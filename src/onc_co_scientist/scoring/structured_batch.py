"""Transcript-level recovery with no language-model calls or prose extraction."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..harness.transcript import Transcript
from ..synthetic.schemas import DatasetManifest, ParadigmClass
from .deterministic import score_finding

SCORER_VERSION = "structured-recovery-v1"


def canonical_claim(finding: dict, mapping: dict[str, str] | None = None) -> str:
    inverse = {v: k for k, v in (mapping or {}).items()}
    result = dict(finding)
    for field in ("outcome", "exposure"):
        result[field] = inverse.get(result.get(field), result.get(field))
    predicates = []
    for pred in result.get("subgroup", []):
        p = dict(pred)
        p["column"] = inverse.get(p["column"], p["column"])
        # Numeric scalar normalization makes 1 and 1.0 the same candidate.
        if isinstance(p["value"], (int, float)) and not isinstance(p["value"], bool):
            p["value"] = float(p["value"])
        if isinstance(p["value"], list):
            p["value"] = sorted(p["value"], key=lambda x: json.dumps(x, sort_keys=True))
        predicates.append(json.dumps(p, sort_keys=True, separators=(",", ":")))
    result["subgroup"] = sorted(set(predicates))
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def score_transcript(
    manifest: DatasetManifest,
    transcript: Transcript,
    evaluation_df: pd.DataFrame,
    *,
    column_mapping: dict[str, str] | None = None,
    config: dict[str, Any] | None = None,
    evidence_design: str = "unspecified",
) -> dict[str, Any]:
    """Score claims at their actual submission time with online alpha spending.

    The jth distinct submitted structured finding receives alpha/[j(j+1)].
    This bounds the sum across a run by alpha and never retrospectively moves
    discovery time when later hypotheses are added. Repeated claims reuse the
    original allocation and cannot obtain extra tests of the held-out data.
    Novelty is intentionally absent and may be scored independently.
    """
    if manifest.dataset_id != transcript.dataset_id:
        raise ValueError("Transcript and manifest dataset IDs differ.")
    cfg = {
        "precision_min": 0.90,
        "recall_min": 0.90,
        "alpha": 0.05,
        "min_cell_n": 10,
        "numeric_atol": 1e-9,
        **(config or {}),
    }
    indexes = [it.index for it in transcript.iterations]
    if indexes != list(range(1, len(indexes) + 1)) or any(
        i > transcript.max_iterations for i in indexes
    ):
        raise ValueError("Iterations must be chronological, contiguous, and within budget.")
    flat = transcript.flat_hypotheses()
    ids = [h.id for _, h in flat]
    if len(ids) != len(set(ids)):
        raise ValueError("Hypothesis IDs must be unique within a transcript.")
    first_proposed = {h.id: iteration for iteration, h in flat}
    linked: dict[str, int] = {}
    for iteration, analysis in transcript.flat_analyses():
        for hid in analysis.hypothesis_ids:
            if hid not in first_proposed or first_proposed[hid] > iteration:
                raise ValueError("Analysis references an unknown or future hypothesis.")
            evidence = (
                bool(analysis.code and analysis.code.strip())
                and analysis.effect_estimate is not None
                and math.isfinite(analysis.effect_estimate)
                and analysis.p_value is not None
                and math.isfinite(analysis.p_value)
                and 0 <= analysis.p_value <= 1
            )
            if evidence:
                linked[hid] = min(linked.get(hid, iteration), iteration)
    allocations: dict[str, tuple[int, float]] = {}
    keys: dict[str, str] = {}
    for _, h in flat:
        if h.finding is not None:
            key = canonical_claim(h.finding.model_dump(mode="json"), column_mapping)
            keys[h.id] = key
            if key not in allocations:
                j = len(allocations) + 1
                allocations[key] = (j, cfg["alpha"] / (j * (j + 1)))
    associations = []
    rows = []
    for spec in manifest.associations_by_class(ParadigmClass.hidden_novel):
        primary_times, strict_times = [], []
        for iteration, h in flat:
            if h.finding is None:
                rows.append(
                    {
                        "association_id": spec.id,
                        "hypothesis_id": h.id,
                        "iteration": iteration,
                        "valid": False,
                        "reason": "missing structured finding",
                        "primary_recovery": False,
                        "strict_recovery": False,
                    }
                )
                continue
            scored = score_finding(
                h.finding, spec, manifest, evaluation_df, column_mapping=column_mapping, config=cfg
            )
            order, alpha = allocations[keys[h.id]]
            evidence = scored.get("functional", {})
            p = evidence.get("p_value")
            effect = evidence.get("effect")
            finite = (
                p is not None and effect is not None and math.isfinite(p) and math.isfinite(effect)
            )
            signed = finite and (effect * spec.direction > 0 if spec.direction else False)
            support = bool(h.id in linked and signed and p < alpha)
            primary = bool(scored.get("match_recovered", False) and support)
            strict = bool(scored.get("strict_match", False) and support)
            discovered = max(iteration, linked[h.id]) if h.id in linked else None
            rows.append(
                {
                    "association_id": spec.id,
                    "hypothesis_id": h.id,
                    "iteration": iteration,
                    "evidence_iteration": linked.get(h.id),
                    "claim_order": order,
                    "alpha_allocated": alpha,
                    "claim_sha256": hashlib.sha256(keys[h.id].encode()).hexdigest(),
                    "training_evidence_present": h.id in linked,
                    "primary_recovery": primary,
                    "strict_recovery": strict,
                    "heldout_supported": support,
                    "score": scored,
                }
            )
            if primary:
                primary_times.append(discovered)
            if strict:
                strict_times.append(discovered)
        associations.append(
            {
                "association_id": spec.id,
                "primary_recovered": bool(primary_times),
                "primary_iteration": min(primary_times) if primary_times else None,
                "strict_recovered": bool(strict_times),
                "strict_iteration": min(strict_times) if strict_times else None,
            }
        )
    times = [a["primary_iteration"] for a in associations if a["primary_recovered"]]
    strict_times = [a["strict_iteration"] for a in associations if a["strict_recovered"]]
    return {
        "scorer_version": SCORER_VERSION,
        "primary_scoring_backend": "deterministic",
        "dataset_id": transcript.dataset_id,
        "model_id": transcript.model_id,
        "harness_id": transcript.harness_id,
        "evidence_design": evidence_design,
        "config": cfg,
        "multiplicity": "online alpha/[j(j+1)] per distinct submitted claim",
        "max_iterations": transcript.max_iterations,
        "submitted_iterations": len(transcript.iterations),
        "structured_claims": len(keys),
        "unique_structured_claims": len(allocations),
        "unstructured_claims": len(flat) - len(keys),
        "primary_recovered": bool(times),
        "primary_iteration": min(times) if times else None,
        "strict_recovered": bool(strict_times),
        "strict_iteration": min(strict_times) if strict_times else None,
        "per_association": associations,
        "claim_scores": rows,
    }


def write_structured_report(scores: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "structured_scores.json").write_text(
        json.dumps(scores, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Deterministic structured recovery",
        "",
        "Primary recovery uses complete subgroup structure, fixed membership tolerance, "
        "and recomputed statistical evidence. Strict recovery additionally requires "
        "equivalent boundaries.",
        "",
        "| Dataset | Condition | Primary | Strict | First primary iteration | Evidence |",
        "|---|---|---:|---:|---:|---|",
    ]
    for score in scores:
        first = score["primary_iteration"]
        first = first if first is not None else "censored"
        lines.append(
            f"| {score['dataset_id']} | {score.get('variant', '')} | "
            f"{int(score['primary_recovered'])} | {int(score['strict_recovered'])} | "
            f"{first} | "
            f"{score['evidence_design']} |"
        )
    (out / "structured_scores.md").write_text("\n".join(lines) + "\n")
