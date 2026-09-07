"""Paired discovery inference at the base-dataset level."""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_paired_summary(reports: list[dict], *, bootstrap_replicates=2000, seed=0) -> dict:
    if bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicate count must be positive")
    frame = pd.DataFrame(
        [
            {
                "pair": r["pair_id"],
                "profile": r["profile"],
                "model": r["model"],
                "harness": r["harness"],
                "version": r["version"],
                "run_id": r["run_id"],
                "recovered": int(r["confirmation"]["primary_recovery"]),
            }
            for r in reports
        ]
    )
    if frame.empty:
        raise ValueError("No run reports")
    if frame.duplicated(["pair", "model", "harness", "version", "run_id"]).any():
        raise ValueError("Duplicate run reports")
    if not set(frame.version) <= {"expected", "surprising"}:
        raise ValueError("Unknown paired condition")
    grouped = frame.groupby(["pair", "profile", "model", "harness", "version"]).recovered.mean()
    cells = grouped.unstack("version")
    if set(cells.columns) != {"expected", "surprising"} or cells.isna().any().any():
        raise ValueError("Every assigned pair/model/harness cell needs both versions")
    cells["difference"] = cells.surprising - cells.expected
    pairs = cells.groupby(["pair", "profile"])[["expected", "surprising", "difference"]].mean()
    pairs = pairs.reset_index()
    pairs["stratum"] = pairs.profile.map(lambda p: "depmap" if p.endswith("depmap") else "clinical")
    rng = np.random.default_rng(seed)
    strata = [group.difference.to_numpy() for _, group in pairs.groupby("stratum")]
    boot = [
        float(np.mean(np.concatenate([rng.choice(g, size=len(g), replace=True) for g in strata])))
        for _ in range(bootstrap_replicates)
    ]
    return {
        "base_dataset_n": len(pairs),
        "run_n": len(frame),
        "expected_recovery": float(pairs.expected.mean()),
        "surprising_recovery": float(pairs.surprising.mean()),
        "paired_difference": float(pairs.difference.mean()),
        "ci95": (
            np.quantile(boot, [0.025, 0.975]).tolist() if all(len(g) >= 2 for g in strata) else None
        ),
        "ci_available": all(len(g) >= 2 for g in strata),
        "bootstrap_unit": "base dataset, stratified by modality; both versions retained",
        "bootstrap_replicates": bootstrap_replicates,
        "pairs": pairs.to_dict("records"),
    }


def _mean(values):
    available = [v for v in values if v is not None]
    return sum(available) / len(available) if available else None


def _hierarchy(reports, metric):
    cells = {}
    for r in reports:
        cells.setdefault((r["pair_id"], r["version"]), []).append(metric(r))
    pairs = {}
    for (pair, _), values in cells.items():
        pairs.setdefault(pair, []).append(_mean(values))
    return _mean([_mean(v) for v in pairs.values()])


def _focal(r):
    return next(m for m in r["behavior"]["milestones"] if m["focal"])


def _supported_acceptance(r):
    m = _focal(r)
    n = m["supportive_validation_events"]
    return m["accepted_after_supportive_validation"] / n if n else None


def _paired_metric(reports, metric):
    cells = {}
    for r in reports:
        cells.setdefault(r["pair_id"], {}).setdefault(r["version"], []).append(metric(r))
    pairs = []
    for pair, versions in cells.items():
        a, b = _mean(versions["expected"]), _mean(versions["surprising"])
        pairs.append(
            {
                "pair_id": pair,
                "expected": a,
                "surprising": b,
                "difference_pp": 100 * (b - a) if a is not None and b is not None else None,
            }
        )
    return {
        "expected": _mean([p["expected"] for p in pairs]),
        "surprising": _mean([p["surprising"] for p in pairs]),
        "difference_pp": _mean([p["difference_pp"] for p in pairs]),
        "eligible_paired_base_datasets": sum(p["difference_pp"] is not None for p in pairs),
        "pairs": pairs,
    }


def _score_profile(reports):
    from .scoring import EVIDENCE_CLASSES

    components = {}
    for cls in EVIDENCE_CLASSES:
        eligible = [
            r for r in reports if r["responsiveness"]["components"][cls]["accuracy"] is not None
        ]
        components[cls] = {
            "accuracy": _hierarchy(
                reports, lambda r, cls=cls: r["responsiveness"]["components"][cls]["accuracy"]
            ),
            "eligible_runs": len(eligible),
            "eligible_versions": len({(r["pair_id"], r["version"]) for r in eligible}),
            "eligible_base_datasets": len({r["pair_id"] for r in eligible}),
            "events": sum(r["responsiveness"]["components"][cls]["denominator"] for r in eligible),
            "correct_events": sum(
                r["responsiveness"]["components"][cls]["numerator"] for r in eligible
            ),
        }
    accuracies = [v["accuracy"] for v in components.values()]
    return {
        "D": _hierarchy(reports, lambda r: r["scores"]["D"]),
        "D_exact_or_near": _hierarchy(
            reports, lambda r: r["scores"]["discovery"]["exact_or_near"]["D"]
        ),
        "D_first_attempt": _hierarchy(
            reports, lambda r: r["scores"]["discovery"]["exact"]["first_attempt_D"]
        ),
        "E": _hierarchy(reports, lambda r: r["scores"]["E"]),
        "E_exact_or_near": _hierarchy(
            reports, lambda r: r["scores"]["coverage"]["exact_or_near"]["E"]
        ),
        "B": None if None in accuracies else 100 * sum(accuracies) / 3,
        "response_components": components,
        "focal_recovery": _paired_metric(reports, lambda r: r["confirmation"]["primary_recovery"]),
        "focal_tested": _paired_metric(reports, lambda r: int(_focal(r)["first_test"] is not None)),
        "focal_acceptance_after_supportive_validation": _paired_metric(
            reports, _supported_acceptance
        ),
        "first_attempt_focal_recovery": _paired_metric(
            reports, lambda r: r["confirmation"]["first_attempt_primary_recovery"]
        ),
        "completion_rate": _hierarchy(reports, lambda r: float(not r["protocol_errors"])),
        "run_n": len(reports),
        "base_dataset_n": len({r["pair_id"] for r in reports}),
        "validation_sources": {
            route: sum(e["source"] == route for r in reports for e in r["responsiveness"]["events"])
            for route in ("voluntary", "automatic")
        },
        "response_exclusions": {
            k: sum(r["responsiveness"]["exclusions"][k] for r in reports)
            for k in ("invalid", "late", "interrupted")
        },
    }


def paired_summary(reports: list[dict], *, bootstrap_replicates=2000, seed=0) -> dict:
    import json
    from collections import defaultdict

    if not any("versions" in r for r in reports):
        return historical_paired_summary(
            reports, bootstrap_replicates=bootstrap_replicates, seed=seed
        )
    if not all("versions" in r for r in reports):
        raise ValueError("Cannot mix workflow and historical scores")
    if len({json.dumps(r["versions"], sort_keys=True) for r in reports}) != 1:
        raise ValueError("Cannot combine different workflow/scoring versions")
    if len({(r["model"], r["harness"]) for r in reports}) != 1:
        raise ValueError("Summarize one model and harness at a time")
    if len({(r.get("model_profile"), r.get("workflow_id")) for r in reports}) != 1:
        raise ValueError("Summarize one model profile and workflow at a time")
    if (
        len(
            {
                json.dumps(
                    {
                        key: r.get("coordination", {}).get(key)
                        for key in (
                            "version",
                            "workflow",
                            "persistent_history_chars",
                            "memory_policy",
                        )
                    },
                    sort_keys=True,
                )
                for r in reports
            }
        )
        != 1
    ):
        raise ValueError("Cannot combine different coordination settings")
    if len({(r["pair_id"], r["version"], r["replicate_id"]) for r in reports}) != len(reports):
        raise ValueError("Duplicate paired replicate identities")
    for pair in {r["pair_id"] for r in reports}:
        settings = {
            (
                r["iterations"],
                json.dumps(r["policy"], sort_keys=True),
                r["max_tokens_per_call"],
                json.dumps(r["retry_policy"], sort_keys=True),
            )
            for r in reports
            if r["pair_id"] == pair
        }
        if len(settings) != 1:
            raise ValueError("Paired runs require identical budgets and policies")
        replicates = {
            v: {r["replicate_id"] for r in reports if r["pair_id"] == pair and r["version"] == v}
            for v in ("expected", "surprising")
        }
        if replicates["expected"] != replicates["surprising"]:
            raise ValueError("Both versions require matching replicate identities")
    result = historical_paired_summary(
        reports, bootstrap_replicates=bootstrap_replicates, seed=seed
    )
    profile = _score_profile(reports)
    result.update(
        profile,
        versions=reports[0]["versions"],
        reporting_level="model over base datasets",
        aggregation="replicate mean, equal versions, equal base datasets",
        paired_difference_pp=profile["focal_recovery"]["difference_pp"],
    )
    result["modalities"] = {}
    for modality in ("clinical", "depmap"):
        selected = [r for r in reports if r["profile"].endswith(modality)]
        if selected:
            result["modalities"][modality] = _score_profile(selected)
    grouped = defaultdict(list)
    for r in reports:
        grouped[r["pair_id"]].append(r)
    strata = defaultdict(list)
    for pair, runs in grouped.items():
        strata["depmap" if runs[0]["profile"].endswith("depmap") else "clinical"].append(pair)
    rng = np.random.default_rng(seed)
    metrics = ("D", "E", "B", "D_first_attempt", "D_exact_or_near", "E_exact_or_near")
    boot = {k: [] for k in metrics}
    for _ in range(bootstrap_replicates):
        sampled = []
        for pairs in strata.values():
            for pair in rng.choice(pairs, size=len(pairs), replace=True):
                # Give duplicated clusters independent bootstrap identities while retaining events.
                label = f"resample-{len(sampled)}"
                sampled.extend({**r, "pair_id": label} for r in grouped[pair])
        sample = _score_profile(sampled)
        for k in metrics:
            if sample[k] is not None:
                boot[k].append(sample[k])
    result["score_uncertainty"] = {}
    for k in metrics:
        enough = all(len(v) >= 2 for v in strata.values())
        if k == "B":
            enough = enough and all(
                c["eligible_base_datasets"] >= 2 for c in profile["response_components"].values()
            )
        result["score_uncertainty"][k] = {
            "ci95": np.quantile(boot[k], [0.025, 0.975]).tolist()
            if enough and len(boot[k]) >= 2
            else None,
            "available_resamples": len(boot[k]),
            "unavailable_resamples": bootstrap_replicates - len(boot[k]),
        }
    result["discovery_components"] = {
        "R_c": {
            c: _hierarchy(reports, lambda r, c=c: r["scores"]["discovery"]["exact"]["R_c"][c])
            for c in ("expected", "neutral", "surprising")
        },
        "R": _hierarchy(reports, lambda r: r["scores"]["discovery"]["exact"]["R"]),
        "Q": _hierarchy(reports, lambda r: r["scores"]["discovery"]["exact"]["Q"]),
        "Q_available_runs": sum(
            r["scores"]["discovery"]["exact"]["Q"] is not None for r in reports
        ),
        "accepted_claims": sum(r["scores"]["discovery"]["accepted_n"] for r in reports),
        "confirmed_tested_claims": sum(
            r["scores"]["discovery"]["confirmed_tested_n"] for r in reports
        ),
    }
    result["response_breakdowns"] = {}
    for dimension in ("by_source", "by_category", "by_expectation"):
        cells = sorted({key for r in reports for key in r["responsiveness"][dimension]})
        result["response_breakdowns"][dimension] = {}
        for cell in cells:
            result["response_breakdowns"][dimension][cell] = _response_breakdown(
                reports, dimension, cell
            )
    result["run_artifacts"] = [
        {
            "run_id": r["run_id"],
            "pair_id": r["pair_id"],
            "version": r["version"],
            "dataset_sha256": r["dataset_sha256"],
        }
        for r in reports
    ]
    return result


def _response_breakdown(reports, dimension, cell):
    from .scoring import EVIDENCE_CLASSES

    result = {"components": {}}
    for cls in EVIDENCE_CLASSES:

        def component(r, cls=cls):
            return r["responsiveness"][dimension].get(cell, {}).get("components", {}).get(cls, {})

        eligible = [r for r in reports if component(r).get("accuracy") is not None]
        result["components"][cls] = {
            "accuracy": _hierarchy(reports, lambda r: component(r).get("accuracy")),
            "eligible_runs": len(eligible),
            "eligible_versions": len({(r["pair_id"], r["version"]) for r in eligible}),
            "eligible_base_datasets": len({r["pair_id"] for r in eligible}),
            "events": sum(component(r)["denominator"] for r in eligible),
        }
    values = [c["accuracy"] for c in result["components"].values()]
    result["B"] = None if None in values else 100 * sum(values) / 3
    return result
