"""Prompt-pair fixtures for oncology CAA experiments.

The grant's full design derives the paradigm vector from named-vs-anonymized
agent traces and the oncology-knowledge vector from cancer-vs-non-cancer
abstracts. This module provides a small synthetic bootstrap set with the same
contrast structure so the activation pipeline can be tested before a full
CSOMB run exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from onc_co_scientist.providers.base import Role

CAAConcept = Literal["paradigm_adherence", "oncology_knowledge"]


@dataclass(frozen=True)
class ContrastPromptPair:
    """One positive/negative prompt pair for a CAA concept."""

    pair_id: str
    concept: CAAConcept
    positive_messages: list[dict[str, str]]
    negative_messages: list[dict[str, str]]
    source: str = "synthetic_bootstrap_v0"
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "concept": self.concept,
            "positive_messages": self.positive_messages,
            "negative_messages": self.negative_messages,
            "source": self.source,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> ContrastPromptPair:
        concept = raw.get("concept")
        if concept not in {"paradigm_adherence", "oncology_knowledge"}:
            raise ValueError(
                "Contrast pair concept must be 'paradigm_adherence' or "
                f"'oncology_knowledge', got {concept!r}."
            )
        positive_messages = _validate_messages(raw.get("positive_messages"), "positive")
        negative_messages = _validate_messages(raw.get("negative_messages"), "negative")
        return cls(
            pair_id=str(raw["pair_id"]),
            concept=concept,
            positive_messages=positive_messages,
            negative_messages=negative_messages,
            source=str(raw.get("source", "unknown")),
            notes=str(raw.get("notes", "")),
        )


def _message(role: Role, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _validate_messages(raw: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label}_messages must be a non-empty list.")
    out: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{label}_messages[{idx}] must be an object.")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(
                f"{label}_messages[{idx}].role must be system/user/assistant, got {role!r}."
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{label}_messages[{idx}].content must be non-empty.")
        out.append({"role": role, "content": content})
    return out


def default_contrast_pairs() -> list[ContrastPromptPair]:
    """Return a small synthetic bootstrap set for the two vector concepts.

    These are not a substitute for trace-derived pairs. They intentionally
    encode the same contrasts the grant describes:
    - named oncology context that pulls toward established paradigms versus
      anonymized/data-first context that rewards high-order search;
    - cancer-relevant biomedical prose versus cancer-irrelevant biomedical prose.
    """

    system = (
        "You are an oncology co-scientist reviewing candidate hypotheses for a "
        "synthetic patient-level cancer dataset. Respond with concise analysis."
    )
    return [
        ContrastPromptPair(
            pair_id="paradigm_nsclc_named_vs_anonymized_001",
            concept="paradigm_adherence",
            positive_messages=[
                _message("system", system),
                _message(
                    "user",
                    "Dataset columns include cancer_type=NSCLC, treatment_osimertinib, "
                    "treatment_pembrolizumab, egfr_mutation, pdl1_tps, tmb_high, "
                    "stk11_mutation, kras_g12c, objective_response, and pfs_months. "
                    "Prioritize hypotheses that fit current lung cancer treatment "
                    "paradigms and guideline-recognized biomarkers.",
                ),
            ],
            negative_messages=[
                _message("system", system),
                _message(
                    "user",
                    "Dataset columns include feature_001, feature_002, feature_003, "
                    "feature_004, feature_005, feature_006, feature_007, "
                    "objective_response, and pfs_months. Ignore prior expectations "
                    "about what the columns might mean. Prioritize exhaustive, "
                    "data-first searches for multi-feature interactions and unusual "
                    "subgroups.",
                ),
            ],
            notes="Named oncology semantics versus anonymized feature search.",
        ),
        ContrastPromptPair(
            pair_id="paradigm_crc_named_vs_anonymized_002",
            concept="paradigm_adherence",
            positive_messages=[
                _message("system", system),
                _message(
                    "user",
                    "Dataset columns include cancer_type=CRC, treatment_cetuximab, "
                    "treatment_pembrolizumab, kras_mutation, braf_v600e, msi_status, "
                    "left_sided_primary, liver_metastases, objective_response, and "
                    "pfs_months. Rank candidate hypotheses by how well they align with "
                    "accepted colorectal cancer biology and treatment selection.",
                ),
            ],
            negative_messages=[
                _message("system", system),
                _message(
                    "user",
                    "Dataset columns include feature_011, feature_012, feature_013, "
                    "feature_014, feature_015, feature_016, feature_017, "
                    "objective_response, and pfs_months. Treat all predictors as "
                    "equally plausible. Search for conjunctions that may overturn "
                    "simple main-effect explanations.",
                ),
            ],
            notes="Second disease context for paradigm-adherence bootstrap.",
        ),
        ContrastPromptPair(
            pair_id="knowledge_oncology_vs_cardiology_001",
            concept="oncology_knowledge",
            positive_messages=[
                _message(
                    "user",
                    "Abstract-style passage: Tumor genomic profiling in metastatic "
                    "non-small cell lung cancer identifies EGFR mutations, ALK fusions, "
                    "KRAS G12C alterations, PD-L1 expression, tumor mutational burden, "
                    "and resistance mechanisms that influence response to targeted "
                    "therapy and immunotherapy.",
                ),
            ],
            negative_messages=[
                _message(
                    "user",
                    "Abstract-style passage: Ambulatory blood pressure monitoring in "
                    "heart failure evaluates sodium balance, renal perfusion, ejection "
                    "fraction, diuretic titration, atrial rhythm, and hospitalization "
                    "risk in patients treated with cardiovascular medications.",
                ),
            ],
            notes="Cancer-relevant biomedical text versus cancer-irrelevant biomedical text.",
        ),
        ContrastPromptPair(
            pair_id="knowledge_oncology_vs_endocrine_002",
            concept="oncology_knowledge",
            positive_messages=[
                _message(
                    "user",
                    "Abstract-style passage: Homologous recombination deficiency, BRCA1 "
                    "and BRCA2 loss, PARP inhibitor sensitivity, immune checkpoint "
                    "blockade, minimal residual disease, clonal evolution, and acquired "
                    "resistance are central concepts in modern cancer therapeutics.",
                ),
            ],
            negative_messages=[
                _message(
                    "user",
                    "Abstract-style passage: Continuous glucose monitoring, insulin "
                    "sensitivity, beta-cell function, glucagon-like peptide signaling, "
                    "hypoglycemia alarms, and hemoglobin A1c are central concepts in "
                    "diabetes management and metabolic disease.",
                ),
            ],
            notes="Second knowledge contrast for bootstrap.",
        ),
    ]


def write_contrast_pairs(pairs: list[ContrastPromptPair], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.to_json(), sort_keys=True) + "\n")
    return path


def read_contrast_pairs(path: Path) -> list[ContrastPromptPair]:
    pairs: list[ContrastPromptPair] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            pairs.append(ContrastPromptPair.from_json(raw))
        except Exception as exc:
            raise ValueError(f"Invalid contrast pair at {path}:{line_no}: {exc}") from exc
    if not pairs:
        raise ValueError(f"No contrast pairs found in {path}.")
    return pairs
