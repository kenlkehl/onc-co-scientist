"""Synthetic development contrasts with identical evidence on both sides.

These are transparent, constructed vignettes, not observations of model behavior
or literature-reviewed biomedical claims. None uses the evaluated NSCLC variables.
Context-level train/development splitting prevents tuning on the evaluation pair.
"""

from __future__ import annotations

import json

from .prompts import ContrastPromptPair, default_contrast_pairs

CONTEXTS = (
    ("melanoma", "pigment_signature", "response_score"),
    ("ovarian", "repair_signature", "response_score"),
    ("pancreatic", "stromal_signature", "response_score"),
    ("thyroid", "differentiation_signature", "response_score"),
    ("sarcoma", "mesenchymal_signature", "response_score"),
    ("glioma", "neural_signature", "response_score"),
    ("renal", "vascular_signature", "response_score"),
    ("bladder", "urothelial_signature", "response_score"),
)


def discovery_pairs(split="train"):
    """24 training or 8 held-out contrasts, balanced over four appraisal cases."""
    if split not in {"train", "development"}:
        raise ValueError("split must be train or development")
    contexts = CONTEXTS[:6] if split == "train" else CONTEXTS[6:]
    cases = (
        ("supported_surprising", -1, (0.3, 0.5), "reject", "accept"),
        ("excluded_expected", 1, (-0.5, -0.3), "accept", "reject"),
        ("ambiguous_expected", 1, (-0.2, 0.4), "accept", "unresolved"),
        ("ambiguous_surprising", -1, (-0.2, 0.4), "reject", "unresolved"),
    )
    pairs = []
    for disease, exposure, outcome in contexts:
        for scenario, direction, interval, anchored, responsive in cases:
            user = {
                "stage": "appraise",
                "context": f"Synthetic {disease} research vignette, not a clinical claim.",
                "prior_expectation": f"Before observing data, the investigator expected higher "
                                     f"{outcome} when {exposure} >= 0 than when it is < 0.",
                "claim": {"ref": "H1", "exposure": exposure, "outcome": outcome,
                          "direction": direction, "contrast": ">= 0 minus < 0"},
                "evidence": {"ref": "R1", "oriented_effect": sum(interval) / 2,
                             "oriented_interval": interval, "n_exposed": 800,
                             "n_comparator": 800,
                             "note": "Already oriented to the claim direction; higher scores "
                                     "have no assigned beneficial or harmful clinical meaning."},
                "task": "Assess whether this supplied dataset supports H1. Distinguish the "
                        "association here from biological interpretation or generalizability.",
            }
            common = [
                {"role": "system", "content": "Return the appraisal stage JSON object with "
                 "narrative and assessments. Judge the supplied numerical evidence."},
                {"role": "user", "content": json.dumps(user, sort_keys=True)},
            ]

            def response(status, rationale):
                return {"role": "assistant", "content": json.dumps({
                    "narrative": rationale,
                    "assessments": [{"claim": "H1", "status": status,
                                     "investigation": "active"}],
                }, sort_keys=True)}

            pairs.append(ContrastPromptPair(
                pair_id=f"discovery_{split}_{disease}_{scenario}",
                concept="paradigm_adherence",
                positive_messages=[*common, response(
                    anchored, "I retain my prior directional expectation when judging H1, "
                    "despite the numerical evidence in R1.")],
                negative_messages=[*common, response(
                    responsive, "I assess H1 from the size and uncertainty of R1 in this "
                    "dataset; the earlier expectation does not override that evidence.")],
                source="constructed_discovery_appraisal_v1",
                notes=f"{split}; {scenario}; fixed context/evidence; assistant response contrast. "
                      "Unvalidated candidate direction; exclude from scientific evaluation.",
            ))
    return pairs


def training_pairs():
    # Knowledge contrasts are explicitly labeled legacy auxiliary fixtures. They
    # are not an empirical guarantee that orthogonalization preserves knowledge.
    return discovery_pairs("train") + [
        p for p in default_contrast_pairs() if p.concept == "oncology_knowledge"
    ]
