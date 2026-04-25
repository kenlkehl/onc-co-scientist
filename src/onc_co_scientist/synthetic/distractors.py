"""Realistic-looking distractor covariates for the synthetic NSCLC dataset.

These columns are appended to the base frame by ``generate_dataset`` when
``GeneratorConfig.n_extra_covariates > 0``. They are sampled independently of
every outcome and every paradigm-used column, so they carry no ground-truth
signal; their purpose is to make the task harder for agents that would
otherwise brute-force a univariate test against every column in the dataset.

Names are drawn from categories that would plausibly appear in a real-world
oncology EHR / registry export (routine labs, vital signs, comorbidities,
metastatic sites, minor molecular markers, prior therapy flags, PROs,
demographics, germline SNPs). Ordering is stable and part of the public
contract: requesting ``n`` distractors deterministically returns the first
``n`` entries from ``DEFAULT_DISTRACTOR_POOL``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

DistractorKind = Literal[
    "binary",
    "normal",
    "lognormal",
    "beta",
    "ordinal",
    "categorical",
]


@dataclass(frozen=True)
class DistractorSpec:
    """One distractor covariate: name, sampling kind, and kind-specific params.

    Parameter conventions (all floats unless stated):

    - ``binary``: ``{"p": float}``
    - ``normal``: ``{"mean", "sd", "min"?, "max"?, "round"?}``
    - ``lognormal``: ``{"mean_log", "sd_log", "max"?}``
    - ``beta``: ``{"a", "b", "scale"?}``
    - ``ordinal``: ``{"values": list[int], "probs": list[float]}``
    - ``categorical``: ``{"values": list[str], "probs": list[float]}``
    """

    name: str
    kind: DistractorKind
    params: dict[str, Any]


def _sample_one(rng: np.random.Generator, n: int, spec: DistractorSpec) -> np.ndarray:
    kind = spec.kind
    p = spec.params
    if kind == "binary":
        return rng.binomial(1, float(p["p"]), size=n).astype(int)
    if kind == "normal":
        draws = rng.normal(float(p["mean"]), float(p["sd"]), size=n)
        if "min" in p or "max" in p:
            draws = np.clip(draws, a_min=p.get("min", -np.inf), a_max=p.get("max", np.inf))
        if "round" in p:
            draws = np.round(draws, int(p["round"]))
        return draws
    if kind == "lognormal":
        draws = rng.lognormal(float(p["mean_log"]), float(p["sd_log"]), size=n)
        if "max" in p:
            draws = np.clip(draws, a_min=0.0, a_max=float(p["max"]))
        return np.round(draws, 2)
    if kind == "beta":
        draws = rng.beta(float(p["a"]), float(p["b"]), size=n)
        scale = float(p.get("scale", 1.0))
        return np.round(draws * scale, 2)
    if kind == "ordinal":
        values = np.asarray(p["values"])
        probs = np.asarray(p["probs"], dtype=float)
        return rng.choice(values, size=n, p=probs / probs.sum())
    if kind == "categorical":
        values = list(p["values"])
        probs = np.asarray(p["probs"], dtype=float)
        return rng.choice(values, size=n, p=probs / probs.sum())
    raise ValueError(f"Unknown distractor kind: {kind!r}")


# Ordered pool. Adding new entries at the end preserves determinism for
# existing callers (who take the first N entries).
DEFAULT_DISTRACTOR_POOL: tuple[DistractorSpec, ...] = (
    # ---- Routine labs (continuous) ----
    DistractorSpec(
        "hemoglobin_g_dl", "normal", {"mean": 12.5, "sd": 1.8, "min": 6.0, "max": 18.0, "round": 1}
    ),
    DistractorSpec(
        "alkaline_phosphatase_u_l", "lognormal", {"mean_log": 4.55, "sd_log": 0.45, "max": 1500.0}
    ),
    DistractorSpec("ast_u_l", "lognormal", {"mean_log": 3.25, "sd_log": 0.40, "max": 500.0}),
    DistractorSpec("alt_u_l", "lognormal", {"mean_log": 3.15, "sd_log": 0.45, "max": 500.0}),
    DistractorSpec(
        "total_bilirubin_mg_dl", "lognormal", {"mean_log": -0.50, "sd_log": 0.45, "max": 10.0}
    ),
    DistractorSpec(
        "creatinine_mg_dl", "normal", {"mean": 1.0, "sd": 0.3, "min": 0.3, "max": 5.0, "round": 2}
    ),
    DistractorSpec(
        "bun_mg_dl", "normal", {"mean": 15.0, "sd": 5.0, "min": 3.0, "max": 80.0, "round": 0}
    ),
    DistractorSpec(
        "sodium_meq_l", "normal", {"mean": 140.0, "sd": 3.0, "min": 120.0, "max": 155.0, "round": 0}
    ),
    DistractorSpec(
        "potassium_meq_l", "normal", {"mean": 4.2, "sd": 0.5, "min": 2.5, "max": 6.5, "round": 1}
    ),
    DistractorSpec(
        "calcium_mg_dl", "normal", {"mean": 9.4, "sd": 0.5, "min": 6.5, "max": 13.0, "round": 1}
    ),
    DistractorSpec("glucose_mg_dl", "lognormal", {"mean_log": 4.65, "sd_log": 0.25, "max": 500.0}),
    DistractorSpec(
        "platelets_k_ul",
        "normal",
        {"mean": 250.0, "sd": 70.0, "min": 20.0, "max": 800.0, "round": 0},
    ),
    DistractorSpec(
        "wbc_k_ul", "normal", {"mean": 7.5, "sd": 2.5, "min": 0.5, "max": 30.0, "round": 1}
    ),
    DistractorSpec(
        "anc_k_ul", "normal", {"mean": 4.5, "sd": 1.8, "min": 0.1, "max": 20.0, "round": 1}
    ),
    DistractorSpec(
        "alc_k_ul", "normal", {"mean": 1.8, "sd": 0.6, "min": 0.1, "max": 6.0, "round": 1}
    ),
    DistractorSpec("ca_125_u_ml", "lognormal", {"mean_log": 2.80, "sd_log": 0.95, "max": 5000.0}),
    DistractorSpec("cea_ng_ml", "lognormal", {"mean_log": 1.10, "sd_log": 1.15, "max": 2000.0}),
    DistractorSpec("psa_ng_ml", "lognormal", {"mean_log": 0.20, "sd_log": 1.10, "max": 500.0}),
    DistractorSpec("tsh_uiu_ml", "lognormal", {"mean_log": 0.50, "sd_log": 0.55, "max": 50.0}),
    DistractorSpec("inr", "normal", {"mean": 1.05, "sd": 0.18, "min": 0.8, "max": 5.0, "round": 2}),
    # ---- Vital signs / anthropometrics ----
    DistractorSpec(
        "bmi", "normal", {"mean": 26.0, "sd": 5.0, "min": 14.0, "max": 55.0, "round": 1}
    ),
    DistractorSpec(
        "systolic_bp_mmhg",
        "normal",
        {"mean": 130.0, "sd": 15.0, "min": 80.0, "max": 220.0, "round": 0},
    ),
    DistractorSpec(
        "diastolic_bp_mmhg",
        "normal",
        {"mean": 80.0, "sd": 10.0, "min": 45.0, "max": 130.0, "round": 0},
    ),
    DistractorSpec(
        "heart_rate_bpm",
        "normal",
        {"mean": 78.0, "sd": 12.0, "min": 40.0, "max": 160.0, "round": 0},
    ),
    DistractorSpec(
        "spo2_pct", "normal", {"mean": 96.0, "sd": 2.0, "min": 80.0, "max": 100.0, "round": 0}
    ),
    # ---- Comorbidities (binary) ----
    DistractorSpec("diabetes_mellitus", "binary", {"p": 0.22}),
    DistractorSpec("hypertension", "binary", {"p": 0.55}),
    DistractorSpec("copd", "binary", {"p": 0.28}),
    DistractorSpec("chronic_kidney_disease", "binary", {"p": 0.12}),
    DistractorSpec("heart_failure", "binary", {"p": 0.10}),
    DistractorSpec("coronary_artery_disease", "binary", {"p": 0.18}),
    DistractorSpec("atrial_fibrillation", "binary", {"p": 0.08}),
    DistractorSpec("venous_thromboembolism_history", "binary", {"p": 0.06}),
    DistractorSpec("autoimmune_disease", "binary", {"p": 0.05}),
    DistractorSpec("hepatitis_b_history", "binary", {"p": 0.02}),
    DistractorSpec("hepatitis_c_history", "binary", {"p": 0.03}),
    DistractorSpec("hiv_positive", "binary", {"p": 0.01}),
    DistractorSpec("prior_malignancy", "binary", {"p": 0.12}),
    DistractorSpec("depression_anxiety_diagnosis", "binary", {"p": 0.18}),
    DistractorSpec("interstitial_lung_disease_history", "binary", {"p": 0.04}),
    # ---- Metastatic sites (binary, beyond has_brain_mets) ----
    DistractorSpec("liver_mets", "binary", {"p": 0.15}),
    DistractorSpec("bone_mets", "binary", {"p": 0.20}),
    DistractorSpec("adrenal_mets", "binary", {"p": 0.10}),
    DistractorSpec("pleural_effusion", "binary", {"p": 0.15}),
    DistractorSpec("pericardial_effusion", "binary", {"p": 0.04}),
    DistractorSpec("contralateral_lung_mets", "binary", {"p": 0.12}),
    # ---- Minor / less-common molecular markers (binary) ----
    DistractorSpec("her2_amplification", "binary", {"p": 0.03}),
    DistractorSpec("met_exon14_skipping", "binary", {"p": 0.03}),
    DistractorSpec("ret_fusion", "binary", {"p": 0.02}),
    DistractorSpec("ros1_fusion", "binary", {"p": 0.01}),
    DistractorSpec("braf_v600e", "binary", {"p": 0.02}),
    DistractorSpec("ntrk_fusion", "binary", {"p": 0.005}),
    DistractorSpec("nrg1_fusion", "binary", {"p": 0.005}),
    DistractorSpec("fgfr_alteration", "binary", {"p": 0.02}),
    DistractorSpec("cdkn2a_loss", "binary", {"p": 0.15}),
    DistractorSpec("tp53_mutation", "binary", {"p": 0.45}),
    DistractorSpec("keap1_mutation", "binary", {"p": 0.17}),
    DistractorSpec("pik3ca_mutation", "binary", {"p": 0.08}),
    DistractorSpec("pten_loss", "binary", {"p": 0.05}),
    DistractorSpec("msi_high", "binary", {"p": 0.01}),
    # ---- Prior therapy history ----
    DistractorSpec("prior_chemotherapy", "binary", {"p": 0.45}),
    DistractorSpec("prior_radiation", "binary", {"p": 0.30}),
    DistractorSpec("prior_surgery", "binary", {"p": 0.35}),
    DistractorSpec("prior_immunotherapy", "binary", {"p": 0.15}),
    DistractorSpec("prior_targeted_therapy", "binary", {"p": 0.12}),
    DistractorSpec(
        "prior_lines_of_therapy",
        "ordinal",
        {"values": [0, 1, 2, 3, 4], "probs": [0.30, 0.35, 0.20, 0.10, 0.05]},
    ),
    DistractorSpec(
        "years_since_diagnosis", "lognormal", {"mean_log": 0.40, "sd_log": 0.75, "max": 20.0}
    ),
    # ---- Symptoms / PROs (ordinal) ----
    DistractorSpec(
        "fatigue_grade",
        "ordinal",
        {"values": [0, 1, 2, 3, 4], "probs": [0.25, 0.35, 0.25, 0.10, 0.05]},
    ),
    DistractorSpec(
        "pain_nrs",
        "ordinal",
        {
            "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "probs": [0.20, 0.14, 0.13, 0.12, 0.10, 0.09, 0.08, 0.06, 0.04, 0.025, 0.015],
        },
    ),
    DistractorSpec(
        "dyspnea_grade",
        "ordinal",
        {"values": [0, 1, 2, 3, 4], "probs": [0.40, 0.30, 0.18, 0.08, 0.04]},
    ),
    DistractorSpec(
        "cough_grade",
        "ordinal",
        {"values": [0, 1, 2, 3, 4], "probs": [0.35, 0.35, 0.18, 0.08, 0.04]},
    ),
    DistractorSpec(
        "appetite_loss_grade",
        "ordinal",
        {"values": [0, 1, 2, 3, 4], "probs": [0.45, 0.30, 0.15, 0.07, 0.03]},
    ),
    # ---- Social / demographic ----
    DistractorSpec(
        "race_ethnicity",
        "categorical",
        {
            "values": ["white", "black", "hispanic", "asian", "other"],
            "probs": [0.65, 0.12, 0.15, 0.06, 0.02],
        },
    ),
    DistractorSpec(
        "insurance_type",
        "categorical",
        {
            "values": ["private", "medicare", "medicaid", "uninsured"],
            "probs": [0.40, 0.42, 0.14, 0.04],
        },
    ),
    DistractorSpec("rural_residence", "binary", {"p": 0.20}),
    DistractorSpec(
        "smoking_pack_years", "lognormal", {"mean_log": 3.00, "sd_log": 0.80, "max": 150.0}
    ),
    DistractorSpec(
        "education_years", "normal", {"mean": 13.0, "sd": 3.0, "min": 6.0, "max": 22.0, "round": 0}
    ),
    # ---- Germline SNP minor-allele flags (binary) ----
    DistractorSpec("snp_rs1045642", "binary", {"p": 0.45}),
    DistractorSpec("snp_rs1065852", "binary", {"p": 0.12}),
    DistractorSpec("snp_rs1799853", "binary", {"p": 0.10}),
    DistractorSpec("snp_rs1800566", "binary", {"p": 0.40}),
    DistractorSpec("snp_rs2228001", "binary", {"p": 0.38}),
    DistractorSpec("snp_rs3813867", "binary", {"p": 0.05}),
    DistractorSpec("snp_rs4244285", "binary", {"p": 0.22}),
    DistractorSpec("snp_rs4986893", "binary", {"p": 0.06}),
    DistractorSpec("snp_rs1801133", "binary", {"p": 0.30}),
    DistractorSpec("snp_rs1800896", "binary", {"p": 0.25}),
    DistractorSpec("snp_rs1800629", "binary", {"p": 0.18}),
    DistractorSpec("snp_rs2228570", "binary", {"p": 0.35}),
    DistractorSpec("snp_rs1801131", "binary", {"p": 0.30}),
    DistractorSpec("snp_rs429358", "binary", {"p": 0.15}),
    DistractorSpec("snp_rs7412", "binary", {"p": 0.08}),
    DistractorSpec("snp_rs662", "binary", {"p": 0.30}),
    DistractorSpec("snp_rs2298771", "binary", {"p": 0.20}),
    DistractorSpec("snp_rs2032582", "binary", {"p": 0.42}),
    DistractorSpec("snp_rs1128503", "binary", {"p": 0.40}),
    DistractorSpec("snp_rs1800470", "binary", {"p": 0.45}),
    DistractorSpec("snp_rs1799983", "binary", {"p": 0.35}),
    DistractorSpec("snp_rs4880", "binary", {"p": 0.48}),
    DistractorSpec("snp_rs1050828", "binary", {"p": 0.08}),
    DistractorSpec("snp_rs4363657", "binary", {"p": 0.12}),
    DistractorSpec("snp_rs2070744", "binary", {"p": 0.40}),
    DistractorSpec("snp_rs6025", "binary", {"p": 0.03}),
    DistractorSpec("snp_rs1801197", "binary", {"p": 0.22}),
    DistractorSpec("snp_rs20417", "binary", {"p": 0.25}),
)


def _assert_pool_is_unique(pool: tuple[DistractorSpec, ...]) -> None:
    seen: set[str] = set()
    for spec in pool:
        if spec.name in seen:
            raise ValueError(f"Duplicate distractor name in pool: {spec.name!r}")
        seen.add(spec.name)


_assert_pool_is_unique(DEFAULT_DISTRACTOR_POOL)


def sample_distractors(
    rng: np.random.Generator,
    n_patients: int,
    n: int,
    pool: tuple[DistractorSpec, ...] = DEFAULT_DISTRACTOR_POOL,
) -> dict[str, np.ndarray]:
    """Return the first ``n`` distractor columns as a ``name -> array`` dict.

    Columns are sampled independently of each other and of the caller's own
    RNG state; the same ``rng`` is consumed in pool order so output is fully
    deterministic given the rng's seed.
    """
    if n < 0:
        raise ValueError(f"n_extra_covariates must be >= 0, got {n}")
    if n > len(pool):
        raise ValueError(
            f"Requested {n} distractor covariates but pool only has "
            f"{len(pool)}. Either lower n_extra_covariates or extend "
            f"DEFAULT_DISTRACTOR_POOL in src/onc_co_scientist/synthetic/distractors.py."
        )
    return {spec.name: _sample_one(rng, n_patients, spec) for spec in pool[:n]}
