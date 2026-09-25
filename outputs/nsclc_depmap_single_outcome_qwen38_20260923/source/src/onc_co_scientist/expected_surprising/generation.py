"""Explicit additive DGPs and paired outcome generation.

All scored contrasts are observable mean differences (or differences of mean
differences), evaluated under the entire model. They carry no causal claim.
Clinical survival is analyzed on the log-month scale in this first release.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..synthetic.cancer_types import get_profile
from ..synthetic.generator import GeneratorConfig, _append_distractor_covariates
from .review_policy import reject_nested_comparison, require_current_candidate, require_current_pair
from .schemas import FULL_RUN_ITERATIONS, Condition, Discovery, Outcome, PairSpec, ReviewedCandidate


def base_frame(profile: str, n: int, seed: int, *, version: int = 1) -> pd.DataFrame:
    config = GeneratorConfig(dataset_id="paired", cancer_type=profile, patient_n=n, seed=seed)
    cancer = get_profile(profile)
    frame = _append_distractor_covariates(cancer.base_frame_fn(config), config, cancer)
    frame = frame.drop(columns=[c for c in frame if c.startswith(("dependency_", "__internal_"))])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 814]))
    for name in ("research_signature_a", "research_signature_b", "research_signature_c"):
        frame[name] = rng.normal(size=n)
    if version >= 2:
        # Constructed assay markers have no assigned gene/pathway or established
        # directional role. They supply neutral parent comparisons, rather than
        # renaming an established association through arbitrary subgroup cutoffs.
        marker_rng = np.random.default_rng(np.random.SeedSequence([seed, 816]))
        for label in ("d", "e", "f"):
            frame[f"research_marker_{label}_positive"] = marker_rng.binomial(1, 0.5, n)
        if "ecog_ps" in frame:
            frame["ecog_ps_ge_2"] = (frame["ecog_ps"] >= 2).astype(int)
        # Candidate contrast encodings, not asserted prognostic thresholds.
        # The literature and realism gates must support the exact cutoff used.
        for source, cutoff, operator, label in (
            ("crp_mg_l", 10, "ge", "crp_mg_l_ge_10"),
            ("albumin_g_dl", 3.5, "le", "albumin_g_dl_le_3_5"),
            ("nlr", 3, "ge", "nlr_ge_3"),
        ):
            if source in frame:
                values = frame[source] >= cutoff if operator == "ge" else frame[source] <= cutoff
                frame[label] = values.astype(int)
    if profile.endswith("depmap"):
        # Development prevalences provide recoverable representation; these
        # are simulation design values, not estimates of population prevalence.
        feature_rng = np.random.default_rng(np.random.SeedSequence([seed, 815]))
        for name in ("tp53_loss", "mtap_loss", "smarca4_loss", "arid1a_loss", "keap1_loss"):
            frame[name] = feature_rng.binomial(1, 0.20, n)
        disease_features = {
            "prostate_depmap": (
                "ar_amplification",
                "androgen_receptor_positive",
                "tmprss2_erg_fusion",
                "cdk12_loss",
            ),
            "aml_depmap": (
                "flt3_itd",
                "npm1_mutation",
                "kmt2a_rearrangement",
                "idh1_mutation",
                "idh2_mutation",
                "dnmt3a_mutation",
            ),
        }
        for name in disease_features.get(profile, ()):
            frame[name] = feature_rng.binomial(1, 0.30, n)
    return frame


def mask(frame: pd.DataFrame, conditions: list[Condition]) -> np.ndarray:
    result = np.ones(len(frame), dtype=bool)
    for c in conditions:
        if c.variable not in frame:
            raise ValueError(f"Unknown condition variable: {c.variable}")
        values = frame[c.variable]
        if c.op == "eq":
            selected = values == c.value
        elif c.op == "ge":
            selected = values >= c.value
        else:
            selected = values <= c.value
        result &= selected.fillna(False).to_numpy(dtype=bool)
    return result


def version_discoveries(spec: PairSpec, version: str) -> list[Discovery]:
    if version not in {"expected", "surprising"}:
        raise ValueError("Version must be expected or surprising")
    discoveries = [d.model_copy(deep=True) for d in spec.discoveries]
    if version == "surprising":
        d = next(d for d in discoveries if d.hypothesis.id == spec.focal_id)
        d.category = "surprising"
        d.hypothesis.direction *= -1
    return discoveries


def conditional_means(spec: PairSpec, frame: pd.DataFrame, version: str) -> dict[str, np.ndarray]:
    means = {o.name: np.full(len(frame), o.intercept, dtype=float) for o in spec.outcomes}
    for d in version_discoveries(spec, version):
        h = d.hypothesis
        x = (frame[h.exposure] == h.exposed).to_numpy(dtype=float)
        eligible = mask(frame, h.eligibility)
        subgroup = mask(frame, h.subgroup)
        # For interaction targets, beta is the added subgroup slope. For a
        # subgroup mean difference, beta is the complete slope inside S.
        inside = h.direction * d.magnitude
        if h.contrast == "interaction":
            slope = d.outside_coefficient + inside * subgroup
        else:
            slope = np.where(subgroup, inside, d.outside_coefficient)
        means[h.outcome] += x * eligible * slope
    return means


def sample(spec: PairSpec, version: str, *, seed: int, n: int | None = None) -> pd.DataFrame:
    frame = base_frame(spec.profile, n or spec.n, seed, version=spec.generation_version)
    means = conditional_means(spec, frame, version)
    # Stream identity is keyed by outcome name; adding an unrelated endpoint
    # cannot move another endpoint's residual draws.
    for o in spec.outcomes:
        key = int.from_bytes(hashlib.sha256(o.name.encode()).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 191, key]))
        frame[o.name] = means[o.name] + rng.normal(0, o.sigma, len(frame))
    return frame


def compile_pair(
    profile: str,
    reviewed: list[ReviewedCandidate],
    *,
    seed: int,
    n: int,
    focal_mode: str = "overall",
    pair_id: str | None = None,
    expected_count: int = 4,
) -> PairSpec:
    expected = [r for r in reviewed if r.candidate.proposed_category == "expected"]
    neutral = [r for r in reviewed if r.candidate.proposed_category == "neutral"]
    if expected_count not in {3, 4}:
        raise ValueError("Expected candidate count must be three or four")
    neutral_count = 6 - expected_count
    if len(expected) < expected_count or len(neutral) < neutral_count:
        raise ValueError(
            f"Need {expected_count} supported expected and {neutral_count} neutral candidates"
        )
    selected = expected[:expected_count] + neutral[:neutral_count]
    for i, item in enumerate(selected):
        require_current_candidate(item, profile)
        reject_nested_comparison(item.candidate, selected[:i])
    depmap = profile.endswith("depmap")
    magnitude, sigma, delta = (0.8, 0.18, 0.15) if depmap else (0.45, 0.30, 0.10)
    discoveries = []
    for i, r in enumerate(selected):
        h = r.candidate.hypothesis.model_copy(deep=True)
        category = (
            "neutral"
            if i >= expected_count
            else ("surprising" if i == expected_count - 1 else "expected")
        )
        expected_direction = h.direction if i < expected_count else None
        if category == "surprising":
            h.direction *= -1
        outside = 0.0
        if i == 0 and focal_mode == "subgroup":
            if h.subgroup or h.contrast != "mean_difference":
                raise ValueError("Focal parent must be an overall mean difference")
            h.subgroup = [
                Condition(variable="research_signature_a", op="ge", value=0),
                Condition(variable="research_signature_b", op="ge", value=0),
            ]
            outside = expected_direction * magnitude
        discoveries.append(
            Discovery(
                hypothesis=h,
                category=category,
                expected_direction=expected_direction,
                magnitude=magnitude,
                outside_coefficient=outside,
                evidence_id=r.candidate.hypothesis.id,
                paradigm_bearing_variables=(
                    [c.variable for c in h.subgroup] if i == 0 and focal_mode == "subgroup" else []
                ),
            )
        )
    outcome_names = {d.hypothesis.outcome for d in discoveries}
    if depmap:
        outcome_names.update(
            f"dependency_{gene}"
            for gene in (
                "BRAF",
                "EGFR",
                "ERBB2",
                "KRAS",
                "RIT1",
                "KIF18A",
                "TMED10",
                "DHX9",
                "POLR2A",
                "RPL11",
            )
        )
    outcomes = [
        Outcome(
            name=name,
            intercept=-0.25 if depmap else 2.5,
            sigma=sigma,
            delta=delta,
            units="CRISPR dependency score" if depmap else "log(months)",
        )
        for name in sorted(outcome_names)
    ]
    return PairSpec(
        schema_version="2.0.0",
        generation_version=2,
        pair_id=pair_id or f"es-v2-{profile}-{seed}",
        profile=profile,
        seed=seed,
        n=n,
        focal_id=discoveries[0].hypothesis.id,
        focal_mode=focal_mode,
        discoveries=discoveries,
        outcomes=outcomes,
        evidence=selected,
    )


def equations(spec: PairSpec, version: str) -> str:
    lines = [
        f"Pair: {spec.pair_id}; condition: {version}",
        "I[...] denotes a Boolean indicator. All coefficients below are on the outcome scale.",
    ]
    for o in spec.outcomes:
        lines.append(
            f"\n{o.name} = {o.intercept} + "
            + " + ".join(
                d.hypothesis.id for d in spec.discoveries if d.hypothesis.outcome == o.name
            )
            + f" + Normal(0, {o.sigma}^2)"
        )
    for d in version_discoveries(spec, version):
        h = d.hypothesis
        s = " AND ".join(f"{c.variable} {c.op} {c.value}" for c in h.subgroup) or "TRUE"
        e = " AND ".join(f"{c.variable} {c.op} {c.value}" for c in h.eligibility) or "TRUE"
        beta = h.direction * d.magnitude
        slope = (
            f"({d.outside_coefficient} + {beta}*I[{s}])"
            if h.contrast == "interaction"
            else f"({beta}*I[{s}] + {d.outside_coefficient}*I[NOT({s})])"
        )
        lines.append(f"{h.id} = I[{h.exposure} == {h.exposed}]*I[{e}]*{slope}")
    return "\n".join(lines) + "\n"


def write_pair(spec: PairSpec, root: Path, *, historical_replay: bool = False) -> dict:
    """Write opaque public tasks; retain assignment, literature and DGP privately."""
    if not historical_replay:
        require_current_pair(spec)
    private = root / "private" / spec.pair_id
    if private.exists():
        raise FileExistsError(f"Pair already exists: {private}")
    private.mkdir(parents=True)
    (private / "pair.json").write_text(spec.model_dump_json(indent=2))
    mapping = {}
    for version in ("expected", "surprising"):
        task_id = hashlib.sha256(f"{spec.pair_id}:{version}".encode()).hexdigest()[:16]
        public = root / "public" / task_id
        public.mkdir(parents=True, exist_ok=False)
        frame = sample(spec, version, seed=spec.seed)
        frame.to_parquet(public / "dataset.parquet", index=False)
        dictionary = {
            c: {"dtype": str(frame[c].dtype), "missing_n": int(frame[c].isna().sum())}
            for c in frame
        }
        (public / "data_dictionary.json").write_text(json.dumps(dictionary, indent=2))
        task = {
            "task_id": task_id,
            "profile": spec.profile,
            "n": spec.n,
            "iterations": (
                10 if historical_replay and spec.profile.endswith("depmap") else FULL_RUN_ITERATIONS
            ),
            "validation_limit": 10,
            "outcomes": [o.model_dump(exclude={"intercept", "sigma"}) for o in spec.outcomes],
        }
        (public / "task.json").write_text(json.dumps(task, indent=2))
        (public / "instructions.md").write_text(
            "Identify scientifically meaningful discoveries through iterative hypothesis "
            "generation, "
            "analysis, and evidence appraisal. Record hypotheses before analysis and link every "
            "decision to its result. State associations on the supplied endpoint scale. These "
            "synthetic data support descriptive associations. Clinical log_pfs_months is natural "
            "log progression-free survival in months, fully observed without censoring. "
            "Dependency scores are continuous; more negative values mean stronger dependency. "
            "Research signatures are constructed standardized assays with arbitrary units. "
            "Research markers D, E, and F, when present, are constructed binary assays with "
            "no assigned gene, pathway, or clinical role. "
            "Conditions define eligibility and subgroups. A mean_difference compares exposed "
            "minus comparator within eligibility and subgroup. An interaction subtracts that "
            "comparison in the subgroup complement within the same eligible population. "
            "A signed effect must exceed the endpoint's delta to support acceptance. You may "
            "request up to ten independent validation results, at most one per iteration. "
            "Each uses a 99.5% interval. Accept if its lower bound exceeds delta; reject the "
            "minimum-effect claim if its upper bound is below delta; otherwise remain unresolved.\n"
        )
        (private / f"{version}_equation.txt").write_text(equations(spec, version))
        mapping[version] = {
            "task_id": task_id,
            "sha256": hashlib.sha256((public / "dataset.parquet").read_bytes()).hexdigest(),
        }
    (private / "assignment.json").write_text(json.dumps(mapping, indent=2))
    return mapping
