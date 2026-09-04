"""Deterministic structural recovery and independent numerical confirmation.

No model, text parser, supplied significance flag, or answer-key repair is used.
The evaluator frame and manifest must use canonical (named) columns. Only the
candidate is unmasked, using the inverse of the stored named-to-masked map.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..harness.structured import Predicate, StructuredFinding, _satisfies
from ..synthetic.schemas import AssociationForm, AssociationSpec, DatasetManifest


def canonical_name(name: str | None, mapping: dict[str, str] | None) -> str | None:
    return {v: k for k, v in (mapping or {}).items()}.get(name, name)


def normalize_predicates(predicates: list[Predicate]) -> list[Predicate]:
    """Remove syntactic duplicates and logically redundant same-column bounds.

    Equivalence never follows merely from coincident masks on observed rows.
    """
    groups: dict[str, list[Predicate]] = {}
    for p in predicates:
        groups.setdefault(p.column, []).append(p)
    result = []
    for col, ps in sorted(groups.items()):
        allowed = [
            p.value if p.operator == "in" else [p.value] for p in ps if p.operator in {"eq", "in"}
        ]
        if allowed:
            values = [v for v in allowed[0] if all(_satisfies(v, p) for p in ps)]
            values = list({str(type(v)) + repr(v): v for v in values}.values())
            values.sort(key=repr)
            if not values:
                raise ValueError(f"Empty subgroup constraints for {col}.")
            result.append(
                Predicate(
                    column=col,
                    operator="eq" if len(values) == 1 else "in",
                    value=values[0] if len(values) == 1 else values,
                )
            )
            continue
        lower = [p for p in ps if p.operator in {"ge", "gt"}]
        upper = [p for p in ps if p.operator in {"le", "lt"}]
        bounds = []
        if lower:
            value = max(p.value for p in lower)
            op = "gt" if any(p.value == value and p.operator == "gt" for p in lower) else "ge"
            bounds.append(Predicate(column=col, operator=op, value=value))
        if upper:
            value = min(p.value for p in upper)
            op = "lt" if any(p.value == value and p.operator == "lt" for p in upper) else "le"
            bounds.append(Predicate(column=col, operator=op, value=value))
        if (
            len(bounds) == 2
            and bounds[0].value == bounds[1].value
            and bounds[0].operator == "ge"
            and bounds[1].operator == "le"
        ):
            bounds = [Predicate(column=col, operator="eq", value=bounds[0].value)]
        result.extend(bounds)
        excluded = []
        for p in ps:
            if p.operator == "ne":
                excluded.append(p.value)
            if p.operator == "not_in":
                excluded.extend(p.value)
        excluded = [v for v in excluded if all(_satisfies(v, p) for p in bounds)]
        excluded = list({str(type(v)) + repr(v): v for v in excluded}.values())
        excluded.sort(key=repr)
        if excluded:
            result.append(
                Predicate(
                    column=col,
                    operator="ne" if len(excluded) == 1 else "not_in",
                    value=excluded[0] if len(excluded) == 1 else excluded,
                )
            )
    return result


def _expected(spec: AssociationSpec) -> list[Predicate]:
    result = []
    for col, value in (spec.subgroup.predicate if spec.subgroup else {}).items():
        if isinstance(value, dict):
            if set(value) - {"min", "max"} or not value:
                raise ValueError("Unsupported answer-key predicate.")
            for key, op in [("min", "ge"), ("max", "le")]:
                if key in value:
                    result.append(Predicate(column=col, operator=op, value=value[key]))
        else:
            result.append(Predicate(column=col, operator="eq", value=value))
    return normalize_predicates(result)


def subgroup_mask(df: pd.DataFrame, predicates: list[Predicate]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for p in predicates:
        if p.column not in df:
            raise ValueError(f"Unknown subgroup column: {p.column}")
        s, v = df[p.column], p.value
        if p.operator == "eq":
            selected = s.eq(v)
        elif p.operator == "ne":
            selected = s.ne(v)
        elif p.operator == "lt":
            selected = s.lt(v)
        elif p.operator == "le":
            selected = s.le(v)
        elif p.operator == "gt":
            selected = s.gt(v)
        elif p.operator == "ge":
            selected = s.ge(v)
        elif p.operator == "in":
            selected = s.isin(v)
        else:
            selected = ~s.isin(v)
        mask &= selected.fillna(False) & s.notna()
    return mask


def _contrast(groups: list[np.ndarray], signs: list[int]) -> dict:
    effect = float(sum(sign * a.mean() for sign, a in zip(signs, groups, strict=True)))
    variances = np.array([a.var(ddof=1) / len(a) for a in groups])
    variance = float(variances.sum())
    if variance == 0:
        # A noiseless synthetic contrast is deterministic, not a noisy t estimate.
        return {
            "effect": effect,
            "p_value": 1.0 if effect == 0 else 0.0,
            "standard_error": 0.0,
            "df": None,
            "constant_cells": True,
        }
    denominator = sum(v**2 / (len(a) - 1) for v, a in zip(variances, groups, strict=True))
    degrees = float(variance**2 / denominator)
    se = math.sqrt(variance)
    return {
        "effect": effect,
        "p_value": float(2 * stats.t.sf(abs(effect / se), degrees)),
        "standard_error": se,
        "df": degrees,
        "constant_cells": False,
    }


def evaluate_finding(
    finding: StructuredFinding | dict,
    evaluation_df: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
    min_cell_n: int = 10,
) -> dict:
    """Truth-free Welch confirmation of the candidate's own outcome and contrast."""
    f = (
        finding
        if isinstance(finding, StructuredFinding)
        else StructuredFinding.model_validate(finding)
    )
    f = StructuredFinding.model_validate(
        {
            **f.model_dump(),
            "outcome": canonical_name(f.outcome, column_mapping),
            "exposure": canonical_name(f.exposure, column_mapping),
            "subgroup": [
                {**p.model_dump(), "column": canonical_name(p.column, column_mapping)}
                for p in f.subgroup
            ],
        }
    )
    if f.outcome not in evaluation_df:
        raise ValueError(f"Unknown outcome: {f.outcome}")
    mask = subgroup_mask(evaluation_df, f.subgroup)
    y = pd.to_numeric(evaluation_df[f.outcome], errors="coerce")
    usable = np.isfinite(y)
    # Missing subgroup covariates are unclassifiable, not members of the complement.
    for p in f.subgroup:
        usable &= evaluation_df[p.column].notna()
    if f.exposure is None:
        groups = [y[usable & mask].to_numpy(float), y[usable & ~mask].to_numpy(float)]
        labels, signs = ["subgroup", "complement"], [1, -1]
    else:
        if f.exposure not in evaluation_df:
            raise ValueError(f"Unknown exposure: {f.exposure}")
        t = pd.to_numeric(evaluation_df[f.exposure], errors="coerce")
        if not t.dropna().isin([0, 1]).all():
            raise ValueError("Treatment contrasts require a binary 0/1 exposure.")
        usable &= t.notna()
        groups = [y[usable & mask & t.eq(v)].to_numpy(float) for v in [1, 0]]
        labels, signs = ["inside_exposed", "inside_control"], [1, -1]
        if f.contrast == "treatment_interaction":
            groups += [y[usable & ~mask & t.eq(v)].to_numpy(float) for v in [1, 0]]
            labels += ["outside_exposed", "outside_control"]
            signs += [-1, 1]
    counts = {label: len(a) for label, a in zip(labels, groups, strict=True)}
    if any(len(a) < min_cell_n for a in groups):
        return {
            "effect": None,
            "p_value": None,
            "finite": False,
            "cell_n": counts,
            "reason": "insufficient cell size",
        }
    result = _contrast(groups, signs)
    result.update(
        cell_n=counts,
        finite=math.isfinite(result["effect"]) and math.isfinite(result["p_value"]),
        contrast=f.contrast,
        estimand="mean difference" if len(groups) == 2 else "difference of mean differences",
    )
    return result


def _same_value(a: Any, b: Any, atol: float) -> bool:
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return abs(float(a) - float(b)) <= atol
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(any(_same_value(x, y, atol) for y in b) for x in a)
    return a == b


def _structure(candidate: list[Predicate], expected: list[Predicate], atol: float) -> dict:
    candidate = normalize_predicates(candidate)
    used, matched, missing, approximate = set(), [], [], []
    for expected_pred in expected:
        exact = next(
            (
                j
                for j, p in enumerate(candidate)
                if j not in used
                and p.column == expected_pred.column
                and p.operator == expected_pred.operator
                and _same_value(p.value, expected_pred.value, atol)
            ),
            None,
        )
        approx = next(
            (
                j
                for j, p in enumerate(candidate)
                if j not in used
                and p.column == expected_pred.column
                and (
                    (p.operator in {"ge", "gt"} and expected_pred.operator in {"ge", "gt"})
                    or (p.operator in {"le", "lt"} and expected_pred.operator in {"le", "lt"})
                )
            ),
            None,
        )
        if exact is not None:
            used.add(exact)
            matched.append(expected_pred.model_dump())
        elif approx is not None:
            used.add(approx)
            approximate.append(
                {
                    "expected": expected_pred.model_dump(),
                    "submitted": candidate[approx].model_dump(),
                }
            )
        else:
            missing.append(expected_pred.model_dump())
    extra = [p.model_dump() for j, p in enumerate(candidate) if j not in used]
    return {
        "matched": matched,
        "approximate": approximate,
        "missing": missing,
        "extra": extra,
        "complete": not missing and not extra,
        "predicate_equivalent": not missing and not extra and not approximate,
    }


def score_finding(
    finding: StructuredFinding | dict,
    spec: AssociationSpec,
    manifest: DatasetManifest,
    evaluation_df: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
    config: dict | None = None,
) -> dict:
    cfg = {
        "precision_min": 0.90,
        "recall_min": 0.90,
        "min_cell_n": 10,
        "numeric_atol": 1e-9,
        "alpha": 0.05,
        **(config or {}),
    }
    try:
        f = (
            finding
            if isinstance(finding, StructuredFinding)
            else StructuredFinding.model_validate(finding)
        )
        candidate = [
            Predicate(
                column=canonical_name(p.column, column_mapping), operator=p.operator, value=p.value
            )
            for p in f.subgroup
        ]
        expected = _expected(spec)
        structure = _structure(candidate, expected, cfg["numeric_atol"])
        truth_mask, proposed_mask = (
            subgroup_mask(evaluation_df, expected),
            subgroup_mask(evaluation_df, candidate),
        )
        tp = int((truth_mask & proposed_mask).sum())
        fp, fn = int((~truth_mask & proposed_mask).sum()), int((truth_mask & ~proposed_mask).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        evidence = evaluate_finding(f, evaluation_df, column_mapping, cfg["min_cell_n"])
        treatments = [v for v in spec.variables if v in manifest.treatment_columns]
        if len(treatments) > 1:
            raise ValueError("This scorer requires one prespecified treatment exposure.")
        exposure = treatments[0] if treatments else None
        expected_contrast = (
            "subgroup_difference"
            if exposure is None
            else "treatment_interaction"
            if spec.form in {AssociationForm.interaction, AssociationForm.subgroup_conditional}
            else "treatment_effect"
        )
        structure.update(
            target=canonical_name(f.outcome, column_mapping) == spec.outcome,
            exposure=canonical_name(f.exposure, column_mapping) == exposure,
            contrast=f.contrast == expected_contrast,
            direction=f.direction == spec.direction,
        )
        target = all(structure[k] for k in ["target", "exposure", "contrast", "direction"])
        matched = bool(
            target
            and structure["complete"]
            and precision >= cfg["precision_min"]
            and recall >= cfg["recall_min"]
        )
        strict = bool(target and structure["predicate_equivalent"])
        effect, p = evidence["effect"], evidence["p_value"]
        correct_direction = bool(
            evidence["finite"] and spec.direction != 0 and effect * spec.direction > 0
        )
        functional = {
            **evidence,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
        return {
            "valid": True,
            "match_recovered": matched,
            "strict_match": strict,
            "recovered": bool(matched and correct_direction and p < cfg["alpha"]),
            "evidence_direction_ok": correct_direction,
            "structural": structure,
            "functional": functional,
        }
    except (ValueError, TypeError, KeyError) as exc:
        return {
            "valid": False,
            "reason": str(exc),
            "recovered": False,
            "match_recovered": False,
            "strict_match": False,
            "structural": {},
            "functional": {},
        }


def compute_evidence(finding, manifest, evaluation_df, column_mapping=None):
    """Compatibility alias; manifest is not read by this truth-free helper."""
    return evaluate_finding(finding, evaluation_df, column_mapping)
