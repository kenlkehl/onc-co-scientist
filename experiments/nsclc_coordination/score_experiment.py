#!/usr/bin/env python3
"""Deterministically score the named clinical NSCLC coordination experiment.

The scorer is deliberately answer-key-specific and contains no model calls.  It
reads the co-scientist harness ``run.json``/``artifacts.json`` outputs, scores
each scientific checkpoint against the planted sotorasib association, and
writes run-level and workflow-level summaries.

Structured ``artifact.claims`` records are preferred.  A conservative text
fallback supports legacy artifacts, but it never infers evidentiary support
from a hypothesis statement alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

RecoveryLevel = Literal["none", "component", "near", "exact"]

SCORER_VERSION = "nsclc-semantic-workflow-grid-v2"
ASSOCIATION_ID = "buried_sotorasib_krasg12c_alkwt_brca2wt_male"
DATASET_ID = "ds001_nsclc"
EXPOSURE = "treatment_sotorasib"
OUTCOME = "pfs_months"
EXPECTED_PREDICATES: dict[str, int] = {
    "kras_g12c": 1,
    "alk_fusion": 0,
    "brca2_mutation": 0,
    "sex_female": 0,
}
PLANTED_EFFECT_MONTHS = 5.0
OBSERVED = {
    "subgroup_n": 3266,
    "exposed_n": 1154,
    "comparator_n": 2112,
    "mean_difference_months": 4.985,
}

LEVEL_RANK: dict[RecoveryLevel, int] = {
    "none": 0,
    "component": 1,
    "near": 2,
    "exact": 3,
}
BASE_STAGES = ("hypothesis_generation", "analysis", "critique", "synthesis")
FINAL_STAGES = {"synthesis", "synthesis_consensus", "federated_synthesis"}


@dataclass(frozen=True)
class TruthSpec:
    """Condition-native answer key loaded only in the evaluator process."""

    dataset_id: str
    association_id: str
    semantic_condition: str
    exposure: str
    outcome: str
    direction: int
    subgroup_predicates: dict[str, int]
    effect_size: float | None = None
    manifest_sha256: str | None = None
    display_mapping: dict[str, str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "association_id": self.association_id,
            "semantic_condition": self.semantic_condition,
            "exposure": self.exposure,
            "outcome": self.outcome,
            "direction": self.direction,
            "subgroup_predicates": dict(self.subgroup_predicates),
            "effect_size": self.effect_size,
            "manifest_sha256": self.manifest_sha256,
            "display_mapping": dict(self.display_mapping or {}),
        }


DEFAULT_TRUTH = TruthSpec(
    dataset_id=DATASET_ID,
    association_id=ASSOCIATION_ID,
    semantic_condition="named",
    exposure=EXPOSURE,
    outcome=OUTCOME,
    direction=1,
    subgroup_predicates=dict(EXPECTED_PREDICATES),
    effect_size=PLANTED_EFFECT_MONTHS,
)


@dataclass(frozen=True)
class ClaimScore:
    """One claim's semantic and evidence-supported recovery result."""

    claim_id: str
    text: str
    recovery_level: RecoveryLevel
    supported: bool
    declared_supported: bool | None
    matched_predicates: tuple[str, ...]
    missing_predicates: tuple[str, ...]
    contradictory_predicates: tuple[str, ...]
    source: Literal["structured", "text"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "recovery_level": self.recovery_level,
            "supported": self.supported,
            "declared_supported": self.declared_supported,
            "matched_predicates": list(self.matched_predicates),
            "missing_predicates": list(self.missing_predicates),
            "contradictory_predicates": list(self.contradictory_predicates),
            "source": self.source,
        }


def truth_record(truth: TruthSpec = DEFAULT_TRUTH) -> dict[str, Any]:
    """Return the frozen answer key and observed seed-0 descriptive values."""

    payload = truth.as_dict()
    payload["direction_label"] = "positive_benefit" if truth.direction > 0 else "negative"
    if truth == DEFAULT_TRUTH:
        payload["observed"] = dict(OBSERVED)
    return payload


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_truth(
    manifest_path: Path,
    *,
    semantic_condition: str,
    column_mapping_path: Path | None = None,
) -> TruthSpec:
    """Derive the condition-native target from one private evaluator manifest."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    associations = manifest.get("associations", [])
    if not isinstance(associations, list) or len(associations) != 1:
        raise ValueError(f"Expected exactly one NSCLC association in {manifest_path}")
    association = associations[0]
    variables = {str(item) for item in association.get("variables", [])}
    treatments = {str(item) for item in manifest.get("treatment_columns", [])}
    exposures = sorted(variables.intersection(treatments))
    if len(exposures) != 1:
        raise ValueError(f"Could not derive one target exposure from {manifest_path}")
    subgroup = association.get("subgroup", {})
    predicates = subgroup.get("predicate", {}) if isinstance(subgroup, Mapping) else {}
    if not isinstance(predicates, Mapping) or not predicates:
        raise ValueError(f"Target association has no subgroup predicates: {manifest_path}")
    direction = int(association.get("direction", 0))
    if direction not in {-1, 1}:
        raise ValueError(f"Target direction must be -1 or 1: {manifest_path}")
    display_mapping: dict[str, str] | None = None
    if column_mapping_path is not None:
        named_to_masked = json.loads(column_mapping_path.read_text(encoding="utf-8"))
        if not isinstance(named_to_masked, Mapping):
            raise ValueError(f"Malformed evaluator column mapping: {column_mapping_path}")
        inverse = {str(masked): str(named) for named, masked in named_to_masked.items()}
        if semantic_condition == "masked":
            display_mapping = inverse
    effect = _finite_number(association.get("effect_size"))
    return TruthSpec(
        dataset_id=str(manifest.get("dataset_id", "unknown")),
        association_id=str(association.get("id", "unknown")),
        semantic_condition=semantic_condition,
        exposure=exposures[0],
        outcome=str(association.get("outcome", "")),
        direction=direction,
        subgroup_predicates={str(key): int(value) for key, value in predicates.items()},
        effect_size=effect,
        manifest_sha256=_file_sha256(manifest_path),
        display_mapping=display_mapping,
    )


def _normal_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _binary_value(value: Any, *, variable: str = "") -> int | None:
    if isinstance(value, bool):
        return int(value)
    number = _finite_number(value)
    if number in {0.0, 1.0}:
        return int(number)
    token = _normal_name(value)
    if variable == "sex_female":
        if token in {"male", "man", "men", "m", "not_female"}:
            return 0
        if token in {"female", "woman", "women", "f"}:
            return 1
    if token in {
        "1",
        "true",
        "yes",
        "positive",
        "present",
        "mutant",
        "mutated",
        "fusion_positive",
    }:
        return 1
    if token in {
        "0",
        "false",
        "no",
        "negative",
        "absent",
        "wild_type",
        "wildtype",
        "wt",
        "unmutated",
        "fusion_negative",
    }:
        return 0
    return None


def _canonical_predicate_name(value: Any, truth: TruthSpec) -> str | None:
    token = _normal_name(value)
    native = {_normal_name(name): name for name in truth.subgroup_predicates}
    if token in native:
        return native[token]
    aliases = {
        "kras_g12c": "kras_g12c",
        "krasg12c": "kras_g12c",
        "kras_g12c_status": "kras_g12c",
        "alk": "alk_fusion",
        "alk_status": "alk_fusion",
        "alk_fusion": "alk_fusion",
        "alkfusion": "alk_fusion",
        "brca2": "brca2_mutation",
        "brca2_status": "brca2_mutation",
        "brca2_mutation": "brca2_mutation",
        "brca2mutation": "brca2_mutation",
        "sex": "sex_female",
        "gender": "sex_female",
        "male": "sex_female",
        "sex_female": "sex_female",
        "sexfemale": "sex_female",
    }
    candidate = aliases.get(token)
    return candidate if candidate in truth.subgroup_predicates else None


def _predicate_values_from_text(text: str, truth: TruthSpec) -> dict[str, int]:
    """Extract only explicit binary subgroup assignments from natural language."""

    lowered = text.lower()
    compact = re.sub(r"[`*]", "", lowered)
    found: dict[str, int] = {}

    correct_patterns: dict[str, tuple[str, ...]] = {
        "kras_g12c": (
            r"kras[_\s-]*g12c\s*(?:=|is|:)?\s*(?:1|positive|present|mutant|mutated)",
            r"kras[_\s-]*g12c[-\s]*(?:mutant|mutated|positive)",
            r"(?:with|among)\s+(?:patients\s+)?(?:with\s+)?kras[_\s-]*g12c",
        ),
        "alk_fusion": (
            r"alk[_\s-]*(?:fusion)?\s*(?:=|is|:)?\s*(?:0|negative|absent|wild[-\s]*type|wt)",
            r"alk[-\s]*(?:wild[-\s]*type|wt|negative)",
            r"without\s+(?:an?\s+)?alk[_\s-]*fusion",
        ),
        "brca2_mutation": (
            r"brca2[_\s-]*(?:mutation|status)?\s*(?:=|is|:)?\s*(?:0|negative|absent|wild[-\s]*type|wt|unmutated)",
            r"brca2[-\s]*(?:wild[-\s]*type|wt|negative|unmutated)",
            r"without\s+(?:a\s+)?brca2[_\s-]*mutation",
        ),
        "sex_female": (
            r"sex[_\s-]*female\s*(?:=|is|:)?\s*0",
            r"\b(?:male|males|men)\b",
            r"\bnot\s+female\b",
        ),
    }
    wrong_patterns: dict[str, tuple[str, ...]] = {
        "kras_g12c": (
            r"kras[_\s-]*g12c\s*(?:=|is|:)?\s*(?:0|negative|absent|wild[-\s]*type|wt)",
            r"without\s+(?:a\s+)?kras[_\s-]*g12c",
        ),
        "alk_fusion": (
            r"alk[_\s-]*fusion\s*(?:=|is|:)?\s*(?:1|positive|present)",
            r"alk[-\s]*fusion[-\s]*(?:positive|present)",
        ),
        "brca2_mutation": (
            r"brca2[_\s-]*mutation\s*(?:=|is|:)?\s*(?:1|positive|present|mutant|mutated)",
            r"brca2[-\s]*(?:mutant|mutated|positive)",
        ),
        "sex_female": (
            r"sex[_\s-]*female\s*(?:=|is|:)?\s*1",
            r"\b(?:female|females|women)\b",
        ),
    }

    for variable, patterns in correct_patterns.items():
        if variable in truth.subgroup_predicates and any(
            re.search(pattern, compact) for pattern in patterns
        ):
            found[variable] = truth.subgroup_predicates[variable]
    for variable, patterns in wrong_patterns.items():
        if variable in truth.subgroup_predicates and any(
            re.search(pattern, compact) for pattern in patterns
        ):
            # Explicit contradictions take precedence over a loose positive pattern.
            found[variable] = 1 - truth.subgroup_predicates[variable]
    for variable, expected in truth.subgroup_predicates.items():
        escaped = re.escape(variable).replace(r"\_", r"[_\s-]*")
        expected_words = "true|yes|positive|present" if expected else "false|no|negative|absent"
        correct = (
            rf"\b{escaped}\b\s*(?:=|is|:)?\s*"
            rf"(?:{expected}|{expected_words})\b"
        )
        wrong_value = 1 - expected
        wrong_words = (
            "true|yes|positive|present" if wrong_value else "false|no|negative|absent"
        )
        wrong = (
            rf"\b{escaped}\b\s*(?:=|is|:)?\s*"
            rf"(?:{wrong_value}|{wrong_words})\b"
        )
        if re.search(correct, compact):
            found[variable] = expected
        if re.search(wrong, compact):
            found[variable] = wrong_value
    return found


def _structured_predicates(
    claim: Mapping[str, Any], text: str, truth: TruthSpec
) -> tuple[dict[str, int], set[str]]:
    raw = None
    for key in ("subgroup_predicates", "predicates", "subgroup", "modifiers"):
        if key in claim and claim[key] not in (None, "", [], {}):
            raw = claim[key]
            break

    parsed: dict[str, int] = {}
    extras: set[str] = set()

    def add(name: Any, value: Any, operator: Any = "eq") -> None:
        token = _normal_name(name)
        if token in {"cancer_type", "cohort", "disease", "lineage"}:
            return
        canonical = _canonical_predicate_name(name, truth)
        if canonical is None:
            extras.add(token or "unknown")
            return
        if canonical == "sex_female" and token in {"male"} and value in (None, ""):
            parsed[canonical] = 0
            return
        binary = _binary_value(value, variable=canonical)
        if binary is not None:
            normalized_operator = _normal_name(operator or "eq")
            if normalized_operator == "eq":
                parsed[canonical] = binary
            elif normalized_operator == "ne":
                parsed[canonical] = 1 - binary
            else:
                # Threshold/set operators on a binary answer-key predicate are
                # not treated as equivalent without an explicit normalization.
                parsed[canonical] = 1 - truth.subgroup_predicates[canonical]

    if isinstance(raw, Mapping):
        # A single {variable, value} predicate and a mapping of predicates are both accepted.
        if any(key in raw for key in ("variable", "column", "field", "name")):
            name = next(
                (raw[key] for key in ("variable", "column", "field", "name") if key in raw),
                "",
            )
            value = next(
                (raw[key] for key in ("value", "equals", "level", "status") if key in raw),
                None,
            )
            add(name, value, raw.get("operator", "eq"))
        else:
            for name, value in raw.items():
                add(name, value)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping):
                name = next(
                    (item[key] for key in ("variable", "column", "field", "name") if key in item),
                    "",
                )
                value = next(
                    (item[key] for key in ("value", "equals", "level", "status") if key in item),
                    None,
                )
                add(name, value, item.get("operator", "eq"))
            elif isinstance(item, str):
                parsed.update(_predicate_values_from_text(item, truth))
    elif isinstance(raw, str):
        parsed.update(_predicate_values_from_text(raw, truth))

    # Natural language may complete an otherwise partially structured claim.
    for name, value in _predicate_values_from_text(text, truth).items():
        parsed.setdefault(name, value)
    return parsed, extras


def _is_exposure(value: Any, truth: TruthSpec) -> bool:
    token = _normal_name(value)
    expected = _normal_name(truth.exposure)
    aliases = {expected, f"{expected}_1", f"{expected}_true"}
    if expected.startswith("treatment_"):
        short = expected.removeprefix("treatment_")
        aliases.update({short, f"{short}_1", f"{short}_true"})
    return token in aliases


def _is_outcome(value: Any, truth: TruthSpec) -> bool:
    token = _normal_name(value)
    expected = _normal_name(truth.outcome)
    aliases = {expected}
    if expected == "pfs_months":
        aliases.update({"pfs", "progression_free_survival", "progression_free_survival_months"})
    return token in aliases


def _direction(value: Any) -> int:
    number = _finite_number(value)
    if number is not None:
        return 1 if number > 0 else (-1 if number < 0 else 0)
    token = _normal_name(value)
    positive_words = ("positive", "benefit", "longer", "higher", "increase", "improv")
    if any(word in token for word in positive_words):
        return 1
    if any(word in token for word in ("negative", "harm", "shorter", "lower", "decrease", "worse")):
        return -1
    return 0


def _text_direction(text: str) -> int:
    lowered = text.lower()
    wrong = (
        r"(?:shorter|worse|decreased?|reduced?)\s+(?:median\s+|mean\s+)?"
        r"(?:pfs|progression[-\s]+free survival)",
        r"(?:negative|harmful)\s+(?:treatment\s+)?effect",
        r"(?:pfs|progression[-\s]+free survival).{0,25}(?:shorter|worse|decreased?)",
    )
    right = (
        r"(?:longer|higher|increased?|improved?|extended?)\s+(?:median\s+|mean\s+)?"
        r"(?:pfs|progression[-\s]+free survival)",
        r"(?:positive|beneficial)\s+(?:treatment\s+)?effect",
        r"(?:pfs|progression[-\s]+free survival).{0,25}(?:longer|higher|increased?|improved?)",
        r"\bbenefit(?:ed|s|ing)?\b",
        r"\+\s*\d+(?:\.\d+)?\s*(?:months?|mo)\b",
        r"\d+(?:\.\d+)?\s*(?:months?|mo)\s+longer\b",
    )
    if any(re.search(pattern, lowered) for pattern in wrong):
        return -1
    if any(re.search(pattern, lowered) for pattern in right):
        return 1
    return 0


def _text_mentions_exposure(text: str, truth: TruthSpec) -> bool:
    names = [truth.exposure]
    if truth.exposure.startswith("treatment_"):
        names.append(truth.exposure.removeprefix("treatment_"))
    patterns = [re.escape(name).replace("_", r"[_\s-]*") for name in names]
    return any(re.search(rf"\b{pattern}\b", text.lower()) for pattern in patterns)


def _text_mentions_outcome(text: str, truth: TruthSpec) -> bool:
    expected = re.escape(truth.outcome).replace("_", r"[_\s-]*")
    if _normal_name(truth.outcome) == "pfs_months":
        expected = r"(?:pfs(?:[_\s-]*months)?|progression[-\s]+free survival)"
    return bool(re.search(rf"\b{expected}\b", text.lower()))


def _canonical_claim(
    claim: Mapping[str, Any], text: str, truth: TruthSpec
) -> dict[str, Any]:
    estimand = claim.get("estimand") if isinstance(claim.get("estimand"), Mapping) else {}
    exposure = next(
        (
            claim[key]
            for key in ("exposure", "driver", "treatment")
            if key in claim and claim[key] not in (None, "")
        ),
        estimand.get("exposure") or estimand.get("treatment"),
    )
    outcome = next(
        (
            claim[key]
            for key in ("outcome", "endpoint")
            if key in claim and claim[key] not in (None, "")
        ),
        estimand.get("outcome") or estimand.get("endpoint"),
    )
    direction_value = next(
        (
            claim[key]
            for key in ("direction", "effect_direction")
            if key in claim and claim[key] not in (None, "")
        ),
        estimand.get("direction"),
    )
    direction = _direction(direction_value)
    if not direction:
        direction = _direction(
            next(
                (claim[key] for key in ("effect_estimate", "estimate", "effect") if key in claim),
                None,
            )
        )
    if not direction:
        direction = _text_direction(text)
    predicates, extras = _structured_predicates(claim, text, truth)
    return {
        "exposure_ok": (
            _is_exposure(exposure, truth)
            if exposure not in (None, "")
            else _text_mentions_exposure(text, truth)
        ),
        "outcome_ok": (
            _is_outcome(outcome, truth)
            if outcome not in (None, "")
            else _text_mentions_outcome(text, truth)
        ),
        "direction": direction,
        "predicates": predicates,
        "extra_predicates": extras,
    }


def _recovery_level(
    canonical: Mapping[str, Any], truth: TruthSpec
) -> tuple[RecoveryLevel, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    predicates = canonical["predicates"]
    matched = tuple(
        name
        for name, expected in truth.subgroup_predicates.items()
        if predicates.get(name) == expected
    )
    contradictory = tuple(
        name
        for name, expected in truth.subgroup_predicates.items()
        if name in predicates and predicates[name] != expected
    )
    missing = tuple(name for name in truth.subgroup_predicates if name not in predicates)
    extra = tuple(sorted(canonical["extra_predicates"]))

    if not canonical["exposure_ok"] or not canonical["outcome_ok"]:
        return "none", matched, missing, contradictory + extra
    if canonical["direction"] not in {0, truth.direction} or contradictory or extra:
        return "none", matched, missing, contradictory + extra
    if len(matched) == len(truth.subgroup_predicates) and canonical["direction"] == truth.direction:
        return "exact", matched, missing, contradictory
    if (
        len(matched) == len(truth.subgroup_predicates) - 1
        and len(missing) == 1
        and canonical["direction"] == truth.direction
    ):
        return "near", matched, missing, contradictory
    if matched:
        return "component", matched, missing, contradictory
    return "none", matched, missing, contradictory


def _declared_support(claim: Mapping[str, Any]) -> bool | None:
    for key in ("evidence_supported", "supported", "is_supported"):
        if key in claim and isinstance(claim[key], bool):
            return claim[key]
    status = _normal_name(claim.get("status", ""))
    if status in {"supported", "accepted", "confirmed", "validated", "selected", "final"}:
        return True
    if status in {"unsupported", "rejected", "refuted", "null", "uncertain", "dropped"}:
        return False
    return None


def _numeric_from_keys(record: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in record:
            number = _finite_number(record[key])
            if number is not None:
                return number
    return None


def _p_value_from_text(text: str) -> float | None:
    match = re.search(r"\bp\s*(?:value\s*)?(?:=|<|<=)\s*(0?\.\d+|1(?:\.0+)?)", text.lower())
    return float(match.group(1)) if match else None


def _subgroup_n_from_text(text: str) -> int | None:
    match = re.search(
        r"\b(?:subgroup[_\s-]*n|n)\s*(?:=|:)\s*([1-9]\d*)\b",
        text.lower(),
    )
    return int(match.group(1)) if match else None


def _named_n_from_text(text: str, label: str) -> int | None:
    token = re.escape(label).replace(r"\_", r"[_\s-]*")
    match = re.search(rf"\b{token}\s*(?:=|:)\s*([1-9]\d*)\b", text.lower())
    return int(match.group(1)) if match else None


def _positive_effect_from_text(text: str) -> bool:
    lowered = text.lower()
    if _text_direction(lowered) < 0:
        return False
    patterns = (
        r"(?:effect(?:_estimate)?|estimate|difference|diff|mean difference|treatment effect)"
        r"\s*(?:=|:|of)?\s*\+?\s*(\d+(?:\.\d+)?)",
        r"\+\s*(\d+(?:\.\d+)?)\s*(?:months?|mo)\b",
        r"(\d+(?:\.\d+)?)\s*(?:months?|mo)\s+(?:longer|benefit)",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _ci_bounds(record: Mapping[str, Any]) -> tuple[float, float] | None:
    for key in ("confidence_interval", "ci", "ci_95", "95_ci"):
        raw = record.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
            lower, upper = _finite_number(raw[0]), _finite_number(raw[1])
            if lower is not None and upper is not None:
                return lower, upper
        if isinstance(raw, Mapping):
            lower = _numeric_from_keys(raw, ("lower", "low", "lower_bound"))
            upper = _numeric_from_keys(raw, ("upper", "high", "upper_bound"))
            if lower is not None and upper is not None:
                return lower, upper
    lower = _numeric_from_keys(record, ("ci_lower", "lower_ci", "confidence_lower"))
    upper = _numeric_from_keys(record, ("ci_upper", "upper_ci", "confidence_upper"))
    if lower is not None and upper is not None:
        return lower, upper
    return None


def _analysis_supports(
    analysis: Mapping[str, Any], *, alpha: float, expected_direction: int
) -> bool:
    """Apply the locked direction, uncertainty, N, and evidence gate."""

    effect = _numeric_from_keys(
        analysis,
        (
            "effect_estimate",
            "estimate",
            "mean_difference",
            "difference",
            "treatment_effect",
            "effect_months",
        ),
    )
    text = " ".join(
        _string(analysis.get(key, ""))
        for key in ("result", "result_summary", "summary", "evidence", "statistic")
    )
    positive = effect is not None and effect * expected_direction > 0
    if effect is None:
        text_direction = _text_direction(text)
        positive = text_direction == expected_direction or (
            expected_direction > 0 and _positive_effect_from_text(text)
        )
    if not positive:
        return False

    p_value = _numeric_from_keys(analysis, ("p_value", "pvalue", "p"))
    if p_value is None:
        p_value = _p_value_from_text(text)
    ci = _ci_bounds(analysis)
    uncertainty_supports = bool(
        (p_value is not None and p_value < alpha)
        or (
            ci is not None
            and ((expected_direction > 0 and ci[0] > 0) or (expected_direction < 0 and ci[1] < 0))
        )
    )
    if not uncertainty_supports:
        return False

    subgroup_n = _numeric_from_keys(analysis, ("subgroup_n", "n", "sample_size"))
    if subgroup_n is None:
        subgroup_n = _subgroup_n_from_text(text)
    if subgroup_n is None or subgroup_n <= 0:
        return False
    exposed_n = _numeric_from_keys(
        analysis, ("exposed_n", "treated_n", "treatment_n", "n_exposed")
    )
    if exposed_n is None:
        exposed_n = _named_n_from_text(text, "exposed_n")
    comparator_n = _numeric_from_keys(
        analysis, ("comparator_n", "control_n", "unexposed_n", "n_comparator")
    )
    if comparator_n is None:
        comparator_n = _named_n_from_text(text, "comparator_n")
    if exposed_n is None or exposed_n <= 0 or comparator_n is None or comparator_n <= 0:
        return False

    evidence_keys = (
        "evidence",
        "evidence_ids",
        "evidence_id",
        "source_ids",
        "source_id",
        "artifact",
        "artifact_path",
        "file",
        "file_path",
    )
    return any(analysis.get(key) not in (None, "", [], {}) for key in evidence_keys)


def _analysis_id(record: Mapping[str, Any]) -> str:
    return str(record.get("analysis_id") or record.get("id") or "")


def _analysis_claim_ids(record: Mapping[str, Any]) -> set[str]:
    raw = record.get("claim_ids", record.get("hypothesis_ids", []))
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, Sequence):
        return {str(item) for item in raw}
    claim_id = record.get("claim_id", record.get("hypothesis_id"))
    return {str(claim_id)} if claim_id not in (None, "") else set()


def _evidence_references(claim: Mapping[str, Any]) -> bool:
    for key in ("analysis_ids", "evidence_ids", "evidence", "supporting_evidence"):
        value = claim.get(key)
        if value not in (None, "", [], {}):
            return True
    return False


def _candidate_analyses(
    claim: Mapping[str, Any],
    analyses: Sequence[Mapping[str, Any]],
    *,
    only_claim: bool,
) -> list[Mapping[str, Any]]:
    claim_id = str(claim.get("claim_id") or claim.get("id") or claim.get("hypothesis_id") or "")
    raw_ids = claim.get("analysis_ids", [])
    analysis_ids = {str(raw_ids)} if isinstance(raw_ids, str) else {str(item) for item in raw_ids}
    linked = [
        analysis
        for analysis in analyses
        if (_analysis_id(analysis) and _analysis_id(analysis) in analysis_ids)
        or (claim_id and claim_id in _analysis_claim_ids(analysis))
    ]
    if linked:
        return linked
    # Never attach an unlinked analysis merely because only one claim exists.
    # That shortcut makes an orphaned or contradictory result look supported.
    return []


def _score_structured_claim(
    claim: Mapping[str, Any],
    analyses: Sequence[Mapping[str, Any]],
    *,
    only_claim: bool,
    alpha: float,
    index: int,
    truth: TruthSpec,
) -> ClaimScore:
    text = _string(
        claim.get("text")
        or claim.get("statement")
        or claim.get("hypothesis")
        or claim.get("summary")
        or ""
    )
    canonical = _canonical_claim(claim, text, truth)
    level, matched, missing, contradictory = _recovery_level(canonical, truth)
    declared = _declared_support(claim)
    candidates = _candidate_analyses(claim, analyses, only_claim=only_claim)
    quantitative_support = _analysis_supports(
        claim, alpha=alpha, expected_direction=truth.direction
    ) or any(
        _analysis_supports(
            analysis, alpha=alpha, expected_direction=truth.direction
        )
        for analysis in candidates
    )
    # The current strict artifact contract always emits ``supported``; require
    # that declaration plus independently checkable statistics.
    supported = bool(level != "none" and declared is True and quantitative_support)
    claim_id = str(
        claim.get("claim_id") or claim.get("id") or claim.get("hypothesis_id") or f"claim-{index}"
    )
    return ClaimScore(
        claim_id=claim_id,
        text=text,
        recovery_level=level,
        supported=supported,
        declared_supported=declared,
        matched_predicates=matched,
        missing_predicates=missing,
        contradictory_predicates=contradictory,
        source="structured",
    )


def _artifact_text_candidates(artifact: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    hypotheses = artifact.get("hypotheses", [])
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    if isinstance(hypotheses, Sequence):
        for item in hypotheses:
            if isinstance(item, Mapping):
                value = item.get("text") or item.get("statement") or item.get("hypothesis")
                if value:
                    candidates.append(_string(value))
            elif item:
                candidates.append(_string(item))
    for key in ("final_answer", "summary", "handoff"):
        value = artifact.get(key)
        if value not in (None, "", [], {}):
            candidates.append(_string(value))
    # Preserve order while removing exact duplicates.
    return list(dict.fromkeys(candidates))


def _score_text_claim(
    text: str,
    artifact: Mapping[str, Any],
    analyses: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    index: int,
    truth: TruthSpec,
) -> ClaimScore:
    pseudo_claim = {"text": text}
    canonical = _canonical_claim(pseudo_claim, text, truth)
    level, matched, missing, contradictory = _recovery_level(canonical, truth)
    evidence_text = " ".join(
        [
            text,
            _string(artifact.get("evidence", "")),
            _string(artifact.get("analyses", "")),
        ]
    )
    text_support = _analysis_supports(
        {
            "result_summary": evidence_text,
            "evidence": artifact.get("evidence", []),
        },
        alpha=alpha,
        expected_direction=truth.direction,
    )
    quantitative_support = text_support or any(
        _analysis_supports(
            analysis, alpha=alpha, expected_direction=truth.direction
        )
        for analysis in analyses
    )
    support_word = bool(
        re.search(
            r"\b(?:supported|confirmed|statistically significant|validated)\b",
            evidence_text.lower(),
        )
    )
    supported = bool(level != "none" and quantitative_support and support_word)
    return ClaimScore(
        claim_id=f"text-{index}",
        text=text,
        recovery_level=level,
        supported=supported,
        declared_supported=True if support_word else None,
        matched_predicates=matched,
        missing_predicates=missing,
        contradictory_predicates=contradictory,
        source="text",
    )


def score_artifact(
    artifact: Mapping[str, Any],
    *,
    prior_analyses: Sequence[Mapping[str, Any]] = (),
    alpha: float = 0.05,
    truth: TruthSpec = DEFAULT_TRUTH,
) -> dict[str, Any]:
    """Score one normalized harness artifact.

    Prior analyses are considered only when linked by IDs to a structured
    claim.  Legacy unlinked analyses are restricted to the current artifact.
    """

    current_analyses_raw = artifact.get("analyses", [])
    if isinstance(current_analyses_raw, Mapping):
        current_analyses = [current_analyses_raw]
    elif isinstance(current_analyses_raw, Sequence) and not isinstance(
        current_analyses_raw, (str, bytes)
    ):
        current_analyses = [item for item in current_analyses_raw if isinstance(item, Mapping)]
    else:
        current_analyses = []
    analyses = [*prior_analyses, *current_analyses]

    raw_claims = artifact.get("claims", [])
    if isinstance(raw_claims, Mapping):
        raw_claims = [raw_claims]
    structured_claims = (
        [item for item in raw_claims if isinstance(item, Mapping)]
        if isinstance(raw_claims, Sequence) and not isinstance(raw_claims, (str, bytes))
        else []
    )
    if structured_claims:
        claim_scores = [
            _score_structured_claim(
                claim,
                analyses,
                only_claim=len(structured_claims) == 1,
                alpha=alpha,
                index=index,
                truth=truth,
            )
            for index, claim in enumerate(structured_claims, start=1)
        ]
    else:
        claim_scores = [
            _score_text_claim(
                text,
                artifact,
                current_analyses,
                alpha=alpha,
                index=index,
                truth=truth,
            )
            for index, text in enumerate(_artifact_text_candidates(artifact), start=1)
        ]

    best = max(claim_scores, key=lambda item: LEVEL_RANK[item.recovery_level], default=None)
    supported_claims = [item for item in claim_scores if item.supported]
    best_supported = max(
        supported_claims,
        key=lambda item: LEVEL_RANK[item.recovery_level],
        default=None,
    )
    return {
        "recovery_level": best.recovery_level if best else "none",
        "supported_recovery_level": best_supported.recovery_level if best_supported else "none",
        "supported": best_supported is not None,
        "target_supported": bool(
            best_supported and LEVEL_RANK[best_supported.recovery_level] >= LEVEL_RANK["near"]
        ),
        "best_claim_id": best.claim_id if best else None,
        "best_supported_claim_id": best_supported.claim_id if best_supported else None,
        "claims": [item.as_dict() for item in claim_scores],
        "analysis_count": len(current_analyses),
    }


def _call_index(record: Mapping[str, Any], fallback: int) -> int:
    number = _finite_number(record.get("call_index"))
    if number is not None:
        return int(number)
    match = re.search(r":c(\d+)$", str(record.get("request_id", "")))
    return int(match.group(1)) if match else fallback


def _canonical_stage(stage_id: str) -> str:
    if stage_id.endswith("_consensus"):
        return stage_id[: -len("_consensus")]
    return stage_id


def _is_deliberative(run: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]) -> bool:
    mode = _normal_name(run.get("workflow_mode", ""))
    workflow_id = _normal_name(run.get("workflow_id", ""))
    return (
        mode == "deliberative"
        or "deliberative" in workflow_id
        or any(str(item.get("stage_id", "")).endswith("_consensus") for item in artifacts)
    )


def _is_checkpoint(stage_id: str, *, deliberative: bool) -> bool:
    if stage_id in {"federated_synthesis", "independent_verification"}:
        return True
    if deliberative:
        return stage_id.endswith("_consensus") and _canonical_stage(stage_id) in BASE_STAGES
    return stage_id in BASE_STAGES


def _usage(record: Mapping[str, Any]) -> dict[str, float]:
    raw = record.get("usage", {})
    if not isinstance(raw, Mapping):
        raw = {}
    return {
        "input_tokens": _finite_number(raw.get("input_tokens")) or 0.0,
        "output_tokens": _finite_number(raw.get("output_tokens")) or 0.0,
        "tool_calls": _finite_number(raw.get("tool_calls")) or 0.0,
        "cost_usd": _finite_number(raw.get("cost_usd")) or 0.0,
        "duration_seconds": _finite_number(raw.get("duration_seconds")) or 0.0,
    }


def _best_stage(
    scores: Sequence[Mapping[str, Any]], canonical_stage: str
) -> Mapping[str, Any] | None:
    candidates = [item for item in scores if item["canonical_stage"] == canonical_stage]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            LEVEL_RANK[item["score"]["supported_recovery_level"]],
            LEVEL_RANK[item["score"]["recovery_level"]],
            item["call_index"],
        ),
    )


def _final_stage(scores: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    central = [item for item in scores if item["stage_id"] == "federated_synthesis"]
    if central:
        return max(central, key=lambda item: item["call_index"])
    synthesis = [item for item in scores if item["canonical_stage"] == "synthesis"]
    return max(synthesis, key=lambda item: item["call_index"]) if synthesis else None


def _level_or_none(stage: Mapping[str, Any] | None, key: str) -> RecoveryLevel:
    return stage["score"][key] if stage else "none"


def _target_supported(stage: Mapping[str, Any] | None) -> bool:
    return bool(stage and stage["score"]["target_supported"])


def score_run(
    run: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 0.05,
    source_run_dir: Path | None = None,
    truth: TruthSpec = DEFAULT_TRUTH,
) -> dict[str, Any]:
    """Score iteration-aware checkpoints without substituting a historical best for terminal."""

    ordered = sorted(
        enumerate(artifacts, start=1),
        key=lambda pair: _call_index(pair[1], pair[0]),
    )
    prior_analyses: list[Mapping[str, Any]] = []
    scored_all: list[dict[str, Any]] = []
    for fallback, record in ordered:
        raw_artifact = record.get("artifact", {})
        artifact = raw_artifact if isinstance(raw_artifact, Mapping) else {}
        artifact_score = score_artifact(
            artifact,
            prior_analyses=prior_analyses,
            alpha=alpha,
            truth=truth,
        )
        stage_id = str(record.get("stage_id", "unknown"))
        canonical_stage = str(record.get("canonical_stage") or _canonical_stage(stage_id))
        iteration_index = int(_finite_number(record.get("iteration_index")) or 1)
        malformed = not isinstance(raw_artifact, Mapping) or not all(
            isinstance(artifact.get(key), str) and bool(str(artifact.get(key)).strip())
            for key in ("summary", "handoff")
        )
        if canonical_stage == "synthesis":
            malformed = malformed or artifact.get("final_answer") is None
        elif artifact.get("final_answer") is not None:
            malformed = True
        scored_all.append(
            {
                "call_index": _call_index(record, fallback),
                "request_id": record.get("request_id"),
                "call_slot": record.get("call_slot"),
                "stage_id": stage_id,
                "canonical_stage": canonical_stage,
                "iteration_index": iteration_index,
                "max_iterations": int(
                    _finite_number(record.get("max_iterations"))
                    or _finite_number(run.get("iteration_policy", {}).get("iterations"))
                    or 1
                ),
                "stage_index": int(_finite_number(record.get("stage_index")) or 0),
                "stage_position": int(_finite_number(record.get("stage_position")) or 1),
                "terminal": bool(record.get("terminal", False)),
                "agent_id": record.get("agent_id"),
                "site_id": record.get("site_id"),
                "round": record.get("round"),
                "position_kind": record.get("position_kind"),
                "malformed": malformed or record.get("artifact_valid") is False,
                "usage": _usage(record),
                "score": artifact_score,
            }
        )
        raw_analyses = artifact.get("analyses", [])
        if isinstance(raw_analyses, Mapping):
            prior_analyses.append(raw_analyses)
        elif isinstance(raw_analyses, Sequence) and not isinstance(raw_analyses, (str, bytes)):
            prior_analyses.extend(item for item in raw_analyses if isinstance(item, Mapping))

    deliberative = _is_deliberative(run, artifacts)
    checkpoints = [
        item for item in scored_all if _is_checkpoint(item["stage_id"], deliberative=deliberative)
    ]
    for index, checkpoint in enumerate(checkpoints, start=1):
        checkpoint["checkpoint_index"] = index

    def last_stage(stage: str) -> Mapping[str, Any] | None:
        candidates = [item for item in checkpoints if item["canonical_stage"] == stage]
        return max(
            candidates,
            key=lambda item: (item["iteration_index"], item["call_index"]),
            default=None,
        )

    analysis = last_stage("analysis")
    critique = last_stage("critique")
    final = _final_stage(checkpoints)
    analysis_supported = _target_supported(analysis)
    critique_supported = _target_supported(critique)
    final_supported = _target_supported(final)
    final_supported_exact = bool(final and final["score"]["supported_recovery_level"] == "exact")

    first_recovery = next(
        (item for item in checkpoints if item["score"]["target_supported"]),
        None,
    )
    first_exact = next(
        (
            item
            for item in checkpoints
            if item["score"]["supported_recovery_level"] == "exact"
        ),
        None,
    )
    syntheses = [item for item in checkpoints if item["canonical_stage"] == "synthesis"]
    ever_exact_synthesis = next(
        (
            item
            for item in syntheses
            if item["score"]["supported_recovery_level"] == "exact"
        ),
        None,
    )
    ever_near_or_exact_synthesis = any(
        LEVEL_RANK[item["score"]["supported_recovery_level"]] >= LEVEL_RANK["near"]
        for item in syntheses
    )
    ever_component_synthesis = any(
        LEVEL_RANK[item["score"]["supported_recovery_level"]] >= LEVEL_RANK["component"]
        for item in syntheses
    )
    max_iterations = int(
        _finite_number(run.get("iteration_policy", {}).get("iterations"))
        or _finite_number(run.get("terminal_iteration"))
        or max((item["iteration_index"] for item in checkpoints), default=1)
    )
    recovery_cap = len(checkpoints)
    recovery_event = first_recovery is not None
    recovery_time = first_recovery["checkpoint_index"] if first_recovery else recovery_cap

    usage_keys = (
        "input_tokens",
        "output_tokens",
        "tool_calls",
        "cost_usd",
        "duration_seconds",
    )
    total_usage = {key: 0.0 for key in usage_keys}
    for item in scored_all:
        for key in total_usage:
            total_usage[key] += item["usage"][key]
    failure_duration = sum(
        _finite_number(item.get("duration_seconds")) or 0.0
        for item in run.get("call_failures", [])
        if isinstance(item, Mapping)
    )
    total_usage["duration_seconds"] += failure_duration
    total_tokens = total_usage["input_tokens"] + total_usage["output_tokens"]
    successful_calls = int(_finite_number(run.get("agent_calls")) or len(scored_all))
    calls = int(_finite_number(run.get("call_attempts")) or successful_calls)
    final_exact_recovered = int(final_supported_exact)
    final_near_or_exact_recovered = int(final_supported)
    tokens_to_first = None
    calls_to_first = None
    if first_recovery:
        calls_to_first = first_recovery["call_index"]
        tokens_to_first = sum(
            item["usage"]["input_tokens"] + item["usage"]["output_tokens"]
            for item in scored_all
            if item["call_index"] <= first_recovery["call_index"]
        )

    survival_eligible = analysis_supported
    critique_rescue_eligible = not analysis_supported and critique is not None
    post_critique_rescue = bool(not analysis_supported and final_supported and critique is not None)
    workflow_id = str(run.get("workflow_id", "unknown"))

    critique_rescues = 0
    synthesis_rescues = 0
    for iteration in sorted({item["iteration_index"] for item in checkpoints}):
        current = [item for item in checkpoints if item["iteration_index"] == iteration]
        current_analysis = next(
            (item for item in current if item["canonical_stage"] == "analysis"), None
        )
        current_critique = next(
            (item for item in current if item["canonical_stage"] == "critique"), None
        )
        current_synthesis = next(
            (item for item in current if item["canonical_stage"] == "synthesis"), None
        )
        if not _target_supported(current_analysis) and _target_supported(current_critique):
            critique_rescues += 1
        if not _target_supported(current_analysis) and _target_supported(current_synthesis):
            synthesis_rescues += 1

    later_loss = False
    if ever_exact_synthesis is not None:
        later_loss = any(
            item["iteration_index"] > ever_exact_synthesis["iteration_index"]
            and item["score"]["supported_recovery_level"] != "exact"
            for item in syntheses
        )
    unsupported_convergence = any(
        item["score"]["recovery_level"] in {"near", "exact"}
        and item["score"]["supported_recovery_level"] == "none"
        for item in syntheses
    )
    malformed_count = sum(bool(item["malformed"]) for item in scored_all)
    timed_out = (
        str(run.get("error_type", "")) in {"TimeoutError", "TimeoutExpired"}
        or "timeout" in str(run.get("stop_reason", "")).lower()
        or int(_finite_number(run.get("timeout_count")) or 0) > 0
    )
    duration_hours = total_usage["duration_seconds"] / 3600.0
    primary_recovered = ever_exact_synthesis is not None
    terminal_exact = final_supported_exact

    return {
        "scorer_version": SCORER_VERSION,
        "association_id": truth.association_id,
        "truth": truth_record(truth),
        "run_id": run.get("run_id") or (source_run_dir.name if source_run_dir else None),
        "source_run_dir": str(source_run_dir.resolve()) if source_run_dir else None,
        "status": run.get("status", "unknown"),
        "task_id": run.get("task_id"),
        "semantic_condition": run.get("semantic_condition", truth.semantic_condition),
        "workflow_id": workflow_id,
        "workflow_mode": run.get("workflow_mode"),
        "model_profile": run.get("model_profile"),
        "model_id": run.get("model_id"),
        "replicate": run.get("replicate"),
        "deliberative_consensus_only": deliberative,
        "artifact_count": len(scored_all),
        "checkpoint_count": len(checkpoints),
        "checkpoint_scores": checkpoints,
        "iterations_planned": max_iterations,
        "iterations_completed": int(
            _finite_number(run.get("iterations_completed"))
            or max((item["iteration_index"] for item in syntheses), default=0)
        ),
        "terminal_iteration": (
            int(_finite_number(final.get("iteration_index")) or 0) if final else None
        ),
        "stop_reason": run.get("stop_reason"),
        "analysis_recovery_level": _level_or_none(analysis, "recovery_level"),
        "analysis_supported_recovery_level": _level_or_none(analysis, "supported_recovery_level"),
        "critique_recovery_level": _level_or_none(critique, "recovery_level"),
        "critique_supported_recovery_level": _level_or_none(critique, "supported_recovery_level"),
        "final_recovery_level": _level_or_none(final, "recovery_level"),
        "final_supported_recovery_level": _level_or_none(final, "supported_recovery_level"),
        "final_supported_exact": final_supported_exact,
        "final_supported_near_or_exact": final_supported,
        "ever_supported_exact_synthesis": primary_recovered,
        "ever_supported_near_or_exact_synthesis": ever_near_or_exact_synthesis,
        "ever_supported_component_synthesis": ever_component_synthesis,
        "terminal_supported_exact": terminal_exact,
        "first_supported_exact_iteration": (
            first_exact["iteration_index"] if first_exact else None
        ),
        "first_supported_exact_call": first_exact["call_index"] if first_exact else None,
        "first_supported_exact_censored_iteration": (
            first_exact["iteration_index"] if first_exact else max_iterations
        ),
        "first_supported_exact_synthesis_iteration": (
            ever_exact_synthesis["iteration_index"] if ever_exact_synthesis else None
        ),
        "first_supported_exact_synthesis_call": (
            ever_exact_synthesis["call_index"] if ever_exact_synthesis else None
        ),
        "later_exact_loss": later_loss,
        "exact_persisted_to_terminal": terminal_exact if primary_recovered else None,
        "unsupported_convergence": unsupported_convergence,
        "malformed_output_count": malformed_count,
        "timed_out": timed_out,
        "analysis_to_final_survival_eligible": survival_eligible,
        "analysis_to_final_survival": final_supported if survival_eligible else None,
        "discovery_loss": (not final_supported) if survival_eligible else None,
        "critique_rescue_eligible": critique_rescue_eligible,
        "critique_rescue": critique_supported if critique_rescue_eligible else None,
        "post_critique_rescue": post_critique_rescue if critique_rescue_eligible else None,
        "critique_rescue_count": critique_rescues,
        "synthesis_rescue_count": synthesis_rescues,
        "time_to_recovery": {
            "event": recovery_event,
            "checkpoint": recovery_time,
            "cap": recovery_cap,
            "censored": not recovery_event,
            "stage_id": first_recovery["stage_id"] if first_recovery else None,
            "call_index": first_recovery["call_index"] if first_recovery else None,
        },
        "usage": {
            **{key: round(value, 6) for key, value in total_usage.items()},
            "total_tokens": int(total_tokens),
            "agent_calls": calls,
            "successful_agent_calls": successful_calls,
            "call_attempts": calls,
            "tokens_to_first_recovery": (
                int(tokens_to_first) if tokens_to_first is not None else None
            ),
            "agent_calls_to_first_recovery": calls_to_first,
            "supported_final_exact_per_1k_tokens": (
                1000.0 * final_exact_recovered / total_tokens if total_tokens else None
            ),
            "supported_final_near_or_exact_per_1k_tokens": (
                1000.0 * final_near_or_exact_recovered / total_tokens if total_tokens else None
            ),
            "supported_final_exact_per_agent_call": (
                final_exact_recovered / calls if calls else None
            ),
            "supported_final_near_or_exact_per_agent_call": (
                final_near_or_exact_recovered / calls if calls else None
            ),
            "supported_ever_exact_per_agent_call": (
                int(primary_recovered) / calls if calls else None
            ),
            "supported_ever_exact_per_1k_output_tokens": (
                1000.0 * int(primary_recovered) / total_usage["output_tokens"]
                if total_usage["output_tokens"]
                else None
            ),
            "supported_ever_exact_per_wall_clock_hour": (
                int(primary_recovered) / duration_hours if duration_hours else None
            ),
        },
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_run_dirs(paths: Sequence[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path.is_file() and path.name == "run.json":
            found.add(path.parent)
        elif (path / "run.json").is_file():
            found.add(path)
        elif (path / "runs").is_dir():
            found.update(item.parent for item in (path / "runs").glob("*/run.json"))
        elif path.is_dir():
            found.update(item.parent for item in path.rglob("run.json"))
    return sorted(found)


def _load_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    aggregate = run_dir / "artifacts.json"
    if aggregate.is_file():
        payload = _read_json(aggregate)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    # Failed/partial runs can still be scored from controller-normalized calls.
    records: list[dict[str, Any]] = []
    for path in sorted((run_dir / "calls").glob("call_*/normalized_response.json")):
        payload = _read_json(path)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    if not total:
        return None, None
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _aggregate_group(
    group: Sequence[Mapping[str, Any]], workflow_id: str, semantic_condition: str = "unspecified"
) -> dict[str, Any]:
    final_levels = {level: 0 for level in LEVEL_RANK}
    supported_levels = {level: 0 for level in LEVEL_RANK}
    for run in group:
        final_levels[run["final_recovery_level"]] += 1
        supported_levels[run["final_supported_recovery_level"]] += 1

    final_exact = sum(bool(run["final_supported_exact"]) for run in group)
    final_near_or_exact = sum(bool(run["final_supported_near_or_exact"]) for run in group)
    survival_runs = [run for run in group if run["analysis_to_final_survival_eligible"]]
    survival_n = sum(bool(run["analysis_to_final_survival"]) for run in survival_runs)
    loss_n = sum(bool(run["discovery_loss"]) for run in survival_runs)
    rescue_runs = [run for run in group if run["critique_rescue_eligible"]]
    rescue_n = sum(bool(run["critique_rescue"]) for run in rescue_runs)
    post_rescue_n = sum(bool(run["post_critique_rescue"]) for run in rescue_runs)
    recovery_runs = [run for run in group if run["time_to_recovery"]["event"]]
    total_tokens = sum(run["usage"]["total_tokens"] for run in group)
    total_calls = sum(run["usage"]["agent_calls"] for run in group)
    capped_times = [float(run["time_to_recovery"]["checkpoint"]) for run in group]
    event_times = [float(run["time_to_recovery"]["checkpoint"]) for run in recovery_runs]
    ever_exact = sum(bool(run.get("ever_supported_exact_synthesis")) for run in group)
    ever_near = sum(bool(run.get("ever_supported_near_or_exact_synthesis")) for run in group)
    ever_component = sum(bool(run.get("ever_supported_component_synthesis")) for run in group)
    primary_low, primary_high = _wilson_interval(ever_exact, len(group))
    terminal_low, terminal_high = _wilson_interval(final_exact, len(group))
    total_output_tokens = sum(run["usage"]["output_tokens"] for run in group)
    total_tool_calls = sum(run["usage"]["tool_calls"] for run in group)
    total_duration_seconds = sum(run["usage"]["duration_seconds"] for run in group)

    return {
        "semantic_condition": semantic_condition,
        "workflow_id": workflow_id,
        "n_runs": len(group),
        "final_recovery_counts": final_levels,
        "final_supported_recovery_counts": supported_levels,
        "final_supported_exact_n": final_exact,
        "final_supported_exact_rate": _rate(final_exact, len(group)),
        "primary_ever_supported_exact_n": ever_exact,
        "primary_ever_supported_exact_rate": _rate(ever_exact, len(group)),
        "primary_ever_supported_exact_wilson_low": primary_low,
        "primary_ever_supported_exact_wilson_high": primary_high,
        "terminal_supported_exact_wilson_low": terminal_low,
        "terminal_supported_exact_wilson_high": terminal_high,
        "ever_supported_near_or_exact_n": ever_near,
        "ever_supported_component_n": ever_component,
        "final_supported_near_or_exact_n": final_near_or_exact,
        "final_supported_near_or_exact_rate": _rate(final_near_or_exact, len(group)),
        "analysis_to_final_survival_numerator": survival_n,
        "analysis_to_final_survival_denominator": len(survival_runs),
        "analysis_to_final_survival_rate": _rate(survival_n, len(survival_runs)),
        "discovery_loss_numerator": loss_n,
        "discovery_loss_denominator": len(survival_runs),
        "discovery_loss_rate": _rate(loss_n, len(survival_runs)),
        "critique_rescue_numerator": rescue_n,
        "critique_rescue_denominator": len(rescue_runs),
        "critique_rescue_rate": _rate(rescue_n, len(rescue_runs)),
        "post_critique_rescue_numerator": post_rescue_n,
        "post_critique_rescue_denominator": len(rescue_runs),
        "post_critique_rescue_rate": _rate(post_rescue_n, len(rescue_runs)),
        "time_to_recovery_events": len(recovery_runs),
        "time_to_recovery_censored": len(group) - len(recovery_runs),
        "time_to_recovery_event_rate": _rate(len(recovery_runs), len(group)),
        "median_recovery_checkpoint_among_events": _median(event_times),
        "restricted_mean_recovery_checkpoint": _mean(capped_times),
        "total_tokens": total_tokens,
        "total_output_tokens": total_output_tokens,
        "total_agent_calls": total_calls,
        "total_tool_calls": total_tool_calls,
        "total_duration_seconds": total_duration_seconds,
        "timeout_n": sum(bool(run.get("timed_out")) for run in group),
        "malformed_output_n": sum(
            int(run.get("malformed_output_count", 0)) for run in group
        ),
        "unsupported_convergence_n": sum(
            bool(run.get("unsupported_convergence")) for run in group
        ),
        "later_exact_loss_n": sum(bool(run.get("later_exact_loss")) for run in group),
        "supported_final_exact_per_1k_tokens": (
            1000.0 * final_exact / total_tokens if total_tokens else None
        ),
        "supported_final_near_or_exact_per_1k_tokens": (
            1000.0 * final_near_or_exact / total_tokens if total_tokens else None
        ),
        "supported_final_exact_per_100_agent_calls": (
            100.0 * final_exact / total_calls if total_calls else None
        ),
        "supported_final_near_or_exact_per_100_agent_calls": (
            100.0 * final_near_or_exact / total_calls if total_calls else None
        ),
        "tokens_per_supported_final_exact": (total_tokens / final_exact if final_exact else None),
        "agent_calls_per_supported_final_exact": (
            total_calls / final_exact if final_exact else None
        ),
        "supported_ever_exact_per_agent_call": (
            ever_exact / total_calls if total_calls else None
        ),
        "supported_ever_exact_per_1k_output_tokens": (
            1000.0 * ever_exact / total_output_tokens if total_output_tokens else None
        ),
        "supported_ever_exact_per_wall_clock_hour": (
            3600.0 * ever_exact / total_duration_seconds if total_duration_seconds else None
        ),
    }


def aggregate_scores(run_scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for score in run_scores:
        key = (
            str(score.get("semantic_condition", "unspecified")),
            str(score.get("workflow_id", "unknown")),
        )
        groups[key].append(score)
    by_cell = [
        _aggregate_group(groups[key], key[1], key[0]) for key in sorted(groups)
    ]
    truth_by_task = {
        str(score.get("task_id")): score.get("truth")
        for score in run_scores
        if score.get("task_id") is not None
    }
    paired: list[dict[str, Any]] = []
    conditions = sorted({str(item.get("semantic_condition")) for item in run_scores})
    workflows = sorted({str(item.get("workflow_id")) for item in run_scores})
    by_key = {
        (
            str(item.get("semantic_condition")),
            str(item.get("workflow_id")),
            int(_finite_number(item.get("replicate")) or 0),
        ): item
        for item in run_scores
    }
    for condition in conditions:
        for left_index, left in enumerate(workflows):
            for right in workflows[left_index + 1 :]:
                pairs = [
                    (by_key[(condition, left, replicate)], by_key[(condition, right, replicate)])
                    for replicate in sorted(
                        {
                            key[2]
                            for key in by_key
                            if key[0] == condition
                            and (condition, left, key[2]) in by_key
                            and (condition, right, key[2]) in by_key
                        }
                    )
                ]
                if pairs:
                    paired.append(
                        {
                            "contrast": "workflow_within_condition",
                            "semantic_condition": condition,
                            "left": left,
                            "right": right,
                            "n_pairs": len(pairs),
                            "mean_primary_difference": _mean(
                                [
                                    float(bool(a["ever_supported_exact_synthesis"]))
                                    - float(bool(b["ever_supported_exact_synthesis"]))
                                    for a, b in pairs
                                ]
                            ),
                            "mean_terminal_difference": _mean(
                                [
                                    float(bool(a["terminal_supported_exact"]))
                                    - float(bool(b["terminal_supported_exact"]))
                                    for a, b in pairs
                                ]
                            ),
                        }
                    )
    if len(conditions) == 2:
        left_condition, right_condition = conditions
        for workflow in workflows:
            pairs = [
                (
                    by_key[(left_condition, workflow, replicate)],
                    by_key[(right_condition, workflow, replicate)],
                )
                for replicate in sorted(
                    {
                        key[2]
                        for key in by_key
                        if (left_condition, workflow, key[2]) in by_key
                        and (right_condition, workflow, key[2]) in by_key
                    }
                )
            ]
            if pairs:
                paired.append(
                    {
                        "contrast": "semantic_within_workflow",
                        "workflow_id": workflow,
                        "left": left_condition,
                        "right": right_condition,
                        "n_pairs": len(pairs),
                        "mean_primary_difference": _mean(
                            [
                                float(bool(a["ever_supported_exact_synthesis"]))
                                - float(bool(b["ever_supported_exact_synthesis"]))
                                for a, b in pairs
                            ]
                        ),
                        "mean_terminal_difference": _mean(
                            [
                                float(bool(a["terminal_supported_exact"]))
                                - float(bool(b["terminal_supported_exact"]))
                                for a, b in pairs
                            ]
                        ),
                    }
                )
    return {
        "schema_version": "1",
        "scorer_version": SCORER_VERSION,
        "truth": truth_record(),
        "truth_by_task": truth_by_task,
        "n_runs": len(run_scores),
        "by_cell": by_cell,
        # Compatibility alias; cells remain condition-separated and are never pooled.
        "by_workflow": by_cell,
        "paired_descriptive_contrasts": paired,
        "runs": list(run_scores),
    }


CSV_FIELDS = (
    "workflow_id",
    "n_runs",
    "final_supported_exact_n",
    "final_supported_exact_rate",
    "final_supported_near_or_exact_n",
    "final_supported_near_or_exact_rate",
    "analysis_to_final_survival_numerator",
    "analysis_to_final_survival_denominator",
    "analysis_to_final_survival_rate",
    "discovery_loss_numerator",
    "discovery_loss_denominator",
    "discovery_loss_rate",
    "critique_rescue_numerator",
    "critique_rescue_denominator",
    "critique_rescue_rate",
    "post_critique_rescue_numerator",
    "post_critique_rescue_denominator",
    "post_critique_rescue_rate",
    "time_to_recovery_events",
    "time_to_recovery_censored",
    "median_recovery_checkpoint_among_events",
    "restricted_mean_recovery_checkpoint",
    "total_tokens",
    "total_agent_calls",
    "supported_final_exact_per_1k_tokens",
    "supported_final_near_or_exact_per_1k_tokens",
    "supported_final_exact_per_100_agent_calls",
    "supported_final_near_or_exact_per_100_agent_calls",
    "tokens_per_supported_final_exact",
    "agent_calls_per_supported_final_exact",
)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_report(aggregate: Mapping[str, Any]) -> str:
    lines = [
        "# NSCLC semantic × workflow grid score",
        "",
        "This deterministic evaluator used each condition's private manifest in its native "
        "column namespace. No model judge was used.",
        "",
        f"Runs scored: **{aggregate['n_runs']}**.",
        "",
        "| Semantic condition | Workflow | Runs | Ever exact synthesis | Terminal exact | "
        "Ever near/exact | Timeouts | Malformed | Calls | Output tokens | Exact /1k output |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["by_cell"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["semantic_condition"],
                    row["workflow_id"],
                    str(row["n_runs"]),
                    f"{row['primary_ever_supported_exact_n']}/{row['n_runs']}",
                    f"{row['final_supported_exact_n']}/{row['n_runs']}",
                    f"{row['ever_supported_near_or_exact_n']}/{row['n_runs']}",
                    str(row["timeout_n"]),
                    str(row["malformed_output_n"]),
                    str(row["total_agent_calls"]),
                    _fmt(row["total_output_tokens"]),
                    _fmt(row["supported_ever_exact_per_1k_output_tokens"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Endpoint definitions",
            "",
            "- Primary: evidence-supported exact recovery at a synthesis checkpoint on or "
            "before iteration 20.",
            "- Key secondary: evidence-supported exact recovery retained in the terminal "
            "iteration-20 synthesis.",
            "- Support requires a directionally compatible quantitative effect, uncertainty "
            "or p-value, and subgroup, exposed, and comparator sample sizes.",
            "- Deliberative chairs are checkpoints; peer artifacts are diagnostics.",
            "- The comparison is native-resource. Calls, output tokens, tool calls, and wall "
            "time are reported rather than treated as matched.",
            "- With five planned replicates per cell, all intervals and paired contrasts are "
            "descriptive; no confirmatory workflow-superiority p-values are reported.",
            "",
            "## Failures and provenance",
            "",
            "Technical failures: "
            f"**{sum(run['status'] != 'completed' for run in aggregate['runs'])}**; "
            f"timeouts: **{sum(bool(run.get('timed_out')) for run in aggregate['runs'])}**.",
            "",
            f"Scorer version: `{aggregate['scorer_version']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.items()} for row in rows
        )


def write_outputs(aggregate: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aggregate.json"
    csv_path = out_dir / "aggregate.csv"
    run_path = out_dir / "run_scores.csv"
    iteration_path = out_dir / "iteration_scores.csv"
    cell_path = out_dir / "cell_summary.csv"
    resource_path = out_dir / "resource_summary.csv"
    markdown_path = out_dir / "report.md"
    json_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    run_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    for run in aggregate["runs"]:
        run_rows.append(
            {
                key: value
                for key, value in run.items()
                if key not in {"checkpoint_scores", "truth", "usage"}
            }
            | {f"usage_{key}": value for key, value in run["usage"].items()}
        )
        for checkpoint in run["checkpoint_scores"]:
            score = checkpoint["score"]
            iteration_rows.append(
                {
                    "run_id": run["run_id"],
                    "task_id": run["task_id"],
                    "semantic_condition": run["semantic_condition"],
                    "workflow_id": run["workflow_id"],
                    "replicate": run["replicate"],
                    "iteration_index": checkpoint["iteration_index"],
                    "stage_id": checkpoint["stage_id"],
                    "canonical_stage": checkpoint["canonical_stage"],
                    "checkpoint_index": checkpoint["checkpoint_index"],
                    "call_index": checkpoint["call_index"],
                    "terminal": checkpoint["terminal"],
                    "malformed": checkpoint["malformed"],
                    "recovery_level": score["recovery_level"],
                    "supported_recovery_level": score["supported_recovery_level"],
                    "target_supported": score["target_supported"],
                    **{
                        f"usage_{key}": value
                        for key, value in checkpoint["usage"].items()
                    },
                }
            )
    cell_rows = list(aggregate["by_cell"])
    resource_rows = [
        {
            key: row.get(key)
            for key in (
                "semantic_condition",
                "workflow_id",
                "n_runs",
                "total_agent_calls",
                "total_output_tokens",
                "total_tool_calls",
                "total_duration_seconds",
                "supported_ever_exact_per_agent_call",
                "supported_ever_exact_per_1k_output_tokens",
                "supported_ever_exact_per_wall_clock_hour",
            )
        }
        for row in cell_rows
    ]
    _write_csv(run_path, run_rows)
    _write_csv(iteration_path, iteration_rows)
    _write_csv(cell_path, cell_rows)
    _write_csv(resource_path, resource_rows)
    _write_csv(csv_path, cell_rows)
    markdown_path.write_text(_markdown_report(aggregate), encoding="utf-8")
    return {
        "json": json_path,
        "csv": csv_path,
        "run_scores": run_path,
        "iteration_scores": iteration_path,
        "cell_summary": cell_path,
        "resource_summary": resource_path,
        "markdown": markdown_path,
    }


def score_experiment(
    paths: Sequence[Path],
    *,
    out_dir: Path,
    alpha: float = 0.05,
    include_smoke: bool = False,
) -> dict[str, Any]:
    run_dirs = _discover_run_dirs(paths)
    if not include_smoke:
        run_dirs = [
            path
            for path in run_dirs
            if not any("smoke" in part.lower() for part in path.parts)
        ]
    if not run_dirs:
        raise ValueError("No run.json files found in the supplied paths.")
    truth_cache: dict[tuple[str, str, str | None], TruthSpec] = {}
    run_scores = []
    for run_dir in run_dirs:
        run = _read_json(run_dir / "run.json")
        truth = DEFAULT_TRUTH
        root = run_dir.parent.parent if run_dir.parent.name == "runs" else run_dir.parent
        index_path = root / "private_evaluation_index.json"
        if index_path.is_file():
            index = _read_json(index_path)
            task_id = str(run.get("task_id", ""))
            task_entry = index.get("tasks", {}).get(task_id)
            if not isinstance(task_entry, Mapping):
                raise ValueError(f"No private evaluator entry for task {task_id!r}")
            manifest_path = Path(str(task_entry.get("path", ""))).resolve(strict=True)
            if _file_sha256(manifest_path) != task_entry.get("sha256"):
                raise ValueError(f"Private evaluator manifest hash changed: {manifest_path}")
            semantic_condition = str(
                run.get("semantic_condition") or task_entry.get("semantic_condition")
            )
            if semantic_condition != str(task_entry.get("semantic_condition")):
                raise ValueError(
                    f"Run condition does not match evaluator entry for task {task_id!r}"
                )
            mapping_entry = index.get("assets", {}).get("column_mapping")
            mapping_path: Path | None = None
            if isinstance(mapping_entry, Mapping):
                mapping_path = Path(str(mapping_entry.get("path", ""))).resolve(strict=True)
                if _file_sha256(mapping_path) != mapping_entry.get("sha256"):
                    raise ValueError(f"Evaluator column mapping hash changed: {mapping_path}")
            cache_key = (
                str(manifest_path),
                semantic_condition,
                str(mapping_path) if mapping_path else None,
            )
            truth = truth_cache.setdefault(
                cache_key,
                load_truth(
                    manifest_path,
                    semantic_condition=semantic_condition,
                    column_mapping_path=mapping_path,
                ),
            )
        run_scores.append(
            score_run(
                run,
                _load_artifacts(run_dir),
                alpha=alpha,
                source_run_dir=run_dir,
                truth=truth,
            )
        )
    aggregate = aggregate_scores(run_scores)
    aggregate["significance_threshold"] = alpha
    aggregate["source_run_dirs"] = [str(path) for path in run_dirs]
    write_outputs(aggregate, out_dir)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, nargs="+", help="Experiment roots or run directories."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Directory for aggregate JSON/CSV/MD."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Two-sided significance threshold."
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Score explicitly supplied smoke roots; excluded by default from main analyses.",
    )
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    aggregate = score_experiment(
        args.results,
        out_dir=args.out,
        alpha=args.alpha,
        include_smoke=args.include_smoke,
    )
    print(f"Scored {aggregate['n_runs']} run(s).")
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
