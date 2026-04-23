"""Match harness-proposed hypotheses to ground-truth AssociationSpec entries.

Two matchers are supported:

- ``RegexMatcher``: deterministic, offline. Uses variable-name tokens drawn from
  the AssociationSpec, plus optional subgroup predicate tokens, to check whether
  a hypothesis string mentions the right columns and a direction. Adequate for
  tests and clean MVP runs.

- ``LLMMatcher``: semantic, provider-backed. Sends the hypothesis text and the
  natural-language description of the ground-truth association to an
  LLMProvider and asks for a yes/no judgment. Robust to phrasing variation
  (e.g., "EGFR mutation" vs "mutant EGFR", "response" vs "objective response").

Both matchers expose the same ``matches(hypothesis_text, spec) -> bool`` API.
The scoring pipeline defaults to ``RegexMatcher`` so scoring is reproducible
without network access; users can plug in ``LLMMatcher`` via config.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..providers.base import ChatMessage, LLMProvider
from ..synthetic.schemas import AssociationSpec

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@runtime_checkable
class HypothesisMatcher(Protocol):
    def matches(self, hypothesis_text: str, spec: AssociationSpec) -> bool: ...


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _variable_keywords(spec: AssociationSpec) -> list[set[str]]:
    """Build a list of keyword-sets. A hypothesis must mention at least one token
    from each set for the regex matcher to accept it."""
    keyword_sets: list[set[str]] = []
    for var in spec.variables:
        tokens = {var.lower()}
        parts = var.split("_")
        tokens.update(p.lower() for p in parts if len(p) > 2)
        # Canonical aliases for common oncology variables.
        aliases = _ALIAS_MAP.get(var, ())
        tokens.update(a.lower() for a in aliases)
        keyword_sets.append(tokens)
    if spec.subgroup is not None:
        for col in spec.subgroup.predicate:
            tokens = {col.lower()}
            parts = col.split("_")
            tokens.update(p.lower() for p in parts if len(p) > 2)
            tokens.update(a.lower() for a in _ALIAS_MAP.get(col, ()))
            keyword_sets.append(tokens)
    return keyword_sets


_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "treatment_io": ("immunotherapy", "checkpoint", "pembrolizumab", "nivolumab"),
    "egfr_mutation": ("egfr", "egfrm"),
    "pdl1_tps": ("pdl1", "pd-l1"),
    "tmb_high": ("tmb", "mutational", "burden"),
    "kras_g12c": ("kras", "g12c"),
    "treatment_kras_g12c_inhibitor": ("sotorasib", "adagrasib", "g12c", "kras"),
    "treatment_x": ("therapy", "drug"),
    "biomarker_z_high": ("biomarker", "marker", "subgroup"),
    "progression_free_months": ("pfs", "progression", "survival"),
    "objective_response": ("response", "orr", "responder"),
}


@dataclass
class RegexMatcher:
    """Heuristic matcher used as the deterministic fallback."""

    require_direction: bool = False

    def matches(self, hypothesis_text: str, spec: AssociationSpec) -> bool:
        tokens = _tokens(hypothesis_text)
        for keyword_set in _variable_keywords(spec):
            if tokens.isdisjoint(keyword_set):
                return False
        if self.require_direction:
            if spec.direction > 0 and not _mentions_increase(hypothesis_text):
                return False
            if spec.direction < 0 and not _mentions_decrease(hypothesis_text):
                return False
        return True


_INCREASE_TOKENS = frozenset(
    {
        "increase",
        "increases",
        "higher",
        "greater",
        "more",
        "improve",
        "improves",
        "longer",
        "better",
        "positive",
    }
)
_DECREASE_TOKENS = frozenset(
    {
        "decrease",
        "decreases",
        "lower",
        "less",
        "fewer",
        "worse",
        "shorter",
        "reduce",
        "reduces",
        "negative",
    }
)


def _mentions_increase(text: str) -> bool:
    return not _tokens(text).isdisjoint(_INCREASE_TOKENS)


def _mentions_decrease(text: str) -> bool:
    return not _tokens(text).isdisjoint(_DECREASE_TOKENS)


@dataclass
class LLMMatcher:
    """Semantic matcher backed by an LLMProvider."""

    provider: LLMProvider
    temperature: float = 0.0

    _SYSTEM = (
        "You compare a hypothesis proposed by an analyst to a ground-truth "
        "association embedded in a synthetic dataset. Decide whether the "
        "analyst hypothesis semantically corresponds to the ground-truth "
        "association. Variable names, directions, and subgroups must all align; "
        "paraphrasing and synonym substitution are fine. Respond with a JSON "
        'object of the form {"matches": true} or {"matches": false}.'
    )

    def matches(self, hypothesis_text: str, spec: AssociationSpec) -> bool:
        user = (
            f"Analyst hypothesis:\n{hypothesis_text}\n\n"
            f"Ground-truth association:\n{spec.natural_language_description}\n\n"
            f"Variables: {', '.join(spec.variables)}\n"
            f"Direction: {spec.direction}\n"
            f"Effect size: {spec.effect_size}"
        )
        response = self.provider.chat(
            [ChatMessage(role="user", content=user)],
            temperature=self.temperature,
            max_tokens=64,
            system=self._SYSTEM,
        )
        text = response.text.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return False
        return bool(payload.get("matches"))
