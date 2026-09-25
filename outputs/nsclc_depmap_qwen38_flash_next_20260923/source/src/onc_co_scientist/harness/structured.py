"""Closed, typed scientific claims shared by all research-agent backends."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Scalar = str | int | float | bool
Operator = Literal["eq", "ne", "lt", "le", "gt", "ge", "in", "not_in"]


class Predicate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    column: str = Field(min_length=1)
    operator: Operator
    value: Scalar | list[Scalar]

    @model_validator(mode="after")
    def validate_value(self) -> Predicate:
        values = self.value if isinstance(self.value, list) else [self.value]
        if not values or any(isinstance(v, float) and not math.isfinite(v) for v in values):
            raise ValueError("Predicate values must be nonempty and finite.")
        if (self.operator in {"in", "not_in"}) != isinstance(self.value, list):
            raise ValueError("Only in/not_in operators require a list.")
        if self.operator in {"lt", "le", "gt", "ge"} and (
            not isinstance(self.value, (float, int)) or isinstance(self.value, bool)
        ):
            raise ValueError("Ordered comparison requires a numeric scalar.")
        return self


def _satisfies(value: Scalar, predicate: Predicate) -> bool:
    op, bound = predicate.operator, predicate.value
    if op == "eq":
        return value == bound
    if op == "ne":
        return value != bound
    if op == "in":
        return value in bound
    if op == "not_in":
        return value not in bound
    try:
        if op == "gt":
            return value > bound
        if op == "ge":
            return value >= bound
        if op == "lt":
            return value < bound
        return value <= bound
    except TypeError:
        return False


class StructuredFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    outcome: str = Field(min_length=1)
    exposure: str | None = None
    contrast: Literal["treatment_effect", "treatment_interaction", "subgroup_difference"]
    direction: Literal[-1, 0, 1]
    subgroup: list[Predicate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_conjunction(self) -> StructuredFinding:
        if self.contrast.startswith("treatment_") and not self.exposure:
            raise ValueError("Treatment contrasts require an exposure.")
        if self.contrast == "subgroup_difference" and self.exposure is not None:
            raise ValueError("Subgroup mean differences do not have a treatment exposure.")
        if self.outcome == self.exposure or any(p.column == self.outcome for p in self.subgroup):
            raise ValueError("The outcome cannot define its own exposure or subgroup.")
        if self.exposure and any(p.column == self.exposure for p in self.subgroup):
            raise ValueError("Subgroup definitions cannot condition on the contrasted exposure.")
        groups: dict[str, list[Predicate]] = {}
        for p in self.subgroup:
            groups.setdefault(p.column, []).append(p)
        for col, ps in groups.items():
            equalities = [p.value for p in ps if p.operator == "eq"]
            allowed = [p.value for p in ps if p.operator == "in"]
            if equalities or allowed:
                candidates = equalities[:1] if equalities else allowed[0]
                if not any(all(_satisfies(value, p) for p in ps) for value in candidates):
                    raise ValueError(f"Contradictory predicates for {col}.")
            lower = [p for p in ps if p.operator in {"gt", "ge"}]
            upper = [p for p in ps if p.operator in {"lt", "le"}]
            if lower and upper:
                lo = max(p.value for p in lower)
                hi = min(p.value for p in upper)
                lo_open = any(p.value == lo and p.operator == "gt" for p in lower)
                hi_open = any(p.value == hi and p.operator == "lt" for p in upper)
                if lo > hi or (lo == hi and (lo_open or hi_open)):
                    raise ValueError(f"Contradictory bounds for {col}.")
        return self
