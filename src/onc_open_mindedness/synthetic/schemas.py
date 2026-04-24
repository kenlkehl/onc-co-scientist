"""Pydantic schemas for synthetic dataset ground truth.

The `DatasetManifest` is the canonical record of what was *actually* simulated
into a bundle; scoring reads it to decide whether a harness-proposed hypothesis
matches a real embedded association.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ParadigmClass(StrEnum):
    """Grant-defined categories for each embedded association.

    - `concordant`: the association matches currently accepted oncology
      paradigms (e.g., immune checkpoint inhibitors are less effective in
      EGFR-mutant lung cancer).
    - `discordant`: the direction of effect is inverted or unexpected relative
      to current consensus.
    - `hidden_novel`: a subgroup in which a broadly ineffective therapy is
      exceptionally active, signaling an undiscovered biomarker. Only
      recoverable by models willing to entertain heterogeneity.
    """

    concordant = "concordant"
    discordant = "discordant"
    hidden_novel = "hidden_novel"


class AssociationForm(StrEnum):
    """Structural form of an embedded association in the data-generating process."""

    main_effect = "main_effect"
    interaction = "interaction"
    subgroup_conditional = "subgroup_conditional"


class SubgroupSpec(BaseModel):
    """A Boolean subgroup defined by a column-level predicate dictionary.

    The subgroup is the set of rows for which every entry in `predicate`
    evaluates true. Keys are column names; values may be scalars (equality)
    or `{"min": x, "max": y}` range dicts.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    predicate: dict[str, object]
    description: str


class AssociationSpec(BaseModel):
    """A single ground-truth association embedded in a synthetic dataset."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable identifier within a dataset manifest.")
    paradigm_class: ParadigmClass
    form: AssociationForm
    variables: list[str] = Field(
        description="Column names participating in this association "
        "(treatment, biomarker, outcome, interacting feature, etc.)."
    )
    outcome: str = Field(description="Outcome column affected by this association.")
    direction: int = Field(
        description="Sign of the effect on the outcome (+1 increases, -1 decreases, 0 null).",
        ge=-1,
        le=1,
    )
    effect_size: float = Field(
        description="Signed effect magnitude on the outcome scale "
        "(linear coefficient for continuous outcomes, log-odds for binary)."
    )
    subgroup: SubgroupSpec | None = Field(
        default=None,
        description="For subgroup-conditional associations (hidden-novel in particular), "
        "the subgroup in which this association is active.",
    )
    natural_language_description: str = Field(
        description="A plain-English statement of the association, used by the scoring "
        "matcher to align harness-proposed hypotheses with ground truth."
    )


class DatasetManifest(BaseModel):
    """Full ground-truth manifest for one synthetic dataset bundle."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    seed: int
    patient_n: int
    columns: list[str] = Field(
        description="All column names present in dataset.parquet, in order."
    )
    treatment_columns: list[str]
    outcome_columns: list[str]
    covariate_columns: list[str]
    associations: list[AssociationSpec]
    generator_version: str = "0.2.0"
    notes: str | None = None

    def associations_by_class(self, paradigm_class: ParadigmClass) -> list[AssociationSpec]:
        return [a for a in self.associations if a.paradigm_class == paradigm_class]
