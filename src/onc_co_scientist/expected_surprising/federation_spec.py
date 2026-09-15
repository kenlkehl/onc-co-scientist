"""Independent site-count and partition dimensions for the scientific grid."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class SitePartition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default="random", pattern=r"^[A-Za-z0-9_-]+$")
    mode: Literal["random", "heterogeneous"] = "random"
    by: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check(self):
        if (self.mode == "heterogeneous") != bool(self.by):
            raise ValueError(
                "Heterogeneous partitions require public covariate names in by; "
                "random partitions do not use by"
            )
        return self


class FederationCell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sites: int = Field(ge=1)
    partition: SitePartition = Field(default_factory=SitePartition)
    seed: int = 0
    context_policy: Literal["legacy", "federated_v2"] = "legacy"

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        data = handler(self)
        if self.context_policy == "legacy":
            data.pop("context_policy", None)
        return data

    @property
    def label(self):
        return f"n{self.sites}-{self.partition.id}" if self.sites > 1 else "n1"


class FederationGrid(BaseModel):
    model_config = ConfigDict(extra="forbid")
    site_counts: list[int] = Field(default_factory=lambda: [1], min_length=1)
    partitions: list[SitePartition] = Field(default_factory=lambda: [SitePartition()], min_length=1)
    seed: int = 0
    orchestrator_model_profile: str | None = None
    context_policy: Literal["legacy", "federated_v2"] = "legacy"

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        data = handler(self)
        if self.context_policy == "legacy":
            data.pop("context_policy", None)
        return data

    @model_validator(mode="after")
    def check(self):
        if any(n < 1 for n in self.site_counts) or len(set(self.site_counts)) != len(
            self.site_counts
        ):
            raise ValueError("site_counts must be distinct positive integers")
        if len({p.id for p in self.partitions}) != len(self.partitions):
            raise ValueError("Partition IDs must be unique")
        return self

    def cells(self):
        return [
            FederationCell(
                sites=n,
                partition=p,
                seed=self.seed,
                context_policy=self.context_policy if n > 1 else "legacy",
            )
            for n in self.site_counts
            for p in ([SitePartition()] if n == 1 else self.partitions)
        ]
