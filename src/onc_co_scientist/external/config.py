"""Versioned external-runner configuration, independent of controller workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from ..expected_surprising.schemas import StrictModel


class ExternalSpec(StrictModel):
    schema_version: Literal["external-native-1"] = "external-native-1"
    experiment_id: str = "nsclc-biomni-qwen38-xhigh"
    input_root: Path
    output_root: Path
    pair_id: str = "es-v2-nsclc_clinical-42000"
    runner: Literal["biomni"] = "biomni"
    execution_mode: Literal["external-native"] = "external-native"
    resource_policy: Literal["full"] = "full"
    biomni_root: Path = Path("/home/kenneth_kehl/biomni")
    biomni_commit: str = "400c1f366b96a35ca253e13c9b06c5076af41d65"
    python: Path = Path("/home/kenneth_kehl/biomni/.venv/bin/python")
    data_lake: Path = Path("/home/kenneth_kehl/biomni/data/biomni_data/data_lake")
    model: str = "Inferact/Qwen3.8-27B-NVFP4"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str | None = None
    reasoning_effort: Literal["xhigh"] = "xhigh"
    max_tokens: int = Field(default=125000, ge=1)
    completion_policy: Literal["fixed", "adaptive"] = "fixed"
    context_guard_tokens: int = Field(default=256, ge=0)
    min_completion_tokens: int = Field(default=1024, ge=1)
    context_length: int = Field(default=262144, ge=1)
    temperature: float = Field(default=0.7, ge=0)
    max_requests: int = Field(default=900, ge=1)
    request_timeout: int = Field(default=14400, ge=1)
    tool_timeout: int = Field(default=600, ge=1)
    rounds: int = Field(default=25, ge=1, le=25)
    replicates: int = Field(default=10, ge=1)
    schedule_seed: int = 20260908
    stage_failure_policy: Literal["zero_run", "retain_scientific_scores"] = "zero_run"

    @model_validator(mode="after")
    def check_paths(self):
        if self.min_completion_tokens > self.max_tokens:
            raise ValueError("Minimum completion allowance exceeds the ceiling")
        if self.context_length <= self.max_tokens:
            raise ValueError("Context must leave space for prompts plus the completion ceiling")
        for field in ("input_root", "output_root", "biomni_root", "python", "data_lake"):
            path = getattr(self, field).expanduser()
            # Resolving a venv's interpreter symlink bypasses its site-packages.
            value = Path(os.path.abspath(path)) if field == "python" else path.resolve()
            setattr(self, field, value)
        return self


def load_spec(path):
    data = yaml.safe_load(Path(path).read_text())
    # Relative paths are relative to the configuration, not the caller's cwd.
    for key in ("input_root", "output_root", "biomni_root", "python", "data_lake"):
        if key in data and not Path(data[key]).expanduser().is_absolute():
            data[key] = str(Path(path).resolve().parent / data[key])
    return ExternalSpec.model_validate(data)
