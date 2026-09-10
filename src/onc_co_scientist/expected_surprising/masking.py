"""Opaque public twins with private, invertible scoring and sampling translations."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from ..synthetic.anonymize import build_column_mapping, build_value_mapping, mask_frame, remap_value
from .evaluation import WorkflowValidationService
from .generation import base_frame, conditional_means, sample
from .packaging import sha256
from .review_policy import require_current_pair
from .schemas import Hypothesis, PairSpec
from .scoring import comparison_key

MASKING_VERSION = "columns-and-text-levels-v1"


class SemanticMask:
    def __init__(self, payload):
        if payload["version"] != MASKING_VERSION:
            raise ValueError("Unknown semantic masking version")
        self.payload = payload
        self.columns = payload["columns"]
        self.values = payload["values"]
        if len(set(self.columns.values())) != len(self.columns):
            raise ValueError("Column aliases must be bijective")
        if any(len(set(v.values())) != len(v) for v in self.values.values()):
            raise ValueError("Level aliases must be bijective within each column")

    def frame(self, frame):
        return mask_frame(frame, self.columns, self.values)

    def hypothesis(self, h, *, inverse=False):
        columns = {v: k for k, v in self.columns.items()} if inverse else self.columns
        values = (
            {
                self.columns.get(k, k): {v: n for n, v in levels.items()}
                for k, levels in self.values.items()
            }
            if inverse
            else self.values
        )
        data = h.model_dump()
        for key in ("exposed", "comparator"):
            data[key] = remap_value(data[key], values.get(h.exposure, {}))
        for key in ("exposure", "outcome"):
            data[key] = columns.get(data[key], data[key])
        for key in ("eligibility", "subgroup"):
            data[key] = [
                {
                    **c,
                    "variable": columns.get(c["variable"], c["variable"]),
                    "value": remap_value(c["value"], values.get(c["variable"], {})),
                }
                for c in data[key]
            ]
        return Hypothesis.model_validate(data)

    def spec(self, original):
        # Runtime coordinate transform only. The reviewed original remains frozen on disk;
        # transformed review hashes are deliberately never asserted to be new reviews.
        spec = original.model_copy(deep=True)
        for d in spec.discoveries:
            d.hypothesis = self.hypothesis(d.hypothesis)
            d.paradigm_bearing_variables = [
                self.columns.get(c, c) for c in d.paradigm_bearing_variables
            ]
        for e in spec.evidence:
            e.candidate.hypothesis = self.hypothesis(e.candidate.hypothesis)
        for o in spec.outcomes:
            o.name = self.columns.get(o.name, o.name)
        return spec

    def original_comparison_key(self, key):
        if not key:
            return key
        outcome, exposure, a, b, contrast, eligibility, subgroup, _ = json.loads(key)
        h = Hypothesis(
            id="seed",
            outcome=outcome,
            exposure=exposure,
            exposed=json.loads(a),
            comparator=json.loads(b),
            contrast=contrast,
            eligibility=[dict(variable=c, op=op, value=json.loads(v)) for c, op, v in eligibility],
            subgroup=[dict(variable=c, op=op, value=json.loads(v)) for c, op, v in subgroup],
            direction=1,
        )
        return comparison_key(self.hypothesis(h, inverse=True))


class MaskedValidationService(WorkflowValidationService):
    def __init__(self, original, version, replicate_id, policy, masking):
        self.original = original
        self.masking = masking
        super().__init__(masking.spec(original), version, replicate_id, policy)

    def derive_seed(self, namespace, key):
        return super().derive_seed(namespace, self.masking.original_comparison_key(key))

    def sample_frame(self, seed):
        return self.masking.frame(sample(self.original, self.version, seed=seed))

    def mean_frame(self, seed, n):
        frame = base_frame(self.original.profile, n, seed, version=self.original.generation_version)
        for name, values in conditional_means(self.original, frame, self.version).items():
            frame[name] = values
        return self.masking.frame(frame)


def mask_package(source: Path, out: Path, *, seed: int = 20260910):
    """Copy a frozen pair package, masking its existing rows rather than regenerating."""
    source, out = source.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(out)
    plans = []
    for path in sorted((source / "private").glob("*/pair.json")):
        spec = PairSpec.model_validate_json(path.read_text())
        require_current_pair(spec)
        assignment = json.loads((path.parent / "assignment.json").read_text())
        frames = []
        for item in assignment.values():
            dataset = source / "public" / item["task_id"] / "dataset.parquet"
            if sha256(dataset) != item["sha256"]:
                raise ValueError(f"Source checksum mismatch: {dataset}")
            frames.append(pd.read_parquet(dataset))
        columns = build_column_mapping(list(frames[0]), [o.name for o in spec.outcomes], seed=seed)
        if spec.profile.endswith("depmap"):
            from ..synthetic.anonymize import extend_outcome_mapping

            columns = extend_outcome_mapping(columns, [o.name for o in spec.outcomes], seed=seed)
        values = build_value_mapping(
            pd.concat(frames, ignore_index=True),
            seed=seed,
            id_columns=("patient_id", "cell_line_id"),
        )
        masking = SemanticMask(
            dict(version=MASKING_VERSION, seed=seed, columns=columns, values=values)
        )
        plans.append((path, assignment, frames, masking))
    if not plans:
        raise ValueError("No source pairs")
    out.mkdir(parents=True)
    audit = {"version": MASKING_VERSION, "source_root": str(source), "pairs": []}
    for path, assignment, frames, masking in plans:
        private = out / "private" / path.parent.name
        shutil.copytree(path.parent, private)
        workflow = json.loads((private / "workflow.json").read_text())
        workflow["masking"] = masking.payload
        (private / "workflow.json").write_text(json.dumps(workflow, indent=2))
        (private / "masking.json").write_text(json.dumps(masking.payload, indent=2))
        for (_version, item), frame in zip(assignment.items(), frames, strict=True):
            original_task = item["task_id"]
            task_id = hashlib.sha256(
                f"{MASKING_VERSION}:{seed}:{original_task}".encode()
            ).hexdigest()[:16]
            public = out / "public" / task_id
            original = source / "public" / original_task
            public.mkdir(parents=True)
            masked = masking.frame(frame)
            masked.to_parquet(public / "dataset.parquet", index=False)
            dictionary = {
                c: {"dtype": str(masked[c].dtype), "missing_n": int(masked[c].isna().sum())}
                for c in masked
            }
            (public / "data_dictionary.json").write_text(json.dumps(dictionary, indent=2))
            task = json.loads((original / "task.json").read_text())
            task["task_id"] = task_id
            for o in task["outcomes"]:
                o["name"] = masking.columns.get(o["name"], o["name"])
            (public / "task.json").write_text(json.dumps(task, indent=2))
            instructions = (original / "instructions.md").read_text()
            # Remove semantic descriptions of the now-opaque research assays.
            start = instructions.find("Research signatures are")
            end = instructions.find("A mean_difference", start)
            if start >= 0 and end >= 0:
                instructions = instructions[:start] + instructions[end:]
            for name, alias in sorted(masking.columns.items(), key=lambda x: -len(x[0])):
                instructions = instructions.replace(name, alias)
            instructions += (
                "\nPredictor names and text categorical values use opaque labels. "
                "Levels are nominal labels, not ordered numbers. Numeric values, "
                "outcome scales, and missingness are preserved.\n"
            )
            (public / "instructions.md").write_text(instructions)
            shutil.copyfile(original / "stage_schema.json", public / "stage_schema.json")
            item.update(
                task_id=task_id,
                source_task_id=original_task,
                source_sha256=item["sha256"],
                sha256=sha256(public / "dataset.parquet"),
            )
        (private / "assignment.json").write_text(json.dumps(assignment, indent=2))
        audit["pairs"].append(
            {
                "pair_id": path.parent.name,
                "assignment": assignment,
                "masked_columns": len(masking.columns),
                "categorical_columns": len(masking.values),
            }
        )
    (out / "package_manifest.json").write_text(json.dumps(audit, indent=2))
    return audit
