"""Excluded native smoke gate, always on freshly generated disposable rows."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from ..expected_surprising.generation import sample
from ..expected_surprising.schemas import ValidationPolicy
from .runner import BiomniRunner, preflight, provenance, run_cell, validate_inputs
from .transport import fingerprint, write_json


class SmokeRunner(BiomniRunner):
    def run(self, spec, public, scratch, config_path, broker):
        config = json.loads(config_path.read_text())
        if spec.rounds == 25:
            config["prompt"] += """
This is an excluded full-length integration pilot on fresh synthetic rows.
Conduct the full research task across all 25 reporting rounds. Also compute the mean
of [2,4,6] with native Python and save {"mean":4.0} in /work/smoke_mean.json.
Exercise voluntary validation and assess newly delivered evidence and due checkpoints.
"""
            write_json(config_path, config)
            return super().run(spec, public, scratch, config_path, broker)
        config["prompt"] += """
This is an excluded integration smoke on fresh synthetic rows. Keep the research small:
compute the mean of [2,4,6] with Python and save {"mean":4.0} in /work/smoke_mean.json.
Choose one binary exposure from the public dictionary, register a mean_difference claim for
log_pfs_months, request its canonical analysis, appraise it, and exercise voluntary validation.
Complete all six reporting rounds, inspecting evidence and due checkpoints. Extensive
search is unnecessary. Save the final report. Use native Python and benchmark_exchange.
"""
        write_json(config_path, config)
        return super().run(spec, public, scratch, config_path, broker)


def run_smoke(spec, *, full_length=False):
    endpoint = preflight(spec)
    tasks, pair, _ = validate_inputs(spec)
    frozen = provenance(spec, tasks)
    digest = fingerprint(frozen)
    smoke_root = spec.output_root / "excluded_smoke" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    rounds = 25 if full_length else 6
    rows = pair.n if full_length else 2000
    smoke_spec = spec.model_copy(
        update={"output_root": smoke_root, "rounds": rounds, "replicates": 1}
    )
    policy = ValidationPolicy.default(pair.profile, rounds)
    outcomes = []
    for index, task in enumerate(tasks):
        public = smoke_root / "inputs" / task.id
        public.mkdir(parents=True)
        frame = sample(pair, task.semantic_condition, seed=987654321 + index, n=rows)
        frame.to_parquet(public / "dataset.parquet", index=False)
        for name in ("task.json", "data_dictionary.json"):
            value = json.loads((task.public_workspace / name).read_text())
            if name == "task.json":
                value.update(n=rows, iterations=rounds)
            write_json(public / name, value)
        disposable = task.model_copy(update={"public_workspace": public})
        result = run_cell(smoke_spec, disposable, pair, policy, 1, digest, runner=SmokeRunner())
        root = smoke_root / "runs" / result["run_id"]
        checks = {
            "completed": result["status"] == "completed",
            "accounting": result["usage"]["usage_complete"],
        }
        mean_file = root / "scratch" / "smoke_mean.json"
        checks["native_python"] = (
            mean_file.exists() and json.loads(mean_file.read_text()).get("mean") == 4
        )
        requests = [json.loads(p.read_text()) for p in sorted((root / "llm").glob("*.json"))]
        checks["thinking"] = any(
            r.get("response", {}).get("choices", [{}])[0].get("message", {}).get("reasoning")
            or r.get("response", {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("reasoning_content")
            for r in requests
        )
        checks["request_settings"] = bool(requests) and all(
            r["request"]["max_tokens"]
            == (
                min(
                    spec.max_tokens,
                    spec.context_length - r["prompt_token_count"] - spec.context_guard_tokens,
                )
                if spec.completion_policy == "adaptive" and "prompt_token_count" in r
                else spec.max_tokens
            )
            and r["request"]["chat_template_kwargs"]
            == {"enable_thinking": True, "reasoning_effort": "xhigh"}
            for r in requests
        )
        from .transport import without_reasoning

        checks["reasoning_not_replayed"] = all(
            m == without_reasoning(m)
            for r in requests
            for m in r["request"]["messages"]
            if m.get("role") == "assistant"
        )
        report = json.loads((root / "report.json").read_text())
        checks["validation_exchange"] = bool(report["responsiveness"]["events"])
        checks["deadline_assessments"] = any(
            e.get("delayed") for e in report["responsiveness"]["events"]
        )
        namespace = root / "scratch" / "native_checkpoint.json"
        checks["checkpoint"] = namespace.exists() and json.loads(namespace.read_text())["resumable"]
        outcomes.append({"run": result, "checks": checks})
    gate = {
        "status": "passed" if all(all(o["checks"].values()) for o in outcomes) else "failed",
        "fingerprint": digest,
        "endpoint": endpoint,
        "excluded": True,
        "rounds": rounds,
        "rows": rows,
        "runs": outcomes,
        "smoke_root": str(smoke_root),
    }
    write_json(smoke_root / "gate.json", gate)
    write_json(spec.output_root / "smoke_gate.json", gate)
    return gate
