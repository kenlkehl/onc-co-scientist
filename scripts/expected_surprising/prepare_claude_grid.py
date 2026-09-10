"""Freeze a Claude Vertex expected/surprising grid with named and masked twins.

Preparation and dry-run planning make no model requests. Start each generated
config explicitly with ``ocs harness run-experiment --config PATH``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from onc_co_scientist.expected_surprising.masking import mask_package
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import run_experiment


def prepare_grid(
    source: Path,
    out: Path,
    *,
    project_id: str,
    region: str = "global",
    model: str = "claude-opus-5",
    effort: str = "medium",
    iterations: int = 6,
    replicates: int = 1,
    max_parallel: int = 2,
    max_tokens: int = 32768,
    pair_ids: list[str] | None = None,
):
    from onc_co_scientist.providers.anthropic_vertex import AnthropicVertexConfig

    AnthropicVertexConfig(model_id=model, reasoning_effort=effort)  # offline validation
    source, out = source.resolve(), out.resolve()
    if not project_id.strip():
        raise ValueError("A project ID is required to freeze the Vertex target")
    # Validate the package and protocol before copying any data.
    template = (
        Path(__file__).resolve().parents[2] / "configs/expected_surprising.claude_vertex.yaml"
    )
    raw = yaml.safe_load(template.read_text())
    raw["expected_surprising"].update(
        root=str(source),
        pair_ids=pair_ids or [],
        max_tokens_per_call=max_tokens,
    )
    raw["iteration_policy"]["iterations"] = iterations
    raw.update(replicates=replicates, max_parallel=max_parallel)
    raw["budget"]["max_agent_calls"] = iterations * 4 * 3 * 3
    raw["models"][0].update(model_id=model, reasoning_effort=effort)
    raw["models"][0]["provider_config"].update(
        model_id=model,
        reasoning_effort=effort,
        project_id=project_id,
        region=region,
    )
    # ExperimentSpec enforces the scientific iteration/release policy and budgets.
    from onc_co_scientist.expected_surprising.experiment import import_tasks
    from onc_co_scientist.harness.experiment import ExperimentSpec

    if any((source / "private").glob("*/masking.json")):
        raise ValueError("Source is already masked; provide the named package")
    spec = ExperimentSpec.model_validate(raw)
    import_tasks(spec.expected_surprising, iterations)
    if out == source or source in out.parents:
        raise ValueError("Output must be outside the source package")
    out.mkdir(parents=True, exist_ok=False)
    configs = []
    for condition in ("named", "masked"):
        root = out / condition
        root.mkdir()
        package = root / "input_data"
        if condition == "named":
            shutil.copytree(source, package)
        else:
            mask_package(source, package)
        raw.update(
            experiment_id=f"claude_vertex_{condition}",
            output_root=str(root / "results"),
        )
        raw["expected_surprising"]["root"] = str(package)
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        spec = load_experiment_spec(path)
        summary = run_experiment(spec, dry_run=True)
        configs.append({"condition": condition, "config": str(path), "runs": summary["n_runs"]})
    (out / "grid.json").write_text(json.dumps(configs, indent=2) + "\n")
    return configs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Current reviewed ledger package")
    p.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", default="global")
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--max-parallel", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--pair-id", action="append", dest="pair_ids")
    args = p.parse_args()
    print(json.dumps(prepare_grid(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
