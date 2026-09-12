"""Freeze the clinical named/masked federation grid without making model calls."""

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from onc_co_scientist.expected_surprising.federation_spec import FederationCell
from onc_co_scientist.expected_surprising.site_statistics import partition_frame
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import run_experiment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    previous = repo / "data/expected_surprising_ledger/full_runs"
    sources = {
        "named": previous / "20260908_clinical10pct_workflows",
        "masked": previous / "20260910_clinical10pct_masked_workflows",
    }
    raw = yaml.safe_load((sources["masked"] / "config.yaml").read_text())
    for source in sources.values():
        frozen = json.loads((source / "frozen_manifest.json").read_text())
        for name, expected in {
            **frozen["input_hashes"],
            "config.yaml": frozen["config_sha256"],
        }.items():
            if digest(source / name) != expected:
                raise ValueError(f"Source input changed: {source / name}")
    # Retain the latest frozen local-model deployment settings, and the prior Claude profile.
    local = yaml.safe_load((previous / "20260910_masked_vllm_recommended/config.yaml").read_text())
    gemma = yaml.safe_load((previous / "20260911_masked_gemma_8060/config.yaml").read_text())
    claude = yaml.safe_load(
        (previous / "20260910_claude_opus5_medium/named/config.yaml").read_text()
    )
    replacements = {m["id"]: m for m in local["models"] + gemma["models"]}
    raw["models"] = [copy.deepcopy(replacements.get(m["id"], m)) for m in raw["models"]]
    raw["models"] += copy.deepcopy(claude["models"])
    raw["federation"] = dict(
        site_counts=[2, 4], seed=20260908, partitions=[dict(id="random", mode="random")]
    )
    # Existing single-site cap is 3x the largest healthy workflow. Preserve that margin.
    raw["budget"]["max_agent_calls"] = 3 * 4 * 25 * (4 * 3 + 2)
    raw["expected_surprising"].update(
        peer_failure_policy="chair_with_available", stage_failure_policy="retain_scientific_scores"
    )
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        repo / "src", out / "source/src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in ("prepare_federated_grid.py", "run_federated_grid.py"):
        shutil.copy2(repo / "scripts/expected_surprising" / name, out / "source" / name)
    selections = {m["id"]: [] for m in raw["models"]}
    all_rows, partition_audits = [], {}
    for condition, source in sources.items():
        folder = out / condition
        folder.mkdir()
        src = yaml.safe_load((source / "config.yaml").read_text())
        shutil.copytree(Path(src["expected_surprising"]["root"]), folder / "input_data")
        config = copy.deepcopy(raw)
        config.update(
            experiment_id=f"{out.name}_{condition}",
            output_root=str(folder),
            description=f"{condition}: 2/4 random sites; same clinical data; 25 iterations",
        )
        config["expected_surprising"]["root"] = str(folder / "input_data")
        (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        spec = load_experiment_spec(folder / "config.yaml")
        run_experiment(spec, dry_run=True)
        plans = json.loads((folder / "plan.json").read_text())
        assert len(plans) == 840
        for p in plans:
            item = {"condition": condition, "run_id": p["run_id"]}
            selections[p["model_profile"]].append(item)
            all_rows.append({"condition": condition, **p})
        # Verify all four presentations use identical random memberships in each repeat.
        for task in spec.tasks:
            frame = pd.read_parquet(task.public_workspace / "dataset.parquet")
            for n in (2, 4):
                for repeat in range(1, 11):
                    _, audit = partition_frame(
                        frame,
                        FederationCell(sites=n, seed=20260908),
                        task.metadata["pair_id"],
                        f"replicate-{repeat:03d}",
                    )
                    key = f"n{n}-r{repeat:03d}"
                    if key in partition_audits:
                        assert partition_audits[key] == audit
                    partition_audits[key] = audit
    for model, rows in selections.items():
        # Interleave named and masked, keeping each condition's seeded plan order.
        named = [r for r in rows if r["condition"] == "named"]
        masked = [r for r in rows if r["condition"] == "masked"]
        write(
            out / "selections" / f"{model}.json",
            [r for pair in zip(named, masked, strict=True) for r in pair],
        )
    write(out / "grid.json", all_rows)
    write(out / "partition_audit.json", partition_audits)
    released = ["sol_medium", "terra_medium", "luna_medium"]
    write(
        out / "release_policy.json",
        dict(
            released_models=released,
            held_models=[m for m in selections if m not in released] + ["biomni_native"],
            instruction="Only Sol/Terra/Luna released. Others require explicit user release.",
        ),
    )
    # Biomni is a native tool-execution harness, not a provider for the three controller workflows.
    # Reserve its comparable native cells explicitly, without mislabeling Qwen controller runs.
    biomni = []
    for c in ("named", "masked"):
        shutil.copy2(
            repo
            / "configs"
            / ("biomni.nsclc.masked.yaml" if c == "masked" else "biomni.nsclc.yaml"),
            out / f"biomni_{c}_reference.yaml",
        )
        for version in ("expected", "surprising"):
            for n in (2, 4):
                for repeat in range(1, 11):
                    biomni.append(
                        dict(
                            condition=c,
                            semantic_condition=version,
                            sites=n,
                            partition="random",
                            replicate=repeat,
                            workflow="biomni-native",
                            model="Inferact/Qwen3.8-27B-NVFP4",
                            reasoning_effort="xhigh",
                            status="held_requires_native_federation_adapter",
                        )
                    )
    write(out / "biomni_reserved_plan.json", biomni)
    cells = Counter(
        (
            p["condition"],
            p["model_profile"],
            p["workflow_id"],
            p["semantic_condition"],
            p["site_count"],
        )
        for p in all_rows
    )
    assert len(cells) == 168 and set(cells.values()) == {10}
    (out / "README.md").write_text(
        "# Federated clinical grid — September 12, 2026\n\n"
        "2 and 4 randomly partitioned sites × named/masked × expected/surprising × "
        "persistent/sequential/deliberative × 10 repeats; 25 iterations. "
        "The prior single-site experiments are the controls. Central and sites use "
        "the same model.\n\n"
        "1,680 controller runs across Sol, Terra, Luna, Astra, Claude Opus 5, Qwen and Gemma. "
        "Only the 720 Sol/Terra/Luna runs are released, with 10 workers per model (30 total). "
        "All other models remain held until the user explicitly releases them.\n\n"
        "Same frozen source datasets, medium reasoning, prior provider settings, "
        "125,000 output-token "
        "ceiling, 1,800-second call timeout, 12 shared analyses per iteration, and "
        "unchanged validation "
        "policy. The shared agent-call cap is 4,200 (three times the largest healthy "
        "1,400-call run), "
        "including central directions/decisions, site peers/chairs, and retries.\n\n"
        "Biomni: 80 native cells are reserved in biomni_reserved_plan.json. Its "
        "existing external-native "
        "harness does not implement federation. These cells require a native "
        "federation adapter before "
        "they can run; the reference configurations retain its Qwen/xhigh settings. They are not "
        "included in the 1,680 executable controller runs.\n\n"
        "Source, inputs, plans, selections and configs are SHA-256 frozen. "
        "release_policy.json is a "
        "separate mutable launch gate. Each model has a process lock and durable "
        "progress under control/. "
        "Resume uses the frozen source and verified call journals.\n\n"
        "Launch/resume one released model with:\n\n```bash\n"
        f"PYTHONPATH={out}/source/src /tmp/ocs-es-refactor-venv/bin/python "
        f"{out}/source/run_federated_grid.py --root {out} --model sol_medium --workers 10\n"
        "```\n\nAppend `--resume` only for an existing execution. "
        "Do not use the generic matrix runner: it does not enforce the release gate.\n"
    )
    hashes = {
        str(p.relative_to(out)): digest(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "release_policy.json"
    }
    write(
        out / "frozen_manifest.json",
        dict(
            frozen_at=datetime.now(UTC).isoformat(),
            hashes=hashes,
            source_grids={c: str(p) for c, p in sources.items()},
            runs=len(all_rows),
            released_runs=720,
            reserved_biomni_runs=80,
        ),
    )
    print(json.dumps(dict(root=str(out), runs=len(all_rows), released_runs=720, cells=len(cells))))


if __name__ == "__main__":
    main()
