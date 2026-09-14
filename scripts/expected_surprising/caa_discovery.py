"""Freeze and run matched CAA discovery experiments using the current controller."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

from onc_co_scientist.expected_surprising.experiment import run_cell
from onc_co_scientist.expected_surprising.masking import SemanticMask
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans, run_experiment
from onc_co_scientist.interventions.discovery_pairs import discovery_pairs, training_pairs
from onc_co_scientist.interventions.prompts import write_contrast_pairs

ROOT = Path(__file__).resolve().parents[2]
LATEST = ROOT / "data/expected_surprising_ledger/full_runs/20260910_vllm_recommended"
MASKED = ROOT / "data/expected_surprising_ledger/full_runs/20260910_masked_vllm_recommended"
DEPENDENCIES = ("numpy", "pandas", "scipy", "pydantic", "pyarrow", "openai")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    files = [*root.glob("*/config.yaml"), root / "schedule.json",
             root / "server_manifest.json", *root.glob("*/plan.json"),
             *root.glob("*/resolved_spec.json"), *root.glob("*/schedule.json")]
    for folder in (root / "source", root / "unmasked/input_data", root / "masked/input_data"):
        files.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(root)): sha(p) for p in sorted(files)}


def validate_twins(named, masked, pair_ids):
    """Verify private pair identity and exact row-preserving semantic transformation."""
    for pair_id in pair_ids:
        left, right = (p / "private" / pair_id for p in (named, masked))
        if (left / "pair.json").read_bytes() != (right / "pair.json").read_bytes():
            raise ValueError("Named/masked private pair mismatch")
        mapping = SemanticMask(read(right / "masking.json"))
        if read(right / "workflow.json").get("masking") != mapping.payload:
            raise ValueError("Runtime and audit masking maps differ")
        assignments = [read(p / "assignment.json") for p in (left, right)]
        for version in ("expected", "surprising"):
            frames = []
            for package, assignment in zip((named, masked), assignments, strict=True):
                item = assignment[version]
                dataset = package / "public" / item["task_id"] / "dataset.parquet"
                if sha(dataset) != item["sha256"]:
                    raise ValueError("Frozen dataset checksum mismatch")
                frames.append(pd.read_parquet(dataset))
            pd.testing.assert_frame_equal(mapping.frame(frames[0]), frames[1], check_exact=True)


def validate_server(manifest, arms):
    fingerprint = manifest.get("fingerprint")
    unsigned = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != fingerprint:
        raise ValueError("Invalid server manifest fingerprint")
    if manifest.get("protocol") != "caa-discovery-1" or manifest.get("dtype") != "bfloat16":
        raise ValueError("Discovery CAA requires the fingerprinted BF16 server")
    if manifest.get("compact_agent_context"):
        raise ValueError("Server context compaction would change the scientific prompts")
    available = {a["model_id"]: a for a in manifest["aliases"]}
    if len(set(arms)) != len(arms) or any(a not in available for a in arms):
        raise ValueError("Select distinct model aliases from the server manifest")
    selected = [available[a] for a in arms]
    if sum(a["scale"] is None for a in selected) != 1 or len(selected) < 2:
        raise ValueError("Select exactly one unsteered control and at least one intervention")
    return selected


def prepare(*, out, named, masked, base_config, server_manifest, arms,
            replicates=10, workflows=("persistent",), smoke=False, rationale):
    out, named, masked = out.resolve(), named.resolve(), masked.resolve()
    if out.exists():
        raise FileExistsError(out)
    if replicates < 1:
        raise ValueError("replicates must be positive")
    manifest = read(server_manifest)
    selected = validate_server(manifest, arms)
    raw = yaml.safe_load(base_config.read_text())
    pair_ids = raw["expected_surprising"]["pair_ids"]
    if len(pair_ids) != 1:
        raise ValueError("This initial interaction analysis requires one fixed dataset pair")
    validate_twins(named, masked, pair_ids)
    available = {w["id"]: w for w in raw["workflows"]}
    if not workflows or any(w not in available for w in workflows):
        raise ValueError("Unknown or empty workflow selection")
    out.mkdir(parents=True)
    source = out / "source"
    shutil.copytree(ROOT / "src/onc_co_scientist", source / "src/onc_co_scientist",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("caa_discovery.py", "build_presentation_results.py"):
        destination = source / "scripts/expected_surprising" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts/expected_surprising" / name, destination)
    write(out / "server_manifest.json", manifest)
    jobs = []
    for view, package in (("unmasked", named), ("masked", masked)):
        owner = out / view
        shutil.copytree(package, owner / "input_data")
        config = copy.deepcopy(raw)
        config.update(experiment_id=f"{out.name}-{view}", output_root=str(owner),
                      description="Matched same-backend BF16 CAA discovery experiment",
                      replicates=replicates, max_parallel=1,
                      workflows=[available[w] for w in workflows])
        config["expected_surprising"]["root"] = str(owner / "input_data")
        if smoke:
            config["iteration_policy"]["iterations"] = 6
            config["expected_surprising"]["max_tokens_per_call"] = 2048
        config["models"] = [{
            "id": a["arm"], "model_id": a["model_id"], "adapter": "provider",
            "reasoning_effort": "medium", "provider_config": {
                "kind": "caa_openai", "model_id": a["model_id"],
                "base_url": "http://127.0.0.1:8765/v1", "timeout_s": 1800,
                "reasoning_effort": "medium",
                "server_fingerprint": manifest["fingerprint"], "sampling_profile": "gemma4",
                "disable_thinking_on_final_retry": True, "json_object_output": False,
            },
        } for a in selected]
        (owner / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        spec = load_experiment_spec(owner / "config.yaml")
        run_experiment(spec, dry_run=True)
        jobs.extend({"view": view, "run_id": p["run_id"], "replicate": p["replicate"]}
                    for p in read(owner / "plan.json"))
    # One serial GPU server; shuffle all views, versions and arms within replicate.
    rng = random.Random(raw["schedule_seed"])
    schedule = []
    for replicate in sorted({j["replicate"] for j in jobs}):
        block = [j for j in jobs if j["replicate"] == replicate]
        rng.shuffle(block)
        schedule.extend(block)
    write(out / "schedule.json", schedule)
    write(out / "freeze.json", {
        "schema": "caa-discovery-campaign-1", "purpose": "engineering smoke" if smoke else
            "preliminary fixed-pair scientific experiment",
        "selection_rationale": rationale, "arms": selected, "pair_ids": pair_ids,
        "baseline": "fresh BF16 unsteered arm, not historical FP8 vLLM results",
        "backend_difference": "Prompt JSON and shared validation; no constrained JSON decoding",
        "source_packages": {"unmasked": str(named), "masked": str(masked)},
        "source_config_sha256": sha(base_config), "files": inventory(out),
        "dependencies": {d: importlib.metadata.version(d) for d in DEPENDENCIES},
        "n_runs": len(schedule),
    })
    return read(out / "freeze.json")


def verify(root):
    frozen = read(root / "freeze.json")
    if inventory(root) != frozen["files"]:
        raise ValueError("Frozen campaign inputs, configuration or implementation changed")
    for name, version in frozen["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Frozen dependency changed: {name}")
    return frozen


def run(root, *, limit=None):
    verify(root)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    specs = {v: load_experiment_spec(root / v / "config.yaml") for v in ("unmasked", "masked")}
    server = read(root / "server_manifest.json")
    url = specs["unmasked"].models[0].provider_config["base_url"] + "/caa"
    with urllib.request.urlopen(url, timeout=1800) as response:
        if json.load(response)["fingerprint"] != server["fingerprint"]:
            raise ValueError("Live server differs from the frozen campaign")
    plans = {v: {p.run_id: p for p in build_run_plans(s)} for v, s in specs.items()}
    executed = 0
    for job in read(root / "schedule.json"):
        view, run_id = job["view"], job["run_id"]
        spec = specs[view]
        result_path = root / view / "runs" / run_id / "run.json"
        # Preserve terminal failures as observations; never reroll a failed cell.
        if result_path.exists() and read(result_path).get("status") in {"completed", "failed"}:
            continue
        verify(root)
        result = run_cell(spec, plans[view][run_id], root / view, spec.fingerprint(), resume=True)
        print(view, run_id, result["status"], flush=True)
        executed += 1
        if limit is not None and executed >= limit:
            break


def summarize(root):
    from build_presentation_results import (
        aggregate,
        aggregate_cross_condition,
        inspect_job,
        write_csv,
    )
    frozen = verify(root)
    rows = [inspect_job((plan, view, root / view)) for view in ("unmasked", "masked")
            for plan in read(root / view / "plan.json")]
    control = next(a["model_id"] for a in frozen["arms"] if a["scale"] is None)
    summaries = {}
    for cohort in ("finished", "completed"):
        table, cells, groups = aggregate(rows, cohort)
        cross = aggregate_cross_condition(rows, cohort, table, cells, groups)
        interactions = [r for r in cross if r["metric"] == "label_surprise_interaction"]
        contrasts = []
        for row in interactions:
            baseline = next(r for r in interactions if r["llm"] == control
                            and r["workflow"] == row["workflow"])
            if row["llm"] != control:
                contrasts.append({"arm": row["llm"], "workflow": row["workflow"],
                    "delta_interaction_pp": (row["value"] - baseline["value"]
                        if row["value"] is not None and baseline["value"] is not None else None),
                    "ci95": None, "note": "Descriptive contrast; no treatment-effect CI yet"})
        summaries[cohort] = dict(metrics=table, cells=cells, cross_condition=cross,
                                 intervention_minus_control=contrasts)
        write_csv(root / "analysis" / f"{cohort}_metrics.csv", table)
        write_csv(root / "analysis" / f"{cohort}_cells.csv", cells)
    write(root / "analysis/summary.json", {
        "purpose": frozen["purpose"], "runs": rows, "cohorts": summaries,
        "interpretation": "Lower I is not sufficient: inspect surprising gains, expected harms, "
            "F1, evidence responsiveness, and technical failures. Fixed-pair results do not "
            "establish general paradigm-anchoring mitigation. Queued runs are never zeros.",
    })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pairs = commands.add_parser("pairs")
    pairs.add_argument("--out", type=Path, required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--named", type=Path, default=LATEST / "input_data")
    prep.add_argument("--masked", type=Path, default=MASKED / "input_data")
    prep.add_argument("--base-config", type=Path, default=LATEST / "config.yaml")
    prep.add_argument("--server-manifest", type=Path, required=True)
    prep.add_argument("--arm", dest="arms", action="append", required=True)
    prep.add_argument("--replicates", type=int, default=10)
    prep.add_argument("--workflow", dest="workflows", action="append")
    prep.add_argument("--smoke", action="store_true")
    prep.add_argument("--rationale", required=True,
                      help="Record development-only arm selection or explicitly unvalidated pilot")
    for command in ("run", "summarize", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("root", type=Path)
        if command == "run":
            sub.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "pairs":
        args.out.mkdir(parents=True, exist_ok=False)
        write_contrast_pairs(training_pairs(), args.out / "train.jsonl")
        write_contrast_pairs(discovery_pairs("development"), args.out / "development.jsonl")
        write(args.out / "manifest.json", {"status": "unvalidated constructed contrasts",
              "train_pairs": 26, "development_pairs": 8,
              "pooling": "last token; no generation prompt; no thinking",
              "sign": "positive anchored minus negative evidence-responsive",
              "caution": "Template and response-length confounds remain; not a validated axis"})
    elif args.command == "prepare":
        kwargs = vars(args).copy()
        kwargs.pop("command")
        kwargs["workflows"] = args.workflows or ["persistent"]
        print(json.dumps({k: v for k, v in prepare(**kwargs).items() if k != "files"}, indent=2))
    else:
        root = args.root.resolve()
        verify(root)
        frozen_script = root / "source/scripts/expected_surprising/caa_discovery.py"
        if args.command != "verify" and Path(__file__).resolve() != frozen_script:
            env = {**os.environ, "PYTHONPATH": str(root / "source/src")}
            raise SystemExit(subprocess.call([sys.executable, str(frozen_script),
                                             *sys.argv[1:]], env=env))
        if args.command == "run":
            run(root, limit=args.limit)
        elif args.command == "summarize":
            summarize(root)
        else:
            print("Frozen campaign verified")


if __name__ == "__main__":
    main()
