"""Prepare and supervise a frozen four-condition native Biomni/Luna campaign."""

import argparse
import fcntl
import json
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import yaml

from onc_co_scientist.external.config import load_spec
from onc_co_scientist.external.runner import provenance, run_cell, validate_inputs, write_reports
from onc_co_scientist.external.smoke import run_smoke
from onc_co_scientist.external.transport import fingerprint, write_json


def prepare(root, source, tools):
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        repo / "src", root / "source/src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(__file__, root / "source/run_biomni_azure.py")
    shutil.copy2(
        repo / "scripts/expected_surprising/download_biomni_data.py",
        root / "source/download_biomni_data.py",
    )
    shutil.copy2(
        repo / "scripts/expected_surprising/probe_biomni_azure.py",
        root / "source/probe_biomni_azure.py",
    )
    plan = []
    for condition in ("named", "masked"):
        folder = root / condition
        shutil.copytree(source / condition / "input_data", folder / "input_data")
        config = {
            "experiment_id": f"{root.name}_{condition}",
            "input_root": str(folder / "input_data"),
            "output_root": str(folder),
            "biomni_root": str(tools / "biomni"),
            "python": str(tools / "biomni-runtime/bin/python"),
            "data_lake": str(tools / "biomni/data/biomni_data/data_lake"),
            "biomni_commit": "400c1f366b96a35ca253e13c9b06c5076af41d65",
            "model": "gpt-5.6-luna",
            "reasoning_effort": "medium",
            "llm_backend": "azure",
            "azure_native_protocol": "structured",
            "base_url": "https://vlmd5aakf765s-openai-eastus2.openai.azure.com/openai/v1",
            "max_tokens": 125000,
            "completion_policy": "fixed",
            "context_length": 1050000,
            "max_requests": 900,
            "request_timeout": 14400,
            "tool_timeout": 600,
            "rounds": 25,
            "replicates": 25,
            "schedule_seed": 20260912,
            "stage_failure_policy": "zero_run",
        }
        (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        spec = load_spec(folder / "config.yaml")
        tasks, _, _ = validate_inputs(spec)
        for task in tasks:
            for replicate in range(1, 26):
                plan.append(
                    {
                        "condition": condition,
                        "version": task.semantic_condition,
                        "task_id": task.id,
                        "replicate": replicate,
                        "run_id": f"{task.id}__biomni_native__r{replicate:03d}",
                    }
                )
    random.Random(20260912).shuffle(plan)
    assert len(plan) == 100
    write_json(root / "plan.json", plan)
    write_json(root / "execution.json", {"status": "prepared", "planned_runs": len(plan)})
    (root / "README.md").write_text(
        "# Native Biomni with Luna on Azure\n\n"
        "100 runs: masked/unmasked × expected/surprising × 25 replicates. "
        "25 reporting rounds per run, 50,000 rows per dataset. "
        "GPT-5.6 Luna, medium reasoning; native Biomni A1 planning, retrieval "
        "and code execution.\n\n"
        "Biomni source is pinned to 400c1f366b96a35ca253e13c9b06c5076af41d65. "
        "Installed base scientific Python environment and full published data lake; "
        "not the complete E1 collection of specialist executables. Unavailable tools remain "
        "visible as native failures. Workers receive only public inputs and their scratch "
        "directory; private evaluation and Azure credentials stay in the broker.\n\n"
        "Azure Responses requests reuse the existing adapter's Entra refresh and shared "
        "deployment pacing. A schema-constrained action/content response is decoded into "
        "one native execute or solution message; helpers keep their text contracts. "
        "No Codex CLI scientist is invoked. Fixed 125,000 output-token "
        "ceiling including reasoning, 900 total model requests per run, service context "
        "truncation disabled. Raw usage/responses are audited. Rate-limit and explicit auth "
        "rejections retry at transport level; ambiguous network failures are retained.\n\n"
        "Four excluded six-round native integration pilots on fresh synthetic rows must "
        "pass before formal admission. Formal failures are retained under zero_run. "
        "Formal run fingerprints freeze source, inputs, runtime packages and data lake. "
        "Content hashes are checked at launch and finish; file metadata is checked before "
        "each admission to avoid rereading the entire lake for every replicate.\n\n"
        "[Live status](STATUS.md) · [Plan](plan.json) · [Execution](execution.json)\n"
    )
    return root


def inventory(specs, root):
    paths = set()
    for folder in (
        root / "source",
        *(s.input_root for s in specs),
        specs[0].biomni_root / "biomni",
        specs[0].data_lake,
    ):
        paths.update(
            p
            for p in folder.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    paths.update(s.output_root / "config.yaml" for s in specs)
    environment = specs[0].python.parent.parent
    paths.add(environment / "pyvenv.cfg")
    paths.update(environment.glob("lib/python*/site-packages/*.dist-info/METADATA"))
    paths.update(environment.glob("lib/python*/site-packages/*.pth"))
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(paths)}


def status(root, phase):
    plan = json.loads((root / "plan.json").read_text())
    counts = {}
    for row in plan:
        key = f"{row['condition']} {row['version']}"
        count = counts.setdefault(key, {"completed": 0, "failed": 0, "running": 0, "pending": 0})
        path = root / row["condition"] / "runs" / row["run_id"] / "run.json"
        state = json.loads(path.read_text())["status"] if path.exists() else "pending"
        count[state] += 1
    now = datetime.now(UTC).isoformat()
    write_json(root / "status.json", {"updated_at": now, "phase": phase, "counts": counts})
    lines = [
        "# Biomni / Luna experiment",
        "",
        f"Updated: {now}",
        "",
        f"Status: **{phase}**",
        "",
        "| Condition | Complete | Failed | Running | Pending |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, c in sorted(counts.items()):
        lines.append(
            f"| {key} | {c['completed']} | {c['failed']} | {c['running']} | {c['pending']} |"
        )
    lines += [
        "",
        "25 replicates per condition; 25 reporting rounds per replicate.",
        "Pilots use separate synthetic rows and are excluded from these counts.",
    ]
    if phase == "downloading_biomni_resources":
        spec = load_spec(root / "named/config.yaml")
        manifest = spec.biomni_root / "data/download_manifest.json"
        resources = json.loads(manifest.read_text()) if manifest.exists() else {}
        size = sum(p.stat().st_size for p in spec.data_lake.glob("*") if p.is_file())
        lines += [
            "",
            f"Biomni resources verified: {len(resources)} / 76. "
            f"Local downloads, including partial files: {size / 1e9:.2f} GB.",
        ]
    temp = root / "STATUS.tmp"
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(root / "STATUS.md")


def await_native_probe(root, probe, spec):
    """Wait on the trusted diagnostic's receipt before starting any pilot."""
    required = {
        "receipt_matches_unseen_input",
        "two_native_executions",
        "observation_returned_to_model",
        "final_uses_actual_observation",
        "usage_complete",
        "structured_command_used",
        "retriever_remains_text",
    }
    started = time.monotonic()
    while True:
        receipt = json.loads((probe / "status.json").read_text())
        if receipt["status"] == "passed":
            checks = receipt.get("checks", {})
            if not all(checks.get(k) is True for k in required):
                raise ValueError("Native execution diagnostic has incomplete checks")
            tested = json.loads((probe / "spec.json").read_text())
            for key in (
                "model",
                "llm_backend",
                "base_url",
                "reasoning_effort",
                "azure_api",
                "azure_native_protocol",
                "biomni_commit",
            ):
                if tested[key] != getattr(spec, key):
                    raise ValueError(f"Native execution diagnostic differs: {key}")
            write_json(root / "native_protocol_gate.json", {"probe": str(probe), **receipt})
            return
        if receipt["status"] != "running":
            raise ValueError("Native execution diagnostic failed; pilots remain unstarted")
        if time.monotonic() - started >= spec.request_timeout:
            raise TimeoutError("Native execution diagnostic did not finish in time")
        time.sleep(10)


def execute(root, workers, probe=None):
    if not Path(__file__).resolve().is_relative_to(root / "source"):
        raise ValueError("Launch the frozen source/run_biomni_azure.py with frozen PYTHONPATH")
    specs = [load_spec(root / c / "config.yaml") for c in ("named", "masked")]
    phase = ["native_protocol_probe" if probe else "excluded_pilots"]
    done = threading.Event()

    def monitor():
        while not done.is_set():
            status(root, phase[0])
            done.wait(30)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        write_json(
            root / "execution.json",
            {
                "status": phase[0],
                "started_at": time.time(),
                "planned_runs": 100,
                "workers": workers,
            },
        )
        if probe:
            await_native_probe(root, probe, specs[0])
            phase[0] = "excluded_pilots"
            write_json(
                root / "execution.json",
                {
                    "status": phase[0],
                    "planned_runs": 100,
                    "native_protocol_probe_passed": True,
                },
            )
        with ThreadPoolExecutor(max_workers=2) as pool:
            gates = list(pool.map(run_smoke, specs))
        if any(g["status"] != "passed" for g in gates):
            raise RuntimeError("Excluded native pilot failed; formal runs remain unstarted")
        phase[0] = "freezing_formal_provenance"
        contexts = {}
        for spec, gate in zip(specs, gates, strict=True):
            tasks, pair, policy = validate_inputs(spec)
            frozen = provenance(spec, tasks)
            digest = fingerprint(frozen)
            if gate["fingerprint"] != digest:
                raise ValueError("Pilot fingerprint changed")
            write_json(spec.output_root / "provenance.json", frozen)
            contexts[spec.output_root.name] = (spec, {t.id: t for t in tasks}, pair, policy, digest)
        baseline = inventory(specs, root)
        write_json(root / "resource_inventory.json", baseline)
        phase[0] = "formal_running"
        write_json(
            root / "execution.json",
            {"status": phase[0], "planned_runs": 100, "workers": workers, "pilots_passed": True},
        )

        def cell(row):
            # Metadata of immutable frozen inputs/resources; content hashes also checked
            # at both ends of the campaign. There is no native access to these writers.
            if inventory(specs, root) != baseline:
                raise ValueError("Frozen resources changed during campaign")
            spec, tasks, pair, policy, digest = contexts[row["condition"]]
            return row["condition"], run_cell(
                spec, tasks[row["task_id"]], pair, policy, row["replicate"], digest, resume=True
            )

        results = {"named": [], "masked": []}
        plan = json.loads((root / "plan.json").read_text())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(cell, row) for row in plan]
            for future in as_completed(pending):
                condition, result = future.result()
                results[condition].append(result)
                write_reports(root / condition, results[condition])
                print(condition, result["run_id"], result["status"], flush=True)
        phase[0] = "final_integrity_check"
        for spec, tasks, _, _, digest in contexts.values():
            if fingerprint(provenance(spec, list(tasks.values()))) != digest:
                raise ValueError("Final resource content verification failed")
        phase[0] = "completed"
        write_json(
            root / "execution.json",
            {
                "status": phase[0],
                "finished_at": time.time(),
                "planned_runs": 100,
                "terminal_runs": sum(map(len, results.values())),
                "failed_runs": sum(r["status"] == "failed" for rs in results.values() for r in rs),
            },
        )
    except BaseException as exc:
        phase[0] = "stopped"
        write_json(
            root / "execution.json",
            {"status": phase[0], "error": f"{type(exc).__name__}: {exc}", "at": time.time()},
        )
        raise
    finally:
        done.set()
        thread.join()
        status(root, phase[0])


def bootstrap(root, workers):
    """Keep resource downloads, excluded pilots, and formal work in one durable job."""
    spec = load_spec(root / "named/config.yaml")
    phase = "downloading_biomni_resources"
    write_json(root / "execution.json", {"status": phase, "planned_runs": 100})
    status(root, phase)
    process = subprocess.Popen(
        [sys.executable, str(root / "source/download_biomni_data.py"), str(spec.biomni_root)]
    )
    while process.poll() is None:
        status(root, phase)
        time.sleep(30)
    if process.returncode:
        write_json(
            root / "execution.json",
            {
                "status": "stopped",
                "planned_runs": 100,
                "error": "Biomni resource download failed before pilots or formal admission",
            },
        )
        status(root, "resource_download_failed")
        raise RuntimeError("Biomni resource download failed")
    execute(root, workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "bootstrap"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--tools", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--probe", type=Path, help="Wait for an existing excluded native probe")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        print(prepare(root, args.source.resolve(), args.tools.resolve()))
    else:
        with (root / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command == "bootstrap":
                if args.probe:
                    parser.error("--probe is supported only with run")
                bootstrap(root, args.workers)
            else:
                execute(root, args.workers, args.probe.resolve() if args.probe else None)
