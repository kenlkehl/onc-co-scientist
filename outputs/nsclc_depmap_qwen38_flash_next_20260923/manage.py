"""Prepare, execute, and report the isolated NSCLC Flash-Next grid.

Run each workflow separately. Later workflows require a reporting receipt for
the prior one, so scientific work cannot advance before its results are shared.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
FROZEN = ROOT / "source" / "src"
sys.path.insert(0, str(FROZEN if FROZEN.exists() else REPO / "src"))

import pandas as pd
import yaml
from onc_co_scientist.expected_surprising.experiment import run_cell
from onc_co_scientist.expected_surprising.experiment_report import write_report
from onc_co_scientist.expected_surprising.masking import SemanticMask, mask_package
from onc_co_scientist.harness.durable_io import atomic_write_json, atomic_write_text
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans, run_experiment
from onc_co_scientist.providers.vllm_openai import VLLMConfig, VLLMProvider

PAIR = "es-v2-nsclc_depmap-42005"
MODEL = "Inferact/Qwen3.8-Flash-Next-NVFP4"
URL = "http://sn4622130540:8001/v1"
WORKFLOWS = ("persistent", "sequential", "deliberative")
CONDITIONS = ("unmasked", "masked")
PARALLEL = 10


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if (ROOT / "frozen_manifest.json").exists():
        raise FileExistsError("Already prepared; use run or status")
    source = REPO / "data/expected_surprising_ledger_25_iterations"
    unmasked = ROOT / "input_data/unmasked"
    shutil.copytree(source / "private" / PAIR, unmasked / "private" / PAIR)
    assignment = json.loads((unmasked / "private" / PAIR / "assignment.json").read_text())
    for item in assignment.values():
        shutil.copytree(source / "public" / item["task_id"], unmasked / "public" / item["task_id"])
        assert sha(unmasked / "public" / item["task_id"] / "dataset.parquet") == item["sha256"]
    atomic_write_json(unmasked / "package_manifest.json", {
        "source_root": str(source), "pair_id": PAIR, "assignment": assignment,
        "selected_at": now(), "numerical_datasets_regenerated": False,
    })
    masked = ROOT / "input_data/masked"
    mask_audit = mask_package(unmasked, masked, seed=20260910)
    masking = SemanticMask(json.loads((masked / "private" / PAIR / "masking.json").read_text()))
    masked_assignment = json.loads((masked / "private" / PAIR / "assignment.json").read_text())
    for version, item in assignment.items():
        frame = pd.read_parquet(unmasked / "public" / item["task_id"] / "dataset.parquet")
        twin = pd.read_parquet(masked / "public" / masked_assignment[version]["task_id"] / "dataset.parquet")
        pd.testing.assert_frame_equal(masking.frame(frame), twin, check_exact=True)
    atomic_write_json(ROOT / "masking_audit.json", {**mask_audit, "exact_frame_parity": True})
    shutil.copytree(REPO / "src", FROZEN, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    provider = dict(kind="vllm_openai", model_id=MODEL, base_url=URL,
                    timeout_s=7200, max_retries=2, reasoning_effort="xhigh",
                    sampling_profile="qwen3_8", json_object_output=True,
                    disable_thinking_on_final_retry=False)
    inventory = []
    for workflow in WORKFLOWS:
        for condition in CONDITIONS:
            root = ROOT / workflow / condition
            root.mkdir(parents=True)
            w = dict(id=workflow, mode=workflow)
            if workflow == "deliberative":
                w.update(agents_per_stage=2, deliberation_rounds=1)
            raw = dict(
                experiment_id=f"nsclc_depmap_flash_next_20260923_{workflow}_{condition}",
                description="Fresh matched NSCLC DepMap grid with Qwen recommended thinking sampling and xhigh effort.",
                output_root=str(root), workspace_strategy="copy", schedule_seed=20260923,
                expected_surprising=dict(root=str(ROOT / "input_data" / condition), pair_ids=[PAIR],
                    max_tokens_per_call=125000, max_retries_per_stage=2,
                    persistent_history_chars=120000, peer_failure_policy="chair_with_available",
                    stage_failure_policy="retain_scientific_scores"),
                iteration_policy=dict(iterations=25), replicates=10, max_parallel=PARALLEL,
                budget=dict(max_agent_calls=900, max_runtime_seconds_per_call=7200),
                models=[dict(id="qwen38_flash_next_xhigh", model_id=MODEL, adapter="provider",
                             reasoning_effort="xhigh", provider_config=provider)],
                workflows=[w],
            )
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(raw, sort_keys=False))
            spec = load_experiment_spec(config)
            dry = run_experiment(spec, dry_run=True)
            rows = json.loads((root / "plan.json").read_text())
            assert dry["n_runs"] == 20
            assert collections.Counter(r["semantic_condition"] for r in rows) == {"expected": 10, "surprising": 10}
            inventory.append(dict(workflow=workflow, masking=condition, runs=20,
                                  config=str(config), fingerprint=spec.fingerprint()))
    atomic_write_json(ROOT / "grid.json", inventory)
    shutil.copy2("/tmp/qwen_flash_next_preflight.json", ROOT / "preflight_response.json")
    response = json.loads((ROOT / "preflight_response.json").read_text())
    assert response["model"] == MODEL
    assert response["choices"][0]["finish_reason"] == "stop"
    assert json.loads(response["choices"][0]["message"]["content"])["ready"] is True
    atomic_write_json(ROOT / "sampling.json", dict(
        model=MODEL, base_url=URL, reasoning_effort="xhigh", enable_thinking=True,
        temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=0.0, repetition_penalty=1.0,
        disable_thinking_on_final_retry=False, max_tokens=125000,
        source="https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage",
        preflight_excluded_from_scientific_runs=True,
    ))
    paths = [ROOT / "manage.py", ROOT / "grid.json", ROOT / "sampling.json", ROOT / "masking_audit.json"]
    paths += [p for d in (ROOT / "input_data", ROOT / "source") for p in d.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts]
    paths += [ROOT / w / c / f for w in WORKFLOWS for c in CONDITIONS for f in ("config.yaml", "plan.json", "schedule.json")]
    atomic_write_json(ROOT / "frozen_manifest.json", dict(
        prepared_at=now(), hashes={str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)},
        design="3 workflows x 2 versions x 2 masking conditions x 10 repeats = 120 runs",
        iterations=25, phase_order=list(WORKFLOWS), max_parallel=PARALLEL,
    ))
    atomic_write_text(ROOT / "PROTOCOL.md", "\n".join([
        "# NSCLC DepMap Qwen3.8 Flash Next grid", "",
        "120 fresh runs: persistent, then sequential, then deliberative; 40 runs per phase.",
        "Each phase contains ten repeats of expected/masked, expected/unmasked, surprising/masked, and surprising/unmasked.",
        "25 iterations per run (explore, analyze, appraise, synthesize). The existing reviewed numerical datasets are copied unchanged.",
        "Masked twins use the existing audited bijective transform; exact frame parity is verified. Evaluator mappings remain private.",
        "Qwen model: Inferact/Qwen3.8-Flash-Next-NVFP4 at http://sn4622130540:8001/v1; 262144-token advertised context.",
        "Qwen thinking sampling: temperature 1.0, top_p .95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1. Reasoning xhigh throughout, including retries.",
        "125000 completion tokens per call; 7200-second HTTP timeout; two SDK transport retries; two stage repair retries. JSON-object responses.",
        "Ten concurrent runs across both masking conditions. Persistent history budget 120000 characters. Sequential has fresh sessions. Deliberative uses two peer drafts and one chair per stage.",
        "Existing repaired-harness policies: chair with available peers; retain scored scientific work after stage failures. All failures remain in assigned denominators and are reported separately.",
        "Focal primary recovery is reported alongside the existing exact discovery score D; repeats measure stochastic variability on one synthetic dataset pair, not uncertainty across datasets.",
        "A phase must finish and receive a reporting receipt before the next phase can launch. Source, configurations, schedule, and datasets are hashed before launch.",
        "", "Sampling source: https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage", "",
    ]))
    print(json.dumps(dict(root=str(ROOT), runs=120, phases=3, parity=True)), flush=True)


def verify():
    manifest = json.loads((ROOT / "frozen_manifest.json").read_text())
    for name, expected in manifest["hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    import onc_co_scientist.expected_surprising.coordination as module
    assert Path(module.__file__).is_relative_to(FROZEN)


def status(workflow):
    rows = []
    for condition in CONDITIONS:
        root = ROOT / workflow / condition
        plan = json.loads((root / "plan.json").read_text())
        for version in ("expected", "surprising"):
            selected = [p for p in plan if p["semantic_condition"] == version]
            terminal, calls, errors, tokens, started = [], 0, 0, 0, 0
            for p in selected:
                run = root / "runs" / p["run_id"]
                started += int(run.exists())
                if (run / "run.json").exists():
                    terminal.append(json.loads((run / "run.json").read_text()))
                for path in (run / "calls").glob("*.json"):
                    r = json.loads(path.read_text())["result"]
                    calls += 1
                    errors += int(bool(r.get("error")))
                    tokens += (r.get("usage") or {}).get("output_tokens") or 0
            rows.append(dict(version=version, masking=condition, assigned=10, started=started,
                             terminal=len(terminal), completed=sum(r["status"] == "completed" for r in terminal),
                             failed=sum(r["status"] == "failed" for r in terminal), calls=calls,
                             call_errors=errors, known_output_tokens=tokens))
    payload = dict(updated_at=now(), workflow=workflow, cells=rows)
    atomic_write_json(ROOT / workflow / "progress.json", payload)
    return payload


def report(workflow, results_by_condition, specs, plans):
    cells = []
    for condition in CONDITIONS:
        spec = specs[condition]
        root = ROOT / workflow / condition
        results = results_by_condition[condition]
        failures = sum(r["status"] == "failed" for r in results)
        atomic_write_json(root / "summary.json", dict(
            experiment_id=spec.experiment_id, spec_fingerprint=spec.fingerprint(),
            status="completed_with_failures" if failures else "completed", n_runs=20,
            n_completed=20-failures, n_failed=failures, runs=results,
        ))
        write_report(spec, plans[condition], results, root)
        for version in ("expected", "surprising"):
            selected = [r for r in results if r["semantic_condition"] == version]
            assert len(selected) == 10
            reports = [json.loads(Path(r["scientific_report"]).read_text())
                       for r in selected if r.get("scientific_report")]
            ds = [r["scores"]["discovery"]["exact"]["D"] for r in reports]
            cells.append(dict(version=version, masking=condition, assigned=10,
                              completed=sum(r["status"] == "completed" for r in selected),
                              failed=sum(r["status"] == "failed" for r in selected),
                              focal_recovery=sum(r.get("primary_recovery", 0) for r in selected),
                              mean_exact_D=sum(ds)/10,
                              calls=sum(r.get("agent_calls", 0) for r in selected),
                              known_output_tokens=sum((r.get("usage") or {}).get("output_tokens") or 0 for r in selected)))
    atomic_write_json(ROOT / workflow / "RESULTS.json", dict(workflow=workflow, completed_at=now(), cells=cells))
    lines = [f"# {workflow.title()} results", "", f"Completed {now()}", "",
             "| Version | Masking | Completed | Failed | Focal recovery / 10 | Mean exact D | Calls | Known output tokens |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in cells:
        lines.append(f"| {c['version']} | {c['masking']} | {c['completed']} | {c['failed']} | {c['focal_recovery']} | {c['mean_exact_D']:.3f} | {c['calls']} | {c['known_output_tokens']:,} |")
    lines += ["", "All ten assigned repeats remain in each denominator. Scientific scores are retained for partially failed runs under the declared harness policy. See condition-specific expected_surprising_report.md files for full scoring and token-accounting limitations.",
              "Repeats characterize model variability on one synthetic dataset pair; they do not estimate generalization across datasets.", ""]
    atomic_write_text(ROOT / workflow / "RESULTS.md", "\n".join(lines))
    return cells


def run(workflow, resume=False):
    verify()
    phase = ROOT / workflow
    lock = (ROOT / "execution.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    index = WORKFLOWS.index(workflow)
    if index:
        previous = ROOT / WORKFLOWS[index-1]
        receipt = json.loads((previous / "reported.json").read_text())
        assert receipt["results_sha256"] == sha(previous / "RESULTS.md")
    if (phase / "execution.json").exists() and not resume:
        raise FileExistsError("Phase already launched; exact replay requires --resume")
    provider = VLLMProvider(VLLMConfig(model_id=MODEL, base_url=URL, timeout_s=30))
    ids = [m.id for m in provider._client.models.list().data]
    assert MODEL in ids, ids
    execution = dict(workflow=workflow, pid=os.getpid(), started_at=now(), status="running",
                     max_parallel=PARALLEL, assigned=40, resume=resume)
    atomic_write_json(phase / "execution.json", execution)
    specs = {c: load_experiment_spec(phase / c / "config.yaml") for c in CONDITIONS}
    plans = {c: build_run_plans(specs[c]) for c in CONDITIONS}
    schedule = [(c, p) for c in CONDITIONS for p in plans[c]]
    random.Random(20260923).shuffle(schedule)
    # Interleave all four cells within each repeat to keep the launch balanced.
    schedule.sort(key=lambda x: x[1].replicate)
    atomic_write_json(phase / "launch_schedule.json", [dict(masking=c, **p.public_dict()) for c, p in schedule])
    results = {c: [] for c in CONDITIONS}
    pending = list(schedule)
    active = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            while pending or active:
                while pending and len(active) < PARALLEL:
                    c, p = pending.pop(0)
                    f = pool.submit(run_cell, specs[c], p, phase / c, specs[c].fingerprint(), resume=resume)
                    active[f] = (c, p)
                done, _ = concurrent.futures.wait(active, timeout=30,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    c, p = active.pop(f)
                    try:
                        r = f.result()
                    except Exception as error:
                        r = dict(**p.public_dict(), status="failed", error=str(error),
                                 error_type=type(error).__name__, traceback=traceback.format_exc(),
                                 agent_calls=0, usage={}, primary_recovery=0,
                                 stop_reason="worker_error")
                        atomic_write_json(phase / c / "runs" / p.run_id / "run.json", r)
                    results[c].append(r)
                    print(json.dumps(dict(event="run_finished", masking=c, run_id=p.run_id,
                                          status=r["status"], focal_recovery=r.get("primary_recovery"))), flush=True)
                snapshot = status(workflow)
                execution.update(updated_at=now(), terminal=sum(len(r) for r in results.values()),
                                 active=len(active), queued=len(pending))
                atomic_write_json(phase / "execution.json", execution)
                print(json.dumps(snapshot), flush=True)
        cells = report(workflow, results, specs, plans)
        execution.update(status="awaiting_report", completed_at=now(), cells=cells)
        atomic_write_json(phase / "execution.json", execution)
    except BaseException as error:
        execution.update(status="driver_error", error=repr(error), traceback=traceback.format_exc())
        atomic_write_json(phase / "execution.json", execution)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "status", "reported"))
    parser.add_argument("--workflow", choices=WORKFLOWS, default="persistent")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run(args.workflow, args.resume)
    elif args.action == "status":
        print(json.dumps(status(args.workflow), indent=2))
    else:
        path = ROOT / args.workflow / "RESULTS.md"
        assert path.exists()
        atomic_write_json(path.parent / "reported.json", dict(reported_at=now(), results_sha256=sha(path)))


if __name__ == "__main__":
    main()
