"""Run paired repeats concurrently through configured providers with frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import threading
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path

import yaml
from smoke_vllm import MeteredProvider, summarize

from onc_co_scientist.expected_surprising.research import now
from onc_co_scientist.expected_surprising.rollout import run
from onc_co_scientist.expected_surprising.schemas import (
    PairSpec,
    ValidationPolicy,
    WorkflowVersions,
)
from onc_co_scientist.expected_surprising.summary import paired_summary


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prepare(config_path, out):
    config = yaml.safe_load(config_path.read_text())
    data = Path(config["data"]).resolve()
    root = Path(__file__).resolve().parents[2]
    profile, repeats = config["profile"], config["replicates"]
    if repeats < 1 or config["workers_per_endpoint"] < 1:
        raise ValueError("Positive repeats and worker counts required")
    labels = [endpoint["label"] for endpoint in config["endpoints"]]
    if len(set(labels)) != len(labels):
        raise ValueError("Endpoint labels must be unique")
    policy = ValidationPolicy.default(profile, config["iterations"])
    pair_paths = list((data / "private").glob(f"es-v2-{profile}-*/pair.json"))
    if len(pair_paths) != 1:
        raise ValueError("Expected exactly one matching pair")
    pair_path = pair_paths[0]
    spec = PairSpec.model_validate_json(pair_path.read_text())
    assignment = json.loads((pair_path.parent / "assignment.json").read_text())
    source_smoke = json.loads(Path(config["source_smoke_manifest"]).read_text())
    smoke_tasks = {t["version"]: t for t in source_smoke["tasks"] if t["pair_id"] == spec.pair_id}
    public_paths = []
    packaged_versions = None
    for version in ("expected", "surprising"):
        public = data / "public" / assignment[version]["task_id"]
        task = json.loads((public / "task.json").read_text())
        WorkflowVersions.model_validate(task["versions"])
        if packaged_versions is not None and task["versions"] != packaged_versions:
            raise ValueError("Paired task workflow versions differ")
        packaged_versions = task["versions"]
        if task["iterations"] != config["iterations"]:
            raise ValueError("Full runs must use the complete packaged iteration budget")
        sha = digest(public / "dataset.parquet")
        if sha != assignment[version]["sha256"] or sha != smoke_tasks[version]["sha256"]:
            raise ValueError("Dataset differs from the requested smoke pair")
        public_paths.append(public)
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    shutil.copytree(root / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(
        root / "scripts/expected_surprising",
        source / "scripts/expected_surprising",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    frozen_data = out / "input_data"
    for public in public_paths:
        shutil.copytree(public, frozen_data / "public" / public.name)
    shutil.copytree(pair_path.parent, frozen_data / "private" / spec.pair_id)
    shutil.copy2(config_path, out / "config.yaml")
    shutil.copy2(config["source_smoke_manifest"], out / "source_smoke_manifest.json")
    repetitions = list(range(1, repeats + 1))
    rng = random.Random(config["order_seed"])
    rng.shuffle(repetitions)
    order = []
    for rep in repetitions:
        versions = ["expected", "surprising"]
        rng.shuffle(versions)
        order.extend((rep, version) for version in versions)
    jobs = []
    for endpoint in config["endpoints"]:
        for rep, version in order:
            jobs.append(
                {
                    "run_id": f"{endpoint['label']}-{profile}-{version}-r{rep:02d}",
                    "endpoint": endpoint["label"],
                    "version": version,
                    "replicate_id": f"full-clinical-repeat-{rep:02d}",
                    "public": str(frozen_data / "public" / assignment[version]["task_id"]),
                    "dataset_sha256": assignment[version]["sha256"],
                }
            )
    manifest = {
        "prepared_at": now(),
        "purpose": "Full paired clinical model comparison",
        "config": config,
        "versions": packaged_versions,
        "policy": policy.model_dump(),
        "validation_alpha": 0.05 / policy.max_comparisons,
        "pair_id": spec.pair_id,
        "pair": str(frozen_data / "private" / spec.pair_id / "pair.json"),
        "jobs": jobs,
        "assigned_runs": len(jobs),
        "expected_stages_per_run": 4 * config["iterations"],
        "temperature": (
            0
            if all(e["provider"]["kind"] != "codex_cli" for e in config["endpoints"])
            else "CLI default; not directly configurable"
        ),
        "repeat_design": (
            "Fresh sessions; distinct validation/confirmation seeds per repeat; "
            "matching repeat IDs across paired versions and models. "
            "Sampling controls are transport-specific."
        ),
        "source_hashes": {str(p.relative_to(out)): digest(p) for p in sorted(source.rglob("*.py"))},
        "input_hashes": {
            str(p.relative_to(out)): digest(p)
            for p in sorted(frozen_data.rglob("*"))
            if p.is_file()
        },
        "dependencies": {
            n: importlib.metadata.version(n)
            for n in ["openai", "numpy", "pandas", "scipy", "pydantic"]
        },
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"prepared": str(out), "runs": len(jobs), "policy": manifest["policy"]}))


def lines(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        with suppress(json.JSONDecodeError):
            records.append(json.loads(line))
    return records


_progress = {}


def progress(path):
    state = _progress.setdefault(
        path,
        {
            "offset": 0,
            "successful_stages": 0,
            "attempt_errors": 0,
            "last_iteration": None,
            "last_stage": None,
        },
    )
    if path.exists():
        with path.open() as handle:
            handle.seek(state["offset"])
            while True:
                line = handle.readline()
                if not line.endswith("\n"):
                    break
                record = json.loads(line)
                if record["kind"] == "stage":
                    state["successful_stages"] += 1
                    payload = record.get("record", record)
                    state["last_iteration"] = payload["iteration"]
                    state["last_stage"] = payload["stage"]
                elif record["kind"] == "attempt_error":
                    state["attempt_errors"] += 1
                state["offset"] = handle.tell()
    return {k: v for k, v in state.items() if k != "offset"}


def write_status(out, manifest, states):
    snapshot = []
    for job in manifest["jobs"]:
        run_id = job["run_id"]
        completed = progress(out / "runs" / run_id / "transcript.jsonl")
        calls = lines(out / "api_metadata" / f"{run_id}.jsonl")
        snapshot.append(
            {
                **job,
                **states[run_id],
                **completed,
                "completed_api_calls": len(calls),
                "completion_tokens": sum(
                    (c.get("usage") or {}).get("completion_tokens", 0) for c in calls
                ),
            }
        )
    execution_path = out / "execution.json"
    driver_pid = (
        json.loads(execution_path.read_text())["pid"] if execution_path.exists() else os.getpid()
    )
    value = {"updated_at": now(), "pid": driver_pid, "runs": snapshot}
    write_json(out / "status.json", value)
    expected = manifest["expected_stages_per_run"]
    text = [
        "# Full clinical runs",
        "",
        f"Updated {value['updated_at']} · process {driver_pid}",
        "",
        f"{len(snapshot)} runs; {manifest['config']['iterations']} iterations each. "
        f"Up to {manifest['config']['workers_per_endpoint']} concurrent runs per endpoint.",
        "",
        "| Endpoint | Queued | Running | Finished | Failed | Successful stages |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for endpoint in manifest["config"]["endpoints"]:
        selected = [r for r in snapshot if r["endpoint"] == endpoint["label"]]
        counts = [
            sum(r["status"] == status for r in selected)
            for status in ("queued", "running", "finished", "failed")
        ]
        text.append(
            f"| {endpoint['label']} | "
            + " | ".join(map(str, counts))
            + f" | {sum(r['successful_stages'] for r in selected)} / {len(selected) * expected} |"
        )
    text += [
        "",
        "Finished runs may contain protocol errors, shown in their reports and scores.",
        "",
        "[Settings](manifest.json) · [Progress](status.json) · [Run summaries](summary.json)",
        "",
        "| Run | State | Successful stages | Last completed step |",
        "|---|---|---:|---|",
    ]
    for r in snapshot:
        step = f"{r['last_iteration']}: {r['last_stage']}" if r["last_iteration"] else "—"
        text.append(
            f"| {r['run_id']} | {r['status']} | {r['successful_stages']} / {expected} | {step} |"
        )
    temporary = out / "STATUS.md.tmp"
    temporary.write_text("\n".join(text) + "\n")
    temporary.replace(out / "STATUS.md")


def write_results(out, manifest, summaries, results):
    def number(value):
        return "Unavailable" if value is None else f"{value:.3f}"

    text = [
        "# Full clinical comparison",
        "",
        f"{len(manifest['jobs'])} assigned runs: 10 expected and 10 surprising per model, "
        "25 iterations per run, on the same clinical dataset pair used in the smoke test.",
        "",
        "| Model | Recall R | Precision P | F1* | Coverage E | Response to evidence B |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, summary in summaries.items():
        components = summary.get("discovery_components", {})
        values = [components.get("R"), components.get("Q")]
        values += [summary.get(key) for key in ("D", "E", "B")]
        text.append(f"| {label} | " + " | ".join(map(number, values)) + " |")
    text += [
        "",
        "| Model | Expected focal recovery | Surprising focal recovery | Difference |",
        "|---|---:|---:|---:|",
    ]
    for label, summary in summaries.items():
        values = [
            summary.get(key)
            for key in ("expected_recovery", "surprising_recovery", "paired_difference")
        ]
        text.append(f"| {label} | " + " | ".join(map(number, values)) + " |")
    complete = sum(r.get("protocol_complete", False) for r in results)
    text += [
        "",
        f"{complete}/{len(results)} runs completed every stage without exhausted retries.",
        "",
        "Recall R is category-balanced sensitivity to embedded discoveries. Precision P is the "
        "fraction of final accepted claims that were tested and independently confirmed, including "
        "additional discoveries. R and P range from 0 to 1.",
        "",
        "Within each run, F1* = 200 × R × P / (R + P), on a 0–100 scale. The asterisk marks "
        "category-balanced recall and confirmation-based precision, rather than one ordinary "
        "confusion matrix. An unrecovered stage error sets that run's primary F1* and focal "
        "recovery to zero under the existing failure rule; descriptive R and P are retained. "
        "The table averages run scores, so its F1* cannot be reconstructed from the displayed "
        "mean R and P, especially when runs incur this failure penalty.",
        "",
        "**Coverage E: how broadly and how early did the agent test the embedded discoveries?** "
        "At the end of each iteration, calculate the fraction of expected, neutral, and surprising "
        "targets tested so far. Average those three fractions, then average across all 25 iterations:",
        "",
        "**E = 100 × average over iterations of [(expected coverage + neutral coverage + "
        "surprising coverage) / 3].**",
        "",
        "A target counts after a valid test of its exact comparison, regardless of the claimed "
        "direction or whether the agent accepts it. Repeated tests give no extra credit. "
        "Testing every target in iteration 1 gives E = 100; first testing them all in iteration "
        "25 gives E = 4. The denominator stays at the configured iteration budget even if a run "
        "ends early, carrying its achieved coverage forward. Near matches are reported separately.",
        "",
        "**Response to evidence B: how often did the agent's later decision agree with the "
        "evaluator's private interval rule?** For each independent validation result delivered "
        "in iteration t, score the assessment at synthesis in iteration t + 2. Agreement earns "
        "1; disagreement or a missing due assessment earns 0. Both agent-requested and automatic "
        "validation count, including comparisons outside the embedded targets.",
        "",
        "| Private evidence class | Interval rule | Decision that earns credit |",
        "|---|---|---|",
        "| Supported | Lower bound > private cutoff | Accept |",
        "| Excluded | Upper bound < private cutoff | Reject |",
        "| Ambiguous | Interval includes or touches private cutoff | Unresolved |",
        "",
        "Intervals are oriented to the claim's direction. Here the private cutoff is 0.10 "
        "natural-log PFS units. Excluded means the interval rules out an effect at least that "
        "large; it does not necessarily rule out any association. Ambiguous means uncertain "
        "relative to that cutoff, not necessarily uncertain about the effect's sign.",
        "",
        "**B = 100 × (supported agreement + excluded agreement + ambiguous agreement) / 3.**",
        "",
        "Calculate each agreement fraction within a run, average available run fractions within "
        "each dataset version, then average versions equally (and base datasets equally when "
        "there is more than one). Finally average the three evidence classes equally. Thus B "
        "is not the fraction of all events correct or a simple average of run-level B scores. "
        "Runs without examples of a class do not contribute to that class's mean; if a class "
        "has no examples at the reporting level, B is unavailable. Invalid results, deadlines "
        "past the iteration budget, and checkpoints not reached because a run was interrupted "
        "are excluded. E and B both range from 0 to 100.",
        "",
        (
            "Agents were told that a relative difference in outcome of 10% or greater is "
            "clinically significant. No log-scale calculation or mechanical decision rule was "
            "given to agents. The evaluator remains unchanged at 0.10 natural-log PFS units "
            "(approximately 11%); the public 10% guidance and private cutoff are close but "
            "not identical. Acceptance did not require validation first."
            if manifest["config"].get("clinical_significance_guidance")
            else "Agents judged effects without prescribed minima. Final confirmation used the private "
            "clinical cutoff of 0.10 natural-log PFS units. Agents did not need to request validation "
            "before accepting claims. The two-iteration checkpoint did not cap investigation."
        ),
        "",
        "Repeated runs were averaged within each version, then versions equally. B was averaged "
        "within evidence class before combining classes. This experiment contains only one base "
        "dataset, so confidence intervals over base datasets are unavailable.",
        "",
        "[Paired summaries and denominators](paired_summaries.json) · [Run details](summary.json)",
        "[Settings and source hashes](manifest.json) · [Execution record](execution.json)",
    ]
    if any(e["provider"]["kind"] == "codex_cli" for e in manifest["config"]["endpoints"]):
        text += [
            "",
            "Codex models used medium reasoning and five concurrent sessions per model. "
            "The CLI uses its default sampling and a 125,000 sampled-token rollout budget; "
            "these controls differ from the vLLM temperature and max-completion-token settings. "
            "Native CLI events and effective command arguments are archived under codex_calls.",
        ]
    diagnostic = out / "B_DIAGNOSTIC.md"
    text += ["", "| Model | Requested reasoning | Requested tier | Runs with all stages successful |", "|---|---|---|---:|"]
    for endpoint in manifest["config"]["endpoints"]:
        provider = endpoint["provider"]
        completed = [r for r in results if r.get("endpoint") == endpoint["label"]]
        text.append(
            f"| {endpoint['label']} | {provider.get('reasoning_effort', 'Server default')} | "
            f"{provider.get('service_tier', 'Server default')} | "
            f"{sum(r.get('protocol_complete', False) for r in completed)}/{len(completed)} |"
        )
    text += ["", "vLLM accepts the requested controls, but its model/template determines their effect; "
             "a successful request does not establish that reasoning budgets match Codex. "
             "The local servers have no verified Priority service tier."]
    if diagnostic.exists():
        text += ["", diagnostic.read_text().strip()]
    (out / "RESULTS.md").write_text("\n".join(text) + "\n")


def execute(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if (out / "execution.json").exists():
        raise FileExistsError(
            "This batch was already launched; existing runs cannot be overwritten"
        )
    for name, expected in {**manifest["source_hashes"], **manifest["input_hashes"]}.items():
        if digest(out / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    import onc_co_scientist.expected_surprising.workflow as workflow

    if not Path(workflow.__file__).resolve().is_relative_to(out / "source"):
        raise ValueError("Launch with PYTHONPATH pointing to this batch's source/src snapshot")
    config = manifest["config"]
    servers = {}
    for endpoint in config["endpoints"]:
        provider = endpoint["provider"]
        if provider["kind"] == "codex_cli":
            servers[endpoint["label"]] = {
                "transport": "codex_cli",
                "model": provider["model_id"],
                "reasoning_effort": provider["reasoning_effort"],
                "temperature_control": "CLI default",
                "token_budget_control": "Codex sampled-token rollout budget",
            }
            continue
        with urllib.request.urlopen(
            provider["base_url"].rstrip("/") + "/models", timeout=20
        ) as response:
            servers[endpoint["label"]] = json.load(response)
        if provider["model_id"] not in {m["id"] for m in servers[endpoint["label"]]["data"]}:
            raise ValueError(f"Requested model is not served at {endpoint['label']}")
    execution = {
        "started_at": now(),
        "pid": os.getpid(),
        "servers": servers,
        "workflow_source": workflow.__file__,
    }
    write_json(out / "execution.json", execution)
    (out / "api_metadata").mkdir()
    spec = PairSpec.model_validate_json(Path(manifest["pair"]).read_text())
    states = {job["run_id"]: {"status": "queued"} for job in manifest["jobs"]}
    results, lock, stop = [], threading.Lock(), threading.Event()

    def monitor():
        while not stop.wait(30):
            with lock:
                write_status(out, manifest, states)

    def one(job, provider_config):
        run_id = job["run_id"]
        with lock:
            states[run_id] = {"status": "running", "started_at": now()}
        run_out = out / "runs" / run_id
        try:
            call_config = dict(provider_config)
            if call_config["kind"] == "codex_cli":
                call_config["audit_dir"] = str(out / "codex_calls" / run_id)
            provider = MeteredProvider(
                call_config, out / "api_metadata" / f"{run_id}.jsonl", run_id
            )
            report = run(
                spec,
                job["version"],
                Path(job["public"]),
                run_out,
                provider,
                run_id=run_id,
                iterations=config["iterations"],
                max_tokens_per_call=config["max_tokens_per_call"],
                max_retries_per_stage=config["max_retries_per_stage"],
                policy=manifest["policy"],
                replicate_id=job["replicate_id"],
            )
            result = {
                **summarize(report, run_out, config["iterations"]),
                "endpoint": job["endpoint"],
                "replicate_id": job["replicate_id"],
            }
        except Exception as exc:
            result = {
                "run_id": run_id,
                "endpoint": job["endpoint"],
                "completed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        with lock:
            states[run_id].update(
                status="finished" if result["completed"] else "failed", completed_at=now()
            )
            results.append(result)
            write_json(out / "summary.json", sorted(results, key=lambda r: r["run_id"]))
        return result

    write_json(out / "summary.json", [])
    write_status(out, manifest, states)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    pools = []
    try:
        futures = []
        for endpoint in config["endpoints"]:
            pool = ThreadPoolExecutor(max_workers=config["workers_per_endpoint"])
            pools.append(pool)
            futures += [
                pool.submit(one, job, endpoint["provider"])
                for job in manifest["jobs"]
                if job["endpoint"] == endpoint["label"]
            ]
        for future in as_completed(futures):
            result = future.result()
            print(f"RUN COMPLETE {result['run_id']}: completed={result['completed']}", flush=True)
        summaries = {}
        for endpoint in config["endpoints"]:
            jobs = [j for j in manifest["jobs"] if j["endpoint"] == endpoint["label"]]
            paths = [out / "runs" / j["run_id"] / "report.json" for j in jobs]
            summaries[endpoint["label"]] = (
                paired_summary([json.loads(p.read_text()) for p in paths])
                if all(p.exists() for p in paths)
                else {"unavailable": "An assigned run failed before producing its report"}
            )
        write_json(out / "paired_summaries.json", summaries)
        write_results(out, manifest, summaries, results)
        execution["completed_at"] = now()
        execution["assigned_runs_finished"] = len(results)
        execution["protocol_complete_runs"] = sum(
            r.get("protocol_complete", False) for r in results
        )
        write_json(out / "execution.json", execution)
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
        stop.set()
        watcher.join()
        write_status(out, manifest, states)
    return 0 if all(r.get("protocol_complete", False) for r in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.execute:
        return execute(out)
    if not args.config:
        parser.error("--config is required to prepare a batch")
    prepare(args.config, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
