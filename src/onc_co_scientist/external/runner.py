"""Run and score native external agents without substituting a custom research loop."""

from __future__ import annotations

import csv
import json
import os
import random
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Protocol

import pandas as pd

from ..expected_surprising.experiment import import_tasks
from ..expected_surprising.packaging import sha256
from ..expected_surprising.prompting import SCIENTIFIC_GUIDANCE, Judgment, Proposal
from ..expected_surprising.review_policy import require_current_pair
from ..expected_surprising.schemas import PairSpec, ValidationPolicy, WorkflowVersions
from ..expected_surprising.workflow import finalize_workflow
from .exchanges import ExchangeGateway, NativeController
from .sandbox import sandbox_command
from .transport import RunBroker, fingerprint, http_json, usage_summary, write_json


class ExternalRunner(Protocol):
    """An engine owns native cognition and tools; the gateway owns benchmark state."""

    def run(self, spec, public, scratch, config_path, broker): ...


class BiomniRunner:
    def run(self, spec, public, scratch, config_path, broker):
        command = sandbox_command(spec, public, scratch, config_path)
        with (scratch.parent / "worker.log").open("a") as output:
            process = subprocess.Popen(
                command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                while process.poll() is None:
                    if broker.fatal:
                        process.terminate()
                        break
                    time.sleep(0.25)
                return process.wait(timeout=30)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise


def task_prompt(rounds, *, masked=False):
    prompt = f"""Investigate diverse, scientifically meaningful comparisons in
/public/dataset.parquet.
Use previous results to allocate investigation. Broad exploration and focused follow-up are both
useful. The public dictionary and task metadata are in /public. Clinical log_pfs_months is natural
log progression-free survival, fully observed without censoring. Research signatures and markers
are constructed assays with no assigned clinical or biological role.
A mean_difference is exposed minus comparator within eligibility and subgroup. An interaction
subtracts that comparison in the subgroup complement in the same eligible population.
{SCIENTIFIC_GUIDANCE}
Use your native planning, code, data lake, research tools and scientific judgment. Retain code,
results, null findings and a final narrative in /work. External knowledge informs interpretation;
claims about this dataset must be supported by its evidence.

Complete {rounds} reporting rounds using benchmark_exchange(request) from Python, printing its
responses. These are reporting/evidence exchanges, not prescribed research roles or analysis stages.
Each request has request_id (unique string), round (starting at 1), action and payload.
- state: payload {{}}; obtain the public ledger and due assessments.
- register: payload {{"proposals": [...]}} using the proposal schema below. Register claims before
  requesting their canonical analysis; initial directions and judgments are immutable. New claims
  receive H references. Refinements/reversals are separate claims linked to their parent.
- analyze: payload {{"run_analyses": ["H1", ...]}}. Up to 12 canonical comparisons per round.
  You may conduct additional native research; official exploration counts canonical tested claims.
- assess: payload {{"assessments": [...]}} using the judgment schema below. Assessments are your
  decisions, never inferred from prose. Omitted evidence attaches displayed direct results.
  Returned estimates/intervals are ALREADY SIGNED to the claimed direction; do not sign them again.
- validate: payload {{"validate": "H1"}} after appraising its discovery evidence. One new voluntary
  request per round and ten total; cached validation is reused. Discovery intervals are 95% Welch
  intervals requiring at least 20 observations per cell. Independent intervals account for the
  maximum number of validation comparisons. Assess every newly delivered result.
- prepare_close: payload {{}} after appraising current evidence; releases scheduled validation and
  seals research submissions for this round. It may deliver new evidence. Assess it before closing.
- close: payload {{"narrative": "...", "assessments": [...]}}. Explicitly reassess every
  due original claim, even if also investigating a refinement. The service maintains
  the accepted claim set.
Automatic evidence releases occur on the configured schedule; reassessment is due two rounds later.
Use the returned assessments_due and round fields. Complete all rounds, then provide your final
<solution>. Do not call close on behalf of later rounds without considering their evidence.
Preserve uncertainty and use accept/reject/unresolved honestly; no particular decision is mandated.
Native data access means recorded initial expectations are not certified as blinded priors.

Proposal schema: {json.dumps(Proposal.model_json_schema())}
Judgment schema: {json.dumps(Judgment.model_json_schema())}
"""
    if masked:
        prompt = prompt.replace(
            "Research signatures and markers\n"
            "are constructed assays with no assigned clinical or biological role.",
            "Predictor names and text categorical values use opaque labels. "
            "Levels are nominal, not ordered numbers. Numeric values, outcome scales, "
            "and missingness are preserved.",
        )
    return prompt


def task_masking(task):
    package = json.loads((task.private_evaluation_path.parent / "workflow.json").read_text())
    if package.get("masking"):
        from ..expected_surprising.masking import SemanticMask

        return SemanticMask(package["masking"])
    return None


def imported_tasks(spec):
    from types import SimpleNamespace

    return import_tasks(
        SimpleNamespace(root=spec.input_root, pair_ids=[spec.pair_id], profiles=[]), spec.rounds
    )


def validate_inputs(spec):
    tasks = imported_tasks(spec)
    pair = PairSpec.model_validate_json(tasks[0].private_evaluation_path.read_text())
    require_current_pair(pair)
    for task in tasks:
        metadata = json.loads((task.public_workspace / "task.json").read_text())
        frame = pd.read_parquet(task.public_workspace / "dataset.parquet")
        if len(frame) != metadata["n"] or len(frame) != pair.n:
            raise ValueError("Parquet row count disagrees with task and pair")
        if not all(o["name"] in frame for o in metadata["outcomes"]):
            raise ValueError("Missing task outcome")
    package = json.loads((tasks[0].private_evaluation_path.parent / "workflow.json").read_text())
    if package["versions"] != WorkflowVersions().model_dump():
        raise ValueError("Unsupported packaged scoring/workflow versions")
    policy = (
        ValidationPolicy.model_validate(package["policy"])
        if spec.rounds == 25
        else ValidationPolicy.default(pair.profile, spec.rounds)
    )
    policy.for_budget(spec.rounds)
    return tasks, pair, policy


def provenance(spec, tasks):
    root = Path(__file__).parents[1]
    code = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}
    native = {
        str(p.relative_to(spec.biomni_root)): sha256(p)
        for p in sorted((spec.biomni_root / "biomni").rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    return {
        "config": spec.model_dump(mode="json"),
        "implementation": code,
        "biomni_source": native,
        "public": {
            task.id: {
                p.name: sha256(p) for p in sorted(task.public_workspace.iterdir()) if p.is_file()
            }
            for task in tasks
        },
        "private": {
            p.name: sha256(p)
            for p in tasks[0].private_evaluation_path.parent.iterdir()
            if p.is_file()
        },
        "worker_environment": subprocess.check_output(
            [str(spec.python), "-m", "pip", "freeze"], text=True
        )
        if (spec.python.parent / "pip").exists()
        else subprocess.check_output(
            [
                str(spec.python),
                "-c",
                "import importlib.metadata,json; "
                "print(json.dumps(sorted((d.metadata['Name'], d.version) "
                "for d in importlib.metadata.distributions())))",
            ],
            text=True,
        ),
        "data_lake": {
            str(p.relative_to(spec.data_lake)): sha256(p)
            for p in sorted(spec.data_lake.rglob("*"))
            if p.is_file()
        },
    }


def preflight(spec, *, live=True):
    tasks, pair, policy = validate_inputs(spec)
    if not spec.python.exists() or not spec.data_lake.is_dir():
        raise ValueError("Biomni Python environment or data lake missing")
    commit = subprocess.check_output(
        ["git", "-C", str(spec.biomni_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != spec.biomni_commit:
        raise ValueError("Biomni commit differs from pinned configuration")
    subprocess.run(
        ["git", "-C", str(spec.biomni_root), "diff", "--exit-code", "HEAD", "--", "biomni"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(spec.python),
            "-c",
            "import langchain_openai,langgraph,cloudpickle,pandas,scipy,statsmodels",
        ],
        check=True,
    )
    result = {
        "status": "static_passed",
        "model": spec.model,
        "biomni_commit": commit,
        "tasks": [t.id for t in tasks],
        "rows": pair.n,
        "policy": policy.model_dump(),
        "resource_policy": "full",
        "environment_scope": "installed Biomni base environment; not complete E1",
        "external_resources": (
            "Available without cloud LLM credentials; unavailable tools remain audited failures"
        ),
    }
    if live:
        key = os.environ[spec.api_key_env] if spec.api_key_env else "EMPTY"
        inventory = http_json(spec.base_url.rstrip("/") + "/models", key=key)
        found = next((m for m in inventory["data"] if m["id"] == spec.model), None)
        if not found or found.get("max_model_len", 0) < spec.context_length:
            raise ValueError("Served model must expose max_model_len >= configured 262144 context")
        result.update(status="endpoint_passed", endpoint_inventory=inventory)
    return result


def run_experiment(spec, *, resume=False):
    # Gate is intentionally required before creating any formal scientific request.
    gate_path = spec.output_root / "smoke_gate.json"
    if not gate_path.exists():
        raise ValueError("Run external-smoke with this exact configuration before formal execution")
    gate = json.loads(gate_path.read_text())
    tasks, pair, policy = validate_inputs(spec)
    preflight(spec)
    frozen = provenance(spec, tasks)
    digest = fingerprint(frozen)
    if gate.get("fingerprint") != digest or gate.get("status") != "passed":
        raise ValueError("Smoke gate missing, failed, or stale for these inputs/configuration/code")
    if spec.completion_policy == "adaptive" and gate.get("rounds", 0) < spec.rounds:
        raise ValueError("Adaptive completion policy requires a full-length excluded pilot")
    manifest = spec.output_root / "provenance.json"
    if manifest.exists() and json.loads(manifest.read_text()) != frozen:
        raise ValueError("Experiment provenance changed; use a new output root")
    write_json(manifest, frozen)
    schedule = [(task, replicate) for task in tasks for replicate in range(1, spec.replicates + 1)]
    random.Random(spec.schedule_seed).shuffle(schedule)
    write_json(
        spec.output_root / "plan.json", [{"task_id": t.id, "replicate": r} for t, r in schedule]
    )
    results = []
    for task, replicate in schedule:
        if fingerprint(provenance(spec, tasks)) != digest:
            raise ValueError(
                "Frozen implementation, inputs, or resources changed during the campaign"
            )
        preflight(spec)
        result = run_cell(spec, task, pair, policy, replicate, digest, resume=resume)
        results.append(result)
        write_reports(spec.output_root, results)
    return results


def run_cell(spec, task, pair, policy, replicate, digest, *, resume=False, runner=None):
    run_id = f"{task.id}__biomni_native__r{replicate:03d}"
    root = spec.output_root / "runs" / run_id
    run_path = root / "run.json"
    if run_path.exists():
        existing = json.loads(run_path.read_text())
        if not resume:
            raise FileExistsError("Run exists; use --resume or a fresh output root")
        if existing["fingerprint"] != digest:
            raise ValueError("Run fingerprint changed")
        if existing["status"] in {"completed", "failed"}:
            if sha256(root / "report.json") != existing["report_sha256"]:
                raise ValueError("Saved scientific report changed")
            for name, expected in existing.get("artifact_sha256", {}).items():
                path = root / name
                if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != expected:
                    raise ValueError("Saved native run artifact changed")
            return existing
    elif root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("Partial run exists; do not overwrite")
    root.mkdir(parents=True, exist_ok=True)
    scratch, public = root / "scratch", root / "public"
    scratch.mkdir(exist_ok=True)
    public.mkdir(exist_ok=True)
    for name in ("dataset.parquet", "task.json", "data_dictionary.json"):
        source, target = task.public_workspace / name, public / name
        if target.exists():
            if sha256(source) != sha256(target):
                raise ValueError("Copied public file changed")
        else:
            shutil.copyfile(source, target)
    controller = NativeController(
        pair,
        task.semantic_condition,
        pd.read_parquet(public / "dataset.parquet"),
        policy,
        f"replicate-{replicate:03d}",
    )
    masking = task_masking(task)
    if masking:
        from ..expected_surprising.masking import MaskedValidationService

        controller.spec = masking.spec(pair)
        controller.outcomes = {o.name: o for o in controller.spec.outcomes}
        controller.service = MaskedValidationService(
            pair, task.semantic_condition, f"replicate-{replicate:03d}", policy, masking
        )
    gateway = ExchangeGateway(controller, spec.rounds)
    broker = RunBroker(spec, gateway, root, secrets.token_hex(32))
    journal = root / "exchanges.jsonl"
    native_resume = False
    checkpoint_path = scratch / "native_checkpoint.json"
    if journal.exists() or broker.requests or checkpoint_path.exists():
        for line in journal.read_text().splitlines() if journal.exists() else []:
            event = json.loads(line)
            if gateway.exchange(event["request"]) != event["response"]:
                raise ValueError("Scientific exchange replay differs")
            broker.exchange_count += 1
        if not checkpoint_path.exists():
            raise ValueError("No native checkpoint; prior native work will not be replayed")
        checkpoint = json.loads(checkpoint_path.read_text())
        if (
            not checkpoint["resumable"]
            or checkpoint["exchange_count"] != broker.exchange_count
            or checkpoint["llm_requests"] != len(broker.requests)
        ):
            raise ValueError("No consistent native checkpoint; arbitrary code will not be replayed")
        native_resume = True
    started = time.time()
    write_json(run_path, {"fingerprint": digest, "status": "running", "run_id": run_id})
    url = broker.start()
    config_path = root / "worker_config.json"
    write_json(
        config_path,
        {
            "broker": url,
            "secret": broker.secret,
            "model": spec.model,
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
            "request_timeout": spec.request_timeout,
            "tool_timeout": spec.tool_timeout,
            "prompt": task_prompt(spec.rounds, masked=masking is not None),
            "run_id": run_id,
            "resume": native_resume,
        },
    )
    failure = None
    try:
        code = (runner or BiomniRunner()).run(spec, public, scratch, config_path, broker)
        if code or gateway.state["completed_rounds"] != spec.rounds:
            failure = (
                broker.fatal or f"native_exit_{code}_rounds_{gateway.state['completed_rounds']}"
            )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        broker.stop()
        # A broker token is ephemeral and not scientific provenance.
        config_path.unlink(missing_ok=True)
    errors = [{"error": failure}] if failure else []
    report = finalize_workflow(
        controller,
        public,
        root,
        model_id=spec.model,
        run_id=run_id,
        iterations=spec.rounds,
        versions=WorkflowVersions().model_dump(),
        stage_failure_policy=spec.stage_failure_policy,
        errors=errors,
        exhausted=errors,
    )
    report.update(
        harness="external-native",
        workflow_id="biomni-native",
        workflow_mode="external-native",
        model_profile="biomni-qwen38-xhigh",
        resource_policy="full",
        clock="reporting_round",
        completed_rounds=gateway.state["completed_rounds"],
        completion_rate=gateway.state["completed_rounds"] / spec.rounds,
        termination=failure or "reporting_rounds_complete",
        initial_expectations_blinded=False,
        completion_policy=spec.completion_policy,
        max_completion_tokens=spec.max_tokens,
        context_length=spec.context_length,
        context_guard_tokens=spec.context_guard_tokens,
        min_completion_tokens=spec.min_completion_tokens,
        reasoning_history_policy="audit_only",
        dataset_view="masked" if masking else "named",
    )
    assignment = json.loads((task.private_evaluation_path.parent / "assignment.json").read_text())
    origin = assignment[task.semantic_condition]
    report["source_dataset_sha256"] = origin.get("source_sha256", origin["sha256"])
    report.pop("successful_stages", None)
    report.pop("expected_stages", None)
    report["retry_policy"] = {
        "atomic_exchange": True,
        "automatic_llm_retries": 0,
        "stage_failure_policy": spec.stage_failure_policy,
    }
    usage = usage_summary(broker.requests)
    report["usage"] = usage
    # Tokens are the common resource axis. Reporting rounds are intentionally
    # not treated as equivalent to stages or native graph steps.
    by_tokens = []
    if journal.exists():
        seen = set()
        for line in journal.read_text().splitlines():
            event = json.loads(line)
            request = event["request"]
            if request["request_id"] in seen or request["action"] != "close":
                continue
            seen.add(request["request_id"])
            at = usage_summary(broker.requests[: event["llm_requests"]])
            total = at["input_tokens"] + at["output_tokens"] if at["usage_complete"] else None
            by_tokens.append(
                {
                    "round": request["round"],
                    "total_tokens": total,
                    "coverage": {
                        category: curve[request["round"] - 1]
                        for category, curve in report["scores"]["coverage"]["exact"][
                            "curves"
                        ].items()
                    },
                }
            )
    report["exploration_by_tokens"] = by_tokens
    write_json(root / "report.json", report)
    result = {
        "run_id": run_id,
        "task_id": task.id,
        "version": task.semantic_condition,
        "pair_id": pair.pair_id,
        "replicate": replicate,
        "fingerprint": digest,
        "status": "failed" if failure else "completed",
        "error": failure,
        "completed_rounds": gateway.state["completed_rounds"],
        "usage": usage,
        "duration_seconds": time.time() - started,
        "report_sha256": sha256(root / "report.json"),
    }
    audited = [
        root / "exchanges.jsonl",
        *(root / "llm").glob("*.json"),
        *(scratch.glob("native_*")),
        *(scratch.glob("network_events.jsonl")),
    ]
    result["artifact_sha256"] = {
        str(path.relative_to(root)): sha256(path)
        for path in audited
        if path.is_file() and not path.is_symlink()
    }
    write_json(run_path, result)
    return result


def write_reports(root, results):
    rows = []
    for run in results:
        report = json.loads((root / "runs" / run["run_id"] / "report.json").read_text())
        scores = report["scores"]
        discovery = scores["discovery"]["exact"]
        rows.append(
            {
                "run_id": run["run_id"],
                "version": run["version"],
                "status": run["status"],
                "R": discovery["R"],
                "P": discovery["Q"],
                "F1_star": scores["D"],
                "E": scores["E"],
                "B": scores["B"],
                "rounds": run["completed_rounds"],
                **run["usage"],
            }
        )
    write_json(
        root / "summary.json",
        {
            "runs": rows,
            "interpretation": (
                "Descriptive repeats of one dataset pair; full Biomni resources, "
                "reporting-round clock; no across-dataset confidence intervals. "
                "Missing precision or responsiveness stays null."
            ),
        },
    )
    with (root / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Biomni native benchmark",
        "",
        "Full resources; 25 reporting rounds. R and P are fractions; F1*, E and B are 0–100.",
        "Blank values are unavailable. Failures remain in the run table. "
        "Repeats are descriptive for one dataset pair.",
        "",
        "| Run | Version | Status | R | P | F1* | E | B | Input tokens | Output tokens |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                "" if row[k] is None else str(row[k])
                for k in (
                    "run_id",
                    "version",
                    "status",
                    "R",
                    "P",
                    "F1_star",
                    "E",
                    "B",
                    "input_tokens",
                    "output_tokens",
                )
            )
            + " |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
