#!/usr/bin/env python3
"""Run the GPT-5.6 Luna sequential information-cascade assay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from cascade import (
    CHAIR_CONDITIONS,
    NETWORK_ORDERS,
    SCENARIOS,
    chain_prompt,
    chair_prompt,
    full_log_bayes_factor,
    network_id,
    private_log_bayes_factor,
    private_prompt,
    validate_report,
    write_summary,
)

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "cascade_report.schema.json"
MACOS_CODEX = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
T = TypeVar("T")


@dataclass(frozen=True)
class CascadeCall:
    replicate: int
    scenario_id: str
    network_id: str
    agent_id: str
    stage: str
    condition: str
    prompt: str

    @property
    def call_id(self) -> str:
        return (
            f"r{self.replicate:02d}__{self.scenario_id}__{self.network_id}__"
            f"{self.stage}__{self.condition}__{self.agent_id}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "luna_cascade",
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--codex",
        type=Path,
        help="Path to the Codex CLI (defaults to PATH, then the macOS app bundle).",
    )
    return parser.parse_args()


def resolve_codex_executable(explicit: Path | None = None) -> Path | None:
    """Resolve the Codex CLI without overriding an explicit user selection."""
    if explicit is not None:
        return explicit.expanduser().resolve()
    if discovered := shutil.which("codex"):
        return Path(discovered).resolve()
    if MACOS_CODEX.is_file():
        return MACOS_CODEX
    return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _load_response(path: Path, call: CascadeCall) -> dict[str, Any]:
    response = json.loads(_strip_json_fence(path.read_text(encoding="utf-8")))
    validate_report(
        response,
        scenario_id=call.scenario_id,
        expected_network_id=call.network_id,
        agent_id=call.agent_id,
        stage=call.stage,
        condition=call.condition,
    )
    return response


def _event_stats(raw: str) -> tuple[int, int, int, list[str]]:
    tool_items: set[tuple[str, str]] = set()
    item_types: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if isinstance(item, dict) and isinstance(item.get("type"), str):
            item_type = item["type"]
            item_types.add(item_type)
            if item_type not in {"agent_message", "reasoning"}:
                tool_items.add((str(item.get("id", "unknown")), item_type))
        usage = event.get("usage")
        if isinstance(usage, dict):
            input_tokens = max(input_tokens, int(usage.get("input_tokens", 0) or 0))
            output_tokens = max(output_tokens, int(usage.get("output_tokens", 0) or 0))
    return len(tool_items), input_tokens, output_tokens, sorted(item_types)


def _command(
    args: argparse.Namespace,
    workspace: Path,
    schema_path: Path,
    response_path: Path,
    prompt: str,
) -> list[str]:
    return [
        str(args.codex),
        "exec",
        "--model",
        args.model,
        "-c",
        f"model_reasoning_effort={args.reasoning_effort}",
        "-c",
        "approval_policy=never",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--json",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(response_path),
        "-C",
        str(workspace),
        prompt,
    ]


def _resume_record(call: CascadeCall, args: argparse.Namespace) -> dict[str, Any] | None:
    call_dir = args.out / "raw" / call.call_id
    response_path = call_dir / "response.json"
    meta_path = call_dir / "meta.json"
    if not args.resume or not response_path.exists() or not meta_path.exists():
        return None
    stored_prompt = (call_dir / "prompt.txt").read_text(encoding="utf-8")
    if stored_prompt != call.prompt:
        raise RuntimeError(f"Refusing to resume changed prompt: {call.call_id}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "prompt_sha256": text_sha256(call.prompt),
        "schema_sha256": file_sha256(SCHEMA_PATH),
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise RuntimeError(
                f"Refusing to resume {call.call_id}: {key}={meta.get(key)!r}, expected {expected!r}"
            )
    response = _load_response(response_path, call)
    return {**meta, "response": response, "resumed": True}


def run_call(call: CascadeCall, args: argparse.Namespace) -> dict[str, Any]:
    resumed = _resume_record(call, args)
    if resumed is not None:
        return resumed
    call_dir = args.out / "raw" / call.call_id
    response_path = call_dir / "response.json"
    meta_path = call_dir / "meta.json"
    call_dir.mkdir(parents=True, exist_ok=True)
    (call_dir / "prompt.txt").write_text(call.prompt, encoding="utf-8")
    if args.dry_run:
        response = {
            "scenario_id": call.scenario_id,
            "network_id": call.network_id,
            "agent_id": call.agent_id,
            "stage": call.stage,
            "condition": call.condition,
            "choice": "inconclusive",
            "confidence": 0,
            "estimated_log_bayes_factor": 0.0,
            "evidence_ids": [],
            "reasoning_summary": "dry run",
            "minority_report": "",
            "changed_from_private": False,
            "change_reason": "not_applicable" if call.stage != "chain" else "no_change",
        }
        response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        meta = _meta(call, args, attempts=0, duration=0.0, stats=(0, 0, 0, []))
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return {**meta, "response": response, "resumed": False}

    existing_attempts = [
        int(path.name.split("_")[-1])
        for path in call_dir.glob("attempt_[0-9][0-9]")
        if path.is_dir()
    ]
    base_attempt = max(existing_attempts, default=0)
    last_error = "unknown failure"
    for retry_index in range(1, args.max_retries + 2):
        attempt_number = base_attempt + retry_index
        attempt_dir = call_dir / f"attempt_{attempt_number:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix="luna-cascade-") as tmp:
            workspace = Path(tmp)
            schema_copy = workspace / SCHEMA_PATH.name
            shutil.copyfile(SCHEMA_PATH, schema_copy)
            response_tmp = workspace / "response.json"
            command = _command(args, workspace, schema_copy, response_tmp, call.prompt)
            (attempt_dir / "command.json").write_text(
                json.dumps(command[:-1] + ["<PROMPT IN prompt.txt>"], indent=2) + "\n",
                encoding="utf-8",
            )
            started = time.monotonic()
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                duration = time.monotonic() - started
                last_error = f"timeout after {duration:.1f}s"
                (attempt_dir / "stderr.log").write_text(str(exc), encoding="utf-8")
                continue
            duration = time.monotonic() - started
            (attempt_dir / "events.jsonl").write_text(result.stdout, encoding="utf-8")
            (attempt_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0:
                diagnostic = result.stderr[-1000:] or result.stdout[-2000:]
                last_error = f"codex exit {result.returncode}: {diagnostic}"
                continue
            if not response_tmp.exists():
                last_error = "Codex completed without output-last-message"
                continue
            try:
                response = _load_response(response_tmp, call)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = f"invalid structured response: {exc}"
                (attempt_dir / "invalid_response.txt").write_text(
                    response_tmp.read_text(encoding="utf-8"), encoding="utf-8"
                )
                continue
            shutil.copyfile(response_tmp, response_path)
            stats = _event_stats(result.stdout)
            meta = _meta(
                call,
                args,
                attempts=attempt_number,
                duration=duration,
                stats=stats,
            )
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            return {**meta, "response": response, "resumed": False}
    raise RuntimeError(f"{call.call_id} failed after retries: {last_error}")


def _meta(
    call: CascadeCall,
    args: argparse.Namespace,
    *,
    attempts: int,
    duration: float,
    stats: tuple[int, int, int, list[str]],
) -> dict[str, Any]:
    tool_events, input_tokens, output_tokens, item_types = stats
    return {
        "replicate": call.replicate,
        "scenario_id": call.scenario_id,
        "network_id": call.network_id,
        "agent_id": call.agent_id,
        "stage": call.stage,
        "condition": call.condition,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "prompt_sha256": text_sha256(call.prompt),
        "schema_sha256": file_sha256(SCHEMA_PATH),
        "tool_events": tool_events,
        "item_types": item_types,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "attempts": attempts,
        "duration_seconds": round(duration, 3),
    }


def execute_calls(
    calls: list[CascadeCall], args: argparse.Namespace, *, stage_name: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    print(f"Starting {stage_name}: {len(calls)} Luna calls with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures: dict[Future[dict[str, Any]], CascadeCall] = {
            pool.submit(run_call, call, args): call for call in calls
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            call = futures[future]
            try:
                record = future.result()
                records.append(record)
                print(
                    f"[{stage_name} {completed}/{len(calls)}] {call.call_id}: "
                    f"{record['response']['choice']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{call.call_id}: {exc}")
                print(f"[{stage_name} {completed}/{len(calls)}] FAILED {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("Stage failures:\n" + "\n".join(failures))
    return records


def private_calls(args: argparse.Namespace) -> list[CascadeCall]:
    return [
        CascadeCall(
            replicate=replicate,
            scenario_id=scenario.scenario_id,
            network_id="private",
            agent_id=packet.principal_id,
            stage="private",
            condition="private",
            prompt=private_prompt(scenario, packet.principal_id),
        )
        for replicate in range(1, args.replicates + 1)
        for scenario in SCENARIOS
        for packet in scenario.private_packets
    ]


def _private_index(records: list[dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, Any]]:
    return {
        (record["replicate"], record["scenario_id"], record["agent_id"]): record["response"]
        for record in records
    }


def run_network_path(
    replicate: int,
    scenario: Any,
    order_name: str,
    private: dict[tuple[int, str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    order = NETWORK_ORDERS[order_name]
    prior = {order[0]: private[(replicate, scenario.scenario_id, order[0])]}
    records: list[dict[str, Any]] = []
    for principal_id in order[1:]:
        private_report = private[(replicate, scenario.scenario_id, principal_id)]
        call = CascadeCall(
            replicate=replicate,
            scenario_id=scenario.scenario_id,
            network_id=network_id(scenario, order_name),
            agent_id=principal_id,
            stage="chain",
            condition="verdict_chain",
            prompt=chain_prompt(
                scenario,
                order_name,
                principal_id,
                private_report,
                prior,
            ),
        )
        record = run_call(call, args)
        response = record["response"]
        expected_changed = response["choice"] != private_report["choice"]
        if response["changed_from_private"] != expected_changed:
            raise RuntimeError(
                f"{call.call_id}: changed_from_private disagrees with observed choice change"
            )
        records.append(record)
        prior[principal_id] = response
    return records


def execute_networks(
    private_records: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    private = _private_index(private_records)
    paths = [
        (replicate, scenario, order_name)
        for replicate in range(1, args.replicates + 1)
        for scenario in SCENARIOS
        for order_name in NETWORK_ORDERS
    ]
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    print(
        f"Starting sequential chains: {len(paths)} paths, 3 Luna calls per path",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=min(args.workers, len(paths))) as pool:
        futures: dict[Future[list[dict[str, Any]]], tuple[int, Any, str]] = {
            pool.submit(run_network_path, replicate, scenario, order_name, private, args): (
                replicate,
                scenario,
                order_name,
            )
            for replicate, scenario, order_name in paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            replicate, scenario, order_name = futures[future]
            path_id = f"r{replicate:02d}__{network_id(scenario, order_name)}"
            try:
                path_records = future.result()
                records.extend(path_records)
                choices = " -> ".join(record["response"]["choice"] for record in path_records)
                print(
                    f"[chain {completed}/{len(paths)}] {path_id}: downstream {choices}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{path_id}: {exc}")
                print(f"[chain {completed}/{len(paths)}] FAILED {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("Chain failures:\n" + "\n".join(failures))
    return records


def chair_calls(
    private_records: list[dict[str, Any]],
    chain_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[CascadeCall]:
    private = _private_index(private_records)
    chain = {
        (record["replicate"], record["network_id"], record["agent_id"]): record["response"]
        for record in chain_records
    }
    calls: list[CascadeCall] = []
    for replicate in range(1, args.replicates + 1):
        for scenario in SCENARIOS:
            for order_name, order in NETWORK_ORDERS.items():
                net_id = network_id(scenario, order_name)
                reports = {order[0]: private[(replicate, scenario.scenario_id, order[0])]}
                reports.update(
                    {
                        principal_id: chain[(replicate, net_id, principal_id)]
                        for principal_id in order[1:]
                    }
                )
                for condition in CHAIR_CONDITIONS:
                    calls.append(
                        CascadeCall(
                            replicate=replicate,
                            scenario_id=scenario.scenario_id,
                            network_id=net_id,
                            agent_id="chair",
                            stage="chair",
                            condition=condition,
                            prompt=chair_prompt(scenario, order_name, condition, reports),
                        )
                    )
    return calls


def expected_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "replicates": args.replicates,
        "seed": args.seed,
        "schema_sha256": file_sha256(SCHEMA_PATH),
        "cascade_code_sha256": file_sha256(HERE / "cascade.py"),
        "runner_code_sha256": file_sha256(Path(__file__).resolve()),
        "conditions": list(CHAIR_CONDITIONS),
        "network_orders": {key: list(value) for key, value in NETWORK_ORDERS.items()},
        "scenarios": [
            {
                **asdict(scenario),
                "full_log_bayes_factor": full_log_bayes_factor(scenario),
                "private_log_bayes_factors": {
                    packet.principal_id: private_log_bayes_factor(scenario, packet.principal_id)
                    for packet in scenario.private_packets
                },
            }
            for scenario in SCENARIOS
        ],
    }


def prepare_output(args: argparse.Namespace) -> None:
    manifest = expected_manifest(args)
    manifest_path = args.out / "manifest.json"
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume:
            raise SystemExit(
                f"Output directory is non-empty; pass --resume or choose another: {args.out}"
            )
        if not manifest_path.exists():
            raise SystemExit("Cannot resume without manifest.json")
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored != manifest:
            raise SystemExit(
                "Refusing to resume: manifest does not match current configuration/code"
            )
        return
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.replicates < 1 or args.workers < 1:
        raise SystemExit("--replicates and --workers must be positive")
    args.codex = resolve_codex_executable(args.codex)
    if not args.dry_run and (args.codex is None or not args.codex.is_file()):
        location = args.codex or "'codex' on PATH or the macOS app bundle"
        raise SystemExit(f"Codex CLI not found: {location}; pass --codex PATH")
    prepare_output(args)

    private_records = execute_calls(private_calls(args), args, stage_name="private")
    chain_records = execute_networks(private_records, args)
    chair_records = execute_calls(
        chair_calls(private_records, chain_records, args),
        args,
        stage_name="chair",
    )
    records = sorted(
        private_records + chain_records + chair_records,
        key=lambda record: (
            record["replicate"],
            record["scenario_id"],
            record["network_id"],
            record["stage"],
            record["condition"],
            record["agent_id"],
        ),
    )
    (args.out / "records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    summary = write_summary(records, args.out)
    print(summary["interpretation"], flush=True)
    print(f"Report: {args.out / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
