#!/usr/bin/env python3
"""Run the controlled federated-analysis pilot with fresh Codex CLI agents."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pilot import (
    CONDITIONS,
    SCENARIOS,
    initial_prompt,
    lineage_meta_analysis,
    revision_prompt,
    validate_report,
    write_summary,
)

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "agent_report.schema.json"
MACOS_CODEX = Path("/Applications/ChatGPT.app/Contents/Resources/codex")


def _resolve_codex_path(override: Path | None = None) -> Path:
    """Resolve an explicit Codex path, then PATH, then the macOS app bundle."""

    if override is not None:
        return override.expanduser().resolve()
    if discovered := shutil.which("codex"):
        return Path(discovered).resolve()
    return MACOS_CODEX


@dataclass(frozen=True)
class AgentCall:
    replicate: int
    scenario_id: str
    agent_id: str
    round_name: str
    condition: str
    prompt: str

    @property
    def call_id(self) -> str:
        return f"r{self.replicate:02d}__{self.scenario_id}__{self.condition}__{self.agent_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "luna_pilot",
        help="Output directory. Reuse the same path with --resume after interruption.",
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="low"
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
        default=None,
        help="Path to the Codex CLI (defaults to PATH, then the macOS app bundle).",
    )
    return parser.parse_args()


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _load_response(path: Path, call: AgentCall) -> dict[str, Any]:
    response = json.loads(_strip_json_fence(path.read_text(encoding="utf-8")))
    validate_report(
        response,
        scenario_id=call.scenario_id,
        agent_id=call.agent_id,
        round_name=call.round_name,
        condition=call.condition,
    )
    return response


def _event_stats(raw: str) -> tuple[int, int, int]:
    tool_types = {
        "command_execution",
        "mcp_tool_call",
        "web_search",
        "file_change",
        "computer_initialize_state",
    }
    tool_events = 0
    input_tokens = 0
    output_tokens = 0

    def visit(value: Any) -> None:
        nonlocal tool_events
        if isinstance(value, dict):
            if value.get("type") in tool_types:
                tool_events += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        visit(event)
        usage = event.get("usage") if isinstance(event, dict) else None
        if isinstance(usage, dict):
            input_tokens = max(input_tokens, int(usage.get("input_tokens", 0) or 0))
            output_tokens = max(output_tokens, int(usage.get("output_tokens", 0) or 0))
    return tool_events, input_tokens, output_tokens


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


def run_call(call: AgentCall, args: argparse.Namespace) -> dict[str, Any]:
    call_dir = args.out / "raw" / call.call_id
    final_response_path = call_dir / "response.json"
    final_meta_path = call_dir / "meta.json"
    if args.resume and final_response_path.exists() and final_meta_path.exists():
        response = _load_response(final_response_path, call)
        meta = json.loads(final_meta_path.read_text(encoding="utf-8"))
        return {**meta, "response": response, "resumed": True}

    call_dir.mkdir(parents=True, exist_ok=True)
    (call_dir / "prompt.txt").write_text(call.prompt, encoding="utf-8")
    if args.dry_run:
        stub = {
            "scenario_id": call.scenario_id,
            "agent_id": call.agent_id,
            "round": call.round_name,
            "condition": call.condition,
            "decision": "inconclusive",
            "confidence": 0,
            "estimated_pooled_effect": 0.0,
            "evidence_ids": [],
            "source_ids": [],
            "reasoning_summary": "dry run",
            "minority_report": "",
            "changed_from_initial": False,
            "change_reason": "not_applicable" if call.round_name == "initial" else "no_change",
        }
        final_response_path.write_text(json.dumps(stub, indent=2) + "\n", encoding="utf-8")
        meta = {
            "replicate": call.replicate,
            "scenario_id": call.scenario_id,
            "agent_id": call.agent_id,
            "round": call.round_name,
            "condition": call.condition,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "tool_events": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "attempts": 0,
            "duration_seconds": 0.0,
        }
        final_meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return {**meta, "response": stub, "resumed": False}

    last_error = "unknown failure"
    for attempt in range(1, args.max_retries + 2):
        attempt_dir = call_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="luna-groupthink-") as tmp:
            workspace = Path(tmp)
            schema_copy = workspace / "agent_report.schema.json"
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
            shutil.copyfile(response_tmp, final_response_path)
            tool_events, input_tokens, output_tokens = _event_stats(result.stdout)
            meta = {
                "replicate": call.replicate,
                "scenario_id": call.scenario_id,
                "agent_id": call.agent_id,
                "round": call.round_name,
                "condition": call.condition,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "tool_events": tool_events,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "attempts": attempt,
                "duration_seconds": round(duration, 3),
            }
            final_meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            return {**meta, "response": response, "resumed": False}
    raise RuntimeError(f"{call.call_id} failed after retries: {last_error}")


def execute_stage(
    calls: list[AgentCall], args: argparse.Namespace, *, stage_name: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    print(f"Starting {stage_name}: {len(calls)} Luna calls with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures: dict[Future[dict[str, Any]], AgentCall] = {
            pool.submit(run_call, call, args): call for call in calls
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            call = futures[future]
            try:
                record = future.result()
                records.append(record)
                decision = record["response"]["decision"]
                print(
                    f"[{stage_name} {completed}/{len(calls)}] {call.call_id}: {decision}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - collect all independent call failures
                failures.append(f"{call.call_id}: {exc}")
                print(f"[{stage_name} {completed}/{len(calls)}] FAILED {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("Stage failures:\n" + "\n".join(failures))
    return records


def initial_calls(args: argparse.Namespace) -> list[AgentCall]:
    calls: list[AgentCall] = []
    for replicate in range(1, args.replicates + 1):
        for scenario in SCENARIOS:
            for report in scenario.reports:
                calls.append(
                    AgentCall(
                        replicate=replicate,
                        scenario_id=scenario.scenario_id,
                        agent_id=report.principal_id,
                        round_name="initial",
                        condition="private",
                        prompt=initial_prompt(scenario, report),
                    )
                )
    return calls


def revision_calls(
    args: argparse.Namespace, initial_records: list[dict[str, Any]]
) -> list[AgentCall]:
    indexed = {
        (record["replicate"], record["scenario_id"], record["agent_id"]): record["response"]
        for record in initial_records
    }
    calls: list[AgentCall] = []
    for replicate in range(1, args.replicates + 1):
        for scenario in SCENARIOS:
            reports = {
                report.principal_id: indexed[(replicate, scenario.scenario_id, report.principal_id)]
                for report in scenario.reports
            }
            for condition in CONDITIONS:
                for report in scenario.reports:
                    calls.append(
                        AgentCall(
                            replicate=replicate,
                            scenario_id=scenario.scenario_id,
                            agent_id=report.principal_id,
                            round_name="revision",
                            condition=condition,
                            prompt=revision_prompt(
                                scenario,
                                report.principal_id,
                                reports,
                                condition,
                                seed=args.seed + replicate,
                            ),
                        )
                    )
    return calls


def main() -> int:
    args = parse_args()
    args.codex = _resolve_codex_path(args.codex)
    if args.replicates < 1 or args.workers < 1:
        raise SystemExit("--replicates and --workers must be positive")
    if not args.dry_run and not args.codex.is_file():
        raise SystemExit(f"Codex CLI not found: {args.codex}")
    if args.out.exists() and not args.resume and any(args.out.iterdir()):
        raise SystemExit(
            f"Output directory is non-empty; pass --resume or choose another: {args.out}"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "replicates": args.replicates,
        "workers": args.workers,
        "seed": args.seed,
        "conditions": list(CONDITIONS),
        "scenarios": [
            {
                **asdict(scenario),
                "lineage_meta_analysis": {
                    "pooled_effect": lineage_meta_analysis(scenario)[0],
                    "standard_error": lineage_meta_analysis(scenario)[1],
                    "decision": lineage_meta_analysis(scenario)[2],
                },
            }
            for scenario in SCENARIOS
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    initial_records = execute_stage(initial_calls(args), args, stage_name="initial")
    revision_records = execute_stage(
        revision_calls(args, initial_records), args, stage_name="revision"
    )
    records = sorted(
        initial_records + revision_records,
        key=lambda r: (
            r["replicate"],
            r["scenario_id"],
            r["condition"],
            r["agent_id"],
        ),
    )
    (args.out / "records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    summary = write_summary(records, args.out)
    print(summary["interpretation"], flush=True)
    print(f"Report: {args.out / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
