"""Small, provider-neutral runner for the native Chat Completions protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .research_budget import validate_completion, validate_step
from .transcript import IterationRecord, Transcript

MAX_OUTPUT = 16_000

EXECUTE_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
    "additionalProperties": False,
}


def _submission_schema() -> dict:
    schema = IterationRecord.model_json_schema()
    definitions = schema.pop("$defs", {})
    hypothesis = definitions["HypothesisRecord"]
    hypothesis["required"] = sorted(set(hypothesis.get("required", [])) | {"finding"})
    hypothesis["properties"]["finding"] = {"$ref": "#/$defs/StructuredFinding"}
    return {
        "type": "object",
        "properties": {"iteration": schema},
        "$defs": definitions,
        "required": ["iteration"],
        "additionalProperties": False,
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Run Python in the task workspace. State is fresh; save artifacts locally."
            ),
            "parameters": EXECUTE_SCHEMA,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_iteration",
            "description": (
                "Submit a validated record immediately after an actual research iteration."
            ),
            "parameters": _submission_schema(),
        },
    },
]


class StructuredRunner:
    def __init__(
        self,
        workspace: str | Path,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        timeout: float = 120.0,
        max_turns: int = 100,
        max_tool_calls: int = 200,
        max_generated_tokens: int = 100_000,
        max_tokens_per_call: int = 4096,
        python_timeout: float = 30.0,
        harness_id: str = "structured-runner@1",
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.service_tier = "default" if service_tier == "standard" else service_tier
        self.timeout = timeout
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.max_generated_tokens = max_generated_tokens
        self.max_tokens_per_call = max_tokens_per_call
        self.python_timeout = python_timeout
        self.harness_id = harness_id
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.log_path = self.workspace / "runner_transcript.jsonl"
        self._tokens_used = 0

    def _request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_completion_tokens": max(
                1, min(self.max_tokens_per_call, self.max_generated_tokens - self._tokens_used)
            ),
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.service_tier:
            body["service_tier"] = self.service_tier
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        deadline = time.monotonic() + self.timeout
        last: Exception | None = None
        for attempt in range(3):
            try:
                remaining = max(0.1, deadline - time.monotonic())
                with urllib.request.urlopen(req, timeout=remaining) as response:
                    raw = response.read(MAX_OUTPUT * 8 + 1)
                if len(raw) > MAX_OUTPUT * 8:
                    raise RuntimeError("endpoint response exceeds limit")
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("endpoint returned malformed JSON") from exc
                self._log(body, result)
                if self.service_tier == "default" and result.get("service_tier") not in {
                    "default",
                    "standard",
                }:
                    raise RuntimeError("endpoint did not verify requested Standard service tier")
                return result
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, *range(500, 600)):
                    raise RuntimeError(f"endpoint HTTP {exc.code}") from exc
                last = exc
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.2 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"endpoint transport failed: {last}") from last

    def _log(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        # Requests contain no credential; this also prevents accidental secret logging.
        with self.log_path.open("a", encoding="utf-8") as fh:
            json.dump(
                {
                    "request": request,
                    "response": response,
                    "usage": response.get("usage"),
                    "service_tier": response.get("service_tier"),
                },
                fh,
                separators=(",", ":"),
            )
            fh.write("\n")

    def _python(self, code: str) -> str:
        try:
            from .runtime import run_subprocess_in_group

            env = {
                k: v
                for k, v in os.environ.items()
                if k
                in {"PATH", "LANG", "LC_ALL", "TMPDIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
            }
            scripts = self.workspace / "executed_code"
            scripts.mkdir(exist_ok=True)
            code_file = scripts / f"{time.time_ns()}.py"
            code_file.write_text(code, encoding="utf-8")
            p = run_subprocess_in_group(
                [sys.executable, str(code_file)],
                cwd=self.workspace,
                env=env,
                timeout=self.python_timeout,
            )
            out = (p.stdout + ("\n" + p.stderr if p.stderr else "")).strip()
            return (
                f"exit_code={p.returncode}\n"
                + out[:MAX_OUTPUT]
                + ("\n[output clipped]" if len(out) > MAX_OUTPUT else "")
            )
        except subprocess.TimeoutExpired as exc:
            return f"Python timed out after {self.python_timeout:g}s: {(exc.stdout or '')!s}"[
                :MAX_OUTPUT
            ]
        except Exception as exc:
            return f"Python execution error: {exc}"

    def _submit(self, value: Any, submitted: list[dict[str, Any]], max_iterations: int) -> str:
        if not isinstance(value, dict):
            return "submit_iteration requires an iteration object"
        try:
            record = IterationRecord.model_validate(value)
        except Exception as exc:
            return "format error: " + str(exc)[:2000]
        if record.index > max_iterations:
            return "format error: iteration index exceeds max_iterations"
        if record.index != len(submitted) + 1:
            return f"format error: expected iteration {len(submitted) + 1}"
        target = self.workspace / "iterations" / f"{record.index:03d}.json"
        if target.exists():
            return f"Iteration {record.index} is immutable and already submitted"
        known = {h["id"] for old in submitted for h in old.get("proposed_hypotheses", [])}
        current = [h.id for h in record.proposed_hypotheses]
        if len(current) != len(set(current)) or set(current) & known:
            return "format error: hypothesis IDs must be unique"
        if any(h.finding is None for h in record.proposed_hypotheses):
            return "format error: each nonempty hypothesis requires finding"
        allowed = known | set(current)
        if any(
            hid not in allowed for analysis in record.analyses for hid in analysis.hypothesis_ids
        ):
            return "format error: analysis references unknown hypothesis"
        metadata_path = self.workspace / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        artifact_hashes = None
        if metadata.get("fixed_research_budget"):
            try:
                artifact_hashes = validate_step(
                    self.workspace,
                    record,
                    submitted,
                    sequential_outputs=metadata.get("require_sequential_outputs", False),
                )
            except (ValueError, OSError) as exc:
                return f"format error: {exc}"
        path = self.workspace / "iterations"
        path.mkdir(exist_ok=True)
        payload = json.dumps(record.model_dump(mode="json"), indent=2) + "\n"
        with target.open("x", encoding="utf-8") as record_file:
            record_file.write(payload)
        submitted.append(record.model_dump(mode="json"))
        event = {
            "index": record.index,
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "utc": datetime.now(UTC).isoformat(),
        }
        if artifact_hashes is not None:
            event["research_artifact_sha256"] = artifact_hashes
        with (self.workspace / "submission_events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, separators=(",", ":")) + "\n")
        return f"Accepted iteration {record.index}"

    def run(self) -> Transcript:
        metadata = json.loads((self.workspace / "metadata.json").read_text(encoding="utf-8"))
        instructions = (
            (self.workspace / "agent_instructions.md").read_text(encoding="utf-8")
            if (self.workspace / "agent_instructions.md").exists()
            else "Analyze the dataset."
        )
        max_iterations = int(metadata["max_iterations"])
        if (self.workspace / "iterations").exists() or self.log_path.exists():
            raise RuntimeError("workspace already contains runner state; use submit/finalize")
        system = (
            "You are a scientific analysis agent. Follow the task instructions below. "
            "Use only files in the task workspace; do not seek answer keys. Call "
            "execute_python for analysis and "
            "call submit_iteration with a complete validated record for every actual "
            "iteration before finishing.\n\n" + instructions
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "Begin the analysis and submit iterations using the provided tools.",
            },
        ]
        submitted: list[dict[str, Any]] = []
        turns = tools = tokens = 0
        while (
            turns < self.max_turns
            and tools < self.max_tool_calls
            and tokens < self.max_generated_tokens
        ):
            result = self._request(messages)
            turns += 1
            usage = result.get("usage")
            if not isinstance(usage, dict) or usage.get("completion_tokens") is None:
                raise RuntimeError("endpoint response omitted completion token usage")
            tokens += int(usage["completion_tokens"])
            self._tokens_used = tokens
            choice = (result.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                if metadata.get("fixed_research_budget"):
                    try:
                        validate_completion(
                            metadata, [IterationRecord.model_validate(x) for x in submitted]
                        )
                    except ValueError as exc:
                        messages.append(
                            {
                                "role": "user",
                                "content": str(exc)
                                + ". Continue actual research; do not pad records.",
                            }
                        )
                        continue
                break
            for call in calls:
                if tools >= self.max_tool_calls:
                    break
                tools += 1
                name = (call.get("function") or {}).get("name")
                raw = (call.get("function") or {}).get("arguments", "{}")
                try:
                    args = json.loads(raw)
                except Exception:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                if name == "execute_python":
                    output = self._python(args.get("code", ""))
                elif name == "submit_iteration":
                    output = self._submit(args.get("iteration"), submitted, max_iterations)
                else:
                    output = f"Unknown tool {name!r}"
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": output,
                }
                messages.append(tool_message)
                with (self.workspace / "tool_events.jsonl").open("a", encoding="utf-8") as log:
                    log.write(
                        json.dumps(
                            {
                                "utc": datetime.now(UTC).isoformat(),
                                "call": call,
                                "result": tool_message,
                            }
                        )
                        + "\n"
                    )
        if not submitted:
            raise RuntimeError("agent produced no submitted iterations")
        iterations = [IterationRecord.model_validate(x) for x in submitted]
        iterations.sort(key=lambda x: x.index)
        validate_completion(metadata, iterations)
        transcript = Transcript(
            dataset_id=str(metadata["dataset_id"]),
            model_id=self.model,
            harness_id=self.harness_id,
            max_iterations=max_iterations,
            iterations=iterations,
        )
        (self.workspace / "transcript.json").write_text(
            json.dumps(transcript.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )
        (self.workspace / "runtime_metadata.json").write_text(
            json.dumps(
                {
                    "model_id": self.model,
                    "returned_model": result.get("model"),
                    "service_tier_returned": result.get("service_tier"),
                    "harness_id": self.harness_id,
                    "reasoning_effort": self.reasoning_effort,
                    "turns": turns,
                    "tool_calls": tools,
                    "generated_tokens": tokens,
                    "service_tier_requested": self.service_tier,
                    "stop_reason": (
                        (result.get("choices") or [{}])[0].get("finish_reason")
                        if "result" in locals()
                        else "budget"
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return transcript


def finalize_workspace(
    workspace: str | Path,
    *,
    model_id: str | None = None,
    harness_id: str | None = None,
    write_output: bool = True,
) -> Transcript:
    root = Path(workspace)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    runtime_path = root / "runtime_metadata.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    records = []
    for path in sorted((root / "iterations").glob("*.json")):
        records.append(IterationRecord.model_validate_json(path.read_text(encoding="utf-8")))
    if not records:
        raise ValueError("no iteration records; workspace is incomplete")
    indexes = [record.index for record in records]
    if len(indexes) != len(set(indexes)):
        raise ValueError("duplicate iteration indexes")
    if any(index > int(meta["max_iterations"]) for index in indexes):
        raise ValueError("iteration index exceeds max_iterations")
    records.sort(key=lambda record: record.index)
    if [r.index for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("missing iteration records")
    validate_completion(meta, records)
    events = root / "submission_events.jsonl"
    if not events.exists():
        raise ValueError("submission event log is missing")
    if events.exists():
        entries = [
            json.loads(line)
            for line in events.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [e.get("index") for e in entries] != indexes:
            raise ValueError("submission events do not match record sequence")
        for record in records:
            path = root / "iterations" / f"{record.index:03d}.json"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not any(
                e.get("index") == record.index and e.get("sha256") == digest for e in entries
            ):
                raise ValueError("iteration integrity event missing or mismatched")
            if meta.get("fixed_research_budget"):
                artifact_hashes = validate_step(
                    root,
                    record,
                    [r.model_dump() for r in records if r.index < record.index],
                    sequential_outputs=meta.get("require_sequential_outputs", False),
                )
                event = next(e for e in entries if e["index"] == record.index)
                if event.get("research_artifact_sha256") != artifact_hashes:
                    raise ValueError("research artifact integrity event missing or mismatched")
    seen = set()
    for record in records:
        current = [h.id for h in record.proposed_hypotheses]
        if len(set(current)) != len(current) or seen.intersection(current):
            raise ValueError("duplicate hypothesis IDs")
        if any(h.finding is None for h in record.proposed_hypotheses):
            raise ValueError("structured finding is missing")
        seen.update(current)
        if any(hid not in seen for a in record.analyses for hid in a.hypothesis_ids):
            raise ValueError("unknown or future hypothesis reference")
    transcript = Transcript(
        dataset_id=str(meta["dataset_id"]),
        model_id=model_id or str(runtime.get("model_id", meta.get("model_id", "external"))),
        harness_id=harness_id or str(runtime.get("harness_id", meta.get("harness_id", "external"))),
        max_iterations=int(meta["max_iterations"]),
        iterations=records,
    )
    if write_output:
        (root / "transcript.json").write_text(
            json.dumps(transcript.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )
    return transcript


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--workspace", required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--reasoning-effort")
    run.add_argument("--service-tier", choices=["standard", "default", "priority", "fast"])
    run.add_argument("--timeout", type=float, default=120)
    run.add_argument("--max-turns", type=int, default=100)
    run.add_argument("--max-tool-calls", type=int, default=200)
    run.add_argument("--max-generated-tokens", type=int, default=100000)
    run.add_argument("--max-tokens-per-call", type=int, default=4096)
    run.add_argument("--python-timeout", type=float, default=30)
    submit = sub.add_parser("submit")
    submit.add_argument("--workspace", required=True)
    submit.add_argument("--record", required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--workspace", required=True)
    final.add_argument("--model-id")
    final.add_argument("--harness-id")
    a = parser.parse_args()
    if a.command == "run":
        StructuredRunner(
            a.workspace,
            base_url=a.base_url,
            model=a.model,
            api_key=os.environ.get(a.api_key_env, ""),
            reasoning_effort=a.reasoning_effort,
            service_tier=a.service_tier,
            timeout=a.timeout,
            max_turns=a.max_turns,
            max_tool_calls=a.max_tool_calls,
            max_generated_tokens=a.max_generated_tokens,
            max_tokens_per_call=a.max_tokens_per_call,
            python_timeout=a.python_timeout,
        ).run()
    elif a.command == "finalize":
        finalize_workspace(a.workspace, model_id=a.model_id, harness_id=a.harness_id)
    else:
        meta = json.loads((Path(a.workspace) / "metadata.json").read_text())
        helper = StructuredRunner(
            a.workspace, base_url="http://127.0.0.1", model=str(meta.get("model_id", "external"))
        )
        prior = [
            json.loads(p.read_text())
            for p in sorted((Path(a.workspace) / "iterations").glob("*.json"))
        ]
        result = helper._submit(
            json.loads(Path(a.record).read_text()), prior, int(meta["max_iterations"])
        )
        print(result)
        return 0 if result.startswith("Accepted") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
