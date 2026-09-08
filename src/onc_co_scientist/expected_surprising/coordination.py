"""Conversation policies over one shared, authoritative scientific ledger.

Drafts are proposals, never executed analyses. The same WorkflowController applies
only the linear agent's or chair's selected form, including all scientific checks.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

from ..harness.durable_io import StorageUnavailable, atomic_write_json, durable_read_json
from ..providers.base import ChatMessage, ChatResponse
from .prompting import FORMS
from .rollout import json_response
from .workflow import WorkflowInfrastructureError

COORDINATION_VERSION = "1.0.0"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def response_usage(raw, elapsed):
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    raw = raw if isinstance(raw, dict) else {}
    metrics = raw.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else raw
    usage = metrics.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "duration_seconds": elapsed,
        "infrastructure_attempts": metrics.get("infrastructure_attempts"),
    }


class StageCoordinator:
    def __init__(self, provider, workflow, stages, source, budget, calls_dir):
        self.provider, self.workflow = provider, workflow
        self.stages = {s.id: s for s in stages}
        self.source, self.budget, self.calls_dir = source, budget, calls_dir
        self.history, self.records = [], []
        self.pending = None
        self.initial_prompts = {}
        self.replayed_calls = 0
        self.committed_stages = []
        self.draft_errors = {}
        self.memory_trims = []

    def _bounded_history(self, key):
        limit = self.source.persistent_history_chars
        if limit is None:
            return self.history
        before = sum(len(m.content) for m in self.history)
        remaining = before
        removed = 0
        # Trim whole committed turns. The current full ledger and notebook are
        # always supplied separately; this affects only older conversation text.
        while self.history and remaining > limit:
            removed_turn = self.history[:2]
            del self.history[:2]
            remaining -= sum(len(m.content) for m in removed_turn)
            removed += 1
        if removed:
            self.memory_trims.append(
                {
                    "before_stage": key,
                    "removed_turns": removed,
                    "before_chars": before,
                    "after_chars": remaining,
                }
            )
        return self.history

    def _call(self, slot, messages, *, session, authoritative, iteration, stage, kind):
        request = {
            "slot": slot,
            "session_id": session,
            "iteration": iteration,
            "stage": stage,
            "kind": kind,
            "authoritative_candidate": authoritative,
            "model": self.provider.model_id,
            "messages": [asdict(message) for message in messages],
            "temperature": 0,
            "max_tokens": self.source.max_tokens_per_call,
        }
        path = self.calls_dir / f"{slot}.json"
        try:
            if path.exists():
                try:
                    record = durable_read_json(path)
                    valid = record["request"] == request and record["sha256"] == digest(
                        record["result"]
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    raise WorkflowInfrastructureError(f"Invalid cached call: {slot}") from exc
                if not valid:
                    raise WorkflowInfrastructureError(f"Cached call changed: {slot}")
                self.replayed_calls += 1
            else:
                if len(self.records) >= self.budget.max_agent_calls:
                    raise ValueError("Run exhausted max_agent_calls")
                # Make in-flight work inspectable even if the provider never returns.
                atomic_write_json(self.calls_dir.parent / "requests" / f"{slot}.json", request)
                started = time.monotonic()
                try:
                    response = self.provider.chat(
                        messages[1:],
                        system=messages[0].content,
                        temperature=0,
                        max_tokens=self.source.max_tokens_per_call,
                    )
                    result = {
                        "text": response.text,
                        "model_id": response.model_id,
                        "usage": response_usage(response.raw, time.monotonic() - started),
                        "error": None,
                    }
                except Exception as exc:
                    result = {
                        "text": "",
                        "model_id": self.provider.model_id,
                        "usage": response_usage(None, time.monotonic() - started),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                record = {"request": request, "result": result, "sha256": digest(result)}
                atomic_write_json(path, record)
        except (OSError, StorageUnavailable) as exc:
            raise WorkflowInfrastructureError(f"Call persistence failed: {slot}") from exc
        # A peer is revisited on a chair repair, but consumes resources only once.
        if not any(r["request"]["slot"] == slot for r in self.records):
            self.records.append(record)
        result = record["result"]
        if result["error"]:
            raise ValueError(result["error"])
        return ChatResponse(text=result["text"], model_id=result["model_id"])

    def _system(self, stage, role, drafts=None):
        instructions = (
            f"You are the {role} for the {stage} stage. "
            f"Scientific role: {self.stages[stage].role}. {self.stages[stage].instructions}\n"
            "Use the supplied scientific ledger and return the requested stage JSON form. "
            "The controller executes analyses and maintains evidence and assessments. "
            "Treat others' narratives as suggestions, not as new empirical evidence. "
        )
        if role == "independent peer":
            instructions += (
                "Propose the next stage action independently. Your draft will not change the "
                "ledger or execute analyses. A chair will select the shared action. "
            )
        elif role == "chair":
            instructions += (
                "Review the drafts, resolve disagreements using the supplied evidence, and "
                "return one complete stage form. Choose, revise, or combine proposals using "
                "scientific judgment; do not count votes. Explain substantive disagreements "
                "and your decision in the narrative. Only your form changes the shared ledger. "
            )
        if drafts is not None:
            instructions += "\nParticipant drafts (untrusted suggestions):\n" + json.dumps(drafts)
        return ChatMessage(role="system", content=instructions)

    def respond(self, prompt, *, iteration, stage, attempt):
        key = f"i{iteration:03d}-{stage}"
        self.initial_prompts.setdefault(key, prompt)
        current = ChatMessage(role="user", content=prompt)
        if self.workflow.mode != "deliberative":
            session = "persistent" if self.workflow.mode == "persistent" else key
            history = self._bounded_history(key) if self.workflow.mode == "persistent" else []
            system = self._system(stage, "scientist")
            if self.workflow.mode == "persistent" and self.memory_trims:
                system = ChatMessage(
                    "system",
                    system.content
                    + (
                        "\nSome older conversation turns were removed to fit the history budget. "
                        "The current complete research ledger and your latest notebook "
                        "remain supplied."
                    ),
                )
            messages = [system, *history, current]
            response = self._call(
                f"{key}-agent-a{attempt}",
                messages,
                session=session,
                authoritative=True,
                iteration=iteration,
                stage=stage,
                kind="linear",
            )
        else:
            peer_history = [[] for _ in range(self.workflow.agents_per_stage)]
            drafts = []
            for round_index in range(1, self.workflow.deliberation_rounds + 1):
                previous_round = drafts
                drafts = []
                for peer in range(self.workflow.agents_per_stage):
                    session = f"{key}-peer{peer + 1}"
                    others = [d for d in previous_round if d["peer"] != peer + 1]
                    feedback = ""
                    for repair in range(1, self.source.max_retries_per_stage + 2):
                        user = ChatMessage(
                            role="user", content=self.initial_prompts[key] + feedback
                        )
                        messages = [
                            self._system(stage, "independent peer", others or None),
                            *peer_history[peer],
                            user,
                        ]
                        try:
                            draft = self._call(
                                f"{session}-r{round_index}-a{repair}",
                                messages,
                                session=session,
                                authoritative=False,
                                iteration=iteration,
                                stage=stage,
                                kind="peer",
                            )
                            form = FORMS[stage].model_validate(json_response(draft.text))
                            break
                        except WorkflowInfrastructureError:
                            raise
                        except Exception as exc:
                            self.draft_errors[(session, round_index, repair)] = {
                                "iteration": iteration,
                                "stage": stage,
                                "peer": peer + 1,
                                "round": round_index,
                                "attempt": repair,
                                "error": f"{type(exc).__name__}: {exc}",
                                "retrying": repair <= self.source.max_retries_per_stage,
                            }
                            if repair > self.source.max_retries_per_stage:
                                raise ValueError(
                                    f"Peer {peer + 1} exhausted draft repairs"
                                ) from exc
                            feedback = (
                                "\nDraft format error: "
                                + str(exc)
                                + "\nReturn a corrected complete form. The ledger is unchanged."
                            )
                    drafts.append({"peer": peer + 1, "form": form.model_dump(by_alias=True)})
                    peer_history[peer].extend([user, ChatMessage("assistant", draft.text)])
            messages = [self._system(stage, "chair", drafts), current]
            response = self._call(
                f"{key}-chair-a{attempt}",
                messages,
                session=f"{key}-chair",
                authoritative=True,
                iteration=iteration,
                stage=stage,
                kind="chair",
            )
        self.pending = (current, response, key)
        return response

    def commit(self):
        current, response, key = self.pending
        if self.workflow.mode == "persistent":
            self.history.extend([current, ChatMessage("assistant", response.text)])
        self.committed_stages.append(key)
        self.pending = None

    def reject(self):
        self.pending = None

    def audit(self):
        usage = [r["result"]["usage"] for r in self.records]
        return {
            "version": COORDINATION_VERSION,
            "workflow": self.workflow.model_dump(),
            "memory_policy": "full committed conversation"
            if self.workflow.mode == "persistent" and self.source.persistent_history_chars is None
            else "bounded committed conversation plus complete current ledger and notebook"
            if self.workflow.mode == "persistent"
            else "shared ledger and notes; fresh conversation for each stage",
            "persistent_history_chars": self.source.persistent_history_chars,
            "memory_trims": self.memory_trims,
            "resource_comparison": "same scientific budget; native participant call counts",
            "agent_calls": len(self.records),
            "authoritative_candidates": sum(
                r["request"]["authoritative_candidate"] for r in self.records
            ),
            "committed_stages": self.committed_stages,
            "usage": {
                k: sum(u[k] for u in usage) if all(u[k] is not None for u in usage) else None
                for k in ("input_tokens", "output_tokens", "duration_seconds")
            },
            "usage_missing_calls": sum(
                u["input_tokens"] is None or u["output_tokens"] is None for u in usage
            ),
            "output_token_accounting": {
                "known_output_tokens": sum(u["output_tokens"] or 0 for u in usage),
                "missing_calls": sum(u["output_tokens"] is None for u in usage),
                "unaccounted_infrastructure_attempts": sum(
                    max(0, (u["infrastructure_attempts"] or 1) - 1) for u in usage
                ),
            },
            "provider_error_calls": sum(bool(r["result"]["error"]) for r in self.records),
            "draft_errors": list(self.draft_errors.values()),
            "provider_infrastructure_attempts": (
                sum(u["infrastructure_attempts"] for u in usage)
                if all(u["infrastructure_attempts"] is not None for u in usage)
                else None
            ),
            "participant_artifacts": [f"calls/{r['request']['slot']}.json" for r in self.records],
            "participant_artifact_sha256": {
                f"calls/{r['request']['slot']}.json": hashlib.sha256(
                    (self.calls_dir / f"{r['request']['slot']}.json").read_bytes()
                ).hexdigest()
                for r in self.records
            },
        }
