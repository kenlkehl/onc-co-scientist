"""Codex CLI transport for the structured, controller-executed research workflow."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .base import ChatMessage, ChatResponse

DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "code_mode",
    "code_mode_host",
    "code_mode_only",
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "image_generation",
    "in_app_browser",
    "plugins",
    "remote_plugin",
    "hooks",
    "multi_agent",
    "multi_agent_v2",
    "skill_search",
    "workspace_dependencies",
    "memories",
    "view_image",
    "goals",
    "sleep_tool",
)


@dataclass(frozen=True)
class CodexCLIConfig:
    model_id: str
    reasoning_effort: str = "medium"
    service_tier: str | None = None
    executable: str = "/home/klkehl/.local/bin/codex"
    audit_dir: str = "/tmp/ocs-codex-provider"
    timeout_s: float = 1800
    usage_retry_s: float = 900
    resume_audit: bool = False
    backend: str = "chatgpt"
    azure_endpoint: str | None = None
    azure_cli_executable: str = "/usr/bin/az"
    azure_auth_retries: int = 3
    azure_auth_retry_s: float = 2.0
    azure_pacing_dir: str = "/tmp/ocs-azure-pacing"
    azure_tokens_per_minute: int = 333000
    azure_output_reserve_tokens: int = 8192
    azure_request_interval_s: float = 10.0
    azure_rate_limit_base_s: float = 60.0
    azure_rate_limit_cap_s: float = 900.0

    def __post_init__(self):
        if (
            type(self.azure_tokens_per_minute) is not int
            or self.azure_tokens_per_minute <= 0
            or type(self.azure_output_reserve_tokens) is not int
            or self.azure_output_reserve_tokens <= 0
            or any(
                not math.isfinite(v) or v < 0
                for v in (
                    self.azure_request_interval_s,
                    self.azure_rate_limit_base_s,
                    self.azure_rate_limit_cap_s,
                )
            )
            or self.azure_rate_limit_cap_s < self.azure_rate_limit_base_s
        ):
            raise ValueError("Azure pacing requires a positive quota and finite nonnegative delays")
        if (
            type(self.azure_auth_retries) is not int
            or not 0 <= self.azure_auth_retries <= 10
            or not math.isfinite(self.azure_auth_retry_s)
            or not 0 <= self.azure_auth_retry_s <= 60
        ):
            raise ValueError("Azure auth retries require 0–10 retries and a finite 0–60s delay")
        if self.backend not in {"chatgpt", "azure"}:
            raise ValueError("Codex backend must be chatgpt or azure")
        if self.backend == "azure":
            url = urlsplit(self.azure_endpoint or "")
            if (
                url.scheme != "https"
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
                or url.path.rstrip("/") != "/openai/v1"
            ):
                raise ValueError("Azure requires an HTTPS /openai/v1 endpoint without credentials")
        elif self.azure_endpoint is not None:
            raise ValueError("azure_endpoint requires backend=azure")


class AzureRequestLease:
    """Mutable admission state protected by the deployment's separate flock inode."""

    def __init__(self, config, path, attempt_dir):
        self.config, self.path, self.attempt_dir = config, path, attempt_dir
        self.state = json.loads(path.read_text()) if path.exists() else {}

    def save(self):
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.path)

    def admit(self, estimated_tokens):
        budget = max(1, int(self.config.azure_tokens_per_minute * 0.9))
        reservation = min(estimated_tokens, budget)
        while True:
            now = time.time()
            recent = [r for r in self.state.get("reservations", []) if r["at"] + 60 > now]
            delay = max(
                0,
                self.state.get("cooldown_until", 0) - now,
                self.state.get("next_request_at", 0) - now,
            )
            if sum(r["tokens"] for r in recent) + reservation > budget:
                delay = max(delay, min(r["at"] + 60 for r in recent) - now)
            if delay <= 0:
                self.state.update(
                    reservations=[*recent, dict(at=now, tokens=reservation)],
                    next_request_at=now + self.config.azure_request_interval_s,
                )
                self.save()
                (self.attempt_dir / "pacing.json").write_text(
                    json.dumps(
                        dict(
                            admitted_at=now,
                            estimated_tokens=estimated_tokens,
                            reserved_tokens=reservation,
                            tokens_per_minute=self.config.azure_tokens_per_minute,
                            single_request_over_estimated_budget=estimated_tokens > budget,
                            state_path=str(self.path),
                        ),
                        indent=2,
                    )
                )
                return
            (self.attempt_dir / "pacing_wait.json").write_text(
                json.dumps(
                    dict(
                        updated_at=now,
                        wait_s=delay,
                        reason="shared Azure quota admission",
                    )
                )
            )
            time.sleep(min(delay, 60))

    def throttled(self, events, stderr):
        text = (
            json.dumps([e for e in events if e.get("type") in {"error", "turn.failed"}])
            + "\n"
            + stderr
        ).lower()
        count = self.state.get("consecutive_rate_limits", 0) + 1
        delay = min(
            self.config.azure_rate_limit_cap_s,
            self.config.azure_rate_limit_base_s * 2 ** min(count - 1, 20),
        )
        delay = min(self.config.azure_rate_limit_cap_s, delay + random.uniform(0, delay * 0.1))
        hints = [float(v) for v in re.findall(r"retry[ -]after[\s:=]+(\d+(?:\.\d+)?)", text)]
        hints += [float(v) / 1000 for v in re.findall(r"retry-after-ms[\s:=]+(\d+)", text)]
        delay = max([delay, *hints])
        self.state.update(
            consecutive_rate_limits=count,
            cooldown_until=max(self.state.get("cooldown_until", 0), time.time() + delay),
        )
        self.save()
        (self.attempt_dir / "rate_limit_retry.json").write_text(
            json.dumps(
                dict(
                    reason="azure_rate_limit",
                    shared_cooldown_until=self.state["cooldown_until"],
                    cooldown_s=delay,
                    consecutive_rate_limits=count,
                    scientific_stage_retry_consumed=False,
                ),
                indent=2,
            )
        )

    def output_reservation(self, max_tokens):
        # Admission estimate only: never changes the scientific output ceiling.
        observed = self.state.get("recent_output_tokens", [])
        learned = math.ceil(max(observed, default=0) * 1.5) + 1024
        return min(max_tokens, max(self.config.azure_output_reserve_tokens, learned))

    def succeeded(self, usage=None):
        self.state["consecutive_rate_limits"] = 0
        output = (usage or {}).get("output_tokens")
        if type(output) is int and output >= 0:
            self.state["recent_output_tokens"] = [
                *self.state.get("recent_output_tokens", []),
                output,
            ][-20:]
        self.save()


class CodexCLIProvider:
    """Use fresh CLI sessions; the controller supplies the public research context."""

    _quota_lock = threading.Lock()
    _quota_until = 0.0

    def __init__(self, config: CodexCLIConfig):
        self.config = config
        self.root = Path(config.audit_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=config.resume_audit)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir(exist_ok=config.resume_audit)
        self.instructions = self.root / "instructions.txt"
        instructions = (
            "You are participating in a structured research workflow. Follow the supplied task "
            "and return its requested JSON record. All evidence is supplied in the user message; "
            "analyses are executed by an external controller. "
            "No external tools are part of this task."
        )
        if self.instructions.exists():
            if self.instructions.read_text() != instructions:
                raise ValueError("Cannot resume Codex audit with changed instructions")
        else:
            self.instructions.write_text(instructions)
        # Stage journals replay completed responses; any new calls append without
        # overwriting completed or interrupted transport attempts.
        self.calls = max(
            (int(p.name[5:]) for p in self.root.glob("call-*") if p.name[5:].isdigit()),
            default=0,
        )

    @property
    def model_id(self):
        return self.config.model_id

    def command(self, output: Path, max_tokens: int):
        permissions = '{filesystem={":root"="deny",":minimal"="read"},network={enabled=false}}'
        settings = {
            "model_reasoning_effort": self.config.reasoning_effort,
            "approval_policy": "never",
            "allow_login_shell": False,
            "web_search": "disabled",
            "project_doc_max_bytes": 0,
            "model_instructions_file": str(self.instructions),
            "include_apps_instructions": False,
            "include_collaboration_mode_instructions": False,
            "skills.include_instructions": False,
            "skills.bundled.enabled": False,
            "features.skip_host_skill_discovery": True,
            "tools.update_plan.enabled": False,
        }
        if self.config.backend == "chatgpt":
            settings["forced_login_method"] = "chatgpt"
        else:
            settings.update(
                {
                    "model_provider": "ocs_azure",
                    "model_providers.ocs_azure.name": "Azure experiment transport",
                    "model_providers.ocs_azure.base_url": self.config.azure_endpoint,
                    "model_providers.ocs_azure.wire_api": "responses",
                    "model_providers.ocs_azure.env_key": "OCS_AZURE_ACCESS_TOKEN",
                    "model_providers.ocs_azure.requires_openai_auth": False,
                }
            )
        if self.config.service_tier is not None:
            settings["service_tier"] = self.config.service_tier
        command = [
            self.config.executable,
            "exec",
            "--model",
            self.model_id,
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "--skip-git-repo-check",
            "--json",
            "--color",
            "never",
            "-C",
            str(self.workspace),
            "--output-last-message",
            str(output),
        ]
        for key, value in settings.items():
            command += ["-c", f"{key}={json.dumps(value)}"]
        command += [
            "-c",
            f"permissions.benchmark={permissions}",
            "-c",
            'default_permissions="benchmark"',
        ]
        command += [
            "-c",
            "features.rollout_budget={enabled=true,limit_tokens="
            + str(max_tokens)
            + ",prefill_token_weight=0,sampling_token_weight=1,reminder_at_remaining_tokens=[]}",
        ]
        for feature in DISABLED_FEATURES:
            command += ["--disable", feature]
        return [*command, "-"]

    def environment(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"OPENAI_API_KEY", "CODEX_API_KEY", "OCS_AZURE_ACCESS_TOKEN"}
        }
        if self.config.backend == "azure":
            # .bashrc uses this Entra flow. Refresh for every fresh CLI attempt:
            # inheriting its initial token would fail during a multi-day grid.
            for retry in range(self.config.azure_auth_retries + 1):
                try:
                    result = subprocess.run(
                        [
                            self.config.azure_cli_executable,
                            "account",
                            "get-access-token",
                            "--resource=https://cognitiveservices.azure.com/",
                            "--query",
                            "accessToken",
                            "--output",
                            "tsv",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    token = result.stdout.strip()
                    if result.returncode == 0 and token and not any(c.isspace() for c in token):
                        break
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if retry == self.config.azure_auth_retries:
                    # Never include token command output in errors or audit files.
                    raise RuntimeError("Azure token refresh failed; no personal-account fallback")
                time.sleep(self.auth_retry_delay(retry))
            env["OCS_AZURE_ACCESS_TOKEN"] = token
        return env

    def auth_retry_delay(self, retry):
        return min(60.0, self.config.azure_auth_retry_s * 2**retry)

    @contextmanager
    def request_slot(self, attempt_dir, prompt, max_tokens):
        """One request in flight per deployment, shared across threads AND processes."""
        if self.config.backend != "azure":
            yield None
            return
        root = Path(self.config.azure_pacing_dir)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = hashlib.sha256(
            f"{self.config.azure_endpoint.rstrip('/')}|{self.model_id}".encode()
        ).hexdigest()
        with (root / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            lease = AzureRequestLease(self.config, root / f"{key}.json", attempt_dir)
            output_reserve = lease.output_reservation(max_tokens)
            estimate = math.ceil(len(prompt.encode()) / 3) + output_reserve + 8000
            lease.admit(estimate)
            (attempt_dir / "reservation_estimate.json").write_text(
                json.dumps(
                    dict(
                        output_reserve_tokens=output_reserve,
                        scientific_output_ceiling=max_tokens,
                        input_estimate_tokens=math.ceil(len(prompt.encode()) / 3),
                        cli_overhead_tokens=8000,
                    ),
                    indent=2,
                )
            )
            yield lease

    @staticmethod
    def azure_rate_limit_failure(events, stderr):
        errors = [e for e in events if e.get("type") in {"error", "turn.failed"}]
        text = (json.dumps(errors) + "\n" + stderr).lower()
        if "insufficient_quota" in text or "billing_hard_limit" in text:
            return False
        return bool(re.search(r"\b429\b|rate[_ -]?limit(?:[_ -]exceeded|ed)?", text))

    @staticmethod
    def azure_auth_failure(events, stderr):
        # Classify error records only: quoted HTTP errors in a successful model
        # response are task content, not evidence that authentication failed.
        errors = [e for e in events if e.get("type") in {"error", "turn.failed"}]
        text = (json.dumps(errors) + "\n" + stderr).lower()
        return bool(
            re.search(
                r"\b401\b|\binvalid_authentication_token\b|\btoken_expired\b|"
                r"\bexpired(?:\s+access)?\s+token\b|"
                r"\b(?:access\s+)?token\s+(?:has\s+|is\s+)?expired\b|"
                r"\bidx10223\b",
                text,
            )
        )

    def chat(self, messages: list[ChatMessage], *, temperature=0.0, max_tokens=125000, system=None):
        self.calls += 1
        call = self.root / f"call-{self.calls:04d}"
        call.mkdir()
        prompt = "\n\n".join(([system] if system else []) + [m.content for m in messages])
        (call / "prompt.txt").write_text(prompt)
        attempt = 0
        auth_retries = 0
        rate_limit_retries = 0
        while True:
            with self._quota_lock:
                delay = max(0, self._quota_until - time.time())
            if delay:
                (self.root / "quota_wait.json").write_text(
                    json.dumps(
                        {
                            "resume_after_unix": time.time() + delay,
                            "reason": "Account usage limit; retrying the same stage later.",
                        }
                    )
                )
                time.sleep(delay)
            attempt += 1
            attempt_dir = call / f"attempt-{attempt:04d}"
            attempt_dir.mkdir()
            final_path = attempt_dir / "final.txt"
            command = self.command(final_path, max_tokens)
            (attempt_dir / "command.json").write_text(json.dumps(command, indent=2))
            with self.request_slot(attempt_dir, prompt, max_tokens) as lease:
                env = self.environment()
                with (
                    (attempt_dir / "events.jsonl").open("w") as stdout,
                    (attempt_dir / "stderr.log").open("w") as stderr,
                ):
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        cwd=self.workspace,
                        env=env,
                        start_new_session=True,
                    )
                    try:
                        process.communicate(prompt, timeout=self.config.timeout_s)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
                        raise TimeoutError(
                            f"Codex exceeded {self.config.timeout_s}s; see {attempt_dir}"
                        ) from None
                raw = (attempt_dir / "events.jsonl").read_text()
                errors = (attempt_dir / "stderr.log").read_text()
                events = [json.loads(line) for line in raw.splitlines() if line.strip()]
                # Reconnect warnings are top-level errors too. Resolve the final
                # turn state instead of discarding a later successful completion.
                terminal = next(
                    (
                        event
                        for event in reversed(events)
                        if event.get("type")
                        in {"turn.started", "turn.completed", "turn.failed", "error"}
                    ),
                    {},
                )
                completed = terminal.get("type") == "turn.completed" and process.returncode == 0
                if (
                    not completed
                    and self.config.backend == "azure"
                    and self.azure_rate_limit_failure(events, errors)
                ):
                    rate_limit_retries += 1
                    lease.throttled(events, errors)
                    continue  # No scientific error journal or stage retry is consumed.
                if (
                    not completed
                    and self.config.backend == "azure"
                    and self.azure_auth_failure(events, errors)
                ):
                    exhausted = auth_retries >= self.config.azure_auth_retries
                    delay = 0 if exhausted else self.auth_retry_delay(auth_retries)
                    (attempt_dir / "auth_retry.json").write_text(
                        json.dumps(
                            {
                                "reason": "azure_authentication_rejected",
                                "retry": not exhausted,
                                "retry_delay_s": delay,
                                "next_attempt_refreshes_token": not exhausted,
                            },
                            indent=2,
                        )
                    )
                    if exhausted:
                        raise RuntimeError(
                            f"Azure authentication failed after {auth_retries} retries; "
                            "check Azure login/access; no personal-account fallback; "
                            f"see {attempt_dir}"
                        )
                    auth_retries += 1
                    time.sleep(delay)
                    continue
                failure_text = (raw + "\n" + errors).lower()
                if not completed and any(
                    s in failure_text
                    for s in (
                        "usage_limit_reached",
                        "you've hit your usage limit",
                        "usage limit exceeded",
                    )
                ):
                    with self._quota_lock:
                        type(self)._quota_until = time.time() + self.config.usage_retry_s
                    continue
                if process.returncode:
                    raise RuntimeError(
                        f"Codex exited {process.returncode}: {(errors or raw)[-1600:]}"
                    )
                if not completed:
                    raise RuntimeError(f"Codex turn failed; see {attempt_dir}")
                tool_items = [
                    event
                    for event in events
                    if event.get("item", {}).get("type")
                    in {"command_execution", "mcp_tool_call", "web_search", "file_change"}
                ]
                if tool_items:
                    raise RuntimeError(
                        f"Unexpected tool use in controller-only task; see {attempt_dir}"
                    )
                usage = terminal.get("usage") or {}
                if lease is not None:
                    lease.succeeded(usage)
                if not final_path.exists():
                    raise RuntimeError(f"Codex returned no final message; see {attempt_dir}")
                response = final_path.read_text()
                metrics = {
                    "finish_reason": "stop",
                    "response_chars": len(response),
                    "usage": {
                        "prompt_tokens": usage.get("input_tokens"),
                        "completion_tokens": usage.get("output_tokens"),
                        "cached_input_tokens": usage.get("cached_input_tokens", 0),
                    },
                    "reasoning_effort": self.config.reasoning_effort,
                    "service_tier_requested": self.config.service_tier,
                    "temperature_requested": temperature,
                    "temperature_control": "CLI default",
                    "token_budget_control": "Codex sampled-token rollout budget; no prefill charge",
                    "cli_usage": usage,
                    "audit_dir": str(attempt_dir),
                    "infrastructure_attempts": attempt,
                    "azure_auth_retries": auth_retries,
                    "azure_rate_limit_retries": rate_limit_retries,
                    "recovered_error_count": sum(e.get("type") == "error" for e in events),
                    "tool_items": 0,
                    "backend": self.config.backend,
                    "azure_endpoint": self.config.azure_endpoint,
                }
                (call / "metrics.json").write_text(json.dumps(metrics, indent=2))
                (self.root / "quota_wait.json").unlink(missing_ok=True)
                return ChatResponse(text=response, model_id=self.model_id, raw={"metrics": metrics})
