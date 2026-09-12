"""Codex CLI transport for the structured, controller-executed research workflow."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
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

    def __post_init__(self):
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
            except (OSError, subprocess.TimeoutExpired):
                raise RuntimeError(
                    "Azure token refresh failed; no personal-account fallback"
                ) from None
            token = result.stdout.strip()
            if result.returncode or not token or any(c.isspace() for c in token):
                # Never include token command output in errors or audit files.
                raise RuntimeError("Azure token refresh failed; no personal-account fallback")
            env["OCS_AZURE_ACCESS_TOKEN"] = token
        return env

    def chat(self, messages: list[ChatMessage], *, temperature=0.0, max_tokens=125000, system=None):
        self.calls += 1
        call = self.root / f"call-{self.calls:04d}"
        call.mkdir()
        prompt = "\n\n".join(([system] if system else []) + [m.content for m in messages])
        (call / "prompt.txt").write_text(prompt)
        attempt = 0
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
                raise RuntimeError(f"Codex exited {process.returncode}: {(errors or raw)[-1600:]}")
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
                "recovered_error_count": sum(e.get("type") == "error" for e in events),
                "tool_items": 0,
                "backend": self.config.backend,
                "azure_endpoint": self.config.azure_endpoint,
            }
            (call / "metrics.json").write_text(json.dumps(metrics, indent=2))
            (self.root / "quota_wait.json").unlink(missing_ok=True)
            return ChatResponse(text=response, model_id=self.model_id, raw={"metrics": metrics})
