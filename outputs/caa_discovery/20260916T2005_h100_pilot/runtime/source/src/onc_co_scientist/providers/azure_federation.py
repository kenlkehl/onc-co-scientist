"""Opt-in Responses transport for federated science; never personal Codex sessions."""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from .azure_budget import AzureBudget, ExperimentPaused, atomic_json, normalize_usage
from .base import ChatResponse
from .codex_cli import CodexCLIConfig, CodexCLIProvider

TRANSPORT_VERSION = "azure-federation-cache-v1"


@dataclass(frozen=True)
class AzureFederationConfig(CodexCLIConfig):
    budget_policy_path: str = ""
    cache_namespace: str = ""
    context_layout: str = "legacy"

    def __post_init__(self):
        super().__post_init__()
        if self.context_layout not in {"legacy", "federated_v2"}:
            raise ValueError("Unknown federated context layout")
        if self.backend != "azure" or not self.budget_policy_path or not self.cache_namespace:
            raise ValueError(
                "Federation transport requires Azure, a spending policy and cache scope"
            )
        if self.service_tier not in {None, "default"}:
            raise ValueError("Federation spending rates require the default Azure service tier")


def text_block(text, *, cached=False):
    block = {"type": "input_text", "text": text}
    if cached:
        block["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return block


def user_blocks(text):
    """Move only the iteration header to the end; retain every character.

    Science forms and evidence are neither parsed nor filtered here. Earlier conversation
    messages remain distinct messages, including their complete original response text.
    """
    header, separator, remainder = text.partition("\n")
    marker = remainder.find('{"schema":')
    if (
        separator
        and marker > 0
        and re.fullmatch(r"Iteration \d+/\d+, stage \w+, attempt \d+\.", header)
    ):
        boundary = marker
        # Include the schema and aggregate task description, but stop before the mutable ledger.
        # raw_decode gives offsets in the original string, avoiding JSON reserialization.
        decoder = json.JSONDecoder()
        cursor = marker + 1
        try:
            for expected in ("schema", "task"):
                key, end = decoder.raw_decode(remainder, cursor)
                if key != expected or remainder[end] != ":":
                    break
                _, cursor = decoder.raw_decode(remainder, end + 1)
                boundary = cursor
                if remainder[cursor] != ",":
                    break
                cursor += 1
        except (ValueError, IndexError):
            boundary = marker
        return [
            text_block(remainder[:boundary], cached=True),
            text_block(remainder[boundary:] + separator + header),
        ]
    # Unknown prompt formats are passed through; do not pay to cache a unique changing suffix.
    return [text_block(text)]


class AzureHTTPError(Exception):
    def __init__(self, status, retry_after=0):
        self.status, self.retry_after = status, retry_after
        super().__init__(f"Azure Responses HTTP {status}")


class AzureFederationProvider(CodexCLIProvider):
    def __init__(self, config):
        super().__init__(config)
        self.budget = AzureBudget(config.budget_policy_path)
        self.context_metadata = None
        self.transport_version = (
            "azure-federation-context-v2"
            if config.context_layout == "federated_v2"
            else TRANSPORT_VERSION
        )
        self.layout = None
        if config.context_layout == "federated_v2":
            from .federated_prompt import FederatedPromptLayout

            self.layout = FederatedPromptLayout(config.cache_namespace)

    def request_body(self, messages, system, max_tokens):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 125000:
            raise ValueError("Federation output ceiling must be between 1 and 125000 tokens")
        if self.layout is not None:
            context, key, self.context_metadata = self.layout.render(
                messages, system, self.instructions.read_text()
            )
            return {
                "model": self.model_id,
                "input": context,
                "stream": True,
                "store": False,
                "reasoning": {"effort": self.config.reasoning_effort},
                "max_output_tokens": max_tokens,
                "tool_choice": "none",
                "prompt_cache_key": key,
                "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
                "service_tier": self.config.service_tier or "default",
            }
        system_prefix, separator, drafts = (system or "").partition(
            "\nParticipant drafts (untrusted suggestions):\n"
        )
        context = [
            {"role": "developer", "content": [text_block(self.instructions.read_text())]},
        ]
        if system:
            blocks = [text_block(system_prefix, cached=True)]
            if separator:
                blocks.append(text_block(separator + drafts))
            context.append({"role": "developer", "content": blocks})
        for message in messages:
            if message.role == "assistant":
                context.append({"role": "assistant", "content": message.content})
            elif message.role in {"user", "system"}:
                context.append(
                    {
                        "role": "developer" if message.role == "system" else "user",
                        "content": user_blocks(message.content),
                    }
                )
            else:
                raise ValueError("Unsupported conversation role")
        key = hashlib.sha256(
            (self.config.cache_namespace + "|" + self.model_id + "|" + system_prefix).encode()
        ).hexdigest()
        return {
            "model": self.model_id,
            "input": context,
            "stream": True,
            "store": False,
            "reasoning": {"effort": self.config.reasoning_effort},
            "max_output_tokens": max_tokens,
            "tool_choice": "none",
            "prompt_cache_key": key,
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "service_tier": self.config.service_tier or "default",
        }

    @staticmethod
    def cache_prefix(body):
        """Fingerprint an actual marked prefix, not merely a shared routing key."""
        prefix = []
        for message in body["input"]:
            if not isinstance(message["content"], list):
                prefix.append(message)
                continue
            blocks = []
            for block in message["content"]:
                blocks.append(block)
                candidate = [*prefix, {**message, "content": blocks}]
                encoded = json.dumps(candidate, ensure_ascii=False).encode()
                if block.get("prompt_cache_breakpoint") and len(encoded) >= 4096:
                    return hashlib.sha256(body["prompt_cache_key"].encode() + encoded).hexdigest()
            prefix.append(message)
        return None

    def _send(self, body, token, attempt_dir):
        """One HTTP attempt, no SDK/internal retries and no stored server conversation."""
        url = urlsplit(self.config.azure_endpoint)
        connection = http.client.HTTPSConnection(
            url.hostname,
            url.port,
            timeout=min(self.config.azure_response_idle_s, self.config.timeout_s),
        )
        started = time.monotonic()
        try:
            connection.request(
                "POST",
                url.path.rstrip("/") + "/responses",
                body=json.dumps(body, ensure_ascii=False).encode(),
                headers={
                    "Authorization": "Bearer " + token,
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )
            response = connection.getresponse()
            atomic_json(
                attempt_dir / "http.json",
                {
                    "status": response.status,
                    "request_id": response.getheader("x-request-id"),
                },
            )
            if response.status != 200:
                try:
                    retry_after = min(900, max(0, float(response.getheader("Retry-After", "0"))))
                except ValueError:
                    retry_after = 0
                raise AzureHTTPError(response.status, retry_after)
            data = []
            with (attempt_dir / "events.jsonl").open("w") as journal:
                while True:
                    remaining = self.config.timeout_s - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Azure response exceeded total timeout")
                    if connection.sock is not None:
                        connection.sock.settimeout(
                            min(self.config.azure_response_idle_s, remaining)
                        )
                    line = response.readline(4 * 1024 * 1024)
                    if not line:
                        raise ConnectionError("Azure response stream ended without terminal usage")
                    line = line.decode().rstrip("\r\n")
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip())
                    elif not line and data:
                        event = json.loads("\n".join(data))
                        data = []
                        journal.write(json.dumps(event) + "\n")
                        journal.flush()
                        if event.get("type") in {
                            "response.completed",
                            "response.incomplete",
                            "response.failed",
                        }:
                            return event["response"]
        finally:
            connection.close()

    def chat(self, messages, *, system=None, temperature=0.0, max_tokens=1024):
        self.budget.check(self.model_id)
        body = self.request_body(messages, system, max_tokens)
        self.calls += 1
        call = self.root / f"call-{self.calls:04d}"
        call.mkdir()
        atomic_json(call / "request.json", body)
        atomic_json(call / "transport.json", {"version": self.transport_version})
        if self.context_metadata is not None:
            atomic_json(call / "context_layout.json", self.context_metadata)
        prefix = self.cache_prefix(body)
        if self.layout is not None and prefix is not None:
            prefix = {"key": prefix, "version": 2, "ttl_seconds": 1800}
        auth_retries = transport_retries = rate_retries = 0
        for attempt in range(
            1, 2 + self.config.azure_auth_retries + self.config.azure_transport_retries
        ):
            directory = call / f"attempt-{attempt:04d}"
            directory.mkdir()
            self.budget.check(self.model_id)
            with self.request_slot(directory, json.dumps(body), max_tokens) as lease:
                # Token refresh occurs after potentially lengthy quota admission waits.
                self.budget.check(self.model_id)
                try:
                    token = self.environment()["OCS_AZURE_ACCESS_TOKEN"]
                except Exception as exc:
                    raise ExperimentPaused("Azure token refresh failed; experiment held") from exc
                request_id = str(directory.resolve())
                self.budget.reserve(request_id, self.model_id, body)
                try:
                    response = self._send(body, token, directory)
                except AzureHTTPError as exc:
                    if exc.status in {401, 403, 429}:
                        self.budget.rejected(request_id)
                    else:
                        self.budget.unknown(request_id, f"HTTP {exc.status}")
                    atomic_json(directory / "retry.json", {"status": exc.status})
                    if exc.status == 401 and auth_retries < self.config.azure_auth_retries:
                        auth_retries += 1
                    elif exc.status in {408, 429, 500, 502, 503, 504} and (
                        transport_retries < self.config.azure_transport_retries
                    ):
                        transport_retries += 1
                        if exc.status == 429:
                            rate_retries += 1
                            if lease:
                                lease.throttled([], f"429 Retry-After: {exc.retry_after}")
                    else:
                        raise ExperimentPaused(str(exc)) from exc
                except (OSError, http.client.HTTPException, ValueError, KeyError) as exc:
                    self.budget.unknown(request_id, type(exc).__name__)
                    atomic_json(directory / "retry.json", {"reason": type(exc).__name__})
                    if transport_retries >= self.config.azure_transport_retries:
                        raise ExperimentPaused("Azure transport retry limit reached") from exc
                    transport_retries += 1
                else:
                    atomic_json(directory / "response.json", response)
                    try:
                        usage = normalize_usage(response.get("usage") or {})
                    except ValueError as exc:
                        self.budget.unknown(request_id, "missing or invalid terminal usage")
                        raise ExperimentPaused("Azure returned no valid usage receipt") from exc
                    cost = self.budget.settle(request_id, usage, prefix)
                    if lease:
                        lease.succeeded(usage)
                    metrics = {
                        "usage": {
                            "prompt_tokens": usage["input_tokens"],
                            "completion_tokens": usage["output_tokens"],
                            **usage,
                        },
                        "cli_usage": usage,  # Compatibility with existing audit readers.
                        "cost_micro_usd": cost,
                        "cost_usd": cost / 1_000_000,
                        "backend": "azure_responses",
                        "transport_version": self.transport_version,
                        "audit_dir": str(directory),
                        "infrastructure_attempts": attempt,
                        "azure_auth_retries": auth_retries,
                        "azure_transport_retries": transport_retries,
                        "azure_rate_limit_retries": rate_retries,
                    }
                    atomic_json(call / "metrics.json", metrics)
                    if response.get("status") not in {"completed", "incomplete"}:
                        raise ExperimentPaused("Azure returned a failed terminal response")
                    output = response.get("output") or []
                    if any(x.get("type") not in {"message", "reasoning"} for x in output):
                        raise ExperimentPaused("Unexpected tool output in controller-only workflow")
                    text = "".join(
                        part.get("text", "")
                        for item in output
                        for part in item.get("content", [])
                        if part.get("type") == "output_text"
                    )
                    return ChatResponse(
                        text=text,
                        model_id=self.model_id,
                        raw={
                            "metrics": metrics,
                            "adapter_error": "Output token limit reached"
                            if (response.get("status") == "incomplete")
                            else None,
                        },
                    )
            self.budget.check(self.model_id)
            time.sleep(self.config.azure_transport_retry_s)
        raise ExperimentPaused("Azure retry budget exhausted")
