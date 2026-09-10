"""Audited local model proxy and framework-neutral benchmark exchange transport."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def http_json(url, data=None, *, timeout=30, key="EMPTY"):
    request = urllib.request.Request(
        url,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def without_reasoning(message):
    """Keep final text/tool calls; reasoning belongs only in raw response audits."""
    message = copy.deepcopy(message)
    for key in ("reasoning", "reasoning_content", "thinking"):
        message.pop(key, None)
    content = message.get("content")
    if isinstance(content, list):
        content = [
            block
            for block in content
            if block.get("type") not in {"reasoning", "thinking", "redacted_thinking"}
        ]
        message["content"] = content
    text = content if isinstance(content, str) else json.dumps(content)
    if "<think>" in text.lower() or "</think>" in text.lower():
        raise ValueError("Reasoning must be separated from final content")
    return message


def model_request(spec, body):
    body = dict(body)
    if body.get("stream"):
        raise ValueError("This audited adapter requires non-streamed responses")
    body.pop("max_completion_tokens", None)
    body["messages"] = [
        without_reasoning(m) if m.get("role") == "assistant" else copy.deepcopy(m)
        for m in body["messages"]
    ]
    body.update(
        model=spec.model,
        max_tokens=spec.max_tokens,
        stream=False,
        reasoning_effort=spec.reasoning_effort,
    )
    body.setdefault("temperature", spec.temperature)
    body["chat_template_kwargs"] = {
        **body.get("chat_template_kwargs", {}),
        "enable_thinking": True,
        "reasoning_effort": spec.reasoning_effort,
    }
    return body


def usage_summary(records):
    known = [r.get("response", {}).get("usage") or {} for r in records]
    details = [u.get("completion_tokens_details") or {} for u in known]
    complete = all("prompt_tokens" in u and "completion_tokens" in u for u in known)
    return {
        "requests": len(records),
        "usage_complete": complete,
        "input_tokens": sum(u.get("prompt_tokens", 0) for u in known) if complete else None,
        "output_tokens": sum(u.get("completion_tokens", 0) for u in known) if complete else None,
        "known_input_tokens": sum(u.get("prompt_tokens", 0) for u in known),
        "known_output_tokens": sum(u.get("completion_tokens", 0) for u in known),
        "reasoning_tokens": sum(d["reasoning_tokens"] for d in details)
        if details and all(d.get("reasoning_tokens") is not None for d in details)
        else None,
        "unknown_usage_requests": sum(
            not ("prompt_tokens" in u and "completion_tokens" in u) for u in known
        ),
        "cost_usd": None,
        "duration_seconds": sum(r.get("duration_seconds", 0) for r in records),
    }


class RunBroker:
    def __init__(self, spec, gateway, root, secret):
        self.spec, self.gateway, self.root, self.secret = spec, gateway, Path(root), secret
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.requests = [
            json.loads(p.read_text()) for p in sorted((self.root / "llm").glob("*.json"))
        ]
        self.exchange_count = 0
        self.fatal = None
        self.key = os.environ[spec.api_key_env] if spec.api_key_env else "EMPTY"

    def exchange(self, body):
        with self.lock:
            response = self.gateway.exchange(body)
            self.exchange_count += 1
            append_json(
                self.root / "exchanges.jsonl",
                {
                    "request": body,
                    "response": response,
                    "llm_requests": len(self.requests),
                    "time": time.time(),
                },
            )
            return response

    def complete(self, body):
        with self.lock:
            if len(self.requests) >= self.spec.max_requests:
                self.fatal = "request_budget_exhausted"
                raise ValueError(self.fatal)
            body = model_request(self.spec, body)
            index = len(self.requests) + 1
            path = self.root / "llm" / f"{index:06d}.json"
            record = {
                "index": index,
                "request": body,
                "started_at": time.time(),
                "status": "inflight",
            }
            self.requests.append(record)
            write_json(path, record)
        start = time.monotonic()
        try:
            base = self.spec.base_url.removesuffix("/v1").rstrip("/")
            count = http_json(
                base + "/tokenize",
                {
                    k: v
                    for k, v in {
                        "model": self.spec.model,
                        "messages": body["messages"],
                        "tools": body.get("tools"),
                        "add_generation_prompt": True,
                        "chat_template_kwargs": body["chat_template_kwargs"],
                    }.items()
                    if v is not None
                },
                key=self.key,
            )
            available = self.spec.context_length - count["count"]
            record.update(
                prompt_token_count=count["count"],
                completion_policy=self.spec.completion_policy,
                requested_max_tokens=self.spec.max_tokens,
            )
            if self.spec.completion_policy == "adaptive":
                body["max_tokens"] = min(
                    self.spec.max_tokens, available - self.spec.context_guard_tokens
                )
                exhausted = body["max_tokens"] < self.spec.min_completion_tokens
            else:
                exhausted = available < self.spec.max_tokens
            if exhausted:
                self.fatal = "context_exhausted"
                raise ValueError("Required completion allowance exceeds configured context")
            record["effective_max_tokens"] = body["max_tokens"]
            write_json(path, record)
            result = http_json(
                self.spec.base_url.rstrip("/") + "/chat/completions",
                body,
                timeout=self.spec.request_timeout,
                key=self.key,
            )
            record.update(response=result, prompt_token_count=count["count"], status="completed")
            choice = result.get("choices", [{}])[0]
            if choice.get("finish_reason") == "length":
                self.fatal = "completion_length_exhausted"
                raise ValueError("Length-truncated output cannot be executed as complete code")
            # Do not mutate the raw response/usage retained in the audit record.
            clean_result = copy.deepcopy(result)
            try:
                for item in clean_result.get("choices", []):
                    item["message"] = without_reasoning(item.get("message", {}))
            except ValueError:
                self.fatal = "reasoning_not_separated"
                raise
            return clean_result
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["duration_seconds"] = time.monotonic() - start
            write_json(path, record)

    def start(self):
        broker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.headers.get("Authorization") != f"Bearer {broker.secret}":
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if not 0 < length <= 20_000_000:
                        raise ValueError("Invalid request size")
                    body = json.loads(self.rfile.read(length))
                    if self.path == "/exchange":
                        result = broker.exchange(body)
                    elif self.path == "/v1/chat/completions":
                        result = broker.complete(body)
                    elif self.path == "/checkpoint":
                        result = {
                            "llm_requests": len(broker.requests),
                            "exchange_count": broker.exchange_count,
                        }
                    else:
                        raise ValueError("Unsupported broker route")
                    status = 200
                except Exception as exc:
                    status, result = (
                        400,
                        {"error": {"message": str(exc), "type": type(exc).__name__}},
                    )
                    append_json(
                        broker.root / "errors.jsonl",
                        {"path": self.path, "error": result, "time": time.time()},
                    )
                raw = json.dumps(result).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # Drain bounded upstream requests before freezing final usage/artifact
        # hashes, including helper work still active when the worker terminates.
        self.server.daemon_threads = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
