import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from onc_co_scientist.harness.structured_runner import StructuredRunner, finalize_workspace


@pytest.mark.parametrize("tier", [None, "standard"])
def test_endpoint_tool_turns_and_immutable_submission(tmp_path, tier):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"dataset_id": "d", "max_iterations": 1, "model_id": "m", "harness_id": "h"})
    )
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            n = len(seen)
            if n == 1:
                msg = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {
                                "name": "execute_python",
                                "arguments": '{"code":"print(2+2)"}',
                            },
                        }
                    ],
                }
            elif n == 2:
                record = {
                    "index": 1,
                    "proposed_hypotheses": [
                        {
                            "id": "h1",
                            "text": "finding",
                            "finding": {
                                "outcome": "y",
                                "exposure": "x",
                                "contrast": "treatment_effect",
                                "direction": 1,
                                "subgroup": [],
                            },
                        }
                    ],
                    "analyses": [],
                }
                msg = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "y",
                            "type": "function",
                            "function": {
                                "name": "submit_iteration",
                                "arguments": json.dumps({"iteration": record}),
                            },
                        }
                    ],
                }
            else:
                msg = {"role": "assistant", "content": "done"}
            payload = json.dumps(
                {
                    "choices": [{"message": msg, "finish_reason": "tool_calls"}],
                    "usage": {"completion_tokens": 2},
                    "service_tier": "default",
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result = StructuredRunner(
            tmp_path,
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="local-served-model",
            max_turns=5,
            max_generated_tokens=20,
            service_tier=tier,
        ).run()
    finally:
        server.shutdown()
    assert [it.index for it in result.iterations] == [1]
    assert result.model_id == "local-served-model"
    assert seen[0]["model"] == "local-served-model"
    if tier is None:
        assert "service_tier" not in seen[0]
    else:
        assert seen[0]["service_tier"] == "default"
    assert "reasoning_effort" not in seen[0]
    assert seen[1]["max_completion_tokens"] < seen[0]["max_completion_tokens"]
    from jsonschema import Draft202012Validator

    schema = seen[0]["tools"][1]["function"]["parameters"]
    Draft202012Validator(schema).validate(
        {"iteration": result.iterations[0].model_dump(mode="json")}
    )
    assert "Authorization" not in (tmp_path / "runner_transcript.jsonl").read_text()
    assert "4" in (tmp_path / "runner_transcript.jsonl").read_text()
    finalized = finalize_workspace(tmp_path)
    assert finalized.iterations[0].index == 1
    assert finalized.model_id == "local-served-model"


@pytest.mark.parametrize("returned", [None, "priority", "fast"])
def test_standard_tier_requires_matching_response(tmp_path, monkeypatch, returned):
    import io

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps({"service_tier": returned}).encode()),
    )
    runner = StructuredRunner(
        tmp_path, base_url="http://localhost", model="m", service_tier="standard"
    )
    with pytest.raises(RuntimeError, match="verify requested Standard"):
        runner._request([])
    assert (tmp_path / "runner_transcript.jsonl").exists()


def test_unknown_tool_and_malformed_arguments_are_safe(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"dataset_id": "d", "max_iterations": 1}))
    (tmp_path / "iterations").mkdir()
    (tmp_path / "iterations" / "001.json").write_text(
        json.dumps({"index": 1, "proposed_hypotheses": []})
    )
    with pytest.raises(ValueError, match="event log"):
        finalize_workspace(tmp_path)


def test_receipt_detects_tamper_and_gaps(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"dataset_id": "d", "max_iterations": 3, "model_id": "actual"})
    )
    r = StructuredRunner(tmp_path, base_url="http://127.0.0.1", model="actual")
    prior = []
    assert "expected iteration 1" in r._submit({"index": 2, "proposed_hypotheses": []}, prior, 3)
    assert r._submit({"index": 1, "proposed_hypotheses": []}, prior, 3).startswith("Accepted")
    assert finalize_workspace(tmp_path).model_id == "actual"
    (tmp_path / "iterations/001.json").write_text(
        json.dumps({"index": 1, "proposed_hypotheses": [], "changed": True})
    )
    with pytest.raises(ValueError, match="integrity"):
        finalize_workspace(tmp_path)


def test_auth_error_not_retried(tmp_path, monkeypatch):
    from urllib.error import HTTPError

    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.invalid", 401, "unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    runner = StructuredRunner(tmp_path, base_url="https://example.invalid/v1", model="test")
    with pytest.raises(RuntimeError, match="401"):
        runner._request([])
    assert len(calls) == 1


def test_analysis_subprocess_does_not_receive_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-that-must-not-reach-code")
    runner = StructuredRunner(tmp_path, base_url="http://127.0.0.1", model="m")
    result = runner._python("import os; print(os.environ.get('OPENAI_API_KEY', 'absent'))")
    assert "absent" in result
    assert "secret-that" not in result
