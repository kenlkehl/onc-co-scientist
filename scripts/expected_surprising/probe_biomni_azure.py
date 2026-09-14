"""Excluded native A1 execute/observe diagnostic; never executes code on the host."""

import argparse
import json
import secrets
import time
from pathlib import Path

from onc_co_scientist.external.config import load_spec
from onc_co_scientist.external.runner import BiomniRunner
from onc_co_scientist.external.transport import RunBroker, usage_summary, write_json


def run_probe(config, root):
    root.mkdir(parents=True, exist_ok=False)
    public, scratch = root / "public", root / "scratch"
    public.mkdir()
    scratch.mkdir()
    challenge = {
        "nonce": secrets.token_hex(24),
        "values": [secrets.randbelow(10000) for _ in range(7)],
    }
    write_json(public / "challenge.json", challenge)
    spec = load_spec(config).model_copy(
        update={
            "azure_native_protocol": "structured",
            "max_tokens": 8192,
            "max_requests": 8,
            "output_root": root,
        }
    )
    write_json(root / "spec.json", spec.model_dump(mode="json"))
    write_json(root / "status.json", {"status": "running", "started_at": time.time()})
    broker = RunBroker(spec, None, root, secrets.token_hex(32))
    config_path = root / "worker_config.json"
    write_json(
        config_path,
        {
            "broker": broker.start(),
            "secret": broker.secret,
            "model": spec.model,
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
            "llm_backend": spec.llm_backend,
            "reasoning_effort": spec.reasoning_effort,
            "request_timeout": spec.request_timeout,
            "tool_timeout": spec.tool_timeout,
            "run_id": "excluded_native_protocol_probe",
            "resume": False,
            "prompt": """Test the real persistent Python interpreter in two separate executions.
First execute Python to load /public/challenge.json into a variable named
probe_challenge and print the complete object. Yield and inspect the observation.
In a second execution, use the existing probe_challenge variable (without rereading
the input file) to write /work/receipt.json with its nonce and the integer sum of
its values, under keys nonce and total. Print the receipt and inspect the result.
Then finish with the exact observed nonce and total in your final answer.
The file contents are unknown until you actually execute code. This is a software
integration test; no biomedical research or benchmark_exchange calls are needed.""",
        },
    )
    try:
        code = BiomniRunner().run(spec, public, scratch, config_path, broker)
        if code or broker.fatal:
            raise RuntimeError(f"Native worker failed: exit={code}, fatal={broker.fatal}")
        receipt = json.loads((scratch / "receipt.json").read_text())
        expected = {"nonce": challenge["nonce"], "total": sum(challenge["values"])}
        events = [
            json.loads(line) for line in (scratch / "native_events.jsonl").read_text().splitlines()
        ]
        result = json.loads((scratch / "native_result.json").read_text())
        observed = any(
            m["role"] == "assistant"
            and m.get("content", "").startswith("<observation>")
            and challenge["nonce"] in m["content"]
            for r in broker.requests
            for m in r["request"]["messages"]
        )
        checks = {
            "receipt_matches_unseen_input": receipt == expected,
            "two_native_executions": sum(e["node"] == "execute" for e in events) >= 2,
            "observation_returned_to_model": observed,
            "final_uses_actual_observation": all(
                str(v) in result["final_text"] for v in expected.values()
            ),
            "usage_complete": usage_summary(broker.requests)["usage_complete"],
            "structured_command_used": any(
                r.get("native_protocol") == "structured" for r in broker.requests
            ),
            "retriever_remains_text": broker.requests[0].get("native_protocol") == "text",
        }
        if not all(checks.values()):
            raise RuntimeError(f"Native probe checks failed: {checks}")
        write_json(
            root / "status.json",
            {
                "status": "passed",
                "finished_at": time.time(),
                "checks": checks,
                "nodes": [e["node"] for e in events],
                "usage": usage_summary(broker.requests),
            },
        )
    except Exception as exc:
        write_json(
            root / "status.json",
            {
                "status": "failed",
                "finished_at": time.time(),
                "error": f"{type(exc).__name__}: {exc}",
                "usage": usage_summary(broker.requests),
            },
        )
        raise
    finally:
        broker.stop()
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run_probe(args.config, args.root.resolve())
