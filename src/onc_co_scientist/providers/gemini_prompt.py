"""Fresh, tool-free Gemini calls for the controlled groupthink experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from .gemini_vertex import GeminiVertexClient, GeminiVertexConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--project-id")
    p.add_argument("--location")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    p.add_argument("--schema", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=240)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("prompt")
    a = p.parse_args()
    schema = json.loads(a.schema.read_text())
    client = GeminiVertexClient(
        GeminiVertexConfig(
            model_id=a.model,
            project_id=a.project_id,
            location=a.location,
            timeout_s=a.timeout,
            reasoning_effort=a.reasoning_effort,
            max_retries=0,
        )
    )
    result = client.complete(
        model=a.model,
        messages=[{"role": "user", "content": a.prompt}],
        max_tokens=a.max_tokens,
        response_format={"type": "json_schema", "json_schema": {"schema": schema}},
    )
    choice = result["choices"][0]
    if choice["finish_reason"] == "length":
        raise RuntimeError("Gemini report truncated")
    payload = json.loads(choice["message"]["content"])
    Draft202012Validator(schema).validate(payload)
    a.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "type": "turn.completed",
                "provider": "gemini-vertex",
                "model": result["model"],
                "project_id": client.project_id,
                "location": client.location,
                "usage": {
                    "input_tokens": result["usage"]["prompt_tokens"],
                    "output_tokens": result["usage"]["completion_tokens"],
                },
            }
        )
    )


if __name__ == "__main__":
    main()
