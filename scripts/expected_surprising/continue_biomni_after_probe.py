"""Admit fresh native pilots only after an excluded wire-format diagnostic succeeds."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def validate_probe(response):
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Diagnostic did not return a complete native response")
    message = choices[0].get("message", {})
    content = message.get("content") or ""
    blocks = re.findall(r"<execute>(.*?)</execute>", content, re.DOTALL)
    if len(blocks) != 1 or not blocks[0].strip() or "<solution>" in content:
        raise ValueError("Diagnostic did not return exactly one native execution command")
    if message.get("tool_calls") or message.get("refusal"):
        raise ValueError("Diagnostic returned an incompatible API action")


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def wait_and_run(root, probe, pid):
    try:
        while True:
            state = json.loads((probe / "status.json").read_text())
            if state["status"] == "completed":
                raw = json.loads((probe / "response.json").read_text())
                validate_probe(raw)
                write(
                    root / "transport_diagnostic.json",
                    {
                        "status": "passed",
                        "excluded": True,
                        "source": str(probe),
                        "test": "unchanged native pilot prompt returns one complete execute block",
                        "usage": raw.get("usage"),
                    },
                )
                break
            if state["status"] == "failed":
                raise RuntimeError("Excluded Chat Completions diagnostic failed")
            os.kill(pid, 0)
            phase = (
                "awaiting_azure_slot" if state["status"] == "waiting_quota" else "transport_probe"
            )
            write(
                root / "execution.json",
                {
                    "status": phase,
                    "planned_runs": 100,
                    "formal_runs_started": 0,
                    "updated_at": time.time(),
                },
            )
            text = (
                "# Biomni / Luna — native Chat Completions validation\n\n"
                f"Status: **{phase}**. Updated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}.\n\n"
                "All 76 Biomni resources are installed. The excluded API-format diagnostic "
                "must return a native execution command before four fresh integration "
                "pilots start. "
                "All four pilots must pass before admitting the 100 formal runs.\n\n"
                "Formal runs started: **0 / 100**.\n"
            )
            temp = root / "STATUS.tmp"
            temp.write_text(text)
            temp.replace(root / "STATUS.md")
            time.sleep(30)
        env = {**os.environ, "PYTHONPATH": str(root / "source/src"), "PYTHONDONTWRITEBYTECODE": "1"}
        os.execve(
            sys.executable,
            [
                sys.executable,
                str(root / "source/run_biomni_azure.py"),
                "run",
                "--root",
                str(root),
                "--workers",
                "4",
            ],
            env,
        )
    except BaseException as exc:
        write(
            root / "execution.json",
            {
                "status": "stopped_before_pilots",
                "planned_runs": 100,
                "formal_runs_started": 0,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        (root / "STATUS.md").write_text(
            "# Biomni / Luna\n\n**Stopped before revised pilots.**\n\n"
            f"{type(exc).__name__}: {exc}\n\nAll 100 formal runs remain unstarted.\n"
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    wait_and_run(args.root.resolve(), args.probe.resolve(), args.pid)
