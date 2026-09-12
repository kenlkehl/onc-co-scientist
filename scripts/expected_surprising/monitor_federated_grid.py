"""Refresh a local progress page without launching or modifying scientific runs."""

import argparse
import fcntl
import json
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def receipt(path, cache, *, native=False):
    if path in cache:
        return cache[path]
    record = read(path, {})
    value = record if native else record.get("result")
    if not value:
        return Counter()
    usage = value.get("response", {}).get("usage", {}) if native else value.get("usage", {})
    tokens = usage.get("completion_tokens" if native else "output_tokens")
    counts = Counter(
        calls=1,
        tokens=tokens or 0,
        unknown=tokens is None,
        errors=(value.get("status") == "failed" if native else bool(value.get("error"))),
    )
    if not native or value.get("status") in {"completed", "failed"}:
        cache[path] = counts
    return counts


def refresh(root, cache=None):
    cache = {} if cache is None else cache
    policy = read(root / "release_policy.json", {})
    groups = defaultdict(Counter)
    for plan in read(root / "grid.json", []):
        key = (plan["model_profile"], plan["condition"], plan["site_count"])
        run = root / plan["condition"] / "runs" / plan["run_id"]
        result = read(run / "run.json", {})
        released = plan["model_profile"] in policy.get("released_models", [])
        state = result.get("status") or (
            "running" if run.exists() else "queued" if released else "held"
        )
        groups[key][state] += 1
        if run.exists():
            for path in (run / "calls").rglob("*.json"):
                groups[key].update(receipt(path, cache))
    native = read(root / "biomni" / "plan.json")
    for plan in native or read(root / "biomni_reserved_plan.json", []):
        key = ("biomni_native", plan["condition"], plan["sites"])
        run = root / "biomni" / plan.get("run_path", "not_started")
        result = read(run / "run.json", {})
        groups[key][result.get("status", "held")] += 1
        if run.exists():
            for path in run.rglob("llm/*.json"):
                groups[key].update(receipt(path, cache, native=True))
    totals = sum(groups.values(), Counter())
    stamp = datetime.now(UTC).isoformat()
    lines = [
        "# Live federated experiment progress",
        "",
        f"Updated **{stamp}**",
        "",
        "2 and 4 random sites · named/masked × expected/surprising · 10 repeats · 25 "
        "rounds/iterations.",
        "",
        f"**Running {totals['running']} · Queued {totals['queued']} · "
        f"Completed {totals['completed']} · Failed {totals['failed']} · Held {totals['held']}**",
        "",
        "Refreshes every 30 seconds. Biomni and all unreleased models stay held. "
        "Calls and observed output tokens include active runs, peers, central agents and retries. "
        "Missing token receipts are counted separately. Native reporting rounds and controller "
        "iterations are different clocks.",
        "",
        "| Model | View | Sites | Held | Queued | Active | Done | Failed | Calls | "
        "Errors | Output tokens | Unknown usage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (model, view, sites), c in sorted(groups.items()):
        values = [
            c[k]
            for k in (
                "held",
                "queued",
                "running",
                "completed",
                "failed",
                "calls",
                "errors",
                "tokens",
                "unknown",
            )
        ]
        lines.append(
            f"| {model} | {view} | {sites} | " + " | ".join(f"{v:,}" for v in values) + " |"
        )
    lines += [
        "",
        "[Grid documentation](README.md) · [Release policy](release_policy.json) · "
        "[Biomni setup](biomni/README.md)",
    ]
    temporary = root / "LIVE_PROGRESS.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(root / "LIVE_PROGRESS.md")
    temporary = root / "live_progress.json.tmp"
    temporary.write_text(
        json.dumps(
            dict(
                updated_at=stamp,
                totals=totals,
                groups=[
                    dict(model=m, condition=c, sites=n, **v) for (m, c, n), v in groups.items()
                ],
            ),
            indent=2,
        )
    )
    temporary.replace(root / "live_progress.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    lock = (args.root / "progress_monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cache = {}
    while True:
        refresh(args.root.resolve(), cache)
        if args.once:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
