"""Refresh a local progress page without launching or modifying scientific runs."""

import argparse
import fcntl
import json
import threading
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


def refresh(root, cache=None, *, audits=None, publish=True, audit_stamp=None):
    cache = {} if cache is None else cache
    transition = read(root / "azure_transition.json", {})
    transferred = {(r["condition"], r["run_id"]) for r in transition.get("queued", [])}
    azure_root = Path(transition["target_root"]) if transition.get("target_root") else None
    existing = {
        condition: {p.name for p in (root / condition / "runs").iterdir()}
        if (root / condition / "runs").exists()
        else set()
        for condition in ("named", "masked")
    }
    policy = read(root / "release_policy.json", {})
    groups = defaultdict(Counter)
    for plan in read(root / "grid.json", []):
        key = (plan["model_profile"], plan["condition"], plan["site_count"])
        run = root / plan["condition"] / "runs" / plan["run_id"]
        exists = plan["run_id"] in existing[plan["condition"]]
        if (plan["condition"], plan["run_id"]) in transferred:
            run = azure_root / plan["condition"] / "runs" / plan["run_id"]
            exists = run.is_dir()
        result = read(run / "run.json", {}) if exists else {}
        released = plan["model_profile"] in policy.get("released_models", [])
        state = result.get("status") or ("running" if exists else "queued" if released else "held")
        groups[key][state] += 1
        if exists and audits is None:
            for path in (run / "calls").rglob("*.json"):
                groups[key].update(receipt(path, cache))
    native = read(root / "biomni" / "plan.json")
    for plan in native or read(root / "biomni_reserved_plan.json", []):
        key = ("biomni_native", plan["condition"], plan["sites"])
        run = root / "biomni" / plan.get("run_path", "not_started")
        result = read(run / "run.json", {})
        groups[key][result.get("status", "held")] += 1
        if run.exists() and audits is None:
            for path in run.rglob("llm/*.json"):
                groups[key].update(receipt(path, cache, native=True))
    if audits is not None:
        for key in groups:
            groups[key].update(audits.get(key, {}))
    if not publish:
        return {
            key: Counter({name: values[name] for name in ("calls", "errors", "tokens", "unknown")})
            for key, values in groups.items()
        }
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
        "Usage scan: " + (audit_stamp or "initial scan pending; displayed usage is incomplete")
        if audits is not None
        else "Usage scan completed with this refresh.",
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
    if transition:
        lines += [
            "",
            "## Azure transition",
            "",
            f"Status: **{transition['status']}**. {len(transition.get('active', []))} "
            f"original runs retained; {len(transferred)} queued identities reserved for Azure.",
            "Only experiment runs change providers. Unreleased models remain held. "
            "Legacy driver errors on reservation files are admission events, "
            "not scientific failures.",
            f"[Transition record]({root / 'azure_transition.json'}) · "
            f"[Azure bundle]({azure_root / 'README.md'})",
        ]
    temporary = root / "LIVE_PROGRESS.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(root / "LIVE_PROGRESS.md")
    temporary = root / "live_progress.json.tmp"
    temporary.write_text(
        json.dumps(
            dict(
                updated_at=stamp,
                usage_updated_at=audit_stamp,
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
    root = args.root.resolve()
    if args.once:
        refresh(root)
        return
    latest, guard = {"totals": {}, "stamp": None}, threading.Lock()

    def scan():
        cache = {}
        while True:
            try:
                totals = refresh(root, cache, publish=False)
                with guard:
                    latest.update(totals=totals, stamp=datetime.now(UTC).isoformat())
            except Exception as exc:
                print("Usage scan failed:", repr(exc), flush=True)
            time.sleep(30)

    threading.Thread(target=scan, daemon=True).start()
    while True:
        with guard:
            totals, stamp = latest["totals"], latest["stamp"]
        refresh(root, audits=totals, audit_stamp=stamp)
        time.sleep(30)


if __name__ == "__main__":
    main()
