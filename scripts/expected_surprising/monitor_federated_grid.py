"""Refresh a local progress page without launching or modifying scientific runs."""

import argparse
import fcntl
import json
import threading
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
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
        input_tokens=usage.get("prompt_tokens" if native else "input_tokens") or 0,
        input_unknown=usage.get("prompt_tokens" if native else "input_tokens") is None,
        cost_unknown=1,
    )
    if not native or value.get("status") in {"completed", "failed"}:
        cache[path] = counts
    return counts


def provider_receipt(path, cache, rates):
    """Read small provider receipts, never multi-megabyte scientific prompt journals."""
    key = (path, json.dumps(rates, sort_keys=True))
    if key in cache:
        return cache[key]
    data = read(path)
    if data is None:
        return Counter(calls=1, unknown=1, input_unknown=1, cost_unknown=1)
    usage = data.get("cli_usage") or data.get("usage", {})
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    cached, written = usage.get("cached_input_tokens"), usage.get("cache_write_input_tokens")
    counts = Counter(
        calls=1,
        tokens=out or 0,
        input_tokens=inp or 0,
        cache_reads=cached or 0,
        cache_writes=written or 0,
        unknown=out is None,
        input_unknown=inp is None,
        infrastructure_retries=max(0, data.get("infrastructure_attempts", 1) - 1),
    )
    cost = data.get("cost_micro_usd")
    if (
        cost is None
        and rates
        and all(type(n) is int and n >= 0 for n in (inp, out, cached, written))
        and cached + written <= inp
    ):
        tariff = rates["long" if inp > rates["short_context_tokens"] else "short"]
        cost = int(
            sum(
                Decimal(str(tariff[k])) * n
                for k, n in (
                    ("input", inp - cached - written),
                    ("cached", cached),
                    ("write", written),
                    ("output", out),
                )
            ).to_integral_value(rounding=ROUND_CEILING)
        )
    counts["cost_micro_usd"] = cost or 0
    counts["cost_unknown"] = cost is None
    cache[key] = counts
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
    spending = read(root / "control/cache_fix_v1/spend_policy.json", {})
    spend_state = read(root / "control/cache_fix_v1/spend_state.json", {})
    price_models = {v: k for k, v in spending.get("model_profiles", {}).items()}
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
        driver = read(root / "control" / plan["model_profile"] / "execution.json", {})
        paused = driver.get("status") == "paused"
        finished_state = next(
            (
                r.get("status")
                for r in driver.get("finished", [])
                if r["run_id"] == plan["run_id"] and r["condition"] == plan["condition"]
            ),
            None,
        )
        state = (
            result.get("status")
            or finished_state
            or (
                "paused"
                if exists and (paused or not released)
                else "running"
                if exists
                else "queued"
                if released
                else "held"
            )
        )
        if (
            not result.get("status")
            and not finished_state
            and released
            and driver.get("status") == "running"
            and "active" in driver
        ):
            state = (
                "running"
                if any(
                    r["run_id"] == plan["run_id"] and r["condition"] == plan["condition"]
                    for r in driver["active"]
                )
                else "queued"
            )
        groups[key][state] += 1
        if exists and audits is None:
            provider_audits = [
                p for p in (run / "provider_audit", run / "central_provider_audit") if p.is_dir()
            ]
            if provider_audits:
                model_name = price_models.get(plan["model_profile"])
                rates = spending.get("rates", {}).get(model_name)
                for audit in provider_audits:
                    # Mixed-model central providers require explicit per-receipt model pricing.
                    audit_rates = rates if audit.name == "provider_audit" else None
                    for call in audit.glob("call-*"):
                        if call.is_dir():
                            groups[key].update(
                                provider_receipt(call / "metrics.json", cache, audit_rates)
                            )
            else:
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
            key: Counter(
                {
                    name: values[name]
                    for name in (
                        "calls",
                        "errors",
                        "tokens",
                        "unknown",
                        "input_tokens",
                        "input_unknown",
                        "cache_reads",
                        "cache_writes",
                        "cost_micro_usd",
                        "cost_unknown",
                        "infrastructure_retries",
                    )
                }
            )
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
        f"Completed {totals['completed']} · Failed {totals['failed']} · "
        f"Paused {totals['paused']} · Held {totals['held']}**",
        "",
        "Refreshes every 30 seconds. Biomni and all unreleased models stay held. "
        "Input, output, cache reads/writes and estimated cost include recorded provider receipts. "
        "Missing token receipts are counted separately. Native reporting rounds and controller "
        "iterations are different clocks.",
        f"**Recorded cost estimate: ${totals['cost_micro_usd'] / 1_000_000:,.2f}** · "
        f"Input {totals['input_tokens']:,} · Cache reads {totals['cache_reads']:,} · "
        f"Cache writes {totals['cache_writes']:,} · "
        f"Receipts with unknown cost {totals['cost_unknown']:,}. Not an Azure invoice; "
        "unreported attempts can add charges.",
        "Usage scan: " + (audit_stamp or "initial scan pending; displayed usage is incomplete")
        if audits is not None
        else "Usage scan completed with this refresh.",
        "",
        "| Model | View | Sites | Held | Queued | Active | Paused | Done | Failed | Calls | "
        "Errors | Output tokens | Unknown usage | Input tokens | Cache reads | Cache writes | "
        "Cost USD | Unknown cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (model, view, sites), c in sorted(groups.items()):
        values = [
            c[k]
            for k in (
                "held",
                "queued",
                "running",
                "paused",
                "completed",
                "failed",
                "calls",
                "errors",
                "tokens",
                "unknown",
                "input_tokens",
                "cache_reads",
                "cache_writes",
            )
        ]
        lines.append(
            f"| {model} | {view} | {sites} | "
            + " | ".join(f"{v:,}" for v in values)
            + f" | ${c['cost_micro_usd'] / 1_000_000:,.2f} | {c['cost_unknown']:,} |"
        )
    if spending:
        reserved = sum(
            a["reserved_micro_usd"]
            for a in spend_state.get("attempts", {}).values()
            if a["status"] in {"reserved", "unknown"}
        )
        lines += [
            "",
            f"Additional budget: ${spending['additional_budget_usd']:,.2f}; "
            f"settled ${spend_state.get('spent_micro_usd', 0) / 1_000_000:,.2f}; "
            f"reserved/unknown ${reserved / 1_000_000:,.2f}. "
            f"Gate: {'enabled' if spending.get('enabled') else 'held'}. "
            f"{spend_state.get('hold_reason', '')}",
        ]
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
        if transition.get("deadline"):
            lines += [
                "",
                f"**Personal-account cutoff: {transition['deadline']}.** "
                "Unfinished initial runs resume on Azure using their saved call journals.",
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
