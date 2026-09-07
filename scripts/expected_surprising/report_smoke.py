"""Render a readable smoke report and retain the source of its scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def fmt(value):
    return "Unavailable" if value is None else f"{value:.2f}"


def label(report):
    profile = report["profile"]
    kind = {
        "nsclc_clinical": "Clinical",
        "nsclc_depmap": "Cell-line",
        "clinical": "Clinical",
        "depmap": "Cell-line",
    }.get(profile, profile)
    return f"{kind} — {report['version']}"


def write_report(smoke, out, audit=None):
    smoke = Path(smoke).resolve()
    manifest = json.loads((smoke / "manifest.json").read_text())
    if manifest["config"]["versions"]["workflow"] not in {"appraisal-3.1.0", "appraisal-3.2.0"}:
        raise ValueError("This report requires an agent-judgment workflow")
    paired = json.loads((smoke / "paired_summary.json").read_text())
    summaries = json.loads((smoke / "summary.json").read_text())
    reports = [
        json.loads(path.read_text())
        for task in manifest["tasks"]
        if (path := smoke / "runs" / task["run_id"] / "report.json").exists()
    ]
    metadata = [
        json.loads(line)
        for path in sorted((smoke / "api_metadata").glob("*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    usage = {
        key: sum((record.get("usage") or {}).get(key, 0) for record in metadata)
        for key in ("completion_tokens", "prompt_tokens", "total_tokens")
    }
    elapsed = (
        datetime.fromisoformat(manifest["completed_at"])
        - datetime.fromisoformat(manifest["started_at"])
    ).total_seconds()
    config = manifest["config"]
    provider = config["provider"]
    total_stages = 4 * manifest["iterations"] * len(manifest["tasks"])
    successful = sum(r["successful_stages"] for r in reports)
    errors = sum(len(r["attempt_errors"]) for r in reports)
    recovered = sum(len(r["recovered_stages"]) for r in reports)
    exhausted = sum(len(r["protocol_errors"]) for r in reports)

    lines = [
        "# Expected/surprising smoke test",
        "",
        f"We tested {provider['model_id']} on {len(manifest['tasks'])} synthetic datasets. "
        "The agent judged effect sizes and conclusions without being given a minimum effect "
        "to aim for.",
        "",
        f"**{successful}/{total_stages} steps completed.** There were {errors} failed attempts, "
        f"{recovered} successful retries, and {exhausted} steps that could not be completed.",
        "",
        f"## {manifest['iterations']} rounds per run",
        "",
        f"Each run had **{manifest['iterations']} rounds**, called iterations in the saved data. "
        "Each round has four steps: explore, analyze, appraise, and synthesize.",
        "",
        "**The two-round rule is a follow-up deadline, not a cap on the run.** "
        "The agent assesses validation evidence when it arrives, then reassesses it two rounds "
        "later. For example, a result received in round 2 is assessed again in round 4.",
        "",
        "The agent can remain unresolved and keep investigating within the run. "
        "The response score uses its assessment at that scheduled check.",
        "",
        "The standard full runs allow 25 rounds for both clinical and cell-line data. "
        f"These shorter {manifest['iterations']}-round runs are smoke tests.",
        "",
        "## Recall, precision, and F1*",
        "",
        "- **Recall (R)** is sensitivity: how many of the embedded discoveries the agent tested, "
        "accepted, and independently confirmed. Expected, neutral, and surprising discoveries "
        "receive equal weight.",
        "- **Precision (P)** is positive predictive value: the confirmed fraction of the agent's "
        "final accepted findings. "
        "A finding must have been tested and then confirmed on fresh data. Additional discoveries "
        "can count, even when they were not in the embedded target list.",
        "- **F1\\*** combines recall and precision using the familiar F1 formula, scaled to 100: "
        "**200 × R × P / (R + P)**.",
        "",
        "\\* This is an F1-style score. Recall is averaged across three discovery categories, "
        "and precision can include additional findings. The two measures therefore do not come "
        "from one ordinary classification confusion matrix.",
        "",
        "**The agent does not have to validate before accepting.** The benchmark tests its final "
        "accepted findings on fresh data after the run. A finding accepted earlier can still "
        "count toward precision.",
        "",
        "Confirmation still uses the benchmark's private effect cutoffs: 0.10 on natural-log "
        "PFS and 0.15 dependency-score units. An unconfirmed finding can have a real positive "
        "effect that does not clearly exceed that cutoff. Disagreement with this scoring rule "
        "does not by itself make the agent's scientific judgment unreasonable.",
        "",
        "## Results",
        "",
        "Recall and precision range from 0 to 1. F1\\* ranges from 0 to 100.",
        "",
        "| Run | Recall (R) | Precision (P) | F1\\* | Confirmed / accepted findings |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in reports:
        d = r["scores"]["discovery"]
        exact = d["exact"]
        lines.append(
            f"| {label(r)} | {fmt(exact['R'])} | {fmt(exact['Q'])} | "
            f"{fmt(r['scores']['D'])} | {d['confirmed_tested_n']} / {d['accepted_n']} |"
        )
    lines += [
        "",
        "Each expected/surprising pair reverses one featured finding, the focal discovery. "
        "Recovering it requires testing the exact comparison, accepting its correct direction, "
        "and confirming it on fresh data.",
        "",
        "| Run | Focal discovery recovered? |",
        "|---|---|",
    ]
    for r in reports:
        lines.append(f"| {label(r)} | {'Yes' if r['confirmation']['primary_recovery'] else 'No'} |")
    lines += [
        "",
        "## Exploration and response to evidence",
        "",
        "**Exploration coverage (E)** measures how broadly and how early the agent tested the "
        "embedded comparisons. It gives credit for testing, regardless of acceptance or direction.",
        "",
        "**Evidence responsiveness (B)** measures whether the agent's assessment two rounds "
        "after validation agrees with the benchmark's private rule. It gives equal weight to "
        "three situations:",
        "",
        "- The whole interval is above the private cutoff: the reference says accept.",
        "- The whole interval is below the cutoff: the reference says reject.",
        "- The interval crosses or touches the cutoff: the reference says unresolved.",
        "",
        "Both scores range from 0 to 100. A response score is unavailable when a run has no "
        "scorable example of one of those three situations.",
        "",
        "| Run | Coverage (E) | Response to evidence (B) |",
        "|---|---:|---:|",
    ]
    for r in reports:
        lines.append(f"| {label(r)} | {fmt(r['scores']['E'])} | {fmt(r['scores']['B'])} |")
    if "D" in paired:
        lines += [
            "",
            f"**Across the {len(manifest['tasks'])} runs:** F1\\* **{fmt(paired['D'])}**, "
            f"coverage **{fmt(paired['E'])}**, and response to evidence **{fmt(paired['B'])}**.",
            "",
            "For F1\\* and coverage, each run gets equal weight. For response to evidence, "
            "we average each of the three situations separately before combining them. "
            "The overall response score can therefore be available even when some individual "
            "runs are missing a situation.",
        ]
    lines += [
        "",
        "These are small workflow checks. They are too few to rank models or establish whether "
        "the instruction change improved performance.",
        "",
        "<details>",
        "<summary>Run settings, detailed counts, and source files</summary>",
        "",
        "## Run settings",
        "",
        f"- {config['max_tokens_per_call']:,} completion tokens per call, including reasoning.",
        f"- Up to {config['max_retries_per_stage']} retries for a failed step. "
        "This retry limit is separate from the number of rounds.",
        f"- {provider['timeout_s'] / 60:g}-minute timeout per call; "
        f"{manifest['workers']} runs can execute at once.",
        f"- {elapsed / 60:.1f} minutes elapsed; {len(metadata)} calls; "
        f"{usage['completion_tokens']:,} completion tokens used.",
        f"- Server: {provider['base_url']}. The controller can use other configured providers.",
    ]
    for profile, policy in config["validation_policies"].items():
        bound = policy["voluntary_limit"] + len(policy["release_iterations"])
        lines.append(
            f"- {label({'profile': profile, 'version': ''}).rstrip(' —')}: "
            f"scheduled validation in rounds {', '.join(map(str, policy['release_iterations']))}; "
            f"follow-up {policy['response_window']} rounds later; at most "
            f"{policy['voluntary_limit']} voluntary requests. Validation alpha = 0.05 / {bound}."
        )
    lines += [
        "",
        "The agent also assesses newly delivered evidence immediately. A result arriving too "
        "late for the follow-up does not enter the response score. Invalid results and interrupted "
        "follow-ups are also excluded; a missing assessment at a due deadline counts "
        "as disagreement.",
        "",
        "If a step fails even after its retries, the run's primary F1\\* and focal recovery "
        "are zero. Separate scores based on first attempts are retained in the saved results.",
    ]
    if audit:
        audit = Path(audit).resolve()
        verification = json.loads(audit.read_text())
        lines += [
            "",
            f"**Audit: {'passed' if verification['passed'] else 'failed'}.** "
            f"The checks cover {verification['packages']['task_n']} dataset packages, "
            "the code used for the runs, cutoff removal from agent inputs, validation timing, "
            f"required responses, and score arithmetic. [Audit details]({audit}).",
        ]
    lines += [
        "",
        "## Counts behind the scores",
        "",
        "Each response cell below is agreements with the private rule / scorable results. "
        "A 0/0 cell means no examples were available.",
        "",
        "| Run | Above cutoff | Below cutoff | Crosses cutoff | Too late |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in reports:
        b = r["responsiveness"]
        counts = [
            f"{b['components'][c]['numerator']} / {b['components'][c]['denominator']}"
            for c in ("supported", "excluded", "ambiguous")
        ]
        lines.append(f"| {label(r)} | {' | '.join(counts)} | {b['exclusions']['late']} |")
    lines += [
        "",
        "| Run | Recall: expected / neutral / surprising | Validation: requested / automatic |",
        "|---|---|---:|",
    ]
    for r in reports:
        rates = " / ".join(
            fmt(r["scores"]["discovery"]["exact"]["R_c"][c])
            for c in ("expected", "neutral", "surprising")
        )
        events = r["responsiveness"]["events"]
        routes = [sum(e["source"] == s for e in events) for s in ("voluntary", "automatic")]
        lines.append(f"| {label(r)} | {rates} | {routes[0]} / {routes[1]} |")
    if "D" in paired:
        lines += [
            "",
            "| Dataset type | F1\\* | Coverage (E) | Response to evidence (B) |",
            "|---|---:|---:|---:|",
        ]
        for name, p in paired["modalities"].items():
            kind = {"clinical": "Clinical", "depmap": "Cell-line"}.get(name, name)
            lines.append(f"| {kind} | {fmt(p['D'])} | {fmt(p['E'])} | {fmt(p['B'])} |")
    lines += [
        "",
        "There is only one expected/surprising pair for each dataset type here, so the report "
        "does not estimate uncertainty across different datasets.",
        "",
        "## Errors and source files",
        "",
    ]
    for r in reports:
        for e in r["attempt_errors"]:
            lines.append(
                f"- {label(r)}, round {e['iteration']}, {e['stage']}, attempt "
                f"{e['attempt']}: {e['error']}"
            )
    for s in summaries:
        if not s["completed"]:
            lines.append(f"- {s['run_id']} failed: {s.get('error', 'unspecified error')}")
    lines += [
        "",
        "The report labels are updated; the original numerical records retain their historical "
        "keys: P corresponds to Q, and F1\\* corresponds to D. "
        "Values and scoring rules are unchanged.",
        "",
        f"- [Settings, versions, dates, and file checksums]({smoke / 'manifest.json'})",
        f"- [Summary across runs]({smoke / 'paired_summary.json'})",
        f"- [Detailed run summary]({smoke / 'summary.json'})",
    ]
    for r in reports:
        run = r["run_id"]
        lines.append(
            f"- {label(r)}: [results]({smoke / 'runs' / run / 'report.json'}), "
            f"[agent transcript]({smoke / 'runs' / run / 'transcript.jsonl'}), "
            f"[call details]({smoke / 'api_metadata' / (run + '.jsonl')})."
        )
    lines += ["", "</details>"]
    Path(out).write_text("\n".join(lines) + "\n")
    return {
        "manifest": manifest,
        "paired_summary": paired,
        "usage": usage,
        "provider_call_n": len(metadata),
        "elapsed_s": elapsed,
        "report_render": {
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "markdown_sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest(),
            "label_to_record_key": {"R": "R", "P": "Q", "F1*": "D", "E": "E", "B": "B"},
            "presentation_only": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    evidence = write_report(args.smoke, args.out, args.audit)
    if args.archive:
        evidence["raw_artifact_sha256"] = {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.smoke.rglob("*"))
            if p.is_file()
        }
        args.out.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
