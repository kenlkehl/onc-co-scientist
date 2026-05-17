"""AB orchestration and synthesis helpers for the Gemma 31B CAA benchmark."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .harness.task_spec import build_tasks


@dataclass(frozen=True)
class ABArm:
    name: str
    codex_profile: str
    model_alias: str
    scale: float


AB_ARMS: dict[str, ABArm] = {
    "control": ABArm(
        name="control",
        codex_profile="gemma-caa-control",
        model_alias="gemma4-31b-control",
        scale=0.0,
    ),
    "neg002": ABArm(
        name="neg002",
        codex_profile="gemma-caa-neg002",
        model_alias="gemma4-31b-caa-l40-neg002",
        scale=-0.02,
    ),
    "neg005": ABArm(
        name="neg005",
        codex_profile="gemma-caa-neg005",
        model_alias="gemma4-31b-caa-l40-neg005",
        scale=-0.05,
    ),
    "neg010": ABArm(
        name="neg010",
        codex_profile="gemma-caa-neg010",
        model_alias="gemma4-31b-caa-l40-neg010",
        scale=-0.10,
    ),
}

SUMMARY_METRICS = [
    "frac_novel",
    "buried_score_named",
    "buried_score_anonymized",
    "fraction_uncovered_named",
    "fraction_uncovered_anonymized",
    "fraction_near_or_better_named",
    "fraction_near_or_better_anonymized",
    "fraction_component_or_better_named",
    "fraction_component_or_better_anonymized",
]


def parse_arm_names(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("At least one AB arm is required.")
    unknown = [name for name in names if name not in AB_ARMS]
    if unknown:
        valid = ", ".join(AB_ARMS)
        raise ValueError(f"Unknown AB arm(s): {', '.join(unknown)}. Valid arms: {valid}.")
    return names


def arm_root(root: Path, arm_name: str, *, stage: str | None = None) -> Path:
    return root / stage / arm_name if stage else root / arm_name


def run_ab_benchmark(
    *,
    root: Path,
    synth_root: Path,
    stage: str | None,
    arms: list[str],
    replicates: int,
    max_iterations: int,
    jobs: int = 1,
    harness_spec: str = "codex",
    harness_profile: str = "codex",
    python_env: Path | None = None,
    judge: str = "anthropic-vertex",
    judge_cli: str = "auto",
    judge_model: str | None = None,
    judge_batch_size: int = 10,
    cache_dir: Path | None = None,
    no_judge_cache: bool = False,
    build_only: bool = False,
    skip_build: bool = False,
    skip_harness: bool = False,
    skip_score: bool = False,
    dry_run: bool = False,
) -> list[list[str]]:
    """Build/run/score the requested arms sequentially.

    Returned commands are the harness/score subprocesses that were executed or
    would be executed in ``dry_run`` mode.
    """

    if replicates < 1:
        raise ValueError("replicates must be >= 1.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1.")
    repo_root = Path(__file__).resolve().parents[2]
    run_harness = repo_root / "scripts" / "run_harness.sh"
    if not run_harness.is_file():
        raise FileNotFoundError(f"Cannot find harness runner at {run_harness}.")

    commands: list[list[str]] = []
    for arm_name in arms:
        arm = AB_ARMS[arm_name]
        base = arm_root(root, arm_name, stage=stage)
        tasks_root = base / "tasks"
        score_root = base / "score"
        if not skip_build and not dry_run:
            build_tasks(
                synth_root,
                tasks_root,
                max_iterations=max_iterations,
                python_env=python_env,
            )
        if build_only:
            continue

        harness_cmd = [
            str(run_harness),
            harness_spec,
            str(tasks_root),
            "--profile",
            harness_profile,
            "--extra-args",
            f"--profile {arm.codex_profile}",
            "--jobs",
            str(jobs),
            "--replicates",
            str(replicates),
        ]
        if python_env is not None:
            harness_cmd.extend(["--python-env", str(python_env)])
        if not skip_harness:
            commands.append(harness_cmd)
            if not dry_run:
                env = os.environ.copy()
                env.setdefault("OPENAI_API_KEY", "EMPTY")
                subprocess.run(harness_cmd, check=True, cwd=repo_root, env=env)

        score_cmd = [
            sys.executable,
            "-m",
            "onc_co_scientist.cli",
            "score",
            "batch",
            "--synth-root",
            str(synth_root),
            "--tasks-root",
            str(tasks_root),
            "--out",
            str(score_root),
            "--judge",
            judge,
            "--judge-cli",
            judge_cli,
            "--judge-batch-size",
            str(judge_batch_size),
        ]
        if judge_model is not None:
            score_cmd.extend(["--judge-model", judge_model])
        if cache_dir is not None:
            score_cmd.extend(["--cache-dir", str(cache_dir)])
        if no_judge_cache:
            score_cmd.append("--no-judge-cache")
        if not skip_score:
            commands.append(score_cmd)
            if not dry_run:
                subprocess.run(score_cmd, check=True, cwd=repo_root)
    return commands


def summarize_ab(root: Path, out: Path, *, stage: str | None = None) -> dict[str, Any]:
    """Read arm score JSON files and write Markdown/CSV synthesis reports."""

    score_files = discover_score_files(root, stage=stage)
    if not score_files:
        detail = f" under stage {stage!r}" if stage else ""
        raise FileNotFoundError(f"No batch_score.json files found under {root}{detail}.")

    rows: list[dict[str, Any]] = []
    per_bundle_rows: list[dict[str, Any]] = []
    for score_file in score_files:
        payload = json.loads(score_file.read_text(encoding="utf-8"))
        detected_stage, arm_name = _stage_arm_from_score_path(root, score_file)
        arm = AB_ARMS.get(arm_name)
        scale = arm.scale if arm is not None else _scale_from_arm_name(arm_name)
        row = {
            "stage": detected_stage,
            "arm": arm_name,
            "scale": scale,
            "score_path": str(score_file),
            "n_bundles": payload.get("n_bundles"),
            "n_replicates_total": payload.get("n_replicates_total"),
        }
        for metric in SUMMARY_METRICS:
            row[metric] = payload.get(metric)
        row.update(_summed_recovery_counts(payload, variant="named", prefix="named"))
        row.update(_summed_recovery_counts(payload, variant="anonymized", prefix="anonymized"))
        rows.append(row)

        for bundle in payload.get("per_bundle", []):
            bundle_row = {
                "stage": detected_stage,
                "arm": arm_name,
                "scale": scale,
                "dataset_id": bundle.get("dataset_id"),
                "variant": bundle.get("variant"),
                "n_replicates": bundle.get("n_replicates"),
                "frac_novel_mean": bundle.get("frac_novel_mean"),
                "buried_score_mean": bundle.get("buried_score_mean"),
                "fraction_uncovered": bundle.get("fraction_uncovered"),
                "fraction_near_or_better": bundle.get("fraction_near_or_better"),
                "fraction_component_or_better": bundle.get("fraction_component_or_better"),
            }
            for level in ("exact", "near", "component", "none"):
                bundle_row[f"recovery_{level}"] = (bundle.get("recovery_level_counts") or {}).get(
                    level, 0
                )
            per_bundle_rows.append(bundle_row)

    rows.sort(key=lambda row: (row["stage"], _scale_sort_key(row.get("scale")), row["arm"]))
    _add_control_deltas(rows)

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "ab_summary.csv", rows)
    _write_csv(out / "per_bundle.csv", per_bundle_rows)
    (out / "ab_summary.md").write_text(render_ab_markdown(rows), encoding="utf-8")
    return {"summary_rows": rows, "per_bundle_rows": per_bundle_rows}


def discover_score_files(root: Path, *, stage: str | None = None) -> list[Path]:
    if stage:
        return sorted((root / stage).glob("*/score/batch_score.json"))
    direct = sorted(root.glob("*/score/batch_score.json"))
    staged = sorted(root.glob("*/*/score/batch_score.json"))
    return direct + [path for path in staged if path not in direct]


def render_ab_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# CAA Gemma 31B AB Summary", ""]
    for stage in _ordered_stages(rows):
        stage_rows = [row for row in rows if row["stage"] == stage]
        heading = stage or "flat"
        lines.append(f"## Stage: {heading}")
        lines.append("")
        lines.append(
            "| arm | scale | n_reps | frac_novel | delta | buried_named | delta | "
            "buried_anon | delta | uncovered_named | uncovered_anon | near_named | near_anon |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in stage_rows:
            lines.append(
                "| {arm} | {scale} | {n_reps} | {frac} | {frac_d} | {bn} | {bn_d} | "
                "{ba} | {ba_d} | {un} | {ua} | {nn} | {na} |".format(
                    arm=row["arm"],
                    scale=_fmt(row.get("scale"), 2),
                    n_reps=row.get("n_replicates_total", ""),
                    frac=_fmt(row.get("frac_novel")),
                    frac_d=_fmt(row.get("delta_frac_novel")),
                    bn=_fmt(row.get("buried_score_named")),
                    bn_d=_fmt(row.get("delta_buried_score_named")),
                    ba=_fmt(row.get("buried_score_anonymized")),
                    ba_d=_fmt(row.get("delta_buried_score_anonymized")),
                    un=_fmt(row.get("fraction_uncovered_named")),
                    ua=_fmt(row.get("fraction_uncovered_anonymized")),
                    nn=_fmt(row.get("fraction_near_or_better_named")),
                    na=_fmt(row.get("fraction_near_or_better_anonymized")),
                )
            )
        lines.append("")
        lines.append("### Recovery-Level Counts")
        lines.append("")
        lines.append(
            "| arm | named exact/near/component/none | anonymized exact/near/component/none |"
        )
        lines.append("|---|---:|---:|")
        for row in stage_rows:
            named_counts = "/".join(str(row.get(f"named_recovery_{level}", 0)) for level in _LEVELS)
            anon_counts = "/".join(
                str(row.get(f"anonymized_recovery_{level}", 0)) for level in _LEVELS
            )
            lines.append(f"| {row['arm']} | {named_counts} | {anon_counts} |")
        lines.append("")
        lines.append("### Dose Response")
        lines.append("")
        for row in stage_rows:
            lines.append(
                "- scale {scale}: frac_novel={frac}, buried_named={bn}, "
                "buried_anonymized={ba}, uncovered_named={un}, uncovered_anonymized={ua}".format(
                    scale=_fmt(row.get("scale"), 2),
                    frac=_fmt(row.get("frac_novel")),
                    bn=_fmt(row.get("buried_score_named")),
                    ba=_fmt(row.get("buried_score_anonymized")),
                    un=_fmt(row.get("fraction_uncovered_named")),
                    ua=_fmt(row.get("fraction_uncovered_anonymized")),
                )
            )
        lines.append("")
    return "\n".join(lines)


_LEVELS = ("exact", "near", "component", "none")


def _stage_arm_from_score_path(root: Path, score_file: Path) -> tuple[str, str]:
    rel = score_file.relative_to(root)
    parts = rel.parts
    if len(parts) >= 4 and parts[2] == "score":
        return parts[0], parts[1]
    if len(parts) >= 3 and parts[1] == "score":
        return "", parts[0]
    return "", score_file.parent.parent.name


def _summed_recovery_counts(
    payload: dict[str, Any],
    *,
    variant: str,
    prefix: str,
) -> dict[str, int]:
    counts = {f"{prefix}_recovery_{level}": 0 for level in _LEVELS}
    for bundle in payload.get("per_bundle", []):
        if bundle.get("variant") != variant:
            continue
        raw_counts = bundle.get("recovery_level_counts") or {}
        for level in _LEVELS:
            counts[f"{prefix}_recovery_{level}"] += int(raw_counts.get(level, 0) or 0)
    return counts


def _add_control_deltas(rows: list[dict[str, Any]]) -> None:
    controls = {
        row["stage"]: row
        for row in rows
        if row["arm"] == "control" or float(row.get("scale") or 0.0) == 0.0
    }
    for row in rows:
        control = controls.get(row["stage"])
        for metric in SUMMARY_METRICS:
            value = row.get(metric)
            base = control.get(metric) if control else None
            row[f"delta_{metric}"] = _numeric_delta(value, base)


def _numeric_delta(value: Any, base: Any) -> float | None:
    if value is None or base is None:
        return None
    try:
        return float(value) - float(base)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ordered_stages(rows: list[dict[str, Any]]) -> list[str]:
    stages: list[str] = []
    for row in rows:
        stage = row["stage"]
        if stage not in stages:
            stages.append(stage)
    return stages


def _scale_sort_key(value: Any) -> float:
    if value is None:
        return 999.0
    try:
        return -float(value)
    except (TypeError, ValueError):
        return 999.0


def _scale_from_arm_name(arm_name: str) -> float | None:
    if arm_name == "control":
        return 0.0
    match = arm_name.removeprefix("neg")
    if match.isdigit():
        return -int(match) / 100.0
    return None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)
