from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from onc_co_scientist.caa_ab import summarize_ab


def _write_score(root: Path, stage: str, arm: str, *, frac_novel: float, uncovered: float) -> None:
    score_dir = root / stage / arm / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_bundles": 2,
        "n_replicates_total": 4,
        "n_bundles_named": 1,
        "n_bundles_anonymized": 1,
        "frac_novel": frac_novel,
        "buried_score_named": 10.0,
        "buried_score_anonymized": 14.0,
        "fraction_uncovered_named": uncovered,
        "fraction_uncovered_anonymized": 0.5,
        "fraction_near_or_better_named": uncovered,
        "fraction_near_or_better_anonymized": 0.75,
        "fraction_component_or_better_named": 1.0,
        "fraction_component_or_better_anonymized": 1.0,
        "per_bundle": [
            {
                "dataset_id": "ds001_crc",
                "variant": "named",
                "n_replicates": 2,
                "frac_novel_mean": frac_novel,
                "buried_score_mean": 10.0,
                "fraction_uncovered": uncovered,
                "fraction_near_or_better": uncovered,
                "fraction_component_or_better": 1.0,
                "recovery_level_counts": {
                    "exact": 1,
                    "near": 1,
                    "component": 0,
                    "none": 0,
                },
            },
            {
                "dataset_id": "ds001_crc",
                "variant": "anonymized",
                "n_replicates": 2,
                "frac_novel_mean": None,
                "buried_score_mean": 14.0,
                "fraction_uncovered": 0.5,
                "fraction_near_or_better": 0.75,
                "fraction_component_or_better": 1.0,
                "recovery_level_counts": {
                    "exact": 1,
                    "near": 0,
                    "component": 1,
                    "none": 0,
                },
            },
        ],
    }
    (score_dir / "batch_score.json").write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_ab_writes_deltas_and_per_bundle_csv(tmp_path: Path) -> None:
    root = tmp_path / "ab"
    _write_score(root, "pilot", "control", frac_novel=0.25, uncovered=0.5)
    _write_score(root, "pilot", "neg002", frac_novel=0.40, uncovered=0.75)

    out = tmp_path / "summary"
    result = summarize_ab(root, out, stage="pilot")

    assert len(result["summary_rows"]) == 2
    neg = next(row for row in result["summary_rows"] if row["arm"] == "neg002")
    assert neg["delta_frac_novel"] == pytest.approx(0.15)
    assert neg["delta_fraction_uncovered_named"] == pytest.approx(0.25)
    assert neg["named_recovery_exact"] == 1
    assert neg["anonymized_recovery_component"] == 1

    assert (out / "ab_summary.md").exists()
    md = (out / "ab_summary.md").read_text(encoding="utf-8")
    assert "Dose Response" in md
    assert "Recovery-Level Counts" in md

    summary_rows = list(csv.DictReader((out / "ab_summary.csv").open(encoding="utf-8")))
    assert {row["arm"] for row in summary_rows} == {"control", "neg002"}
    neg_csv = next(row for row in summary_rows if row["arm"] == "neg002")
    assert float(neg_csv["delta_frac_novel"]) == pytest.approx(0.15)

    per_bundle_rows = list(csv.DictReader((out / "per_bundle.csv").open(encoding="utf-8")))
    assert len(per_bundle_rows) == 4
    assert {"named", "anonymized"} == {row["variant"] for row in per_bundle_rows}
