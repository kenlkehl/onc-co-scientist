"""End-to-end CLI test for ``ocs score batch``."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from onc_co_scientist.cli import app
from onc_co_scientist.harness.task_spec import build_tasks
from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.synthetic.cancer_types import CancerType
from onc_co_scientist.synthetic.generator import GeneratorConfig
from onc_co_scientist.synthetic.io import read_manifest
from onc_co_scientist.synthetic.multi import (
    generate_multi_dataset,
    write_multi_bundle_pair,
)


def _build_two_cancer_synth_and_tasks(tmp_path: Path) -> tuple[Path, Path]:
    base = GeneratorConfig(
        dataset_id="ds_batch",
        patient_n=40,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
        n_extra_covariates=4,
    )
    chosen = [CancerType.crc, CancerType.breast]
    bundles = generate_multi_dataset(base, chosen)
    synth_root = tmp_path / "ds"
    write_multi_bundle_pair(bundles, synth_root, anon_seed=0)
    tasks_root = tmp_path / "tasks"
    build_tasks(synth_root, tasks_root, max_iterations=2)
    return synth_root, tasks_root


def _write_transcript(path: Path, manifest_dataset_id: str) -> None:
    payload = Transcript(
        dataset_id=manifest_dataset_id,
        model_id="fake-model",
        harness_id="fake-harness@1",
        max_iterations=2,
        iterations=[],
    ).model_dump_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def test_score_batch_writes_report_with_replicate_aggregation(tmp_path: Path) -> None:
    synth_root, tasks_root = _build_two_cancer_synth_and_tasks(tmp_path)

    # Drop two replicate transcripts under each bundle.
    for ct in ("crc", "breast"):
        for variant in ("named", "anonymized"):
            manifest = read_manifest(synth_root / ct / variant)
            for idx in (1, 2):
                run_dir = tasks_root / ct / variant / "runs" / f"run_{idx:03d}"
                _write_transcript(run_dir / "transcript.json", manifest.dataset_id)

    runner = CliRunner()
    out_dir = tmp_path / "score"
    result = runner.invoke(
        app,
        [
            "score",
            "batch",
            "--synth-root",
            str(synth_root),
            "--tasks-root",
            str(tasks_root),
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    assert (out_dir / "batch_score.json").exists()
    assert (out_dir / "batch_score.md").exists()

    payload = json.loads((out_dir / "batch_score.json").read_text())
    assert payload["n_bundles"] == 4  # 2 cancer types × 2 variants
    assert payload["n_replicates_total"] == 8
    for bundle in payload["per_bundle"]:
        assert bundle["n_replicates"] == 2

    # Markdown shows per-bundle sections.
    md = (out_dir / "batch_score.md").read_text()
    assert "Batch Scoring Report" in md
    assert "n_replicates=2" in md


def test_score_batch_skips_bundles_without_transcripts(tmp_path: Path) -> None:
    synth_root, tasks_root = _build_two_cancer_synth_and_tasks(tmp_path)

    # Only populate one bundle's runs/.
    target_manifest = read_manifest(synth_root / "crc" / "named")
    run_dir = tasks_root / "crc" / "named" / "runs" / "run_001"
    _write_transcript(run_dir / "transcript.json", target_manifest.dataset_id)

    runner = CliRunner()
    out_dir = tmp_path / "score"
    result = runner.invoke(
        app,
        [
            "score",
            "batch",
            "--synth-root",
            str(synth_root),
            "--tasks-root",
            str(tasks_root),
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads((out_dir / "batch_score.json").read_text())
    assert payload["n_bundles"] == 1
    assert payload["n_replicates_total"] == 1
    assert "warning" in result.output.lower() or "skipping" in result.output.lower()


def test_score_batch_errors_when_no_transcripts(tmp_path: Path) -> None:
    synth_root, tasks_root = _build_two_cancer_synth_and_tasks(tmp_path)

    runner = CliRunner()
    out_dir = tmp_path / "score"
    result = runner.invoke(
        app,
        [
            "score",
            "batch",
            "--synth-root",
            str(synth_root),
            "--tasks-root",
            str(tasks_root),
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code != 0
    assert "No replicate transcripts" in result.output or "No replicate transcripts" in str(result.exception)
