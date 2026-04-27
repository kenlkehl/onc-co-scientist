"""End-to-end CLI tests for ``ocs score run`` and ``ocs score batch``.

Uses ``--judge stub`` with a JSON config so the LLM is never called.
"""

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


def _build_synth_and_tasks(tmp_path: Path) -> tuple[Path, Path]:
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


def _write_transcript(path: Path, dataset_id: str) -> None:
    payload = Transcript(
        dataset_id=dataset_id,
        model_id="fake-model",
        harness_id="fake-harness@1",
        max_iterations=2,
        iterations=[
            {
                "index": 1,
                "proposed_hypotheses": [
                    {"id": "h1", "text": "feature_037 modifies pfs"}
                ],
                "analyses": [],
            }
        ],
    ).model_dump_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def _stub_config(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "novel_phrases": ["feature_"],
                "match_phrases": {},
            }
        )
    )
    return path


def test_score_batch_skips_anonymized_bundles_and_writes_report(tmp_path: Path) -> None:
    synth_root, tasks_root = _build_synth_and_tasks(tmp_path)
    # Drop two replicate transcripts under each bundle (named AND anonymized
    # — to verify anonymized bundles are filtered out by the CLI, not by the
    # absence of transcripts).
    for ct in ("crc", "breast"):
        for variant in ("named", "anonymized"):
            manifest = read_manifest(synth_root / ct / variant)
            for idx in (1, 2):
                run_dir = tasks_root / ct / variant / "runs" / f"run_{idx:03d}"
                _write_transcript(run_dir / "transcript.json", manifest.dataset_id)

    runner = CliRunner()
    out_dir = tmp_path / "score"
    stub_cfg = _stub_config(tmp_path / "stub.json")
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
            "--judge",
            "stub",
            "--stub-config",
            str(stub_cfg),
            "--no-judge-cache",
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads((out_dir / "batch_score.json").read_text())
    # 2 cancer types × named only = 2 bundles; anonymized excluded.
    assert payload["n_bundles"] == 2
    assert payload["n_replicates_total"] == 4
    for bundle in payload["per_bundle"]:
        assert bundle["n_replicates"] == 2
        # The single hypothesis "feature_037 ..." matches our novel phrase.
        assert bundle["frac_novel_mean"] == 1.0

    md = (out_dir / "batch_score.md").read_text()
    assert "Batch Scoring Report" in md
    assert "Novelty %" in md
    assert "Buried discovery iteration" in md

    judgments = (out_dir / "batch_judgments.jsonl").read_text().splitlines()
    assert len(judgments) == 4  # 2 bundles × 2 reps × 1 hypothesis each


def test_score_batch_errors_when_no_transcripts(tmp_path: Path) -> None:
    synth_root, tasks_root = _build_synth_and_tasks(tmp_path)
    runner = CliRunner()
    out_dir = tmp_path / "score"
    stub_cfg = _stub_config(tmp_path / "stub.json")
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
            "--judge",
            "stub",
            "--stub-config",
            str(stub_cfg),
            "--no-judge-cache",
        ],
    )
    assert result.exit_code != 0
    combined = result.output + str(result.exception or "")
    assert "No replicate transcripts" in combined


def test_score_run_writes_single_replicate_report(tmp_path: Path) -> None:
    synth_root, tasks_root = _build_synth_and_tasks(tmp_path)
    bundle = synth_root / "crc" / "named"
    manifest = read_manifest(bundle)
    transcript_path = (
        tasks_root / "crc" / "named" / "runs" / "run_001" / "transcript.json"
    )
    _write_transcript(transcript_path, manifest.dataset_id)

    runner = CliRunner()
    out_dir = tmp_path / "score"
    stub_cfg = _stub_config(tmp_path / "stub.json")
    result = runner.invoke(
        app,
        [
            "score",
            "run",
            "--dataset",
            str(bundle),
            "--transcript",
            str(transcript_path),
            "--out",
            str(out_dir),
            "--judge",
            "stub",
            "--stub-config",
            str(stub_cfg),
            "--no-judge-cache",
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads((out_dir / "batch_score.json").read_text())
    assert payload["n_bundles"] == 1
    assert payload["n_replicates_total"] == 1
    assert payload["per_bundle"][0]["frac_novel_mean"] == 1.0
