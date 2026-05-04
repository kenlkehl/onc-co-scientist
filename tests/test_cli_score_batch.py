"""End-to-end CLI tests for ``ocs score run`` and ``ocs score batch``.

Uses ``--judge stub`` with a JSON config so the LLM is never called.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from onc_co_scientist.cli import JudgeBackend, _build_judge, app
from onc_co_scientist.harness.task_spec import build_tasks
from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.scoring import ClaudeCliJudge, CodexCliJudge
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
        min_buried_treated_subgroup_n=0,
        n_extra_covariates=4,
    )
    chosen = [CancerType.crc, CancerType.breast]
    bundles = generate_multi_dataset(base, chosen)
    synth_root = tmp_path / "ds"
    write_multi_bundle_pair(bundles, synth_root, anon_seed=0)
    tasks_root = tmp_path / "tasks"
    build_tasks(synth_root, tasks_root, max_iterations=2)
    return synth_root, tasks_root


def _write_transcript(path: Path, dataset_id: str, *, encoding: str = "utf-8") -> None:
    payload = Transcript(
        dataset_id=dataset_id,
        model_id="fake-model",
        harness_id="fake-harness@1",
        max_iterations=2,
        iterations=[
            {
                "index": 1,
                "proposed_hypotheses": [{"id": "h1", "text": "feature_037 modifies pfs"}],
                "analyses": [],
            }
        ],
    ).model_dump_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding=encoding)


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


def test_build_judge_supports_claude_and_codex_cli_backends() -> None:
    claude = _build_judge(
        JudgeBackend.claude_cli,
        judge_cli="auto",
        judge_model=None,
        batch_size=3,
        cache_dir=None,
        stub_config_path=None,
    )
    codex = _build_judge(
        JudgeBackend.codex_cli,
        judge_cli="auto",
        judge_model="gpt-5.4",
        batch_size=4,
        cache_dir=None,
        stub_config_path=None,
    )

    assert isinstance(claude, ClaudeCliJudge)
    assert claude.cli == "claude"
    assert claude.batch_size == 3
    assert isinstance(codex, CodexCliJudge)
    assert codex.cli == "codex"
    assert codex.model_id == "gpt-5.4"
    assert codex.batch_size == 4


def test_score_batch_scores_named_and_anonymized_and_writes_report(tmp_path: Path) -> None:
    synth_root, tasks_root = _build_synth_and_tasks(tmp_path)
    # Drop two replicate transcripts under each bundle (named AND
    # anonymized) — both variants should now be scored, with novelty
    # only computed for named.
    for ct in ("crc", "breast"):
        for variant in ("named", "anonymized"):
            manifest = read_manifest(synth_root / ct / variant)
            for idx in (1, 2):
                run_dir = tasks_root / ct / variant / "runs" / f"run_{idx:03d}"
                encoding = (
                    "utf-8-sig" if ct == "crc" and variant == "named" and idx == 1 else "utf-8"
                )
                _write_transcript(
                    run_dir / "transcript.json",
                    manifest.dataset_id,
                    encoding=encoding,
                )

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

    payload = json.loads((out_dir / "batch_score.json").read_text(encoding="utf-8"))
    # 2 cancer types × 2 variants = 4 bundles; 2 replicates each.
    assert payload["n_bundles"] == 4
    assert payload["n_bundles_named"] == 2
    assert payload["n_bundles_anonymized"] == 2
    assert payload["n_replicates_total"] == 8

    by_variant: dict[str, list] = {"named": [], "anonymized": []}
    for b in payload["per_bundle"]:
        by_variant[b["variant"]].append(b)
    assert len(by_variant["named"]) == 2
    assert len(by_variant["anonymized"]) == 2

    for bundle in by_variant["named"]:
        assert bundle["n_replicates"] == 2
        # The single hypothesis "feature_037 ..." matches our novel phrase.
        assert bundle["frac_novel_mean"] == 1.0
    for bundle in by_variant["anonymized"]:
        assert bundle["n_replicates"] == 2
        # Novelty is not computed for anonymized → mean is None.
        assert bundle["frac_novel_mean"] is None

    md = (out_dir / "batch_score.md").read_text(encoding="utf-8")
    assert "Batch Scoring Report" in md
    assert "Novelty %" in md
    assert "Buried discovery iteration — named" in md
    assert "Buried discovery iteration — anonymized" in md
    assert "Fraction near-or-better recovery" in md
    assert "component-or-better recovery" in md

    # Only named replicates contribute to the judgments JSONL.
    judgments = (out_dir / "batch_judgments.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(judgments) == 4  # 2 named bundles × 2 reps × 1 hypothesis each

    # Match-judgment trace covers every (replicate, association, hypothesis)
    # tuple. The base config sets n_buried_signatures=1 → 1 hidden_novel
    # association per bundle; 4 bundles × 2 reps × 1 hypothesis × 1
    # association = 8 lines.
    match_lines = (out_dir / "batch_match_judgments.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(match_lines) == 8
    sample = json.loads(match_lines[0])
    assert {
        "dataset_id",
        "variant",
        "replicate",
        "association_id",
        "iteration",
        "hypothesis_id",
        "text",
        "matches",
        "recovery_level",
        "rationale",
    } <= sample.keys()


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
    transcript_path = tasks_root / "crc" / "named" / "runs" / "run_001" / "transcript.json"
    _write_transcript(transcript_path, manifest.dataset_id, encoding="utf-8-sig")

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

    payload = json.loads((out_dir / "batch_score.json").read_text(encoding="utf-8"))
    assert payload["n_bundles"] == 1
    assert payload["n_replicates_total"] == 1
    assert payload["per_bundle"][0]["variant"] == "named"
    assert payload["per_bundle"][0]["frac_novel_mean"] == 1.0
