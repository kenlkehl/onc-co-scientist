"""CLI integration tests for the multi-cancer ``ocs synth generate`` command."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from onc_co_scientist.cli import app, _parse_cancer_types
from onc_co_scientist.synthetic.cancer_types import CancerType, all_cancer_types


def _write_minimal_config(path: Path, patient_n: int = 80) -> Path:
    config = {
        "dataset_id": "ds_test",
        "patient_n": patient_n,
        "seed": 0,
        "n_buried_signatures": 1,
        "min_buried_treated_subgroup_n": 0,
        "n_extra_covariates": 5,
    }
    path.write_text(yaml.safe_dump(config))
    return path


def test_parse_cancer_types_defaults_to_all() -> None:
    assert _parse_cancer_types("all") == all_cancer_types()
    assert _parse_cancer_types("") == all_cancer_types()


def test_parse_cancer_types_subset_preserves_order_and_dedupes() -> None:
    assert _parse_cancer_types("crc, nsclc, crc") == [
        CancerType.crc,
        CancerType.nsclc,
    ]


def test_parse_cancer_types_rejects_unknown() -> None:
    import typer

    try:
        _parse_cancer_types("nsclc, melanoma")
    except typer.BadParameter as exc:
        assert "melanoma" in str(exc)
    else:
        raise AssertionError("expected BadParameter for unknown cancer type")


def test_default_cli_writes_all_five_cancer_types(tmp_path: Path) -> None:
    runner = CliRunner()
    cfg = _write_minimal_config(tmp_path / "synth.yaml")
    out_dir = tmp_path / "ds"

    result = runner.invoke(
        app,
        [
            "synth",
            "generate",
            "--config",
            str(cfg),
            "--out",
            str(out_dir),
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.stdout

    # Each cancer type gets its own subfolder containing named/+anonymized/.
    expected = {ct.value for ct in all_cancer_types()}
    assert {p.name for p in out_dir.iterdir() if p.is_dir()} == expected

    for ct in all_cancer_types():
        named = out_dir / ct.value / "named" / "manifest.json"
        anon = out_dir / ct.value / "anonymized" / "manifest.json"
        column_map = out_dir / ct.value / "anonymized" / "column_mapping.json"
        assert named.exists(), f"missing {named}"
        assert anon.exists(), f"missing {anon}"
        assert column_map.exists(), f"missing {column_map}"
        manifest = json.loads(named.read_text())
        assert manifest["cancer_type"] == ct.value
        assert manifest["dataset_id"] == f"ds_test_{ct.value}"


def test_subset_cli_writes_only_selected_cancer_types(tmp_path: Path) -> None:
    runner = CliRunner()
    cfg = _write_minimal_config(tmp_path / "synth.yaml")
    out_dir = tmp_path / "ds_subset"

    result = runner.invoke(
        app,
        [
            "synth",
            "generate",
            "--config",
            str(cfg),
            "--out",
            str(out_dir),
            "--cancer-types",
            "crc,breast",
        ],
    )
    assert result.exit_code == 0, result.stdout

    written = {p.name for p in out_dir.iterdir() if p.is_dir()}
    assert written == {"crc", "breast"}
    assert (out_dir / "crc" / "named" / "manifest.json").exists()
    assert (out_dir / "breast" / "anonymized" / "manifest.json").exists()


def test_cli_named_only_variant_writes_bundle_root_per_cancer(
    tmp_path: Path,
) -> None:
    """With --variant named, the bundle is written directly under the
    cancer-type folder (no extra named/ subdir)."""
    runner = CliRunner()
    cfg = _write_minimal_config(tmp_path / "synth.yaml")
    out_dir = tmp_path / "ds_named"

    result = runner.invoke(
        app,
        [
            "synth",
            "generate",
            "--config",
            str(cfg),
            "--out",
            str(out_dir),
            "--cancer-types",
            "nsclc",
            "--variant",
            "named",
        ],
    )
    assert result.exit_code == 0, result.stdout

    # Bundle root sits directly under <out>/<cancer_type>/.
    assert (out_dir / "nsclc" / "manifest.json").exists()
    assert (out_dir / "nsclc" / "public" / "dataset.parquet").exists()
    # And there is no separate named/ subdirectory.
    assert not (out_dir / "nsclc" / "named").exists()


def test_cli_harness_build_task_batches_synth_root(tmp_path: Path) -> None:
    """Pointing `ocs harness build-task --dataset` at a synth output root
    should produce one task bundle per discovered named/anonymized folder,
    mirroring the input tree under --out."""
    runner = CliRunner()
    cfg = _write_minimal_config(tmp_path / "synth.yaml", patient_n=60)
    synth_root = tmp_path / "ds"

    gen = runner.invoke(
        app,
        [
            "synth",
            "generate",
            "--config",
            str(cfg),
            "--out",
            str(synth_root),
            "--cancer-types",
            "crc,breast",
            "--seed",
            "0",
        ],
    )
    assert gen.exit_code == 0, gen.stdout

    tasks_root = tmp_path / "tasks"
    result = runner.invoke(
        app,
        [
            "harness",
            "build-task",
            "--dataset",
            str(synth_root),
            "--out",
            str(tasks_root),
            "--max-iterations",
            "3",
        ],
    )
    assert result.exit_code == 0, result.stdout

    for ct in ("crc", "breast"):
        for variant in ("named", "anonymized"):
            task_dir = tasks_root / ct / variant
            assert (task_dir / "agent_instructions.md").exists()
            assert (task_dir / "dataset.parquet").exists()
            assert not (task_dir / "manifest.json").exists()


def test_cli_harness_build_task_single_bundle_compat(tmp_path: Path) -> None:
    """Pointing --dataset at a single bundle still writes one task bundle
    directly into --out (backwards-compat with the per-bundle workflow)."""
    runner = CliRunner()
    cfg = _write_minimal_config(tmp_path / "synth.yaml", patient_n=60)
    synth_root = tmp_path / "ds"

    gen = runner.invoke(
        app,
        [
            "synth",
            "generate",
            "--config",
            str(cfg),
            "--out",
            str(synth_root),
            "--cancer-types",
            "nsclc",
        ],
    )
    assert gen.exit_code == 0, gen.stdout

    bundle = synth_root / "nsclc" / "anonymized"
    task_dir = tmp_path / "single_task"
    result = runner.invoke(
        app,
        [
            "harness",
            "build-task",
            "--dataset",
            str(bundle),
            "--out",
            str(task_dir),
            "--max-iterations",
            "2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (task_dir / "agent_instructions.md").exists()
    assert (task_dir / "dataset.parquet").exists()
    # And no nested per-cancer-type subdir was created.
    assert not (task_dir / "nsclc").exists()


def test_cli_harness_build_task_errors_on_empty_dir(tmp_path: Path) -> None:
    runner = CliRunner()
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(
        app,
        [
            "harness",
            "build-task",
            "--dataset",
            str(empty),
            "--out",
            str(tmp_path / "tasks"),
        ],
    )
    assert result.exit_code != 0


def test_cli_rejects_unknown_cancer_type(tmp_path: Path) -> None:
    runner = CliRunner()
    cfg = _write_minimal_config(tmp_path / "synth.yaml")
    out_dir = tmp_path / "bad"

    result = runner.invoke(
        app,
        [
            "synth",
            "generate",
            "--config",
            str(cfg),
            "--out",
            str(out_dir),
            "--cancer-types",
            "melanoma",
        ],
    )
    assert result.exit_code != 0
    assert "melanoma" in result.stdout or "melanoma" in (result.stderr or "")
