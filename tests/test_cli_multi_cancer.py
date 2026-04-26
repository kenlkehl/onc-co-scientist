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
