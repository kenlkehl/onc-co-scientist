from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from onc_co_scientist.cli import app
from onc_co_scientist.interventions import (
    VectorBundle,
    default_contrast_pairs,
    metadata_path_for,
    orthogonalize,
    read_contrast_pairs,
    write_contrast_pairs,
)
from onc_co_scientist.interventions.caa import parse_layers


def test_default_contrast_pairs_round_trip(tmp_path):
    pairs = default_contrast_pairs()
    assert {pair.concept for pair in pairs} == {
        "paradigm_adherence",
        "oncology_knowledge",
    }

    path = tmp_path / "pairs.jsonl"
    write_contrast_pairs(pairs, path)
    loaded = read_contrast_pairs(path)
    assert [pair.pair_id for pair in loaded] == [pair.pair_id for pair in pairs]


def test_orthogonalize_removes_projection():
    vector = np.array([1.0, 2.0], dtype=np.float32)
    against = np.array([1.0, 0.0], dtype=np.float32)
    out = orthogonalize(vector, against)
    assert np.allclose(out, np.array([0.0, 2.0], dtype=np.float32))
    assert np.isclose(float(np.dot(out, against)), 0.0)


def test_vector_bundle_save_load_and_orthogonalize(tmp_path):
    bundle = VectorBundle(
        vectors={
            "paradigm_adherence": {3: np.array([1.0, 1.0], dtype=np.float32)},
            "oncology_knowledge": {3: np.array([1.0, 0.0], dtype=np.float32)},
        },
        metadata={"model_id": "tiny"},
    ).with_orthogonalized()

    assert "paradigm_orthogonalized" in bundle.concepts()
    assert np.allclose(
        bundle.vector("paradigm_orthogonalized", 3),
        np.array([0.0, 1.0], dtype=np.float32),
    )

    path = tmp_path / "vectors.npz"
    bundle.save(path)
    assert metadata_path_for(path).exists()
    reloaded = VectorBundle.load(path)
    assert reloaded.layers_for("paradigm_orthogonalized") == [3]
    assert np.allclose(
        reloaded.vector("paradigm_orthogonalized", 3),
        np.array([0.0, 1.0], dtype=np.float32),
    )


def test_parse_layers():
    assert parse_layers("middle", n_layers=10) == [5]
    assert parse_layers("last:3", n_layers=10) == [7, 8, 9]
    assert parse_layers("3,1,3", n_layers=10) == [1, 3]


def test_caa_write_pairs_cli(tmp_path):
    runner = CliRunner()
    out = tmp_path / "pairs.jsonl"
    result = runner.invoke(app, ["caa", "write-pairs", "--out", str(out)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {row["concept"] for row in rows} == {
        "paradigm_adherence",
        "oncology_knowledge",
    }
