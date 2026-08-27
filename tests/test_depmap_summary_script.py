from scripts.summarize_depmap_metadata import _resolve_bundle_dir


def _stub_bundle(path):
    (path / "public").mkdir(parents=True)
    (path / "manifest.json").touch()
    (path / "public" / "dataset.parquet").touch()
    return path


def test_resolve_bundle_dir_accepts_single_variant_layout(tmp_path):
    bundle = _stub_bundle(tmp_path / "crc_depmap")

    assert _resolve_bundle_dir(tmp_path / "crc_depmap") == bundle


def test_resolve_bundle_dir_accepts_default_paired_layout(tmp_path):
    bundle = _stub_bundle(tmp_path / "crc_depmap" / "named")

    assert _resolve_bundle_dir(tmp_path / "crc_depmap") == bundle
