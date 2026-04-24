from onc_open_mindedness.synthetic.generator import GeneratorConfig, generate_dataset
from onc_open_mindedness.synthetic.io import (
    DATASET_FILENAME,
    DESCRIPTION_FILENAME,
    MANIFEST_FILENAME,
    PUBLIC_SUBDIR,
    public_dir,
    read_description,
    read_frame,
    read_manifest,
    write_bundle,
)


def test_write_and_read_bundle_roundtrip(tmp_path):
    cfg = GeneratorConfig(
        dataset_id="ds_roundtrip",
        patient_n=80,
        seed=3,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
    )
    bundle = generate_dataset(cfg)
    out = write_bundle(bundle, tmp_path / "ds")
    manifest = read_manifest(out)
    frame = read_frame(out)
    desc = read_description(out)
    assert manifest == bundle.manifest
    assert list(frame.columns) == list(bundle.frame.columns)
    assert len(frame) == bundle.manifest.patient_n
    assert "Synthetic NSCLC dataset" in desc


def test_bundle_layout_isolates_manifest_from_agent_view(tmp_path):
    cfg = GeneratorConfig(
        dataset_id="ds_layout",
        patient_n=40,
        seed=1,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
    )
    bundle = generate_dataset(cfg)
    out = write_bundle(bundle, tmp_path / "ds")

    # Manifest sits at the top level only.
    assert (out / MANIFEST_FILENAME).exists()
    assert not (public_dir(out) / MANIFEST_FILENAME).exists()

    # Agent-safe files sit under public/ only.
    public = out / PUBLIC_SUBDIR
    assert public.is_dir()
    assert (public / DATASET_FILENAME).exists()
    assert (public / DESCRIPTION_FILENAME).exists()
    assert not (out / DATASET_FILENAME).exists()
    assert not (out / DESCRIPTION_FILENAME).exists()
