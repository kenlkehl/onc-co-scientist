from onc_open_mindedness.synthetic.generator import GeneratorConfig, generate_dataset
from onc_open_mindedness.synthetic.io import (
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
    assert "Synthetic oncology dataset" in desc
