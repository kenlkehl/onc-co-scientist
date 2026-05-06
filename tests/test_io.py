from onc_co_scientist.synthetic.cancer_types import CancerType
from onc_co_scientist.synthetic.generator import GeneratorConfig, generate_dataset
from onc_co_scientist.synthetic.io import (
    DATASET_FILENAME,
    DESCRIPTION_FILENAME,
    MANIFEST_FILENAME,
    PUBLIC_SUBDIR,
    discover_bundles,
    public_dir,
    read_description,
    read_frame,
    read_manifest,
    write_bundle,
)
from onc_co_scientist.synthetic.multi import (
    generate_multi_dataset,
    write_multi_bundle,
    write_multi_bundle_pair,
)


def test_write_and_read_bundle_roundtrip(tmp_path):
    cfg = GeneratorConfig(
        dataset_id="ds_roundtrip",
        patient_n=80,
        seed=3,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
        min_buried_treated_subgroup_n=0,
    )
    bundle = generate_dataset(cfg)
    out = write_bundle(bundle, tmp_path / "ds")
    manifest = read_manifest(out)
    frame = read_frame(out)
    desc = read_description(out)
    assert manifest == bundle.manifest
    assert list(frame.columns) == list(bundle.frame.columns)
    assert len(frame) == bundle.manifest.patient_n
    assert "Oncology patient cohort" in desc


def test_bundle_layout_isolates_manifest_from_agent_view(tmp_path):
    cfg = GeneratorConfig(
        dataset_id="ds_layout",
        patient_n=40,
        seed=1,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
        min_buried_treated_subgroup_n=0,
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


def test_discover_bundles_walks_paired_layout(tmp_path):
    """Both named/ and anonymized/ twins under each cancer type are found."""
    base = GeneratorConfig(
        dataset_id="ds_discover",
        patient_n=60,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
        min_buried_treated_subgroup_n=0,
        n_extra_covariates=5,
    )
    chosen = [CancerType.crc_clinical, CancerType.breast_clinical]
    bundles = generate_multi_dataset(base, chosen)
    write_multi_bundle_pair(bundles, tmp_path / "ds", anon_seed=0)

    found = discover_bundles(tmp_path / "ds")
    expected = sorted(
        tmp_path / "ds" / ct.value / variant for ct in chosen for variant in ("named", "anonymized")
    )
    assert found == expected


def test_discover_bundles_walks_single_variant_layout(tmp_path):
    """With variant=named/anonymized, the bundle sits directly under <ct>/."""
    base = GeneratorConfig(
        dataset_id="ds_single",
        patient_n=40,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
        min_buried_treated_subgroup_n=0,
        n_extra_covariates=4,
    )
    chosen = [CancerType.nsclc_clinical, CancerType.aml_clinical]
    bundles = generate_multi_dataset(base, chosen)
    write_multi_bundle(bundles, tmp_path / "ds", anonymize=False)

    found = discover_bundles(tmp_path / "ds")
    expected = sorted(tmp_path / "ds" / ct.value for ct in chosen)
    assert found == expected


def test_discover_bundles_returns_empty_for_non_synth_dir(tmp_path):
    (tmp_path / "unrelated.txt").write_text("hi")
    assert discover_bundles(tmp_path) == []
