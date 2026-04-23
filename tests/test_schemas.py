from onc_open_mindedness.synthetic.schemas import (
    AssociationForm,
    AssociationSpec,
    DatasetManifest,
    ParadigmClass,
    SubgroupSpec,
)


def test_association_spec_roundtrip():
    spec = AssociationSpec(
        id="a1",
        paradigm_class=ParadigmClass.discordant,
        form=AssociationForm.interaction,
        variables=["treatment_io", "egfr_mutation", "progression_free_months"],
        outcome="progression_free_months",
        direction=1,
        effect_size=2.5,
        natural_language_description="IO is more effective in EGFR-mutant patients.",
    )
    restored = AssociationSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec


def test_dataset_manifest_roundtrip_with_subgroup():
    spec = AssociationSpec(
        id="a2",
        paradigm_class=ParadigmClass.hidden_novel,
        form=AssociationForm.subgroup_conditional,
        variables=["treatment_x", "biomarker_z_high", "objective_response"],
        outcome="objective_response",
        direction=1,
        effect_size=2.0,
        subgroup=SubgroupSpec(
            name="bz_high",
            predicate={"biomarker_z_high": 1},
            description="High biomarker Z",
        ),
        natural_language_description="Drug X works only in biomarker-Z-high subgroup.",
    )
    manifest = DatasetManifest(
        dataset_id="ds_test",
        seed=42,
        patient_n=100,
        columns=["patient_id", "treatment_x", "biomarker_z_high", "objective_response"],
        treatment_columns=["treatment_x"],
        outcome_columns=["objective_response"],
        covariate_columns=["biomarker_z_high"],
        associations=[spec],
    )
    restored = DatasetManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert restored.associations_by_class(ParadigmClass.hidden_novel) == [spec]
    assert restored.associations_by_class(ParadigmClass.concordant) == []
