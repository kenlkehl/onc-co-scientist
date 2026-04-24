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
        variables=[
            "treatment_pembrolizumab",
            "egfr_mutation",
            "progression_free_months",
        ],
        outcome="progression_free_months",
        direction=1,
        effect_size=2.5,
        natural_language_description=(
            "Pembrolizumab is more effective in EGFR-mutant patients."
        ),
    )
    restored = AssociationSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec


def test_dataset_manifest_roundtrip_with_subgroup():
    spec = AssociationSpec(
        id="a2",
        paradigm_class=ParadigmClass.hidden_novel,
        form=AssociationForm.subgroup_conditional,
        variables=["treatment_olaparib", "brca2_mutation", "objective_response"],
        outcome="objective_response",
        direction=1,
        effect_size=2.0,
        subgroup=SubgroupSpec(
            name="brca2_mutant",
            predicate={"brca2_mutation": 1},
            description="BRCA2-mutant patients",
        ),
        natural_language_description=(
            "Olaparib works only in the BRCA2-mutant subgroup."
        ),
    )
    manifest = DatasetManifest(
        dataset_id="ds_test",
        seed=42,
        patient_n=100,
        columns=[
            "patient_id",
            "treatment_olaparib",
            "brca2_mutation",
            "objective_response",
        ],
        treatment_columns=["treatment_olaparib"],
        outcome_columns=["objective_response"],
        covariate_columns=["brca2_mutation"],
        associations=[spec],
    )
    restored = DatasetManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert restored.associations_by_class(ParadigmClass.hidden_novel) == [spec]
    assert restored.associations_by_class(ParadigmClass.concordant) == []
