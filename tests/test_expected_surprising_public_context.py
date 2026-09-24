"""Regression coverage for equivalent named/masked measurement context."""

import copy
import json

import pandas as pd
import pytest

from onc_co_scientist.expected_surprising.generation import write_pair
from onc_co_scientist.expected_surprising.masking import SemanticMask, mask_package
from onc_co_scientist.expected_surprising.packaging import repackage
from onc_co_scientist.expected_surprising.prompting import public_outcomes
from onc_co_scientist.expected_surprising.schemas import ValidationPolicy, WorkflowStageRecord
from onc_co_scientist.expected_surprising.workflow import run_workflow
from onc_co_scientist.external.runner import task_prompt
from onc_co_scientist.providers.base import ChatResponse
from tests.test_expected_surprising import attest, pair


@pytest.fixture(params=["nsclc_clinical", "nsclc_depmap"])
def packages(request, tmp_path):
    spec = pair(request.param)
    # A non-default cutoff catches modality constants and stale public metadata.
    spec.outcomes[0].delta = 0.234
    attest(spec)
    write_pair(spec, tmp_path / "original")
    manifest = repackage(tmp_path / "original", tmp_path / "named")
    mask_package(tmp_path / "named", tmp_path / "masked")
    private = tmp_path / "masked/private" / spec.pair_id
    masking = SemanticMask(json.loads((private / "masking.json").read_text()))
    assignment = json.loads((private / "assignment.json").read_text())
    return spec, manifest, masking, assignment


def test_packages_preserve_identifiers_assay_context_thresholds_and_data(packages, tmp_path):
    spec, manifest, masking, assignment = packages
    identifier = "cell_line_id" if spec.profile.endswith("depmap") else "patient_id"
    assert identifier not in masking.columns
    assert identifier not in masking.values
    assert masking.payload["version"] == "columns-and-text-levels-v2"
    for item in manifest["tasks"]:
        named = tmp_path / "named/public" / item["task_id"]
        masked = tmp_path / "masked/public" / assignment[item["version"]]["task_id"]
        original = pd.read_parquet(named / "dataset.parquet")
        actual = pd.read_parquet(masked / "dataset.parquet")
        pd.testing.assert_frame_equal(actual, masking.frame(original))
        pd.testing.assert_series_equal(actual[identifier], original[identifier])

        named_task = json.loads((named / "task.json").read_text())
        masked_task = json.loads((masked / "task.json").read_text())
        assert named_task["outcomes"] == public_outcomes(spec.outcomes)
        assert masked_task["outcomes"] == public_outcomes(masking.spec(spec).outcomes)
        assert (
            named_task["versions"]["prompt"] == masked_task["versions"]["prompt"] == "ledger-1.1.0"
        )
        instructions = (named / "instructions.md").read_text()
        for name, alias in sorted(masking.columns.items(), key=lambda x: -len(x[0])):
            instructions = instructions.replace(name, alias)
        actual_instructions = (masked / "instructions.md").read_text()
        assert actual_instructions.split("\nPredictor names and text categorical values")[0] == (
            instructions
        )
        for column in original:
            if column.startswith(("research_signature_", "research_marker_")):
                assert masking.columns[column] in actual_instructions
                assert column not in actual_instructions
        assert "standardized assays in arbitrary units" in actual_instructions
        assert (
            "binary assays with no assigned gene, pathway, or clinical role" in actual_instructions
        )


@pytest.mark.parametrize("old_identifier_alias", [False, True])
@pytest.mark.parametrize("legacy_interface", [False, True])
def test_runtime_filters_ids_and_uses_evaluator_thresholds(
    packages, tmp_path, old_identifier_alias, legacy_interface
):
    spec, _, masking, assignment = packages
    public = tmp_path / "masked/public" / assignment["expected"]["task_id"]
    identifier = "cell_line_id" if spec.profile.endswith("depmap") else "patient_id"
    if old_identifier_alias:
        payload = copy.deepcopy(masking.payload)
        payload["version"] = "columns-and-text-levels-v1"
        payload["columns"][identifier] = "feature_old_row_id"
        masking = SemanticMask(payload)
        frame = pd.read_parquet(public / "dataset.parquet")
        frame.rename(columns={identifier: "feature_old_row_id"}).to_parquet(
            public / "dataset.parquet", index=False
        )
    task = json.loads((public / "task.json").read_text())
    for outcome in task["outcomes"]:
        outcome.pop("effect_size_threshold")  # Historical packages lacked this field.
    if legacy_interface:
        task["versions"].update(
            workflow="appraisal-3.1.0", prompt="exploration-3.1.0", schema_version="stage-3.0.0"
        )
    (public / "task.json").write_text(json.dumps(task))

    class Scientist:
        model_id = "context-regression"
        prompts = []

        def chat(self, messages, **kwargs):
            prompt = messages[0].content
            self.prompts.append(prompt)
            stage = prompt.splitlines()[0].split()[3].rstrip(",")
            text = (
                WorkflowStageRecord(iteration=1, stage=stage).model_dump_json()
                if legacy_interface
                else "{}"
            )
            return ChatResponse(text=text, model_id=self.model_id)

    scientist = Scientist()
    report = run_workflow(
        spec,
        "expected",
        public,
        tmp_path / "run",
        scientist,
        run_id="context-regression",
        iterations=1,
        policy=ValidationPolicy(release_iterations=[]),
        masking=masking.payload,
    )
    assert report["successful_stages"] == 4
    assert not report["protocol_errors"]
    for prompt in scientist.prompts:
        assert identifier not in prompt and "feature_old_row_id" not in prompt
        assert "10%" not in prompt
        payload = json.loads(prompt[prompt.index('{"schema":') :])
        visible_task = payload["history"][0]["task"] if legacy_interface else payload["task"]
        assert visible_task["outcomes"] == public_outcomes(masking.spec(spec).outcomes)
        if not legacy_interface:
            for outcome in visible_task["outcomes"]:
                assert f"{outcome['name']}: {outcome['effect_size_threshold']:g}" in prompt
            assert "absolute differences on the supplied outcome scale" in prompt


def test_native_prompts_retain_the_same_assay_description_and_thresholds(packages):
    spec, _, masking, _ = packages
    named = task_prompt(25, outcomes=public_outcomes(spec.outcomes))
    masked = task_prompt(25, outcomes=public_outcomes(masking.spec(spec).outcomes), masked=True)
    for prompt in (named, masked):
        assert "constructed assays with no assigned clinical or biological role" in prompt
        assert "0.234" in prompt
        assert "10%" not in prompt
        assert "absolute differences on the supplied outcome scale" in prompt
    for name, alias in sorted(masking.columns.items(), key=lambda x: -len(x[0])):
        named = named.replace(name, alias)
    assert masked.split("\nPredictor names and text categorical values")[0] == named
