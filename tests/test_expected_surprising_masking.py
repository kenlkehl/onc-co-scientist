import json
from pathlib import Path

import pandas as pd
import pytest

from onc_co_scientist.expected_surprising.evaluation import WorkflowValidationService
from onc_co_scientist.expected_surprising.generation import sample, version_discoveries
from onc_co_scientist.expected_surprising.masking import (
    MaskedValidationService,
    SemanticMask,
    mask_package,
)
from onc_co_scientist.expected_surprising.schemas import Hypothesis, PairSpec, ValidationPolicy
from onc_co_scientist.expected_surprising.scoring import assign, comparison_key, estimate
from onc_co_scientist.synthetic.anonymize import build_value_mapping, mask_frame

SOURCE = Path(
    "data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/input_data"
)


def test_text_masking_missing_mixed_unused_and_column_scope():
    frame = pd.DataFrame(
        {
            "patient_id": ["a", "b", "c", "d"],
            "a": ["yes", None, "no", 1],
            "b": pd.Categorical(["yes", "no", None, "yes"], categories=["yes", "no", "unused"]),
        }
    )
    values = build_value_mapping(frame, seed=7)
    assert "patient_id" not in values
    assert values == build_value_mapping(frame.iloc[::-1], seed=7)
    masked = mask_frame(frame, {"a": "feature_1", "b": "feature_2"}, values)
    assert masked["feature_1"].iloc[3] == 1
    assert masked["feature_1"].isna().tolist() == frame["a"].isna().tolist()
    inverse_values = {c: {v: k for k, v in levels.items()} for c, levels in values.items()}
    restored = mask_frame(
        masked.rename(columns={"feature_1": "a", "feature_2": "b"}), {}, inverse_values
    )
    pd.testing.assert_frame_equal(restored, frame)
    with pytest.raises(ValueError, match="Unmapped"):
        mask_frame(pd.DataFrame({"a": ["new"]}), {}, values)


@pytest.fixture(scope="module")
def masked_package(tmp_path_factory):
    if not SOURCE.exists():
        pytest.skip("Frozen clinical pair unavailable")
    root = tmp_path_factory.mktemp("masked") / "package"
    mask_package(SOURCE, root)
    private = next((root / "private").iterdir())
    spec = PairSpec.model_validate_json((private / "pair.json").read_text())
    masking = SemanticMask(json.loads((private / "masking.json").read_text()))
    return root, private, spec, masking


@pytest.mark.parametrize("version", ["expected", "surprising"])
def test_pair_values_discoveries_validation_seeds_and_confirmation(masked_package, version):
    root, private, spec, masking = masked_package
    assignment = json.loads((private / "assignment.json").read_text())
    item = assignment[version]
    original = pd.read_parquet(SOURCE / "public" / item["source_task_id"] / "dataset.parquet")
    actual = pd.read_parquet(root / "public" / item["task_id"] / "dataset.parquet")
    pd.testing.assert_frame_equal(actual, masking.frame(original))
    assert len(masking.values) == 2
    assert (private / "pair.json").read_bytes() == (
        SOURCE / "private" / spec.pair_id / "pair.json"
    ).read_bytes()
    for name, levels in masking.values.items():
        assert set(actual[masking.columns[name]]) == set(levels.values())
        assert not set(actual[masking.columns[name]]) & set(levels)
    policy = ValidationPolicy.default(spec.profile, 25)
    named = WorkflowValidationService(spec, version, "replicate-001", policy)
    masked = MaskedValidationService(spec, version, "replicate-001", policy, masking)
    truths = version_discoveries(spec, version)
    named_h = [d.hypothesis for d in truths]
    # Include text eligibility and a non-target to test full-DGP adjudication.
    named_h += [
        Hypothesis(
            id="text",
            exposure="sex_female",
            outcome="log_pfs_months",
            direction=1,
            eligibility=[dict(variable="histology", value=next(iter(masking.values["histology"])))],
        )
    ]
    masked_h = [masking.hypothesis(h) for h in named_h]
    for i, (h, mh) in enumerate(zip(named_h, masked_h, strict=True)):
        assert masking.hypothesis(mh, inverse=True) == h
        a = estimate(original, h, delta=0.1, alpha=0.05, result_id="a")
        b = estimate(actual, mh, delta=0.1, alpha=0.05, result_id="a")
        assert a == b
        assert named.derive_seed("independent-validation", comparison_key(h)) == masked.derive_seed(
            "independent-validation", comparison_key(mh)
        )
        nr, _ = named.deliver(h, i + 1, "voluntary")
        mr, _ = masked.deliver(mh, i + 1, "voluntary")
        assert nr.model_dump(exclude={"id"}) == mr.model_dump(exclude={"id"})
    nconfirm, mconfirm = named.confirm(named_h), masked.confirm(masked_h)
    assert nconfirm == mconfirm
    assert assign(masked_h, version_discoveries(masking.spec(spec), version)) == assign(
        named_h, truths
    )
    # Independent samples, including text levels, retain exact numeric draws.
    pd.testing.assert_frame_equal(
        masked.sample_frame(123), masking.frame(sample(spec, version, seed=123))
    )


def test_package_private_map_and_public_nonleakage(masked_package):
    root, private, spec, masking = masked_package
    for public in (root / "public").iterdir():
        assert not any(
            "mapping" in p.name or p.name in {"pair.json", "workflow.json", "masking.json"}
            for p in public.iterdir()
        )
        text = "\n".join(p.read_text() for p in public.iterdir() if p.suffix in {".json", ".md"})
        assert not any(column in text for column in masking.columns)
        assert "10%" not in text or "clinically" in text
    with pytest.raises(FileExistsError):
        mask_package(SOURCE, root)


def test_masked_matrix_all_workflows_and_public_prompts(tmp_path, monkeypatch):
    import yaml

    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import run_experiment
    from tests.test_expected_surprising_experiment import config, providers, reports

    _, pairs, raw, config_path = config(tmp_path)
    mask_package(tmp_path / "packages", tmp_path / "opaque")
    private = next((tmp_path / "opaque/private").iterdir())
    masking = SemanticMask(json.loads((private / "masking.json").read_text()))
    raw["expected_surprising"]["root"] = "opaque"
    config_path.write_text(yaml.safe_dump(raw))
    made = providers(monkeypatch, masking.spec(pairs[0]))
    result = run_experiment(load_experiment_spec(config_path))
    assert result["n_completed"] == 6
    scientific = reports(result)
    assert all(r["successful_stages"] == 24 for r in scientific)
    for version in ("expected", "surprising"):
        group = [r for r in scientific if r["version"] == version]
        assert all(r["scores"] == group[0]["scores"] for r in group)
    for provider in made:
        text = "\n".join(m.content for call in provider.messages for m in call)
        assert not any(column in text for column in masking.columns)
        assert not any(
            value in text
            for levels in masking.values.values()
            for value in levels
            if len(value) > 5 and value != "current"
        )


def test_endpoint_gate_requires_complete_parent(tmp_path):
    from scripts.expected_surprising.run_masked_grid import parent_finished
    path = tmp_path / 'execution.json'
    assert not parent_finished(path)
    path.write_text('{')
    assert not parent_finished(path)
    state = dict(selected=['a', 'b'], finished=[{'run_id': 'a'}], worker_errors=[])
    path.write_text(json.dumps(state))
    assert not parent_finished(path)
    state['completed_at'] = '2026-09-10'
    path.write_text(json.dumps(state))
    assert not parent_finished(path)
    state['finished'].append({'run_id': 'b'})
    path.write_text(json.dumps(state))
    assert parent_finished(path)
