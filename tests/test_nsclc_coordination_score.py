from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "nsclc_coordination"
sys.path.insert(0, str(EXPERIMENT))

from score_experiment import (  # noqa: E402
    load_truth,
    score_artifact,
    score_experiment,
    score_run,
)


def _claim(
    claim_id: str,
    predicates: dict[str, int],
    *,
    direction: str = "positive",
    supported: bool = True,
) -> dict:
    return {
        "claim_id": claim_id,
        "text": "Sotorasib improves PFS in the specified subgroup.",
        "exposure": "treatment_sotorasib",
        "outcome": "pfs_months",
        "direction": direction,
        "subgroup_predicates": predicates,
        "supported": supported,
        "analysis_ids": [f"a-{claim_id}"],
    }


def _analysis(claim_id: str) -> dict:
    return {
        "analysis_id": f"a-{claim_id}",
        "claim_ids": [claim_id],
        "effect_estimate": 4.985,
        "p_value": 0.001,
        "significant": True,
        "subgroup_n": 3266,
        "exposed_n": 1154,
        "comparator_n": 2112,
        "evidence": ["analysis.csv#sotorasib-subgroup"],
    }


EXACT_PREDICATES = {
    "kras_g12c": 1,
    "alk_fusion": 0,
    "brca2_mutation": 0,
    "sex_female": 0,
}


def _record(
    call: int,
    stage: str,
    artifact: dict,
    *,
    round_index: int = 1,
    iteration_index: int = 1,
) -> dict:
    artifact = dict(artifact)
    artifact.setdefault(
        "final_answer",
        ({"conclusion": "final", "supported_claim_indices": [0]}
         if stage.removesuffix("_consensus") == "synthesis"
         else None),
    )
    return {
        "request_id": f"run:c{call:04d}",
        "stage_id": stage,
        "agent_id": f"agent-{call}",
        "session_id": f"session-{call}",
        "site_id": None,
        "round": round_index,
        "iteration_index": iteration_index,
        "max_iterations": iteration_index,
        "artifact": artifact,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "tool_calls": 1,
            "cost_usd": 0.01,
            "duration_seconds": 1.0,
        },
    }


def _artifact(claim_id: str, predicates: dict[str, int]) -> dict:
    return {
        "summary": "structured result",
        "handoff": "structured handoff",
        "claims": [_claim(claim_id, predicates)],
        "analyses": [_analysis(claim_id)],
        "evidence": ["analysis table"],
    }


def test_structured_claims_score_exact_near_component_and_evidence_support() -> None:
    exact = score_artifact(_artifact("exact", EXACT_PREDICATES))
    near = score_artifact(
        _artifact(
            "near",
            {key: value for key, value in EXACT_PREDICATES.items() if key != "sex_female"},
        )
    )
    component = score_artifact(_artifact("component", {"kras_g12c": 1}))
    wrong_sign = score_artifact(
        {
            **_artifact("wrong", EXACT_PREDICATES),
            "analyses": [
                {
                    **_analysis("wrong"),
                    "effect_estimate": -4.985,
                }
            ],
        }
    )

    assert exact["recovery_level"] == "exact"
    assert exact["supported_recovery_level"] == "exact"
    assert near["recovery_level"] == "near"
    assert component["recovery_level"] == "component"
    assert wrong_sign["recovery_level"] == "exact"
    assert wrong_sign["supported_recovery_level"] == "none"


def test_current_strict_claim_shape_scores_without_legacy_analysis_link() -> None:
    strict_claim = {
        "exposure": "treatment_sotorasib",
        "outcome": "pfs_months",
        "direction": "positive",
        "subgroup": [
            {"variable": variable, "operator": "eq", "value": value}
            for variable, value in EXACT_PREDICATES.items()
        ],
        "comparator": "no sotorasib in the same subgroup",
        "effect_estimate": 4.985,
        "effect_unit": "months",
        "p_value": 0.001,
        "subgroup_n": 3266,
        "exposed_n": 1154,
        "comparator_n": 2112,
        "supported": True,
        "confidence": 0.98,
        "evidence": ["analysis.csv#strict-claim"],
    }
    score = score_artifact(
        {
            "summary": "strict artifact",
            "handoff": "strict handoff",
            "claims": [strict_claim],
            "analyses": [],
        }
    )

    assert score["recovery_level"] == "exact"
    assert score["supported_recovery_level"] == "exact"

    strict_claim["subgroup"][1]["operator"] = "ne"
    contradictory = score_artifact(
        {
            "summary": "strict artifact",
            "handoff": "strict handoff",
            "claims": [strict_claim],
            "analyses": [],
        }
    )
    assert contradictory["recovery_level"] == "none"


def test_positive_binary_exposure_encoding_is_canonicalized() -> None:
    artifact = _artifact("encoded-exposure", EXACT_PREDICATES)
    artifact["claims"][0]["exposure"] = "treatment_sotorasib=1"
    encoded = score_artifact(artifact)
    assert encoded["recovery_level"] == "exact"

    artifact["claims"][0]["exposure"] = "treatment_sotorasib=0"
    unexposed = score_artifact(artifact)
    assert unexposed["recovery_level"] == "none"


def test_text_fallback_is_conservative() -> None:
    statement = (
        "Sotorasib produced 4.985 months longer PFS (p<0.001; n=3266; exposed_n=1154; "
        "comparator_n=2112) in KRAS G12C-mutant, ALK-wild-type, BRCA2-wild-type male "
        "patients; this finding was supported."
    )
    exact = score_artifact(
        {
            "summary": statement,
            "handoff": statement,
            "hypotheses": [],
            "analyses": [],
            "evidence": [
                "analysis.csv: mean difference +4.985 months; p<0.001; n=3266; "
                "exposed_n=1154; comparator_n=2112"
            ],
        }
    )
    unsupported = score_artifact(
        {
            "summary": "Sotorasib may improve PFS in KRAS G12C-mutant patients.",
            "handoff": "hypothesis only",
            "hypotheses": [],
            "analyses": [],
            "evidence": [],
        }
    )

    assert exact["recovery_level"] == "exact"
    assert exact["supported_recovery_level"] == "exact"
    assert unsupported["recovery_level"] == "component"
    assert unsupported["supported_recovery_level"] == "none"


def test_support_gate_rejects_missing_n_and_orphaned_analysis() -> None:
    missing_n_analysis = _analysis("missing-n")
    missing_n_analysis.pop("subgroup_n")
    missing_n = score_artifact(
        {
            "summary": "structured result",
            "handoff": "structured handoff",
            "claims": [_claim("missing-n", EXACT_PREDICATES)],
            "analyses": [missing_n_analysis],
        }
    )
    orphan = score_artifact(
        {
            "summary": "structured result",
            "handoff": "structured handoff",
            "claims": [_claim("target", EXACT_PREDICATES)],
            "analyses": [_analysis("different-claim")],
        }
    )

    assert missing_n["recovery_level"] == "exact"
    assert missing_n["supported_recovery_level"] == "none"
    assert orphan["recovery_level"] == "exact"
    assert orphan["supported_recovery_level"] == "none"


def test_deliberative_uses_consensus_checkpoints_and_scores_survival() -> None:
    artifacts = [
        _record(1, "hypothesis_generation", _artifact("peer-exact", EXACT_PREDICATES)),
        _record(
            2,
            "hypothesis_generation_consensus",
            {"summary": "none", "handoff": "none"},
            round_index=3,
        ),
        _record(3, "analysis", _artifact("peer-analysis", EXACT_PREDICATES)),
        _record(4, "analysis_consensus", _artifact("analysis", EXACT_PREDICATES), round_index=3),
        _record(5, "critique_consensus", _artifact("critique", EXACT_PREDICATES), round_index=3),
        _record(6, "synthesis_consensus", _artifact("final", EXACT_PREDICATES), round_index=3),
    ]
    run = {
        "run_id": "delib-run",
        "workflow_id": "deliberative",
        "workflow_mode": "deliberative",
        "agent_calls": 6,
        "status": "completed",
    }
    score = score_run(run, artifacts)

    assert [item["stage_id"] for item in score["checkpoint_scores"]] == [
        "hypothesis_generation_consensus",
        "analysis_consensus",
        "critique_consensus",
        "synthesis_consensus",
    ]
    assert score["time_to_recovery"] == {
        "event": True,
        "checkpoint": 2,
        "cap": 4,
        "censored": False,
        "stage_id": "analysis_consensus",
        "call_index": 4,
    }
    assert score["analysis_to_final_survival"] is True
    assert score["discovery_loss"] is False
    assert score["usage"]["tokens_to_first_recovery"] == 600


def test_critique_rescue_and_analysis_loss_are_explicit_denominators() -> None:
    null = {"summary": "No target found.", "handoff": "No target found."}
    rescue_artifacts = [
        _record(1, "hypothesis_generation", null),
        _record(2, "analysis", null),
        _record(3, "critique", _artifact("rescued", EXACT_PREDICATES)),
        _record(4, "synthesis", _artifact("final", EXACT_PREDICATES)),
    ]
    rescue = score_run(
        {"run_id": "rescue", "workflow_id": "sequential", "workflow_mode": "sequential"},
        rescue_artifacts,
    )
    assert rescue["critique_rescue_eligible"] is True
    assert rescue["critique_rescue"] is True
    assert rescue["post_critique_rescue"] is True
    assert rescue["analysis_to_final_survival"] is None

    loss_artifacts = [
        _record(1, "hypothesis_generation", null),
        _record(2, "analysis", _artifact("found", EXACT_PREDICATES)),
        _record(3, "critique", null),
        _record(4, "synthesis", null),
    ]
    loss = score_run(
        {"run_id": "loss", "workflow_id": "sequential", "workflow_mode": "sequential"},
        loss_artifacts,
    )
    assert loss["analysis_to_final_survival_eligible"] is True
    assert loss["analysis_to_final_survival"] is False
    assert loss["discovery_loss"] is True


def test_near_counts_for_transition_but_not_primary_final_exact() -> None:
    near_predicates = {key: value for key, value in EXACT_PREDICATES.items() if key != "sex_female"}
    artifacts = [
        _record(1, "hypothesis_generation", {"summary": "none", "handoff": "none"}),
        _record(2, "analysis", _artifact("analysis-near", near_predicates)),
        _record(3, "critique", _artifact("critique-near", near_predicates)),
        _record(4, "synthesis", _artifact("final-near", near_predicates)),
    ]
    score = score_run(
        {"run_id": "near", "workflow_id": "persistent", "workflow_mode": "persistent"},
        artifacts,
    )

    assert score["final_supported_recovery_level"] == "near"
    assert score["final_supported_exact"] is False
    assert score["final_supported_near_or_exact"] is True
    assert score["analysis_to_final_survival"] is True
    assert score["usage"]["supported_final_exact_per_1k_tokens"] == 0.0
    assert score["usage"]["supported_final_near_or_exact_per_1k_tokens"] > 0


def test_experiment_writes_deterministic_json_csv_and_markdown(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    run_dir = result_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    run = {
        "run_id": "run-1",
        "workflow_id": "persistent",
        "workflow_mode": "persistent",
        "agent_calls": 4,
        "status": "completed",
    }
    artifacts = [
        _record(1, "hypothesis_generation", {"summary": "none", "handoff": "none"}),
        _record(2, "analysis", _artifact("analysis", EXACT_PREDICATES)),
        _record(3, "critique", _artifact("critique", EXACT_PREDICATES)),
        _record(4, "synthesis", _artifact("final", EXACT_PREDICATES)),
    ]
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts), encoding="utf-8")

    out = tmp_path / "score"
    first = score_experiment([result_root], out_dir=out)
    first_json = (out / "aggregate.json").read_text(encoding="utf-8")
    second = score_experiment([result_root], out_dir=out)

    assert first == second
    assert first_json == (out / "aggregate.json").read_text(encoding="utf-8")
    assert first["by_workflow"][0]["analysis_to_final_survival_rate"] == 1.0
    assert first["by_workflow"][0]["final_supported_exact_rate"] == 1.0
    assert first["by_workflow"][0]["supported_final_exact_per_1k_tokens"] > 0
    assert (out / "aggregate.csv").is_file()
    assert (out / "run_scores.csv").is_file()
    assert (out / "iteration_scores.csv").is_file()
    assert (out / "cell_summary.csv").is_file()
    assert (out / "resource_summary.csv").is_file()
    assert "No model judge was used" in (out / "report.md").read_text(encoding="utf-8")


def test_masked_truth_scores_opaque_exact_near_component_contradiction_and_text() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "example_data_clinical_all_claude"
        / "ds001"
        / "nsclc"
    )
    truth = load_truth(
        root / "anonymized" / "manifest.json",
        semantic_condition="masked",
        column_mapping_path=root / "anonymized" / "column_mapping.json",
    )
    exact_predicates = {
        "feature_016": 1,
        "feature_028": 0,
        "feature_005": 0,
        "feature_031": 0,
    }

    def masked_artifact(predicates: dict[str, int]) -> dict:
        return {
            "summary": "opaque result",
            "handoff": "opaque handoff",
            "claims": [
                {
                    "exposure": "feature_018",
                    "outcome": "pfs_months",
                    "direction": "positive",
                    "subgroup": [
                        {"variable": variable, "operator": "eq", "value": value}
                        for variable, value in predicates.items()
                    ],
                    "effect_estimate": 4.985,
                    "p_value": 0.001,
                    "subgroup_n": 3266,
                    "exposed_n": 1154,
                    "comparator_n": 2112,
                    "supported": True,
                    "evidence": ["opaque-analysis.csv"],
                }
            ],
        }

    exact = score_artifact(masked_artifact(exact_predicates), truth=truth)
    near = score_artifact(
        masked_artifact(
            {
                key: value
                for key, value in exact_predicates.items()
                if key != "feature_031"
            }
        ),
        truth=truth,
    )
    component = score_artifact(masked_artifact({"feature_016": 1}), truth=truth)
    contradiction = score_artifact(
        masked_artifact({**exact_predicates, "feature_028": 1}), truth=truth
    )
    statement = (
        "feature_018 increases pfs_months by +4.985 months (p<0.001; n=3266; "
        "exposed_n=1154; comparator_n=2112) when feature_016=1, feature_028=0, "
        "feature_005=0, and feature_031=0; supported."
    )
    text_score = score_artifact(
        {
            "summary": statement,
            "handoff": statement,
            "evidence": [statement],
        },
        truth=truth,
    )

    assert exact["supported_recovery_level"] == "exact"
    assert near["supported_recovery_level"] == "near"
    assert component["supported_recovery_level"] == "component"
    assert contradiction["recovery_level"] == "none"
    assert text_score["supported_recovery_level"] == "exact"
    assert truth.exposure == "feature_018"
    assert set(truth.subgroup_predicates) == set(exact_predicates)


def test_ever_recovery_is_not_substituted_for_terminal_iteration() -> None:
    null = {"summary": "none", "handoff": "none"}
    artifacts = [
        _record(1, "hypothesis_generation", null, iteration_index=1),
        _record(2, "analysis", null, iteration_index=1),
        _record(3, "critique", null, iteration_index=1),
        _record(4, "synthesis", _artifact("found", EXACT_PREDICATES), iteration_index=1),
        _record(5, "hypothesis_generation", null, iteration_index=2),
        _record(6, "analysis", null, iteration_index=2),
        _record(7, "critique", null, iteration_index=2),
        _record(8, "synthesis", null, iteration_index=2),
    ]
    score = score_run(
        {
            "run_id": "trajectory",
            "workflow_id": "sequential",
            "workflow_mode": "sequential",
            "semantic_condition": "named",
            "iteration_policy": {"iterations": 2, "completion_mode": "fixed"},
            "iterations_completed": 2,
            "terminal_iteration": 2,
            "status": "completed",
        },
        artifacts,
    )

    assert score["ever_supported_exact_synthesis"] is True
    assert score["terminal_supported_exact"] is False
    assert score["final_supported_recovery_level"] == "none"
    assert score["first_supported_exact_synthesis_iteration"] == 1
    assert score["later_exact_loss"] is True
    assert score["exact_persisted_to_terminal"] is False


def test_aggregation_keeps_all_six_semantic_workflow_cells_separate(tmp_path: Path) -> None:
    source_root = (
        Path(__file__).resolve().parents[1]
        / "example_data_clinical_all_claude"
        / "ds001"
        / "nsclc"
    )
    result_root = tmp_path / "main-results"
    index = {
        "schema_version": "1",
        "tasks": {
            "named-task": {
                "semantic_condition": "named",
                "path": str(source_root / "named" / "manifest.json"),
                "sha256": "a844619fceb456a5ef4d9b5ba3dff5e7f07363eb83554226601f845ed22ce064",
            },
            "masked-task": {
                "semantic_condition": "masked",
                "path": str(source_root / "anonymized" / "manifest.json"),
                "sha256": "e78bb273b53647fdc41af4d0e7b34ffbb0d0dec86b0e09b78e689be764ca0409",
            },
        },
        "assets": {
            "column_mapping": {
                "path": str(source_root / "anonymized" / "column_mapping.json"),
                "sha256": "6d291a3d653803b12ad150654635201b079c6dd37e09fd7037fd0bcbf08d9cd7",
            }
        },
    }
    result_root.mkdir(parents=True)
    (result_root / "private_evaluation_index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    workflows = ("persistent", "sequential", "deliberative")
    for condition, task_id in (("named", "named-task"), ("masked", "masked-task")):
        truth = load_truth(
            Path(index["tasks"][task_id]["path"]),
            semantic_condition=condition,
            column_mapping_path=Path(index["assets"]["column_mapping"]["path"]),
        )
        predicates = truth.subgroup_predicates
        for workflow in workflows:
            run_id = f"{condition}-{workflow}"
            run_dir = result_root / "runs" / run_id
            run_dir.mkdir(parents=True)
            run = {
                "run_id": run_id,
                "task_id": task_id,
                "semantic_condition": condition,
                "workflow_id": workflow,
                "workflow_mode": workflow,
                "replicate": 1,
                "agent_calls": 4,
                "status": "completed",
            }
            claim = {
                "summary": "result",
                "handoff": "handoff",
                "claims": [
                    {
                        "exposure": truth.exposure,
                        "outcome": truth.outcome,
                        "direction": "positive",
                        "subgroup": [
                            {"variable": key, "operator": "eq", "value": value}
                            for key, value in predicates.items()
                        ],
                        "effect_estimate": 4.9,
                        "p_value": 0.001,
                        "subgroup_n": 3266,
                        "exposed_n": 1154,
                        "comparator_n": 2112,
                        "supported": True,
                        "evidence": ["analysis.csv"],
                    }
                ],
            }
            stages = ["hypothesis_generation", "analysis", "critique", "synthesis"]
            artifacts = [
                _record(
                    call,
                    stage,
                    (
                        claim
                        if stage == "synthesis"
                        else {"summary": "none", "handoff": "none"}
                    ),
                )
                for call, stage in enumerate(stages, start=1)
            ]
            if workflow == "deliberative":
                for artifact in artifacts:
                    artifact["stage_id"] += "_consensus"
            (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
            (run_dir / "artifacts.json").write_text(json.dumps(artifacts), encoding="utf-8")

    out = tmp_path / "scored"
    aggregate = score_experiment([result_root], out_dir=out)

    assert aggregate["n_runs"] == 6
    assert {
        (row["semantic_condition"], row["workflow_id"]) for row in aggregate["by_cell"]
    } == {
        (condition, workflow)
        for condition in ("named", "masked")
        for workflow in workflows
    }
    assert all(row["n_runs"] == 1 for row in aggregate["by_cell"])
    assert len((out / "cell_summary.csv").read_text(encoding="utf-8").splitlines()) == 7
