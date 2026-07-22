from __future__ import annotations

import sys
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "groupthink_pilot"
sys.path.insert(0, str(PILOT_DIR))

from pilot import (  # noqa: E402
    CONDITIONS,
    SCENARIOS,
    lineage_meta_analysis,
    revision_prompt,
    summarize_records,
)
from run_pilot import MACOS_CODEX, _resolve_codex_path  # noqa: E402


def _response(scenario_id: str, agent_id: str, condition: str, decision: str) -> dict:
    return {
        "scenario_id": scenario_id,
        "agent_id": agent_id,
        "round": "initial" if condition == "private" else "revision",
        "condition": condition,
        "decision": decision,
        "confidence": 80,
        "estimated_pooled_effect": -0.2 if decision == "benefit" else 0.2,
        "evidence_ids": ["E1"],
        "source_ids": ["source"],
        "reasoning_summary": "test",
        "minority_report": "",
        "changed_from_initial": False,
        "change_reason": "not_applicable" if condition == "private" else "no_change",
    }


def test_scenario_truth_is_balanced_and_executable() -> None:
    truths = []
    for scenario in SCENARIOS:
        pooled, pooled_se, truth = lineage_meta_analysis(scenario)
        assert abs(pooled) > 1.96 * pooled_se
        assert truth == scenario.expected_truth
        truths.append(truth)
    assert truths.count("benefit") == 2
    assert truths.count("harm") == 2


def test_prompts_isolate_social_signal_and_add_protocol() -> None:
    scenario = SCENARIOS[0]
    initial = {
        report.principal_id: _response(
            scenario.scenario_id,
            report.principal_id,
            "private",
            "harm" if report.principal_id != "P4" else "benefit",
        )
        for report in scenario.reports
    }
    evidence = revision_prompt(scenario, "P4", initial, "artifact_exchange", seed=1)
    social = revision_prompt(scenario, "P4", initial, "network_context", seed=1)
    protocol = revision_prompt(scenario, "P4", initial, "lineage_protocol", seed=1)
    assert "network dashboard" not in evidence
    assert "network dashboard" in social
    assert "Count independent source_id values" not in social
    assert "Count independent source_id values" in protocol


def test_codex_path_resolution_prefers_override_then_path(monkeypatch) -> None:
    override = Path("custom") / "codex"
    assert _resolve_codex_path(override) == override.resolve()

    discovered = Path("tools") / "codex"
    monkeypatch.setattr("run_pilot.shutil.which", lambda command: str(discovered))
    assert _resolve_codex_path() == discovered.resolve()

    monkeypatch.setattr("run_pilot.shutil.which", lambda command: None)
    assert _resolve_codex_path() == MACOS_CODEX


def test_summary_detects_false_consensus_and_protocol_rescue() -> None:
    records = []
    for scenario in SCENARIOS:
        truth = scenario.expected_truth
        wrong = "harm" if truth == "benefit" else "benefit"
        for report in scenario.reports:
            initial_decision = truth if report.principal_id == "P4" else wrong
            records.append(
                {
                    "replicate": 1,
                    "scenario_id": scenario.scenario_id,
                    "agent_id": report.principal_id,
                    "condition": "private",
                    "response": _response(
                        scenario.scenario_id, report.principal_id, "private", initial_decision
                    ),
                }
            )
            for condition in CONDITIONS:
                decision = wrong if condition == "network_context" else truth
                records.append(
                    {
                        "replicate": 1,
                        "scenario_id": scenario.scenario_id,
                        "agent_id": report.principal_id,
                        "condition": condition,
                        "response": _response(
                            scenario.scenario_id, report.principal_id, condition, decision
                        ),
                    }
                )
    summary = summarize_records(records)
    assert summary["condition_aggregates"]["network_context"]["false_consensus_networks"] == 4
    assert summary["condition_aggregates"]["artifact_exchange"]["false_consensus_networks"] == 0
    assert summary["paired_contrasts"]["protocol_rescue_vs_social"] == 16
    assert "observed a preliminary social-signal groupthink pattern" in summary["interpretation"]
