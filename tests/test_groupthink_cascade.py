import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "groupthink_cascade"
sys.path.insert(0, str(EXPERIMENT))

from cascade import (  # noqa: E402
    NETWORK_ORDERS,
    SCENARIOS,
    full_log_bayes_factor,
    network_id,
    private_log_bayes_factor,
    sign_choice,
    summarize_records,
)
from run_cascade import resolve_codex_executable  # noqa: E402


def test_mirrored_truth_and_private_split() -> None:
    for scenario in SCENARIOS:
        assert sign_choice(full_log_bayes_factor(scenario)) == scenario.expected_truth
        choices = [
            sign_choice(private_log_bayes_factor(scenario, packet.principal_id))
            for packet in scenario.private_packets
        ]
        assert choices.count(scenario.expected_truth) == 2


def test_orders_use_identical_evidence_multiset() -> None:
    expected = {packet.principal_id for packet in SCENARIOS[0].private_packets}
    for order in NETWORK_ORDERS.values():
        assert set(order) == expected


def test_codex_executable_override_takes_precedence(monkeypatch, tmp_path: Path) -> None:
    explicit = tmp_path / "custom-codex"
    monkeypatch.setattr("run_cascade.shutil.which", lambda _: str(tmp_path / "path-codex"))

    assert resolve_codex_executable(explicit) == explicit


def test_codex_executable_is_discovered_on_path(monkeypatch, tmp_path: Path) -> None:
    discovered = tmp_path / "path-codex"
    monkeypatch.setattr("run_cascade.shutil.which", lambda _: str(discovered))

    assert resolve_codex_executable() == discovered


def test_codex_executable_uses_existing_macos_fallback(monkeypatch, tmp_path: Path) -> None:
    fallback = tmp_path / "codex"
    fallback.touch()
    monkeypatch.setattr("run_cascade.shutil.which", lambda _: None)
    monkeypatch.setattr("run_cascade.MACOS_CODEX", fallback)

    assert resolve_codex_executable() == fallback


def _record(
    *,
    scenario_id: str,
    network: str,
    agent_id: str,
    stage: str,
    condition: str,
    choice: str,
) -> dict:
    return {
        "replicate": 1,
        "scenario_id": scenario_id,
        "network_id": network,
        "agent_id": agent_id,
        "stage": stage,
        "condition": condition,
        "model": "gpt-5.6-luna",
        "reasoning_effort": "low",
        "tool_events": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "response": {
            "scenario_id": scenario_id,
            "network_id": network,
            "agent_id": agent_id,
            "stage": stage,
            "condition": condition,
            "choice": choice,
            "confidence": 80,
            "estimated_log_bayes_factor": 0.0,
            "evidence_ids": [],
            "reasoning_summary": "test",
            "minority_report": "",
            "changed_from_private": False,
            "change_reason": "not_applicable" if stage != "chain" else "no_change",
        },
    }


def test_summary_detects_adverse_cascade() -> None:
    records = []
    for scenario in SCENARIOS:
        private_choice = {}
        for packet in scenario.private_packets:
            choice = sign_choice(private_log_bayes_factor(scenario, packet.principal_id))
            private_choice[packet.principal_id] = choice
            records.append(
                _record(
                    scenario_id=scenario.scenario_id,
                    network="private",
                    agent_id=packet.principal_id,
                    stage="private",
                    condition="private",
                    choice=choice,
                )
            )
        wrong = "H2" if scenario.expected_truth == "H1" else "H1"
        for order_name, order in NETWORK_ORDERS.items():
            net = network_id(scenario, order_name)
            for principal_id in order[1:]:
                choice = wrong if order_name == "adverse" else private_choice[principal_id]
                record = _record(
                    scenario_id=scenario.scenario_id,
                    network=net,
                    agent_id=principal_id,
                    stage="chain",
                    condition="verdict_chain",
                    choice=choice,
                )
                record["response"]["changed_from_private"] = choice != private_choice[principal_id]
                record["response"]["change_reason"] = (
                    "peer_verdicts" if record["response"]["changed_from_private"] else "no_change"
                )
                records.append(record)
            for condition in (
                "artifacts_only",
                "verdicts_only",
                "artifacts_plus_verdicts",
                "minority_protocol",
            ):
                records.append(
                    _record(
                        scenario_id=scenario.scenario_id,
                        network=net,
                        agent_id="chair",
                        stage="chair",
                        condition=condition,
                        choice=scenario.expected_truth,
                    )
                )
    summary = summarize_records(records)
    assert summary["chain_aggregate"]["cascade_initiations"] == 4
    assert summary["chain_aggregate"]["cascade_lock_ins"] == 2
