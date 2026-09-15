import copy
import json
from types import SimpleNamespace

import pytest

from onc_co_scientist.expected_surprising.coordination import StageCoordinator
from onc_co_scientist.expected_surprising.federated_history import FederatedHistoryCoordinator
from onc_co_scientist.expected_surprising.federation_spec import FederationGrid
from onc_co_scientist.providers.base import ChatMessage, ChatResponse
from onc_co_scientist.providers.federated_prompt import FederatedPromptLayout, extract, restore

SYSTEM = "You are a scientist on one federated site team. Current stage: analyze."


def payload(site="site_1"):
    return {
        "schema": {"required": ["assessment"]},
        "task": {"goal": "Find supported associations"},
        "claims": [
            {
                "ref": "H1",
                "comparison": {"exposure": "x", "outcome": "y"},
                "initial_expectation": {"anticipated_direction": 1},
                "current": {"status": "unresolved"},
                "evidence": [
                    {
                        "ref": "R1",
                        "estimate": -0.12345678901234,
                        "diagnostic": "aggregate",
                        "valid": True,
                    }
                ],
            }
        ],
        "federation": {"site": site, "local_analyses": {"H1": {"estimate": 0.5}}},
        "validation": {"remaining": 3},
        "research_notes": "Observe R1",
    }


def message(value):
    return ChatMessage(
        "user",
        "Iteration 1/25, stage analyze, attempt 1.\nResearch goal.\n"
        + json.dumps(value, separators=(",", ":"))
        + "\nRepair: keep all references.",
    )


def test_current_ledger_roundtrip_preserves_every_value_and_association():
    value = payload()
    value["claims"].append(copy.deepcopy(value["claims"][0]))
    value["claims"][1]["ref"] = "H2"  # Same evidence may support multiple claims.
    before = copy.deepcopy(value)
    assert restore(*extract(value)) == value
    assert value == before
    assert extract(value)[3]["claims"][1]["evidence_refs"] == ["R1"]


def test_changed_stage_state_and_peer_drafts_do_not_invalidate_reference_catalog():
    layout = FederatedPromptLayout("run-A")
    first, key, _ = layout.render([message(payload())], SYSTEM, "Stable controller instructions")
    changed = payload()
    changed["schema"] = {"required": ["narrative"]}
    changed["claims"][0]["current"]["status"] = "reject"
    changed["research_notes"] = "Reject H1 because R1 has opposite sign"
    changed["validation"]["remaining"] = 2
    system = SYSTEM.replace("analyze", "appraise") + "\nParticipant drafts: conflicting suggestions"
    second, key2, _ = layout.render([message(changed)], system, "Stable controller instructions")
    assert key2 == key
    assert first[:2] == second[:2]
    assert second[2]["role"] == "developer" and second[2]["content"][0]["text"] == system
    assert (
        json.loads(second[-1]["content"].split("Research goal.\n")[1].split("\nRepair:")[0])[
            "claims"
        ][0]["current"]["status"]
        == "reject"
    )
    assert second[-1]["content"].endswith("Repair: keep all references.")


def test_new_evidence_appends_but_removed_evidence_never_leaks_back():
    layout = FederatedPromptLayout("run")
    first, _, _ = layout.render([message(payload())], SYSTEM, "Stable")
    updated = payload()
    updated["claims"][0]["evidence"].append({"ref": "R2", "estimate": 0.3})
    second, _, _ = layout.render([message(updated)], SYSTEM, "Stable")
    a = "".join(b["text"] for b in first[1]["content"])
    b = "".join(b["text"] for b in second[1]["content"])
    assert b.startswith(a)
    third, _, _ = layout.render([message(payload())], SYSTEM, "Stable")
    assert "R2" not in "".join(b["text"] for b in third[1]["content"])


def test_scope_isolation_and_conflicting_reference_rejected():
    layout = FederatedPromptLayout("run")
    _, key, _ = layout.render([message(payload())], SYSTEM, "Stable")
    other = payload("site_2")
    other["claims"][0]["evidence"][0]["estimate"] = 0.8
    _, other_key, _ = layout.render([message(other)], SYSTEM, "Stable")
    assert key != other_key
    conflicting = payload()
    conflicting["claims"][0]["evidence"][0]["estimate"] = 0.9
    with pytest.raises(ValueError, match="changed meaning"):
        layout.render([message(conflicting)], SYSTEM, "Stable")
    with pytest.raises(ValueError, match="single-site"):
        layout.render([message(payload())], "You are a scientist.", "Stable")


def coordinator(cls, limit):
    return cls(
        SimpleNamespace(),
        SimpleNamespace(mode="persistent"),
        [],
        SimpleNamespace(persistent_history_chars=limit),
        SimpleNamespace(),
        None,
    )


def test_history_preserves_complete_decisions_despite_large_old_ledgers():
    new = coordinator(FederatedHistoryCoordinator, 3000)
    old = coordinator(StageCoordinator, 3000)
    answer = '{"narrative":"R1 contradicts H1; investigate the subgroup"}'
    for i in range(3):
        for c in (new, old):
            c.pending = (
                ChatMessage("user", "ledger " * 30000),
                ChatResponse(text=answer, model_id="test"),
                f"i{i + 1:03d}-appraise",
            )
            c.commit()
            c._bounded_history(f"i{i + 2:03d}-explore")
    assert old.history == []
    assert [m.content for m in new.history if m.role == "assistant"] == [answer] * 3
    assert "Historical iteration 1, stage appraise" in new.history[0].content
    assert "ledger ledger" not in "".join(m.content for m in new.history)


def test_newest_complete_response_survives_soft_limit():
    c = coordinator(FederatedHistoryCoordinator, 100)
    c.pending = (message(payload()), ChatResponse(text="x" * 500, model_id="test"), "i001-analyze")
    c.commit()
    assert c._bounded_history("i002-explore")[-1].content == "x" * 500
    assert c.memory_trims[-1]["newest_turn_exceeds_soft_limit"]


def test_legacy_serialization_and_one_site_are_unchanged():
    legacy = FederationGrid()
    assert legacy.model_dump() == {
        "site_counts": [1],
        "partitions": [{"id": "random", "mode": "random", "by": []}],
        "seed": 0,
        "orchestrator_model_profile": None,
    }
    assert "context_policy" not in legacy.cells()[0].model_dump()
    new = FederationGrid(site_counts=[1, 2], context_policy="federated_v2")
    one, two = new.cells()
    assert one.context_policy == "legacy" and "context_policy" not in one.model_dump()
    assert two.model_dump()["context_policy"] == "federated_v2"


def test_large_catalog_keeps_previous_breakpoint_on_append():
    layout = FederatedPromptLayout("run")
    first_payload = payload()
    first_payload["task"]["stable_reference"] = "Aggregate reference. " * 2000
    first, key, _ = layout.render([message(first_payload)], SYSTEM, "Stable")
    second_payload = copy.deepcopy(first_payload)
    second_payload["claims"][0]["evidence"].append({"ref": "R2", "estimate": 0.9})
    second, key2, _ = layout.render([message(second_payload)], SYSTEM, "Stable")
    assert key == key2
    assert second[1]["content"][: len(first[1]["content"])] == first[1]["content"]
    assert len([b for b in second[1]["content"] if "prompt_cache_breakpoint" in b]) <= 4


def test_history_only_commits_accepted_candidates_and_replay_is_deterministic():
    a = coordinator(FederatedHistoryCoordinator, 120000)
    b = coordinator(FederatedHistoryCoordinator, 120000)
    for c in (a, b):
        c.pending = (
            message(payload()),
            ChatResponse(text="rejected", model_id="test"),
            "i001-explore",
        )
        c.reject()
        assert c.history == []
        for i in range(3):
            c.pending = (
                message(payload()),
                ChatResponse(text=f"complete response {i}", model_id="test"),
                f"i{i + 1:03d}-appraise",
            )
            c.commit()
    assert a.history == b.history
    assert a.committed_stages == b.committed_stages


def test_cell_provider_uses_v2_only_for_multisite_opt_in(monkeypatch, tmp_path):
    from onc_co_scientist.expected_surprising import experiment
    from onc_co_scientist.expected_surprising.federation_spec import FederationCell
    from onc_co_scientist.providers.azure_federation import AzureFederationProvider

    sentinel = object()
    monkeypatch.setattr(experiment, "get_provider", lambda _: sentinel)
    config = dict(
        kind="codex_cli",
        backend="azure",
        model_id="gpt-5.6-terra",
        azure_endpoint="https://example.openai.azure.com/openai/v1",
        audit_dir=str(tmp_path / "audit"),
        budget_policy_path=str(tmp_path / "policy.json"),
    )
    for cell in (None, FederationCell(sites=1), FederationCell(sites=2)):
        assert experiment._cell_provider(config, SimpleNamespace(federation=cell)) is sentinel
    p = experiment._cell_provider(
        config, SimpleNamespace(federation=FederationCell(sites=2, context_policy="federated_v2"))
    )
    assert isinstance(p, AzureFederationProvider)
    assert p.transport_version == "azure-federation-context-v2"
    body = p.request_body([message(payload())], SYSTEM, 125000)
    assert body["max_output_tokens"] == 125000
    assert p.context_metadata["current_ledger_roundtrip"]


def test_evidence_direction_is_preserved_per_claim_not_collapsed_by_r_id():
    value = payload()
    reversed_claim = copy.deepcopy(value["claims"][0])
    reversed_claim["ref"] = "H2"
    reversed_claim["comparison"]["direction"] = -1
    reversed_claim["evidence"][0]["estimate"] *= -1
    value["claims"].append(reversed_claim)
    task, definitions, evidence, state = extract(value)
    assert evidence["H1", "R1"]["estimate"] == -evidence["H2", "R1"]["estimate"]
    assert restore(task, definitions, evidence, state) == value
