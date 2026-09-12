"""Offline contracts for external coordination of native, site-local Biomni."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from onc_co_scientist.expected_surprising.packaging import sha256
from onc_co_scientist.external.federation import NativeFederationSpec, NativeSite, SiteGateway
from onc_co_scientist.external.federation_runner import execute, run_cell
from onc_co_scientist.external.runner import task_masking, validate_inputs
from onc_co_scientist.external.transport import RequestBudget, RunBroker, write_json
from tests.test_expected_surprising_experiment import config


class NativeFixture:
    """Model-free stand-in at the native worker boundary, not a replacement harness."""

    def __init__(self, proposal):
        self.proposal = proposal
        self.dispatches = []
        self.frames = {}

    def run(self, spec, public, scratch, config_path, broker):
        settings = json.loads(config_path.read_text())
        site = public.parent.name
        self.frames[site] = pd.read_parquet(public / "dataset.parquet")
        self.dispatches.append((site, settings["run_id"], settings["continuation"]))
        assert settings["exchange_doc"].startswith("Exchange with your site coordinator")
        assert settings["require_namespace_hash"]
        context = broker.gateway.context
        handoff = {"narrative": "Local native research completed."}
        if broker.gateway.round == 1 and context["phase"] == "research":
            handoff["proposals"] = [self.proposal]
        broker.exchange(
            dict(
                request_id="site-handoff",
                round=broker.gateway.round,
                action="handoff",
                payload=handoff,
            )
        )
        namespace = scratch / "native_namespace.pkl"
        namespace.write_bytes(b"opaque-native-namespace-for-fixture-only")
        write_json(
            scratch / "native_checkpoint.json",
            dict(
                resumable=True,
                dispatch_id=settings["run_id"],
                namespace_sha256=sha256(namespace),
                llm_requests=len(broker.requests),
                exchange_count=broker.exchange_count,
            ),
        )
        return 0


class CentralFixture:
    def __init__(self):
        self.steps = Counter()
        self.prompts = []

    def complete(self, body):
        prompt = body["messages"][0]["content"]
        self.prompts.append(prompt)
        payload = json.loads(prompt.rsplit("\n", 1)[1])
        state = payload["ledger"]
        if payload["schema"]["title"] == "CentralDirection":
            response = {"goal": "Investigate the shared question", "site_instructions": {}}
        else:
            number, sealed = state["round"], state["sealed"]
            key = (number, sealed)
            step = self.steps[key]
            self.steps[key] += 1
            if sealed:
                actions = ["assess", "close"]
            elif number == 1:
                actions = ["register", "analyze", "assess", "validate", "assess", "prepare_close"]
            else:
                actions = ["assess", "prepare_close"]
            action = actions[step]
            data = {}
            if action == "register":
                data = {"proposals": payload["handoffs"]["site_1"]["proposals"]}
            if action == "analyze":
                data = {"run_analyses": ["H1"]}
            if action == "validate":
                data = {"validate": "H1"}
            if action == "assess":
                data = {
                    "assessments": [
                        dict(claim=row["ref"], status="unresolved", investigation="active")
                        for row in state["claims"]
                    ]
                }
            response = dict(request_id="central-request", round=number, action=action, payload=data)
        return {"choices": [{"message": {"content": json.dumps(response)}}]}


@pytest.mark.parametrize("sites,masked", [(2, False), (4, False), (2, True), (4, True)])
def test_full_native_rounds_isolation_aggregation_and_terminal_reuse(
    tmp_path, monkeypatch, sites, masked
):
    matrix, _, _, _ = config(tmp_path)
    source = matrix.expected_surprising.root
    if masked:
        from onc_co_scientist.expected_surprising.masking import mask_package

        target = tmp_path / "masked"
        mask_package(source, target)
        source = target
    spec = NativeFederationSpec(
        input_root=source,
        output_root=tmp_path / "federation",
        sites=sites,
        rounds=6,
        pair_id=matrix.tasks[0].metadata["pair_id"],
    )
    tasks, pair, policy = validate_inputs(spec)
    task = tasks[0]
    hypothesis = pair.discoveries[0].hypothesis
    masking = task_masking(task)
    if masking:
        hypothesis = masking.hypothesis(hypothesis)
    proposal = {
        **hypothesis.model_dump(exclude={"id"}),
        "anticipated_direction": hypothesis.direction,
        "initial_status": "unresolved",
    }
    native, central = NativeFixture(proposal), CentralFixture()
    monkeypatch.setattr(RunBroker, "start", lambda self: "http://fixture.invalid")
    monkeypatch.setattr(RunBroker, "stop", lambda self: None)
    result = run_cell(
        spec,
        task,
        pair,
        policy,
        1,
        "test-fingerprint",
        runner=native,
        central_factory=lambda *args: central,
    )
    assert result["status"] == "completed", result.get("error")
    assert result["completed_rounds"] == 6
    assert len(native.dispatches) == sites * 6 * 2
    assert sum(not continuation for _, _, continuation in native.dispatches) == sites
    source_frame = pd.read_parquet(task.public_workspace / "dataset.parquet")
    assert sum(len(f) for f in native.frames.values()) == len(source_frame)
    id_column = "patient_id"
    memberships = [set(f[id_column]) for f in native.frames.values()]
    assert len(set.union(*memberships)) == sum(map(len, memberships))
    text = "\n".join(central.prompts)
    assert str(source_frame[id_column].iloc[0]) not in text
    for private in ('"discoveries":', '"delta":', '"expected_direction":'):
        assert private not in text
    root = spec.output_root / "runs" / result["run_id"]
    report = json.loads((root / "report.json").read_text())
    assert report["clock"] == "reporting_round"
    assert report["federation"]["condition"]["sites"] == sites
    assert set(report["scoped_usage"]) == {"central", *native.frames}
    calls = len(native.dispatches), len(central.prompts)
    again = run_cell(
        spec,
        task,
        pair,
        policy,
        1,
        "test-fingerprint",
        resume=True,
        runner=native,
        central_factory=lambda *args: central,
    )
    assert again == result
    assert (len(native.dispatches), len(central.prompts)) == calls
    # Reconstruct from completed dispatches and central journals after an interrupted finalization.
    write_json(
        root / "run.json",
        dict(run_id=result["run_id"], fingerprint="test-fingerprint", status="running"),
    )
    replay = run_cell(
        spec,
        task,
        pair,
        policy,
        1,
        "test-fingerprint",
        resume=True,
        runner=native,
        central_factory=lambda *args: central,
    )
    assert replay["status"] == "completed", replay.get("error")
    assert (len(native.dispatches), len(central.prompts)) == calls


def test_sites_cannot_spend_scientific_budgets_or_export_arbitrary_fields():
    gateway = SiteGateway(1, {"ledger": {"claims": []}})
    for action in ("register", "analyze", "validate", "assess", "close"):
        with pytest.raises(ValueError):
            gateway.exchange(dict(request_id=action, round=1, action=action, payload={}))
    with pytest.raises(ValueError):
        gateway.exchange(
            dict(request_id="raw", round=1, action="handoff", payload={"rows": [[1, 2]]})
        )
    request = dict(
        request_id="one", round=1, action="handoff", payload={"narrative": "aggregate review"}
    )
    assert gateway.exchange(request) == gateway.exchange(request)
    with pytest.raises(ValueError, match="reused"):
        gateway.exchange({**request, "payload": {}})
    with pytest.raises(ValueError, match="One handoff"):
        gateway.exchange({**request, "request_id": "two"})


def test_shared_budget_is_atomic_across_sites_and_central():
    budget = RequestBudget(7)

    def reserve(_):
        try:
            budget.reserve()
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(reserve, range(30)))
    assert sum(results) == budget.used == 7


def test_shared_budget_counts_failed_requests(tmp_path, monkeypatch):
    from onc_co_scientist.external import transport

    spec = NativeFederationSpec(input_root=tmp_path, output_root=tmp_path / "out")
    budget = RequestBudget(1)
    brokers = [
        RunBroker(spec, None, tmp_path / s, "fixture", shared_budget=budget)
        for s in ("central", "site_1")
    ]

    def fail(*args, **kwargs):
        raise TimeoutError("offline failure")

    monkeypatch.setattr(transport, "http_json", fail)
    with pytest.raises(TimeoutError):
        brokers[0].complete({"messages": []})
    with pytest.raises(ValueError, match="shared_request_budget"):
        brokers[1].complete({"messages": []})
    assert budget.used == 1 and len(brokers[0].requests) == 1 and not brokers[1].requests


@pytest.mark.parametrize("pilot", [True, False])
def test_held_gate_precedes_preflight_and_all_model_calls(tmp_path, monkeypatch, pilot):
    from onc_co_scientist.external import federation_runner

    root = tmp_path / "biomni"
    root.mkdir()
    write_json(tmp_path / "release_policy.json", {"released_models": ["sol_medium"]})

    def forbidden(*args, **kwargs):
        pytest.fail("Held arm reached preflight or live calls")

    monkeypatch.setattr(federation_runner, "preflight", forbidden)
    with pytest.raises(ValueError, match="Biomni is held"):
        execute(root, pilot=pilot)
    assert list(root.iterdir()) == []


def test_native_unsafe_checkpoint_fails_without_running_code(tmp_path):
    spec = NativeFederationSpec(input_root=tmp_path, output_root=tmp_path / "out")

    class Forbidden:
        def run(self, *args):
            pytest.fail("Unsafe checkpoint executed native code")

    site = NativeSite(
        spec,
        "site_1",
        tmp_path / "public",
        tmp_path / "site",
        RequestBudget(5),
        runner=Forbidden(),
        resume=True,
    )
    write_json(site.scratch / "native_checkpoint.json", {"resumable": False})
    with pytest.raises(ValueError, match="safe native namespace"):
        site.dispatch(1, "research", {})


def test_cached_central_malformed_json_does_not_make_new_requests(tmp_path):
    from onc_co_scientist.external.federation import NativeFederation

    class InvalidCentral:
        calls = 0

        def complete(self, body):
            self.calls += 1
            return {"choices": [{"message": {"content": "not json"}}]}

    coordinator = object.__new__(NativeFederation)
    coordinator.root, coordinator.central = tmp_path, InvalidCentral()
    for _ in range(2):
        with pytest.raises(ValueError):
            coordinator.ask("bad", {}, {})
    assert coordinator.central.calls == 1
