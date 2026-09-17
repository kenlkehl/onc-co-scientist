"""Offline transport, cost admission and journal-preserving cancellation checks."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from onc_co_scientist.providers.azure_budget import (
    AzureBudget,
    ExperimentPaused,
    atomic_json,
    normalize_usage,
    usage_cost_micro,
)
from onc_co_scientist.providers.azure_federation import (
    AzureFederationConfig,
    AzureFederationProvider,
    AzureHTTPError,
    user_blocks,
)
from onc_co_scientist.providers.base import ChatMessage

MODEL = "gpt-5.6-sol"
RATES = {
    "short_context_tokens": 272000,
    "context_window": 1050000,
    "short": {"input": 4, "cached": 0.4, "write": 5, "output": 20},
    "long": {"input": 8, "cached": 0.8, "write": 10, "output": 30},
}


@pytest.fixture
def policy(tmp_path):
    release = tmp_path / "release.json"
    atomic_json(release, {"released_models": ["sol_medium"]})
    path = tmp_path / "spend_policy.json"
    atomic_json(
        path,
        {
            "enabled": True,
            "additional_budget_usd": 100,
            "release_policy_path": str(release),
            "rates_valid_until": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            "rates": {MODEL: RATES},
            "model_profiles": {MODEL: "sol_medium"},
            "max_unknown_attempts": 3,
            "max_reusable_prefix_misses": 2,
        },
    )
    atomic_json(tmp_path / "spend_state.json", {"spent_micro_usd": 0, "attempts": {}})
    return path


def make_provider(tmp_path, policy, **options):
    return AzureFederationProvider(
        AzureFederationConfig(
            model_id=MODEL,
            backend="azure",
            azure_endpoint="https://example.openai.azure.com/openai/v1",
            audit_dir=str(tmp_path / "audit"),
            budget_policy_path=str(policy),
            cache_namespace="federation/run/site",
            azure_transport_retry_s=0,
            **options,
        )
    )


def usage(cached=0, written=800):
    return {
        "input_tokens": 1000,
        "output_tokens": 100,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
        "output_tokens_details": {"reasoning_tokens": 30},
    }


def completed(cached=0, written=800):
    return {
        "status": "completed",
        "usage": usage(cached, written),
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"notes":"R1 supports H1"}'}],
            }
        ],
    }


def prompt(i=1):
    return (
        f"Iteration {i}/25, stage analyze, attempt 1.\n"
        + ("Research goal. " * 400)
        + '\n{"schema":{},"evidence":"R1"}'
    )


def mock_transport(monkeypatch, provider, outcomes):
    captures, tokens = [], []
    monkeypatch.setattr(provider, "request_slot", lambda *a: nullcontext(None))

    def env():
        tokens.append(len(tokens) + 1)
        return {"OCS_AZURE_ACCESS_TOKEN": f"fake-token-{tokens[-1]}"}

    def send(body, token, directory):
        captures.append((body, token))
        result = outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(provider, "environment", env)
    monkeypatch.setattr(provider, "_send", send)
    return captures, tokens


def test_cache_keys_boundaries_and_all_evidence_preserved(tmp_path, policy):
    p = make_provider(tmp_path, policy)
    system = "Research goal: use aggregate evidence only."
    old = ChatMessage("user", prompt())
    answer = ChatMessage("assistant", '{"claim":"H1","evidence":["R1"]}')
    first = p.request_body([old], system, 125000)
    second = p.request_body([old, answer, ChatMessage("user", prompt(2))], system, 125000)
    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert second["input"][: len(first["input"])] == first["input"]
    assert second["input"][-2] == {"role": "assistant", "content": answer.content}
    assert second["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert second["max_output_tokens"] == 125000
    assert first["input"][-1]["content"][0] == second["input"][-1]["content"][0]
    assert "prompt_cache_breakpoint" not in second["input"][-1]["content"][-1]
    blocks = user_blocks(prompt(2))
    payload, _, header = blocks[1]["text"].rpartition("\n")
    assert header + "\n" + blocks[0]["text"] + payload == prompt(2)
    assert user_blocks("opaque evidence R1") == [
        {"type": "input_text", "text": "opaque evidence R1"}
    ]
    assert p.cache_prefix(first) == p.cache_prefix(second)
    other = make_provider(tmp_path / "other", policy)
    object.__setattr__(other.config, "cache_namespace", "other/site")
    assert (
        other.request_body([old], system, 125000)["prompt_cache_key"] != first["prompt_cache_key"]
    )


def test_drafts_are_not_in_stable_key_or_explicit_system_prefix(tmp_path, policy):
    p = make_provider(tmp_path, policy)
    a = p.request_body([], "Chair\nParticipant drafts (untrusted suggestions):\nA", 100)
    b = p.request_body([], "Chair\nParticipant drafts (untrusted suggestions):\nB", 100)
    assert a["prompt_cache_key"] == b["prompt_cache_key"]
    assert a["input"][1]["content"][0] == b["input"][1]["content"][0]
    assert a["input"][1]["content"][1]["text"].endswith("A")


def test_expiry_retry_refreshes_token_and_accounts_once(tmp_path, policy, monkeypatch):
    p = make_provider(tmp_path, policy)
    calls, tokens = mock_transport(monkeypatch, p, [AzureHTTPError(401), completed()])
    result = p.chat([ChatMessage("user", prompt())])
    assert tokens == [1, 2]
    assert calls[0][0] == calls[1][0]
    assert result.raw["metrics"]["azure_auth_retries"] == 1
    state = json.loads((policy.parent / "spend_state.json").read_text())
    assert [a["status"] for a in state["attempts"].values()] == ["rejected", "settled"]
    assert state["spent_micro_usd"] == 6800
    assert "fake-token" not in "".join(x.read_text() for x in p.root.rglob("*.json"))


def test_unknown_attempt_retains_reservation_across_retry_and_restart(
    tmp_path, policy, monkeypatch
):
    p = make_provider(tmp_path, policy)
    mock_transport(monkeypatch, p, [TimeoutError(), completed()])
    p.chat([ChatMessage("user", "aggregate context")])
    state = json.loads((policy.parent / "spend_state.json").read_text())
    records = list(state["attempts"].values())
    assert [a["status"] for a in records] == ["unknown", "settled"]
    assert records[0]["reserved_micro_usd"] > state["spent_micro_usd"]
    gate = AzureBudget(policy)
    with gate.locked():
        assert gate.state["attempts"] == state["attempts"]


def test_missing_usage_and_cache_miss_hold_without_fabricated_science(
    tmp_path, policy, monkeypatch
):
    p = make_provider(tmp_path, policy)
    mock_transport(monkeypatch, p, [{"status": "completed", "usage": {}}])
    with pytest.raises(ExperimentPaused, match="usage"):
        p.chat([ChatMessage("user", "context")])
    assert not (p.root / "call-0001/metrics.json").exists()
    assert (
        next(
            iter(json.loads((policy.parent / "spend_state.json").read_text())["attempts"].values())
        )["status"]
        == "unknown"
    )
    mock_transport(monkeypatch, p, [completed(), completed(), completed()])
    for _ in range(3):
        p.chat([ChatMessage("user", prompt())])
    with pytest.raises(ExperimentPaused, match="cache reads"):
        p.chat([ChatMessage("user", prompt())])


def test_release_gate_prevents_auth_and_requests(tmp_path, policy, monkeypatch):
    p = make_provider(tmp_path, policy)
    atomic_json(tmp_path / "release.json", {"released_models": []})
    calls, tokens = mock_transport(monkeypatch, p, [])
    with pytest.raises(ExperimentPaused, match="release"):
        p.chat([ChatMessage("user", "hello")])
    assert not calls and not tokens
    assert not list(p.root.glob("call-*"))


def test_parallel_admission_does_not_overspend(tmp_path, policy):
    config = json.loads(policy.read_text())
    config["additional_budget_usd"] = 0.4
    atomic_json(policy, config)
    body = {"max_output_tokens": 100, "input": []}

    def reserve(n):
        try:
            return AzureBudget(policy).reserve(str(n), MODEL, body)
        except ExperimentPaused:
            return 0

    with ThreadPoolExecutor(max_workers=12) as pool:
        values = list(pool.map(reserve, range(20)))
    assert sum(v > 0 for v in values) == 1
    assert sum(values) <= 400000
    with pytest.raises(ExperimentPaused, match="duplicate"):
        AzureBudget(policy).reserve(str(values.index(max(values))), MODEL, body)


def test_cost_bands_and_reasoning_not_double_counted():
    u = normalize_usage(usage(cached=800, written=0))
    assert usage_cost_micro(u, RATES) == 3120
    u.update(input_tokens=300000, cached_input_tokens=100000, cache_write_input_tokens=100000)
    assert usage_cost_micro(u, RATES) == 1883000
    with pytest.raises(ValueError):
        normalize_usage(usage(cached=900, written=900))
    for field in ("input_tokens", "output_tokens"):
        broken = usage()
        broken[field] = None
        with pytest.raises(ValueError):
            normalize_usage(broken)


def test_cancellation_bypasses_scientific_repair_and_journal(tmp_path):
    from onc_co_scientist.expected_surprising.coordination import StageCoordinator

    class PausedProvider:
        model_id = MODEL

        def chat(self, *a, **k):
            raise ExperimentPaused("budget hold")

    c = StageCoordinator(
        PausedProvider(),
        None,
        [],
        SimpleNamespace(max_retries_per_stage=2, max_tokens_per_call=100),
        SimpleNamespace(max_agent_calls=100),
        tmp_path / "calls",
    )
    with pytest.raises(ExperimentPaused):
        c._call(
            "i001-analyze-a1",
            [ChatMessage("system", "goal"), ChatMessage("user", "task")],
            session="session",
            authoritative=True,
            iteration=1,
            stage="analyze",
            kind="linear",
        )
    assert not list((tmp_path / "calls").glob("*.json"))
    assert not c.records


def test_stream_wire_no_network(tmp_path, policy, monkeypatch):
    p = make_provider(tmp_path, policy)
    captured = {}

    class Connection:
        sock = None

        def __init__(self, host, port, timeout):
            captured["host"] = host

        def request(self, method, path, body, headers):
            captured.update(method=method, path=path, body=json.loads(body), headers=headers)

        def getresponse(self):
            r = io.BytesIO(
                (
                    "data: "
                    + json.dumps({"type": "response.completed", "response": completed()})
                    + "\n\n"
                ).encode()
            )
            r.status = 200
            r.getheader = lambda *a: None
            return r

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(
        "onc_co_scientist.providers.azure_federation.http.client.HTTPSConnection", Connection
    )
    body = p.request_body([ChatMessage("user", prompt())], "goal", 125000)
    directory = tmp_path / "stream"
    directory.mkdir()
    result = p._send(body, "fake-only", directory)
    assert result == completed()
    assert captured["path"] == "/openai/v1/responses"
    assert captured["body"] == body
    assert captured["headers"]["Authorization"] == "Bearer fake-only"
    assert captured["closed"]


def test_stable_schema_and_task_cached_but_ledger_and_iteration_not_cached(tmp_path, policy):
    p = make_provider(tmp_path, policy)

    def content(i):
        return f"Iteration {i}/25, stage appraise, attempt 1.\nResearch goal\n" + json.dumps(
            {
                "schema": {"properties": {"claims": {"type": "array"}}},
                "task": {"description": "aggregate only", "n": 123},
                "claims": [{"id": "H1", "evidence": f"R{i}"}],
            },
            separators=(",", ":"),
        )

    a, b = [user_blocks(content(i)) for i in (1, 2)]
    assert a[0] == b[0]
    assert '"n":123' in a[0]["text"] and '"id":"H1"' not in a[0]["text"]
    for i, blocks in enumerate((a, b), 1):
        tail, _, header = blocks[1]["text"].rpartition("\n")
        assert header + "\n" + blocks[0]["text"] + tail == content(i)
    assert p.request_body([ChatMessage("user", content(1))], "goal", 100)["store"] is False


@pytest.mark.parametrize("change", ["disabled", "nan", "negative_ledger", "expired", "zero_limit"])
def test_budget_fails_closed_on_invalid_or_disabled_policy(policy, change):
    data = json.loads(policy.read_text())
    if change == "disabled":
        data["enabled"] = False
    elif change == "nan":
        data["additional_budget_usd"] = "NaN"
    elif change == "expired":
        data["rates_valid_until"] = "2020-01-01T00:00:00+00:00"
    elif change == "zero_limit":
        data["max_unknown_attempts"] = 0
    else:
        atomic_json(policy.parent / "spend_state.json", {"spent_micro_usd": -1, "attempts": {}})
    atomic_json(policy, data)
    with pytest.raises(ExperimentPaused):
        AzureBudget(policy).reserve("never", MODEL, {"max_output_tokens": 100})


def test_429_releases_reservation_and_refreshes_token(tmp_path, policy, monkeypatch):
    p = make_provider(tmp_path, policy)
    calls, tokens = mock_transport(monkeypatch, p, [AzureHTTPError(429), completed()])
    p.chat([ChatMessage("user", "aggregates")])
    assert len(calls) == len(tokens) == 2
    attempts = json.loads((policy.parent / "spend_state.json").read_text())["attempts"]
    assert [a["status"] for a in attempts.values()] == ["rejected", "settled"]


def test_repeated_missing_receipts_latch_shared_hold(tmp_path, policy):
    gate = AzureBudget(policy)
    for n in range(3):
        gate.reserve(str(n), MODEL, {"max_output_tokens": 100})
        gate.unknown(str(n), "stream disconnected")
    with pytest.raises(ExperimentPaused, match="Unknown"):
        AzureBudget(policy).check(MODEL)


def test_incomplete_output_is_accounted_before_scientific_retry(tmp_path, policy, monkeypatch):
    p = make_provider(tmp_path, policy)
    result = completed()
    result["status"] = "incomplete"
    mock_transport(monkeypatch, p, [result])
    response = p.chat([ChatMessage("user", "aggregate context")])
    assert response.raw["adapter_error"] == "Output token limit reached"
    assert json.loads((policy.parent / "spend_state.json").read_text())["spent_micro_usd"] > 0


def test_completed_call_replay_does_not_touch_azure_or_budget(tmp_path):
    from onc_co_scientist.expected_surprising.coordination import StageCoordinator, digest

    class NoCalls:
        model_id = MODEL

        def chat(self, *a, **kw):
            raise AssertionError("Replay must not call the model")

    source = SimpleNamespace(max_retries_per_stage=2, max_tokens_per_call=100)
    c = StageCoordinator(
        NoCalls(), None, [], source, SimpleNamespace(max_agent_calls=100), tmp_path
    )
    messages = [ChatMessage("system", "goal"), ChatMessage("user", "evidence R1")]
    request = {
        "slot": "i001-analyze-a1",
        "session_id": "s",
        "iteration": 1,
        "stage": "analyze",
        "kind": "linear",
        "authoritative_candidate": True,
        "model": MODEL,
        "messages": [{"role": x.role, "content": x.content} for x in messages],
        "temperature": 0,
        "max_tokens": 100,
        "final_retry": False,
    }
    result = {"text": '{"evidence":["R1"]}', "error": None, "model_id": MODEL}
    path = tmp_path / "i001-analyze-a1.json"
    atomic_json(path, {"request": request, "result": result, "sha256": digest(result)})
    old = path.read_bytes()
    response = c._call(
        request["slot"],
        messages,
        session="s",
        authoritative=True,
        iteration=1,
        stage="analyze",
        kind="linear",
    )
    assert response.text == result["text"] and c.replayed_calls == 1
    assert path.read_bytes() == old


def test_v2_cache_guard_observes_eligible_usage_and_expiry(policy):
    gate = AzureBudget(policy)
    prefix = {"key": "actual-marked-prefix", "version": 2, "ttl_seconds": 1800}
    body = {"max_output_tokens": 100}

    def settle(name, cached=0, written=0):
        gate.reserve(name, MODEL, body)
        return gate.settle(
            name,
            dict(
                input_tokens=3000,
                output_tokens=10,
                cached_input_tokens=cached,
                cache_write_input_tokens=written,
                reasoning_output_tokens=0,
            ),
            prefix,
        )

    # An uncacheable request does not establish an expected hit.
    settle("short1")
    settle("short2")
    state_path = policy.with_name("spend_state.json")
    key = MODEL + ":v2:actual-marked-prefix"
    assert json.loads(state_path.read_text())["cache_prefixes"][key]["misses"] == 0
    settle("write", written=2000)
    settle("miss1", written=2000)
    with gate.locked():
        assert gate.state["cache_prefixes"][key]["misses"] == 1
        gate.state["cache_prefixes"][key]["last_eligible_at"] = (
            datetime.now(UTC) - timedelta(hours=1)
        ).isoformat()
        gate.save()
    settle("expired", written=2000)
    assert json.loads(state_path.read_text())["cache_prefixes"][key]["misses"] == 0
    settle("hit", cached=2000)
    settle("miss2", written=2000)
    settle("miss3", written=2000)
    state = json.loads(state_path.read_text())
    assert "within the cache TTL" in state["hold_reason"]
    assert state["attempts"]["miss3"]["status"] == "settled"
    assert state["attempts"]["miss3"]["cache_observation"]["misses"] == 2
    with pytest.raises(ExperimentPaused):
        gate.reserve("held", MODEL, body)


def test_uncached_transport_preserves_scientific_content(tmp_path, policy):
    cached = make_provider(tmp_path / "cached", policy)
    off = make_provider(tmp_path / "off", policy, prompt_caching=False)
    messages = [ChatMessage("user", prompt())]
    expected = cached.request_body(messages, "goal", 125000)
    actual = off.request_body(messages, "goal", 125000)
    for m in expected["input"]:
        if isinstance(m["content"], list):
            for b in m["content"]:
                b.pop("prompt_cache_breakpoint", None)
    assert actual == expected
    assert off.cache_prefix(actual) is None
    assert actual["prompt_cache_options"]["mode"] == "explicit"


@pytest.mark.parametrize("v2", [False, True])
def test_cache_observation_can_be_nonblocking_without_disabling_spend_guard(policy, v2):
    p = json.loads(policy.read_text())
    p["cache_miss_pause_enabled"] = False
    atomic_json(policy, p)
    gate = AzureBudget(policy)
    prefix = {"key": "stable", "version": 2, "ttl_seconds": 1800} if v2 else "stable"
    for i in range(8):
        gate.reserve(str(i), MODEL, {"max_output_tokens": 100, "input": []})
        u = normalize_usage(usage())
        u.update(input_tokens=2000, cache_write_input_tokens=1500)
        gate.settle(str(i), u, prefix)
    gate.check(MODEL)
    state = json.loads(gate.state_path.read_text())
    assert next(iter(state["cache_prefixes"].values()))["misses"] == 7
    p["additional_budget_usd"] = state["spent_micro_usd"] / 1_000_000
    atomic_json(policy, p)
    with pytest.raises(ExperimentPaused, match="exhausted"):
        gate.check(MODEL)


@pytest.mark.parametrize("partial", [False, True])
def test_streamed_rate_limit_retries_only_before_output(tmp_path, policy, monkeypatch, partial):
    p = make_provider(tmp_path, policy, azure_transport_retries=1)
    monkeypatch.setattr(p, "request_slot", lambda *args: nullcontext(None))
    tokens = []

    def environment():
        tokens.append(1)
        return {"OCS_AZURE_ACCESS_TOKEN": "fake-only"}

    monkeypatch.setattr(p, "environment", environment)
    failed = {
        "status": "failed",
        "error": {"code": "rate_limit_exceeded"},
        "usage": None,
        "output": [],
    }
    streams = [[{"type": "response.created", "response": {"output": []}}]]
    if partial:
        streams[0].append({"type": "response.output_text.delta", "delta": "partial"})
    streams[0] += [
        {"type": "error", "code": "rate_limit_exceeded"},
        {"type": "response.failed", "response": failed},
    ]
    streams.append([{"type": "response.completed", "response": completed()}])

    class Connection:
        sock = None

        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def getresponse(self):
            r = io.BytesIO(
                "".join("data: " + json.dumps(e) + "\n\n" for e in streams.pop(0)).encode()
            )
            r.status = 200
            r.getheader = lambda *args: None
            return r

    monkeypatch.setattr(
        "onc_co_scientist.providers.azure_federation.http.client.HTTPSConnection", Connection
    )
    if partial:
        with pytest.raises(ExperimentPaused, match="usage"):
            p.chat([ChatMessage("user", "aggregate context")])
    else:
        result = p.chat([ChatMessage("user", "aggregate context")])
        assert result.raw["metrics"]["azure_rate_limit_retries"] == 1
        assert len(tokens) == 2
    attempts = list(
        json.loads((policy.parent / "spend_state.json").read_text())["attempts"].values()
    )
    assert [a["status"] for a in attempts] == (["unknown"] if partial else ["rejected", "settled"])


@pytest.mark.parametrize("stop", [None, "retry_limit", "unknown_limit", "budget"])
@pytest.mark.parametrize("receipt", [None, {}])
def test_streamed_server_error_retains_cost_and_retries_bounded(
    tmp_path, policy, monkeypatch, stop, receipt
):
    config = json.loads(policy.read_text())
    if stop == "unknown_limit":
        config["max_unknown_attempts"] = 1
    if stop == "budget":
        config["additional_budget_usd"] = 0.4
    atomic_json(policy, config)
    p = make_provider(tmp_path, policy, azure_transport_retries=0 if stop == "retry_limit" else 1)
    monkeypatch.setattr(p, "request_slot", lambda *args: nullcontext(None))
    tokens, bodies, sleeps = [], [], []

    def environment():
        tokens.append(1)
        return {"OCS_AZURE_ACCESS_TOKEN": "fake-only"}

    monkeypatch.setattr(p, "environment", environment)
    monkeypatch.setattr("onc_co_scientist.providers.azure_federation.time.sleep", sleeps.append)
    failed = {
        "status": "failed",
        "error": {"code": "server_error"},
        "usage": receipt,
        "output": [{"type": "reasoning", "content": []}],
    }
    streams = [
        [
            {"type": "response.output_item.added", "item": {"type": "reasoning"}},
            {"type": "response.failed", "response": failed},
        ],
        [{"type": "response.completed", "response": completed()}],
    ]

    class Connection:
        sock = None

        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            bodies.append(kwargs["body"])

        def close(self):
            pass

        def getresponse(self):
            r = io.BytesIO(
                "".join("data: " + json.dumps(e) + "\n\n" for e in streams.pop(0)).encode()
            )
            r.status = 200
            r.getheader = lambda *args: None
            return r

    monkeypatch.setattr(
        "onc_co_scientist.providers.azure_federation.http.client.HTTPSConnection", Connection
    )
    if stop:
        with pytest.raises(ExperimentPaused):
            p.chat([ChatMessage("user", "aggregate context")])
    else:
        result = p.chat([ChatMessage("user", "aggregate context")])
        assert result.raw["metrics"]["azure_transport_retries"] == 1
        assert bodies[0] == bodies[1]
        assert len(tokens) == 2
        assert sleeps == [0]
    state = json.loads((policy.parent / "spend_state.json").read_text())
    attempts = list(state["attempts"].values())
    assert [a["status"] for a in attempts] == (["unknown"] if stop else ["unknown", "settled"])
    assert attempts[0]["reserved_micro_usd"] > 0
    assert len(bodies) == (1 if stop else 2)
    first = p.root / "call-0001/attempt-0001"
    assert json.loads((first / "response.json").read_text()) == failed
    assert json.loads((first / "stream_failure.json").read_text())["output_activity"]
