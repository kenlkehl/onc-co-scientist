"""Long-response and fail-fast regression tests; no real weights or GPU launches."""

import asyncio
import io
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import yaml

from onc_co_scientist.caa_server import CAAInferenceEngine, create_app
from onc_co_scientist.expected_surprising.workflow import WorkflowInfrastructureError
from onc_co_scientist.interventions import VectorBundle, caa
from onc_co_scientist.providers.base import ChatMessage, ProviderInfrastructureError
from onc_co_scientist.providers.caa_openai import CAAConfig, CAAProvider
from tests.test_caa_discovery import campaign
from tests.test_caa_discovery import frozen_campaign as campaign_fixture
from tests.test_expected_surprising_experiment import Scientist, providers


@pytest.fixture(name="frozen_campaign")
def frozen_campaign_fixture(tmp_path):
    return campaign_fixture.__wrapped__(tmp_path)


@pytest.mark.parametrize("smoke", [True, False])
def test_iteration_mode_never_reduces_response_budget(frozen_campaign, tmp_path, smoke):
    root, _, _ = frozen_campaign
    out = tmp_path / "second"
    base = yaml.safe_load((tmp_path / "matrix.yaml").read_text())
    base["iteration_policy"]["iterations"] = 25
    base["budget"]["max_agent_calls"] = 900
    base_path = tmp_path / "full.yaml"
    base_path.write_text(yaml.safe_dump(base))
    frozen = campaign.prepare(
        out=out, named=root / "unmasked/input_data", masked=root / "masked/input_data",
        base_config=base_path, server_manifest=root / "server_manifest.json",
        arms=["gemma4-control", "gemma4-caa"], replicates=1, smoke=smoke,
        rationale="long-response contract",
    )
    raw = yaml.safe_load((out / "unmasked/config.yaml").read_text())
    assert raw["expected_surprising"]["max_tokens_per_call"] == 100000
    assert raw["iteration_policy"]["iterations"] == (6 if smoke else 25)
    assert raw["budget"]["max_runtime_seconds_per_call"] == 21600
    assert raw["models"][0]["provider_config"]["timeout_s"] == 21600
    assert raw["models"][0]["provider_config"]["max_retries"] == 0
    assert frozen["max_tokens_per_call"] == 100000


@pytest.mark.parametrize("status", [400, 500, 503, "timeout", "connection"])
def test_caa_sdk_never_retries_infrastructure_failures(status, monkeypatch):
    from openai import OpenAI

    calls = []

    def transport(request):
        calls.append(json.loads(request.content))
        if status == "timeout":
            raise httpx.ReadTimeout("test timeout", request=request)
        if status == "connection":
            raise httpx.ConnectError("test disconnect", request=request)
        return httpx.Response(status, json={"error": {"message": "CUDA out of memory"}})

    original = OpenAI
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        monkeypatch.setattr("openai.OpenAI", lambda **kw: original(http_client=client, **kw))
        provider = CAAProvider(CAAConfig(model_id="gemma4-control", server_fingerprint="a" * 64))
        assert provider._client.max_retries == 0
        with pytest.raises(ProviderInfrastructureError):
            provider.chat([ChatMessage("user", "test")], max_tokens=100000)
        assert len(calls) == 1
        assert calls[0]["max_tokens"] == 100000
    with pytest.raises(ValueError, match="retries"):
        CAAConfig(model_id="x", server_fingerprint="a" * 64, max_retries=2)


@pytest.mark.parametrize("field,value", [
    ("caa_provenance", ["invalid"]), ("usage", ["invalid"]),
    ("caa_provenance", {"fingerprint": "a" * 64, "arm": ["invalid"]}),
])
def test_malformed_provenance_is_infrastructure_not_scientific_error(monkeypatch, field, value):
    raw = {"caa_provenance": {"fingerprint": "a" * 64, "arm": {"model_id": "gemma4-control"}},
           "usage": {"prompt_tokens": 1, "completion_tokens": 2}, field: value}
    monkeypatch.setattr(CAAProvider, "_build_client", lambda self: None)
    monkeypatch.setattr("onc_co_scientist.providers.vllm_openai.VLLMProvider.chat",
                        lambda *a, **kw: SimpleNamespace(raw=raw))
    provider = CAAProvider(CAAConfig(model_id="gemma4-control", server_fingerprint="a" * 64))
    with pytest.raises(ProviderInfrastructureError):
        provider.chat([])


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses"])
def test_generation_failure_latches_server_and_health(tmp_path, monkeypatch, endpoint):
    engine = CAAInferenceEngine(model_path="fixture", vector_file=tmp_path / "x.npz")
    engine.model, engine.processor = object(), object()
    engine.manifest = {"fingerprint": "a" * 64}
    calls = []

    def oom(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(caa, "generate_messages_unsteered", oom)
    payload = {"model": "gemma4-31b-control", "messages": [{"role": "user", "content": "x"}],
               "input": "x", "max_tokens": 100000}
    class Request:
        async def json(self):
            return payload

    # Invoke the real async route handlers without sockets or ASGI worker threads.
    routes = {r.path: r.endpoint for r in create_app(engine, load_on_startup=False).routes}
    for _ in range(2):
        response = asyncio.run(routes[endpoint](Request()))
        assert response.status_code == 503
        assert response.headers["x-should-retry"] == "false"
        assert json.loads(response.body)["error"]["type"] == "caa_inference_unavailable"
    assert routes["/health"]().status_code == 503
    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == 100000
    assert calls[0]["prefill_chunk_size"] == 2048


def test_real_controller_persists_fatal_call_and_halts_matrix(frozen_campaign, monkeypatch):
    root, _, pair = frozen_campaign

    class Broken(Scientist):
        def chat(self, messages, **kwargs):
            self.messages.append(messages)
            assert kwargs["max_tokens"] == 100000
            raise ProviderInfrastructureError("CUDA out of memory")

    made = providers(monkeypatch, pair, Broken)
    manifest = (root / "server_manifest.json").read_bytes()
    monkeypatch.setattr(campaign.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(manifest))
    with pytest.raises(WorkflowInfrastructureError, match="out of memory"):
        campaign.run(root, limit=8)
    assert len(made) == 1 and len(made[0].messages) == 1
    calls = list(root.glob("*/runs/*/calls/*.json"))
    assert len(calls) == 1
    assert json.loads(calls[0].read_text())["result"]["infrastructure_error"] is True
    assert campaign.read(root / "halt.json")["failure_category"] == "infrastructure"
    # Cache replay is also fatal (no duplicate provider call), never a repair.
    owner = calls[0].parents[3]
    spec = campaign.load_experiment_spec(owner / "config.yaml")
    # Test replay directly at the coordinator to avoid terminal-run short circuit.
    from onc_co_scientist.expected_surprising.coordination import StageCoordinator
    coordinator = StageCoordinator(made[0], spec.workflows[0], spec.stages,
                                   spec.expected_surprising, spec.budget, calls[0].parent)
    request = campaign.read(calls[0])["request"]
    with pytest.raises(WorkflowInfrastructureError, match="out of memory"):
        coordinator._call(request["slot"], [ChatMessage(**m) for m in request["messages"]],
                          session=request["session_id"], authoritative=True,
                          iteration=request["iteration"], stage=request["stage"],
                          kind=request["kind"])
    assert len(made[0].messages) == 1
    with pytest.raises(RuntimeError, match="Campaign halted"):
        campaign.run(root)
    assert len(list(root.glob("*/runs/*/run.json"))) == 1
    monkeypatch.syspath_prepend(str(campaign.ROOT / "scripts/expected_surprising"))
    summary = campaign.summarize(root)
    assert all(r["value"] is None for r in summary["finished"]["cross_condition"])


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_default_one_cell_gate_and_failed_cell_stop(frozen_campaign, monkeypatch, status):
    root, _, _ = frozen_campaign
    manifest = (root / "server_manifest.json").read_bytes()
    monkeypatch.setattr(campaign.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(manifest))
    calls = []

    def run_cell(spec, plan, owner, *args, **kwargs):
        calls.append(plan.run_id)
        result = {"run_id": plan.run_id, "status": status}
        campaign.write(owner / "runs" / plan.run_id / "run.json", result)
        return result

    monkeypatch.setattr(campaign, "run_cell", run_cell)
    if status == "completed":
        campaign.run(root)
        assert len(calls) == 1
        campaign.run(root)  # Skip the first terminal success; advance by one only.
        assert len(calls) == len(set(calls)) == 2
    else:
        with pytest.raises(RuntimeError, match="campaign halted"):
            campaign.run(root, limit=8)
        assert len(calls) == 1
        assert campaign.read(root / "halt.json")["failure_category"] == "failed_cell"


def test_context_guard_rejects_before_device_transfer_and_does_not_shrink_budget(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(
                                text_config=SimpleNamespace(max_position_embeddings=131072)),
                            generation_config=SimpleNamespace(eos_token_id=1))
    model.generate = lambda **kw: calls.append(kw) or torch.tensor([[3, 4, 1]])
    monkeypatch.setattr(caa, "render_messages", lambda *a, **kw: "x")
    monkeypatch.setattr(caa, "tokenize_text", lambda *a: {"input_ids": torch.tensor([[3, 4]])})
    monkeypatch.setattr(caa, "_first_device", lambda m: "cpu")
    monkeypatch.setattr(caa, "_move_inputs", lambda *a: pytest.fail("must reject before transfer"))
    with pytest.raises(RuntimeError, match="No tokens were truncated"):
        caa.generate_messages_unsteered(messages=[], processor=None, model=model,
                                        max_new_tokens=100000, max_context_tokens=100001)
    with pytest.raises(RuntimeError, match="131072"):
        caa.generate_messages_unsteered(messages=[], processor=None, model=model,
                                        max_new_tokens=131072, max_context_tokens=200000)
    assert not calls
    monkeypatch.setattr(caa, "_move_inputs", lambda inputs, device: inputs)
    monkeypatch.setattr(caa, "decode_tokens", lambda *a: "{}")
    caa.generate_messages_unsteered(messages=[], processor=None, model=model,
                                    max_new_tokens=100000, max_context_tokens=131072,
                                    prefill_chunk_size=2048, cache_implementation="dynamic")
    assert calls[0]["max_new_tokens"] == 100000
    assert calls[0]["prefill_chunk_size"] == 2048


@pytest.mark.parametrize("steered", [False, True])
def test_tiny_gemma_chunked_prefill_matches_unchunked(steered):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4ForCausalLM"):
        pytest.skip("native Gemma 4 support required")
    torch.manual_seed(123)
    config = transformers.Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=64,
        max_position_embeddings=256, sliding_window=32,
        layer_types=["sliding_attention", "full_attention"],
    )
    model = transformers.Gemma4ForCausalLM(config).eval()
    model.set_attn_implementation("sdpa")
    tokens = torch.randint(3, 64, (1, 48))
    bundle = VectorBundle({"candidate": {1: np.ones(32, dtype=np.float32)}})
    from contextlib import nullcontext
    context = caa.SteeringHooks(model=model, vector_bundle=bundle, concept="candidate",
                                layers=[1], mode="add", scale=-.05) if steered else nullcontext()
    with torch.inference_mode(), context:
        outputs = [model.generate(
            tokens, max_new_tokens=4, do_sample=False, cache_implementation="dynamic",
            prefill_chunk_size=chunk, return_dict_in_generate=True, output_scores=True,
        ) for chunk in (None, 16)]
    torch.testing.assert_close(outputs[0].sequences, outputs[1].sequences)
    for left, right in zip(outputs[0].scores, outputs[1].scores, strict=True):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)
