"""CPU contract tests for the matched-discovery CAA path (no model downloads)."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from onc_co_scientist.caa_server import (
    CAAInferenceEngine,
    build_chat_completion_payload,
    build_responses_payload,
)
from onc_co_scientist.expected_surprising.masking import mask_package
from onc_co_scientist.interventions import VectorBundle
from onc_co_scientist.interventions.caa import GenerationOutput, render_messages
from onc_co_scientist.interventions.discovery_pairs import discovery_pairs, training_pairs
from onc_co_scientist.providers.caa_openai import CAAConfig, CAAProvider
from tests.test_expected_surprising_experiment import config, providers

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/expected_surprising/caa_discovery.py"
spec = importlib.util.spec_from_file_location("caa_discovery", SCRIPT)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_development_contrasts_hold_context_and_evidence_fixed():
    training, development = discovery_pairs(), discovery_pairs("development")
    assert len(training) == 24 and len(development) == 8 and len(training_pairs()) == 26
    for pair in [*training, *development]:
        assert pair.positive_messages[:-1] == pair.negative_messages[:-1]
        assert pair.positive_messages[-1]["role"] == "assistant"
        assert pair.positive_messages[-1] != pair.negative_messages[-1]
        assert "nsclc" not in json.dumps(pair.to_json()).lower()
    assert not ({p.pair_id.split("_")[2] for p in training}
                & {p.pair_id.split("_")[2] for p in development})


def test_random_control_norms_and_reproducibility():
    bundle = VectorBundle({"paradigm_orthogonalized": {2: np.array([1., 2., 3.])}})
    first, second = bundle.with_random_control(), bundle.with_random_control()
    np.testing.assert_array_equal(first.vector("random_norm_matched", 2),
                                  second.vector("random_norm_matched", 2))
    assert np.isclose(np.linalg.norm(first.vector("random_norm_matched", 2)), np.sqrt(14))
    assert "random_norm_matched" not in bundle.vectors


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_protocol_preserves_raw_usage_and_truncation(reason):
    output = GenerationOutput('{"ok":true}', prompt_tokens=20, completion_tokens=60,
                              finish_reason=reason, raw_text="thought plus final answer")
    output.provenance = {"fingerprint": "a" * 64}
    payload = {"model": "gemma4-31b-control"}
    chat = build_chat_completion_payload(request_payload=payload, generated_text=output)
    assert chat["usage"] == {"prompt_tokens": 20, "completion_tokens": 60, "total_tokens": 80}
    assert chat["choices"][0]["finish_reason"] == reason
    assert chat["raw_generation"] == output.raw_text
    response = build_responses_payload(request_payload=payload, generated_text=output)
    assert response["usage"]["output_tokens"] == 60
    assert (response["status"] == "incomplete") == (reason == "length")


def test_request_thinking_and_fingerprint_enforced(tmp_path, monkeypatch):
    calls = []
    engine = CAAInferenceEngine(model_path="fixture", vector_file=tmp_path / "x.npz")
    engine.model, engine.processor = object(), object()
    engine.manifest = {"fingerprint": "a" * 64}

    def generate(**kwargs):
        calls.append(kwargs)
        return GenerationOutput("{}", prompt_tokens=4, completion_tokens=2,
                                finish_reason="stop", raw_text="{}")

    monkeypatch.setattr("onc_co_scientist.interventions.caa.generate_messages_unsteered", generate)
    payload = {"model": "gemma4-31b-control", "messages": [{"role": "user", "content": "x"}],
               "caa_expected_fingerprint": "a" * 64}
    for thinking in (True, False):
        output, _, _ = engine.generate_for_request({**payload,
            "chat_template_kwargs": {"enable_thinking": thinking}}, chat_messages=True)
        assert calls[-1]["enable_thinking"] is thinking
        assert output.provenance["enable_thinking"] is thinking
    with pytest.raises(ValueError, match="fingerprint"):
        engine.generate_for_request({**payload, "caa_expected_fingerprint": "b" * 64})
    with pytest.raises(ValueError, match="constrained JSON"):
        engine.generate_for_request({**payload, "response_format": {"type": "json_object"}})
    assert len(calls) == 2
    received = {}

    def template(messages, **kwargs):
        received.update(kwargs)
        return "prompt"

    render_messages(SimpleNamespace(apply_chat_template=template), [],
                    add_generation_prompt=True, enable_thinking=True)
    assert received["enable_thinking"] is True


def test_provider_sampling_and_provenance(monkeypatch):
    monkeypatch.setattr(CAAProvider, "_build_client", lambda self: None)
    provider = CAAProvider(CAAConfig(model_id="gemma4-control", server_fingerprint="a" * 64))
    options = provider.generation_options(final_retry=True, truncation_failures=2)
    assert options["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert options["extra_body"]["caa_expected_fingerprint"] == "a" * 64
    assert options["temperature"] == 1 and "response_format" not in options
    response = SimpleNamespace(raw={"caa_provenance": {"fingerprint": "a" * 64,
                "arm": {"model_id": "gemma4-control"}},
                "usage": {"prompt_tokens": 1, "completion_tokens": 2}})
    monkeypatch.setattr("onc_co_scientist.providers.vllm_openai.VLLMProvider.chat",
                        lambda *args, **kwargs: response)
    assert provider.chat([]) is response
    response.raw["caa_provenance"]["fingerprint"] = "b" * 64
    with pytest.raises(RuntimeError, match="provenance"):
        provider.chat([])
    with pytest.raises(ValueError, match="fingerprint"):
        CAAConfig(model_id="bad")


@pytest.mark.parametrize("last_token,reason", [(106, "stop"), (9, "length")])
def test_generation_counts_and_checkpoint_stop_ids(last_token, reason):
    torch = pytest.importorskip("torch")
    from onc_co_scientist.interventions.caa import generate_messages_unsteered

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = SimpleNamespace(eos_token_id=[1, 106])

        def generate(self, **kwargs):
            assert torch.is_inference_mode_enabled()
            assert "eos_token_id" not in kwargs
            return torch.tensor([[3, 4, 8, last_token]])

    class Processor:
        eos_token_id = 1
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return {"input_ids": torch.tensor([[3, 4]])}

        def decode(self, *args, **kwargs):
            return "raw thought and final"

        def parse_response(self, text):
            return {"content": "{}"}

    output = generate_messages_unsteered(messages=[], processor=Processor(), model=Model(),
                                         max_new_tokens=2, temperature=0)
    assert output.finish_reason == reason
    assert output.usage["prompt_tokens"] == output.usage["completion_tokens"] == 2
    assert str(output) == ("{}" if reason == "stop" else "raw thought and final")


def test_hooks_apply_at_prefill_and_decode_and_remove():
    torch = pytest.importorskip("torch")
    from onc_co_scientist.interventions.caa import SteeringHooks

    layers = torch.nn.ModuleList([torch.nn.Identity()])
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    bundle = VectorBundle({"test": {0: np.array([1., 2.], dtype=np.float32)}})
    for scale in (0., -0.5, 0.5):
        with SteeringHooks(model=model, vector_bundle=bundle, concept="test", layers=[0],
                           mode="add", scale=scale):
            for length in (5, 1):
                result = layers[0](torch.zeros(1, length, 2))
                torch.testing.assert_close(
                    result, torch.tensor([scale, 2 * scale]).expand_as(result)
                )
        assert not layers[0]._forward_hooks
    with pytest.raises(RuntimeError), SteeringHooks(
        model=model, vector_bundle=bundle, concept="test", layers=[0], mode="add", scale=1,
    ):
        raise RuntimeError("generation failed")
    assert not layers[0]._forward_hooks


def test_openai_client_round_trip_preserves_usage_and_rejects_truncation(tmp_path, monkeypatch):
    import httpx
    from openai import OpenAI

    from onc_co_scientist.providers.base import ChatMessage

    engine = CAAInferenceEngine(model_path="fixture", vector_file=tmp_path / "v.npz")
    engine.processor, engine.model = object(), object()
    engine.manifest = {"fingerprint": "a" * 64}
    reason = "stop"

    def generate(**kwargs):
        return GenerationOutput("{}", prompt_tokens=11, completion_tokens=22,
                                finish_reason=reason, raw_text="raw thought and {}")

    monkeypatch.setattr("onc_co_scientist.interventions.caa.generate_messages_unsteered", generate)
    provider = CAAProvider(CAAConfig(model_id="gemma4-31b-control", server_fingerprint="a" * 64))

    def transport(request):
        payload = json.loads(request.content)
        text, alias, _ = engine.generate_for_request(payload, chat_messages=True)
        body = build_chat_completion_payload(request_payload=payload, generated_text=text,
                                             model_alias=alias)
        return httpx.Response(200, json=body)

    # Exercise actual SDK serialization without sockets, ASGI threads or GPU dependencies.
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        provider._client.close()
        provider._client = OpenAI(
            api_key="EMPTY", base_url="http://testserver/v1", http_client=client
        )
        result = provider.chat([ChatMessage("user", "hello")])
        assert result.text == "{}"
        assert result.raw["usage"]["completion_tokens"] == 22
        assert result.raw["raw_generation"] == "raw thought and {}"
        reason = "length"
        result = provider.chat([ChatMessage("user", "hello")])
        assert result.text == "" and "adapter_error" in result.raw


@pytest.fixture
def frozen_campaign(tmp_path):
    _, pairs, raw, path = config(tmp_path)
    raw["expected_surprising"]["pair_ids"] = [pairs[0].pair_id]
    raw["schedule_seed"] = 7
    raw["budget"]["max_runtime_seconds_per_call"] = 1800
    path.write_text(yaml.safe_dump(raw))
    mask_package(tmp_path / "packages", tmp_path / "masked")
    manifest = {"protocol": "caa-discovery-1", "dtype": "bfloat16",
                "compact_agent_context": False, "aliases": [
        {"model_id": "gemma4-control", "arm": "control", "scale": None},
        {"model_id": "gemma4-caa", "arm": "candidate", "scale": -.05}]}
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    campaign.write(tmp_path / "manifest.json", manifest)
    root = tmp_path / "campaign"
    frozen = campaign.prepare(out=root, named=tmp_path / "packages", masked=tmp_path / "masked",
        base_config=path, server_manifest=tmp_path / "manifest.json",
        arms=["gemma4-control", "gemma4-caa"], replicates=1, rationale="unvalidated unit fixture")
    return root, frozen, pairs[0]


def test_freeze_balances_all_conditions_and_detects_tampering(frozen_campaign):
    root, frozen, _ = frozen_campaign
    assert frozen["n_runs"] == 8
    assert campaign.verify(root) == frozen
    jobs = campaign.read(root / "schedule.json")
    assert len({(j["view"], j["run_id"]) for j in jobs}) == 8
    raw = yaml.safe_load((root / "masked/config.yaml").read_text())
    assert raw["models"][0]["provider_config"]["json_object_output"] is False
    assert raw["models"][0]["provider_config"]["kind"] == "caa_openai"
    with (root / "masked/config.yaml").open("a") as stream:
        stream.write("\n# change\n")
    with pytest.raises(ValueError, match="changed"):
        campaign.verify(root)


def test_shared_controller_and_missing_run_summary(frozen_campaign, monkeypatch):
    root, _, pair = frozen_campaign
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    initial = campaign.summarize(root)
    assert all(r["value"] is None for r in initial["finished"]["cross_condition"])
    assert len(initial["finished"]["intervention_minus_control"]) == 1
    assert initial["finished"]["intervention_minus_control"][0]["delta_interaction_pp"] is None
    providers(monkeypatch, pair)
    monkeypatch.setattr("tests.test_expected_surprising_workflow.ScriptedScientist.model_id",
                        "gemma4-control")
    # Exercise the real current scientific controller, not the legacy CAA gate.
    spec = campaign.load_experiment_spec(root / "unmasked/config.yaml")
    plan = campaign.build_run_plans(spec)[0]
    result = campaign.run_cell(spec, plan, root / "unmasked", spec.fingerprint(), resume=False)
    assert result["status"] == "completed"
    summary = campaign.summarize(root)
    assert all(r["value"] is None for r in summary["finished"]["cross_condition"])
    rows = campaign.read(root / "analysis/summary.json")["runs"]
    assert sum(r["report_available"] for r in rows) == 1
    assert sum(r["status"] == "queued" for r in rows) == 7
