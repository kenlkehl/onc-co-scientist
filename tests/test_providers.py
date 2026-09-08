import pytest

from onc_co_scientist.providers.registry import get_provider


def test_registry_rejects_unknown_kind():
    with pytest.raises(ValueError):
        get_provider({"kind": "nope"})


def test_vllm_config_requires_model_id():
    with pytest.raises(ValueError):
        get_provider({"kind": "vllm_openai"})


def test_vllm_registry_passes_explicit_reasoning_and_tier(monkeypatch):
    from types import SimpleNamespace
    from onc_co_scientist.providers.vllm_openai import VLLMProvider
    from onc_co_scientist.providers.base import ChatMessage
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ready'))])
    monkeypatch.setattr(VLLMProvider, '_build_client', lambda self: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    provider = get_provider({'kind': 'vllm_openai', 'model_id': 'test', 'reasoning_effort': 'medium', 'service_tier': 'default'})
    assert provider.chat([ChatMessage(role='user', content='hi')]).text == 'ready'
    assert captured['reasoning_effort'] == 'medium'
    assert captured['service_tier'] == 'default'
    captured.clear()
    get_provider({'kind': 'vllm_openai', 'model_id': 'test'}).chat([])
    assert 'reasoning_effort' not in captured and 'service_tier' not in captured
