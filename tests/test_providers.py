import pytest

from onc_open_mindedness.providers.registry import get_provider


def test_registry_rejects_unknown_kind():
    with pytest.raises(ValueError):
        get_provider({"kind": "nope"})


def test_vllm_config_requires_model_id():
    with pytest.raises(ValueError):
        get_provider({"kind": "vllm_openai"})
