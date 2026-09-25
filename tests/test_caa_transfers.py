"""CPU-only contracts for the opt-in, model-scoped GPU transfer workaround."""


import pytest

from onc_co_scientist.interventions import caa, transfers


def fake_cuda_tensor():
    torch = pytest.importorskip("torch")

    class FakeCudaTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda:1")

        def cpu(self):
            return torch.ones(2)

    return torch.Tensor._make_subclass(FakeCudaTensor, torch.ones(2))


def test_stages_only_cross_gpu_tensors_and_preserves_skipped_keys():
    value = fake_cuda_tensor()
    data = {"ordinary": (value, [value]), "past_key_values": value, "number": 7}
    staged = transfers.stage_cross_gpu(data, "cuda:0", "past_key_values")
    assert staged["ordinary"][0].device.type == "cpu"
    assert staged["ordinary"][1][0].device.type == "cpu"
    assert staged["past_key_values"] is value and staged["number"] == 7
    assert transfers.stage_cross_gpu(value, "cuda:1") is value
    assert transfers.stage_cross_gpu(value, None) is value


def test_hook_preserves_callers_device_without_global_patch(monkeypatch):
    torch = pytest.importorskip("torch")
    hooks = pytest.importorskip("accelerate.hooks")
    original_send = hooks.send_to_device
    inner = hooks.AlignDevicesHook(execution_device="cuda:0", io_same_device=True)
    model = torch.nn.Identity()
    model._hf_hook = inner
    value = fake_cuda_tensor()

    def pre(module, *args, **kwargs):
        assert args[0].device.type == "cpu"
        inner.input_device = torch.device("cpu")
        return args, kwargs

    monkeypatch.setattr(inner, "pre_forward", pre)
    monkeypatch.setattr(inner, "post_forward", lambda module, output: output)
    transfers.install_cpu_staged_transfers(model)
    model._hf_hook.pre_forward(model, value)
    assert inner.input_device == torch.device("cuda:1")
    assert hooks.send_to_device is original_send
    assert model.caa_inter_gpu_transfer == "cpu"


@pytest.mark.parametrize("kwargs", [
    {"inter_gpu_transfer": "unknown"},
    {"inter_gpu_transfer": "cpu", "compile_forward": True},
])
def test_rejects_invalid_transfer_policy_before_loading(kwargs):
    with pytest.raises(ValueError):
        caa.load_transformers_text_model("unused", **kwargs)
