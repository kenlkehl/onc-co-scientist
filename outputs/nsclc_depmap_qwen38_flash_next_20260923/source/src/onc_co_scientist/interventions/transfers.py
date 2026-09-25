"""Opt-in CPU staging of Accelerate's inter-GPU activation transfers.

Weights remain on their assigned GPUs. Only this model's dispatch hooks are
wrapped; no global monkeypatch, package modification, or driver change is made.
"""

from collections.abc import Mapping


def stage_cross_gpu(value, target, skip_keys=None):
    """Move cross-GPU tensors to CPU before Accelerate sends them to the target."""
    import torch
    from accelerate.utils.operations import honor_type

    if target is None:
        return value
    target = torch.device("cuda", target) if isinstance(target, int) else torch.device(target)
    if isinstance(value, torch.Tensor):
        if value.device.type == target.type == "cuda" and value.device != target:
            return value.cpu()
        return value
    if isinstance(value, (tuple, list)):
        return honor_type(value, (stage_cross_gpu(v, target, skip_keys) for v in value))
    if isinstance(value, Mapping):
        skip = [skip_keys] if isinstance(skip_keys, str) else (skip_keys or [])
        return type(value)({
            k: v if k in skip else stage_cross_gpu(v, target, skip) for k, v in value.items()
        })
    return value


def install_cpu_staged_transfers(model):
    """Wrap existing dispatch hooks without reinitializing or redispatching weights."""
    from accelerate.hooks import AlignDevicesHook, ModelHook, SequentialHook
    from accelerate.utils.operations import find_device

    class CpuStagedHook(ModelHook):
        def __init__(self, inner):
            self.inner = inner
            self.no_grad = inner.no_grad

        def pre_forward(self, module, *args, **kwargs):
            inner = self.inner
            original_device = find_device([args, kwargs]) if inner.io_same_device else None
            args = stage_cross_gpu(args, inner.execution_device)
            kwargs = stage_cross_gpu(kwargs, inner.execution_device, inner.skip_keys)
            args, kwargs = inner.pre_forward(module, *args, **kwargs)
            if inner.io_same_device:
                # Staging must not change the caller's original output device.
                inner.input_device = original_device
            return args, kwargs

        def post_forward(self, module, output):
            if self.inner.io_same_device:
                output = stage_cross_gpu(output, self.inner.input_device, self.inner.skip_keys)
            return self.inner.post_forward(module, output)

        def detach_hook(self, module):
            return self.inner.detach_hook(module)

    def wrap(hook):
        if isinstance(hook, AlignDevicesHook):
            return CpuStagedHook(hook)
        if isinstance(hook, SequentialHook):
            hook.hooks = tuple(wrap(child) for child in hook.hooks)
        return hook

    for module in model.modules():
        if hasattr(module, "_hf_hook"):
            module._hf_hook = wrap(module._hf_hook)
    model.caa_inter_gpu_transfer = "cpu"
