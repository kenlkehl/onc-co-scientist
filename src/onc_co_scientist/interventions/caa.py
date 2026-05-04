"""Contrastive activation addition prototype.

This module keeps heavyweight ML imports lazy so the benchmark and scoring
tests can run without a CUDA/Transformers environment. The implementation is a
prototype suitable for deriving and trying CAA-style residual-stream vectors;
it does not edit model weights on disk.
"""

from __future__ import annotations

import datetime as dt
import inspect
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .prompts import ContrastPromptPair
from .vectors import VectorBundle

ActivationPosition = Literal["last", "mean"]
SteeringMode = Literal["add", "ablate"]


def parse_layers(raw: str | None, *, n_layers: int) -> list[int]:
    """Parse layer selection strings.

    Supported forms: ``all``, ``middle``, ``last:N``, or comma-separated
    zero-based layer indices such as ``20,30,40,50``.
    """

    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    text = (raw or "middle").strip().lower()
    if text == "all":
        return list(range(n_layers))
    if text == "middle":
        return [n_layers // 2]
    if text.startswith("last:"):
        count = int(text.split(":", 1)[1])
        if count <= 0:
            raise ValueError("last:N layer count must be positive.")
        start = max(0, n_layers - count)
        return list(range(start, n_layers))
    layers = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not layers:
        raise ValueError(f"No layers selected from {raw!r}.")
    bad = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if bad:
        raise ValueError(f"Layer(s) out of range for {n_layers} layers: {bad}.")
    return sorted(dict.fromkeys(layers))


def load_transformers_text_model(
    model_id_or_path: str,
    *,
    cache_dir: Path | None = None,
    local_files_only: bool = True,
    dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = False,
):
    """Load a text-generation model plus processor/tokenizer via Transformers."""

    torch = _import_torch()
    try:
        import transformers
        from transformers import AutoProcessor, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "CAA experiments require the interventions extra: uv pip install -e '.[interventions]'"
        ) from exc

    common_kwargs: dict[str, Any] = {
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}

    try:
        processor = AutoProcessor.from_pretrained(model_id_or_path, **common_kwargs)
    except Exception:
        processor = AutoTokenizer.from_pretrained(model_id_or_path, **common_kwargs)

    dtype_arg: Any
    if dtype == "auto":
        dtype_arg = "auto"
    else:
        try:
            dtype_arg = getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype {dtype!r}.") from exc

    model_kwargs = {
        **common_kwargs,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    model_classes = [transformers.AutoModelForCausalLM]
    for class_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            model_classes.append(model_class)

    # Current Gemma model cards use `dtype`; older Transformers uses
    # `torch_dtype`. Try the documented path first, then fall back. If the
    # causal-LM auto class does not support a multimodal config, try newer
    # multimodal auto classes when available.
    last_config_error: Exception | None = None
    for model_class in model_classes:
        try:
            model = model_class.from_pretrained(
                model_id_or_path,
                dtype=dtype_arg,
                **model_kwargs,
            )
            break
        except TypeError:
            model = model_class.from_pretrained(
                model_id_or_path,
                torch_dtype=dtype_arg,
                **model_kwargs,
            )
            break
        except (KeyError, ValueError) as exc:
            last_config_error = exc
    else:
        assert last_config_error is not None
        raise last_config_error
    model.eval()
    return processor, model


def derive_caa_vectors(
    pairs: list[ContrastPromptPair],
    *,
    processor,
    model,
    layers: list[int],
    position: ActivationPosition = "last",
    enable_thinking: bool = False,
) -> VectorBundle:
    """Derive mean positive-minus-negative activation vectors for each concept."""

    if not pairs:
        raise ValueError("At least one contrast pair is required.")
    if not layers:
        raise ValueError("At least one layer is required.")

    accum: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        positive = collect_prompt_activations(
            pair.positive_messages,
            processor=processor,
            model=model,
            layers=layers,
            position=position,
            enable_thinking=enable_thinking,
        )
        negative = collect_prompt_activations(
            pair.negative_messages,
            processor=processor,
            model=model,
            layers=layers,
            position=position,
            enable_thinking=enable_thinking,
        )
        for layer in layers:
            accum[pair.concept][layer].append(positive[layer] - negative[layer])

    vectors: dict[str, dict[int, np.ndarray]] = {}
    for concept, layer_map in accum.items():
        vectors[concept] = {}
        for layer, values in layer_map.items():
            vectors[concept][layer] = np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)

    metadata = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "model_id": getattr(getattr(model, "config", None), "_name_or_path", None),
        "layers": layers,
        "position": position,
        "enable_thinking": enable_thinking,
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "concept": pair.concept,
                "source": pair.source,
            }
            for pair in pairs
        ],
    }
    return VectorBundle(vectors=vectors, metadata=metadata).with_orthogonalized()


def collect_prompt_activations(
    messages: list[dict[str, str]],
    *,
    processor,
    model,
    layers: list[int],
    position: ActivationPosition,
    enable_thinking: bool = False,
) -> dict[int, np.ndarray]:
    """Run one prompt and return selected hidden-state activations by layer."""

    torch = _import_torch()
    text = render_messages(
        processor,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenize_text(processor, text)
    inputs = _move_inputs(inputs, _first_device(model))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    n_transformer_layers = len(hidden_states) - 1
    bad = [layer for layer in layers if layer < 0 or layer >= n_transformer_layers]
    if bad:
        raise ValueError(f"Layer(s) out of range for {n_transformer_layers} hidden layers: {bad}.")

    attention_mask = inputs.get("attention_mask")
    out: dict[int, np.ndarray] = {}
    for layer in layers:
        # hidden_states[0] is embeddings; hidden_states[layer + 1] is after layer.
        layer_hidden = hidden_states[layer + 1][0]
        if position == "mean":
            if attention_mask is None:
                activation = layer_hidden.mean(dim=0)
            else:
                mask = attention_mask[0].to(dtype=torch.bool, device=layer_hidden.device)
                activation = layer_hidden[mask].mean(dim=0)
        elif position == "last":
            idx = _last_token_index(inputs)
            activation = layer_hidden[idx]
        else:
            raise ValueError(f"Unknown activation position {position!r}.")
        out[layer] = activation.detach().float().cpu().numpy().astype(np.float32)
    return out


def generate_with_vector(
    *,
    prompt: str,
    system: str | None,
    vector_bundle: VectorBundle,
    concept: str,
    layers: list[int] | None,
    processor,
    model,
    mode: SteeringMode = "add",
    scale: float = -1.0,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    enable_thinking: bool = False,
) -> str:
    """Generate text while applying additive steering or runtime ablation."""

    selected_layers = layers if layers is not None else vector_bundle.layers_for(concept)
    if not selected_layers:
        raise ValueError(f"No layers available for concept {concept!r}.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    text = render_messages(
        processor,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = _move_inputs(tokenize_text(processor, text), _first_device(model))
    input_len = int(inputs["input_ids"].shape[-1])
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    pad_token_id = _processor_token_id(processor, "pad_token_id")
    eos_token_id = _processor_token_id(processor, "eos_token_id")
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id

    with SteeringHooks(
        model=model,
        vector_bundle=vector_bundle,
        concept=concept,
        layers=selected_layers,
        mode=mode,
        scale=scale,
    ):
        outputs = model.generate(**inputs, **kwargs)

    generated = outputs[0][input_len:]
    decoded = decode_tokens(processor, generated)
    parser = getattr(processor, "parse_response", None)
    if callable(parser):
        try:
            parsed = parser(decoded)
            if isinstance(parsed, str):
                return parsed
            if isinstance(parsed, dict):
                return str(parsed.get("content") or parsed.get("text") or parsed)
            return str(parsed)
        except Exception:
            return decoded
    return decoded


class SteeringHooks:
    """Context manager that applies vectors to decoder-layer residual outputs."""

    def __init__(
        self,
        *,
        model,
        vector_bundle: VectorBundle,
        concept: str,
        layers: Iterable[int],
        mode: SteeringMode,
        scale: float,
    ) -> None:
        self.model = model
        self.vector_bundle = vector_bundle
        self.concept = concept
        self.layers = list(layers)
        self.mode = mode
        self.scale = scale
        self.handles = []

    def __enter__(self) -> SteeringHooks:
        torch = _import_torch()
        decoder_layers = find_decoder_layers(self.model)
        for layer_idx in self.layers:
            if layer_idx < 0 or layer_idx >= len(decoder_layers):
                raise ValueError(
                    f"Layer {layer_idx} out of range for {len(decoder_layers)} decoder layers."
                )
            vector_np = self.vector_bundle.vector(self.concept, layer_idx)
            vector = torch.tensor(vector_np, dtype=torch.float32)

            def hook(_module, _inputs, output, *, vector=vector):
                hidden, rebuild = _split_layer_output(output)
                if hidden.shape[-1] != vector.shape[0]:
                    raise ValueError(
                        f"Vector dim {vector.shape[0]} does not match hidden dim "
                        f"{hidden.shape[-1]}."
                    )
                v = vector.to(device=hidden.device, dtype=hidden.dtype)
                if self.mode == "add":
                    adjusted = hidden + self.scale * v.view(1, 1, -1)
                elif self.mode == "ablate":
                    v_float = v.float()
                    norm = v_float.norm()
                    if float(norm) == 0.0:
                        adjusted = hidden
                    else:
                        unit = (v_float / norm).to(dtype=hidden.dtype)
                        coeff = hidden.float().matmul(unit.float()).to(dtype=hidden.dtype)
                        adjusted = hidden - self.scale * coeff.unsqueeze(-1) * unit.view(1, 1, -1)
                else:
                    raise ValueError(f"Unknown steering mode {self.mode!r}.")
                return rebuild(adjusted)

            self.handles.append(decoder_layers[layer_idx].register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def render_messages(
    processor,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    template = getattr(processor, "apply_chat_template", None)
    if callable(template):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        signature = inspect.signature(template)
        if "enable_thinking" in signature.parameters:
            kwargs["enable_thinking"] = enable_thinking
        return template(messages, **kwargs)
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:")
    return "\n".join(rendered)


def tokenize_text(processor, text: str):
    if callable(processor):
        return processor(text=text, return_tensors="pt")
    raise TypeError("Processor/tokenizer is not callable.")


def decode_tokens(processor, token_ids) -> str:
    decoder = getattr(processor, "decode", None)
    if callable(decoder):
        return decoder(token_ids, skip_special_tokens=False)
    tokenizer = getattr(processor, "tokenizer", None)
    decoder = getattr(tokenizer, "decode", None)
    if callable(decoder):
        return decoder(token_ids, skip_special_tokens=False)
    raise TypeError("Processor/tokenizer does not expose decode().")


def _processor_token_id(processor, name: str) -> int | None:
    value = getattr(processor, name, None)
    if value is not None:
        return int(value)
    tokenizer = getattr(processor, "tokenizer", None)
    value = getattr(tokenizer, name, None)
    return int(value) if value is not None else None


def find_decoder_layers(model) -> list:
    candidates = [
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if (isinstance(obj, (list, tuple)) or hasattr(obj, "__len__")) and len(obj) > 0:
            return list(obj)
    raise ValueError(
        "Could not locate decoder layers on the model. Add its module path to "
        "find_decoder_layers()."
    )


def infer_num_layers(model) -> int:
    config = getattr(model, "config", None)
    config_layers = getattr(config, "num_hidden_layers", None)
    if config_layers is not None:
        return int(config_layers)
    text_config = getattr(config, "text_config", None)
    text_config_layers = getattr(text_config, "num_hidden_layers", None)
    if text_config_layers is not None:
        return int(text_config_layers)
    return len(find_decoder_layers(model))


def _split_layer_output(output):
    if isinstance(output, tuple):
        hidden = output[0]

        def rebuild(adjusted):
            return (adjusted, *output[1:])

        return hidden, rebuild

    def rebuild(adjusted):
        return adjusted

    return output, rebuild


def _last_token_index(inputs) -> int:
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        return int(inputs["input_ids"].shape[-1] - 1)
    return int(attention_mask[0].sum().item() - 1)


def _move_inputs(inputs, device):
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _first_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "CAA experiments require torch. Install with: uv pip install -e '.[interventions]'"
        ) from exc
    return torch
