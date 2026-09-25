"""Fail closed on numerical failures; never silently sanitize scientific vectors."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onc_co_scientist.interventions import VectorBundle, caa


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_vectors_rejected_on_construction_and_load(tmp_path, bad):
    with pytest.raises(ValueError, match="Non-finite CAA vector"):
        VectorBundle({"candidate": {3: np.array([1.0, bad])}})
    path = tmp_path / "corrupt.npz"
    np.savez(path, candidate__layer_3=np.array([bad]))
    with pytest.raises(ValueError, match="Non-finite CAA vector"):
        VectorBundle.load(path)


def test_mutated_nonfinite_vector_not_saved(tmp_path):
    bundle = VectorBundle({"candidate": {3: np.array([1.0])}})
    bundle.vectors["candidate"][3][0] = np.nan
    path = tmp_path / "vectors.npz"
    with pytest.raises(ValueError, match="Non-finite CAA vector"):
        bundle.save(path)
    assert not path.exists()


@pytest.mark.parametrize(
    "text,reason,valid",
    [
        ('{"status":"ready"}', "stop", True),
        ('{"status":"ready"}', "length", False),
        ('{"status":"wrong"}', "stop", False),
        ("unparsed thought", "stop", False),
    ],
)
def test_smoke_requires_valid_stopped_answer(text, reason, valid):
    path = Path(__file__).parents[1] / "scripts/expected_surprising/caa_local_smoke.py"
    spec = spec_from_file_location("caa_local_smoke", path)
    smoke = module_from_spec(spec)
    spec.loader.exec_module(smoke)
    output = caa.GenerationOutput(
        text, prompt_tokens=5, completion_tokens=10, finish_reason=reason, raw_text=text
    )
    assert all(smoke.validate_generation(output).values()) is valid


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_extraction_rejects_nonfinite_activation(monkeypatch, bad):
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **kwargs):
            return SimpleNamespace(
                hidden_states=(
                    torch.zeros(1, 2, 2),
                    torch.tensor([[[1.0, 1.0], [bad, 2.0]]]),
                )
            )

    monkeypatch.setattr(caa, "render_messages", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(caa, "tokenize_text", lambda *args: {"input_ids": torch.tensor([[1, 2]])})
    with pytest.raises(ValueError, match="Non-finite prompt activation at layer 0"):
        caa.collect_prompt_activations(
            [],
            processor=None,
            model=Model(),
            layers=[0],
            position="last",
        )
