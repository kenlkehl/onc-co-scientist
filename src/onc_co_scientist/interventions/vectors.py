"""Vector artifact helpers for CAA and activation steering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def orthogonalize(vector: np.ndarray, against: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return the component of ``vector`` orthogonal to ``against``."""

    v = np.asarray(vector, dtype=np.float32)
    k = np.asarray(against, dtype=np.float32)
    denom = float(np.dot(k, k))
    if denom <= eps:
        return v.copy()
    projection = (float(np.dot(v, k)) / denom) * k
    return (v - projection).astype(np.float32)


def metadata_path_for(vector_path: Path) -> Path:
    if vector_path.suffix == ".npz":
        return vector_path.with_suffix(".json")
    return vector_path.with_suffix(vector_path.suffix + ".json")


@dataclass
class VectorBundle:
    """A serializable set of per-concept, per-layer activation vectors."""

    vectors: dict[str, dict[int, np.ndarray]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def concepts(self) -> list[str]:
        return sorted(self.vectors)

    def layers_for(self, concept: str) -> list[int]:
        return sorted(self.vectors.get(concept, {}))

    def vector(self, concept: str, layer: int) -> np.ndarray:
        try:
            return self.vectors[concept][layer]
        except KeyError as exc:
            raise KeyError(f"No vector for concept={concept!r}, layer={layer}.") from exc

    def with_orthogonalized(
        self,
        *,
        source_concept: str = "paradigm_adherence",
        against_concept: str = "oncology_knowledge",
        out_concept: str = "paradigm_orthogonalized",
    ) -> VectorBundle:
        out: dict[str, dict[int, np.ndarray]] = {
            concept: {layer: vec.copy() for layer, vec in layer_map.items()}
            for concept, layer_map in self.vectors.items()
        }
        source_layers = out.get(source_concept, {})
        against_layers = out.get(against_concept, {})
        shared_layers = sorted(set(source_layers) & set(against_layers))
        if not shared_layers:
            return VectorBundle(vectors=out, metadata={**self.metadata})
        out[out_concept] = {
            layer: orthogonalize(source_layers[layer], against_layers[layer])
            for layer in shared_layers
        }
        metadata = {**self.metadata}
        metadata["orthogonalization"] = {
            "source_concept": source_concept,
            "against_concept": against_concept,
            "out_concept": out_concept,
            "layers": shared_layers,
        }
        return VectorBundle(vectors=out, metadata=metadata)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for concept, layer_map in self.vectors.items():
            for layer, vector in layer_map.items():
                arrays[_array_key(concept, layer)] = np.asarray(vector, dtype=np.float32)
        np.savez_compressed(path, **arrays)

        metadata = {
            **self.metadata,
            "concepts": {
                concept: {
                    "layers": sorted(layer_map),
                    "norms": {
                        str(layer): float(np.linalg.norm(vector))
                        for layer, vector in sorted(layer_map.items())
                    },
                }
                for concept, layer_map in self.vectors.items()
            },
        }
        metadata_path_for(path).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> VectorBundle:
        loaded = np.load(path)
        vectors: dict[str, dict[int, np.ndarray]] = {}
        for key in loaded.files:
            concept, layer = _parse_array_key(key)
            vectors.setdefault(concept, {})[layer] = loaded[key].astype(np.float32)
        metadata_path = metadata_path_for(path)
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        return cls(vectors=vectors, metadata=metadata)


def _array_key(concept: str, layer: int) -> str:
    return f"{concept}__layer_{layer}"


def _parse_array_key(key: str) -> tuple[str, int]:
    marker = "__layer_"
    if marker not in key:
        raise ValueError(f"Invalid vector array key {key!r}.")
    concept, raw_layer = key.rsplit(marker, 1)
    return concept, int(raw_layer)
