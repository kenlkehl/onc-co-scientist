"""Trusted, row-local execution; only sufficient statistics leave a site."""

import hashlib

import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy.stats import t

from .schemas import EvidenceResult
from .scoring import cells


def partition_frame(frame, cell, pair_id, replicate_id):
    if cell.sites > len(frame):
        raise ValueError("More sites than observations")
    for name in cell.partition.by:
        if name not in frame or name in {"patient_id", "cell_line_id"}:
            raise ValueError(f"Partition covariate is unavailable or an identifier: {name}")
    seed = int.from_bytes(
        hashlib.sha256(
            f"{pair_id}:{replicate_id}:{cell.seed}:{cell.partition.id}:{cell.sites}".encode()
        ).digest()[:8],
        "little",
    )
    order = np.random.default_rng(seed).permutation(len(frame))
    if cell.partition.mode == "heterogeneous":
        # Stable sorting after shuffling gives reproducible random tie-breaking.
        ordered = frame.iloc[order].reset_index(drop=True)
        ranks = ordered.sort_values(
            cell.partition.by, kind="stable", na_position="last"
        ).index.to_numpy()
        order = order[ranks]
    chunks = np.array_split(order, cell.sites)
    sites = {f"site_{i + 1}": frame.iloc[idx].copy() for i, idx in enumerate(chunks)}
    audit = {
        f"site_{i + 1}": {
            "rows": len(idx),
            "membership_sha256": hashlib.sha256(np.sort(idx).astype("<i8").tobytes()).hexdigest(),
        }
        for i, idx in enumerate(chunks)
    }
    return sites, audit


def sufficient_statistics(frame, hypothesis, minimum=20):
    try:
        groups, signs = cells(frame, hypothesis)
    except (ValueError, KeyError, TypeError) as error:
        return {"valid": False, "diagnostic": str(error), "cells": []}
    stats = []
    for group, sign in zip(groups, signs, strict=True):
        n = len(group)
        stats.append(
            {
                "n": n,
                "sign": sign,
                "mean": float(np.mean(group)) if n >= minimum else None,
                "variance": float(np.var(group, ddof=1)) if n >= minimum else None,
            }
        )
    return {
        "valid": all(c["mean"] is not None for c in stats),
        "diagnostic": "site_sufficient_statistics"
        if all(c["mean"] is not None for c in stats)
        else "suppressed_small_site_cell",
        "cells": stats,
    }


def combine_statistics(site_stats, hypothesis, *, delta, alpha, result_id):
    kwargs = dict(id=result_id, hypothesis_id=hypothesis.id, delta=delta, alpha=alpha)
    if not site_stats or any(not s["valid"] for s in site_stats):
        return EvidenceResult(
            **kwargs, valid=False, cell_n=[], diagnostic="missing_or_suppressed_site_cells"
        )
    combined = []
    for cells_at_position in zip(*(s["cells"] for s in site_stats), strict=True):
        n = sum(c["n"] for c in cells_at_position)
        mean = sum(c["n"] * c["mean"] for c in cells_at_position) / n
        variance = sum(
            (c["n"] - 1) * c["variance"] + c["n"] * (c["mean"] - mean) ** 2
            for c in cells_at_position
        ) / (n - 1)
        sign = cells_at_position[0]["sign"]
        if any(c["sign"] != sign for c in cells_at_position):
            raise ValueError("Incompatible site contrasts")
        combined.append((n, mean, variance, sign))
    components = np.array([v / n for n, _, v, _ in combined])
    value = sum(m * sign for _, m, _, sign in combined)
    variance = float(components.sum())
    if variance <= 0 or not np.isfinite(variance):
        return EvidenceResult(
            **kwargs, cell_n=[c[0] for c in combined], valid=False, diagnostic="invalid_variance"
        )
    denominator = sum(v * v / (c[0] - 1) for v, c in zip(components, combined, strict=True))
    df = variance * variance / denominator if denominator else float("inf")
    half = float(t.ppf(1 - alpha / 2, df) * np.sqrt(variance))
    return EvidenceResult(
        **kwargs,
        estimate=value,
        lower=value - half,
        upper=value + half,
        cell_n=[c[0] for c in combined],
        valid=True,
        diagnostic="pooled_site_sufficient_statistics",
    )


def public_result(result):
    return result.model_dump(exclude={"delta"})


def public_site_context(frame, task):
    # No samples, IDs, file paths or arbitrary row-valued objects are exposed.
    safe = frame.drop(columns=["patient_id", "cell_line_id"], errors="ignore")
    stats = {}
    for name, series in safe.items():
        count = int(series.count())
        item = {"dtype": str(series.dtype), "count": count}
        if count >= 20:
            if is_numeric_dtype(series.dtype):
                item.update(mean=float(series.mean()), std=float(series.std()))
            else:
                item["levels"] = {
                    str(level): int(n) for level, n in series.value_counts().items() if n >= 20
                }
        stats[name] = item
    return {**task, "observations": len(frame), "variables": stats}
