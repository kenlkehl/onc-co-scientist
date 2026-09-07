"""Paired discovery inference at the base-dataset level."""

from __future__ import annotations

import numpy as np
import pandas as pd


def paired_summary(reports: list[dict], *, bootstrap_replicates=2000, seed=0) -> dict:
    if bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicate count must be positive")
    frame = pd.DataFrame(
        [
            {
                "pair": r["pair_id"],
                "profile": r["profile"],
                "model": r["model"],
                "harness": r["harness"],
                "version": r["version"],
                "run_id": r["run_id"],
                "recovered": int(r["confirmation"]["primary_recovery"]),
            }
            for r in reports
        ]
    )
    if frame.empty:
        raise ValueError("No run reports")
    if frame.duplicated(["pair", "model", "harness", "version", "run_id"]).any():
        raise ValueError("Duplicate run reports")
    if not set(frame.version) <= {"expected", "surprising"}:
        raise ValueError("Unknown paired condition")
    grouped = frame.groupby(["pair", "profile", "model", "harness", "version"]).recovered.mean()
    cells = grouped.unstack("version")
    if set(cells.columns) != {"expected", "surprising"} or cells.isna().any().any():
        raise ValueError("Every assigned pair/model/harness cell needs both versions")
    cells["difference"] = cells.surprising - cells.expected
    pairs = cells.groupby(["pair", "profile"])[["expected", "surprising", "difference"]].mean()
    pairs = pairs.reset_index()
    pairs["stratum"] = pairs.profile.map(lambda p: "depmap" if p.endswith("depmap") else "clinical")
    rng = np.random.default_rng(seed)
    strata = [group.difference.to_numpy() for _, group in pairs.groupby("stratum")]
    boot = [
        float(np.mean(np.concatenate([rng.choice(g, size=len(g), replace=True) for g in strata])))
        for _ in range(bootstrap_replicates)
    ]
    return {
        "base_dataset_n": len(pairs),
        "run_n": len(frame),
        "expected_recovery": float(pairs.expected.mean()),
        "surprising_recovery": float(pairs.surprising.mean()),
        "paired_difference": float(pairs.difference.mean()),
        "ci95": (
            np.quantile(boot, [0.025, 0.975]).tolist() if all(len(g) >= 2 for g in strata) else None
        ),
        "ci_available": all(len(g) >= 2 for g in strata),
        "bootstrap_unit": "base dataset, stratified by modality; both versions retained",
        "bootstrap_replicates": bootstrap_replicates,
        "pairs": pairs.to_dict("records"),
    }
