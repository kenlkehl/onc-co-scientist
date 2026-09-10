# Masked clinical workflow grid, September 10, 2026

This grid repeats the six-model, three-workflow clinical benchmark using masked copies of its exact expected and surprising datasets. Each contains 50,000 rows. All 43 predictor names are replaced by seeded `feature_NNN` labels. Smoking-status and histology values use seeded, column-scoped `level_NNN` labels. Numerical values, row order, missingness, patient identifiers, and the clinical outcome `log_pfs_months` are preserved. Levels are nominal, not ordinal. One private mapping is shared by both versions.

The 360 runs comprise Luna, Terra, Sol, Astra, Qwen 3.8 27B, and Gemma 4 31B × persistent, sequential, and deliberative workflows × expected/surprising × ten repeats. Settings match the original grid: medium reasoning, default service tier, 25 iterations, 125,000 output tokens per call, 900-call ceiling, two technical retries, 1,800 seconds per call, 120,000-character persistent history, two deliberative peers and one chair, and at most 30 concurrent runs. Scientific analysis and validation budgets and public 10% clinical-significance guidance are unchanged.

All models use the repaired September 9 harness, including available-peer fallback and retention of scientific scores after failed stages. The ongoing unmasked comparison mixes original and repaired implementations, so a pooled difference against that run is not solely a masking effect. This limitation applies especially to Qwen/Gemma and the retained older Codex cells.

The reviewed original pair and literature records remain private and unchanged. At runtime, hypotheses are expressed in masked coordinates. Validation samples and full-DGP conditional means are generated from the original specification, then masked. Seeds for independent validation and scheduled selection translate canonical comparisons back to their original names and values. The same claim therefore receives the same numerical draws, intervals, and confirmation results as its named equivalent. Mapping files never enter public workspaces or participant prompts.

The new grid owns its source snapshot, input package, run IDs, call journals, logs, lock, and reports. Codex cells start immediately. Qwen/Gemma cells remain queued until the original vLLM continuation records completion; the new driver checks read-only every 30 seconds and starts them automatically. It never stops, resumes, rewrites, or changes the source of the unmasked run. Scheduling order may therefore differ from the original randomized plan while identities and scientific seeds remain fixed.

Validation includes categorical round trips, missing and mixed values, unused declared levels, private/public separation, both paired datasets, all six ground-truth discoveries, independent validation and final confirmation equivalence, non-target full-DGP adjudication, and end-to-end execution of all three workflows. The live Luna deliberative smoke is separate from all 360 scored runs, and its costs remain separate.

- [Live status](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260910_clinical10pct_masked_workflows/STATUS.md)
- [Live calls and token totals](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260910_clinical10pct_masked_workflows/LIVE_PROGRESS.md)
- [Configuration](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260910_clinical10pct_masked_workflows/config.yaml)
- [Frozen source/input hashes and masking provenance](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260910_clinical10pct_masked_workflows/frozen_manifest.json)
- [Expected dataset](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260910_clinical10pct_masked_workflows/input_data/public/a561a12e2075ee78/dataset.parquet)
- [Surprising dataset](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260910_clinical10pct_masked_workflows/input_data/public/1a87643398283251/dataset.parquet)
- [Final results, when complete](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260910_clinical10pct_masked_workflows/RESULTS.md)

The driver writes the final report automatically. Restart only this driver with its frozen `source/run_masked_grid.py --out <this-grid-root> --resume` and frozen `source/src` on `PYTHONPATH`; the process lock prevents duplicate drivers. Resume replays cached calls and preserves implementation/input hash checks. Never launch the new config with the generic runner, which does not implement the endpoint queue.
