# Matched masked versus unmasked interim comparison

Uses the prior 176-completed masked snapshot. Only runs completed on both sides are compared, matching model, workflow, expected/surprising version, and replicate. Deleted local-model results are excluded. Differences are masked minus unmasked. Means weight the matched runs equally.

Unmasked snapshot: 2026-09-10T23:36:16.232752+00:00.
Matched completed runs: 172; masked completed runs excluded because their unmasked counterparts were not completed: 4.

## By workflow

| Model / workflow | Matched n | Unmasked focal recovery | Masked focal recovery | Unmasked F1* | Masked F1* | Δ F1* |
|---|---:|---:|---:|---:|---:|---:|
| astra_medium / deliberative | 11 | 11/11 | 11/11 | 80.6 | 78.7 | -1.9 |
| astra_medium / persistent | 16 | 16/16 | 16/16 | 79.4 | 78.6 | -0.8 |
| astra_medium / sequential | 15 | 15/15 | 15/15 | 78.2 | 79.8 | +1.5 |
| luna_medium / deliberative | 13 | 0/13 | 9/13 | 9.0 | 45.4 | +36.4 |
| luna_medium / persistent | 12 | 2/12 | 7/12 | 17.8 | 52.0 | +34.2 |
| luna_medium / sequential | 15 | 1/15 | 10/15 | 8.0 | 56.0 | +48.0 |
| qwen_3_8_27b / persistent | 2 | 2/2 | 2/2 | 80.0 | 80.0 | +0.0 |
| qwen_3_8_27b / sequential | 1 | 1/1 | 1/1 | 50.0 | 50.0 | +0.0 |
| sol_medium / deliberative | 10 | 6/10 | 10/10 | 56.2 | 78.1 | +21.8 |
| sol_medium / persistent | 15 | 14/15 | 15/15 | 72.6 | 78.1 | +5.4 |
| sol_medium / sequential | 15 | 13/15 | 15/15 | 71.4 | 77.2 | +5.8 |
| terra_medium / deliberative | 15 | 8/15 | 14/15 | 59.5 | 77.7 | +18.1 |
| terra_medium / persistent | 17 | 13/17 | 16/17 | 67.1 | 76.9 | +9.8 |
| terra_medium / sequential | 15 | 12/15 | 15/15 | 65.3 | 78.6 | +13.3 |

F1* is the existing 0–100 discovery score. Focal recovery requires independent confirmation. This completed-only analysis excludes failed runs on either side and can be affected by which runs finish first. Repeated runs share one base dataset pair; no across-dataset confidence intervals or causal masking-effect claim is justified. The unmasked Codex cohort mixes retained original and repaired implementations; masked Codex uses its frozen implementation. Qwen/Gemma use matching newly recommended caller settings on both sides. Per-pair generation provenance is retained in paired_runs.csv.
