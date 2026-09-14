# Interim masked benchmark metrics

Snapshot: 2026-09-10T23:33:55.604358+00:00.

{'completed': 176, 'failed': 5, 'active': 36, 'queued': 143}

Completed runs only in the performance tables; failures are counted separately. Pending runs are never scored as failures. Previous deleted local-model runs are excluded. Means weight available completed runs equally; expected/surprising and workflow completion counts are uneven. These are descriptive interim results, not a final ranking. Only one base dataset pair is represented, so no across-dataset confidence intervals are estimated.

## Focal recovery

Recovery requires recovery of the focal planted finding and independent evaluator confirmation.

| Model | Workflow | Complete | Failed | Expected recovered/n | Surprising recovered/n | Matched pairs | Matched S−E (pp) |
|---|---|---:|---:|---|---|---:|---:|
| astra_medium | deliberative | 11 | 0 | 5/5 (100%) | 6/6 (100%) | 5 | 0.0 |
| astra_medium | persistent | 16 | 0 | 8/8 (100%) | 8/8 (100%) | 8 | 0.0 |
| astra_medium | sequential | 16 | 0 | 8/8 (100%) | 8/8 (100%) | 8 | 0.0 |
| gemma_4_31b | deliberative | 0 | 0 | — | — | 0 | — |
| gemma_4_31b | persistent | 0 | 0 | — | — | 0 | — |
| gemma_4_31b | sequential | 0 | 0 | — | — | 0 | — |
| luna_medium | deliberative | 13 | 1 | 5/7 (71%) | 4/6 (67%) | 6 | -16.7 |
| luna_medium | persistent | 14 | 2 | 5/7 (71%) | 4/7 (57%) | 6 | -16.7 |
| luna_medium | sequential | 15 | 1 | 5/7 (71%) | 5/8 (62%) | 7 | -14.3 |
| qwen_3_8_27b | deliberative | 0 | 0 | — | — | 0 | — |
| qwen_3_8_27b | persistent | 2 | 0 | 1/1 (100%) | 1/1 (100%) | 1 | 0.0 |
| qwen_3_8_27b | sequential | 1 | 0 | 1/1 (100%) | — | 0 | — |
| sol_medium | deliberative | 10 | 0 | 5/5 (100%) | 5/5 (100%) | 5 | 0.0 |
| sol_medium | persistent | 15 | 0 | 8/8 (100%) | 7/7 (100%) | 7 | 0.0 |
| sol_medium | sequential | 15 | 0 | 7/7 (100%) | 8/8 (100%) | 7 | 0.0 |
| terra_medium | deliberative | 16 | 0 | 7/8 (88%) | 8/8 (100%) | 8 | 12.5 |
| terra_medium | persistent | 17 | 0 | 8/8 (100%) | 8/9 (89%) | 8 | -12.5 |
| terra_medium | sequential | 15 | 1 | 7/7 (100%) | 8/8 (100%) | 7 | 0.0 |

Matched differences use only replicates with both versions completed. Separate recovery columns use all completed runs in each version.

## Supporting metrics

R: category-balanced planted-finding recall; P: independently confirmed fraction of accepted claims; F1*: mean per-run harmonic discovery score; E: exploration coverage; B: evidence responsiveness (available runs only). R and P are percentages; F1*, E, and B are on 0–100 scales.

| Model | Workflow | R % | P % | F1* | E | B (n) | Mean calls | Mean known output tokens |
|---|---|---:|---:|---:|---:|---|---:|---:|
| astra_medium | deliberative | 66.7 | 96.2 | 78.7 | 64.1 | 99.0 (11) | 303.6 | 265305.5 |
| astra_medium | persistent | 67.7 | 93.9 | 78.6 | 64.4 | 94.7 (16) | 104.0 | 88291.1 |
| astra_medium | sequential | 70.8 | 92.0 | 79.7 | 65.6 | 97.3 (14) | 106.7 | 81971.2 |
| gemma_4_31b | deliberative | — | — | — | — | — (0) | — | — |
| gemma_4_31b | persistent | — | — | — | — | — (0) | — | — |
| gemma_4_31b | sequential | — | — | — | — | — (0) | — | — |
| luna_medium | deliberative | 33.8 | 90.3 | 45.4 | 39.7 | 86.8 (8) | 304.1 | 223319.9 |
| luna_medium | persistent | 42.1 | 85.0 | 53.2 | 45.0 | 83.3 (8) | 102.3 | 72988.9 |
| luna_medium | sequential | 43.0 | 88.1 | 56.0 | 47.1 | 74.9 (8) | 102.5 | 71019.0 |
| qwen_3_8_27b | deliberative | — | — | — | — | — (0) | — | — |
| qwen_3_8_27b | persistent | 66.7 | 100.0 | 80.0 | 61.0 | 61.1 (1) | 105.0 | 181035.5 |
| qwen_3_8_27b | sequential | 33.3 | 100.0 | 50.0 | 60.9 | 66.7 (1) | 104.0 | 373135.0 |
| sol_medium | deliberative | 66.7 | 94.3 | 78.1 | 61.4 | 91.7 (10) | 301.1 | 543048.5 |
| sol_medium | persistent | 66.7 | 94.5 | 78.1 | 61.8 | 92.9 (13) | 102.2 | 170331.0 |
| sol_medium | sequential | 66.7 | 91.9 | 77.2 | 61.6 | 87.1 (14) | 101.4 | 175103.9 |
| terra_medium | deliberative | 65.3 | 97.5 | 77.8 | 59.7 | 93.6 (11) | 300.2 | 165841.8 |
| terra_medium | persistent | 63.7 | 98.0 | 76.9 | 64.0 | 92.3 (13) | 100.5 | 60894.8 |
| terra_medium | sequential | 65.6 | 98.5 | 78.6 | 62.3 | 93.1 (10) | 100.2 | 61145.2 |

Known token counts are lower bounds where provider or infrastructure usage is missing; missingness is recorded per run in run_metrics.csv. Tokens and calls here cover completed runs only. Report SHA-256 checks passed for every included terminal run.
