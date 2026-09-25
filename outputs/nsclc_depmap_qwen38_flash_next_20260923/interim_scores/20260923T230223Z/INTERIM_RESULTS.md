# Provisional persistent scores

Snapshot: 2026-09-23T23:02:23.450582+00:00

No full 25-iteration runs had completed. Each row summarizes the latest completed iteration of all ten running repeats; iteration counts differ. These are provisional scores, not phase results.

| Version | Masking | Iterations reached | Focal / 10 | Mean exact D (F1*) | Recall R % | Confirmed-claim fraction P % |
|---|---|---:|---:|---:|---:|---:|
| expected | masked | 6–12 | 0 | 13.6 | 10.0 | 59.5 |
| expected | unmasked | 6–12 | 0 | 53.4 | 44.4 | 84.7 |
| surprising | masked | 10–15 | 0 | 10.7 | 6.7 | 73.6 |
| surprising | unmasked | 8–13 | 0 | 61.3 | 50.0 | 79.6 |

Focal recovery requires exact recovery and independent confirmation. R equally weights expected, neutral, and surprising planted-finding categories. P is the fraction of accepted claims that were tested and confirmed on fresh evaluator data. D/F1* is the harmonic mean of R and P on a 0–100 scale. Each metric is averaged separately across runs; mean D need not equal the harmonic mean of mean R and mean P.

Runs are unfinished and have unequal iteration budgets at this snapshot.
Current acceptances and confirmation thresholds can change before iteration 25.
One synthetic dataset pair; repeats measure stochastic model variability, not generalization across datasets.
E and B are omitted because this is an interim discovery-state evaluation, not a completed workflow report.

P is unavailable when a run has no accepted claims; its averages use these available-run denominators: expected/masked: 7/10; expected/unmasked: 10/10; surprising/masked: 6/10; surprising/unmasked: 10/10. R, D, and focal recovery retain all ten runs per cell.
