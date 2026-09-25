# Provisional persistent scores

Snapshot: 2026-09-23T23:02:23.450582+00:00

Each row summarizes all ten repeats at exactly six fully completed iterations. These are provisional scores, not phase results.

| Version | Masking | Iterations reached | Focal / 10 | Mean exact D (F1*) | Recall R % | Confirmed-claim fraction P % |
|---|---|---:|---:|---:|---:|---:|
| expected | masked | 6–6 | 0 | 9.5 | 6.7 | 38.8 |
| expected | unmasked | 6–6 | 0 | 54.8 | 44.4 | 89.6 |
| surprising | masked | 6–6 | 0 | 10.5 | 6.7 | 66.7 |
| surprising | unmasked | 6–6 | 0 | 64.8 | 50.0 | 92.3 |

Focal recovery requires exact recovery and independent confirmation. R equally weights expected, neutral, and surprising planted-finding categories. P is the fraction of accepted claims that were tested and confirmed on fresh evaluator data. D/F1* is the harmonic mean of R and P on a 0–100 scale. Each metric is averaged separately across runs; mean D need not equal the harmonic mean of mean R and mean P.

P is unavailable when a run has no accepted claims; its averages use these available-run denominators: expected/masked: 7/10; expected/unmasked: 10/10; surprising/masked: 4/10; surprising/unmasked: 10/10. R, D, and focal recovery retain all ten runs per cell.

All 40 runs are compared at exactly six completed iterations; the full 25-iteration runs remain unfinished.
Current acceptances and confirmation thresholds can change before iteration 25.
One synthetic dataset pair; repeats measure stochastic model variability, not generalization across datasets.
E and B are omitted because this is an interim discovery-state evaluation, not a completed workflow report.
