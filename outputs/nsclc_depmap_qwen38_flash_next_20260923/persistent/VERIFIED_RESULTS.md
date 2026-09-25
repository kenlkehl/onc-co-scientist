# Verified persistent results: 16-outcome NSCLC DepMap

| Version | Masking | Exact D | Exact R | P | Focal / 10 | Complete | Failed | Known output tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| expected | unmasked | 50.620 | 0.4444 | 0.7288 | 0 | 10 | 0 | 12,636,414 |
| surprising | unmasked | 57.219 | 0.5000 | 0.6717 | 0 | 9 | 1 | 12,340,488 |
| expected | masked | 19.336 | 0.1667 | 0.5104 | 0 | 10 | 0 | 12,387,626 |
| surprising | masked | 13.571 | 0.0833 | 0.6820 | 0 | 10 | 0 | 12,185,533 |

D (F1*) is on a 0–100 scale. R is category-balanced exact recovery/recall. Q is the machine-readable name for P, the fraction of final accepted claims confirmed on fresh evaluator data. R and Q are on a 0–1 scale. D and R average all ten assigned repeats; Q averages only runs with accepted claims, and its denominator is recorded as Q_defined_runs in VERIFIED_RESULTS.json.

- Scores and counts match the original RESULTS.json and all 40 individual scientific reports; report hashes verified. All assigned repeats remain in denominators, including the partially failed run.
- Known input/output tokens include saved calls from a run whose aggregate usage is null because one timeout has unknown usage. The original generated summary treats that null aggregate as zero and undercounts known output tokens; the corrected known totals are reported here without changing scientific scores or frozen artifacts.
- Known tokens are not complete physical usage: 70 interrupted unfinished requests, one recorded timeout, and any failed physical SDK retry attempts have unavailable usage.
- Repeats measure model variability on one dataset pair, not generalization across datasets.
- Original 16-outcome frozen prompts retain the masked row-identifier exposure, missing masked assay descriptions, and relative-versus-absolute threshold mismatch documented in the masking audit. The composite campaign uses the corrected protocol.

Execution and resource totals:

- calls: 4,254
- successful_stages: 3,999
- expected_stages: 4,000
- attempts_with_errors: 255
- repaired_stages: 242
- exhausted_stages: 1
- degraded_stages: 0
- known_input_tokens: 262,555,037
- known_output_tokens: 49,550,061

P is defined for 9/10 repeats in each masked cell and 10/10 repeats in each unmasked cell.
