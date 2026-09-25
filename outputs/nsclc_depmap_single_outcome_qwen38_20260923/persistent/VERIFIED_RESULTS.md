# Verified persistent results: composite-outcome NSCLC DepMap

| Version | Masking | Exact D | Exact R | P | E | B | Focal / 10 | Complete | Failed | Known output tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expected | unmasked | 66.304 | 0.5556 | 0.8261 | 52.1333 | 81.0780 | 0 | 10 | 0 | 12,740,085 |
| surprising | unmasked | 53.888 | 0.4500 | 0.7352 | 40.9333 | 74.0000 | 0 | 10 | 0 | 12,521,233 |
| expected | masked | 64.309 | 0.5444 | 0.7922 | 45.9111 | 75.9325 | 0 | 10 | 0 | 14,093,379 |
| surprising | masked | 57.846 | 0.4833 | 0.7262 | 37.6000 | 70.5278 | 2 | 10 | 0 | 14,982,056 |

D (F1*) is on a 0–100 scale. R is category-balanced exact recovery/recall. Q is the machine-readable name for P, the fraction of final accepted claims confirmed on fresh evaluator data. R and Q are on a 0–1 scale. D and R average all ten assigned repeats; Q averages only runs with accepted claims, and its denominator is recorded as Q_defined_runs in VERIFIED_RESULTS.json.

E is exact-target exploration coverage over the full iteration budget, rewarding earlier testing. B is evidence responsiveness, balanced across supported, excluded, and ambiguous evidence. Both use a 0–100 scale. B evidence and eligible-run counts are recorded in the JSON; it is conditional on evidence encountered.

- Discovery scores and run counts match RESULTS.json and all 40 individual scientific reports; report hashes verified. All assigned repeats remain in D/R/E denominators, including any partially failed runs. P averages runs with accepted claims; B averages eligible runs within each evidence class, then gives the three classes equal weight.
- Known input/output tokens include saved calls when a run aggregate is null. Any known output tokens omitted by the generated summary are quantified separately, without changing scientific scores or frozen artifacts.
- Provider-reported known tokens do not establish complete physical usage: unobserved failed SDK retry attempts can have unavailable usage. Explicit interruption and unknown-usage counts are reported separately.
- Repeats measure model variability on one dataset pair, not generalization across datasets.
- Original 16-outcome frozen prompts retain the masked row-identifier exposure, missing masked assay descriptions, and relative-versus-absolute threshold mismatch documented in the masking audit. The composite campaign uses the corrected protocol.
- Known interrupted unfinished requests for this campaign/phase: 0; their partial usage is unavailable. Saved call records with unavailable usage: 0.

Execution and resource totals:

- calls: 4,341
- successful_stages: 4,000
- expected_stages: 4,000
- attempts_with_errors: 341
- repaired_stages: 324
- exhausted_stages: 0
- degraded_stages: 0
- known_input_tokens: 263,739,807
- known_output_tokens: 54,336,753
