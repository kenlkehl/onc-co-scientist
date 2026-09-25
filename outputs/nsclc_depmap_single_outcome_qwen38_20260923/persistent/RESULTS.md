# Persistent results

Completed 2026-09-24T12:42:35.215558+00:00

| Version | Masking | Completed | Failed | Focal recovery / 10 | Mean exact D | Calls | Known output tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| expected | unmasked | 10 | 0 | 0 | 66.304 | 1092 | 12,740,085 |
| surprising | unmasked | 10 | 0 | 0 | 53.888 | 1080 | 12,521,233 |
| expected | masked | 10 | 0 | 0 | 64.309 | 1085 | 14,093,379 |
| surprising | masked | 10 | 0 | 2 | 57.846 | 1084 | 14,982,056 |

All ten assigned repeats remain in each denominator. Scientific scores are retained for partially failed runs under the declared harness policy. See condition-specific expected_surprising_report.md files for full scoring and token-accounting limitations.
Repeats characterize model variability on one synthetic dataset pair; they do not estimate generalization across datasets.
