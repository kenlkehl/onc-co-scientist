# Persistent results

Completed 2026-09-24T07:39:29.928558+00:00

| Version | Masking | Completed | Failed | Focal recovery / 10 | Mean exact D | Calls | Known output tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| expected | unmasked | 10 | 0 | 0 | 50.620 | 1071 | 12,636,414 |
| surprising | unmasked | 9 | 1 | 0 | 57.219 | 1082 | 12,340,488 |
| expected | masked | 10 | 0 | 0 | 19.336 | 1048 | 11,106,216 |
| surprising | masked | 10 | 0 | 0 | 13.571 | 1053 | 12,185,533 |

All ten assigned repeats remain in each denominator. Scientific scores are retained for partially failed runs under the declared harness policy. See condition-specific expected_surprising_report.md files for full scoring and token-accounting limitations.
Repeats characterize model variability on one synthetic dataset pair; they do not estimate generalization across datasets.
