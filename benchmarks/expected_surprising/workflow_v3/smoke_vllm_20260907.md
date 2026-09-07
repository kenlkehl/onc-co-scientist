# Six-iteration appraisal-first workflow smoke report

This is the archived smoke with prescribed minimum effects. The [new smoke report](../workflow_v3_1/smoke_vllm_20260907.md) removes those prescriptions from agent inputs and spells out every metric name.

Completed September 7, 2026 using `Inferact/Qwen3.8-27B-NVFP4` at `http://sn4622130540:8000/v1`. The workflow itself remains provider independent.

The four assigned runs completed 96 of 96 stages. There were 4 failed attempts, 4 recovered stages, and 0 exhausted stages. All mechanical audit checks passed. All 20 package hashes match the frozen v2 release.

Each run used six iterations, automatic release in iterations **2 and 4**, a **two-iteration** response window, at most ten voluntary slots, and `V_max = 12` (`alpha = 0.05 / 12`). The allowance remained 125,000 completion tokens per call including reasoning, two retries per stage, and a 1,800-second timeout, with two workers and a shared paired replicate ID. Schedules, version IDs, dependency versions, source hashes, and assignments are in the manifest.

| Run | Successful stages | Repairs / exhausted | D | E | B | Exact confirmed focal |
|---|---:|---:|---:|---:|---:|---:|
| nsclc_clinical-expected | 24/24 | 1 / 0 | 50.00 | 64.81 | Unavailable | 1 |
| nsclc_clinical-surprising | 24/24 | 1 / 0 | 49.32 | 66.67 | Unavailable | 0 |
| nsclc_depmap-expected | 24/24 | 1 / 0 | 0.00 | 0.00 | Unavailable | 0 |
| nsclc_depmap-surprising | 24/24 | 1 / 0 | 50.00 | 33.33 | Unavailable | 0 |

| Run | R expected / neutral / surprising | R | Q | Accepted / tested-confirmed | Exact-or-near D |
|---|---|---:|---:|---:|---:|
| nsclc_clinical-expected | 1.00 / 0.00 / 0.00 | 0.33 | 1.00 | 51 / 51 | 50.00 |
| nsclc_clinical-surprising | 1.00 / 0.00 / 0.00 | 0.33 | 0.95 | 19 / 18 | 49.32 |
| nsclc_depmap-expected | 0.00 / 0.00 / 0.00 | 0.00 | 1.00 | 12 / 12 | 0.00 |
| nsclc_depmap-surprising | 0.50 / 0.00 / 0.50 | 0.33 | 1.00 | 30 / 30 | 50.00 |

| Run | Voluntary / automatic deliveries | Scheduled slots requested / available | Supported correct / n | Excluded correct / n | Ambiguous correct / n | Invalid / late / interrupted |
|---|---:|---:|---:|---:|---:|---:|
| nsclc_clinical-expected | 6 / 1 | 1 / 2 | 4 / 4 | 1 / 1 | 0 / 0 | 0 / 2 / 0 |
| nsclc_clinical-surprising | 6 / 1 | 1 / 2 | 4 / 4 | 1 / 1 | 0 / 0 | 0 / 2 / 0 |
| nsclc_depmap-expected | 5 / 1 | 1 / 2 | 3 / 3 | 1 / 1 | 0 / 0 | 0 / 2 / 0 |
| nsclc_depmap-surprising | 6 / 2 | 0 / 2 | 5 / 5 | 1 / 1 | 0 / 0 | 0 / 2 / 0 |

| Run | Focal first test | Correct direction registered | First accepted | Final confirmed |
|---|---:|---:|---:|---:|
| nsclc_clinical-expected | 1 | 1 | 1 | 1 |
| nsclc_clinical-surprising | 1 | Not reached | Not reached | 0 |
| nsclc_depmap-expected | Not reached | Not reached | Not reached | 0 |
| nsclc_depmap-surprising | Not reached | Not reached | Not reached | 0 |

| Run | Untested final acceptances | Unconfirmed accepted claims | Final evidence excludes claim | Voluntarily validated before first acceptance / tested first acceptances |
|---|---:|---:|---:|---:|
| nsclc_clinical-expected | 0 | 0 | 0 | 0 / 51 |
| nsclc_clinical-surprising | 0 | 1 | 0 | 0 / 19 |
| nsclc_depmap-expected | 0 | 0 | 0 | 0 / 12 |
| nsclc_depmap-surprising | 0 | 0 | 0 | 0 / 30 |

The expected cell-line run restricted all 65 valid discovery tests to `has_crispr_qc = 1`. The embedded targets have no eligibility restriction. Under the preserved exact/near matching rules, this yields D = E = 0, even though all 12 final claims were independently confirmed as additional claims (Q = 1). This is a comparison-scope mismatch, not an assertion that those associations are false.

B uses explicit assessments at the second subsequent iteration. Missing due decisions would count as errors; missing evidence classes leave complete B unavailable without redistributing their weights. Late voluntary evidence remains visible through its immediate response. The JSON reports also retain source/category/expectation subsets, coverage curves, first-acceptance validation histories, follow-up windows, and separate unsupported-acceptance counts.

| Reporting unit | D | E | B | Expected focal recovery | Surprising focal recovery | Difference, pp |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 37.33 | 41.20 | Unavailable | 0.50 | 0.00 | -50.00 |
| clinical | 49.66 | 65.74 | Unavailable | 1.00 | 0.00 | -100.00 |
| depmap | 25.00 | 16.67 | Unavailable | 0.00 | 0.00 | 0.00 |

Aggregation averages replicates, then gives the two versions and base datasets equal weight. Responsiveness is aggregated separately within each evidence class before calculating B. There is only one base dataset per modality in this smoke, so uncertainty intervals are unavailable. Conditional focal testing and acceptance after supportive validation are reported with their opportunity pools in the paired JSON. These are development workflow checks, not a formal model ranking.

The batch ran from 2026-09-07T17:14:20.929190+00:00 to 2026-09-07T18:28:50.300563+00:00, about 74.5 minutes. It recorded 100 provider calls and 1,590,402 completion tokens. 4 calls stopped at the token limit. Every failed attempt and repair remains in its transcript. Strict first-attempt discovery scores are retained separately from the primary retry-policy scores.

`nsclc_clinical-expected` technical attempts:

- Iteration 4, synthesize, attempt 1: ValueError: Empty response; return a complete JSON stage record

`nsclc_clinical-surprising` technical attempts:

- Iteration 5, appraise, attempt 1: ValueError: Empty response; return a complete JSON stage record

`nsclc_depmap-expected` technical attempts:

- Iteration 3, explore, attempt 1: ValueError: Empty response; return a complete JSON stage record

`nsclc_depmap-surprising` technical attempts:

- Iteration 4, appraise, attempt 1: ValueError: Empty response; return a complete JSON stage record

The earlier partial development batch `20260907T170155Z` was deliberately stopped to correct comparison-orientation diagnostics. Its files are retained and excluded from every number above. The final batch used the finalized runtime source; the audit verifies source and runner hashes.

Local validation: **61 tests passed**, including 40 historical cases and 21 workflow cases; Ruff and the dataset/package hash audit passed. Full-budget scripted runs exercised 25 clinical and 10 cell-line iterations through the generic provider interface.

Artifacts:

- [Machine-readable audit](smoke_audit.json), [package audit](package_audit.json), [package manifest](package_manifest.json).
- [Run manifest](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/manifest.json), [paired summary](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/paired_summary.json), [detailed run summary](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/summary.json).
- `nsclc_clinical-expected`: [report.json](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/runs/nsclc_clinical-expected/report.json), [transcript.jsonl](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/runs/nsclc_clinical-expected/transcript.jsonl), [API metadata](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/api_metadata/nsclc_clinical-expected.jsonl).
- `nsclc_clinical-surprising`: [report.json](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/runs/nsclc_clinical-surprising/report.json), [transcript.jsonl](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/runs/nsclc_clinical-surprising/transcript.jsonl), [API metadata](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/api_metadata/nsclc_clinical-surprising.jsonl).
- `nsclc_depmap-expected`: [report.json](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/runs/nsclc_depmap-expected/report.json), [transcript.jsonl](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/runs/nsclc_depmap-expected/transcript.jsonl), [API metadata](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/api_metadata/nsclc_depmap-expected.jsonl).
- `nsclc_depmap-surprising`: [report.json](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/runs/nsclc_depmap-surprising/report.json), [transcript.jsonl](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/runs/nsclc_depmap-surprising/transcript.jsonl), [API metadata](/data1/ken/onc-co-scientist/data/expected_surprising_workflow_refactor/smoke/20260907T171358Z/api_metadata/nsclc_depmap-surprising.jsonl).

Reproduce using the [guide commands](../../../docs/EXPECTED_SURPRISING_GUIDE.md#running-and-inspecting-an-evaluation). Use a new output directory and verify the endpoint’s served model first. Numerical DGP v2, old public packages, historical transcripts/scores, and the separate named/masked workflow remain unchanged.
