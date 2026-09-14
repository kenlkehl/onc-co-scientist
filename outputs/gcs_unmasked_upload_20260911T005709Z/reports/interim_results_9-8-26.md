# Interim clinical workflow results — September 8, 2026

**Snapshot: September 08, 2026 at 10:14:21 PM EDT.** 130 of 360 runs had finished: **98 completed all 100 scientific stages; 32 finished with unrecovered errors.** Runs still active or queued are excluded, not scored as failures. This snapshot remains fixed while the experiment continues. All six models requested medium reasoning and standard/default service tier.

**Preliminary grant data.** The frozen Codex adapter sometimes discarded a completed response after a reconnect warning. Retrying could change later decisions, and exhausted retries trigger score penalties. The current run continues unchanged at your request; the fix applies to subsequent experiments. Token totals below recover usage from saved native completion events, but scientific scores retain the original decisions and penalties.

**What stands out:** Among all finished runs including failures, Astra persistent has the highest F1* (78.4), and Terra sequential has the highest B (99.0). Luna’s recovery is low despite frequent protocol completion. Qwen has only two fully successful runs so far. These are descriptive results from uneven, partially finished groups, not final rankings.

**New interpretation issue:** at least 7 of Gemma’s 12 fully completed runs used an inflated clinical threshold in their final narrative: about 0.206 or 0.25 log-PFS units. Six explicitly computed 10% of mean log-PFS; the seventh used the same inflated 0.25 threshold. A 10% PFS increase is about 0.095 log units. This can make a reproducible effect of about 0.15 log units look clinically unimportant. Protocol completion therefore does not guarantee correct interpretation.

## Runs that completed every stage

These are the runs marked `completed`: 25 iterations of explore, analyze, appraise, and synthesize. Recovered retries are allowed. This table describes successful executions and can favor conditions with more failures; the next table includes those failures.

R and P are percentages. F1*, E, and B use a 0–100 scale. Focal columns show recovered runs / observed runs for each dataset version. “Exp / Sur” gives the number of runs observed in each version. A dash means no runs or no eligible evidence, never zero.

| Model | Workflow | Completed | Exp / Sur | Recall R % | Precision P % | F1* | Exploration E | Evidence response B | Focal expected | Focal surprising |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Luna | persistent | 8 | 4 / 4 | 15.3 | 62.3 | 21.2 | 29.4 | 82.3 | 2/4 | 0/4 |
| Luna | sequential | 7 | 3 / 4 | 6.0 | 75.1 | 9.8 | 28.9 | 79.9 | 0/3 | 0/4 |
| Luna | deliberative | 8 | 4 / 4 | 8.3 | 58.9 | 13.3 | 29.2 | 81.3 | 1/4 | 0/4 |
| Terra | persistent | 8 | 4 / 4 | 53.5 | 97.7 | 66.4 | 57.6 | 89.6 | 3/4 | 3/4 |
| Terra | sequential | 8 | 4 / 4 | 50.0 | 95.6 | 63.6 | 62.6 | 99.0 | 4/4 | 2/4 |
| Terra | deliberative | 7 | 4 / 3 | 45.8 | 93.7 | 58.9 | 58.1 | 88.9 | 4/4 | 0/3 |
| Sol | persistent | 6 | 3 / 3 | 66.7 | 92.9 | 77.5 | 65.6 | 80.6 | 3/3 | 3/3 |
| Sol | sequential | 7 | 3 / 4 | 66.7 | 91.3 | 77.0 | 66.0 | 87.2 | 3/3 | 4/4 |
| Sol | deliberative | 2 | 1 / 1 | 41.7 | 91.0 | 53.3 | 40.7 | 76.7 | 1/1 | 0/1 |
| Astra | persistent | 9 | 4 / 5 | 66.7 | 95.4 | 78.4 | 65.6 | 91.3 | 4/4 | 5/5 |
| Astra | sequential | 8 | 4 / 4 | 62.5 | 96.9 | 75.2 | 65.4 | 100.0 | 4/4 | 4/4 |
| Astra | deliberative | 6 | 3 / 3 | 69.4 | 98.9 | 81.5 | 66.7 | 95.8 | 3/3 | 3/3 |
| Qwen 3.8 27B | persistent | 2 | 1 / 1 | 66.7 | 100.0 | 80.0 | 64.9 | — | 1/1 | 1/1 |
| Qwen 3.8 27B | sequential | 0 | 0 / 0 | — | — | — | — | — | — | — |
| Qwen 3.8 27B | deliberative | 0 | 0 / 0 | — | — | — | — | — | — | — |
| Gemma 4 31B | persistent | 6 | 3 / 3 | 55.6 | 94.0 | 68.4 | 59.5 | 63.9 | 3/3 | 3/3 |
| Gemma 4 31B | sequential | 4 | 2 / 2 | 66.7 | 95.8 | 78.5 | 63.7 | 63.9 | 2/2 | 2/2 |
| Gemma 4 31B | deliberative | 2 | 2 / 0 | 50.0 | 100.0 | 65.0 | 65.6 | 50.0 | 2/2 | — |

Qwen persistent has no B value in the successful-only table because its two successful runs had no eligible ambiguous-evidence events. Gemma deliberative has only expected-version completions so far; its successful-only scores describe that version alone.

## All finished runs, including failures

This is the more useful operational comparison. An unrecovered stage error makes F1* and focal recovery zero for that run, even when it found something useful. R, P, E, and B still describe the scientific record that exists. These are interim means, not the final balanced paired experiment.

| Model | Workflow | Finished / 20 | Failed | Exp / Sur | Recall R % | Precision P % | F1* | Exploration E | Evidence response B | Focal expected | Focal surprising |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Luna | persistent | 9/20 | 1 | 5 / 4 | 17.2 | 63.8 | 19.3 | 32.9 | 79.6 | 2/5 | 0/4 |
| Luna | sequential | 8/20 | 1 | 4 / 4 | 9.7 | 74.1 | 8.9 | 33.5 | 80.9 | 0/4 | 0/4 |
| Luna | deliberative | 8/20 | 0 | 4 / 4 | 8.3 | 58.9 | 13.3 | 29.2 | 81.3 | 1/4 | 0/4 |
| Terra | persistent | 8/20 | 0 | 4 / 4 | 53.5 | 97.7 | 66.4 | 57.6 | 89.6 | 3/4 | 3/4 |
| Terra | sequential | 8/20 | 0 | 4 / 4 | 50.0 | 95.6 | 63.6 | 62.6 | 99.0 | 4/4 | 2/4 |
| Terra | deliberative | 8/20 | 1 | 4 / 4 | 50.0 | 94.4 | 53.0 | 59.7 | 90.3 | 4/4 | 0/4 |
| Sol | persistent | 8/20 | 2 | 4 / 4 | 66.7 | 92.2 | 58.1 | 65.5 | 86.8 | 3/4 | 3/4 |
| Sol | sequential | 8/20 | 1 | 4 / 4 | 59.7 | 90.2 | 67.1 | 59.0 | 85.5 | 3/4 | 4/4 |
| Sol | deliberative | 7/20 | 5 | 3 / 4 | 48.6 | 92.4 | 16.6 | 52.5 | 85.1 | 1/3 | 0/4 |
| Astra | persistent | 9/20 | 0 | 4 / 5 | 66.7 | 95.4 | 78.4 | 65.6 | 91.3 | 4/4 | 5/5 |
| Astra | sequential | 9/20 | 1 | 4 / 5 | 62.5 | 94.8 | 67.3 | 65.4 | 97.2 | 4/4 | 4/5 |
| Astra | deliberative | 8/20 | 2 | 4 / 4 | 68.8 | 98.7 | 61.1 | 66.2 | 95.5 | 3/4 | 3/4 |
| Qwen 3.8 27B | persistent | 8/20 | 6 | 4 / 4 | 58.3 | 100.0 | 20.0 | 64.9 | 64.2 | 1/4 | 1/4 |
| Qwen 3.8 27B | sequential | 4/20 | 4 | 2 / 2 | 50.0 | 100.0 | 0.0 | 63.3 | 61.8 | 0/2 | 0/2 |
| Qwen 3.8 27B | deliberative | 1/20 | 1 | 0 / 1 | 66.7 | 100.0 | 0.0 | 65.3 | 66.7 | — | 0/1 |
| Gemma 4 31B | persistent | 8/20 | 2 | 4 / 4 | 54.2 | 94.2 | 51.3 | 59.9 | 64.6 | 3/4 | 3/4 |
| Gemma 4 31B | sequential | 6/20 | 2 | 3 / 3 | 61.1 | 97.2 | 52.3 | 63.7 | 63.7 | 2/3 | 2/3 |
| Gemma 4 31B | deliberative | 5/20 | 3 | 3 / 2 | 61.1 | 100.0 | 21.7 | 65.1 | 59.0 | 2/3 | 0/2 |

## Output tokens for all finished runs

Totals include peers, chairs, repairs, and provider retries. Means divide by the number of finished runs, including failures; they are ordinary per-run means. Native Codex completion usage replaces the coordinator total, so the same response is not counted twice. Reasoning tokens are already part of output tokens and are not added again. vLLM totals use the coordinator’s provider-reported usage.

| Model | Workflow | Original observed | Reconciled observed | Mean / finished run | Recovered tokens | Original calls missing usage | Native attempts without completion |
|---|---|---:|---:|---:|---:|---:|---:|
| Luna | persistent | 793,851 | 794,113 | 88,235 | 262 | 1 | 0 |
| Luna | sequential | 707,350 | 714,875 | 89,359 | 7,525 | 9 | 0 |
| Luna | deliberative | 1,978,154 | 1,979,833 | 247,479 | 1,679 | 2 | 0 |
| Terra | persistent | 440,553 | 440,553 | 55,069 | 0 | 0 | 0 |
| Terra | sequential | 459,002 | 459,002 | 57,375 | 0 | 0 | 0 |
| Terra | deliberative | 1,312,458 | 1,322,293 | 165,287 | 9,835 | 11 | 1 |
| Sol | persistent | 1,463,716 | 1,476,828 | 184,604 | 13,112 | 13 | 0 |
| Sol | sequential | 1,477,733 | 1,484,879 | 185,610 | 7,146 | 8 | 0 |
| Sol | deliberative | 4,049,875 | 4,119,176 | 588,454 | 69,301 | 33 | 0 |
| Astra | persistent | 740,177 | 741,620 | 82,402 | 1,443 | 2 | 1 |
| Astra | sequential | 696,932 | 701,722 | 77,969 | 4,790 | 8 | 1 |
| Astra | deliberative | 1,969,040 | 1,984,069 | 248,009 | 15,029 | 18 | 0 |
| Qwen 3.8 27B | persistent | 7,321,515 | 7,321,515 | 915,189 | 0 | 0 | 0 |
| Qwen 3.8 27B | sequential | 9,927,356 | 9,927,356 | 2,481,839 | 0 | 0 | 0 |
| Qwen 3.8 27B | deliberative | 4,257,100 | 4,257,100 | 4,257,100 | 0 | 0 | 0 |
| Gemma 4 31B | persistent | 2,179,672 | 2,179,672 | 272,459 | 0 | 0 | 0 |
| Gemma 4 31B | sequential | 2,320,209 | 2,320,209 | 386,702 | 0 | 0 | 0 |
| Gemma 4 31B | deliberative | 4,032,977 | 4,032,977 | 806,595 | 0 | 0 | 0 |

Across these 130 finished runs: **46,257,792 observed output tokens**, including **130,122 recovered from native records** beyond the original 46,127,670. 3 native attempts have no completion usage; their unknown consumption is not imputed as zero. These totals exclude ongoing and queued runs and therefore differ from the live progress log.

## Pipeline audit

Checked all 130 terminal scientific reports and their saved call records. Frozen source, input, and configuration checksums: 0 mismatches across 98 files. Audit discrepancies: 0.

The audit found no mismatches in report or call checksums, plan identities, dataset assignments, stage counts, call counts, or score calculations. All runs marked completed reached 25 iterations and committed each of the 100 stages exactly once. R/P/F1*, E, B, and focal recovery were recomputed from the saved scientific record using the frozen scoring code and matched the reports. This verifies scoring consistency; it does not rerun the models or regenerate independent validation datasets.

Codex native records contain **102 completed attempts with error warnings** and **3 attempts without a completion event** (3 explicitly report model capacity). The first category is the known adapter defect, not evidence that the server never produced an answer. Failures in the vLLM runs include malformed JSON and invalid stage forms or references; a finished run can preserve useful analyses while failing the protocol.

A review of final synthesis narratives found the inflated Gemma threshold in 3/6 completed persistent runs, 3/4 sequential runs, and 1/2 deliberative runs. This is a conservative count of explicit statements, not an exhaustive correctness score. For example, expected/persistent repeat 1 dismissed the marker-D effect of about 0.15 log units because it was below 0.206. [Reviewed examples and exact run IDs](/data1/ken/onc-co-scientist/outputs/interim_results_9-8-26/threshold_interpretation.json). For a future rerun, specifying that the 10% refers to PFS duration in months would clarify the intended meaning without prescribing a formula. The experiment and agent instructions were not changed during this audit.

Gemma and Qwen scored zero on the ambiguous-evidence component of B in every observed workflow. This is not an absent evidence category: across their finished runs, Gemma had 18 eligible ambiguous events (14 reject, 3 accept, 1 missing due assessment), and Qwen had 3 (all reject); the reference response was unresolved. Qwen’s component is therefore based on very little evidence. [Response counts](/data1/ken/onc-co-scientist/outputs/interim_results_9-8-26/ambiguous_evidence_responses.json).

Saved requests confirm the intended participant roles: single-agent calls in persistent/sequential workflows, and independent peers plus an authoritative chair in deliberative workflows. Peer drafts do not execute analyses. Persistent requests contained 2–12 messages as bounded history accumulated; sequential and deliberative requests each contained two messages, with deliberative drafts supplied to the chair in its system message. The audit checks Codex command settings for medium reasoning and default service tier. vLLM configuration requests the same settings, but this does not establish equal effective reasoning budgets across providers.

## Reading the metrics

- **R — recovery/recall:** fraction of embedded findings that were accepted, tested, and independently confirmed, giving equal weight to expected, neutral, and surprising findings. This is category-balanced recall. Exact matches are used here.
- **P — confirmed-claim fraction / precision:** accepted, tested, independently confirmed claims divided by all accepted claims. Additional discoveries can count. An agent may accept a claim before independent validation; validation determines whether the evaluator credits it. P is unavailable when there are no accepted claims.
- **F1\* — discovery performance:** each run gets `200 × R × P / (R + P)` using R/P as fractions; zero if both are zero or P is unavailable. An unrecovered stage error also forces it to zero. The asterisk marks category-balanced recall and this failure rule, which make it differ from ordinary F1. The table averages per-run scores, so it is not the harmonic mean of the displayed R and P.
- **E — exploration coverage:** at each of the 25 iterations, calculate the fraction of target comparisons tested validly in each category; average the three categories, then average all 25 iterations and multiply by 100. Early coverage earns more credit. Repeating a test adds no coverage; acceptance is unnecessary.
- **B — response to evidence:** orient each validation interval in the claim’s direction. If its lower bound exceeds the private minimum, the credited response is accept; if its upper bound is below the minimum, reject; if it spans the minimum, unresolved. Score the explicit assessment at synthesis two iterations after evidence release. Average accuracy separately within supported, excluded, and ambiguous evidence, then average the three categories and multiply by 100. Missing due assessments receive zero; invalid or too-late evidence is excluded. If a category is absent for the condition, B is unavailable rather than reweighted.

The **two-iteration window is a response deadline for B**, not a two-iteration run cap. Every run has a 25-iteration budget. Public instructions say that a relative outcome difference of at least 10% is clinically significant. The private evaluator still uses 0.10 natural-log PFS units (about a 10.5% geometric-mean difference); these thresholds are close but not identical.

Within each model/workflow, the tables first average available runs within each dataset version, then give the observed versions equal weight. B follows that procedure separately for each evidence category. Unavailable P values are omitted and their counts are retained in the audit JSON. If only one version is available, its value is shown descriptively. Unfinished runs and missing partners are never invented or given zero scores. The final paired analysis requires all assigned repeats; these interim tables do not estimate that final contrast. There is only one underlying clinical dataset pair, so no across-dataset confidence intervals or model ranking claims are warranted.

Persistent retains a bounded history of committed conversation (120,000 characters) plus the full current ledger and notes. Sequential starts each stage with a fresh conversation plus that ledger and notes. Deliberative uses two independent peers and one chair per stage. All share the same scientific analysis and validation budgets, but deliberative uses about 300 participant calls per clean run versus 100 for the other workflows.

## Reproducibility

[Fixed snapshot and run IDs](/data1/ken/onc-co-scientist/outputs/interim_results_9-8-26/snapshot.json) · [Per-run audit, denominators, and token reconciliation](/data1/ken/onc-co-scientist/outputs/interim_results_9-8-26/audit_results.json) · [Read-only audit script](/data1/ken/onc-co-scientist/outputs/interim_results_9-8-26/audit.py) · [Report renderer](/data1/ken/onc-co-scientist/outputs/interim_results_9-8-26/render.py)

[Frozen experiment configuration](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/config.yaml) · [Known adapter issue](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/PROVIDER_ADAPTER_ISSUE.md) · [Live progress](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/LIVE_PROGRESS.md)
