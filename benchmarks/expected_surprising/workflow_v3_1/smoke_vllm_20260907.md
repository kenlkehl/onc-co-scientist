# Expected/surprising smoke test

We tested Inferact/Qwen3.8-27B-NVFP4 on 4 synthetic datasets. The agent judged effect sizes and conclusions without being given a minimum effect to aim for.

**96/96 steps completed.** There were 3 failed attempts, 3 successful retries, and 0 steps that could not be completed.

## Six rounds per run

Each run had **6 rounds**, called iterations in the saved data. Each round has four steps: explore, analyze, appraise, and synthesize.

**The two-round rule is a follow-up deadline, not a cap on the run.** The agent assesses validation evidence when it arrives, then reassesses it two rounds later. For example, a result received in round 2 is assessed again in round 4.

The agent can remain unresolved and keep investigating within the run. The response score uses its assessment at that scheduled check.

The standard full runs allow 25 rounds for clinical data and 10 for cell-line data. These shorter 6-round runs are smoke tests.

## Recall, precision, and F1*

- **Recall (R)** is sensitivity: how many of the embedded discoveries the agent tested, accepted, and independently confirmed. Expected, neutral, and surprising discoveries receive equal weight.
- **Precision (P)** is positive predictive value: the confirmed fraction of the agent's final accepted findings. A finding must have been tested and then confirmed on fresh data. Additional discoveries can count, even when they were not in the embedded target list.
- **F1\*** combines recall and precision using the familiar F1 formula, scaled to 100: **200 × R × P / (R + P)**.

\* This is an F1-style score. Recall is averaged across three discovery categories, and precision can include additional findings. The two measures therefore do not come from one ordinary classification confusion matrix.

**The agent does not have to validate before accepting.** The benchmark tests its final accepted findings on fresh data after the run. A finding accepted earlier can still count toward precision.

Confirmation still uses the benchmark's private effect cutoffs: 0.10 on natural-log PFS and 0.15 dependency-score units. An unconfirmed finding can have a real positive effect that does not clearly exceed that cutoff. Disagreement with this scoring rule does not by itself make the agent's scientific judgment unreasonable.

## Results

Recall and precision range from 0 to 1. F1\* ranges from 0 to 100.

| Run | Recall (R) | Precision (P) | F1\* | Confirmed / accepted findings |
|---|---:|---:|---:|---:|
| Clinical — expected | 0.00 | 0.00 | 0.00 | 0 / 3 |
| Clinical — surprising | 0.67 | 1.00 | 80.00 | 24 / 24 |
| Cell-line — expected | 0.44 | 0.54 | 48.87 | 19 / 35 |
| Cell-line — surprising | 0.00 | 0.68 | 0.00 | 23 / 34 |

Each expected/surprising pair reverses one featured finding, the focal discovery. Recovering it requires testing the exact comparison, accepting its correct direction, and confirming it on fresh data.

| Run | Focal discovery recovered? |
|---|---|
| Clinical — expected | No |
| Clinical — surprising | Yes |
| Cell-line — expected | No |
| Cell-line — surprising | No |

## Exploration and response to evidence

**Exploration coverage (E)** measures how broadly and how early the agent tested the embedded comparisons. It gives credit for testing, regardless of acceptance or direction.

**Evidence responsiveness (B)** measures whether the agent's assessment two rounds after validation agrees with the benchmark's private rule. It gives equal weight to three situations:

- The whole interval is above the private cutoff: the reference says accept.
- The whole interval is below the cutoff: the reference says reject.
- The interval crosses or touches the cutoff: the reference says unresolved.

Both scores range from 0 to 100. A response score is unavailable when a run has no scorable example of one of those three situations.

| Run | Coverage (E) | Response to evidence (B) |
|---|---:|---:|
| Clinical — expected | 0.00 | Unavailable |
| Clinical — surprising | 36.11 | Unavailable |
| Cell-line — expected | 44.44 | Unavailable |
| Cell-line — surprising | 0.00 | 33.33 |

**Across the 4 runs:** F1\* **32.22**, coverage **20.14**, and response to evidence **36.11**.

For F1\* and coverage, each run gets equal weight. For response to evidence, we average each of the three situations separately before combining them. The overall response score can therefore be available even when some individual runs are missing a situation.

These are small workflow checks. They are too few to rank models or establish whether the instruction change improved performance.

<details>
<summary>Run settings, detailed counts, and source files</summary>

## Run settings

- 125,000 completion tokens per call, including reasoning.
- Up to 2 retries for a failed step. This retry limit is separate from the number of rounds.
- 30-minute timeout per call; 2 runs can execute at once.
- 71.3 minutes elapsed; 99 calls; 1,482,429 completion tokens used.
- Server: http://sn4622130540:8000/v1. The controller can use other configured providers.
- Clinical: scheduled validation in rounds 2, 4; follow-up 2 rounds later; at most 10 voluntary requests. Validation alpha = 0.05 / 12.
- Cell-line: scheduled validation in rounds 2, 4; follow-up 2 rounds later; at most 10 voluntary requests. Validation alpha = 0.05 / 12.

The agent also assesses newly delivered evidence immediately. A result arriving too late for the follow-up does not enter the response score. Invalid results and interrupted follow-ups are also excluded; a missing assessment at a due deadline counts as disagreement.

If a step fails even after its retries, the run's primary F1\* and focal recovery are zero. Separate scores based on first attempts are retained in the saved results.

**Audit: passed.** The checks cover 20 dataset packages, the code used for the runs, cutoff removal from agent inputs, validation timing, required responses, and score arithmetic. [Audit details](/data1/ken/onc-co-scientist/benchmarks/expected_surprising/workflow_v3_1/smoke_audit.json).

## Counts behind the scores

Each response cell below is agreements with the private rule / scorable results. A 0/0 cell means no examples were available.

| Run | Above cutoff | Below cutoff | Crosses cutoff | Too late |
|---|---:|---:|---:|---:|
| Clinical — expected | 0 / 0 | 1 / 3 | 0 / 1 | 2 |
| Clinical — surprising | 1 / 1 | 0 / 4 | 0 / 0 | 2 |
| Cell-line — expected | 4 / 4 | 0 / 1 | 0 / 0 | 2 |
| Cell-line — surprising | 3 / 3 | 0 / 1 | 0 / 1 | 2 |

| Run | Recall: expected / neutral / surprising | Validation: requested / automatic |
|---|---|---:|
| Clinical — expected | 0.00 / 0.00 / 0.00 | 4 / 2 |
| Clinical — surprising | 1.00 / 0.00 / 1.00 | 5 / 2 |
| Cell-line — expected | 0.33 / 0.00 / 1.00 | 6 / 1 |
| Cell-line — surprising | 0.00 / 0.00 / 0.00 | 6 / 1 |

| Dataset type | F1\* | Coverage (E) | Response to evidence (B) |
|---|---:|---:|---:|
| Clinical | 40.00 | 18.06 | 38.89 |
| Cell-line | 24.44 | 22.22 | 33.33 |

There is only one expected/surprising pair for each dataset type here, so the report does not estimate uncertainty across different datasets.

## Errors and source files

- Clinical — expected, round 3, appraise, attempt 1: ValueError: Validation requires a previously registered hypothesis
- Clinical — expected, round 4, appraise, attempt 1: ValueError: Empty response; return a complete JSON stage record
- Clinical — surprising, round 6, synthesize, attempt 1: ValueError: Empty response; return a complete JSON stage record

The report labels are updated; the original numerical records retain their historical keys: P corresponds to Q, and F1\* corresponds to D. Values and scoring rules are unchanged.

- [Settings, versions, dates, and file checksums](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/manifest.json)
- [Summary across runs](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/paired_summary.json)
- [Detailed run summary](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/summary.json)
- Clinical — expected: [results](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/runs/nsclc_clinical-expected/report.json), [agent transcript](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/runs/nsclc_clinical-expected/transcript.jsonl), [call details](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/api_metadata/nsclc_clinical-expected.jsonl).
- Clinical — surprising: [results](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/runs/nsclc_clinical-surprising/report.json), [agent transcript](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/runs/nsclc_clinical-surprising/transcript.jsonl), [call details](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/api_metadata/nsclc_clinical-surprising.jsonl).
- Cell-line — expected: [results](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/runs/nsclc_depmap-expected/report.json), [agent transcript](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/runs/nsclc_depmap-expected/transcript.jsonl), [call details](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/api_metadata/nsclc_depmap-expected.jsonl).
- Cell-line — surprising: [results](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/runs/nsclc_depmap-surprising/report.json), [agent transcript](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/runs/nsclc_depmap-surprising/transcript.jsonl), [call details](/data1/ken/onc-co-scientist/data/expected_surprising_agent_judgment/smoke/20260907T190300Z/api_metadata/nsclc_depmap-surprising.jsonl).

</details>
