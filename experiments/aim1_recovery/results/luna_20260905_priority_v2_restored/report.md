# Aim 1 structured recovery v2

Finalized 250/250 planned attempts: 246 completed and 4 terminal failures. Failures remain in every primary denominator with zero recovery credit.

Primary recovery measures complete subgroup identity, correct outcome/exposure/direction and ≥90% subgroup precision and recall. Clinical treatment effects within the subgroup and treatment interactions both qualify. Strict recovery requires equivalent boundaries. Statistical support does not gate either identity endpoint.

Confirmation is secondary: finite linked discovery analysis plus held-out, correctly signed evidence for the candidate's declared contrast, with alpha=0.05/[j(j+1)] per distinct claim. Interaction confirmation counts only explicitly submitted treatment interactions. Novelty judging is optional and was not used by this report.

| Family | Condition | N | Identity | Strict identity | Confirmed | Interaction confirmed |
|---|---|---:|---:|---:|---:|---:|
| clinical | named | 100 | 3 | 3 | 3 | 2 |
| clinical | anonymized | 100 | 1 | 1 | 1 | 0 |
| depmap | named | 25 | 1 | 0 | 1 | 0 |
| depmap | anonymized | 25 | 0 | 0 | 0 | 0 |

Protocol and limitations

All research runs must complete 25 clinical or 10 DepMap iterations, including screening, multivariable exploration, refinement, and robustness. Records retain script/output hashes; these checks do not establish equal tokens/compute or certify scientific originality. Neutral examples are identical across naming conditions. Research sessions receive no recovery feedback. This is a revised protocol on fixed archived cohorts, not an isolated test of model ability or scoring alone.

Requested model: gpt-5.6-luna; reasoning: medium; service: priority. Actual models: ['gpt-5.6-luna']. See launch evidence and runtime metadata for capability/telemetry verification.

Restoration and protocol limitations

The runtime reset lost records after the 187-completion checkpoint. Conversation reported at least 197 completions and one unidentified failed attempt; its exact replicate cannot be recovered. Results from the reconstructed batch must be labelled accordingly.
Jobs 0188 and 0189 have non-atomic checkpoint receipt/record mismatches; retained with zero primary credit. Original checkpoint evidence is preserved.
Conservatively exclude jobs 0188 through 0203, the range potentially affected by post-checkpoint loss. This is not an exact mapping of the lost failure.
Fixed submitted iterations and saved analyses do not establish 25/10 separate adaptive LLM reasoning turns. At least one resumed run used templated rationale text across many iterations. Report this limitation without changing the frozen scientific recovery criteria.
Job 0219 failed a submitted output hash check; job 0229 reached its iteration cap without the required robustness action. Both remain in the denominator with zero credit, without altering their records or retrying.
Two orchestration/summary helper scripts for job 0196 were written at repository root. Copies are retained under recovery_evidence. The linked analysis artifacts pass the frozen record checks. Task isolation was instruction-based, not an operating-system boundary.

Sensitivity excluding documented or potentially reset-affected jobs

| Family | Condition | Excluded | Remaining N | Identity | Strict |
|---|---|---:|---:|---:|---:|
| clinical | anonymized | 4 | 96 | 1 | 1 |
| clinical | named | 11 | 89 | 3 | 3 |
| depmap | anonymized | 0 | 25 | 0 | 0 |
| depmap | named | 1 | 24 | 1 | 0 |

Diagnostics and reproducibility

Only three named and one masked clinical runs submitted any complete target subgroup; all four received primary credit. Three named DepMap runs submitted complete subgroup structures, but only one met the membership precision/recall thresholds. These are post hoc descriptive diagnostics, not changes to the scoring rules. The fixed-budget reconstructed batch has lower recovery than the earlier short-budget v2-rescored batch (clinical 25/100 named and 12/100 masked); the present records do not isolate why.

All 250 input hashes and all 246 valid full-budget transcripts passed validation. Valid runs contain 5,415 submitted iterations and 7,668 structured claims. Restoring the final pre-scoring archive and rerunning the scorer without model calls reproduced run_scores.csv, group_scores.csv and structured_scores.json byte-for-byte.
