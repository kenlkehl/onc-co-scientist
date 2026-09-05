# Aim 1 structured recovery pilot

Completed 250/250 planned runs.

Models recorded in transcripts: gpt-5.6-luna. Harnesses: chatgpt-work-structured-v1.

For Work runs, the protocol requests Luna 5.6, medium reasoning, priority/Fast mode as advertised by the model tool. Each replicate starts with no inherited conversation. Per-response tier and token telemetry are unavailable for Work subagents. Endpoint runs record the supplied model and returned service metadata in runtime_metadata.json.

## Recovery

| Family | Condition | N | Primary | Strict | RMST iterations |
|---|---|---:|---:|---:|---:|
| clinical | named | 100 | 9/100 | 6/100 | 23.18 |
| clinical | anonymized | 100 | 5/100 | 5/100 | 24.01 |
| depmap | named | 25 | 0/25 | 0/25 | 10.0 |
| depmap | anonymized | 25 | 0/25 | 0/25 | 10.0 |

Excluded setup batch: {"n_launched": 12, "reason": "Incorrect dereferenced virtualenv path; excluded in entirety before scoring; all formal jobs fresh."}

Technical interruptions are retained in coordinator_state.json and summary.json. Affected jobs continued from their original saved work and submitted records, without recovery feedback. Some required a replacement agent context when the original runtime session became unavailable.

Sensitivity excluding jobs that required replacement agent contexts (the primary figure retains all planned runs):

| Family | Condition | Excluded | Remaining N | Primary | Strict |
|---|---|---:|---:|---:|---:|
| clinical | anonymized | 3 | 97 | 5/97 | 5/97 |
| clinical | named | 2 | 98 | 9/98 | 6/98 |
| depmap | anonymized | 3 | 22 | 0/22 | 0/22 |
| depmap | named | 0 | 25 | 0/25 | 0/25 |

Primary recovery requires the complete defining structure, subgroup precision and recall ≥0.90, correct outcome/exposure/contrast/direction, and independent support on the held-out 20% of rows. Strict recovery additionally requires equivalent boundaries. The jth distinct submitted claim receives alpha=0.05/[j(j+1)], preventing repeated held-out tests and future look-ahead in discovery time. Claims must have a linked executed discovery analysis recorded before credit. Novelty does not enter recovery scoring.

The original iteration caps and replicate counts are retained. Early stopping is allowed by the original task instructions. RMST is the mean of min(discovery iteration, task cap), with non-recovery assigned the cap; compare ratios only with the task-specific caps in mind.

This rerun changes the model, output protocol, numeric tolerance, confirmatory analysis, held-out data split, and examples. Original examples that mentioned planted DepMap variables were replaced with neutral placeholders. Thresholds were selected during development after inspecting archived examples and frozen before fresh experiments; this is not an independently preregistered validation.

The retained launch hashes are historical. The scoring implementation commit and hashes are recorded separately in implementation_for_scoring.json; software versions are in environment.json. The frozen scientific criteria are in protocol.json.

Validation notes, when present, are retained in validation_notes.json. Failed analyses recorded as NaN remain unchanged in the original records and cannot satisfy the finite-evidence requirement. Record comparison treats matching NaNs as identical while distinguishing them from null and finite values.

Work sessions share a filesystem; task isolation was implemented through separate copies and explicit instructions, not an OS access boundary. Archived cohorts and public column/value semantics were retained, including category labels in masked data. Counts describe stochastic repeats on these fixed cohorts, not independent biological datasets.

Files: run_scores.csv, group_scores.csv, structured_scores.json, transcripts/, aim1_recovery.png/.pdf/.svg, and discovery_curves.png/.pdf.
