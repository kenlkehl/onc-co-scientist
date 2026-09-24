# NSCLC DepMap masking audit and harness corrections

The September 23 Qwen grid uses `Inferact/Qwen3.8-Flash-Next-NVFP4`, thinking enabled,
reasoning effort xhigh, and the Qwen thinking sampling profile on
`http://sn4622130540:8001/v1`. The original design is 40 persistent runs, followed by
40 sequential and 40 deliberative runs: ten repeats in each expected/surprising ×
masked/unmasked cell, 25 iterations per run. Persistent is currently running at
concurrency 40. No full run had finished at the audit checkpoint.

The complete local campaign, source snapshot, configuration, traces, and private
audit are under `outputs/nsclc_depmap_qwen38_flash_next_20260923/`.

## Equal-budget interim scores

Forty snapshots captured at 2026-09-23 23:02 UTC were truncated to exactly six
completed iterations and scored offline, without sending feedback to the agents.
D is category-balanced discovery F1 on a 0–100 scale; all ten assigned repeats
remain in each mean.

| Version | Named D | Masked D |
|---|---:|---:|
| Expected | 54.8 | 9.5 |
| Surprising | 64.8 | 10.5 |

The audit found exact numerical parity for 4,894 observed comparisons after
renaming, equal validation seeds, and unchanged confirmation/scoring for all 40
accepted-claim sets after transforming them between naming conditions. Correct
oracle queries recover all six targets in both versions and naming conditions.

The main observed gap was target selection. By iteration six, named agents tested
each of MSI–WRN, SMARCA4–SMARCA2, and KEAP1–NFE2L2 in 18/20 runs, versus 0/20,
2/20, and 4/20 for masked agents. Query counts and invalid-analysis counts were
similar; masked agents needed fewer repairs. Neither condition found the exact
focal subgroup by this checkpoint.

## Clinical versus DepMap search space

| NSCLC package | Clinical | DepMap |
|---|---:|---:|
| Observations | 50,000 | 2,000 |
| Predictors excluding identifiers/outcomes | 43 | 65 |
| Outcomes | 1 | 16 |
| Predictor × outcome combinations | 43 | 1,040 |
| Binary predictor × outcome combinations | 21 | 496 |
| Planted targets | 6 | 6 |

Both allow 12 analyses per iteration. Initial prompts contain marginal column
summaries, not joint association screens. DepMap therefore has about 24 times as
many initial predictor–outcome pairs under the same query budget. These counts
omit thresholds, subgroup choices, interactions, and directions; they are a
search-space comparison, not a measured causal decomposition of score differences.

## Corrections for future campaigns

- Preserve `patient_id` and `cell_line_id` during masking. Filter identifiers and
  historical aliases from runtime summaries in both prompt interfaces.
- Publish the evaluator's outcome-specific absolute effect-size thresholds in
  task metadata and prompts. Current clinical thresholds are 0.10 natural-log PFS
  units; DepMap thresholds are 0.15 dependency-score units. Remove the incompatible
  generic 10% relative-difference instruction.
- Preserve standardized/binary assay descriptions, translating their column
  names to the same opaque aliases as the data.
- Version new packages as `appraisal-3.3.0` / `ledger-1.1.0`, and masking as
  `columns-and-text-levels-v2`. Historical masking maps remain readable.

Validation: 139 focused tests passed, including both modalities, both dataset
versions, all three workflows, legacy identifier aliases, and native-agent
integration. All 167 frozen artifacts in the original running grid still match
their recorded hashes. That grid continues under its original protocol; the fixes
must be evaluated in a separate campaign.
