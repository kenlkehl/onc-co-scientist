# DepMap masked versus unmasked audit

The evidence strongly favors a real advantage from meaningful names guiding the search in this benchmark. I found no data, numerical-analysis, validation-seed, or discovery-score error that penalizes identical claims merely because they are masked. There are prompt-level imperfections, so this is not a perfectly isolated test of semantic priors.

This audit uses the frozen persistent snapshots captured at 2026-09-23 23:02 UTC, with all 40 runs truncated to exactly six completed iterations for the performance comparison. No full 25-iteration run had finished at that snapshot. All work was offline: no model calls, agent-visible feedback, live journal edits, or changes to the running experiments.

## Equal-budget result

| Dataset version | Unmasked mean exact D | Masked mean exact D | Difference | Repeats per condition |
|---|---:|---:|---:|---:|
| expected | 54.8 | 9.5 | +45.3 | 10 |
| surprising | 64.8 | 10.5 | +54.3 | 10 |

D is the category-balanced discovery F1* score on a 0–100 scale. All ten repeats remain in each mean. The unmasked score is higher in 8/10 paired expected repeats and 10/10 surprising repeats. These are stochastic repeats on one synthetic dataset pair, not evidence of generalization across datasets. Focal exact confirmed recovery remains zero in all 40 six-iteration snapshots.

## Numerical and scoring checks

- Both observed datasets match their masked twins exactly after the declared bijective transform: values, rows, order, missingness, and types are preserved.
- All 4,894 distinct valid comparisons recorded across the 40 longer snapshots produce exactly equal numerical estimates and intervals before and after renaming. Their private-validation seeds are equal as well.
- Each of the 40 observed accepted-claim sets was transformed to the opposite naming condition and independently confirmed again. Numerical confirmation results and exact/near discovery R, P, D, and claim counts were unchanged in every run.
- Supplying all six correct planted hypotheses produces D = 100 and focal recovery = 1 in both naming conditions and both versions. Twelve paired voluntary-validation checks on these targets also match numerically.
- Initial requests have identical model, maximum output budget, temperature, sampling options, and retry settings.

See [parity_checks.json](parity_checks.json), [audit.py](audit.py), and [prompt_checks.json](prompt_checks.json).

## Where the gap appears

Counts below pool expected and surprising versions, giving 20 runs per naming condition. A tested target means the exact comparison was tested with valid evidence, regardless of the initially proposed direction.

| Planted comparison | Named: tested in iteration 1 | Masked: tested in iteration 1 | Named: tested by iteration 6 | Masked: tested by iteration 6 |
|---|---:|---:|---:|---:|
| MSI–WRN | 17/20 | 0/20 | 18/20 | 0/20 |
| SMARCA4–SMARCA2 | 16/20 | 0/20 | 18/20 | 2/20 |
| KEAP1–NFE2L2 | 17/20 | 0/20 | 18/20 | 4/20 |

The named agents propose biologically interpretable pairs immediately. Masked agents describe broad screening based on prevalence, balanced cell sizes, outcome variance, and arbitrary feature/outcome combinations. This is visible before evidence is supplied. Neither condition reaches the exact focal or neutral subgroup targets within the matched six-iteration checkpoint.

The analysis budgets are essentially equal: 1,426 executed comparisons for named runs versus 1,413 for masked runs out of 1,440 available per naming condition. Each has eight invalid analyses. Named runs actually have more rejected attempts needing repair (39 versus 15), and neither condition has an exhausted stage by this checkpoint. Thus unused analysis slots, invalid tests, or retry penalties do not explain the masked deficit.

All planted-target comparisons tested by this checkpoint are eventually represented in the observed confirmed-recovery totals. The gap therefore tracks which comparisons were investigated. Even replacing every run’s confirmed-claim fraction with 100% while holding its observed recall fixed only raises masked mean D to 10.0 (expected) and 10.7 (surprising), versus the observed 9.5 and 10.5. Most of this particular D gap is missing target coverage, not precision penalties.

See [behavior.json](behavior.json), [precision_sensitivity.json](precision_sensitivity.json), and [matched scores](matched_iteration_6/INTERIM_RESULTS.json).

## Harness issues and limits

1. **Identifier prompt asymmetry: a real masking defect.** `cell_line_id` is renamed to `feature_003`, but the prompt excludes identifiers only by the literal names `patient_id` and `cell_line_id`. Consequently masked prompts list 82 variables instead of 81, including a 2,000-level row identifier. No inspected hypothesis uses it as an exposure or condition; only two masked runs mention it in their narratives. There is no evidence that it explains the large target-selection gap, but it should be fixed for a clean future experiment.
   - [Mask construction](/data1/ken/onc-co-scientist/outputs/nsclc_depmap_qwen38_flash_next_20260923/source/src/onc_co_scientist/expected_surprising/masking.py:132); [prompt identifier filter](/data1/ken/onc-co-scientist/outputs/nsclc_depmap_qwen38_flash_next_20260923/source/src/onc_co_scientist/expected_surprising/workflow.py:580).

2. **Shared instruction/evaluator mismatch.** Both conditions receive a “10% relative” clinical-significance instruction. The frozen DepMap outcomes instead use an absolute score threshold `delta = 0.15`, and confirmation requires the signed interval’s lower bound to exceed it. One masked response explicitly interprets 10% of an outcome mean near −0.25 as roughly 0.025. This is a substantive ambiguity that can affect acceptance and precision in both conditions; it does not explain the first-iteration difference in target selection. Its total causal effect requires a controlled follow-up, not a claim from this audit.
   - [Scientific guidance](/data1/ken/onc-co-scientist/outputs/nsclc_depmap_qwen38_flash_next_20260923/source/src/onc_co_scientist/expected_surprising/prompting.py:76); [evidence classification](/data1/ken/onc-co-scientist/outputs/nsclc_depmap_qwen38_flash_next_20260923/source/src/onc_co_scientist/expected_surprising/scoring.py:156).

3. **Masking changes more than gene labels.** It also hides predictor identities and categorical levels and removes the paragraph describing synthetic research signatures/markers. The observed effect combines lost domain semantics with these contextual differences. A gene-name-only causal interpretation would require a tighter ablation.
   - [Instruction transformation](/data1/ken/onc-co-scientist/outputs/nsclc_depmap_qwen38_flash_next_20260923/source/src/onc_co_scientist/expected_surprising/masking.py:178).

For a future clean comparison, preserve and exclude identifiers consistently, retain equivalent nonsemantic assay/type descriptions, and align the stated effect-size criterion with the prespecified evaluator. Repeat matched runs after freezing those changes. The current grid remains unchanged so its accumulated history stays interpretable.
