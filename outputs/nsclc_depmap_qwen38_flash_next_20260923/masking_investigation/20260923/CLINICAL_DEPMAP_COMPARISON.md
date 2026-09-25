# Clinical versus DepMap discovery search space

Offline inspection of the actual expected/surprising ledger packages. Dataset dimensions are measured from each expected-version public parquet; paired versions have the same schema. No live experiment settings or model inputs were changed.

| Profile | Rows | Total columns | Predictors excluding ID/outcomes | Outcomes | Predictor × outcome pairs | Binary predictors | Binary × outcome pairs | Planted targets | Overall / subgroup targets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aml_clinical | 50,000 | 47 | 45 | 1 | 45 | 23 | 23 | 6 | 3 / 3 |
| aml_depmap | 2,000 | 88 | 71 | 16 | 1,136 | 36 | 576 | 6 | 2 / 4 |
| breast_clinical | 50,000 | 49 | 47 | 1 | 47 | 25 | 25 | 6 | 0 / 6 |
| breast_depmap | 2,000 | 80 | 65 | 14 | 910 | 30 | 420 | 6 | 3 / 3 |
| crc_clinical | 50,000 | 45 | 43 | 1 | 43 | 22 | 22 | 6 | 0 / 6 |
| crc_depmap | 2,000 | 81 | 65 | 15 | 975 | 30 | 450 | 6 | 4 / 2 |
| nsclc_clinical | 50,000 | 45 | 43 | 1 | 43 | 21 | 21 | 6 | 4 / 2 |
| nsclc_depmap | 2,000 | 82 | 65 | 16 | 1,040 | 31 | 496 | 6 | 3 / 3 |
| prostate_clinical | 50,000 | 44 | 42 | 1 | 42 | 19 | 19 | 6 | 0 / 6 |
| prostate_depmap | 2,000 | 86 | 69 | 16 | 1,104 | 37 | 592 | 6 | 3 / 3 |

## Interpretation

- For NSCLC, the coarse predictor–outcome space is 1,040 versus 43, a 24.2-fold difference. Restricting to observed two-level predictors still gives 496 versus 21 pairs, a 23.6-fold difference. These counts are search-space proxies, not counts of all legal hypotheses: they omit categorical contrasts, continuous thresholds, eligibility, subgroup boundaries, interactions, and direction.
- Both task packages allow 12 analysis requests per iteration. Six iterations supply at most 72 analyses and the full 25-iteration run supplies at most 300. Exhaustively screening the simple binary pairs would require 42 iterations for NSCLC DepMap versus two for clinical, before follow-up. This assumes an efficient enumeration with no repeated comparisons or extra refinements; it is not a prediction of actual random sampling.
- Clinical tasks have one outcome, log_pfs_months. Clinical masking preserves that endpoint name. DepMap has 14–16 outcomes across these five packages and masks their gene identities as well as predictor identities. NSCLC DepMap has six distinct outcomes carrying the planted effects and ten additional outcomes.
- The model initially sees marginal summaries from frame.describe(include="all"), not a predictor–outcome association matrix. The exposed workflow requests individual comparisons; it provides no bulk all-pairs screening action. Masked agents therefore must allocate their limited queries without knowing the associations. Outcome variance and prevalence can guide selection, so this is not literally uniform random search.
- NSCLC clinical has four overall planted comparisons and two subgroup comparisons; NSCLC DepMap has three of each. Its focal target also requires two specific signature conditions, whereas the clinical focal target is overall. This adds complexity for NSCLC, but is not universal across profiles: some clinical packages have all six targets restricted to subgroups.
- DepMap has 2,000 observations versus 50,000 for clinical, which can reduce precision. However its planted coefficient is larger (0.8 versus 0.45) and residual noise smaller (0.18 versus 0.30). Coefficient/residual-noise ratios alone are not power comparisons because prevalence, subgroup size, other planted effects, and confirmation thresholds also matter. The earlier parity audit recovered all DepMap targets with the correct oracle queries; matched six-iteration run results show missed query selection rather than failure to detect queried planted targets.
- Consequently the present evidence supports a substantial search-budget explanation, not a quantified causal decomposition of the clinical–DepMap score gap. A controlled follow-up could equalize outcome count or give both modalities an identical prespecified bulk screening step, while keeping the existing experiment unchanged.

## Sources

- [Measured dimensions](dataset_dimensions.json)
- [Masked versus unmasked numerical and behavioral audit](REPORT.md)
- Original packages: `/data1/ken/onc-co-scientist/data/expected_surprising_ledger_25_iterations`.
- Frozen implementation: `outputs/nsclc_depmap_qwen38_flash_next_20260923/source/src/onc_co_scientist/expected_surprising/{masking.py,prompting.py,workflow.py}`.
