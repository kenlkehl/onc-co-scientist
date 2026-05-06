# Oncology Co-Scientist Benchmark — Batch Scoring Report

- **Bundles scored:** 2 (1 named, 1 anonymized)
- **Replicates (total):** 50
- **Novelty %** (named only, unweighted mean of bundle means): 0.628
- **Buried discovery iteration — named** (lower = uncovers earlier; exact only; no exact = max_iterations + 1): 9.640
- **Buried discovery iteration — anonymized** (exact only; no exact = max_iterations + 1): 10.880
- **Fraction of replicates uncovering buried — named:** 0.400
- **Fraction of replicates uncovering buried — anonymized:** 0.040
- **Fraction near-or-better recovery — named:** 0.960
- **Fraction near-or-better recovery — anonymized:** 0.920
- **Fraction component-or-better recovery — named:** 1.000
- **Fraction component-or-better recovery — anonymized:** 1.000

## Per-bundle detail (mean ± SD across replicates)

### ds001_depmap

#### Named (n_replicates=25)
- frac_novel: 0.628 ± 0.064
- buried_score: 9.64 ± 2.22
- replicates uncovered: 10/25
- near-or-better recovery: 24/25
- component-or-better recovery: 25/25
- recovery levels: exact=10, near=14, component=1, none=0

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | gpt-5-codex | codex-cli-local@2026-05-04 | 0.564 | 11 | — | near | 6 | Brain-lineage cell lines have more negative `dependency_EGFR` scores than non-brain cell lines.<br>Cell lines in the top quartile of `growth_rate_doublings_per_day` (>=0.964) have more negative `dependency_TMED10` scores than slower-growing cell lines.<br>Cell lines with `myc_amplification` equal to 1 have more negative `dependency_RPL11` scores than cell lines without `myc_amplification`. |
| 002 | gpt-5-codex | codex-cli-local-parquet-analysis | 0.605 | 11 | — | near | 4 | Cell lines with `nf1_loss == 1` have more negative `dependency_RIT1` scores than cell lines without NF1 loss.<br>Cell lines with `myc_amplification == 1` and `copy_number_8q24 >= 0.428` have more negative `dependency_RPL11` scores than other cell lines.<br>Cell lines with `myc_amplification == 1` have slightly more negative `dependency_DHX9` scores than cell lines without MYC amplification. |
| 003 | gpt-5-codex | codex-cli@local-parquet-node-analysis | 0.783 | 11 | — | near | 5 | Cell lines with `nf1_loss=1` have more negative `dependency_RIT1` scores than `nf1_loss=0` lines, and `lineage=brain` plus `nf1_loss=1` defines a stronger RIT1-dependent subgroup.<br>Cell lines with `lineage=colorectal`, `apc_mutation=1`, or high `wnt_activity_score` have more negative `dependency_KIF18A` scores than other cell lines.<br>Within `lineage=colorectal`, the subgroup `apc_mutation=1` and `smad4_loss=0` has much more negative `dependency_KIF18A`, while `smad4_loss=1` suppresses this KIF18A dependency. |
| 004 | gpt-5-codex | codex-local-parquet-node@2026-05-04 | 0.620 | 11 | — | near | 6 | Cell lines with `nf1_loss` equal to 1 have more negative `dependency_RIT1` scores than cell lines without `nf1_loss`.<br>After accounting for `nf1_loss`, cell lines with `kras_mutation` equal to 1 have more negative `dependency_RIT1` scores than cell lines without `kras_mutation`.<br>Cell lines with `cdkn2a_loss` equal to 1 have more negative `dependency_TMED10` scores than cell lines without `cdkn2a_loss`. |
| 005 | gpt-5-codex | codex-cli@0.128.0 | 0.674 | 11 | — | near | 5 | Cell lines with `nf1_loss` equal to 1 have more negative `dependency_RIT1` scores than cell lines with `nf1_loss` equal to 0.<br>Cell lines in the top quartile of `growth_rate_doublings_per_day` have more negative `dependency_TMED10` scores than slower-growing cell lines.<br>Cell lines in the top quartile of `growth_rate_doublings_per_day` have more negative `dependency_DHX9` scores than slower-growing cell lines. |
| 006 | gpt-5-codex | codex-cli@local-parquet-js-reader | 0.650 | 11 | — | near | 5 | Cell lines with `nf1_loss` equal to 1 have more negative `dependency_RIT1` scores than cell lines without NF1 loss.<br>The strongest `dependency_RIT1` subgroup is `nf1_loss` equal to 1 combined with low `rnaseq_MYC_log2_tpm` at or below 4.230.<br>Colorectal cell lines (`lineage` equal to colorectal) have more negative `dependency_KIF18A` scores than non-colorectal cell lines. |
| 007 | gpt-5-codex | codex-direct-js-parquet@2026-05-04 | 0.700 | 11 | — | component | 6 | The strongest BRAF dependency subgroup is cell lines where `braf_v600e` equals 1 and `brca2_loss` equals 1; this subgroup has more negative `dependency_BRAF` than all other cell lines.<br>EGFR dependency is concentrated in the subgroup where `egfr_amplification` equals 1 and `lineage_subtype` is "basal"; this subgroup has more negative `dependency_EGFR` than other cell lines.<br>KRAS dependency is strongest in cell lines where `kras_mutation` equals 1 and `lineage_subtype` is "her2_enriched"; this subgroup has more negative `dependency_KRAS` than other cell lines. |
| 008 | gpt-5-codex | codex-cli-local-node-parquet-analysis | 0.650 | 11 | — | near | 6 | Within `egfr_amplification` positive cell lines, high `rnaseq_MYC_log2_tpm` (top quartile, >=5.669) and low `mutation_burden` (bottom quartile, <=3.230) mark a subgroup with especially negative `dependency_EGFR`.<br>The subgroup with `erbb2_amplification` equal to 1, `lineage` equal to breast, and high `ifn_gamma_signature` (top quartile, >=0.602) has especially negative `dependency_ERBB2`.<br>Among `kras_mutation` positive lines, the subgroup with high `growth_rate_doublings_per_day` (top quartile, >=0.964) and `lineage` equal to hematopoietic has especially negative `dependency_KRAS`. |
| 009 | gpt-5-codex | codex-cli@standard-library-parquet-analysis | 0.516 | 10 | 10 | exact | 10 | Cell lines with `nf1_loss=1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss=0`, indicating RAS-pathway rewiring creates RIT1 dependency.<br>`lineage=colorectal` cell lines have more negative `dependency_KIF18A` scores than non-colorectal lines.<br>Cell lines with `apc_mutation=1` have more negative `dependency_KIF18A` scores than cell lines with `apc_mutation=0`. |
| 010 | gpt-5-codex | codex-cli@manual-node-parquet-analysis | 0.533 | 10 | 10 | exact | 10 | Cell lines with `nf1_loss == 1` have more negative `dependency_RIT1` scores than cell lines with intact NF1.<br>Cell lines with both `lineage == "brain"` and `nf1_loss == 1` have especially negative `dependency_RIT1` scores compared with all other cell lines.<br>Cell lines with `lineage == "colorectal"` have more negative `dependency_KIF18A` scores than non-colorectal cell lines. |
| 011 | gpt-5-codex | codex-cli-local@2026-05-04 | 0.564 | 11 | — | near | 6 | Brain-lineage cell lines have more negative `dependency_EGFR` scores than non-brain cell lines.<br>Cell lines in the top quartile of `growth_rate_doublings_per_day` (>=0.964) have more negative `dependency_TMED10` scores than slower-growing cell lines.<br>Cell lines with `myc_amplification` equal to 1 have more negative `dependency_RPL11` scores than cell lines without `myc_amplification`. |
| 012 | gpt-5-codex | codex-cli@manual-node-parquet-analysis | 0.533 | 10 | 10 | exact | 10 | Cell lines with `nf1_loss == 1` have more negative `dependency_RIT1` scores than cell lines with intact NF1.<br>Cell lines with both `lineage == "brain"` and `nf1_loss == 1` have especially negative `dependency_RIT1` scores compared with all other cell lines.<br>Cell lines with `lineage == "colorectal"` have more negative `dependency_KIF18A` scores than non-colorectal cell lines. |
| 013 | gpt-5-codex | codex-cli@local-node-parquet-analysis | 0.633 | 11 | — | near | 6 | Cell lines with `nf1_loss=1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss=0`, consistent with RAS-pathway dependency.<br>Cell lines with high `growth_rate_doublings_per_day` (>=0.964, the 75th percentile) have more negative `dependency_POLR2A` scores than slower-growing lines.<br>Cell lines with high `growth_rate_doublings_per_day` (>=0.964, the 75th percentile) have more negative `dependency_TMED10` scores than slower-growing lines. |
| 014 | gpt-5-codex | codex-cli@local | 0.629 | 8 | 8 | exact | 8 | The apparent association between `lineage=brain` and more negative `dependency_EGFR` is explained by `egfr_amplification` and will not remain independently negative after adjustment.<br>The apparent associations between `lineage=breast` or `lineage=gastric` and more negative `dependency_ERBB2` are explained by `erbb2_amplification` and will not remain independently negative after adjustment.<br>Among `kras_mutation` positive lines, `lineage=pancreatic` does not further strengthen `dependency_KRAS`; the `kras_mutation` by `lineage=pancreatic` interaction should be near zero. |
| 015 | gpt-5-codex | codex-cli@local-task-bundle | 0.667 | 6 | 6 | exact | 6 | Among `kras_mutation=1` lines, `lineage=pancreatic` does not further strengthen `dependency_KRAS`; the interaction between `kras_mutation` and `lineage=pancreatic` should be near zero.<br>Cell lines with `nf1_loss=1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss=0`, independent of growth and Cas9 activity.<br>Cell lines with high `growth_rate_doublings_per_day` (`>=0.964`, the upper quartile) have more negative `dependency_TMED10` scores than lower-growth lines. |
| 016 | gpt-5-codex | codex-cli-local-parquet-analysis | 0.706 | 7 | 7 | exact | 7 | Cell lines with `nf1_loss == 1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss == 0`, after accounting for `growth_rate_doublings_per_day` and `cas9_activity_score`.<br>Fast-growing cell lines with `growth_rate_doublings_per_day >= 0.964` have more negative `dependency_TMED10` scores than slower-growing cell lines.<br>Higher `cas9_activity_score` is associated with more negative `dependency_TMED10` scores, consistent with a screen-quality component in this outcome. |
| 017 | gpt-5-codex | codex-cli-local@2026-05-05 | 0.700 | 4 | 4 | exact | 4 | Cell lines with `nf1_loss=1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss=0`, after accounting for lineage, culture type, screen batch, Cas9 activity, and growth rate.<br>Cell lines with `lineage='colorectal'` and `apc_mutation=1` have more negative `dependency_KIF18A` scores than all other cell lines.<br>Cell lines with `lineage='colorectal'`, `apc_mutation=1`, and high `wnt_activity_score` have more negative `dependency_KIF18A` scores than all other cell lines. |
| 018 | gpt-5-codex | codex-cli-local-analysis | 0.618 | 11 | — | near | 5 | Cell lines with `nf1_loss=1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss=0`, after adjusting for lineage and screen covariates.<br>The `egfr_amplification` effect on `dependency_EGFR` is even more negative in `lineage=brain` than in non-brain lineages.<br>The `nf1_loss` effect on `dependency_RIT1` is even more negative in `lineage=brain` than in non-brain lineages. |
| 019 | gpt-5-codex | codex-cli@local-analysis | 0.703 | 10 | 10 | exact | 10 | Cell lines with `nf1_loss` = 1 have more negative `dependency_RIT1` scores than cell lines with `nf1_loss` = 0.<br>Among cell lines with `braf_v600e` = 1, `lineage` = skin has more negative `dependency_BRAF` scores than non-skin `braf_v600e` = 1 lines.<br>Among cell lines with `egfr_amplification` = 1, `growth_rate_doublings_per_day` >= 0.964 has more negative `dependency_EGFR` scores than lower-growth EGFR-amplified lines. |
| 020 | gpt-5-codex | codex-cli@local-analysis | 0.575 | 11 | — | near | 5 | Within `egfr_amplification=1` cell lines, `rnaseq_MYC_log2_tpm>=5.669` is associated with more negative `dependency_EGFR` than lower MYC RNA.<br>Within `egfr_amplification=1` cell lines, `lineage_subtype=luminal` suppresses EGFR dependency, giving less negative `dependency_EGFR` than other EGFR-amplified lines.<br>Cell lines with `nf1_loss=1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss=0`. |
| 021 | gpt-5-codex | codex-local-analysis@2026-05-05 | 0.600 | 5 | 5 | exact | 5 | Among cell lines with `braf_v600e == 1`, `lineage == skin` cell lines have more negative `dependency_BRAF` scores than non-skin BRAF V600E cell lines.<br>Among cell lines with `kras_mutation == 1`, `lineage == pancreatic` cell lines have more negative `dependency_KRAS` scores than non-pancreatic KRAS-mutant cell lines.<br>Cell lines with both `egfr_amplification == 1` and `lineage == brain` have more negative `dependency_EGFR` scores than all other cell lines. |
| 022 | gpt-5-codex | codex-cli@local | 0.600 | 11 | — | near | 5 | Among `erbb2_amplification=1` cell lines, `lineage_subtype=her2_enriched` has more negative `dependency_ERBB2` scores than other ERBB2-amplified cell lines.<br>Among `egfr_amplification=1` cell lines, `lineage=brain` has more negative `dependency_EGFR` scores than non-brain EGFR-amplified cell lines.<br>Cell lines with `nf1_loss=1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss=0`. |
| 023 | gpt-5-codex | codex-cli@local | 0.676 | 11 | — | near | 4 | Cell lines with `nf1_loss=1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss=0`, consistent with RAS-pathway rewiring producing RIT1 dependency.<br>A one-standard-deviation increase in `growth_rate_doublings_per_day` is associated with more negative `dependency_TMED10` scores, but the effect is expected to be weaker than for `dependency_POLR2A`.<br>A one-standard-deviation increase in `growth_rate_doublings_per_day` is associated with more negative `dependency_DHX9` scores, suggesting a proliferation-linked DHX9 dependency component. |
| 024 | gpt-5-codex | codex-cli@2026-05-05 | 0.600 | 11 | — | near | 5 | The `egfr_amplification` effect on `dependency_EGFR` is more negative in `lineage="brain"` than in non-brain lineages.<br>`smad4_loss=1` further increases KRAS dependency among `kras_mutation=1` lines, making `dependency_KRAS` more negative than expected from the `kras_mutation` and `smad4_loss` main effects.<br>Cell lines with `nf1_loss=1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss=0`, after adjusting for growth rate, Cas9 activity, and screen batch. |
| 025 | gpt-5-codex | codex-cli@local | 0.606 | 6 | 6 | exact | 6 | Cell lines with `nf1_loss=1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss=0`.<br>Cell lines with `apc_mutation=1` have more negative `dependency_KIF18A` scores than cell lines with `apc_mutation=0`.<br>Cell lines with `myc_amplification=1` have more negative `dependency_RPL11` scores than cell lines with `myc_amplification=0`. |

#### Anonymized (n_replicates=25)
- buried_score: 10.88 ± 0.60
- replicates uncovered: 1/25
- near-or-better recovery: 23/25
- component-or-better recovery: 25/25
- recovery levels: exact=1, near=22, component=2, none=0

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | gpt-5-codex | codex-node-parquet-analysis@2026-05-04 | 11 | — | near | 5 |
| 002 | gpt-5-codex | codex-cli@0.128.0 | 11 | — | near | 7 |
| 003 | gpt-5-codex | codex-local-parquet-node@2026-05-04 | 11 | — | near | 10 |
| 004 | gpt-5-codex | codex-cli@local-parquet-reader | 11 | — | near | 9 |
| 005 | gpt-5-codex | codex-cli-local-depmap-analysis@2026-05-04 | 11 | — | near | 4 |
| 006 | gpt-5-codex | codex-cli@local-node-parquet-analysis | 11 | — | near | 4 |
| 007 | gpt-5-codex | codex-cli@local-node-analysis | 11 | — | near | 5 |
| 008 | gpt-5-codex | codex-cli-local-powershell-parquet-analysis | 11 | — | component | 4 |
| 009 | gpt-5-codex | codex-cli@local-node-parquet-analysis | 11 | — | component | 3 |
| 010 | gpt-5-codex | codex-cli@local | 11 | — | near | 5 |
| 011 | gpt-5-codex | codex-cli@depmap-task-bundle | 11 | — | near | 6 |
| 012 | gpt-5-codex | codex-local-parquet-analysis@2026-05-04 | 11 | — | near | 6 |
| 013 | gpt-5-codex | codex-cli@depmap-task-bundle-root | 11 | — | near | 6 |
| 014 | gpt-5-codex | codex-cli@depmap-task-bundle-root | 11 | — | near | 6 |
| 015 | gpt-5-codex | codex-cli-manual-local-analysis | 11 | — | near | 4 |
| 016 | gpt-5-codex | codex-cli-manual-local-analysis | 11 | — | near | 4 |
| 017 | gpt-5-codex | codex-local-js-parquet-analysis | 8 | 8 | exact | 8 |
| 018 | gpt-5-codex | codex-cli@local | 11 | — | near | 4 |
| 019 | gpt-5-codex | codex-local-statistical-analysis@2026-05-05 | 11 | — | near | 4 |
| 020 | gpt-5-codex | codex-cli@local | 11 | — | near | 6 |
| 021 | gpt-5-codex | codex-cli-local@2026-05-05 | 11 | — | near | 5 |
| 022 | gpt-5-codex | codex-cli-local@2026-05-05 | 11 | — | near | 6 |
| 023 | gpt-5-codex | codex-cli@2026-05-05 | 11 | — | near | 4 |
| 024 | gpt-5-codex | codex-cli@local-analysis | 11 | — | near | 5 |
| 025 | gpt-5-codex | codex-cli@workspace | 11 | — | near | 5 |
