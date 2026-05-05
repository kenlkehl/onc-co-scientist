# Oncology Co-Scientist Benchmark — Batch Scoring Report

- **Bundles scored:** 2 (1 named, 1 anonymized)
- **Replicates (total):** 10
- **Novelty %** (named only, unweighted mean of bundle means): 0.627
- **Buried discovery iteration — named** (lower = uncovers earlier; exact only; no exact = max_iterations + 1): 11.000
- **Buried discovery iteration — anonymized** (exact only; no exact = max_iterations + 1): 11.000
- **Fraction of replicates uncovering buried — named:** 0.000
- **Fraction of replicates uncovering buried — anonymized:** 0.000
- **Fraction near-or-better recovery — named:** 0.600
- **Fraction near-or-better recovery — anonymized:** 1.000
- **Fraction component-or-better recovery — named:** 1.000
- **Fraction component-or-better recovery — anonymized:** 1.000

## Per-bundle detail (mean ± SD across replicates)

### ds001_depmap

#### Named (n_replicates=5)
- frac_novel: 0.627 ± 0.081
- buried_score: 11.00 ± 0.00
- replicates uncovered: 0/5
- near-or-better recovery: 3/5
- component-or-better recovery: 5/5
- recovery levels: exact=0, near=3, component=2, none=0

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | gpt-5-codex | codex-direct-depmap-analysis@2026-05-04 | 0.586 | 11 | — | component | 4 | Cell lines with `nf1_loss == 1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss == 0`.<br>The `dependency_RIT1` effect is stronger in the subgroup with `lineage == "brain"` and `nf1_loss == 1` than in all other cell lines.<br>Cell lines with `lineage == "colorectal"` have more negative `dependency_KIF18A` scores than non-colorectal cell lines. |
| 002 | gpt-5-codex | codex-direct-depmap-analysis@2026-05-04 | 0.586 | 11 | — | component | 4 | Cell lines with `nf1_loss == 1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss == 0`.<br>The `dependency_RIT1` effect is stronger in the subgroup with `lineage == "brain"` and `nf1_loss == 1` than in all other cell lines.<br>Cell lines with `lineage == "colorectal"` have more negative `dependency_KIF18A` scores than non-colorectal cell lines. |
| 003 | gpt-5-codex | codex-cli-local-node-parquet@2026-05-04 | 0.771 | 11 | — | near | 5 | Among `braf_v600e=1` lines, the subgroup with `growth_rate_doublings_per_day>=0.964` and `pten_loss=1` has especially negative `dependency_BRAF` scores compared with all other lines.<br>Among `egfr_amplification=1` lines, the subgroup with `growth_rate_doublings_per_day>=0.964` and `copy_number_8q24>=0.428` has especially negative `dependency_EGFR` scores compared with all other lines.<br>Among `erbb2_amplification=1` lines, breast-lineage lines with `ifn_gamma_signature>=0.519` have especially negative `dependency_ERBB2` scores compared with all other lines. |
| 004 | gpt-5-codex | codex-local-node-parquet-analysis@2026-05-04 | 0.606 | 11 | — | near | 7 | Cell lines in the top quartile of growth_rate_doublings_per_day (>= 0.964) have more negative dependency_DHX9 scores than all other cell lines.<br>Cell lines in the top quartile of growth_rate_doublings_per_day (>= 0.964) have more negative dependency_TMED10 scores than all other cell lines.<br>Brain lineage cell lines have more negative dependency_EGFR scores than non-brain lineage cell lines. |
| 005 | gpt-5-codex | codex-cli-node-parquet-analysis | 0.586 | 11 | — | near | 6 | Cell lines with `nf1_loss == 1` have more negative `dependency_RIT1` scores than cell lines with `nf1_loss == 0`.<br>Cell lines with `lineage == "colorectal"` have more negative `dependency_KIF18A` scores than non-colorectal cell lines.<br>Cell lines with `apc_mutation == 1` have more negative `dependency_KIF18A` scores than cell lines with `apc_mutation == 0`. |

#### Anonymized (n_replicates=5)
- buried_score: 11.00 ± 0.00
- replicates uncovered: 0/5
- near-or-better recovery: 5/5
- component-or-better recovery: 5/5
- recovery levels: exact=0, near=5, component=0, none=0

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | gpt-5-codex | codex-cli-manual-node-parquet-analysis | 11 | — | near | 6 |
| 002 | gpt-5-codex | codex-local-parquet-js@2026-05-04 | 11 | — | near | 7 |
| 003 | gpt-5-codex | codex-cli-manual-node-parquet-analysis | 11 | — | near | 6 |
| 004 | gpt-5-codex | codex-cli-manual-depmap-analysis | 11 | — | near | 5 |
| 005 | gpt-5-codex | codex-cli@2026-05-04-local | 11 | — | near | 7 |
