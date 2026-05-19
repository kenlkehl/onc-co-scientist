# Oncology Co-Scientist Benchmark — Batch Scoring Report

- **Bundles scored:** 10 (5 named, 5 anonymized)
- **Replicates (total):** 10
- **Novelty %** (named only, unweighted mean of bundle means): 0.445
- **Buried discovery iteration — named** (lower = uncovers earlier; exact recoveries only): n/a
- **Buried discovery iteration — anonymized** (exact recoveries only): n/a
- **Fraction of replicates uncovering buried — named:** 0.000
- **Fraction of replicates uncovering buried — anonymized:** 0.000
- **Fraction near-or-better recovery — named:** 0.000
- **Fraction near-or-better recovery — anonymized:** 0.000
- **Fraction component-or-better recovery — named:** 0.400
- **Fraction component-or-better recovery — anonymized:** 0.000

## Per-bundle detail (mean ± SD; buried iteration over exact recoveries only)

### ds001_aml

#### Named (n_replicates=1)
- frac_novel: 0.143
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | qwen35-9b | codex-cli@1.0.0 | 0.143 | n/a | — | none | n/a | The effect of treatment_midostaurin on objective_response differs by idh1_mutation status.<br>The effect of treatment_midostaurin on objective_response varies by idh1_mutation status.<br>The effect of treatment_7plus3 on objective_response is strongest in patients with age_years=1. |

#### Anonymized (n_replicates=1)
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | qwen35-9b | codex-cli@1.0.0 | n/a | — | none | n/a |

### ds001_breast

#### Named (n_replicates=1)
- frac_novel: 0.615
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 1/1
- recovery levels: exact=0, near=0, component=1, none=0

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | qwen35-9b | codex-cli@1.0.0 | 0.615 | n/a | — | component | 2 | The effect of treatment_tamoxifen on pfs_months differs by her2_positive status.<br>The effect of treatment_tamoxifen on pfs_months differs by brca1_mutation status.<br>The effect of treatment_tamoxifen on pfs_months differs by brca2_mutation status. |

#### Anonymized (n_replicates=1)
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | qwen35-9b | codex-cli | n/a | — | none | n/a |

### ds001_crc

#### Named (n_replicates=1)
- frac_novel: 0.000
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | codex | codex-failure | 0.000 | n/a | — | none | n/a | — |

#### Anonymized (n_replicates=1)
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | qwen35-9b | codex-cli@1.0.0 | n/a | — | none | n/a |

### ds001_nsclc

#### Named (n_replicates=1)
- frac_novel: 0.710
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 1/1
- recovery levels: exact=0, near=0, component=1, none=0

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | codex-cli | codex-cli@1.0.0 | 0.710 | n/a | — | component | 2 | The effect of treatment_pembrolizumab on pfs_months differs by sex_female.<br>The effect of treatment_pembrolizumab on pfs_months differs by smoking_status.<br>The effect of treatment_pembrolizumab on pfs_months differs by ecog_ps. |

#### Anonymized (n_replicates=1)
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | codex | codex-failure | n/a | — | none | n/a |

### ds001_prostate

#### Named (n_replicates=1)
- frac_novel: 0.759
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | qwen35-9b | codex-cli@1.0.0 | 0.759 | n/a | — | none | n/a | The effect of treatment_enzalutamide on objective_response differs by brca2_mutation status.<br>The effect of treatment_enzalutamide on objective_response differs by msi_high status.<br>The effect of treatment_enzalutamide on objective_response differs by psma_high status. |

#### Anonymized (n_replicates=1)
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | qwen35-9b | codex-cli@1.0.0 | n/a | — | none | n/a |
