# Oncology Co-Scientist Benchmark — Batch Scoring Report

- **Bundles scored:** 10 (5 named, 5 anonymized)
- **Replicates (total):** 10
- **Novelty %** (named only, unweighted mean of bundle means): 0.448
- **Buried discovery iteration — named** (lower = uncovers earlier; exact recoveries only): n/a
- **Buried discovery iteration — anonymized** (exact recoveries only): n/a
- **Fraction of replicates uncovering buried — named:** 0.000
- **Fraction of replicates uncovering buried — anonymized:** 0.000
- **Fraction near-or-better recovery — named:** 0.000
- **Fraction near-or-better recovery — anonymized:** 0.000
- **Fraction component-or-better recovery — named:** 0.400
- **Fraction component-or-better recovery — anonymized:** 0.200

## Per-bundle detail (mean ± SD; buried iteration over exact recoveries only)

### ds001_aml

#### Named (n_replicates=1)
- frac_novel: 0.467
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | qwen35-9b-bf16 | codex-cli@pilot | 0.467 | n/a | — | none | n/a | Proportion of female patients differs between those with objective_response=1 and those with objective_response=0.<br>The effect of treatment_midostaurin on objective_response differs between patients aged 60+ and those under 60.<br>The effect of treatment_midostaurin on objective_response differs between female and male patients. |

#### Anonymized (n_replicates=1)
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | codex | codex-failure | n/a | — | none | n/a |

### ds001_breast

#### Named (n_replicates=1)
- frac_novel: 0.087
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | qwen35-9b | codex-cli | 0.087 | n/a | — | none | n/a | The effect of treatment_olaparib on PFS differs by pr_positive status.<br>The effect of treatment_palbociclib on PFS differs by her2_low status. |

#### Anonymized (n_replicates=1)
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 1/1
- recovery levels: exact=0, near=0, component=1, none=0

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | codex-cli | codex-cli@1.0.0 | n/a | — | component | 1 |

### ds001_crc

#### Named (n_replicates=1)
- frac_novel: 0.400
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | codex-cli@1.0.0 | oncology-analysis-harness | 0.400 | n/a | — | none | n/a | Patients with treatment_cetuximab=1 have lower pfs_months than those with treatment_cetuximab!=1.<br>Patients with treatment_bevacizumab=1 have lower pfs_months than those with treatment_bevacizumab!=1.<br>Patients with treatment_trastuzumab_tucatinib=1 have lower pfs_months than those with treatment_trastuzumab_tucatinib!=1. |

#### Anonymized (n_replicates=1)
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | codex | codex-failure | n/a | — | none | n/a |

### ds001_nsclc

#### Named (n_replicates=1)
- frac_novel: 0.671
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 1/1
- recovery levels: exact=0, near=0, component=1, none=0

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | codex-cli | codex-cli@1.0.0 | 0.671 | n/a | — | component | 3 | Patients with age_years in higher quartile have higher pfs_months.<br>Patients with sex_female in higher quartile have higher pfs_months.<br>Patients with smoking_status in higher quartile have higher pfs_months. |

#### Anonymized (n_replicates=1)
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | qwen35-9b | codex-cli@1.0.0 | n/a | — | none | n/a |

### ds001_prostate

#### Named (n_replicates=1)
- frac_novel: 0.615
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 1/1
- recovery levels: exact=0, near=0, component=1, none=0

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | codex-cli | codex-cli@1.0.0 | 0.615 | n/a | — | component | 3 | Patients with age_years=1 have a different objective_response rate compared to those with age_years=0.<br>Patients with sex_female=1 have a different objective_response rate compared to those with sex_female=0.<br>The effect of treatment_enzalutamide on objective_response differs between patients with brca2_mutation=1 and those with brca2_mutation=0. |

#### Anonymized (n_replicates=1)
- buried_score (exact recoveries only): n/a
- replicates uncovered: 0/1
- near-or-better recovery: 0/1
- component-or-better recovery: 0/1
- recovery levels: exact=0, near=0, component=0, none=1

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | codex | codex-failure | n/a | — | none | n/a |
