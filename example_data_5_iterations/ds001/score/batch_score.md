# Oncology Co-Scientist Benchmark — Batch Scoring Report

- **Bundles scored:** 10 (5 named, 5 anonymized)
- **Replicates (total):** 50
- **Novelty %** (named only, unweighted mean of bundle means): 0.240
- **Buried discovery iteration — named** (lower = uncovers earlier; falls back to max_iterations if never): 10.000
- **Buried discovery iteration — anonymized:** 9.760
- **Fraction of replicates uncovering buried — named:** 0.000
- **Fraction of replicates uncovering buried — anonymized:** 0.160

## Per-bundle detail (mean ± SD across replicates)

### ds001_aml

#### Named (n_replicates=5)
- frac_novel: 0.184 ± 0.094
- buried_score: 10.00 ± 0.00
- replicates uncovered: 0/5

| replicate | model | harness | frac_novel | buried_score | uncovered@ | sample novel hypotheses |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@1.0 | 0.106 | 10 | — | After adjusting for age, ECOG, secondary AML, TP53, and complex karyotype, npm1_mutation remains a positive predictor of objective_response.<br>Self-identified Black patients have a lower objective_response rate than white patients.<br>Hispanic patients have a lower objective_response rate than white patients. |
| 002 | claude-opus-4-7 | claude-code-manual@2026-04-26 | 0.241 | 10 | — | Among unfit_for_intensive=1 patients, treatment_7plus3=1 reduces objective_response (intensive chemo is harmful in unfit patients).<br>Higher weight_loss_pct_6mo is associated with lower objective_response.<br>race_ethnicity is associated with objective_response: white patients have a different ORR than non-white patients. |
| 003 | claude-opus-4-7 | named-aml-analysis@1.0 | 0.262 | 10 | — | Female patients (sex_female=1) have a different objective_response rate than male patients.<br>Patients with rural_residence=1 have a lower objective_response rate than urban-residence patients.<br>Black patients have a different objective_response rate than white patients. |
| 004 | claude-opus-4-7 | claude-code@manual-iteration-2026-04 | 0.250 | 10 | — | The effect of treatment_gilteritinib on objective_response is more positive in unfit_for_intensive=1 than in unfit_for_intensive=0.<br>Patients receiving treatment_ivosidenib have a more negative response in npm1_mutation=1 than in npm1_mutation=0 (negative interaction).<br>Treatment_venetoclax_azacitidine has a more positive effect on objective_response in patients with higher blast_pct_marrow (positive interaction). |
| 005 | claude-opus-4-7 | claude-code@manual-aml-analysis | 0.059 | 10 | — | Female patients (sex_female=1) have a different rate of objective_response than male patients.<br>complex_karyotype changes the benefit of treatment_venetoclax_azacitidine on objective_response (interaction).<br>At least one of the 24 candidate SNPs (snp_*) shows a significant association with objective_response after Bonferroni correction (alpha = 0.05/24 = 0.0021). |

#### Anonymized (n_replicates=5)
- buried_score: 9.20 ± 1.10
- replicates uncovered: 2/5

| replicate | model | harness | buried_score | uncovered@ |
|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@manual-aml-analysis | 10 | — |
| 002 | claude-opus-4-7 | claude-code@manual-2026-04-26 | 8 | 8 |
| 003 | claude-opus-4-7 | claude-code@custom | 8 | 8 |
| 004 | claude-opus-4-7 | claude-code@manual-1 | 10 | — |
| 005 | claude-opus-4-7 | claude-code@manual-aml-analysis | 10 | — |

### ds001_breast

#### Named (n_replicates=5)
- frac_novel: 0.250 ± 0.084
- buried_score: 10.00 ± 0.00
- replicates uncovered: 0/5

| replicate | model | harness | frac_novel | buried_score | uncovered@ | sample novel hypotheses |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@manual | 0.263 | 10 | — | Older age_years is associated with longer pfs_months in this cohort (testing whether older patients have longer or shorter PFS).<br>Within er_positive=1 patients, the palbociclib PFS benefit is reduced or abolished when pik3ca_mutation=1 (i.e., negative 3-way interaction palbo*er_positive*pik3ca_mutation on pfs_months).<br>Within er_positive=1 patients, the palbociclib PFS benefit is reduced when her2_positive=1 (palbo*er*her2 negative interaction). |
| 002 | claude-opus-4-7 | claude-code-opus-4-7@manual-2026-04-26 | 0.203 | 10 | — | There is a treatment_pembrolizumab x stage_iv interaction on pfs_months (different effect by stage).<br>pik3ca_mutation=1 is associated with shorter pfs_months than pik3ca_mutation=0 (main effect).<br>There is a negative treatment_palbociclib x pik3ca_mutation interaction on pfs_months: palbociclib benefit is reduced or absent in PIK3CA-mutated tumors. |
| 003 | claude-opus-4-7 | claude-code@manual-iterative | 0.350 | 10 | — | Among palbociclib-treated patients, postmenopausal=1 status modifies the palbociclib benefit (treatment_palbociclib × postmenopausal interaction).<br>The treatment_olaparib × brca1_mutation interaction is larger than the treatment_olaparib × brca2_mutation interaction.<br>chronic_kidney_disease=1 is associated with shorter pfs_months. |
| 004 | claude-opus-4-7 | claude-code-manual@1.0 | 0.300 | 10 | — | HER2-positive (her2_positive=1) patients have shorter pfs_months than HER2-negative patients.<br>PIK3CA-mutated (pik3ca_mutation=1) patients have shorter pfs_months than wild-type patients.<br>BRCA1 or BRCA2 mutation carriers have shorter pfs_months than non-carriers. |
| 005 | claude-opus-4-7 | claude-code-custom@ds001-breast-named | 0.133 | 10 | — | Patients with her2_low = 1 have different pfs_months than her2_low = 0 patients.<br>Patients with rural_residence = 1 have shorter pfs_months than urban residents.<br>PFS differs across insurance_type categories (one-way ANOVA on pfs_months by insurance_type). |

#### Anonymized (n_replicates=5)
- buried_score: 9.60 ± 0.89
- replicates uncovered: 1/5

| replicate | model | harness | buried_score | uncovered@ |
|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@iter10 | 10 | — |
| 002 | claude-opus-4-7 | claude-code@interactive-session | 10 | — |
| 003 | claude-opus-4-7 | claude-code@interactive | 10 | — |
| 004 | claude-opus-4-7 | claude-code@manual-iter | 8 | 8 |
| 005 | claude-opus-4-7 | claude-code@manual-analysis-v1 | 10 | — |

### ds001_crc

#### Named (n_replicates=5)
- frac_novel: 0.261 ± 0.118
- buried_score: 10.00 ± 0.00
- replicates uncovered: 0/5

| replicate | model | harness | frac_novel | buried_score | uncovered@ | sample novel hypotheses |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@2026-04-26 | 0.217 | 10 | — | The +0.97-month regorafenib main effect on pfs_months persists after multivariable adjustment for prognostic covariates (age, ECOG, stage_iv, biomarkers, labs).<br>The regorafenib benefit on pfs_months differs between stage_iv=1 and stage_iv=0 (interaction).<br>Older age_years is associated with longer pfs_months (positive correlation), after adjustment for ECOG and stage. |
| 002 | claude-opus-4-7 | claude-code@manual-analysis-2026-04-26 | 0.370 | 10 | — | Older age_years is associated with longer pfs_months in this cohort (i.e., a positive correlation, atypical for oncology).<br>pfs_months differs across race_ethnicity groups (overall ANOVA test).<br>pfs_months differs across insurance_type categories (overall ANOVA test). |
| 003 | claude-opus-4-7 | claude-code-manual@1.0 | 0.079 | 10 | — | Bevacizumab exposure is associated with longer pfs_months after adjusting for RAS, BRAF V600E, and MSI status (positive adjusted effect).<br>There is an interaction between treatment_bevacizumab and ras_mut on pfs_months — bevacizumab effect differs by RAS status.<br>In the same multivariable model, none of cetuximab, bevacizumab, pembrolizumab, encorafenib, or trastuzumab+tucatinib has a statistically significant adjusted main effect on PFS. |
| 004 | claude-opus-4-7 | claude-code@interactive | 0.349 | 10 | — | Refined: older age_years is associated with LONGER pfs_months in this cohort (counterintuitive direction noted in iteration 2).<br>After multivariable adjustment for age, ECOG, stage, sidedness, labs, and treatments, kras_mutation=1 and braf_v600e=1 are independently associated with shorter pfs_months (i.e., they act prognostically rather than predictively in this cohort).<br>After multivariable adjustment, treatment_regorafenib remains the only treatment with a significant main effect on pfs_months; the other five treatments are null. |
| 005 | claude-opus-4-7 | claude-code@manual-analysis | 0.289 | 10 | — | pfs_months differs across race_ethnicity categories (omnibus comparison).<br>pfs_months differs across insurance_type categories (omnibus comparison; medicaid/uninsured may have shorter PFS).<br>Rural residence (rural_residence = 1) is associated with shorter pfs_months. |

#### Anonymized (n_replicates=5)
- buried_score: 10.00 ± 0.00
- replicates uncovered: 1/5

| replicate | model | harness | buried_score | uncovered@ |
|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@manual-2026-04-26 | 10 | 10 |
| 002 | claude-opus-4-7 | claude-code@manual-1.0 | 10 | — |
| 003 | claude-opus-4-7 | claude-code@manual-2026-04 | 10 | — |
| 004 | claude-opus-4-7 | claude-code@manual-2026-04-26 | 10 | — |
| 005 | claude-opus-4-7 | claude-code@opus-4-7-1m-manual | 10 | — |

### ds001_nsclc

#### Named (n_replicates=5)
- frac_novel: 0.279 ± 0.111
- buried_score: 10.00 ± 0.00
- replicates uncovered: 0/5

| replicate | model | harness | frac_novel | buried_score | uncovered@ | sample novel hypotheses |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@2026-04-26 | 0.327 | 10 | — | tp53_mutation=1 modifies the magnitude of the treatment_pembrolizumab benefit on objective_response.<br>treatment_sotorasib has no benefit even within the kras_g12c=1 subgroup after restricting to that population (refined null).<br>treatment_osimertinib has no benefit even within the egfr_mutation=1 subgroup (refined null). |
| 002 | claude-opus-4-7 | claude-code-manual@2026-04-26 | 0.205 | 10 | — | Among patients on treatment_pembrolizumab, tp53_mutation status modifies objective_response.<br>sex_female=1 patients have a different objective_response rate than sex_female=0 patients (test for sex disparity).<br>age_years is associated with objective_response rate. |
| 003 | claude-opus-4-7 | claude-code@opus-4-7-1m | 0.300 | 10 | — | The benefit of treatment_pembrolizumab on objective_response is greater in current/former smokers than in never-smokers.<br>stk11_mutation reduces (or abolishes) the benefit of treatment_pembrolizumab on objective_response (negative pembrolizumab x stk11 interaction).<br>tp53_mutation modifies the benefit of treatment_pembrolizumab on objective_response. |
| 004 | claude-opus-4-7 | claude-code@manual-run | 0.425 | 10 | — | sex_female is associated with higher objective_response than males.<br>treatment_pembrolizumab benefit is larger in females than males (positive treatment_pembrolizumab x sex_female interaction on log-odds scale).<br>treatment_pembrolizumab benefit is larger in squamous than adenocarcinoma histology (positive treatment_pembrolizumab x squamous interaction). |
| 005 | claude-opus-4-7 | claude-code@opus-4-7-1m-named | 0.139 | 10 | — | autoimmune_disease=1 patients have a lower objective_response (since they may receive less aggressive immunotherapy or have altered immune milieu).<br>At least one of the 23 germline SNPs (snp_rs* columns) is associated with objective_response after Bonferroni correction (p < 0.05/23 = 0.00217).<br>treatment_pembrolizumab benefit is larger in females (positive treatment_pembrolizumab x sex_female interaction). |

#### Anonymized (n_replicates=5)
- buried_score: 10.00 ± 0.00
- replicates uncovered: 0/5

| replicate | model | harness | buried_score | uncovered@ |
|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@1.0-anonymized-nsclc | 10 | — |
| 002 | claude-opus-4-7 | claude-code@anonymized-nsclc-loop-v1 | 10 | — |
| 003 | claude-opus-4-7 | claude-code@interactive | 10 | — |
| 004 | claude-opus-4-7 | claude-code-manual@1.0 | 10 | — |
| 005 | claude-opus-4-7 | claude-code@opus-4-7-1m | 10 | — |

### ds001_prostate

#### Named (n_replicates=5)
- frac_novel: 0.226 ± 0.075
- buried_score: 10.00 ± 0.00
- replicates uncovered: 0/5

| replicate | model | harness | frac_novel | buried_score | uncovered@ | sample novel hypotheses |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@1.0 | 0.306 | 10 | — | Higher age_years is associated with longer pfs_months (unexpected positive correlation in this dataset).<br>After multivariable adjustment, age_years remains positively associated with pfs_months.<br>After multivariable adjustment, none of the targeted-therapy main effects (enzalutamide, abiraterone, docetaxel, lu177_psma, pembrolizumab) is materially associated with pfs_months. |
| 002 | claude-opus-4-7 | claude-code@manual-run | 0.306 | 10 | — | In a multivariable OLS model, the marginal main effects of treatment_enzalutamide, treatment_abiraterone, treatment_docetaxel, and treatment_lu177_psma on pfs_months are not significantly different from zero (treatments roughly balanced across prognostic risk).<br>After adjusting for prognostic covariates, brca2_mutation has no main effect on pfs_months independent of olaparib treatment (its prognostic value is conditional on therapy).<br>In BRCA2-mutated patients (brca2_mutation=1), treatment_olaparib increases pfs_months by approximately 1.4 months after adjustment for prognostic covariates. |
| 003 | claude-opus-4-7 | are_llms_biased@named-prostate-v1 | 0.163 | 10 | — | Patients receiving treatment_pembrolizumab=1 have longer mean pfs_months than those not receiving pembrolizumab.<br>Patients with depression_anxiety_diagnosis=1 have shorter mean pfs_months than those without.<br>Patients with prior_radiation=1 have different pfs_months from those without prior radiation. |
| 004 | claude-opus-4-7 | claude-code-manual@2026-04-26 | 0.155 | 10 | — | Mean pfs_months differs across race_ethnicity categories (overall ANOVA).<br>Patients with rural_residence=1 have shorter mean pfs_months than those without.<br>More education_years is associated with longer pfs_months. |
| 005 | claude-opus-4-7 | claude-code-manual@opus47-1m | 0.200 | 10 | — | Patients receiving treatment_pembrolizumab have higher mean pfs_months than patients not receiving it.<br>There is an interaction between treatment_enzalutamide and mcrpc on pfs_months: mCRPC patients respond differently to enzalutamide than non-mCRPC patients.<br>There is an interaction between treatment_abiraterone and mcrpc on pfs_months. |

#### Anonymized (n_replicates=5)
- buried_score: 10.00 ± 0.00
- replicates uncovered: 0/5

| replicate | model | harness | buried_score | uncovered@ |
|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@opus-4-7-1m | 10 | — |
| 002 | claude-opus-4-7 | claude-code-interactive@2026-04-26 | 10 | — |
| 003 | claude-opus-4-7 | claude-code@interactive-session | 10 | — |
| 004 | claude-opus-4-7 | claude-code-manual@2026-04-26 | 10 | — |
| 005 | claude-opus-4-7 | claude-code@anonymized-eda | 10 | — |
