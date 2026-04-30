# Oncology Co-Scientist Benchmark — Batch Scoring Report

- **Bundles scored:** 10 (5 named, 5 anonymized)
- **Replicates (total):** 200
- **Novelty %** (named only, unweighted mean of bundle means): 0.330
- **Buried discovery iteration — named** (lower = uncovers earlier; falls back to max_iterations if never): 24.190
- **Buried discovery iteration — anonymized:** 23.590
- **Fraction of replicates uncovering buried — named:** 0.150
- **Fraction of replicates uncovering buried — anonymized:** 0.150

## Per-bundle detail (mean ± SD across replicates)

### ds001_aml

#### Named (n_replicates=20)
- frac_novel: 0.405 ± 0.087
- buried_score: 24.35 ± 2.48
- replicates uncovered: 5/20

| replicate | model | harness | frac_novel | buried_score | uncovered@ | sample novel hypotheses |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@named-aml-1 | 0.563 | 25 | — | Patients with flt3_tkd=1 have a different objective_response rate than patients with flt3_tkd=0.<br>There is a positive interaction between flt3_itd and treatment_midostaurin on objective_response in a logistic regression (interaction term coefficient > 0).<br>There is a positive interaction between flt3_itd and treatment_gilteritinib on objective_response in a logistic regression (interaction term coefficient > 0). |
| 002 | claude-opus-4-7 | claude-code-manual@2026-04-27 | 0.347 | 25 | — | Among tp53_mutation=1 patients, treatment_venetoclax_azacitidine=1 yields a lower objective_response rate than expected (negative interaction; TP53 abrogates venetoclax benefit).<br>Higher albumin_g_dl is associated with a higher probability of objective_response.<br>Higher crp_mg_l is associated with a lower probability of objective_response. |
| 003 | claude-opus-4-7 | claude-code@named-aml-analysis | 0.329 | 25 | — | Objective response rate differs across race_ethnicity categories (any difference, omnibus).<br>Black patients have a lower objective_response rate than white patients.<br>Hispanic patients have a different objective_response rate than non-Hispanic white patients. |
| 004 | claude-opus-4-7 | claude-code-named@manual | 0.426 | 25 | 25 | A combined inflammation index (low albumin <3.5 g/dL plus high CRP >10 mg/L) shows a stepwise decrease in ORR with increasing index value (0,1,2).<br>There is a positive interaction between IDH1 mutation and treatment_venetoclax_azacitidine: ORR is substantially higher with ven+aza specifically among IDH1+ patients than among IDH1- patients.<br>Objective response rate differs by race_ethnicity, with Black patients having lower ORR than White patients (reference). |
| 005 | claude-opus-4-7 | claude-code@manual | 0.388 | 14 | 14 | Female patients (sex_female=1) have a different objective_response rate than male patients.<br>Higher albumin_g_dl is associated with higher objective_response.<br>Greater weight_loss_pct_6mo is associated with lower objective_response. |
| 006 | claude-opus-4-7 | claude-code-manual@2026-04-27 | 0.347 | 25 | — | Among tp53_mutation=1 patients, treatment_venetoclax_azacitidine=1 yields a lower objective_response rate than expected (negative interaction; TP53 abrogates venetoclax benefit).<br>Higher albumin_g_dl is associated with a higher probability of objective_response.<br>Higher crp_mg_l is associated with a lower probability of objective_response. |
| 007 | claude-opus-4-7 | claude-code-interactive@2026-04-27 | 0.379 | 25 | 25 | There is a positive interaction between idh1_mutation and treatment_venetoclax_azacitidine: IDH1-mutated patients gain more benefit (higher objective_response rate) from venetoclax-azacitidine than IDH1-wildtype patients.<br>There is a positive interaction between npm1_mutation and treatment_venetoclax_azacitidine: NPM1-mutated patients gain more benefit from venetoclax-azacitidine than NPM1-wildtype patients.<br>There is a negative interaction between npm1_mutation and treatment_ivosidenib: NPM1-mutated patients respond worse to ivosidenib than NPM1-wildtype patients (i.e., NPM1+ patients should not be put on a non-IDH targeted therapy when alternatives exist). |
| 008 | claude-opus-4-7 | claude-code-manual@2026-04-27 | 0.347 | 25 | — | Among tp53_mutation=1 patients, treatment_venetoclax_azacitidine=1 yields a lower objective_response rate than expected (negative interaction; TP53 abrogates venetoclax benefit).<br>Higher albumin_g_dl is associated with a higher probability of objective_response.<br>Higher crp_mg_l is associated with a lower probability of objective_response. |
| 009 | claude-opus-4-7 | claude-code@manual-named | 0.333 | 23 | 23 | Higher albumin_g_dl is associated with higher objective_response (positive nutritional/inflammatory marker).<br>There is a positive interaction between treatment_venetoclax_azacitidine and npm1_mutation on objective_response (greater treatment benefit in NPM1-mutated patients).<br>Among NPM1-mutated patients, treatment_venetoclax_azacitidine recipients have higher objective_response than non-recipients. |
| 010 | claude-opus-4-7 | claude-code@aml-named-1 | 0.431 | 25 | — | Female sex (sex_female=1) is associated with a different probability of objective_response than male sex.<br>Higher crp_mg_l (systemic inflammation) is associated with lower objective_response.<br>Higher albumin_g_dl (better nutritional/host status) is associated with higher objective_response. |
| 011 | claude-opus-4-7 | claude-code@named-aml-iter25 | 0.301 | 25 | — | Female patients (`sex_female`=1) have a different objective_response rate than male patients.<br>There is a negative interaction between `treatment_7plus3` and `tp53_mutation`: 7+3 is less effective (relative) in TP53-mutant patients than TP53-wildtype.<br>Within TP53-mutant patients, those receiving `treatment_venetoclax_azacitidine` have a different objective_response rate than those not receiving it. |
| 012 | claude-opus-4-7 | claude-code@manual-analysis | 0.487 | 25 | — | Female sex (sex_female=1) is associated with a different objective_response rate compared to male patients.<br>treatment_7plus3 × tp53_mutation interaction: 7+3 effect on objective_response differs by tp53_mutation status.<br>treatment_venetoclax_azacitidine × tp53_mutation interaction: ven/aza effect differs by tp53_mutation status. |
| 013 | claude-opus-4-7 | claude-code@inline-harness | 0.542 | 25 | — | Patients with idh1_mutation==1 have a higher objective_response rate than patients without IDH1 mutation.<br>After adjusting for age, ECOG, AML genetics, baseline labs, and treatments, weight_loss_pct_6mo is independently associated with a lower log-odds of objective_response.<br>After adjusting for age, ECOG, AML genetics, baseline labs, and treatments, albumin_g_dl is independently associated with a higher log-odds of objective_response. |
| 014 | claude-opus-4-7 | claude-code-manual@2026-04-27 | 0.347 | 25 | — | Among tp53_mutation=1 patients, treatment_venetoclax_azacitidine=1 yields a lower objective_response rate than expected (negative interaction; TP53 abrogates venetoclax benefit).<br>Higher albumin_g_dl is associated with a higher probability of objective_response.<br>Higher crp_mg_l is associated with a lower probability of objective_response. |
| 015 | claude-opus-4-7 | claude-code-named-bundle@2026-04-28 | 0.479 | 25 | — | Among `idh2_mutation`=1 patients, `treatment_ivosidenib`=1 (off-target) yields a similar `objective_response` rate to no ivosidenib.<br>Among `idh1_mutation`=1 patients, `treatment_enasidenib`=1 (off-target) yields a similar `objective_response` rate to no enasidenib.<br>Higher `platelets_k_ul` at baseline is associated with higher `objective_response` rate. |
| 016 | claude-opus-4-7 | claude-code@inline-aml-2026-04-28 | 0.348 | 25 | — | There is a positive interaction (on the log-odds scale) between flt3_itd and treatment_midostaurin in predicting objective_response (i.e., midostaurin's effect is larger in FLT3-ITD+ patients).<br>There is a positive interaction (on the log-odds scale) between flt3_itd and treatment_gilteritinib in predicting objective_response.<br>There is a positive interaction between unfit_for_intensive and treatment_venetoclax_azacitidine on the log-odds of objective_response (effect larger in unfit subgroup). |
| 017 | claude-opus-4-7 | claude-code@interactive-1.0 | 0.294 | 25 | — | Patients with idh1_mutation have higher objective_response rate than patients without idh1_mutation.<br>Higher serum albumin (albumin_g_dl) is associated with higher objective_response (i.e., better nutritional status predicts response).<br>Patients with race_ethnicity == 'black' have lower objective_response than patients with race_ethnicity == 'white'. |
| 018 | claude-opus-4-7 | claude-code@aml-named-1.0 | 0.341 | 25 | — | tp53_mutation=1 patients have a smaller (or negative) ORR benefit from treatment_venetoclax_azacitidine than tp53_mutation=0 patients (negative interaction).<br>The favorable effect of npm1_mutation on ORR is attenuated when flt3_itd=1 (negative npm1_mutation × flt3_itd interaction).<br>heart_failure=1 is associated with lower ORR. |
| 019 | claude-opus-4-7 | claude-code-manual@2026-04 | 0.550 | 25 | — | Patients with flt3_tkd=1 have a higher objective_response rate than those without.<br>tp53_mutation-positive patients have a relatively lower objective_response benefit from treatment_venetoclax_azacitidine than TP53-negative patients (negative interaction).<br>complex_karyotype-positive patients have a relatively lower objective_response benefit from treatment_venetoclax_azacitidine than complex_karyotype-negative patients (negative interaction). |
| 020 | claude-opus-4-7 | claude-code-interactive@1.0 | 0.511 | 25 | 25 | Venetoclax/azacitidine treatment (treatment_venetoclax_azacitidine) shows a strong positive interaction with IDH1 mutation on objective_response — IDH1+ patients benefit much more from ven/aza than IDH1- patients do.<br>In a multivariable logistic model that simultaneously includes age, sex, ECOG, secondary AML, fitness, key cytogenetics/mutations, marrow blasts, WBC, albumin, weight loss, CRP, and the six treatments, ECOG performance status is the single strongest negative independent predictor of objective_response.<br>In the same multivariable model, IDH1 mutation remains a strong positive independent predictor of objective_response after adjustment. |

#### Anonymized (n_replicates=20)
- buried_score: 23.80 ± 5.37
- replicates uncovered: 2/20

| replicate | model | harness | buried_score | uncovered@ |
|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@manual-iter-2026-04-27 | 25 | — |
| 002 | claude-opus-4-7 | claude-code@anonymized-aml-v1 | 25 | — |
| 003 | claude-opus-4-7 | claude-code-interactive@manual | 1 | 1 |
| 004 | claude-opus-4-7 | claude-code-bundle@manual | 25 | — |
| 005 | claude-opus-4-7 | claude-code-bundle@manual | 25 | — |
| 006 | claude-opus-4-7 | claude-code@aml-anonymized-runner | 25 | — |
| 007 | claude-opus-4-7 | claude-code@interactive-2026-04 | 25 | — |
| 008 | claude-opus-4-7 | claude-code@opus-4-7-1m | 25 | — |
| 009 | claude-opus-4-7 | claude-code@aml-anonymized-1 | 25 | — |
| 010 | claude-opus-4-7 | claude-code@analysis-1 | 25 | — |
| 011 | claude-opus-4-7 | claude-code-manual@2026-04-27 | 25 | — |
| 012 | claude-opus-4-7 | claude-code-opus-4-7@1m-context | 25 | — |
| 013 | claude-opus-4-7 | claude-code-self@my-analysis-2026-04-28 | 25 | — |
| 014 | claude-opus-4-7 | claude-code-manual@aml-anonymized-2026-04-28 | 25 | — |
| 015 | claude-opus-4-7 | claude-code-opus-4-7@anonymized-aml | 25 | 25 |
| 016 | claude-opus-4-7 | claude-code-manual@1.0 | 25 | — |
| 017 | claude-opus-4-7 | claude-code@manual | 25 | — |
| 018 | claude-opus-4-7 | claude-code@manual-2026-04-28 | 25 | — |
| 019 | claude-opus-4-7 | claude-code-direct@2026-04-28 | 25 | — |
| 020 | claude-opus-4-7 | manual-claude-code-session@2026-04-28 | 25 | — |

### ds001_breast

#### Named (n_replicates=20)
- frac_novel: 0.259 ± 0.099
- buried_score: 22.80 ± 5.57
- replicates uncovered: 4/20

| replicate | model | harness | frac_novel | buried_score | uncovered@ | sample novel hypotheses |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-opus47-1m@manual-2026-04-27 | 0.434 | 15 | 15 | treatment_palbociclib benefit is attenuated or abolished in pik3ca_mutation=1 patients relative to pik3ca_mutation=0.<br>treatment_pembrolizumab effect on pfs_months differs between has_brain_mets=1 and has_brain_mets=0.<br>treatment_pembrolizumab has no positive PFS effect in any major breast cancer subgroup tested (MSI-H, TNBC, ER+, HER2+, brain mets, stage IV). |
| 002 | claude-opus-4-7 | manual-claude-code@2026-04-27 | 0.141 | 25 | — | Mean pfs_months differs between patients with her2_low=1 and patients with her2_low=0.<br>Among er_positive=1 patients, mean pfs_months is higher when both treatment_palbociclib=1 AND treatment_tamoxifen=1 than in patients receiving neither.<br>Sacituzumab govitecan extends pfs_months more in her2_low=1 patients than in her2_low=0 patients (positive interaction). |
| 003 | claude-opus-4-7 | manual-claude-code@2026-04-27 | 0.141 | 25 | — | Mean pfs_months differs between patients with her2_low=1 and patients with her2_low=0.<br>Among er_positive=1 patients, mean pfs_months is higher when both treatment_palbociclib=1 AND treatment_tamoxifen=1 than in patients receiving neither.<br>Sacituzumab govitecan extends pfs_months more in her2_low=1 patients than in her2_low=0 patients (positive interaction). |
| 004 | claude-opus-4-7 | claude-code@interactive-2026-04-27 | 0.207 | 25 | — | Patients with her2_low == 1 have different mean pfs_months than HER2-zero patients.<br>The PFS benefit of treatment_palbociclib is greater in postmenopausal == 1 than postmenopausal == 0 patients.<br>The PFS benefit of treatment_sacituzumab_govitecan is greater in her2_low == 1 patients (positive treatment×HER2-low interaction). |
| 005 | claude-opus-4-7 | claude-code@manual-1.0 | 0.125 | 25 | — | There is a positive interaction between her2_low and treatment_sacituzumab_govitecan on pfs_months: sacituzumab govitecan prolongs pfs_months more in HER2-low patients than in HER2-not-low patients.<br>Mean pfs_months differs between race_ethnicity = 'black' and race_ethnicity = 'white' patients (testing socio-demographic disparity).<br>Mean pfs_months is shorter for uninsured patients (insurance_type = 'uninsured') than for privately insured patients (insurance_type = 'private'). |
| 006 | claude-opus-4-7 | claude-code@manual-analysis | 0.306 | 25 | — | treatment_palbociclib provides a larger PFS benefit in postmenopausal=1 patients than in postmenopausal=0 patients.<br>Mean pfs_months differs across race_ethnicity strata (white, black, hispanic, asian, other).<br>Mean pfs_months differs across insurance_type strata (private, medicare, medicaid, uninsured); uninsured patients have shorter PFS than privately insured. |
| 007 | claude-opus-4-7 | claude-code@opus-4.7-1m | 0.200 | 25 | — | Higher age_years is associated with higher pfs_months (positive linear association in the overall cohort).<br>Within ER-positive patients, treatment_tamoxifen prolongs pfs_months more in postmenopausal patients than in pre-menopausal patients (positive interaction).<br>Within HR-positive/HER2-negative patients, treatment_palbociclib provides additional pfs_months benefit in PIK3CA-mutated (pik3ca_mutation=1) patients on top of any baseline benefit (positive treatment_palbociclib × pik3ca_mutation interaction). |
| 008 | claude-opus-4-7 | claude-code@manual-iterative-v1 | 0.192 | 25 | — | Patients with her2_low tumors have different mean pfs_months than her2-non-low patients.<br>Patients with pik3ca_mutation have different mean pfs_months than wild-type.<br>The PFS benefit of treatment_sacituzumab_govitecan is larger in her2_low patients than in her2-non-low patients (positive interaction term). |
| 009 | claude-opus-4-7 | claude-code@manual-2026-04-27 | 0.412 | 25 | — | HER2-low tumors (her2_low=1) are associated with longer pfs_months than HER2-zero.<br>There is a positive interaction between TNBC status and treatment_pembrolizumab on pfs_months: TNBC patients benefit more from pembrolizumab than non-TNBC.<br>Higher serum albumin (albumin_g_dl) is associated with longer pfs_months. |
| 010 | claude-opus-4-7 | claude-code@manual-iter-v1 | 0.373 | 25 | — | There is a positive treatment_sacituzumab_govitecan-by-her2_low interaction on pfs_months: the benefit of treatment_sacituzumab_govitecan is greater when her2_low=1 than when her2_low=0.<br>Mean pfs_months differs between sex_female=1 and sex_female=0 patients (positive coef = female longer PFS).<br>Mean pfs_months is lower in rural_residence=1 than rural_residence=0. |
| 011 | claude-opus-4-7 | claude-code@manual-iter-v1 | 0.283 | 25 | — | HER2-low patients have different mean pfs_months than non-HER2-low patients.<br>The PFS benefit of treatment_sacituzumab_govitecan is larger in HER2-low patients than in non-HER2-low patients (positive interaction effect).<br>Mean pfs_months differs across race_ethnicity groups (white, black, hispanic, asian, other). |
| 012 | claude-opus-4-7 | claude-code-manual@2026-04 | 0.326 | 25 | — | treatment_sacituzumab_govitecan improves pfs_months specifically in her2_low patients (TROP2-targeting ADC active in HER2-low disease).<br>her2_positive patients have shorter pfs_months than HER2-negative patients overall (more aggressive biology in this cohort, where targeted therapy benefit is not realized).<br>her2_low patients have longer pfs_months than HER2-not-low patients. |
| 013 | claude-opus-4-7 | claude-code@manual-analysis | 0.347 | 25 | 25 | Patients with her2_low == 1 have longer mean pfs_months than patients with her2_low == 0.<br>The treatment_palbociclib effect on pfs_months is attenuated in her2_positive patients (negative treatment-by-HER2 interaction).<br>Within her2_negative patients, treatment_palbociclib increases pfs_months but within her2_positive patients it does not. |
| 014 | claude-opus-4-7 | claude-code-interactive@manual | 0.165 | 25 | — | HER2-low (her2_low=1) patients have different mean pfs_months than HER2-not-low patients.<br>Sacituzumab govitecan benefit on pfs_months differs by her2_low status (interaction treatment_sacituzumab_govitecan × her2_low).<br>Mean pfs_months differs across race_ethnicity categories (omnibus ANOVA). |
| 015 | claude-opus-4-7 | claude-code@manual-2026-04-28 | 0.215 | 10 | 10 | Sacituzumab govitecan provides greater PFS benefit in HER2-low patients than HER2-not-low (positive treatment_sacituzumab_govitecan × her2_low interaction).<br>Tamoxifen provides greater PFS benefit in postmenopausal patients than premenopausal patients (positive treatment_tamoxifen × postmenopausal interaction).<br>Pembrolizumab benefit differs by tp53 mutation status (treatment_pembrolizumab × tp53_mutation interaction). |
| 016 | claude-opus-4-7 | claude-code@iterative-named | 0.367 | 6 | 6 | Mean pfs_months differs between her2_low=1 and her2_low=0 patients.<br>Higher albumin_g_dl is associated with shorter pfs_months (linear association).<br>treatment_sacituzumab_govitecan has a more favourable effect on pfs_months among her2_low=1 patients (positive interaction term). |
| 017 | claude-opus-4-7 | claude-code-custom@ds001-breast-v1 | 0.315 | 25 | — | In her2_low=1 patients, treatment_sacituzumab_govitecan is associated with higher mean pfs_months than no sacituzumab.<br>In her2_low=0 patients, treatment_sacituzumab_govitecan is not associated with higher mean pfs_months.<br>There is a positive treatment_sacituzumab_govitecan × her2_low interaction in an OLS of pfs_months. |
| 018 | claude-opus-4-7 | claude-code-custom@1.0 | 0.110 | 25 | — | The pfs_months benefit of treatment_sacituzumab_govitecan differs (likely greater) in her2_low=1 patients vs her2_low=0.<br>After adjusting for the same covariates, treatment_pembrolizumab does NOT have a significant positive main-effect coefficient on pfs_months (effect concentrated in MSI-high subgroup).<br>Greater years_since_diagnosis is associated with different pfs_months. |
| 019 | claude-opus-4-7 | claude-code@opus-4-7-1m-named | 0.247 | 25 | — | There is a positive interaction between treatment_sacituzumab_govitecan and her2_low on pfs_months: sacituzumab govitecan improves PFS in HER2-low patients.<br>Higher fatigue_grade (CTCAE-style 0-4) is associated with shorter pfs_months.<br>Higher pain_nrs (numeric rating scale) is associated with shorter pfs_months. |
| 020 | claude-opus-4-7 | claude-code@manual-iterative-2026-04 | 0.270 | 25 | — | her2_low=1 is associated with longer pfs_months than her2_low=0.<br>msi_high=1 is associated with different pfs_months than msi_high=0.<br>There is an interaction between treatment_sacituzumab_govitecan and her2_low: sacituzumab benefit is concentrated in HER2-low patients. |

#### Anonymized (n_replicates=20)
- buried_score: 21.85 ± 6.23
- replicates uncovered: 7/20

| replicate | model | harness | buried_score | uncovered@ |
|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@manual-1.0 | 25 | — |
| 002 | claude-opus-4-7 | claude-code-manual@1.0 | 25 | — |
| 003 | claude-opus-4-7 | claude-code@interactive | 25 | 25 |
| 004 | claude-opus-4-7 | claude-code@manual-bash | 25 | — |
| 005 | claude-opus-4-7 | claude-code-manual@1.0.0 | 25 | — |
| 006 | claude-opus-4-7 | claude-code@interactive-1.0 | 11 | 11 |
| 007 | claude-opus-4-7 | claude-code@manual-analysis | 24 | 24 |
| 008 | claude-opus-4-7 | claude-code@manual-2026-04-27 | 24 | 24 |
| 009 | claude-opus-4-7 | claude-code@local | 25 | — |
| 010 | claude-opus-4-7 | claude-code-manual@1.0.0 | 25 | — |
| 011 | claude-opus-4-7 | claude-code@manual | 25 | — |
| 012 | claude-opus-4-7 | claude-code-interactive@manual | 25 | — |
| 013 | claude-opus-4-7 | claude-code-manual@1.0 | 10 | 10 |
| 014 | claude-opus-4-7 | claude-code@interactive | 10 | 10 |
| 015 | claude-opus-4-7 | claude-code@manual-1.0 | 25 | — |
| 016 | claude-opus-4-7 | claude-code-interactive@2026-04-28 | 8 | 8 |
| 017 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 25 | — |
| 018 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 25 | — |
| 019 | claude-opus-4-7 | manual-iterative-analysis@v1 | 25 | — |
| 020 | claude-opus-4-7 | claude-code@manual-2026-04 | 25 | — |

### ds001_crc

#### Named (n_replicates=20)
- frac_novel: 0.315 ± 0.102
- buried_score: 24.60 ± 1.57
- replicates uncovered: 3/20

| replicate | model | harness | frac_novel | buried_score | uncovered@ | sample novel hypotheses |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@named-task-runner | 0.391 | 25 | — | Patients with diabetes_mellitus=1 have different pfs_months than those without.<br>Patients with hypertension=1 have different pfs_months than those without.<br>Patients with copd=1 have shorter pfs_months than those without. |
| 002 | claude-opus-4-7 | claude-code@manual-analysis | 0.493 | 18 | 18 | Race/ethnicity categories differ in pfs_months.<br>Insurance type categories differ in pfs_months.<br>rural_residence=1 patients have different pfs_months than rural_residence=0. |
| 003 | claude-opus-4-7 | claude-code-named-direct@1.0.0 | 0.203 | 25 | — | Female sex (sex_female=1) is associated with longer pfs_months than male sex.<br>NRAS mutation (nras_mutation=1) is associated with shorter pfs_months than wild-type.<br>Bevacizumab benefit is similar between right-sided and left-sided primaries (no significant bevacizumab × right-sided interaction). |
| 004 | claude-opus-4-7 | claude-code-direct@2026-04-28 | 0.239 | 25 | — | Patients with ntrk_fusion=1 have shorter mean pfs_months than those with ntrk_fusion=0.<br>Female patients (sex_female=1) have different mean pfs_months than male patients.<br>Patients with autoimmune_disease=1 have shorter mean pfs_months than those with autoimmune_disease=0. |
| 005 | claude-opus-4-7 | manual-iter-analysis@1.0 | 0.411 | 25 | — | There is a negative treatment_cetuximab x nras_mutation interaction on pfs_months.<br>In KRAS-mutated patients, treatment_bevacizumab is associated with longer pfs_months than no bevacizumab.<br>Among patients with prior_lines_of_therapy >= 2, treatment_regorafenib is associated with longer pfs_months than no regorafenib. |
| 006 | claude-opus-4-7 | claude-code-custom@manual | 0.250 | 25 | — | A composite symptom-burden score (sum of pain_nrs, fatigue_grade, dyspnea_grade, cough_grade, appetite_loss_grade) is negatively associated with pfs_months.<br>Mean pfs_months differs by rural_residence (rural vs urban).<br>Higher education_years is associated with longer pfs_months (a socioeconomic-status proxy). |
| 007 | claude-opus-4-7 | claude-code@analysis-script | 0.307 | 25 | — | diabetes_mellitus=1 patients have different pfs_months than diabetes_mellitus=0 (negative effect expected).<br>hypertension=1 patients have different pfs_months than hypertension=0 (negative effect expected).<br>copd=1 patients have different pfs_months than copd=0 (negative effect expected). |
| 008 | claude-opus-4-7 | claude-code@in-process-analysis | 0.483 | 25 | — | Older age_years is associated with longer pfs_months (positive correlation).<br>Mean pfs_months differs across insurance_type categories.<br>Patients with rural_residence=1 have different pfs_months than rural_residence=0. |
| 009 | claude-opus-4-7 | claude-code@manual-pipeline-1 | 0.315 | 25 | — | Right-sided primary tumors (right_sided_primary=1) have shorter pfs_months than left-sided primaries.<br>Mean pfs_months differs between female (sex_female=1) and male patients.<br>NRAS-mutant (nras_mutation=1) tumors have different pfs_months versus NRAS wild-type. |
| 010 | claude-opus-4-7 | claude-code@manual-iterative | 0.243 | 25 | — | rural_residence == 1 is associated with shorter pfs_months than urban (rural_residence == 0).<br>Mean pfs_months differs across race_ethnicity categories.<br>Mean pfs_months differs across insurance_type categories. |
| 011 | claude-opus-4-7 | claude-code@manual-iterative-2026-04-28 | 0.141 | 25 | — | Within `braf_v600e` = 1, the combined-effect estimate for receiving both encorafenib and cetuximab exceeds the sum of their separate effects on `pfs_months` (super-additive interaction).<br>Patients with `rural_residence` = 1 have shorter `pfs_months` than those with `rural_residence` = 0 (access disparity).<br>Higher `education_years` is associated with longer `pfs_months` (positive slope). |
| 012 | claude-opus-4-7 | claude-code-manual@1.0 | 0.205 | 25 | — | There is an interaction between treatment_bevacizumab and venous_thromboembolism_history on pfs_months.<br>Patients with heart_failure=1 have shorter pfs_months than patients without heart failure.<br>Patients with chronic_kidney_disease=1 have shorter pfs_months than patients without CKD. |
| 013 | claude-opus-4-7 | claude-code-named@2026-04-28 | 0.406 | 25 | — | diabetes_mellitus=1 is associated with shorter pfs_months than diabetes_mellitus=0.<br>chronic_kidney_disease=1 is associated with shorter pfs_months than chronic_kidney_disease=0.<br>heart_failure=1 is associated with shorter pfs_months than heart_failure=0. |
| 014 | claude-opus-4-7 | claude-code-manual@1.0 | 0.395 | 24 | 24 | braf_v600e = 1 is associated with shorter pfs_months than braf_v600e = 0.<br>nras_mutation = 1 is associated with shorter pfs_months than nras_mutation = 0.<br>In a multivariable OLS model with age, sex, ECOG, stage, sidedness, key biomarkers, CEA, albumin, LDH, weight loss, CRP, NLR, and all six treatments, the disease-severity terms (ecog_ps, stage_iv, albumin, weight loss) remain the dominant predictors of pfs_months. |
| 015 | claude-opus-4-7 | claude-code-custom@manual-1.0 | 0.246 | 25 | — | Higher fatigue_grade is associated with shorter pfs_months (adjusted for other symptoms).<br>Higher pain_nrs is associated with shorter pfs_months (adjusted for other symptoms).<br>treatment_regorafenib has a non-zero (likely small) main effect on pfs_months and the effect may differ by prior_lines_of_therapy. |
| 016 | claude-opus-4-7 | manual-claude-code-run@2026-04-28 | 0.225 | 25 | — | Female sex (sex_female=1) is associated with longer pfs_months than male (positive mean difference).<br>Mean pfs_months differs across race_ethnicity categories (overall ANOVA test).<br>Mean pfs_months differs across insurance_type categories (overall ANOVA test). |
| 017 | claude-opus-4-7 | claude-code@manual-harness | 0.390 | 25 | — | nras_mutation=1 is associated with shorter pfs_months than nras_mutation=0 (analogous to KRAS).<br>ntrk_fusion=1 is associated with longer pfs_months than ntrk_fusion=0 (NTRK fusions are usually targetable and rare).<br>The treatment_X effect on pfs_months differs between stage_iv=1 and stage_iv=0 patients for at least one drug (cetuximab, bevacizumab, pembrolizumab, regorafenib): treatment x stage_iv interaction term will be significant. |
| 018 | claude-opus-4-7 | claude-code@manual | 0.429 | 25 | 25 | nras_mutation=1 is associated with longer pfs_months (positive effect, contrary to typical clinical expectation).<br>Older age_years is associated with longer pfs_months (positive slope), opposite to the usual aging-as-poor-prognosis expectation.<br>treatment_regorafenib has a positive main effect on pfs_months (longer PFS in regorafenib-treated patients) after adjustment for prognostic covariates. |
| 019 | claude-opus-4-7 | claude-code@local-ds001-crc | 0.241 | 25 | — | Adrenal mets, contralateral lung mets, and pericardial effusion each predict shorter pfs_months.<br>Higher calcium and platelets each have measurable associations with pfs_months.<br>Bevacizumab effect on pfs_months differs by tumor side (treatment × right_sided_primary interaction). |
| 020 | claude-opus-4-7 | claude-code@manual-iterative-2026-04 | 0.288 | 25 | — | treatment_regorafenib effect on pfs_months differs by kras_mutation status.<br>Within kras_mutation==0, treatment_regorafenib increases mean pfs_months substantially; within kras_mutation==1, treatment_regorafenib has little or no effect on pfs_months.<br>treatment_regorafenib effect on pfs_months is also modified by nras_mutation status (benefit largest in NRAS WT). |

#### Anonymized (n_replicates=20)
- buried_score: 25.00 ± 0.00
- replicates uncovered: 0/20

| replicate | model | harness | buried_score | uncovered@ |
|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@opus-4-7-1m-anonymized-v1 | 25 | — |
| 002 | claude-opus-4-7 | claude-code-cli@anonymized-bundle-run | 25 | — |
| 003 | claude-opus-4-7 | claude-code@interactive-session | 25 | — |
| 004 | claude-opus-4-7 | claude-code-manual@1.0 | 25 | — |
| 005 | claude-opus-4-7 | claude-code-manual@1.0 | 25 | — |
| 006 | claude-opus-4-7 | claude-code-interactive@manual-2026-04-28 | 25 | — |
| 007 | claude-opus-4-7 | claude-code-manual@1.0 | 25 | — |
| 008 | claude-opus-4-7 | claude-code@interactive-analysis-2026-04-28 | 25 | — |
| 009 | claude-opus-4-7 | manual-claude-code-session@2026-04-28 | 25 | — |
| 010 | claude-opus-4-7 | claude-code-manual@1.0 | 25 | — |
| 011 | claude-opus-4-7 | claude-code@manual-iter25 | 25 | — |
| 012 | claude-opus-4-7 | claude-code@manual-2026-04-28 | 25 | — |
| 013 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 25 | — |
| 014 | claude-opus-4-7 | claude-code-custom@v1 | 25 | — |
| 015 | claude-opus-4-7 | claude-code@manual-iter-2026-04-28 | 25 | — |
| 016 | claude-opus-4-7 | claude-code@user-driven-2026-04 | 25 | — |
| 017 | claude-opus-4-7 | claude-code@interactive-2026-04-28 | 25 | — |
| 018 | claude-opus-4-7 | claude-code@manual-script-2026-04-28 | 25 | — |
| 019 | claude-opus-4-7 | claude-code@interactive-1 | 25 | — |
| 020 | claude-opus-4-7 | claude-code@manual-2026-04-28 | 25 | — |

### ds001_nsclc

#### Named (n_replicates=20)
- frac_novel: 0.317 ± 0.104
- buried_score: 24.20 ± 2.48
- replicates uncovered: 3/20

| replicate | model | harness | frac_novel | buried_score | uncovered@ | sample novel hypotheses |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@nsclc-named-1 | 0.410 | 18 | 18 | There is a negative interaction: stk11_mutation=1 attenuates the objective_response benefit of treatment_pembrolizumab.<br>There is a negative interaction: keap1_mutation=1 attenuates the objective_response benefit of treatment_pembrolizumab.<br>Female patients (sex_female=1) have a higher objective_response rate than male patients. |
| 002 | claude-opus-4-7 | claude-code@named-bundle-runner | 0.283 | 25 | — | The effect of treatment_pembrolizumab on objective_response is less positive (or negative) in patients with stk11_mutation=1 (negative interaction between treatment_pembrolizumab and stk11_mutation).<br>Among patients receiving treatment_pembrolizumab, those with keap1_mutation=1 have a lower objective_response rate than those with keap1_mutation=0.<br>The effect of treatment_pembrolizumab on objective_response is less positive in patients with keap1_mutation=1 (negative interaction). |
| 003 | claude-opus-4-7 | claude-code@manual-script | 0.274 | 25 | — | adrenal_mets==1 is associated with a LOWER objective_response rate.<br>pleural_effusion==1 is associated with a LOWER objective_response rate.<br>Among patients with egfr_mutation==1, sex_female==1 patients have a HIGHER objective_response rate. |
| 004 | claude-opus-4-7 | are-llms-biased@v1 | 0.436 | 25 | — | Building on the STK11 result (h9.1), the benefit of treatment_pembrolizumab on objective_response is also reduced in keap1_mutation=1 patients (negative pembrolizumab×keap1_mutation interaction).<br>Black patients (race_ethnicity=='black') have a different objective_response rate than non-Black patients.<br>Hispanic patients (race_ethnicity=='hispanic') have a different objective_response rate than non-Hispanic patients. |
| 005 | claude-opus-4-7 | claude-code-manual@1.0 | 0.491 | 25 | — | stk11_mutation=1 patients gain less (or zero) objective_response from treatment_pembrolizumab compared to stk11_mutation=0 patients (negative interaction).<br>objective_response rate differs across race_ethnicity categories.<br>objective_response rate differs across insurance_type categories. |
| 006 | claude-opus-4-7 | claude-code-opus47-1m@ds001-nsclc-named | 0.210 | 25 | — | There is an interaction between squamous histology and `treatment_pembrolizumab` for `objective_response`.<br>There is an interaction between `age_years` and `treatment_pembrolizumab` for `objective_response`.<br>There is an interaction between `sex_female` and `treatment_pembrolizumab` for `objective_response`. |
| 007 | claude-opus-4-7 | claude-code@nsclc-named-1 | 0.475 | 25 | — | keap1_mutation=1 attenuates the pembrolizumab benefit on objective_response, similar to STK11.<br>Pembrolizumab benefit on objective_response is larger in sex_female=1 than sex_female=0.<br>Pembrolizumab benefit declines as ecog_ps increases; in ecog_ps>=2 there is no benefit. |
| 008 | claude-opus-4-7 | claude-code-custom@2026-04-28 | 0.213 | 25 | — | In stk11_mutation=1 patients, pembrolizumab benefit on objective_response is reduced (negative interaction) vs stk11_mutation=0.<br>In keap1_mutation=1 patients, pembrolizumab benefit on objective_response is reduced (negative interaction) vs keap1_mutation=0.<br>Among pembrolizumab-treated patients, ever-smokers have higher objective_response than never-smokers. |
| 009 | claude-opus-4-7 | claude-code@manual-2026-04-28 | 0.098 | 16 | 16 | keap1_mutation modifies the effect of treatment_pembrolizumab on objective_response: pembrolizumab response is lower in keap1_mutation=1 patients (negative interaction).<br>sex_female modifies the effect of treatment_pembrolizumab on objective_response.<br>Among patients with pdl1_tps>=0.5 AND tmb_high=1, treatment_pembrolizumab is associated with markedly higher objective_response than no pembrolizumab (combined biomarker enrichment). |
| 010 | claude-opus-4-7 | claude-code-direct@1 | 0.231 | 25 | — | The objective_response benefit of treatment_pembrolizumab is reduced in keap1_mutation=1 patients; i.e., the interaction keap1_mutation x treatment_pembrolizumab has a negative coefficient.<br>Among patients with histology='squamous', treatment_pembrolizumab effect on objective_response differs from the effect among adenocarcinoma patients.<br>nrg1_fusion=1 is associated with a different objective_response rate than nrg1_fusion=0 (targetable oncogenic driver may correlate with response patterns). |
| 011 | claude-opus-4-7 | claude-code-do-analysis@1 | 0.420 | 25 | — | Female sex (sex_female=1) is associated with higher objective_response than male.<br>The response benefit of treatment_pembrolizumab over no pembrolizumab is different (interaction) between squamous and adenocarcinoma histology.<br>adrenal_mets=1 is associated with lower objective_response than adrenal_mets=0. |
| 012 | claude-opus-4-7 | claude-code-manual@opus-4-7-2026-04-28 | 0.404 | 25 | — | stk11_mutation-positive patients derive less benefit from treatment_pembrolizumab on objective_response than stk11_mutation-negative patients (negative STK11 x pembrolizumab interaction; immunotherapy resistance).<br>keap1_mutation-positive patients derive less benefit from treatment_pembrolizumab on objective_response than keap1_mutation-negative patients (negative KEAP1 x pembrolizumab interaction).<br>Female patients (sex_female=1) have a different objective_response rate than male patients. |
| 013 | claude-opus-4-7 | claude-code-named-bundle@1 | 0.357 | 25 | — | Female patients (sex_female=1) have a different probability of objective_response than male patients.<br>Patients with stk11_mutation = 1 derive less benefit (smaller increase in objective_response) from treatment_pembrolizumab than stk11_mutation = 0 patients (negative interaction).<br>Patients with keap1_mutation = 1 derive less benefit from treatment_pembrolizumab than keap1_mutation = 0 patients (negative interaction). |
| 014 | claude-opus-4-7 | claude-code-self-harness@1.0 | 0.231 | 25 | — | Never-smokers (smoking_status == 'never') derive less benefit from treatment_pembrolizumab on objective_response than ever-smokers (negative interaction between pembrolizumab and never-smoker status).<br>There is an interaction between treatment_pembrolizumab and histology=='squamous' on objective_response.<br>The objective_response rate differs across race_ethnicity categories. |
| 015 | claude-opus-4-7 | claude-code@interactive | 0.291 | 25 | 25 | Patients with STK11 mutation (stk11_mutation=1) have a different overall objective response rate than those without it.<br>STK11 mutations attenuate the response benefit of pembrolizumab (negative treatment_pembrolizumab x stk11_mutation interaction).<br>Pembrolizumab benefit on response differs between squamous and adenocarcinoma histology (treatment_pembrolizumab x squamous interaction non-zero). |
| 016 | claude-opus-4-7 | claude-code-iterative@v1 | 0.298 | 25 | — | Among treatment_pembrolizumab recipients, keap1_mutation is associated with lower objective_response.<br>There is a negative interaction between treatment_pembrolizumab and keap1_mutation on objective_response.<br>Patients with adrenal_mets=1 have lower objective_response than those without adrenal metastases. |
| 017 | claude-opus-4-7 | claude-code-opus-4-7-1m@manual | 0.348 | 25 | — | There is a positive multiplicative interaction between treatment_pembrolizumab and pdl1_tps on objective_response (the slope of pdl1_tps on log-odds of response is steeper among pembrolizumab-treated patients).<br>There is a positive multiplicative interaction between treatment_sotorasib and kras_g12c on objective_response.<br>Higher albumin_g_dl is associated with a higher objective_response rate. |
| 018 | claude-opus-4-7 | claude-code@anthropic-cli | 0.206 | 25 | — | Among patients on treatment_pembrolizumab, stk11_mutation=1 reduces the magnitude of treatment-associated improvement in objective_response (negative interaction).<br>Patients with keap1_mutation=1 derive less objective_response benefit from treatment_pembrolizumab than keap1_mutation=0 patients (negative interaction).<br>The effect of treatment_pembrolizumab on objective_response differs by sex_female (interaction). |
| 019 | claude-opus-4-7 | claude-code-manual@1.0 | 0.279 | 25 | — | tmb_high amplifies the positive effect of treatment_pembrolizumab on ORR.<br>Medicaid/uninsured insurance is associated with different ORR than medicare/private.<br>rural_residence is associated with different ORR. |
| 020 | claude-opus-4-7 | claude-code@native-direct | 0.382 | 25 | — | stk11_mutation diminishes the response benefit of treatment_pembrolizumab (negative interaction term).<br>egfr_mutation diminishes the response benefit of treatment_pembrolizumab (negative interaction term).<br>Within smoking_status strata (current/former/never), the response benefit of treatment_pembrolizumab differs (smokers benefit more than never-smokers). |

#### Anonymized (n_replicates=20)
- buried_score: 23.35 ± 5.20
- replicates uncovered: 4/20

| replicate | model | harness | buried_score | uncovered@ |
|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@interactive | 25 | 25 |
| 002 | claude-opus-4-7 | claude-code@nsclc-anon-1 | 25 | — |
| 003 | claude-opus-4-7 | claude-code@manual-2026-04-28 | 25 | — |
| 004 | claude-opus-4-7 | claude-code-direct@2026-04-28 | 25 | — |
| 005 | claude-opus-4-7 | claude-code@manual-2026-04-28 | 25 | — |
| 006 | claude-opus-4-7 | claude-code@opus-4-7-1m | 25 | — |
| 007 | claude-opus-4-7 | claude-code-manual@nsclc-anon-2026-04-28 | 25 | — |
| 008 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 25 | — |
| 009 | claude-opus-4-7 | claude-code@1.0 | 25 | — |
| 010 | claude-opus-4-7 | claude-code@nsclc-anon-2026-04 | 25 | — |
| 011 | claude-opus-4-7 | claude-code@manual-iterative | 5 | 5 |
| 012 | claude-opus-4-7 | claude-code@manual-iteration | 25 | — |
| 013 | claude-opus-4-7 | claude-code@dataset-analysis-2026-04-28 | 25 | — |
| 014 | claude-opus-4-7 | claude-code@manual-run | 25 | — |
| 015 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 25 | — |
| 016 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 12 | 12 |
| 017 | claude-opus-4-7 | claude-code@manual-run-2026-04-28 | 25 | — |
| 018 | claude-opus-4-7 | claude-code@manual-agent-2026-04-28 | 25 | — |
| 019 | claude-opus-4-7 | claude-code@manual-iter | 25 | — |
| 020 | claude-opus-4-7 | claude-code-opus-4-7-1m@adhoc | 25 | 25 |

### ds001_prostate

#### Named (n_replicates=20)
- frac_novel: 0.357 ± 0.079
- buried_score: 25.00 ± 0.00
- replicates uncovered: 0/20

| replicate | model | harness | frac_novel | buried_score | uncovered@ | sample novel hypotheses |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@1.0 | 0.223 | 25 | — | Patients with insurance_type='medicaid' have lower mean pfs_months than patients with insurance_type='private'.<br>Patients with rural_residence=1 have lower mean pfs_months than urban patients.<br>Patients with race_ethnicity='black' have different mean pfs_months than 'white' patients. |
| 002 | claude-opus-4-7 | claude-code@custom-prostate-analysis | 0.425 | 25 | — | Patients with psma_high=1 have different mean pfs_months than psma_high=0.<br>The olaparib × brca2_mutation interaction on pfs_months remains positive and significant after adjusting for age_years, ecog_ps, mcrpc, albumin_g_dl, ldh_u_l, psa_ng_ml, weight_loss_pct_6mo.<br>The olaparib × brca2_mutation interaction on pfs_months is observed within the mcrpc=0 subgroup as well. |
| 003 | claude-opus-4-7 | claude-code-manual@1.0 | 0.453 | 25 | — | treatment_pembrolizumab effect on pfs_months differs by tp53_mutation status.<br>Older age_years is associated with longer pfs_months in this cohort (positive slope) — opposite of clinical expectation.<br>Mean pfs_months differs across insurance_type categories. |
| 004 | claude-opus-4-7 | claude-code-manual@1.0 | 0.389 | 25 | — | Patients with chronic_kidney_disease=1 have lower mean pfs_months than patients with chronic_kidney_disease=0.<br>Patients with heart_failure=1 have lower mean pfs_months than patients with heart_failure=0.<br>Patients with coronary_artery_disease=1 have lower mean pfs_months than patients with coronary_artery_disease=0. |
| 005 | claude-opus-4-7 | claude-code@manual-2026-04-28 | 0.407 | 25 | — | Patients with psma_high=1 have higher mean pfs_months than patients with psma_high=0.<br>Docetaxel benefit on pfs_months differs by visceral metastases status (interaction).<br>Within brca2_mutation=0 patients, mean pfs_months differs between treatment_olaparib=1 and treatment_olaparib=0 (direction expected to favor treatment_olaparib when biomarker matches predicted indication). |
| 006 | claude-opus-4-7 | manual-claude-code@1.0 | 0.421 | 25 | — | Mean pfs_months differs across race_ethnicity categories.<br>Mean pfs_months differs across insurance_type categories.<br>Patients with rural_residence=1 have different mean pfs_months than those with rural_residence=0. |
| 007 | claude-opus-4-7 | claude-code-custom@1.0 | 0.427 | 25 | — | treatment_docetaxel is more effective (greater increase in pfs_months) in patients with visceral_mets than in patients without visceral_mets.<br>tp53_mutation modifies the effect of treatment_olaparib on pfs_months.<br>tp53_mutation modifies the effect of treatment_pembrolizumab on pfs_months. |
| 008 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 0.283 | 25 | — | There is a negative interaction between treatment_docetaxel and tp53_mutation (p53-deficient tumors respond less well to docetaxel).<br>race_ethnicity is associated with pfs_months: black, hispanic, asian, or other patients differ from white patients.<br>insurance_type is associated with pfs_months: medicare, medicaid, or uninsured differ from privately insured patients. |
| 009 | claude-opus-4-7 | claude-code-named@1.0 | 0.230 | 25 | — | Older age (age_years) is associated with longer pfs_months (positive slope), even after adjusting for ECOG, PSA, mCRPC, albumin and visceral mets — likely reflecting selection of fitter, less aggressive disease in older patients.<br>Mean pfs_months differs across race_ethnicity categories (omnibus F-test).<br>Mean pfs_months differs across insurance_type categories (omnibus F-test). |
| 010 | claude-opus-4-7 | claude-code@named-prostate-1 | 0.381 | 25 | — | In a univariate OLS, older age (age_years) is associated with longer pfs_months.<br>The PFS effect of treatment_enzalutamide differs between mCRPC (mcrpc=1) and hormone-sensitive (mcrpc=0) patients (mcrpc × treatment_enzalutamide interaction).<br>The PFS effect of treatment_olaparib differs between mCRPC (mcrpc=1) and hormone-sensitive (mcrpc=0) patients (mcrpc × treatment_olaparib interaction). |
| 011 | claude-opus-4-7 | claude-code@manual-iter | 0.443 | 25 | — | Receiving treatment_pembrolizumab=1 is associated with longer pfs_months than not receiving it (positive coefficient when all treatments are mutually adjusted).<br>After adjusting for ECOG, mCRPC, albumin, log PSA, log LDH, hemoglobin, and visceral mets, age_years has a positive (suppressed) coefficient on pfs_months, indicating its raw negative effect is mediated by these prognostic factors.<br>Higher fatigue_grade is associated with shorter pfs_months after adjusting for ECOG. |
| 012 | claude-opus-4-7 | claude-code@iterative-eda-2026.04 | 0.400 | 25 | — | Patients with psma_high=1 have different mean pfs_months than psma_high=0.<br>Higher fatigue_grade is associated with shorter pfs_months.<br>Higher pain_nrs is associated with shorter pfs_months. |
| 013 | claude-opus-4-7 | claude-code@manual-iteration-v1 | 0.235 | 25 | — | Log-transformed PSA shows a stronger linear association with pfs_months than raw PSA.<br>PSMA-high patients (psma_high=1) have different pfs_months than PSMA-low.<br>There is a different pfs_months effect of treatment_docetaxel between patients with and without visceral_mets (interaction). |
| 014 | claude-opus-4-7 | claude-code-manual@1.0 | 0.267 | 25 | — | There is a positive treatment_docetaxel x visceral_mets interaction on pfs_months: visceral-mets patients gain more PFS from docetaxel than non-visceral patients.<br>Mean pfs_months differs across race_ethnicity categories (white/black/hispanic/asian/other).<br>Mean pfs_months differs across insurance_type categories (medicare/private/medicaid/uninsured). |
| 015 | claude-opus-4-7 | claude-code-manual@1.0 | 0.362 | 25 | — | Mean pfs_months differs across race_ethnicity strata.<br>Mean pfs_months differs across insurance_type strata.<br>rural_residence patients have a different mean pfs_months than non-rural patients. |
| 016 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 0.277 | 25 | — | Mean pfs_months differs across race_ethnicity categories (white, black, hispanic, asian, other).<br>Mean pfs_months differs across insurance_type categories (medicare, private, medicaid, uninsured).<br>Patients with rural_residence=1 have shorter pfs_months than those without. |
| 017 | claude-opus-4-7 | claude-code@manual-iteration | 0.372 | 25 | — | Refined: in this cohort, older age_years is associated with LONGER pfs_months, with each additional year adding roughly +0.17 months of PFS in a univariate linear model.<br>After adjusting for age, ECOG, albumin, log(PSA), weight loss, LDH, and visceral mets in a multivariable OLS, mCRPC will lose its univariate negative association with pfs_months (i.e., mCRPC's apparent prognostic effect is confounded by other prognostic factors).<br>The olaparib × BRCA2 interaction effect on pfs_months remains large and statistically significant after adjusting for age, ECOG, albumin, weight loss, and log(PSA). |
| 018 | claude-opus-4-7 | claude-code-agent@manual-iter-2026-04-28 | 0.468 | 25 | — | The PFS effect of treatment_abiraterone is more negative in ar_v7_positive=1 patients than in ar_v7_positive=0 patients (negative interaction).<br>msi_high=1 patients have different pfs_months than msi_high=0 patients (direction uncertain in unadjusted comparison).<br>psma_high=1 patients have different pfs_months than psma_high=0 patients. |
| 019 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 0.330 | 25 | — | psma_high=1 is associated with longer pfs_months.<br>Black patients (race_ethnicity='black') have shorter mean pfs_months than white patients.<br>Hispanic patients have shorter mean pfs_months than white patients. |
| 020 | claude-opus-4-7 | claude-code@manual-driver | 0.349 | 25 | — | Older age_years is associated with longer pfs_months (Pearson correlation > 0).<br>The BRCA2 x olaparib interaction remains positive and significant after adjustment for age, ECOG, mCRPC, visceral mets, albumin, weight loss, and PSA.<br>There is a negative interaction between ar_v7_positive and treatment_abiraterone on pfs_months. |

#### Anonymized (n_replicates=20)
- buried_score: 23.95 ± 3.27
- replicates uncovered: 2/20

| replicate | model | harness | buried_score | uncovered@ |
|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@opus-4-7-1m | 13 | 13 |
| 002 | claude-opus-4-7 | claude-code@manual-2026-04-28 | 16 | 16 |
| 003 | claude-opus-4-7 | claude-code-interactive@manual-2026-04-28 | 25 | — |
| 004 | claude-opus-4-7 | claude-code@manual-prostate-analysis-v1 | 25 | — |
| 005 | claude-opus-4-7 | claude-code-interactive@manual-2026-04-28 | 25 | — |
| 006 | claude-opus-4-7 | claude-code-interactive@manual-1 | 25 | — |
| 007 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 25 | — |
| 008 | claude-opus-4-7 | claude-code@interactive-1 | 25 | — |
| 009 | claude-opus-4-7 | claude-code@interactive | 25 | — |
| 010 | claude-opus-4-7 | claude-code-manual@1.0 | 25 | — |
| 011 | claude-opus-4-7 | claude-code@manual-iter | 25 | — |
| 012 | claude-opus-4-7 | claude-code-direct@manual-2026-04-28 | 25 | — |
| 013 | claude-opus-4-7 | claude-code@interactive-opus-4-7 | 25 | — |
| 014 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 25 | — |
| 015 | claude-opus-4-7 | claude-code-manual@1.0 | 25 | — |
| 016 | claude-opus-4-7 | claude-code-interactive@manual-1 | 25 | — |
| 017 | claude-opus-4-7 | claude-code@manual-prostate-2026-04-28 | 25 | — |
| 018 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 25 | — |
| 019 | claude-opus-4-7 | claude-code-manual@2026-04-28 | 25 | — |
| 020 | claude-opus-4-7 | custom-python-driver@1.0 | 25 | — |
