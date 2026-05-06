# Oncology Co-Scientist Benchmark — Batch Scoring Report

- **Bundles scored:** 10 (5 named, 5 anonymized)
- **Replicates (total):** 200
- **Novelty %** (named only, unweighted mean of bundle means): 0.383
- **Buried discovery iteration — named** (lower = uncovers earlier; exact only; no exact = max_iterations + 1): 17.670
- **Buried discovery iteration — anonymized** (exact only; no exact = max_iterations + 1): 12.330
- **Fraction of replicates uncovering buried — named:** 0.720
- **Fraction of replicates uncovering buried — anonymized:** 0.920
- **Fraction near-or-better recovery — named:** 0.950
- **Fraction near-or-better recovery — anonymized:** 1.000
- **Fraction component-or-better recovery — named:** 1.000
- **Fraction component-or-better recovery — anonymized:** 1.000

## Per-bundle detail (mean ± SD across replicates)

### ds001_aml

#### Named (n_replicates=20)
- frac_novel: 0.334 ± 0.083
- buried_score: 14.20 ± 4.73
- replicates uncovered: 20/20
- near-or-better recovery: 20/20
- component-or-better recovery: 20/20
- recovery levels: exact=20, near=0, component=0, none=0

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@manual-analysis | 0.302 | 13 | 13 | exact | 13 | After multivariable adjustment, the strongest independent predictors of objective_response include ecog_ps (negative), unfit_for_intensive (positive), npm1_mutation (positive), tp53_mutation (negative), complex_karyotype (negative), blast_pct_marrow (negative), wbc_k_per_ul (negative), albumin_g_dl (positive), weight_loss_pct_6mo (negative), crp_mg_l (negative), and treatment_venetoclax_azacitidine (positive).<br>The ven/aza response benefit is concentrated in patients who are BOTH npm1_mutation=1 AND unfit_for_intensive=1; in either-or-neither subgroups the benefit is small or absent.<br>Among npm1+/unfit+ patients on ven/aza, tp53_mutation=1 still suppresses the response benefit; the ven/aza effect is therefore concentrated in npm1+ AND unfit+ AND tp53- patients. |
| 002 | claude-opus-4-7 | claude-code-manual@1.0 | 0.382 | 7 | 7 | exact | 7 | Across all 6 treatments x 29 candidate effect modifiers, at least one treatment-by-feature interaction surpasses Bonferroni significance for predicting objective_response.<br>Among patients with npm1_mutation=1, treatment_venetoclax_azacitidine markedly increases probability of objective_response relative to npm1_mutation=1 patients not receiving venetoclax_azacitidine (positive interaction treatment_venetoclax_azacitidine x npm1_mutation).<br>In patients with npm1_mutation=0, treatment_venetoclax_azacitidine has essentially no effect on objective_response. |
| 003 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 0.241 | 18 | 18 | exact | 18 | Liver/kidney chemistry, electrolytes (AST, ALT, ALP, bilirubin, creatinine, BUN, sodium, potassium, calcium, hemoglobin) have no meaningful association with objective_response after main-effect screening.<br>Within IDH1-mutated patients, ivosidenib benefit is concentrated in tp53_mutation==0 / complex_karyotype==0 cytogenetically favorable subset.<br>Among unfit_for_intensive==1, npm1_mutation==1 patients, treatment_venetoclax_azacitidine effect on objective_response is further amplified when tp53_mutation==0 (TP53 wild-type required). |
| 004 | claude-opus-4-7 | claude-code@manual-aml-analysis | 0.431 | 8 | 8 | exact | 8 | In multivariable logistic regression with all 35 baseline features and treatments, the strongest independent predictors of objective_response are ECOG (negative), NPM1 (positive), unfit_for_intensive (positive), treatment_venetoclax_azacitidine (positive), weight_loss (negative), blast_pct_marrow (negative), albumin (positive), complex_karyotype (negative), TP53 (negative), CRP (negative), and WBC (negative); other treatments will show no main effect.<br>treatment_venetoclax_azacitidine has a larger positive effect on objective_response in npm1_mutation=1 patients (positive interaction with NPM1).<br>treatment_venetoclax_azacitidine has a smaller (or absent) effect on objective_response in tp53_mutation=1 patients (negative interaction with TP53). |
| 005 | claude-opus-4-7 | claude-code@manual-aml-analysis-v1 | 0.348 | 14 | 14 | exact | 14 | Patients with unfit_for_intensive=1 have a higher objective_response rate than patients with unfit_for_intensive=0 (paradoxical association in this cohort).<br>sex_female is associated with objective_response rate.<br>None of the four targeted agents (midostaurin, gilteritinib, ivosidenib, enasidenib) shows a positive interaction with its canonical target mutation; therefore none of them produces a clinically meaningful improvement in objective_response in this cohort, even within mutation-positive subgroups. |
| 006 | claude-opus-4-7 | claude-code@opus-4-7-aml-named | 0.184 | 10 | 10 | exact | 10 | Female patients (sex_female=1) have a different objective_response rate than male patients.<br>In a multivariable logistic model adjusting for all baseline features, the strongest independent positive predictors of objective_response are unfit_for_intensive, npm1_mutation, treatment_venetoclax_azacitidine, and albumin_g_dl, while ecog_ps, weight_loss_pct_6mo, blast_pct_marrow, complex_karyotype, and tp53_mutation are independent negative predictors.<br>The treatment_venetoclax_azacitidine effect on objective_response is concentrated in patients who are BOTH npm1_mutation=1 AND unfit_for_intensive=1; the effect is small or zero in patients with only one or neither of these features. |
| 007 | claude-opus-4-7 | claude-code-manual@1.0 | 0.309 | 15 | 15 | exact | 15 | Sex_female is associated with objective_response.<br>After multivariable adjustment, treatment_midostaurin/gilteritinib/ivosidenib/enasidenib/7+3 each show no independent main effect on objective_response.<br>Within unfit_for_intensive=1, the ven+aza ORR benefit is concentrated in npm1_mutation=1 patients and is essentially absent in npm1=0 patients. |
| 008 | claude-opus-4-7 | claude-code@named-aml-1 | 0.292 | 14 | 14 | exact | 14 | Female sex (sex_female==1) is associated with a different objective_response rate than male sex.<br>Hemoglobin, LDH, NLR, and other lab markers have no detectable association with objective_response.<br>unfit_for_intensive==1 patients have higher objective_response rates than unfit_for_intensive==0 patients (paradoxical association driven by treatment selection). |
| 009 | claude-opus-4-7 | claude-code-manual@1.0 | 0.262 | 11 | 11 | exact | 11 | After adjusting for all available baseline features and treatments, the multivariable-adjusted association of NPM1 mutation with objective_response remains positive and significant.<br>After multivariable adjustment, the targeted small-molecule treatments (midostaurin, gilteritinib, ivosidenib, enasidenib) and treatment_7plus3 each have an adjusted main effect on objective_response that is not statistically distinguishable from zero.<br>Joint subgroup hypothesis: among patients who are NPM1-mutated AND unfit_for_intensive AND TP53-wild-type, treatment_venetoclax_azacitidine produces a much larger increase in objective_response than in any other subgroup. |
| 010 | claude-opus-4-7[1m] | manual-claude-code-session@2026-05-03 | 0.471 | 12 | 12 | exact | 12 | Among unfit_for_intensive=0 (fit) patients, treatment_venetoclax_azacitidine does NOT increase objective_response.<br>Targeted therapy/mutation match interactions are weak: treatment_midostaurin × flt3_itd, treatment_gilteritinib × flt3_itd, treatment_ivosidenib × idh1_mutation, treatment_enasidenib × idh2_mutation are not significant predictors of objective_response.<br>In the same model, treatment_venetoclax_azacitidine is independently associated with higher objective_response (main effect), while the four targeted therapies (midostaurin, gilteritinib, ivosidenib, enasidenib) and 7+3 are not. |
| 011 | claude-opus-4-7 | claude-code-custom@1.0 | 0.395 | 16 | 16 | exact | 16 | complex_karyotype abrogates the treatment_venetoclax_azacitidine response benefit (negative interaction).<br>The NPM1+ main-effect benefit on objective_response is driven entirely by NPM1+ patients who received VEN/AZA, not an intrinsic NPM1+ response advantage.<br>The unfit_for_intensive main-effect benefit on objective_response is similarly driven entirely by unfit patients who received VEN/AZA, not an intrinsic unfit-patient advantage. |
| 012 | claude-opus-4-7 | manual-claude-code@2026-05-03 | 0.416 | 12 | 12 | exact | 12 | Female patients (sex_female=1) have a different objective_response rate than males.<br>Higher blast_pct_marrow is associated with lower objective_response.<br>Higher weight_loss_pct_6mo is associated with lower objective_response. |
| 013 | claude-opus-4-7 | claude-code-manual@kk-2026-05-03 | 0.272 | 17 | 17 | exact | 17 | objective_response rate differs between sex_female=1 and sex_female=0.<br>creatinine_mg_dl is associated with objective_response such that values associated with worse fitness (different creatinine_mg_dl predicts higher ORR).<br>bun_mg_dl is associated with objective_response such that values associated with worse fitness (different bun_mg_dl predicts higher ORR). |
| 014 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 0.245 | 13 | 13 | exact | 13 | Within unfit_for_intensive=1 AND npm1_mutation=1 patients, treatment_venetoclax_azacitidine produces a much larger increase in objective_response than in any other unfit/npm1 cell.<br>Within unfit_for_intensive=1 AND npm1_mutation=1 patients, the ven/aza response benefit is abolished if tp53_mutation=1.<br>Within unfit_for_intensive=1 AND npm1_mutation=1 patients, the ven/aza response benefit is abolished if complex_karyotype=1. |
| 015 | claude-opus-4-7 | claude-code-manual@opus-4-7-1m | 0.442 | 11 | 11 | exact | 11 | sex_female associates with objective_response.<br>flt3_tkd is associated with a different objective_response rate vs patients without it.<br>Higher wbc_k_per_ul is associated with a different probability of objective_response. |
| 016 | claude-opus-4-7 | manual-iterative-analysis@2026-05-03 | 0.415 | 11 | 11 | exact | 11 | Higher potassium_meq_l (a marker of disease burden / poor nutrition / organ stress) is associated with LOWER objective_response.<br>Higher calcium_mg_dl (a marker of disease burden / poor nutrition / organ stress) is associated with HIGHER objective_response.<br>After adjustment for the same covariates, treatment_midostaurin, treatment_gilteritinib, treatment_ivosidenib, treatment_enasidenib, and treatment_7plus3 each have NO independent main-effect association with objective_response. |
| 017 | claude-opus-4-7 | claude-code-aml-named@1.0 | 0.429 | 18 | 18 | exact | 18 | Patients flagged unfit_for_intensive have a higher rate of objective_response than fit patients (because the unfit subgroup is preferentially treated with venetoclax-based therapy).<br>Female sex is associated with a different rate of objective_response than male sex.<br>Higher blast_pct_marrow is associated with lower objective_response. |
| 018 | claude-opus-4-7 | claude-code@named-task | 0.290 | 25 | 25 | exact | 25 | Lower sodium_meq_l is associated with lower objective_response rate (i.e., higher sodium = higher response).<br>potassium_meq_l is associated with objective_response rate.<br>calcium_mg_dl is associated with objective_response rate. |
| 019 | claude-opus-4-7 | claude-code-direct@v1 | 0.295 | 25 | 25 | exact | 25 | Female sex (sex_female=1) is associated with a different probability of objective_response than male sex.<br>Higher sodium_meq_l (within range) is associated with higher objective_response.<br>Higher potassium_meq_l is associated with lower objective_response. |
| 020 | claude-opus-4-7 | claude-code@manual | 0.255 | 14 | 14 | exact | 14 | The effect of `treatment_venetoclax_azacitidine` on `objective_response` is more positive in patients with `npm1_mutation` set than in those without.<br>The benefit of `treatment_venetoclax_azacitidine` is jointly concentrated in patients with both `npm1_mutation` set AND `unfit_for_intensive` set, with little to no benefit when only one of those conditions holds.<br>Within the NPM1+ AND unfit_for_intensive subgroup, the `treatment_venetoclax_azacitidine` benefit on `objective_response` is suppressed when `tp53_mutation` is also set. |

#### Anonymized (n_replicates=20)
- buried_score: 9.65 ± 2.58
- replicates uncovered: 20/20
- near-or-better recovery: 20/20
- component-or-better recovery: 20/20
- recovery levels: exact=20, near=0, component=0, none=0

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 9 | 9 | exact | 9 |
| 002 | claude-opus-4-7 | claude-code-manual@v1 | 13 | 13 | exact | 13 |
| 003 | claude-opus-4-7 | claude-code@manual-analysis | 10 | 10 | exact | 10 |
| 004 | claude-opus-4-7 | claude-code-manual@2026-05 | 8 | 8 | exact | 8 |
| 005 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 11 | 11 | exact | 11 |
| 006 | claude-opus-4-7 | claude-code@native-2026-05 | 8 | 8 | exact | 8 |
| 007 | claude-opus-4-7 | claude-code@inline | 10 | 10 | exact | 10 |
| 008 | claude-opus-4-7 | claude-code-direct@2026-05-03 | 8 | 8 | exact | 8 |
| 009 | claude-opus-4-7 | claude-code@manual-iterative-2026-05-03 | 12 | 12 | exact | 12 |
| 010 | claude-opus-4-7 | claude-code@manual-analysis-2026-05-03 | 7 | 7 | exact | 7 |
| 011 | claude-opus-4-7 | claude-code@anonymized-aml | 8 | 8 | exact | 8 |
| 012 | claude-opus-4-7 | claude-code@manual-2026.05.03 | 10 | 10 | exact | 10 |
| 013 | claude-opus-4-7 | claude-code@manual | 8 | 8 | exact | 8 |
| 014 | claude-opus-4-7 | claude-code@manual-harness-1 | 15 | 15 | exact | 15 |
| 015 | claude-opus-4-7 | claude-code@manual-1.0 | 10 | 10 | exact | 10 |
| 016 | claude-opus-4-7 | claude-code@manual-analysis | 7 | 7 | exact | 7 |
| 017 | claude-opus-4-7 | claude-code-cli@opus-4-7-1m | 15 | 15 | exact | 15 |
| 018 | claude-opus-4-7 | claude-code-manual@1.0 | 7 | 7 | exact | 7 |
| 019 | claude-opus-4-7 | claude-code@opus-4-7-anonymized | 11 | 11 | exact | 11 |
| 020 | claude-opus-4-7 | claude-code@interactive-session | 6 | 6 | exact | 6 |

### ds001_breast

#### Named (n_replicates=20)
- frac_novel: 0.371 ± 0.139
- buried_score: 20.85 ± 5.72
- replicates uncovered: 10/20
- near-or-better recovery: 18/20
- component-or-better recovery: 20/20
- recovery levels: exact=10, near=8, component=2, none=0

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 0.358 | 19 | 19 | exact | 19 | Older age_years is associated with longer pfs_months (positive Pearson correlation).<br>pik3ca_mutation=1 is associated with shorter pfs_months than pik3ca_mutation=0 after adjustment.<br>The treatment_palbociclib effect on pfs_months is smaller (or absent) in pik3ca_mutation=1 patients than in pik3ca_mutation=0 patients (negative treatment_palbociclib x pik3ca_mutation interaction). |
| 002 | claude-opus-4-7 | claude-code@manual-iterative | 0.404 | 26 | — | near | 8 | Patients with pik3ca_mutation=1 have shorter pfs_months than wild-type.<br>Patients with brca1_mutation=1 differ in pfs_months versus wild-type.<br>After multivariable adjustment, treatment_tamoxifen has no main effect on pfs_months. |
| 003 | claude-opus-4-7 | claude-code@manual-analysis-v1 | 0.425 | 13 | 13 | exact | 13 | HER2-positive (her2_positive=1) tumors have shorter pfs_months than HER2-negative tumors (more aggressive biology).<br>PIK3CA-mutated tumors (pik3ca_mutation=1) have shorter pfs_months than PIK3CA wild-type tumors.<br>Treatment assignment is roughly random with respect to clinical/biomarker features (synthetic dataset; no confounding by indication). |
| 004 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 0.273 | 26 | — | near | 10 | HER2-low status (her2_low=1) is associated with longer pfs_months than HER2-zero.<br>PIK3CA mutation (pik3ca_mutation=1) is associated with shorter pfs_months.<br>BRCA1 or BRCA2 mutation is associated with shorter pfs_months on average. |
| 005 | claude-opus-4-7 | claude-code@opus-4-7-named-1m | 0.286 | 15 | 15 | exact | 15 | Mean pfs_months is higher in her2_low=1 vs her2_low=0 patients.<br>The effect of treatment_palbociclib on pfs_months is more negative (smaller benefit) in her2_positive=1 than in her2_positive=0 patients.<br>The effect of treatment_trastuzumab on pfs_months is more positive in her2_low=1 than in her2_low=0 patients. |
| 006 | claude-opus-4-7 | claude-code@manual-iter-2026-05-03 | 0.345 | 26 | — | near | 14 | Patients with her2_low=1 have different pfs_months than those without.<br>After multivariable adjustment for demographics, disease severity, biomarkers, and labs, treatment_palbociclib remains the only treatment with a robust positive main effect on pfs_months; trastuzumab, olaparib, pembrolizumab, and tamoxifen show no positive adjusted main effect.<br>The palbociclib benefit is concentrated specifically in er_positive=1 AND her2_positive=0 patients (regardless of pr_positive), with essentially no benefit in any other ER/HER2 cell. |
| 007 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 0.314 | 26 | — | near | 15 | Tamoxifen produces longer pfs_months in postmenopausal patients than in premenopausal patients.<br>Palbociclib's PFS benefit is reduced or abolished in HER2-positive patients (palbociclib×HER2 interaction is negative).<br>Palbociclib's PFS benefit is reduced or abolished in PIK3CA-mutant patients (palbociclib×PIK3CA interaction is negative). |
| 008 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 0.644 | 13 | 13 | exact | 13 | age_years is positively correlated with pfs_months (continuous main effect).<br>tumor_size_cm is positively correlated with pfs_months (continuous main effect).<br>potassium_meq_l is positively correlated with pfs_months (continuous main effect). |
| 009 | claude-opus-4-7[1m] | claude-code-cli@manual | 0.073 | 26 | — | component | 5 | Higher potassium_meq_l is associated with shorter pfs_months (negative slope).<br>Across all six treatments and 12 binary modifiers, at least one significant treatment×modifier interaction on pfs_months exists beyond the obvious receptor matches.<br>Treatment effects are blunted in patients with ecog_ps>=2: every treatment × ecog_ps interaction is non-positive (treatment effect smaller as ECOG rises). |
| 010 | claude-opus-4-7 | claude-code-manual-run@2026-05-03 | 0.306 | 26 | — | near | 25 | her2_low=1 patients have different mean pfs_months than non-HER2-low patients.<br>Higher albumin_g_dl (better nutrition/liver synthetic function) is associated with longer pfs_months.<br>Higher ldh_u_l (tumor burden marker) is associated with shorter pfs_months. |
| 011 | claude-opus-4-7 | manual-claude-code-session@2026-05-03 | 0.380 | 26 | — | near | 15 | her2_low=1 is associated with longer pfs_months than her2_low=0.<br>brca2_mutation=1 is associated with shorter pfs_months than brca2_mutation=0.<br>After adjusting for all features and treatments simultaneously, stage_iv, has_brain_mets, ecog_ps, her2_positive, pik3ca_mutation, ki67_pct, weight_loss_pct_6mo, and ldh_u_l independently shorten pfs_months, while er_positive, age_years, and albumin_g_dl independently lengthen it. |
| 012 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 0.390 | 26 | — | component | 1 | Patients with pik3ca_mutation = 1 have shorter pfs_months than patients without the mutation.<br>Patients with her2_positive = 1 have shorter pfs_months than patients without HER2 amplification.<br>Patients with her2_low = 1 have longer pfs_months than patients with her2_low = 0. |
| 013 | claude-opus-4-7 | claude-code-cli@interactive-session | 0.452 | 26 | — | near | 19 | Mean pfs_months is lower in patients with postmenopausal=1 than postmenopausal=0.<br>Mean pfs_months is differs in patients with sex_female=1 than sex_female=0.<br>pfs_months is positively associated with age_years as a continuous predictor. |
| 014 | claude-opus-4-7 | claude-code-custom@local | 0.692 | 17 | 17 | exact | 17 | Higher total_bilirubin_mg_dl is associated with a different pfs_months (linear association).<br>Higher creatinine_mg_dl is associated with a different pfs_months (linear association).<br>Higher bun_mg_dl is associated with a different pfs_months (linear association). |
| 015 | claude-opus-4-7 | claude-code@manual-run | 0.208 | 19 | 19 | exact | 19 | After adjustment for demographics, performance status, stage, brain mets, ER/PR/HER2/HER2-low, BRCA1/BRCA2/PIK3CA, Ki67, tumor size, albumin, LDH, weight loss, CRP, NLR, hemoglobin, and alkaline phosphatase, only treatment_palbociclib retains a clinically meaningful adjusted PFS benefit; the other five treatments do not.<br>Within ER-positive HER2-negative patients, palbociclib × pik3ca_mutation interaction is negative: palbociclib prolongs pfs_months in PIK3CA wild-type but not in PIK3CA-mutated patients.<br>Within ER-positive HER2-negative patients, palbociclib effect on pfs_months declines with rising ki67_pct (negative tx*ki67 interaction). |
| 016 | claude-opus-4-7 | claude-code@opus-4-7-1m | 0.517 | 11 | 11 | exact | 11 | pik3ca_mutation=1 is associated with shorter pfs_months than pik3ca_mutation=0.<br>her2_low=1 is associated with longer pfs_months than her2_low=0.<br>The palbociclib effect on pfs_months is negatively modified by pik3ca_mutation (no benefit in PIK3CA-mut). |
| 017 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 0.321 | 14 | 14 | exact | 14 | her2_low=1 patients have different pfs_months than her2_low=0.<br>Routine labs (hemoglobin, alk phos, AST, ALT, bilirubin, creatinine, BUN, sodium, potassium, calcium, tumor_size_cm) have negligible association with pfs_months.<br>In a multivariable OLS model of pfs_months on all features and treatments, treatment_palbociclib retains a substantial positive coefficient while tamoxifen, trastuzumab, olaparib, sacituzumab govitecan, and pembrolizumab do not. |
| 018 | claude-opus-4-7 | claude-code@manual-script | 0.357 | 15 | 15 | exact | 15 | brca1_mutation or brca2_mutation positivity is associated with shorter pfs_months.<br>sex_female has a main effect on pfs_months in this breast cohort.<br>In a multivariable OLS for pfs_months including all features and treatments, age_years, ecog_ps, stage_iv, has_brain_mets, weight_loss_pct_6mo, albumin_g_dl, ki67_pct, er_positive, her2_positive, pik3ca_mutation, ldh_u_l, and treatment_palbociclib will each have independent effects with the directions seen in univariate analysis. |
| 019 | claude-opus-4-7 | claude-code-custom@2026-05-03 | 0.256 | 26 | — | near | 17 | After adjustment for stage, ECOG, age, key biomarkers, and labs, treatment_pembrolizumab remains associated with pfs_months.<br>treatment_sacituzumab_govitecan × her2_low interaction on pfs is positive.<br>treatment_palbociclib × postmenopausal interaction on pfs is positive. |
| 020 | claude-opus-4-7 | claude-code-fresh@2026-05-03 | 0.418 | 21 | 21 | exact | 21 | PIK3CA-mutated (pik3ca_mutation=1) patients have shorter pfs_months than PIK3CA wild-type patients.<br>Among ER-positive patients, treatment_tamoxifen prolongs pfs_months more in postmenopausal than in premenopausal patients.<br>Treatment_palbociclib's PFS benefit is larger in PR-positive patients (positive palbo*PR interaction). |

#### Anonymized (n_replicates=20)
- buried_score: 15.90 ± 7.06
- replicates uncovered: 15/20
- near-or-better recovery: 20/20
- component-or-better recovery: 20/20
- recovery levels: exact=15, near=5, component=0, none=0

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 13 | 13 | exact | 13 |
| 002 | claude-opus-4-7 | claude-code@native | 24 | 24 | exact | 24 |
| 003 | claude-opus-4-7 | claude-code-manual@1.0 | 11 | 11 | exact | 11 |
| 004 | claude-opus-4-7 | claude-code-manual@1.0 | 8 | 8 | exact | 8 |
| 005 | claude-opus-4-7 | claude-code-manual@1.0 | 26 | — | near | 7 |
| 006 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 17 | 17 | exact | 17 |
| 007 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 26 | — | near | 6 |
| 008 | claude-opus-4-7 | claude-code@manual-analysis-1.0 | 26 | — | near | 7 |
| 009 | claude-opus-4-7 | manual-claude-code@2026-05-03 | 26 | — | near | 25 |
| 010 | claude-opus-4-7 | claude-code-interactive@manual | 9 | 9 | exact | 9 |
| 011 | claude-opus-4-7 | claude-code@hand-driven-2026-05-03 | 12 | 12 | exact | 12 |
| 012 | claude-opus-4-7 | claude-code-interactive@2026-05-03 | 26 | — | near | 11 |
| 013 | claude-opus-4-7 | claude-code@manual-1m | 16 | 16 | exact | 16 |
| 014 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 17 | 17 | exact | 17 |
| 015 | claude-opus-4-7 | claude-code@interactive-session | 11 | 11 | exact | 11 |
| 016 | claude-opus-4-7 | claude-code-manual@1.0 | 12 | 12 | exact | 12 |
| 017 | claude-opus-4-7 | claude-code@manual-2026-05 | 9 | 9 | exact | 9 |
| 018 | claude-opus-4-7 | claude-code@manual | 9 | 9 | exact | 9 |
| 019 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 12 | 12 | exact | 12 |
| 020 | claude-opus-4-7 | claude-code@manual-analysis | 8 | 8 | exact | 8 |

### ds001_crc

#### Named (n_replicates=20)
- frac_novel: 0.398 ± 0.098
- buried_score: 18.75 ± 6.27
- replicates uncovered: 16/20
- near-or-better recovery: 20/20
- component-or-better recovery: 20/20
- recovery levels: exact=16, near=4, component=0, none=0

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@local | 0.385 | 19 | 19 | exact | 19 | Older age_years is associated with longer pfs_months (positive slope) in this colorectal cohort.<br>After multivariable adjustment for age_years, ecog_ps, stage_iv, biomarkers, and labs, treatment_regorafenib retains a positive main effect on pfs_months while the other five treatments do not.<br>treatment_bevacizumab effect on pfs_months differs by right_sided_primary status. |
| 002 | claude-opus-4-7 | claude-code@manual-analyst-v1 | 0.429 | 11 | 11 | exact | 11 | After adjusting for prognostic features (ECOG, stage_iv, age_years, sex_female, right_sided_primary, albumin_g_dl, weight_loss_pct_6mo, cea_ng_ml, ldh_u_l, crp_mg_l, nlr, hemoglobin_g_dl, biomarkers), only treatment_regorafenib has a non-zero average pfs_months effect among the 6 treatments.<br>Regorafenib-treated patients are not meaningfully different from non-regorafenib patients on baseline ECOG, stage_iv, age_years, albumin_g_dl, weight_loss_pct_6mo, or biomarker frequencies (i.e. its main effect is not driven by baseline confounding).<br>For each treatment, treatment×feature interactions (with each of the 27 candidate features one at a time, controlling for ECOG/stage/age/albumin/weight loss) reveal substantial heterogeneity. The only treatment with such heterogeneity is regorafenib. |
| 003 | claude-opus-4-7 | claude-code@manual-iteration | 0.360 | 26 | — | near | 13 | In a multivariable OLS model of pfs_months on all 33 candidate predictors (including all six treatments), the strongest independent predictors of pfs_months are age_years (positive), ecog_ps (negative), stage_iv (negative), albumin_g_dl (positive), weight_loss_pct_6mo (negative), kras_mutation (negative), braf_v600e (negative), right_sided_primary (negative), cea_ng_ml (negative), and treatment_regorafenib (positive); other treatments (cetuximab, bevacizumab, pembrolizumab, encorafenib, trastuzumab_tucatinib) have null adjusted main effects.<br>Cetuximab differentially affects pfs_months by braf_v600e status.<br>Bevacizumab improves pfs_months in some clinically meaningful subgroup defined by stage_iv x right_sided_primary. |
| 004 | claude-opus-4-7 | claude-code@manual-bundle | 0.352 | 14 | 14 | exact | 14 | sex_female is associated with a difference in pfs_months versus male patients.<br>The regorafenib effect on pfs_months is modified by kras_mutation: regorafenib improves PFS in KRAS wild-type but has reduced/no benefit in KRAS-mutant.<br>The regorafenib effect on pfs_months is modified by braf_v600e: regorafenib improves PFS in BRAF wild-type but has reduced/no benefit in BRAF V600E. |
| 005 | claude-opus-4-7 | claude-code@manual-bundle | 0.315 | 14 | 14 | exact | 14 | The regorafenib effect on pfs_months is modified by kras_mutation: regorafenib improves PFS in KRAS wild-type but loses its benefit in KRAS-mutant.<br>The regorafenib effect on pfs_months is modified by braf_v600e: regorafenib improves PFS in BRAF wild-type but loses its benefit in BRAF V600E.<br>The regorafenib effect on pfs_months is modified by right_sided_primary: regorafenib improves PFS in left-sided but loses its benefit in right-sided primary. |
| 006 | claude-opus-4-7 | manual-claude-code@1.0 | 0.500 | 21 | 21 | exact | 21 | The PFS effect of treatment_regorafenib differs by kras_mutation status — benefit concentrated in KRAS-WT.<br>The PFS effect of treatment_regorafenib differs by braf_v600e status — benefit concentrated in BRAF-WT.<br>The PFS effect of treatment_regorafenib differs by right_sided_primary — benefit concentrated in left-sided. |
| 007 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 0.517 | 10 | 10 | exact | 10 | treatment_regorafenib benefit on pfs_months is differential by kras_mutation status, with larger benefit in kras_mutation=0.<br>treatment_regorafenib benefit on pfs_months is differential by braf_v600e status, with larger benefit in braf_v600e=0.<br>treatment_regorafenib benefit on pfs_months is differential by right_sided_primary, with larger benefit in right_sided_primary=0. |
| 008 | claude-opus-4-7 | claude-code@self-run | 0.287 | 8 | 8 | exact | 8 | sodium_meq_l is correlated with pfs_months (Pearson r != 0).<br>potassium_meq_l is correlated with pfs_months (Pearson r != 0).<br>calcium_mg_dl is correlated with pfs_months (Pearson r != 0). |
| 009 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 0.220 | 11 | 11 | exact | 11 | After adjusting for prognostic features, treatment_regorafenib retains a substantial positive main effect on pfs_months, while no other treatment shows a non-zero adjusted main effect.<br>The pfs_months benefit of treatment_regorafenib is concentrated in kras_mutation=0 (KRAS wild-type), with little or no benefit in kras_mutation=1 (negative interaction).<br>The pfs_months benefit of treatment_regorafenib is concentrated in braf_v600e=0 (BRAF wild-type), with little or no benefit in braf_v600e=1 (negative interaction). |
| 010 | claude-opus-4-7 | claude-code-named@1.0 | 0.314 | 18 | 18 | exact | 18 | Effect of treatment_bevacizumab on pfs_months differs by stage_iv.<br>Effect of treatment_bevacizumab on pfs_months differs by braf_v600e.<br>Effect of treatment_regorafenib on pfs_months is larger in KRAS-wt than KRAS-mutant (interaction). |
| 011 | claude-opus-4-7 | claude-code@1.0-named-crc | 0.325 | 26 | — | near | 25 | treatment_pembrolizumab improves pfs_months in right-sided tumors (right_sided_primary=1) more than in left-sided.<br>There is an interaction between treatment_pembrolizumab and braf_v600e on pfs_months.<br>treatment_regorafenib improves mean pfs_months in patients with ECOG 0 (ecog_ps=0) vs no regorafenib. |
| 012 | claude-opus-4-7 | claude-code-named@2026-05-03 | 0.415 | 14 | 14 | exact | 14 | The treatment_regorafenib effect on pfs_months is larger in kras_mutation=0 than in kras_mutation=1 (regorafenib benefit concentrated in KRAS wild-type).<br>The treatment_regorafenib effect on pfs_months is larger in braf_v600e=0 than in braf_v600e=1.<br>The treatment_regorafenib effect on pfs_months is larger in right_sided_primary=0 (left-sided primaries) than in right_sided_primary=1. |
| 013 | claude-opus-4-7 | claude-code@custom-eval-harness | 0.378 | 25 | 25 | exact | 25 | Within HER2-amplified, the trastuzumab/tucatinib effect on `pfs_months` is greater when patients are also RAS wild-type (`kras_mutation`=0 AND `nras_mutation`=0).<br>Bevacizumab effect on `pfs_months` differs by KRAS mutation status (treatment_bevacizumab×kras_mutation interaction is non-zero).<br>Bevacizumab effect on `pfs_months` differs by tumor sidedness (treatment_bevacizumab×right_sided_primary interaction is non-zero). |
| 014 | claude-opus-4-7 | claude-code@manual-iter | 0.417 | 25 | 25 | exact | 25 | Treatment_bevacizumab produces longer pfs_months in right-sided primary tumor patients than in left-sided.<br>Treatment_regorafenib produces a larger pfs_months benefit in patients with low CEA (cea_ng_ml at or below median) than in patients with high CEA.<br>Treatment_regorafenib effect on pfs_months differs across ECOG performance status categories. |
| 015 | claude-opus-4-7 | claude-code@inline | 0.573 | 25 | 25 | exact | 25 | Higher nlr is associated with longer pfs_months (linear slope hypothesis: nonzero).<br>Higher hemoglobin_g_dl is associated with shorter pfs_months (linear slope hypothesis: nonzero).<br>Higher ast_u_l is associated with longer pfs_months (linear slope hypothesis: nonzero). |
| 016 | claude-opus-4-7 | claude-code-manual@local | 0.612 | 22 | 22 | exact | 22 | There is a treatment_bevacizumab × kras_mutation interaction on pfs_months.<br>There is a treatment_bevacizumab × braf_v600e interaction on pfs_months.<br>There is a treatment_bevacizumab × msi_high interaction on pfs_months. |
| 017 | claude-opus-4-7 | claude-code-manual@1 | 0.339 | 20 | 20 | exact | 20 | potassium_meq_l is correlated with pfs_months (any direction).<br>After adjusting for all features simultaneously (OLS), age_years, ecog_ps, stage_iv, right_sided_primary, kras_mutation, braf_v600e, albumin_g_dl, weight_loss_pct_6mo, cea_ng_ml, ldh_u_l and treatment_regorafenib remain independently associated with pfs_months.<br>After adjusting for clinical covariates, the canonical targeted-therapy associations (cetuximab in RAS-WT, pembrolizumab in MSI-high, encorafenib in BRAF V600E, trastuzumab/tucatinib in HER2+) remain non-significant. |
| 018 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 0.458 | 14 | 14 | exact | 14 | Older age_years is associated with longer pfs_months (positive Pearson correlation).<br>After adjustment, treatment_regorafenib remains the only systemic-therapy variable with a non-zero main effect on pfs_months; the other five treatments have null adjusted main effects.<br>treatment_pembrolizumab effect varies by ecog_ps with greater benefit in ECOG 0. |
| 019 | claude-opus-4-7 | claude-code-opus-4-7@manual-2026-05-03 | 0.422 | 26 | — | near | 5 | Older age_years is associated with longer pfs_months (positive correlation).<br>In a multivariable OLS model with all features and treatments, treatment_regorafenib has an independent positive effect on pfs_months.<br>Regorafenib effect on PFS is modified by stage_iv (stronger in metastatic vs non-metastatic). |
| 020 | claude-opus-4-7 | claude-code@named-task-2026-05-03 | 0.342 | 26 | — | near | 13 | sodium_meq_l is associated with pfs_months.<br>potassium_meq_l is associated with pfs_months.<br>calcium_mg_dl is associated with pfs_months. |

#### Anonymized (n_replicates=20)
- buried_score: 12.60 ± 5.03
- replicates uncovered: 20/20
- near-or-better recovery: 20/20
- component-or-better recovery: 20/20
- recovery levels: exact=20, near=0, component=0, none=0

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 10 | 10 | exact | 10 |
| 002 | claude-opus-4-7 | claude-code@opus-4-7-1m-manual | 11 | 11 | exact | 11 |
| 003 | claude-opus-4-7 | claude-code-interactive@manual | 6 | 6 | exact | 6 |
| 004 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 12 | 12 | exact | 12 |
| 005 | claude-opus-4-7 | claude-code@inline-2026-05-03 | 7 | 7 | exact | 7 |
| 006 | claude-opus-4-7 | claude-code@opus-4-7-1m | 10 | 10 | exact | 10 |
| 007 | claude-opus-4-7 | claude-code@manual-analysis-1 | 11 | 11 | exact | 11 |
| 008 | claude-opus-4-7 | claude-code@inline-analysis | 9 | 9 | exact | 9 |
| 009 | claude-opus-4-7 | claude-code@manual-bash-harness | 13 | 13 | exact | 13 |
| 010 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 9 | 9 | exact | 9 |
| 011 | claude-opus-4-7 | claude-code-manual@1.0 | 10 | 10 | exact | 10 |
| 012 | claude-opus-4-7 | claude-code@manual-analysis | 15 | 15 | exact | 15 |
| 013 | claude-opus-4-7 | claude-code@manual-analysis-2026-05-03 | 9 | 9 | exact | 9 |
| 014 | claude-opus-4-7 | claude-code-interactive@manual-2026-05-03 | 11 | 11 | exact | 11 |
| 015 | claude-opus-4-7 | claude-code@opus-4-7-1m | 18 | 18 | exact | 18 |
| 016 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 20 | 20 | exact | 20 |
| 017 | claude-opus-4-7 | claude-code@manual-iterative | 16 | 16 | exact | 16 |
| 018 | claude-opus-4-7 | claude-code@manual-iter-2026-05-03 | 21 | 21 | exact | 21 |
| 019 | claude-opus-4-7 | claude-code@manual | 9 | 9 | exact | 9 |
| 020 | claude-opus-4-7 | claude-code@local | 25 | 25 | exact | 25 |

### ds001_nsclc

#### Named (n_replicates=20)
- frac_novel: 0.374 ± 0.105
- buried_score: 20.85 ± 7.65
- replicates uncovered: 8/20
- near-or-better recovery: 18/20
- component-or-better recovery: 20/20
- recovery levels: exact=8, near=10, component=2, none=0

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@1.0 | 0.256 | 7 | 7 | exact | 7 | brca2_mutation=1 patients have different mean pfs_months than brca2_mutation=0 patients (marginal).<br>treatment_osimertinib improves pfs_months more in alk_fusion=1 patients than alk_fusion=0 patients (off-label/unexpected).<br>The age–pfs_months relationship is monotonically increasing across the age range (no inverted-U shape). |
| 002 | claude-opus-4-7 | claude-code@manual-1 | 0.397 | 6 | 6 | exact | 6 | Sex modifies the sotorasib effect within KRAS G12C+ patients: the benefit is concentrated in males (sex_female=0).<br>Within KRAS G12C+ male (sex_female=0) patients, treatment_sotorasib raises pfs_months by >4 months on average.<br>Within KRAS G12C+ female (sex_female=1) patients, treatment_sotorasib has no effect on pfs_months. |
| 003 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 0.464 | 26 | — | near | 10 | Older age_years is associated with longer pfs_months (positive correlation in this cohort).<br>kras_g12c == 1 is associated with longer pfs_months than kras_g12c == 0.<br>stk11_mutation is associated with shorter pfs_months. |
| 004 | claude-opus-4-7 | claude-code@manual-analysis-2026-05-03 | 0.375 | 26 | — | near | 9 | Pembrolizumab is harmful (or less beneficial) in stk11_mutation=1 patients, even in PD-L1 high subset (negative interaction with STK11).<br>Pembrolizumab benefit is concentrated in patients with pdl1_tps>=0.5 AND tmb_high=1 AND stk11_mutation=0.<br>Osimertinib's lack of effect in EGFR+ patients persists when stratified by histology, smoking_status, has_brain_mets, stage_iv, ECOG, STK11, TMB and sex. |
| 005 | claude-opus-4-7 | claude-code-manual@opus47 | 0.611 | 10 | 10 | exact | 10 | Smoking_status differs in pfs_months across categories: current smokers have shorter pfs than former or never.<br>Female patients (sex_female=1) have shorter mean pfs_months than males.<br>Pembrolizumab effect on pfs_months increases with PD-L1 TPS quartile (i.e., effect largest in Q4). |
| 006 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 0.289 | 26 | — | near | 25 | Female patients (sex_female=1) have shorter pfs_months than male patients.<br>Among stk11_mutation=1 patients, treatment_pembrolizumab=1 has reduced or no benefit relative to wildtype; the treatment_pembrolizumab × stk11_mutation interaction is negative.<br>Adding the four treatments and their key biomarker interactions (pembro×pdl1_tps, pembro×tmb_high, pembro×stk11_mutation, sotorasib×kras_g12c, olaparib×brca2_mutation, osimertinib×egfr_mutation, osimertinib×has_brain_mets) to the multivariable model materially improves R^2, and only the sotorasib×kras_g12c interaction remains significant. |
| 007 | claude-opus-4-7 | claude-code@manual-loop-1 | 0.283 | 26 | — | near | 25 | Patients with kras_g12c=1 have longer pfs_months than KRAS G12C wild-type patients.<br>Patients with alk_fusion=1 have shorter pfs_months than ALK-negative patients.<br>Routine labs (hemoglobin_g_dl, alkaline_phosphatase_u_l, ast_u_l, alt_u_l, total_bilirubin_mg_dl, creatinine_mg_dl, bun_mg_dl, sodium_meq_l, potassium_meq_l, calcium_mg_dl) each have measurable main effects on pfs_months. |
| 008 | claude-opus-4-7 | claude-code@biased-llms-eval | 0.234 | 26 | — | component | 4 | The treatment_pembrolizumab effect on pfs_months is smaller (more negative) in patients with stk11_mutation == 1 — i.e., negative treatment_pembrolizumab × stk11_mutation interaction.<br>Within the subgroup stk11_mutation == 0 (STK11 wild-type), treatment_pembrolizumab == 1 patients have longer mean pfs_months than treatment_pembrolizumab == 0 patients.<br>A multivariable OLS of pfs_months on all clinical features (age, sex, smoking, ECOG, histology, stage, brain mets, mutations, biomarkers, labs) explains a large fraction of variance (R² > 0.5). |
| 009 | claude-opus-4-7 | claude-code@manual-iterations | 0.303 | 26 | — | near | 24 | Patients with brca2_mutation=1 have shorter pfs_months than wild-type patients.<br>After adjustment, treatment_olaparib still shows no overall effect.<br>After adjustment, treatment_osimertinib still shows no overall effect. |
| 010 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 0.400 | 26 | — | near | 7 | Treatment assignment appears randomized: biomarker prevalence does not differ between treated and untreated groups for any of the four treatments.<br>Within the egfr_mutation=1 subgroup, treatment_osimertinib does not change mean pfs_months.<br>Within the brca2_mutation=1 subgroup, treatment_olaparib does not change mean pfs_months. |
| 011 | claude-opus-4-7 | claude-code-manual@1.0 | 0.413 | 9 | 9 | exact | 9 | Higher age_years is associated with longer pfs_months (positive correlation).<br>sex_female = 1 is associated with shorter pfs_months than sex_female = 0.<br>brca2_mutation = 1 is associated with shorter pfs_months. |
| 012 | claude-opus-4-7 | claude-code@local-2026-05-03 | 0.255 | 26 | — | near | 22 | After joint adjustment, the apparent age and sex effects on pfs_months persist or change direction; specifically, female sex remains independently associated with shorter PFS.<br>The pfs_months benefit of treatment_pembrolizumab is reduced (less positive / negative interaction) in stk11_mutation==1 patients.<br>The pfs_months benefit of treatment_pembrolizumab differs between adenocarcinoma and squamous histology. |
| 013 | claude-opus-4-7 | claude-code@named-bundle | 0.554 | 22 | 22 | exact | 22 | KRAS G12C-mutant patients (kras_g12c=1) have longer pfs_months than kras_g12c=0 patients.<br>ALK-fusion-positive patients (alk_fusion=1) have shorter pfs_months than alk_fusion=0 patients.<br>There is a treatment_pembrolizumab × stage_iv interaction on pfs_months (i.e., the pembrolizumab effect differs by stage_iv). |
| 014 | claude-opus-4-7 | manual-claude-code@named-2026-05-03 | 0.466 | 15 | 15 | exact | 15 | Patients with stk11_mutation=1 have higher mean pfs_months than those with stk11_mutation=0.<br>Patients with brca2_mutation=1 have lower mean pfs_months than those with brca2_mutation=0.<br>Patients with tmb_high=1 have lower mean pfs_months than those with tmb_high=0. |
| 015 | claude-opus-4-7 | claude-code-manual@1.0 | 0.421 | 26 | — | component | 3 | kras_g12c=1 is associated with longer pfs_months than kras_g12c=0 in adjusted model.<br>brca2_mutation=1 is associated with shorter pfs_months than brca2_mutation=0 in adjusted model.<br>tmb_high=1 is associated with shorter pfs_months than tmb_high=0 in adjusted model. |
| 016 | claude-opus-4-7 | claude-code-manual@1.0 | 0.453 | 12 | 12 | exact | 12 | KRAS G12C mutation (kras_g12c=1) is associated with longer pfs_months in the overall cohort.<br>ALK fusion (alk_fusion=1) is associated with shorter pfs_months in the overall cohort.<br>After adjusting for age_years, sex_female, ecog_ps, stage_iv, has_brain_mets, histology, smoking_status, all biomarkers (egfr_mutation, kras_g12c, alk_fusion, stk11_mutation, brca2_mutation, tmb_high, pdl1_tps), and key labs (albumin_g_dl, ldh_u_l, weight_loss_pct_6mo, crp_mg_l, nlr, hemoglobin_g_dl), each treatment retains a positive main effect on pfs_months. |
| 017 | claude-opus-4-7 | claude-code@kk-ds001-nsclc-named-2026-05 | 0.274 | 24 | 24 | exact | 24 | The treatment_pembrolizumab effect on pfs_months is greater in non-squamous adenocarcinoma patients (negative treatment x squamous interaction).<br>The treatment_osimertinib effect on pfs_months differs by alk_fusion status.<br>The treatment_osimertinib effect on pfs_months differs by has_brain_mets status. |
| 018 | claude-opus-4-7 | claude-code-manual@kk-2026-05-03 | 0.333 | 26 | — | near | 14 | stk11_mutation+ patients have different pfs_months than stk11_mutation- patients (overall sample).<br>brca2_mutation+ patients have different pfs_months than brca2_mutation- patients (overall sample).<br>After adjustment for all measured features in a multivariable OLS model on log(pfs_months+0.1), age_years, ecog_ps, stage_iv, has_brain_mets, albumin_g_dl, weight_loss_pct_6mo, ldh_u_l, smoking status, histology, kras_g12c, alk_fusion remain independently associated with PFS, while pdl1_tps, tmb_high, egfr_mutation, stk11_mutation, and most chemistries are not. |
| 019 | claude-opus-4-7 | manual-claude-code@nsclc-named-2026-05-03 | 0.426 | 26 | — | near | 25 | kras_g12c=1 is associated with longer pfs_months than kras_g12c=0 (positive univariate beta).<br>alk_fusion=1 is associated with shorter pfs_months than alk_fusion=0 (negative univariate beta).<br>tmb_high=1 is associated with shorter pfs_months than tmb_high=0 (negative univariate beta). |
| 020 | claude-opus-4-7 | claude-code@manual-run | 0.281 | 26 | — | near | 15 | Older age_years is associated with longer pfs_months in this cohort.<br>At least one feature shows a significant treatment_sotorasib x feature interaction on pfs_months beyond kras_g12c (suggesting an additional modifier of sotorasib effect).<br>At least one feature shows a significant treatment_olaparib x feature interaction on pfs_months. |

#### Anonymized (n_replicates=20)
- buried_score: 13.60 ± 6.00
- replicates uncovered: 17/20
- near-or-better recovery: 20/20
- component-or-better recovery: 20/20
- recovery levels: exact=17, near=3, component=0, none=0

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 10 | 10 | exact | 10 |
| 002 | claude-opus-4-7 | claude-code-manual@1.0.0 | 7 | 7 | exact | 7 |
| 003 | claude-opus-4-7 | claude-code@manual-1m-context | 15 | 15 | exact | 15 |
| 004 | claude-opus-4-7 | claude-code-manual-driver@2026-05-03 | 16 | 16 | exact | 16 |
| 005 | claude-opus-4-7 | claude-code-manual@1.0 | 9 | 9 | exact | 9 |
| 006 | claude-opus-4-7 | claude-code@interactive-1.0 | 14 | 14 | exact | 14 |
| 007 | claude-opus-4-7 | claude-code@manual-run | 13 | 13 | exact | 13 |
| 008 | claude-opus-4-7 | claude-code@opus-4-7-1m | 10 | 10 | exact | 10 |
| 009 | claude-opus-4-7 | claude-code@manual-run-2026-05-03 | 14 | 14 | exact | 14 |
| 010 | claude-opus-4-7 | claude-code@nsclc-anonymized-v1 | 12 | 12 | exact | 12 |
| 011 | claude-opus-4-7 | claude-code@manual | 26 | — | near | 11 |
| 012 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 9 | 9 | exact | 9 |
| 013 | claude-opus-4-7 | claude-code@manual-script | 6 | 6 | exact | 6 |
| 014 | claude-opus-4-7 | claude-code@nsclc-anonymized-1 | 26 | — | near | 13 |
| 015 | claude-opus-4-7 | claude-code@interactive-1m | 26 | — | near | 10 |
| 016 | claude-opus-4-7 | claude-code-interactive@anonymized-2026-05 | 9 | 9 | exact | 9 |
| 017 | claude-opus-4-7 | claude-code@opus-4-7-1m | 14 | 14 | exact | 14 |
| 018 | claude-opus-4-7 | claude-code@2.0 | 9 | 9 | exact | 9 |
| 019 | claude-opus-4-7 | claude-code@manual-loop | 14 | 14 | exact | 14 |
| 020 | claude-opus-4-7 | claude-code@opus-4-7-1m-anonymized | 13 | 13 | exact | 13 |

### ds001_prostate

#### Named (n_replicates=20)
- frac_novel: 0.437 ± 0.087
- buried_score: 13.70 ± 6.11
- replicates uncovered: 18/20
- near-or-better recovery: 19/20
- component-or-better recovery: 20/20
- recovery levels: exact=18, near=1, component=1, none=0

| replicate | model | harness | frac_novel | buried_score | exact@ | recovery | recovery@ | sample novel hypotheses |
|---|---|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 0.431 | 12 | 12 | exact | 12 | brca2_mutation=1 is associated with a lower rate of objective_response than brca2_mutation=0.<br>msi_high=1 is associated with a lower rate of objective_response than msi_high=0.<br>After adjustment for ECOG, PSA, mcrpc, brca2, ar_v7, msi_high, albumin, weight loss, CRP, and other treatments, treatment_enzalutamide retains a positive association with objective_response. |
| 002 | claude-opus-4-7 | claude-code@manual-iter | 0.421 | 4 | 4 | exact | 4 | Patients with brca2_mutation=1 have a lower objective_response rate than patients with brca2_mutation=0.<br>Patients with msi_high=1 have a lower objective_response rate than patients with msi_high=0.<br>Treatment_enzalutamide raises objective_response in mcrpc=0 patients but provides no benefit in mcrpc=1 patients (castration-resistance modifies effect). |
| 003 | claude-opus-4-7 | claude-code@manual-prostate-named | 0.391 | 14 | 14 | exact | 14 | Patients with brca2_mutation=1 have a lower objective_response rate than patients with brca2_mutation=0.<br>After controlling for all features and treatments, treatment_enzalutamide retains a significant positive effect on objective_response.<br>After adjustment, treatment_abiraterone, treatment_docetaxel, treatment_olaparib, treatment_lu177_psma, and treatment_pembrolizumab each have a near-zero average effect on objective_response. |
| 004 | claude-opus-4-7 | claude-code@manual | 0.423 | 13 | 13 | exact | 13 | Patients with msi_high=1 have a lower rate of objective_response than patients with msi_high=0.<br>Patients with psma_high=1 have a different rate of objective_response than patients with psma_high=0.<br>Higher crp_mg_l is associated with a lower rate of objective_response. |
| 005 | claude-opus-4-7 | claude-code@manual | 0.408 | 14 | 14 | exact | 14 | Patients with brca2_mutation have a lower rate of objective_response than patients without brca2_mutation.<br>Patients with msi_high status have a lower rate of objective_response than patients without msi_high status.<br>The treatment_enzalutamide effect on objective_response is smaller (or absent) in mcrpc patients than in non-mcrpc patients (negative interaction). |
| 006 | claude-opus-4-7 | claude-code-manual@1.0 | 0.673 | 5 | 5 | exact | 5 | brca2_mutation=1 is associated with lower objective_response rate than brca2_mutation=0.<br>msi_high=1 is associated with lower objective_response rate than msi_high=0.<br>After adjusting for all baseline features and other treatments, treatment_enzalutamide remains positively associated with objective_response while treatment_abiraterone, treatment_docetaxel, treatment_olaparib, treatment_lu177_psma, and treatment_pembrolizumab show no significant adjusted main effect on objective_response. |
| 007 | claude-opus-4-7 | claude-code@manual-iteration-2026-05-03 | 0.413 | 7 | 7 | exact | 7 | Patients with brca2_mutation=1 have a lower rate of objective_response than patients with brca2_mutation=0.<br>Patients with msi_high=1 have a lower rate of objective_response than patients with msi_high=0.<br>treatment_enzalutamide increases objective_response more strongly in patients with mcrpc=0 (hormone-sensitive) than in mcrpc=1 (castration-resistant). |
| 008 | claude-opus-4-7 | claude-code@interactive-2026-05 | 0.561 | 12 | 12 | exact | 12 | `visceral_mets` and `gleason_score` have negligible main effects on `objective_response`.<br>Patients with `brca2_mutation` = 1 have a lower `objective_response` rate than patients with brca2_mutation = 0.<br>Patients with `msi_high` = 1 have a lower `objective_response` rate than msi_high = 0 patients. |
| 009 | claude-opus-4-7 | custom-react@analysis-1 | 0.500 | 12 | 12 | exact | 12 | After adjusting for all features and the other treatments, treatment_enzalutamide remains positively associated with objective_response.<br>After adjustment, treatment_abiraterone is associated with objective_response.<br>After adjustment, treatment_docetaxel is associated with objective_response. |
| 010 | claude-opus-4-7 | claude-code@manual-1.0 | 0.490 | 17 | 17 | exact | 17 | BRCA2-mutated patients (brca2_mutation=1) have a lower objective_response rate than wildtype patients.<br>In a multivariable logistic model adjusting for all clinical/biomarker/lab features, treatment_enzalutamide retains a strong positive effect on objective_response and the other five treatments do not.<br>BRCA2 mutation also abrogates the benefit of enzalutamide, separately from AR-V7 (treatment_enzalutamide × brca2_mutation interaction with negative sign). |
| 011 | claude-opus-4-7 | claude-code-custom@2026-05-03 | 0.286 | 25 | 25 | exact | 25 | Higher age_years is associated with a difference in objective_response rate (continuous association).<br>The treatment effect of treatment_enzalutamide is smaller in patients with mcrpc=1 than mcrpc=0 (negative enzalutamide x mcrpc interaction).<br>The treatment effect of treatment_enzalutamide is smaller in patients with higher ecog_ps. |
| 012 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 0.418 | 26 | — | near | 25 | Patients with brca2_mutation=1 have a LOWER objective_response rate than brca2_mutation=0.<br>Patients with msi_high=1 have a LOWER objective_response rate than msi_high=0.<br>treatment_olaparib=1 is associated with HIGHER objective_response. |
| 013 | claude-opus-4-7 | claude-code@manual-analysis-1.0 | 0.472 | 10 | 10 | exact | 10 | Patients with brca2_mutation=1 have a lower objective_response rate than patients with brca2_mutation=0.<br>Patients with msi_high=1 have a lower objective_response rate than patients with msi_high=0.<br>The treatment_enzalutamide effect on objective_response is greater in patients with mcrpc=0 than in patients with mcrpc=1 (positive interaction with non-mCRPC, equivalent to negative treatment_enzalutamide x mcrpc interaction). |
| 014 | claude-opus-4-7 | claude-code-named@1.0 | 0.327 | 13 | 13 | exact | 13 | The effect of treatment_enzalutamide on objective_response is attenuated in patients with mcrpc=1 compared to mcrpc=0.<br>The effect of treatment_abiraterone on objective_response is attenuated in patients with mcrpc=1 compared to mcrpc=0.<br>The effect of treatment_docetaxel on objective_response is attenuated in patients with mcrpc=1 compared to mcrpc=0. |
| 015 | claude-opus-4-7[1m] | claude-code-manual@2026-05-03 | 0.446 | 16 | 16 | exact | 16 | Patients receiving treatment_pembrolizumab have a higher rate of objective_response than those not receiving treatment_pembrolizumab.<br>potassium_meq_l is associated with objective_response (non-zero beta in logistic regression).<br>Higher calcium_mg_dl is associated with a lower objective_response rate (negative beta in logistic regression). |
| 016 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 0.352 | 10 | 10 | exact | 10 | Patients with brca2_mutation=1 have lower objective_response rate than brca2_mutation=0 patients.<br>Patients with msi_high=1 have lower objective_response rate than msi_high=0 patients.<br>In a fully adjusted multivariable logistic model for objective_response, treatment_enzalutamide remains a strong positive predictor; mcrpc, brca2_mutation, ar_v7_positive, msi_high, ecog_ps, log_psa, weight_loss_pct_6mo, and crp_mg_l remain negative; albumin_g_dl remains positive; the other treatments do not contribute. |
| 017 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 0.431 | 12 | 12 | exact | 12 | brca2_mutation=1 is associated with a lower rate of objective_response than brca2_mutation=0.<br>msi_high=1 is associated with a lower rate of objective_response than msi_high=0.<br>After adjustment for ECOG, PSA, mcrpc, brca2, ar_v7, msi_high, albumin, weight loss, CRP, and other treatments, treatment_enzalutamide retains a positive association with objective_response. |
| 018 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 0.500 | 11 | 11 | exact | 11 | Higher hemoglobin_g_dl (continuous) is associated with a negative change in the log-odds of objective_response (univariable logistic per 1 SD).<br>Higher alkaline_phosphatase_u_l (continuous) is associated with a positive change in the log-odds of objective_response (univariable logistic per 1 SD).<br>Higher alt_u_l (continuous) is associated with a positive change in the log-odds of objective_response (univariable logistic per 1 SD). |
| 019 | claude-opus-4-7 | claude-code-opus-4-7@manual-iter | 0.321 | 15 | 15 | exact | 15 | creatinine_mg_dl is associated with objective_response (higher Cr -> lower ORR).<br>total_bilirubin_mg_dl is associated with objective_response (higher bilirubin -> lower ORR).<br>ast_u_l is associated with objective_response (higher AST -> lower ORR). |
| 020 | claude-opus-4-7 | claude-code-named@2026-05 | 0.470 | 26 | — | component | 1 | Treatment with treatment_pembrolizumab is associated with higher objective_response than no pembrolizumab.<br>brca2_mutation = 1 patients have lower objective_response rate than brca2_mutation = 0 patients (BRCA2 alterations are a poor-prognostic marker in this cohort).<br>msi_high = 1 patients have lower objective_response rate than msi_high = 0 patients. |

#### Anonymized (n_replicates=20)
- buried_score: 9.90 ± 3.84
- replicates uncovered: 20/20
- near-or-better recovery: 20/20
- component-or-better recovery: 20/20
- recovery levels: exact=20, near=0, component=0, none=0

| replicate | model | harness | buried_score | exact@ | recovery | recovery@ |
|---|---|---|---|---|---|---|
| 001 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 7 | 7 | exact | 7 |
| 002 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 15 | 15 | exact | 15 |
| 003 | claude-opus-4-7 | claude-code-manual@1 | 9 | 9 | exact | 9 |
| 004 | claude-opus-4-7 | claude-code@interactive | 11 | 11 | exact | 11 |
| 005 | claude-opus-4-7 | claude-code@anonymized-ds001-prostate | 8 | 8 | exact | 8 |
| 006 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 11 | 11 | exact | 11 |
| 007 | claude-opus-4-7 | claude-code-manual@2026-05 | 11 | 11 | exact | 11 |
| 008 | claude-opus-4-7 | claude-code-manual@1.0 | 6 | 6 | exact | 6 |
| 009 | claude-opus-4-7 | claude-code@opus-4-7-1m | 10 | 10 | exact | 10 |
| 010 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 13 | 13 | exact | 13 |
| 011 | claude-opus-4-7 | claude-code@interactive-2026.05.03 | 11 | 11 | exact | 11 |
| 012 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 8 | 8 | exact | 8 |
| 013 | claude-opus-4-7 | claude-code-manual@1.0 | 4 | 4 | exact | 4 |
| 014 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 21 | 21 | exact | 21 |
| 015 | claude-opus-4-7 | claude-code-manual@2026-05-03 | 5 | 5 | exact | 5 |
| 016 | claude-opus-4-7 | claude-code-manual@opus-4-7-1m | 8 | 8 | exact | 8 |
| 017 | claude-opus-4-7 | claude-code@manual-2026-05-03 | 9 | 9 | exact | 9 |
| 018 | claude-opus-4-7 | claude-code@manual | 9 | 9 | exact | 9 |
| 019 | claude-opus-4-7 | claude-code@manual-iter-2026-05-03 | 8 | 8 | exact | 8 |
| 020 | claude-opus-4-7 | manual-claude-code-bundle@2026-05-03 | 14 | 14 | exact | 14 |
