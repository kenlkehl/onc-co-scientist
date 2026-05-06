"""Build transcript.json and analysis_summary.txt from analysis results."""
import json

transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-fresh@2026-05-03",
    "max_iterations": 25,
    "iterations": []
}

# --------- Iteration 1 ---------
transcript["iterations"].append({
    "index": 1,
    "proposed_hypotheses": [
        {"id": "h1_age", "text": "Older age_years is associated with longer pfs_months (positive correlation)."},
        {"id": "h1_ecog", "text": "Higher ecog_ps is associated with shorter pfs_months."},
        {"id": "h1_stage", "text": "Patients with stage_iv=1 have shorter pfs_months than patients with stage_iv=0."},
        {"id": "h1_brain", "text": "Patients with has_brain_mets=1 have shorter pfs_months than patients with has_brain_mets=0."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h1_age"], "code": "stats.pearsonr(df['age_years'], df['pfs_months'])",
         "result_summary": "Pearson correlation between age_years and pfs_months r=+0.702 (p<1e-300). Mean PFS rises monotonically from 0.5 mo at age 30s to 9.9 mo at age 90s.",
         "p_value": 0.0, "effect_estimate": 0.702, "significant": True},
        {"hypothesis_ids": ["h1_ecog"], "code": "df.groupby('ecog_ps')['pfs_months'].mean()",
         "result_summary": "Mean PFS by ecog_ps: 0=5.64, 1=4.44, 2=3.29 (one-way ANOVA F-test p<<1e-300).",
         "p_value": 0.0, "effect_estimate": -1.18, "significant": True},
        {"hypothesis_ids": ["h1_stage"], "code": "stats.ttest_ind(df.loc[df['stage_iv']==1,'pfs_months'], df.loc[df['stage_iv']==0,'pfs_months'])",
         "result_summary": "Mean PFS stage_iv=1: 3.62 mo vs stage_iv=0: 5.15 mo; difference -1.54 mo (Welch t-test p<<1e-300).",
         "p_value": 0.0, "effect_estimate": -1.54, "significant": True},
        {"hypothesis_ids": ["h1_brain"], "code": "stats.ttest_ind(df.loc[df['has_brain_mets']==1,'pfs_months'], df.loc[df['has_brain_mets']==0,'pfs_months'])",
         "result_summary": "Mean PFS has_brain_mets=1: 3.75 mo vs has_brain_mets=0: 4.80 mo; difference -1.05 mo (Welch t-test p=1.5e-178).",
         "p_value": 1.5e-178, "effect_estimate": -1.05, "significant": True},
    ]
})

# --------- Iteration 2 ---------
transcript["iterations"].append({
    "index": 2,
    "proposed_hypotheses": [
        {"id": "h2_er", "text": "ER-positive (er_positive=1) patients have longer pfs_months than ER-negative patients."},
        {"id": "h2_pr", "text": "PR-positive (pr_positive=1) patients have longer pfs_months than PR-negative patients."},
        {"id": "h2_her2", "text": "HER2-positive (her2_positive=1) patients have shorter pfs_months than HER2-negative patients."},
        {"id": "h2_her2low", "text": "HER2-low (her2_low=1) patients have longer pfs_months than HER2-not-low patients."},
        {"id": "h2_pik3ca", "text": "PIK3CA-mutated (pik3ca_mutation=1) patients have shorter pfs_months than PIK3CA wild-type patients."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h2_er"], "code": "stats.ttest_ind(by er_positive)",
         "result_summary": "Mean PFS er_positive=1: 4.86 mo vs er_positive=0: 4.30 mo; difference +0.57 mo (p=3.7e-143).",
         "p_value": 3.7e-143, "effect_estimate": 0.568, "significant": True},
        {"hypothesis_ids": ["h2_pr"], "code": "stats.ttest_ind(by pr_positive)",
         "result_summary": "Mean PFS pr_positive=1: 4.82 mo vs pr_positive=0: 4.46 mo; difference +0.36 mo (p=5.8e-59).",
         "p_value": 5.8e-59, "effect_estimate": 0.362, "significant": True},
        {"hypothesis_ids": ["h2_her2"], "code": "stats.ttest_ind(by her2_positive)",
         "result_summary": "Mean PFS her2_positive=1: 4.32 mo vs her2_positive=0: 4.78 mo; difference -0.46 mo (p=6.1e-70).",
         "p_value": 6.1e-70, "effect_estimate": -0.459, "significant": True},
        {"hypothesis_ids": ["h2_her2low"], "code": "stats.ttest_ind(by her2_low)",
         "result_summary": "Mean PFS her2_low=1: 4.76 mo vs her2_low=0: 4.65 mo; difference +0.11 mo (p=1.0e-6).",
         "p_value": 1.0e-6, "effect_estimate": 0.114, "significant": True},
        {"hypothesis_ids": ["h2_pik3ca"], "code": "stats.ttest_ind(by pik3ca_mutation)",
         "result_summary": "Mean PFS pik3ca_mutation=1: 4.33 mo vs pik3ca_mutation=0: 4.89 mo; difference -0.56 mo (p=6.1e-144).",
         "p_value": 6.1e-144, "effect_estimate": -0.559, "significant": True},
    ]
})

# --------- Iteration 3 ---------
transcript["iterations"].append({
    "index": 3,
    "proposed_hypotheses": [
        {"id": "h3_brca1", "text": "BRCA1-mutation carriers have different mean pfs_months than non-carriers (suspected lower)."},
        {"id": "h3_brca2", "text": "BRCA2-mutation carriers have different mean pfs_months than non-carriers (suspected lower)."},
        {"id": "h3_alb", "text": "Higher albumin_g_dl is associated with longer pfs_months (positive correlation)."},
        {"id": "h3_wl", "text": "Higher weight_loss_pct_6mo is associated with shorter pfs_months (negative correlation)."},
        {"id": "h3_ki67", "text": "Higher ki67_pct is associated with shorter pfs_months (negative correlation)."},
        {"id": "h3_node", "text": "Node-positive disease is associated with shorter pfs_months than node-negative."},
        {"id": "h3_postmeno", "text": "Postmenopausal status is associated with different pfs_months than premenopausal."},
        {"id": "h3_sex", "text": "Female sex is associated with different pfs_months than male sex."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h3_brca1"], "code": "stats.ttest_ind(by brca1_mutation)",
         "result_summary": "Mean PFS brca1=1: 4.67 vs brca1=0: 4.69; diff -0.02 mo, p=0.72 (NS).",
         "p_value": 0.72, "effect_estimate": -0.024, "significant": False},
        {"hypothesis_ids": ["h3_brca2"], "code": "stats.ttest_ind(by brca2_mutation)",
         "result_summary": "Mean PFS brca2=1: 4.57 vs brca2=0: 4.70; diff -0.13 mo, p=0.081 (borderline).",
         "p_value": 0.081, "effect_estimate": -0.127, "significant": False},
        {"hypothesis_ids": ["h3_alb"], "code": "stats.pearsonr(albumin_g_dl, pfs_months)",
         "result_summary": "Pearson r=+0.100 (p=9.4e-112).",
         "p_value": 9.4e-112, "effect_estimate": 0.100, "significant": True},
        {"hypothesis_ids": ["h3_wl"], "code": "stats.pearsonr(weight_loss_pct_6mo, pfs_months)",
         "result_summary": "Pearson r=-0.116 (p=4.1e-148).",
         "p_value": 4.1e-148, "effect_estimate": -0.116, "significant": True},
        {"hypothesis_ids": ["h3_ki67"], "code": "stats.pearsonr(ki67_pct, pfs_months)",
         "result_summary": "Pearson r=-0.091 (p=3.0e-93).",
         "p_value": 3.0e-93, "effect_estimate": -0.091, "significant": True},
        {"hypothesis_ids": ["h3_node"], "code": "stats.ttest_ind(by node_positive)",
         "result_summary": "Mean PFS node+: 4.68 vs node-: 4.70; diff -0.02 mo, p=0.29 (NS).",
         "p_value": 0.29, "effect_estimate": -0.024, "significant": False},
        {"hypothesis_ids": ["h3_postmeno"], "code": "stats.ttest_ind(by postmenopausal)",
         "result_summary": "Mean PFS postmeno: 4.68 vs premeno: 4.71; diff -0.02 mo, p=0.31 (NS).",
         "p_value": 0.31, "effect_estimate": -0.023, "significant": False},
        {"hypothesis_ids": ["h3_sex"], "code": "stats.ttest_ind(by sex_female)",
         "result_summary": "Mean PFS female: 4.69 vs male: 4.70; diff -0.01 mo, p=0.56 (NS).",
         "p_value": 0.56, "effect_estimate": -0.013, "significant": False},
    ]
})

# --------- Iteration 4 ---------
transcript["iterations"].append({
    "index": 4,
    "proposed_hypotheses": [
        {"id": "h4_palbo", "text": "Patients on treatment_palbociclib=1 have longer pfs_months than those off palbociclib (positive main effect)."},
        {"id": "h4_tam", "text": "Patients on treatment_tamoxifen=1 have longer pfs_months than those off tamoxifen."},
        {"id": "h4_tras", "text": "Patients on treatment_trastuzumab=1 have longer pfs_months than those off trastuzumab."},
        {"id": "h4_ola", "text": "Patients on treatment_olaparib=1 have longer pfs_months than those off olaparib."},
        {"id": "h4_saci", "text": "Patients on treatment_sacituzumab_govitecan=1 have longer pfs_months than those off."},
        {"id": "h4_pembro", "text": "Patients on treatment_pembrolizumab=1 have longer pfs_months than those off pembrolizumab."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h4_palbo"], "code": "stats.ttest_ind(by treatment_palbociclib)",
         "result_summary": "Mean PFS palbo=1: 5.41 vs palbo=0: 4.31; difference +1.10 mo (p<<1e-300). Strong positive main effect.",
         "p_value": 0.0, "effect_estimate": 1.098, "significant": True},
        {"hypothesis_ids": ["h4_tam"], "code": "stats.ttest_ind(by treatment_tamoxifen)",
         "result_summary": "Mean PFS tam=1: 4.72 vs tam=0: 4.68; difference +0.04 mo (p=0.077, NS).",
         "p_value": 0.077, "effect_estimate": 0.043, "significant": False},
        {"hypothesis_ids": ["h4_tras"], "code": "stats.ttest_ind(by treatment_trastuzumab)",
         "result_summary": "Mean PFS tras=1: 4.64 vs tras=0: 4.71; difference -0.06 mo (p=0.029, slightly negative overall).",
         "p_value": 0.029, "effect_estimate": -0.060, "significant": True},
        {"hypothesis_ids": ["h4_ola"], "code": "stats.ttest_ind(by treatment_olaparib)",
         "result_summary": "Mean PFS ola=1: 4.66 vs ola=0: 4.70; difference -0.04 mo (p=0.33, NS).",
         "p_value": 0.33, "effect_estimate": -0.036, "significant": False},
        {"hypothesis_ids": ["h4_saci"], "code": "stats.ttest_ind(by treatment_sacituzumab_govitecan)",
         "result_summary": "Mean PFS saci=1: 4.69 vs saci=0: 4.69; difference -0.01 mo (p=0.85, NS).",
         "p_value": 0.85, "effect_estimate": -0.007, "significant": False},
        {"hypothesis_ids": ["h4_pembro"], "code": "stats.ttest_ind(by treatment_pembrolizumab)",
         "result_summary": "Mean PFS pembro=1: 4.63 vs pembro=0: 4.70; difference -0.07 mo (p=0.018, slightly negative overall).",
         "p_value": 0.018, "effect_estimate": -0.074, "significant": True},
    ]
})

# --------- Iteration 5 ---------
transcript["iterations"].append({
    "index": 5,
    "proposed_hypotheses": [
        {"id": "h5_tam_er", "text": "Treatment_tamoxifen prolongs pfs_months only in ER-positive patients (positive interaction with er_positive)."},
        {"id": "h5_tam_pm", "text": "Among ER-positive patients, treatment_tamoxifen prolongs pfs_months more in postmenopausal than in premenopausal patients."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h5_tam_er"], "code": "stratified t-tests by er_positive",
         "result_summary": "Within ER+: mean PFS tam=1 4.78 vs tam=0 4.81; diff +0.02 mo, p=0.45. Within ER-: tam=1 4.59 vs tam=0 4.51; diff +0.08 mo, p=0.030. Tamoxifen does not show meaningful benefit in ER+ subgroup; small positive in ER-. Hypothesis refuted.",
         "p_value": 0.45, "effect_estimate": 0.023, "significant": False},
        {"hypothesis_ids": ["h5_tam_pm"], "code": "stratified t-tests within ER+ by postmenopausal",
         "result_summary": "Within ER+/postmeno: tam diff +0.01 mo, p=0.81. Within ER+/premeno: tam diff +0.04 mo, p=0.38. No menopausal modification of tamoxifen effect within ER+; both effectively null. Hypothesis refuted.",
         "p_value": 0.81, "effect_estimate": 0.010, "significant": False},
    ]
})

# --------- Iteration 6 ---------
transcript["iterations"].append({
    "index": 6,
    "proposed_hypotheses": [
        {"id": "h6_palbo_er", "text": "Treatment_palbociclib prolongs pfs_months substantially more in ER-positive patients than in ER-negative patients (positive palbo*ER interaction)."},
        {"id": "h6_palbo_her2", "text": "Treatment_palbociclib's PFS benefit is reduced or abolished in HER2-positive patients (negative palbo*HER2 interaction)."},
        {"id": "h6_palbo_pr", "text": "Treatment_palbociclib's PFS benefit is larger in PR-positive patients (positive palbo*PR interaction)."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h6_palbo_er"], "code": "smf.ols('pfs_months ~ treatment_palbociclib*er_positive', df)",
         "result_summary": "Stratified: ER+ palbo diff +1.58 mo (p<<1e-300); ER- palbo diff -0.02 mo (p=0.63). OLS interaction palbo:er_positive = +1.594, p=9.3e-229.",
         "p_value": 9.3e-229, "effect_estimate": 1.594, "significant": True},
        {"hypothesis_ids": ["h6_palbo_her2"], "code": "smf.ols('pfs_months ~ treatment_palbociclib*her2_positive', df)",
         "result_summary": "Stratified: HER2- palbo diff +1.35 mo (p<<1e-300); HER2+ palbo diff -0.01 mo (p=0.77). OLS interaction palbo:her2_positive = -1.364, p=2.2e-118.",
         "p_value": 2.2e-118, "effect_estimate": -1.364, "significant": True},
        {"hypothesis_ids": ["h6_palbo_pr"], "code": "smf.ols('pfs_months ~ treatment_palbociclib*pr_positive', df)",
         "result_summary": "OLS interaction palbo:pr_positive = +1.093, p=2.6e-117. PR+ patients gain greater palbo benefit.",
         "p_value": 2.6e-117, "effect_estimate": 1.093, "significant": True},
    ]
})

# --------- Iteration 7 ---------
transcript["iterations"].append({
    "index": 7,
    "proposed_hypotheses": [
        {"id": "h7_palbo_pik", "text": "Treatment_palbociclib's PFS benefit is reduced or abolished in PIK3CA-mutated patients (negative palbo*PIK3CA interaction)."},
        {"id": "h7_palbo_pm", "text": "Treatment_palbociclib's effect is similar in pre- vs post-menopausal patients (no large interaction with postmenopausal)."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h7_palbo_pik"], "code": "smf.ols('pfs_months ~ treatment_palbociclib*pik3ca_mutation', df) and stratified",
         "result_summary": "Stratified: PIK3CA-WT palbo diff +1.66 mo (p<<1e-300); PIK3CA-mut palbo diff +0.04 mo (p=0.29). Interaction palbo:pik3ca_mutation = -1.623, p=4.0e-255.",
         "p_value": 4.0e-255, "effect_estimate": -1.623, "significant": True},
        {"hypothesis_ids": ["h7_palbo_pm"], "code": "stratified by postmenopausal within ER+",
         "result_summary": "ER+/postmeno: palbo diff +1.50 mo (p=4e-283); ER+/premeno: palbo diff +1.69 mo (p=5e-230). Interaction palbo:postmenopausal = -0.13, p=0.004 (small attenuation in postmeno but benefit remains huge in both).",
         "p_value": 0.004, "effect_estimate": -0.134, "significant": True},
    ]
})

# --------- Iteration 8 ---------
transcript["iterations"].append({
    "index": 8,
    "proposed_hypotheses": [
        {"id": "h8_palbo_er_pik", "text": "Within ER-positive patients, treatment_palbociclib's PFS benefit is concentrated in PIK3CA wild-type patients and is abolished in PIK3CA-mutated patients (significant palbo*ER+*PIK3CA three-way interaction)."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h8_palbo_er_pik"], "code": "stratified 4 cells: ER x PIK3CA",
         "result_summary": "ER+/PIK3CA-WT: palbo diff +2.38 mo (p<<1e-300, n=8034 vs 14754). ER+/PIK3CA-mut: palbo diff +0.04 mo (p=0.39). ER-/PIK3CA-WT: -0.05 (p=0.28). ER-/PIK3CA-mut: +0.04 (p=0.52). Three-way OLS palbo:er_positive:pik3ca_mutation = -2.40, p<<1e-300.",
         "p_value": 0.0, "effect_estimate": -2.403, "significant": True},
    ]
})

# --------- Iteration 9 ---------
transcript["iterations"].append({
    "index": 9,
    "proposed_hypotheses": [
        {"id": "h9_palbo_full", "text": "Treatment_palbociclib prolongs pfs_months only in patients who are simultaneously ER-positive AND HER2-negative AND PIK3CA wild-type; it has no effect outside this subgroup."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h9_palbo_full"], "code": "stratified 8 cells: ER x HER2 x PIK3CA",
         "result_summary": "ER+/HER2-/PIK3CA-WT (n=18681): palbo diff +2.91 mo (p<<1e-300, on=6571 vs off=12110). ER+/HER2-/PIK3CA-mut: diff +0.02 (p=0.62). ER+/HER2+/PIK3CA-WT: diff -0.02 (p=0.81). All other cells diff between -0.13 and +0.09 (NS). Outside the eligible subgroup overall: palbo diff +0.003 mo (p=0.91, n=10857 vs 20462). Adjusted interaction palbo:eligible = +2.92, p<<1e-300.",
         "p_value": 0.0, "effect_estimate": 2.915, "significant": True},
    ]
})

# --------- Iteration 10 ---------
transcript["iterations"].append({
    "index": 10,
    "proposed_hypotheses": [
        {"id": "h10_palbo_ki67", "text": "Within the ER+/HER2-/PIK3CA-WT subgroup, the magnitude of the treatment_palbociclib PFS benefit decreases as ki67_pct increases (negative palbo*ki67 interaction within the subgroup)."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h10_palbo_ki67"], "code": "ols('pfs_months ~ treatment_palbociclib*ki67_pct', subset)",
         "result_summary": "Within ER+/HER2-/PIK3CA-WT (n=18681), OLS interaction palbo:ki67_pct = -0.135 mo per percentage-point ki67 (p<<1e-300). Implied palbo benefit: ~+5.0 mo at ki67=0, ~+3.0 mo at ki67=15, ~+1.0 mo at ki67=30. The whole-cohort palbo:ki67 interaction is also strongly negative (-0.052, p=1.5e-174).",
         "p_value": 0.0, "effect_estimate": -0.135, "significant": True},
    ]
})

# --------- Iteration 11 ---------
transcript["iterations"].append({
    "index": 11,
    "proposed_hypotheses": [
        {"id": "h11_tras_her2", "text": "Treatment_trastuzumab prolongs pfs_months in HER2-positive patients (positive trastuzumab*HER2 interaction)."},
        {"id": "h11_tras_adj", "text": "After adjustment for clinical and biomarker covariates, treatment_trastuzumab has a clinically meaningful positive main effect on pfs_months."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h11_tras_her2"], "code": "stratified by her2_positive",
         "result_summary": "HER2+: tras diff -0.08 mo (p=0.15, n=1890 vs 7170). HER2-: tras diff -0.05 mo (p=0.10). OLS interaction tras:her2_positive = -0.028, p=0.69. Trastuzumab does NOT prolong PFS in HER2+ patients in this dataset. Hypothesis refuted.",
         "p_value": 0.69, "effect_estimate": -0.028, "significant": False},
        {"hypothesis_ids": ["h11_tras_adj"], "code": "joint OLS with covariates",
         "result_summary": "Adjusted joint model: trastuzumab beta = -0.010, p=0.45 (NS). Within HER2+ alone after adjustment: beta=-0.000, p=0.95. Robust null effect; hypothesis refuted.",
         "p_value": 0.95, "effect_estimate": -0.010, "significant": False},
    ]
})

# --------- Iteration 12 ---------
transcript["iterations"].append({
    "index": 12,
    "proposed_hypotheses": [
        {"id": "h12_ola_brca", "text": "Treatment_olaparib prolongs pfs_months in patients with any BRCA1 or BRCA2 mutation (positive interaction)."},
        {"id": "h12_ola_brca1_only", "text": "Treatment_olaparib prolongs pfs_months specifically in BRCA1-mutated patients (positive interaction)."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h12_ola_brca"], "code": "smf.ols('pfs_months ~ treatment_olaparib*any_brca', df) and stratified",
         "result_summary": "anyBRCA+: ola diff +0.35 mo (p=0.041, n=239 vs 2260). anyBRCA-: ola diff -0.06 mo (p=0.14). OLS interaction olaparib:any_brca = +0.402, p=0.021.",
         "p_value": 0.021, "effect_estimate": 0.402, "significant": True},
        {"hypothesis_ids": ["h12_ola_brca1_only"], "code": "stratified by brca1_mutation",
         "result_summary": "BRCA1+: ola diff +0.40 mo (p=0.12, n=120). BRCA1-: ola diff -0.05 (p=0.21). OLS interaction olaparib:brca1_mutation = +0.446, p=0.066. Suggestive but underpowered.",
         "p_value": 0.066, "effect_estimate": 0.446, "significant": False},
    ]
})

# --------- Iteration 13 ---------
transcript["iterations"].append({
    "index": 13,
    "proposed_hypotheses": [
        {"id": "h13_ola_hr_brca", "text": "Treatment_olaparib's PFS benefit is concentrated in patients who are simultaneously hormone-receptor-negative (er_positive=0 AND pr_positive=0) AND have any BRCA1 or BRCA2 mutation."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h13_ola_hr_brca"], "code": "stratified 4 cells: HR x any_brca",
         "result_summary": "HR-/anyBRCA+ (n=629): ola diff +0.83 mo (p=0.004, on=46 vs off=583). HR-/BRCA-: -0.04 (p=0.54). HR+/BRCA+: +0.19 (p=0.34). HR+/BRCA-: -0.07 (p=0.16). OLS interaction treatment_olaparib*HR-_AND_BRCA+ = +0.876, p=0.023.",
         "p_value": 0.023, "effect_estimate": 0.876, "significant": True},
    ]
})

# --------- Iteration 14 ---------
transcript["iterations"].append({
    "index": 14,
    "proposed_hypotheses": [
        {"id": "h14_saci_tnbc", "text": "Treatment_sacituzumab_govitecan prolongs pfs_months in TNBC (er_negative AND pr_negative AND her2_negative) patients (positive saci*TNBC interaction)."},
        {"id": "h14_saci_her2low", "text": "Treatment_sacituzumab_govitecan prolongs pfs_months in HER2-low (her2_low=1) patients (positive saci*her2_low interaction)."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h14_saci_tnbc"], "code": "stratified by TNBC",
         "result_summary": "TNBC: saci diff -0.09 mo (p=0.22, n=1084 vs 9412). non-TNBC: saci diff +0.02 mo (p=0.66). No benefit in TNBC; hypothesis refuted.",
         "p_value": 0.22, "effect_estimate": -0.085, "significant": False},
        {"hypothesis_ids": ["h14_saci_her2low"], "code": "stratified by her2_low",
         "result_summary": "HER2-low: saci diff -0.03 mo (p=0.64). OLS interaction saci:her2_low = -0.038, p=0.63. No HER2-low benefit; hypothesis refuted.",
         "p_value": 0.63, "effect_estimate": -0.031, "significant": False},
    ]
})

# --------- Iteration 15 ---------
transcript["iterations"].append({
    "index": 15,
    "proposed_hypotheses": [
        {"id": "h15_pembro_tnbc", "text": "Treatment_pembrolizumab prolongs pfs_months in TNBC patients (positive pembro*TNBC interaction) and is neutral or harmful in non-TNBC patients."},
        {"id": "h15_pembro_burden", "text": "Within TNBC, the treatment_pembrolizumab effect on pfs_months is larger when high inflammatory or proliferative-burden surrogates (high ki67_pct, high ldh_u_l, or high crp_mg_l) are present."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h15_pembro_tnbc"], "code": "smf.ols('pfs_months ~ treatment_pembrolizumab*tnbc', df) + stratified",
         "result_summary": "TNBC: pembro diff +0.07 mo (p=0.22). non-TNBC: pembro diff -0.11 mo (p=0.0017). OLS interaction pembro:tnbc = +0.185, p=0.015. Adjusted: pembro main = -0.046 (p=0.012), pembro:tnbc = +0.068 (p=0.091). The interaction is real but small; pembro slightly harms non-TNBC and is neutral-to-slight-positive in TNBC.",
         "p_value": 0.015, "effect_estimate": 0.185, "significant": True},
        {"hypothesis_ids": ["h15_pembro_burden"], "code": "stratified within TNBC by ki67/ldh/crp/nlr cutoffs",
         "result_summary": "Within TNBC, none of ki67>=20, ldh>=300, crp>=10, nlr>=4 yielded a significant pembro effect (all p>0.18). No clean inflammatory-modifier subgroup within TNBC; hypothesis refuted.",
         "p_value": 0.18, "effect_estimate": -0.07, "significant": False},
    ]
})

# --------- Iteration 16 ---------
transcript["iterations"].append({
    "index": 16,
    "proposed_hypotheses": [
        {"id": "h16_screen", "text": "Across all six treatments, an exhaustive single-variable interaction screen (treatment x each of 31 features) on pfs_months will show that strong interaction effects (|effect|>0.3 mo and p<1e-10) are specific to treatment_palbociclib*{er_positive, pr_positive, her2_positive, pik3ca_mutation, ki67_pct, her2_low}, and that no other treatment-feature pair will reach this stringent threshold."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h16_screen"], "code": "smf.ols(pfs_months ~ tx*feature) for every treatment x feature",
         "result_summary": "Strongest interactions across the 31x6 grid: palbo:pik3ca -1.62 (p=4e-255); palbo:er +1.59 (p=9e-229); palbo:ki67 -0.052 (p=1.5e-174); palbo:her2 -1.36 (p=2e-118); palbo:pr +1.09 (p=3e-117); palbo:her2_low +0.42 (p=9e-19). For all other treatments the smallest interaction p-values were >=0.005 (tras:tumor_size_cm p=0.018; ola:creatinine p=0.005; saci:albumin p=0.013; pembro:er p=0.050). Hypothesis confirmed.",
         "p_value": 4e-255, "effect_estimate": -1.623, "significant": True},
    ]
})

# --------- Iteration 17 ---------
transcript["iterations"].append({
    "index": 17,
    "proposed_hypotheses": [
        {"id": "h17_joint", "text": "In a multivariable OLS model adjusted for age_years, ecog_ps, stage_iv, has_brain_mets, albumin_g_dl, weight_loss_pct_6mo, ki67_pct, er_positive, pr_positive, her2_positive, her2_low, brca1_mutation, brca2_mutation, pik3ca_mutation, and postmenopausal (and including all six treatments), only treatment_palbociclib will retain a significant adjusted main effect on pfs_months."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h17_joint"], "code": "smf.ols joint model with all treatments and covariates",
         "result_summary": "Adjusted main effects (months): palbociclib +1.10 (p<<1e-300, sig); tamoxifen -0.008 (p=0.50); trastuzumab -0.010 (p=0.45); olaparib -0.026 (p=0.13); sacituzumab +0.033 (p=0.05, borderline); pembrolizumab -0.020 (p=0.15). Only palbociclib retains a clinically and statistically meaningful effect.",
         "p_value": 0.0, "effect_estimate": 1.102, "significant": True},
    ]
})

# --------- Iteration 18 ---------
transcript["iterations"].append({
    "index": 18,
    "proposed_hypotheses": [
        {"id": "h18_2way_palbo", "text": "An exhaustive two-way binary subgroup search for treatment_palbociclib will identify the largest signed effect within ER+ AND PIK3CA-WT (with HER2- selection further refining the cell)."},
        {"id": "h18_2way_others", "text": "An exhaustive two-way binary subgroup search for tamoxifen, trastuzumab, sacituzumab, and pembrolizumab will not reveal any 2-way subgroup with |diff|>0.5 mo at p<0.01."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h18_2way_palbo"], "code": "exhaustive 2-way binary subgroup loop for treatment_palbociclib",
         "result_summary": "Palbo: every cell within ER+/HER2-/PIK3CA-WT shows diff > +2 mo (e.g., ER+&PIK0: +2.38; ER+&HER2- broader: +1.50; the triple cell ER+/HER2-/PIK3CA-WT: +2.91). All cells outside this combination show |diff|<0.13 mo. Hypothesis confirmed.",
         "p_value": 0.0, "effect_estimate": 2.908, "significant": True},
        {"hypothesis_ids": ["h18_2way_others"], "code": "exhaustive 2-way binary subgroup loop",
         "result_summary": "Top significant 2-way subgroups: olaparib in HR-/BRCA+ +0.88 (p=0.004); olaparib in PR-/BRCA1+ +1.02 (p=0.034, n=34); pembrolizumab in BRCA1+ subgroups -0.4 to -0.7 (p~0.01-0.04, n~90-130). Sacituzumab, tamoxifen, and trastuzumab show no 2-way subgroup with |diff|>0.5 at robust p. The olaparib HR-/BRCA+ signal is the only secondary positive finding; hypothesis partially refuted by this olaparib signal.",
         "p_value": 0.004, "effect_estimate": 0.829, "significant": True},
    ]
})

# --------- Iteration 19 ---------
transcript["iterations"].append({
    "index": 19,
    "proposed_hypotheses": [
        {"id": "h19_pembro_brca", "text": "Treatment_pembrolizumab is associated with shorter pfs_months specifically in BRCA1-mutation carriers (negative pembro*BRCA1 interaction)."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h19_pembro_brca"], "code": "smf.ols('pfs_months ~ treatment_pembrolizumab*brca1_mutation', df) and adjusted",
         "result_summary": "Unadjusted interaction pembro:brca1_mutation = -0.188, p=0.34 (NS). Adjusted: -0.004, p=0.97 (NS). The pembro decrement seen in BRCA1+ 2-way subgroups appears driven by chance/multiple testing. Hypothesis refuted.",
         "p_value": 0.97, "effect_estimate": 0.004, "significant": False},
    ]
})

# --------- Iteration 20 ---------
transcript["iterations"].append({
    "index": 20,
    "proposed_hypotheses": [
        {"id": "h20_age_strong", "text": "After adjusting for stage_iv, has_brain_mets, and ecog_ps, age_years remains a strong positive predictor of pfs_months at greater than +0.1 month per year of age."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h20_age_strong"], "code": "smf.ols('pfs_months ~ age_years + stage_iv + has_brain_mets + ecog_ps', df)",
         "result_summary": "Adjusted age beta = +0.176 mo per year (p<<1e-300). Stage IV beta -1.55, brain mets -0.99, ECOG -1.17. Age is the dominant prognostic feature in this dataset.",
         "p_value": 0.0, "effect_estimate": 0.176, "significant": True},
    ]
})

# --------- Iteration 21 ---------
transcript["iterations"].append({
    "index": 21,
    "proposed_hypotheses": [
        {"id": "h21_palbo_final", "text": "FINAL: treatment_palbociclib improves pfs_months exclusively in patients who are simultaneously er_positive=1 AND her2_positive=0 AND pik3ca_mutation=0. The within-subgroup effect is approximately +2.9 months and the magnitude declines with rising ki67_pct (approximately -0.14 mo per percentage-point ki67 within the subgroup). Outside this subgroup the palbociclib effect on pfs_months is indistinguishable from zero."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h21_palbo_final"], "code": "subgroup test + adjusted interaction",
         "result_summary": "Within ER+/HER2-/PIK3CA-WT (n=18681, on=6571 vs off=12110): palbo diff = +2.91 mo (p<<1e-300). Outside (n=31319, on=10857 vs off=20462): palbo diff = +0.003 mo (p=0.91). Adjusted interaction palbo:eligible = +2.92, p<<1e-300. Within-subgroup palbo*ki67 = -0.135 mo per pp ki67 (p<<1e-300). Hypothesis fully supported.",
         "p_value": 0.0, "effect_estimate": 2.915, "significant": True},
    ]
})

# --------- Iteration 22 ---------
transcript["iterations"].append({
    "index": 22,
    "proposed_hypotheses": [
        {"id": "h22_ola_final", "text": "FINAL: treatment_olaparib improves pfs_months mainly in patients who are simultaneously hormone-receptor-negative (er_positive=0 AND pr_positive=0) AND carry any BRCA1 or BRCA2 mutation; the within-subgroup effect is approximately +0.83 months. Outside this combination the olaparib effect is null or trends slightly negative."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h22_ola_final"], "code": "subgroup test + interaction with target indicator",
         "result_summary": "HR-/anyBRCA+ (n=629, on=46 vs off=583): ola diff +0.83 mo (p=0.0038). HR-/BRCA-: -0.04 (p=0.54). HR+/BRCA+: +0.19 (p=0.34). HR+/BRCA-: -0.07 (p=0.16). OLS interaction olaparib*HR-_AND_BRCA+ = +0.876, p=0.023. Effect concentrated in this subgroup but supported by modest n; broader anyBRCA+ subgroup also shows +0.35 (p=0.041).",
         "p_value": 0.023, "effect_estimate": 0.876, "significant": True},
    ]
})

# --------- Iteration 23 ---------
transcript["iterations"].append({
    "index": 23,
    "proposed_hypotheses": [
        {"id": "h23_tras_final", "text": "FINAL: treatment_trastuzumab does NOT prolong pfs_months in any clinically defined subgroup, including HER2-positive patients."},
        {"id": "h23_saci_final", "text": "FINAL: treatment_sacituzumab_govitecan does NOT prolong pfs_months in any clinically defined subgroup, including TNBC and HER2-low patients."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h23_tras_final"], "code": "stratified by HER2/HR/stage/postmeno; adjusted",
         "result_summary": "HER2+ alone: diff -0.08 (p=0.15). HER2+ adjusted joint model: beta=-0.000 (p=0.95). HER2+/HR+/postmeno: -0.13 (p=0.12). All deeper subgroups null. Hypothesis supported (no positive subgroup).",
         "p_value": 0.95, "effect_estimate": -0.060, "significant": False},
        {"hypothesis_ids": ["h23_saci_final"], "code": "stratified TNBC, HER2-low, HR-/HER2-/stage_iv, brain_mets, etc.",
         "result_summary": "All sacituzumab subgroup effects are between -0.26 and +0.36 mo; only one cell (HR-/HER2-/stage_iv: -0.26, p=0.018) borderline negative; no plausible benefit subgroup. Hypothesis supported.",
         "p_value": 0.018, "effect_estimate": -0.262, "significant": True},
    ]
})

# --------- Iteration 24 ---------
transcript["iterations"].append({
    "index": 24,
    "proposed_hypotheses": [
        {"id": "h24_pembro_final", "text": "FINAL: treatment_pembrolizumab provides at most a small (~+0.07-0.18 mo) PFS benefit limited to TNBC patients (er_positive=0 AND pr_positive=0 AND her2_positive=0). In non-TNBC patients pembrolizumab is associated with a small (~-0.1 mo) decrement in pfs_months."},
        {"id": "h24_tam_final", "text": "FINAL: treatment_tamoxifen does not prolong pfs_months in any clinically defined subgroup; effects are uniformly small (|diff|<0.12 mo) and do not survive multiple-testing correction."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h24_pembro_final"], "code": "stratified TNBC + adjusted interaction",
         "result_summary": "TNBC: pembro diff +0.07 mo (p=0.22). non-TNBC: -0.11 mo (p=0.0017). OLS interaction pembro:tnbc = +0.185 (p=0.015). Adjusted: main = -0.046 (p=0.012), interaction = +0.068 (p=0.091). Hypothesis supported: small positive in TNBC, small negative in non-TNBC.",
         "p_value": 0.015, "effect_estimate": 0.185, "significant": True},
        {"hypothesis_ids": ["h24_tam_final"], "code": "stratified ER x postmeno, ECOG, stage; exhaustive 2-way",
         "result_summary": "Largest tamoxifen subgroup: ER-/premeno +0.11 mo (p=0.066). ER+/postmeno: +0.01 (p=0.81). All adjusted main and interaction effects |beta|<0.06 with p>0.05. Hypothesis supported.",
         "p_value": 0.81, "effect_estimate": 0.010, "significant": False},
    ]
})

# --------- Iteration 25 ---------
transcript["iterations"].append({
    "index": 25,
    "proposed_hypotheses": [
        {"id": "h25_summary", "text": "FINAL SUMMARY: The treatment-effect subgroup map for ds001_breast is: (1) treatment_palbociclib +2.9 mo PFS in er_positive=1 AND her2_positive=0 AND pik3ca_mutation=0 (modulated by ki67_pct); (2) treatment_olaparib +0.8 mo PFS in er_positive=0 AND pr_positive=0 AND (brca1_mutation=1 OR brca2_mutation=1); (3) treatment_pembrolizumab small +0.07-0.18 mo PFS interaction with TNBC; (4) treatment_tamoxifen, treatment_trastuzumab, and treatment_sacituzumab_govitecan show no clinically meaningful subgroup benefit. Strongest non-treatment prognostic factors: age_years (+0.18 mo/yr), ecog_ps (-1.17/level), stage_iv (-1.55), has_brain_mets (-0.99), albumin_g_dl (positive), weight_loss_pct_6mo (negative), ki67_pct (negative)."},
    ],
    "analyses": [
        {"hypothesis_ids": ["h25_summary"], "code": "consolidation of prior iterations",
         "result_summary": "Final subgroup hypotheses fully tested in iterations 9, 13, 15, 21, 22, 23, 24. Palbociclib: ER+/HER2-/PIK3CA-WT in-subgroup diff +2.91 mo (p<<1e-300); out-of-subgroup +0.003 (p=0.91). Olaparib: HR-/anyBRCA+ in-subgroup diff +0.83 (p=0.0038). Pembrolizumab: TNBC subgroup interaction +0.185 (p=0.015). Other treatments: no positive subgroup.",
         "p_value": 0.0, "effect_estimate": 2.915, "significant": True},
    ]
})

with open('transcript.json','w') as fp:
    json.dump(transcript, fp, indent=2)
print('Wrote transcript.json with', len(transcript['iterations']), 'iterations')

summary = """ANALYSIS SUMMARY -- ds001_breast (50,000 patients)
===================================================

Outcome: pfs_months (progression-free survival in months; mean 4.69, range 0.0-16.66).

OVERVIEW OF APPROACH
--------------------
Across 25 iterations I worked from main effects to multivariable adjusted
interactions to systematic two- and three-way subgroup discovery. Treatments
considered: tamoxifen, palbociclib, trastuzumab, olaparib, sacituzumab
govitecan, and pembrolizumab. For every treatment with usable variation I
ran a single-feature interaction screen (treatment x each clinical /
biomarker / lab feature) and an exhaustive two-way binary-subgroup search;
for the strongest signals I confirmed with three-way subgroup definitions
and adjusted models that included age_years, ecog_ps, stage_iv,
has_brain_mets, albumin_g_dl, weight_loss_pct_6mo, ki67_pct,
hormone-receptor markers, HER2 status, BRCA1/2, PIK3CA, and postmenopausal
status.

PROGNOSTIC (NON-TREATMENT) FINDINGS
-----------------------------------
* age_years: r=+0.70 with PFS unadjusted; +0.176 mo/year adjusted for
  stage/brain mets/ECOG (p<<1e-300). The strongest single predictor in this
  cohort.
* ecog_ps: PFS 5.64/4.44/3.29 mo at ECOG 0/1/2; adjusted -1.17 mo per level.
* stage_iv: -1.54 mo unadjusted, -1.55 adjusted.
* has_brain_mets: -1.05 mo unadjusted, -0.99 adjusted.
* er_positive: +0.57 mo; pr_positive: +0.36 mo; her2_positive: -0.46 mo.
* pik3ca_mutation: -0.56 mo; her2_low: +0.11 mo (small but highly sig).
* albumin_g_dl: r=+0.10 (positive); weight_loss_pct_6mo: r=-0.12; ki67_pct:
  r=-0.09. Other labs (ldh, crp, nlr, hemoglobin, alkaline phosphatase, AST,
  ALT, bilirubin, creatinine, BUN, sodium, potassium, calcium) were either
  null or had only nominal correlations (|r|<0.02).
* node_positive, postmenopausal, and sex_female are not associated with PFS
  at the marginal level in this cohort.
* brca1_mutation and brca2_mutation are not strong main-effect prognostic
  markers (NS or only borderline at p=0.08).

TREATMENT MAIN EFFECTS
----------------------
* treatment_palbociclib: +1.10 mo unadjusted (p<<1e-300); +1.10 adjusted.
  The only treatment with a clinically meaningful adjusted main effect.
* treatment_tamoxifen: +0.04 mo (p=0.077); adjusted -0.01 (NS).
* treatment_trastuzumab: -0.06 mo (p=0.029); adjusted -0.01 (NS). Robust
  null even within HER2+ patients.
* treatment_olaparib: -0.04 mo (p=0.33); adjusted -0.03 (NS).
* treatment_sacituzumab_govitecan: -0.01 (p=0.85); adjusted +0.03 (p=0.05,
  borderline but tiny magnitude).
* treatment_pembrolizumab: -0.07 mo (p=0.018); adjusted -0.02 (NS).

TREATMENT-EFFECT HETEROGENEITY (KEY FINDINGS)
---------------------------------------------

(1) PALBOCICLIB -- strong, well-defined subgroup
    * The benefit is concentrated in patients who are ER-positive AND
      HER2-negative AND PIK3CA wild-type.
    * Within this subgroup (n=18,681; on palbo n=6,571 vs off n=12,110):
      mean PFS 7.21 vs 4.30 mo, diff = +2.91 mo (p<<1e-300).
    * Outside this subgroup (n=31,319): palbo diff = +0.003 mo (p=0.91).
    * Adjusted interaction palbo x eligible_subgroup = +2.92, p<<1e-300.
    * Three-way interaction palbo x er_positive x pik3ca_mutation = -2.40
      (p<<1e-300): the ER+ benefit is fully reversed by PIK3CA mutation.
    * Within ER+/PIK3CA-WT, palbo benefit is also fully suppressed by HER2+
      (interaction palbo x her2_positive within ER+/PIK3CA-WT = -2.93,
      p=3e-227).
    * Within the eligible (ER+/HER2-/PIK3CA-WT) subgroup, the magnitude of
      the palbociclib benefit decreases as ki67_pct rises:
      interaction palbo x ki67_pct = -0.135 mo per percentage-point ki67
      (p<<1e-300). Implied: ~+5.0 mo at ki67=0, ~+3.0 mo at ki67=15,
      ~+1.0 mo at ki67=30.
    FINAL palbociclib subgroup hypothesis:
      Treatment: treatment_palbociclib.
      Outcome: pfs_months (longer is better).
      Direction: positive.
      Subgroup predicates: er_positive=1 AND her2_positive=0 AND
                            pik3ca_mutation=0.
      Effect modifier within the subgroup: ki67_pct (smaller benefit at
                            higher ki67).

(2) OLAPARIB -- modest, refined subgroup
    * Stratified by hormone-receptor status (HR = er_positive=1 OR
      pr_positive=1) and any BRCA mutation (brca1_mutation=1 OR
      brca2_mutation=1):
        HR-/anyBRCA+ (n=629; on=46 vs off=583): diff = +0.83 mo (p=0.0038).
        HR-/BRCA-  (n=12,205): diff = -0.04 mo (p=0.54).
        HR+/BRCA+  (n=1,870):  diff = +0.19 mo (p=0.34).
        HR+/BRCA-  (n=35,296): diff = -0.07 mo (p=0.16).
    * Interaction olaparib x (HR- AND anyBRCA+) = +0.876 (p=0.023).
    * Broader interaction olaparib x anyBRCA+ = +0.402 (p=0.021); benefit
      is strongest when HR is also negative.
    * Within BRCA1+ alone: ola diff +0.40 (p=0.12); within BRCA2+ alone:
      +0.30 (p=0.19). Both individually underpowered but consistent in
      direction.
    FINAL olaparib subgroup hypothesis:
      Treatment: treatment_olaparib.
      Outcome: pfs_months (longer is better).
      Direction: positive.
      Subgroup predicates: er_positive=0 AND pr_positive=0 AND
                            (brca1_mutation=1 OR brca2_mutation=1).

(3) PEMBROLIZUMAB -- small TNBC interaction
    * TNBC = er_positive=0 AND pr_positive=0 AND her2_positive=0.
    * TNBC: pembro diff +0.07 mo (p=0.22). non-TNBC: -0.11 mo (p=0.0017).
    * OLS interaction pembro x TNBC = +0.185 (p=0.015).
    * Adjusted: pembro main effect = -0.046 (p=0.012), interaction +0.068
      (p=0.091).
    * No additional within-TNBC modifier (ki67, ldh, crp, nlr) was supported.
    * The signal is real but small in magnitude; outside TNBC, pembrolizumab
      is associated with a small decrement in PFS.
    FINAL pembrolizumab subgroup hypothesis:
      Treatment: treatment_pembrolizumab.
      Outcome: pfs_months (longer is better).
      Direction: weakly positive in TNBC; weakly negative outside TNBC.
      Subgroup predicates: er_positive=0 AND pr_positive=0 AND
                            her2_positive=0.

(4) TAMOXIFEN -- no subgroup
    * No clinically meaningful effect anywhere. Largest stratified estimate
      was ER-/premeno: +0.11 mo (p=0.066). ER+/postmeno: +0.01 (p=0.81).
      The biology-suggested subgroup (ER+ AND postmenopausal) shows no
      benefit.
    FINAL tamoxifen subgroup hypothesis:
      No clinically meaningful subgroup; treatment_tamoxifen has no positive
      effect on pfs_months in this cohort.

(5) TRASTUZUMAB -- no subgroup
    * Within HER2+ (n=9,060): diff -0.08 mo (p=0.15). Adjusted within HER2+:
      beta=-0.000 (p=0.95). HER2+/HR+/postmeno: -0.13 (p=0.12). All deeper
      subgroups null or borderline negative.
    FINAL trastuzumab subgroup hypothesis:
      No clinically meaningful subgroup; treatment_trastuzumab has no
      positive effect on pfs_months in this cohort, including HER2+ disease.

(6) SACITUZUMAB GOVITECAN -- no subgroup
    * TNBC: -0.09 (p=0.22). HER2-low: -0.03 (p=0.64). HER2-low/TNBC:
      -0.07 (p=0.49). HR-/HER2-/stage_iv: -0.26 (p=0.018) -- only nominally
      adverse. No subgroup with a positive effect.
    FINAL sacituzumab subgroup hypothesis:
      No clinically meaningful positive subgroup; treatment_sacituzumab_
      govitecan has no positive effect on pfs_months in this cohort,
      including TNBC and HER2-low disease.

OVERALL CONCLUSIONS
-------------------
Of the six treatments studied in ds001_breast, only TWO show
treatment-effect heterogeneity that is both statistically robust and
clinically meaningful:
* treatment_palbociclib confers a large (+2.9 mo) PFS benefit, but ONLY in
  patients who are ER+ AND HER2- AND PIK3CA-WT. The benefit is further
  modulated by ki67_pct (smaller as ki67 rises).
* treatment_olaparib confers a modest (+0.8 mo) PFS benefit, concentrated in
  patients who are HR- AND BRCA1/BRCA2-mutated.
A third treatment (treatment_pembrolizumab) shows a small TNBC-restricted
interaction (~+0.07-0.18 mo) that is statistically significant but
clinically modest. The remaining three treatments (tamoxifen, trastuzumab,
sacituzumab govitecan) show no clinically meaningful PFS subgroup benefit
despite many candidate subgroups being tested.

Strongest prognostic features (effect on pfs_months, adjusted): age_years
(+0.18 mo/year), ecog_ps (-1.17 per level), stage_iv (-1.55), has_brain_mets
(-0.99), pik3ca_mutation (-0.56), her2_positive (-0.46), and biological-
fitness markers albumin_g_dl (positive) and weight_loss_pct_6mo (negative).

Notable null findings (against textbook biology): trastuzumab in HER2+
patients shows zero benefit on PFS in this cohort; tamoxifen in ER+ shows
zero benefit; sacituzumab in TNBC and HER2-low shows zero benefit. These
are robust nulls supported by both unadjusted stratified analyses and
adjusted multivariable models.
"""

with open('analysis_summary.txt','w') as fp:
    fp.write(summary)
print('Wrote analysis_summary.txt', len(summary), 'chars')
