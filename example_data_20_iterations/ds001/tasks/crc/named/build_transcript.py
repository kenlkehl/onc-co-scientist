"""Build transcript.json from results.json with structured iterations and hypotheses."""
import json

R = json.load(open("results.json"))


def A(rkey, hids, code=""):
    """Build an analysis record from a results.json key."""
    r = R[rkey]
    return {
        "hypothesis_ids": hids,
        "code": code or None,
        "result_summary": r["result_summary"],
        "p_value": r["p_value"],
        "effect_estimate": r["effect_estimate"],
        "significant": r["significant"],
    }


def H(hid, text, kind="novel"):
    return {"id": hid, "text": text, "kind": kind}


iterations = []

# Iteration 1 — prognostic main effects
iterations.append({
    "index": 1,
    "proposed_hypotheses": [
        H("h1_stageiv", "Patients with stage_iv=1 have shorter mean pfs_months than patients with stage_iv=0."),
        H("h1_liver", "Patients with liver_mets=1 have shorter mean pfs_months than patients with liver_mets=0."),
        H("h1_bone", "Patients with bone_mets=1 have shorter mean pfs_months than patients with bone_mets=0."),
        H("h1_ecog", "Higher ecog_ps is associated with shorter pfs_months (negative beta in OLS)."),
    ],
    "analyses": [
        A("i1_stage_iv", ["h1_stageiv"], "stats.ttest_ind(df.loc[df.stage_iv==1,'pfs_months'], df.loc[df.stage_iv==0,'pfs_months'], equal_var=False)"),
        A("i1_liver_mets", ["h1_liver"], "stats.ttest_ind by liver_mets"),
        A("i1_bone_mets", ["h1_bone"], "stats.ttest_ind by bone_mets"),
        A("i1_ecog_ps", ["h1_ecog"], "OLS pfs_months ~ ecog_ps"),
    ],
})

# Iteration 2 — lab/biomarker prognostic
iterations.append({
    "index": 2,
    "proposed_hypotheses": [
        H("h2_alb", "Higher albumin_g_dl is associated with longer pfs_months (positive beta)."),
        H("h2_ldh", "Higher ldh_u_l is associated with shorter pfs_months (negative beta)."),
        H("h2_crp", "Higher crp_mg_l is associated with shorter pfs_months (negative beta)."),
        H("h2_nlr", "Higher nlr (neutrophil-to-lymphocyte ratio) is associated with shorter pfs_months."),
        H("h2_wl", "Higher weight_loss_pct_6mo is associated with shorter pfs_months (negative beta)."),
        H("h2_cea", "Higher cea_ng_ml is associated with shorter pfs_months (negative beta)."),
        H("h2_hgb", "Higher hemoglobin_g_dl is associated with longer pfs_months (positive beta)."),
    ],
    "analyses": [
        A("i2_albumin_g_dl", ["h2_alb"], "OLS pfs_months ~ albumin_g_dl"),
        A("i2_ldh_u_l", ["h2_ldh"], "OLS pfs_months ~ ldh_u_l"),
        A("i2_crp_mg_l", ["h2_crp"], "OLS pfs_months ~ crp_mg_l"),
        A("i2_nlr", ["h2_nlr"], "OLS pfs_months ~ nlr"),
        A("i2_weight_loss_pct_6mo", ["h2_wl"], "OLS pfs_months ~ weight_loss_pct_6mo"),
        A("i2_cea_ng_ml", ["h2_cea"], "OLS pfs_months ~ cea_ng_ml"),
        A("i2_hemoglobin_g_dl", ["h2_hgb"], "OLS pfs_months ~ hemoglobin_g_dl"),
    ],
})

# Iteration 3 — genomic biomarker main effects
iterations.append({
    "index": 3,
    "proposed_hypotheses": [
        H("h3_kras", "Patients with kras_mutation=1 have shorter pfs_months than kras_mutation=0 (negative diff)."),
        H("h3_nras", "Patients with nras_mutation=1 have shorter pfs_months than nras_mutation=0 (negative diff)."),
        H("h3_braf", "Patients with braf_v600e=1 have shorter pfs_months than braf_v600e=0 (negative diff)."),
        H("h3_msi", "Patients with msi_high=1 have longer pfs_months than msi_high=0 (positive diff)."),
        H("h3_her2", "Patients with her2_amplified=1 have a different mean pfs_months than her2_amplified=0."),
        H("h3_ntrk", "Patients with ntrk_fusion=1 have a different mean pfs_months than ntrk_fusion=0."),
        H("h3_right", "Patients with right_sided_primary=1 have shorter pfs_months than left-sided (negative diff)."),
    ],
    "analyses": [
        A("i3_kras_mutation", ["h3_kras"]),
        A("i3_nras_mutation", ["h3_nras"]),
        A("i3_braf_v600e", ["h3_braf"]),
        A("i3_msi_high", ["h3_msi"]),
        A("i3_her2_amplified", ["h3_her2"]),
        A("i3_ntrk_fusion", ["h3_ntrk"]),
        A("i3_right_sided_primary", ["h3_right"]),
    ],
})

# Iteration 4 — treatment main effects (unadjusted)
iterations.append({
    "index": 4,
    "proposed_hypotheses": [
        H("h4_cetux", "Patients on treatment_cetuximab=1 have different pfs_months than those off (raw mean comparison)."),
        H("h4_bev", "Patients on treatment_bevacizumab=1 have different pfs_months than those off."),
        H("h4_pembro", "Patients on treatment_pembrolizumab=1 have different pfs_months than those off."),
        H("h4_enc", "Patients on treatment_encorafenib=1 have different pfs_months than those off."),
        H("h4_tt", "Patients on treatment_trastuzumab_tucatinib=1 have different pfs_months than those off."),
        H("h4_rego", "Patients on treatment_regorafenib=1 have different pfs_months than those off."),
    ],
    "analyses": [
        A("i4_treatment_cetuximab", ["h4_cetux"]),
        A("i4_treatment_bevacizumab", ["h4_bev"]),
        A("i4_treatment_pembrolizumab", ["h4_pembro"]),
        A("i4_treatment_encorafenib", ["h4_enc"]),
        A("i4_treatment_trastuzumab_tucatinib", ["h4_tt"]),
        A("i4_treatment_regorafenib", ["h4_rego"]),
    ],
})

# Iteration 5 — canonical biomarker-treatment interactions
iterations.append({
    "index": 5,
    "proposed_hypotheses": [
        H("h5_cetux_kras", "There is a negative interaction between treatment_cetuximab and kras_mutation on pfs_months: cetuximab benefit (positive effect) is reduced or reversed in KRAS-mutant patients (interaction term negative)."),
        H("h5_cetux_nras", "There is a negative interaction between treatment_cetuximab and nras_mutation on pfs_months."),
        H("h5_cetux_braf", "There is a negative interaction between treatment_cetuximab and braf_v600e on pfs_months."),
        H("h5_pembro_msi", "There is a positive interaction between treatment_pembrolizumab and msi_high on pfs_months: pembrolizumab benefits MSI-high patients more (interaction term positive)."),
        H("h5_enc_braf", "There is a positive interaction between treatment_encorafenib and braf_v600e on pfs_months: encorafenib benefits BRAF V600E patients more."),
        H("h5_tt_her2", "There is a positive interaction between treatment_trastuzumab_tucatinib and her2_amplified on pfs_months."),
    ],
    "analyses": [
        A("i5_cetux_x_kras", ["h5_cetux_kras"], "OLS pfs ~ cetux + kras + cetux:kras"),
        A("i5_cetux_x_nras", ["h5_cetux_nras"]),
        A("i5_cetux_x_braf", ["h5_cetux_braf"]),
        A("i5_pembro_x_msi", ["h5_pembro_msi"]),
        A("i5_enco_x_braf", ["h5_enc_braf"]),
        A("i5_tt_x_her2", ["h5_tt_her2"]),
    ],
})

# Iteration 6 — biomarker-stratified subgroup analyses
iterations.append({
    "index": 6,
    "proposed_hypotheses": [
        H("h6_cetux_krasWT", "Among kras_mutation=0, treatment_cetuximab=1 has longer pfs_months than treatment_cetuximab=0 (positive diff).", kind="refined"),
        H("h6_cetux_krasMUT", "Among kras_mutation=1, treatment_cetuximab does not improve pfs_months (no positive diff).", kind="refined"),
        H("h6_cetux_nrasWT", "Among nras_mutation=0, treatment_cetuximab=1 has longer pfs_months than off.", kind="refined"),
        H("h6_cetux_nrasMUT", "Among nras_mutation=1, treatment_cetuximab does not improve pfs_months.", kind="refined"),
        H("h6_cetux_brafWT", "Among braf_v600e=0, treatment_cetuximab=1 has longer pfs_months than off.", kind="refined"),
        H("h6_cetux_brafMUT", "Among braf_v600e=1, treatment_cetuximab does not improve pfs_months.", kind="refined"),
        H("h6_pembro_msiH", "Among msi_high=1, treatment_pembrolizumab=1 has substantially longer pfs_months than off (positive diff).", kind="refined"),
        H("h6_pembro_msiL", "Among msi_high=0, treatment_pembrolizumab=1 has no benefit on pfs_months.", kind="refined"),
        H("h6_enc_brafMUT", "Among braf_v600e=1, treatment_encorafenib=1 has longer pfs_months than off (positive diff).", kind="refined"),
        H("h6_enc_brafWT", "Among braf_v600e=0, treatment_encorafenib=1 has no benefit on pfs_months."),
        H("h6_tt_her2A", "Among her2_amplified=1, treatment_trastuzumab_tucatinib=1 has longer pfs_months than off."),
        H("h6_tt_her2N", "Among her2_amplified=0, treatment_trastuzumab_tucatinib=1 has no benefit."),
    ],
    "analyses": [
        A("i6_treatment_cetuximab_in_kras_mutation=0", ["h6_cetux_krasWT"]),
        A("i6_treatment_cetuximab_in_kras_mutation=1", ["h6_cetux_krasMUT"]),
        A("i6_treatment_cetuximab_in_nras_mutation=0", ["h6_cetux_nrasWT"]),
        A("i6_treatment_cetuximab_in_nras_mutation=1", ["h6_cetux_nrasMUT"]),
        A("i6_treatment_cetuximab_in_braf_v600e=0", ["h6_cetux_brafWT"]),
        A("i6_treatment_cetuximab_in_braf_v600e=1", ["h6_cetux_brafMUT"]),
        A("i6_treatment_pembrolizumab_in_msi_high=1", ["h6_pembro_msiH"]),
        A("i6_treatment_pembrolizumab_in_msi_high=0", ["h6_pembro_msiL"]),
        A("i6_treatment_encorafenib_in_braf_v600e=1", ["h6_enc_brafMUT"]),
        A("i6_treatment_encorafenib_in_braf_v600e=0", ["h6_enc_brafWT"]),
        A("i6_treatment_trastuzumab_tucatinib_in_her2_amplified=1", ["h6_tt_her2A"]),
        A("i6_treatment_trastuzumab_tucatinib_in_her2_amplified=0", ["h6_tt_her2N"]),
    ],
})

# Iteration 7 — symptom and comorbidity prognostic effects
iterations.append({
    "index": 7,
    "proposed_hypotheses": [
        H("h7_fatigue", "Higher fatigue_grade is associated with shorter pfs_months."),
        H("h7_pain", "Higher pain_nrs is associated with shorter pfs_months."),
        H("h7_dyspnea", "Higher dyspnea_grade is associated with shorter pfs_months."),
        H("h7_appetite", "Higher appetite_loss_grade is associated with shorter pfs_months."),
        H("h7_cough", "Higher cough_grade is associated with shorter pfs_months."),
        H("h7_diabetes", "Patients with diabetes_mellitus=1 have different pfs_months than those without."),
        H("h7_htn", "Patients with hypertension=1 have different pfs_months than those without."),
        H("h7_copd", "Patients with copd=1 have shorter pfs_months than those without."),
        H("h7_ckd", "Patients with chronic_kidney_disease=1 have shorter pfs_months than those without."),
        H("h7_hf", "Patients with heart_failure=1 have shorter pfs_months."),
        H("h7_cad", "Patients with coronary_artery_disease=1 have different pfs_months."),
        H("h7_afib", "Patients with atrial_fibrillation=1 have different pfs_months."),
        H("h7_vte", "Patients with venous_thromboembolism_history=1 have different pfs_months."),
    ],
    "analyses": [
        A("i7_fatigue_grade", ["h7_fatigue"]),
        A("i7_pain_nrs", ["h7_pain"]),
        A("i7_dyspnea_grade", ["h7_dyspnea"]),
        A("i7_appetite_loss_grade", ["h7_appetite"]),
        A("i7_cough_grade", ["h7_cough"]),
        A("i7_diabetes_mellitus", ["h7_diabetes"]),
        A("i7_hypertension", ["h7_htn"]),
        A("i7_copd", ["h7_copd"]),
        A("i7_chronic_kidney_disease", ["h7_ckd"]),
        A("i7_heart_failure", ["h7_hf"]),
        A("i7_coronary_artery_disease", ["h7_cad"]),
        A("i7_atrial_fibrillation", ["h7_afib"]),
        A("i7_venous_thromboembolism_history", ["h7_vte"]),
    ],
})

# Iteration 8 — demographics and vitals
iterations.append({
    "index": 8,
    "proposed_hypotheses": [
        H("h8_age", "Older age_years is associated with shorter pfs_months (negative beta)."),
        H("h8_sex", "Female sex (sex_female=1) is associated with different pfs_months than male."),
        H("h8_rural", "Rural residence (rural_residence=1) is associated with shorter pfs_months."),
        H("h8_smoke", "Higher smoking_pack_years is associated with shorter pfs_months."),
        H("h8_bmi", "Higher bmi is associated with longer pfs_months (positive beta)."),
        H("h8_sbp", "Higher systolic_bp_mmhg is associated with shorter pfs_months."),
        H("h8_hr", "Higher heart_rate_bpm is associated with shorter pfs_months."),
        H("h8_spo2", "Higher spo2_pct is associated with longer pfs_months (positive beta)."),
        H("h8_edu", "More education_years is associated with longer pfs_months."),
    ],
    "analyses": [
        A("i8_age_years", ["h8_age"]),
        A("i8_sex_female", ["h8_sex"]),
        A("i8_rural_residence", ["h8_rural"]),
        A("i8_smoking_pack_years", ["h8_smoke"]),
        A("i8_bmi", ["h8_bmi"]),
        A("i8_systolic_bp_mmhg", ["h8_sbp"]),
        A("i8_heart_rate_bpm", ["h8_hr"]),
        A("i8_spo2_pct", ["h8_spo2"]),
        A("i8_education_years", ["h8_edu"]),
    ],
})

# Iteration 9 — race/ethnicity and insurance
iterations.append({
    "index": 9,
    "proposed_hypotheses": [
        H("h9_race", "Mean pfs_months differs across race_ethnicity groups."),
        H("h9_ins", "Mean pfs_months differs across insurance_type groups."),
    ],
    "analyses": [
        A("i9_race_ethnicity", ["h9_race"], "stats.f_oneway across race groups"),
        A("i9_insurance_type", ["h9_ins"], "stats.f_oneway across insurance groups"),
    ],
})

# Iteration 10 — prior treatment / disease history
iterations.append({
    "index": 10,
    "proposed_hypotheses": [
        H("h10_priorchemo", "prior_chemotherapy=1 is associated with shorter pfs_months."),
        H("h10_priorrad", "prior_radiation=1 is associated with different pfs_months."),
        H("h10_priorsurg", "prior_surgery=1 is associated with longer pfs_months."),
        H("h10_priorimmuno", "prior_immunotherapy=1 is associated with different pfs_months."),
        H("h10_priortarg", "prior_targeted_therapy=1 is associated with different pfs_months."),
        H("h10_priorlines", "More prior_lines_of_therapy is associated with shorter pfs_months."),
        H("h10_yrsdx", "More years_since_diagnosis is associated with longer pfs_months."),
        H("h10_priormalig", "Patients with prior_malignancy=1 have different pfs_months."),
    ],
    "analyses": [
        A("i10_prior_chemotherapy", ["h10_priorchemo"]),
        A("i10_prior_radiation", ["h10_priorrad"]),
        A("i10_prior_surgery", ["h10_priorsurg"]),
        A("i10_prior_immunotherapy", ["h10_priorimmuno"]),
        A("i10_prior_targeted_therapy", ["h10_priortarg"]),
        A("i10_prior_lines_of_therapy", ["h10_priorlines"]),
        A("i10_years_since_diagnosis", ["h10_yrsdx"]),
        A("i10_prior_malignancy", ["h10_priormalig"]),
    ],
})

# Iteration 11 — covariate-adjusted treatment effects
iterations.append({
    "index": 11,
    "proposed_hypotheses": [
        H("h11_cetux_adj", "After adjusting for age, ECOG, stage_iv, albumin, LDH, and liver_mets, treatment_cetuximab is associated with longer pfs_months (positive beta).", kind="refined"),
        H("h11_bev_adj", "After adjustment, treatment_bevacizumab is associated with longer pfs_months."),
        H("h11_pembro_adj", "After adjustment, treatment_pembrolizumab is associated with longer pfs_months."),
        H("h11_enc_adj", "After adjustment, treatment_encorafenib is associated with longer pfs_months."),
        H("h11_tt_adj", "After adjustment, treatment_trastuzumab_tucatinib is associated with longer pfs_months."),
        H("h11_rego_adj", "After adjustment, treatment_regorafenib is associated with longer pfs_months."),
    ],
    "analyses": [
        A("i11_adj_treatment_cetuximab", ["h11_cetux_adj"]),
        A("i11_adj_treatment_bevacizumab", ["h11_bev_adj"]),
        A("i11_adj_treatment_pembrolizumab", ["h11_pembro_adj"]),
        A("i11_adj_treatment_encorafenib", ["h11_enc_adj"]),
        A("i11_adj_treatment_trastuzumab_tucatinib", ["h11_tt_adj"]),
        A("i11_adj_treatment_regorafenib", ["h11_rego_adj"]),
    ],
})

# Iteration 12 — laterality x cetuximab
iterations.append({
    "index": 12,
    "proposed_hypotheses": [
        H("h12_cetux_lat", "There is a negative interaction between treatment_cetuximab and right_sided_primary on pfs_months: cetuximab benefit is attenuated or reversed in right-sided tumors."),
        H("h12_cetux_rightRASwt", "Among right-sided RAS-WT (KRAS=0 & NRAS=0) patients, treatment_cetuximab=1 has shorter or equal pfs_months than off (no benefit).", kind="refined"),
        H("h12_cetux_leftRASwt", "Among left-sided RAS-WT (KRAS=0 & NRAS=0) patients, treatment_cetuximab=1 has longer pfs_months than off.", kind="refined"),
    ],
    "analyses": [
        A("i12_cetux_x_rightsided", ["h12_cetux_lat"]),
        A("i12_cetux_in_rightRASwt", ["h12_cetux_rightRASwt"]),
        A("i12_cetux_in_leftRASwt", ["h12_cetux_leftRASwt"]),
    ],
})

# Iteration 13 — triple-marker subgroup
iterations.append({
    "index": 13,
    "proposed_hypotheses": [
        H("h13_tripleWT", "Among patients with KRAS=0, NRAS=0, and BRAF V600E=0 (triple wild-type), treatment_cetuximab=1 has longer pfs_months than off (positive diff).", kind="refined"),
    ],
    "analyses": [
        A("i13_cetux_in_tripleWT", ["h13_tripleWT"]),
    ],
})

# Iteration 14 — SNP scan
snp_cols = [
    "snp_rs1045642","snp_rs1065852","snp_rs1799853","snp_rs1800566","snp_rs2228001",
    "snp_rs3813867","snp_rs4244285","snp_rs4986893","snp_rs1801133","snp_rs1800896",
    "snp_rs1800629","snp_rs2228570","snp_rs1801131","snp_rs429358","snp_rs7412",
    "snp_rs662","snp_rs2298771","snp_rs2032582","snp_rs1128503","snp_rs1800470",
    "snp_rs1799983","snp_rs4880","snp_rs1050828","snp_rs4363657","snp_rs2070744",
    "snp_rs6025","snp_rs1801197"
]
hyps_14 = [H(f"h14_{c}", f"There is a non-zero linear association between {c} (0/1/2 dose) and pfs_months in OLS.") for c in snp_cols]
iterations.append({
    "index": 14,
    "proposed_hypotheses": hyps_14,
    "analyses": [A(f"i14_{c}", [f"h14_{c}"], f"OLS pfs ~ {c}") for c in snp_cols],
})

# Iteration 15 — BRAF combo (encorafenib + cetuximab)
iterations.append({
    "index": 15,
    "proposed_hypotheses": [
        H("h15_braf_table", "Within braf_v600e=1, mean pfs_months differs across the four cells defined by treatment_encorafenib × treatment_cetuximab (descriptive)."),
        H("h15_enc_x_cetux_braf", "Within braf_v600e=1, the interaction between treatment_encorafenib and treatment_cetuximab on pfs_months is positive (combo synergy)."),
    ],
    "analyses": [
        A("i15_braf_combo_table", ["h15_braf_table"]),
        A("i15_enc_x_cetux_in_braf", ["h15_enc_x_cetux_braf"]),
    ],
})

# Iteration 16 — bevacizumab interactions
iterations.append({
    "index": 16,
    "proposed_hypotheses": [
        H("h16_bev_lat", "There is a non-zero interaction between treatment_bevacizumab and right_sided_primary on pfs_months."),
        H("h16_bev_kras", "There is a non-zero interaction between treatment_bevacizumab and kras_mutation on pfs_months."),
    ],
    "analyses": [
        A("i16_bev_x_rightsided", ["h16_bev_lat"]),
        A("i16_bev_x_kras", ["h16_bev_kras"]),
    ],
})

# Iteration 17 — regorafenib × prior lines
iterations.append({
    "index": 17,
    "proposed_hypotheses": [
        H("h17_rego_lines", "There is a non-zero interaction between treatment_regorafenib and prior_lines_of_therapy on pfs_months (regorafenib effect varies with line of therapy)."),
    ],
    "analyses": [
        A("i17_rego_x_priorlines", ["h17_rego_lines"]),
    ],
})

# Iteration 18 — pembrolizumab in MSI-H, adjusted
iterations.append({
    "index": 18,
    "proposed_hypotheses": [
        H("h18_pembro_msi_adj", "Within msi_high=1, after adjusting for age, ECOG, stage_iv, albumin, and LDH, treatment_pembrolizumab=1 is associated with longer pfs_months (positive beta).", kind="refined"),
    ],
    "analyses": [
        A("i18_pembro_in_msi_adj", ["h18_pembro_msi_adj"]),
    ],
})

# Iteration 19 — trastuzumab+tucatinib in HER2-amp, adjusted
iterations.append({
    "index": 19,
    "proposed_hypotheses": [
        H("h19_tt_her2_adj", "Within her2_amplified=1, after adjusting for age, ECOG, stage_iv, albumin, and LDH, treatment_trastuzumab_tucatinib=1 is associated with longer pfs_months.", kind="refined"),
    ],
    "analyses": [
        A("i19_tt_in_her2_adj", ["h19_tt_her2_adj"]),
    ],
})

# Iteration 20 — pembrolizumab access by insurance among MSI-H
iterations.append({
    "index": 20,
    "proposed_hypotheses": [
        H("h20_pembro_ins", "Among msi_high=1 patients, the proportion receiving treatment_pembrolizumab differs across insurance_type categories (chi-square)."),
    ],
    "analyses": [
        A("i20_pembro_access_by_insurance", ["h20_pembro_ins"]),
    ],
})

# Iteration 21 — BRAF combo vs neither
iterations.append({
    "index": 21,
    "proposed_hypotheses": [
        H("h21_braf_combo", "Within braf_v600e=1, patients receiving both treatment_encorafenib and treatment_cetuximab have longer pfs_months than patients receiving neither (positive diff).", kind="refined"),
    ],
    "analyses": [
        A("i21_braf_combo_vs_neither", ["h21_braf_combo"]),
    ],
})

# Iteration 22 — ECOG x stage IV
iterations.append({
    "index": 22,
    "proposed_hypotheses": [
        H("h22_ecog_stage", "There is a non-zero interaction between ecog_ps and stage_iv on pfs_months: the negative effect of ECOG is amplified or attenuated in stage IV vs non-stage-IV."),
    ],
    "analyses": [
        A("i22_ecog_x_stageiv", ["h22_ecog_stage"]),
    ],
})

# Iteration 23 — full multivariable OLS
iterations.append({
    "index": 23,
    "proposed_hypotheses": [
        H("h23_multivar", "In a comprehensive multivariable OLS for pfs_months including demographics, prognostic features, biomarkers, and all six treatments, the strongest independent predictors will be ecog_ps, stage_iv, albumin_g_dl, weight_loss_pct_6mo, and treatment_regorafenib; targeted-therapy treatments (cetuximab, pembrolizumab, encorafenib, trastuzumab+tucatinib, bevacizumab) will not have substantial independent effects.", kind="refined"),
    ],
    "analyses": [
        A("i23_full_OLS", ["h23_multivar"], "OLS pfs_months ~ comprehensive covariates"),
    ],
})

# Iteration 24 — targeted-treatment delivery audit
iterations.append({
    "index": 24,
    "proposed_hypotheses": [
        H("h24_pembro_audit", "Treatment_pembrolizumab is preferentially given to msi_high=1 patients vs msi_high=0 (positive proportion difference)."),
        H("h24_enc_audit", "Treatment_encorafenib is preferentially given to braf_v600e=1 patients vs braf_v600e=0."),
        H("h24_tt_audit", "Treatment_trastuzumab_tucatinib is preferentially given to her2_amplified=1 patients vs her2_amplified=0."),
        H("h24_cetux_audit", "Treatment_cetuximab is preferentially withheld from kras_mutation=1 patients (negative proportion difference)."),
    ],
    "analyses": [
        A("i24_audit_treatment_pembrolizumab_in_msi_high", ["h24_pembro_audit"]),
        A("i24_audit_treatment_encorafenib_in_braf_v600e", ["h24_enc_audit"]),
        A("i24_audit_treatment_trastuzumab_tucatinib_in_her2_amplified", ["h24_tt_audit"]),
        A("i24_audit_treatment_cetuximab_in_kras_mutation", ["h24_cetux_audit"]),
    ],
})

# Iteration 25 — final pembro re-check & prior immuno modifier
iterations.append({
    "index": 25,
    "proposed_hypotheses": [
        H("h25_pembro_msi_re", "Re-confirm: there is a positive interaction between treatment_pembrolizumab and msi_high on pfs_months (final check)."),
        H("h25_pembro_priorimmuno", "Within msi_high=1, the interaction between treatment_pembrolizumab and prior_immunotherapy is non-zero (prior immunotherapy modifies pembro response)."),
    ],
    "analyses": [
        A("i25_pembro_msi_recheck", ["h25_pembro_msi_re"]),
        A("i25_pembro_x_priorimmuno_in_msi", ["h25_pembro_priorimmuno"]),
    ],
})

transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@named-task-runner",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

# Quick validation
total_h = sum(len(it["proposed_hypotheses"]) for it in iterations)
total_a = sum(len(it["analyses"]) for it in iterations)
print(f"Wrote transcript.json: {len(iterations)} iterations, {total_h} hypotheses, {total_a} analyses")
