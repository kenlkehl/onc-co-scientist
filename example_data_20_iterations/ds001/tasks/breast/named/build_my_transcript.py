"""Assembles transcript.json from analyses stored in my_results.json."""
import json

with open("my_results.json") as fh:
    R = json.load(fh)


def A(*tags, hyp_ids):
    """Build analysis records from result tags."""
    out = []
    for tag in tags:
        if tag not in R:
            continue
        r = R[tag]
        out.append({
            "hypothesis_ids": hyp_ids,
            "code": f"# tag={tag} -- see my_full_analysis.py",
            "result_summary": r["result_summary"],
            "p_value": r["p_value"],
            "effect_estimate": r["effect_estimate"],
            "significant": r["significant"],
        })
    return out


iterations = []

# ----- ITERATION 1 -----
iterations.append({
    "index": 1,
    "proposed_hypotheses": [
        {"id": "h1.1", "text": "Patients receiving treatment_tamoxifen have a different mean pfs_months than patients not receiving treatment_tamoxifen.", "kind": "novel"},
        {"id": "h1.2", "text": "Patients receiving treatment_palbociclib have a higher mean pfs_months than patients not receiving treatment_palbociclib.", "kind": "novel"},
        {"id": "h1.3", "text": "Patients receiving treatment_trastuzumab have a higher mean pfs_months than patients not receiving treatment_trastuzumab.", "kind": "novel"},
        {"id": "h1.4", "text": "Patients receiving treatment_olaparib have a different mean pfs_months than patients not receiving treatment_olaparib.", "kind": "novel"},
        {"id": "h1.5", "text": "Patients receiving treatment_sacituzumab_govitecan have a different mean pfs_months than patients not receiving it.", "kind": "novel"},
        {"id": "h1.6", "text": "Patients receiving treatment_pembrolizumab have a different mean pfs_months than patients not receiving it.", "kind": "novel"},
    ],
    "analyses":
        A("i1_treatment_tamoxifen_main", hyp_ids=["h1.1"]) +
        A("i1_treatment_palbociclib_main", hyp_ids=["h1.2"]) +
        A("i1_treatment_trastuzumab_main", hyp_ids=["h1.3"]) +
        A("i1_treatment_olaparib_main", hyp_ids=["h1.4"]) +
        A("i1_treatment_sacituzumab_govitecan_main", hyp_ids=["h1.5"]) +
        A("i1_treatment_pembrolizumab_main", hyp_ids=["h1.6"]),
})

# ----- ITERATION 2 -----
iterations.append({
    "index": 2,
    "proposed_hypotheses": [
        {"id": "h2.1", "text": "In er_positive=1 patients, treatment_tamoxifen is associated with higher mean pfs_months than no tamoxifen (i.e., tamoxifen benefit is concentrated in ER+ disease).", "kind": "refined"},
        {"id": "h2.2", "text": "In er_positive=0 patients, treatment_tamoxifen is not associated with higher mean pfs_months (tamoxifen has no benefit in ER- disease).", "kind": "refined"},
        {"id": "h2.3", "text": "There is a positive treatment_tamoxifen × er_positive interaction in an OLS of pfs_months: the tamoxifen effect is more positive in ER+ than in ER- patients.", "kind": "refined"},
    ],
    "analyses":
        A("i2_tam_in_erpos", hyp_ids=["h2.1"]) +
        A("i2_tam_in_erneg", hyp_ids=["h2.2"]) +
        A("i2_tam_x_er_interaction", hyp_ids=["h2.3"]),
})

# ----- ITERATION 3 -----
iterations.append({
    "index": 3,
    "proposed_hypotheses": [
        {"id": "h3.1", "text": "In her2_positive=1 patients, treatment_trastuzumab is associated with higher mean pfs_months than no trastuzumab (HER2-targeted benefit in HER2+ disease).", "kind": "refined"},
        {"id": "h3.2", "text": "In her2_positive=0 patients, treatment_trastuzumab is not associated with higher mean pfs_months.", "kind": "refined"},
        {"id": "h3.3", "text": "There is a positive treatment_trastuzumab × her2_positive interaction in an OLS of pfs_months.", "kind": "refined"},
    ],
    "analyses":
        A("i3_tras_in_her2pos", hyp_ids=["h3.1"]) +
        A("i3_tras_in_her2neg", hyp_ids=["h3.2"]) +
        A("i3_tras_x_her2_interaction", hyp_ids=["h3.3"]),
})

# ----- ITERATION 4 -----
iterations.append({
    "index": 4,
    "proposed_hypotheses": [
        {"id": "h4.1", "text": "In patients with brca1_mutation=1 OR brca2_mutation=1, treatment_olaparib is associated with higher mean pfs_months than no olaparib.", "kind": "refined"},
        {"id": "h4.2", "text": "In BRCA1/2 wild-type patients, treatment_olaparib is not associated with higher mean pfs_months.", "kind": "refined"},
        {"id": "h4.3", "text": "There is a positive treatment_olaparib × (brca1 OR brca2) interaction in an OLS of pfs_months.", "kind": "refined"},
    ],
    "analyses":
        A("i4_olap_in_brca_any", hyp_ids=["h4.1"]) +
        A("i4_olap_in_brca_wt", hyp_ids=["h4.2"]) +
        A("i4_olap_x_brca_interaction", hyp_ids=["h4.3"]),
})

# ----- ITERATION 5 -----
iterations.append({
    "index": 5,
    "proposed_hypotheses": [
        {"id": "h5.1", "text": "In her2_low=1 patients, treatment_sacituzumab_govitecan is associated with higher mean pfs_months than no sacituzumab.", "kind": "refined"},
        {"id": "h5.2", "text": "In her2_low=0 patients, treatment_sacituzumab_govitecan is not associated with higher mean pfs_months.", "kind": "refined"},
        {"id": "h5.3", "text": "There is a positive treatment_sacituzumab_govitecan × her2_low interaction in an OLS of pfs_months.", "kind": "refined"},
    ],
    "analyses":
        A("i5_sg_in_her2low", hyp_ids=["h5.1"]) +
        A("i5_sg_in_her2_not_low", hyp_ids=["h5.2"]) +
        A("i5_sg_x_her2low_interaction", hyp_ids=["h5.3"]),
})

# ----- ITERATION 6 -----
iterations.append({
    "index": 6,
    "proposed_hypotheses": [
        {"id": "h6.1", "text": "In msi_high=1 patients, treatment_pembrolizumab is associated with higher mean pfs_months than no pembrolizumab.", "kind": "refined"},
        {"id": "h6.2", "text": "In msi_high=0 patients, treatment_pembrolizumab is not associated with higher mean pfs_months.", "kind": "refined"},
        {"id": "h6.3", "text": "There is a positive treatment_pembrolizumab × msi_high interaction in an OLS of pfs_months.", "kind": "refined"},
    ],
    "analyses":
        A("i6_pembro_in_msi_high", hyp_ids=["h6.1"]) +
        A("i6_pembro_in_msi_low", hyp_ids=["h6.2"]) +
        A("i6_pembro_x_msi_interaction", hyp_ids=["h6.3"]),
})

# ----- ITERATION 7 -----
iterations.append({
    "index": 7,
    "proposed_hypotheses": [
        {"id": "h7.1", "text": "stage_iv=1 patients have lower mean pfs_months than stage_iv=0 patients.", "kind": "novel"},
        {"id": "h7.2", "text": "has_brain_mets=1 patients have lower mean pfs_months than has_brain_mets=0 patients.", "kind": "novel"},
        {"id": "h7.3", "text": "liver_mets=1 patients have lower mean pfs_months than liver_mets=0 patients.", "kind": "novel"},
        {"id": "h7.4", "text": "bone_mets=1 patients have lower mean pfs_months than bone_mets=0 patients.", "kind": "novel"},
        {"id": "h7.5", "text": "node_positive=1 patients have lower mean pfs_months than node_positive=0 patients.", "kind": "novel"},
        {"id": "h7.6", "text": "pleural_effusion=1 patients have lower mean pfs_months than pleural_effusion=0 patients.", "kind": "novel"},
        {"id": "h7.7", "text": "adrenal_mets=1 patients have lower mean pfs_months than adrenal_mets=0 patients.", "kind": "novel"},
    ],
    "analyses":
        A("i7_stage_iv_main", hyp_ids=["h7.1"]) +
        A("i7_has_brain_mets_main", hyp_ids=["h7.2"]) +
        A("i7_liver_mets_main", hyp_ids=["h7.3"]) +
        A("i7_bone_mets_main", hyp_ids=["h7.4"]) +
        A("i7_node_positive_main", hyp_ids=["h7.5"]) +
        A("i7_pleural_effusion_main", hyp_ids=["h7.6"]) +
        A("i7_adrenal_mets_main", hyp_ids=["h7.7"]),
})

# ----- ITERATION 8 -----
iterations.append({
    "index": 8,
    "proposed_hypotheses": [
        {"id": "h8.1", "text": "Higher ecog_ps is associated with lower pfs_months (negative Spearman correlation).", "kind": "novel"},
        {"id": "h8.2", "text": "Higher fatigue_grade is associated with lower pfs_months.", "kind": "novel"},
        {"id": "h8.3", "text": "Higher pain_nrs is associated with lower pfs_months.", "kind": "novel"},
        {"id": "h8.4", "text": "Higher dyspnea_grade is associated with lower pfs_months.", "kind": "novel"},
        {"id": "h8.5", "text": "Higher appetite_loss_grade is associated with lower pfs_months.", "kind": "novel"},
        {"id": "h8.6", "text": "Higher weight_loss_pct_6mo is associated with lower pfs_months.", "kind": "novel"},
    ],
    "analyses":
        A("i8_ecog_ps_corr", hyp_ids=["h8.1"]) +
        A("i8_fatigue_grade_corr", hyp_ids=["h8.2"]) +
        A("i8_pain_nrs_corr", hyp_ids=["h8.3"]) +
        A("i8_dyspnea_grade_corr", hyp_ids=["h8.4"]) +
        A("i8_appetite_loss_grade_corr", hyp_ids=["h8.5"]) +
        A("i8_weight_loss_pct_6mo_corr", hyp_ids=["h8.6"]),
})

# ----- ITERATION 9 -----
iterations.append({
    "index": 9,
    "proposed_hypotheses": [
        {"id": "h9.1", "text": "Higher albumin_g_dl is associated with higher pfs_months.", "kind": "novel"},
        {"id": "h9.2", "text": "Higher ldh_u_l is associated with lower pfs_months.", "kind": "novel"},
        {"id": "h9.3", "text": "Higher crp_mg_l is associated with lower pfs_months.", "kind": "novel"},
        {"id": "h9.4", "text": "Higher nlr (neutrophil-to-lymphocyte ratio) is associated with lower pfs_months.", "kind": "novel"},
        {"id": "h9.5", "text": "Higher hemoglobin_g_dl is associated with higher pfs_months.", "kind": "novel"},
        {"id": "h9.6", "text": "Higher alkaline_phosphatase_u_l is associated with lower pfs_months.", "kind": "novel"},
        {"id": "h9.7", "text": "Higher calcium_mg_dl is associated with lower pfs_months.", "kind": "novel"},
        {"id": "h9.8", "text": "Higher platelets_k_ul is associated with higher pfs_months.", "kind": "novel"},
        {"id": "h9.9", "text": "Higher alc_k_ul (absolute lymphocyte count) is associated with higher pfs_months.", "kind": "novel"},
        {"id": "h9.10", "text": "Higher anc_k_ul (absolute neutrophil count) is associated with lower pfs_months.", "kind": "novel"},
        {"id": "h9.11", "text": "Higher ki67_pct is associated with lower pfs_months.", "kind": "novel"},
        {"id": "h9.12", "text": "Higher tumor_size_cm is associated with lower pfs_months.", "kind": "novel"},
    ],
    "analyses":
        A("i9_albumin_g_dl_corr", hyp_ids=["h9.1"]) +
        A("i9_ldh_u_l_corr", hyp_ids=["h9.2"]) +
        A("i9_crp_mg_l_corr", hyp_ids=["h9.3"]) +
        A("i9_nlr_corr", hyp_ids=["h9.4"]) +
        A("i9_hemoglobin_g_dl_corr", hyp_ids=["h9.5"]) +
        A("i9_alkaline_phosphatase_u_l_corr", hyp_ids=["h9.6"]) +
        A("i9_calcium_mg_dl_corr", hyp_ids=["h9.7"]) +
        A("i9_platelets_k_ul_corr", hyp_ids=["h9.8"]) +
        A("i9_alc_k_ul_corr", hyp_ids=["h9.9"]) +
        A("i9_anc_k_ul_corr", hyp_ids=["h9.10"]) +
        A("i9_ki67_pct_corr", hyp_ids=["h9.11"]) +
        A("i9_tumor_size_cm_corr", hyp_ids=["h9.12"]),
})

# ----- ITERATION 10 -----
iterations.append({
    "index": 10,
    "proposed_hypotheses": [
        {"id": "h10.1", "text": "Older age_years is associated with lower pfs_months (negative Spearman correlation).", "kind": "novel"},
        {"id": "h10.2", "text": "Female (sex_female=1) patients have a different mean pfs_months than male patients.", "kind": "novel"},
        {"id": "h10.3", "text": "Mean pfs_months differs across race_ethnicity categories (one-way ANOVA).", "kind": "novel"},
        {"id": "h10.4", "text": "Black patients have lower mean pfs_months than white patients.", "kind": "novel"},
        {"id": "h10.5", "text": "Mean pfs_months differs across insurance_type categories (one-way ANOVA).", "kind": "novel"},
        {"id": "h10.6", "text": "Uninsured patients have lower mean pfs_months than privately insured patients.", "kind": "novel"},
        {"id": "h10.7", "text": "Patients with rural_residence=1 have lower mean pfs_months than urban residents.", "kind": "novel"},
    ],
    "analyses":
        A("i10_age_corr", hyp_ids=["h10.1"]) +
        A("i10_female_main", hyp_ids=["h10.2"]) +
        A("i10_race_anova", hyp_ids=["h10.3"]) +
        A("i10_black_vs_white", hyp_ids=["h10.4"]) +
        A("i10_insurance_anova", hyp_ids=["h10.5"]) +
        A("i10_uninsured_vs_private", hyp_ids=["h10.6"]) +
        A("i10_rural_main", hyp_ids=["h10.7"]),
})

# ----- ITERATION 11 -----
iterations.append({
    "index": 11,
    "proposed_hypotheses": [
        {"id": "h11.1", "text": "In postmenopausal=1 patients, treatment_tamoxifen is associated with higher mean pfs_months than no tamoxifen.", "kind": "refined"},
        {"id": "h11.2", "text": "In postmenopausal=0 (premenopausal) patients, treatment_tamoxifen is associated with higher mean pfs_months than no tamoxifen.", "kind": "refined"},
        {"id": "h11.3", "text": "There is a non-zero treatment_tamoxifen × postmenopausal interaction in an OLS of pfs_months.", "kind": "refined"},
    ],
    "analyses":
        A("i11_tam_in_post", hyp_ids=["h11.1"]) +
        A("i11_tam_in_pre", hyp_ids=["h11.2"]) +
        A("i11_tam_x_post_interaction", hyp_ids=["h11.3"]),
})

# ----- ITERATION 12 -----
iterations.append({
    "index": 12,
    "proposed_hypotheses": [
        {"id": "h12.1", "text": "After adjusting for age, sex, ECOG, stage IV, brain/liver/bone mets, ER, HER2, BRCA, albumin, LDH, NLR, and other treatments, ecog_ps is independently associated with lower pfs_months.", "kind": "refined"},
        {"id": "h12.2", "text": "After multivariable adjustment, stage_iv is independently associated with lower pfs_months.", "kind": "refined"},
        {"id": "h12.3", "text": "After multivariable adjustment, has_brain_mets is independently associated with lower pfs_months.", "kind": "refined"},
        {"id": "h12.4", "text": "After multivariable adjustment, liver_mets is independently associated with lower pfs_months.", "kind": "refined"},
        {"id": "h12.5", "text": "After multivariable adjustment, bone_mets is independently associated with lower pfs_months.", "kind": "refined"},
        {"id": "h12.6", "text": "After multivariable adjustment, higher albumin_g_dl is independently associated with higher pfs_months.", "kind": "refined"},
        {"id": "h12.7", "text": "After multivariable adjustment, higher ldh_u_l is independently associated with lower pfs_months.", "kind": "refined"},
        {"id": "h12.8", "text": "After multivariable adjustment, higher nlr is independently associated with lower pfs_months.", "kind": "refined"},
        {"id": "h12.9", "text": "After multivariable adjustment, treatment_tamoxifen is independently associated with higher pfs_months.", "kind": "refined"},
        {"id": "h12.10", "text": "After multivariable adjustment, treatment_palbociclib is independently associated with higher pfs_months.", "kind": "refined"},
        {"id": "h12.11", "text": "After multivariable adjustment, treatment_trastuzumab is independently associated with higher pfs_months.", "kind": "refined"},
        {"id": "h12.12", "text": "After multivariable adjustment, treatment_olaparib is independently associated with higher pfs_months.", "kind": "refined"},
        {"id": "h12.13", "text": "After multivariable adjustment, treatment_sacituzumab_govitecan is independently associated with higher pfs_months.", "kind": "refined"},
        {"id": "h12.14", "text": "After multivariable adjustment, treatment_pembrolizumab is independently associated with higher pfs_months.", "kind": "refined"},
    ],
    "analyses":
        A("i12_mv_ecog_ps", hyp_ids=["h12.1"]) +
        A("i12_mv_stage_iv", hyp_ids=["h12.2"]) +
        A("i12_mv_has_brain_mets", hyp_ids=["h12.3"]) +
        A("i12_mv_liver_mets", hyp_ids=["h12.4"]) +
        A("i12_mv_bone_mets", hyp_ids=["h12.5"]) +
        A("i12_mv_albumin_g_dl", hyp_ids=["h12.6"]) +
        A("i12_mv_ldh_u_l", hyp_ids=["h12.7"]) +
        A("i12_mv_nlr", hyp_ids=["h12.8"]) +
        A("i12_mv_treatment_tamoxifen", hyp_ids=["h12.9"]) +
        A("i12_mv_treatment_palbociclib", hyp_ids=["h12.10"]) +
        A("i12_mv_treatment_trastuzumab", hyp_ids=["h12.11"]) +
        A("i12_mv_treatment_olaparib", hyp_ids=["h12.12"]) +
        A("i12_mv_treatment_sacituzumab_govitecan", hyp_ids=["h12.13"]) +
        A("i12_mv_treatment_pembrolizumab", hyp_ids=["h12.14"]),
})

# ----- ITERATION 13 -----
iterations.append({
    "index": 13,
    "proposed_hypotheses": [
        {"id": "h13.1", "text": "diabetes_mellitus=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h13.2", "text": "hypertension=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h13.3", "text": "chronic_kidney_disease=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h13.4", "text": "heart_failure=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h13.5", "text": "copd=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h13.6", "text": "depression_anxiety_diagnosis=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h13.7", "text": "autoimmune_disease=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h13.8", "text": "prior_malignancy=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h13.9", "text": "venous_thromboembolism_history=1 patients have lower mean pfs_months.", "kind": "novel"},
    ],
    "analyses":
        A("i13_diabetes_mellitus_main", hyp_ids=["h13.1"]) +
        A("i13_hypertension_main", hyp_ids=["h13.2"]) +
        A("i13_chronic_kidney_disease_main", hyp_ids=["h13.3"]) +
        A("i13_heart_failure_main", hyp_ids=["h13.4"]) +
        A("i13_copd_main", hyp_ids=["h13.5"]) +
        A("i13_depression_anxiety_diagnosis_main", hyp_ids=["h13.6"]) +
        A("i13_autoimmune_disease_main", hyp_ids=["h13.7"]) +
        A("i13_prior_malignancy_main", hyp_ids=["h13.8"]) +
        A("i13_venous_thromboembolism_history_main", hyp_ids=["h13.9"]),
})

# ----- ITERATION 14 -----
iterations.append({
    "index": 14,
    "proposed_hypotheses": [
        {"id": "h14.1", "text": "In er_positive=1 patients, treatment_palbociclib is associated with higher mean pfs_months than no palbociclib (CDK4/6 benefit in ER+).", "kind": "refined"},
        {"id": "h14.2", "text": "In er_positive=0 patients, treatment_palbociclib is not associated with higher mean pfs_months.", "kind": "refined"},
        {"id": "h14.3", "text": "There is a positive treatment_palbociclib × er_positive interaction in an OLS of pfs_months.", "kind": "refined"},
        {"id": "h14.4", "text": "In ER+/HER2- patients (er_positive=1 AND her2_positive=0), treatment_palbociclib is associated with higher mean pfs_months than no palbociclib.", "kind": "refined"},
    ],
    "analyses":
        A("i14_palbo_in_erpos", hyp_ids=["h14.1"]) +
        A("i14_palbo_in_erneg", hyp_ids=["h14.2"]) +
        A("i14_palbo_x_er_interaction", hyp_ids=["h14.3"]) +
        A("i14_palbo_in_erpos_her2neg", hyp_ids=["h14.4"]),
})

# ----- ITERATION 15 -----
iterations.append({
    "index": 15,
    "proposed_hypotheses": [
        {"id": "h15.1", "text": "snp_rs1045642 dose is associated with pfs_months (Spearman).", "kind": "novel"},
        {"id": "h15.2", "text": "snp_rs1065852 dose is associated with pfs_months.", "kind": "novel"},
        {"id": "h15.3", "text": "snp_rs1799853 dose is associated with pfs_months.", "kind": "novel"},
        {"id": "h15.4", "text": "snp_rs4244285 dose is associated with pfs_months.", "kind": "novel"},
        {"id": "h15.5", "text": "snp_rs1801133 dose is associated with pfs_months.", "kind": "novel"},
        {"id": "h15.6", "text": "snp_rs429358 dose is associated with pfs_months.", "kind": "novel"},
        {"id": "h15.7", "text": "snp_rs7412 dose is associated with pfs_months.", "kind": "novel"},
        {"id": "h15.8", "text": "snp_rs1050828 dose is associated with pfs_months.", "kind": "novel"},
        {"id": "h15.9", "text": "snp_rs1800629 dose is associated with pfs_months.", "kind": "novel"},
    ],
    "analyses":
        A("i15_snp_rs1045642_corr", hyp_ids=["h15.1"]) +
        A("i15_snp_rs1065852_corr", hyp_ids=["h15.2"]) +
        A("i15_snp_rs1799853_corr", hyp_ids=["h15.3"]) +
        A("i15_snp_rs4244285_corr", hyp_ids=["h15.4"]) +
        A("i15_snp_rs1801133_corr", hyp_ids=["h15.5"]) +
        A("i15_snp_rs429358_corr", hyp_ids=["h15.6"]) +
        A("i15_snp_rs7412_corr", hyp_ids=["h15.7"]) +
        A("i15_snp_rs1050828_corr", hyp_ids=["h15.8"]) +
        A("i15_snp_rs1800629_corr", hyp_ids=["h15.9"]),
})

# ----- ITERATION 16 -----
iterations.append({
    "index": 16,
    "proposed_hypotheses": [
        {"id": "h16.1", "text": "In brca1_mutation=1 patients, treatment_olaparib is associated with higher mean pfs_months than no olaparib.", "kind": "refined"},
        {"id": "h16.2", "text": "In brca2_mutation=1 patients, treatment_olaparib is associated with higher mean pfs_months than no olaparib.", "kind": "refined"},
    ],
    "analyses":
        A("i16_olap_in_brca1_mutation", hyp_ids=["h16.1"]) +
        A("i16_olap_in_brca2_mutation", hyp_ids=["h16.2"]),
})

# ----- ITERATION 17 -----
iterations.append({
    "index": 17,
    "proposed_hypotheses": [
        {"id": "h17.1", "text": "In her2_amplification=1 patients, treatment_trastuzumab is associated with higher mean pfs_months than no trastuzumab.", "kind": "refined"},
        {"id": "h17.2", "text": "In her2_amplification=0 patients, treatment_trastuzumab is not associated with higher mean pfs_months.", "kind": "refined"},
    ],
    "analyses":
        A("i17_tras_in_her2amp", hyp_ids=["h17.1"]) +
        A("i17_tras_in_her2unamp", hyp_ids=["h17.2"]),
})

# ----- ITERATION 18 -----
iterations.append({
    "index": 18,
    "proposed_hypotheses": [
        {"id": "h18.1", "text": "In sex_female=1 patients, treatment_tamoxifen is associated with higher mean pfs_months.", "kind": "refined"},
        {"id": "h18.2", "text": "In sex_female=0 patients, treatment_tamoxifen is associated with higher mean pfs_months.", "kind": "refined"},
    ],
    "analyses":
        A("i18_tam_in_female", hyp_ids=["h18.1"]) +
        A("i18_tam_in_male", hyp_ids=["h18.2"]),
})

# ----- ITERATION 19 -----
iterations.append({
    "index": 19,
    "proposed_hypotheses": [
        {"id": "h19.1", "text": "In white patients, treatment_palbociclib is associated with higher mean pfs_months than no palbociclib.", "kind": "refined"},
        {"id": "h19.2", "text": "In black patients, treatment_palbociclib is associated with higher mean pfs_months than no palbociclib.", "kind": "refined"},
        {"id": "h19.3", "text": "In hispanic patients, treatment_palbociclib is associated with higher mean pfs_months than no palbociclib.", "kind": "refined"},
        {"id": "h19.4", "text": "In asian patients, treatment_palbociclib is associated with higher mean pfs_months than no palbociclib.", "kind": "refined"},
        {"id": "h19.5", "text": "In white patients, treatment_pembrolizumab is associated with higher mean pfs_months than no pembrolizumab.", "kind": "refined"},
        {"id": "h19.6", "text": "In black patients, treatment_pembrolizumab is associated with higher mean pfs_months than no pembrolizumab.", "kind": "refined"},
        {"id": "h19.7", "text": "In hispanic patients, treatment_pembrolizumab is associated with higher mean pfs_months than no pembrolizumab.", "kind": "refined"},
        {"id": "h19.8", "text": "In asian patients, treatment_pembrolizumab is associated with higher mean pfs_months than no pembrolizumab.", "kind": "refined"},
        {"id": "h19.9", "text": "In white patients, treatment_olaparib is associated with higher mean pfs_months than no olaparib.", "kind": "refined"},
        {"id": "h19.10", "text": "In black patients, treatment_olaparib is associated with higher mean pfs_months than no olaparib.", "kind": "refined"},
        {"id": "h19.11", "text": "In hispanic patients, treatment_olaparib is associated with higher mean pfs_months than no olaparib.", "kind": "refined"},
        {"id": "h19.12", "text": "In asian patients, treatment_olaparib is associated with higher mean pfs_months than no olaparib.", "kind": "refined"},
    ],
    "analyses":
        A("i19_treatment_palbociclib_in_white", hyp_ids=["h19.1"]) +
        A("i19_treatment_palbociclib_in_black", hyp_ids=["h19.2"]) +
        A("i19_treatment_palbociclib_in_hispanic", hyp_ids=["h19.3"]) +
        A("i19_treatment_palbociclib_in_asian", hyp_ids=["h19.4"]) +
        A("i19_treatment_pembrolizumab_in_white", hyp_ids=["h19.5"]) +
        A("i19_treatment_pembrolizumab_in_black", hyp_ids=["h19.6"]) +
        A("i19_treatment_pembrolizumab_in_hispanic", hyp_ids=["h19.7"]) +
        A("i19_treatment_pembrolizumab_in_asian", hyp_ids=["h19.8"]) +
        A("i19_treatment_olaparib_in_white", hyp_ids=["h19.9"]) +
        A("i19_treatment_olaparib_in_black", hyp_ids=["h19.10"]) +
        A("i19_treatment_olaparib_in_hispanic", hyp_ids=["h19.11"]) +
        A("i19_treatment_olaparib_in_asian", hyp_ids=["h19.12"]),
})

# ----- ITERATION 20 -----
iterations.append({
    "index": 20,
    "proposed_hypotheses": [
        {"id": "h20.1", "text": "Higher prior_lines_of_therapy is associated with lower pfs_months (Spearman).", "kind": "novel"},
        {"id": "h20.2", "text": "prior_chemotherapy=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h20.3", "text": "prior_radiation=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h20.4", "text": "prior_surgery=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h20.5", "text": "prior_immunotherapy=1 patients have lower mean pfs_months.", "kind": "novel"},
        {"id": "h20.6", "text": "prior_targeted_therapy=1 patients have lower mean pfs_months.", "kind": "novel"},
    ],
    "analyses":
        A("i20_prior_lines_corr", hyp_ids=["h20.1"]) +
        A("i20_prior_chemotherapy_main", hyp_ids=["h20.2"]) +
        A("i20_prior_radiation_main", hyp_ids=["h20.3"]) +
        A("i20_prior_surgery_main", hyp_ids=["h20.4"]) +
        A("i20_prior_immunotherapy_main", hyp_ids=["h20.5"]) +
        A("i20_prior_targeted_therapy_main", hyp_ids=["h20.6"]),
})

# ----- ITERATION 21 -----
iterations.append({
    "index": 21,
    "proposed_hypotheses": [
        {"id": "h21.1", "text": "pik3ca_mutation=1 patients have lower mean pfs_months than pik3ca_mutation=0 patients.", "kind": "novel"},
        {"id": "h21.2", "text": "In pik3ca_mutation=1 patients, treatment_olaparib is associated with higher mean pfs_months than no olaparib.", "kind": "refined"},
    ],
    "analyses":
        A("i21_pik3ca_main", hyp_ids=["h21.1"]) +
        A("i21_olap_in_pik3ca", hyp_ids=["h21.2"]),
})

# ----- ITERATION 22 -----
iterations.append({
    "index": 22,
    "proposed_hypotheses": [
        {"id": "h22.1", "text": "In her2_low=1 AND her2_positive=0 patients, treatment_trastuzumab is associated with higher mean pfs_months than no trastuzumab (off-label HER2-low benefit).", "kind": "refined"},
        {"id": "h22.2", "text": "In stage_iv=1 patients, treatment_pembrolizumab is associated with higher mean pfs_months than no pembrolizumab.", "kind": "refined"},
        {"id": "h22.3", "text": "In stage_iv=0 patients, treatment_pembrolizumab is associated with higher mean pfs_months than no pembrolizumab.", "kind": "refined"},
    ],
    "analyses":
        A("i22_tras_in_her2low_only", hyp_ids=["h22.1"]) +
        A("i22_pembro_in_stage4", hyp_ids=["h22.2"]) +
        A("i22_pembro_in_not_stage4", hyp_ids=["h22.3"]),
})

# ----- ITERATION 23 -----
iterations.append({
    "index": 23,
    "proposed_hypotheses": [
        {"id": "h23.1", "text": "In below-median-albumin patients, treatment_palbociclib is associated with higher mean pfs_months than no palbociclib.", "kind": "refined"},
        {"id": "h23.2", "text": "In at-or-above-median-albumin patients, treatment_palbociclib is associated with higher mean pfs_months than no palbociclib.", "kind": "refined"},
        {"id": "h23.3", "text": "In below-median-albumin patients, treatment_pembrolizumab is associated with higher mean pfs_months than no pembrolizumab.", "kind": "refined"},
        {"id": "h23.4", "text": "In at-or-above-median-albumin patients, treatment_pembrolizumab is associated with higher mean pfs_months than no pembrolizumab.", "kind": "refined"},
    ],
    "analyses":
        A("i23_treatment_palbociclib_low_alb", hyp_ids=["h23.1"]) +
        A("i23_treatment_palbociclib_high_alb", hyp_ids=["h23.2"]) +
        A("i23_treatment_pembrolizumab_low_alb", hyp_ids=["h23.3"]) +
        A("i23_treatment_pembrolizumab_high_alb", hyp_ids=["h23.4"]),
})

# ----- ITERATION 24 -----
iterations.append({
    "index": 24,
    "proposed_hypotheses": [
        {"id": "h24.1", "text": "After adjusting for age, ECOG, stage IV, albumin, LDH, NLR within the ER+/HER2- subgroup, treatment_palbociclib is independently associated with higher pfs_months.", "kind": "refined"},
        {"id": "h24.2", "text": "After adjusting for age, ECOG, stage IV, albumin, LDH, NLR within the HER2+ subgroup, treatment_trastuzumab is independently associated with higher pfs_months.", "kind": "refined"},
        {"id": "h24.3", "text": "After adjusting for age, ECOG, stage IV, albumin, LDH, NLR within the BRCA1/2-mutated subgroup, treatment_olaparib is independently associated with higher pfs_months.", "kind": "refined"},
    ],
    "analyses":
        A("i24_palbo_in_erpos_her2neg_adjusted", hyp_ids=["h24.1"]) +
        A("i24_tras_in_her2pos_adjusted", hyp_ids=["h24.2"]) +
        A("i24_olap_in_brcaany_adjusted", hyp_ids=["h24.3"]),
})

# ----- ITERATION 25 -----
iterations.append({
    "index": 25,
    "proposed_hypotheses": [
        {"id": "h25.1", "text": "Patients receiving any biomarker-matched targeted therapy (tamoxifen+ER, palbociclib+ER+/HER2-, trastuzumab+HER2+, olaparib+BRCA1/2, sacituzumab+HER2-low/HER2-) have higher mean pfs_months than patients not receiving a biomarker-matched targeted therapy.", "kind": "refined"},
        {"id": "h25.2", "text": "After adjusting for age, ECOG, stage IV, albumin, LDH, and NLR, the biomarker-matched-treatment indicator remains independently associated with higher pfs_months.", "kind": "refined"},
    ],
    "analyses":
        A("i25_biomarker_matched_tx_main", hyp_ids=["h25.1"]) +
        A("i25_biomarker_matched_tx_adjusted", hyp_ids=["h25.2"]),
})

transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-custom@ds001-breast-v1",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json", "w") as fh:
    json.dump(transcript, fh, indent=2)

# Sanity check
n_hyp = sum(len(it["proposed_hypotheses"]) for it in iterations)
n_ana = sum(len(it["analyses"]) for it in iterations)
print(f"Wrote transcript.json with {len(iterations)} iterations, {n_hyp} hypotheses, {n_ana} analyses")
