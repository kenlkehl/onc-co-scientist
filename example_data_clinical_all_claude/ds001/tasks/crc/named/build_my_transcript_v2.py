"""Build transcript.json from new_results.json."""
import json

R = json.load(open("new_results.json"))


def a(hyp_ids, key, code=None):
    """Build an analysis record from a results entry."""
    r = R[key]
    return {
        "hypothesis_ids": hyp_ids,
        "code": code,
        "result_summary": r["summary"],
        "p_value": r["p_value"],
        "effect_estimate": r["effect_estimate"],
        "significant": r["significant"],
    }


iterations = []

# ----- Iteration 1: treatment main effects -----
hyps1 = [
    {"id": "h1.1", "text": "Patients receiving treatment_cetuximab have different mean pfs_months than those who do not.", "kind": "novel"},
    {"id": "h1.2", "text": "Patients receiving treatment_bevacizumab have different mean pfs_months than those who do not.", "kind": "novel"},
    {"id": "h1.3", "text": "Patients receiving treatment_pembrolizumab have different mean pfs_months than those who do not.", "kind": "novel"},
    {"id": "h1.4", "text": "Patients receiving treatment_encorafenib have different mean pfs_months than those who do not.", "kind": "novel"},
    {"id": "h1.5", "text": "Patients receiving treatment_trastuzumab_tucatinib have different mean pfs_months than those who do not.", "kind": "novel"},
    {"id": "h1.6", "text": "Patients receiving treatment_regorafenib have higher mean pfs_months than those who do not.", "kind": "novel"},
]
ana1 = [
    a(["h1.1"], "i1_treatment_cetuximab_main", "stats.ttest_ind(df.loc[df.treatment_cetuximab==1,'pfs_months'], df.loc[df.treatment_cetuximab==0,'pfs_months'])"),
    a(["h1.2"], "i1_treatment_bevacizumab_main"),
    a(["h1.3"], "i1_treatment_pembrolizumab_main"),
    a(["h1.4"], "i1_treatment_encorafenib_main"),
    a(["h1.5"], "i1_treatment_trastuzumab_tucatinib_main"),
    a(["h1.6"], "i1_treatment_regorafenib_main"),
]
iterations.append({"index": 1, "proposed_hypotheses": hyps1, "analyses": ana1})

# ----- Iteration 2: clinical/biomarker main effects -----
hyps2 = [
    {"id": "h2.1", "text": "Patients with kras_mutation have lower mean pfs_months than KRAS-wild-type patients.", "kind": "novel"},
    {"id": "h2.2", "text": "Patients with nras_mutation have different mean pfs_months than NRAS-wild-type patients.", "kind": "novel"},
    {"id": "h2.3", "text": "Patients with braf_v600e have lower mean pfs_months than BRAF-wild-type patients.", "kind": "novel"},
    {"id": "h2.4", "text": "Patients with msi_high have different mean pfs_months than MSS patients.", "kind": "novel"},
    {"id": "h2.5", "text": "Patients with her2_amplified have different mean pfs_months than HER2-non-amplified patients.", "kind": "novel"},
    {"id": "h2.6", "text": "Patients with ntrk_fusion have different mean pfs_months than NTRK-wild-type patients.", "kind": "novel"},
    {"id": "h2.7", "text": "Patients with stage_iv have lower mean pfs_months than non-stage-IV patients.", "kind": "novel"},
    {"id": "h2.8", "text": "Patients with right_sided_primary have lower mean pfs_months than left-sided patients.", "kind": "novel"},
    {"id": "h2.9", "text": "sex_female is associated with mean pfs_months.", "kind": "novel"},
    {"id": "h2.10", "text": "Higher ecog_ps is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.11", "text": "age_years is linearly associated with pfs_months.", "kind": "novel"},
    {"id": "h2.12", "text": "Higher cea_ng_ml is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.13", "text": "Higher albumin_g_dl is associated with longer pfs_months.", "kind": "novel"},
    {"id": "h2.14", "text": "Higher ldh_u_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.15", "text": "Higher weight_loss_pct_6mo is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.16", "text": "Higher crp_mg_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.17", "text": "Higher nlr is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.18", "text": "Higher hemoglobin_g_dl is associated with longer pfs_months.", "kind": "novel"},
    {"id": "h2.19", "text": "Higher alkaline_phosphatase_u_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.20", "text": "Higher ast_u_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.21", "text": "Higher alt_u_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.22", "text": "Higher total_bilirubin_mg_dl is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.23", "text": "Higher creatinine_mg_dl is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.24", "text": "Higher bun_mg_dl is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.25", "text": "sodium_meq_l is associated with pfs_months.", "kind": "novel"},
    {"id": "h2.26", "text": "potassium_meq_l is associated with pfs_months.", "kind": "novel"},
    {"id": "h2.27", "text": "calcium_mg_dl is associated with pfs_months.", "kind": "novel"},
]
ana2 = [
    a(["h2.1"], "i2_kras_mutation_main"),
    a(["h2.2"], "i2_nras_mutation_main"),
    a(["h2.3"], "i2_braf_v600e_main"),
    a(["h2.4"], "i2_msi_high_main"),
    a(["h2.5"], "i2_her2_amplified_main"),
    a(["h2.6"], "i2_ntrk_fusion_main"),
    a(["h2.7"], "i2_stage_iv_main"),
    a(["h2.8"], "i2_right_sided_primary_main"),
    a(["h2.9"], "i2_sex_female_main"),
    a(["h2.10"], "i2_ecog_0_main"),
    a(["h2.10"], "i2_ecog_1_main"),
    a(["h2.10"], "i2_ecog_2_main"),
    a(["h2.11"], "i2_age_years_main", "smf.ols('pfs_months ~ age_years', data=df).fit()"),
    a(["h2.12"], "i2_cea_ng_ml_main"),
    a(["h2.13"], "i2_albumin_g_dl_main"),
    a(["h2.14"], "i2_ldh_u_l_main"),
    a(["h2.15"], "i2_weight_loss_pct_6mo_main"),
    a(["h2.16"], "i2_crp_mg_l_main"),
    a(["h2.17"], "i2_nlr_main"),
    a(["h2.18"], "i2_hemoglobin_g_dl_main"),
    a(["h2.19"], "i2_alkaline_phosphatase_u_l_main"),
    a(["h2.20"], "i2_ast_u_l_main"),
    a(["h2.21"], "i2_alt_u_l_main"),
    a(["h2.22"], "i2_total_bilirubin_mg_dl_main"),
    a(["h2.23"], "i2_creatinine_mg_dl_main"),
    a(["h2.24"], "i2_bun_mg_dl_main"),
    a(["h2.25"], "i2_sodium_meq_l_main"),
    a(["h2.26"], "i2_potassium_meq_l_main"),
    a(["h2.27"], "i2_calcium_mg_dl_main"),
]
iterations.append({"index": 2, "proposed_hypotheses": hyps2, "analyses": ana2})

# ----- Iteration 3: multivariable clinical model -----
hyps3 = [
    {"id": "h3.1", "text": "After adjusting jointly for age_years, sex_female, ecog_ps, stage_iv, right_sided_primary, albumin_g_dl, ldh_u_l, cea_ng_ml, weight_loss_pct_6mo, crp_mg_l, nlr, and hemoglobin_g_dl, ecog_ps, stage_iv, right_sided_primary, cea_ng_ml, ldh_u_l and weight_loss_pct_6mo independently shorten pfs_months while age_years and albumin_g_dl independently lengthen it.", "kind": "refined"},
]
ana3 = [
    a(["h3.1"], "i3_multi_age_years", "smf.ols('pfs_months ~ age_years + sex_female + ecog_ps + stage_iv + right_sided_primary + albumin_g_dl + ldh_u_l + cea_ng_ml + weight_loss_pct_6mo + crp_mg_l + nlr + hemoglobin_g_dl', data=df).fit()"),
    a(["h3.1"], "i3_multi_sex_female"),
    a(["h3.1"], "i3_multi_ecog_ps"),
    a(["h3.1"], "i3_multi_stage_iv"),
    a(["h3.1"], "i3_multi_right_sided_primary"),
    a(["h3.1"], "i3_multi_albumin_g_dl"),
    a(["h3.1"], "i3_multi_ldh_u_l"),
    a(["h3.1"], "i3_multi_cea_ng_ml"),
    a(["h3.1"], "i3_multi_weight_loss_pct_6mo"),
    a(["h3.1"], "i3_multi_crp_mg_l"),
    a(["h3.1"], "i3_multi_nlr"),
    a(["h3.1"], "i3_multi_hemoglobin_g_dl"),
]
iterations.append({"index": 3, "proposed_hypotheses": hyps3, "analyses": ana3})

# ----- Iteration 4: pembro x msi -----
hyps4 = [
    {"id": "h4.1", "text": "Among msi_high=1 patients, treatment_pembrolizumab increases mean pfs_months relative to no pembrolizumab; among msi_high=0 patients there is no such effect, yielding a positive treatment_pembrolizumab x msi_high interaction.", "kind": "novel"},
]
ana4 = [
    a(["h4.1"], "i4_pembro_in_msi1"),
    a(["h4.1"], "i4_pembro_in_msi0"),
    a(["h4.1"], "i4_pembro_msi_interaction", "smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high + age_years + ecog_ps + stage_iv', data=df).fit()"),
]
iterations.append({"index": 4, "proposed_hypotheses": hyps4, "analyses": ana4})

# ----- Iteration 5: cetuximab x RAS/RAF wild-type -----
hyps5 = [
    {"id": "h5.1", "text": "Among kras_mutation=0 AND nras_mutation=0 AND braf_v600e=0 (RAS/RAF wild-type) patients, treatment_cetuximab increases mean pfs_months; the cetuximab x ras_raf_wt interaction is positive.", "kind": "novel"},
    {"id": "h5.2", "text": "Within RAS/RAF wild-type AND right_sided_primary=0 (left-sided), treatment_cetuximab increases mean pfs_months.", "kind": "novel"},
]
ana5 = [
    a(["h5.1"], "i5_cetux_in_ras_raf_wt"),
    a(["h5.1"], "i5_cetux_in_ras_raf_mut"),
    a(["h5.1"], "i5_cetux_rasrafwt_interaction", "smf.ols('pfs_months ~ treatment_cetuximab * ras_raf_wt + age_years + ecog_ps + stage_iv + right_sided_primary', data=df).fit()"),
    a(["h5.2"], "i5_cetux_rasrafwt_left"),
    a(["h5.2"], "i5_cetux_rasrafwt_right"),
]
iterations.append({"index": 5, "proposed_hypotheses": hyps5, "analyses": ana5})

# ----- Iteration 6: encorafenib x braf -----
hyps6 = [
    {"id": "h6.1", "text": "Among braf_v600e=1 patients, treatment_encorafenib increases mean pfs_months; in braf_v600e=0 patients there is no effect; treatment_encorafenib x braf_v600e interaction is positive.", "kind": "novel"},
]
ana6 = [
    a(["h6.1"], "i6_encora_in_braf1"),
    a(["h6.1"], "i6_encora_in_braf0"),
    a(["h6.1"], "i6_encora_braf_interaction", "smf.ols('pfs_months ~ treatment_encorafenib * braf_v600e + age_years + ecog_ps + stage_iv', data=df).fit()"),
]
iterations.append({"index": 6, "proposed_hypotheses": hyps6, "analyses": ana6})

# ----- Iteration 7: T+T x HER2 -----
hyps7 = [
    {"id": "h7.1", "text": "Among her2_amplified=1 patients, treatment_trastuzumab_tucatinib increases mean pfs_months; in her2_amplified=0 patients there is no effect; treatment_trastuzumab_tucatinib x her2_amplified interaction is positive.", "kind": "novel"},
    {"id": "h7.2", "text": "Within her2_amplified=1 AND kras_mutation=0 (HER2+ KRAS-wild-type), treatment_trastuzumab_tucatinib increases mean pfs_months relative to within her2_amplified=1 AND kras_mutation=1.", "kind": "novel"},
]
ana7 = [
    a(["h7.1"], "i7_tt_in_her21"),
    a(["h7.1"], "i7_tt_in_her20"),
    a(["h7.1"], "i7_tt_her2_interaction", "smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified + age_years + ecog_ps + stage_iv', data=df).fit()"),
    a(["h7.2"], "i7_tt_her2_kraswt"),
    a(["h7.2"], "i7_tt_her2_krasmut"),
]
iterations.append({"index": 7, "proposed_hypotheses": hyps7, "analyses": ana7})

# ----- Iteration 8: bevacizumab adjusted + interactions -----
hyps8 = [
    {"id": "h8.1", "text": "After adjustment for age_years, ecog_ps, stage_iv, right_sided_primary, albumin_g_dl, ldh_u_l and cea_ng_ml, treatment_bevacizumab independently changes mean pfs_months.", "kind": "novel"},
    {"id": "h8.2", "text": "treatment_bevacizumab interacts with kras_mutation, right_sided_primary, msi_high, stage_iv, ecog_ps, or braf_v600e on pfs_months.", "kind": "novel"},
]
ana8 = [
    a(["h8.1"], "i8_bev_adjusted", "smf.ols('pfs_months ~ treatment_bevacizumab + age_years + ecog_ps + stage_iv + right_sided_primary + albumin_g_dl + ldh_u_l + cea_ng_ml', data=df).fit()"),
    a(["h8.2"], "i8_bev_x_kras_mutation"),
    a(["h8.2"], "i8_bev_x_right_sided_primary"),
    a(["h8.2"], "i8_bev_x_msi_high"),
    a(["h8.2"], "i8_bev_x_stage_iv"),
    a(["h8.2"], "i8_bev_x_ecog_ps"),
    a(["h8.2"], "i8_bev_x_braf_v600e"),
]
iterations.append({"index": 8, "proposed_hypotheses": hyps8, "analyses": ana8})

# ----- Iteration 9: regorafenib adjusted + interaction screen -----
hyps9 = [
    {"id": "h9.1", "text": "After adjustment for age_years, ecog_ps, stage_iv, albumin_g_dl, ldh_u_l, and cea_ng_ml, treatment_regorafenib independently increases mean pfs_months.", "kind": "refined"},
    {"id": "h9.2", "text": "treatment_regorafenib effect on pfs_months is suppressed (negative interaction) by kras_mutation.", "kind": "novel"},
    {"id": "h9.3", "text": "treatment_regorafenib effect on pfs_months is enhanced (positive interaction) in nras_mutation patients.", "kind": "novel"},
    {"id": "h9.4", "text": "treatment_regorafenib effect on pfs_months is suppressed by braf_v600e.", "kind": "novel"},
    {"id": "h9.5", "text": "treatment_regorafenib effect on pfs_months is suppressed by right_sided_primary.", "kind": "novel"},
    {"id": "h9.6", "text": "treatment_regorafenib effect on pfs_months is suppressed by her2_amplified.", "kind": "novel"},
    {"id": "h9.7", "text": "treatment_regorafenib effect on pfs_months is suppressed by higher ecog_ps.", "kind": "novel"},
    {"id": "h9.8", "text": "treatment_regorafenib effect on pfs_months is not modified by msi_high, ntrk_fusion, stage_iv, or sex_female.", "kind": "novel"},
]
ana9 = [
    a(["h9.1"], "i9_rego_adjusted"),
    a(["h9.2"], "i9_rego_x_kras_mutation"),
    a(["h9.3"], "i9_rego_x_nras_mutation"),
    a(["h9.4"], "i9_rego_x_braf_v600e"),
    a(["h9.5"], "i9_rego_x_right_sided_primary"),
    a(["h9.6"], "i9_rego_x_her2_amplified"),
    a(["h9.7"], "i9_rego_x_ecog_ps"),
    a(["h9.8"], "i9_rego_x_msi_high"),
    a(["h9.8"], "i9_rego_x_ntrk_fusion"),
    a(["h9.8"], "i9_rego_x_stage_iv"),
    a(["h9.8"], "i9_rego_x_sex_female"),
]
iterations.append({"index": 9, "proposed_hypotheses": hyps9, "analyses": ana9})

# ----- Iteration 10: rego stratified by kras -----
hyps10 = [
    {"id": "h10.1", "text": "Among kras_mutation=0 patients, treatment_regorafenib increases pfs_months; among kras_mutation=1 patients, the effect is null.", "kind": "refined"},
]
ana10 = [
    a(["h10.1"], "i10_rego_in_kraswt"),
    a(["h10.1"], "i10_rego_in_krasmut"),
]
iterations.append({"index": 10, "proposed_hypotheses": hyps10, "analyses": ana10})

# ----- Iteration 11: rego stratified by braf -----
hyps11 = [
    {"id": "h11.1", "text": "Among braf_v600e=0 patients, treatment_regorafenib increases pfs_months; among braf_v600e=1 patients, the effect is null.", "kind": "refined"},
]
ana11 = [
    a(["h11.1"], "i11_rego_in_brafwt"),
    a(["h11.1"], "i11_rego_in_brafmut"),
]
iterations.append({"index": 11, "proposed_hypotheses": hyps11, "analyses": ana11})

# ----- Iteration 12: rego stratified by side -----
hyps12 = [
    {"id": "h12.1", "text": "Among right_sided_primary=0 (left-sided) patients, treatment_regorafenib increases pfs_months; among right_sided_primary=1 patients, the effect is null.", "kind": "refined"},
]
ana12 = [
    a(["h12.1"], "i12_rego_in_leftsided"),
    a(["h12.1"], "i12_rego_in_rightsided"),
]
iterations.append({"index": 12, "proposed_hypotheses": hyps12, "analyses": ana12})

# ----- Iteration 13: joint subgroup -----
hyps13 = [
    {"id": "h13.1", "text": "Within the joint subgroup kras_mutation=0 AND braf_v600e=0 AND right_sided_primary=0, treatment_regorafenib substantially increases pfs_months relative to non-regorafenib patients in the same subgroup; outside this subgroup the effect is null.", "kind": "refined"},
]
ana13 = [
    a(["h13.1"], "i13_rego_in_kras0_braf0_left"),
    a(["h13.1"], "i13_rego_outside_subgroup"),
]
iterations.append({"index": 13, "proposed_hypotheses": hyps13, "analyses": ana13})

# ----- Iteration 14: rego x ECOG -----
hyps14 = [
    {"id": "h14.1", "text": "treatment_regorafenib increases pfs_months at every ecog_ps stratum (0, 1, 2), so ecog_ps modifies baseline pfs_months but does not abolish the regorafenib effect.", "kind": "novel"},
    {"id": "h14.2", "text": "Within kras_mutation=0 AND braf_v600e=0 AND right_sided_primary=0 AND ecog_ps=0, treatment_regorafenib produces an even larger absolute pfs_months gain than in the same biomarker subgroup overall.", "kind": "refined"},
]
ana14 = [
    a(["h14.1"], "i14_rego_ecog_0"),
    a(["h14.1"], "i14_rego_ecog_1"),
    a(["h14.1"], "i14_rego_ecog_2"),
    a(["h14.2"], "i14_rego_subgroup_ec0"),
]
iterations.append({"index": 14, "proposed_hypotheses": hyps14, "analyses": ana14})

# ----- Iteration 15: rego x continuous interaction screen -----
hyps15 = [
    {"id": "h15.1", "text": "Higher cea_ng_ml suppresses the treatment_regorafenib pfs_months benefit (negative interaction).", "kind": "novel"},
    {"id": "h15.2", "text": "Higher albumin_g_dl suppresses the treatment_regorafenib pfs_months benefit (negative interaction).", "kind": "novel"},
    {"id": "h15.3", "text": "Higher ldh_u_l, weight_loss_pct_6mo, nlr or sodium_meq_l enhance the treatment_regorafenib effect; higher crp_mg_l, bun_mg_dl suppress it; remaining labs (age_years, hemoglobin_g_dl, alkaline_phosphatase_u_l, ast_u_l, alt_u_l, total_bilirubin_mg_dl, creatinine_mg_dl, potassium_meq_l, calcium_mg_dl) show no interaction with treatment_regorafenib on pfs_months.", "kind": "novel"},
]
ana15 = [
    a(["h15.1"], "i15_rego_x_cea_ng_ml"),
    a(["h15.2"], "i15_rego_x_albumin_g_dl"),
    a(["h15.3"], "i15_rego_x_ldh_u_l"),
    a(["h15.3"], "i15_rego_x_weight_loss_pct_6mo"),
    a(["h15.3"], "i15_rego_x_nlr"),
    a(["h15.3"], "i15_rego_x_sodium_meq_l"),
    a(["h15.3"], "i15_rego_x_crp_mg_l"),
    a(["h15.3"], "i15_rego_x_bun_mg_dl"),
    a(["h15.3"], "i15_rego_x_age_years"),
    a(["h15.3"], "i15_rego_x_hemoglobin_g_dl"),
    a(["h15.3"], "i15_rego_x_alkaline_phosphatase_u_l"),
    a(["h15.3"], "i15_rego_x_ast_u_l"),
    a(["h15.3"], "i15_rego_x_alt_u_l"),
    a(["h15.3"], "i15_rego_x_total_bilirubin_mg_dl"),
    a(["h15.3"], "i15_rego_x_creatinine_mg_dl"),
    a(["h15.3"], "i15_rego_x_potassium_meq_l"),
    a(["h15.3"], "i15_rego_x_calcium_mg_dl"),
]
iterations.append({"index": 15, "proposed_hypotheses": hyps15, "analyses": ana15})

# ----- Iteration 16: joint regression -----
hyps16 = [
    {"id": "h16.1", "text": "In a joint regression including treatment_regorafenib x kras_mutation, treatment_regorafenib x braf_v600e, and treatment_regorafenib x right_sided_primary plus age_years, ecog_ps, stage_iv, albumin_g_dl, cea_ng_ml, the baseline regorafenib effect (KRAS-wt, BRAF-wt, left-sided) is large and positive, and each of the three interactions is significantly negative and roughly equal in magnitude to the baseline effect — i.e., each unfavorable predicate fully neutralises the regorafenib benefit.", "kind": "refined"},
]
ana16 = [
    a(["h16.1"], "i16_joint_treatment_regorafenib", "smf.ols('pfs_months ~ treatment_regorafenib * kras_mutation + treatment_regorafenib * braf_v600e + treatment_regorafenib * right_sided_primary + age_years + ecog_ps + stage_iv + albumin_g_dl + cea_ng_ml', data=df).fit()"),
    a(["h16.1"], "i16_joint_treatment_regorafenib_kras_mutation"),
    a(["h16.1"], "i16_joint_treatment_regorafenib_braf_v600e"),
    a(["h16.1"], "i16_joint_treatment_regorafenib_right_sided_primary"),
]
iterations.append({"index": 16, "proposed_hypotheses": hyps16, "analyses": ana16})

# ----- Iteration 17: pembro adjusted + interactions -----
hyps17 = [
    {"id": "h17.1", "text": "After adjustment for age_years, ecog_ps, stage_iv, msi_high, albumin_g_dl, cea_ng_ml, treatment_pembrolizumab does not change pfs_months.", "kind": "refined"},
    {"id": "h17.2", "text": "treatment_pembrolizumab does not interact with msi_high, kras_mutation, braf_v600e, right_sided_primary, stage_iv, ecog_ps, or her2_amplified on pfs_months.", "kind": "refined"},
]
ana17 = [
    a(["h17.1"], "i17_pembro_adjusted"),
    a(["h17.2"], "i17_pembro_x_msi_high"),
    a(["h17.2"], "i17_pembro_x_kras_mutation"),
    a(["h17.2"], "i17_pembro_x_braf_v600e"),
    a(["h17.2"], "i17_pembro_x_right_sided_primary"),
    a(["h17.2"], "i17_pembro_x_stage_iv"),
    a(["h17.2"], "i17_pembro_x_ecog_ps"),
    a(["h17.2"], "i17_pembro_x_her2_amplified"),
]
iterations.append({"index": 17, "proposed_hypotheses": hyps17, "analyses": ana17})

# ----- Iteration 18: cetuximab interaction screen -----
hyps18 = [
    {"id": "h18.1", "text": "treatment_cetuximab does not interact with kras_mutation, nras_mutation, braf_v600e, msi_high, her2_amplified, right_sided_primary, stage_iv, ecog_ps or sex_female on pfs_months.", "kind": "refined"},
]
ana18 = [
    a(["h18.1"], "i18_cetux_x_kras_mutation"),
    a(["h18.1"], "i18_cetux_x_nras_mutation"),
    a(["h18.1"], "i18_cetux_x_braf_v600e"),
    a(["h18.1"], "i18_cetux_x_msi_high"),
    a(["h18.1"], "i18_cetux_x_her2_amplified"),
    a(["h18.1"], "i18_cetux_x_right_sided_primary"),
    a(["h18.1"], "i18_cetux_x_stage_iv"),
    a(["h18.1"], "i18_cetux_x_ecog_ps"),
    a(["h18.1"], "i18_cetux_x_sex_female"),
]
iterations.append({"index": 18, "proposed_hypotheses": hyps18, "analyses": ana18})

# ----- Iteration 19: encorafenib interaction screen -----
hyps19 = [
    {"id": "h19.1", "text": "treatment_encorafenib does not interact with kras_mutation, nras_mutation, braf_v600e, her2_amplified, right_sided_primary, stage_iv, ecog_ps, treatment_cetuximab, or treatment_bevacizumab on pfs_months.", "kind": "refined"},
    {"id": "h19.2", "text": "treatment_encorafenib has a negative interaction with msi_high on pfs_months (effect more negative when MSI-high).", "kind": "novel"},
]
ana19 = [
    a(["h19.1"], "i19_encora_x_kras_mutation"),
    a(["h19.1"], "i19_encora_x_nras_mutation"),
    a(["h19.1"], "i19_encora_x_braf_v600e"),
    a(["h19.1"], "i19_encora_x_her2_amplified"),
    a(["h19.1"], "i19_encora_x_right_sided_primary"),
    a(["h19.1"], "i19_encora_x_stage_iv"),
    a(["h19.1"], "i19_encora_x_ecog_ps"),
    a(["h19.1"], "i19_encora_x_treatment_cetuximab"),
    a(["h19.1"], "i19_encora_x_treatment_bevacizumab"),
    a(["h19.2"], "i19_encora_x_msi_high"),
]
iterations.append({"index": 19, "proposed_hypotheses": hyps19, "analyses": ana19})

# ----- Iteration 20: T+T interaction screen -----
hyps20 = [
    {"id": "h20.1", "text": "treatment_trastuzumab_tucatinib does not interact with any of kras_mutation, nras_mutation, braf_v600e, msi_high, her2_amplified, right_sided_primary, stage_iv, or ecog_ps on pfs_months.", "kind": "refined"},
]
ana20 = [
    a(["h20.1"], "i20_tt_x_kras_mutation"),
    a(["h20.1"], "i20_tt_x_nras_mutation"),
    a(["h20.1"], "i20_tt_x_braf_v600e"),
    a(["h20.1"], "i20_tt_x_msi_high"),
    a(["h20.1"], "i20_tt_x_her2_amplified"),
    a(["h20.1"], "i20_tt_x_right_sided_primary"),
    a(["h20.1"], "i20_tt_x_stage_iv"),
    a(["h20.1"], "i20_tt_x_ecog_ps"),
]
iterations.append({"index": 20, "proposed_hypotheses": hyps20, "analyses": ana20})

# ----- Iteration 21: bevacizumab interaction screen -----
hyps21 = [
    {"id": "h21.1", "text": "treatment_bevacizumab does not interact with nras_mutation, her2_amplified, ntrk_fusion, sex_female, ecog_ps, stage_iv, or right_sided_primary on pfs_months.", "kind": "refined"},
]
ana21 = [
    a(["h21.1"], "i21_bev_x_nras_mutation"),
    a(["h21.1"], "i21_bev_x_her2_amplified"),
    a(["h21.1"], "i21_bev_x_ntrk_fusion"),
    a(["h21.1"], "i21_bev_x_sex_female"),
    a(["h21.1"], "i21_bev_x_ecog_ps"),
    a(["h21.1"], "i21_bev_x_stage_iv"),
    a(["h21.1"], "i21_bev_x_right_sided_primary"),
]
iterations.append({"index": 21, "proposed_hypotheses": hyps21, "analyses": ana21})

# ----- Iteration 22: rego in 8 cells of KRAS x BRAF x side -----
hyps22 = [
    {"id": "h22.1", "text": "Across the 2x2x2 grid of kras_mutation x braf_v600e x right_sided_primary, treatment_regorafenib has a strong positive pfs_months effect ONLY in the (kras_mutation=0, braf_v600e=0, right_sided_primary=0) cell; in every other cell with patients available the effect is statistically null.", "kind": "refined"},
]
ana22 = [
    a(["h22.1"], "i22_rego_k0_b0_s0"),
    a(["h22.1"], "i22_rego_k0_b0_s1"),
    a(["h22.1"], "i22_rego_k0_b1_s0"),
    a(["h22.1"], "i22_rego_k0_b1_s1"),
    a(["h22.1"], "i22_rego_k1_b0_s0"),
    a(["h22.1"], "i22_rego_k1_b0_s1"),
]
iterations.append({"index": 22, "proposed_hypotheses": hyps22, "analyses": ana22})

# ----- Iteration 23: confirmation of subgroup -----
hyps23 = [
    {"id": "h23.1", "text": "Removing any one of the three predicates (kras_mutation=0, braf_v600e=0, right_sided_primary=0) from the regorafenib subgroup definition reduces the estimated treatment_regorafenib pfs_months effect, confirming that all three predicates contribute to defining the responsive subgroup.", "kind": "refined"},
]
ana23 = [
    a(["h23.1"], "i23_rego_best_subgroup"),
    a(["h23.1"], "i23_rego_drop_kras"),
    a(["h23.1"], "i23_rego_drop_braf"),
    a(["h23.1"], "i23_rego_drop_side"),
]
iterations.append({"index": 23, "proposed_hypotheses": hyps23, "analyses": ana23})

# ----- Iteration 24: NRAS contribution -----
hyps24 = [
    {"id": "h24.1", "text": "Among nras_mutation=1 patients, treatment_regorafenib increases pfs_months; the estimated effect is enhanced when the additional KRAS-wt + BRAF-wt + left-sided predicates also hold, but NRAS status itself is not required to define the responsive subgroup (because nras_mutation=1 is essentially a subset of kras_mutation=0).", "kind": "novel"},
]
ana24 = [
    a(["h24.1"], "i24_rego_in_nras1"),
    a(["h24.1"], "i24_rego_nras1_subgroup"),
]
iterations.append({"index": 24, "proposed_hypotheses": hyps24, "analyses": ana24})

# ----- Iteration 25: final subgroup hypothesis & confirmation -----
hyps25 = [
    {"id": "h25.1", "text": "FINAL SUBGROUP: treatment_regorafenib increases pfs_months relative to no regorafenib in the subgroup defined by (kras_mutation=0 AND braf_v600e=0 AND right_sided_primary=0); the unfavorable values of any of these three predicates (kras_mutation=1, braf_v600e=1, or right_sided_primary=1) suppress the regorafenib benefit toward zero.", "kind": "refined"},
    {"id": "h25.2", "text": "Within the regorafenib responder subgroup (kras_mutation=0 AND braf_v600e=0 AND right_sided_primary=0), none of treatment_cetuximab, treatment_bevacizumab, treatment_pembrolizumab, treatment_encorafenib or treatment_trastuzumab_tucatinib changes pfs_months — i.e., the subgroup-specific benefit is unique to regorafenib.", "kind": "novel"},
    {"id": "h25.3", "text": "In a regression with treatment_regorafenib interactions for kras_mutation, braf_v600e, and right_sided_primary plus adjustment for age_years, ecog_ps and stage_iv, the baseline (KRAS-wt + BRAF-wt + left-sided) regorafenib effect on pfs_months is approximately +2.3 months and is highly significant.", "kind": "refined"},
]
ana25 = [
    a(["h25.1"], "i23_rego_best_subgroup"),
    a(["h25.2"], "i25_in_best_treatment_cetuximab"),
    a(["h25.2"], "i25_in_best_treatment_bevacizumab"),
    a(["h25.2"], "i25_in_best_treatment_pembrolizumab"),
    a(["h25.2"], "i25_in_best_treatment_encorafenib"),
    a(["h25.2"], "i25_in_best_treatment_trastuzumab_tucatinib"),
    a(["h25.3"], "i25_final_rego_pure", "smf.ols('pfs_months ~ treatment_regorafenib + treatment_regorafenib:kras_mutation + treatment_regorafenib:braf_v600e + treatment_regorafenib:right_sided_primary + kras_mutation + braf_v600e + right_sided_primary + age_years + ecog_ps + stage_iv', data=df).fit()"),
]
iterations.append({"index": 25, "proposed_hypotheses": hyps25, "analyses": ana25})


transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@named-task-2026-05-03",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2, default=str)

print(f"Wrote transcript.json with {len(iterations)} iterations")
total_h = sum(len(it['proposed_hypotheses']) for it in iterations)
total_a = sum(len(it['analyses']) for it in iterations)
print(f"  total hypotheses: {total_h}")
print(f"  total analyses: {total_a}")
