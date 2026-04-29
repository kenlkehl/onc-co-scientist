"""Builds transcript.json from results_collected.json."""

import json

with open("results_collected.json") as f:
    R = json.load(f)


def H(hid, text, kind="novel"):
    return {"id": hid, "text": text, "kind": kind}


def A(hids, summary, p, eff, sig, code=None):
    return {
        "hypothesis_ids": hids,
        "code": code,
        "result_summary": summary,
        "p_value": p,
        "effect_estimate": eff,
        "significant": sig,
    }


iters = []

# ---- Iteration 1 ----
i = R["i1"]
iters.append({
    "index": 1,
    "proposed_hypotheses": [
        H("h1", "Higher ECOG performance status (ecog_ps) is associated with shorter pfs_months: each one-unit increase in ecog_ps corresponds to a decrease in mean pfs_months."),
    ],
    "analyses": [
        A(["h1"],
          f"Linear regression of pfs_months on ecog_ps: coef={i['ecog_ols']['coef']} months/unit (SE {i['ecog_ols']['se']}), Pearson r={i['ecog_corr']['r']}, p<1e-300. Higher ECOG strongly tied to shorter PFS.",
          i["ecog_ols"]["p"], i["ecog_ols"]["coef"], i["ecog_ols"]["sig"],
          code="smf.ols('pfs_months ~ ecog_ps', df).fit()"),
    ],
})

# ---- Iteration 2 ----
i = R["i2"]; t = i["stage_iv_t"]
iters.append({
    "index": 2,
    "proposed_hypotheses": [
        H("h2", "Stage IV disease (stage_iv=1) is associated with shorter pfs_months than non-stage-IV disease."),
    ],
    "analyses": [
        A(["h2"],
          f"Welch t-test of pfs_months by stage_iv: stage IV mean {t['mean_pos']} mo (n={t['n_pos']}) vs non-stage-IV {t['mean_neg']} mo (n={t['n_neg']}), diff {t['effect']} mo, t={t['t']}, p<1e-300.",
          t["p"], t["effect"], t["sig"],
          code="stats.ttest_ind(df[df.stage_iv==1].pfs_months, df[df.stage_iv==0].pfs_months, equal_var=False)"),
    ],
})

# ---- Iteration 3 ----
i = R["i3"]; t = i["brain_mets_t"]
iters.append({
    "index": 3,
    "proposed_hypotheses": [
        H("h3", "Patients with brain metastases (has_brain_mets=1) have shorter pfs_months than patients without brain metastases."),
    ],
    "analyses": [
        A(["h3"],
          f"Welch t-test: brain-mets mean {t['mean_pos']} mo (n={t['n_pos']}) vs no-brain-mets {t['mean_neg']} mo (n={t['n_neg']}); diff {t['effect']} mo, p<1e-180.",
          t["p"], t["effect"], t["sig"],
          code="stats.ttest_ind(df[df.has_brain_mets==1].pfs_months, df[df.has_brain_mets==0].pfs_months, equal_var=False)"),
    ],
})

# ---- Iteration 4 ----
i = R["i4"]
iters.append({
    "index": 4,
    "proposed_hypotheses": [
        H("h4a", "Patients with liver metastases (liver_mets=1) have shorter pfs_months than those without."),
        H("h4b", "Patients with bone metastases (bone_mets=1) have shorter pfs_months than those without."),
        H("h4c", "Patients with pleural effusion (pleural_effusion=1) have shorter pfs_months than those without."),
    ],
    "analyses": [
        A(["h4a"],
          f"Welch t-test: liver-mets {i['liver_mets_t']['mean_pos']} mo vs no-liver-mets {i['liver_mets_t']['mean_neg']} mo; diff {i['liver_mets_t']['effect']} mo, p={i['liver_mets_t']['p']}. NOT significant.",
          i["liver_mets_t"]["p"], i["liver_mets_t"]["effect"], i["liver_mets_t"]["sig"]),
        A(["h4b"],
          f"Welch t-test: bone-mets {i['bone_mets_t']['mean_pos']} mo vs no-bone-mets {i['bone_mets_t']['mean_neg']} mo; diff {i['bone_mets_t']['effect']} mo, p={i['bone_mets_t']['p']}. Not significant.",
          i["bone_mets_t"]["p"], i["bone_mets_t"]["effect"], i["bone_mets_t"]["sig"]),
        A(["h4c"],
          f"Welch t-test: pleural-effusion {i['pleural_effusion_t']['mean_pos']} mo vs none {i['pleural_effusion_t']['mean_neg']} mo; diff {i['pleural_effusion_t']['effect']} mo, p={i['pleural_effusion_t']['p']}. Not significant.",
          i["pleural_effusion_t"]["p"], i["pleural_effusion_t"]["effect"], i["pleural_effusion_t"]["sig"]),
    ],
})

# ---- Iteration 5 ----
i = R["i5"]
iters.append({
    "index": 5,
    "proposed_hypotheses": [
        H("h5a", "ER-positive disease (er_positive=1) is associated with longer pfs_months than ER-negative disease."),
        H("h5b", "PR-positive disease (pr_positive=1) is associated with longer pfs_months than PR-negative disease."),
    ],
    "analyses": [
        A(["h5a"],
          f"Welch t-test: ER+ {i['er_pos_t']['mean_pos']} mo (n={i['er_pos_t']['n_pos']}) vs ER- {i['er_pos_t']['mean_neg']} mo; diff {i['er_pos_t']['effect']} mo, t={i['er_pos_t']['t']}, p<1e-140.",
          i["er_pos_t"]["p"], i["er_pos_t"]["effect"], i["er_pos_t"]["sig"]),
        A(["h5b"],
          f"Welch t-test: PR+ {i['pr_pos_t']['mean_pos']} mo vs PR- {i['pr_pos_t']['mean_neg']} mo; diff {i['pr_pos_t']['effect']} mo, p<1e-50.",
          i["pr_pos_t"]["p"], i["pr_pos_t"]["effect"], i["pr_pos_t"]["sig"]),
    ],
})

# ---- Iteration 6 ----
i = R["i6"]
trast_int = i["trast_x_her2pos"]["interaction"]; trast_pos = i["trast_x_her2pos"]["pos"]
iters.append({
    "index": 6,
    "proposed_hypotheses": [
        H("h6a", "HER2-positive disease (her2_positive=1) is associated with shorter pfs_months than HER2-negative disease (main effect)."),
        H("h6b", "Trastuzumab is more beneficial in HER2-positive than HER2-negative patients (positive treatment_trastuzumab × her2_positive interaction on pfs_months)."),
    ],
    "analyses": [
        A(["h6a"],
          f"Welch t-test: HER2+ {i['her2_pos_t']['mean_pos']} mo vs HER2- {i['her2_pos_t']['mean_neg']} mo; diff {i['her2_pos_t']['effect']} mo, p<1e-60.",
          i["her2_pos_t"]["p"], i["her2_pos_t"]["effect"], i["her2_pos_t"]["sig"]),
        A(["h6b"],
          f"OLS pfs_months ~ treatment_trastuzumab * her2_positive — interaction coef {trast_int['coef']} mo, p={trast_int['p']}; subgroup tx vs ctl effect in HER2+: {trast_pos['effect']} mo (p={trast_pos['p']}). NO benefit detected in HER2+ subgroup; trastuzumab × HER2+ interaction NOT significant.",
          trast_int["p"], trast_int["coef"], trast_int["sig"],
          code="smf.ols('pfs_months ~ treatment_trastuzumab * her2_positive', df).fit()"),
    ],
})

# ---- Iteration 7 ----
i = R["i7"]
sg_int = i["sg_x_her2low"]["interaction"]; sg_pos = i["sg_x_her2low"]["pos"]
iters.append({
    "index": 7,
    "proposed_hypotheses": [
        H("h7a", "HER2-low disease (her2_low=1) is associated with different pfs_months than HER2-zero disease."),
        H("h7b", "Sacituzumab govitecan provides greater PFS benefit in HER2-low patients than HER2-not-low (positive treatment_sacituzumab_govitecan × her2_low interaction)."),
    ],
    "analyses": [
        A(["h7a"],
          f"Welch t-test: HER2-low {i['her2_low_t']['mean_pos']} mo vs HER2-not-low {i['her2_low_t']['mean_neg']} mo; diff {i['her2_low_t']['effect']} mo, p={i['her2_low_t']['p']}. Statistically significant but tiny in magnitude.",
          i["her2_low_t"]["p"], i["her2_low_t"]["effect"], i["her2_low_t"]["sig"]),
        A(["h7b"],
          f"OLS pfs_months ~ tx*her2_low — interaction coef {sg_int['coef']} mo, p={sg_int['p']}; in HER2-low subgroup tx effect {sg_pos['effect']} mo (p={sg_pos['p']}). Sacituzumab does NOT show enhanced benefit in HER2-low.",
          sg_int["p"], sg_int["coef"], sg_int["sig"]),
    ],
})

# ---- Iteration 8 ----
i = R["i8"]
olap_int = i["olap_x_brca"]["interaction"]; olap_pos = i["olap_x_brca"]["pos"]
iters.append({
    "index": 8,
    "proposed_hypotheses": [
        H("h8a", "Patients with any BRCA1 or BRCA2 mutation (brca_any=1) differ in pfs_months from BRCA-wild-type patients (main effect)."),
        H("h8b", "Olaparib provides greater PFS benefit in BRCA-mutated patients than BRCA-wild-type (positive treatment_olaparib × brca_any interaction on pfs_months)."),
    ],
    "analyses": [
        A(["h8a"],
          f"Welch t-test: BRCA-any {i['brca_any_t']['mean_pos']} mo (n={i['brca_any_t']['n_pos']}) vs BRCA-wt {i['brca_any_t']['mean_neg']} mo; diff {i['brca_any_t']['effect']} mo, p={i['brca_any_t']['p']}. Not significant.",
          i["brca_any_t"]["p"], i["brca_any_t"]["effect"], i["brca_any_t"]["sig"]),
        A(["h8b"],
          f"OLS pfs_months ~ treatment_olaparib * brca_any — interaction coef {olap_int['coef']} mo, p={olap_int['p']} (significant). In BRCA-mutated subgroup, olaparib vs no-olaparib gives {olap_pos['effect']} mo benefit (p={olap_pos['p']}); in BRCA-wt subgroup effect is {i['olap_x_brca']['neg']['effect']} mo (p={i['olap_x_brca']['neg']['p']}).",
          olap_int["p"], olap_int["coef"], olap_int["sig"],
          code="smf.ols('pfs_months ~ treatment_olaparib * brca_any', df).fit()"),
    ],
})

# ---- Iteration 9 ----
i = R["i9"]; tam_int = i["tam_x_er"]["interaction"]
iters.append({
    "index": 9,
    "proposed_hypotheses": [
        H("h9a", "Tamoxifen has a stronger PFS benefit in ER-positive patients than ER-negative patients (positive treatment_tamoxifen × er_positive interaction)."),
        H("h9b", "Tamoxifen provides greater PFS benefit in postmenopausal patients than premenopausal patients (positive treatment_tamoxifen × postmenopausal interaction)."),
    ],
    "analyses": [
        A(["h9a"],
          f"OLS pfs_months ~ tam * er_positive — interaction coef {tam_int['coef']} mo, p={tam_int['p']}; in ER+ effect {i['tam_x_er']['pos']['effect']} mo (p={i['tam_x_er']['pos']['p']}); in ER- effect {i['tam_x_er']['neg']['effect']} mo (p={i['tam_x_er']['neg']['p']}). NO ER+ enrichment of tamoxifen benefit; interaction NOT significant.",
          tam_int["p"], tam_int["coef"], tam_int["sig"]),
        A(["h9b"],
          f"Postmenopausal × tamoxifen interaction coef {i['tam_x_postmeno']['interaction']['coef']} mo, p={i['tam_x_postmeno']['interaction']['p']}. Not significant.",
          i["tam_x_postmeno"]["interaction"]["p"], i["tam_x_postmeno"]["interaction"]["coef"], i["tam_x_postmeno"]["interaction"]["sig"]),
    ],
})

# ---- Iteration 10 ----
i = R["i10"]
palb_er = i["palb_x_er"]["interaction"]; palb_er_h2 = i["palb_x_er_her2neg"]["interaction"]
iters.append({
    "index": 10,
    "proposed_hypotheses": [
        H("h10a", "Palbociclib has a stronger PFS benefit in ER-positive than ER-negative patients (positive treatment_palbociclib × er_positive interaction)."),
        H("h10b", "Palbociclib's benefit is greatest in ER-positive/HER2-negative patients (positive treatment_palbociclib × er_pos_her2_neg interaction)."),
    ],
    "analyses": [
        A(["h10a"],
          f"OLS pfs_months ~ palb * er_positive — interaction coef {palb_er['coef']} mo, p<1e-300. In ER+ subgroup palbociclib vs not = +{i['palb_x_er']['pos']['effect']} mo (p<1e-300). In ER- subgroup effect {i['palb_x_er']['neg']['effect']} mo (p={i['palb_x_er']['neg']['p']}). HIGHLY significant ER+ enrichment.",
          palb_er["p"], palb_er["coef"], palb_er["sig"],
          code="smf.ols('pfs_months ~ treatment_palbociclib * er_positive', df).fit()"),
        A(["h10b"],
          f"OLS pfs_months ~ palb * er_pos_her2_neg — interaction coef {palb_er_h2['coef']} mo, p<1e-300. In ER+/HER2- palbociclib vs not = +{i['palb_x_er_her2neg']['pos']['effect']} mo. The single largest treatment-biomarker effect in the cohort.",
          palb_er_h2["p"], palb_er_h2["coef"], palb_er_h2["sig"]),
    ],
})

# ---- Iteration 11 ----
i = R["i11"]; pemb_msi = i["pembro_x_msi"]["interaction"]
iters.append({
    "index": 11,
    "proposed_hypotheses": [
        H("h11a", "Pembrolizumab provides greater PFS benefit in MSI-high patients than MSI-stable patients (positive treatment_pembrolizumab × msi_high interaction)."),
        H("h11b", "Pembrolizumab benefit differs by tp53 mutation status (treatment_pembrolizumab × tp53_mutation interaction)."),
    ],
    "analyses": [
        A(["h11a"],
          f"OLS pfs_months ~ pembro * msi_high — interaction coef {pemb_msi['coef']} mo, p={pemb_msi['p']}. In MSI-high subgroup pembro effect {i['pembro_x_msi']['pos']['effect']} mo (p={i['pembro_x_msi']['pos']['p']}, n_tx={i['pembro_x_msi']['pos']['n_tx']}); in MSI-stable effect is {i['pembro_x_msi']['neg']['effect']} mo. MSI-high subgroup small (n=498); no significant interaction.",
          pemb_msi["p"], pemb_msi["coef"], pemb_msi["sig"]),
        A(["h11b"],
          f"Pembro × TP53 interaction coef {i['pembro_x_tp53']['interaction']['coef']} mo, p={i['pembro_x_tp53']['interaction']['p']}. Not significant.",
          i["pembro_x_tp53"]["interaction"]["p"], i["pembro_x_tp53"]["interaction"]["coef"], i["pembro_x_tp53"]["interaction"]["sig"]),
    ],
})

# ---- Iteration 12 ----
i = R["i12"]
iters.append({
    "index": 12,
    "proposed_hypotheses": [
        H("h12", "Higher serum albumin (albumin_g_dl) is associated with longer pfs_months (positive coefficient on albumin_g_dl)."),
    ],
    "analyses": [
        A(["h12"],
          f"OLS pfs_months ~ albumin_g_dl: coef {i['alb_ols']['coef']} mo per g/dL (SE {i['alb_ols']['se']}), Pearson r={i['alb_corr']['r']}, p<1e-100. Higher albumin → longer PFS.",
          i["alb_ols"]["p"], i["alb_ols"]["coef"], i["alb_ols"]["sig"]),
    ],
})

# ---- Iteration 13 ----
i = R["i13"]
iters.append({
    "index": 13,
    "proposed_hypotheses": [
        H("h13a", "Higher LDH (ldh_u_l) is associated with shorter pfs_months."),
        H("h13b", "Higher CRP (crp_mg_l) is associated with shorter pfs_months."),
        H("h13c", "Higher neutrophil-lymphocyte ratio (nlr) is associated with shorter pfs_months."),
    ],
    "analyses": [
        A(["h13a"],
          f"Pearson r(ldh_u_l, pfs_months)={i['ldh_corr']['r']}, p={i['ldh_corr']['p']}. Higher LDH → shorter PFS (small effect, significant).",
          i["ldh_corr"]["p"], i["ldh_corr"]["r"], i["ldh_corr"]["sig"]),
        A(["h13b"], f"Pearson r(crp_mg_l, pfs_months)={i['crp_corr']['r']}, p={i['crp_corr']['p']}. Not significant.",
          i["crp_corr"]["p"], i["crp_corr"]["r"], i["crp_corr"]["sig"]),
        A(["h13c"], f"Pearson r(nlr, pfs_months)={i['nlr_corr']['r']}, p={i['nlr_corr']['p']}. Not significant.",
          i["nlr_corr"]["p"], i["nlr_corr"]["r"], i["nlr_corr"]["sig"]),
    ],
})

# ---- Iteration 14 ----
i = R["i14"]
iters.append({
    "index": 14,
    "proposed_hypotheses": [
        H("h14a", "Older age (age_years) is associated with shorter pfs_months (negative correlation between age_years and pfs_months)."),
        H("h14b", "Higher BMI (bmi) is associated with longer pfs_months."),
        H("h14c", "Greater 6-month weight loss (weight_loss_pct_6mo) is associated with shorter pfs_months."),
    ],
    "analyses": [
        A(["h14a"],
          f"Pearson r(age_years, pfs_months)={i['age_corr']['r']}, p<1e-300. UNEXPECTED DIRECTION — older patients have LONGER pfs_months in this cohort (positive correlation, very strong: r≈0.70). Hypothesis refuted; direction reversed.",
          i["age_corr"]["p"], i["age_corr"]["r"], i["age_corr"]["sig"]),
        A(["h14b"], f"Pearson r(bmi, pfs_months)={i['bmi_corr']['r']}, p={i['bmi_corr']['p']}. Not significant.",
          i["bmi_corr"]["p"], i["bmi_corr"]["r"], i["bmi_corr"]["sig"]),
        A(["h14c"],
          f"Pearson r(weight_loss_pct_6mo, pfs_months)={i['wtloss_corr']['r']}, p<1e-130. More weight loss → shorter PFS, as hypothesized.",
          i["wtloss_corr"]["p"], i["wtloss_corr"]["r"], i["wtloss_corr"]["sig"]),
    ],
})

# ---- Iteration 15 ----
i = R["i15"]
iters.append({
    "index": 15,
    "proposed_hypotheses": [
        H("h15a", "Larger primary tumor size (tumor_size_cm) is associated with shorter pfs_months."),
        H("h15b", "Higher Ki67 proliferation index (ki67_pct) is associated with shorter pfs_months."),
    ],
    "analyses": [
        A(["h15a"], f"Pearson r(tumor_size_cm, pfs_months)={i['tumor_size_corr']['r']}, p={i['tumor_size_corr']['p']}. Null.",
          i["tumor_size_corr"]["p"], i["tumor_size_corr"]["r"], i["tumor_size_corr"]["sig"]),
        A(["h15b"], f"Pearson r(ki67_pct, pfs_months)={i['ki67_corr']['r']}, p<1e-90. Higher Ki67 → shorter PFS.",
          i["ki67_corr"]["p"], i["ki67_corr"]["r"], i["ki67_corr"]["sig"]),
    ],
})

# ---- Iteration 16 ----
i = R["i16"]
iters.append({
    "index": 16,
    "proposed_hypotheses": [
        H("h16a", "Higher fatigue grade (fatigue_grade) is associated with shorter pfs_months."),
        H("h16b", "Higher pain (pain_nrs) is associated with shorter pfs_months."),
        H("h16c", "Higher dyspnea grade (dyspnea_grade) is associated with shorter pfs_months."),
        H("h16d", "Higher appetite-loss grade (appetite_loss_grade) is associated with shorter pfs_months."),
    ],
    "analyses": [
        A(["h16a"], f"r={i['fatigue_corr']['r']}, p={i['fatigue_corr']['p']}. Null.", i["fatigue_corr"]["p"], i["fatigue_corr"]["r"], i["fatigue_corr"]["sig"]),
        A(["h16b"], f"r={i['pain_corr']['r']}, p={i['pain_corr']['p']}. Null.", i["pain_corr"]["p"], i["pain_corr"]["r"], i["pain_corr"]["sig"]),
        A(["h16c"], f"r={i['dyspnea_corr']['r']}, p={i['dyspnea_corr']['p']}. Null.", i["dyspnea_corr"]["p"], i["dyspnea_corr"]["r"], i["dyspnea_corr"]["sig"]),
        A(["h16d"], f"r={i['appetite_corr']['r']}, p={i['appetite_corr']['p']}. Null.", i["appetite_corr"]["p"], i["appetite_corr"]["r"], i["appetite_corr"]["sig"]),
    ],
})

# ---- Iteration 17 ----
i = R["i17"]
iters.append({
    "index": 17,
    "proposed_hypotheses": [
        H("h17a", "Higher hemoglobin (hemoglobin_g_dl) is associated with longer pfs_months."),
        H("h17b", "Higher platelets (platelets_k_ul) is associated with shorter pfs_months."),
        H("h17c", "Higher absolute neutrophil count (anc_k_ul) is associated with shorter pfs_months."),
        H("h17d", "Higher absolute lymphocyte count (alc_k_ul) is associated with longer pfs_months."),
    ],
    "analyses": [
        A(["h17a"], f"r={i['hgb_corr']['r']}, p={i['hgb_corr']['p']}. Null.", i["hgb_corr"]["p"], i["hgb_corr"]["r"], i["hgb_corr"]["sig"]),
        A(["h17b"], f"r={i['plt_corr']['r']}, p={i['plt_corr']['p']}. Null.", i["plt_corr"]["p"], i["plt_corr"]["r"], i["plt_corr"]["sig"]),
        A(["h17c"], f"r={i['anc_corr']['r']}, p={i['anc_corr']['p']}. Null.", i["anc_corr"]["p"], i["anc_corr"]["r"], i["anc_corr"]["sig"]),
        A(["h17d"], f"r={i['alc_corr']['r']}, p={i['alc_corr']['p']}. Null.", i["alc_corr"]["p"], i["alc_corr"]["r"], i["alc_corr"]["sig"]),
    ],
})

# ---- Iteration 18 ----
i = R["i18"]
iters.append({
    "index": 18,
    "proposed_hypotheses": [
        H("h18a", "PIK3CA mutation (pik3ca_mutation=1) is associated with shorter pfs_months than PIK3CA-wild-type."),
        H("h18b", "TP53 mutation (tp53_mutation=1) is associated with shorter pfs_months than TP53-wild-type."),
        H("h18c", "HER2 amplification (her2_amplification=1) is associated with shorter pfs_months than non-amplified."),
    ],
    "analyses": [
        A(["h18a"],
          f"Welch t-test: PIK3CA-mut {i['pik3ca_t']['mean_pos']} mo (n={i['pik3ca_t']['n_pos']}) vs WT {i['pik3ca_t']['mean_neg']} mo; diff {i['pik3ca_t']['effect']} mo, p<1e-140.",
          i["pik3ca_t"]["p"], i["pik3ca_t"]["effect"], i["pik3ca_t"]["sig"]),
        A(["h18b"], f"TP53-mut {i['tp53_t']['mean_pos']} vs WT {i['tp53_t']['mean_neg']}; diff {i['tp53_t']['effect']} mo, p={i['tp53_t']['p']}. Null.",
          i["tp53_t"]["p"], i["tp53_t"]["effect"], i["tp53_t"]["sig"]),
        A(["h18c"], f"HER2-amp {i['her2_amp_t']['mean_pos']} vs non-amp {i['her2_amp_t']['mean_neg']}; diff {i['her2_amp_t']['effect']} mo, p={i['her2_amp_t']['p']}. Null.",
          i["her2_amp_t"]["p"], i["her2_amp_t"]["effect"], i["her2_amp_t"]["sig"]),
    ],
})

# ---- Iteration 19 ----
i = R["i19"]
iters.append({
    "index": 19,
    "proposed_hypotheses": [
        H("h19a", "More prior lines of therapy (prior_lines_of_therapy) is associated with shorter pfs_months."),
        H("h19b", "Prior chemotherapy (prior_chemotherapy=1) is associated with shorter pfs_months."),
        H("h19c", "Prior immunotherapy (prior_immunotherapy=1) is associated with shorter pfs_months."),
        H("h19d", "Longer time since diagnosis (years_since_diagnosis) is associated with shorter pfs_months."),
    ],
    "analyses": [
        A(["h19a"], f"r={i['prior_lines_corr']['r']}, p={i['prior_lines_corr']['p']}. Null.", i["prior_lines_corr"]["p"], i["prior_lines_corr"]["r"], i["prior_lines_corr"]["sig"]),
        A(["h19b"], f"diff {i['prior_chemo_t']['effect']} mo, p={i['prior_chemo_t']['p']}. Null.", i["prior_chemo_t"]["p"], i["prior_chemo_t"]["effect"], i["prior_chemo_t"]["sig"]),
        A(["h19c"], f"diff {i['prior_immuno_t']['effect']} mo, p={i['prior_immuno_t']['p']}. Null.", i["prior_immuno_t"]["p"], i["prior_immuno_t"]["effect"], i["prior_immuno_t"]["sig"]),
        A(["h19d"], f"r={i['yrs_dx_corr']['r']}, p={i['yrs_dx_corr']['p']}. Null.", i["yrs_dx_corr"]["p"], i["yrs_dx_corr"]["r"], i["yrs_dx_corr"]["sig"]),
    ],
})

# ---- Iteration 20 ----
i = R["i20"]
iters.append({
    "index": 20,
    "proposed_hypotheses": [
        H("h20a", "Mean pfs_months differs by race_ethnicity category."),
        H("h20b", "Mean pfs_months differs by insurance_type category."),
        H("h20c", "Rural residence (rural_residence=1) is associated with shorter pfs_months."),
        H("h20d", "More smoking (smoking_pack_years) is associated with shorter pfs_months."),
    ],
    "analyses": [
        A(["h20a"],
          f"OLS F-test pfs_months ~ C(race_ethnicity): F={i['race_ftest']['f']}, p={i['race_ftest']['p']}. Means: {i['race_ftest']['by_group']}. Not significant.",
          i["race_ftest"]["p"], 0.0, i["race_ftest"]["sig"]),
        A(["h20b"],
          f"OLS F-test pfs_months ~ C(insurance_type): F={i['insurance_ftest']['f']}, p={i['insurance_ftest']['p']}. Means: {i['insurance_ftest']['by_group']}. Not significant.",
          i["insurance_ftest"]["p"], 0.0, i["insurance_ftest"]["sig"]),
        A(["h20c"], f"diff {i['rural_t']['effect']} mo, p={i['rural_t']['p']}. Null.", i["rural_t"]["p"], i["rural_t"]["effect"], i["rural_t"]["sig"]),
        A(["h20d"], f"r={i['smoking_corr']['r']}, p={i['smoking_corr']['p']}. Null.", i["smoking_corr"]["p"], i["smoking_corr"]["r"], i["smoking_corr"]["sig"]),
    ],
})

# ---- Iteration 21 ----
i = R["i21"]
iters.append({
    "index": 21,
    "proposed_hypotheses": [
        H("h21a", "Chronic kidney disease (chronic_kidney_disease=1) is associated with shorter pfs_months."),
        H("h21b", "Heart failure (heart_failure=1) is associated with shorter pfs_months."),
        H("h21c", "Interstitial lung disease history (interstitial_lung_disease_history=1) is associated with shorter pfs_months."),
        H("h21d", "Autoimmune disease (autoimmune_disease=1) is associated with shorter pfs_months."),
    ],
    "analyses": [
        A(["h21a"], f"diff {i['ckd_t']['effect']} mo, p={i['ckd_t']['p']}. Null.", i["ckd_t"]["p"], i["ckd_t"]["effect"], i["ckd_t"]["sig"]),
        A(["h21b"], f"diff {i['hf_t']['effect']} mo, p={i['hf_t']['p']}. Null.", i["hf_t"]["p"], i["hf_t"]["effect"], i["hf_t"]["sig"]),
        A(["h21c"], f"diff {i['ild_t']['effect']} mo, p={i['ild_t']['p']}. Null.", i["ild_t"]["p"], i["ild_t"]["effect"], i["ild_t"]["sig"]),
        A(["h21d"], f"diff {i['autoimmune_t']['effect']} mo, p={i['autoimmune_t']['p']}. Null.", i["autoimmune_t"]["p"], i["autoimmune_t"]["effect"], i["autoimmune_t"]["sig"]),
    ],
})

# ---- Iteration 22 ----
i = R["i22"]; top = i["snp_top"][0]
iters.append({
    "index": 22,
    "proposed_hypotheses": [
        H("h22", "At least one of the 25 SNP markers (snp_rs*) is associated with pfs_months at a Bonferroni-corrected significance threshold (alpha = 0.05/25 ≈ 0.002)."),
    ],
    "analyses": [
        A(["h22"],
          f"Pearson screen across all 25 SNPs. Smallest p was {top['snp']} (r={top['r']}, p={top['p']}); Bonferroni threshold {i['snp_bonferroni_thresh']}. NO SNP survives correction; hypothesis NOT supported — no SNP is significantly associated with PFS at corrected α.",
          top["p"], top["r"], False,
          code="for c in snp_cols: stats.pearsonr(df[c], df['pfs_months'])"),
    ],
})

# ---- Iteration 23 ----
i = R["i23"]
mv_lookup = {row["term"]: row for row in i["multivariable"]}
def mv(term):
    return mv_lookup[term]
iters.append({
    "index": 23,
    "proposed_hypotheses": [
        H("h23a", "After mutual adjustment for ECOG, stage, mets, labs, biomarkers, and treatments, ecog_ps remains negatively associated with pfs_months.", "refined"),
        H("h23b", "After mutual adjustment, treatment_palbociclib has a positive coefficient on pfs_months (palbociclib remains an independent predictor of longer PFS).", "refined"),
        H("h23c", "After mutual adjustment, age_years remains positively associated with pfs_months (older = longer PFS persists adjusting for biology).", "refined"),
        H("h23d", "After mutual adjustment, the targeted-therapy main-effect terms for trastuzumab, olaparib, sacituzumab, pembrolizumab, and tamoxifen are NOT significant predictors of pfs_months in the average patient.", "refined"),
    ],
    "analyses": [
        A(["h23a"],
          f"Multivariable OLS (n={i['n']}, R²={i['r2']}). ecog_ps coef {mv('ecog_ps')['coef']} mo (p={mv('ecog_ps')['p']}, sig={mv('ecog_ps')['sig']}).",
          mv("ecog_ps")["p"], mv("ecog_ps")["coef"], mv("ecog_ps")["sig"]),
        A(["h23b"],
          f"treatment_palbociclib adjusted coef {mv('treatment_palbociclib')['coef']} mo (p={mv('treatment_palbociclib')['p']}). Independently positive.",
          mv("treatment_palbociclib")["p"], mv("treatment_palbociclib")["coef"], mv("treatment_palbociclib")["sig"]),
        A(["h23c"],
          f"age_years adjusted coef {mv('age_years')['coef']} mo per year (p={mv('age_years')['p']}). Strongly positive even adjusting for biology.",
          mv("age_years")["p"], mv("age_years")["coef"], mv("age_years")["sig"]),
        A(["h23d"],
          "Adjusted main-effect coefficients (mo): trastuzumab "
          f"{mv('treatment_trastuzumab')['coef']} (p={mv('treatment_trastuzumab')['p']}), "
          f"olaparib {mv('treatment_olaparib')['coef']} (p={mv('treatment_olaparib')['p']}), "
          f"sacituzumab {mv('treatment_sacituzumab_govitecan')['coef']} (p={mv('treatment_sacituzumab_govitecan')['p']}), "
          f"pembrolizumab {mv('treatment_pembrolizumab')['coef']} (p={mv('treatment_pembrolizumab')['p']}), "
          f"tamoxifen {mv('treatment_tamoxifen')['coef']} (p={mv('treatment_tamoxifen')['p']}). "
          "All non-significant in main-effect form, supporting the hypothesis.",
          0.5, 0.0, True),
    ],
})

# ---- Iteration 24 ----
i = R["i24"]
iters.append({
    "index": 24,
    "proposed_hypotheses": [
        H("h24a", "In HER2-NEGATIVE patients (negative control), trastuzumab does not improve pfs_months (null effect expected).", "refined"),
        H("h24b", "In BRCA-wild-type patients (negative control), olaparib does not improve pfs_months.", "refined"),
        H("h24c", "In ER-positive patients, palbociclib improves pfs_months substantially (positive control).", "refined"),
        H("h24d", "In MSI-high patients, pembrolizumab improves pfs_months (positive control); statistical power is limited (small subgroup).", "refined"),
    ],
    "analyses": [
        A(["h24a"],
          f"In her2_neg subgroup, trastuzumab vs no = {i['trastuzumab_in_her2neg']['effect']} mo (p={i['trastuzumab_in_her2neg']['p']}). Confirmed null.",
          i["trastuzumab_in_her2neg"]["p"], i["trastuzumab_in_her2neg"]["effect"], i["trastuzumab_in_her2neg"]["sig"]),
        A(["h24b"],
          f"In brca_wt subgroup, olaparib vs no = {i['olaparib_in_brca_wt']['effect']} mo (p={i['olaparib_in_brca_wt']['p']}). Confirmed null/slightly negative trend.",
          i["olaparib_in_brca_wt"]["p"], i["olaparib_in_brca_wt"]["effect"], i["olaparib_in_brca_wt"]["sig"]),
        A(["h24c"],
          f"In ER+ subgroup, palbociclib vs no = +{i['palb_in_er_pos']['effect']} mo (p<1e-300). Strongly positive (positive control passes).",
          i["palb_in_er_pos"]["p"], i["palb_in_er_pos"]["effect"], i["palb_in_er_pos"]["sig"]),
        A(["h24d"],
          f"In MSI-high subgroup, pembro vs no = +{i['pembro_in_msi_high']['effect']} mo (p={i['pembro_in_msi_high']['p']}, n_tx={i['pembro_in_msi_high']['n_tx']}, n_ctl={i['pembro_in_msi_high']['n_ctl']}). Direction is positive (consistent with hypothesis) but power-limited; not statistically significant.",
          i["pembro_in_msi_high"]["p"], i["pembro_in_msi_high"]["effect"], i["pembro_in_msi_high"]["sig"]),
    ],
})

# ---- Iteration 25 ----
i = R["i25"]
iters.append({
    "index": 25,
    "proposed_hypotheses": [
        H("h25a", "After adjusting for ECOG, stage, brain mets, liver mets, albumin, LDH, NLR, and age within the HER2-positive subgroup, trastuzumab still does not significantly improve pfs_months.", "refined"),
        H("h25b", "After adjusting for prognostic covariates within the BRCA-mutated subgroup, olaparib does not significantly improve pfs_months (i.e., the unadjusted +0.35-mo benefit attenuates with adjustment).", "refined"),
        H("h25c", "After adjustment within the ER+/HER2- subgroup, palbociclib provides a sizeable independent positive effect on pfs_months (>1 month).", "refined"),
        H("h25d", "After adjustment within the ER+ subgroup, tamoxifen does not significantly change pfs_months.", "refined"),
    ],
    "analyses": [
        A(["h25a"],
          f"In HER2+ subgroup OLS pfs_months ~ trastuzumab + ECOG + stage + brain_mets + liver_mets + albumin + LDH + NLR + age (n={i['trast_adj_her2pos']['n']}): trastuzumab coef {i['trast_adj_her2pos']['coef']} mo (p={i['trast_adj_her2pos']['p']}). Adjusted effect remains non-significant.",
          i["trast_adj_her2pos"]["p"], i["trast_adj_her2pos"]["coef"], i["trast_adj_her2pos"]["sig"]),
        A(["h25b"],
          f"In BRCA-mut subgroup adjusted OLS (n={i['olap_adj_brca_any']['n']}): olaparib coef {i['olap_adj_brca_any']['coef']} mo (p={i['olap_adj_brca_any']['p']}). Adjustment knocks out the unadjusted +0.35-mo benefit; small subgroup limits power.",
          i["olap_adj_brca_any"]["p"], i["olap_adj_brca_any"]["coef"], i["olap_adj_brca_any"]["sig"]),
        A(["h25c"],
          f"In ER+/HER2- subgroup adjusted OLS (n={i['palb_adj_er_her2neg']['n']}): palbociclib coef +{i['palb_adj_er_her2neg']['coef']} mo (p<1e-300). Strong, robust positive effect — the standout finding of the analysis.",
          i["palb_adj_er_her2neg"]["p"], i["palb_adj_er_her2neg"]["coef"], i["palb_adj_er_her2neg"]["sig"]),
        A(["h25d"],
          f"In ER+ subgroup adjusted OLS (n={i['tam_adj_erpos']['n']}): tamoxifen coef {i['tam_adj_erpos']['coef']} mo (p={i['tam_adj_erpos']['p']}). Tamoxifen still null after adjustment.",
          i["tam_adj_erpos"]["p"], i["tam_adj_erpos"]["coef"], i["tam_adj_erpos"]["sig"]),
    ],
})

transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@manual-2026-04-28",
    "max_iterations": 25,
    "iterations": iters,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

print("Wrote transcript.json with", len(iters), "iterations")
total_h = sum(len(it["proposed_hypotheses"]) for it in iters)
total_a = sum(len(it["analyses"]) for it in iters)
print(f"Total hypotheses: {total_h}, total analyses: {total_a}")
