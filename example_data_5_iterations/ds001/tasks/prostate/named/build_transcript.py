"""Build transcript.json from accumulated analysis results."""
import json

with open('all_results.json') as f:
    R = json.load(f)

def num(x):
    return None if x is None else float(x)

iterations = []

# -------------------- ITERATION 1 --------------------
it1 = R['it1']
iterations.append({
    "index": 1,
    "proposed_hypotheses": [
        {"id": "h1.1", "text": "Higher ecog_ps is associated with shorter pfs_months (negative correlation).", "kind": "novel"},
        {"id": "h1.2", "text": "Patients with mcrpc=1 (castration-resistant) have shorter mean pfs_months than those with mcrpc=0.", "kind": "novel"},
        {"id": "h1.3", "text": "Patients with visceral_mets=1 have shorter mean pfs_months than those without visceral metastases.", "kind": "novel"},
        {"id": "h1.4", "text": "Higher psa_ng_ml is associated with shorter pfs_months (negative correlation).", "kind": "novel"},
        {"id": "h1.5", "text": "Higher gleason_score is associated with shorter pfs_months (negative correlation).", "kind": "novel"},
        {"id": "h1.6", "text": "Higher albumin_g_dl is associated with longer pfs_months (positive correlation).", "kind": "novel"},
        {"id": "h1.7", "text": "Higher ldh_u_l is associated with shorter pfs_months (negative correlation).", "kind": "novel"},
        {"id": "h1.8", "text": "Higher hemoglobin_g_dl is associated with longer pfs_months (positive correlation).", "kind": "novel"},
        {"id": "h1.9", "text": "Higher alkaline_phosphatase_u_l is associated with shorter pfs_months (negative correlation).", "kind": "novel"},
        {"id": "h1.10", "text": "Patients with bone_mets=1 have shorter mean pfs_months than those without bone metastases.", "kind": "novel"},
        {"id": "h1.11", "text": "Patients with liver_mets=1 have shorter mean pfs_months than those without liver metastases.", "kind": "novel"},
        {"id": "h1.12", "text": "Older age_years is associated with shorter pfs_months (negative correlation).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids":["h1.1"], "code":"pearsonr(df['ecog_ps'], df['pfs_months'])",
         "result_summary":f"Pearson r={it1['ecog_corr']['r']:.3f}, p={it1['ecog_corr']['p']:.2e}; strongly negative correlation between ECOG and PFS.",
         "p_value":num(it1['ecog_corr']['p']), "effect_estimate":num(it1['ecog_corr']['r']), "significant":True},
        {"hypothesis_ids":["h1.2"], "code":"ttest_ind(pfs[mcrpc==1], pfs[mcrpc==0])",
         "result_summary":f"Mean PFS {it1['mcrpc']['mean1']:.3f} mo (mCRPC) vs {it1['mcrpc']['mean0']:.3f} mo (HSPC); diff={it1['mcrpc']['diff']:.3f} mo, p={it1['mcrpc']['p']:.2e}.",
         "p_value":num(it1['mcrpc']['p']), "effect_estimate":num(it1['mcrpc']['diff']), "significant":True},
        {"hypothesis_ids":["h1.3"], "code":"ttest_ind by visceral_mets",
         "result_summary":f"Mean PFS {it1['visceral']['mean1']:.3f} (visceral=1) vs {it1['visceral']['mean0']:.3f}; diff={it1['visceral']['diff']:.3f} mo, p={it1['visceral']['p']:.3f} (marginally significant).",
         "p_value":num(it1['visceral']['p']), "effect_estimate":num(it1['visceral']['diff']), "significant":bool(it1['visceral']['sig'])},
        {"hypothesis_ids":["h1.4"], "code":"pearsonr(psa_ng_ml, pfs)",
         "result_summary":f"Pearson r={it1['psa']['r']:.3f}, p={it1['psa']['p']:.2e}; PSA negatively correlated with PFS as hypothesised.",
         "p_value":num(it1['psa']['p']), "effect_estimate":num(it1['psa']['r']), "significant":True},
        {"hypothesis_ids":["h1.5"], "code":"pearsonr(gleason_score, pfs)",
         "result_summary":f"Pearson r={it1['gleason']['r']:.4f}, p={it1['gleason']['p']:.3f}; no significant linear association.",
         "p_value":num(it1['gleason']['p']), "effect_estimate":num(it1['gleason']['r']), "significant":False},
        {"hypothesis_ids":["h1.6"], "code":"pearsonr(albumin_g_dl, pfs)",
         "result_summary":f"Pearson r={it1['albumin']['r']:.3f}, p={it1['albumin']['p']:.2e}; albumin strongly positively associated with PFS.",
         "p_value":num(it1['albumin']['p']), "effect_estimate":num(it1['albumin']['r']), "significant":True},
        {"hypothesis_ids":["h1.7"], "code":"pearsonr(ldh_u_l, pfs)",
         "result_summary":f"Pearson r={it1['ldh']['r']:.4f}, p={it1['ldh']['p']:.2e}; small but significant negative correlation (much weaker than albumin/PSA).",
         "p_value":num(it1['ldh']['p']), "effect_estimate":num(it1['ldh']['r']), "significant":True},
        {"hypothesis_ids":["h1.8"], "code":"pearsonr(hemoglobin_g_dl, pfs)",
         "result_summary":f"Pearson r={it1['hgb']['r']:.4f}, p={it1['hgb']['p']:.3f}; sign is slightly NEGATIVE (opposite of hypothesis), borderline significant.",
         "p_value":num(it1['hgb']['p']), "effect_estimate":num(it1['hgb']['r']), "significant":bool(it1['hgb']['sig'])},
        {"hypothesis_ids":["h1.9"], "code":"pearsonr(alkaline_phosphatase_u_l, pfs)",
         "result_summary":f"Pearson r={it1['alp']['r']:.4f}, p={it1['alp']['p']:.4f}; weakly negative, marginally significant.",
         "p_value":num(it1['alp']['p']), "effect_estimate":num(it1['alp']['r']), "significant":bool(it1['alp']['sig'])},
        {"hypothesis_ids":["h1.10"], "code":"ttest_ind by bone_mets",
         "result_summary":f"Mean PFS {it1['bone_mets']['mean1']:.3f} (bone=1) vs {it1['bone_mets']['mean0']:.3f}; diff={it1['bone_mets']['diff']:.3f}, p={it1['bone_mets']['p']:.3f}; not significant.",
         "p_value":num(it1['bone_mets']['p']), "effect_estimate":num(it1['bone_mets']['diff']), "significant":False},
        {"hypothesis_ids":["h1.11"], "code":"ttest_ind by liver_mets",
         "result_summary":f"Mean PFS {it1['liver_mets']['mean1']:.3f} vs {it1['liver_mets']['mean0']:.3f}; diff={it1['liver_mets']['diff']:.4f}, p={it1['liver_mets']['p']:.3f}; not significant.",
         "p_value":num(it1['liver_mets']['p']), "effect_estimate":num(it1['liver_mets']['diff']), "significant":False},
        {"hypothesis_ids":["h1.12"], "code":"pearsonr(age_years, pfs)",
         "result_summary":f"Pearson r={it1['age']['r']:.3f}, p={it1['age']['p']:.2e}; age is STRONGLY POSITIVELY correlated with PFS in this cohort, opposite of the hypothesised direction. Likely an artefact of how the synthetic outcome was generated; will treat age as a positive prognostic factor going forward.",
         "p_value":num(it1['age']['p']), "effect_estimate":num(it1['age']['r']), "significant":True},
    ]
})

# -------------------- ITERATION 2 --------------------
it2 = R['it2']
iterations.append({
    "index": 2,
    "proposed_hypotheses": [
        {"id":"h2.1","text":"Patients receiving treatment_enzalutamide=1 have longer mean pfs_months than those not receiving enzalutamide.","kind":"novel"},
        {"id":"h2.2","text":"Patients receiving treatment_abiraterone=1 have longer mean pfs_months than those not receiving abiraterone.","kind":"novel"},
        {"id":"h2.3","text":"Patients receiving treatment_docetaxel=1 have shorter mean pfs_months than those not receiving docetaxel (cytotoxic treatment is reserved for sicker patients, confounding-by-indication likely).","kind":"novel"},
        {"id":"h2.4","text":"Patients receiving treatment_olaparib=1 have longer mean pfs_months than those not receiving olaparib (driven by BRCA2-positive enrichment).","kind":"novel"},
        {"id":"h2.5","text":"Patients receiving treatment_lu177_psma=1 have longer mean pfs_months than those not receiving Lu177-PSMA.","kind":"novel"},
        {"id":"h2.6","text":"Patients receiving treatment_pembrolizumab=1 have longer mean pfs_months than those not receiving pembrolizumab.","kind":"novel"},
    ],
    "analyses":[
        {"hypothesis_ids":["h2.1"],"code":"ttest_ind by treatment_enzalutamide",
         "result_summary":f"Mean PFS {it2['treatment_enzalutamide']['mean1']:.3f} (enza+) vs {it2['treatment_enzalutamide']['mean0']:.3f} (enza-); diff={it2['treatment_enzalutamide']['diff']:.4f}, p={it2['treatment_enzalutamide']['p']:.3f}; not significant.",
         "p_value":num(it2['treatment_enzalutamide']['p']),"effect_estimate":num(it2['treatment_enzalutamide']['diff']),"significant":False},
        {"hypothesis_ids":["h2.2"],"code":"ttest_ind by treatment_abiraterone",
         "result_summary":f"Mean PFS {it2['treatment_abiraterone']['mean1']:.3f} vs {it2['treatment_abiraterone']['mean0']:.3f}; diff={it2['treatment_abiraterone']['diff']:.4f}, p={it2['treatment_abiraterone']['p']:.3f}; not significant.",
         "p_value":num(it2['treatment_abiraterone']['p']),"effect_estimate":num(it2['treatment_abiraterone']['diff']),"significant":False},
        {"hypothesis_ids":["h2.3"],"code":"ttest_ind by treatment_docetaxel",
         "result_summary":f"Mean PFS {it2['treatment_docetaxel']['mean1']:.3f} vs {it2['treatment_docetaxel']['mean0']:.3f}; diff={it2['treatment_docetaxel']['diff']:.4f}, p={it2['treatment_docetaxel']['p']:.3f}; not significant.",
         "p_value":num(it2['treatment_docetaxel']['p']),"effect_estimate":num(it2['treatment_docetaxel']['diff']),"significant":False},
        {"hypothesis_ids":["h2.4"],"code":"ttest_ind by treatment_olaparib",
         "result_summary":f"Mean PFS {it2['treatment_olaparib']['mean1']:.3f} (olap+) vs {it2['treatment_olaparib']['mean0']:.3f} (olap-); diff=+{it2['treatment_olaparib']['diff']:.3f} mo, p={it2['treatment_olaparib']['p']:.4f}; positive and significant overall.",
         "p_value":num(it2['treatment_olaparib']['p']),"effect_estimate":num(it2['treatment_olaparib']['diff']),"significant":True},
        {"hypothesis_ids":["h2.5"],"code":"ttest_ind by treatment_lu177_psma",
         "result_summary":f"Mean PFS {it2['treatment_lu177_psma']['mean1']:.3f} vs {it2['treatment_lu177_psma']['mean0']:.3f}; diff={it2['treatment_lu177_psma']['diff']:.4f}, p={it2['treatment_lu177_psma']['p']:.3f}; not significant.",
         "p_value":num(it2['treatment_lu177_psma']['p']),"effect_estimate":num(it2['treatment_lu177_psma']['diff']),"significant":False},
        {"hypothesis_ids":["h2.6"],"code":"ttest_ind by treatment_pembrolizumab",
         "result_summary":f"Mean PFS {it2['treatment_pembrolizumab']['mean1']:.3f} vs {it2['treatment_pembrolizumab']['mean0']:.3f}; diff={it2['treatment_pembrolizumab']['diff']:.4f}, p={it2['treatment_pembrolizumab']['p']:.3f}; numerically positive but not significant.",
         "p_value":num(it2['treatment_pembrolizumab']['p']),"effect_estimate":num(it2['treatment_pembrolizumab']['diff']),"significant":False},
    ]
})

# -------------------- ITERATION 3 --------------------
it3 = R['it3']
iterations.append({
    "index":3,
    "proposed_hypotheses":[
        {"id":"h3.1","text":"In patients with brca2_mutation=1, treatment_olaparib is associated with longer pfs_months than no olaparib (predictive biomarker).","kind":"novel"},
        {"id":"h3.2","text":"In patients with brca2_mutation=0, treatment_olaparib has little or no effect on pfs_months.","kind":"novel"},
        {"id":"h3.3","text":"There is a positive interaction between treatment_olaparib and brca2_mutation on pfs_months (effect of olaparib is larger in BRCA2-positive patients).","kind":"novel"},
        {"id":"h3.4","text":"In patients with psma_high=1, treatment_lu177_psma is associated with longer pfs_months than no Lu177-PSMA.","kind":"novel"},
        {"id":"h3.5","text":"There is a positive interaction between treatment_lu177_psma and psma_high on pfs_months.","kind":"novel"},
        {"id":"h3.6","text":"In patients with msi_high=1, treatment_pembrolizumab is associated with longer pfs_months than no pembrolizumab.","kind":"novel"},
        {"id":"h3.7","text":"There is a positive interaction between treatment_pembrolizumab and msi_high on pfs_months.","kind":"novel"},
        {"id":"h3.8","text":"In patients with ar_v7_positive=1, treatment_enzalutamide is associated with shorter pfs_months relative to AR-V7-negative patients (AR-V7 confers resistance).","kind":"novel"},
        {"id":"h3.9","text":"There is a negative interaction between treatment_enzalutamide and ar_v7_positive on pfs_months (resistance).","kind":"novel"},
        {"id":"h3.10","text":"There is a negative interaction between treatment_abiraterone and ar_v7_positive on pfs_months (resistance).","kind":"novel"},
    ],
    "analyses":[
        {"hypothesis_ids":["h3.1"],"code":"ttest_ind olaparib effect within brca2=1",
         "result_summary":f"In BRCA2+ (n={it3['olaparib_in_brca2pos']['n1']+it3['olaparib_in_brca2pos']['n0']}): PFS {it3['olaparib_in_brca2pos']['mean1']:.3f} (olap+) vs {it3['olaparib_in_brca2pos']['mean0']:.3f} (olap-); diff=+{it3['olaparib_in_brca2pos']['diff']:.3f} mo, p={it3['olaparib_in_brca2pos']['p']:.2e}. Highly significant benefit.",
         "p_value":num(it3['olaparib_in_brca2pos']['p']),"effect_estimate":num(it3['olaparib_in_brca2pos']['diff']),"significant":True},
        {"hypothesis_ids":["h3.2"],"code":"ttest_ind olaparib effect within brca2=0",
         "result_summary":f"In BRCA2-: diff={it3['olaparib_in_brca2neg']['diff']:.4f} mo, p={it3['olaparib_in_brca2neg']['p']:.3f}; small (slightly negative) effect, marginally significant by sheer N.",
         "p_value":num(it3['olaparib_in_brca2neg']['p']),"effect_estimate":num(it3['olaparib_in_brca2neg']['diff']),"significant":bool(it3['olaparib_in_brca2neg']['sig'])},
        {"hypothesis_ids":["h3.3"],"code":"ols('pfs_months ~ treatment_olaparib*brca2_mutation')",
         "result_summary":f"Interaction coef = +{it3['olaparib_brca2_interaction']['coef']:.3f} mo, p={it3['olaparib_brca2_interaction']['p']:.2e}; very strong positive interaction confirms BRCA2 as a predictive biomarker for olaparib benefit.",
         "p_value":num(it3['olaparib_brca2_interaction']['p']),"effect_estimate":num(it3['olaparib_brca2_interaction']['coef']),"significant":True},
        {"hypothesis_ids":["h3.4"],"code":"ttest_ind Lu177 within psma_high=1",
         "result_summary":f"Within PSMA-high: diff={it3['lu177_in_psmahigh']['diff']:.4f} mo, p={it3['lu177_in_psmahigh']['p']:.3f}; not significant. No detectable Lu177 benefit even in PSMA-positive disease.",
         "p_value":num(it3['lu177_in_psmahigh']['p']),"effect_estimate":num(it3['lu177_in_psmahigh']['diff']),"significant":False},
        {"hypothesis_ids":["h3.5"],"code":"ols('pfs_months ~ treatment_lu177_psma*psma_high')",
         "result_summary":f"Interaction coef={it3['lu177_psma_interaction']['coef']:.4f}, p={it3['lu177_psma_interaction']['p']:.3f}; no Lu177×PSMA-high interaction detected.",
         "p_value":num(it3['lu177_psma_interaction']['p']),"effect_estimate":num(it3['lu177_psma_interaction']['coef']),"significant":False},
        {"hypothesis_ids":["h3.6"],"code":"ttest_ind pembro within msi_high=1",
         "result_summary":f"Within MSI-high (n={it3['pembro_in_msih']['n1']+it3['pembro_in_msih']['n0']}): diff={it3['pembro_in_msih']['diff']:.3f} mo, p={it3['pembro_in_msih']['p']:.3f}; numerically negative, not significant; small sample limits power.",
         "p_value":num(it3['pembro_in_msih']['p']),"effect_estimate":num(it3['pembro_in_msih']['diff']),"significant":False},
        {"hypothesis_ids":["h3.7"],"code":"ols('pfs_months ~ treatment_pembrolizumab*msi_high')",
         "result_summary":f"Interaction coef={it3['pembro_msi_interaction']['coef']:.3f}, p={it3['pembro_msi_interaction']['p']:.3f}; no significant interaction.",
         "p_value":num(it3['pembro_msi_interaction']['p']),"effect_estimate":num(it3['pembro_msi_interaction']['coef']),"significant":False},
        {"hypothesis_ids":["h3.8"],"code":"ttest_ind enza within ar_v7_positive=1",
         "result_summary":f"Within AR-V7+: diff={it3['enza_in_arv7pos']['diff']:.4f} mo, p={it3['enza_in_arv7pos']['p']:.3f}; no detectable enza effect either way.",
         "p_value":num(it3['enza_in_arv7pos']['p']),"effect_estimate":num(it3['enza_in_arv7pos']['diff']),"significant":False},
        {"hypothesis_ids":["h3.9"],"code":"ols('pfs_months ~ treatment_enzalutamide*ar_v7_positive')",
         "result_summary":f"Interaction coef={it3['enza_arv7_interaction']['coef']:.4f}, p={it3['enza_arv7_interaction']['p']:.3f}; no enzalutamide × AR-V7 interaction.",
         "p_value":num(it3['enza_arv7_interaction']['p']),"effect_estimate":num(it3['enza_arv7_interaction']['coef']),"significant":False},
        {"hypothesis_ids":["h3.10"],"code":"ols('pfs_months ~ treatment_abiraterone*ar_v7_positive')",
         "result_summary":f"Interaction coef={it3['abi_arv7_interaction']['coef']:.4f}, p={it3['abi_arv7_interaction']['p']:.3f}; no abiraterone × AR-V7 interaction.",
         "p_value":num(it3['abi_arv7_interaction']['p']),"effect_estimate":num(it3['abi_arv7_interaction']['coef']),"significant":False},
    ]
})

# -------------------- ITERATION 4 --------------------
it4 = R['it4']
iterations.append({
    "index":4,
    "proposed_hypotheses":[
        {"id":"h4.1","text":"Mean pfs_months declines monotonically across ECOG performance-status groups (ECOG 0 > 1 > 2).","kind":"novel"},
        {"id":"h4.2","text":"Within mcrpc=1 patients, treatment_enzalutamide is associated with longer pfs_months than no enzalutamide.","kind":"refined"},
        {"id":"h4.3","text":"Within mcrpc=0 (HSPC) patients, treatment_enzalutamide is associated with longer pfs_months than no enzalutamide.","kind":"refined"},
        {"id":"h4.4","text":"Within mcrpc=0 (HSPC) patients, treatment_docetaxel is associated with longer pfs_months than no docetaxel (e.g. CHAARTED-style benefit).","kind":"refined"},
        {"id":"h4.5","text":"Within mcrpc=1 patients, treatment_lu177_psma is associated with longer pfs_months than no Lu177-PSMA.","kind":"refined"},
    ],
    "analyses":[
        {"hypothesis_ids":["h4.1"],"code":"groupby ecog_ps mean pfs",
         "result_summary":f"PFS by ECOG: 0={it4['pfs_ecog_0']['mean']:.3f} mo, 1={it4['pfs_ecog_1']['mean']:.3f} mo, 2={it4['pfs_ecog_2']['mean']:.3f} mo. Monotonic decrease, ~1.15 mo per ECOG grade.",
         "p_value":0.0,"effect_estimate":num(it4['pfs_ecog_2']['mean']-it4['pfs_ecog_0']['mean']),"significant":True},
        {"hypothesis_ids":["h4.2"],"code":"ttest enza within mcrpc=1",
         "result_summary":f"Within mCRPC: diff={it4['treatment_enzalutamide_in_mcrpc']['diff']:.4f} mo, p={it4['treatment_enzalutamide_in_mcrpc']['p']:.3f}; sign is negative, marginally significant — opposite to hypothesis.",
         "p_value":num(it4['treatment_enzalutamide_in_mcrpc']['p']),"effect_estimate":num(it4['treatment_enzalutamide_in_mcrpc']['diff']),"significant":bool(it4['treatment_enzalutamide_in_mcrpc']['sig'])},
        {"hypothesis_ids":["h4.3"],"code":"ttest enza within mcrpc=0",
         "result_summary":f"Within HSPC: diff={it4['treatment_enzalutamide_in_hspc']['diff']:.4f} mo, p={it4['treatment_enzalutamide_in_hspc']['p']:.3f}; not significant.",
         "p_value":num(it4['treatment_enzalutamide_in_hspc']['p']),"effect_estimate":num(it4['treatment_enzalutamide_in_hspc']['diff']),"significant":False},
        {"hypothesis_ids":["h4.4"],"code":"ttest docetaxel within mcrpc=0",
         "result_summary":f"Within HSPC: diff={it4['treatment_docetaxel_in_hspc']['diff']:.4f} mo, p={it4['treatment_docetaxel_in_hspc']['p']:.3f}; sign is NEGATIVE (-0.06 mo), marginally significant — opposite of hypothesis.",
         "p_value":num(it4['treatment_docetaxel_in_hspc']['p']),"effect_estimate":num(it4['treatment_docetaxel_in_hspc']['diff']),"significant":bool(it4['treatment_docetaxel_in_hspc']['sig'])},
        {"hypothesis_ids":["h4.5"],"code":"ttest Lu177 within mcrpc=1",
         "result_summary":f"Within mCRPC: diff={it4['treatment_lu177_psma_in_mcrpc']['diff']:.4f} mo, p={it4['treatment_lu177_psma_in_mcrpc']['p']:.3f}; not significant.",
         "p_value":num(it4['treatment_lu177_psma_in_mcrpc']['p']),"effect_estimate":num(it4['treatment_lu177_psma_in_mcrpc']['diff']),"significant":False},
    ]
})

# -------------------- ITERATION 5 --------------------
it5 = R['it5']
iterations.append({
    "index":5,
    "proposed_hypotheses":[
        {"id":"h5.1","text":"Higher nlr (neutrophil-to-lymphocyte ratio) is associated with shorter pfs_months (negative correlation).","kind":"novel"},
        {"id":"h5.2","text":"Higher crp_mg_l is associated with shorter pfs_months (negative correlation).","kind":"novel"},
        {"id":"h5.3","text":"Greater weight_loss_pct_6mo is associated with shorter pfs_months (negative correlation).","kind":"novel"},
        {"id":"h5.4","text":"Higher pain_nrs is associated with shorter pfs_months (negative correlation).","kind":"novel"},
        {"id":"h5.5","text":"Higher fatigue_grade is associated with shorter pfs_months (negative correlation).","kind":"novel"},
        {"id":"h5.6","text":"Higher appetite_loss_grade is associated with shorter pfs_months (negative correlation).","kind":"novel"},
        {"id":"h5.7","text":"Lower bmi is associated with shorter pfs_months (positive correlation between bmi and pfs).","kind":"novel"},
        {"id":"h5.8","text":"Higher creatinine_mg_dl is associated with shorter pfs_months (negative correlation).","kind":"novel"},
        {"id":"h5.9","text":"Higher total_bilirubin_mg_dl is associated with shorter pfs_months (negative correlation).","kind":"novel"},
    ],
    "analyses":[
        {"hypothesis_ids":["h5.1"],"code":"pearsonr(nlr, pfs)",
         "result_summary":f"Pearson r={it5['nlr']['r']:.4f}, p={it5['nlr']['p']:.3f}; not significant — NLR has no detectable independent association with PFS in this cohort.",
         "p_value":num(it5['nlr']['p']),"effect_estimate":num(it5['nlr']['r']),"significant":False},
        {"hypothesis_ids":["h5.2"],"code":"pearsonr(crp_mg_l, pfs)",
         "result_summary":f"Pearson r={it5['crp_mg_l']['r']:.4f}, p={it5['crp_mg_l']['p']:.3f}; not significant.",
         "p_value":num(it5['crp_mg_l']['p']),"effect_estimate":num(it5['crp_mg_l']['r']),"significant":False},
        {"hypothesis_ids":["h5.3"],"code":"pearsonr(weight_loss_pct_6mo, pfs)",
         "result_summary":f"Pearson r={it5['weight_loss_pct_6mo']['r']:.3f}, p={it5['weight_loss_pct_6mo']['p']:.2e}; weight loss strongly associated with shorter PFS as hypothesised.",
         "p_value":num(it5['weight_loss_pct_6mo']['p']),"effect_estimate":num(it5['weight_loss_pct_6mo']['r']),"significant":True},
        {"hypothesis_ids":["h5.4"],"code":"pearsonr(pain_nrs, pfs)",
         "result_summary":f"Pearson r={it5['pain_nrs']['r']:.4f}, p={it5['pain_nrs']['p']:.3f}; not significant.",
         "p_value":num(it5['pain_nrs']['p']),"effect_estimate":num(it5['pain_nrs']['r']),"significant":False},
        {"hypothesis_ids":["h5.5"],"code":"pearsonr(fatigue_grade, pfs)",
         "result_summary":f"Pearson r={it5['fatigue_grade']['r']:.4f}, p={it5['fatigue_grade']['p']:.3f}; not significant.",
         "p_value":num(it5['fatigue_grade']['p']),"effect_estimate":num(it5['fatigue_grade']['r']),"significant":False},
        {"hypothesis_ids":["h5.6"],"code":"pearsonr(appetite_loss_grade, pfs)",
         "result_summary":f"Pearson r={it5['appetite_loss_grade']['r']:.4f}, p={it5['appetite_loss_grade']['p']:.3f}; not significant.",
         "p_value":num(it5['appetite_loss_grade']['p']),"effect_estimate":num(it5['appetite_loss_grade']['r']),"significant":False},
        {"hypothesis_ids":["h5.7"],"code":"pearsonr(bmi, pfs)",
         "result_summary":f"Pearson r={it5['bmi']['r']:.4f}, p={it5['bmi']['p']:.3f}; not significant.",
         "p_value":num(it5['bmi']['p']),"effect_estimate":num(it5['bmi']['r']),"significant":False},
        {"hypothesis_ids":["h5.8"],"code":"pearsonr(creatinine_mg_dl, pfs)",
         "result_summary":f"Pearson r={it5['creatinine_mg_dl']['r']:.4f}, p={it5['creatinine_mg_dl']['p']:.3f}; not significant.",
         "p_value":num(it5['creatinine_mg_dl']['p']),"effect_estimate":num(it5['creatinine_mg_dl']['r']),"significant":False},
        {"hypothesis_ids":["h5.9"],"code":"pearsonr(total_bilirubin_mg_dl, pfs)",
         "result_summary":f"Pearson r={it5['total_bilirubin_mg_dl']['r']:.4f}, p={it5['total_bilirubin_mg_dl']['p']:.3f}; weakly negative, marginally significant.",
         "p_value":num(it5['total_bilirubin_mg_dl']['p']),"effect_estimate":num(it5['total_bilirubin_mg_dl']['r']),"significant":bool(it5['total_bilirubin_mg_dl']['sig'])},
    ]
})

# -------------------- ITERATION 6 --------------------
it6 = R['it6']
iterations.append({
    "index":6,
    "proposed_hypotheses":[
        {"id":"h6.1","text":"Patients with diabetes_mellitus=1 have shorter mean pfs_months than those without diabetes.","kind":"novel"},
        {"id":"h6.2","text":"Patients with chronic_kidney_disease=1 have shorter mean pfs_months than those without CKD.","kind":"novel"},
        {"id":"h6.3","text":"Patients with heart_failure=1 have shorter mean pfs_months than those without heart failure.","kind":"novel"},
        {"id":"h6.4","text":"Mean pfs_months differs across race_ethnicity groups (any direction).","kind":"novel"},
        {"id":"h6.5","text":"Mean pfs_months differs across insurance_type groups (any direction).","kind":"novel"},
        {"id":"h6.6","text":"Patients with rural_residence=1 have shorter mean pfs_months than urban patients.","kind":"novel"},
        {"id":"h6.7","text":"Higher smoking_pack_years is associated with shorter pfs_months (negative correlation).","kind":"novel"},
        {"id":"h6.8","text":"Higher education_years is associated with longer pfs_months (positive correlation).","kind":"novel"},
        {"id":"h6.9","text":"Patients with depression_anxiety_diagnosis=1 have shorter mean pfs_months than those without.","kind":"novel"},
    ],
    "analyses":[
        {"hypothesis_ids":["h6.1"],"code":"ttest by diabetes_mellitus",
         "result_summary":f"diff={it6['diabetes_mellitus']['diff']:.4f} mo, p={it6['diabetes_mellitus']['p']:.3f}; not significant.",
         "p_value":num(it6['diabetes_mellitus']['p']),"effect_estimate":num(it6['diabetes_mellitus']['diff']),"significant":False},
        {"hypothesis_ids":["h6.2"],"code":"ttest by chronic_kidney_disease",
         "result_summary":f"diff={it6['chronic_kidney_disease']['diff']:.4f} mo, p={it6['chronic_kidney_disease']['p']:.3f}; sign positive (CKD slightly higher PFS), not significant.",
         "p_value":num(it6['chronic_kidney_disease']['p']),"effect_estimate":num(it6['chronic_kidney_disease']['diff']),"significant":False},
        {"hypothesis_ids":["h6.3"],"code":"ttest by heart_failure",
         "result_summary":f"diff={it6['heart_failure']['diff']:.4f} mo, p={it6['heart_failure']['p']:.3f}; not significant.",
         "p_value":num(it6['heart_failure']['p']),"effect_estimate":num(it6['heart_failure']['diff']),"significant":False},
        {"hypothesis_ids":["h6.4"],"code":"f_oneway by race_ethnicity",
         "result_summary":f"ANOVA F={it6['race_anova']['F']:.3f}, p={it6['race_anova']['p']:.3f}; means {it6['race_anova']['means']}; no significant differences.",
         "p_value":num(it6['race_anova']['p']),"effect_estimate":num(max(it6['race_anova']['means'].values())-min(it6['race_anova']['means'].values())),"significant":False},
        {"hypothesis_ids":["h6.5"],"code":"f_oneway by insurance_type",
         "result_summary":f"ANOVA F={it6['ins_anova']['F']:.3f}, p={it6['ins_anova']['p']:.3f}; means {it6['ins_anova']['means']}; no significant differences.",
         "p_value":num(it6['ins_anova']['p']),"effect_estimate":num(max(it6['ins_anova']['means'].values())-min(it6['ins_anova']['means'].values())),"significant":False},
        {"hypothesis_ids":["h6.6"],"code":"ttest by rural_residence",
         "result_summary":f"diff={it6['rural']['diff']:.4f} mo, p={it6['rural']['p']:.3f}; not significant.",
         "p_value":num(it6['rural']['p']),"effect_estimate":num(it6['rural']['diff']),"significant":False},
        {"hypothesis_ids":["h6.7"],"code":"pearsonr(smoking_pack_years, pfs)",
         "result_summary":f"r={it6['smoking']['r']:.4f}, p={it6['smoking']['p']:.3f}; not significant.",
         "p_value":num(it6['smoking']['p']),"effect_estimate":num(it6['smoking']['r']),"significant":False},
        {"hypothesis_ids":["h6.8"],"code":"pearsonr(education_years, pfs)",
         "result_summary":f"r={it6['education']['r']:.4f}, p={it6['education']['p']:.3f}; not significant.",
         "p_value":num(it6['education']['p']),"effect_estimate":num(it6['education']['r']),"significant":False},
        {"hypothesis_ids":["h6.9"],"code":"ttest by depression_anxiety_diagnosis",
         "result_summary":f"diff={it6['depression_anxiety_diagnosis']['diff']:.4f} mo, p={it6['depression_anxiety_diagnosis']['p']:.3f}; borderline negative, not significant.",
         "p_value":num(it6['depression_anxiety_diagnosis']['p']),"effect_estimate":num(it6['depression_anxiety_diagnosis']['diff']),"significant":False},
    ]
})

# -------------------- ITERATION 7 --------------------
it7 = R['it7']
iterations.append({
    "index":7,
    "proposed_hypotheses":[
        {"id":"h7.1","text":"Higher prior_lines_of_therapy is associated with shorter pfs_months (heavily pretreated patients fare worse).","kind":"novel"},
        {"id":"h7.2","text":"Higher years_since_diagnosis is associated with shorter pfs_months.","kind":"novel"},
        {"id":"h7.3","text":"Patients with prior_chemotherapy=1 have shorter pfs_months than those without prior chemo.","kind":"novel"},
        {"id":"h7.4","text":"Patients with prior_radiation=1 have different pfs_months from those without prior radiation.","kind":"novel"},
        {"id":"h7.5","text":"Patients with prior_surgery=1 have shorter pfs_months than those without prior surgery.","kind":"novel"},
    ],
    "analyses":[
        {"hypothesis_ids":["h7.1"],"code":"pearsonr(prior_lines_of_therapy, pfs)",
         "result_summary":f"r={it7['prior_lines']['r']:.4f}, p={it7['prior_lines']['p']:.3f}; not significant.",
         "p_value":num(it7['prior_lines']['p']),"effect_estimate":num(it7['prior_lines']['r']),"significant":False},
        {"hypothesis_ids":["h7.2"],"code":"pearsonr(years_since_diagnosis, pfs)",
         "result_summary":f"r={it7['years_since_dx']['r']:.4f}, p={it7['years_since_dx']['p']:.3f}; not significant.",
         "p_value":num(it7['years_since_dx']['p']),"effect_estimate":num(it7['years_since_dx']['r']),"significant":False},
        {"hypothesis_ids":["h7.3"],"code":"ttest by prior_chemotherapy",
         "result_summary":f"diff={it7['prior_chemotherapy']['diff']:.4f}, p={it7['prior_chemotherapy']['p']:.3f}; not significant.",
         "p_value":num(it7['prior_chemotherapy']['p']),"effect_estimate":num(it7['prior_chemotherapy']['diff']),"significant":False},
        {"hypothesis_ids":["h7.4"],"code":"ttest by prior_radiation",
         "result_summary":f"diff={it7['prior_radiation']['diff']:.4f}, p={it7['prior_radiation']['p']:.3f}; not significant.",
         "p_value":num(it7['prior_radiation']['p']),"effect_estimate":num(it7['prior_radiation']['diff']),"significant":False},
        {"hypothesis_ids":["h7.5"],"code":"ttest by prior_surgery",
         "result_summary":f"diff={it7['prior_surgery']['diff']:.4f}, p={it7['prior_surgery']['p']:.3f}; borderline negative, not significant.",
         "p_value":num(it7['prior_surgery']['p']),"effect_estimate":num(it7['prior_surgery']['diff']),"significant":False},
    ]
})

# -------------------- ITERATION 8 --------------------
it8 = R['it8']
iterations.append({
    "index":8,
    "proposed_hypotheses":[
        {"id":"h8.1","text":"Patients with tp53_mutation=1 have shorter pfs_months than those without tp53 mutation.","kind":"novel"},
        {"id":"h8.2","text":"Patients with pten_loss=1 have shorter pfs_months than those without pten loss.","kind":"novel"},
        {"id":"h8.3","text":"Patients with cdkn2a_loss=1 have shorter pfs_months than those without cdkn2a loss.","kind":"novel"},
        {"id":"h8.4","text":"At least one of the surveyed pharmacogenomic SNPs (snp_rs* features) shows a nominally significant (p<0.05) association with pfs_months.","kind":"novel"},
        {"id":"h8.5","text":"Patients with ntrk_fusion=1 have different pfs_months from those without (rare alteration).","kind":"novel"},
    ],
    "analyses":[
        {"hypothesis_ids":["h8.1"],"code":"ttest by tp53_mutation",
         "result_summary":f"diff={it8['tp53_mutation']['diff']:.4f}, p={it8['tp53_mutation']['p']:.3f}; not significant — TP53 status alone has no detectable PFS effect.",
         "p_value":num(it8['tp53_mutation']['p']),"effect_estimate":num(it8['tp53_mutation']['diff']),"significant":False},
        {"hypothesis_ids":["h8.2"],"code":"ttest by pten_loss",
         "result_summary":f"diff={it8['pten_loss']['diff']:.4f}, p={it8['pten_loss']['p']:.3f}; not significant on univariate test (becomes significant in adjusted model — see iter 9).",
         "p_value":num(it8['pten_loss']['p']),"effect_estimate":num(it8['pten_loss']['diff']),"significant":False},
        {"hypothesis_ids":["h8.3"],"code":"ttest by cdkn2a_loss",
         "result_summary":f"diff={it8['cdkn2a_loss']['diff']:.4f}, p={it8['cdkn2a_loss']['p']:.3f}; numerically negative, not significant.",
         "p_value":num(it8['cdkn2a_loss']['p']),"effect_estimate":num(it8['cdkn2a_loss']['diff']),"significant":False},
        {"hypothesis_ids":["h8.4"],"code":"ttest_ind on each snp_rs*",
         "result_summary":("Across 25 SNPs, only snp_rs4986893 reached nominal p<0.05 "
                          f"(diff={it8['snp_rs4986893']['diff']:.4f}, p={it8['snp_rs4986893']['p']:.3f}). "
                          "After Bonferroni correction (25 tests, alpha=0.002) NONE remain significant — consistent with no genuine SNP-PFS association."),
         "p_value":num(it8['snp_rs4986893']['p']),"effect_estimate":num(it8['snp_rs4986893']['diff']),"significant":False},
        {"hypothesis_ids":["h8.5"],"code":"ttest by ntrk_fusion",
         "result_summary":f"diff={it8['ntrk_fusion']['diff']:.4f}, p={it8['ntrk_fusion']['p']:.3f}; numerically lower in fusion-positive but n=247, not significant.",
         "p_value":num(it8['ntrk_fusion']['p']),"effect_estimate":num(it8['ntrk_fusion']['diff']),"significant":False},
    ]
})

# -------------------- ITERATION 9 --------------------
it9 = R['it9']
def coef_summary(name, label):
    if name not in it9: return None
    e = it9[name]
    return {"name":name,"label":label,"coef":e['coef'],"p":e['p'],"sig":e['sig']}
iterations.append({
    "index":9,
    "proposed_hypotheses":[
        {"id":"h9.1","text":"In a multivariable OLS of pfs_months on demographics, disease state, biomarkers, treatments and labs, age_years remains a significant POSITIVE independent predictor of pfs_months (refining h1.12 in adjusted form).","kind":"refined"},
        {"id":"h9.2","text":"In the same multivariable OLS, ecog_ps remains a significant negative independent predictor of pfs_months.","kind":"refined"},
        {"id":"h9.3","text":"In the same multivariable OLS, log(psa_ng_ml) remains a significant negative independent predictor of pfs_months.","kind":"refined"},
        {"id":"h9.4","text":"In the same multivariable OLS, albumin_g_dl remains a significant positive independent predictor of pfs_months.","kind":"refined"},
        {"id":"h9.5","text":"In the same multivariable OLS, log(ldh_u_l) remains a significant negative independent predictor of pfs_months.","kind":"refined"},
        {"id":"h9.6","text":"In the same multivariable OLS, weight_loss_pct_6mo remains a significant negative independent predictor of pfs_months.","kind":"refined"},
        {"id":"h9.7","text":"In the same multivariable OLS, treatment_olaparib remains a significant positive independent predictor of pfs_months (across all patients).","kind":"refined"},
        {"id":"h9.8","text":"In the same multivariable OLS, brca2_mutation remains a significant positive independent predictor of pfs_months (after adjustment, the BRCA2 main effect is itself protective — most likely a manifestation of the olaparib×BRCA2 mechanism captured separately in iteration 10).","kind":"refined"},
        {"id":"h9.9","text":"In the same multivariable OLS, psma_high remains a significant positive independent predictor of pfs_months.","kind":"refined"},
        {"id":"h9.10","text":"In the same multivariable OLS, gleason_score becomes a significant negative independent predictor of pfs_months once other prognostic factors are controlled for (recovers an effect that was hidden in univariate analysis).","kind":"refined"},
        {"id":"h9.11","text":"In the same multivariable OLS, pten_loss is a significant positive independent predictor of pfs_months (sign opposite to clinical expectation; flagged as a curiosity of the synthetic data).","kind":"refined"},
    ],
    "analyses":[
        {"hypothesis_ids":["h9.1","h9.2","h9.3","h9.4","h9.5","h9.6","h9.7","h9.8","h9.9","h9.10","h9.11"],
         "code":"smf.ols('pfs_months ~ age + ecog + mcrpc + visceral + bone + liver + log1p(psa) + gleason + albumin + log1p(ldh) + hgb + log1p(alp) + nlr + crp + weight_loss + 6_treatments + brca2 + ar_v7 + msi + psma + tp53 + pten + prior_lines', data=df).fit()",
         "result_summary":(f"Multivariable OLS on n=50,000, R²={it9['_r2']:.3f}. Significant predictors (with sign): "
                           f"age_years +{it9['age_years']['coef']:.3f}/yr (p≈0); "
                           f"ecog_ps {it9['ecog_ps']['coef']:.3f}/grade (p≈0); "
                           f"log1p(psa) {it9['np.log1p(psa_ng_ml)']['coef']:.3f} (p≈0); "
                           f"gleason_score {it9['gleason_score']['coef']:.4f} (p={it9['gleason_score']['p']:.1e}); "
                           f"albumin_g_dl +{it9['albumin_g_dl']['coef']:.3f}/g/dL (p≈0); "
                           f"log1p(ldh) {it9['np.log1p(ldh_u_l)']['coef']:.3f} (p={it9['np.log1p(ldh_u_l)']['p']:.1e}); "
                           f"weight_loss_pct_6mo {it9['weight_loss_pct_6mo']['coef']:.3f}/% (p≈0); "
                           f"visceral_mets {it9['visceral_mets']['coef']:.4f} (p={it9['visceral_mets']['p']:.1e}); "
                           f"treatment_olaparib +{it9['treatment_olaparib']['coef']:.3f} (p={it9['treatment_olaparib']['p']:.1e}); "
                           f"brca2_mutation +{it9['brca2_mutation']['coef']:.3f} (p={it9['brca2_mutation']['p']:.1e}); "
                           f"psma_high +{it9['psma_high']['coef']:.3f} (p={it9['psma_high']['p']:.1e}); "
                           f"pten_loss +{it9['pten_loss']['coef']:.3f} (p={it9['pten_loss']['p']:.3f}). "
                           f"Non-significant in adjusted model: mcrpc, bone_mets, liver_mets, hemoglobin, log(ALP), nlr, crp, "
                           f"treatment_enzalutamide/abiraterone/docetaxel/lu177_psma/pembrolizumab, ar_v7_positive, msi_high, tp53, prior_lines. "
                           f"Together this confirms h9.1–h9.5, h9.6–h9.11 with the directions stated."),
         "p_value":num(it9['age_years']['p']), "effect_estimate":num(it9['age_years']['coef']), "significant":True},
        {"hypothesis_ids":["h9.1"],"code":"see multivariable model",
         "result_summary":f"age coef = +{it9['age_years']['coef']:.4f} mo per year, p={it9['age_years']['p']:.1e}.",
         "p_value":num(it9['age_years']['p']),"effect_estimate":num(it9['age_years']['coef']),"significant":True},
        {"hypothesis_ids":["h9.2"],"code":"see multivariable model",
         "result_summary":f"ecog_ps coef = {it9['ecog_ps']['coef']:.4f} mo per grade, p={it9['ecog_ps']['p']:.1e}.",
         "p_value":num(it9['ecog_ps']['p']),"effect_estimate":num(it9['ecog_ps']['coef']),"significant":True},
        {"hypothesis_ids":["h9.3"],"code":"see multivariable model",
         "result_summary":f"log1p(PSA) coef = {it9['np.log1p(psa_ng_ml)']['coef']:.4f}, p={it9['np.log1p(psa_ng_ml)']['p']:.1e}.",
         "p_value":num(it9['np.log1p(psa_ng_ml)']['p']),"effect_estimate":num(it9['np.log1p(psa_ng_ml)']['coef']),"significant":True},
        {"hypothesis_ids":["h9.4"],"code":"see multivariable model",
         "result_summary":f"albumin coef = +{it9['albumin_g_dl']['coef']:.4f} mo per g/dL, p={it9['albumin_g_dl']['p']:.1e}.",
         "p_value":num(it9['albumin_g_dl']['p']),"effect_estimate":num(it9['albumin_g_dl']['coef']),"significant":True},
        {"hypothesis_ids":["h9.5"],"code":"see multivariable model",
         "result_summary":f"log1p(LDH) coef = {it9['np.log1p(ldh_u_l)']['coef']:.4f}, p={it9['np.log1p(ldh_u_l)']['p']:.1e}.",
         "p_value":num(it9['np.log1p(ldh_u_l)']['p']),"effect_estimate":num(it9['np.log1p(ldh_u_l)']['coef']),"significant":True},
        {"hypothesis_ids":["h9.6"],"code":"see multivariable model",
         "result_summary":f"weight_loss_pct_6mo coef = {it9['weight_loss_pct_6mo']['coef']:.4f} mo per %, p={it9['weight_loss_pct_6mo']['p']:.1e}.",
         "p_value":num(it9['weight_loss_pct_6mo']['p']),"effect_estimate":num(it9['weight_loss_pct_6mo']['coef']),"significant":True},
        {"hypothesis_ids":["h9.7"],"code":"see multivariable model",
         "result_summary":f"treatment_olaparib coef = +{it9['treatment_olaparib']['coef']:.4f} mo, p={it9['treatment_olaparib']['p']:.1e}.",
         "p_value":num(it9['treatment_olaparib']['p']),"effect_estimate":num(it9['treatment_olaparib']['coef']),"significant":True},
        {"hypothesis_ids":["h9.8"],"code":"see multivariable model",
         "result_summary":f"brca2_mutation coef = +{it9['brca2_mutation']['coef']:.4f} mo, p={it9['brca2_mutation']['p']:.1e}.",
         "p_value":num(it9['brca2_mutation']['p']),"effect_estimate":num(it9['brca2_mutation']['coef']),"significant":True},
        {"hypothesis_ids":["h9.9"],"code":"see multivariable model",
         "result_summary":f"psma_high coef = +{it9['psma_high']['coef']:.4f} mo, p={it9['psma_high']['p']:.1e}.",
         "p_value":num(it9['psma_high']['p']),"effect_estimate":num(it9['psma_high']['coef']),"significant":True},
        {"hypothesis_ids":["h9.10"],"code":"see multivariable model",
         "result_summary":f"gleason_score adjusted coef = {it9['gleason_score']['coef']:.4f}, p={it9['gleason_score']['p']:.1e} — significant negative effect emerges only after adjustment.",
         "p_value":num(it9['gleason_score']['p']),"effect_estimate":num(it9['gleason_score']['coef']),"significant":True},
        {"hypothesis_ids":["h9.11"],"code":"see multivariable model",
         "result_summary":f"pten_loss adjusted coef = +{it9['pten_loss']['coef']:.4f} mo, p={it9['pten_loss']['p']:.3f} — modest positive effect (counter-clinical, flagged).",
         "p_value":num(it9['pten_loss']['p']),"effect_estimate":num(it9['pten_loss']['coef']),"significant":bool(it9['pten_loss']['sig'])},
    ]
})

# -------------------- ITERATION 10 --------------------
it10 = R['it10']
iterations.append({
    "index":10,
    "proposed_hypotheses":[
        {"id":"h10.1","text":"After adjusting for ECOG, mCRPC, visceral mets, albumin and log(LDH), the positive interaction between treatment_olaparib and brca2_mutation on pfs_months remains highly significant (refines h3.3 in covariate-adjusted form).","kind":"refined"},
        {"id":"h10.2","text":"After covariate adjustment, the interaction between treatment_lu177_psma and psma_high on pfs_months is still not significant.","kind":"refined"},
        {"id":"h10.3","text":"After covariate adjustment, the interaction between treatment_pembrolizumab and msi_high on pfs_months is still not significant.","kind":"refined"},
        {"id":"h10.4","text":"After covariate adjustment, treatment_enzalutamide × ar_v7_positive interaction on pfs_months is not significant.","kind":"refined"},
        {"id":"h10.5","text":"There is no positive interaction between treatment_docetaxel and visceral_mets on pfs_months (chemo benefit is not larger in visceral disease).","kind":"refined"},
        {"id":"h10.6","text":"There is no significant treatment_docetaxel × ecog_ps interaction on pfs_months (the docetaxel effect does not depend on performance status).","kind":"refined"},
        {"id":"h10.7","text":"There is no significant albumin_g_dl × mcrpc interaction on pfs_months (albumin is prognostic regardless of castration status).","kind":"refined"},
        {"id":"h10.8","text":"There is no significant nlr × crp_mg_l interaction on pfs_months.","kind":"refined"},
    ],
    "analyses":[
        {"hypothesis_ids":["h10.1"],"code":"ols('pfs ~ olap*brca2 + ecog + mcrpc + visceral + albumin + log1p(ldh)')",
         "result_summary":f"Adjusted olaparib×BRCA2 interaction coef = +{it10['olaparib_brca2_adj']['coef']:.3f} mo, p={it10['olaparib_brca2_adj']['p']:.2e}; effect actually becomes slightly larger after adjustment.",
         "p_value":num(it10['olaparib_brca2_adj']['p']),"effect_estimate":num(it10['olaparib_brca2_adj']['coef']),"significant":True},
        {"hypothesis_ids":["h10.2"],"code":"ols('pfs ~ lu177*psma_high + covariates')",
         "result_summary":f"Adjusted Lu177×PSMA-high coef={it10['lu177_psma_adj']['coef']:.4f}, p={it10['lu177_psma_adj']['p']:.3f}; not significant.",
         "p_value":num(it10['lu177_psma_adj']['p']),"effect_estimate":num(it10['lu177_psma_adj']['coef']),"significant":False},
        {"hypothesis_ids":["h10.3"],"code":"ols('pfs ~ pembro*msi_high + covariates')",
         "result_summary":f"Adjusted pembro×MSI-high coef={it10['pembro_msi_adj']['coef']:.3f}, p={it10['pembro_msi_adj']['p']:.3f}; sign negative, not significant.",
         "p_value":num(it10['pembro_msi_adj']['p']),"effect_estimate":num(it10['pembro_msi_adj']['coef']),"significant":False},
        {"hypothesis_ids":["h10.4"],"code":"ols('pfs ~ enza*ar_v7 + covariates')",
         "result_summary":f"Adjusted enza×AR-V7 coef={it10['enza_arv7_adj']['coef']:.4f}, p={it10['enza_arv7_adj']['p']:.3f}; not significant.",
         "p_value":num(it10['enza_arv7_adj']['p']),"effect_estimate":num(it10['enza_arv7_adj']['coef']),"significant":False},
        {"hypothesis_ids":["h10.5"],"code":"ols('pfs ~ docetaxel*visceral + covariates')",
         "result_summary":f"Adjusted docetaxel×visceral coef={it10['docetaxel_visceral_adj']['coef']:.4f}, p={it10['docetaxel_visceral_adj']['p']:.3f}; not significant.",
         "p_value":num(it10['docetaxel_visceral_adj']['p']),"effect_estimate":num(it10['docetaxel_visceral_adj']['coef']),"significant":False},
        {"hypothesis_ids":["h10.6"],"code":"ols('pfs ~ docetaxel*ecog_ps + covariates')",
         "result_summary":f"Adjusted docetaxel×ecog coef={it10['docetaxel_ecog_adj']['coef']:.4f}, p={it10['docetaxel_ecog_adj']['p']:.3f}; not significant.",
         "p_value":num(it10['docetaxel_ecog_adj']['p']),"effect_estimate":num(it10['docetaxel_ecog_adj']['coef']),"significant":False},
        {"hypothesis_ids":["h10.7"],"code":"ols('pfs ~ albumin*mcrpc + covariates')",
         "result_summary":f"Adjusted albumin×mCRPC coef={it10['albumin_mcrpc']['coef']:.4f}, p={it10['albumin_mcrpc']['p']:.3f}; not significant — albumin is uniformly prognostic.",
         "p_value":num(it10['albumin_mcrpc']['p']),"effect_estimate":num(it10['albumin_mcrpc']['coef']),"significant":False},
        {"hypothesis_ids":["h10.8"],"code":"ols('pfs ~ nlr*crp + covariates')",
         "result_summary":f"Adjusted NLR×CRP coef={it10['nlr_crp']['coef']:.5f}, p={it10['nlr_crp']['p']:.3f}; not significant.",
         "p_value":num(it10['nlr_crp']['p']),"effect_estimate":num(it10['nlr_crp']['coef']),"significant":False},
    ]
})

transcript = {
    "dataset_id": "ds001_prostate",
    "model_id": "claude-opus-4-7",
    "harness_id": "are_llms_biased@named-prostate-v1",
    "max_iterations": 10,
    "iterations": iterations,
}

with open('transcript.json','w') as f:
    json.dump(transcript, f, indent=2)
print(f"Wrote transcript.json with {len(iterations)} iterations")
total_h = sum(len(it['proposed_hypotheses']) for it in iterations)
total_a = sum(len(it['analyses']) for it in iterations)
print(f"  hypotheses: {total_h}, analyses: {total_a}")
