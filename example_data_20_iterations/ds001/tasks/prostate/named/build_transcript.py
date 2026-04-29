"""Build transcript.json from completed analyses."""
import json

iterations = []

iterations.append({
    "index": 1,
    "proposed_hypotheses": [
        {"id": "h1", "text": "Patients receiving treatment_olaparib have a different mean pfs_months than those not receiving it (two-sided).", "kind": "novel"},
        {"id": "h2", "text": "Patients receiving treatment_pembrolizumab have a different mean pfs_months than those not receiving it.", "kind": "novel"},
        {"id": "h3", "text": "Patients receiving treatment_lu177_psma have a different mean pfs_months than those not receiving it.", "kind": "novel"},
        {"id": "h4", "text": "Patients receiving treatment_docetaxel have a different mean pfs_months than those not receiving it.", "kind": "novel"},
        {"id": "h5", "text": "Patients receiving treatment_enzalutamide have a different mean pfs_months than those not receiving it.", "kind": "novel"},
        {"id": "h6", "text": "Patients receiving treatment_abiraterone have a different mean pfs_months than those not receiving it.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h1"], "code": "stats.ttest_ind(pfs[olap==1], pfs[olap==0])",
         "result_summary": "Mean PFS 3.84 (olaparib) vs 3.74 (no olaparib); Welch t-test p=0.0033, n_tx=5098, n_no=44902.",
         "p_value": 0.00329, "effect_estimate": 0.094, "significant": True},
        {"hypothesis_ids": ["h2"], "code": "stats.ttest_ind(pfs[pembro==1], pfs[pembro==0])",
         "result_summary": "Mean PFS 3.81 (pembro) vs 3.74 (no pembro); p=0.137, n_tx=2384.",
         "p_value": 0.137, "effect_estimate": 0.064, "significant": False},
        {"hypothesis_ids": ["h3"], "code": "stats.ttest_ind(pfs[lu==1], pfs[lu==0])",
         "result_summary": "Mean PFS 3.74 (Lu177-PSMA) vs 3.74 (no Lu177-PSMA); p=0.872, n_tx=7504.",
         "p_value": 0.872, "effect_estimate": -0.004, "significant": False},
        {"hypothesis_ids": ["h4"], "code": "stats.ttest_ind(pfs[doce==1], pfs[doce==0])",
         "result_summary": "Mean PFS 3.73 (docetaxel) vs 3.75 (no docetaxel); p=0.381, n_tx=15187.",
         "p_value": 0.381, "effect_estimate": -0.017, "significant": False},
        {"hypothesis_ids": ["h5"], "code": "stats.ttest_ind(pfs[enza==1], pfs[enza==0])",
         "result_summary": "Mean PFS 3.73 (enzalutamide) vs 3.75 (no); p=0.153, n_tx=20076.",
         "p_value": 0.153, "effect_estimate": -0.026, "significant": False},
        {"hypothesis_ids": ["h6"], "code": "stats.ttest_ind(pfs[abi==1], pfs[abi==0])",
         "result_summary": "Mean PFS 3.74 (abiraterone) vs 3.75 (no); p=0.776, n_tx=14991.",
         "p_value": 0.776, "effect_estimate": -0.006, "significant": False},
    ],
})

iterations.append({
    "index": 2,
    "proposed_hypotheses": [
        {"id": "h7", "text": "Older age_years is associated with longer pfs_months (Pearson correlation > 0).", "kind": "novel"},
        {"id": "h8", "text": "Higher ecog_ps is associated with shorter pfs_months (correlation < 0).", "kind": "novel"},
        {"id": "h9", "text": "Patients with mcrpc=1 have shorter mean pfs_months than mcrpc=0.", "kind": "novel"},
        {"id": "h10", "text": "Higher psa_ng_ml is associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h11", "text": "Higher albumin_g_dl is associated with longer pfs_months.", "kind": "novel"},
        {"id": "h12", "text": "Greater weight_loss_pct_6mo is associated with shorter pfs_months.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h7"], "code": "stats.pearsonr(age_years, pfs_months)",
         "result_summary": "Strong positive correlation r=+0.856, p<1e-300; older patients have longer PFS in this cohort. By age decile mean PFS rises from 0.85 (decile 0, mean age 47) to 6.86 (decile 9, mean age 82).",
         "p_value": 0.0, "effect_estimate": 0.856, "significant": True},
        {"hypothesis_ids": ["h8"], "code": "stats.pearsonr(ecog_ps, pfs_months)",
         "result_summary": "r=-0.388, p<1e-300; higher ECOG PS strongly predicts shorter PFS.",
         "p_value": 0.0, "effect_estimate": -0.388, "significant": True},
        {"hypothesis_ids": ["h9"], "code": "stats.ttest_ind(pfs[mcrpc==1], pfs[mcrpc==0])",
         "result_summary": "Mean PFS 3.49 (mCRPC) vs 4.01 (non-mCRPC); diff=-0.521 months, p=1.6e-181.",
         "p_value": 1.58e-181, "effect_estimate": -0.521, "significant": True},
        {"hypothesis_ids": ["h10"], "code": "stats.pearsonr(psa_ng_ml, pfs_months)",
         "result_summary": "r=-0.153, p=3.5e-260; higher PSA modestly predicts shorter PFS.",
         "p_value": 3.48e-260, "effect_estimate": -0.153, "significant": True},
        {"hypothesis_ids": ["h11"], "code": "stats.pearsonr(albumin_g_dl, pfs_months)",
         "result_summary": "r=+0.123, p=1.5e-166; higher albumin predicts longer PFS.",
         "p_value": 1.47e-166, "effect_estimate": 0.123, "significant": True},
        {"hypothesis_ids": ["h12"], "code": "stats.pearsonr(weight_loss_pct_6mo, pfs_months)",
         "result_summary": "r=-0.143, p=9.9e-226; greater 6-month weight loss predicts shorter PFS.",
         "p_value": 9.91e-226, "effect_estimate": -0.143, "significant": True},
    ],
})

iterations.append({
    "index": 3,
    "proposed_hypotheses": [
        {"id": "h13", "text": "Patients with visceral_mets=1 have shorter mean pfs_months than visceral_mets=0.", "kind": "novel"},
        {"id": "h14", "text": "Patients with liver_mets=1 have shorter mean pfs_months than liver_mets=0.", "kind": "novel"},
        {"id": "h15", "text": "Patients with bone_mets=1 have shorter mean pfs_months than bone_mets=0.", "kind": "novel"},
        {"id": "h16", "text": "Patients with pleural_effusion=1 have shorter mean pfs_months than those without.", "kind": "novel"},
        {"id": "h17", "text": "Higher gleason_score is associated with shorter pfs_months (after adjustment for age, ECOG, mCRPC, albumin).", "kind": "novel"},
        {"id": "h18", "text": "Higher ldh_u_l is associated with shorter pfs_months after adjustment for age, ECOG, mCRPC, albumin.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h13"], "code": "stats.ttest_ind(pfs[visc==1], pfs[visc==0])",
         "result_summary": "diff=-0.048 months, p=0.034; visceral mets associated with marginally shorter PFS.",
         "p_value": 0.0345, "effect_estimate": -0.048, "significant": True},
        {"hypothesis_ids": ["h14"], "code": "stats.ttest_ind(pfs[liver==1], pfs[liver==0])",
         "result_summary": "diff=-0.002 months, p=0.95; liver mets show no detectable main effect.",
         "p_value": 0.948, "effect_estimate": -0.002, "significant": False},
        {"hypothesis_ids": ["h15"], "code": "stats.ttest_ind(pfs[bone==1], pfs[bone==0])",
         "result_summary": "diff=+0.011, p=0.62; bone mets show no main effect (sign opposite to expected).",
         "p_value": 0.624, "effect_estimate": 0.011, "significant": False},
        {"hypothesis_ids": ["h16"], "code": "stats.ttest_ind(pfs[pleu==1], pfs[pleu==0])",
         "result_summary": "diff=-0.043, p=0.087; trend toward shorter PFS, not significant.",
         "p_value": 0.0871, "effect_estimate": -0.043, "significant": False},
        {"hypothesis_ids": ["h17"], "code": "OLS pfs ~ gleason + age + ecog + mcrpc + albumin",
         "result_summary": "Univariate r=-0.003 (p=0.49). Adjusted gleason beta=-0.0084 per point, p=0.0016 - small but significant.",
         "p_value": 0.00155, "effect_estimate": -0.0084, "significant": True},
        {"hypothesis_ids": ["h18"], "code": "OLS pfs ~ ldh + age + ecog + mcrpc + albumin",
         "result_summary": "Univariate r=-0.017 (p=2e-4). Adjusted ldh beta=-0.00050 per U/L, p=3.8e-51 - robust prognostic effect after adjustment.",
         "p_value": 3.76e-51, "effect_estimate": -0.0005, "significant": True},
    ],
})

iterations.append({
    "index": 4,
    "proposed_hypotheses": [
        {"id": "h19", "text": "Patients with brca2_mutation=1 have a different mean pfs_months than brca2_mutation=0 (overall).", "kind": "novel"},
        {"id": "h20", "text": "There is a positive interaction between brca2_mutation and treatment_olaparib on pfs_months: olaparib is more effective in BRCA2-mutated patients than in BRCA2-wildtype patients.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h19"], "code": "stats.ttest_ind(pfs[brca2==1], pfs[brca2==0])",
         "result_summary": "Mean PFS 3.84 (BRCA2+) vs 3.74 (BRCA2-); diff=+0.100, p=0.0021. Modest positive main effect, suspected to be driven by olaparib subgroup.",
         "p_value": 0.00205, "effect_estimate": 0.100, "significant": True},
        {"hypothesis_ids": ["h20"], "code": "OLS pfs ~ brca2 + olap + brca2:olap",
         "result_summary": "Interaction beta = +1.619 months, p=4.3e-61 - extremely strong positive interaction. BRCA2 and olaparib main effects ~0; the synergy is concentrated entirely in the BRCA2+ x olaparib cell.",
         "p_value": 4.33e-61, "effect_estimate": 1.619, "significant": True},
    ],
})

iterations.append({
    "index": 5,
    "proposed_hypotheses": [
        {"id": "h21", "text": "Within brca2_mutation=1 patients, treatment_olaparib increases mean pfs_months by >1 month vs no olaparib.", "kind": "refined"},
        {"id": "h22", "text": "Within brca2_mutation=0 patients, treatment_olaparib has no clinically meaningful (>0.1 month) positive effect on pfs_months.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h21"], "code": "stats.ttest_ind(pfs[brca2==1 & olap==1], pfs[brca2==1 & olap==0])",
         "result_summary": "BRCA2+ patients: mean PFS 5.22 (olap, n=529) vs 3.67 (no olap, n=4467); diff=+1.545 months, p=4.0e-27.",
         "p_value": 3.95e-27, "effect_estimate": 1.545, "significant": True},
        {"hypothesis_ids": ["h22"], "code": "stats.ttest_ind(pfs[brca2==0 & olap==1], pfs[brca2==0 & olap==0])",
         "result_summary": "BRCA2- patients: mean PFS 3.67 (olap, n=4569) vs 3.74 (no olap, n=40435); diff=-0.073, p=0.019. Olaparib does not benefit (and may slightly hurt) BRCA2-wildtype patients.",
         "p_value": 0.0191, "effect_estimate": -0.073, "significant": True},
    ],
})

iterations.append({
    "index": 6,
    "proposed_hypotheses": [
        {"id": "h23", "text": "The BRCA2 x olaparib interaction is preserved (positive, large) within the mcrpc=1 subgroup.", "kind": "refined"},
        {"id": "h24", "text": "The BRCA2 x olaparib interaction remains positive and significant after adjustment for age, ECOG, mCRPC, visceral mets, albumin, weight loss, and PSA.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h23"], "code": "stratify mcrpc==1; ttest by olap within brca2==1",
         "result_summary": "Within mCRPC=1, BRCA2+ patients: olaparib effect = +1.548 months (p=7.9e-16, n_tx=296, n_no=2428). Within non-mCRPC, BRCA2+: +1.564 (p=2.4e-13). The olaparib benefit in BRCA2+ is essentially identical across mCRPC status.",
         "p_value": 7.91e-16, "effect_estimate": 1.548, "significant": True},
        {"hypothesis_ids": ["h24"], "code": "OLS pfs ~ brca2 + olap + brca2:olap + age + ecog + mcrpc + visceral + albumin + weight_loss + psa",
         "result_summary": "Adjusted: brca2 beta=+0.003 (p=0.63), olap beta=-0.003 (p=0.66), brca2*olap beta=+1.439 (p<<1e-300). Model R^2=0.951. Interaction is fully robust to clinical adjustment.",
         "p_value": 0.0, "effect_estimate": 1.439, "significant": True},
    ],
})

iterations.append({
    "index": 7,
    "proposed_hypotheses": [
        {"id": "h25", "text": "There is a positive interaction between msi_high and treatment_pembrolizumab on pfs_months: pembrolizumab is more effective in MSI-high patients.", "kind": "novel"},
        {"id": "h26", "text": "Among msi_high=1 patients, treatment_pembrolizumab increases mean pfs_months vs no pembrolizumab.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h25"], "code": "OLS pfs ~ msi + pembro + msi:pembro",
         "result_summary": "Interaction beta=-0.340, p=0.15. After adjusting for age/ECOG/mCRPC/albumin: ix beta=+0.012, p=0.87. No detectable MSI x pembrolizumab synergy.",
         "p_value": 0.153, "effect_estimate": -0.340, "significant": False},
        {"hypothesis_ids": ["h26"], "code": "ttest within MSI-high",
         "result_summary": "MSI-high (n=1528): pembrolizumab PFS 3.53 (n=79) vs 3.79 (n=1449), diff=-0.265, p=0.24. Numerically opposite the expected direction; subgroup small.",
         "p_value": 0.239, "effect_estimate": -0.265, "significant": False},
    ],
})

iterations.append({
    "index": 8,
    "proposed_hypotheses": [
        {"id": "h27", "text": "There is a positive interaction between psma_high and treatment_lu177_psma on pfs_months: Lu177-PSMA is more effective in PSMA-high patients.", "kind": "novel"},
        {"id": "h28", "text": "Among psma_high=1 patients, treatment_lu177_psma increases mean pfs_months vs no Lu177-PSMA.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h27"], "code": "OLS pfs ~ psma + lu + psma:lu",
         "result_summary": "Interaction beta=-0.017, p=0.74. No detectable PSMA-high x Lu177-PSMA synergy in this cohort.",
         "p_value": 0.737, "effect_estimate": -0.017, "significant": False},
        {"hypothesis_ids": ["h28"], "code": "stratify psma==1; ttest by lu",
         "result_summary": "PSMA-high (n=29962): Lu177 PFS 3.74 (n=4486) vs 3.75 (n=25476), diff=-0.011. PSMA-low: Lu177 PFS 3.74 (n=3018) vs 3.73 (n=17020), diff=+0.006. No benefit detected in either stratum.",
         "p_value": None, "effect_estimate": -0.011, "significant": False},
    ],
})

iterations.append({
    "index": 9,
    "proposed_hypotheses": [
        {"id": "h29", "text": "There is a negative interaction between ar_v7_positive and treatment_enzalutamide on pfs_months: enzalutamide is less effective in AR-V7-positive patients.", "kind": "novel"},
        {"id": "h30", "text": "There is a negative interaction between ar_v7_positive and treatment_abiraterone on pfs_months.", "kind": "novel"},
        {"id": "h31", "text": "Within ar_v7_positive=1, treatment_enzalutamide produces shorter pfs_months than no enzalutamide (resistance signal).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h29"], "code": "OLS pfs ~ arv7 + enza + arv7:enza",
         "result_summary": "Interaction beta=+0.002, p=0.96. No interaction detected.",
         "p_value": 0.957, "effect_estimate": 0.002, "significant": False},
        {"hypothesis_ids": ["h30"], "code": "OLS pfs ~ arv7 + abi + arv7:abi",
         "result_summary": "Interaction beta=+0.048, p=0.33. No interaction detected.",
         "p_value": 0.334, "effect_estimate": 0.048, "significant": False},
        {"hypothesis_ids": ["h31"], "code": "ttest pfs by enza within arv7==1",
         "result_summary": "AR-V7+ subgroup: enza PFS 3.71 (n=4019) vs 3.74 (n=6019), diff=-0.024, p=0.56. No resistance signal observed.",
         "p_value": 0.557, "effect_estimate": -0.024, "significant": False},
    ],
})

iterations.append({
    "index": 10,
    "proposed_hypotheses": [
        {"id": "h32", "text": "There is a negative interaction between ecog_ps and treatment_docetaxel on pfs_months: docetaxel is less effective in poor performance status patients.", "kind": "novel"},
        {"id": "h33", "text": "There is a negative interaction between ecog_ps and treatment_pembrolizumab on pfs_months.", "kind": "novel"},
        {"id": "h34", "text": "There is a negative interaction between ecog_ps and treatment_olaparib on pfs_months.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h32"], "code": "OLS pfs ~ ecog + doce + ecog:doce",
         "result_summary": "Interaction beta=+0.012, p=0.66. No interaction.",
         "p_value": 0.663, "effect_estimate": 0.012, "significant": False},
        {"hypothesis_ids": ["h33"], "code": "OLS pfs ~ ecog + pembro + ecog:pembro",
         "result_summary": "Interaction beta=-0.041, p=0.48. No interaction.",
         "p_value": 0.483, "effect_estimate": -0.041, "significant": False},
        {"hypothesis_ids": ["h34"], "code": "OLS pfs ~ ecog + olap + ecog:olap",
         "result_summary": "Interaction beta=-0.037, p=0.36. No interaction (the BRCA2 axis dominates olaparib effects, not ECOG).",
         "p_value": 0.364, "effect_estimate": -0.037, "significant": False},
    ],
})

iterations.append({
    "index": 11,
    "proposed_hypotheses": [
        {"id": "h35", "text": "treatment_docetaxel increases pfs_months more in visceral_mets=1 patients than visceral_mets=0 (chemotherapy benefit in aggressive disease).", "kind": "novel"},
        {"id": "h36", "text": "treatment_lu177_psma is more effective (longer PFS) in patients with bone_mets=1 vs bone_mets=0.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h35"], "code": "OLS pfs ~ visceral + doce + visceral:doce; stratified",
         "result_summary": "Interaction beta=+0.052, p=0.29. Within visceral=1, doce diff=+0.024 (p=0.58); within visceral=0, doce diff=-0.027 (p=0.21). No clear differential benefit.",
         "p_value": 0.294, "effect_estimate": 0.052, "significant": False},
        {"hypothesis_ids": ["h36"], "code": "OLS pfs ~ bone + lu + bone:lu (adj age,ecog)",
         "result_summary": "Interaction beta=-0.029, p=0.19. Direction is opposite to the hypothesis but not significant.",
         "p_value": 0.186, "effect_estimate": -0.029, "significant": False},
    ],
})

iterations.append({
    "index": 12,
    "proposed_hypotheses": [
        {"id": "h37", "text": "Patients with chronic_kidney_disease=1 have shorter mean pfs_months than those without.", "kind": "novel"},
        {"id": "h38", "text": "Patients with heart_failure=1 have shorter mean pfs_months than those without.", "kind": "novel"},
        {"id": "h39", "text": "Patients with depression_anxiety_diagnosis=1 have shorter mean pfs_months than those without.", "kind": "novel"},
        {"id": "h40", "text": "Patients with diabetes_mellitus=1 have shorter mean pfs_months than those without.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h37"], "code": "ttest pfs by ckd",
         "result_summary": "diff=+0.033, p=0.23. No effect (opposite direction of hypothesis).",
         "p_value": 0.231, "effect_estimate": 0.033, "significant": False},
        {"hypothesis_ids": ["h38"], "code": "ttest pfs by hf",
         "result_summary": "diff=-0.003, p=0.92. No effect.",
         "p_value": 0.916, "effect_estimate": -0.003, "significant": False},
        {"hypothesis_ids": ["h39"], "code": "ttest pfs by depression",
         "result_summary": "diff=-0.044, p=0.061. Trend, not significant.",
         "p_value": 0.0606, "effect_estimate": -0.044, "significant": False},
        {"hypothesis_ids": ["h40"], "code": "ttest pfs by dm",
         "result_summary": "diff=-0.029, p=0.19. No effect.",
         "p_value": 0.19, "effect_estimate": -0.029, "significant": False},
    ],
})

iterations.append({
    "index": 13,
    "proposed_hypotheses": [
        {"id": "h41", "text": "Mean pfs_months differs between black vs non-black patients (race_ethnicity).", "kind": "novel"},
        {"id": "h42", "text": "Mean pfs_months differs between hispanic vs non-hispanic patients.", "kind": "novel"},
        {"id": "h43", "text": "Mean pfs_months differs between asian vs non-asian patients.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h41"], "code": "ttest pfs[race==black] vs others",
         "result_summary": "diff=-0.004, p=0.89, n_black=6064. No racial disparity in PFS detected.",
         "p_value": 0.889, "effect_estimate": -0.004, "significant": False},
        {"hypothesis_ids": ["h42"], "code": "ttest pfs[race==hispanic] vs others",
         "result_summary": "diff=-0.008, p=0.76, n_hispanic=7562. No effect.",
         "p_value": 0.757, "effect_estimate": -0.008, "significant": False},
        {"hypothesis_ids": ["h43"], "code": "ttest pfs[race==asian] vs others",
         "result_summary": "diff=+0.057, p=0.14, n_asian=3019. No significant difference.",
         "p_value": 0.136, "effect_estimate": 0.057, "significant": False},
    ],
})

iterations.append({
    "index": 14,
    "proposed_hypotheses": [
        {"id": "h44", "text": "Uninsured patients have shorter mean pfs_months than insured patients.", "kind": "novel"},
        {"id": "h45", "text": "Medicaid patients have shorter mean pfs_months than non-Medicaid patients.", "kind": "novel"},
        {"id": "h46", "text": "Rural residence is associated with shorter mean pfs_months.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h44"], "code": "ttest pfs by insurance==uninsured",
         "result_summary": "diff=-0.019, p=0.68, n_uninsured=1994. No SDOH effect detected.",
         "p_value": 0.683, "effect_estimate": -0.019, "significant": False},
        {"hypothesis_ids": ["h45"], "code": "ttest pfs by insurance==medicaid",
         "result_summary": "diff=-0.004, p=0.88, n_medicaid=7087. No effect.",
         "p_value": 0.88, "effect_estimate": -0.004, "significant": False},
        {"hypothesis_ids": ["h46"], "code": "ttest pfs by rural_residence",
         "result_summary": "diff=+0.015, p=0.50. No effect (opposite sign to hypothesis).",
         "p_value": 0.504, "effect_estimate": 0.015, "significant": False},
    ],
})

iterations.append({
    "index": 15,
    "proposed_hypotheses": [
        {"id": "h47", "text": "At least one of the catalogued SNPs (snp_rs* fields) is associated with pfs_months at uncorrected p<0.01.", "kind": "novel"},
        {"id": "h48", "text": "snp_rs4986893 (lowest-p SNP from screen) is independently associated with pfs_months after adjustment for age, ECOG, mCRPC, albumin.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h47"], "code": "for each snp: pearsonr(snp, pfs)",
         "result_summary": "Min p across 28 SNPs = 0.0145 (snp_rs4986893, r=-0.011). At Bonferroni alpha 0.05/28=0.0018, no SNP is significant. No SNP achieves uncorrected p<0.01.",
         "p_value": 0.0145, "effect_estimate": -0.011, "significant": False},
        {"hypothesis_ids": ["h48"], "code": "OLS pfs ~ snp_rs4986893 + age + ecog + mcrpc + albumin",
         "result_summary": "Top SNP signal does not survive scrutiny; SNP block is essentially non-informative for PFS.",
         "p_value": None, "effect_estimate": -0.011, "significant": False},
    ],
})

iterations.append({
    "index": 16,
    "proposed_hypotheses": [
        {"id": "h49", "text": "There is a multiplicative (negative) interaction between albumin_g_dl and weight_loss_pct_6mo on pfs_months: low albumin compounds the harm of weight loss.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h49"], "code": "OLS pfs ~ albumin + weight_loss + albumin:weight_loss + age + ecog",
         "result_summary": "Interaction beta=-0.0004, p=0.75. Albumin and weight loss act additively on PFS; no synergistic cachexia effect detected. Both main effects remain strongly significant (alb +0.48, wl -0.075, both p<<0.001).",
         "p_value": 0.753, "effect_estimate": -0.0004, "significant": False},
    ],
})

iterations.append({
    "index": 17,
    "proposed_hypotheses": [
        {"id": "h50", "text": "Within ecog_ps=2 (frail) patients, treatment_olaparib still improves pfs_months relative to no olaparib.", "kind": "refined"},
        {"id": "h51", "text": "Within ecog_ps=2 patients, treatment_docetaxel decreases mean pfs_months (toxicity outweighs benefit in frail patients).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h50"], "code": "ttest pfs by olap within ECOG=2",
         "result_summary": "ECOG=2 subgroup (n=7437): olaparib diff=+0.148 months, p=0.044. Olaparib retains a positive effect even in frail patients (driven by BRCA2+ subset).",
         "p_value": 0.0439, "effect_estimate": 0.148, "significant": True},
        {"hypothesis_ids": ["h51"], "code": "ttest pfs by doce within ECOG=2",
         "result_summary": "ECOG=2: docetaxel diff=-0.008, p=0.86. No frailty toxicity signal detected.",
         "p_value": 0.863, "effect_estimate": -0.008, "significant": False},
    ],
})

iterations.append({
    "index": 18,
    "proposed_hypotheses": [
        {"id": "h52", "text": "A composite symptom burden score (pain_nrs + fatigue_grade + dyspnea_grade + cough_grade + appetite_loss_grade) is negatively correlated with pfs_months.", "kind": "novel"},
        {"id": "h53", "text": "pain_nrs alone is negatively correlated with pfs_months.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h52"], "code": "pearsonr(symptom_composite, pfs)",
         "result_summary": "r=-0.0004, p=0.93. Symptom burden has no detectable relationship to PFS in this cohort.",
         "p_value": 0.934, "effect_estimate": -0.0004, "significant": False},
        {"hypothesis_ids": ["h53"], "code": "pearsonr(pain_nrs, pfs)",
         "result_summary": "r=-0.001, p=0.80. No effect.",
         "p_value": 0.797, "effect_estimate": -0.001, "significant": False},
    ],
})

iterations.append({
    "index": 19,
    "proposed_hypotheses": [
        {"id": "h54", "text": "The BRCA2 x olaparib synergy is itself modified by mcrpc status (3-way interaction term BRCA2:olaparib:mCRPC on pfs_months is non-zero).", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h54"], "code": "OLS pfs ~ b + o + m + b:o + b:m + o:m + b:o:m",
         "result_summary": "3-way (b:o:m) beta=+0.003, p=0.99. The 2-way b:o interaction is +1.627 (p=9e-29). The synergy is essentially identical in mCRPC and non-mCRPC contexts; mCRPC is purely additive (-0.51 main effect).",
         "p_value": 0.987, "effect_estimate": 0.003, "significant": False},
    ],
})

iterations.append({
    "index": 20,
    "proposed_hypotheses": [
        {"id": "h55", "text": "A multivariable OLS predicting pfs_months from age, ECOG, mCRPC, visceral_mets, PSA, albumin, weight_loss, BRCA2, AR-V7, MSI, PSMA, the six treatments, and the BRCA2:olap, MSI:pembro, PSMA:lu interactions explains substantial PFS variance, with the BRCA2:olap interaction the dominant treatment-related coefficient.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h55"], "code": "OLS pfs ~ comprehensive feature set with 3 interactions",
         "result_summary": "R^2=0.951. Largest absolute coefficients: ecog (-1.15), albumin (+0.48), mcrpc (-0.38), age (+0.17). brca2:olap interaction beta=+1.439 (p<<1e-300) is the only treatment-related coefficient with a clinically meaningful magnitude. msi:pembro and psma:lu interactions are not significant. Main treatment effects are tiny.",
         "p_value": 0.0, "effect_estimate": 1.439, "significant": True},
    ],
})

iterations.append({
    "index": 21,
    "proposed_hypotheses": [
        {"id": "h56", "text": "Within brca2_mutation=0 patients, no treatment (enzalutamide, abiraterone, docetaxel, olaparib, Lu177-PSMA, pembrolizumab) produces a clinically meaningful (>0.1 month) positive improvement in pfs_months.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h56"], "code": "stratify brca2==0; ttest each treatment",
         "result_summary": "In BRCA2- (n=45004): olap diff=-0.073 (p=0.019, slight harm), enza -0.026 (p=0.21), abi ~0, doce -0.020, lu -0.005, pembro +0.064 (p=0.13). Across all six treatments, no clinically meaningful PFS gain in BRCA2-wildtype patients. The reservoir of benefit is concentrated in BRCA2+ x olaparib.",
         "p_value": 0.0191, "effect_estimate": -0.073, "significant": True},
    ],
})

iterations.append({
    "index": 22,
    "proposed_hypotheses": [
        {"id": "h57", "text": "Within msi_high=0 patients, treatment_pembrolizumab provides a small positive effect on pfs_months (i.e., the apparent overall pembro main effect is not driven by a small MSI-high signal).", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h57"], "code": "ttest pfs by pembro within MSI-low",
         "result_summary": "MSI-low subgroup (n=48472): pembro PFS 3.81 (n=2305) vs 3.74 (n=46167); diff=+0.075, p=0.086. Direction matches main-effect estimate; the MSI-high subgroup (n=79 on pembro) is too small to drive the overall pembro main effect.",
         "p_value": 0.0864, "effect_estimate": 0.075, "significant": False},
    ],
})

iterations.append({
    "index": 23,
    "proposed_hypotheses": [
        {"id": "h58", "text": "years_since_diagnosis is correlated with pfs_months (longer disease history -> different outcomes).", "kind": "novel"},
        {"id": "h59", "text": "Total number of concurrent treatments (sum across the six treatment indicators) is positively correlated with pfs_months.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h58"], "code": "pearsonr(years_since_dx, pfs)",
         "result_summary": "r=-0.001, p=0.90. No detectable relationship.",
         "p_value": 0.897, "effect_estimate": -0.001, "significant": False},
        {"hypothesis_ids": ["h59"], "code": "groupby n_treatments mean(pfs)",
         "result_summary": "Mean PFS by treatment count: 0->3.76, 1->3.74, 2->3.72, 3->3.80, 4->3.78, 5->3.91. Modal patients (1 treatment) have similar PFS to untreated; trend with multiple treatments is small and non-monotonic.",
         "p_value": None, "effect_estimate": 0.02, "significant": False},
    ],
})

iterations.append({
    "index": 24,
    "proposed_hypotheses": [
        {"id": "h60", "text": "Consistent with a prostate cancer cohort, sex_female=1 prevalence is essentially zero (and therefore sex effects on pfs_months cannot be tested).", "kind": "novel"},
        {"id": "h61", "text": "The strong positive age-PFS correlation (r=+0.86) survives adjustment for ECOG, mCRPC, albumin, weight loss, PSA - i.e., it is not confounded by these factors.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h60"], "code": "df['sex_female'].sum()",
         "result_summary": "0 of 50,000 patients have sex_female=1. Cohort is uniformly male, consistent with prostate cancer; sex contrast is therefore not estimable.",
         "p_value": None, "effect_estimate": 0.0, "significant": False},
        {"hypothesis_ids": ["h61"], "code": "OLS pfs ~ age + ecog + mcrpc + albumin + weight_loss + psa (and full model)",
         "result_summary": "In the comprehensive multivariable model (iter 20), age beta=+0.174 per year (p<<1e-300) - very strong positive coefficient that persists after extensive adjustment. The direction is opposite to typical clinical expectation and likely reflects a synthetic data-generation pattern rather than a real biological effect.",
         "p_value": 0.0, "effect_estimate": 0.174, "significant": True},
    ],
})

iterations.append({
    "index": 25,
    "proposed_hypotheses": [
        {"id": "h62", "text": "Across all hypotheses tested, the BRCA2 x olaparib interaction is the only therapy-relevant signal with both clinical magnitude (>1 month) and overwhelming statistical support; other targeted-therapy biomarker pairs (MSI/pembrolizumab, PSMA/Lu177-PSMA, AR-V7/ARSI) show no detectable interactions on PFS.", "kind": "refined"},
        {"id": "h63", "text": "Prognosis in this cohort is dominated by performance status (ECOG, negative), age (positive direction), mCRPC status (negative), albumin (positive), weight loss (negative), PSA (negative), and LDH (negative); demographics, comorbidities, symptom burden, SDOH, and the catalogued SNPs add little prognostic information.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h62"], "code": "synthesis across iters 4-9, 19-22",
         "result_summary": "Confirmed. Olaparib in BRCA2+ adds ~+1.5 months PFS (p=4e-27); the same drug shows -0.07 months in BRCA2-wildtype patients (p=0.02). MSI:pembro p=0.15, PSMA:Lu177 p=0.74, AR-V7:enza p=0.96. The pharmacogenomic axis is BRCA2-specific.",
         "p_value": 4.33e-61, "effect_estimate": 1.619, "significant": True},
        {"hypothesis_ids": ["h63"], "code": "synthesis across iters 2,3,12-15,18,20",
         "result_summary": "Confirmed. Comprehensive OLS R^2=0.951 driven almost entirely by age, ECOG, mCRPC, albumin, weight_loss, PSA, LDH and the BRCA2:olap interaction. Race, insurance, rurality, comorbidities, symptom burden, and 28 SNPs are all null.",
         "p_value": 0.0, "effect_estimate": 0.951, "significant": True},
    ],
})

transcript = {
    "dataset_id": "ds001_prostate",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@manual-driver",
    "max_iterations": 25,
    "iterations": iterations,
}

with open('transcript.json', 'w') as f:
    json.dump(transcript, f, indent=2)

print('iterations:', len(iterations))
hyps = sum(len(it['proposed_hypotheses']) for it in iterations)
analyses = sum(len(it['analyses']) for it in iterations)
print('hypotheses:', hyps)
print('analyses:', analyses)
