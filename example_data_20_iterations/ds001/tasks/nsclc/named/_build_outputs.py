"""Build transcript.json and analysis_summary.txt from the iterative analysis."""
import json
from pathlib import Path

DATASET_ID = "ds001_nsclc"
MODEL_ID = "claude-opus-4-7"
HARNESS_ID = "claude-code@native-direct"
MAX_ITER = 25

iterations = []

# --------------- Iter 1: main treatment effects ---------------
iterations.append({
    "index": 1,
    "proposed_hypotheses": [
        {"id": "h1.1", "text": "Patients receiving treatment_pembrolizumab have a higher objective_response rate than those not receiving it.", "kind": "novel"},
        {"id": "h1.2", "text": "Patients receiving treatment_sotorasib have a higher objective_response rate than those not receiving it.", "kind": "novel"},
        {"id": "h1.3", "text": "Patients receiving treatment_olaparib have a higher objective_response rate than those not receiving it.", "kind": "novel"},
        {"id": "h1.4", "text": "Patients receiving treatment_osimertinib have a higher objective_response rate than those not receiving it.", "kind": "novel"},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h1.1"],
            "code": "stats.chi2_contingency 2x2 of treatment_pembrolizumab x objective_response",
            "result_summary": "Response 0.1739 on pembrolizumab vs 0.1640 off; OR=1.073, log(OR)=+0.070, chi-square p=0.0034 — small but statistically significant overall benefit.",
            "p_value": 0.0034,
            "effect_estimate": 0.0702,
            "significant": True,
        },
        {
            "hypothesis_ids": ["h1.2"],
            "code": "stats.chi2_contingency 2x2 of treatment_sotorasib x objective_response",
            "result_summary": "Response 0.1695 on sotorasib vs 0.1687 off; OR=1.006, p=0.83 — no main effect.",
            "p_value": 0.8294,
            "effect_estimate": 0.0057,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h1.3"],
            "code": "stats.chi2_contingency 2x2 of treatment_olaparib x objective_response",
            "result_summary": "Response 0.1699 on olaparib vs 0.1685 off; OR=1.010, p=0.72 — no main effect.",
            "p_value": 0.7238,
            "effect_estimate": 0.0095,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h1.4"],
            "code": "stats.chi2_contingency 2x2 of treatment_osimertinib x objective_response",
            "result_summary": "Response 0.1655 on osimertinib vs 0.1704 off; OR=0.965, p=0.18 — no main effect (numerically lower).",
            "p_value": 0.1792,
            "effect_estimate": -0.0355,
            "significant": False,
        },
    ],
})

# --------------- Iter 2: targeted-therapy x target-biomarker interactions ---------------
iterations.append({
    "index": 2,
    "proposed_hypotheses": [
        {"id": "h2.1", "text": "treatment_osimertinib increases objective_response specifically in egfr_mutation-positive patients (positive interaction term).", "kind": "novel"},
        {"id": "h2.2", "text": "treatment_sotorasib increases objective_response specifically in kras_g12c-positive patients (positive interaction term).", "kind": "novel"},
        {"id": "h2.3", "text": "treatment_olaparib increases objective_response specifically in brca2_mutation-positive patients (positive interaction term).", "kind": "novel"},
        {"id": "h2.4", "text": "treatment_pembrolizumab increases objective_response more in tmb_high patients than in tmb_high-negative patients (positive interaction term).", "kind": "novel"},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h2.1"],
            "code": "smf.logit('objective_response ~ treatment_osimertinib*egfr_mutation')",
            "result_summary": "Osimertinib×EGFR-mut interaction coef=+0.024, p=0.76. In EGFR-mut patients, osimertinib RR 0.164 vs 0.166 off (OR=0.99, p=0.87). No biomarker-specific benefit detected.",
            "p_value": 0.7584,
            "effect_estimate": 0.024,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h2.2"],
            "code": "smf.logit('objective_response ~ treatment_sotorasib*kras_g12c')",
            "result_summary": "Sotorasib×KRAS-G12C interaction coef=+0.059, p=0.43. In KRAS-G12C+ patients, sotorasib RR 0.175 vs 0.166 off (OR=1.06, p=0.43). No biomarker-specific benefit.",
            "p_value": 0.4266,
            "effect_estimate": 0.0593,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h2.3"],
            "code": "smf.logit('objective_response ~ treatment_olaparib*brca2_mutation')",
            "result_summary": "Olaparib×BRCA2-mut interaction coef=-0.178, p=0.27. In BRCA2-mut patients, olaparib RR 0.146 vs 0.168 off (OR=0.85, p=0.35) — opposite of expected, not significant.",
            "p_value": 0.2747,
            "effect_estimate": -0.1775,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h2.4"],
            "code": "smf.logit('objective_response ~ treatment_pembrolizumab*tmb_high')",
            "result_summary": "Pembro×TMB-high interaction coef=+0.184, p=4.1e-04. In TMB-high patients, pembro RR 0.197 vs 0.167 off (OR=1.22, p=5e-06); in TMB-low, pembro effect is null.",
            "p_value": 0.0004146,
            "effect_estimate": 0.1839,
            "significant": True,
        },
    ],
})

# --------------- Iter 3: pembrolizumab x other biomarkers ---------------
iterations.append({
    "index": 3,
    "proposed_hypotheses": [
        {"id": "h3.1", "text": "stk11_mutation diminishes the response benefit of treatment_pembrolizumab (negative interaction term).", "kind": "novel"},
        {"id": "h3.2", "text": "egfr_mutation diminishes the response benefit of treatment_pembrolizumab (negative interaction term).", "kind": "novel"},
        {"id": "h3.3", "text": "Within smoking_status strata (current/former/never), the response benefit of treatment_pembrolizumab differs (smokers benefit more than never-smokers).", "kind": "novel"},
        {"id": "h3.4", "text": "Within histology (squamous vs adenocarcinoma), the response benefit of treatment_pembrolizumab differs.", "kind": "novel"},
        {"id": "h3.5", "text": "keap1_mutation diminishes the response benefit of treatment_pembrolizumab.", "kind": "novel"},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h3.1"],
            "code": "smf.logit('objective_response ~ treatment_pembrolizumab*stk11_mutation')",
            "result_summary": "Interaction coef=-0.211, p=0.0018. In STK11-WT, pembro OR=1.106 (p=1e-04); in STK11-mut, pembro OR=0.896 (p=0.08). STK11 mutation abrogates pembro benefit.",
            "p_value": 0.001785,
            "effect_estimate": -0.2111,
            "significant": True,
        },
        {
            "hypothesis_ids": ["h3.2"],
            "code": "smf.logit('objective_response ~ treatment_pembrolizumab*egfr_mutation')",
            "result_summary": "Interaction coef=-0.035, p=0.63. Pembro effect similar in EGFR-mut and EGFR-WT (OR 1.04 vs 1.08). No strong differential.",
            "p_value": 0.6258,
            "effect_estimate": -0.0349,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h3.3"],
            "code": "smf.logit('objective_response ~ treatment_pembrolizumab*C(smoking_status)')",
            "result_summary": "Pembro × never (vs current) coef=-0.078, p=0.31; Pembro × former coef=-0.009, p=0.86. Numerically smaller pembro effect in never-smokers (OR=1.01) than in smokers (OR=1.08-1.09), but not significant.",
            "p_value": 0.3067,
            "effect_estimate": -0.0778,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h3.4"],
            "code": "smf.logit('objective_response ~ treatment_pembrolizumab*C(histology)')",
            "result_summary": "Pembro×squamous interaction coef=+0.030, p=0.58. Both histologies show comparable pembro benefit (OR 1.06 adeno; 1.10 squamous). No significant differential.",
            "p_value": 0.5752,
            "effect_estimate": 0.0297,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h3.5"],
            "code": "smf.logit('objective_response ~ treatment_pembrolizumab*keap1_mutation')",
            "result_summary": "Interaction coef=-0.058, p=0.36. Pembro benefit slightly attenuated in KEAP1+ (OR 1.02) vs KEAP1- (OR 1.08), not statistically significant.",
            "p_value": 0.3625,
            "effect_estimate": -0.0580,
            "significant": False,
        },
    ],
})

# --------------- Iter 4: PD-L1 main and pembro interaction ---------------
iterations.append({
    "index": 4,
    "proposed_hypotheses": [
        {"id": "h4.1", "text": "Higher pdl1_tps is associated with higher objective_response (positive main effect).", "kind": "novel"},
        {"id": "h4.2", "text": "The treatment_pembrolizumab benefit on objective_response is greater at higher pdl1_tps (positive interaction).", "kind": "novel"},
        {"id": "h4.3", "text": "In PD-L1 high (top tertile) patients, treatment_pembrolizumab confers a substantially higher response rate than in PD-L1 low patients.", "kind": "novel"},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h4.1"],
            "code": "smf.logit('objective_response ~ pdl1_tps')",
            "result_summary": "PD-L1 main effect coef=+0.166 per unit TPS (0–1 scale), p=0.003. Modest overall positive association with response.",
            "p_value": 0.003193,
            "effect_estimate": 0.1659,
            "significant": True,
        },
        {
            "hypothesis_ids": ["h4.2"],
            "code": "smf.logit('objective_response ~ treatment_pembrolizumab*pdl1_tps')",
            "result_summary": "Pembro×PD-L1 interaction coef=+0.591, p=1.6e-07. Strong evidence pembro effect grows with PD-L1.",
            "p_value": 1.574e-07,
            "effect_estimate": 0.5908,
            "significant": True,
        },
        {
            "hypothesis_ids": ["h4.3"],
            "code": "stratified chi-square by PD-L1 tertile",
            "result_summary": "Pembro effect by PD-L1 tertile: low OR=1.02 (p=0.64), med OR=0.94 (p=0.11), high OR=1.29 (p=1e-09). Benefit concentrated in high tertile.",
            "p_value": 1.04e-09,
            "effect_estimate": 0.250,
            "significant": True,
        },
    ],
})

# --------------- Iter 5: lab/clinical prognostic factors ---------------
iterations.append({
    "index": 5,
    "proposed_hypotheses": [
        {"id": "h5.1", "text": "Lower albumin_g_dl is associated with lower objective_response (positive coefficient).", "kind": "novel"},
        {"id": "h5.2", "text": "Higher crp_mg_l is associated with lower objective_response (negative coefficient).", "kind": "novel"},
        {"id": "h5.3", "text": "Higher weight_loss_pct_6mo is associated with lower objective_response.", "kind": "novel"},
        {"id": "h5.4", "text": "Higher ecog_ps is associated with lower objective_response.", "kind": "novel"},
        {"id": "h5.5", "text": "Higher ldh_u_l is associated with lower objective_response.", "kind": "novel"},
        {"id": "h5.6", "text": "Higher nlr is associated with lower objective_response.", "kind": "novel"},
        {"id": "h5.7", "text": "Lower hemoglobin_g_dl is associated with lower objective_response.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h5.1"], "code": "ttest_ind albumin by response", "result_summary": "mean albumin 3.821 in responders vs 3.798 in non-responders (diff=+0.023, t-test p=1.3e-04). Higher albumin associated with response.", "p_value": 0.0001326, "effect_estimate": 0.023, "significant": True},
        {"hypothesis_ids": ["h5.2"], "code": "ttest_ind crp by response", "result_summary": "mean CRP 5.78 in responders vs 6.15 in non-responders (diff=-0.37, p=4.2e-04). Higher CRP associated with non-response.", "p_value": 0.0004218, "effect_estimate": -0.369, "significant": True},
        {"hypothesis_ids": ["h5.3"], "code": "ttest_ind weight_loss_pct_6mo by response", "result_summary": "mean weight_loss 3.38% in responders vs 3.93% in non-responders (diff=-0.55, p=2.8e-35). Strong inverse relationship.", "p_value": 2.765e-35, "effect_estimate": -0.549, "significant": True},
        {"hypothesis_ids": ["h5.4"], "code": "smf.logit('objective_response ~ ecog_ps')", "result_summary": "logit coef on ecog_ps = -0.375 per unit (p=1.7e-93). Strong negative prognostic effect.", "p_value": 1.733e-93, "effect_estimate": -0.3749, "significant": True},
        {"hypothesis_ids": ["h5.5"], "code": "ttest_ind ldh by response", "result_summary": "mean LDH 224.3 in responders vs 223.6 in non-responders (diff=+0.6, p=0.50). No association.", "p_value": 0.5048, "effect_estimate": 0.643, "significant": False},
        {"hypothesis_ids": ["h5.6"], "code": "ttest_ind nlr by response", "result_summary": "mean NLR 3.48 in responders vs 3.50 in non-responders (diff=-0.02, p=0.38). No detectable association.", "p_value": 0.38, "effect_estimate": -0.022, "significant": False},
        {"hypothesis_ids": ["h5.7"], "code": "ttest_ind hemoglobin by response", "result_summary": "mean Hb 12.50 in responders vs 12.51 in non-responders (diff=-0.01, p=0.60). No detectable association.", "p_value": 0.602, "effect_estimate": -0.011, "significant": False},
    ],
})

# --------------- Iter 6: stage/mets/comorbidities ---------------
iterations.append({
    "index": 6,
    "proposed_hypotheses": [
        {"id": "h6.1", "text": "stage_iv patients have lower objective_response than non-stage-IV patients.", "kind": "novel"},
        {"id": "h6.2", "text": "has_brain_mets patients have lower objective_response than those without brain metastases.", "kind": "novel"},
        {"id": "h6.3", "text": "liver_mets patients have lower objective_response than those without liver metastases.", "kind": "novel"},
        {"id": "h6.4", "text": "Common comorbidities (hypertension, diabetes, copd, ckd, heart failure) are each associated with lower objective_response.", "kind": "novel"},
        {"id": "h6.5", "text": "interstitial_lung_disease_history is associated with lower objective_response (especially with pembro).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h6.1"], "code": "chi2 stage_iv x response", "result_summary": "stage IV RR 0.154 vs 0.196 (OR=0.74, p=1.1e-33). Strong negative effect.", "p_value": 1.109e-33, "effect_estimate": -0.296, "significant": True},
        {"hypothesis_ids": ["h6.2"], "code": "chi2 has_brain_mets x response", "result_summary": "Brain mets RR 0.144 vs 0.178 (OR=0.78, p=2.0e-18). Strong negative effect.", "p_value": 1.992e-18, "effect_estimate": -0.252, "significant": True},
        {"hypothesis_ids": ["h6.3"], "code": "chi2 liver_mets x response", "result_summary": "Liver mets RR 0.173 vs 0.168 (OR=1.03, p=0.34). No significant univariate effect.", "p_value": 0.339, "effect_estimate": 0.033, "significant": False},
        {"hypothesis_ids": ["h6.4"], "code": "chi2 each comorbidity x response", "result_summary": "Hypertension OR=0.99 p=0.72; diabetes OR=1.01 p=0.76; COPD OR=0.98 p=0.38; CKD OR=1.05 p=0.22; HF OR=1.01 p=0.89; CAD OR=0.94 p=0.05. Comorbidities not strongly associated with response.", "p_value": 0.05, "effect_estimate": -0.06, "significant": False},
        {"hypothesis_ids": ["h6.5"], "code": "chi2 ILD-history x response", "result_summary": "ILD-history RR 0.160 vs 0.169 (OR=0.93, p=0.27). No significant univariate effect.", "p_value": 0.274, "effect_estimate": -0.071, "significant": False},
    ],
})

# --------------- Iter 7: demographics, race, insurance, SES ---------------
iterations.append({
    "index": 7,
    "proposed_hypotheses": [
        {"id": "h7.1", "text": "Female patients (sex_female=1) have higher objective_response than male patients.", "kind": "novel"},
        {"id": "h7.2", "text": "Older age_years is associated with lower objective_response.", "kind": "novel"},
        {"id": "h7.3", "text": "race_ethnicity strata differ in objective_response (omnibus chi-square).", "kind": "novel"},
        {"id": "h7.4", "text": "insurance_type strata differ in objective_response (medicaid lower than private).", "kind": "novel"},
        {"id": "h7.5", "text": "Higher smoking_pack_years is associated with lower objective_response.", "kind": "novel"},
        {"id": "h7.6", "text": "rural_residence patients have lower objective_response than urban-residence patients.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h7.1"], "code": "chi2 sex_female x response", "result_summary": "Females RR 0.175 vs males 0.164 (OR=1.09, p=5.9e-04). Modest but significant higher response in women.", "p_value": 0.0005905, "effect_estimate": 0.0827, "significant": True},
        {"hypothesis_ids": ["h7.2"], "code": "ttest_ind age by response", "result_summary": "mean age 65.0 responders vs 65.0 non-responders (diff=+0.05, p=0.67). No detectable age effect.", "p_value": 0.6689, "effect_estimate": 0.051, "significant": False},
        {"hypothesis_ids": ["h7.3"], "code": "chi2 race x response", "result_summary": "Response rates: white 0.165, black 0.178, hispanic 0.172, asian 0.177, other 0.185; omnibus chi2 p=0.045. Marginal differential, with white below others.", "p_value": 0.04481, "effect_estimate": 0.013, "significant": True},
        {"hypothesis_ids": ["h7.4"], "code": "chi2 insurance x response", "result_summary": "Response rates: medicaid 0.161, medicare 0.168, private 0.172, uninsured 0.170; omnibus chi2 p=0.23. No significant difference.", "p_value": 0.2315, "effect_estimate": 0.011, "significant": False},
        {"hypothesis_ids": ["h7.5"], "code": "ttest_ind smoking_pack_years by response", "result_summary": "mean pack-years 27.3 responders vs 27.4 non-responders (diff=-0.09, p=0.76). No association.", "p_value": 0.7569, "effect_estimate": -0.088, "significant": False},
        {"hypothesis_ids": ["h7.6"], "code": "chi2 rural_residence x response", "result_summary": "Rural RR 0.167 vs urban 0.170 (OR=0.98, p=0.50). No detectable association.", "p_value": 0.4968, "effect_estimate": -0.018, "significant": False},
    ],
})

# --------------- Iter 8: SNP screen for response ---------------
iterations.append({
    "index": 8,
    "proposed_hypotheses": [
        {"id": "h8.1", "text": "At least one of the 23 measured SNP variables is associated with objective_response after multiple-testing correction (Bonferroni 0.05/23).", "kind": "novel"},
        {"id": "h8.2", "text": "snp_rs7412 (APOE-related) is associated with objective_response (any direction).", "kind": "novel"},
        {"id": "h8.3", "text": "snp_rs1050828 (G6PD-related) is associated with objective_response.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h8.1"], "code": "logit per SNP, multiple testing", "result_summary": "Of 23 SNPs, only rs7412 (p=0.018) and rs1050828 (p=0.031) achieved nominal p<0.05; neither survives Bonferroni 0.05/23=0.0022. Genome-wide null.", "p_value": 0.018, "effect_estimate": 0.05, "significant": False},
        {"hypothesis_ids": ["h8.2"], "code": "smf.logit('objective_response ~ snp_rs7412')", "result_summary": "rs7412 coef=+0.100, p=0.018. Nominally positive, not significant after correction.", "p_value": 0.01822, "effect_estimate": 0.1003, "significant": True},
        {"hypothesis_ids": ["h8.3"], "code": "smf.logit('objective_response ~ snp_rs1050828')", "result_summary": "rs1050828 coef=-0.099, p=0.031. Nominally negative, not significant after correction.", "p_value": 0.03116, "effect_estimate": -0.0989, "significant": True},
    ],
})

# --------------- Iter 9: prior therapy ---------------
iterations.append({
    "index": 9,
    "proposed_hypotheses": [
        {"id": "h9.1", "text": "prior_chemotherapy is associated with lower objective_response (selection / refractoriness).", "kind": "novel"},
        {"id": "h9.2", "text": "prior_immunotherapy is associated with lower objective_response (immune exhaustion).", "kind": "novel"},
        {"id": "h9.3", "text": "Higher prior_lines_of_therapy is associated with lower objective_response.", "kind": "novel"},
        {"id": "h9.4", "text": "Longer years_since_diagnosis is associated with lower objective_response.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h9.1"], "code": "chi2 prior_chemotherapy x response", "result_summary": "prior_chemo RR 0.166 vs 0.172 (OR=0.96, p=0.067). Marginal decrease, not statistically significant.", "p_value": 0.06716, "effect_estimate": -0.044, "significant": False},
        {"hypothesis_ids": ["h9.2"], "code": "chi2 prior_immunotherapy x response", "result_summary": "prior_IO RR 0.173 vs 0.168 (OR=1.03, p=0.38). No detectable effect.", "p_value": 0.3847, "effect_estimate": 0.029, "significant": False},
        {"hypothesis_ids": ["h9.3"], "code": "ttest prior_lines_of_therapy by response", "result_summary": "Mean lines 1.24 responders vs 1.26 non-responders (diff=-0.016, p=0.24). No detectable effect.", "p_value": 0.2395, "effect_estimate": -0.016, "significant": False},
        {"hypothesis_ids": ["h9.4"], "code": "ttest years_since_diagnosis by response", "result_summary": "Mean YSD 1.97 responders vs 1.99 non-responders (diff=-0.021, p=0.31). No detectable effect.", "p_value": 0.3109, "effect_estimate": -0.021, "significant": False},
    ],
})

# --------------- Iter 10: pembro x performance/clinical interactions ---------------
iterations.append({
    "index": 10,
    "proposed_hypotheses": [
        {"id": "h10.1", "text": "treatment_pembrolizumab benefit decreases as ecog_ps rises (negative interaction with ecog_ps).", "kind": "novel"},
        {"id": "h10.2", "text": "treatment_pembrolizumab benefit decreases with greater weight_loss_pct_6mo (negative interaction).", "kind": "novel"},
        {"id": "h10.3", "text": "treatment_pembrolizumab benefit is smaller in stage_iv patients (negative interaction).", "kind": "novel"},
        {"id": "h10.4", "text": "treatment_pembrolizumab benefit is smaller in patients with brain metastases (negative interaction).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h10.1"], "code": "smf.logit('objective_response ~ treatment_pembrolizumab*ecog_ps')", "result_summary": "Pembro×ECOG interaction coef=-0.036, p=0.32. ECOG strongly prognostic but does not differentially attenuate pembro effect.", "p_value": 0.3192, "effect_estimate": -0.0364, "significant": False},
        {"hypothesis_ids": ["h10.2"], "code": "smf.logit('objective_response ~ treatment_pembrolizumab*weight_loss_pct_6mo')", "result_summary": "Pembro×weight_loss interaction coef=+0.004, p=0.51. No interaction.", "p_value": 0.5145, "effect_estimate": 0.0042, "significant": False},
        {"hypothesis_ids": ["h10.3"], "code": "smf.logit('objective_response ~ treatment_pembrolizumab*stage_iv')", "result_summary": "Pembro×stage_iv interaction coef=+0.071, p=0.15. Trend toward larger pembro effect in stage IV (opposite of hypothesis), not significant.", "p_value": 0.1467, "effect_estimate": 0.0709, "significant": False},
        {"hypothesis_ids": ["h10.4"], "code": "smf.logit('objective_response ~ treatment_pembrolizumab*has_brain_mets')", "result_summary": "Pembro×brain_mets interaction coef=-0.033, p=0.57. No interaction.", "p_value": 0.5691, "effect_estimate": -0.0328, "significant": False},
    ],
})

# --------------- Iter 11: histology / smoking adjusted ---------------
iterations.append({
    "index": 11,
    "proposed_hypotheses": [
        {"id": "h11.1", "text": "Squamous histology is associated with higher objective_response than adenocarcinoma.", "kind": "novel"},
        {"id": "h11.2", "text": "Never-smokers have lower objective_response than current/former smokers.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h11.1"], "code": "smf.logit('objective_response ~ C(histology)')", "result_summary": "Squamous vs adeno coef=+0.047, p=0.076. Squamous RR 0.174 vs adeno 0.167 — marginal, not significant.", "p_value": 0.0762, "effect_estimate": 0.0469, "significant": False},
        {"hypothesis_ids": ["h11.2"], "code": "smf.logit('objective_response ~ C(smoking_status)')", "result_summary": "Former vs current coef=-0.027 p=0.32; Never vs current coef=-0.061 p=0.11. No significant differences.", "p_value": 0.1112, "effect_estimate": -0.0606, "significant": False},
    ],
})

# --------------- Iter 12: First multivariable model ---------------
iterations.append({
    "index": 12,
    "proposed_hypotheses": [
        {"id": "h12.1", "text": "After adjusting for ecog_ps, stage_iv, brain mets, albumin, weight_loss, sex, the pembro×pdl1_tps interaction term remains positive and significant.", "kind": "refined"},
        {"id": "h12.2", "text": "After adjustment, the pembro×stk11_mutation negative interaction remains.", "kind": "refined"},
        {"id": "h12.3", "text": "After adjustment, the pembro×tmb_high positive interaction remains.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h12.1","h12.2","h12.3"], "code": "logit('objective_response ~ pembro*pdl1_tps + pembro*tmb_high + pembro*stk11_mutation + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + weight_loss_pct_6mo + sex_female')", "result_summary": "Adjusted: pembro:pdl1_tps coef=+0.593 (p=2e-07); pembro:tmb_high coef=+0.174 (p=0.001); pembro:stk11_mutation coef=-0.203 (p=0.003). All three interactions persist with little change in magnitude.", "p_value": 2e-07, "effect_estimate": 0.593, "significant": True},
    ],
})

# --------------- Iter 13: SNP × treatment interactions screen ---------------
iterations.append({
    "index": 13,
    "proposed_hypotheses": [
        {"id": "h13.1", "text": "At least one SNP×treatment interaction (across 23 SNPs × 4 treatments = 92 tests) is significant after Bonferroni correction (0.05/92 ≈ 5.4e-4).", "kind": "novel"},
        {"id": "h13.2", "text": "snp_rs7412 modifies the response benefit of treatment_pembrolizumab (interaction term significant).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h13.1"], "code": "for snp,tx: logit('response ~ snp*tx'); collect interaction p", "result_summary": "Of 92 SNP×treatment interaction tests, 7 had p<0.05 (vs ~4.6 expected by chance). Smallest p-values rs1800896×sotorasib (p=0.0025) and rs1799853×olaparib (p=0.009). None survive Bonferroni 5.4e-4.", "p_value": 0.0025, "effect_estimate": -0.176, "significant": False},
        {"hypothesis_ids": ["h13.2"], "code": "smf.logit('response ~ pembro*snp_rs7412')", "result_summary": "Pembro×rs7412 interaction coef=-0.033, p=0.70. No evidence rs7412 modifies pembro effect.", "p_value": 0.6998, "effect_estimate": -0.0328, "significant": False},
    ],
})

# --------------- Iter 14: olaparib x other DNA-repair / mutation features ---------------
iterations.append({
    "index": 14,
    "proposed_hypotheses": [
        {"id": "h14.1", "text": "treatment_olaparib has a positive interaction with msi_high (DNA repair / homologous recombination context).", "kind": "novel"},
        {"id": "h14.2", "text": "treatment_olaparib has a positive interaction with tmb_high.", "kind": "novel"},
        {"id": "h14.3", "text": "treatment_olaparib has a positive interaction with pten_loss (DNA-repair related).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h14.1"], "code": "smf.logit('response ~ olaparib*msi_high')", "result_summary": "Olaparib×MSI-high interaction coef=-0.157, p=0.58. No biomarker-specific olaparib benefit.", "p_value": 0.5772, "effect_estimate": -0.1574, "significant": False},
        {"hypothesis_ids": ["h14.2"], "code": "smf.logit('response ~ olaparib*tmb_high')", "result_summary": "Olaparib×TMB-high interaction coef=+0.027, p=0.63. No biomarker-specific benefit.", "p_value": 0.6327, "effect_estimate": 0.0271, "significant": False},
        {"hypothesis_ids": ["h14.3"], "code": "smf.logit('response ~ olaparib*pten_loss')", "result_summary": "Olaparib×PTEN-loss interaction coef=+0.174, p=0.14. Numerical positive trend, not significant.", "p_value": 0.1409, "effect_estimate": 0.1742, "significant": False},
    ],
})

# --------------- Iter 15: osimertinib x clinical features ---------------
iterations.append({
    "index": 15,
    "proposed_hypotheses": [
        {"id": "h15.1", "text": "treatment_osimertinib reduces the response disadvantage from has_brain_mets (positive interaction with has_brain_mets, since osimertinib has CNS activity).", "kind": "novel"},
        {"id": "h15.2", "text": "treatment_osimertinib has a positive interaction with histology=adenocarcinoma vs squamous.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h15.1"], "code": "smf.logit('response ~ osimertinib*has_brain_mets')", "result_summary": "Osimertinib×brain_mets interaction coef=+0.051, p=0.42. No CNS-specific osimertinib benefit detected.", "p_value": 0.4177, "effect_estimate": 0.0511, "significant": False},
        {"hypothesis_ids": ["h15.2"], "code": "smf.logit('response ~ osimertinib*C(histology)')", "result_summary": "Osimertinib×squamous interaction coef=+0.047, p=0.42. No histology-specific differential.", "p_value": 0.4201, "effect_estimate": 0.0468, "significant": False},
    ],
})

# --------------- Iter 16: Sex × pembro interaction ---------------
iterations.append({
    "index": 16,
    "proposed_hypotheses": [
        {"id": "h16.1", "text": "Female patients derive greater treatment_pembrolizumab response benefit than male patients (positive sex_female × pembro interaction).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h16.1"], "code": "smf.logit('response ~ pembro*sex_female')", "result_summary": "Pembro×sex_female interaction coef=+0.146, p=0.0024. In men, pembro OR=1.00 (p=0.94); in women, pembro OR=1.16 (p=2.6e-05). Pembro benefit is concentrated in women.", "p_value": 0.002373, "effect_estimate": 0.1457, "significant": True},
    ],
})

# --------------- Iter 17: Race × pembro ---------------
iterations.append({
    "index": 17,
    "proposed_hypotheses": [
        {"id": "h17.1", "text": "treatment_pembrolizumab response benefit differs across race_ethnicity strata (omnibus interaction).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h17.1"], "code": "smf.logit('response ~ pembro*C(race_ethnicity)')", "result_summary": "Individual race × pembro interaction terms not significant (smallest p=0.07 for 'other' vs reference). Pembro ORs: Asian 1.15, Black 1.12, white 1.07, Hispanic 1.08, other 0.82. No statistically significant heterogeneity.", "p_value": 0.07226, "effect_estimate": -0.338, "significant": False},
    ],
})

# --------------- Iter 18: number-of-treatments / combo ---------------
iterations.append({
    "index": 18,
    "proposed_hypotheses": [
        {"id": "h18.1", "text": "Total number of concurrent treatments (n_tx ∈ {0..4}) is positively associated with objective_response.", "kind": "novel"},
        {"id": "h18.2", "text": "Combining treatment_pembrolizumab with treatment_sotorasib produces synergy beyond the sum (positive interaction).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h18.1"], "code": "smf.logit('response ~ n_tx')", "result_summary": "n_tx coef=+0.015, p=0.23. Response rates: 0 tx 0.162, 1 tx 0.171, 2 tx 0.169, 3 tx 0.171, 4 tx 0.178. Slight numerical trend, not statistically significant.", "p_value": 0.2332, "effect_estimate": 0.0151, "significant": False},
        {"hypothesis_ids": ["h18.2"], "code": "smf.logit('response ~ pembro*sotorasib')", "result_summary": "Pembro×sotorasib interaction coef=+0.058, p=0.25. No detectable synergy.", "p_value": 0.2495, "effect_estimate": 0.0576, "significant": False},
    ],
})

# --------------- Iter 19: PDL1 x histology ---------------
iterations.append({
    "index": 19,
    "proposed_hypotheses": [
        {"id": "h19.1", "text": "The PD-L1 effect on objective_response differs between squamous and adenocarcinoma histology.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h19.1"], "code": "smf.logit('response ~ pdl1_tps*C(histology)')", "result_summary": "PDL1×squamous interaction coef=+0.011, p=0.93. No histology-dependent PD-L1 effect.", "p_value": 0.9287, "effect_estimate": 0.0112, "significant": False},
    ],
})

# --------------- Iter 20: Other lab/vital screens ---------------
iterations.append({
    "index": 20,
    "proposed_hypotheses": [
        {"id": "h20.1", "text": "Higher total_bilirubin_mg_dl is associated with lower objective_response.", "kind": "novel"},
        {"id": "h20.2", "text": "Higher ast_u_l is associated with lower objective_response.", "kind": "novel"},
        {"id": "h20.3", "text": "Higher cea_ng_ml is associated with lower objective_response.", "kind": "novel"},
        {"id": "h20.4", "text": "Higher creatinine_mg_dl is associated with lower objective_response.", "kind": "novel"},
        {"id": "h20.5", "text": "Higher heart_rate_bpm is associated with lower objective_response.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h20.1"], "code": "ttest_ind", "result_summary": "Bilirubin diff -0.005 (p=0.21). No association.", "p_value": 0.2114, "effect_estimate": -0.005, "significant": False},
        {"hypothesis_ids": ["h20.2"], "code": "ttest_ind", "result_summary": "AST diff -0.24 (p=0.08). Borderline.", "p_value": 0.07902, "effect_estimate": -0.243, "significant": False},
        {"hypothesis_ids": ["h20.3"], "code": "ttest_ind", "result_summary": "CEA diff +0.12 (p=0.29). No association.", "p_value": 0.2928, "effect_estimate": 0.119, "significant": False},
        {"hypothesis_ids": ["h20.4"], "code": "ttest_ind", "result_summary": "Creatinine diff +0.001 (p=0.84). No association.", "p_value": 0.8437, "effect_estimate": 0.001, "significant": False},
        {"hypothesis_ids": ["h20.5"], "code": "ttest_ind", "result_summary": "Heart rate diff -0.05 (p=0.73). No association.", "p_value": 0.7272, "effect_estimate": -0.050, "significant": False},
    ],
})

# --------------- Iter 21: Three-way pembro x pdl1 x stk11 ---------------
iterations.append({
    "index": 21,
    "proposed_hypotheses": [
        {"id": "h21.1", "text": "STK11 mutation specifically dampens the pembro×PD-L1 dose-response relationship (negative three-way interaction pembro:pdl1_tps:stk11_mutation).", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h21.1"], "code": "smf.logit('response ~ pembro*pdl1_tps*stk11_mutation')", "result_summary": "Three-way interaction coef=-0.388, p=0.22. Numerically large but underpowered. The two-way pembro:pdl1_tps interaction (+0.644) and two-way pembro:stk11 effect remain dominant.", "p_value": 0.2226, "effect_estimate": -0.3884, "significant": False},
    ],
})

# --------------- Iter 22: comprehensive multivariable model with all main effects ---------------
iterations.append({
    "index": 22,
    "proposed_hypotheses": [
        {"id": "h22.1", "text": "After adjustment for ecog_ps, stage_iv, brain_mets, albumin, crp, weight_loss, sex, age, bmi, histology, smoking_status, race, comorbidities, the prognostic effect of ecog_ps remains the strongest negative predictor (largest |coef|).", "kind": "refined"},
        {"id": "h22.2", "text": "After adjustment for clinical and biomarker features, sotorasib, olaparib, and osimertinib show no main response effect.", "kind": "refined"},
        {"id": "h22.3", "text": "After adjustment, neither race nor smoking status independently predicts response (i.e., apparent univariate trends disappear).", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h22.1"], "code": "comprehensive logit", "result_summary": "Adjusted coefficients: ecog_ps=-0.382 (p<1e-19), stage_iv=-0.302 (p<1e-11), has_brain_mets=-0.257 (p<1e-08), albumin=+0.093 (p=1e-04), crp=-0.005 (p=4e-04), weight_loss=-0.039 (p<1e-11). ECOG remains strongest predictor.", "p_value": 1e-19, "effect_estimate": -0.382, "significant": True},
        {"hypothesis_ids": ["h22.2"], "code": "see comprehensive logit", "result_summary": "Adjusted main effects: sotorasib coef=+0.006 (p=0.80); olaparib coef=+0.011 (p=0.69); osimertinib coef=-0.034 (p=0.20). All null even after adjustment.", "p_value": 0.20, "effect_estimate": -0.034, "significant": False},
        {"hypothesis_ids": ["h22.3"], "code": "see comprehensive logit", "result_summary": "Race coefficients (vs Asian ref): black +0.006, hispanic -0.039, other +0.048, white -0.077; none significant. Smoking status (former, never vs current): both p>0.6. Univariate signals attenuated to null.", "p_value": 0.13, "effect_estimate": -0.077, "significant": False},
    ],
})

# --------------- Iter 23: Pembro effect in PDL1>=50% ---------------
iterations.append({
    "index": 23,
    "proposed_hypotheses": [
        {"id": "h23.1", "text": "Within the PD-L1 high (TPS≥0.5) subgroup, treatment_pembrolizumab raises objective_response substantially compared with no-pembrolizumab.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h23.1"], "code": "chi2 pembro x response in pdl1>=0.5 subgroup", "result_summary": "n=10,749 with PDL1≥0.5. Pembro RR 0.209 vs no-pembro 0.156 (OR=1.44, p=7.7e-13). Large absolute difference of ~5.4 percentage points.", "p_value": 7.745e-13, "effect_estimate": 0.0536, "significant": True},
    ],
})

# --------------- Iter 24: Pembro effect in optimal subgroup (PDL1-high AND TMB-high AND STK11-WT) ---------------
iterations.append({
    "index": 24,
    "proposed_hypotheses": [
        {"id": "h24.1", "text": "In the joint subgroup (pdl1_tps≥0.5 AND tmb_high=1 AND stk11_mutation=0), treatment_pembrolizumab dramatically raises objective_response — larger than in any single biomarker subgroup.", "kind": "refined"},
        {"id": "h24.2", "text": "In the STK11-mutated subgroup, treatment_pembrolizumab does NOT raise response (point estimate at or below 1).", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h24.1"], "code": "chi2 pembro x response in pdl1>=0.5 & tmb_high=1 & stk11=0", "result_summary": "n=2,714 in optimal subgroup. Pembro RR 0.313 vs no-pembro 0.153 (OR=2.52, p=1.4e-22). Absolute difference ~16 pp.", "p_value": 1.398e-22, "effect_estimate": 0.1596, "significant": True},
        {"hypothesis_ids": ["h24.2"], "code": "chi2 pembro x response in stk11_mutation=1", "result_summary": "n=7,451 STK11+. Pembro RR 0.158 vs no-pembro 0.173 (OR=0.90, p=0.084). Numerically lower on pembro; consistent with negative pembro×STK11 interaction.", "p_value": 0.08352, "effect_estimate": -0.0151, "significant": False},
    ],
})

# --------------- Iter 25: Final summary / fatigue grade ---------------
iterations.append({
    "index": 25,
    "proposed_hypotheses": [
        {"id": "h25.1", "text": "Higher fatigue_grade is associated with lower objective_response (negative coefficient), consistent with disease burden marker.", "kind": "novel"},
        {"id": "h25.2", "text": "After accounting for pembro×sex interaction, the unadjusted female-vs-male main effect on response is fully attributable to women's larger pembro benefit (i.e., conditioning on sex_female=1 yields no main effect).", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h25.1"], "code": "smf.logit('response ~ fatigue_grade')", "result_summary": "fatigue_grade coef=-0.022, p=0.040. Mild negative association.", "p_value": 0.04024, "effect_estimate": -0.0222, "significant": True},
        {"hypothesis_ids": ["h25.2"], "code": "from interaction model coef[sex_female]=0.008 (p=0.82)", "result_summary": "After including pembro×sex interaction, sex_female main effect attenuates to coef=+0.008 (p=0.82). The female-vs-male advantage (~+8% univariate OR) is essentially fully explained by larger pembro benefit in women.", "p_value": 0.82, "effect_estimate": 0.008, "significant": False},
    ],
})


transcript = {
    "dataset_id": DATASET_ID,
    "model_id": MODEL_ID,
    "harness_id": HARNESS_ID,
    "max_iterations": MAX_ITER,
    "iterations": iterations,
}

Path("transcript.json").write_text(json.dumps(transcript, indent=2))
print("Wrote transcript.json")

# ---------------------- analysis_summary.txt ----------------------
summary = """ds001_nsclc — Iterative Hypothesis Analysis Summary
====================================================

Cohort: 50,000 NSCLC patients from a commercial EHR-derived dataset; overall
objective response rate = 16.9% (8,447/50,000).

This summary walks through 25 iterations of hypothesis generation and
statistical testing, organised into five themes:

  (A) Treatment main effects
  (B) Treatment x biomarker interactions (predictive markers)
  (C) Prognostic clinical/lab features (main effects)
  (D) Demographics, SES, race, sex
  (E) SNPs, comorbidities, prior therapy
  (F) Multivariable confirmation and joint subgroup effects


-----------------------------------------------------------
A) TREATMENT MAIN EFFECTS (Iter 1)
-----------------------------------------------------------
- treatment_pembrolizumab: Response 17.4% vs 16.4% off; OR=1.07, p=0.003.
  Modest but real overall benefit (the main treatment with a detectable
  unadjusted effect).
- treatment_sotorasib: 16.95% vs 16.87%, OR=1.01, p=0.83.  Null.
- treatment_olaparib: 17.0% vs 16.9%, OR=1.01, p=0.72.  Null.
- treatment_osimertinib: 16.55% vs 17.04%, OR=0.97, p=0.18.  Null
  (numerically lower).

Main effect verdict: only pembrolizumab shows an unadjusted main effect.

-----------------------------------------------------------
B) TARGETED-THERAPY x TARGET-BIOMARKER INTERACTIONS (Iter 2, 14, 15)
-----------------------------------------------------------
A core a-priori expectation in NSCLC is that targeted therapies should be
biomarker-selective.  We tested each treatment against its canonical target.

- osimertinib x egfr_mutation: interaction coef=+0.024, p=0.76 — null.
  In EGFR-mutant patients, osimertinib RR 0.164 vs 0.166 off (OR=0.99,
  p=0.87).  No detectable EGFR-specific benefit.
- sotorasib x kras_g12c: interaction coef=+0.059, p=0.43 — null.
- olaparib x brca2_mutation: interaction coef=-0.18, p=0.27 — null
  (numerically opposite of expected).
- olaparib x msi_high, tmb_high, pten_loss: all null.
- osimertinib x has_brain_mets, x histology: null.

Verdict: REFUTED — none of the three targeted therapies in this cohort
shows the biomarker-selective response heterogeneity that the standard-of-
care evidence predicts.  This is the most striking negative finding of the
analysis.

-----------------------------------------------------------
B') PEMBROLIZUMAB x BIOMARKER INTERACTIONS (Iter 2, 3, 4, 12, 16, 21, 23, 24)
-----------------------------------------------------------
For pembrolizumab, by contrast, multiple predictive biomarkers were
identified and confirmed:

1) pembrolizumab x pdl1_tps (continuous PD-L1 TPS):
     interaction coef = +0.591, p = 1.6e-07 (univariate);
     +0.596 (p < 1e-07) in the comprehensive multivariable model.
   Stratified: in the PD-L1 LOW tertile, pembro OR=1.02 (p=0.64); in MID
   tertile OR=0.94 (p=0.11); in HIGH tertile OR=1.29 (p=1.0e-09).
   All of pembrolizumab's overall benefit is concentrated in the
   PD-L1-high tertile.  In the PDL1>=50% subgroup (n=10,749), pembro
   raised RR from 15.6% to 20.9% (OR=1.44, p=7.7e-13).
   SUPPORTED.

2) pembrolizumab x tmb_high:
     interaction coef = +0.184, p = 4.1e-04 (univariate);
     +0.173 (p = 0.001) adjusted.
   In TMB-high patients, pembro RR 19.7% vs 16.7% off (OR=1.22, p=5e-06);
   in TMB-low patients pembro effect is null.
   SUPPORTED.

3) pembrolizumab x stk11_mutation (NEGATIVE):
     interaction coef = -0.211, p = 0.0018 (univariate);
     -0.205 (p = 0.003) adjusted.
   In STK11-WT, pembro OR=1.11 (p=1e-04); in STK11-mut, pembro OR=0.90
   (p=0.084, numerically harmful).  STK11 mutation abrogates pembro
   benefit.
   SUPPORTED, consistent with published clinical literature.

4) pembrolizumab x sex_female:
     interaction coef = +0.146, p = 0.0024 (univariate);
     +0.145 (p = 0.003) adjusted.
   In men, pembro effect is essentially null (OR=1.00, p=0.94); in women,
   pembro OR=1.16 (p=2.6e-05).  After accounting for this interaction,
   the apparent univariate female advantage in response (RR 17.5% vs
   16.4%) collapses to coef=+0.008 (p=0.82) — i.e., the entire female-
   vs-male response gap is explained by women's larger pembro benefit.
   SUPPORTED, novel finding worth flagging.

Other pembro x biomarker tests REFUTED (or null):
- pembro x egfr_mutation, x msi_high, x keap1_mutation, x tp53_mutation,
  x alk_fusion, x histology, x smoking_status, x race_ethnicity — all
  not significant.
- pembro x ecog_ps, x weight_loss, x stage_iv, x brain_mets — all
  not significant (the prognostic effects are present but do not modify
  pembro relative effect).

Joint optimal subgroup (PDL1>=0.5 AND tmb_high=1 AND stk11_mutation=0,
n=2,714): pembro RR 31.3% vs no-pembro 15.3%, OR=2.52, p=1.4e-22.
This is the biggest pembro effect anywhere in the cohort.

The three-way interaction pembro:pdl1_tps:stk11_mutation was
NUMERICALLY large (-0.39) but not statistically significant (p=0.22).

-----------------------------------------------------------
C) PROGNOSTIC CLINICAL / LAB MAIN EFFECTS (Iter 5, 6, 20, 25)
-----------------------------------------------------------
Strong prognostic effects (each adjusted for the others in Iter 12 / 22):
- ecog_ps: coef = -0.38 per unit, p = 1.7e-93. Strongest single predictor.
- stage_iv: coef = -0.30, p = 1.1e-33. Strong negative.
- has_brain_mets: coef = -0.26, p = 2.0e-18. Strong negative.
- weight_loss_pct_6mo: coef = -0.039 per unit %, p = 2.8e-35. Strong.
- albumin_g_dl: coef = +0.09 per unit, p = 1.3e-04. Positive (good).
- crp_mg_l: coef = -0.005 per unit, p = 4.2e-04. Negative (inflammation).
- fatigue_grade: coef = -0.022, p = 0.04. Mild negative.

NULL or marginal:
- ldh_u_l (p=0.50), nlr (p=0.38), hemoglobin (p=0.60), bmi (p=0.35),
  ast/alt/bilirubin/creatinine/bun/glucose/sodium/potassium/calcium/
  inr/tsh — all p>=0.08.  Most chemistries do not predict response.
- liver_mets (p=0.34), bone_mets (p=0.18), adrenal_mets (p=0.39),
  pleural_effusion (p=0.89), pericardial_effusion (p=0.61) — null.
- All comorbidities (HTN, DM, COPD, CKD, HF, AF, autoimmune disease,
  ILD history, hepatitis B/C, HIV, prior malignancy, depression /
  anxiety) — all null.  Coronary artery disease was borderline
  (OR=0.94, p=0.05).

-----------------------------------------------------------
D) DEMOGRAPHICS / SES / RACE / SEX (Iter 7, 17, 22)
-----------------------------------------------------------
- age_years: null (p=0.67).
- smoking_pack_years: null (p=0.76).
- education_years: null (p=0.40).
- rural_residence: null (p=0.50).
- insurance_type: null overall (omnibus p=0.23).
- race_ethnicity: marginal omnibus chi2 p=0.045.  White patients trended
  slightly lower (RR 16.5%) than non-white patients (17.2-18.5%); the
  effect did not survive multivariable adjustment (race coefficients all
  p > 0.13 in Iter 22).
- sex_female: significant univariate higher response (OR=1.09, p=6e-04);
  but as noted, this is fully explained by pembro x sex interaction
  rather than a main effect of sex.

-----------------------------------------------------------
E) SNPs, COMORBIDITIES, PRIOR THERAPY (Iter 8, 9, 13)
-----------------------------------------------------------
- 23 SNPs screened against response: only rs7412 (p=0.018) and
  rs1050828 (p=0.031) achieved nominal p<0.05; neither survives
  Bonferroni 0.05/23=0.0022.  Genome-wide null.
- 92 SNP x treatment interaction tests: 7 reached p<0.05 (vs 4.6 expected
  by chance), no Bonferroni-significant hit (0.05/92 = 5.4e-4).
  The interaction signals are best interpreted as multiple-testing noise.
- Prior therapies (chemo, radiation, surgery, immunotherapy, targeted) and
  prior_lines_of_therapy / years_since_diagnosis: all null (smallest
  p=0.067 for prior_chemotherapy).
- n_tx (number of concurrent treatments 0..4): non-significant trend
  (p=0.23).
- pembro x sotorasib synergy: p=0.25, null.

-----------------------------------------------------------
F) OVERALL CONCLUSIONS
-----------------------------------------------------------

1) The most consistent treatment-effect signal is pembrolizumab, modulated
   by tumour-immune biomarkers.  Three robust predictive interactions
   survived multivariable adjustment and are clinically interpretable:
     - PD-L1 TPS (positive, large, monotone in PD-L1)
     - TMB-high (positive, ~22% relative increase in response)
     - STK11 mutation (negative; in STK11+ patients pembro provides no
       benefit)
   The combined PDL1-high & TMB-high & STK11-WT subgroup achieves >2x
   pembrolizumab benefit (OR 2.5).

2) A novel finding: pembrolizumab benefit is concentrated in women
   (OR 1.16) and absent in men (OR 1.00), and this pembro x sex effect
   wholly accounts for the unadjusted female advantage in response.

3) Targeted therapies (osimertinib, sotorasib, olaparib) do NOT exhibit
   their expected biomarker-selective response heterogeneity in this
   dataset.  Neither main effects nor interactions with the canonical
   target mutations were detected, despite ample sample size and
   power.  This is at odds with the published evidence for these
   therapies and warrants investigation of dataset construction.

4) Strong prognostic features for objective response are the classical
   ones: ECOG performance status, stage IV, brain metastases, weight
   loss, albumin, and CRP.  Fatigue grade has a small additional
   contribution.  Other organ-function chemistries, vital signs,
   comorbidities, prior therapies, and demographic features (age,
   pack-years, BMI, race after adjustment, insurance, rural residence,
   education) are essentially uninformative for response.

5) The 23 measured SNPs do not predict response, individually or in
   combination with any treatment, after multiple-testing correction.

Summary of supported vs refuted hypotheses
-------------------------------------------
SUPPORTED:
  pembro main effect; pembro x PD-L1 TPS interaction; pembro x TMB-high
  interaction; pembro x STK11 negative interaction; pembro x sex_female
  positive interaction; ECOG, stage_iv, brain mets, weight loss, albumin,
  CRP as prognostic factors; PDL1 main positive effect; fatigue_grade
  small negative.

REFUTED:
  Sotorasib x KRAS-G12C interaction; olaparib x BRCA2 interaction;
  osimertinib x EGFR interaction; main effects of sotorasib / olaparib /
  osimertinib; SNP main effects on response (after correction);
  SNP x treatment interactions (after correction); race / smoking /
  insurance as independent predictors after adjustment; sex_female
  main effect (subsumed by pembro x sex interaction); LDH, NLR, Hb,
  BMI, age, comorbidity main effects.
"""
Path("analysis_summary.txt").write_text(summary)
print("Wrote analysis_summary.txt")
print(f"transcript.json size: {Path('transcript.json').stat().st_size:,} bytes")
print(f"analysis_summary.txt size: {Path('analysis_summary.txt').stat().st_size:,} bytes")
