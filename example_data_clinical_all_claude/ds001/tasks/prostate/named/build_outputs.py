"""Build transcript.json and analysis_summary.txt from documented analysis results."""
import json

iterations = []

# --- Iteration 1: treatment main effects ---
iterations.append({
    "index": 1,
    "proposed_hypotheses": [
        {"id": "h1.1", "text": "Patients receiving treatment_enzalutamide have a higher rate of objective_response than patients not receiving treatment_enzalutamide.", "kind": "novel"},
        {"id": "h1.2", "text": "Patients receiving treatment_abiraterone have a different rate of objective_response than patients not receiving treatment_abiraterone.", "kind": "novel"},
        {"id": "h1.3", "text": "Patients receiving treatment_docetaxel have a different rate of objective_response than patients not receiving treatment_docetaxel.", "kind": "novel"},
        {"id": "h1.4", "text": "Patients receiving treatment_olaparib have a different rate of objective_response than patients not receiving treatment_olaparib.", "kind": "novel"},
        {"id": "h1.5", "text": "Patients receiving treatment_lu177_psma have a different rate of objective_response than patients not receiving treatment_lu177_psma.", "kind": "novel"},
        {"id": "h1.6", "text": "Patients receiving treatment_pembrolizumab have a different rate of objective_response than patients not receiving treatment_pembrolizumab.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h1.1"], "code": "chi2 of treatment_enzalutamide vs objective_response",
         "result_summary": "treatment_enzalutamide+: rr=0.3614 (n=20076); -: rr=0.1589 (n=29924); diff=+0.2024.",
         "p_value": 0.0, "effect_estimate": 0.2024, "significant": True},
        {"hypothesis_ids": ["h1.2"], "code": "chi2 of treatment_abiraterone vs objective_response",
         "result_summary": "treatment_abiraterone+: rr=0.2381 (n=14991); -: rr=0.2411 (n=35009); diff=-0.0031.",
         "p_value": 0.470, "effect_estimate": -0.0031, "significant": False},
        {"hypothesis_ids": ["h1.3"], "code": "chi2 of treatment_docetaxel vs objective_response",
         "result_summary": "treatment_docetaxel+: rr=0.2399 (n=15187); -: rr=0.2404 (n=34813); diff=-0.0005.",
         "p_value": 0.914, "effect_estimate": -0.0005, "significant": False},
        {"hypothesis_ids": ["h1.4"], "code": "chi2 of treatment_olaparib vs objective_response",
         "result_summary": "treatment_olaparib+: rr=0.2393 (n=5098); -: rr=0.2403 (n=44902); diff=-0.0010.",
         "p_value": 0.886, "effect_estimate": -0.0010, "significant": False},
        {"hypothesis_ids": ["h1.5"], "code": "chi2 of treatment_lu177_psma vs objective_response",
         "result_summary": "treatment_lu177_psma+: rr=0.2427 (n=7504); -: rr=0.2398 (n=42496); diff=+0.0029.",
         "p_value": 0.600, "effect_estimate": 0.0029, "significant": False},
        {"hypothesis_ids": ["h1.6"], "code": "chi2 of treatment_pembrolizumab vs objective_response",
         "result_summary": "treatment_pembrolizumab+: rr=0.2387 (n=2384); -: rr=0.2403 (n=47616); diff=-0.0016.",
         "p_value": 0.876, "effect_estimate": -0.0016, "significant": False},
    ],
})

# --- Iteration 2: feature main effects (binary + ordinal) ---
iterations.append({
    "index": 2,
    "proposed_hypotheses": [
        {"id": "h2.1", "text": "Patients with mcrpc=1 have a lower objective_response rate than patients with mcrpc=0.", "kind": "novel"},
        {"id": "h2.2", "text": "Patients with visceral_mets=1 have a lower objective_response rate than patients with visceral_mets=0.", "kind": "novel"},
        {"id": "h2.3", "text": "Patients with brca2_mutation=1 have a lower objective_response rate than patients with brca2_mutation=0.", "kind": "novel"},
        {"id": "h2.4", "text": "Patients with ar_v7_positive=1 have a lower objective_response rate than patients with ar_v7_positive=0.", "kind": "novel"},
        {"id": "h2.5", "text": "Patients with msi_high=1 have a different objective_response rate than patients with msi_high=0.", "kind": "novel"},
        {"id": "h2.6", "text": "Patients with psma_high=1 have a different objective_response rate than patients with psma_high=0.", "kind": "novel"},
        {"id": "h2.7", "text": "Higher ecog_ps is associated with lower objective_response rates.", "kind": "novel"},
        {"id": "h2.8", "text": "Higher gleason_score is associated with lower objective_response rates.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h2.1"], "result_summary": "mcrpc+: rr=0.1535 (n=27481); mcrpc-: rr=0.3460 (n=22519); diff=-0.1925.", "p_value": 0.0, "effect_estimate": -0.1925, "significant": True},
        {"hypothesis_ids": ["h2.2"], "result_summary": "visceral+: rr=0.2395 (n=9973); visceral-: rr=0.2404 (n=40027); diff=-0.0008.", "p_value": 0.871, "effect_estimate": -0.0008, "significant": False},
        {"hypothesis_ids": ["h2.3"], "result_summary": "brca2+: rr=0.1497 (n=4996); brca2-: rr=0.2503 (n=45004); diff=-0.1005.", "p_value": 5.43e-56, "effect_estimate": -0.1005, "significant": True},
        {"hypothesis_ids": ["h2.4"], "result_summary": "ar_v7+: rr=0.1599 (n=10038); ar_v7-: rr=0.2604 (n=39962); diff=-0.1005.", "p_value": 1.90e-98, "effect_estimate": -0.1005, "significant": True},
        {"hypothesis_ids": ["h2.5"], "result_summary": "msi_high+: rr=0.1760 (n=1528); msi_high-: rr=0.2422 (n=48472); diff=-0.0662.", "p_value": 2.97e-9, "effect_estimate": -0.0662, "significant": True},
        {"hypothesis_ids": ["h2.6"], "result_summary": "psma_high+: rr=0.2386 (n=29962); psma_high-: rr=0.2426 (n=20038); diff=-0.0040.", "p_value": 0.305, "effect_estimate": -0.0040, "significant": False},
        {"hypothesis_ids": ["h2.7"], "result_summary": "ECOG 0 rr=0.282 (n=17592), ECOG 1 rr=0.228 (n=24971), ECOG 2 rr=0.182 (n=7437); chi2 trend p=5.4e-72; per-1-step ECOG diff approx -0.05.", "p_value": 5.4e-72, "effect_estimate": -0.0500, "significant": True},
        {"hypothesis_ids": ["h2.8"], "result_summary": "Gleason 6,7,8,9,10 rr=0.243,0.237,0.242,0.242,0.242; chi2 p=0.77; effectively flat.", "p_value": 0.77, "effect_estimate": -0.001, "significant": False},
    ],
})

# --- Iteration 3: continuous feature main effects ---
iterations.append({
    "index": 3,
    "proposed_hypotheses": [
        {"id": "h3.1", "text": "Higher psa_ng_ml is associated with lower objective_response.", "kind": "novel"},
        {"id": "h3.2", "text": "Lower albumin_g_dl is associated with lower objective_response.", "kind": "novel"},
        {"id": "h3.3", "text": "Higher weight_loss_pct_6mo is associated with lower objective_response.", "kind": "novel"},
        {"id": "h3.4", "text": "Higher crp_mg_l is associated with lower objective_response.", "kind": "novel"},
        {"id": "h3.5", "text": "Higher ldh_u_l is associated with lower objective_response.", "kind": "novel"},
        {"id": "h3.6", "text": "Higher nlr is associated with lower objective_response.", "kind": "novel"},
        {"id": "h3.7", "text": "Lower hemoglobin_g_dl is associated with lower objective_response.", "kind": "novel"},
        {"id": "h3.8", "text": "age_years is associated with objective_response.", "kind": "novel"},
        {"id": "h3.9", "text": "Higher alkaline_phosphatase_u_l is associated with lower objective_response.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h3.1"], "result_summary": "Mean PSA in responders 28.2 vs 47.2 in non-responders; t-test p=1.5e-137; diff=-19.0 ng/mL.", "p_value": 1.5e-137, "effect_estimate": -18.989, "significant": True},
        {"hypothesis_ids": ["h3.2"], "result_summary": "Mean albumin 3.820 in responders vs 3.796 in non-responders; t-test p=3.6e-6; diff=+0.024 g/dL.", "p_value": 3.6e-6, "effect_estimate": 0.024, "significant": True},
        {"hypothesis_ids": ["h3.3"], "result_summary": "Mean weight loss 3.50% in responders vs 3.95% non-responders; t-test p=6.8e-31; diff=-0.454.", "p_value": 6.8e-31, "effect_estimate": -0.454, "significant": True},
        {"hypothesis_ids": ["h3.4"], "result_summary": "Mean CRP 5.69 in responders vs 6.23 in non-responders; t-test p=9.3e-9; diff=-0.533 mg/L.", "p_value": 9.3e-9, "effect_estimate": -0.533, "significant": True},
        {"hypothesis_ids": ["h3.5"], "result_summary": "Mean LDH 223.8 in responders vs 224.1 in non-responders; t-test p=0.68; effectively null.", "p_value": 0.68, "effect_estimate": -0.347, "significant": False},
        {"hypothesis_ids": ["h3.6"], "result_summary": "Mean NLR 3.486 in responders vs 3.497 in non-responders; t-test p=0.61; null.", "p_value": 0.61, "effect_estimate": -0.011, "significant": False},
        {"hypothesis_ids": ["h3.7"], "result_summary": "Mean hemoglobin 12.50 vs 12.51; t-test p=0.51; null.", "p_value": 0.51, "effect_estimate": -0.012, "significant": False},
        {"hypothesis_ids": ["h3.8"], "result_summary": "Mean age 65.0 in both groups; t-test p=0.82; null.", "p_value": 0.82, "effect_estimate": 0.024, "significant": False},
        {"hypothesis_ids": ["h3.9"], "result_summary": "Mean ALP 104.7 vs 104.2; t-test p=0.34; null.", "p_value": 0.34, "effect_estimate": 0.498, "significant": False},
    ],
})

# --- Iteration 4: Multivariable adjusted main effects ---
iterations.append({
    "index": 4,
    "proposed_hypotheses": [
        {"id": "h4.1", "text": "After controlling for all features and treatments, treatment_enzalutamide retains a significant positive effect on objective_response.", "kind": "refined"},
        {"id": "h4.2", "text": "After adjustment, treatment_abiraterone, treatment_docetaxel, treatment_olaparib, treatment_lu177_psma, and treatment_pembrolizumab each have a near-zero average effect on objective_response.", "kind": "refined"},
        {"id": "h4.3", "text": "After adjustment, mcrpc, ar_v7_positive, brca2_mutation, msi_high, ecog_ps, log(psa), log(crp), and weight_loss_pct_6mo remain independently negatively associated with objective_response, while albumin remains positively associated.", "kind": "refined"},
        {"id": "h4.4", "text": "After adjustment, gleason_score, age_years, hemoglobin_g_dl, ldh_u_l, alkaline_phosphatase_u_l, ast/alt/bilirubin/creatinine/bun and the electrolytes have no independent association with objective_response.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h4.1"], "code": "Logit(objective_response ~ all features + treatments)",
         "result_summary": "Adjusted logit coefficient for treatment_enzalutamide = +1.202 (z=52.7, p<1e-300).",
         "p_value": 0.0, "effect_estimate": 1.202, "significant": True},
        {"hypothesis_ids": ["h4.2"], "result_summary": "Adjusted logit coefficients (p): abiraterone -0.025 (0.31); docetaxel +0.002 (0.94); olaparib +0.026 (0.49); lu177_psma +0.030 (0.34); pembrolizumab +0.014 (0.79).",
         "p_value": 0.31, "effect_estimate": 0.0, "significant": False},
        {"hypothesis_ids": ["h4.3"], "result_summary": "Adjusted logit coefficients: mcrpc -1.07 (p~0); ar_v7_positive -0.69 (p=2e-108); brca2_mutation -0.74 (p=4e-64); msi_high -0.43 (p=3e-9); ecog_ps -0.33 per step (p=1e-82); log(PSA) -0.082 (p=2e-12); log(CRP) -0.090 (p=2e-10); weight_loss -0.037/% (p=1e-33); albumin +0.126/g/dL (p=2e-8).",
         "p_value": 1e-8, "effect_estimate": -1.07, "significant": True},
        {"hypothesis_ids": ["h4.4"], "result_summary": "Adjusted p-values for gleason_score 0.98, age 0.73, Hb 0.72, LDH 0.65, ALP 0.14, AST 0.07, ALT 0.59, bilirubin 0.19, creatinine 0.72, BUN 0.93, sodium 0.66, potassium 0.08, calcium 0.69 — none significant.",
         "p_value": 0.5, "effect_estimate": 0.0, "significant": False},
    ],
})

# --- Iteration 5: Targeted predictive biomarker x treatment interactions ---
iterations.append({
    "index": 5,
    "proposed_hypotheses": [
        {"id": "h5.1", "text": "treatment_olaparib increases objective_response more in patients with brca2_mutation=1 than in those with brca2_mutation=0 (PARP-inhibitor synthetic lethality).", "kind": "novel"},
        {"id": "h5.2", "text": "treatment_pembrolizumab increases objective_response more in patients with msi_high=1 than in those with msi_high=0 (immunotherapy in MSI-H tumors).", "kind": "novel"},
        {"id": "h5.3", "text": "treatment_lu177_psma increases objective_response more in patients with psma_high=1 than in those with psma_high=0 (PSMA-targeted radioligand).", "kind": "novel"},
        {"id": "h5.4", "text": "treatment_enzalutamide increases objective_response less (or not at all) in patients with ar_v7_positive=1 than in those with ar_v7_positive=0 (AR-V7 confers AR-antagonist resistance).", "kind": "novel"},
        {"id": "h5.5", "text": "treatment_abiraterone increases objective_response less in patients with ar_v7_positive=1 than in those with ar_v7_positive=0.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h5.1"], "code": "Subgroup analysis + logit interaction enz*brca2",
         "result_summary": "In BRCA2+: olaparib rr=0.121 (n=529) vs 0.153 (n=4467); diff=-0.032. In BRCA2-: diff=+0.003. Olaparib does NOT preferentially help BRCA2+ patients in this dataset; if anything trend is negative.",
         "p_value": 0.058, "effect_estimate": -0.0321, "significant": False},
        {"hypothesis_ids": ["h5.2"], "result_summary": "In MSI-high: pembrolizumab rr=0.177 (n=79) vs 0.176 (n=1449); diff=+0.001 (Fisher p~1.0). In MSI-low: diff=-0.001. No subgroup benefit.",
         "p_value": 1.0, "effect_estimate": 0.0012, "significant": False},
        {"hypothesis_ids": ["h5.3"], "result_summary": "In PSMA-high: lu177_psma rr=0.238 (n=4486) vs 0.239 (n=25476); diff=-0.001 (p=0.88). In PSMA-low: diff=+0.009. No PSMA-conditional benefit.",
         "p_value": 0.88, "effect_estimate": -0.0011, "significant": False},
        {"hypothesis_ids": ["h5.4"], "result_summary": "In AR-V7-: enzalutamide rr=0.411 vs 0.160 (diff=+0.251, p<1e-300). In AR-V7+: rr=0.165 vs 0.157 (diff=+0.009, p=0.27). Interaction logit coef -1.33 (p=1.4e-101) — strong negative modification.",
         "p_value": 1.4e-101, "effect_estimate": -1.327, "significant": True},
        {"hypothesis_ids": ["h5.5"], "result_summary": "In AR-V7-: abiraterone rr=0.257 vs 0.262 (diff=-0.005). In AR-V7+: 0.163 vs 0.158 (diff=+0.005). Interaction logit coef +0.07 (p=0.26). No effect.",
         "p_value": 0.26, "effect_estimate": 0.0741, "significant": False},
    ],
})

# --- Iteration 6: enzalutamide x mCRPC interaction ---
iterations.append({
    "index": 6,
    "proposed_hypotheses": [
        {"id": "h6.1", "text": "treatment_enzalutamide increases objective_response substantially more in patients with mcrpc=0 (hormone-sensitive disease) than in patients with mcrpc=1 (castration-resistant).", "kind": "novel"},
        {"id": "h6.2", "text": "treatment_enzalutamide effect is similar across visceral_mets status.", "kind": "novel"},
        {"id": "h6.3", "text": "treatment_enzalutamide effect is similar across ecog_ps levels.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h6.1"], "code": "Stratified rates + logit enz*mcrpc",
         "result_summary": "mcrpc=0: enz rr=0.610 vs 0.169, diff=+0.440 (p<1e-300). mcrpc=1: 0.158 vs 0.151, diff=+0.008 (p=0.09). Adjusted enz*mcrpc interaction coef = -2.02 (p<1e-300).",
         "p_value": 0.0, "effect_estimate": -2.0231, "significant": True},
        {"hypothesis_ids": ["h6.2"], "result_summary": "visceral=0: enz diff=+0.203; visceral=1: enz diff=+0.200. Interaction coef -0.009 (p=0.86). No modification.",
         "p_value": 0.86, "effect_estimate": -0.0094, "significant": False},
        {"hypothesis_ids": ["h6.3"], "result_summary": "ECOG 0,1,2 enz diffs +0.207, +0.201, +0.198 — essentially identical. Interaction logit coef +0.14 per step (p=1.6e-5; small in magnitude).",
         "p_value": 1.6e-5, "effect_estimate": 0.1414, "significant": True},
    ],
})

# --- Iteration 7: Joint AR-V7 x mCRPC enzalutamide subgroup ---
iterations.append({
    "index": 7,
    "proposed_hypotheses": [
        {"id": "h7.1", "text": "Within the joint subgroup ar_v7_positive=0 AND mcrpc=0, treatment_enzalutamide produces the largest absolute increase in objective_response, while in ar_v7_positive=0 AND mcrpc=1 the effect is essentially absent.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h7.1"],
         "result_summary": "AR-V7=0 & mcrpc=0 (n=18000): enz rr=0.717 vs 0.171, diff=+0.546. AR-V7=0 & mcrpc=1 (n=21962): rr=0.159 vs 0.150, diff=+0.008.",
         "p_value": 0.0, "effect_estimate": 0.5459, "significant": True},
    ],
})

# --- Iteration 8: systematic interaction screen ---
iterations.append({
    "index": 8,
    "proposed_hypotheses": [
        {"id": "h8.1", "text": "Among all 6 treatments × 7 candidate modifiers (mcrpc, visceral_mets, brca2_mutation, ar_v7_positive, msi_high, psma_high, ecog_ps), only treatment_enzalutamide shows large modifier interactions on objective_response, and its strongest negative modifiers are mcrpc, ar_v7_positive, brca2_mutation, and msi_high.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h8.1"], "code": "42 logit(objective_response ~ tx + mod + tx*mod) regressions",
         "result_summary": "Top interactions all involve enzalutamide: enz*mcrpc -1.978 (p<1e-300), enz*ar_v7 -1.237 (p=1.4e-93), enz*brca2 -1.138 (p=3.3e-42), enz*msi_high -1.283 (p=1.8e-19), enz*ecog +0.141 (p=1.6e-5). All non-enzalutamide interactions p>0.01 except marginal lu177*visceral (p=0.01) and abiraterone*brca2 (p=0.03), neither survives multiple testing.",
         "p_value": 0.0, "effect_estimate": -1.978, "significant": True},
    ],
})

# --- Iteration 9: secondary modifiers within enz-responsive core ---
iterations.append({
    "index": 9,
    "proposed_hypotheses": [
        {"id": "h9.1", "text": "Within the AR-V7- AND mcrpc=0 subgroup, brca2_mutation=1 abolishes the treatment_enzalutamide benefit.", "kind": "refined"},
        {"id": "h9.2", "text": "Within the AR-V7- AND mcrpc=0 subgroup, msi_high=1 abolishes the treatment_enzalutamide benefit.", "kind": "refined"},
        {"id": "h9.3", "text": "Within the AR-V7- AND mcrpc=0 subgroup, psma_high status does NOT meaningfully modify the treatment_enzalutamide benefit.", "kind": "novel"},
        {"id": "h9.4", "text": "Within the AR-V7- AND mcrpc=0 subgroup, visceral_mets does NOT meaningfully modify the treatment_enzalutamide benefit.", "kind": "novel"},
        {"id": "h9.5", "text": "Within the AR-V7- AND mcrpc=0 subgroup, ecog_ps does NOT meaningfully modify the treatment_enzalutamide benefit.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h9.1"], "result_summary": "Core (n=18000): brca2-: enz diff=+0.607; brca2+: enz diff=+0.004. Interaction coef -2.80 (p=6.6e-90).",
         "p_value": 6.6e-90, "effect_estimate": -2.798, "significant": True},
        {"hypothesis_ids": ["h9.2"], "result_summary": "Core: msi-: enz diff=+0.563; msi-high: enz diff=-0.022. Interaction coef -2.74 (p=9.6e-34).",
         "p_value": 9.6e-34, "effect_estimate": -2.738, "significant": True},
        {"hypothesis_ids": ["h9.3"], "result_summary": "Core: psma_high=0 enz diff=+0.568; psma_high=1 enz diff=+0.531. Small interaction coef -0.21 (p=6e-3); only modestly attenuates effect.",
         "p_value": 0.006, "effect_estimate": -0.2068, "significant": True},
        {"hypothesis_ids": ["h9.4"], "result_summary": "Core: visceral_mets=0 enz diff=+0.549; visceral=1 enz diff=+0.535. Interaction coef -0.07 (p=0.44). Not modifying.",
         "p_value": 0.44, "effect_estimate": -0.0699, "significant": False},
        {"hypothesis_ids": ["h9.5"], "result_summary": "Core: ECOG 0,1,2 enz diffs +0.558, +0.541, +0.528. Interaction coef +0.046 (p=0.41). Not modifying.",
         "p_value": 0.41, "effect_estimate": 0.0455, "significant": False},
    ],
})

# --- Iteration 10: Continuous modifiers of enzalutamide effect ---
iterations.append({
    "index": 10,
    "proposed_hypotheses": [
        {"id": "h10.1", "text": "Higher psa_ng_ml is associated with smaller treatment_enzalutamide benefit on objective_response.", "kind": "novel"},
        {"id": "h10.2", "text": "Other continuous labs (albumin, LDH, hemoglobin, ALP, CRP, NLR, weight_loss, age, calcium) do not meaningfully modify the treatment_enzalutamide benefit.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h10.1"], "code": "Logit(response ~ enz + lab_z + enz*lab_z) on full sample",
         "result_summary": "enz × log(PSA) interaction: coef -0.81 (p=1.2e-72). Higher PSA → smaller enzalutamide benefit (note: PSA is correlated with mcrpc).",
         "p_value": 1.2e-72, "effect_estimate": -0.8054, "significant": True},
        {"hypothesis_ids": ["h10.2"], "result_summary": "Interaction coefs (p): albumin -0.04 (0.06); LDH -0.002 (0.91); Hb +0.013 (0.56); ALP +0.015 (0.48); CRP +0.038 (0.13); NLR -0.014 (0.50); weight_loss +0.027 (0.23); age -0.002 (0.92); calcium -0.003 (0.87). None are clinically meaningful.",
         "p_value": 0.5, "effect_estimate": 0.0, "significant": False},
    ],
})

# --- Iteration 11: Pembrolizumab fine subgroups ---
iterations.append({
    "index": 11,
    "proposed_hypotheses": [
        {"id": "h11.1", "text": "treatment_pembrolizumab benefit is concentrated in a narrower subgroup defined by msi_high=1 plus one or more clinical features (e.g., mcrpc=0, ecog_ps=0, ar_v7_positive=0, visceral_mets=0).", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h11.1"], "result_summary": "msi_high=1 & visceral_mets=0: diff=+0.009 (n_t=66); msi_high=1 & ecog=0: diff=-0.017 (n_t=29); msi_high=1 & mcrpc=0: diff=+0.100 (n_t=34, p>0.2); msi_high=1 & mcrpc=1: diff=-0.072 (n_t=45); msi_high=1 & ar_v7=0: diff=-0.005 (n_t=65). No subgroup achieves significance; samples small.",
         "p_value": 0.23, "effect_estimate": 0.10, "significant": False},
    ],
})

# --- Iteration 12: Olaparib fine subgroups ---
iterations.append({
    "index": 12,
    "proposed_hypotheses": [
        {"id": "h12.1", "text": "treatment_olaparib benefit may emerge in BRCA2+ patients restricted by mcrpc status or AR-V7 status (e.g., BRCA2+ AND mcrpc=0).", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h12.1"], "result_summary": "BRCA2+ & mcrpc=0: diff=-0.021 (n_t=233); BRCA2+ & mcrpc=1: diff=-0.041 (n_t=296); BRCA2+ & ar_v7=0: diff=-0.043 (n_t=423); BRCA2+ & ar_v7=1: diff=+0.011 (n_t=106). None positive or significant.",
         "p_value": 0.19, "effect_estimate": -0.043, "significant": False},
    ],
})

# --- Iteration 13: Lu177-PSMA fine subgroups ---
iterations.append({
    "index": 13,
    "proposed_hypotheses": [
        {"id": "h13.1", "text": "treatment_lu177_psma benefit may emerge in PSMA-high patients restricted by mcrpc status, visceral_mets status, or ECOG status.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h13.1"], "result_summary": "psma_high=1 & mcrpc=1 diff=+0.009 (n_t=2468); psma_high=1 & mcrpc=0 diff=-0.012 (n_t=2018); psma_high=1 & visceral=0 diff=+0.004 (n_t=3567); psma_high=1 & ecog=0 diff=-0.005 (n_t=1607). All near zero.",
         "p_value": 0.5, "effect_estimate": 0.0092, "significant": False},
    ],
})

# --- Iteration 14: Joint 4-marker subgroup ---
iterations.append({
    "index": 14,
    "proposed_hypotheses": [
        {"id": "h14.1", "text": "Within the subgroup ar_v7_positive=0 AND mcrpc=0 AND brca2_mutation=0 AND msi_high=0, treatment_enzalutamide produces a very large absolute increase in objective_response (~0.62), while none of the other five treatments produces a meaningful effect within the same subgroup.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h14.1"], "result_summary": "Joint subgroup n=15681. Enzalutamide rr=0.798 (n_t=6325) vs 0.172 (n_c=9356); diff=+0.626 (chi2 p<1e-300). Within same subgroup: abiraterone diff=-0.012, docetaxel +0.005, olaparib -0.001, lu177_psma -0.007, pembrolizumab -0.017 — all null.",
         "p_value": 0.0, "effect_estimate": 0.6257, "significant": True},
    ],
})

# --- Iteration 15: PSA tertiles within core ---
iterations.append({
    "index": 15,
    "proposed_hypotheses": [
        {"id": "h15.1", "text": "Within the joint enzalutamide-responsive subgroup, baseline psa_ng_ml does NOT further modify the treatment_enzalutamide effect (the earlier PSA × enz interaction was driven by PSA's correlation with mcrpc).", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h15.1"], "result_summary": "PSA tertile in core: low (PSA<4.8) diff=+0.635; mid (4.8-11.3) diff=+0.617; high (>11.3) diff=+0.625. Logit enz×log(PSA) within core: coef -0.047 (p=0.34). PSA does not modify within-subgroup effect.",
         "p_value": 0.34, "effect_estimate": -0.0468, "significant": False},
    ],
})

# --- Iteration 16: Exhaustive 2-feature subgroups for non-enz treatments ---
iterations.append({
    "index": 16,
    "proposed_hypotheses": [
        {"id": "h16.1", "text": "An exhaustive search over all 2-of-6 binary clinical/biomarker subgroups finds no subgroup in which abiraterone, docetaxel, olaparib, lu177_psma, or pembrolizumab produces a statistically significant change in objective_response.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h16.1"], "result_summary": "Top |effect| 2-feature subgroups: abiraterone msi+psma- diff=-0.049 p=0.18; docetaxel mcrpc-msi+ diff=-0.029 p=0.45; olaparib mcrpc-msi+ diff=-0.071 p=0.19; lu177 ar_v7+msi+ diff=+0.082 p=0.26; pembrolizumab mcrpc-msi+ diff=+0.100 p=0.23. None significant.",
         "p_value": 0.18, "effect_estimate": -0.049, "significant": False},
    ],
})

# --- Iteration 17: 3-feature subgroups for non-enz treatments ---
iterations.append({
    "index": 17,
    "proposed_hypotheses": [
        {"id": "h17.1", "text": "An exhaustive search over all 3-of-6 binary clinical/biomarker subgroups (with adequate sample sizes) finds no subgroup in which abiraterone, docetaxel, olaparib, lu177_psma, or pembrolizumab produces a statistically significant change in objective_response.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h17.1"], "result_summary": "Best 3-feature subgroups (effect, p): abiraterone mcrpc-msi+psma- -0.072 p=0.24; docetaxel ar_v7+brca2+psma- +0.062 p=0.19; olaparib mcrpc-msi+visc- -0.101 p=0.10; lu177 ar_v7+msi+visc- +0.148 p=0.07; pembrolizumab ar_v7+brca2+psma- -0.095 p=0.26. None significant.",
         "p_value": 0.07, "effect_estimate": 0.148, "significant": False},
    ],
})

# --- Iteration 18: Feature x feature interactions on baseline response ---
iterations.append({
    "index": 18,
    "proposed_hypotheses": [
        {"id": "h18.1", "text": "Pairs of negative biomarkers (mcrpc, ar_v7, brca2, msi_high, ecog_ps) act roughly additively (no large interaction) on baseline objective_response in untreated/non-enz patients.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h18.1"], "result_summary": "Among 10 pairwise interactions in non-enz subset (n=29924): only ar_v7×brca2 (+0.27, p=0.04) and brca2×ecog (-0.20, p=0.02) marginal; both small and not survive multiple testing. Effects of mcrpc, ar_v7, brca2, msi_high, ecog are essentially additive on response.",
         "p_value": 0.04, "effect_estimate": 0.267, "significant": False},
    ],
})

# --- Iteration 19: Visceral mets conditional ---
iterations.append({
    "index": 19,
    "proposed_hypotheses": [
        {"id": "h19.1", "text": "visceral_mets has no association with objective_response either overall or within mcrpc=0 or mcrpc=1 strata.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h19.1"], "result_summary": "Overall diff +0.001 (p=0.95); mcrpc=0 diff -0.001 (p=0.92); mcrpc=1 diff +0.000 (p=0.95). Visceral_mets is non-prognostic.",
         "p_value": 0.92, "effect_estimate": -0.001, "significant": False},
    ],
})

# --- Iteration 20: Treatment combinations ---
iterations.append({
    "index": 20,
    "proposed_hypotheses": [
        {"id": "h20.1", "text": "Patients receive multiple concurrent treatments, and number of treatments is positively associated with objective_response (likely confounded by enzalutamide receipt).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h20.1"], "result_summary": "n_treatments distribution: 0:10572, 1:19943, 2:14031, 3:4636, 4:764, 5:53, 6:1. Response by # tx: 0→0.156, 1→0.233, 2→0.284, 3→0.321, 4→0.300, 5→0.359. Positive association consistent with enzalutamide presence in patients with more therapies.",
         "p_value": 0.0, "effect_estimate": 0.04, "significant": True},
    ],
})

# --- Iteration 21: Adjusted enzalutamide model with interactions ---
iterations.append({
    "index": 21,
    "proposed_hypotheses": [
        {"id": "h21.1", "text": "After fitting a logistic model with enzalutamide × {mcrpc, ar_v7, brca2, msi} interactions plus all other covariates, each interaction is strongly negative (effect-suppressing) and treatment_enzalutamide retains a large positive baseline effect among the 4-negative reference subgroup.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h21.1"], "result_summary": "Model coefs: enzalutamide +2.63 (p<1e-300); enz×mcrpc -2.23 (p<1e-300); enz×ar_v7 -1.64 (p=1.3e-139); enz×brca2 -1.55 (p=6.0e-67); enz×msi -1.67 (p=1.8e-27). Confirms 4-marker subgroup model.",
         "p_value": 0.0, "effect_estimate": 2.628, "significant": True},
    ],
})

# --- Iteration 22: Decision-tree heterogeneity ---
iterations.append({
    "index": 22,
    "proposed_hypotheses": [
        {"id": "h22.1", "text": "A decision-tree summary of the data-driven enzalutamide treatment effect (T-learner with gradient boosting) selects mcrpc, ar_v7_positive, and brca2_mutation as the dominant splits and assigns mean ITE +0.59 to the leaf {mcrpc=0, ar_v7=0, brca2=0} and ITE near 0 to all other leaves.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h22.1"], "code": "T-learner GradientBoostingRegressor + DecisionTreeRegressor(max_depth=3) on ITE",
         "result_summary": "Tree splits: mcrpc->ar_v7->brca2; leaf mean ITE: {mcrpc=0,ar_v7=0,brca2=0}=0.59, {mcrpc=0,ar_v7=0,brca2=1}=0.05, {mcrpc=0,ar_v7=1,...}=0.04 and 0.01, {mcrpc=1,...}=-0.03 to 0.02. Mean ITE in joint 4-marker core subgroup =+0.609; in each isolating subgroup (mcrpc+, ar_v7+, brca2+, msi-high) ~+0.01.",
         "p_value": 0.0, "effect_estimate": 0.609, "significant": True},
    ],
})

# --- Iteration 23: Necessity check ---
iterations.append({
    "index": 23,
    "proposed_hypotheses": [
        {"id": "h23.1", "text": "Each of the four conditions ar_v7_positive=0, mcrpc=0, brca2_mutation=0, and msi_high=0 is individually necessary: dropping any one of them and admitting the corresponding 'positive' patients shows essentially no enzalutamide benefit in the added stratum.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h23.1"], "result_summary": "Adding-back single-positive strata to core (each holds the other 3 negative): only AR-V7+ added back: enz diff +0.022 (n_t=1573); only mCRPC=1: +0.007 (n_t=7677); only BRCA2+: +0.004 (n_t=710); only MSI-high: -0.025 (n_t=189). All near zero, contrasting with +0.626 in 4-negative core.",
         "p_value": 0.5, "effect_estimate": 0.022, "significant": False},
    ],
})

# --- Iteration 24: Combination effects ---
iterations.append({
    "index": 24,
    "proposed_hypotheses": [
        {"id": "h24.1", "text": "Within the enzalutamide-responsive 4-negative core subgroup, adding any of the other five treatments to enzalutamide does not further increase objective_response.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h24.1"], "result_summary": "In core enz-treated patients (n=6325, rr=0.798), response by additional agent: +abiraterone 0.790 vs 0.801 (-0.011); +docetaxel 0.801 vs 0.797 (+0.004); +olaparib 0.773 vs 0.801 (-0.028); +lu177 0.809 vs 0.796 (+0.014); +pembro 0.823 vs 0.797 (+0.026). All combinations roughly equivalent — none provides a meaningful add-on.",
         "p_value": 0.5, "effect_estimate": -0.001, "significant": False},
    ],
})

# --- Iteration 25: Final adjusted analyses of preferred biological subgroups + final hypotheses ---
iterations.append({
    "index": 25,
    "proposed_hypotheses": [
        {"id": "h25.1", "text": "treatment_olaparib has no benefit on objective_response even within BRCA2+ patients (adjusted logit coef = -0.27, p=0.05).", "kind": "refined"},
        {"id": "h25.2", "text": "treatment_pembrolizumab has no benefit on objective_response even within MSI-high patients (adjusted logit coef = -0.006, p=0.99).", "kind": "refined"},
        {"id": "h25.3", "text": "treatment_lu177_psma has no benefit on objective_response within PSMA-high or PSMA-high+mCRPC patients.", "kind": "refined"},
        {"id": "h25.4", "text": "treatment_abiraterone has no benefit on objective_response within mCRPC patients.", "kind": "refined"},
        {"id": "h25.5", "text": "treatment_docetaxel has no benefit on objective_response even within mCRPC + visceral_mets patients.", "kind": "refined"},
        {"id": "h25.6", "text": "FINAL SUBGROUP HYPOTHESIS: treatment_enzalutamide increases objective_response by ~+0.63 absolute (from ~0.17 to ~0.80) in the joint subgroup ar_v7_positive=0 AND mcrpc=0 AND brca2_mutation=0 AND msi_high=0; outside this subgroup the treatment has no measurable benefit (the effect is suppressed by any of: ar_v7_positive=1, mcrpc=1, brca2_mutation=1, or msi_high=1).", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h25.1"], "result_summary": "Olaparib in BRCA2+ (n=4996), adjusted for ECOG, mcrpc, ar_v7, msi, psma, visceral: coef=-0.269, p=0.055; rr 0.121 vs 0.153.", "p_value": 0.055, "effect_estimate": -0.269, "significant": False},
        {"hypothesis_ids": ["h25.2"], "result_summary": "Pembrolizumab in MSI-high (n=1528), adjusted: coef=-0.006, p=0.99; rr 0.177 vs 0.176.", "p_value": 0.985, "effect_estimate": -0.006, "significant": False},
        {"hypothesis_ids": ["h25.3"], "result_summary": "Lu177-PSMA in PSMA-high: coef=-0.007, p=0.87; in PSMA-high+mCRPC+: coef=+0.064, p=0.29.", "p_value": 0.29, "effect_estimate": 0.064, "significant": False},
        {"hypothesis_ids": ["h25.4"], "result_summary": "Abiraterone in mCRPC+ (n=27481), adjusted: coef=+0.0045, p=0.90; rr 0.154 vs 0.153.", "p_value": 0.90, "effect_estimate": 0.005, "significant": False},
        {"hypothesis_ids": ["h25.5"], "result_summary": "Docetaxel in mCRPC+ & visceral+ (n=5510): coef=+0.045, p=0.58; rr 0.159 vs 0.152.", "p_value": 0.58, "effect_estimate": 0.045, "significant": False},
        {"hypothesis_ids": ["h25.6"], "code": "Subgroup chi2; logit with single subgroup interaction term",
         "result_summary": "Joint subgroup (ar_v7=0 & mcrpc=0 & brca2=0 & msi=0): n=15681; enz rr=0.798 (n_t=6325) vs control rr=0.172 (n_c=9356); absolute diff=+0.626 (chi2 p<1e-300). Single-indicator model: enz coef +0.058 (p=0.06), enz×core coef +2.93 (p<1e-300). Outside the subgroup the enzalutamide effect is essentially null in every stratum tested.",
         "p_value": 0.0, "effect_estimate": 0.6257, "significant": True},
    ],
})

transcript = {
    "dataset_id": "ds001_prostate",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@manual-prostate-named",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)
print("Wrote transcript.json with", len(iterations), "iterations")
