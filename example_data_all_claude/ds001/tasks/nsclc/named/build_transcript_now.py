"""Build transcript.json and analysis_summary.txt from results.json."""
import json

with open('results.json', 'r') as f:
    R = json.load(f)


def A(hyp_ids, key):
    """Build an analysis record for the given results key, addressing the given hypotheses."""
    r = R[key]
    return {
        "hypothesis_ids": hyp_ids if isinstance(hyp_ids, list) else [hyp_ids],
        "code": key,
        "result_summary": r["result_summary"],
        "p_value": r["p_value"],
        "effect_estimate": r["effect_estimate"],
        "significant": r["significant"],
    }


def H(hid, text, kind="novel"):
    return {"id": hid, "text": text, "kind": kind}


iterations = []

# -------- Iteration 1: clinical baseline features and PFS --------
iterations.append({
    "index": 1,
    "proposed_hypotheses": [
        H("h1", "Higher ECOG performance status (ecog_ps=2 vs ecog_ps=0) is associated with shorter pfs_months."),
        H("h2", "Patients with stage_iv=1 have shorter pfs_months than patients with stage_iv=0."),
        H("h3", "Patients with has_brain_mets=1 have shorter pfs_months than patients with has_brain_mets=0."),
        H("h4", "Female patients (sex_female=1) have different pfs_months than male patients."),
    ],
    "analyses": [
        A("h1", "h1_ecog2_vs_ecog0"),
        A("h2", "h2_stageiv"),
        A("h3", "h3_brainmets"),
        A("h4", "h4_female"),
    ],
})

# -------- Iteration 2: treatment main effects --------
iterations.append({
    "index": 2,
    "proposed_hypotheses": [
        H("h5", "Patients receiving treatment_pembrolizumab have longer pfs_months than those who do not, on average."),
        H("h6", "Patients receiving treatment_sotorasib have longer pfs_months than those who do not, on average."),
        H("h7", "Patients receiving treatment_olaparib have longer pfs_months than those who do not, on average."),
        H("h8", "Patients receiving treatment_osimertinib have longer pfs_months than those who do not, on average."),
    ],
    "analyses": [
        A("h5", "h_main_treatment_pembrolizumab"),
        A("h6", "h_main_treatment_sotorasib"),
        A("h7", "h_main_treatment_olaparib"),
        A("h8", "h_main_treatment_osimertinib"),
    ],
})

# -------- Iteration 3: biomarker main effects on PFS --------
iterations.append({
    "index": 3,
    "proposed_hypotheses": [
        H("h9", "egfr_mutation+ patients have different pfs_months than egfr_mutation- patients (overall sample)."),
        H("h10", "kras_g12c+ patients have different pfs_months than kras_g12c- patients (overall sample)."),
        H("h11", "alk_fusion+ patients have shorter pfs_months than alk_fusion- patients (overall sample)."),
        H("h12", "stk11_mutation+ patients have different pfs_months than stk11_mutation- patients (overall sample)."),
        H("h13", "brca2_mutation+ patients have different pfs_months than brca2_mutation- patients (overall sample)."),
        H("h14", "tmb_high+ patients have different pfs_months than tmb_high- patients (overall sample)."),
    ],
    "analyses": [
        A("h9", "h_main_egfr_mutation"),
        A("h10", "h_main_kras_g12c"),
        A("h11", "h_main_alk_fusion"),
        A("h12", "h_main_stk11_mutation"),
        A("h13", "h_main_brca2_mutation"),
        A("h14", "h_main_tmb_high"),
    ],
})

# -------- Iteration 4: continuous lab features and PFS --------
iterations.append({
    "index": 4,
    "proposed_hypotheses": [
        H("h15", "Higher serum albumin_g_dl is associated with longer pfs_months (positive Pearson correlation)."),
        H("h16", "Higher ldh_u_l is associated with shorter pfs_months (negative Pearson correlation)."),
        H("h17", "Higher weight_loss_pct_6mo is associated with shorter pfs_months (negative Pearson correlation)."),
        H("h18", "Higher crp_mg_l is associated with shorter pfs_months (negative Pearson correlation)."),
        H("h19", "Higher nlr (neutrophil-to-lymphocyte ratio) is associated with shorter pfs_months (negative Pearson correlation)."),
        H("h20", "Higher pdl1_tps is associated with longer pfs_months (positive Pearson correlation)."),
        H("h21", "Older age_years is associated with shorter pfs_months (negative Pearson correlation)."),
    ],
    "analyses": [
        A("h15", "h_corr_albumin_g_dl"),
        A("h16", "h_corr_ldh_u_l"),
        A("h17", "h_corr_weight_loss_pct_6mo"),
        A("h18", "h_corr_crp_mg_l"),
        A("h19", "h_corr_nlr"),
        A("h20", "h_corr_pdl1_tps"),
        A("h21", "h_corr_age_years"),
        A("h21", "h_age_log_pfs"),
    ],
})

# -------- Iteration 5: smoking, histology, and remaining lab correlates --------
iterations.append({
    "index": 5,
    "proposed_hypotheses": [
        H("h22", "Never-smokers (smoking_status='never') have longer pfs_months than current smokers (smoking_status='current')."),
        H("h23", "Patients with adenocarcinoma histology have longer pfs_months than patients with squamous histology."),
        H("h24", "Hemoglobin_g_dl is correlated with pfs_months."),
        H("h25", "Liver enzymes (ast_u_l, alt_u_l, total_bilirubin_mg_dl, alkaline_phosphatase_u_l) and renal markers (creatinine_mg_dl, bun_mg_dl) are correlated with pfs_months."),
    ],
    "analyses": [
        A("h22", "h_smoking_never_vs_current"),
        A("h23", "h_hist_adeno_vs_sq"),
        A("h24", "h_corr_hemoglobin_g_dl"),
        A("h25", "h_corr_alkaline_phosphatase_u_l"),
        A("h25", "h_corr_ast_u_l"),
        A("h25", "h_corr_alt_u_l"),
        A("h25", "h_corr_total_bilirubin_mg_dl"),
        A("h25", "h_corr_creatinine_mg_dl"),
        A("h25", "h_corr_bun_mg_dl"),
        A("h25", "h_corr_sodium_meq_l"),
        A("h25", "h_corr_potassium_meq_l"),
        A("h25", "h_corr_calcium_mg_dl"),
    ],
})

# -------- Iteration 6: targeted therapy x matched biomarker (canonical) --------
iterations.append({
    "index": 6,
    "proposed_hypotheses": [
        H("h26", "treatment_osimertinib improves pfs_months specifically in egfr_mutation+ patients (positive osimertinib*egfr_mutation interaction on log PFS), with little or no effect in egfr_mutation- patients."),
        H("h27", "treatment_sotorasib improves pfs_months specifically in kras_g12c+ patients (positive sotorasib*kras_g12c interaction on log PFS), with little or no effect in kras_g12c- patients."),
        H("h28", "treatment_olaparib improves pfs_months specifically in brca2_mutation+ patients (positive olaparib*brca2_mutation interaction on log PFS), with little or no effect in brca2_mutation- patients."),
    ],
    "analyses": [
        A("h26", "h_osi_egfr_in_egfr_mutationpos"),
        A("h26", "h_osi_egfr_in_egfr_mutationneg"),
        A("h26", "h_osi_egfr_interaction_egfr_mutation"),
        A("h27", "h_sot_kras_in_kras_g12cpos"),
        A("h27", "h_sot_kras_in_kras_g12cneg"),
        A("h27", "h_sot_kras_interaction_kras_g12c"),
        A("h28", "h_ola_brca2_in_brca2_mutationpos"),
        A("h28", "h_ola_brca2_in_brca2_mutationneg"),
        A("h28", "h_ola_brca2_interaction_brca2_mutation"),
    ],
})

# -------- Iteration 7: pembrolizumab biomarker heterogeneity --------
iterations.append({
    "index": 7,
    "proposed_hypotheses": [
        H("h29", "treatment_pembrolizumab improves pfs_months more in pdl1_high (TPS>=0.5) patients than in pdl1_low patients (positive interaction)."),
        H("h30", "treatment_pembrolizumab improves pfs_months more in tmb_high+ patients than in tmb_high- patients (positive interaction)."),
        H("h31", "treatment_pembrolizumab is less effective in stk11_mutation+ patients than in stk11_mutation- patients (negative interaction)."),
        H("h32", "treatment_pembrolizumab is less effective in never-smokers than in ever-smokers (negative interaction with never-smoker status)."),
    ],
    "analyses": [
        A("h29", "h_pem_in_pdl1_highpos"),
        A("h29", "h_pem_in_pdl1_highneg"),
        A("h29", "h_pem_interaction_pdl1_high"),
        A("h30", "h_pem_in_tmb_highpos"),
        A("h30", "h_pem_in_tmb_highneg"),
        A("h30", "h_pem_interaction_tmb_high"),
        A("h31", "h_pem_in_stk11_mutationpos"),
        A("h31", "h_pem_in_stk11_mutationneg"),
        A("h31", "h_pem_interaction_stk11_mutation"),
        A("h32", "h_pem_in_eversmoker"),
        A("h32", "h_pem_in_neversmoker"),
        A("h32", "h_pem_interaction_eversmoker"),
    ],
})

# -------- Iteration 8: full multivariable adjusted model --------
iterations.append({
    "index": 8,
    "proposed_hypotheses": [
        H("h33", "After adjustment for all measured features in a multivariable OLS model on log(pfs_months+0.1), age_years, ecog_ps, stage_iv, has_brain_mets, albumin_g_dl, weight_loss_pct_6mo, ldh_u_l, smoking status, histology, kras_g12c, alk_fusion remain independently associated with PFS, while pdl1_tps, tmb_high, egfr_mutation, stk11_mutation, and most chemistries are not."),
        H("h34", "After adjustment, only treatment_sotorasib retains a positive PFS effect on log scale; treatment_pembrolizumab, treatment_olaparib, and treatment_osimertinib do not."),
    ],
    "analyses": [
        A("h33", "h_adj_ecog_ps"),
        A("h33", "h_adj_stage_iv"),
        A("h33", "h_adj_has_brain_mets"),
        A("h33", "h_adj_pdl1_tps"),
        A("h33", "h_adj_tmb_high"),
        A("h33", "h_adj_albumin_g_dl"),
        A("h33", "h_adj_ldh_u_l"),
        A("h33", "h_adj_nlr"),
        A("h33", "h_adj_crp_mg_l"),
        A("h33", "h_adj_weight_loss_pct_6mo"),
        A("h33", "h_adj_egfr_mutation"),
        A("h33", "h_adj_kras_g12c"),
        A("h33", "h_adj_alk_fusion"),
        A("h33", "h_adj_stk11_mutation"),
        A("h33", "h_adj_brca2_mutation"),
        A("h34", "h_adj_treatment_pembrolizumab"),
        A("h34", "h_adj_treatment_sotorasib"),
        A("h34", "h_adj_treatment_olaparib"),
        A("h34", "h_adj_treatment_osimertinib"),
    ],
})

# -------- Iteration 9: joint pembrolizumab subgroups --------
iterations.append({
    "index": 9,
    "proposed_hypotheses": [
        H("h35", "treatment_pembrolizumab improves pfs_months in the joint subgroup of pdl1_high=1 AND stk11_mutation=0."),
        H("h36", "treatment_pembrolizumab improves pfs_months in the joint subgroup of pdl1_high=1 AND tmb_high=1."),
        H("h37", "treatment_pembrolizumab improves pfs_months in the joint subgroup pdl1_high=1 AND tmb_high=1 AND stk11_mutation=0 (the most clinically favorable immunotherapy candidates)."),
        H("h38", "treatment_pembrolizumab effect is suppressed in pdl1_high+ AND stk11_mutation+ (STK11 confers resistance)."),
    ],
    "analyses": [
        A("h35", "h_pem_pdl1high_stk11neg"),
        A("h36", "h_pem_pdl1high_tmbhigh"),
        A("h37", "h_pem_pdl1high_tmbhigh_stk11neg"),
        A("h38", "h_pem_pdl1high_stk11pos"),
    ],
})

# -------- Iteration 10: comprehensive interaction screen for each treatment --------
iterations.append({
    "index": 10,
    "proposed_hypotheses": [
        H("h39", "Among all measured features, the strongest treatment-effect modifier of treatment_sotorasib is kras_g12c (consistent with its molecular target), and the second-strongest modifiers include sex_female and age_years."),
        H("h40", "No measured feature shows a clinically meaningful interaction with treatment_pembrolizumab; the largest interaction p-values per modifier exceed conventional thresholds."),
        H("h41", "No measured feature shows a clinically meaningful interaction with treatment_olaparib; including the brca2_mutation interaction itself."),
        H("h42", "No measured feature shows a clinically meaningful interaction with treatment_osimertinib; including the egfr_mutation interaction itself."),
    ],
    "analyses": [
        A("h39", "h_screen_treatment_sotorasib_top1_kras_g12c"),
        A("h39", "h_screen_treatment_sotorasib_top2_sex_female"),
        A("h39", "h_screen_treatment_sotorasib_top3_age_years"),
        A("h39", "h_screen_treatment_sotorasib_top4_alk_fusion"),
        A("h39", "h_screen_treatment_sotorasib_top5_albumin_g_dl"),
        A("h40", "h_screen_treatment_pembrolizumab_top1_weight_loss_pct_6mo"),
        A("h40", "h_screen_treatment_pembrolizumab_top2_bun_mg_dl"),
        A("h40", "h_screen_treatment_pembrolizumab_top3_ldh_u_l"),
        A("h40", "h_screen_treatment_pembrolizumab_top4_stage_iv"),
        A("h40", "h_screen_treatment_pembrolizumab_top5_pdl1_tps"),
        A("h41", "h_screen_treatment_olaparib_top1_bun_mg_dl"),
        A("h41", "h_screen_treatment_olaparib_top2_age_years"),
        A("h41", "h_screen_treatment_olaparib_top3_ecog_ps"),
        A("h42", "h_screen_treatment_osimertinib_top1_alk_fusion"),
        A("h42", "h_screen_treatment_osimertinib_top2_albumin_g_dl"),
        A("h42", "h_screen_treatment_osimertinib_top3_pdl1_tps"),
    ],
})

# -------- Iteration 11: tree-based subgroup discovery --------
iterations.append({
    "index": 11,
    "proposed_hypotheses": [
        H("h43", "A CART regression of treatment_sotorasib's estimated treatment effect (counterfactual: random forest trained on controls) on baseline features identifies a subgroup with very large positive sotorasib effect (>3 mo improvement) and a separate subgroup with no benefit."),
        H("h44", "CART subgroup analyses for treatment_pembrolizumab, treatment_olaparib, and treatment_osimertinib do NOT reveal subgroups with very large positive treatment effects; the across-leaf range of estimated effects is small."),
    ],
    "analyses": [
        A("h43", "h_tree_top_treatment_sotorasib"),
        A("h43", "h_tree_bot_treatment_sotorasib"),
        A("h44", "h_tree_top_treatment_pembrolizumab"),
        A("h44", "h_tree_bot_treatment_pembrolizumab"),
        A("h44", "h_tree_top_treatment_olaparib"),
        A("h44", "h_tree_bot_treatment_olaparib"),
        A("h44", "h_tree_top_treatment_osimertinib"),
        A("h44", "h_tree_bot_treatment_osimertinib"),
    ],
})

# -------- Iteration 12: refined sotorasib subgroup search within KRAS+ --------
iterations.append({
    "index": 12,
    "proposed_hypotheses": [
        H("h45", "Within KRAS+ patients, sex_female modifies treatment_sotorasib effect: PFS gain from sotorasib is much larger in male KRAS+ patients than in female KRAS+ patients (negative sotorasib*sex_female interaction within KRAS+).", "novel"),
        H("h46", "Within KRAS+ patients, treatment_sotorasib effect on log PFS is also modified by age_years and ecog_ps and has_brain_mets and albumin_g_dl, but the effect size of these modifiers is small relative to sex.", "novel"),
        H("h47", "stk11_mutation does NOT meaningfully modify treatment_sotorasib effect within KRAS+ patients (interaction near zero).", "novel"),
    ],
    "analyses": [
        A("h45", "h_sot_kras_int_sex_female"),
        A("h45", "h_sot_kras_sex0"),
        A("h45", "h_sot_kras_sex1"),
        A("h46", "h_sot_kras_int_age_years"),
        A("h46", "h_sot_kras_int_ecog_ps"),
        A("h46", "h_sot_kras_int_has_brain_mets"),
        A("h46", "h_sot_kras_int_albumin_g_dl"),
        A("h47", "h_sot_kras_int_stk11_mutation"),
        A("h47", "h_sot_kras_stk110"),
        A("h47", "h_sot_kras_stk111"),
    ],
})

# -------- Iteration 13: confirmatory triple interaction and per-subgroup sotorasib effect --------
iterations.append({
    "index": 13,
    "proposed_hypotheses": [
        H("h48", "The three-way interaction treatment_sotorasib * kras_g12c * sex_female on log PFS in the full cohort is large and negative, confirming that sotorasib's benefit is concentrated in KRAS+ males.", "refined"),
        H("h49", "Within KRAS+ males, treatment_sotorasib gives a substantial PFS benefit (~4.5-4.8 months) that is consistent across ECOG, brain mets, STK11, and age strata.", "refined"),
        H("h50", "Within KRAS+ females, treatment_sotorasib confers no PFS benefit across ECOG strata.", "refined"),
    ],
    "analyses": [
        A("h48", "h_sot_kras_sex_3way"),
        A("h49", "h_sot_kras_male_ecog0"),
        A("h49", "h_sot_kras_male_ecog1"),
        A("h49", "h_sot_kras_male_ecog2"),
        A("h49", "h_sot_kras_male_brain0"),
        A("h49", "h_sot_kras_male_brain1"),
        A("h49", "h_sot_kras_male_stk110"),
        A("h49", "h_sot_kras_male_stk111"),
        A("h49", "h_sot_kras_male_age_tertile0"),
        A("h49", "h_sot_kras_male_age_tertile1"),
        A("h49", "h_sot_kras_male_age_tertile2"),
        A("h50", "h_sot_kras_female_ecog0"),
        A("h50", "h_sot_kras_female_ecog1"),
        A("h50", "h_sot_kras_female_ecog2"),
    ],
})

# -------- Iteration 14: best-supported subgroup hypotheses (FINAL) --------
iterations.append({
    "index": 14,
    "proposed_hypotheses": [
        H("h51",
          "FINAL subgroup hypothesis for treatment_sotorasib: treatment_sotorasib improves pfs_months by approximately 4.6 months in patients with kras_g12c=1 AND sex_female=0 (KRAS G12C-positive males); the benefit is essentially absent in kras_g12c=1 AND sex_female=1 (KRAS G12C-positive females) and is essentially absent in kras_g12c=0 patients regardless of sex.",
          "refined"),
        H("h52",
          "FINAL subgroup hypothesis for treatment_pembrolizumab: no biomarker subgroup (including pdl1_high, tmb_high, stk11_mutation status, smoking status, or any combination of these) yields a clinically or statistically meaningful PFS benefit from treatment_pembrolizumab in this dataset; the adjusted treatment effect on log PFS is approximately 0 (p>0.9).",
          "refined"),
        H("h53",
          "FINAL subgroup hypothesis for treatment_olaparib: treatment_olaparib does not improve pfs_months even in the brca2_mutation+ subgroup; the olaparib*brca2_mutation interaction is near zero (p~0.97).",
          "refined"),
        H("h54",
          "FINAL subgroup hypothesis for treatment_osimertinib: treatment_osimertinib does not improve pfs_months even in the egfr_mutation+ subgroup; the osimertinib*egfr_mutation interaction is near zero (p~0.43).",
          "refined"),
    ],
    "analyses": [
        A("h51", "h_final_sotorasib_subgroup"),
        A("h52", "h_final_pembrolizumab_adj"),
        A("h52", "h_pem_best_subgroup"),
        A("h52", "h_pem_worst_subgroup"),
        A("h52", "h_pem_pdl1_smoker_stk11neg_goodecog"),
        A("h53", "h_final_olaparib_adj"),
        A("h53", "h_ola_best_subgroup"),
        A("h54", "h_final_osimertinib_adj"),
        A("h54", "h_osi_best_subgroup"),
        A("h54", "h_osi_egfr_smoke_never"),
        A("h54", "h_osi_egfr_smoke_former"),
        A("h54", "h_osi_egfr_smoke_current"),
    ],
})

transcript = {
    "dataset_id": "ds001_nsclc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-manual@kk-2026-05-03",
    "max_iterations": 25,
    "iterations": iterations,
}

with open('transcript.json', 'w') as f:
    json.dump(transcript, f, indent=2)
print(f"Wrote transcript.json with {len(iterations)} iterations.")
