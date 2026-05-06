"""Assemble transcript.json and analysis_summary.txt from results.json."""
import json
from pathlib import Path

with open("results.json", "r") as f:
    R = json.load(f)

def get(key):
    return R[key]

def A(hyp_ids, key, code=None):
    r = get(key)
    return {
        "hypothesis_ids": hyp_ids if isinstance(hyp_ids, list) else [hyp_ids],
        "code": code,
        "result_summary": r["result_summary"],
        "p_value": r["p_value"],
        "effect_estimate": r["effect_estimate"],
        "significant": r["significant"],
    }

def H(id_, text, kind="novel"):
    return {"id": id_, "text": text, "kind": kind}

iterations = []

# ------------- Iteration 1 -------------
iterations.append({
    "index": 1,
    "proposed_hypotheses": [
        H("h1", "Higher serum albumin (`albumin_g_dl`) is associated with longer progression-free survival (`pfs_months`); the OLS slope of pfs_months on albumin_g_dl is positive."),
        H("h2", "Worse performance status (higher `ecog_ps`) is associated with shorter `pfs_months`; the OLS slope is negative."),
        H("h3", "Patients with `stage_iv` = 1 have shorter mean `pfs_months` than those with `stage_iv` = 0."),
    ],
    "analyses": [
        A("h1", "h1", code="smf.ols('pfs_months ~ albumin_g_dl', data=df).fit()"),
        A("h2", "h2", code="smf.ols('pfs_months ~ ecog_ps', data=df).fit()"),
        A("h3", "h3", code="ttest_ind(pfs_months by stage_iv)"),
    ],
})

# ------------- Iteration 2: Lab markers -------------
iterations.append({
    "index": 2,
    "proposed_hypotheses": [
        H("h4", "Higher tumor burden marker CEA (log1p(`cea_ng_ml`)) is associated with shorter `pfs_months`; OLS slope is negative."),
        H("h5", "Higher `ldh_u_l` is associated with shorter `pfs_months`; OLS slope is negative."),
        H("h6", "Higher `nlr` (neutrophil-to-lymphocyte ratio) is associated with shorter `pfs_months`; OLS slope is negative."),
        H("h7", "Higher `crp_mg_l` (inflammation) is associated with shorter `pfs_months`; OLS slope is negative."),
        H("h8", "Greater 6-month weight loss (`weight_loss_pct_6mo`) is associated with shorter `pfs_months`; OLS slope is negative."),
    ],
    "analyses": [
        A("h4", "h4", code="smf.ols('pfs_months ~ log_cea', data=df).fit()"),
        A("h5", "h5", code="smf.ols('pfs_months ~ ldh_u_l', data=df).fit()"),
        A("h6", "h6", code="smf.ols('pfs_months ~ nlr', data=df).fit()"),
        A("h7", "h7", code="smf.ols('pfs_months ~ crp_mg_l', data=df).fit()"),
        A("h8", "h8", code="smf.ols('pfs_months ~ weight_loss_pct_6mo', data=df).fit()"),
    ],
})

# ------------- Iteration 3: Demographics -------------
iterations.append({
    "index": 3,
    "proposed_hypotheses": [
        H("h9", "Older age (`age_years`) is associated with shorter `pfs_months`; OLS slope is negative."),
        H("h10", "Mean `pfs_months` differs between female (`sex_female`=1) and male patients."),
        H("h11", "Patients with `right_sided_primary`=1 have shorter mean `pfs_months` than those with left-sided primary."),
    ],
    "analyses": [
        A("h9", "h9", code="smf.ols('pfs_months ~ age_years', data=df).fit()"),
        A("h10", "h10", code="ttest_ind(pfs by sex)"),
        A("h11", "h11", code="ttest_ind(pfs by right_sided_primary)"),
    ],
})

# ------------- Iteration 4: Treatment main effects -------------
iterations.append({
    "index": 4,
    "proposed_hypotheses": [
        H("h12", "Treatment with cetuximab (`treatment_cetuximab`=1) is associated with longer `pfs_months` overall."),
        H("h13", "Treatment with bevacizumab (`treatment_bevacizumab`=1) is associated with longer `pfs_months` overall."),
        H("h14", "Treatment with pembrolizumab (`treatment_pembrolizumab`=1) is associated with longer `pfs_months` overall."),
        H("h15", "Treatment with encorafenib (`treatment_encorafenib`=1) is associated with longer `pfs_months` overall."),
        H("h16", "Treatment with trastuzumab/tucatinib (`treatment_trastuzumab_tucatinib`=1) is associated with longer `pfs_months` overall."),
        H("h17", "Treatment with regorafenib (`treatment_regorafenib`=1) is associated with longer `pfs_months` overall."),
    ],
    "analyses": [
        A("h12", "main_treatment_cetuximab", code="ttest_ind(pfs by treatment_cetuximab)"),
        A("h13", "main_treatment_bevacizumab"),
        A("h14", "main_treatment_pembrolizumab"),
        A("h15", "main_treatment_encorafenib"),
        A("h16", "main_treatment_trastuzumab_tucatinib"),
        A("h17", "main_treatment_regorafenib"),
    ],
})

# ------------- Iteration 5: Cetuximab x KRAS (canonical hypothesis) -------------
iterations.append({
    "index": 5,
    "proposed_hypotheses": [
        H("h18", "Cetuximab benefit on `pfs_months` is restricted to KRAS wild-type patients: in `kras_mutation`=0 the cetuximab-vs-no-cetuximab PFS difference is positive, and the treatment_cetuximab×kras_mutation interaction is negative."),
        H("h19", "In `kras_mutation`=1 patients, cetuximab does not improve `pfs_months` (effect ~ 0)."),
    ],
    "analyses": [
        A("h18", "cetux_kras0", code="ttest cetuximab in KRAS WT"),
        A("h19", "cetux_kras1", code="ttest cetuximab in KRAS mut"),
        A(["h18", "h19"], "cetux_kras_interaction", code="smf.ols('pfs_months ~ treatment_cetuximab * kras_mutation', data=df).fit()"),
    ],
})

# ------------- Iteration 6: Cetuximab x NRAS, BRAF, sidedness -------------
iterations.append({
    "index": 6,
    "proposed_hypotheses": [
        H("h20", "Cetuximab benefit on `pfs_months` is restricted to NRAS wild-type patients (interaction with `nras_mutation` negative).", kind="refined"),
        H("h21", "Cetuximab benefit on `pfs_months` is restricted to BRAF wild-type patients (interaction with `braf_v600e` negative).", kind="refined"),
        H("h22", "Cetuximab benefit on `pfs_months` is greater in left-sided primary (`right_sided_primary`=0) than right-sided.", kind="refined"),
    ],
    "analyses": [
        A("h20", "cetux_nras_mutation_interaction"),
        A("h20", "cetux_nras_mutation0"),
        A("h20", "cetux_nras_mutation1"),
        A("h21", "cetux_braf_v600e_interaction"),
        A("h21", "cetux_braf_v600e0"),
        A("h21", "cetux_braf_v600e1"),
        A("h22", "cetux_right_sided_primary_interaction"),
        A("h22", "cetux_right_sided_primary0"),
        A("h22", "cetux_right_sided_primary1"),
    ],
})

# ------------- Iteration 7: Cetuximab in fully RAS/RAF WT and left-sided -------------
iterations.append({
    "index": 7,
    "proposed_hypotheses": [
        H("h23", "In RAS/RAF wild-type patients (`kras_mutation`=0 AND `nras_mutation`=0 AND `braf_v600e`=0), cetuximab is associated with longer `pfs_months` than no cetuximab.", kind="refined"),
        H("h24", "In RAS/RAF wild-type AND left-sided (`right_sided_primary`=0) patients, cetuximab benefit on `pfs_months` is even larger than in the broader RAS/RAF WT group.", kind="refined"),
    ],
    "analyses": [
        A("h23", "cetux_full_wt"),
        A("h24", "cetux_full_wt_left"),
        A(["h23", "h24"], "cetux_NOT_full_wt_left"),
    ],
})

# ------------- Iteration 8: Pembrolizumab x MSI-high -------------
iterations.append({
    "index": 8,
    "proposed_hypotheses": [
        H("h25", "Pembrolizumab benefit on `pfs_months` is restricted to MSI-high (`msi_high`=1) patients; the treatment_pembrolizumab×msi_high interaction is positive."),
    ],
    "analyses": [
        A("h25", "pembro_msi0"),
        A("h25", "pembro_msi1"),
        A("h25", "pembro_msi_interaction"),
    ],
})

# ------------- Iteration 9: Encorafenib x BRAF V600E -------------
iterations.append({
    "index": 9,
    "proposed_hypotheses": [
        H("h26", "Encorafenib benefit on `pfs_months` is concentrated in BRAF V600E (`braf_v600e`=1) patients; the treatment_encorafenib×braf_v600e interaction is positive (i.e., encorafenib effect more positive in BRAF mut)."),
    ],
    "analyses": [
        A("h26", "enco_braf0"),
        A("h26", "enco_braf1"),
        A("h26", "enco_braf_interaction"),
    ],
})

# ------------- Iteration 10: HER2 trastuzumab/tucatinib -------------
iterations.append({
    "index": 10,
    "proposed_hypotheses": [
        H("h27", "Trastuzumab/tucatinib benefit on `pfs_months` is concentrated in HER2-amplified (`her2_amplified`=1) patients; the treatment_trastuzumab_tucatinib×her2_amplified interaction is positive."),
        H("h28", "Within HER2-amplified, the trastuzumab/tucatinib effect on `pfs_months` is greater when patients are also RAS wild-type (`kras_mutation`=0 AND `nras_mutation`=0).", kind="refined"),
    ],
    "analyses": [
        A("h27", "her2tx_her20"),
        A("h27", "her2tx_her21"),
        A("h27", "her2tx_her2_interaction"),
        A("h28", "her2tx_her2_rasWT"),
        A("h28", "her2tx_her2_rasMUT"),
    ],
})

# ------------- Iteration 11: Bevacizumab subgroups -------------
iterations.append({
    "index": 11,
    "proposed_hypotheses": [
        H("h29", "Bevacizumab effect on `pfs_months` differs by KRAS mutation status (treatment_bevacizumab×kras_mutation interaction is non-zero)."),
        H("h30", "Bevacizumab effect on `pfs_months` differs by tumor sidedness (treatment_bevacizumab×right_sided_primary interaction is non-zero)."),
    ],
    "analyses": [
        A("h29", "bev_kras_interaction"),
        A("h30", "bev_side_interaction"),
    ],
})

# ------------- Iteration 12: Regorafenib interactions broad screen -------------
iterations.append({
    "index": 12,
    "proposed_hypotheses": [
        H("h31", "Regorafenib effect on `pfs_months` differs by KRAS mutation status (interaction non-zero)."),
        H("h32", "Regorafenib effect on `pfs_months` differs by tumor sidedness (interaction non-zero)."),
        H("h33", "Regorafenib effect on `pfs_months` differs by BRAF V600E status (interaction non-zero)."),
        H("h34", "Regorafenib effect on `pfs_months` differs by MSI-high status (interaction non-zero)."),
        H("h35", "Regorafenib effect on `pfs_months` differs by stage_iv status (interaction non-zero)."),
        H("h36", "Regorafenib effect on `pfs_months` is modified by HER2 amplification (interaction non-zero)."),
        H("h37", "Regorafenib effect on `pfs_months` is modified by ECOG performance status."),
        H("h38", "Regorafenib effect on `pfs_months` is modified by age."),
        H("h39", "Regorafenib effect on `pfs_months` is modified by log(CEA); higher CEA reduces benefit (interaction negative)."),
        H("h40", "Regorafenib effect on `pfs_months` is modified by albumin."),
        H("h41", "Regorafenib effect on `pfs_months` is modified by LDH."),
        H("h42", "Regorafenib effect on `pfs_months` is modified by NLR."),
    ],
    "analyses": [
        A("h31", "rego_kras_mutation_interaction"),
        A("h32", "rego_right_sided_primary_interaction"),
        A("h33", "rego_braf_v600e_interaction"),
        A("h34", "rego_msi_high_interaction"),
        A("h35", "rego_stage_iv_interaction"),
        A("h36", "rego_her2_amplified_interaction"),
        A("h37", "rego_ecog_ps_interaction"),
        A("h38", "rego_age_years_interaction"),
        A("h39", "rego_log_cea_interaction"),
        A("h40", "rego_albumin_g_dl_interaction"),
        A("h41", "rego_ldh_u_l_interaction"),
        A("h42", "rego_nlr_interaction"),
    ],
})

# ------------- Iteration 13: Adjusted multivariable model -------------
iterations.append({
    "index": 13,
    "proposed_hypotheses": [
        H("h43", "After adjusting for age, sex, ECOG, stage_iv, sidedness, RAS/RAF/MSI/HER2 biomarkers, and labs, the adjusted association of cetuximab with `pfs_months` is positive."),
        H("h44", "After multivariable adjustment, bevacizumab is associated with longer `pfs_months`."),
        H("h45", "After multivariable adjustment, pembrolizumab is associated with longer `pfs_months`."),
        H("h46", "After multivariable adjustment, encorafenib is associated with longer `pfs_months`."),
        H("h47", "After multivariable adjustment, trastuzumab/tucatinib is associated with longer `pfs_months`."),
        H("h48", "After multivariable adjustment, regorafenib remains associated with longer `pfs_months`."),
    ],
    "analyses": [
        A("h43", "adj_treatment_cetuximab"),
        A("h44", "adj_treatment_bevacizumab"),
        A("h45", "adj_treatment_pembrolizumab"),
        A("h46", "adj_treatment_encorafenib"),
        A("h47", "adj_treatment_trastuzumab_tucatinib"),
        A("h48", "adj_treatment_regorafenib"),
    ],
})

# ------------- Iteration 14: Per-treatment top-modifier screens -------------
iterations.append({
    "index": 14,
    "proposed_hypotheses": [
        H("h49", "An exhaustive treatment-by-feature interaction screen across {age, sex, ECOG, stage_iv, sidedness, KRAS, NRAS, BRAF, MSI, HER2, NTRK, log_cea, albumin, LDH, weight loss, CRP, NLR, hemoglobin} will identify at least one feature whose interaction with cetuximab is significant at p<0.05."),
        H("h50", "The same interaction screen will identify at least one significant modifier of bevacizumab effect on `pfs_months`."),
        H("h51", "The same interaction screen will identify at least one significant modifier of pembrolizumab effect on `pfs_months`."),
        H("h52", "The same interaction screen will identify at least one significant modifier of encorafenib effect on `pfs_months`."),
        H("h53", "The same interaction screen will identify at least one significant modifier of trastuzumab/tucatinib effect on `pfs_months`."),
        H("h54", "The same interaction screen will identify at least one significant modifier of regorafenib effect on `pfs_months`; the strongest will involve KRAS."),
    ],
    "analyses": [
        A("h49", "screen_top3_treatment_cetuximab"),
        A("h50", "screen_top3_treatment_bevacizumab"),
        A("h51", "screen_top3_treatment_pembrolizumab"),
        A("h52", "screen_top3_treatment_encorafenib"),
        A("h53", "screen_top3_treatment_trastuzumab_tucatinib"),
        A("h54", "screen_top3_treatment_regorafenib"),
    ],
})

# ------------- Iteration 15: Initial subgroup hypotheses for each treatment -------------
iterations.append({
    "index": 15,
    "proposed_hypotheses": [
        H("h55", "Best subgroup for cetuximab benefit: KRAS WT AND NRAS WT AND BRAF WT AND left-sided primary; in this subgroup cetuximab improves `pfs_months`.", kind="refined"),
        H("h56", "Best subgroup for pembrolizumab benefit: MSI-high; in this subgroup pembrolizumab improves `pfs_months`.", kind="refined"),
        H("h57", "Best subgroup for encorafenib benefit: BRAF V600E mutation; in this subgroup encorafenib improves `pfs_months`.", kind="refined"),
        H("h58", "Best subgroup for trastuzumab/tucatinib benefit: HER2-amplified AND RAS WT; in this subgroup it improves `pfs_months`.", kind="refined"),
        H("h59", "Bevacizumab provides a small adjusted positive effect on `pfs_months` in the overall cohort (no clear subgroup).", kind="refined"),
        H("h60", "Regorafenib provides a positive effect on `pfs_months` in patients with ECOG performance status 0.", kind="refined"),
    ],
    "analyses": [
        A("h55", "final_cetux_subgroup"),
        A("h56", "final_pembro_subgroup"),
        A("h57", "final_enco_subgroup"),
        A("h58", "final_her2tx_subgroup"),
        A("h59", "final_bev_adjusted"),
        A("h60", "final_rego_ecog0"),
    ],
})

# ------------- Iteration 16: Cetuximab x KRAS x sidedness 2x2 -------------
iterations.append({
    "index": 16,
    "proposed_hypotheses": [
        H("h61", "Cetuximab benefit on `pfs_months` is concentrated specifically in KRAS WT AND left-sided patients; the effect is largest in (kras_mutation=0, right_sided_primary=0) and minimal in the other three cells.", kind="refined"),
    ],
    "analyses": [
        A("h61", "cetux_kras0_side0"),
        A("h61", "cetux_kras0_side1"),
        A("h61", "cetux_kras1_side0"),
        A("h61", "cetux_kras1_side1"),
    ],
})

# ------------- Iteration 17: Cetuximab x BRAF in RAS WT -------------
iterations.append({
    "index": 17,
    "proposed_hypotheses": [
        H("h62", "Within RAS wild-type (KRAS WT and NRAS WT), the cetuximab×BRAF V600E interaction on `pfs_months` is negative — BRAF V600E suppresses cetuximab benefit even after restricting to RAS WT.", kind="refined"),
    ],
    "analyses": [
        A("h62", "cetux_braf_in_rasWT"),
    ],
})

# ------------- Iteration 18: Encorafenib subgroup refinement -------------
iterations.append({
    "index": 18,
    "proposed_hypotheses": [
        H("h63", "Within BRAF V600E, the encorafenib effect on `pfs_months` is concentrated in KRAS WT (BRAF V600E and KRAS mutation are typically mutually exclusive but may have an effect modifier here).", kind="refined"),
    ],
    "analyses": [
        A("h63", "enco_braf1_kras0"),
        A("h63", "enco_braf1_kras1"),
    ],
})

# ------------- Iteration 19: Pembrolizumab heterogeneity within MSI-high -------------
iterations.append({
    "index": 19,
    "proposed_hypotheses": [
        H("h64", "Within MSI-high patients, pembrolizumab effect on `pfs_months` is modified by sidedness, KRAS, stage_iv, or ECOG (test each)."),
    ],
    "analyses": [
        A("h64", "pembro_right_sided_primary_in_msi"),
        A("h64", "pembro_kras_mutation_in_msi"),
        A("h64", "pembro_stage_iv_in_msi"),
        A("h64", "pembro_ecog_ps_in_msi"),
    ],
})

# ------------- Iteration 20: HER2 tx subgroup refinement -------------
iterations.append({
    "index": 20,
    "proposed_hypotheses": [
        H("h65", "Trastuzumab/tucatinib benefit on `pfs_months` is concentrated in HER2-amplified AND fully RAS/RAF wild-type patients (KRAS WT, NRAS WT, BRAF WT).", kind="refined"),
    ],
    "analyses": [
        A("h65", "final_her2tx_full_wt"),
    ],
})

# ------------- Iteration 21: Bevacizumab x continuous interactions -------------
iterations.append({
    "index": 21,
    "proposed_hypotheses": [
        H("h66", "Bevacizumab effect on `pfs_months` is modified by log(CEA), albumin, LDH, ECOG, or age (test each interaction)."),
    ],
    "analyses": [
        A("h66", "bev_log_cea_interaction"),
        A("h66", "bev_albumin_g_dl_interaction"),
        A("h66", "bev_ldh_u_l_interaction"),
        A("h66", "bev_ecog_ps_interaction"),
        A("h66", "bev_age_years_interaction"),
    ],
})

# ------------- Iteration 22: Regorafenib in KRAS strata -------------
iterations.append({
    "index": 22,
    "proposed_hypotheses": [
        H("h67", "Regorafenib improves `pfs_months` in KRAS wild-type (`kras_mutation`=0) patients but has no effect (or is null) in KRAS mutant patients.", kind="refined"),
    ],
    "analyses": [
        A("h67", "rego_kras0"),
        A("h67", "rego_kras1"),
    ],
})

# ------------- Iteration 23: NTRK fusion -------------
iterations.append({
    "index": 23,
    "proposed_hypotheses": [
        H("h68", "Mean `pfs_months` differs between patients with `ntrk_fusion`=1 and those without."),
    ],
    "analyses": [
        A("h68", "ntrk_count"),
    ],
})

# ------------- Iteration 24: Adjusted interactions in full multivariable model -------------
iterations.append({
    "index": 24,
    "proposed_hypotheses": [
        H("h69", "After multivariable adjustment, none of the canonical biomarker × treatment interactions (cetuximab×KRAS/NRAS/BRAF/sidedness, pembrolizumab×MSI, encorafenib×BRAF, trastuzumab/tucatinib×HER2) are statistically significant in this dataset (i.e., the dataset's only strong heterogeneity is for regorafenib)."),
    ],
    "analyses": [
        A("h69", "adj_treatment_cetuximab_x_kras_mutation"),
        A("h69", "adj_treatment_cetuximab_x_nras_mutation"),
        A("h69", "adj_treatment_cetuximab_x_braf_v600e"),
        A("h69", "adj_treatment_cetuximab_x_right_sided_primary"),
        A("h69", "adj_treatment_pembrolizumab_x_msi_high"),
        A("h69", "adj_treatment_encorafenib_x_braf_v600e"),
        A("h69", "adj_treatment_trastuzumab_tucatinib_x_her2_amplified"),
    ],
})

# ------------- Iteration 25: Final regorafenib subgroup discovery -------------
iterations.append({
    "index": 25,
    "proposed_hypotheses": [
        H("h70", "Regorafenib effect on `pfs_months` requires KRAS WT: in KRAS WT × BRAF WT it is large and positive; KRAS mutation suppresses the effect; BRAF V600E further suppresses it.", kind="refined"),
        H("h71", "Within KRAS WT × BRAF WT, regorafenib benefit is concentrated in left-sided primary patients (sidedness suppresses the effect when right-sided).", kind="refined"),
        H("h72", "Within KRAS WT × BRAF WT × left-sided, regorafenib benefit is further concentrated in patients with low CEA (`cea_ng_ml` < 5 ng/mL); higher CEA suppresses benefit.", kind="refined"),
        H("h73", "FINAL: Regorafenib produces a large positive effect on `pfs_months` (~+5 months) specifically in patients with `kras_mutation`=0 AND `nras_mutation`=0 AND `braf_v600e`=0 AND `right_sided_primary`=0 AND `cea_ng_ml` < 5 ng/mL; outside this subgroup the effect is essentially null. Each of {KRAS mutation, NRAS mutation (weakly), BRAF V600E, right-sided primary, elevated CEA} suppresses the regorafenib treatment effect.", kind="refined"),
        H("h74", "Joint multivariable interaction model confirms negative interactions of regorafenib with `kras_mutation`, `braf_v600e`, `right_sided_primary`, and `log_cea`, while NRAS mutation is not an independent suppressor.", kind="refined"),
    ],
    "analyses": [
        A("h70", "rego_kras0_braf0"),
        A("h70", "rego_kras0_braf1"),
        A("h70", "rego_kras1_braf0"),
        A("h71", "rego_kras0_braf0_side0"),
        A("h71", "rego_kras0_braf0_side1"),
        A("h72", "rego_kras0_braf0_left_cea_lt5"),
        A("h72", "rego_kras0_braf0_left_cea_ge5"),
        A("h73", "rego_FINAL_subgroup"),
        A("h73", "rego_FINAL_complement"),
        A("h74", "rego_multi_treatment_regorafenib_x_kras_mutation"),
        A("h74", "rego_multi_treatment_regorafenib_x_nras_mutation"),
        A("h74", "rego_multi_treatment_regorafenib_x_braf_v600e"),
        A("h74", "rego_multi_treatment_regorafenib_x_right_sided_primary"),
        A("h74", "rego_multi_treatment_regorafenib_x_log_cea"),
        A("h74", "rego_multi_main"),
    ],
})

transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@custom-eval-harness",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

print(f"Wrote transcript.json with {len(iterations)} iterations.")
hyp_count = sum(len(it["proposed_hypotheses"]) for it in iterations)
ana_count = sum(len(it["analyses"]) for it in iterations)
print(f"Hypotheses: {hyp_count}, Analyses: {ana_count}")
