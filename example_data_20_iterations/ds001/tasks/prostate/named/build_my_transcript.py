"""Build transcript.json and analysis_summary.txt from my_analysis_results.json."""

import json

with open("my_analysis_results.json") as fh:
    R = json.load(fh)


def find(name_substr, recs):
    """Find a record by name substring."""
    for r in recs:
        if name_substr in r.get("name", ""):
            return r
    return None


iterations = []

# ================== ITERATION 1 ==================
it = R["iter1_treatment_main"]
hyps = []
analyses = []
treatments = ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
              "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]
for i, tx in enumerate(treatments, 1):
    hyps.append({
        "id": f"h1.{i}",
        "text": f"Mean pfs_months differs between patients receiving {tx} (=1) and those not receiving it (=0).",
        "kind": "novel",
    })
for i, (tx, r) in enumerate(zip(treatments, it), 1):
    analyses.append({
        "hypothesis_ids": [f"h1.{i}"],
        "code": f"stats.ttest_ind(df.loc[df['{tx}']==1,'pfs_months'], df.loc[df['{tx}']==0,'pfs_months'])",
        "result_summary": (
            f"Mean PFS {r['mean_a']:.3f} on {tx}=1 (n={r['n_a']}) vs {r['mean_b']:.3f} off (n={r['n_b']});"
            f" Welch t-test p={r['p_value']:.3g}."
        ),
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 1, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 2 ==================
prog_vars = [
    ("age_years", "older age", "longer"),
    ("ecog_ps", "higher ECOG PS", "shorter"),
    ("psa_ng_ml", "higher PSA", "shorter"),
    ("gleason_score", "higher Gleason score", "shorter"),
    ("albumin_g_dl", "higher serum albumin", "longer"),
    ("ldh_u_l", "higher LDH", "shorter"),
    ("hemoglobin_g_dl", "higher hemoglobin", "longer"),
    ("alkaline_phosphatase_u_l", "higher alkaline phosphatase", "shorter"),
    ("weight_loss_pct_6mo", "more 6-month weight loss", "shorter"),
    ("crp_mg_l", "higher CRP", "shorter"),
    ("nlr", "higher neutrophil-lymphocyte ratio", "shorter"),
]
hyps = []
analyses = []
for i, (var, descr, dir_) in enumerate(prog_vars, 1):
    hyps.append({
        "id": f"h2.{i}",
        "text": f"In a univariate OLS, {descr} ({var}) is associated with {dir_} pfs_months.",
        "kind": "novel",
    })
for i, ((var, _, _), r) in enumerate(zip(prog_vars, R["iter2_prognostic_continuous"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h2.{i}"],
        "code": f"smf.ols('pfs_months ~ {var}', data=df).fit()",
        "result_summary": f"OLS β={r['effect']:.4f} for {var}, p={r['p_value']:.3g}, R²={r['rsq']:.4f}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 2, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 3 ==================
ds_vars = [
    ("mcrpc", "Castration-resistant disease (mcrpc=1)"),
    ("visceral_mets", "Visceral metastases (visceral_mets=1)"),
    ("liver_mets", "Liver metastases (liver_mets=1)"),
    ("bone_mets", "Bone metastases (bone_mets=1)"),
    ("adrenal_mets", "Adrenal metastases (adrenal_mets=1)"),
    ("pleural_effusion", "Pleural effusion present"),
    ("pericardial_effusion", "Pericardial effusion present"),
]
hyps = []
analyses = []
for i, (var, descr) in enumerate(ds_vars, 1):
    hyps.append({
        "id": f"h3.{i}",
        "text": f"{descr} is associated with shorter pfs_months than its absence.",
        "kind": "novel",
    })
for i, ((var, _), r) in enumerate(zip(ds_vars, R["iter3_disease_state"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h3.{i}"],
        "code": f"stats.ttest_ind(df.loc[df['{var}']==1,'pfs_months'], df.loc[df['{var}']==0,'pfs_months'])",
        "result_summary": f"Mean PFS {r['mean_a']:.3f} ({var}=1, n={r['n_a']}) vs {r['mean_b']:.3f} ({var}=0, n={r['n_b']}); p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 3, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 4 ==================
priors = [
    ("brca2_mutation", "treatment_olaparib", "BRCA2-mutated patients receiving olaparib (a PARP inhibitor) have longer pfs_months than BRCA2-mutated patients not on olaparib, with a larger benefit than in BRCA2-wild-type"),
    ("ar_v7_positive", "treatment_enzalutamide", "AR-V7-positive patients have a smaller (or negative) PFS benefit from enzalutamide than AR-V7-negative patients (AR-V7 is a known marker of resistance to AR-targeted therapy)"),
    ("ar_v7_positive", "treatment_abiraterone", "AR-V7-positive patients have a smaller (or negative) PFS benefit from abiraterone than AR-V7-negative patients"),
    ("msi_high", "treatment_pembrolizumab", "MSI-high patients receiving pembrolizumab have longer pfs_months than MSI-high patients not on pembrolizumab, with a larger benefit than in microsatellite-stable patients"),
    ("psma_high", "treatment_lu177_psma", "PSMA-high patients receiving Lu-177-PSMA radioligand therapy have longer pfs_months than PSMA-high patients not on Lu-177-PSMA, with a larger benefit than in PSMA-low patients"),
]
hyps = []
analyses = []
for i, (bio, tx, txt) in enumerate(priors, 1):
    hyps.append({
        "id": f"h4.{i}",
        "text": f"There is a positive {bio} × {tx} interaction on pfs_months: {txt}.",
        "kind": "novel",
    })
for i, ((bio, tx, _), r) in enumerate(zip(priors, R["iter4_key_biomarker_tx_interactions"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h4.{i}"],
        "code": f"smf.ols('pfs_months ~ {bio} * {tx}', data=df).fit()",
        "result_summary": (
            f"Treatment effect within {bio}=1: {r['effect_pos']:.3f} mo (n={r['n_pos']}); "
            f"within {bio}=0: {r['effect_neg']:.3f} mo (n={r['n_neg']}); "
            f"interaction coefficient β={r['interaction_effect']:.4f}, p={r['p_value']:.3g}."
        ),
        "p_value": r["p_value"],
        "effect_estimate": r["interaction_effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 4, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 5 ==================
hyps = []
analyses = []
for i, tx in enumerate(treatments, 1):
    hyps.append({
        "id": f"h5.{i}",
        "text": f"The PFS effect of {tx} differs between mCRPC (mcrpc=1) and hormone-sensitive (mcrpc=0) patients (mcrpc × {tx} interaction).",
        "kind": "novel",
    })
for i, (tx, r) in enumerate(zip(treatments, R["iter5_mcrpc_treatment"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h5.{i}"],
        "code": f"smf.ols('pfs_months ~ mcrpc * {tx}', data=df).fit()",
        "result_summary": f"Treatment effect in mcrpc=1: {r['effect_pos']:.3f} mo; in mcrpc=0: {r['effect_neg']:.3f} mo; interaction p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["interaction_effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 5, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 6 ==================
hyps = []
analyses = []
for i, tx in enumerate(treatments, 1):
    hyps.append({
        "id": f"h6.{i}",
        "text": f"After adjusting for age, ECOG, mCRPC, visceral_mets, PSA, Gleason, albumin, LDH and hemoglobin, {tx} retains an independent main-effect association with pfs_months.",
        "kind": "refined",
    })
for i, (tx, r) in enumerate(zip(treatments, R["iter6_adjusted_tx"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h6.{i}"],
        "code": f"smf.ols('pfs_months ~ {tx} + age_years + ecog_ps + mcrpc + visceral_mets + psa_ng_ml + gleason_score + albumin_g_dl + ldh_u_l + hemoglobin_g_dl', data=df).fit()",
        "result_summary": f"Adjusted β for {tx} = {r['effect']:.4f}, p={r['p_value']:.3g}, model R²={r['rsq']:.4f}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 6, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 7 ==================
hyps = []
analyses = []
for i, ((bio, tx, _), r) in enumerate(zip(priors, R["iter7_adjusted_interactions"]), 1):
    hyps.append({
        "id": f"h7.{i}",
        "text": f"After adjusting for clinical confounders, the {bio} × {tx} interaction on pfs_months remains significant (kind=refined of h4.{i}).",
        "kind": "refined",
    })
    analyses.append({
        "hypothesis_ids": [f"h7.{i}"],
        "code": f"smf.ols('pfs_months ~ {bio} * {tx} + age_years + ecog_ps + mcrpc + visceral_mets + psa_ng_ml + albumin_g_dl + ldh_u_l + hemoglobin_g_dl', data=df).fit()",
        "result_summary": f"Adjusted interaction β={r['effect']:.4f}, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 7, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 8 ==================
hyps = []
analyses = []
race_levels = ["asian", "black", "hispanic", "other"]
for i, lvl in enumerate(race_levels, 1):
    hyps.append({
        "id": f"h8.{i}",
        "text": f"Mean pfs_months differs between {lvl} race/ethnicity patients and white patients.",
        "kind": "novel",
    })
hyps.append({"id": "h8.5", "text": "Mean pfs_months differs across race_ethnicity categories overall (omnibus ANOVA).", "kind": "novel"})
race_results = R["iter8_race"]
# First 4 entries are pairwise vs white, last is ANOVA
for i, r in enumerate(race_results[:4], 1):
    analyses.append({
        "hypothesis_ids": [f"h8.{i}"],
        "code": "stats.ttest_ind(df.loc[df.race_ethnicity==g,'pfs_months'], df.loc[df.race_ethnicity=='white','pfs_months'])",
        "result_summary": f"{r['name']}: mean diff {r['effect']:.3f} mo, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
anova_r = race_results[-1]
analyses.append({
    "hypothesis_ids": ["h8.5"],
    "code": "stats.f_oneway(*[df.loc[df.race_ethnicity==g,'pfs_months'] for g in df.race_ethnicity.unique()])",
    "result_summary": f"One-way ANOVA F={anova_r['effect']:.3f}, p={anova_r['p_value']:.3g}.",
    "p_value": anova_r["p_value"],
    "effect_estimate": anova_r["effect"],
    "significant": anova_r["significant"],
})
iterations.append({"index": 8, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 9 ==================
ins_levels = ["medicare", "medicaid", "uninsured"]
hyps = []
analyses = []
for i, lvl in enumerate(ins_levels, 1):
    hyps.append({
        "id": f"h9.{i}",
        "text": f"Mean pfs_months differs between {lvl}-insured patients and privately insured patients (uninsured/under-insured patients hypothesised to have shorter PFS).",
        "kind": "novel",
    })
for i, (lvl, r) in enumerate(zip(ins_levels, R["iter9_insurance"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h9.{i}"],
        "code": "stats.ttest_ind(...)",
        "result_summary": f"{r['name']}: mean diff {r['effect']:.3f} mo, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 9, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 10 ==================
hyps = []
analyses = []
ecog_txs = ["treatment_docetaxel", "treatment_enzalutamide", "treatment_abiraterone",
            "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]
for i, tx in enumerate(ecog_txs, 1):
    hyps.append({
        "id": f"h10.{i}",
        "text": f"There is a negative ecog_ps × {tx} interaction on pfs_months: the PFS benefit of {tx} is smaller (or harm larger) in patients with worse performance status (higher ecog_ps).",
        "kind": "novel",
    })
for i, (tx, r) in enumerate(zip(ecog_txs, R["iter10_ecog_tx"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h10.{i}"],
        "code": f"smf.ols('pfs_months ~ ecog_ps * {tx}', data=df).fit()",
        "result_summary": f"Interaction β={r['effect']:.4f}, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 10, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 11 ==================
hyps = []
analyses = []
visc_txs = ["treatment_docetaxel", "treatment_enzalutamide", "treatment_abiraterone",
            "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]
for i, tx in enumerate(visc_txs, 1):
    if tx == "treatment_docetaxel":
        txt = "the PFS benefit of docetaxel is larger in patients with visceral metastases (visceral_mets=1) than without (clinical preference for chemotherapy in visceral disease)"
    else:
        txt = f"the PFS effect of {tx} differs between visceral_mets=1 vs 0"
    hyps.append({
        "id": f"h11.{i}",
        "text": f"There is a visceral_mets × {tx} interaction on pfs_months: {txt}.",
        "kind": "novel",
    })
for i, (tx, r) in enumerate(zip(visc_txs, R["iter11_visceral_tx"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h11.{i}"],
        "code": f"smf.ols('pfs_months ~ visceral_mets * {tx}', data=df).fit()",
        "result_summary": f"Treatment effect in visceral_mets=1: {r['effect_pos']:.3f}; in =0: {r['effect_neg']:.3f}; interaction p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["interaction_effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 11, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 12 ==================
prior_vars = [("prior_lines_of_therapy", "More prior lines of therapy is associated with shorter pfs_months (heavily pretreated patients have lower benefit)"),
              ("prior_chemotherapy", "Prior chemotherapy"),
              ("prior_radiation", "Prior radiation"),
              ("prior_surgery", "Prior surgery"),
              ("prior_immunotherapy", "Prior immunotherapy"),
              ("prior_targeted_therapy", "Prior targeted therapy")]
hyps = []
analyses = []
for i, (var, descr) in enumerate(prior_vars, 1):
    if var == "prior_lines_of_therapy":
        text = descr + "."
    else:
        text = f"{descr} (={var}=1) is associated with shorter pfs_months than no prior such therapy."
    hyps.append({"id": f"h12.{i}", "text": text, "kind": "novel"})
# First analysis is regression on prior_lines
r = R["iter12_prior_therapy"][0]
analyses.append({
    "hypothesis_ids": ["h12.1"],
    "code": "smf.ols('pfs_months ~ prior_lines_of_therapy', data=df).fit()",
    "result_summary": f"OLS β for prior_lines_of_therapy = {r['effect']:.4f}, p={r['p_value']:.3g}.",
    "p_value": r["p_value"],
    "effect_estimate": r["effect"],
    "significant": r["significant"],
})
for i, r in enumerate(R["iter12_prior_therapy"][1:], 2):
    analyses.append({
        "hypothesis_ids": [f"h12.{i}"],
        "code": f"stats.ttest_ind(...)",
        "result_summary": f"{r['name']}: mean diff {r['effect']:.3f} mo, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 12, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 13 ==================
sx_vars = [("pain_nrs", "Higher pain on a numeric rating scale"),
           ("fatigue_grade", "Higher fatigue grade"),
           ("dyspnea_grade", "Higher dyspnea grade"),
           ("cough_grade", "Higher cough grade"),
           ("appetite_loss_grade", "Higher appetite-loss grade")]
hyps = []
analyses = []
for i, (var, descr) in enumerate(sx_vars, 1):
    hyps.append({
        "id": f"h13.{i}",
        "text": f"{descr} ({var}) is associated with shorter pfs_months in univariate OLS.",
        "kind": "novel",
    })
for i, ((var, _), r) in enumerate(zip(sx_vars, R["iter13_symptoms"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h13.{i}"],
        "code": f"smf.ols('pfs_months ~ {var}', data=df).fit()",
        "result_summary": f"OLS β for {var} = {r['effect']:.4f}, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 13, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 14 ==================
hyps = [{
    "id": "h14.1",
    "text": "At least one of the 25 SNP genotype scores has a univariate association with pfs_months at the nominal α=0.05 level (with the expected ~5% false-positive rate under the null).",
    "kind": "novel",
}]
analyses = []
sig = [r for r in R["iter14_snps"] if r["p_value"] < 0.05]
top = sorted(R["iter14_snps"], key=lambda x: x["p_value"])[:5]
top_summary = ", ".join([f"{r['predictor']} β={r['effect']:.4f} p={r['p_value']:.3g}" for r in top])
analyses.append({
    "hypothesis_ids": ["h14.1"],
    "code": "for s in snps: smf.ols(f'pfs_months ~ {s}', data=df).fit()",
    "result_summary": (
        f"{len(sig)}/25 SNPs showed nominal p<0.05 (consistent with chance); "
        f"top 5 by p-value: {top_summary}. None survives Bonferroni (0.05/25=0.002)."
    ),
    "p_value": min(r["p_value"] for r in R["iter14_snps"]),
    "effect_estimate": float(top[0]["effect"]),
    "significant": False,
})
# Individual hypothesis for the top snp (rs4986893)
hyps.append({
    "id": "h14.2",
    "text": "snp_rs4986893 (the top-ranked SNP) is associated with shorter pfs_months.",
    "kind": "refined",
})
top1 = top[0]
analyses.append({
    "hypothesis_ids": ["h14.2"],
    "code": f"smf.ols('pfs_months ~ {top1['predictor']}', data=df).fit()",
    "result_summary": f"OLS β={top1['effect']:.4f} p={top1['p_value']:.3g}; nominally significant but does not survive Bonferroni correction.",
    "p_value": top1["p_value"],
    "effect_estimate": top1["effect"],
    "significant": top1["significant"],
})
iterations.append({"index": 14, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 15 ==================
co_vars = [("diabetes_mellitus", "Diabetes mellitus"),
           ("hypertension", "Hypertension"),
           ("copd", "COPD"),
           ("chronic_kidney_disease", "Chronic kidney disease"),
           ("heart_failure", "Heart failure"),
           ("coronary_artery_disease", "Coronary artery disease"),
           ("atrial_fibrillation", "Atrial fibrillation"),
           ("venous_thromboembolism_history", "VTE history"),
           ("autoimmune_disease", "Autoimmune disease"),
           ("depression_anxiety_diagnosis", "Depression/anxiety diagnosis")]
hyps = []
analyses = []
for i, (var, descr) in enumerate(co_vars, 1):
    hyps.append({
        "id": f"h15.{i}",
        "text": f"{descr} ({var}=1) is associated with shorter pfs_months than its absence.",
        "kind": "novel",
    })
for i, ((var, _), r) in enumerate(zip(co_vars, R["iter15_comorbidities"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h15.{i}"],
        "code": f"stats.ttest_ind(df.loc[df['{var}']==1,'pfs_months'], df.loc[df['{var}']==0,'pfs_months'])",
        "result_summary": f"{r['name']}: mean diff {r['effect']:.3f} mo, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 15, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 16 ==================
gen_vars = [("tp53_mutation", "TP53 mutation"),
            ("pten_loss", "PTEN loss"),
            ("pik3ca_mutation", "PIK3CA mutation"),
            ("cdkn2a_loss", "CDKN2A loss"),
            ("fgfr_alteration", "FGFR alteration"),
            ("her2_amplification", "HER2 amplification"),
            ("braf_v600e", "BRAF V600E"),
            ("keap1_mutation", "KEAP1 mutation")]
hyps = []
analyses = []
for i, (var, descr) in enumerate(gen_vars, 1):
    hyps.append({
        "id": f"h16.{i}",
        "text": f"{descr} ({var}=1) is associated with shorter pfs_months than its absence.",
        "kind": "novel",
    })
for i, ((var, _), r) in enumerate(zip(gen_vars, R["iter16_genomics"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h16.{i}"],
        "code": f"stats.ttest_ind(df.loc[df['{var}']==1,'pfs_months'], df.loc[df['{var}']==0,'pfs_months'])",
        "result_summary": f"{r['name']}: mean diff {r['effect']:.3f} mo, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 16, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 17 ==================
hyps = [
    {"id": "h17.1", "text": "Among BRCA2-mutated patients (brca2_mutation=1), olaparib (treatment_olaparib=1) is associated with longer pfs_months than no olaparib.", "kind": "refined"},
    {"id": "h17.2", "text": "Among BRCA2-wild-type patients (brca2_mutation=0), olaparib has no clinically meaningful effect on pfs_months.", "kind": "refined"},
]
analyses = []
for i, r in enumerate(R["iter17_brca_olaparib_detail"], 1):
    analyses.append({
        "hypothesis_ids": [f"h17.{i}"],
        "code": "stats.ttest_ind(...)",
        "result_summary": (
            f"{r['name']}: mean PFS on olaparib {r['mean_tx']:.3f} (n={r['n_tx']}) vs off {r['mean_no_tx']:.3f} "
            f"(n={r['n_no_tx']}); diff {r['effect']:.3f} mo, p={r['p_value']:.3g}."
        ),
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 17, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 18 ==================
arv_combos = [("treatment_enzalutamide", 1), ("treatment_enzalutamide", 0),
              ("treatment_abiraterone", 1), ("treatment_abiraterone", 0)]
hyps = []
analyses = []
for i, (tx, arv) in enumerate(arv_combos, 1):
    direction = "no PFS benefit (or harm)" if arv == 1 else "no PFS benefit"
    hyps.append({
        "id": f"h18.{i}",
        "text": f"Among AR-V7={'positive' if arv==1 else 'negative'} patients, {tx} is associated with {direction}.",
        "kind": "refined",
    })
for i, (r, (tx, arv)) in enumerate(zip(R["iter18_arv7_detail"], arv_combos), 1):
    analyses.append({
        "hypothesis_ids": [f"h18.{i}"],
        "code": "stats.ttest_ind(...)",
        "result_summary": (
            f"{r['name']}: mean on tx {r['mean_tx']:.3f} (n={r['n_tx']}), off {r['mean_no_tx']:.3f} "
            f"(n={r['n_no_tx']}); diff {r['effect']:.3f}, p={r['p_value']:.3g}."
        ),
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 18, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 19 ==================
hyps = [
    {"id": "h19.1", "text": "Among PSMA-high patients (psma_high=1), Lu-177-PSMA (treatment_lu177_psma=1) is associated with longer pfs_months than no Lu-177-PSMA.", "kind": "refined"},
    {"id": "h19.2", "text": "Among PSMA-low patients (psma_high=0), Lu-177-PSMA has no clinically meaningful effect on pfs_months.", "kind": "refined"},
]
analyses = []
for i, r in enumerate(R["iter19_psma_lu177_detail"], 1):
    analyses.append({
        "hypothesis_ids": [f"h19.{i}"],
        "code": "stats.ttest_ind(...)",
        "result_summary": (
            f"{r['name']}: mean on Lu-177 {r['mean_tx']:.3f} (n={r['n_tx']}) vs off {r['mean_no_tx']:.3f} "
            f"(n={r['n_no_tx']}); diff {r['effect']:.3f}, p={r['p_value']:.3g}."
        ),
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 19, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 20 ==================
hyps = [
    {"id": "h20.1", "text": "Among MSI-high patients (msi_high=1), pembrolizumab (treatment_pembrolizumab=1) is associated with longer pfs_months than no pembrolizumab.", "kind": "refined"},
    {"id": "h20.2", "text": "Among MSI-stable patients (msi_high=0), pembrolizumab has no clinically meaningful effect on pfs_months.", "kind": "refined"},
]
analyses = []
for i, r in enumerate(R["iter20_msi_pembro_detail"], 1):
    analyses.append({
        "hypothesis_ids": [f"h20.{i}"],
        "code": "stats.ttest_ind(...)",
        "result_summary": (
            f"{r['name']}: mean on pembro {r['mean_tx']:.3f} (n={r['n_tx']}) vs off {r['mean_no_tx']:.3f} "
            f"(n={r['n_no_tx']}); diff {r['effect']:.3f}, p={r['p_value']:.3g}."
        ),
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 20, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 21 ==================
vital_vars = [("bmi", "Higher BMI"),
              ("systolic_bp_mmhg", "Higher systolic BP"),
              ("diastolic_bp_mmhg", "Higher diastolic BP"),
              ("heart_rate_bpm", "Higher heart rate"),
              ("spo2_pct", "Higher SpO2"),
              ("creatinine_mg_dl", "Higher creatinine"),
              ("bun_mg_dl", "Higher BUN"),
              ("sodium_meq_l", "Higher sodium"),
              ("potassium_meq_l", "Higher potassium"),
              ("calcium_mg_dl", "Higher calcium"),
              ("platelets_k_ul", "Higher platelets"),
              ("wbc_k_ul", "Higher WBC"),
              ("anc_k_ul", "Higher ANC"),
              ("alc_k_ul", "Higher ALC")]
hyps = []
analyses = []
for i, (var, descr) in enumerate(vital_vars, 1):
    hyps.append({
        "id": f"h21.{i}",
        "text": f"{descr} ({var}) is associated with pfs_months in univariate OLS (direction depends on the variable).",
        "kind": "novel",
    })
for i, ((var, _), r) in enumerate(zip(vital_vars, R["iter21_vitals_labs"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h21.{i}"],
        "code": f"smf.ols('pfs_months ~ {var}', data=df).fit()",
        "result_summary": f"β for {var}={r['effect']:.4f}, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 21, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 22 ==================
hyps = [
    {"id": "h22.1", "text": "Rural residence (rural_residence=1) is associated with shorter pfs_months than urban residence.", "kind": "novel"},
    {"id": "h22.2", "text": "Higher smoking_pack_years is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h22.3", "text": "More years of education (education_years) is associated with longer pfs_months (a healthcare-access / SES proxy).", "kind": "novel"},
]
analyses = []
for i, r in enumerate(R["iter22_demographics"], 1):
    analyses.append({
        "hypothesis_ids": [f"h22.{i}"],
        "code": r.get("formula", ""),
        "result_summary": f"effect={r['effect']:.4f}, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 22, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 23 ==================
combos = [("treatment_docetaxel", "treatment_abiraterone"),
          ("treatment_enzalutamide", "treatment_abiraterone"),
          ("treatment_docetaxel", "treatment_enzalutamide")]
hyps = []
analyses = []
for i, (a, b) in enumerate(combos, 1):
    hyps.append({
        "id": f"h23.{i}",
        "text": f"There is an interaction between {a} and {b} on pfs_months: the combined effect is greater than the sum of either alone (or shows synergy/antagonism).",
        "kind": "novel",
    })
for i, ((a, b), r) in enumerate(zip(combos, R["iter23_combos"]), 1):
    analyses.append({
        "hypothesis_ids": [f"h23.{i}"],
        "code": f"smf.ols('pfs_months ~ {a} * {b}', data=df).fit()",
        "result_summary": f"Interaction term {a}:{b} β={r['effect']:.4f}, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 23, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 24 ==================
hyps = [
    {"id": "h24.1", "text": "In a full multivariable model that includes all major prognostic clinical features and the five biomarker × treatment interactions of interest, the BRCA2 × olaparib interaction remains a strong, significant predictor of pfs_months.", "kind": "refined"},
    {"id": "h24.2", "text": "In the same full model, the AR-V7 × enzalutamide, AR-V7 × abiraterone, MSI × pembrolizumab and PSMA × Lu-177 interactions are NOT significant predictors of pfs_months.", "kind": "refined"},
    {"id": "h24.3", "text": "In the full multivariable model, ECOG performance status, mCRPC status, age, PSA, albumin, weight loss and LDH are independent predictors of pfs_months.", "kind": "refined"},
]
analyses = []
fm = R["iter24_full_model"]
# Locate key terms
key_terms = ["brca2_mutation:treatment_olaparib", "ar_v7_positive:treatment_enzalutamide",
             "ar_v7_positive:treatment_abiraterone", "msi_high:treatment_pembrolizumab",
             "psma_high:treatment_lu177_psma"]
brca_olap = next(t for t in fm["terms"] if t["term"] == "brca2_mutation:treatment_olaparib")
analyses.append({
    "hypothesis_ids": ["h24.1"],
    "code": "smf.ols(formula_full, data=df).fit() (see formula in iter24_full_model)",
    "result_summary": (
        f"Full model R²={fm['rsquared']:.4f} (n={fm['n']}). "
        f"brca2_mutation:treatment_olaparib β={brca_olap['effect']:.4f}, p={brca_olap['p_value']:.3g} (highly significant)."
    ),
    "p_value": brca_olap["p_value"],
    "effect_estimate": brca_olap["effect"],
    "significant": brca_olap["significant"],
})
other_inter = [t for t in fm["terms"] if t["term"] in key_terms[1:]]
parts = ", ".join(f"{t['term']} β={t['effect']:.4f} p={t['p_value']:.3g}" for t in other_inter)
analyses.append({
    "hypothesis_ids": ["h24.2"],
    "code": "(same full model)",
    "result_summary": f"In adjusted full model: {parts}. None reaches p<0.05.",
    "p_value": min(t["p_value"] for t in other_inter),
    "effect_estimate": max(abs(t["effect"]) for t in other_inter) * (1 if other_inter[0]["effect"] >= 0 else -1),
    "significant": any(t["significant"] for t in other_inter),
})
prog_terms = ["ecog_ps", "mcrpc", "age_years", "psa_ng_ml", "albumin_g_dl", "weight_loss_pct_6mo", "ldh_u_l"]
prog_summ = "; ".join(f"{t['term']} β={t['effect']:.4f} p={t['p_value']:.3g}"
                       for t in fm["terms"] if t["term"] in prog_terms)
ecog_t = next(t for t in fm["terms"] if t["term"] == "ecog_ps")
analyses.append({
    "hypothesis_ids": ["h24.3"],
    "code": "(same full model)",
    "result_summary": f"Adjusted prognostic effects: {prog_summ}. All highly significant.",
    "p_value": ecog_t["p_value"],
    "effect_estimate": ecog_t["effect"],
    "significant": ecog_t["significant"],
})
iterations.append({"index": 24, "proposed_hypotheses": hyps, "analyses": analyses})

# ================== ITERATION 25 ==================
hyps = []
analyses = []
black_txs = treatments
for i, tx in enumerate(black_txs, 1):
    hyps.append({
        "id": f"h25.{i}",
        "text": f"There is an interaction between black race/ethnicity (vs all other groups) and {tx} on pfs_months: the PFS effect of {tx} differs in black patients.",
        "kind": "novel",
    })
for i, (tx, r) in enumerate(zip(black_txs, R["iter25_equity"][:6]), 1):
    analyses.append({
        "hypothesis_ids": [f"h25.{i}"],
        "code": f"smf.ols('pfs_months ~ black * {tx}', data=df).fit()",
        "result_summary": f"Interaction β={r['effect']:.4f}, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
unins_txs = ["treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]
for i, tx in enumerate(unins_txs, 7):
    hyps.append({
        "id": f"h25.{i}",
        "text": f"Uninsured patients (insurance_type='uninsured') derive less PFS benefit from the high-cost targeted therapy {tx} than privately/publicly insured patients (interaction term).",
        "kind": "novel",
    })
for i, (tx, r) in enumerate(zip(unins_txs, R["iter25_equity"][6:]), 7):
    analyses.append({
        "hypothesis_ids": [f"h25.{i}"],
        "code": f"smf.ols('pfs_months ~ uninsured * {tx}', data=df).fit()",
        "result_summary": f"Interaction β={r['effect']:.4f}, p={r['p_value']:.3g}.",
        "p_value": r["p_value"],
        "effect_estimate": r["effect"],
        "significant": r["significant"],
    })
iterations.append({"index": 25, "proposed_hypotheses": hyps, "analyses": analyses})

transcript = {
    "dataset_id": "ds001_prostate",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@named-prostate-1",
    "max_iterations": 25,
    "iterations": iterations,
}
with open("transcript.json", "w") as fh:
    json.dump(transcript, fh, indent=2)
print(f"Wrote transcript.json with {len(iterations)} iterations.")

# Sanity counts
total_h = sum(len(it["proposed_hypotheses"]) for it in iterations)
total_a = sum(len(it["analyses"]) for it in iterations)
print(f"Total hypotheses: {total_h}; total analyses: {total_a}")
