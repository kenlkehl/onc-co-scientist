"""Build transcript.json from _results.json."""
import json

with open("_results.json") as f:
    R = json.load(f)


def hf(p):
    try:
        return float(p)
    except Exception:
        return None


# Build iterations
iters = []

# ---- ITER 1 ----
iter1 = R["iter1_tx_main_effects"]
hyps = []
analyses = []
for i, tx in enumerate([
    "treatment_midostaurin", "treatment_gilteritinib", "treatment_ivosidenib",
    "treatment_enasidenib", "treatment_venetoclax_azacitidine", "treatment_7plus3"]):
    hid = f"h1_{tx}"
    hyps.append({
        "id": hid,
        "text": f"Patients receiving {tx}=1 have a higher objective_response rate than those with {tx}=0.",
        "kind": "novel",
    })
    r = iter1[tx]
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2_contingency on objective_response by {tx}",
        "result_summary": r["summary"],
        "p_value": hf(r["p_value"]),
        "effect_estimate": hf(r["effect_estimate"]),
        "significant": hf(r["p_value"]) < 0.05 if r["p_value"] is not None else None,
    })
iters.append({"index": 1, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 2 ----
iter2 = R["iter2_mutation_main_effects"]
hyps, analyses = [], []
descs = {
    "flt3_itd": ("flt3_itd=1", "higher", "FLT3-ITD historically confers high relapse risk but elevated WBC"),
    "flt3_tkd": ("flt3_tkd=1", "higher", "FLT3-TKD typically more inhibitor-responsive"),
    "idh1_mutation": ("idh1_mutation=1", "higher", "IDH1 mutation considered intermediate risk"),
    "idh2_mutation": ("idh2_mutation=1", "higher", "IDH2 mutation considered intermediate risk"),
    "npm1_mutation": ("npm1_mutation=1", "higher", "NPM1 mutation is favorable risk in AML"),
    "tp53_mutation": ("tp53_mutation=1", "lower", "TP53 mutation is adverse risk in AML"),
    "complex_karyotype": ("complex_karyotype=1", "lower", "Complex karyotype is adverse risk"),
    "secondary_aml": ("secondary_aml=1", "lower", "Secondary AML responds worse than de novo"),
}
for m, (label, direction, _) in descs.items():
    hid = f"h2_{m}"
    hyps.append({
        "id": hid,
        "text": f"Patients with {label} have a {direction} objective_response rate than those without.",
        "kind": "novel",
    })
    r = iter2[m]
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2_contingency on objective_response by {m}",
        "result_summary": r["summary"],
        "p_value": hf(r["p_value"]),
        "effect_estimate": hf(r["effect_estimate"]),
        "significant": hf(r["p_value"]) < 0.05 if r["p_value"] is not None else None,
    })
iters.append({"index": 2, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 3 — FLT3-ITD × midostaurin ----
r = R["iter3_flt3itd_x_midostaurin"]
hid = "h3_flt3itd_mido"
hyps = [{
    "id": hid,
    "text": "FLT3-ITD-positive patients show greater objective_response benefit from treatment_midostaurin than FLT3-ITD-negative patients (positive interaction).",
    "kind": "novel",
}]
analyses = [{
    "hypothesis_ids": [hid],
    "code": "logit('objective_response ~ flt3_itd * treatment_midostaurin', df)",
    "result_summary": r["summary"],
    "p_value": hf(r["interaction_p"]),
    "effect_estimate": hf(r["interaction_beta"]),
    "significant": hf(r["interaction_p"]) < 0.05 if r["interaction_p"] is not None else None,
}]
iters.append({"index": 3, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 4 — FLT3-ITD × gilteritinib ----
r = R["iter4_flt3itd_x_gilteritinib"]
hid = "h4_flt3itd_gilt"
hyps = [{
    "id": hid,
    "text": "FLT3-ITD-positive patients show greater objective_response benefit from treatment_gilteritinib than FLT3-ITD-negative patients (positive interaction).",
    "kind": "novel",
}]
analyses = [{
    "hypothesis_ids": [hid],
    "code": "logit('objective_response ~ flt3_itd * treatment_gilteritinib', df)",
    "result_summary": r["summary"],
    "p_value": hf(r["interaction_p"]),
    "effect_estimate": hf(r["interaction_beta"]),
    "significant": hf(r["interaction_p"]) < 0.05 if r["interaction_p"] is not None else None,
}]
iters.append({"index": 4, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 5 — IDH1 × ivosidenib ----
r = R["iter5_idh1_x_ivosidenib"]
hid = "h5_idh1_ivo"
hyps = [{
    "id": hid,
    "text": "idh1_mutation-positive patients show greater objective_response benefit from treatment_ivosidenib than idh1_mutation-negative patients (positive interaction).",
    "kind": "novel",
}]
analyses = [{
    "hypothesis_ids": [hid],
    "code": "logit('objective_response ~ idh1_mutation * treatment_ivosidenib', df)",
    "result_summary": r["summary"],
    "p_value": hf(r["interaction_p"]),
    "effect_estimate": hf(r["interaction_beta"]),
    "significant": hf(r["interaction_p"]) < 0.05 if r["interaction_p"] is not None else None,
}]
iters.append({"index": 5, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 6 — IDH2 × enasidenib ----
r = R["iter6_idh2_x_enasidenib"]
hid = "h6_idh2_ena"
hyps = [{
    "id": hid,
    "text": "idh2_mutation-positive patients show greater objective_response benefit from treatment_enasidenib than idh2_mutation-negative patients (positive interaction).",
    "kind": "novel",
}]
analyses = [{
    "hypothesis_ids": [hid],
    "code": "logit('objective_response ~ idh2_mutation * treatment_enasidenib', df)",
    "result_summary": r["summary"],
    "p_value": hf(r["interaction_p"]),
    "effect_estimate": hf(r["interaction_beta"]),
    "significant": hf(r["interaction_p"]) < 0.05 if r["interaction_p"] is not None else None,
}]
iters.append({"index": 6, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 7 — Age, ECOG, fitness ----
r = R["iter7_age_ecog_fitness"]
hyps, analyses = [], []
hyps.append({"id": "h7_age", "text": "Higher age_years is associated with lower objective_response rate.", "kind": "novel"})
hyps.append({"id": "h7_ecog", "text": "Higher ecog_ps is associated with lower objective_response rate.", "kind": "novel"})
hyps.append({"id": "h7_unfit", "text": "Patients flagged unfit_for_intensive=1 have a lower objective_response rate than fit patients.", "kind": "novel"})
analyses.append({
    "hypothesis_ids": ["h7_age"],
    "code": "logit('objective_response ~ age_years', df)",
    "result_summary": r["age_years"]["summary"],
    "p_value": hf(r["age_years"]["p_value"]),
    "effect_estimate": hf(r["age_years"]["beta"]),
    "significant": hf(r["age_years"]["p_value"]) < 0.05,
})
analyses.append({
    "hypothesis_ids": ["h7_ecog"],
    "code": "logit('objective_response ~ ecog_ps', df)",
    "result_summary": r["ecog_ps"]["summary"],
    "p_value": hf(r["ecog_ps"]["p_value"]),
    "effect_estimate": hf(r["ecog_ps"]["beta"]),
    "significant": hf(r["ecog_ps"]["p_value"]) < 0.05,
})
analyses.append({
    "hypothesis_ids": ["h7_unfit"],
    "code": "chi2_contingency on objective_response by unfit_for_intensive",
    "result_summary": r["unfit_for_intensive"]["summary"],
    "p_value": hf(r["unfit_for_intensive"]["p_value"]),
    "effect_estimate": hf(r["unfit_for_intensive"]["effect_estimate"]),
    "significant": hf(r["unfit_for_intensive"]["p_value"]) < 0.05,
})
iters.append({"index": 7, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 8 — Disease burden ----
r = R["iter8_disease_burden"]
hyps, analyses = [], []
mapping = {
    "wbc_k_per_ul": "Higher wbc_k_per_ul is associated with lower objective_response rate.",
    "blast_pct_marrow": "Higher blast_pct_marrow is associated with lower objective_response rate.",
    "ldh_u_l": "Higher ldh_u_l is associated with lower objective_response rate.",
}
for col, txt in mapping.items():
    hid = f"h8_{col}"
    hyps.append({"id": hid, "text": txt, "kind": "novel"})
    rr = r[col]
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit('objective_response ~ {col}', df)",
        "result_summary": rr["summary"],
        "p_value": hf(rr["p_value"]),
        "effect_estimate": hf(rr["beta"]),
        "significant": hf(rr["p_value"]) < 0.05,
    })
iters.append({"index": 8, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 9 — Inflammation/nutrition ----
r = R["iter9_inflammation_nutrition"]
hyps, analyses = [], []
mapping = {
    "albumin_g_dl": ("Higher albumin_g_dl is associated with higher objective_response rate (better nutrition).", "+"),
    "crp_mg_l": ("Higher crp_mg_l is associated with lower objective_response rate (more inflammation).", "-"),
    "nlr": ("Higher nlr (neutrophil-to-lymphocyte ratio) is associated with lower objective_response rate.", "-"),
}
for col, (txt, _) in mapping.items():
    hid = f"h9_{col}"
    hyps.append({"id": hid, "text": txt, "kind": "novel"})
    rr = r[col]
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit('objective_response ~ {col}', df)",
        "result_summary": rr["summary"],
        "p_value": hf(rr["p_value"]),
        "effect_estimate": hf(rr["beta"]),
        "significant": hf(rr["p_value"]) < 0.05,
    })
iters.append({"index": 9, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 10 — TP53 × intensive treatments ----
r = R["iter10_tp53_x_intensive"]
hyps, analyses = [], []
for tx in ["treatment_7plus3", "treatment_venetoclax_azacitidine"]:
    hid = f"h10_tp53_{tx}"
    hyps.append({
        "id": hid,
        "text": f"tp53_mutation-positive patients have a relatively lower objective_response benefit from {tx} than TP53-negative patients (negative interaction).",
        "kind": "novel",
    })
    rr = r[tx]
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit('objective_response ~ tp53_mutation * {tx}', df)",
        "result_summary": rr["summary"],
        "p_value": hf(rr["interaction_p"]),
        "effect_estimate": hf(rr["interaction_beta"]),
        "significant": hf(rr["interaction_p"]) < 0.05,
    })
iters.append({"index": 10, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 11 — NPM1 × 7+3 ----
r = R["iter11_npm1_x_7plus3"]
hid = "h11_npm1_7plus3"
hyps = [{
    "id": hid,
    "text": "npm1_mutation-positive patients have a greater objective_response benefit from treatment_7plus3 than NPM1-negative patients (positive interaction).",
    "kind": "novel",
}]
analyses = [{
    "hypothesis_ids": [hid],
    "code": "logit('objective_response ~ npm1_mutation * treatment_7plus3', df)",
    "result_summary": r["summary"],
    "p_value": hf(r["interaction_p"]),
    "effect_estimate": hf(r["interaction_beta"]),
    "significant": hf(r["interaction_p"]) < 0.05,
}]
iters.append({"index": 11, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 12 — Complex karyotype × treatments ----
r = R["iter12_complex_kary_x_tx"]
hyps, analyses = [], []
for tx in ["treatment_7plus3", "treatment_venetoclax_azacitidine"]:
    hid = f"h12_ck_{tx}"
    hyps.append({
        "id": hid,
        "text": f"complex_karyotype-positive patients have a relatively lower objective_response benefit from {tx} than complex_karyotype-negative patients (negative interaction).",
        "kind": "novel",
    })
    rr = r[tx]
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit('objective_response ~ complex_karyotype * {tx}', df)",
        "result_summary": rr["summary"],
        "p_value": hf(rr["interaction_p"]),
        "effect_estimate": hf(rr["interaction_beta"]),
        "significant": hf(rr["interaction_p"]) < 0.05,
    })
iters.append({"index": 12, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 13 — Sociodemographic ----
r = R["iter13_sociodemographic"]
hyps, analyses = [], []
hyps.append({"id": "h13_sex", "text": "Female patients (sex_female=1) have a different objective_response rate than males.", "kind": "novel"})
analyses.append({
    "hypothesis_ids": ["h13_sex"],
    "code": "chi2_contingency by sex_female",
    "result_summary": r["sex_female"]["summary"],
    "p_value": hf(r["sex_female"]["p_value"]),
    "effect_estimate": hf(r["sex_female"]["effect_estimate"]),
    "significant": hf(r["sex_female"]["p_value"]) < 0.05,
})
hyps.append({"id": "h13_rural", "text": "Rural-residing patients (rural_residence=1) have a lower objective_response rate than urban patients.", "kind": "novel"})
analyses.append({
    "hypothesis_ids": ["h13_rural"],
    "code": "chi2_contingency by rural_residence",
    "result_summary": r["rural_residence"]["summary"],
    "p_value": hf(r["rural_residence"]["p_value"]),
    "effect_estimate": hf(r["rural_residence"]["effect_estimate"]),
    "significant": hf(r["rural_residence"]["p_value"]) < 0.05,
})
hyps.append({"id": "h13_edu", "text": "Higher education_years is associated with higher objective_response rate.", "kind": "novel"})
analyses.append({
    "hypothesis_ids": ["h13_edu"],
    "code": "logit('objective_response ~ education_years', df)",
    "result_summary": r["education_years"]["summary"],
    "p_value": hf(r["education_years"]["p_value"]),
    "effect_estimate": hf(r["education_years"]["beta"]),
    "significant": hf(r["education_years"]["p_value"]) < 0.05,
})
hyps.append({"id": "h13_smk", "text": "Higher smoking_pack_years is associated with lower objective_response rate.", "kind": "novel"})
analyses.append({
    "hypothesis_ids": ["h13_smk"],
    "code": "logit('objective_response ~ smoking_pack_years', df)",
    "result_summary": r["smoking_pack_years"]["summary"],
    "p_value": hf(r["smoking_pack_years"]["p_value"]),
    "effect_estimate": hf(r["smoking_pack_years"]["beta"]),
    "significant": hf(r["smoking_pack_years"]["p_value"]) < 0.05,
})
hyps.append({"id": "h13_race", "text": "Objective_response rate differs across race_ethnicity categories.", "kind": "novel"})
analyses.append({
    "hypothesis_ids": ["h13_race"],
    "code": "chi2_contingency on objective_response × race_ethnicity",
    "result_summary": r["race_ethnicity"]["summary"],
    "p_value": hf(r["race_ethnicity"]["p_value"]),
    "effect_estimate": None,
    "significant": hf(r["race_ethnicity"]["p_value"]) < 0.05,
})
hyps.append({"id": "h13_ins", "text": "Objective_response rate differs across insurance_type categories.", "kind": "novel"})
analyses.append({
    "hypothesis_ids": ["h13_ins"],
    "code": "chi2_contingency on objective_response × insurance_type",
    "result_summary": r["insurance_type"]["summary"],
    "p_value": hf(r["insurance_type"]["p_value"]),
    "effect_estimate": None,
    "significant": hf(r["insurance_type"]["p_value"]) < 0.05,
})
iters.append({"index": 13, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 14 — unfit × venaza ----
r = R["iter14_unfit_x_venaza"]
hid = "h14_unfit_venaza"
hyps = [{
    "id": hid,
    "text": "unfit_for_intensive-positive patients have a greater objective_response benefit from treatment_venetoclax_azacitidine than fit patients (positive interaction).",
    "kind": "novel",
}]
analyses = [{
    "hypothesis_ids": [hid],
    "code": "logit('objective_response ~ unfit_for_intensive * treatment_venetoclax_azacitidine', df)",
    "result_summary": r["summary"],
    "p_value": hf(r["interaction_p"]),
    "effect_estimate": hf(r["interaction_beta"]),
    "significant": hf(r["interaction_p"]) < 0.05,
}]
iters.append({"index": 14, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 15 — Comorbidities ----
r = R["iter15_comorbidities"]
hyps, analyses = [], []
for c, rr in r.items():
    hid = f"h15_{c}"
    hyps.append({
        "id": hid,
        "text": f"Patients with {c}=1 have a different objective_response rate than those without.",
        "kind": "novel",
    })
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2_contingency on objective_response by {c}",
        "result_summary": rr["summary"],
        "p_value": hf(rr["p_value"]),
        "effect_estimate": hf(rr["effect_estimate"]),
        "significant": hf(rr["p_value"]) < 0.05,
    })
iters.append({"index": 15, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 16 — SNPs ----
r = R["iter16_snps"]
hyps, analyses = [], []
for c, rr in r.items():
    hid = f"h16_{c}"
    hyps.append({
        "id": hid,
        "text": f"Carriers of the variant allele at {c} have a different objective_response rate than non-carriers.",
        "kind": "novel",
    })
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2_contingency or logit on objective_response by {c}",
        "result_summary": rr["summary"],
        "p_value": hf(rr["p_value"]),
        "effect_estimate": hf(rr.get("effect_estimate", rr.get("beta"))),
        "significant": hf(rr["p_value"]) < 0.05,
    })
iters.append({"index": 16, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 17 — ECOG × 7+3 ----
r = R["iter17_ecog_x_7plus3"]
hid = "h17_ecog_7plus3"
hyps = [{
    "id": hid,
    "text": "Higher ecog_ps reduces the objective_response benefit of treatment_7plus3 (negative interaction; intensive therapy works less well in poor PS patients).",
    "kind": "novel",
}]
analyses = [{
    "hypothesis_ids": [hid],
    "code": "logit('objective_response ~ ecog_ps * treatment_7plus3', df)",
    "result_summary": r["summary"],
    "p_value": hf(r["interaction_p"]),
    "effect_estimate": hf(r["interaction_beta"]),
    "significant": hf(r["interaction_p"]) < 0.05,
}]
iters.append({"index": 17, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 18 — risk groups ----
r = R["iter18_risk_groups"]
hyps, analyses = [], []
hyps.append({"id": "h18_fav", "text": "Patients in the favorable risk group (NPM1-positive AND FLT3-ITD-negative) have a higher objective_response rate than non-favorable patients.", "kind": "novel"})
analyses.append({
    "hypothesis_ids": ["h18_fav"],
    "code": "fav_risk = (npm1_mutation==1) & (flt3_itd==0); chi2_contingency",
    "result_summary": r["favorable"]["summary"],
    "p_value": hf(r["favorable"]["p_value"]),
    "effect_estimate": hf(r["favorable"]["effect_estimate"]),
    "significant": hf(r["favorable"]["p_value"]) < 0.05,
})
hyps.append({"id": "h18_adv", "text": "Patients in the adverse risk group (TP53-positive OR complex_karyotype) have a lower objective_response rate than non-adverse patients.", "kind": "novel"})
analyses.append({
    "hypothesis_ids": ["h18_adv"],
    "code": "adv_risk = (tp53_mutation==1) | (complex_karyotype==1); chi2_contingency",
    "result_summary": r["adverse"]["summary"],
    "p_value": hf(r["adverse"]["p_value"]),
    "effect_estimate": hf(r["adverse"]["effect_estimate"]),
    "significant": hf(r["adverse"]["p_value"]) < 0.05,
})
iters.append({"index": 18, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 19 — Multivariable ----
r = R["iter19_multivariable"]
hyps, analyses = [], []
hyps.append({"id": "h19_mv", "text": "Adjusting for confounders in a multivariable logistic regression, ECOG performance status remains the dominant predictor of objective_response (negative coefficient).", "kind": "refined"})
hyps.append({"id": "h19_mv_idh1", "text": "After multivariable adjustment, idh1_mutation remains a positive independent predictor of objective_response.", "kind": "refined"})
hyps.append({"id": "h19_mv_alb", "text": "After multivariable adjustment, higher albumin_g_dl remains an independent positive predictor of objective_response.", "kind": "refined"})
hyps.append({"id": "h19_mv_blast", "text": "After multivariable adjustment, higher blast_pct_marrow remains an independent negative predictor of objective_response.", "kind": "refined"})
coefs = r["coefficients"]
analyses.append({
    "hypothesis_ids": ["h19_mv"],
    "code": "logit on full model; report ecog_ps",
    "result_summary": f"ecog_ps adj beta={coefs['ecog_ps']['beta']:+.4f} (OR={coefs['ecog_ps']['OR']:.3f}, p={coefs['ecog_ps']['p_value']:.3g}); dominant signal in adjusted model.",
    "p_value": hf(coefs["ecog_ps"]["p_value"]),
    "effect_estimate": hf(coefs["ecog_ps"]["beta"]),
    "significant": hf(coefs["ecog_ps"]["p_value"]) < 0.05,
})
analyses.append({
    "hypothesis_ids": ["h19_mv_idh1"],
    "code": "logit on full model; report idh1_mutation",
    "result_summary": f"idh1_mutation adj beta={coefs['idh1_mutation']['beta']:+.4f} (OR={coefs['idh1_mutation']['OR']:.3f}, p={coefs['idh1_mutation']['p_value']:.3g})",
    "p_value": hf(coefs["idh1_mutation"]["p_value"]),
    "effect_estimate": hf(coefs["idh1_mutation"]["beta"]),
    "significant": hf(coefs["idh1_mutation"]["p_value"]) < 0.05,
})
analyses.append({
    "hypothesis_ids": ["h19_mv_alb"],
    "code": "logit on full model; report albumin_g_dl",
    "result_summary": f"albumin_g_dl adj beta={coefs['albumin_g_dl']['beta']:+.4f} (OR/unit={coefs['albumin_g_dl']['OR']:.3f}, p={coefs['albumin_g_dl']['p_value']:.3g})",
    "p_value": hf(coefs["albumin_g_dl"]["p_value"]),
    "effect_estimate": hf(coefs["albumin_g_dl"]["beta"]),
    "significant": hf(coefs["albumin_g_dl"]["p_value"]) < 0.05,
})
analyses.append({
    "hypothesis_ids": ["h19_mv_blast"],
    "code": "logit on full model; report blast_pct_marrow",
    "result_summary": f"blast_pct_marrow adj beta={coefs['blast_pct_marrow']['beta']:+.5f} (OR/unit={coefs['blast_pct_marrow']['OR']:.4f}, p={coefs['blast_pct_marrow']['p_value']:.3g})",
    "p_value": hf(coefs["blast_pct_marrow"]["p_value"]),
    "effect_estimate": hf(coefs["blast_pct_marrow"]["beta"]),
    "significant": hf(coefs["blast_pct_marrow"]["p_value"]) < 0.05,
})
iters.append({"index": 19, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 20 — Symptoms ----
r = R["iter20_symptoms"]
hyps, analyses = [], []
mapping = {
    "fatigue_grade": "Higher fatigue_grade is associated with lower objective_response.",
    "pain_nrs": "Higher pain_nrs is associated with lower objective_response.",
    "dyspnea_grade": "Higher dyspnea_grade is associated with lower objective_response.",
    "cough_grade": "Higher cough_grade is associated with lower objective_response.",
    "appetite_loss_grade": "Higher appetite_loss_grade is associated with lower objective_response.",
    "weight_loss_pct_6mo": "Higher weight_loss_pct_6mo is associated with lower objective_response.",
}
for col, txt in mapping.items():
    hid = f"h20_{col}"
    hyps.append({"id": hid, "text": txt, "kind": "novel"})
    rr = r[col]
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit('objective_response ~ {col}', df)",
        "result_summary": rr["summary"],
        "p_value": hf(rr["p_value"]),
        "effect_estimate": hf(rr["beta"]),
        "significant": hf(rr["p_value"]) < 0.05,
    })
iters.append({"index": 20, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 21 — Blood counts ----
r = R["iter21_blood_counts"]
hyps, analyses = [], []
mapping = {
    "hemoglobin_g_dl": "Higher hemoglobin_g_dl is associated with higher objective_response.",
    "platelets_k_ul": "Higher platelets_k_ul is associated with higher objective_response.",
    "anc_k_ul": "Higher anc_k_ul is associated with higher objective_response.",
    "alc_k_ul": "Higher alc_k_ul is associated with higher objective_response.",
}
for col, txt in mapping.items():
    hid = f"h21_{col}"
    hyps.append({"id": hid, "text": txt, "kind": "novel"})
    rr = r[col]
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit('objective_response ~ {col}', df)",
        "result_summary": rr["summary"],
        "p_value": hf(rr["p_value"]),
        "effect_estimate": hf(rr["beta"]),
        "significant": hf(rr["p_value"]) < 0.05,
    })
iters.append({"index": 21, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 22 — FLT3-TKD × FLT3 inhibitors ----
r = R["iter22_flt3tkd_x_flt3i"]
hyps, analyses = [], []
for tx in ["treatment_midostaurin", "treatment_gilteritinib"]:
    hid = f"h22_tkd_{tx}"
    hyps.append({
        "id": hid,
        "text": f"flt3_tkd-positive patients have a greater objective_response benefit from {tx} than FLT3-TKD-negative patients (positive interaction).",
        "kind": "novel",
    })
    rr = r[tx]
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit('objective_response ~ flt3_tkd * {tx}', df)",
        "result_summary": rr["summary"],
        "p_value": hf(rr["interaction_p"]),
        "effect_estimate": hf(rr["interaction_beta"]),
        "significant": hf(rr["interaction_p"]) < 0.05,
    })
iters.append({"index": 22, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 23 — Age × 7+3 ----
r = R["iter23_age_x_7plus3"]
hid = "h23_age_7plus3"
hyps = [{
    "id": hid,
    "text": "Older patients (higher age_years) gain less objective_response benefit from treatment_7plus3 than younger patients (negative interaction).",
    "kind": "novel",
}]
analyses = [{
    "hypothesis_ids": [hid],
    "code": "logit('objective_response ~ age_years * treatment_7plus3', df)",
    "result_summary": r["summary"],
    "p_value": hf(r["interaction_p"]),
    "effect_estimate": hf(r["interaction_beta"]),
    "significant": hf(r["interaction_p"]) < 0.05,
}]
iters.append({"index": 23, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 24 — TP53 × venaza ----
r = R["iter24_tp53_x_venaza"]
hid = "h24_tp53_venaza"
hyps = [{
    "id": hid,
    "text": "tp53_mutation-positive patients gain less objective_response benefit from treatment_venetoclax_azacitidine than TP53-negative patients (negative interaction).",
    "kind": "refined",
}]
analyses = [{
    "hypothesis_ids": [hid],
    "code": "logit('objective_response ~ tp53_mutation * treatment_venetoclax_azacitidine', df)",
    "result_summary": r["summary"],
    "p_value": hf(r["interaction_p"]),
    "effect_estimate": hf(r["interaction_beta"]),
    "significant": hf(r["interaction_p"]) < 0.05,
}]
iters.append({"index": 24, "proposed_hypotheses": hyps, "analyses": analyses})

# ---- ITER 25 — Matched-therapy summary ----
r = R["iter25_matched_therapy"]
hyps, analyses = [], []
hyps.append({
    "id": "h25_matched_unadj",
    "text": "Patients receiving biomarker-matched targeted therapy (FLT3-mutated on midostaurin/gilteritinib, IDH1-mutated on ivosidenib, IDH2-mutated on enasidenib) have a higher objective_response rate than unmatched patients.",
    "kind": "refined",
})
analyses.append({
    "hypothesis_ids": ["h25_matched_unadj"],
    "code": "matched_therapy indicator + chi2_contingency",
    "result_summary": r["unadjusted"]["summary"],
    "p_value": hf(r["unadjusted"]["p_value"]),
    "effect_estimate": hf(r["unadjusted"]["effect_estimate"]),
    "significant": hf(r["unadjusted"]["p_value"]) < 0.05,
})
hyps.append({
    "id": "h25_matched_adj",
    "text": "After adjusting for age_years, ecog_ps, tp53_mutation, complex_karyotype, secondary_aml, albumin_g_dl, and nlr, biomarker-matched targeted therapy is independently associated with higher objective_response.",
    "kind": "refined",
})
analyses.append({
    "hypothesis_ids": ["h25_matched_adj"],
    "code": "logit('objective_response ~ matched_therapy + age_years + ecog_ps + tp53_mutation + complex_karyotype + secondary_aml + albumin_g_dl + nlr', df)",
    "result_summary": r["adjusted"]["summary"],
    "p_value": hf(r["adjusted"]["p_value"]),
    "effect_estimate": hf(r["adjusted"]["beta"]),
    "significant": hf(r["adjusted"]["p_value"]) < 0.05,
})
iters.append({"index": 25, "proposed_hypotheses": hyps, "analyses": analyses})


transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-manual@2026-04",
    "max_iterations": 25,
    "iterations": iters,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

print(f"Wrote transcript.json with {len(iters)} iterations.")
total_h = sum(len(it["proposed_hypotheses"]) for it in iters)
total_a = sum(len(it["analyses"]) for it in iters)
print(f"Total hypotheses: {total_h}, total analyses: {total_a}")
