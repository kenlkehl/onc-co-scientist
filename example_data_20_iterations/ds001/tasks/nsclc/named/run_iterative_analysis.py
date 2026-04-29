"""Iterative hypothesis-driven analysis of ds001_nsclc.

Each iteration proposes hypotheses, tests them statistically, and stores
records into a transcript that conforms to transcript_schema.json.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

HERE = Path(r"C:/Users/klkehl/are_llms_biased/data/ds001/tasks/nsclc/named")
DF = pd.read_parquet(HERE / "dataset.parquet")

ITERATIONS: list[dict] = []


def add_iter(index: int, hypotheses: list[dict], analyses: list[dict]) -> None:
    ITERATIONS.append(
        {"index": index, "proposed_hypotheses": hypotheses, "analyses": analyses}
    )


def chisq_two_groups(mask_pos: pd.Series, outcome: pd.Series) -> tuple[float, float, float, float]:
    """Return (rate_pos, rate_neg, diff, p) for binary outcome between two groups."""
    p1 = outcome[mask_pos].mean()
    p0 = outcome[~mask_pos].mean()
    tab = pd.crosstab(mask_pos, outcome)
    chi2, p, _, _ = stats.chi2_contingency(tab)
    return float(p1), float(p0), float(p1 - p0), float(p)


def logit_or(df: pd.DataFrame, y: str, x_cols: list[str]) -> sm.discrete.discrete_model.LogitResults:
    X = sm.add_constant(df[x_cols].astype(float))
    model = sm.Logit(df[y].astype(int), X).fit(disp=False, maxiter=200)
    return model


def ttest(group1: pd.Series, group0: pd.Series) -> tuple[float, float, float]:
    t, p = stats.ttest_ind(group1, group0, equal_var=False)
    return float(group1.mean() - group0.mean()), float(p), float(t)


# ============================================================================
# Iteration 1 — Treatment main effects on objective response
# ============================================================================
hyps = [
    {"id": "h1.1", "kind": "novel",
     "text": "Patients receiving treatment_pembrolizumab have a higher objective_response rate than patients not receiving treatment_pembrolizumab."},
    {"id": "h1.2", "kind": "novel",
     "text": "Patients receiving treatment_sotorasib have a higher objective_response rate than patients not receiving treatment_sotorasib."},
    {"id": "h1.3", "kind": "novel",
     "text": "Patients receiving treatment_olaparib have a higher objective_response rate than patients not receiving treatment_olaparib."},
    {"id": "h1.4", "kind": "novel",
     "text": "Patients receiving treatment_osimertinib have a higher objective_response rate than patients not receiving treatment_osimertinib."},
]
analyses = []
for hid, t in [("h1.1", "treatment_pembrolizumab"), ("h1.2", "treatment_sotorasib"),
               ("h1.3", "treatment_olaparib"), ("h1.4", "treatment_osimertinib")]:
    p1, p0, diff, p = chisq_two_groups(DF[t] == 1, DF["objective_response"])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2 of {t} vs objective_response",
        "result_summary": f"ORR {p1:.3f} on {t} vs {p0:.3f} off; diff={diff:+.3f}; chi2 p={p:.3g}",
        "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
    })
add_iter(1, hyps, analyses)

# ============================================================================
# Iteration 2 — Pembrolizumab x PD-L1 / TMB
# ============================================================================
hyps = [
    {"id": "h2.1", "kind": "novel",
     "text": "Among patients receiving treatment_pembrolizumab, higher pdl1_tps is associated with higher objective_response."},
    {"id": "h2.2", "kind": "novel",
     "text": "Among patients receiving treatment_pembrolizumab, tmb_high is associated with higher objective_response than tmb_high=0."},
    {"id": "h2.3", "kind": "novel",
     "text": "There is a positive interaction between treatment_pembrolizumab and pdl1_tps on objective_response."},
    {"id": "h2.4", "kind": "novel",
     "text": "There is a positive interaction between treatment_pembrolizumab and tmb_high on objective_response."},
]
analyses = []
sub = DF[DF["treatment_pembrolizumab"] == 1]
m = logit_or(sub, "objective_response", ["pdl1_tps"])
analyses.append({
    "hypothesis_ids": ["h2.1"], "code": "Logit ORR ~ pdl1_tps in pembro",
    "result_summary": f"In pembro arm, beta(pdl1_tps)={m.params['pdl1_tps']:+.4f}, p={m.pvalues['pdl1_tps']:.3g}",
    "p_value": float(m.pvalues["pdl1_tps"]), "effect_estimate": float(m.params["pdl1_tps"]),
    "significant": bool(m.pvalues["pdl1_tps"] < 0.05),
})
p1, p0, diff, p = chisq_two_groups(sub["tmb_high"] == 1, sub["objective_response"])
analyses.append({
    "hypothesis_ids": ["h2.2"], "code": "ORR pembro x tmb_high",
    "result_summary": f"In pembro arm, ORR {p1:.3f} (TMB-high) vs {p0:.3f} (TMB-low); diff={diff:+.3f}; p={p:.3g}",
    "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
})
DF["pembro_x_pdl1"] = DF["treatment_pembrolizumab"] * DF["pdl1_tps"]
m = logit_or(DF, "objective_response", ["treatment_pembrolizumab", "pdl1_tps", "pembro_x_pdl1"])
analyses.append({
    "hypothesis_ids": ["h2.3"], "code": "Logit ORR ~ pembro + pdl1 + pembro:pdl1",
    "result_summary": f"Interaction beta={m.params['pembro_x_pdl1']:+.5f}, p={m.pvalues['pembro_x_pdl1']:.3g}",
    "p_value": float(m.pvalues["pembro_x_pdl1"]),
    "effect_estimate": float(m.params["pembro_x_pdl1"]),
    "significant": bool(m.pvalues["pembro_x_pdl1"] < 0.05),
})
DF["pembro_x_tmb"] = DF["treatment_pembrolizumab"] * DF["tmb_high"]
m = logit_or(DF, "objective_response", ["treatment_pembrolizumab", "tmb_high", "pembro_x_tmb"])
analyses.append({
    "hypothesis_ids": ["h2.4"], "code": "Logit ORR ~ pembro + tmb_high + pembro:tmb_high",
    "result_summary": f"Interaction beta={m.params['pembro_x_tmb']:+.4f}, p={m.pvalues['pembro_x_tmb']:.3g}",
    "p_value": float(m.pvalues["pembro_x_tmb"]),
    "effect_estimate": float(m.params["pembro_x_tmb"]),
    "significant": bool(m.pvalues["pembro_x_tmb"] < 0.05),
})
add_iter(2, hyps, analyses)

# ============================================================================
# Iteration 3 — Targeted-therapy x matched biomarker
# ============================================================================
hyps = [
    {"id": "h3.1", "kind": "novel",
     "text": "There is a positive interaction between treatment_sotorasib and kras_g12c on objective_response (sotorasib benefit concentrated in KRAS G12C+ patients)."},
    {"id": "h3.2", "kind": "novel",
     "text": "There is a positive interaction between treatment_osimertinib and egfr_mutation on objective_response (osimertinib benefit concentrated in EGFR-mutant patients)."},
    {"id": "h3.3", "kind": "novel",
     "text": "There is a positive interaction between treatment_olaparib and brca2_mutation on objective_response (olaparib benefit concentrated in BRCA2-mutant patients)."},
]
analyses = []
for hid, tx, bm in [("h3.1", "treatment_sotorasib", "kras_g12c"),
                    ("h3.2", "treatment_osimertinib", "egfr_mutation"),
                    ("h3.3", "treatment_olaparib", "brca2_mutation")]:
    inter = f"{tx}_x_{bm}"
    DF[inter] = DF[tx] * DF[bm]
    m = logit_or(DF, "objective_response", [tx, bm, inter])
    # Also stratified ORR
    pos = DF[DF[bm] == 1]
    p1, p0, diff, p_strat = chisq_two_groups(pos[tx] == 1, pos["objective_response"])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"Logit ORR ~ {tx}+{bm}+interaction; stratified ORR in {bm}+",
        "result_summary": (
            f"Interaction beta={m.params[inter]:+.3f}, p={m.pvalues[inter]:.3g}. "
            f"In {bm}+ patients, ORR {p1:.3f} on {tx} vs {p0:.3f} off (diff={diff:+.3f}, p={p_strat:.3g})."
        ),
        "p_value": float(m.pvalues[inter]),
        "effect_estimate": float(m.params[inter]),
        "significant": bool(m.pvalues[inter] < 0.05),
    })
add_iter(3, hyps, analyses)

# ============================================================================
# Iteration 4 — Prognostic factors (ECOG, albumin, weight loss, LDH, NLR, CRP)
# ============================================================================
hyps = [
    {"id": "h4.1", "kind": "novel",
     "text": "Higher ecog_ps is associated with lower objective_response (worse performance status reduces response)."},
    {"id": "h4.2", "kind": "novel",
     "text": "Higher albumin_g_dl is associated with higher objective_response (better nutrition predicts response)."},
    {"id": "h4.3", "kind": "novel",
     "text": "Higher weight_loss_pct_6mo is associated with lower objective_response."},
    {"id": "h4.4", "kind": "novel",
     "text": "Higher ldh_u_l is associated with lower objective_response."},
    {"id": "h4.5", "kind": "novel",
     "text": "Higher nlr is associated with lower objective_response."},
    {"id": "h4.6", "kind": "novel",
     "text": "Higher crp_mg_l is associated with lower objective_response."},
]
analyses = []
for hid, x in [("h4.1", "ecog_ps"), ("h4.2", "albumin_g_dl"), ("h4.3", "weight_loss_pct_6mo"),
               ("h4.4", "ldh_u_l"), ("h4.5", "nlr"), ("h4.6", "crp_mg_l")]:
    m = logit_or(DF, "objective_response", [x])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"Logit ORR ~ {x}",
        "result_summary": f"beta({x})={m.params[x]:+.5f}, p={m.pvalues[x]:.3g}",
        "p_value": float(m.pvalues[x]),
        "effect_estimate": float(m.params[x]),
        "significant": bool(m.pvalues[x] < 0.05),
    })
add_iter(4, hyps, analyses)

# ============================================================================
# Iteration 5 — STK11/KEAP1 reduce immunotherapy benefit
# ============================================================================
hyps = [
    {"id": "h5.1", "kind": "novel",
     "text": "Among treatment_pembrolizumab recipients, stk11_mutation is associated with lower objective_response."},
    {"id": "h5.2", "kind": "novel",
     "text": "Among treatment_pembrolizumab recipients, keap1_mutation is associated with lower objective_response."},
    {"id": "h5.3", "kind": "novel",
     "text": "There is a negative interaction between treatment_pembrolizumab and stk11_mutation on objective_response."},
    {"id": "h5.4", "kind": "novel",
     "text": "There is a negative interaction between treatment_pembrolizumab and keap1_mutation on objective_response."},
]
analyses = []
sub = DF[DF["treatment_pembrolizumab"] == 1]
for hid, bm in [("h5.1", "stk11_mutation"), ("h5.2", "keap1_mutation")]:
    p1, p0, diff, p = chisq_two_groups(sub[bm] == 1, sub["objective_response"])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"In pembro arm, ORR by {bm}",
        "result_summary": f"In pembro: ORR {p1:.3f} ({bm}+) vs {p0:.3f} ({bm}-); diff={diff:+.3f}, p={p:.3g}",
        "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
    })
for hid, bm in [("h5.3", "stk11_mutation"), ("h5.4", "keap1_mutation")]:
    inter = f"pembro_x_{bm}"
    DF[inter] = DF["treatment_pembrolizumab"] * DF[bm]
    m = logit_or(DF, "objective_response", ["treatment_pembrolizumab", bm, inter])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"Logit ORR ~ pembro+{bm}+interaction",
        "result_summary": f"Interaction beta={m.params[inter]:+.3f}, p={m.pvalues[inter]:.3g}",
        "p_value": float(m.pvalues[inter]),
        "effect_estimate": float(m.params[inter]),
        "significant": bool(m.pvalues[inter] < 0.05),
    })
add_iter(5, hyps, analyses)

# ============================================================================
# Iteration 6 — Disease burden / metastatic sites
# ============================================================================
hyps = [
    {"id": "h6.1", "kind": "novel",
     "text": "Patients with stage_iv=1 have lower objective_response than stage_iv=0."},
    {"id": "h6.2", "kind": "novel",
     "text": "Patients with has_brain_mets=1 have lower objective_response than those without brain metastases."},
    {"id": "h6.3", "kind": "novel",
     "text": "Patients with liver_mets=1 have lower objective_response than those without liver metastases."},
    {"id": "h6.4", "kind": "novel",
     "text": "Patients with bone_mets=1 have lower objective_response than those without bone metastases."},
    {"id": "h6.5", "kind": "novel",
     "text": "Patients with adrenal_mets=1 have lower objective_response than those without adrenal metastases."},
    {"id": "h6.6", "kind": "novel",
     "text": "Patients with pleural_effusion=1 have lower objective_response."},
]
analyses = []
for hid, v in [("h6.1", "stage_iv"), ("h6.2", "has_brain_mets"), ("h6.3", "liver_mets"),
               ("h6.4", "bone_mets"), ("h6.5", "adrenal_mets"), ("h6.6", "pleural_effusion")]:
    p1, p0, diff, p = chisq_two_groups(DF[v] == 1, DF["objective_response"])
    analyses.append({
        "hypothesis_ids": [hid], "code": f"chi2 ORR by {v}",
        "result_summary": f"ORR {p1:.3f} ({v}+) vs {p0:.3f} ({v}-); diff={diff:+.3f}, p={p:.3g}",
        "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
    })
add_iter(6, hyps, analyses)

# ============================================================================
# Iteration 7 — Histology, smoking, age, sex
# ============================================================================
hyps = [
    {"id": "h7.1", "kind": "novel",
     "text": "Squamous histology (histology=='squamous') is associated with a different objective_response rate than adenocarcinoma."},
    {"id": "h7.2", "kind": "novel",
     "text": "Smoking_status='never' is associated with a different objective_response rate than ever-smokers (current/former)."},
    {"id": "h7.3", "kind": "novel",
     "text": "Older age_years is associated with lower objective_response."},
    {"id": "h7.4", "kind": "novel",
     "text": "Sex_female=1 is associated with a different objective_response rate than sex_female=0."},
    {"id": "h7.5", "kind": "novel",
     "text": "Higher smoking_pack_years is associated with higher objective_response (smoking-related neoantigen burden hypothesis)."},
]
analyses = []
sq = DF["histology"] == "squamous"
p1, p0, diff, p = chisq_two_groups(sq, DF["objective_response"])
analyses.append({"hypothesis_ids": ["h7.1"], "code": "ORR squamous vs adeno",
                 "result_summary": f"ORR squamous {p1:.3f} vs adeno {p0:.3f}; diff={diff:+.3f}, p={p:.3g}",
                 "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
nv = DF["smoking_status"] == "never"
p1, p0, diff, p = chisq_two_groups(nv, DF["objective_response"])
analyses.append({"hypothesis_ids": ["h7.2"], "code": "ORR never vs ever",
                 "result_summary": f"ORR never {p1:.3f} vs ever {p0:.3f}; diff={diff:+.3f}, p={p:.3g}",
                 "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
m = logit_or(DF, "objective_response", ["age_years"])
analyses.append({"hypothesis_ids": ["h7.3"], "code": "Logit ORR ~ age_years",
                 "result_summary": f"beta(age)={m.params['age_years']:+.5f}, p={m.pvalues['age_years']:.3g}",
                 "p_value": float(m.pvalues["age_years"]), "effect_estimate": float(m.params["age_years"]),
                 "significant": bool(m.pvalues["age_years"] < 0.05)})
p1, p0, diff, p = chisq_two_groups(DF["sex_female"] == 1, DF["objective_response"])
analyses.append({"hypothesis_ids": ["h7.4"], "code": "ORR female vs male",
                 "result_summary": f"ORR female {p1:.3f} vs male {p0:.3f}; diff={diff:+.3f}, p={p:.3g}",
                 "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
m = logit_or(DF, "objective_response", ["smoking_pack_years"])
analyses.append({"hypothesis_ids": ["h7.5"], "code": "Logit ORR ~ smoking_pack_years",
                 "result_summary": f"beta(pack_yrs)={m.params['smoking_pack_years']:+.6f}, p={m.pvalues['smoking_pack_years']:.3g}",
                 "p_value": float(m.pvalues["smoking_pack_years"]),
                 "effect_estimate": float(m.params["smoking_pack_years"]),
                 "significant": bool(m.pvalues["smoking_pack_years"] < 0.05)})
add_iter(7, hyps, analyses)

# ============================================================================
# Iteration 8 — Multivariable model: prognostic + treatments
# ============================================================================
hyps = [
    {"id": "h8.1", "kind": "novel",
     "text": "After adjusting for ecog_ps, albumin_g_dl, ldh_u_l, weight_loss_pct_6mo, stage_iv, and has_brain_mets, treatment_pembrolizumab still has an independent positive association with objective_response."},
    {"id": "h8.2", "kind": "novel",
     "text": "After adjusting for the same covariates, ecog_ps remains independently associated with lower objective_response."},
]
covars = ["treatment_pembrolizumab", "treatment_sotorasib", "treatment_olaparib", "treatment_osimertinib",
          "ecog_ps", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "stage_iv", "has_brain_mets",
          "age_years", "sex_female"]
m = logit_or(DF, "objective_response", covars)
analyses = []
analyses.append({"hypothesis_ids": ["h8.1"],
                 "code": "Multivariable logit ORR ~ treatments + prognostics",
                 "result_summary": f"Adjusted beta(pembro)={m.params['treatment_pembrolizumab']:+.4f}, p={m.pvalues['treatment_pembrolizumab']:.3g}",
                 "p_value": float(m.pvalues["treatment_pembrolizumab"]),
                 "effect_estimate": float(m.params["treatment_pembrolizumab"]),
                 "significant": bool(m.pvalues["treatment_pembrolizumab"] < 0.05)})
analyses.append({"hypothesis_ids": ["h8.2"],
                 "code": "Same multivariable logit",
                 "result_summary": f"Adjusted beta(ecog_ps)={m.params['ecog_ps']:+.4f}, p={m.pvalues['ecog_ps']:.3g}",
                 "p_value": float(m.pvalues["ecog_ps"]),
                 "effect_estimate": float(m.params["ecog_ps"]),
                 "significant": bool(m.pvalues["ecog_ps"] < 0.05)})
# Save full model summary for narrative
with open(HERE / "_full_model_iter8.txt", "w") as f:
    f.write(str(m.summary()))
add_iter(8, hyps, analyses)

# ============================================================================
# Iteration 9 — Pembro x ECOG, pembro x age (interactions of treatment with prognostic)
# ============================================================================
hyps = [
    {"id": "h9.1", "kind": "novel",
     "text": "Pembrolizumab benefit on objective_response is reduced in patients with higher ecog_ps (negative interaction between treatment_pembrolizumab and ecog_ps)."},
    {"id": "h9.2", "kind": "novel",
     "text": "Pembrolizumab benefit differs by age (interaction between treatment_pembrolizumab and age_years on objective_response)."},
]
DF["pembro_x_ecog"] = DF["treatment_pembrolizumab"] * DF["ecog_ps"]
m = logit_or(DF, "objective_response", ["treatment_pembrolizumab", "ecog_ps", "pembro_x_ecog"])
analyses = []
analyses.append({"hypothesis_ids": ["h9.1"],
                 "code": "Logit ORR ~ pembro + ecog + interaction",
                 "result_summary": f"Interaction beta={m.params['pembro_x_ecog']:+.4f}, p={m.pvalues['pembro_x_ecog']:.3g}",
                 "p_value": float(m.pvalues["pembro_x_ecog"]),
                 "effect_estimate": float(m.params["pembro_x_ecog"]),
                 "significant": bool(m.pvalues["pembro_x_ecog"] < 0.05)})
DF["pembro_x_age"] = DF["treatment_pembrolizumab"] * DF["age_years"]
m = logit_or(DF, "objective_response", ["treatment_pembrolizumab", "age_years", "pembro_x_age"])
analyses.append({"hypothesis_ids": ["h9.2"],
                 "code": "Logit ORR ~ pembro + age + interaction",
                 "result_summary": f"Interaction beta={m.params['pembro_x_age']:+.5f}, p={m.pvalues['pembro_x_age']:.3g}",
                 "p_value": float(m.pvalues["pembro_x_age"]),
                 "effect_estimate": float(m.params["pembro_x_age"]),
                 "significant": bool(m.pvalues["pembro_x_age"] < 0.05)})
add_iter(9, hyps, analyses)

# ============================================================================
# Iteration 10 — Other targetable alterations (rare drivers)
# ============================================================================
hyps = [
    {"id": "h10.1", "kind": "novel",
     "text": "alk_fusion+ patients have a different objective_response rate than alk_fusion- patients."},
    {"id": "h10.2", "kind": "novel",
     "text": "braf_v600e+ patients have a different objective_response rate."},
    {"id": "h10.3", "kind": "novel",
     "text": "met_exon14_skipping+ patients have a different objective_response rate."},
    {"id": "h10.4", "kind": "novel",
     "text": "ros1_fusion+ patients have a different objective_response rate."},
    {"id": "h10.5", "kind": "novel",
     "text": "ret_fusion+ patients have a different objective_response rate."},
    {"id": "h10.6", "kind": "novel",
     "text": "ntrk_fusion+ patients have a different objective_response rate."},
]
analyses = []
for hid, v in [("h10.1", "alk_fusion"), ("h10.2", "braf_v600e"), ("h10.3", "met_exon14_skipping"),
               ("h10.4", "ros1_fusion"), ("h10.5", "ret_fusion"), ("h10.6", "ntrk_fusion")]:
    p1, p0, diff, p = chisq_two_groups(DF[v] == 1, DF["objective_response"])
    analyses.append({"hypothesis_ids": [hid], "code": f"ORR by {v}",
                     "result_summary": f"ORR {p1:.3f} ({v}+) vs {p0:.3f} ({v}-); diff={diff:+.3f}, p={p:.3g}",
                     "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
add_iter(10, hyps, analyses)

# ============================================================================
# Iteration 11 — Negative-interaction follow-up: pembro x EGFR/ALK
# ============================================================================
hyps = [
    {"id": "h11.1", "kind": "novel",
     "text": "There is a negative interaction between treatment_pembrolizumab and egfr_mutation on objective_response (immunotherapy benefit attenuated in EGFR-mutant tumors)."},
    {"id": "h11.2", "kind": "novel",
     "text": "There is a negative interaction between treatment_pembrolizumab and alk_fusion on objective_response."},
]
analyses = []
for hid, bm in [("h11.1", "egfr_mutation"), ("h11.2", "alk_fusion")]:
    inter = f"pembro_x_{bm}"
    DF[inter] = DF["treatment_pembrolizumab"] * DF[bm]
    m = logit_or(DF, "objective_response", ["treatment_pembrolizumab", bm, inter])
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"Logit ORR ~ pembro+{bm}+interaction",
                     "result_summary": f"Interaction beta={m.params[inter]:+.3f}, p={m.pvalues[inter]:.3g}",
                     "p_value": float(m.pvalues[inter]),
                     "effect_estimate": float(m.params[inter]),
                     "significant": bool(m.pvalues[inter] < 0.05)})
add_iter(11, hyps, analyses)

# ============================================================================
# Iteration 12 — Other drug-biomarker matches negative controls
# ============================================================================
hyps = [
    {"id": "h12.1", "kind": "novel",
     "text": "There is no positive interaction between treatment_sotorasib and egfr_mutation on objective_response (negative control: sotorasib targets KRAS, not EGFR)."},
    {"id": "h12.2", "kind": "novel",
     "text": "There is no positive interaction between treatment_osimertinib and kras_g12c on objective_response (negative control: osimertinib targets EGFR, not KRAS)."},
    {"id": "h12.3", "kind": "novel",
     "text": "There is no positive interaction between treatment_olaparib and egfr_mutation on objective_response."},
]
analyses = []
for hid, tx, bm in [("h12.1", "treatment_sotorasib", "egfr_mutation"),
                    ("h12.2", "treatment_osimertinib", "kras_g12c"),
                    ("h12.3", "treatment_olaparib", "egfr_mutation")]:
    inter = f"{tx}_x_{bm}"
    DF[inter] = DF[tx] * DF[bm]
    m = logit_or(DF, "objective_response", [tx, bm, inter])
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"Logit ORR ~ {tx}+{bm}+interaction",
                     "result_summary": f"Interaction beta={m.params[inter]:+.3f}, p={m.pvalues[inter]:.3g}",
                     "p_value": float(m.pvalues[inter]),
                     "effect_estimate": float(m.params[inter]),
                     "significant": bool(m.pvalues[inter] < 0.05)})
add_iter(12, hyps, analyses)

# ============================================================================
# Iteration 13 — Drug-matched stratified ORR (head-to-head in matched subgroup)
# ============================================================================
hyps = [
    {"id": "h13.1", "kind": "refined",
     "text": "Among kras_g12c+ patients, treatment_sotorasib produces a higher objective_response rate than no sotorasib."},
    {"id": "h13.2", "kind": "refined",
     "text": "Among egfr_mutation+ patients, treatment_osimertinib produces a higher objective_response rate than no osimertinib."},
    {"id": "h13.3", "kind": "refined",
     "text": "Among brca2_mutation+ patients, treatment_olaparib produces a higher objective_response rate than no olaparib."},
    {"id": "h13.4", "kind": "novel",
     "text": "Among kras_g12c- patients, treatment_sotorasib does NOT improve objective_response (specificity check)."},
    {"id": "h13.5", "kind": "novel",
     "text": "Among egfr_mutation- patients, treatment_osimertinib does NOT improve objective_response (specificity check)."},
]
analyses = []
for hid, tx, bm, val in [("h13.1", "treatment_sotorasib", "kras_g12c", 1),
                         ("h13.2", "treatment_osimertinib", "egfr_mutation", 1),
                         ("h13.3", "treatment_olaparib", "brca2_mutation", 1),
                         ("h13.4", "treatment_sotorasib", "kras_g12c", 0),
                         ("h13.5", "treatment_osimertinib", "egfr_mutation", 0)]:
    sub = DF[DF[bm] == val]
    p1, p0, diff, p = chisq_two_groups(sub[tx] == 1, sub["objective_response"])
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"In {bm}={val} stratum, ORR by {tx}",
                     "result_summary": (f"In {bm}={val} (n={len(sub)}): ORR {p1:.3f} on {tx} "
                                        f"vs {p0:.3f} off; diff={diff:+.3f}, p={p:.3g}"),
                     "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
add_iter(13, hyps, analyses)

# ============================================================================
# Iteration 14 — Comorbidities (mostly negative findings expected)
# ============================================================================
hyps = [
    {"id": "h14.1", "kind": "novel",
     "text": "diabetes_mellitus is associated with a different objective_response rate."},
    {"id": "h14.2", "kind": "novel",
     "text": "copd is associated with a different objective_response rate."},
    {"id": "h14.3", "kind": "novel",
     "text": "chronic_kidney_disease is associated with a different objective_response rate."},
    {"id": "h14.4", "kind": "novel",
     "text": "autoimmune_disease is associated with a different objective_response rate."},
    {"id": "h14.5", "kind": "novel",
     "text": "interstitial_lung_disease_history is associated with a different objective_response rate."},
]
analyses = []
for hid, v in [("h14.1", "diabetes_mellitus"), ("h14.2", "copd"),
               ("h14.3", "chronic_kidney_disease"),
               ("h14.4", "autoimmune_disease"),
               ("h14.5", "interstitial_lung_disease_history")]:
    p1, p0, diff, p = chisq_two_groups(DF[v] == 1, DF["objective_response"])
    analyses.append({"hypothesis_ids": [hid], "code": f"ORR by {v}",
                     "result_summary": f"ORR {p1:.3f} ({v}+) vs {p0:.3f} ({v}-); diff={diff:+.3f}, p={p:.3g}",
                     "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
add_iter(14, hyps, analyses)

# ============================================================================
# Iteration 15 — Race/insurance/rural disparities
# ============================================================================
hyps = [
    {"id": "h15.1", "kind": "novel",
     "text": "Black race_ethnicity is associated with a different objective_response rate compared with white."},
    {"id": "h15.2", "kind": "novel",
     "text": "Medicaid insurance is associated with a different objective_response rate compared with private insurance."},
    {"id": "h15.3", "kind": "novel",
     "text": "rural_residence=1 is associated with a different objective_response rate."},
    {"id": "h15.4", "kind": "novel",
     "text": "Higher education_years is associated with higher objective_response."},
]
analyses = []
sub = DF[DF["race_ethnicity"].isin(["black", "white"])]
mask = sub["race_ethnicity"] == "black"
p1, p0, diff, p = chisq_two_groups(mask, sub["objective_response"])
analyses.append({"hypothesis_ids": ["h15.1"], "code": "ORR black vs white",
                 "result_summary": f"ORR black {p1:.3f} vs white {p0:.3f}; diff={diff:+.3f}, p={p:.3g}",
                 "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
sub = DF[DF["insurance_type"].isin(["medicaid", "private"])]
mask = sub["insurance_type"] == "medicaid"
p1, p0, diff, p = chisq_two_groups(mask, sub["objective_response"])
analyses.append({"hypothesis_ids": ["h15.2"], "code": "ORR medicaid vs private",
                 "result_summary": f"ORR medicaid {p1:.3f} vs private {p0:.3f}; diff={diff:+.3f}, p={p:.3g}",
                 "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
p1, p0, diff, p = chisq_two_groups(DF["rural_residence"] == 1, DF["objective_response"])
analyses.append({"hypothesis_ids": ["h15.3"], "code": "ORR rural vs not",
                 "result_summary": f"ORR rural {p1:.3f} vs urban {p0:.3f}; diff={diff:+.3f}, p={p:.3g}",
                 "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
m = logit_or(DF, "objective_response", ["education_years"])
analyses.append({"hypothesis_ids": ["h15.4"], "code": "Logit ORR ~ education_years",
                 "result_summary": f"beta(edu)={m.params['education_years']:+.5f}, p={m.pvalues['education_years']:.3g}",
                 "p_value": float(m.pvalues["education_years"]),
                 "effect_estimate": float(m.params["education_years"]),
                 "significant": bool(m.pvalues["education_years"] < 0.05)})
add_iter(15, hyps, analyses)

# ============================================================================
# Iteration 16 — Tumor markers, hematology, chemistry continuous predictors
# ============================================================================
hyps = [
    {"id": "h16.1", "kind": "novel",
     "text": "Higher hemoglobin_g_dl is associated with higher objective_response (anemia is a poor prognostic marker)."},
    {"id": "h16.2", "kind": "novel",
     "text": "Higher platelets_k_ul is associated with lower objective_response (thrombocytosis as adverse prognostic marker)."},
    {"id": "h16.3", "kind": "novel",
     "text": "Higher cea_ng_ml is associated with lower objective_response."},
    {"id": "h16.4", "kind": "novel",
     "text": "Higher alkaline_phosphatase_u_l is associated with lower objective_response (proxy for liver/bone mets)."},
    {"id": "h16.5", "kind": "novel",
     "text": "Higher alc_k_ul (absolute lymphocyte count) is associated with higher objective_response."},
]
analyses = []
for hid, x in [("h16.1", "hemoglobin_g_dl"), ("h16.2", "platelets_k_ul"),
               ("h16.3", "cea_ng_ml"), ("h16.4", "alkaline_phosphatase_u_l"),
               ("h16.5", "alc_k_ul")]:
    m = logit_or(DF, "objective_response", [x])
    analyses.append({"hypothesis_ids": [hid], "code": f"Logit ORR ~ {x}",
                     "result_summary": f"beta({x})={m.params[x]:+.6f}, p={m.pvalues[x]:.3g}",
                     "p_value": float(m.pvalues[x]),
                     "effect_estimate": float(m.params[x]),
                     "significant": bool(m.pvalues[x] < 0.05)})
add_iter(16, hyps, analyses)

# ============================================================================
# Iteration 17 — Symptom burden
# ============================================================================
hyps = [
    {"id": "h17.1", "kind": "novel",
     "text": "Higher fatigue_grade is associated with lower objective_response."},
    {"id": "h17.2", "kind": "novel",
     "text": "Higher pain_nrs is associated with lower objective_response."},
    {"id": "h17.3", "kind": "novel",
     "text": "Higher dyspnea_grade is associated with lower objective_response."},
    {"id": "h17.4", "kind": "novel",
     "text": "Higher cough_grade is associated with lower objective_response."},
    {"id": "h17.5", "kind": "novel",
     "text": "Higher appetite_loss_grade is associated with lower objective_response."},
]
analyses = []
for hid, x in [("h17.1", "fatigue_grade"), ("h17.2", "pain_nrs"),
               ("h17.3", "dyspnea_grade"), ("h17.4", "cough_grade"),
               ("h17.5", "appetite_loss_grade")]:
    m = logit_or(DF, "objective_response", [x])
    analyses.append({"hypothesis_ids": [hid], "code": f"Logit ORR ~ {x}",
                     "result_summary": f"beta({x})={m.params[x]:+.5f}, p={m.pvalues[x]:.3g}",
                     "p_value": float(m.pvalues[x]),
                     "effect_estimate": float(m.params[x]),
                     "significant": bool(m.pvalues[x] < 0.05)})
add_iter(17, hyps, analyses)

# ============================================================================
# Iteration 18 — Prior therapy lines
# ============================================================================
hyps = [
    {"id": "h18.1", "kind": "novel",
     "text": "Higher prior_lines_of_therapy is associated with lower objective_response."},
    {"id": "h18.2", "kind": "novel",
     "text": "prior_immunotherapy=1 is associated with lower objective_response (refractory population)."},
    {"id": "h18.3", "kind": "novel",
     "text": "prior_chemotherapy=1 is associated with lower objective_response."},
    {"id": "h18.4", "kind": "novel",
     "text": "prior_radiation=1 is associated with a different objective_response rate."},
    {"id": "h18.5", "kind": "novel",
     "text": "Higher years_since_diagnosis is associated with a different objective_response rate."},
]
analyses = []
for hid, x in [("h18.1", "prior_lines_of_therapy"), ("h18.2", "prior_immunotherapy"),
               ("h18.3", "prior_chemotherapy"), ("h18.4", "prior_radiation"),
               ("h18.5", "years_since_diagnosis")]:
    if DF[x].dtype == bool or set(DF[x].unique()) <= {0, 1}:
        p1, p0, diff, p = chisq_two_groups(DF[x] == 1, DF["objective_response"])
        analyses.append({"hypothesis_ids": [hid], "code": f"ORR by {x}",
                         "result_summary": f"ORR {p1:.3f} ({x}+) vs {p0:.3f} ({x}-); diff={diff:+.3f}, p={p:.3g}",
                         "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
    else:
        m = logit_or(DF, "objective_response", [x])
        analyses.append({"hypothesis_ids": [hid], "code": f"Logit ORR ~ {x}",
                         "result_summary": f"beta({x})={m.params[x]:+.5f}, p={m.pvalues[x]:.3g}",
                         "p_value": float(m.pvalues[x]),
                         "effect_estimate": float(m.params[x]),
                         "significant": bool(m.pvalues[x] < 0.05)})
add_iter(18, hyps, analyses)

# ============================================================================
# Iteration 19 — TP53 / PIK3CA / additional co-mutations
# ============================================================================
hyps = [
    {"id": "h19.1", "kind": "novel",
     "text": "tp53_mutation is associated with a different objective_response rate."},
    {"id": "h19.2", "kind": "novel",
     "text": "pik3ca_mutation is associated with a different objective_response rate."},
    {"id": "h19.3", "kind": "novel",
     "text": "pten_loss is associated with a different objective_response rate."},
    {"id": "h19.4", "kind": "novel",
     "text": "cdkn2a_loss is associated with a different objective_response rate."},
    {"id": "h19.5", "kind": "novel",
     "text": "her2_amplification is associated with a different objective_response rate."},
]
analyses = []
for hid, v in [("h19.1", "tp53_mutation"), ("h19.2", "pik3ca_mutation"),
               ("h19.3", "pten_loss"), ("h19.4", "cdkn2a_loss"),
               ("h19.5", "her2_amplification")]:
    p1, p0, diff, p = chisq_two_groups(DF[v] == 1, DF["objective_response"])
    analyses.append({"hypothesis_ids": [hid], "code": f"ORR by {v}",
                     "result_summary": f"ORR {p1:.3f} ({v}+) vs {p0:.3f} ({v}-); diff={diff:+.3f}, p={p:.3g}",
                     "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
add_iter(19, hyps, analyses)

# ============================================================================
# Iteration 20 — Vital signs / BMI
# ============================================================================
hyps = [
    {"id": "h20.1", "kind": "novel",
     "text": "Higher bmi is associated with higher objective_response (obesity-paradox in immunotherapy)."},
    {"id": "h20.2", "kind": "novel",
     "text": "Lower spo2_pct is associated with lower objective_response."},
    {"id": "h20.3", "kind": "novel",
     "text": "Higher heart_rate_bpm is associated with lower objective_response."},
    {"id": "h20.4", "kind": "novel",
     "text": "Higher systolic_bp_mmhg is associated with a different objective_response rate."},
]
analyses = []
for hid, x in [("h20.1", "bmi"), ("h20.2", "spo2_pct"), ("h20.3", "heart_rate_bpm"),
               ("h20.4", "systolic_bp_mmhg")]:
    m = logit_or(DF, "objective_response", [x])
    analyses.append({"hypothesis_ids": [hid], "code": f"Logit ORR ~ {x}",
                     "result_summary": f"beta({x})={m.params[x]:+.5f}, p={m.pvalues[x]:.3g}",
                     "p_value": float(m.pvalues[x]),
                     "effect_estimate": float(m.params[x]),
                     "significant": bool(m.pvalues[x] < 0.05)})
add_iter(20, hyps, analyses)

# ============================================================================
# Iteration 21 — Pharmacogenomic SNPs (negative-control candidates)
# ============================================================================
hyps = [
    {"id": "h21.1", "kind": "novel",
     "text": "snp_rs1045642 carrier status is not associated with objective_response (negative control)."},
    {"id": "h21.2", "kind": "novel",
     "text": "snp_rs429358 (APOE) carrier status is not associated with objective_response."},
    {"id": "h21.3", "kind": "novel",
     "text": "snp_rs1801133 (MTHFR) carrier status is not associated with objective_response."},
    {"id": "h21.4", "kind": "novel",
     "text": "Across 23 SNPs, the count of significant associations with objective_response at alpha=0.05 is approximately consistent with the null."},
]
snps = [c for c in DF.columns if c.startswith("snp_rs")]
analyses = []
for hid, v in [("h21.1", "snp_rs1045642"), ("h21.2", "snp_rs429358"), ("h21.3", "snp_rs1801133")]:
    p1, p0, diff, p = chisq_two_groups(DF[v] == 1, DF["objective_response"])
    analyses.append({"hypothesis_ids": [hid], "code": f"ORR by {v}",
                     "result_summary": f"ORR {p1:.3f} ({v}+) vs {p0:.3f} ({v}-); diff={diff:+.3f}, p={p:.3g}",
                     "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
sig = 0
ps = []
for s in snps:
    p1, p0, diff, p = chisq_two_groups(DF[s] == 1, DF["objective_response"])
    ps.append(p)
    if p < 0.05:
        sig += 1
analyses.append({"hypothesis_ids": ["h21.4"],
                 "code": "23 chi-square tests of SNPs vs ORR",
                 "result_summary": f"{sig}/{len(snps)} SNPs achieved p<0.05; expected under null ~{0.05*len(snps):.1f}",
                 "p_value": None,
                 "effect_estimate": float(sig - 0.05 * len(snps)),
                 "significant": bool(sig > 2 * 0.05 * len(snps))})
add_iter(21, hyps, analyses)

# ============================================================================
# Iteration 22 — Continuous treatment x prognostic interactions (full model)
# ============================================================================
hyps = [
    {"id": "h22.1", "kind": "novel",
     "text": "After full multivariable adjustment, treatment_pembrolizumab x pdl1_tps interaction is positive on objective_response."},
    {"id": "h22.2", "kind": "novel",
     "text": "After full multivariable adjustment, treatment_pembrolizumab x stk11_mutation interaction is negative on objective_response."},
]
covars = ["treatment_pembrolizumab", "treatment_sotorasib", "treatment_olaparib", "treatment_osimertinib",
          "ecog_ps", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "stage_iv", "has_brain_mets",
          "age_years", "sex_female", "pdl1_tps", "tmb_high", "egfr_mutation", "kras_g12c",
          "alk_fusion", "stk11_mutation", "keap1_mutation", "brca2_mutation",
          "pembro_x_pdl1", "pembro_x_tmb", "pembro_x_stk11_mutation"]
m = logit_or(DF, "objective_response", covars)
with open(HERE / "_full_model_iter22.txt", "w") as f:
    f.write(str(m.summary()))
analyses = []
analyses.append({"hypothesis_ids": ["h22.1"], "code": "Adjusted Logit with pembro:pdl1",
                 "result_summary": f"Adjusted interaction beta(pembro:pdl1)={m.params['pembro_x_pdl1']:+.5f}, p={m.pvalues['pembro_x_pdl1']:.3g}",
                 "p_value": float(m.pvalues["pembro_x_pdl1"]),
                 "effect_estimate": float(m.params["pembro_x_pdl1"]),
                 "significant": bool(m.pvalues["pembro_x_pdl1"] < 0.05)})
analyses.append({"hypothesis_ids": ["h22.2"], "code": "Adjusted Logit with pembro:stk11",
                 "result_summary": f"Adjusted interaction beta(pembro:stk11)={m.params['pembro_x_stk11_mutation']:+.4f}, p={m.pvalues['pembro_x_stk11_mutation']:.3g}",
                 "p_value": float(m.pvalues["pembro_x_stk11_mutation"]),
                 "effect_estimate": float(m.params["pembro_x_stk11_mutation"]),
                 "significant": bool(m.pvalues["pembro_x_stk11_mutation"] < 0.05)})
add_iter(22, hyps, analyses)

# ============================================================================
# Iteration 23 — Prior surgery, MSI-high, prior_targeted_therapy
# ============================================================================
hyps = [
    {"id": "h23.1", "kind": "novel",
     "text": "prior_surgery=1 is associated with higher objective_response (lower disease burden)."},
    {"id": "h23.2", "kind": "novel",
     "text": "msi_high=1 is associated with higher objective_response."},
    {"id": "h23.3", "kind": "novel",
     "text": "prior_targeted_therapy=1 is associated with a different objective_response rate."},
    {"id": "h23.4", "kind": "novel",
     "text": "venous_thromboembolism_history is associated with lower objective_response."},
]
analyses = []
for hid, v in [("h23.1", "prior_surgery"), ("h23.2", "msi_high"),
               ("h23.3", "prior_targeted_therapy"),
               ("h23.4", "venous_thromboembolism_history")]:
    p1, p0, diff, p = chisq_two_groups(DF[v] == 1, DF["objective_response"])
    analyses.append({"hypothesis_ids": [hid], "code": f"ORR by {v}",
                     "result_summary": f"ORR {p1:.3f} ({v}+) vs {p0:.3f} ({v}-); diff={diff:+.3f}, p={p:.3g}",
                     "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05)})
add_iter(23, hyps, analyses)

# ============================================================================
# Iteration 24 — Compound multivariable model with biomarker-matched interactions
# ============================================================================
hyps = [
    {"id": "h24.1", "kind": "refined",
     "text": "After multivariable adjustment, the treatment_sotorasib x kras_g12c interaction remains a positive predictor of objective_response."},
    {"id": "h24.2", "kind": "refined",
     "text": "After multivariable adjustment, the treatment_osimertinib x egfr_mutation interaction remains a positive predictor of objective_response."},
    {"id": "h24.3", "kind": "refined",
     "text": "After multivariable adjustment, the treatment_olaparib x brca2_mutation interaction remains a positive predictor of objective_response."},
]
DF["sot_x_kras"] = DF["treatment_sotorasib"] * DF["kras_g12c"]
DF["osi_x_egfr"] = DF["treatment_osimertinib"] * DF["egfr_mutation"]
DF["ola_x_brca2"] = DF["treatment_olaparib"] * DF["brca2_mutation"]
covars = ["treatment_pembrolizumab", "treatment_sotorasib", "treatment_olaparib", "treatment_osimertinib",
          "ecog_ps", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "stage_iv", "has_brain_mets",
          "age_years", "sex_female",
          "kras_g12c", "egfr_mutation", "brca2_mutation",
          "sot_x_kras", "osi_x_egfr", "ola_x_brca2"]
m = logit_or(DF, "objective_response", covars)
with open(HERE / "_full_model_iter24.txt", "w") as f:
    f.write(str(m.summary()))
analyses = []
for hid, key in [("h24.1", "sot_x_kras"), ("h24.2", "osi_x_egfr"), ("h24.3", "ola_x_brca2")]:
    analyses.append({"hypothesis_ids": [hid], "code": "Multivariable logit with all matched-biomarker interactions",
                     "result_summary": f"Adjusted beta({key})={m.params[key]:+.3f}, p={m.pvalues[key]:.3g}",
                     "p_value": float(m.pvalues[key]),
                     "effect_estimate": float(m.params[key]),
                     "significant": bool(m.pvalues[key] < 0.05)})
add_iter(24, hyps, analyses)

# ============================================================================
# Iteration 25 — Liver function (AST/ALT/bilirubin), kidney function, glucose, sodium
# ============================================================================
hyps = [
    {"id": "h25.1", "kind": "novel",
     "text": "Higher ast_u_l is associated with lower objective_response."},
    {"id": "h25.2", "kind": "novel",
     "text": "Higher alt_u_l is associated with lower objective_response."},
    {"id": "h25.3", "kind": "novel",
     "text": "Higher total_bilirubin_mg_dl is associated with lower objective_response."},
    {"id": "h25.4", "kind": "novel",
     "text": "Higher creatinine_mg_dl is associated with a different objective_response rate."},
    {"id": "h25.5", "kind": "novel",
     "text": "Lower sodium_meq_l (hyponatremia) is associated with lower objective_response."},
    {"id": "h25.6", "kind": "novel",
     "text": "Higher calcium_mg_dl (hypercalcemia, possibly bone-mets-related) is associated with lower objective_response."},
]
analyses = []
for hid, x in [("h25.1", "ast_u_l"), ("h25.2", "alt_u_l"),
               ("h25.3", "total_bilirubin_mg_dl"), ("h25.4", "creatinine_mg_dl"),
               ("h25.5", "sodium_meq_l"), ("h25.6", "calcium_mg_dl")]:
    m = logit_or(DF, "objective_response", [x])
    analyses.append({"hypothesis_ids": [hid], "code": f"Logit ORR ~ {x}",
                     "result_summary": f"beta({x})={m.params[x]:+.5f}, p={m.pvalues[x]:.3g}",
                     "p_value": float(m.pvalues[x]),
                     "effect_estimate": float(m.params[x]),
                     "significant": bool(m.pvalues[x] < 0.05)})
add_iter(25, hyps, analyses)

# ============================================================================
# Emit transcript.json
# ============================================================================
transcript = {
    "dataset_id": "ds001_nsclc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-iterative@v1",
    "max_iterations": 25,
    "iterations": ITERATIONS,
}
with open(HERE / "transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)
print(f"Wrote transcript.json with {len(ITERATIONS)} iterations")

# Also dump a concise summary for narrative writing
sig_table = []
for it in ITERATIONS:
    for a in it["analyses"]:
        sig_table.append({
            "iter": it["index"],
            "hids": ",".join(a["hypothesis_ids"]),
            "p": a.get("p_value"),
            "eff": a.get("effect_estimate"),
            "sig": a.get("significant"),
            "summary": a["result_summary"],
        })
with open(HERE / "_summary_table.json", "w") as f:
    json.dump(sig_table, f, indent=2)
print(f"Wrote _summary_table.json with {len(sig_table)} analyses")
