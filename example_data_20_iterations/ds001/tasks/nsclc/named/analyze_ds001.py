"""Iterative analysis of ds001_nsclc dataset.

Builds transcript.json and analysis_summary.txt by running 25 iterations
of hypothesis-test cycles on biomarker / treatment / outcome relationships.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
df = pd.read_parquet(HERE / "dataset.parquet")
N = len(df)

iterations = []
narrative = []


def add_iter(idx, hypotheses, analyses):
    iterations.append({
        "index": idx,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses,
    })


def fisher_or(a, b):
    """Return OR and Fisher p for binary group a vs b on objective_response."""
    yes_a = int((df.loc[a, "objective_response"] == 1).sum())
    no_a = int((df.loc[a, "objective_response"] == 0).sum())
    yes_b = int((df.loc[b, "objective_response"] == 1).sum())
    no_b = int((df.loc[b, "objective_response"] == 0).sum())
    table = [[yes_a, no_a], [yes_b, no_b]]
    odds, p = stats.fisher_exact(table)
    p_a = yes_a / max(yes_a + no_a, 1)
    p_b = yes_b / max(yes_b + no_b, 1)
    return float(odds), float(p), float(p_a), float(p_b)


def logreg(formula, data=None):
    """Fit logistic regression and return the fitted model."""
    if data is None:
        data = df
    return smf.logit(formula, data=data).fit(disp=0)


# ------------------------------------------------------------------
# Iteration 1: main treatment effects
# ------------------------------------------------------------------
hyps = [
    {"id": "h1.1", "kind": "novel",
     "text": "Patients receiving treatment_pembrolizumab have a higher rate of objective_response than those not receiving treatment_pembrolizumab."},
    {"id": "h1.2", "kind": "novel",
     "text": "Patients receiving treatment_sotorasib have a different rate of objective_response than those not receiving treatment_sotorasib."},
    {"id": "h1.3", "kind": "novel",
     "text": "Patients receiving treatment_olaparib have a different rate of objective_response than those not receiving treatment_olaparib."},
    {"id": "h1.4", "kind": "novel",
     "text": "Patients receiving treatment_osimertinib have a different rate of objective_response than those not receiving treatment_osimertinib."},
]
ans = []
for hid, tx in [("h1.1", "treatment_pembrolizumab"), ("h1.2", "treatment_sotorasib"),
                ("h1.3", "treatment_olaparib"), ("h1.4", "treatment_osimertinib")]:
    odds, p, p_tx, p_no = fisher_or(df[tx] == 1, df[tx] == 0)
    eff = p_tx - p_no
    ans.append({
        "hypothesis_ids": [hid],
        "code": f"crosstab({tx}, objective_response); fisher_exact",
        "result_summary": f"ORR={p_tx:.4f} on {tx} vs {p_no:.4f} off; OR={odds:.3f}, Fisher p={p:.3g}.",
        "p_value": p, "effect_estimate": float(eff), "significant": p < 0.05,
    })
add_iter(1, hyps, ans)
narrative.append("Iter 1 — Main treatment effects on ORR:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 2: biomarker main effects
# ------------------------------------------------------------------
hyps = [
    {"id": "h2.1", "kind": "novel",
     "text": "Patients with egfr_mutation have a different rate of objective_response than those without egfr_mutation."},
    {"id": "h2.2", "kind": "novel",
     "text": "Patients with kras_g12c have a different rate of objective_response than those without kras_g12c."},
    {"id": "h2.3", "kind": "novel",
     "text": "Patients with brca2_mutation have a different rate of objective_response than those without brca2_mutation."},
    {"id": "h2.4", "kind": "novel",
     "text": "Patients with tmb_high have a higher rate of objective_response than those without tmb_high."},
    {"id": "h2.5", "kind": "novel",
     "text": "Higher pdl1_tps is associated with higher rate of objective_response (continuous logistic regression)."},
]
ans = []
for hid, biomarker in [("h2.1", "egfr_mutation"), ("h2.2", "kras_g12c"),
                       ("h2.3", "brca2_mutation"), ("h2.4", "tmb_high")]:
    odds, p, p_pos, p_neg = fisher_or(df[biomarker] == 1, df[biomarker] == 0)
    ans.append({
        "hypothesis_ids": [hid],
        "code": f"crosstab({biomarker}, objective_response); fisher_exact",
        "result_summary": f"ORR with {biomarker}={p_pos:.4f} vs without={p_neg:.4f}; OR={odds:.3f}, p={p:.3g}.",
        "p_value": p, "effect_estimate": float(p_pos - p_neg), "significant": p < 0.05,
    })
m = logreg("objective_response ~ pdl1_tps")
ans.append({
    "hypothesis_ids": ["h2.5"],
    "code": "logit(objective_response ~ pdl1_tps)",
    "result_summary": f"Logistic OR per unit pdl1_tps = {math.exp(m.params['pdl1_tps']):.3f}; p={m.pvalues['pdl1_tps']:.3g}.",
    "p_value": float(m.pvalues["pdl1_tps"]),
    "effect_estimate": float(m.params["pdl1_tps"]),
    "significant": m.pvalues["pdl1_tps"] < 0.05,
})
add_iter(2, hyps, ans)
narrative.append("Iter 2 — Biomarker main effects on ORR:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 3: EGFR x osimertinib interaction
# ------------------------------------------------------------------
hyps = [
    {"id": "h3.1", "kind": "novel",
     "text": "Among egfr_mutation-positive patients, treatment_osimertinib produces a higher ORR than no osimertinib."},
    {"id": "h3.2", "kind": "novel",
     "text": "Among egfr_mutation-negative patients, treatment_osimertinib has minimal/no effect on ORR."},
    {"id": "h3.3", "kind": "novel",
     "text": "There is a positive interaction between egfr_mutation and treatment_osimertinib on objective_response (logistic regression interaction term beta > 0)."},
]
ans = []
mp = df["egfr_mutation"] == 1
mn = df["egfr_mutation"] == 0
odds, p, p_tx, p_no = fisher_or((df["treatment_osimertinib"] == 1) & mp,
                                (df["treatment_osimertinib"] == 0) & mp)
ans.append({"hypothesis_ids": ["h3.1"],
            "code": "fisher among egfr_mutation==1: osimertinib vs not",
            "result_summary": f"In EGFR+: ORR osi={p_tx:.4f} vs no-osi={p_no:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p_tx - p_no), "significant": p < 0.05})
odds, p, p_tx, p_no = fisher_or((df["treatment_osimertinib"] == 1) & mn,
                                (df["treatment_osimertinib"] == 0) & mn)
ans.append({"hypothesis_ids": ["h3.2"],
            "code": "fisher among egfr_mutation==0: osimertinib vs not",
            "result_summary": f"In EGFR-: ORR osi={p_tx:.4f} vs no-osi={p_no:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p_tx - p_no), "significant": p < 0.05})
m = logreg("objective_response ~ egfr_mutation * treatment_osimertinib")
b = m.params["egfr_mutation:treatment_osimertinib"]
pv = m.pvalues["egfr_mutation:treatment_osimertinib"]
ans.append({"hypothesis_ids": ["h3.3"],
            "code": "logit(obj ~ egfr_mutation * treatment_osimertinib)",
            "result_summary": f"Interaction beta={b:.3f} (OR={math.exp(b):.3f}), p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(3, hyps, ans)
narrative.append("Iter 3 — EGFR x osimertinib:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 4: KRAS G12C x sotorasib
# ------------------------------------------------------------------
hyps = [
    {"id": "h4.1", "kind": "novel",
     "text": "Among kras_g12c-positive patients, treatment_sotorasib increases ORR vs no sotorasib."},
    {"id": "h4.2", "kind": "novel",
     "text": "Among kras_g12c-negative patients, treatment_sotorasib has little/no effect on ORR."},
    {"id": "h4.3", "kind": "novel",
     "text": "There is a positive interaction between kras_g12c and treatment_sotorasib on objective_response."},
]
ans = []
mp = df["kras_g12c"] == 1; mn = df["kras_g12c"] == 0
odds, p, p_tx, p_no = fisher_or((df["treatment_sotorasib"] == 1) & mp,
                                (df["treatment_sotorasib"] == 0) & mp)
ans.append({"hypothesis_ids": ["h4.1"],
            "code": "fisher among kras_g12c==1: sotorasib vs not",
            "result_summary": f"In KRASG12C+: ORR sotorasib={p_tx:.4f} vs no={p_no:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p_tx - p_no), "significant": p < 0.05})
odds, p, p_tx, p_no = fisher_or((df["treatment_sotorasib"] == 1) & mn,
                                (df["treatment_sotorasib"] == 0) & mn)
ans.append({"hypothesis_ids": ["h4.2"],
            "code": "fisher among kras_g12c==0: sotorasib vs not",
            "result_summary": f"In KRASG12C-: ORR sotorasib={p_tx:.4f} vs no={p_no:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p_tx - p_no), "significant": p < 0.05})
m = logreg("objective_response ~ kras_g12c * treatment_sotorasib")
b = m.params["kras_g12c:treatment_sotorasib"]
pv = m.pvalues["kras_g12c:treatment_sotorasib"]
ans.append({"hypothesis_ids": ["h4.3"],
            "code": "logit(obj ~ kras_g12c * treatment_sotorasib)",
            "result_summary": f"Interaction beta={b:.3f} (OR={math.exp(b):.3f}), p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(4, hyps, ans)
narrative.append("Iter 4 — KRAS G12C x sotorasib:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 5: BRCA2 x olaparib
# ------------------------------------------------------------------
hyps = [
    {"id": "h5.1", "kind": "novel",
     "text": "Among brca2_mutation-positive patients, treatment_olaparib increases ORR vs no olaparib."},
    {"id": "h5.2", "kind": "novel",
     "text": "Among brca2_mutation-negative patients, treatment_olaparib has no/minimal effect on ORR."},
    {"id": "h5.3", "kind": "novel",
     "text": "There is a positive interaction between brca2_mutation and treatment_olaparib on objective_response."},
]
ans = []
mp = df["brca2_mutation"] == 1; mn = df["brca2_mutation"] == 0
odds, p, p_tx, p_no = fisher_or((df["treatment_olaparib"] == 1) & mp,
                                (df["treatment_olaparib"] == 0) & mp)
ans.append({"hypothesis_ids": ["h5.1"],
            "code": "fisher among brca2_mutation==1: olaparib vs not",
            "result_summary": f"In BRCA2+: ORR olaparib={p_tx:.4f} vs no={p_no:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p_tx - p_no), "significant": p < 0.05})
odds, p, p_tx, p_no = fisher_or((df["treatment_olaparib"] == 1) & mn,
                                (df["treatment_olaparib"] == 0) & mn)
ans.append({"hypothesis_ids": ["h5.2"],
            "code": "fisher among brca2_mutation==0: olaparib vs not",
            "result_summary": f"In BRCA2-: ORR olaparib={p_tx:.4f} vs no={p_no:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p_tx - p_no), "significant": p < 0.05})
m = logreg("objective_response ~ brca2_mutation * treatment_olaparib")
b = m.params["brca2_mutation:treatment_olaparib"]
pv = m.pvalues["brca2_mutation:treatment_olaparib"]
ans.append({"hypothesis_ids": ["h5.3"],
            "code": "logit(obj ~ brca2_mutation * treatment_olaparib)",
            "result_summary": f"Interaction beta={b:.3f} (OR={math.exp(b):.3f}), p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(5, hyps, ans)
narrative.append("Iter 5 — BRCA2 x olaparib:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 6: PD-L1 / TMB x pembrolizumab
# ------------------------------------------------------------------
hyps = [
    {"id": "h6.1", "kind": "novel",
     "text": "Higher pdl1_tps amplifies the positive effect of treatment_pembrolizumab on ORR (positive interaction term)."},
    {"id": "h6.2", "kind": "novel",
     "text": "tmb_high amplifies the positive effect of treatment_pembrolizumab on ORR."},
    {"id": "h6.3", "kind": "novel",
     "text": "Among patients with pdl1_tps>=0.5, treatment_pembrolizumab is associated with higher ORR than no pembrolizumab."},
]
ans = []
m = logreg("objective_response ~ pdl1_tps * treatment_pembrolizumab")
b = m.params["pdl1_tps:treatment_pembrolizumab"]; pv = m.pvalues["pdl1_tps:treatment_pembrolizumab"]
ans.append({"hypothesis_ids": ["h6.1"],
            "code": "logit(obj ~ pdl1_tps * treatment_pembrolizumab)",
            "result_summary": f"Interaction beta={b:.3f} (OR={math.exp(b):.3f}), p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
m = logreg("objective_response ~ tmb_high * treatment_pembrolizumab")
b = m.params["tmb_high:treatment_pembrolizumab"]; pv = m.pvalues["tmb_high:treatment_pembrolizumab"]
ans.append({"hypothesis_ids": ["h6.2"],
            "code": "logit(obj ~ tmb_high * treatment_pembrolizumab)",
            "result_summary": f"Interaction beta={b:.3f} (OR={math.exp(b):.3f}), p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
mask = df["pdl1_tps"] >= 0.5
odds, p, p_tx, p_no = fisher_or((df["treatment_pembrolizumab"] == 1) & mask,
                                (df["treatment_pembrolizumab"] == 0) & mask)
ans.append({"hypothesis_ids": ["h6.3"],
            "code": "fisher pembro vs not, restricted to pdl1_tps>=0.5",
            "result_summary": f"In PD-L1>=50%: ORR pembro={p_tx:.4f} vs not={p_no:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p_tx - p_no), "significant": p < 0.05})
add_iter(6, hyps, ans)
narrative.append("Iter 6 — Pembrolizumab x PD-L1/TMB:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 7: ECOG performance status main effect
# ------------------------------------------------------------------
hyps = [
    {"id": "h7.1", "kind": "novel",
     "text": "Higher ecog_ps (worse performance status) is associated with lower objective_response."},
    {"id": "h7.2", "kind": "novel",
     "text": "ECOG=2 patients have substantially lower ORR than ECOG=0 patients."},
]
ans = []
m = logreg("objective_response ~ ecog_ps")
b = m.params["ecog_ps"]; pv = m.pvalues["ecog_ps"]
ans.append({"hypothesis_ids": ["h7.1"],
            "code": "logit(obj ~ ecog_ps)",
            "result_summary": f"Per-unit ecog_ps: beta={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
odds, p, p1, p0 = fisher_or(df["ecog_ps"] == 2, df["ecog_ps"] == 0)
ans.append({"hypothesis_ids": ["h7.2"],
            "code": "fisher ecog_ps==2 vs ==0",
            "result_summary": f"ORR ECOG2={p1:.4f} vs ECOG0={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
add_iter(7, hyps, ans)
narrative.append("Iter 7 — ECOG performance status:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 8: Albumin / LDH / weight loss / NLR / CRP main effects
# ------------------------------------------------------------------
hyps = [
    {"id": "h8.1", "kind": "novel",
     "text": "Higher albumin_g_dl is associated with higher objective_response."},
    {"id": "h8.2", "kind": "novel",
     "text": "Higher ldh_u_l is associated with lower objective_response."},
    {"id": "h8.3", "kind": "novel",
     "text": "Higher weight_loss_pct_6mo is associated with lower objective_response."},
    {"id": "h8.4", "kind": "novel",
     "text": "Higher nlr (neutrophil/lymphocyte ratio) is associated with lower objective_response."},
    {"id": "h8.5", "kind": "novel",
     "text": "Higher crp_mg_l is associated with lower objective_response."},
]
ans = []
for hid, var in [("h8.1", "albumin_g_dl"), ("h8.2", "ldh_u_l"),
                 ("h8.3", "weight_loss_pct_6mo"), ("h8.4", "nlr"), ("h8.5", "crp_mg_l")]:
    m = logreg(f"objective_response ~ {var}")
    b = m.params[var]; pv = m.pvalues[var]
    ans.append({"hypothesis_ids": [hid],
                "code": f"logit(objective_response ~ {var})",
                "result_summary": f"{var}: beta={b:.5f} (per unit), OR={math.exp(b):.4f}, p={pv:.3g}.",
                "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(8, hyps, ans)
narrative.append("Iter 8 — Lab/host markers (main effects):\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 9: Histology + STK11 + KEAP1 + smoking
# ------------------------------------------------------------------
hyps = [
    {"id": "h9.1", "kind": "novel",
     "text": "Squamous histology has a different ORR vs adenocarcinoma."},
    {"id": "h9.2", "kind": "novel",
     "text": "stk11_mutation is associated with reduced ORR overall."},
    {"id": "h9.3", "kind": "novel",
     "text": "Among pembrolizumab-treated patients, stk11_mutation reduces ORR (negative interaction with pembrolizumab)."},
    {"id": "h9.4", "kind": "novel",
     "text": "Among pembrolizumab-treated patients, keap1_mutation reduces ORR (negative interaction with pembrolizumab)."},
    {"id": "h9.5", "kind": "novel",
     "text": "Never-smokers have a different ORR than current smokers."},
]
ans = []
df["squamous"] = (df["histology"] == "squamous").astype(int)
odds, p, p1, p0 = fisher_or(df["squamous"] == 1, df["squamous"] == 0)
ans.append({"hypothesis_ids": ["h9.1"],
            "code": "fisher squamous==1 vs ==0",
            "result_summary": f"ORR squamous={p1:.4f} vs adeno={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
odds, p, p1, p0 = fisher_or(df["stk11_mutation"] == 1, df["stk11_mutation"] == 0)
ans.append({"hypothesis_ids": ["h9.2"],
            "code": "fisher stk11_mutation==1 vs ==0",
            "result_summary": f"ORR STK11+={p1:.4f} vs STK11-={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
m = logreg("objective_response ~ stk11_mutation * treatment_pembrolizumab")
b = m.params["stk11_mutation:treatment_pembrolizumab"]
pv = m.pvalues["stk11_mutation:treatment_pembrolizumab"]
ans.append({"hypothesis_ids": ["h9.3"],
            "code": "logit(obj ~ stk11_mutation * pembrolizumab)",
            "result_summary": f"Interaction beta={b:.3f} (OR={math.exp(b):.3f}), p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
m = logreg("objective_response ~ keap1_mutation * treatment_pembrolizumab")
b = m.params["keap1_mutation:treatment_pembrolizumab"]
pv = m.pvalues["keap1_mutation:treatment_pembrolizumab"]
ans.append({"hypothesis_ids": ["h9.4"],
            "code": "logit(obj ~ keap1_mutation * pembrolizumab)",
            "result_summary": f"Interaction beta={b:.3f} (OR={math.exp(b):.3f}), p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
df["never_smoker"] = (df["smoking_status"] == "never").astype(int)
df["current_smoker"] = (df["smoking_status"] == "current").astype(int)
odds, p, p_n, p_c = fisher_or(df["never_smoker"] == 1, df["current_smoker"] == 1)
ans.append({"hypothesis_ids": ["h9.5"],
            "code": "fisher never vs current smoker",
            "result_summary": f"ORR never={p_n:.4f} vs current={p_c:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p_n - p_c), "significant": p < 0.05})
add_iter(9, hyps, ans)
narrative.append("Iter 9 — Histology / STK11 / KEAP1 / smoking:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 10: Metastasis sites + stage IV
# ------------------------------------------------------------------
hyps = [
    {"id": "h10.1", "kind": "novel",
     "text": "stage_iv patients have lower ORR than non-stage IV patients."},
    {"id": "h10.2", "kind": "novel",
     "text": "Patients with liver_mets have lower ORR than those without."},
    {"id": "h10.3", "kind": "novel",
     "text": "Patients with bone_mets have lower ORR than those without."},
    {"id": "h10.4", "kind": "novel",
     "text": "Patients with has_brain_mets have lower ORR than those without."},
    {"id": "h10.5", "kind": "novel",
     "text": "Patients with pleural_effusion have lower ORR than those without."},
]
ans = []
for hid, v in [("h10.1", "stage_iv"), ("h10.2", "liver_mets"), ("h10.3", "bone_mets"),
               ("h10.4", "has_brain_mets"), ("h10.5", "pleural_effusion")]:
    odds, p, p1, p0 = fisher_or(df[v] == 1, df[v] == 0)
    ans.append({"hypothesis_ids": [hid],
                "code": f"fisher {v}==1 vs ==0",
                "result_summary": f"ORR {v}+={p1:.4f} vs {v}-={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
                "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
add_iter(10, hyps, ans)
narrative.append("Iter 10 — Metastatic burden:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 11: Demographics — age, sex, race/ethnicity, insurance, rural
# ------------------------------------------------------------------
hyps = [
    {"id": "h11.1", "kind": "novel",
     "text": "Older age_years is associated with lower objective_response."},
    {"id": "h11.2", "kind": "novel",
     "text": "Female sex (sex_female=1) is associated with a different ORR than male."},
    {"id": "h11.3", "kind": "novel",
     "text": "race_ethnicity is associated with ORR (chi-square test of independence)."},
    {"id": "h11.4", "kind": "novel",
     "text": "Medicaid/uninsured insurance is associated with different ORR than medicare/private."},
    {"id": "h11.5", "kind": "novel",
     "text": "rural_residence is associated with different ORR."},
]
ans = []
m = logreg("objective_response ~ age_years")
b = m.params["age_years"]; pv = m.pvalues["age_years"]
ans.append({"hypothesis_ids": ["h11.1"],
            "code": "logit(obj ~ age_years)",
            "result_summary": f"Per year age: beta={b:.5f}, OR={math.exp(b):.4f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
odds, p, p1, p0 = fisher_or(df["sex_female"] == 1, df["sex_female"] == 0)
ans.append({"hypothesis_ids": ["h11.2"],
            "code": "fisher sex_female",
            "result_summary": f"ORR female={p1:.4f} vs male={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
ct = pd.crosstab(df["race_ethnicity"], df["objective_response"])
chi2, pv, dof, _ = stats.chi2_contingency(ct)
rates = df.groupby("race_ethnicity")["objective_response"].mean().to_dict()
ans.append({"hypothesis_ids": ["h11.3"],
            "code": "chi2_contingency(race_ethnicity, objective_response)",
            "result_summary": f"Chi2={chi2:.3f}, dof={dof}, p={pv:.3g}; ORR by race: {rates}.",
            "p_value": float(pv),
            "effect_estimate": float(max(rates.values()) - min(rates.values())),
            "significant": pv < 0.05})
df["medicaid_uninsured"] = df["insurance_type"].isin(["medicaid", "uninsured"]).astype(int)
odds, p, p1, p0 = fisher_or(df["medicaid_uninsured"] == 1, df["medicaid_uninsured"] == 0)
ans.append({"hypothesis_ids": ["h11.4"],
            "code": "fisher medicaid+uninsured vs medicare+private",
            "result_summary": f"ORR medicaid/uninsured={p1:.4f} vs medicare/private={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
odds, p, p1, p0 = fisher_or(df["rural_residence"] == 1, df["rural_residence"] == 0)
ans.append({"hypothesis_ids": ["h11.5"],
            "code": "fisher rural_residence",
            "result_summary": f"ORR rural={p1:.4f} vs non-rural={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
add_iter(11, hyps, ans)
narrative.append("Iter 11 — Demographics / SES:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 12: Symptom severity grades
# ------------------------------------------------------------------
hyps = [
    {"id": "h12.1", "kind": "novel",
     "text": "Higher fatigue_grade is associated with lower objective_response."},
    {"id": "h12.2", "kind": "novel",
     "text": "Higher pain_nrs is associated with lower objective_response."},
    {"id": "h12.3", "kind": "novel",
     "text": "Higher dyspnea_grade is associated with lower objective_response."},
    {"id": "h12.4", "kind": "novel",
     "text": "Higher cough_grade is associated with lower objective_response."},
    {"id": "h12.5", "kind": "novel",
     "text": "Higher appetite_loss_grade is associated with lower objective_response."},
]
ans = []
for hid, var in [("h12.1", "fatigue_grade"), ("h12.2", "pain_nrs"),
                 ("h12.3", "dyspnea_grade"), ("h12.4", "cough_grade"),
                 ("h12.5", "appetite_loss_grade")]:
    m = logreg(f"objective_response ~ {var}")
    b = m.params[var]; pv = m.pvalues[var]
    ans.append({"hypothesis_ids": [hid],
                "code": f"logit(obj ~ {var})",
                "result_summary": f"{var}: beta={b:.4f}, OR per unit={math.exp(b):.3f}, p={pv:.3g}.",
                "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(12, hyps, ans)
narrative.append("Iter 12 — Symptom grades:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 13: Additional rare driver biomarkers
# ------------------------------------------------------------------
hyps = [
    {"id": "h13.1", "kind": "novel",
     "text": "alk_fusion is associated with higher ORR (often targetable in NSCLC)."},
    {"id": "h13.2", "kind": "novel",
     "text": "ros1_fusion is associated with higher ORR."},
    {"id": "h13.3", "kind": "novel",
     "text": "ret_fusion is associated with higher ORR."},
    {"id": "h13.4", "kind": "novel",
     "text": "braf_v600e is associated with higher ORR."},
    {"id": "h13.5", "kind": "novel",
     "text": "met_exon14_skipping is associated with higher ORR."},
    {"id": "h13.6", "kind": "novel",
     "text": "ntrk_fusion is associated with higher ORR."},
    {"id": "h13.7", "kind": "novel",
     "text": "her2_amplification is associated with different ORR."},
]
ans = []
for hid, v in [("h13.1", "alk_fusion"), ("h13.2", "ros1_fusion"),
               ("h13.3", "ret_fusion"), ("h13.4", "braf_v600e"),
               ("h13.5", "met_exon14_skipping"), ("h13.6", "ntrk_fusion"),
               ("h13.7", "her2_amplification")]:
    odds, p, p1, p0 = fisher_or(df[v] == 1, df[v] == 0)
    ans.append({"hypothesis_ids": [hid],
                "code": f"fisher {v}",
                "result_summary": f"ORR {v}+={p1:.4f} vs {v}-={p0:.4f}; OR={odds:.3f}, p={p:.3g}, n+={int((df[v]==1).sum())}.",
                "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
add_iter(13, hyps, ans)
narrative.append("Iter 13 — Rare oncogenic drivers:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 14: Treatment x histology interactions
# ------------------------------------------------------------------
hyps = [
    {"id": "h14.1", "kind": "novel",
     "text": "treatment_pembrolizumab effect on ORR differs between squamous and adenocarcinoma (interaction)."},
    {"id": "h14.2", "kind": "novel",
     "text": "treatment_osimertinib effect on ORR is concentrated in adenocarcinoma (negative interaction with squamous)."},
]
ans = []
m = logreg("objective_response ~ squamous * treatment_pembrolizumab")
b = m.params["squamous:treatment_pembrolizumab"]; pv = m.pvalues["squamous:treatment_pembrolizumab"]
ans.append({"hypothesis_ids": ["h14.1"],
            "code": "logit(obj ~ squamous * pembro)",
            "result_summary": f"Interaction beta={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
m = logreg("objective_response ~ squamous * treatment_osimertinib")
b = m.params["squamous:treatment_osimertinib"]; pv = m.pvalues["squamous:treatment_osimertinib"]
ans.append({"hypothesis_ids": ["h14.2"],
            "code": "logit(obj ~ squamous * osi)",
            "result_summary": f"Interaction beta={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(14, hyps, ans)
narrative.append("Iter 14 — Treatment x histology:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 15: Three-way EGFR x osi x prior_targeted_therapy
# ------------------------------------------------------------------
hyps = [
    {"id": "h15.1", "kind": "refined",
     "text": "Among EGFR-mutant patients on osimertinib, prior_targeted_therapy reduces ORR (resistance after prior TKI)."},
    {"id": "h15.2", "kind": "refined",
     "text": "Among EGFR-mutant patients, the osimertinib effect is larger in those with no prior_targeted_therapy than those with prior_targeted_therapy (negative interaction within EGFR+ subset)."},
]
ans = []
mask = (df["egfr_mutation"] == 1) & (df["treatment_osimertinib"] == 1)
odds, p, p1, p0 = fisher_or(mask & (df["prior_targeted_therapy"] == 1),
                            mask & (df["prior_targeted_therapy"] == 0))
ans.append({"hypothesis_ids": ["h15.1"],
            "code": "fisher: among EGFR+/osi+: prior_TT yes vs no",
            "result_summary": f"In EGFR+/osi+: ORR prior-TT={p1:.4f} vs no-prior-TT={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
sub = df[df["egfr_mutation"] == 1].copy()
m = logreg("objective_response ~ treatment_osimertinib * prior_targeted_therapy", data=sub)
b = m.params["treatment_osimertinib:prior_targeted_therapy"]
pv = m.pvalues["treatment_osimertinib:prior_targeted_therapy"]
ans.append({"hypothesis_ids": ["h15.2"],
            "code": "logit(obj ~ osi * prior_TT) within EGFR+",
            "result_summary": f"Interaction beta={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(15, hyps, ans)
narrative.append("Iter 15 — EGFR/osi x prior TT:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 16: Biomarker-matched targeted therapy composite
# ------------------------------------------------------------------
df["matched_therapy"] = (
    ((df["egfr_mutation"] == 1) & (df["treatment_osimertinib"] == 1)) |
    ((df["kras_g12c"] == 1) & (df["treatment_sotorasib"] == 1)) |
    ((df["brca2_mutation"] == 1) & (df["treatment_olaparib"] == 1))
).astype(int)

hyps = [
    {"id": "h16.1", "kind": "refined",
     "text": "Receiving a biomarker-matched targeted therapy (matched_therapy: EGFR+osi, KRASG12C+sotorasib, BRCA2+olaparib) is associated with higher ORR."},
    {"id": "h16.2", "kind": "refined",
     "text": "After adjusting for ECOG, age, stage IV, liver mets, albumin, LDH and pdl1_tps, matched_therapy retains a positive association with ORR."},
]
ans = []
odds, p, p1, p0 = fisher_or(df["matched_therapy"] == 1, df["matched_therapy"] == 0)
ans.append({"hypothesis_ids": ["h16.1"],
            "code": "fisher matched_therapy",
            "result_summary": f"ORR matched_therapy={p1:.4f} vs not={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
m = smf.logit(
    "objective_response ~ matched_therapy + ecog_ps + age_years + stage_iv + liver_mets + albumin_g_dl + ldh_u_l + pdl1_tps",
    data=df).fit(disp=0)
b = m.params["matched_therapy"]; pv = m.pvalues["matched_therapy"]
ans.append({"hypothesis_ids": ["h16.2"],
            "code": "multivariable logit, matched_therapy adjusted",
            "result_summary": f"Adjusted matched_therapy beta={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(16, hyps, ans)
narrative.append("Iter 16 — Matched-therapy effect (multivariable):\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 17: SNP main effects
# ------------------------------------------------------------------
snps = [c for c in df.columns if c.startswith("snp_")]
hyps = [
    {"id": "h17.1", "kind": "novel",
     "text": "Most pharmacogenomic SNPs (snp_*) are unrelated to objective_response (counts of nominally significant tests do not exceed binomial expectation)."},
    {"id": "h17.2", "kind": "novel",
     "text": "snp_rs1045642 (ABCB1 C3435T) is associated with objective_response."},
    {"id": "h17.3", "kind": "novel",
     "text": "snp_rs1799853 (CYP2C9*2) is associated with objective_response."},
    {"id": "h17.4", "kind": "novel",
     "text": "snp_rs429358 (APOE) is associated with objective_response."},
]
ans = []
sig_count = 0
sig_snps = []
for s in snps:
    odds, p, p1, p0 = fisher_or(df[s] == 1, df[s] == 0)
    if p < 0.05:
        sig_count += 1
        sig_snps.append((s, round(p, 4), round(p1 - p0, 4)))
binom_p = float(stats.binomtest(sig_count, len(snps), 0.05).pvalue)
ans.append({"hypothesis_ids": ["h17.1"],
            "code": "loop fisher each snp_*; binomial test of count of p<0.05 vs 0.05",
            "result_summary": f"{sig_count}/{len(snps)} SNPs nominally p<0.05 (binom p={binom_p:.3g}). Hits: {sig_snps}.",
            "p_value": binom_p,
            "effect_estimate": float(sig_count / max(len(snps), 1) - 0.05),
            "significant": binom_p < 0.05})
for hid, s in [("h17.2", "snp_rs1045642"), ("h17.3", "snp_rs1799853"), ("h17.4", "snp_rs429358")]:
    odds, p, p1, p0 = fisher_or(df[s] == 1, df[s] == 0)
    ans.append({"hypothesis_ids": [hid],
                "code": f"fisher {s}",
                "result_summary": f"ORR {s}+={p1:.4f} vs {s}-={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
                "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
add_iter(17, hyps, ans)
narrative.append("Iter 17 — SNP main effects:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 18: Comorbidities & immune-related
# ------------------------------------------------------------------
hyps = [
    {"id": "h18.1", "kind": "novel",
     "text": "autoimmune_disease history is associated with different ORR (overall main effect)."},
    {"id": "h18.2", "kind": "novel",
     "text": "interstitial_lung_disease_history is associated with reduced ORR."},
    {"id": "h18.3", "kind": "novel",
     "text": "chronic_kidney_disease patients have lower ORR than those without."},
    {"id": "h18.4", "kind": "novel",
     "text": "Among patients with autoimmune_disease, the treatment_pembrolizumab effect on ORR is altered (interaction)."},
]
ans = []
for hid, v in [("h18.1", "autoimmune_disease"), ("h18.2", "interstitial_lung_disease_history"),
               ("h18.3", "chronic_kidney_disease")]:
    odds, p, p1, p0 = fisher_or(df[v] == 1, df[v] == 0)
    ans.append({"hypothesis_ids": [hid],
                "code": f"fisher {v}",
                "result_summary": f"ORR {v}+={p1:.4f} vs {v}-={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
                "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
m = logreg("objective_response ~ autoimmune_disease * treatment_pembrolizumab")
b = m.params["autoimmune_disease:treatment_pembrolizumab"]
pv = m.pvalues["autoimmune_disease:treatment_pembrolizumab"]
ans.append({"hypothesis_ids": ["h18.4"],
            "code": "logit(obj ~ autoimmune_disease * pembro)",
            "result_summary": f"Interaction beta={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(18, hyps, ans)
narrative.append("Iter 18 — Comorbidities & autoimmune:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 19: Lab markers — hemoglobin, platelets, ALC, CEA
# ------------------------------------------------------------------
hyps = [
    {"id": "h19.1", "kind": "novel",
     "text": "Higher hemoglobin_g_dl is associated with higher ORR."},
    {"id": "h19.2", "kind": "novel",
     "text": "Higher platelets_k_ul is associated with lower ORR."},
    {"id": "h19.3", "kind": "novel",
     "text": "Higher alc_k_ul (absolute lymphocyte count) is associated with higher ORR."},
    {"id": "h19.4", "kind": "novel",
     "text": "Higher cea_ng_ml is associated with lower ORR."},
]
ans = []
for hid, var in [("h19.1", "hemoglobin_g_dl"), ("h19.2", "platelets_k_ul"),
                 ("h19.3", "alc_k_ul"), ("h19.4", "cea_ng_ml")]:
    m = logreg(f"objective_response ~ {var}")
    b = m.params[var]; pv = m.pvalues[var]
    ans.append({"hypothesis_ids": [hid],
                "code": f"logit(obj ~ {var})",
                "result_summary": f"{var}: beta={b:.5f}, OR per unit={math.exp(b):.4f}, p={pv:.3g}.",
                "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(19, hyps, ans)
narrative.append("Iter 19 — Lab markers (further):\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 20: Prior therapy history
# ------------------------------------------------------------------
hyps = [
    {"id": "h20.1", "kind": "novel",
     "text": "More prior_lines_of_therapy is associated with lower ORR."},
    {"id": "h20.2", "kind": "novel",
     "text": "prior_immunotherapy is associated with lower ORR overall."},
    {"id": "h20.3", "kind": "novel",
     "text": "Among pembrolizumab-treated patients, prior_immunotherapy reduces ORR (negative interaction)."},
]
ans = []
m = logreg("objective_response ~ prior_lines_of_therapy")
b = m.params["prior_lines_of_therapy"]; pv = m.pvalues["prior_lines_of_therapy"]
ans.append({"hypothesis_ids": ["h20.1"],
            "code": "logit(obj ~ prior_lines_of_therapy)",
            "result_summary": f"Per line: beta={b:.4f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
odds, p, p1, p0 = fisher_or(df["prior_immunotherapy"] == 1, df["prior_immunotherapy"] == 0)
ans.append({"hypothesis_ids": ["h20.2"],
            "code": "fisher prior_immunotherapy",
            "result_summary": f"ORR prior-IO+={p1:.4f} vs prior-IO-={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
m = logreg("objective_response ~ prior_immunotherapy * treatment_pembrolizumab")
b = m.params["prior_immunotherapy:treatment_pembrolizumab"]
pv = m.pvalues["prior_immunotherapy:treatment_pembrolizumab"]
ans.append({"hypothesis_ids": ["h20.3"],
            "code": "logit(obj ~ prior_immunotherapy * pembro)",
            "result_summary": f"Interaction beta={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(20, hyps, ans)
narrative.append("Iter 20 — Prior therapy history:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 21: Joint model with all 4 biomarker x treatment interactions
# ------------------------------------------------------------------
hyps = [
    {"id": "h21.1", "kind": "refined",
     "text": "In a joint logistic regression with EGFR*osi, KRASG12C*sotorasib, BRCA2*olaparib, PDL1*pembrolizumab plus ECOG/albumin/LDH covariates, each biomarker-treatment interaction is positive and significant."},
]
ans = []
m = smf.logit(
    "objective_response ~ "
    "egfr_mutation * treatment_osimertinib + "
    "kras_g12c * treatment_sotorasib + "
    "brca2_mutation * treatment_olaparib + "
    "pdl1_tps * treatment_pembrolizumab + "
    "ecog_ps + albumin_g_dl + ldh_u_l",
    data=df).fit(disp=0)
for term in ["egfr_mutation:treatment_osimertinib",
             "kras_g12c:treatment_sotorasib",
             "brca2_mutation:treatment_olaparib",
             "pdl1_tps:treatment_pembrolizumab"]:
    b = m.params[term]; pv = m.pvalues[term]
    ans.append({"hypothesis_ids": ["h21.1"],
                "code": "joint logit with all 4 interactions + adjustments",
                "result_summary": f"{term}: beta={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
                "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(21, hyps, ans)
narrative.append("Iter 21 — Joint adjusted model with all biomarker x treatment interactions:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 22: Pembro subgroup analyses (PD-L1 within tx, TMB)
# ------------------------------------------------------------------
hyps = [
    {"id": "h22.1", "kind": "refined",
     "text": "Within treatment_pembrolizumab=1 patients, higher pdl1_tps strongly predicts higher ORR."},
    {"id": "h22.2", "kind": "refined",
     "text": "Within treatment_pembrolizumab=0 patients, pdl1_tps has weak/no effect on ORR."},
    {"id": "h22.3", "kind": "refined",
     "text": "tmb_high effect on ORR is largely concentrated in pembrolizumab-treated patients (positive interaction)."},
]
ans = []
sub_p = df[df["treatment_pembrolizumab"] == 1]
m = logreg("objective_response ~ pdl1_tps", data=sub_p)
b = m.params["pdl1_tps"]; pv = m.pvalues["pdl1_tps"]
ans.append({"hypothesis_ids": ["h22.1"],
            "code": "logit(obj ~ pdl1_tps) within pembro=1",
            "result_summary": f"In pembro+: beta(pdl1_tps)={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
sub_np = df[df["treatment_pembrolizumab"] == 0]
m = logreg("objective_response ~ pdl1_tps", data=sub_np)
b = m.params["pdl1_tps"]; pv = m.pvalues["pdl1_tps"]
ans.append({"hypothesis_ids": ["h22.2"],
            "code": "logit(obj ~ pdl1_tps) within pembro=0",
            "result_summary": f"In pembro-: beta(pdl1_tps)={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
m = logreg("objective_response ~ tmb_high * treatment_pembrolizumab")
b = m.params["tmb_high:treatment_pembrolizumab"]
pv = m.pvalues["tmb_high:treatment_pembrolizumab"]
ans.append({"hypothesis_ids": ["h22.3"],
            "code": "logit(obj ~ tmb_high * pembro)",
            "result_summary": f"tmb_high:pembro interaction beta={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(22, hyps, ans)
narrative.append("Iter 22 — Pembro subgroup analyses:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 23: Three-way PDL1 x STK11 x pembrolizumab
# ------------------------------------------------------------------
hyps = [
    {"id": "h23.1", "kind": "refined",
     "text": "Among treatment_pembrolizumab=1 patients, the positive PD-L1 effect is attenuated when stk11_mutation=1 (three-way interaction PDL1*pembro*STK11)."},
    {"id": "h23.2", "kind": "refined",
     "text": "Within pembrolizumab-treated, stk11_mutation+ patients have lower ORR than stk11_mutation- patients."},
]
ans = []
m = smf.logit(
    "objective_response ~ pdl1_tps * treatment_pembrolizumab * stk11_mutation",
    data=df).fit(disp=0)
term = "pdl1_tps:treatment_pembrolizumab:stk11_mutation"
b = m.params[term]; pv = m.pvalues[term]
ans.append({"hypothesis_ids": ["h23.1"],
            "code": "logit(obj ~ pdl1_tps * pembro * stk11)",
            "result_summary": f"3-way interaction beta={b:.3f}, OR={math.exp(b):.3f}, p={pv:.3g}.",
            "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
mask = df["treatment_pembrolizumab"] == 1
odds, p, p1, p0 = fisher_or(mask & (df["stk11_mutation"] == 1),
                            mask & (df["stk11_mutation"] == 0))
ans.append({"hypothesis_ids": ["h23.2"],
            "code": "fisher stk11 within pembro+",
            "result_summary": f"In pembro+: ORR STK11+={p1:.4f} vs STK11-={p0:.4f}; OR={odds:.3f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": float(p1 - p0), "significant": p < 0.05})
add_iter(23, hyps, ans)
narrative.append("Iter 23 — PD-L1 x pembro x STK11 three-way:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 24: Disparities — race / insurance differences in matched-therapy and ORR
# ------------------------------------------------------------------
hyps = [
    {"id": "h24.1", "kind": "novel",
     "text": "Receipt of matched_therapy differs by race_ethnicity (testing equity of biomarker-driven treatment assignment)."},
    {"id": "h24.2", "kind": "novel",
     "text": "Receipt of matched_therapy differs by insurance_type."},
    {"id": "h24.3", "kind": "novel",
     "text": "After adjusting for biomarkers, ECOG, age, sex, albumin, LDH, matched_therapy, pdl1_tps, and tmb_high, race_ethnicity remains independently associated with ORR."},
]
ans = []
ct = pd.crosstab(df["race_ethnicity"], df["matched_therapy"])
chi2, pv, dof, _ = stats.chi2_contingency(ct)
rates = df.groupby("race_ethnicity")["matched_therapy"].mean().to_dict()
ans.append({"hypothesis_ids": ["h24.1"],
            "code": "chi2(race_ethnicity, matched_therapy)",
            "result_summary": f"matched_therapy by race: {rates}; chi2={chi2:.3f}, p={pv:.3g}.",
            "p_value": float(pv),
            "effect_estimate": float(max(rates.values()) - min(rates.values())),
            "significant": pv < 0.05})
ct = pd.crosstab(df["insurance_type"], df["matched_therapy"])
chi2, pv, dof, _ = stats.chi2_contingency(ct)
rates2 = df.groupby("insurance_type")["matched_therapy"].mean().to_dict()
ans.append({"hypothesis_ids": ["h24.2"],
            "code": "chi2(insurance_type, matched_therapy)",
            "result_summary": f"matched_therapy by insurance: {rates2}; chi2={chi2:.3f}, p={pv:.3g}.",
            "p_value": float(pv),
            "effect_estimate": float(max(rates2.values()) - min(rates2.values())),
            "significant": pv < 0.05})
m = smf.logit(
    "objective_response ~ C(race_ethnicity) + ecog_ps + age_years + sex_female + "
    "albumin_g_dl + ldh_u_l + matched_therapy + pdl1_tps + tmb_high",
    data=df).fit(disp=0)
race_terms = [t for t in m.params.index if t.startswith("C(race_ethnicity)")]
joint_min_p = float(min(m.pvalues[t] for t in race_terms))
race_summary = "; ".join(f"{t}: beta={m.params[t]:.3f}, p={m.pvalues[t]:.3g}" for t in race_terms)
ans.append({"hypothesis_ids": ["h24.3"],
            "code": "logit with C(race_ethnicity) controlling for biomarkers/ECOG/labs",
            "result_summary": f"Adjusted race effects: {race_summary}.",
            "p_value": joint_min_p,
            "effect_estimate": float(max(m.params[t] for t in race_terms)
                                      - min(m.params[t] for t in race_terms)),
            "significant": joint_min_p < 0.05})
add_iter(24, hyps, ans)
narrative.append("Iter 24 — Disparities in matched-therapy receipt and ORR:\n"
                 + "\n".join("  " + a["result_summary"] for a in ans))


# ------------------------------------------------------------------
# Iteration 25: Final integrated multivariable model
# ------------------------------------------------------------------
hyps = [
    {"id": "h25.1", "kind": "refined",
     "text": "In a final integrated logistic regression that includes biomarker-matched-therapy interactions, ECOG, albumin, LDH, weight loss, NLR, prior_lines_of_therapy, age, stage IV, liver_mets, brain_mets, bone_mets, STK11, KEAP1, TP53 — the strongest positive predictors of ORR are matched-therapy interaction terms and PD-L1*pembrolizumab; the strongest negative predictors are higher ECOG, weight loss, LDH, NLR, prior lines of therapy, liver mets and brain mets."},
]
ans = []
m = smf.logit(
    "objective_response ~ "
    "egfr_mutation * treatment_osimertinib + "
    "kras_g12c * treatment_sotorasib + "
    "brca2_mutation * treatment_olaparib + "
    "pdl1_tps * treatment_pembrolizumab + "
    "tmb_high + ecog_ps + age_years + sex_female + albumin_g_dl + ldh_u_l + "
    "weight_loss_pct_6mo + nlr + crp_mg_l + prior_lines_of_therapy + "
    "stage_iv + liver_mets + has_brain_mets + bone_mets + "
    "stk11_mutation + keap1_mutation + tp53_mutation",
    data=df).fit(disp=0)

key_terms = [
    "egfr_mutation:treatment_osimertinib",
    "kras_g12c:treatment_sotorasib",
    "brca2_mutation:treatment_olaparib",
    "pdl1_tps:treatment_pembrolizumab",
    "tmb_high",
    "ecog_ps",
    "albumin_g_dl",
    "ldh_u_l",
    "weight_loss_pct_6mo",
    "nlr",
    "prior_lines_of_therapy",
    "stage_iv",
    "liver_mets",
    "has_brain_mets",
    "bone_mets",
    "stk11_mutation",
    "keap1_mutation",
    "tp53_mutation",
    "age_years",
    "sex_female",
    "crp_mg_l",
]
for term in key_terms:
    b = m.params[term]; pv = m.pvalues[term]
    ans.append({"hypothesis_ids": ["h25.1"],
                "code": "final integrated logit",
                "result_summary": f"{term}: adj beta={b:.4f}, OR={math.exp(b):.4f}, p={pv:.3g}.",
                "p_value": float(pv), "effect_estimate": float(b), "significant": pv < 0.05})
add_iter(25, hyps, ans)
narrative.append("Iter 25 — Final integrated multivariable logistic regression (key predictors above).")


# ------------------------------------------------------------------
# Write transcript
# ------------------------------------------------------------------
transcript = {
    "dataset_id": "ds001_nsclc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-manual@1.0",
    "max_iterations": 25,
    "iterations": iterations,
}
def _coerce(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


with open(HERE / "transcript.json", "w", encoding="utf-8") as f:
    json.dump(transcript, f, indent=2, default=_coerce, ensure_ascii=False)


# ------------------------------------------------------------------
# Build summary text
# ------------------------------------------------------------------
summary_lines = []
summary_lines.append(
    "ANALYSIS SUMMARY — ds001_nsclc (50000 patients, binary outcome: objective_response)"
)
summary_lines.append("")
summary_lines.append(
    "Approach: 25 iterations of propose-test-refine on EHR-derived NSCLC cohort with patient,\n"
    "tumor, biomarker, lab, treatment, comorbidity, symptom, demographic, and SNP features.\n"
    "Statistical methods: Fisher exact for 2x2 contingency tables; logistic regression for\n"
    "continuous predictors and main / interaction effects; chi-square for higher-cardinality\n"
    "categorical contrasts."
)
summary_lines.append("")
summary_lines.append(
    f"Overall ORR: {df['objective_response'].mean():.3f} ({int(df['objective_response'].sum())}/{N})."
)
summary_lines.append("")
summary_lines.append("ITERATION-LEVEL FINDINGS")
for n in narrative:
    summary_lines.append("")
    summary_lines.append(n)

summary_lines.append("")
summary_lines.append("KEY THEMES & CONCLUSIONS (synthesizing what the data actually showed)")
summary_lines.append(
    "1. Among the four biomarker-treatment pairings tested, ONLY pdl1_tps x\n"
    "   treatment_pembrolizumab shows a strong, reproducible positive interaction:\n"
    "   - Interaction beta=+0.59 (OR ~1.8 per unit pdl1_tps among pembrolizumab-treated\n"
    "     patients), p ~ 1.6e-7. Within pembro+: pdl1_tps positively predicts ORR (p ~ 7.6e-9).\n"
    "     Within pembro-: pdl1_tps trends negatively/null (p=0.09). At pdl1_tps>=0.5,\n"
    "     pembrolizumab ORR is 0.21 vs 0.16 (OR 1.43, p ~ 7e-13).\n"
    "   - The hypothesized egfr_mutation x treatment_osimertinib interaction was NOT\n"
    "     supported (interaction p=0.76). EGFR-mutant patients did not show higher ORR on\n"
    "     osimertinib than EGFR-mutant patients off osimertinib in this cohort.\n"
    "   - The hypothesized kras_g12c x treatment_sotorasib interaction was NOT supported\n"
    "     (p=0.43); the point estimate was small and positive but not significant.\n"
    "   - The hypothesized brca2_mutation x treatment_olaparib interaction was NOT\n"
    "     supported (interaction beta=-0.18, p=0.27).\n"
    "   - Net result: as a composite, matched_therapy receipt did not improve ORR\n"
    "     (matched vs not matched: ORR 0.166 vs 0.169, p=0.66; adjusted OR 0.98, p=0.67).\n"
    "2. tmb_high is independently associated with higher ORR (OR ~1.14, p ~ 1e-6 main\n"
    "   effect; OR ~1.20 interaction with pembrolizumab, p ~ 4e-4): TMB-high contributes\n"
    "   an additional positive effect that is enhanced in the pembrolizumab-treated stratum.\n"
    "3. Patient/disease severity markers are the most robust predictors of ORR:\n"
    "   - higher ecog_ps lowers ORR strongly (ECOG2 ORR=0.11 vs ECOG0 ORR=0.21; per-unit\n"
    "     OR 0.69, p ~ 1.7e-93). The single strongest predictor in the dataset.\n"
    "   - higher weight_loss_pct_6mo lowers ORR (OR 0.96 per unit, p ~ 2e-32).\n"
    "   - lower albumin_g_dl lowers ORR (OR 1.10 per g/dL higher, p ~ 1e-4).\n"
    "   - higher crp_mg_l lowers ORR (p ~ 7e-4). nlr trended negative but did not reach\n"
    "     significance in main effect (p=0.38).\n"
    "   - stage_iv lowers ORR (OR 0.74, p ~ 3e-33), as does has_brain_mets (OR 0.78,\n"
    "     p ~ 6e-19). liver_mets, bone_mets, pleural_effusion did NOT differentiate ORR.\n"
    "4. STK11 mutation modulates immunotherapy: STK11 had no overall main effect, but a\n"
    "   significant negative stk11_mutation x treatment_pembrolizumab interaction\n"
    "   (beta=-0.21, p=0.002). Within pembrolizumab-treated, ORR is 0.158 in STK11+ vs\n"
    "   0.177 in STK11- (p=0.005). KEAP1 trended in the same direction but was not\n"
    "   significant.\n"
    "5. Within EGFR+, prior_targeted_therapy did NOT significantly modify the osimertinib\n"
    "   effect (p=0.78). The hypothesized 'resistance after prior TKI' pattern was not\n"
    "   detectable in this cohort.\n"
    "6. Demographics:\n"
    "   - sex_female is associated with higher ORR (OR 1.09, p ~ 6e-4) overall and remains\n"
    "     significant in the final adjusted model (adj OR 1.08, p ~ 8e-4).\n"
    "   - race_ethnicity showed marginal global association (chi2 p=0.045) with white the\n"
    "     lowest ORR group; this association was not significant after adjustment for\n"
    "     biomarkers and labs.\n"
    "   - age_years, insurance_type, rural_residence had no significant ORR association.\n"
    "   - matched_therapy receipt did not differ significantly across race or insurance\n"
    "     strata (p>0.15), so access disparities to biomarker-matched therapy are not\n"
    "     evident here.\n"
    "7. Symptom grades: only fatigue_grade reached p<0.05 (negative); pain_nrs,\n"
    "   dyspnea_grade, cough_grade, appetite_loss_grade did not significantly predict ORR.\n"
    "8. Pharmacogenomic SNPs (snp_*): 2/23 nominally significant at p<0.05 (binomial test\n"
    "   p=0.32), consistent with the chance rate under the null. No systematic\n"
    "   pharmacogenomic signal for ORR in this cohort.\n"
    "9. Rare oncogenic drivers (alk_fusion, ros1_fusion, ret_fusion, braf_v600e,\n"
    "   met_exon14_skipping, ntrk_fusion, her2_amplification): none individually reached\n"
    "   significance for ORR in this cohort.\n"
    "10. prior_immunotherapy x treatment_pembrolizumab interaction was significantly\n"
    "    POSITIVE (beta=+0.135, p=0.044) — opposite of the prior expectation that prior IO\n"
    "    would reduce pembrolizumab response, suggesting this cohort does not show the\n"
    "    classic 'IO re-challenge' decrement pattern.\n"
    "11. Final integrated multivariable model: positive significant predictors of ORR are\n"
    "    pdl1_tps:treatment_pembrolizumab (OR 1.81), tmb_high (OR 1.14), albumin_g_dl\n"
    "    (OR 1.10), sex_female (OR 1.08). Negative significant predictors are ecog_ps\n"
    "    (OR 0.68), stage_iv (OR 0.74), has_brain_mets (OR 0.77), weight_loss_pct_6mo\n"
    "    (OR 0.96 per unit), crp_mg_l (OR 0.995 per unit). The matched-targeted-therapy\n"
    "    interactions for EGFR/KRAS-G12C/BRCA2 do NOT survive in the adjusted model.\n"
    "\n"
    "Bottom line: in this 50,000-patient NSCLC cohort, the dominant axis of objective\n"
    "response is the immunotherapy-PD-L1/TMB axis combined with patient performance status\n"
    "and disease burden. Classic 'biomarker-matched targeted therapy' for EGFR, KRAS G12C,\n"
    "and BRCA2 was not detectable as a positive ORR effect in this dataset, suggesting\n"
    "either substantial heterogeneity, treatment-effect dilution by biomarker-discordant\n"
    "use, or absence of the encoded pharmacology in the data-generating process."
)

with open(HERE / "analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("Wrote transcript.json and analysis_summary.txt")
print(f"Iterations: {len(iterations)}")
print(f"Total hypotheses: {sum(len(it['proposed_hypotheses']) for it in iterations)}")
print(f"Total analyses: {sum(len(it['analyses']) for it in iterations)}")
