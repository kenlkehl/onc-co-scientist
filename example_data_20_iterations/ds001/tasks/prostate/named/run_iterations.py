"""Iterative analysis of ds001_prostate cohort.

Runs up to 25 iterations of hypothesis -> analysis -> refinement, capturing
each in a transcript.json conforming to transcript_schema.json. Also emits
analysis_summary.txt at the end.
"""
from __future__ import annotations

import json
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")
OUT = "pfs_months"

ITERATIONS: list[dict[str, Any]] = []


def _round(x, n=4):
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        if math.isnan(float(x)) or math.isinf(float(x)):
            return None
        return round(float(x), n)
    except Exception:
        return None


def add_iteration(idx: int, hypotheses: list[dict], analyses: list[dict]):
    ITERATIONS.append({
        "index": idx,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses,
    })


def t_test(group_col: str, group_val=1, outcome: str = OUT, df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = DF
    a = df.loc[df[group_col] == group_val, outcome].astype(float).values
    b = df.loc[df[group_col] != group_val, outcome].astype(float).values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = float(np.mean(a) - np.mean(b))
    return {
        "n_in": int(len(a)),
        "n_out": int(len(b)),
        "mean_in": float(np.mean(a)),
        "mean_out": float(np.mean(b)),
        "diff": diff,
        "t": float(t),
        "p_value": float(p),
    }


def ols_fit(formula: str, df: pd.DataFrame | None = None):
    if df is None:
        df = DF
    return smf.ols(formula, data=df).fit()


def beta_p(model, term: str) -> tuple[float, float]:
    return float(model.params[term]), float(model.pvalues[term])


# ------------------- Iteration 1 ------------------- #
# Baseline: ECOG PS, mCRPC, visceral mets — should worsen PFS.

m1 = ols_fit(f"{OUT} ~ ecog_ps + mcrpc + visceral_mets + age_years")
beta_ecog, p_ecog = beta_p(m1, "ecog_ps")
beta_mcrpc, p_mcrpc = beta_p(m1, "mcrpc")
beta_vm, p_vm = beta_p(m1, "visceral_mets")
beta_age, p_age = beta_p(m1, "age_years")

add_iteration(
    1,
    [
        {"id": "h1.1", "text": "Higher ECOG performance status (ecog_ps) is associated with shorter pfs_months (negative coefficient).", "kind": "novel"},
        {"id": "h1.2", "text": "Patients with mCRPC (mcrpc=1) have shorter pfs_months than patients without mCRPC.", "kind": "novel"},
        {"id": "h1.3", "text": "Presence of visceral metastases (visceral_mets=1) is associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h1.4", "text": "Older age (age_years) is associated with shorter pfs_months.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h1.1"], "code": "smf.ols('pfs_months ~ ecog_ps + mcrpc + visceral_mets + age_years', df).fit()",
         "result_summary": f"OLS coefficient for ecog_ps = {beta_ecog:.3f} months per unit (p={p_ecog:.3g}).",
         "p_value": _round(p_ecog), "effect_estimate": _round(beta_ecog), "significant": p_ecog < 0.05},
        {"hypothesis_ids": ["h1.2"], "code": "same model, mcrpc term",
         "result_summary": f"OLS coefficient for mcrpc = {beta_mcrpc:.3f} months (p={p_mcrpc:.3g}).",
         "p_value": _round(p_mcrpc), "effect_estimate": _round(beta_mcrpc), "significant": p_mcrpc < 0.05},
        {"hypothesis_ids": ["h1.3"], "code": "same model, visceral_mets term",
         "result_summary": f"OLS coefficient for visceral_mets = {beta_vm:.3f} months (p={p_vm:.3g}).",
         "p_value": _round(p_vm), "effect_estimate": _round(beta_vm), "significant": p_vm < 0.05},
        {"hypothesis_ids": ["h1.4"], "code": "same model, age_years term",
         "result_summary": f"OLS coefficient for age_years = {beta_age:.4f} months/year (p={p_age:.3g}).",
         "p_value": _round(p_age), "effect_estimate": _round(beta_age), "significant": p_age < 0.05},
    ],
)
print("Iter1 done")

# ------------------- Iteration 2 ------------------- #
# Lab markers: PSA (log), LDH, albumin, hemoglobin
DF["log_psa"] = np.log1p(DF["psa_ng_ml"])
DF["log_ldh"] = np.log(DF["ldh_u_l"])

m2 = ols_fit(f"{OUT} ~ log_psa + log_ldh + albumin_g_dl + hemoglobin_g_dl + alkaline_phosphatase_u_l")
b_psa, p_psa = beta_p(m2, "log_psa")
b_ldh, p_ldh = beta_p(m2, "log_ldh")
b_alb, p_alb = beta_p(m2, "albumin_g_dl")
b_hgb, p_hgb = beta_p(m2, "hemoglobin_g_dl")
b_alp, p_alp = beta_p(m2, "alkaline_phosphatase_u_l")

add_iteration(
    2,
    [
        {"id": "h2.1", "text": "Higher log(PSA) is associated with shorter pfs_months (negative coefficient).", "kind": "novel"},
        {"id": "h2.2", "text": "Higher LDH (log_ldh) is associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h2.3", "text": "Higher albumin_g_dl is associated with longer pfs_months (positive coefficient).", "kind": "novel"},
        {"id": "h2.4", "text": "Higher hemoglobin_g_dl is associated with longer pfs_months.", "kind": "novel"},
        {"id": "h2.5", "text": "Higher alkaline_phosphatase_u_l is associated with shorter pfs_months.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h2.1"], "code": "OLS pfs ~ log_psa + log_ldh + albumin + hgb + alp",
         "result_summary": f"log_psa beta={b_psa:.4f} months (p={p_psa:.3g}).", "p_value": _round(p_psa), "effect_estimate": _round(b_psa), "significant": p_psa<0.05},
        {"hypothesis_ids": ["h2.2"], "code": "same model",
         "result_summary": f"log_ldh beta={b_ldh:.4f} months (p={p_ldh:.3g}).", "p_value": _round(p_ldh), "effect_estimate": _round(b_ldh), "significant": p_ldh<0.05},
        {"hypothesis_ids": ["h2.3"], "code": "same model",
         "result_summary": f"albumin_g_dl beta={b_alb:.4f} months/(g/dL) (p={p_alb:.3g}).", "p_value": _round(p_alb), "effect_estimate": _round(b_alb), "significant": p_alb<0.05},
        {"hypothesis_ids": ["h2.4"], "code": "same model",
         "result_summary": f"hemoglobin_g_dl beta={b_hgb:.4f} months/(g/dL) (p={p_hgb:.3g}).", "p_value": _round(p_hgb), "effect_estimate": _round(b_hgb), "significant": p_hgb<0.05},
        {"hypothesis_ids": ["h2.5"], "code": "same model",
         "result_summary": f"alkaline_phosphatase_u_l beta={b_alp:.5f} months/U/L (p={p_alp:.3g}).", "p_value": _round(p_alp), "effect_estimate": _round(b_alp), "significant": p_alp<0.05},
    ],
)
print("Iter2 done")

# ------------------- Iteration 3 ------------------- #
# Metastatic site main effects: liver, bone, adrenal, pleural effusion
m3 = ols_fit(f"{OUT} ~ liver_mets + bone_mets + adrenal_mets + visceral_mets + pleural_effusion")
res3 = {}
for term in ["liver_mets", "bone_mets", "adrenal_mets", "visceral_mets", "pleural_effusion"]:
    res3[term] = beta_p(m3, term)

add_iteration(
    3,
    [
        {"id": "h3.1", "text": "Liver metastases (liver_mets=1) are associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h3.2", "text": "Bone metastases (bone_mets=1) are associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h3.3", "text": "Adrenal metastases (adrenal_mets=1) are associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h3.4", "text": "Pleural effusion (pleural_effusion=1) is associated with shorter pfs_months.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": [f"h3.{i+1}"], "code": "OLS pfs ~ liver+bone+adrenal+visceral+pleural",
         "result_summary": f"{term} beta={res3[term][0]:.4f} (p={res3[term][1]:.3g}).",
         "p_value": _round(res3[term][1]), "effect_estimate": _round(res3[term][0]), "significant": res3[term][1]<0.05}
        for i, term in enumerate(["liver_mets", "bone_mets", "adrenal_mets", "pleural_effusion"])
    ],
)
print("Iter3 done")

# ------------------- Iteration 4 ------------------- #
# Treatment main effects (univariable, then mutually adjusted)
trt_cols = ["treatment_enzalutamide","treatment_abiraterone","treatment_docetaxel",
            "treatment_olaparib","treatment_lu177_psma","treatment_pembrolizumab"]
m4 = ols_fit(f"{OUT} ~ " + " + ".join(trt_cols))
res4 = {t: beta_p(m4, t) for t in trt_cols}

hyps4 = []
ans4 = []
for i,t in enumerate(trt_cols):
    hid = f"h4.{i+1}"
    drug = t.replace("treatment_","")
    hyps4.append({"id": hid, "text": f"Receiving {t}=1 is associated with longer pfs_months than not receiving it (positive coefficient when all treatments are mutually adjusted).", "kind": "novel"})
    b,p = res4[t]
    ans4.append({"hypothesis_ids":[hid], "code": "OLS pfs ~ all 6 treatment indicators",
                 "result_summary": f"{t} beta={b:.4f} months (p={p:.3g}).",
                 "p_value": _round(p), "effect_estimate": _round(b), "significant": p<0.05})
add_iteration(4, hyps4, ans4)
print("Iter4 done")

# ------------------- Iteration 5 ------------------- #
# Biomarker main effects on PFS (without treatment interactions)
biomarkers = ["brca2_mutation","ar_v7_positive","msi_high","psma_high","tp53_mutation","pten_loss"]
m5 = ols_fit(f"{OUT} ~ " + " + ".join(biomarkers))
res5 = {b: beta_p(m5, b) for b in biomarkers}
hyps5 = []
ans5 = []
descs = {
    "brca2_mutation": "BRCA2 mutation",
    "ar_v7_positive": "AR-V7 positivity",
    "msi_high": "MSI-high status",
    "psma_high": "PSMA-high expression",
    "tp53_mutation": "TP53 mutation",
    "pten_loss": "PTEN loss",
}
for i,b in enumerate(biomarkers):
    hid=f"h5.{i+1}"
    hyps5.append({"id":hid,"text":f"{descs[b]} ({b}=1) is associated with a difference in pfs_months in the overall cohort (any direction; tested two-sided).","kind":"novel"})
    bv,pv = res5[b]
    ans5.append({"hypothesis_ids":[hid],"code":"OLS pfs ~ 6 biomarkers",
                 "result_summary": f"{b} beta={bv:.4f} months (p={pv:.3g}).",
                 "p_value": _round(pv),"effect_estimate":_round(bv),"significant":pv<0.05})
add_iteration(5, hyps5, ans5)
print("Iter5 done")

# ------------------- Iteration 6 ------------------- #
# Olaparib x BRCA2 interaction (PARP inhibitor expected benefit in BRCA-mutant)
m6 = ols_fit(f"{OUT} ~ treatment_olaparib * brca2_mutation + ecog_ps + mcrpc + log_psa + log_ldh + albumin_g_dl")
b_int, p_int = beta_p(m6, "treatment_olaparib:brca2_mutation")
b_main, p_main = beta_p(m6, "treatment_olaparib")
# stratified means
strat = {}
for brca in [0,1]:
    for ola in [0,1]:
        sub = DF[(DF.brca2_mutation==brca)&(DF.treatment_olaparib==ola)][OUT]
        strat[(brca,ola)] = (float(sub.mean()), int(len(sub)))
diff_brca1 = strat[(1,1)][0]-strat[(1,0)][0]
diff_brca0 = strat[(0,1)][0]-strat[(0,0)][0]

add_iteration(
    6,
    [
        {"id":"h6.1","text":"There is a positive interaction between treatment_olaparib and brca2_mutation on pfs_months: the PFS benefit of olaparib is larger in BRCA2-mutated patients than in BRCA2 wild-type patients (positive interaction coefficient).","kind":"novel"},
        {"id":"h6.2","text":"Among patients with brca2_mutation=1, mean pfs_months is higher in those receiving treatment_olaparib than in those not receiving it (positive simple difference).","kind":"refined"},
    ],
    [
        {"hypothesis_ids":["h6.1"],
         "code":"OLS pfs ~ treatment_olaparib*brca2_mutation + clinical covariates",
         "result_summary": f"Interaction coef = {b_int:.4f} months (p={p_int:.3g}). Olaparib main effect (BRCA2=0) = {b_main:.4f} months (p={p_main:.3g}).",
         "p_value": _round(p_int),"effect_estimate":_round(b_int),"significant":p_int<0.05},
        {"hypothesis_ids":["h6.2"],
         "code": "stratified means: BRCA2+/Ola+ vs BRCA2+/Ola- ",
         "result_summary": f"BRCA2+ Ola+ mean PFS={strat[(1,1)][0]:.2f} (n={strat[(1,1)][1]}); BRCA2+ Ola- mean PFS={strat[(1,0)][0]:.2f} (n={strat[(1,0)][1]}); diff={diff_brca1:.3f}; BRCA2- diff (Ola+ - Ola-)={diff_brca0:.3f}.",
         "p_value": None, "effect_estimate": _round(diff_brca1), "significant": None},
    ],
)
print("Iter6 done")

# ------------------- Iteration 7 ------------------- #
# Pembrolizumab x MSI-high interaction
m7 = ols_fit(f"{OUT} ~ treatment_pembrolizumab * msi_high + ecog_ps + log_ldh + albumin_g_dl")
b_int7, p_int7 = beta_p(m7, "treatment_pembrolizumab:msi_high")
b_main7, p_main7 = beta_p(m7, "treatment_pembrolizumab")
strat7 = {}
for msi in [0,1]:
    for pem in [0,1]:
        sub = DF[(DF.msi_high==msi)&(DF.treatment_pembrolizumab==pem)][OUT]
        strat7[(msi,pem)] = (float(sub.mean()), int(len(sub)))
diff_msi1 = strat7[(1,1)][0]-strat7[(1,0)][0]
diff_msi0 = strat7[(0,1)][0]-strat7[(0,0)][0]

add_iteration(
    7,
    [
        {"id":"h7.1","text":"There is a positive interaction between treatment_pembrolizumab and msi_high on pfs_months: pembrolizumab's PFS benefit is larger in MSI-high patients (positive interaction coefficient).","kind":"novel"},
        {"id":"h7.2","text":"Among patients with msi_high=1, mean pfs_months is higher in those receiving treatment_pembrolizumab than in those not receiving it.","kind":"refined"},
    ],
    [
        {"hypothesis_ids":["h7.1"],"code":"OLS pfs ~ treatment_pembrolizumab*msi_high + covariates",
         "result_summary": f"Interaction coef={b_int7:.4f} (p={p_int7:.3g}). Pembrolizumab main (MSI=0) = {b_main7:.4f} (p={p_main7:.3g}).",
         "p_value": _round(p_int7),"effect_estimate":_round(b_int7),"significant":p_int7<0.05},
        {"hypothesis_ids":["h7.2"],"code":"stratified means MSI=1",
         "result_summary": f"MSI+ Pem+ mean PFS={strat7[(1,1)][0]:.2f} (n={strat7[(1,1)][1]}); MSI+ Pem- mean PFS={strat7[(1,0)][0]:.2f} (n={strat7[(1,0)][1]}); diff={diff_msi1:.3f}; MSI- diff={diff_msi0:.3f}.",
         "p_value": None,"effect_estimate":_round(diff_msi1),"significant":None},
    ],
)
print("Iter7 done")

# ------------------- Iteration 8 ------------------- #
# Lu177-PSMA x PSMA-high interaction
m8 = ols_fit(f"{OUT} ~ treatment_lu177_psma * psma_high + ecog_ps + log_ldh + log_psa + albumin_g_dl")
b_int8, p_int8 = beta_p(m8, "treatment_lu177_psma:psma_high")
b_main8, p_main8 = beta_p(m8, "treatment_lu177_psma")
strat8 = {}
for psma in [0,1]:
    for lu in [0,1]:
        sub = DF[(DF.psma_high==psma)&(DF.treatment_lu177_psma==lu)][OUT]
        strat8[(psma,lu)] = (float(sub.mean()), int(len(sub)))
diff_psma1 = strat8[(1,1)][0]-strat8[(1,0)][0]
diff_psma0 = strat8[(0,1)][0]-strat8[(0,0)][0]

add_iteration(
    8,
    [
        {"id":"h8.1","text":"There is a positive interaction between treatment_lu177_psma and psma_high on pfs_months: Lu177-PSMA's benefit is larger when psma_high=1 (positive interaction coefficient).","kind":"novel"},
        {"id":"h8.2","text":"Among psma_high=1 patients, mean pfs_months is higher in those receiving treatment_lu177_psma than not.","kind":"refined"},
    ],
    [
        {"hypothesis_ids":["h8.1"],"code":"OLS pfs ~ treatment_lu177_psma*psma_high + covariates",
         "result_summary": f"Interaction coef={b_int8:.4f} (p={p_int8:.3g}). Lu177 main (PSMA-low)={b_main8:.4f} (p={p_main8:.3g}).",
         "p_value": _round(p_int8),"effect_estimate":_round(b_int8),"significant":p_int8<0.05},
        {"hypothesis_ids":["h8.2"],"code":"stratified means PSMA-high",
         "result_summary": f"PSMA+ Lu177+ PFS={strat8[(1,1)][0]:.2f} (n={strat8[(1,1)][1]}); PSMA+ Lu177- PFS={strat8[(1,0)][0]:.2f} (n={strat8[(1,0)][1]}); diff={diff_psma1:.3f}; PSMA- diff={diff_psma0:.3f}.",
         "p_value": None,"effect_estimate":_round(diff_psma1),"significant":None},
    ],
)
print("Iter8 done")

# ------------------- Iteration 9 ------------------- #
# Enzalutamide x AR-V7 interaction (negative interaction expected -- AR-V7 confers resistance)
m9 = ols_fit(f"{OUT} ~ treatment_enzalutamide * ar_v7_positive + ecog_ps + log_psa + log_ldh + albumin_g_dl")
b_int9, p_int9 = beta_p(m9, "treatment_enzalutamide:ar_v7_positive")
b_main9, p_main9 = beta_p(m9, "treatment_enzalutamide")
# Same for abiraterone
m9b = ols_fit(f"{OUT} ~ treatment_abiraterone * ar_v7_positive + ecog_ps + log_psa + log_ldh + albumin_g_dl")
b_int9b, p_int9b = beta_p(m9b, "treatment_abiraterone:ar_v7_positive")
b_main9b, p_main9b = beta_p(m9b, "treatment_abiraterone")

strat9 = {}
for arv in [0,1]:
    for enz in [0,1]:
        sub = DF[(DF.ar_v7_positive==arv)&(DF.treatment_enzalutamide==enz)][OUT]
        strat9[(arv,enz)] = (float(sub.mean()), int(len(sub)))
diff_arv1 = strat9[(1,1)][0]-strat9[(1,0)][0]
diff_arv0 = strat9[(0,1)][0]-strat9[(0,0)][0]

add_iteration(
    9,
    [
        {"id":"h9.1","text":"There is a negative interaction between treatment_enzalutamide and ar_v7_positive on pfs_months: AR-V7 positivity reduces the benefit of enzalutamide (interaction coefficient is negative).","kind":"novel"},
        {"id":"h9.2","text":"There is a negative interaction between treatment_abiraterone and ar_v7_positive on pfs_months: AR-V7 positivity reduces the benefit of abiraterone (interaction coefficient is negative).","kind":"novel"},
        {"id":"h9.3","text":"Among ar_v7_positive=1 patients, the simple PFS difference (enzalutamide on minus off) is smaller (less positive or negative) than in AR-V7-negative patients.","kind":"refined"},
    ],
    [
        {"hypothesis_ids":["h9.1"],"code":"OLS pfs ~ enza*ar_v7 + covariates",
         "result_summary": f"Interaction coef={b_int9:.4f} (p={p_int9:.3g}). Enza main (AR-V7=0)={b_main9:.4f} (p={p_main9:.3g}).",
         "p_value": _round(p_int9),"effect_estimate":_round(b_int9),"significant":p_int9<0.05},
        {"hypothesis_ids":["h9.2"],"code":"OLS pfs ~ abi*ar_v7 + covariates",
         "result_summary": f"Interaction coef={b_int9b:.4f} (p={p_int9b:.3g}). Abi main (AR-V7=0)={b_main9b:.4f} (p={p_main9b:.3g}).",
         "p_value": _round(p_int9b),"effect_estimate":_round(b_int9b),"significant":p_int9b<0.05},
        {"hypothesis_ids":["h9.3"],"code":"stratified means by ar_v7",
         "result_summary": f"AR-V7+ Enza+ - Enza- diff={diff_arv1:.3f}; AR-V7- diff={diff_arv0:.3f}.",
         "p_value": None,"effect_estimate":_round(diff_arv1-diff_arv0),"significant":None},
    ],
)
print("Iter9 done")

# Save partial transcript checkpoint
def save_partial():
    transcript = {
        "dataset_id": "ds001_prostate",
        "model_id": "claude-opus-4-7",
        "harness_id": "claude-code@manual-iter",
        "max_iterations": 25,
        "iterations": ITERATIONS,
    }
    with open("transcript.json","w") as f:
        json.dump(transcript, f, indent=2, default=str)

save_partial()
print("Partial saved after 9 iterations")
