"""Final confirmatory: best vs complement enzalutamide subgroup, and other treatments in their target subgroups."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

ROOT = Path(r"C:\Users\klkehl\are_llms_biased\data\ds001\tasks\prostate\named")
df = pd.read_parquet(ROOT / "dataset.parquet")
y = df["objective_response"].astype(int)


def proportion_test(yes_in, n_in, yes_out, n_out):
    p1 = yes_in / max(n_in, 1)
    p0 = yes_out / max(n_out, 1)
    pooled = (yes_in + yes_out) / max(n_in + n_out, 1)
    se = math.sqrt(pooled * (1 - pooled) * (1 / max(n_in, 1) + 1 / max(n_out, 1))) if pooled not in (0, 1) else 0
    if se == 0:
        return p1 - p0, 1.0
    z = (p1 - p0) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return p1 - p0, p


def stratum(name, mask, t):
    sub = df[mask]
    yi = int(((sub[t] == 1) & (sub["objective_response"] == 1)).sum())
    ni = int((sub[t] == 1).sum())
    yo = int(((sub[t] == 0) & (sub["objective_response"] == 1)).sum())
    no = int((sub[t] == 0).sum())
    if ni == 0 or no == 0:
        return None
    diff, p = proportion_test(yi, ni, yo, no)
    return {
        "label": name, "treatment": t,
        "n_on": ni, "n_off": no, "rate_on": yi / ni, "rate_off": yo / no,
        "diff": diff, "p": p,
    }


out = {}

# --- Enzalutamide best 4-feature subgroup ---
best_mask = (df["mcrpc"] == 0) & (df["ar_v7_positive"] == 0) & (df["brca2_mutation"] == 0) & (df["msi_high"] == 0)
out["enz_best4"] = stratum("mcrpc=0 & ar_v7_neg & brca2_wt & msi_neg", best_mask, "treatment_enzalutamide")
out["enz_complement_of_best4"] = stratum("complement of best 4 (any unfavorable)", ~best_mask, "treatment_enzalutamide")

# Add PSA constraint
best_mask_psa = best_mask & (df["psa_ng_ml"] <= 15.86)
out["enz_best4_lowpsa"] = stratum("best4 + psa<=15.86", best_mask_psa, "treatment_enzalutamide")

# --- Olaparib in BRCA2+ subgroup definitions ---
bm = (df["brca2_mutation"] == 1)
out["olaparib_brca2pos"] = stratum("brca2_mutation==1", bm, "treatment_olaparib")
out["olaparib_brca2pos_arv7neg"] = stratum("brca2+ & ar_v7_neg", bm & (df["ar_v7_positive"] == 0), "treatment_olaparib")
out["olaparib_brca2pos_msi_low"] = stratum("brca2+ & msi_low", bm & (df["msi_high"] == 0), "treatment_olaparib")

# --- Pembro in MSI-high subgroups ---
mh = (df["msi_high"] == 1)
out["pembro_msihigh"] = stratum("msi_high==1", mh, "treatment_pembrolizumab")
out["pembro_msihigh_nonmcrpc"] = stratum("msi_high & mcrpc=0", mh & (df["mcrpc"] == 0), "treatment_pembrolizumab")
out["pembro_msihigh_arv7neg"] = stratum("msi_high & ar_v7_neg", mh & (df["ar_v7_positive"] == 0), "treatment_pembrolizumab")

# --- Lu177-PSMA in PSMA-high subgroups ---
ph = (df["psma_high"] == 1)
out["lu177_psmahigh"] = stratum("psma_high==1", ph, "treatment_lu177_psma")
out["lu177_psmahigh_nonmcrpc"] = stratum("psma_high & mcrpc=0", ph & (df["mcrpc"] == 0), "treatment_lu177_psma")
out["lu177_psmahigh_visceral0"] = stratum("psma_high & visceral=0", ph & (df["visceral_mets"] == 0), "treatment_lu177_psma")

# --- Final full multivariable logistic regression with key interactions ---
features = [
    "treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
    "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab",
    "age_years", "ecog_ps", "mcrpc", "visceral_mets", "psa_ng_ml",
    "gleason_score", "brca2_mutation", "ar_v7_positive", "msi_high", "psma_high",
    "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr",
    "hemoglobin_g_dl", "alkaline_phosphatase_u_l",
]
X = df[features].copy()
X["enz_x_arv7"] = df["treatment_enzalutamide"] * df["ar_v7_positive"]
X["enz_x_brca2"] = df["treatment_enzalutamide"] * df["brca2_mutation"]
X["enz_x_msi"] = df["treatment_enzalutamide"] * df["msi_high"]
X["enz_x_mcrpc"] = df["treatment_enzalutamide"] * df["mcrpc"]
X["enz_x_psa"] = df["treatment_enzalutamide"] * df["psa_ng_ml"]
Xc = sm.add_constant(X)
mod = sm.Logit(y, Xc).fit(disp=False, maxiter=400)
out["multivar_model"] = {}
for c in mod.params.index:
    out["multivar_model"][c] = {
        "coef": float(mod.params[c]),
        "p": float(mod.pvalues[c]),
        "or": float(np.exp(mod.params[c])),
    }

# --- Test enz benefit within complement-of-best subgroup (should be null/small) ---
for ablation in [
    ("drop mcrpc=0 only (mcrpc=1)", df["mcrpc"] == 1),
    ("drop ar_v7=0 only (ar_v7_positive=1)", df["ar_v7_positive"] == 1),
    ("drop brca2=0 only (brca2_mutation=1)", df["brca2_mutation"] == 1),
    ("drop msi=0 only (msi_high=1)", df["msi_high"] == 1),
]:
    out[f"enz_in_{ablation[0]}"] = stratum(ablation[0], ablation[1], "treatment_enzalutamide")

with open(ROOT / "_my_analysis" / "results_final.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

print("ENZA best subgroup:")
print(out["enz_best4"])
print("ENZA complement:")
print(out["enz_complement_of_best4"])
print()
print("Sub-strata where enz fails (single negative modifier):")
for k in [
    "enz_in_drop mcrpc=0 only (mcrpc=1)",
    "enz_in_drop ar_v7=0 only (ar_v7_positive=1)",
    "enz_in_drop brca2=0 only (brca2_mutation=1)",
    "enz_in_drop msi=0 only (msi_high=1)",
]:
    print(k, out[k])
print()
print("Olaparib in BRCA2+:", out["olaparib_brca2pos"])
print("Pembro in MSI-high:", out["pembro_msihigh"])
print("Lu177 in PSMA-high:", out["lu177_psmahigh"])

print()
print("Key multivariable coefs:")
for c in [
    "treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
    "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab",
    "enz_x_arv7", "enz_x_brca2", "enz_x_msi", "enz_x_mcrpc", "enz_x_psa",
]:
    info = out["multivar_model"][c]
    print(f"  {c:25s} coef={info['coef']:+.4f} OR={info['or']:.3f} p={info['p']:.3g}")
