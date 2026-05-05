"""End-to-end analysis script for ds001_prostate. Produces results.json with iteration outputs."""
import json
import math
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import logit

ROOT = Path(r"C:\Users\klkehl\are_llms_biased\data\ds001\tasks\prostate\named")
df = pd.read_parquet(ROOT / "dataset.parquet")

OUTCOME = "objective_response"
TREATMENTS = [
    "treatment_enzalutamide",
    "treatment_abiraterone",
    "treatment_docetaxel",
    "treatment_olaparib",
    "treatment_lu177_psma",
    "treatment_pembrolizumab",
]
BINARY_FEATURES = [
    "mcrpc",
    "visceral_mets",
    "brca2_mutation",
    "ar_v7_positive",
    "msi_high",
    "psma_high",
]
CONTINUOUS_FEATURES = [
    "age_years", "ecog_ps", "psa_ng_ml", "gleason_score",
    "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr",
    "hemoglobin_g_dl", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
    "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
    "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl",
]

results = {}


def proportion_test(yes_in, n_in, yes_out, n_out):
    """Two-proportion z-test return (rate_in - rate_out, p)."""
    p1 = yes_in / max(n_in, 1)
    p0 = yes_out / max(n_out, 1)
    pooled = (yes_in + yes_out) / max(n_in + n_out, 1)
    se = math.sqrt(pooled * (1 - pooled) * (1 / max(n_in, 1) + 1 / max(n_out, 1))) if pooled not in (0, 1) else 0
    if se == 0:
        return p1 - p0, 1.0
    z = (p1 - p0) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return p1 - p0, p


# === ITER 1: Univariable treatment main effects on objective_response ===
iter1 = []
for t in TREATMENTS:
    yes_in = int(((df[t] == 1) & (df[OUTCOME] == 1)).sum())
    n_in = int((df[t] == 1).sum())
    yes_out = int(((df[t] == 0) & (df[OUTCOME] == 1)).sum())
    n_out = int((df[t] == 0).sum())
    diff, p = proportion_test(yes_in, n_in, yes_out, n_out)
    iter1.append({
        "treatment": t,
        "rate_on": yes_in / n_in,
        "rate_off": yes_out / n_out,
        "diff": diff,
        "p": p,
        "n_on": n_in,
        "n_off": n_out,
    })
results["iter1_treatment_main_effects"] = iter1

# === ITER 2: Adjusted treatment effects via multivariable logistic regression ===
X = df[TREATMENTS + BINARY_FEATURES + CONTINUOUS_FEATURES].copy()
y = df[OUTCOME].astype(int)
Xc = sm.add_constant(X)
mod_full = sm.Logit(y, Xc).fit(disp=False, maxiter=200)
iter2 = []
for c in TREATMENTS + BINARY_FEATURES + CONTINUOUS_FEATURES:
    iter2.append({
        "var": c,
        "coef": float(mod_full.params[c]),
        "p": float(mod_full.pvalues[c]),
        "or": float(np.exp(mod_full.params[c])),
    })
results["iter2_multivar_logit"] = iter2

# === ITER 3: Prognostic feature univariable analyses (rate diffs / OR) ===
iter3 = []
for f in BINARY_FEATURES:
    yi = int(((df[f] == 1) & (df[OUTCOME] == 1)).sum())
    ni = int((df[f] == 1).sum())
    yo = int(((df[f] == 0) & (df[OUTCOME] == 1)).sum())
    no = int((df[f] == 0).sum())
    diff, p = proportion_test(yi, ni, yo, no)
    iter3.append({"feature": f, "rate_pos": yi / ni, "rate_neg": yo / no, "diff": diff, "p": p})
for f in CONTINUOUS_FEATURES:
    g1 = df.loc[df[OUTCOME] == 1, f]
    g0 = df.loc[df[OUTCOME] == 0, f]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    iter3.append({
        "feature": f,
        "mean_resp": float(g1.mean()),
        "mean_nonresp": float(g0.mean()),
        "diff": float(g1.mean() - g0.mean()),
        "p": float(t.pvalue),
    })
results["iter3_feature_main_effects"] = iter3

# === ITER 4: Treatment x biomarker interaction screen (logistic with interaction) ===
# Build a per-treatment, per-binary-feature interaction model adjusted for other treatments and key prognostics
adj_covs = [
    "age_years", "ecog_ps", "mcrpc", "visceral_mets", "psa_ng_ml",
    "gleason_score", "albumin_g_dl", "ldh_u_l", "hemoglobin_g_dl",
]
iter4 = []
for t in TREATMENTS:
    other_t = [o for o in TREATMENTS if o != t]
    for f in BINARY_FEATURES:
        Xi = df[[t, f] + other_t + adj_covs].copy()
        Xi[f"{t}_x_{f}"] = df[t] * df[f]
        Xc2 = sm.add_constant(Xi)
        try:
            mi = sm.Logit(y, Xc2).fit(disp=False, maxiter=200)
            iter4.append({
                "treatment": t,
                "feature": f,
                "main_t_coef": float(mi.params[t]),
                "main_t_p": float(mi.pvalues[t]),
                "main_f_coef": float(mi.params[f]),
                "main_f_p": float(mi.pvalues[f]),
                "interaction_coef": float(mi.params[f"{t}_x_{f}"]),
                "interaction_p": float(mi.pvalues[f"{t}_x_{f}"]),
                "interaction_or": float(np.exp(mi.params[f"{t}_x_{f}"])),
            })
        except Exception as e:
            iter4.append({"treatment": t, "feature": f, "error": str(e)})
results["iter4_treatment_binary_feature_interactions"] = iter4

# === ITER 5: Stratified treatment effects within each binary biomarker ===
iter5 = []
for t in TREATMENTS:
    for f in BINARY_FEATURES:
        for fv in [0, 1]:
            sub = df[df[f] == fv]
            yi = int(((sub[t] == 1) & (sub[OUTCOME] == 1)).sum())
            ni = int((sub[t] == 1).sum())
            yo = int(((sub[t] == 0) & (sub[OUTCOME] == 1)).sum())
            no = int((sub[t] == 0).sum())
            if ni == 0 or no == 0:
                continue
            diff, p = proportion_test(yi, ni, yo, no)
            iter5.append({
                "treatment": t, "feature": f, "feature_value": fv,
                "rate_on": yi / ni, "rate_off": yo / no, "diff": diff, "p": p,
                "n_on": ni, "n_off": no,
            })
results["iter5_stratified_tx_effects_by_binary"] = iter5


# === ITER 6: Treatment x continuous feature interactions (logit) ===
iter6 = []
for t in TREATMENTS:
    other_t = [o for o in TREATMENTS if o != t]
    for f in CONTINUOUS_FEATURES:
        Xi = df[[t, f] + other_t + [c for c in adj_covs if c != f]].copy()
        Xi[f"{t}_x_{f}"] = df[t] * df[f]
        Xc2 = sm.add_constant(Xi)
        try:
            mi = sm.Logit(y, Xc2).fit(disp=False, maxiter=200)
            iter6.append({
                "treatment": t, "feature": f,
                "interaction_coef": float(mi.params[f"{t}_x_{f}"]),
                "interaction_p": float(mi.pvalues[f"{t}_x_{f}"]),
            })
        except Exception as e:
            iter6.append({"treatment": t, "feature": f, "error": str(e)})
results["iter6_treatment_continuous_interactions"] = iter6


# === ITER 7: Per-treatment exhaustive 2-binary-feature subgroup screen (top hits per tx) ===
def subgroup_diff(sub):
    yi = int(((sub[t] == 1) & (sub[OUTCOME] == 1)).sum())
    ni = int((sub[t] == 1).sum())
    yo = int(((sub[t] == 0) & (sub[OUTCOME] == 1)).sum())
    no = int((sub[t] == 0).sum())
    if ni < 30 or no < 30:
        return None
    diff, p = proportion_test(yi, ni, yo, no)
    return ni, no, yi / ni, yo / no, diff, p


iter7 = {}
for t in TREATMENTS:
    rows = []
    for (f1, f2) in combinations(BINARY_FEATURES, 2):
        for v1 in (0, 1):
            for v2 in (0, 1):
                sub = df[(df[f1] == v1) & (df[f2] == v2)]
                r = subgroup_diff(sub)
                if r is None:
                    continue
                ni, no, ron, roff, diff, p = r
                rows.append({
                    "f1": f1, "v1": v1, "f2": f2, "v2": v2,
                    "n_on": ni, "n_off": no, "rate_on": ron, "rate_off": roff,
                    "diff": diff, "p": p,
                })
    rows.sort(key=lambda r: r["p"])
    iter7[t] = rows[:10]
results["iter7_two_binary_feature_subgroups_top10"] = iter7


# === ITER 8: Specific hypothesis tests ===
# olaparib in BRCA2-mutant patients
def test_subgroup(tname, mask, label):
    sub = df[mask]
    yi = int(((sub[tname] == 1) & (sub[OUTCOME] == 1)).sum())
    ni = int((sub[tname] == 1).sum())
    yo = int(((sub[tname] == 0) & (sub[OUTCOME] == 1)).sum())
    no = int((sub[tname] == 0).sum())
    if ni == 0 or no == 0:
        return None
    diff, p = proportion_test(yi, ni, yo, no)
    return {
        "treatment": tname, "subgroup": label,
        "n_on": ni, "n_off": no, "rate_on": yi / ni, "rate_off": yo / no,
        "diff": diff, "p": p,
    }


iter8 = []
iter8.append(test_subgroup("treatment_olaparib", df["brca2_mutation"] == 1, "brca2_mutation==1"))
iter8.append(test_subgroup("treatment_olaparib", df["brca2_mutation"] == 0, "brca2_mutation==0"))
iter8.append(test_subgroup("treatment_pembrolizumab", df["msi_high"] == 1, "msi_high==1"))
iter8.append(test_subgroup("treatment_pembrolizumab", df["msi_high"] == 0, "msi_high==0"))
iter8.append(test_subgroup("treatment_lu177_psma", df["psma_high"] == 1, "psma_high==1"))
iter8.append(test_subgroup("treatment_lu177_psma", df["psma_high"] == 0, "psma_high==0"))
iter8.append(test_subgroup("treatment_enzalutamide", df["ar_v7_positive"] == 1, "ar_v7_positive==1"))
iter8.append(test_subgroup("treatment_enzalutamide", df["ar_v7_positive"] == 0, "ar_v7_positive==0"))
iter8.append(test_subgroup("treatment_abiraterone", df["ar_v7_positive"] == 1, "ar_v7_positive==1"))
iter8.append(test_subgroup("treatment_abiraterone", df["ar_v7_positive"] == 0, "ar_v7_positive==0"))
results["iter8_targeted_biomarker_subgroups"] = iter8


# === ITER 9: Refine olaparib BRCA2 - test joint subgroup with additional modifier ===
# Within BRCA2+, does olaparib effect depend on visceral_mets, ECOG, mCRPC, etc.?
iter9 = []
brca = df[df["brca2_mutation"] == 1]
for f in BINARY_FEATURES:
    if f == "brca2_mutation":
        continue
    for fv in (0, 1):
        sub = brca[brca[f] == fv]
        if len(sub) < 100:
            continue
        yi = int(((sub["treatment_olaparib"] == 1) & (sub[OUTCOME] == 1)).sum())
        ni = int((sub["treatment_olaparib"] == 1).sum())
        yo = int(((sub["treatment_olaparib"] == 0) & (sub[OUTCOME] == 1)).sum())
        no = int((sub["treatment_olaparib"] == 0).sum())
        if ni == 0 or no == 0:
            continue
        diff, p = proportion_test(yi, ni, yo, no)
        iter9.append({
            "modifier": f, "modifier_value": fv,
            "n_on": ni, "n_off": no, "rate_on": yi / ni, "rate_off": yo / no,
            "diff": diff, "p": p,
        })
# Continuous within BRCA2+: median split tests for ECOG and visceral
iter9_cont = []
for f in ["ecog_ps", "visceral_mets"]:
    for thresh in (None,):
        if f == "ecog_ps":
            for cut in (0, 1):
                sub_low = brca[brca[f] <= cut]
                sub_hi = brca[brca[f] > cut]
                for label, sub in [(f"{f}_le_{cut}", sub_low), (f"{f}_gt_{cut}", sub_hi)]:
                    if len(sub) < 100:
                        continue
                    yi = int(((sub["treatment_olaparib"] == 1) & (sub[OUTCOME] == 1)).sum())
                    ni = int((sub["treatment_olaparib"] == 1).sum())
                    yo = int(((sub["treatment_olaparib"] == 0) & (sub[OUTCOME] == 1)).sum())
                    no = int((sub["treatment_olaparib"] == 0).sum())
                    if ni == 0 or no == 0:
                        continue
                    diff, p = proportion_test(yi, ni, yo, no)
                    iter9_cont.append({
                        "split": label, "n_on": ni, "n_off": no,
                        "rate_on": yi / ni, "rate_off": yo / no, "diff": diff, "p": p,
                    })
results["iter9_olaparib_within_brca2_refinement"] = {"binary": iter9, "continuous_splits": iter9_cont}


# === ITER 10: refine pembrolizumab MSI-high subgroup ===
iter10 = []
msi = df[df["msi_high"] == 1]
print("msi-high count:", len(msi))
for f in BINARY_FEATURES:
    if f == "msi_high":
        continue
    for fv in (0, 1):
        sub = msi[msi[f] == fv]
        if len(sub) < 30:
            continue
        yi = int(((sub["treatment_pembrolizumab"] == 1) & (sub[OUTCOME] == 1)).sum())
        ni = int((sub["treatment_pembrolizumab"] == 1).sum())
        yo = int(((sub["treatment_pembrolizumab"] == 0) & (sub[OUTCOME] == 1)).sum())
        no = int((sub["treatment_pembrolizumab"] == 0).sum())
        if ni == 0 or no == 0:
            continue
        diff, p = proportion_test(yi, ni, yo, no)
        iter10.append({
            "modifier": f, "modifier_value": fv,
            "n_on": ni, "n_off": no, "rate_on": yi / ni, "rate_off": yo / no,
            "diff": diff, "p": p,
        })
results["iter10_pembro_within_msi_refinement"] = iter10


# === ITER 11: Refine lu177-psma in PSMA-high ===
iter11 = []
psm = df[df["psma_high"] == 1]
for f in BINARY_FEATURES:
    if f == "psma_high":
        continue
    for fv in (0, 1):
        sub = psm[psm[f] == fv]
        if len(sub) < 100:
            continue
        yi = int(((sub["treatment_lu177_psma"] == 1) & (sub[OUTCOME] == 1)).sum())
        ni = int((sub["treatment_lu177_psma"] == 1).sum())
        yo = int(((sub["treatment_lu177_psma"] == 0) & (sub[OUTCOME] == 1)).sum())
        no = int((sub["treatment_lu177_psma"] == 0).sum())
        if ni == 0 or no == 0:
            continue
        diff, p = proportion_test(yi, ni, yo, no)
        iter11.append({
            "modifier": f, "modifier_value": fv,
            "n_on": ni, "n_off": no, "rate_on": yi / ni, "rate_off": yo / no,
            "diff": diff, "p": p,
        })
results["iter11_lu177_within_psmahigh_refinement"] = iter11


# === ITER 12: Refine enzalutamide and abiraterone in AR-V7 negative ===
iter12 = []
arv7n = df[df["ar_v7_positive"] == 0]
arv7p = df[df["ar_v7_positive"] == 1]
for tname in ("treatment_enzalutamide", "treatment_abiraterone"):
    for label, sub in [("ar_v7_neg", arv7n), ("ar_v7_pos", arv7p)]:
        for f in ["mcrpc", "visceral_mets"]:
            for fv in (0, 1):
                ss = sub[sub[f] == fv]
                if len(ss) < 200:
                    continue
                yi = int(((ss[tname] == 1) & (ss[OUTCOME] == 1)).sum())
                ni = int((ss[tname] == 1).sum())
                yo = int(((ss[tname] == 0) & (ss[OUTCOME] == 1)).sum())
                no = int((ss[tname] == 0).sum())
                if ni == 0 or no == 0:
                    continue
                diff, p = proportion_test(yi, ni, yo, no)
                iter12.append({
                    "treatment": tname, "stratum": label,
                    "modifier": f, "modifier_value": fv,
                    "n_on": ni, "n_off": no, "rate_on": yi / ni, "rate_off": yo / no,
                    "diff": diff, "p": p,
                })
results["iter12_androgen_targeted_refinement"] = iter12


# === ITER 13: Three-binary-feature subgroup search per treatment (top hits) ===
iter13 = {}
for t in TREATMENTS:
    rows = []
    for (f1, f2, f3) in combinations(BINARY_FEATURES, 3):
        for v1 in (0, 1):
            for v2 in (0, 1):
                for v3 in (0, 1):
                    sub = df[(df[f1] == v1) & (df[f2] == v2) & (df[f3] == v3)]
                    yi = int(((sub[t] == 1) & (sub[OUTCOME] == 1)).sum())
                    ni = int((sub[t] == 1).sum())
                    yo = int(((sub[t] == 0) & (sub[OUTCOME] == 1)).sum())
                    no = int((sub[t] == 0).sum())
                    if ni < 30 or no < 30:
                        continue
                    diff, p = proportion_test(yi, ni, yo, no)
                    rows.append({
                        "f1": f1, "v1": v1, "f2": f2, "v2": v2, "f3": f3, "v3": v3,
                        "n_on": ni, "n_off": no, "rate_on": yi / ni, "rate_off": yo / no,
                        "diff": diff, "p": p,
                    })
    rows.sort(key=lambda r: (r["p"], -abs(r["diff"])))
    iter13[t] = rows[:10]
results["iter13_three_binary_subgroups_top10"] = iter13


# === ITER 14: Continuous-feature stratified treatment effect (median splits and ECOG levels) ===
iter14 = []
for t in TREATMENTS:
    for f in ["age_years", "ldh_u_l", "albumin_g_dl", "psa_ng_ml", "alkaline_phosphatase_u_l", "hemoglobin_g_dl", "crp_mg_l"]:
        med = df[f].median()
        for label, mask in [(f"{f}_le_{med:.2f}", df[f] <= med), (f"{f}_gt_{med:.2f}", df[f] > med)]:
            sub = df[mask]
            yi = int(((sub[t] == 1) & (sub[OUTCOME] == 1)).sum())
            ni = int((sub[t] == 1).sum())
            yo = int(((sub[t] == 0) & (sub[OUTCOME] == 1)).sum())
            no = int((sub[t] == 0).sum())
            if ni == 0 or no == 0:
                continue
            diff, p = proportion_test(yi, ni, yo, no)
            iter14.append({
                "treatment": t, "feature": f, "split": label,
                "n_on": ni, "n_off": no, "rate_on": yi / ni, "rate_off": yo / no,
                "diff": diff, "p": p,
            })
    for cut in (0, 1):
        for label, mask in [(f"ecog_le_{cut}", df["ecog_ps"] <= cut), (f"ecog_gt_{cut}", df["ecog_ps"] > cut)]:
            sub = df[mask]
            yi = int(((sub[t] == 1) & (sub[OUTCOME] == 1)).sum())
            ni = int((sub[t] == 1).sum())
            yo = int(((sub[t] == 0) & (sub[OUTCOME] == 1)).sum())
            no = int((sub[t] == 0).sum())
            if ni == 0 or no == 0:
                continue
            diff, p = proportion_test(yi, ni, yo, no)
            iter14.append({
                "treatment": t, "feature": "ecog_ps", "split": label,
                "n_on": ni, "n_off": no, "rate_on": yi / ni, "rate_off": yo / no,
                "diff": diff, "p": p,
            })
results["iter14_continuous_split_stratified"] = iter14


# === ITER 15: Final confirmatory multivariable model with chosen interactions ===
# Add specific interaction terms based on prior exploration
X = df[TREATMENTS + BINARY_FEATURES + CONTINUOUS_FEATURES].copy()
X["enz_x_arv7"] = df["treatment_enzalutamide"] * df["ar_v7_positive"]
X["abi_x_arv7"] = df["treatment_abiraterone"] * df["ar_v7_positive"]
X["ola_x_brca2"] = df["treatment_olaparib"] * df["brca2_mutation"]
X["pembro_x_msi"] = df["treatment_pembrolizumab"] * df["msi_high"]
X["lu_x_psma"] = df["treatment_lu177_psma"] * df["psma_high"]
Xc = sm.add_constant(X)
mod_int = sm.Logit(y, Xc).fit(disp=False, maxiter=300)
iter15 = []
for c in ["enz_x_arv7", "abi_x_arv7", "ola_x_brca2", "pembro_x_msi", "lu_x_psma"] + TREATMENTS:
    iter15.append({
        "var": c,
        "coef": float(mod_int.params[c]),
        "p": float(mod_int.pvalues[c]),
        "or": float(np.exp(mod_int.params[c])),
    })
results["iter15_final_interaction_logit"] = iter15

# Save results
with open(ROOT / "_my_analysis" / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("Saved results.")
print("Iter1 main effects:")
for r in iter1:
    print(f"  {r['treatment']}: rate_on={r['rate_on']:.3f}, rate_off={r['rate_off']:.3f}, diff={r['diff']:.3f}, p={r['p']:.3g}")
