"""Fix the tree heterogeneity screen and append to results_full.json."""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
df = pd.read_parquet("dataset.parquet")
OUTCOME = "objective_response"

with open("results_full.json") as f:
    results = json.load(f)


def logistic(y, X):
    X = sm.add_constant(X.astype(float), has_constant="add")
    return sm.Logit(y.astype(int).values, X.values).fit(disp=0, maxiter=200), list(X.columns)


def all_feature_interactions(treatment, biomarker_filter):
    sub = df[df[biomarker_filter] == 1].reset_index(drop=True)
    if sub.shape[0] < 200:
        return None
    res = {}
    for col in [
        "ecog_ps",
        "visceral_mets",
        "mcrpc",
        "albumin_g_dl",
        "ldh_u_l",
        "psa_ng_ml",
        "gleason_score",
        "alkaline_phosphatase_u_l",
        "hemoglobin_g_dl",
        "weight_loss_pct_6mo",
        "crp_mg_l",
        "nlr",
        "age_years",
    ]:
        x = sub[col].astype(float).copy()
        if col in ("psa_ng_ml", "ldh_u_l", "crp_mg_l", "nlr", "alkaline_phosphatase_u_l"):
            x = np.log1p(x)
        T = sub[treatment].astype(float)
        X = pd.DataFrame({"T": T.values, "X": x.values, "TX": (T * x).values})
        try:
            m, cols = logistic(sub[OUTCOME], X)
            ti = cols.index("T"); xi = cols.index("X"); txi = cols.index("TX")
            res[col] = {
                "coef_T": float(m.params[ti]),
                "p_T": float(m.pvalues[ti]),
                "coef_X": float(m.params[xi]),
                "p_X": float(m.pvalues[xi]),
                "coef_TX": float(m.params[txi]),
                "p_TX": float(m.pvalues[txi]),
            }
        except Exception as exc:
            res[col] = {"error": str(exc)}
    return res


for key, t, b in [
    ("tree_olaparib_brca2", "treatment_olaparib", "brca2_mutation"),
    ("tree_pembro_msi", "treatment_pembrolizumab", "msi_high"),
    ("tree_lu177_psma_pos", "treatment_lu177_psma", "psma_high"),
    ("tree_enza_arv7neg", "treatment_enzalutamide", None),  # placeholder; replaced below
]:
    if b is None:
        continue
    results[key] = all_feature_interactions(t, b)

# Also: enzalutamide effect heterogeneity in AR-V7 negative subgroup
sub = df[df["ar_v7_positive"] == 0].reset_index(drop=True)
res = {}
for col in [
    "ecog_ps", "visceral_mets", "mcrpc", "albumin_g_dl", "ldh_u_l", "psa_ng_ml",
    "gleason_score", "alkaline_phosphatase_u_l", "hemoglobin_g_dl",
    "weight_loss_pct_6mo", "crp_mg_l", "nlr", "age_years",
]:
    x = sub[col].astype(float).copy()
    if col in ("psa_ng_ml", "ldh_u_l", "crp_mg_l", "nlr", "alkaline_phosphatase_u_l"):
        x = np.log1p(x)
    T = sub["treatment_enzalutamide"].astype(float)
    X = pd.DataFrame({"T": T.values, "X": x.values, "TX": (T * x).values})
    try:
        m, cols = logistic(sub[OUTCOME], X)
        ti = cols.index("T"); xi = cols.index("X"); txi = cols.index("TX")
        res[col] = {
            "coef_T": float(m.params[ti]), "p_T": float(m.pvalues[ti]),
            "coef_X": float(m.params[xi]), "p_X": float(m.pvalues[xi]),
            "coef_TX": float(m.params[txi]), "p_TX": float(m.pvalues[txi]),
        }
    except Exception as exc:
        res[col] = {"error": str(exc)}
results["tree_enza_in_arv7neg"] = res

# Joint subgroup: enza in PSA-low + AR-V7- (or other multi-modifier definitions)
def jstrat(treatment, masks):
    mask = pd.Series(True, index=df.index)
    label = []
    for nm, m in masks.items():
        mask = mask & m
        label.append(nm)
    sub = df[mask]
    if (sub[treatment] == 1).sum() < 5 or (sub[treatment] == 0).sum() < 5:
        return None
    a = sub.loc[sub[treatment] == 1, OUTCOME]
    b = sub.loc[sub[treatment] == 0, OUTCOME]
    counts = np.array([a.sum(), b.sum()])
    nobs = np.array([len(a), len(b)])
    from statsmodels.stats.proportion import proportions_ztest
    _, p = proportions_ztest(counts, nobs)
    return {
        "rate_t": float(a.mean()), "rate_c": float(b.mean()),
        "diff": float(a.mean() - b.mean()), "p": float(p),
        "n_t": int(nobs[0]), "n_c": int(nobs[1]), "label": "+".join(label),
    }


# Enza in AR-V7- + low PSA (since main heterogeneity hint is psa_low x enza shows huge effect: 48% vs 17%)
results["joint_enza_arv7neg_psa_low"] = jstrat(
    "treatment_enzalutamide",
    {"arv7_neg": df["ar_v7_positive"] == 0, "psa_low": df["psa_ng_ml"] <= df["psa_ng_ml"].median()},
)
results["joint_enza_arv7neg_psa_high"] = jstrat(
    "treatment_enzalutamide",
    {"arv7_neg": df["ar_v7_positive"] == 0, "psa_high": df["psa_ng_ml"] > df["psa_ng_ml"].median()},
)
# Enza in AR-V7- + ECOG<=1
results["joint_enza_arv7neg_ecog01"] = jstrat(
    "treatment_enzalutamide",
    {"arv7_neg": df["ar_v7_positive"] == 0, "ecog<=1": df["ecog_ps"] <= 1},
)
results["joint_enza_arv7neg_ecog2"] = jstrat(
    "treatment_enzalutamide",
    {"arv7_neg": df["ar_v7_positive"] == 0, "ecog=2": df["ecog_ps"] == 2},
)

# Enza in AR-V7- + low PSA + ECOG<=1 (the "best response" subgroup)
results["joint_enza_arv7neg_psa_low_ecog01"] = jstrat(
    "treatment_enzalutamide",
    {
        "arv7_neg": df["ar_v7_positive"] == 0,
        "psa_low": df["psa_ng_ml"] <= df["psa_ng_ml"].median(),
        "ecog<=1": df["ecog_ps"] <= 1,
    },
)

# Olaparib BRCA2+ albumin_low
results["joint_olap_brca2_alblow_ecog01"] = jstrat(
    "treatment_olaparib",
    {
        "brca2": df["brca2_mutation"] == 1,
        "alb_low": df["albumin_g_dl"] < df["albumin_g_dl"].median(),
        "ecog<=1": df["ecog_ps"] <= 1,
    },
)

# mcRPC stratified treatment effects (because mcrpc had massive multivar coef)
for t in [
    "treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
    "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab",
]:
    for label, mask in [("mcrpc1", df["mcrpc"] == 1), ("mcrpc0", df["mcrpc"] == 0)]:
        sub = df[mask]
        a = sub.loc[sub[t] == 1, OUTCOME]
        b = sub.loc[sub[t] == 0, OUTCOME]
        if len(a) < 20 or len(b) < 20:
            continue
        from statsmodels.stats.proportion import proportions_ztest
        counts = np.array([a.sum(), b.sum()])
        nobs = np.array([len(a), len(b)])
        _, p = proportions_ztest(counts, nobs)
        results[f"strat_{t}_{label}"] = {
            "rate_t": float(a.mean()), "rate_c": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "p": float(p),
            "n_t": int(nobs[0]), "n_c": int(nobs[1]),
        }

# mcrpc effect on outcome
sub_mc = df["mcrpc"] == 1
a = df.loc[sub_mc, OUTCOME]; b = df.loc[~sub_mc, OUTCOME]
from statsmodels.stats.proportion import proportions_ztest
counts = np.array([a.sum(), b.sum()]); nobs = np.array([len(a), len(b)])
_, p = proportions_ztest(counts, nobs)
results["main_mcrpc"] = {
    "rate_a": float(a.mean()), "rate_b": float(b.mean()),
    "diff": float(a.mean() - b.mean()), "z": None, "p": float(p),
    "n_a": int(len(a)), "n_b": int(len(b)),
    "events_a": int(a.sum()), "events_b": int(b.sum()),
}

with open("results_full.json", "w") as f:
    json.dump(results, f, indent=2, default=lambda x: None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x)

print("Tree screens fixed. New results:")
print("tree_olaparib_brca2 sample:")
for col, d in results["tree_olaparib_brca2"].items():
    print(f"  {col}: coef_T={d['coef_T']:+.4f} p_T={d['p_T']:.2e} coef_TX={d['coef_TX']:+.4f} p_TX={d['p_TX']:.2e}")
print()
print("Joint enza:")
for k in ["joint_enza_arv7neg_psa_low", "joint_enza_arv7neg_psa_high", "joint_enza_arv7neg_ecog01", "joint_enza_arv7neg_ecog2", "joint_enza_arv7neg_psa_low_ecog01"]:
    print(f"  {k}: {results[k]}")
print()
print("Strat by mcrpc:")
for k,v in results.items():
    if k.startswith("strat_treatment_") and ("mcrpc" in k):
        print(f"  {k}: rate_t={v['rate_t']:.3f} rate_c={v['rate_c']:.3f} diff={v['diff']:+.3f} p={v['p']:.2e} n_t={v['n_t']}")
