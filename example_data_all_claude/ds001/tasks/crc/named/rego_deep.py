"""Refine the regorafenib subgroup hypothesis."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
df = pd.read_parquet("dataset.parquet")


def fmt(x, p=4):
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    return float(round(float(x), p))


def subgroup_ttest(predicate, treatment, label):
    sub = df[predicate]
    if sub[treatment].sum() < 10 or (sub[treatment] == 0).sum() < 10:
        return {"label": label, "n_total": int(len(sub)), "note": "insufficient"}
    a = sub.loc[sub[treatment] == 1, "pfs_months"]
    b = sub.loc[sub[treatment] == 0, "pfs_months"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "label": label,
        "n_total": int(len(sub)),
        "n_treated": int(sub[treatment].sum()),
        "mean_treated": fmt(a.mean()),
        "mean_untreated": fmt(b.mean()),
        "diff": fmt(a.mean() - b.mean()),
        "p_value": fmt(p, 8),
    }


out = {}

# Left + KRAS-wt
pred = (df["right_sided_primary"] == 0) & (df["kras_mutation"] == 0)
out["left_kraswt"] = subgroup_ttest(pred, "treatment_regorafenib", "Rego in LEFT + KRAS-wt")

# Left + KRAS-wt + BRAF-wt
pred = (
    (df["right_sided_primary"] == 0)
    & (df["kras_mutation"] == 0)
    & (df["braf_v600e"] == 0)
)
out["left_kraswt_brafwt"] = subgroup_ttest(
    pred, "treatment_regorafenib", "Rego in LEFT + KRAS-wt + BRAF-wt"
)

# Left + KRAS-wt + BRAF-wt + NRAS-wt
pred = (
    (df["right_sided_primary"] == 0)
    & (df["kras_mutation"] == 0)
    & (df["braf_v600e"] == 0)
    & (df["nras_mutation"] == 0)
)
out["left_panwt"] = subgroup_ttest(
    pred, "treatment_regorafenib", "Rego in LEFT + panwt"
)

pred_neg = ~(
    (df["right_sided_primary"] == 0)
    & (df["kras_mutation"] == 0)
    & (df["braf_v600e"] == 0)
)
out["not_left_kraswt_brafwt"] = subgroup_ttest(
    pred_neg, "treatment_regorafenib", "Rego in NOT(LEFT+KRAS-wt+BRAF-wt)"
)

for f in ["right_sided_primary", "kras_mutation", "braf_v600e", "nras_mutation"]:
    out[f"pos_{f}"] = subgroup_ttest(df[f] == 1, "treatment_regorafenib", f"Rego in {f}=1")
    out[f"neg_{f}"] = subgroup_ttest(df[f] == 0, "treatment_regorafenib", f"Rego in {f}=0")

df["cea_q"] = pd.qcut(df["cea_ng_ml"], 4, labels=False)
for q in range(4):
    pred = df["cea_q"] == q
    out[f"cea_q{q}"] = subgroup_ttest(
        pred, "treatment_regorafenib", f"Rego in CEA quartile {q}"
    )

median_cea = df["cea_ng_ml"].median()
pred = (
    (df["right_sided_primary"] == 0)
    & (df["kras_mutation"] == 0)
    & (df["braf_v600e"] == 0)
    & (df["cea_ng_ml"] < median_cea)
)
out["left_kraswt_brafwt_lowcea"] = subgroup_ttest(
    pred, "treatment_regorafenib", "Rego in LEFT+KRAS-wt+BRAF-wt+lowCEA"
)
pred = (
    (df["right_sided_primary"] == 0)
    & (df["kras_mutation"] == 0)
    & (df["braf_v600e"] == 0)
    & (df["cea_ng_ml"] >= median_cea)
)
out["left_kraswt_brafwt_highcea"] = subgroup_ttest(
    pred, "treatment_regorafenib", "Rego in LEFT+KRAS-wt+BRAF-wt+highCEA"
)

pred_fav = (
    (df["right_sided_primary"] == 0)
    & (df["kras_mutation"] == 0)
    & (df["braf_v600e"] == 0)
)
sub = df[pred_fav].copy()
out["fav_n"] = int(len(sub))
out["fav_n_treated"] = int(sub["treatment_regorafenib"].sum())
sub["t"] = sub["treatment_regorafenib"]
for f in [
    "age_years",
    "ecog_ps",
    "cea_ng_ml",
    "albumin_g_dl",
    "ldh_u_l",
    "weight_loss_pct_6mo",
    "crp_mg_l",
    "nlr",
    "hemoglobin_g_dl",
]:
    sub2 = sub.copy()
    sub2["f"] = (sub2[f] - sub2[f].mean()) / sub2[f].std()
    sub2["tf"] = sub2["t"] * sub2["f"]
    m = sm.OLS(sub2["pfs_months"], sm.add_constant(sub2[["t", "f", "tf"]])).fit()
    out[f"fav_inter_{f}"] = {
        "coef_tf": fmt(m.params["tf"]),
        "p_tf": fmt(m.pvalues["tf"], 6),
        "coef_t": fmt(m.params["t"]),
        "p_t": fmt(m.pvalues["t"], 6),
    }

m3 = smf.ols(
    "pfs_months ~ treatment_regorafenib * right_sided_primary + treatment_regorafenib * kras_mutation + treatment_regorafenib * braf_v600e + treatment_regorafenib * nras_mutation",
    data=df,
).fit()
out["multi_interaction_summary"] = {
    "r2": fmt(m3.rsquared),
    "params": {c: {"coef": fmt(m3.params[c]), "p": fmt(m3.pvalues[c], 6)} for c in m3.params.index},
}

sub = df.copy()
sub["t"] = sub["treatment_pembrolizumab"]
for f in [
    "msi_high",
    "kras_mutation",
    "right_sided_primary",
    "ecog_ps",
    "stage_iv",
    "braf_v600e",
    "her2_amplified",
    "ntrk_fusion",
    "sex_female",
]:
    sub2 = sub.copy()
    sub2["m"] = sub2[f]
    sub2["tm"] = sub2["t"] * sub2["m"]
    m = sm.OLS(sub2["pfs_months"], sm.add_constant(sub2[["t", "m", "tm"]])).fit()
    out[f"pembro_inter_{f}"] = {
        "coef_tm": fmt(m.params["tm"]),
        "p_tm": fmt(m.pvalues["tm"], 6),
    }

with open("rego_subgroup.json", "w") as fh:
    json.dump(out, fh, indent=2, default=str)

print("=== Regorafenib refined subgroups ===")
for k in [
    "left_kraswt",
    "left_kraswt_brafwt",
    "left_panwt",
    "not_left_kraswt_brafwt",
    "left_kraswt_brafwt_lowcea",
    "left_kraswt_brafwt_highcea",
]:
    v = out[k]
    print(
        f"{k}: n={v.get('n_total')}, n_treated={v.get('n_treated')}, "
        f"mean_t={v.get('mean_treated')}, mean_u={v.get('mean_untreated')}, "
        f"diff={v.get('diff')}, p={v.get('p_value')}"
    )

print("\n=== CEA quartiles (regorafenib effect) ===")
for q in range(4):
    v = out[f"cea_q{q}"]
    print(f"q{q}: n={v.get('n_total')}, diff={v.get('diff')}, p={v.get('p_value')}")

print("\n=== Continuous interactions in favorable subgroup ===")
for k in [k for k in out if k.startswith("fav_inter_")]:
    print(f"  {k}: {out[k]}")

print("\n=== 4-way interaction params ===")
for c, p in out["multi_interaction_summary"]["params"].items():
    print(f"  {c}: coef={p['coef']}, p={p['p']}")

print("\n=== Pembro interactions ===")
for k in [k for k in out if k.startswith("pembro_inter_")]:
    print(f"  {k}: {out[k]}")
