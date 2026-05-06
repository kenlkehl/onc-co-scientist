"""Joint subgroup search for regorafenib treatment effect heterogeneity.
The screen showed three modifiers (KRAS, right-sided, BRAF V600E) — test joint definition."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet("dataset.parquet")
out = {}


def tx_effect(mask, label):
    sub = df[mask]
    a = sub.loc[sub.treatment_regorafenib == 1, "pfs_months"]
    b = sub.loc[sub.treatment_regorafenib == 0, "pfs_months"]
    if len(a) < 3 or len(b) < 3:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "subgroup": label,
        "n_pos": int(len(a)), "n_neg": int(len(b)),
        "mean_pos": float(a.mean()), "mean_neg": float(b.mean()),
        "diff": float(a.mean() - b.mean()),
        "p": float(p),
    }


# ----- Step 1: regorafenib effect within each modifier alone -----
print("Single modifiers:")
single = [
    (df.kras_mutation == 0, "KRAS-wt"),
    (df.kras_mutation == 1, "KRAS-mut"),
    (df.right_sided_primary == 0, "Left-sided"),
    (df.right_sided_primary == 1, "Right-sided"),
    (df.braf_v600e == 0, "BRAF-wt"),
    (df.braf_v600e == 1, "BRAF-V600E"),
    (df.nras_mutation == 0, "NRAS-wt"),
    (df.nras_mutation == 1, "NRAS-mut"),
]
out["single"] = []
for m, l in single:
    r = tx_effect(m, l)
    out["single"].append(r)
    print(f"  rego in {l:15s}: n+={r['n_pos']:5d} n-={r['n_neg']:5d}  diff={r['diff']:+.3f}  p={r['p']:.3g}")

# ----- Step 2: pairs -----
print("\nPairs:")
pairs = [
    ((df.kras_mutation == 0) & (df.right_sided_primary == 0), "KRAS-wt & Left-sided"),
    ((df.kras_mutation == 0) & (df.braf_v600e == 0), "KRAS-wt & BRAF-wt"),
    ((df.right_sided_primary == 0) & (df.braf_v600e == 0), "Left-sided & BRAF-wt"),
]
out["pairs"] = []
for m, l in pairs:
    r = tx_effect(m, l)
    out["pairs"].append(r)
    print(f"  rego in {l:30s}: n+={r['n_pos']:5d} n-={r['n_neg']:5d}  diff={r['diff']:+.3f}  p={r['p']:.3g}")

# ----- Step 3: triple -----
print("\nTriples:")
triples = [
    ((df.kras_mutation == 0) & (df.right_sided_primary == 0) & (df.braf_v600e == 0),
     "KRAS-wt & Left-sided & BRAF-wt"),
    ((df.kras_mutation == 0) & (df.right_sided_primary == 0) & (df.braf_v600e == 0) & (df.nras_mutation == 0),
     "KRAS-wt & Left-sided & BRAF-wt & NRAS-wt (RAS/BRAF-wt left)"),
]
out["triples"] = []
for m, l in triples:
    r = tx_effect(m, l)
    out["triples"].append(r)
    print(f"  rego in {l:55s}: n+={r['n_pos']:5d} n-={r['n_neg']:5d}  diff={r['diff']:+.3f}  p={r['p']:.3g}")

# Complement subgroups (treatment effect should be small or absent)
print("\nComplement (any modifier unfavorable):")
comp = [
    (~((df.kras_mutation == 0) & (df.right_sided_primary == 0) & (df.braf_v600e == 0)),
     "any of KRAS-mut OR right OR BRAF-V600E"),
    ((df.kras_mutation == 1), "KRAS-mut (regardless)"),
    ((df.braf_v600e == 1), "BRAF-V600E (regardless)"),
    ((df.right_sided_primary == 1), "Right-sided (regardless)"),
]
out["complement"] = []
for m, l in comp:
    r = tx_effect(m, l)
    out["complement"].append(r)
    print(f"  rego in {l:50s}: n+={r['n_pos']:5d} n-={r['n_neg']:5d}  diff={r['diff']:+.3f}  p={r['p']:.3g}")

# ----- Step 4: 3-way interaction OLS -----
print("\nJoint OLS with 3-way:")
df["_left"] = (df.right_sided_primary == 0).astype(int)
df["_kraswt"] = (df.kras_mutation == 0).astype(int)
df["_brafwt"] = (df.braf_v600e == 0).astype(int)
df["_favorable"] = (df._left & df._kraswt & df._brafwt).astype(int)
m = smf.ols("pfs_months ~ treatment_regorafenib * _favorable", data=df).fit()
print(m.params)
print(m.pvalues)
out["favorable_joint"] = {
    "params": {k: float(v) for k, v in m.params.items()},
    "pvalues": {k: float(v) for k, v in m.pvalues.items()},
}

# Also test: every continuous feature x rego in favorable
print("\nContinuous-feature interactions with rego (in favorable subgroup):")
sub = df[df._favorable == 1].copy()
cont_feats = ["age_years", "ecog_ps", "albumin_g_dl", "ldh_u_l",
              "weight_loss_pct_6mo", "cea_ng_ml", "crp_mg_l", "nlr",
              "hemoglobin_g_dl"]
out["cont_in_favorable"] = []
for c in cont_feats:
    f = f"pfs_months ~ treatment_regorafenib * {c}"
    m = smf.ols(f, data=sub).fit()
    key = f"treatment_regorafenib:{c}"
    out["cont_in_favorable"].append({
        "feature": c,
        "interaction_coef": float(m.params.get(key, np.nan)),
        "interaction_p": float(m.pvalues.get(key, np.nan)),
    })
for r in sorted(out["cont_in_favorable"], key=lambda x: x["interaction_p"]):
    print(f"  rego x {r['feature']:25s}  int={r['interaction_coef']:+.4f}  p={r['interaction_p']:.3g}")

with open("rego_subgroup.json", "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nSaved rego_subgroup.json")
