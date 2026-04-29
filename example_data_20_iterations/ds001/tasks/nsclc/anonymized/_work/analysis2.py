"""Iteration 4-8: multivariable logistic, interactions, subgroup analyses."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
df = pd.read_parquet(BUNDLE / "dataset.parquet")
y = df["objective_response"].astype(int)

# Top features identified in iters 1-3
top_bin = ["feature_013", "feature_067", "feature_006", "feature_007", "feature_039"]
top_cont = ["feature_051", "feature_011", "feature_099", "feature_063", "feature_092"]
cat = ["feature_123", "feature_043", "feature_057", "feature_005"]

# ---------------- Iteration 4: multivariable logistic ----------------
df["objective_response"] = df["objective_response"].astype(int)
formula = ("objective_response ~ feature_013 + feature_067 + feature_006 + feature_007 + feature_039 "
           "+ feature_051 + feature_011 + feature_099 + feature_063 + feature_092 "
           "+ C(feature_123) + C(feature_043) + C(feature_057) + C(feature_005)")
m4 = smf.logit(formula, data=df).fit(disp=False)
print("--- Multivariable logistic (Iteration 4) ---")
print(m4.summary().tables[1])
mv_summary = pd.DataFrame({"coef": m4.params, "se": m4.bse, "z": m4.tvalues,
                           "p": m4.pvalues, "OR": np.exp(m4.params)})
mv_summary.to_csv(HERE / "iter4_multivariable.csv")

# ---------------- Iteration 5: pairwise interactions among top features ----------------
inter_results = []
top_all = top_bin + top_cont
for i, a in enumerate(top_all):
    for b in top_all[i + 1:]:
        f = (f"objective_response ~ {a} + {b} + {a}:{b} + feature_013 + feature_067 + feature_006 "
             f"+ feature_051 + feature_011")
        # avoid duplicating main effects
        try:
            mm = smf.logit(f, data=df).fit(disp=False)
            ikey = f"{a}:{b}"
            coef = mm.params.get(ikey, np.nan)
            p = mm.pvalues.get(ikey, np.nan)
            inter_results.append(dict(a=a, b=b, coef=coef, p_value=p, OR=np.exp(coef) if pd.notna(coef) else np.nan))
        except Exception as e:
            inter_results.append(dict(a=a, b=b, error=str(e)))
inter_df = pd.DataFrame(inter_results).sort_values("p_value")
inter_df.to_csv(HERE / "iter5_pairwise_interactions.csv", index=False)
print("\n--- Top 20 pairwise interactions ---")
print(inter_df.head(20).to_string(index=False))

# ---------------- Iteration 6: feature_051 interactions with all binary features (PS-like effect modification) ----------------
# feature_051 is 0/1/2 with strong main effect (OR_per_unit ~ 0.69)
ps_int = []
for c in (top_bin + ["feature_076", "feature_112", "feature_021", "feature_053", "feature_074", "feature_100"]):
    f = f"objective_response ~ feature_051 * {c} + feature_011 + feature_013 + feature_067 + feature_006"
    try:
        mm = smf.logit(f, data=df).fit(disp=False)
        ikey = f"feature_051:{c}"
        ps_int.append(dict(b=c, coef=mm.params[ikey], p=mm.pvalues[ikey],
                           OR=float(np.exp(mm.params[ikey]))))
    except Exception as e:
        ps_int.append(dict(b=c, error=str(e)))
ps_int_df = pd.DataFrame(ps_int).sort_values("p")
ps_int_df.to_csv(HERE / "iter6_feature051_x_binary_interactions.csv", index=False)
print("\n--- feature_051 x binary interactions ---")
print(ps_int_df.to_string(index=False))

# ---------------- Iteration 7: feature_011 interactions ----------------
f11_int = []
for c in (top_bin + cat[:2]):
    if c in cat:
        f = f"objective_response ~ feature_011 * C({c}) + feature_051 + feature_013 + feature_067"
    else:
        f = f"objective_response ~ feature_011 * {c} + feature_051 + feature_013 + feature_067"
    try:
        mm = smf.logit(f, data=df).fit(disp=False)
        # Print all interaction terms
        ikeys = [k for k in mm.params.index if "feature_011:" in k or ":feature_011" in k]
        for ik in ikeys:
            f11_int.append(dict(b=c, term=ik, coef=mm.params[ik], p=mm.pvalues[ik]))
    except Exception as e:
        f11_int.append(dict(b=c, error=str(e)))
f11_int_df = pd.DataFrame(f11_int).sort_values("p")
f11_int_df.to_csv(HERE / "iter7_feature011_interactions.csv", index=False)
print("\n--- feature_011 interactions ---")
print(f11_int_df.to_string(index=False))

# ---------------- Iteration 8: histology-stratified analyses ----------------
strat_rows = []
for level in df["feature_043"].unique():
    sub = df[df["feature_043"] == level]
    f = "objective_response ~ feature_013 + feature_067 + feature_006 + feature_051 + feature_011 + feature_099"
    mm = smf.logit(f, data=sub).fit(disp=False)
    for term in mm.params.index:
        if term == "Intercept":
            continue
        strat_rows.append(dict(stratum_var="feature_043", level=level, term=term,
                               coef=mm.params[term], p=mm.pvalues[term],
                               OR=float(np.exp(mm.params[term]))))
strat_df = pd.DataFrame(strat_rows)
strat_df.to_csv(HERE / "iter8_histology_strat.csv", index=False)
print("\n--- Histology-stratified ORs ---")
print(strat_df.to_string(index=False))

print("\nDone iter 4-8.")
