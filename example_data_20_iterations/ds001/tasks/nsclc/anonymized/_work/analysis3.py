"""Iteration 9-15: deeper interactions, refined effects."""
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

# ---------------- Iteration 9: 3-way interaction feature_006*feature_007*feature_039 with feature_092 ----------------
# First check whether features 006/007/039 are jointly meaningful (e.g. combo-therapy regimen)
ct = pd.crosstab([df["feature_006"], df["feature_007"]], df["feature_039"])
print("--- Cross-tab feature_006 x feature_007 x feature_039 ---")
print(ct)

# Combine: build a regimen indicator = sum of 006, 007, 039
df["regimen_count"] = df["feature_006"] + df["feature_007"] + df["feature_039"]
print(df.groupby("regimen_count").agg(n=("objective_response", "size"),
                                       resp_rate=("objective_response", "mean")))

# Logit with regimen_count, feature_092 and their interaction
m9 = smf.logit("objective_response ~ regimen_count * feature_092 + feature_051 + feature_011 + feature_013 + feature_067", data=df).fit(disp=False)
print(m9.summary().tables[1])

# ---------------- Iteration 10: feature_092 quartile-stratified response rates ----------------
df["f092_q"] = pd.qcut(df["feature_092"], 4, labels=False, duplicates="drop")
g = df.groupby("f092_q").agg(n=("objective_response", "size"),
                             rr=("objective_response", "mean"),
                             pd_l1_min=("feature_092", "min"),
                             pd_l1_max=("feature_092", "max"))
print("\n--- feature_092 quartile response rates ---")
print(g)

# Test linear trend
m10 = smf.logit("objective_response ~ feature_092", data=df).fit(disp=False)
print("\nfeature_092 main effect logit coef:", m10.params["feature_092"], "p=", m10.pvalues["feature_092"])

# ---------------- Iteration 11: feature_013 and feature_067 — co-occurrence? ----------------
ct13_67 = pd.crosstab(df["feature_013"], df["feature_067"])
print("\n--- feature_013 x feature_067 cross-tab ---")
print(ct13_67)
# Joint model
m11 = smf.logit("objective_response ~ feature_013 * feature_067 + feature_051 + feature_011 + feature_006 + feature_007 + feature_039 + feature_092", data=df).fit(disp=False)
print(m11.summary().tables[1])

# ---------------- Iteration 12: Race / insurance disparities controlling for clinical features ----------------
m12a = smf.logit("objective_response ~ C(feature_123) + feature_051 + feature_011 + feature_013 + feature_067 + feature_006 + feature_007 + feature_039 + feature_092 + feature_099 + C(feature_005)", data=df).fit(disp=False)
print("\n--- Race-adjusted (feature_123) ---")
print(m12a.summary().tables[1])

# ---------------- Iteration 13: smoking status and histology effects ----------------
m13 = smf.logit("objective_response ~ C(feature_057) * C(feature_043) + feature_051 + feature_011 + feature_006 + feature_007 + feature_039", data=df).fit(disp=False)
print("\n--- Smoking x Histology ---")
print(m13.summary().tables[1])

# ---------------- Iteration 14: feature_011 dose-response (categorize) ----------------
df["f011_cat"] = pd.cut(df["feature_011"], bins=[-0.1, 0, 2, 5, 10, 30], labels=["0", "1-2", "3-5", "6-10", "11+"])
g11 = df.groupby("f011_cat").agg(n=("objective_response", "size"),
                                 rr=("objective_response", "mean"))
print("\n--- feature_011 categories ---")
print(g11)

# ---------------- Iteration 15: refined screen for additional binary features (after adjustment) ----------------
covars = "feature_051 + feature_011 + feature_013 + feature_067 + feature_006 + feature_007 + feature_039 + feature_092 + feature_099"
extra_bin = [c for c in df.columns if c not in ("patient_id","objective_response","feature_013","feature_067","feature_006","feature_007","feature_039","feature_051","feature_011","feature_099","feature_063","feature_092","f092_q","regimen_count","f011_cat") and df[c].nunique() == 2 and df[c].dtype != object]
adj_rows = []
for c in extra_bin:
    f = f"objective_response ~ {c} + " + covars
    try:
        mm = smf.logit(f, data=df).fit(disp=False)
        adj_rows.append(dict(feature=c, coef=mm.params[c], p=mm.pvalues[c],
                             OR=float(np.exp(mm.params[c]))))
    except Exception as e:
        adj_rows.append(dict(feature=c, error=str(e)))
adj_df = pd.DataFrame(adj_rows).sort_values("p")
adj_df.to_csv(HERE / "iter15_adjusted_binary.csv", index=False)
print("\n--- Adjusted binary screen (top 10) ---")
print(adj_df.head(10).to_string(index=False))

print("\nDone iter 9-15.")
