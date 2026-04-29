"""Iteration 16-22: deeper interaction structure and refined modelling."""
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
df["regimen_count"] = df["feature_006"] + df["feature_007"] + df["feature_039"]

# ---------------- Iteration 16: each of feature_006/007/039 interacts with feature_092 separately, controlling for the others ----------------
# Joint model with all three two-way interactions
m16 = smf.logit("objective_response ~ feature_006*feature_092 + feature_007*feature_092 + feature_039*feature_092 + feature_051 + feature_011 + feature_013 + feature_067",
                data=df).fit(disp=False)
print("--- Each of 006/007/039 x feature_092 ---")
print(m16.summary().tables[1])

# ---------------- Iteration 17: response-rate by regimen_count and feature_092 (binarised at median of feature_092) ----------------
df["f092_hi"] = (df["feature_092"] > df["feature_092"].median()).astype(int)
g17 = df.groupby(["regimen_count", "f092_hi"]).agg(n=("objective_response", "size"),
                                                     rr=("objective_response", "mean")).round(3)
print("\n--- Response by regimen_count and feature_092 high/low ---")
print(g17)

# Specifically look at high-tertile feature_092
df["f092_t"] = pd.qcut(df["feature_092"], 3, labels=False, duplicates="drop")
g17b = df.groupby(["regimen_count", "f092_t"]).agg(n=("objective_response", "size"),
                                                     rr=("objective_response", "mean")).round(3)
print("\n--- Response by regimen_count and feature_092 tertile ---")
print(g17b)

# ---------------- Iteration 18: examine whether feature_006 has predictive value only when feature_092 is high ----------------
# Within feature_092 strata, ORs of regimen_count (=feature_006+feature_007+feature_039)
for t in sorted(df["f092_t"].unique()):
    sub = df[df["f092_t"] == t]
    mm = smf.logit("objective_response ~ regimen_count + feature_051 + feature_011 + feature_013 + feature_067",
                   data=sub).fit(disp=False)
    print(f"\nfeature_092 tertile {t}: regimen_count coef={mm.params['regimen_count']:.4f}, "
          f"OR={np.exp(mm.params['regimen_count']):.3f}, p={mm.pvalues['regimen_count']:.2g}")

# ---------------- Iteration 19: feature_051 x feature_011 (PS x mets count) ----------------
m19 = smf.logit("objective_response ~ feature_051 * feature_011 + feature_013 + feature_067 + feature_006 + feature_007 + feature_039 + feature_092",
                data=df).fit(disp=False)
print("\n--- feature_051 x feature_011 ---")
print(m19.summary().tables[1])

# ---------------- Iteration 20: insurance disparity, controlling and unadjusted ----------------
m20a = smf.logit("objective_response ~ C(feature_005)", data=df).fit(disp=False)
print("\n--- Unadjusted insurance ---")
print(m20a.summary().tables[1])
m20b = smf.logit("objective_response ~ C(feature_005) + feature_051 + feature_011 + feature_013 + feature_067 + feature_006 + feature_007 + feature_039 + feature_092 + feature_099 + C(feature_123) + C(feature_043)",
                 data=df).fit(disp=False)
print("\n--- Adjusted insurance ---")
print(m20b.summary().tables[1])

# ---------------- Iteration 21: feature_099 dose-response ----------------
df["f099_q"] = pd.qcut(df["feature_099"], 4, labels=False, duplicates="drop")
g21 = df.groupby("f099_q").agg(n=("objective_response", "size"),
                                rr=("objective_response", "mean"),
                                f099_min=("feature_099", "min"),
                                f099_max=("feature_099", "max"))
print("\n--- feature_099 quartiles ---")
print(g21)

# ---------------- Iteration 22: PD-L1-like feature_092 by treatment regimen, refined ----------------
# Check whether feature_092 main effect goes negative if all 006/007/039 = 0
sub_no = df[df["regimen_count"] == 0]
sub_full = df[df["regimen_count"] == 3]
m22a = smf.logit("objective_response ~ feature_092 + feature_051 + feature_011 + feature_013 + feature_067", data=sub_no).fit(disp=False)
m22b = smf.logit("objective_response ~ feature_092 + feature_051 + feature_011 + feature_013 + feature_067", data=sub_full).fit(disp=False)
print("\n--- feature_092 effect within regimen_count=0 ---")
print(m22a.summary().tables[1])
print(f"OR per unit: {np.exp(m22a.params['feature_092']):.2f}")
print("\n--- feature_092 effect within regimen_count=3 ---")
print(m22b.summary().tables[1])
print(f"OR per unit: {np.exp(m22b.params['feature_092']):.2f}")

print("\nDone iter 16-22.")
