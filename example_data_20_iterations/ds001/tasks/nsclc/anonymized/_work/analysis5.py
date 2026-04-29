"""Iteration 23-25: final analyses."""
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

# ---------------- Iteration 23: feature_092 effect across regimen_count strata (response rate by quintile) ----------------
df["f092_q5"] = pd.qcut(df["feature_092"], 5, labels=False, duplicates="drop")
g23 = df.groupby(["regimen_count", "f092_q5"]).agg(n=("objective_response", "size"),
                                                     rr=("objective_response", "mean")).round(3)
print("--- Response rate by regimen_count (rows) and feature_092 quintile (cols) ---")
print(g23.unstack())

# ---------------- Iteration 24: feature_076 and feature_112 - corroborate adjusted screen ----------------
m24a = smf.logit("objective_response ~ feature_076 + feature_051 + feature_011 + feature_013 + feature_067 + feature_006 + feature_007 + feature_039 + feature_092",
                 data=df).fit(disp=False)
print("\n--- feature_076 adjusted ---")
print(m24a.summary().tables[1].as_text().split("feature_076")[1].split("\n")[0])
m24b = smf.logit("objective_response ~ feature_112 + feature_051 + feature_011 + feature_013 + feature_067 + feature_006 + feature_007 + feature_039 + feature_092",
                 data=df).fit(disp=False)
print("\n--- feature_112 adjusted ---")
print(m24b.summary().tables[1].as_text().split("feature_112")[1].split("\n")[0])

# ---------------- Iteration 25: final synthesis model ----------------
final_formula = ("objective_response ~ feature_051 + feature_011 + feature_013 + feature_067 "
                 "+ feature_006 + feature_007 + feature_039 + feature_092 + feature_006:feature_092 + feature_007:feature_092 + feature_039:feature_092 "
                 "+ feature_099 + feature_063 + feature_076 + feature_112 + C(feature_005)")
final_m = smf.logit(final_formula, data=df).fit(disp=False)
print("\n--- Final synthesis model ---")
print(final_m.summary().tables[1])

# Goodness-of-fit
print(f"\nLog-likelihood: {final_m.llf:.1f}")
print(f"Pseudo R^2: {final_m.prsquared:.4f}")
print(f"AIC: {final_m.aic:.1f}")

# Save
out = pd.DataFrame({"coef": final_m.params, "se": final_m.bse, "z": final_m.tvalues,
                    "p": final_m.pvalues, "OR": np.exp(final_m.params)})
out.to_csv(HERE / "iter25_final_model.csv")

# ---------------- Sanity: response rate by regimen_count alone, within feature_092 top quintile ----------------
top_q = df["f092_q5"].max()
sub = df[df["f092_q5"] == top_q]
print("\n--- Within feature_092 top quintile, response rate by regimen_count ---")
print(sub.groupby("regimen_count").agg(n=("objective_response", "size"),
                                         rr=("objective_response", "mean")).round(3))

print("\nDone iter 23-25.")
