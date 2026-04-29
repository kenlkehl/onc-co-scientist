"""Iterative analysis of ds001_nsclc anonymized dataset.

Runs a series of analyses, records each result, and writes
out structured artifacts that the harness will turn into transcript.json.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
df = pd.read_parquet(BUNDLE / "dataset.parquet")
y = df["objective_response"].astype(int).values

# Classify features
binary_cols = []
multi_int_cols = []
cont_cols = []
cat_cols = []
for c in df.columns:
    if c in ("patient_id", "objective_response"):
        continue
    s = df[c]
    if s.dtype == object:
        cat_cols.append(c)
    elif s.nunique() == 2:
        binary_cols.append(c)
    elif np.issubdtype(s.dtype, np.integer) and s.nunique() <= 11:
        multi_int_cols.append(c)
    else:
        cont_cols.append(c)

print(f"binary: {len(binary_cols)}, ord-int: {len(multi_int_cols)}, cont: {len(cont_cols)}, cat: {len(cat_cols)}")

# ---------------- Iteration 1: univariate binary screen ----------------
bin_rows = []
for c in binary_cols:
    x = df[c].astype(int).values
    ct = pd.crosstab(df[c], df["objective_response"])
    if ct.shape == (2, 2):
        # OR via 2x2
        a = ct.iloc[1, 1]; b = ct.iloc[1, 0]; cc = ct.iloc[0, 1]; d = ct.iloc[0, 0]
        # add 0.5 if any zero
        if min(a, b, cc, d) == 0:
            a += 0.5; b += 0.5; cc += 0.5; d += 0.5
        log_or = np.log((a * d) / (b * cc))
        se = np.sqrt(1/a + 1/b + 1/cc + 1/d)
        z = log_or / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        rr1 = a / (a + b)  # response rate when feature=1
        rr0 = cc / (cc + d)  # response rate when feature=0
        bin_rows.append(dict(feature=c, n1=int(df[c].sum()), n0=int(len(df) - df[c].sum()),
                             rr_when_1=rr1, rr_when_0=rr0, log_or=log_or, OR=np.exp(log_or),
                             p_value=p))
bin_df = pd.DataFrame(bin_rows).sort_values("p_value")
bin_df.to_csv(HERE / "iter1_binary_screen.csv", index=False)
print("Top 25 binary features by p-value:")
print(bin_df.head(25).to_string(index=False))

# ---------------- Iteration 2: categorical screen ----------------
cat_rows = []
for c in cat_cols:
    ct = pd.crosstab(df[c], df["objective_response"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    rates = (ct[1] / ct.sum(axis=1)).to_dict()
    cat_rows.append(dict(feature=c, levels=str(rates), chi2=chi2, dof=dof, p_value=p))
cat_df = pd.DataFrame(cat_rows).sort_values("p_value")
cat_df.to_csv(HERE / "iter2_categorical_screen.csv", index=False)
print("\nCategorical features:")
print(cat_df.to_string(index=False))

# ---------------- Iteration 3: continuous screen ----------------
cont_rows = []
all_cont = cont_cols + multi_int_cols
for c in all_cont:
    x = df[c].astype(float).values
    # Logistic regression of y on x (single covariate)
    X = sm.add_constant(x)
    try:
        m = sm.Logit(y, X).fit(disp=False)
        coef = m.params[1]
        p = m.pvalues[1]
        # Standardised effect: per SD
        sd = np.std(x)
        cont_rows.append(dict(feature=c, mean_resp=float(np.mean(x[y == 1])),
                              mean_nonresp=float(np.mean(x[y == 0])),
                              sd=sd, beta=coef, beta_per_sd=coef * sd,
                              OR_per_sd=float(np.exp(coef * sd)),
                              p_value=float(p)))
    except Exception as e:
        cont_rows.append(dict(feature=c, error=str(e)))
cont_df = pd.DataFrame(cont_rows).sort_values("p_value")
cont_df.to_csv(HERE / "iter3_continuous_screen.csv", index=False)
print("\nTop 25 continuous features by p-value:")
print(cont_df.head(25).to_string(index=False))

# Save categorisations for next iter
with open(HERE / "feature_types.json", "w") as f:
    json.dump(dict(binary=binary_cols, multi_int=multi_int_cols,
                   cont=cont_cols, cat=cat_cols), f, indent=2)
print("\nDone iter 1-3.")
