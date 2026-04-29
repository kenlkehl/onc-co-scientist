"""Statistical analysis pipeline for ds001_aml.

Computes univariate, multivariable, subgroup, and interaction analyses
of objective_response on 124 features, used to populate transcript.json.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

OUT = "results.json"
df = pd.read_parquet("dataset.parquet")
y = df["objective_response"].astype(int)

features = [c for c in df.columns if c.startswith("feature_")]

# Classify features
binary, multicat, continuous = [], [], []
for c in features:
    if df[c].dtype == "object":
        multicat.append(c)
    elif df[c].nunique() == 2:
        binary.append(c)
    elif df[c].nunique() <= 10:
        multicat.append(c)
    else:
        continuous.append(c)

results = {"binary_univariate": [], "continuous_univariate": [],
           "multicat_univariate": [], "interactions": [],
           "subgroup": [], "multivariable": {}, "race_insurance": {}}


def logit_summary(X, y, name):
    """Return (coef, se, z, p, OR) for a single feature using statsmodels logit."""
    Xc = sm.add_constant(X.astype(float))
    try:
        m = sm.Logit(y, Xc).fit(disp=0, maxiter=100)
        coef = m.params.iloc[1]
        se = m.bse.iloc[1]
        p = m.pvalues.iloc[1]
        return coef, se, p
    except Exception as e:
        return None, None, None


print("=== Binary univariate ===")
for c in binary:
    rate1 = y[df[c] == 1].mean()
    rate0 = y[df[c] == 0].mean()
    n1 = int((df[c] == 1).sum())
    n0 = int((df[c] == 0).sum())
    coef, se, p = logit_summary(df[[c]].rename(columns={c: "x"})[["x"]], y, c)
    if coef is None:
        continue
    OR = float(np.exp(coef))
    results["binary_univariate"].append({
        "feature": c, "n1": n1, "n0": n0,
        "rate_x1": float(rate1), "rate_x0": float(rate0),
        "diff": float(rate1 - rate0),
        "log_or": float(coef), "p": float(p), "OR": OR
    })

print("=== Continuous univariate ===")
for c in continuous:
    x = df[c].astype(float)
    # standardize for comparable effect estimate
    xs = (x - x.mean()) / x.std()
    coef, se, p = logit_summary(pd.DataFrame({"x": xs}), y, c)
    if coef is None:
        continue
    # Also: mean of x in responders vs non-responders
    m_resp = float(x[y == 1].mean())
    m_nonresp = float(x[y == 0].mean())
    results["continuous_univariate"].append({
        "feature": c, "mean_resp": m_resp, "mean_nonresp": m_nonresp,
        "log_or_per_sd": float(coef), "p": float(p),
        "OR_per_sd": float(np.exp(coef))
    })

print("=== Multi-cat univariate ===")
for c in multicat:
    # use formula
    try:
        f = f"objective_response ~ C(Q('{c}'))"
        m = smf.logit(f, data=df).fit(disp=0, maxiter=100)
        # collapse: log-likelihood ratio test for the categorical effect
        # compare to null
        m0 = smf.logit("objective_response ~ 1", data=df).fit(disp=0)
        lr = 2 * (m.llf - m0.llf)
        ddf = int(m.df_model)
        p = float(stats.chi2.sf(lr, ddf))
        # Per-level rates
        rates = df.groupby(c)["objective_response"].agg(["mean", "count"]).reset_index()
        rates_dict = {str(r[c]): {"rate": float(r["mean"]), "n": int(r["count"])}
                      for _, r in rates.iterrows()}
        results["multicat_univariate"].append({
            "feature": c, "lr_p": p, "df": ddf, "rates": rates_dict
        })
    except Exception as e:
        print(f"  multicat fail {c}: {e}")

# Save partial
with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Wrote", OUT)
