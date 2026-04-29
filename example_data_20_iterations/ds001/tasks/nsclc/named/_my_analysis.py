"""Iterative hypothesis-testing analysis of ds001_nsclc.

Generates transcript.json and analysis_summary.txt.
"""
import json
import math
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")
N = len(DF)

ITERATIONS = []  # collected as we go
ANALYSIS_NOTES = []  # narrative text per iteration


def add_iter(idx, hypotheses, analyses, narrative):
    ITERATIONS.append({
        "index": idx,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses,
    })
    ANALYSIS_NOTES.append((idx, narrative))


# ---------- helper analysis functions ----------

def chi2_or(df, col, outcome="objective_response"):
    """Return (effect=log-OR, p-value, OR, n1_resp, n0_resp, rate1, rate0)."""
    a = int(((df[col] == 1) & (df[outcome] == 1)).sum())
    b = int(((df[col] == 1) & (df[outcome] == 0)).sum())
    c = int(((df[col] == 0) & (df[outcome] == 1)).sum())
    d = int(((df[col] == 0) & (df[outcome] == 0)).sum())
    table = np.array([[a, b], [c, d]])
    if (table.min() == 0):
        # use Haldane correction
        table_c = table + 0.5
        odds_ratio = (table_c[0, 0] * table_c[1, 1]) / (table_c[0, 1] * table_c[1, 0])
    else:
        odds_ratio = (a * d) / (b * c)
    chi2, p, _, _ = stats.chi2_contingency(table)
    n1 = a + b
    n0 = c + d
    rate1 = a / n1 if n1 else float("nan")
    rate0 = c / n0 if n0 else float("nan")
    return {
        "log_or": math.log(odds_ratio),
        "or": odds_ratio,
        "p": p,
        "rate_yes": rate1,
        "rate_no": rate0,
        "n_yes": n1,
        "n_no": n0,
    }


def cont_assoc(df, col, outcome="objective_response"):
    """Logistic regression of outcome on a continuous column. Use Wald via statsmodels OR a simple t-test on means."""
    g1 = df.loc[df[outcome] == 1, col].dropna()
    g0 = df.loc[df[outcome] == 0, col].dropna()
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    return {
        "diff_resp_minus_nonresp": float(g1.mean() - g0.mean()),
        "p": float(p),
        "mean_resp": float(g1.mean()),
        "mean_non": float(g0.mean()),
    }


def logit_fit(df, formula):
    import statsmodels.formula.api as smf
    return smf.logit(formula, data=df).fit(disp=False)


def fmt_p(p):
    if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
        return "NA"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


# Pre-compute treatment counts and overall rate
OVERALL_RATE = DF["objective_response"].mean()
print(f"Overall response rate: {OVERALL_RATE:.4f} ({DF['objective_response'].sum()}/{N})")
