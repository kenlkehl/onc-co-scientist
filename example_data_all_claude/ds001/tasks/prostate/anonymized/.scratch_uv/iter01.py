"""Iteration 1: univariate screen of every feature vs objective_response."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].astype(int).values

# Drop degenerate
features = [c for c in df.columns if c not in ('patient_id', 'objective_response', 'feature_030')]

results = []
for c in features:
    x = df[c].values
    if df[c].nunique() == 2:
        # Chi-square
        a = y[x == 1].mean()
        b = y[x == 0].mean()
        ct = pd.crosstab(df[c], df['objective_response'])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        # Risk diff
        rd = a - b
        results.append({'feature': c, 'kind': 'binary', 'rate_pos': a, 'rate_neg': b,
                        'risk_diff': rd, 'p': p, 'n_pos': int((x==1).sum())})
    elif df[c].nunique() <= 12:
        ct = pd.crosstab(df[c], df['objective_response'])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        rates = df.groupby(c)['objective_response'].mean().to_dict()
        # logistic slope (treat as continuous)
        X = sm.add_constant(x.astype(float))
        try:
            mod = sm.Logit(y, X).fit(disp=0)
            beta = mod.params[1]
            p_logit = mod.pvalues[1]
        except Exception:
            beta = np.nan; p_logit = np.nan
        results.append({'feature': c, 'kind': 'ordinal', 'rates': rates,
                        'beta': beta, 'p': p_logit, 'p_chi2': p})
    else:
        # logistic regression
        X = sm.add_constant(x.astype(float))
        try:
            mod = sm.Logit(y, X).fit(disp=0)
            beta = mod.params[1]
            p = mod.pvalues[1]
        except Exception:
            beta, p = np.nan, np.nan
        # standardized OR per SD
        sd = np.std(x)
        results.append({'feature': c, 'kind': 'continuous', 'beta': beta, 'p': p,
                        'or_per_sd': float(np.exp(beta * sd)) if not np.isnan(beta) else np.nan,
                        'mean': float(x.mean()), 'std': float(sd)})

# rank by p
results.sort(key=lambda r: r.get('p', 1.0) if r.get('p') is not None and not np.isnan(r.get('p',1.0)) else 1.0)
for r in results:
    print(r)
