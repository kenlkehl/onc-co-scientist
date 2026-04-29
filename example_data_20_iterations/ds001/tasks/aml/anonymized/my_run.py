"""
Comprehensive analysis of the AML dataset.
Strategy: univariate screen of every feature against objective_response,
then probe interactions, subgroups, and multivariable adjustments.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
print('Loaded', df.shape)
y = df['objective_response'].astype(int)
print('Response rate:', y.mean())

# Classify features
feature_cols = [c for c in df.columns if c.startswith('feature_')]
binary_cols = []
multi_cols = []  # ordinal/multi-level int
cont_cols = []
cat_cols = []
for c in feature_cols:
    if df[c].dtype == object:
        cat_cols.append(c)
    elif df[c].nunique() == 2:
        binary_cols.append(c)
    elif df[c].dtype.kind in 'iu' and df[c].nunique() <= 11:
        multi_cols.append(c)
    else:
        cont_cols.append(c)

print(f'Binary: {len(binary_cols)}, Multi-level int: {len(multi_cols)}, Continuous: {len(cont_cols)}, Categorical: {len(cat_cols)}')

# ----- Step 1: Univariate screening -----
results = []

for c in binary_cols:
    x = df[c].astype(int)
    p1 = y[x==1].mean()
    p0 = y[x==0].mean()
    n1 = (x==1).sum()
    n0 = (x==0).sum()
    # logistic regression for OR
    try:
        m = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
        beta = m.params[c]
        pval = m.pvalues[c]
        OR = np.exp(beta)
    except Exception:
        beta, pval, OR = np.nan, np.nan, np.nan
    results.append({'feature': c, 'kind': 'binary', 'n1': n1, 'n0': n0,
                    'rate1': p1, 'rate0': p0, 'diff': p1 - p0,
                    'beta': beta, 'OR': OR, 'p': pval})

for c in multi_cols:
    x = df[c].astype(float)
    try:
        m = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
        beta = m.params[c]
        pval = m.pvalues[c]
    except Exception:
        beta, pval = np.nan, np.nan
    rate0 = y[x==x.min()].mean()
    rateMax = y[x==x.max()].mean()
    results.append({'feature': c, 'kind': 'ordinal',
                    'n_levels': df[c].nunique(),
                    'rate_min': rate0, 'rate_max': rateMax,
                    'diff': rateMax - rate0, 'beta': beta,
                    'OR': np.exp(beta) if not np.isnan(beta) else np.nan,
                    'p': pval})

for c in cont_cols:
    x = df[c].astype(float)
    # standardized for comparable beta
    z = (x - x.mean()) / x.std()
    try:
        m = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
        beta = m.params[0 if isinstance(m.params, np.ndarray) else m.params.index[1]]
        beta = m.params.iloc[1]
        pval = m.pvalues.iloc[1]
    except Exception:
        beta, pval = np.nan, np.nan
    results.append({'feature': c, 'kind': 'continuous',
                    'mean_resp1': x[y==1].mean(),
                    'mean_resp0': x[y==0].mean(),
                    'diff': x[y==1].mean() - x[y==0].mean(),
                    'beta_per_sd': beta,
                    'OR_per_sd': np.exp(beta) if not np.isnan(beta) else np.nan,
                    'p': pval})

for c in cat_cols:
    # one-hot encode and run multinomial-style chi-square
    tab = pd.crosstab(df[c], y)
    try:
        chi2, pval, dof, _ = stats.chi2_contingency(tab)
    except Exception:
        chi2, pval = np.nan, np.nan
    rates = (tab[1] / tab.sum(axis=1)).to_dict()
    results.append({'feature': c, 'kind': 'categorical',
                    'rates': rates, 'chi2': chi2, 'p': pval})

univ = pd.DataFrame(results)
univ_sorted = univ.sort_values('p', na_position='last')
univ_sorted.to_csv('univariate_my.csv', index=False)
print('\n=== Top 30 univariate hits ===')
print(univ_sorted.head(30)[['feature','kind','diff','beta','OR','beta_per_sd','OR_per_sd','p']].to_string())
print('\nTotal significant at 0.05:', (univ['p'] < 0.05).sum(), 'of', len(univ))
print('Significant at Bonferroni (p < 0.05/124):', (univ['p'] < 0.05/124).sum())
