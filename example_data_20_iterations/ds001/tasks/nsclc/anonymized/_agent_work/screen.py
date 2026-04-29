"""Univariate screen of every feature against objective_response."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].astype(int)

obj_cols = ['feature_057', 'feature_043', 'feature_123', 'feature_005']
feat_cols = [c for c in df.columns if c not in ('patient_id', 'objective_response')]

results = []

for c in feat_cols:
    s = df[c]
    if c in obj_cols:
        # categorical: chi-square
        ct = pd.crosstab(s, y)
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        # also report response rate by level
        rates = y.groupby(s).mean().to_dict()
        results.append({
            'feature': c,
            'kind': 'categorical',
            'n_levels': s.nunique(),
            'p_value': p,
            'effect': None,
            'rates': str(rates),
        })
    elif s.dtype == 'int64' and s.nunique() == 2:
        # binary: logistic regression coefficient
        X = sm.add_constant(s.astype(float).values)
        try:
            res = sm.Logit(y.values, X).fit(disp=0)
            beta = res.params[1]
            p = res.pvalues[1]
            r1 = y[s == 1].mean()
            r0 = y[s == 0].mean()
            results.append({
                'feature': c,
                'kind': 'binary',
                'n_pos': int(s.sum()),
                'rate_1': r1,
                'rate_0': r0,
                'OR': float(np.exp(beta)),
                'effect': float(beta),
                'p_value': float(p),
            })
        except Exception as e:
            results.append({'feature': c, 'kind': 'binary', 'error': str(e)})
    else:
        # continuous or ordinal int
        X = sm.add_constant(s.astype(float).values)
        try:
            res = sm.Logit(y.values, X).fit(disp=0)
            beta = res.params[1]
            p = res.pvalues[1]
            results.append({
                'feature': c,
                'kind': 'continuous' if s.dtype == 'float64' else 'ordinal',
                'mean': float(s.mean()),
                'std': float(s.std()),
                'OR_per_unit': float(np.exp(beta)),
                'effect': float(beta),
                'p_value': float(p),
            })
        except Exception as e:
            results.append({'feature': c, 'kind': 'cont', 'error': str(e)})

screen = pd.DataFrame(results)
screen = screen.sort_values('p_value')
screen.to_csv('univariate_screen.csv', index=False)
print('rows:', len(screen))
print()
print('top 30 by p-value:')
print(screen.head(30).to_string())
