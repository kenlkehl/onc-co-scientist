"""Iteration 8: within unfavorable subgroup, look for any remaining heterogeneity
where feature_008 might still confer benefit.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

df = pd.read_parquet('../dataset.parquet')
unfav = ((df.feature_013==1) | (df.feature_015==1) | (df.feature_021==1) | (df.feature_027==1)).values
sub = df[unfav].reset_index(drop=True)
print('n unfavorable =', len(sub))

# Overall feature_008 effect within unfavorable
rr1 = sub.loc[sub.feature_008==1, 'objective_response'].mean()
rr0 = sub.loc[sub.feature_008==0, 'objective_response'].mean()
print(f'overall: n8=1: {(sub.feature_008==1).sum()} RR={rr1:.3f} | n8=0: {(sub.feature_008==0).sum()} RR={rr0:.3f} | diff={rr1-rr0:+.3f}')

# Fit logistic with feature_008 and all features and interactions of feature_008 with others
features = [c for c in sub.columns if c not in ('patient_id', 'objective_response', 'feature_030',
                                                  'feature_008')]
records = []
for cov in features:
    if not pd.api.types.is_numeric_dtype(sub[cov]):
        continue
    raw = sub[cov].astype(float).values
    if sub[cov].nunique() > 2:
        x_cov = (raw - raw.mean()) / raw.std()
    else:
        x_cov = raw
    x_tx = sub['feature_008'].astype(float).values
    x_int = x_tx * x_cov
    X = np.column_stack([np.ones(len(sub)), x_tx, x_cov, x_int])
    try:
        mod = sm.Logit(sub['objective_response'].astype(int).values, X).fit(disp=0, maxiter=100)
        beta_int, p_int = mod.params[3], mod.pvalues[3]
    except Exception:
        beta_int, p_int = np.nan, np.nan
    records.append({'cov': cov, 'beta_int': beta_int, 'p_int': p_int})

rec = pd.DataFrame(records).sort_values('p_int')
print('\n--- top 10 interactions within unfavorable ---')
print(rec.head(10).to_string(index=False))

print('\n=== Independent-sub-subgroups: within unfavorable, restrict to single positive marker ===')
for marker in ['feature_013','feature_015','feature_021','feature_027']:
    others = [m for m in ['feature_013','feature_015','feature_021','feature_027'] if m != marker]
    mask = (df[marker] == 1)
    for o in others:
        mask = mask & (df[o] == 0)
    s = df[mask]
    rr1 = s.loc[s.feature_008==1, 'objective_response'].mean()
    rr0 = s.loc[s.feature_008==0, 'objective_response'].mean()
    n1 = (s.feature_008==1).sum(); n0 = (s.feature_008==0).sum()
    print(f' {marker}=1, others=0: n={len(s)} | n8=1={n1} RR={rr1:.3f} | n8=0={n0} RR={rr0:.3f} | diff={rr1-rr0:+.3f}')
