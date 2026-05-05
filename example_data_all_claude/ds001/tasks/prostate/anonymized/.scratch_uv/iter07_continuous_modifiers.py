"""Iteration 7: within the favorable subgroup, screen for additional modifiers.

Within the favorable subgroup (feature_013=0 AND feature_015=0 AND feature_021=0
AND feature_027=0), is there further heterogeneity in feature_008's effect?
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
fav = ((df.feature_013==0) & (df.feature_015==0) & (df.feature_021==0) & (df.feature_027==0)).values
sub = df[fav].reset_index(drop=True)
print('n favorable =', len(sub))

# Standardize continuous features
features = [c for c in sub.columns if c not in ('patient_id', 'objective_response', 'feature_030',
                                                  'feature_013','feature_015','feature_021','feature_027')]
covs = [c for c in features if c != 'feature_008']

records = []
for cov in covs:
    if not pd.api.types.is_numeric_dtype(sub[cov]):
        continue
    x_tx = sub['feature_008'].astype(float).values
    raw = sub[cov].astype(float).values
    if sub[cov].nunique() > 2:
        x_cov = (raw - raw.mean()) / raw.std()
    else:
        x_cov = raw
    x_int = x_tx * x_cov
    X = np.column_stack([np.ones(len(sub)), x_tx, x_cov, x_int])
    try:
        mod = sm.Logit(sub['objective_response'].astype(int).values, X).fit(disp=0, maxiter=100)
        beta_int, p_int = mod.params[3], mod.pvalues[3]
        beta_cov, p_cov = mod.params[2], mod.pvalues[2]
    except Exception as e:
        beta_int, p_int = np.nan, np.nan
        beta_cov, p_cov = np.nan, np.nan
    records.append({'cov': cov, 'beta_int': beta_int, 'p_int': p_int,
                    'beta_cov_main': beta_cov, 'p_cov_main': p_cov, 'nuniq': sub[cov].nunique()})

rec = pd.DataFrame(records).sort_values('p_int')
print('\n--- Within favorable subgroup, feature_008 x covariate interactions sorted by p_int ---')
print(rec.to_string(index=False))

print('\n=== Effect of feature_008 by feature_001 level (within favorable) ===')
for v in sorted(sub.feature_001.unique()):
    s = sub[sub.feature_001 == v]
    if len(s) < 30: continue
    rr1 = s.loc[s.feature_008==1, 'objective_response'].mean()
    rr0 = s.loc[s.feature_008==0, 'objective_response'].mean()
    n1, n0 = (s.feature_008==1).sum(), (s.feature_008==0).sum()
    print(f' feature_001={v}: n={len(s)} | n8=1={n1} RR={rr1:.3f} | n8=0={n0} RR={rr0:.3f} | diff={rr1-rr0:+.3f}')

print('\n=== Effect of feature_008 by feature_022 quartile (within favorable, with all four off) ===')
qs = sub['feature_022'].quantile([0.25, 0.5, 0.75])
print('quantiles:', qs.to_dict())
for q_idx in range(4):
    if q_idx == 0:
        m = sub.feature_022 <= qs.iloc[0]
    elif q_idx == 3:
        m = sub.feature_022 > qs.iloc[2]
    else:
        m = (sub.feature_022 > qs.iloc[q_idx-1]) & (sub.feature_022 <= qs.iloc[q_idx])
    s = sub[m]
    rr1 = s.loc[s.feature_008==1, 'objective_response'].mean()
    rr0 = s.loc[s.feature_008==0, 'objective_response'].mean()
    n1, n0 = (s.feature_008==1).sum(), (s.feature_008==0).sum()
    print(f' Q{q_idx+1}: n={len(s)} | n8=1={n1} RR={rr1:.3f} | n8=0={n0} RR={rr0:.3f} | diff={rr1-rr0:+.3f}')
