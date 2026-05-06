"""Subgroup analyses: explore the f016 x f018 synergy and find suppressors."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/nsclc/anonymized/dataset.parquet")
df['smoke_current'] = (df['feature_001'] == 'current').astype(int)
df['smoke_former'] = (df['feature_001'] == 'former').astype(int)
df['smoke_never']  = (df['feature_001'] == 'never').astype(int)
df['hist_squamous'] = (df['feature_006'] == 'squamous').astype(int)

binary_cols = [c for c in df.columns if df[c].dtype in ['int64'] and df[c].nunique() == 2 and c != 'feature_014']
float_cols = [c for c in df.columns if df[c].dtype == 'float64' and c != 'pfs_months']

results = {}

# 1. 2x2 table for feature_016 x feature_018
print('=== 2x2 PFS for feature_016 x feature_018 ===')
tab = df.groupby(['feature_016', 'feature_018'])['pfs_months'].agg(['mean', 'std', 'count'])
print(tab)
results['f016_x_f018'] = tab.reset_index().to_dict(orient='records')

# 2. Within feature_016=1 vs feature_016=0, what is effect of feature_018?
print('\n=== feature_018 effect within feature_016 strata ===')
strata = []
for v in [0, 1]:
    sub = df[df['feature_016'] == v]
    g0 = sub.loc[sub['feature_018'] == 0, 'pfs_months']
    g1 = sub.loc[sub['feature_018'] == 1, 'pfs_months']
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    strata.append({'feature_016': v, 'mean_f018=0': g0.mean(), 'mean_f018=1': g1.mean(),
                   'diff': g1.mean() - g0.mean(), 'n0': len(g0), 'n1': len(g1),
                   't': float(t), 'p': float(p)})
print(pd.DataFrame(strata))
results['f018_within_f016'] = strata

# 3. Within feature_016=1 (the responder pool), screen interactions of feature_018 with all other features
print('\n=== Within feature_016=1 (n=6368), interaction of feature_018 with other features ===')
sub = df[df['feature_016'] == 1].copy()
print('n =', len(sub))
hetero = []
all_other = (binary_cols + ['feature_014'] + float_cols +
             ['smoke_current', 'smoke_former', 'hist_squamous'])
for mod in all_other:
    if mod in ['feature_016', 'feature_018']:
        continue
    if sub[mod].nunique() < 2:
        continue
    d = sub[['feature_018', mod, 'pfs_months']].copy()
    d['inter'] = d['feature_018'] * d[mod]
    Xi = sm.add_constant(d[['feature_018', mod, 'inter']])
    try:
        mi = sm.OLS(d['pfs_months'], Xi).fit()
        hetero.append({'mod': mod,
                       'beta_f018': float(mi.params['feature_018']),
                       'beta_mod': float(mi.params[mod]),
                       'beta_inter': float(mi.params['inter']),
                       'p_inter': float(mi.pvalues['inter'])})
    except Exception as e:
        print('skip', mod, e)
hetero_df = pd.DataFrame(hetero).sort_values('p_inter')
print(hetero_df.head(20).to_string())
results['hetero_within_f016'] = hetero

# 4. Stratify f018 effect by feature_031 (since f016*f031 interaction was strong)
print('\n=== feature_018 effect within feature_016=1, stratified by feature_031 ===')
joint = []
for f31 in [0, 1]:
    sub2 = df[(df['feature_016'] == 1) & (df['feature_031'] == f31)]
    g0 = sub2.loc[sub2['feature_018'] == 0, 'pfs_months']
    g1 = sub2.loc[sub2['feature_018'] == 1, 'pfs_months']
    if len(g0) > 1 and len(g1) > 1:
        t, p = stats.ttest_ind(g1, g0, equal_var=False)
        joint.append({'feature_031': f31, 'n0': len(g0), 'n1': len(g1),
                      'mean0': g0.mean(), 'mean1': g1.mean(),
                      'diff': g1.mean() - g0.mean(), 't': float(t), 'p': float(p)})
print(pd.DataFrame(joint))
results['f018_within_f016_f031'] = joint

# 5. Look at three-way interaction: feature_016 x feature_018 x feature_031
print('\n=== 3-way: f016 x f018 x f031 mean PFS ===')
tab3 = df.groupby(['feature_016', 'feature_018', 'feature_031'])['pfs_months'].agg(['mean', 'count'])
print(tab3)
results['three_way_f016_f018_f031'] = tab3.reset_index().to_dict(orient='records')

# 6. Effect within f016=1 & f031=0 — the "responder" pool — does f018 give a clean effect?
print('\n=== Within f016=1 & f031=0 (best responder pool?) — look at distribution ===')
core = df[(df['feature_016'] == 1) & (df['feature_031'] == 0)]
print('n =', len(core))
print('PFS by f018:')
print(core.groupby('feature_018')['pfs_months'].agg(['mean', 'std', 'count']))

# 7. Which features predict whether someone is in the f016=1 strata?
print('\n=== What predicts feature_016=1? ===')
feats_for_pred = [c for c in (binary_cols + ['feature_014'] + float_cols +
                              ['smoke_current', 'smoke_former', 'hist_squamous'])
                  if c not in ['feature_016', 'feature_018']]
X = sm.add_constant(df[feats_for_pred])
m = sm.Logit(df['feature_016'], X).fit(disp=False)
preds = []
for name, b, p in zip(m.params.index, m.params.values, m.pvalues.values):
    preds.append({'var': name, 'beta': float(b), 'p': float(p)})
preds_df = pd.DataFrame(preds).sort_values('p')
print(preds_df.head(15).to_string())
results['predict_f016'] = preds

# 8. What predicts feature_018=1?
print('\n=== What predicts feature_018=1? ===')
X = sm.add_constant(df[feats_for_pred])
m = sm.Logit(df['feature_018'], X).fit(disp=False)
preds = []
for name, b, p in zip(m.params.index, m.params.values, m.pvalues.values):
    preds.append({'var': name, 'beta': float(b), 'p': float(p)})
preds_df = pd.DataFrame(preds).sort_values('p')
print(preds_df.head(15).to_string())
results['predict_f018'] = preds

with open("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/nsclc/anonymized/work/subgroups.json", 'w') as f:
    json.dump(results, f, default=str, indent=2)
print('\nDone.')
