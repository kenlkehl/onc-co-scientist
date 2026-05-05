"""Validate the f016 x f018 x f031 subgroup; explore other candidate treatments;
check three-way subgroups exhaustively for other treatment candidates."""
import json, itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/nsclc/anonymized/dataset.parquet")
df['smoke_current'] = (df['feature_001'] == 'current').astype(int)
df['smoke_former'] = (df['feature_001'] == 'former').astype(int)
df['hist_squamous'] = (df['feature_006'] == 'squamous').astype(int)

results = {}

binary_cols = [c for c in df.columns if df[c].dtype in ['int64'] and df[c].nunique() == 2]
float_cols = [c for c in df.columns if df[c].dtype == 'float64' and c != 'pfs_months']

# 1. Adjusted: does the triple subgroup effect persist after adjusting for confounders?
print('=== Adjusted multivariable model with triple interaction term ===')
df['triple'] = ((df['feature_016'] == 1) & (df['feature_018'] == 1) & (df['feature_031'] == 0)).astype(int)
adj_cols = ([c for c in binary_cols if c not in ['feature_016', 'feature_018', 'feature_031']] +
            ['feature_014'] + float_cols + ['smoke_current', 'smoke_former', 'hist_squamous',
            'feature_016', 'feature_018', 'feature_031', 'triple'])
X = sm.add_constant(df[adj_cols])
m = sm.OLS(df['pfs_months'], X).fit()
print(m.summary())
results['adjusted_triple'] = {'beta_triple': float(m.params['triple']),
                              'p_triple': float(m.pvalues['triple']),
                              'rsq': float(m.rsquared),
                              'beta_f016': float(m.params['feature_016']),
                              'beta_f018': float(m.params['feature_018']),
                              'beta_f031': float(m.params['feature_031'])}

# 2. Adjusted within f016=1 stratum: does feature_018 effect (modified by f031) persist?
print('\n=== Within f016=1: regress on confounders + f018 + f031 + f018*f031 ===')
sub = df[df['feature_016'] == 1].copy()
sub['f018xf031'] = sub['feature_018'] * sub['feature_031']
adj_cols2 = ([c for c in binary_cols if c not in ['feature_016']] +
             ['feature_014'] + float_cols + ['smoke_current', 'smoke_former', 'hist_squamous', 'f018xf031'])
X = sm.add_constant(sub[adj_cols2])
m = sm.OLS(sub['pfs_months'], X).fit()
print('beta f018 =', m.params['feature_018'], 'p =', m.pvalues['feature_018'])
print('beta f031 =', m.params['feature_031'], 'p =', m.pvalues['feature_031'])
print('beta f018*f031 =', m.params['f018xf031'], 'p =', m.pvalues['f018xf031'])
results['within_f016_adj'] = {'beta_f018': float(m.params['feature_018']),
                               'beta_f031': float(m.params['feature_031']),
                               'beta_f018xf031': float(m.params['f018xf031']),
                               'p_f018': float(m.pvalues['feature_018']),
                               'p_f031': float(m.pvalues['feature_031']),
                               'p_inter': float(m.pvalues['f018xf031'])}

# 3. Check whether *any* other binary suppressor variable nullifies the f018 effect within f016=1
print('\n=== Within f016=1, find any binary variable s.t. f018 effect varies by it ===')
sub = df[df['feature_016'] == 1].copy()
binary_excl = [c for c in binary_cols if c not in ['feature_016', 'feature_018']]
sub_h = []
for mod in binary_excl:
    if sub[mod].nunique() < 2: continue
    g00 = sub[(sub['feature_018']==0) & (sub[mod]==0)]['pfs_months']
    g10 = sub[(sub['feature_018']==1) & (sub[mod]==0)]['pfs_months']
    g01 = sub[(sub['feature_018']==0) & (sub[mod]==1)]['pfs_months']
    g11 = sub[(sub['feature_018']==1) & (sub[mod]==1)]['pfs_months']
    if min(len(g00),len(g10),len(g01),len(g11)) < 30: continue
    eff_when0 = g10.mean() - g00.mean()
    eff_when1 = g11.mean() - g01.mean()
    sub_h.append({'mod': mod, 'n00':len(g00),'n10':len(g10),'n01':len(g01),'n11':len(g11),
                  'pfs_00':g00.mean(),'pfs_10':g10.mean(),'pfs_01':g01.mean(),'pfs_11':g11.mean(),
                  'eff_when_mod0': eff_when0, 'eff_when_mod1': eff_when1,
                  'diff': eff_when0 - eff_when1})
sub_h_df = pd.DataFrame(sub_h).sort_values('diff', key=abs, ascending=False)
print(sub_h_df.to_string())
results['mod_within_f016'] = sub_h

# 4. Repeat heterogeneity for feature_023 (other big main effect, treatment-like) ---
print('\n=== feature_023 heterogeneity (binary modifiers; full sample) ===')
hetero23 = []
all_other = [c for c in binary_cols + ['feature_014'] + float_cols +
             ['smoke_current','smoke_former','hist_squamous'] if c != 'feature_023']
for mod in all_other:
    d = df[['feature_023', mod, 'pfs_months']].copy()
    d['inter'] = d['feature_023'] * d[mod]
    Xi = sm.add_constant(d[['feature_023', mod, 'inter']])
    mi = sm.OLS(d['pfs_months'], Xi).fit()
    hetero23.append({'mod': mod, 'beta_tx': float(mi.params['feature_023']),
                     'beta_mod': float(mi.params[mod]),
                     'beta_inter': float(mi.params['inter']),
                     'p_inter': float(mi.pvalues['inter'])})
hetero23_df = pd.DataFrame(hetero23).sort_values('p_inter')
print(hetero23_df.head(15).to_string())
results['hetero_f023'] = hetero23

# 5. Repeat for feature_011
print('\n=== feature_011 heterogeneity (binary modifiers; full sample) ===')
hetero11 = []
for mod in [c for c in all_other if c != 'feature_011']:
    d = df[['feature_011', mod, 'pfs_months']].copy()
    d['inter'] = d['feature_011'] * d[mod]
    Xi = sm.add_constant(d[['feature_011', mod, 'inter']])
    mi = sm.OLS(d['pfs_months'], Xi).fit()
    hetero11.append({'mod': mod, 'beta_tx': float(mi.params['feature_011']),
                     'beta_mod': float(mi.params[mod]),
                     'beta_inter': float(mi.params['inter']),
                     'p_inter': float(mi.pvalues['inter'])})
hetero11_df = pd.DataFrame(hetero11).sort_values('p_inter')
print(hetero11_df.head(15).to_string())
results['hetero_f011'] = hetero11

# 6. Check if the triple subgroup is robust — different sample and bootstrap sanity
print('\n=== Verification: PFS by triple-positive group ===')
tab = df.groupby('triple')['pfs_months'].agg(['mean', 'std', 'count', 'median'])
print(tab)
results['triple_table'] = tab.reset_index().to_dict(orient='records')

# 7. Smoking interactions with histology / treatment within triple
print('\n=== Within triple-positive subgroup, what predicts PFS variation? ===')
sub3 = df[df['triple'] == 1]
print('n =', len(sub3))
from scipy.stats import pearsonr
preds_in_triple = []
for c in float_cols:
    r, p = pearsonr(sub3[c], sub3['pfs_months'])
    preds_in_triple.append({'col': c, 'r': float(r), 'p': float(p)})
for c in binary_cols + ['smoke_current','smoke_former','hist_squamous']:
    if c == 'triple' or sub3[c].nunique() < 2: continue
    g0 = sub3.loc[sub3[c] == 0, 'pfs_months']
    g1 = sub3.loc[sub3[c] == 1, 'pfs_months']
    if len(g0) > 5 and len(g1) > 5:
        t, p = stats.ttest_ind(g1, g0, equal_var=False)
        preds_in_triple.append({'col': c, 'mean1': g1.mean(), 'mean0': g0.mean(),
                                'diff': g1.mean()-g0.mean(), 'p': float(p)})
preds_in_triple_df = pd.DataFrame(preds_in_triple).sort_values('p')
print(preds_in_triple_df.head(20).to_string())
results['preds_in_triple'] = preds_in_triple

with open("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/nsclc/anonymized/work/validate.json", 'w') as f:
    json.dump(results, f, default=str, indent=2)
print('\nDone.')
