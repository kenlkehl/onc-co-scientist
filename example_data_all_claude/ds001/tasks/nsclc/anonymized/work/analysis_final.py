"""Final battery: refine the subgroup with additional suppressors, screen for other
synergies, and lock down all numbers needed for the transcript."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/nsclc/anonymized/dataset.parquet")
df['smoke_current'] = (df['feature_001'] == 'current').astype(int)
df['smoke_former'] = (df['feature_001'] == 'former').astype(int)
df['hist_squamous'] = (df['feature_006'] == 'squamous').astype(int)

binary_cols = [c for c in df.columns if df[c].dtype in ['int64'] and df[c].nunique() == 2]
float_cols = [c for c in df.columns if df[c].dtype == 'float64' and c != 'pfs_months']
results = {}

# 1. Refine the responder subgroup: add f028=0 and f005=0 to the triple
print('=== Refined responder subgroup ===')
df['triple'] = ((df['feature_016'] == 1) & (df['feature_018'] == 1) & (df['feature_031'] == 0)).astype(int)
df['quint'] = ((df['feature_016'] == 1) & (df['feature_018'] == 1) & (df['feature_031'] == 0)
              & (df['feature_028'] == 0) & (df['feature_005'] == 0)).astype(int)

print('triple:', df.groupby('triple')['pfs_months'].agg(['mean','std','count','median']))
print('quint:', df.groupby('quint')['pfs_months'].agg(['mean','std','count','median']))

# Within triple-positive, effect of f028 alone, f005 alone, and combined
sub_t = df[df['triple'] == 1]
print('\n=== Within triple, f028 effect ===')
g0 = sub_t.loc[sub_t['feature_028']==0, 'pfs_months']
g1 = sub_t.loc[sub_t['feature_028']==1, 'pfs_months']
t, p = stats.ttest_ind(g1, g0, equal_var=False)
print(f'f028=0: mean={g0.mean():.3f} n={len(g0)}; f028=1: mean={g1.mean():.3f} n={len(g1)}; diff={g1.mean()-g0.mean():.3f}, p={p:.3e}')
results['triple_f028'] = {'mean0': float(g0.mean()), 'mean1': float(g1.mean()), 'diff': float(g1.mean()-g0.mean()), 'n0': len(g0), 'n1': len(g1), 'p': float(p)}

print('=== Within triple, f005 effect ===')
g0 = sub_t.loc[sub_t['feature_005']==0, 'pfs_months']
g1 = sub_t.loc[sub_t['feature_005']==1, 'pfs_months']
t, p = stats.ttest_ind(g1, g0, equal_var=False)
print(f'f005=0: mean={g0.mean():.3f} n={len(g0)}; f005=1: mean={g1.mean():.3f} n={len(g1)}; diff={g1.mean()-g0.mean():.3f}, p={p:.3e}')
results['triple_f005'] = {'mean0': float(g0.mean()), 'mean1': float(g1.mean()), 'diff': float(g1.mean()-g0.mean()), 'n0': len(g0), 'n1': len(g1), 'p': float(p)}

# Within f016=1, f018=1, f031=0, screen all binary features for whether they modify f018 effect
# Already did, but let's lock down numbers for f028, f005

# 2. Adjust for confounders: triple + quint vs. confounders
print('\n=== Adjusted: PFS ~ confounders + quint ===')
adj_cols = ([c for c in binary_cols if c not in ['feature_016','feature_018','feature_031','feature_028','feature_005']] +
            ['feature_014'] + float_cols + ['smoke_current','smoke_former','hist_squamous',
            'feature_016','feature_018','feature_031','feature_028','feature_005','quint'])
X = sm.add_constant(df[adj_cols])
m = sm.OLS(df['pfs_months'], X).fit()
print('beta_quint=', m.params['quint'], 'p=', m.pvalues['quint'])
print('rsq=', m.rsquared)
results['quint_adj'] = {'beta': float(m.params['quint']), 'p': float(m.pvalues['quint']), 'rsq': float(m.rsquared)}

# 3. Continuous suppressors: within triple-positive, do continuous features modify PFS?
print('\n=== Within triple, continuous predictors of PFS (Pearson) ===')
sub_t = df[df['triple'] == 1]
cont_in_triple = []
for c in float_cols:
    r, p = stats.pearsonr(sub_t[c], sub_t['pfs_months'])
    cont_in_triple.append({'col': c, 'r': float(r), 'p': float(p)})
print(pd.DataFrame(cont_in_triple).sort_values('p').head(10).to_string())
results['cont_in_triple'] = cont_in_triple

# 4. Sanity: if I exclude the triple subgroup, how do the main effects look?
print('\n=== Outside triple-positive: main effects of f016, f018, f031 ===')
sub_out = df[df['triple'] == 0]
for c in ['feature_016','feature_018','feature_031']:
    g0 = sub_out.loc[sub_out[c]==0, 'pfs_months']
    g1 = sub_out.loc[sub_out[c]==1, 'pfs_months']
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    print(f'{c}: diff={g1.mean()-g0.mean():.4f}, p={p:.3e}')

# 5. Check feature_015 as continuous interaction modifier
print('\n=== feature_015 (likely age) modifies f018 effect within f016=1, f031=0 ===')
sub = df[(df['feature_016']==1) & (df['feature_031']==0)].copy()
sub['inter'] = sub['feature_018'] * sub['feature_015']
X = sm.add_constant(sub[['feature_018','feature_015','inter']])
m = sm.OLS(sub['pfs_months'], X).fit()
print(m.summary().tables[1])
results['f018_x_f015_in_responder'] = {
    'beta_f018': float(m.params['feature_018']),
    'beta_f015': float(m.params['feature_015']),
    'beta_inter': float(m.params['inter']),
    'p_inter': float(m.pvalues['inter'])
}

# 6. Sanity: similar architectures? screen each binary x binary x binary three-way for ANY synergy
print('\n=== Three-way binary synergy screen: top 15 ===')
from itertools import combinations
threeway = []
b_useful = [c for c in binary_cols if df[c].nunique() == 2]
for a, b, c in combinations(b_useful, 3):
    # 2x2x2 mean PFS, find max - min
    means = df.groupby([a,b,c])['pfs_months'].mean()
    rng = means.max() - means.min()
    if rng > 2:
        # F-test: contrast cell with rest
        idx_max = means.idxmax()
        sel = ((df[a]==idx_max[0]) & (df[b]==idx_max[1]) & (df[c]==idx_max[2]))
        g_in = df.loc[sel, 'pfs_months']
        g_out = df.loc[~sel, 'pfs_months']
        if len(g_in) >= 50:
            t, p = stats.ttest_ind(g_in, g_out, equal_var=False)
            threeway.append({'a':a,'b':b,'c':c,
                             'a_v':int(idx_max[0]),'b_v':int(idx_max[1]),'c_v':int(idx_max[2]),
                             'mean_in':float(g_in.mean()),'n_in':int(len(g_in)),
                             'mean_out':float(g_out.mean()),'rng':float(rng),
                             'p':float(p)})
threeway_df = pd.DataFrame(threeway).sort_values('rng', ascending=False)
print(threeway_df.head(15).to_string())
results['threeway'] = threeway

# 7. Final summary numbers
final = {
    'triple_n': int(df['triple'].sum()),
    'triple_pfs_mean': float(df.loc[df['triple']==1,'pfs_months'].mean()),
    'rest_pfs_mean': float(df.loc[df['triple']==0,'pfs_months'].mean()),
    'quint_n': int(df['quint'].sum()),
    'quint_pfs_mean': float(df.loc[df['quint']==1,'pfs_months'].mean()),
    'rest_quint_mean': float(df.loc[df['quint']==0,'pfs_months'].mean()),
}
print('\n=== Final summary ===')
print(json.dumps(final, indent=2))
results['final'] = final

with open("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/nsclc/anonymized/work/final.json", 'w') as f:
    json.dump(results, f, default=str, indent=2)
print('\nDone.')
