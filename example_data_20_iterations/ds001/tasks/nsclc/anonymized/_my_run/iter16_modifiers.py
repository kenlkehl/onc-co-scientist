"""Iteration 16: test other potential effect modifiers of triple-combo benefit in biomarker-high.
Age, ECOG (f051), histology, smoking, sex (find candidate)."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values
df['triple'] = ((df['feature_006']==1)&(df['feature_007']==1)&(df['feature_039']==1)).astype(int)
df['f092_hi'] = (df['feature_092']>=0.5).astype(int)
df['triple_x_hi'] = df['triple'] * df['f092_hi']

# Combine triple+hi as the effective "responsive treatment" indicator and test if effect varies by:
# age, f051 (ECOG), histology, smoking, race, sex-candidates
df['age_z'] = (df['feature_078']-df['feature_078'].mean())/df['feature_078'].std()

# We'll use a logit with: triple_hi = triple AND f092_hi (~5x more potent indicator)
# vs. triple OR f092_hi alone (no synergy) — contrasted in the 4-way already shown
# Now: do effect modifiers attenuate or amplify the synergy?

# Use all-data model with interaction to test modifier × triple_hi
print('=== Effect modifiers of (triple × biomarker_hi) ===')
modifiers = {
    'age_z': df['age_z'].values,
    'feature_051': ((df['feature_051']-df['feature_051'].mean())/df['feature_051'].std()).values,
    'squamous': (df['feature_043']=='squamous').astype(float).values,
    'never_smoker': (df['feature_057']=='never').astype(float).values,
    'former_smoker': (df['feature_057']=='former').astype(float).values,
    'feature_011_z': ((df['feature_011']-df['feature_011'].mean())/df['feature_011'].std()).values,
    'feature_013': df['feature_013'].astype(float).values,
    'feature_067': df['feature_067'].astype(float).values,
    'feature_099_z': ((df['feature_099']-df['feature_099'].mean())/df['feature_099'].std()).values,
}

for name, mod in modifiers.items():
    data = pd.DataFrame({
        'mod': mod,
        'triple': df['triple'].astype(float),
        'hi': df['f092_hi'].astype(float),
    })
    data['triple_hi'] = data['triple']*data['hi']
    data['mod_triple_hi'] = data['mod']*data['triple_hi']
    data['mod_triple'] = data['mod']*data['triple']
    data['mod_hi'] = data['mod']*data['hi']
    X_full = sm.add_constant(data).astype(float)
    X_no = X_full.drop(columns=['mod_triple_hi'])
    res_f = sm.Logit(y, X_full).fit(disp=False, maxiter=200)
    res_n = sm.Logit(y, X_no).fit(disp=False, maxiter=200)
    lr = 2*(res_f.llf - res_n.llf)
    p = 1 - chi2_dist.cdf(lr, df=1)
    coef = res_f.params['mod_triple_hi']
    print(f'  {name:18s}: mod_triple_hi coef={coef:+.3f}, LR p={p:.3g}')

# Now also: simple ORR table within (triple=1, hi=1) by various subgroups
print('\n=== ORR within triple_combo + biomarker_hi (n=712) by subgroups ===')
sub = df[(df['triple']==1)&(df['f092_hi']==1)].copy()
print(f'N={len(sub)}, overall ORR={sub["objective_response"].mean():.3f}')
sub['age_quartile'] = pd.qcut(sub['feature_078'], 4)
print('\nBy age quartile:')
print(sub.groupby('age_quartile', observed=True)['objective_response'].agg(['mean','count']).to_string())
print('\nBy ECOG-like (feature_051):')
print(sub.groupby('feature_051')['objective_response'].agg(['mean','count']).to_string())
print('\nBy histology:')
print(sub.groupby('feature_043')['objective_response'].agg(['mean','count']).to_string())
print('\nBy smoking:')
print(sub.groupby('feature_057')['objective_response'].agg(['mean','count']).to_string())
