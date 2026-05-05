"""Full analysis script for ds001_breast - generates results.json for transcript construction."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet('dataset.parquet')
y = df['pfs_months'].values

binary_features = [
    'sex_female','stage_iv','has_brain_mets','node_positive','postmenopausal',
    'er_positive','pr_positive','her2_positive','her2_low',
    'brca1_mutation','brca2_mutation','pik3ca_mutation',
]
treatments = [
    'treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
    'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab',
]
ordinal_features = ['ecog_ps']
continuous_features = [
    'age_years','ki67_pct','tumor_size_cm','albumin_g_dl','ldh_u_l',
    'weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
    'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
    'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl',
]

results = {}

def diff_test(g1, g0):
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    return float(np.mean(g1) - np.mean(g0)), float(p), float(np.mean(g1)), float(np.mean(g0))

# ---- Iteration 1: main effects of binary clinical/biomarker features on PFS ----
results['main_binary'] = {}
for f in binary_features + treatments + ordinal_features:
    if f == 'ecog_ps':
        # Use linear regression; report slope per unit
        x = df[f].values.astype(float)
        slope, intercept, r, p, se = stats.linregress(x, y)
        means = df.groupby(f)['pfs_months'].mean().to_dict()
        results['main_binary'][f] = {'slope_or_diff': float(slope), 'p': float(p),
                                     'group_means': {str(k): float(v) for k, v in means.items()}}
    else:
        g1 = y[df[f] == 1]
        g0 = y[df[f] == 0]
        diff, p, m1, m0 = diff_test(g1, g0)
        results['main_binary'][f] = {'slope_or_diff': diff, 'p': p, 'mean_pos': m1, 'mean_neg': m0,
                                     'n_pos': int(len(g1)), 'n_neg': int(len(g0))}

# ---- Iteration 2: main effects of continuous features on PFS via linear regression ----
results['main_continuous'] = {}
for f in continuous_features:
    x = df[f].values.astype(float)
    slope, intercept, r, p, se = stats.linregress(x, y)
    # Tertile means for context
    q = pd.qcut(df[f], 3, labels=['low','mid','high'], duplicates='drop')
    tmeans = df.groupby(q, observed=True)['pfs_months'].mean().to_dict()
    results['main_continuous'][f] = {'slope': float(slope), 'p': float(p), 'r': float(r),
                                     'tertile_means': {str(k): float(v) for k, v in tmeans.items()}}

# ---- Iteration 3: multivariable regression of PFS ----
X_cols = binary_features + treatments + ordinal_features + continuous_features
X = df[X_cols].astype(float).values
X_full = sm.add_constant(X)
ols = sm.OLS(y, X_full).fit()
results['multivariable_main'] = {
    'columns': ['const'] + X_cols,
    'coefs': [float(c) for c in ols.params],
    'pvals': [float(p) for p in ols.pvalues],
    'rsq': float(ols.rsquared),
}

# Save partial results so far
with open('results_v2.json', 'w') as fp:
    json.dump(results, fp, indent=2)
print('Iter 1-3 done. R^2 of main multivariable model:', ols.rsquared)
print('Top significant features (p < 0.001):')
for col, coef, p in zip(['const']+X_cols, ols.params, ols.pvalues):
    if p < 0.001:
        print(f'  {col}: coef={coef:+.4f}, p={p:.2e}')
