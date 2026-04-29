import pandas as pd
import numpy as np
import scipy.stats as ss
import statsmodels.formula.api as smf
import json

df = pd.read_parquet('dataset.parquet')
results = {}

# === Iter 1: feature_078 effect ===
r, p = ss.pearsonr(df['feature_078'], df['pfs_months'])
m = smf.ols('pfs_months ~ feature_078', data=df).fit()
results['iter1_feature_078'] = {
    'pearson_r': float(r), 'pearson_p': float(p),
    'ols_slope': float(m.params['feature_078']),
    'ols_p': float(m.pvalues['feature_078']),
    'intercept': float(m.params['Intercept']),
}

# === Iter 2: feature_057 ===
m = smf.ols('pfs_months ~ feature_057', data=df).fit()
results['iter2_feature_057'] = {
    'slope': float(m.params['feature_057']),
    'p': float(m.pvalues['feature_057']),
    'means': {str(v): float(df.loc[df['feature_057'] == v, 'pfs_months'].mean()) for v in sorted(df['feature_057'].unique())},
    'ns': {str(v): int((df['feature_057'] == v).sum()) for v in sorted(df['feature_057'].unique())},
}

# === Iter 3: feature_051 ===
g0 = df.loc[df['feature_051'] == 0, 'pfs_months']
g1 = df.loc[df['feature_051'] == 1, 'pfs_months']
t, p = ss.ttest_ind(g0, g1)
results['iter3_feature_051'] = {
    'mean_0': float(g0.mean()), 'mean_1': float(g1.mean()),
    'diff_1_minus_0': float(g1.mean() - g0.mean()), 'p': float(p),
    'n_0': int(len(g0)), 'n_1': int(len(g1)),
}

# === Iter 4: feature_038 ===
g0 = df.loc[df['feature_038'] == 0, 'pfs_months']
g1 = df.loc[df['feature_038'] == 1, 'pfs_months']
t, p = ss.ttest_ind(g0, g1)
results['iter4_feature_038'] = {
    'mean_0': float(g0.mean()), 'mean_1': float(g1.mean()),
    'diff_1_minus_0': float(g1.mean() - g0.mean()), 'p': float(p),
    'n_0': int(len(g0)), 'n_1': int(len(g1)),
}

# === Iter 5: feature_013, feature_043, feature_109, feature_067 ===
res5 = {}
for c in ['feature_013', 'feature_043', 'feature_109', 'feature_067']:
    g0 = df.loc[df[c] == 0, 'pfs_months']
    g1 = df.loc[df[c] == 1, 'pfs_months']
    t, p = ss.ttest_ind(g0, g1)
    res5[c] = {'mean_0': float(g0.mean()), 'mean_1': float(g1.mean()),
               'diff': float(g1.mean() - g0.mean()), 'p': float(p),
               'n_0': int(len(g0)), 'n_1': int(len(g1))}
results['iter5_more_binaries'] = res5

# === Iter 6: continuous lab-like ===
res6 = {}
for c in ['feature_092', 'feature_099', 'feature_009']:
    r, p = ss.pearsonr(df[c], df['pfs_months'])
    res6[c] = {'r': float(r), 'p': float(p)}
results['iter6_continuous'] = res6

# === Iter 7: race ===
groups = {cat: df.loc[df['feature_064'] == cat, 'pfs_months'].values for cat in df['feature_064'].unique()}
f, p = ss.f_oneway(*groups.values())
results['iter7_race'] = {
    'overall_F': float(f), 'overall_p': float(p),
    'means': {k: float(v.mean()) for k, v in groups.items()},
    'ns': {k: int(len(v)) for k, v in groups.items()},
}
white_mean = groups['white'].mean()
res7p = {}
for cat, vals in groups.items():
    if cat == 'white':
        continue
    t, pp = ss.ttest_ind(groups['white'], vals)
    res7p[cat] = {'diff_vs_white': float(vals.mean() - white_mean), 'p': float(pp)}
results['iter7_race']['pairwise_vs_white'] = res7p

# === Iter 8: insurance ===
groups = {cat: df.loc[df['feature_018'] == cat, 'pfs_months'].values for cat in df['feature_018'].unique()}
f, p = ss.f_oneway(*groups.values())
results['iter8_insurance'] = {
    'overall_F': float(f), 'overall_p': float(p),
    'means': {k: float(v.mean()) for k, v in groups.items()},
    'ns': {k: int(len(v)) for k, v in groups.items()},
}

# === Iter 9: multivariable model ===
m = smf.ols('pfs_months ~ feature_078 + feature_057 + feature_051 + feature_038 + feature_013 + feature_043 + feature_099 + feature_092', data=df).fit()
results['iter9_mvm'] = {
    'r2': float(m.rsquared),
    'params': {k: float(v) for k, v in m.params.items()},
    'pvalues': {k: float(v) for k, v in m.pvalues.items()},
}

# === Iter 10: interaction 038 x 051 ===
m = smf.ols('pfs_months ~ feature_038 * feature_051 + feature_078 + feature_057', data=df).fit()
results['iter10_int_038x051'] = {
    'interaction_coef': float(m.params['feature_038:feature_051']),
    'interaction_p': float(m.pvalues['feature_038:feature_051']),
    'main_038': float(m.params['feature_038']),
    'main_051': float(m.params['feature_051']),
}

# === Iter 11: feature_038 by race ===
res11 = {}
for cat in df['feature_064'].unique():
    sub = df[df['feature_064'] == cat]
    g0 = sub.loc[sub['feature_038'] == 0, 'pfs_months']
    g1 = sub.loc[sub['feature_038'] == 1, 'pfs_months']
    if len(g0) > 10 and len(g1) > 10:
        t, p = ss.ttest_ind(g0, g1)
        res11[cat] = {'diff': float(g1.mean() - g0.mean()), 'p': float(p),
                      'n_0': int(len(g0)), 'n_1': int(len(g1))}
m_full = smf.ols('pfs_months ~ feature_038 * C(feature_064) + feature_078 + feature_057 + feature_051', data=df).fit()
m_red = smf.ols('pfs_months ~ feature_038 + C(feature_064) + feature_078 + feature_057 + feature_051', data=df).fit()
F = ((m_red.ssr - m_full.ssr) / 4) / (m_full.ssr / m_full.df_resid)
pf = 1 - ss.f.cdf(F, 4, m_full.df_resid)
res11['_interaction_F'] = float(F)
res11['_interaction_p'] = float(pf)
results['iter11_038_by_race'] = res11

# === Iter 12: feature_051 by race ===
res12 = {}
for cat in df['feature_064'].unique():
    sub = df[df['feature_064'] == cat]
    g0 = sub.loc[sub['feature_051'] == 0, 'pfs_months']
    g1 = sub.loc[sub['feature_051'] == 1, 'pfs_months']
    if len(g0) > 10 and len(g1) > 10:
        t, p = ss.ttest_ind(g0, g1)
        res12[cat] = {'diff': float(g1.mean() - g0.mean()), 'p': float(p),
                      'n_0': int(len(g0)), 'n_1': int(len(g1))}
m_full = smf.ols('pfs_months ~ feature_051 * C(feature_064) + feature_078 + feature_057 + feature_038', data=df).fit()
m_red = smf.ols('pfs_months ~ feature_051 + C(feature_064) + feature_078 + feature_057 + feature_038', data=df).fit()
F = ((m_red.ssr - m_full.ssr) / 4) / (m_full.ssr / m_full.df_resid)
pf = 1 - ss.f.cdf(F, 4, m_full.df_resid)
res12['_interaction_F'] = float(F)
res12['_interaction_p'] = float(pf)
results['iter12_051_by_race'] = res12

# === Iter 13: feature_038 by insurance ===
res13 = {}
for cat in df['feature_018'].unique():
    sub = df[df['feature_018'] == cat]
    g0 = sub.loc[sub['feature_038'] == 0, 'pfs_months']
    g1 = sub.loc[sub['feature_038'] == 1, 'pfs_months']
    if len(g0) > 10 and len(g1) > 10:
        t, p = ss.ttest_ind(g0, g1)
        res13[cat] = {'diff': float(g1.mean() - g0.mean()), 'p': float(p)}
m_full = smf.ols('pfs_months ~ feature_038 * C(feature_018) + feature_078 + feature_057 + feature_051', data=df).fit()
m_red = smf.ols('pfs_months ~ feature_038 + C(feature_018) + feature_078 + feature_057 + feature_051', data=df).fit()
F = ((m_red.ssr - m_full.ssr) / 3) / (m_full.ssr / m_full.df_resid)
pf = 1 - ss.f.cdf(F, 3, m_full.df_resid)
res13['_interaction_F'] = float(F)
res13['_interaction_p'] = float(pf)
results['iter13_038_by_insurance'] = res13

# === Iter 14: feature_038 x feature_057 ===
m = smf.ols('pfs_months ~ feature_038 * feature_057 + feature_078 + feature_051', data=df).fit()
strat14 = {}
for v in sorted(df['feature_057'].unique()):
    sub = df[df['feature_057'] == v]
    g0 = sub.loc[sub['feature_038'] == 0, 'pfs_months']
    g1 = sub.loc[sub['feature_038'] == 1, 'pfs_months']
    t, p = ss.ttest_ind(g0, g1)
    strat14[str(v)] = {'diff': float(g1.mean() - g0.mean()), 'p': float(p),
                       'n_0': int(len(g0)), 'n_1': int(len(g1))}
results['iter14_int_038x057'] = {
    'interaction_coef': float(m.params['feature_038:feature_057']),
    'interaction_p': float(m.pvalues['feature_038:feature_057']),
    'stratified': strat14,
}

# === Iter 15: feature_078 x feature_038 ===
m = smf.ols('pfs_months ~ feature_078 * feature_038 + feature_057 + feature_051', data=df).fit()
results['iter15_int_age038'] = {
    'interaction_coef': float(m.params['feature_078:feature_038']),
    'interaction_p': float(m.pvalues['feature_078:feature_038']),
}

# === Iter 16: feature_099 x feature_092 multivariable ===
m = smf.ols('pfs_months ~ feature_099 + feature_092 + feature_078 + feature_057', data=df).fit()
results['iter16_lab_adjusted'] = {
    'feature_099_coef': float(m.params['feature_099']),
    'feature_099_p': float(m.pvalues['feature_099']),
    'feature_092_coef': float(m.params['feature_092']),
    'feature_092_p': float(m.pvalues['feature_092']),
}

# === Iter 17: feature_071 ===
r, p = ss.pearsonr(df['feature_071'], df['pfs_months'])
m = smf.ols('pfs_months ~ feature_071 + feature_078 + feature_057', data=df).fit()
results['iter17_feature_071'] = {
    'pearson_r': float(r), 'pearson_p': float(p),
    'adj_coef': float(m.params['feature_071']), 'adj_p': float(m.pvalues['feature_071']),
}

# === Iter 18: 5-level ordinals ===
res18 = {}
for c in ['feature_025', 'feature_075', 'feature_026', 'feature_096', 'feature_033']:
    r, p = ss.pearsonr(df[c], df['pfs_months'])
    res18[c] = {'r': float(r), 'p': float(p)}
results['iter18_5level_ords'] = res18

# === Iter 19: feature_092 x feature_038 ===
m = smf.ols('pfs_months ~ feature_092 * feature_038 + feature_078 + feature_057 + feature_051', data=df).fit()
results['iter19_int_092x038'] = {
    'interaction_coef': float(m.params['feature_092:feature_038']),
    'interaction_p': float(m.pvalues['feature_092:feature_038']),
}

# === Iter 20: feature_051 x feature_057 ===
m = smf.ols('pfs_months ~ feature_051 * feature_057 + feature_078 + feature_038', data=df).fit()
strat20 = {}
for v in sorted(df['feature_057'].unique()):
    sub = df[df['feature_057'] == v]
    g0 = sub.loc[sub['feature_051'] == 0, 'pfs_months']
    g1 = sub.loc[sub['feature_051'] == 1, 'pfs_months']
    t, p = ss.ttest_ind(g0, g1)
    strat20[str(v)] = {'diff': float(g1.mean() - g0.mean()), 'p': float(p)}
results['iter20_int_051x057'] = {
    'interaction_coef': float(m.params['feature_051:feature_057']),
    'interaction_p': float(m.pvalues['feature_051:feature_057']),
    'stratified': strat20,
}

# === Iter 21: feature_078 by race ===
res21 = {}
for cat in df['feature_064'].unique():
    sub = df[df['feature_064'] == cat]
    if len(sub) > 100:
        r, p = ss.pearsonr(sub['feature_078'], sub['pfs_months'])
        res21[cat] = {'r': float(r), 'p': float(p), 'n': int(len(sub))}
results['iter21_age_by_race'] = res21

# === Iter 22: feature_106, feature_005, feature_122, feature_088 ===
res22 = {}
for c in ['feature_106', 'feature_005', 'feature_122', 'feature_088', 'feature_007']:
    g0 = df.loc[df[c] == 0, 'pfs_months']
    g1 = df.loc[df[c] == 1, 'pfs_months']
    t, p = ss.ttest_ind(g0, g1)
    res22[c] = {'diff': float(g1.mean() - g0.mean()), 'p': float(p),
                'n_0': int(len(g0)), 'n_1': int(len(g1))}
results['iter22_other_binaries'] = res22

# === Iter 23: 3-way feature_038 x feature_051 stratified by race ===
res23 = {}
for cat in df['feature_064'].unique():
    sub = df[df['feature_064'] == cat]
    if len(sub) > 200:
        try:
            mm = smf.ols('pfs_months ~ feature_038 * feature_051 + feature_078 + feature_057', data=sub).fit()
            res23[cat] = {
                'int_coef': float(mm.params.get('feature_038:feature_051', np.nan)),
                'int_p': float(mm.pvalues.get('feature_038:feature_051', np.nan)),
                'n': int(len(sub)),
            }
        except Exception as e:
            res23[cat] = {'error': str(e)}
results['iter23_3way'] = res23

# === Iter 24: comprehensive multivariable ===
m = smf.ols('pfs_months ~ feature_078 + feature_057 + feature_051 + feature_038 + feature_013 + feature_043 + feature_099 + feature_092 + feature_109 + feature_067 + C(feature_064) + C(feature_018)', data=df).fit()
results['iter24_full_mvm'] = {
    'r2': float(m.rsquared),
    'r2_adj': float(m.rsquared_adj),
    'params': {k: float(v) for k, v in m.params.items()},
    'pvalues': {k: float(v) for k, v in m.pvalues.items()},
}

# === Iter 25: feature_078 x feature_057 ===
m = smf.ols('pfs_months ~ feature_078 * feature_057 + feature_051 + feature_038', data=df).fit()
results['iter25_int_age_ps'] = {
    'interaction_coef': float(m.params['feature_078:feature_057']),
    'interaction_p': float(m.pvalues['feature_078:feature_057']),
}

with open('analysis_results.json', 'w') as fout:
    json.dump(results, fout, indent=2, default=str)

print('Done. R^2 of full model:', results['iter24_full_mvm']['r2'])
print('R^2 adj:', results['iter24_full_mvm']['r2_adj'])
