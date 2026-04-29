import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings, json
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

R = {}

def reg(formula, key, data=df):
    m = smf.ols(formula, data=data).fit()
    R[key] = {
        'coef': m.params.to_dict(),
        'pvalues': m.pvalues.to_dict(),
        'n': int(m.nobs),
        'r2': float(m.rsquared),
    }
    return m

# ===== Iteration 1: demographics main effects =====
m = reg('pfs_months ~ feature_078', 'i1_age')
print('i1_age coef=%.5f p=%.3e' % (m.params['feature_078'], m.pvalues['feature_078']))

groups = [g['pfs_months'].values for _, g in df.groupby('feature_018')]
f, p = stats.f_oneway(*groups)
race_means = df.groupby('feature_018')['pfs_months'].mean().to_dict()
R['i1_race_anova'] = {'F': float(f), 'p': float(p), 'means': race_means}
print('i1_race anova p=%.3e means=%s' % (p, race_means))

groups = [g['pfs_months'].values for _, g in df.groupby('feature_045')]
f, p = stats.f_oneway(*groups)
ins_means = df.groupby('feature_045')['pfs_months'].mean().to_dict()
R['i1_insurance_anova'] = {'F': float(f), 'p': float(p), 'means': ins_means}
print('i1_insurance anova p=%.3e means=%s' % (p, ins_means))

# ===== Iteration 2: top clinical features =====
r, p = stats.spearmanr(df['feature_057'], df['pfs_months'])
R['i2_f057_spearman'] = {'r': float(r), 'p': float(p)}
m = smf.ols('pfs_months ~ feature_057', data=df).fit()
R['i2_f057_lin'] = {'coef': float(m.params['feature_057']), 'p': float(m.pvalues['feature_057'])}
print('i2_f057 spearman r=%.4f p=%.3e' % (r, p))

r, p = stats.spearmanr(df['feature_013'], df['pfs_months'])
R['i2_f013_spearman'] = {'r': float(r), 'p': float(p)}
print('i2_f013 spearman r=%.4f p=%.3e' % (r, p))

g0 = df.loc[df['feature_051']==0, 'pfs_months']
g1 = df.loc[df['feature_051']==1, 'pfs_months']
t, p = stats.ttest_ind(g1, g0)
R['i2_f051_tt'] = {'mean_diff': float(g1.mean()-g0.mean()), 'p': float(p), 'mean_0': float(g0.mean()), 'mean_1': float(g1.mean()), 'n0': int(len(g0)), 'n1': int(len(g1))}
print('i2_f051 diff=%.4f p=%.3e' % (g1.mean()-g0.mean(), p))

# ===== Iteration 3: more clinical features =====
for col in ['feature_006', 'feature_009', 'feature_109', 'feature_039', 'feature_038', 'feature_103', 'feature_028', 'feature_084', 'feature_092', 'feature_094', 'feature_014']:
    if df[col].nunique() == 2:
        g0 = df.loc[df[col]==0, 'pfs_months']
        g1 = df.loc[df[col]==1, 'pfs_months']
        t, p = stats.ttest_ind(g1, g0)
        R['i3_%s_tt' % col] = {'mean_diff': float(g1.mean()-g0.mean()), 'p': float(p), 'n0': int(len(g0)), 'n1': int(len(g1))}
    else:
        r, p = stats.spearmanr(df[col], df['pfs_months'])
        R['i3_%s_spearman' % col] = {'r': float(r), 'p': float(p)}
    print('i3_%s' % col, R.get('i3_%s_tt' % col) or R.get('i3_%s_spearman' % col))

# ===== Iteration 4: full multivariable model =====
formula = 'pfs_months ~ feature_078 + C(feature_057) + np.log1p(feature_013) + feature_051 + feature_006 + feature_009 + C(feature_018) + C(feature_045)'
m = smf.ols(formula, data=df).fit()
R['i4_multivar'] = {
    'coef': m.params.to_dict(),
    'pvalues': m.pvalues.to_dict(),
    'n': int(m.nobs),
    'r2': float(m.rsquared),
    'r2_adj': float(m.rsquared_adj),
}
print('i4_multivar R2=%.4f' % m.rsquared)
for c in m.params.index:
    if 'feature_018' in c or 'feature_045' in c:
        print('  %s coef=%.4f p=%.4f' % (c, m.params[c], m.pvalues[c]))

# ===== Iteration 5: pairwise demographics =====
white_pfs = df.loc[df['feature_018']=='white', 'pfs_months']
for race in ['black', 'hispanic', 'asian', 'other']:
    r_pfs = df.loc[df['feature_018']==race, 'pfs_months']
    t, p = stats.ttest_ind(r_pfs, white_pfs)
    R['i5_race_%s_vs_white' % race] = {'mean_diff': float(r_pfs.mean()-white_pfs.mean()), 'p': float(p), 'n_race': int(len(r_pfs)), 'n_white': int(len(white_pfs))}
    print('i5 %s vs white: diff=%.4f p=%.4f' % (race, r_pfs.mean()-white_pfs.mean(), p))

priv_pfs = df.loc[df['feature_045']=='private', 'pfs_months']
for ins in ['medicare', 'medicaid', 'uninsured']:
    i_pfs = df.loc[df['feature_045']==ins, 'pfs_months']
    t, p = stats.ttest_ind(i_pfs, priv_pfs)
    R['i5_ins_%s_vs_private' % ins] = {'mean_diff': float(i_pfs.mean()-priv_pfs.mean()), 'p': float(p), 'n_ins': int(len(i_pfs)), 'n_priv': int(len(priv_pfs))}
    print('i5 %s vs private: diff=%.4f p=%.4f' % (ins, i_pfs.mean()-priv_pfs.mean(), p))

# ===== Iterations 6-9: interactions =====
from scipy.stats import f as f_dist

def joint_interaction_test(full_formula, reduced_formula, df_data, mark_terms):
    mF = smf.ols(full_formula, data=df_data).fit()
    mR = smf.ols(reduced_formula, data=df_data).fit()
    rss_F = (mF.resid**2).sum()
    rss_R = (mR.resid**2).sum()
    int_terms = [c for c in mF.params.index if any(t in c for t in mark_terms)]
    df_diff = len(int_terms)
    if df_diff == 0:
        return None
    F = ((rss_R-rss_F)/df_diff) / (rss_F/mF.df_resid)
    p_joint = 1 - f_dist.cdf(F, df_diff, mF.df_resid)
    int_co = {c: float(mF.params[c]) for c in int_terms}
    int_pv = {c: float(mF.pvalues[c]) for c in int_terms}
    return {'F': float(F), 'p': float(p_joint), 'df': df_diff, 'coef': int_co, 'pvalues': int_pv}

top_binary = ['feature_051', 'feature_109', 'feature_039', 'feature_043', 'feature_035']

# i6: race x binary
for col in top_binary:
    res = joint_interaction_test(
        'pfs_months ~ %s * C(feature_018) + feature_078 + C(feature_057)' % col,
        'pfs_months ~ %s + C(feature_018) + feature_078 + C(feature_057)' % col,
        df, ['%s:' % col, ':%s' % col]
    )
    R['i6_race_x_%s' % col] = res
    print('i6 race x %s joint p=%.4f' % (col, res['p']))

# i7: insurance x binary
for col in top_binary:
    res = joint_interaction_test(
        'pfs_months ~ %s * C(feature_045) + feature_078 + C(feature_057)' % col,
        'pfs_months ~ %s + C(feature_045) + feature_078 + C(feature_057)' % col,
        df, ['%s:' % col, ':%s' % col]
    )
    R['i7_ins_x_%s' % col] = res
    print('i7 ins x %s joint p=%.4f' % (col, res['p']))

# i8: age x binary
for col in top_binary:
    fm = 'pfs_months ~ %s * feature_078 + C(feature_057)' % col
    m = smf.ols(fm, data=df).fit()
    int_term = '%s:feature_078' % col
    if int_term not in m.params.index:
        continue
    R['i8_age_x_%s' % col] = {'coef': float(m.params[int_term]), 'p': float(m.pvalues[int_term])}
    print('i8 age x %s coef=%.5f p=%.4f' % (col, m.params[int_term], m.pvalues[int_term]))

# i9: stage x binary
for col in top_binary:
    res = joint_interaction_test(
        'pfs_months ~ C(%s) * C(feature_057) + feature_078' % col,
        'pfs_months ~ C(%s) + C(feature_057) + feature_078' % col,
        df, ['C(%s)[T.1]:C(feature_057)' % col]
    )
    R['i9_stage_x_%s' % col] = res
    print('i9 stage x %s joint p=%.4f' % (col, res['p']))

# ===== Iteration 10: adjusted binary feature screen =====
binary_cols = [c for c in df.columns if c not in ['patient_id','pfs_months'] and df[c].dtype != 'object' and df[c].nunique() == 2]
adj_results = []
for col in binary_cols:
    if df[col].std() == 0: continue
    fm = 'pfs_months ~ %s + feature_078 + C(feature_057)' % col
    try:
        m = smf.ols(fm, data=df).fit()
        adj_results.append((col, float(m.params[col]), float(m.pvalues[col])))
    except Exception as e:
        pass

adj_results.sort(key=lambda x: x[2])
print('Top adjusted binary:')
for r in adj_results[:25]:
    print('  %s coef=%+.4f p=%.4e' % (r[0], r[1], r[2]))
R['i10_adj_binary_screen_top25'] = [{'feature': r[0], 'coef': r[1], 'p': r[2]} for r in adj_results[:25]]
R['i10_adj_binary_screen_full'] = {r[0]: {'coef': r[1], 'p': r[2]} for r in adj_results}

with open('_work/analyses.json', 'w') as fp:
    json.dump(R, fp, indent=2)
print('Saved analyses.json')
