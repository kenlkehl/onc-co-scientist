import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f as f_dist
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings, json
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

with open('_work/analyses.json') as fp:
    R = json.load(fp)

def joint_interaction_test(full_formula, reduced_formula, data, mark_terms):
    mF = smf.ols(full_formula, data=data).fit()
    mR = smf.ols(reduced_formula, data=data).fit()
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

# ===== Iteration 11: PSA (feature_013) quartile-stratified PFS =====
df['_psa_q'] = pd.qcut(df['feature_013'], 4, labels=False)
psa_q_means = df.groupby('_psa_q')['pfs_months'].agg(['mean','count']).to_dict()
R['i11_psa_quartile_pfs'] = {f'q{int(k)}': {'mean': float(v), 'count': int(psa_q_means['count'][k])} for k,v in psa_q_means['mean'].items()}
print('i11 PSA quartile means:', R['i11_psa_quartile_pfs'])

# trend test - linear regression of mean PFS on quartile rank
m = smf.ols('pfs_months ~ _psa_q', data=df).fit()
R['i11_psa_q_trend'] = {'coef': float(m.params['_psa_q']), 'p': float(m.pvalues['_psa_q'])}
print('i11 trend coef=%.4f p=%.3e' % (m.params['_psa_q'], m.pvalues['_psa_q']))

# ===== Iteration 12: feature_057 stage interaction with PSA =====
res = joint_interaction_test(
    'pfs_months ~ np.log1p(feature_013) * C(feature_057) + feature_078',
    'pfs_months ~ np.log1p(feature_013) + C(feature_057) + feature_078',
    df, ['np.log1p(feature_013):']
)
R['i12_psa_x_stage'] = res
print('i12 PSA x stage joint p=%.4f' % res['p'])

# Check effect within each stage
for s in [0, 1, 2]:
    sub = df[df['feature_057']==s]
    r, p = stats.spearmanr(sub['feature_013'], sub['pfs_months'])
    R[f'i12_psa_within_stage{s}'] = {'r': float(r), 'p': float(p), 'n': int(len(sub))}
    print('  stage %d: spearman r=%.4f p=%.3e n=%d' % (s, r, p, len(sub)))

# ===== Iteration 13: race-stratified analysis of top predictors =====
for race in ['white','black','hispanic','asian','other']:
    sub = df[df['feature_018']==race]
    # f051 effect within race
    g0 = sub.loc[sub['feature_051']==0, 'pfs_months']
    g1 = sub.loc[sub['feature_051']==1, 'pfs_months']
    if len(g0)>5 and len(g1)>5:
        t, p = stats.ttest_ind(g1, g0)
        R[f'i13_f051_within_{race}'] = {'mean_diff': float(g1.mean()-g0.mean()), 'p': float(p), 'n0': int(len(g0)), 'n1': int(len(g1))}
        print('i13 %s: f051 diff=%.4f p=%.4f' % (race, g1.mean()-g0.mean(), p))

# ===== Iteration 14: insurance-stratified analysis of top predictors =====
for ins in ['private','medicare','medicaid','uninsured']:
    sub = df[df['feature_045']==ins]
    g0 = sub.loc[sub['feature_051']==0, 'pfs_months']
    g1 = sub.loc[sub['feature_051']==1, 'pfs_months']
    if len(g0)>5 and len(g1)>5:
        t, p = stats.ttest_ind(g1, g0)
        R[f'i14_f051_within_{ins}'] = {'mean_diff': float(g1.mean()-g0.mean()), 'p': float(p), 'n0': int(len(g0)), 'n1': int(len(g1))}
        print('i14 %s: f051 diff=%.4f p=%.4f' % (ins, g1.mean()-g0.mean(), p))

# ===== Iteration 15: continuous biomarker × top binary feature_051 =====
for col in ['feature_009', 'feature_006', 'feature_092', 'feature_094', 'feature_038', 'feature_028', 'feature_084']:
    fm = 'pfs_months ~ feature_051 * %s + feature_078 + C(feature_057)' % col
    m = smf.ols(fm, data=df).fit()
    int_t = 'feature_051:%s' % col
    if int_t in m.params.index:
        R[f'i15_f051_x_{col}'] = {'coef': float(m.params[int_t]), 'p': float(m.pvalues[int_t])}
        print('i15 f051 x %s coef=%.5f p=%.4f' % (col, m.params[int_t], m.pvalues[int_t]))

# ===== Iteration 16: combined effect of feature_109 and feature_039 (both protective) =====
df['_both_pf'] = (df['feature_109'].astype(int) + df['feature_039'].astype(int)).astype(int)
m = smf.ols('pfs_months ~ C(_both_pf) + feature_078 + C(feature_057)', data=df).fit()
print('i16 combined feature_109+feature_039:')
for c in m.params.index:
    if '_both_pf' in c:
        print('  %s coef=%.4f p=%.4f' % (c, m.params[c], m.pvalues[c]))
        R[f'i16_combo_109_039_{c}'] = {'coef': float(m.params[c]), 'p': float(m.pvalues[c])}

# ===== Iteration 17: race × age interaction =====
res = joint_interaction_test(
    'pfs_months ~ feature_078 * C(feature_018) + C(feature_057)',
    'pfs_months ~ feature_078 + C(feature_018) + C(feature_057)',
    df, ['feature_078:C(feature_018)', 'C(feature_018)']
)
# pick interaction terms only
mF = smf.ols('pfs_months ~ feature_078 * C(feature_018) + C(feature_057)', data=df).fit()
mR = smf.ols('pfs_months ~ feature_078 + C(feature_018) + C(feature_057)', data=df).fit()
rss_F = (mF.resid**2).sum(); rss_R = (mR.resid**2).sum()
int_terms = [c for c in mF.params.index if 'feature_078:C(feature_018)' in c]
df_diff = len(int_terms)
F = ((rss_R-rss_F)/df_diff) / (rss_F/mF.df_resid)
p_joint = 1 - f_dist.cdf(F, df_diff, mF.df_resid)
R['i17_age_x_race'] = {'F': float(F), 'p': float(p_joint), 'df': int(df_diff)}
print('i17 age x race joint p=%.4f' % p_joint)

# ===== Iteration 18: PSA × race =====
mF = smf.ols('pfs_months ~ np.log1p(feature_013) * C(feature_018) + feature_078 + C(feature_057)', data=df).fit()
mR = smf.ols('pfs_months ~ np.log1p(feature_013) + C(feature_018) + feature_078 + C(feature_057)', data=df).fit()
rss_F = (mF.resid**2).sum(); rss_R = (mR.resid**2).sum()
int_terms = [c for c in mF.params.index if 'np.log1p(feature_013):C(feature_018)' in c]
df_diff = len(int_terms)
F = ((rss_R-rss_F)/df_diff) / (rss_F/mF.df_resid)
p_joint = 1 - f_dist.cdf(F, df_diff, mF.df_resid)
R['i18_psa_x_race'] = {'F': float(F), 'p': float(p_joint), 'df': int(df_diff)}
print('i18 PSA x race joint p=%.4f' % p_joint)

# ===== Iteration 19: feature_057 × race =====
mF = smf.ols('pfs_months ~ C(feature_057) * C(feature_018) + feature_078', data=df).fit()
mR = smf.ols('pfs_months ~ C(feature_057) + C(feature_018) + feature_078', data=df).fit()
rss_F = (mF.resid**2).sum(); rss_R = (mR.resid**2).sum()
int_terms = [c for c in mF.params.index if 'C(feature_057)' in c and 'C(feature_018)' in c]
df_diff = len(int_terms)
F = ((rss_R-rss_F)/df_diff) / (rss_F/mF.df_resid)
p_joint = 1 - f_dist.cdf(F, df_diff, mF.df_resid)
R['i19_stage_x_race'] = {'F': float(F), 'p': float(p_joint), 'df': int(df_diff)}
print('i19 stage x race joint p=%.4f' % p_joint)

# ===== Iteration 20: feature_057 × insurance =====
mF = smf.ols('pfs_months ~ C(feature_057) * C(feature_045) + feature_078', data=df).fit()
mR = smf.ols('pfs_months ~ C(feature_057) + C(feature_045) + feature_078', data=df).fit()
rss_F = (mF.resid**2).sum(); rss_R = (mR.resid**2).sum()
int_terms = [c for c in mF.params.index if 'C(feature_057)' in c and 'C(feature_045)' in c]
df_diff = len(int_terms)
F = ((rss_R-rss_F)/df_diff) / (rss_F/mF.df_resid)
p_joint = 1 - f_dist.cdf(F, df_diff, mF.df_resid)
R['i20_stage_x_ins'] = {'F': float(F), 'p': float(p_joint), 'df': int(df_diff)}
print('i20 stage x ins joint p=%.4f' % p_joint)

# ===== Iteration 21: continuous variable (feature_006) effect by stage =====
for s in [0,1,2]:
    sub = df[df['feature_057']==s]
    r, p = stats.spearmanr(sub['feature_006'], sub['pfs_months'])
    R[f'i21_f006_within_stage{s}'] = {'r': float(r), 'p': float(p), 'n': int(len(sub))}
    print('i21 f006 within stage %d: r=%.4f p=%.3e' % (s, r, p))

# ===== Iteration 22: prevalence of features by race (treatment receipt disparities) =====
print('\ni22 binary feature prevalence by race (top binary features):')
for col in ['feature_051','feature_109','feature_039','feature_043','feature_057']:
    if df[col].nunique() <= 5:
        # chi-square independence
        ct = pd.crosstab(df[col], df['feature_018'])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        prev_by_race = df.groupby('feature_018')[col].mean().to_dict() if df[col].nunique()==2 else None
        R[f'i22_prev_{col}_by_race'] = {'chi2': float(chi2), 'p': float(p), 'prev_by_race': prev_by_race}
        print('  %s: chi2 p=%.4f, prev=%s' % (col, p, prev_by_race))

# ===== Iteration 23: prevalence of features by insurance =====
print('\ni23 binary feature prevalence by insurance:')
for col in ['feature_051','feature_109','feature_039','feature_043','feature_057']:
    if df[col].nunique() <= 5:
        ct = pd.crosstab(df[col], df['feature_045'])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        prev_by_ins = df.groupby('feature_045')[col].mean().to_dict() if df[col].nunique()==2 else None
        R[f'i23_prev_{col}_by_ins'] = {'chi2': float(chi2), 'p': float(p), 'prev_by_ins': prev_by_ins}
        print('  %s: chi2 p=%.4f, prev=%s' % (col, p, prev_by_ins))

# ===== Iteration 24: age (feature_078) prevalence by race/insurance =====
mage_race = df.groupby('feature_018')['feature_078'].mean().to_dict()
groups_ar = [g['feature_078'].values for _, g in df.groupby('feature_018')]
f_ar, p_ar = stats.f_oneway(*groups_ar)
R['i24_age_by_race'] = {'mean_age': mage_race, 'F': float(f_ar), 'p': float(p_ar)}
print('i24 age by race: ANOVA p=%.4f means=%s' % (p_ar, mage_race))

mage_ins = df.groupby('feature_045')['feature_078'].mean().to_dict()
groups_ai = [g['feature_078'].values for _, g in df.groupby('feature_045')]
f_ai, p_ai = stats.f_oneway(*groups_ai)
R['i24_age_by_ins'] = {'mean_age': mage_ins, 'F': float(f_ai), 'p': float(p_ai)}
print('i24 age by ins: ANOVA p=%.4f means=%s' % (p_ai, mage_ins))

# ===== Iteration 25: kitchen-sink final model with key interactions =====
formula = ('pfs_months ~ feature_078 + C(feature_057) + np.log1p(feature_013) + '
           'feature_051 + feature_006 + feature_009 + feature_109 + feature_039 + '
           'C(feature_018) + C(feature_045) + feature_078:feature_051 + '
           'C(feature_057):feature_051')
m = smf.ols(formula, data=df).fit()
R['i25_final_model'] = {
    'coef': {c: float(v) for c,v in m.params.items()},
    'pvalues': {c: float(v) for c,v in m.pvalues.items()},
    'r2': float(m.rsquared),
    'r2_adj': float(m.rsquared_adj),
    'n': int(m.nobs),
}
print('\ni25 final model R2=%.4f, R2_adj=%.4f' % (m.rsquared, m.rsquared_adj))
print('Significant terms (p<0.05):')
for c in m.params.index:
    if m.pvalues[c] < 0.05 and c != 'Intercept':
        print('  %s coef=%+.5f p=%.4e' % (c, m.params[c], m.pvalues[c]))

with open('_work/analyses.json', 'w') as fp:
    json.dump(R, fp, indent=2)
print('\nSaved.')
