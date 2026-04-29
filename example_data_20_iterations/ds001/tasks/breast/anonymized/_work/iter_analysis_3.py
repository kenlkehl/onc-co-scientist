"""Iterations 16-25."""
import pandas as pd
import numpy as np
import json
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
y = df['pfs_months']

iterations = json.load(open('_work/iters_partial.json'))

def add(it, hyps, ans):
    iterations.append({'index': it, 'proposed_hypotheses': hyps, 'analyses': ans})

def regress(formula, data=None):
    return smf.ols(formula, data=data if data is not None else df).fit()

# --- ITERATION 16: Does the protective effect of feature_042 persist after adjusting for feature_080 (the largest confounder)? ---
hyps = [
    {'id': 'h16.1', 'text': "After adjusting for feature_080, the protective effect of feature_042 on pfs_months is reduced (i.e., feature_080 partly mediates or confounds feature_042 effect).", 'kind': 'novel'},
    {'id': 'h16.2', 'text': "After adjusting for feature_080, the harmful effect of feature_056 on pfs_months remains essentially the same.", 'kind': 'novel'},
    {'id': 'h16.3', 'text': "feature_042=1 patients have a higher mean feature_080 than feature_042=0 patients (selection / case-mix effect).", 'kind': 'novel'},
    {'id': 'h16.4', 'text': "feature_056=1 patients have a different mean feature_080 than feature_056=0 patients.", 'kind': 'novel'},
]
ans = []
m_unadj = regress("pfs_months ~ feature_042")
m_adj = regress("pfs_months ~ feature_042 + feature_080")
ans.append({
    'hypothesis_ids': ['h16.1'],
    'code': "ols(pfs ~ f042) vs ols(pfs ~ f042 + f080)",
    'result_summary': f"Unadjusted f042 beta = {float(m_unadj.params['feature_042']):+.3f} (p={float(m_unadj.pvalues['feature_042']):.2e}). Adjusted for f080: beta = {float(m_adj.params['feature_042']):+.3f} (p={float(m_adj.pvalues['feature_042']):.2e}).",
    'p_value': float(m_adj.pvalues['feature_042']), 'effect_estimate': float(m_adj.params['feature_042']) - float(m_unadj.params['feature_042']),
    'significant': bool(abs(m_adj.params['feature_042'] - m_unadj.params['feature_042']) > 0.05),
})
m_unadj56 = regress("pfs_months ~ feature_056")
m_adj56 = regress("pfs_months ~ feature_056 + feature_080")
ans.append({
    'hypothesis_ids': ['h16.2'],
    'code': "ols(pfs ~ f056) vs ols(pfs ~ f056 + f080)",
    'result_summary': f"Unadjusted f056 beta = {float(m_unadj56.params['feature_056']):+.3f}. Adjusted for f080: beta = {float(m_adj56.params['feature_056']):+.3f} (p={float(m_adj56.pvalues['feature_056']):.2e}).",
    'p_value': float(m_adj56.pvalues['feature_056']), 'effect_estimate': float(m_adj56.params['feature_056']) - float(m_unadj56.params['feature_056']),
    'significant': bool(abs(m_adj56.params['feature_056'] - m_unadj56.params['feature_056']) > 0.05),
})
mn1 = float(df.loc[df['feature_042']==1, 'feature_080'].mean()); mn0 = float(df.loc[df['feature_042']==0, 'feature_080'].mean())
t,p = stats.ttest_ind(df.loc[df['feature_042']==1, 'feature_080'], df.loc[df['feature_042']==0, 'feature_080'], equal_var=False)
ans.append({
    'hypothesis_ids': ['h16.3'],
    'code': "ttest of feature_080 by feature_042",
    'result_summary': f"Mean feature_080: f042=1 -> {mn1:.2f}, f042=0 -> {mn0:.2f}. Diff={mn1-mn0:+.3f}, t-test p={float(p):.3e}.",
    'p_value': float(p), 'effect_estimate': float(mn1-mn0), 'significant': bool(p<0.05),
})
mn1 = float(df.loc[df['feature_056']==1, 'feature_080'].mean()); mn0 = float(df.loc[df['feature_056']==0, 'feature_080'].mean())
t,p = stats.ttest_ind(df.loc[df['feature_056']==1, 'feature_080'], df.loc[df['feature_056']==0, 'feature_080'], equal_var=False)
ans.append({
    'hypothesis_ids': ['h16.4'],
    'code': "ttest of feature_080 by feature_056",
    'result_summary': f"Mean feature_080: f056=1 -> {mn1:.2f}, f056=0 -> {mn0:.2f}. Diff={mn1-mn0:+.3f}, t-test p={float(p):.3e}.",
    'p_value': float(p), 'effect_estimate': float(mn1-mn0), 'significant': bool(p<0.05),
})
add(16, hyps, ans)

# --- ITERATION 17: race-stratified outcomes after adjusting for clinical case-mix ---
hyps = [
    {'id': 'h17.1', 'text': "After adjusting for the strong predictors (feature_080, feature_063, feature_042, feature_056), pfs_months still differs across racial groups (feature_011) - i.e., a residual disparity exists.", 'kind': 'novel'},
    {'id': 'h17.2', 'text': "After adjusting for the strong predictors, pfs_months still differs across insurance groups (feature_089).", 'kind': 'novel'},
    {'id': 'h17.3', 'text': "Distributions of feature_042 (treatment-like) differ by race (feature_011) -- chi-square test of independence.", 'kind': 'novel'},
    {'id': 'h17.4', 'text': "Distributions of feature_042 (treatment-like) differ by insurance (feature_089).", 'kind': 'novel'},
]
ans = []
m_full = regress("pfs_months ~ feature_080 + feature_063 + feature_042 + feature_056 + C(feature_011)")
race_terms = [k for k in m_full.params.index if k.startswith('C(feature_011)')]
ft = m_full.f_test(race_terms)
ans.append({
    'hypothesis_ids': ['h17.1'],
    'code': "ols(pfs ~ f080 + f063 + f042 + f056 + C(feature_011)); F-test on race terms",
    'result_summary': f"Adjusted joint F-test for race terms: p={float(ft.pvalue):.3e}. Race coefs (vs reference): " + "; ".join(f"{k.replace('C(feature_011)[T.','')[:-1]}={float(m_full.params[k]):+.3f}" for k in race_terms),
    'p_value': float(ft.pvalue), 'effect_estimate': float(max(m_full.params[k] for k in race_terms) - min(m_full.params[k] for k in race_terms)) if race_terms else 0,
    'significant': bool(ft.pvalue<0.05),
})
m_full2 = regress("pfs_months ~ feature_080 + feature_063 + feature_042 + feature_056 + C(feature_089)")
ins_terms = [k for k in m_full2.params.index if k.startswith('C(feature_089)')]
ft2 = m_full2.f_test(ins_terms)
ans.append({
    'hypothesis_ids': ['h17.2'],
    'code': "ols(pfs ~ f080 + f063 + f042 + f056 + C(feature_089))",
    'result_summary': f"Adjusted joint F-test for insurance terms: p={float(ft2.pvalue):.3e}. Insurance coefs: " + "; ".join(f"{k.replace('C(feature_089)[T.','')[:-1]}={float(m_full2.params[k]):+.3f}" for k in ins_terms),
    'p_value': float(ft2.pvalue), 'effect_estimate': float(max(m_full2.params[k] for k in ins_terms) - min(m_full2.params[k] for k in ins_terms)) if ins_terms else 0,
    'significant': bool(ft2.pvalue<0.05),
})
ct = pd.crosstab(df['feature_011'], df['feature_042'])
chi2, p, _, _ = stats.chi2_contingency(ct)
prop = ct.div(ct.sum(axis=1), axis=0)[1]
ans.append({
    'hypothesis_ids': ['h17.3'],
    'code': "chi2_contingency of feature_011 vs feature_042",
    'result_summary': f"Chi-square: chi2={chi2:.3f}, p={p:.3e}. Proportion feature_042=1 by race: " + "; ".join(f"{r}={prop[r]:.3f}" for r in prop.index),
    'p_value': float(p), 'effect_estimate': float(prop.max()-prop.min()), 'significant': bool(p<0.05),
})
ct = pd.crosstab(df['feature_089'], df['feature_042'])
chi2, p, _, _ = stats.chi2_contingency(ct)
prop = ct.div(ct.sum(axis=1), axis=0)[1]
ans.append({
    'hypothesis_ids': ['h17.4'],
    'code': "chi2_contingency of feature_089 vs feature_042",
    'result_summary': f"Chi-square: chi2={chi2:.3f}, p={p:.3e}. Proportion feature_042=1 by insurance: " + "; ".join(f"{r}={prop[r]:.3f}" for r in prop.index),
    'p_value': float(p), 'effect_estimate': float(prop.max()-prop.min()), 'significant': bool(p<0.05),
})
add(17, hyps, ans)

# --- ITERATION 18: variable importance via drop-1 R^2 ---
hyps = [
    {'id': 'h18.1', 'text': "Among feature_080, feature_063, feature_056, feature_042, feature_048, feature_111, feature_015, feature_040, feature_039, feature_019, feature_067 and feature_101, dropping feature_080 from the multivariable model causes by far the largest decrease in R^2 (i.e. it is the single most important predictor).", 'kind': 'novel'},
]
ans = []
predictors = ['feature_080','feature_063','feature_042','feature_056','feature_048','feature_111','feature_015','feature_040','feature_039','feature_019','feature_067','feature_101']
m_full = regress("pfs_months ~ " + " + ".join(predictors))
r2_full = m_full.rsquared
drops = {}
for p_ in predictors:
    rest = [x for x in predictors if x != p_]
    m_d = regress("pfs_months ~ " + " + ".join(rest))
    drops[p_] = float(r2_full - m_d.rsquared)
ranked = sorted(drops.items(), key=lambda kv: -kv[1])
ans.append({
    'hypothesis_ids': ['h18.1'],
    'code': "Drop-1 R^2 reduction on the 12-predictor multivariable model",
    'result_summary': f"Full R2={r2_full:.4f}. Drop-1 dR2 (sorted): " + ", ".join(f"{k}={v:.4f}" for k,v in ranked) + ".",
    'p_value': None, 'effect_estimate': float(ranked[0][1]), 'significant': bool(ranked[0][0]=='feature_080'),
})
add(18, hyps, ans)

# --- ITERATION 19: feature_063 x feature_042 interaction (does treatment benefit depend on performance status?) ---
hyps = [
    {'id': 'h19.1', 'text': "There is a significant interaction between feature_063 (ordinal 0-2) and feature_042 (binary): the protective effect of feature_042 differs by feature_063 level.", 'kind': 'novel'},
    {'id': 'h19.2', 'text': "There is a significant interaction between feature_063 and feature_056 on pfs_months.", 'kind': 'novel'},
]
ans = []
for hid, c1, c2 in [('h19.1','feature_063','feature_042'),('h19.2','feature_063','feature_056')]:
    m = regress(f"pfs_months ~ {c1} * {c2}")
    inter = f"{c1}:{c2}"
    b = float(m.params[inter]); p = float(m.pvalues[inter])
    eff = {}
    for lv in sorted(df[c1].unique()):
        sub = df[df[c1]==lv]
        a = sub.loc[sub[c2]==1, 'pfs_months'].values
        bb = sub.loc[sub[c2]==0, 'pfs_months'].values
        t,pp = stats.ttest_ind(a, bb, equal_var=False)
        eff[int(lv)] = (float(a.mean()-bb.mean()), float(pp), int(len(a)), int(len(bb)))
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"ols(pfs ~ {c1} * {c2})",
        'result_summary': f"Interaction beta = {b:+.3f}, p={p:.3e}. Effect of {c2} stratified by {c1}: {eff}.",
        'p_value': p, 'effect_estimate': b, 'significant': bool(p<0.05),
    })
add(19, hyps, ans)

# --- ITERATION 20: Does feature_080 effect depend on race? ---
hyps = [
    {'id': 'h20.1', 'text': "The slope of feature_080 on pfs_months differs by race (feature_011 x feature_080 interaction).", 'kind': 'novel'},
    {'id': 'h20.2', 'text': "The slope of feature_080 on pfs_months differs by insurance (feature_089 x feature_080 interaction).", 'kind': 'novel'},
]
ans = []
for hid, cat in [('h20.1','feature_011'),('h20.2','feature_089')]:
    m = regress(f"pfs_months ~ feature_080 * C({cat})")
    inter_terms = [k for k in m.params.index if k.startswith('feature_080:C(')]
    ft = m.f_test(inter_terms)
    p = float(ft.pvalue)
    slopes = {}
    for lv in df[cat].unique():
        sub = df[df[cat]==lv]
        mm = smf.ols("pfs_months ~ feature_080", data=sub).fit()
        slopes[lv] = (float(mm.params['feature_080']), int(len(sub)))
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"ols(pfs ~ feature_080 * C({cat})); F-test on interaction terms",
        'result_summary': f"Joint F-test on feature_080 x C({cat}) interaction: p={p:.3e}. Stratified slopes: {slopes}.",
        'p_value': p, 'effect_estimate': float(max(v[0] for v in slopes.values()) - min(v[0] for v in slopes.values())),
        'significant': bool(p<0.05),
    })
add(20, hyps, ans)

# --- ITERATION 21: Predictors of extreme outcomes (long PFS > 10 mo vs short PFS < 2 mo) ---
hyps = [
    {'id': 'h21.1', 'text': "Among patients with pfs_months > 10 mo, feature_080 distribution is concentrated at the high end (>= 75); among patients with pfs_months < 2, feature_080 is concentrated at the low end.", 'kind': 'novel'},
    {'id': 'h21.2', 'text': "feature_042 prevalence is significantly higher among long-PFS (>10mo) than short-PFS (<2mo) patients.", 'kind': 'novel'},
    {'id': 'h21.3', 'text': "feature_056 prevalence is significantly higher among short-PFS (<2mo) than long-PFS (>10mo) patients.", 'kind': 'novel'},
]
ans = []
long_p = df[y > 10]; short_p = df[y < 2]
m_long = float(long_p['feature_080'].mean()); m_short = float(short_p['feature_080'].mean())
t,p = stats.ttest_ind(long_p['feature_080'], short_p['feature_080'], equal_var=False)
ans.append({
    'hypothesis_ids': ['h21.1'],
    'code': "feature_080 distribution: long_PFS vs short_PFS",
    'result_summary': f"Mean feature_080 in long-PFS (>10mo, n={len(long_p)}) = {m_long:.2f}; short-PFS (<2mo, n={len(short_p)}) = {m_short:.2f}. Diff={m_long-m_short:+.2f}, p={float(p):.3e}.",
    'p_value': float(p), 'effect_estimate': float(m_long-m_short), 'significant': bool(p<0.05),
})
ct = np.array([[long_p['feature_042'].sum(), len(long_p)-long_p['feature_042'].sum()],
               [short_p['feature_042'].sum(), len(short_p)-short_p['feature_042'].sum()]])
chi2, p, _, _ = stats.chi2_contingency(ct)
ans.append({
    'hypothesis_ids': ['h21.2'],
    'code': "chi2 of feature_042 in long vs short PFS strata",
    'result_summary': f"Prevalence feature_042=1: long-PFS={long_p['feature_042'].mean():.3f}, short-PFS={short_p['feature_042'].mean():.3f}. Chi-square chi2={chi2:.1f}, p={float(p):.3e}.",
    'p_value': float(p), 'effect_estimate': float(long_p['feature_042'].mean() - short_p['feature_042'].mean()), 'significant': bool(p<0.05),
})
ct = np.array([[long_p['feature_056'].sum(), len(long_p)-long_p['feature_056'].sum()],
               [short_p['feature_056'].sum(), len(short_p)-short_p['feature_056'].sum()]])
chi2, p, _, _ = stats.chi2_contingency(ct)
ans.append({
    'hypothesis_ids': ['h21.3'],
    'code': "chi2 of feature_056 in long vs short PFS strata",
    'result_summary': f"Prevalence feature_056=1: long-PFS={long_p['feature_056'].mean():.3f}, short-PFS={short_p['feature_056'].mean():.3f}. Chi-square chi2={chi2:.1f}, p={float(p):.3e}.",
    'p_value': float(p), 'effect_estimate': float(long_p['feature_056'].mean() - short_p['feature_056'].mean()), 'significant': bool(p<0.05),
})
add(21, hyps, ans)

# --- ITERATION 22: All previously untested binary features that did rise in screen — feature_086, feature_077, feature_031, feature_005 ---
hyps = [
    {'id': 'h22.1', 'text': "feature_086 (binary, ~15%) is associated with pfs_months in the screening direction (negative).", 'kind': 'novel'},
    {'id': 'h22.2', 'text': "feature_077 (binary, ~35%) is positively associated with pfs_months but the effect is small (<0.1 mo).", 'kind': 'novel'},
    {'id': 'h22.3', 'text': "feature_031 (binary, ~20%) is negatively associated with pfs_months (small).", 'kind': 'novel'},
    {'id': 'h22.4', 'text': "feature_005 (binary, ~30%) is negatively associated with pfs_months.", 'kind': 'novel'},
    {'id': 'h22.5', 'text': "feature_034 (binary, ~37%) is positively associated with pfs_months (small).", 'kind': 'novel'},
]
ans = []
def t_(c):
    a = df.loc[df[c]==1,'pfs_months'].values; b = df.loc[df[c]==0,'pfs_months'].values
    t,p = stats.ttest_ind(a,b,equal_var=False)
    return float(a.mean()-b.mean()), float(p), int(len(a)), int(len(b))
for hid, c in [('h22.1','feature_086'),('h22.2','feature_077'),('h22.3','feature_031'),('h22.4','feature_005'),('h22.5','feature_034')]:
    e, p, n1, n0 = t_(c)
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"ttest on {c}",
        'result_summary': f"{c}: diff={e:+.3f} mo (n1={n1}, n0={n0}), p={p:.3e}.",
        'p_value': p, 'effect_estimate': e, 'significant': bool(p<0.05),
    })
add(22, hyps, ans)

# --- ITERATION 23: rare features tested within the elderly subgroup (feature_080>=75) ---
hyps = [
    {'id': 'h23.1', 'text': "Within elderly patients (feature_080 >= 75), the protective effect of feature_042 is preserved (>= 0.5 mo).", 'kind': 'novel'},
    {'id': 'h23.2', 'text': "Within young patients (feature_080 < 50), the protective effect of feature_042 is preserved (>= 0.5 mo).", 'kind': 'novel'},
    {'id': 'h23.3', 'text': "Within elderly patients (feature_080 >= 75), the harmful effect of feature_056 is preserved.", 'kind': 'novel'},
    {'id': 'h23.4', 'text': "Within young patients (feature_080 < 50), the harmful effect of feature_056 is preserved.", 'kind': 'novel'},
]
ans = []
elder = df[df['feature_080']>=75]; young = df[df['feature_080']<50]
for hid, sub, label, c in [('h23.1', elder, 'elderly>=75','feature_042'),('h23.2', young, 'young<50','feature_042'),('h23.3', elder, 'elderly>=75','feature_056'),('h23.4', young, 'young<50','feature_056')]:
    a = sub.loc[sub[c]==1,'pfs_months'].values; b = sub.loc[sub[c]==0,'pfs_months'].values
    t,p = stats.ttest_ind(a,b,equal_var=False)
    e = float(a.mean()-b.mean())
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"ttest of {c} within {label}",
        'result_summary': f"{label}: {c}=1 mean={a.mean():.3f} (n={len(a)}); {c}=0 mean={b.mean():.3f} (n={len(b)}); diff={e:+.3f} mo, p={float(p):.3e}.",
        'p_value': float(p), 'effect_estimate': e, 'significant': bool(p<0.05),
    })
add(23, hyps, ans)

# --- ITERATION 24: residuals after subtracting predicted - what remains? ---
hyps = [
    {'id': 'h24.1', 'text': "After conditioning on the 12 strong predictors (the iter-6 multivariable model), the residuals show no remaining association with race feature_011 (no residual race disparity).", 'kind': 'novel'},
    {'id': 'h24.2', 'text': "After conditioning on the 12 strong predictors, the residuals show no remaining association with insurance feature_089.", 'kind': 'novel'},
]
ans = []
predictors = "feature_080 + feature_063 + feature_042 + feature_056 + feature_048 + feature_111 + feature_015 + feature_040 + feature_039 + feature_019 + feature_067 + feature_101"
m_full = regress(f"pfs_months ~ {predictors}")
df['_resid'] = m_full.resid
groups = [df.loc[df['feature_011']==g, '_resid'].values for g in df['feature_011'].unique()]
F, p = stats.f_oneway(*groups)
means = {g: float(df.loc[df['feature_011']==g, '_resid'].mean()) for g in df['feature_011'].unique()}
ans.append({
    'hypothesis_ids': ['h24.1'],
    'code': "f_oneway of residuals across feature_011",
    'result_summary': f"Residual ANOVA across race: F={F:.3f}, p={float(p):.3e}. Mean residual by race: {means}.",
    'p_value': float(p), 'effect_estimate': float(max(means.values())-min(means.values())), 'significant': bool(p<0.05),
})
groups = [df.loc[df['feature_089']==g, '_resid'].values for g in df['feature_089'].unique()]
F, p = stats.f_oneway(*groups)
means = {g: float(df.loc[df['feature_089']==g, '_resid'].mean()) for g in df['feature_089'].unique()}
ans.append({
    'hypothesis_ids': ['h24.2'],
    'code': "f_oneway of residuals across feature_089",
    'result_summary': f"Residual ANOVA across insurance: F={F:.3f}, p={float(p):.3e}. Mean residual by insurance: {means}.",
    'p_value': float(p), 'effect_estimate': float(max(means.values())-min(means.values())), 'significant': bool(p<0.05),
})
df.drop(columns=['_resid'], inplace=True)
add(24, hyps, ans)

# --- ITERATION 25: Out-of-sample R^2 of the multivariable OLS model + univariate explained variance ---
hyps = [
    {'id': 'h25.1', 'text': "A multivariable OLS model containing all 127 features (with one-hot encoding for the 2 categoricals) achieves an out-of-sample R^2 of at least 0.70 for predicting pfs_months under 5-fold cross-validation.", 'kind': 'novel'},
    {'id': 'h25.2', 'text': "feature_080 alone explains at least 50% of pfs_months variance (R^2 >= 0.50).", 'kind': 'novel'},
]
ans = []
X = pd.get_dummies(df.drop(columns=['patient_id','pfs_months']), drop_first=True).astype(float)
X = sm.add_constant(X)
yv = y.values
n = len(df)
rng = np.random.RandomState(42)
idx = rng.permutation(n)
folds = np.array_split(idx, 5)
r2_oos = []
for k in range(5):
    test_idx = folds[k]
    train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
    Xtr = X.iloc[train_idx].values; ytr = yv[train_idx]
    Xte = X.iloc[test_idx].values; yte = yv[test_idx]
    # OLS via pseudoinverse for stability
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    pred = Xte @ beta
    ss_res = ((yte - pred)**2).sum()
    ss_tot = ((yte - yte.mean())**2).sum()
    r2_oos.append(1 - ss_res/ss_tot)
r2_oos = np.array(r2_oos)
ans.append({
    'hypothesis_ids': ['h25.1'],
    'code': "5-fold CV OLS via numpy lstsq on all 127 features (one-hot encoded)",
    'result_summary': f"Cross-validated out-of-sample R2 (5-fold OLS): mean={r2_oos.mean():.4f}, std={r2_oos.std():.4f}, folds={list(np.round(r2_oos,4))}.",
    'p_value': None, 'effect_estimate': float(r2_oos.mean()), 'significant': bool(r2_oos.mean()>=0.70),
})
m = regress("pfs_months ~ feature_080")
ans.append({
    'hypothesis_ids': ['h25.2'],
    'code': "ols(pfs_months ~ feature_080).rsquared",
    'result_summary': f"feature_080 alone: R2={m.rsquared:.4f} (= {m.rsquared*100:.1f}% of variance).",
    'p_value': None, 'effect_estimate': float(m.rsquared), 'significant': bool(m.rsquared>=0.50),
})
add(25, hyps, ans)

with open('_work/iters_partial.json','w') as f:
    json.dump(iterations, f, indent=2, default=str)
print(f"Iterations 16-25 done. Total: {len(iterations)}")
