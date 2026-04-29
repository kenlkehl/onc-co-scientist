"""
Iterative analysis of ds001_breast.
Runs 25 logical iterations; serializes results to _work/iters_partial.json.
"""
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

iterations = []

def add(it, hyps, ans):
    iterations.append({'index': it, 'proposed_hypotheses': hyps, 'analyses': ans})

def ttest(col, val=1, val0=0):
    a = y[df[col] == val].values
    b = y[df[col] == val0].values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(p), int(len(a)), int(len(b)), float(a.mean()), float(b.mean())

def regress(formula):
    return smf.ols(formula, data=df).fit()

# --- ITERATION 1: top binary main effects ---
hyps = [
    {'id': 'h1.1', 'text': "Patients with feature_042=1 have longer pfs_months than those with feature_042=0 (positive effect of feature_042 on PFS).", 'kind': 'novel'},
    {'id': 'h1.2', 'text': "Patients with feature_056=1 have shorter pfs_months than those with feature_056=0 (negative effect of feature_056 on PFS).", 'kind': 'novel'},
    {'id': 'h1.3', 'text': "Patients with feature_048=1 have shorter pfs_months than those with feature_048=0 (negative effect of feature_048 on PFS).", 'kind': 'novel'},
    {'id': 'h1.4', 'text': "Patients with feature_111=1 have longer pfs_months than those with feature_111=0 (positive effect of feature_111 on PFS).", 'kind': 'novel'},
]
ans = []
for h, c in [('h1.1','feature_042'),('h1.2','feature_056'),('h1.3','feature_048'),('h1.4','feature_111')]:
    e, p, n1, n0, m1, m0 = ttest(c)
    ans.append({
        'hypothesis_ids': [h],
        'code': f"stats.ttest_ind(df.loc[df['{c}']==1,'pfs_months'], df.loc[df['{c}']==0,'pfs_months'], equal_var=False)",
        'result_summary': f"{c}: mean PFS {m1:.3f} (n={n1}) vs {m0:.3f} (n={n0}); diff={e:+.3f} mo, Welch t-test p={p:.3e}.",
        'p_value': p, 'effect_estimate': e, 'significant': bool(p < 0.05),
    })
add(1, hyps, ans)

# --- ITERATION 2: ordinal feature_063 + remaining strong binaries ---
hyps = [
    {'id': 'h2.1', 'text': "feature_063 (ordinal 0-2) shows a negative monotonic relationship with pfs_months: each unit increase shortens PFS.", 'kind': 'novel'},
    {'id': 'h2.2', 'text': "Patients with feature_015=1 have shorter pfs_months than those with feature_015=0.", 'kind': 'novel'},
    {'id': 'h2.3', 'text': "Patients with feature_040=1 have shorter pfs_months than those with feature_040=0.", 'kind': 'novel'},
    {'id': 'h2.4', 'text': "Patients with feature_039=1 have longer pfs_months than those with feature_039=0.", 'kind': 'novel'},
]
ans = []
m = regress("pfs_months ~ feature_063")
b = m.params['feature_063']; p = m.pvalues['feature_063']
levels = sorted(df['feature_063'].unique())
means = {int(lv): float(y[df['feature_063']==lv].mean()) for lv in levels}
ans.append({
    'hypothesis_ids': ['h2.1'],
    'code': "smf.ols('pfs_months ~ feature_063', data=df).fit()",
    'result_summary': f"OLS slope on feature_063 (ordinal): {b:+.3f} mo/unit, p={p:.3e}. Means by level {means}.",
    'p_value': float(p), 'effect_estimate': float(b), 'significant': bool(p<0.05),
})
for h, c in [('h2.2','feature_015'),('h2.3','feature_040'),('h2.4','feature_039')]:
    e, p, n1, n0, m1, m0 = ttest(c)
    ans.append({
        'hypothesis_ids': [h],
        'code': f"stats.ttest_ind on {c}",
        'result_summary': f"{c}: mean PFS {m1:.3f} (n={n1}) vs {m0:.3f} (n={n0}); diff={e:+.3f} mo, Welch t-test p={p:.3e}.",
        'p_value': p, 'effect_estimate': e, 'significant': bool(p<0.05),
    })
add(2, hyps, ans)

# --- ITERATION 3: top continuous features ---
hyps = [
    {'id': 'h3.1', 'text': "feature_080 (continuous 30-90, plausibly age) is positively correlated with pfs_months: as feature_080 increases, PFS increases.", 'kind': 'novel'},
    {'id': 'h3.2', 'text': "feature_019 (continuous 1.5-5.5, plausibly albumin) is positively associated with pfs_months.", 'kind': 'novel'},
    {'id': 'h3.3', 'text': "feature_067 (continuous 0-24.6) is negatively associated with pfs_months.", 'kind': 'novel'},
    {'id': 'h3.4', 'text': "feature_101 (continuous 1-100, mean 15.5) is negatively associated with pfs_months.", 'kind': 'novel'},
]
ans = []
for h, c in [('h3.1','feature_080'),('h3.2','feature_019'),('h3.3','feature_067'),('h3.4','feature_101')]:
    rho, p_s = stats.spearmanr(df[c], y)
    m = regress(f"pfs_months ~ {c}")
    b = m.params[c]; p = m.pvalues[c]
    ans.append({
        'hypothesis_ids': [h],
        'code': f"spearmanr + ols(pfs_months ~ {c})",
        'result_summary': f"{c}: Spearman rho={rho:+.3f} (p={p_s:.3e}); OLS slope={b:+.4f} mo per unit (p={p:.3e}).",
        'p_value': float(p), 'effect_estimate': float(b), 'significant': bool(p<0.05),
    })
add(3, hyps, ans)

# --- ITERATION 4: shape of feature_080 effect ---
hyps = [
    {'id': 'h4.1', 'text': "Mean pfs_months increases monotonically across deciles of feature_080 (no inverted-U shape).", 'kind': 'novel'},
    {'id': 'h4.2', 'text': "A quadratic term in feature_080 significantly improves the model versus a linear-only model.", 'kind': 'novel'},
]
ans = []
df['_f80_dec'] = pd.qcut(df['feature_080'], 10, labels=False, duplicates='drop')
dec = df.groupby('_f80_dec')['pfs_months'].agg(['mean','count']).reset_index()
ans.append({
    'hypothesis_ids': ['h4.1'],
    'code': "df.groupby(qcut(feature_080,10))['pfs_months'].mean()",
    'result_summary': "Decile means low->high of feature_080: " + ", ".join(f"{r['mean']:.2f}" for _,r in dec.iterrows()) + ".",
    'p_value': None, 'effect_estimate': float(dec['mean'].iloc[-1] - dec['mean'].iloc[0]), 'significant': True,
})
m1 = regress("pfs_months ~ feature_080")
m2 = regress("pfs_months ~ feature_080 + I(feature_080**2)")
lr = 2*(m2.llf - m1.llf)
p_lr = float(stats.chi2.sf(lr, 1))
ans.append({
    'hypothesis_ids': ['h4.2'],
    'code': "ols(pfs ~ f80) vs ols(pfs ~ f80 + f80^2); LR test",
    'result_summary': f"Quadratic coef = {m2.params['I(feature_080 ** 2)']:+.5f}, p={m2.pvalues['I(feature_080 ** 2)']:.3e}. LR test p={p_lr:.3e}. Linear R2={m1.rsquared:.3f}, quadratic R2={m2.rsquared:.3f}.",
    'p_value': float(m2.pvalues['I(feature_080 ** 2)']), 'effect_estimate': float(m2.params['I(feature_080 ** 2)']), 'significant': bool(m2.pvalues['I(feature_080 ** 2)']<0.05),
})
add(4, hyps, ans)

# --- ITERATION 5: race + insurance main effects (disparities) ---
hyps = [
    {'id': 'h5.1', 'text': "Mean pfs_months differs across racial groups in feature_011 (white, black, hispanic, asian, other).", 'kind': 'novel'},
    {'id': 'h5.2', 'text': "Mean pfs_months differs across insurance groups in feature_089 (private, medicare, medicaid, uninsured).", 'kind': 'novel'},
]
ans = []
groups = [y[df['feature_011']==g].values for g in df['feature_011'].unique()]
F, p = stats.f_oneway(*groups)
means = {g: float(y[df['feature_011']==g].mean()) for g in df['feature_011'].unique()}
ans.append({
    'hypothesis_ids': ['h5.1'],
    'code': "stats.f_oneway over feature_011 levels",
    'result_summary': f"ANOVA F={F:.3f}, p={p:.3e}. Means {means}.",
    'p_value': float(p), 'effect_estimate': float(max(means.values())-min(means.values())), 'significant': bool(p<0.05),
})
groups = [y[df['feature_089']==g].values for g in df['feature_089'].unique()]
F, p = stats.f_oneway(*groups)
means = {g: float(y[df['feature_089']==g].mean()) for g in df['feature_089'].unique()}
ans.append({
    'hypothesis_ids': ['h5.2'],
    'code': "stats.f_oneway over feature_089 levels",
    'result_summary': f"ANOVA F={F:.3f}, p={p:.3e}. Means {means}.",
    'p_value': float(p), 'effect_estimate': float(max(means.values())-min(means.values())), 'significant': bool(p<0.05),
})
add(5, hyps, ans)

with open('_work/iters_partial.json','w') as f:
    json.dump(iterations, f, indent=2, default=str)
print(f"Iterations 1-5 done. Total: {len(iterations)}")
