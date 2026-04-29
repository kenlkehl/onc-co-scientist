"""Iterations 6-15."""
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

def regress(formula):
    return smf.ols(formula, data=df).fit()

# --- ITERATION 6: multivariable model with the leading predictors ---
hyps = [
    {'id': 'h6.1', 'text': "In a multivariable OLS model containing feature_080, feature_063, feature_042, feature_056, feature_048, feature_111, feature_015, feature_040, feature_039, feature_019, feature_067, and feature_101, all twelve features remain individually significant (p<0.05) with the same direction of effect as in their univariate tests.", 'kind': 'novel'},
]
ans = []
formula = ("pfs_months ~ feature_080 + feature_063 + feature_042 + feature_056 + feature_048 + "
           "feature_111 + feature_015 + feature_040 + feature_039 + feature_019 + feature_067 + feature_101")
m = regress(formula)
coefs = {k: (float(m.params[k]), float(m.pvalues[k])) for k in m.params.index if k != 'Intercept'}
sig_count = sum(1 for k,(b,p) in coefs.items() if p < 0.05)
ans.append({
    'hypothesis_ids': ['h6.1'],
    'code': f"smf.ols('{formula}', data=df).fit()",
    'result_summary': f"Multivariable OLS R2={m.rsquared:.3f}, n=50000. Coefficients (beta, p): " + "; ".join(f"{k}={b:+.4f}, p={p:.2e}" for k,(b,p) in coefs.items()) + f". {sig_count}/12 features p<0.05.",
    'p_value': None, 'effect_estimate': float(m.rsquared), 'significant': bool(sig_count==12),
})
add(6, hyps, ans)

# --- ITERATION 7: interactions among the 4 strongest binary predictors ---
hyps = [
    {'id': 'h7.1', 'text': "There is a significant interaction between feature_042 and feature_056 on pfs_months: the protective effect of feature_042 differs by feature_056 status.", 'kind': 'novel'},
    {'id': 'h7.2', 'text': "There is a significant interaction between feature_042 and feature_048 on pfs_months.", 'kind': 'novel'},
    {'id': 'h7.3', 'text': "There is a significant interaction between feature_056 and feature_111 on pfs_months.", 'kind': 'novel'},
]
ans = []
for hid, c1, c2 in [('h7.1','feature_042','feature_056'),('h7.2','feature_042','feature_048'),('h7.3','feature_056','feature_111')]:
    m = regress(f"pfs_months ~ {c1} * {c2}")
    inter = f"{c1}:{c2}"
    b = float(m.params[inter]); p = float(m.pvalues[inter])
    # cell means for context
    cells = {(int(a),int(c)): float(y[(df[c1]==a)&(df[c2]==c)].mean()) for a in [0,1] for c in [0,1]}
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"smf.ols('pfs_months ~ {c1} * {c2}').fit()",
        'result_summary': f"{c1} x {c2} interaction beta = {b:+.3f}, p={p:.3e}. Cell means {cells}.",
        'p_value': p, 'effect_estimate': b, 'significant': bool(p<0.05),
    })
add(7, hyps, ans)

# --- ITERATION 8: feature_080 (continuous) interactions with the strongest binaries ---
hyps = [
    {'id': 'h8.1', 'text': "The slope of feature_080 on pfs_months differs by feature_056 status (significant feature_080 x feature_056 interaction).", 'kind': 'novel'},
    {'id': 'h8.2', 'text': "The slope of feature_080 on pfs_months differs by feature_042 status (significant feature_080 x feature_042 interaction).", 'kind': 'novel'},
    {'id': 'h8.3', 'text': "The slope of feature_080 on pfs_months differs by feature_063 level (significant feature_080 x feature_063 interaction).", 'kind': 'novel'},
]
ans = []
for hid, c1, c2 in [('h8.1','feature_080','feature_056'),('h8.2','feature_080','feature_042'),('h8.3','feature_080','feature_063')]:
    m = regress(f"pfs_months ~ {c1} * {c2}")
    inter = f"{c1}:{c2}"
    b = float(m.params[inter]); p = float(m.pvalues[inter])
    # main slope and slope-in-strata
    if c2 == 'feature_063':
        slopes = {}
        for lv in [0,1,2]:
            sub = df[df[c2]==lv]
            mm = smf.ols(f"pfs_months ~ {c1}", data=sub).fit()
            slopes[int(lv)] = float(mm.params[c1])
    else:
        slopes = {}
        for v in [0,1]:
            sub = df[df[c2]==v]
            mm = smf.ols(f"pfs_months ~ {c1}", data=sub).fit()
            slopes[int(v)] = float(mm.params[c1])
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"smf.ols('pfs_months ~ {c1} * {c2}').fit()",
        'result_summary': f"Interaction beta = {b:+.4f}, p={p:.3e}. Stratified slopes of {c1}: {slopes}.",
        'p_value': p, 'effect_estimate': b, 'significant': bool(p<0.05),
    })
add(8, hyps, ans)

# --- ITERATION 9: race-stratified treatment effects (feature_042 protective effect by race) ---
hyps = [
    {'id': 'h9.1', 'text': "The protective effect of feature_042 on pfs_months differs by race (feature_011): testing feature_042 x feature_011 interaction.", 'kind': 'novel'},
    {'id': 'h9.2', 'text': "The harmful effect of feature_056 on pfs_months differs by race (feature_011 interaction).", 'kind': 'novel'},
]
ans = []
for hid, c1 in [('h9.1','feature_042'),('h9.2','feature_056')]:
    m = regress(f"pfs_months ~ {c1} * C(feature_011)")
    inter_terms = [k for k in m.params.index if k.startswith(f'{c1}:')]
    f_test = m.f_test(' = '.join([f"{t}=0" for t in inter_terms]) if False else inter_terms)
    p_inter = float(f_test.pvalue)
    # effect of c1 within each race
    eff = {}
    for r in df['feature_011'].unique():
        sub = df[df['feature_011']==r]
        a = sub.loc[sub[c1]==1, 'pfs_months'].values
        b = sub.loc[sub[c1]==0, 'pfs_months'].values
        if len(a)>=20 and len(b)>=20:
            t,p = stats.ttest_ind(a,b,equal_var=False)
            eff[r] = (float(a.mean()-b.mean()), float(p), int(len(a)), int(len(b)))
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"smf.ols('pfs_months ~ {c1} * C(feature_011)').fit(); F-test on interaction terms",
        'result_summary': f"Joint F-test on {c1} x feature_011 interaction terms: p={p_inter:.3e}. Race-stratified effect of {c1} (mean1-mean0, t-test p, n1, n0): {eff}.",
        'p_value': p_inter, 'effect_estimate': float(np.std([v[0] for v in eff.values()])) if eff else 0.0, 'significant': bool(p_inter<0.05),
    })
add(9, hyps, ans)

# --- ITERATION 10: insurance-stratified (treatment effect) ---
hyps = [
    {'id': 'h10.1', 'text': "The protective effect of feature_042 on pfs_months differs by insurance type (feature_089 interaction).", 'kind': 'novel'},
    {'id': 'h10.2', 'text': "The harmful effect of feature_056 on pfs_months differs by insurance type (feature_089 interaction).", 'kind': 'novel'},
]
ans = []
for hid, c1 in [('h10.1','feature_042'),('h10.2','feature_056')]:
    m = regress(f"pfs_months ~ {c1} * C(feature_089)")
    inter_terms = [k for k in m.params.index if k.startswith(f'{c1}:')]
    f_test = m.f_test(inter_terms)
    p_inter = float(f_test.pvalue)
    eff = {}
    for r in df['feature_089'].unique():
        sub = df[df['feature_089']==r]
        a = sub.loc[sub[c1]==1, 'pfs_months'].values
        b = sub.loc[sub[c1]==0, 'pfs_months'].values
        if len(a)>=20 and len(b)>=20:
            t,p = stats.ttest_ind(a,b,equal_var=False)
            eff[r] = (float(a.mean()-b.mean()), float(p), int(len(a)), int(len(b)))
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"smf.ols('pfs_months ~ {c1} * C(feature_089)').fit()",
        'result_summary': f"Joint F-test on {c1} x feature_089 interaction terms: p={p_inter:.3e}. Insurance-stratified effect of {c1}: {eff}.",
        'p_value': p_inter, 'effect_estimate': float(np.std([v[0] for v in eff.values()])) if eff else 0.0, 'significant': bool(p_inter<0.05),
    })
add(10, hyps, ans)

# --- ITERATION 11: lab values (sodium, calcium, hemoglobin, BMI etc.) ---
hyps = [
    {'id': 'h11.1', 'text': "feature_035 (continuous 6-18, plausibly hemoglobin g/dL) is positively associated with pfs_months.", 'kind': 'novel'},
    {'id': 'h11.2', 'text': "feature_007 (continuous 128-152, plausibly sodium mEq/L) is associated with pfs_months.", 'kind': 'novel'},
    {'id': 'h11.3', 'text': "feature_006 (continuous 7.3-11.8, plausibly calcium mg/dL) is associated with pfs_months.", 'kind': 'novel'},
    {'id': 'h11.4', 'text': "feature_079 (continuous 14-47.7, plausibly BMI) is associated with pfs_months.", 'kind': 'novel'},
    {'id': 'h11.5', 'text': "feature_096 (continuous 0.3-2.1, plausibly creatinine) is associated with pfs_months.", 'kind': 'novel'},
]
ans = []
for hid, c in [('h11.1','feature_035'),('h11.2','feature_007'),('h11.3','feature_006'),('h11.4','feature_079'),('h11.5','feature_096')]:
    rho, p_s = stats.spearmanr(df[c], y)
    m = regress(f"pfs_months ~ {c}")
    b = float(m.params[c]); p = float(m.pvalues[c])
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"spearmanr + ols(pfs_months ~ {c})",
        'result_summary': f"{c}: Spearman rho={rho:+.4f} (p={p_s:.3e}); OLS slope={b:+.4f} mo/unit (p={p:.3e}).",
        'p_value': p, 'effect_estimate': b, 'significant': bool(p<0.05),
    })
add(11, hyps, ans)

# --- ITERATION 12: tumor-marker-like features ---
hyps = [
    {'id': 'h12.1', 'text': "feature_059 (continuous 0.3-897, mean 26; possibly tumor marker such as CA15-3) is negatively associated with pfs_months.", 'kind': 'novel'},
    {'id': 'h12.2', 'text': "feature_027 (continuous 0.02-516, mean 5.8) is negatively associated with pfs_months.", 'kind': 'novel'},
    {'id': 'h12.3', 'text': "feature_046 (continuous 0.02-116, mean 2.2) is negatively associated with pfs_months.", 'kind': 'novel'},
    {'id': 'h12.4', 'text': "feature_064 (continuous 0.7-150, mean 27) is negatively associated with pfs_months.", 'kind': 'novel'},
    {'id': 'h12.5', 'text': "feature_053 (continuous 48-830, mean 224; possibly LDH) is negatively associated with pfs_months.", 'kind': 'novel'},
]
ans = []
for hid, c in [('h12.1','feature_059'),('h12.2','feature_027'),('h12.3','feature_046'),('h12.4','feature_064'),('h12.5','feature_053')]:
    rho, p_s = stats.spearmanr(df[c], y)
    m = regress(f"pfs_months ~ {c}")
    b = float(m.params[c]); p = float(m.pvalues[c])
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"spearmanr + ols(pfs_months ~ {c})",
        'result_summary': f"{c}: Spearman rho={rho:+.4f} (p={p_s:.3e}); OLS slope={b:+.6f} mo/unit (p={p:.3e}).",
        'p_value': p, 'effect_estimate': b, 'significant': bool(p<0.05),
    })
add(12, hyps, ans)

# --- ITERATION 13: log-transform of skewed markers ---
hyps = [
    {'id': 'h13.1', 'text': "log(feature_059+1) is more strongly negatively associated with pfs_months than the raw feature_059 (right-skewed marker).", 'kind': 'novel'},
    {'id': 'h13.2', 'text': "log(feature_027+1) is more strongly negatively associated with pfs_months than the raw feature_027.", 'kind': 'novel'},
]
ans = []
for hid, c in [('h13.1','feature_059'),('h13.2','feature_027')]:
    df['_lg'] = np.log1p(df[c])
    rho_raw, p_raw = stats.spearmanr(df[c], y)
    m_raw = regress(f"pfs_months ~ {c}")
    m_log = smf.ols("pfs_months ~ _lg", data=df).fit()
    b_log = float(m_log.params['_lg']); p_log = float(m_log.pvalues['_lg'])
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"ols(pfs_months ~ log1p({c})) vs ols(pfs_months ~ {c})",
        'result_summary': f"Raw {c}: OLS slope={float(m_raw.params[c]):+.5f}, p={float(m_raw.pvalues[c]):.3e}, R2={m_raw.rsquared:.4f}. log1p({c}): slope={b_log:+.4f}, p={p_log:.3e}, R2={m_log.rsquared:.4f}. Spearman rho (raw)={rho_raw:+.4f}.",
        'p_value': p_log, 'effect_estimate': b_log, 'significant': bool(p_log<0.05),
    })
    df.drop(columns=['_lg'], inplace=True)
add(13, hyps, ans)

# --- ITERATION 14: ordinal-coded features (h2.1 found 063 strong; what about 021, 024, 127, 001, 073 ?) ---
hyps = [
    {'id': 'h14.1', 'text': "Ordinal feature_021 (0-4) has a non-zero linear association with pfs_months.", 'kind': 'novel'},
    {'id': 'h14.2', 'text': "Ordinal feature_024 (0-4) has a non-zero linear association with pfs_months.", 'kind': 'novel'},
    {'id': 'h14.3', 'text': "Ordinal feature_127 (0-4) has a non-zero linear association with pfs_months.", 'kind': 'novel'},
    {'id': 'h14.4', 'text': "Ordinal feature_001 (0-4) has a non-zero linear association with pfs_months.", 'kind': 'novel'},
    {'id': 'h14.5', 'text': "Ordinal feature_073 (0-4) has a non-zero linear association with pfs_months.", 'kind': 'novel'},
    {'id': 'h14.6', 'text': "Ordinal feature_016 (0-10) has a non-zero linear association with pfs_months.", 'kind': 'novel'},
]
ans = []
for hid, c in [('h14.1','feature_021'),('h14.2','feature_024'),('h14.3','feature_127'),('h14.4','feature_001'),('h14.5','feature_073'),('h14.6','feature_016')]:
    m = regress(f"pfs_months ~ {c}")
    b = float(m.params[c]); p = float(m.pvalues[c])
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"ols(pfs_months ~ {c})",
        'result_summary': f"{c}: OLS slope={b:+.4f} mo/unit, p={p:.3e}.",
        'p_value': p, 'effect_estimate': b, 'significant': bool(p<0.05),
    })
add(14, hyps, ans)

# --- ITERATION 15: smaller-prevalence binary features that haven't been tested individually ---
hyps = [
    {'id': 'h15.1', 'text': "Patients with feature_017=1 (rare, ~2.6%) have shorter pfs_months than feature_017=0.", 'kind': 'novel'},
    {'id': 'h15.2', 'text': "Patients with feature_094=1 (rare, ~2.4%) have shorter pfs_months than feature_094=0.", 'kind': 'novel'},
    {'id': 'h15.3', 'text': "Patients with feature_058=1 (rare, ~1.0%) have shorter pfs_months than feature_058=0.", 'kind': 'novel'},
    {'id': 'h15.4', 'text': "Patients with feature_118=1 (very rare, ~0.5%) have shorter pfs_months than feature_118=0.", 'kind': 'novel'},
    {'id': 'h15.5', 'text': "Patients with feature_126=1 (very rare, ~0.6%) have shorter pfs_months than feature_126=0.", 'kind': 'novel'},
]
ans = []
def ttest(col):
    a = y[df[col] == 1].values
    b = y[df[col] == 0].values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(p), int(len(a)), int(len(b)), float(a.mean()), float(b.mean())
for hid, c in [('h15.1','feature_017'),('h15.2','feature_094'),('h15.3','feature_058'),('h15.4','feature_118'),('h15.5','feature_126')]:
    e, p, n1, n0, m1, m0 = ttest(c)
    ans.append({
        'hypothesis_ids': [hid],
        'code': f"stats.ttest_ind on {c}",
        'result_summary': f"{c}: mean PFS {m1:.3f} (n={n1}) vs {m0:.3f} (n={n0}); diff={e:+.3f} mo, Welch p={p:.3e}.",
        'p_value': p, 'effect_estimate': e, 'significant': bool(p<0.05),
    })
add(15, hyps, ans)

with open('_work/iters_partial.json','w') as f:
    json.dump(iterations, f, indent=2, default=str)
print(f"Iterations 6-15 done. Total: {len(iterations)}")
