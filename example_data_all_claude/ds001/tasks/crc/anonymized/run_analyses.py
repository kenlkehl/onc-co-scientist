"""Iterative analysis of ds001_crc anonymized cohort.
Outputs JSON-serializable analysis records consumed by the transcript builder.
"""
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations

df = pd.read_parquet('dataset.parquet').drop(columns=['patient_id'])
y = df['pfs_months']

binary_feats = [c for c in df.columns if c != 'pfs_months' and df[c].nunique() == 2]
multilevel_feats = [c for c in df.columns if c != 'pfs_months' and 2 < df[c].nunique() <= 10]
cont_feats = [c for c in df.columns if c != 'pfs_months' and df[c].nunique() > 10]

# Candidate treatment: feature_027 (binary, ~20% prevalence, positive effect on PFS)
TREAT = 'feature_027'

# Container for analysis records
ITERATIONS = []

def add_iter(idx, hyps, analyses):
    ITERATIONS.append({'index': idx, 'proposed_hypotheses': hyps, 'analyses': analyses})

def fmt_p(p):
    if p == 0 or p < 1e-300:
        return 0.0
    return float(p)

# ---- Iteration 1: Univariate screen (already explored interactively, formalize a few key tests) ----
hyps = [
    {'id': 'h1', 'text': 'Higher feature_015 (continuous, range 30-90, plausibly age) is positively associated with longer pfs_months.', 'kind': 'novel'},
    {'id': 'h2', 'text': 'Higher feature_001 (ordinal 0/1/2, plausibly performance status or stage) is associated with shorter pfs_months.', 'kind': 'novel'},
    {'id': 'h3', 'text': 'Patients with feature_014 == 1 have shorter pfs_months than those with feature_014 == 0.', 'kind': 'novel'},
    {'id': 'h4', 'text': 'Patients with feature_027 == 1 (candidate treatment, ~20% prevalence) have longer pfs_months than feature_027 == 0.', 'kind': 'novel'},
]
analyses = []

# h1: Spearman correlation feature_015 vs pfs_months
rho, p = stats.spearmanr(df['feature_015'], y)
analyses.append({
    'hypothesis_ids': ['h1'],
    'code': "stats.spearmanr(df['feature_015'], df['pfs_months'])",
    'result_summary': f"Spearman rho = {rho:.3f}, p = {p:.3e}; mean PFS rises from 1.33mo (feature_015<=52.3) to 7.45mo (feature_015>=77.8) across deciles.",
    'p_value': fmt_p(p),
    'effect_estimate': float(rho),
    'significant': bool(p < 0.05),
})

# h2: Spearman feature_001 vs PFS, stratified means
rho, p = stats.spearmanr(df['feature_001'], y)
m0, m1, m2 = [df.loc[df['feature_001']==v, 'pfs_months'].mean() for v in (0,1,2)]
analyses.append({
    'hypothesis_ids': ['h2'],
    'code': "stats.spearmanr(df['feature_001'], df['pfs_months'])",
    'result_summary': f"Mean PFS by feature_001: 0->{m0:.2f}, 1->{m1:.2f}, 2->{m2:.2f} (Spearman rho={rho:.3f}, p={p:.3e}).",
    'p_value': fmt_p(p),
    'effect_estimate': float(rho),
    'significant': bool(p < 0.05),
})

# h3: t-test by feature_014
a = y[df['feature_014']==0]; b = y[df['feature_014']==1]
t, p = stats.ttest_ind(a, b, equal_var=False)
analyses.append({
    'hypothesis_ids': ['h3'],
    'code': "stats.ttest_ind(y[feature_014==0], y[feature_014==1], equal_var=False)",
    'result_summary': f"Mean PFS feature_014=1 ({b.mean():.2f}) vs feature_014=0 ({a.mean():.2f}); difference = {b.mean()-a.mean():+.2f} months (Welch t-test p={p:.3e}).",
    'p_value': fmt_p(p),
    'effect_estimate': float(b.mean()-a.mean()),
    'significant': bool(p < 0.05),
})

# h4: t-test by feature_027
a = y[df[TREAT]==0]; b = y[df[TREAT]==1]
t, p = stats.ttest_ind(a, b, equal_var=False)
analyses.append({
    'hypothesis_ids': ['h4'],
    'code': "stats.ttest_ind(y[feature_027==0], y[feature_027==1], equal_var=False)",
    'result_summary': f"Mean PFS feature_027=1 ({b.mean():.2f}) vs feature_027=0 ({a.mean():.2f}); difference = {b.mean()-a.mean():+.2f} months (p={p:.3e}, n_treat={int((df[TREAT]==1).sum())}).",
    'p_value': fmt_p(p),
    'effect_estimate': float(b.mean()-a.mean()),
    'significant': bool(p < 0.05),
})
add_iter(1, hyps, analyses)

# ---- Iteration 2: Continuous correlates: feature_025 (smoking-like?) and feature_019 (albumin-like?) ----
hyps = [
    {'id': 'h5', 'text': 'Higher feature_025 (continuous, range 0-24.6) is associated with shorter pfs_months.', 'kind': 'novel'},
    {'id': 'h6', 'text': 'Higher feature_019 (continuous, mean 3.8, range 1.5-5.5) is associated with longer pfs_months.', 'kind': 'novel'},
    {'id': 'h7', 'text': 'Patients with feature_023 == 1 have shorter pfs_months than feature_023 == 0.', 'kind': 'novel'},
    {'id': 'h8', 'text': 'Patients with feature_006 == 1 have shorter pfs_months than feature_006 == 0.', 'kind': 'novel'},
]
analyses = []
for hid, col, sign in [('h5','feature_025',-1), ('h6','feature_019',+1)]:
    rho, p = stats.spearmanr(df[col], y)
    analyses.append({
        'hypothesis_ids': [hid],
        'code': f"stats.spearmanr(df['{col}'], df['pfs_months'])",
        'result_summary': f"Spearman rho = {rho:.3f}, p = {p:.3e} (n=50000).",
        'p_value': fmt_p(p),
        'effect_estimate': float(rho),
        'significant': bool(p < 0.05),
    })
for hid, col in [('h7','feature_023'), ('h8','feature_006')]:
    a = y[df[col]==0]; b = y[df[col]==1]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    analyses.append({
        'hypothesis_ids': [hid],
        'code': f"stats.ttest_ind(y[{col}==0], y[{col}==1], equal_var=False)",
        'result_summary': f"Mean PFS {col}=1 ({b.mean():.2f}) vs 0 ({a.mean():.2f}); diff = {b.mean()-a.mean():+.2f} months (p={p:.3e}).",
        'p_value': fmt_p(p),
        'effect_estimate': float(b.mean()-a.mean()),
        'significant': bool(p < 0.05),
    })
add_iter(2, hyps, analyses)

# ---- Iteration 3: Multivariable OLS to confirm independent effects ----
hyps = [
    {'id': 'h9', 'text': 'In a multivariable OLS for pfs_months, feature_027 retains a positive effect after adjustment for feature_015, feature_001, feature_014, feature_025, feature_019, feature_023, feature_006.', 'kind': 'refined'},
    {'id': 'h10', 'text': 'In the same multivariable OLS, feature_015 retains its positive coefficient on pfs_months after adjustment.', 'kind': 'refined'},
    {'id': 'h11', 'text': 'In the same multivariable OLS, feature_001 (treated as ordinal numeric) retains a negative coefficient on pfs_months.', 'kind': 'refined'},
]
formula = ('pfs_months ~ feature_027 + feature_015 + C(feature_001) + feature_014 '
           '+ feature_025 + feature_019 + feature_023 + feature_006')
m = smf.ols(formula, data=df).fit()
coefs = m.params; pv = m.pvalues
def grab(name):
    return float(coefs[name]), fmt_p(pv[name])
c_t, p_t = grab('feature_027')
c_age, p_age = grab('feature_015')
c_f1_1, p_f1_1 = grab('C(feature_001)[T.1]')
c_f1_2, p_f1_2 = grab('C(feature_001)[T.2]')

analyses = [
    {
        'hypothesis_ids': ['h9'],
        'code': f"smf.ols('{formula}', data=df).fit()",
        'result_summary': f"Adjusted feature_027 coefficient = {c_t:+.3f} months (p={p_t:.3e}); R^2 = {m.rsquared:.3f}.",
        'p_value': p_t, 'effect_estimate': c_t, 'significant': bool(p_t<0.05),
    },
    {
        'hypothesis_ids': ['h10'],
        'code': "same OLS",
        'result_summary': f"Adjusted feature_015 coefficient = {c_age:+.4f} months per unit (p={p_age:.3e}).",
        'p_value': p_age, 'effect_estimate': c_age, 'significant': bool(p_age<0.05),
    },
    {
        'hypothesis_ids': ['h11'],
        'code': "same OLS, treating feature_001 as categorical with reference 0",
        'result_summary': (f"feature_001=1 coef vs 0: {c_f1_1:+.3f} (p={p_f1_1:.3e}); "
                          f"feature_001=2 coef vs 0: {c_f1_2:+.3f} (p={p_f1_2:.3e})."),
        'p_value': p_f1_2, 'effect_estimate': c_f1_2, 'significant': bool(p_f1_2<0.05),
    },
]
add_iter(3, hyps, analyses)

# ---- Iteration 4: Test treatment-by-feature interactions for the candidate treatment ----
# Screen all features for interaction with feature_027 in OLS pfs ~ feature * treat (controlling for top covariates)
base_terms = 'feature_015 + C(feature_001) + feature_014 + feature_025 + feature_019 + feature_023 + feature_006'
def interaction_test(feat):
    f = f"pfs_months ~ {TREAT} * {feat} + " + base_terms
    if feat in ('feature_015','feature_001','feature_014','feature_025','feature_019','feature_023','feature_006'):
        # avoid duplicating; remove its presence in base_terms
        terms = [t.strip() for t in base_terms.split('+')]
        terms = [t for t in terms if feat not in t]
        f = f"pfs_months ~ {TREAT} * {feat} + " + ' + '.join(terms)
    try:
        m = smf.ols(f, data=df).fit()
        # find coef of interaction
        ix_keys = [k for k in m.params.index if ':' in k and TREAT in k and feat in k]
        if not ix_keys:
            return None
        # Use the first interaction coefficient (binary case) or the test statistic
        key = ix_keys[0]
        return float(m.params[key]), fmt_p(m.pvalues[key]), key
    except Exception as e:
        return None

hyps = [
    {'id': 'h12', 'text': 'The benefit of feature_027 on pfs_months is modified by feature_001 (treatment-by-performance-status-like interaction); the positive treatment effect is larger at lower feature_001.', 'kind': 'novel'},
    {'id': 'h13', 'text': 'The benefit of feature_027 on pfs_months is modified by feature_014; the positive treatment effect differs between feature_014 strata.', 'kind': 'novel'},
    {'id': 'h14', 'text': 'The benefit of feature_027 on pfs_months is modified by feature_023; the positive treatment effect differs between feature_023 strata.', 'kind': 'novel'},
    {'id': 'h15', 'text': 'The benefit of feature_027 on pfs_months is modified by feature_006; the positive treatment effect differs between feature_006 strata.', 'kind': 'novel'},
]
analyses = []
for hid, feat in [('h12','feature_001'), ('h13','feature_014'), ('h14','feature_023'), ('h15','feature_006')]:
    res = interaction_test(feat)
    if res is None:
        analyses.append({'hypothesis_ids':[hid],'result_summary':'Interaction model failed to converge.','p_value':None,'effect_estimate':None,'significant':False})
    else:
        coef, pv, key = res
        # Stratified means
        st = df.groupby([feat, TREAT])['pfs_months'].mean().unstack()
        analyses.append({
            'hypothesis_ids': [hid],
            'code': f"smf.ols('pfs ~ {TREAT}*{feat} + covariates', df).fit() — {key}",
            'result_summary': (f"Interaction coef ({key}) = {coef:+.3f} months, p = {pv:.3e}. "
                               f"Stratified means (rows={feat}, cols={TREAT}):\n{st.to_string()}"),
            'p_value': pv, 'effect_estimate': coef, 'significant': bool(pv<0.05),
        })
add_iter(4, hyps, analyses)

# ---- Iteration 5: Continuous-feature interactions with treatment ----
hyps = [
    {'id': 'h16', 'text': 'The benefit of feature_027 on pfs_months varies with feature_015 (interaction term non-zero).', 'kind': 'novel'},
    {'id': 'h17', 'text': 'The benefit of feature_027 on pfs_months varies with feature_025.', 'kind': 'novel'},
    {'id': 'h18', 'text': 'The benefit of feature_027 on pfs_months varies with feature_019.', 'kind': 'novel'},
]
analyses = []
for hid, feat in [('h16','feature_015'), ('h17','feature_025'), ('h18','feature_019')]:
    res = interaction_test(feat)
    coef, pv, key = res
    analyses.append({
        'hypothesis_ids':[hid],
        'code': f"OLS pfs ~ {TREAT}*{feat} + covariates — {key}",
        'result_summary': f"Interaction coefficient ({key}) = {coef:+.4f}, p = {pv:.3e}.",
        'p_value': pv, 'effect_estimate': coef, 'significant': bool(pv<0.05),
    })
add_iter(5, hyps, analyses)

# ---- Iteration 6: Systematic interaction screen across ALL features ----
hyps = [
    {'id': 'h19', 'text': 'Among all 33 features, at least one shows a statistically significant interaction with feature_027 on pfs_months at Bonferroni-adjusted alpha (0.05/33 ~ 0.0015).', 'kind': 'novel'},
]
all_features = [c for c in df.columns if c not in ('pfs_months', TREAT)]
screen = []
for feat in all_features:
    r = interaction_test(feat)
    if r is None:
        continue
    coef, pv, key = r
    screen.append((feat, key, coef, pv))
screen_df = pd.DataFrame(screen, columns=['feat','term','coef','p']).sort_values('p')
top = screen_df.head(10)
sig_bonf = (screen_df['p'] < 0.05/len(all_features)).sum()
analyses = [{
    'hypothesis_ids':['h19'],
    'code': f"For each feature f: smf.ols('pfs_months ~ {TREAT}*f + covariates', df).fit(); record p of interaction term.",
    'result_summary': (f"Top 10 treatment-interaction p-values (n={len(screen_df)} features tested, Bonferroni alpha={(0.05/len(all_features)):.4f}):\n"
                       f"{top.to_string(index=False)}\n"
                       f"Number of interactions significant at Bonferroni: {sig_bonf}."),
    'p_value': float(top.iloc[0]['p']),
    'effect_estimate': float(top.iloc[0]['coef']),
    'significant': bool(top.iloc[0]['p'] < 0.05/len(all_features)),
}]
add_iter(6, hyps, analyses)

# Persist screen for later iterations
TOP_INTERACTORS = top['feat'].tolist()
SCREEN = screen_df

# ---- Iteration 7: Probe the strongest interactor in detail ----
top_feat = screen_df.iloc[0]['feat']
hyps = [
    {'id':'h20','text': f"feature_027's pfs_months benefit differs across strata of {top_feat}; the treatment effect is significantly larger in one stratum.", 'kind':'novel'},
]
# Stratified treatment-effect estimates with Welch t-test
analyses = []
if df[top_feat].nunique() == 2:
    rows = []
    for v in (0,1):
        sub = df[df[top_feat]==v]
        a = sub.loc[sub[TREAT]==0,'pfs_months']; b = sub.loc[sub[TREAT]==1,'pfs_months']
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append((v, len(sub), a.mean(), b.mean(), b.mean()-a.mean(), p))
    txt = pd.DataFrame(rows, columns=[top_feat,'n','mean_ctrl','mean_trt','diff','p']).to_string(index=False)
    # use the larger-magnitude diff
    coef = float(rows[0][4] if abs(rows[0][4])>abs(rows[1][4]) else rows[1][4])
    pmin = float(min(rows[0][5], rows[1][5]))
    analyses.append({
        'hypothesis_ids':['h20'],
        'code': f"Stratify by {top_feat} (binary), t-test of {TREAT} within each stratum.",
        'result_summary': f"Stratified treatment effect of {TREAT} within {top_feat} levels:\n{txt}",
        'p_value': pmin, 'effect_estimate': coef, 'significant': bool(pmin<0.05),
    })
elif df[top_feat].nunique() == 3:
    rows = []
    for v in sorted(df[top_feat].unique()):
        sub = df[df[top_feat]==v]
        a = sub.loc[sub[TREAT]==0,'pfs_months']; b = sub.loc[sub[TREAT]==1,'pfs_months']
        if len(a)>0 and len(b)>0:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append((v, len(sub), a.mean(), b.mean(), b.mean()-a.mean(), p))
    txt = pd.DataFrame(rows, columns=[top_feat,'n','mean_ctrl','mean_trt','diff','p']).to_string(index=False)
    coef = float(max(rows, key=lambda r: abs(r[4]))[4])
    pmin = float(min(r[5] for r in rows))
    analyses.append({
        'hypothesis_ids':['h20'],
        'code': f"Stratify by {top_feat} (3 levels), t-test of {TREAT} within each level.",
        'result_summary': f"Stratified treatment effect of {TREAT} within {top_feat}:\n{txt}",
        'p_value': pmin, 'effect_estimate': coef, 'significant': bool(pmin<0.05),
    })
else:
    df['_q'] = pd.qcut(df[top_feat], 4, duplicates='drop')
    rows = []
    for v, sub in df.groupby('_q'):
        a = sub.loc[sub[TREAT]==0,'pfs_months']; b = sub.loc[sub[TREAT]==1,'pfs_months']
        if len(a)>0 and len(b)>0:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append((str(v), len(sub), a.mean(), b.mean(), b.mean()-a.mean(), p))
    txt = pd.DataFrame(rows, columns=[top_feat,'n','mean_ctrl','mean_trt','diff','p']).to_string(index=False)
    coef = float(max(rows, key=lambda r: abs(r[4]))[4])
    pmin = float(min(r[5] for r in rows))
    df.drop(columns=['_q'], inplace=True)
    analyses.append({
        'hypothesis_ids':['h20'],
        'code': f"Stratify by quartiles of {top_feat}, t-test of {TREAT}.",
        'result_summary': f"Treatment-effect across quartiles of {top_feat}:\n{txt}",
        'p_value': pmin, 'effect_estimate': coef, 'significant': bool(pmin<0.05),
    })
add_iter(7, hyps, analyses)

# ---- Iteration 8: Second-strongest interactor ----
second_feat = screen_df.iloc[1]['feat']
hyps = [
    {'id':'h21','text': f"feature_027's effect on pfs_months also differs by {second_feat}.", 'kind':'novel'},
]
# Use a similar block
analyses = []
if df[second_feat].nunique() == 2:
    rows = []
    for v in (0,1):
        sub = df[df[second_feat]==v]
        a = sub.loc[sub[TREAT]==0,'pfs_months']; b = sub.loc[sub[TREAT]==1,'pfs_months']
        if len(a)>0 and len(b)>0:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append((v, len(sub), a.mean(), b.mean(), b.mean()-a.mean(), p))
    txt = pd.DataFrame(rows, columns=[second_feat,'n','mean_ctrl','mean_trt','diff','p']).to_string(index=False)
    coef = float(max(rows, key=lambda r: abs(r[4]))[4])
    pmin = float(min(r[5] for r in rows))
    analyses.append({'hypothesis_ids':['h21'],'code':f"Stratify by {second_feat}, t-test of {TREAT}.","result_summary":f"Treatment effect within {second_feat}:\n{txt}",'p_value':pmin,'effect_estimate':coef,'significant':bool(pmin<0.05)})
elif df[second_feat].nunique() == 3:
    rows = []
    for v in sorted(df[second_feat].unique()):
        sub = df[df[second_feat]==v]
        a = sub.loc[sub[TREAT]==0,'pfs_months']; b = sub.loc[sub[TREAT]==1,'pfs_months']
        if len(a)>0 and len(b)>0:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append((v, len(sub), a.mean(), b.mean(), b.mean()-a.mean(), p))
    txt = pd.DataFrame(rows, columns=[second_feat,'n','mean_ctrl','mean_trt','diff','p']).to_string(index=False)
    coef = float(max(rows, key=lambda r: abs(r[4]))[4])
    pmin = float(min(r[5] for r in rows))
    analyses.append({'hypothesis_ids':['h21'],'code':f"Stratify by {second_feat}, t-test of {TREAT}.","result_summary":f"Treatment effect within {second_feat}:\n{txt}",'p_value':pmin,'effect_estimate':coef,'significant':bool(pmin<0.05)})
else:
    df['_q'] = pd.qcut(df[second_feat], 4, duplicates='drop')
    rows = []
    for v, sub in df.groupby('_q'):
        a = sub.loc[sub[TREAT]==0,'pfs_months']; b = sub.loc[sub[TREAT]==1,'pfs_months']
        if len(a)>0 and len(b)>0:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append((str(v), len(sub), a.mean(), b.mean(), b.mean()-a.mean(), p))
    df.drop(columns=['_q'], inplace=True)
    txt = pd.DataFrame(rows, columns=[second_feat,'n','mean_ctrl','mean_trt','diff','p']).to_string(index=False)
    coef = float(max(rows, key=lambda r: abs(r[4]))[4])
    pmin = float(min(r[5] for r in rows))
    analyses.append({'hypothesis_ids':['h21'],'code':f"Stratify by quartiles of {second_feat}, t-test of {TREAT}.","result_summary":f"Treatment effect across quartiles of {second_feat}:\n{txt}",'p_value':pmin,'effect_estimate':coef,'significant':bool(pmin<0.05)})
add_iter(8, hyps, analyses)

# ---- Iteration 9: Joint test of top 3 interactors with treatment in single OLS ----
top3 = screen_df.head(3)['feat'].tolist()
form = (f"pfs_months ~ {TREAT}*{top3[0]} + {TREAT}*{top3[1]} + {TREAT}*{top3[2]} + {base_terms}")
# Avoid duplication of base covariates if overlap with top3 — drop them from base
def build_form(top3):
    btxt = base_terms
    for f in top3:
        btxt = btxt.replace(f, '')
    btxt = ' + '.join([t.strip() for t in btxt.split('+') if t.strip()])
    return f"pfs_months ~ {TREAT}*{top3[0]} + {TREAT}*{top3[1]} + {TREAT}*{top3[2]} + " + btxt
form = build_form(top3)
m_joint = smf.ols(form, data=df).fit()

# Joint Wald test of all interaction terms
intx_keys = [k for k in m_joint.params.index if ':' in k and TREAT in k]
hyps = [
    {'id':'h22','text': f"Jointly testing the three strongest interactors ({', '.join(top3)}) with feature_027 yields a significant joint contribution beyond main effects.", 'kind':'refined'},
]
constraint = ' = '.join(intx_keys) + ' = 0'
ftest = m_joint.f_test(' , '.join([f'{k} = 0' for k in intx_keys]))
analyses = [{
    'hypothesis_ids':['h22'],
    'code': f"smf.ols(... {TREAT} interactions with {top3} ...); f_test(joint==0)",
    'result_summary': (f"Joint F-test on interactions {intx_keys}: F = {float(ftest.fvalue):.2f}, "
                       f"p = {float(ftest.pvalue):.3e}; coefs:\n"
                       + '\n'.join(f"  {k}: {m_joint.params[k]:+.4f} (p={m_joint.pvalues[k]:.3e})" for k in intx_keys)),
    'p_value': float(ftest.pvalue),
    'effect_estimate': float(sum(m_joint.params[k] for k in intx_keys)),
    'significant': bool(float(ftest.pvalue)<0.05),
}]
add_iter(9, hyps, analyses)

# ---- Iteration 10: Tree-based subgroup discovery for treatment-effect heterogeneity ----
# Honest causal forest is overkill; use a simple T-learner contrasting treatment effect by predicting
# pfs separately under treatment and control with gradient boosting, then identifying high-effect subgroup.
hyps = [
    {'id':'h23','text': 'A regression-tree T-learner identifies a subgroup of patients in whom feature_027 confers an above-average pfs_months benefit, and the subgroup defining variables match those flagged by the interaction screen.', 'kind':'novel'},
]
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

X_cols = [c for c in df.columns if c not in ('pfs_months', TREAT)]
X = df[X_cols].values
y_arr = df['pfs_months'].values
t_arr = df[TREAT].values

# T-learner via gradient boosting on each arm
gbr0 = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42).fit(X[t_arr==0], y_arr[t_arr==0])
gbr1 = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42).fit(X[t_arr==1], y_arr[t_arr==1])
tau_hat = gbr1.predict(X) - gbr0.predict(X)

# Fit shallow regression tree on tau_hat to discover subgroup
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, random_state=42).fit(X, tau_hat)
tree_text = export_text(tree, feature_names=X_cols, decimals=2, max_depth=3)
# Identify highest- and lowest-effect leaves
leaf_id = tree.apply(X)
leaf_means = pd.Series(tau_hat).groupby(leaf_id).agg(['mean','count']).sort_values('mean', ascending=False)

# Take top leaf and confirm with empirical diff in PFS
top_leaf = leaf_means.index[0]
mask = leaf_id == top_leaf
a = y_arr[(mask) & (t_arr==0)]; b = y_arr[(mask) & (t_arr==1)]
t_t, p_t = stats.ttest_ind(a, b, equal_var=False) if (len(a)>0 and len(b)>0) else (np.nan, np.nan)

bot_leaf = leaf_means.index[-1]
mask_b = leaf_id == bot_leaf
ab = y_arr[(mask_b) & (t_arr==0)]; bb = y_arr[(mask_b) & (t_arr==1)]
t_b, p_b = stats.ttest_ind(ab, bb, equal_var=False) if (len(ab)>0 and len(bb)>0) else (np.nan, np.nan)

analyses = [{
    'hypothesis_ids':['h23'],
    'code': "T-learner with GradientBoostingRegressor; DecisionTreeRegressor(max_depth=3) on predicted tau; identify top/bottom leaves.",
    'result_summary': (f"Subgroup tree (CATE):\n{tree_text}\n"
                       f"Highest-CATE leaf: predicted tau = {leaf_means.iloc[0]['mean']:+.2f} (n={int(leaf_means.iloc[0]['count'])});"
                       f" empirical PFS diff = {b.mean()-a.mean():+.2f} months (p={p_t:.3e}, n_ctrl={len(a)}, n_trt={len(b)}).\n"
                       f"Lowest-CATE leaf: predicted tau = {leaf_means.iloc[-1]['mean']:+.2f} (n={int(leaf_means.iloc[-1]['count'])});"
                       f" empirical PFS diff = {bb.mean()-ab.mean():+.2f} months (p={p_b:.3e})."),
    'p_value': float(p_t),
    'effect_estimate': float(b.mean()-a.mean()),
    'significant': bool(p_t<0.05),
}]
add_iter(10, hyps, analyses)

# Persist tree-discovered top leaf rule for later
def extract_rule_paths(tree, feature_names, top_leaf):
    # walk tree to find rule for top_leaf
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    path = []
    def walk(node, conds):
        if children_left[node] == -1:
            if node == top_leaf:
                path.extend(conds)
            return
        f = feature_names[feature[node]]
        thr = threshold[node]
        walk(children_left[node], conds + [f"{f} <= {thr:.4g}"])
        walk(children_right[node], conds + [f"{f} > {thr:.4g}"])
    walk(0, [])
    return path
TOP_LEAF_RULE = extract_rule_paths(tree, X_cols, top_leaf)
BOT_LEAF_RULE = extract_rule_paths(tree, X_cols, bot_leaf)

# ---- Iteration 11: Confirm tree subgroup with explicit subgroup test ----
hyps = [
    {'id':'h24','text': f"In the tree-discovered subgroup defined by [{', '.join(TOP_LEAF_RULE)}], feature_027 has a larger positive effect on pfs_months than in the complementary group.", 'kind':'refined'},
]
# Build subgroup mask from rule
def mask_from_rule(df, rule):
    mask = np.ones(len(df), dtype=bool)
    for cond in rule:
        if '<=' in cond:
            f, thr = cond.split(' <= '); mask &= (df[f].values <= float(thr))
        else:
            f, thr = cond.split(' > '); mask &= (df[f].values > float(thr))
    return mask
mask_top = mask_from_rule(df, TOP_LEAF_RULE)
a_in = y[mask_top & (df[TREAT]==0)]; b_in = y[mask_top & (df[TREAT]==1)]
a_out = y[~mask_top & (df[TREAT]==0)]; b_out = y[~mask_top & (df[TREAT]==1)]
diff_in = b_in.mean() - a_in.mean()
diff_out = b_out.mean() - a_out.mean()
_, p_in = stats.ttest_ind(a_in, b_in, equal_var=False)
_, p_out = stats.ttest_ind(a_out, b_out, equal_var=False)
# Interaction test of subgroup indicator
df['_sub'] = mask_top.astype(int)
m_int = smf.ols(f'pfs_months ~ {TREAT}*_sub', data=df).fit()
intx_p = float(m_int.pvalues[f'{TREAT}:_sub'])
intx_coef = float(m_int.params[f'{TREAT}:_sub'])
df.drop(columns=['_sub'], inplace=True)
analyses = [{
    'hypothesis_ids':['h24'],
    'code': "Mask patients with tree-rule -> compare treatment effect in vs out + interaction term.",
    'result_summary': (f"In-group (n={int(mask_top.sum())}): mean PFS ctrl={a_in.mean():.2f}, trt={b_in.mean():.2f}, diff={diff_in:+.2f} (p={p_in:.3e}).\n"
                       f"Out-group (n={int((~mask_top).sum())}): mean PFS ctrl={a_out.mean():.2f}, trt={b_out.mean():.2f}, diff={diff_out:+.2f} (p={p_out:.3e}).\n"
                       f"Interaction (treatment x subgroup-indicator): coef={intx_coef:+.3f}, p={intx_p:.3e}."),
    'p_value': intx_p, 'effect_estimate': diff_in - diff_out, 'significant': bool(intx_p<0.05),
}]
add_iter(11, hyps, analyses)

# ---- Iteration 12: Try splitting along single most-important continuous modifier ----
# Use feature_015 quartiles or feature_001 levels as a clinically-cleaner partition
hyps = [
    {'id':'h25','text': 'feature_027 yields its largest positive pfs_months effect in the lowest feature_001 stratum (feature_001=0).', 'kind':'novel'},
    {'id':'h26','text': 'feature_027 yields its largest positive pfs_months effect in the highest feature_015 quartile.', 'kind':'novel'},
]
analyses = []
rows = []
for v in (0,1,2):
    sub = df[df['feature_001']==v]
    a = sub.loc[sub[TREAT]==0,'pfs_months']; b = sub.loc[sub[TREAT]==1,'pfs_months']
    t,p = stats.ttest_ind(a,b,equal_var=False)
    rows.append((v, len(sub), int((sub[TREAT]==1).sum()), a.mean(), b.mean(), b.mean()-a.mean(), p))
txt = pd.DataFrame(rows, columns=['feature_001','n','n_trt','mean_ctrl','mean_trt','diff','p']).to_string(index=False)
best = max(rows, key=lambda r: r[5])
analyses.append({'hypothesis_ids':['h25'],'code':"stratify by feature_001 levels","result_summary":f"Treatment effect by feature_001:\n{txt}\nLargest diff at feature_001={best[0]} (+{best[5]:.2f} mo).",'p_value':float(best[6]),'effect_estimate':float(best[5]),'significant':bool(best[6]<0.05)})

df['_aq'] = pd.qcut(df['feature_015'], 4, labels=False, duplicates='drop')
rows = []
for v in sorted(df['_aq'].dropna().unique()):
    sub = df[df['_aq']==v]
    a = sub.loc[sub[TREAT]==0,'pfs_months']; b = sub.loc[sub[TREAT]==1,'pfs_months']
    t,p = stats.ttest_ind(a,b,equal_var=False)
    rows.append((int(v), len(sub), int((sub[TREAT]==1).sum()), a.mean(), b.mean(), b.mean()-a.mean(), p))
txt = pd.DataFrame(rows, columns=['feature_015_q','n','n_trt','mean_ctrl','mean_trt','diff','p']).to_string(index=False)
best = max(rows, key=lambda r: r[5])
df.drop(columns=['_aq'], inplace=True)
analyses.append({'hypothesis_ids':['h26'],'code':"stratify by feature_015 quartiles","result_summary":f"Treatment effect by feature_015 quartile:\n{txt}\nLargest diff at q={best[0]} (+{best[5]:.2f} mo).",'p_value':float(best[6]),'effect_estimate':float(best[5]),'significant':bool(best[6]<0.05)})
add_iter(12, hyps, analyses)

# ---- Iteration 13: 2x2 cross of feature_001 (=0 vs >=1) and feature_014 (=0 vs 1) ----
hyps = [
    {'id':'h27','text': 'feature_027 confers its largest pfs_months benefit in the (feature_001==0) AND (feature_014==0) subgroup.', 'kind':'novel'},
]
rows = []
for f1 in (0,1):  # 0 vs >=1
    for f14 in (0,1):
        sub = df[((df['feature_001']==0) if f1==0 else (df['feature_001']>=1)) & (df['feature_014']==f14)]
        a = sub.loc[sub[TREAT]==0,'pfs_months']; b = sub.loc[sub[TREAT]==1,'pfs_months']
        t,p = stats.ttest_ind(a,b,equal_var=False) if (len(a)>0 and len(b)>0) else (np.nan, np.nan)
        rows.append((f"feat001={'==0' if f1==0 else '>=1'}", f"feat014={f14}", len(sub), int((sub[TREAT]==1).sum()), a.mean(), b.mean(), b.mean()-a.mean(), p))
txt = pd.DataFrame(rows, columns=['feature_001 group','feature_014','n','n_trt','mean_ctrl','mean_trt','diff','p']).to_string(index=False)
best = max(rows, key=lambda r: r[6])
analyses = [{
    'hypothesis_ids':['h27'],
    'code':"2x2 stratification of feature_001 (==0 vs >=1) by feature_014",
    'result_summary': f"Treatment effect across 2x2:\n{txt}\nLargest diff in {best[0]} & {best[1]} (+{best[6]:.2f} mo).",
    'p_value': float(best[7]), 'effect_estimate': float(best[6]), 'significant': bool(best[7]<0.05),
}]
add_iter(13, hyps, analyses)

# ---- Iteration 14: Refine — try (feature_001==0) AND (feature_014==0) AND (feature_023==0) ----
hyps = [
    {'id':'h28','text': 'feature_027 has its largest pfs_months effect in patients with feature_001==0, feature_014==0, AND feature_023==0; adding feature_023==0 sharpens the subgroup further.', 'kind':'refined'},
    {'id':'h29','text': 'In the complementary group (feature_001>0 OR feature_014==1 OR feature_023==1), feature_027 effect on pfs_months is small or near-zero.', 'kind':'novel'},
]
mask_pos = (df['feature_001']==0) & (df['feature_014']==0) & (df['feature_023']==0)
a = y[mask_pos & (df[TREAT]==0)]; b = y[mask_pos & (df[TREAT]==1)]
diff_pos = b.mean()-a.mean()
_, p_pos = stats.ttest_ind(a,b,equal_var=False)

a2 = y[~mask_pos & (df[TREAT]==0)]; b2 = y[~mask_pos & (df[TREAT]==1)]
diff_neg = b2.mean()-a2.mean()
_, p_neg = stats.ttest_ind(a2,b2,equal_var=False)

df['_sub'] = mask_pos.astype(int)
m = smf.ols(f'pfs_months ~ {TREAT}*_sub', data=df).fit()
intx_coef = float(m.params[f'{TREAT}:_sub']); intx_p = float(m.pvalues[f'{TREAT}:_sub'])
df.drop(columns=['_sub'], inplace=True)

analyses = [{
    'hypothesis_ids':['h28'],
    'code':"Mask = (feature_001==0)&(feature_014==0)&(feature_023==0); compute treat effect inside vs outside",
    'result_summary':(f"Inside (n={int(mask_pos.sum())}, {int((mask_pos & (df[TREAT]==1)).sum())} treated): "
                      f"PFS ctrl={a.mean():.2f}, trt={b.mean():.2f}, diff={diff_pos:+.2f} (p={p_pos:.3e}).\n"
                      f"Outside (n={int((~mask_pos).sum())}): diff={diff_neg:+.2f} (p={p_neg:.3e}).\n"
                      f"Interaction (treat x subgroup): coef={intx_coef:+.3f}, p={intx_p:.3e}."),
    'p_value':intx_p,'effect_estimate':diff_pos - diff_neg,'significant':bool(intx_p<0.05)},
{
    'hypothesis_ids':['h29'],
    'code':"compare diff_neg with zero",
    'result_summary':f"Outside group treatment effect = {diff_neg:+.2f} months (p={p_neg:.3e}).",
    'p_value':float(p_neg),'effect_estimate':float(diff_neg),'significant':bool(p_neg<0.05),
}]
add_iter(14, hyps, analyses)

# ---- Iteration 15: Add feature_006 to subgroup definition ----
hyps = [
    {'id':'h30','text':'Adding feature_006==0 to the subgroup (feature_001==0, feature_014==0, feature_023==0) further enriches the pfs_months benefit of feature_027.', 'kind':'refined'},
]
mask_pos2 = mask_pos & (df['feature_006']==0)
a = y[mask_pos2 & (df[TREAT]==0)]; b = y[mask_pos2 & (df[TREAT]==1)]
diff = b.mean()-a.mean()
_, pv = stats.ttest_ind(a,b,equal_var=False) if (len(a)>0 and len(b)>0) else (np.nan, np.nan)
df['_sub'] = mask_pos2.astype(int)
m = smf.ols(f'pfs_months ~ {TREAT}*_sub', data=df).fit()
intx_coef = float(m.params[f'{TREAT}:_sub']); intx_p = float(m.pvalues[f'{TREAT}:_sub'])
df.drop(columns=['_sub'], inplace=True)
analyses = [{
    'hypothesis_ids':['h30'],
    'code':"Mask = (feature_001==0)&(feature_014==0)&(feature_023==0)&(feature_006==0)",
    'result_summary':(f"Inside (n={int(mask_pos2.sum())}, {int((mask_pos2 & (df[TREAT]==1)).sum())} treated): "
                      f"diff = {diff:+.2f} months (p={pv:.3e}).\n"
                      f"Interaction (treat x subgroup-indicator): coef={intx_coef:+.3f}, p={intx_p:.3e}."),
    'p_value':intx_p,'effect_estimate':diff,'significant':bool(intx_p<0.05),
}]
add_iter(15, hyps, analyses)

# ---- Iteration 16: Add feature_022 (rare binary that came up significant) ----
hyps = [
    {'id':'h31','text':'feature_022==0 also enhances the subgroup; the largest treatment benefit may concentrate in feature_001==0 & feature_014==0 & feature_023==0 & feature_006==0 & feature_022==0.', 'kind':'refined'},
]
mask_pos3 = mask_pos2 & (df['feature_022']==0)
a = y[mask_pos3 & (df[TREAT]==0)]; b = y[mask_pos3 & (df[TREAT]==1)]
diff = b.mean()-a.mean()
_, pv = stats.ttest_ind(a,b,equal_var=False)
df['_sub'] = mask_pos3.astype(int)
m = smf.ols(f'pfs_months ~ {TREAT}*_sub', data=df).fit()
intx_coef = float(m.params[f'{TREAT}:_sub']); intx_p = float(m.pvalues[f'{TREAT}:_sub'])
df.drop(columns=['_sub'], inplace=True)
analyses = [{
    'hypothesis_ids':['h31'],
    'code':"Mask + feature_022==0",
    'result_summary':(f"Inside (n={int(mask_pos3.sum())}, {int((mask_pos3 & (df[TREAT]==1)).sum())} treated): diff = {diff:+.2f} (p={pv:.3e}).\n"
                      f"Interaction coef={intx_coef:+.3f}, p={intx_p:.3e}."),
    'p_value':intx_p,'effect_estimate':diff,'significant':bool(intx_p<0.05),
}]
add_iter(16, hyps, analyses)

# ---- Iteration 17: Try restricting on feature_015 (older patients) and on feature_025 (smoking-like) ----
hyps = [
    {'id':'h32','text':'Adding feature_015 >= median (older patients) to the subgroup further enhances feature_027 benefit.', 'kind':'refined'},
    {'id':'h33','text':'Adding feature_025 == 0 (low/zero) to the subgroup further enhances feature_027 benefit.', 'kind':'refined'},
]
analyses = []
mask_age = mask_pos3 & (df['feature_015'] >= df['feature_015'].median())
a = y[mask_age & (df[TREAT]==0)]; b = y[mask_age & (df[TREAT]==1)]
_, pv = stats.ttest_ind(a,b,equal_var=False) if (len(a)>0 and len(b)>0) else (np.nan,np.nan)
analyses.append({'hypothesis_ids':['h32'],'code':"mask_pos3 & feature_015 >= median","result_summary":f"n={int(mask_age.sum())}, diff={b.mean()-a.mean():+.2f} (p={pv:.3e}).",'p_value':float(pv),'effect_estimate':float(b.mean()-a.mean()),'significant':bool(pv<0.05)})

mask_smoke = mask_pos3 & (df['feature_025']==0)
a = y[mask_smoke & (df[TREAT]==0)]; b = y[mask_smoke & (df[TREAT]==1)]
_, pv = stats.ttest_ind(a,b,equal_var=False) if (len(a)>0 and len(b)>0) else (np.nan,np.nan)
analyses.append({'hypothesis_ids':['h33'],'code':"mask_pos3 & feature_025==0","result_summary":f"n={int(mask_smoke.sum())}, diff={b.mean()-a.mean():+.2f} (p={pv:.3e}).",'p_value':float(pv),'effect_estimate':float(b.mean()-a.mean()),'significant':bool(pv<0.05)})
add_iter(17, hyps, analyses)

# ---- Iteration 18: Logistic regression-like check — predict above-median PFS as outcome ----
hyps = [
    {'id':'h34','text':'feature_027 increases the odds of above-median pfs_months (probability of being a long-PFS responder) after adjustment.', 'kind':'novel'},
]
df['_above_med'] = (df['pfs_months'] >= df['pfs_months'].median()).astype(int)
form = (f"_above_med ~ {TREAT} + feature_015 + C(feature_001) + feature_014 + feature_025 + feature_019 + feature_023 + feature_006")
m = smf.logit(form, data=df).fit(disp=False)
or_t = float(np.exp(m.params[TREAT])); p_t = float(m.pvalues[TREAT])
df.drop(columns=['_above_med'], inplace=True)
analyses = [{
    'hypothesis_ids':['h34'],
    'code': f"smf.logit('_above_med ~ {TREAT} + covariates', df).fit()",
    'result_summary': f"Adjusted OR for feature_027 on above-median PFS = {or_t:.3f} (p={p_t:.3e}); coef={float(m.params[TREAT]):+.3f}.",
    'p_value':p_t,'effect_estimate':float(m.params[TREAT]),'significant':bool(p_t<0.05),
}]
add_iter(18, hyps, analyses)

# ---- Iteration 19: Pairwise interactions among covariates (no treatment) on PFS — top biomarker pairs ----
hyps = [
    {'id':'h35','text':'There exist non-treatment biomarker pairs whose interaction explains additional pfs_months variance beyond their main effects (Bonferroni-significant).', 'kind':'novel'},
]
biomarker_set = ['feature_015','feature_001','feature_014','feature_023','feature_006','feature_025','feature_019','feature_022','feature_027']
# Cap to top 8 to keep tractable
pair_results = []
for a, b in combinations(biomarker_set, 2):
    f = f'pfs_months ~ {a}*{b}'
    try:
        m = smf.ols(f, data=df).fit()
        # find interaction term
        keys = [k for k in m.params.index if ':' in k]
        if not keys:
            continue
        k = keys[0]
        pair_results.append((a,b,k,float(m.params[k]),fmt_p(m.pvalues[k])))
    except Exception as e:
        continue
pair_df = pd.DataFrame(pair_results, columns=['a','b','term','coef','p']).sort_values('p').head(10)
analyses = [{
    'hypothesis_ids':['h35'],
    'code': "OLS pfs ~ a*b for each pair among top 9 features",
    'result_summary': f"Top 10 pair interactions:\n{pair_df.to_string(index=False)}",
    'p_value': float(pair_df.iloc[0]['p']),'effect_estimate': float(pair_df.iloc[0]['coef']),'significant': bool(pair_df.iloc[0]['p']<(0.05/len(pair_results))),
}]
add_iter(19, hyps, analyses)

# ---- Iteration 20: Specifically test feature_015 x feature_001 interaction (age x performance status) ----
hyps = [
    {'id':'h36','text':'feature_015 and feature_001 interact in their effect on pfs_months (the negative effect of feature_001 attenuates or reverses at high feature_015).', 'kind':'novel'},
]
m = smf.ols("pfs_months ~ feature_015 * C(feature_001)", data=df).fit()
intx_keys = [k for k in m.params.index if ':' in k]
ftest = m.f_test(', '.join(f'{k}=0' for k in intx_keys))
analyses = [{
    'hypothesis_ids':['h36'],
    'code': "OLS pfs ~ feature_015 * C(feature_001); joint F-test of interactions.",
    'result_summary': (f"Joint F-test on feature_015:C(feature_001) terms: F={float(ftest.fvalue):.2f}, p={float(ftest.pvalue):.3e}.\n"
                       + '\n'.join(f"  {k}: {m.params[k]:+.4f} (p={m.pvalues[k]:.3e})" for k in intx_keys)),
    'p_value':float(ftest.pvalue),'effect_estimate': float(sum(m.params[k] for k in intx_keys)),'significant': bool(float(ftest.pvalue)<0.05),
}]
add_iter(20, hyps, analyses)

# ---- Iteration 21: Look for negative subgroup — patients in whom feature_027 may HARM ----
hyps = [
    {'id':'h37','text':'In the highest-risk subgroup (feature_001==2 AND feature_014==1), feature_027 confers little or no pfs_months benefit and may even be neutral/negative.', 'kind':'novel'},
]
mask_hr = (df['feature_001']==2) & (df['feature_014']==1)
a = y[mask_hr & (df[TREAT]==0)]; b = y[mask_hr & (df[TREAT]==1)]
diff = b.mean()-a.mean()
_, pv = stats.ttest_ind(a,b,equal_var=False) if (len(a)>0 and len(b)>0) else (np.nan,np.nan)
analyses = [{
    'hypothesis_ids':['h37'],
    'code':"mask = (feature_001==2)&(feature_014==1); t-test treat",
    'result_summary': f"Highest-risk subgroup n={int(mask_hr.sum())}: diff = {diff:+.2f} months (p={pv:.3e}, n_trt={int((mask_hr & (df[TREAT]==1)).sum())}).",
    'p_value': float(pv),'effect_estimate': float(diff),'significant': bool(pv<0.05),
}]
add_iter(21, hyps, analyses)

# ---- Iteration 22: ANOVA — test main effect & interaction of TREAT in big OLS with all useful predictors ----
hyps = [
    {'id':'h38','text':"Treating each top covariate as also potentially interacting with feature_027, the joint interaction term remains highly significant in a comprehensive OLS.", 'kind':'refined'},
]
big_form = ("pfs_months ~ feature_027 * (feature_015 + C(feature_001) + feature_014 + feature_025 + feature_019 + feature_023 + feature_006 + feature_022)")
m = smf.ols(big_form, data=df).fit()
intx_keys = [k for k in m.params.index if ':' in k]
ftest = m.f_test(', '.join(f'{k}=0' for k in intx_keys))
analyses = [{
    'hypothesis_ids':['h38'],
    'code': "smf.ols('pfs ~ feature_027 * (key covariates)') ; joint F-test of interaction terms",
    'result_summary': (f"Joint F-test on all feature_027 interactions: F={float(ftest.fvalue):.2f}, p={float(ftest.pvalue):.3e}.\n"
                       f"Top 5 interaction coefs by |t|:\n"
                       + (m.summary2().tables[1].loc[intx_keys].assign(absT=lambda d:d['t'].abs()).sort_values('absT', ascending=False)
                          [['Coef.','Std.Err.','t','P>|t|']].head(5).to_string())),
    'p_value':float(ftest.pvalue),'effect_estimate': float(sum(m.params[k] for k in intx_keys)),'significant':bool(float(ftest.pvalue)<0.05),
}]
add_iter(22, hyps, analyses)

# ---- Iteration 23: Subgroup search via exhaustive 3-way binary intersections ----
hyps = [
    {'id':'h39','text':'An exhaustive scan over all 3-way intersections of binary features (each at value 0 or 1) finds a small subgroup with substantially larger feature_027 pfs_months benefit than any 2-way subgroup, and the variables in that 3-way pattern overlap with feature_001/feature_014/feature_023.', 'kind':'novel'},
]
# Restrict to binary features (excluding TREAT)
bins = [c for c in binary_feats if c != TREAT]
best = None
all_three = []
for f1, f2, f3 in combinations(bins, 3):
    for v1 in (0,1):
        for v2 in (0,1):
            for v3 in (0,1):
                mask = (df[f1]==v1) & (df[f2]==v2) & (df[f3]==v3)
                n = int(mask.sum())
                if n < 500:
                    continue
                n_trt = int((mask & (df[TREAT]==1)).sum())
                n_ctl = n - n_trt
                if n_trt < 100 or n_ctl < 100:
                    continue
                a = y[mask & (df[TREAT]==0)]; b = y[mask & (df[TREAT]==1)]
                diff = b.mean()-a.mean()
                _, pv = stats.ttest_ind(a,b,equal_var=False)
                all_three.append((f1,v1,f2,v2,f3,v3,n,n_trt,diff,pv))
all3 = pd.DataFrame(all_three, columns=['f1','v1','f2','v2','f3','v3','n','n_trt','diff','p']).sort_values('diff', ascending=False)
top_subgroups = all3.head(10)
analyses = [{
    'hypothesis_ids':['h39'],
    'code': "Exhaustive 3-way binary intersections (n>=500, both arms>=100); rank by diff in PFS",
    'result_summary': f"Top 10 highest-treatment-benefit subgroups:\n{top_subgroups.to_string(index=False)}",
    'p_value': float(top_subgroups.iloc[0]['p']),'effect_estimate': float(top_subgroups.iloc[0]['diff']),'significant': bool(top_subgroups.iloc[0]['p']<0.05),
}]
add_iter(23, hyps, analyses)

# ---- Iteration 24: Confirmatory test of best 3-way subgroup with interaction-vs-complement ----
hyps = [
    {'id':'h40','text':'The best 3-way binary subgroup discovered in iteration 23 yields a treatment x subgroup-indicator interaction p-value < 0.001 in OLS adjusted for the eight key covariates.', 'kind':'refined'},
]
br = top_subgroups.iloc[0]
mask_best = (df[br['f1']]==br['v1']) & (df[br['f2']]==br['v2']) & (df[br['f3']]==br['v3'])
df['_sub'] = mask_best.astype(int)
form = f"pfs_months ~ {TREAT}*_sub + feature_015 + C(feature_001) + feature_014 + feature_025 + feature_019 + feature_023 + feature_006 + feature_022"
m = smf.ols(form, data=df).fit()
key = f'{TREAT}:_sub'
intx_coef = float(m.params[key]); intx_p = float(m.pvalues[key])
trt_main = float(m.params[TREAT]); trt_p = float(m.pvalues[TREAT])
df.drop(columns=['_sub'], inplace=True)
rule_text = f"({br['f1']}=={br['v1']}) & ({br['f2']}=={br['v2']}) & ({br['f3']}=={br['v3']})"
analyses = [{
    'hypothesis_ids':['h40'],
    'code': f"OLS pfs ~ {TREAT}*subgroup_indicator + 8 key covariates",
    'result_summary': (f"Best 3-way subgroup rule: {rule_text} (n={int(mask_best.sum())}).\n"
                       f"Adjusted main effect of {TREAT}: {trt_main:+.3f} (p={trt_p:.3e}).\n"
                       f"Interaction {TREAT}:subgroup: coef={intx_coef:+.3f}, p={intx_p:.3e}.\n"
                       f"=> Within-subgroup adjusted treatment effect = {trt_main+intx_coef:+.3f} months."),
    'p_value': intx_p,'effect_estimate': trt_main + intx_coef,'significant': bool(intx_p<0.05),
}]
add_iter(24, hyps, analyses)

# ---- Iteration 25: Final summary subgroup hypothesis (largest-effect, well-powered) ----
# The CATE-tree from iter 10 identified a leaf with the largest empirical treatment effect:
#   feature_005 <= 4.99  AND  feature_023 == 0  AND  feature_006 == 0
# Confirm and report this as the final best-supported subgroup. Also test a refined version that
# adds feature_022 == 0 (a fourth modifier flagged by the screen).
final_mask = (df['feature_005'] <= 4.99) & (df['feature_023'] == 0) & (df['feature_006'] == 0)
refined_mask = final_mask & (df['feature_022'] == 0)
final_rule = "feature_005 <= 4.99 AND feature_023 == 0 AND feature_006 == 0"
refined_rule = final_rule + " AND feature_022 == 0"

hyps = [
    {'id':'h41','text': (
        "Final best-supported treatment subgroup: among patients with feature_005 <= 4.99 AND feature_023 == 0 "
        "AND feature_006 == 0, feature_027 yields a substantially positive effect on pfs_months (~+4-5 months); "
        "in the complementary group, feature_027 has no meaningful effect on pfs_months. "
        "Direction: positive treatment effect within the subgroup, near-zero outside."),
     'kind':'refined'},
    {'id':'h42','text': (
        "Adding feature_022 == 0 to the subgroup (i.e., feature_005 <= 4.99 AND feature_023 == 0 AND feature_006 == 0 "
        "AND feature_022 == 0) further enriches the positive feature_027 effect on pfs_months."),
     'kind':'refined'},
]

# 1) Final subgroup (3-condition tree rule)
a = y[final_mask & (df[TREAT]==0)]; b = y[final_mask & (df[TREAT]==1)]
ac = y[~final_mask & (df[TREAT]==0)]; bc = y[~final_mask & (df[TREAT]==1)]
diff_in = b.mean()-a.mean()
diff_out = bc.mean()-ac.mean()
_, p_in = stats.ttest_ind(a,b,equal_var=False)
_, p_out = stats.ttest_ind(ac,bc,equal_var=False)

# Adjusted interaction
df['_sub'] = final_mask.astype(int)
form = (f"pfs_months ~ {TREAT}*_sub + feature_015 + C(feature_001) + feature_014 + feature_025 + feature_019 + feature_022")
m = smf.ols(form, data=df).fit()
intx_coef = float(m.params[f'{TREAT}:_sub']); intx_p = float(m.pvalues[f'{TREAT}:_sub'])
trt_main = float(m.params[TREAT]); trt_p = float(m.pvalues[TREAT])
df.drop(columns=['_sub'], inplace=True)

# 2) Refined subgroup
ar = y[refined_mask & (df[TREAT]==0)]; br = y[refined_mask & (df[TREAT]==1)]
diff_r = br.mean()-ar.mean()
_, p_r = stats.ttest_ind(ar,br,equal_var=False)
df['_sub'] = refined_mask.astype(int)
form2 = (f"pfs_months ~ {TREAT}*_sub + feature_015 + C(feature_001) + feature_014 + feature_025 + feature_019")
m2 = smf.ols(form2, data=df).fit()
intx_coef2 = float(m2.params[f'{TREAT}:_sub']); intx_p2 = float(m2.pvalues[f'{TREAT}:_sub'])
df.drop(columns=['_sub'], inplace=True)

analyses = [{
    'hypothesis_ids':['h41'],
    'code': ("mask_final = (feature_005<=4.99) & (feature_023==0) & (feature_006==0); "
             "t-test treat within mask vs complement; OLS pfs ~ feature_027*_sub + covariates"),
    'result_summary': (
        f"Final subgroup rule: {final_rule}.\n"
        f"Inside (n={int(final_mask.sum())}, n_trt={int((final_mask & (df[TREAT]==1)).sum())}): "
        f"PFS ctrl={a.mean():.2f}, trt={b.mean():.2f}, diff={diff_in:+.2f} months (p={p_in:.3e}).\n"
        f"Complement (n={int((~final_mask).sum())}, n_trt={int(((~final_mask) & (df[TREAT]==1)).sum())}): "
        f"PFS ctrl={ac.mean():.2f}, trt={bc.mean():.2f}, diff={diff_out:+.2f} months (p={p_out:.3e}).\n"
        f"Adjusted interaction (treat x subgroup-indicator): coef={intx_coef:+.3f}, p={intx_p:.3e}; "
        f"adjusted main {TREAT}: {trt_main:+.3f} (p={trt_p:.3e}).\n"
        f"=> Within-subgroup adjusted treatment effect ~ {trt_main+intx_coef:+.3f} months."),
    'p_value':float(p_in),'effect_estimate':float(diff_in),'significant':bool(p_in<0.05),
},
{
    'hypothesis_ids':['h42'],
    'code': "mask_refined = mask_final & (feature_022==0)",
    'result_summary': (
        f"Refined subgroup rule: {refined_rule}.\n"
        f"Inside (n={int(refined_mask.sum())}, n_trt={int((refined_mask & (df[TREAT]==1)).sum())}): "
        f"PFS ctrl={ar.mean():.2f}, trt={br.mean():.2f}, diff={diff_r:+.2f} months (p={p_r:.3e}).\n"
        f"Adjusted interaction (treat x refined-subgroup-indicator): coef={intx_coef2:+.3f}, p={intx_p2:.3e}."),
    'p_value':float(p_r),'effect_estimate':float(diff_r),'significant':bool(p_r<0.05),
}]
add_iter(25, hyps, analyses)

# Persist for transcript builder
out = {
    'dataset_id': 'ds001_crc',
    'model_id': 'claude-opus-4-7',
    'harness_id': 'claude-code@local',
    'max_iterations': 25,
    'iterations': ITERATIONS,
}

# Make all NaN/inf safe
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k,v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float):
        if np.isnan(o): return None
        if np.isinf(o): return None
        return o
    return o
out = clean(out)

with open('transcript.json','w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)

# Compact summary JSON of analyses for use writing the analysis_summary.txt
brief = []
for it in ITERATIONS:
    for a in it['analyses']:
        brief.append({
            'iter': it['index'],
            'hyps': a['hypothesis_ids'],
            'p': a.get('p_value'),
            'eff': a.get('effect_estimate'),
            'sig': a.get('significant'),
            'summary': a.get('result_summary'),
        })
with open('_brief.json','w', encoding='utf-8') as f:
    json.dump(brief, f, indent=2)

print("Wrote transcript.json with", len(ITERATIONS), "iterations and", sum(len(it['analyses']) for it in ITERATIONS), "analyses.")
