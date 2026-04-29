"""
Iterative analysis of ds001_crc dataset.
Runs each iteration's tests, accumulates results that we'll compile into transcript.json.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import defaultdict

df = pd.read_parquet('dataset.parquet')
print('Loaded:', df.shape)

results = {'iterations': []}

def add_iter(idx, hypotheses, analyses):
    results['iterations'].append({
        'index': idx,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses,
    })

# ============================================================
# ITERATION 1 - Identify strongest main effects on PFS
# ============================================================
hyps1 = [
    {'id': 'h1_1', 'text': 'feature_078 (continuous, range 30-90) is positively associated with pfs_months in this cohort.', 'kind': 'novel'},
    {'id': 'h1_2', 'text': 'feature_057 (ordinal, levels 0-2) is negatively associated with pfs_months: higher levels predict shorter PFS.', 'kind': 'novel'},
    {'id': 'h1_3', 'text': 'feature_051 (binary) is associated with shorter pfs_months when present (=1) than when absent (=0).', 'kind': 'novel'},
    {'id': 'h1_4', 'text': 'feature_038 (binary) is associated with longer pfs_months when present (=1) than when absent (=0).', 'kind': 'novel'},
]

an1 = []
# Pearson + linear regression for feature_078
slope, intercept, r, p, se = stats.linregress(df['feature_078'], df['pfs_months'])
an1.append({
    'hypothesis_ids': ['h1_1'],
    'code': "stats.linregress(df['feature_078'], df['pfs_months'])",
    'result_summary': f"Linear regression: slope={slope:.4f} months per unit feature_078, r={r:.4f}, p={p:.3e}. Strong positive association.",
    'p_value': float(p), 'effect_estimate': float(slope), 'significant': bool(p < 0.05),
})
# feature_057 ordinal
slope, intercept, r, p, se = stats.linregress(df['feature_057'], df['pfs_months'])
an1.append({
    'hypothesis_ids': ['h1_2'],
    'code': "stats.linregress(df['feature_057'], df['pfs_months'])",
    'result_summary': f"Linear regression on ordinal feature_057: slope={slope:.4f} months/level (negative), r={r:.4f}, p={p:.3e}. Means: lvl0={df.loc[df.feature_057==0,'pfs_months'].mean():.2f}, lvl1={df.loc[df.feature_057==1,'pfs_months'].mean():.2f}, lvl2={df.loc[df.feature_057==2,'pfs_months'].mean():.2f}.",
    'p_value': float(p), 'effect_estimate': float(slope), 'significant': bool(p < 0.05),
})
# feature_051 binary
g1 = df.loc[df.feature_051==1, 'pfs_months']; g0 = df.loc[df.feature_051==0, 'pfs_months']
t, p = stats.ttest_ind(g1, g0)
diff = g1.mean() - g0.mean()
an1.append({
    'hypothesis_ids': ['h1_3'],
    'code': "stats.ttest_ind(df.loc[df.feature_051==1,'pfs_months'], df.loc[df.feature_051==0,'pfs_months'])",
    'result_summary': f"T-test feature_051: mean(=1)={g1.mean():.3f} vs mean(=0)={g0.mean():.3f}, diff={diff:.3f} months, p={p:.3e}.",
    'p_value': float(p), 'effect_estimate': float(diff), 'significant': bool(p < 0.05),
})
# feature_038 binary
g1 = df.loc[df.feature_038==1, 'pfs_months']; g0 = df.loc[df.feature_038==0, 'pfs_months']
t, p = stats.ttest_ind(g1, g0)
diff = g1.mean() - g0.mean()
an1.append({
    'hypothesis_ids': ['h1_4'],
    'code': "stats.ttest_ind(df.loc[df.feature_038==1,'pfs_months'], df.loc[df.feature_038==0,'pfs_months'])",
    'result_summary': f"T-test feature_038: mean(=1)={g1.mean():.3f} vs mean(=0)={g0.mean():.3f}, diff={diff:.3f} months, p={p:.3e}.",
    'p_value': float(p), 'effect_estimate': float(diff), 'significant': bool(p < 0.05),
})
add_iter(1, hyps1, an1)

# ============================================================
# ITERATION 2 - Other binary and ordinal features showing strong associations
# ============================================================
hyps2 = [
    {'id': 'h2_1', 'text': 'feature_099 (continuous, range 0-24.6) is negatively associated with pfs_months.', 'kind': 'novel'},
    {'id': 'h2_2', 'text': 'feature_092 (continuous, range 1.5-5.5) is positively associated with pfs_months.', 'kind': 'novel'},
    {'id': 'h2_3', 'text': 'feature_013 (binary) is associated with shorter pfs_months when =1 than when =0.', 'kind': 'novel'},
    {'id': 'h2_4', 'text': 'feature_043 (binary) is associated with shorter pfs_months when =1 than when =0.', 'kind': 'novel'},
    {'id': 'h2_5', 'text': 'feature_009 (continuous, range 0-777) is negatively associated with pfs_months.', 'kind': 'novel'},
]

an2 = []
for hid, col in [('h2_1','feature_099'),('h2_2','feature_092'),('h2_5','feature_009')]:
    slope, intercept, r, p, se = stats.linregress(df[col], df['pfs_months'])
    an2.append({
        'hypothesis_ids': [hid],
        'code': f"stats.linregress(df['{col}'], df['pfs_months'])",
        'result_summary': f"Linear regression {col}: slope={slope:.5f} months/unit, r={r:.4f}, p={p:.3e}.",
        'p_value': float(p), 'effect_estimate': float(slope), 'significant': bool(p < 0.05),
    })
for hid, col in [('h2_3','feature_013'),('h2_4','feature_043')]:
    g1 = df.loc[df[col]==1, 'pfs_months']; g0 = df.loc[df[col]==0, 'pfs_months']
    t, p = stats.ttest_ind(g1, g0)
    diff = g1.mean() - g0.mean()
    an2.append({
        'hypothesis_ids': [hid],
        'code': f"stats.ttest_ind(df.loc[df.{col}==1,'pfs_months'], df.loc[df.{col}==0,'pfs_months'])",
        'result_summary': f"T-test {col}: mean(=1)={g1.mean():.3f} vs mean(=0)={g0.mean():.3f}, diff={diff:.3f} months, p={p:.3e}.",
        'p_value': float(p), 'effect_estimate': float(diff), 'significant': bool(p < 0.05),
    })
add_iter(2, hyps2, an2)

# ============================================================
# ITERATION 3 - Categorical: race-like (feature_064) and insurance (feature_018)
# ============================================================
hyps3 = [
    {'id': 'h3_1', 'text': 'pfs_months differs across feature_064 categories (white/hispanic/black/asian/other).', 'kind': 'novel'},
    {'id': 'h3_2', 'text': 'Among feature_064, black patients have shorter mean pfs_months than white patients (disparity).', 'kind': 'novel'},
    {'id': 'h3_3', 'text': 'pfs_months differs across feature_018 categories (medicare/private/medicaid/uninsured).', 'kind': 'novel'},
    {'id': 'h3_4', 'text': 'Within feature_018, uninsured patients have shorter mean pfs_months than privately insured patients.', 'kind': 'novel'},
]

an3 = []
groups = [df.loc[df.feature_064==v,'pfs_months'].values for v in df.feature_064.unique()]
f, p = stats.f_oneway(*groups)
means = {v: float(df.loc[df.feature_064==v,'pfs_months'].mean()) for v in df.feature_064.unique()}
an3.append({
    'hypothesis_ids': ['h3_1'],
    'code': "stats.f_oneway by feature_064",
    'result_summary': f"ANOVA across feature_064: F={f:.2f}, p={p:.3e}. Means: {means}",
    'p_value': float(p), 'effect_estimate': float(max(means.values())-min(means.values())), 'significant': bool(p < 0.05),
})
g_b = df.loc[df.feature_064=='black','pfs_months']; g_w = df.loc[df.feature_064=='white','pfs_months']
t, p = stats.ttest_ind(g_b, g_w)
diff = g_b.mean() - g_w.mean()
an3.append({
    'hypothesis_ids': ['h3_2'],
    'code': "stats.ttest_ind black vs white feature_064",
    'result_summary': f"Black mean={g_b.mean():.3f} vs white mean={g_w.mean():.3f}, diff={diff:.3f} months, p={p:.3e}.",
    'p_value': float(p), 'effect_estimate': float(diff), 'significant': bool(p < 0.05),
})
groups = [df.loc[df.feature_018==v,'pfs_months'].values for v in df.feature_018.unique()]
f, p = stats.f_oneway(*groups)
means = {v: float(df.loc[df.feature_018==v,'pfs_months'].mean()) for v in df.feature_018.unique()}
an3.append({
    'hypothesis_ids': ['h3_3'],
    'code': "stats.f_oneway by feature_018",
    'result_summary': f"ANOVA across feature_018: F={f:.2f}, p={p:.3e}. Means: {means}",
    'p_value': float(p), 'effect_estimate': float(max(means.values())-min(means.values())), 'significant': bool(p < 0.05),
})
g_un = df.loc[df.feature_018=='uninsured','pfs_months']; g_pr = df.loc[df.feature_018=='private','pfs_months']
t, p = stats.ttest_ind(g_un, g_pr)
diff = g_un.mean() - g_pr.mean()
an3.append({
    'hypothesis_ids': ['h3_4'],
    'code': "stats.ttest_ind uninsured vs private feature_018",
    'result_summary': f"Uninsured mean={g_un.mean():.3f} vs private mean={g_pr.mean():.3f}, diff={diff:.3f} months, p={p:.3e}.",
    'p_value': float(p), 'effect_estimate': float(diff), 'significant': bool(p < 0.05),
})
add_iter(3, hyps3, an3)

# Save partial
with open('iter_partial.json','w') as f: json.dump(results, f, indent=2)
print('Iter 1-3 done.')
