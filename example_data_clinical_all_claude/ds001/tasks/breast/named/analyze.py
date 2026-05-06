"""Comprehensive analysis of ds001_breast for transcript generation."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_parquet("dataset.parquet")
print("N =", len(df))

OUT = {}

def lr(formula_x, y='pfs_months', data=None):
    """Run OLS, return (coef, p) for first non-intercept term."""
    if data is None:
        data = df
    X = sm.add_constant(data[formula_x])
    yv = data[y]
    m = sm.OLS(yv, X).fit()
    return m

def ttest_groups(col, val_col='pfs_months'):
    a = df.loc[df[col] == 1, val_col]
    b = df.loc[df[col] == 0, val_col]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return a.mean() - b.mean(), float(p), float(a.mean()), float(b.mean()), len(a), len(b)

def correlation(col, val_col='pfs_months'):
    r, p = stats.pearsonr(df[col], df[val_col])
    return float(r), float(p)

# === ITER 1: Main effects of binary features on pfs_months ===
print("\n=== ITER 1: Binary main effects ===")
binary_cols = ['sex_female','stage_iv','has_brain_mets','node_positive','postmenopausal',
               'er_positive','pr_positive','her2_positive','her2_low','brca1_mutation',
               'brca2_mutation','pik3ca_mutation']
iter1 = {}
for c in binary_cols:
    diff, p, m1, m0, n1, n0 = ttest_groups(c)
    iter1[c] = {'diff': diff, 'p': p, 'mean1': m1, 'mean0': m0, 'n1': n1, 'n0': n0}
    print(f"  {c}: diff={diff:.3f}, p={p:.3e}, n1={n1}, mean1={m1:.3f}, mean0={m0:.3f}")
OUT['iter1'] = iter1

# === ITER 2: Continuous main effects on pfs_months ===
print("\n=== ITER 2: Continuous main effects ===")
cont_cols = ['age_years','ecog_ps','ki67_pct','tumor_size_cm','albumin_g_dl','ldh_u_l',
             'weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
             'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
             'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']
iter2 = {}
for c in cont_cols:
    r, p = correlation(c)
    iter2[c] = {'r': r, 'p': p}
    print(f"  {c}: r={r:.4f}, p={p:.3e}")
OUT['iter2'] = iter2

# === ITER 3: Treatment main effects ===
print("\n=== ITER 3: Treatment main effects ===")
tx_cols = ['treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
           'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab']
iter3 = {}
for c in tx_cols:
    diff, p, m1, m0, n1, n0 = ttest_groups(c)
    iter3[c] = {'diff': diff, 'p': p, 'mean1': m1, 'mean0': m0, 'n1': n1, 'n0': n0}
    print(f"  {c}: diff={diff:.3f}, p={p:.3e}, n1={n1}, mean1={m1:.3f}, mean0={m0:.3f}")
OUT['iter3'] = iter3

with open("results_iter1_3.json","w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("\nSaved results_iter1_3.json")
