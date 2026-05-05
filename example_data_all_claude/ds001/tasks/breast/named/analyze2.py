"""Iter 4-12: Treatment-by-biomarker interactions."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_parquet("dataset.parquet")
OUT = {}

def interaction_test(tx, mod, y='pfs_months', extra_cov=None):
    """OLS: y ~ tx + mod + tx*mod (+ covariates). Return interaction coef and p."""
    cols = [tx, mod]
    if extra_cov:
        cols = cols + list(extra_cov)
    X = df[cols].copy()
    X[f'{tx}_x_{mod}'] = df[tx] * df[mod]
    X = sm.add_constant(X)
    m = sm.OLS(df[y], X).fit()
    coef = m.params[f'{tx}_x_{mod}']
    p = m.pvalues[f'{tx}_x_{mod}']
    # Stratified means
    s11 = df.loc[(df[tx]==1) & (df[mod]==1), y].mean()
    s10 = df.loc[(df[tx]==1) & (df[mod]==0), y].mean()
    s01 = df.loc[(df[tx]==0) & (df[mod]==1), y].mean()
    s00 = df.loc[(df[tx]==0) & (df[mod]==0), y].mean()
    n11 = ((df[tx]==1) & (df[mod]==1)).sum()
    n10 = ((df[tx]==1) & (df[mod]==0)).sum()
    n01 = ((df[tx]==0) & (df[mod]==1)).sum()
    n00 = ((df[tx]==0) & (df[mod]==0)).sum()
    eff_in_mod = s11 - s01
    eff_off_mod = s10 - s00
    return {
        'tx': tx, 'mod': mod,
        'interaction_coef': float(coef), 'interaction_p': float(p),
        'effect_in_mod1': float(eff_in_mod), 'effect_in_mod0': float(eff_off_mod),
        'n11': int(n11), 'n10': int(n10), 'n01': int(n01), 'n00': int(n00),
        'm11': float(s11), 'm10': float(s10), 'm01': float(s01), 'm00': float(s00),
    }

binary_mods = ['stage_iv','has_brain_mets','node_positive','postmenopausal',
               'er_positive','pr_positive','her2_positive','her2_low',
               'brca1_mutation','brca2_mutation','pik3ca_mutation','sex_female']

tx_cols = ['treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
           'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab']

# Iter 4-9: tx-by-binary screen
print("\n=== ITER 4-9: Treatment x binary modifier interaction screen ===")
all_int = []
for tx in tx_cols:
    for mod in binary_mods:
        try:
            r = interaction_test(tx, mod)
            all_int.append(r)
            sig = "***" if r['interaction_p'] < 0.001 else ("**" if r['interaction_p']<0.01 else ("*" if r['interaction_p']<0.05 else ""))
            print(f"  {tx:38s} x {mod:18s}: int_coef={r['interaction_coef']:+.3f}  p={r['interaction_p']:.2e}  eff_in_mod1={r['effect_in_mod1']:+.3f}  eff_in_mod0={r['effect_in_mod0']:+.3f}  n11={r['n11']}  {sig}")
        except Exception as e:
            print(f"  {tx} x {mod}: ERROR {e}")

OUT['interactions_binary'] = all_int

# === Iter 10: Treatment x continuous (key labs) ===
print("\n=== ITER 10: Treatment x continuous biomarker interaction screen ===")
cont_mods = ['age_years','ecog_ps','ki67_pct','albumin_g_dl','weight_loss_pct_6mo','tumor_size_cm','ldh_u_l']
cont_int = []
for tx in tx_cols:
    for mod in cont_mods:
        # Standardize continuous mod
        m_std = (df[mod] - df[mod].mean()) / df[mod].std()
        X = pd.DataFrame({tx: df[tx], mod: m_std, 'inter': df[tx]*m_std})
        X = sm.add_constant(X)
        fit = sm.OLS(df['pfs_months'], X).fit()
        coef = fit.params['inter']; p = fit.pvalues['inter']
        cont_int.append({'tx':tx,'mod':mod,'coef':float(coef),'p':float(p)})
        sig = "***" if p < 0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        print(f"  {tx:38s} x {mod:18s}: coef(per +1SD)={coef:+.3f}  p={p:.2e}  {sig}")

OUT['interactions_continuous'] = cont_int

with open("results_iter4_10.json","w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("\nSaved results_iter4_10.json")
