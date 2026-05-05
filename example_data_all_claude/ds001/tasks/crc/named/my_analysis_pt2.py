"""Refinement analyses for regorafenib heterogeneity."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import json

df = pd.read_parquet('dataset.parquet')

def strat_compare(tx, mask, label):
    sub = df[mask]
    if sub[tx].sum() < 5 or (~sub[tx].astype(bool)).sum() < 5:
        print(f"  {label}: SKIP (n_tx={sub[tx].sum()}, n_ctrl={(~sub[tx].astype(bool)).sum()})")
        return None
    a = sub.loc[sub[tx]==1,'pfs_months']
    b = sub.loc[sub[tx]==0,'pfs_months']
    diff = a.mean() - b.mean()
    t,p = stats.ttest_ind(a, b, equal_var=False)
    print(f"  {label}: n_tx={len(a)}, n_ctrl={len(b)}, mean_tx={a.mean():.3f}, mean_ctrl={b.mean():.3f}, diff={diff:.3g}, p={p:.4g}")
    return {'n_tx': len(a), 'n_ctrl': len(b), 'mean_tx': a.mean(), 'mean_ctrl': b.mean(), 'diff': diff, 'p': p}

results = {}

# ============================
# Regorafenib heterogeneity exploration
# ============================
print("\n=== Regorafenib stratified by simultaneous KRAS, BRAF, sidedness ===")

# Composite: KRAS-WT + BRAF-WT + left-sided
m1 = (df['kras_mutation']==0)&(df['braf_v600e']==0)&(df['right_sided_primary']==0)
results['rego_left_KRASWT_BRAFWT'] = strat_compare('treatment_regorafenib', m1,
    'rego: left + KRAS-WT + BRAF-WT')

m2 = ~m1
results['rego_complement'] = strat_compare('treatment_regorafenib', m2,
    'rego: complement (right OR KRAS-mut OR BRAF-mut)')

# Decompose: try each violator
print("\nDecompose violators:")
m_a = (df['right_sided_primary']==1)&(df['kras_mutation']==0)&(df['braf_v600e']==0)
results['rego_right_KRASWT_BRAFWT'] = strat_compare('treatment_regorafenib', m_a, 'rego: right + KRAS-WT + BRAF-WT')

m_b = (df['right_sided_primary']==0)&(df['kras_mutation']==1)&(df['braf_v600e']==0)
results['rego_left_KRASmut_BRAFWT'] = strat_compare('treatment_regorafenib', m_b, 'rego: left + KRAS-mut + BRAF-WT')

m_c = (df['right_sided_primary']==0)&(df['kras_mutation']==0)&(df['braf_v600e']==1)
results['rego_left_KRASWT_BRAFmut'] = strat_compare('treatment_regorafenib', m_c, 'rego: left + KRAS-WT + BRAF-mut')

# Three "pure" groups all fail simultaneously
m_d = (df['right_sided_primary']==1)&(df['kras_mutation']==1)
results['rego_right_KRASmut'] = strat_compare('treatment_regorafenib', m_d, 'rego: right + KRAS-mut')

# Add NRAS into the mix (NRAS interaction enhanced!)
print("\nWith NRAS:")
m_e = (df['kras_mutation']==0)&(df['braf_v600e']==0)&(df['right_sided_primary']==0)&(df['nras_mutation']==0)
results['rego_left_panWT'] = strat_compare('treatment_regorafenib', m_e, 'rego: left + pan-WT (no KRAS/NRAS/BRAF)')

m_f = (df['kras_mutation']==0)&(df['braf_v600e']==0)&(df['right_sided_primary']==0)&(df['nras_mutation']==1)
results['rego_left_NRASmut_only'] = strat_compare('treatment_regorafenib', m_f, 'rego: left + NRAS-mut + KRAS-WT + BRAF-WT')

# Run the 3-way interaction model: rego x kras x braf
print("\n=== Three-way interaction: rego x kras_mut x braf_v600e ===")
sub = df.copy()
sub['rk'] = sub['treatment_regorafenib']*sub['kras_mutation']
sub['rb'] = sub['treatment_regorafenib']*sub['braf_v600e']
sub['kb'] = sub['kras_mutation']*sub['braf_v600e']
sub['rkb'] = sub['rk']*sub['braf_v600e']
sub['rs'] = sub['treatment_regorafenib']*sub['right_sided_primary']
sub['rks'] = sub['rk']*sub['right_sided_primary']

X = sub[['treatment_regorafenib','kras_mutation','braf_v600e','right_sided_primary','rk','rb','kb','rs']]
m = sm.OLS(sub['pfs_months'], sm.add_constant(X)).fit()
print(m.summary())

# ============================
# Verify regorafenib effect specifically by subgroup
# ============================
print("\n=== Regorafenib by ALL combos of {right, kras, braf} ===")
for r in [0,1]:
    for k in [0,1]:
        for b in [0,1]:
            mask = (df['right_sided_primary']==r)&(df['kras_mutation']==k)&(df['braf_v600e']==b)
            label = f"rego: right={r}, kras={k}, braf={b}"
            res = strat_compare('treatment_regorafenib', mask, label)
            if res:
                results[label] = res

# Combine into one strict subgroup hypothesis: regorafenib benefits left-sided KRAS-WT BRAF-WT
# Test it cleanly
print("\n=== FINAL: regorafenib in left + KRAS-WT + BRAF-WT (predicted strong benefit) vs anyone failing this ===")
benefit_mask = (df['right_sided_primary']==0)&(df['kras_mutation']==0)&(df['braf_v600e']==0)
print(f"Benefit subgroup size: {benefit_mask.sum()}, complement: {(~benefit_mask).sum()}")
strat_compare('treatment_regorafenib', benefit_mask, 'FINAL rego benefit subgroup')
strat_compare('treatment_regorafenib', ~benefit_mask, 'FINAL rego non-benefit (any of right/kras/braf)')

# Also test if NRAS matters within the benefit subgroup
print("\n=== Within rego benefit subgroup, does NRAS matter? ===")
m_within_nras1 = benefit_mask & (df['nras_mutation']==1)
m_within_nras0 = benefit_mask & (df['nras_mutation']==0)
strat_compare('treatment_regorafenib', m_within_nras0, 'rego: benefit subgroup + NRAS WT')
strat_compare('treatment_regorafenib', m_within_nras1, 'rego: benefit subgroup + NRAS mut')

# Test interaction of regorafenib x other features in the supposed benefit subgroup
print("\n=== Test interactions within benefit subgroup ===")
sub2 = df[benefit_mask].copy()
print(f"Subset size: {len(sub2)}")
for feat in ['ecog_ps','stage_iv','msi_high','her2_amplified','ntrk_fusion','sex_female']:
    sub2['ix'] = sub2['treatment_regorafenib']*sub2[feat]
    X = sub2[['treatment_regorafenib', feat, 'ix']]
    m = sm.OLS(sub2['pfs_months'], sm.add_constant(X)).fit()
    print(f"  rego x {feat} within benefit subgroup: int_coef={m.params['ix']:.4g}, p={m.pvalues['ix']:.4g}")

# ============================
# Re-examine the NRAS finding (positive effect on PFS, opposite of KRAS)
# ============================
print("\n=== NRAS deeper look ===")
print(f"  Mean PFS NRAS+: {df.loc[df['nras_mutation']==1,'pfs_months'].mean():.3f}")
print(f"  Mean PFS NRAS-: {df.loc[df['nras_mutation']==0,'pfs_months'].mean():.3f}")
# Adjusted
X = df[['nras_mutation','kras_mutation','braf_v600e','msi_high','her2_amplified','ecog_ps','stage_iv','age_years','albumin_g_dl','weight_loss_pct_6mo','cea_ng_ml']]
m = sm.OLS(df['pfs_months'], sm.add_constant(X)).fit()
print(f"  NRAS adjusted coef: {m.params['nras_mutation']:.4g}, p={m.pvalues['nras_mutation']:.4g}")

# In rego strata
print("Within KRAS WT only (so NRAS mut matters):")
m_kw = (df['kras_mutation']==0)
strat_compare('treatment_regorafenib', m_kw & (df['nras_mutation']==1), 'rego: KRAS-WT + NRAS-mut')
strat_compare('treatment_regorafenib', m_kw & (df['nras_mutation']==0), 'rego: KRAS-WT + NRAS-WT')

# ============================
# Continuous modifiers within subgroup
# ============================
print("\n=== Within benefit subgroup, continuous modifiers ===")
for feat in ['cea_ng_ml','albumin_g_dl','ldh_u_l','crp_mg_l','nlr','weight_loss_pct_6mo','age_years','hemoglobin_g_dl']:
    sub2['ix'] = sub2['treatment_regorafenib']*sub2[feat]
    X = sub2[['treatment_regorafenib', feat, 'ix']]
    m = sm.OLS(sub2['pfs_months'], sm.add_constant(X)).fit()
    print(f"  rego x {feat} within benefit subgroup: int_coef={m.params['ix']:.4g}, p={m.pvalues['ix']:.4g}")

# ============================
# Test rego x cea interaction observed in main run
# ============================
print("\n=== Confirming rego x cea (continuous): is it in non-benefit group?")
for label, mask in [
    ('left+kraswt+brafwt', benefit_mask),
    ('rest', ~benefit_mask),
]:
    sub2 = df[mask].copy()
    sub2['ix'] = sub2['treatment_regorafenib']*sub2['cea_ng_ml']
    X = sub2[['treatment_regorafenib', 'cea_ng_ml', 'ix']]
    m = sm.OLS(sub2['pfs_months'], sm.add_constant(X)).fit()
    print(f"  {label}: rego coef={m.params['treatment_regorafenib']:.3g}, ix coef={m.params['ix']:.4g}, p={m.pvalues['ix']:.4g}")

with open('my_results_pt2.json','w') as f:
    json.dump(results, f, default=str, indent=2)

print("\nDONE.")
