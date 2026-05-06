"""Comprehensive analyses for ds001_crc dataset."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import json

df = pd.read_parquet('dataset.parquet')

results = {}

def linreg(y, X, label):
    Xc = sm.add_constant(X)
    m = sm.OLS(y, Xc).fit()
    return m

def report(name, est, p, sig=None):
    if sig is None:
        sig = (p is not None) and (p < 0.05)
    results[name] = {'effect': float(est) if est is not None else None,
                     'p': float(p) if p is not None else None,
                     'significant': bool(sig)}
    print(f"{name}: est={est:.4g}, p={p:.4g}, sig={sig}")

# ============================
# Iteration 1: prevalences
# ============================
print("\n=== Prevalences/distributions ===")
for c in ['sex_female','stage_iv','right_sided_primary','kras_mutation','nras_mutation',
          'braf_v600e','msi_high','her2_amplified','ntrk_fusion',
          'treatment_cetuximab','treatment_bevacizumab','treatment_pembrolizumab',
          'treatment_encorafenib','treatment_trastuzumab_tucatinib','treatment_regorafenib']:
    print(f"  {c}: mean={df[c].mean():.4f}")
print(f"  pfs_months: mean={df['pfs_months'].mean():.3f}, sd={df['pfs_months'].std():.3f}, median={df['pfs_months'].median():.3f}")
print(f"  ecog_ps distribution:\n{df['ecog_ps'].value_counts().sort_index()}")

# ============================
# Iteration 2-3: Main effects of continuous features on PFS
# ============================
print("\n=== Main effects (continuous) on pfs_months ===")
cont_feats = ['age_years','cea_ng_ml','albumin_g_dl','ldh_u_l','weight_loss_pct_6mo',
              'crp_mg_l','nlr','hemoglobin_g_dl','alkaline_phosphatase_u_l',
              'ast_u_l','alt_u_l','total_bilirubin_mg_dl','creatinine_mg_dl',
              'bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']

for c in cont_feats:
    m = linreg(df['pfs_months'], df[[c]], c)
    report(f"PFS ~ {c}", m.params[c], m.pvalues[c])

# ============================
# Iteration 4: Main effects (binary features, no treatments) on PFS
# ============================
print("\n=== Main effects (binary, demographics/biomarkers) on pfs_months ===")
binary_nontx = ['sex_female','stage_iv','right_sided_primary','kras_mutation',
                'nras_mutation','braf_v600e','msi_high','her2_amplified','ntrk_fusion','ecog_ps']
for c in binary_nontx:
    m = linreg(df['pfs_months'], df[[c]], c)
    report(f"PFS ~ {c}", m.params[c], m.pvalues[c])

# ============================
# Iteration 5: Treatment main effects
# ============================
print("\n=== Treatment main effects on pfs_months (unadjusted) ===")
treatments = ['treatment_cetuximab','treatment_bevacizumab','treatment_pembrolizumab',
              'treatment_encorafenib','treatment_trastuzumab_tucatinib','treatment_regorafenib']
for t in treatments:
    m = linreg(df['pfs_months'], df[[t]], t)
    report(f"PFS ~ {t} (unadjusted)", m.params[t], m.pvalues[t])

# ============================
# Iteration 6: Treatment main effects, adjusted
# ============================
print("\n=== Treatment main effects on pfs_months (adjusted for prognostics & biomarkers) ===")
adjust_cols = ['age_years','sex_female','ecog_ps','stage_iv','right_sided_primary',
               'kras_mutation','nras_mutation','braf_v600e','msi_high','her2_amplified',
               'ntrk_fusion','cea_ng_ml','albumin_g_dl','ldh_u_l','weight_loss_pct_6mo',
               'crp_mg_l','nlr','hemoglobin_g_dl']

for t in treatments:
    X = df[[t] + adjust_cols].copy()
    m = linreg(df['pfs_months'], X, t)
    report(f"PFS ~ {t} | adjusted", m.params[t], m.pvalues[t])

# ============================
# Iteration 7-12: Treatment x biomarker interactions
# ============================
print("\n=== Treatment x biomarker interactions ===")

def interaction_test(tx, marker):
    """Test if treatment effect differs by marker. Reports interaction coef."""
    sub = df.copy()
    sub['interaction'] = sub[tx] * sub[marker]
    X = sub[[tx, marker, 'interaction']]
    m = linreg(sub['pfs_months'], X, f"{tx} x {marker}")
    eff_in = m.params[tx] + m.params['interaction']  # effect when marker=1
    eff_out = m.params[tx]  # effect when marker=0
    return m.params['interaction'], m.pvalues['interaction'], eff_in, eff_out

interactions_to_test = [
    # Cetuximab: should not work in RAS/BRAF mutated
    ('treatment_cetuximab','kras_mutation'),
    ('treatment_cetuximab','nras_mutation'),
    ('treatment_cetuximab','braf_v600e'),
    ('treatment_cetuximab','right_sided_primary'),
    # Pembrolizumab: works in MSI-H
    ('treatment_pembrolizumab','msi_high'),
    ('treatment_pembrolizumab','kras_mutation'),
    ('treatment_pembrolizumab','braf_v600e'),
    # Encorafenib: works in BRAF V600E
    ('treatment_encorafenib','braf_v600e'),
    ('treatment_encorafenib','kras_mutation'),
    ('treatment_encorafenib','msi_high'),
    # Trastuzumab/tucatinib: works in HER2 amplified
    ('treatment_trastuzumab_tucatinib','her2_amplified'),
    ('treatment_trastuzumab_tucatinib','kras_mutation'),
    # Bevacizumab: any modifiers?
    ('treatment_bevacizumab','kras_mutation'),
    ('treatment_bevacizumab','braf_v600e'),
    ('treatment_bevacizumab','msi_high'),
    ('treatment_bevacizumab','right_sided_primary'),
    # Regorafenib
    ('treatment_regorafenib','kras_mutation'),
    ('treatment_regorafenib','braf_v600e'),
    ('treatment_regorafenib','msi_high'),
    # NTRK fusion
    ('treatment_pembrolizumab','ntrk_fusion'),
]

interaction_results = {}
for tx, m_marker in interactions_to_test:
    coef, pval, eff_in, eff_out = interaction_test(tx, m_marker)
    key = f"{tx} x {m_marker}"
    interaction_results[key] = {'interaction_coef': coef, 'p': pval,
                                 'effect_marker_pos': eff_in, 'effect_marker_neg': eff_out}
    print(f"  {key}: int_coef={coef:.4g}, p={pval:.4g}, eff(marker=1)={eff_in:.3g}, eff(marker=0)={eff_out:.3g}")
    report(key, coef, pval)

# ============================
# Iteration 13-15: Stratified treatment effects
# ============================
print("\n=== Stratified mean PFS comparisons ===")

def strat_compare(tx, mask, label):
    sub = df[mask]
    if sub[tx].sum() < 10 or (~sub[tx].astype(bool)).sum() < 10:
        return None, None, None
    a = sub.loc[sub[tx]==1,'pfs_months']
    b = sub.loc[sub[tx]==0,'pfs_months']
    diff = a.mean() - b.mean()
    t,p = stats.ttest_ind(a, b, equal_var=False)
    print(f"  {label}: n_tx={len(a)}, n_ctrl={len(b)}, diff={diff:.3g}, p={p:.4g}")
    report(label, diff, p)
    return diff, p, len(sub)

# Cetuximab in RAS-WT/BRAF-WT vs RAS-mut or BRAF-mut
mask_pan_wt = (df['kras_mutation']==0)&(df['nras_mutation']==0)&(df['braf_v600e']==0)
strat_compare('treatment_cetuximab', mask_pan_wt, 'cetux PFS in pan-RAS/BRAF WT')
strat_compare('treatment_cetuximab', ~mask_pan_wt, 'cetux PFS in any RAS/BRAF mut')

# Cetuximab in left-sided pan-WT (gold standard)
mask_left_panwt = mask_pan_wt & (df['right_sided_primary']==0)
mask_right_panwt = mask_pan_wt & (df['right_sided_primary']==1)
strat_compare('treatment_cetuximab', mask_left_panwt, 'cetux PFS in left-sided pan-WT')
strat_compare('treatment_cetuximab', mask_right_panwt, 'cetux PFS in right-sided pan-WT')

# Pembro in MSI-H
strat_compare('treatment_pembrolizumab', df['msi_high']==1, 'pembro PFS in MSI-H')
strat_compare('treatment_pembrolizumab', df['msi_high']==0, 'pembro PFS in MSS')

# Encorafenib in BRAF V600E
strat_compare('treatment_encorafenib', df['braf_v600e']==1, 'encora PFS in BRAF V600E')
strat_compare('treatment_encorafenib', df['braf_v600e']==0, 'encora PFS in BRAF WT')

# Trastuzumab/tucatinib in HER2 amplified
strat_compare('treatment_trastuzumab_tucatinib', df['her2_amplified']==1, 'her2 tx PFS in HER2 amp')
strat_compare('treatment_trastuzumab_tucatinib', df['her2_amplified']==0, 'her2 tx PFS in HER2 negative')

# Bevacizumab universal?
strat_compare('treatment_bevacizumab', df['kras_mutation']==1, 'bev PFS in KRAS mut')
strat_compare('treatment_bevacizumab', df['kras_mutation']==0, 'bev PFS in KRAS WT')

# Regorafenib
strat_compare('treatment_regorafenib', df['msi_high']==1, 'rego PFS in MSI-H')
strat_compare('treatment_regorafenib', df['msi_high']==0, 'rego PFS in MSS')

# ============================
# Iteration 16-17: Three-way subgroup definitions for cetuximab
# ============================
print("\n=== Three-way subgroup checks (cetuximab) ===")
# left-sided + KRAS WT + NRAS WT + BRAF WT
mask = (df['right_sided_primary']==0)&(df['kras_mutation']==0)&(df['nras_mutation']==0)&(df['braf_v600e']==0)
strat_compare('treatment_cetuximab', mask, 'cetux: left + KRAS-WT + NRAS-WT + BRAF-WT')
strat_compare('treatment_cetuximab', ~mask, 'cetux: anyone failing left+KRAS-WT+NRAS-WT+BRAF-WT')

# Each violator separately:
m1 = (df['right_sided_primary']==1)&(df['kras_mutation']==0)&(df['nras_mutation']==0)&(df['braf_v600e']==0)
strat_compare('treatment_cetuximab', m1, 'cetux: right + KRAS-WT + NRAS-WT + BRAF-WT')
m2 = (df['right_sided_primary']==0)&(df['kras_mutation']==1)&(df['nras_mutation']==0)&(df['braf_v600e']==0)
strat_compare('treatment_cetuximab', m2, 'cetux: left + KRAS-mut only')
m3 = (df['right_sided_primary']==0)&(df['kras_mutation']==0)&(df['nras_mutation']==1)&(df['braf_v600e']==0)
strat_compare('treatment_cetuximab', m3, 'cetux: left + NRAS-mut only')
m4 = (df['right_sided_primary']==0)&(df['kras_mutation']==0)&(df['nras_mutation']==0)&(df['braf_v600e']==1)
strat_compare('treatment_cetuximab', m4, 'cetux: left + BRAF-mut only')

# Encorafenib + BRAF + maybe MSS-only?
print("\n=== Encorafenib subgroup definitions ===")
strat_compare('treatment_encorafenib', (df['braf_v600e']==1)&(df['msi_high']==0), 'encora: BRAF + MSS')
strat_compare('treatment_encorafenib', (df['braf_v600e']==1)&(df['msi_high']==1), 'encora: BRAF + MSI-H')

# Trastuzumab/tucatinib + HER2 + RAS-WT?
print("\n=== Trastuzumab/tucatinib subgroup definitions ===")
strat_compare('treatment_trastuzumab_tucatinib', (df['her2_amplified']==1)&(df['kras_mutation']==0)&(df['nras_mutation']==0), 'her2-tx: HER2 + RAS-WT')
strat_compare('treatment_trastuzumab_tucatinib', (df['her2_amplified']==1)&((df['kras_mutation']==1)|(df['nras_mutation']==1)), 'her2-tx: HER2 + RAS-mut')

# ============================
# Iteration 18: NTRK fusion (rare biomarker)
# ============================
print("\n=== NTRK ===")
mask_ntrk = df['ntrk_fusion']==1
print(f"  NTRK+ count: {mask_ntrk.sum()}, mean PFS NTRK+: {df.loc[mask_ntrk,'pfs_months'].mean():.3f}, NTRK-: {df.loc[~mask_ntrk,'pfs_months'].mean():.3f}")
m = linreg(df['pfs_months'], df[['ntrk_fusion']], 'ntrk')
report('PFS ~ ntrk_fusion', m.params['ntrk_fusion'], m.pvalues['ntrk_fusion'])

# ============================
# Iteration 19: Systematic interaction screen for each treatment with binary features
# ============================
print("\n=== Systematic interaction screen: each treatment x each binary feature ===")
binary_modifiers = ['sex_female','stage_iv','right_sided_primary','kras_mutation','nras_mutation',
                    'braf_v600e','msi_high','her2_amplified','ntrk_fusion']
for t in treatments:
    for b in binary_modifiers:
        coef, pval, eff_in, eff_out = interaction_test(t, b)
        if pval < 0.01:
            print(f"  *** {t} x {b}: int_coef={coef:.3g}, p={pval:.4g}, eff(b=1)={eff_in:.3g}, eff(b=0)={eff_out:.3g}")
        else:
            print(f"      {t} x {b}: int_coef={coef:.3g}, p={pval:.4g}")
        report(f"INT_{t}_x_{b}", coef, pval)

# ============================
# Iteration 20: Continuous modifier screen for major treatments
# ============================
print("\n=== Continuous modifier screen for treatments ===")
for t in treatments:
    for c in ['ecog_ps','age_years','albumin_g_dl','ldh_u_l','crp_mg_l','nlr','cea_ng_ml','weight_loss_pct_6mo']:
        sub = df.copy()
        sub['ix'] = sub[t]*sub[c]
        X = sub[[t, c, 'ix']]
        m = linreg(sub['pfs_months'], X, '')
        if m.pvalues['ix'] < 0.01:
            print(f"  *** {t} x {c}: int_coef={m.params['ix']:.4g}, p={m.pvalues['ix']:.4g}")
        report(f"INT_{t}_x_{c}", m.params['ix'], m.pvalues['ix'])

# ============================
# Iteration 21: Multivariable adjusted main effect regression with all features
# ============================
print("\n=== Full multivariable PFS model ===")
all_features = adjust_cols + treatments + ['alkaline_phosphatase_u_l','ast_u_l','alt_u_l',
                                           'total_bilirubin_mg_dl','creatinine_mg_dl','bun_mg_dl',
                                           'sodium_meq_l','potassium_meq_l','calcium_mg_dl']
X = df[all_features]
m = linreg(df['pfs_months'], X, 'full')
print(m.summary())

for c in m.params.index:
    if c == 'const':
        continue
    report(f"MV_{c}", m.params[c], m.pvalues[c])

# ============================
# Iteration 22: ECOG and stage as treatment effect modifiers
# ============================
print("\n=== Treatment x ECOG/stage interactions ===")
for t in treatments:
    coef, pval, eff_in, eff_out = interaction_test(t, 'ecog_ps')  # ecog is 0/1/2
    print(f"  {t} x ecog_ps (continuous-like): int_coef={coef:.4g}, p={pval:.4g}")
    report(f"INT_{t}_x_ecog_ps", coef, pval)

# ============================
# Iteration 23: Final composite subgroup tests
# ============================
print("\n=== Final composite subgroup hypotheses ===")
# Cetuximab benefit: left-sided + pan-WT
m1 = mask_left_panwt
strat_compare('treatment_cetuximab', m1, 'FINAL cetux: left + pan-WT')
# Versus complement for confirmation
strat_compare('treatment_cetuximab', ~m1, 'FINAL cetux: complement of (left + pan-WT)')

# Pembrolizumab: MSI-H only
strat_compare('treatment_pembrolizumab', df['msi_high']==1, 'FINAL pembro: MSI-H')
strat_compare('treatment_pembrolizumab', df['msi_high']==0, 'FINAL pembro: MSS')

# Encorafenib: BRAF V600E
strat_compare('treatment_encorafenib', df['braf_v600e']==1, 'FINAL encora: BRAF V600E')
strat_compare('treatment_encorafenib', df['braf_v600e']==0, 'FINAL encora: BRAF WT')

# Trastuzumab/tucatinib: HER2 amplified
strat_compare('treatment_trastuzumab_tucatinib', df['her2_amplified']==1, 'FINAL HER2tx: HER2 amp')
strat_compare('treatment_trastuzumab_tucatinib', df['her2_amplified']==0, 'FINAL HER2tx: HER2 neg')

# Bevacizumab: any heterogeneity?
print("\n=== Bevacizumab subgroup analysis ===")
strat_compare('treatment_bevacizumab', df['stage_iv']==1, 'bev: stage IV')
strat_compare('treatment_bevacizumab', df['stage_iv']==0, 'bev: not stage IV')

# Regorafenib
print("\n=== Regorafenib in different ECOG strata ===")
strat_compare('treatment_regorafenib', df['ecog_ps']==0, 'rego: ECOG 0')
strat_compare('treatment_regorafenib', df['ecog_ps']==1, 'rego: ECOG 1')
strat_compare('treatment_regorafenib', df['ecog_ps']==2, 'rego: ECOG 2')

# Tx interactions among each other
print("\n=== Cetuximab x Bevacizumab interaction (combo therapy?) ===")
sub = df.copy()
sub['ix'] = sub['treatment_cetuximab']*sub['treatment_bevacizumab']
m = linreg(sub['pfs_months'], sub[['treatment_cetuximab','treatment_bevacizumab','ix']], '')
print(f"  cetux*bev coef={m.params['ix']:.4g}, p={m.pvalues['ix']:.4g}")
report('INT_cetux_x_bev', m.params['ix'], m.pvalues['ix'])

# Save raw results
with open('my_results.json','w') as f:
    json.dump({k: v for k, v in results.items()}, f, indent=2, default=str)

print("\nDONE. Results saved to my_results.json")
