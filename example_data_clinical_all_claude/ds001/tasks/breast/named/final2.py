import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

df = pd.read_parquet('dataset.parquet')
df['hr_pos_her2_neg'] = (((df['er_positive']==1) | (df['pr_positive']==1)) & (df['her2_positive']==0)).astype(int)
df['brca_mut'] = ((df['brca1_mutation']==1) | (df['brca2_mutation']==1)).astype(int)
df['tnbc'] = ((df['er_positive']==0) & (df['pr_positive']==0) & (df['her2_positive']==0)).astype(int)

results = {}

# Olaparib in BRCA+/node+ — unadjusted very promising
print("=== Olaparib in BRCA+/node+ ===")
sub = df[(df['brca_mut']==1) & (df['node_positive']==1)]
m = smf.ols('pfs_months ~ treatment_olaparib + age_years + ecog_ps + stage_iv + has_brain_mets + er_positive + her2_positive + albumin_g_dl + ki67_pct', data=sub).fit()
print(f'  n={len(sub)}, n_treated={(sub["treatment_olaparib"]==1).sum()}, coef={m.params["treatment_olaparib"]:.4f}, p={m.pvalues["treatment_olaparib"]:.3g}')
results['ola_BRCAposNodepos_adj'] = {'coef': float(m.params['treatment_olaparib']), 'p': float(m.pvalues['treatment_olaparib']),
                                       'n_treated': int((sub['treatment_olaparib']==1).sum()), 'n_total': int(len(sub))}

# Pure unadjusted Olaparib in BRCA mut
print("\nOlaparib in BRCA mut (unadjusted simple t-test, no covariates):")
sub = df[df['brca_mut']==1]
a = sub.loc[sub['treatment_olaparib']==1, 'pfs_months']
b = sub.loc[sub['treatment_olaparib']==0, 'pfs_months']
ts = stats.ttest_ind(a, b, equal_var=False)
print(f'  n_treated={len(a)}, n_control={len(b)}, ola+={a.mean():.3f}, ola-={b.mean():.3f}, diff={a.mean()-b.mean():.3f}, p={ts.pvalue:.3g}')

# Test interaction: olaparib*brca_mut in full data with rich adjustment
print("\nOlaparib x BRCA full-data interaction (richly adjusted):")
m = smf.ols('pfs_months ~ treatment_olaparib*brca_mut + age_years + ecog_ps + stage_iv + has_brain_mets + er_positive + pr_positive + her2_positive + pik3ca_mutation + albumin_g_dl + ki67_pct + weight_loss_pct_6mo + tumor_size_cm', data=df).fit()
key = 'treatment_olaparib:brca_mut'
print(f'  inter coef={m.params[key]:.4f}, p={m.pvalues[key]:.3g}')
results['ola_brca_inter_richadj'] = {'coef': float(m.params[key]), 'p': float(m.pvalues[key])}

# Tamoxifen: search ER+ in many subgroups, including continuous moderators
print("\n=== Tamoxifen × continuous in ER+ ===")
sub = df[df['er_positive']==1]
for cont in ['age_years','ki67_pct','tumor_size_cm','albumin_g_dl','crp_mg_l','nlr','weight_loss_pct_6mo','ldh_u_l']:
    formula = 'pfs_months ~ treatment_tamoxifen*' + cont + ' + age_years + ecog_ps + stage_iv + her2_positive + pik3ca_mutation'
    m = smf.ols(formula, data=sub).fit()
    key = 'treatment_tamoxifen:' + cont
    if key in m.params.index:
        p = float(m.pvalues[key])
        if p < 0.1:
            print(f'  ER+, tam x {cont}: coef={m.params[key]:.5f}, p={p:.3g}')

# Sacituzumab in brain mets adjusted - confirmed; refine: brain + ER+
print("\nSacituz: brain + ER+:")
sub = df[(df['has_brain_mets']==1) & (df['er_positive']==1)]
m = smf.ols('pfs_months ~ treatment_sacituzumab_govitecan + age_years + ecog_ps + stage_iv + her2_positive + albumin_g_dl + ki67_pct', data=sub).fit()
print(f'  n={len(sub)}, n_treated={(sub["treatment_sacituzumab_govitecan"]==1).sum()}, coef={m.params["treatment_sacituzumab_govitecan"]:.4f}, p={m.pvalues["treatment_sacituzumab_govitecan"]:.3g}')
results['sacituz_brain_ER_adj'] = {'coef': float(m.params['treatment_sacituzumab_govitecan']), 'p': float(m.pvalues['treatment_sacituzumab_govitecan']),
                                    'n_treated': int((sub['treatment_sacituzumab_govitecan']==1).sum()), 'n_total': int(len(sub))}

# Sacituz: brain + ER+/HER2- specifically
print("Sacituz: brain + ER+/HER2-:")
sub = df[(df['has_brain_mets']==1) & (df['er_positive']==1) & (df['her2_positive']==0)]
m = smf.ols('pfs_months ~ treatment_sacituzumab_govitecan + age_years + ecog_ps + stage_iv + albumin_g_dl + ki67_pct', data=sub).fit()
print(f'  n={len(sub)}, n_treated={(sub["treatment_sacituzumab_govitecan"]==1).sum()}, coef={m.params["treatment_sacituzumab_govitecan"]:.4f}, p={m.pvalues["treatment_sacituzumab_govitecan"]:.3g}')

# 3-way interaction: sacituz * brain_mets * er
m = smf.ols('pfs_months ~ treatment_sacituzumab_govitecan*has_brain_mets*er_positive + age_years + ecog_ps + stage_iv + her2_positive + pik3ca_mutation + albumin_g_dl + ki67_pct', data=df).fit()
key = 'treatment_sacituzumab_govitecan:has_brain_mets:er_positive'
if key in m.params.index:
    print(f'  3-way sacituz:brain:er coef={m.params[key]:.4f}, p={m.pvalues[key]:.3g}')

# Sex effect with treatments — check tamoxifen x sex
print("\nTamoxifen x sex_female:")
m = smf.ols('pfs_months ~ treatment_tamoxifen*sex_female + age_years + ecog_ps + stage_iv + er_positive + her2_positive + pik3ca_mutation', data=df).fit()
key = 'treatment_tamoxifen:sex_female'
print(f'  coef={m.params[key]:.4f}, p={m.pvalues[key]:.3g}')

# Sex distribution by treatment
print("\nSex distribution by treatment:")
for t in ['treatment_tamoxifen','treatment_palbociclib','treatment_pembrolizumab','treatment_olaparib','treatment_sacituzumab_govitecan','treatment_trastuzumab']:
    print(f'  {t}: n_female={((df[t]==1) & (df["sex_female"]==1)).sum()} / n_total={(df[t]==1).sum()}')

# Final palbociclib confirmation: 4-way (ER+/HER2-/PIK3CA-wt/postmenopausal)
print("\n=== Palbo refined: ER+/HER2-/PIK3CA-wt subgroups ===")
sub_base = df[(df['er_positive']==1) & (df['her2_positive']==0) & (df['pik3ca_mutation']==0)]
for f in ['postmenopausal','sex_female','node_positive','stage_iv','has_brain_mets','brca_mut']:
    for v in [0, 1]:
        s = sub_base[sub_base[f]==v]
        a = s.loc[s['treatment_palbociclib']==1, 'pfs_months']
        b = s.loc[s['treatment_palbociclib']==0, 'pfs_months']
        if len(a)<30 or len(b)<30: continue
        ts = stats.ttest_ind(a, b, equal_var=False)
        print(f'  ER+/HER2-/PIK3CA-wt, {f}={v}: palbo+={a.mean():.3f}(n={len(a)}), -={b.mean():.3f}(n={len(b)}), diff={a.mean()-b.mean():.3f}, p={ts.pvalue:.3g}')

with open('final2_results.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print("DONE")
