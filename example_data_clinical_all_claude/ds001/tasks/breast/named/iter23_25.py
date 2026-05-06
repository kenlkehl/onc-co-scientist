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

# Iter 23: Pembro deep search in TNBC
print("=== Iter 23: Pembro x TNBC modifiers ===")
sub = df[df['tnbc']==1]
print(f'TNBC subset n={len(sub)}, pembro n={(sub["treatment_pembrolizumab"]==1).sum()}')
for mod in ['stage_iv','has_brain_mets','node_positive','postmenopausal','sex_female','brca_mut','pik3ca_mutation']:
    formula = 'pfs_months ~ treatment_pembrolizumab*' + mod + ' + age_years + ecog_ps + albumin_g_dl'
    m = smf.ols(formula, data=sub).fit()
    key = 'treatment_pembrolizumab:' + mod
    if key in m.params.index:
        print(f'  TNBC+{mod}: pembro main coef={m.params["treatment_pembrolizumab"]:.4f} p={m.pvalues["treatment_pembrolizumab"]:.3g}, inter={m.params[key]:.4f} p={m.pvalues[key]:.3g}')

# Pembro x continuous in TNBC
print("\nPembro x continuous in TNBC:")
for cont in ['age_years','ki67_pct','tumor_size_cm','albumin_g_dl','crp_mg_l','nlr','weight_loss_pct_6mo','ldh_u_l','hemoglobin_g_dl']:
    formula = 'pfs_months ~ treatment_pembrolizumab*' + cont + ' + age_years + ecog_ps + stage_iv'
    m = smf.ols(formula, data=sub).fit()
    key = 'treatment_pembrolizumab:' + cont
    if key in m.params.index:
        p = float(m.pvalues[key])
        if p < 0.1:
            print(f'  TNBC, pembro x {cont}: coef={m.params[key]:.5f}, p={p:.3g}')

# Iter 24: Olaparib deep dive
print("\n=== Iter 24: Olaparib x BRCA modifiers ===")
sub = df[df['brca_mut']==1]
print(f'BRCA mut subset n={len(sub)}, olaparib n={(sub["treatment_olaparib"]==1).sum()}')
for mod in ['her2_positive','er_positive','tnbc','stage_iv','has_brain_mets','node_positive']:
    a = sub.loc[(sub[mod]==0) & (sub['treatment_olaparib']==1), 'pfs_months']
    b = sub.loc[(sub[mod]==0) & (sub['treatment_olaparib']==0), 'pfs_months']
    if len(a)>=10 and len(b)>=10:
        ts = stats.ttest_ind(a, b, equal_var=False)
        print(f'  BRCA+, {mod}=0: ola+={a.mean():.3f}(n={len(a)}), -={b.mean():.3f}(n={len(b)}), diff={a.mean()-b.mean():.3f}, p={ts.pvalue:.3g}')
    a = sub.loc[(sub[mod]==1) & (sub['treatment_olaparib']==1), 'pfs_months']
    b = sub.loc[(sub[mod]==1) & (sub['treatment_olaparib']==0), 'pfs_months']
    if len(a)>=10 and len(b)>=10:
        ts = stats.ttest_ind(a, b, equal_var=False)
        print(f'  BRCA+, {mod}=1: ola+={a.mean():.3f}(n={len(a)}), -={b.mean():.3f}(n={len(b)}), diff={a.mean()-b.mean():.3f}, p={ts.pvalue:.3g}')

# Iter 25: Final adjusted models
print("\n=== Iter 25: Final adjusted models ===")
# Palbociclib in ER+/HER2-/PIK3CA-wt
sub = df[(df['er_positive']==1) & (df['her2_positive']==0) & (df['pik3ca_mutation']==0)]
m = smf.ols('pfs_months ~ treatment_palbociclib + age_years + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + ki67_pct + weight_loss_pct_6mo', data=sub).fit()
results['palbo_ERposHER2negPIK3CAwt_adj'] = {'coef': float(m.params['treatment_palbociclib']), 'p': float(m.pvalues['treatment_palbociclib']), 'n_treated': int((sub['treatment_palbociclib']==1).sum()), 'n_total': int(len(sub))}
print(f'Palbo in ER+/HER2-/PIK3CA-wt (adj): coef={m.params["treatment_palbociclib"]:.4f}, p={m.pvalues["treatment_palbociclib"]:.3g}')

# Compare ER+/HER2-/PIK3CA-mut: palbo NOT effective
sub_mut = df[(df['er_positive']==1) & (df['her2_positive']==0) & (df['pik3ca_mutation']==1)]
m2 = smf.ols('pfs_months ~ treatment_palbociclib + age_years + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + ki67_pct + weight_loss_pct_6mo', data=sub_mut).fit()
results['palbo_ERposHER2negPIK3CAmut_adj'] = {'coef': float(m2.params['treatment_palbociclib']), 'p': float(m2.pvalues['treatment_palbociclib']), 'n_treated': int((sub_mut['treatment_palbociclib']==1).sum())}
print(f'Palbo in ER+/HER2-/PIK3CA-mut (adj): coef={m2.params["treatment_palbociclib"]:.4f}, p={m2.pvalues["treatment_palbociclib"]:.3g}')

# Trastuzumab in HER2+
sub = df[df['her2_positive']==1]
m = smf.ols('pfs_months ~ treatment_trastuzumab + age_years + ecog_ps + stage_iv + has_brain_mets + er_positive + albumin_g_dl + ki67_pct', data=sub).fit()
results['tras_HER2pos_adj'] = {'coef': float(m.params['treatment_trastuzumab']), 'p': float(m.pvalues['treatment_trastuzumab'])}
print(f'Trastuzumab in HER2+ (adj): coef={m.params["treatment_trastuzumab"]:.4f}, p={m.pvalues["treatment_trastuzumab"]:.3g}')

# Olaparib in BRCA mutated
sub = df[df['brca_mut']==1]
m = smf.ols('pfs_months ~ treatment_olaparib + age_years + ecog_ps + stage_iv + has_brain_mets + er_positive + her2_positive + albumin_g_dl + ki67_pct', data=sub).fit()
results['ola_BRCApos_adj'] = {'coef': float(m.params['treatment_olaparib']), 'p': float(m.pvalues['treatment_olaparib']), 'n_treated': int((sub['treatment_olaparib']==1).sum()), 'n_total': int(len(sub))}
print(f'Olaparib in BRCA mutated (adj): coef={m.params["treatment_olaparib"]:.4f}, p={m.pvalues["treatment_olaparib"]:.3g}')

# Olaparib in BRCA mutated AND HER2-
sub = df[(df['brca_mut']==1) & (df['her2_positive']==0)]
m = smf.ols('pfs_months ~ treatment_olaparib + age_years + ecog_ps + stage_iv + has_brain_mets + er_positive + albumin_g_dl + ki67_pct', data=sub).fit()
results['ola_BRCAposHER2neg_adj'] = {'coef': float(m.params['treatment_olaparib']), 'p': float(m.pvalues['treatment_olaparib']), 'n_treated': int((sub['treatment_olaparib']==1).sum()), 'n_total': int(len(sub))}
print(f'Olaparib in BRCA mut + HER2- (adj): coef={m.params["treatment_olaparib"]:.4f}, p={m.pvalues["treatment_olaparib"]:.3g}')

# Pembrolizumab in TNBC
sub = df[df['tnbc']==1]
m = smf.ols('pfs_months ~ treatment_pembrolizumab + age_years + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + ki67_pct', data=sub).fit()
results['pembro_TNBC_adj'] = {'coef': float(m.params['treatment_pembrolizumab']), 'p': float(m.pvalues['treatment_pembrolizumab']), 'n_treated': int((sub['treatment_pembrolizumab']==1).sum()), 'n_total': int(len(sub))}
print(f'Pembrolizumab in TNBC (adj): coef={m.params["treatment_pembrolizumab"]:.4f}, p={m.pvalues["treatment_pembrolizumab"]:.3g}')

# Sacituzumab in brain mets
sub = df[df['has_brain_mets']==1]
m = smf.ols('pfs_months ~ treatment_sacituzumab_govitecan + age_years + ecog_ps + stage_iv + er_positive + her2_positive + albumin_g_dl + ki67_pct', data=sub).fit()
results['sacituz_brain_adj'] = {'coef': float(m.params['treatment_sacituzumab_govitecan']), 'p': float(m.pvalues['treatment_sacituzumab_govitecan']), 'n_treated': int((sub['treatment_sacituzumab_govitecan']==1).sum())}
print(f'Sacituz in brain mets (adj): coef={m.params["treatment_sacituzumab_govitecan"]:.4f}, p={m.pvalues["treatment_sacituzumab_govitecan"]:.3g}')

# Tamoxifen in ER+
sub = df[df['er_positive']==1]
m = smf.ols('pfs_months ~ treatment_tamoxifen + age_years + ecog_ps + stage_iv + has_brain_mets + her2_positive + albumin_g_dl + ki67_pct', data=sub).fit()
results['tam_ERpos_adj'] = {'coef': float(m.params['treatment_tamoxifen']), 'p': float(m.pvalues['treatment_tamoxifen'])}
print(f'Tamoxifen in ER+ (adj): coef={m.params["treatment_tamoxifen"]:.4f}, p={m.pvalues["treatment_tamoxifen"]:.3g}')

with open('iter23_25_results.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print("DONE")
