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
df['premeno'] = (1 - df['postmenopausal']).astype(int)

results = {}

# Pembro in TNBC by premenopausal
print("=== Pembro: TNBC + premenopausal ===")
for tn in [0, 1]:
    for pm in [0, 1]:
        sub = df[(df['tnbc']==tn) & (df['postmenopausal']==pm)]
        a = sub.loc[sub['treatment_pembrolizumab']==1, 'pfs_months']
        b = sub.loc[sub['treatment_pembrolizumab']==0, 'pfs_months']
        if len(a)<20 or len(b)<20: continue
        ts = stats.ttest_ind(a, b, equal_var=False)
        print(f'  TNBC={tn}, postmeno={pm}: pembro+={a.mean():.3f}(n={len(a)}), -={b.mean():.3f}(n={len(b)}), diff={a.mean()-b.mean():.3f}, p={ts.pvalue:.3g}')

# TNBC + premenopausal as the subgroup
print("\nPembrolizumab in TNBC + premenopausal (adj):")
sub = df[(df['tnbc']==1) & (df['postmenopausal']==0)]
m = smf.ols('pfs_months ~ treatment_pembrolizumab + age_years + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + ki67_pct', data=sub).fit()
print(f'  n={len(sub)}, n_treated={(sub["treatment_pembrolizumab"]==1).sum()}, coef={m.params["treatment_pembrolizumab"]:.4f}, p={m.pvalues["treatment_pembrolizumab"]:.3g}')
results['pembro_TNBC_premeno_adj'] = {
    'coef': float(m.params['treatment_pembrolizumab']),
    'p': float(m.pvalues['treatment_pembrolizumab']),
    'n_treated': int((sub['treatment_pembrolizumab']==1).sum()),
    'n_total': int(len(sub))}

# Three-way interaction test
m = smf.ols('pfs_months ~ treatment_pembrolizumab*tnbc*postmenopausal + age_years + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + ki67_pct', data=df).fit()
key = 'treatment_pembrolizumab:tnbc:postmenopausal'
if key in m.params.index:
    print(f'  3-way interaction pembro:tnbc:postmenopausal: coef={m.params[key]:.4f}, p={m.pvalues[key]:.3g}')
results['pembro_3way_inter'] = {'coef': float(m.params[key]), 'p': float(m.pvalues[key])}

# Sacituzumab brain mets — confirm
print("\n=== Sacituzumab in brain mets ===")
sub = df[df['has_brain_mets']==1]
print(f'  n={len(sub)}, n_treated={(sub["treatment_sacituzumab_govitecan"]==1).sum()}')
m = smf.ols('pfs_months ~ treatment_sacituzumab_govitecan + age_years + ecog_ps + stage_iv + er_positive + her2_positive + albumin_g_dl + ki67_pct + weight_loss_pct_6mo', data=sub).fit()
print(f'  Sacituz adjusted coef={m.params["treatment_sacituzumab_govitecan"]:.4f}, p={m.pvalues["treatment_sacituzumab_govitecan"]:.3g}')

# Interaction in full data
m = smf.ols('pfs_months ~ treatment_sacituzumab_govitecan*has_brain_mets + age_years + ecog_ps + stage_iv + er_positive + her2_positive + albumin_g_dl + ki67_pct', data=df).fit()
key = 'treatment_sacituzumab_govitecan:has_brain_mets'
print(f'  Sacituz x brain interaction: coef={m.params[key]:.4f}, p={m.pvalues[key]:.3g}')
results['sacituz_brain_inter'] = {'coef': float(m.params[key]), 'p': float(m.pvalues[key])}

# Olaparib BRCA + ER-
print("\n=== Olaparib in BRCA+ER- ===")
sub = df[(df['brca_mut']==1) & (df['er_positive']==0)]
m = smf.ols('pfs_months ~ treatment_olaparib + age_years + ecog_ps + stage_iv + has_brain_mets + her2_positive + albumin_g_dl + ki67_pct', data=sub).fit()
print(f'  n={len(sub)}, n_treated={(sub["treatment_olaparib"]==1).sum()}, coef={m.params["treatment_olaparib"]:.4f}, p={m.pvalues["treatment_olaparib"]:.3g}')
results['ola_BRCAposERneg_adj'] = {'coef': float(m.params['treatment_olaparib']), 'p': float(m.pvalues['treatment_olaparib']),
                                    'n_treated': int((sub['treatment_olaparib']==1).sum()), 'n_total': int(len(sub))}

# Three-way interaction
m = smf.ols('pfs_months ~ treatment_olaparib*brca_mut*er_positive + age_years + ecog_ps + stage_iv + has_brain_mets + her2_positive + albumin_g_dl + ki67_pct', data=df).fit()
key = 'treatment_olaparib:brca_mut:er_positive'
if key in m.params.index:
    print(f'  3-way interaction ola:brca:er_positive: coef={m.params[key]:.4f}, p={m.pvalues[key]:.3g}')
    results['ola_3way_inter'] = {'coef': float(m.params[key]), 'p': float(m.pvalues[key])}

# Sacituzumab in brain mets — explore for further refinement
print('\nSacituz in brain mets x ECOG, x ER:')
sub = df[df['has_brain_mets']==1]
for er in [0, 1]:
    s = sub[sub['er_positive']==er]
    a = s.loc[s['treatment_sacituzumab_govitecan']==1, 'pfs_months']
    b = s.loc[s['treatment_sacituzumab_govitecan']==0, 'pfs_months']
    if len(a)<10: continue
    ts = stats.ttest_ind(a, b, equal_var=False)
    print(f'  brain+, ER={er}: SG+={a.mean():.3f}(n={len(a)}), -={b.mean():.3f}(n={len(b)}), diff={a.mean()-b.mean():.3f}, p={ts.pvalue:.3g}')

# Tamoxifen check various subgroups
print('\n=== Tamoxifen — search broader subgroups ===')
for er in [0,1]:
    for h in [0,1]:
        for pm in [0,1]:
            sub = df[(df['er_positive']==er) & (df['her2_positive']==h) & (df['postmenopausal']==pm)]
            a = sub.loc[sub['treatment_tamoxifen']==1, 'pfs_months']
            b = sub.loc[sub['treatment_tamoxifen']==0, 'pfs_months']
            if len(a)<50 or len(b)<50: continue
            ts = stats.ttest_ind(a, b, equal_var=False)
            if ts.pvalue < 0.1:
                print(f'  ER={er},HER2={h},postmeno={pm}: tam+={a.mean():.3f}(n={len(a)}), -={b.mean():.3f}(n={len(b)}), diff={a.mean()-b.mean():.3f}, p={ts.pvalue:.3g}')

# Tamoxifen × postmenopausal
print('\nTamoxifen x postmenopausal:')
m = smf.ols('pfs_months ~ treatment_tamoxifen*postmenopausal + age_years + ecog_ps + stage_iv + er_positive + her2_positive', data=df).fit()
key = 'treatment_tamoxifen:postmenopausal'
print(f'  coef={m.params[key]:.4f}, p={m.pvalues[key]:.3g}')

# Trastuzumab — check by ECOG, brain mets etc
print('\n=== Trastuzumab in HER2+ subset by other features ===')
sub = df[df['her2_positive']==1]
for f in ['er_positive','postmenopausal','stage_iv','has_brain_mets','node_positive','brca_mut','pik3ca_mutation']:
    for v in [0, 1]:
        s = sub[sub[f]==v]
        a = s.loc[s['treatment_trastuzumab']==1, 'pfs_months']
        b = s.loc[s['treatment_trastuzumab']==0, 'pfs_months']
        if len(a)<30 or len(b)<30: continue
        ts = stats.ttest_ind(a, b, equal_var=False)
        if ts.pvalue < 0.1:
            print(f'  HER2+, {f}={v}: tras+={a.mean():.3f}(n={len(a)}), -={b.mean():.3f}(n={len(b)}), diff={a.mean()-b.mean():.3f}, p={ts.pvalue:.3g}')

with open('final_check_results.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print("DONE")
