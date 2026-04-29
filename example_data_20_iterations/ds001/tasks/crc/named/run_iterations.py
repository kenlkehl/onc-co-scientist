"""Iterative hypothesis testing on ds001_crc."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
results = {}

def ttest(name, mask):
    a = df.loc[mask, 'pfs_months']
    b = df.loc[~mask, 'pfs_months']
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {'n_a': int(mask.sum()), 'n_b': int((~mask).sum()),
            'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
            'effect': float(a.mean() - b.mean()), 'p': float(p)}

def regress(formula, focus=None):
    m = smf.ols(formula, data=df).fit()
    if focus:
        return {'coef': float(m.params[focus]), 'p': float(m.pvalues[focus]),
                'ci_lo': float(m.conf_int().loc[focus, 0]),
                'ci_hi': float(m.conf_int().loc[focus, 1])}
    return m

def stratified_treatment_effect(tx, subset_mask, label_a, label_b):
    """Effect of treatment within a subset."""
    sub = df.loc[subset_mask]
    a = sub.loc[sub[tx] == 1, 'pfs_months']
    b = sub.loc[sub[tx] == 0, 'pfs_months']
    if len(a) < 5 or len(b) < 5:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {'subset': f'{label_a} vs {label_b}', 'n_tx': int(len(a)), 'n_no_tx': int(len(b)),
            'mean_tx': float(a.mean()), 'mean_no_tx': float(b.mean()),
            'effect': float(a.mean() - b.mean()), 'p': float(p)}

# ITERATION 1: stage_iv, ecog_ps, age main effects
print("=== Iteration 1: Baseline prognostic factors ===")
results['i1'] = {}
results['i1']['stage_iv'] = ttest('stage_iv', df['stage_iv'] == 1)
results['i1']['ecog_ge2'] = ttest('ecog_ge2', df['ecog_ps'] >= 2)
results['i1']['age'] = regress('pfs_months ~ age_years', focus='age_years')
print(results['i1'])

# ITERATION 2: KRAS, NRAS, BRAF main effects on PFS
print("\n=== Iteration 2: Driver mutation main effects ===")
results['i2'] = {}
results['i2']['kras'] = ttest('kras', df['kras_mutation'] == 1)
results['i2']['nras'] = ttest('nras', df['nras_mutation'] == 1)
results['i2']['braf'] = ttest('braf', df['braf_v600e'] == 1)
print(results['i2'])

# ITERATION 3: MSI-high, HER2 amp main effects
print("\n=== Iteration 3: MSI-high, HER2 main effects ===")
results['i3'] = {}
results['i3']['msi'] = ttest('msi', df['msi_high'] == 1)
results['i3']['her2'] = ttest('her2', df['her2_amplified'] == 1)
results['i3']['right_sided'] = ttest('right_sided', df['right_sided_primary'] == 1)
print(results['i3'])

# ITERATION 4: Treatment main effects
print("\n=== Iteration 4: Treatment main effects ===")
results['i4'] = {}
for tx in ['treatment_cetuximab','treatment_bevacizumab','treatment_pembrolizumab',
          'treatment_encorafenib','treatment_trastuzumab_tucatinib','treatment_regorafenib']:
    results['i4'][tx] = ttest(tx, df[tx] == 1)
print(results['i4'])

# ITERATION 5: Cetuximab × KRAS interaction (key biology)
print("\n=== Iteration 5: Cetuximab × KRAS interaction ===")
results['i5'] = {}
m = smf.ols('pfs_months ~ treatment_cetuximab * kras_mutation', data=df).fit()
results['i5']['interaction_coef'] = float(m.params['treatment_cetuximab:kras_mutation'])
results['i5']['interaction_p'] = float(m.pvalues['treatment_cetuximab:kras_mutation'])
results['i5']['cetux_main'] = float(m.params['treatment_cetuximab'])
results['i5']['kras_main'] = float(m.params['kras_mutation'])
results['i5']['cetux_in_kras_wt'] = stratified_treatment_effect('treatment_cetuximab', df['kras_mutation']==0, 'KRAS-WT cetux', 'KRAS-WT no-cetux')
results['i5']['cetux_in_kras_mut'] = stratified_treatment_effect('treatment_cetuximab', df['kras_mutation']==1, 'KRAS-mut cetux', 'KRAS-mut no-cetux')
print(results['i5'])

# ITERATION 6: Cetuximab × NRAS interaction
print("\n=== Iteration 6: Cetuximab × NRAS interaction ===")
results['i6'] = {}
m = smf.ols('pfs_months ~ treatment_cetuximab * nras_mutation', data=df).fit()
results['i6']['interaction_coef'] = float(m.params['treatment_cetuximab:nras_mutation'])
results['i6']['interaction_p'] = float(m.pvalues['treatment_cetuximab:nras_mutation'])
results['i6']['cetux_in_nras_wt'] = stratified_treatment_effect('treatment_cetuximab', df['nras_mutation']==0, 'NRAS-WT cetux', 'NRAS-WT no-cetux')
results['i6']['cetux_in_nras_mut'] = stratified_treatment_effect('treatment_cetuximab', df['nras_mutation']==1, 'NRAS-mut cetux', 'NRAS-mut no-cetux')
print(results['i6'])

# ITERATION 7: Cetuximab × BRAF V600E interaction (cetuximab less effective)
print("\n=== Iteration 7: Cetuximab × BRAF V600E ===")
results['i7'] = {}
m = smf.ols('pfs_months ~ treatment_cetuximab * braf_v600e', data=df).fit()
results['i7']['interaction_coef'] = float(m.params['treatment_cetuximab:braf_v600e'])
results['i7']['interaction_p'] = float(m.pvalues['treatment_cetuximab:braf_v600e'])
results['i7']['cetux_in_braf_wt'] = stratified_treatment_effect('treatment_cetuximab', df['braf_v600e']==0, 'BRAF-WT cetux', 'BRAF-WT no-cetux')
results['i7']['cetux_in_braf_mut'] = stratified_treatment_effect('treatment_cetuximab', df['braf_v600e']==1, 'BRAF-mut cetux', 'BRAF-mut no-cetux')
print(results['i7'])

# ITERATION 8: Cetuximab × right-sided primary
print("\n=== Iteration 8: Cetuximab × right-sided primary ===")
results['i8'] = {}
m = smf.ols('pfs_months ~ treatment_cetuximab * right_sided_primary', data=df).fit()
results['i8']['interaction_coef'] = float(m.params['treatment_cetuximab:right_sided_primary'])
results['i8']['interaction_p'] = float(m.pvalues['treatment_cetuximab:right_sided_primary'])
results['i8']['cetux_in_left'] = stratified_treatment_effect('treatment_cetuximab', df['right_sided_primary']==0, 'left cetux', 'left no-cetux')
results['i8']['cetux_in_right'] = stratified_treatment_effect('treatment_cetuximab', df['right_sided_primary']==1, 'right cetux', 'right no-cetux')
print(results['i8'])

# ITERATION 9: Pembrolizumab × MSI-high interaction (key)
print("\n=== Iteration 9: Pembrolizumab × MSI-high ===")
results['i9'] = {}
m = smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high', data=df).fit()
results['i9']['interaction_coef'] = float(m.params['treatment_pembrolizumab:msi_high'])
results['i9']['interaction_p'] = float(m.pvalues['treatment_pembrolizumab:msi_high'])
results['i9']['pembro_main'] = float(m.params['treatment_pembrolizumab'])
results['i9']['msi_main'] = float(m.params['msi_high'])
results['i9']['pembro_in_mss'] = stratified_treatment_effect('treatment_pembrolizumab', df['msi_high']==0, 'MSS pembro', 'MSS no-pembro')
results['i9']['pembro_in_msi'] = stratified_treatment_effect('treatment_pembrolizumab', df['msi_high']==1, 'MSI-H pembro', 'MSI-H no-pembro')
print(results['i9'])

# ITERATION 10: Encorafenib × BRAF V600E interaction
print("\n=== Iteration 10: Encorafenib × BRAF V600E ===")
results['i10'] = {}
m = smf.ols('pfs_months ~ treatment_encorafenib * braf_v600e', data=df).fit()
results['i10']['interaction_coef'] = float(m.params['treatment_encorafenib:braf_v600e'])
results['i10']['interaction_p'] = float(m.pvalues['treatment_encorafenib:braf_v600e'])
results['i10']['enco_in_braf_wt'] = stratified_treatment_effect('treatment_encorafenib', df['braf_v600e']==0, 'BRAF-WT enco', 'BRAF-WT no-enco')
results['i10']['enco_in_braf_mut'] = stratified_treatment_effect('treatment_encorafenib', df['braf_v600e']==1, 'BRAF-mut enco', 'BRAF-mut no-enco')
print(results['i10'])

# ITERATION 11: Trastuzumab/tucatinib × HER2 amplified
print("\n=== Iteration 11: Tras/tuca × HER2 amplified ===")
results['i11'] = {}
m = smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified', data=df).fit()
key = 'treatment_trastuzumab_tucatinib:her2_amplified'
results['i11']['interaction_coef'] = float(m.params[key])
results['i11']['interaction_p'] = float(m.pvalues[key])
results['i11']['trtu_in_her2_neg'] = stratified_treatment_effect('treatment_trastuzumab_tucatinib', df['her2_amplified']==0, 'HER2- tras', 'HER2- no-tras')
results['i11']['trtu_in_her2_pos'] = stratified_treatment_effect('treatment_trastuzumab_tucatinib', df['her2_amplified']==1, 'HER2+ tras', 'HER2+ no-tras')
print(results['i11'])

# ITERATION 12: Lab markers main effects on PFS
print("\n=== Iteration 12: Lab/biochem prognostic markers ===")
results['i12'] = {}
for col in ['cea_ng_ml','albumin_g_dl','ldh_u_l','crp_mg_l','nlr','weight_loss_pct_6mo','hemoglobin_g_dl']:
    r = regress(f'pfs_months ~ {col}', focus=col)
    results['i12'][col] = r
print(results['i12'])

# ITERATION 13: Metastasis sites
print("\n=== Iteration 13: Metastasis site prognostic ===")
results['i13'] = {}
for col in ['liver_mets','bone_mets','adrenal_mets','pleural_effusion','pericardial_effusion']:
    results['i13'][col] = ttest(col, df[col] == 1)
print(results['i13'])

# ITERATION 14: Comorbidities
print("\n=== Iteration 14: Comorbidities ===")
results['i14'] = {}
for col in ['diabetes_mellitus','hypertension','copd','chronic_kidney_disease','heart_failure','coronary_artery_disease','autoimmune_disease','prior_malignancy']:
    results['i14'][col] = ttest(col, df[col] == 1)
print(results['i14'])

# ITERATION 15: Symptoms (fatigue, pain, dyspnea)
print("\n=== Iteration 15: Symptom grades ===")
results['i15'] = {}
for col in ['fatigue_grade','pain_nrs','dyspnea_grade','cough_grade','appetite_loss_grade']:
    r = regress(f'pfs_months ~ {col}', focus=col)
    results['i15'][col] = r
print(results['i15'])

# ITERATION 16: Sex effect, age × sex
print("\n=== Iteration 16: Sex, race, insurance ===")
results['i16'] = {}
results['i16']['sex_female'] = ttest('sex_female', df['sex_female']==1)
results['i16']['rural'] = ttest('rural', df['rural_residence']==1)
m = smf.ols('pfs_months ~ C(race_ethnicity)', data=df).fit()
results['i16']['race_anova_p'] = float(m.f_pvalue)
m = smf.ols('pfs_months ~ C(insurance_type)', data=df).fit()
results['i16']['insurance_anova_p'] = float(m.f_pvalue)
print(results['i16'])

# ITERATION 17: Prior therapy
print("\n=== Iteration 17: Prior therapy / lines ===")
results['i17'] = {}
for col in ['prior_chemotherapy','prior_radiation','prior_surgery','prior_immunotherapy','prior_targeted_therapy']:
    results['i17'][col] = ttest(col, df[col]==1)
results['i17']['prior_lines'] = regress('pfs_months ~ prior_lines_of_therapy', focus='prior_lines_of_therapy')
results['i17']['years_since_dx'] = regress('pfs_months ~ years_since_diagnosis', focus='years_since_diagnosis')
print(results['i17'])

# ITERATION 18: Bevacizumab × subgroups (any selective benefit?)
print("\n=== Iteration 18: Bevacizumab subgroup effects ===")
results['i18'] = {}
for sub_col, sub_label in [('kras_mutation','KRAS-mut'),('right_sided_primary','right'),('stage_iv','stage IV')]:
    m = smf.ols(f'pfs_months ~ treatment_bevacizumab * {sub_col}', data=df).fit()
    key = f'treatment_bevacizumab:{sub_col}'
    results['i18'][f'bev_x_{sub_col}'] = {'coef': float(m.params[key]), 'p': float(m.pvalues[key])}
print(results['i18'])

# ITERATION 19: Pembrolizumab × MSS / non-MSI
print("\n=== Iteration 19: Pembro effect deeper - by MSI categories ===")
results['i19'] = {}
# MSI-H pembro stratified
mh = df['msi_high']==1
mss = df['msi_high']==0
results['i19']['pembro_mss_only'] = stratified_treatment_effect('treatment_pembrolizumab', mss, 'MSS pembro', 'MSS no-pembro')
results['i19']['pembro_msi_only'] = stratified_treatment_effect('treatment_pembrolizumab', mh, 'MSI pembro', 'MSI no-pembro')
# Combined effect: in MSI-H, pembro should help; in MSS, it should not
print(results['i19'])

# ITERATION 20: Multivariable regression of all key prognostic + treatment factors
print("\n=== Iteration 20: Multivariable PFS model ===")
formula = ('pfs_months ~ age_years + sex_female + ecog_ps + stage_iv + right_sided_primary + '
           'kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified + '
           'cea_ng_ml + albumin_g_dl + ldh_u_l + nlr + crp_mg_l + liver_mets + '
           'treatment_cetuximab + treatment_bevacizumab + treatment_pembrolizumab + '
           'treatment_encorafenib + treatment_trastuzumab_tucatinib + treatment_regorafenib')
m = smf.ols(formula, data=df).fit()
results['i20'] = {'r2': float(m.rsquared), 'n': int(m.nobs)}
results['i20']['coefs'] = {k: {'coef': float(v), 'p': float(m.pvalues[k])} for k,v in m.params.items()}
print('R2:', m.rsquared)
print(m.summary().tables[1])

# ITERATION 21: Multivariable with key interactions
print("\n=== Iteration 21: Multivariable with biomarker × treatment interactions ===")
formula = ('pfs_months ~ age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + cea_ng_ml + '
           'treatment_cetuximab*kras_mutation + treatment_pembrolizumab*msi_high + '
           'treatment_encorafenib*braf_v600e + treatment_trastuzumab_tucatinib*her2_amplified')
m = smf.ols(formula, data=df).fit()
results['i21'] = {'r2': float(m.rsquared)}
key_terms = ['treatment_cetuximab:kras_mutation','treatment_pembrolizumab:msi_high',
             'treatment_encorafenib:braf_v600e','treatment_trastuzumab_tucatinib:her2_amplified']
for k in key_terms:
    results['i21'][k] = {'coef': float(m.params[k]), 'p': float(m.pvalues[k])}
print('R2:', m.rsquared)
for k in key_terms:
    print(k, m.params[k], 'p=', m.pvalues[k])

# ITERATION 22: SNP screening for PFS effect
print("\n=== Iteration 22: SNP screening ===")
snp_cols = [c for c in df.columns if c.startswith('snp_')]
results['i22'] = {}
for s in snp_cols:
    r = regress(f'pfs_months ~ {s}', focus=s)
    results['i22'][s] = r
sig = [(k,v) for k,v in results['i22'].items() if v['p'] < 0.05]
print(f'SNPs with p<0.05: {len(sig)} of {len(snp_cols)}')
for k,v in sorted(sig, key=lambda x: x[1]['p'])[:5]:
    print(' ', k, v)

# ITERATION 23: Older patients subgroup - cetux+KRAS still holds?
print("\n=== Iteration 23: Older subgroup ===")
results['i23'] = {}
older = df['age_years'] >= 70
younger = df['age_years'] < 70
m = smf.ols('pfs_months ~ treatment_cetuximab * kras_mutation', data=df[older]).fit()
results['i23']['cetux_kras_in_older'] = {'coef': float(m.params['treatment_cetuximab:kras_mutation']), 'p': float(m.pvalues['treatment_cetuximab:kras_mutation'])}
m = smf.ols('pfs_months ~ treatment_cetuximab * kras_mutation', data=df[younger]).fit()
results['i23']['cetux_kras_in_younger'] = {'coef': float(m.params['treatment_cetuximab:kras_mutation']), 'p': float(m.pvalues['treatment_cetuximab:kras_mutation'])}
# Also pembro x msi in older
m = smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high', data=df[older]).fit()
results['i23']['pembro_msi_in_older'] = {'coef': float(m.params['treatment_pembrolizumab:msi_high']), 'p': float(m.pvalues['treatment_pembrolizumab:msi_high'])}
print(results['i23'])

# ITERATION 24: Triple effect - Pembro effect in MSI-high stratified by stage IV
print("\n=== Iteration 24: 3-way interactions for pembro/msi by ECOG ===")
results['i24'] = {}
# stratified pembro effect among MSI-high by stage
mh_stg4 = (df['msi_high']==1) & (df['stage_iv']==1)
mh_no4  = (df['msi_high']==1) & (df['stage_iv']==0)
results['i24']['pembro_msi_stg4'] = stratified_treatment_effect('treatment_pembrolizumab', mh_stg4, 'MSI+stIV pembro', 'MSI+stIV no')
results['i24']['pembro_msi_nostg4'] = stratified_treatment_effect('treatment_pembrolizumab', mh_no4, 'MSI+nostIV pembro', 'MSI+nostIV no')
# Cetuximab in left-sided KRAS-WT (best responder per biology)
left_wt = (df['right_sided_primary']==0) & (df['kras_mutation']==0) & (df['nras_mutation']==0) & (df['braf_v600e']==0)
results['i24']['cetux_left_allwt'] = stratified_treatment_effect('treatment_cetuximab', left_wt, 'left+all-WT cetux', 'left+all-WT no-cetux')
right_wt = (df['right_sided_primary']==1) & (df['kras_mutation']==0) & (df['nras_mutation']==0) & (df['braf_v600e']==0)
results['i24']['cetux_right_allwt'] = stratified_treatment_effect('treatment_cetuximab', right_wt, 'right+all-WT cetux', 'right+all-WT no-cetux')
print(results['i24'])

# ITERATION 25: Final summary - check regorafenib for any subgroup, and combinations
print("\n=== Iteration 25: Regorafenib + other ===")
results['i25'] = {}
# regorafenib by prior_lines
for col in ['kras_mutation','braf_v600e','msi_high','prior_chemotherapy']:
    m = smf.ols(f'pfs_months ~ treatment_regorafenib * {col}', data=df).fit()
    key = f'treatment_regorafenib:{col}'
    results['i25'][f'rego_x_{col}'] = {'coef': float(m.params[key]), 'p': float(m.pvalues[key])}
# Also examine triple-WT and BRAF subset
print(results['i25'])

with open('all_results.json','w') as f:
    json.dump(results, f, indent=2, default=str)

print('\nDONE -- results saved to all_results.json')
