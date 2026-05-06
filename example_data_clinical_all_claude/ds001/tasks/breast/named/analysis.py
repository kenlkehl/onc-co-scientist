"""Systematic analysis for ds001_breast: pfs_months outcome."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
results = {}
out = 'pfs_months'


def t_test(col, label=None):
    a = df.loc[df[col] == 1, out]
    b = df.loc[df[col] == 0, out]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        'mean_yes': float(a.mean()),
        'mean_no': float(b.mean()),
        'diff': float(a.mean() - b.mean()),
        'n_yes': int(len(a)),
        'n_no': int(len(b)),
        't': float(t),
        'p': float(p),
    }


def corr(col):
    r, p = stats.pearsonr(df[col], df[out])
    return {'r': float(r), 'p': float(p)}


def anova(col):
    groups = [df.loc[df[col] == k, out] for k in sorted(df[col].unique())]
    F, p = stats.f_oneway(*groups)
    return {'F': float(F), 'p': float(p),
            'means': {int(k): float(df.loc[df[col] == k, out].mean()) for k in sorted(df[col].unique())}}


print('=== ITER 1: Demographic/staging main effects ===')
r1 = {}
r1['age'] = corr('age_years')
r1['sex_female'] = t_test('sex_female')
r1['postmenopausal'] = t_test('postmenopausal')
r1['stage_iv'] = t_test('stage_iv')
r1['has_brain_mets'] = t_test('has_brain_mets')
r1['node_positive'] = t_test('node_positive')
r1['ecog_ps'] = anova('ecog_ps')
results['iter1'] = r1
print(json.dumps(r1, indent=2))

print('\n=== ITER 2: Tumor biology / receptor main effects ===')
r2 = {}
r2['er_positive'] = t_test('er_positive')
r2['pr_positive'] = t_test('pr_positive')
r2['her2_positive'] = t_test('her2_positive')
r2['her2_low'] = t_test('her2_low')
r2['brca1_mutation'] = t_test('brca1_mutation')
r2['brca2_mutation'] = t_test('brca2_mutation')
r2['pik3ca_mutation'] = t_test('pik3ca_mutation')
r2['ki67_pct'] = corr('ki67_pct')
r2['tumor_size_cm'] = corr('tumor_size_cm')
results['iter2'] = r2
print(json.dumps(r2, indent=2))

print('\n=== ITER 3: Labs / cachexia main effects ===')
r3 = {}
for c in ['albumin_g_dl', 'ldh_u_l', 'crp_mg_l', 'nlr', 'weight_loss_pct_6mo',
          'hemoglobin_g_dl', 'alkaline_phosphatase_u_l', 'ast_u_l', 'alt_u_l',
          'total_bilirubin_mg_dl', 'creatinine_mg_dl', 'bun_mg_dl',
          'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl']:
    r3[c] = corr(c)
results['iter3'] = r3
print(json.dumps(r3, indent=2))

print('\n=== ITER 4: Treatment main effects ===')
r4 = {}
for t in ['treatment_tamoxifen', 'treatment_palbociclib', 'treatment_trastuzumab',
          'treatment_olaparib', 'treatment_sacituzumab_govitecan', 'treatment_pembrolizumab']:
    r4[t] = t_test(t)
results['iter4'] = r4
print(json.dumps(r4, indent=2))

with open('analysis_part1.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('\nSaved analysis_part1.json')
