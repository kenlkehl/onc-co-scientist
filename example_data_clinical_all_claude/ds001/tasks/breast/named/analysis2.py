"""Iter 5-12: treatment x biomarker interactions, multivariable models."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
out = 'pfs_months'
res = {}


def stratified(treatment, modifier):
    """Effect of treatment within modifier=1 vs modifier=0."""
    rows = {}
    for m in [1, 0]:
        sub = df[df[modifier] == m]
        a = sub.loc[sub[treatment] == 1, out]
        b = sub.loc[sub[treatment] == 0, out]
        if len(a) > 0 and len(b) > 0:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows[f'{modifier}={m}'] = {
                'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
                'diff': float(a.mean() - b.mean()),
                'n_tx': int(len(a)), 'n_ctrl': int(len(b)),
                't': float(t), 'p': float(p),
            }
    # Interaction test via OLS
    f = f'{out} ~ {treatment} * {modifier}'
    m = smf.ols(f, data=df).fit()
    inter_term = f'{treatment}:{modifier}'
    rows['interaction'] = {
        'coef': float(m.params[inter_term]),
        'p': float(m.pvalues[inter_term]),
    }
    return rows


print('=== ITER 5: Tamoxifen x ER status, x PR status, x postmenopausal ===')
res['iter5_tamoxifen'] = {
    'x_er': stratified('treatment_tamoxifen', 'er_positive'),
    'x_pr': stratified('treatment_tamoxifen', 'pr_positive'),
    'x_postmeno': stratified('treatment_tamoxifen', 'postmenopausal'),
}
print(json.dumps(res['iter5_tamoxifen'], indent=2))

print('\n=== ITER 6: Palbociclib x ER, x HER2, x postmeno ===')
res['iter6_palbociclib'] = {
    'x_er': stratified('treatment_palbociclib', 'er_positive'),
    'x_her2': stratified('treatment_palbociclib', 'her2_positive'),
    'x_postmeno': stratified('treatment_palbociclib', 'postmenopausal'),
    'x_pik3ca': stratified('treatment_palbociclib', 'pik3ca_mutation'),
}
print(json.dumps(res['iter6_palbociclib'], indent=2))

print('\n=== ITER 7: Trastuzumab x HER2, x HER2-low ===')
res['iter7_trastuzumab'] = {
    'x_her2': stratified('treatment_trastuzumab', 'her2_positive'),
    'x_her2low': stratified('treatment_trastuzumab', 'her2_low'),
    'x_er': stratified('treatment_trastuzumab', 'er_positive'),
}
print(json.dumps(res['iter7_trastuzumab'], indent=2))

print('\n=== ITER 8: Olaparib x BRCA1, x BRCA2, x BRCA-any ===')
df['brca_any'] = ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)).astype(int)
res['iter8_olaparib'] = {
    'x_brca1': stratified('treatment_olaparib', 'brca1_mutation'),
    'x_brca2': stratified('treatment_olaparib', 'brca2_mutation'),
    'x_brca_any': stratified('treatment_olaparib', 'brca_any'),
    'x_her2': stratified('treatment_olaparib', 'her2_positive'),
    'x_er': stratified('treatment_olaparib', 'er_positive'),
}
print(json.dumps(res['iter8_olaparib'], indent=2))

print('\n=== ITER 9: Sacituzumab x HER2, HER2-low, ER, triple-neg ===')
df['triple_neg'] = ((df['er_positive'] == 0) & (df['pr_positive'] == 0) & (df['her2_positive'] == 0)).astype(int)
res['iter9_sacituzumab'] = {
    'x_her2': stratified('treatment_sacituzumab_govitecan', 'her2_positive'),
    'x_her2low': stratified('treatment_sacituzumab_govitecan', 'her2_low'),
    'x_er': stratified('treatment_sacituzumab_govitecan', 'er_positive'),
    'x_tnbc': stratified('treatment_sacituzumab_govitecan', 'triple_neg'),
}
print(json.dumps(res['iter9_sacituzumab'], indent=2))

print('\n=== ITER 10: Pembrolizumab x TNBC, ki67-high, PIK3CA, HER2 ===')
df['ki67_high'] = (df['ki67_pct'] >= 20).astype(int)
res['iter10_pembrolizumab'] = {
    'x_tnbc': stratified('treatment_pembrolizumab', 'triple_neg'),
    'x_ki67_high': stratified('treatment_pembrolizumab', 'ki67_high'),
    'x_pik3ca': stratified('treatment_pembrolizumab', 'pik3ca_mutation'),
    'x_her2': stratified('treatment_pembrolizumab', 'her2_positive'),
    'x_er': stratified('treatment_pembrolizumab', 'er_positive'),
    'x_pdl1_proxy_high_nlr': stratified('treatment_pembrolizumab', 'stage_iv'),
}
print(json.dumps(res['iter10_pembrolizumab'], indent=2))

print('\n=== ITER 11: Multivariable OLS — full prognostic model ===')
covars = [
    'age_years', 'sex_female', 'ecog_ps', 'stage_iv', 'has_brain_mets',
    'node_positive', 'postmenopausal', 'er_positive', 'pr_positive',
    'her2_positive', 'her2_low', 'brca1_mutation', 'brca2_mutation',
    'pik3ca_mutation', 'ki67_pct', 'tumor_size_cm', 'albumin_g_dl',
    'ldh_u_l', 'weight_loss_pct_6mo', 'crp_mg_l', 'nlr',
    'treatment_tamoxifen', 'treatment_palbociclib', 'treatment_trastuzumab',
    'treatment_olaparib', 'treatment_sacituzumab_govitecan',
    'treatment_pembrolizumab',
]
formula = f'{out} ~ ' + ' + '.join(covars)
m11 = smf.ols(formula, data=df).fit()
res['iter11_full_model'] = {
    'r_squared': float(m11.rsquared),
    'coefs': {k: {'coef': float(v), 'p': float(m11.pvalues[k])} for k, v in m11.params.items()},
}
print(f"R^2 = {m11.rsquared:.4f}")
for k, v in m11.params.items():
    print(f'  {k:35s} {v:+9.4f}  p={m11.pvalues[k]:.3g}')

print('\n=== ITER 12: Joint mechanism-specific subgroup confirmations ===')
# tamoxifen in ER+/postmeno
sub = df[(df['er_positive'] == 1) & (df['postmenopausal'] == 1)]
a = sub.loc[sub['treatment_tamoxifen'] == 1, out]
b = sub.loc[sub['treatment_tamoxifen'] == 0, out]
t, p = stats.ttest_ind(a, b, equal_var=False)
res['iter12_tamox_er_postmeno'] = {
    'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
    'diff': float(a.mean() - b.mean()), 'n_tx': int(len(a)), 'n_ctrl': int(len(b)),
    'p': float(p),
}
# palbo in ER+/HER2-/postmeno
sub = df[(df['er_positive'] == 1) & (df['her2_positive'] == 0) & (df['postmenopausal'] == 1)]
a = sub.loc[sub['treatment_palbociclib'] == 1, out]
b = sub.loc[sub['treatment_palbociclib'] == 0, out]
t, p = stats.ttest_ind(a, b, equal_var=False)
res['iter12_palbo_erpos_her2neg_postmeno'] = {
    'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
    'diff': float(a.mean() - b.mean()), 'n_tx': int(len(a)), 'n_ctrl': int(len(b)),
    'p': float(p),
}
# trastuzumab in HER2+
sub = df[df['her2_positive'] == 1]
a = sub.loc[sub['treatment_trastuzumab'] == 1, out]
b = sub.loc[sub['treatment_trastuzumab'] == 0, out]
t, p = stats.ttest_ind(a, b, equal_var=False)
res['iter12_trast_her2pos'] = {
    'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
    'diff': float(a.mean() - b.mean()), 'n_tx': int(len(a)), 'n_ctrl': int(len(b)),
    'p': float(p),
}
# olaparib in BRCA-any
sub = df[df['brca_any'] == 1]
a = sub.loc[sub['treatment_olaparib'] == 1, out]
b = sub.loc[sub['treatment_olaparib'] == 0, out]
t, p = stats.ttest_ind(a, b, equal_var=False)
res['iter12_olap_brca_any'] = {
    'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
    'diff': float(a.mean() - b.mean()), 'n_tx': int(len(a)), 'n_ctrl': int(len(b)),
    'p': float(p),
}
# pembro in TNBC
sub = df[df['triple_neg'] == 1]
a = sub.loc[sub['treatment_pembrolizumab'] == 1, out]
b = sub.loc[sub['treatment_pembrolizumab'] == 0, out]
t, p = stats.ttest_ind(a, b, equal_var=False)
res['iter12_pembro_tnbc'] = {
    'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
    'diff': float(a.mean() - b.mean()), 'n_tx': int(len(a)), 'n_ctrl': int(len(b)),
    'p': float(p),
}
# sacituzumab in TNBC
sub = df[df['triple_neg'] == 1]
a = sub.loc[sub['treatment_sacituzumab_govitecan'] == 1, out]
b = sub.loc[sub['treatment_sacituzumab_govitecan'] == 0, out]
t, p = stats.ttest_ind(a, b, equal_var=False)
res['iter12_sacituzumab_tnbc'] = {
    'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
    'diff': float(a.mean() - b.mean()), 'n_tx': int(len(a)), 'n_ctrl': int(len(b)),
    'p': float(p),
}

print(json.dumps({k: v for k, v in res.items() if 'iter12' in k}, indent=2))

with open('analysis_part2.json', 'w') as f:
    json.dump(res, f, indent=2, default=str)
print('Saved analysis_part2.json')
