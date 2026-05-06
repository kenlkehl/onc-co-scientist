"""Iterations 1-5: baseline analyses."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet('dataset.parquet')

results = {}

# ---------- Iteration 1: Main effects of treatments on PFS ----------
treatments = [
    'treatment_tamoxifen', 'treatment_palbociclib', 'treatment_trastuzumab',
    'treatment_olaparib', 'treatment_sacituzumab_govitecan', 'treatment_pembrolizumab'
]
iter1 = {}
for t in treatments:
    yes = df.loc[df[t] == 1, 'pfs_months']
    no = df.loc[df[t] == 0, 'pfs_months']
    tt = stats.ttest_ind(yes, no, equal_var=False)
    iter1[t] = {
        'mean_on': float(yes.mean()),
        'mean_off': float(no.mean()),
        'diff': float(yes.mean() - no.mean()),
        'n_on': int(len(yes)), 'n_off': int(len(no)),
        'p_value': float(tt.pvalue), 't_stat': float(tt.statistic),
    }
results['iter1_treatment_main_effects'] = iter1

# ---------- Iteration 2: Continuous features vs PFS (Pearson r) ----------
cont_feats = ['age_years', 'ki67_pct', 'tumor_size_cm', 'albumin_g_dl', 'ldh_u_l',
              'weight_loss_pct_6mo', 'crp_mg_l', 'nlr', 'hemoglobin_g_dl',
              'alkaline_phosphatase_u_l', 'ast_u_l', 'alt_u_l',
              'total_bilirubin_mg_dl', 'creatinine_mg_dl', 'bun_mg_dl',
              'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl']
iter2 = {}
for f in cont_feats:
    r, p = stats.pearsonr(df[f], df['pfs_months'])
    iter2[f] = {'pearson_r': float(r), 'p_value': float(p)}
results['iter2_continuous_main_effects'] = iter2

# ---------- Iteration 3: Binary features vs PFS (mean diff) ----------
bin_feats = ['sex_female', 'stage_iv', 'has_brain_mets', 'node_positive',
             'postmenopausal', 'er_positive', 'pr_positive', 'her2_positive',
             'her2_low', 'brca1_mutation', 'brca2_mutation', 'pik3ca_mutation']
iter3 = {}
for f in bin_feats:
    yes = df.loc[df[f] == 1, 'pfs_months']
    no = df.loc[df[f] == 0, 'pfs_months']
    tt = stats.ttest_ind(yes, no, equal_var=False)
    iter3[f] = {
        'mean_yes': float(yes.mean()), 'mean_no': float(no.mean()),
        'diff': float(yes.mean() - no.mean()),
        'p_value': float(tt.pvalue), 't_stat': float(tt.statistic),
        'n_yes': int(len(yes)), 'n_no': int(len(no)),
    }
# Also test ECOG (ordinal 0/1/2)
ecog_groups = [df.loc[df['ecog_ps'] == k, 'pfs_months'] for k in [0, 1, 2]]
ecog_anova = stats.f_oneway(*ecog_groups)
iter3['ecog_ps_anova'] = {
    'mean_0': float(ecog_groups[0].mean()),
    'mean_1': float(ecog_groups[1].mean()),
    'mean_2': float(ecog_groups[2].mean()),
    'p_value': float(ecog_anova.pvalue),
    'f_stat': float(ecog_anova.statistic),
}
results['iter3_binary_main_effects'] = iter3

# ---------- Iteration 4: Multivariable OLS for PFS ----------
all_feats = cont_feats + bin_feats + ['ecog_ps'] + treatments
X = df[all_feats].copy()
X = sm.add_constant(X)
y = df['pfs_months']
model = sm.OLS(y, X).fit()
iter4 = {
    'r_squared': float(model.rsquared),
    'adj_r_squared': float(model.rsquared_adj),
    'n': int(model.nobs),
    'coefficients': {name: {'coef': float(model.params[name]),
                            'se': float(model.bse[name]),
                            'p_value': float(model.pvalues[name])}
                     for name in model.params.index},
}
results['iter4_multivariable_ols'] = iter4

# ---------- Iteration 5: Pre-specified treatment-biomarker pairings ----------
pairs = [
    ('treatment_tamoxifen', 'er_positive'),
    ('treatment_tamoxifen', 'postmenopausal'),
    ('treatment_palbociclib', 'er_positive'),
    ('treatment_trastuzumab', 'her2_positive'),
    ('treatment_olaparib', 'brca1_mutation'),
    ('treatment_olaparib', 'brca2_mutation'),
    ('treatment_sacituzumab_govitecan', 'her2_low'),
    ('treatment_pembrolizumab', 'has_brain_mets'),
]
iter5 = {}
for treat, marker in pairs:
    sub_pos = df[df[marker] == 1]
    sub_neg = df[df[marker] == 0]
    eff_pos = sub_pos.loc[sub_pos[treat] == 1, 'pfs_months'].mean() - sub_pos.loc[sub_pos[treat] == 0, 'pfs_months'].mean()
    eff_neg = sub_neg.loc[sub_neg[treat] == 1, 'pfs_months'].mean() - sub_neg.loc[sub_neg[treat] == 0, 'pfs_months'].mean()
    # Interaction test via OLS
    sub = df[[treat, marker, 'pfs_months']].copy()
    sub['interaction'] = sub[treat] * sub[marker]
    Xi = sm.add_constant(sub[[treat, marker, 'interaction']])
    m = sm.OLS(sub['pfs_months'], Xi).fit()
    iter5[f'{treat}_x_{marker}'] = {
        'effect_in_marker_pos': float(eff_pos),
        'effect_in_marker_neg': float(eff_neg),
        'interaction_coef': float(m.params['interaction']),
        'interaction_p': float(m.pvalues['interaction']),
        'treat_main_in_full': float(m.params[treat]),
        'treat_main_p': float(m.pvalues[treat]),
        'n_marker_pos': int(len(sub_pos)),
        'n_marker_neg': int(len(sub_neg)),
    }
results['iter5_treatment_biomarker_pairings'] = iter5

with open('iters_1_5_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("Done iters 1-5")
print(json.dumps(results, indent=2, default=str)[:5000])
