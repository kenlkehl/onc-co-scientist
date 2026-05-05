"""Full analysis to be saved to results.json. We run all iterations end-to-end."""
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

def linreg(formula, data=df, label=None):
    m = smf.ols(formula, data=data).fit()
    return m

def basic_t(a, b):
    res = stats.ttest_ind(a, b, equal_var=False)
    return res.statistic, res.pvalue, a.mean() - b.mean()


# =========================================================
# ITERATION 1: distributions and PFS overall + correlations
# =========================================================
out = {}
out['pfs_mean'] = float(df['pfs_months'].mean())
out['pfs_median'] = float(df['pfs_months'].median())
out['pfs_std'] = float(df['pfs_months'].std())

# Correlations of continuous vars with pfs_months
cont = ['age_years', 'ki67_pct', 'tumor_size_cm', 'albumin_g_dl', 'ldh_u_l',
        'weight_loss_pct_6mo', 'crp_mg_l', 'nlr', 'hemoglobin_g_dl',
        'alkaline_phosphatase_u_l', 'ast_u_l', 'alt_u_l', 'total_bilirubin_mg_dl',
        'creatinine_mg_dl', 'bun_mg_dl', 'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl']
out['continuous_corrs'] = {}
for c in cont:
    r, p = stats.pearsonr(df[c], df['pfs_months'])
    out['continuous_corrs'][c] = {'r': float(r), 'p': float(p)}

# Binary features main effects
bins = ['sex_female', 'stage_iv', 'has_brain_mets', 'node_positive',
        'postmenopausal', 'er_positive', 'pr_positive', 'her2_positive',
        'her2_low', 'brca1_mutation', 'brca2_mutation', 'pik3ca_mutation',
        'treatment_tamoxifen', 'treatment_palbociclib', 'treatment_trastuzumab',
        'treatment_olaparib', 'treatment_sacituzumab_govitecan', 'treatment_pembrolizumab']
out['binary_main_effects'] = {}
for b in bins:
    a = df.loc[df[b] == 1, 'pfs_months']
    nz = df.loc[df[b] == 0, 'pfs_months']
    t, p, diff = basic_t(a, nz)
    out['binary_main_effects'][b] = {
        'mean_pos': float(a.mean()), 'mean_neg': float(nz.mean()),
        'diff': float(diff), 'p': float(p), 'n_pos': int(len(a)), 'n_neg': int(len(nz))
    }

# ECOG PS as ordinal
ecog_means = df.groupby('ecog_ps')['pfs_months'].mean().to_dict()
ecog_corr = stats.pearsonr(df['ecog_ps'], df['pfs_months'])
out['ecog_means'] = {int(k): float(v) for k, v in ecog_means.items()}
out['ecog_pearson'] = {'r': float(ecog_corr[0]), 'p': float(ecog_corr[1])}

results['iter01_distributions_main_effects'] = out

# =========================================================
# ITERATION 2: regress pfs on all features (multivariable)
# =========================================================
features = bins + cont + ['ecog_ps']
X = sm.add_constant(df[features].astype(float))
y = df['pfs_months']
m = sm.OLS(y, X).fit()
out2 = {'rsquared': float(m.rsquared), 'rsquared_adj': float(m.rsquared_adj),
        'coefs': {k: {'beta': float(m.params[k]), 'p': float(m.pvalues[k])} for k in m.params.index}}
results['iter02_multivariable_main_effects'] = out2

# =========================================================
# ITERATION 3: treatment x receptor interactions
# =========================================================
def interaction(treat, modifier, df=df):
    # Subgroups
    g = df.groupby([modifier, treat])['pfs_months'].agg(['mean', 'std', 'count']).reset_index()
    # Interaction effect via OLS
    m = smf.ols(f'pfs_months ~ {treat} * {modifier}', data=df).fit()
    coef = m.params.get(f'{treat}:{modifier}', np.nan)
    pval = m.pvalues.get(f'{treat}:{modifier}', np.nan)
    # Effect of treatment within each modifier level
    eff_when_mod1 = df.loc[(df[modifier] == 1) & (df[treat] == 1), 'pfs_months'].mean() - \
                    df.loc[(df[modifier] == 1) & (df[treat] == 0), 'pfs_months'].mean()
    eff_when_mod0 = df.loc[(df[modifier] == 0) & (df[treat] == 1), 'pfs_months'].mean() - \
                    df.loc[(df[modifier] == 0) & (df[treat] == 0), 'pfs_months'].mean()
    # p-values for each effect
    p_mod1 = stats.ttest_ind(df.loc[(df[modifier] == 1) & (df[treat] == 1), 'pfs_months'],
                              df.loc[(df[modifier] == 1) & (df[treat] == 0), 'pfs_months']).pvalue
    p_mod0 = stats.ttest_ind(df.loc[(df[modifier] == 0) & (df[treat] == 1), 'pfs_months'],
                              df.loc[(df[modifier] == 0) & (df[treat] == 0), 'pfs_months']).pvalue
    return {'interaction_beta': float(coef), 'interaction_p': float(pval),
            'eff_treat_when_mod1': float(eff_when_mod1), 'p_when_mod1': float(p_mod1),
            'eff_treat_when_mod0': float(eff_when_mod0), 'p_when_mod0': float(p_mod0),
            'group_table': g.to_dict('records')}

out3 = {}
# Trastuzumab x HER2
out3['trastuzumab_x_her2_positive'] = interaction('treatment_trastuzumab', 'her2_positive')
# Tamoxifen x ER
out3['tamoxifen_x_er_positive'] = interaction('treatment_tamoxifen', 'er_positive')
# Palbociclib x ER
out3['palbociclib_x_er_positive'] = interaction('treatment_palbociclib', 'er_positive')
# Palbociclib x postmenopausal
out3['palbociclib_x_postmenopausal'] = interaction('treatment_palbociclib', 'postmenopausal')
# Olaparib x BRCA1
out3['olaparib_x_brca1'] = interaction('treatment_olaparib', 'brca1_mutation')
# Olaparib x BRCA2
out3['olaparib_x_brca2'] = interaction('treatment_olaparib', 'brca2_mutation')
# Pembrolizumab x ER negative
df['er_negative'] = 1 - df['er_positive']
out3['pembrolizumab_x_er_negative'] = interaction('treatment_pembrolizumab', 'er_negative')
# Pembrolizumab x triple negative (er-, pr-, her2-)
df['triple_negative'] = ((df['er_positive'] == 0) & (df['pr_positive'] == 0) & (df['her2_positive'] == 0)).astype(int)
out3['pembrolizumab_x_triple_negative'] = interaction('treatment_pembrolizumab', 'triple_negative')
# Sacituzumab x triple negative
out3['sacituzumab_x_triple_negative'] = interaction('treatment_sacituzumab_govitecan', 'triple_negative')

# Tamoxifen x postmenopausal
out3['tamoxifen_x_postmenopausal'] = interaction('treatment_tamoxifen', 'postmenopausal')
# Trastuzumab x her2_low
out3['trastuzumab_x_her2_low'] = interaction('treatment_trastuzumab', 'her2_low')
# Pembrolizumab x stage_iv
out3['pembrolizumab_x_stage_iv'] = interaction('treatment_pembrolizumab', 'stage_iv')
# Olaparib x BRCA either
df['brca_any'] = ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)).astype(int)
out3['olaparib_x_brca_any'] = interaction('treatment_olaparib', 'brca_any')

results['iter03_treatment_biomarker_interactions'] = out3

# =========================================================
# ITERATION 4: continuous features regression with treatments adj
# =========================================================
out4 = {}
for tx in ['treatment_tamoxifen', 'treatment_palbociclib', 'treatment_trastuzumab',
           'treatment_olaparib', 'treatment_sacituzumab_govitecan', 'treatment_pembrolizumab']:
    m = smf.ols(f'pfs_months ~ {tx} + age_years + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + ldh_u_l + crp_mg_l + nlr', data=df).fit()
    out4[tx] = {'beta': float(m.params[tx]), 'p': float(m.pvalues[tx]),
                'rsquared': float(m.rsquared)}
results['iter04_treatment_adjusted'] = out4

# =========================================================
# ITERATION 5: lab biomarker categories (high LDH, low albumin)
# =========================================================
out5 = {}
df['ldh_high'] = (df['ldh_u_l'] > df['ldh_u_l'].median()).astype(int)
df['albumin_low'] = (df['albumin_g_dl'] < df['albumin_g_dl'].median()).astype(int)
df['crp_high'] = (df['crp_mg_l'] > df['crp_mg_l'].median()).astype(int)
df['nlr_high'] = (df['nlr'] > df['nlr'].median()).astype(int)
df['hgb_low'] = (df['hemoglobin_g_dl'] < df['hemoglobin_g_dl'].median()).astype(int)
for var in ['ldh_high', 'albumin_low', 'crp_high', 'nlr_high', 'hgb_low']:
    a = df.loc[df[var] == 1, 'pfs_months']
    b = df.loc[df[var] == 0, 'pfs_months']
    t, p = stats.ttest_ind(a, b, equal_var=False)
    out5[var] = {'mean_pos': float(a.mean()), 'mean_neg': float(b.mean()),
                 'diff': float(a.mean() - b.mean()), 'p': float(p)}

# Pembrolizumab x ldh_high
out5['pembrolizumab_x_ldh_high'] = interaction('treatment_pembrolizumab', 'ldh_high')
# Pembrolizumab x crp_high
out5['pembrolizumab_x_crp_high'] = interaction('treatment_pembrolizumab', 'crp_high')

results['iter05_lab_biomarkers'] = out5

# =========================================================
# ITERATION 6: brain mets, stage_iv interactions with treatments
# =========================================================
out6 = {}
for tx in ['treatment_tamoxifen', 'treatment_palbociclib', 'treatment_trastuzumab',
           'treatment_olaparib', 'treatment_sacituzumab_govitecan', 'treatment_pembrolizumab']:
    out6[f'{tx}_x_stage_iv'] = interaction(tx, 'stage_iv')
    out6[f'{tx}_x_has_brain_mets'] = interaction(tx, 'has_brain_mets')
results['iter06_stage_brain_interactions'] = out6

# =========================================================
# ITERATION 7: PIK3CA x palbociclib  (alpelisib not given but PIK3CA is biomarker)
# =========================================================
out7 = {}
out7['palbociclib_x_pik3ca'] = interaction('treatment_palbociclib', 'pik3ca_mutation')
# Three-way: palbociclib in ER+ postmenopausal
df['er_postmen'] = ((df['er_positive'] == 1) & (df['postmenopausal'] == 1)).astype(int)
out7['palbociclib_x_er_postmenopausal'] = interaction('treatment_palbociclib', 'er_postmen')
# Trastuzumab in HER2+ stage IV
df['her2_stage_iv'] = ((df['her2_positive'] == 1) & (df['stage_iv'] == 1)).astype(int)
out7['trastuzumab_x_her2pos_stage_iv'] = interaction('treatment_trastuzumab', 'her2_stage_iv')
# Pembrolizumab in TNBC stage IV
df['tnbc_stage_iv'] = ((df['triple_negative'] == 1) & (df['stage_iv'] == 1)).astype(int)
out7['pembrolizumab_x_tnbc_stage_iv'] = interaction('treatment_pembrolizumab', 'tnbc_stage_iv')
# Olaparib in BRCA stage IV
df['brca_stage_iv'] = ((df['brca_any'] == 1) & (df['stage_iv'] == 1)).astype(int)
out7['olaparib_x_brca_stage_iv'] = interaction('treatment_olaparib', 'brca_stage_iv')
# Pembro+saci x TNBC
out7['pembrolizumab_in_tnbc_only'] = {
    'eff': float(df.loc[(df['triple_negative'] == 1) & (df['treatment_pembrolizumab'] == 1), 'pfs_months'].mean() -
                 df.loc[(df['triple_negative'] == 1) & (df['treatment_pembrolizumab'] == 0), 'pfs_months'].mean()),
    'n_tnbc': int(df['triple_negative'].sum()),
    'n_tnbc_pembro': int(df.loc[df['triple_negative'] == 1, 'treatment_pembrolizumab'].sum())
}

results['iter07_three_way_subgroups'] = out7

# =========================================================
# ITERATION 8: Confirm subgroup with all signal modifiers (multivariable interaction)
# =========================================================
out8 = {}

# For trastuzumab, full model with HER2 and stage_iv
m = smf.ols('pfs_months ~ treatment_trastuzumab * her2_positive + stage_iv + age_years + ecog_ps + albumin_g_dl + ldh_u_l + crp_mg_l', data=df).fit()
out8['trastuzumab_her2pos_full_model'] = {k: {'beta': float(m.params[k]), 'p': float(m.pvalues[k])} for k in m.params.index}

# For palbociclib, full model with ER and postmenopausal
m = smf.ols('pfs_months ~ treatment_palbociclib * er_positive * postmenopausal + age_years + ecog_ps + albumin_g_dl', data=df).fit()
out8['palbociclib_er_postmen_full'] = {k: {'beta': float(m.params[k]), 'p': float(m.pvalues[k])} for k in m.params.index}

# For olaparib, full model with brca_any
m = smf.ols('pfs_months ~ treatment_olaparib * brca_any + age_years + ecog_ps + stage_iv + albumin_g_dl', data=df).fit()
out8['olaparib_brca_full'] = {k: {'beta': float(m.params[k]), 'p': float(m.pvalues[k])} for k in m.params.index}

# Pembrolizumab x triple_negative full
m = smf.ols('pfs_months ~ treatment_pembrolizumab * triple_negative + age_years + ecog_ps + stage_iv + albumin_g_dl', data=df).fit()
out8['pembrolizumab_tnbc_full'] = {k: {'beta': float(m.params[k]), 'p': float(m.pvalues[k])} for k in m.params.index}

# Sacituzumab x triple_negative full
m = smf.ols('pfs_months ~ treatment_sacituzumab_govitecan * triple_negative + age_years + ecog_ps + stage_iv + albumin_g_dl', data=df).fit()
out8['sacituzumab_tnbc_full'] = {k: {'beta': float(m.params[k]), 'p': float(m.pvalues[k])} for k in m.params.index}

# Tamoxifen x ER positive full
m = smf.ols('pfs_months ~ treatment_tamoxifen * er_positive + age_years + ecog_ps + stage_iv + albumin_g_dl', data=df).fit()
out8['tamoxifen_er_full'] = {k: {'beta': float(m.params[k]), 'p': float(m.pvalues[k])} for k in m.params.index}

results['iter08_subgroup_confirmation'] = out8

# =========================================================
# ITERATION 9: HER2-low and trastuzumab in HER2-low only?
# =========================================================
out9 = {}
# HER2-low subgroup
df['her2_zero'] = ((df['her2_positive'] == 0) & (df['her2_low'] == 0)).astype(int)
out9['trastuzumab_in_her2_zero'] = interaction('treatment_trastuzumab', 'her2_zero')
out9['trastuzumab_in_her2_low_only'] = {
    'mean_treated_her2_low': float(df.loc[(df['her2_low'] == 1) & (df['her2_positive'] == 0) & (df['treatment_trastuzumab'] == 1), 'pfs_months'].mean()),
    'mean_untreated_her2_low': float(df.loc[(df['her2_low'] == 1) & (df['her2_positive'] == 0) & (df['treatment_trastuzumab'] == 0), 'pfs_months'].mean()),
}
results['iter09_her2_low'] = out9

# =========================================================
# ITERATION 10: Massive interaction screen: each treatment vs each binary
# =========================================================
out10 = {}
treatments = ['treatment_tamoxifen', 'treatment_palbociclib', 'treatment_trastuzumab',
              'treatment_olaparib', 'treatment_sacituzumab_govitecan', 'treatment_pembrolizumab']
binary_modifiers = ['sex_female', 'stage_iv', 'has_brain_mets', 'node_positive',
                    'postmenopausal', 'er_positive', 'pr_positive', 'her2_positive',
                    'her2_low', 'brca1_mutation', 'brca2_mutation', 'pik3ca_mutation',
                    'triple_negative', 'brca_any',
                    'ldh_high', 'albumin_low', 'crp_high', 'nlr_high', 'hgb_low']
for tx in treatments:
    out10[tx] = {}
    for mod in binary_modifiers:
        if tx == mod:
            continue
        try:
            m = smf.ols(f'pfs_months ~ {tx} * {mod}', data=df).fit()
            ip = m.pvalues.get(f'{tx}:{mod}', np.nan)
            ic = m.params.get(f'{tx}:{mod}', np.nan)
            out10[tx][mod] = {'interaction_beta': float(ic), 'interaction_p': float(ip)}
        except Exception:
            out10[tx][mod] = {'error': 'failed'}
results['iter10_full_interaction_screen'] = out10

# =========================================================
# ITERATION 11: Within-subgroup pure analysis: palbo in ER+ postmen
# =========================================================
out11 = {}
# Palbociclib effect within ER+ postmenopausal
sub = df[(df['er_positive'] == 1) & (df['postmenopausal'] == 1)]
m1 = sub.loc[sub['treatment_palbociclib'] == 1, 'pfs_months'].mean()
m0 = sub.loc[sub['treatment_palbociclib'] == 0, 'pfs_months'].mean()
t, p = stats.ttest_ind(sub.loc[sub['treatment_palbociclib'] == 1, 'pfs_months'],
                        sub.loc[sub['treatment_palbociclib'] == 0, 'pfs_months'])
out11['palbo_er_postmen'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                              'diff': float(m1 - m0), 'p': float(p),
                              'n_treated': int((sub['treatment_palbociclib'] == 1).sum()),
                              'n_untreated': int((sub['treatment_palbociclib'] == 0).sum())}

# Palbo in ER+ premenopausal vs postmen
sub_pre = df[(df['er_positive'] == 1) & (df['postmenopausal'] == 0)]
m1 = sub_pre.loc[sub_pre['treatment_palbociclib'] == 1, 'pfs_months'].mean()
m0 = sub_pre.loc[sub_pre['treatment_palbociclib'] == 0, 'pfs_months'].mean()
t, p = stats.ttest_ind(sub_pre.loc[sub_pre['treatment_palbociclib'] == 1, 'pfs_months'],
                        sub_pre.loc[sub_pre['treatment_palbociclib'] == 0, 'pfs_months'])
out11['palbo_er_premen'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                             'diff': float(m1 - m0), 'p': float(p)}

# Palbo in ER- (any menopause)
sub_neg = df[df['er_positive'] == 0]
m1 = sub_neg.loc[sub_neg['treatment_palbociclib'] == 1, 'pfs_months'].mean()
m0 = sub_neg.loc[sub_neg['treatment_palbociclib'] == 0, 'pfs_months'].mean()
t, p = stats.ttest_ind(sub_neg.loc[sub_neg['treatment_palbociclib'] == 1, 'pfs_months'],
                        sub_neg.loc[sub_neg['treatment_palbociclib'] == 0, 'pfs_months'])
out11['palbo_er_neg'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                          'diff': float(m1 - m0), 'p': float(p)}

results['iter11_palbo_subgroup_dissection'] = out11

# =========================================================
# ITERATION 12: Olaparib in BRCA1, BRCA2, neither
# =========================================================
out12 = {}
for label, mask in [('brca1', df['brca1_mutation'] == 1),
                     ('brca2', df['brca2_mutation'] == 1),
                     ('brca_any', df['brca_any'] == 1),
                     ('brca_none', df['brca_any'] == 0)]:
    sub = df[mask]
    m1 = sub.loc[sub['treatment_olaparib'] == 1, 'pfs_months'].mean()
    m0 = sub.loc[sub['treatment_olaparib'] == 0, 'pfs_months'].mean()
    t, p = stats.ttest_ind(sub.loc[sub['treatment_olaparib'] == 1, 'pfs_months'],
                            sub.loc[sub['treatment_olaparib'] == 0, 'pfs_months'])
    out12[f'olaparib_in_{label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                       'diff': float(m1 - m0) if not np.isnan(m1 - m0) else None,
                                       'p': float(p) if not np.isnan(p) else None,
                                       'n_treated': int((sub['treatment_olaparib'] == 1).sum()),
                                       'n_untreated': int((sub['treatment_olaparib'] == 0).sum())}
results['iter12_olaparib_brca'] = out12

# =========================================================
# ITERATION 13: trastuzumab in HER2+ vs HER2-low vs HER2-zero
# =========================================================
out13 = {}
for label, mask in [('her2pos', df['her2_positive'] == 1),
                     ('her2low', (df['her2_positive'] == 0) & (df['her2_low'] == 1)),
                     ('her2zero', (df['her2_positive'] == 0) & (df['her2_low'] == 0))]:
    sub = df[mask]
    m1 = sub.loc[sub['treatment_trastuzumab'] == 1, 'pfs_months'].mean()
    m0 = sub.loc[sub['treatment_trastuzumab'] == 0, 'pfs_months'].mean()
    t, p = stats.ttest_ind(sub.loc[sub['treatment_trastuzumab'] == 1, 'pfs_months'],
                            sub.loc[sub['treatment_trastuzumab'] == 0, 'pfs_months'])
    out13[f'trast_in_{label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                    'diff': float(m1 - m0) if not np.isnan(m1 - m0) else None,
                                    'p': float(p) if not np.isnan(p) else None,
                                    'n_treated': int((sub['treatment_trastuzumab'] == 1).sum())}
results['iter13_trastuzumab_her2'] = out13

# =========================================================
# ITERATION 14: Pembrolizumab in TNBC stratified
# =========================================================
out14 = {}
for label, mask in [('tnbc', df['triple_negative'] == 1),
                     ('non_tnbc', df['triple_negative'] == 0),
                     ('tnbc_stage_iv', (df['triple_negative'] == 1) & (df['stage_iv'] == 1)),
                     ('tnbc_non_stage_iv', (df['triple_negative'] == 1) & (df['stage_iv'] == 0))]:
    sub = df[mask]
    if len(sub) == 0:
        continue
    m1 = sub.loc[sub['treatment_pembrolizumab'] == 1, 'pfs_months'].mean()
    m0 = sub.loc[sub['treatment_pembrolizumab'] == 0, 'pfs_months'].mean()
    t, p = stats.ttest_ind(sub.loc[sub['treatment_pembrolizumab'] == 1, 'pfs_months'],
                            sub.loc[sub['treatment_pembrolizumab'] == 0, 'pfs_months'])
    out14[f'pembro_in_{label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                     'diff': float(m1 - m0) if not np.isnan(m1 - m0) else None,
                                     'p': float(p) if not np.isnan(p) else None,
                                     'n_treated': int((sub['treatment_pembrolizumab'] == 1).sum())}
results['iter14_pembro_tnbc_subgroups'] = out14

# =========================================================
# ITERATION 15: Sacituzumab in TNBC subsets
# =========================================================
out15 = {}
for label, mask in [('tnbc', df['triple_negative'] == 1),
                     ('non_tnbc', df['triple_negative'] == 0),
                     ('hr_pos', (df['er_positive'] == 1) | (df['pr_positive'] == 1)),
                     ('her2_pos', df['her2_positive'] == 1)]:
    sub = df[mask]
    if len(sub) == 0:
        continue
    m1 = sub.loc[sub['treatment_sacituzumab_govitecan'] == 1, 'pfs_months'].mean()
    m0 = sub.loc[sub['treatment_sacituzumab_govitecan'] == 0, 'pfs_months'].mean()
    t, p = stats.ttest_ind(sub.loc[sub['treatment_sacituzumab_govitecan'] == 1, 'pfs_months'],
                            sub.loc[sub['treatment_sacituzumab_govitecan'] == 0, 'pfs_months'])
    out15[f'saci_in_{label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                   'diff': float(m1 - m0) if not np.isnan(m1 - m0) else None,
                                   'p': float(p) if not np.isnan(p) else None,
                                   'n_treated': int((sub['treatment_sacituzumab_govitecan'] == 1).sum())}
results['iter15_sacituzumab_subgroups'] = out15

# =========================================================
# ITERATION 16: Tamoxifen in ER+ pre vs post; also PR+
# =========================================================
out16 = {}
for label, mask in [('er_pos', df['er_positive'] == 1),
                     ('er_neg', df['er_positive'] == 0),
                     ('er_pos_premen', (df['er_positive'] == 1) & (df['postmenopausal'] == 0)),
                     ('er_pos_postmen', (df['er_positive'] == 1) & (df['postmenopausal'] == 1)),
                     ('pr_pos', df['pr_positive'] == 1),
                     ('er_pr_pos', (df['er_positive'] == 1) & (df['pr_positive'] == 1))]:
    sub = df[mask]
    if len(sub) == 0:
        continue
    m1 = sub.loc[sub['treatment_tamoxifen'] == 1, 'pfs_months'].mean()
    m0 = sub.loc[sub['treatment_tamoxifen'] == 0, 'pfs_months'].mean()
    t, p = stats.ttest_ind(sub.loc[sub['treatment_tamoxifen'] == 1, 'pfs_months'],
                            sub.loc[sub['treatment_tamoxifen'] == 0, 'pfs_months'])
    out16[f'tam_in_{label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                  'diff': float(m1 - m0) if not np.isnan(m1 - m0) else None,
                                  'p': float(p) if not np.isnan(p) else None,
                                  'n_treated': int((sub['treatment_tamoxifen'] == 1).sum())}
results['iter16_tamoxifen_subgroups'] = out16

# =========================================================
# ITERATION 17: All treatments in stage_iv and non-stage_iv
# =========================================================
out17 = {}
for tx in treatments:
    for label, mask in [('stage_iv', df['stage_iv'] == 1),
                         ('non_stage_iv', df['stage_iv'] == 0)]:
        sub = df[mask]
        m1 = sub.loc[sub[tx] == 1, 'pfs_months'].mean()
        m0 = sub.loc[sub[tx] == 0, 'pfs_months'].mean()
        t, p = stats.ttest_ind(sub.loc[sub[tx] == 1, 'pfs_months'],
                                sub.loc[sub[tx] == 0, 'pfs_months'])
        out17[f'{tx}_in_{label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                      'diff': float(m1 - m0) if not np.isnan(m1 - m0) else None,
                                      'p': float(p) if not np.isnan(p) else None,
                                      'n_treated': int((sub[tx] == 1).sum())}
results['iter17_treatments_by_stage'] = out17

# =========================================================
# ITERATION 18: Treatment effect heterogeneity by lab biomarkers
# =========================================================
out18 = {}
for tx in treatments:
    for mod in ['ldh_high', 'crp_high', 'nlr_high', 'albumin_low', 'hgb_low']:
        try:
            m = smf.ols(f'pfs_months ~ {tx} * {mod}', data=df).fit()
            ip = m.pvalues.get(f'{tx}:{mod}', np.nan)
            ic = m.params.get(f'{tx}:{mod}', np.nan)
            out18[f'{tx}_x_{mod}'] = {'interaction_beta': float(ic), 'interaction_p': float(ip)}
        except Exception as e:
            out18[f'{tx}_x_{mod}'] = {'error': str(e)}
results['iter18_treatment_x_lab_screen'] = out18

# =========================================================
# ITERATION 19: Brain mets and treatments
# =========================================================
out19 = {}
for tx in treatments:
    for label, mask in [('brain_mets', df['has_brain_mets'] == 1),
                         ('no_brain_mets', df['has_brain_mets'] == 0)]:
        sub = df[mask]
        if len(sub) == 0 or (sub[tx] == 1).sum() < 5 or (sub[tx] == 0).sum() < 5:
            continue
        m1 = sub.loc[sub[tx] == 1, 'pfs_months'].mean()
        m0 = sub.loc[sub[tx] == 0, 'pfs_months'].mean()
        t, p = stats.ttest_ind(sub.loc[sub[tx] == 1, 'pfs_months'],
                                sub.loc[sub[tx] == 0, 'pfs_months'])
        out19[f'{tx}_in_{label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                      'diff': float(m1 - m0) if not np.isnan(m1 - m0) else None,
                                      'p': float(p) if not np.isnan(p) else None,
                                      'n_treated': int((sub[tx] == 1).sum())}
results['iter19_treatment_brain_mets'] = out19

# =========================================================
# ITERATION 20: Final candidate subgroup definitions confirmation
# Test the exact best subgroup definitions and quantify treatment effect
# =========================================================
out20 = {}

# Hypothesis: trastuzumab benefit in HER2-positive (regardless of other features)
# Define and test
sub = df[df['her2_positive'] == 1]
m1 = sub.loc[sub['treatment_trastuzumab'] == 1, 'pfs_months'].mean()
m0 = sub.loc[sub['treatment_trastuzumab'] == 0, 'pfs_months'].mean()
t, p = stats.ttest_ind(sub.loc[sub['treatment_trastuzumab'] == 1, 'pfs_months'],
                        sub.loc[sub['treatment_trastuzumab'] == 0, 'pfs_months'])
out20['trastuzumab_in_her2pos'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                     'diff': float(m1 - m0), 'p': float(p),
                                     'n_treated': int((sub['treatment_trastuzumab'] == 1).sum())}

# Tamoxifen in ER+
sub = df[df['er_positive'] == 1]
m1 = sub.loc[sub['treatment_tamoxifen'] == 1, 'pfs_months'].mean()
m0 = sub.loc[sub['treatment_tamoxifen'] == 0, 'pfs_months'].mean()
t, p = stats.ttest_ind(sub.loc[sub['treatment_tamoxifen'] == 1, 'pfs_months'],
                        sub.loc[sub['treatment_tamoxifen'] == 0, 'pfs_months'])
out20['tamoxifen_in_erpos'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                 'diff': float(m1 - m0), 'p': float(p),
                                 'n_treated': int((sub['treatment_tamoxifen'] == 1).sum())}

# Palbociclib in ER+ postmenopausal
sub = df[(df['er_positive'] == 1) & (df['postmenopausal'] == 1)]
m1 = sub.loc[sub['treatment_palbociclib'] == 1, 'pfs_months'].mean()
m0 = sub.loc[sub['treatment_palbociclib'] == 0, 'pfs_months'].mean()
t, p = stats.ttest_ind(sub.loc[sub['treatment_palbociclib'] == 1, 'pfs_months'],
                        sub.loc[sub['treatment_palbociclib'] == 0, 'pfs_months'])
out20['palbociclib_in_er_postmen'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                        'diff': float(m1 - m0), 'p': float(p),
                                        'n_treated': int((sub['treatment_palbociclib'] == 1).sum())}

# Olaparib in BRCA-any
sub = df[df['brca_any'] == 1]
m1 = sub.loc[sub['treatment_olaparib'] == 1, 'pfs_months'].mean()
m0 = sub.loc[sub['treatment_olaparib'] == 0, 'pfs_months'].mean()
t, p = stats.ttest_ind(sub.loc[sub['treatment_olaparib'] == 1, 'pfs_months'],
                        sub.loc[sub['treatment_olaparib'] == 0, 'pfs_months'])
out20['olaparib_in_brca_any'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                   'diff': float(m1 - m0), 'p': float(p),
                                   'n_treated': int((sub['treatment_olaparib'] == 1).sum())}

# Pembrolizumab in TNBC
sub = df[df['triple_negative'] == 1]
m1 = sub.loc[sub['treatment_pembrolizumab'] == 1, 'pfs_months'].mean()
m0 = sub.loc[sub['treatment_pembrolizumab'] == 0, 'pfs_months'].mean()
t, p = stats.ttest_ind(sub.loc[sub['treatment_pembrolizumab'] == 1, 'pfs_months'],
                        sub.loc[sub['treatment_pembrolizumab'] == 0, 'pfs_months'])
out20['pembrolizumab_in_tnbc'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                    'diff': float(m1 - m0), 'p': float(p),
                                    'n_treated': int((sub['treatment_pembrolizumab'] == 1).sum())}

# Sacituzumab in TNBC
sub = df[df['triple_negative'] == 1]
m1 = sub.loc[sub['treatment_sacituzumab_govitecan'] == 1, 'pfs_months'].mean()
m0 = sub.loc[sub['treatment_sacituzumab_govitecan'] == 0, 'pfs_months'].mean()
t, p = stats.ttest_ind(sub.loc[sub['treatment_sacituzumab_govitecan'] == 1, 'pfs_months'],
                        sub.loc[sub['treatment_sacituzumab_govitecan'] == 0, 'pfs_months'])
out20['sacituzumab_in_tnbc'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                  'diff': float(m1 - m0), 'p': float(p),
                                  'n_treated': int((sub['treatment_sacituzumab_govitecan'] == 1).sum())}

results['iter20_subgroup_confirmation'] = out20

# =========================================================
# ITERATION 21: Test refined subgroup defs - does adding stage_iv or other modifiers help?
# E.g., trastuzumab in HER2+ stage IV vs non-stage IV
# =========================================================
out21 = {}

# Trastuzumab in HER2+ x stage_iv
sub = df[df['her2_positive'] == 1]
m = smf.ols('pfs_months ~ treatment_trastuzumab * stage_iv', data=sub).fit()
out21['trast_her2pos_x_stage_iv'] = {k: {'beta': float(m.params[k]), 'p': float(m.pvalues[k])} for k in m.params.index}

# Specific cells
for stage_label, stage_mask in [('stage_iv', sub['stage_iv'] == 1),
                                  ('non_stage_iv', sub['stage_iv'] == 0)]:
    s = sub[stage_mask]
    m1 = s.loc[s['treatment_trastuzumab'] == 1, 'pfs_months'].mean()
    m0 = s.loc[s['treatment_trastuzumab'] == 0, 'pfs_months'].mean()
    t, p = stats.ttest_ind(s.loc[s['treatment_trastuzumab'] == 1, 'pfs_months'],
                            s.loc[s['treatment_trastuzumab'] == 0, 'pfs_months'])
    out21[f'trast_her2pos_{stage_label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                                'diff': float(m1 - m0), 'p': float(p),
                                                'n': int(len(s))}

# Pembrolizumab in TNBC x stage_iv
sub = df[df['triple_negative'] == 1]
m = smf.ols('pfs_months ~ treatment_pembrolizumab * stage_iv', data=sub).fit()
out21['pembro_tnbc_x_stage_iv'] = {k: {'beta': float(m.params[k]), 'p': float(m.pvalues[k])} for k in m.params.index}

for stage_label, stage_mask in [('stage_iv', sub['stage_iv'] == 1),
                                  ('non_stage_iv', sub['stage_iv'] == 0)]:
    s = sub[stage_mask]
    m1 = s.loc[s['treatment_pembrolizumab'] == 1, 'pfs_months'].mean()
    m0 = s.loc[s['treatment_pembrolizumab'] == 0, 'pfs_months'].mean()
    t, p = stats.ttest_ind(s.loc[s['treatment_pembrolizumab'] == 1, 'pfs_months'],
                            s.loc[s['treatment_pembrolizumab'] == 0, 'pfs_months'])
    out21[f'pembro_tnbc_{stage_label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                              'diff': float(m1 - m0), 'p': float(p),
                                              'n': int(len(s))}

# Olaparib in BRCA1 vs BRCA2
sub = df[df['brca_any'] == 1]
m = smf.ols('pfs_months ~ treatment_olaparib * brca1_mutation', data=sub).fit()
out21['olap_brca_x_brca1'] = {k: {'beta': float(m.params[k]), 'p': float(m.pvalues[k])} for k in m.params.index}

# Sacituzumab in TNBC stage IV vs not
sub = df[df['triple_negative'] == 1]
for stage_label, stage_mask in [('stage_iv', sub['stage_iv'] == 1),
                                  ('non_stage_iv', sub['stage_iv'] == 0)]:
    s = sub[stage_mask]
    m1 = s.loc[s['treatment_sacituzumab_govitecan'] == 1, 'pfs_months'].mean()
    m0 = s.loc[s['treatment_sacituzumab_govitecan'] == 0, 'pfs_months'].mean()
    t, p = stats.ttest_ind(s.loc[s['treatment_sacituzumab_govitecan'] == 1, 'pfs_months'],
                            s.loc[s['treatment_sacituzumab_govitecan'] == 0, 'pfs_months'])
    out21[f'saci_tnbc_{stage_label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                            'diff': float(m1 - m0), 'p': float(p),
                                            'n': int(len(s))}

results['iter21_refined_subgroups'] = out21

# =========================================================
# ITERATION 22: ECOG PS heterogeneity for treatments
# =========================================================
out22 = {}
for tx in treatments:
    sub_low = df[df['ecog_ps'] == 0]
    sub_high = df[df['ecog_ps'] >= 1]
    m1l = sub_low.loc[sub_low[tx] == 1, 'pfs_months'].mean()
    m0l = sub_low.loc[sub_low[tx] == 0, 'pfs_months'].mean()
    m1h = sub_high.loc[sub_high[tx] == 1, 'pfs_months'].mean()
    m0h = sub_high.loc[sub_high[tx] == 0, 'pfs_months'].mean()
    out22[f'{tx}_ecog0'] = {'diff': float(m1l - m0l)}
    out22[f'{tx}_ecog_ge1'] = {'diff': float(m1h - m0h)}
    m = smf.ols(f'pfs_months ~ {tx} * ecog_ps', data=df).fit()
    out22[f'{tx}_x_ecog_interaction'] = {'beta': float(m.params.get(f'{tx}:ecog_ps', np.nan)),
                                          'p': float(m.pvalues.get(f'{tx}:ecog_ps', np.nan))}
results['iter22_ecog_heterogeneity'] = out22

# =========================================================
# ITERATION 23: Confirm by full multivariable adjustment of subgroup signals
# =========================================================
out23 = {}

# Trastuzumab effect with full covariate adjustment, stratified by HER2
m_her2pos = smf.ols('pfs_months ~ treatment_trastuzumab + age_years + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + ldh_u_l + crp_mg_l + nlr',
                     data=df[df['her2_positive'] == 1]).fit()
out23['trast_in_her2pos_adjusted'] = {'beta': float(m_her2pos.params['treatment_trastuzumab']),
                                        'p': float(m_her2pos.pvalues['treatment_trastuzumab'])}
m_her2neg = smf.ols('pfs_months ~ treatment_trastuzumab + age_years + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + ldh_u_l + crp_mg_l + nlr',
                     data=df[df['her2_positive'] == 0]).fit()
out23['trast_in_her2neg_adjusted'] = {'beta': float(m_her2neg.params['treatment_trastuzumab']),
                                        'p': float(m_her2neg.pvalues['treatment_trastuzumab'])}

# Palbociclib in ER+ postmen vs others, adjusted
m1 = smf.ols('pfs_months ~ treatment_palbociclib + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + crp_mg_l',
              data=df[(df['er_positive'] == 1) & (df['postmenopausal'] == 1)]).fit()
out23['palbo_in_er_postmen_adjusted'] = {'beta': float(m1.params['treatment_palbociclib']),
                                           'p': float(m1.pvalues['treatment_palbociclib'])}
m1 = smf.ols('pfs_months ~ treatment_palbociclib + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + crp_mg_l',
              data=df[~((df['er_positive'] == 1) & (df['postmenopausal'] == 1))]).fit()
out23['palbo_outside_subgroup_adjusted'] = {'beta': float(m1.params['treatment_palbociclib']),
                                              'p': float(m1.pvalues['treatment_palbociclib'])}

# Olaparib in BRCA-any adjusted
m1 = smf.ols('pfs_months ~ treatment_olaparib + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + crp_mg_l',
              data=df[df['brca_any'] == 1]).fit()
out23['olap_in_brca_adjusted'] = {'beta': float(m1.params['treatment_olaparib']),
                                    'p': float(m1.pvalues['treatment_olaparib'])}
m1 = smf.ols('pfs_months ~ treatment_olaparib + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + crp_mg_l',
              data=df[df['brca_any'] == 0]).fit()
out23['olap_outside_brca_adjusted'] = {'beta': float(m1.params['treatment_olaparib']),
                                         'p': float(m1.pvalues['treatment_olaparib'])}

# Pembro in TNBC adjusted
m1 = smf.ols('pfs_months ~ treatment_pembrolizumab + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + crp_mg_l',
              data=df[df['triple_negative'] == 1]).fit()
out23['pembro_in_tnbc_adjusted'] = {'beta': float(m1.params['treatment_pembrolizumab']),
                                      'p': float(m1.pvalues['treatment_pembrolizumab'])}
m1 = smf.ols('pfs_months ~ treatment_pembrolizumab + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + crp_mg_l',
              data=df[df['triple_negative'] == 0]).fit()
out23['pembro_outside_tnbc_adjusted'] = {'beta': float(m1.params['treatment_pembrolizumab']),
                                           'p': float(m1.pvalues['treatment_pembrolizumab'])}

# Sacituzumab in TNBC adjusted
m1 = smf.ols('pfs_months ~ treatment_sacituzumab_govitecan + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + crp_mg_l',
              data=df[df['triple_negative'] == 1]).fit()
out23['saci_in_tnbc_adjusted'] = {'beta': float(m1.params['treatment_sacituzumab_govitecan']),
                                    'p': float(m1.pvalues['treatment_sacituzumab_govitecan'])}
m1 = smf.ols('pfs_months ~ treatment_sacituzumab_govitecan + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + crp_mg_l',
              data=df[df['triple_negative'] == 0]).fit()
out23['saci_outside_tnbc_adjusted'] = {'beta': float(m1.params['treatment_sacituzumab_govitecan']),
                                         'p': float(m1.pvalues['treatment_sacituzumab_govitecan'])}

# Tamoxifen in ER+ adjusted
m1 = smf.ols('pfs_months ~ treatment_tamoxifen + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + crp_mg_l',
              data=df[df['er_positive'] == 1]).fit()
out23['tam_in_erpos_adjusted'] = {'beta': float(m1.params['treatment_tamoxifen']),
                                    'p': float(m1.pvalues['treatment_tamoxifen'])}
m1 = smf.ols('pfs_months ~ treatment_tamoxifen + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + crp_mg_l',
              data=df[df['er_positive'] == 0]).fit()
out23['tam_in_erneg_adjusted'] = {'beta': float(m1.params['treatment_tamoxifen']),
                                    'p': float(m1.pvalues['treatment_tamoxifen'])}

results['iter23_full_adjusted_subgroups'] = out23

# =========================================================
# ITERATION 24: Check if effect concentrated in subset (e.g., TNBC + ECOG 0)
# =========================================================
out24 = {}

# Pembro in TNBC, by ECOG and stage_iv
for label, mask in [('tnbc_ecog0', (df['triple_negative'] == 1) & (df['ecog_ps'] == 0)),
                     ('tnbc_ecog_ge1', (df['triple_negative'] == 1) & (df['ecog_ps'] >= 1)),
                     ('tnbc_stage_iv', (df['triple_negative'] == 1) & (df['stage_iv'] == 1)),
                     ('tnbc_non_stage_iv', (df['triple_negative'] == 1) & (df['stage_iv'] == 0)),
                     ('tnbc_postmen', (df['triple_negative'] == 1) & (df['postmenopausal'] == 1)),
                     ('tnbc_premen', (df['triple_negative'] == 1) & (df['postmenopausal'] == 0)),
                     ('tnbc_ldh_high', (df['triple_negative'] == 1) & (df['ldh_high'] == 1)),
                     ('tnbc_ldh_normal', (df['triple_negative'] == 1) & (df['ldh_high'] == 0)),
                     ('tnbc_albumin_high', (df['triple_negative'] == 1) & (df['albumin_low'] == 0)),
                     ('tnbc_albumin_low', (df['triple_negative'] == 1) & (df['albumin_low'] == 1))]:
    sub = df[mask]
    if len(sub) == 0 or (sub['treatment_pembrolizumab'] == 1).sum() < 5:
        continue
    m1 = sub.loc[sub['treatment_pembrolizumab'] == 1, 'pfs_months'].mean()
    m0 = sub.loc[sub['treatment_pembrolizumab'] == 0, 'pfs_months'].mean()
    t, p = stats.ttest_ind(sub.loc[sub['treatment_pembrolizumab'] == 1, 'pfs_months'],
                            sub.loc[sub['treatment_pembrolizumab'] == 0, 'pfs_months'])
    out24[f'pembro_in_{label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                     'diff': float(m1 - m0), 'p': float(p),
                                     'n_treated': int((sub['treatment_pembrolizumab'] == 1).sum())}

# Trastuzumab subdivisions
for label, mask in [('her2pos_ecog0', (df['her2_positive'] == 1) & (df['ecog_ps'] == 0)),
                     ('her2pos_ecog_ge1', (df['her2_positive'] == 1) & (df['ecog_ps'] >= 1)),
                     ('her2pos_stage_iv', (df['her2_positive'] == 1) & (df['stage_iv'] == 1)),
                     ('her2pos_non_stage_iv', (df['her2_positive'] == 1) & (df['stage_iv'] == 0))]:
    sub = df[mask]
    if len(sub) == 0 or (sub['treatment_trastuzumab'] == 1).sum() < 5:
        continue
    m1 = sub.loc[sub['treatment_trastuzumab'] == 1, 'pfs_months'].mean()
    m0 = sub.loc[sub['treatment_trastuzumab'] == 0, 'pfs_months'].mean()
    t, p = stats.ttest_ind(sub.loc[sub['treatment_trastuzumab'] == 1, 'pfs_months'],
                            sub.loc[sub['treatment_trastuzumab'] == 0, 'pfs_months'])
    out24[f'trast_in_{label}'] = {'mean_treated': float(m1), 'mean_untreated': float(m0),
                                    'diff': float(m1 - m0), 'p': float(p),
                                    'n_treated': int((sub['treatment_trastuzumab'] == 1).sum())}

results['iter24_finer_subgroups'] = out24

# =========================================================
# ITERATION 25: Final summary numbers (treatment effects in best subgroup)
# =========================================================
out25 = {}

# Summary table
summary = []
for tx, label, mask in [
    ('treatment_trastuzumab', 'her2pos', df['her2_positive'] == 1),
    ('treatment_trastuzumab', 'her2neg', df['her2_positive'] == 0),
    ('treatment_tamoxifen', 'erpos', df['er_positive'] == 1),
    ('treatment_tamoxifen', 'erneg', df['er_positive'] == 0),
    ('treatment_palbociclib', 'er_postmen', (df['er_positive'] == 1) & (df['postmenopausal'] == 1)),
    ('treatment_palbociclib', 'not_er_postmen', ~((df['er_positive'] == 1) & (df['postmenopausal'] == 1))),
    ('treatment_olaparib', 'brca_any', df['brca_any'] == 1),
    ('treatment_olaparib', 'brca_none', df['brca_any'] == 0),
    ('treatment_pembrolizumab', 'tnbc', df['triple_negative'] == 1),
    ('treatment_pembrolizumab', 'non_tnbc', df['triple_negative'] == 0),
    ('treatment_sacituzumab_govitecan', 'tnbc', df['triple_negative'] == 1),
    ('treatment_sacituzumab_govitecan', 'non_tnbc', df['triple_negative'] == 0),
]:
    sub = df[mask]
    m1 = sub.loc[sub[tx] == 1, 'pfs_months'].mean()
    m0 = sub.loc[sub[tx] == 0, 'pfs_months'].mean()
    n1 = (sub[tx] == 1).sum()
    n0 = (sub[tx] == 0).sum()
    if n1 > 5 and n0 > 5:
        t, p = stats.ttest_ind(sub.loc[sub[tx] == 1, 'pfs_months'],
                                sub.loc[sub[tx] == 0, 'pfs_months'])
    else:
        t, p = np.nan, np.nan
    summary.append({'treatment': tx, 'subgroup': label,
                    'n_treated': int(n1), 'n_untreated': int(n0),
                    'mean_treated': float(m1) if not np.isnan(m1) else None,
                    'mean_untreated': float(m0) if not np.isnan(m0) else None,
                    'diff': float(m1 - m0) if not np.isnan(m1 - m0) else None,
                    'p': float(p) if not np.isnan(p) else None})
out25['final_summary'] = summary
results['iter25_final_summary'] = out25

# Save
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("Done. Results saved to results.json")
