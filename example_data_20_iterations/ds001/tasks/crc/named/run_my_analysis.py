"""
Run all statistical analyses for the CRC dataset and dump results to JSON.
Each entry corresponds to a hypothesis-analysis pair used in transcript.json.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
OUT = []

def add(hid, text, kind, code, summary, p, eff, sig):
    OUT.append({
        'iter': None,
        'hid': hid,
        'text': text,
        'kind': kind,
        'code': code,
        'result_summary': summary,
        'p_value': float(p) if p is not None and not (isinstance(p, float) and np.isnan(p)) else None,
        'effect_estimate': float(eff) if eff is not None and not (isinstance(eff, float) and np.isnan(eff)) else None,
        'significant': bool(sig) if sig is not None else None,
    })

def ttest_means(g1, g0, label1, label0):
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    diff = float(g1.mean() - g0.mean())
    return diff, float(p), f"Mean PFS {label1}={g1.mean():.3f} vs {label0}={g0.mean():.3f}; diff={diff:+.3f} months (Welch t p={p:.2e})."

def ols_coef(formula, term):
    mod = smf.ols(formula, data=df).fit()
    eff = float(mod.params[term])
    p = float(mod.pvalues[term])
    return eff, p, mod

# ============================================================
# ITERATION 1: ECOG performance status & PFS
# ============================================================
i = 1
g_high = df.loc[df['ecog_ps'] >= 2, 'pfs_months']
g_low = df.loc[df['ecog_ps'] < 2, 'pfs_months']
diff, p, summ = ttest_means(g_high, g_low, 'ECOG>=2', 'ECOG<2')
add('h1_1', 'Patients with poor performance status (ecog_ps >= 2) have shorter pfs_months than those with ecog_ps < 2.', 'novel',
    "df.groupby(df['ecog_ps']>=2)['pfs_months'].mean(); Welch t-test", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ ecog_ps', 'ecog_ps')
add('h1_2', 'Higher ecog_ps (continuous, 0-4) is associated with shorter pfs_months in linear regression.', 'novel',
    "smf.ols('pfs_months ~ ecog_ps', data=df).fit()",
    f"OLS coef for ecog_ps = {eff:+.3f} months per 1-point increase, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 2: Stage IV
# ============================================================
i = 2
g1 = df.loc[df['stage_iv']==1, 'pfs_months']
g0 = df.loc[df['stage_iv']==0, 'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'stage_iv=1', 'stage_iv=0')
add('h2_1', 'Patients with stage_iv=1 have shorter pfs_months than patients with stage_iv=0.', 'novel',
    "Welch t on stage_iv groups", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# Albumin
eff, p, _ = ols_coef('pfs_months ~ albumin_g_dl', 'albumin_g_dl')
add('h2_2', 'Higher albumin_g_dl is associated with longer pfs_months.', 'novel',
    "smf.ols('pfs_months ~ albumin_g_dl').fit()",
    f"OLS slope = {eff:+.3f} months per g/dL, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 3: CEA, LDH, weight_loss
# ============================================================
i = 3
# log-CEA
df['log_cea'] = np.log1p(df['cea_ng_ml'].clip(lower=0))
eff, p, _ = ols_coef('pfs_months ~ log_cea', 'log_cea')
add('h3_1', 'Higher serum CEA (cea_ng_ml) is associated with shorter pfs_months.', 'novel',
    "smf.ols('pfs_months ~ log1p(cea_ng_ml)').fit()",
    f"OLS slope on log1p(cea) = {eff:+.3f} months per log-unit, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ ldh_u_l', 'ldh_u_l')
add('h3_2', 'Higher ldh_u_l is associated with shorter pfs_months.', 'novel',
    "smf.ols('pfs_months ~ ldh_u_l').fit()",
    f"OLS slope = {eff*100:+.4f} months per 100 U/L (raw coef={eff:+.5f}), p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ weight_loss_pct_6mo', 'weight_loss_pct_6mo')
add('h3_3', 'Greater weight_loss_pct_6mo is associated with shorter pfs_months.', 'novel',
    "smf.ols('pfs_months ~ weight_loss_pct_6mo').fit()",
    f"OLS slope = {eff:+.3f} months per 1% weight loss, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 4: BRAF V600E main effect
# ============================================================
i = 4
g1 = df.loc[df['braf_v600e']==1,'pfs_months']
g0 = df.loc[df['braf_v600e']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'braf_v600e=1', 'braf_v600e=0')
add('h4_1', 'Patients with braf_v600e=1 have shorter pfs_months than patients with braf_v600e=0.', 'novel',
    "Welch t on braf_v600e groups", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# MSI-H main effect
g1 = df.loc[df['msi_high']==1,'pfs_months']
g0 = df.loc[df['msi_high']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'msi_high=1', 'msi_high=0')
add('h4_2', 'Patients with msi_high=1 have different pfs_months than patients with msi_high=0.', 'novel',
    "Welch t on msi_high", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 5: Cetuximab × KRAS interaction (classical CRC biology)
# ============================================================
i = 5
mod = smf.ols('pfs_months ~ treatment_cetuximab * kras_mutation', data=df).fit()
eff_int = float(mod.params['treatment_cetuximab:kras_mutation'])
p_int = float(mod.pvalues['treatment_cetuximab:kras_mutation'])
eff_main = float(mod.params['treatment_cetuximab'])
p_main = float(mod.pvalues['treatment_cetuximab'])
add('h5_1', 'In KRAS wild-type patients (kras_mutation=0), treatment_cetuximab is associated with longer pfs_months than no cetuximab; this benefit is absent or reversed in KRAS-mutated patients (kras_mutation=1) — a negative cetuximab × kras_mutation interaction.', 'novel',
    "smf.ols('pfs_months ~ treatment_cetuximab*kras_mutation').fit()",
    f"Cetuximab main effect (KRAS wt) = {eff_main:+.3f} mo (p={p_main:.2e}); cetuximab×kras_mutation interaction = {eff_int:+.3f} mo (p={p_int:.2e}).",
    p_int, eff_int, p_int<0.05)
OUT[-1]['iter'] = i

# Cetuximab effect within KRAS WT only
sub = df[df['kras_mutation']==0]
g1 = sub.loc[sub['treatment_cetuximab']==1,'pfs_months']; g0 = sub.loc[sub['treatment_cetuximab']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'cetux+ KRAS wt', 'cetux- KRAS wt')
add('h5_2', 'Among KRAS wild-type patients (kras_mutation=0), treatment_cetuximab is associated with longer pfs_months than no cetuximab.', 'novel',
    "Welch t on cetuximab within kras_mutation==0 subgroup", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# Within KRAS-mutated patients
sub = df[df['kras_mutation']==1]
g1 = sub.loc[sub['treatment_cetuximab']==1,'pfs_months']; g0 = sub.loc[sub['treatment_cetuximab']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'cetux+ KRAS mut', 'cetux- KRAS mut')
add('h5_3', 'Among KRAS-mutated patients (kras_mutation=1), treatment_cetuximab is associated with shorter pfs_months than no cetuximab (resistance / harm).', 'novel',
    "Welch t on cetuximab within kras_mutation==1", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 6: Cetuximab × NRAS, Cetuximab × BRAF
# ============================================================
i = 6
mod = smf.ols('pfs_months ~ treatment_cetuximab * nras_mutation', data=df).fit()
eff = float(mod.params['treatment_cetuximab:nras_mutation']); p = float(mod.pvalues['treatment_cetuximab:nras_mutation'])
add('h6_1', 'There is a negative interaction between treatment_cetuximab and nras_mutation on pfs_months: cetuximab benefit is reduced/reversed in NRAS-mutated patients.', 'novel',
    "smf.ols('pfs_months ~ treatment_cetuximab*nras_mutation').fit()",
    f"Interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

mod = smf.ols('pfs_months ~ treatment_cetuximab * braf_v600e', data=df).fit()
eff = float(mod.params['treatment_cetuximab:braf_v600e']); p = float(mod.pvalues['treatment_cetuximab:braf_v600e'])
add('h6_2', 'There is a negative interaction between treatment_cetuximab and braf_v600e on pfs_months: cetuximab benefit is reduced/reversed in BRAF V600E-mutated patients.', 'novel',
    "smf.ols('pfs_months ~ treatment_cetuximab*braf_v600e').fit()",
    f"Interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# Within RAS/BRAF wild-type patients (the canonical cetuximab indication)
sub = df[(df['kras_mutation']==0)&(df['nras_mutation']==0)&(df['braf_v600e']==0)]
g1 = sub.loc[sub['treatment_cetuximab']==1,'pfs_months']; g0 = sub.loc[sub['treatment_cetuximab']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'cetux+ RAS/BRAF wt', 'cetux- RAS/BRAF wt')
add('h6_3', 'Among patients who are RAS/BRAF wild-type (kras_mutation=0, nras_mutation=0, braf_v600e=0), treatment_cetuximab is associated with longer pfs_months than no cetuximab.', 'refined',
    "Welch t on cetuximab in triple-WT subgroup", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 7: Pembrolizumab × MSI-H
# ============================================================
i = 7
mod = smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high', data=df).fit()
eff = float(mod.params['treatment_pembrolizumab:msi_high']); p = float(mod.pvalues['treatment_pembrolizumab:msi_high'])
add('h7_1', 'There is a positive interaction between treatment_pembrolizumab and msi_high on pfs_months: pembrolizumab benefit is greater in MSI-high patients.', 'novel',
    "smf.ols('pfs_months ~ treatment_pembrolizumab*msi_high').fit()",
    f"Interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# Within MSI-high
sub = df[df['msi_high']==1]
g1 = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months']; g0 = sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'pembro+ MSI-H', 'pembro- MSI-H')
add('h7_2', 'Among MSI-high patients (msi_high=1), treatment_pembrolizumab is associated with longer pfs_months than no pembrolizumab.', 'novel',
    "Welch t on pembrolizumab within msi_high==1", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# Within MSS
sub = df[df['msi_high']==0]
g1 = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months']; g0 = sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'pembro+ MSS', 'pembro- MSS')
add('h7_3', 'Among MSS patients (msi_high=0), treatment_pembrolizumab is NOT associated with longer pfs_months versus no pembrolizumab (null/negative effect).', 'novel',
    "Welch t on pembrolizumab within msi_high==0", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 8: Encorafenib × BRAF V600E
# ============================================================
i = 8
mod = smf.ols('pfs_months ~ treatment_encorafenib * braf_v600e', data=df).fit()
eff = float(mod.params['treatment_encorafenib:braf_v600e']); p = float(mod.pvalues['treatment_encorafenib:braf_v600e'])
add('h8_1', 'There is a positive interaction between treatment_encorafenib and braf_v600e on pfs_months: encorafenib benefit is greater in BRAF V600E-mutant patients.', 'novel',
    "smf.ols('pfs_months ~ treatment_encorafenib*braf_v600e').fit()",
    f"Interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

sub = df[df['braf_v600e']==1]
g1 = sub.loc[sub['treatment_encorafenib']==1,'pfs_months']; g0 = sub.loc[sub['treatment_encorafenib']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'encora+ BRAFmut', 'encora- BRAFmut')
add('h8_2', 'Among BRAF V600E-mutant patients (braf_v600e=1), treatment_encorafenib is associated with longer pfs_months than no encorafenib.', 'novel',
    "Welch t on encorafenib within braf_v600e==1", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

sub = df[df['braf_v600e']==0]
g1 = sub.loc[sub['treatment_encorafenib']==1,'pfs_months']; g0 = sub.loc[sub['treatment_encorafenib']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'encora+ BRAFwt', 'encora- BRAFwt')
add('h8_3', 'Among BRAF wild-type patients (braf_v600e=0), treatment_encorafenib is NOT associated with longer pfs_months than no encorafenib.', 'novel',
    "Welch t on encorafenib within braf_v600e==0", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 9: Trastuzumab/tucatinib × HER2
# ============================================================
i = 9
mod = smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified', data=df).fit()
eff = float(mod.params['treatment_trastuzumab_tucatinib:her2_amplified']); p = float(mod.pvalues['treatment_trastuzumab_tucatinib:her2_amplified'])
add('h9_1', 'There is a positive interaction between treatment_trastuzumab_tucatinib and her2_amplified on pfs_months: HER2-targeted therapy is more beneficial in HER2-amplified patients.', 'novel',
    "smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib*her2_amplified').fit()",
    f"Interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

sub = df[df['her2_amplified']==1]
g1 = sub.loc[sub['treatment_trastuzumab_tucatinib']==1,'pfs_months']; g0 = sub.loc[sub['treatment_trastuzumab_tucatinib']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'TT+ HER2amp', 'TT- HER2amp')
add('h9_2', 'Among HER2-amplified patients (her2_amplified=1), treatment_trastuzumab_tucatinib is associated with longer pfs_months than no trastuzumab/tucatinib.', 'novel',
    "Welch t on trastuzumab_tucatinib within her2_amplified==1", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

sub = df[df['her2_amplified']==0]
g1 = sub.loc[sub['treatment_trastuzumab_tucatinib']==1,'pfs_months']; g0 = sub.loc[sub['treatment_trastuzumab_tucatinib']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'TT+ HER2-', 'TT- HER2-')
add('h9_3', 'Among HER2-non-amplified patients (her2_amplified=0), treatment_trastuzumab_tucatinib is NOT associated with longer pfs_months.', 'novel',
    "Welch t on trastuzumab_tucatinib within her2_amplified==0", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 10: Bevacizumab main effect (largely biomarker-agnostic)
# ============================================================
i = 10
g1 = df.loc[df['treatment_bevacizumab']==1,'pfs_months']; g0 = df.loc[df['treatment_bevacizumab']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'bev+', 'bev-')
add('h10_1', 'treatment_bevacizumab is associated with longer pfs_months than no bevacizumab in the overall cohort.', 'novel',
    "Welch t on bevacizumab", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# Bevacizumab × KRAS interaction (should be small)
mod = smf.ols('pfs_months ~ treatment_bevacizumab * kras_mutation', data=df).fit()
eff = float(mod.params['treatment_bevacizumab:kras_mutation']); p = float(mod.pvalues['treatment_bevacizumab:kras_mutation'])
add('h10_2', 'The benefit of treatment_bevacizumab on pfs_months does NOT differ between KRAS-mutated and KRAS wild-type patients (no interaction).', 'novel',
    "smf.ols('pfs_months ~ treatment_bevacizumab*kras_mutation').fit()",
    f"Interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# Bevacizumab × VTE history — concern but check
mod = smf.ols('pfs_months ~ treatment_bevacizumab * venous_thromboembolism_history', data=df).fit()
eff = float(mod.params['treatment_bevacizumab:venous_thromboembolism_history']); p = float(mod.pvalues['treatment_bevacizumab:venous_thromboembolism_history'])
add('h10_3', 'There is an interaction between treatment_bevacizumab and venous_thromboembolism_history on pfs_months.', 'novel',
    "smf.ols('pfs_months ~ treatment_bevacizumab*venous_thromboembolism_history').fit()",
    f"Interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 11: Right-sided primary
# ============================================================
i = 11
g1 = df.loc[df['right_sided_primary']==1,'pfs_months']; g0 = df.loc[df['right_sided_primary']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'right-sided', 'left-sided')
add('h11_1', 'Patients with right_sided_primary=1 have shorter pfs_months than patients with right_sided_primary=0.', 'novel',
    "Welch t on right_sided_primary", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# Cetuximab × right-sided primary
mod = smf.ols('pfs_months ~ treatment_cetuximab * right_sided_primary', data=df).fit()
eff = float(mod.params['treatment_cetuximab:right_sided_primary']); p = float(mod.pvalues['treatment_cetuximab:right_sided_primary'])
add('h11_2', 'There is a negative interaction between treatment_cetuximab and right_sided_primary on pfs_months: cetuximab benefit is reduced or reversed in right-sided primaries.', 'novel',
    "smf.ols('pfs_months ~ treatment_cetuximab*right_sided_primary').fit()",
    f"Interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# Cetuximab in left-sided RAS/BRAF wt subset
sub = df[(df['kras_mutation']==0)&(df['nras_mutation']==0)&(df['braf_v600e']==0)&(df['right_sided_primary']==0)]
g1 = sub.loc[sub['treatment_cetuximab']==1,'pfs_months']; g0 = sub.loc[sub['treatment_cetuximab']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'cetux+ leftRASwt', 'cetux- leftRASwt')
add('h11_3', 'Among left-sided RAS/BRAF wild-type patients, treatment_cetuximab is associated with longer pfs_months than no cetuximab (largest expected benefit).', 'refined',
    "Welch t on cetuximab in left-sided triple-WT subgroup", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 12: Sex, age, race effects
# ============================================================
i = 12
g1 = df.loc[df['sex_female']==1,'pfs_months']; g0 = df.loc[df['sex_female']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'female', 'male')
add('h12_1', 'Female patients (sex_female=1) have different pfs_months than male patients (sex_female=0).', 'novel',
    "Welch t on sex_female", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ age_years', 'age_years')
add('h12_2', 'Higher age_years is associated with shorter pfs_months.', 'novel',
    "smf.ols('pfs_months ~ age_years').fit()",
    f"OLS slope = {eff:+.4f} months per year of age, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# Race
mod = smf.ols('pfs_months ~ C(race_ethnicity)', data=df).fit()
fp = float(mod.f_pvalue) if mod.f_pvalue is not None else None
# Use anova to get group F-test
anova = sm.stats.anova_lm(mod, typ=2)
p_race = float(anova.loc['C(race_ethnicity)','PR(>F)'])
group_means = df.groupby('race_ethnicity')['pfs_months'].mean().sort_values()
eff = float(group_means.max() - group_means.min())
add('h12_3', 'Mean pfs_months differs across categories of race_ethnicity.', 'novel',
    "anova_lm of pfs_months ~ C(race_ethnicity)",
    f"ANOVA F-test p={p_race:.2e}; group means range = {eff:.3f} months.", p_race, eff, p_race<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 13: Liver/bone mets, multivariable model of clinical features
# ============================================================
i = 13
g1 = df.loc[df['liver_mets']==1,'pfs_months']; g0 = df.loc[df['liver_mets']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'liver_mets=1', 'liver_mets=0')
add('h13_1', 'Patients with liver_mets=1 have shorter pfs_months than patients without liver mets.', 'novel',
    "Welch t on liver_mets", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

g1 = df.loc[df['bone_mets']==1,'pfs_months']; g0 = df.loc[df['bone_mets']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'bone_mets=1', 'bone_mets=0')
add('h13_2', 'Patients with bone_mets=1 have shorter pfs_months than patients without bone mets.', 'novel',
    "Welch t on bone_mets", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# Multivariable
mod = smf.ols('pfs_months ~ ecog_ps + stage_iv + albumin_g_dl + log_cea + ldh_u_l + weight_loss_pct_6mo + age_years + liver_mets + bone_mets + right_sided_primary', data=df).fit()
eff = float(mod.params['ecog_ps']); p = float(mod.pvalues['ecog_ps'])
add('h13_3', 'In a multivariable model adjusting for stage_iv, albumin, log_cea, ldh, weight_loss, age, liver_mets, bone_mets, and right_sided_primary, ecog_ps remains independently associated with shorter pfs_months.', 'novel',
    "smf.ols multivariable",
    f"Adjusted ecog_ps coef = {eff:+.3f} mo per point (p={p:.2e}); model R²={mod.rsquared:.4f}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 14: Inflammation markers — CRP, NLR, hemoglobin
# ============================================================
i = 14
eff, p, _ = ols_coef('pfs_months ~ crp_mg_l', 'crp_mg_l')
add('h14_1', 'Higher crp_mg_l is associated with shorter pfs_months.', 'novel',
    "smf.ols('pfs_months ~ crp_mg_l')", f"OLS slope = {eff:+.5f} mo per mg/L, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ nlr', 'nlr')
add('h14_2', 'Higher nlr (neutrophil-to-lymphocyte ratio) is associated with shorter pfs_months.', 'novel',
    "smf.ols('pfs_months ~ nlr')", f"OLS slope = {eff:+.4f} mo per unit, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ hemoglobin_g_dl', 'hemoglobin_g_dl')
add('h14_3', 'Higher hemoglobin_g_dl is associated with longer pfs_months.', 'novel',
    "smf.ols('pfs_months ~ hemoglobin_g_dl')", f"OLS slope = {eff:+.3f} mo per g/dL, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 15: Liver function tests
# ============================================================
i = 15
eff, p, _ = ols_coef('pfs_months ~ alkaline_phosphatase_u_l', 'alkaline_phosphatase_u_l')
add('h15_1', 'Higher alkaline_phosphatase_u_l is associated with shorter pfs_months.', 'novel',
    "smf.ols", f"OLS slope = {eff:+.6f} mo per U/L, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ ast_u_l', 'ast_u_l')
add('h15_2', 'Higher ast_u_l is associated with shorter pfs_months.', 'novel',
    "smf.ols", f"OLS slope = {eff:+.5f} mo per U/L, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ total_bilirubin_mg_dl', 'total_bilirubin_mg_dl')
add('h15_3', 'Higher total_bilirubin_mg_dl is associated with shorter pfs_months.', 'novel',
    "smf.ols", f"OLS slope = {eff:+.3f} mo per mg/dL, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 16: Symptoms
# ============================================================
i = 16
eff, p, _ = ols_coef('pfs_months ~ fatigue_grade', 'fatigue_grade')
add('h16_1', 'Higher fatigue_grade is associated with shorter pfs_months.', 'novel',
    "smf.ols", f"OLS slope = {eff:+.3f} mo per grade, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ pain_nrs', 'pain_nrs')
add('h16_2', 'Higher pain_nrs is associated with shorter pfs_months.', 'novel',
    "smf.ols", f"OLS slope = {eff:+.3f} mo per pain point, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ appetite_loss_grade', 'appetite_loss_grade')
add('h16_3', 'Higher appetite_loss_grade is associated with shorter pfs_months.', 'novel',
    "smf.ols", f"OLS slope = {eff:+.3f} mo per grade, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 17: Comorbidities
# ============================================================
i = 17
g1 = df.loc[df['heart_failure']==1,'pfs_months']; g0 = df.loc[df['heart_failure']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'HF=1', 'HF=0')
add('h17_1', 'Patients with heart_failure=1 have shorter pfs_months than patients without heart failure.', 'novel',
    "Welch t on heart_failure", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

g1 = df.loc[df['chronic_kidney_disease']==1,'pfs_months']; g0 = df.loc[df['chronic_kidney_disease']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'CKD=1', 'CKD=0')
add('h17_2', 'Patients with chronic_kidney_disease=1 have shorter pfs_months than patients without CKD.', 'novel',
    "Welch t on chronic_kidney_disease", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

g1 = df.loc[df['diabetes_mellitus']==1,'pfs_months']; g0 = df.loc[df['diabetes_mellitus']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'DM=1', 'DM=0')
add('h17_3', 'Patients with diabetes_mellitus=1 have shorter pfs_months than patients without diabetes.', 'novel',
    "Welch t on diabetes_mellitus", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 18: Prior therapy lines, prior chemo
# ============================================================
i = 18
eff, p, _ = ols_coef('pfs_months ~ prior_lines_of_therapy', 'prior_lines_of_therapy')
add('h18_1', 'More prior_lines_of_therapy is associated with shorter pfs_months.', 'novel',
    "smf.ols", f"OLS slope = {eff:+.3f} mo per prior line, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

g1 = df.loc[df['prior_chemotherapy']==1,'pfs_months']; g0 = df.loc[df['prior_chemotherapy']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'priorchemo=1','priorchemo=0')
add('h18_2', 'Patients with prior_chemotherapy=1 have shorter pfs_months than patients without prior chemo.', 'novel',
    "Welch t", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ years_since_diagnosis', 'years_since_diagnosis')
add('h18_3', 'Longer years_since_diagnosis is associated with shorter pfs_months (more advanced/refractory disease).', 'novel',
    "smf.ols", f"OLS slope = {eff:+.3f} mo per year, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 19: SNPs — pharmacogenomics
# ============================================================
i = 19
# UGT1A1 (rs4148323 not in this set; closest analog is rs8175347 not present). Use
# rs4244285 (CYP2C19*2), rs1799853 (CYP2C9*2), rs1801133 (MTHFR C677T) — all
# clinically relevant for chemo metabolism.
eff, p, _ = ols_coef('pfs_months ~ snp_rs1801133', 'snp_rs1801133')
add('h19_1', 'snp_rs1801133 (MTHFR C677T) genotype dose is associated with pfs_months (potentially affecting fluoropyrimidine response).', 'novel',
    "smf.ols('pfs_months ~ snp_rs1801133')", f"OLS slope = {eff:+.3f} mo per allele, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ snp_rs4244285', 'snp_rs4244285')
add('h19_2', 'snp_rs4244285 (CYP2C19*2) genotype dose is associated with pfs_months.', 'novel',
    "smf.ols('pfs_months ~ snp_rs4244285')", f"OLS slope = {eff:+.3f} mo per allele, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# Multi-SNP screen
snp_cols = [c for c in df.columns if c.startswith('snp_')]
results = []
for s in snp_cols:
    eff_, p_, _ = ols_coef(f'pfs_months ~ {s}', s)
    results.append((s, eff_, p_))
results_df = pd.DataFrame(results, columns=['snp','eff','p']).sort_values('p')
top_snp = results_df.iloc[0]
add('h19_3', 'At least one of the SNPs in this dataset (snp_rs* columns) is associated with pfs_months at p<0.05.', 'novel',
    "Loop over snp_* univariate OLS",
    f"Top SNP = {top_snp['snp']} eff={top_snp['eff']:+.3f} mo, p={top_snp['p']:.2e}; {(results_df['p']<0.05).sum()}/{len(results_df)} SNPs nominally significant.",
    float(top_snp['p']), float(top_snp['eff']), float(top_snp['p'])<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 20: Sociodemographic — insurance, rural, education
# ============================================================
i = 20
mod = smf.ols('pfs_months ~ C(insurance_type)', data=df).fit()
anova = sm.stats.anova_lm(mod, typ=2)
p_ins = float(anova.loc['C(insurance_type)','PR(>F)'])
gm = df.groupby('insurance_type')['pfs_months'].mean()
eff = float(gm.max() - gm.min())
add('h20_1', 'Mean pfs_months differs across insurance_type categories.', 'novel',
    "anova_lm of pfs_months ~ C(insurance_type)",
    f"F-test p={p_ins:.2e}; max-min group mean diff = {eff:.3f} mo.",
    p_ins, eff, p_ins<0.05)
OUT[-1]['iter'] = i

g1 = df.loc[df['rural_residence']==1,'pfs_months']; g0 = df.loc[df['rural_residence']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'rural=1','rural=0')
add('h20_2', 'Patients with rural_residence=1 have different pfs_months than patients with rural_residence=0.', 'novel',
    "Welch t on rural_residence", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

eff, p, _ = ols_coef('pfs_months ~ education_years', 'education_years')
add('h20_3', 'Higher education_years is associated with longer pfs_months.', 'novel',
    "smf.ols('pfs_months ~ education_years')", f"OLS slope = {eff:+.4f} mo per year, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 21: Smoking; pleural/pericardial effusion (lung-like features)
# ============================================================
i = 21
eff, p, _ = ols_coef('pfs_months ~ smoking_pack_years', 'smoking_pack_years')
add('h21_1', 'More smoking_pack_years is associated with shorter pfs_months.', 'novel',
    "smf.ols", f"OLS slope = {eff:+.4f} mo per pack-year, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

g1 = df.loc[df['pleural_effusion']==1,'pfs_months']; g0 = df.loc[df['pleural_effusion']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'pleff=1','pleff=0')
add('h21_2', 'Patients with pleural_effusion=1 have shorter pfs_months than those without pleural effusion.', 'novel',
    "Welch t on pleural_effusion", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

g1 = df.loc[df['pericardial_effusion']==1,'pfs_months']; g0 = df.loc[df['pericardial_effusion']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'periceff=1','periceff=0')
add('h21_3', 'Patients with pericardial_effusion=1 have shorter pfs_months than those without pericardial effusion.', 'novel',
    "Welch t on pericardial_effusion", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 22: Other treatments — regorafenib (later-line), cetuximab × triple-WT × left
# ============================================================
i = 22
g1 = df.loc[df['treatment_regorafenib']==1,'pfs_months']; g0 = df.loc[df['treatment_regorafenib']==0,'pfs_months']
diff, p, summ = ttest_means(g1, g0, 'rego+','rego-')
add('h22_1', 'treatment_regorafenib is associated with different pfs_months than no regorafenib in the overall cohort.', 'novel',
    "Welch t on regorafenib", summ, p, diff, p<0.05)
OUT[-1]['iter'] = i

# Regorafenib × prior_lines_of_therapy interaction
mod = smf.ols('pfs_months ~ treatment_regorafenib * prior_lines_of_therapy', data=df).fit()
eff = float(mod.params['treatment_regorafenib:prior_lines_of_therapy']); p = float(mod.pvalues['treatment_regorafenib:prior_lines_of_therapy'])
add('h22_2', 'There is an interaction between treatment_regorafenib and prior_lines_of_therapy on pfs_months.', 'novel',
    "smf.ols('pfs_months ~ treatment_regorafenib*prior_lines_of_therapy').fit()",
    f"Interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# pembrolizumab × tp53_mutation
mod = smf.ols('pfs_months ~ treatment_pembrolizumab * tp53_mutation', data=df).fit()
eff = float(mod.params['treatment_pembrolizumab:tp53_mutation']); p = float(mod.pvalues['treatment_pembrolizumab:tp53_mutation'])
add('h22_3', 'There is an interaction between treatment_pembrolizumab and tp53_mutation on pfs_months.', 'novel',
    "smf.ols('pfs_months ~ treatment_pembrolizumab*tp53_mutation').fit()",
    f"Interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 23: Comprehensive multivariable PFS model
# ============================================================
i = 23
formula = ('pfs_months ~ age_years + sex_female + ecog_ps + stage_iv + right_sided_primary + '
           'kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified + '
           'albumin_g_dl + log_cea + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr + hemoglobin_g_dl + '
           'liver_mets + bone_mets + heart_failure + chronic_kidney_disease + '
           'treatment_cetuximab + treatment_bevacizumab + treatment_pembrolizumab + treatment_encorafenib + '
           'treatment_trastuzumab_tucatinib + treatment_regorafenib + '
           'treatment_cetuximab:kras_mutation + treatment_cetuximab:nras_mutation + treatment_cetuximab:braf_v600e + '
           'treatment_pembrolizumab:msi_high + treatment_encorafenib:braf_v600e + '
           'treatment_trastuzumab_tucatinib:her2_amplified + '
           'prior_lines_of_therapy + years_since_diagnosis')
mod = smf.ols(formula, data=df).fit()
# Test the panel of expected interactions all together via a Wald
keys = ['treatment_cetuximab:kras_mutation','treatment_cetuximab:nras_mutation','treatment_cetuximab:braf_v600e',
        'treatment_pembrolizumab:msi_high','treatment_encorafenib:braf_v600e','treatment_trastuzumab_tucatinib:her2_amplified']
wald = mod.f_test(' = '.join(keys) + ' = 0' if False else ', '.join([f'{k} = 0' for k in keys]))
eff_panel_p = float(wald.pvalue)
add('h23_1', 'Adjusting for clinical, lab, biomarker, and treatment covariates, the joint set of biomarker × matched-treatment interactions (cetuximab×KRAS, cetuximab×NRAS, cetuximab×BRAF, pembrolizumab×MSI, encorafenib×BRAF, trastuzumab/tucatinib×HER2) is jointly significant for pfs_months.', 'novel',
    "Wald test on multivariable model",
    f"Joint Wald F p={eff_panel_p:.2e}; model R²={mod.rsquared:.4f}.",
    eff_panel_p, float(mod.rsquared), eff_panel_p<0.05)
OUT[-1]['iter'] = i

# Adjusted cetuximab×KRAS interaction (specifically in the multivariable model)
eff = float(mod.params['treatment_cetuximab:kras_mutation']); p = float(mod.pvalues['treatment_cetuximab:kras_mutation'])
add('h23_2', 'In the multivariable model, the cetuximab × kras_mutation interaction remains negative and significant for pfs_months after adjustment.', 'refined',
    "multivariable smf.ols",
    f"Adjusted interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# Adjusted pembrolizumab×MSI
eff = float(mod.params['treatment_pembrolizumab:msi_high']); p = float(mod.pvalues['treatment_pembrolizumab:msi_high'])
add('h23_3', 'In the multivariable model, the pembrolizumab × msi_high interaction remains positive and significant for pfs_months after adjustment.', 'refined',
    "multivariable smf.ols",
    f"Adjusted interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 24: Sex × treatment interactions (look for treatment-effect heterogeneity)
# ============================================================
i = 24
mod = smf.ols('pfs_months ~ treatment_pembrolizumab * sex_female', data=df).fit()
eff = float(mod.params['treatment_pembrolizumab:sex_female']); p = float(mod.pvalues['treatment_pembrolizumab:sex_female'])
add('h24_1', 'There is an interaction between treatment_pembrolizumab and sex_female on pfs_months.', 'novel',
    "smf.ols('pfs_months ~ treatment_pembrolizumab*sex_female').fit()",
    f"Interaction coef = {eff:+.3f} mo, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

mod = smf.ols('pfs_months ~ treatment_bevacizumab * age_years', data=df).fit()
eff = float(mod.params['treatment_bevacizumab:age_years']); p = float(mod.pvalues['treatment_bevacizumab:age_years'])
add('h24_2', 'There is an interaction between treatment_bevacizumab and age_years on pfs_months (e.g., reduced benefit at older ages).', 'novel',
    "smf.ols('pfs_months ~ treatment_bevacizumab*age_years').fit()",
    f"Interaction coef = {eff:+.4f} mo per year, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

mod = smf.ols('pfs_months ~ treatment_cetuximab * ecog_ps', data=df).fit()
eff = float(mod.params['treatment_cetuximab:ecog_ps']); p = float(mod.pvalues['treatment_cetuximab:ecog_ps'])
add('h24_3', 'There is an interaction between treatment_cetuximab and ecog_ps on pfs_months: cetuximab benefit varies with performance status.', 'novel',
    "smf.ols('pfs_months ~ treatment_cetuximab*ecog_ps').fit()",
    f"Interaction coef = {eff:+.3f} mo per point, p={p:.2e}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# ============================================================
# ITERATION 25: Final synthesis / robustness — does the cetuximab-by-RAS-WT subgroup remain after adjustment?
# ============================================================
i = 25
# Adjusted cetuximab effect within RAS/BRAF wt
sub = df[(df['kras_mutation']==0)&(df['nras_mutation']==0)&(df['braf_v600e']==0)].copy()
sub['log_cea'] = np.log1p(sub['cea_ng_ml'].clip(lower=0))
mod = smf.ols('pfs_months ~ treatment_cetuximab + age_years + sex_female + ecog_ps + stage_iv + albumin_g_dl + log_cea + ldh_u_l + right_sided_primary', data=sub).fit()
eff = float(mod.params['treatment_cetuximab']); p = float(mod.pvalues['treatment_cetuximab'])
add('h25_1', 'Among RAS/BRAF wild-type patients, treatment_cetuximab remains associated with longer pfs_months after adjustment for age, sex, ecog_ps, stage_iv, albumin, log-CEA, ldh, and right-sided primary.', 'refined',
    "smf.ols multivariable on triple-WT subgroup",
    f"Adjusted cetuximab coef = {eff:+.3f} mo (p={p:.2e}); n={len(sub)}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# Adjusted pembrolizumab effect within MSI-H
sub = df[df['msi_high']==1].copy()
sub['log_cea'] = np.log1p(sub['cea_ng_ml'].clip(lower=0))
mod = smf.ols('pfs_months ~ treatment_pembrolizumab + age_years + sex_female + ecog_ps + stage_iv + albumin_g_dl + log_cea + ldh_u_l', data=sub).fit()
eff = float(mod.params['treatment_pembrolizumab']); p = float(mod.pvalues['treatment_pembrolizumab'])
add('h25_2', 'Among MSI-high patients, treatment_pembrolizumab remains associated with longer pfs_months after adjustment for age, sex, ecog_ps, stage_iv, albumin, log-CEA, and ldh.', 'refined',
    "smf.ols multivariable on MSI-H subgroup",
    f"Adjusted pembrolizumab coef = {eff:+.3f} mo (p={p:.2e}); n={len(sub)}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# Adjusted encorafenib effect within BRAF V600E
sub = df[df['braf_v600e']==1].copy()
sub['log_cea'] = np.log1p(sub['cea_ng_ml'].clip(lower=0))
mod = smf.ols('pfs_months ~ treatment_encorafenib + age_years + sex_female + ecog_ps + stage_iv + albumin_g_dl + log_cea + ldh_u_l', data=sub).fit()
eff = float(mod.params['treatment_encorafenib']); p = float(mod.pvalues['treatment_encorafenib'])
add('h25_3', 'Among BRAF V600E-mutant patients, treatment_encorafenib remains associated with longer pfs_months after adjustment for age, sex, ecog_ps, stage_iv, albumin, log-CEA, and ldh.', 'refined',
    "smf.ols multivariable on BRAF V600E subgroup",
    f"Adjusted encorafenib coef = {eff:+.3f} mo (p={p:.2e}); n={len(sub)}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# Adjusted trastuzumab/tucatinib effect within HER2-amplified
sub = df[df['her2_amplified']==1].copy()
sub['log_cea'] = np.log1p(sub['cea_ng_ml'].clip(lower=0))
mod = smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib + age_years + sex_female + ecog_ps + stage_iv + albumin_g_dl + log_cea + ldh_u_l', data=sub).fit()
eff = float(mod.params['treatment_trastuzumab_tucatinib']); p = float(mod.pvalues['treatment_trastuzumab_tucatinib'])
add('h25_4', 'Among HER2-amplified patients, treatment_trastuzumab_tucatinib remains associated with longer pfs_months after adjustment for age, sex, ecog_ps, stage_iv, albumin, log-CEA, and ldh.', 'refined',
    "smf.ols multivariable on HER2-amp subgroup",
    f"Adjusted T/T coef = {eff:+.3f} mo (p={p:.2e}); n={len(sub)}.", p, eff, p<0.05)
OUT[-1]['iter'] = i

# Save
with open('all_my_results.json','w') as f:
    json.dump(OUT, f, indent=2)
print(f"Wrote {len(OUT)} analyses to all_my_results.json")
