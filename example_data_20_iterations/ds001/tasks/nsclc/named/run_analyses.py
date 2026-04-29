"""Run all analyses in one pass, save results to results.json"""
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


def logit_main(varname, outcome='objective_response'):
    X = df[[varname]].copy()
    if df[varname].dtype == object:
        X = pd.get_dummies(X, drop_first=True).astype(float)
    X = sm.add_constant(X.astype(float))
    model = sm.Logit(df[outcome], X).fit(disp=False)
    target_terms = [c for c in X.columns if c != 'const']
    target = target_terms[0]
    coef = model.params[target]
    p = model.pvalues[target]
    return {'coef_log_odds': float(coef), 'odds_ratio': float(np.exp(coef)), 'p_value': float(p)}


def interaction_test(treatment, biomarker, outcome='objective_response'):
    sub = df[[treatment, biomarker, outcome]].copy().astype(float)
    sub['inter'] = sub[treatment] * sub[biomarker]
    X = sm.add_constant(sub[[treatment, biomarker, 'inter']])
    model = sm.Logit(sub[outcome], X).fit(disp=False)
    coef = model.params['inter']
    p = model.pvalues['inter']
    rates = {}
    for t in [0, 1]:
        for b in [0, 1]:
            mask = (df[treatment] == t) & (df[biomarker] == b)
            if mask.sum() > 0:
                rates[f't{t}_b{b}'] = float(df.loc[mask, outcome].mean())
    return {
        'interaction_coef_log_odds': float(coef),
        'interaction_odds_ratio': float(np.exp(coef)),
        'p_value': float(p),
        'response_rates': rates,
        'treatment_main_p': float(model.pvalues[treatment]),
        'biomarker_main_p': float(model.pvalues[biomarker]),
    }


# ITER 1: Treatment main effects
for tx in ['treatment_pembrolizumab', 'treatment_sotorasib', 'treatment_olaparib', 'treatment_osimertinib']:
    rate1 = df.loc[df[tx] == 1, 'objective_response'].mean()
    rate0 = df.loc[df[tx] == 0, 'objective_response'].mean()
    out = logit_main(tx)
    out['rate_treated'] = float(rate1)
    out['rate_untreated'] = float(rate0)
    out['rate_diff'] = float(rate1 - rate0)
    results[f'tx_main_{tx}'] = out

# ITER 2: ECOG, stage, brain mets, age, sex
for v in ['ecog_ps', 'stage_iv', 'has_brain_mets', 'age_years', 'sex_female']:
    out = logit_main(v)
    results[f'main_{v}'] = out

# ITER 3: Biomarker main
for v in ['egfr_mutation', 'kras_g12c', 'alk_fusion', 'brca2_mutation', 'tmb_high',
          'stk11_mutation', 'keap1_mutation', 'tp53_mutation', 'pdl1_tps']:
    out = logit_main(v)
    if df[v].nunique() == 2:
        out['rate_pos'] = float(df.loc[df[v] == 1, 'objective_response'].mean())
        out['rate_neg'] = float(df.loc[df[v] == 0, 'objective_response'].mean())
    results[f'main_{v}'] = out

# ITER 4: EGFR x osimertinib
results['inter_egfr_osi'] = interaction_test('treatment_osimertinib', 'egfr_mutation')

# ITER 5: KRAS G12C x sotorasib
results['inter_krasg12c_soto'] = interaction_test('treatment_sotorasib', 'kras_g12c')

# ITER 6: BRCA2 x olaparib
results['inter_brca2_olap'] = interaction_test('treatment_olaparib', 'brca2_mutation')

# ITER 7: PD-L1 x pembro
df['pdl1_high'] = (df['pdl1_tps'] >= 0.5).astype(int)
results['inter_pdl1high_pembro'] = interaction_test('treatment_pembrolizumab', 'pdl1_high')

sub = df[['treatment_pembrolizumab', 'pdl1_tps', 'objective_response']].copy().astype(float)
sub['inter'] = sub['treatment_pembrolizumab'] * sub['pdl1_tps']
X = sm.add_constant(sub[['treatment_pembrolizumab', 'pdl1_tps', 'inter']])
m = sm.Logit(sub['objective_response'], X).fit(disp=False)
results['inter_pdl1cont_pembro'] = {
    'interaction_coef_log_odds': float(m.params['inter']),
    'p_value': float(m.pvalues['inter']),
    'pembro_main': float(m.params['treatment_pembrolizumab']),
    'pdl1_main': float(m.params['pdl1_tps']),
}

# ITER 8: TMB x pembro
results['inter_tmb_pembro'] = interaction_test('treatment_pembrolizumab', 'tmb_high')

# ITER 9: STK11 / KEAP1 x pembro
results['inter_stk11_pembro'] = interaction_test('treatment_pembrolizumab', 'stk11_mutation')
results['inter_keap1_pembro'] = interaction_test('treatment_pembrolizumab', 'keap1_mutation')

# ITER 10: ALK x osi/pembro
results['inter_alk_osi'] = interaction_test('treatment_osimertinib', 'alk_fusion')
results['inter_alk_pembro'] = interaction_test('treatment_pembrolizumab', 'alk_fusion')

# ITER 11: Lab markers
for v in ['albumin_g_dl', 'ldh_u_l', 'nlr', 'crp_mg_l', 'weight_loss_pct_6mo', 'hemoglobin_g_dl']:
    out = logit_main(v)
    results[f'main_{v}'] = out

# ITER 12: Mets
for v in ['liver_mets', 'bone_mets', 'adrenal_mets', 'pleural_effusion', 'pericardial_effusion']:
    out = logit_main(v)
    out['rate_pos'] = float(df.loc[df[v] == 1, 'objective_response'].mean())
    out['rate_neg'] = float(df.loc[df[v] == 0, 'objective_response'].mean())
    results[f'mets_{v}'] = out

# ITER 13: Histology / smoking
df['squamous'] = (df['histology'] == 'squamous').astype(int)
out = logit_main('squamous')
out['rate_squamous'] = float(df.loc[df['squamous'] == 1, 'objective_response'].mean())
out['rate_adeno'] = float(df.loc[df['squamous'] == 0, 'objective_response'].mean())
results['main_squamous'] = out

for s in ['current', 'former', 'never']:
    df[f'smoke_{s}'] = (df['smoking_status'] == s).astype(int)
    out = logit_main(f'smoke_{s}')
    out['rate_in_group'] = float(df.loc[df[f'smoke_{s}'] == 1, 'objective_response'].mean())
    out['rate_other'] = float(df.loc[df[f'smoke_{s}'] == 0, 'objective_response'].mean())
    results[f'smoke_{s}'] = out

# ITER 14: smoking x treatment
sub = df[['treatment_pembrolizumab', 'smoking_pack_years', 'objective_response']].copy().astype(float)
sub['inter'] = sub['treatment_pembrolizumab'] * sub['smoking_pack_years']
X = sm.add_constant(sub[['treatment_pembrolizumab', 'smoking_pack_years', 'inter']])
m = sm.Logit(sub['objective_response'], X).fit(disp=False)
results['inter_packyrs_pembro'] = {
    'interaction_coef_log_odds': float(m.params['inter']),
    'p_value': float(m.pvalues['inter']),
}
results['inter_never_pembro'] = interaction_test('treatment_pembrolizumab', 'smoke_never')
results['inter_never_osi'] = interaction_test('treatment_osimertinib', 'smoke_never')

# ITER 15: Sex x treatment
results['inter_sex_pembro'] = interaction_test('treatment_pembrolizumab', 'sex_female')
results['inter_sex_osi'] = interaction_test('treatment_osimertinib', 'sex_female')
results['inter_sex_soto'] = interaction_test('treatment_sotorasib', 'sex_female')
results['inter_sex_olap'] = interaction_test('treatment_olaparib', 'sex_female')

# ITER 16: Race
for r in ['white', 'black', 'hispanic', 'asian', 'other']:
    df[f'race_{r}'] = (df['race_ethnicity'] == r).astype(int)
    out = logit_main(f'race_{r}')
    out['rate_in_group'] = float(df.loc[df[f'race_{r}'] == 1, 'objective_response'].mean())
    out['rate_other'] = float(df.loc[df[f'race_{r}'] == 0, 'objective_response'].mean())
    results[f'race_{r}'] = out
results['inter_asian_osi'] = interaction_test('treatment_osimertinib', 'race_asian')
results['inter_black_pembro'] = interaction_test('treatment_pembrolizumab', 'race_black')

# ITER 17: Insurance / SES
for ins in ['medicare', 'private', 'medicaid', 'uninsured']:
    df[f'ins_{ins}'] = (df['insurance_type'] == ins).astype(int)
    out = logit_main(f'ins_{ins}')
    out['rate_in_group'] = float(df.loc[df[f'ins_{ins}'] == 1, 'objective_response'].mean())
    out['rate_other'] = float(df.loc[df[f'ins_{ins}'] == 0, 'objective_response'].mean())
    results[f'ins_{ins}'] = out
for v in ['rural_residence', 'education_years']:
    out = logit_main(v)
    results[f'soc_{v}'] = out

# ITER 18: Comorbidities
for v in ['autoimmune_disease', 'interstitial_lung_disease_history', 'copd',
          'chronic_kidney_disease', 'heart_failure', 'diabetes_mellitus', 'hypertension']:
    out = logit_main(v)
    results[f'comorb_{v}'] = out
results['inter_autoimm_pembro'] = interaction_test('treatment_pembrolizumab', 'autoimmune_disease')
results['inter_ild_pembro'] = interaction_test('treatment_pembrolizumab', 'interstitial_lung_disease_history')

# ITER 19: Prior therapies
for v in ['prior_chemotherapy', 'prior_radiation', 'prior_surgery', 'prior_immunotherapy',
          'prior_targeted_therapy', 'prior_lines_of_therapy', 'years_since_diagnosis']:
    out = logit_main(v)
    results[f'prior_{v}'] = out
results['inter_priorIO_pembro'] = interaction_test('treatment_pembrolizumab', 'prior_immunotherapy')

# ITER 20: Symptoms
for v in ['fatigue_grade', 'pain_nrs', 'dyspnea_grade', 'cough_grade', 'appetite_loss_grade']:
    out = logit_main(v)
    results[f'sym_{v}'] = out

# ITER 21: SNPs
snp_cols = [c for c in df.columns if c.startswith('snp_')]
for s in snp_cols:
    out = logit_main(s)
    results[f'snp_main_{s}'] = out

# ITER 22: SNP x treatment for first 6 SNPs
for s in snp_cols[:6]:
    for tx in ['treatment_pembrolizumab', 'treatment_sotorasib', 'treatment_olaparib', 'treatment_osimertinib']:
        try:
            results[f'inter_{s}_x_{tx}'] = interaction_test(tx, s)
        except Exception as e:
            results[f'inter_{s}_x_{tx}'] = {'error': str(e)}

# ITER 23: ECOG / brain mets x treatment
df['ecog_high'] = (df['ecog_ps'] >= 2).astype(int)
results['inter_ecoghigh_pembro'] = interaction_test('treatment_pembrolizumab', 'ecog_high')
results['inter_ecoghigh_osi'] = interaction_test('treatment_osimertinib', 'ecog_high')
results['inter_brain_pembro'] = interaction_test('treatment_pembrolizumab', 'has_brain_mets')
results['inter_brain_osi'] = interaction_test('treatment_osimertinib', 'has_brain_mets')

# ITER 24: Multivariable
formula = ('objective_response ~ treatment_pembrolizumab + treatment_sotorasib + '
           'treatment_olaparib + treatment_osimertinib + '
           'egfr_mutation + kras_g12c + brca2_mutation + pdl1_tps + tmb_high + '
           'stk11_mutation + keap1_mutation + ecog_ps + stage_iv + has_brain_mets + '
           'albumin_g_dl + ldh_u_l + nlr + age_years + sex_female + '
           'treatment_osimertinib:egfr_mutation + treatment_sotorasib:kras_g12c + '
           'treatment_olaparib:brca2_mutation + treatment_pembrolizumab:pdl1_tps + '
           'treatment_pembrolizumab:stk11_mutation')
mv = smf.logit(formula, data=df).fit(disp=False)
results['multivariable'] = {
    'params': {k: float(v) for k, v in mv.params.items()},
    'pvalues': {k: float(v) for k, v in mv.pvalues.items()},
    'odds_ratios': {k: float(np.exp(v)) for k, v in mv.params.items()},
}

# ITER 25: Final synthesis -- TMB x pembro stratified by PDL1
for pdl1_status in [0, 1]:
    sub = df[df['pdl1_high'] == pdl1_status][['treatment_pembrolizumab', 'tmb_high', 'objective_response']].copy().astype(float)
    sub['inter'] = sub['treatment_pembrolizumab'] * sub['tmb_high']
    X = sm.add_constant(sub[['treatment_pembrolizumab', 'tmb_high', 'inter']])
    m = sm.Logit(sub['objective_response'], X).fit(disp=False)
    results[f'tmb_pembro_in_pdl1_{pdl1_status}'] = {
        'inter_coef': float(m.params['inter']),
        'inter_p': float(m.pvalues['inter']),
        'n': int(len(sub)),
    }

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('Wrote results.json with', len(results), 'entries')
