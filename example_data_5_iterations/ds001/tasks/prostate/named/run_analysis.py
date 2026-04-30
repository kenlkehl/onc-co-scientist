"""Iterative analysis script for ds001_prostate. Outputs JSON + text summaries."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
print(f"Loaded {df.shape}")

results = {}

def mean_diff_test(grp1, grp0, label):
    """Welch t-test for difference in PFS means; effect = mean(grp1) - mean(grp0)."""
    a = df.loc[grp1, 'pfs_months'].values
    b = df.loc[grp0, 'pfs_months'].values
    if len(a) < 5 or len(b) < 5:
        return {'label': label, 'n1': len(a), 'n0': len(b), 'mean1': float(np.mean(a)) if len(a) else None,
                'mean0': float(np.mean(b)) if len(b) else None, 'diff': None, 'p': None, 'sig': False}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {'label': label, 'n1': len(a), 'n0': len(b), 'mean1': float(np.mean(a)),
            'mean0': float(np.mean(b)), 'diff': float(np.mean(a) - np.mean(b)),
            'p': float(p), 'sig': bool(p < 0.05)}

def cont_corr(col, label):
    """Pearson correlation between continuous predictor and pfs_months."""
    r, p = stats.pearsonr(df[col].values, df['pfs_months'].values)
    return {'label': label, 'r': float(r), 'p': float(p), 'sig': bool(p < 0.05)}

def ols_test(formula, focal_term, label):
    """Run OLS on PFS, return coef + p for focal term."""
    model = smf.ols(formula, data=df).fit()
    coef = float(model.params[focal_term])
    p = float(model.pvalues[focal_term])
    return {'label': label, 'formula': formula, 'term': focal_term, 'coef': coef, 'p': p, 'sig': bool(p < 0.05)}

# ====================== ITERATION 1: prognostic main effects ======================
it1 = {}
it1['ecog_corr'] = cont_corr('ecog_ps', 'ECOG vs PFS')
it1['mcrpc'] = mean_diff_test(df['mcrpc']==1, df['mcrpc']==0, 'mCRPC vs hormone-sensitive')
it1['visceral'] = mean_diff_test(df['visceral_mets']==1, df['visceral_mets']==0, 'visceral mets')
it1['psa'] = cont_corr('psa_ng_ml', 'PSA vs PFS')
it1['gleason'] = cont_corr('gleason_score', 'Gleason vs PFS')
it1['albumin'] = cont_corr('albumin_g_dl', 'albumin vs PFS')
it1['ldh'] = cont_corr('ldh_u_l', 'LDH vs PFS')
it1['hgb'] = cont_corr('hemoglobin_g_dl', 'hemoglobin vs PFS')
it1['alp'] = cont_corr('alkaline_phosphatase_u_l', 'ALP vs PFS')
it1['bone_mets'] = mean_diff_test(df['bone_mets']==1, df['bone_mets']==0, 'bone mets')
it1['liver_mets'] = mean_diff_test(df['liver_mets']==1, df['liver_mets']==0, 'liver mets')
it1['age'] = cont_corr('age_years', 'age vs PFS')
results['it1'] = it1
print("IT1 done")

# ====================== ITERATION 2: treatment main effects ======================
it2 = {}
for t in ['treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel',
          'treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab']:
    it2[t] = mean_diff_test(df[t]==1, df[t]==0, t)
results['it2'] = it2
print("IT2 done")

# ====================== ITERATION 3: biomarker-treatment interactions ======================
it3 = {}
# BRCA2 × olaparib
it3['olaparib_in_brca2pos'] = mean_diff_test(
    (df['treatment_olaparib']==1) & (df['brca2_mutation']==1),
    (df['treatment_olaparib']==0) & (df['brca2_mutation']==1),
    'olaparib effect within BRCA2+')
it3['olaparib_in_brca2neg'] = mean_diff_test(
    (df['treatment_olaparib']==1) & (df['brca2_mutation']==0),
    (df['treatment_olaparib']==0) & (df['brca2_mutation']==0),
    'olaparib effect within BRCA2-')
m = smf.ols('pfs_months ~ treatment_olaparib * brca2_mutation', data=df).fit()
it3['olaparib_brca2_interaction'] = {'label':'olaparib*BRCA2 interaction',
    'coef': float(m.params['treatment_olaparib:brca2_mutation']),
    'p': float(m.pvalues['treatment_olaparib:brca2_mutation']),
    'sig': bool(m.pvalues['treatment_olaparib:brca2_mutation'] < 0.05)}

# PSMA × Lu177
it3['lu177_in_psmahigh'] = mean_diff_test(
    (df['treatment_lu177_psma']==1) & (df['psma_high']==1),
    (df['treatment_lu177_psma']==0) & (df['psma_high']==1),
    'Lu177 in PSMA-high')
it3['lu177_in_psmalow'] = mean_diff_test(
    (df['treatment_lu177_psma']==1) & (df['psma_high']==0),
    (df['treatment_lu177_psma']==0) & (df['psma_high']==0),
    'Lu177 in PSMA-low')
m = smf.ols('pfs_months ~ treatment_lu177_psma * psma_high', data=df).fit()
it3['lu177_psma_interaction'] = {'label':'Lu177*PSMA interaction',
    'coef': float(m.params['treatment_lu177_psma:psma_high']),
    'p': float(m.pvalues['treatment_lu177_psma:psma_high']),
    'sig': bool(m.pvalues['treatment_lu177_psma:psma_high'] < 0.05)}

# MSI × pembrolizumab
it3['pembro_in_msih'] = mean_diff_test(
    (df['treatment_pembrolizumab']==1) & (df['msi_high']==1),
    (df['treatment_pembrolizumab']==0) & (df['msi_high']==1),
    'pembro in MSI-high')
it3['pembro_in_mss'] = mean_diff_test(
    (df['treatment_pembrolizumab']==1) & (df['msi_high']==0),
    (df['treatment_pembrolizumab']==0) & (df['msi_high']==0),
    'pembro in MSS')
m = smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high', data=df).fit()
it3['pembro_msi_interaction'] = {'label':'pembro*MSI interaction',
    'coef': float(m.params['treatment_pembrolizumab:msi_high']),
    'p': float(m.pvalues['treatment_pembrolizumab:msi_high']),
    'sig': bool(m.pvalues['treatment_pembrolizumab:msi_high'] < 0.05)}

# AR-V7 × enzalutamide
it3['enza_in_arv7pos'] = mean_diff_test(
    (df['treatment_enzalutamide']==1) & (df['ar_v7_positive']==1),
    (df['treatment_enzalutamide']==0) & (df['ar_v7_positive']==1),
    'enza in AR-V7+')
it3['enza_in_arv7neg'] = mean_diff_test(
    (df['treatment_enzalutamide']==1) & (df['ar_v7_positive']==0),
    (df['treatment_enzalutamide']==0) & (df['ar_v7_positive']==0),
    'enza in AR-V7-')
m = smf.ols('pfs_months ~ treatment_enzalutamide * ar_v7_positive', data=df).fit()
it3['enza_arv7_interaction'] = {'label':'enza*AR-V7 interaction',
    'coef': float(m.params['treatment_enzalutamide:ar_v7_positive']),
    'p': float(m.pvalues['treatment_enzalutamide:ar_v7_positive']),
    'sig': bool(m.pvalues['treatment_enzalutamide:ar_v7_positive'] < 0.05)}

# AR-V7 × abiraterone
it3['abi_in_arv7pos'] = mean_diff_test(
    (df['treatment_abiraterone']==1) & (df['ar_v7_positive']==1),
    (df['treatment_abiraterone']==0) & (df['ar_v7_positive']==1),
    'abi in AR-V7+')
it3['abi_in_arv7neg'] = mean_diff_test(
    (df['treatment_abiraterone']==1) & (df['ar_v7_positive']==0),
    (df['treatment_abiraterone']==0) & (df['ar_v7_positive']==0),
    'abi in AR-V7-')
m = smf.ols('pfs_months ~ treatment_abiraterone * ar_v7_positive', data=df).fit()
it3['abi_arv7_interaction'] = {'label':'abi*AR-V7 interaction',
    'coef': float(m.params['treatment_abiraterone:ar_v7_positive']),
    'p': float(m.pvalues['treatment_abiraterone:ar_v7_positive']),
    'sig': bool(m.pvalues['treatment_abiraterone:ar_v7_positive'] < 0.05)}

results['it3'] = it3
print("IT3 done")

# ====================== ITERATION 4: subgroup heterogeneity by clinical state ======================
it4 = {}
# treatment effects within mCRPC vs HSPC
for t in ['treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel','treatment_lu177_psma']:
    it4[f'{t}_in_mcrpc'] = mean_diff_test(
        (df[t]==1) & (df['mcrpc']==1), (df[t]==0) & (df['mcrpc']==1), f'{t} in mCRPC')
    it4[f'{t}_in_hspc'] = mean_diff_test(
        (df[t]==1) & (df['mcrpc']==0), (df[t]==0) & (df['mcrpc']==0), f'{t} in HSPC')

# ECOG groups
for ecog in [0, 1, 2]:
    sub = df[df['ecog_ps']==ecog]
    it4[f'pfs_ecog_{ecog}'] = {'label': f'PFS in ECOG={ecog}', 'n': int(len(sub)),
        'mean': float(sub['pfs_months'].mean()), 'median': float(sub['pfs_months'].median())}
results['it4'] = it4
print("IT4 done")

# ====================== ITERATION 5: inflammation/nutrition markers ======================
it5 = {}
for col in ['nlr','crp_mg_l','weight_loss_pct_6mo','pain_nrs','fatigue_grade',
            'appetite_loss_grade','dyspnea_grade','platelets_k_ul','wbc_k_ul',
            'calcium_mg_dl','sodium_meq_l','creatinine_mg_dl','bun_mg_dl','total_bilirubin_mg_dl',
            'ast_u_l','alt_u_l','bmi','spo2_pct']:
    it5[col] = cont_corr(col, f'{col} vs PFS')
results['it5'] = it5
print("IT5 done")

# ====================== ITERATION 6: comorbidities & demographics ======================
it6 = {}
for c in ['diabetes_mellitus','hypertension','copd','chronic_kidney_disease',
          'heart_failure','coronary_artery_disease','atrial_fibrillation',
          'venous_thromboembolism_history','autoimmune_disease','depression_anxiety_diagnosis']:
    it6[c] = mean_diff_test(df[c]==1, df[c]==0, c)
# race/ethnicity ANOVA
groups = [df.loc[df['race_ethnicity']==r,'pfs_months'].values for r in df['race_ethnicity'].unique()]
F, p = stats.f_oneway(*groups)
race_means = df.groupby('race_ethnicity')['pfs_months'].mean().to_dict()
it6['race_anova'] = {'label':'PFS across race/ethnicity','F':float(F),'p':float(p),
                     'means':{k:float(v) for k,v in race_means.items()},'sig':bool(p<0.05)}
# insurance ANOVA
groups = [df.loc[df['insurance_type']==r,'pfs_months'].values for r in df['insurance_type'].unique()]
F, p = stats.f_oneway(*groups)
ins_means = df.groupby('insurance_type')['pfs_months'].mean().to_dict()
it6['ins_anova'] = {'label':'PFS across insurance','F':float(F),'p':float(p),
                    'means':{k:float(v) for k,v in ins_means.items()},'sig':bool(p<0.05)}
it6['rural'] = mean_diff_test(df['rural_residence']==1, df['rural_residence']==0, 'rural residence')
it6['smoking'] = cont_corr('smoking_pack_years', 'smoking pack-years')
it6['education'] = cont_corr('education_years', 'education years')
results['it6'] = it6
print("IT6 done")

# ====================== ITERATION 7: prior therapy and disease history ======================
it7 = {}
it7['prior_lines'] = cont_corr('prior_lines_of_therapy', 'prior lines of therapy')
it7['years_since_dx'] = cont_corr('years_since_diagnosis', 'years since diagnosis')
for c in ['prior_chemotherapy','prior_radiation','prior_surgery','prior_immunotherapy','prior_targeted_therapy']:
    it7[c] = mean_diff_test(df[c]==1, df[c]==0, c)
results['it7'] = it7
print("IT7 done")

# ====================== ITERATION 8: SNP and other genomic main effects ======================
it8 = {}
snps = [c for c in df.columns if c.startswith('snp_')]
for s in snps:
    it8[s] = mean_diff_test(df[s]==1, df[s]==0, s)
for g in ['tp53_mutation','pten_loss','her2_amplification','cdkn2a_loss','pik3ca_mutation',
          'fgfr_alteration','ntrk_fusion','braf_v600e','ros1_fusion','ret_fusion',
          'met_exon14_skipping','nrg1_fusion','keap1_mutation']:
    it8[g] = mean_diff_test(df[g]==1, df[g]==0, g)
results['it8'] = it8
print("IT8 done")

# ====================== ITERATION 9: multivariable model ======================
it9 = {}
formula = ('pfs_months ~ age_years + ecog_ps + mcrpc + visceral_mets + bone_mets + liver_mets'
           ' + np.log1p(psa_ng_ml) + gleason_score + albumin_g_dl + np.log1p(ldh_u_l)'
           ' + hemoglobin_g_dl + np.log1p(alkaline_phosphatase_u_l) + nlr + crp_mg_l + weight_loss_pct_6mo'
           ' + treatment_enzalutamide + treatment_abiraterone + treatment_docetaxel'
           ' + treatment_olaparib + treatment_lu177_psma + treatment_pembrolizumab'
           ' + brca2_mutation + ar_v7_positive + msi_high + psma_high + tp53_mutation + pten_loss'
           ' + prior_lines_of_therapy')
m = smf.ols(formula, data=df).fit()
for k, v in m.params.items():
    it9[k] = {'coef': float(v), 'p': float(m.pvalues[k]), 'sig': bool(m.pvalues[k] < 0.05)}
it9['_r2'] = float(m.rsquared)
it9['_n'] = int(m.nobs)
results['it9'] = it9
print(f"IT9 done R2={m.rsquared:.4f}")

# ====================== ITERATION 10: targeted interactions among top predictors ======================
it10 = {}
# Refined interactions
m = smf.ols('pfs_months ~ treatment_olaparib * brca2_mutation + ecog_ps + mcrpc + visceral_mets + albumin_g_dl + np.log1p(ldh_u_l)', data=df).fit()
it10['olaparib_brca2_adj'] = {'coef': float(m.params['treatment_olaparib:brca2_mutation']),
    'p': float(m.pvalues['treatment_olaparib:brca2_mutation']),
    'sig': bool(m.pvalues['treatment_olaparib:brca2_mutation']<0.05)}
m = smf.ols('pfs_months ~ treatment_lu177_psma * psma_high + ecog_ps + mcrpc + visceral_mets + albumin_g_dl + np.log1p(ldh_u_l)', data=df).fit()
it10['lu177_psma_adj'] = {'coef': float(m.params['treatment_lu177_psma:psma_high']),
    'p': float(m.pvalues['treatment_lu177_psma:psma_high']),
    'sig': bool(m.pvalues['treatment_lu177_psma:psma_high']<0.05)}
m = smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high + ecog_ps + mcrpc + visceral_mets + albumin_g_dl + np.log1p(ldh_u_l)', data=df).fit()
it10['pembro_msi_adj'] = {'coef': float(m.params['treatment_pembrolizumab:msi_high']),
    'p': float(m.pvalues['treatment_pembrolizumab:msi_high']),
    'sig': bool(m.pvalues['treatment_pembrolizumab:msi_high']<0.05)}

# Three-way: enzalutamide x AR-V7 x mCRPC
m = smf.ols('pfs_months ~ treatment_enzalutamide * ar_v7_positive + ecog_ps + mcrpc + visceral_mets + albumin_g_dl', data=df).fit()
it10['enza_arv7_adj'] = {'coef': float(m.params['treatment_enzalutamide:ar_v7_positive']),
    'p': float(m.pvalues['treatment_enzalutamide:ar_v7_positive']),
    'sig': bool(m.pvalues['treatment_enzalutamide:ar_v7_positive']<0.05)}

# Docetaxel × visceral mets (chemo benefit in visceral)
m = smf.ols('pfs_months ~ treatment_docetaxel * visceral_mets + ecog_ps + mcrpc + albumin_g_dl + np.log1p(ldh_u_l)', data=df).fit()
it10['docetaxel_visceral_adj'] = {'coef': float(m.params['treatment_docetaxel:visceral_mets']),
    'p': float(m.pvalues['treatment_docetaxel:visceral_mets']),
    'sig': bool(m.pvalues['treatment_docetaxel:visceral_mets']<0.05)}

# Treatment x ECOG (ECOG-2 patients respond worse?)
m = smf.ols('pfs_months ~ treatment_docetaxel * ecog_ps + mcrpc + visceral_mets + albumin_g_dl', data=df).fit()
it10['docetaxel_ecog_adj'] = {'coef': float(m.params['treatment_docetaxel:ecog_ps']),
    'p': float(m.pvalues['treatment_docetaxel:ecog_ps']),
    'sig': bool(m.pvalues['treatment_docetaxel:ecog_ps']<0.05)}

# albumin x mcrpc
m = smf.ols('pfs_months ~ albumin_g_dl * mcrpc + ecog_ps + visceral_mets', data=df).fit()
it10['albumin_mcrpc'] = {'coef': float(m.params['albumin_g_dl:mcrpc']),
    'p': float(m.pvalues['albumin_g_dl:mcrpc']),
    'sig': bool(m.pvalues['albumin_g_dl:mcrpc']<0.05)}

# nlr x crp (inflammation composite)
m = smf.ols('pfs_months ~ nlr * crp_mg_l + ecog_ps + mcrpc + albumin_g_dl', data=df).fit()
it10['nlr_crp'] = {'coef': float(m.params['nlr:crp_mg_l']),
    'p': float(m.pvalues['nlr:crp_mg_l']),
    'sig': bool(m.pvalues['nlr:crp_mg_l']<0.05)}

results['it10'] = it10
print("IT10 done")

with open('all_results.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print("Saved all_results.json")
