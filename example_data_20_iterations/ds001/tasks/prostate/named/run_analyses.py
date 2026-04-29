"""Run analyses for ds001_prostate dataset."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
print(f"n={len(df)}")
results = {}

def ttest(group_var, outcome='pfs_months'):
    a = df.loc[df[group_var] == 1, outcome]
    b = df.loc[df[group_var] == 0, outcome]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        'mean_1': float(a.mean()),
        'mean_0': float(b.mean()),
        'diff': float(a.mean() - b.mean()),
        't': float(t),
        'p': float(p),
        'n_1': int(len(a)),
        'n_0': int(len(b)),
    }

def linreg(x_var, outcome='pfs_months', covariates=None):
    if covariates is None:
        covariates = []
    cols = [x_var] + covariates
    X = df[cols].copy()
    X = sm.add_constant(X)
    y = df[outcome]
    model = sm.OLS(y, X).fit()
    return {
        'beta': float(model.params[x_var]),
        'p': float(model.pvalues[x_var]),
        'se': float(model.bse[x_var]),
    }

def interaction_test(treatment, biomarker, outcome='pfs_months'):
    # Test treatment*biomarker interaction
    formula = f"{outcome} ~ {treatment} * {biomarker}"
    model = smf.ols(formula, data=df).fit()
    interaction_term = f"{treatment}:{biomarker}"
    return {
        'beta_treatment': float(model.params[treatment]),
        'beta_biomarker': float(model.params[biomarker]),
        'beta_interaction': float(model.params[interaction_term]),
        'p_interaction': float(model.pvalues[interaction_term]),
        'mean_t1_b1': float(df.loc[(df[treatment] == 1) & (df[biomarker] == 1), outcome].mean()),
        'mean_t1_b0': float(df.loc[(df[treatment] == 1) & (df[biomarker] == 0), outcome].mean()),
        'mean_t0_b1': float(df.loc[(df[treatment] == 0) & (df[biomarker] == 1), outcome].mean()),
        'mean_t0_b0': float(df.loc[(df[treatment] == 0) & (df[biomarker] == 0), outcome].mean()),
        'effect_in_b1': float(df.loc[(df[treatment] == 1) & (df[biomarker] == 1), outcome].mean() -
                              df.loc[(df[treatment] == 0) & (df[biomarker] == 1), outcome].mean()),
        'effect_in_b0': float(df.loc[(df[treatment] == 1) & (df[biomarker] == 0), outcome].mean() -
                              df.loc[(df[treatment] == 0) & (df[biomarker] == 0), outcome].mean()),
    }

# ============ ITER 1: Treatment main effects ============
treatments = ['treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel',
              'treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab']
for t in treatments:
    results[f'main_{t}'] = ttest(t)
    print(t, results[f'main_{t}'])

# ============ ITER 2: Biomarker main effects ============
biomarkers = ['mcrpc','visceral_mets','brca2_mutation','ar_v7_positive','msi_high','psma_high',
              'liver_mets','bone_mets','adrenal_mets']
for b in biomarkers:
    results[f'main_{b}'] = ttest(b)
    print(b, results[f'main_{b}'])

# ============ ITER 3: Continuous clinical features ============
cont_features = ['age_years','ecog_ps','gleason_score','psa_ng_ml','albumin_g_dl','ldh_u_l',
                 'crp_mg_l','nlr','hemoglobin_g_dl','alkaline_phosphatase_u_l',
                 'weight_loss_pct_6mo','prior_lines_of_therapy','years_since_diagnosis',
                 'pain_nrs','fatigue_grade','bmi','smoking_pack_years','education_years']
for c in cont_features:
    results[f'lin_{c}'] = linreg(c)
    print(c, results[f'lin_{c}'])

# ============ ITER 4-6: Treatment × biomarker interactions (key clinical hypotheses) ============
# Olaparib × BRCA2: PARP inhibitor benefit in BRCA2-mutated
# Lu177-PSMA × PSMA-high: PSMA-targeted therapy benefit in PSMA-high
# Pembrolizumab × MSI-high: PD-1 inhibitor benefit in MSI-high
# Enzalutamide × AR-V7: AR-V7 confers resistance
# Abiraterone × AR-V7: AR-V7 confers resistance
key_interactions = [
    ('treatment_olaparib','brca2_mutation'),
    ('treatment_lu177_psma','psma_high'),
    ('treatment_pembrolizumab','msi_high'),
    ('treatment_enzalutamide','ar_v7_positive'),
    ('treatment_abiraterone','ar_v7_positive'),
    ('treatment_olaparib','ar_v7_positive'),
    ('treatment_pembrolizumab','brca2_mutation'),
    ('treatment_docetaxel','visceral_mets'),
    ('treatment_enzalutamide','mcrpc'),
    ('treatment_abiraterone','mcrpc'),
]
for t, b in key_interactions:
    key = f'inter_{t}_{b}'
    results[key] = interaction_test(t, b)
    print(key, results[key])

# ============ ITER 7: SDOH effects ============
# Race/ethnicity: dummy
race_dummies = pd.get_dummies(df['race_ethnicity'], prefix='race', drop_first=False).astype(int)
df_race = pd.concat([df[['pfs_months']], race_dummies], axis=1)
# Use white as reference
for race_cat in ['black','hispanic','asian','other']:
    col = f'race_{race_cat}'
    if col in df_race.columns:
        a = df.loc[df['race_ethnicity'] == race_cat, 'pfs_months']
        b = df.loc[df['race_ethnicity'] == 'white', 'pfs_months']
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results[f'sdoh_race_{race_cat}_vs_white'] = {
            'mean_cat': float(a.mean()),
            'mean_white': float(b.mean()),
            'diff': float(a.mean() - b.mean()),
            'p': float(p),
            'n_cat': int(len(a)),
            'n_white': int(len(b)),
        }
        print(f'race_{race_cat}_vs_white', results[f'sdoh_race_{race_cat}_vs_white'])

# Insurance
for ins_cat in ['medicaid','uninsured','medicare']:
    a = df.loc[df['insurance_type'] == ins_cat, 'pfs_months']
    b = df.loc[df['insurance_type'] == 'private', 'pfs_months']
    t, p = stats.ttest_ind(a, b, equal_var=False)
    results[f'sdoh_ins_{ins_cat}_vs_private'] = {
        'mean_cat': float(a.mean()),
        'mean_priv': float(b.mean()),
        'diff': float(a.mean() - b.mean()),
        'p': float(p),
        'n_cat': int(len(a)),
        'n_priv': int(len(b)),
    }
    print(f'ins_{ins_cat}_vs_private', results[f'sdoh_ins_{ins_cat}_vs_private'])

# Rural residence
results['sdoh_rural'] = ttest('rural_residence')
print('rural', results['sdoh_rural'])

# ============ ITER 8: Comorbidities ============
comorbidities = ['diabetes_mellitus','hypertension','copd','chronic_kidney_disease',
                 'heart_failure','coronary_artery_disease','atrial_fibrillation']
for c in comorbidities:
    results[f'comorbid_{c}'] = ttest(c)
    print(c, results[f'comorbid_{c}'])

# ============ ITER 9: Adjusted treatment effects ============
# Multivariable regression: each treatment adjusted for ECOG, age, mcrpc, visceral_mets, ldh, albumin
covariates = ['age_years','ecog_ps','mcrpc','visceral_mets','ldh_u_l','albumin_g_dl','psa_ng_ml','gleason_score']
for t in treatments:
    results[f'adj_{t}'] = linreg(t, covariates=covariates)
    print(f'adj_{t}', results[f'adj_{t}'])

# ============ ITER 10: SNP main effects ============
snps = [c for c in df.columns if c.startswith('snp_')]
snp_results = []
for s in snps:
    r = ttest(s)
    snp_results.append((s, r['diff'], r['p']))
    results[f'snp_{s}'] = r
# Print most significant SNPs
snp_results.sort(key=lambda x: x[2])
print('Top 5 SNPs by p-value:')
for s, d, p in snp_results[:5]:
    print(f'  {s}: diff={d:.4f}, p={p:.4f}')

# ============ ITER 11: Symptom and PS effects ============
symptoms = ['fatigue_grade','pain_nrs','dyspnea_grade','cough_grade','appetite_loss_grade']
for s in symptoms:
    results[f'symptom_{s}'] = linreg(s)
    print(s, results[f'symptom_{s}'])

# ============ ITER 12: Composite/multivariable model ============
# Full multivariable model with all key features
key_features = ['age_years','ecog_ps','mcrpc','visceral_mets','liver_mets','bone_mets','psa_ng_ml',
                'gleason_score','brca2_mutation','ar_v7_positive','msi_high','psma_high',
                'albumin_g_dl','ldh_u_l','crp_mg_l','nlr','hemoglobin_g_dl','weight_loss_pct_6mo',
                'prior_lines_of_therapy','treatment_enzalutamide','treatment_abiraterone',
                'treatment_docetaxel','treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab',
                'pain_nrs','fatigue_grade']
X = sm.add_constant(df[key_features])
y = df['pfs_months']
mv_model = sm.OLS(y, X).fit()
print('\n\nMultivariable model R^2:', mv_model.rsquared)
print(mv_model.summary().tables[1])

mv_results = {}
for var in key_features:
    mv_results[var] = {
        'beta': float(mv_model.params[var]),
        'p': float(mv_model.pvalues[var]),
        'se': float(mv_model.bse[var]),
    }
results['multivariable_model'] = {
    'r_squared': float(mv_model.rsquared),
    'coefs': mv_results,
}

# Save all results
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\n\nSaved analysis_results.json")
