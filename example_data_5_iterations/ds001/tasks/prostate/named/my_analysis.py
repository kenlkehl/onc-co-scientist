"""Iterative analysis of ds001_prostate. Outputs results to stdout."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
print(f"N = {len(df)}; PFS mean = {df['pfs_months'].mean():.3f}, median = {df['pfs_months'].median():.3f}\n")

results = {}

def lin_reg(formula, label):
    m = smf.ols(formula, data=df).fit()
    return m

def report(label, est, p, sig=None):
    if sig is None:
        sig = (p < 0.05) if p is not None else None
    s = "*" if sig else " "
    print(f"  {s} {label}: est={est:.4f}, p={p:.4g}")
    results[label] = {'est': float(est), 'p': float(p), 'sig': bool(sig)}


# ===== ITERATION 1: Main treatment effects on PFS =====
print("=" * 70)
print("ITERATION 1: Main treatment effects on PFS")
print("=" * 70)
treatments = ['treatment_enzalutamide', 'treatment_abiraterone', 'treatment_docetaxel',
              'treatment_olaparib', 'treatment_lu177_psma', 'treatment_pembrolizumab']
for t in treatments:
    m1 = df.loc[df[t] == 1, 'pfs_months']
    m0 = df.loc[df[t] == 0, 'pfs_months']
    diff = m1.mean() - m0.mean()
    tres = stats.ttest_ind(m1, m0, equal_var=False)
    report(f"PFS_diff_{t}", diff, tres.pvalue)


# ===== ITERATION 2: Prognostic / clinical features =====
print()
print("=" * 70)
print("ITERATION 2: Prognostic features on PFS")
print("=" * 70)
# ECOG
for ps in [1, 2]:
    sub = df.loc[df['ecog_ps'] == ps, 'pfs_months']
    ref = df.loc[df['ecog_ps'] == 0, 'pfs_months']
    diff = sub.mean() - ref.mean()
    tres = stats.ttest_ind(sub, ref, equal_var=False)
    report(f"PFS_ECOG{ps}_vs_0", diff, tres.pvalue)
# Visceral mets
for col in ['visceral_mets', 'liver_mets', 'bone_mets', 'mcrpc']:
    m1 = df.loc[df[col] == 1, 'pfs_months']
    m0 = df.loc[df[col] == 0, 'pfs_months']
    tres = stats.ttest_ind(m1, m0, equal_var=False)
    report(f"PFS_diff_{col}", m1.mean() - m0.mean(), tres.pvalue)
# Continuous: regression coefficient (per unit)
for col in ['age_years', 'albumin_g_dl', 'ldh_u_l', 'hemoglobin_g_dl',
            'alkaline_phosphatase_u_l', 'crp_mg_l', 'nlr', 'psa_ng_ml',
            'gleason_score', 'weight_loss_pct_6mo']:
    m = lin_reg(f"pfs_months ~ {col}", col)
    est = m.params[col]
    p = m.pvalues[col]
    report(f"PFS_per_unit_{col}", est, p)


# ===== ITERATION 3: Biomarker-treatment interactions =====
print()
print("=" * 70)
print("ITERATION 3: Biomarker-treatment interactions")
print("=" * 70)
# Olaparib x BRCA2
m = lin_reg("pfs_months ~ treatment_olaparib * brca2_mutation", "olaparib_x_brca2")
est = m.params['treatment_olaparib:brca2_mutation']
p = m.pvalues['treatment_olaparib:brca2_mutation']
report("interaction_olaparib_x_brca2", est, p)
# Lu177-PSMA x PSMA-high
m = lin_reg("pfs_months ~ treatment_lu177_psma * psma_high", "lu177_x_psma")
est = m.params['treatment_lu177_psma:psma_high']
p = m.pvalues['treatment_lu177_psma:psma_high']
report("interaction_lu177_x_psma_high", est, p)
# Pembrolizumab x MSI-high
m = lin_reg("pfs_months ~ treatment_pembrolizumab * msi_high", "pembro_x_msi")
est = m.params['treatment_pembrolizumab:msi_high']
p = m.pvalues['treatment_pembrolizumab:msi_high']
report("interaction_pembro_x_msi_high", est, p)

# Subgroup means for these
for tcol, bcol in [('treatment_olaparib','brca2_mutation'),
                    ('treatment_lu177_psma','psma_high'),
                    ('treatment_pembrolizumab','msi_high')]:
    print(f"\n  Subgroup PFS for {tcol} x {bcol}:")
    for tv in [0,1]:
        for bv in [0,1]:
            sub = df[(df[tcol]==tv) & (df[bcol]==bv)]
            print(f"    {tcol}={tv}, {bcol}={bv}: n={len(sub)}, mean={sub['pfs_months'].mean():.3f}")


# ===== ITERATION 4: AR-V7 resistance to ARSI =====
print()
print("=" * 70)
print("ITERATION 4: AR-V7 modifies enzalutamide/abiraterone effect")
print("=" * 70)
m = lin_reg("pfs_months ~ treatment_enzalutamide * ar_v7_positive", "enza_x_arv7")
est = m.params['treatment_enzalutamide:ar_v7_positive']
p = m.pvalues['treatment_enzalutamide:ar_v7_positive']
report("interaction_enzalutamide_x_arv7", est, p)

m = lin_reg("pfs_months ~ treatment_abiraterone * ar_v7_positive", "abi_x_arv7")
est = m.params['treatment_abiraterone:ar_v7_positive']
p = m.pvalues['treatment_abiraterone:ar_v7_positive']
report("interaction_abiraterone_x_arv7", est, p)

# Subgroup PFS for ARV7
print("\n  Subgroup PFS - enzalutamide x AR-V7:")
for tv in [0,1]:
    for bv in [0,1]:
        sub = df[(df['treatment_enzalutamide']==tv) & (df['ar_v7_positive']==bv)]
        print(f"    enza={tv}, arv7={bv}: n={len(sub)}, mean={sub['pfs_months'].mean():.3f}")
print("\n  Subgroup PFS - abiraterone x AR-V7:")
for tv in [0,1]:
    for bv in [0,1]:
        sub = df[(df['treatment_abiraterone']==tv) & (df['ar_v7_positive']==bv)]
        print(f"    abi={tv}, arv7={bv}: n={len(sub)}, mean={sub['pfs_months'].mean():.3f}")


# ===== ITERATION 5: ARSI in mCRPC vs HSPC =====
print()
print("=" * 70)
print("ITERATION 5: Treatment effect modified by mCRPC status")
print("=" * 70)
for tcol in ['treatment_enzalutamide', 'treatment_abiraterone', 'treatment_docetaxel']:
    m = lin_reg(f"pfs_months ~ {tcol} * mcrpc", "mcrpc_int")
    est = m.params[f'{tcol}:mcrpc']
    p = m.pvalues[f'{tcol}:mcrpc']
    report(f"interaction_{tcol}_x_mcrpc", est, p)

# Gleason
for tcol in ['treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel']:
    m = lin_reg(f"pfs_months ~ {tcol} * gleason_score", "gleason")
    est = m.params[f'{tcol}:gleason_score']
    p = m.pvalues[f'{tcol}:gleason_score']
    report(f"interaction_{tcol}_x_gleason", est, p)


# ===== ITERATION 6: Inflammatory markers and PFS, plus interaction with chemo =====
print()
print("=" * 70)
print("ITERATION 6: Inflammation markers; pembrolizumab x ECOG/PD markers")
print("=" * 70)
# CRP and NLR adjusted for treatments
m = lin_reg("pfs_months ~ crp_mg_l + nlr + albumin_g_dl + ldh_u_l + ecog_ps + " +
            " + ".join(treatments), "multi_inflammatory")
for c in ['crp_mg_l','nlr','albumin_g_dl','ldh_u_l','ecog_ps']:
    report(f"adj_PFS_per_unit_{c}", m.params[c], m.pvalues[c])

# Docetaxel x visceral mets / liver mets
for tcol in ['treatment_docetaxel']:
    for bcol in ['visceral_mets','liver_mets','bone_mets']:
        m = lin_reg(f"pfs_months ~ {tcol} * {bcol}", "subgrp")
        est = m.params[f'{tcol}:{bcol}']
        p = m.pvalues[f'{tcol}:{bcol}']
        report(f"interaction_{tcol}_x_{bcol}", est, p)


# ===== ITERATION 7: Comorbidities =====
print()
print("=" * 70)
print("ITERATION 7: Comorbidities effect on PFS")
print("=" * 70)
comorbs = ['diabetes_mellitus','hypertension','copd','chronic_kidney_disease',
           'heart_failure','coronary_artery_disease','atrial_fibrillation',
           'venous_thromboembolism_history','autoimmune_disease',
           'depression_anxiety_diagnosis','prior_malignancy']
for c in comorbs:
    m1 = df.loc[df[c]==1, 'pfs_months']
    m0 = df.loc[df[c]==0, 'pfs_months']
    tres = stats.ttest_ind(m1, m0, equal_var=False)
    report(f"PFS_diff_{c}", m1.mean()-m0.mean(), tres.pvalue)


# ===== ITERATION 8: Demographic disparities =====
print()
print("=" * 70)
print("ITERATION 8: Demographic / socioeconomic effects")
print("=" * 70)
# race
for race in ['white','hispanic','black','asian','other']:
    sub = df.loc[df['race_ethnicity']==race, 'pfs_months']
    ref = df.loc[df['race_ethnicity']!=race, 'pfs_months']
    tres = stats.ttest_ind(sub, ref, equal_var=False)
    report(f"PFS_diff_race_{race}_vs_other", sub.mean()-ref.mean(), tres.pvalue)
# insurance
for ins in ['medicare','private','medicaid','uninsured']:
    sub = df.loc[df['insurance_type']==ins, 'pfs_months']
    ref = df.loc[df['insurance_type']!=ins, 'pfs_months']
    tres = stats.ttest_ind(sub, ref, equal_var=False)
    report(f"PFS_diff_insurance_{ins}_vs_other", sub.mean()-ref.mean(), tres.pvalue)
# rural
m1 = df.loc[df['rural_residence']==1, 'pfs_months']
m0 = df.loc[df['rural_residence']==0, 'pfs_months']
tres = stats.ttest_ind(m1, m0, equal_var=False)
report("PFS_diff_rural_residence", m1.mean()-m0.mean(), tres.pvalue)
# education years
m = lin_reg("pfs_months ~ education_years", "ed")
report("PFS_per_year_education", m.params['education_years'], m.pvalues['education_years'])
# smoking
m = lin_reg("pfs_months ~ smoking_pack_years", "smoke")
report("PFS_per_pack_year_smoking", m.params['smoking_pack_years'], m.pvalues['smoking_pack_years'])


# ===== ITERATION 9: SNP screen =====
print()
print("=" * 70)
print("ITERATION 9: SNP screen for PFS associations")
print("=" * 70)
snp_cols = [c for c in df.columns if c.startswith('snp_')]
snp_results = []
for s in snp_cols:
    m = lin_reg(f"pfs_months ~ {s}", "snp")
    snp_results.append((s, m.params[s], m.pvalues[s]))
snp_results.sort(key=lambda x: x[2])
print("  Top 10 SNPs by p-value:")
for s, est, p in snp_results[:10]:
    report(f"PFS_per_allele_{s}", est, p)
# Note: with 25 SNPs tested, Bonferroni-adjusted alpha = 0.05/25 = 0.002
print(f"\n  Bonferroni threshold = 0.05/{len(snp_cols)} = {0.05/len(snp_cols):.4g}")


# ===== ITERATION 10: Multivariable adjusted model =====
print()
print("=" * 70)
print("ITERATION 10: Multivariable model + key interactions")
print("=" * 70)
formula = ("pfs_months ~ age_years + ecog_ps + mcrpc + visceral_mets + bone_mets + "
           "liver_mets + albumin_g_dl + ldh_u_l + hemoglobin_g_dl + crp_mg_l + nlr + "
           "alkaline_phosphatase_u_l + psa_ng_ml + gleason_score + weight_loss_pct_6mo + "
           "treatment_enzalutamide + treatment_abiraterone + treatment_docetaxel + "
           "treatment_olaparib + treatment_lu177_psma + treatment_pembrolizumab + "
           "treatment_olaparib:brca2_mutation + brca2_mutation + "
           "treatment_lu177_psma:psma_high + psma_high + "
           "treatment_pembrolizumab:msi_high + msi_high + "
           "treatment_enzalutamide:ar_v7_positive + ar_v7_positive")
m = lin_reg(formula, "full")
print(f"\n  R-squared: {m.rsquared:.4f}, n={int(m.nobs)}")
key_terms = ['ecog_ps','mcrpc','visceral_mets','liver_mets','albumin_g_dl','ldh_u_l',
             'hemoglobin_g_dl','crp_mg_l','nlr','psa_ng_ml','gleason_score',
             'weight_loss_pct_6mo','treatment_olaparib:brca2_mutation',
             'treatment_lu177_psma:psma_high','treatment_pembrolizumab:msi_high',
             'treatment_enzalutamide:ar_v7_positive']
for t in key_terms:
    if t in m.params.index:
        report(f"adj_{t}", m.params[t], m.pvalues[t])


# Save raw results to JSON
with open('my_results.json','w') as f:
    json.dump(results, f, indent=2)
print("\nSaved my_results.json")
