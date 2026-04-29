"""Additional analyses for ds001_prostate."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
results = {}

# ---- Refined olaparib × BRCA2 within mCRPC ----
sub = df[df['mcrpc'] == 1]
formula = "pfs_months ~ treatment_olaparib * brca2_mutation"
m = smf.ols(formula, data=sub).fit()
print('Olaparib×BRCA2 in mCRPC:')
print(m.summary().tables[1])
results['olaparib_brca2_in_mcrpc'] = {
    'beta_interaction': float(m.params['treatment_olaparib:brca2_mutation']),
    'p_interaction': float(m.pvalues['treatment_olaparib:brca2_mutation']),
    'mean_t1_b1': float(sub.loc[(sub['treatment_olaparib']==1)&(sub['brca2_mutation']==1),'pfs_months'].mean()),
    'mean_t0_b1': float(sub.loc[(sub['treatment_olaparib']==0)&(sub['brca2_mutation']==1),'pfs_months'].mean()),
    'n_t1_b1': int(((sub['treatment_olaparib']==1)&(sub['brca2_mutation']==1)).sum()),
    'n_t0_b1': int(((sub['treatment_olaparib']==0)&(sub['brca2_mutation']==1)).sum()),
}

# ---- Adjusted olaparib × BRCA2 ----
formula_adj = "pfs_months ~ treatment_olaparib * brca2_mutation + age_years + ecog_ps + mcrpc + albumin_g_dl + ldh_u_l + psa_ng_ml + weight_loss_pct_6mo"
m_adj = smf.ols(formula_adj, data=df).fit()
print('Adjusted olaparib×brca2:')
print(m_adj.summary().tables[1])
results['adj_olaparib_brca2'] = {
    'beta_interaction': float(m_adj.params['treatment_olaparib:brca2_mutation']),
    'p_interaction': float(m_adj.pvalues['treatment_olaparib:brca2_mutation']),
    'beta_olaparib_main': float(m_adj.params['treatment_olaparib']),
    'p_olaparib_main': float(m_adj.pvalues['treatment_olaparib']),
}

# ---- Pembrolizumab × MSI-high adjusted ----
formula_pm = "pfs_months ~ treatment_pembrolizumab * msi_high + age_years + ecog_ps + mcrpc + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo"
m_pm = smf.ols(formula_pm, data=df).fit()
print('\nAdjusted pembro×MSI:')
print(m_pm.summary().tables[1])
results['adj_pembro_msi'] = {
    'beta_interaction': float(m_pm.params['treatment_pembrolizumab:msi_high']),
    'p_interaction': float(m_pm.pvalues['treatment_pembrolizumab:msi_high']),
}

# ---- Lu177-PSMA × PSMA-high adjusted ----
formula_lu = "pfs_months ~ treatment_lu177_psma * psma_high + age_years + ecog_ps + mcrpc + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo"
m_lu = smf.ols(formula_lu, data=df).fit()
print('\nAdjusted Lu177×PSMAhigh:')
print(m_lu.summary().tables[1])
results['adj_lu177_psma_high'] = {
    'beta_interaction': float(m_lu.params['treatment_lu177_psma:psma_high']),
    'p_interaction': float(m_lu.pvalues['treatment_lu177_psma:psma_high']),
}

# ---- ECOG subgroup: treatment effects in fit (ECOG 0-1) vs unfit (ECOG 2+) ----
fit_idx = df['ecog_ps'] <= 1
treatments = ['treatment_olaparib','treatment_pembrolizumab','treatment_docetaxel','treatment_enzalutamide','treatment_abiraterone','treatment_lu177_psma']
for t in treatments:
    a = df.loc[fit_idx & (df[t]==1), 'pfs_months']
    b = df.loc[fit_idx & (df[t]==0), 'pfs_months']
    t_stat, p = stats.ttest_ind(a, b, equal_var=False)
    a2 = df.loc[~fit_idx & (df[t]==1), 'pfs_months']
    b2 = df.loc[~fit_idx & (df[t]==0), 'pfs_months']
    t2, p2 = stats.ttest_ind(a2, b2, equal_var=False)
    results[f'ecog_subgroup_{t}'] = {
        'fit_diff': float(a.mean() - b.mean()), 'fit_p': float(p),
        'unfit_diff': float(a2.mean() - b2.mean()), 'unfit_p': float(p2),
    }
    print(f'{t}: fit_diff={a.mean()-b.mean():.3f} (p={p:.3f}), unfit_diff={a2.mean()-b2.mean():.3f} (p={p2:.3f})')

# ---- ECOG main effect ----
m_e = smf.ols("pfs_months ~ ecog_ps", data=df).fit()
print('\nECOG main:', m_e.params['ecog_ps'], m_e.pvalues['ecog_ps'])

# ---- Albumin, LDH categorical ----
df['albumin_low'] = (df['albumin_g_dl'] < 3.5).astype(int)
df['ldh_high'] = (df['ldh_u_l'] > df['ldh_u_l'].median()).astype(int)
results['albumin_low_main'] = {
    'mean_low': float(df.loc[df['albumin_low']==1,'pfs_months'].mean()),
    'mean_norm': float(df.loc[df['albumin_low']==0,'pfs_months'].mean()),
    'diff': float(df.loc[df['albumin_low']==1,'pfs_months'].mean() - df.loc[df['albumin_low']==0,'pfs_months'].mean()),
    'p': float(stats.ttest_ind(df.loc[df['albumin_low']==1,'pfs_months'], df.loc[df['albumin_low']==0,'pfs_months'], equal_var=False).pvalue),
}
print('Albumin low:', results['albumin_low_main'])

# ---- Treatment combos: any AR-targeting vs none in mCRPC ----
df['ar_targeting'] = ((df['treatment_enzalutamide']==1)|(df['treatment_abiraterone']==1)).astype(int)
sub = df[df['mcrpc']==1]
a = sub.loc[sub['ar_targeting']==1, 'pfs_months']
b = sub.loc[sub['ar_targeting']==0, 'pfs_months']
t, p = stats.ttest_ind(a, b, equal_var=False)
results['ar_targeting_mcrpc'] = {'mean_yes': float(a.mean()), 'mean_no': float(b.mean()), 'diff': float(a.mean()-b.mean()), 'p': float(p)}
print('AR-targeting in mCRPC:', results['ar_targeting_mcrpc'])

# ---- Olaparib × BRCA2 in non-mCRPC (sanity) ----
sub2 = df[df['mcrpc']==0]
m2 = smf.ols("pfs_months ~ treatment_olaparib * brca2_mutation", data=sub2).fit()
results['olaparib_brca2_in_nonmcrpc'] = {
    'beta_interaction': float(m2.params['treatment_olaparib:brca2_mutation']),
    'p_interaction': float(m2.pvalues['treatment_olaparib:brca2_mutation']),
}
print('Olaparib×BRCA2 in non-mCRPC:', results['olaparib_brca2_in_nonmcrpc'])

# ---- BRCA2 main effect adjusted ----
m_b = smf.ols("pfs_months ~ brca2_mutation + age_years + ecog_ps + mcrpc + albumin_g_dl + ldh_u_l", data=df).fit()
results['brca2_adj'] = {'beta': float(m_b.params['brca2_mutation']), 'p': float(m_b.pvalues['brca2_mutation'])}
print('BRCA2 adjusted:', results['brca2_adj'])

# ---- Race × treatment interactions (do Black patients get same olaparib benefit?) ----
df['race_black'] = (df['race_ethnicity'] == 'black').astype(int)
m_r = smf.ols("pfs_months ~ treatment_olaparib * race_black + age_years + ecog_ps", data=df).fit()
results['olaparib_race_black'] = {
    'beta_interaction': float(m_r.params['treatment_olaparib:race_black']),
    'p_interaction': float(m_r.pvalues['treatment_olaparib:race_black']),
}
print('Olaparib×race_black:', results['olaparib_race_black'])

# ---- Insurance × treatment ----
df['priv_ins'] = (df['insurance_type'] == 'private').astype(int)
m_i = smf.ols("pfs_months ~ treatment_olaparib * priv_ins + age_years + ecog_ps", data=df).fit()
results['olaparib_private_ins'] = {
    'beta_interaction': float(m_i.params['treatment_olaparib:priv_ins']),
    'p_interaction': float(m_i.pvalues['treatment_olaparib:priv_ins']),
}
print('Olaparib×private:', results['olaparib_private_ins'])

# ---- Adjusted SNP × treatment for top SNPs ----
# rs4986893: looked promising
top_snps = ['snp_rs4986893','snp_rs429358','snp_rs1065852']
for s in top_snps:
    m_s = smf.ols(f"pfs_months ~ {s} + age_years + ecog_ps + mcrpc + albumin_g_dl", data=df).fit()
    results[f'snp_{s}_adj'] = {'beta': float(m_s.params[s]), 'p': float(m_s.pvalues[s])}
    print(f'{s} adj:', results[f'snp_{s}_adj'])

# ---- Visceral mets adjusted ----
m_v = smf.ols("pfs_months ~ visceral_mets + age_years + ecog_ps + mcrpc + albumin_g_dl", data=df).fit()
results['visceral_adj'] = {'beta': float(m_v.params['visceral_mets']), 'p': float(m_v.pvalues['visceral_mets'])}
print('Visceral adj:', results['visceral_adj'])

# ---- High symptom burden composite ----
df['symptom_score'] = df['fatigue_grade'] + df['pain_nrs']/2 + df['dyspnea_grade'] + df['appetite_loss_grade']
m_sy = smf.ols("pfs_months ~ symptom_score + age_years + ecog_ps", data=df).fit()
results['symptom_score_adj'] = {'beta': float(m_sy.params['symptom_score']), 'p': float(m_sy.pvalues['symptom_score'])}
print('Symptom score adj:', results['symptom_score_adj'])

# ---- mcrpc × treatment effects ----
for t in ['treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel']:
    m_m = smf.ols(f"pfs_months ~ {t} * mcrpc + age_years + ecog_ps + albumin_g_dl", data=df).fit()
    inter_var = f'{t}:mcrpc'
    results[f'{t}_mcrpc_inter'] = {
        'beta_interaction': float(m_m.params[inter_var]),
        'p_interaction': float(m_m.pvalues[inter_var]),
    }
    print(f'{t}*mcrpc:', results[f'{t}_mcrpc_inter'])

# ---- Save ----
with open('analysis_results2.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved analysis_results2.json")
