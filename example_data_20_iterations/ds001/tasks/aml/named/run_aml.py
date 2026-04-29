import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings, json
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
out = {}

def logit_single(formula, data=df):
    return smf.logit(formula, data=data).fit(disp=0, maxiter=100)

def iact(biom, treat):
    f = f'objective_response ~ {biom} * {treat}'
    m = smf.logit(f, data=df).fit(disp=0, maxiter=100)
    iname = f'{biom}:{treat}'
    return {
        'inter_coef': float(m.params[iname]),
        'inter_p': float(m.pvalues[iname]),
        'main_treat': float(m.params[treat]),
        'main_treat_p': float(m.pvalues[treat]),
        'main_biom': float(m.params[biom]),
        'main_biom_p': float(m.pvalues[biom]),
    }

# Single-variable models for prognostics
for f in ['ecog_ps','age_years','sex_female','weight_loss_pct_6mo','blast_pct_marrow',
          'idh1_mutation','albumin_g_dl','crp_mg_l','wbc_k_per_ul',
          'treatment_venetoclax_azacitidine','atrial_fibrillation',
          'treatment_midostaurin','treatment_gilteritinib','treatment_ivosidenib',
          'treatment_enasidenib','treatment_7plus3',
          'flt3_itd','flt3_tkd','idh2_mutation','npm1_mutation','tp53_mutation',
          'complex_karyotype','secondary_aml','unfit_for_intensive',
          'snp_rs1050828','snp_rs2032582','snp_rs1799853','snp_rs3813867',
          'ast_u_l','alt_u_l','total_bilirubin_mg_dl','creatinine_mg_dl',
          'ldh_u_l','platelets_k_ul','hemoglobin_g_dl','nlr','inr',
          'fatigue_grade','pain_nrs','dyspnea_grade','cough_grade','appetite_loss_grade',
          'rural_residence','smoking_pack_years','education_years',
          'prior_chemotherapy','prior_radiation','prior_surgery',
          'prior_immunotherapy','prior_targeted_therapy','prior_lines_of_therapy']:
    m = logit_single(f'objective_response ~ {f}')
    out[f'main_{f}'] = {'coef': float(m.params[f]), 'p': float(m.pvalues[f])}

# Interactions
out['inter_flt3itd_x_mido'] = iact('flt3_itd','treatment_midostaurin')
out['inter_flt3itd_x_gilt'] = iact('flt3_itd','treatment_gilteritinib')
out['inter_flt3tkd_x_gilt'] = iact('flt3_tkd','treatment_gilteritinib')
out['inter_flt3tkd_x_mido'] = iact('flt3_tkd','treatment_midostaurin')
out['inter_idh1_x_ivo'] = iact('idh1_mutation','treatment_ivosidenib')
out['inter_idh2_x_ena'] = iact('idh2_mutation','treatment_enasidenib')
out['inter_npm1_x_venaza'] = iact('npm1_mutation','treatment_venetoclax_azacitidine')
out['inter_tp53_x_venaza'] = iact('tp53_mutation','treatment_venetoclax_azacitidine')
out['inter_tp53_x_7p3'] = iact('tp53_mutation','treatment_7plus3')
out['inter_ck_x_7p3'] = iact('complex_karyotype','treatment_7plus3')
out['inter_ck_x_venaza'] = iact('complex_karyotype','treatment_venetoclax_azacitidine')
out['inter_unfit_x_venaza'] = iact('unfit_for_intensive','treatment_venetoclax_azacitidine')
out['inter_unfit_x_7p3'] = iact('unfit_for_intensive','treatment_7plus3')
out['inter_secondary_x_venaza'] = iact('secondary_aml','treatment_venetoclax_azacitidine')

# Subgroup-restricted treatment effects
def sub_effect(filter_col, val, treat):
    sub = df[df[filter_col]==val]
    m = smf.logit(f'objective_response ~ {treat}', data=sub).fit(disp=0, maxiter=100)
    return {'coef': float(m.params[treat]), 'p': float(m.pvalues[treat]), 'n': int(len(sub))}

out['ivo_in_idh1+'] = sub_effect('idh1_mutation', 1, 'treatment_ivosidenib')
out['ena_in_idh2+'] = sub_effect('idh2_mutation', 1, 'treatment_enasidenib')
out['gilt_in_flt3itd+'] = sub_effect('flt3_itd', 1, 'treatment_gilteritinib')
out['mido_in_flt3itd+'] = sub_effect('flt3_itd', 1, 'treatment_midostaurin')
out['mido_in_flt3tkd+'] = sub_effect('flt3_tkd', 1, 'treatment_midostaurin')
out['venaza_in_npm1+'] = sub_effect('npm1_mutation', 1, 'treatment_venetoclax_azacitidine')
out['venaza_in_unfit'] = sub_effect('unfit_for_intensive', 1, 'treatment_venetoclax_azacitidine')
out['7p3_in_unfit'] = sub_effect('unfit_for_intensive', 1, 'treatment_7plus3')
out['7p3_in_ck'] = sub_effect('complex_karyotype', 1, 'treatment_7plus3')

# Multivariable
form = ('objective_response ~ age_years + ecog_ps + weight_loss_pct_6mo + blast_pct_marrow + '
        'albumin_g_dl + crp_mg_l + wbc_k_per_ul + idh1_mutation + idh2_mutation + tp53_mutation + '
        'complex_karyotype + npm1_mutation + flt3_itd + secondary_aml + '
        'treatment_venetoclax_azacitidine + treatment_7plus3 + treatment_midostaurin + '
        'treatment_gilteritinib + treatment_ivosidenib + treatment_enasidenib + atrial_fibrillation + sex_female')
m_mv = smf.logit(form, data=df).fit(disp=0, maxiter=300)
out['multivariable_coefs'] = {k: float(v) for k,v in m_mv.params.items()}
out['multivariable_p'] = {k: float(v) for k,v in m_mv.pvalues.items()}

# Symptom composite
df2 = df.copy()
df2['symp_sum'] = df2[['fatigue_grade','pain_nrs','dyspnea_grade','cough_grade','appetite_loss_grade']].sum(axis=1)
m = smf.logit('objective_response ~ symp_sum', data=df2).fit(disp=0)
out['main_symp_sum'] = {'coef': float(m.params['symp_sum']), 'p': float(m.pvalues['symp_sum'])}

# Race / insurance LRT
m = smf.logit('objective_response ~ C(race_ethnicity)', data=df).fit(disp=0)
out['race_lrt_p'] = float(m.llr_pvalue)
out['race_rr'] = df.groupby('race_ethnicity')['objective_response'].mean().to_dict()
m = smf.logit('objective_response ~ C(insurance_type)', data=df).fit(disp=0)
out['ins_lrt_p'] = float(m.llr_pvalue)
out['ins_rr'] = df.groupby('insurance_type')['objective_response'].mean().to_dict()

# Interactions: afib x age
m = smf.logit('objective_response ~ atrial_fibrillation * age_years', data=df).fit(disp=0)
out['inter_afib_x_age'] = {
    'coef': float(m.params['atrial_fibrillation:age_years']),
    'p': float(m.pvalues['atrial_fibrillation:age_years'])
}
m = smf.logit('objective_response ~ ecog_ps * treatment_venetoclax_azacitidine', data=df).fit(disp=0)
out['inter_ecog_x_venaza'] = {
    'coef': float(m.params['ecog_ps:treatment_venetoclax_azacitidine']),
    'p': float(m.pvalues['ecog_ps:treatment_venetoclax_azacitidine'])
}
m = smf.logit('objective_response ~ age_years * treatment_7plus3', data=df).fit(disp=0)
out['inter_age_x_7p3'] = {
    'coef': float(m.params['age_years:treatment_7plus3']),
    'p': float(m.pvalues['age_years:treatment_7plus3'])
}

# Composite high-risk
df2['high_risk'] = ((df2['tp53_mutation']==1)|(df2['complex_karyotype']==1)).astype(int)
m = smf.logit('objective_response ~ high_risk', data=df2).fit(disp=0)
out['main_high_risk'] = {'coef': float(m.params['high_risk']), 'p': float(m.pvalues['high_risk'])}
m = smf.logit('objective_response ~ high_risk * treatment_venetoclax_azacitidine', data=df2).fit(disp=0)
out['inter_highrisk_x_venaza'] = {
    'coef': float(m.params['high_risk:treatment_venetoclax_azacitidine']),
    'p': float(m.pvalues['high_risk:treatment_venetoclax_azacitidine'])
}

# All SNPs scan
snp_cols = [c for c in df.columns if c.startswith('snp_')]
snp_scan = []
for s in snp_cols:
    m = smf.logit(f'objective_response ~ {s}', data=df).fit(disp=0)
    snp_scan.append((s, float(m.params[s]), float(m.pvalues[s])))
snp_scan.sort(key=lambda x: x[2])
out['snp_scan_top5'] = [{'snp': s, 'coef': c, 'p': p} for s,c,p in snp_scan[:5]]
out['snp_scan_count_p_lt_0.05'] = sum(1 for _,_,p in snp_scan if p < 0.05)
out['snp_count'] = len(snp_scan)

# IDH1 x venaza
m = smf.logit('objective_response ~ idh1_mutation * treatment_venetoclax_azacitidine', data=df).fit(disp=0)
out['inter_idh1_x_venaza'] = {
    'coef': float(m.params['idh1_mutation:treatment_venetoclax_azacitidine']),
    'p': float(m.pvalues['idh1_mutation:treatment_venetoclax_azacitidine'])
}

# Subgroup response rates
out['rr_by_idh1'] = df.groupby('idh1_mutation')['objective_response'].mean().to_dict()
out['rr_by_ecog'] = df.groupby('ecog_ps')['objective_response'].mean().to_dict()
out['rr_by_afib'] = df.groupby('atrial_fibrillation')['objective_response'].mean().to_dict()
out['rr_by_ck'] = df.groupby('complex_karyotype')['objective_response'].mean().to_dict()
out['rr_by_tp53'] = df.groupby('tp53_mutation')['objective_response'].mean().to_dict()
out['rr_by_npm1'] = df.groupby('npm1_mutation')['objective_response'].mean().to_dict()
out['mean_age_by_response'] = df.groupby('objective_response')['age_years'].mean().to_dict()
out['mean_albumin_by_response'] = df.groupby('objective_response')['albumin_g_dl'].mean().to_dict()
out['mean_ecog_by_response'] = df.groupby('objective_response')['ecog_ps'].mean().to_dict()
out['mean_blast_by_response'] = df.groupby('objective_response')['blast_pct_marrow'].mean().to_dict()
out['mean_crp_by_response'] = df.groupby('objective_response')['crp_mg_l'].mean().to_dict()
out['mean_wl_by_response'] = df.groupby('objective_response')['weight_loss_pct_6mo'].mean().to_dict()
out['mean_wbc_by_response'] = df.groupby('objective_response')['wbc_k_per_ul'].mean().to_dict()

# fishers / chi tests for several
for f in ['atrial_fibrillation','complex_karyotype','tp53_mutation','idh1_mutation','npm1_mutation']:
    tab = pd.crosstab(df[f], df['objective_response'])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    out[f'chi2_{f}'] = {'p': float(p), 'rr1': float(df.loc[df[f]==1,'objective_response'].mean()),
                       'rr0': float(df.loc[df[f]==0,'objective_response'].mean())}

with open('results.json','w') as f:
    json.dump(out, f, indent=2, default=str)
print('done')
