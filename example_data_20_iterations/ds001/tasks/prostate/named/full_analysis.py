"""Full 25-iteration analysis for ds001_prostate. Produces transcript.json and analysis_summary.txt."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet('dataset.parquet')
N = len(df)
OUTCOME = 'pfs_months'

iterations = []  # list of dicts

def add_iter(idx, hyps, analyses):
    iterations.append({
        'index': idx,
        'proposed_hypotheses': hyps,
        'analyses': analyses,
    })

def t_test_binary(col, hyp_id, label_text):
    a = df.loc[df[col] == 1, OUTCOME]
    b = df.loc[df[col] == 0, OUTCOME]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    eff = float(a.mean() - b.mean())
    return {
        'hypothesis_ids': [hyp_id],
        'code': f"stats.ttest_ind(df.loc[df['{col}']==1,'{OUTCOME}'], df.loc[df['{col}']==0,'{OUTCOME}'])",
        'result_summary': f"{label_text}: mean({col}=1)={a.mean():.3f} vs mean=0={b.mean():.3f}, diff={eff:+.4f}, t={t:.3f}, p={p:.3g} (n1={len(a)}, n0={len(b)})",
        'p_value': float(p),
        'effect_estimate': eff,
        'significant': bool(p < 0.05),
    }

def pearson(col, hyp_id, label_text):
    r, p = stats.pearsonr(df[col], df[OUTCOME])
    return {
        'hypothesis_ids': [hyp_id],
        'code': f"stats.pearsonr(df['{col}'], df['{OUTCOME}'])",
        'result_summary': f"{label_text}: r={r:+.4f}, p={p:.3g}",
        'p_value': float(p),
        'effect_estimate': float(r),
        'significant': bool(p < 0.05),
    }

def ols_reg(formula_cols, hyp_id, label_text, target=None, slope_for=None):
    X = df[formula_cols].copy()
    X = sm.add_constant(X)
    y = df[OUTCOME]
    res = sm.OLS(y, X).fit()
    if slope_for is None:
        slope_for = formula_cols[0]
    coef = float(res.params[slope_for])
    p = float(res.pvalues[slope_for])
    return {
        'hypothesis_ids': [hyp_id],
        'code': f"sm.OLS(df['{OUTCOME}'], sm.add_constant(df[{formula_cols!r}])).fit()  # coef of {slope_for}",
        'result_summary': f"{label_text}: adj-coef({slope_for})={coef:+.4f}, p={p:.3g}, R²={res.rsquared:.4f}, n={int(res.nobs)}",
        'p_value': p,
        'effect_estimate': coef,
        'significant': bool(p < 0.05),
    }

def interaction_lm(treat, marker, hyp_id, label_text):
    X = df[[treat, marker]].copy()
    X['interaction'] = X[treat] * X[marker]
    X = sm.add_constant(X)
    res = sm.OLS(df[OUTCOME], X).fit()
    coef = float(res.params['interaction'])
    p = float(res.pvalues['interaction'])
    return {
        'hypothesis_ids': [hyp_id],
        'code': f"OLS(pfs ~ {treat} + {marker} + {treat}:{marker})  # interaction term",
        'result_summary': f"{label_text}: interaction({treat}×{marker}) β={coef:+.4f}, p={p:.3g}",
        'p_value': p,
        'effect_estimate': coef,
        'significant': bool(p < 0.05),
    }

def subgroup_effect(treat, sub_col, sub_val, hyp_id, label_text):
    sub = df[df[sub_col] == sub_val]
    a = sub.loc[sub[treat] == 1, OUTCOME]
    b = sub.loc[sub[treat] == 0, OUTCOME]
    if len(a) < 10 or len(b) < 10:
        return {
            'hypothesis_ids': [hyp_id],
            'code': f"subgroup ({sub_col}={sub_val}) PFS by {treat}",
            'result_summary': f"{label_text}: insufficient sample (n1={len(a)}, n0={len(b)})",
            'p_value': None,
            'effect_estimate': 0.0,
            'significant': False,
        }
    t, p = stats.ttest_ind(a, b, equal_var=False)
    eff = float(a.mean() - b.mean())
    return {
        'hypothesis_ids': [hyp_id],
        'code': f"stats.ttest_ind(df[df.{sub_col}=={sub_val} & {treat}==1].pfs, df[df.{sub_col}=={sub_val} & {treat}==0].pfs)",
        'result_summary': f"{label_text}: among {sub_col}={sub_val} (n={len(sub)}): mean({treat}=1)={a.mean():.3f} vs ={b.mean():.3f}, diff={eff:+.4f}, t={t:.3f}, p={p:.3g}",
        'p_value': float(p),
        'effect_estimate': eff,
        'significant': bool(p < 0.05),
    }

# ========================= Iteration 1: Treatment main effects =========================
hyps_1 = [
    {'id':'h1_1','text':'Patients receiving treatment_enzalutamide have higher mean pfs_months than those not receiving it.','kind':'novel'},
    {'id':'h1_2','text':'Patients receiving treatment_abiraterone have higher mean pfs_months than those not receiving it.','kind':'novel'},
    {'id':'h1_3','text':'Patients receiving treatment_docetaxel have higher mean pfs_months than those not receiving it.','kind':'novel'},
    {'id':'h1_4','text':'Patients receiving treatment_olaparib have higher mean pfs_months than those not receiving it.','kind':'novel'},
    {'id':'h1_5','text':'Patients receiving treatment_lu177_psma have higher mean pfs_months than those not receiving it.','kind':'novel'},
    {'id':'h1_6','text':'Patients receiving treatment_pembrolizumab have higher mean pfs_months than those not receiving it.','kind':'novel'},
]
ana_1 = [
    t_test_binary('treatment_enzalutamide','h1_1','PFS by treatment_enzalutamide'),
    t_test_binary('treatment_abiraterone','h1_2','PFS by treatment_abiraterone'),
    t_test_binary('treatment_docetaxel','h1_3','PFS by treatment_docetaxel'),
    t_test_binary('treatment_olaparib','h1_4','PFS by treatment_olaparib'),
    t_test_binary('treatment_lu177_psma','h1_5','PFS by treatment_lu177_psma'),
    t_test_binary('treatment_pembrolizumab','h1_6','PFS by treatment_pembrolizumab'),
]
add_iter(1, hyps_1, ana_1)

# ========================= Iteration 2: disease-state main effects =========================
hyps_2 = [
    {'id':'h2_1','text':'Patients with mcrpc=1 (castration-resistant) have lower mean pfs_months than those with mcrpc=0.','kind':'novel'},
    {'id':'h2_2','text':'Patients with visceral_mets=1 have lower mean pfs_months than those with visceral_mets=0.','kind':'novel'},
    {'id':'h2_3','text':'Patients with liver_mets=1 have lower mean pfs_months than those with liver_mets=0.','kind':'novel'},
    {'id':'h2_4','text':'Patients with bone_mets=1 have lower mean pfs_months than those with bone_mets=0.','kind':'novel'},
    {'id':'h2_5','text':'Higher ecog_ps is correlated with lower pfs_months (Pearson r negative).','kind':'novel'},
    {'id':'h2_6','text':'Higher gleason_score is correlated with lower pfs_months.','kind':'novel'},
]
ana_2 = [
    t_test_binary('mcrpc','h2_1','PFS by mcrpc'),
    t_test_binary('visceral_mets','h2_2','PFS by visceral_mets'),
    t_test_binary('liver_mets','h2_3','PFS by liver_mets'),
    t_test_binary('bone_mets','h2_4','PFS by bone_mets'),
    pearson('ecog_ps','h2_5','PFS vs ecog_ps'),
    pearson('gleason_score','h2_6','PFS vs gleason_score'),
]
add_iter(2, hyps_2, ana_2)

# ========================= Iteration 3: prognostic labs =========================
hyps_3 = [
    {'id':'h3_1','text':'Higher psa_ng_ml is correlated with lower pfs_months.','kind':'novel'},
    {'id':'h3_2','text':'Higher ldh_u_l is correlated with lower pfs_months.','kind':'novel'},
    {'id':'h3_3','text':'Higher albumin_g_dl is correlated with higher pfs_months.','kind':'novel'},
    {'id':'h3_4','text':'Higher alkaline_phosphatase_u_l is correlated with lower pfs_months.','kind':'novel'},
    {'id':'h3_5','text':'Higher hemoglobin_g_dl is correlated with higher pfs_months.','kind':'novel'},
    {'id':'h3_6','text':'Higher crp_mg_l is correlated with lower pfs_months.','kind':'novel'},
    {'id':'h3_7','text':'Higher nlr is correlated with lower pfs_months.','kind':'novel'},
    {'id':'h3_8','text':'Greater weight_loss_pct_6mo is correlated with lower pfs_months.','kind':'novel'},
]
ana_3 = [
    pearson('psa_ng_ml','h3_1','PFS vs psa_ng_ml'),
    pearson('ldh_u_l','h3_2','PFS vs ldh_u_l'),
    pearson('albumin_g_dl','h3_3','PFS vs albumin_g_dl'),
    pearson('alkaline_phosphatase_u_l','h3_4','PFS vs alkaline_phosphatase_u_l'),
    pearson('hemoglobin_g_dl','h3_5','PFS vs hemoglobin_g_dl'),
    pearson('crp_mg_l','h3_6','PFS vs crp_mg_l'),
    pearson('nlr','h3_7','PFS vs nlr'),
    pearson('weight_loss_pct_6mo','h3_8','PFS vs weight_loss_pct_6mo'),
]
add_iter(3, hyps_3, ana_3)

# ========================= Iteration 4: age & demographics =========================
hyps_4 = [
    {'id':'h4_1','text':'Older age_years is correlated with pfs_months (direction unspecified, suspect lower).','kind':'novel'},
    {'id':'h4_2','text':'Mean pfs_months differs across race_ethnicity categories.','kind':'novel'},
    {'id':'h4_3','text':'Mean pfs_months differs across insurance_type categories.','kind':'novel'},
    {'id':'h4_4','text':'Patients with rural_residence=1 have different mean pfs_months than those with rural_residence=0.','kind':'novel'},
    {'id':'h4_5','text':'Higher education_years is correlated with higher pfs_months.','kind':'novel'},
    {'id':'h4_6','text':'Higher smoking_pack_years is correlated with lower pfs_months.','kind':'novel'},
]
ana_4 = [pearson('age_years','h4_1','PFS vs age_years')]
# race_ethnicity: ANOVA across categories
groups_race = [df.loc[df['race_ethnicity']==v, OUTCOME].values for v in df['race_ethnicity'].unique()]
F_r, p_r = stats.f_oneway(*groups_race)
means_r = df.groupby('race_ethnicity')[OUTCOME].mean().to_dict()
ana_4.append({'hypothesis_ids':['h4_2'],
    'code':"stats.f_oneway(*[df.loc[df.race_ethnicity==v,'pfs_months'].values for v in df.race_ethnicity.unique()])",
    'result_summary':f"PFS across race_ethnicity (ANOVA): F={F_r:.3f}, p={p_r:.3g}; means={ {k:round(v,3) for k,v in means_r.items()} }",
    'p_value':float(p_r),'effect_estimate':float(max(means_r.values())-min(means_r.values())),'significant':bool(p_r<0.05)})
groups_ins = [df.loc[df['insurance_type']==v, OUTCOME].values for v in df['insurance_type'].unique()]
F_i, p_i = stats.f_oneway(*groups_ins)
means_i = df.groupby('insurance_type')[OUTCOME].mean().to_dict()
ana_4.append({'hypothesis_ids':['h4_3'],
    'code':"stats.f_oneway(*[df.loc[df.insurance_type==v,'pfs_months'].values for v in df.insurance_type.unique()])",
    'result_summary':f"PFS across insurance_type (ANOVA): F={F_i:.3f}, p={p_i:.3g}; means={ {k:round(v,3) for k,v in means_i.items()} }",
    'p_value':float(p_i),'effect_estimate':float(max(means_i.values())-min(means_i.values())),'significant':bool(p_i<0.05)})
ana_4.append(t_test_binary('rural_residence','h4_4','PFS by rural_residence'))
ana_4.append(pearson('education_years','h4_5','PFS vs education_years'))
ana_4.append(pearson('smoking_pack_years','h4_6','PFS vs smoking_pack_years'))
add_iter(4, hyps_4, ana_4)

# ========================= Iteration 5: comorbidities =========================
comorbids = ['diabetes_mellitus','hypertension','copd','chronic_kidney_disease','heart_failure',
             'coronary_artery_disease','atrial_fibrillation','venous_thromboembolism_history',
             'autoimmune_disease','prior_malignancy','depression_anxiety_diagnosis']
hyps_5 = [{'id':f'h5_{i+1}','text':f'Patients with {c}=1 have lower mean pfs_months than those with {c}=0.','kind':'novel'}
          for i,c in enumerate(comorbids)]
ana_5 = [t_test_binary(c, f'h5_{i+1}', f'PFS by {c}') for i,c in enumerate(comorbids)]
add_iter(5, hyps_5, ana_5)

# ========================= Iteration 6: symptom grades =========================
sx = ['fatigue_grade','pain_nrs','dyspnea_grade','cough_grade','appetite_loss_grade']
hyps_6 = [{'id':f'h6_{i+1}','text':f'Higher {s} is correlated with lower pfs_months.','kind':'novel'} for i,s in enumerate(sx)]
ana_6 = [pearson(s, f'h6_{i+1}', f'PFS vs {s}') for i,s in enumerate(sx)]
add_iter(6, hyps_6, ana_6)

# ========================= Iteration 7: Prior therapies =========================
priors = ['prior_chemotherapy','prior_radiation','prior_surgery','prior_immunotherapy','prior_targeted_therapy']
hyps_7 = [{'id':f'h7_{i+1}','text':f'Patients with {p}=1 have lower mean pfs_months than those with {p}=0.','kind':'novel'} for i,p in enumerate(priors)]
hyps_7 += [
    {'id':'h7_6','text':'Higher prior_lines_of_therapy is correlated with lower pfs_months.','kind':'novel'},
    {'id':'h7_7','text':'Higher years_since_diagnosis is correlated with lower pfs_months.','kind':'novel'},
]
ana_7 = [t_test_binary(p, f'h7_{i+1}', f'PFS by {p}') for i,p in enumerate(priors)]
ana_7.append(pearson('prior_lines_of_therapy','h7_6','PFS vs prior_lines_of_therapy'))
ana_7.append(pearson('years_since_diagnosis','h7_7','PFS vs years_since_diagnosis'))
add_iter(7, hyps_7, ana_7)

# ========================= Iteration 8: targeted-therapy biomarkers (treatment×marker interactions) =========================
hyps_8 = [
    {'id':'h8_1','text':'Olaparib has a larger PFS benefit in brca2_mutation=1 patients than brca2_mutation=0 (positive treatment_olaparib×brca2_mutation interaction).','kind':'novel'},
    {'id':'h8_2','text':'Pembrolizumab has a larger PFS benefit in msi_high=1 patients than msi_high=0 (positive treatment_pembrolizumab×msi_high interaction).','kind':'novel'},
    {'id':'h8_3','text':'Lu177-PSMA has a larger PFS benefit in psma_high=1 patients than psma_high=0 (positive treatment_lu177_psma×psma_high interaction).','kind':'novel'},
    {'id':'h8_4','text':'Enzalutamide has a smaller PFS benefit in ar_v7_positive=1 patients than ar_v7_positive=0 (negative treatment_enzalutamide×ar_v7_positive interaction).','kind':'novel'},
    {'id':'h8_5','text':'Abiraterone has a smaller PFS benefit in ar_v7_positive=1 patients than ar_v7_positive=0 (negative treatment_abiraterone×ar_v7_positive interaction).','kind':'novel'},
]
ana_8 = [
    interaction_lm('treatment_olaparib','brca2_mutation','h8_1','olaparib×brca2_mutation interaction'),
    interaction_lm('treatment_pembrolizumab','msi_high','h8_2','pembrolizumab×msi_high interaction'),
    interaction_lm('treatment_lu177_psma','psma_high','h8_3','lu177_psma×psma_high interaction'),
    interaction_lm('treatment_enzalutamide','ar_v7_positive','h8_4','enzalutamide×ar_v7_positive interaction'),
    interaction_lm('treatment_abiraterone','ar_v7_positive','h8_5','abiraterone×ar_v7_positive interaction'),
]
add_iter(8, hyps_8, ana_8)

# ========================= Iteration 9: subgroup analyses for matched markers =========================
hyps_9 = [
    {'id':'h9_1','text':'Within brca2_mutation=1, patients on treatment_olaparib have higher mean pfs_months than those off treatment_olaparib.','kind':'refined'},
    {'id':'h9_2','text':'Within brca2_mutation=0, treatment_olaparib does NOT improve mean pfs_months (refined null).','kind':'refined'},
    {'id':'h9_3','text':'Within msi_high=1, treatment_pembrolizumab improves mean pfs_months.','kind':'refined'},
    {'id':'h9_4','text':'Within psma_high=1, treatment_lu177_psma improves mean pfs_months.','kind':'refined'},
    {'id':'h9_5','text':'Within ar_v7_positive=1, treatment_enzalutamide does NOT improve mean pfs_months (refined null).','kind':'refined'},
]
ana_9 = [
    subgroup_effect('treatment_olaparib','brca2_mutation',1,'h9_1','olaparib in BRCA2+'),
    subgroup_effect('treatment_olaparib','brca2_mutation',0,'h9_2','olaparib in BRCA2-'),
    subgroup_effect('treatment_pembrolizumab','msi_high',1,'h9_3','pembrolizumab in MSI-high'),
    subgroup_effect('treatment_lu177_psma','psma_high',1,'h9_4','lu177_psma in PSMA-high'),
    subgroup_effect('treatment_enzalutamide','ar_v7_positive',1,'h9_5','enzalutamide in AR-V7+'),
]
add_iter(9, hyps_9, ana_9)

# ========================= Iteration 10: more labs =========================
more_labs = ['calcium_mg_dl','sodium_meq_l','potassium_meq_l','glucose_mg_dl','total_bilirubin_mg_dl',
             'creatinine_mg_dl','bun_mg_dl','platelets_k_ul','wbc_k_ul','anc_k_ul','alc_k_ul','inr',
             'ast_u_l','alt_u_l','tsh_uiu_ml','ca_125_u_ml','cea_ng_ml']
hyps_10 = [{'id':f'h10_{i+1}','text':f'{lab} is correlated with pfs_months (any direction).','kind':'novel'} for i,lab in enumerate(more_labs)]
ana_10 = [pearson(lab, f'h10_{i+1}', f'PFS vs {lab}') for i,lab in enumerate(more_labs)]
add_iter(10, hyps_10, ana_10)

# ========================= Iteration 11: vitals & BMI =========================
vitals = ['bmi','systolic_bp_mmhg','diastolic_bp_mmhg','heart_rate_bpm','spo2_pct']
hyps_11 = [{'id':f'h11_{i+1}','text':f'{v} is correlated with pfs_months.','kind':'novel'} for i,v in enumerate(vitals)]
ana_11 = [pearson(v, f'h11_{i+1}', f'PFS vs {v}') for i,v in enumerate(vitals)]
add_iter(11, hyps_11, ana_11)

# ========================= Iteration 12: genomic alterations (non-targeted) =========================
genomic = ['her2_amplification','met_exon14_skipping','ret_fusion','ros1_fusion','braf_v600e','ntrk_fusion',
           'nrg1_fusion','fgfr_alteration','cdkn2a_loss','tp53_mutation','keap1_mutation','pik3ca_mutation','pten_loss']
hyps_12 = [{'id':f'h12_{i+1}','text':f'Patients with {g}=1 have different mean pfs_months than those with {g}=0.','kind':'novel'} for i,g in enumerate(genomic)]
ana_12 = [t_test_binary(g, f'h12_{i+1}', f'PFS by {g}') for i,g in enumerate(genomic)]
add_iter(12, hyps_12, ana_12)

# ========================= Iteration 13: rare comorbidities & host history =========================
rare = ['hepatitis_b_history','hepatitis_c_history','hiv_positive','interstitial_lung_disease_history',
        'pleural_effusion','pericardial_effusion','adrenal_mets','contralateral_lung_mets']
hyps_13 = [{'id':f'h13_{i+1}','text':f'Patients with {c}=1 have different mean pfs_months than those with {c}=0.','kind':'novel'} for i,c in enumerate(rare)]
ana_13 = [t_test_binary(c, f'h13_{i+1}', f'PFS by {c}') for i,c in enumerate(rare)]
add_iter(13, hyps_13, ana_13)

# ========================= Iteration 14: SNP screen (all 25) =========================
snps = [c for c in df.columns if c.startswith('snp_')]
hyps_14 = [{'id':f'h14_{i+1}','text':f'{s} is correlated with pfs_months.','kind':'novel'} for i,s in enumerate(snps)]
ana_14 = [pearson(s, f'h14_{i+1}', f'PFS vs {s}') for i,s in enumerate(snps)]
add_iter(14, hyps_14, ana_14)

# ========================= Iteration 15: multivariable prognostic model =========================
hyps_15 = [
    {'id':'h15_1','text':'In a multivariable OLS adjusting for age_years, ecog_ps, mcrpc, visceral_mets, albumin_g_dl, ldh_u_l, psa_ng_ml, weight_loss_pct_6mo, alkaline_phosphatase_u_l, hemoglobin_g_dl, the coefficient on ecog_ps remains negative and significant.','kind':'refined'},
    {'id':'h15_2','text':'In the same multivariable OLS, the coefficient on albumin_g_dl remains positive and significant.','kind':'refined'},
    {'id':'h15_3','text':'In the same multivariable OLS, the coefficient on mcrpc remains negative and significant.','kind':'refined'},
    {'id':'h15_4','text':'In the same multivariable OLS, the coefficient on age_years remains positive and significant (older = longer pfs in this synthetic cohort).','kind':'refined'},
]
mv_cols = ['age_years','ecog_ps','mcrpc','visceral_mets','albumin_g_dl','ldh_u_l','psa_ng_ml',
           'weight_loss_pct_6mo','alkaline_phosphatase_u_l','hemoglobin_g_dl']
ana_15 = [
    ols_reg(mv_cols,'h15_1','MV PFS regression',slope_for='ecog_ps'),
    ols_reg(mv_cols,'h15_2','MV PFS regression',slope_for='albumin_g_dl'),
    ols_reg(mv_cols,'h15_3','MV PFS regression',slope_for='mcrpc'),
    ols_reg(mv_cols,'h15_4','MV PFS regression',slope_for='age_years'),
]
add_iter(15, hyps_15, ana_15)

# ========================= Iteration 16: olaparib×brca2 deeper, and pembro×msi deeper, conditional on mcrpc =========================
hyps_16 = [
    {'id':'h16_1','text':'Within mcrpc=1 patients, olaparib has a larger PFS benefit in brca2_mutation=1 than brca2_mutation=0.','kind':'refined'},
    {'id':'h16_2','text':'Within mcrpc=0 patients, olaparib has a larger PFS benefit in brca2_mutation=1 than brca2_mutation=0.','kind':'refined'},
    {'id':'h16_3','text':'Within visceral_mets=1, treatment_docetaxel improves PFS more than in visceral_mets=0 (positive interaction).','kind':'novel'},
]
def interaction_in_subgroup(treat, marker, sub_col, sub_val, hyp_id, label):
    sub = df[df[sub_col] == sub_val]
    X = sub[[treat, marker]].copy()
    X['interaction'] = X[treat] * X[marker]
    X = sm.add_constant(X)
    res = sm.OLS(sub[OUTCOME], X).fit()
    coef = float(res.params['interaction'])
    p = float(res.pvalues['interaction'])
    return {
        'hypothesis_ids':[hyp_id],
        'code':f"OLS in {sub_col}={sub_val}: pfs ~ {treat} + {marker} + {treat}:{marker}",
        'result_summary':f"{label}: in {sub_col}={sub_val} (n={len(sub)}), interaction β={coef:+.4f}, p={p:.3g}",
        'p_value':p,'effect_estimate':coef,'significant':bool(p<0.05)
    }
ana_16 = [
    interaction_in_subgroup('treatment_olaparib','brca2_mutation','mcrpc',1,'h16_1','olaparib×brca2 in mCRPC'),
    interaction_in_subgroup('treatment_olaparib','brca2_mutation','mcrpc',0,'h16_2','olaparib×brca2 in non-mCRPC'),
    interaction_lm('treatment_docetaxel','visceral_mets','h16_3','docetaxel×visceral_mets interaction'),
]
add_iter(16, hyps_16, ana_16)

# ========================= Iteration 17: treatment×ECOG interactions =========================
hyps_17 = [
    {'id':'h17_1','text':'The PFS benefit of treatment_docetaxel decreases as ecog_ps increases (negative treatment_docetaxel×ecog_ps interaction).','kind':'novel'},
    {'id':'h17_2','text':'The PFS benefit of treatment_olaparib does not depend on ecog_ps (interaction null).','kind':'novel'},
    {'id':'h17_3','text':'The PFS benefit of treatment_enzalutamide decreases as ecog_ps increases.','kind':'novel'},
]
ana_17 = [
    interaction_lm('treatment_docetaxel','ecog_ps','h17_1','docetaxel×ecog_ps interaction'),
    interaction_lm('treatment_olaparib','ecog_ps','h17_2','olaparib×ecog_ps interaction'),
    interaction_lm('treatment_enzalutamide','ecog_ps','h17_3','enzalutamide×ecog_ps interaction'),
]
add_iter(17, hyps_17, ana_17)

# ========================= Iteration 18: refined olaparib in BRCA2+ adjusted for prognostic confounders =========================
hyps_18 = [
    {'id':'h18_1','text':'Within brca2_mutation=1 patients, the adjusted coefficient of treatment_olaparib on pfs_months remains positive and significant after controlling for ecog_ps, mcrpc, age_years, albumin_g_dl, ldh_u_l, weight_loss_pct_6mo.','kind':'refined'},
    {'id':'h18_2','text':'Within brca2_mutation=0 patients, the adjusted coefficient of treatment_olaparib on pfs_months is null after controlling for the same prognostic confounders.','kind':'refined'},
]
def adj_treat_in_subgroup(treat, sub_col, sub_val, hyp_id, label, adj=mv_cols):
    sub = df[df[sub_col] == sub_val]
    cols = [treat] + adj
    X = sub[cols].copy()
    X = sm.add_constant(X)
    res = sm.OLS(sub[OUTCOME], X).fit()
    coef = float(res.params[treat])
    p = float(res.pvalues[treat])
    return {
        'hypothesis_ids':[hyp_id],
        'code':f"OLS within {sub_col}={sub_val}: pfs ~ {treat} + {adj}",
        'result_summary':f"{label}: in {sub_col}={sub_val} (n={len(sub)}), adj-β({treat})={coef:+.4f}, p={p:.3g}",
        'p_value':p,'effect_estimate':coef,'significant':bool(p<0.05)
    }
ana_18 = [
    adj_treat_in_subgroup('treatment_olaparib','brca2_mutation',1,'h18_1','adj olaparib in BRCA2+'),
    adj_treat_in_subgroup('treatment_olaparib','brca2_mutation',0,'h18_2','adj olaparib in BRCA2-'),
]
add_iter(18, hyps_18, ana_18)

# ========================= Iteration 19: composite "matched-therapy" effect =========================
df['matched_therapy'] = (
    ((df['treatment_olaparib']==1) & (df['brca2_mutation']==1)).astype(int) +
    ((df['treatment_pembrolizumab']==1) & (df['msi_high']==1)).astype(int) +
    ((df['treatment_lu177_psma']==1) & (df['psma_high']==1)).astype(int)
)
hyps_19 = [
    {'id':'h19_1','text':'Patients with any matched_therapy (olaparib+BRCA2 OR pembrolizumab+MSI-high OR Lu177+PSMA-high) have higher mean pfs_months than those with no matched therapy.','kind':'refined'},
    {'id':'h19_2','text':'Adjusted for prognostic covariates (age, ECOG, mCRPC, albumin, LDH, weight_loss), the matched_therapy indicator remains a positive predictor of pfs_months.','kind':'refined'},
]
a = df.loc[df['matched_therapy']>=1, OUTCOME]; b = df.loc[df['matched_therapy']==0, OUTCOME]
t_m, p_m = stats.ttest_ind(a, b, equal_var=False)
eff_m = float(a.mean() - b.mean())
ana_19 = [{
    'hypothesis_ids':['h19_1'],
    'code':"matched_therapy = (olaparib&brca2)|(pembro&msi)|(lu177&psma); ttest",
    'result_summary':f"Matched vs unmatched: mean(matched≥1)={a.mean():.3f} (n={len(a)}) vs mean=0={b.mean():.3f} (n={len(b)}), diff={eff_m:+.4f}, p={p_m:.3g}",
    'p_value':float(p_m),'effect_estimate':eff_m,'significant':bool(p_m<0.05)
}]
ana_19.append(ols_reg(['matched_therapy','age_years','ecog_ps','mcrpc','albumin_g_dl','ldh_u_l','weight_loss_pct_6mo'],'h19_2','adjusted matched_therapy effect',slope_for='matched_therapy'))
add_iter(19, hyps_19, ana_19)

# ========================= Iteration 20: treatment-treatment interactions =========================
hyps_20 = [
    {'id':'h20_1','text':'The PFS effect of treatment_enzalutamide is augmented when patients are also on treatment_abiraterone (positive enzalutamide×abiraterone interaction).','kind':'novel'},
    {'id':'h20_2','text':'The PFS effect of treatment_docetaxel is augmented when patients are also on treatment_enzalutamide.','kind':'novel'},
    {'id':'h20_3','text':'The PFS effect of treatment_olaparib in BRCA2+ patients is further augmented by concurrent treatment_abiraterone (3-way interaction not directly tested; instead test olaparib×abiraterone in BRCA2+).','kind':'novel'},
]
ana_20 = [
    interaction_lm('treatment_enzalutamide','treatment_abiraterone','h20_1','enzalutamide×abiraterone interaction'),
    interaction_lm('treatment_docetaxel','treatment_enzalutamide','h20_2','docetaxel×enzalutamide interaction'),
    interaction_in_subgroup('treatment_olaparib','treatment_abiraterone','brca2_mutation',1,'h20_3','olaparib×abiraterone in BRCA2+'),
]
add_iter(20, hyps_20, ana_20)

# ========================= Iteration 21: age age and treatment effects =========================
hyps_21 = [
    {'id':'h21_1','text':'The PFS benefit of treatment_olaparib varies with age_years (treatment_olaparib×age_years interaction nonzero).','kind':'novel'},
    {'id':'h21_2','text':'The PFS benefit of treatment_docetaxel varies with age_years (negative interaction expected).','kind':'novel'},
    {'id':'h21_3','text':'In a quadratic model age_years + age_years², pfs_months has a non-monotonic relationship with age (significant quadratic term).','kind':'novel'},
]
ana_21 = [
    interaction_lm('treatment_olaparib','age_years','h21_1','olaparib×age_years interaction'),
    interaction_lm('treatment_docetaxel','age_years','h21_2','docetaxel×age_years interaction'),
]
df['age2'] = df['age_years']**2
res_q = sm.OLS(df[OUTCOME], sm.add_constant(df[['age_years','age2']])).fit()
coef_q = float(res_q.params['age2']); p_q = float(res_q.pvalues['age2'])
ana_21.append({'hypothesis_ids':['h21_3'],
    'code':"OLS pfs ~ age_years + age_years**2",
    'result_summary':f"Quadratic age model: β(age²)={coef_q:+.6f}, p={p_q:.3g}, R²={res_q.rsquared:.4f}",
    'p_value':p_q,'effect_estimate':coef_q,'significant':bool(p_q<0.05)})
add_iter(21, hyps_21, ana_21)

# ========================= Iteration 22: subgroup biomarker effects in mCRPC vs non-mCRPC =========================
hyps_22 = [
    {'id':'h22_1','text':'In mcrpc=1 patients, mean pfs_months is lower for ar_v7_positive=1 vs 0.','kind':'novel'},
    {'id':'h22_2','text':'In mcrpc=1 patients, mean pfs_months is higher for psma_high=1 vs 0.','kind':'novel'},
    {'id':'h22_3','text':'In mcrpc=0 patients, ar_v7_positive does not affect mean pfs_months.','kind':'novel'},
]
ana_22 = [
    subgroup_effect('ar_v7_positive','mcrpc',1,'h22_1','ar_v7 in mCRPC'),
    subgroup_effect('psma_high','mcrpc',1,'h22_2','psma_high in mCRPC'),
    subgroup_effect('ar_v7_positive','mcrpc',0,'h22_3','ar_v7 in non-mCRPC'),
]
add_iter(22, hyps_22, ana_22)

# ========================= Iteration 23: confirm signs of top continuous predictors via standardized betas =========================
hyps_23 = [
    {'id':'h23_1','text':'In a univariate model age_years vs pfs_months, the slope coefficient is positive and large in magnitude.','kind':'refined'},
    {'id':'h23_2','text':'In a univariate model ecog_ps vs pfs_months, the slope coefficient is negative.','kind':'refined'},
    {'id':'h23_3','text':'In a univariate model albumin_g_dl vs pfs_months, the slope is positive.','kind':'refined'},
    {'id':'h23_4','text':'In a univariate model weight_loss_pct_6mo vs pfs_months, the slope is negative.','kind':'refined'},
]
def univariate_ols(col, hyp_id, label):
    X = sm.add_constant(df[[col]])
    res = sm.OLS(df[OUTCOME], X).fit()
    coef = float(res.params[col]); p = float(res.pvalues[col])
    return {'hypothesis_ids':[hyp_id],
            'code':f"OLS pfs ~ {col}",
            'result_summary':f"{label}: β({col})={coef:+.4f}, p={p:.3g}, R²={res.rsquared:.4f}",
            'p_value':p,'effect_estimate':coef,'significant':bool(p<0.05)}
ana_23 = [
    univariate_ols('age_years','h23_1','univariate age'),
    univariate_ols('ecog_ps','h23_2','univariate ECOG'),
    univariate_ols('albumin_g_dl','h23_3','univariate albumin'),
    univariate_ols('weight_loss_pct_6mo','h23_4','univariate weight_loss'),
]
add_iter(23, hyps_23, ana_23)

# ========================= Iteration 24: check whether SES variables remain after adjustment =========================
hyps_24 = [
    {'id':'h24_1','text':'After adjusting for age_years, ecog_ps, mcrpc, albumin_g_dl, ldh_u_l, weight_loss_pct_6mo, the rural_residence indicator does not predict pfs_months.','kind':'refined'},
    {'id':'h24_2','text':'After adjustment, education_years does not predict pfs_months.','kind':'refined'},
    {'id':'h24_3','text':'After adjustment, smoking_pack_years does not predict pfs_months.','kind':'refined'},
]
adj_set = ['age_years','ecog_ps','mcrpc','albumin_g_dl','ldh_u_l','weight_loss_pct_6mo']
ana_24 = [
    ols_reg(['rural_residence']+adj_set,'h24_1','adjusted rural_residence',slope_for='rural_residence'),
    ols_reg(['education_years']+adj_set,'h24_2','adjusted education_years',slope_for='education_years'),
    ols_reg(['smoking_pack_years']+adj_set,'h24_3','adjusted smoking_pack_years',slope_for='smoking_pack_years'),
]
add_iter(24, hyps_24, ana_24)

# ========================= Iteration 25: final composite model =========================
hyps_25 = [
    {'id':'h25_1','text':'A multivariable OLS combining the top significant prognostic predictors (age_years, ecog_ps, mcrpc, albumin_g_dl, ldh_u_l, psa_ng_ml, weight_loss_pct_6mo, alkaline_phosphatase_u_l, hemoglobin_g_dl) plus a matched_therapy indicator and the olaparib×brca2_mutation interaction explains substantially more variance than prognostic features alone (matched_therapy coefficient positive and significant).','kind':'refined'},
    {'id':'h25_2','text':'In the final model, the olaparib×brca2_mutation interaction term is positive and significant.','kind':'refined'},
]
final_cols = ['age_years','ecog_ps','mcrpc','albumin_g_dl','ldh_u_l','psa_ng_ml',
              'weight_loss_pct_6mo','alkaline_phosphatase_u_l','hemoglobin_g_dl',
              'matched_therapy','treatment_olaparib','brca2_mutation']
df['olaparib_x_brca2'] = df['treatment_olaparib'] * df['brca2_mutation']
final_cols2 = final_cols + ['olaparib_x_brca2']
res_f = sm.OLS(df[OUTCOME], sm.add_constant(df[final_cols2])).fit()
coef_mt = float(res_f.params['matched_therapy']); p_mt = float(res_f.pvalues['matched_therapy'])
coef_int = float(res_f.params['olaparib_x_brca2']); p_int = float(res_f.pvalues['olaparib_x_brca2'])
ana_25 = [
    {'hypothesis_ids':['h25_1'],
     'code':"OLS pfs ~ prognostic + matched_therapy + olaparib + brca2 + olaparib*brca2",
     'result_summary':f"Final MV model R²={res_f.rsquared:.4f}, n={int(res_f.nobs)}; β(matched_therapy)={coef_mt:+.4f}, p={p_mt:.3g}",
     'p_value':p_mt,'effect_estimate':coef_mt,'significant':bool(p_mt<0.05)},
    {'hypothesis_ids':['h25_2'],
     'code':"same model, coef of olaparib_x_brca2",
     'result_summary':f"Final MV model β(olaparib×brca2)={coef_int:+.4f}, p={p_int:.3g}, R²={res_f.rsquared:.4f}",
     'p_value':p_int,'effect_estimate':coef_int,'significant':bool(p_int<0.05)},
]
add_iter(25, hyps_25, ana_25)

# ============================== Build transcript ==============================
transcript = {
    'dataset_id':'ds001_prostate',
    'model_id':'claude-opus-4-7',
    'harness_id':'manual-claude-code@1.0',
    'max_iterations':25,
    'iterations':iterations,
}
with open('transcript.json','w') as fh:
    json.dump(transcript, fh, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))

# Print a compact summary
print(f"Iterations recorded: {len(iterations)}")
total_h = sum(len(i['proposed_hypotheses']) for i in iterations)
total_a = sum(len(i['analyses']) for i in iterations)
print(f"Total hypotheses: {total_h}")
print(f"Total analyses: {total_a}")
sig = sum(1 for it in iterations for a in it['analyses'] if a.get('significant'))
print(f"Significant: {sig}/{total_a}")

# Brief summary by iteration
for it in iterations:
    print(f"--- Iter {it['index']} ---")
    for a in it['analyses']:
        eff = a.get('effect_estimate'); p = a.get('p_value')
        eff_s = f"{eff:+.4f}" if eff is not None else "NA"
        p_s = f"{p:.3g}" if p is not None else "NA"
        print(f"  {a['hypothesis_ids']} eff={eff_s} p={p_s} sig={a.get('significant')}")
        print(f"    {a['result_summary']}")

print("\nTranscript written.")
