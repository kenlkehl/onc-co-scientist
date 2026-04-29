"""Additional targeted analyses for the transcript."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

with open('analysis_results.json') as f:
    results = json.load(f)

# ============ Additional Iter: Olaparib×BRCA2 stratified estimates ============
# Print the conditional effects already computed
o_b = results.get('inter_treatment_olaparib_brca2_mutation', {})
print(f"Olaparib in BRCA2+ effect: {o_b.get('effect_in_b1'):.3f} months")
print(f"Olaparib in BRCA2- effect: {o_b.get('effect_in_b0'):.3f} months")

# Test BRCA2+ olaparib t-test
sub = df[df['brca2_mutation'] == 1]
a = sub.loc[sub['treatment_olaparib'] == 1, 'pfs_months']
b = sub.loc[sub['treatment_olaparib'] == 0, 'pfs_months']
t, p = stats.ttest_ind(a, b, equal_var=False)
results['strat_olap_in_brca2pos'] = dict(diff=float(a.mean()-b.mean()), p=float(p), n_on=int(len(a)), n_off=int(len(b)))
print(f"BRCA2+: olaparib vs not, n={len(a)} vs {len(b)}, diff={a.mean()-b.mean():.3f}, p={p:.4g}")

# BRCA2- olaparib t-test
sub = df[df['brca2_mutation'] == 0]
a = sub.loc[sub['treatment_olaparib'] == 1, 'pfs_months']
b = sub.loc[sub['treatment_olaparib'] == 0, 'pfs_months']
t, p = stats.ttest_ind(a, b, equal_var=False)
results['strat_olap_in_brca2neg'] = dict(diff=float(a.mean()-b.mean()), p=float(p), n_on=int(len(a)), n_off=int(len(b)))
print(f"BRCA2-: olaparib vs not, diff={a.mean()-b.mean():.3f}, p={p:.4g}")

# ============ Other genes ============
genes = ['tp53_mutation', 'pten_loss', 'cdkn2a_loss', 'pik3ca_mutation',
         'her2_amplification', 'fgfr_alteration', 'keap1_mutation',
         'met_exon14_skipping', 'ret_fusion', 'ros1_fusion', 'braf_v600e',
         'ntrk_fusion', 'nrg1_fusion']
for g in genes:
    a = df.loc[df[g] == 1, 'pfs_months']
    b = df.loc[df[g] == 0, 'pfs_months']
    if len(a) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results[f'gene_{g}'] = dict(diff=float(a.mean()-b.mean()), p=float(p), n_on=int(len(a)))
        print(f"{g}: n_on={len(a)}, diff={a.mean()-b.mean():.3f}, p={p:.4g}")

# ============ Additional symptom interactions adjusted ============
# Adjusted comorbidities (adjusted for age, ecog)
covars = ['age_years', 'ecog_ps', 'mcrpc', 'gleason_score', 'albumin_g_dl', 'ldh_u_l', 'psa_ng_ml']
comorb = ['diabetes_mellitus','hypertension','copd','chronic_kidney_disease',
          'heart_failure','coronary_artery_disease','atrial_fibrillation',
          'autoimmune_disease','depression_anxiety_diagnosis','prior_malignancy']
for c in comorb:
    formula = f"pfs_months ~ {c} + " + " + ".join(covars)
    m = smf.ols(formula, data=df).fit()
    results[f'adj_comorb_{c}'] = dict(coef=float(m.params[c]), p=float(m.pvalues[c]))

# ============ Vital signs and other labs ============
others = ['systolic_bp_mmhg', 'diastolic_bp_mmhg', 'heart_rate_bpm', 'spo2_pct', 'bmi',
          'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl', 'glucose_mg_dl', 'tsh_uiu_ml',
          'ca_125_u_ml', 'cea_ng_ml', 'inr', 'wbc_k_ul', 'anc_k_ul', 'alc_k_ul',
          'platelets_k_ul', 'bun_mg_dl', 'ast_u_l', 'alt_u_l', 'total_bilirubin_mg_dl',
          'creatinine_mg_dl']
for v in others:
    m = smf.ols(f"pfs_months ~ {v}", data=df).fit()
    results[f'lin_{v}'] = dict(beta=float(m.params[v]), p=float(m.pvalues[v]))

# ============ ECOG-stratified treatment effects ============
treatments = ['treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel',
              'treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab']
for tx in treatments:
    sub_eff = {}
    for ev in [0,1,2]:
        sub = df[df['ecog_ps']==ev]
        a = sub.loc[sub[tx]==1, 'pfs_months']
        b = sub.loc[sub[tx]==0, 'pfs_months']
        if len(a)>5 and len(b)>5:
            t,p = stats.ttest_ind(a, b, equal_var=False)
            sub_eff[ev] = dict(diff=float(a.mean()-b.mean()), p=float(p), n_on=int(len(a)))
    results[f'ecog_strat_{tx}'] = sub_eff

# ============ Full multivariable model with ALL genes and treatments ============
genes_all = ['brca2_mutation','tp53_mutation','pten_loss','cdkn2a_loss','pik3ca_mutation',
             'her2_amplification','fgfr_alteration','keap1_mutation','msi_high',
             'ar_v7_positive','psma_high','met_exon14_skipping','ret_fusion','ros1_fusion',
             'braf_v600e','ntrk_fusion','nrg1_fusion']
formula_genes = "pfs_months ~ " + " + ".join(genes_all + treatments +
    ['age_years','ecog_ps','gleason_score','mcrpc','visceral_mets','albumin_g_dl',
     'ldh_u_l','psa_ng_ml','weight_loss_pct_6mo'])
m_genes_full = smf.ols(formula_genes, data=df).fit()
gene_adj = {}
for g in genes_all:
    gene_adj[g] = dict(coef=float(m_genes_full.params[g]), p=float(m_genes_full.pvalues[g]))
results['adj_genes'] = gene_adj
print("Adjusted gene coefficients:")
for g, v in gene_adj.items():
    print(f"  {g}: coef={v['coef']:.3f}, p={v['p']:.4g}")

# ============ Adjusted treatment-biomarker interactions ============
def adj_int(treatment, biomarker):
    formula = (f"pfs_months ~ {treatment} * {biomarker} + age_years + ecog_ps + "
               f"gleason_score + mcrpc + visceral_mets + albumin_g_dl + ldh_u_l + "
               f"psa_ng_ml + weight_loss_pct_6mo")
    m = smf.ols(formula, data=df).fit()
    interaction_term = f"{treatment}:{biomarker}"
    return float(m.params[interaction_term]), float(m.pvalues[interaction_term])

for (tx,bm) in [('treatment_olaparib','brca2_mutation'),
                ('treatment_pembrolizumab','msi_high'),
                ('treatment_lu177_psma','psma_high'),
                ('treatment_enzalutamide','ar_v7_positive'),
                ('treatment_abiraterone','ar_v7_positive')]:
    coef, p = adj_int(tx,bm)
    results[f'adj_int_{tx}_{bm}'] = dict(coef=coef, p=p)
    print(f"adj_int {tx}*{bm}: coef={coef:.3f}, p={p:.4g}")

# ============ Adjusted race/insurance ============
m_ar = smf.ols(
    "pfs_months ~ C(race_ethnicity) + age_years + ecog_ps + gleason_score + mcrpc + "
    "visceral_mets + albumin_g_dl + ldh_u_l + psa_ng_ml + weight_loss_pct_6mo",
    data=df).fit()
race_adj = {}
for k in m_ar.params.index:
    if 'race_ethnicity' in k:
        race_adj[k] = dict(coef=float(m_ar.params[k]), p=float(m_ar.pvalues[k]))
results['adj_race'] = race_adj
print("Adjusted race:", race_adj)

m_ai = smf.ols(
    "pfs_months ~ C(insurance_type) + age_years + ecog_ps + gleason_score + mcrpc + "
    "visceral_mets + albumin_g_dl + ldh_u_l + psa_ng_ml + weight_loss_pct_6mo",
    data=df).fit()
ins_adj = {}
for k in m_ai.params.index:
    if 'insurance_type' in k:
        ins_adj[k] = dict(coef=float(m_ai.params[k]), p=float(m_ai.pvalues[k]))
results['adj_insurance'] = ins_adj
print("Adjusted insurance:", ins_adj)

# ============ Composite Halabi-style risk ============
df['low_albumin'] = (df['albumin_g_dl'] < 3.5).astype(int)
df['high_ldh'] = (df['ldh_u_l'] > 250).astype(int)
df['high_alp'] = (df['alkaline_phosphatase_u_l'] > 150).astype(int)
df['high_ecog'] = (df['ecog_ps'] >= 1).astype(int)
df['risk_score'] = df['low_albumin']+df['high_ldh']+df['high_alp']+df['high_ecog']
m_r = smf.ols("pfs_months ~ risk_score", data=df).fit()
results['composite_risk_score'] = dict(coef=float(m_r.params['risk_score']),
                                        p=float(m_r.pvalues['risk_score']))
for s in [0,1,2,3,4]:
    sub = df[df['risk_score']==s]
    if len(sub)>20:
        results[f'risk_score_{s}'] = dict(n=int(len(sub)),
                                           mean=float(sub['pfs_months'].mean()))

# ============ Mets count ============
df['mets_count'] = df['visceral_mets']+df['liver_mets']+df['bone_mets']+df['adrenal_mets']
m_m = smf.ols("pfs_months ~ mets_count", data=df).fit()
results['mets_count'] = dict(coef=float(m_m.params['mets_count']),
                              p=float(m_m.pvalues['mets_count']))

# ============ Triple interaction olaparib×BRCA2 by mCRPC ============
for mv in [0,1]:
    sub = df[df['mcrpc']==mv]
    m = smf.ols("pfs_months ~ treatment_olaparib * brca2_mutation", data=sub).fit()
    interaction_term = "treatment_olaparib:brca2_mutation"
    results[f'mcrpc{mv}_olap_brca2'] = dict(
        coef=float(m.params[interaction_term]),
        p=float(m.pvalues[interaction_term]),
        n=int(len(sub)))

# ============ Age subgroup ============
df['age_75plus'] = (df['age_years']>=75).astype(int)
for tx in treatments:
    m = smf.ols(f"pfs_months ~ {tx} * age_75plus", data=df).fit()
    interaction_term = f"{tx}:age_75plus"
    results[f'age75_int_{tx}'] = dict(coef=float(m.params[interaction_term]),
                                       p=float(m.pvalues[interaction_term]))

# Save
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved {len(results)} entries.")
