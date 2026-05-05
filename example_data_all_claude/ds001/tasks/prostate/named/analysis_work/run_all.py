"""Systematic analyses across 25 iterations for ds001_prostate."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('../dataset.parquet')
print(f"Loaded {len(df)} patients, {df.shape[1]} columns")

# Treatments and outcome
TREATMENTS = ['treatment_enzalutamide', 'treatment_abiraterone', 'treatment_docetaxel',
              'treatment_olaparib', 'treatment_lu177_psma', 'treatment_pembrolizumab']
OUTCOME = 'objective_response'
BIOMARKERS = ['brca2_mutation', 'ar_v7_positive', 'msi_high', 'psma_high']
CONTINUOUS = ['age_years', 'psa_ng_ml', 'albumin_g_dl', 'ldh_u_l', 'weight_loss_pct_6mo',
              'crp_mg_l', 'nlr', 'hemoglobin_g_dl', 'alkaline_phosphatase_u_l',
              'ast_u_l', 'alt_u_l', 'total_bilirubin_mg_dl', 'creatinine_mg_dl',
              'bun_mg_dl', 'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl']
BINARY_FEATS = ['mcrpc', 'visceral_mets', 'brca2_mutation', 'ar_v7_positive',
                'msi_high', 'psma_high']

results = {}

def diff_prop(g1, g0, name=''):
    n1, n0 = len(g1), len(g0)
    p1 = g1.mean() if n1>0 else np.nan
    p0 = g0.mean() if n0>0 else np.nan
    diff = p1 - p0
    if n1==0 or n0==0:
        return diff, np.nan
    # Two-proportion z-test
    table = np.array([[g1.sum(), n1-g1.sum()], [g0.sum(), n0-g0.sum()]])
    if (table==0).any() or n1<5 or n0<5:
        try:
            _, p = stats.fisher_exact(table)
        except Exception:
            p = np.nan
    else:
        chi2, p, _, _ = stats.chi2_contingency(table)
    return diff, p

def logit_or(formula, data):
    try:
        m = smf.logit(formula, data=data).fit(disp=0, maxiter=100)
        return m
    except Exception as e:
        print(f"Failed: {formula}: {e}")
        return None

# === ITERATION 1: Marginal main-effect of each treatment on objective response ===
it1 = []
for t in TREATMENTS:
    g1 = df.loc[df[t]==1, OUTCOME]
    g0 = df.loc[df[t]==0, OUTCOME]
    diff, p = diff_prop(g1, g0)
    it1.append({'treatment': t, 'rr_treated': float(g1.mean()), 'rr_untreated': float(g0.mean()),
                'diff': float(diff), 'p_value': float(p), 'n_treated': int(len(g1))})
results['iter1_treatment_main'] = it1
print("Iter1:", it1)

# === ITERATION 2: Continuous feature univariate associations with outcome (logistic regression) ===
it2 = []
for c in CONTINUOUS:
    m = logit_or(f"{OUTCOME} ~ {c}", df)
    if m is None:
        continue
    coef = float(m.params[c])
    p = float(m.pvalues[c])
    it2.append({'feature': c, 'logit_coef': coef, 'p_value': p, 'OR': float(np.exp(coef))})
results['iter2_cont_main'] = it2

# === ITERATION 3: Binary feature univariate associations ===
it3 = []
for b in BINARY_FEATS:
    g1 = df.loc[df[b]==1, OUTCOME]
    g0 = df.loc[df[b]==0, OUTCOME]
    diff, p = diff_prop(g1, g0)
    it3.append({'feature': b, 'rr_pos': float(g1.mean()), 'rr_neg': float(g0.mean()),
                'diff': float(diff), 'p_value': float(p)})
results['iter3_bin_main'] = it3

# === ITERATION 4: Disease state (mcrpc, visceral_mets, ecog_ps) effects ===
it4 = []
for f in ['mcrpc','visceral_mets','ecog_ps']:
    if f=='ecog_ps':
        # Spearman or treat as categorical
        m = logit_or(f"{OUTCOME} ~ {f}", df)
        coef = float(m.params[f])
        p = float(m.pvalues[f])
        it4.append({'feature': f, 'logit_coef': coef, 'p_value': p, 'OR': float(np.exp(coef))})
    else:
        g1 = df.loc[df[f]==1, OUTCOME]
        g0 = df.loc[df[f]==0, OUTCOME]
        diff, p = diff_prop(g1, g0)
        it4.append({'feature': f, 'rr_pos': float(g1.mean()), 'rr_neg': float(g0.mean()),
                    'diff': float(diff), 'p_value': float(p)})
results['iter4_disease'] = it4

# === ITERATION 5: Olaparib x BRCA2 interaction ===
df['olap_brca2'] = df['treatment_olaparib']*df['brca2_mutation']
m = logit_or(f"{OUTCOME} ~ treatment_olaparib*brca2_mutation", df)
inter_p = float(m.pvalues['treatment_olaparib:brca2_mutation'])
inter_coef = float(m.params['treatment_olaparib:brca2_mutation'])
# Stratified RR
g_olap_brca = df.loc[(df['treatment_olaparib']==1)&(df['brca2_mutation']==1), OUTCOME]
g_olap_no = df.loc[(df['treatment_olaparib']==1)&(df['brca2_mutation']==0), OUTCOME]
g_no_olap_brca = df.loc[(df['treatment_olaparib']==0)&(df['brca2_mutation']==1), OUTCOME]
g_no_olap_no = df.loc[(df['treatment_olaparib']==0)&(df['brca2_mutation']==0), OUTCOME]
te_brca = g_olap_brca.mean() - g_no_olap_brca.mean()
te_nobrca = g_olap_no.mean() - g_no_olap_no.mean()
results['iter5_olap_brca'] = {'inter_coef': inter_coef, 'inter_p': inter_p,
    'rr_olap_brca': float(g_olap_brca.mean()), 'rr_no_olap_brca': float(g_no_olap_brca.mean()),
    'rr_olap_nobrca': float(g_olap_no.mean()), 'rr_no_olap_nobrca': float(g_no_olap_no.mean()),
    'te_brca_pos': float(te_brca), 'te_brca_neg': float(te_nobrca),
    'n_olap_brca': int(len(g_olap_brca)), 'n_olap_nobrca': int(len(g_olap_no))}

# === ITERATION 6: Pembro x MSI-high interaction ===
m = logit_or(f"{OUTCOME} ~ treatment_pembrolizumab*msi_high", df)
inter_p = float(m.pvalues['treatment_pembrolizumab:msi_high'])
inter_coef = float(m.params['treatment_pembrolizumab:msi_high'])
g11 = df.loc[(df['treatment_pembrolizumab']==1)&(df['msi_high']==1), OUTCOME]
g10 = df.loc[(df['treatment_pembrolizumab']==1)&(df['msi_high']==0), OUTCOME]
g01 = df.loc[(df['treatment_pembrolizumab']==0)&(df['msi_high']==1), OUTCOME]
g00 = df.loc[(df['treatment_pembrolizumab']==0)&(df['msi_high']==0), OUTCOME]
results['iter6_pembro_msi'] = {'inter_coef': inter_coef, 'inter_p': inter_p,
    'rr_pem_msi': float(g11.mean()), 'rr_nopem_msi': float(g01.mean()),
    'rr_pem_nomsi': float(g10.mean()), 'rr_nopem_nomsi': float(g00.mean()),
    'te_msi_pos': float(g11.mean()-g01.mean()), 'te_msi_neg': float(g10.mean()-g00.mean()),
    'n_pem_msi': int(len(g11)), 'n_pem_nomsi': int(len(g10))}

# === ITERATION 7: Lu177-PSMA x PSMA-high interaction ===
m = logit_or(f"{OUTCOME} ~ treatment_lu177_psma*psma_high", df)
inter_p = float(m.pvalues['treatment_lu177_psma:psma_high'])
inter_coef = float(m.params['treatment_lu177_psma:psma_high'])
g11 = df.loc[(df['treatment_lu177_psma']==1)&(df['psma_high']==1), OUTCOME]
g10 = df.loc[(df['treatment_lu177_psma']==1)&(df['psma_high']==0), OUTCOME]
g01 = df.loc[(df['treatment_lu177_psma']==0)&(df['psma_high']==1), OUTCOME]
g00 = df.loc[(df['treatment_lu177_psma']==0)&(df['psma_high']==0), OUTCOME]
results['iter7_lu177_psma'] = {'inter_coef': inter_coef, 'inter_p': inter_p,
    'rr_lu_psma': float(g11.mean()), 'rr_nolu_psma': float(g01.mean()),
    'rr_lu_nopsma': float(g10.mean()), 'rr_nolu_nopsma': float(g00.mean()),
    'te_psma_pos': float(g11.mean()-g01.mean()), 'te_psma_neg': float(g10.mean()-g00.mean()),
    'n_lu_psma': int(len(g11)), 'n_lu_nopsma': int(len(g10))}

# === ITERATION 8: AR-V7 x enzalutamide and AR-V7 x abiraterone ===
it8 = {}
for t in ['treatment_enzalutamide', 'treatment_abiraterone']:
    m = logit_or(f"{OUTCOME} ~ {t}*ar_v7_positive", df)
    inter_term = f'{t}:ar_v7_positive'
    inter_p = float(m.pvalues[inter_term])
    inter_coef = float(m.params[inter_term])
    g11 = df.loc[(df[t]==1)&(df['ar_v7_positive']==1), OUTCOME]
    g10 = df.loc[(df[t]==1)&(df['ar_v7_positive']==0), OUTCOME]
    g01 = df.loc[(df[t]==0)&(df['ar_v7_positive']==1), OUTCOME]
    g00 = df.loc[(df[t]==0)&(df['ar_v7_positive']==0), OUTCOME]
    it8[t] = {'inter_coef': inter_coef, 'inter_p': inter_p,
        'te_arv7_pos': float(g11.mean()-g01.mean()), 'te_arv7_neg': float(g10.mean()-g00.mean()),
        'rr_t_arv7p': float(g11.mean()), 'rr_not_arv7p': float(g01.mean()),
        'rr_t_arv7n': float(g10.mean()), 'rr_not_arv7n': float(g00.mean())}
results['iter8_arv7'] = it8

# === ITERATION 9: Treatment x ECOG, mcrpc, visceral mets ===
it9 = {}
for t in TREATMENTS:
    sub = {}
    for f in ['ecog_ps', 'mcrpc', 'visceral_mets']:
        m = logit_or(f"{OUTCOME} ~ {t}*{f}", df)
        inter_term = f'{t}:{f}'
        if inter_term in m.pvalues:
            sub[f] = {'inter_coef': float(m.params[inter_term]), 'inter_p': float(m.pvalues[inter_term])}
    it9[t] = sub
results['iter9_clin_inter'] = it9

# === ITERATION 10: Multivariable adjusted main effect of each treatment, controlling for prognostics ===
adj_covs = ['age_years', 'ecog_ps', 'mcrpc', 'visceral_mets', 'psa_ng_ml',
            'albumin_g_dl', 'ldh_u_l', 'hemoglobin_g_dl', 'alkaline_phosphatase_u_l',
            'gleason_score', 'crp_mg_l', 'nlr', 'weight_loss_pct_6mo']
it10 = {}
for t in TREATMENTS:
    formula = f"{OUTCOME} ~ {t} + " + " + ".join(adj_covs)
    m = logit_or(formula, df)
    if m is not None:
        it10[t] = {'adj_coef': float(m.params[t]), 'adj_p': float(m.pvalues[t]),
                   'adj_OR': float(np.exp(m.params[t]))}
results['iter10_adjusted'] = it10

# === ITERATION 11: Multivariable model with all treatments together ===
formula = f"{OUTCOME} ~ " + " + ".join(TREATMENTS) + " + " + " + ".join(adj_covs)
m = logit_or(formula, df)
it11 = {}
for t in TREATMENTS:
    it11[t] = {'coef': float(m.params[t]), 'p': float(m.pvalues[t]), 'OR': float(np.exp(m.params[t]))}
results['iter11_all_tx'] = it11

# === ITERATION 12: Continuous biomarker x treatment interactions (LDH, albumin, CRP, NLR) ===
it12 = {}
for t in TREATMENTS:
    sub = {}
    for f in ['ldh_u_l', 'albumin_g_dl', 'crp_mg_l', 'nlr', 'psa_ng_ml', 'alkaline_phosphatase_u_l']:
        m = logit_or(f"{OUTCOME} ~ {t}*{f}", df)
        inter_term = f'{t}:{f}'
        if m is not None and inter_term in m.pvalues:
            sub[f] = {'inter_coef': float(m.params[inter_term]), 'inter_p': float(m.pvalues[inter_term])}
    it12[t] = sub
results['iter12_cont_inter'] = it12

# === ITERATION 13: Joint subgroup test - Olaparib effect in BRCA2+ patients with good performance status (ECOG 0-1) ===
it13 = {}
# Olaparib in BRCA2+ AND good ECOG (<=1) AND good albumin (>3.5)
for cond_name, cond in [
    ('brca2_pos', df['brca2_mutation']==1),
    ('brca2_pos_ecog01', (df['brca2_mutation']==1)&(df['ecog_ps']<=1)),
    ('brca2_pos_alb_high', (df['brca2_mutation']==1)&(df['albumin_g_dl']>=3.5)),
    ('brca2_pos_ecog01_alb_high', (df['brca2_mutation']==1)&(df['ecog_ps']<=1)&(df['albumin_g_dl']>=3.5)),
]:
    sub = df.loc[cond]
    g1 = sub.loc[sub['treatment_olaparib']==1, OUTCOME]
    g0 = sub.loc[sub['treatment_olaparib']==0, OUTCOME]
    diff, p = diff_prop(g1, g0)
    it13[cond_name] = {'rr_olap': float(g1.mean()) if len(g1)>0 else None,
        'rr_no_olap': float(g0.mean()) if len(g0)>0 else None,
        'diff': float(diff) if not np.isnan(diff) else None, 'p': float(p) if not np.isnan(p) else None,
        'n_olap': int(len(g1)), 'n_no_olap': int(len(g0))}
results['iter13_olap_subgroup'] = it13

# === ITERATION 14: Joint subgroup test - Pembrolizumab in MSI-high with various modifiers ===
it14 = {}
for cond_name, cond in [
    ('msi_high', df['msi_high']==1),
    ('msi_high_ecog01', (df['msi_high']==1)&(df['ecog_ps']<=1)),
    ('msi_high_alb_high', (df['msi_high']==1)&(df['albumin_g_dl']>=3.5)),
    ('msi_high_no_visceral', (df['msi_high']==1)&(df['visceral_mets']==0)),
    ('msi_high_low_ldh', (df['msi_high']==1)&(df['ldh_u_l']<250)),
    ('msi_high_ecog01_alb_high', (df['msi_high']==1)&(df['ecog_ps']<=1)&(df['albumin_g_dl']>=3.5)),
]:
    sub = df.loc[cond]
    g1 = sub.loc[sub['treatment_pembrolizumab']==1, OUTCOME]
    g0 = sub.loc[sub['treatment_pembrolizumab']==0, OUTCOME]
    diff, p = diff_prop(g1, g0)
    it14[cond_name] = {'rr_pem': float(g1.mean()) if len(g1)>0 else None,
        'rr_no_pem': float(g0.mean()) if len(g0)>0 else None,
        'diff': float(diff) if not np.isnan(diff) else None, 'p': float(p) if not np.isnan(p) else None,
        'n_pem': int(len(g1)), 'n_no_pem': int(len(g0))}
results['iter14_pembro_subgroup'] = it14

# === ITERATION 15: Joint subgroup - Lu177 in PSMA-high with modifiers ===
it15 = {}
for cond_name, cond in [
    ('psma_high', df['psma_high']==1),
    ('psma_high_ecog01', (df['psma_high']==1)&(df['ecog_ps']<=1)),
    ('psma_high_alb_high', (df['psma_high']==1)&(df['albumin_g_dl']>=3.5)),
    ('psma_high_no_visceral', (df['psma_high']==1)&(df['visceral_mets']==0)),
    ('psma_high_low_ldh', (df['psma_high']==1)&(df['ldh_u_l']<250)),
    ('psma_high_ecog01_alb_high', (df['psma_high']==1)&(df['ecog_ps']<=1)&(df['albumin_g_dl']>=3.5)),
    ('psma_high_alb_high_low_ldh', (df['psma_high']==1)&(df['albumin_g_dl']>=3.5)&(df['ldh_u_l']<250)),
]:
    sub = df.loc[cond]
    g1 = sub.loc[sub['treatment_lu177_psma']==1, OUTCOME]
    g0 = sub.loc[sub['treatment_lu177_psma']==0, OUTCOME]
    diff, p = diff_prop(g1, g0)
    it15[cond_name] = {'rr_lu': float(g1.mean()) if len(g1)>0 else None,
        'rr_no_lu': float(g0.mean()) if len(g0)>0 else None,
        'diff': float(diff) if not np.isnan(diff) else None, 'p': float(p) if not np.isnan(p) else None,
        'n_lu': int(len(g1)), 'n_no_lu': int(len(g0))}
results['iter15_lu177_subgroup'] = it15

# === ITERATION 16: Enzalutamide and Abiraterone in AR-V7 negative subgroup ===
it16 = {}
for t in ['treatment_enzalutamide', 'treatment_abiraterone']:
    for cond_name, cond in [
        ('arv7_neg', df['ar_v7_positive']==0),
        ('arv7_neg_no_visceral', (df['ar_v7_positive']==0)&(df['visceral_mets']==0)),
        ('arv7_neg_ecog01', (df['ar_v7_positive']==0)&(df['ecog_ps']<=1)),
        ('arv7_pos', df['ar_v7_positive']==1),
    ]:
        sub = df.loc[cond]
        g1 = sub.loc[sub[t]==1, OUTCOME]
        g0 = sub.loc[sub[t]==0, OUTCOME]
        diff, p = diff_prop(g1, g0)
        it16[f"{t}__{cond_name}"] = {'rr_t': float(g1.mean()) if len(g1)>0 else None,
            'rr_not': float(g0.mean()) if len(g0)>0 else None,
            'diff': float(diff) if not np.isnan(diff) else None, 'p': float(p) if not np.isnan(p) else None,
            'n_t': int(len(g1)), 'n_not': int(len(g0))}
results['iter16_arv7_subgroup'] = it16

# === ITERATION 17: Docetaxel - check for prognostic subgroups (ECOG, visceral, mcrpc) ===
it17 = {}
for cond_name, cond in [
    ('all', pd.Series([True]*len(df))),
    ('mcrpc_pos', df['mcrpc']==1),
    ('mcrpc_neg', df['mcrpc']==0),
    ('visceral', df['visceral_mets']==1),
    ('no_visceral', df['visceral_mets']==0),
    ('ecog01', df['ecog_ps']<=1),
    ('ecog2', df['ecog_ps']==2),
    ('alb_high', df['albumin_g_dl']>=3.5),
    ('alb_low', df['albumin_g_dl']<3.5),
]:
    sub = df.loc[cond]
    g1 = sub.loc[sub['treatment_docetaxel']==1, OUTCOME]
    g0 = sub.loc[sub['treatment_docetaxel']==0, OUTCOME]
    diff, p = diff_prop(g1, g0)
    it17[cond_name] = {'rr_t': float(g1.mean()) if len(g1)>0 else None,
        'rr_not': float(g0.mean()) if len(g0)>0 else None,
        'diff': float(diff) if not np.isnan(diff) else None, 'p': float(p) if not np.isnan(p) else None,
        'n_t': int(len(g1)), 'n_not': int(len(g0))}
results['iter17_docetaxel_subgroup'] = it17

# === ITERATION 18: Cross-treatment effect modification - is the predictive effect of BRCA2 on olaparib response moderated by other factors? ===
it18 = {}
# Three-way: olaparib x brca2 x ecog
m = logit_or("objective_response ~ treatment_olaparib*brca2_mutation*ecog_ps", df)
it18['olap_brca2_ecog_3way'] = {
    'three_way_coef': float(m.params.get('treatment_olaparib:brca2_mutation:ecog_ps', np.nan)),
    'three_way_p': float(m.pvalues.get('treatment_olaparib:brca2_mutation:ecog_ps', np.nan))}
# olaparib x brca2 x visceral
m = logit_or("objective_response ~ treatment_olaparib*brca2_mutation*visceral_mets", df)
it18['olap_brca2_visceral_3way'] = {
    'three_way_coef': float(m.params.get('treatment_olaparib:brca2_mutation:visceral_mets', np.nan)),
    'three_way_p': float(m.pvalues.get('treatment_olaparib:brca2_mutation:visceral_mets', np.nan))}
# pembro x msi x ecog
m = logit_or("objective_response ~ treatment_pembrolizumab*msi_high*ecog_ps", df)
it18['pem_msi_ecog_3way'] = {
    'three_way_coef': float(m.params.get('treatment_pembrolizumab:msi_high:ecog_ps', np.nan)),
    'three_way_p': float(m.pvalues.get('treatment_pembrolizumab:msi_high:ecog_ps', np.nan))}
# lu177 x psma x ecog
m = logit_or("objective_response ~ treatment_lu177_psma*psma_high*ecog_ps", df)
it18['lu_psma_ecog_3way'] = {
    'three_way_coef': float(m.params.get('treatment_lu177_psma:psma_high:ecog_ps', np.nan)),
    'three_way_p': float(m.pvalues.get('treatment_lu177_psma:psma_high:ecog_ps', np.nan))}
results['iter18_three_way'] = it18

# === ITERATION 19: Systematic treatment x feature interaction screen for each treatment ===
# All features
ALL_FEATS = ['age_years','ecog_ps','mcrpc','visceral_mets','psa_ng_ml','gleason_score',
             'brca2_mutation','ar_v7_positive','msi_high','psma_high','albumin_g_dl',
             'ldh_u_l','weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
             'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
             'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']
it19 = {}
for t in TREATMENTS:
    feat_results = []
    for f in ALL_FEATS:
        m = logit_or(f"{OUTCOME} ~ {t}*{f}", df)
        inter = f'{t}:{f}'
        if m is not None and inter in m.pvalues:
            feat_results.append({'feature': f, 'coef': float(m.params[inter]), 'p': float(m.pvalues[inter])})
    feat_results.sort(key=lambda x: x['p'])
    it19[t] = feat_results
results['iter19_screen'] = it19
# Print top 5 strongest interactions per treatment
for t, lst in it19.items():
    print(f"\nTop interactions for {t}:")
    for r in lst[:5]:
        print(f"  {r['feature']}: coef={r['coef']:.4f}, p={r['p']:.2e}")

# === ITERATION 20: Refined predictive subgroups - test ALL combinations of binary modifiers for each treatment ===
# For each treatment, find the binary feature pair that maximizes the treatment effect
it20 = {}
for t, top_feats in [
    ('treatment_olaparib', ['brca2_mutation','psma_high','msi_high','ar_v7_positive','mcrpc','visceral_mets']),
    ('treatment_pembrolizumab', ['msi_high','brca2_mutation','psma_high','ar_v7_positive','mcrpc','visceral_mets']),
    ('treatment_lu177_psma', ['psma_high','brca2_mutation','msi_high','ar_v7_positive','mcrpc','visceral_mets']),
    ('treatment_enzalutamide', ['ar_v7_positive','mcrpc','visceral_mets','brca2_mutation','psma_high','msi_high']),
    ('treatment_abiraterone', ['ar_v7_positive','mcrpc','visceral_mets','brca2_mutation','psma_high','msi_high']),
    ('treatment_docetaxel', ['mcrpc','visceral_mets','ar_v7_positive','brca2_mutation','psma_high','msi_high']),
]:
    sub_results = []
    for f1, f2 in combinations(top_feats, 2):
        # Try all 4 combos of f1,f2 directions
        for d1, d2 in [(1,1),(1,0),(0,1),(0,0)]:
            mask = (df[f1]==d1)&(df[f2]==d2)
            sub = df.loc[mask]
            if len(sub) < 50: continue
            g1 = sub.loc[sub[t]==1, OUTCOME]
            g0 = sub.loc[sub[t]==0, OUTCOME]
            if len(g1) < 20 or len(g0) < 20: continue
            diff, p = diff_prop(g1, g0)
            sub_results.append({'subgroup': f"{f1}={d1},{f2}={d2}", 'diff': float(diff),
                                'p': float(p), 'n_t': int(len(g1)), 'n_not': int(len(g0)),
                                'rr_t': float(g1.mean()), 'rr_not': float(g0.mean())})
    sub_results.sort(key=lambda x: x['diff'], reverse=True)
    it20[t] = sub_results[:10]  # top 10 by treatment effect
results['iter20_subgroup_search'] = it20

# === ITERATION 21: Three-way binary subgroup search ===
it21 = {}
for t, top_feats in [
    ('treatment_olaparib', ['brca2_mutation','mcrpc','visceral_mets']),
    ('treatment_pembrolizumab', ['msi_high','mcrpc','visceral_mets']),
    ('treatment_lu177_psma', ['psma_high','mcrpc','visceral_mets']),
]:
    sub_results = []
    for d1, d2, d3 in [(1,1,1),(1,1,0),(1,0,1),(1,0,0),(0,1,1),(0,1,0),(0,0,1),(0,0,0)]:
        mask = (df[top_feats[0]]==d1)&(df[top_feats[1]]==d2)&(df[top_feats[2]]==d3)
        sub = df.loc[mask]
        if len(sub) < 50: continue
        g1 = sub.loc[sub[t]==1, OUTCOME]
        g0 = sub.loc[sub[t]==0, OUTCOME]
        if len(g1) < 10 or len(g0) < 10: continue
        diff, p = diff_prop(g1, g0)
        sub_results.append({
            'subgroup': f"{top_feats[0]}={d1},{top_feats[1]}={d2},{top_feats[2]}={d3}",
            'diff': float(diff), 'p': float(p), 'n_t': int(len(g1)), 'n_not': int(len(g0)),
            'rr_t': float(g1.mean()), 'rr_not': float(g0.mean())})
    sub_results.sort(key=lambda x: x['diff'], reverse=True)
    it21[t] = sub_results
results['iter21_three_way_subgroup'] = it21

# === ITERATION 22: For each predictive treatment, test for "suppressor" variables that reduce treatment effect ===
# E.g., olaparib in BRCA2 - does visceral mets or high LDH suppress the effect?
it22 = {}
for t, biomark in [('treatment_olaparib','brca2_mutation'),
                   ('treatment_pembrolizumab','msi_high'),
                   ('treatment_lu177_psma','psma_high')]:
    sub_responder = df.loc[df[biomark]==1]  # biomarker-positive subgroup
    res = []
    for modifier in ['ecog_ps','mcrpc','visceral_mets','ar_v7_positive']:
        for level_label, mask in [('low', sub_responder[modifier]==0), ('high', sub_responder[modifier]>0 if modifier!='ecog_ps' else sub_responder[modifier]>=2)]:
            s = sub_responder.loc[mask]
            if len(s) < 30: continue
            g1 = s.loc[s[t]==1, OUTCOME]
            g0 = s.loc[s[t]==0, OUTCOME]
            if len(g1)<5 or len(g0)<5: continue
            diff, p = diff_prop(g1, g0)
            res.append({'modifier': modifier, 'level': level_label, 'diff': float(diff),
                        'p': float(p) if not np.isnan(p) else None,
                        'rr_t': float(g1.mean()), 'rr_not': float(g0.mean()),
                        'n_t': int(len(g1)), 'n_not': int(len(g0))})
    it22[t] = res
results['iter22_suppressors'] = it22

# === ITERATION 23: Final refined treatment-effect heterogeneity model with all biomarker interactions ===
# Olaparib model
m = logit_or(f"{OUTCOME} ~ treatment_olaparib*brca2_mutation + ecog_ps + visceral_mets + albumin_g_dl + ldh_u_l + age_years + mcrpc", df)
it23 = {}
it23['olaparib_adj'] = {
    'inter_coef': float(m.params['treatment_olaparib:brca2_mutation']),
    'inter_p': float(m.pvalues['treatment_olaparib:brca2_mutation']),
    'olap_main_coef': float(m.params['treatment_olaparib']),
    'olap_main_p': float(m.pvalues['treatment_olaparib'])}
m = logit_or(f"{OUTCOME} ~ treatment_pembrolizumab*msi_high + ecog_ps + visceral_mets + albumin_g_dl + ldh_u_l + age_years + mcrpc", df)
it23['pembro_adj'] = {
    'inter_coef': float(m.params['treatment_pembrolizumab:msi_high']),
    'inter_p': float(m.pvalues['treatment_pembrolizumab:msi_high']),
    'pem_main_coef': float(m.params['treatment_pembrolizumab']),
    'pem_main_p': float(m.pvalues['treatment_pembrolizumab'])}
m = logit_or(f"{OUTCOME} ~ treatment_lu177_psma*psma_high + ecog_ps + visceral_mets + albumin_g_dl + ldh_u_l + age_years + mcrpc", df)
it23['lu_adj'] = {
    'inter_coef': float(m.params['treatment_lu177_psma:psma_high']),
    'inter_p': float(m.pvalues['treatment_lu177_psma:psma_high']),
    'lu_main_coef': float(m.params['treatment_lu177_psma']),
    'lu_main_p': float(m.pvalues['treatment_lu177_psma'])}
results['iter23_adj_inter'] = it23

# === ITERATION 24: Best-supported predictive subgroup definitions, using BRCA2/MSI/PSMA + clinical filters ===
# Final hypothesis: olaparib benefits brca2+, pembro benefits msi+, lu177 benefits psma+
# But are there subgroups that don't benefit? E.g., brca2+ but ECOG 2 - no benefit?
it24 = {}
# Olaparib
sub = df.loc[df['brca2_mutation']==1]
for ecog in [0,1,2]:
    s = sub.loc[sub['ecog_ps']==ecog]
    if len(s) < 30: continue
    g1 = s.loc[s['treatment_olaparib']==1, OUTCOME]
    g0 = s.loc[s['treatment_olaparib']==0, OUTCOME]
    if len(g1)<5 or len(g0)<5: continue
    diff, p = diff_prop(g1, g0)
    it24[f'olap_brca2_ecog{ecog}'] = {'diff': float(diff), 'p': float(p),
        'rr_t': float(g1.mean()), 'rr_not': float(g0.mean()), 'n_t': int(len(g1)), 'n_not': int(len(g0))}
# Pembro
sub = df.loc[df['msi_high']==1]
for ecog in [0,1,2]:
    s = sub.loc[sub['ecog_ps']==ecog]
    if len(s) < 30: continue
    g1 = s.loc[s['treatment_pembrolizumab']==1, OUTCOME]
    g0 = s.loc[s['treatment_pembrolizumab']==0, OUTCOME]
    if len(g1)<5 or len(g0)<5: continue
    diff, p = diff_prop(g1, g0)
    it24[f'pem_msi_ecog{ecog}'] = {'diff': float(diff), 'p': float(p),
        'rr_t': float(g1.mean()), 'rr_not': float(g0.mean()), 'n_t': int(len(g1)), 'n_not': int(len(g0))}
# Lu177
sub = df.loc[df['psma_high']==1]
for ecog in [0,1,2]:
    s = sub.loc[sub['ecog_ps']==ecog]
    if len(s) < 30: continue
    g1 = s.loc[s['treatment_lu177_psma']==1, OUTCOME]
    g0 = s.loc[s['treatment_lu177_psma']==0, OUTCOME]
    if len(g1)<5 or len(g0)<5: continue
    diff, p = diff_prop(g1, g0)
    it24[f'lu_psma_ecog{ecog}'] = {'diff': float(diff), 'p': float(p),
        'rr_t': float(g1.mean()), 'rr_not': float(g0.mean()), 'n_t': int(len(g1)), 'n_not': int(len(g0))}
results['iter24_predictive_by_ecog'] = it24

# === ITERATION 25: Final hypothesis statements - confirmed predictive subgroups ===
# Test the strongest sub-hypotheses with confidence intervals
from scipy.stats import norm
def or_ci(a,b,c,d):
    """OR for 2x2: [[a treated/responder, b treated/no resp], [c untreated/responder, d untreated/no resp]]"""
    if min(a,b,c,d) == 0:
        a,b,c,d = a+0.5,b+0.5,c+0.5,d+0.5
    or_ = (a*d)/(b*c)
    se = np.sqrt(1/a+1/b+1/c+1/d)
    return or_, np.exp(np.log(or_)-1.96*se), np.exp(np.log(or_)+1.96*se)

it25 = {}
# Olaparib in BRCA2+
sub = df.loc[df['brca2_mutation']==1]
g1 = sub.loc[sub['treatment_olaparib']==1, OUTCOME]
g0 = sub.loc[sub['treatment_olaparib']==0, OUTCOME]
a,b,c,d = g1.sum(), len(g1)-g1.sum(), g0.sum(), len(g0)-g0.sum()
or_, lo, hi = or_ci(a,b,c,d)
diff, p = diff_prop(g1, g0)
it25['olap_brca2'] = {'OR': float(or_), 'OR_lo': float(lo), 'OR_hi': float(hi),
    'rr_t': float(g1.mean()), 'rr_not': float(g0.mean()), 'diff': float(diff), 'p': float(p),
    'n_t': int(len(g1)), 'n_not': int(len(g0))}
# Pembro in MSI-high
sub = df.loc[df['msi_high']==1]
g1 = sub.loc[sub['treatment_pembrolizumab']==1, OUTCOME]
g0 = sub.loc[sub['treatment_pembrolizumab']==0, OUTCOME]
a,b,c,d = g1.sum(), len(g1)-g1.sum(), g0.sum(), len(g0)-g0.sum()
or_, lo, hi = or_ci(a,b,c,d)
diff, p = diff_prop(g1, g0)
it25['pem_msi'] = {'OR': float(or_), 'OR_lo': float(lo), 'OR_hi': float(hi),
    'rr_t': float(g1.mean()), 'rr_not': float(g0.mean()), 'diff': float(diff), 'p': float(p),
    'n_t': int(len(g1)), 'n_not': int(len(g0))}
# Lu177 in PSMA-high
sub = df.loc[df['psma_high']==1]
g1 = sub.loc[sub['treatment_lu177_psma']==1, OUTCOME]
g0 = sub.loc[sub['treatment_lu177_psma']==0, OUTCOME]
a,b,c,d = g1.sum(), len(g1)-g1.sum(), g0.sum(), len(g0)-g0.sum()
or_, lo, hi = or_ci(a,b,c,d)
diff, p = diff_prop(g1, g0)
it25['lu_psma'] = {'OR': float(or_), 'OR_lo': float(lo), 'OR_hi': float(hi),
    'rr_t': float(g1.mean()), 'rr_not': float(g0.mean()), 'diff': float(diff), 'p': float(p),
    'n_t': int(len(g1)), 'n_not': int(len(g0))}
# Same drugs in biomarker-NEGATIVE (sanity check)
for t, bm, label in [('treatment_olaparib','brca2_mutation','olap_brca2_neg'),
                     ('treatment_pembrolizumab','msi_high','pem_msi_neg'),
                     ('treatment_lu177_psma','psma_high','lu_psma_neg')]:
    sub = df.loc[df[bm]==0]
    g1 = sub.loc[sub[t]==1, OUTCOME]
    g0 = sub.loc[sub[t]==0, OUTCOME]
    a,b,c,d = g1.sum(), len(g1)-g1.sum(), g0.sum(), len(g0)-g0.sum()
    or_, lo, hi = or_ci(a,b,c,d)
    diff, p = diff_prop(g1, g0)
    it25[label] = {'OR': float(or_), 'OR_lo': float(lo), 'OR_hi': float(hi),
        'rr_t': float(g1.mean()), 'rr_not': float(g0.mean()), 'diff': float(diff), 'p': float(p),
        'n_t': int(len(g1)), 'n_not': int(len(g0))}
results['iter25_final'] = it25

with open('all_results.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved all_results.json")
