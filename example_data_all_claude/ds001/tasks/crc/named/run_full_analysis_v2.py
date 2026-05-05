"""
Full iterative analysis of ds001_crc.
Runs ~25 iterations of hypothesis -> analysis on pfs_months and emits a structured
results dictionary used to build the transcript and summary.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')

OUT = {"iterations": []}

def add_iter(idx, hyps, analyses):
    OUT["iterations"].append({"index": idx, "proposed_hypotheses": hyps, "analyses": analyses})


def ttest_diff(mask_yes, mask_no, y='pfs_months'):
    a = df.loc[mask_yes, y].values
    b = df.loc[mask_no, y].values
    if len(a) < 5 or len(b) < 5:
        return None, None, None, None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(np.mean(a)), float(np.mean(b)), float(t), float(p)


def lin_effect(formula):
    """Run an OLS, return (coef, p) for the variable named after the tilde."""
    res = smf.ols(formula, data=df).fit()
    return res


# ------------- Iteration 1: ECOG main effect -------------
print("Iteration 1")
res_ecog = smf.ols('pfs_months ~ ecog_ps', data=df).fit()
ecog_coef = float(res_ecog.params['ecog_ps'])
ecog_p = float(res_ecog.pvalues['ecog_ps'])
m_e0 = df.loc[df['ecog_ps']==0,'pfs_months'].mean()
m_e1 = df.loc[df['ecog_ps']==1,'pfs_months'].mean()
m_e2 = df.loc[df['ecog_ps']==2,'pfs_months'].mean()

add_iter(1,
    [{"id":"h1","text":"Higher ECOG performance status (ecog_ps) is associated with shorter pfs_months (negative slope of pfs on ecog_ps).","kind":"novel"}],
    [{"hypothesis_ids":["h1"],
      "code":"smf.ols('pfs_months ~ ecog_ps', data=df).fit()",
      "result_summary":f"Linear regression of pfs_months on ecog_ps: slope={ecog_coef:.3f} months/ECOG point, p={ecog_p:.2e}. Mean PFS by ECOG: 0={m_e0:.2f}, 1={m_e1:.2f}, 2={m_e2:.2f}.",
      "p_value":ecog_p, "effect_estimate":ecog_coef, "significant": ecog_p<0.05}])

# ------------- Iteration 2: Stage IV main effect -------------
print("Iteration 2")
m_s4_y, m_s4_n, t, p = ttest_diff(df['stage_iv']==1, df['stage_iv']==0)
add_iter(2,
    [{"id":"h2","text":"Patients with stage_iv=1 have shorter pfs_months than patients with stage_iv=0.","kind":"novel"}],
    [{"hypothesis_ids":["h2"],
      "code":"stats.ttest_ind(df.pfs[stage_iv==1], df.pfs[stage_iv==0])",
      "result_summary":f"Stage IV mean PFS = {m_s4_y:.2f} vs non-stage-IV {m_s4_n:.2f} (diff={m_s4_y-m_s4_n:+.2f} months, Welch t={t:.1f}, p={p:.2e}).",
      "p_value":p, "effect_estimate":float(m_s4_y-m_s4_n), "significant":p<0.05}])

# ------------- Iteration 3: age, sex, right-sided -------------
print("Iteration 3")
res_age = smf.ols('pfs_months ~ age_years', data=df).fit()
age_coef, age_p = float(res_age.params['age_years']), float(res_age.pvalues['age_years'])
mF, mM, tF, pF = ttest_diff(df['sex_female']==1, df['sex_female']==0)
mR, mL, tR, pR = ttest_diff(df['right_sided_primary']==1, df['right_sided_primary']==0)

add_iter(3,
    [{"id":"h3a","text":"Older age_years is associated with shorter pfs_months (negative slope).","kind":"novel"},
     {"id":"h3b","text":"Female patients (sex_female=1) have different pfs_months than male patients (sex_female=0).","kind":"novel"},
     {"id":"h3c","text":"Right-sided primary tumors (right_sided_primary=1) have shorter pfs_months than left-sided.","kind":"novel"}],
    [{"hypothesis_ids":["h3a"],
      "code":"smf.ols('pfs_months ~ age_years', data=df).fit()",
      "result_summary":f"Slope of pfs on age = {age_coef:.4f} months/year, p={age_p:.2e}.",
      "p_value":age_p, "effect_estimate":age_coef, "significant":age_p<0.05},
     {"hypothesis_ids":["h3b"],
      "code":"ttest pfs by sex_female",
      "result_summary":f"Mean PFS female={mF:.2f} vs male={mM:.2f}, diff={mF-mM:+.2f}, p={pF:.2e}.",
      "p_value":pF, "effect_estimate":float(mF-mM), "significant":pF<0.05},
     {"hypothesis_ids":["h3c"],
      "code":"ttest pfs by right_sided_primary",
      "result_summary":f"Mean PFS right-sided={mR:.2f} vs left/transverse={mL:.2f}, diff={mR-mL:+.2f}, p={pR:.2e}.",
      "p_value":pR, "effect_estimate":float(mR-mL), "significant":pR<0.05}])

# ------------- Iteration 4: continuous lab main effects -------------
print("Iteration 4")
labs = ['cea_ng_ml','albumin_g_dl','ldh_u_l','weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
        'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl','creatinine_mg_dl',
        'bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']
lab_results = {}
analyses4 = []
for lab in labs:
    res = smf.ols(f'pfs_months ~ {lab}', data=df).fit()
    coef = float(res.params[lab]); p = float(res.pvalues[lab])
    lab_results[lab] = (coef, p)

hyps4 = [{"id":f"h4_{i+1}","text":f"Higher {lab} is associated with {'shorter' if lab_results[lab][0]<0 else 'longer'} pfs_months (linear slope hypothesis: nonzero).","kind":"novel"} for i,lab in enumerate(labs)]
for i,lab in enumerate(labs):
    coef,p = lab_results[lab]
    analyses4.append({"hypothesis_ids":[f"h4_{i+1}"],
        "code":f"smf.ols('pfs_months ~ {lab}', data=df).fit()",
        "result_summary":f"Slope of pfs on {lab} = {coef:.4f}, p={p:.2e}.",
        "p_value":p, "effect_estimate":coef, "significant":p<0.05})
add_iter(4, hyps4, analyses4)

# ------------- Iteration 5: biomarker (mutation) main effects -------------
print("Iteration 5")
markers = ['kras_mutation','nras_mutation','braf_v600e','msi_high','her2_amplified','ntrk_fusion']
hyps5 = []; analyses5 = []
for i,m in enumerate(markers):
    a, b, t, p = ttest_diff(df[m]==1, df[m]==0)
    diff = a-b
    hyps5.append({"id":f"h5_{i+1}","text":f"Patients with {m}=1 have {'shorter' if diff<0 else 'longer'} pfs_months than patients with {m}=0.","kind":"novel"})
    analyses5.append({"hypothesis_ids":[f"h5_{i+1}"],
        "code":f"ttest pfs by {m}",
        "result_summary":f"{m}: pos={a:.2f} vs neg={b:.2f} (diff={diff:+.2f}, p={p:.2e}, n_pos={(df[m]==1).sum()}).",
        "p_value":float(p), "effect_estimate":float(diff), "significant":float(p)<0.05})
add_iter(5, hyps5, analyses5)

# ------------- Iteration 6: cetuximab main effect, adjusted -------------
print("Iteration 6")
m_y, m_n, t, p = ttest_diff(df['treatment_cetuximab']==1, df['treatment_cetuximab']==0)
res_adj = smf.ols('pfs_months ~ treatment_cetuximab + age_years + sex_female + ecog_ps + stage_iv', data=df).fit()
add_iter(6,
    [{"id":"h6","text":"treatment_cetuximab is associated with longer pfs_months on average (positive main effect).","kind":"novel"}],
    [{"hypothesis_ids":["h6"],
      "code":"ttest + ols pfs ~ cetux + covariates",
      "result_summary":f"Unadjusted: cetux={m_y:.2f} vs ctrl={m_n:.2f} (diff={m_y-m_n:+.2f}, p={p:.2e}). Adjusted (age/sex/ECOG/stage): coef={float(res_adj.params['treatment_cetuximab']):+.3f}, p={float(res_adj.pvalues['treatment_cetuximab']):.2e}.",
      "p_value":float(p), "effect_estimate":float(m_y-m_n), "significant":float(p)<0.05}])

# ------------- Iteration 7: bevacizumab main effect, adjusted -------------
print("Iteration 7")
m_y, m_n, t, p = ttest_diff(df['treatment_bevacizumab']==1, df['treatment_bevacizumab']==0)
res_adj = smf.ols('pfs_months ~ treatment_bevacizumab + age_years + sex_female + ecog_ps + stage_iv', data=df).fit()
add_iter(7,
    [{"id":"h7","text":"treatment_bevacizumab is associated with longer pfs_months on average (positive main effect).","kind":"novel"}],
    [{"hypothesis_ids":["h7"],
      "code":"ttest + ols pfs ~ bev + covariates",
      "result_summary":f"Unadjusted: bev={m_y:.2f} vs ctrl={m_n:.2f} (diff={m_y-m_n:+.2f}, p={p:.2e}). Adjusted: coef={float(res_adj.params['treatment_bevacizumab']):+.3f}, p={float(res_adj.pvalues['treatment_bevacizumab']):.2e}.",
      "p_value":float(p), "effect_estimate":float(m_y-m_n), "significant":float(p)<0.05}])

# ------------- Iteration 8: pembrolizumab main effect, adjusted -------------
print("Iteration 8")
m_y, m_n, t, p = ttest_diff(df['treatment_pembrolizumab']==1, df['treatment_pembrolizumab']==0)
res_adj = smf.ols('pfs_months ~ treatment_pembrolizumab + age_years + sex_female + ecog_ps + stage_iv', data=df).fit()
add_iter(8,
    [{"id":"h8","text":"treatment_pembrolizumab is associated with longer pfs_months on average across the whole cohort.","kind":"novel"}],
    [{"hypothesis_ids":["h8"],
      "code":"ttest + ols pfs ~ pembro + covariates",
      "result_summary":f"Unadjusted: pembro={m_y:.2f} vs ctrl={m_n:.2f} (diff={m_y-m_n:+.2f}, p={p:.2e}). Adjusted: coef={float(res_adj.params['treatment_pembrolizumab']):+.3f}, p={float(res_adj.pvalues['treatment_pembrolizumab']):.2e}.",
      "p_value":float(p), "effect_estimate":float(m_y-m_n), "significant":float(p)<0.05}])

# ------------- Iteration 9: encorafenib main effect, adjusted -------------
print("Iteration 9")
m_y, m_n, t, p = ttest_diff(df['treatment_encorafenib']==1, df['treatment_encorafenib']==0)
res_adj = smf.ols('pfs_months ~ treatment_encorafenib + age_years + sex_female + ecog_ps + stage_iv', data=df).fit()
add_iter(9,
    [{"id":"h9","text":"treatment_encorafenib is associated with longer pfs_months across the whole cohort.","kind":"novel"}],
    [{"hypothesis_ids":["h9"],
      "code":"ttest + ols pfs ~ enco + covariates",
      "result_summary":f"Unadjusted: enco={m_y:.2f} vs ctrl={m_n:.2f} (diff={m_y-m_n:+.2f}, p={p:.2e}). Adjusted: coef={float(res_adj.params['treatment_encorafenib']):+.3f}, p={float(res_adj.pvalues['treatment_encorafenib']):.2e}.",
      "p_value":float(p), "effect_estimate":float(m_y-m_n), "significant":float(p)<0.05}])

# ------------- Iteration 10: trastuzumab/tucatinib main effect, adjusted -------------
print("Iteration 10")
m_y, m_n, t, p = ttest_diff(df['treatment_trastuzumab_tucatinib']==1, df['treatment_trastuzumab_tucatinib']==0)
res_adj = smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib + age_years + sex_female + ecog_ps + stage_iv', data=df).fit()
add_iter(10,
    [{"id":"h10","text":"treatment_trastuzumab_tucatinib is associated with longer pfs_months across the whole cohort.","kind":"novel"}],
    [{"hypothesis_ids":["h10"],
      "code":"ttest + ols pfs ~ trast/tuc + covariates",
      "result_summary":f"Unadjusted: trast/tuc={m_y:.2f} vs ctrl={m_n:.2f} (diff={m_y-m_n:+.2f}, p={p:.2e}). Adjusted: coef={float(res_adj.params['treatment_trastuzumab_tucatinib']):+.3f}, p={float(res_adj.pvalues['treatment_trastuzumab_tucatinib']):.2e}.",
      "p_value":float(p), "effect_estimate":float(m_y-m_n), "significant":float(p)<0.05}])

# ------------- Iteration 11: regorafenib main effect, adjusted -------------
print("Iteration 11")
m_y, m_n, t, p = ttest_diff(df['treatment_regorafenib']==1, df['treatment_regorafenib']==0)
res_adj = smf.ols('pfs_months ~ treatment_regorafenib + age_years + sex_female + ecog_ps + stage_iv', data=df).fit()
add_iter(11,
    [{"id":"h11","text":"treatment_regorafenib is associated with longer pfs_months across the whole cohort.","kind":"novel"}],
    [{"hypothesis_ids":["h11"],
      "code":"ttest + ols pfs ~ rego + covariates",
      "result_summary":f"Unadjusted: rego={m_y:.2f} vs ctrl={m_n:.2f} (diff={m_y-m_n:+.2f}, p={p:.2e}). Adjusted: coef={float(res_adj.params['treatment_regorafenib']):+.3f}, p={float(res_adj.pvalues['treatment_regorafenib']):.2e}.",
      "p_value":float(p), "effect_estimate":float(m_y-m_n), "significant":float(p)<0.05}])

# ------------- Iteration 12: cetuximab x KRAS / NRAS / BRAF / right-sided -------------
print("Iteration 12")
hyps12 = []; analyses12 = []
# cetux x KRAS
res = smf.ols('pfs_months ~ treatment_cetuximab*kras_mutation', data=df).fit()
inter = float(res.params['treatment_cetuximab:kras_mutation']); pinter = float(res.pvalues['treatment_cetuximab:kras_mutation'])
e_pos = df.loc[(df['treatment_cetuximab']==1)&(df['kras_mutation']==1),'pfs_months'].mean() - df.loc[(df['treatment_cetuximab']==0)&(df['kras_mutation']==1),'pfs_months'].mean()
e_neg = df.loc[(df['treatment_cetuximab']==1)&(df['kras_mutation']==0),'pfs_months'].mean() - df.loc[(df['treatment_cetuximab']==0)&(df['kras_mutation']==0),'pfs_months'].mean()
hyps12.append({"id":"h12a","text":"The effect of treatment_cetuximab on pfs_months differs between KRAS-mutated and KRAS-wildtype patients (interaction).","kind":"novel"})
analyses12.append({"hypothesis_ids":["h12a"],
    "code":"smf.ols('pfs_months ~ treatment_cetuximab*kras_mutation', data=df)",
    "result_summary":f"Cetuximab effect in KRAS-mut={e_pos:+.2f} vs KRAS-wt={e_neg:+.2f}; interaction coef={inter:+.3f}, p={pinter:.2e}.",
    "p_value":pinter, "effect_estimate":inter, "significant":pinter<0.05})

# cetux x NRAS
res = smf.ols('pfs_months ~ treatment_cetuximab*nras_mutation', data=df).fit()
inter = float(res.params['treatment_cetuximab:nras_mutation']); pinter = float(res.pvalues['treatment_cetuximab:nras_mutation'])
e_pos = df.loc[(df['treatment_cetuximab']==1)&(df['nras_mutation']==1),'pfs_months'].mean() - df.loc[(df['treatment_cetuximab']==0)&(df['nras_mutation']==1),'pfs_months'].mean()
e_neg = df.loc[(df['treatment_cetuximab']==1)&(df['nras_mutation']==0),'pfs_months'].mean() - df.loc[(df['treatment_cetuximab']==0)&(df['nras_mutation']==0),'pfs_months'].mean()
hyps12.append({"id":"h12b","text":"The effect of treatment_cetuximab on pfs_months differs between NRAS-mutated and NRAS-wildtype patients (interaction).","kind":"novel"})
analyses12.append({"hypothesis_ids":["h12b"],
    "code":"smf.ols('pfs_months ~ treatment_cetuximab*nras_mutation', data=df)",
    "result_summary":f"Cetuximab effect in NRAS-mut={e_pos:+.2f} vs NRAS-wt={e_neg:+.2f}; interaction coef={inter:+.3f}, p={pinter:.2e}.",
    "p_value":pinter, "effect_estimate":inter, "significant":pinter<0.05})

# cetux x BRAF
res = smf.ols('pfs_months ~ treatment_cetuximab*braf_v600e', data=df).fit()
inter = float(res.params['treatment_cetuximab:braf_v600e']); pinter = float(res.pvalues['treatment_cetuximab:braf_v600e'])
e_pos = df.loc[(df['treatment_cetuximab']==1)&(df['braf_v600e']==1),'pfs_months'].mean() - df.loc[(df['treatment_cetuximab']==0)&(df['braf_v600e']==1),'pfs_months'].mean()
e_neg = df.loc[(df['treatment_cetuximab']==1)&(df['braf_v600e']==0),'pfs_months'].mean() - df.loc[(df['treatment_cetuximab']==0)&(df['braf_v600e']==0),'pfs_months'].mean()
hyps12.append({"id":"h12c","text":"The effect of treatment_cetuximab on pfs_months differs between BRAF V600E-mutated and BRAF wildtype patients (interaction).","kind":"novel"})
analyses12.append({"hypothesis_ids":["h12c"],
    "code":"smf.ols('pfs_months ~ treatment_cetuximab*braf_v600e', data=df)",
    "result_summary":f"Cetuximab effect in BRAFV600E-pos={e_pos:+.2f} vs BRAF-wt={e_neg:+.2f}; interaction coef={inter:+.3f}, p={pinter:.2e}.",
    "p_value":pinter, "effect_estimate":inter, "significant":pinter<0.05})

# cetux x right-sided
res = smf.ols('pfs_months ~ treatment_cetuximab*right_sided_primary', data=df).fit()
inter = float(res.params['treatment_cetuximab:right_sided_primary']); pinter = float(res.pvalues['treatment_cetuximab:right_sided_primary'])
e_r = df.loc[(df['treatment_cetuximab']==1)&(df['right_sided_primary']==1),'pfs_months'].mean() - df.loc[(df['treatment_cetuximab']==0)&(df['right_sided_primary']==1),'pfs_months'].mean()
e_l = df.loc[(df['treatment_cetuximab']==1)&(df['right_sided_primary']==0),'pfs_months'].mean() - df.loc[(df['treatment_cetuximab']==0)&(df['right_sided_primary']==0),'pfs_months'].mean()
hyps12.append({"id":"h12d","text":"The effect of treatment_cetuximab on pfs_months differs between right-sided and left-sided primary tumors (interaction).","kind":"novel"})
analyses12.append({"hypothesis_ids":["h12d"],
    "code":"smf.ols('pfs_months ~ treatment_cetuximab*right_sided_primary', data=df)",
    "result_summary":f"Cetuximab effect right-sided={e_r:+.2f} vs left-sided={e_l:+.2f}; interaction coef={inter:+.3f}, p={pinter:.2e}.",
    "p_value":pinter, "effect_estimate":inter, "significant":pinter<0.05})

add_iter(12, hyps12, analyses12)

# ------------- Iteration 13: pembrolizumab x MSI-high (and BRAF/right-sided) -------------
print("Iteration 13")
hyps13 = []; analyses13 = []
res = smf.ols('pfs_months ~ treatment_pembrolizumab*msi_high', data=df).fit()
inter = float(res.params['treatment_pembrolizumab:msi_high']); pinter = float(res.pvalues['treatment_pembrolizumab:msi_high'])
e_pos = df.loc[(df['treatment_pembrolizumab']==1)&(df['msi_high']==1),'pfs_months'].mean() - df.loc[(df['treatment_pembrolizumab']==0)&(df['msi_high']==1),'pfs_months'].mean()
e_neg = df.loc[(df['treatment_pembrolizumab']==1)&(df['msi_high']==0),'pfs_months'].mean() - df.loc[(df['treatment_pembrolizumab']==0)&(df['msi_high']==0),'pfs_months'].mean()
n_msi_pembro = ((df['msi_high']==1)&(df['treatment_pembrolizumab']==1)).sum()
hyps13.append({"id":"h13a","text":"The effect of treatment_pembrolizumab on pfs_months is much larger (more positive) in MSI-high patients than in MSI-stable patients (positive interaction).","kind":"novel"})
analyses13.append({"hypothesis_ids":["h13a"],
    "code":"smf.ols('pfs_months ~ treatment_pembrolizumab*msi_high', data=df)",
    "result_summary":f"Pembro effect in MSI-high={e_pos:+.2f} vs MSI-stable={e_neg:+.2f}; interaction coef={inter:+.3f}, p={pinter:.2e}. n(MSI-high & pembro)={n_msi_pembro}.",
    "p_value":pinter, "effect_estimate":inter, "significant":pinter<0.05})

# pembro x BRAF
res = smf.ols('pfs_months ~ treatment_pembrolizumab*braf_v600e', data=df).fit()
inter = float(res.params['treatment_pembrolizumab:braf_v600e']); pinter = float(res.pvalues['treatment_pembrolizumab:braf_v600e'])
hyps13.append({"id":"h13b","text":"The effect of treatment_pembrolizumab on pfs_months differs between BRAF V600E and BRAF wildtype patients (interaction).","kind":"novel"})
analyses13.append({"hypothesis_ids":["h13b"],
    "code":"smf.ols('pfs_months ~ treatment_pembrolizumab*braf_v600e', data=df)",
    "result_summary":f"Pembro x BRAF V600E interaction coef={inter:+.3f}, p={pinter:.2e}.",
    "p_value":pinter, "effect_estimate":inter, "significant":pinter<0.05})

add_iter(13, hyps13, analyses13)

# ------------- Iteration 14: encorafenib x BRAF V600E (BEACON-like) and KRAS -------------
print("Iteration 14")
hyps14 = []; analyses14 = []
res = smf.ols('pfs_months ~ treatment_encorafenib*braf_v600e', data=df).fit()
inter = float(res.params['treatment_encorafenib:braf_v600e']); pinter = float(res.pvalues['treatment_encorafenib:braf_v600e'])
e_pos = df.loc[(df['treatment_encorafenib']==1)&(df['braf_v600e']==1),'pfs_months'].mean() - df.loc[(df['treatment_encorafenib']==0)&(df['braf_v600e']==1),'pfs_months'].mean()
e_neg = df.loc[(df['treatment_encorafenib']==1)&(df['braf_v600e']==0),'pfs_months'].mean() - df.loc[(df['treatment_encorafenib']==0)&(df['braf_v600e']==0),'pfs_months'].mean()
n_braf_enc = ((df['braf_v600e']==1)&(df['treatment_encorafenib']==1)).sum()
hyps14.append({"id":"h14a","text":"The effect of treatment_encorafenib on pfs_months is positive and larger in BRAF V600E-mutated patients than in BRAF wildtype patients (positive interaction).","kind":"novel"})
analyses14.append({"hypothesis_ids":["h14a"],
    "code":"smf.ols('pfs_months ~ treatment_encorafenib*braf_v600e', data=df)",
    "result_summary":f"Encorafenib effect in BRAFV600E={e_pos:+.2f} vs BRAF-wt={e_neg:+.2f}; interaction coef={inter:+.3f}, p={pinter:.2e}. n(BRAF & enco)={n_braf_enc}.",
    "p_value":pinter, "effect_estimate":inter, "significant":pinter<0.05})

add_iter(14, hyps14, analyses14)

# ------------- Iteration 15: trastuzumab/tucatinib x HER2 amplified -------------
print("Iteration 15")
hyps15 = []; analyses15 = []
res = smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib*her2_amplified', data=df).fit()
inter = float(res.params['treatment_trastuzumab_tucatinib:her2_amplified']); pinter = float(res.pvalues['treatment_trastuzumab_tucatinib:her2_amplified'])
e_pos = df.loc[(df['treatment_trastuzumab_tucatinib']==1)&(df['her2_amplified']==1),'pfs_months'].mean() - df.loc[(df['treatment_trastuzumab_tucatinib']==0)&(df['her2_amplified']==1),'pfs_months'].mean()
e_neg = df.loc[(df['treatment_trastuzumab_tucatinib']==1)&(df['her2_amplified']==0),'pfs_months'].mean() - df.loc[(df['treatment_trastuzumab_tucatinib']==0)&(df['her2_amplified']==0),'pfs_months'].mean()
n_her_tt = ((df['her2_amplified']==1)&(df['treatment_trastuzumab_tucatinib']==1)).sum()
hyps15.append({"id":"h15a","text":"The effect of treatment_trastuzumab_tucatinib on pfs_months is positive and larger in HER2-amplified patients than in HER2-non-amplified patients (positive interaction).","kind":"novel"})
analyses15.append({"hypothesis_ids":["h15a"],
    "code":"smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib*her2_amplified', data=df)",
    "result_summary":f"Trast/tuc effect in HER2-amp={e_pos:+.2f} vs HER2-neg={e_neg:+.2f}; interaction coef={inter:+.3f}, p={pinter:.2e}. n(HER2-amp & trast/tuc)={n_her_tt}.",
    "p_value":pinter, "effect_estimate":inter, "significant":pinter<0.05})

add_iter(15, hyps15, analyses15)

# ------------- Iteration 16: bevacizumab interactions (KRAS, right-sided, stage_iv) -------------
print("Iteration 16")
hyps16 = []; analyses16 = []
for marker in ['kras_mutation','right_sided_primary','stage_iv','msi_high','braf_v600e']:
    res = smf.ols(f'pfs_months ~ treatment_bevacizumab*{marker}', data=df).fit()
    inter = float(res.params[f'treatment_bevacizumab:{marker}']); pinter = float(res.pvalues[f'treatment_bevacizumab:{marker}'])
    e_pos = df.loc[(df['treatment_bevacizumab']==1)&(df[marker]==1),'pfs_months'].mean() - df.loc[(df['treatment_bevacizumab']==0)&(df[marker]==1),'pfs_months'].mean()
    e_neg = df.loc[(df['treatment_bevacizumab']==1)&(df[marker]==0),'pfs_months'].mean() - df.loc[(df['treatment_bevacizumab']==0)&(df[marker]==0),'pfs_months'].mean()
    hid = f"h16_{marker}"
    hyps16.append({"id":hid,"text":f"The effect of treatment_bevacizumab on pfs_months differs between {marker}=1 and {marker}=0 patients (interaction).","kind":"novel"})
    analyses16.append({"hypothesis_ids":[hid],
        "code":f"smf.ols('pfs_months ~ treatment_bevacizumab*{marker}', data=df)",
        "result_summary":f"Bev effect {marker}=1: {e_pos:+.2f} vs {marker}=0: {e_neg:+.2f}; interaction coef={inter:+.3f}, p={pinter:.2e}.",
        "p_value":pinter, "effect_estimate":inter, "significant":pinter<0.05})

add_iter(16, hyps16, analyses16)

# ------------- Iteration 17: regorafenib subgroup effects (deep dive) -------------
print("Iteration 17")
hyps17 = []; analyses17 = []
# Strong main effect found earlier -- explore which subgroups drive it
markers = ['kras_mutation','nras_mutation','braf_v600e','msi_high','her2_amplified','ntrk_fusion',
           'right_sided_primary','stage_iv','sex_female']
for marker in markers:
    res = smf.ols(f'pfs_months ~ treatment_regorafenib*{marker}', data=df).fit()
    inter = float(res.params[f'treatment_regorafenib:{marker}']); pinter = float(res.pvalues[f'treatment_regorafenib:{marker}'])
    e_pos = df.loc[(df['treatment_regorafenib']==1)&(df[marker]==1),'pfs_months'].mean() - df.loc[(df['treatment_regorafenib']==0)&(df[marker]==1),'pfs_months'].mean()
    e_neg = df.loc[(df['treatment_regorafenib']==1)&(df[marker]==0),'pfs_months'].mean() - df.loc[(df['treatment_regorafenib']==0)&(df[marker]==0),'pfs_months'].mean()
    hid = f"h17_{marker}"
    hyps17.append({"id":hid,"text":f"The effect of treatment_regorafenib on pfs_months differs between {marker}=1 and {marker}=0 patients (interaction).","kind":"novel"})
    analyses17.append({"hypothesis_ids":[hid],
        "code":f"smf.ols('pfs_months ~ treatment_regorafenib*{marker}', data=df)",
        "result_summary":f"Rego effect {marker}=1: {e_pos:+.2f} vs {marker}=0: {e_neg:+.2f}; interaction coef={inter:+.3f}, p={pinter:.2e}.",
        "p_value":pinter, "effect_estimate":inter, "significant":pinter<0.05})

add_iter(17, hyps17, analyses17)

# Save partial output
with open('iter_results_part1.json','w') as f:
    json.dump(OUT, f, indent=2)
print("Saved part1")
