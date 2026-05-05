"""Iterative analysis of ds001_aml — proposes & tests hypotheses across iterations."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import logit
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
N = len(df)

TX = ['treatment_midostaurin','treatment_gilteritinib','treatment_ivosidenib',
      'treatment_enasidenib','treatment_venetoclax_azacitidine','treatment_7plus3']
MUT = ['flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation','npm1_mutation','tp53_mutation']
LABS = ['wbc_k_per_ul','blast_pct_marrow','albumin_g_dl','ldh_u_l','crp_mg_l','nlr',
        'hemoglobin_g_dl','alkaline_phosphatase_u_l','ast_u_l','alt_u_l',
        'total_bilirubin_mg_dl','creatinine_mg_dl','bun_mg_dl','sodium_meq_l',
        'potassium_meq_l','calcium_mg_dl']
DEMO = ['age_years','sex_female','ecog_ps','secondary_aml','unfit_for_intensive',
        'complex_karyotype','weight_loss_pct_6mo']

iterations = []  # list of dicts {index, proposed_hypotheses, analyses}
LOG = []  # narrative for summary

def fmt(x, d=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "NA"
    return f"{x:.{d}f}"

def diff_in_means_binary_outcome(df, group_col, outcome_col='objective_response'):
    """Returns response-rate difference (group1 - group0), p-value via chi-square."""
    g1 = df.loc[df[group_col]==1, outcome_col]
    g0 = df.loc[df[group_col]==0, outcome_col]
    if len(g1) < 2 or len(g0) < 2:
        return None, None, len(g1), len(g0)
    p1, p0 = g1.mean(), g0.mean()
    tab = pd.crosstab(df[group_col], df[outcome_col])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    return p1-p0, p, len(g1), len(g0)

def logistic_or(df, x_col, y_col='objective_response', adjust=None):
    """Logistic regression OR for x_col on y_col, optionally adjusted."""
    cols = [x_col]
    if adjust:
        cols += [c for c in adjust if c != x_col]
    X = df[cols].astype(float).copy()
    X = sm.add_constant(X)
    y = df[y_col].astype(int).values
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        coef = model.params[x_col]
        p = model.pvalues[x_col]
        return coef, np.exp(coef), p
    except Exception as e:
        return None, None, None

def chi2_or(df, x_col, y_col='objective_response'):
    tab = pd.crosstab(df[x_col], df[y_col])
    if tab.shape != (2,2):
        return None, None
    chi2, p, _, _ = stats.chi2_contingency(tab)
    a,b = tab.iloc[1,1], tab.iloc[1,0]
    c,d = tab.iloc[0,1], tab.iloc[0,0]
    if b==0 or c==0 or d==0 or a==0:
        return None, p
    or_ = (a*d)/(b*c)
    return or_, p

# ============ ITERATION 1: feature main effects ===============
it1_hyps = []
it1_an = []

# H1.1: Older age -> lower response
hid='h1.1'
it1_hyps.append({"id":hid,"text":"Higher age_years is associated with lower probability of objective_response.","kind":"novel"})
coef,or_,p = logistic_or(df,'age_years')
it1_an.append({"hypothesis_ids":[hid],
               "code":"sm.Logit(objective_response ~ age_years)",
               "result_summary":f"Logistic regression coef for age_years = {fmt(coef)} (OR per yr={fmt(or_,3)}), p={fmt(p,2e=False) if False else fmt(p,6)}.",
               "p_value":float(p) if p is not None else None,
               "effect_estimate":float(coef) if coef is not None else None,
               "significant": (p is not None and p<0.05)})

# H1.2: ECOG higher -> lower response
hid='h1.2'
it1_hyps.append({"id":hid,"text":"Higher ecog_ps is associated with lower probability of objective_response.","kind":"novel"})
coef,or_,p = logistic_or(df,'ecog_ps')
it1_an.append({"hypothesis_ids":[hid],
               "code":"sm.Logit(objective_response ~ ecog_ps)",
               "result_summary":f"OR per ECOG unit = {fmt(or_,3)}, coef={fmt(coef)}, p={fmt(p,6)}.",
               "p_value":float(p) if p is not None else None,
               "effect_estimate":float(coef) if coef is not None else None,
               "significant": (p is not None and p<0.05)})

# H1.3: tp53_mutation -> lower
hid='h1.3'
it1_hyps.append({"id":hid,"text":"tp53_mutation is associated with lower probability of objective_response.","kind":"novel"})
diff,pchi,n1,n0 = diff_in_means_binary_outcome(df,'tp53_mutation')
it1_an.append({"hypothesis_ids":[hid],
               "code":"chi2 tp53_mutation x objective_response",
               "result_summary":f"Response in tp53+ {fmt(df[df.tp53_mutation==1].objective_response.mean(),4)} vs tp53- {fmt(df[df.tp53_mutation==0].objective_response.mean(),4)}; diff={fmt(diff,4)}, chi2 p={fmt(pchi,6)}.",
               "p_value":float(pchi) if pchi is not None else None,
               "effect_estimate":float(diff) if diff is not None else None,
               "significant":(pchi is not None and pchi<0.05)})

# H1.4: complex_karyotype -> lower
hid='h1.4'
it1_hyps.append({"id":hid,"text":"complex_karyotype is associated with lower probability of objective_response.","kind":"novel"})
diff,pchi,n1,n0 = diff_in_means_binary_outcome(df,'complex_karyotype')
it1_an.append({"hypothesis_ids":[hid],
               "code":"chi2 complex_karyotype x objective_response",
               "result_summary":f"Response in CK+ {fmt(df[df.complex_karyotype==1].objective_response.mean(),4)} vs CK- {fmt(df[df.complex_karyotype==0].objective_response.mean(),4)}; diff={fmt(diff,4)}, p={fmt(pchi,6)}.",
               "p_value":float(pchi) if pchi is not None else None,
               "effect_estimate":float(diff) if diff is not None else None,
               "significant":(pchi is not None and pchi<0.05)})

# H1.5: npm1_mutation -> higher
hid='h1.5'
it1_hyps.append({"id":hid,"text":"npm1_mutation is associated with higher probability of objective_response.","kind":"novel"})
diff,pchi,_,_ = diff_in_means_binary_outcome(df,'npm1_mutation')
it1_an.append({"hypothesis_ids":[hid],
               "code":"chi2 npm1_mutation x objective_response",
               "result_summary":f"Response in NPM1+ {fmt(df[df.npm1_mutation==1].objective_response.mean(),4)} vs NPM1- {fmt(df[df.npm1_mutation==0].objective_response.mean(),4)}; diff={fmt(diff,4)}, p={fmt(pchi,6)}.",
               "p_value":float(pchi) if pchi is not None else None,
               "effect_estimate":float(diff) if diff is not None else None,
               "significant":(pchi is not None and pchi<0.05)})

# H1.6: secondary_aml -> lower
hid='h1.6'
it1_hyps.append({"id":hid,"text":"secondary_aml is associated with lower probability of objective_response.","kind":"novel"})
diff,pchi,_,_ = diff_in_means_binary_outcome(df,'secondary_aml')
it1_an.append({"hypothesis_ids":[hid],
               "code":"chi2 secondary_aml x objective_response",
               "result_summary":f"Response in 2°AML {fmt(df[df.secondary_aml==1].objective_response.mean(),4)} vs de novo {fmt(df[df.secondary_aml==0].objective_response.mean(),4)}; diff={fmt(diff,4)}, p={fmt(pchi,6)}.",
               "p_value":float(pchi) if pchi is not None else None,
               "effect_estimate":float(diff) if diff is not None else None,
               "significant":(pchi is not None and pchi<0.05)})

# H1.7: unfit_for_intensive -> lower
hid='h1.7'
it1_hyps.append({"id":hid,"text":"unfit_for_intensive flag is associated with lower probability of objective_response.","kind":"novel"})
diff,pchi,_,_ = diff_in_means_binary_outcome(df,'unfit_for_intensive')
it1_an.append({"hypothesis_ids":[hid],
               "code":"chi2 unfit_for_intensive x objective_response",
               "result_summary":f"Response unfit+ {fmt(df[df.unfit_for_intensive==1].objective_response.mean(),4)} vs fit {fmt(df[df.unfit_for_intensive==0].objective_response.mean(),4)}; diff={fmt(diff,4)}, p={fmt(pchi,6)}.",
               "p_value":float(pchi) if pchi is not None else None,
               "effect_estimate":float(diff) if diff is not None else None,
               "significant":(pchi is not None and pchi<0.05)})

iterations.append({"index":1,"proposed_hypotheses":it1_hyps,"analyses":it1_an})

# print quickly
for an in it1_an:
    print(an["result_summary"])

with open("iter1_dump.json","w") as f:
    json.dump(iterations,f,indent=2)
print("Iter 1 done.")
