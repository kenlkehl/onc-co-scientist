"""Iterative hypothesis-driven analysis of ds001_nsclc."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
N = len(df)
y = df['objective_response'].astype(int)

iterations = []


def add_iter(idx, hypotheses, analyses):
    iterations.append({
        'index': idx,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses,
    })


def diff_in_props(col_or_mask, sub_df=None):
    """Return (rr_yes, rr_no, diff, p, n_yes, n_no) for binary predictor."""
    d = sub_df if sub_df is not None else df
    if isinstance(col_or_mask, str):
        m = d[col_or_mask] == 1
    else:
        m = col_or_mask
    yes = d.loc[m, 'objective_response']
    no = d.loc[~m, 'objective_response']
    if len(yes) == 0 or len(no) == 0:
        return None
    table = np.array([[yes.sum(), len(yes) - yes.sum()],
                      [no.sum(), len(no) - no.sum()]])
    chi2, p, _, _ = stats.chi2_contingency(table)
    return float(yes.mean()), float(no.mean()), float(yes.mean() - no.mean()), float(p), int(len(yes)), int(len(no))


def logistic(formula_cols, target='objective_response', d=None):
    d = d if d is not None else df
    X = d[formula_cols].copy()
    X = sm.add_constant(X)
    model = sm.Logit(d[target], X).fit(disp=0)
    return model


# ------------------------------ ITERATION 1 ------------------------------
hyps = [
    {'id': 'h1.1', 'text': 'Patients receiving treatment_pembrolizumab have a higher objective_response rate than those not receiving it.', 'kind': 'novel'},
    {'id': 'h1.2', 'text': 'Patients receiving treatment_sotorasib have a higher objective_response rate than those not receiving it.', 'kind': 'novel'},
    {'id': 'h1.3', 'text': 'Patients receiving treatment_olaparib have a higher objective_response rate than those not receiving it.', 'kind': 'novel'},
    {'id': 'h1.4', 'text': 'Patients receiving treatment_osimertinib have a higher objective_response rate than those not receiving it.', 'kind': 'novel'},
]
analyses = []
for tx, hid in [('treatment_pembrolizumab', 'h1.1'),
                ('treatment_sotorasib', 'h1.2'),
                ('treatment_olaparib', 'h1.3'),
                ('treatment_osimertinib', 'h1.4')]:
    r1, r0, diff, p, n1, n0 = diff_in_props(tx)
    analyses.append({
        'hypothesis_ids': [hid],
        'code': f"chi2 of {tx} vs objective_response",
        'result_summary': f"Response rate {r1:.3f} (n={n1}) on {tx} vs {r0:.3f} (n={n0}) off; diff={diff:+.3f}, chi2 p={p:.3g}.",
        'p_value': p,
        'effect_estimate': diff,
        'significant': bool(p < 0.05),
    })
add_iter(1, hyps, analyses)


# ------------------------------ ITERATION 2 ------------------------------
hyps = [
    {'id': 'h2.1', 'text': 'Among patients with egfr_mutation==1, those receiving treatment_osimertinib have a higher objective_response rate than EGFR-mutant patients not receiving treatment_osimertinib.', 'kind': 'novel'},
    {'id': 'h2.2', 'text': 'Among patients with egfr_mutation==0, treatment_osimertinib does not increase objective_response rate.', 'kind': 'novel'},
    {'id': 'h2.3', 'text': 'There is a positive interaction between egfr_mutation and treatment_osimertinib on objective_response (the benefit of osimertinib is larger in EGFR-mutant patients).', 'kind': 'novel'},
]
analyses = []
sub = df[df['egfr_mutation'] == 1]
r1, r0, diff, p, n1, n0 = diff_in_props('treatment_osimertinib', sub_df=sub)
analyses.append({'hypothesis_ids': ['h2.1'],
                 'code': 'subset egfr_mutation==1; chi2 osimertinib vs response',
                 'result_summary': f"In EGFR+ subset (N={len(sub)}): response {r1:.3f} on osimertinib (n={n1}) vs {r0:.3f} off (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
sub = df[df['egfr_mutation'] == 0]
r1, r0, diff, p, n1, n0 = diff_in_props('treatment_osimertinib', sub_df=sub)
analyses.append({'hypothesis_ids': ['h2.2'],
                 'code': 'subset egfr_mutation==0; chi2 osimertinib vs response',
                 'result_summary': f"In EGFR- subset (N={len(sub)}): response {r1:.3f} on osimertinib (n={n1}) vs {r0:.3f} off (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
df['_egfr_x_osi'] = df['egfr_mutation'] * df['treatment_osimertinib']
m = logistic(['egfr_mutation', 'treatment_osimertinib', '_egfr_x_osi'])
b = float(m.params['_egfr_x_osi']); p = float(m.pvalues['_egfr_x_osi'])
analyses.append({'hypothesis_ids': ['h2.3'],
                 'code': 'logit(response ~ egfr + osimertinib + egfr*osimertinib)',
                 'result_summary': f"Interaction log-OR (egfr_mutation x treatment_osimertinib) = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(2, hyps, analyses)


# ------------------------------ ITERATION 3 ------------------------------
hyps = [
    {'id': 'h3.1', 'text': 'Among patients with kras_g12c==1, those receiving treatment_sotorasib have a higher objective_response rate than KRAS G12C+ patients not receiving treatment_sotorasib.', 'kind': 'novel'},
    {'id': 'h3.2', 'text': 'Among patients with kras_g12c==0, treatment_sotorasib does not increase objective_response rate.', 'kind': 'novel'},
    {'id': 'h3.3', 'text': 'There is a positive interaction between kras_g12c and treatment_sotorasib on objective_response.', 'kind': 'novel'},
]
analyses = []
sub = df[df['kras_g12c'] == 1]
r1, r0, diff, p, n1, n0 = diff_in_props('treatment_sotorasib', sub_df=sub)
analyses.append({'hypothesis_ids': ['h3.1'],
                 'code': 'subset kras_g12c==1; chi2 sotorasib vs response',
                 'result_summary': f"In KRAS G12C+ subset (N={len(sub)}): response {r1:.3f} on sotorasib (n={n1}) vs {r0:.3f} off (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
sub = df[df['kras_g12c'] == 0]
r1, r0, diff, p, n1, n0 = diff_in_props('treatment_sotorasib', sub_df=sub)
analyses.append({'hypothesis_ids': ['h3.2'],
                 'code': 'subset kras_g12c==0; chi2 sotorasib vs response',
                 'result_summary': f"In KRAS G12C- subset (N={len(sub)}): response {r1:.3f} on sotorasib (n={n1}) vs {r0:.3f} off (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
df['_kras_x_sot'] = df['kras_g12c'] * df['treatment_sotorasib']
m = logistic(['kras_g12c', 'treatment_sotorasib', '_kras_x_sot'])
b = float(m.params['_kras_x_sot']); p = float(m.pvalues['_kras_x_sot'])
analyses.append({'hypothesis_ids': ['h3.3'],
                 'code': 'logit(response ~ kras + sotorasib + kras*sotorasib)',
                 'result_summary': f"Interaction log-OR (kras_g12c x treatment_sotorasib) = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(3, hyps, analyses)


# ------------------------------ ITERATION 4 ------------------------------
hyps = [
    {'id': 'h4.1', 'text': 'Higher pdl1_tps is associated with a higher objective_response rate among patients on treatment_pembrolizumab.', 'kind': 'novel'},
    {'id': 'h4.2', 'text': 'Higher pdl1_tps is NOT associated with response among patients off treatment_pembrolizumab.', 'kind': 'novel'},
    {'id': 'h4.3', 'text': 'There is a positive interaction between pdl1_tps and treatment_pembrolizumab on objective_response.', 'kind': 'novel'},
]
analyses = []
sub = df[df['treatment_pembrolizumab'] == 1]
m = logistic(['pdl1_tps'], d=sub)
b = float(m.params['pdl1_tps']); p = float(m.pvalues['pdl1_tps'])
analyses.append({'hypothesis_ids': ['h4.1'],
                 'code': 'subset pembro==1; logit(response ~ pdl1_tps)',
                 'result_summary': f"Among pembrolizumab patients (N={len(sub)}): log-OR per unit pdl1_tps = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
sub = df[df['treatment_pembrolizumab'] == 0]
m = logistic(['pdl1_tps'], d=sub)
b = float(m.params['pdl1_tps']); p = float(m.pvalues['pdl1_tps'])
analyses.append({'hypothesis_ids': ['h4.2'],
                 'code': 'subset pembro==0; logit(response ~ pdl1_tps)',
                 'result_summary': f"Among pembrolizumab-untreated (N={len(sub)}): log-OR per unit pdl1_tps = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
df['_pdl1_x_pembro'] = df['pdl1_tps'] * df['treatment_pembrolizumab']
m = logistic(['pdl1_tps', 'treatment_pembrolizumab', '_pdl1_x_pembro'])
b = float(m.params['_pdl1_x_pembro']); p = float(m.pvalues['_pdl1_x_pembro'])
analyses.append({'hypothesis_ids': ['h4.3'],
                 'code': 'logit(response ~ pdl1 + pembro + pdl1*pembro)',
                 'result_summary': f"Interaction log-OR (pdl1_tps x treatment_pembrolizumab) = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(4, hyps, analyses)


# ------------------------------ ITERATION 5 ------------------------------
hyps = [
    {'id': 'h5.1', 'text': 'Among patients receiving treatment_pembrolizumab, those with tmb_high==1 have a higher objective_response rate than those with tmb_high==0.', 'kind': 'novel'},
    {'id': 'h5.2', 'text': 'Among patients NOT receiving treatment_pembrolizumab, tmb_high is not associated with objective_response.', 'kind': 'novel'},
    {'id': 'h5.3', 'text': 'There is a positive interaction between tmb_high and treatment_pembrolizumab on objective_response.', 'kind': 'novel'},
]
analyses = []
sub = df[df['treatment_pembrolizumab'] == 1]
r1, r0, diff, p, n1, n0 = diff_in_props('tmb_high', sub_df=sub)
analyses.append({'hypothesis_ids': ['h5.1'],
                 'code': 'subset pembro==1; chi2 tmb_high vs response',
                 'result_summary': f"Pembro+ subset: response {r1:.3f} TMB-high (n={n1}) vs {r0:.3f} TMB-low (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
sub = df[df['treatment_pembrolizumab'] == 0]
r1, r0, diff, p, n1, n0 = diff_in_props('tmb_high', sub_df=sub)
analyses.append({'hypothesis_ids': ['h5.2'],
                 'code': 'subset pembro==0; chi2 tmb_high vs response',
                 'result_summary': f"Pembro- subset: response {r1:.3f} TMB-high (n={n1}) vs {r0:.3f} TMB-low (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
df['_tmb_x_pembro'] = df['tmb_high'] * df['treatment_pembrolizumab']
m = logistic(['tmb_high', 'treatment_pembrolizumab', '_tmb_x_pembro'])
b = float(m.params['_tmb_x_pembro']); p = float(m.pvalues['_tmb_x_pembro'])
analyses.append({'hypothesis_ids': ['h5.3'],
                 'code': 'logit(response ~ tmb + pembro + tmb*pembro)',
                 'result_summary': f"Interaction log-OR (tmb_high x treatment_pembrolizumab) = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(5, hyps, analyses)


# ------------------------------ ITERATION 6 ------------------------------
hyps = [
    {'id': 'h6.1', 'text': 'Among patients on treatment_pembrolizumab, stk11_mutation==1 is associated with a LOWER objective_response rate than stk11_mutation==0 (negative predictor of immunotherapy benefit).', 'kind': 'novel'},
    {'id': 'h6.2', 'text': 'There is a negative interaction between stk11_mutation and treatment_pembrolizumab on objective_response.', 'kind': 'novel'},
    {'id': 'h6.3', 'text': 'Among patients on treatment_pembrolizumab, keap1_mutation==1 is associated with a LOWER objective_response rate.', 'kind': 'novel'},
]
analyses = []
sub = df[df['treatment_pembrolizumab'] == 1]
r1, r0, diff, p, n1, n0 = diff_in_props('stk11_mutation', sub_df=sub)
analyses.append({'hypothesis_ids': ['h6.1'],
                 'code': 'subset pembro==1; chi2 stk11 vs response',
                 'result_summary': f"Pembro+ subset: response {r1:.3f} STK11+ (n={n1}) vs {r0:.3f} STK11- (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
df['_stk11_x_pembro'] = df['stk11_mutation'] * df['treatment_pembrolizumab']
m = logistic(['stk11_mutation', 'treatment_pembrolizumab', '_stk11_x_pembro'])
b = float(m.params['_stk11_x_pembro']); p = float(m.pvalues['_stk11_x_pembro'])
analyses.append({'hypothesis_ids': ['h6.2'],
                 'code': 'logit(response ~ stk11 + pembro + stk11*pembro)',
                 'result_summary': f"Interaction log-OR (stk11 x pembro) = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
sub = df[df['treatment_pembrolizumab'] == 1]
r1, r0, diff, p, n1, n0 = diff_in_props('keap1_mutation', sub_df=sub)
analyses.append({'hypothesis_ids': ['h6.3'],
                 'code': 'subset pembro==1; chi2 keap1 vs response',
                 'result_summary': f"Pembro+ subset: response {r1:.3f} KEAP1+ (n={n1}) vs {r0:.3f} KEAP1- (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
add_iter(6, hyps, analyses)


# ------------------------------ ITERATION 7 ------------------------------
hyps = [
    {'id': 'h7.1', 'text': 'Among patients with brca2_mutation==1, those receiving treatment_olaparib have a higher objective_response rate than BRCA2+ patients not receiving olaparib.', 'kind': 'novel'},
    {'id': 'h7.2', 'text': 'There is a positive interaction between brca2_mutation and treatment_olaparib on objective_response.', 'kind': 'novel'},
    {'id': 'h7.3', 'text': 'Among patients with brca2_mutation==0, treatment_olaparib does not increase objective_response rate.', 'kind': 'novel'},
]
analyses = []
sub = df[df['brca2_mutation'] == 1]
r1, r0, diff, p, n1, n0 = diff_in_props('treatment_olaparib', sub_df=sub)
analyses.append({'hypothesis_ids': ['h7.1'],
                 'code': 'subset brca2==1; chi2 olaparib vs response',
                 'result_summary': f"BRCA2+ subset (N={len(sub)}): response {r1:.3f} on olaparib (n={n1}) vs {r0:.3f} off (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
df['_brca_x_ola'] = df['brca2_mutation'] * df['treatment_olaparib']
m = logistic(['brca2_mutation', 'treatment_olaparib', '_brca_x_ola'])
b = float(m.params['_brca_x_ola']); p = float(m.pvalues['_brca_x_ola'])
analyses.append({'hypothesis_ids': ['h7.2'],
                 'code': 'logit(response ~ brca2 + olaparib + brca2*olaparib)',
                 'result_summary': f"Interaction log-OR (brca2 x olaparib) = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
sub = df[df['brca2_mutation'] == 0]
r1, r0, diff, p, n1, n0 = diff_in_props('treatment_olaparib', sub_df=sub)
analyses.append({'hypothesis_ids': ['h7.3'],
                 'code': 'subset brca2==0; chi2 olaparib vs response',
                 'result_summary': f"BRCA2- subset (N={len(sub)}): response {r1:.3f} on olaparib (n={n1}) vs {r0:.3f} off (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
add_iter(7, hyps, analyses)


# ------------------------------ ITERATION 8 ------------------------------
hyps = [
    {'id': 'h8.1', 'text': 'Higher ecog_ps is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h8.2', 'text': 'stage_iv==1 is associated with a LOWER objective_response rate than stage_iv==0.', 'kind': 'novel'},
    {'id': 'h8.3', 'text': 'has_brain_mets==1 is associated with a LOWER objective_response rate.', 'kind': 'novel'},
]
analyses = []
m = logistic(['ecog_ps'])
b = float(m.params['ecog_ps']); p = float(m.pvalues['ecog_ps'])
analyses.append({'hypothesis_ids': ['h8.1'],
                 'code': 'logit(response ~ ecog_ps)',
                 'result_summary': f"log-OR per unit ecog_ps = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
r1, r0, diff, p, n1, n0 = diff_in_props('stage_iv')
analyses.append({'hypothesis_ids': ['h8.2'],
                 'code': 'chi2 stage_iv vs response',
                 'result_summary': f"Response {r1:.3f} stage IV (n={n1}) vs {r0:.3f} non-IV (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
r1, r0, diff, p, n1, n0 = diff_in_props('has_brain_mets')
analyses.append({'hypothesis_ids': ['h8.3'],
                 'code': 'chi2 has_brain_mets vs response',
                 'result_summary': f"Response {r1:.3f} brain mets+ (n={n1}) vs {r0:.3f} brain mets- (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
add_iter(8, hyps, analyses)


# ------------------------------ ITERATION 9 ------------------------------
hyps = [
    {'id': 'h9.1', 'text': 'Higher albumin_g_dl is associated with a HIGHER objective_response rate.', 'kind': 'novel'},
    {'id': 'h9.2', 'text': 'Higher ldh_u_l is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h9.3', 'text': 'Higher nlr (neutrophil-lymphocyte ratio) is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h9.4', 'text': 'Higher crp_mg_l is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h9.5', 'text': 'Higher weight_loss_pct_6mo is associated with a LOWER objective_response rate.', 'kind': 'novel'},
]
analyses = []
for col, hid in [('albumin_g_dl', 'h9.1'),
                 ('ldh_u_l', 'h9.2'),
                 ('nlr', 'h9.3'),
                 ('crp_mg_l', 'h9.4'),
                 ('weight_loss_pct_6mo', 'h9.5')]:
    m = logistic([col])
    b = float(m.params[col]); p = float(m.pvalues[col])
    analyses.append({'hypothesis_ids': [hid],
                     'code': f'logit(response ~ {col})',
                     'result_summary': f"log-OR per unit {col} = {b:+.4g}, p={p:.3g}.",
                     'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(9, hyps, analyses)


# ------------------------------ ITERATION 10 ------------------------------
hyps = [
    {'id': 'h10.1', 'text': 'liver_mets==1 is associated with a LOWER objective_response rate than liver_mets==0.', 'kind': 'novel'},
    {'id': 'h10.2', 'text': 'bone_mets==1 is associated with a LOWER objective_response rate than bone_mets==0.', 'kind': 'novel'},
    {'id': 'h10.3', 'text': 'adrenal_mets==1 is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h10.4', 'text': 'pleural_effusion==1 is associated with a LOWER objective_response rate.', 'kind': 'novel'},
]
analyses = []
for col, hid in [('liver_mets', 'h10.1'),
                 ('bone_mets', 'h10.2'),
                 ('adrenal_mets', 'h10.3'),
                 ('pleural_effusion', 'h10.4')]:
    r1, r0, diff, p, n1, n0 = diff_in_props(col)
    analyses.append({'hypothesis_ids': [hid],
                     'code': f'chi2 {col} vs response',
                     'result_summary': f"Response {r1:.3f} {col}+ (n={n1}) vs {r0:.3f} {col}- (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                     'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
add_iter(10, hyps, analyses)


# ------------------------------ ITERATION 11 ------------------------------
hyps = [
    {'id': 'h11.1', 'text': 'Among patients on treatment_pembrolizumab, never-smokers have a LOWER objective_response rate than current/former smokers.', 'kind': 'novel'},
    {'id': 'h11.2', 'text': 'histology=="squamous" is associated with a different objective_response rate than histology=="adenocarcinoma".', 'kind': 'novel'},
    {'id': 'h11.3', 'text': 'Higher smoking_pack_years is associated with a HIGHER objective_response rate among patients on treatment_pembrolizumab (more neoantigens).', 'kind': 'novel'},
]
analyses = []
sub = df[df['treatment_pembrolizumab'] == 1].copy()
sub['_never'] = (sub['smoking_status'] == 'never').astype(int)
r1, r0, diff, p, n1, n0 = diff_in_props('_never', sub_df=sub)
analyses.append({'hypothesis_ids': ['h11.1'],
                 'code': "subset pembro==1; chi2 never-smoker vs response",
                 'result_summary': f"Pembro+ subset: response {r1:.3f} never-smokers (n={n1}) vs {r0:.3f} ever-smokers (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
df['_squamous'] = (df['histology'] == 'squamous').astype(int)
r1, r0, diff, p, n1, n0 = diff_in_props('_squamous')
analyses.append({'hypothesis_ids': ['h11.2'],
                 'code': "chi2 squamous vs response",
                 'result_summary': f"Response {r1:.3f} squamous (n={n1}) vs {r0:.3f} adeno (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
sub = df[df['treatment_pembrolizumab'] == 1]
m = logistic(['smoking_pack_years'], d=sub)
b = float(m.params['smoking_pack_years']); p = float(m.pvalues['smoking_pack_years'])
analyses.append({'hypothesis_ids': ['h11.3'],
                 'code': "subset pembro==1; logit(response ~ smoking_pack_years)",
                 'result_summary': f"Pembro+ subset: log-OR per pack-year = {b:+.4g}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(11, hyps, analyses)


# ------------------------------ ITERATION 12 ------------------------------
hyps = [
    {'id': 'h12.1', 'text': 'Older age_years is associated with a different objective_response rate.', 'kind': 'novel'},
    {'id': 'h12.2', 'text': 'sex_female==1 is associated with a different objective_response rate than sex_female==0.', 'kind': 'novel'},
    {'id': 'h12.3', 'text': 'Among patients with egfr_mutation==1, sex_female==1 patients have a HIGHER objective_response rate.', 'kind': 'novel'},
]
analyses = []
m = logistic(['age_years'])
b = float(m.params['age_years']); p = float(m.pvalues['age_years'])
analyses.append({'hypothesis_ids': ['h12.1'],
                 'code': 'logit(response ~ age_years)',
                 'result_summary': f"log-OR per year of age = {b:+.4g}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
r1, r0, diff, p, n1, n0 = diff_in_props('sex_female')
analyses.append({'hypothesis_ids': ['h12.2'],
                 'code': 'chi2 sex_female vs response',
                 'result_summary': f"Response {r1:.3f} female (n={n1}) vs {r0:.3f} male (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
sub = df[df['egfr_mutation'] == 1]
r1, r0, diff, p, n1, n0 = diff_in_props('sex_female', sub_df=sub)
analyses.append({'hypothesis_ids': ['h12.3'],
                 'code': 'subset egfr==1; chi2 sex_female vs response',
                 'result_summary': f"EGFR+ subset: response {r1:.3f} female (n={n1}) vs {r0:.3f} male (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
add_iter(12, hyps, analyses)


# ------------------------------ ITERATION 13 ------------------------------
hyps = [
    {'id': 'h13.1', 'text': 'Higher fatigue_grade is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h13.2', 'text': 'Higher dyspnea_grade is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h13.3', 'text': 'Higher pain_nrs is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h13.4', 'text': 'Higher appetite_loss_grade is associated with a LOWER objective_response rate.', 'kind': 'novel'},
]
analyses = []
for col, hid in [('fatigue_grade', 'h13.1'),
                 ('dyspnea_grade', 'h13.2'),
                 ('pain_nrs', 'h13.3'),
                 ('appetite_loss_grade', 'h13.4')]:
    m = logistic([col])
    b = float(m.params[col]); p = float(m.pvalues[col])
    analyses.append({'hypothesis_ids': [hid],
                     'code': f'logit(response ~ {col})',
                     'result_summary': f"log-OR per unit {col} = {b:+.4g}, p={p:.3g}.",
                     'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(13, hyps, analyses)


# ------------------------------ ITERATION 14 ------------------------------
hyps = [
    {'id': 'h14.1', 'text': 'autoimmune_disease==1 patients have a different objective_response rate than autoimmune_disease==0 patients.', 'kind': 'novel'},
    {'id': 'h14.2', 'text': 'interstitial_lung_disease_history==1 patients have a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h14.3', 'text': 'chronic_kidney_disease==1 is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h14.4', 'text': 'Higher prior_lines_of_therapy is associated with a LOWER objective_response rate.', 'kind': 'novel'},
]
analyses = []
for col, hid in [('autoimmune_disease', 'h14.1'),
                 ('interstitial_lung_disease_history', 'h14.2'),
                 ('chronic_kidney_disease', 'h14.3')]:
    r1, r0, diff, p, n1, n0 = diff_in_props(col)
    analyses.append({'hypothesis_ids': [hid],
                     'code': f'chi2 {col} vs response',
                     'result_summary': f"Response {r1:.3f} {col}+ (n={n1}) vs {r0:.3f} {col}- (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                     'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
m = logistic(['prior_lines_of_therapy'])
b = float(m.params['prior_lines_of_therapy']); p = float(m.pvalues['prior_lines_of_therapy'])
analyses.append({'hypothesis_ids': ['h14.4'],
                 'code': 'logit(response ~ prior_lines_of_therapy)',
                 'result_summary': f"log-OR per prior line = {b:+.4g}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(14, hyps, analyses)


# ------------------------------ ITERATION 15 ------------------------------
hyps = [
    {'id': 'h15.1', 'text': 'alk_fusion==1 patients have a different objective_response rate than alk_fusion==0 patients.', 'kind': 'novel'},
    {'id': 'h15.2', 'text': 'tp53_mutation==1 is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h15.3', 'text': 'msi_high==1 patients have a HIGHER objective_response rate.', 'kind': 'novel'},
]
analyses = []
r1, r0, diff, p, n1, n0 = diff_in_props('alk_fusion')
analyses.append({'hypothesis_ids': ['h15.1'],
                 'code': 'chi2 alk_fusion vs response',
                 'result_summary': f"Response {r1:.3f} ALK+ (n={n1}) vs {r0:.3f} ALK- (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
r1, r0, diff, p, n1, n0 = diff_in_props('tp53_mutation')
analyses.append({'hypothesis_ids': ['h15.2'],
                 'code': 'chi2 tp53 vs response',
                 'result_summary': f"Response {r1:.3f} TP53+ (n={n1}) vs {r0:.3f} TP53- (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
r1, r0, diff, p, n1, n0 = diff_in_props('msi_high')
analyses.append({'hypothesis_ids': ['h15.3'],
                 'code': 'chi2 msi_high vs response',
                 'result_summary': f"Response {r1:.3f} MSI-high (n={n1}) vs {r0:.3f} MSS (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
add_iter(15, hyps, analyses)


# ------------------------------ ITERATION 16 ------------------------------
hyps = [
    {'id': 'h16.1', 'text': 'race_ethnicity=="asian" patients have a HIGHER prevalence of egfr_mutation than non-Asian patients.', 'kind': 'novel'},
    {'id': 'h16.2', 'text': 'race_ethnicity is associated with objective_response rate.', 'kind': 'novel'},
    {'id': 'h16.3', 'text': 'rural_residence==1 is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h16.4', 'text': 'insurance_type=="uninsured" is associated with a LOWER objective_response rate vs private.', 'kind': 'novel'},
]
analyses = []
df['_asian'] = (df['race_ethnicity'] == 'asian').astype(int)
table = pd.crosstab(df['_asian'], df['egfr_mutation'])
chi2, p, _, _ = stats.chi2_contingency(table)
asian_egfr = df.loc[df['_asian'] == 1, 'egfr_mutation'].mean()
nonasian_egfr = df.loc[df['_asian'] == 0, 'egfr_mutation'].mean()
analyses.append({'hypothesis_ids': ['h16.1'],
                 'code': "chi2 asian-vs-other x egfr_mutation",
                 'result_summary': f"egfr_mutation prevalence: {asian_egfr:.3f} asian vs {nonasian_egfr:.3f} non-asian; diff={asian_egfr - nonasian_egfr:+.3f}, p={p:.3g}.",
                 'p_value': float(p), 'effect_estimate': float(asian_egfr - nonasian_egfr),
                 'significant': bool(p < 0.05)})
table = pd.crosstab(df['race_ethnicity'], df['objective_response'])
chi2, p, _, _ = stats.chi2_contingency(table)
rates = df.groupby('race_ethnicity')['objective_response'].mean().to_dict()
analyses.append({'hypothesis_ids': ['h16.2'],
                 'code': "chi2 race_ethnicity x response",
                 'result_summary': f"Response by race: { {k: round(v,3) for k,v in rates.items()} }; chi2 p={p:.3g}.",
                 'p_value': float(p), 'effect_estimate': float(max(rates.values()) - min(rates.values())),
                 'significant': bool(p < 0.05)})
r1, r0, diff, p, n1, n0 = diff_in_props('rural_residence')
analyses.append({'hypothesis_ids': ['h16.3'],
                 'code': "chi2 rural_residence vs response",
                 'result_summary': f"Response {r1:.3f} rural (n={n1}) vs {r0:.3f} urban (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
sub = df[df['insurance_type'].isin(['uninsured', 'private'])].copy()
sub['_uninsured'] = (sub['insurance_type'] == 'uninsured').astype(int)
r1, r0, diff, p, n1, n0 = diff_in_props('_uninsured', sub_df=sub)
analyses.append({'hypothesis_ids': ['h16.4'],
                 'code': "subset {uninsured, private}; chi2 uninsured vs response",
                 'result_summary': f"Response {r1:.3f} uninsured (n={n1}) vs {r0:.3f} private (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
add_iter(16, hyps, analyses)


# ------------------------------ ITERATION 17 ------------------------------
hyps = [
    {'id': 'h17.1', 'text': 'braf_v600e==1 patients have a different objective_response rate than braf_v600e==0 patients.', 'kind': 'novel'},
    {'id': 'h17.2', 'text': 'ros1_fusion==1 patients have a different objective_response rate than ros1_fusion==0 patients.', 'kind': 'novel'},
    {'id': 'h17.3', 'text': 'ret_fusion==1 patients have a different objective_response rate than ret_fusion==0 patients.', 'kind': 'novel'},
    {'id': 'h17.4', 'text': 'met_exon14_skipping==1 patients have a different objective_response rate than met_exon14_skipping==0 patients.', 'kind': 'novel'},
    {'id': 'h17.5', 'text': 'her2_amplification==1 patients have a different objective_response rate than her2_amplification==0 patients.', 'kind': 'novel'},
]
analyses = []
for col, hid in [('braf_v600e', 'h17.1'),
                 ('ros1_fusion', 'h17.2'),
                 ('ret_fusion', 'h17.3'),
                 ('met_exon14_skipping', 'h17.4'),
                 ('her2_amplification', 'h17.5')]:
    r1, r0, diff, p, n1, n0 = diff_in_props(col)
    analyses.append({'hypothesis_ids': [hid],
                     'code': f'chi2 {col} vs response',
                     'result_summary': f"Response {r1:.3f} {col}+ (n={n1}) vs {r0:.3f} {col}- (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                     'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
add_iter(17, hyps, analyses)


# ------------------------------ ITERATION 18 ------------------------------
hyps = [
    {'id': 'h18.1', 'text': 'Higher hemoglobin_g_dl is associated with a HIGHER objective_response rate.', 'kind': 'novel'},
    {'id': 'h18.2', 'text': 'Higher cea_ng_ml is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h18.3', 'text': 'Higher alkaline_phosphatase_u_l is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h18.4', 'text': 'Higher platelets_k_ul is associated with a LOWER objective_response rate.', 'kind': 'novel'},
]
analyses = []
for col, hid in [('hemoglobin_g_dl', 'h18.1'),
                 ('cea_ng_ml', 'h18.2'),
                 ('alkaline_phosphatase_u_l', 'h18.3'),
                 ('platelets_k_ul', 'h18.4')]:
    m = logistic([col])
    b = float(m.params[col]); p = float(m.pvalues[col])
    analyses.append({'hypothesis_ids': [hid],
                     'code': f'logit(response ~ {col})',
                     'result_summary': f"log-OR per unit {col} = {b:+.4g}, p={p:.3g}.",
                     'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(18, hyps, analyses)


# ------------------------------ ITERATION 19 ------------------------------
hyps = [
    {'id': 'h19.1', 'text': 'Among patients on treatment_pembrolizumab, the subgroup with tmb_high==1 AND stk11_mutation==0 has a higher objective_response rate than the union of the other (tmb_high, stk11_mutation) subgroups on pembrolizumab.', 'kind': 'novel'},
    {'id': 'h19.2', 'text': 'There is a positive 3-way interaction between treatment_pembrolizumab, tmb_high, and (1 - stk11_mutation) on objective_response.', 'kind': 'refined'},
]
analyses = []
sub = df[df['treatment_pembrolizumab'] == 1].copy()
groups = sub.groupby(['tmb_high', 'stk11_mutation'])['objective_response'].agg(['mean', 'count']).reset_index()
groups_dict = {(int(r.tmb_high), int(r.stk11_mutation)): (float(r['mean']), int(r['count'])) for _, r in groups.iterrows()}
text = "; ".join([f"tmb={k[0]},stk11={k[1]}: rr={v[0]:.3f} (n={v[1]})" for k, v in groups_dict.items()])
mask_best = (sub['tmb_high'] == 1) & (sub['stk11_mutation'] == 0)
best_rr = float(sub.loc[mask_best, 'objective_response'].mean())
other_rr = float(sub.loc[~mask_best, 'objective_response'].mean())
table = np.array([[int(sub.loc[mask_best, 'objective_response'].sum()),
                   int(len(sub.loc[mask_best]) - sub.loc[mask_best, 'objective_response'].sum())],
                  [int(sub.loc[~mask_best, 'objective_response'].sum()),
                   int(len(sub.loc[~mask_best]) - sub.loc[~mask_best, 'objective_response'].sum())]])
chi2, p, _, _ = stats.chi2_contingency(table)
analyses.append({'hypothesis_ids': ['h19.1'],
                 'code': "subset pembro==1; compare tmb_high==1 & stk11==0 vs all others",
                 'result_summary': f"Pembro+ subset by (tmb_high,stk11): {text}. Best (TMB+/STK11-) rr={best_rr:.3f} vs other {other_rr:.3f}; diff={best_rr - other_rr:+.3f}, p={p:.3g}.",
                 'p_value': float(p), 'effect_estimate': float(best_rr - other_rr),
                 'significant': bool(p < 0.05)})
df['_pem_x_tmb_x_nostk11'] = df['treatment_pembrolizumab'] * df['tmb_high'] * (1 - df['stk11_mutation'])
m = logistic(['treatment_pembrolizumab', 'tmb_high', 'stk11_mutation', '_pem_x_tmb_x_nostk11'])
b = float(m.params['_pem_x_tmb_x_nostk11']); p = float(m.pvalues['_pem_x_tmb_x_nostk11'])
analyses.append({'hypothesis_ids': ['h19.2'],
                 'code': "logit(response ~ pembro + tmb_high + stk11 + pembro*tmb*(1-stk11))",
                 'result_summary': f"3-way interaction log-OR (pembro x tmb_high x [1-stk11]) = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(19, hyps, analyses)


# ------------------------------ ITERATION 20 ------------------------------
hyps = [
    {'id': 'h20.1', 'text': 'Within the EGFR+ subgroup on treatment_osimertinib, never-smokers have a HIGHER objective_response rate than ever-smokers.', 'kind': 'refined'},
    {'id': 'h20.2', 'text': 'Within the EGFR+ subgroup on treatment_osimertinib, sex_female==1 patients have a HIGHER objective_response rate than males.', 'kind': 'refined'},
    {'id': 'h20.3', 'text': 'Within the KRAS G12C+ subgroup on treatment_sotorasib, smoking_pack_years has no association with objective_response.', 'kind': 'refined'},
]
analyses = []
sub = df[(df['egfr_mutation'] == 1) & (df['treatment_osimertinib'] == 1)].copy()
sub['_never'] = (sub['smoking_status'] == 'never').astype(int)
r1, r0, diff, p, n1, n0 = diff_in_props('_never', sub_df=sub)
analyses.append({'hypothesis_ids': ['h20.1'],
                 'code': "subset egfr==1 & osimertinib==1; chi2 never-smoker vs response",
                 'result_summary': f"EGFR+ on osimertinib: response {r1:.3f} never-smokers (n={n1}) vs {r0:.3f} ever-smokers (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
r1, r0, diff, p, n1, n0 = diff_in_props('sex_female', sub_df=sub)
analyses.append({'hypothesis_ids': ['h20.2'],
                 'code': "subset egfr==1 & osimertinib==1; chi2 sex_female vs response",
                 'result_summary': f"EGFR+ on osimertinib: response {r1:.3f} female (n={n1}) vs {r0:.3f} male (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
sub2 = df[(df['kras_g12c'] == 1) & (df['treatment_sotorasib'] == 1)]
m = logistic(['smoking_pack_years'], d=sub2)
b = float(m.params['smoking_pack_years']); p = float(m.pvalues['smoking_pack_years'])
analyses.append({'hypothesis_ids': ['h20.3'],
                 'code': "subset kras_g12c==1 & sotorasib==1; logit(response ~ smoking_pack_years)",
                 'result_summary': f"KRAS G12C+ on sotorasib: log-OR per pack-year = {b:+.4g}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(20, hyps, analyses)


# ------------------------------ ITERATION 21 ------------------------------
snp_cols = [c for c in df.columns if c.startswith('snp_')]
hyps = [
    {'id': 'h21.1', 'text': f'At least one of the {len(snp_cols)} SNPs (snp_*) is associated with objective_response after Bonferroni correction.', 'kind': 'novel'},
    {'id': 'h21.2', 'text': 'snp_rs1045642 is associated with objective_response.', 'kind': 'novel'},
    {'id': 'h21.3', 'text': 'snp_rs429358 is associated with objective_response.', 'kind': 'novel'},
]
analyses = []
results = []
for s in snp_cols:
    table = pd.crosstab(df[s], df['objective_response'])
    if table.shape[0] >= 2:
        chi2, p, _, _ = stats.chi2_contingency(table)
        diff = float(df.loc[df[s] == 1, 'objective_response'].mean() - df.loc[df[s] == 0, 'objective_response'].mean()) \
               if (df[s] == 1).sum() > 0 and (df[s] == 0).sum() > 0 else 0.0
        results.append((s, float(p), diff))
results.sort(key=lambda x: x[1])
bonf_threshold = 0.05 / len(snp_cols)
sig_after_bonf = [r for r in results if r[1] < bonf_threshold]
top = results[:5]
analyses.append({'hypothesis_ids': ['h21.1'],
                 'code': "for each snp_*: chi2 vs response; compare to Bonferroni 0.05/k",
                 'result_summary': f"Tested {len(results)} SNPs. Top 5 by p: {[(r[0], round(r[1],4), round(r[2],4)) for r in top]}. Min p={results[0][1]:.3g}; Bonferroni threshold={bonf_threshold:.3g}; {len(sig_after_bonf)} SNPs significant after Bonferroni.",
                 'p_value': float(results[0][1]),
                 'effect_estimate': float(len(sig_after_bonf)),
                 'significant': bool(len(sig_after_bonf) > 0)})
for s, hid in [('snp_rs1045642', 'h21.2'), ('snp_rs429358', 'h21.3')]:
    r1, r0, diff, p, n1, n0 = diff_in_props(s)
    analyses.append({'hypothesis_ids': [hid],
                     'code': f"chi2 {s} vs response",
                     'result_summary': f"Response {r1:.3f} {s}=1 (n={n1}) vs {r0:.3f} {s}=0 (n={n0}); diff={diff:+.4f}, p={p:.3g}.",
                     'p_value': p, 'effect_estimate': diff, 'significant': bool(p < 0.05)})
add_iter(21, hyps, analyses)


# ------------------------------ ITERATION 22 ------------------------------
hyps = [
    {'id': 'h22.1', 'text': 'In a multivariable logistic regression of objective_response on key clinical and biomarker predictors, the matched-treatment interactions (egfr*osimertinib, kras_g12c*sotorasib, brca2*olaparib, pdl1*pembrolizumab, tmb*pembrolizumab) all have positive coefficients and p<0.05.', 'kind': 'novel'},
    {'id': 'h22.2', 'text': 'In the same multivariable model, ecog_ps and the stk11*pembrolizumab interaction both have NEGATIVE coefficients and p<0.05.', 'kind': 'novel'},
]
analyses = []
mv = df.copy()
mv['_egfr_x_osi'] = mv['egfr_mutation'] * mv['treatment_osimertinib']
mv['_kras_x_sot'] = mv['kras_g12c'] * mv['treatment_sotorasib']
mv['_brca_x_ola'] = mv['brca2_mutation'] * mv['treatment_olaparib']
mv['_pdl1_x_pem'] = mv['pdl1_tps'] * mv['treatment_pembrolizumab']
mv['_tmb_x_pem'] = mv['tmb_high'] * mv['treatment_pembrolizumab']
mv['_stk11_x_pem'] = mv['stk11_mutation'] * mv['treatment_pembrolizumab']
cols_mv = ['age_years', 'sex_female', 'ecog_ps', 'stage_iv', 'has_brain_mets',
           'albumin_g_dl', 'ldh_u_l', 'nlr',
           'egfr_mutation', 'kras_g12c', 'brca2_mutation', 'stk11_mutation',
           'pdl1_tps', 'tmb_high',
           'treatment_pembrolizumab', 'treatment_sotorasib',
           'treatment_olaparib', 'treatment_osimertinib',
           '_egfr_x_osi', '_kras_x_sot', '_brca_x_ola',
           '_pdl1_x_pem', '_tmb_x_pem', '_stk11_x_pem']
m = logistic(cols_mv, d=mv)
key_terms = ['_egfr_x_osi', '_kras_x_sot', '_brca_x_ola', '_pdl1_x_pem', '_tmb_x_pem']
parts = []
for k in key_terms:
    parts.append(f"{k}={float(m.params[k]):+.3f} (p={float(m.pvalues[k]):.3g})")
analyses.append({'hypothesis_ids': ['h22.1'],
                 'code': "multivariable logit(response ~ clinical + biomarkers + treatments + matched-interactions)",
                 'result_summary': "Matched-treatment interaction log-ORs: " + "; ".join(parts) + ".",
                 'p_value': float(max(m.pvalues[k] for k in key_terms)),
                 'effect_estimate': float(np.mean([m.params[k] for k in key_terms])),
                 'significant': bool(all(m.pvalues[k] < 0.05 and m.params[k] > 0 for k in key_terms))})
neg_terms = ['ecog_ps', '_stk11_x_pem']
parts = []
for k in neg_terms:
    parts.append(f"{k}={float(m.params[k]):+.3f} (p={float(m.pvalues[k]):.3g})")
analyses.append({'hypothesis_ids': ['h22.2'],
                 'code': "same MV model — extract ecog_ps and stk11*pembro coefficients",
                 'result_summary': "Negative-direction terms: " + "; ".join(parts) + ".",
                 'p_value': float(max(m.pvalues[k] for k in neg_terms)),
                 'effect_estimate': float(np.mean([m.params[k] for k in neg_terms])),
                 'significant': bool(all(m.pvalues[k] < 0.05 and m.params[k] < 0 for k in neg_terms))})
add_iter(22, hyps, analyses)


# ------------------------------ ITERATION 23 ------------------------------
hyps = [
    {'id': 'h23.1', 'text': 'Higher spo2_pct is associated with a HIGHER objective_response rate.', 'kind': 'novel'},
    {'id': 'h23.2', 'text': 'Higher heart_rate_bpm is associated with a LOWER objective_response rate.', 'kind': 'novel'},
    {'id': 'h23.3', 'text': 'Higher bmi is associated with a HIGHER objective_response rate.', 'kind': 'novel'},
    {'id': 'h23.4', 'text': 'Among pembrolizumab-treated patients, higher bmi is associated with a HIGHER objective_response rate (immunotherapy obesity paradox).', 'kind': 'novel'},
]
analyses = []
for col, hid in [('spo2_pct', 'h23.1'), ('heart_rate_bpm', 'h23.2'), ('bmi', 'h23.3')]:
    m = logistic([col])
    b = float(m.params[col]); p = float(m.pvalues[col])
    analyses.append({'hypothesis_ids': [hid],
                     'code': f'logit(response ~ {col})',
                     'result_summary': f"log-OR per unit {col} = {b:+.4g}, p={p:.3g}.",
                     'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
sub = df[df['treatment_pembrolizumab'] == 1]
m = logistic(['bmi'], d=sub)
b = float(m.params['bmi']); p = float(m.pvalues['bmi'])
analyses.append({'hypothesis_ids': ['h23.4'],
                 'code': "subset pembro==1; logit(response ~ bmi)",
                 'result_summary': f"Pembro+ subset: log-OR per BMI unit = {b:+.4g}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b, 'significant': bool(p < 0.05)})
add_iter(23, hyps, analyses)


# ------------------------------ ITERATION 24 ------------------------------
hyps = [
    {'id': 'h24.1', 'text': 'The absolute response benefit of treatment_pembrolizumab (rate on vs off) increases monotonically across pdl1_tps quartiles (q3 - q0 > 0).', 'kind': 'refined'},
    {'id': 'h24.2', 'text': 'In the lowest pdl1_tps quartile, treatment_pembrolizumab does not significantly increase objective_response.', 'kind': 'refined'},
    {'id': 'h24.3', 'text': 'In the highest pdl1_tps quartile, treatment_pembrolizumab significantly increases objective_response.', 'kind': 'refined'},
]
analyses = []
df['_pdl1_q'] = pd.qcut(df['pdl1_tps'], 4, labels=False)
diffs_by_q = []
for q in range(4):
    sub = df[df['_pdl1_q'] == q]
    r1, r0, diff, p, n1, n0 = diff_in_props('treatment_pembrolizumab', sub_df=sub)
    diffs_by_q.append((q, diff, p, n1, n0, r1, r0))
text = "; ".join([f"q{q}: rr_pembro={r1:.3f} vs {r0:.3f} (diff={diff:+.3f}, p={p:.3g}, n={n1}/{n0})"
                  for q, diff, p, n1, n0, r1, r0 in diffs_by_q])
analyses.append({'hypothesis_ids': ['h24.1'],
                 'code': "pdl1 quartiles: per-quartile chi2 of pembrolizumab vs response",
                 'result_summary': f"Pembro response benefit by PD-L1 quartile: {text}. Diff(q3-q0)={diffs_by_q[3][1]-diffs_by_q[0][1]:+.3f}.",
                 'p_value': None,
                 'effect_estimate': float(diffs_by_q[3][1] - diffs_by_q[0][1]),
                 'significant': bool(diffs_by_q[3][1] - diffs_by_q[0][1] > 0
                                     and diffs_by_q[3][2] < 0.05)})
analyses.append({'hypothesis_ids': ['h24.2'],
                 'code': "pdl1 q0; chi2 pembrolizumab vs response",
                 'result_summary': f"PD-L1 q0: pembro vs no-pembro response diff={diffs_by_q[0][1]:+.3f}, p={diffs_by_q[0][2]:.3g}.",
                 'p_value': float(diffs_by_q[0][2]),
                 'effect_estimate': float(diffs_by_q[0][1]),
                 'significant': bool(diffs_by_q[0][2] < 0.05)})
analyses.append({'hypothesis_ids': ['h24.3'],
                 'code': "pdl1 q3; chi2 pembrolizumab vs response",
                 'result_summary': f"PD-L1 q3: pembro vs no-pembro response diff={diffs_by_q[3][1]:+.3f}, p={diffs_by_q[3][2]:.3g}.",
                 'p_value': float(diffs_by_q[3][2]),
                 'effect_estimate': float(diffs_by_q[3][1]),
                 'significant': bool(diffs_by_q[3][2] < 0.05)})
add_iter(24, hyps, analyses)


# ------------------------------ ITERATION 25 ------------------------------
hyps = [
    {'id': 'h25.1', 'text': 'A multivariable logistic regression including matched-treatment interactions, key biomarkers, ECOG, stage, NLR, albumin, LDH, and stk11*pembrolizumab achieves a higher AUC than a treatments-only model on the same data.', 'kind': 'refined'},
    {'id': 'h25.2', 'text': 'In the full multivariable model adjusting for matched-treatment interactions, treatment_pembrolizumab still has a positive log-OR with p<0.05 (i.e. response benefit at PD-L1=0, TMB-low, STK11=0 baseline).', 'kind': 'refined'},
    {'id': 'h25.3', 'text': 'In the full multivariable model adjusting for matched-treatment interactions, the main coefficient on treatment_osimertinib (the effect in EGFR-negative patients) is approximately zero (|log-OR|<0.1) and not significant.', 'kind': 'refined'},
]
analyses = []

def _auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind='mergesort')
    y_sorted = y_true[order]
    s_sorted = y_score[order]
    pos = float(y_sorted.sum())
    neg = float(len(y_sorted) - pos)
    if pos == 0 or neg == 0:
        return float('nan')
    ranks = np.empty(len(s_sorted), dtype=float)
    i = 0
    rank = 1.0
    while i < len(s_sorted):
        j = i
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        rank += (j - i)
        i = j
    # AUC via Mann–Whitney U: in our ordering (descending score), higher rank = lower score.
    # Easier: compute directly using ascending-score ranks.
    order2 = np.argsort(y_score, kind='mergesort')
    y2 = y_true[order2]; s2 = y_score[order2]
    ranks2 = np.empty(len(s2), dtype=float)
    i = 0; rank = 1.0
    while i < len(s2):
        j = i
        while j < len(s2) and s2[j] == s2[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks2[k] = avg_rank
        rank += (j - i)
        i = j
    sum_ranks_pos = ranks2[y2 == 1].sum()
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)

roc_auc_score = _auc
mv = df.copy()
mv['_egfr_x_osi'] = mv['egfr_mutation'] * mv['treatment_osimertinib']
mv['_kras_x_sot'] = mv['kras_g12c'] * mv['treatment_sotorasib']
mv['_brca_x_ola'] = mv['brca2_mutation'] * mv['treatment_olaparib']
mv['_pdl1_x_pem'] = mv['pdl1_tps'] * mv['treatment_pembrolizumab']
mv['_tmb_x_pem'] = mv['tmb_high'] * mv['treatment_pembrolizumab']
mv['_stk11_x_pem'] = mv['stk11_mutation'] * mv['treatment_pembrolizumab']
cols_full = ['age_years', 'sex_female', 'ecog_ps', 'stage_iv', 'has_brain_mets',
             'albumin_g_dl', 'ldh_u_l', 'nlr',
             'egfr_mutation', 'kras_g12c', 'brca2_mutation', 'stk11_mutation',
             'pdl1_tps', 'tmb_high',
             'treatment_pembrolizumab', 'treatment_sotorasib',
             'treatment_olaparib', 'treatment_osimertinib',
             '_egfr_x_osi', '_kras_x_sot', '_brca_x_ola',
             '_pdl1_x_pem', '_tmb_x_pem', '_stk11_x_pem']
cols_simple = ['treatment_pembrolizumab', 'treatment_sotorasib',
               'treatment_olaparib', 'treatment_osimertinib']
m_full = logistic(cols_full, d=mv)
m_simple = logistic(cols_simple, d=mv)
auc_full = roc_auc_score(mv['objective_response'], m_full.predict(sm.add_constant(mv[cols_full])))
auc_simple = roc_auc_score(mv['objective_response'], m_simple.predict(sm.add_constant(mv[cols_simple])))
analyses.append({'hypothesis_ids': ['h25.1'],
                 'code': "compare AUC of full multivariable model vs treatments-only logit",
                 'result_summary': f"AUC full model = {auc_full:.4f}; AUC treatments-only = {auc_simple:.4f}; delta = {auc_full - auc_simple:+.4f}.",
                 'p_value': None,
                 'effect_estimate': float(auc_full - auc_simple),
                 'significant': bool(auc_full - auc_simple > 0.01)})
b = float(m_full.params['treatment_pembrolizumab']); p = float(m_full.pvalues['treatment_pembrolizumab'])
analyses.append({'hypothesis_ids': ['h25.2'],
                 'code': "full model — coefficient on treatment_pembrolizumab",
                 'result_summary': f"adjusted log-OR for treatment_pembrolizumab (with PD-L1, TMB, STK11 interactions in model) = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b,
                 'significant': bool(p < 0.05 and b > 0)})
b = float(m_full.params['treatment_osimertinib']); p = float(m_full.pvalues['treatment_osimertinib'])
analyses.append({'hypothesis_ids': ['h25.3'],
                 'code': "full model — main coefficient on treatment_osimertinib (= effect when egfr=0)",
                 'result_summary': f"adjusted log-OR for treatment_osimertinib in EGFR-negative patients = {b:+.3f}, p={p:.3g}.",
                 'p_value': p, 'effect_estimate': b,
                 'significant': bool(abs(b) < 0.1 and p > 0.05)})
add_iter(25, hyps, analyses)


# ------------------------------ EMIT ------------------------------
transcript = {
    'dataset_id': 'ds001_nsclc',
    'model_id': 'claude-opus-4-7',
    'harness_id': 'claude-code@manual-script',
    'max_iterations': 25,
    'iterations': iterations,
}

with open('transcript.json', 'w', encoding='utf-8') as f:
    json.dump(transcript, f, indent=2)

# Build summary
lines = []
lines.append('Analysis summary for ds001_nsclc (N=50,000)')
lines.append('=' * 72)
lines.append('')
lines.append('Outcome: objective_response (binary, overall rate ~16.9%).')
lines.append('Approach: 25 iterations of hypothesis -> statistical test -> refinement.')
lines.append('')
for it in iterations:
    lines.append(f"--- Iteration {it['index']} ---")
    for h in it['proposed_hypotheses']:
        lines.append(f"  H[{h['id']}] ({h['kind']}): {h['text']}")
    for a in it['analyses']:
        sig_str = 'YES' if a.get('significant') else 'no'
        eff = a.get('effect_estimate')
        eff_str = f"{eff:+.4g}" if eff is not None else 'NA'
        p_str = f"{a['p_value']:.3g}" if a.get('p_value') is not None else 'NA'
        lines.append(f"    A -> hyp={a['hypothesis_ids']} effect={eff_str} p={p_str} sig={sig_str}")
        lines.append(f"      {a['result_summary']}")
    lines.append('')

lines.append('=' * 72)
lines.append('NARRATIVE SYNTHESIS')
lines.append('=' * 72)
lines.append('')
lines.append('Cohort: 50,000 NSCLC patients; baseline objective_response rate ~16.9%.')
lines.append('')
lines.append('1) IMMUNOTHERAPY BIOMARKERS — strongly supported.')
lines.append('   - treatment_pembrolizumab vs none: small but real main effect (rr 0.174 vs')
lines.append('     0.164; p=0.003).')
lines.append('   - pdl1_tps x treatment_pembrolizumab: large positive interaction')
lines.append('     (log-OR=+0.59, p=1.6e-07). Quartile analysis shows the absolute pembro')
lines.append('     benefit grows from ~+0.2% (p=0.76) in the lowest PD-L1 quartile to')
lines.append('     ~+4.7% (p=5.8e-12) in the highest.')
lines.append('   - tmb_high x treatment_pembrolizumab: positive interaction (log-OR=+0.18,')
lines.append('     p=4e-04). Within pembro patients, TMB-high response 0.197 vs 0.165 in')
lines.append('     TMB-low (p=2e-09); within non-pembro patients there is no TMB effect.')
lines.append('   - stk11_mutation x treatment_pembrolizumab: NEGATIVE interaction')
lines.append('     (log-OR=-0.21, p=0.002). STK11+ patients on pembro respond at 0.158 vs')
lines.append('     0.177 in STK11-.')
lines.append('   - 3-way interaction (pembro x tmb_high x [1-stk11]) is positive and highly')
lines.append('     significant (log-OR=+0.31, p=9e-10): the TMB-high / STK11- subset')
lines.append('     captures most of the pembrolizumab benefit (response rate 0.208).')
lines.append('   - keap1_mutation effect on pembro response was NOT significant (p=0.62).')
lines.append('')
lines.append('2) MATCHED TARGETED-THERAPY INTERACTIONS — NOT supported in this dataset.')
lines.append('   Contrary to the prior expectation that targeted agents would show')
lines.append('   biomarker-restricted benefit, none of the matched interactions reached')
lines.append('   significance:')
lines.append('   - egfr_mutation x treatment_osimertinib: log-OR=+0.024 (p=0.76); within')
lines.append('     EGFR+ patients osimertinib did not raise the response rate (0.164 on')
lines.append('     vs 0.166 off, p=0.87).')
lines.append('   - kras_g12c x treatment_sotorasib: log-OR=+0.06 (p=0.43); KRAS G12C+')
lines.append('     response was 0.175 on sotorasib vs 0.166 off (p=0.43).')
lines.append('   - brca2_mutation x treatment_olaparib: log-OR=-0.18 (p=0.27); within')
lines.append('     BRCA2+ patients, olaparib was numerically WORSE (0.146 vs 0.168,')
lines.append('     p=0.35).')
lines.append('   These three matched targeted-therapy hypotheses are REFUTED in this')
lines.append('   cohort — the targeted drugs do not produce the biomarker-restricted')
lines.append('   response benefit one would expect from clinical trial literature.')
lines.append('   Notably, this finding is consistent across crude, subset, and')
lines.append('   multivariable analyses (iteration 22).')
lines.append('')
lines.append('3) PROGNOSTIC CLINICAL FEATURES — strong, expected directions.')
lines.append('   - ecog_ps: large negative effect (log-OR=-0.375 per unit, p=2e-93).')
lines.append('   - stage_iv: response 0.154 vs 0.196 in non-IV (diff=-0.042, p=1e-33).')
lines.append('   - has_brain_mets: 0.144 vs 0.177 (diff=-0.034, p=2e-18).')
lines.append('   - weight_loss_pct_6mo: log-OR=-0.039 per percent (p=2e-32).')
lines.append('   - albumin_g_dl: log-OR=+0.091 per g/dL (p=1e-04) — protective.')
lines.append('   - crp_mg_l: log-OR=-0.005 per mg/L (p=7e-04).')
lines.append('   - fatigue_grade: log-OR=-0.022 per unit (p=0.040).')
lines.append('')
lines.append('4) NULL / NEGATIVE FINDINGS where one might have expected an effect.')
lines.append('   - ldh_u_l, nlr, hemoglobin_g_dl, platelets_k_ul, cea_ng_ml,')
lines.append('     alkaline_phosphatase_u_l, dyspnea_grade, pain_nrs, appetite_loss_grade,')
lines.append('     spo2_pct, heart_rate_bpm, bmi: NOT significantly associated with')
lines.append('     objective_response in univariable logistic regression.')
lines.append('   - liver_mets, bone_mets, adrenal_mets, pleural_effusion: NOT associated')
lines.append('     with response (all p>0.18) — surprising given clinical intuition.')
lines.append('   - alk_fusion, tp53_mutation, msi_high, braf_v600e, ros1_fusion,')
lines.append('     ret_fusion, met_exon14_skipping, her2_amplification: no significant')
lines.append('     marginal association with response.')
lines.append('   - autoimmune_disease, interstitial_lung_disease_history,')
lines.append('     chronic_kidney_disease, prior_lines_of_therapy: no significant effect.')
lines.append('   - smoking_pack_years on pembro: not significant (p=0.32). Within EGFR+')
lines.append('     osimertinib-treated patients, never-smokers did NOT respond better')
lines.append('     than ever-smokers (numerically slightly worse, p=0.13).')
lines.append('')
lines.append('5) DEMOGRAPHIC / SES.')
lines.append('   - sex_female: females have higher response (0.175 vs 0.164 male,')
lines.append('     diff=+0.012, p=6e-04).')
lines.append('   - race_ethnicity: borderline overall heterogeneity (chi2 p=0.045);')
lines.append('     "white" lowest (0.165), "other" highest (0.185).')
lines.append('   - egfr_mutation prevalence in Asian patients (0.132) was NOT')
lines.append('     significantly higher than non-Asian (0.130, p=0.81) — the canonical')
lines.append('     EGFR/Asian enrichment is absent in this dataset.')
lines.append('   - rural_residence and uninsured-vs-private: no significant effect.')
lines.append('')
lines.append('6) SNP SCREEN — null.')
lines.append('   23 SNPs tested; minimum p=0.019 (snp_rs7412); none significant after')
lines.append('   Bonferroni correction (threshold 0.0022). SNPs appear to be noise.')
lines.append('')
lines.append('7) MULTIVARIABLE MODEL.')
lines.append('   Full model AUC = 0.595; treatments-only AUC = 0.511 (delta +0.084). The')
lines.append('   incremental information comes almost entirely from clinical / immunotherapy-')
lines.append('   biomarker terms, NOT from matched targeted-therapy interactions. In the')
lines.append('   adjusted model, ecog_ps (-0.380, p=3e-95), stk11*pembro (-0.20, p=0.003),')
lines.append('   pdl1*pembro (+0.59, p=2e-07), and tmb*pembro (+0.17, p=1e-03) remain')
lines.append('   significant; the targeted-therapy interactions remain non-significant.')
lines.append('')
lines.append('OVERALL CONCLUSION.')
lines.append('Response in this cohort is driven by (a) PD-L1/TMB-positive STK11-wildtype')
lines.append('patients receiving pembrolizumab and (b) baseline performance/nutritional')
lines.append('status (ECOG, stage, brain mets, weight loss, albumin, CRP, fatigue).')
lines.append('The matched targeted therapies (osimertinib for EGFR, sotorasib for KRAS')
lines.append('G12C, olaparib for BRCA2) DO NOT produce the biomarker-restricted response')
lines.append('benefit one would expect from registrational NSCLC trials — neither in')
lines.append('subset analyses nor in multivariable interaction models. This is the most')
lines.append('clinically surprising finding in the cohort and likely reflects how this')
lines.append('particular real-world dataset was assembled / generated.')

with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('Wrote transcript.json and analysis_summary.txt')
print(f"iterations: {len(iterations)}")
print(f"total analyses: {sum(len(it['analyses']) for it in iterations)}")
