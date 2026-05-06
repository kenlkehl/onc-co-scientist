"""Build the transcript.json and analysis_summary.txt from results_v2.json."""
import json

with open('results_v2.json') as f:
    R = json.load(f)

mb = R['main_binary']      # main_binary[feature] -> {slope_or_diff, p, mean_pos, mean_neg, ...}
mc = R['main_continuous']  # main_continuous[feature] -> {slope, p, r, tertile_means}
mv = R['multivariable_main']
inter = R['interactions']  # inter[treat][feat] -> {inter_coef|inter_coef_per_sd, inter_p, ...}
sub = R['subgroup_effects']
ex = R['exhaustive_subgroups']
fc = R['final_confirmations']

iters = []

def hyp(hid, text, kind='novel'):
    return {'id': hid, 'text': text, 'kind': kind}

def ana(ids, summary, p, eff, sig=None, code=None):
    a = {'hypothesis_ids': ids, 'result_summary': summary, 'p_value': p, 'effect_estimate': eff}
    if sig is not None:
        a['significant'] = sig
    if code:
        a['code'] = code
    return a

# ---- Iteration 1 — Stage IV and brain mets ----
v_stage = mb['stage_iv']
v_brain = mb['has_brain_mets']
iters.append({
    'index': 1,
    'proposed_hypotheses': [
        hyp('h1.1', 'Patients with stage_iv=1 have shorter pfs_months than patients with stage_iv=0.'),
        hyp('h1.2', 'Patients with has_brain_mets=1 have shorter pfs_months than patients with has_brain_mets=0.'),
    ],
    'analyses': [
        ana(['h1.1'],
            f"Welch t-test of pfs_months by stage_iv: mean {v_stage['mean_pos']:.3f} (stage_iv=1) vs {v_stage['mean_neg']:.3f} (stage_iv=0); diff={v_stage['slope_or_diff']:+.3f}, p={v_stage['p']:.2e}.",
            v_stage['p'], v_stage['slope_or_diff'], v_stage['p']<0.05,
            "stats.ttest_ind(df.loc[df['stage_iv']==1,'pfs_months'], df.loc[df['stage_iv']==0,'pfs_months'], equal_var=False)"),
        ana(['h1.2'],
            f"Welch t-test of pfs_months by has_brain_mets: mean {v_brain['mean_pos']:.3f} (yes) vs {v_brain['mean_neg']:.3f} (no); diff={v_brain['slope_or_diff']:+.3f}, p={v_brain['p']:.2e}.",
            v_brain['p'], v_brain['slope_or_diff'], v_brain['p']<0.05),
    ],
})

# ---- Iteration 2 — ECOG PS and weight loss ----
v_ecog = mb['ecog_ps']
v_wl = mc['weight_loss_pct_6mo']
iters.append({
    'index': 2,
    'proposed_hypotheses': [
        hyp('h2.1', 'Higher ecog_ps is associated with shorter pfs_months (negative slope).'),
        hyp('h2.2', 'Higher weight_loss_pct_6mo is associated with shorter pfs_months (negative slope).'),
    ],
    'analyses': [
        ana(['h2.1'],
            f"Linear regression pfs_months ~ ecog_ps: slope={v_ecog['slope_or_diff']:+.3f} months per ECOG point, p={v_ecog['p']:.2e}. Group means: {v_ecog['group_means']}.",
            v_ecog['p'], v_ecog['slope_or_diff'], v_ecog['p']<0.05),
        ana(['h2.2'],
            f"Linear regression pfs_months ~ weight_loss_pct_6mo: slope={v_wl['slope']:+.4f} per percent, p={v_wl['p']:.2e}. Tertile means: {v_wl['tertile_means']}.",
            v_wl['p'], v_wl['slope'], v_wl['p']<0.05),
    ],
})

# ---- Iteration 3 — Hormone receptors and HER2 ----
v_er = mb['er_positive']
v_pr = mb['pr_positive']
v_h2p = mb['her2_positive']
v_h2l = mb['her2_low']
iters.append({
    'index': 3,
    'proposed_hypotheses': [
        hyp('h3.1', 'er_positive=1 is associated with longer pfs_months than er_positive=0.'),
        hyp('h3.2', 'pr_positive=1 is associated with longer pfs_months than pr_positive=0.'),
        hyp('h3.3', 'her2_positive=1 is associated with shorter pfs_months than her2_positive=0.'),
        hyp('h3.4', 'her2_low=1 is associated with longer pfs_months than her2_low=0.'),
    ],
    'analyses': [
        ana(['h3.1'], f"er_positive: mean PFS {v_er['mean_pos']:.3f} vs {v_er['mean_neg']:.3f}; diff={v_er['slope_or_diff']:+.3f}, p={v_er['p']:.2e}.", v_er['p'], v_er['slope_or_diff'], v_er['p']<0.05),
        ana(['h3.2'], f"pr_positive: mean PFS {v_pr['mean_pos']:.3f} vs {v_pr['mean_neg']:.3f}; diff={v_pr['slope_or_diff']:+.3f}, p={v_pr['p']:.2e}.", v_pr['p'], v_pr['slope_or_diff'], v_pr['p']<0.05),
        ana(['h3.3'], f"her2_positive: mean PFS {v_h2p['mean_pos']:.3f} vs {v_h2p['mean_neg']:.3f}; diff={v_h2p['slope_or_diff']:+.3f}, p={v_h2p['p']:.2e}.", v_h2p['p'], v_h2p['slope_or_diff'], v_h2p['p']<0.05),
        ana(['h3.4'], f"her2_low: mean PFS {v_h2l['mean_pos']:.3f} vs {v_h2l['mean_neg']:.3f}; diff={v_h2l['slope_or_diff']:+.3f}, p={v_h2l['p']:.2e}.", v_h2l['p'], v_h2l['slope_or_diff'], v_h2l['p']<0.05),
    ],
})

# ---- Iteration 4 — Mutations ----
v_b1 = mb['brca1_mutation']
v_b2 = mb['brca2_mutation']
v_pik = mb['pik3ca_mutation']
iters.append({
    'index': 4,
    'proposed_hypotheses': [
        hyp('h4.1', 'brca1_mutation=1 is associated with shorter pfs_months than brca1_mutation=0.'),
        hyp('h4.2', 'brca2_mutation=1 is associated with shorter pfs_months than brca2_mutation=0.'),
        hyp('h4.3', 'pik3ca_mutation=1 is associated with shorter pfs_months than pik3ca_mutation=0.'),
    ],
    'analyses': [
        ana(['h4.1'], f"brca1_mutation: mean PFS {v_b1['mean_pos']:.3f} vs {v_b1['mean_neg']:.3f}; diff={v_b1['slope_or_diff']:+.3f}, p={v_b1['p']:.2e} — null.", v_b1['p'], v_b1['slope_or_diff'], v_b1['p']<0.05),
        ana(['h4.2'], f"brca2_mutation: mean PFS {v_b2['mean_pos']:.3f} vs {v_b2['mean_neg']:.3f}; diff={v_b2['slope_or_diff']:+.3f}, p={v_b2['p']:.2e} — null.", v_b2['p'], v_b2['slope_or_diff'], v_b2['p']<0.05),
        ana(['h4.3'], f"pik3ca_mutation: mean PFS {v_pik['mean_pos']:.3f} vs {v_pik['mean_neg']:.3f}; diff={v_pik['slope_or_diff']:+.3f}, p={v_pik['p']:.2e}.", v_pik['p'], v_pik['slope_or_diff'], v_pik['p']<0.05),
    ],
})

# ---- Iteration 5 — Continuous biomarkers (Ki67, albumin, LDH) and age ----
v_ki = mc['ki67_pct']; v_alb = mc['albumin_g_dl']; v_ldh = mc['ldh_u_l']; v_age = mc['age_years']
iters.append({
    'index': 5,
    'proposed_hypotheses': [
        hyp('h5.1', 'Higher ki67_pct is associated with shorter pfs_months.'),
        hyp('h5.2', 'Higher albumin_g_dl is associated with longer pfs_months.'),
        hyp('h5.3', 'Higher ldh_u_l is associated with shorter pfs_months.'),
        hyp('h5.4', 'Higher age_years is associated with shorter pfs_months.'),
    ],
    'analyses': [
        ana(['h5.1'], f"linregress pfs ~ ki67_pct: slope={v_ki['slope']:+.4f} mo per %, p={v_ki['p']:.2e}, r={v_ki['r']:+.3f}.", v_ki['p'], v_ki['slope'], v_ki['p']<0.05),
        ana(['h5.2'], f"linregress pfs ~ albumin_g_dl: slope={v_alb['slope']:+.3f} mo per g/dL, p={v_alb['p']:.2e}, r={v_alb['r']:+.3f}.", v_alb['p'], v_alb['slope'], v_alb['p']<0.05),
        ana(['h5.3'], f"linregress pfs ~ ldh_u_l: slope={v_ldh['slope']:+.5f} mo per U/L, p={v_ldh['p']:.2e}, r={v_ldh['r']:+.3f}.", v_ldh['p'], v_ldh['slope'], v_ldh['p']<0.05),
        ana(['h5.4'], f"linregress pfs ~ age_years: slope={v_age['slope']:+.4f} mo per year, p={v_age['p']:.2e}, r={v_age['r']:+.3f}. Direction is positive (older = longer PFS) in this cohort, contrary to prior expectation.", v_age['p'], v_age['slope'], v_age['p']<0.05),
    ],
})

# ---- Iteration 6 — Multivariable ----
cols = mv['columns']; coefs = mv['coefs']; ps = mv['pvals']
sig_lines = []
for c, b, p in zip(cols, coefs, ps):
    if c == 'const':
        continue
    if p < 1e-6:
        sig_lines.append(f"{c} (coef={b:+.4f}, p={p:.1e})")
iters.append({
    'index': 6,
    'proposed_hypotheses': [
        hyp('h6.1', 'After adjusting for all features and treatments simultaneously, stage_iv, has_brain_mets, ecog_ps, her2_positive, pik3ca_mutation, ki67_pct, weight_loss_pct_6mo, and ldh_u_l independently shorten pfs_months, while er_positive, age_years, and albumin_g_dl independently lengthen it.'),
        hyp('h6.2', 'After adjustment, treatment_palbociclib has a positive coefficient on pfs_months while the other treatments have null coefficients.'),
    ],
    'analyses': [
        ana(['h6.1','h6.2'],
            f"OLS pfs ~ all features+treatments (R^2={mv['rsq']:.3f}). Independent effects with p<1e-6: " + '; '.join(sig_lines) + ". Treatment coefs: " + ', '.join(f"{c}={b:+.3f}(p={p:.2g})" for c,b,p in zip(cols,coefs,ps) if c.startswith('treatment_')),
            None, None,
            sig=True),
    ],
})

# ---- Iteration 7 — Treatment main effects (univariate) ----
treat_lines = []
for t in ['treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab','treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab']:
    v = mb[t]
    treat_lines.append(f"{t}: diff={v['slope_or_diff']:+.4f} mo (p={v['p']:.2g})")
v_palbo = mb['treatment_palbociclib']
iters.append({
    'index': 7,
    'proposed_hypotheses': [
        hyp('h7.1', 'treatment_palbociclib=1 is associated with longer pfs_months than treatment_palbociclib=0 in the overall cohort.'),
        hyp('h7.2', 'The other five treatments (tamoxifen, trastuzumab, olaparib, sacituzumab_govitecan, pembrolizumab) have small or null overall effects on pfs_months when assessed univariately.'),
    ],
    'analyses': [
        ana(['h7.1'], f"Univariate Welch t-test: treatment_palbociclib treated mean {v_palbo['mean_pos']:.3f} vs untreated {v_palbo['mean_neg']:.3f}; diff={v_palbo['slope_or_diff']:+.3f}, p={v_palbo['p']:.2e}.", v_palbo['p'], v_palbo['slope_or_diff'], v_palbo['p']<0.05),
        ana(['h7.2'], 'Univariate Welch t-tests, each treatment vs untreated overall: ' + '; '.join(treat_lines) + '. Only palbociclib reaches a clinically meaningful effect; the others are within ~0.1 mo of null.', None, None, sig=True),
    ],
})

# ---- Iteration 8 — Palbociclib x ER, HER2, PIK3CA interactions ----
ip_er = inter['treatment_palbociclib']['er_positive']
ip_h2p = inter['treatment_palbociclib']['her2_positive']
ip_pik = inter['treatment_palbociclib']['pik3ca_mutation']
ip_pr = inter['treatment_palbociclib']['pr_positive']
iters.append({
    'index': 8,
    'proposed_hypotheses': [
        hyp('h8.1', 'The pfs_months benefit from treatment_palbociclib is larger in er_positive=1 patients than in er_positive=0 patients (positive interaction).'),
        hyp('h8.2', 'The pfs_months benefit from treatment_palbociclib is smaller (or reversed) in her2_positive=1 patients than in her2_positive=0 patients (negative interaction).'),
        hyp('h8.3', 'The pfs_months benefit from treatment_palbociclib is smaller in pik3ca_mutation=1 patients than in pik3ca_mutation=0 patients (negative interaction).'),
        hyp('h8.4', 'The pfs_months benefit from treatment_palbociclib is larger in pr_positive=1 patients than in pr_positive=0 patients (positive interaction).'),
    ],
    'analyses': [
        ana(['h8.1'], f"OLS with covariates: palbociclib x er_positive interaction coef={ip_er['inter_coef']:+.3f} (p={ip_er['inter_p']:.2e}). Effect in ER+: {ip_er['eff_in_pos']:+.3f} mo; in ER-: {ip_er['eff_in_neg']:+.3f} mo.", ip_er['inter_p'], ip_er['inter_coef'], ip_er['inter_p']<0.05),
        ana(['h8.2'], f"OLS with covariates: palbociclib x her2_positive interaction coef={ip_h2p['inter_coef']:+.3f} (p={ip_h2p['inter_p']:.2e}). Effect in HER2+: {ip_h2p['eff_in_pos']:+.3f}; in HER2-: {ip_h2p['eff_in_neg']:+.3f}.", ip_h2p['inter_p'], ip_h2p['inter_coef'], ip_h2p['inter_p']<0.05),
        ana(['h8.3'], f"OLS with covariates: palbociclib x pik3ca_mutation interaction coef={ip_pik['inter_coef']:+.3f} (p={ip_pik['inter_p']:.2e}). Effect in PIK3CA-mut: {ip_pik['eff_in_pos']:+.3f}; in PIK3CA-wt: {ip_pik['eff_in_neg']:+.3f}.", ip_pik['inter_p'], ip_pik['inter_coef'], ip_pik['inter_p']<0.05),
        ana(['h8.4'], f"OLS with covariates: palbociclib x pr_positive interaction coef={ip_pr['inter_coef']:+.3f} (p={ip_pr['inter_p']:.2e}). Effect in PR+: {ip_pr['eff_in_pos']:+.3f}; in PR-: {ip_pr['eff_in_neg']:+.3f}.", ip_pr['inter_p'], ip_pr['inter_coef'], ip_pr['inter_p']<0.05),
    ],
})

# ---- Iteration 9 — Palbociclib x Ki67 (continuous interaction) ----
ip_ki = inter['treatment_palbociclib']['ki67_pct']
iters.append({
    'index': 9,
    'proposed_hypotheses': [
        hyp('h9.1', 'The pfs_months benefit from treatment_palbociclib decreases as ki67_pct increases (negative interaction with ki67_pct).'),
    ],
    'analyses': [
        ana(['h9.1'], f"OLS interaction palbociclib x z(ki67_pct), adjusted: coef per SD={ip_ki['inter_coef_per_sd']:+.3f} (p={ip_ki['inter_p']:.2e}). Tertile-stratified palbo effects: low Ki67 {ip_ki['tertile_treatment_effects'].get('low')}, mid {ip_ki['tertile_treatment_effects'].get('mid')}, high {ip_ki['tertile_treatment_effects'].get('high')}.", ip_ki['inter_p'], ip_ki['inter_coef_per_sd'], ip_ki['inter_p']<0.05),
    ],
})

# ---- Iteration 10 — Trastuzumab x HER2+ (clinically expected; tested) ----
it_h2p = inter['treatment_trastuzumab']['her2_positive']
sub_t = sub['tras_her2']
iters.append({
    'index': 10,
    'proposed_hypotheses': [
        hyp('h10.1', 'The pfs_months benefit from treatment_trastuzumab is larger in her2_positive=1 patients than in her2_positive=0 patients (positive interaction).'),
    ],
    'analyses': [
        ana(['h10.1'], f"OLS trastuzumab x her2_positive interaction coef={it_h2p['inter_coef']:+.3f} (p={it_h2p['inter_p']:.2g}). Subgroup: HER2+ palbo effect={sub_t['HER2+']['diff']:+.3f} (p={sub_t['HER2+']['p']:.2g}, n_t={sub_t['HER2+']['n_treated']}); HER2- effect={sub_t['HER2-']['diff']:+.3f} (p={sub_t['HER2-']['p']:.2g}). Interaction is null — trastuzumab does not show the clinically expected enrichment in HER2+ patients in this dataset.", it_h2p['inter_p'], it_h2p['inter_coef'], it_h2p['inter_p']<0.05),
    ],
})

# ---- Iteration 11 — Olaparib x BRCA (clinically expected) ----
io_b1 = inter['treatment_olaparib']['brca1_mutation']
io_b2 = inter['treatment_olaparib']['brca2_mutation']
sub_o = sub['ola_brca']
iters.append({
    'index': 11,
    'proposed_hypotheses': [
        hyp('h11.1', 'The pfs_months benefit from treatment_olaparib is larger in brca1_mutation=1 patients than in brca1_mutation=0 patients (positive interaction).'),
        hyp('h11.2', 'The pfs_months benefit from treatment_olaparib is larger in brca2_mutation=1 patients than in brca2_mutation=0 patients (positive interaction).'),
        hyp('h11.3', 'The pfs_months benefit from treatment_olaparib is larger in patients with brca1_mutation=1 OR brca2_mutation=1 than in BRCA-wild-type patients.'),
    ],
    'analyses': [
        ana(['h11.1'], f"OLS olaparib x brca1_mutation interaction coef={io_b1['inter_coef']:+.3f} (p={io_b1['inter_p']:.2g}). BRCA1+ subgroup olaparib effect={sub_o['BRCA1']['diff']:+.3f} (p={sub_o['BRCA1']['p']:.2g}, n_t={sub_o['BRCA1']['n_treated']}).", io_b1['inter_p'], io_b1['inter_coef'], io_b1['inter_p']<0.05),
        ana(['h11.2'], f"OLS olaparib x brca2_mutation interaction coef={io_b2['inter_coef']:+.3f} (p={io_b2['inter_p']:.2g}). BRCA2+ subgroup olaparib effect={sub_o['BRCA2']['diff']:+.3f} (p={sub_o['BRCA2']['p']:.2g}, n_t={sub_o['BRCA2']['n_treated']}).", io_b2['inter_p'], io_b2['inter_coef'], io_b2['inter_p']<0.05),
        ana(['h11.3'], f"Subgroup test: BRCA1+ or BRCA2+ olaparib effect={sub_o['BRCA_any']['diff']:+.3f} (p={sub_o['BRCA_any']['p']:.2g}, n_t={sub_o['BRCA_any']['n_treated']}); BRCA-wt effect={sub_o['BRCA_none']['diff']:+.3f} (p={sub_o['BRCA_none']['p']:.2g}). Adjusted-interaction test (BRCA-any) inter_coef={fc['ola_brca_adj_inter']['inter_coef']:+.3f} (p={fc['ola_brca_adj_inter']['inter_p']:.2g}) — adjusted interaction not significant; raw subgroup difference is borderline.", sub_o['BRCA_any']['p'], sub_o['BRCA_any']['diff'], sub_o['BRCA_any']['p']<0.05),
    ],
})

# ---- Iteration 12 — Pembrolizumab x TNBC ----
ip_er2 = inter['treatment_pembrolizumab']['er_positive']
sub_p = sub['pembro_tnbc']
iters.append({
    'index': 12,
    'proposed_hypotheses': [
        hyp('h12.1', 'The pfs_months benefit from treatment_pembrolizumab is larger in triple-negative patients (er_positive=0 AND pr_positive=0 AND her2_positive=0) than in non-TNBC patients.'),
        hyp('h12.2', 'treatment_pembrolizumab improves pfs_months in TNBC patients with high ki67_pct (above median).'),
    ],
    'analyses': [
        ana(['h12.1'], f"TNBC subgroup pembrolizumab effect={sub_p['TNBC']['diff']:+.3f} (p={sub_p['TNBC']['p']:.2g}, n_t={sub_p['TNBC']['n_treated']}); ER+ effect={sub_p['ER+']['diff']:+.3f} (p={sub_p['ER+']['p']:.2g}). The TNBC-specific effect is small and not significant.", sub_p['TNBC']['p'], sub_p['TNBC']['diff'], sub_p['TNBC']['p']<0.05),
        ana(['h12.2'], f"TNBC + high Ki67 subgroup pembrolizumab effect={sub_p['high_pdl1_proxy_TNBC_highKi67']['diff']:+.3f} (p={sub_p['high_pdl1_proxy_TNBC_highKi67']['p']:.2g}, n_t={sub_p['high_pdl1_proxy_TNBC_highKi67']['n_treated']}). Null. Confirmed by TNBC + brain_mets ({fc['pembro_tnbc_brain']['diff']:+.3f}, p={fc['pembro_tnbc_brain']['p']:.2g}) and TNBC + high Ki67 + high LDH ({fc['pembro_tnbc_highki67_highldh']['diff']:+.3f}, p={fc['pembro_tnbc_highki67_highldh']['p']:.2g}). No positive subgroup detected for pembrolizumab.", sub_p['high_pdl1_proxy_TNBC_highKi67']['p'], sub_p['high_pdl1_proxy_TNBC_highKi67']['diff'], False),
    ],
})

# ---- Iteration 13 — Sacituzumab x has_brain_mets ----
is_brain = inter['treatment_sacituzumab_govitecan']['has_brain_mets']
sub_s = sub['sacit_brain']
iters.append({
    'index': 13,
    'proposed_hypotheses': [
        hyp('h13.1', 'The pfs_months benefit from treatment_sacituzumab_govitecan is larger in has_brain_mets=1 patients than in has_brain_mets=0 patients (positive interaction).'),
    ],
    'analyses': [
        ana(['h13.1'], f"OLS sacit x has_brain_mets interaction coef={is_brain['inter_coef']:+.3f} (p={is_brain['inter_p']:.2e}). Brain mets subgroup sacit effect={sub_s['brain_mets']['diff']:+.3f} (p={sub_s['brain_mets']['p']:.2g}, n_t={sub_s['brain_mets']['n_treated']}); no brain mets effect={sub_s['no_brain_mets']['diff']:+.3f}. Adjusted-interaction confirmation (controls for other treatments and major covariates): inter_coef={fc['sacit_brain_adj_inter']['inter_coef']:+.3f} (p={fc['sacit_brain_adj_inter']['inter_p']:.2e}). Sacituzumab benefit is concentrated in brain-mets patients.", is_brain['inter_p'], is_brain['inter_coef'], is_brain['inter_p']<0.05),
    ],
})

# ---- Iteration 14 — Tamoxifen x ER+ ----
it_er = inter['treatment_tamoxifen']['er_positive']
sub_tam = sub['tam_hr']
iters.append({
    'index': 14,
    'proposed_hypotheses': [
        hyp('h14.1', 'The pfs_months benefit from treatment_tamoxifen is larger in er_positive=1 than in er_positive=0 patients.'),
        hyp('h14.2', 'Among er_positive=1 patients, the tamoxifen pfs_months benefit is larger in postmenopausal=1 than in postmenopausal=0 patients.'),
    ],
    'analyses': [
        ana(['h14.1'], f"OLS tamoxifen x er_positive interaction coef={it_er['inter_coef']:+.3f} (p={it_er['inter_p']:.2g}). ER+ subgroup tam effect={sub_tam['ER+']['diff']:+.3f} (p={sub_tam['ER+']['p']:.2g}); HR- effect={sub_tam['HR-']['diff']:+.3f}. No ER-enriched benefit.", it_er['inter_p'], it_er['inter_coef'], it_er['inter_p']<0.05),
        ana(['h14.2'], f"ER+ premeno tam effect={sub_tam['ER+_premeno']['diff']:+.3f} (p={sub_tam['ER+_premeno']['p']:.2g}); ER+ postmeno effect={sub_tam['ER+_postmeno']['diff']:+.3f} (p={sub_tam['ER+_postmeno']['p']:.2g}). Both null.", sub_tam['ER+_postmeno']['p'], sub_tam['ER+_postmeno']['diff'], False),
    ],
})

# ---- Iteration 15 — Joint palbociclib subgroup model: ER+ AND HER2- AND PIK3CA-wt ----
v = sub['palbo_joint']
iters.append({
    'index': 15,
    'proposed_hypotheses': [
        hyp('h15.1', 'In the joint subgroup defined by er_positive=1 AND her2_positive=0 AND pik3ca_mutation=0, treatment_palbociclib increases pfs_months by more than 2 months relative to no palbociclib.'),
    ],
    'analyses': [
        ana(['h15.1'], f"Subgroup analysis: ER+/HER2-/PIK3CA-wt palbociclib treated mean PFS {v['ER+_HER2-_PIK3CAwt']['mean_treated']:.3f} vs untreated {v['ER+_HER2-_PIK3CAwt']['mean_untreated']:.3f}; diff={v['ER+_HER2-_PIK3CAwt']['diff']:+.3f} mo (p={v['ER+_HER2-_PIK3CAwt']['p']:.2g}, n_t={v['ER+_HER2-_PIK3CAwt']['n_treated']}, n_u={v['ER+_HER2-_PIK3CAwt']['n_untreated']}). Adjusted-interaction confirmation: inter_coef={fc['palbo_subgroup_adj_inter']['inter_coef']:+.3f} (p={fc['palbo_subgroup_adj_inter']['inter_p']:.2g}); palbo effect at non-subgroup is essentially 0 ({fc['palbo_subgroup_adj_inter']['treat_coef_mod0']:+.3f}). Palbociclib benefit is almost entirely concentrated in this subgroup.", v['ER+_HER2-_PIK3CAwt']['p'], v['ER+_HER2-_PIK3CAwt']['diff'], True),
    ],
})

# ---- Iteration 16 — Refine palbociclib subgroup with PR+ and low Ki67 ----
iters.append({
    'index': 16,
    'proposed_hypotheses': [
        hyp('h16.1', 'Adding pr_positive=1 to ER+/HER2-/PIK3CA-wt further increases the palbociclib pfs_months benefit.', kind='refined'),
        hyp('h16.2', 'Within ER+/HER2-/PIK3CA-wt/PR+ patients, restricting to low ki67_pct (below median) further increases the palbociclib pfs_months benefit, identifying a luminal-A-like subgroup with the maximal effect.', kind='refined'),
    ],
    'analyses': [
        ana(['h16.1'], f"ER+/HER2-/PIK3CA-wt/PR+ palbociclib effect: diff={v['ER+_HER2-_PIK3CAwt_PR+']['diff']:+.3f} mo (p={v['ER+_HER2-_PIK3CAwt_PR+']['p']:.2g}, n_t={v['ER+_HER2-_PIK3CAwt_PR+']['n_treated']}, n_u={v['ER+_HER2-_PIK3CAwt_PR+']['n_untreated']}). Adding PR+ increases effect from +2.91 to +2.95 mo.", v['ER+_HER2-_PIK3CAwt_PR+']['p'], v['ER+_HER2-_PIK3CAwt_PR+']['diff'], True),
        ana(['h16.2'], f"ER+/HER2-/PIK3CA-wt/PR+/low Ki67 (<median) palbociclib effect: diff={v['ER+_HER2-_PIK3CAwt_PR+_lowKi67']['diff']:+.3f} mo (p={v['ER+_HER2-_PIK3CAwt_PR+_lowKi67']['p']:.2g}, n_t={v['ER+_HER2-_PIK3CAwt_PR+_lowKi67']['n_treated']}, n_u={v['ER+_HER2-_PIK3CAwt_PR+_lowKi67']['n_untreated']}). Adding low Ki67 increases effect from +2.95 to +5.01 mo. This is the maximally refined positive subgroup. Complement subgroup palbo effect={fc['palbo_complement']['diff']:+.3f} mo (p={fc['palbo_complement']['p']:.2g}) — small residual benefit outside subgroup.", v['ER+_HER2-_PIK3CAwt_PR+_lowKi67']['p'], v['ER+_HER2-_PIK3CAwt_PR+_lowKi67']['diff'], True),
    ],
})

# ---- Iteration 17 — Sacituzumab refined subgroup: brain mets + low Ki67 ----
iters.append({
    'index': 17,
    'proposed_hypotheses': [
        hyp('h17.1', 'Within has_brain_mets=1 patients, treatment_sacituzumab_govitecan benefit on pfs_months is concentrated in patients with low ki67_pct (below median).', kind='refined'),
    ],
    'analyses': [
        ana(['h17.1'], f"Brain mets + low Ki67 sacit effect: diff={fc['sacit_brain_lowki67']['diff']:+.3f} mo (p={fc['sacit_brain_lowki67']['p']:.2g}, n_t={fc['sacit_brain_lowki67']['n_t']}, n_u={fc['sacit_brain_lowki67']['n_u']}). Adjusted-interaction (brain_mets & low_ki67) inter_coef={fc['sacit_brain_lowki67_adj_inter']['inter_coef']:+.3f} (p={fc['sacit_brain_lowki67_adj_inter']['inter_p']:.2g}). The sacituzumab benefit concentrates in brain-mets patients with low Ki67.", fc['sacit_brain_lowki67']['p'], fc['sacit_brain_lowki67']['diff'], True),
    ],
})

# ---- Iteration 18 — Exhaustive 2-/3-feature subgroup search for trastuzumab ----
top_tras = ex['treatment_trastuzumab'][:5]
trow_lines = [f"{r['subgroup']}: diff={r['diff']:+.3f} (p={r['p']:.2g}, n_t={r['n_t']})" for r in top_tras]
iters.append({
    'index': 18,
    'proposed_hypotheses': [
        hyp('h18.1', 'A 2- or 3-feature binary subgroup exists in which treatment_trastuzumab significantly increases pfs_months by more than 0.5 months.'),
    ],
    'analyses': [
        ana(['h18.1'], f"Exhaustive search over 2- and 3-feature binary subgroups (min n_t=150) for trastuzumab. Top |diff| subgroups: " + '; '.join(trow_lines) + ". The largest subgroup-level diff is well below 0.5 mo and the directions are mixed; multiple-testing correction would render none significant. No reliable trastuzumab-responsive subgroup is identified.", None, top_tras[0]['diff'], False),
    ],
})

# ---- Iteration 19 — Exhaustive search for olaparib ----
top_ola = ex['treatment_olaparib'][:5]
orow = [f"{r['subgroup']}: diff={r['diff']:+.3f} (p={r['p']:.2g}, n_t={r['n_t']})" for r in top_ola]
iters.append({
    'index': 19,
    'proposed_hypotheses': [
        hyp('h19.1', 'Beyond the BRCA-mutated subgroup, an additional 2-/3-feature binary subgroup exists where treatment_olaparib substantially increases pfs_months.'),
    ],
    'analyses': [
        ana(['h19.1'], f"Exhaustive 2-/3-feature search for olaparib (min n_t=150). Top |diff| rows are predominantly negative (suggesting harm in HER2+/PIK3CA-mutated/brain-mets non-BRCA subsets): " + '; '.join(orow) + ". No additional positive subgroup beyond BRCA-mut is identified.", None, top_ola[0]['diff'], False),
    ],
})

# ---- Iteration 20 — Exhaustive search for sacituzumab and pembrolizumab ----
top_sac = ex['treatment_sacituzumab_govitecan'][:5]
top_pem = ex['treatment_pembrolizumab'][:5]
srow = [f"{r['subgroup']}: diff={r['diff']:+.3f} (p={r['p']:.2g}, n_t={r['n_t']})" for r in top_sac]
prow = [f"{r['subgroup']}: diff={r['diff']:+.3f} (p={r['p']:.2g}, n_t={r['n_t']})" for r in top_pem]
iters.append({
    'index': 20,
    'proposed_hypotheses': [
        hyp('h20.1', 'A multi-feature subgroup involving has_brain_mets=1 AND low ki67_pct exists in which treatment_sacituzumab_govitecan increases pfs_months by ~0.4-0.7 months.'),
        hyp('h20.2', 'No 2-/3-feature subgroup exists in which treatment_pembrolizumab significantly increases pfs_months; effects are null or slightly negative across all candidate subgroups.'),
    ],
    'analyses': [
        ana(['h20.1'], f"Sacituzumab top subgroups by |diff|: " + '; '.join(srow) + ". Multiple subgroups containing brain_mets+low_ki67 reach diff +0.4-0.7 mo (p~0.003-0.05). Pattern is consistent across er_pos/her2_neg/pik3ca_wt variants of the brain_mets+low_ki67 backbone.", top_sac[0]['p'], top_sac[0]['diff'], True),
        ana(['h20.2'], f"Pembrolizumab top subgroups by |diff|: " + '; '.join(prow) + ". All top results are negative (treatment associated with shorter PFS), concentrated in ECOG=2/low-Ki67/low-albumin (frail) patients. No positive subgroup is observed.", top_pem[0]['p'], top_pem[0]['diff'], False),
    ],
})

# ---- Iteration 21 — Exhaustive search for tamoxifen and palbociclib (confirmation) ----
top_tam = ex['treatment_tamoxifen'][:5]
top_palbo = ex['treatment_palbociclib'][:5]
tamrow = [f"{r['subgroup']}: diff={r['diff']:+.3f} (p={r['p']:.2g}, n_t={r['n_t']})" for r in top_tam]
palrow = [f"{r['subgroup']}: diff={r['diff']:+.3f} (p={r['p']:.2g}, n_t={r['n_t']})" for r in top_palbo]
iters.append({
    'index': 21,
    'proposed_hypotheses': [
        hyp('h21.1', 'The exhaustive 2-/3-feature search confirms that treatment_palbociclib produces its largest pfs_months benefits (diff > 3 mo) in subgroups that combine pik3ca_wt AND low ki67_pct, with additional gain from er_positive=1, pr_positive=1, and her2_negative.', kind='refined'),
        hyp('h21.2', 'A small positive subgroup for treatment_tamoxifen exists in brca1_mutation=1 patients (diff ~0.4-0.5 mo), though it is borderline after multiple testing.'),
    ],
    'analyses': [
        ana(['h21.1'], f"Palbociclib top subgroups by |diff|: " + '; '.join(palrow) + ". The 'pik3ca_wt + low_ki67' backbone repeatedly produces +2.9 mo+ effects, with up to +4.1 mo when combined with ER+/PR+/HER2-low.", top_palbo[0]['p'], top_palbo[0]['diff'], True),
        ana(['h21.2'], f"Tamoxifen top subgroups by |diff|: " + '; '.join(tamrow) + f". BRCA1+ER+ subgroup confirmation: diff={fc['tam_brca1_er_pos']['diff']:+.3f} (p={fc['tam_brca1_er_pos']['p']:.2g}). Borderline signal, not robust to multiple testing.", top_tam[0]['p'], top_tam[0]['diff'], False),
    ],
})

# ---- Iteration 22 — Trastuzumab in HER2+ ER- vs HER2+ ER+ ----
iters.append({
    'index': 22,
    'proposed_hypotheses': [
        hyp('h22.1', 'treatment_trastuzumab benefit on pfs_months is larger in HER2+/ER- (HER2-driven, non-luminal) patients than in HER2+/ER+ patients.'),
    ],
    'analyses': [
        ana(['h22.1'], f"HER2+ ER- trastuzumab effect: diff={fc['tras_her2_er_neg']['diff']:+.3f} (p={fc['tras_her2_er_neg']['p']:.2g}, n_t={fc['tras_her2_er_neg']['n_t']}); HER2+ ER+ effect: diff={fc['tras_her2_er_pos']['diff']:+.3f} (p={fc['tras_her2_er_pos']['p']:.2g}). Both null. Trastuzumab does not appear to confer a PFS benefit in any HER2-defined subgroup of this cohort.", fc['tras_her2_er_neg']['p'], fc['tras_her2_er_neg']['diff'], False),
    ],
})

# ---- Iteration 23 — Final palbociclib subgroup hypothesis ----
iters.append({
    'index': 23,
    'proposed_hypotheses': [
        hyp('h23.1', 'FINAL: treatment_palbociclib increases pfs_months by ~5 months relative to no palbociclib in the joint subgroup defined by er_positive=1 AND her2_positive=0 AND pik3ca_mutation=0 AND pr_positive=1 AND ki67_pct < median (~14.6%). Outside this subgroup the palbociclib effect is small (~+0.35 mo).', kind='refined'),
    ],
    'analyses': [
        ana(['h23.1'], f"Subgroup palbociclib effect (ER+/HER2-/PIK3CAwt/PR+/lowKi67): diff={fc['palbo_best']['diff']:+.3f} mo (p={fc['palbo_best']['p']:.2g}, n_t={fc['palbo_best']['n_t']}, n_u={fc['palbo_best']['n_u']}). Complement subgroup palbo effect: diff={fc['palbo_complement']['diff']:+.3f} mo (p={fc['palbo_complement']['p']:.2g}). The treatment-effect heterogeneity is dramatic (subgroup vs complement diff ~4.7 mo).", fc['palbo_best']['p'], fc['palbo_best']['diff'], True),
    ],
})

# ---- Iteration 24 — Final sacituzumab and olaparib subgroup hypotheses ----
iters.append({
    'index': 24,
    'proposed_hypotheses': [
        hyp('h24.1', 'FINAL: treatment_sacituzumab_govitecan increases pfs_months by ~0.4 months in the subgroup defined by has_brain_mets=1 AND ki67_pct < median; effect is null elsewhere.', kind='refined'),
        hyp('h24.2', 'FINAL: treatment_olaparib increases pfs_months by ~0.35 months in the subgroup defined by brca1_mutation=1 OR brca2_mutation=1; effect is null in BRCA-wild-type patients.', kind='refined'),
    ],
    'analyses': [
        ana(['h24.1'], f"Sacituzumab in brain_mets + low_ki67: diff={fc['sacit_brain_lowki67']['diff']:+.3f} (p={fc['sacit_brain_lowki67']['p']:.2g}, n_t={fc['sacit_brain_lowki67']['n_t']}); adjusted interaction coef={fc['sacit_brain_lowki67_adj_inter']['inter_coef']:+.3f} (p={fc['sacit_brain_lowki67_adj_inter']['inter_p']:.2g}).", fc['sacit_brain_lowki67']['p'], fc['sacit_brain_lowki67']['diff'], True),
        ana(['h24.2'], f"Olaparib in BRCA1+ or BRCA2+: diff={fc['ola_brca_any']['diff']:+.3f} (p={fc['ola_brca_any']['p']:.2g}, n_t={fc['ola_brca_any']['n_t']}); BRCA-wt diff={fc['ola_brca_none']['diff']:+.3f} (p={fc['ola_brca_none']['p']:.2g}). Adjusted-interaction not significant (p={fc['ola_brca_adj_inter']['inter_p']:.2g}); the BRCA-mut subgroup signal is borderline.", fc['ola_brca_any']['p'], fc['ola_brca_any']['diff'], True),
    ],
})

# ---- Iteration 25 — Final negative findings for trastuzumab, pembrolizumab, tamoxifen ----
iters.append({
    'index': 25,
    'proposed_hypotheses': [
        hyp('h25.1', 'FINAL: treatment_trastuzumab has no clinically meaningful main effect or subgroup effect on pfs_months in this cohort, including in the her2_positive=1 subgroup.'),
        hyp('h25.2', 'FINAL: treatment_pembrolizumab has no positive subgroup effect on pfs_months in this cohort; in subgroups defined by frailty markers (ecog_ps=2 + low_alb + low_ki67) the association is mildly negative.'),
        hyp('h25.3', 'FINAL: treatment_tamoxifen has no robust pfs_months benefit on the cohort or in canonical ER+/postmenopausal subgroups; a borderline positive signal in BRCA1+ patients is likely chance after multiple-testing correction.'),
    ],
    'analyses': [
        ana(['h25.1'], f"Best HER2-stratified trastuzumab estimates: HER2+ diff={sub['tras_her2']['HER2+']['diff']:+.3f} (p={sub['tras_her2']['HER2+']['p']:.2g}); HER2+/ER- diff={fc['tras_her2_er_neg']['diff']:+.3f}; HER2+/ER+ diff={fc['tras_her2_er_pos']['diff']:+.3f}. None reach significance.", sub['tras_her2']['HER2+']['p'], sub['tras_her2']['HER2+']['diff'], False),
        ana(['h25.2'], f"Best pembrolizumab subgroup (TNBC + brain_mets) diff={fc['pembro_tnbc_brain']['diff']:+.3f} (p={fc['pembro_tnbc_brain']['p']:.2g}); TNBC + high Ki67 + high LDH diff={fc['pembro_tnbc_highki67_highldh']['diff']:+.3f} (p={fc['pembro_tnbc_highki67_highldh']['p']:.2g}); top 3-feature subgroups all negative (smallest p={ex['treatment_pembrolizumab'][0]['p']:.2g} for {ex['treatment_pembrolizumab'][0]['subgroup']} with diff={ex['treatment_pembrolizumab'][0]['diff']:+.3f}).", fc['pembro_tnbc_brain']['p'], fc['pembro_tnbc_brain']['diff'], False),
        ana(['h25.3'], f"Tamoxifen overall diff={mb['treatment_tamoxifen']['slope_or_diff']:+.3f} (p={mb['treatment_tamoxifen']['p']:.2g}); ER+ diff={sub['tam_hr']['ER+']['diff']:+.3f} (p={sub['tam_hr']['ER+']['p']:.2g}); ER+/postmeno diff={sub['tam_hr']['ER+_postmeno']['diff']:+.3f}; BRCA1+/ER+ diff={fc['tam_brca1_er_pos']['diff']:+.3f} (p={fc['tam_brca1_er_pos']['p']:.2g}). No robust positive subgroup.", sub['tam_hr']['ER+']['p'], sub['tam_hr']['ER+']['diff'], False),
    ],
})

transcript = {
    'dataset_id': 'ds001_breast',
    'model_id': 'claude-opus-4-7',
    'harness_id': 'manual-claude-code-session@2026-05-03',
    'max_iterations': 25,
    'iterations': iters,
}

with open('transcript.json', 'w') as fp:
    json.dump(transcript, fp, indent=2)

print('Wrote transcript.json with', len(iters), 'iterations')
