"""Construct transcript.json from results.json"""
import json

r = json.load(open('results.json'))


def fmt_or(x):
    return f"OR={x['odds_ratio']:.3f}, p={x['p_value']:.3g}"


def fmt_int(x):
    return f"interaction OR={x['interaction_odds_ratio']:.3f}, p={x['p_value']:.3g}"


iterations = []


# -------- Iteration 1: Treatment main effects --------
i = 1
hyps = [
    {"id": "h1.1", "text": "Patients receiving treatment_pembrolizumab have a higher mean objective_response than patients not receiving treatment_pembrolizumab.", "kind": "novel"},
    {"id": "h1.2", "text": "Patients receiving treatment_sotorasib have a higher mean objective_response than patients not receiving treatment_sotorasib.", "kind": "novel"},
    {"id": "h1.3", "text": "Patients receiving treatment_olaparib have a higher mean objective_response than patients not receiving treatment_olaparib.", "kind": "novel"},
    {"id": "h1.4", "text": "Patients receiving treatment_osimertinib have a higher mean objective_response than patients not receiving treatment_osimertinib.", "kind": "novel"},
]
analyses = []
for tx, hid in [('treatment_pembrolizumab', 'h1.1'), ('treatment_sotorasib', 'h1.2'),
                ('treatment_olaparib', 'h1.3'), ('treatment_osimertinib', 'h1.4')]:
    d = r[f'tx_main_{tx}']
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"sm.Logit(df['objective_response'], sm.add_constant(df[['{tx}']])).fit()",
        "result_summary": f"Response rate {d['rate_treated']:.3f} on {tx} vs {d['rate_untreated']:.3f} off; logit {fmt_or(d)}.",
        "p_value": d['p_value'],
        "effect_estimate": d['rate_diff'],
        "significant": d['p_value'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 2: Clinical / demographic prognostic factors --------
i = 2
hyps = [
    {"id": "h2.1", "text": "Higher ecog_ps is associated with lower probability of objective_response.", "kind": "novel"},
    {"id": "h2.2", "text": "Patients with stage_iv = 1 have lower probability of objective_response than those with stage_iv = 0.", "kind": "novel"},
    {"id": "h2.3", "text": "Patients with has_brain_mets = 1 have lower probability of objective_response than those without.", "kind": "novel"},
    {"id": "h2.4", "text": "Older age_years is associated with lower probability of objective_response.", "kind": "novel"},
    {"id": "h2.5", "text": "Female patients (sex_female=1) have a different probability of objective_response than male patients.", "kind": "novel"},
]
analyses = []
for v, hid in [('ecog_ps', 'h2.1'), ('stage_iv', 'h2.2'), ('has_brain_mets', 'h2.3'),
                ('age_years', 'h2.4'), ('sex_female', 'h2.5')]:
    d = r[f'main_{v}']
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"sm.Logit(df['objective_response'], sm.add_constant(df[['{v}']])).fit()",
        "result_summary": f"Logistic regression of objective_response on {v}: {fmt_or(d)}.",
        "p_value": d['p_value'],
        "effect_estimate": d['coef_log_odds'],
        "significant": d['p_value'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 3: Biomarker main effects --------
i = 3
hyps = [
    {"id": "h3.1", "text": "Patients with tmb_high = 1 have a higher probability of objective_response than tmb_high = 0.", "kind": "novel"},
    {"id": "h3.2", "text": "Higher pdl1_tps is associated with higher probability of objective_response.", "kind": "novel"},
    {"id": "h3.3", "text": "egfr_mutation, kras_g12c, alk_fusion, brca2_mutation, stk11_mutation, keap1_mutation, and tp53_mutation each have a non-zero main effect on objective_response.", "kind": "novel"},
]
analyses = []
d = r['main_tmb_high']
analyses.append({
    "hypothesis_ids": ["h3.1"],
    "code": "logit(objective_response ~ tmb_high)",
    "result_summary": f"Response 0.182 in tmb_high=1 vs 0.164 in tmb_high=0; {fmt_or(d)}.",
    "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
})
d = r['main_pdl1_tps']
analyses.append({
    "hypothesis_ids": ["h3.2"],
    "code": "logit(objective_response ~ pdl1_tps)",
    "result_summary": f"pdl1_tps continuous {fmt_or(d)} per unit (i.e. 0->1).",
    "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
})
for v in ['egfr_mutation', 'kras_g12c', 'alk_fusion', 'brca2_mutation', 'stk11_mutation', 'keap1_mutation', 'tp53_mutation']:
    d = r[f'main_{v}']
    analyses.append({
        "hypothesis_ids": ["h3.3"],
        "code": f"logit(objective_response ~ {v})",
        "result_summary": f"{v}: {fmt_or(d)}; rates pos {d['rate_pos']:.3f} vs neg {d['rate_neg']:.3f}.",
        "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 4: EGFR x osimertinib interaction --------
i = 4
hyps = [
    {"id": "h4.1", "text": "Among patients with egfr_mutation = 1, treatment_osimertinib increases objective_response more than it does among egfr_mutation = 0 patients (positive interaction on the log-odds scale).", "kind": "novel"},
]
d = r['inter_egfr_osi']
rr = d['response_rates']
analyses = [{
    "hypothesis_ids": ["h4.1"],
    "code": "logit(objective_response ~ treatment_osimertinib*egfr_mutation)",
    "result_summary": (f"Cell response rates: osi-/egfr- {rr['t0_b0']:.3f}, osi-/egfr+ {rr['t0_b1']:.3f}, "
                      f"osi+/egfr- {rr['t1_b0']:.3f}, osi+/egfr+ {rr['t1_b1']:.3f}; {fmt_int(d)}. "
                      "No detectable interaction; osimertinib does not selectively increase response in EGFR-mutant patients."),
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
}]
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 5: KRAS G12C x sotorasib --------
i = 5
hyps = [
    {"id": "h5.1", "text": "Among patients with kras_g12c = 1, treatment_sotorasib increases objective_response more than among kras_g12c = 0 patients (positive interaction).", "kind": "novel"},
]
d = r['inter_krasg12c_soto']
rr = d['response_rates']
analyses = [{
    "hypothesis_ids": ["h5.1"],
    "code": "logit(objective_response ~ treatment_sotorasib*kras_g12c)",
    "result_summary": (f"Cell rates: soto-/kras- {rr['t0_b0']:.3f}, soto-/kras+ {rr['t0_b1']:.3f}, "
                      f"soto+/kras- {rr['t1_b0']:.3f}, soto+/kras+ {rr['t1_b1']:.3f}; {fmt_int(d)}. "
                      "No statistical interaction detected."),
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
}]
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 6: BRCA2 x olaparib --------
i = 6
hyps = [
    {"id": "h6.1", "text": "Among patients with brca2_mutation = 1, treatment_olaparib increases objective_response more than among brca2_mutation = 0 patients (positive interaction).", "kind": "novel"},
]
d = r['inter_brca2_olap']
rr = d['response_rates']
analyses = [{
    "hypothesis_ids": ["h6.1"],
    "code": "logit(objective_response ~ treatment_olaparib*brca2_mutation)",
    "result_summary": (f"Cell rates: olap-/brca- {rr['t0_b0']:.3f}, olap-/brca+ {rr['t0_b1']:.3f}, "
                      f"olap+/brca- {rr['t1_b0']:.3f}, olap+/brca+ {rr['t1_b1']:.3f}; {fmt_int(d)}. "
                      "Point estimate is in the wrong direction (negative) and not significant."),
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
}]
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 7: PD-L1 x pembrolizumab --------
i = 7
hyps = [
    {"id": "h7.1", "text": "Patients with pdl1_tps >= 0.5 (high PD-L1) derive a larger increase in objective_response from treatment_pembrolizumab than patients with PD-L1 < 0.5 (positive treatment_pembrolizumab x pdl1_high interaction).", "kind": "novel"},
    {"id": "h7.2", "text": "On the continuous pdl1_tps scale, the treatment_pembrolizumab x pdl1_tps interaction on objective_response is positive.", "kind": "novel"},
]
d = r['inter_pdl1high_pembro']
rr = d['response_rates']
analyses = [{
    "hypothesis_ids": ["h7.1"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*pdl1_high)",
    "result_summary": (f"pembro-/pdl1lo {rr['t0_b0']:.3f}, pembro-/pdl1hi {rr['t0_b1']:.3f}, "
                      f"pembro+/pdl1lo {rr['t1_b0']:.3f}, pembro+/pdl1hi {rr['t1_b1']:.3f}; "
                      f"{fmt_int(d)}. Strongly positive interaction: pembro lifts response from ~16% to ~21% in PD-L1 high but not in PD-L1 low."),
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
}]
d2 = r['inter_pdl1cont_pembro']
analyses.append({
    "hypothesis_ids": ["h7.2"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*pdl1_tps)",
    "result_summary": f"Continuous pdl1_tps x pembrolizumab interaction beta={d2['interaction_coef_log_odds']:.3f}, p={d2['p_value']:.3g}.",
    "p_value": d2['p_value'], "effect_estimate": d2['interaction_coef_log_odds'], "significant": d2['p_value'] < 0.05,
})
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 8: TMB x pembrolizumab --------
i = 8
hyps = [
    {"id": "h8.1", "text": "Patients with tmb_high = 1 derive a larger increase in objective_response from treatment_pembrolizumab than patients with tmb_high = 0 (positive interaction).", "kind": "novel"},
]
d = r['inter_tmb_pembro']
rr = d['response_rates']
analyses = [{
    "hypothesis_ids": ["h8.1"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*tmb_high)",
    "result_summary": (f"pembro-/tmb- {rr['t0_b0']:.3f}, pembro-/tmb+ {rr['t0_b1']:.3f}, "
                      f"pembro+/tmb- {rr['t1_b0']:.3f}, pembro+/tmb+ {rr['t1_b1']:.3f}; {fmt_int(d)}. "
                      "Positive interaction: pembrolizumab benefit is concentrated in TMB-high tumors."),
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
}]
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 9: STK11 / KEAP1 x pembrolizumab (resistance hypothesis) --------
i = 9
hyps = [
    {"id": "h9.1", "text": "Patients with stk11_mutation = 1 derive less benefit (smaller increase in objective_response) from treatment_pembrolizumab than stk11_mutation = 0 patients (negative interaction).", "kind": "novel"},
    {"id": "h9.2", "text": "Patients with keap1_mutation = 1 derive less benefit from treatment_pembrolizumab than keap1_mutation = 0 patients (negative interaction).", "kind": "novel"},
]
d = r['inter_stk11_pembro']
rr = d['response_rates']
analyses = [{
    "hypothesis_ids": ["h9.1"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*stk11_mutation)",
    "result_summary": (f"pembro-/stk- {rr['t0_b0']:.3f}, pembro-/stk+ {rr['t0_b1']:.3f}, "
                      f"pembro+/stk- {rr['t1_b0']:.3f}, pembro+/stk+ {rr['t1_b1']:.3f}; {fmt_int(d)}. "
                      "Negative interaction: STK11 mutation abrogates pembrolizumab response."),
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
}]
d2 = r['inter_keap1_pembro']
rr2 = d2['response_rates']
analyses.append({
    "hypothesis_ids": ["h9.2"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*keap1_mutation)",
    "result_summary": (f"pembro-/keap- {rr2['t0_b0']:.3f}, pembro-/keap+ {rr2['t0_b1']:.3f}, "
                      f"pembro+/keap- {rr2['t1_b0']:.3f}, pembro+/keap+ {rr2['t1_b1']:.3f}; {fmt_int(d2)}. "
                      "No detectable KEAP1 effect on pembrolizumab response."),
    "p_value": d2['p_value'], "effect_estimate": d2['interaction_coef_log_odds'], "significant": d2['p_value'] < 0.05,
})
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 10: ALK fusion --------
i = 10
hyps = [
    {"id": "h10.1", "text": "Patients with alk_fusion = 1 have a larger increase in objective_response from treatment_osimertinib than alk_fusion = 0 patients (positive interaction).", "kind": "novel"},
    {"id": "h10.2", "text": "Patients with alk_fusion = 1 derive less benefit (smaller increase in objective_response) from treatment_pembrolizumab than alk_fusion = 0 patients (negative interaction, oncogene-driven biology).", "kind": "novel"},
]
d = r['inter_alk_osi']; rr = d['response_rates']
analyses = [{
    "hypothesis_ids": ["h10.1"],
    "code": "logit(objective_response ~ treatment_osimertinib*alk_fusion)",
    "result_summary": (f"osi-/alk- {rr['t0_b0']:.3f}, osi-/alk+ {rr['t0_b1']:.3f}, osi+/alk- {rr['t1_b0']:.3f}, osi+/alk+ {rr['t1_b1']:.3f}; {fmt_int(d)}. "
                      "No osimertinib advantage in ALK+ tumors (as expected — osimertinib targets EGFR, not ALK)."),
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
}]
d2 = r['inter_alk_pembro']; rr2 = d2['response_rates']
analyses.append({
    "hypothesis_ids": ["h10.2"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*alk_fusion)",
    "result_summary": f"pembro x alk_fusion interaction {fmt_int(d2)}; null result.",
    "p_value": d2['p_value'], "effect_estimate": d2['interaction_coef_log_odds'], "significant": d2['p_value'] < 0.05,
})
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 11: Inflammatory / nutritional labs --------
i = 11
hyps = [
    {"id": "h11.1", "text": "Higher albumin_g_dl is associated with higher probability of objective_response.", "kind": "novel"},
    {"id": "h11.2", "text": "Higher ldh_u_l is associated with lower probability of objective_response.", "kind": "novel"},
    {"id": "h11.3", "text": "Higher nlr is associated with lower probability of objective_response.", "kind": "novel"},
    {"id": "h11.4", "text": "Higher crp_mg_l is associated with lower probability of objective_response.", "kind": "novel"},
    {"id": "h11.5", "text": "Higher weight_loss_pct_6mo is associated with lower probability of objective_response.", "kind": "novel"},
]
analyses = []
for v, hid in [('albumin_g_dl', 'h11.1'), ('ldh_u_l', 'h11.2'), ('nlr', 'h11.3'),
                ('crp_mg_l', 'h11.4'), ('weight_loss_pct_6mo', 'h11.5')]:
    d = r[f'main_{v}']
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit(objective_response ~ {v})",
        "result_summary": f"{v}: log-odds beta={d['coef_log_odds']:+.4f}, {fmt_or(d)}.",
        "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 12: Metastasis sites --------
i = 12
hyps = [
    {"id": "h12.1", "text": "Patients with liver_mets = 1 have lower objective_response than those without.", "kind": "novel"},
    {"id": "h12.2", "text": "Patients with bone_mets = 1 have lower objective_response than those without.", "kind": "novel"},
    {"id": "h12.3", "text": "Patients with adrenal_mets = 1 have lower objective_response than those without.", "kind": "novel"},
    {"id": "h12.4", "text": "Patients with pleural_effusion = 1 have lower objective_response than those without.", "kind": "novel"},
    {"id": "h12.5", "text": "Patients with pericardial_effusion = 1 have lower objective_response than those without.", "kind": "novel"},
]
analyses = []
for v, hid in [('liver_mets', 'h12.1'), ('bone_mets', 'h12.2'), ('adrenal_mets', 'h12.3'),
                ('pleural_effusion', 'h12.4'), ('pericardial_effusion', 'h12.5')]:
    d = r[f'mets_{v}']
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit(objective_response ~ {v})",
        "result_summary": f"{v}: rate pos {d['rate_pos']:.3f} vs neg {d['rate_neg']:.3f}; {fmt_or(d)}.",
        "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 13: Histology / smoking_status --------
i = 13
hyps = [
    {"id": "h13.1", "text": "Patients with squamous histology have a different objective_response than those with adenocarcinoma.", "kind": "novel"},
    {"id": "h13.2", "text": "Smoking_status (current vs former vs never) is associated with objective_response.", "kind": "novel"},
]
analyses = []
d = r['main_squamous']
analyses.append({
    "hypothesis_ids": ["h13.1"],
    "code": "logit(objective_response ~ I(histology=='squamous'))",
    "result_summary": f"squamous response {d['rate_squamous']:.3f} vs adeno {d['rate_adeno']:.3f}; {fmt_or(d)}.",
    "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
})
for s in ['current', 'former', 'never']:
    d = r[f'smoke_{s}']
    analyses.append({
        "hypothesis_ids": ["h13.2"],
        "code": f"logit(objective_response ~ I(smoking_status=='{s}'))",
        "result_summary": f"smoking_status={s}: rate {d['rate_in_group']:.3f} vs other {d['rate_other']:.3f}; {fmt_or(d)}.",
        "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 14: smoking x treatment --------
i = 14
hyps = [
    {"id": "h14.1", "text": "Higher smoking_pack_years amplifies the objective_response benefit of treatment_pembrolizumab (positive interaction).", "kind": "novel"},
    {"id": "h14.2", "text": "Never-smokers (smoking_status='never') derive a smaller objective_response benefit from treatment_pembrolizumab (negative interaction).", "kind": "novel"},
    {"id": "h14.3", "text": "Never-smokers derive a larger objective_response benefit from treatment_osimertinib than ever-smokers (positive interaction, EGFR-enriched).", "kind": "novel"},
]
d1 = r['inter_packyrs_pembro']
analyses = [{
    "hypothesis_ids": ["h14.1"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*smoking_pack_years)",
    "result_summary": f"smoking_pack_years x pembrolizumab interaction beta={d1['interaction_coef_log_odds']:+.4f}, p={d1['p_value']:.3g}; null.",
    "p_value": d1['p_value'], "effect_estimate": d1['interaction_coef_log_odds'], "significant": d1['p_value'] < 0.05,
}]
d2 = r['inter_never_pembro']
analyses.append({
    "hypothesis_ids": ["h14.2"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*I(smoking_status=='never'))",
    "result_summary": f"never-smoker x pembrolizumab interaction {fmt_int(d2)}; null.",
    "p_value": d2['p_value'], "effect_estimate": d2['interaction_coef_log_odds'], "significant": d2['p_value'] < 0.05,
})
d3 = r['inter_never_osi']
analyses.append({
    "hypothesis_ids": ["h14.3"],
    "code": "logit(objective_response ~ treatment_osimertinib*I(smoking_status=='never'))",
    "result_summary": f"never-smoker x osimertinib {fmt_int(d3)}; trend toward negative interaction (p~0.09).",
    "p_value": d3['p_value'], "effect_estimate": d3['interaction_coef_log_odds'], "significant": d3['p_value'] < 0.05,
})
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 15: Sex x treatment --------
i = 15
hyps = [
    {"id": "h15.1", "text": "Female patients derive a larger increase in objective_response from treatment_pembrolizumab than male patients (positive interaction).", "kind": "refined"},
    {"id": "h15.2", "text": "Female patients derive a different increase in objective_response from treatment_osimertinib than male patients.", "kind": "novel"},
    {"id": "h15.3", "text": "Female patients derive a different increase in objective_response from treatment_sotorasib than male patients.", "kind": "novel"},
    {"id": "h15.4", "text": "Female patients derive a different increase in objective_response from treatment_olaparib than male patients.", "kind": "novel"},
]
analyses = []
for hid, key in [("h15.1", 'inter_sex_pembro'), ("h15.2", 'inter_sex_osi'),
                  ("h15.3", 'inter_sex_soto'), ("h15.4", 'inter_sex_olap')]:
    d = r[key]; rr = d['response_rates']
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit(objective_response ~ {key.replace('inter_sex_', 'treatment_')}*sex_female)",
        "result_summary": (f"male/no {rr['t0_b0']:.3f}, female/no {rr['t0_b1']:.3f}, "
                          f"male/yes {rr['t1_b0']:.3f}, female/yes {rr['t1_b1']:.3f}; {fmt_int(d)}."),
        "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 16: Race / ethnicity --------
i = 16
hyps = [
    {"id": "h16.1", "text": "Race/ethnicity is associated with objective_response (each race indicator is tested for a non-zero effect, with white as one comparator).", "kind": "novel"},
    {"id": "h16.2", "text": "Asian patients (race_ethnicity='asian') derive a larger increase in objective_response from treatment_osimertinib than non-Asian patients (positive interaction).", "kind": "novel"},
    {"id": "h16.3", "text": "Black patients (race_ethnicity='black') derive a different increase in objective_response from treatment_pembrolizumab than non-Black patients.", "kind": "novel"},
]
analyses = []
for race in ['white', 'black', 'hispanic', 'asian', 'other']:
    d = r[f'race_{race}']
    analyses.append({
        "hypothesis_ids": ["h16.1"],
        "code": f"logit(objective_response ~ I(race_ethnicity=='{race}'))",
        "result_summary": f"race={race}: in-group {d['rate_in_group']:.3f} vs other {d['rate_other']:.3f}; {fmt_or(d)}.",
        "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
d = r['inter_asian_osi']
analyses.append({
    "hypothesis_ids": ["h16.2"],
    "code": "logit(objective_response ~ treatment_osimertinib*I(race_ethnicity=='asian'))",
    "result_summary": f"Asian x osimertinib {fmt_int(d)}; null.",
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
})
d = r['inter_black_pembro']
analyses.append({
    "hypothesis_ids": ["h16.3"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*I(race_ethnicity=='black'))",
    "result_summary": f"Black x pembrolizumab {fmt_int(d)}; null.",
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
})
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 17: Insurance / SES --------
i = 17
hyps = [
    {"id": "h17.1", "text": "Insurance type (medicare/private/medicaid/uninsured) is associated with objective_response.", "kind": "novel"},
    {"id": "h17.2", "text": "rural_residence = 1 is associated with lower objective_response than rural_residence = 0.", "kind": "novel"},
    {"id": "h17.3", "text": "Higher education_years is associated with higher objective_response.", "kind": "novel"},
]
analyses = []
for ins in ['medicare', 'private', 'medicaid', 'uninsured']:
    d = r[f'ins_{ins}']
    analyses.append({
        "hypothesis_ids": ["h17.1"],
        "code": f"logit(objective_response ~ I(insurance_type=='{ins}'))",
        "result_summary": f"insurance={ins}: in-group {d['rate_in_group']:.3f} vs other {d['rate_other']:.3f}; {fmt_or(d)}.",
        "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
for v, hid in [('rural_residence', 'h17.2'), ('education_years', 'h17.3')]:
    d = r[f'soc_{v}']
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit(objective_response ~ {v})",
        "result_summary": f"{v}: {fmt_or(d)}.",
        "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 18: Comorbidities + autoimmune x pembro --------
i = 18
hyps = [
    {"id": "h18.1", "text": "Major comorbidities (autoimmune_disease, interstitial_lung_disease_history, copd, chronic_kidney_disease, heart_failure, diabetes_mellitus, hypertension) each have a non-zero main effect on objective_response.", "kind": "novel"},
    {"id": "h18.2", "text": "Patients with autoimmune_disease = 1 derive a smaller increase in objective_response from treatment_pembrolizumab than those without (negative interaction).", "kind": "novel"},
    {"id": "h18.3", "text": "Patients with interstitial_lung_disease_history = 1 derive a smaller increase in objective_response from treatment_pembrolizumab than those without (negative interaction).", "kind": "novel"},
]
analyses = []
for v in ['autoimmune_disease', 'interstitial_lung_disease_history', 'copd',
          'chronic_kidney_disease', 'heart_failure', 'diabetes_mellitus', 'hypertension']:
    d = r[f'comorb_{v}']
    analyses.append({
        "hypothesis_ids": ["h18.1"],
        "code": f"logit(objective_response ~ {v})",
        "result_summary": f"{v}: {fmt_or(d)}.",
        "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
d = r['inter_autoimm_pembro']
analyses.append({
    "hypothesis_ids": ["h18.2"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*autoimmune_disease)",
    "result_summary": f"autoimmune x pembrolizumab {fmt_int(d)}; null.",
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
})
d = r['inter_ild_pembro']
analyses.append({
    "hypothesis_ids": ["h18.3"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*interstitial_lung_disease_history)",
    "result_summary": f"ILD x pembrolizumab {fmt_int(d)}; null.",
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
})
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 19: Prior therapies --------
i = 19
hyps = [
    {"id": "h19.1", "text": "Patients with prior_chemotherapy = 1 have lower objective_response than those without.", "kind": "novel"},
    {"id": "h19.2", "text": "Higher prior_lines_of_therapy is associated with lower objective_response.", "kind": "novel"},
    {"id": "h19.3", "text": "Patients with prior_immunotherapy = 1 derive a smaller increase in objective_response from treatment_pembrolizumab (negative interaction).", "kind": "novel"},
    {"id": "h19.4", "text": "Other prior treatments (prior_radiation, prior_surgery, prior_immunotherapy, prior_targeted_therapy) and years_since_diagnosis each have a non-zero main effect on objective_response.", "kind": "novel"},
]
analyses = []
for v, hid in [('prior_chemotherapy', 'h19.1'), ('prior_lines_of_therapy', 'h19.2')]:
    d = r[f'prior_{v}']
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit(objective_response ~ {v})",
        "result_summary": f"{v}: {fmt_or(d)}.",
        "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
d = r['inter_priorIO_pembro']
analyses.append({
    "hypothesis_ids": ["h19.3"],
    "code": "logit(objective_response ~ treatment_pembrolizumab*prior_immunotherapy)",
    "result_summary": f"prior_immunotherapy x pembrolizumab {fmt_int(d)}; positive (not negative) interaction at borderline significance.",
    "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
})
for v in ['prior_radiation', 'prior_surgery', 'prior_immunotherapy', 'prior_targeted_therapy', 'years_since_diagnosis']:
    d = r[f'prior_{v}']
    analyses.append({
        "hypothesis_ids": ["h19.4"],
        "code": f"logit(objective_response ~ {v})",
        "result_summary": f"{v}: {fmt_or(d)}.",
        "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 20: Symptoms --------
i = 20
hyps = [
    {"id": "h20.1", "text": "Higher fatigue_grade is associated with lower objective_response.", "kind": "novel"},
    {"id": "h20.2", "text": "Higher pain_nrs is associated with lower objective_response.", "kind": "novel"},
    {"id": "h20.3", "text": "Higher dyspnea_grade is associated with lower objective_response.", "kind": "novel"},
    {"id": "h20.4", "text": "Higher cough_grade is associated with lower objective_response.", "kind": "novel"},
    {"id": "h20.5", "text": "Higher appetite_loss_grade is associated with lower objective_response.", "kind": "novel"},
]
analyses = []
for v, hid in [('fatigue_grade', 'h20.1'), ('pain_nrs', 'h20.2'), ('dyspnea_grade', 'h20.3'),
                ('cough_grade', 'h20.4'), ('appetite_loss_grade', 'h20.5')]:
    d = r[f'sym_{v}']
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit(objective_response ~ {v})",
        "result_summary": f"{v}: {fmt_or(d)}.",
        "p_value": d['p_value'], "effect_estimate": d['coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 21: SNP main effects --------
i = 21
hyps = [
    {"id": "h21.1", "text": "At least one of the 23 candidate drug-metabolism SNPs (snp_rs*) has a non-zero main effect on objective_response.", "kind": "novel"},
]
analyses = []
sig_snps = []
for k, v in r.items():
    if k.startswith('snp_main_'):
        analyses.append({
            "hypothesis_ids": ["h21.1"],
            "code": f"logit(objective_response ~ {k.replace('snp_main_', '')})",
            "result_summary": f"{k.replace('snp_main_', '')}: {fmt_or(v)}.",
            "p_value": v['p_value'], "effect_estimate": v['coef_log_odds'], "significant": v['p_value'] < 0.05,
        })
        if v['p_value'] < 0.05:
            sig_snps.append((k.replace('snp_main_', ''), v['p_value']))
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 22: SNP x treatment interactions --------
i = 22
hyps = [
    {"id": "h22.1", "text": "At least one drug-metabolism SNP modifies the objective_response benefit of one of the four treatments (non-zero SNP x treatment interaction).", "kind": "novel"},
]
analyses = []
for k, v in r.items():
    if k.startswith('inter_snp_') and 'p_value' in v:
        analyses.append({
            "hypothesis_ids": ["h22.1"],
            "code": k,
            "result_summary": f"{k}: {fmt_int(v)}.",
            "p_value": v['p_value'], "effect_estimate": v['interaction_coef_log_odds'], "significant": v['p_value'] < 0.05,
        })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 23: ECOG, brain mets x treatment --------
i = 23
hyps = [
    {"id": "h23.1", "text": "Patients with ecog_ps >= 2 (ecog_high=1) derive a smaller increase in objective_response from treatment_pembrolizumab than ecog_ps <= 1 patients (negative interaction).", "kind": "novel"},
    {"id": "h23.2", "text": "Patients with ecog_ps >= 2 derive a different increase in objective_response from treatment_osimertinib than ecog_ps <= 1 patients.", "kind": "novel"},
    {"id": "h23.3", "text": "Patients with has_brain_mets = 1 derive a smaller increase in objective_response from treatment_pembrolizumab than those without brain mets (negative interaction).", "kind": "novel"},
    {"id": "h23.4", "text": "Patients with has_brain_mets = 1 derive a different increase in objective_response from treatment_osimertinib than those without brain mets.", "kind": "novel"},
]
analyses = []
for hid, key in [("h23.1", 'inter_ecoghigh_pembro'), ("h23.2", 'inter_ecoghigh_osi'),
                  ("h23.3", 'inter_brain_pembro'), ("h23.4", 'inter_brain_osi')]:
    d = r[key]; rr = d['response_rates']
    analyses.append({
        "hypothesis_ids": [hid],
        "code": key,
        "result_summary": (f"Cell rates t0/b0={rr['t0_b0']:.3f}, t0/b1={rr['t0_b1']:.3f}, "
                          f"t1/b0={rr['t1_b0']:.3f}, t1/b1={rr['t1_b1']:.3f}; {fmt_int(d)}."),
        "p_value": d['p_value'], "effect_estimate": d['interaction_coef_log_odds'], "significant": d['p_value'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 24: Multivariable model --------
i = 24
hyps = [
    {"id": "h24.1", "text": "Adjusting for clinical, biomarker, and laboratory covariates, the treatment_pembrolizumab x pdl1_tps interaction remains a positive, statistically significant effect on objective_response.", "kind": "refined"},
    {"id": "h24.2", "text": "Adjusting for the same covariates, the treatment_pembrolizumab x stk11_mutation interaction remains a negative effect on objective_response.", "kind": "refined"},
    {"id": "h24.3", "text": "After adjustment, ECOG performance status, stage_iv, has_brain_mets, weight_loss_pct_6mo, and albumin_g_dl remain independently associated with objective_response in the expected directions.", "kind": "refined"},
    {"id": "h24.4", "text": "Adjusting for the same covariates, the treatment-by-driver-mutation interactions for osimertinib x egfr_mutation, sotorasib x kras_g12c, and olaparib x brca2_mutation remain non-significant.", "kind": "refined"},
]
mv = r['multivariable']
analyses = []
analyses.append({
    "hypothesis_ids": ["h24.1"],
    "code": "logit multivariable model with biomarker x treatment interactions",
    "result_summary": (f"In adjusted model, treatment_pembrolizumab:pdl1_tps beta={mv['params']['treatment_pembrolizumab:pdl1_tps']:+.3f} "
                      f"(OR={mv['odds_ratios']['treatment_pembrolizumab:pdl1_tps']:.3f}), p={mv['pvalues']['treatment_pembrolizumab:pdl1_tps']:.3g}."),
    "p_value": mv['pvalues']['treatment_pembrolizumab:pdl1_tps'],
    "effect_estimate": mv['params']['treatment_pembrolizumab:pdl1_tps'],
    "significant": mv['pvalues']['treatment_pembrolizumab:pdl1_tps'] < 0.05,
})
analyses.append({
    "hypothesis_ids": ["h24.2"],
    "code": "logit multivariable model",
    "result_summary": (f"treatment_pembrolizumab:stk11_mutation beta={mv['params']['treatment_pembrolizumab:stk11_mutation']:+.3f} "
                      f"(OR={mv['odds_ratios']['treatment_pembrolizumab:stk11_mutation']:.3f}), p={mv['pvalues']['treatment_pembrolizumab:stk11_mutation']:.3g}."),
    "p_value": mv['pvalues']['treatment_pembrolizumab:stk11_mutation'],
    "effect_estimate": mv['params']['treatment_pembrolizumab:stk11_mutation'],
    "significant": mv['pvalues']['treatment_pembrolizumab:stk11_mutation'] < 0.05,
})
for term in ['ecog_ps', 'stage_iv', 'has_brain_mets', 'albumin_g_dl']:
    analyses.append({
        "hypothesis_ids": ["h24.3"],
        "code": "logit multivariable model",
        "result_summary": f"adjusted {term}: beta={mv['params'][term]:+.4f}, OR={mv['odds_ratios'][term]:.3f}, p={mv['pvalues'][term]:.3g}.",
        "p_value": mv['pvalues'][term], "effect_estimate": mv['params'][term],
        "significant": mv['pvalues'][term] < 0.05,
    })
for term in ['treatment_osimertinib:egfr_mutation', 'treatment_sotorasib:kras_g12c', 'treatment_olaparib:brca2_mutation']:
    analyses.append({
        "hypothesis_ids": ["h24.4"],
        "code": "logit multivariable model",
        "result_summary": f"adjusted {term}: beta={mv['params'][term]:+.4f}, OR={mv['odds_ratios'][term]:.3f}, p={mv['pvalues'][term]:.3g}.",
        "p_value": mv['pvalues'][term], "effect_estimate": mv['params'][term],
        "significant": mv['pvalues'][term] < 0.05,
    })
analyses.append({
    "hypothesis_ids": ["h24.3"],
    "code": "logit multivariable model",
    "result_summary": f"adjusted tmb_high: beta={mv['params']['tmb_high']:+.4f}, OR={mv['odds_ratios']['tmb_high']:.3f}, p={mv['pvalues']['tmb_high']:.3g}.",
    "p_value": mv['pvalues']['tmb_high'], "effect_estimate": mv['params']['tmb_high'],
    "significant": mv['pvalues']['tmb_high'] < 0.05,
})
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})

# -------- Iteration 25: Stratified TMB x pembro by PD-L1 --------
i = 25
hyps = [
    {"id": "h25.1", "text": "The treatment_pembrolizumab x tmb_high interaction on objective_response is concentrated in patients with pdl1_tps >= 0.5 (PD-L1 high) and is essentially absent in PD-L1 low patients.", "kind": "refined"},
]
analyses = []
for s in [0, 1]:
    d = r[f'tmb_pembro_in_pdl1_{s}']
    analyses.append({
        "hypothesis_ids": ["h25.1"],
        "code": f"logit(objective_response ~ treatment_pembrolizumab*tmb_high) restricted to pdl1_high == {s}",
        "result_summary": (f"In pdl1_high={s} subgroup (n={d['n']}), pembrolizumab x tmb_high interaction "
                          f"beta={d['inter_coef']:+.3f}, p={d['inter_p']:.3g}."),
        "p_value": d['inter_p'], "effect_estimate": d['inter_coef'],
        "significant": d['inter_p'] < 0.05,
    })
iterations.append({"index": i, "proposed_hypotheses": hyps, "analyses": analyses})


transcript = {
    "dataset_id": "ds001_nsclc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-named-bundle@1",
    "max_iterations": 25,
    "iterations": iterations,
}

with open('transcript.json', 'w') as f:
    json.dump(transcript, f, indent=2, default=str)
print(f'Wrote transcript.json with {len(iterations)} iterations')
ntot = sum(len(it['analyses']) for it in iterations)
print(f'Total analyses: {ntot}')
nhyp = sum(len(it['proposed_hypotheses']) for it in iterations)
print(f'Total hypotheses: {nhyp}')
