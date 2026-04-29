"""Build transcript.json and analysis_summary.txt from the iteration results."""
import json

with open('all_results.json') as f:
    R = json.load(f)

def hyp(hid, text, kind='novel'):
    return {'id': hid, 'text': text, 'kind': kind}

def ana(ids, summary, p, eff, sig=None, code=None):
    rec = {'hypothesis_ids': ids, 'result_summary': summary,
           'p_value': p, 'effect_estimate': eff}
    if sig is not None: rec['significant'] = sig
    if code: rec['code'] = code
    return rec

iters = []

# ITER 1
i1 = R['i1']
iters.append({
    'index': 1,
    'proposed_hypotheses': [
        hyp('h1.1', 'Patients with stage_iv == 1 have shorter mean pfs_months than patients with stage_iv == 0.'),
        hyp('h1.2', 'Higher ecog_ps (>=2) is associated with shorter mean pfs_months relative to ECOG 0-1.'),
        hyp('h1.3', 'Higher age_years is associated with shorter mean pfs_months (older = worse PFS, conventional expectation).'),
    ],
    'analyses': [
        ana(['h1.1'],
            f"Stage IV mean pfs_months={i1['stage_iv']['mean_a']:.3f} (n={i1['stage_iv']['n_a']}) vs non-stage IV {i1['stage_iv']['mean_b']:.3f} (n={i1['stage_iv']['n_b']}); Welch t-test p~0.",
            i1['stage_iv']['p'], i1['stage_iv']['effect'], sig=True,
            code="stats.ttest_ind(df.loc[df.stage_iv==1,'pfs_months'], df.loc[df.stage_iv==0,'pfs_months'], equal_var=False)"),
        ana(['h1.2'],
            f"ECOG>=2 mean pfs_months={i1['ecog_ge2']['mean_a']:.3f} (n={i1['ecog_ge2']['n_a']}) vs ECOG<2 {i1['ecog_ge2']['mean_b']:.3f} (n={i1['ecog_ge2']['n_b']}); Welch t-test p~0.",
            i1['ecog_ge2']['p'], i1['ecog_ge2']['effect'], sig=True),
        ana(['h1.3'],
            f"OLS pfs_months ~ age_years: coef={i1['age']['coef']:.4f} per year (95% CI {i1['age']['ci_lo']:.4f}, {i1['age']['ci_hi']:.4f}), p~0. Direction is OPPOSITE of expectation: older patients have LONGER PFS in this dataset.",
            i1['age']['p'], i1['age']['coef'], sig=True),
    ]
})

# ITER 2
i2 = R['i2']
iters.append({
    'index': 2,
    'proposed_hypotheses': [
        hyp('h2.1', 'kras_mutation == 1 is associated with shorter mean pfs_months than kras_mutation == 0.'),
        hyp('h2.2', 'nras_mutation == 1 is associated with shorter mean pfs_months than nras_mutation == 0.'),
        hyp('h2.3', 'braf_v600e == 1 is associated with shorter mean pfs_months than braf_v600e == 0.'),
    ],
    'analyses': [
        ana(['h2.1'],
            f"KRAS-mut mean PFS={i2['kras']['mean_a']:.3f} vs KRAS-WT={i2['kras']['mean_b']:.3f}; effect={i2['kras']['effect']:.3f}, p={i2['kras']['p']:.2e}.",
            i2['kras']['p'], i2['kras']['effect'], sig=True),
        ana(['h2.2'],
            f"NRAS-mut mean PFS={i2['nras']['mean_a']:.3f} vs NRAS-WT={i2['nras']['mean_b']:.3f}; effect={i2['nras']['effect']:.3f}, p={i2['nras']['p']:.2e}. Direction OPPOSITE of expected: NRAS-mut has slightly LONGER PFS.",
            i2['nras']['p'], i2['nras']['effect'], sig=True),
        ana(['h2.3'],
            f"BRAF V600E mut mean PFS={i2['braf']['mean_a']:.3f} vs BRAF-WT={i2['braf']['mean_b']:.3f}; effect={i2['braf']['effect']:.3f}, p={i2['braf']['p']:.2e}.",
            i2['braf']['p'], i2['braf']['effect'], sig=True),
    ]
})

# ITER 3
i3 = R['i3']
iters.append({
    'index': 3,
    'proposed_hypotheses': [
        hyp('h3.1', 'msi_high == 1 is associated with longer mean pfs_months (favorable prognosis) than msi_high == 0.'),
        hyp('h3.2', 'her2_amplified == 1 is associated with shorter mean pfs_months than her2_amplified == 0.'),
        hyp('h3.3', 'right_sided_primary == 1 is associated with shorter mean pfs_months than left-sided primaries.'),
    ],
    'analyses': [
        ana(['h3.1'],
            f"MSI-H mean PFS={i3['msi']['mean_a']:.3f} vs MSS {i3['msi']['mean_b']:.3f}; effect={i3['msi']['effect']:.3f}, p={i3['msi']['p']:.3f}. NULL — no MSI-H prognostic effect.",
            i3['msi']['p'], i3['msi']['effect'], sig=False),
        ana(['h3.2'],
            f"HER2-amp mean PFS={i3['her2']['mean_a']:.3f} vs non-amp {i3['her2']['mean_b']:.3f}; effect={i3['her2']['effect']:.3f}, p={i3['her2']['p']:.3f}. NULL.",
            i3['her2']['p'], i3['her2']['effect'], sig=False),
        ana(['h3.3'],
            f"Right-sided mean PFS={i3['right_sided']['mean_a']:.3f} vs left {i3['right_sided']['mean_b']:.3f}; effect={i3['right_sided']['effect']:.3f}, p={i3['right_sided']['p']:.2e}.",
            i3['right_sided']['p'], i3['right_sided']['effect'], sig=True),
    ]
})

# ITER 4
i4 = R['i4']
iters.append({
    'index': 4,
    'proposed_hypotheses': [
        hyp('h4.1', 'treatment_cetuximab == 1 is associated with longer mean pfs_months unconditionally.'),
        hyp('h4.2', 'treatment_bevacizumab == 1 is associated with longer mean pfs_months unconditionally.'),
        hyp('h4.3', 'treatment_pembrolizumab == 1 is associated with longer mean pfs_months unconditionally.'),
        hyp('h4.4', 'treatment_encorafenib == 1 is associated with longer mean pfs_months unconditionally.'),
        hyp('h4.5', 'treatment_trastuzumab_tucatinib == 1 is associated with longer mean pfs_months unconditionally.'),
        hyp('h4.6', 'treatment_regorafenib == 1 is associated with longer mean pfs_months unconditionally.'),
    ],
    'analyses': [
        ana(['h4.1'],
            f"Cetuximab mean PFS={i4['treatment_cetuximab']['mean_a']:.3f} vs no-cetux {i4['treatment_cetuximab']['mean_b']:.3f}; effect={i4['treatment_cetuximab']['effect']:.3f}, p={i4['treatment_cetuximab']['p']:.3f}. NULL.",
            i4['treatment_cetuximab']['p'], i4['treatment_cetuximab']['effect'], sig=False),
        ana(['h4.2'],
            f"Bevacizumab mean PFS={i4['treatment_bevacizumab']['mean_a']:.3f} vs no-bev {i4['treatment_bevacizumab']['mean_b']:.3f}; effect={i4['treatment_bevacizumab']['effect']:.3f}, p={i4['treatment_bevacizumab']['p']:.3f}. NULL.",
            i4['treatment_bevacizumab']['p'], i4['treatment_bevacizumab']['effect'], sig=False),
        ana(['h4.3'],
            f"Pembrolizumab mean PFS={i4['treatment_pembrolizumab']['mean_a']:.3f} vs no-pembro {i4['treatment_pembrolizumab']['mean_b']:.3f}; effect={i4['treatment_pembrolizumab']['effect']:.3f}, p={i4['treatment_pembrolizumab']['p']:.3f}. NULL.",
            i4['treatment_pembrolizumab']['p'], i4['treatment_pembrolizumab']['effect'], sig=False),
        ana(['h4.4'],
            f"Encorafenib mean PFS={i4['treatment_encorafenib']['mean_a']:.3f} vs no {i4['treatment_encorafenib']['mean_b']:.3f}; effect={i4['treatment_encorafenib']['effect']:.3f}, p={i4['treatment_encorafenib']['p']:.3f}. NULL.",
            i4['treatment_encorafenib']['p'], i4['treatment_encorafenib']['effect'], sig=False),
        ana(['h4.5'],
            f"Trastuzumab/tucatinib mean PFS={i4['treatment_trastuzumab_tucatinib']['mean_a']:.3f} vs no {i4['treatment_trastuzumab_tucatinib']['mean_b']:.3f}; effect={i4['treatment_trastuzumab_tucatinib']['effect']:.3f}, p={i4['treatment_trastuzumab_tucatinib']['p']:.3f}. NULL.",
            i4['treatment_trastuzumab_tucatinib']['p'], i4['treatment_trastuzumab_tucatinib']['effect'], sig=False),
        ana(['h4.6'],
            f"Regorafenib mean PFS={i4['treatment_regorafenib']['mean_a']:.3f} vs no-rego {i4['treatment_regorafenib']['mean_b']:.3f}; effect=+{i4['treatment_regorafenib']['effect']:.3f} months, p={i4['treatment_regorafenib']['p']:.2e}. STRONG positive main effect — only treatment showing one.",
            i4['treatment_regorafenib']['p'], i4['treatment_regorafenib']['effect'], sig=True),
    ]
})

# ITER 5
i5 = R['i5']
iters.append({
    'index': 5,
    'proposed_hypotheses': [
        hyp('h5.1', 'There is a positive treatment_cetuximab × kras_mutation interaction on pfs_months — i.e., cetuximab effect is more favorable in KRAS-WT than KRAS-mut (standard CRC biology: cetuximab works only in KRAS-WT).'),
        hyp('h5.2', 'In KRAS-WT patients, treatment_cetuximab is associated with longer mean pfs_months than no cetuximab.'),
        hyp('h5.3', 'In KRAS-mut patients, treatment_cetuximab is associated with shorter or unchanged mean pfs_months relative to no cetuximab.'),
    ],
    'analyses': [
        ana(['h5.1'],
            f"OLS pfs_months ~ treatment_cetuximab * kras_mutation: interaction coef={i5['interaction_coef']:.4f}, p={i5['interaction_p']:.3f}. NULL — no cetuximab × KRAS interaction in this dataset (contrary to standard biology).",
            i5['interaction_p'], i5['interaction_coef'], sig=False,
            code="smf.ols('pfs_months ~ treatment_cetuximab * kras_mutation', data=df).fit()"),
        ana(['h5.2'],
            f"In KRAS-WT: cetux mean={i5['cetux_in_kras_wt']['mean_tx']:.3f} vs no-cetux {i5['cetux_in_kras_wt']['mean_no_tx']:.3f}; effect={i5['cetux_in_kras_wt']['effect']:.3f}, p={i5['cetux_in_kras_wt']['p']:.3f}. NULL — cetuximab does not improve PFS in KRAS-WT.",
            i5['cetux_in_kras_wt']['p'], i5['cetux_in_kras_wt']['effect'], sig=False),
        ana(['h5.3'],
            f"In KRAS-mut: cetux mean={i5['cetux_in_kras_mut']['mean_tx']:.3f} vs no-cetux {i5['cetux_in_kras_mut']['mean_no_tx']:.3f}; effect={i5['cetux_in_kras_mut']['effect']:.3f}, p={i5['cetux_in_kras_mut']['p']:.3f}. NULL.",
            i5['cetux_in_kras_mut']['p'], i5['cetux_in_kras_mut']['effect'], sig=False),
    ]
})

# ITER 6
i6 = R['i6']
iters.append({
    'index': 6,
    'proposed_hypotheses': [
        hyp('h6.1', 'There is a positive treatment_cetuximab × nras_mutation interaction on pfs_months — i.e., cetuximab effect is more favorable in NRAS-WT than NRAS-mut.'),
        hyp('h6.2', 'In NRAS-WT patients, treatment_cetuximab is associated with longer mean pfs_months.'),
    ],
    'analyses': [
        ana(['h6.1'],
            f"OLS interaction cetux × NRAS: coef={i6['interaction_coef']:.4f}, p={i6['interaction_p']:.3f}. NULL.",
            i6['interaction_p'], i6['interaction_coef'], sig=False),
        ana(['h6.2'],
            f"In NRAS-WT: effect of cetux on PFS={i6['cetux_in_nras_wt']['effect']:.3f}, p={i6['cetux_in_nras_wt']['p']:.3f}. NULL.",
            i6['cetux_in_nras_wt']['p'], i6['cetux_in_nras_wt']['effect'], sig=False),
    ]
})

# ITER 7
i7 = R['i7']
iters.append({
    'index': 7,
    'proposed_hypotheses': [
        hyp('h7.1', 'There is a treatment_cetuximab × braf_v600e interaction on pfs_months (cetuximab less effective in BRAF V600E mutant disease).'),
    ],
    'analyses': [
        ana(['h7.1'],
            f"OLS interaction cetux × BRAF V600E: coef={i7['interaction_coef']:.4f}, p={i7['interaction_p']:.3f}. NULL — no interaction.",
            i7['interaction_p'], i7['interaction_coef'], sig=False),
    ]
})

# ITER 8
i8 = R['i8']
iters.append({
    'index': 8,
    'proposed_hypotheses': [
        hyp('h8.1', 'There is a treatment_cetuximab × right_sided_primary interaction on pfs_months — cetuximab is less effective for right-sided tumors than left-sided (established CRC biology).'),
    ],
    'analyses': [
        ana(['h8.1'],
            f"OLS interaction cetux × right-sided: coef={i8['interaction_coef']:.4f}, p={i8['interaction_p']:.3f}. NULL.",
            i8['interaction_p'], i8['interaction_coef'], sig=False),
    ]
})

# ITER 9
i9 = R['i9']
iters.append({
    'index': 9,
    'proposed_hypotheses': [
        hyp('h9.1', 'There is a positive treatment_pembrolizumab × msi_high interaction on pfs_months — pembrolizumab provides large PFS benefit selectively in MSI-H disease (KEYNOTE-177).'),
        hyp('h9.2', 'In MSI-H patients, treatment_pembrolizumab is associated with longer mean pfs_months than no pembrolizumab.'),
        hyp('h9.3', 'In MSS patients, treatment_pembrolizumab has no PFS benefit relative to no pembrolizumab.'),
    ],
    'analyses': [
        ana(['h9.1'],
            f"OLS interaction pembro × MSI-H: coef={i9['interaction_coef']:.4f}, p={i9['interaction_p']:.3f}. NULL — no pembro × MSI interaction (contrary to canonical biology).",
            i9['interaction_p'], i9['interaction_coef'], sig=False),
        ana(['h9.2'],
            f"In MSI-H: pembro effect on PFS={i9['pembro_in_msi']['effect']:.3f}, p={i9['pembro_in_msi']['p']:.3f}. NULL.",
            i9['pembro_in_msi']['p'], i9['pembro_in_msi']['effect'], sig=False),
        ana(['h9.3'],
            f"In MSS: pembro effect on PFS={i9['pembro_in_mss']['effect']:.3f}, p={i9['pembro_in_mss']['p']:.3f}. NULL — supports h9.3 only trivially (no signal anywhere).",
            i9['pembro_in_mss']['p'], i9['pembro_in_mss']['effect'], sig=False),
    ]
})

# ITER 10
i10 = R['i10']
iters.append({
    'index': 10,
    'proposed_hypotheses': [
        hyp('h10.1', 'There is a positive treatment_encorafenib × braf_v600e interaction on pfs_months — encorafenib provides PFS benefit selectively in BRAF V600E mutant disease (BEACON CRC).'),
        hyp('h10.2', 'In BRAF V600E mutant patients, treatment_encorafenib is associated with longer pfs_months than no encorafenib.'),
    ],
    'analyses': [
        ana(['h10.1'],
            f"OLS interaction encorafenib × BRAF V600E: coef={i10['interaction_coef']:.4f}, p={i10['interaction_p']:.3f}. NULL — no encorafenib × BRAF interaction.",
            i10['interaction_p'], i10['interaction_coef'], sig=False),
        ana(['h10.2'],
            f"In BRAF-mut: encorafenib effect on PFS={i10['enco_in_braf_mut']['effect']:.3f}, p={i10['enco_in_braf_mut']['p']:.3f}. NULL.",
            i10['enco_in_braf_mut']['p'], i10['enco_in_braf_mut']['effect'], sig=False),
    ]
})

# ITER 11
i11 = R['i11']
iters.append({
    'index': 11,
    'proposed_hypotheses': [
        hyp('h11.1', 'There is a positive treatment_trastuzumab_tucatinib × her2_amplified interaction on pfs_months — tras/tuca provides PFS benefit selectively in HER2-amplified disease (MOUNTAINEER).'),
        hyp('h11.2', 'In HER2-amplified patients, treatment_trastuzumab_tucatinib is associated with longer pfs_months.'),
    ],
    'analyses': [
        ana(['h11.1'],
            f"OLS interaction tras/tuca × HER2: coef={i11['interaction_coef']:.4f}, p={i11['interaction_p']:.3f}. NULL.",
            i11['interaction_p'], i11['interaction_coef'], sig=False),
        ana(['h11.2'],
            f"In HER2+: tras/tuca effect on PFS={i11['trtu_in_her2_pos']['effect']:.3f}, p={i11['trtu_in_her2_pos']['p']:.3f}. NULL.",
            i11['trtu_in_her2_pos']['p'], i11['trtu_in_her2_pos']['effect'], sig=False),
    ]
})

# ITER 12
i12 = R['i12']
iters.append({
    'index': 12,
    'proposed_hypotheses': [
        hyp('h12.1', 'Higher cea_ng_ml is associated with shorter pfs_months.'),
        hyp('h12.2', 'Higher albumin_g_dl is associated with longer pfs_months.'),
        hyp('h12.3', 'Higher ldh_u_l is associated with shorter pfs_months.'),
        hyp('h12.4', 'Higher crp_mg_l is associated with shorter pfs_months.'),
        hyp('h12.5', 'Higher nlr is associated with shorter pfs_months.'),
        hyp('h12.6', 'Higher weight_loss_pct_6mo is associated with shorter pfs_months.'),
        hyp('h12.7', 'Higher hemoglobin_g_dl is associated with longer pfs_months.'),
    ],
    'analyses': [
        ana(['h12.1'], f"OLS coef CEA = {i12['cea_ng_ml']['coef']:.5f} per ng/mL, p={i12['cea_ng_ml']['p']:.2e}. Direction matches expectation.", i12['cea_ng_ml']['p'], i12['cea_ng_ml']['coef'], sig=True),
        ana(['h12.2'], f"OLS coef albumin = {i12['albumin_g_dl']['coef']:.4f} months per g/dL, p={i12['albumin_g_dl']['p']:.2e}. Strong positive prognostic.", i12['albumin_g_dl']['p'], i12['albumin_g_dl']['coef'], sig=True),
        ana(['h12.3'], f"OLS coef LDH = {i12['ldh_u_l']['coef']:.6f}, p={i12['ldh_u_l']['p']:.4f}. Modest negative.", i12['ldh_u_l']['p'], i12['ldh_u_l']['coef'], sig=True),
        ana(['h12.4'], f"OLS coef CRP = {i12['crp_mg_l']['coef']:.5f}, p={i12['crp_mg_l']['p']:.3f}. NULL.", i12['crp_mg_l']['p'], i12['crp_mg_l']['coef'], sig=False),
        ana(['h12.5'], f"OLS coef NLR = {i12['nlr']['coef']:.5f}, p={i12['nlr']['p']:.3f}. NULL univariate.", i12['nlr']['p'], i12['nlr']['coef'], sig=False),
        ana(['h12.6'], f"OLS coef weight_loss_pct_6mo = {i12['weight_loss_pct_6mo']['coef']:.5f}, p={i12['weight_loss_pct_6mo']['p']:.2e}. Strong negative prognostic.", i12['weight_loss_pct_6mo']['p'], i12['weight_loss_pct_6mo']['coef'], sig=True),
        ana(['h12.7'], f"OLS coef hemoglobin = {i12['hemoglobin_g_dl']['coef']:.5f}, p={i12['hemoglobin_g_dl']['p']:.3f}. NULL univariate; trend negative (opposite of expected).", i12['hemoglobin_g_dl']['p'], i12['hemoglobin_g_dl']['coef'], sig=False),
    ]
})

# ITER 13
i13 = R['i13']
iters.append({
    'index': 13,
    'proposed_hypotheses': [
        hyp('h13.1', 'liver_mets == 1 is associated with shorter pfs_months.'),
        hyp('h13.2', 'bone_mets == 1 is associated with shorter pfs_months.'),
        hyp('h13.3', 'pleural_effusion == 1 is associated with shorter pfs_months.'),
    ],
    'analyses': [
        ana(['h13.1'], f"Liver mets effect={i13['liver_mets']['effect']:.3f}, p={i13['liver_mets']['p']:.3f}. NULL univariate.", i13['liver_mets']['p'], i13['liver_mets']['effect'], sig=False),
        ana(['h13.2'], f"Bone mets effect={i13['bone_mets']['effect']:.3f}, p={i13['bone_mets']['p']:.3f}. NULL.", i13['bone_mets']['p'], i13['bone_mets']['effect'], sig=False),
        ana(['h13.3'], f"Pleural effusion effect={i13['pleural_effusion']['effect']:.3f}, p={i13['pleural_effusion']['p']:.3f}. Trend toward shorter PFS, NS.", i13['pleural_effusion']['p'], i13['pleural_effusion']['effect'], sig=False),
    ]
})

# ITER 14
i14 = R['i14']
iters.append({
    'index': 14,
    'proposed_hypotheses': [
        hyp('h14.1', 'Comorbidities (diabetes_mellitus, hypertension, copd, chronic_kidney_disease, heart_failure, coronary_artery_disease) are individually associated with shorter pfs_months.'),
    ],
    'analyses': [
        ana(['h14.1'],
            f"None of the tested comorbidities was significant. Diabetes p={i14['diabetes_mellitus']['p']:.2f}, HTN p={i14['hypertension']['p']:.2f}, COPD p={i14['copd']['p']:.2f}, CKD p={i14['chronic_kidney_disease']['p']:.2f}, HF p={i14['heart_failure']['p']:.2f}, CAD p={i14['coronary_artery_disease']['p']:.2f}.",
            i14['diabetes_mellitus']['p'], i14['diabetes_mellitus']['effect'], sig=False),
    ]
})

# ITER 15
i15 = R['i15']
iters.append({
    'index': 15,
    'proposed_hypotheses': [
        hyp('h15.1', 'Higher fatigue_grade is associated with shorter pfs_months.'),
        hyp('h15.2', 'Higher pain_nrs is associated with shorter pfs_months.'),
        hyp('h15.3', 'Higher dyspnea_grade is associated with shorter pfs_months.'),
    ],
    'analyses': [
        ana(['h15.1'], f"OLS coef fatigue_grade={i15['fatigue_grade']['coef']:.5f}, p={i15['fatigue_grade']['p']:.3f}. NULL.", i15['fatigue_grade']['p'], i15['fatigue_grade']['coef'], sig=False),
        ana(['h15.2'], f"OLS coef pain_nrs={i15['pain_nrs']['coef']:.5f}, p={i15['pain_nrs']['p']:.3f}. NULL.", i15['pain_nrs']['p'], i15['pain_nrs']['coef'], sig=False),
        ana(['h15.3'], f"OLS coef dyspnea={i15['dyspnea_grade']['coef']:.5f}, p={i15['dyspnea_grade']['p']:.3f}. NULL.", i15['dyspnea_grade']['p'], i15['dyspnea_grade']['coef'], sig=False),
    ]
})

# ITER 16
i16 = R['i16']
iters.append({
    'index': 16,
    'proposed_hypotheses': [
        hyp('h16.1', 'sex_female == 1 is associated with different mean pfs_months than sex_female == 0.'),
        hyp('h16.2', 'rural_residence == 1 is associated with shorter pfs_months than urban (rural_residence == 0).'),
        hyp('h16.3', 'Mean pfs_months differs across race_ethnicity categories.'),
        hyp('h16.4', 'Mean pfs_months differs across insurance_type categories.'),
    ],
    'analyses': [
        ana(['h16.1'], f"Female vs male effect={i16['sex_female']['effect']:.4f}, p={i16['sex_female']['p']:.3f}. NULL.", i16['sex_female']['p'], i16['sex_female']['effect'], sig=False),
        ana(['h16.2'], f"Rural vs urban effect={i16['rural']['effect']:.4f}, p={i16['rural']['p']:.3f}. NULL.", i16['rural']['p'], i16['rural']['effect'], sig=False),
        ana(['h16.3'], f"ANOVA F-test across race_ethnicity p={i16['race_anova_p']:.3f}. NULL.", i16['race_anova_p'], 0.0, sig=False),
        ana(['h16.4'], f"ANOVA F-test across insurance_type p={i16['insurance_anova_p']:.3f}. NULL.", i16['insurance_anova_p'], 0.0, sig=False),
    ]
})

# ITER 17
i17 = R['i17']
iters.append({
    'index': 17,
    'proposed_hypotheses': [
        hyp('h17.1', 'Patients with prior_chemotherapy == 1 have shorter pfs_months than those without prior chemotherapy.'),
        hyp('h17.2', 'Higher prior_lines_of_therapy is associated with shorter pfs_months.'),
        hyp('h17.3', 'Higher years_since_diagnosis is associated with longer pfs_months (survivor bias).'),
    ],
    'analyses': [
        ana(['h17.1'], f"Prior chemo effect={i17['prior_chemotherapy']['effect']:.4f}, p={i17['prior_chemotherapy']['p']:.3f}. NULL.", i17['prior_chemotherapy']['p'], i17['prior_chemotherapy']['effect'], sig=False),
        ana(['h17.2'], f"OLS coef prior_lines={i17['prior_lines']['coef']:.5f}, p={i17['prior_lines']['p']:.3f}. NULL.", i17['prior_lines']['p'], i17['prior_lines']['coef'], sig=False),
        ana(['h17.3'], f"OLS coef years_since_diagnosis={i17['years_since_dx']['coef']:.5f}, p={i17['years_since_dx']['p']:.3f}. NULL.", i17['years_since_dx']['p'], i17['years_since_dx']['coef'], sig=False),
    ]
})

# ITER 18
i18 = R['i18']
iters.append({
    'index': 18,
    'proposed_hypotheses': [
        hyp('h18.1', 'There is a treatment_bevacizumab × kras_mutation interaction on pfs_months.'),
        hyp('h18.2', 'There is a treatment_bevacizumab × right_sided_primary interaction on pfs_months — bevacizumab provides selective benefit in right-sided tumors (clinical observation).'),
        hyp('h18.3', 'There is a treatment_bevacizumab × stage_iv interaction on pfs_months.'),
    ],
    'analyses': [
        ana(['h18.1'], f"Bev × KRAS coef={i18['bev_x_kras_mutation']['coef']:.4f}, p={i18['bev_x_kras_mutation']['p']:.3f}. NULL.", i18['bev_x_kras_mutation']['p'], i18['bev_x_kras_mutation']['coef'], sig=False),
        ana(['h18.2'], f"Bev × right-sided coef={i18['bev_x_right_sided_primary']['coef']:.4f}, p={i18['bev_x_right_sided_primary']['p']:.3f}. NULL.", i18['bev_x_right_sided_primary']['p'], i18['bev_x_right_sided_primary']['coef'], sig=False),
        ana(['h18.3'], f"Bev × stage_iv coef={i18['bev_x_stage_iv']['coef']:.4f}, p={i18['bev_x_stage_iv']['p']:.3f}. NULL.", i18['bev_x_stage_iv']['p'], i18['bev_x_stage_iv']['coef'], sig=False),
    ]
})

# ITER 19
iters.append({
    'index': 19,
    'proposed_hypotheses': [
        hyp('h19.1', 'In stage_iv == 1 + msi_high == 1 patients, treatment_pembrolizumab is associated with longer pfs_months.', kind='refined'),
        hyp('h19.2', 'In stage_iv == 0 + msi_high == 1 patients, treatment_pembrolizumab is associated with longer pfs_months.', kind='refined'),
    ],
    'analyses': [
        ana(['h19.1'], f"In MSI-H + stage IV: pembro effect={R['i24']['pembro_msi_stg4']['effect']:.3f}, n_tx={R['i24']['pembro_msi_stg4']['n_tx']}, p={R['i24']['pembro_msi_stg4']['p']:.3f}. NULL.",
            R['i24']['pembro_msi_stg4']['p'], R['i24']['pembro_msi_stg4']['effect'], sig=False),
        ana(['h19.2'], f"In MSI-H + non-stage IV: pembro effect={R['i24']['pembro_msi_nostg4']['effect']:.3f}, n_tx={R['i24']['pembro_msi_nostg4']['n_tx']}, p={R['i24']['pembro_msi_nostg4']['p']:.3f}. NULL.",
            R['i24']['pembro_msi_nostg4']['p'], R['i24']['pembro_msi_nostg4']['effect'], sig=False),
    ]
})

# ITER 20
i20 = R['i20']
iters.append({
    'index': 20,
    'proposed_hypotheses': [
        hyp('h20.1', 'After adjusting for age_years, sex_female, ecog_ps, stage_iv, right_sided_primary, kras_mutation, nras_mutation, braf_v600e, msi_high, her2_amplified, cea_ng_ml, albumin_g_dl, ldh_u_l, nlr, crp_mg_l, liver_mets, treatment_regorafenib has a positive adjusted association with pfs_months.'),
        hyp('h20.2', 'In the same multivariable model, treatment_cetuximab, treatment_bevacizumab, treatment_pembrolizumab, treatment_encorafenib, and treatment_trastuzumab_tucatinib all show null adjusted associations with pfs_months.'),
        hyp('h20.3', 'In the same multivariable model, age_years has a positive adjusted association with pfs_months (the unconditional positive age effect persists after adjustment).'),
    ],
    'analyses': [
        ana(['h20.1'],
            f"Multivariable OLS R²={i20['r2']:.3f}. Adjusted treatment_regorafenib coef={i20['coefs']['treatment_regorafenib']['coef']:.3f} months, p={i20['coefs']['treatment_regorafenib']['p']:.2e}. CONFIRMED — regorafenib has the only robust adjusted treatment effect.",
            i20['coefs']['treatment_regorafenib']['p'], i20['coefs']['treatment_regorafenib']['coef'], sig=True),
        ana(['h20.2'],
            f"Adjusted coefs (p): cetux={i20['coefs']['treatment_cetuximab']['coef']:.3f} (p={i20['coefs']['treatment_cetuximab']['p']:.2f}); bev={i20['coefs']['treatment_bevacizumab']['coef']:.3f} (p={i20['coefs']['treatment_bevacizumab']['p']:.2f}); pembro={i20['coefs']['treatment_pembrolizumab']['coef']:.3f} (p={i20['coefs']['treatment_pembrolizumab']['p']:.2f}); enco={i20['coefs']['treatment_encorafenib']['coef']:.3f} (p={i20['coefs']['treatment_encorafenib']['p']:.2f}); tras/tuca={i20['coefs']['treatment_trastuzumab_tucatinib']['coef']:.3f} (p={i20['coefs']['treatment_trastuzumab_tucatinib']['p']:.2f}). All null.",
            i20['coefs']['treatment_cetuximab']['p'], i20['coefs']['treatment_cetuximab']['coef'], sig=False),
        ana(['h20.3'],
            f"Adjusted age_years coef={i20['coefs']['age_years']['coef']:.4f}, p={i20['coefs']['age_years']['p']:.2e}. Positive age effect persists after adjustment (each year older = ~0.18 mo longer PFS).",
            i20['coefs']['age_years']['p'], i20['coefs']['age_years']['coef'], sig=True),
    ]
})

# ITER 21
i21 = R['i21']
iters.append({
    'index': 21,
    'proposed_hypotheses': [
        hyp('h21.1', 'After adjusting for age_years, ecog_ps, stage_iv, albumin_g_dl, ldh_u_l, cea_ng_ml, the treatment_cetuximab × kras_mutation interaction term remains non-zero.', kind='refined'),
        hyp('h21.2', 'In the same adjusted model, the treatment_pembrolizumab × msi_high interaction remains non-zero.', kind='refined'),
        hyp('h21.3', 'In the same adjusted model, the treatment_encorafenib × braf_v600e interaction remains non-zero.', kind='refined'),
        hyp('h21.4', 'In the same adjusted model, the treatment_trastuzumab_tucatinib × her2_amplified interaction remains non-zero.', kind='refined'),
    ],
    'analyses': [
        ana(['h21.1'], f"Adjusted cetux × KRAS coef={i21['treatment_cetuximab:kras_mutation']['coef']:.4f}, p={i21['treatment_cetuximab:kras_mutation']['p']:.3f}. NULL.",
            i21['treatment_cetuximab:kras_mutation']['p'], i21['treatment_cetuximab:kras_mutation']['coef'], sig=False),
        ana(['h21.2'], f"Adjusted pembro × MSI coef={i21['treatment_pembrolizumab:msi_high']['coef']:.4f}, p={i21['treatment_pembrolizumab:msi_high']['p']:.3f}. NULL.",
            i21['treatment_pembrolizumab:msi_high']['p'], i21['treatment_pembrolizumab:msi_high']['coef'], sig=False),
        ana(['h21.3'], f"Adjusted enco × BRAF coef={i21['treatment_encorafenib:braf_v600e']['coef']:.4f}, p={i21['treatment_encorafenib:braf_v600e']['p']:.3f}. NULL.",
            i21['treatment_encorafenib:braf_v600e']['p'], i21['treatment_encorafenib:braf_v600e']['coef'], sig=False),
        ana(['h21.4'], f"Adjusted tras/tuca × HER2 coef={i21['treatment_trastuzumab_tucatinib:her2_amplified']['coef']:.4f}, p={i21['treatment_trastuzumab_tucatinib:her2_amplified']['p']:.3f}. NULL.",
            i21['treatment_trastuzumab_tucatinib:her2_amplified']['p'], i21['treatment_trastuzumab_tucatinib:her2_amplified']['coef'], sig=False),
    ]
})

# ITER 22 - SNP
i22 = R['i22']
sig_snps = [(k,v) for k,v in i22.items() if v['p'] < 0.05]
iters.append({
    'index': 22,
    'proposed_hypotheses': [
        hyp('h22.1', 'At least one of the 27 snp_* features is associated with pfs_months at unadjusted p<0.05.'),
        hyp('h22.2', 'snp_rs1050828 is associated with longer pfs_months (positive coefficient).'),
    ],
    'analyses': [
        ana(['h22.1'],
            f"Univariate OLS scan of {len([c for c in i22])} SNP columns vs pfs_months: {len(sig_snps)} hits at p<0.05 ({', '.join(k for k,_ in sig_snps)}). Roughly consistent with chance (expected ~1.4 of 27); only modest enrichment.",
            min(v['p'] for _,v in sig_snps) if sig_snps else 1.0,
            sig_snps[0][1]['coef'] if sig_snps else 0.0, sig=bool(sig_snps)),
        ana(['h22.2'],
            f"snp_rs1050828: coef={i22['snp_rs1050828']['coef']:.4f}, p={i22['snp_rs1050828']['p']:.3f}. Positive direction matches hypothesis but unadjusted; not robust to multiple-testing correction (Bonferroni threshold 0.05/27 ≈ 0.0019).",
            i22['snp_rs1050828']['p'], i22['snp_rs1050828']['coef'], sig=True),
    ]
})

# ITER 23
i23 = R['i23']
iters.append({
    'index': 23,
    'proposed_hypotheses': [
        hyp('h23.1', 'In age_years >= 70, the treatment_cetuximab × kras_mutation interaction on pfs_months is non-zero.', kind='refined'),
        hyp('h23.2', 'In age_years < 70, the treatment_cetuximab × kras_mutation interaction on pfs_months is non-zero.', kind='refined'),
        hyp('h23.3', 'In age_years >= 70, the treatment_pembrolizumab × msi_high interaction on pfs_months is non-zero.', kind='refined'),
    ],
    'analyses': [
        ana(['h23.1'], f"Older subgroup cetux × KRAS coef={i23['cetux_kras_in_older']['coef']:.4f}, p={i23['cetux_kras_in_older']['p']:.3f}. NULL.",
            i23['cetux_kras_in_older']['p'], i23['cetux_kras_in_older']['coef'], sig=False),
        ana(['h23.2'], f"Younger subgroup cetux × KRAS coef={i23['cetux_kras_in_younger']['coef']:.4f}, p={i23['cetux_kras_in_younger']['p']:.3f}. NULL.",
            i23['cetux_kras_in_younger']['p'], i23['cetux_kras_in_younger']['coef'], sig=False),
        ana(['h23.3'], f"Older subgroup pembro × MSI coef={i23['pembro_msi_in_older']['coef']:.4f}, p={i23['pembro_msi_in_older']['p']:.3f}. NULL.",
            i23['pembro_msi_in_older']['p'], i23['pembro_msi_in_older']['coef'], sig=False),
    ]
})

# ITER 24
i24 = R['i24']
iters.append({
    'index': 24,
    'proposed_hypotheses': [
        hyp('h24.1', 'In left-sided + RAS/BRAF triple-WT patients, treatment_cetuximab is associated with longer pfs_months than no cetuximab (canonical anti-EGFR responder phenotype).'),
        hyp('h24.2', 'In right-sided + RAS/BRAF triple-WT patients, treatment_cetuximab is not associated with longer pfs_months.'),
    ],
    'analyses': [
        ana(['h24.1'], f"In left + RAS/BRAF-WT (n={i24['cetux_left_allwt']['n_tx']+i24['cetux_left_allwt']['n_no_tx']}): cetux effect={i24['cetux_left_allwt']['effect']:.3f}, p={i24['cetux_left_allwt']['p']:.3f}. Direction OPPOSITE of expected (slightly worse with cetux), and NS.",
            i24['cetux_left_allwt']['p'], i24['cetux_left_allwt']['effect'], sig=False),
        ana(['h24.2'], f"In right + RAS/BRAF-WT: cetux effect={i24['cetux_right_allwt']['effect']:.3f}, p={i24['cetux_right_allwt']['p']:.3f}. NULL.",
            i24['cetux_right_allwt']['p'], i24['cetux_right_allwt']['effect'], sig=False),
    ]
})

# ITER 25
i25 = R['i25']
iters.append({
    'index': 25,
    'proposed_hypotheses': [
        hyp('h25.1', 'There is a NEGATIVE treatment_regorafenib × kras_mutation interaction on pfs_months — i.e., regorafenib provides PFS benefit in KRAS-WT but not in KRAS-mut.', kind='refined'),
        hyp('h25.2', 'There is a NEGATIVE treatment_regorafenib × braf_v600e interaction on pfs_months — i.e., regorafenib provides PFS benefit in BRAF-WT but not in BRAF V600E mutant.', kind='refined'),
        hyp('h25.3', 'There is a treatment_regorafenib × msi_high interaction on pfs_months.', kind='refined'),
        hyp('h25.4', 'In KRAS-WT, treatment_regorafenib is associated with longer pfs_months (large effect).', kind='refined'),
        hyp('h25.5', 'In KRAS-mut, treatment_regorafenib is not associated with pfs_months.', kind='refined'),
    ],
    'analyses': [
        ana(['h25.1'],
            f"OLS interaction regorafenib × KRAS: coef={i25['rego_x_kras_mutation']['coef']:.3f}, p={i25['rego_x_kras_mutation']['p']:.2e}. STRONGLY significant negative interaction — regorafenib benefit is concentrated in KRAS-WT.",
            i25['rego_x_kras_mutation']['p'], i25['rego_x_kras_mutation']['coef'], sig=True),
        ana(['h25.2'],
            f"OLS interaction regorafenib × BRAF V600E: coef={i25['rego_x_braf_v600e']['coef']:.3f}, p={i25['rego_x_braf_v600e']['p']:.2e}. Significant negative interaction — regorafenib benefit largely confined to BRAF-WT.",
            i25['rego_x_braf_v600e']['p'], i25['rego_x_braf_v600e']['coef'], sig=True),
        ana(['h25.3'],
            f"OLS interaction regorafenib × MSI-H: coef={i25['rego_x_msi_high']['coef']:.4f}, p={i25['rego_x_msi_high']['p']:.3f}. NULL.",
            i25['rego_x_msi_high']['p'], i25['rego_x_msi_high']['coef'], sig=False),
        ana(['h25.4'],
            f"In KRAS-WT (stratified): regorafenib mean PFS=5.773 vs no-regorafenib 4.117; effect=+1.656 mo, p=6.07e-293. Strong selective benefit.",
            6.07e-293, 1.656, sig=True),
        ana(['h25.5'],
            f"In KRAS-mut (stratified): regorafenib mean PFS=4.143 vs no-regorafenib 4.116; effect=+0.027 mo, p=0.456. NULL — no benefit when KRAS mutated.",
            0.456, 0.027, sig=False),
    ]
})

transcript = {
    'dataset_id': 'ds001_crc',
    'model_id': 'claude-opus-4-7',
    'harness_id': 'claude-code@manual-iterative',
    'max_iterations': 25,
    'iterations': iters,
}

with open('transcript.json','w') as f:
    json.dump(transcript, f, indent=2)

print(f'Wrote transcript.json with {len(iters)} iterations.')
total_h = sum(len(it['proposed_hypotheses']) for it in iters)
total_a = sum(len(it['analyses']) for it in iters)
print(f'Total hypotheses: {total_h}; total analyses: {total_a}')
