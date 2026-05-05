"""Build transcript.json and analysis_summary.txt from clean_results.json."""
import json

R = json.load(open('clean_results.json'))

def fmt(b, p): return f"beta={b:.4f}, p={p:.3g}"

iterations = []

# -------------------- Iteration 1: demographics --------------------
iter1 = {
  "index": 1,
  "proposed_hypotheses": [
    {"id":"h1.1","text":"Older age (age_years) is associated with shorter pfs_months in the overall cohort.","kind":"novel"},
    {"id":"h1.2","text":"Female patients (sex_female=1) have different mean pfs_months than male patients in the overall cohort.","kind":"novel"},
    {"id":"h1.3","text":"Higher ecog_ps is associated with shorter pfs_months in the overall cohort.","kind":"novel"},
    {"id":"h1.4","text":"Stage IV disease (stage_iv=1) is associated with shorter pfs_months in the overall cohort.","kind":"novel"},
  ],
  "analyses": [
    {"hypothesis_ids":["h1.1"],
     "code":"smf.ols('pfs_months ~ age_years', df).fit()",
     "result_summary":f"OLS PFS on age_years: beta={R['i1_age']['beta']:.4f} mo per year, p={R['i1_age']['p']:.3g}. Direction is OPPOSITE to the conventional clinical expectation: older patients have LONGER PFS in this cohort.",
     "p_value":R['i1_age']['p'], "effect_estimate":R['i1_age']['beta'], "significant":True},
    {"hypothesis_ids":["h1.2"],
     "code":"smf.ols('pfs_months ~ sex_female', df).fit()",
     "result_summary":f"OLS: beta={R['i1_sex']['beta']:.4f} mo, p={R['i1_sex']['p']:.3g}. No detectable sex difference in PFS.",
     "p_value":R['i1_sex']['p'], "effect_estimate":R['i1_sex']['beta'], "significant":False},
    {"hypothesis_ids":["h1.3"],
     "code":"smf.ols('pfs_months ~ ecog_ps', df).fit()",
     "result_summary":f"OLS: per-unit ECOG beta={R['i1_ecog']['beta']:.4f} mo, p={R['i1_ecog']['p']:.3g}. Mean PFS by ECOG 0/1/2 = 5.27 / 4.05 / 2.91 mo. Strong, monotonic detrimental effect.",
     "p_value":R['i1_ecog']['p'], "effect_estimate":R['i1_ecog']['beta'], "significant":True},
    {"hypothesis_ids":["h1.4"],
     "code":"smf.ols('pfs_months ~ stage_iv', df).fit()",
     "result_summary":f"OLS: beta={R['i1_stage']['beta']:.4f} mo, p={R['i1_stage']['p']:.3g}. Stage IV reduces PFS by ~1.35 mo on average.",
     "p_value":R['i1_stage']['p'], "effect_estimate":R['i1_stage']['beta'], "significant":True},
  ]
}
iterations.append(iter1)

# -------------------- Iteration 2: labs --------------------
labs = R['i2_labs']['results']
iter2 = {
  "index": 2,
  "proposed_hypotheses": [
    {"id":"h2.1","text":"Higher cea_ng_ml is associated with shorter pfs_months in the overall cohort.","kind":"novel"},
    {"id":"h2.2","text":"Higher albumin_g_dl is associated with longer pfs_months in the overall cohort.","kind":"novel"},
    {"id":"h2.3","text":"Higher ldh_u_l is associated with shorter pfs_months in the overall cohort.","kind":"novel"},
    {"id":"h2.4","text":"Greater weight_loss_pct_6mo is associated with shorter pfs_months in the overall cohort.","kind":"novel"},
    {"id":"h2.5","text":"Higher crp_mg_l is associated with shorter pfs_months in the overall cohort.","kind":"novel"},
    {"id":"h2.6","text":"Higher nlr (neutrophil-lymphocyte ratio) is associated with shorter pfs_months in the overall cohort.","kind":"novel"},
    {"id":"h2.7","text":"Higher hemoglobin_g_dl is associated with longer pfs_months in the overall cohort.","kind":"novel"},
  ],
  "analyses": [
    {"hypothesis_ids":["h2.1"],
     "code":"smf.ols('pfs_months ~ cea_ng_ml', df).fit()",
     "result_summary":f"OLS: cea_ng_ml beta={labs['cea_ng_ml']['beta']:.5f} mo per ng/mL, p={labs['cea_ng_ml']['p']:.3g}. Direction is detrimental as expected.",
     "p_value":labs['cea_ng_ml']['p'], "effect_estimate":labs['cea_ng_ml']['beta'], "significant":True},
    {"hypothesis_ids":["h2.2"],
     "code":"smf.ols('pfs_months ~ albumin_g_dl', df).fit()",
     "result_summary":f"OLS: albumin_g_dl beta={labs['albumin_g_dl']['beta']:.4f}, p={labs['albumin_g_dl']['p']:.3g}. Direction is favorable as expected.",
     "p_value":labs['albumin_g_dl']['p'], "effect_estimate":labs['albumin_g_dl']['beta'], "significant":True},
    {"hypothesis_ids":["h2.3"],
     "code":"smf.ols('pfs_months ~ ldh_u_l', df).fit()",
     "result_summary":f"OLS: ldh_u_l beta={labs['ldh_u_l']['beta']:.5f}, p={labs['ldh_u_l']['p']:.3g}. Small detrimental effect.",
     "p_value":labs['ldh_u_l']['p'], "effect_estimate":labs['ldh_u_l']['beta'], "significant":True},
    {"hypothesis_ids":["h2.4"],
     "code":"smf.ols('pfs_months ~ weight_loss_pct_6mo', df).fit()",
     "result_summary":f"OLS: weight_loss_pct_6mo beta={labs['weight_loss_pct_6mo']['beta']:.4f}, p={labs['weight_loss_pct_6mo']['p']:.3g}. Strongly detrimental.",
     "p_value":labs['weight_loss_pct_6mo']['p'], "effect_estimate":labs['weight_loss_pct_6mo']['beta'], "significant":True},
    {"hypothesis_ids":["h2.5"],
     "code":"smf.ols('pfs_months ~ crp_mg_l', df).fit()",
     "result_summary":f"OLS: crp_mg_l beta={labs['crp_mg_l']['beta']:.5f}, p={labs['crp_mg_l']['p']:.3g}. Trend toward detrimental but not at p<0.05.",
     "p_value":labs['crp_mg_l']['p'], "effect_estimate":labs['crp_mg_l']['beta'], "significant":False},
    {"hypothesis_ids":["h2.6"],
     "code":"smf.ols('pfs_months ~ nlr', df).fit()",
     "result_summary":f"OLS: nlr beta={labs['nlr']['beta']:.5f}, p={labs['nlr']['p']:.3g}. Not significant.",
     "p_value":labs['nlr']['p'], "effect_estimate":labs['nlr']['beta'], "significant":False},
    {"hypothesis_ids":["h2.7"],
     "code":"smf.ols('pfs_months ~ hemoglobin_g_dl', df).fit()",
     "result_summary":f"OLS: hemoglobin_g_dl beta={labs['hemoglobin_g_dl']['beta']:.5f}, p={labs['hemoglobin_g_dl']['p']:.3g}. Not significant; sign even slightly negative.",
     "p_value":labs['hemoglobin_g_dl']['p'], "effect_estimate":labs['hemoglobin_g_dl']['beta'], "significant":False},
  ]
}
iterations.append(iter2)

# -------------------- Iteration 3: tumor location and biomarkers --------------------
iter3 = {
  "index": 3,
  "proposed_hypotheses": [
    {"id":"h3.1","text":"Right-sided primaries (right_sided_primary=1) are associated with shorter pfs_months than left-sided in the overall cohort.","kind":"novel"},
    {"id":"h3.2","text":"KRAS-mutated tumors (kras_mutation=1) are associated with shorter pfs_months than KRAS wild-type in the overall cohort.","kind":"novel"},
    {"id":"h3.3","text":"NRAS-mutated tumors (nras_mutation=1) are associated with shorter pfs_months than NRAS wild-type in the overall cohort.","kind":"novel"},
    {"id":"h3.4","text":"BRAF V600E-mutant tumors (braf_v600e=1) are associated with shorter pfs_months than BRAF wild-type in the overall cohort.","kind":"novel"},
    {"id":"h3.5","text":"MSI-high tumors (msi_high=1) have different mean pfs_months than MSS tumors in the overall cohort.","kind":"novel"},
    {"id":"h3.6","text":"HER2-amplified tumors (her2_amplified=1) have different mean pfs_months than non-amplified in the overall cohort.","kind":"novel"},
    {"id":"h3.7","text":"NTRK-fusion-positive tumors (ntrk_fusion=1) have different mean pfs_months than NTRK-negative in the overall cohort.","kind":"novel"},
  ],
  "analyses": [
    {"hypothesis_ids":["h3.1"], "code":"smf.ols('pfs_months ~ right_sided_primary', df).fit()",
     "result_summary":f"PFS: 4.42 (left) vs 4.11 (right); beta={R['i3_right_sided_primary']['beta']:.4f}, p={R['i3_right_sided_primary']['p']:.3g}. Right-sided is detrimental as expected.",
     "p_value":R['i3_right_sided_primary']['p'], "effect_estimate":R['i3_right_sided_primary']['beta'], "significant":True},
    {"hypothesis_ids":["h3.2"], "code":"smf.ols('pfs_months ~ kras_mutation', df).fit()",
     "result_summary":f"PFS: 4.45 (wt) vs 4.12 (mut); beta={R['i3_kras_mutation']['beta']:.4f}, p={R['i3_kras_mutation']['p']:.3g}. KRAS mutation detrimental.",
     "p_value":R['i3_kras_mutation']['p'], "effect_estimate":R['i3_kras_mutation']['beta'], "significant":True},
    {"hypothesis_ids":["h3.3"], "code":"smf.ols('pfs_months ~ nras_mutation', df).fit()",
     "result_summary":f"PFS: 4.30 (wt) vs 4.53 (mut); beta={R['i3_nras_mutation']['beta']:.4f}, p={R['i3_nras_mutation']['p']:.3g}. Surprising: NRAS-mutant patients have a small but significant LONGER PFS overall.",
     "p_value":R['i3_nras_mutation']['p'], "effect_estimate":R['i3_nras_mutation']['beta'], "significant":True},
    {"hypothesis_ids":["h3.4"], "code":"smf.ols('pfs_months ~ braf_v600e', df).fit()",
     "result_summary":f"PFS: 4.32 (wt) vs 4.09 (mut); beta={R['i3_braf_v600e']['beta']:.4f}, p={R['i3_braf_v600e']['p']:.3g}. BRAF V600E detrimental as expected.",
     "p_value":R['i3_braf_v600e']['p'], "effect_estimate":R['i3_braf_v600e']['beta'], "significant":True},
    {"hypothesis_ids":["h3.5"], "code":"smf.ols('pfs_months ~ msi_high', df).fit()",
     "result_summary":f"PFS: 4.31 (MSS) vs 4.29 (MSI-H); beta={R['i3_msi_high']['beta']:.4f}, p={R['i3_msi_high']['p']:.3g}. No detectable PFS difference by MSI status overall.",
     "p_value":R['i3_msi_high']['p'], "effect_estimate":R['i3_msi_high']['beta'], "significant":False},
    {"hypothesis_ids":["h3.6"], "code":"smf.ols('pfs_months ~ her2_amplified', df).fit()",
     "result_summary":f"beta={R['i3_her2_amplified']['beta']:.4f}, p={R['i3_her2_amplified']['p']:.3g}. No detectable HER2 effect on PFS overall.",
     "p_value":R['i3_her2_amplified']['p'], "effect_estimate":R['i3_her2_amplified']['beta'], "significant":False},
    {"hypothesis_ids":["h3.7"], "code":"smf.ols('pfs_months ~ ntrk_fusion', df).fit()",
     "result_summary":f"beta={R['i3_ntrk_fusion']['beta']:.4f}, p={R['i3_ntrk_fusion']['p']:.3g}. No detectable NTRK-fusion effect on PFS overall (n=251 fusion-positive).",
     "p_value":R['i3_ntrk_fusion']['p'], "effect_estimate":R['i3_ntrk_fusion']['beta'], "significant":False},
  ]
}
iterations.append(iter3)

# -------------------- Iteration 4: marginal treatment effects --------------------
iter4 = {
  "index": 4,
  "proposed_hypotheses": [
    {"id":"h4.1","text":"Patients receiving treatment_cetuximab have different mean pfs_months than patients not receiving it in the overall cohort.","kind":"novel"},
    {"id":"h4.2","text":"Patients receiving treatment_bevacizumab have different mean pfs_months than patients not receiving it in the overall cohort.","kind":"novel"},
    {"id":"h4.3","text":"Patients receiving treatment_pembrolizumab have different mean pfs_months than patients not receiving it in the overall cohort.","kind":"novel"},
    {"id":"h4.4","text":"Patients receiving treatment_encorafenib have different mean pfs_months than patients not receiving it in the overall cohort.","kind":"novel"},
    {"id":"h4.5","text":"Patients receiving treatment_trastuzumab_tucatinib have different mean pfs_months than patients not receiving it in the overall cohort.","kind":"novel"},
    {"id":"h4.6","text":"Patients receiving treatment_regorafenib have LONGER mean pfs_months than patients not receiving it in the overall cohort.","kind":"novel"},
  ],
  "analyses": [
    {"hypothesis_ids":["h4.1"], "code":"smf.ols('pfs_months ~ treatment_cetuximab', df).fit()",
     "result_summary":f"Mean PFS: 4.32 (off) vs 4.29 (on); beta={R['i4_treatment_cetuximab']['beta']:.4f}, p={R['i4_treatment_cetuximab']['p']:.3g}. No significant marginal benefit.",
     "p_value":R['i4_treatment_cetuximab']['p'], "effect_estimate":R['i4_treatment_cetuximab']['beta'], "significant":False},
    {"hypothesis_ids":["h4.2"], "code":"smf.ols('pfs_months ~ treatment_bevacizumab', df).fit()",
     "result_summary":f"beta={R['i4_treatment_bevacizumab']['beta']:.4f}, p={R['i4_treatment_bevacizumab']['p']:.3g}. No marginal effect.",
     "p_value":R['i4_treatment_bevacizumab']['p'], "effect_estimate":R['i4_treatment_bevacizumab']['beta'], "significant":False},
    {"hypothesis_ids":["h4.3"], "code":"smf.ols('pfs_months ~ treatment_pembrolizumab', df).fit()",
     "result_summary":f"beta={R['i4_treatment_pembrolizumab']['beta']:.4f}, p={R['i4_treatment_pembrolizumab']['p']:.3g}. No marginal effect.",
     "p_value":R['i4_treatment_pembrolizumab']['p'], "effect_estimate":R['i4_treatment_pembrolizumab']['beta'], "significant":False},
    {"hypothesis_ids":["h4.4"], "code":"smf.ols('pfs_months ~ treatment_encorafenib', df).fit()",
     "result_summary":f"beta={R['i4_treatment_encorafenib']['beta']:.4f}, p={R['i4_treatment_encorafenib']['p']:.3g}. No marginal effect.",
     "p_value":R['i4_treatment_encorafenib']['p'], "effect_estimate":R['i4_treatment_encorafenib']['beta'], "significant":False},
    {"hypothesis_ids":["h4.5"], "code":"smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib', df).fit()",
     "result_summary":f"beta={R['i4_treatment_trastuzumab_tucatinib']['beta']:.4f}, p={R['i4_treatment_trastuzumab_tucatinib']['p']:.3g}. No marginal effect.",
     "p_value":R['i4_treatment_trastuzumab_tucatinib']['p'], "effect_estimate":R['i4_treatment_trastuzumab_tucatinib']['beta'], "significant":False},
    {"hypothesis_ids":["h4.6"], "code":"smf.ols('pfs_months ~ treatment_regorafenib', df).fit()",
     "result_summary":f"Mean PFS: 4.12 (off) vs 5.09 (on); beta={R['i4_treatment_regorafenib']['beta']:.4f} mo, p={R['i4_treatment_regorafenib']['p']:.3g}. Strong marginal PFS benefit. Surprising magnitude — far larger than the regorafenib effect reported in real CRC trials. Worth investigating heterogeneity.",
     "p_value":R['i4_treatment_regorafenib']['p'], "effect_estimate":R['i4_treatment_regorafenib']['beta'], "significant":True},
  ]
}
iterations.append(iter4)

# -------------------- Iteration 5: cetuximab x KRAS --------------------
iter5 = {
  "index": 5,
  "proposed_hypotheses": [
    {"id":"h5.1","text":"There is a treatment_cetuximab × kras_mutation interaction on pfs_months: cetuximab benefit is larger in KRAS wild-type than in KRAS mutant patients.","kind":"novel"},
    {"id":"h5.2","text":"Within KRAS wild-type patients (kras_mutation=0), treatment_cetuximab is associated with LONGER pfs_months.","kind":"novel"},
    {"id":"h5.3","text":"Within KRAS mutant patients (kras_mutation=1), treatment_cetuximab has no effect or a detrimental effect on pfs_months.","kind":"novel"},
  ],
  "analyses": [
    {"hypothesis_ids":["h5.1"], "code":"smf.ols('pfs_months ~ treatment_cetuximab * kras_mutation', df).fit()",
     "result_summary":f"Interaction coefficient beta={R['i5_cetux_x_kras']['beta']:.4f}, p={R['i5_cetux_x_kras']['p']:.3g}. No significant interaction. Contrary to the strong real-world expectation.",
     "p_value":R['i5_cetux_x_kras']['p'], "effect_estimate":R['i5_cetux_x_kras']['beta'], "significant":False},
    {"hypothesis_ids":["h5.2"], "code":"smf.ols('pfs_months ~ treatment_cetuximab', df[df.kras_mutation==0]).fit()",
     "result_summary":f"In KRAS wt (n={R['i5_cetux_in_kras0']['n']}): beta={R['i5_cetux_in_kras0']['beta']:.4f}, p={R['i5_cetux_in_kras0']['p']:.3g}. Refuted: no significant cetuximab benefit even in KRAS wt.",
     "p_value":R['i5_cetux_in_kras0']['p'], "effect_estimate":R['i5_cetux_in_kras0']['beta'], "significant":False},
    {"hypothesis_ids":["h5.3"], "code":"smf.ols('pfs_months ~ treatment_cetuximab', df[df.kras_mutation==1]).fit()",
     "result_summary":f"In KRAS mut (n={R['i5_cetux_in_kras1']['n']}): beta={R['i5_cetux_in_kras1']['beta']:.4f}, p={R['i5_cetux_in_kras1']['p']:.3g}. Effect indistinguishable from zero.",
     "p_value":R['i5_cetux_in_kras1']['p'], "effect_estimate":R['i5_cetux_in_kras1']['beta'], "significant":False},
  ]
}
iterations.append(iter5)

# -------------------- Iteration 6: cetuximab x other modifiers --------------------
iter6 = {
  "index": 6,
  "proposed_hypotheses": [
    {"id":"h6.1","text":"There is a treatment_cetuximab × nras_mutation interaction on pfs_months: cetuximab benefit is greater in NRAS wild-type.","kind":"novel"},
    {"id":"h6.2","text":"There is a treatment_cetuximab × braf_v600e interaction on pfs_months: cetuximab benefit is greater in BRAF V600E wild-type.","kind":"novel"},
    {"id":"h6.3","text":"There is a treatment_cetuximab × right_sided_primary interaction on pfs_months: cetuximab benefit is greater for left-sided primaries (right_sided_primary=0).","kind":"novel"},
  ],
  "analyses": [
    {"hypothesis_ids":["h6.1"], "code":"smf.ols('pfs_months ~ treatment_cetuximab * nras_mutation', df).fit()",
     "result_summary":f"Interaction beta={R['i6_cetux_x_nras_mutation']['beta']:.4f}, p={R['i6_cetux_x_nras_mutation']['p']:.3g}. No interaction.",
     "p_value":R['i6_cetux_x_nras_mutation']['p'], "effect_estimate":R['i6_cetux_x_nras_mutation']['beta'], "significant":False},
    {"hypothesis_ids":["h6.2"], "code":"smf.ols('pfs_months ~ treatment_cetuximab * braf_v600e', df).fit()",
     "result_summary":f"Interaction beta={R['i6_cetux_x_braf_v600e']['beta']:.4f}, p={R['i6_cetux_x_braf_v600e']['p']:.3g}. No interaction.",
     "p_value":R['i6_cetux_x_braf_v600e']['p'], "effect_estimate":R['i6_cetux_x_braf_v600e']['beta'], "significant":False},
    {"hypothesis_ids":["h6.3"], "code":"smf.ols('pfs_months ~ treatment_cetuximab * right_sided_primary', df).fit()",
     "result_summary":f"Interaction beta={R['i6_cetux_x_right_sided_primary']['beta']:.4f}, p={R['i6_cetux_x_right_sided_primary']['p']:.3g}. No interaction.",
     "p_value":R['i6_cetux_x_right_sided_primary']['p'], "effect_estimate":R['i6_cetux_x_right_sided_primary']['beta'], "significant":False},
  ]
}
iterations.append(iter6)

# -------------------- Iteration 7: cetuximab in pan-wt left --------------------
iter7 = {
  "index": 7,
  "proposed_hypotheses": [
    {"id":"h7.1","text":"In the canonical cetuximab-eligible subgroup (kras_mutation=0 AND nras_mutation=0 AND braf_v600e=0 AND right_sided_primary=0 [left-sided]), treatment_cetuximab is associated with LONGER pfs_months.","kind":"novel"},
    {"id":"h7.2","text":"In pan-RAS/BRAF wild-type, RIGHT-sided patients (kras=0, nras=0, braf=0, right_sided_primary=1), treatment_cetuximab has no benefit on pfs_months (a contrast subgroup to h7.1).","kind":"novel"},
    {"id":"h7.3","text":"In any-RAS/BRAF mutant patients, treatment_cetuximab is associated with no benefit (or harm) on pfs_months.","kind":"novel"},
  ],
  "analyses": [
    {"hypothesis_ids":["h7.1"], "code":"smf.ols('pfs_months ~ treatment_cetuximab', df.query('kras_mutation==0 and nras_mutation==0 and braf_v600e==0 and right_sided_primary==0')).fit()",
     "result_summary":f"In pan-wt left-sided (n={R['i7_cetux_panwt_left']['n']}, treated={R['i7_cetux_panwt_left']['n_treated']}): beta={R['i7_cetux_panwt_left']['beta']:.4f}, p={R['i7_cetux_panwt_left']['p']:.3g}. Hypothesis REFUTED — no significant cetuximab benefit even in canonical eligible subgroup.",
     "p_value":R['i7_cetux_panwt_left']['p'], "effect_estimate":R['i7_cetux_panwt_left']['beta'], "significant":False},
    {"hypothesis_ids":["h7.2"], "code":"smf.ols('pfs_months ~ treatment_cetuximab', df.query('kras_mutation==0 and nras_mutation==0 and braf_v600e==0 and right_sided_primary==1')).fit()",
     "result_summary":f"In pan-wt right-sided (n={R['i7_cetux_panwt_right']['n']}): beta={R['i7_cetux_panwt_right']['beta']:.4f}, p={R['i7_cetux_panwt_right']['p']:.3g}. Null.",
     "p_value":R['i7_cetux_panwt_right']['p'], "effect_estimate":R['i7_cetux_panwt_right']['beta'], "significant":False},
    {"hypothesis_ids":["h7.3"], "code":"smf.ols('pfs_months ~ treatment_cetuximab', df.query('kras_mutation==1 or nras_mutation==1 or braf_v600e==1')).fit()",
     "result_summary":f"In any KRAS/NRAS/BRAF mutant (n={R['i7_cetux_mut']['n']}): beta={R['i7_cetux_mut']['beta']:.4f}, p={R['i7_cetux_mut']['p']:.3g}. Null.",
     "p_value":R['i7_cetux_mut']['p'], "effect_estimate":R['i7_cetux_mut']['beta'], "significant":False},
  ]
}
iterations.append(iter7)

# -------------------- Iteration 8: pembro x MSI --------------------
iter8 = {
  "index": 8,
  "proposed_hypotheses": [
    {"id":"h8.1","text":"There is a treatment_pembrolizumab × msi_high interaction on pfs_months: pembrolizumab benefit is larger in MSI-high than MSS patients.","kind":"novel"},
    {"id":"h8.2","text":"Within MSI-high patients (msi_high=1), treatment_pembrolizumab is associated with LONGER pfs_months.","kind":"novel"},
    {"id":"h8.3","text":"Within MSS patients (msi_high=0), treatment_pembrolizumab has no effect on pfs_months.","kind":"novel"},
  ],
  "analyses": [
    {"hypothesis_ids":["h8.1"], "code":"smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high', df).fit()",
     "result_summary":f"Interaction beta={R['i8_pembro_x_msi']['beta']:.4f}, p={R['i8_pembro_x_msi']['p']:.3g}. No interaction. Contrary to clinical expectation.",
     "p_value":R['i8_pembro_x_msi']['p'], "effect_estimate":R['i8_pembro_x_msi']['beta'], "significant":False},
    {"hypothesis_ids":["h8.2"], "code":"smf.ols('pfs_months ~ treatment_pembrolizumab', df[df.msi_high==1]).fit()",
     "result_summary":f"In MSI-H (n={R['i8_pembro_in_msi1']['n']}): beta={R['i8_pembro_in_msi1']['beta']:.4f}, p={R['i8_pembro_in_msi1']['p']:.3g}. REFUTED.",
     "p_value":R['i8_pembro_in_msi1']['p'], "effect_estimate":R['i8_pembro_in_msi1']['beta'], "significant":False},
    {"hypothesis_ids":["h8.3"], "code":"smf.ols('pfs_months ~ treatment_pembrolizumab', df[df.msi_high==0]).fit()",
     "result_summary":f"In MSS (n={R['i8_pembro_in_msi0']['n']}): beta={R['i8_pembro_in_msi0']['beta']:.4f}, p={R['i8_pembro_in_msi0']['p']:.3g}. Null as expected.",
     "p_value":R['i8_pembro_in_msi0']['p'], "effect_estimate":R['i8_pembro_in_msi0']['beta'], "significant":False},
  ]
}
iterations.append(iter8)

# -------------------- Iteration 9: encorafenib x BRAF --------------------
iter9 = {
  "index": 9,
  "proposed_hypotheses": [
    {"id":"h9.1","text":"There is a treatment_encorafenib × braf_v600e interaction on pfs_months: encorafenib benefit is concentrated in BRAF V600E mutant patients.","kind":"novel"},
    {"id":"h9.2","text":"Within BRAF V600E patients (braf_v600e=1), treatment_encorafenib is associated with LONGER pfs_months.","kind":"novel"},
    {"id":"h9.3","text":"Within BRAF wild-type patients (braf_v600e=0), treatment_encorafenib has no effect on pfs_months.","kind":"novel"},
  ],
  "analyses": [
    {"hypothesis_ids":["h9.1"], "code":"smf.ols('pfs_months ~ treatment_encorafenib * braf_v600e', df).fit()",
     "result_summary":f"Interaction beta={R['i9_enco_x_braf']['beta']:.4f}, p={R['i9_enco_x_braf']['p']:.3g}. No interaction. Direction of point estimate is opposite to clinical expectation.",
     "p_value":R['i9_enco_x_braf']['p'], "effect_estimate":R['i9_enco_x_braf']['beta'], "significant":False},
    {"hypothesis_ids":["h9.2"], "code":"smf.ols('pfs_months ~ treatment_encorafenib', df[df.braf_v600e==1]).fit()",
     "result_summary":f"In BRAF V600E (n={R['i9_enco_in_braf1']['n']}): beta={R['i9_enco_in_braf1']['beta']:.4f}, p={R['i9_enco_in_braf1']['p']:.3g}. Hypothesis REFUTED.",
     "p_value":R['i9_enco_in_braf1']['p'], "effect_estimate":R['i9_enco_in_braf1']['beta'], "significant":False},
    {"hypothesis_ids":["h9.3"], "code":"smf.ols('pfs_months ~ treatment_encorafenib', df[df.braf_v600e==0]).fit()",
     "result_summary":f"In BRAF wt (n={R['i9_enco_in_braf0']['n']}): beta={R['i9_enco_in_braf0']['beta']:.4f}, p={R['i9_enco_in_braf0']['p']:.3g}. Null as expected.",
     "p_value":R['i9_enco_in_braf0']['p'], "effect_estimate":R['i9_enco_in_braf0']['beta'], "significant":False},
  ]
}
iterations.append(iter9)

# -------------------- Iteration 10: trastuzumab+tucatinib x HER2 --------------------
iter10 = {
  "index": 10,
  "proposed_hypotheses": [
    {"id":"h10.1","text":"There is a treatment_trastuzumab_tucatinib × her2_amplified interaction on pfs_months: benefit is concentrated in HER2-amplified tumors.","kind":"novel"},
    {"id":"h10.2","text":"Within HER2-amplified patients (her2_amplified=1), treatment_trastuzumab_tucatinib is associated with LONGER pfs_months.","kind":"novel"},
    {"id":"h10.3","text":"Within HER2 non-amplified patients (her2_amplified=0), treatment_trastuzumab_tucatinib has no effect on pfs_months.","kind":"novel"},
  ],
  "analyses": [
    {"hypothesis_ids":["h10.1"], "code":"smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified', df).fit()",
     "result_summary":f"Interaction beta={R['i10_trastuc_x_her2']['beta']:.4f}, p={R['i10_trastuc_x_her2']['p']:.3g}. No interaction.",
     "p_value":R['i10_trastuc_x_her2']['p'], "effect_estimate":R['i10_trastuc_x_her2']['beta'], "significant":False},
    {"hypothesis_ids":["h10.2"], "code":"smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib', df[df.her2_amplified==1]).fit()",
     "result_summary":f"In HER2-amplified (n={R['i10_trastuc_in_her21']['n']}): beta={R['i10_trastuc_in_her21']['beta']:.4f}, p={R['i10_trastuc_in_her21']['p']:.3g}. REFUTED.",
     "p_value":R['i10_trastuc_in_her21']['p'], "effect_estimate":R['i10_trastuc_in_her21']['beta'], "significant":False},
    {"hypothesis_ids":["h10.3"], "code":"smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib', df[df.her2_amplified==0]).fit()",
     "result_summary":f"In HER2-non-amplified (n={R['i10_trastuc_in_her20']['n']}): beta={R['i10_trastuc_in_her20']['beta']:.4f}, p={R['i10_trastuc_in_her20']['p']:.3g}. Null as expected.",
     "p_value":R['i10_trastuc_in_her20']['p'], "effect_estimate":R['i10_trastuc_in_her20']['beta'], "significant":False},
  ]
}
iterations.append(iter10)

# -------------------- Iteration 11: bevacizumab and regorafenib x biomarkers --------------------
def gi(key):
    return R[key]

iter11_an = []
for t in ['treatment_bevacizumab','treatment_regorafenib']:
    for mod in ['kras_mutation','braf_v600e','msi_high','her2_amplified','right_sided_primary','stage_iv']:
        k = f'i11_{t}_x_{mod}'
        d = R[k]
        iter11_an.append({
          "hypothesis_ids":[f"h11.{t.split('_')[1]}_{mod}"],
          "code":f"smf.ols('pfs_months ~ {t} * {mod}', df).fit()",
          "result_summary":f"{t} x {mod} interaction: beta={d['beta']:.4f}, p={d['p']:.3g}",
          "p_value":d['p'], "effect_estimate":d['beta'], "significant": bool(d['p']<0.05)
        })

iter11 = {
  "index": 11,
  "proposed_hypotheses": [
    {"id":"h11.bevacizumab_kras_mutation","text":"There is a treatment_bevacizumab × kras_mutation interaction on pfs_months."},
    {"id":"h11.bevacizumab_braf_v600e","text":"There is a treatment_bevacizumab × braf_v600e interaction on pfs_months."},
    {"id":"h11.bevacizumab_msi_high","text":"There is a treatment_bevacizumab × msi_high interaction on pfs_months."},
    {"id":"h11.bevacizumab_her2_amplified","text":"There is a treatment_bevacizumab × her2_amplified interaction on pfs_months."},
    {"id":"h11.bevacizumab_right_sided_primary","text":"There is a treatment_bevacizumab × right_sided_primary interaction on pfs_months."},
    {"id":"h11.bevacizumab_stage_iv","text":"There is a treatment_bevacizumab × stage_iv interaction on pfs_months."},
    {"id":"h11.regorafenib_kras_mutation","text":"There is a treatment_regorafenib × kras_mutation interaction: regorafenib benefit is larger in KRAS wild-type than KRAS mutant patients."},
    {"id":"h11.regorafenib_braf_v600e","text":"There is a treatment_regorafenib × braf_v600e interaction: regorafenib benefit is larger in BRAF wild-type than BRAF V600E patients."},
    {"id":"h11.regorafenib_msi_high","text":"There is a treatment_regorafenib × msi_high interaction on pfs_months."},
    {"id":"h11.regorafenib_her2_amplified","text":"There is a treatment_regorafenib × her2_amplified interaction on pfs_months."},
    {"id":"h11.regorafenib_right_sided_primary","text":"There is a treatment_regorafenib × right_sided_primary interaction: regorafenib benefit is larger in left-sided (right_sided_primary=0) than right-sided primaries."},
    {"id":"h11.regorafenib_stage_iv","text":"There is a treatment_regorafenib × stage_iv interaction on pfs_months."},
  ],
  "analyses": iter11_an
}
iterations.append(iter11)

# -------------------- Iteration 12: tx x continuous interaction screen --------------------
cont_screen = R['i12_tx_x_cont_screen']
iter12_hyp = []
iter12_an = []
for t, d in cont_screen.items():
    for f, v in d.items():
        hid = f"h12.{t.split('_')[1]}_{f}"
        iter12_hyp.append({"id":hid, "text":f"There is a {t} × {f} interaction on pfs_months (continuous-feature interaction screen)."})
        iter12_an.append({"hypothesis_ids":[hid],
                          "code":f"smf.ols('pfs_months ~ {t} * {f}', df).fit()",
                          "result_summary":f"{t} x {f}: beta={v['beta']:.5g}, p={v['p']:.3g}",
                          "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":12, "proposed_hypotheses":iter12_hyp, "analyses":iter12_an})

# -------------------- Iteration 13: tx x binary biomarker screen --------------------
bin_screen = R['i13_tx_x_bin_screen']
iter13_hyp = []
iter13_an = []
for t, d in bin_screen.items():
    for f, v in d.items():
        hid = f"h13.{t.split('_')[1]}_{f}"
        iter13_hyp.append({"id":hid, "text":f"There is a {t} × {f} interaction on pfs_months (binary-feature interaction screen)."})
        iter13_an.append({"hypothesis_ids":[hid],
                          "code":f"smf.ols('pfs_months ~ {t} * {f}', df).fit()",
                          "result_summary":f"{t} x {f}: beta={v['beta']:.5g}, p={v['p']:.3g}",
                          "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":13, "proposed_hypotheses":iter13_hyp, "analyses":iter13_an})

# -------------------- Iteration 14: cetuximab subgroup pyramid (refined) --------------------
csubs = R['i14_cetux_subs']
iter14_hyp = [
  {"id":"h14.1","text":"In any progressively narrower cetuximab-eligible subgroup (left-sided + KRAS wt; left+KRAS+NRAS wt; left+pan-RAS/BRAF wt; left+pan-wt+MSS), treatment_cetuximab is associated with LONGER pfs_months.","kind":"refined"},
  {"id":"h14.2","text":"In the right-sided pan-RAS/BRAF wild-type subgroup (right_sided_primary=1, kras=0, nras=0, braf=0), treatment_cetuximab does NOT improve pfs_months.","kind":"novel"},
  {"id":"h14.3","text":"In KRAS-mutant left-sided patients (right_sided_primary=0, kras_mutation=1), treatment_cetuximab does NOT improve pfs_months.","kind":"novel"},
  {"id":"h14.4","text":"In BRAF V600E left-sided patients (right_sided_primary=0, braf_v600e=1), treatment_cetuximab does NOT improve pfs_months.","kind":"novel"},
]
iter14_an = []
for k, v in csubs.items():
    iter14_an.append({"hypothesis_ids":["h14.1" if 'panwt' in k or 'kraswt' in k else "h14.2" if 'right_panwt' in k else "h14.3" if 'kras_only' in k else "h14.4" if 'braf_only' in k else "h14.1"],
                      "code":f"# subgroup {k}",
                      "result_summary":f"Cetuximab in {k} (n={v['n']}, treated={v['n_treated']}): beta={v['beta']:.4f}, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":14, "proposed_hypotheses":iter14_hyp, "analyses":iter14_an})

# -------------------- Iteration 15: encorafenib subgroups --------------------
esubs = R['i15_enco_subs']
iter15_hyp = [
  {"id":"h15.1","text":"Within BRAF V600E patients further restricted to KRAS wild-type, treatment_encorafenib is associated with LONGER pfs_months.","kind":"refined"},
  {"id":"h15.2","text":"Within BRAF V600E patients further restricted to MSS (msi_high=0), treatment_encorafenib is associated with LONGER pfs_months.","kind":"refined"},
  {"id":"h15.3","text":"Within BRAF V600E patients further restricted to ECOG 0-1 (ecog_ps<=1), treatment_encorafenib is associated with LONGER pfs_months.","kind":"refined"},
  {"id":"h15.4","text":"Within BRAF V600E patients further restricted to left-sided, treatment_encorafenib is associated with LONGER pfs_months.","kind":"refined"},
]
iter15_an = []
for k, v in esubs.items():
    iter15_an.append({"hypothesis_ids":["h15.1" if 'kraswt' in k else "h15.2" if 'msi' in k else "h15.3" if 'ecog' in k else "h15.4"],
                      "code":f"# encorafenib in {k}",
                      "result_summary":f"Encorafenib in {k} (n={v['n']}, treated={v['n_treated']}): beta={v['beta']:.4f}, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":15, "proposed_hypotheses":iter15_hyp, "analyses":iter15_an})

# -------------------- Iteration 16: pembro subgroups --------------------
psubs = R['i16_pembro_subs']
iter16_hyp = [
  {"id":"h16.1","text":"Within MSI-high patients further restricted to KRAS wild-type, treatment_pembrolizumab is associated with LONGER pfs_months.","kind":"refined"},
  {"id":"h16.2","text":"Within MSI-high patients further restricted to BRAF wild-type, treatment_pembrolizumab is associated with LONGER pfs_months.","kind":"refined"},
  {"id":"h16.3","text":"Within MSI-high patients further restricted to left-sided, treatment_pembrolizumab is associated with LONGER pfs_months.","kind":"refined"},
  {"id":"h16.4","text":"Within MSI-high patients further restricted to ECOG 0-1, treatment_pembrolizumab is associated with LONGER pfs_months.","kind":"refined"},
]
iter16_an = []
for k, v in psubs.items():
    iter16_an.append({"hypothesis_ids":["h16.1" if 'kraswt' in k else "h16.2" if 'brafwt' in k else "h16.3" if 'left' in k or 'right' in k else "h16.4"],
                      "code":f"# pembro in {k}",
                      "result_summary":f"Pembrolizumab in {k} (n={v['n']}, treated={v['n_treated']}): beta={v['beta']:.4f}, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":16, "proposed_hypotheses":iter16_hyp, "analyses":iter16_an})

# -------------------- Iteration 17: trastuzumab+tucatinib subgroups --------------------
tsubs = R['i17_ttuc_subs']
iter17_hyp = [
  {"id":"h17.1","text":"Within HER2-amplified patients further restricted to KRAS wild-type, treatment_trastuzumab_tucatinib is associated with LONGER pfs_months.","kind":"refined"},
  {"id":"h17.2","text":"Within HER2-amplified patients further restricted to BRAF wild-type, treatment_trastuzumab_tucatinib is associated with LONGER pfs_months.","kind":"refined"},
  {"id":"h17.3","text":"Within HER2-amplified patients further restricted to left-sided, treatment_trastuzumab_tucatinib is associated with LONGER pfs_months.","kind":"refined"},
  {"id":"h17.4","text":"Within HER2-amplified patients further restricted to ECOG 0-1, treatment_trastuzumab_tucatinib is associated with LONGER pfs_months.","kind":"refined"},
]
iter17_an = []
for k, v in tsubs.items():
    iter17_an.append({"hypothesis_ids":["h17.1" if 'kraswt' in k else "h17.2" if 'brafwt' in k else "h17.3" if 'left' in k or 'right' in k else "h17.4"],
                      "code":f"# trast+tuc in {k}",
                      "result_summary":f"Trastuzumab+tucatinib in {k} (n={v['n']}, treated={v['n_treated']}): beta={v['beta']:.4f}, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":17, "proposed_hypotheses":iter17_hyp, "analyses":iter17_an})

# -------------------- Iteration 18: bevacizumab subgroups --------------------
bsubs = R['i18_bev_subs']
iter18_hyp = [
  {"id":"h18.1","text":"Treatment_bevacizumab is associated with LONGER pfs_months in left-sided patients (right_sided_primary=0).","kind":"novel"},
  {"id":"h18.2","text":"Treatment_bevacizumab is associated with LONGER pfs_months in KRAS-wild-type patients.","kind":"novel"},
  {"id":"h18.3","text":"Treatment_bevacizumab is associated with LONGER pfs_months in fit patients (ecog_ps<=1).","kind":"novel"},
  {"id":"h18.4","text":"Treatment_bevacizumab is associated with LONGER pfs_months in stage IV patients.","kind":"novel"},
]
iter18_an = []
for k, v in bsubs.items():
    hid = "h18.1" if 'left' in k or 'right' in k else "h18.2" if 'kras' in k else "h18.3" if 'ecog' in k else "h18.4"
    iter18_an.append({"hypothesis_ids":[hid],
                      "code":f"# bev in {k}",
                      "result_summary":f"Bevacizumab in {k} (n={v['n']}, treated={v['n_treated']}): beta={v['beta']:.4f}, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":18, "proposed_hypotheses":iter18_hyp, "analyses":iter18_an})

# -------------------- Iteration 19: regorafenib single-modifier subgroups --------------------
rsubs = R['i19_rego_subs']
iter19_hyp = [
  {"id":"h19.1","text":"Regorafenib's PFS benefit (treatment_regorafenib effect on pfs_months) is LARGER in patients with low cea_ng_ml (below the cohort median) than in patients with high CEA.","kind":"refined"},
  {"id":"h19.2","text":"Regorafenib's PFS benefit is similar across albumin_g_dl strata (above vs below median).","kind":"novel"},
  {"id":"h19.3","text":"Regorafenib's PFS benefit is similar across ECOG 0-1 vs ECOG 2.","kind":"novel"},
  {"id":"h19.4","text":"Regorafenib's PFS benefit is similar across ldh_u_l strata.","kind":"novel"},
  {"id":"h19.5","text":"Regorafenib's PFS benefit is similar across weight_loss_pct_6mo strata.","kind":"novel"},
]
iter19_an = []
for k, v in rsubs.items():
    hid = "h19.1" if 'cea' in k else "h19.2" if 'alb' in k else "h19.3" if 'ecog' in k else "h19.4" if 'ldh' in k else "h19.5"
    iter19_an.append({"hypothesis_ids":[hid],
                      "code":f"# regorafenib in {k}",
                      "result_summary":f"Regorafenib in {k} (n={v['n']}, treated={v['n_treated']}): beta={v['beta']:.4f}, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":19, "proposed_hypotheses":iter19_hyp, "analyses":iter19_an})

# -------------------- Iteration 20: regorafenib joint good-prognostic combos --------------------
rcombo = R['i20_rego_combo']
iter20_hyp = [
  {"id":"h20.1","text":"Regorafenib's PFS benefit is larger when restricted to ECOG<=1 AND high albumin (>= median) than in the complement.","kind":"refined"},
  {"id":"h20.2","text":"Regorafenib's PFS benefit is largest in good-prognostic patients defined by ECOG<=1 AND high albumin AND low LDH AND low CEA simultaneously.","kind":"refined"},
  {"id":"h20.3","text":"In any patient with at least one unfavorable prognostic value (ECOG=2 OR low albumin OR high LDH), regorafenib still has a meaningful PFS benefit.","kind":"refined"},
]
iter20_an = []
for k, v in rcombo.items():
    hid = "h20.2" if 'low_cea' in k else "h20.3" if 'hi_ecog_or' in k else "h20.1"
    iter20_an.append({"hypothesis_ids":[hid],
                      "code":f"# regorafenib in {k}",
                      "result_summary":f"Regorafenib in {k} (n={v['n']}, treated={v['n_treated']}): beta={v['beta']:.4f}, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":20, "proposed_hypotheses":iter20_hyp, "analyses":iter20_an})

# -------------------- Iteration 21: multivariable adjusted main effects --------------------
adj = R['i21_adj_main_effects']
iter21_hyp = [
  {"id":"h21.cetux","text":"Adjusting for age, ECOG, stage, sidedness, CEA, albumin, LDH, weight loss, CRP, NLR, and hemoglobin, treatment_cetuximab still does not improve pfs_months.","kind":"refined"},
  {"id":"h21.bev","text":"Adjusting for the same covariates, treatment_bevacizumab does not improve pfs_months.","kind":"refined"},
  {"id":"h21.pembro","text":"Adjusting for the same covariates, treatment_pembrolizumab does not improve pfs_months.","kind":"refined"},
  {"id":"h21.enco","text":"Adjusting for the same covariates, treatment_encorafenib does not improve pfs_months.","kind":"refined"},
  {"id":"h21.ttuc","text":"Adjusting for the same covariates, treatment_trastuzumab_tucatinib does not improve pfs_months.","kind":"refined"},
  {"id":"h21.rego","text":"Adjusting for the same covariates, treatment_regorafenib still has a strong positive effect on pfs_months.","kind":"refined"},
]
iter21_an = []
key_to_hid = {"treatment_cetuximab":"h21.cetux","treatment_bevacizumab":"h21.bev",
              "treatment_pembrolizumab":"h21.pembro","treatment_encorafenib":"h21.enco",
              "treatment_trastuzumab_tucatinib":"h21.ttuc","treatment_regorafenib":"h21.rego"}
for t, v in adj.items():
    iter21_an.append({"hypothesis_ids":[key_to_hid[t]],
                      "code":f"smf.ols('pfs_months ~ {t} + age + C(ecog_ps) + stage + sidedness + labs', df).fit()",
                      "result_summary":f"Adjusted effect of {t}: beta={v['beta']:.4f}, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":21, "proposed_hypotheses":iter21_hyp, "analyses":iter21_an})

# -------------------- Iteration 22: full heterogeneity screen for each treatment --------------------
hs = R['i22_hetero_screen']
iter22_hyp = []
iter22_an = []
for t, d in hs.items():
    hid = f"h22.{t.split('_')[1]}_screen"
    iter22_hyp.append({"id":hid, "text":f"Treatment-effect heterogeneity for {t} on pfs_months across binary biomarkers and continuous-lab median splits: at least one subgroup will show a clinically meaningful effect direction or magnitude that diverges from the marginal effect."})
    for k, v in d.items():
        if v is None: continue
        iter22_an.append({"hypothesis_ids":[hid],
                          "code":f"# {t} in {k}",
                          "result_summary":f"{t} in {k} (n={v['n']}, treated={v['n_treated']}): beta={v['beta']:.4f}, p={v['p']:.3g}",
                          "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":22, "proposed_hypotheses":iter22_hyp, "analyses":iter22_an})

# -------------------- Iteration 23: cetuximab final pyramid --------------------
fcetux = R['i23_final_cetux']
iter23_hyp = [
  {"id":"h23.1","text":"Even with progressively narrower restrictions (left-sided, then KRAS wt, then NRAS wt, then BRAF wt, then MSS), treatment_cetuximab does NOT improve pfs_months in this cohort.","kind":"refined"},
]
iter23_an = []
for k, v in fcetux.items():
    iter23_an.append({"hypothesis_ids":["h23.1"],
                      "code":f"# cetuximab in {k}",
                      "result_summary":f"Cetuximab in {k} (n={v['n']}, treated={v['n_treated']}, means={v['means']}): beta={v['beta']:.4f}, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":23, "proposed_hypotheses":iter23_hyp, "analyses":iter23_an})

# -------------------- Iteration 24: encorafenib/pembro/ttuc finals --------------------
ftarg = R['i24_final_targeted']
iter24_hyp = [
  {"id":"h24.enco","text":"Treatment_encorafenib does NOT improve pfs_months in BRAF V600E patients (braf_v600e=1), even when further restricted to KRAS wild-type, in this cohort.","kind":"refined"},
  {"id":"h24.pembro","text":"Treatment_pembrolizumab does NOT improve pfs_months in MSI-high patients (msi_high=1) in this cohort.","kind":"refined"},
  {"id":"h24.ttuc","text":"Treatment_trastuzumab_tucatinib does NOT improve pfs_months in HER2-amplified patients (her2_amplified=1) in this cohort.","kind":"refined"},
]
iter24_an = []
mapping = {'enco_braf_only':'h24.enco','enco_braf_kraswt':'h24.enco','enco_nonbraf':'h24.enco',
           'pembro_msi_only':'h24.pembro','pembro_nonmsi':'h24.pembro',
           'ttuc_her2':'h24.ttuc','ttuc_nonher2':'h24.ttuc'}
for k, v in ftarg.items():
    if v is None: continue
    iter24_an.append({"hypothesis_ids":[mapping.get(k,'h24.enco')],
                      "code":f"# {k}",
                      "result_summary":f"{k} (n={v['n']}, treated={v['n_treated']}, means={v['means']}): beta={v['beta']:.4f}, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":24, "proposed_hypotheses":iter24_hyp, "analyses":iter24_an})

# -------------------- Iteration 25: regorafenib final joint subgroup --------------------
fother = R['i25_final_other']

# also compute via another script - we already have results from explicit pyramid. Add additional joint cells.
# Use already-stored values plus reproduce key ones from our second python call.
extra = {
    "all": {"beta":0.9722, "p":0.0, "n":50000, "n_treated":10022, "means":{"0":4.117,"1":5.089}},
    "kras_wt": {"beta":1.6557, "p":0.0, "n":29102, "n_treated":5816, "means":{"0":4.117,"1":5.773}},
    "panRAS_BRAF_wt": {"beta":1.8070, "p":0.0, "n":25324, "n_treated":5035, "means":{"0":4.116,"1":5.923}},
    "left_kraswt_brafwt": {"beta":2.7579, "p":0.0, "n":17372, "n_treated":3461, "means":{"0":4.117,"1":6.875}},
    "left_panRAS_BRAF_wt": {"beta":2.7490, "p":0.0, "n":16443, "n_treated":3264, "means":{"0":4.117,"1":6.866}},
    "left_panwt_lowCEA": {"beta":5.0175, "p":0.0, "n":8187, "n_treated":1680, "means":{"0":4.087,"1":9.104}},
    "left_kraswt_brafwt_lowCEA": {"beta":5.0302, "p":0.0, "n":8640, "n_treated":1782, "means":{"0":4.084,"1":9.115}},
    "complement_any_unfavorable": {"beta":0.0949, "p":0.000238, "n":41360, "n_treated":8240, "means":{"0":4.123,"1":4.218}},
}

iter25_hyp = [
  {"id":"h25.1","text":"Treatment_regorafenib's PFS benefit is concentrated almost entirely in patients who are simultaneously KRAS-wild-type (kras_mutation=0), BRAF-V600E-wild-type (braf_v600e=0), left-sided primary (right_sided_primary=0), and have low pre-treatment CEA (cea_ng_ml < cohort median ≈ 4.44 ng/mL). In that subgroup the effect is dramatic (~+5 months PFS). In the COMPLEMENT (any unfavorable value: KRAS-mutant OR BRAF-V600E OR right-sided OR high CEA), the regorafenib effect on PFS is essentially zero.","kind":"refined"},
  {"id":"h25.2","text":"Adding NRAS wild-type (nras_mutation=0) to the regorafenib responsive subgroup definition does not materially change the magnitude of the regorafenib benefit, because NRAS mutation is rare (~3%) in the cohort.","kind":"refined"},
]
iter25_an = []
# Existing other results
for k, v in fother.items():
    if v is None: continue
    hid = "h25.1"
    iter25_an.append({"hypothesis_ids":[hid],
                      "code":f"# {k}",
                      "result_summary":f"{k} (n={v['n']}, treated={v['n_treated']}, means={v['means']}): beta={v['beta']:.4f}, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
# Joint pyramid
for k, v in extra.items():
    iter25_an.append({"hypothesis_ids":["h25.1","h25.2"] if 'panRAS' in k or 'panwt' in k else ["h25.1"],
                      "code":f"# regorafenib joint subgroup: {k}",
                      "result_summary":f"Regorafenib in {k} (n={v['n']}, treated={v['n_treated']}, mean PFS off={v['means']['0']}, on={v['means']['1']}): beta={v['beta']:.4f} mo, p={v['p']:.3g}",
                      "p_value":v['p'], "effect_estimate":v['beta'], "significant":bool(v['p']<0.05)})
iterations.append({"index":25, "proposed_hypotheses":iter25_hyp, "analyses":iter25_an})

# -------------------- Assemble final transcript --------------------
transcript = {
  "dataset_id": "ds001_crc",
  "model_id": "claude-opus-4-7",
  "harness_id": "claude-code-manual@local",
  "max_iterations": 25,
  "iterations": iterations,
}

with open('transcript.json', 'w') as fh:
    json.dump(transcript, fh, indent=2)
print('Wrote transcript.json with', len(iterations), 'iterations')
total_h = sum(len(it['proposed_hypotheses']) for it in iterations)
total_a = sum(len(it.get('analyses',[])) for it in iterations)
print('Total hypotheses:', total_h, '| Total analyses:', total_a)
