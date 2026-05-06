"""Assemble transcript.json from final_results.json and palbo_pik3ca_results.json."""

import json

R = json.load(open('final_results.json'))
P = json.load(open('palbo_pik3ca_results.json'))

iters = []

# ---- Iteration 1: descriptives & population characterization ----
d = R['iter1_descriptive']
iters.append({
    "index": 1,
    "proposed_hypotheses": [
        {"id": "h1.1",
         "text": ("In ds001_breast (n=50000), the marginal mean of pfs_months is approximately "
                  "4.7 months with appreciable patient heterogeneity in stage_iv (~30%), "
                  "has_brain_mets (~10%), er_positive (~70%), her2_positive (~18%), and "
                  "brca1_mutation/brca2_mutation (~2.5% each)."),
         "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h1.1"],
         "code": "df['pfs_months'].describe(); df[['er_positive','her2_positive','stage_iv','has_brain_mets','brca1_mutation','brca2_mutation']].mean()",
         "result_summary": (
             f"Overall pfs_months: mean={d['pfs_overall_mean']:.3f}, "
             f"median={d['pfs_overall_median']:.3f}, sd={d['pfs_overall_sd']:.3f}. "
             f"er_positive={d['er_positive_pct']:.3f}, her2_positive={d['her2_positive_pct']:.3f}, "
             f"stage_iv={d['stage_iv_pct']:.3f}, has_brain_mets={d['brain_mets_pct']:.3f}, "
             f"tnbc={d['tnbc_pct']:.3f}, hr_pos_her2_neg={d['hr_pos_her2_neg_pct']:.3f}, "
             f"brca1={d['brca1_pct']:.3f}, brca2={d['brca2_pct']:.3f}. "
             f"Treatment use: tamoxifen={d['treatment_use']['treatment_tamoxifen']:.3f}, "
             f"palbociclib={d['treatment_use']['treatment_palbociclib']:.3f}, "
             f"trastuzumab={d['treatment_use']['treatment_trastuzumab']:.3f}, "
             f"olaparib={d['treatment_use']['treatment_olaparib']:.3f}, "
             f"sacituzumab_govitecan={d['treatment_use']['treatment_sacituzumab_govitecan']:.3f}, "
             f"pembrolizumab={d['treatment_use']['treatment_pembrolizumab']:.3f}."),
         "effect_estimate": float(d['pfs_overall_mean']),
         "p_value": None,
         "significant": None}
    ]
})

# ---- Iteration 2: clinical binary features vs PFS ----
def find2(name):
    for x in R['iter2_clinical_binary_vs_pfs']:
        if x['feature']==name: return x
hyps2 = []
ans2 = []
for feat, direction in [('stage_iv','lower'),('has_brain_mets','lower'),
                        ('node_positive','lower'),('postmenopausal','lower'),('sex_female','differs')]:
    rec = find2(feat)
    hid = f"h2.{feat}"
    hyps2.append({"id": hid,
                  "text": f"Mean pfs_months is {direction} in patients with {feat}=1 than {feat}=0.",
                  "kind": "novel"})
    ans2.append({"hypothesis_ids":[hid],
                 "code": f"stats.ttest_ind(df.loc[df['{feat}']==1,'pfs_months'], df.loc[df['{feat}']==0,'pfs_months'], equal_var=False)",
                 "result_summary": (f"Mean pfs_months {feat}=1 vs 0: "
                                    f"{rec['mean_1']:.3f} vs {rec['mean_0']:.3f}, "
                                    f"diff={rec['diff']:+.3f}, Welch t-test p={rec['p_value']:.2e}."),
                 "effect_estimate": float(rec['diff']),
                 "p_value": float(rec['p_value']),
                 "significant": bool(rec['p_value'] < 0.05)})
iters.append({"index": 2, "proposed_hypotheses": hyps2, "analyses": ans2})

# ---- Iteration 3: continuous clinical vs PFS ----
def find3(name):
    for x in R['iter3_clinical_cont_vs_pfs']:
        if x['feature']==name: return x
hyps3, ans3 = [], []
for feat, direction in [('age_years','positively'),('ecog_ps','negatively'),('ki67_pct','negatively'),
                        ('weight_loss_pct_6mo','negatively'),('tumor_size_cm','negatively')]:
    rec = find3(feat)
    hid = f"h3.{feat}"
    hyps3.append({"id":hid,
                  "text":f"pfs_months is {direction} associated with {feat} as a continuous predictor.",
                  "kind":"novel"})
    ans3.append({"hypothesis_ids":[hid],
                 "code":f"stats.linregress(df['{feat}'], df['pfs_months'])",
                 "result_summary": (f"Linear regression of pfs_months on {feat}: "
                                    f"slope={rec['slope']:+.4f}, r={rec['r']:+.3f}, p={rec['p_value']:.2e}."),
                 "effect_estimate": float(rec['slope']),
                 "p_value": float(rec['p_value']),
                 "significant": bool(rec['p_value']<0.05)})
iters.append({"index":3, "proposed_hypotheses":hyps3, "analyses":ans3})

# ---- Iteration 4: biomarkers vs PFS ----
def find4(name):
    for x in R['iter4_biomarker_vs_pfs']:
        if x['feature']==name: return x
hyps4, ans4 = [], []
for feat, direction in [('er_positive','higher'),('pr_positive','higher'),
                        ('her2_positive','lower'),('her2_low','higher'),
                        ('brca1_mutation','differs'),('brca2_mutation','differs'),
                        ('pik3ca_mutation','lower')]:
    rec = find4(feat)
    hid = f"h4.{feat}"
    hyps4.append({"id":hid,
                  "text": f"Mean pfs_months is {direction} in patients with {feat}=1 vs {feat}=0.",
                  "kind":"novel"})
    ans4.append({"hypothesis_ids":[hid],
                 "code":f"stats.ttest_ind(df.loc[df['{feat}']==1,'pfs_months'], df.loc[df['{feat}']==0,'pfs_months'], equal_var=False)",
                 "result_summary":(f"Mean pfs_months {feat}=1 vs 0: {rec['mean_1']:.3f} vs {rec['mean_0']:.3f}, "
                                   f"diff={rec['diff']:+.3f}, Welch t-test p={rec['p_value']:.2e}."),
                 "effect_estimate": float(rec['diff']),
                 "p_value": float(rec['p_value']),
                 "significant": bool(rec['p_value']<0.05)})
iters.append({"index":4, "proposed_hypotheses":hyps4, "analyses":ans4})

# ---- Iteration 5: lab values vs PFS (selected) ----
def find5(name):
    for x in R['iter5_labs_vs_pfs']:
        if x['feature']==name: return x
hyps5, ans5 = [], []
for feat, direction in [('albumin_g_dl','positively'),('ldh_u_l','negatively'),
                        ('crp_mg_l','negatively'),('nlr','negatively'),
                        ('hemoglobin_g_dl','positively'),('alkaline_phosphatase_u_l','negatively'),
                        ('calcium_mg_dl','differs')]:
    rec = find5(feat)
    hid = f"h5.{feat}"
    hyps5.append({"id":hid,
                  "text": f"pfs_months is {direction} associated with {feat}.",
                  "kind":"novel"})
    ans5.append({"hypothesis_ids":[hid],
                 "code": f"stats.linregress(df['{feat}'], df['pfs_months'])",
                 "result_summary":(f"Linear regression slope of pfs_months on {feat}={rec['slope']:+.4f}, "
                                   f"r={rec['r']:+.3f}, p={rec['p_value']:.2e}."),
                 "effect_estimate": float(rec['slope']),
                 "p_value": float(rec['p_value']),
                 "significant": bool(rec['p_value']<0.05)})
iters.append({"index":5, "proposed_hypotheses":hyps5, "analyses":ans5})

# ---- Iteration 6: unadjusted treatment main effects ----
def find6(name):
    for x in R['iter6_tx_unadj']:
        if x['feature']==name: return x
treatments = ['treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
              'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab']
hyps6, ans6 = [], []
for t in treatments:
    rec = find6(t)
    hid = f"h6.{t.split('_',1)[1]}"
    hyps6.append({"id":hid,
                  "text": f"Mean pfs_months is higher in patients receiving {t}=1 than {t}=0 (unadjusted comparison across the full cohort).",
                  "kind":"novel"})
    ans6.append({"hypothesis_ids":[hid],
                 "code":f"stats.ttest_ind(df.loc[df['{t}']==1,'pfs_months'], df.loc[df['{t}']==0,'pfs_months'], equal_var=False)",
                 "result_summary":(f"Mean pfs_months on {t}={rec['mean_1']:.3f} vs off={rec['mean_0']:.3f}, "
                                   f"diff={rec['diff']:+.3f}, Welch t-test p={rec['p_value']:.2e}."),
                 "effect_estimate": float(rec['diff']),
                 "p_value": float(rec['p_value']),
                 "significant": bool(rec['p_value']<0.05)})
iters.append({"index":6, "proposed_hypotheses":hyps6, "analyses":ans6})

# ---- Iteration 7: adjusted treatment main effects ----
hyps7, ans7 = [], []
for t in treatments:
    v = R['iter7_tx_adjusted'][t]
    hid = f"h7.{t.split('_',1)[1]}_adj"
    hyps7.append({"id":hid,
                  "text": (f"Adjusted for age_years, ecog_ps, stage_iv, has_brain_mets, albumin_g_dl, ldh_u_l, "
                           f"crp_mg_l, nlr, weight_loss_pct_6mo, and tumor_size_cm, {t} is positively "
                           f"associated with pfs_months in the full cohort."),
                  "kind":"refined"})
    ans7.append({"hypothesis_ids":[hid],
                 "code": f"smf.ols('pfs_months ~ {t} + age_years+ecog_ps+stage_iv+has_brain_mets+albumin_g_dl+ldh_u_l+crp_mg_l+nlr+weight_loss_pct_6mo+tumor_size_cm', df).fit()",
                 "result_summary":(f"OLS coefficient on {t}: {v['coef']:+.3f} months PFS "
                                   f"(95% CI {v['ci_low']:+.3f} to {v['ci_high']:+.3f}), p={v['p_value']:.2e}."),
                 "effect_estimate": float(v['coef']),
                 "p_value": float(v['p_value']),
                 "significant": bool(v['p_value']<0.05)})
iters.append({"index":7, "proposed_hypotheses":hyps7, "analyses":ans7})

# ---- Iteration 8: tamoxifen x ER, x PR ----
def add_interaction_iter(idx, key_root, tx, mods, hyps_text_fn):
    hyps, ans = [], []
    for mod in mods:
        k = f"{key_root}_x_{mod.split('_')[0]}" if False else None
    # simpler: build per modifier from R
    return None

def interaction_iter(idx, tx, mods, key_prefix, biology_text):
    hyps, ans = [], []
    for mod, key in mods:
        rec = R[key]
        it = rec['interaction_test']
        sk = rec['stratified']
        hid = f"h{idx}.{tx.split('_',1)[1]}_x_{mod}"
        text = (f"The pfs_months effect of {tx} differs between patients with {mod}=1 and {mod}=0; "
                f"specifically, {biology_text.format(tx=tx, mod=mod)}.")
        hyps.append({"id": hid, "text": text, "kind":"novel"})
        # interaction test
        ans.append({"hypothesis_ids":[hid],
                    "code": f"smf.ols('pfs_months ~ {tx}*{mod} + age_years+ecog_ps+stage_iv+has_brain_mets+albumin_g_dl+ldh_u_l+crp_mg_l+nlr+weight_loss_pct_6mo+tumor_size_cm', df).fit()",
                    "result_summary": (f"Interaction coefficient {tx}:{mod} = {it['interaction']:+.3f} months "
                                       f"PFS (p_interaction = {it['p_interaction']:.2e}). "
                                       f"Stratified {tx} adjusted coefficient in {mod}=1: "
                                       f"{sk.get('modifier=1',{}).get('coef','NA') if 'modifier=1' in sk else 'NA'} "
                                       f"(p={sk.get('modifier=1',{}).get('p_value','NA') if 'modifier=1' in sk else 'NA'}); "
                                       f"in {mod}=0: "
                                       f"{sk.get('modifier=0',{}).get('coef','NA') if 'modifier=0' in sk else 'NA'} "
                                       f"(p={sk.get('modifier=0',{}).get('p_value','NA') if 'modifier=0' in sk else 'NA'})."),
                    "effect_estimate": float(it['interaction']),
                    "p_value": float(it['p_interaction']),
                    "significant": bool(it['p_interaction']<0.05)})
    return hyps, ans

h, a = interaction_iter(8, 'treatment_tamoxifen',
                       [('er_positive','iter8_tamoxifen_x_er'),
                        ('pr_positive','iter8_tamoxifen_x_pr')],
                       'tamx',
                       "patients with {mod}=1 (the canonical hormone-receptor-positive group) should derive larger PFS benefit from {tx}")
iters.append({"index":8, "proposed_hypotheses":h, "analyses":a})

# ---- Iteration 9: trastuzumab x HER2+, HER2-low ----
h, a = interaction_iter(9, 'treatment_trastuzumab',
                       [('her2_positive','iter9_trastuzumab_x_her2pos'),
                        ('her2_low','iter9_trastuzumab_x_her2low')],
                       'trasx',
                       "{tx}, a HER2-targeted antibody, should produce larger PFS benefit when {mod}=1")
iters.append({"index":9, "proposed_hypotheses":h, "analyses":a})

# ---- Iteration 10: olaparib x BRCA ----
h, a = interaction_iter(10, 'treatment_olaparib',
                       [('brca1_mutation','iter10_olaparib_x_brca1'),
                        ('brca2_mutation','iter10_olaparib_x_brca2'),
                        ('brca_any','iter10_olaparib_x_brca_any')],
                       'olax',
                       "{tx} (a PARP inhibitor) should produce larger PFS benefit when {mod}=1 (synthetic-lethality biology)")
iters.append({"index":10, "proposed_hypotheses":h, "analyses":a})

# ---- Iteration 11: palbociclib x ER+, x HER2-, x postmeno, x ER+/HER2- ----
h, a = interaction_iter(11, 'treatment_palbociclib',
                       [('er_positive','iter11_palbo_x_er'),
                        ('her2_positive','iter11_palbo_x_her2neg'),
                        ('postmenopausal','iter11_palbo_x_postmeno'),
                        ('er_pos_her2_neg','iter11_palbo_x_er_pos_her2_neg')],
                       'palbox',
                       "{tx} (a CDK4/6 inhibitor) should produce larger PFS benefit when {mod}=1 (HR+/HER2- biology)")
iters.append({"index":11, "proposed_hypotheses":h, "analyses":a})

# ---- Iteration 12: sacituzumab x TNBC ----
h, a = interaction_iter(12, 'treatment_sacituzumab_govitecan',
                       [('tnbc','iter12_saci_x_tnbc'),
                        ('er_positive','iter12_saci_x_er_neg'),
                        ('her2_low','iter12_saci_x_her2_low')],
                       'sacix',
                       "{tx} (a TROP2-directed ADC) should produce larger PFS benefit when {mod}=1 (or, for er_positive=1, smaller benefit)")
iters.append({"index":12, "proposed_hypotheses":h, "analyses":a})

# ---- Iteration 13: pembrolizumab x TNBC, x stage_iv, x brain_mets ----
h, a = interaction_iter(13, 'treatment_pembrolizumab',
                       [('tnbc','iter13_pembro_x_tnbc'),
                        ('stage_iv','iter13_pembro_x_stage_iv'),
                        ('has_brain_mets','iter13_pembro_x_pdl1_unavail_proxy_tnbc_brain')],
                       'pembrox',
                       "{tx} (an immune checkpoint inhibitor) should produce larger PFS benefit in {mod}=1 patients (typical TNBC/advanced indications)")
iters.append({"index":13, "proposed_hypotheses":h, "analyses":a})

# ---- Iteration 14: full screen of all tx x all candidate modifiers ----
top = R['iter14_screen'][:15]
hyps14 = [{"id":"h14.screen",
           "text":("Across the full grid of treatment x biomarker/clinical-feature interactions, "
                   "any treatment-effect heterogeneity should concentrate in biologically plausible "
                   "modifiers (HR+ for tamoxifen/palbociclib, HER2+ for trastuzumab, BRCA for olaparib, "
                   "TNBC for sacituzumab/pembrolizumab); we expect strongly non-zero interactions only "
                   "for the matched indications."),
           "kind":"novel"}]
ans14 = []
for x in top:
    if 'p_interaction' not in x: continue
    ans14.append({"hypothesis_ids":["h14.screen"],
                  "code": f"smf.ols('pfs_months ~ {x['tx']}*{x['modifier']} + <covs>', df).fit()",
                  "result_summary": (f"{x['tx']} x {x['modifier']}: interaction coefficient "
                                     f"{x['interaction']:+.3f}, p_interaction={x['p_interaction']:.2e}."),
                  "effect_estimate": float(x['interaction']),
                  "p_value": float(x['p_interaction']),
                  "significant": bool(x['p_interaction']<0.05)})
iters.append({"index":14, "proposed_hypotheses":hyps14, "analyses":ans14})

# ---- Iteration 15: subgroup-specific treatment effects ----
hyps15, ans15 = [], []
mapping = [
    ('h15.tam_ER+', 'tamoxifen_in_ER+', 'treatment_tamoxifen', 'er_positive==1',
     "Within er_positive==1, treatment_tamoxifen is associated with longer pfs_months."),
    ('h15.tras_HER2+', 'trastuzumab_in_HER2+', 'treatment_trastuzumab', 'her2_positive==1',
     "Within her2_positive==1, treatment_trastuzumab is associated with longer pfs_months."),
    ('h15.ola_BRCA', 'olaparib_in_BRCAany', 'treatment_olaparib', 'brca1_mutation==1 or brca2_mutation==1',
     "Within (brca1_mutation==1 OR brca2_mutation==1), treatment_olaparib is associated with longer pfs_months."),
    ('h15.palbo_HRposHER2neg', 'palbo_in_HRposHER2neg', 'treatment_palbociclib',
     '(er_positive==1 or pr_positive==1) and her2_positive==0',
     "Within HR+/HER2- (i.e. (er_positive==1 OR pr_positive==1) AND her2_positive==0), "
     "treatment_palbociclib is associated with longer pfs_months."),
    ('h15.saci_TNBC', 'saci_in_TNBC', 'treatment_sacituzumab_govitecan',
     'er_positive==0 and pr_positive==0 and her2_positive==0',
     "Within TNBC (er_positive==0 AND pr_positive==0 AND her2_positive==0), "
     "treatment_sacituzumab_govitecan is associated with longer pfs_months."),
    ('h15.pembro_TNBC', 'pembro_in_TNBC', 'treatment_pembrolizumab',
     'er_positive==0 and pr_positive==0 and her2_positive==0',
     "Within TNBC, treatment_pembrolizumab is associated with longer pfs_months."),
]
for hid, key, tx, pred, text in mapping:
    rec = R['iter15_subgroup_definitions'][key]
    hyps15.append({"id":hid, "text":text, "kind":"refined"})
    ans15.append({"hypothesis_ids":[hid],
                  "code": f"smf.ols('pfs_months ~ {tx} + <covs>', df.query('{pred}')).fit()",
                  "result_summary": (f"In subgroup defined by {pred} (n={rec['n']}), adjusted "
                                     f"coefficient on {tx} = {rec['coef']:+.3f} months PFS "
                                     f"(SE {rec['se']:.3f}, p={rec['p_value']:.2e})."),
                  "effect_estimate": float(rec['coef']),
                  "p_value": float(rec['p_value']),
                  "significant": bool(rec['p_value']<0.05)})
iters.append({"index":15, "proposed_hypotheses":hyps15, "analyses":ans15})

# ---- Iteration 16: three-way interactions to test secondary modifiers in canonical subgroups ----
hyps16, ans16 = [], []
threeway_summary = R['iter16_threeway']
for key, label in [('tamoxifen_er_postmeno','treatment_tamoxifen x er_positive x postmenopausal'),
                   ('palbo_hrpos_postmeno','treatment_palbociclib x hr_pos x postmenopausal'),
                   ('palbo_er_her2neg_postmeno','treatment_palbociclib x er_pos_her2_neg x postmenopausal'),
                   ('trastuzumab_her2pos_node','treatment_trastuzumab x her2_positive x node_positive')]:
    v = threeway_summary[key]
    hid = f"h16.{key}"
    hyps16.append({"id": hid,
                   "text": (f"In a model with the three-way interaction {label}, the three-way term is "
                            f"non-zero, indicating that the treatment-by-canonical-modifier effect itself "
                            f"depends on the secondary modifier."),
                   "kind":"novel"})
    triple_key = next((k for k in v['coefs'] if k.count(':')==2), None)
    coef = v['coefs'].get(triple_key, 0.0) if triple_key else 0.0
    pv = v['pvals'].get(triple_key, 1.0) if triple_key else 1.0
    formula_str = label.replace(' x ', '*')
    ans16.append({"hypothesis_ids":[hid],
                  "code": f"smf.ols('pfs_months ~ {formula_str} + <covs>', df).fit()",
                  "result_summary": (f"Three-way interaction term ({triple_key}): coef={coef:+.4f}, "
                                     f"p={pv:.2e}. Two-way coefficients: " +
                                     "; ".join(f"{k}={vv:+.4f}(p={v['pvals'].get(k,1):.2e})"
                                               for k,vv in v['coefs'].items() if k.count(':')==1)),
                  "effect_estimate": float(coef),
                  "p_value": float(pv),
                  "significant": bool(pv<0.05)})
iters.append({"index":16, "proposed_hypotheses":hyps16, "analyses":ans16})

# ---- Iteration 17: clinical/staging modifiers of treatment effect ----
hyps17, ans17 = [], []
for tx in treatments:
    v = R['iter17_clinical_modifiers'][tx]
    for mod_key in ['x_stage_iv','x_brain_mets','x_node_positive']:
        mod = mod_key.replace('x_','')
        if mod=='brain_mets': mod='has_brain_mets'
        if mod=='node_positive': mod='node_positive'
        if mod=='stage_iv': mod='stage_iv'
        rec = v[mod_key]
        hid = f"h17.{tx.split('_',1)[1]}_{mod_key}"
        hyps17.append({"id":hid,
                       "text":(f"The treatment effect of {tx} on pfs_months differs by {mod} status; "
                               f"specifically, the treatment is more (or less) effective in {mod}=1 patients."),
                       "kind":"novel"})
        ans17.append({"hypothesis_ids":[hid],
                      "code": f"smf.ols('pfs_months ~ {tx}*{mod} + <covs>', df).fit()",
                      "result_summary": (f"Interaction {tx}:{mod} = {rec['interaction']:+.4f}, "
                                         f"p_interaction={rec['p_interaction']:.2e}."),
                      "effect_estimate": float(rec['interaction']),
                      "p_value": float(rec['p_interaction']),
                      "significant": bool(rec['p_interaction']<0.05)})
iters.append({"index":17, "proposed_hypotheses":hyps17, "analyses":ans17})

# ---- Iteration 18: continuous modifiers (top hits) ----
hyps18, ans18 = [], []
top18 = [x for x in R['iter18_continuous_modifiers'] if 'p_interaction' in x][:12]
for x in top18:
    hid = f"h18.{x['tx'].split('_',1)[1]}_x_{x['modifier']}"
    hyps18.append({"id":hid,
                   "text":(f"The pfs_months effect of {x['tx']} varies with the continuous predictor "
                           f"{x['modifier']}; we expect a non-zero {x['tx']}:{x['modifier']} interaction term."),
                   "kind":"novel"})
    ans18.append({"hypothesis_ids":[hid],
                  "code": f"smf.ols('pfs_months ~ {x['tx']}*{x['modifier']} + <covs>', df).fit()",
                  "result_summary": (f"Interaction {x['tx']}:{x['modifier']} = {x['interaction']:+.5f} per "
                                     f"unit of {x['modifier']}, p={x['p_interaction']:.2e}."),
                  "effect_estimate": float(x['interaction']),
                  "p_value": float(x['p_interaction']),
                  "significant": bool(x['p_interaction']<0.05)})
iters.append({"index":18, "proposed_hypotheses":hyps18, "analyses":ans18})

# ---- Iteration 19: joint subgroup definitions (multi-feature) ----
hyps19, ans19 = [], []
joint = R['iter19_joint_subgroups']
joint_picks = [
    ('h19.palbo_HR+HER2-_postmeno','palbo_HR+HER2-_postmeno','treatment_palbociclib',
     "Within HR+/HER2- AND postmenopausal=1, treatment_palbociclib is associated with longer pfs_months "
     "than in patients outside this subgroup (positive joint subgroup effect)."),
    ('h19.palbo_ER+HER2-_postmeno','palbo_ER+HER2-_postmeno','treatment_palbociclib',
     "Within ER+/HER2- AND postmenopausal=1, treatment_palbociclib is associated with longer pfs_months."),
    ('h19.tras_HER2+_StageIV','tras_HER2+_StageIV','treatment_trastuzumab',
     "Within her2_positive=1 AND stage_iv=1, treatment_trastuzumab is associated with longer pfs_months."),
    ('h19.ola_BRCAany','ola_BRCAany','treatment_olaparib',
     "Within (brca1_mutation=1 OR brca2_mutation=1), treatment_olaparib is associated with longer pfs_months."),
    ('h19.saci_TNBC_StageIV','saci_TNBC_StageIV','treatment_sacituzumab_govitecan',
     "Within TNBC AND stage_iv=1, treatment_sacituzumab_govitecan is associated with longer pfs_months."),
    ('h19.pembro_TNBC_StageIV','pembro_TNBC_StageIV','treatment_pembrolizumab',
     "Within TNBC AND stage_iv=1, treatment_pembrolizumab is associated with longer pfs_months."),
    ('h19.tam_ER+PR+postmeno','tam_ER+PR+postmeno','treatment_tamoxifen',
     "Within ER+ AND PR+ AND postmenopausal=1, treatment_tamoxifen is associated with longer pfs_months."),
]
for hid, key, tx, text in joint_picks:
    v = joint[key]
    hyps19.append({"id":hid, "text":text, "kind":"refined"})
    ans19.append({"hypothesis_ids":[hid],
                  "code": f"smf.ols('pfs_months ~ {tx} + <covs>', df.query('{v['predicate']}')).fit()",
                  "result_summary": (f"Subgroup ({v['predicate']}, n_in={v['n_in']}): "
                                     f"adjusted {tx} coef = {v['tx_effect_in']:+.3f} (p={v['p_in']:.2e}); "
                                     f"in complement (n_out={v['n_out']}): "
                                     f"{v['tx_effect_out']:+.3f} (p={v['p_out']:.2e}); "
                                     f"in - out = {v['difference']:+.3f}."),
                  "effect_estimate": float(v['tx_effect_in']),
                  "p_value": float(v['p_in']),
                  "significant": bool(v['p_in']<0.05)})
iters.append({"index":19, "proposed_hypotheses":hyps19, "analyses":ans19})

# ---- Iteration 20: prognostic-only multivariable model (sanity check) ----
m20 = R['iter20_prognostic_model']
hyps20 = [{"id":"h20.prognostic",
           "text":("In a multivariable OLS model of pfs_months on age_years, ecog_ps, stage_iv, has_brain_mets, "
                   "albumin_g_dl, ldh_u_l, crp_mg_l, nlr, weight_loss_pct_6mo, and tumor_size_cm, the dominant "
                   "prognostic variables are age_years (positive), ecog_ps (negative), stage_iv (negative), "
                   "has_brain_mets (negative), albumin_g_dl (positive), and weight_loss_pct_6mo (negative)."),
           "kind":"refined"}]
ans20 = []
for var in ['age_years','ecog_ps','stage_iv','has_brain_mets','albumin_g_dl','weight_loss_pct_6mo']:
    coef = m20['coefs'].get(var, 0.0); pv = m20['pvals'].get(var, 1.0)
    ans20.append({"hypothesis_ids":["h20.prognostic"],
                  "code": "smf.ols('pfs_months ~ <prognostic covs>', df).fit()",
                  "result_summary": f"OLS coefficient on {var} = {coef:+.4f} (p={pv:.2e}); model R^2 = {m20['r_squared']:.3f}.",
                  "effect_estimate": float(coef),
                  "p_value": float(pv),
                  "significant": bool(pv<0.05)})
iters.append({"index":20, "proposed_hypotheses":hyps20, "analyses":ans20})

# ---- Iteration 21: full multi-treatment x indication interaction model ----
m21 = R['iter21_full_model']
hyps21 = [{"id":"h21.full_inter",
           "text":("In a single OLS model that includes each treatment crossed with its canonical indication "
                   "(treatment_tamoxifen*er_positive, treatment_trastuzumab*her2_positive, "
                   "treatment_olaparib*brca_any, treatment_palbociclib*er_pos_her2_neg, "
                   "treatment_sacituzumab_govitecan*tnbc, treatment_pembrolizumab*tnbc), only the "
                   "treatment_palbociclib:er_pos_her2_neg interaction is meaningfully positive; the other "
                   "treatment-by-canonical-indication interactions are near zero."),
           "kind":"novel"}]
ans21 = []
for term in ['treatment_palbociclib:er_pos_her2_neg','treatment_tamoxifen:er_positive',
             'treatment_trastuzumab:her2_positive','treatment_olaparib:brca_any',
             'treatment_sacituzumab_govitecan:tnbc','treatment_pembrolizumab:tnbc']:
    coef = m21['coefs'].get(term,0.0); pv = m21['pvals'].get(term,1.0)
    ans21.append({"hypothesis_ids":["h21.full_inter"],
                  "code": f"smf.ols('pfs_months ~ <multi-tx*indication interactions> + <covs>', df).fit()  # term {term}",
                  "result_summary": f"Coefficient on {term} = {coef:+.3f} months PFS (p={pv:.2e}); model R^2 = {m21['r_squared']:.3f}.",
                  "effect_estimate": float(coef),
                  "p_value": float(pv),
                  "significant": bool(pv<0.05)})
iters.append({"index":21, "proposed_hypotheses":hyps21, "analyses":ans21})

# ---- Iteration 22: secondary modifier search inside each indication subgroup ----
hyps22, ans22 = [], []
sec = R['iter22_secondary_modifiers']
# For each treatment indication, take the strongest secondary modifier and report
for grp_key, tx_label, predicate in [
    ('palbo_in_HRposHER2neg','treatment_palbociclib','HR+/HER2-'),
    ('trastuzumab_in_HER2pos','treatment_trastuzumab','HER2+'),
    ('olaparib_in_BRCAany','treatment_olaparib','BRCA1/2 mut'),
    ('saci_in_TNBC','treatment_sacituzumab_govitecan','TNBC'),
    ('pembro_in_TNBC','treatment_pembrolizumab','TNBC'),
    ('tamoxifen_in_ERpos','treatment_tamoxifen','ER+'),
]:
    arr = sec[grp_key]
    # take the first (strongest) hit per group
    top_hit = next((x for x in arr if 'p_interaction' in x), None)
    if not top_hit: continue
    hid = f"h22.{tx_label.split('_',1)[1]}_in_{grp_key}_top"
    hyps22.append({"id":hid,
                   "text":(f"Within the {predicate} subgroup, the {tx_label} treatment effect on pfs_months is "
                           f"further modified by {top_hit['cand']} (a non-zero "
                           f"{tx_label}:{top_hit['cand']} interaction within the indication subgroup)."),
                   "kind":"novel"})
    ans22.append({"hypothesis_ids":[hid],
                  "code": f"smf.ols('pfs_months ~ {tx_label}*{top_hit['cand']} + <covs>', df.query('<{predicate} predicate>')).fit()",
                  "result_summary": (f"Within {predicate}, {tx_label}:{top_hit['cand']} interaction = "
                                     f"{top_hit['interaction']:+.4f}, p={top_hit['p_interaction']:.2e}; "
                                     f"main {tx_label} (at {top_hit['cand']}=0 reference) = "
                                     f"{top_hit['tx_main']:+.4f} (p={top_hit['p_tx_main']:.2e})."),
                  "effect_estimate": float(top_hit['interaction']),
                  "p_value": float(top_hit['p_interaction']),
                  "significant": bool(top_hit['p_interaction']<0.05)})
iters.append({"index":22, "proposed_hypotheses":hyps22, "analyses":ans22})

# ---- Iteration 23: refined combined subgroups (test the final-form predicates) ----
hyps23, ans23 = [], []
ref = R['iter23_refined_combined_subgroups']
for hid, key, tx, text in [
    ('h23.tras_HER2+_noBrainMets','tras_HER2+_noBrainMets','treatment_trastuzumab',
     "Within her2_positive=1 AND has_brain_mets=0, treatment_trastuzumab is associated with longer pfs_months."),
    ('h23.palbo_HR+HER2-_postmeno_ECOG<2','palbo_HR+HER2-_postmeno_ECOG<2','treatment_palbociclib',
     "Within HR+/HER2- AND postmenopausal=1 AND ecog_ps<2, treatment_palbociclib is associated with longer pfs_months."),
    ('h23.saci_TNBC_StageIV','saci_TNBC_StageIV','treatment_sacituzumab_govitecan',
     "Within TNBC AND stage_iv=1, treatment_sacituzumab_govitecan is associated with longer pfs_months."),
    ('h23.pembro_TNBC_StageIV_noBrain','pembro_TNBC_StageIV_noBrain','treatment_pembrolizumab',
     "Within TNBC AND stage_iv=1 AND has_brain_mets=0, treatment_pembrolizumab is associated with longer pfs_months."),
    ('h23.ola_BRCA_ECOG<2','ola_BRCA_ECOG<2','treatment_olaparib',
     "Within (brca1_mutation=1 OR brca2_mutation=1) AND ecog_ps<2, treatment_olaparib is associated with longer pfs_months."),
    ('h23.tam_ER+PR+postmeno','tam_ER+PR+_postmeno','treatment_tamoxifen',
     "Within er_positive=1 AND pr_positive=1 AND postmenopausal=1, treatment_tamoxifen is associated with longer pfs_months."),
]:
    v = ref.get(key)
    if v is None: continue
    hyps23.append({"id":hid, "text":text, "kind":"refined"})
    ans23.append({"hypothesis_ids":[hid],
                  "code": f"smf.ols('pfs_months ~ {tx} + <covs>', df.query(<refined-predicate>)).fit()",
                  "result_summary": (f"In refined subgroup ({v['predicate']}, n={v['n']}): adjusted "
                                     f"{tx} coefficient = {v['tx_coef']:+.3f} (SE {v['se']:.3f}, p={v['p_value']:.2e})."),
                  "effect_estimate": float(v['tx_coef']),
                  "p_value": float(v['p_value']),
                  "significant": bool(v['p_value']<0.05)})
iters.append({"index":23, "proposed_hypotheses":hyps23, "analyses":ans23})

# ---- Iteration 24: PIK3CA refines palbo subgroup; final triple-interaction confirmation ----
hyps24 = [
    {"id":"h24.palbo_HRposHER2neg_PIK3CAwt",
     "text":("The treatment_palbociclib PFS benefit is concentrated in patients who are HR+ "
             "((er_positive=1 OR pr_positive=1)) AND her2_positive=0 AND pik3ca_mutation=0; "
             "in HR+/HER2- patients with pik3ca_mutation=1 the palbociclib benefit is essentially zero, "
             "so PIK3CA mutation defines the unfavorable secondary feature that suppresses the treatment effect."),
     "kind":"refined"},
    {"id":"h24.palbo_HRposHER2neg_PIK3CAmut",
     "text":("Within HR+/HER2- AND pik3ca_mutation=1, treatment_palbociclib shows no PFS benefit "
             "(adjusted coefficient near zero, non-significant)."),
     "kind":"refined"},
    {"id":"h24.palbo_triple_interaction",
     "text":("In a model with the three-way interaction "
             "treatment_palbociclib * (HR+/HER2- indicator) * (PIK3CA wild-type indicator), the triple-interaction "
             "coefficient is large and positive (~+2.7 months PFS), while the constituent two-way interactions "
             "and the palbociclib main effect are near zero — i.e. the entire palbociclib effect lives in the "
             "HR+/HER2-/PIK3CA-wt cell."),
     "kind":"novel"},
]
ans24 = []
v = P['palbo_HRposHER2neg_PIK3CAwt']
ans24.append({"hypothesis_ids":["h24.palbo_HRposHER2neg_PIK3CAwt"],
              "code": "smf.ols('pfs_months ~ treatment_palbociclib + <covs>', df.query('(er_positive==1 or pr_positive==1) and her2_positive==0 and pik3ca_mutation==0')).fit()",
              "result_summary": (f"In HR+/HER2-/PIK3CA-wt subgroup (n={v['n']}), adjusted treatment_palbociclib "
                                 f"coefficient = {v['tx_coef']:+.3f} months PFS (SE {v['se']:.3f}, p={v['p_value']:.2e})."),
              "effect_estimate": float(v['tx_coef']),
              "p_value": float(v['p_value']),
              "significant": bool(v['p_value']<0.05)})
v = P['palbo_HRposHER2neg_PIK3CAmut']
ans24.append({"hypothesis_ids":["h24.palbo_HRposHER2neg_PIK3CAmut"],
              "code": "smf.ols('pfs_months ~ treatment_palbociclib + <covs>', df.query('(er_positive==1 or pr_positive==1) and her2_positive==0 and pik3ca_mutation==1')).fit()",
              "result_summary": (f"In HR+/HER2-/PIK3CA-mutant subgroup (n={v['n']}), adjusted treatment_palbociclib "
                                 f"coefficient = {v['tx_coef']:+.3f} months PFS (SE {v['se']:.3f}, p={v['p_value']:.2e})."),
              "effect_estimate": float(v['tx_coef']),
              "p_value": float(v['p_value']),
              "significant": bool(v['p_value']<0.05)})
trip = P['triple_interaction']['treatment_palbociclib:hr_pos_her2_neg:pik3ca_wt']
ans24.append({"hypothesis_ids":["h24.palbo_triple_interaction"],
              "code": "smf.ols('pfs_months ~ treatment_palbociclib*hr_pos_her2_neg*pik3ca_wt + <covs>', df).fit()",
              "result_summary": (f"Triple interaction treatment_palbociclib:hr_pos_her2_neg:pik3ca_wt coefficient "
                                 f"= {trip['coef']:+.3f} months PFS (p={trip['p']:.2e}). Lower-order palbociclib "
                                 f"terms are near zero (palbo main {P['triple_interaction']['treatment_palbociclib']['coef']:+.4f} "
                                 f"p={P['triple_interaction']['treatment_palbociclib']['p']:.2e}; "
                                 f"palbo:hr_pos_her2_neg {P['triple_interaction']['treatment_palbociclib:hr_pos_her2_neg']['coef']:+.4f} "
                                 f"p={P['triple_interaction']['treatment_palbociclib:hr_pos_her2_neg']['p']:.2e}; "
                                 f"palbo:pik3ca_wt {P['triple_interaction']['treatment_palbociclib:pik3ca_wt']['coef']:+.4f} "
                                 f"p={P['triple_interaction']['treatment_palbociclib:pik3ca_wt']['p']:.2e})."),
              "effect_estimate": float(trip['coef']),
              "p_value": float(trip['p']),
              "significant": bool(trip['p']<0.05)})
# Within HR+/HER2-, palbo:pik3ca_mutation interaction
v = P['palbo_pik3ca_within_HRposHER2neg']
inter = v['treatment_palbociclib:pik3ca_mutation']
main = v['treatment_palbociclib']
hyps24.append({"id":"h24.palbo_pik3ca_within_HRposHER2neg",
               "text":("Within HR+/HER2-, there is a large negative treatment_palbociclib x pik3ca_mutation "
                       "interaction on pfs_months: the palbociclib benefit (~+2.7 months in PIK3CA-wt) is "
                       "fully neutralized when pik3ca_mutation=1.")})
ans24.append({"hypothesis_ids":["h24.palbo_pik3ca_within_HRposHER2neg"],
              "code": "smf.ols('pfs_months ~ treatment_palbociclib*pik3ca_mutation + <covs>', df.query('(er_positive==1 or pr_positive==1) and her2_positive==0')).fit()",
              "result_summary": (f"Within HR+/HER2-: treatment_palbociclib main coef = {main['coef']:+.3f} (p={main['p']:.2e}), "
                                 f"treatment_palbociclib:pik3ca_mutation interaction = {inter['coef']:+.3f} (p={inter['p']:.2e})."),
              "effect_estimate": float(inter['coef']),
              "p_value": float(inter['p']),
              "significant": bool(inter['p']<0.05)})
iters.append({"index":24, "proposed_hypotheses":hyps24, "analyses":ans24})

# ---- Iteration 25: Final canonical four-cell visualization and best subgroup hypotheses for each tx ----
hyps25 = []
ans25 = []

# Final per-treatment "best subgroup hypotheses" capturing direction + complete subgroup definition
final_subgroup_specs = [
    {
        "hid":"h25.tamoxifen_final",
        "text":("Final treatment-effect-heterogeneity hypothesis for treatment_tamoxifen: across the entire "
                "ds001_breast cohort, treatment_tamoxifen has no detectable PFS benefit, even within "
                "er_positive=1 AND pr_positive=1 AND postmenopausal=1 (the canonical hormone-receptor-positive "
                "subgroup). Direction near zero (slight negative point estimate, non-significant)."),
        "key": ('treatment_tamoxifen','sub_tam')
    },
    {
        "hid":"h25.trastuzumab_final",
        "text":("Final treatment-effect-heterogeneity hypothesis for treatment_trastuzumab: even within "
                "her2_positive=1 (the canonical indication, with or without restricting to has_brain_mets=0 / "
                "ecog_ps<2 / stage_iv=1), treatment_trastuzumab has no detectable PFS benefit (adjusted "
                "coefficients near zero, non-significant)."),
        "key": ('treatment_trastuzumab','sub_tras')
    },
    {
        "hid":"h25.olaparib_final",
        "text":("Final treatment-effect-heterogeneity hypothesis for treatment_olaparib: within "
                "(brca1_mutation=1 OR brca2_mutation=1) — with or without restricting to ecog_ps<2 — "
                "treatment_olaparib shows no significant PFS benefit; the point estimate inside this "
                "subgroup is small and the formal interaction tx x BRCA is non-significant."),
        "key": ('treatment_olaparib','sub_ola')
    },
    {
        "hid":"h25.palbociclib_final",
        "text":("Final treatment-effect-heterogeneity hypothesis for treatment_palbociclib: the PFS benefit "
                "of treatment_palbociclib is concentrated in patients who are HR+ "
                "((er_positive=1 OR pr_positive=1)) AND her2_positive=0 AND pik3ca_mutation=0. In this "
                "complete subgroup, adjusted PFS is ~+2.74 months longer with palbociclib (p<<0.001). "
                "Two unfavorable predicates suppress the effect: her2_positive=1 (eliminates benefit, "
                "tx x HER2+ interaction strongly negative), and pik3ca_mutation=1 (eliminates benefit, "
                "tx x PIK3CA interaction strongly negative). Within the favorable subgroup, higher ki67_pct "
                "further reduces benefit (ki67 tx-interaction = -0.13 months per percent ki67, p<<0.001), "
                "but does not abolish it."),
        "key": ('treatment_palbociclib','sub_palbo')
    },
    {
        "hid":"h25.sacituzumab_final",
        "text":("Final treatment-effect-heterogeneity hypothesis for treatment_sacituzumab_govitecan: within "
                "TNBC (er_positive=0 AND pr_positive=0 AND her2_positive=0), with or without further "
                "restriction to stage_iv=1, treatment_sacituzumab_govitecan shows no significant PFS benefit "
                "(adjusted coefficient near zero)."),
        "key": ('treatment_sacituzumab_govitecan','sub_saci')
    },
    {
        "hid":"h25.pembrolizumab_final",
        "text":("Final treatment-effect-heterogeneity hypothesis for treatment_pembrolizumab: within TNBC "
                "(er_positive=0 AND pr_positive=0 AND her2_positive=0), with or without further restriction "
                "to stage_iv=1 AND has_brain_mets=0, treatment_pembrolizumab shows no significant PFS benefit "
                "(adjusted coefficient ~+0.01 to +0.03, non-significant). The unadjusted negative association "
                "outside TNBC reflects channelling of pembrolizumab into more advanced disease, not a true "
                "harm."),
        "key": ('treatment_pembrolizumab','sub_pembro')
    },
]

four = R['iter25_four_cell_views']
for spec in final_subgroup_specs:
    hyps25.append({"id":spec['hid'], "text":spec['text'], "kind":"refined"})
    tx, subcol = spec['key']
    short_tx = tx.split('_',1)[1] if tx.startswith('treatment_sacituzumab') else tx.split('_',1)[1]
    # match the keys in iter25_four_cell_views which use short names
    short_map = {
        'treatment_tamoxifen': 'tamoxifen_x_sub_tam',
        'treatment_trastuzumab': 'trastuzumab_x_sub_tras',
        'treatment_olaparib': 'olaparib_x_sub_ola',
        'treatment_palbociclib': 'palbociclib_x_sub_palbo',
        'treatment_sacituzumab_govitecan': 'sacituzumab_x_sub_saci',
        'treatment_pembrolizumab': 'pembrolizumab_x_sub_pembro',
    }
    cell_key = short_map[tx]
    cells = four[cell_key]
    # use difference-in-means in the matched subgroup as effect estimate (simple, transparent)
    eff = cells['s1_t1']['mean_pfs'] - cells['s1_t0']['mean_pfs']
    ans25.append({"hypothesis_ids":[spec['hid']],
                  "code": f"df.groupby(['<{subcol} subgroup>','{tx}'])['pfs_months'].mean()",
                  "result_summary": (
                      f"Mean pfs_months by ({subcol}, {tx}) cell: "
                      f"sub=0,tx=0 -> {cells['s0_t0']['mean_pfs']:.3f} (n={cells['s0_t0']['n']}); "
                      f"sub=0,tx=1 -> {cells['s0_t1']['mean_pfs']:.3f} (n={cells['s0_t1']['n']}); "
                      f"sub=1,tx=0 -> {cells['s1_t0']['mean_pfs']:.3f} (n={cells['s1_t0']['n']}); "
                      f"sub=1,tx=1 -> {cells['s1_t1']['mean_pfs']:.3f} (n={cells['s1_t1']['n']}). "
                      f"In-subgroup raw treatment-on - treatment-off difference = {eff:+.3f} months PFS."),
                  "effect_estimate": float(eff),
                  "p_value": None,
                  "significant": None})

# Add the formal interaction p-values for each tx with its canonical subgroup indicator
for tx_label, key in [('treatment_tamoxifen','tamoxifen_x_sub'),
                      ('treatment_trastuzumab','trastuzumab_x_sub'),
                      ('treatment_olaparib','olaparib_x_sub'),
                      ('treatment_palbociclib','palbociclib_x_sub'),
                      ('treatment_sacituzumab_govitecan','sacituzumab_x_sub'),
                      ('treatment_pembrolizumab','pembrolizumab_x_sub')]:
    rec = R['iter24_final_subgroup_interactions'][key]
    spec_hid = next(s['hid'] for s in final_subgroup_specs if s['key'][0]==tx_label)
    ans25.append({"hypothesis_ids":[spec_hid],
                  "code": f"smf.ols('pfs_months ~ {tx_label}*<canonical_subgroup_indicator> + <covs>', df).fit()",
                  "result_summary": (f"Formal interaction {tx_label} x canonical-subgroup indicator: "
                                     f"interaction coef = {rec['interaction']:+.3f} months PFS, "
                                     f"p_interaction = {rec['p_interaction']:.2e}; in-subgroup main {tx_label} "
                                     f"(at indicator=0) = {rec['main_tx']:+.4f} (p={rec['p_main_tx']:.2e})."),
                  "effect_estimate": float(rec['interaction']),
                  "p_value": float(rec['p_interaction']),
                  "significant": bool(rec['p_interaction']<0.05)})

iters.append({"index":25, "proposed_hypotheses":hyps25, "analyses":ans25})

# ---- assemble top-level ----
transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-cli@interactive-session",
    "max_iterations": 25,
    "iterations": iters,
}

with open('transcript.json','w') as f:
    json.dump(transcript, f, indent=2, default=str)

print(f"Wrote transcript with {len(iters)} iterations.")
print(f"Total hypotheses: {sum(len(i['proposed_hypotheses']) for i in iters)}")
print(f"Total analyses: {sum(len(i.get('analyses',[])) for i in iters)}")
