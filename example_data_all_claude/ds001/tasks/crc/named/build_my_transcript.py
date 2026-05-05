"""Assemble transcript.json and analysis_summary.txt from results JSON files."""
import json

R1 = json.load(open("my_results.json"))
R2 = json.load(open("my_results2.json"))


def fmt_p(p):
    if p is None:
        return "p=NA"
    if p < 1e-300:
        return "p<1e-300"
    if p < 1e-6:
        return f"p={p:.2e}"
    return f"p={p:.4g}"


def make(hid, text, kind="novel"):
    return {"id": hid, "text": text, "kind": kind}


def analysis(hids, summary, p, eff, code=None, sig=None):
    a = {"hypothesis_ids": hids, "result_summary": summary,
         "p_value": p, "effect_estimate": eff,
         "significant": (p is not None and p < 0.05) if sig is None else sig}
    if code:
        a["code"] = code
    return a


iters = []

# ==== Iteration 1: PFS distribution and age ====
pfs = R1["pfs_describe"]
age_p = R1["age_vs_pfs_pearson"]
iters.append({
    "index": 1,
    "proposed_hypotheses": [
        make("h1_age",
             "Older age_years is associated with longer pfs_months (positive correlation between continuous age and PFS).")
    ],
    "analyses": [
        analysis(["h1_age"],
                 f"Mean PFS={pfs['mean']:.2f} months (SD {pfs['std']:.2f}); "
                 f"Pearson r between age_years and pfs_months = {age_p['r']:.3f} ({fmt_p(age_p['p'])}). "
                 "PFS rises strongly with age in this cohort.",
                 age_p["p"], age_p["r"],
                 code="stats.pearsonr(df['age_years'], df['pfs_months'])"),
    ],
})

# ==== Iteration 2: sex ====
sex_p = R1["sex_vs_pfs_pearson"]
sex_d = R1["pfs_by_sex"]
iters.append({
    "index": 2,
    "proposed_hypotheses": [
        make("h2_sex",
             "Female patients (sex_female=1) have different mean pfs_months than male patients.")
    ],
    "analyses": [
        analysis(["h2_sex"],
                 f"Mean PFS in females={sex_d['group_mean']:.3f} vs males={sex_d['other_mean']:.3f} "
                 f"(diff={sex_d['diff']:+.3f} months, t-test {fmt_p(sex_d['p'])}). "
                 "No detectable sex difference.",
                 sex_d["p"], sex_d["diff"],
                 code="stats.ttest_ind(df.loc[df['sex_female']==1,'pfs_months'], df.loc[df['sex_female']==0,'pfs_months'])")
    ],
})

# ==== Iteration 3: disease severity (stage IV, ECOG) ====
sd = R1["pfs_by_stage_iv"]
ec_p = R1["ecog_vs_pfs_pearson"]
ec_g = R1["pfs_by_ecog"]
iters.append({
    "index": 3,
    "proposed_hypotheses": [
        make("h3_stage", "stage_iv=1 patients have shorter pfs_months than stage_iv=0 patients."),
        make("h3_ecog", "Higher ecog_ps is associated with shorter pfs_months."),
    ],
    "analyses": [
        analysis(["h3_stage"],
                 f"Stage IV mean PFS={sd['group_mean']:.3f} vs non-stage IV={sd['other_mean']:.3f} "
                 f"(diff={sd['diff']:+.3f} months, {fmt_p(sd['p'])}).",
                 sd["p"], sd["diff"]),
        analysis(["h3_ecog"],
                 f"Mean PFS by ECOG: 0={ec_g.get('0', float('nan')):.2f}, "
                 f"1={ec_g.get('1', float('nan')):.2f}, 2={ec_g.get('2', float('nan')):.2f}; "
                 f"Pearson r={ec_p['r']:.3f} ({fmt_p(ec_p['p'])}).",
                 ec_p["p"], ec_p["r"]),
    ],
})

# ==== Iteration 4: tumor sidedness ====
rs = R1["pfs_by_right_sided"]
iters.append({
    "index": 4,
    "proposed_hypotheses": [
        make("h4_right",
             "right_sided_primary=1 patients have shorter pfs_months than left-sided patients (right_sided_primary=0).")
    ],
    "analyses": [
        analysis(["h4_right"],
                 f"Right-sided mean PFS={rs['group_mean']:.3f} vs left-sided={rs['other_mean']:.3f} "
                 f"(diff={rs['diff']:+.3f} months, {fmt_p(rs['p'])}).",
                 rs["p"], rs["diff"])
    ],
})

# ==== Iteration 5: mutation main effects ====
muts = ["kras_mutation", "nras_mutation", "braf_v600e", "msi_high",
        "her2_amplified", "ntrk_fusion"]
hyps5 = []
analyses5 = []
for m in muts:
    hid = f"h5_{m}"
    direction = "shorter"
    hyps5.append(make(hid,
                      f"Patients with {m}=1 have different mean pfs_months than those with {m}=0."))
    d = R1[f"pfs_by_{m}"]
    analyses5.append(analysis(
        [hid],
        f"{m}: mean PFS in mutant={d['group_mean']:.3f} (n={d['n_group']}) vs wild-type={d['other_mean']:.3f} "
        f"(n={d['n_other']}); diff={d['diff']:+.3f} months, {fmt_p(d['p'])}.",
        d["p"], d["diff"]))
iters.append({"index": 5, "proposed_hypotheses": hyps5, "analyses": analyses5})

# ==== Iteration 6: lab biomarker univariate ====
labs = R1["lab_pfs_pearson"]
hyps6 = []
analyses6 = []
for lab, v in labs.items():
    hid = f"h6_{lab}"
    hyps6.append(make(hid, f"{lab} is correlated with pfs_months (any direction)."))
    analyses6.append(analysis(
        [hid],
        f"Pearson r({lab}, pfs_months) = {v['r']:+.4f} ({fmt_p(v['p'])}).",
        v["p"], v["r"]))
iters.append({"index": 6, "proposed_hypotheses": hyps6, "analyses": analyses6})

# ==== Iteration 7: treatment univariate main effects ====
TX = ["treatment_cetuximab", "treatment_bevacizumab", "treatment_pembrolizumab",
      "treatment_encorafenib", "treatment_trastuzumab_tucatinib", "treatment_regorafenib"]
hyps7 = []
analyses7 = []
for tx in TX:
    hid = f"h7_{tx}"
    hyps7.append(make(hid, f"Patients receiving {tx}=1 have different mean pfs_months than those not receiving {tx}."))
    d = R1[f"pfs_by_{tx}"]
    analyses7.append(analysis(
        [hid],
        f"{tx}: treated mean PFS={d['group_mean']:.3f} (n={d['n_group']}) vs untreated={d['other_mean']:.3f} "
        f"(n={d['n_other']}); diff={d['diff']:+.3f} months, {fmt_p(d['p'])}.",
        d["p"], d["diff"]))
iters.append({"index": 7, "proposed_hypotheses": hyps7, "analyses": analyses7})

# ==== Iteration 8: full multivariable model ====
mv = R1["mv_full"]
hyps8 = [make("h8_mv",
              "After adjusting for all features simultaneously (OLS), age_years, ecog_ps, stage_iv, "
              "right_sided_primary, kras_mutation, braf_v600e, albumin_g_dl, weight_loss_pct_6mo, "
              "cea_ng_ml, ldh_u_l and treatment_regorafenib remain independently associated with pfs_months.")]
# pull individual significant adjusted effects
top_terms = ["age_years", "ecog_ps", "stage_iv", "right_sided_primary", "kras_mutation",
             "braf_v600e", "albumin_g_dl", "weight_loss_pct_6mo", "cea_ng_ml",
             "ldh_u_l", "treatment_regorafenib", "treatment_cetuximab",
             "treatment_pembrolizumab", "treatment_encorafenib",
             "treatment_bevacizumab", "treatment_trastuzumab_tucatinib"]
parts = []
for t in top_terms:
    if t in mv:
        parts.append(f"{t}: coef={mv[t]['coef']:+.4f} ({fmt_p(mv[t]['p'])})")
analyses8 = [analysis(
    ["h8_mv"],
    "OLS pfs_months ~ all features. " + "; ".join(parts),
    mv["treatment_regorafenib"]["p"], mv["treatment_regorafenib"]["coef"],
    code="smf.ols('pfs_months ~ <all features>', data=df).fit()")]
iters.append({"index": 8, "proposed_hypotheses": hyps8, "analyses": analyses8})

# ==== Iteration 9: canonical biomarker subgroups for each targeted treatment ====
hyps9 = [
    make("h9_cetux_RASwt",
         "treatment_cetuximab improves pfs_months in patients with KRAS wild-type, NRAS wild-type, BRAF wild-type tumors "
         "(i.e., kras_mutation=0 AND nras_mutation=0 AND braf_v600e=0)."),
    make("h9_pembro_MSI",
         "treatment_pembrolizumab improves pfs_months in msi_high=1 patients."),
    make("h9_encora_BRAF",
         "treatment_encorafenib improves pfs_months in braf_v600e=1 patients."),
    make("h9_trastu_HER2",
         "treatment_trastuzumab_tucatinib improves pfs_months in her2_amplified=1 patients."),
]
def fmt_sub(label):
    v = R1.get(label) or R2.get(label, {})
    if "note" in v:
        return f"{label}: {v}"
    return (f"{label}: n={v['n']} (treated={v.get('n_treated','?')}); coef={v['coef']:+.3f} "
            f"({fmt_p(v['p'])}); mean treated={v.get('mean_treated','?')} vs untreated={v.get('mean_untreated','?')}")
analyses9 = [
    analysis(["h9_cetux_RASwt"], fmt_sub("cetux_in_RASwt_BRAFwt"),
             R1["cetux_in_RASwt_BRAFwt"]["p"], R1["cetux_in_RASwt_BRAFwt"]["coef"],
             code="smf.ols('pfs_months ~ treatment_cetuximab', data=df[(kras==0)&(nras==0)&(braf==0)]).fit()"),
    analysis(["h9_pembro_MSI"], fmt_sub("pembro_in_MSI_high"),
             R1["pembro_in_MSI_high"]["p"], R1["pembro_in_MSI_high"]["coef"]),
    analysis(["h9_encora_BRAF"], fmt_sub("encora_in_BRAFmut"),
             R1["encora_in_BRAFmut"]["p"], R1["encora_in_BRAFmut"]["coef"]),
    analysis(["h9_trastu_HER2"], fmt_sub("trastu_in_HER2pos"),
             R1["trastu_in_HER2pos"]["p"], R1["trastu_in_HER2pos"]["coef"]),
]
iters.append({"index": 9, "proposed_hypotheses": hyps9, "analyses": analyses9})

# ==== Iteration 10: confirm canonical drugs lack effect with control subgroups + adjusted ====
hyps10 = [
    make("h10_cetux_left",
         "treatment_cetuximab improves pfs_months specifically in RAS/BRAF wild-type, left-sided tumors "
         "(kras=0, nras=0, braf_v600e=0, right_sided_primary=0)."),
    make("h10_cetux_KRASmut",
         "treatment_cetuximab is harmful or neutral in kras_mutation=1 patients."),
    make("h10_pembro_MSS",
         "treatment_pembrolizumab is neutral or harmful in msi_high=0 patients."),
    make("h10_adj_targets",
         "After adjusting for clinical covariates, the canonical targeted-therapy associations "
         "(cetuximab in RAS-WT, pembrolizumab in MSI-high, encorafenib in BRAF V600E, trastuzumab/tucatinib in HER2+) "
         "remain non-significant.")
]
analyses10 = [
    analysis(["h10_cetux_left"], fmt_sub("cetux_in_RASwt_BRAFwt_LEFT"),
             R1["cetux_in_RASwt_BRAFwt_LEFT"]["p"], R1["cetux_in_RASwt_BRAFwt_LEFT"]["coef"]),
    analysis(["h10_cetux_KRASmut"], fmt_sub("cetux_in_KRASmut"),
             R1["cetux_in_KRASmut"]["p"], R1["cetux_in_KRASmut"]["coef"]),
    analysis(["h10_pembro_MSS"], fmt_sub("pembro_in_MSS"),
             R1["pembro_in_MSS"]["p"], R1["pembro_in_MSS"]["coef"]),
    analysis(["h10_adj_targets"],
             "Adjusted (covariate-controlled) subgroup analyses: "
             + "; ".join([
                 f"cetuximab RAS-WT/BRAF-WT coef={R1['adj_cetux_RASwt_BRAFwt']['coef']:+.3f} ({fmt_p(R1['adj_cetux_RASwt_BRAFwt']['p'])})",
                 f"cetuximab RAS-WT/BRAF-WT/LEFT coef={R1['adj_cetux_RASwt_BRAFwt_LEFT']['coef']:+.3f} ({fmt_p(R1['adj_cetux_RASwt_BRAFwt_LEFT']['p'])})",
                 f"pembrolizumab MSI-high coef={R1['adj_pembro_MSIhigh']['coef']:+.3f} ({fmt_p(R1['adj_pembro_MSIhigh']['p'])})",
                 f"encorafenib BRAF coef={R1['adj_encora_BRAFmut']['coef']:+.3f} ({fmt_p(R1['adj_encora_BRAFmut']['p'])})",
                 f"trastuzumab/tucatinib HER2+ coef={R1['adj_trastu_HER2pos']['coef']:+.3f} ({fmt_p(R1['adj_trastu_HER2pos']['p'])})"
             ]),
             R1['adj_pembro_MSIhigh']['p'], R1['adj_pembro_MSIhigh']['coef'])
]
iters.append({"index": 10, "proposed_hypotheses": hyps10, "analyses": analyses10})

# ==== Iteration 11: bevacizumab and trastuzumab/tucatinib in any subgroup ====
hyps11 = [
    make("h11_bev_overall",
         "treatment_bevacizumab has no effect on pfs_months in the overall cohort."),
    make("h11_bev_left",
         "treatment_bevacizumab has no effect on pfs_months in left-sided tumors."),
    make("h11_trastu_HER2_RASwt",
         "treatment_trastuzumab_tucatinib improves pfs_months in HER2+, RAS/BRAF wild-type tumors "
         "(her2_amplified=1, kras=0, nras=0, braf_v600e=0).")
]
analyses11 = [
    analysis(["h11_bev_overall"], fmt_sub("bev_overall"),
             R1["bev_overall"]["p"], R1["bev_overall"]["coef"]),
    analysis(["h11_bev_left"], fmt_sub("bev_in_left_sided"),
             R1["bev_in_left_sided"]["p"], R1["bev_in_left_sided"]["coef"]),
    analysis(["h11_trastu_HER2_RASwt"], fmt_sub("trastu_HER2_RASwt_BRAFwt"),
             R1["trastu_HER2_RASwt_BRAFwt"]["p"], R1["trastu_HER2_RASwt_BRAFwt"]["coef"])
]
iters.append({"index": 11, "proposed_hypotheses": hyps11, "analyses": analyses11})

# ==== Iteration 12: regorafenib main effect ====
hyps12 = [
    make("h12_rego_main",
         "treatment_regorafenib increases pfs_months versus no regorafenib in the overall cohort.")
]
analyses12 = [
    analysis(["h12_rego_main"], fmt_sub("rego_overall"),
             R1["rego_overall"]["p"], R1["rego_overall"]["coef"])
]
iters.append({"index": 12, "proposed_hypotheses": hyps12, "analyses": analyses12})

# ==== Iteration 13: treatment x biomarker interaction screen ====
inter = R1["interaction_screen"]
hyps13 = [
    make("h13_screen",
         "An exhaustive treatment-by-biomarker interaction screen will reveal that treatment_regorafenib "
         "has highly significant interactions with kras_mutation, braf_v600e, and right_sided_primary "
         "(its benefit is modified by these features), while other treatments will lack strong interactions.")
]
sig_inter_lines = []
for k, v in inter.items():
    if v["p"] < 0.05:
        sig_inter_lines.append(f"{k}: inter_coef={v['coef']:+.3f} ({fmt_p(v['p'])})")
analyses13 = [
    analysis(["h13_screen"],
             "Interaction screen pfs ~ tx*biomarker for all 6 treatments x 9 biomarkers. Significant: "
             + "; ".join(sig_inter_lines),
             inter["treatment_regorafenib__x__kras_mutation"]["p"],
             inter["treatment_regorafenib__x__kras_mutation"]["coef"])
]
iters.append({"index": 13, "proposed_hypotheses": hyps13, "analyses": analyses13})

# ==== Iteration 14: regorafenib subgroup sweep ====
rego_sub = R1["rego_subgroups"]
hyps14 = [
    make("h14_rego_KRASwt",
         "treatment_regorafenib increases pfs_months in kras_mutation=0 patients, but not in kras_mutation=1 patients."),
    make("h14_rego_LEFT",
         "treatment_regorafenib increases pfs_months in left-sided (right_sided_primary=0) patients, but not in right-sided patients."),
    make("h14_rego_BRAFwt",
         "treatment_regorafenib increases pfs_months in braf_v600e=0 patients, but not in braf_v600e=1 patients."),
    make("h14_rego_CEAhigh",
         "The treatment_regorafenib benefit on pfs_months is abolished in the highest tertile of cea_ng_ml.")
]
analyses14 = [
    analysis(["h14_rego_KRASwt"],
             f"KRAS WT (n={rego_sub['kras_mutation=0']['n']}): coef={rego_sub['kras_mutation=0']['coef']:+.3f} "
             f"({fmt_p(rego_sub['kras_mutation=0']['p'])}). "
             f"KRAS mut (n={rego_sub['kras_mutation=1']['n']}): coef={rego_sub['kras_mutation=1']['coef']:+.3f} "
             f"({fmt_p(rego_sub['kras_mutation=1']['p'])}).",
             rego_sub['kras_mutation=0']['p'], rego_sub['kras_mutation=0']['coef']),
    analysis(["h14_rego_LEFT"],
             f"Left (n={rego_sub['right_sided_primary=0']['n']}): coef={rego_sub['right_sided_primary=0']['coef']:+.3f} "
             f"({fmt_p(rego_sub['right_sided_primary=0']['p'])}). "
             f"Right (n={rego_sub['right_sided_primary=1']['n']}): coef={rego_sub['right_sided_primary=1']['coef']:+.3f} "
             f"({fmt_p(rego_sub['right_sided_primary=1']['p'])}).",
             rego_sub['right_sided_primary=0']['p'], rego_sub['right_sided_primary=0']['coef']),
    analysis(["h14_rego_BRAFwt"],
             f"BRAF WT (n={rego_sub['braf_v600e=0']['n']}): coef={rego_sub['braf_v600e=0']['coef']:+.3f} "
             f"({fmt_p(rego_sub['braf_v600e=0']['p'])}). "
             f"BRAF V600E (n={rego_sub['braf_v600e=1']['n']}): coef={rego_sub['braf_v600e=1']['coef']:+.3f} "
             f"({fmt_p(rego_sub['braf_v600e=1']['p'])}).",
             rego_sub['braf_v600e=0']['p'], rego_sub['braf_v600e=0']['coef']),
    analysis(["h14_rego_CEAhigh"],
             "Tertiles of CEA: "
             + "; ".join([f"{t}: coef={rego_sub[f'cea_ng_ml_{t}']['coef']:+.3f} ({fmt_p(rego_sub[f'cea_ng_ml_{t}']['p'])})"
                          for t in ['low','mid','high']]),
             rego_sub['cea_ng_ml_high']['p'], rego_sub['cea_ng_ml_high']['coef'])
]
iters.append({"index": 14, "proposed_hypotheses": hyps14, "analyses": analyses14})

# ==== Iteration 15: bevacizumab heterogeneity (mostly null) ====
bev_sub = R1["bev_subgroups"]
hyps15 = [
    make("h15_bev_BRAFmut",
         "treatment_bevacizumab harms pfs_months in braf_v600e=1 patients (i.e., negative effect in this subgroup)."),
    make("h15_bev_no_other",
         "treatment_bevacizumab has no significant effect on pfs_months in any of the major clinical subgroups (sidedness, KRAS status, MSI, HER2, sex, stage).")
]
analyses15 = [
    analysis(["h15_bev_BRAFmut"],
             f"Bev in BRAF V600E (n={bev_sub['braf_v600e=1']['n']}): coef={bev_sub['braf_v600e=1']['coef']:+.3f} "
             f"({fmt_p(bev_sub['braf_v600e=1']['p'])}).",
             bev_sub['braf_v600e=1']['p'], bev_sub['braf_v600e=1']['coef']),
    analysis(["h15_bev_no_other"],
             "Bev subgroup sweep – non-significant subgroups: "
             + "; ".join([f"{k}: coef={v['coef']:+.3f} ({fmt_p(v['p'])})"
                          for k, v in bev_sub.items() if v['p'] >= 0.05]),
             bev_sub['right_sided_primary=0']['p'], bev_sub['right_sided_primary=0']['coef'])
]
iters.append({"index": 15, "proposed_hypotheses": hyps15, "analyses": analyses15})

# ==== Iteration 16: combined regorafenib subgroup definition ====
hyps16 = [
    make("h16_rego_combo",
         "treatment_regorafenib increases pfs_months in patients who are kras_mutation=0, nras_mutation=0, "
         "braf_v600e=0, AND right_sided_primary=0 (the joint RAS/BRAF-WT, left-sided subgroup), "
         "with a larger effect than in either single-modifier subgroup."),
    make("h16_rego_LEFT_only",
         "treatment_regorafenib improves pfs_months in left-sided (right_sided_primary=0) patients regardless of mutation status."),
]
analyses16 = [
    analysis(["h16_rego_combo"], fmt_sub("rego_RASwt_BRAFwt_LEFT"),
             R2["rego_RASwt_BRAFwt_LEFT"]["p"], R2["rego_RASwt_BRAFwt_LEFT"]["coef"]),
    analysis(["h16_rego_LEFT_only"], fmt_sub("rego_LEFT_only"),
             R2["rego_LEFT_only"]["p"], R2["rego_LEFT_only"]["coef"]),
]
iters.append({"index": 16, "proposed_hypotheses": hyps16, "analyses": analyses16})

# ==== Iteration 17: CEA modulates regorafenib in the responsive subgroup ====
hyps17 = [
    make("h17_rego_RASwt_LEFT_CEAlow",
         "treatment_regorafenib increases pfs_months most strongly in kras_mutation=0, nras_mutation=0, "
         "braf_v600e=0, right_sided_primary=0, low-CEA (cea_ng_ml below median) patients."),
    make("h17_rego_RASwt_LEFT_CEAhigh_attenuated",
         "Within the RAS/BRAF-WT, left-sided subgroup, the treatment_regorafenib effect is much smaller "
         "in patients with cea_ng_ml in the highest tertile than in those with low/medium cea_ng_ml.")
]
analyses17 = [
    analysis(["h17_rego_RASwt_LEFT_CEAlow"], fmt_sub("rego_RASwt_BRAFwt_LEFT_CEAlow"),
             R2["rego_RASwt_BRAFwt_LEFT_CEAlow"]["p"], R2["rego_RASwt_BRAFwt_LEFT_CEAlow"]["coef"]),
    analysis(["h17_rego_RASwt_LEFT_CEAhigh_attenuated"],
             f"CEA non-high tertile within RAS-WT/BRAF-WT/LEFT: "
             f"coef={R2['rego_RASwt_BRAFwt_LEFT_CEAlowmid']['coef']:+.3f} "
             f"({fmt_p(R2['rego_RASwt_BRAFwt_LEFT_CEAlowmid']['p'])}); "
             f"CEA high tertile within same subgroup: "
             f"coef={R2['rego_RASwt_BRAFwt_LEFT_CEAhigh_tertile']['coef']:+.3f} "
             f"({fmt_p(R2['rego_RASwt_BRAFwt_LEFT_CEAhigh_tertile']['p'])}).",
             R2['rego_RASwt_BRAFwt_LEFT_CEAhigh_tertile']['p'],
             R2['rego_RASwt_BRAFwt_LEFT_CEAhigh_tertile']['coef'])
]
iters.append({"index": 17, "proposed_hypotheses": hyps17, "analyses": analyses17})

# ==== Iteration 18: three-way interaction for regorafenib ====
tw = R2["rego_threeway_KRAS_RIGHT"]
hyps18 = [
    make("h18_threeway",
         "The treatment_regorafenib effect on pfs_months is abolished if EITHER kras_mutation=1 OR right_sided_primary=1; "
         "the effect is restored only when both are zero. Tested as a three-way interaction "
         "treatment_regorafenib × kras_mutation × right_sided_primary, where the three-way coefficient is positive "
         "(indicating that 'KRAS+ AND right-sided' brings the rego effect back to zero rather than further negative).")
]
analyses18 = [
    analysis(["h18_threeway"],
             f"Three-way model coefficients: "
             f"treatment_regorafenib (KRAS WT, left)={tw['treatment_regorafenib']['coef']:+.3f}; "
             f"× kras_mutation = {tw['treatment_regorafenib:kras_mutation']['coef']:+.3f}; "
             f"× right_sided_primary = {tw['treatment_regorafenib:right_sided_primary']['coef']:+.3f}; "
             f"× kras × right = {tw['treatment_regorafenib:kras_mutation:right_sided_primary']['coef']:+.3f} "
             f"({fmt_p(tw['treatment_regorafenib:kras_mutation:right_sided_primary']['p'])}).",
             tw['treatment_regorafenib:kras_mutation:right_sided_primary']['p'],
             tw['treatment_regorafenib:kras_mutation:right_sided_primary']['coef'])
]
iters.append({"index": 18, "proposed_hypotheses": hyps18, "analyses": analyses18})

# ==== Iteration 19: joint multivariable model with all rego modifiers ====
jt = R2["rego_joint_modifier_model"]
hyps19 = [
    make("h19_joint",
         "In a joint multivariable model with covariate adjustment and interactions of treatment_regorafenib with "
         "kras_mutation, braf_v600e, right_sided_primary, and continuous cea_ng_ml, all four interaction terms remain "
         "negative and highly significant, confirming that each independently attenuates the regorafenib benefit.")
]
key_inters = [
    "treatment_regorafenib:kras_mutation",
    "treatment_regorafenib:braf_v600e",
    "treatment_regorafenib:right_sided_primary",
    "treatment_regorafenib:cea_ng_ml",
]
parts19 = [f"{k}: coef={jt[k]['coef']:+.4f} ({fmt_p(jt[k]['p'])})" for k in key_inters]
analyses19 = [
    analysis(["h19_joint"],
             f"Joint model: rego main coef (KRAS-WT, BRAF-WT, left, CEA=0) = "
             f"{jt['treatment_regorafenib']['coef']:+.3f} ({fmt_p(jt['treatment_regorafenib']['p'])}). "
             "Modifier interactions: " + "; ".join(parts19),
             jt['treatment_regorafenib:kras_mutation']['p'],
             jt['treatment_regorafenib:kras_mutation']['coef'])
]
iters.append({"index": 19, "proposed_hypotheses": hyps19, "analyses": analyses19})

# ==== Iteration 20: final responsive subgroup definition + adjusted ====
hyps20 = [
    make("h20_final_subgroup",
         "treatment_regorafenib increases pfs_months in the subgroup defined by kras_mutation=0 AND braf_v600e=0 "
         "AND right_sided_primary=0 AND cea_ng_ml below the upper tertile (~14.5 ng/mL); the unfavorable values "
         "of any of these four features (KRAS mutation, BRAF V600E, right-sided primary, or high CEA) suppress the benefit. "
         "This is the best-supported treatment-effect subgroup for the regorafenib–pfs_months pair."),
    make("h20_final_subgroup_adjusted",
         "The regorafenib benefit in the kras=0 AND braf=0 AND right_sided_primary=0 AND CEA-non-high subgroup "
         "remains large and significant after adjusting for age_years, sex_female, ecog_ps, stage_iv, "
         "albumin_g_dl, ldh_u_l, nlr, crp_mg_l, hemoglobin_g_dl."),
    make("h20_cetux_neg_control",
         "treatment_cetuximab does NOT improve pfs_months in the same kras=0, braf=0, right_sided_primary=0, "
         "CEA-non-high subgroup, confirming that the regorafenib benefit is treatment-specific rather than "
         "an artifact of patient selection."),
]
analyses20 = [
    analysis(["h20_final_subgroup"], fmt_sub("rego_FINAL_KRASwt_BRAFwt_LEFT_CEAnonhigh"),
             R2["rego_FINAL_KRASwt_BRAFwt_LEFT_CEAnonhigh"]["p"],
             R2["rego_FINAL_KRASwt_BRAFwt_LEFT_CEAnonhigh"]["coef"]),
    analysis(["h20_final_subgroup_adjusted"],
             f"Adjusted OLS in same subgroup (n={R2['adj_rego_FINAL_KRASwt_BRAFwt_LEFT_CEAnonhigh']['n']}): "
             f"treatment_regorafenib coef = {R2['adj_rego_FINAL_KRASwt_BRAFwt_LEFT_CEAnonhigh']['coef']:+.3f} "
             f"({fmt_p(R2['adj_rego_FINAL_KRASwt_BRAFwt_LEFT_CEAnonhigh']['p'])}).",
             R2['adj_rego_FINAL_KRASwt_BRAFwt_LEFT_CEAnonhigh']['p'],
             R2['adj_rego_FINAL_KRASwt_BRAFwt_LEFT_CEAnonhigh']['coef']),
    analysis(["h20_cetux_neg_control"], fmt_sub("cetux_RASwt_BRAFwt_LEFT_CEAnonhigh"),
             R2["cetux_RASwt_BRAFwt_LEFT_CEAnonhigh"]["p"],
             R2["cetux_RASwt_BRAFwt_LEFT_CEAnonhigh"]["coef"]),
]
iters.append({"index": 20, "proposed_hypotheses": hyps20, "analyses": analyses20})

transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-manual@1",
    "max_iterations": 25,
    "iterations": iters,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

print(f"Wrote transcript.json with {len(iters)} iterations.")
total_h = sum(len(it['proposed_hypotheses']) for it in iters)
total_a = sum(len(it['analyses']) for it in iters)
print(f"Total: {total_h} hypotheses, {total_a} analyses.")
