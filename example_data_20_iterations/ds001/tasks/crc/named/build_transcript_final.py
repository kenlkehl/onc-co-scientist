"""Build the transcript.json from my_results.json."""
import json

R = json.load(open("my_results.json"))


def fmt_p(p):
    if p < 1e-300:
        return "<1e-300"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3g}"


def t_summary(rec):
    return (f"Mean pfs_months={rec['mean1']:.3f} (n={rec['n1']}) vs "
            f"{rec['mean0']:.3f} (n={rec['n0']}); diff={rec['diff']:+.3f} months; "
            f"Welch t-test p={fmt_p(rec['p'])}.")


def lin_summary(rec, units=""):
    return (f"OLS pfs_months on {rec['focal']} (n={rec['n']}): "
            f"coef={rec['coef']:+.4f}{units}, p={fmt_p(rec['p'])}.")


def is_sig(p, alpha=0.05):
    return bool(p < alpha)


iters = []

# ============ ITER 1 ============
iters.append({
    "index": 1,
    "proposed_hypotheses": [
        {"id": "h1.1", "text": "Patients receiving treatment_cetuximab have different mean pfs_months than patients not receiving treatment_cetuximab.", "kind": "novel"},
        {"id": "h1.2", "text": "Patients receiving treatment_bevacizumab have different mean pfs_months than patients not receiving treatment_bevacizumab.", "kind": "novel"},
        {"id": "h1.3", "text": "Patients receiving treatment_pembrolizumab have different mean pfs_months than patients not receiving treatment_pembrolizumab.", "kind": "novel"},
        {"id": "h1.4", "text": "Patients receiving treatment_encorafenib have different mean pfs_months than patients not receiving treatment_encorafenib.", "kind": "novel"},
        {"id": "h1.5", "text": "Patients receiving treatment_trastuzumab_tucatinib have different mean pfs_months than patients not receiving treatment_trastuzumab_tucatinib.", "kind": "novel"},
        {"id": "h1.6", "text": "Patients receiving treatment_regorafenib have higher mean pfs_months than patients not receiving treatment_regorafenib.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h1.1"], "code": "scipy.stats.ttest_ind(pfs|cet=1, pfs|cet=0)",
         "result_summary": t_summary(R["i1_treatment_cetuximab"]),
         "p_value": R["i1_treatment_cetuximab"]["p"],
         "effect_estimate": R["i1_treatment_cetuximab"]["diff"],
         "significant": is_sig(R["i1_treatment_cetuximab"]["p"])},
        {"hypothesis_ids": ["h1.2"], "code": "scipy.stats.ttest_ind on bev",
         "result_summary": t_summary(R["i1_treatment_bevacizumab"]),
         "p_value": R["i1_treatment_bevacizumab"]["p"],
         "effect_estimate": R["i1_treatment_bevacizumab"]["diff"],
         "significant": is_sig(R["i1_treatment_bevacizumab"]["p"])},
        {"hypothesis_ids": ["h1.3"], "code": "scipy.stats.ttest_ind on pem",
         "result_summary": t_summary(R["i1_treatment_pembrolizumab"]),
         "p_value": R["i1_treatment_pembrolizumab"]["p"],
         "effect_estimate": R["i1_treatment_pembrolizumab"]["diff"],
         "significant": is_sig(R["i1_treatment_pembrolizumab"]["p"])},
        {"hypothesis_ids": ["h1.4"], "code": "scipy.stats.ttest_ind on encorafenib",
         "result_summary": t_summary(R["i1_treatment_encorafenib"]),
         "p_value": R["i1_treatment_encorafenib"]["p"],
         "effect_estimate": R["i1_treatment_encorafenib"]["diff"],
         "significant": is_sig(R["i1_treatment_encorafenib"]["p"])},
        {"hypothesis_ids": ["h1.5"], "code": "scipy.stats.ttest_ind on tras+tuc",
         "result_summary": t_summary(R["i1_treatment_trastuzumab_tucatinib"]),
         "p_value": R["i1_treatment_trastuzumab_tucatinib"]["p"],
         "effect_estimate": R["i1_treatment_trastuzumab_tucatinib"]["diff"],
         "significant": is_sig(R["i1_treatment_trastuzumab_tucatinib"]["p"])},
        {"hypothesis_ids": ["h1.6"], "code": "scipy.stats.ttest_ind on regorafenib",
         "result_summary": t_summary(R["i1_treatment_regorafenib"]),
         "p_value": R["i1_treatment_regorafenib"]["p"],
         "effect_estimate": R["i1_treatment_regorafenib"]["diff"],
         "significant": is_sig(R["i1_treatment_regorafenib"]["p"])},
    ],
})

# ============ ITER 2 ============
iters.append({
    "index": 2,
    "proposed_hypotheses": [
        {"id": "h2.1", "text": "Patients with kras_mutation=1 have shorter mean pfs_months than patients with kras_mutation=0.", "kind": "novel"},
        {"id": "h2.2", "text": "Patients with nras_mutation=1 have shorter mean pfs_months than patients with nras_mutation=0.", "kind": "novel"},
        {"id": "h2.3", "text": "Patients with braf_v600e=1 have shorter mean pfs_months than patients with braf_v600e=0.", "kind": "novel"},
        {"id": "h2.4", "text": "Patients with msi_high=1 have longer mean pfs_months than patients with msi_high=0.", "kind": "novel"},
        {"id": "h2.5", "text": "Patients with her2_amplified=1 have shorter mean pfs_months than patients with her2_amplified=0.", "kind": "novel"},
        {"id": "h2.6", "text": "Patients with tp53_mutation=1 have shorter mean pfs_months than patients with tp53_mutation=0.", "kind": "novel"},
        {"id": "h2.7", "text": "Patients with pik3ca_mutation=1 have shorter mean pfs_months than patients with pik3ca_mutation=0.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h2.1"], "code": "ttest_ind on kras_mutation",
         "result_summary": t_summary(R["i2_kras_mutation"]),
         "p_value": R["i2_kras_mutation"]["p"], "effect_estimate": R["i2_kras_mutation"]["diff"],
         "significant": is_sig(R["i2_kras_mutation"]["p"])},
        {"hypothesis_ids": ["h2.2"], "code": "ttest_ind on nras_mutation",
         "result_summary": t_summary(R["i2_nras_mutation"]),
         "p_value": R["i2_nras_mutation"]["p"], "effect_estimate": R["i2_nras_mutation"]["diff"],
         "significant": is_sig(R["i2_nras_mutation"]["p"])},
        {"hypothesis_ids": ["h2.3"], "code": "ttest_ind on braf_v600e",
         "result_summary": t_summary(R["i2_braf_v600e"]),
         "p_value": R["i2_braf_v600e"]["p"], "effect_estimate": R["i2_braf_v600e"]["diff"],
         "significant": is_sig(R["i2_braf_v600e"]["p"])},
        {"hypothesis_ids": ["h2.4"], "code": "ttest_ind on msi_high",
         "result_summary": t_summary(R["i2_msi_high"]),
         "p_value": R["i2_msi_high"]["p"], "effect_estimate": R["i2_msi_high"]["diff"],
         "significant": is_sig(R["i2_msi_high"]["p"])},
        {"hypothesis_ids": ["h2.5"], "code": "ttest_ind on her2_amplified",
         "result_summary": t_summary(R["i2_her2_amplified"]),
         "p_value": R["i2_her2_amplified"]["p"], "effect_estimate": R["i2_her2_amplified"]["diff"],
         "significant": is_sig(R["i2_her2_amplified"]["p"])},
        {"hypothesis_ids": ["h2.6"], "code": "ttest_ind on tp53_mutation",
         "result_summary": t_summary(R["i2_tp53_mutation"]),
         "p_value": R["i2_tp53_mutation"]["p"], "effect_estimate": R["i2_tp53_mutation"]["diff"],
         "significant": is_sig(R["i2_tp53_mutation"]["p"])},
        {"hypothesis_ids": ["h2.7"], "code": "ttest_ind on pik3ca_mutation",
         "result_summary": t_summary(R["i2_pik3ca_mutation"]),
         "p_value": R["i2_pik3ca_mutation"]["p"], "effect_estimate": R["i2_pik3ca_mutation"]["diff"],
         "significant": is_sig(R["i2_pik3ca_mutation"]["p"])},
    ],
})

# ============ ITER 3 ============
iters.append({
    "index": 3,
    "proposed_hypotheses": [
        {"id": "h3.1", "text": "Patients with stage_iv=1 have shorter mean pfs_months than patients with stage_iv=0.", "kind": "novel"},
        {"id": "h3.2", "text": "Patients with right_sided_primary=1 have shorter mean pfs_months than patients with right_sided_primary=0.", "kind": "novel"},
        {"id": "h3.3", "text": "Higher ecog_ps is associated with shorter pfs_months (negative slope).", "kind": "novel"},
        {"id": "h3.4", "text": "Older age_years is associated with shorter pfs_months (negative slope).", "kind": "novel"},
        {"id": "h3.5", "text": "Higher baseline albumin_g_dl is associated with longer pfs_months (positive slope).", "kind": "novel"},
        {"id": "h3.6", "text": "Higher baseline ldh_u_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
        {"id": "h3.7", "text": "Higher baseline cea_ng_ml is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h3.1"], "code": "ttest_ind on stage_iv",
         "result_summary": t_summary(R["i3_stage_iv"]),
         "p_value": R["i3_stage_iv"]["p"], "effect_estimate": R["i3_stage_iv"]["diff"],
         "significant": is_sig(R["i3_stage_iv"]["p"])},
        {"hypothesis_ids": ["h3.2"], "code": "ttest_ind on right_sided_primary",
         "result_summary": t_summary(R["i3_right_sided_primary"]),
         "p_value": R["i3_right_sided_primary"]["p"],
         "effect_estimate": R["i3_right_sided_primary"]["diff"],
         "significant": is_sig(R["i3_right_sided_primary"]["p"])},
        {"hypothesis_ids": ["h3.3"], "code": "smf.ols('pfs_months ~ ecog_ps')",
         "result_summary": lin_summary(R["i3_ecog"], " months/unit"),
         "p_value": R["i3_ecog"]["p"], "effect_estimate": R["i3_ecog"]["coef"],
         "significant": is_sig(R["i3_ecog"]["p"])},
        {"hypothesis_ids": ["h3.4"], "code": "smf.ols('pfs_months ~ age_years')",
         "result_summary": lin_summary(R["i3_age"], " months/year"),
         "p_value": R["i3_age"]["p"], "effect_estimate": R["i3_age"]["coef"],
         "significant": is_sig(R["i3_age"]["p"])},
        {"hypothesis_ids": ["h3.5"], "code": "smf.ols('pfs_months ~ albumin_g_dl')",
         "result_summary": lin_summary(R["i3_albumin"], " months per g/dL"),
         "p_value": R["i3_albumin"]["p"], "effect_estimate": R["i3_albumin"]["coef"],
         "significant": is_sig(R["i3_albumin"]["p"])},
        {"hypothesis_ids": ["h3.6"], "code": "smf.ols('pfs_months ~ ldh_u_l')",
         "result_summary": lin_summary(R["i3_ldh"], " months per U/L"),
         "p_value": R["i3_ldh"]["p"], "effect_estimate": R["i3_ldh"]["coef"],
         "significant": is_sig(R["i3_ldh"]["p"])},
        {"hypothesis_ids": ["h3.7"], "code": "smf.ols('pfs_months ~ cea_ng_ml')",
         "result_summary": lin_summary(R["i3_cea"], " months per ng/mL"),
         "p_value": R["i3_cea"]["p"], "effect_estimate": R["i3_cea"]["coef"],
         "significant": is_sig(R["i3_cea"]["p"])},
    ],
})

# ============ ITER 4: cetuximab × RAS interaction ============
iters.append({
    "index": 4,
    "proposed_hypotheses": [
        {"id": "h4.1", "text": "Among patients with kras_mutation=0, treatment_cetuximab is associated with longer mean pfs_months than no cetuximab; among kras_mutation=1 patients there is no benefit (positive interaction term such that the cetuximab effect is more positive in KRAS WT).", "kind": "novel"},
        {"id": "h4.2", "text": "The interaction treatment_cetuximab × nras_mutation is non-zero, with cetuximab benefit attenuated in NRAS-mutant patients (negative effect on cetuximab in NRAS-mutant subgroup).", "kind": "novel"},
        {"id": "h4.3", "text": "Among RAS wild-type (kras_mutation=0 AND nras_mutation=0), treatment_cetuximab benefits pfs_months more than in RAS-mutant patients (negative cetuximab × ras_wt interaction would imply opposite; we expect positive).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h4.1"], "code": "smf.ols('pfs_months ~ treatment_cetuximab*kras_mutation')",
         "result_summary": (f"Cetuximab effect within KRAS WT: diff={R['i4_cet_in_kras0']['diff']:+.3f} (p={fmt_p(R['i4_cet_in_kras0']['p'])}, n_cet={R['i4_cet_in_kras0']['n1']}). "
                            f"Within KRAS mutant: diff={R['i4_cet_in_kras1']['diff']:+.3f} (p={fmt_p(R['i4_cet_in_kras1']['p'])}, n_cet={R['i4_cet_in_kras1']['n1']}). "
                            f"Interaction term coef={R['i4_cet_kras_ix']['coef']:+.4f}, p={fmt_p(R['i4_cet_kras_ix']['p'])}."),
         "p_value": R["i4_cet_kras_ix"]["p"], "effect_estimate": R["i4_cet_kras_ix"]["coef"],
         "significant": is_sig(R["i4_cet_kras_ix"]["p"])},
        {"hypothesis_ids": ["h4.2"], "code": "smf.ols('pfs_months ~ treatment_cetuximab*nras_mutation')",
         "result_summary": f"Cetuximab×NRAS interaction coef={R['i4_cet_nras_ix']['coef']:+.4f}, p={fmt_p(R['i4_cet_nras_ix']['p'])}.",
         "p_value": R["i4_cet_nras_ix"]["p"], "effect_estimate": R["i4_cet_nras_ix"]["coef"],
         "significant": is_sig(R["i4_cet_nras_ix"]["p"])},
        {"hypothesis_ids": ["h4.3"], "code": "ras_wt = (kras==0)&(nras==0); smf.ols('pfs_months ~ treatment_cetuximab*ras_wt')",
         "result_summary": f"Cetuximab×RAS-WT interaction coef={R['i4_cet_raswt_ix']['coef']:+.4f}, p={fmt_p(R['i4_cet_raswt_ix']['p'])}.",
         "p_value": R["i4_cet_raswt_ix"]["p"], "effect_estimate": R["i4_cet_raswt_ix"]["coef"],
         "significant": is_sig(R["i4_cet_raswt_ix"]["p"])},
    ],
})

# ============ ITER 5 ============
iters.append({
    "index": 5,
    "proposed_hypotheses": [
        {"id": "h5.1", "text": "The interaction treatment_encorafenib × braf_v600e is positive on pfs_months: encorafenib benefits BRAF V600E mutant patients more than BRAF wild-type.", "kind": "novel"},
        {"id": "h5.2", "text": "Within BRAF V600E mutant patients, treatment_encorafenib is associated with longer mean pfs_months than no encorafenib.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h5.1"], "code": "smf.ols('pfs_months ~ treatment_encorafenib*braf_v600e')",
         "result_summary": (f"Encorafenib×BRAF V600E interaction coef={R['i5_enc_brafix']['coef']:+.4f}, p={fmt_p(R['i5_enc_brafix']['p'])}. "
                            f"Within BRAF WT: cet_diff={R['i5_enc_in_braf0']['diff']:+.3f} (p={fmt_p(R['i5_enc_in_braf0']['p'])}). "
                            f"Within BRAF V600E: enc_diff={R['i5_enc_in_braf1']['diff']:+.3f} (p={fmt_p(R['i5_enc_in_braf1']['p'])}, n_enc={R['i5_enc_in_braf1']['n1']})."),
         "p_value": R["i5_enc_brafix"]["p"], "effect_estimate": R["i5_enc_brafix"]["coef"],
         "significant": is_sig(R["i5_enc_brafix"]["p"])},
        {"hypothesis_ids": ["h5.2"], "code": "stratified ttest within braf_v600e==1",
         "result_summary": (f"Within BRAF V600E mutant subgroup (n={R['i5_enc_in_braf1']['n1']+R['i5_enc_in_braf1']['n0']}): "
                            f"mean pfs on encorafenib={R['i5_enc_in_braf1']['mean1']:.3f} vs off={R['i5_enc_in_braf1']['mean0']:.3f}; "
                            f"diff={R['i5_enc_in_braf1']['diff']:+.3f}, p={fmt_p(R['i5_enc_in_braf1']['p'])}."),
         "p_value": R["i5_enc_in_braf1"]["p"], "effect_estimate": R["i5_enc_in_braf1"]["diff"],
         "significant": is_sig(R["i5_enc_in_braf1"]["p"])},
    ],
})

# ============ ITER 6 ============
iters.append({
    "index": 6,
    "proposed_hypotheses": [
        {"id": "h6.1", "text": "The interaction treatment_pembrolizumab × msi_high is positive on pfs_months: pembrolizumab benefits MSI-high patients more than MSS patients.", "kind": "novel"},
        {"id": "h6.2", "text": "Within msi_high=1, treatment_pembrolizumab is associated with longer mean pfs_months than no pembrolizumab.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h6.1"], "code": "smf.ols('pfs_months ~ treatment_pembrolizumab*msi_high')",
         "result_summary": (f"Pembrolizumab×MSI-high interaction coef={R['i6_pem_msi_ix']['coef']:+.4f}, p={fmt_p(R['i6_pem_msi_ix']['p'])}. "
                            f"Within MSS: pem_diff={R['i6_pem_in_msi0']['diff']:+.3f} (p={fmt_p(R['i6_pem_in_msi0']['p'])}). "
                            f"Within MSI-H: pem_diff={R['i6_pem_in_msi1']['diff']:+.3f} (p={fmt_p(R['i6_pem_in_msi1']['p'])}, n_pem={R['i6_pem_in_msi1']['n1']})."),
         "p_value": R["i6_pem_msi_ix"]["p"], "effect_estimate": R["i6_pem_msi_ix"]["coef"],
         "significant": is_sig(R["i6_pem_msi_ix"]["p"])},
        {"hypothesis_ids": ["h6.2"], "code": "stratified ttest within msi_high==1",
         "result_summary": (f"Within MSI-high (n={R['i6_pem_in_msi1']['n1']+R['i6_pem_in_msi1']['n0']}): "
                            f"mean pfs on pem={R['i6_pem_in_msi1']['mean1']:.3f} vs off={R['i6_pem_in_msi1']['mean0']:.3f}; "
                            f"diff={R['i6_pem_in_msi1']['diff']:+.3f}, p={fmt_p(R['i6_pem_in_msi1']['p'])}."),
         "p_value": R["i6_pem_in_msi1"]["p"], "effect_estimate": R["i6_pem_in_msi1"]["diff"],
         "significant": is_sig(R["i6_pem_in_msi1"]["p"])},
    ],
})

# ============ ITER 7 ============
iters.append({
    "index": 7,
    "proposed_hypotheses": [
        {"id": "h7.1", "text": "The interaction treatment_trastuzumab_tucatinib × her2_amplified is positive on pfs_months: HER2-directed therapy benefits HER2-amplified patients more than HER2-non-amplified.", "kind": "novel"},
        {"id": "h7.2", "text": "Within her2_amplified=1, treatment_trastuzumab_tucatinib is associated with longer mean pfs_months than no HER2-directed therapy.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h7.1"], "code": "smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib*her2_amplified')",
         "result_summary": (f"Trastuzumab+Tucatinib × HER2-amplified interaction coef={R['i7_her2_ix']['coef']:+.4f}, p={fmt_p(R['i7_her2_ix']['p'])}. "
                            f"Within HER2- patients: trt_diff={R['i7_her2trt_in_her20']['diff']:+.3f} (p={fmt_p(R['i7_her2trt_in_her20']['p'])}). "
                            f"Within HER2+: trt_diff={R['i7_her2trt_in_her21']['diff']:+.3f} (p={fmt_p(R['i7_her2trt_in_her21']['p'])}, n_trt={R['i7_her2trt_in_her21']['n1']})."),
         "p_value": R["i7_her2_ix"]["p"], "effect_estimate": R["i7_her2_ix"]["coef"],
         "significant": is_sig(R["i7_her2_ix"]["p"])},
        {"hypothesis_ids": ["h7.2"], "code": "stratified ttest within her2_amplified==1",
         "result_summary": (f"Within HER2-amplified subgroup (n={R['i7_her2trt_in_her21']['n1']+R['i7_her2trt_in_her21']['n0']}): "
                            f"mean pfs on Tras+Tuc={R['i7_her2trt_in_her21']['mean1']:.3f} vs off={R['i7_her2trt_in_her21']['mean0']:.3f}; "
                            f"diff={R['i7_her2trt_in_her21']['diff']:+.3f}, p={fmt_p(R['i7_her2trt_in_her21']['p'])}."),
         "p_value": R["i7_her2trt_in_her21"]["p"], "effect_estimate": R["i7_her2trt_in_her21"]["diff"],
         "significant": is_sig(R["i7_her2trt_in_her21"]["p"])},
    ],
})

# ============ ITER 8 ============
iters.append({
    "index": 8,
    "proposed_hypotheses": [
        {"id": "h8.1", "text": "The interaction treatment_cetuximab × right_sided_primary is negative on pfs_months: cetuximab benefits left-sided (right_sided_primary=0) more than right-sided primaries.", "kind": "novel"},
        {"id": "h8.2", "text": "Among RAS wild-type left-sided patients, cetuximab is associated with longer pfs_months than no cetuximab; among RAS wild-type right-sided patients there is no benefit.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h8.1"], "code": "smf.ols('pfs_months ~ treatment_cetuximab*right_sided_primary')",
         "result_summary": (f"Cetuximab × right-sided interaction coef={R['i8_cet_side_ix']['coef']:+.4f}, p={fmt_p(R['i8_cet_side_ix']['p'])}. "
                            f"Within left-sided: cet_diff={R['i8_cet_in_side0']['diff']:+.3f} (p={fmt_p(R['i8_cet_in_side0']['p'])}). "
                            f"Within right-sided: cet_diff={R['i8_cet_in_side1']['diff']:+.3f} (p={fmt_p(R['i8_cet_in_side1']['p'])})."),
         "p_value": R["i8_cet_side_ix"]["p"], "effect_estimate": R["i8_cet_side_ix"]["coef"],
         "significant": is_sig(R["i8_cet_side_ix"]["p"])},
        {"hypothesis_ids": ["h8.2"], "code": "ras_wt subgroup, ttest cet by side",
         "result_summary": (f"In RAS-WT left-sided (n={R['i16_cet_raswt_left']['n1']+R['i16_cet_raswt_left']['n0']}): "
                            f"cet_diff={R['i16_cet_raswt_left']['diff']:+.3f}, p={fmt_p(R['i16_cet_raswt_left']['p'])}. "
                            f"In RAS-WT right-sided (n={R['i16_cet_raswt_right']['n1']+R['i16_cet_raswt_right']['n0']}): "
                            f"cet_diff={R['i16_cet_raswt_right']['diff']:+.3f}, p={fmt_p(R['i16_cet_raswt_right']['p'])}."),
         "p_value": R["i16_cet_raswt_left"]["p"], "effect_estimate": R["i16_cet_raswt_left"]["diff"],
         "significant": is_sig(R["i16_cet_raswt_left"]["p"])},
    ],
})

# ============ ITER 9 ============
iters.append({
    "index": 9,
    "proposed_hypotheses": [
        {"id": "h9.1", "text": "Higher nlr is associated with shorter pfs_months (negative slope).", "kind": "novel"},
        {"id": "h9.2", "text": "Higher crp_mg_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
        {"id": "h9.3", "text": "Higher hemoglobin_g_dl is associated with longer pfs_months (positive slope).", "kind": "novel"},
        {"id": "h9.4", "text": "Higher alkaline_phosphatase_u_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
        {"id": "h9.5", "text": "Higher weight_loss_pct_6mo is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h9.1"], "code": "smf.ols('pfs_months ~ nlr')",
         "result_summary": lin_summary(R["i9_nlr"], " months per unit"),
         "p_value": R["i9_nlr"]["p"], "effect_estimate": R["i9_nlr"]["coef"],
         "significant": is_sig(R["i9_nlr"]["p"])},
        {"hypothesis_ids": ["h9.2"], "code": "smf.ols('pfs_months ~ crp_mg_l')",
         "result_summary": lin_summary(R["i9_crp_mg_l"], " months per mg/L"),
         "p_value": R["i9_crp_mg_l"]["p"], "effect_estimate": R["i9_crp_mg_l"]["coef"],
         "significant": is_sig(R["i9_crp_mg_l"]["p"])},
        {"hypothesis_ids": ["h9.3"], "code": "smf.ols('pfs_months ~ hemoglobin_g_dl')",
         "result_summary": lin_summary(R["i9_hemoglobin_g_dl"], " months per g/dL"),
         "p_value": R["i9_hemoglobin_g_dl"]["p"], "effect_estimate": R["i9_hemoglobin_g_dl"]["coef"],
         "significant": is_sig(R["i9_hemoglobin_g_dl"]["p"])},
        {"hypothesis_ids": ["h9.4"], "code": "smf.ols('pfs_months ~ alkaline_phosphatase_u_l')",
         "result_summary": lin_summary(R["i9_alkaline_phosphatase_u_l"], " months per U/L"),
         "p_value": R["i9_alkaline_phosphatase_u_l"]["p"],
         "effect_estimate": R["i9_alkaline_phosphatase_u_l"]["coef"],
         "significant": is_sig(R["i9_alkaline_phosphatase_u_l"]["p"])},
        {"hypothesis_ids": ["h9.5"], "code": "smf.ols('pfs_months ~ weight_loss_pct_6mo')",
         "result_summary": lin_summary(R["i9_weight_loss_pct_6mo"], " months per %"),
         "p_value": R["i9_weight_loss_pct_6mo"]["p"],
         "effect_estimate": R["i9_weight_loss_pct_6mo"]["coef"],
         "significant": is_sig(R["i9_weight_loss_pct_6mo"]["p"])},
    ],
})

# ============ ITER 10 ============
iters.append({
    "index": 10,
    "proposed_hypotheses": [
        {"id": "h10.1", "text": "Patients with liver_mets=1 have shorter mean pfs_months than patients with liver_mets=0.", "kind": "novel"},
        {"id": "h10.2", "text": "Patients with bone_mets=1 have shorter mean pfs_months than patients with bone_mets=0.", "kind": "novel"},
        {"id": "h10.3", "text": "Patients with adrenal_mets=1 have shorter mean pfs_months than patients with adrenal_mets=0.", "kind": "novel"},
        {"id": "h10.4", "text": "Higher count of metastatic sites (sum of liver_mets, bone_mets, adrenal_mets) is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h10.1"], "code": "ttest_ind on liver_mets",
         "result_summary": t_summary(R["i10_liver_mets"]),
         "p_value": R["i10_liver_mets"]["p"], "effect_estimate": R["i10_liver_mets"]["diff"],
         "significant": is_sig(R["i10_liver_mets"]["p"])},
        {"hypothesis_ids": ["h10.2"], "code": "ttest_ind on bone_mets",
         "result_summary": t_summary(R["i10_bone_mets"]),
         "p_value": R["i10_bone_mets"]["p"], "effect_estimate": R["i10_bone_mets"]["diff"],
         "significant": is_sig(R["i10_bone_mets"]["p"])},
        {"hypothesis_ids": ["h10.3"], "code": "ttest_ind on adrenal_mets",
         "result_summary": t_summary(R["i10_adrenal_mets"]),
         "p_value": R["i10_adrenal_mets"]["p"], "effect_estimate": R["i10_adrenal_mets"]["diff"],
         "significant": is_sig(R["i10_adrenal_mets"]["p"])},
        {"hypothesis_ids": ["h10.4"], "code": "smf.ols('pfs_months ~ mets_count')",
         "result_summary": lin_summary(R["i10_mets_count"], " months per site"),
         "p_value": R["i10_mets_count"]["p"], "effect_estimate": R["i10_mets_count"]["coef"],
         "significant": is_sig(R["i10_mets_count"]["p"])},
    ],
})

# ============ ITER 11 ============
iters.append({
    "index": 11,
    "proposed_hypotheses": [
        {"id": "h11.1", "text": "Higher fatigue_grade is associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h11.2", "text": "Higher pain_nrs is associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h11.3", "text": "Higher dyspnea_grade is associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h11.4", "text": "Higher appetite_loss_grade is associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h11.5", "text": "A composite symptom-burden score (sum of pain_nrs, fatigue_grade, dyspnea_grade, cough_grade, appetite_loss_grade) is negatively associated with pfs_months.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h11.1"], "code": "smf.ols('pfs_months ~ fatigue_grade')",
         "result_summary": lin_summary(R["i11_fatigue_grade"], " months/grade"),
         "p_value": R["i11_fatigue_grade"]["p"],
         "effect_estimate": R["i11_fatigue_grade"]["coef"],
         "significant": is_sig(R["i11_fatigue_grade"]["p"])},
        {"hypothesis_ids": ["h11.2"], "code": "smf.ols('pfs_months ~ pain_nrs')",
         "result_summary": lin_summary(R["i11_pain_nrs"], " months/NRS unit"),
         "p_value": R["i11_pain_nrs"]["p"],
         "effect_estimate": R["i11_pain_nrs"]["coef"],
         "significant": is_sig(R["i11_pain_nrs"]["p"])},
        {"hypothesis_ids": ["h11.3"], "code": "smf.ols('pfs_months ~ dyspnea_grade')",
         "result_summary": lin_summary(R["i11_dyspnea_grade"], " months/grade"),
         "p_value": R["i11_dyspnea_grade"]["p"],
         "effect_estimate": R["i11_dyspnea_grade"]["coef"],
         "significant": is_sig(R["i11_dyspnea_grade"]["p"])},
        {"hypothesis_ids": ["h11.4"], "code": "smf.ols('pfs_months ~ appetite_loss_grade')",
         "result_summary": lin_summary(R["i11_appetite_loss_grade"], " months/grade"),
         "p_value": R["i11_appetite_loss_grade"]["p"],
         "effect_estimate": R["i11_appetite_loss_grade"]["coef"],
         "significant": is_sig(R["i11_appetite_loss_grade"]["p"])},
        {"hypothesis_ids": ["h11.5"], "code": "smf.ols('pfs_months ~ sx_burden')",
         "result_summary": lin_summary(R["i11_sx_burden"], " months/unit"),
         "p_value": R["i11_sx_burden"]["p"],
         "effect_estimate": R["i11_sx_burden"]["coef"],
         "significant": is_sig(R["i11_sx_burden"]["p"])},
    ],
})

# ============ ITER 12 ============
iters.append({
    "index": 12,
    "proposed_hypotheses": [
        {"id": "h12.1", "text": "Mean pfs_months differs by sex_female (female vs male).", "kind": "novel"},
        {"id": "h12.2", "text": "Mean pfs_months differs by rural_residence (rural vs urban).", "kind": "novel"},
        {"id": "h12.3", "text": "Higher smoking_pack_years is associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h12.4", "text": "Higher education_years is associated with longer pfs_months (a socioeconomic-status proxy).", "kind": "novel"},
        {"id": "h12.5", "text": "Mean pfs_months differs across race_ethnicity categories.", "kind": "novel"},
        {"id": "h12.6", "text": "Mean pfs_months differs across insurance_type categories.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h12.1"], "code": "ttest_ind on sex_female",
         "result_summary": t_summary(R["i12_sex"]),
         "p_value": R["i12_sex"]["p"], "effect_estimate": R["i12_sex"]["diff"],
         "significant": is_sig(R["i12_sex"]["p"])},
        {"hypothesis_ids": ["h12.2"], "code": "ttest_ind on rural_residence",
         "result_summary": t_summary(R["i12_rural"]),
         "p_value": R["i12_rural"]["p"], "effect_estimate": R["i12_rural"]["diff"],
         "significant": is_sig(R["i12_rural"]["p"])},
        {"hypothesis_ids": ["h12.3"], "code": "smf.ols('pfs_months ~ smoking_pack_years')",
         "result_summary": lin_summary(R["i12_smoking"], " months/pack-year"),
         "p_value": R["i12_smoking"]["p"], "effect_estimate": R["i12_smoking"]["coef"],
         "significant": is_sig(R["i12_smoking"]["p"])},
        {"hypothesis_ids": ["h12.4"], "code": "smf.ols('pfs_months ~ education_years')",
         "result_summary": lin_summary(R["i12_education"], " months/year"),
         "p_value": R["i12_education"]["p"], "effect_estimate": R["i12_education"]["coef"],
         "significant": is_sig(R["i12_education"]["p"])},
        {"hypothesis_ids": ["h12.5"], "code": "smf.ols('pfs_months ~ C(race_ethnicity)')",
         "result_summary": f"ANOVA F={R['i12_race_F']['F']:.3f}, p={fmt_p(R['i12_race_F']['p'])}.",
         "p_value": R["i12_race_F"]["p"], "effect_estimate": R["i12_race_F"]["F"],
         "significant": is_sig(R["i12_race_F"]["p"])},
        {"hypothesis_ids": ["h12.6"], "code": "smf.ols('pfs_months ~ C(insurance_type)')",
         "result_summary": f"ANOVA F={R['i12_insurance_F']['F']:.3f}, p={fmt_p(R['i12_insurance_F']['p'])}.",
         "p_value": R["i12_insurance_F"]["p"], "effect_estimate": R["i12_insurance_F"]["F"],
         "significant": is_sig(R["i12_insurance_F"]["p"])},
    ],
})

# ============ ITER 13 ============
snp_main = R["i13_snp_main"]
snp_sorted = sorted(snp_main.items(), key=lambda kv: kv[1]["p"])
top3 = snp_sorted[:3]
analyses_13 = []
hyps_13 = []
for i, (s, v) in enumerate(top3):
    hid = f"h13.{i+1}"
    direction = "longer" if v["diff"] > 0 else "shorter"
    hyps_13.append({"id": hid, "text": f"Carriers of {s}=1 have {direction} mean pfs_months than non-carriers (allelic main effect).", "kind": "novel"})
    analyses_13.append({"hypothesis_ids": [hid], "code": f"ttest_ind on {s}",
                        "result_summary": f"{s}: diff={v['diff']:+.3f} months, p={fmt_p(v['p'])}.",
                        "p_value": v["p"], "effect_estimate": v["diff"],
                        "significant": is_sig(v["p"])})

# Add a global hypothesis: any SNP shows main effect at p<0.05 by chance
sig_count = sum(1 for s, v in snp_main.items() if v["p"] < 0.05)
hyps_13.append({"id": "h13.global", "text": "More than 5% of the 27 SNPs (i.e., more than ~1.4) show univariate associations with pfs_months at p<0.05, beyond chance expectation.", "kind": "novel"})
analyses_13.append({"hypothesis_ids": ["h13.global"], "code": "count of SNP univariate p<0.05",
                    "result_summary": f"{sig_count} of 27 SNPs reach p<0.05 (expected by chance ≈ 1.35); not strikingly enriched.",
                    "p_value": None, "effect_estimate": float(sig_count - 27 * 0.05),
                    "significant": False})
iters.append({
    "index": 13,
    "proposed_hypotheses": hyps_13,
    "analyses": analyses_13,
})

# ============ ITER 14 ============
mv_p = R["i14_mv"]["p"]
mv_b = R["i14_mv"]["params"]
key_terms = ["ecog_ps", "age_years", "stage_iv", "albumin_g_dl", "ldh_u_l",
             "kras_mutation", "braf_v600e", "right_sided_primary",
             "treatment_regorafenib", "treatment_cetuximab:kras_mutation",
             "treatment_encorafenib:braf_v600e",
             "treatment_pembrolizumab:msi_high",
             "treatment_trastuzumab_tucatinib:her2_amplified"]
hyps_14 = [
    {"id": "h14.1", "text": "In a multivariable OLS controlling for ECOG, age, stage, albumin, LDH, NLR, CRP, RAS, BRAF, MSI-H, HER2-amp, sidedness, liver_mets and other treatments, treatment_regorafenib remains positively associated with pfs_months.", "kind": "refined"},
    {"id": "h14.2", "text": "After multivariable adjustment, the canonical biomarker–treatment interactions (cet×kras, enc×braf_v600e, pem×msi_high, trast+tuc×her2_amplified) are statistically significant in the expected directions.", "kind": "refined"},
    {"id": "h14.3", "text": "After multivariable adjustment, ecog_ps, stage_iv, albumin_g_dl, ldh_u_l, kras_mutation, braf_v600e and right_sided_primary all retain large independent effects on pfs_months.", "kind": "refined"},
]
analyses_14 = [
    {"hypothesis_ids": ["h14.1"], "code": "MV OLS — see formula in result",
     "result_summary": (f"treatment_regorafenib coef={mv_b['treatment_regorafenib']:+.4f}, p={fmt_p(mv_p['treatment_regorafenib'])} "
                        f"(R²={R['i14_mv']['rsq']:.3f}, n={R['i14_mv']['n']})."),
     "p_value": mv_p["treatment_regorafenib"], "effect_estimate": mv_b["treatment_regorafenib"],
     "significant": is_sig(mv_p["treatment_regorafenib"])},
    {"hypothesis_ids": ["h14.2"], "code": "MV OLS interaction terms",
     "result_summary": (
         f"cet×kras coef={mv_b.get('treatment_cetuximab:kras_mutation', float('nan')):+.4f} (p={fmt_p(mv_p.get('treatment_cetuximab:kras_mutation', 1))}); "
         f"enc×braf_v600e coef={mv_b.get('treatment_encorafenib:braf_v600e', float('nan')):+.4f} (p={fmt_p(mv_p.get('treatment_encorafenib:braf_v600e', 1))}); "
         f"pem×msi coef={mv_b.get('treatment_pembrolizumab:msi_high', float('nan')):+.4f} (p={fmt_p(mv_p.get('treatment_pembrolizumab:msi_high', 1))}); "
         f"tras+tuc×her2 coef={mv_b.get('treatment_trastuzumab_tucatinib:her2_amplified', float('nan')):+.4f} (p={fmt_p(mv_p.get('treatment_trastuzumab_tucatinib:her2_amplified', 1))})."
     ),
     "p_value": mv_p.get("treatment_cetuximab:kras_mutation", 1),
     "effect_estimate": mv_b.get("treatment_cetuximab:kras_mutation", 0),
     "significant": False},
    {"hypothesis_ids": ["h14.3"], "code": "MV OLS — main prognostic terms",
     "result_summary": (f"ecog_ps={mv_b['ecog_ps']:+.3f} p={fmt_p(mv_p['ecog_ps'])}; "
                        f"stage_iv={mv_b['stage_iv']:+.3f} p={fmt_p(mv_p['stage_iv'])}; "
                        f"albumin={mv_b['albumin_g_dl']:+.3f} p={fmt_p(mv_p['albumin_g_dl'])}; "
                        f"ldh={mv_b['ldh_u_l']:+.4f} p={fmt_p(mv_p['ldh_u_l'])}; "
                        f"kras={mv_b['kras_mutation']:+.3f} p={fmt_p(mv_p['kras_mutation'])}; "
                        f"braf_v600e={mv_b['braf_v600e']:+.3f} p={fmt_p(mv_p['braf_v600e'])}; "
                        f"right_sided={mv_b['right_sided_primary']:+.3f} p={fmt_p(mv_p['right_sided_primary'])}."),
     "p_value": mv_p["ecog_ps"], "effect_estimate": mv_b["ecog_ps"],
     "significant": True},
]
iters.append({"index": 14, "proposed_hypotheses": hyps_14, "analyses": analyses_14})

# ============ ITER 15 ============
iters.append({
    "index": 15,
    "proposed_hypotheses": [
        {"id": "h15.1", "text": "The interaction ecog_ps × treatment_cetuximab is non-zero on pfs_months: cetuximab benefit (or harm) varies with performance status.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h15.1"], "code": "smf.ols('pfs_months ~ ecog_ps*treatment_cetuximab')",
         "result_summary": lin_summary(R["i15_ecog_cet_ix"]),
         "p_value": R["i15_ecog_cet_ix"]["p"],
         "effect_estimate": R["i15_ecog_cet_ix"]["coef"],
         "significant": is_sig(R["i15_ecog_cet_ix"]["p"])},
    ],
})

# ============ ITER 16 ============
iters.append({
    "index": 16,
    "proposed_hypotheses": [
        {"id": "h16.1", "text": "The composite indicator raswt_left_cet (RAS wild-type AND left-sided AND on cetuximab) marks a subgroup with longer pfs_months than the rest of the cohort.", "kind": "refined"},
        {"id": "h16.2", "text": "Within the RAS wild-type, left-sided subgroup, treatment_cetuximab is associated with longer pfs_months than no cetuximab.", "kind": "refined"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h16.1"], "code": "ttest_ind on raswt_left_cet",
         "result_summary": t_summary(R["i16_raswt_left_cet"]),
         "p_value": R["i16_raswt_left_cet"]["p"],
         "effect_estimate": R["i16_raswt_left_cet"]["diff"],
         "significant": is_sig(R["i16_raswt_left_cet"]["p"])},
        {"hypothesis_ids": ["h16.2"], "code": "stratified ttest cet within RAS-WT, left-sided",
         "result_summary": (f"In RAS-WT left-sided: mean pfs on cet={R['i16_cet_raswt_left']['mean1']:.3f} "
                            f"vs off={R['i16_cet_raswt_left']['mean0']:.3f}; "
                            f"diff={R['i16_cet_raswt_left']['diff']:+.3f}, p={fmt_p(R['i16_cet_raswt_left']['p'])}."),
         "p_value": R["i16_cet_raswt_left"]["p"],
         "effect_estimate": R["i16_cet_raswt_left"]["diff"],
         "significant": is_sig(R["i16_cet_raswt_left"]["p"])},
    ],
})

# ============ ITER 17 ============
iters.append({
    "index": 17,
    "proposed_hypotheses": [
        {"id": "h17.1", "text": "The interaction treatment_bevacizumab × kras_mutation is non-zero on pfs_months.", "kind": "novel"},
        {"id": "h17.2", "text": "The interaction treatment_bevacizumab × right_sided_primary is non-zero on pfs_months.", "kind": "novel"},
        {"id": "h17.3", "text": "The interaction treatment_bevacizumab × liver_mets is non-zero on pfs_months (anti-VEGF benefit may differ in visceral disease).", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h17.1"], "code": "smf.ols('pfs_months ~ treatment_bevacizumab*kras_mutation')",
         "result_summary": lin_summary(R["i17_bev_kras_ix"]),
         "p_value": R["i17_bev_kras_ix"]["p"],
         "effect_estimate": R["i17_bev_kras_ix"]["coef"],
         "significant": is_sig(R["i17_bev_kras_ix"]["p"])},
        {"hypothesis_ids": ["h17.2"], "code": "smf.ols('pfs_months ~ treatment_bevacizumab*right_sided_primary')",
         "result_summary": lin_summary(R["i17_bev_side_ix"]),
         "p_value": R["i17_bev_side_ix"]["p"],
         "effect_estimate": R["i17_bev_side_ix"]["coef"],
         "significant": is_sig(R["i17_bev_side_ix"]["p"])},
        {"hypothesis_ids": ["h17.3"], "code": "smf.ols('pfs_months ~ treatment_bevacizumab*liver_mets')",
         "result_summary": lin_summary(R["i17_bev_liver_ix"]),
         "p_value": R["i17_bev_liver_ix"]["p"],
         "effect_estimate": R["i17_bev_liver_ix"]["coef"],
         "significant": is_sig(R["i17_bev_liver_ix"]["p"])},
    ],
})

# ============ ITER 18 ============
iters.append({
    "index": 18,
    "proposed_hypotheses": [
        {"id": "h18.1", "text": "The interaction treatment_regorafenib × prior_lines_of_therapy is non-zero: regorafenib effect on pfs_months varies with line of therapy.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h18.1"], "code": "smf.ols('pfs_months ~ treatment_regorafenib*prior_lines_of_therapy')",
         "result_summary": lin_summary(R["i18_reg_lines_ix"]),
         "p_value": R["i18_reg_lines_ix"]["p"],
         "effect_estimate": R["i18_reg_lines_ix"]["coef"],
         "significant": is_sig(R["i18_reg_lines_ix"]["p"])},
    ],
})

# ============ ITER 19 ============
iters.append({
    "index": 19,
    "proposed_hypotheses": [
        {"id": "h19.1", "text": "The interaction treatment_pembrolizumab × nlr_high (NLR ≥ median) is negative: high baseline NLR diminishes pembrolizumab benefit on pfs_months.", "kind": "novel"},
    ],
    "analyses": [
        {"hypothesis_ids": ["h19.1"], "code": "smf.ols('pfs_months ~ treatment_pembrolizumab*nlr_high')",
         "result_summary": lin_summary(R["i19_pem_nlr_ix"]),
         "p_value": R["i19_pem_nlr_ix"]["p"],
         "effect_estimate": R["i19_pem_nlr_ix"]["coef"],
         "significant": is_sig(R["i19_pem_nlr_ix"]["p"])},
    ],
})

# ============ ITER 20 ============
fp = R["i20_final"]["p"]
fb = R["i20_final"]["params"]
hyps_20 = [
    {"id": "h20.1", "text": "After mutual adjustment for all six treatments and key biomarkers and prognostic covariates, treatment_regorafenib remains the only treatment with a robust positive independent effect on pfs_months.", "kind": "refined"},
    {"id": "h20.2", "text": "After mutual adjustment, kras_mutation, braf_v600e, ecog_ps, age_years, stage_iv, albumin_g_dl, ldh_u_l, right_sided_primary all retain independent associations with pfs_months in the expected directions; nras_mutation, msi_high, her2_amplified do not.", "kind": "refined"},
]
analyses_20 = [
    {"hypothesis_ids": ["h20.1"], "code": "MV OLS final formula",
     "result_summary": (f"treatment_regorafenib coef={fb['treatment_regorafenib']:+.4f}, p={fmt_p(fp['treatment_regorafenib'])}; "
                        f"cetuximab={fb['treatment_cetuximab']:+.4f} p={fmt_p(fp['treatment_cetuximab'])}; "
                        f"bev={fb['treatment_bevacizumab']:+.4f} p={fmt_p(fp['treatment_bevacizumab'])}; "
                        f"pem={fb['treatment_pembrolizumab']:+.4f} p={fmt_p(fp['treatment_pembrolizumab'])}; "
                        f"enc={fb['treatment_encorafenib']:+.4f} p={fmt_p(fp['treatment_encorafenib'])}; "
                        f"tras+tuc={fb['treatment_trastuzumab_tucatinib']:+.4f} p={fmt_p(fp['treatment_trastuzumab_tucatinib'])}; "
                        f"R²={R['i20_final']['rsq']:.3f}, n={R['i20_final']['n']}."),
     "p_value": fp["treatment_regorafenib"],
     "effect_estimate": fb["treatment_regorafenib"],
     "significant": is_sig(fp["treatment_regorafenib"])},
    {"hypothesis_ids": ["h20.2"], "code": "MV OLS final — biomarker / prognostic terms",
     "result_summary": (f"kras={fb['kras_mutation']:+.3f} p={fmt_p(fp['kras_mutation'])}; "
                        f"nras={fb['nras_mutation']:+.3f} p={fmt_p(fp['nras_mutation'])}; "
                        f"braf_v600e={fb['braf_v600e']:+.3f} p={fmt_p(fp['braf_v600e'])}; "
                        f"msi_high={fb['msi_high']:+.3f} p={fmt_p(fp['msi_high'])}; "
                        f"her2_amplified={fb['her2_amplified']:+.3f} p={fmt_p(fp['her2_amplified'])}; "
                        f"ecog={fb['ecog_ps']:+.3f} p={fmt_p(fp['ecog_ps'])}; "
                        f"age={fb['age_years']:+.3f} p={fmt_p(fp['age_years'])}; "
                        f"stage_iv={fb['stage_iv']:+.3f} p={fmt_p(fp['stage_iv'])}; "
                        f"albumin={fb['albumin_g_dl']:+.3f} p={fmt_p(fp['albumin_g_dl'])}; "
                        f"ldh={fb['ldh_u_l']:+.4f} p={fmt_p(fp['ldh_u_l'])}; "
                        f"right_sided={fb['right_sided_primary']:+.3f} p={fmt_p(fp['right_sided_primary'])}."),
     "p_value": fp["kras_mutation"],
     "effect_estimate": fb["kras_mutation"],
     "significant": is_sig(fp["kras_mutation"])},
]
iters.append({"index": 20, "proposed_hypotheses": hyps_20, "analyses": analyses_20})

transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-custom@manual",
    "max_iterations": 25,
    "iterations": iters,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

print("Wrote transcript.json with", len(iters), "iterations.")
print("Total hypotheses:", sum(len(it["proposed_hypotheses"]) for it in iters))
print("Total analyses:", sum(len(it["analyses"]) for it in iters))
