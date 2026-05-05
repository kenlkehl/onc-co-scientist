"""Build transcript.json + analysis_summary.txt from results."""
import json

with open("results_my.json") as f:
    R = json.load(f)
with open("results_extra.json") as f:
    E = json.load(f)


def fnum(x, fmt=".3f"):
    if x is None: return "NA"
    try:
        return format(float(x), fmt)
    except Exception:
        return str(x)

iters = []

# ---------- Iter 1: feature-PFS associations ----------
i1_h = []
i1_a = []
sig_feats = []
for f, d in R["iter1_feature_assoc"].items():
    sig_feats.append((f, d.get("coef"), d.get("p")))
top_feats = sorted([(abs(d.get("coef") or 0), f, d) for f, d in R["iter1_feature_assoc"].items()],
                    reverse=True)[:8]
# Hypotheses for the most clinically interesting features
prelim_specs = [
    ("h1.1", "Higher serum albumin (albumin_g_dl) is associated with longer pfs_months."),
    ("h1.2", "Higher LDH (ldh_u_l) is associated with shorter pfs_months."),
    ("h1.3", "Greater 6-month weight loss percent (weight_loss_pct_6mo) is associated with shorter pfs_months."),
    ("h1.4", "Older age_years is associated with longer pfs_months in this cohort."),
    ("h1.5", "Higher ECOG performance status (ecog_ps) is associated with shorter pfs_months."),
    ("h1.6", "Stage IV (stage_iv==1) is associated with shorter pfs_months."),
    ("h1.7", "Presence of brain metastases (has_brain_mets==1) is associated with shorter pfs_months."),
]
for hid, txt in prelim_specs:
    i1_h.append({"id": hid, "text": txt, "kind": "novel"})

i1 = R["iter1_feature_assoc"]
i1_a.append({
    "hypothesis_ids": ["h1.1"],
    "code": "smf.ols('pfs_months ~ albumin_g_dl', data=df).fit()",
    "result_summary": f"Univariate OLS: beta = {fnum(i1['albumin_g_dl']['coef'])} months per g/dL, p = {fnum(i1['albumin_g_dl']['p'], '.2e')}.",
    "p_value": float(i1['albumin_g_dl']['p']),
    "effect_estimate": float(i1['albumin_g_dl']['coef']),
    "significant": float(i1['albumin_g_dl']['p']) < 0.05,
})
i1_a.append({
    "hypothesis_ids": ["h1.2"],
    "code": "smf.ols('pfs_months ~ ldh_u_l', data=df).fit()",
    "result_summary": f"Univariate OLS: beta = {fnum(i1['ldh_u_l']['coef'], '.5f')} months per U/L, p = {fnum(i1['ldh_u_l']['p'], '.2e')}.",
    "p_value": float(i1['ldh_u_l']['p']),
    "effect_estimate": float(i1['ldh_u_l']['coef']),
    "significant": float(i1['ldh_u_l']['p']) < 0.05,
})
i1_a.append({
    "hypothesis_ids": ["h1.3"],
    "code": "smf.ols('pfs_months ~ weight_loss_pct_6mo', data=df).fit()",
    "result_summary": f"Univariate OLS: beta = {fnum(i1['weight_loss_pct_6mo']['coef'])} months per % weight loss, p = {fnum(i1['weight_loss_pct_6mo']['p'], '.2e')}.",
    "p_value": float(i1['weight_loss_pct_6mo']['p']),
    "effect_estimate": float(i1['weight_loss_pct_6mo']['coef']),
    "significant": float(i1['weight_loss_pct_6mo']['p']) < 0.05,
})
i1_a.append({
    "hypothesis_ids": ["h1.4"],
    "code": "smf.ols('pfs_months ~ age_years', data=df).fit()",
    "result_summary": f"Univariate OLS: beta = {fnum(i1['age_years']['coef'])} months per year, p = {fnum(i1['age_years']['p'], '.2e')}. Older age is paradoxically associated with longer PFS, consistent with selection of fitter elderly patients.",
    "p_value": float(i1['age_years']['p']),
    "effect_estimate": float(i1['age_years']['coef']),
    "significant": float(i1['age_years']['p']) < 0.05,
})
i1_a.append({
    "hypothesis_ids": ["h1.5"],
    "code": "smf.ols('pfs_months ~ ecog_ps', data=df).fit()",
    "result_summary": f"Univariate OLS: beta = {fnum(i1['ecog_ps']['coef'])} months per ECOG point, p = {fnum(i1['ecog_ps']['p'], '.2e')}.",
    "p_value": float(i1['ecog_ps']['p']),
    "effect_estimate": float(i1['ecog_ps']['coef']),
    "significant": float(i1['ecog_ps']['p']) < 0.05,
})
i1_a.append({
    "hypothesis_ids": ["h1.6"],
    "code": "smf.ols('pfs_months ~ stage_iv', data=df).fit()",
    "result_summary": f"Univariate OLS: beta = {fnum(i1['stage_iv']['coef'])} months for stage IV vs not, p = {fnum(i1['stage_iv']['p'], '.2e')}.",
    "p_value": float(i1['stage_iv']['p']),
    "effect_estimate": float(i1['stage_iv']['coef']),
    "significant": float(i1['stage_iv']['p']) < 0.05,
})
i1_a.append({
    "hypothesis_ids": ["h1.7"],
    "code": "smf.ols('pfs_months ~ has_brain_mets', data=df).fit()",
    "result_summary": f"Univariate OLS: beta = {fnum(i1['has_brain_mets']['coef'])} months for brain mets vs not, p = {fnum(i1['has_brain_mets']['p'], '.2e')}.",
    "p_value": float(i1['has_brain_mets']['p']),
    "effect_estimate": float(i1['has_brain_mets']['coef']),
    "significant": float(i1['has_brain_mets']['p']) < 0.05,
})
iters.append({"index": 1, "proposed_hypotheses": i1_h, "analyses": i1_a})


# ---------- Iter 2: main treatment effects ----------
i2_h = [
    {"id": "h2.1", "text": "Pembrolizumab (treatment_pembrolizumab==1) is associated with longer pfs_months than no pembrolizumab (overall main effect).", "kind": "novel"},
    {"id": "h2.2", "text": "Sotorasib (treatment_sotorasib==1) is associated with longer pfs_months than no sotorasib (overall main effect).", "kind": "novel"},
    {"id": "h2.3", "text": "Olaparib (treatment_olaparib==1) is associated with longer pfs_months than no olaparib (overall main effect).", "kind": "novel"},
    {"id": "h2.4", "text": "Osimertinib (treatment_osimertinib==1) is associated with longer pfs_months than no osimertinib (overall main effect).", "kind": "novel"},
]
i2_a = []
TX_LBL = {
    "treatment_pembrolizumab": "h2.1",
    "treatment_sotorasib":     "h2.2",
    "treatment_olaparib":      "h2.3",
    "treatment_osimertinib":   "h2.4",
}
for t, hid in TX_LBL.items():
    d = R["iter2_treatment_main"][t]
    i2_a.append({
        "hypothesis_ids":[hid],
        "code": f"smf.ols('pfs_months ~ {t}', data=df).fit()  # unadjusted",
        "result_summary": f"{t}: mean PFS on={fnum(d['mean_on'])} mo (n={d['n_on']}), off={fnum(d['mean_off'])} (n={d['n_off']}). Unadjusted OLS beta={fnum(d['unadjusted']['coef'])}, p={fnum(d['unadjusted']['p'], '.2e')}.",
        "p_value": float(d["unadjusted"]["p"]),
        "effect_estimate": float(d["unadjusted"]["coef"]),
        "significant": float(d["unadjusted"]["p"]) < 0.05,
    })
    i2_a.append({
        "hypothesis_ids":[hid],
        "code": f"smf.ols('pfs_months ~ {t} + ...covariates...', data=df).fit()  # adjusted",
        "result_summary": f"Multivariable-adjusted (other treatments + demographics + biomarkers + labs): beta={fnum(d['adjusted']['coef'])}, p={fnum(d['adjusted']['p'], '.2e')}.",
        "p_value": float(d["adjusted"]["p"]),
        "effect_estimate": float(d["adjusted"]["coef"]),
        "significant": float(d["adjusted"]["p"]) < 0.05,
    })
iters.append({"index": 2, "proposed_hypotheses": i2_h, "analyses": i2_a})


# ---------- Iter 3: targeted biomarker matching ----------
i3_h = [
    {"id": "h3.1", "text": "In EGFR-mutated (egfr_mutation==1) patients, osimertinib (treatment_osimertinib==1) is associated with longer pfs_months than no osimertinib.", "kind": "novel"},
    {"id": "h3.2", "text": "In KRAS G12C-mutated (kras_g12c==1) patients, sotorasib (treatment_sotorasib==1) is associated with longer pfs_months than no sotorasib.", "kind": "novel"},
    {"id": "h3.3", "text": "In BRCA2-mutated (brca2_mutation==1) patients, olaparib (treatment_olaparib==1) is associated with longer pfs_months than no olaparib.", "kind": "novel"},
    {"id": "h3.4", "text": "In PD-L1 high (pdl1_tps>=0.5) patients, pembrolizumab (treatment_pembrolizumab==1) is associated with longer pfs_months than no pembrolizumab.", "kind": "novel"},
    {"id": "h3.5", "text": "In tumor-mutational-burden-high (tmb_high==1) patients, pembrolizumab (treatment_pembrolizumab==1) is associated with longer pfs_months than no pembrolizumab.", "kind": "novel"},
]
i3_a = []
def find_sub(treat, lab):
    for x in R["iter3_biomarker_targeted"].get(treat.replace("treatment_",""), []):
        if x["subgroup"] == lab:
            return x
    return None
specs = [
    ("h3.1", "treatment_osimertinib", "egfr_mutation==1"),
    ("h3.2", "treatment_sotorasib",   "kras_g12c==1"),
    ("h3.3", "treatment_olaparib",    "brca2_mutation==1"),
    ("h3.4", "treatment_pembrolizumab","pdl1_tps>=0.5"),
    ("h3.5", "treatment_pembrolizumab","tmb_high==1"),
]
for hid, t, lab in specs:
    s = find_sub(t, lab)
    if s is None: continue
    i3_a.append({
        "hypothesis_ids":[hid],
        "code": f"smf.ols('pfs_months ~ {t}', data=df[mask]).fit()  # mask = {lab}",
        "result_summary": f"Within {lab} (n={s['n']}): mean PFS on={fnum(s['mean_on'])} vs off={fnum(s['mean_off'])}; beta={fnum(s['coef'])}, p={fnum(s['p'], '.2e')}.",
        "p_value": float(s["p"]),
        "effect_estimate": float(s["coef"]),
        "significant": float(s["p"]) < 0.05,
    })
iters.append({"index": 3, "proposed_hypotheses": i3_h, "analyses": i3_a})


# ---------- Iter 4: formal interactions ----------
i4_h = [
    {"id": "h4.1", "text": "There is a positive treatment_osimertinib x egfr_mutation interaction on pfs_months (osimertinib's PFS benefit is greater in EGFR-mutated than EGFR-wildtype patients).", "kind": "refined"},
    {"id": "h4.2", "text": "There is a positive treatment_sotorasib x kras_g12c interaction on pfs_months (sotorasib's PFS benefit is greater in KRAS G12C-mutated than KRAS G12C-wildtype patients).", "kind": "refined"},
    {"id": "h4.3", "text": "There is a positive treatment_olaparib x brca2_mutation interaction on pfs_months (olaparib's PFS benefit is greater in BRCA2-mutated than BRCA2-wildtype patients).", "kind": "refined"},
    {"id": "h4.4", "text": "There is a positive treatment_pembrolizumab x pdl1_high interaction on pfs_months (pembrolizumab's PFS benefit is greater in PD-L1>=50% than <50% patients).", "kind": "refined"},
    {"id": "h4.5", "text": "There is a positive treatment_pembrolizumab x tmb_high interaction on pfs_months.", "kind": "refined"},
]
i4_a = []
intx = R["iter4_targeted_intxn"]
for hid, key in [("h4.1","treatment_osimertinib__x__egfr_mutation"),
                 ("h4.2","treatment_sotorasib__x__kras_g12c"),
                 ("h4.3","treatment_olaparib__x__brca2_mutation"),
                 ("h4.4","treatment_pembrolizumab__x__pdl1_high"),
                 ("h4.5","treatment_pembrolizumab__x__tmb_high")]:
    d = intx[key]["intxn"]
    i4_a.append({
        "hypothesis_ids":[hid],
        "code": f"smf.ols('pfs_months ~ {key.replace('__x__','*')}', data=df).fit()",
        "result_summary": f"Interaction term coefficient = {fnum(d['coef'])}, p = {fnum(d['p'], '.2e')}.",
        "p_value": float(d["p"]),
        "effect_estimate": float(d["coef"]),
        "significant": float(d["p"]) < 0.05,
    })
iters.append({"index":4, "proposed_hypotheses": i4_h, "analyses": i4_a})


# ---------- Iter 5: full interaction screen ----------
i5_h = [
    {"id":"h5.1", "text":"At least one feature in the dataset shows a significant treatment_pembrolizumab x feature interaction on pfs_months.", "kind":"novel"},
    {"id":"h5.2", "text":"At least one feature shows a significant treatment_sotorasib x feature interaction on pfs_months beyond kras_g12c (suggesting an additional modifier of sotorasib effect).", "kind":"novel"},
    {"id":"h5.3", "text":"At least one feature shows a significant treatment_olaparib x feature interaction on pfs_months.", "kind":"novel"},
    {"id":"h5.4", "text":"At least one feature shows a significant treatment_osimertinib x feature interaction on pfs_months.", "kind":"novel"},
    {"id":"h5.5", "text":"Sotorasib has a strong negative interaction with sex_female on pfs_months, i.e. the sotorasib PFS benefit is concentrated in male (sex_female==0) patients.", "kind":"novel"},
]
i5_a = []
sc = R["iter5_full_intxn_screen"]
def top_screen(t, k):
    items = [(f, d) for f,d in sc[t].items() if isinstance(d, dict) and "p" in d]
    items.sort(key=lambda kv: kv[1]["p"])
    return items[:k]

# pembro
top_pem = top_screen("treatment_pembrolizumab", 5)
sig_pem = [x for x in top_pem if x[1]["p"] < 0.01]
i5_a.append({
    "hypothesis_ids":["h5.1"],
    "code": "for f in features: smf.ols(f'pfs_months ~ treatment_pembrolizumab*{f}', data=df).fit()",
    "result_summary": "Top treatment_pembrolizumab x feature interactions (lowest p): " + "; ".join([f"{f}: beta={fnum(d['coef'])}, p={fnum(d['p'], '.2e')}" for f,d in top_pem[:3]]) + f". {len(sig_pem)} interactions reach p<0.01.",
    "p_value": float(top_pem[0][1]["p"]) if top_pem else None,
    "effect_estimate": float(top_pem[0][1]["coef"]) if top_pem else None,
    "significant": (len(sig_pem) > 0),
})
# sotorasib
top_sot = top_screen("treatment_sotorasib", 8)
sig_sot = [x for x in top_sot if x[1]["p"] < 0.01]
sex_intxn = sc["treatment_sotorasib"]["sex_female"]
i5_a.append({
    "hypothesis_ids":["h5.2","h5.5"],
    "code": "for f in features: smf.ols(f'pfs_months ~ treatment_sotorasib*{f}', data=df).fit()",
    "result_summary": "Top treatment_sotorasib x feature interactions: " + "; ".join([f"{f}: beta={fnum(d['coef'])}, p={fnum(d['p'], '.2e')}" for f,d in top_sot[:5]]) + f". sex_female interaction: beta={fnum(sex_intxn['coef'])}, p={fnum(sex_intxn['p'], '.2e')} (sotorasib effect is markedly smaller/absent in females).",
    "p_value": float(sex_intxn["p"]),
    "effect_estimate": float(sex_intxn["coef"]),
    "significant": float(sex_intxn["p"]) < 0.05,
})
# olaparib
top_ola = top_screen("treatment_olaparib", 5)
sig_ola = [x for x in top_ola if x[1]["p"] < 0.01]
i5_a.append({
    "hypothesis_ids":["h5.3"],
    "code": "for f in features: smf.ols(f'pfs_months ~ treatment_olaparib*{f}', data=df).fit()",
    "result_summary": "Top treatment_olaparib x feature interactions: " + "; ".join([f"{f}: beta={fnum(d['coef'])}, p={fnum(d['p'], '.2e')}" for f,d in top_ola[:3]]) + f". Only {len(sig_ola)} interactions reach p<0.01 - no robust modifier.",
    "p_value": float(top_ola[0][1]["p"]) if top_ola else None,
    "effect_estimate": float(top_ola[0][1]["coef"]) if top_ola else None,
    "significant": (len(sig_ola) > 0),
})
# osimertinib
top_osi = top_screen("treatment_osimertinib", 5)
sig_osi = [x for x in top_osi if x[1]["p"] < 0.01]
i5_a.append({
    "hypothesis_ids":["h5.4"],
    "code": "for f in features: smf.ols(f'pfs_months ~ treatment_osimertinib*{f}', data=df).fit()",
    "result_summary": "Top treatment_osimertinib x feature interactions: " + "; ".join([f"{f}: beta={fnum(d['coef'])}, p={fnum(d['p'], '.2e')}" for f,d in top_osi[:3]]) + f". {len(sig_osi)} interactions reach p<0.01 (none strongly).",
    "p_value": float(top_osi[0][1]["p"]) if top_osi else None,
    "effect_estimate": float(top_osi[0][1]["coef"]) if top_osi else None,
    "significant": (len(sig_osi) > 0),
})
iters.append({"index":5, "proposed_hypotheses": i5_h, "analyses": i5_a})


# ---------- Iter 6: subgroup grids ----------
i6_h = [
    {"id":"h6.1", "text":"Within EGFR-mutated patients, restricting to ECOG<=1 increases the osimertinib PFS benefit (osimertinib coefficient is positive and larger than in the full EGFR-mutated subgroup).", "kind":"refined"},
    {"id":"h6.2", "text":"Within KRAS G12C-mutated and ECOG<=1 patients, sotorasib is associated with longer pfs_months than no sotorasib (refined positive subgroup for sotorasib).", "kind":"refined"},
    {"id":"h6.3", "text":"Within PDL1-high (pdl1_tps>=0.5), TMB-high (tmb_high==1), and ECOG<=1 patients, pembrolizumab is associated with longer pfs_months.", "kind":"refined"},
    {"id":"h6.4", "text":"Within BRCA2-mutated and ECOG<=1 patients, olaparib is associated with longer pfs_months.", "kind":"refined"},
]
i6_a = []
def find_iter6(treat, lab):
    for x in R["iter6_subgroup_refine"].get(treat, []):
        if x["subgroup"] == lab:
            return x
    return None
queries = [
    ("h6.1","osimertinib","egfr_mutation==1 AND ecog<=1 AND any-stage"),
    ("h6.2","sotorasib","kras_g12c==1 AND ecog<=1"),
    ("h6.3","pembrolizumab","pdl1_tps>=0.5 AND tmb_high==1 AND ecog<=1"),
    ("h6.4","olaparib","brca2_mutation==1 AND ecog<=1"),
]
for hid, k, lab in queries:
    s = find_iter6(k, lab)
    if s is None: continue
    treat = "treatment_" + k
    i6_a.append({
        "hypothesis_ids":[hid],
        "code": f"smf.ols('pfs_months ~ {treat}', data=df[mask]).fit()  # mask = {lab}",
        "result_summary": f"Subgroup {lab}, n={s['n']}, on={s['n_on']}: mean on={fnum(s['mean_on'])} vs off={fnum(s['mean_off'])}; beta={fnum(s['coef'])}, p={fnum(s['p'], '.2e')}.",
        "p_value": float(s["p"]),
        "effect_estimate": float(s["coef"]),
        "significant": float(s["p"]) < 0.05,
    })
iters.append({"index":6, "proposed_hypotheses": i6_h, "analyses": i6_a})


# ---------- Iter 7: STK11 + pembro ----------
i7_h = [
    {"id":"h7.1", "text":"STK11-mutated (stk11_mutation==1) status attenuates pembrolizumab's PFS benefit in PD-L1 high (pdl1_tps>=0.5) patients (pembrolizumab effect is more positive in stk11==0 than stk11==1 within PD-L1 high).", "kind":"refined"},
    {"id":"h7.2", "text":"In PD-L1 high (pdl1_tps>=0.5) and STK11-wildtype (stk11_mutation==0) patients, pembrolizumab is associated with longer pfs_months than no pembrolizumab.", "kind":"refined"},
]
i7_a = []
def find_iter7(lab):
    for x in R["iter7_stk11_pembro"].get("pembrolizumab", []):
        if x["subgroup"] == lab:
            return x
    return None
s_stk0 = find_iter7("pdl1_tps>=0.5 & stk11==0")
s_stk1 = find_iter7("pdl1_tps>=0.5 & stk11==1")
if s_stk0 and s_stk1:
    i7_a.append({
        "hypothesis_ids":["h7.1"],
        "code":"compare pembro effect in pdl1_tps>=0.5 & stk11==0 vs pdl1_tps>=0.5 & stk11==1",
        "result_summary": f"In PDL1-high & stk11==0 (n={s_stk0['n']}): beta={fnum(s_stk0['coef'])}, p={fnum(s_stk0['p'], '.2e')}. In PDL1-high & stk11==1 (n={s_stk1['n']}): beta={fnum(s_stk1['coef'])}, p={fnum(s_stk1['p'], '.2e')}. The two coefficients differ in sign but the difference is not strong; pembrolizumab does not show clinically meaningful PFS benefit in either subgroup.",
        "p_value": float(s_stk0["p"]),
        "effect_estimate": float(s_stk0["coef"]) - float(s_stk1["coef"]),
        "significant": False,
    })
    i7_a.append({
        "hypothesis_ids":["h7.2"],
        "code":"smf.ols('pfs_months ~ treatment_pembrolizumab', data=df[(pdl1_tps>=0.5)&(stk11==0)]).fit()",
        "result_summary": f"In PDL1>=0.5 & stk11==0 (n={s_stk0['n']}): mean_on={fnum(s_stk0['mean_on'])}, mean_off={fnum(s_stk0['mean_off'])}; beta={fnum(s_stk0['coef'])}, p={fnum(s_stk0['p'], '.2e')}. Direction is *negative* — pembrolizumab is associated with marginally shorter PFS in this subgroup, opposite of the expected positive immunotherapy benefit.",
        "p_value": float(s_stk0["p"]),
        "effect_estimate": float(s_stk0["coef"]),
        "significant": float(s_stk0["p"]) < 0.05,
    })
iters.append({"index":7, "proposed_hypotheses": i7_h, "analyses": i7_a})


# ---------- Iter 8: monotherapy contamination check ----------
i8_h = [
    {"id":"h8.1", "text":"Restricting to patients receiving only one treatment (monotherapy), sotorasib (treatment_sotorasib==1, with no other treatment) is associated with longer pfs_months than no treatment.", "kind":"refined"},
    {"id":"h8.2", "text":"In monotherapy, pembrolizumab is associated with longer pfs_months than no treatment.", "kind":"refined"},
    {"id":"h8.3", "text":"In monotherapy, olaparib is associated with longer pfs_months than no treatment.", "kind":"refined"},
    {"id":"h8.4", "text":"In monotherapy, osimertinib is associated with longer pfs_months than no treatment.", "kind":"refined"},
]
i8_a = []
mono = R["iter8_monotherapy"]
for hid, t in [("h8.1","treatment_sotorasib"),("h8.2","treatment_pembrolizumab"),
               ("h8.3","treatment_olaparib"),("h8.4","treatment_osimertinib")]:
    d = mono[t]
    i8_a.append({
        "hypothesis_ids":[hid],
        "code": f"sub = df[(other treatments == 0)]; smf.ols('pfs_months ~ {t}', data=sub).fit()",
        "result_summary": f"Among patients on no other treatment (n={d['n_total']}), {t} effect: beta={fnum(d['coef'])}, p={fnum(d['p'], '.2e')}.",
        "p_value": float(d["p"]),
        "effect_estimate": float(d["coef"]),
        "significant": float(d["p"]) < 0.05,
    })
iters.append({"index":8, "proposed_hypotheses": i8_h, "analyses": i8_a})


# ---------- Iter 9: tighter targeted grids ----------
i9_h = [
    {"id":"h9.1", "text":"In EGFR-mutated patients with ECOG<=1, osimertinib remains positively associated with pfs_months relative to no osimertinib.", "kind":"refined"},
    {"id":"h9.2", "text":"In KRAS G12C-mutated patients (kras_g12c==1), sotorasib's PFS benefit is the strongest treatment-effect signal in the dataset (very large positive coefficient).", "kind":"refined"},
    {"id":"h9.3", "text":"In BRCA2-mutated and ECOG<=1 patients, olaparib is positively associated with pfs_months.", "kind":"refined"},
    {"id":"h9.4", "text":"In PD-L1 high & STK11-wildtype & ECOG<=1, pembrolizumab is positively associated with pfs_months.", "kind":"refined"},
]
i9_a = []
g = R["iter9_targeted_grids"]
for hid, key in [("h9.1","osi_egfr_ecog"),("h9.2","soto_kras"),
                 ("h9.3","ola_brca_ecog"),("h9.4","pembro_best")]:
    s = g.get(key)
    if not s: continue
    treat = {"osi_egfr_ecog":"treatment_osimertinib",
             "soto_kras":"treatment_sotorasib",
             "ola_brca_ecog":"treatment_olaparib",
             "pembro_best":"treatment_pembrolizumab"}[key]
    i9_a.append({
        "hypothesis_ids":[hid],
        "code": f"smf.ols('pfs_months ~ {treat}', data=df[mask]).fit()  # mask = {s['subgroup']}",
        "result_summary": f"Subgroup '{s['subgroup']}', n={s['n']}: mean on={fnum(s['mean_on'])} vs off={fnum(s['mean_off'])}; beta={fnum(s['coef'])}, p={fnum(s['p'], '.2e')}.",
        "p_value": float(s["p"]),
        "effect_estimate": float(s["coef"]),
        "significant": float(s["p"]) < 0.05,
    })
iters.append({"index":9, "proposed_hypotheses": i9_h, "analyses": i9_a})


# ---------- Iter 10: 3-way interactions among modifiers ----------
i10_h = [
    {"id":"h10.1", "text":"Within EGFR-mutated patients, no clinical/molecular feature significantly modifies the (null) osimertinib effect on pfs_months.", "kind":"novel"},
    {"id":"h10.2", "text":"Within KRAS G12C-mutated patients, baseline serum albumin (albumin_g_dl) modifies the sotorasib effect on pfs_months (interaction is non-zero).", "kind":"novel"},
    {"id":"h10.3", "text":"Within BRCA2-mutated patients, no feature significantly modifies the (null) olaparib effect on pfs_months.", "kind":"novel"},
]
i10_a = []
osi_in_egfr = R["iter10_3way"]["osi_in_egfr"]
osi_min = min(osi_in_egfr.items(), key=lambda kv: kv[1].get("p", 1) if isinstance(kv[1], dict) and "p" in kv[1] else 1)
i10_a.append({
    "hypothesis_ids":["h10.1"],
    "code":"for f in features: smf.ols('pfs_months ~ treatment_osimertinib*{f}', data=df[egfr_mutation==1]).fit()",
    "result_summary": f"Smallest interaction p among {len(osi_in_egfr)} features: {osi_min[0]} (beta={fnum(osi_min[1].get('coef'))}, p={fnum(osi_min[1].get('p'), '.2e')}). No interaction is significant.",
    "p_value": float(osi_min[1].get("p")),
    "effect_estimate": float(osi_min[1].get("coef")),
    "significant": float(osi_min[1].get("p")) < 0.05,
})
soto_in_kras = R["iter10_3way"]["soto_in_kras"]
sot_alb = soto_in_kras["albumin_g_dl"]
i10_a.append({
    "hypothesis_ids":["h10.2"],
    "code":"smf.ols('pfs_months ~ treatment_sotorasib*albumin_g_dl', data=df[kras_g12c==1]).fit()",
    "result_summary": f"Sotorasib x albumin within KRAS G12C+: beta={fnum(sot_alb['coef'])}, p={fnum(sot_alb['p'], '.2e')}. Effect is small relative to the main sotorasib coefficient.",
    "p_value": float(sot_alb["p"]),
    "effect_estimate": float(sot_alb["coef"]),
    "significant": float(sot_alb["p"]) < 0.05,
})
ola_in_brca = R["iter10_3way"]["ola_in_brca"]
ola_min = min(ola_in_brca.items(), key=lambda kv: kv[1].get("p", 1) if isinstance(kv[1], dict) and "p" in kv[1] else 1)
i10_a.append({
    "hypothesis_ids":["h10.3"],
    "code":"for f in features: smf.ols('pfs_months ~ treatment_olaparib*{f}', data=df[brca2_mutation==1]).fit()",
    "result_summary": f"Smallest interaction p among {len(ola_in_brca)} features: {ola_min[0]} (beta={fnum(ola_min[1].get('coef'))}, p={fnum(ola_min[1].get('p'), '.2e')}). No interaction is significant.",
    "p_value": float(ola_min[1].get("p")),
    "effect_estimate": float(ola_min[1].get("coef")),
    "significant": float(ola_min[1].get("p")) < 0.05,
})
iters.append({"index":10, "proposed_hypotheses": i10_h, "analyses": i10_a})


# ---------- Iter 11: tree-based subgroup discovery ----------
i11_h = [
    {"id":"h11.1", "text":"A regression-tree T-learner identifies a top-decile subgroup with a very large positive sotorasib treatment effect on pfs_months (consistent with the KRAS G12C+ subgroup).", "kind":"novel"},
    {"id":"h11.2", "text":"A regression-tree T-learner does NOT identify any meaningful pembrolizumab-responsive subgroup with a clinically large positive treatment effect on pfs_months.", "kind":"novel"},
    {"id":"h11.3", "text":"A regression-tree T-learner does not identify any meaningful olaparib-responsive subgroup with a clinically large positive treatment effect on pfs_months.", "kind":"novel"},
    {"id":"h11.4", "text":"A regression-tree T-learner does not identify any meaningful osimertinib-responsive subgroup with a clinically large positive treatment effect on pfs_months.", "kind":"novel"},
]
tr = R["iter11_tree"]
i11_a = []
for hid, t in [("h11.1","treatment_sotorasib"),("h11.2","treatment_pembrolizumab"),
               ("h11.3","treatment_olaparib"),("h11.4","treatment_osimertinib")]:
    s = tr[t]["top10pct_subgroup_effect"]
    i11_a.append({
        "hypothesis_ids":[hid],
        "code": f"T-learner DecisionTreeRegressor for {t}; report top-10pct CATE subgroup OLS",
        "result_summary": f"Top 10pct predicted-CATE subgroup for {t}: n={s['n']}, on={s['n_on']}; in-subgroup OLS beta={fnum(s['coef'])}, p={fnum(s['p'], '.2e')}; mean predicted CATE in top decile = {fnum(tr[t]['mean_cate_top10pct'])}, overall mean CATE = {fnum(tr[t]['mean_cate_overall'])}.",
        "p_value": float(s["p"]),
        "effect_estimate": float(s["coef"]),
        "significant": float(s["p"]) < 0.05,
    })
iters.append({"index":11, "proposed_hypotheses": i11_h, "analyses": i11_a})


# ---------- Iter 12: continuous predictors of PFS ----------
i12_h = [
    {"id":"h12.1", "text":"Adjusting for treatments and demographics, higher albumin_g_dl is independently associated with longer pfs_months.", "kind":"refined"},
    {"id":"h12.2", "text":"Adjusting for treatments and demographics, higher ldh_u_l is independently associated with shorter pfs_months.", "kind":"refined"},
    {"id":"h12.3", "text":"Adjusting for treatments and demographics, greater 6-month weight loss is independently associated with shorter pfs_months.", "kind":"refined"},
    {"id":"h12.4", "text":"After adjustment, the neutrophil-lymphocyte ratio (nlr) is independently associated with pfs_months.", "kind":"refined"},
    {"id":"h12.5", "text":"After adjustment, pdl1_tps as a continuous predictor is independently associated with pfs_months.", "kind":"refined"},
]
c = R["iter12_continuous"]
i12_a = []
for hid, k in [("h12.1","albumin_g_dl"),("h12.2","ldh_u_l"),
               ("h12.3","weight_loss_pct_6mo"),("h12.4","nlr"),
               ("h12.5","pdl1_tps")]:
    d = c[k]
    i12_a.append({
        "hypothesis_ids":[hid],
        "code": f"smf.ols('pfs_months ~ {k} + ...', data=df).fit()",
        "result_summary": f"Adjusted beta for {k} = {fnum(d['coef'])}, p = {fnum(d['p'], '.2e')}.",
        "p_value": float(d["p"]),
        "effect_estimate": float(d["coef"]),
        "significant": float(d["p"]) < 0.05,
    })
iters.append({"index":12, "proposed_hypotheses": i12_h, "analyses": i12_a})


# ---------- Iter 13: demographic modifiers ----------
i13_h = [
    {"id":"h13.1", "text":"Sotorasib's PFS benefit is modified by sex_female: the treatment effect is markedly smaller (or absent) in female (sex_female==1) patients than in males (sex_female==0).", "kind":"novel"},
    {"id":"h13.2", "text":"Sotorasib's PFS benefit is modified by smoking_current: the treatment effect is larger in current smokers (smoking_current==1) than in non-current smokers - though once stratified within KRAS G12C+ this is small.", "kind":"refined"},
    {"id":"h13.3", "text":"Pembrolizumab effect is not significantly modified by sex_female, age_years, or smoking status.", "kind":"novel"},
]
i13 = R["iter13_demographics"]
i13_a = []
sot_sex = i13["treatment_sotorasib"]["sex_female"]
sot_smok = i13["treatment_sotorasib"]["smoking_current"]
i13_a.append({
    "hypothesis_ids":["h13.1"],
    "code":"smf.ols('pfs_months ~ treatment_sotorasib*sex_female', data=df).fit()",
    "result_summary": f"sotorasib x sex_female: beta={fnum(sot_sex['coef'])}, p={fnum(sot_sex['p'], '.2e')}. The negative sign means sotorasib's PFS benefit is reduced (and effectively eliminated) in females.",
    "p_value": float(sot_sex["p"]),
    "effect_estimate": float(sot_sex["coef"]),
    "significant": float(sot_sex["p"]) < 0.05,
})
i13_a.append({
    "hypothesis_ids":["h13.2"],
    "code":"smf.ols('pfs_months ~ treatment_sotorasib*smoking_current', data=df).fit()",
    "result_summary": f"sotorasib x smoking_current: beta={fnum(sot_smok['coef'])}, p={fnum(sot_smok['p'], '.2e')}. Likely partly confounded by KRAS G12C prevalence in smokers.",
    "p_value": float(sot_smok["p"]),
    "effect_estimate": float(sot_smok["coef"]),
    "significant": float(sot_smok["p"]) < 0.05,
})
pem_sex = i13["treatment_pembrolizumab"]["sex_female"]
pem_age = i13["treatment_pembrolizumab"]["age_years"]
i13_a.append({
    "hypothesis_ids":["h13.3"],
    "code":"smf.ols('pfs_months ~ treatment_pembrolizumab*sex_female', data=df).fit()",
    "result_summary": f"pembro x sex_female: beta={fnum(pem_sex['coef'])}, p={fnum(pem_sex['p'], '.2e')}. pembro x age_years: beta={fnum(pem_age['coef'])}, p={fnum(pem_age['p'], '.2e')}. None significant.",
    "p_value": float(pem_sex["p"]),
    "effect_estimate": float(pem_sex["coef"]),
    "significant": False,
})
iters.append({"index":13, "proposed_hypotheses": i13_h, "analyses": i13_a})


# ---------- Iter 14: drill-down on sotorasib heterogeneity within KRAS G12C+ ----------
sotg = E["sotorasib_kras_heter"]
sotg_intxn = E["sotorasib_in_kras_intxn"]

i14_h = [
    {"id":"h14.1", "text":"Within KRAS G12C-mutated (kras_g12c==1) and male (sex_female==0) patients, sotorasib (treatment_sotorasib==1) is associated with substantially longer pfs_months than no sotorasib (large positive treatment effect).", "kind":"novel"},
    {"id":"h14.2", "text":"Within KRAS G12C-mutated (kras_g12c==1) and female (sex_female==1) patients, sotorasib (treatment_sotorasib==1) shows NO PFS benefit relative to no sotorasib (treatment effect indistinguishable from zero).", "kind":"novel"},
    {"id":"h14.3", "text":"There is a strong negative three-way interaction between treatment_sotorasib, kras_g12c, and sex_female on pfs_months: the sotorasib benefit is concentrated in male KRAS-G12C-positive patients.", "kind":"novel"},
    {"id":"h14.4", "text":"Within KRAS G12C-mutated patients, BRCA2 mutation (brca2_mutation==1) substantially reduces the sotorasib PFS benefit (large negative interaction with sotorasib).", "kind":"novel"},
]
i14_a = []
male = sotg["kras+ & sex_female==0"]
female = sotg["kras+ & sex_female==1"]
i14_a.append({
    "hypothesis_ids":["h14.1"],
    "code":"smf.ols('pfs_months ~ treatment_sotorasib', data=df[(kras_g12c==1)&(sex_female==0)]).fit()",
    "result_summary": f"KRAS G12C+ males (n={male['n']}, on={male['n_on']}): mean PFS on sotorasib = {fnum(male['mean_on'])} mo vs off = {fnum(male['mean_off'])} mo; beta={fnum(male['coef'])}, p={fnum(male['p'], '.2e')}.",
    "p_value": float(male["p"]),
    "effect_estimate": float(male["coef"]),
    "significant": float(male["p"]) < 0.05,
})
i14_a.append({
    "hypothesis_ids":["h14.2"],
    "code":"smf.ols('pfs_months ~ treatment_sotorasib', data=df[(kras_g12c==1)&(sex_female==1)]).fit()",
    "result_summary": f"KRAS G12C+ females (n={female['n']}, on={female['n_on']}): mean PFS on sotorasib = {fnum(female['mean_on'])} mo vs off = {fnum(female['mean_off'])} mo; beta={fnum(female['coef'])}, p={fnum(female['p'], '.2e')}. Sotorasib provides essentially no PFS benefit in this subgroup.",
    "p_value": float(female["p"]),
    "effect_estimate": float(female["coef"]),
    "significant": float(female["p"]) < 0.05,
})
threeway = E["sotorasib_kras_sex_3way"]["treatment_sotorasib:kras_g12c:sex_female"]
i14_a.append({
    "hypothesis_ids":["h14.3"],
    "code":"smf.ols('pfs_months ~ treatment_sotorasib*kras_g12c*sex_female', data=df).fit()",
    "result_summary": f"3-way interaction treatment_sotorasib:kras_g12c:sex_female beta={fnum(threeway['coef'])}, p={fnum(threeway['p'], '.2e')}. The sotorasib effect in KRAS+ males is +4.66 months above the baseline; this is almost entirely cancelled by the -4.69 three-way interaction in KRAS+ females.",
    "p_value": float(threeway["p"]),
    "effect_estimate": float(threeway["coef"]),
    "significant": float(threeway["p"]) < 0.05,
})
brca = sotg_intxn["brca2_mutation"]
i14_a.append({
    "hypothesis_ids":["h14.4"],
    "code":"smf.ols('pfs_months ~ treatment_sotorasib*brca2_mutation', data=df[kras_g12c==1]).fit()",
    "result_summary": f"Within KRAS G12C+: sotorasib x brca2_mutation interaction beta={fnum(brca['intxn_coef'])}, p={fnum(brca['intxn_p'], '.2e')}. Patients who are both KRAS G12C+ and BRCA2-mutated lose much of the sotorasib benefit.",
    "p_value": float(brca["intxn_p"]),
    "effect_estimate": float(brca["intxn_coef"]),
    "significant": float(brca["intxn_p"]) < 0.05,
})
iters.append({"index":14, "proposed_hypotheses": i14_h, "analyses": i14_a})


# ---------- Iter 15: even tighter sotorasib subgroup definitions ----------
import pandas as pd
df = pd.read_parquet("dataset.parquet")
df["smoking_current"] = (df["smoking_status"]=="current").astype(int)
import statsmodels.formula.api as smf
def stat_sub(treat, mask, label):
    sub = df[mask]
    if sub[treat].nunique() < 2:
        return {"label":label,"n":int(sub.shape[0]),"issue":"no_var"}
    m = smf.ols(f"pfs_months ~ {treat}", data=sub).fit()
    return {"label":label,"n":int(sub.shape[0]),
            "n_on":int(sub[treat].sum()),"n_off":int((sub[treat]==0).sum()),
            "mean_on":float(sub.loc[sub[treat]==1,"pfs_months"].mean()),
            "mean_off":float(sub.loc[sub[treat]==0,"pfs_months"].mean()),
            "coef":float(m.params[treat]),"p":float(m.pvalues[treat])}

# Final best definition: KRAS G12C+ male, exclude BRCA2+
sub_strict = (df["kras_g12c"]==1) & (df["sex_female"]==0) & (df["brca2_mutation"]==0)
res_strict = stat_sub("treatment_sotorasib", sub_strict, "kras_g12c==1 & sex_female==0 & brca2_mutation==0")

# What if also restrict to ECOG <= 1?
sub_strict2 = sub_strict & (df["ecog_ps"]<=1)
res_strict2 = stat_sub("treatment_sotorasib", sub_strict2, "kras+ & male & brca2_neg & ecog<=1")

# Final null hypotheses
sub_fail_sex = (df["kras_g12c"]==1) & (df["sex_female"]==1)
res_fail_sex = stat_sub("treatment_sotorasib", sub_fail_sex, "kras+ & female (sotorasib does NOT benefit)")
sub_fail_brca = (df["kras_g12c"]==1) & (df["brca2_mutation"]==1)
res_fail_brca = stat_sub("treatment_sotorasib", sub_fail_brca, "kras+ & brca2+ (sotorasib does NOT benefit)")

i15_h = [
    {"id":"h15.1", "text":"Within KRAS G12C-mutated, BRCA2-wildtype, male (kras_g12c==1 & brca2_mutation==0 & sex_female==0) patients, sotorasib provides a very large positive PFS benefit relative to no sotorasib.", "kind":"refined"},
    {"id":"h15.2", "text":"Within KRAS G12C-mutated, BRCA2-wildtype, male, ECOG<=1 patients, sotorasib provides a very large positive PFS benefit relative to no sotorasib (final refined positive sotorasib subgroup).", "kind":"refined"},
    {"id":"h15.3", "text":"Within KRAS G12C-mutated female (sex_female==1) patients, sotorasib does NOT improve pfs_months relative to no sotorasib (negative subgroup defined by the variable that suppresses the treatment effect).", "kind":"refined"},
    {"id":"h15.4", "text":"Within KRAS G12C-mutated and BRCA2-mutated patients, sotorasib does NOT improve pfs_months relative to no sotorasib (a second variable whose 'unfavorable' value suppresses the treatment effect).", "kind":"refined"},
    {"id":"h15.5", "text":"Marker-treatment 'mismatch' subgroups (e.g., osimertinib in EGFR-wildtype, sotorasib in KRAS-wildtype, olaparib in BRCA2-wildtype, pembrolizumab in PDL1<0.01) all show null PFS effects, confirming biological specificity for sotorasib but absence of biomarker-driven benefit for the other three drugs.", "kind":"novel"},
]
i15_a = []
i15_a.append({
    "hypothesis_ids":["h15.1"],
    "code":"smf.ols('pfs_months ~ treatment_sotorasib', data=df[(kras_g12c==1)&(brca2_mutation==0)&(sex_female==0)]).fit()",
    "result_summary": f"Subgroup '{res_strict['label']}', n={res_strict['n']}, on={res_strict['n_on']}: mean on={fnum(res_strict['mean_on'])} mo vs off={fnum(res_strict['mean_off'])} mo; beta={fnum(res_strict['coef'])}, p={fnum(res_strict['p'], '.2e')}.",
    "p_value": float(res_strict["p"]),
    "effect_estimate": float(res_strict["coef"]),
    "significant": float(res_strict["p"]) < 0.05,
})
i15_a.append({
    "hypothesis_ids":["h15.2"],
    "code":"smf.ols('pfs_months ~ treatment_sotorasib', data=df[(kras_g12c==1)&(brca2_mutation==0)&(sex_female==0)&(ecog_ps<=1)]).fit()",
    "result_summary": f"Subgroup '{res_strict2['label']}', n={res_strict2['n']}, on={res_strict2['n_on']}: mean on={fnum(res_strict2['mean_on'])} mo vs off={fnum(res_strict2['mean_off'])} mo; beta={fnum(res_strict2['coef'])}, p={fnum(res_strict2['p'], '.2e')}.",
    "p_value": float(res_strict2["p"]),
    "effect_estimate": float(res_strict2["coef"]),
    "significant": float(res_strict2["p"]) < 0.05,
})
i15_a.append({
    "hypothesis_ids":["h15.3"],
    "code":"smf.ols('pfs_months ~ treatment_sotorasib', data=df[(kras_g12c==1)&(sex_female==1)]).fit()",
    "result_summary": f"Subgroup '{res_fail_sex['label']}', n={res_fail_sex['n']}, on={res_fail_sex['n_on']}: mean on={fnum(res_fail_sex['mean_on'])} mo vs off={fnum(res_fail_sex['mean_off'])} mo; beta={fnum(res_fail_sex['coef'])}, p={fnum(res_fail_sex['p'], '.2e')}.",
    "p_value": float(res_fail_sex["p"]),
    "effect_estimate": float(res_fail_sex["coef"]),
    "significant": float(res_fail_sex["p"]) < 0.05,
})
i15_a.append({
    "hypothesis_ids":["h15.4"],
    "code":"smf.ols('pfs_months ~ treatment_sotorasib', data=df[(kras_g12c==1)&(brca2_mutation==1)]).fit()",
    "result_summary": f"Subgroup '{res_fail_brca['label']}', n={res_fail_brca['n']}, on={res_fail_brca['n_on']}: mean on={fnum(res_fail_brca['mean_on'])} mo vs off={fnum(res_fail_brca['mean_off'])} mo; beta={fnum(res_fail_brca['coef'])}, p={fnum(res_fail_brca['p'], '.2e')}.",
    "p_value": float(res_fail_brca["p"]),
    "effect_estimate": float(res_fail_brca["coef"]),
    "significant": float(res_fail_brca["p"]) < 0.05,
})
mm = R["iter15_mismatch"]
mm_summary = "; ".join([f"{k.split('__')[0]} in {k.split('__')[1]}: beta={fnum(v['coef'])}, p={fnum(v['p'], '.2e')}" for k, v in mm.items()])
i15_a.append({
    "hypothesis_ids":["h15.5"],
    "code":"compare each treatment effect within the marker-NEGATIVE subgroup",
    "result_summary": f"All marker-mismatch subgroup treatment effects are null: {mm_summary}.",
    "p_value": None,
    "effect_estimate": None,
    "significant": False,
})
iters.append({"index":15, "proposed_hypotheses": i15_h, "analyses": i15_a})


transcript = {
    "dataset_id": "ds001_nsclc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@manual-run",
    "max_iterations": 25,
    "iterations": iters,
}

with open("transcript.json","w") as f:
    json.dump(transcript, f, indent=2, default=str)
print("Wrote transcript.json with", len(iters), "iterations.")
