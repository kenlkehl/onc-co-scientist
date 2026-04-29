"""Fresh analysis run for ds001_crc — 25 iterations of hypothesis-driven analysis."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(0)
df = pd.read_parquet("dataset.parquet")
N = len(df)

# Helper: signed effect for binary X on continuous Y via mean difference + Welch t
def mean_diff(y_a, y_b):
    m_a = float(np.mean(y_a)); m_b = float(np.mean(y_b))
    t = stats.ttest_ind(y_a, y_b, equal_var=False)
    return m_a - m_b, float(t.pvalue), m_a, m_b, len(y_a), len(y_b)

def by_binary(df_, col, y="pfs_months"):
    a = df_.loc[df_[col] == 1, y].values
    b = df_.loc[df_[col] == 0, y].values
    return mean_diff(a, b)

def linreg(df_, x, y="pfs_months"):
    X = sm.add_constant(df_[x].astype(float).values)
    m = sm.OLS(df_[y].values, X).fit()
    return float(m.params[1]), float(m.pvalues[1])

iterations = []  # list of {index, proposed_hypotheses, analyses}

# ---------- ITERATION 1 ----------
hyps = []
ans = []
hyps.append({"id": "h1.1", "text": "Higher ECOG performance status (ecog_ps) is associated with shorter pfs_months (negative slope).", "kind": "novel"})
hyps.append({"id": "h1.2", "text": "Stage IV disease (stage_iv=1) is associated with shorter pfs_months than non-stage IV.", "kind": "novel"})
hyps.append({"id": "h1.3", "text": "Older age (age_years) is associated with shorter pfs_months (negative slope).", "kind": "novel"})

eff, pv = linreg(df, "ecog_ps")
ans.append({"hypothesis_ids": ["h1.1"], "code": "OLS pfs_months ~ ecog_ps",
            "result_summary": f"Each 1-point increase in ecog_ps changes mean pfs_months by {eff:.3f} months (p={pv:.2e}).",
            "p_value": pv, "effect_estimate": eff, "significant": pv < 0.05})
eff, pv, ma, mb, na, nb = by_binary(df, "stage_iv")
ans.append({"hypothesis_ids": ["h1.2"], "code": "Welch t-test pfs by stage_iv",
            "result_summary": f"Mean pfs_months: stage_iv=1 {ma:.2f} (n={na}) vs stage_iv=0 {mb:.2f} (n={nb}); diff={eff:.3f}, p={pv:.2e}.",
            "p_value": pv, "effect_estimate": eff, "significant": pv < 0.05})
eff, pv = linreg(df, "age_years")
ans.append({"hypothesis_ids": ["h1.3"], "code": "OLS pfs_months ~ age_years",
            "result_summary": f"Each 1-year increase in age changes mean pfs_months by {eff:.4f} months (p={pv:.2e}).",
            "p_value": pv, "effect_estimate": eff, "significant": pv < 0.05})
iterations.append({"index": 1, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 2 ----------
hyps = []; ans = []
hyps.append({"id": "h2.1", "text": "Lower albumin_g_dl is associated with shorter pfs_months (positive slope: higher albumin → longer PFS).", "kind": "novel"})
hyps.append({"id": "h2.2", "text": "Higher weight_loss_pct_6mo is associated with shorter pfs_months (negative slope).", "kind": "novel"})
hyps.append({"id": "h2.3", "text": "Higher LDH (ldh_u_l) is associated with shorter pfs_months (negative slope).", "kind": "novel"})
hyps.append({"id": "h2.4", "text": "Higher CRP (crp_mg_l) is associated with shorter pfs_months (negative slope).", "kind": "novel"})
hyps.append({"id": "h2.5", "text": "Higher neutrophil-to-lymphocyte ratio (nlr) is associated with shorter pfs_months (negative slope).", "kind": "novel"})
hyps.append({"id": "h2.6", "text": "Higher baseline CEA (cea_ng_ml) is associated with shorter pfs_months (negative slope).", "kind": "novel"})
for hid, col in [("h2.1","albumin_g_dl"),("h2.2","weight_loss_pct_6mo"),("h2.3","ldh_u_l"),
                  ("h2.4","crp_mg_l"),("h2.5","nlr"),("h2.6","cea_ng_ml")]:
    eff, pv = linreg(df, col)
    ans.append({"hypothesis_ids": [hid], "code": f"OLS pfs_months ~ {col}",
                "result_summary": f"Slope of pfs_months on {col}: {eff:.5f} months/unit (p={pv:.2e}).",
                "p_value": pv, "effect_estimate": eff, "significant": pv < 0.05})
iterations.append({"index": 2, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 3 ----------
hyps = []; ans = []
hyps.append({"id": "h3.1", "text": "Cetuximab (treatment_cetuximab) is associated with longer pfs_months overall (positive mean difference).", "kind": "novel"})
hyps.append({"id": "h3.2", "text": "Bevacizumab (treatment_bevacizumab) is associated with longer pfs_months overall (positive mean difference).", "kind": "novel"})
hyps.append({"id": "h3.3", "text": "Pembrolizumab (treatment_pembrolizumab) is associated with longer pfs_months overall.", "kind": "novel"})
hyps.append({"id": "h3.4", "text": "Encorafenib (treatment_encorafenib) is associated with longer pfs_months overall.", "kind": "novel"})
hyps.append({"id": "h3.5", "text": "Trastuzumab+tucatinib (treatment_trastuzumab_tucatinib) is associated with longer pfs_months overall.", "kind": "novel"})
hyps.append({"id": "h3.6", "text": "Regorafenib (treatment_regorafenib) is associated with shorter pfs_months overall (later-line indication).", "kind": "novel"})
for hid, col in [("h3.1","treatment_cetuximab"),("h3.2","treatment_bevacizumab"),
                  ("h3.3","treatment_pembrolizumab"),("h3.4","treatment_encorafenib"),
                  ("h3.5","treatment_trastuzumab_tucatinib"),("h3.6","treatment_regorafenib")]:
    eff, pv, ma, mb, na, nb = by_binary(df, col)
    ans.append({"hypothesis_ids":[hid], "code": f"Welch t-test pfs by {col}",
                "result_summary": f"Mean pfs_months on {col} {ma:.2f} (n={na}) vs off {mb:.2f} (n={nb}); diff={eff:.3f}, p={pv:.2e}.",
                "p_value": pv, "effect_estimate": eff, "significant": pv < 0.05})
iterations.append({"index": 3, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 4 — refining with biomarker subgroups ----------
hyps = []; ans = []
hyps.append({"id":"h4.1","text":"Cetuximab benefit (longer pfs_months) is restricted to or stronger in patients without KRAS, NRAS, or BRAF V600E mutations (RAS/RAF wild-type) than in patients with any of these mutations (positive interaction in WT subgroup).","kind":"refined"})
hyps.append({"id":"h4.2","text":"Among RAS/RAF wild-type patients, cetuximab benefit (longer pfs_months) is greater in left-sided (right_sided_primary=0) than right-sided primaries.","kind":"refined"})
hyps.append({"id":"h4.3","text":"In the KRAS-mutant subgroup, cetuximab is associated with no benefit or harm on pfs_months (mean difference ≤0).","kind":"refined"})

df["rasraf_wt"] = ((df["kras_mutation"]==0) & (df["nras_mutation"]==0) & (df["braf_v600e"]==0)).astype(int)

# h4.1: cetuximab effect in WT vs MUT
sub_wt = df[df["rasraf_wt"]==1]
sub_mut = df[df["rasraf_wt"]==0]
eff_wt, pv_wt, ma_wt, mb_wt, na_wt, nb_wt = by_binary(sub_wt, "treatment_cetuximab")
eff_mut, pv_mut, ma_mut, mb_mut, na_mut, nb_mut = by_binary(sub_mut, "treatment_cetuximab")
# Interaction: regression with cetux*rasraf_wt
m = smf.ols("pfs_months ~ treatment_cetuximab * rasraf_wt", data=df).fit()
inter_eff = float(m.params["treatment_cetuximab:rasraf_wt"])
inter_pv = float(m.pvalues["treatment_cetuximab:rasraf_wt"])
ans.append({"hypothesis_ids":["h4.1"], "code":"OLS pfs_months ~ treatment_cetuximab * rasraf_wt",
            "result_summary": (f"Cetuximab effect in WT: {eff_wt:+.3f} mo (p={pv_wt:.2e}); in MUT: {eff_mut:+.3f} mo (p={pv_mut:.2e}). "
                               f"Interaction term cetux*WT = {inter_eff:+.3f} mo (p={inter_pv:.2e})."),
            "p_value": inter_pv, "effect_estimate": inter_eff, "significant": inter_pv < 0.05})

# h4.2: among WT, cetux*right_sided
m2 = smf.ols("pfs_months ~ treatment_cetuximab * right_sided_primary", data=sub_wt).fit()
inter2_eff = float(m2.params["treatment_cetuximab:right_sided_primary"])
inter2_pv = float(m2.pvalues["treatment_cetuximab:right_sided_primary"])
# stratified means
wt_left = sub_wt[sub_wt["right_sided_primary"]==0]
wt_right = sub_wt[sub_wt["right_sided_primary"]==1]
e_left, p_left, _, _, _, _ = by_binary(wt_left, "treatment_cetuximab")
e_right, p_right, _, _, _, _ = by_binary(wt_right, "treatment_cetuximab")
ans.append({"hypothesis_ids":["h4.2"], "code":"OLS pfs_months ~ treatment_cetuximab * right_sided_primary (WT subset)",
            "result_summary": (f"In RAS/RAF WT, cetux effect on pfs_months: left-sided {e_left:+.3f} mo (p={p_left:.2e}); "
                               f"right-sided {e_right:+.3f} mo (p={p_right:.2e}). "
                               f"Cetux*right_sided interaction = {inter2_eff:+.3f} mo (p={inter2_pv:.2e})."),
            "p_value": inter2_pv, "effect_estimate": inter2_eff, "significant": inter2_pv < 0.05})

# h4.3: in KRAS-mutant patients
kras_sub = df[df["kras_mutation"]==1]
eff_k, pv_k, ma_k, mb_k, na_k, nb_k = by_binary(kras_sub, "treatment_cetuximab")
ans.append({"hypothesis_ids":["h4.3"], "code":"Welch t-test pfs by treatment_cetuximab in KRAS-mutant subset",
            "result_summary": (f"In KRAS-mutant patients, cetuximab effect on pfs_months = {eff_k:+.3f} mo "
                               f"({ma_k:.2f} vs {mb_k:.2f}, n={na_k}/{nb_k}, p={pv_k:.2e})."),
            "p_value": pv_k, "effect_estimate": eff_k, "significant": pv_k < 0.05})
iterations.append({"index": 4, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 5 — pembrolizumab × MSI ----------
hyps = []; ans = []
hyps.append({"id":"h5.1","text":"Pembrolizumab benefit on pfs_months is restricted to MSI-high (msi_high=1) patients (positive interaction).","kind":"refined"})
hyps.append({"id":"h5.2","text":"In MSI-high patients, pembrolizumab is associated with longer pfs_months than no pembrolizumab (positive mean difference).","kind":"refined"})
hyps.append({"id":"h5.3","text":"In microsatellite-stable (msi_high=0) patients, pembrolizumab is not associated with longer pfs_months (effect ~0).","kind":"refined"})

m = smf.ols("pfs_months ~ treatment_pembrolizumab * msi_high", data=df).fit()
inter_eff = float(m.params["treatment_pembrolizumab:msi_high"])
inter_pv = float(m.pvalues["treatment_pembrolizumab:msi_high"])
sub_msi = df[df["msi_high"]==1]
sub_mss = df[df["msi_high"]==0]
e_msi, p_msi, ma_msi, mb_msi, na_msi, nb_msi = by_binary(sub_msi, "treatment_pembrolizumab")
e_mss, p_mss, _, _, _, _ = by_binary(sub_mss, "treatment_pembrolizumab")
ans.append({"hypothesis_ids":["h5.1"], "code":"OLS pfs_months ~ treatment_pembrolizumab * msi_high",
            "result_summary": f"Pembro*MSI interaction = {inter_eff:+.3f} mo (p={inter_pv:.2e}); pembro effect MSI-high {e_msi:+.3f} vs MSS {e_mss:+.3f}.",
            "p_value": inter_pv, "effect_estimate": inter_eff, "significant": inter_pv < 0.05})
ans.append({"hypothesis_ids":["h5.2"], "code":"Welch t-test pfs by pembro in MSI-high subset",
            "result_summary": f"MSI-high pembro effect: {e_msi:+.3f} mo ({ma_msi:.2f} vs {mb_msi:.2f}, n={na_msi}/{nb_msi}, p={p_msi:.2e}).",
            "p_value": p_msi, "effect_estimate": e_msi, "significant": p_msi < 0.05})
ans.append({"hypothesis_ids":["h5.3"], "code":"Welch t-test pfs by pembro in MSS subset",
            "result_summary": f"MSS pembro effect: {e_mss:+.3f} mo (p={p_mss:.2e}).",
            "p_value": p_mss, "effect_estimate": e_mss, "significant": p_mss < 0.05})
iterations.append({"index": 5, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 6 — encorafenib × BRAF V600E ----------
hyps = []; ans = []
hyps.append({"id":"h6.1","text":"Encorafenib benefit on pfs_months is restricted to BRAF V600E-mutant patients (positive interaction with braf_v600e).","kind":"refined"})
hyps.append({"id":"h6.2","text":"In BRAF V600E-mutant patients, encorafenib is associated with longer pfs_months than no encorafenib (positive mean difference).","kind":"refined"})
hyps.append({"id":"h6.3","text":"In BRAF V600E-wild-type patients, encorafenib is not associated with longer pfs_months (effect ~0).","kind":"refined"})

m = smf.ols("pfs_months ~ treatment_encorafenib * braf_v600e", data=df).fit()
inter_eff = float(m.params["treatment_encorafenib:braf_v600e"])
inter_pv = float(m.pvalues["treatment_encorafenib:braf_v600e"])
sub_braf = df[df["braf_v600e"]==1]; sub_braf_wt = df[df["braf_v600e"]==0]
e_braf, p_braf, ma_braf, mb_braf, na_braf, nb_braf = by_binary(sub_braf, "treatment_encorafenib")
e_brafwt, p_brafwt, _, _, _, _ = by_binary(sub_braf_wt, "treatment_encorafenib")
ans.append({"hypothesis_ids":["h6.1"], "code":"OLS pfs_months ~ treatment_encorafenib * braf_v600e",
            "result_summary": f"Encorafenib*BRAFV600E interaction = {inter_eff:+.3f} mo (p={inter_pv:.2e}); BRAF-mut effect {e_braf:+.3f} vs WT {e_brafwt:+.3f}.",
            "p_value": inter_pv, "effect_estimate": inter_eff, "significant": inter_pv < 0.05})
ans.append({"hypothesis_ids":["h6.2"], "code":"Welch t-test pfs by encorafenib in BRAF V600E subset",
            "result_summary": f"BRAF V600E-mut encorafenib effect: {e_braf:+.3f} mo ({ma_braf:.2f} vs {mb_braf:.2f}, n={na_braf}/{nb_braf}, p={p_braf:.2e}).",
            "p_value": p_braf, "effect_estimate": e_braf, "significant": p_braf < 0.05})
ans.append({"hypothesis_ids":["h6.3"], "code":"Welch t-test pfs by encorafenib in BRAF V600E WT subset",
            "result_summary": f"BRAF V600E WT encorafenib effect: {e_brafwt:+.3f} mo (p={p_brafwt:.2e}).",
            "p_value": p_brafwt, "effect_estimate": e_brafwt, "significant": p_brafwt < 0.05})
iterations.append({"index": 6, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 7 — trastuzumab+tucatinib × HER2 ----------
hyps = []; ans = []
hyps.append({"id":"h7.1","text":"Trastuzumab+tucatinib benefit on pfs_months is restricted to HER2-amplified (her2_amplified=1) patients (positive interaction).","kind":"refined"})
hyps.append({"id":"h7.2","text":"In HER2-amplified patients, trastuzumab+tucatinib is associated with longer pfs_months (positive mean difference).","kind":"refined"})
hyps.append({"id":"h7.3","text":"In HER2-non-amplified patients, trastuzumab+tucatinib is not associated with longer pfs_months (effect ~0).","kind":"refined"})

m = smf.ols("pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified", data=df).fit()
inter_eff = float(m.params["treatment_trastuzumab_tucatinib:her2_amplified"])
inter_pv = float(m.pvalues["treatment_trastuzumab_tucatinib:her2_amplified"])
sub_her2 = df[df["her2_amplified"]==1]; sub_her2wt = df[df["her2_amplified"]==0]
e_h, p_h, ma_h, mb_h, na_h, nb_h = by_binary(sub_her2, "treatment_trastuzumab_tucatinib")
e_hwt, p_hwt, _, _, _, _ = by_binary(sub_her2wt, "treatment_trastuzumab_tucatinib")
ans.append({"hypothesis_ids":["h7.1"], "code":"OLS pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified",
            "result_summary": f"Trastuzumab/tucatinib*HER2 interaction = {inter_eff:+.3f} mo (p={inter_pv:.2e}); HER2-amp effect {e_h:+.3f} vs non-amp {e_hwt:+.3f}.",
            "p_value": inter_pv, "effect_estimate": inter_eff, "significant": inter_pv < 0.05})
ans.append({"hypothesis_ids":["h7.2"], "code":"Welch t-test pfs by trastuzumab_tucatinib in HER2-amp subset",
            "result_summary": f"HER2-amp trastuzumab/tucatinib effect: {e_h:+.3f} mo ({ma_h:.2f} vs {mb_h:.2f}, n={na_h}/{nb_h}, p={p_h:.2e}).",
            "p_value": p_h, "effect_estimate": e_h, "significant": p_h < 0.05})
ans.append({"hypothesis_ids":["h7.3"], "code":"Welch t-test pfs by trastuzumab_tucatinib in HER2 non-amp subset",
            "result_summary": f"HER2-non-amp trastuzumab/tucatinib effect: {e_hwt:+.3f} mo (p={p_hwt:.2e}).",
            "p_value": p_hwt, "effect_estimate": e_hwt, "significant": p_hwt < 0.05})
iterations.append({"index": 7, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 8 — sidedness, sex, race ----------
hyps = []; ans = []
hyps.append({"id":"h8.1","text":"Right-sided primary (right_sided_primary=1) tumors are associated with shorter pfs_months than left-sided (negative mean difference).","kind":"novel"})
hyps.append({"id":"h8.2","text":"Female sex (sex_female=1) is associated with longer pfs_months than male (positive mean difference).","kind":"novel"})
hyps.append({"id":"h8.3","text":"Liver metastases (liver_mets=1) are associated with shorter pfs_months than no liver mets (negative mean difference).","kind":"novel"})

for hid, col in [("h8.1","right_sided_primary"),("h8.2","sex_female"),("h8.3","liver_mets")]:
    eff, pv, ma, mb, na, nb = by_binary(df, col)
    ans.append({"hypothesis_ids":[hid], "code":f"Welch t-test pfs by {col}",
                "result_summary": f"Mean pfs: {col}=1 {ma:.2f} (n={na}) vs =0 {mb:.2f} (n={nb}); diff={eff:.3f}, p={pv:.2e}.",
                "p_value": pv, "effect_estimate": eff, "significant": pv < 0.05})
iterations.append({"index": 8, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 9 — multivariable: prognostic model ----------
hyps = []; ans = []
hyps.append({"id":"h9.1","text":"In a multivariable OLS for pfs_months, ECOG performance status retains a negative coefficient after adjusting for stage_iv, age_years, albumin_g_dl, ldh_u_l, weight_loss_pct_6mo, and crp_mg_l.","kind":"refined"})
hyps.append({"id":"h9.2","text":"In the same multivariable OLS, albumin_g_dl retains a positive coefficient (higher albumin → longer PFS) after adjustment.","kind":"refined"})
hyps.append({"id":"h9.3","text":"In the same multivariable OLS, weight_loss_pct_6mo retains a negative coefficient after adjustment.","kind":"refined"})

m = smf.ols("pfs_months ~ ecog_ps + stage_iv + age_years + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr", data=df).fit()
for hid, col in [("h9.1","ecog_ps"),("h9.2","albumin_g_dl"),("h9.3","weight_loss_pct_6mo")]:
    e = float(m.params[col]); p = float(m.pvalues[col])
    ans.append({"hypothesis_ids":[hid], "code":"OLS pfs_months ~ ecog_ps + stage_iv + age_years + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr",
                "result_summary": f"Adjusted coefficient for {col}: {e:+.4f} mo/unit (p={p:.2e}).",
                "p_value": p, "effect_estimate": e, "significant": p<0.05})
iterations.append({"index": 9, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 10 — symptoms ----------
hyps = []; ans = []
hyps.append({"id":"h10.1","text":"Higher fatigue_grade is associated with shorter pfs_months (negative slope).","kind":"novel"})
hyps.append({"id":"h10.2","text":"Higher pain_nrs is associated with shorter pfs_months (negative slope).","kind":"novel"})
hyps.append({"id":"h10.3","text":"Higher dyspnea_grade is associated with shorter pfs_months (negative slope).","kind":"novel"})
hyps.append({"id":"h10.4","text":"Higher appetite_loss_grade is associated with shorter pfs_months (negative slope).","kind":"novel"})
for hid, col in [("h10.1","fatigue_grade"),("h10.2","pain_nrs"),("h10.3","dyspnea_grade"),("h10.4","appetite_loss_grade")]:
    eff, pv = linreg(df, col)
    ans.append({"hypothesis_ids":[hid], "code":f"OLS pfs_months ~ {col}",
                "result_summary": f"Slope of pfs on {col}: {eff:+.4f} mo/grade (p={pv:.2e}).",
                "p_value": pv, "effect_estimate": eff, "significant": pv<0.05})
iterations.append({"index": 10, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 11 — comorbidities ----------
hyps = []; ans = []
hyps.append({"id":"h11.1","text":"Heart failure (heart_failure=1) is associated with shorter pfs_months (negative mean difference).","kind":"novel"})
hyps.append({"id":"h11.2","text":"Chronic kidney disease (chronic_kidney_disease=1) is associated with shorter pfs_months.","kind":"novel"})
hyps.append({"id":"h11.3","text":"COPD (copd=1) is associated with shorter pfs_months.","kind":"novel"})
hyps.append({"id":"h11.4","text":"Diabetes mellitus (diabetes_mellitus=1) is associated with shorter pfs_months.","kind":"novel"})
for hid, col in [("h11.1","heart_failure"),("h11.2","chronic_kidney_disease"),("h11.3","copd"),("h11.4","diabetes_mellitus")]:
    eff, pv, ma, mb, na, nb = by_binary(df, col)
    ans.append({"hypothesis_ids":[hid], "code":f"Welch t-test pfs by {col}",
                "result_summary": f"Mean pfs: {col}=1 {ma:.2f} vs =0 {mb:.2f}; diff={eff:+.3f}, p={pv:.2e}.",
                "p_value": pv, "effect_estimate": eff, "significant": pv<0.05})
iterations.append({"index": 11, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 12 — prior therapy ----------
hyps = []; ans = []
hyps.append({"id":"h12.1","text":"Greater prior_lines_of_therapy is associated with shorter pfs_months (negative slope).","kind":"novel"})
hyps.append({"id":"h12.2","text":"Prior chemotherapy (prior_chemotherapy=1) is associated with shorter pfs_months.","kind":"novel"})
hyps.append({"id":"h12.3","text":"Prior surgery (prior_surgery=1) is associated with longer pfs_months (often resectable disease has better prognosis).","kind":"novel"})
eff, pv = linreg(df, "prior_lines_of_therapy")
ans.append({"hypothesis_ids":["h12.1"], "code":"OLS pfs_months ~ prior_lines_of_therapy",
            "result_summary": f"Slope of pfs on prior_lines_of_therapy: {eff:+.4f} mo/line (p={pv:.2e}).",
            "p_value": pv, "effect_estimate": eff, "significant": pv<0.05})
for hid, col in [("h12.2","prior_chemotherapy"),("h12.3","prior_surgery")]:
    eff, pv, ma, mb, na, nb = by_binary(df, col)
    ans.append({"hypothesis_ids":[hid], "code":f"Welch t-test pfs by {col}",
                "result_summary": f"Mean pfs: {col}=1 {ma:.2f} vs =0 {mb:.2f}; diff={eff:+.3f}, p={pv:.2e}.",
                "p_value": pv, "effect_estimate": eff, "significant": pv<0.05})
iterations.append({"index": 12, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 13 — labs (CBC, chem) ----------
hyps = []; ans = []
hyps.append({"id":"h13.1","text":"Lower hemoglobin_g_dl is associated with shorter pfs_months (positive slope: higher Hb → longer PFS).","kind":"novel"})
hyps.append({"id":"h13.2","text":"Higher alkaline_phosphatase_u_l is associated with shorter pfs_months (negative slope; reflects liver/bone burden).","kind":"novel"})
hyps.append({"id":"h13.3","text":"Higher total_bilirubin_mg_dl is associated with shorter pfs_months (negative slope).","kind":"novel"})
hyps.append({"id":"h13.4","text":"Higher creatinine_mg_dl is associated with shorter pfs_months (negative slope).","kind":"novel"})
for hid, col in [("h13.1","hemoglobin_g_dl"),("h13.2","alkaline_phosphatase_u_l"),
                  ("h13.3","total_bilirubin_mg_dl"),("h13.4","creatinine_mg_dl")]:
    eff, pv = linreg(df, col)
    ans.append({"hypothesis_ids":[hid], "code":f"OLS pfs_months ~ {col}",
                "result_summary": f"Slope of pfs on {col}: {eff:+.5f} (p={pv:.2e}).",
                "p_value": pv, "effect_estimate": eff, "significant": pv<0.05})
iterations.append({"index": 13, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 14 — race/ethnicity, insurance, rural ----------
hyps = []; ans = []
hyps.append({"id":"h14.1","text":"Mean pfs_months differs across race_ethnicity categories (overall ANOVA test).","kind":"novel"})
hyps.append({"id":"h14.2","text":"Mean pfs_months differs across insurance_type categories (overall ANOVA test).","kind":"novel"})
hyps.append({"id":"h14.3","text":"Rural residence (rural_residence=1) is associated with shorter pfs_months than non-rural.","kind":"novel"})

groups = [df.loc[df["race_ethnicity"]==r, "pfs_months"].values for r in df["race_ethnicity"].unique()]
f1 = stats.f_oneway(*groups)
means_by_race = df.groupby("race_ethnicity")["pfs_months"].mean().to_dict()
# pick effect: max - min mean
max_r = max(means_by_race.values()); min_r = min(means_by_race.values())
ans.append({"hypothesis_ids":["h14.1"], "code":"f_oneway pfs by race_ethnicity",
            "result_summary": f"ANOVA F={f1.statistic:.2f} (p={f1.pvalue:.2e}); group means: {means_by_race}.",
            "p_value": float(f1.pvalue), "effect_estimate": float(max_r - min_r), "significant": bool(f1.pvalue<0.05)})
groups2 = [df.loc[df["insurance_type"]==r, "pfs_months"].values for r in df["insurance_type"].unique()]
f2 = stats.f_oneway(*groups2)
means_by_ins = df.groupby("insurance_type")["pfs_months"].mean().to_dict()
ans.append({"hypothesis_ids":["h14.2"], "code":"f_oneway pfs by insurance_type",
            "result_summary": f"ANOVA F={f2.statistic:.2f} (p={f2.pvalue:.2e}); group means: {means_by_ins}.",
            "p_value": float(f2.pvalue), "effect_estimate": float(max(means_by_ins.values())-min(means_by_ins.values())),
            "significant": bool(f2.pvalue<0.05)})
eff, pv, ma, mb, na, nb = by_binary(df, "rural_residence")
ans.append({"hypothesis_ids":["h14.3"], "code":"Welch t-test pfs by rural_residence",
            "result_summary": f"Mean pfs: rural=1 {ma:.2f} vs =0 {mb:.2f}; diff={eff:+.3f}, p={pv:.2e}.",
            "p_value": pv, "effect_estimate": eff, "significant": pv<0.05})
iterations.append({"index": 14, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 15 — sites of metastasis ----------
hyps = []; ans = []
hyps.append({"id":"h15.1","text":"Bone metastases (bone_mets=1) are associated with shorter pfs_months than no bone mets.","kind":"novel"})
hyps.append({"id":"h15.2","text":"Adrenal metastases (adrenal_mets=1) are associated with shorter pfs_months.","kind":"novel"})
hyps.append({"id":"h15.3","text":"Pleural effusion (pleural_effusion=1) is associated with shorter pfs_months.","kind":"novel"})
for hid, col in [("h15.1","bone_mets"),("h15.2","adrenal_mets"),("h15.3","pleural_effusion")]:
    eff, pv, ma, mb, na, nb = by_binary(df, col)
    ans.append({"hypothesis_ids":[hid], "code":f"Welch t-test pfs by {col}",
                "result_summary": f"Mean pfs: {col}=1 {ma:.2f} vs =0 {mb:.2f}; diff={eff:+.3f}, p={pv:.2e}.",
                "p_value": pv, "effect_estimate": eff, "significant": pv<0.05})
iterations.append({"index": 15, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 16 — bevacizumab subgroup heterogeneity ----------
hyps = []; ans = []
hyps.append({"id":"h16.1","text":"Bevacizumab effect on pfs_months does not depend on KRAS mutation status (interaction term ~0).","kind":"refined"})
hyps.append({"id":"h16.2","text":"Bevacizumab effect on pfs_months does not depend on right_sided_primary (interaction term ~0).","kind":"refined"})
hyps.append({"id":"h16.3","text":"Bevacizumab is associated with longer pfs_months in both KRAS-mutant and KRAS-wild-type subgroups (broad benefit).","kind":"refined"})
m = smf.ols("pfs_months ~ treatment_bevacizumab * kras_mutation", data=df).fit()
e = float(m.params["treatment_bevacizumab:kras_mutation"]); p = float(m.pvalues["treatment_bevacizumab:kras_mutation"])
ans.append({"hypothesis_ids":["h16.1"], "code":"OLS pfs_months ~ treatment_bevacizumab * kras_mutation",
            "result_summary": f"Bevacizumab*KRAS interaction = {e:+.3f} mo (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
m = smf.ols("pfs_months ~ treatment_bevacizumab * right_sided_primary", data=df).fit()
e = float(m.params["treatment_bevacizumab:right_sided_primary"]); p = float(m.pvalues["treatment_bevacizumab:right_sided_primary"])
ans.append({"hypothesis_ids":["h16.2"], "code":"OLS pfs_months ~ treatment_bevacizumab * right_sided_primary",
            "result_summary": f"Bevacizumab*right_sided interaction = {e:+.3f} mo (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
sub_kras = df[df["kras_mutation"]==1]; sub_kraswt = df[df["kras_mutation"]==0]
e1, p1, _, _, _, _ = by_binary(sub_kras, "treatment_bevacizumab")
e2, p2, _, _, _, _ = by_binary(sub_kraswt, "treatment_bevacizumab")
ans.append({"hypothesis_ids":["h16.3"], "code":"Stratified bevacizumab effect by KRAS",
            "result_summary": f"Bevacizumab effect KRAS-mut {e1:+.3f} (p={p1:.2e}); KRAS-WT {e2:+.3f} (p={p2:.2e}).",
            "p_value": min(p1,p2), "effect_estimate": (e1+e2)/2, "significant": (p1<0.05 and p2<0.05)})
iterations.append({"index": 16, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 17 — SNP scan (univariate) ----------
hyps = []; ans = []
snp_cols = [c for c in df.columns if c.startswith("snp_")]
hyps.append({"id":"h17.1","text":f"At least one of the {len(snp_cols)} SNPs (snp_rs* columns) shows a Bonferroni-significant association with pfs_months across the full cohort (additive coding 0/1/2).","kind":"novel"})
results = []
for s in snp_cols:
    eff, pv = linreg(df, s)
    results.append((s, eff, pv))
results.sort(key=lambda r: r[2])
top_s, top_eff, top_pv = results[0]
bonf_thresh = 0.05/len(snp_cols)
n_sig = sum(1 for _, _, p in results if p < bonf_thresh)
ans.append({"hypothesis_ids":["h17.1"], "code":"OLS pfs_months ~ snp (each)",
            "result_summary": (f"Top SNP: {top_s}, slope={top_eff:+.4f} mo/allele, p={top_pv:.2e}. "
                               f"{n_sig}/{len(snp_cols)} SNPs pass Bonferroni (alpha={bonf_thresh:.2e})."),
            "p_value": top_pv, "effect_estimate": top_eff, "significant": top_pv < bonf_thresh})
iterations.append({"index": 17, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 18 — biology-driven SNPs and irinotecan-related ----------
# Note: rs1045642 (ABCB1), rs1065852 (CYP2D6), rs4244285 (CYP2C19), rs1799853 (CYP2C9),
# rs1800629 (TNF), rs1800896 (IL10), rs1801133 (MTHFR C677T)
hyps = []; ans = []
hyps.append({"id":"h18.1","text":"snp_rs1801133 (MTHFR C677T) carriers show a different mean pfs_months than non-carriers (additive slope ≠ 0).","kind":"novel"})
hyps.append({"id":"h18.2","text":"snp_rs1065852 (CYP2D6) genotype is associated with pfs_months (additive slope ≠ 0).","kind":"novel"})
hyps.append({"id":"h18.3","text":"snp_rs4244285 (CYP2C19) genotype is associated with pfs_months (additive slope ≠ 0).","kind":"novel"})
for hid, col in [("h18.1","snp_rs1801133"),("h18.2","snp_rs1065852"),("h18.3","snp_rs4244285")]:
    eff, pv = linreg(df, col)
    ans.append({"hypothesis_ids":[hid], "code":f"OLS pfs_months ~ {col}",
                "result_summary": f"Slope of pfs on {col}: {eff:+.4f} mo/allele (p={pv:.2e}).",
                "p_value": pv, "effect_estimate": eff, "significant": pv<0.05})
iterations.append({"index": 18, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 19 — regorafenib subgroups ----------
hyps = []; ans = []
hyps.append({"id":"h19.1","text":"Regorafenib (treatment_regorafenib=1) is associated with shorter pfs_months even after adjusting for prior_lines_of_therapy (negative coefficient).","kind":"refined"})
hyps.append({"id":"h19.2","text":"Among patients with ≥2 prior_lines_of_therapy, regorafenib is associated with longer pfs_months than no regorafenib (positive mean difference within heavily pretreated patients).","kind":"refined"})
m = smf.ols("pfs_months ~ treatment_regorafenib + prior_lines_of_therapy + ecog_ps + stage_iv", data=df).fit()
e = float(m.params["treatment_regorafenib"]); p = float(m.pvalues["treatment_regorafenib"])
ans.append({"hypothesis_ids":["h19.1"], "code":"OLS pfs_months ~ treatment_regorafenib + prior_lines_of_therapy + ecog_ps + stage_iv",
            "result_summary": f"Adjusted regorafenib coefficient: {e:+.3f} mo (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
sub = df[df["prior_lines_of_therapy"]>=2]
eff, pv, ma, mb, na, nb = by_binary(sub, "treatment_regorafenib")
ans.append({"hypothesis_ids":["h19.2"], "code":"Welch t-test pfs by treatment_regorafenib in heavily pretreated subset (>=2 prior lines)",
            "result_summary": f"Heavily pretreated regorafenib effect: {eff:+.3f} mo ({ma:.2f} vs {mb:.2f}, n={na}/{nb}, p={pv:.2e}).",
            "p_value": pv, "effect_estimate": eff, "significant": pv<0.05})
iterations.append({"index": 19, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 20 — confounding for cetux: re-test cetux in WT after adjustment ----------
hyps = []; ans = []
hyps.append({"id":"h20.1","text":"After adjusting for prognostic covariates (ecog_ps, stage_iv, age_years, albumin_g_dl, ldh_u_l, weight_loss_pct_6mo, crp_mg_l, prior_lines_of_therapy), cetuximab retains a positive coefficient on pfs_months in RAS/RAF wild-type patients.","kind":"refined"})
hyps.append({"id":"h20.2","text":"After the same adjustment, pembrolizumab retains a positive coefficient in MSI-high patients.","kind":"refined"})
hyps.append({"id":"h20.3","text":"After the same adjustment, encorafenib retains a positive coefficient in BRAF V600E-mutant patients.","kind":"refined"})

m = smf.ols("pfs_months ~ treatment_cetuximab + ecog_ps + stage_iv + age_years + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + prior_lines_of_therapy", data=df[df['rasraf_wt']==1]).fit()
e = float(m.params["treatment_cetuximab"]); p = float(m.pvalues["treatment_cetuximab"])
ans.append({"hypothesis_ids":["h20.1"], "code":"OLS pfs_months ~ treatment_cetuximab + covariates (RAS/RAF WT subset)",
            "result_summary": f"Adjusted cetuximab effect (WT): {e:+.3f} mo (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
m = smf.ols("pfs_months ~ treatment_pembrolizumab + ecog_ps + stage_iv + age_years + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + prior_lines_of_therapy", data=df[df['msi_high']==1]).fit()
e = float(m.params["treatment_pembrolizumab"]); p = float(m.pvalues["treatment_pembrolizumab"])
ans.append({"hypothesis_ids":["h20.2"], "code":"OLS pfs_months ~ treatment_pembrolizumab + covariates (MSI-high subset)",
            "result_summary": f"Adjusted pembrolizumab effect (MSI-H): {e:+.3f} mo (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
m = smf.ols("pfs_months ~ treatment_encorafenib + ecog_ps + stage_iv + age_years + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + prior_lines_of_therapy", data=df[df['braf_v600e']==1]).fit()
e = float(m.params["treatment_encorafenib"]); p = float(m.pvalues["treatment_encorafenib"])
ans.append({"hypothesis_ids":["h20.3"], "code":"OLS pfs_months ~ treatment_encorafenib + covariates (BRAFV600E subset)",
            "result_summary": f"Adjusted encorafenib effect (BRAF V600E): {e:+.3f} mo (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
iterations.append({"index": 20, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 21 — three-way: cetux x WT x sidedness ----------
hyps = []; ans = []
hyps.append({"id":"h21.1","text":"In RAS/RAF wild-type, left-sided (right_sided_primary=0) patients, cetuximab is associated with longer pfs_months (positive mean difference); the effect is the largest of any biomarker-defined cetuximab subgroup.","kind":"refined"})
hyps.append({"id":"h21.2","text":"In RAS/RAF wild-type, right-sided patients, cetuximab effect on pfs_months is null or negative.","kind":"refined"})

wt_left = df[(df["rasraf_wt"]==1) & (df["right_sided_primary"]==0)]
wt_right = df[(df["rasraf_wt"]==1) & (df["right_sided_primary"]==1)]
mut_left = df[(df["rasraf_wt"]==0) & (df["right_sided_primary"]==0)]
mut_right = df[(df["rasraf_wt"]==0) & (df["right_sided_primary"]==1)]
e_wl, p_wl, ma, mb, na, nb = by_binary(wt_left, "treatment_cetuximab")
e_wr, p_wr, _, _, _, _ = by_binary(wt_right, "treatment_cetuximab")
e_ml, p_ml, _, _, _, _ = by_binary(mut_left, "treatment_cetuximab")
e_mr, p_mr, _, _, _, _ = by_binary(mut_right, "treatment_cetuximab")
ans.append({"hypothesis_ids":["h21.1"], "code":"Cetuximab effect in WT-left subset (Welch t)",
            "result_summary": (f"Cetux effect WT-left: {e_wl:+.3f} mo (p={p_wl:.2e}); WT-right {e_wr:+.3f}, "
                               f"MUT-left {e_ml:+.3f}, MUT-right {e_mr:+.3f}."),
            "p_value": p_wl, "effect_estimate": e_wl, "significant": p_wl<0.05})
ans.append({"hypothesis_ids":["h21.2"], "code":"Cetuximab effect in WT-right subset (Welch t)",
            "result_summary": f"Cetux effect WT-right: {e_wr:+.3f} mo (p={p_wr:.2e}).",
            "p_value": p_wr, "effect_estimate": e_wr, "significant": p_wr<0.05})
iterations.append({"index": 21, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 22 — pembro × MSI in pembro-treated subset ----------
hyps = []; ans = []
hyps.append({"id":"h22.1","text":"Among pembrolizumab-treated patients, MSI-high status (msi_high=1) is associated with substantially longer pfs_months than MSS (positive mean difference).","kind":"refined"})
hyps.append({"id":"h22.2","text":"Among non-pembrolizumab-treated patients, MSI-high status is not associated with a meaningful pfs_months difference (effect ~0).","kind":"refined"})

pembro = df[df["treatment_pembrolizumab"]==1]
nonpembro = df[df["treatment_pembrolizumab"]==0]
e1, p1, ma1, mb1, na1, nb1 = by_binary(pembro, "msi_high")
e2, p2, ma2, mb2, na2, nb2 = by_binary(nonpembro, "msi_high")
ans.append({"hypothesis_ids":["h22.1"], "code":"Welch t-test pfs by msi_high in pembro-treated subset",
            "result_summary": f"Pembro MSI-H {ma1:.2f} (n={na1}) vs MSS {mb1:.2f} (n={nb1}); diff={e1:+.3f}, p={p1:.2e}.",
            "p_value": p1, "effect_estimate": e1, "significant": p1<0.05})
ans.append({"hypothesis_ids":["h22.2"], "code":"Welch t-test pfs by msi_high in non-pembro subset",
            "result_summary": f"Non-pembro MSI-H {ma2:.2f} (n={na2}) vs MSS {mb2:.2f} (n={nb2}); diff={e2:+.3f}, p={p2:.2e}.",
            "p_value": p2, "effect_estimate": e2, "significant": p2<0.05})
iterations.append({"index": 22, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 23 — common mutations & PFS ----------
hyps = []; ans = []
hyps.append({"id":"h23.1","text":"BRAF V600E mutation is associated with shorter pfs_months overall (negative mean difference).","kind":"novel"})
hyps.append({"id":"h23.2","text":"MSI-high status is associated with longer pfs_months overall (positive mean difference).","kind":"novel"})
hyps.append({"id":"h23.3","text":"TP53 mutation (tp53_mutation=1) is associated with shorter pfs_months overall.","kind":"novel"})
hyps.append({"id":"h23.4","text":"PIK3CA mutation is associated with shorter pfs_months overall.","kind":"novel"})
for hid, col in [("h23.1","braf_v600e"),("h23.2","msi_high"),("h23.3","tp53_mutation"),("h23.4","pik3ca_mutation")]:
    eff, pv, ma, mb, na, nb = by_binary(df, col)
    ans.append({"hypothesis_ids":[hid], "code":f"Welch t-test pfs by {col}",
                "result_summary": f"Mean pfs: {col}=1 {ma:.2f} (n={na}) vs =0 {mb:.2f} (n={nb}); diff={eff:+.3f}, p={pv:.2e}.",
                "p_value": pv, "effect_estimate": eff, "significant": pv<0.05})
iterations.append({"index": 23, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 24 — interaction: bevacizumab × stage_iv ----------
hyps = []; ans = []
hyps.append({"id":"h24.1","text":"Bevacizumab effect on pfs_months is similar across stage_iv and non-stage_iv (interaction term ~0; broad benefit).","kind":"refined"})
hyps.append({"id":"h24.2","text":"In a fully adjusted multivariable model (stage_iv, ecog_ps, age, albumin, ldh, prior_lines_of_therapy, all six treatments, and major biomarkers), bevacizumab retains a positive coefficient on pfs_months.","kind":"refined"})
hyps.append({"id":"h24.3","text":"In the same fully adjusted model, regorafenib retains a negative coefficient on pfs_months.","kind":"refined"})

m = smf.ols("pfs_months ~ treatment_bevacizumab * stage_iv", data=df).fit()
e = float(m.params["treatment_bevacizumab:stage_iv"]); p = float(m.pvalues["treatment_bevacizumab:stage_iv"])
ans.append({"hypothesis_ids":["h24.1"], "code":"OLS pfs_months ~ treatment_bevacizumab * stage_iv",
            "result_summary": f"Bev*stage_iv interaction = {e:+.3f} mo (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
formula = ("pfs_months ~ treatment_cetuximab + treatment_bevacizumab + treatment_pembrolizumab + "
           "treatment_encorafenib + treatment_trastuzumab_tucatinib + treatment_regorafenib + "
           "ecog_ps + stage_iv + age_years + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + "
           "prior_lines_of_therapy + kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified + "
           "right_sided_primary + liver_mets")
m_full = smf.ols(formula, data=df).fit()
e = float(m_full.params["treatment_bevacizumab"]); p = float(m_full.pvalues["treatment_bevacizumab"])
ans.append({"hypothesis_ids":["h24.2"], "code":"Multivariable OLS with all treatments+covariates",
            "result_summary": f"Adjusted bevacizumab coefficient: {e:+.3f} mo (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
e = float(m_full.params["treatment_regorafenib"]); p = float(m_full.pvalues["treatment_regorafenib"])
ans.append({"hypothesis_ids":["h24.3"], "code":"Multivariable OLS with all treatments+covariates",
            "result_summary": f"Adjusted regorafenib coefficient: {e:+.3f} mo (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
iterations.append({"index": 24, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- ITERATION 25 — final: targeted-treatment × matched biomarker, all together ----------
hyps = []; ans = []
hyps.append({"id":"h25.1","text":"Trastuzumab+tucatinib coefficient remains positive in HER2-amplified patients after adjustment for prognostic covariates (ecog_ps, stage_iv, age_years, albumin_g_dl, ldh_u_l, weight_loss_pct_6mo, crp_mg_l, prior_lines_of_therapy).","kind":"refined"})
hyps.append({"id":"h25.2","text":"Cetuximab × RAS/RAF-WT × right_sided_primary: among RAS/RAF WT patients, the cetuximab × right_sided interaction is negative (cetuximab benefit attenuated/reversed in right-sided primaries).","kind":"refined"})
hyps.append({"id":"h25.3","text":"Higher prior_lines_of_therapy is associated with shorter pfs_months even after adjusting for ecog_ps and stage_iv (negative coefficient retained).","kind":"refined"})

m = smf.ols("pfs_months ~ treatment_trastuzumab_tucatinib + ecog_ps + stage_iv + age_years + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + prior_lines_of_therapy", data=df[df["her2_amplified"]==1]).fit()
e = float(m.params["treatment_trastuzumab_tucatinib"]); p = float(m.pvalues["treatment_trastuzumab_tucatinib"])
ans.append({"hypothesis_ids":["h25.1"], "code":"OLS pfs_months ~ trastuzumab_tucatinib + covariates (HER2-amp subset)",
            "result_summary": f"Adjusted trastuzumab/tucatinib effect (HER2-amp): {e:+.3f} mo (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
m = smf.ols("pfs_months ~ treatment_cetuximab * right_sided_primary", data=df[df["rasraf_wt"]==1]).fit()
e = float(m.params["treatment_cetuximab:right_sided_primary"]); p = float(m.pvalues["treatment_cetuximab:right_sided_primary"])
ans.append({"hypothesis_ids":["h25.2"], "code":"OLS pfs_months ~ cetuximab*right_sided (RAS/RAF WT)",
            "result_summary": f"Cetuximab*right_sided interaction in WT: {e:+.3f} mo (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
m = smf.ols("pfs_months ~ prior_lines_of_therapy + ecog_ps + stage_iv", data=df).fit()
e = float(m.params["prior_lines_of_therapy"]); p = float(m.pvalues["prior_lines_of_therapy"])
ans.append({"hypothesis_ids":["h25.3"], "code":"OLS pfs_months ~ prior_lines_of_therapy + ecog_ps + stage_iv",
            "result_summary": f"Adjusted prior_lines_of_therapy slope: {e:+.4f} mo/line (p={p:.2e}).",
            "p_value": p, "effect_estimate": e, "significant": p<0.05})
iterations.append({"index": 25, "proposed_hypotheses": hyps, "analyses": ans})

# ---------- WRITE TRANSCRIPT ----------
transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "manual-claude-code-run@2026-04-28",
    "max_iterations": 25,
    "iterations": iterations,
}
with open("transcript.json","w") as f:
    json.dump(transcript, f, indent=2, default=float)

# Print summary mat to stdout
print(f"Wrote transcript.json with {len(iterations)} iterations.")
total_h = sum(len(it['proposed_hypotheses']) for it in iterations)
total_a = sum(len(it['analyses']) for it in iterations)
print(f"Total hypotheses: {total_h}; total analyses: {total_a}.")
