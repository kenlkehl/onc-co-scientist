"""Run all 25 iterations of analyses; produce results.json with hypothesis IDs, effects, p-values."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
y = "objective_response"

results = {"iterations": []}


def add_iter(idx, hyps, analyses):
    results["iterations"].append({
        "index": idx,
        "proposed_hypotheses": hyps,
        "analyses": analyses,
    })


def chi2_or(df, exposure, outcome=y):
    """2x2 chi-square; effect estimate = log odds ratio."""
    a = df[(df[exposure] == 1) & (df[outcome] == 1)].shape[0]
    b = df[(df[exposure] == 1) & (df[outcome] == 0)].shape[0]
    c = df[(df[exposure] == 0) & (df[outcome] == 1)].shape[0]
    d = df[(df[exposure] == 0) & (df[outcome] == 0)].shape[0]
    table = np.array([[a, b], [c, d]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    # signed log-OR
    if a == 0 or b == 0 or c == 0 or d == 0:
        a += 0.5; b += 0.5; c += 0.5; d += 0.5
    log_or = np.log((a * d) / (b * c))
    p1 = a / (a + b) if (a + b) > 0 else 0
    p0 = c / (c + d) if (c + d) > 0 else 0
    return {
        "p_value": float(p),
        "effect_estimate": float(p1 - p0),
        "significant": bool(p < 0.05),
        "log_or": float(log_or),
        "rate_exposed": float(p1),
        "rate_unexposed": float(p0),
        "n_exposed": int(a + b),
        "n_unexposed": int(c + d),
    }


def cont_logreg(df, predictor, outcome=y):
    """Logistic regression of outcome on continuous predictor; effect = beta (log-OR per unit)."""
    X = sm.add_constant(df[[predictor]].astype(float))
    y_ = df[outcome].astype(int)
    try:
        m = sm.Logit(y_, X).fit(disp=False)
        return {
            "p_value": float(m.pvalues[predictor]),
            "effect_estimate": float(m.params[predictor]),
            "significant": bool(m.pvalues[predictor] < 0.05),
        }
    except Exception as e:
        return {"p_value": None, "effect_estimate": None, "significant": None, "error": str(e)}


def interaction_logreg(df, tx, marker, outcome=y):
    """Logistic regression with interaction term; effect = beta of tx*marker."""
    sub = df[[tx, marker, outcome]].copy()
    sub["inter"] = sub[tx] * sub[marker]
    X = sm.add_constant(sub[[tx, marker, "inter"]].astype(float))
    y_ = sub[outcome].astype(int)
    try:
        m = sm.Logit(y_, X).fit(disp=False)
        return {
            "p_value": float(m.pvalues["inter"]),
            "effect_estimate": float(m.params["inter"]),
            "significant": bool(m.pvalues["inter"] < 0.05),
            "main_tx_beta": float(m.params[tx]),
            "main_marker_beta": float(m.params[marker]),
        }
    except Exception as e:
        return {"p_value": None, "effect_estimate": None, "significant": None, "error": str(e)}


def stratum_rate(df, tx, marker, marker_val, outcome=y):
    sub = df[df[marker] == marker_val]
    if sub.empty:
        return None
    return chi2_or(sub, tx, outcome)


# =====================================================================
# Iteration 1 — Age, sex, ECOG main effects
# =====================================================================
hyps = [
    {"id": "h1_age", "text": "Older age (age_years) is associated with lower probability of objective_response in AML patients."},
    {"id": "h1_sex", "text": "Female sex (sex_female=1) is associated with a different objective_response rate compared to male patients."},
    {"id": "h1_ecog", "text": "Higher ECOG performance status (ecog_ps) is associated with lower probability of objective_response."},
]
analyses = []
r = cont_logreg(df, "age_years")
analyses.append({
    "hypothesis_ids": ["h1_age"],
    "code": "sm.Logit(df.objective_response, sm.add_constant(df.age_years)).fit()",
    "result_summary": f"Logistic regression of response on age_years: beta={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
r = chi2_or(df, "sex_female")
analyses.append({
    "hypothesis_ids": ["h1_sex"],
    "code": "stats.chi2_contingency on sex_female x objective_response",
    "result_summary": f"Response rate female {r['rate_exposed']:.3f} vs male {r['rate_unexposed']:.3f}; chi2 p={r['p_value']:.4g}; rate diff={r['effect_estimate']:.4f}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
r = cont_logreg(df, "ecog_ps")
analyses.append({
    "hypothesis_ids": ["h1_ecog"],
    "code": "sm.Logit(response, ecog_ps)",
    "result_summary": f"Logistic regression of response on ECOG: beta={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
add_iter(1, hyps, analyses)

# =====================================================================
# Iteration 2 — AML driver mutations main effects
# =====================================================================
hyps = [
    {"id": "h2_npm1", "text": "Patients with npm1_mutation=1 have a higher objective_response rate than those without."},
    {"id": "h2_flt3itd", "text": "Patients with flt3_itd=1 have a different objective_response rate than those without."},
    {"id": "h2_idh1", "text": "Patients with idh1_mutation=1 have a higher objective_response rate than those without."},
    {"id": "h2_idh2", "text": "Patients with idh2_mutation=1 have a different objective_response rate than those without."},
    {"id": "h2_tp53", "text": "Patients with tp53_mutation=1 have a lower objective_response rate than those without."},
    {"id": "h2_ck", "text": "Patients with complex_karyotype=1 have a lower objective_response rate than those without."},
]
analyses = []
for hid, col in [("h2_npm1","npm1_mutation"),("h2_flt3itd","flt3_itd"),("h2_idh1","idh1_mutation"),
                 ("h2_idh2","idh2_mutation"),("h2_tp53","tp53_mutation"),("h2_ck","complex_karyotype")]:
    r = chi2_or(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2_contingency(crosstab({col}, objective_response))",
        "result_summary": f"{col}: response {r['rate_exposed']:.3f} (n={r['n_exposed']}) vs {r['rate_unexposed']:.3f} (n={r['n_unexposed']}); rate diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(2, hyps, analyses)

# =====================================================================
# Iteration 3 — Treatment main effects
# =====================================================================
hyps = [
    {"id": "h3_mido", "text": "Receipt of treatment_midostaurin is associated with a higher objective_response rate vs not receiving it."},
    {"id": "h3_gilt", "text": "Receipt of treatment_gilteritinib is associated with a higher objective_response rate vs not receiving it."},
    {"id": "h3_ivo", "text": "Receipt of treatment_ivosidenib is associated with a higher objective_response rate vs not receiving it."},
    {"id": "h3_ena", "text": "Receipt of treatment_enasidenib is associated with a higher objective_response rate vs not receiving it."},
    {"id": "h3_venaza", "text": "Receipt of treatment_venetoclax_azacitidine is associated with a higher objective_response rate vs not receiving it."},
    {"id": "h3_7p3", "text": "Receipt of treatment_7plus3 is associated with a higher objective_response rate vs not receiving it."},
]
analyses = []
for hid, col in [("h3_mido","treatment_midostaurin"),("h3_gilt","treatment_gilteritinib"),
                 ("h3_ivo","treatment_ivosidenib"),("h3_ena","treatment_enasidenib"),
                 ("h3_venaza","treatment_venetoclax_azacitidine"),("h3_7p3","treatment_7plus3")]:
    r = chi2_or(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2 on {col}",
        "result_summary": f"{col}: response on {r['rate_exposed']:.3f} (n={r['n_exposed']}) vs off {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(3, hyps, analyses)

# =====================================================================
# Iteration 4 — Targeted-therapy × biomarker interactions (clinically expected)
# =====================================================================
hyps = [
    {"id": "h4_mido_flt3itd", "text": "treatment_midostaurin × flt3_itd interaction: midostaurin improves objective_response more in flt3_itd=1 patients than in flt3_itd=0 patients."},
    {"id": "h4_gilt_flt3itd", "text": "treatment_gilteritinib × flt3_itd interaction: gilteritinib improves objective_response more in flt3_itd=1 patients than in flt3_itd=0 patients."},
    {"id": "h4_ivo_idh1", "text": "treatment_ivosidenib × idh1_mutation interaction: ivosidenib improves objective_response more in idh1_mutation=1 patients than in idh1_mutation=0 patients."},
    {"id": "h4_ena_idh2", "text": "treatment_enasidenib × idh2_mutation interaction: enasidenib improves objective_response more in idh2_mutation=1 patients than in idh2_mutation=0 patients."},
]
analyses = []
for hid, tx, mt in [("h4_mido_flt3itd","treatment_midostaurin","flt3_itd"),
                    ("h4_gilt_flt3itd","treatment_gilteritinib","flt3_itd"),
                    ("h4_ivo_idh1","treatment_ivosidenib","idh1_mutation"),
                    ("h4_ena_idh2","treatment_enasidenib","idh2_mutation")]:
    r = interaction_logreg(df, tx, mt)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"sm.Logit(y, [const,{tx},{mt},{tx}*{mt}])",
        "result_summary": f"{tx}*{mt} interaction beta={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(4, hyps, analyses)

# =====================================================================
# Iteration 5 — Lab markers main effects
# =====================================================================
hyps = [
    {"id": "h5_wbc", "text": "Higher wbc_k_per_ul is associated with lower probability of objective_response."},
    {"id": "h5_blast", "text": "Higher blast_pct_marrow is associated with lower probability of objective_response."},
    {"id": "h5_alb", "text": "Higher albumin_g_dl is associated with higher probability of objective_response."},
    {"id": "h5_ldh", "text": "Higher ldh_u_l is associated with lower probability of objective_response."},
    {"id": "h5_hgb", "text": "Higher hemoglobin_g_dl is associated with higher probability of objective_response."},
    {"id": "h5_plt", "text": "Higher platelets_k_ul is associated with higher probability of objective_response."},
]
analyses = []
for hid, col in [("h5_wbc","wbc_k_per_ul"),("h5_blast","blast_pct_marrow"),("h5_alb","albumin_g_dl"),
                 ("h5_ldh","ldh_u_l"),("h5_hgb","hemoglobin_g_dl"),("h5_plt","platelets_k_ul")]:
    r = cont_logreg(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"sm.Logit(y, [const,{col}])",
        "result_summary": f"{col}: logistic beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(5, hyps, analyses)

# =====================================================================
# Iteration 6 — Stratified targeted-therapy effects within mutation strata
# =====================================================================
hyps = [
    {"id": "h6_mido_in_itd", "text": "Within flt3_itd=1 patients, treatment_midostaurin is associated with higher objective_response than no midostaurin."},
    {"id": "h6_gilt_in_itd", "text": "Within flt3_itd=1 patients, treatment_gilteritinib is associated with higher objective_response than no gilteritinib."},
    {"id": "h6_ivo_in_idh1", "text": "Within idh1_mutation=1 patients, treatment_ivosidenib is associated with higher objective_response than no ivosidenib."},
    {"id": "h6_ena_in_idh2", "text": "Within idh2_mutation=1 patients, treatment_enasidenib is associated with higher objective_response than no enasidenib."},
]
analyses = []
for hid, tx, mt in [("h6_mido_in_itd","treatment_midostaurin","flt3_itd"),
                    ("h6_gilt_in_itd","treatment_gilteritinib","flt3_itd"),
                    ("h6_ivo_in_idh1","treatment_ivosidenib","idh1_mutation"),
                    ("h6_ena_in_idh2","treatment_enasidenib","idh2_mutation")]:
    r = stratum_rate(df, tx, mt, 1)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2 on {tx} within {mt}==1 stratum",
        "result_summary": f"In {mt}=1: {tx}+ rate {r['rate_exposed']:.3f} vs {tx}- rate {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(6, hyps, analyses)

# =====================================================================
# Iteration 7 — Venetoclax/aza vs 7+3 by fitness (unfit_for_intensive)
# =====================================================================
hyps = [
    {"id": "h7_venaza_unfit", "text": "treatment_venetoclax_azacitidine × unfit_for_intensive interaction: ven/aza improves objective_response more in unfit_for_intensive=1 patients than in unfit_for_intensive=0 patients."},
    {"id": "h7_7p3_unfit", "text": "treatment_7plus3 × unfit_for_intensive interaction: 7+3 improves objective_response more in unfit_for_intensive=0 patients than in unfit_for_intensive=1 patients."},
    {"id": "h7_venaza_in_unfit", "text": "Within unfit_for_intensive=1 patients, treatment_venetoclax_azacitidine is associated with higher objective_response."},
    {"id": "h7_7p3_in_fit", "text": "Within unfit_for_intensive=0 patients, treatment_7plus3 is associated with higher objective_response."},
]
analyses = []
for hid, tx, mt in [("h7_venaza_unfit","treatment_venetoclax_azacitidine","unfit_for_intensive"),
                    ("h7_7p3_unfit","treatment_7plus3","unfit_for_intensive")]:
    r = interaction_logreg(df, tx, mt)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit y ~ {tx} + {mt} + {tx}:{mt}",
        "result_summary": f"interaction beta={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
r = stratum_rate(df, "treatment_venetoclax_azacitidine", "unfit_for_intensive", 1)
analyses.append({
    "hypothesis_ids": ["h7_venaza_in_unfit"],
    "code": "chi2 of ven/aza within unfit_for_intensive==1",
    "result_summary": f"unfit subset: ven/aza+ {r['rate_exposed']:.3f} vs ven/aza- {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
r = stratum_rate(df, "treatment_7plus3", "unfit_for_intensive", 0)
analyses.append({
    "hypothesis_ids": ["h7_7p3_in_fit"],
    "code": "chi2 of 7+3 within unfit_for_intensive==0",
    "result_summary": f"fit subset: 7+3+ {r['rate_exposed']:.3f} vs 7+3- {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
add_iter(7, hyps, analyses)

# =====================================================================
# Iteration 8 — TP53 + complex karyotype prognostic
# =====================================================================
hyps = [
    {"id": "h8_tp53_ck", "text": "Patients with both tp53_mutation=1 AND complex_karyotype=1 have a lower objective_response rate than other patients."},
    {"id": "h8_tp53_x_7p3", "text": "treatment_7plus3 × tp53_mutation interaction: 7+3 effect on objective_response differs by tp53_mutation status."},
    {"id": "h8_tp53_x_venaza", "text": "treatment_venetoclax_azacitidine × tp53_mutation interaction: ven/aza effect differs by tp53_mutation status."},
]
analyses = []
df_ = df.copy()
df_["tp53_ck"] = ((df_["tp53_mutation"] == 1) & (df_["complex_karyotype"] == 1)).astype(int)
r = chi2_or(df_, "tp53_ck")
analyses.append({
    "hypothesis_ids": ["h8_tp53_ck"],
    "code": "chi2 of (tp53 AND complex_karyotype) flag vs response",
    "result_summary": f"tp53+ck combo: {r['rate_exposed']:.3f} (n={r['n_exposed']}) vs others {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
for hid, tx, mt in [("h8_tp53_x_7p3","treatment_7plus3","tp53_mutation"),
                    ("h8_tp53_x_venaza","treatment_venetoclax_azacitidine","tp53_mutation")]:
    r = interaction_logreg(df, tx, mt)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit y ~ {tx}+{mt}+inter",
        "result_summary": f"interaction beta={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(8, hyps, analyses)

# =====================================================================
# Iteration 9 — NPM1 (favorable) interactions
# =====================================================================
hyps = [
    {"id": "h9_npm1_no_flt3", "text": "Patients with npm1_mutation=1 AND flt3_itd=0 (favorable) have higher objective_response than other patients."},
    {"id": "h9_npm1_x_7p3", "text": "treatment_7plus3 × npm1_mutation interaction: 7+3 benefit (vs no 7+3) is greater in npm1_mutation=1 patients."},
]
analyses = []
df_ = df.copy()
df_["npm1_no_itd"] = ((df_["npm1_mutation"] == 1) & (df_["flt3_itd"] == 0)).astype(int)
r = chi2_or(df_, "npm1_no_itd")
analyses.append({
    "hypothesis_ids": ["h9_npm1_no_flt3"],
    "code": "chi2 of (npm1=1 & flt3_itd=0) vs response",
    "result_summary": f"npm1+/flt3itd-: rate {r['rate_exposed']:.3f} (n={r['n_exposed']}) vs others {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
r = interaction_logreg(df, "treatment_7plus3", "npm1_mutation")
analyses.append({
    "hypothesis_ids": ["h9_npm1_x_7p3"],
    "code": "logit y ~ 7+3 + npm1 + inter",
    "result_summary": f"7+3 × npm1 interaction beta={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
add_iter(9, hyps, analyses)

# =====================================================================
# Iteration 10 — Sociodemographic disparities
# =====================================================================
hyps = [
    {"id": "h10_race", "text": "Objective_response rates differ across race_ethnicity categories (white, black, hispanic, asian, other)."},
    {"id": "h10_insurance", "text": "Objective_response rates differ across insurance_type categories (medicare, medicaid, private, uninsured)."},
    {"id": "h10_rural", "text": "Patients with rural_residence=1 have a different objective_response rate than urban patients."},
    {"id": "h10_education", "text": "Higher education_years is associated with higher probability of objective_response."},
]
analyses = []
ct = pd.crosstab(df["race_ethnicity"], df[y])
chi2, p, _, _ = stats.chi2_contingency(ct)
rates = df.groupby("race_ethnicity")[y].mean().to_dict()
analyses.append({
    "hypothesis_ids": ["h10_race"],
    "code": "chi2_contingency(crosstab(race_ethnicity, response))",
    "result_summary": f"race_ethnicity rates {rates}; chi2 p={p:.4g}",
    "p_value": float(p), "effect_estimate": float(max(rates.values())-min(rates.values())), "significant": bool(p<0.05),
})
ct = pd.crosstab(df["insurance_type"], df[y])
chi2, p, _, _ = stats.chi2_contingency(ct)
rates = df.groupby("insurance_type")[y].mean().to_dict()
analyses.append({
    "hypothesis_ids": ["h10_insurance"],
    "code": "chi2_contingency(crosstab(insurance_type, response))",
    "result_summary": f"insurance_type rates {rates}; chi2 p={p:.4g}",
    "p_value": float(p), "effect_estimate": float(max(rates.values())-min(rates.values())), "significant": bool(p<0.05),
})
r = chi2_or(df, "rural_residence")
analyses.append({
    "hypothesis_ids": ["h10_rural"],
    "code": "chi2 rural_residence vs response",
    "result_summary": f"rural {r['rate_exposed']:.3f} vs urban {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
r = cont_logreg(df, "education_years")
analyses.append({
    "hypothesis_ids": ["h10_education"],
    "code": "logit response ~ education_years",
    "result_summary": f"education_years beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
add_iter(10, hyps, analyses)

# =====================================================================
# Iteration 11 — Comorbidities and prior treatment
# =====================================================================
hyps = [
    {"id": "h11_dm", "text": "Patients with diabetes_mellitus=1 have a lower objective_response rate than those without."},
    {"id": "h11_ckd", "text": "Patients with chronic_kidney_disease=1 have a lower objective_response rate than those without."},
    {"id": "h11_hf", "text": "Patients with heart_failure=1 have a lower objective_response rate than those without."},
    {"id": "h11_priormalig", "text": "Patients with prior_malignancy=1 have a lower objective_response rate than those without."},
    {"id": "h11_priorchemo", "text": "Patients with prior_chemotherapy=1 have a lower objective_response rate than those without."},
    {"id": "h11_priorlines", "text": "More prior_lines_of_therapy is associated with lower objective_response."},
]
analyses = []
for hid, col in [("h11_dm","diabetes_mellitus"),("h11_ckd","chronic_kidney_disease"),("h11_hf","heart_failure"),
                 ("h11_priormalig","prior_malignancy"),("h11_priorchemo","prior_chemotherapy")]:
    r = chi2_or(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2 {col}",
        "result_summary": f"{col}: {r['rate_exposed']:.3f} vs {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
r = cont_logreg(df, "prior_lines_of_therapy")
analyses.append({
    "hypothesis_ids": ["h11_priorlines"],
    "code": "logit y ~ prior_lines_of_therapy",
    "result_summary": f"beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
add_iter(11, hyps, analyses)

# =====================================================================
# Iteration 12 — Inflammation and metabolic markers
# =====================================================================
hyps = [
    {"id": "h12_crp", "text": "Higher crp_mg_l is associated with lower probability of objective_response."},
    {"id": "h12_nlr", "text": "Higher nlr is associated with lower probability of objective_response."},
    {"id": "h12_wl", "text": "Higher weight_loss_pct_6mo is associated with lower probability of objective_response."},
    {"id": "h12_bmi", "text": "Higher bmi is associated with higher probability of objective_response."},
]
analyses = []
for hid, col in [("h12_crp","crp_mg_l"),("h12_nlr","nlr"),("h12_wl","weight_loss_pct_6mo"),("h12_bmi","bmi")]:
    r = cont_logreg(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit y ~ {col}",
        "result_summary": f"{col} beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(12, hyps, analyses)

# =====================================================================
# Iteration 13 — Symptom burden
# =====================================================================
hyps = [
    {"id": "h13_fatigue", "text": "Higher fatigue_grade is associated with lower probability of objective_response."},
    {"id": "h13_pain", "text": "Higher pain_nrs is associated with lower probability of objective_response."},
    {"id": "h13_dyspnea", "text": "Higher dyspnea_grade is associated with lower probability of objective_response."},
    {"id": "h13_appetite", "text": "Higher appetite_loss_grade is associated with lower probability of objective_response."},
]
analyses = []
for hid, col in [("h13_fatigue","fatigue_grade"),("h13_pain","pain_nrs"),
                 ("h13_dyspnea","dyspnea_grade"),("h13_appetite","appetite_loss_grade")]:
    r = cont_logreg(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit y ~ {col}",
        "result_summary": f"{col} beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(13, hyps, analyses)

# =====================================================================
# Iteration 14 — Pharmacogenomic SNPs main effects (broad screen)
# =====================================================================
snp_cols = [c for c in df.columns if c.startswith("snp_")]
hyps = []
for s in snp_cols:
    hyps.append({"id": f"h14_{s}", "text": f"The SNP {s} is associated with objective_response (allele dose effect)."})
analyses = []
for s in snp_cols:
    r = cont_logreg(df, s)
    analyses.append({
        "hypothesis_ids": [f"h14_{s}"],
        "code": f"logit y ~ {s}",
        "result_summary": f"{s} beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(14, hyps, analyses)

# =====================================================================
# Iteration 15 — SNP × treatment interaction screen for top SNP
# =====================================================================
# Find the most significant SNP from iter14 then screen interactions
snp_p = []
for s in snp_cols:
    r = cont_logreg(df, s)
    snp_p.append((s, r["p_value"] if r["p_value"] is not None else 1.0))
snp_p.sort(key=lambda x: x[1])
top_snp = snp_p[0][0]

hyps = [
    {"id": "h15_topsnp_x_venaza", "text": f"The {top_snp} SNP modifies the effect of treatment_venetoclax_azacitidine on objective_response (interaction)."},
    {"id": "h15_topsnp_x_7p3", "text": f"The {top_snp} SNP modifies the effect of treatment_7plus3 on objective_response (interaction)."},
    {"id": "h15_topsnp_x_mido", "text": f"The {top_snp} SNP modifies the effect of treatment_midostaurin on objective_response (interaction)."},
]
analyses = []
for hid, tx in [("h15_topsnp_x_venaza","treatment_venetoclax_azacitidine"),
                ("h15_topsnp_x_7p3","treatment_7plus3"),
                ("h15_topsnp_x_mido","treatment_midostaurin")]:
    r = interaction_logreg(df, tx, top_snp)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit y ~ {tx} + {top_snp} + {tx}*{top_snp}",
        "result_summary": f"{tx}*{top_snp} interaction beta={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(15, hyps, analyses)

# =====================================================================
# Iteration 16 — Multivariable model with key features
# =====================================================================
hyps = [
    {"id": "h16_mv_age", "text": "After adjusting for ECOG, blast %, albumin, LDH, NPM1, FLT3-ITD, and TP53, age_years remains independently associated with objective_response."},
    {"id": "h16_mv_npm1", "text": "After adjusting for age, ECOG, blast %, albumin, LDH, FLT3-ITD, TP53, npm1_mutation remains independently associated with objective_response."},
    {"id": "h16_mv_tp53", "text": "After adjusting for age, ECOG, blast %, albumin, LDH, NPM1, FLT3-ITD, tp53_mutation remains independently associated with objective_response."},
    {"id": "h16_mv_alb", "text": "After multivariable adjustment, albumin_g_dl remains independently associated with objective_response."},
]
analyses = []
formula = ("objective_response ~ age_years + ecog_ps + blast_pct_marrow + albumin_g_dl + ldh_u_l + "
           "npm1_mutation + flt3_itd + tp53_mutation + complex_karyotype")
m = smf.logit(formula, data=df).fit(disp=False)
for hid, var in [("h16_mv_age","age_years"),("h16_mv_npm1","npm1_mutation"),
                 ("h16_mv_tp53","tp53_mutation"),("h16_mv_alb","albumin_g_dl")]:
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"smf.logit('{formula}', df).fit()",
        "result_summary": f"{var} adjusted beta={m.params[var]:.4g}, p={m.pvalues[var]:.4g}",
        "p_value": float(m.pvalues[var]), "effect_estimate": float(m.params[var]),
        "significant": bool(m.pvalues[var] < 0.05),
    })
add_iter(16, hyps, analyses)

# =====================================================================
# Iteration 17 — Secondary AML
# =====================================================================
hyps = [
    {"id": "h17_secondary", "text": "Patients with secondary_aml=1 have lower objective_response than de novo AML patients."},
    {"id": "h17_secondary_x_venaza", "text": "treatment_venetoclax_azacitidine × secondary_aml interaction: ven/aza effect differs in secondary AML."},
]
analyses = []
r = chi2_or(df, "secondary_aml")
analyses.append({
    "hypothesis_ids": ["h17_secondary"],
    "code": "chi2 secondary_aml vs response",
    "result_summary": f"secondary {r['rate_exposed']:.3f} vs de novo {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
r = interaction_logreg(df, "treatment_venetoclax_azacitidine", "secondary_aml")
analyses.append({
    "hypothesis_ids": ["h17_secondary_x_venaza"],
    "code": "logit y ~ ven/aza + secondary_aml + interaction",
    "result_summary": f"interaction beta={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
add_iter(17, hyps, analyses)

# =====================================================================
# Iteration 18 — Liver and renal function
# =====================================================================
hyps = [
    {"id": "h18_creat", "text": "Higher creatinine_mg_dl is associated with lower probability of objective_response."},
    {"id": "h18_bili", "text": "Higher total_bilirubin_mg_dl is associated with lower probability of objective_response."},
    {"id": "h18_ast", "text": "Higher ast_u_l is associated with lower probability of objective_response."},
    {"id": "h18_alt", "text": "Higher alt_u_l is associated with lower probability of objective_response."},
]
analyses = []
for hid, col in [("h18_creat","creatinine_mg_dl"),("h18_bili","total_bilirubin_mg_dl"),
                 ("h18_ast","ast_u_l"),("h18_alt","alt_u_l")]:
    r = cont_logreg(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit y ~ {col}",
        "result_summary": f"{col} beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(18, hyps, analyses)

# =====================================================================
# Iteration 19 — Age × treatment interactions (older patients tolerate venaza better)
# =====================================================================
hyps = [
    {"id": "h19_age_x_venaza", "text": "age_years × treatment_venetoclax_azacitidine interaction: ven/aza relative effect on objective_response increases with age."},
    {"id": "h19_age_x_7p3", "text": "age_years × treatment_7plus3 interaction: 7+3 relative effect on objective_response decreases with age."},
    {"id": "h19_ecog_x_7p3", "text": "ecog_ps × treatment_7plus3 interaction: 7+3 effect on objective_response decreases with ECOG."},
]
analyses = []
for hid, tx, mt in [("h19_age_x_venaza","treatment_venetoclax_azacitidine","age_years"),
                    ("h19_age_x_7p3","treatment_7plus3","age_years"),
                    ("h19_ecog_x_7p3","treatment_7plus3","ecog_ps")]:
    r = interaction_logreg(df, tx, mt)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit y ~ {tx} + {mt} + interaction",
        "result_summary": f"{tx}*{mt} interaction beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(19, hyps, analyses)

# =====================================================================
# Iteration 20 — Cell counts (ANC, ALC, platelets) interactions
# =====================================================================
hyps = [
    {"id": "h20_anc", "text": "Higher anc_k_ul is associated with higher probability of objective_response."},
    {"id": "h20_alc", "text": "Higher alc_k_ul is associated with higher probability of objective_response."},
    {"id": "h20_plt_x_venaza", "text": "platelets_k_ul × treatment_venetoclax_azacitidine: at higher platelets, ven/aza objective_response benefit changes."},
]
analyses = []
for hid, col in [("h20_anc","anc_k_ul"),("h20_alc","alc_k_ul")]:
    r = cont_logreg(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit y ~ {col}",
        "result_summary": f"{col} beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
r = interaction_logreg(df, "treatment_venetoclax_azacitidine", "platelets_k_ul")
analyses.append({
    "hypothesis_ids": ["h20_plt_x_venaza"],
    "code": "logit y ~ ven/aza + platelets + inter",
    "result_summary": f"ven/aza × platelets interaction beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
add_iter(20, hyps, analyses)

# =====================================================================
# Iteration 21 — Years since diagnosis
# =====================================================================
hyps = [
    {"id": "h21_ysd", "text": "Higher years_since_diagnosis is associated with lower probability of objective_response."},
    {"id": "h21_pack", "text": "Higher smoking_pack_years is associated with lower probability of objective_response."},
]
analyses = []
for hid, col in [("h21_ysd","years_since_diagnosis"),("h21_pack","smoking_pack_years")]:
    r = cont_logreg(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit y ~ {col}",
        "result_summary": f"{col} beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(21, hyps, analyses)

# =====================================================================
# Iteration 22 — IDH1 + ivosidenib refined; also IDH2 + enasidenib refined
# (focus on response within idh1=1 and idh2=1 separately, account for fitness)
# =====================================================================
hyps = [
    {"id": "h22_ivo_in_idh1_unfit", "text": "Within idh1_mutation=1 AND unfit_for_intensive=1 patients, treatment_ivosidenib is associated with higher objective_response than no ivosidenib."},
    {"id": "h22_ena_in_idh2_unfit", "text": "Within idh2_mutation=1 AND unfit_for_intensive=1 patients, treatment_enasidenib is associated with higher objective_response than no enasidenib."},
    {"id": "h22_idh1_prog", "text": "After adjusting for treatment_ivosidenib and standard covariates, idh1_mutation remains positively associated with objective_response."},
]
analyses = []
sub = df[(df["idh1_mutation"] == 1) & (df["unfit_for_intensive"] == 1)]
r = chi2_or(sub, "treatment_ivosidenib") if not sub.empty else None
if r is not None:
    analyses.append({
        "hypothesis_ids": ["h22_ivo_in_idh1_unfit"],
        "code": "chi2 of ivosidenib in idh1_mutation=1 & unfit=1",
        "result_summary": f"n={sub.shape[0]}; ivo+ rate {r['rate_exposed']:.3f} vs ivo- {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
sub = df[(df["idh2_mutation"] == 1) & (df["unfit_for_intensive"] == 1)]
r = chi2_or(sub, "treatment_enasidenib") if not sub.empty else None
if r is not None:
    analyses.append({
        "hypothesis_ids": ["h22_ena_in_idh2_unfit"],
        "code": "chi2 of enasidenib in idh2_mutation=1 & unfit=1",
        "result_summary": f"n={sub.shape[0]}; ena+ rate {r['rate_exposed']:.3f} vs ena- {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
formula = ("objective_response ~ idh1_mutation + treatment_ivosidenib + age_years + ecog_ps + "
           "albumin_g_dl + tp53_mutation + complex_karyotype + npm1_mutation + flt3_itd")
m = smf.logit(formula, data=df).fit(disp=False)
analyses.append({
    "hypothesis_ids": ["h22_idh1_prog"],
    "code": f"smf.logit('{formula}', df).fit()",
    "result_summary": f"idh1_mutation adjusted beta={m.params['idh1_mutation']:.4g}, p={m.pvalues['idh1_mutation']:.4g}",
    "p_value": float(m.pvalues["idh1_mutation"]),
    "effect_estimate": float(m.params["idh1_mutation"]),
    "significant": bool(m.pvalues["idh1_mutation"] < 0.05),
})
add_iter(22, hyps, analyses)

# =====================================================================
# Iteration 23 — Combinations: 7+3 + venetoclax/aza overlap; treatment interactions
# =====================================================================
hyps = [
    {"id": "h23_7p3_venaza_combo", "text": "Patients receiving both treatment_7plus3=1 AND treatment_venetoclax_azacitidine=1 differ in objective_response from those receiving exactly one."},
    {"id": "h23_mido_x_npm1", "text": "treatment_midostaurin × npm1_mutation interaction: midostaurin effect differs by npm1 status."},
]
analyses = []
df_ = df.copy()
df_["combo"] = ((df_["treatment_7plus3"] == 1) & (df_["treatment_venetoclax_azacitidine"] == 1)).astype(int)
df_["only_one"] = (((df_["treatment_7plus3"] == 1) ^ (df_["treatment_venetoclax_azacitidine"] == 1))).astype(int)
sub = df_[(df_["combo"] == 1) | (df_["only_one"] == 1)]
r = chi2_or(sub, "combo")
analyses.append({
    "hypothesis_ids": ["h23_7p3_venaza_combo"],
    "code": "chi2 (7+3 & venaza) vs (only-one) on response",
    "result_summary": f"both rate {r['rate_exposed']:.3f} (n={r['n_exposed']}) vs single {r['rate_unexposed']:.3f}; diff={r['effect_estimate']:.4f}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
r = interaction_logreg(df, "treatment_midostaurin", "npm1_mutation")
analyses.append({
    "hypothesis_ids": ["h23_mido_x_npm1"],
    "code": "logit y ~ midostaurin + npm1 + inter",
    "result_summary": f"interaction beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
    "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
})
add_iter(23, hyps, analyses)

# =====================================================================
# Iteration 24 — Symptom × treatment, comorbidity × treatment
# =====================================================================
hyps = [
    {"id": "h24_pain_x_7p3", "text": "pain_nrs × treatment_7plus3 interaction: 7+3 effect on response decreases with pain_nrs."},
    {"id": "h24_dm_x_venaza", "text": "diabetes_mellitus × treatment_venetoclax_azacitidine interaction: ven/aza effect differs in diabetics."},
    {"id": "h24_hf_x_7p3", "text": "heart_failure × treatment_7plus3 interaction: 7+3 effect on response is reduced in heart_failure=1 patients."},
]
analyses = []
for hid, tx, mt in [("h24_pain_x_7p3","treatment_7plus3","pain_nrs"),
                    ("h24_dm_x_venaza","treatment_venetoclax_azacitidine","diabetes_mellitus"),
                    ("h24_hf_x_7p3","treatment_7plus3","heart_failure")]:
    r = interaction_logreg(df, tx, mt)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"logit y ~ {tx} + {mt} + interaction",
        "result_summary": f"{tx}*{mt} interaction beta={r['effect_estimate']:.4g}, p={r['p_value']:.4g}",
        "p_value": r["p_value"], "effect_estimate": r["effect_estimate"], "significant": r["significant"],
    })
add_iter(24, hyps, analyses)

# =====================================================================
# Iteration 25 — Final comprehensive multivariable model
# =====================================================================
hyps = [
    {"id": "h25_final_model", "text": "In a multivariable logistic model including age, ECOG, albumin, LDH, blast %, NPM1, FLT3-ITD, IDH1, TP53, complex karyotype, and unfit-for-intensive, multiple variables remain independently and significantly associated with objective_response."},
    {"id": "h25_idh1_mv", "text": "After full multivariable adjustment, idh1_mutation remains positively and significantly associated with objective_response."},
    {"id": "h25_alb_mv", "text": "After full multivariable adjustment, albumin_g_dl remains positively and significantly associated with objective_response."},
    {"id": "h25_age_mv", "text": "After full multivariable adjustment, age_years remains negatively associated with objective_response."},
]
analyses = []
formula = ("objective_response ~ age_years + sex_female + ecog_ps + albumin_g_dl + ldh_u_l + "
           "blast_pct_marrow + wbc_k_per_ul + hemoglobin_g_dl + platelets_k_ul + "
           "npm1_mutation + flt3_itd + idh1_mutation + idh2_mutation + tp53_mutation + "
           "complex_karyotype + secondary_aml + unfit_for_intensive + "
           "treatment_midostaurin + treatment_gilteritinib + treatment_ivosidenib + "
           "treatment_enasidenib + treatment_venetoclax_azacitidine + treatment_7plus3")
m = smf.logit(formula, data=df).fit(disp=False)
sig_vars = [v for v in m.pvalues.index if v != "Intercept" and m.pvalues[v] < 0.05]
analyses.append({
    "hypothesis_ids": ["h25_final_model"],
    "code": f"smf.logit final model: {formula}",
    "result_summary": f"Model significant variables (p<0.05): {sig_vars}; pseudo-R2={m.prsquared:.4f}",
    "p_value": float(min(m.pvalues.drop('Intercept'))) if len(m.pvalues) > 1 else None,
    "effect_estimate": float(m.prsquared),
    "significant": True if len(sig_vars) >= 2 else False,
})
for hid, var in [("h25_idh1_mv","idh1_mutation"),
                 ("h25_alb_mv","albumin_g_dl"),
                 ("h25_age_mv","age_years")]:
    analyses.append({
        "hypothesis_ids": [hid],
        "code": "final multivariable logit",
        "result_summary": f"{var} adjusted beta={m.params[var]:.4g}, p={m.pvalues[var]:.4g}",
        "p_value": float(m.pvalues[var]), "effect_estimate": float(m.params[var]),
        "significant": bool(m.pvalues[var] < 0.05),
    })
add_iter(25, hyps, analyses)

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("DONE; total iterations:", len(results["iterations"]))
print("total analyses:", sum(len(it["analyses"]) for it in results["iterations"]))
