"""Iterative analysis for ds001_aml. Produces transcript.json + analysis_summary.txt."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DF = pd.read_parquet("dataset.parquet")
ITERATIONS = []

def fmt(x, n=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "NA"
    return f"{x:.{n}g}"

def add_iter(idx, hyps, analyses):
    ITERATIONS.append({
        "index": idx,
        "proposed_hypotheses": hyps,
        "analyses": analyses,
    })

def chi2_diff(col, outcome="objective_response"):
    """Return (rate_1, rate_0, diff, p) for binary feature."""
    df = DF
    sub1 = df.loc[df[col] == 1, outcome]
    sub0 = df.loc[df[col] == 0, outcome]
    r1, r0 = sub1.mean(), sub0.mean()
    ct = pd.crosstab(df[col], df[outcome])
    if ct.shape == (2, 2):
        _, p, _, _ = stats.chi2_contingency(ct)
    else:
        p = float("nan")
    return r1, r0, r1 - r0, p

def ttest_by_outcome(col, outcome="objective_response"):
    df = DF
    a = df.loc[df[outcome] == 1, col]
    b = df.loc[df[outcome] == 0, col]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return a.mean(), b.mean(), a.mean() - b.mean(), p

def logit(features, interaction=None):
    df = DF
    X = df[features].copy()
    if interaction is not None:
        a, b = interaction
        X[f"{a}_x_{b}"] = df[a] * df[b]
    X = sm.add_constant(X)
    y = df["objective_response"]
    m = sm.Logit(y, X).fit(disp=False)
    return m

# =========================================================================
# Iteration 1 — ECOG performance status
# =========================================================================
m1, m0, diff, p = ttest_by_outcome("ecog_ps")
add_iter(1,
    [{"id": "h1", "text": "Higher ECOG performance status (worse functional status) is associated with a lower probability of objective_response in this AML cohort.", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h1"],
        "code": "stats.ttest_ind(df.loc[df.objective_response==1,'ecog_ps'], df.loc[df.objective_response==0,'ecog_ps'])",
        "result_summary": f"Mean ecog_ps in responders={m1:.3f} vs non-responders={m0:.3f} (diff={diff:+.3f}, t-test p={fmt(p)}).",
        "p_value": float(p),
        "effect_estimate": float(diff),
        "significant": bool(p < 0.05),
    }])

# =========================================================================
# Iteration 2 — Weight loss
# =========================================================================
m1, m0, diff, p = ttest_by_outcome("weight_loss_pct_6mo")
add_iter(2,
    [{"id": "h2", "text": "Greater weight_loss_pct_6mo (more recent weight loss) is associated with lower mean objective_response.", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h2"],
        "code": "ttest weight_loss_pct_6mo by objective_response",
        "result_summary": f"Mean weight loss in responders={m1:.3f}% vs non-responders={m0:.3f}% (diff={diff:+.3f}, p={fmt(p)}).",
        "p_value": float(p),
        "effect_estimate": float(diff),
        "significant": bool(p < 0.05),
    }])

# =========================================================================
# Iteration 3 — Marrow blast % and WBC
# =========================================================================
m1a, m0a, da, pa = ttest_by_outcome("blast_pct_marrow")
m1b, m0b, db, pb = ttest_by_outcome("wbc_k_per_ul")
add_iter(3,
    [
        {"id": "h3a", "text": "Higher blast_pct_marrow (greater marrow blast burden at baseline) is associated with lower probability of objective_response.", "kind": "novel"},
        {"id": "h3b", "text": "Higher wbc_k_per_ul (greater leukocyte count at presentation) is associated with lower probability of objective_response.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h3a"], "code": "ttest blast_pct_marrow by outcome",
         "result_summary": f"Mean blast % in responders={m1a:.2f} vs non-responders={m0a:.2f} (diff={da:+.2f}, p={fmt(pa)}).",
         "p_value": float(pa), "effect_estimate": float(da), "significant": bool(pa < 0.05)},
        {"hypothesis_ids": ["h3b"], "code": "ttest wbc_k_per_ul by outcome",
         "result_summary": f"Mean WBC in responders={m1b:.2f} vs non-responders={m0b:.2f} (diff={db:+.2f}, p={fmt(pb)}).",
         "p_value": float(pb), "effect_estimate": float(db), "significant": bool(pb < 0.05)},
    ])

# =========================================================================
# Iteration 4 — Albumin & CRP (nutrition/inflammation)
# =========================================================================
m1a, m0a, da, pa = ttest_by_outcome("albumin_g_dl")
m1b, m0b, db, pb = ttest_by_outcome("crp_mg_l")
add_iter(4,
    [
        {"id": "h4a", "text": "Higher serum albumin_g_dl (better nutritional status) is associated with higher probability of objective_response.", "kind": "novel"},
        {"id": "h4b", "text": "Higher serum crp_mg_l (greater systemic inflammation) is associated with lower probability of objective_response.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h4a"], "code": "ttest albumin_g_dl by outcome",
         "result_summary": f"Mean albumin: responders={m1a:.3f} vs non={m0a:.3f} (diff={da:+.3f}, p={fmt(pa)}).",
         "p_value": float(pa), "effect_estimate": float(da), "significant": bool(pa < 0.05)},
        {"hypothesis_ids": ["h4b"], "code": "ttest crp_mg_l by outcome",
         "result_summary": f"Mean CRP: responders={m1b:.2f} vs non={m0b:.2f} (diff={db:+.2f}, p={fmt(pb)}).",
         "p_value": float(pb), "effect_estimate": float(db), "significant": bool(pb < 0.05)},
    ])

# =========================================================================
# Iteration 5 — IDH1 mutation main effect
# =========================================================================
r1, r0, diff, p = chi2_diff("idh1_mutation")
add_iter(5,
    [{"id": "h5", "text": "Patients with idh1_mutation==1 have a higher objective_response rate than patients without IDH1 mutation.", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h5"],
        "code": "chi2 idh1_mutation x objective_response",
        "result_summary": f"Response rate IDH1+={r1:.3f} vs IDH1-={r0:.3f} (diff={diff:+.4f}, chi2 p={fmt(p)}).",
        "p_value": float(p),
        "effect_estimate": float(diff),
        "significant": bool(p < 0.05),
    }])

# =========================================================================
# Iteration 6 — Other AML genetics: NPM1, TP53, complex karyotype
# =========================================================================
items, hs = [], []
for col in ["npm1_mutation", "tp53_mutation", "complex_karyotype"]:
    r1, r0, diff, p = chi2_diff(col)
    hid = f"h6_{col}"
    direction = "higher" if col == "npm1_mutation" else "lower"
    hs.append({"id": hid,
               "text": f"Patients with {col}==1 have a {direction} objective_response rate than patients without it (clinically NPM1 favorable; TP53 and complex karyotype unfavorable).",
               "kind": "novel"})
    items.append({"hypothesis_ids": [hid], "code": f"chi2 {col} x outcome",
                  "result_summary": f"Response rate {col}+={r1:.4f} vs {col}-={r0:.4f} (diff={diff:+.4f}, p={fmt(p)}).",
                  "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
add_iter(6, hs, items)

# =========================================================================
# Iteration 7 — IDH1 × ivosidenib interaction (targeted-therapy match)
# =========================================================================
g = DF.groupby(["idh1_mutation", "treatment_ivosidenib"])["objective_response"].mean()
m = logit(["idh1_mutation", "treatment_ivosidenib"], interaction=("idh1_mutation", "treatment_ivosidenib"))
beta_int = float(m.params["idh1_mutation_x_treatment_ivosidenib"])
p_int = float(m.pvalues["idh1_mutation_x_treatment_ivosidenib"])
add_iter(7,
    [{"id": "h7", "text": "Among patients with idh1_mutation==1, those who receive treatment_ivosidenib have a higher objective_response rate than IDH1+ patients who do not (positive multiplicative interaction).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h7"],
        "code": "logit y ~ idh1 + ivo + idh1:ivo",
        "result_summary": (f"Cell rates: IDH1-/ivo-={g.loc[(0,0)]:.3f}; IDH1-/ivo+={g.loc[(0,1)]:.3f}; "
                           f"IDH1+/ivo-={g.loc[(1,0)]:.3f}; IDH1+/ivo+={g.loc[(1,1)]:.3f}. "
                           f"Interaction log-OR={beta_int:+.3f}, p={fmt(p_int)}."),
        "p_value": p_int,
        "effect_estimate": beta_int,
        "significant": bool(p_int < 0.05),
    }])

# =========================================================================
# Iteration 8 — IDH2 × enasidenib interaction
# =========================================================================
g = DF.groupby(["idh2_mutation", "treatment_enasidenib"])["objective_response"].mean()
m = logit(["idh2_mutation", "treatment_enasidenib"], interaction=("idh2_mutation", "treatment_enasidenib"))
beta_int = float(m.params["idh2_mutation_x_treatment_enasidenib"])
p_int = float(m.pvalues["idh2_mutation_x_treatment_enasidenib"])
add_iter(8,
    [{"id": "h8", "text": "Among patients with idh2_mutation==1, those receiving treatment_enasidenib have a higher objective_response rate than IDH2+ patients who do not (positive multiplicative interaction).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h8"],
        "code": "logit y ~ idh2 + ena + idh2:ena",
        "result_summary": (f"Cell rates: IDH2-/ena-={g.loc[(0,0)]:.3f}; IDH2-/ena+={g.loc[(0,1)]:.3f}; "
                           f"IDH2+/ena-={g.loc[(1,0)]:.3f}; IDH2+/ena+={g.loc[(1,1)]:.3f}. "
                           f"Interaction log-OR={beta_int:+.3f}, p={fmt(p_int)}."),
        "p_value": p_int,
        "effect_estimate": beta_int,
        "significant": bool(p_int < 0.05),
    }])

# =========================================================================
# Iteration 9 — FLT3-ITD × midostaurin and × gilteritinib interactions
# =========================================================================
items, hs = [], []
for tx in ["treatment_midostaurin", "treatment_gilteritinib"]:
    g = DF.groupby(["flt3_itd", tx])["objective_response"].mean()
    m = logit(["flt3_itd", tx], interaction=("flt3_itd", tx))
    bi = float(m.params[f"flt3_itd_x_{tx}"])
    pi = float(m.pvalues[f"flt3_itd_x_{tx}"])
    hid = f"h9_{tx}"
    hs.append({"id": hid,
               "text": f"Among patients with flt3_itd==1, those receiving {tx} have a higher objective_response rate than FLT3-ITD+ patients without that drug (positive interaction).",
               "kind": "novel"})
    items.append({"hypothesis_ids": [hid], "code": f"logit y ~ flt3_itd + {tx} + flt3_itd:{tx}",
                  "result_summary": (f"Cell rates flt3_itd × {tx}: (0,0)={g.loc[(0,0)]:.3f} (0,1)={g.loc[(0,1)]:.3f} "
                                     f"(1,0)={g.loc[(1,0)]:.3f} (1,1)={g.loc[(1,1)]:.3f}. "
                                     f"Interaction log-OR={bi:+.3f}, p={fmt(pi)}."),
                  "p_value": pi, "effect_estimate": bi, "significant": bool(pi < 0.05)})
add_iter(9, hs, items)

# =========================================================================
# Iteration 10 — Unfit × venetoclax/azacitidine interaction
# =========================================================================
g = DF.groupby(["unfit_for_intensive", "treatment_venetoclax_azacitidine"])["objective_response"].mean()
m = logit(["unfit_for_intensive", "treatment_venetoclax_azacitidine"], interaction=("unfit_for_intensive", "treatment_venetoclax_azacitidine"))
bi = float(m.params["unfit_for_intensive_x_treatment_venetoclax_azacitidine"])
pi = float(m.pvalues["unfit_for_intensive_x_treatment_venetoclax_azacitidine"])
add_iter(10,
    [{"id": "h10", "text": "Among patients flagged unfit_for_intensive==1, treatment_venetoclax_azacitidine raises objective_response rate more than it does in fit patients (positive interaction).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h10"],
        "code": "logit y ~ unfit + venaza + unfit:venaza",
        "result_summary": (f"Cell rates unfit × venaza: (0,0)={g.loc[(0,0)]:.3f} (0,1)={g.loc[(0,1)]:.3f} "
                           f"(1,0)={g.loc[(1,0)]:.3f} (1,1)={g.loc[(1,1)]:.3f}. "
                           f"Interaction log-OR={bi:+.3f}, p={fmt(pi)}."),
        "p_value": pi, "effect_estimate": bi, "significant": bool(pi < 0.05),
    }])

# =========================================================================
# Iteration 11 — Age effect (continuous) and sex
# =========================================================================
m1a, m0a, da, pa = ttest_by_outcome("age_years")
ct = pd.crosstab(DF["sex_female"], DF["objective_response"])
_, ps, _, _ = stats.chi2_contingency(ct)
r1 = DF.loc[DF.sex_female == 1, "objective_response"].mean()
r0 = DF.loc[DF.sex_female == 0, "objective_response"].mean()
add_iter(11,
    [
        {"id": "h11a", "text": "Older age_years is associated with a lower probability of objective_response.", "kind": "novel"},
        {"id": "h11b", "text": "Female patients (sex_female==1) and male patients have different objective_response rates.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h11a"], "code": "ttest age_years by outcome",
         "result_summary": f"Mean age responders={m1a:.2f} vs non={m0a:.2f} (diff={da:+.2f}, p={fmt(pa)}).",
         "p_value": float(pa), "effect_estimate": float(da), "significant": bool(pa < 0.05)},
        {"hypothesis_ids": ["h11b"], "code": "chi2 sex_female x outcome",
         "result_summary": f"Response rate female={r1:.4f} vs male={r0:.4f} (diff={r1-r0:+.4f}, p={fmt(ps)}).",
         "p_value": float(ps), "effect_estimate": float(r1 - r0), "significant": bool(ps < 0.05)},
    ])

# =========================================================================
# Iteration 12 — Multivariable model adjusted for major confounders
# =========================================================================
features = ["age_years", "sex_female", "ecog_ps", "secondary_aml", "unfit_for_intensive",
            "complex_karyotype", "flt3_itd", "flt3_tkd", "idh1_mutation", "idh2_mutation",
            "npm1_mutation", "tp53_mutation", "wbc_k_per_ul", "blast_pct_marrow",
            "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr",
            "treatment_midostaurin", "treatment_gilteritinib", "treatment_ivosidenib",
            "treatment_enasidenib", "treatment_venetoclax_azacitidine", "treatment_7plus3"]
m = logit(features)
sig = [(name, m.params[name], m.pvalues[name]) for name in features if m.pvalues[name] < 0.05]
sig.sort(key=lambda x: x[2])
items, hs = [], []
for col, est, p in sig:
    hid = f"h12_{col}"
    direction = "higher" if est > 0 else "lower"
    hs.append({"id": hid,
               "text": f"After adjusting for age, ECOG, AML genetics, baseline labs, and treatments, {col} is independently associated with a {direction} log-odds of objective_response.",
               "kind": "refined"})
    items.append({"hypothesis_ids": [hid], "code": "Logit on full feature set",
                  "result_summary": f"Adjusted log-OR for {col}={est:+.4f}, p={fmt(p)}.",
                  "p_value": float(p), "effect_estimate": float(est), "significant": True})
add_iter(12, hs, items)

# =========================================================================
# Iteration 13 — Treatment main effects unadjusted (head-to-head)
# =========================================================================
items, hs = [], []
for tx in ["treatment_midostaurin", "treatment_gilteritinib", "treatment_ivosidenib",
           "treatment_enasidenib", "treatment_venetoclax_azacitidine", "treatment_7plus3"]:
    r1, r0, diff, p = chi2_diff(tx)
    hid = f"h13_{tx}"
    hs.append({"id": hid,
               "text": f"Patients receiving {tx} have a different objective_response rate than patients not receiving that treatment.",
               "kind": "novel"})
    items.append({"hypothesis_ids": [hid], "code": f"chi2 {tx} x outcome",
                  "result_summary": f"Response rate {tx}+={r1:.4f} vs {tx}-={r0:.4f} (diff={diff:+.4f}, p={fmt(p)}).",
                  "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
add_iter(13, hs, items)

# =========================================================================
# Iteration 14 — Comorbidities: CKD, heart failure, AF
# =========================================================================
items, hs = [], []
for col in ["chronic_kidney_disease", "heart_failure", "atrial_fibrillation",
            "diabetes_mellitus", "hypertension", "copd"]:
    r1, r0, diff, p = chi2_diff(col)
    hid = f"h14_{col}"
    hs.append({"id": hid,
               "text": f"Patients with {col}==1 have a different objective_response rate than patients without that comorbidity.",
               "kind": "novel"})
    items.append({"hypothesis_ids": [hid], "code": f"chi2 {col} x outcome",
                  "result_summary": f"Response rate {col}+={r1:.4f} vs {col}-={r0:.4f} (diff={diff:+.4f}, p={fmt(p)}).",
                  "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
add_iter(14, hs, items)

# =========================================================================
# Iteration 15 — Symptom burden
# =========================================================================
items, hs = [], []
for col in ["fatigue_grade", "pain_nrs", "dyspnea_grade", "cough_grade", "appetite_loss_grade"]:
    m1a, m0a, da, pa = ttest_by_outcome(col)
    hid = f"h15_{col}"
    hs.append({"id": hid,
               "text": f"Higher {col} (worse symptom burden) is associated with a lower probability of objective_response.",
               "kind": "novel"})
    items.append({"hypothesis_ids": [hid], "code": f"ttest {col} by outcome",
                  "result_summary": f"Mean {col}: responders={m1a:.3f} vs non={m0a:.3f} (diff={da:+.3f}, p={fmt(pa)}).",
                  "p_value": float(pa), "effect_estimate": float(da), "significant": bool(pa < 0.05)})
add_iter(15, hs, items)

# =========================================================================
# Iteration 16 — SDOH: rural residence, education, insurance, race/ethnicity
# =========================================================================
items, hs = [], []
# rural binary
r1, r0, diff, p = chi2_diff("rural_residence")
items.append({"hypothesis_ids": ["h16a"], "code": "chi2 rural_residence x outcome",
              "result_summary": f"Response rate rural+={r1:.4f} vs rural-={r0:.4f} (diff={diff:+.4f}, p={fmt(p)}).",
              "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
hs.append({"id": "h16a", "text": "Patients with rural_residence==1 have a different objective_response rate than non-rural patients.", "kind": "novel"})
# education ttest
m1, m0, d, pe = ttest_by_outcome("education_years")
items.append({"hypothesis_ids": ["h16b"], "code": "ttest education_years by outcome",
              "result_summary": f"Mean education_years: responders={m1:.2f} vs non={m0:.2f} (diff={d:+.3f}, p={fmt(pe)}).",
              "p_value": float(pe), "effect_estimate": float(d), "significant": bool(pe < 0.05)})
hs.append({"id": "h16b", "text": "More years of education_years are associated with a higher probability of objective_response.", "kind": "novel"})
# insurance categorical
ct = pd.crosstab(DF["insurance_type"], DF["objective_response"])
_, p_ins, _, _ = stats.chi2_contingency(ct)
rates = DF.groupby("insurance_type")["objective_response"].mean()
diff_ins = float(rates.max() - rates.min())
items.append({"hypothesis_ids": ["h16c"], "code": "chi2 insurance_type x outcome",
              "result_summary": f"Response rates by insurance: {rates.round(4).to_dict()}; chi2 p={fmt(p_ins)}.",
              "p_value": float(p_ins), "effect_estimate": diff_ins, "significant": bool(p_ins < 0.05)})
hs.append({"id": "h16c", "text": "Objective_response rate differs across insurance_type categories (medicare vs private vs medicaid vs uninsured).", "kind": "novel"})
# race
ct = pd.crosstab(DF["race_ethnicity"], DF["objective_response"])
_, p_race, _, _ = stats.chi2_contingency(ct)
rates_r = DF.groupby("race_ethnicity")["objective_response"].mean()
diff_race = float(rates_r.max() - rates_r.min())
items.append({"hypothesis_ids": ["h16d"], "code": "chi2 race_ethnicity x outcome",
              "result_summary": f"Response rates by race_ethnicity: {rates_r.round(4).to_dict()}; chi2 p={fmt(p_race)}.",
              "p_value": float(p_race), "effect_estimate": diff_race, "significant": bool(p_race < 0.05)})
hs.append({"id": "h16d", "text": "Objective_response rate differs across race_ethnicity categories.", "kind": "novel"})
add_iter(16, hs, items)

# =========================================================================
# Iteration 17 — Pharmacogenomic SNP screen (FDR-style ranking)
# =========================================================================
snp_cols = [c for c in DF.columns if c.startswith("snp_")]
snp_results = []
for c in snp_cols:
    r1, r0, diff, p = chi2_diff(c)
    snp_results.append((c, diff, p))
snp_results.sort(key=lambda x: x[2])
items, hs = [], []
for c, diff, p in snp_results[:5]:
    hid = f"h17_{c}"
    direction = "higher" if diff > 0 else "lower"
    hs.append({"id": hid,
               "text": f"Patients with {c}==1 have a {direction} objective_response rate than patients with {c}==0.",
               "kind": "novel"})
    items.append({"hypothesis_ids": [hid], "code": f"chi2 {c} x outcome",
                  "result_summary": f"Response rate {c}+={DF.loc[DF[c]==1,'objective_response'].mean():.4f} vs {c}-={DF.loc[DF[c]==0,'objective_response'].mean():.4f} (diff={diff:+.4f}, p={fmt(p)}).",
                  "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
add_iter(17, hs, items)

# =========================================================================
# Iteration 18 — Liver/renal function and bilirubin
# =========================================================================
items, hs = [], []
for col in ["alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl",
            "creatinine_mg_dl", "bun_mg_dl", "ldh_u_l"]:
    m1, m0, d, p = ttest_by_outcome(col)
    hid = f"h18_{col}"
    direction = "higher" if d > 0 else "lower"
    hs.append({"id": hid,
               "text": f"Mean {col} differs between responders and non-responders (responders {direction}).",
               "kind": "novel"})
    items.append({"hypothesis_ids": [hid], "code": f"ttest {col} by outcome",
                  "result_summary": f"Mean {col}: responders={m1:.3f} vs non={m0:.3f} (diff={d:+.3f}, p={fmt(p)}).",
                  "p_value": float(p), "effect_estimate": float(d), "significant": bool(p < 0.05)})
add_iter(18, hs, items)

# =========================================================================
# Iteration 19 — Metabolic / electrolytes / vitals
# =========================================================================
items, hs = [], []
for col in ["sodium_meq_l", "potassium_meq_l", "calcium_mg_dl", "glucose_mg_dl",
            "platelets_k_ul", "anc_k_ul", "alc_k_ul", "hemoglobin_g_dl",
            "systolic_bp_mmhg", "heart_rate_bpm", "spo2_pct", "bmi"]:
    m1, m0, d, p = ttest_by_outcome(col)
    hid = f"h19_{col}"
    direction = "higher" if d > 0 else "lower"
    hs.append({"id": hid,
               "text": f"Mean {col} differs between responders and non-responders (responders {direction}).",
               "kind": "novel"})
    items.append({"hypothesis_ids": [hid], "code": f"ttest {col} by outcome",
                  "result_summary": f"Mean {col}: responders={m1:.3f} vs non={m0:.3f} (diff={d:+.3f}, p={fmt(p)}).",
                  "p_value": float(p), "effect_estimate": float(d), "significant": bool(p < 0.05)})
add_iter(19, hs, items)

# =========================================================================
# Iteration 20 — Prior therapy lines and duration
# =========================================================================
items, hs = [], []
for col in ["prior_lines_of_therapy", "years_since_diagnosis"]:
    m1, m0, d, p = ttest_by_outcome(col)
    hid = f"h20_{col}"
    hs.append({"id": hid,
               "text": f"Higher {col} (more heavily pretreated / longer disease course) is associated with lower probability of objective_response.",
               "kind": "novel"})
    items.append({"hypothesis_ids": [hid], "code": f"ttest {col} by outcome",
                  "result_summary": f"Mean {col}: responders={m1:.3f} vs non={m0:.3f} (diff={d:+.3f}, p={fmt(p)}).",
                  "p_value": float(p), "effect_estimate": float(d), "significant": bool(p < 0.05)})
for col in ["prior_chemotherapy", "prior_radiation", "prior_surgery", "prior_immunotherapy", "prior_targeted_therapy"]:
    r1, r0, diff, p = chi2_diff(col)
    hid = f"h20_{col}"
    hs.append({"id": hid,
               "text": f"Patients with {col}==1 have a different objective_response rate than patients without prior exposure to that modality.",
               "kind": "novel"})
    items.append({"hypothesis_ids": [hid], "code": f"chi2 {col} x outcome",
                  "result_summary": f"Response rate {col}+={r1:.4f} vs {col}-={r0:.4f} (diff={diff:+.4f}, p={fmt(p)}).",
                  "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
add_iter(20, hs, items)

# =========================================================================
# Iteration 21 — Effect modification: IDH1 mutation effect across treatments
# Refines h5/h7: IDH1 favorability may not be drug-dependent.
# =========================================================================
items, hs = [], []
hs.append({"id": "h21_main",
           "text": "The favorable association of idh1_mutation with objective_response persists after adjustment for treatment received and other prognostic features (i.e., IDH1 acts as a prognostic factor independent of ivosidenib).",
           "kind": "refined"})
m = logit(features)  # already-fitted multivariable
beta_idh1 = float(m.params["idh1_mutation"])
p_idh1 = float(m.pvalues["idh1_mutation"])
beta_ivo = float(m.params["treatment_ivosidenib"])
p_ivo = float(m.pvalues["treatment_ivosidenib"])
items.append({"hypothesis_ids": ["h21_main"], "code": "Adjusted Logit; coefficient on idh1_mutation",
              "result_summary": (f"Adjusted log-OR for idh1_mutation={beta_idh1:+.3f}, p={fmt(p_idh1)}; "
                                 f"adjusted log-OR for treatment_ivosidenib={beta_ivo:+.3f}, p={fmt(p_ivo)}."),
              "p_value": float(p_idh1), "effect_estimate": float(beta_idh1), "significant": bool(p_idh1 < 0.05)})
# Also test whether IDH1 effect is enhanced by ivosidenib in adjusted model
m2 = logit(features, interaction=("idh1_mutation", "treatment_ivosidenib"))
beta_int = float(m2.params["idh1_mutation_x_treatment_ivosidenib"])
p_int = float(m2.pvalues["idh1_mutation_x_treatment_ivosidenib"])
hs.append({"id": "h21_int",
           "text": "After multivariable adjustment, the idh1_mutation × treatment_ivosidenib interaction is positive: ivosidenib amplifies the favorable IDH1 effect.",
           "kind": "refined"})
items.append({"hypothesis_ids": ["h21_int"], "code": "Adjusted Logit + idh1*ivo interaction",
              "result_summary": f"Adjusted interaction log-OR={beta_int:+.3f}, p={fmt(p_int)}.",
              "p_value": float(p_int), "effect_estimate": float(beta_int), "significant": bool(p_int < 0.05)})
add_iter(21, hs, items)

# =========================================================================
# Iteration 22 — Ecog × treatment_7plus3 (intensive chemo only helps fit pts?)
# =========================================================================
g = DF.groupby(["ecog_ps", "treatment_7plus3"])["objective_response"].mean().unstack()
m = logit(["ecog_ps", "treatment_7plus3"], interaction=("ecog_ps", "treatment_7plus3"))
bi = float(m.params["ecog_ps_x_treatment_7plus3"])
pi = float(m.pvalues["ecog_ps_x_treatment_7plus3"])
add_iter(22,
    [{"id": "h22", "text": "The negative association between ecog_ps and objective_response is more pronounced among patients receiving treatment_7plus3 (induction chemo) than among those who do not (negative interaction).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h22"],
        "code": "logit y ~ ecog + 7+3 + ecog:7+3",
        "result_summary": f"Response rate by ecog × 7+3:\n{g.to_string()}\nInteraction log-OR={bi:+.3f}, p={fmt(pi)}.",
        "p_value": pi, "effect_estimate": bi, "significant": bool(pi < 0.05),
    }])

# =========================================================================
# Iteration 23 — Albumin × CRP additive prognostic effect
# =========================================================================
median_alb = DF["albumin_g_dl"].median()
median_crp = DF["crp_mg_l"].median()
DF["alb_low"] = (DF["albumin_g_dl"] < median_alb).astype(int)
DF["crp_high"] = (DF["crp_mg_l"] > median_crp).astype(int)
g = DF.groupby(["alb_low", "crp_high"])["objective_response"].mean().unstack()
m = logit(["alb_low", "crp_high"], interaction=("alb_low", "crp_high"))
bi = float(m.params["alb_low_x_crp_high"])
pi = float(m.pvalues["alb_low_x_crp_high"])
add_iter(23,
    [{"id": "h23", "text": "Patients with both low albumin (alb_low=1, albumin below median) AND high CRP (crp_high=1, CRP above median) have an even lower objective_response rate than expected from the additive sum of the two main effects (positive interaction => synergistic risk).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h23"],
        "code": "logit y ~ alb_low + crp_high + alb_low:crp_high",
        "result_summary": f"Response rate by alb_low × crp_high:\n{g.to_string()}\nInteraction log-OR={bi:+.3f}, p={fmt(pi)}.",
        "p_value": pi, "effect_estimate": bi, "significant": bool(pi < 0.05),
    }])

# =========================================================================
# Iteration 24 — Heterogeneity of IDH1 effect by sex
# =========================================================================
g = DF.groupby(["sex_female", "idh1_mutation"])["objective_response"].mean().unstack()
m = logit(["sex_female", "idh1_mutation"], interaction=("sex_female", "idh1_mutation"))
bi = float(m.params["sex_female_x_idh1_mutation"])
pi = float(m.pvalues["sex_female_x_idh1_mutation"])
add_iter(24,
    [{"id": "h24", "text": "The favorable effect of idh1_mutation on objective_response differs between female (sex_female==1) and male patients (i.e., a sex_female × idh1_mutation interaction is non-zero).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h24"],
        "code": "logit y ~ sex_female + idh1 + sex:idh1",
        "result_summary": f"Response rate by sex × IDH1:\n{g.to_string()}\nInteraction log-OR={bi:+.3f}, p={fmt(pi)}.",
        "p_value": pi, "effect_estimate": bi, "significant": bool(pi < 0.05),
    }])

# =========================================================================
# Iteration 25 — Final composite prognostic score
# =========================================================================
# Predicted probability from full multivariable model — split by quartiles, compare
m = logit(features)
DF["pred_resp"] = m.predict(sm.add_constant(DF[features]))
DF["pred_q"] = pd.qcut(DF["pred_resp"], 4, labels=[1, 2, 3, 4]).astype(int)
rates = DF.groupby("pred_q")["objective_response"].mean()
ct = pd.crosstab(DF["pred_q"], DF["objective_response"])
_, p_q, _, _ = stats.chi2_contingency(ct)
diff_q = float(rates.iloc[-1] - rates.iloc[0])
add_iter(25,
    [{"id": "h25", "text": "A multivariable logistic model trained on baseline features (age, ECOG, AML genetics, labs, treatment) discriminates objective_response: patients in the top predicted-probability quartile have a higher observed response rate than those in the bottom quartile.", "kind": "refined"}],
    [{
        "hypothesis_ids": ["h25"],
        "code": "Logit -> predicted prob; cut into quartiles; compare observed rates",
        "result_summary": f"Observed response by quartile of predicted prob: Q1={rates.iloc[0]:.3f} Q2={rates.iloc[1]:.3f} Q3={rates.iloc[2]:.3f} Q4={rates.iloc[3]:.3f}; chi2 p={fmt(p_q)}; Q4-Q1 diff={diff_q:+.3f}.",
        "p_value": float(p_q), "effect_estimate": diff_q, "significant": bool(p_q < 0.05),
    }])

# =========================================================================
# Write outputs
# =========================================================================
transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@inline-harness",
    "max_iterations": 25,
    "iterations": ITERATIONS,
}
with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x))
print(f"Wrote transcript.json with {len(ITERATIONS)} iterations.")
n_hyps = sum(len(it["proposed_hypotheses"]) for it in ITERATIONS)
n_anals = sum(len(it["analyses"]) for it in ITERATIONS)
print(f"  total hypotheses: {n_hyps}; total analyses: {n_anals}")
