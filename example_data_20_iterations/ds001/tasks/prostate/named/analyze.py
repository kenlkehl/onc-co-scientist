"""Comprehensive analysis of ds001_prostate -> builds transcript.json + analysis_summary.txt."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
N = len(df)
print(f"Loaded {N} rows, {df.shape[1]} cols")

iterations = []  # list of {index, proposed_hypotheses, analyses}


def add_iter(index, hyps, analyses):
    iterations.append({"index": index, "proposed_hypotheses": hyps, "analyses": analyses})


def ttest_by_group(col, group):
    a = df.loc[df[group] == 1, col].astype(float)
    b = df.loc[df[group] == 0, col].astype(float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(p), float(a.mean()), float(b.mean()), len(a), len(b)


def linreg(formula):
    return smf.ols(formula, data=df).fit()


# ------------------------------------------------------------------
# Iteration 1: Main effects of each treatment on PFS
# ------------------------------------------------------------------
hyps1 = []
ans1 = []
treatments = [
    "treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
    "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab",
]
for i, t in enumerate(treatments, 1):
    hid = f"h1.{i}"
    hyps1.append({"id": hid,
                  "text": f"Patients receiving {t}=1 have different mean pfs_months than patients with {t}=0 (overall, unadjusted)."})
    diff, p, m1, m0, n1, n0 = ttest_by_group("pfs_months", t)
    ans1.append({
        "hypothesis_ids": [hid],
        "code": f"stats.ttest_ind(df[df['{t}']==1].pfs_months, df[df['{t}']==0].pfs_months, equal_var=False)",
        "result_summary": f"Mean PFS {m1:.3f} (n={n1}) on {t} vs {m0:.3f} (n={n0}) off; Welch t-test p={p:.3g}.",
        "p_value": p,
        "effect_estimate": diff,
        "significant": bool(p < 0.05),
    })
add_iter(1, hyps1, ans1)


# ------------------------------------------------------------------
# Iteration 2: Disease-state and burden biomarkers (binary)
# ------------------------------------------------------------------
hyps2 = []
ans2 = []
bin_features = [
    ("mcrpc", "lower"), ("visceral_mets", "lower"),
    ("liver_mets", "lower"), ("bone_mets", "lower"), ("adrenal_mets", "lower"),
    ("pleural_effusion", "lower"), ("pericardial_effusion", "lower"),
]
for i, (b, direction) in enumerate(bin_features, 1):
    hid = f"h2.{i}"
    hyps2.append({"id": hid,
                  "text": f"Patients with {b}=1 have {direction} mean pfs_months than patients with {b}=0."})
    diff, p, m1, m0, n1, n0 = ttest_by_group("pfs_months", b)
    ans2.append({
        "hypothesis_ids": [hid],
        "code": f"stats.ttest_ind(df[df['{b}']==1].pfs_months, df[df['{b}']==0].pfs_months, equal_var=False)",
        "result_summary": f"Mean PFS {m1:.3f} (n={n1}) with {b} vs {m0:.3f} (n={n0}) without; Welch p={p:.3g}.",
        "p_value": p,
        "effect_estimate": diff,
        "significant": bool(p < 0.05),
    })
add_iter(2, hyps2, ans2)


# ------------------------------------------------------------------
# Iteration 3: Molecular biomarkers (binary)
# ------------------------------------------------------------------
hyps3 = []
ans3 = []
mol = [
    ("brca2_mutation", "higher"), ("ar_v7_positive", "lower"),
    ("msi_high", "higher"), ("psma_high", "higher"),
    ("tp53_mutation", "lower"), ("pten_loss", "lower"),
    ("her2_amplification", "either"), ("braf_v600e", "either"),
]
for i, (b, direction) in enumerate(mol, 1):
    hid = f"h3.{i}"
    hyps3.append({"id": hid,
                  "text": f"Patients with {b}=1 have {direction} mean pfs_months than patients with {b}=0."})
    diff, p, m1, m0, n1, n0 = ttest_by_group("pfs_months", b)
    ans3.append({
        "hypothesis_ids": [hid],
        "code": f"stats.ttest_ind(df[df['{b}']==1].pfs_months, df[df['{b}']==0].pfs_months, equal_var=False)",
        "result_summary": f"Mean PFS {m1:.3f} (n={n1}) with {b}=1 vs {m0:.3f} (n={n0}) with {b}=0; Welch p={p:.3g}.",
        "p_value": p,
        "effect_estimate": diff,
        "significant": bool(p < 0.05),
    })
add_iter(3, hyps3, ans3)


# ------------------------------------------------------------------
# Iteration 4: Continuous prognostic features vs PFS (Pearson)
# ------------------------------------------------------------------
hyps4 = []
ans4 = []
cont = [
    ("age_years", "negative"), ("ecog_ps", "negative"),
    ("psa_ng_ml", "negative"), ("gleason_score", "negative"),
    ("albumin_g_dl", "positive"), ("ldh_u_l", "negative"),
    ("hemoglobin_g_dl", "positive"), ("alkaline_phosphatase_u_l", "negative"),
    ("crp_mg_l", "negative"), ("nlr", "negative"),
    ("weight_loss_pct_6mo", "negative"),
]
for i, (c, direction) in enumerate(cont, 1):
    hid = f"h4.{i}"
    hyps4.append({"id": hid,
                  "text": f"{c} has a {direction} association with pfs_months (Pearson correlation)."})
    r, p = stats.pearsonr(df[c], df["pfs_months"])
    ans4.append({
        "hypothesis_ids": [hid],
        "code": f"stats.pearsonr(df['{c}'], df['pfs_months'])",
        "result_summary": f"Pearson r={r:.4f} between {c} and pfs_months (p={p:.3g}, n={N}).",
        "p_value": float(p),
        "effect_estimate": float(r),
        "significant": bool(p < 0.05),
    })
add_iter(4, hyps4, ans4)


# ------------------------------------------------------------------
# Iteration 5: Refine — treatment × biomarker interactions (predicted matches)
# ------------------------------------------------------------------
hyps5 = []
ans5 = []
interactions = [
    ("treatment_olaparib", "brca2_mutation",
     "Olaparib benefit (additional pfs_months) is greater in BRCA2-mutated patients than in BRCA2-wildtype patients (positive interaction)."),
    ("treatment_pembrolizumab", "msi_high",
     "Pembrolizumab benefit on pfs_months is greater in MSI-high patients than MSI-low (positive interaction)."),
    ("treatment_lu177_psma", "psma_high",
     "Lu177-PSMA benefit on pfs_months is greater in PSMA-high patients (positive interaction)."),
    ("treatment_enzalutamide", "ar_v7_positive",
     "AR-V7 positivity attenuates or reverses enzalutamide benefit (negative interaction)."),
    ("treatment_abiraterone", "ar_v7_positive",
     "AR-V7 positivity attenuates or reverses abiraterone benefit (negative interaction)."),
    ("treatment_docetaxel", "visceral_mets",
     "Docetaxel benefit on pfs_months differs by visceral metastases status (interaction)."),
]
for i, (tx, mk, text) in enumerate(interactions, 1):
    hid = f"h5.{i}"
    hyps5.append({"id": hid, "text": text})
    formula = f"pfs_months ~ {tx} * {mk}"
    fit = linreg(formula)
    inter_name = f"{tx}:{mk}"
    coef = float(fit.params.get(inter_name, np.nan))
    p = float(fit.pvalues.get(inter_name, np.nan))
    main_tx = float(fit.params.get(tx, np.nan))
    main_mk = float(fit.params.get(mk, np.nan))
    ans5.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('{formula}', data=df).fit()",
        "result_summary": (f"OLS PFS ~ {tx}*{mk}: main {tx} {main_tx:+.3f}, main {mk} {main_mk:+.3f}, "
                            f"interaction {coef:+.3f} (p={p:.3g})."),
        "p_value": p,
        "effect_estimate": coef,
        "significant": bool(p < 0.05) if not np.isnan(p) else None,
    })
add_iter(5, hyps5, ans5)


# ------------------------------------------------------------------
# Iteration 6: Stratified subgroup PFS by treatment for predicted matches
# ------------------------------------------------------------------
hyps6 = []
ans6 = []
subg = [
    ("treatment_olaparib", "brca2_mutation"),
    ("treatment_pembrolizumab", "msi_high"),
    ("treatment_lu177_psma", "psma_high"),
    ("treatment_enzalutamide", "ar_v7_positive"),
    ("treatment_abiraterone", "ar_v7_positive"),
]
for i, (tx, mk) in enumerate(subg, 1):
    for j, mval in enumerate([1, 0]):
        hid = f"h6.{i}.{mval}"
        text = (f"Within {mk}={mval} patients, mean pfs_months differs between {tx}=1 and {tx}=0 "
                f"(direction expected to favor {tx} when biomarker matches predicted indication).")
        hyps6.append({"id": hid, "text": text})
        sub = df[df[mk] == mval]
        a = sub.loc[sub[tx] == 1, "pfs_months"]
        b = sub.loc[sub[tx] == 0, "pfs_months"]
        if len(a) < 5 or len(b) < 5:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        diff = float(a.mean() - b.mean())
        ans6.append({
            "hypothesis_ids": [hid],
            "code": f"sub=df[df['{mk}']=={mval}]; ttest(sub[sub['{tx}']==1].pfs, sub[sub['{tx}']==0].pfs)",
            "result_summary": (f"In {mk}={mval}: PFS {a.mean():.3f} (n={len(a)}) on {tx} vs "
                                f"{b.mean():.3f} (n={len(b)}) off; diff {diff:+.3f}, Welch p={p:.3g}."),
            "p_value": float(p),
            "effect_estimate": diff,
            "significant": bool(p < 0.05),
        })
add_iter(6, hyps6, ans6)


# ------------------------------------------------------------------
# Iteration 7: Comorbidities vs PFS
# ------------------------------------------------------------------
hyps7 = []
ans7 = []
comor = [
    "diabetes_mellitus", "hypertension", "copd", "chronic_kidney_disease",
    "heart_failure", "coronary_artery_disease", "atrial_fibrillation",
    "venous_thromboembolism_history", "autoimmune_disease",
    "hepatitis_b_history", "hepatitis_c_history", "hiv_positive",
    "prior_malignancy", "depression_anxiety_diagnosis",
]
for i, c in enumerate(comor, 1):
    hid = f"h7.{i}"
    hyps7.append({"id": hid, "text": f"{c}=1 is associated with shorter pfs_months than {c}=0."})
    diff, p, m1, m0, n1, n0 = ttest_by_group("pfs_months", c)
    ans7.append({
        "hypothesis_ids": [hid],
        "code": f"ttest by {c}",
        "result_summary": f"PFS {m1:.3f} (n={n1}) with {c} vs {m0:.3f} (n={n0}) without; p={p:.3g}.",
        "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
    })
add_iter(7, hyps7, ans7)


# ------------------------------------------------------------------
# Iteration 8: Symptom severity vs PFS (continuous)
# ------------------------------------------------------------------
hyps8 = []
ans8 = []
sym = ["fatigue_grade", "pain_nrs", "dyspnea_grade", "cough_grade", "appetite_loss_grade"]
for i, s in enumerate(sym, 1):
    hid = f"h8.{i}"
    hyps8.append({"id": hid, "text": f"Higher {s} predicts shorter pfs_months (negative correlation)."})
    r, p = stats.pearsonr(df[s], df["pfs_months"])
    ans8.append({
        "hypothesis_ids": [hid],
        "code": f"stats.pearsonr(df['{s}'], df['pfs_months'])",
        "result_summary": f"Pearson r={r:.4f} (p={p:.3g}, n={N}).",
        "p_value": float(p), "effect_estimate": float(r),
        "significant": bool(p < 0.05),
    })
add_iter(8, hyps8, ans8)


# ------------------------------------------------------------------
# Iteration 9: Demographics / SES — race, insurance, rural, education
# ------------------------------------------------------------------
hyps9 = []
ans9 = []
# Race ANOVA
hid = "h9.1"
hyps9.append({"id": hid, "text": "Mean pfs_months differs across race_ethnicity categories (one-way ANOVA)."})
groups = [df.loc[df.race_ethnicity == lvl, "pfs_months"] for lvl in df.race_ethnicity.unique()]
F, p = stats.f_oneway(*groups)
means = df.groupby("race_ethnicity")["pfs_months"].mean().to_dict()
ans9.append({
    "hypothesis_ids": [hid],
    "code": "f_oneway by race_ethnicity",
    "result_summary": f"One-way ANOVA F={F:.3f}, p={p:.3g}. Means: {means}.",
    "p_value": float(p), "effect_estimate": float(F),
    "significant": bool(p < 0.05),
})
# Insurance ANOVA
hid = "h9.2"
hyps9.append({"id": hid, "text": "Mean pfs_months differs across insurance_type categories."})
groups = [df.loc[df.insurance_type == lvl, "pfs_months"] for lvl in df.insurance_type.unique()]
F, p = stats.f_oneway(*groups)
means = df.groupby("insurance_type")["pfs_months"].mean().to_dict()
ans9.append({
    "hypothesis_ids": [hid],
    "code": "f_oneway by insurance_type",
    "result_summary": f"F={F:.3f}, p={p:.3g}. Means: {means}.",
    "p_value": float(p), "effect_estimate": float(F),
    "significant": bool(p < 0.05),
})
# Rural
hid = "h9.3"
hyps9.append({"id": hid, "text": "Rural residence (rural_residence=1) is associated with shorter pfs_months than non-rural."})
diff, p, m1, m0, n1, n0 = ttest_by_group("pfs_months", "rural_residence")
ans9.append({
    "hypothesis_ids": [hid],
    "code": "ttest rural_residence",
    "result_summary": f"PFS {m1:.3f} rural (n={n1}) vs {m0:.3f} non-rural (n={n0}); p={p:.3g}.",
    "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
})
# Education
hid = "h9.4"
hyps9.append({"id": hid, "text": "education_years is positively correlated with pfs_months."})
r, p = stats.pearsonr(df["education_years"], df["pfs_months"])
ans9.append({
    "hypothesis_ids": [hid],
    "code": "pearsonr(education_years, pfs_months)",
    "result_summary": f"r={r:.4f}, p={p:.3g}.",
    "p_value": float(p), "effect_estimate": float(r), "significant": bool(p < 0.05),
})
# Smoking
hid = "h9.5"
hyps9.append({"id": hid, "text": "smoking_pack_years is negatively correlated with pfs_months."})
r, p = stats.pearsonr(df["smoking_pack_years"], df["pfs_months"])
ans9.append({
    "hypothesis_ids": [hid],
    "code": "pearsonr(smoking_pack_years, pfs_months)",
    "result_summary": f"r={r:.4f}, p={p:.3g}.",
    "p_value": float(p), "effect_estimate": float(r), "significant": bool(p < 0.05),
})
add_iter(9, hyps9, ans9)


# ------------------------------------------------------------------
# Iteration 10: Prior therapy and lines of therapy vs PFS
# ------------------------------------------------------------------
hyps10 = []
ans10 = []
prior = [
    "prior_chemotherapy", "prior_radiation", "prior_surgery",
    "prior_immunotherapy", "prior_targeted_therapy",
]
for i, p_ in enumerate(prior, 1):
    hid = f"h10.{i}"
    hyps10.append({"id": hid, "text": f"{p_}=1 is associated with shorter pfs_months than {p_}=0 (heavily pretreated → poorer prognosis)."})
    diff, pv, m1, m0, n1, n0 = ttest_by_group("pfs_months", p_)
    ans10.append({
        "hypothesis_ids": [hid],
        "code": f"ttest by {p_}",
        "result_summary": f"PFS {m1:.3f} (n={n1}) with {p_} vs {m0:.3f} (n={n0}) without; p={pv:.3g}.",
        "p_value": pv, "effect_estimate": diff, "significant": bool(pv < 0.05),
    })
hid = "h10.6"
hyps10.append({"id": hid, "text": "prior_lines_of_therapy is negatively correlated with pfs_months."})
r, p = stats.pearsonr(df["prior_lines_of_therapy"], df["pfs_months"])
ans10.append({
    "hypothesis_ids": [hid],
    "code": "pearsonr(prior_lines_of_therapy, pfs_months)",
    "result_summary": f"r={r:.4f}, p={p:.3g}.",
    "p_value": float(p), "effect_estimate": float(r), "significant": bool(p < 0.05),
})
hid = "h10.7"
hyps10.append({"id": hid, "text": "years_since_diagnosis is negatively correlated with pfs_months."})
r, p = stats.pearsonr(df["years_since_diagnosis"], df["pfs_months"])
ans10.append({
    "hypothesis_ids": [hid],
    "code": "pearsonr(years_since_diagnosis, pfs_months)",
    "result_summary": f"r={r:.4f}, p={p:.3g}.",
    "p_value": float(p), "effect_estimate": float(r), "significant": bool(p < 0.05),
})
add_iter(10, hyps10, ans10)


# ------------------------------------------------------------------
# Iteration 11: Multivariable regression of PFS on key clinical features
# ------------------------------------------------------------------
hyps11 = []
ans11 = []
hid = "h11.1"
hyps11.append({"id": hid, "text": "After adjustment for ECOG, age, mCRPC, visceral_mets, albumin, LDH, alk-phos, hemoglobin, NLR, and prior_lines_of_therapy, multiple of these remain independent predictors of pfs_months."})
fit = linreg("pfs_months ~ age_years + ecog_ps + mcrpc + visceral_mets + albumin_g_dl + ldh_u_l + alkaline_phosphatase_u_l + hemoglobin_g_dl + nlr + prior_lines_of_therapy + gleason_score + psa_ng_ml")
res = []
for var in fit.params.index:
    if var == "Intercept":
        continue
    res.append(f"{var} {fit.params[var]:+.4f} (p={fit.pvalues[var]:.2g})")
ans11.append({
    "hypothesis_ids": [hid],
    "code": "multivariable OLS",
    "result_summary": "OLS adj-R²=" + f"{fit.rsquared_adj:.4f}. Coefs: " + "; ".join(res),
    "p_value": float(fit.f_pvalue),
    "effect_estimate": float(fit.rsquared_adj),
    "significant": bool(fit.f_pvalue < 0.05),
})
# Save coefs for narrative
mv_coefs = {var: (float(fit.params[var]), float(fit.pvalues[var])) for var in fit.params.index}
add_iter(11, hyps11, ans11)


# ------------------------------------------------------------------
# Iteration 12: ECOG-stratified treatment effects (subgroup interaction)
# ------------------------------------------------------------------
hyps12 = []
ans12 = []
for i, t in enumerate(treatments, 1):
    hid = f"h12.{i}"
    hyps12.append({"id": hid, "text": f"The effect of {t} on pfs_months is modified by ecog_ps (treatment × ECOG interaction)."})
    fit = linreg(f"pfs_months ~ {t} * ecog_ps")
    nm = f"{t}:ecog_ps"
    coef = float(fit.params.get(nm, np.nan))
    p = float(fit.pvalues.get(nm, np.nan))
    ans12.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {t}*ecog_ps', df).fit()",
        "result_summary": f"Interaction {t}:ecog_ps {coef:+.4f}, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05) if not np.isnan(p) else None,
    })
add_iter(12, hyps12, ans12)


# ------------------------------------------------------------------
# Iteration 13: Treatment combinations (e.g., dual therapy) and PFS
# ------------------------------------------------------------------
hyps13 = []
ans13 = []
combos = [
    ("treatment_enzalutamide", "treatment_abiraterone"),
    ("treatment_docetaxel", "treatment_abiraterone"),
    ("treatment_docetaxel", "treatment_enzalutamide"),
    ("treatment_olaparib", "treatment_enzalutamide"),
]
for i, (a, b) in enumerate(combos, 1):
    hid = f"h13.{i}"
    hyps13.append({"id": hid, "text": f"There is a non-zero {a} × {b} interaction on pfs_months (synergy or antagonism)."})
    fit = linreg(f"pfs_months ~ {a} * {b}")
    nm = f"{a}:{b}"
    coef = float(fit.params.get(nm, np.nan))
    p = float(fit.pvalues.get(nm, np.nan))
    ans13.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {a}*{b}', df).fit()",
        "result_summary": f"Interaction {a}:{b} {coef:+.4f}, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05) if not np.isnan(p) else None,
    })
add_iter(13, hyps13, ans13)


# ------------------------------------------------------------------
# Iteration 14: Other lab markers (continuous) vs PFS
# ------------------------------------------------------------------
hyps14 = []
ans14 = []
labs = [
    ("ast_u_l", "negative"), ("alt_u_l", "negative"), ("total_bilirubin_mg_dl", "negative"),
    ("creatinine_mg_dl", "negative"), ("bun_mg_dl", "negative"),
    ("sodium_meq_l", "either"), ("potassium_meq_l", "either"),
    ("calcium_mg_dl", "either"), ("glucose_mg_dl", "either"),
    ("platelets_k_ul", "either"), ("wbc_k_ul", "either"),
    ("anc_k_ul", "either"), ("alc_k_ul", "either"),
    ("ca_125_u_ml", "either"), ("cea_ng_ml", "either"),
    ("tsh_uiu_ml", "either"), ("inr", "either"), ("bmi", "either"),
    ("systolic_bp_mmhg", "either"), ("diastolic_bp_mmhg", "either"),
    ("heart_rate_bpm", "either"), ("spo2_pct", "either"),
]
for i, (c, direction) in enumerate(labs, 1):
    hid = f"h14.{i}"
    hyps14.append({"id": hid, "text": f"{c} has a {direction} (or any) association with pfs_months."})
    r, p = stats.pearsonr(df[c], df["pfs_months"])
    ans14.append({
        "hypothesis_ids": [hid],
        "code": f"pearsonr({c}, pfs_months)",
        "result_summary": f"r={r:.4f}, p={p:.3g}.",
        "p_value": float(p), "effect_estimate": float(r), "significant": bool(p < 0.05),
    })
add_iter(14, hyps14, ans14)


# ------------------------------------------------------------------
# Iteration 15: SNP main effects on PFS (Pearson) — search for any hits
# ------------------------------------------------------------------
hyps15 = []
ans15 = []
snps = [c for c in df.columns if c.startswith("snp_")]
for i, s in enumerate(snps, 1):
    hid = f"h15.{i}"
    hyps15.append({"id": hid, "text": f"{s} is associated with pfs_months (linear additive coding)."})
    r, p = stats.pearsonr(df[s], df["pfs_months"])
    ans15.append({
        "hypothesis_ids": [hid],
        "code": f"pearsonr({s}, pfs_months)",
        "result_summary": f"r={r:.4f}, p={p:.3g}.",
        "p_value": float(p), "effect_estimate": float(r), "significant": bool(p < 0.05),
    })
add_iter(15, hyps15, ans15)


# ------------------------------------------------------------------
# Iteration 16: Refined precision-medicine subgroups — biomarker-positive vs biomarker-negative on-treatment effect
# ------------------------------------------------------------------
hyps16 = []
ans16 = []
match_pairs = [
    ("treatment_olaparib", "brca2_mutation"),
    ("treatment_pembrolizumab", "msi_high"),
    ("treatment_lu177_psma", "psma_high"),
]
for i, (tx, mk) in enumerate(match_pairs, 1):
    hid_a = f"h16.{i}.match"
    hid_b = f"h16.{i}.nomatch"
    hyps16.append({"id": hid_a, "text": f"Among {mk}=1 patients, mean pfs_months is higher with {tx}=1 than {tx}=0 (biomarker-matched)."})
    hyps16.append({"id": hid_b, "text": f"Among {mk}=0 patients, mean pfs_months is similar (no large difference) between {tx}=1 and {tx}=0 (biomarker-unmatched, expect no benefit)."})
    for mval, hid_use in [(1, hid_a), (0, hid_b)]:
        sub = df[df[mk] == mval]
        a = sub.loc[sub[tx] == 1, "pfs_months"]
        b = sub.loc[sub[tx] == 0, "pfs_months"]
        if len(a) < 5 or len(b) < 5:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        diff = float(a.mean() - b.mean())
        ans16.append({
            "hypothesis_ids": [hid_use],
            "code": f"sub=df[df['{mk}']=={mval}]; ttest of pfs by {tx}",
            "result_summary": (f"{mk}={mval}: PFS {a.mean():.3f} (n={len(a)}) on {tx} vs {b.mean():.3f} "
                                f"(n={len(b)}) off; diff {diff:+.3f}, p={p:.3g}."),
            "p_value": float(p), "effect_estimate": diff, "significant": bool(p < 0.05),
        })
add_iter(16, hyps16, ans16)


# ------------------------------------------------------------------
# Iteration 17: AR-V7 attenuation of AR-targeted therapy benefit (resistance)
# ------------------------------------------------------------------
hyps17 = []
ans17 = []
for i, tx in enumerate(["treatment_enzalutamide", "treatment_abiraterone"], 1):
    for mval in [1, 0]:
        hid = f"h17.{i}.{mval}"
        hyps17.append({"id": hid,
                       "text": f"Within ar_v7_positive={mval}, mean pfs_months differs between {tx}=1 and {tx}=0; expected: in AR-V7+ no/negative benefit, in AR-V7- positive benefit."})
        sub = df[df.ar_v7_positive == mval]
        a = sub.loc[sub[tx] == 1, "pfs_months"]
        b = sub.loc[sub[tx] == 0, "pfs_months"]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        diff = float(a.mean() - b.mean())
        ans17.append({
            "hypothesis_ids": [hid],
            "code": f"sub=df[ar_v7=={mval}]; ttest pfs ~ {tx}",
            "result_summary": (f"AR-V7={mval}: PFS {a.mean():.3f} (n={len(a)}) on {tx} vs "
                                f"{b.mean():.3f} (n={len(b)}); diff {diff:+.3f}, p={p:.3g}."),
            "p_value": float(p), "effect_estimate": diff, "significant": bool(p < 0.05),
        })
add_iter(17, hyps17, ans17)


# ------------------------------------------------------------------
# Iteration 18: Visceral / liver mets as effect modifiers of treatments
# ------------------------------------------------------------------
hyps18 = []
ans18 = []
for i, tx in enumerate(treatments, 1):
    hid = f"h18.{i}"
    hyps18.append({"id": hid, "text": f"{tx} effect on pfs_months is modified by visceral_mets (interaction)."})
    fit = linreg(f"pfs_months ~ {tx} * visceral_mets")
    nm = f"{tx}:visceral_mets"
    coef = float(fit.params.get(nm, np.nan))
    p = float(fit.pvalues.get(nm, np.nan))
    ans18.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {tx}*visceral_mets').fit()",
        "result_summary": f"Interaction {tx}:visceral_mets {coef:+.4f}, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05) if not np.isnan(p) else None,
    })
add_iter(18, hyps18, ans18)


# ------------------------------------------------------------------
# Iteration 19: mCRPC × treatment interactions
# ------------------------------------------------------------------
hyps19 = []
ans19 = []
for i, tx in enumerate(treatments, 1):
    hid = f"h19.{i}"
    hyps19.append({"id": hid, "text": f"{tx} effect on pfs_months is modified by mcrpc (interaction)."})
    fit = linreg(f"pfs_months ~ {tx} * mcrpc")
    nm = f"{tx}:mcrpc"
    coef = float(fit.params.get(nm, np.nan))
    p = float(fit.pvalues.get(nm, np.nan))
    ans19.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {tx}*mcrpc').fit()",
        "result_summary": f"Interaction {tx}:mcrpc {coef:+.4f}, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05) if not np.isnan(p) else None,
    })
add_iter(19, hyps19, ans19)


# ------------------------------------------------------------------
# Iteration 20: Final adjusted treatment-effect model with predicted-match interactions
# ------------------------------------------------------------------
hyps20 = []
ans20 = []
hid = "h20.1"
hyps20.append({"id": hid, "text": "After adjustment for clinical covariates and main effects, predicted biomarker × treatment interactions for olaparib×BRCA2, pembrolizumab×MSI-high, and Lu177-PSMA×PSMA-high remain positive on pfs_months."})
formula = ("pfs_months ~ age_years + ecog_ps + mcrpc + visceral_mets + albumin_g_dl + ldh_u_l + "
           "alkaline_phosphatase_u_l + hemoglobin_g_dl + nlr + prior_lines_of_therapy + gleason_score + "
           "psa_ng_ml + brca2_mutation + msi_high + psma_high + ar_v7_positive + "
           "treatment_enzalutamide + treatment_abiraterone + treatment_docetaxel + "
           "treatment_olaparib + treatment_lu177_psma + treatment_pembrolizumab + "
           "treatment_olaparib:brca2_mutation + treatment_pembrolizumab:msi_high + "
           "treatment_lu177_psma:psma_high + treatment_enzalutamide:ar_v7_positive + "
           "treatment_abiraterone:ar_v7_positive")
fit_full = linreg(formula)
key_terms = [
    "treatment_olaparib:brca2_mutation",
    "treatment_pembrolizumab:msi_high",
    "treatment_lu177_psma:psma_high",
    "treatment_enzalutamide:ar_v7_positive",
    "treatment_abiraterone:ar_v7_positive",
]
parts = []
key_results = {}
for k in key_terms:
    c = float(fit_full.params.get(k, np.nan))
    p = float(fit_full.pvalues.get(k, np.nan))
    parts.append(f"{k} {c:+.4f} (p={p:.2g})")
    key_results[k] = (c, p)
ans20.append({
    "hypothesis_ids": [hid],
    "code": "Full multivariable OLS with predicted interactions",
    "result_summary": "Adj-R²=" + f"{fit_full.rsquared_adj:.4f}. Key interactions: " + "; ".join(parts),
    "p_value": float(fit_full.f_pvalue),
    "effect_estimate": float(key_results["treatment_olaparib:brca2_mutation"][0]),
    "significant": bool(fit_full.f_pvalue < 0.05),
})
add_iter(20, hyps20, ans20)


# ------------------------------------------------------------------
# Iteration 21: SES interactions — does insurance / rural modify treatment access patterns?
# ------------------------------------------------------------------
hyps21 = []
ans21 = []
hid = "h21.1"
hyps21.append({"id": hid, "text": "Patients on Medicaid or uninsured have shorter pfs_months than those with private insurance (after adjustment for ECOG and age)."})
df["ins_uninsured"] = (df["insurance_type"] == "uninsured").astype(int)
df["ins_medicaid"] = (df["insurance_type"] == "medicaid").astype(int)
df["ins_medicare"] = (df["insurance_type"] == "medicare").astype(int)
fit = linreg("pfs_months ~ ins_uninsured + ins_medicaid + ins_medicare + age_years + ecog_ps")
parts = [f"{k} {fit.params[k]:+.4f} (p={fit.pvalues[k]:.2g})" for k in ["ins_uninsured","ins_medicaid","ins_medicare"]]
ans21.append({
    "hypothesis_ids": [hid],
    "code": "OLS pfs ~ insurance dummies + age + ECOG",
    "result_summary": "; ".join(parts),
    "p_value": float(fit.pvalues["ins_uninsured"]),
    "effect_estimate": float(fit.params["ins_uninsured"]),
    "significant": bool(fit.pvalues["ins_uninsured"] < 0.05),
})
hid = "h21.2"
hyps21.append({"id": hid, "text": "Black race is associated with different pfs_months than white race after adjusting for age and ECOG."})
df["race_black"] = (df["race_ethnicity"] == "black").astype(int)
df["race_hispanic"] = (df["race_ethnicity"] == "hispanic").astype(int)
df["race_asian"] = (df["race_ethnicity"] == "asian").astype(int)
df["race_other"] = (df["race_ethnicity"] == "other").astype(int)
fit = linreg("pfs_months ~ race_black + race_hispanic + race_asian + race_other + age_years + ecog_ps")
parts = [f"{k} {fit.params[k]:+.4f} (p={fit.pvalues[k]:.2g})" for k in ["race_black","race_hispanic","race_asian","race_other"]]
ans21.append({
    "hypothesis_ids": [hid],
    "code": "OLS pfs ~ race dummies + age + ECOG",
    "result_summary": "; ".join(parts),
    "p_value": float(fit.pvalues["race_black"]),
    "effect_estimate": float(fit.params["race_black"]),
    "significant": bool(fit.pvalues["race_black"] < 0.05),
})
add_iter(21, hyps21, ans21)


# ------------------------------------------------------------------
# Iteration 22: BRCA2 × olaparib & PSMA × Lu177 stratified replication w/ Cohen's d
# ------------------------------------------------------------------
hyps22 = []
ans22 = []
for i, (tx, mk) in enumerate(match_pairs, 1):
    hid = f"h22.{i}"
    hyps22.append({"id": hid, "text": f"Within {mk}=1, the effect size (mean diff) of {tx} on pfs_months is larger than within {mk}=0 (interaction confirmed by stratified diffs)."})
    sub1 = df[df[mk] == 1]; sub0 = df[df[mk] == 0]
    d1 = sub1.loc[sub1[tx] == 1, "pfs_months"].mean() - sub1.loc[sub1[tx] == 0, "pfs_months"].mean()
    d0 = sub0.loc[sub0[tx] == 1, "pfs_months"].mean() - sub0.loc[sub0[tx] == 0, "pfs_months"].mean()
    fit = linreg(f"pfs_months ~ {tx} * {mk}")
    p = float(fit.pvalues.get(f"{tx}:{mk}", np.nan))
    ans22.append({
        "hypothesis_ids": [hid],
        "code": f"stratified diffs + interaction p",
        "result_summary": f"{mk}=1 diff {d1:+.3f}; {mk}=0 diff {d0:+.3f}; interaction p={p:.3g}.",
        "p_value": p, "effect_estimate": float(d1 - d0), "significant": bool(p < 0.05),
    })
add_iter(22, hyps22, ans22)


# ------------------------------------------------------------------
# Iteration 23: TP53 / PTEN loss as adverse modifiers of treatment benefit
# ------------------------------------------------------------------
hyps23 = []
ans23 = []
for i, tx in enumerate(["treatment_enzalutamide","treatment_abiraterone","treatment_docetaxel"], 1):
    for mk in ["tp53_mutation","pten_loss"]:
        hid = f"h23.{tx[10:13]}_{mk}"
        hyps23.append({"id": hid, "text": f"{mk}=1 attenuates the effect of {tx} on pfs_months (negative interaction)."})
        fit = linreg(f"pfs_months ~ {tx} * {mk}")
        nm = f"{tx}:{mk}"
        coef = float(fit.params.get(nm, np.nan))
        p = float(fit.pvalues.get(nm, np.nan))
        ans23.append({
            "hypothesis_ids": [hid],
            "code": f"OLS pfs ~ {tx}*{mk}",
            "result_summary": f"Interaction {nm} {coef:+.4f}, p={p:.3g}.",
            "p_value": p, "effect_estimate": coef,
            "significant": bool(p < 0.05) if not np.isnan(p) else None,
        })
add_iter(23, hyps23, ans23)


# ------------------------------------------------------------------
# Iteration 24: Bone-mets-specific responses + alkaline phosphatase modifier
# ------------------------------------------------------------------
hyps24 = []
ans24 = []
hid = "h24.1"
hyps24.append({"id": hid, "text": "Lu177-PSMA effect on pfs_months is modified by bone_mets status (interaction)."})
fit = linreg("pfs_months ~ treatment_lu177_psma * bone_mets")
nm = "treatment_lu177_psma:bone_mets"
ans24.append({
    "hypothesis_ids": [hid],
    "code": "OLS pfs ~ Lu177*bone_mets",
    "result_summary": f"Interaction {nm} {fit.params[nm]:+.4f}, p={fit.pvalues[nm]:.3g}.",
    "p_value": float(fit.pvalues[nm]), "effect_estimate": float(fit.params[nm]),
    "significant": bool(fit.pvalues[nm] < 0.05),
})
hid = "h24.2"
hyps24.append({"id": hid, "text": "Higher alkaline_phosphatase_u_l predicts larger Lu177-PSMA benefit (positive interaction)."})
fit = linreg("pfs_months ~ treatment_lu177_psma * alkaline_phosphatase_u_l")
nm = "treatment_lu177_psma:alkaline_phosphatase_u_l"
ans24.append({
    "hypothesis_ids": [hid],
    "code": "OLS pfs ~ Lu177*alk_phos",
    "result_summary": f"Interaction {nm} {fit.params[nm]:+.6f}, p={fit.pvalues[nm]:.3g}.",
    "p_value": float(fit.pvalues[nm]), "effect_estimate": float(fit.params[nm]),
    "significant": bool(fit.pvalues[nm] < 0.05),
})
hid = "h24.3"
hyps24.append({"id": hid, "text": "Olaparib effect on pfs_months differs by tp53_mutation status (interaction)."})
fit = linreg("pfs_months ~ treatment_olaparib * tp53_mutation")
nm = "treatment_olaparib:tp53_mutation"
ans24.append({
    "hypothesis_ids": [hid],
    "code": "OLS pfs ~ olaparib*tp53",
    "result_summary": f"Interaction {nm} {fit.params[nm]:+.4f}, p={fit.pvalues[nm]:.3g}.",
    "p_value": float(fit.pvalues[nm]), "effect_estimate": float(fit.params[nm]),
    "significant": bool(fit.pvalues[nm] < 0.05),
})
add_iter(24, hyps24, ans24)


# ------------------------------------------------------------------
# Iteration 25: Final synthesis — biomarker-matched-treatment count vs PFS, and overall summary
# ------------------------------------------------------------------
hyps25 = []
ans25 = []
hid = "h25.1"
hyps25.append({"id": hid, "text": "A composite 'biomarker-matched-therapy' indicator (any of: olaparib×BRCA2, pembrolizumab×MSI-high, Lu177×PSMA-high) is positively associated with pfs_months."})
df["matched_therapy"] = (
    ((df["treatment_olaparib"] == 1) & (df["brca2_mutation"] == 1)) |
    ((df["treatment_pembrolizumab"] == 1) & (df["msi_high"] == 1)) |
    ((df["treatment_lu177_psma"] == 1) & (df["psma_high"] == 1))
).astype(int)
diff, p, m1, m0, n1, n0 = ttest_by_group("pfs_months", "matched_therapy")
ans25.append({
    "hypothesis_ids": [hid],
    "code": "ttest pfs by matched_therapy composite",
    "result_summary": f"PFS {m1:.3f} (n={n1}) matched vs {m0:.3f} (n={n0}) unmatched; diff {diff:+.3f}, p={p:.3g}.",
    "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
})

hid = "h25.2"
hyps25.append({"id": hid, "text": "After adjusting for ECOG, age, mCRPC, visceral_mets, LDH, albumin, the matched_therapy indicator remains positive and significant."})
fit = linreg("pfs_months ~ matched_therapy + age_years + ecog_ps + mcrpc + visceral_mets + albumin_g_dl + ldh_u_l + nlr + prior_lines_of_therapy")
ans25.append({
    "hypothesis_ids": [hid],
    "code": "Adjusted OLS pfs ~ matched_therapy + covariates",
    "result_summary": (f"matched_therapy {fit.params['matched_therapy']:+.4f} "
                        f"(p={fit.pvalues['matched_therapy']:.3g}); ECOG {fit.params['ecog_ps']:+.4f}; "
                        f"mcrpc {fit.params['mcrpc']:+.4f}; albumin {fit.params['albumin_g_dl']:+.4f}."),
    "p_value": float(fit.pvalues["matched_therapy"]),
    "effect_estimate": float(fit.params["matched_therapy"]),
    "significant": bool(fit.pvalues["matched_therapy"] < 0.05),
})

hid = "h25.3"
hyps25.append({"id": hid, "text": "Combining all six treatment indicators (additive), the number of concurrent treatments is associated with pfs_months."})
df["n_treatments"] = df[treatments].sum(axis=1)
r, p = stats.pearsonr(df["n_treatments"], df["pfs_months"])
ans25.append({
    "hypothesis_ids": [hid],
    "code": "pearsonr(n_treatments, pfs_months)",
    "result_summary": f"r={r:.4f}, p={p:.3g}.",
    "p_value": float(p), "effect_estimate": float(r), "significant": bool(p < 0.05),
})
add_iter(25, hyps25, ans25)


# ------------------------------------------------------------------
# Save transcript
# ------------------------------------------------------------------
transcript = {
    "dataset_id": "ds001_prostate",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@manual-2026-04-28",
    "max_iterations": 25,
    "iterations": iterations,
}
with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)
print("Wrote transcript.json")

# Save key data for narrative
import pickle
with open("_narrative_data.pkl", "wb") as f:
    pickle.dump({
        "iterations": iterations,
        "mv_coefs": mv_coefs,
        "key_results": key_results,
    }, f)
print("Wrote _narrative_data.pkl")
