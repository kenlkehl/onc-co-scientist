"""Iterative hypothesis testing on AML cohort."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
out = "objective_response"
y = df[out].values

results = {"iterations": []}


def add_iter(idx, hyps, analyses):
    results["iterations"].append({
        "index": idx,
        "proposed_hypotheses": hyps,
        "analyses": analyses,
    })


def two_prop_test(df, var, val_a, val_b, outcome=out):
    """Test diff in proportion of outcome between two groups."""
    a = df.loc[df[var] == val_a, outcome]
    b = df.loc[df[var] == val_b, outcome]
    p_a, p_b = a.mean(), b.mean()
    n_a, n_b = len(a), len(b)
    # Z test for proportions
    p_pool = (a.sum() + b.sum()) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return p_a - p_b, 1.0, n_a, n_b
    z = (p_a - p_b) / se
    pv = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_a - p_b, pv, n_a, n_b


def logistic_main(formula, data=None):
    if data is None:
        data = df
    m = smf.logit(formula, data=data).fit(disp=0)
    return m


def interaction_logit(tx, mut):
    """Run logistic with interaction tx*mut and return coefs/p for interaction term."""
    m = smf.logit(f"{out} ~ {tx} * {mut}", data=df).fit(disp=0)
    coef = m.params.get(f"{tx}:{mut}", np.nan)
    pv = m.pvalues.get(f"{tx}:{mut}", np.nan)
    return m, coef, pv


# ============================================================
# Iteration 1: baseline demographics & disease features main
# ============================================================
hyps1 = [
    {"id": "h1_age", "text": "Older age (`age_years`) is associated with lower probability of `objective_response`."},
    {"id": "h1_sex", "text": "Female patients (`sex_female`=1) have a different probability of `objective_response` than males."},
    {"id": "h1_ecog", "text": "Higher `ecog_ps` is associated with lower `objective_response` rate."},
]
ana1 = []

# Age
m = logistic_main(f"{out} ~ age_years")
ana1.append({
    "hypothesis_ids": ["h1_age"],
    "code": f"smf.logit('{out} ~ age_years', data=df).fit()",
    "result_summary": f"Logistic regression: log-OR per year of age = {m.params['age_years']:.5f} (p={m.pvalues['age_years']:.3g}).",
    "p_value": float(m.pvalues["age_years"]),
    "effect_estimate": float(m.params["age_years"]),
    "significant": bool(m.pvalues["age_years"] < 0.05),
})

# Sex
diff, pv, na, nb = two_prop_test(df, "sex_female", 1, 0)
ana1.append({
    "hypothesis_ids": ["h1_sex"],
    "code": "two-prop z-test on objective_response by sex_female",
    "result_summary": f"Response: female={df.loc[df.sex_female==1,out].mean():.3f} (n={na}) vs male={df.loc[df.sex_female==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(diff),
    "significant": bool(pv < 0.05),
})

# ECOG
m = logistic_main(f"{out} ~ ecog_ps")
ana1.append({
    "hypothesis_ids": ["h1_ecog"],
    "code": f"smf.logit('{out} ~ ecog_ps', data=df).fit()",
    "result_summary": f"Logistic regression: log-OR per ECOG point = {m.params['ecog_ps']:.4f} (p={m.pvalues['ecog_ps']:.3g}).",
    "p_value": float(m.pvalues["ecog_ps"]),
    "effect_estimate": float(m.params["ecog_ps"]),
    "significant": bool(m.pvalues["ecog_ps"] < 0.05),
})
add_iter(1, hyps1, ana1)


# ============================================================
# Iteration 2: disease biology main effects
# ============================================================
hyps2 = [
    {"id": "h2_secondary", "text": "Patients with `secondary_aml`=1 have a lower `objective_response` rate than de-novo AML."},
    {"id": "h2_complex", "text": "Patients with `complex_karyotype`=1 have a lower `objective_response` rate."},
    {"id": "h2_blasts", "text": "Higher `blast_pct_marrow` is associated with lower `objective_response` rate."},
    {"id": "h2_unfit", "text": "Patients flagged `unfit_for_intensive`=1 have a lower `objective_response` rate."},
]
ana2 = []
for hid, var in [("h2_secondary", "secondary_aml"),
                  ("h2_complex", "complex_karyotype"),
                  ("h2_unfit", "unfit_for_intensive")]:
    diff, pv, na, nb = two_prop_test(df, var, 1, 0)
    ana2.append({
        "hypothesis_ids": [hid],
        "code": f"two-prop z-test on {out} by {var}",
        "result_summary": f"Response: {var}+={df.loc[df[var]==1,out].mean():.3f} (n={na}) vs {var}-={df.loc[df[var]==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(diff),
        "significant": bool(pv < 0.05),
    })
m = logistic_main(f"{out} ~ blast_pct_marrow")
ana2.append({
    "hypothesis_ids": ["h2_blasts"],
    "code": f"smf.logit('{out} ~ blast_pct_marrow', data=df).fit()",
    "result_summary": f"Log-OR per 1% marrow-blast increase = {m.params['blast_pct_marrow']:.5f} (p={m.pvalues['blast_pct_marrow']:.3g}).",
    "p_value": float(m.pvalues["blast_pct_marrow"]),
    "effect_estimate": float(m.params["blast_pct_marrow"]),
    "significant": bool(m.pvalues["blast_pct_marrow"] < 0.05),
})
add_iter(2, hyps2, ana2)


# ============================================================
# Iteration 3: mutation main effects
# ============================================================
hyps3 = [
    {"id": "h3_flt3itd", "text": "`flt3_itd`=1 patients have a different `objective_response` rate than FLT3-ITD-negative patients."},
    {"id": "h3_idh1", "text": "`idh1_mutation`=1 patients have a higher `objective_response` rate than IDH1-wild-type."},
    {"id": "h3_idh2", "text": "`idh2_mutation`=1 patients have a different `objective_response` rate than IDH2-wild-type."},
    {"id": "h3_npm1", "text": "`npm1_mutation`=1 patients have a different `objective_response` rate than NPM1-wild-type."},
    {"id": "h3_tp53", "text": "`tp53_mutation`=1 patients have a lower `objective_response` rate than TP53-wild-type."},
]
ana3 = []
for hid, var in [("h3_flt3itd", "flt3_itd"),
                  ("h3_idh1", "idh1_mutation"),
                  ("h3_idh2", "idh2_mutation"),
                  ("h3_npm1", "npm1_mutation"),
                  ("h3_tp53", "tp53_mutation")]:
    diff, pv, na, nb = two_prop_test(df, var, 1, 0)
    ana3.append({
        "hypothesis_ids": [hid],
        "code": f"two-prop z-test on {out} by {var}",
        "result_summary": f"{var}+: {df.loc[df[var]==1,out].mean():.3f} (n={na}); {var}-: {df.loc[df[var]==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(diff),
        "significant": bool(pv < 0.05),
    })
add_iter(3, hyps3, ana3)


# ============================================================
# Iteration 4: FLT3 inhibitor x FLT3 mutation interactions
# ============================================================
hyps4 = [
    {"id": "h4_mido_itd", "text": "Among `flt3_itd`=1 patients, `treatment_midostaurin`=1 yields a higher `objective_response` rate than no midostaurin (positive treatment-mutation interaction)."},
    {"id": "h4_gilt_itd", "text": "Among `flt3_itd`=1 patients, `treatment_gilteritinib`=1 yields a higher `objective_response` rate than no gilteritinib (positive treatment-mutation interaction)."},
    {"id": "h4_mido_tkd", "text": "Among `flt3_tkd`=1 patients, `treatment_midostaurin`=1 yields a higher `objective_response` rate than no midostaurin."},
    {"id": "h4_gilt_tkd", "text": "Among `flt3_tkd`=1 patients, `treatment_gilteritinib`=1 yields a higher `objective_response` rate than no gilteritinib."},
]
ana4 = []
for hid, tx, mut in [("h4_mido_itd", "treatment_midostaurin", "flt3_itd"),
                      ("h4_gilt_itd", "treatment_gilteritinib", "flt3_itd"),
                      ("h4_mido_tkd", "treatment_midostaurin", "flt3_tkd"),
                      ("h4_gilt_tkd", "treatment_gilteritinib", "flt3_tkd")]:
    m, coef, pv = interaction_logit(tx, mut)
    a = df.loc[(df[tx]==1)&(df[mut]==1),out].mean()
    b = df.loc[(df[tx]==1)&(df[mut]==0),out].mean()
    c = df.loc[(df[tx]==0)&(df[mut]==1),out].mean()
    d = df.loc[(df[tx]==0)&(df[mut]==0),out].mean()
    n_aa = ((df[tx]==1)&(df[mut]==1)).sum()
    ana4.append({
        "hypothesis_ids": [hid],
        "code": f"smf.logit('{out} ~ {tx}*{mut}', data=df).fit()",
        "result_summary": f"Cells: tx+mut+={a:.3f} (n={n_aa}), tx+mut-={b:.3f}, tx-mut+={c:.3f}, tx-mut-={d:.3f}. Tx effect within {mut}+: {a-c:+.4f}. Interaction (log-OR) coef={coef:+.4f}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(a - c),  # absolute response-rate change for tx+ within mutation+
        "significant": bool(pv < 0.05),
    })
add_iter(4, hyps4, ana4)


# ============================================================
# Iteration 5: IDH inhibitor x IDH mutation interactions
# ============================================================
hyps5 = [
    {"id": "h5_ivo_idh1", "text": "Among `idh1_mutation`=1 patients, `treatment_ivosidenib`=1 yields a higher `objective_response` rate than no ivosidenib (positive treatment-mutation interaction)."},
    {"id": "h5_ena_idh2", "text": "Among `idh2_mutation`=1 patients, `treatment_enasidenib`=1 yields a higher `objective_response` rate than no enasidenib (positive treatment-mutation interaction)."},
    {"id": "h5_ivo_idh2", "text": "Among `idh2_mutation`=1 patients, `treatment_ivosidenib`=1 (off-target) yields a similar `objective_response` rate to no ivosidenib."},
    {"id": "h5_ena_idh1", "text": "Among `idh1_mutation`=1 patients, `treatment_enasidenib`=1 (off-target) yields a similar `objective_response` rate to no enasidenib."},
]
ana5 = []
for hid, tx, mut in [("h5_ivo_idh1", "treatment_ivosidenib", "idh1_mutation"),
                      ("h5_ena_idh2", "treatment_enasidenib", "idh2_mutation"),
                      ("h5_ivo_idh2", "treatment_ivosidenib", "idh2_mutation"),
                      ("h5_ena_idh1", "treatment_enasidenib", "idh1_mutation")]:
    m, coef, pv = interaction_logit(tx, mut)
    a = df.loc[(df[tx]==1)&(df[mut]==1),out].mean()
    b = df.loc[(df[tx]==1)&(df[mut]==0),out].mean()
    c = df.loc[(df[tx]==0)&(df[mut]==1),out].mean()
    d = df.loc[(df[tx]==0)&(df[mut]==0),out].mean()
    n_aa = ((df[tx]==1)&(df[mut]==1)).sum()
    ana5.append({
        "hypothesis_ids": [hid],
        "code": f"smf.logit('{out} ~ {tx}*{mut}', data=df).fit()",
        "result_summary": f"Cells: tx+mut+={a:.3f} (n={n_aa}), tx+mut-={b:.3f}, tx-mut+={c:.3f}, tx-mut-={d:.3f}. Tx effect within {mut}+: {a-c:+.4f}. Interaction coef={coef:+.4f}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(a - c),
        "significant": bool(pv < 0.05),
    })
add_iter(5, hyps5, ana5)


# ============================================================
# Iteration 6: NPM1 / ven-aza interactions
# ============================================================
hyps6 = [
    {"id": "h6_venaza_npm1", "text": "Among `npm1_mutation`=1 patients, `treatment_venetoclax_azacitidine`=1 yields a higher `objective_response` rate than no ven/aza (positive interaction)."},
    {"id": "h6_venaza_tp53", "text": "Among `tp53_mutation`=1 patients, `treatment_venetoclax_azacitidine`=1 yields a different `objective_response` rate than no ven/aza."},
    {"id": "h6_venaza_idh", "text": "Among patients with any IDH mutation (`idh1_mutation`=1 OR `idh2_mutation`=1), `treatment_venetoclax_azacitidine`=1 yields a higher response rate."},
    {"id": "h6_venaza_main", "text": "Overall, `treatment_venetoclax_azacitidine`=1 is associated with a higher `objective_response` rate than no ven/aza (main effect)."},
]
ana6 = []
m, coef, pv = interaction_logit("treatment_venetoclax_azacitidine", "npm1_mutation")
tx, mut = "treatment_venetoclax_azacitidine", "npm1_mutation"
a = df.loc[(df[tx]==1)&(df[mut]==1),out].mean()
c = df.loc[(df[tx]==0)&(df[mut]==1),out].mean()
n_aa = ((df[tx]==1)&(df[mut]==1)).sum()
ana6.append({
    "hypothesis_ids": ["h6_venaza_npm1"],
    "code": f"smf.logit('{out} ~ {tx}*{mut}', data=df).fit()",
    "result_summary": f"NPM1+ ven/aza+={a:.3f} (n={n_aa}); NPM1+ ven/aza-={c:.3f}. Tx effect in NPM1+ : {a-c:+.4f}. Interaction coef={coef:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(a - c),
    "significant": bool(pv < 0.05),
})
m, coef, pv = interaction_logit("treatment_venetoclax_azacitidine", "tp53_mutation")
tx, mut = "treatment_venetoclax_azacitidine", "tp53_mutation"
a = df.loc[(df[tx]==1)&(df[mut]==1),out].mean()
c = df.loc[(df[tx]==0)&(df[mut]==1),out].mean()
n_aa = ((df[tx]==1)&(df[mut]==1)).sum()
ana6.append({
    "hypothesis_ids": ["h6_venaza_tp53"],
    "code": f"smf.logit('{out} ~ {tx}*{mut}', data=df).fit()",
    "result_summary": f"TP53+ ven/aza+={a:.3f} (n={n_aa}); TP53+ ven/aza-={c:.3f}. Tx effect in TP53+ : {a-c:+.4f}. Interaction coef={coef:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(a - c),
    "significant": bool(pv < 0.05),
})
df["any_idh"] = ((df["idh1_mutation"] == 1) | (df["idh2_mutation"] == 1)).astype(int)
m, coef, pv = interaction_logit("treatment_venetoclax_azacitidine", "any_idh")
tx, mut = "treatment_venetoclax_azacitidine", "any_idh"
a = df.loc[(df[tx]==1)&(df[mut]==1),out].mean()
c = df.loc[(df[tx]==0)&(df[mut]==1),out].mean()
n_aa = ((df[tx]==1)&(df[mut]==1)).sum()
ana6.append({
    "hypothesis_ids": ["h6_venaza_idh"],
    "code": f"smf.logit('{out} ~ treatment_venetoclax_azacitidine*any_idh', data=df).fit()",
    "result_summary": f"any_idh+ ven/aza+={a:.3f} (n={n_aa}); any_idh+ ven/aza-={c:.3f}. Tx effect={a-c:+.4f}. Interaction coef={coef:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(a - c),
    "significant": bool(pv < 0.05),
})
diff, pv, na, nb = two_prop_test(df, "treatment_venetoclax_azacitidine", 1, 0)
ana6.append({
    "hypothesis_ids": ["h6_venaza_main"],
    "code": "two-prop z-test on objective_response by treatment_venetoclax_azacitidine",
    "result_summary": f"VenAza+: {df.loc[df.treatment_venetoclax_azacitidine==1,out].mean():.3f} vs VenAza-: {df.loc[df.treatment_venetoclax_azacitidine==0,out].mean():.3f}; diff={diff:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(diff),
    "significant": bool(pv < 0.05),
})
add_iter(6, hyps6, ana6)


# ============================================================
# Iteration 7: 7+3 induction interactions / unfit
# ============================================================
hyps7 = [
    {"id": "h7_7p3_unfit", "text": "Among `unfit_for_intensive`=1 patients, `treatment_7plus3`=1 is associated with a lower `objective_response` rate than no 7+3 (negative interaction in unfit subgroup)."},
    {"id": "h7_7p3_complex", "text": "Among `complex_karyotype`=1 patients, `treatment_7plus3`=1 yields a lower `objective_response` rate than no 7+3."},
    {"id": "h7_7p3_age", "text": "The `treatment_7plus3`=1 effect on `objective_response` weakens with older `age_years` (negative interaction with age)."},
    {"id": "h7_venaza_unfit", "text": "Among `unfit_for_intensive`=1 patients, `treatment_venetoclax_azacitidine`=1 yields a higher `objective_response` rate than no ven/aza."},
]
ana7 = []
for hid, tx, mut in [("h7_7p3_unfit", "treatment_7plus3", "unfit_for_intensive"),
                      ("h7_7p3_complex", "treatment_7plus3", "complex_karyotype"),
                      ("h7_venaza_unfit", "treatment_venetoclax_azacitidine", "unfit_for_intensive")]:
    m, coef, pv = interaction_logit(tx, mut)
    a = df.loc[(df[tx]==1)&(df[mut]==1),out].mean()
    c = df.loc[(df[tx]==0)&(df[mut]==1),out].mean()
    n_aa = ((df[tx]==1)&(df[mut]==1)).sum()
    ana7.append({
        "hypothesis_ids": [hid],
        "code": f"smf.logit('{out} ~ {tx}*{mut}', data=df).fit()",
        "result_summary": f"{mut}+ tx+={a:.3f} (n={n_aa}); {mut}+ tx-={c:.3f}. Tx effect in {mut}+: {a-c:+.4f}. Interaction coef={coef:+.4f}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(a - c),
        "significant": bool(pv < 0.05),
    })
m = smf.logit(f"{out} ~ treatment_7plus3 * age_years", data=df).fit(disp=0)
coef = m.params["treatment_7plus3:age_years"]
pv = m.pvalues["treatment_7plus3:age_years"]
ana7.append({
    "hypothesis_ids": ["h7_7p3_age"],
    "code": "smf.logit('objective_response ~ treatment_7plus3 * age_years', data=df).fit()",
    "result_summary": f"Interaction log-OR per year (7+3 x age) = {coef:+.5f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(coef),
    "significant": bool(pv < 0.05),
})
add_iter(7, hyps7, ana7)


# ============================================================
# Iteration 8: laboratory predictors
# ============================================================
hyps8 = [
    {"id": "h8_wbc", "text": "Higher `wbc_k_per_ul` at baseline is associated with lower `objective_response` rate."},
    {"id": "h8_ldh", "text": "Higher `ldh_u_l` at baseline is associated with lower `objective_response` rate."},
    {"id": "h8_albumin", "text": "Higher `albumin_g_dl` at baseline is associated with higher `objective_response` rate."},
    {"id": "h8_hgb", "text": "Higher `hemoglobin_g_dl` at baseline is associated with higher `objective_response` rate."},
    {"id": "h8_plt", "text": "Higher `platelets_k_ul` at baseline is associated with higher `objective_response` rate."},
    {"id": "h8_anc", "text": "Higher `anc_k_ul` at baseline is associated with higher `objective_response` rate."},
]
ana8 = []
for hid, var in [("h8_wbc", "wbc_k_per_ul"),
                  ("h8_ldh", "ldh_u_l"),
                  ("h8_albumin", "albumin_g_dl"),
                  ("h8_hgb", "hemoglobin_g_dl"),
                  ("h8_plt", "platelets_k_ul"),
                  ("h8_anc", "anc_k_ul")]:
    m = logistic_main(f"{out} ~ {var}")
    ana8.append({
        "hypothesis_ids": [hid],
        "code": f"smf.logit('{out} ~ {var}', data=df).fit()",
        "result_summary": f"Log-OR per unit {var} = {m.params[var]:+.5g} (p={m.pvalues[var]:.3g}).",
        "p_value": float(m.pvalues[var]),
        "effect_estimate": float(m.params[var]),
        "significant": bool(m.pvalues[var] < 0.05),
    })
add_iter(8, hyps8, ana8)


# ============================================================
# Iteration 9: inflammation / nutrition / symptoms
# ============================================================
hyps9 = [
    {"id": "h9_crp", "text": "Higher `crp_mg_l` is associated with lower `objective_response` rate."},
    {"id": "h9_nlr", "text": "Higher `nlr` (neutrophil-lymphocyte ratio) is associated with lower `objective_response` rate."},
    {"id": "h9_wtloss", "text": "Higher `weight_loss_pct_6mo` is associated with lower `objective_response` rate."},
    {"id": "h9_fatigue", "text": "Higher `fatigue_grade` is associated with lower `objective_response` rate."},
    {"id": "h9_pain", "text": "Higher `pain_nrs` is associated with lower `objective_response` rate."},
    {"id": "h9_appetite", "text": "Higher `appetite_loss_grade` is associated with lower `objective_response` rate."},
]
ana9 = []
for hid, var in [("h9_crp", "crp_mg_l"),
                  ("h9_nlr", "nlr"),
                  ("h9_wtloss", "weight_loss_pct_6mo"),
                  ("h9_fatigue", "fatigue_grade"),
                  ("h9_pain", "pain_nrs"),
                  ("h9_appetite", "appetite_loss_grade")]:
    m = logistic_main(f"{out} ~ {var}")
    ana9.append({
        "hypothesis_ids": [hid],
        "code": f"smf.logit('{out} ~ {var}', data=df).fit()",
        "result_summary": f"Log-OR per unit {var} = {m.params[var]:+.5g} (p={m.pvalues[var]:.3g}).",
        "p_value": float(m.pvalues[var]),
        "effect_estimate": float(m.params[var]),
        "significant": bool(m.pvalues[var] < 0.05),
    })
add_iter(9, hyps9, ana9)


# ============================================================
# Iteration 10: organ function (hepatic / renal / electrolytes)
# ============================================================
hyps10 = [
    {"id": "h10_bili", "text": "Higher `total_bilirubin_mg_dl` is associated with lower `objective_response` rate."},
    {"id": "h10_creat", "text": "Higher `creatinine_mg_dl` is associated with lower `objective_response` rate."},
    {"id": "h10_inr", "text": "Higher `inr` is associated with lower `objective_response` rate."},
    {"id": "h10_ast", "text": "Higher `ast_u_l` is associated with lower `objective_response` rate."},
    {"id": "h10_alt", "text": "Higher `alt_u_l` is associated with lower `objective_response` rate."},
    {"id": "h10_alkp", "text": "Higher `alkaline_phosphatase_u_l` is associated with lower `objective_response` rate."},
]
ana10 = []
for hid, var in [("h10_bili", "total_bilirubin_mg_dl"),
                  ("h10_creat", "creatinine_mg_dl"),
                  ("h10_inr", "inr"),
                  ("h10_ast", "ast_u_l"),
                  ("h10_alt", "alt_u_l"),
                  ("h10_alkp", "alkaline_phosphatase_u_l")]:
    m = logistic_main(f"{out} ~ {var}")
    ana10.append({
        "hypothesis_ids": [hid],
        "code": f"smf.logit('{out} ~ {var}', data=df).fit()",
        "result_summary": f"Log-OR per unit {var} = {m.params[var]:+.5g} (p={m.pvalues[var]:.3g}).",
        "p_value": float(m.pvalues[var]),
        "effect_estimate": float(m.params[var]),
        "significant": bool(m.pvalues[var] < 0.05),
    })
add_iter(10, hyps10, ana10)


# ============================================================
# Iteration 11: comorbidities main effects
# ============================================================
hyps11 = [
    {"id": "h11_dm", "text": "`diabetes_mellitus`=1 patients have a different `objective_response` rate than non-diabetic patients."},
    {"id": "h11_htn", "text": "`hypertension`=1 patients have a different `objective_response` rate than non-hypertensive patients."},
    {"id": "h11_ckd", "text": "`chronic_kidney_disease`=1 patients have a lower `objective_response` rate than CKD-negative patients."},
    {"id": "h11_chf", "text": "`heart_failure`=1 patients have a lower `objective_response` rate than CHF-negative patients."},
    {"id": "h11_priormalig", "text": "`prior_malignancy`=1 patients have a lower `objective_response` rate than those without prior malignancy."},
    {"id": "h11_copd", "text": "`copd`=1 patients have a lower `objective_response` rate than COPD-negative patients."},
]
ana11 = []
for hid, var in [("h11_dm", "diabetes_mellitus"),
                  ("h11_htn", "hypertension"),
                  ("h11_ckd", "chronic_kidney_disease"),
                  ("h11_chf", "heart_failure"),
                  ("h11_priormalig", "prior_malignancy"),
                  ("h11_copd", "copd")]:
    diff, pv, na, nb = two_prop_test(df, var, 1, 0)
    ana11.append({
        "hypothesis_ids": [hid],
        "code": f"two-prop z-test on {out} by {var}",
        "result_summary": f"{var}+: {df.loc[df[var]==1,out].mean():.3f} (n={na}); {var}-: {df.loc[df[var]==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(diff),
        "significant": bool(pv < 0.05),
    })
add_iter(11, hyps11, ana11)


# ============================================================
# Iteration 12: prior therapy
# ============================================================
hyps12 = [
    {"id": "h12_priorchemo", "text": "`prior_chemotherapy`=1 patients have a lower `objective_response` rate than chemo-naive patients."},
    {"id": "h12_priorlines", "text": "Higher `prior_lines_of_therapy` is associated with lower `objective_response` rate."},
    {"id": "h12_yrsdx", "text": "Longer `years_since_diagnosis` is associated with lower `objective_response` rate."},
    {"id": "h12_priorrad", "text": "`prior_radiation`=1 patients have a different `objective_response` rate than rad-naive patients."},
    {"id": "h12_priortarg", "text": "`prior_targeted_therapy`=1 patients have a different `objective_response` rate."},
]
ana12 = []
diff, pv, na, nb = two_prop_test(df, "prior_chemotherapy", 1, 0)
ana12.append({
    "hypothesis_ids": ["h12_priorchemo"],
    "code": "two-prop z-test on objective_response by prior_chemotherapy",
    "result_summary": f"prior_chemo+: {df.loc[df.prior_chemotherapy==1,out].mean():.3f} (n={na}); prior_chemo-: {df.loc[df.prior_chemotherapy==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(diff),
    "significant": bool(pv < 0.05),
})
m = logistic_main(f"{out} ~ prior_lines_of_therapy")
ana12.append({
    "hypothesis_ids": ["h12_priorlines"],
    "code": "smf.logit('objective_response ~ prior_lines_of_therapy', data=df).fit()",
    "result_summary": f"Log-OR per prior line = {m.params['prior_lines_of_therapy']:+.5f} (p={m.pvalues['prior_lines_of_therapy']:.3g}).",
    "p_value": float(m.pvalues["prior_lines_of_therapy"]),
    "effect_estimate": float(m.params["prior_lines_of_therapy"]),
    "significant": bool(m.pvalues["prior_lines_of_therapy"] < 0.05),
})
m = logistic_main(f"{out} ~ years_since_diagnosis")
ana12.append({
    "hypothesis_ids": ["h12_yrsdx"],
    "code": "smf.logit('objective_response ~ years_since_diagnosis', data=df).fit()",
    "result_summary": f"Log-OR per year since dx = {m.params['years_since_diagnosis']:+.5f} (p={m.pvalues['years_since_diagnosis']:.3g}).",
    "p_value": float(m.pvalues["years_since_diagnosis"]),
    "effect_estimate": float(m.params["years_since_diagnosis"]),
    "significant": bool(m.pvalues["years_since_diagnosis"] < 0.05),
})
for hid, var in [("h12_priorrad", "prior_radiation"), ("h12_priortarg", "prior_targeted_therapy")]:
    diff, pv, na, nb = two_prop_test(df, var, 1, 0)
    ana12.append({
        "hypothesis_ids": [hid],
        "code": f"two-prop z-test on {out} by {var}",
        "result_summary": f"{var}+: {df.loc[df[var]==1,out].mean():.3f} (n={na}); {var}-: {df.loc[df[var]==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(diff),
        "significant": bool(pv < 0.05),
    })
add_iter(12, hyps12, ana12)


# ============================================================
# Iteration 13: refined: NPM1 + FLT3 ITD compound (favorable mod)
# ============================================================
hyps13 = [
    {"id": "h13_npm1_no_itd", "text": "Patients with `npm1_mutation`=1 AND `flt3_itd`=0 (favorable molecular class) have a higher `objective_response` rate than patients with `npm1_mutation`=0 OR `flt3_itd`=1."},
    {"id": "h13_tp53_complex", "text": "Patients with both `tp53_mutation`=1 AND `complex_karyotype`=1 (very-high-risk class) have a lower `objective_response` rate than patients without both."},
    {"id": "h13_npm1_venaza_refined", "text": "REFINED: The benefit of `treatment_venetoclax_azacitidine` on `objective_response` is concentrated in the `npm1_mutation`=1 AND `flt3_itd`=0 favorable subgroup, with a larger absolute response-rate increase than in NPM1-wild-type patients."},
]
ana13 = []
df["fav_npm1"] = ((df["npm1_mutation"] == 1) & (df["flt3_itd"] == 0)).astype(int)
df["high_risk_tp53cx"] = ((df["tp53_mutation"] == 1) & (df["complex_karyotype"] == 1)).astype(int)
diff, pv, na, nb = two_prop_test(df, "fav_npm1", 1, 0)
ana13.append({
    "hypothesis_ids": ["h13_npm1_no_itd"],
    "code": "two-prop z-test on objective_response by (npm1_mutation==1 & flt3_itd==0)",
    "result_summary": f"fav_npm1+: {df.loc[df.fav_npm1==1,out].mean():.3f} (n={na}); fav_npm1-: {df.loc[df.fav_npm1==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(diff),
    "significant": bool(pv < 0.05),
})
diff, pv, na, nb = two_prop_test(df, "high_risk_tp53cx", 1, 0)
ana13.append({
    "hypothesis_ids": ["h13_tp53_complex"],
    "code": "two-prop z-test on objective_response by (tp53==1 & complex_karyotype==1)",
    "result_summary": f"high_risk+: {df.loc[df.high_risk_tp53cx==1,out].mean():.3f} (n={na}); high_risk-: {df.loc[df.high_risk_tp53cx==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(diff),
    "significant": bool(pv < 0.05),
})
m, coef, pv = interaction_logit("treatment_venetoclax_azacitidine", "fav_npm1")
tx, mut = "treatment_venetoclax_azacitidine", "fav_npm1"
a = df.loc[(df[tx]==1)&(df[mut]==1),out].mean()
c = df.loc[(df[tx]==0)&(df[mut]==1),out].mean()
b = df.loc[(df[tx]==1)&(df[mut]==0),out].mean()
d = df.loc[(df[tx]==0)&(df[mut]==0),out].mean()
n_aa = ((df[tx]==1)&(df[mut]==1)).sum()
ana13.append({
    "hypothesis_ids": ["h13_npm1_venaza_refined"],
    "code": f"smf.logit('{out} ~ {tx}*{mut}', data=df).fit()",
    "result_summary": f"Cells: ven+ fav+={a:.3f} (n={n_aa}); ven- fav+={c:.3f}; ven+ fav-={b:.3f}; ven- fav-={d:.3f}. Tx effect in fav+ = {a-c:+.4f}; in fav- = {b-d:+.4f}. Interaction log-OR = {coef:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float((a - c) - (b - d)),  # diff-of-diffs
    "significant": bool(pv < 0.05),
})
add_iter(13, hyps13, ana13)


# ============================================================
# Iteration 14: pharmacogenomic SNPs main effects
# ============================================================
snp_cols = [c for c in df.columns if c.startswith("snp_")]
hyps14 = [
    {"id": f"h14_{snp}", "text": f"Carriers (`{snp}` ≥ 1) have a different `objective_response` rate than non-carriers (`{snp}` = 0)."}
    for snp in snp_cols[:8]
]
ana14 = []
for hyp, snp in zip(hyps14, snp_cols[:8]):
    df["__snp_carrier"] = (df[snp] >= 1).astype(int)
    diff, pv, na, nb = two_prop_test(df, "__snp_carrier", 1, 0)
    ana14.append({
        "hypothesis_ids": [hyp["id"]],
        "code": f"two-prop z-test on {out} by ({snp} >= 1)",
        "result_summary": f"{snp} carrier+: {df.loc[df.__snp_carrier==1,out].mean():.3f} (n={na}); carrier-: {df.loc[df.__snp_carrier==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(diff),
        "significant": bool(pv < 0.05),
    })
df = df.drop(columns="__snp_carrier")
add_iter(14, hyps14, ana14)


# ============================================================
# Iteration 15: more SNPs
# ============================================================
hyps15 = [
    {"id": f"h15_{snp}", "text": f"Carriers (`{snp}` ≥ 1) have a different `objective_response` rate than non-carriers."}
    for snp in snp_cols[8:16]
]
ana15 = []
for hyp, snp in zip(hyps15, snp_cols[8:16]):
    df["__snp_carrier"] = (df[snp] >= 1).astype(int)
    diff, pv, na, nb = two_prop_test(df, "__snp_carrier", 1, 0)
    ana15.append({
        "hypothesis_ids": [hyp["id"]],
        "code": f"two-prop z-test on {out} by ({snp} >= 1)",
        "result_summary": f"{snp} carrier+: {df.loc[df.__snp_carrier==1,out].mean():.3f}; carrier-: {df.loc[df.__snp_carrier==0,out].mean():.3f}; diff={diff:+.4f}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(diff),
        "significant": bool(pv < 0.05),
    })
df = df.drop(columns="__snp_carrier")
add_iter(15, hyps15, ana15)


# ============================================================
# Iteration 16: more SNPs
# ============================================================
hyps16 = [
    {"id": f"h16_{snp}", "text": f"Carriers (`{snp}` ≥ 1) have a different `objective_response` rate than non-carriers."}
    for snp in snp_cols[16:]
]
ana16 = []
for hyp, snp in zip(hyps16, snp_cols[16:]):
    df["__snp_carrier"] = (df[snp] >= 1).astype(int)
    diff, pv, na, nb = two_prop_test(df, "__snp_carrier", 1, 0)
    ana16.append({
        "hypothesis_ids": [hyp["id"]],
        "code": f"two-prop z-test on {out} by ({snp} >= 1)",
        "result_summary": f"{snp} carrier+: {df.loc[df.__snp_carrier==1,out].mean():.3f}; carrier-: {df.loc[df.__snp_carrier==0,out].mean():.3f}; diff={diff:+.4f}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(diff),
        "significant": bool(pv < 0.05),
    })
df = df.drop(columns="__snp_carrier")
add_iter(16, hyps16, ana16)


# ============================================================
# Iteration 17: race / insurance / rural disparities
# ============================================================
hyps17 = [
    {"id": "h17_race", "text": "`objective_response` rate differs across `race_ethnicity` categories (omnibus chi-square)."},
    {"id": "h17_insurance", "text": "`objective_response` rate differs across `insurance_type` categories (omnibus chi-square)."},
    {"id": "h17_rural", "text": "Patients with `rural_residence`=1 have a different `objective_response` rate than urban residents."},
    {"id": "h17_education", "text": "Higher `education_years` is associated with a different `objective_response` rate."},
    {"id": "h17_smoking", "text": "Higher `smoking_pack_years` is associated with a different `objective_response` rate."},
]
ana17 = []
ct = pd.crosstab(df["race_ethnicity"], df[out])
chi2, p, dof, _ = stats.chi2_contingency(ct)
rates = df.groupby("race_ethnicity")[out].mean().to_dict()
ana17.append({
    "hypothesis_ids": ["h17_race"],
    "code": "stats.chi2_contingency(pd.crosstab(df.race_ethnicity, df.objective_response))",
    "result_summary": f"Response rates by race: {rates}. Chi-square ({dof} dof) = {chi2:.2f}, p={p:.3g}.",
    "p_value": float(p),
    "effect_estimate": float(max(rates.values()) - min(rates.values())),
    "significant": bool(p < 0.05),
})
ct = pd.crosstab(df["insurance_type"], df[out])
chi2, p, dof, _ = stats.chi2_contingency(ct)
rates = df.groupby("insurance_type")[out].mean().to_dict()
ana17.append({
    "hypothesis_ids": ["h17_insurance"],
    "code": "stats.chi2_contingency(pd.crosstab(df.insurance_type, df.objective_response))",
    "result_summary": f"Response rates by insurance: {rates}. Chi-square ({dof} dof) = {chi2:.2f}, p={p:.3g}.",
    "p_value": float(p),
    "effect_estimate": float(max(rates.values()) - min(rates.values())),
    "significant": bool(p < 0.05),
})
diff, pv, na, nb = two_prop_test(df, "rural_residence", 1, 0)
ana17.append({
    "hypothesis_ids": ["h17_rural"],
    "code": "two-prop z-test on objective_response by rural_residence",
    "result_summary": f"rural+: {df.loc[df.rural_residence==1,out].mean():.3f} (n={na}); rural-: {df.loc[df.rural_residence==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(diff),
    "significant": bool(pv < 0.05),
})
m = logistic_main(f"{out} ~ education_years")
ana17.append({
    "hypothesis_ids": ["h17_education"],
    "code": "smf.logit('objective_response ~ education_years', data=df).fit()",
    "result_summary": f"Log-OR per year of education = {m.params['education_years']:+.5f} (p={m.pvalues['education_years']:.3g}).",
    "p_value": float(m.pvalues["education_years"]),
    "effect_estimate": float(m.params["education_years"]),
    "significant": bool(m.pvalues["education_years"] < 0.05),
})
m = logistic_main(f"{out} ~ smoking_pack_years")
ana17.append({
    "hypothesis_ids": ["h17_smoking"],
    "code": "smf.logit('objective_response ~ smoking_pack_years', data=df).fit()",
    "result_summary": f"Log-OR per pack-year = {m.params['smoking_pack_years']:+.5f} (p={m.pvalues['smoking_pack_years']:.3g}).",
    "p_value": float(m.pvalues["smoking_pack_years"]),
    "effect_estimate": float(m.params["smoking_pack_years"]),
    "significant": bool(m.pvalues["smoking_pack_years"] < 0.05),
})
add_iter(17, hyps17, ana17)


# ============================================================
# Iteration 18: vital signs and miscellaneous
# ============================================================
hyps18 = [
    {"id": "h18_bmi", "text": "Higher `bmi` is associated with a different `objective_response` rate."},
    {"id": "h18_sbp", "text": "Higher `systolic_bp_mmhg` is associated with a different `objective_response` rate."},
    {"id": "h18_hr", "text": "Higher `heart_rate_bpm` is associated with a different `objective_response` rate."},
    {"id": "h18_spo2", "text": "Lower `spo2_pct` is associated with lower `objective_response` rate."},
    {"id": "h18_dyspnea", "text": "Higher `dyspnea_grade` is associated with lower `objective_response` rate."},
]
ana18 = []
for hid, var in [("h18_bmi", "bmi"),
                  ("h18_sbp", "systolic_bp_mmhg"),
                  ("h18_hr", "heart_rate_bpm"),
                  ("h18_spo2", "spo2_pct"),
                  ("h18_dyspnea", "dyspnea_grade")]:
    m = logistic_main(f"{out} ~ {var}")
    ana18.append({
        "hypothesis_ids": [hid],
        "code": f"smf.logit('{out} ~ {var}', data=df).fit()",
        "result_summary": f"Log-OR per unit {var} = {m.params[var]:+.5g} (p={m.pvalues[var]:.3g}).",
        "p_value": float(m.pvalues[var]),
        "effect_estimate": float(m.params[var]),
        "significant": bool(m.pvalues[var] < 0.05),
    })
add_iter(18, hyps18, ana18)


# ============================================================
# Iteration 19: irrelevant non-AML markers — control hypotheses
# ============================================================
hyps19 = [
    {"id": "h19_ca125", "text": "Higher `ca_125_u_ml` is associated with a different `objective_response` rate (control marker, not expected in AML)."},
    {"id": "h19_cea", "text": "Higher `cea_ng_ml` is associated with a different `objective_response` rate (control marker)."},
    {"id": "h19_psa", "text": "Higher `psa_ng_ml` is associated with a different `objective_response` rate (control marker)."},
    {"id": "h19_tsh", "text": "Higher `tsh_uiu_ml` is associated with a different `objective_response` rate."},
    {"id": "h19_her2", "text": "`her2_amplification`=1 patients have a different `objective_response` rate (control feature, not expected in AML)."},
    {"id": "h19_braf", "text": "`braf_v600e`=1 patients have a different `objective_response` rate (control feature)."},
]
ana19 = []
for hid, var in [("h19_ca125", "ca_125_u_ml"),
                  ("h19_cea", "cea_ng_ml"),
                  ("h19_psa", "psa_ng_ml"),
                  ("h19_tsh", "tsh_uiu_ml")]:
    m = logistic_main(f"{out} ~ {var}")
    ana19.append({
        "hypothesis_ids": [hid],
        "code": f"smf.logit('{out} ~ {var}', data=df).fit()",
        "result_summary": f"Log-OR per unit {var} = {m.params[var]:+.5g} (p={m.pvalues[var]:.3g}).",
        "p_value": float(m.pvalues[var]),
        "effect_estimate": float(m.params[var]),
        "significant": bool(m.pvalues[var] < 0.05),
    })
for hid, var in [("h19_her2", "her2_amplification"), ("h19_braf", "braf_v600e")]:
    diff, pv, na, nb = two_prop_test(df, var, 1, 0)
    ana19.append({
        "hypothesis_ids": [hid],
        "code": f"two-prop z-test on {out} by {var}",
        "result_summary": f"{var}+: {df.loc[df[var]==1,out].mean():.3f} (n={na}); {var}-: {df.loc[df[var]==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(diff),
        "significant": bool(pv < 0.05),
    })
add_iter(19, hyps19, ana19)


# ============================================================
# Iteration 20: refined: enasidenib + idh1 (off-target apparent)
# ============================================================
hyps20 = [
    {"id": "h20_ena_any_idh", "text": "REFINED: Among patients with any IDH mutation (`idh1_mutation`=1 OR `idh2_mutation`=1), `treatment_enasidenib`=1 is associated with a higher `objective_response` rate."},
    {"id": "h20_ivo_any_idh", "text": "REFINED: Among patients with any IDH mutation, `treatment_ivosidenib`=1 is associated with a different `objective_response` rate."},
    {"id": "h20_idh_inhib_any", "text": "REFINED: Receipt of any IDH inhibitor (`treatment_ivosidenib`=1 OR `treatment_enasidenib`=1) within IDH-mutant patients is associated with a different `objective_response` rate than no IDH inhibitor."},
]
ana20 = []
df["any_idh"] = ((df["idh1_mutation"] == 1) | (df["idh2_mutation"] == 1)).astype(int)
df["any_idh_inhib"] = ((df["treatment_ivosidenib"] == 1) | (df["treatment_enasidenib"] == 1)).astype(int)
m, coef, pv = interaction_logit("treatment_enasidenib", "any_idh")
tx, mut = "treatment_enasidenib", "any_idh"
a = df.loc[(df[tx]==1)&(df[mut]==1),out].mean()
c = df.loc[(df[tx]==0)&(df[mut]==1),out].mean()
n_aa = ((df[tx]==1)&(df[mut]==1)).sum()
ana20.append({
    "hypothesis_ids": ["h20_ena_any_idh"],
    "code": f"smf.logit('{out} ~ {tx}*{mut}', data=df).fit()",
    "result_summary": f"any_idh+ ena+={a:.3f} (n={n_aa}); any_idh+ ena-={c:.3f}. Tx effect in IDH+ = {a-c:+.4f}. Interaction p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(a - c),
    "significant": bool(pv < 0.05),
})
m, coef, pv = interaction_logit("treatment_ivosidenib", "any_idh")
tx, mut = "treatment_ivosidenib", "any_idh"
a = df.loc[(df[tx]==1)&(df[mut]==1),out].mean()
c = df.loc[(df[tx]==0)&(df[mut]==1),out].mean()
n_aa = ((df[tx]==1)&(df[mut]==1)).sum()
ana20.append({
    "hypothesis_ids": ["h20_ivo_any_idh"],
    "code": f"smf.logit('{out} ~ {tx}*{mut}', data=df).fit()",
    "result_summary": f"any_idh+ ivo+={a:.3f} (n={n_aa}); any_idh+ ivo-={c:.3f}. Tx effect in IDH+ = {a-c:+.4f}. Interaction p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(a - c),
    "significant": bool(pv < 0.05),
})
m, coef, pv = interaction_logit("any_idh_inhib", "any_idh")
tx, mut = "any_idh_inhib", "any_idh"
a = df.loc[(df[tx]==1)&(df[mut]==1),out].mean()
c = df.loc[(df[tx]==0)&(df[mut]==1),out].mean()
n_aa = ((df[tx]==1)&(df[mut]==1)).sum()
ana20.append({
    "hypothesis_ids": ["h20_idh_inhib_any"],
    "code": "smf.logit('objective_response ~ any_idh_inhib*any_idh', data=df).fit()",
    "result_summary": f"IDH+ + IDHi+={a:.3f} (n={n_aa}); IDH+ + IDHi-={c:.3f}. Tx effect in IDH+ = {a-c:+.4f}. Interaction p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(a - c),
    "significant": bool(pv < 0.05),
})
add_iter(20, hyps20, ana20)


# ============================================================
# Iteration 21: refined: NPM1 mutation status alone, ivosidenib in IDH1 (look closer)
# ============================================================
hyps21 = [
    {"id": "h21_ivo_idh1_refined", "text": "REFINED: Within `idh1_mutation`=1, `treatment_ivosidenib`=1 is associated with a LOWER `objective_response` rate than no ivosidenib (i.e., the apparent IDH1-only main-effect benefit of ~22% is NOT augmented — and may be reduced — by ivosidenib in this dataset)."},
    {"id": "h21_idh1_alone", "text": "REFINED: The IDH1-mutation-associated higher response is concentrated in IDH1+ patients NOT receiving ivosidenib."},
]
ana21 = []
m, coef, pv = interaction_logit("treatment_ivosidenib", "idh1_mutation")
tx, mut = "treatment_ivosidenib", "idh1_mutation"
a = df.loc[(df[tx]==1)&(df[mut]==1),out].mean()
b = df.loc[(df[tx]==1)&(df[mut]==0),out].mean()
c = df.loc[(df[tx]==0)&(df[mut]==1),out].mean()
d = df.loc[(df[tx]==0)&(df[mut]==0),out].mean()
n_aa = ((df[tx]==1)&(df[mut]==1)).sum()
ana21.append({
    "hypothesis_ids": ["h21_ivo_idh1_refined"],
    "code": "smf.logit('objective_response ~ treatment_ivosidenib*idh1_mutation', data=df).fit()",
    "result_summary": f"IDH1+ ivo+={a:.3f} (n={n_aa}); IDH1+ ivo-={c:.3f}; IDH1- ivo+={b:.3f}; IDH1- ivo-={d:.3f}. Tx effect in IDH1+ = {a-c:+.4f}. Interaction log-OR = {coef:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(a - c),
    "significant": bool(pv < 0.05),
})
ct = pd.crosstab(df["idh1_mutation"].astype(str) + "_ivo" + df["treatment_ivosidenib"].astype(str), df[out])
ana21.append({
    "hypothesis_ids": ["h21_idh1_alone"],
    "code": "Stratified response rates by idh1_mutation x treatment_ivosidenib",
    "result_summary": f"Response rates: IDH1+/ivo-={c:.3f}, IDH1+/ivo+={a:.3f}, IDH1-/ivo-={d:.3f}, IDH1-/ivo+={b:.3f}. Difference IDH1+ /ivo- minus overall = {c - df[out].mean():+.4f}.",
    "p_value": None,
    "effect_estimate": float(c - df[out].mean()),
    "significant": None,
})
add_iter(21, hyps21, ana21)


# ============================================================
# Iteration 22: multivariable model with key clinical features
# ============================================================
hyps22 = [
    {"id": "h22_mv_age", "text": "After adjusting for ECOG, mutations, treatment, blasts, and labs, older `age_years` remains independently associated with lower `objective_response`."},
    {"id": "h22_mv_idh1", "text": "After multivariable adjustment, `idh1_mutation`=1 remains independently associated with higher `objective_response`."},
    {"id": "h22_mv_complex", "text": "After multivariable adjustment, `complex_karyotype`=1 remains independently associated with lower `objective_response`."},
    {"id": "h22_mv_unfit", "text": "After multivariable adjustment, `unfit_for_intensive`=1 remains independently associated with lower `objective_response`."},
]
formula = (f"{out} ~ age_years + sex_female + ecog_ps + secondary_aml + unfit_for_intensive "
           "+ complex_karyotype + flt3_itd + flt3_tkd + idh1_mutation + idh2_mutation + npm1_mutation + tp53_mutation "
           "+ wbc_k_per_ul + blast_pct_marrow + albumin_g_dl + ldh_u_l "
           "+ treatment_midostaurin + treatment_gilteritinib + treatment_ivosidenib + treatment_enasidenib "
           "+ treatment_venetoclax_azacitidine + treatment_7plus3")
m_mv = smf.logit(formula, data=df).fit(disp=0)
ana22 = []
for hid, term in [("h22_mv_age", "age_years"),
                   ("h22_mv_idh1", "idh1_mutation"),
                   ("h22_mv_complex", "complex_karyotype"),
                   ("h22_mv_unfit", "unfit_for_intensive")]:
    ana22.append({
        "hypothesis_ids": [hid],
        "code": "Multivariable logistic regression with demographic, biology, lab, and treatment terms",
        "result_summary": f"Adjusted log-OR for {term} = {m_mv.params[term]:+.5g} (p={m_mv.pvalues[term]:.3g}).",
        "p_value": float(m_mv.pvalues[term]),
        "effect_estimate": float(m_mv.params[term]),
        "significant": bool(m_mv.pvalues[term] < 0.05),
    })
add_iter(22, hyps22, ana22)


# ============================================================
# Iteration 23: refined - older + unfit + venaza
# ============================================================
hyps23 = [
    {"id": "h23_old_unfit_venaza", "text": "REFINED: Among older (`age_years`>=70) and `unfit_for_intensive`=1 patients, `treatment_venetoclax_azacitidine`=1 yields a higher `objective_response` rate than other regimens."},
    {"id": "h23_young_fit_7p3", "text": "REFINED: Among younger (`age_years`<60) and `unfit_for_intensive`=0 patients, `treatment_7plus3`=1 yields a higher `objective_response` rate than no 7+3."},
    {"id": "h23_age_venaza_interact", "text": "REFINED: The benefit of `treatment_venetoclax_azacitidine` on `objective_response` increases with `age_years` (positive interaction)."},
]
ana23 = []
sub = df[(df["age_years"] >= 70) & (df["unfit_for_intensive"] == 1)].copy()
diff, pv, na, nb = two_prop_test(sub, "treatment_venetoclax_azacitidine", 1, 0)
ana23.append({
    "hypothesis_ids": ["h23_old_unfit_venaza"],
    "code": "two-prop z-test in subset (age>=70 & unfit) on objective_response by treatment_venetoclax_azacitidine",
    "result_summary": f"Older+unfit subset (n={len(sub)}): VenAza+ {sub.loc[sub.treatment_venetoclax_azacitidine==1,out].mean():.3f} (n={na}) vs VenAza- {sub.loc[sub.treatment_venetoclax_azacitidine==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(diff),
    "significant": bool(pv < 0.05),
})
sub = df[(df["age_years"] < 60) & (df["unfit_for_intensive"] == 0)].copy()
diff, pv, na, nb = two_prop_test(sub, "treatment_7plus3", 1, 0)
ana23.append({
    "hypothesis_ids": ["h23_young_fit_7p3"],
    "code": "two-prop z-test in subset (age<60 & fit) on objective_response by treatment_7plus3",
    "result_summary": f"Younger+fit subset (n={len(sub)}): 7+3+ {sub.loc[sub.treatment_7plus3==1,out].mean():.3f} (n={na}) vs 7+3- {sub.loc[sub.treatment_7plus3==0,out].mean():.3f} (n={nb}); diff={diff:+.4f}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(diff),
    "significant": bool(pv < 0.05),
})
m = smf.logit(f"{out} ~ treatment_venetoclax_azacitidine * age_years", data=df).fit(disp=0)
coef = m.params["treatment_venetoclax_azacitidine:age_years"]
pv = m.pvalues["treatment_venetoclax_azacitidine:age_years"]
ana23.append({
    "hypothesis_ids": ["h23_age_venaza_interact"],
    "code": "smf.logit('objective_response ~ treatment_venetoclax_azacitidine * age_years', data=df).fit()",
    "result_summary": f"Interaction log-OR per year (venaza x age) = {coef:+.5g}, p={pv:.3g}.",
    "p_value": float(pv),
    "effect_estimate": float(coef),
    "significant": bool(pv < 0.05),
})
add_iter(23, hyps23, ana23)


# ============================================================
# Iteration 24: SNP x treatment interactions (pharmacogenomics)
# ============================================================
hyps24 = [
    {"id": "h24_snp_cyp2c19", "text": "Among `snp_rs4244285` (CYP2C19*2) carriers, `treatment_venetoclax_azacitidine`=1 has a different effect on `objective_response` than in non-carriers (gene-drug interaction)."},
    {"id": "h24_snp_apoe", "text": "Among `snp_rs429358` (APOE) carriers, `treatment_7plus3`=1 has a different effect on `objective_response` than in non-carriers."},
    {"id": "h24_snp_mthfr", "text": "Among `snp_rs1801133` (MTHFR C677T) carriers, `treatment_venetoclax_azacitidine`=1 has a different effect on `objective_response`."},
]
ana24 = []
for hid, snp, tx in [("h24_snp_cyp2c19", "snp_rs4244285", "treatment_venetoclax_azacitidine"),
                      ("h24_snp_apoe", "snp_rs429358", "treatment_7plus3"),
                      ("h24_snp_mthfr", "snp_rs1801133", "treatment_venetoclax_azacitidine")]:
    df["__c"] = (df[snp] >= 1).astype(int)
    m = smf.logit(f"{out} ~ {tx} * __c", data=df).fit(disp=0)
    coef = m.params[f"{tx}:__c"]
    pv = m.pvalues[f"{tx}:__c"]
    ana24.append({
        "hypothesis_ids": [hid],
        "code": f"smf.logit('{out} ~ {tx} * (snp>=1)', data=df).fit()",
        "result_summary": f"Interaction log-OR ({tx} x {snp} carrier) = {coef:+.5g}, p={pv:.3g}.",
        "p_value": float(pv),
        "effect_estimate": float(coef),
        "significant": bool(pv < 0.05),
    })
df = df.drop(columns="__c")
add_iter(24, hyps24, ana24)


# ============================================================
# Iteration 25: final synthesis - multivariable with interactions
# ============================================================
hyps25 = [
    {"id": "h25_full_idh1_main", "text": "FINAL: After multivariable adjustment in a model that includes treatment-mutation interaction terms, `idh1_mutation` retains an independent positive association with `objective_response` (regardless of treatment received)."},
    {"id": "h25_venaza_npm1_intx", "text": "FINAL: After multivariable adjustment in a full model with treatment-mutation interactions, the `treatment_venetoclax_azacitidine`:`npm1_mutation` interaction term is positive and statistically significant, indicating venetoclax/azacitidine benefits NPM1-mutant patients more than NPM1-wild-type patients."},
    {"id": "h25_unfit_main", "text": "FINAL: After multivariable adjustment with treatment-mutation interactions, `unfit_for_intensive`=1 remains independently associated with lower `objective_response`."},
    {"id": "h25_fav_npm1_main", "text": "FINAL: After multivariable adjustment with treatment-mutation interactions, the favorable molecular class (NPM1+ / FLT3-ITD-) remains positively associated with `objective_response`."},
]
formula_full = (
    f"{out} ~ age_years + sex_female + ecog_ps + secondary_aml + unfit_for_intensive "
    "+ complex_karyotype + flt3_itd + flt3_tkd + idh1_mutation + idh2_mutation + npm1_mutation + tp53_mutation "
    "+ wbc_k_per_ul + blast_pct_marrow + albumin_g_dl + ldh_u_l "
    "+ treatment_midostaurin*flt3_itd + treatment_gilteritinib*flt3_itd "
    "+ treatment_ivosidenib*idh1_mutation + treatment_enasidenib*idh2_mutation "
    "+ treatment_venetoclax_azacitidine*npm1_mutation + treatment_7plus3"
)
m_full = smf.logit(formula_full, data=df).fit(disp=0)
ana25 = []
for hid, term in [("h25_full_idh1_main", "idh1_mutation"),
                   ("h25_venaza_npm1_intx", "treatment_venetoclax_azacitidine:npm1_mutation"),
                   ("h25_unfit_main", "unfit_for_intensive")]:
    if term in m_full.params:
        ana25.append({
            "hypothesis_ids": [hid],
            "code": "Multivariable logistic with treatment-mutation interactions",
            "result_summary": f"Adjusted log-OR for {term} = {m_full.params[term]:+.5g} (p={m_full.pvalues[term]:.3g}).",
            "p_value": float(m_full.pvalues[term]),
            "effect_estimate": float(m_full.params[term]),
            "significant": bool(m_full.pvalues[term] < 0.05),
        })
# fav_npm1
df["fav_npm1"] = ((df["npm1_mutation"] == 1) & (df["flt3_itd"] == 0)).astype(int)
formula_fav = (f"{out} ~ age_years + sex_female + ecog_ps + secondary_aml + unfit_for_intensive "
               "+ complex_karyotype + tp53_mutation + idh1_mutation + idh2_mutation + fav_npm1 "
               "+ wbc_k_per_ul + blast_pct_marrow + albumin_g_dl + ldh_u_l "
               "+ treatment_midostaurin + treatment_gilteritinib + treatment_ivosidenib + treatment_enasidenib "
               "+ treatment_venetoclax_azacitidine + treatment_7plus3")
m_fav = smf.logit(formula_fav, data=df).fit(disp=0)
ana25.append({
    "hypothesis_ids": ["h25_fav_npm1_main"],
    "code": "Multivariable logistic with fav_npm1 (NPM1+/FLT3-ITD-) feature",
    "result_summary": f"Adjusted log-OR for fav_npm1 = {m_fav.params['fav_npm1']:+.5g} (p={m_fav.pvalues['fav_npm1']:.3g}).",
    "p_value": float(m_fav.pvalues["fav_npm1"]),
    "effect_estimate": float(m_fav.params["fav_npm1"]),
    "significant": bool(m_fav.pvalues["fav_npm1"] < 0.05),
})
add_iter(25, hyps25, ana25)


# Build full transcript
transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-named-bundle@2026-04-28",
    "max_iterations": 25,
    "iterations": results["iterations"],
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

# Pretty print summary of significance for narrative
print("\n=== SUMMARY ===")
for it in results["iterations"]:
    print(f"\n--- Iter {it['index']} ---")
    for h in it["proposed_hypotheses"]:
        # find matching analyses
        relevant = [a for a in it["analyses"] if h["id"] in a["hypothesis_ids"]]
        for a in relevant:
            sig = "SIG" if a.get("significant") else "ns"
            est = a.get("effect_estimate")
            pv = a.get("p_value")
            est_s = f"{est:+.4g}" if est is not None else "NA"
            pv_s = f"{pv:.3g}" if pv is not None else "NA"
            print(f"  [{sig}] {h['id']}: est={est_s}, p={pv_s} -- {h['text'][:100]}")
print("Saved transcript.json")
