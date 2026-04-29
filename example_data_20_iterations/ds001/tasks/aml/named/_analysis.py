"""Comprehensive analysis script for ds001_aml.

Runs all the statistical tests for each iteration and stores the
result (effect estimate, p-value, summary) for downstream transcript
construction.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import logit, ols

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
N = len(df)
results = {}


def _logit_or(formula, data, focal=None):
    """Fit a logistic regression and return the OR/beta/p for `focal` term.

    Returns (beta, OR, p, summary_str).
    """
    model = logit(formula, data=data).fit(disp=False, maxiter=200)
    if focal is None:
        focal = list(model.params.index)[1]  # first non-intercept
    beta = float(model.params[focal])
    p = float(model.pvalues[focal])
    or_ = float(np.exp(beta))
    return beta, or_, p, model


def _orr_diff(mask_pos, mask_neg, name_pos="pos", name_neg="neg", outcome="objective_response"):
    """Return ORR difference, p (chi2), and a summary string."""
    pos = df.loc[mask_pos, outcome]
    neg = df.loc[mask_neg, outcome]
    p_orr = pos.mean()
    n_orr = neg.mean()
    diff = float(p_orr - n_orr)
    table = pd.crosstab(mask_pos.astype(int), df[outcome])
    chi2, pval, _, _ = stats.chi2_contingency(table)
    summary = (
        f"ORR {name_pos}={p_orr:.3f} (n={mask_pos.sum()}) vs {name_neg}={n_orr:.3f} "
        f"(n={mask_neg.sum()}); diff={diff:+.3f}; chi2 p={pval:.3g}"
    )
    return diff, float(pval), summary


# ------------------------------------------------------------------
# ITERATION 1 — Treatment main effects on ORR
# ------------------------------------------------------------------
treatments = [
    "treatment_midostaurin",
    "treatment_gilteritinib",
    "treatment_ivosidenib",
    "treatment_enasidenib",
    "treatment_venetoclax_azacitidine",
    "treatment_7plus3",
]
iter1 = {}
for tx in treatments:
    diff, p, summary = _orr_diff(df[tx] == 1, df[tx] == 0, tx, f"no_{tx}")
    iter1[tx] = {"effect_estimate": diff, "p_value": p, "summary": summary}
results["iter1_tx_main_effects"] = iter1

# ------------------------------------------------------------------
# ITERATION 2 — Mutation / cytogenetics main effects on ORR
# ------------------------------------------------------------------
muts = [
    "flt3_itd",
    "flt3_tkd",
    "idh1_mutation",
    "idh2_mutation",
    "npm1_mutation",
    "tp53_mutation",
    "complex_karyotype",
    "secondary_aml",
]
iter2 = {}
for m in muts:
    diff, p, summary = _orr_diff(df[m] == 1, df[m] == 0, m, f"no_{m}")
    iter2[m] = {"effect_estimate": diff, "p_value": p, "summary": summary}
results["iter2_mutation_main_effects"] = iter2

# ------------------------------------------------------------------
# ITERATION 3 — FLT3-ITD × midostaurin interaction
# ------------------------------------------------------------------
beta, or_, p, m = _logit_or(
    "objective_response ~ flt3_itd * treatment_midostaurin",
    df,
    focal="flt3_itd:treatment_midostaurin",
)
# Stratified ORR
orr_a = df[(df.flt3_itd == 1) & (df.treatment_midostaurin == 1)].objective_response.mean()
orr_b = df[(df.flt3_itd == 1) & (df.treatment_midostaurin == 0)].objective_response.mean()
orr_c = df[(df.flt3_itd == 0) & (df.treatment_midostaurin == 1)].objective_response.mean()
orr_d = df[(df.flt3_itd == 0) & (df.treatment_midostaurin == 0)].objective_response.mean()
results["iter3_flt3itd_x_midostaurin"] = {
    "interaction_beta": beta,
    "interaction_OR": or_,
    "interaction_p": p,
    "stratified": {
        "FLT3+/mido+": orr_a, "FLT3+/mido-": orr_b,
        "FLT3-/mido+": orr_c, "FLT3-/mido-": orr_d,
    },
    "summary": (
        f"FLT3-ITD × midostaurin: interaction beta={beta:+.3f} (OR={or_:.2f}, p={p:.3g}). "
        f"ORR FLT3+/mido+={orr_a:.3f}, FLT3+/mido-={orr_b:.3f}, "
        f"FLT3-/mido+={orr_c:.3f}, FLT3-/mido-={orr_d:.3f}"
    ),
}

# ------------------------------------------------------------------
# ITERATION 4 — FLT3-ITD × gilteritinib
# ------------------------------------------------------------------
beta, or_, p, _ = _logit_or(
    "objective_response ~ flt3_itd * treatment_gilteritinib",
    df,
    focal="flt3_itd:treatment_gilteritinib",
)
orr_a = df[(df.flt3_itd == 1) & (df.treatment_gilteritinib == 1)].objective_response.mean()
orr_b = df[(df.flt3_itd == 1) & (df.treatment_gilteritinib == 0)].objective_response.mean()
orr_c = df[(df.flt3_itd == 0) & (df.treatment_gilteritinib == 1)].objective_response.mean()
orr_d = df[(df.flt3_itd == 0) & (df.treatment_gilteritinib == 0)].objective_response.mean()
results["iter4_flt3itd_x_gilteritinib"] = {
    "interaction_beta": beta, "interaction_OR": or_, "interaction_p": p,
    "stratified": {"FLT3+/gilt+": orr_a, "FLT3+/gilt-": orr_b,
                   "FLT3-/gilt+": orr_c, "FLT3-/gilt-": orr_d},
    "summary": (
        f"FLT3-ITD × gilteritinib: beta={beta:+.3f} (OR={or_:.2f}, p={p:.3g}). "
        f"ORR FLT3+/gilt+={orr_a:.3f}, FLT3+/gilt-={orr_b:.3f}, "
        f"FLT3-/gilt+={orr_c:.3f}, FLT3-/gilt-={orr_d:.3f}"
    ),
}

# ------------------------------------------------------------------
# ITERATION 5 — IDH1 × ivosidenib
# ------------------------------------------------------------------
beta, or_, p, _ = _logit_or(
    "objective_response ~ idh1_mutation * treatment_ivosidenib",
    df,
    focal="idh1_mutation:treatment_ivosidenib",
)
orr_a = df[(df.idh1_mutation == 1) & (df.treatment_ivosidenib == 1)].objective_response.mean()
orr_b = df[(df.idh1_mutation == 1) & (df.treatment_ivosidenib == 0)].objective_response.mean()
orr_c = df[(df.idh1_mutation == 0) & (df.treatment_ivosidenib == 1)].objective_response.mean()
orr_d = df[(df.idh1_mutation == 0) & (df.treatment_ivosidenib == 0)].objective_response.mean()
results["iter5_idh1_x_ivosidenib"] = {
    "interaction_beta": beta, "interaction_OR": or_, "interaction_p": p,
    "stratified": {"IDH1+/ivo+": orr_a, "IDH1+/ivo-": orr_b,
                   "IDH1-/ivo+": orr_c, "IDH1-/ivo-": orr_d},
    "summary": (
        f"IDH1 × ivosidenib: beta={beta:+.3f} (OR={or_:.2f}, p={p:.3g}). "
        f"ORR IDH1+/ivo+={orr_a:.3f}, IDH1+/ivo-={orr_b:.3f}, "
        f"IDH1-/ivo+={orr_c:.3f}, IDH1-/ivo-={orr_d:.3f}"
    ),
}

# ------------------------------------------------------------------
# ITERATION 6 — IDH2 × enasidenib
# ------------------------------------------------------------------
beta, or_, p, _ = _logit_or(
    "objective_response ~ idh2_mutation * treatment_enasidenib",
    df,
    focal="idh2_mutation:treatment_enasidenib",
)
orr_a = df[(df.idh2_mutation == 1) & (df.treatment_enasidenib == 1)].objective_response.mean()
orr_b = df[(df.idh2_mutation == 1) & (df.treatment_enasidenib == 0)].objective_response.mean()
orr_c = df[(df.idh2_mutation == 0) & (df.treatment_enasidenib == 1)].objective_response.mean()
orr_d = df[(df.idh2_mutation == 0) & (df.treatment_enasidenib == 0)].objective_response.mean()
results["iter6_idh2_x_enasidenib"] = {
    "interaction_beta": beta, "interaction_OR": or_, "interaction_p": p,
    "stratified": {"IDH2+/ena+": orr_a, "IDH2+/ena-": orr_b,
                   "IDH2-/ena+": orr_c, "IDH2-/ena-": orr_d},
    "summary": (
        f"IDH2 × enasidenib: beta={beta:+.3f} (OR={or_:.2f}, p={p:.3g}). "
        f"ORR IDH2+/ena+={orr_a:.3f}, IDH2+/ena-={orr_b:.3f}, "
        f"IDH2-/ena+={orr_c:.3f}, IDH2-/ena-={orr_d:.3f}"
    ),
}

# ------------------------------------------------------------------
# ITERATION 7 — Age, ECOG, fitness on ORR (continuous + ordinal)
# ------------------------------------------------------------------
iter7 = {}
for col in ["age_years", "ecog_ps"]:
    beta, or_, p, _ = _logit_or(f"objective_response ~ {col}", df, focal=col)
    iter7[col] = {
        "beta": beta, "OR_per_unit": or_, "p_value": p,
        "summary": f"{col}: beta={beta:+.4f} (OR/unit={or_:.3f}, p={p:.3g})"
    }
diff, p, summary = _orr_diff(df.unfit_for_intensive == 1, df.unfit_for_intensive == 0,
                             "unfit", "fit")
iter7["unfit_for_intensive"] = {"effect_estimate": diff, "p_value": p, "summary": summary}
results["iter7_age_ecog_fitness"] = iter7

# ------------------------------------------------------------------
# ITERATION 8 — Disease burden: WBC, blast %, LDH
# ------------------------------------------------------------------
iter8 = {}
for col in ["wbc_k_per_ul", "blast_pct_marrow", "ldh_u_l"]:
    beta, or_, p, _ = _logit_or(f"objective_response ~ {col}", df, focal=col)
    iter8[col] = {"beta": beta, "OR_per_unit": or_, "p_value": p,
                  "summary": f"{col}: beta={beta:+.5f} (OR/unit={or_:.4f}, p={p:.3g})"}
results["iter8_disease_burden"] = iter8

# ------------------------------------------------------------------
# ITERATION 9 — Inflammation/nutrition: albumin, CRP, NLR
# ------------------------------------------------------------------
iter9 = {}
for col in ["albumin_g_dl", "crp_mg_l", "nlr"]:
    beta, or_, p, _ = _logit_or(f"objective_response ~ {col}", df, focal=col)
    iter9[col] = {"beta": beta, "OR_per_unit": or_, "p_value": p,
                  "summary": f"{col}: beta={beta:+.4f} (OR/unit={or_:.3f}, p={p:.3g})"}
results["iter9_inflammation_nutrition"] = iter9

# ------------------------------------------------------------------
# ITERATION 10 — TP53 × treatments (TP53 confers poor response to intensive tx)
# ------------------------------------------------------------------
iter10 = {}
for tx in ["treatment_7plus3", "treatment_venetoclax_azacitidine"]:
    beta, or_, p, _ = _logit_or(
        f"objective_response ~ tp53_mutation * {tx}", df,
        focal=f"tp53_mutation:{tx}",
    )
    a = df[(df.tp53_mutation == 1) & (df[tx] == 1)].objective_response.mean()
    b = df[(df.tp53_mutation == 1) & (df[tx] == 0)].objective_response.mean()
    c = df[(df.tp53_mutation == 0) & (df[tx] == 1)].objective_response.mean()
    d = df[(df.tp53_mutation == 0) & (df[tx] == 0)].objective_response.mean()
    iter10[tx] = {
        "interaction_beta": beta, "interaction_OR": or_, "interaction_p": p,
        "stratified": {"TP53+/tx+": a, "TP53+/tx-": b, "TP53-/tx+": c, "TP53-/tx-": d},
        "summary": (
            f"TP53 × {tx}: beta={beta:+.3f} (OR={or_:.2f}, p={p:.3g}). "
            f"ORR TP53+/{tx}+={a:.3f}, TP53+/{tx}-={b:.3f}, TP53-/{tx}+={c:.3f}, TP53-/{tx}-={d:.3f}"
        ),
    }
results["iter10_tp53_x_intensive"] = iter10

# ------------------------------------------------------------------
# ITERATION 11 — NPM1 favorable: explore main + interactions with 7+3
# ------------------------------------------------------------------
beta, or_, p, _ = _logit_or(
    "objective_response ~ npm1_mutation * treatment_7plus3",
    df, focal="npm1_mutation:treatment_7plus3",
)
a = df[(df.npm1_mutation == 1) & (df.treatment_7plus3 == 1)].objective_response.mean()
b = df[(df.npm1_mutation == 1) & (df.treatment_7plus3 == 0)].objective_response.mean()
c = df[(df.npm1_mutation == 0) & (df.treatment_7plus3 == 1)].objective_response.mean()
d = df[(df.npm1_mutation == 0) & (df.treatment_7plus3 == 0)].objective_response.mean()
results["iter11_npm1_x_7plus3"] = {
    "interaction_beta": beta, "interaction_OR": or_, "interaction_p": p,
    "stratified": {"NPM1+/7+3+": a, "NPM1+/7+3-": b, "NPM1-/7+3+": c, "NPM1-/7+3-": d},
    "summary": (
        f"NPM1 × 7+3: beta={beta:+.3f} (OR={or_:.2f}, p={p:.3g}). "
        f"ORR NPM1+/7+3+={a:.3f}, NPM1+/7+3-={b:.3f}, NPM1-/7+3+={c:.3f}, NPM1-/7+3-={d:.3f}"
    ),
}

# ------------------------------------------------------------------
# ITERATION 12 — Complex karyotype × treatments
# ------------------------------------------------------------------
iter12 = {}
for tx in ["treatment_7plus3", "treatment_venetoclax_azacitidine"]:
    beta, or_, p, _ = _logit_or(
        f"objective_response ~ complex_karyotype * {tx}", df,
        focal=f"complex_karyotype:{tx}",
    )
    iter12[tx] = {
        "interaction_beta": beta, "interaction_OR": or_, "interaction_p": p,
        "summary": f"complex_karyotype × {tx}: beta={beta:+.3f} (OR={or_:.2f}, p={p:.3g})",
    }
results["iter12_complex_kary_x_tx"] = iter12

# ------------------------------------------------------------------
# ITERATION 13 — Sociodemographic effects on ORR
# ------------------------------------------------------------------
iter13 = {}
# sex
diff, p, summary = _orr_diff(df.sex_female == 1, df.sex_female == 0, "female", "male")
iter13["sex_female"] = {"effect_estimate": diff, "p_value": p, "summary": summary}
# rural
diff, p, summary = _orr_diff(df.rural_residence == 1, df.rural_residence == 0, "rural", "urban")
iter13["rural_residence"] = {"effect_estimate": diff, "p_value": p, "summary": summary}
# education and smoking as continuous
for col in ["education_years", "smoking_pack_years"]:
    beta, or_, p, _ = _logit_or(f"objective_response ~ {col}", df, focal=col)
    iter13[col] = {"beta": beta, "OR_per_unit": or_, "p_value": p,
                   "summary": f"{col}: beta={beta:+.5f} (OR/unit={or_:.4f}, p={p:.3g})"}
# race/ethnicity & insurance: chi2 across categories vs ORR
for col in ["race_ethnicity", "insurance_type"]:
    table = pd.crosstab(df[col], df["objective_response"])
    chi2, pval, _, _ = stats.chi2_contingency(table)
    rates = df.groupby(col)["objective_response"].mean().to_dict()
    iter13[col] = {"chi2": float(chi2), "p_value": float(pval), "rates": rates,
                   "summary": f"{col}: chi2 p={pval:.3g}; rates={ {k: round(v,3) for k,v in rates.items()} }"}
results["iter13_sociodemographic"] = iter13

# ------------------------------------------------------------------
# ITERATION 14 — Unfit_for_intensive × venetoclax+azacitidine
# ------------------------------------------------------------------
beta, or_, p, _ = _logit_or(
    "objective_response ~ unfit_for_intensive * treatment_venetoclax_azacitidine",
    df, focal="unfit_for_intensive:treatment_venetoclax_azacitidine",
)
a = df[(df.unfit_for_intensive == 1) & (df.treatment_venetoclax_azacitidine == 1)].objective_response.mean()
b = df[(df.unfit_for_intensive == 1) & (df.treatment_venetoclax_azacitidine == 0)].objective_response.mean()
c = df[(df.unfit_for_intensive == 0) & (df.treatment_venetoclax_azacitidine == 1)].objective_response.mean()
d = df[(df.unfit_for_intensive == 0) & (df.treatment_venetoclax_azacitidine == 0)].objective_response.mean()
results["iter14_unfit_x_venaza"] = {
    "interaction_beta": beta, "interaction_OR": or_, "interaction_p": p,
    "stratified": {"unfit/venaza+": a, "unfit/venaza-": b, "fit/venaza+": c, "fit/venaza-": d},
    "summary": (
        f"unfit × venaza: beta={beta:+.3f} (OR={or_:.2f}, p={p:.3g}). "
        f"ORR unfit/venaza+={a:.3f}, unfit/venaza-={b:.3f}, fit/venaza+={c:.3f}, fit/venaza-={d:.3f}"
    ),
}

# ------------------------------------------------------------------
# ITERATION 15 — Comorbidity main effects on ORR
# ------------------------------------------------------------------
comorbs = [
    "diabetes_mellitus", "hypertension", "copd", "chronic_kidney_disease",
    "heart_failure", "coronary_artery_disease", "atrial_fibrillation",
    "venous_thromboembolism_history", "autoimmune_disease",
    "hepatitis_b_history", "hepatitis_c_history", "hiv_positive",
    "prior_malignancy", "depression_anxiety_diagnosis",
]
iter15 = {}
for c in comorbs:
    diff, p, summary = _orr_diff(df[c] == 1, df[c] == 0, c, f"no_{c}")
    iter15[c] = {"effect_estimate": diff, "p_value": p, "summary": summary}
results["iter15_comorbidities"] = iter15

# ------------------------------------------------------------------
# ITERATION 16 — SNP main effects (binary; coded 0/1/2 likely allele count)
# ------------------------------------------------------------------
snp_cols = [c for c in df.columns if c.startswith("snp_")]
iter16 = {}
for c in snp_cols:
    # If binary, use chi2; else use logistic regression
    vals = sorted(df[c].unique().tolist())
    if set(vals) <= {0, 1}:
        diff, p, summary = _orr_diff(df[c] == 1, df[c] == 0, f"{c}+", f"{c}-")
        iter16[c] = {"effect_estimate": diff, "p_value": p, "summary": summary}
    else:
        beta, or_, p, _ = _logit_or(f"objective_response ~ {c}", df, focal=c)
        iter16[c] = {"beta": beta, "OR_per_allele": or_, "p_value": p,
                     "summary": f"{c}: beta={beta:+.4f} (OR/allele={or_:.3f}, p={p:.3g})"}
results["iter16_snps"] = iter16

# ------------------------------------------------------------------
# ITERATION 17 — ECOG × treatment_7plus3 interaction (fit patients
# benefit more from intensive therapy)
# ------------------------------------------------------------------
beta, or_, p, _ = _logit_or(
    "objective_response ~ ecog_ps * treatment_7plus3",
    df, focal="ecog_ps:treatment_7plus3",
)
results["iter17_ecog_x_7plus3"] = {
    "interaction_beta": beta, "interaction_OR": or_, "interaction_p": p,
    "summary": f"ECOG × 7+3: beta={beta:+.3f} (OR={or_:.2f}, p={p:.3g})",
}

# ------------------------------------------------------------------
# ITERATION 18 — Combined ELN-like favorable / adverse risk score
# ------------------------------------------------------------------
df["fav_risk"] = ((df.npm1_mutation == 1) & (df.flt3_itd == 0)).astype(int)
df["adv_risk"] = ((df.tp53_mutation == 1) | (df.complex_karyotype == 1)).astype(int)
diff_fav, p_fav, summ_fav = _orr_diff(df.fav_risk == 1, df.fav_risk == 0, "favorable", "non-favorable")
diff_adv, p_adv, summ_adv = _orr_diff(df.adv_risk == 1, df.adv_risk == 0, "adverse", "non-adverse")
results["iter18_risk_groups"] = {
    "favorable": {"effect_estimate": diff_fav, "p_value": p_fav, "summary": summ_fav},
    "adverse": {"effect_estimate": diff_adv, "p_value": p_adv, "summary": summ_adv},
}

# ------------------------------------------------------------------
# ITERATION 19 — Multivariable logistic regression for adjusted effects
# ------------------------------------------------------------------
mv_form = (
    "objective_response ~ age_years + ecog_ps + secondary_aml + complex_karyotype + "
    "flt3_itd + flt3_tkd + idh1_mutation + idh2_mutation + npm1_mutation + tp53_mutation + "
    "wbc_k_per_ul + blast_pct_marrow + albumin_g_dl + ldh_u_l + nlr + "
    "treatment_midostaurin + treatment_gilteritinib + treatment_ivosidenib + "
    "treatment_enasidenib + treatment_venetoclax_azacitidine + treatment_7plus3"
)
mv = logit(mv_form, data=df).fit(disp=False, maxiter=500)
mv_dict = {}
for var in mv.params.index:
    if var == "Intercept":
        continue
    mv_dict[var] = {
        "beta": float(mv.params[var]),
        "OR": float(np.exp(mv.params[var])),
        "p_value": float(mv.pvalues[var]),
    }
results["iter19_multivariable"] = {
    "summary": "Multivariable logistic regression (objective_response ~ key features)",
    "coefficients": mv_dict,
}

# ------------------------------------------------------------------
# ITERATION 20 — Symptom burden on ORR
# ------------------------------------------------------------------
iter20 = {}
for col in ["fatigue_grade", "pain_nrs", "dyspnea_grade", "cough_grade", "appetite_loss_grade",
            "weight_loss_pct_6mo"]:
    beta, or_, p, _ = _logit_or(f"objective_response ~ {col}", df, focal=col)
    iter20[col] = {"beta": beta, "OR_per_unit": or_, "p_value": p,
                   "summary": f"{col}: beta={beta:+.4f} (OR/unit={or_:.3f}, p={p:.3g})"}
results["iter20_symptoms"] = iter20

# ------------------------------------------------------------------
# ITERATION 21 — Hematologic counts on ORR
# ------------------------------------------------------------------
iter21 = {}
for col in ["hemoglobin_g_dl", "platelets_k_ul", "anc_k_ul", "alc_k_ul"]:
    beta, or_, p, _ = _logit_or(f"objective_response ~ {col}", df, focal=col)
    iter21[col] = {"beta": beta, "OR_per_unit": or_, "p_value": p,
                   "summary": f"{col}: beta={beta:+.5f} (OR/unit={or_:.4f}, p={p:.3g})"}
results["iter21_blood_counts"] = iter21

# ------------------------------------------------------------------
# ITERATION 22 — FLT3-TKD × FLT3 inhibitors
# ------------------------------------------------------------------
iter22 = {}
for tx in ["treatment_midostaurin", "treatment_gilteritinib"]:
    beta, or_, p, _ = _logit_or(
        f"objective_response ~ flt3_tkd * {tx}", df,
        focal=f"flt3_tkd:{tx}",
    )
    a = df[(df.flt3_tkd == 1) & (df[tx] == 1)].objective_response.mean()
    b = df[(df.flt3_tkd == 1) & (df[tx] == 0)].objective_response.mean()
    c = df[(df.flt3_tkd == 0) & (df[tx] == 1)].objective_response.mean()
    d = df[(df.flt3_tkd == 0) & (df[tx] == 0)].objective_response.mean()
    iter22[tx] = {
        "interaction_beta": beta, "interaction_OR": or_, "interaction_p": p,
        "stratified": {"TKD+/tx+": a, "TKD+/tx-": b, "TKD-/tx+": c, "TKD-/tx-": d},
        "summary": (
            f"FLT3-TKD × {tx}: beta={beta:+.3f} (OR={or_:.2f}, p={p:.3g}). "
            f"ORR TKD+/{tx}+={a:.3f}, TKD+/{tx}-={b:.3f}, TKD-/{tx}+={c:.3f}, TKD-/{tx}-={d:.3f}"
        ),
    }
results["iter22_flt3tkd_x_flt3i"] = iter22

# ------------------------------------------------------------------
# ITERATION 23 — Age × 7+3 (younger benefits more from intensive)
# ------------------------------------------------------------------
beta, or_, p, _ = _logit_or(
    "objective_response ~ age_years * treatment_7plus3",
    df, focal="age_years:treatment_7plus3",
)
results["iter23_age_x_7plus3"] = {
    "interaction_beta": beta, "interaction_OR": or_, "interaction_p": p,
    "summary": f"age_years × 7+3: beta={beta:+.4f} (OR={or_:.3f}, p={p:.3g})",
}

# ------------------------------------------------------------------
# ITERATION 24 — TP53 × venetoclax+aza interaction
# (TP53-mutated AML is known to do poorly even with venaza)
# ------------------------------------------------------------------
beta, or_, p, _ = _logit_or(
    "objective_response ~ tp53_mutation * treatment_venetoclax_azacitidine",
    df, focal="tp53_mutation:treatment_venetoclax_azacitidine",
)
a = df[(df.tp53_mutation == 1) & (df.treatment_venetoclax_azacitidine == 1)].objective_response.mean()
b = df[(df.tp53_mutation == 1) & (df.treatment_venetoclax_azacitidine == 0)].objective_response.mean()
c = df[(df.tp53_mutation == 0) & (df.treatment_venetoclax_azacitidine == 1)].objective_response.mean()
d = df[(df.tp53_mutation == 0) & (df.treatment_venetoclax_azacitidine == 0)].objective_response.mean()
results["iter24_tp53_x_venaza"] = {
    "interaction_beta": beta, "interaction_OR": or_, "interaction_p": p,
    "stratified": {"TP53+/venaza+": a, "TP53+/venaza-": b, "TP53-/venaza+": c, "TP53-/venaza-": d},
    "summary": (
        f"TP53 × venaza: beta={beta:+.3f} (OR={or_:.2f}, p={p:.3g}). "
        f"ORR TP53+/venaza+={a:.3f}, TP53+/venaza-={b:.3f}, TP53-/venaza+={c:.3f}, TP53-/venaza-={d:.3f}"
    ),
}

# ------------------------------------------------------------------
# ITERATION 25 — Final synthesis: matched-therapy benefit metric
# Predicted ORR with vs without canonical biomarker-matched therapy
# in matched subgroups (FLT3+gilt, IDH1+ivo, IDH2+ena), plus combined
# multivariable test for matched-therapy term.
# ------------------------------------------------------------------
df["matched_therapy"] = (
    ((df.flt3_itd == 1) & ((df.treatment_gilteritinib == 1) | (df.treatment_midostaurin == 1))) |
    ((df.flt3_tkd == 1) & ((df.treatment_gilteritinib == 1) | (df.treatment_midostaurin == 1))) |
    ((df.idh1_mutation == 1) & (df.treatment_ivosidenib == 1)) |
    ((df.idh2_mutation == 1) & (df.treatment_enasidenib == 1))
).astype(int)
diff, p, summary = _orr_diff(df.matched_therapy == 1, df.matched_therapy == 0,
                             "matched", "unmatched")
# adjusted version
adj_form = (
    "objective_response ~ matched_therapy + age_years + ecog_ps + tp53_mutation + "
    "complex_karyotype + secondary_aml + albumin_g_dl + nlr"
)
adj = logit(adj_form, data=df).fit(disp=False, maxiter=500)
results["iter25_matched_therapy"] = {
    "unadjusted": {"effect_estimate": diff, "p_value": p, "summary": summary},
    "adjusted": {
        "beta": float(adj.params["matched_therapy"]),
        "OR": float(np.exp(adj.params["matched_therapy"])),
        "p_value": float(adj.pvalues["matched_therapy"]),
        "summary": (
            f"Adjusted matched_therapy: beta={float(adj.params['matched_therapy']):+.3f} "
            f"(OR={float(np.exp(adj.params['matched_therapy'])):.2f}, "
            f"p={float(adj.pvalues['matched_therapy']):.3g})"
        ),
    },
}

# Save raw
with open("_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Print compact summaries to stdout
def _walk(obj, path=""):
    if isinstance(obj, dict):
        if "summary" in obj and isinstance(obj["summary"], str):
            print(f"[{path}] {obj['summary']}")
        else:
            for k, v in obj.items():
                _walk(v, f"{path}/{k}" if path else k)
_walk(results)
