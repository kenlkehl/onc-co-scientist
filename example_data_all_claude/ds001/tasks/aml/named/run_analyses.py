"""
Iterative analysis of ds001_aml dataset.
Computes a battery of statistical tests across the iterations described in the task brief.
Saves results to results.json for use by the transcript builder.
"""

import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")

results = {}


def fit_logit(data, formula):
    """Fit a logit model and return the params/pvalues dict."""
    m = smf.logit(formula, data=data).fit(disp=False)
    out = {}
    for name in m.params.index:
        out[name] = {
            "coef": float(m.params[name]),
            "p": float(m.pvalues[name]),
            "or": float(np.exp(m.params[name])),
        }
    return out


def chi2_or_fisher(data, group_col, outcome_col="objective_response"):
    """2x2 chi-square; return rates and p-value."""
    sub = data[[group_col, outcome_col]].dropna()
    rate1 = sub.loc[sub[group_col] == 1, outcome_col].mean()
    rate0 = sub.loc[sub[group_col] == 0, outcome_col].mean()
    n1 = int((sub[group_col] == 1).sum())
    n0 = int((sub[group_col] == 0).sum())
    tab = pd.crosstab(sub[group_col], sub[outcome_col])
    if tab.shape == (2, 2):
        chi2, p, _, _ = stats.chi2_contingency(tab)
    else:
        chi2, p = np.nan, np.nan
    return {
        "rate_on": float(rate1),
        "rate_off": float(rate0),
        "diff": float(rate1 - rate0),
        "n_on": n1,
        "n_off": n0,
        "p": float(p),
        "chi2": float(chi2),
    }


# =============================================================================
# ITERATION 1: outcome distribution and main effects of treatments
# =============================================================================
results["it1"] = {}
results["it1"]["overall_orr"] = float(df["objective_response"].mean())
treatments = [
    "treatment_midostaurin",
    "treatment_gilteritinib",
    "treatment_ivosidenib",
    "treatment_enasidenib",
    "treatment_venetoclax_azacitidine",
    "treatment_7plus3",
]
for t in treatments:
    results["it1"][t] = chi2_or_fisher(df, t)

# =============================================================================
# ITERATION 2: demographics + clinical features main effects
# =============================================================================
results["it2"] = {}

# age - logistic regression
m = smf.logit("objective_response ~ age_years", data=df).fit(disp=False)
results["it2"]["age_years"] = {
    "coef": float(m.params["age_years"]),
    "p": float(m.pvalues["age_years"]),
    "or": float(np.exp(m.params["age_years"])),
}
# sex
results["it2"]["sex_female"] = chi2_or_fisher(df, "sex_female")
# ecog as ordinal logit
m = smf.logit("objective_response ~ ecog_ps", data=df).fit(disp=False)
results["it2"]["ecog_ps"] = {
    "coef": float(m.params["ecog_ps"]),
    "p": float(m.pvalues["ecog_ps"]),
    "or": float(np.exp(m.params["ecog_ps"])),
}
results["it2"]["secondary_aml"] = chi2_or_fisher(df, "secondary_aml")
results["it2"]["unfit_for_intensive"] = chi2_or_fisher(df, "unfit_for_intensive")

# =============================================================================
# ITERATION 3: cytogenetic / molecular markers
# =============================================================================
results["it3"] = {}
markers = [
    "complex_karyotype",
    "flt3_itd",
    "flt3_tkd",
    "idh1_mutation",
    "idh2_mutation",
    "npm1_mutation",
    "tp53_mutation",
]
for m_ in markers:
    results["it3"][m_] = chi2_or_fisher(df, m_)

# =============================================================================
# ITERATION 4: continuous lab biomarkers
# =============================================================================
results["it4"] = {}
labs = [
    "wbc_k_per_ul",
    "blast_pct_marrow",
    "albumin_g_dl",
    "ldh_u_l",
    "weight_loss_pct_6mo",
    "crp_mg_l",
    "nlr",
    "hemoglobin_g_dl",
    "alkaline_phosphatase_u_l",
    "ast_u_l",
    "alt_u_l",
    "total_bilirubin_mg_dl",
    "creatinine_mg_dl",
    "bun_mg_dl",
    "sodium_meq_l",
    "potassium_meq_l",
    "calcium_mg_dl",
]
for lab in labs:
    m = smf.logit(f"objective_response ~ {lab}", data=df).fit(disp=False)
    results["it4"][lab] = {
        "coef": float(m.params[lab]),
        "p": float(m.pvalues[lab]),
        "or": float(np.exp(m.params[lab])),
        "mean_resp": float(df.loc[df.objective_response == 1, lab].mean()),
        "mean_nonresp": float(df.loc[df.objective_response == 0, lab].mean()),
    }

# =============================================================================
# ITERATION 5: targeted treatment - biomarker matching
# =============================================================================
# Hypothesized treatment-biomarker pairings from clinical knowledge:
#   midostaurin / gilteritinib -> FLT3 mutated (ITD or TKD)
#   ivosidenib -> IDH1 mutated
#   enasidenib -> IDH2 mutated
#   venetoclax_azacitidine -> unfit_for_intensive
#   7plus3 -> fit (not unfit)
results["it5"] = {}


def stratified_effect(d, treatment, marker, marker_val=1):
    sub = d[d[marker] == marker_val]
    return chi2_or_fisher(sub, treatment)


results["it5"]["mido_in_flt3itd"] = stratified_effect(df, "treatment_midostaurin", "flt3_itd", 1)
results["it5"]["mido_in_flt3itd_neg"] = stratified_effect(df, "treatment_midostaurin", "flt3_itd", 0)
results["it5"]["gilt_in_flt3itd"] = stratified_effect(df, "treatment_gilteritinib", "flt3_itd", 1)
results["it5"]["gilt_in_flt3any"] = chi2_or_fisher(
    df.assign(flt3any=((df.flt3_itd == 1) | (df.flt3_tkd == 1)).astype(int)).query("flt3any==1"),
    "treatment_gilteritinib",
)
results["it5"]["ivo_in_idh1"] = stratified_effect(df, "treatment_ivosidenib", "idh1_mutation", 1)
results["it5"]["ivo_in_idh1_neg"] = stratified_effect(df, "treatment_ivosidenib", "idh1_mutation", 0)
results["it5"]["ena_in_idh2"] = stratified_effect(df, "treatment_enasidenib", "idh2_mutation", 1)
results["it5"]["ena_in_idh2_neg"] = stratified_effect(df, "treatment_enasidenib", "idh2_mutation", 0)
results["it5"]["venaza_in_unfit"] = stratified_effect(df, "treatment_venetoclax_azacitidine", "unfit_for_intensive", 1)
results["it5"]["venaza_in_fit"] = stratified_effect(df, "treatment_venetoclax_azacitidine", "unfit_for_intensive", 0)
results["it5"]["sevenplusthree_in_fit"] = stratified_effect(df, "treatment_7plus3", "unfit_for_intensive", 0)
results["it5"]["sevenplusthree_in_unfit"] = stratified_effect(df, "treatment_7plus3", "unfit_for_intensive", 1)

# =============================================================================
# ITERATION 6: formal interaction tests (logit with interaction term)
# =============================================================================
results["it6"] = {}

def interaction_test(treatment, marker, dat=df):
    f = f"objective_response ~ {treatment} * {marker}"
    m = smf.logit(f, data=dat).fit(disp=False)
    inter = f"{treatment}:{marker}"
    return {
        "main_tx": float(m.params[treatment]),
        "main_mk": float(m.params[marker]),
        "inter": float(m.params[inter]),
        "p_inter": float(m.pvalues[inter]),
        "or_inter": float(np.exp(m.params[inter])),
    }


results["it6"]["mido_x_flt3itd"] = interaction_test("treatment_midostaurin", "flt3_itd")
results["it6"]["gilt_x_flt3itd"] = interaction_test("treatment_gilteritinib", "flt3_itd")
results["it6"]["gilt_x_flt3tkd"] = interaction_test("treatment_gilteritinib", "flt3_tkd")
results["it6"]["ivo_x_idh1"] = interaction_test("treatment_ivosidenib", "idh1_mutation")
results["it6"]["ena_x_idh2"] = interaction_test("treatment_enasidenib", "idh2_mutation")
results["it6"]["venaza_x_unfit"] = interaction_test("treatment_venetoclax_azacitidine", "unfit_for_intensive")
results["it6"]["sevenplusthree_x_unfit"] = interaction_test("treatment_7plus3", "unfit_for_intensive")

# =============================================================================
# ITERATION 7: explore venaza in unfit by other co-modifiers
# =============================================================================
results["it7"] = {}
sub = df[df.unfit_for_intensive == 1]
for mod in ["tp53_mutation", "complex_karyotype", "secondary_aml", "flt3_itd", "npm1_mutation"]:
    for v in [0, 1]:
        s = sub[sub[mod] == v]
        if s.shape[0] >= 50 and s["treatment_venetoclax_azacitidine"].nunique() == 2:
            r = chi2_or_fisher(s, "treatment_venetoclax_azacitidine")
            results["it7"][f"venaza_unfit_{mod}={v}"] = r

# also formal 3-way interaction logits
for mod in ["tp53_mutation", "complex_karyotype"]:
    f = f"objective_response ~ treatment_venetoclax_azacitidine * unfit_for_intensive * {mod}"
    m = smf.logit(f, data=df).fit(disp=False)
    p_lookup = {}
    for k in m.params.index:
        p_lookup[k] = {"coef": float(m.params[k]), "p": float(m.pvalues[k])}
    results["it7"][f"3way_logit_{mod}"] = p_lookup

# =============================================================================
# ITERATION 8: refined ven/aza subgroup — unfit & TP53 wt & not complex karyotype
# =============================================================================
results["it8"] = {}
sub_strict = df[(df.unfit_for_intensive == 1) & (df.tp53_mutation == 0) & (df.complex_karyotype == 0)]
results["it8"]["venaza_strict_subgroup"] = chi2_or_fisher(sub_strict, "treatment_venetoclax_azacitidine")
results["it8"]["n_strict"] = int(sub_strict.shape[0])

# Compare to complement (everyone else)
not_strict = df[~((df.unfit_for_intensive == 1) & (df.tp53_mutation == 0) & (df.complex_karyotype == 0))]
results["it8"]["venaza_outside_strict"] = chi2_or_fisher(not_strict, "treatment_venetoclax_azacitidine")

# Formal 3-way interaction p-value via likelihood ratio test
m_full = smf.logit(
    "objective_response ~ treatment_venetoclax_azacitidine * unfit_for_intensive * tp53_mutation * complex_karyotype",
    data=df,
).fit(disp=False)
m_red = smf.logit(
    "objective_response ~ treatment_venetoclax_azacitidine + unfit_for_intensive + tp53_mutation + complex_karyotype",
    data=df,
).fit(disp=False)
lr = 2 * (m_full.llf - m_red.llf)
df_diff = m_full.df_model - m_red.df_model
p_lr = 1 - stats.chi2.cdf(lr, df_diff)
results["it8"]["lr_test"] = {"lr": float(lr), "df": int(df_diff), "p": float(p_lr)}

# =============================================================================
# ITERATION 9: heterogeneity screen for treatment_venetoclax_azacitidine
# =============================================================================
results["it9"] = {}
candidate_modifiers = [
    "sex_female",
    "secondary_aml",
    "unfit_for_intensive",
    "complex_karyotype",
    "flt3_itd",
    "flt3_tkd",
    "idh1_mutation",
    "idh2_mutation",
    "npm1_mutation",
    "tp53_mutation",
]
for mod in candidate_modifiers:
    f = f"objective_response ~ treatment_venetoclax_azacitidine * {mod}"
    m = smf.logit(f, data=df).fit(disp=False)
    results["it9"][mod] = {
        "p_inter": float(m.pvalues[f"treatment_venetoclax_azacitidine:{mod}"]),
        "coef_inter": float(m.params[f"treatment_venetoclax_azacitidine:{mod}"]),
    }

# Continuous modifiers (split at median)
for cmod in ["age_years", "ecog_ps", "albumin_g_dl", "ldh_u_l", "blast_pct_marrow", "wbc_k_per_ul"]:
    if cmod == "ecog_ps":
        df["_med"] = (df[cmod] >= 1).astype(int)
    else:
        df["_med"] = (df[cmod] > df[cmod].median()).astype(int)
    f = "objective_response ~ treatment_venetoclax_azacitidine * _med"
    m = smf.logit(f, data=df).fit(disp=False)
    results["it9"][f"{cmod}_above_median"] = {
        "p_inter": float(m.pvalues["treatment_venetoclax_azacitidine:_med"]),
        "coef_inter": float(m.params["treatment_venetoclax_azacitidine:_med"]),
    }
df.drop(columns=["_med"], inplace=True, errors="ignore")

# =============================================================================
# ITERATION 10: heterogeneity screen for the targeted agents
# =============================================================================
results["it10"] = {}
for tx in [
    "treatment_midostaurin",
    "treatment_gilteritinib",
    "treatment_ivosidenib",
    "treatment_enasidenib",
    "treatment_7plus3",
]:
    results["it10"][tx] = {}
    for mod in candidate_modifiers + ["age_years_above_median"]:
        if mod == "age_years_above_median":
            d = df.assign(_m=(df.age_years > df.age_years.median()).astype(int))
            f = f"objective_response ~ {tx} * _m"
        else:
            d = df
            f = f"objective_response ~ {tx} * {mod}"
        try:
            m = smf.logit(f, data=d).fit(disp=False)
            inter_name = f"{tx}:_m" if mod == "age_years_above_median" else f"{tx}:{mod}"
            results["it10"][tx][mod] = {
                "p_inter": float(m.pvalues[inter_name]),
                "coef_inter": float(m.params[inter_name]),
            }
        except Exception as e:
            results["it10"][tx][mod] = {"error": str(e)}

# =============================================================================
# ITERATION 11: paradoxical findings — ivosidenib in IDH1+
# =============================================================================
results["it11"] = {}
# stratify ivo in IDH1+ further by tp53, complex karyotype, age, ecog
sub = df[df.idh1_mutation == 1]
results["it11"]["ivo_idh1_overall"] = chi2_or_fisher(sub, "treatment_ivosidenib")
for mod in ["tp53_mutation", "complex_karyotype", "secondary_aml", "unfit_for_intensive"]:
    for v in [0, 1]:
        s = sub[sub[mod] == v]
        if s.shape[0] >= 30 and s["treatment_ivosidenib"].nunique() == 2:
            results["it11"][f"ivo_idh1_{mod}={v}"] = chi2_or_fisher(s, "treatment_ivosidenib")

# also gilteritinib heterogeneity in FLT3-ITD by tp53, etc
sub = df[df.flt3_itd == 1]
results["it11"]["gilt_itd_overall"] = chi2_or_fisher(sub, "treatment_gilteritinib")
for mod in ["tp53_mutation", "complex_karyotype", "secondary_aml", "unfit_for_intensive"]:
    for v in [0, 1]:
        s = sub[sub[mod] == v]
        if s.shape[0] >= 30 and s["treatment_gilteritinib"].nunique() == 2:
            results["it11"][f"gilt_itd_{mod}={v}"] = chi2_or_fisher(s, "treatment_gilteritinib")

# =============================================================================
# ITERATION 12: 7+3 stratified by patient fitness / subgroups
# =============================================================================
results["it12"] = {}
results["it12"]["sevenplusthree_fit_unfit0"] = stratified_effect(df, "treatment_7plus3", "unfit_for_intensive", 0)
results["it12"]["sevenplusthree_fit_unfit1"] = stratified_effect(df, "treatment_7plus3", "unfit_for_intensive", 1)
# Stratify in fit further by tp53 / complex karyotype
sub = df[df.unfit_for_intensive == 0]
for mod in ["tp53_mutation", "complex_karyotype", "secondary_aml"]:
    for v in [0, 1]:
        s = sub[sub[mod] == v]
        if s.shape[0] >= 50 and s["treatment_7plus3"].nunique() == 2:
            results["it12"][f"sevenplusthree_fit_{mod}={v}"] = chi2_or_fisher(s, "treatment_7plus3")

# =============================================================================
# ITERATION 13: multivariable model — all features together
# =============================================================================
results["it13"] = {}
features = [
    "age_years",
    "sex_female",
    "ecog_ps",
    "secondary_aml",
    "unfit_for_intensive",
    "complex_karyotype",
    "flt3_itd",
    "flt3_tkd",
    "idh1_mutation",
    "idh2_mutation",
    "npm1_mutation",
    "tp53_mutation",
    "wbc_k_per_ul",
    "blast_pct_marrow",
    "albumin_g_dl",
    "ldh_u_l",
    "weight_loss_pct_6mo",
    "crp_mg_l",
    "nlr",
    "treatment_midostaurin",
    "treatment_gilteritinib",
    "treatment_ivosidenib",
    "treatment_enasidenib",
    "treatment_venetoclax_azacitidine",
    "treatment_7plus3",
    "hemoglobin_g_dl",
    "alkaline_phosphatase_u_l",
    "ast_u_l",
    "alt_u_l",
    "total_bilirubin_mg_dl",
    "creatinine_mg_dl",
    "bun_mg_dl",
    "sodium_meq_l",
    "potassium_meq_l",
    "calcium_mg_dl",
]
formula = "objective_response ~ " + " + ".join(features)
m = smf.logit(formula, data=df).fit(disp=False, maxiter=200)
results["it13"]["coefs"] = {
    n: {"coef": float(m.params[n]), "p": float(m.pvalues[n]), "or": float(np.exp(m.params[n]))}
    for n in m.params.index
}
results["it13"]["llf"] = float(m.llf)
results["it13"]["pseudo_r2"] = float(1 - m.llf / m.llnull)

# =============================================================================
# ITERATION 14: Final refined model with key interactions
# =============================================================================
results["it14"] = {}
m = smf.logit(
    "objective_response ~ treatment_venetoclax_azacitidine * unfit_for_intensive "
    "+ treatment_venetoclax_azacitidine:tp53_mutation "
    "+ treatment_venetoclax_azacitidine:complex_karyotype "
    "+ treatment_ivosidenib*idh1_mutation "
    "+ treatment_enasidenib*idh2_mutation "
    "+ treatment_midostaurin*flt3_itd "
    "+ treatment_gilteritinib*flt3_itd "
    "+ age_years + sex_female + ecog_ps + secondary_aml + complex_karyotype "
    "+ tp53_mutation + npm1_mutation + flt3_itd + flt3_tkd + idh1_mutation + idh2_mutation "
    "+ albumin_g_dl + ldh_u_l + wbc_k_per_ul + blast_pct_marrow + nlr + crp_mg_l + weight_loss_pct_6mo",
    data=df,
).fit(disp=False, maxiter=200)
results["it14"]["coefs"] = {
    n: {"coef": float(m.params[n]), "p": float(m.pvalues[n]), "or": float(np.exp(m.params[n]))}
    for n in m.params.index
}

# =============================================================================
# ITERATION 15: Confirm strict ven/aza subgroup hypothesis
# =============================================================================
# Compute the venaza effect in a series of nested subgroups
results["it15"] = {}
combos = [
    ("unfit_only", df.unfit_for_intensive == 1),
    ("unfit_tp53wt", (df.unfit_for_intensive == 1) & (df.tp53_mutation == 0)),
    ("unfit_ck0", (df.unfit_for_intensive == 1) & (df.complex_karyotype == 0)),
    ("unfit_tp53wt_ck0",
        (df.unfit_for_intensive == 1) & (df.tp53_mutation == 0) & (df.complex_karyotype == 0)),
    ("fit_only", df.unfit_for_intensive == 0),
    ("fit_tp53wt", (df.unfit_for_intensive == 0) & (df.tp53_mutation == 0)),
    ("fit_tp53mut", (df.unfit_for_intensive == 0) & (df.tp53_mutation == 1)),
    ("unfit_tp53mut", (df.unfit_for_intensive == 1) & (df.tp53_mutation == 1)),
    ("unfit_ck1", (df.unfit_for_intensive == 1) & (df.complex_karyotype == 1)),
]
for name, mask in combos:
    sub = df[mask]
    if sub.shape[0] > 50 and sub["treatment_venetoclax_azacitidine"].nunique() == 2:
        r = chi2_or_fisher(sub, "treatment_venetoclax_azacitidine")
        r["n"] = int(sub.shape[0])
        results["it15"][name] = r

# Save all
with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("Done. Saved results.json with sections:", list(results.keys()))
