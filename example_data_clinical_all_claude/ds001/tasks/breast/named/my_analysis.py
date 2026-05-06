"""Comprehensive 25-iteration analysis of ds001_breast PFS outcomes.

For each iteration we record results that map directly to transcript.json.
"""
from __future__ import annotations

import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")
RESULTS: dict = {}

OUTCOME = "pfs_months"
BIN_COLS = [
    "sex_female", "stage_iv", "has_brain_mets", "node_positive", "postmenopausal",
    "er_positive", "pr_positive", "her2_positive", "her2_low",
    "brca1_mutation", "brca2_mutation", "pik3ca_mutation",
]
CONT_COLS = [
    "age_years", "ki67_pct", "tumor_size_cm", "albumin_g_dl", "ldh_u_l",
    "weight_loss_pct_6mo", "crp_mg_l", "nlr",
    "hemoglobin_g_dl", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
    "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
    "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl",
]
TREATMENTS = [
    "treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
    "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
]

# Confounders we routinely adjust for in multivariable models
ADJUSTERS = [
    "age_years", "sex_female", "ecog_ps", "stage_iv", "has_brain_mets",
    "tumor_size_cm", "albumin_g_dl", "ldh_u_l", "ki67_pct",
]


def add_const(X):
    return sm.add_constant(X, has_constant="add")


def ols_main(col, adj=False):
    """OLS of outcome ~ col, optionally adjusted for confounders."""
    X_cols = [col] + (ADJUSTERS if adj else [])
    X_cols = [c for c in X_cols if c != col] if not adj else X_cols
    Xc = [col] if not adj else list(dict.fromkeys([col] + ADJUSTERS))
    X = DF[Xc].astype(float)
    X = add_const(X)
    y = DF[OUTCOME].astype(float)
    model = sm.OLS(y, X).fit()
    return {
        "n": int(len(y)),
        "coef": float(model.params[col]),
        "se": float(model.bse[col]),
        "t": float(model.tvalues[col]),
        "p": float(model.pvalues[col]),
        "r2": float(model.rsquared),
    }


def ttest_groups(col):
    a = DF.loc[DF[col] == 1, OUTCOME]
    b = DF.loc[DF[col] == 0, OUTCOME]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "mean_pos": float(a.mean()), "mean_neg": float(b.mean()),
        "diff": float(a.mean() - b.mean()), "t": float(t), "p": float(p),
        "n_pos": int(len(a)), "n_neg": int(len(b)),
    }


def interaction_test(treatment, modifier, modifier_kind="binary"):
    """OLS y ~ treatment * modifier + adjusters. Returns interaction coef + p."""
    cols = list(dict.fromkeys([treatment, modifier] + ADJUSTERS))
    X = DF[cols].astype(float).copy()
    X[f"{treatment}_x_{modifier}"] = X[treatment] * X[modifier]
    X = add_const(X)
    y = DF[OUTCOME].astype(float)
    model = sm.OLS(y, X).fit()
    inter_name = f"{treatment}_x_{modifier}"
    return {
        "n": int(len(y)),
        "treatment_coef": float(model.params[treatment]),
        "treatment_p": float(model.pvalues[treatment]),
        "modifier_coef": float(model.params[modifier]),
        "modifier_p": float(model.pvalues[modifier]),
        "interaction_coef": float(model.params[inter_name]),
        "interaction_p": float(model.pvalues[inter_name]),
    }


def subgroup_effect(treatment, mask, label):
    """Estimate treatment effect within a subgroup mask, vs. outside."""
    sub = DF.loc[mask]
    if sub[treatment].nunique() < 2 or len(sub) < 50:
        return {"label": label, "n": int(len(sub)), "n_treated": int(sub[treatment].sum()),
                "effect": None, "p": None}
    a = sub.loc[sub[treatment] == 1, OUTCOME]
    b = sub.loc[sub[treatment] == 0, OUTCOME]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "label": label,
        "n": int(len(sub)), "n_treated": int(len(a)), "n_untreated": int(len(b)),
        "mean_treated": float(a.mean()), "mean_untreated": float(b.mean()),
        "effect": float(a.mean() - b.mean()), "p": float(p),
    }


def adjusted_subgroup_effect(treatment, mask, label):
    sub = DF.loc[mask]
    if sub[treatment].nunique() < 2 or len(sub) < 50:
        return {"label": label, "n": int(len(sub)), "effect": None, "p": None}
    cols = list(dict.fromkeys([treatment] + ADJUSTERS))
    X = sub[cols].astype(float)
    X = add_const(X)
    y = sub[OUTCOME].astype(float)
    model = sm.OLS(y, X).fit()
    return {
        "label": label, "n": int(len(sub)), "n_treated": int(sub[treatment].sum()),
        "adj_effect": float(model.params[treatment]),
        "adj_p": float(model.pvalues[treatment]),
    }


# ---------- ITERATION 1: Outcome distribution and basic feature checks ----------
print("=== Iter 1 ===")
res = {}
res["pfs_summary"] = {
    "mean": float(DF[OUTCOME].mean()),
    "median": float(DF[OUTCOME].median()),
    "sd": float(DF[OUTCOME].std()),
    "min": float(DF[OUTCOME].min()),
    "max": float(DF[OUTCOME].max()),
    "p10": float(DF[OUTCOME].quantile(0.10)),
    "p90": float(DF[OUTCOME].quantile(0.90)),
}
res["ecog_distribution"] = DF["ecog_ps"].value_counts().sort_index().to_dict()
# PFS by ECOG
res["pfs_by_ecog"] = {
    int(k): {"n": int(g.shape[0]), "mean_pfs": float(g[OUTCOME].mean()),
             "sd": float(g[OUTCOME].std())}
    for k, g in DF.groupby("ecog_ps")
}
# ANOVA across ECOG
groups = [DF.loc[DF["ecog_ps"] == k, OUTCOME] for k in [0, 1, 2]]
F, p = stats.f_oneway(*groups)
res["anova_pfs_by_ecog"] = {"F": float(F), "p": float(p)}
RESULTS["iter01"] = res

# ---------- ITERATION 2: Demographic main effects ----------
print("=== Iter 2 ===")
res = {}
res["age_pfs"] = ols_main("age_years")
res["sex_pfs"] = ttest_groups("sex_female")
res["ecog_pfs_ols"] = ols_main("ecog_ps")
RESULTS["iter02"] = res

# ---------- ITERATION 3: Disease burden main effects ----------
print("=== Iter 3 ===")
res = {}
res["stage_iv_pfs"] = ttest_groups("stage_iv")
res["brain_mets_pfs"] = ttest_groups("has_brain_mets")
res["node_positive_pfs"] = ttest_groups("node_positive")
res["tumor_size_pfs"] = ols_main("tumor_size_cm")
RESULTS["iter03"] = res

# ---------- ITERATION 4: Hormonal/HER2 receptor main effects ----------
print("=== Iter 4 ===")
res = {}
for c in ["er_positive", "pr_positive", "her2_positive", "her2_low", "postmenopausal"]:
    res[c + "_pfs"] = ttest_groups(c)
res["ki67_pfs"] = ols_main("ki67_pct")
RESULTS["iter04"] = res

# ---------- ITERATION 5: Mutation main effects ----------
print("=== Iter 5 ===")
res = {}
for c in ["brca1_mutation", "brca2_mutation", "pik3ca_mutation"]:
    res[c + "_pfs"] = ttest_groups(c)
RESULTS["iter05"] = res

# ---------- ITERATION 6: Lab value main effects (organ function) ----------
print("=== Iter 6 ===")
res = {}
for c in ["albumin_g_dl", "ldh_u_l", "alkaline_phosphatase_u_l",
          "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl",
          "creatinine_mg_dl", "bun_mg_dl"]:
    res[c + "_pfs"] = ols_main(c)
RESULTS["iter06"] = res

# ---------- ITERATION 7: Lab value main effects (electrolytes / heme / inflammation) ----------
print("=== Iter 7 ===")
res = {}
for c in ["sodium_meq_l", "potassium_meq_l", "calcium_mg_dl",
          "hemoglobin_g_dl", "crp_mg_l", "nlr", "weight_loss_pct_6mo"]:
    res[c + "_pfs"] = ols_main(c)
RESULTS["iter07"] = res

# ---------- ITERATION 8: Treatment main effects (unadjusted) ----------
print("=== Iter 8 ===")
res = {}
for tr in TREATMENTS:
    res[tr + "_pfs"] = ttest_groups(tr)
RESULTS["iter08"] = res

# ---------- ITERATION 9: Treatment effects adjusted for prognostic factors ----------
print("=== Iter 9 ===")
res = {}
for tr in TREATMENTS:
    res[tr + "_adj"] = ols_main(tr, adj=True)
RESULTS["iter09"] = res

# ---------- ITERATION 10: Multivariable model with all covariates ----------
print("=== Iter 10 ===")
all_feats = (BIN_COLS + ["ecog_ps"] + CONT_COLS + TREATMENTS)
X = DF[all_feats].astype(float)
X = add_const(X)
y = DF[OUTCOME].astype(float)
model = sm.OLS(y, X).fit()
res = {
    "r2": float(model.rsquared),
    "adj_r2": float(model.rsquared_adj),
    "n": int(len(y)),
    "n_features": int(len(all_feats)),
    "params": {k: {"coef": float(v), "p": float(model.pvalues[k]),
                   "se": float(model.bse[k])}
               for k, v in model.params.items()},
}
RESULTS["iter10"] = res

# ---------- ITERATION 11: Biology-driven treatment x biomarker interactions ----------
print("=== Iter 11 ===")
res = {}
res["tamox_x_er"] = interaction_test("treatment_tamoxifen", "er_positive")
res["palbo_x_er"] = interaction_test("treatment_palbociclib", "er_positive")
res["trast_x_her2pos"] = interaction_test("treatment_trastuzumab", "her2_positive")
res["olap_x_brca1"] = interaction_test("treatment_olaparib", "brca1_mutation")
res["olap_x_brca2"] = interaction_test("treatment_olaparib", "brca2_mutation")
res["sacit_x_her2low"] = interaction_test("treatment_sacituzumab_govitecan", "her2_low")
res["pembro_x_her2pos"] = interaction_test("treatment_pembrolizumab", "her2_positive")
RESULTS["iter11"] = res

# ---------- ITERATION 12: Subgroup-based treatment effects within target biomarkers ----------
print("=== Iter 12 ===")
res = {}
res["tamox_in_ER+"] = subgroup_effect("treatment_tamoxifen", DF["er_positive"] == 1, "ER+")
res["tamox_in_ER-"] = subgroup_effect("treatment_tamoxifen", DF["er_positive"] == 0, "ER-")
res["palbo_in_ER+"] = subgroup_effect("treatment_palbociclib", DF["er_positive"] == 1, "ER+")
res["palbo_in_ER-"] = subgroup_effect("treatment_palbociclib", DF["er_positive"] == 0, "ER-")
res["trast_in_HER2+"] = subgroup_effect("treatment_trastuzumab", DF["her2_positive"] == 1, "HER2+")
res["trast_in_HER2-"] = subgroup_effect("treatment_trastuzumab", DF["her2_positive"] == 0, "HER2-")
res["olap_in_BRCA"] = subgroup_effect("treatment_olaparib",
                                       (DF["brca1_mutation"] == 1) | (DF["brca2_mutation"] == 1),
                                       "BRCA1 or BRCA2 mutated")
res["olap_in_noBRCA"] = subgroup_effect("treatment_olaparib",
                                         (DF["brca1_mutation"] == 0) & (DF["brca2_mutation"] == 0),
                                         "no BRCA mutation")
res["sacit_in_HER2low"] = subgroup_effect("treatment_sacituzumab_govitecan",
                                           DF["her2_low"] == 1, "HER2-low")
res["sacit_in_notHER2low"] = subgroup_effect("treatment_sacituzumab_govitecan",
                                              DF["her2_low"] == 0, "not HER2-low")
RESULTS["iter12"] = res

# ---------- ITERATION 13: Treatment x ECOG, stage, postmenopausal interactions ----------
print("=== Iter 13 ===")
res = {}
for tr in TREATMENTS:
    for mod in ["ecog_ps", "stage_iv", "postmenopausal", "has_brain_mets"]:
        key = f"{tr}_x_{mod}"
        try:
            res[key] = interaction_test(tr, mod)
        except Exception as e:
            res[key] = {"error": str(e)}
RESULTS["iter13"] = res

# ---------- ITERATION 14: Systematic treatment x feature interaction screen ----------
print("=== Iter 14 ===")
candidate_modifiers = [
    "age_years", "sex_female", "ecog_ps", "stage_iv", "has_brain_mets", "node_positive",
    "postmenopausal", "er_positive", "pr_positive", "her2_positive", "her2_low",
    "brca1_mutation", "brca2_mutation", "pik3ca_mutation", "ki67_pct", "tumor_size_cm",
    "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr",
    "hemoglobin_g_dl", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
    "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl", "sodium_meq_l",
    "potassium_meq_l", "calcium_mg_dl",
]
screen = {}
for tr in TREATMENTS:
    rows = []
    for mod in candidate_modifiers:
        if mod == tr:
            continue
        try:
            r = interaction_test(tr, mod)
            rows.append({"modifier": mod, **r})
        except Exception as e:
            rows.append({"modifier": mod, "error": str(e)})
    rows = sorted([r for r in rows if "interaction_p" in r], key=lambda r: r["interaction_p"])
    screen[tr] = rows[:8]  # keep top 8 modifiers per treatment
RESULTS["iter14"] = {"screen_top8_per_treatment": screen}

# ---------- ITERATION 15: Refined biomarker subgroup hypotheses for each treatment ----------
print("=== Iter 15 ===")
res = {}
# Examine the strongest interaction per treatment from the screen and quantify subgroup effect
for tr, rows in RESULTS["iter14"]["screen_top8_per_treatment"].items():
    if not rows:
        continue
    top = rows[0]
    mod = top["modifier"]
    if DF[mod].nunique() == 2:
        # binary modifier: treatment effect within mod==1 vs mod==0
        eff_pos = subgroup_effect(tr, DF[mod] == 1, f"{mod}=1")
        eff_neg = subgroup_effect(tr, DF[mod] == 0, f"{mod}=0")
        adj_pos = adjusted_subgroup_effect(tr, DF[mod] == 1, f"{mod}=1")
        adj_neg = adjusted_subgroup_effect(tr, DF[mod] == 0, f"{mod}=0")
        res[tr] = {"top_modifier": mod, "interaction_p": top["interaction_p"],
                   "eff_pos": eff_pos, "eff_neg": eff_neg,
                   "adj_pos": adj_pos, "adj_neg": adj_neg}
    else:
        # continuous modifier: split at median
        med = float(DF[mod].median())
        eff_hi = subgroup_effect(tr, DF[mod] >= med, f"{mod}>=median")
        eff_lo = subgroup_effect(tr, DF[mod] < med, f"{mod}<median")
        res[tr] = {"top_modifier": mod, "interaction_p": top["interaction_p"],
                   "median": med, "eff_hi": eff_hi, "eff_lo": eff_lo}
RESULTS["iter15"] = res

# ---------- ITERATION 16: Joint biomarker-defined subgroups (multi-factor) ----------
print("=== Iter 16 ===")
res = {}
# olaparib in BRCA-mutated AND HRD-related contexts
res["olap_brca1_only"] = subgroup_effect("treatment_olaparib", DF["brca1_mutation"] == 1, "BRCA1+")
res["olap_brca2_only"] = subgroup_effect("treatment_olaparib", DF["brca2_mutation"] == 1, "BRCA2+")
res["olap_brca_anyAge<50"] = subgroup_effect(
    "treatment_olaparib",
    ((DF["brca1_mutation"] == 1) | (DF["brca2_mutation"] == 1)) & (DF["age_years"] < 50),
    "BRCA+ & age<50")
res["olap_brca_postmeno"] = subgroup_effect(
    "treatment_olaparib",
    ((DF["brca1_mutation"] == 1) | (DF["brca2_mutation"] == 1)) & (DF["postmenopausal"] == 1),
    "BRCA+ & postmenopausal")
# tamoxifen ER+ subgroups
res["tamox_ER+_postmeno"] = subgroup_effect(
    "treatment_tamoxifen",
    (DF["er_positive"] == 1) & (DF["postmenopausal"] == 1),
    "ER+ postmenopausal")
res["tamox_ER+_premeno"] = subgroup_effect(
    "treatment_tamoxifen",
    (DF["er_positive"] == 1) & (DF["postmenopausal"] == 0),
    "ER+ premenopausal")
# palbo ER+/HER2-
res["palbo_ER+HER2-"] = subgroup_effect(
    "treatment_palbociclib",
    (DF["er_positive"] == 1) & (DF["her2_positive"] == 0),
    "ER+ HER2-")
# trastuzumab HER2+
res["trast_HER2+_node+"] = subgroup_effect(
    "treatment_trastuzumab",
    (DF["her2_positive"] == 1) & (DF["node_positive"] == 1),
    "HER2+ node+")
res["trast_HER2+_stageIV"] = subgroup_effect(
    "treatment_trastuzumab",
    (DF["her2_positive"] == 1) & (DF["stage_iv"] == 1),
    "HER2+ stage IV")
# sacituzumab in HER2-low TNBC-like
res["sacit_HER2low_ER-"] = subgroup_effect(
    "treatment_sacituzumab_govitecan",
    (DF["her2_low"] == 1) & (DF["er_positive"] == 0),
    "HER2-low & ER-")
res["sacit_HER2low_ER+"] = subgroup_effect(
    "treatment_sacituzumab_govitecan",
    (DF["her2_low"] == 1) & (DF["er_positive"] == 1),
    "HER2-low & ER+")
# pembro: try TNBC-like and PD-L1 surrogates (none available, use nlr/crp high)
res["pembro_TNBC"] = subgroup_effect(
    "treatment_pembrolizumab",
    (DF["er_positive"] == 0) & (DF["pr_positive"] == 0) & (DF["her2_positive"] == 0),
    "TNBC (ER-/PR-/HER2-)")
res["pembro_TNBC_highCRP"] = subgroup_effect(
    "treatment_pembrolizumab",
    (DF["er_positive"] == 0) & (DF["pr_positive"] == 0) & (DF["her2_positive"] == 0)
    & (DF["crp_mg_l"] > DF["crp_mg_l"].median()),
    "TNBC + high CRP")
RESULTS["iter16"] = res

# ---------- ITERATION 17: Adjusted treatment effects in marker-defined subgroups ----------
print("=== Iter 17 ===")
res = {}
res["tamox_ER+_adj"] = adjusted_subgroup_effect("treatment_tamoxifen", DF["er_positive"] == 1, "ER+")
res["tamox_ER-_adj"] = adjusted_subgroup_effect("treatment_tamoxifen", DF["er_positive"] == 0, "ER-")
res["palbo_ER+_adj"] = adjusted_subgroup_effect("treatment_palbociclib", DF["er_positive"] == 1, "ER+")
res["palbo_ER-_adj"] = adjusted_subgroup_effect("treatment_palbociclib", DF["er_positive"] == 0, "ER-")
res["trast_HER2+_adj"] = adjusted_subgroup_effect("treatment_trastuzumab", DF["her2_positive"] == 1, "HER2+")
res["trast_HER2-_adj"] = adjusted_subgroup_effect("treatment_trastuzumab", DF["her2_positive"] == 0, "HER2-")
res["olap_BRCA_adj"] = adjusted_subgroup_effect(
    "treatment_olaparib",
    (DF["brca1_mutation"] == 1) | (DF["brca2_mutation"] == 1), "BRCA+")
res["olap_noBRCA_adj"] = adjusted_subgroup_effect(
    "treatment_olaparib",
    (DF["brca1_mutation"] == 0) & (DF["brca2_mutation"] == 0), "no BRCA")
res["sacit_HER2low_adj"] = adjusted_subgroup_effect(
    "treatment_sacituzumab_govitecan", DF["her2_low"] == 1, "HER2-low")
res["sacit_notHER2low_adj"] = adjusted_subgroup_effect(
    "treatment_sacituzumab_govitecan", DF["her2_low"] == 0, "not HER2-low")
res["pembro_TNBC_adj"] = adjusted_subgroup_effect(
    "treatment_pembrolizumab",
    (DF["er_positive"] == 0) & (DF["pr_positive"] == 0) & (DF["her2_positive"] == 0),
    "TNBC")
res["pembro_notTNBC_adj"] = adjusted_subgroup_effect(
    "treatment_pembrolizumab",
    ~((DF["er_positive"] == 0) & (DF["pr_positive"] == 0) & (DF["her2_positive"] == 0)),
    "not TNBC")
RESULTS["iter17"] = res

# ---------- ITERATION 18: Three-way interactions / further refinement of best subgroups ----------
print("=== Iter 18 ===")
res = {}
# olaparib in BRCA1 vs BRCA2 separately (adjusted)
res["olap_BRCA1_adj"] = adjusted_subgroup_effect("treatment_olaparib",
                                                  DF["brca1_mutation"] == 1, "BRCA1+")
res["olap_BRCA2_adj"] = adjusted_subgroup_effect("treatment_olaparib",
                                                  DF["brca2_mutation"] == 1, "BRCA2+")
# tamoxifen in ER+/PR+ vs ER+/PR-
res["tamox_ER+PR+"] = adjusted_subgroup_effect(
    "treatment_tamoxifen", (DF["er_positive"] == 1) & (DF["pr_positive"] == 1), "ER+/PR+")
res["tamox_ER+PR-"] = adjusted_subgroup_effect(
    "treatment_tamoxifen", (DF["er_positive"] == 1) & (DF["pr_positive"] == 0), "ER+/PR-")
# palbo ER+/HER2- adjusted
res["palbo_ER+HER2-_adj"] = adjusted_subgroup_effect(
    "treatment_palbociclib",
    (DF["er_positive"] == 1) & (DF["her2_positive"] == 0), "ER+/HER2-")
# trastuzumab HER2+ further by ECOG
res["trast_HER2+_ecog<=1"] = adjusted_subgroup_effect(
    "treatment_trastuzumab",
    (DF["her2_positive"] == 1) & (DF["ecog_ps"] <= 1), "HER2+ ECOG<=1")
res["trast_HER2+_ecog2"] = adjusted_subgroup_effect(
    "treatment_trastuzumab",
    (DF["her2_positive"] == 1) & (DF["ecog_ps"] == 2), "HER2+ ECOG=2")
# sacituzumab HER2-low + ER-
res["sacit_HER2low_ER-_adj"] = adjusted_subgroup_effect(
    "treatment_sacituzumab_govitecan",
    (DF["her2_low"] == 1) & (DF["er_positive"] == 0), "HER2-low & ER-")
# pembro: PIK3CA modifier, brain mets modifier
res["pembro_TNBC_brainmets"] = adjusted_subgroup_effect(
    "treatment_pembrolizumab",
    (DF["er_positive"] == 0) & (DF["pr_positive"] == 0) & (DF["her2_positive"] == 0)
    & (DF["has_brain_mets"] == 1), "TNBC + brain mets")
res["pembro_TNBC_noBrain"] = adjusted_subgroup_effect(
    "treatment_pembrolizumab",
    (DF["er_positive"] == 0) & (DF["pr_positive"] == 0) & (DF["her2_positive"] == 0)
    & (DF["has_brain_mets"] == 0), "TNBC no brain mets")
RESULTS["iter18"] = res

# ---------- ITERATION 19: Three-way interaction tests for treatments showing heterogeneity ----------
print("=== Iter 19 ===")
res = {}


def threeway(tr, m1, m2):
    cols = list(dict.fromkeys([tr, m1, m2] + ADJUSTERS))
    X = DF[cols].astype(float).copy()
    X[f"{tr}_x_{m1}"] = X[tr] * X[m1]
    X[f"{tr}_x_{m2}"] = X[tr] * X[m2]
    X[f"{m1}_x_{m2}"] = X[m1] * X[m2]
    X[f"{tr}_x_{m1}_x_{m2}"] = X[tr] * X[m1] * X[m2]
    X = add_const(X)
    y = DF[OUTCOME].astype(float)
    model = sm.OLS(y, X).fit()
    name = f"{tr}_x_{m1}_x_{m2}"
    return {
        "coef_3way": float(model.params[name]),
        "p_3way": float(model.pvalues[name]),
        "coef_t_x_m1": float(model.params[f"{tr}_x_{m1}"]),
        "p_t_x_m1": float(model.pvalues[f"{tr}_x_{m1}"]),
        "coef_t_x_m2": float(model.params[f"{tr}_x_{m2}"]),
        "p_t_x_m2": float(model.pvalues[f"{tr}_x_{m2}"]),
    }


res["palbo_x_ER_x_HER2neg"] = threeway("treatment_palbociclib", "er_positive", "her2_positive")
res["sacit_x_HER2low_x_ER"] = threeway("treatment_sacituzumab_govitecan", "her2_low", "er_positive")
res["pembro_x_ER_x_HER2"] = threeway("treatment_pembrolizumab", "er_positive", "her2_positive")
res["olap_x_BRCA1_x_BRCA2"] = threeway("treatment_olaparib", "brca1_mutation", "brca2_mutation")
res["trast_x_HER2_x_node"] = threeway("treatment_trastuzumab", "her2_positive", "node_positive")
RESULTS["iter19"] = res

# ---------- ITERATION 20: Pairwise binary modifier scan for each treatment ----------
print("=== Iter 20 ===")
binary_mods = [
    "sex_female", "stage_iv", "has_brain_mets", "node_positive", "postmenopausal",
    "er_positive", "pr_positive", "her2_positive", "her2_low",
    "brca1_mutation", "brca2_mutation", "pik3ca_mutation",
]
res = {}
for tr in TREATMENTS:
    rows = []
    for i, m1 in enumerate(binary_mods):
        for m2 in binary_mods[i + 1:]:
            mask_pp = (DF[m1] == 1) & (DF[m2] == 1)
            if mask_pp.sum() < 100:
                continue
            sub = DF.loc[mask_pp]
            if sub[tr].nunique() < 2:
                continue
            a = sub.loc[sub[tr] == 1, OUTCOME]
            b = sub.loc[sub[tr] == 0, OUTCOME]
            if len(a) < 25 or len(b) < 25:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append({
                "subgroup": f"{m1}=1 & {m2}=1",
                "n": int(mask_pp.sum()),
                "n_treated": int(len(a)),
                "effect": float(a.mean() - b.mean()),
                "p": float(p),
            })
    rows = sorted(rows, key=lambda r: r["p"])
    res[tr] = rows[:5]
RESULTS["iter20"] = res

# ---------- ITERATION 21: Continuous-modifier joint subgroup analysis ----------
print("=== Iter 21 ===")
res = {}
# Tamoxifen ER+ further stratified by age tertile
for tr, marker_mask, label in [
    ("treatment_tamoxifen", DF["er_positive"] == 1, "ER+"),
    ("treatment_palbociclib", (DF["er_positive"] == 1) & (DF["her2_positive"] == 0), "ER+/HER2-"),
    ("treatment_trastuzumab", DF["her2_positive"] == 1, "HER2+"),
    ("treatment_olaparib", (DF["brca1_mutation"] == 1) | (DF["brca2_mutation"] == 1), "BRCA+"),
    ("treatment_sacituzumab_govitecan", DF["her2_low"] == 1, "HER2-low"),
    ("treatment_pembrolizumab",
     (DF["er_positive"] == 0) & (DF["pr_positive"] == 0) & (DF["her2_positive"] == 0), "TNBC"),
]:
    sub_tert = []
    for tcol in ["age_years", "ecog_ps", "albumin_g_dl", "ldh_u_l"]:
        if tcol == "ecog_ps":
            buckets = [(0, "ECOG=0"), (1, "ECOG=1"), (2, "ECOG=2")]
            for v, name in buckets:
                m = marker_mask & (DF[tcol] == v)
                eff = subgroup_effect(tr, m, f"{label} & {name}")
                sub_tert.append(eff)
        else:
            qs = DF.loc[marker_mask, tcol].quantile([1 / 3, 2 / 3]).values
            buckets = [
                (DF[tcol] <= qs[0], f"{tcol}<=Q1"),
                ((DF[tcol] > qs[0]) & (DF[tcol] <= qs[1]), f"{tcol} Q2"),
                (DF[tcol] > qs[1], f"{tcol}>Q2"),
            ]
            for cond, name in buckets:
                m = marker_mask & cond
                eff = subgroup_effect(tr, m, f"{label} & {name}")
                sub_tert.append(eff)
    res[tr] = sub_tert
RESULTS["iter21"] = res

# ---------- ITERATION 22: Tree-style two-feature subgroup discovery for each treatment ----------
print("=== Iter 22 ===")


def best_two_feature_subgroup(tr, top_n=5):
    """For each treatment, search all (binary, binary) subgroups for largest |effect| with adequate n."""
    feats = binary_mods
    rows = []
    for i, f1 in enumerate(feats):
        for v1 in [0, 1]:
            for j, f2 in enumerate(feats):
                if j == i:
                    continue
                for v2 in [0, 1]:
                    mask = (DF[f1] == v1) & (DF[f2] == v2)
                    n = int(mask.sum())
                    if n < 200:
                        continue
                    sub = DF.loc[mask]
                    if sub[tr].nunique() < 2:
                        continue
                    a = sub.loc[sub[tr] == 1, OUTCOME]
                    b = sub.loc[sub[tr] == 0, OUTCOME]
                    if len(a) < 50 or len(b) < 50:
                        continue
                    t, p = stats.ttest_ind(a, b, equal_var=False)
                    rows.append({
                        "subgroup": f"{f1}={v1} & {f2}={v2}",
                        "n": n, "n_treated": int(len(a)),
                        "effect": float(a.mean() - b.mean()),
                        "p": float(p),
                    })
    rows = sorted(rows, key=lambda r: -abs(r["effect"]))
    return rows[:top_n]


res = {}
for tr in TREATMENTS:
    res[tr] = best_two_feature_subgroup(tr, top_n=8)
RESULTS["iter22"] = res

# ---------- ITERATION 23: Final candidate best-supported subgroup hypotheses ----------
print("=== Iter 23 ===")
# For each treatment, define a final candidate subgroup based on findings, then test (a) effect within
# subgroup, (b) effect outside, (c) interaction p-value in adjusted model.

def test_subgroup_full(tr, mask, label):
    sub_in = DF.loc[mask]
    sub_out = DF.loc[~mask]
    if sub_in[tr].nunique() < 2 or sub_out[tr].nunique() < 2:
        return None
    a_in = sub_in.loc[sub_in[tr] == 1, OUTCOME]
    b_in = sub_in.loc[sub_in[tr] == 0, OUTCOME]
    a_out = sub_out.loc[sub_out[tr] == 1, OUTCOME]
    b_out = sub_out.loc[sub_out[tr] == 0, OUTCOME]
    eff_in = a_in.mean() - b_in.mean()
    eff_out = a_out.mean() - b_out.mean()
    _, p_in = stats.ttest_ind(a_in, b_in, equal_var=False)
    _, p_out = stats.ttest_ind(a_out, b_out, equal_var=False)

    # Adjusted in-subgroup effect
    cols = list(dict.fromkeys([tr] + ADJUSTERS))
    X = sub_in[cols].astype(float)
    X = add_const(X)
    y = sub_in[OUTCOME].astype(float)
    m_in = sm.OLS(y, X).fit()
    adj_eff_in = float(m_in.params[tr])
    adj_p_in = float(m_in.pvalues[tr])

    # Interaction model: y ~ tr * I(in_subgroup) + adjusters
    DD = DF.copy()
    DD["in_sub"] = mask.astype(int).values
    cols2 = list(dict.fromkeys([tr, "in_sub"] + ADJUSTERS))
    X2 = DD[cols2].astype(float).copy()
    X2["tr_x_in"] = X2[tr] * X2["in_sub"]
    X2 = add_const(X2)
    y2 = DD[OUTCOME].astype(float)
    m2 = sm.OLS(y2, X2).fit()

    return {
        "label": label,
        "n_in": int(mask.sum()),
        "n_treated_in": int(len(a_in)),
        "eff_in": float(eff_in), "p_in": float(p_in),
        "adj_eff_in": adj_eff_in, "adj_p_in": adj_p_in,
        "n_out": int((~mask).sum()),
        "eff_out": float(eff_out), "p_out": float(p_out),
        "interaction_coef": float(m2.params["tr_x_in"]),
        "interaction_p": float(m2.pvalues["tr_x_in"]),
    }


res = {}
res["tamox_ER+"] = test_subgroup_full("treatment_tamoxifen", DF["er_positive"] == 1, "ER+")
res["palbo_ER+HER2-"] = test_subgroup_full(
    "treatment_palbociclib",
    (DF["er_positive"] == 1) & (DF["her2_positive"] == 0), "ER+/HER2-")
res["trast_HER2+"] = test_subgroup_full(
    "treatment_trastuzumab", DF["her2_positive"] == 1, "HER2+")
res["olap_BRCA"] = test_subgroup_full(
    "treatment_olaparib",
    (DF["brca1_mutation"] == 1) | (DF["brca2_mutation"] == 1), "BRCA1 or BRCA2 mutated")
res["sacit_HER2low"] = test_subgroup_full(
    "treatment_sacituzumab_govitecan", DF["her2_low"] == 1, "HER2-low")
res["sacit_HER2low_ER-"] = test_subgroup_full(
    "treatment_sacituzumab_govitecan",
    (DF["her2_low"] == 1) & (DF["er_positive"] == 0), "HER2-low & ER-")
res["pembro_TNBC"] = test_subgroup_full(
    "treatment_pembrolizumab",
    (DF["er_positive"] == 0) & (DF["pr_positive"] == 0) & (DF["her2_positive"] == 0), "TNBC")
RESULTS["iter23"] = res

# ---------- ITERATION 24: Sensitivity — drop confounders with strongest treatment-feature association ----------
print("=== Iter 24 ===")
res = {}
# Re-fit the global multivariable model and pull a clean treatment row + key biomarker rows
all_feats = (BIN_COLS + ["ecog_ps"] + CONT_COLS + TREATMENTS
             + ["treatment_tamoxifen_x_er_positive",
                "treatment_palbociclib_x_er_positive",
                "treatment_trastuzumab_x_her2_positive",
                "treatment_olaparib_x_brca_any",
                "treatment_sacituzumab_govitecan_x_her2_low",
                "treatment_pembrolizumab_x_tnbc"])
DD = DF.copy()
DD["brca_any"] = ((DD["brca1_mutation"] == 1) | (DD["brca2_mutation"] == 1)).astype(int)
DD["tnbc"] = ((DD["er_positive"] == 0) & (DD["pr_positive"] == 0)
              & (DD["her2_positive"] == 0)).astype(int)
DD["treatment_tamoxifen_x_er_positive"] = DD["treatment_tamoxifen"] * DD["er_positive"]
DD["treatment_palbociclib_x_er_positive"] = DD["treatment_palbociclib"] * DD["er_positive"]
DD["treatment_trastuzumab_x_her2_positive"] = DD["treatment_trastuzumab"] * DD["her2_positive"]
DD["treatment_olaparib_x_brca_any"] = DD["treatment_olaparib"] * DD["brca_any"]
DD["treatment_sacituzumab_govitecan_x_her2_low"] = (
    DD["treatment_sacituzumab_govitecan"] * DD["her2_low"])
DD["treatment_pembrolizumab_x_tnbc"] = DD["treatment_pembrolizumab"] * DD["tnbc"]
# Use brca_any/tnbc instead of original brca columns to avoid collinearity in interaction
all_feats = (
    [c for c in BIN_COLS if c not in ("brca1_mutation", "brca2_mutation")] +
    ["brca_any", "tnbc", "ecog_ps"] + CONT_COLS + TREATMENTS +
    ["treatment_tamoxifen_x_er_positive",
     "treatment_palbociclib_x_er_positive",
     "treatment_trastuzumab_x_her2_positive",
     "treatment_olaparib_x_brca_any",
     "treatment_sacituzumab_govitecan_x_her2_low",
     "treatment_pembrolizumab_x_tnbc"]
)
X = DD[all_feats].astype(float)
X = add_const(X)
y = DD[OUTCOME].astype(float)
model = sm.OLS(y, X).fit()
res["joint_interaction_model"] = {
    "r2": float(model.rsquared),
    "n": int(len(y)),
    "params": {k: {"coef": float(v), "p": float(model.pvalues[k]),
                   "se": float(model.bse[k])}
               for k, v in model.params.items()},
}
RESULTS["iter24"] = res

# ---------- ITERATION 25: Summary table — final best-supported subgroup per treatment ----------
print("=== Iter 25 ===")
res = {}
# Restate the strongest subgroup result per treatment using the iteration 23/24 output, with rounded numbers
final = {}
for tr, key in [
    ("treatment_tamoxifen", "tamox_ER+"),
    ("treatment_palbociclib", "palbo_ER+HER2-"),
    ("treatment_trastuzumab", "trast_HER2+"),
    ("treatment_olaparib", "olap_BRCA"),
    ("treatment_sacituzumab_govitecan", "sacit_HER2low"),
    ("treatment_pembrolizumab", "pembro_TNBC"),
]:
    final[tr] = RESULTS["iter23"].get(key)
res["final_subgroup_per_treatment"] = final
RESULTS["iter25"] = res

# Save
with open("my_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print("Done. Results saved.")
