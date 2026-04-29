"""Comprehensive iterative analysis of ds001_breast.

Runs ~25 iterations of hypothesis tests on the breast cancer cohort and
prints results in JSON-serializable form so the harness transcript can be
assembled deterministically.
"""

import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
Y = "pfs_months"


def fmt(x, n=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float(round(x, n))


def ttest(group_col, label=None):
    a = df.loc[df[group_col] == 1, Y]
    b = df.loc[df[group_col] == 0, Y]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    eff = a.mean() - b.mean()
    return {
        "label": label or group_col,
        "n_pos": int((df[group_col] == 1).sum()),
        "n_neg": int((df[group_col] == 0).sum()),
        "mean_pos": fmt(a.mean()),
        "mean_neg": fmt(b.mean()),
        "effect": fmt(eff),
        "t": fmt(t),
        "p": fmt(p, 6),
        "sig": bool(p < 0.05),
    }


def ols(formula, label=None, term=None):
    m = smf.ols(formula, data=df).fit()
    if term is None:
        term = list(m.params.index)[1]
    return {
        "label": label or formula,
        "formula": formula,
        "term": term,
        "coef": fmt(m.params[term]),
        "se": fmt(m.bse[term]),
        "p": fmt(m.pvalues[term], 6),
        "sig": bool(m.pvalues[term] < 0.05),
        "n": int(m.nobs),
    }


def stratified_effect(treatment, modifier):
    """Treatment effect on PFS within modifier=1 vs modifier=0; reports interaction p."""
    out = {}
    for sub, name in [(1, "pos"), (0, "neg")]:
        sub_df = df[df[modifier] == sub]
        a = sub_df.loc[sub_df[treatment] == 1, Y]
        b = sub_df.loc[sub_df[treatment] == 0, Y]
        if len(a) > 5 and len(b) > 5:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            out[name] = {
                "n_tx": int(len(a)),
                "n_ctl": int(len(b)),
                "mean_tx": fmt(a.mean()),
                "mean_ctl": fmt(b.mean()),
                "effect": fmt(a.mean() - b.mean()),
                "p": fmt(p, 6),
            }
    f = f"{Y} ~ {treatment} * {modifier}"
    m = smf.ols(f, data=df).fit()
    inter = f"{treatment}:{modifier}"
    out["interaction"] = {
        "term": inter,
        "coef": fmt(m.params.get(inter)),
        "p": fmt(m.pvalues.get(inter), 6),
        "sig": bool(m.pvalues.get(inter, 1.0) < 0.05),
        "main_treatment_coef_in_modifier_neg": fmt(m.params.get(treatment)),
        "main_modifier_coef": fmt(m.params.get(modifier)),
    }
    return out


def cont_corr(col):
    r, p = stats.pearsonr(df[col], df[Y])
    return {"col": col, "r": fmt(r), "p": fmt(p, 6), "sig": bool(p < 0.05)}


results = {}


# Iter 1: ECOG PS — higher = worse PFS
results["i1"] = {
    "ecog_corr": cont_corr("ecog_ps"),
    "ecog_ols": ols(f"{Y} ~ ecog_ps"),
}

# Iter 2: Stage IV
results["i2"] = {"stage_iv_t": ttest("stage_iv")}

# Iter 3: Brain mets
results["i3"] = {"brain_mets_t": ttest("has_brain_mets")}

# Iter 4: Liver / bone mets
results["i4"] = {
    "liver_mets_t": ttest("liver_mets"),
    "bone_mets_t": ttest("bone_mets"),
    "pleural_effusion_t": ttest("pleural_effusion"),
}

# Iter 5: ER / PR positivity
results["i5"] = {
    "er_pos_t": ttest("er_positive"),
    "pr_pos_t": ttest("pr_positive"),
}

# Iter 6: HER2-positive main effect + trastuzumab interaction
results["i6"] = {
    "her2_pos_t": ttest("her2_positive"),
    "trastuzumab_t": ttest("treatment_trastuzumab"),
    "trast_x_her2pos": stratified_effect("treatment_trastuzumab", "her2_positive"),
}

# Iter 7: HER2-low + sacituzumab govitecan interaction
results["i7"] = {
    "her2_low_t": ttest("her2_low"),
    "sacituzumab_t": ttest("treatment_sacituzumab_govitecan"),
    "sg_x_her2low": stratified_effect("treatment_sacituzumab_govitecan", "her2_low"),
}

# Iter 8: BRCA1/2 + olaparib interaction
df["brca_any"] = ((df["brca1_mutation"] == 1) | (df["brca2_mutation"] == 1)).astype(int)
results["i8"] = {
    "brca_any_t": ttest("brca_any"),
    "olaparib_t": ttest("treatment_olaparib"),
    "olap_x_brca": stratified_effect("treatment_olaparib", "brca_any"),
    "olap_x_brca1": stratified_effect("treatment_olaparib", "brca1_mutation"),
    "olap_x_brca2": stratified_effect("treatment_olaparib", "brca2_mutation"),
}

# Iter 9: Tamoxifen + ER positivity interaction; postmenopausal subgroup
results["i9"] = {
    "tamoxifen_t": ttest("treatment_tamoxifen"),
    "tam_x_er": stratified_effect("treatment_tamoxifen", "er_positive"),
    "tam_x_postmeno": stratified_effect("treatment_tamoxifen", "postmenopausal"),
}

# Iter 10: Palbociclib + ER positivity (and HER2-negative ER+ subgroup)
df["er_pos_her2_neg"] = ((df["er_positive"] == 1) & (df["her2_positive"] == 0)).astype(int)
results["i10"] = {
    "palbociclib_t": ttest("treatment_palbociclib"),
    "palb_x_er": stratified_effect("treatment_palbociclib", "er_positive"),
    "palb_x_er_her2neg": stratified_effect("treatment_palbociclib", "er_pos_her2_neg"),
}

# Iter 11: Pembrolizumab + MSI-high
results["i11"] = {
    "pembro_t": ttest("treatment_pembrolizumab"),
    "pembro_x_msi": stratified_effect("treatment_pembrolizumab", "msi_high"),
    "pembro_x_tp53": stratified_effect("treatment_pembrolizumab", "tp53_mutation"),
}

# Iter 12: Albumin (continuous)
results["i12"] = {
    "alb_corr": cont_corr("albumin_g_dl"),
    "alb_ols": ols(f"{Y} ~ albumin_g_dl"),
}

# Iter 13: LDH, CRP, NLR (inflammation/tumor burden)
results["i13"] = {
    "ldh_corr": cont_corr("ldh_u_l"),
    "crp_corr": cont_corr("crp_mg_l"),
    "nlr_corr": cont_corr("nlr"),
}

# Iter 14: Age, BMI, weight loss
results["i14"] = {
    "age_corr": cont_corr("age_years"),
    "bmi_corr": cont_corr("bmi"),
    "wtloss_corr": cont_corr("weight_loss_pct_6mo"),
}

# Iter 15: Tumor size and Ki67 (proliferation)
results["i15"] = {
    "tumor_size_corr": cont_corr("tumor_size_cm"),
    "ki67_corr": cont_corr("ki67_pct"),
}

# Iter 16: Symptom burden (fatigue, pain, dyspnea)
results["i16"] = {
    "fatigue_corr": cont_corr("fatigue_grade"),
    "pain_corr": cont_corr("pain_nrs"),
    "dyspnea_corr": cont_corr("dyspnea_grade"),
    "appetite_corr": cont_corr("appetite_loss_grade"),
}

# Iter 17: Hemoglobin, platelets, ANC, ALC
results["i17"] = {
    "hgb_corr": cont_corr("hemoglobin_g_dl"),
    "plt_corr": cont_corr("platelets_k_ul"),
    "anc_corr": cont_corr("anc_k_ul"),
    "alc_corr": cont_corr("alc_k_ul"),
}

# Iter 18: PIK3CA + alpelisib? not in dataset, just main effect
results["i18"] = {
    "pik3ca_t": ttest("pik3ca_mutation"),
    "tp53_t": ttest("tp53_mutation"),
    "her2_amp_t": ttest("her2_amplification"),
}

# Iter 19: Prior lines & prior therapy histories
results["i19"] = {
    "prior_lines_corr": cont_corr("prior_lines_of_therapy"),
    "prior_chemo_t": ttest("prior_chemotherapy"),
    "prior_immuno_t": ttest("prior_immunotherapy"),
    "prior_targeted_t": ttest("prior_targeted_therapy"),
    "yrs_dx_corr": cont_corr("years_since_diagnosis"),
}

# Iter 20: SDOH — race/ethnicity, insurance, rural, education
results["i20"] = {
    "rural_t": ttest("rural_residence"),
    "education_corr": cont_corr("education_years"),
    "smoking_corr": cont_corr("smoking_pack_years"),
}
# Race/ethnicity F-test
m = smf.ols(f"{Y} ~ C(race_ethnicity)", data=df).fit()
results["i20"]["race_ftest"] = {
    "f": fmt(m.fvalue),
    "p": fmt(m.f_pvalue, 6),
    "sig": bool(m.f_pvalue < 0.05),
    "by_group": {str(k): fmt(v) for k, v in df.groupby("race_ethnicity")[Y].mean().items()},
}
m2 = smf.ols(f"{Y} ~ C(insurance_type)", data=df).fit()
results["i20"]["insurance_ftest"] = {
    "f": fmt(m2.fvalue),
    "p": fmt(m2.f_pvalue, 6),
    "sig": bool(m2.f_pvalue < 0.05),
    "by_group": {str(k): fmt(v) for k, v in df.groupby("insurance_type")[Y].mean().items()},
}

# Iter 21: Comorbidities (organ dysfunction, autoimmune)
results["i21"] = {
    "ckd_t": ttest("chronic_kidney_disease"),
    "hf_t": ttest("heart_failure"),
    "cad_t": ttest("coronary_artery_disease"),
    "ild_t": ttest("interstitial_lung_disease_history"),
    "autoimmune_t": ttest("autoimmune_disease"),
    "diabetes_t": ttest("diabetes_mellitus"),
}

# Iter 22: SNPs — quick screen of all snp_ columns vs PFS (sorted by p)
snp_cols = [c for c in df.columns if c.startswith("snp_rs")]
snp_results = []
for c in snp_cols:
    r, p = stats.pearsonr(df[c], df[Y])
    snp_results.append({"snp": c, "r": fmt(r), "p": fmt(p, 6)})
snp_results.sort(key=lambda r: r["p"])
results["i22"] = {"snp_top": snp_results[:10], "snp_bonferroni_thresh": fmt(0.05 / len(snp_cols), 6)}

# Iter 23: Multivariable model with key features — biggest predictors
features = [
    "ecog_ps", "stage_iv", "has_brain_mets", "liver_mets", "bone_mets",
    "albumin_g_dl", "ldh_u_l", "nlr", "weight_loss_pct_6mo",
    "er_positive", "pr_positive", "her2_positive", "her2_low",
    "ki67_pct", "tumor_size_cm",
    "treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
    "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
    "age_years", "prior_lines_of_therapy",
]
formula = f"{Y} ~ " + " + ".join(features)
m = smf.ols(formula, data=df).fit()
mv = []
for term in features:
    mv.append({
        "term": term,
        "coef": fmt(m.params[term]),
        "se": fmt(m.bse[term]),
        "p": fmt(m.pvalues[term], 6),
        "sig": bool(m.pvalues[term] < 0.05),
    })
results["i23"] = {"multivariable": mv, "r2": fmt(m.rsquared), "n": int(m.nobs)}

# Iter 24: Targeted-therapy cross-checks — wrong-target negative controls
# trastuzumab in HER2-NEGATIVE; olaparib in BRCA-WT
df["her2_neg"] = (df["her2_positive"] == 0).astype(int)
df["brca_wt"] = ((df["brca1_mutation"] == 0) & (df["brca2_mutation"] == 0)).astype(int)
results["i24"] = {}
for tx, mod, name in [
    ("treatment_trastuzumab", "her2_neg", "trastuzumab_in_her2neg"),
    ("treatment_olaparib", "brca_wt", "olaparib_in_brca_wt"),
    ("treatment_sacituzumab_govitecan", "her2_low", "sg_in_her2low"),
    ("treatment_palbociclib", "er_positive", "palb_in_er_pos"),
    ("treatment_tamoxifen", "er_positive", "tam_in_er_pos"),
    ("treatment_pembrolizumab", "msi_high", "pembro_in_msi_high"),
]:
    sub = df[df[mod] == 1]
    a = sub.loc[sub[tx] == 1, Y]; b = sub.loc[sub[tx] == 0, Y]
    if len(a) > 5 and len(b) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results["i24"][name] = {
            "n_tx": int(len(a)), "n_ctl": int(len(b)),
            "mean_tx": fmt(a.mean()), "mean_ctl": fmt(b.mean()),
            "effect": fmt(a.mean() - b.mean()), "p": fmt(p, 6),
            "sig": bool(p < 0.05),
        }

# Iter 25: Treatment effect adjusted for prognostic covariates (key targeted matches)
adjusters = "ecog_ps + stage_iv + has_brain_mets + liver_mets + albumin_g_dl + ldh_u_l + nlr + age_years"
results["i25"] = {}
for tx, mod, name in [
    ("treatment_trastuzumab", "her2_positive", "trast_adj_her2pos"),
    ("treatment_olaparib", "brca_any", "olap_adj_brca_any"),
    ("treatment_palbociclib", "er_pos_her2_neg", "palb_adj_er_her2neg"),
    ("treatment_sacituzumab_govitecan", "her2_low", "sg_adj_her2low"),
    ("treatment_pembrolizumab", "msi_high", "pembro_adj_msi"),
    ("treatment_tamoxifen", "er_positive", "tam_adj_erpos"),
]:
    sub = df[df[mod] == 1]
    f = f"{Y} ~ {tx} + {adjusters}"
    m = smf.ols(f, data=sub).fit()
    results["i25"][name] = {
        "n": int(m.nobs),
        "coef": fmt(m.params[tx]),
        "se": fmt(m.bse[tx]),
        "p": fmt(m.pvalues[tx], 6),
        "sig": bool(m.pvalues[tx] < 0.05),
    }

with open("results_collected.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps(results, indent=2, default=str))
