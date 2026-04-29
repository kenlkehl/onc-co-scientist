"""Comprehensive analysis of ds001_prostate.

Runs analyses across ~25 iterations worth of hypotheses and dumps a JSON
results dict to my_analysis_results.json that the transcript builder will
consume.
"""

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
N = len(df)
print(f"Loaded {N} rows, {df.shape[1]} columns")

results = {}


def ttest_groups(name, mask_a, mask_b, label_a, label_b, outcome="pfs_months"):
    a = df.loc[mask_a, outcome].dropna()
    b = df.loc[mask_b, outcome].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    eff = a.mean() - b.mean()
    return {
        "name": name,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "label_a": label_a,
        "label_b": label_b,
        "effect": float(eff),
        "t": float(t),
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }


def linreg(name, formula, predictor):
    """Run a linear OLS via statsmodels.formula.api on df, return coefficient
    for `predictor` and p-value."""
    import statsmodels.formula.api as smf
    m = smf.ols(formula, data=df).fit()
    if predictor in m.params.index:
        coef = float(m.params[predictor])
        pval = float(m.pvalues[predictor])
    else:
        # Try matching by partial name
        keys = [k for k in m.params.index if predictor in k]
        if keys:
            coef = float(m.params[keys[0]])
            pval = float(m.pvalues[keys[0]])
        else:
            return None
    return {
        "name": name,
        "formula": formula,
        "predictor": predictor,
        "effect": coef,
        "p_value": pval,
        "significant": bool(pval < 0.05),
        "n": int(m.nobs),
        "rsq": float(m.rsquared),
    }


def interaction_test(name, predictor_a, predictor_b, covariates=None):
    """Test the interaction term predictor_a * predictor_b on pfs_months."""
    import statsmodels.formula.api as smf
    cov = ""
    if covariates:
        cov = " + " + " + ".join(covariates)
    formula = f"pfs_months ~ {predictor_a} * {predictor_b}{cov}"
    m = smf.ols(formula, data=df).fit()
    interaction_term = f"{predictor_a}:{predictor_b}"
    if interaction_term not in m.params.index:
        # Fallback: search for any name containing both
        keys = [k for k in m.params.index if predictor_a in k and predictor_b in k and ":" in k]
        if not keys:
            return None
        interaction_term = keys[0]
    return {
        "name": name,
        "formula": formula,
        "interaction_term": interaction_term,
        "effect": float(m.params[interaction_term]),
        "p_value": float(m.pvalues[interaction_term]),
        "significant": bool(m.pvalues[interaction_term] < 0.05),
        "main_a_effect": float(m.params.get(predictor_a, np.nan)),
        "main_b_effect": float(m.params.get(predictor_b, np.nan)),
    }


def stratified_treatment_effect(name, biomarker, treatment):
    """Compute treatment effect (difference in means) within biomarker+/- and
    test interaction."""
    import statsmodels.formula.api as smf
    pos = df[df[biomarker] == 1]
    neg = df[df[biomarker] == 0]
    eff_pos = pos.loc[pos[treatment] == 1, "pfs_months"].mean() - pos.loc[pos[treatment] == 0, "pfs_months"].mean()
    eff_neg = neg.loc[neg[treatment] == 1, "pfs_months"].mean() - neg.loc[neg[treatment] == 0, "pfs_months"].mean()
    formula = f"pfs_months ~ {biomarker} * {treatment}"
    m = smf.ols(formula, data=df).fit()
    inter = f"{biomarker}:{treatment}"
    return {
        "name": name,
        "biomarker": biomarker,
        "treatment": treatment,
        "effect_pos": float(eff_pos),
        "effect_neg": float(eff_neg),
        "interaction_effect": float(m.params.get(inter, np.nan)),
        "p_value": float(m.pvalues.get(inter, np.nan)),
        "significant": bool(m.pvalues.get(inter, 1.0) < 0.05),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
    }


# ==============================================================================
# ITERATION 1: Main effects of established prostate-cancer-relevant treatments
# ==============================================================================
print("\n=== Iteration 1: Treatment main effects ===")
it1 = []
for tx in ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    r = ttest_groups(f"{tx} main effect on PFS",
                     df[tx] == 1, df[tx] == 0, f"{tx}=1", f"{tx}=0")
    print(f"  {tx}: eff={r['effect']:.3f} p={r['p_value']:.3g}")
    it1.append(r)
results["iter1_treatment_main"] = it1

# ==============================================================================
# ITERATION 2: Established prognostic features (continuous → linear regression)
# ==============================================================================
print("\n=== Iteration 2: Univariate prognostic features ===")
it2 = []
for var in ["age_years", "ecog_ps", "psa_ng_ml", "gleason_score",
            "albumin_g_dl", "ldh_u_l", "hemoglobin_g_dl", "alkaline_phosphatase_u_l",
            "weight_loss_pct_6mo", "crp_mg_l", "nlr"]:
    r = linreg(f"PFS ~ {var}", f"pfs_months ~ {var}", var)
    print(f"  {var}: beta={r['effect']:.4f} p={r['p_value']:.3g}")
    it2.append(r)
results["iter2_prognostic_continuous"] = it2

# ==============================================================================
# ITERATION 3: Disease-state binary features (mCRPC, visceral mets, etc.)
# ==============================================================================
print("\n=== Iteration 3: Binary disease-state features ===")
it3 = []
for var in ["mcrpc", "visceral_mets", "liver_mets", "bone_mets", "adrenal_mets",
            "pleural_effusion", "pericardial_effusion"]:
    r = ttest_groups(f"PFS by {var}", df[var] == 1, df[var] == 0, f"{var}=1", f"{var}=0")
    print(f"  {var}: eff={r['effect']:.3f} p={r['p_value']:.3g}")
    it3.append(r)
results["iter3_disease_state"] = it3

# ==============================================================================
# ITERATION 4: Targeted-therapy biomarker interactions — the key clinical priors
# (BRCA2 × olaparib, AR-V7 × enzalutamide, AR-V7 × abiraterone, MSI × pembro,
#  PSMA × Lu-177)
# ==============================================================================
print("\n=== Iteration 4: Biomarker × treatment interactions (priors) ===")
it4 = []
key_priors = [
    ("brca2_mutation", "treatment_olaparib"),
    ("ar_v7_positive", "treatment_enzalutamide"),
    ("ar_v7_positive", "treatment_abiraterone"),
    ("msi_high", "treatment_pembrolizumab"),
    ("psma_high", "treatment_lu177_psma"),
]
for bio, tx in key_priors:
    r = stratified_treatment_effect(f"{bio} × {tx}", bio, tx)
    print(f"  {bio} × {tx}: pos_eff={r['effect_pos']:.3f}, neg_eff={r['effect_neg']:.3f}, inter_p={r['p_value']:.3g}")
    it4.append(r)
results["iter4_key_biomarker_tx_interactions"] = it4

# ==============================================================================
# ITERATION 5: Each treatment within mCRPC vs hormone-sensitive
# ==============================================================================
print("\n=== Iteration 5: Treatment × mCRPC ===")
it5 = []
for tx in ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    r = stratified_treatment_effect(f"mcrpc × {tx}", "mcrpc", tx)
    print(f"  mcrpc × {tx}: pos_eff={r['effect_pos']:.3f}, neg_eff={r['effect_neg']:.3f}, p={r['p_value']:.3g}")
    it5.append(r)
results["iter5_mcrpc_treatment"] = it5

# ==============================================================================
# ITERATION 6: Multivariable model — adjusted treatment effects
# ==============================================================================
print("\n=== Iteration 6: Multivariable adjusted treatment effects ===")
import statsmodels.formula.api as smf
adj_covs = "age_years + ecog_ps + mcrpc + visceral_mets + psa_ng_ml + gleason_score + albumin_g_dl + ldh_u_l + hemoglobin_g_dl"
it6 = []
for tx in ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    f = f"pfs_months ~ {tx} + {adj_covs}"
    m = smf.ols(f, data=df).fit()
    r = {
        "name": f"adjusted {tx}",
        "formula": f,
        "predictor": tx,
        "effect": float(m.params[tx]),
        "p_value": float(m.pvalues[tx]),
        "significant": bool(m.pvalues[tx] < 0.05),
        "rsq": float(m.rsquared),
    }
    print(f"  {tx} (adj): beta={r['effect']:.4f} p={r['p_value']:.3g}")
    it6.append(r)
results["iter6_adjusted_tx"] = it6

# ==============================================================================
# ITERATION 7: Re-test biomarker × treatment interactions after adjusting for
# clinical covariates (most important targeted ones)
# ==============================================================================
print("\n=== Iteration 7: Adjusted biomarker × treatment interactions ===")
it7 = []
for bio, tx in key_priors:
    f = f"pfs_months ~ {bio} * {tx} + age_years + ecog_ps + mcrpc + visceral_mets + psa_ng_ml + albumin_g_dl + ldh_u_l + hemoglobin_g_dl"
    m = smf.ols(f, data=df).fit()
    inter = f"{bio}:{tx}"
    r = {
        "name": f"adj {bio} × {tx}",
        "formula": f,
        "interaction_term": inter,
        "effect": float(m.params[inter]),
        "p_value": float(m.pvalues[inter]),
        "significant": bool(m.pvalues[inter] < 0.05),
        "main_bio": float(m.params[bio]),
        "main_tx": float(m.params[tx]),
    }
    print(f"  {bio}×{tx}: inter_beta={r['effect']:.4f} p={r['p_value']:.3g}")
    it7.append(r)
results["iter7_adjusted_interactions"] = it7

# ==============================================================================
# ITERATION 8: Race/ethnicity effects
# ==============================================================================
print("\n=== Iteration 8: Race/ethnicity ===")
it8 = []
re_means = df.groupby("race_ethnicity")["pfs_months"].agg(["mean", "count"]).reset_index()
print(re_means)
for re_val in df["race_ethnicity"].unique():
    if re_val == "white":
        continue
    a = df.loc[df["race_ethnicity"] == re_val, "pfs_months"]
    b = df.loc[df["race_ethnicity"] == "white", "pfs_months"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    r = {
        "name": f"PFS {re_val} vs white",
        "effect": float(a.mean() - b.mean()),
        "p_value": float(p),
        "significant": bool(p < 0.05),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }
    print(f"  {re_val} vs white: eff={r['effect']:.3f} p={r['p_value']:.3g}")
    it8.append(r)
# Also: ANOVA across all race/ethnicity groups
groups = [df.loc[df["race_ethnicity"] == g, "pfs_months"] for g in df["race_ethnicity"].unique()]
f_stat, p_anova = stats.f_oneway(*groups)
it8.append({
    "name": "ANOVA across race_ethnicity",
    "effect": float(f_stat),
    "p_value": float(p_anova),
    "significant": bool(p_anova < 0.05),
})
results["iter8_race"] = it8

# ==============================================================================
# ITERATION 9: Insurance type effects
# ==============================================================================
print("\n=== Iteration 9: Insurance type ===")
it9 = []
ins_means = df.groupby("insurance_type")["pfs_months"].agg(["mean", "count"]).reset_index()
print(ins_means)
for ins_val in ["medicare", "medicaid", "uninsured"]:
    a = df.loc[df["insurance_type"] == ins_val, "pfs_months"]
    b = df.loc[df["insurance_type"] == "private", "pfs_months"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    r = {
        "name": f"PFS {ins_val} vs private",
        "effect": float(a.mean() - b.mean()),
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }
    print(f"  {ins_val} vs private: eff={r['effect']:.3f} p={r['p_value']:.3g}")
    it9.append(r)
results["iter9_insurance"] = it9

# ==============================================================================
# ITERATION 10: ECOG performance status × treatment interactions (e.g., ECOG 2 patients
# tolerate docetaxel less)
# ==============================================================================
print("\n=== Iteration 10: ECOG × treatment interactions ===")
it10 = []
for tx in ["treatment_docetaxel", "treatment_enzalutamide", "treatment_abiraterone",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    f = f"pfs_months ~ ecog_ps * {tx}"
    m = smf.ols(f, data=df).fit()
    inter = f"ecog_ps:{tx}"
    r = {
        "name": f"ecog_ps × {tx}",
        "formula": f,
        "effect": float(m.params[inter]),
        "p_value": float(m.pvalues[inter]),
        "significant": bool(m.pvalues[inter] < 0.05),
    }
    print(f"  ecog × {tx}: inter_beta={r['effect']:.4f} p={r['p_value']:.3g}")
    it10.append(r)
results["iter10_ecog_tx"] = it10

# ==============================================================================
# ITERATION 11: Visceral mets × treatment interactions (docetaxel often
# preferred for visceral mets in mCRPC)
# ==============================================================================
print("\n=== Iteration 11: Visceral mets × treatment interactions ===")
it11 = []
for tx in ["treatment_docetaxel", "treatment_enzalutamide", "treatment_abiraterone",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    r = stratified_treatment_effect(f"visceral_mets × {tx}", "visceral_mets", tx)
    print(f"  visceral × {tx}: pos_eff={r['effect_pos']:.3f}, neg_eff={r['effect_neg']:.3f}, p={r['p_value']:.3g}")
    it11.append(r)
results["iter11_visceral_tx"] = it11

# ==============================================================================
# ITERATION 12: Number of prior lines of therapy
# ==============================================================================
print("\n=== Iteration 12: Prior lines / prior therapy effects ===")
it12 = []
r = linreg("PFS ~ prior_lines_of_therapy", "pfs_months ~ prior_lines_of_therapy", "prior_lines_of_therapy")
print(f"  prior_lines_of_therapy: beta={r['effect']:.4f} p={r['p_value']:.3g}")
it12.append(r)
for var in ["prior_chemotherapy", "prior_radiation", "prior_surgery",
            "prior_immunotherapy", "prior_targeted_therapy"]:
    r = ttest_groups(f"PFS by {var}", df[var] == 1, df[var] == 0, f"{var}=1", f"{var}=0")
    print(f"  {var}: eff={r['effect']:.3f} p={r['p_value']:.3g}")
    it12.append(r)
results["iter12_prior_therapy"] = it12

# ==============================================================================
# ITERATION 13: Symptom burden (pain, fatigue, dyspnea, appetite_loss)
# ==============================================================================
print("\n=== Iteration 13: Symptom burden ===")
it13 = []
for var in ["pain_nrs", "fatigue_grade", "dyspnea_grade", "cough_grade", "appetite_loss_grade"]:
    r = linreg(f"PFS ~ {var}", f"pfs_months ~ {var}", var)
    print(f"  {var}: beta={r['effect']:.4f} p={r['p_value']:.3g}")
    it13.append(r)
results["iter13_symptoms"] = it13

# ==============================================================================
# ITERATION 14: SNPs main effects — large multiple testing burden
# ==============================================================================
print("\n=== Iteration 14: SNP main effects ===")
snps = [c for c in df.columns if c.startswith("snp_")]
it14 = []
for s in snps:
    r = linreg(f"PFS ~ {s}", f"pfs_months ~ {s}", s)
    it14.append(r)
sig_snps = [r for r in it14 if r["p_value"] < 0.05]
print(f"  {len(sig_snps)}/{len(it14)} SNPs nominally significant")
for r in sorted(it14, key=lambda x: x["p_value"])[:5]:
    print(f"    {r['predictor']}: beta={r['effect']:.4f} p={r['p_value']:.3g}")
results["iter14_snps"] = it14

# ==============================================================================
# ITERATION 15: Comorbidity effects (heart failure, CKD, COPD, etc.)
# ==============================================================================
print("\n=== Iteration 15: Comorbidity main effects ===")
it15 = []
for var in ["diabetes_mellitus", "hypertension", "copd", "chronic_kidney_disease",
            "heart_failure", "coronary_artery_disease", "atrial_fibrillation",
            "venous_thromboembolism_history", "autoimmune_disease",
            "depression_anxiety_diagnosis"]:
    r = ttest_groups(f"PFS by {var}", df[var] == 1, df[var] == 0, f"{var}=1", f"{var}=0")
    print(f"  {var}: eff={r['effect']:.3f} p={r['p_value']:.3g}")
    it15.append(r)
results["iter15_comorbidities"] = it15

# ==============================================================================
# ITERATION 16: Other tumor genomics (TP53, PTEN, PIK3CA, CDKN2A loss, FGFR)
# ==============================================================================
print("\n=== Iteration 16: Tumor genomic alterations ===")
it16 = []
for var in ["tp53_mutation", "pten_loss", "pik3ca_mutation", "cdkn2a_loss",
            "fgfr_alteration", "her2_amplification", "braf_v600e",
            "keap1_mutation"]:
    r = ttest_groups(f"PFS by {var}", df[var] == 1, df[var] == 0, f"{var}=1", f"{var}=0")
    print(f"  {var}: eff={r['effect']:.3f} p={r['p_value']:.3g}")
    it16.append(r)
results["iter16_genomics"] = it16

# ==============================================================================
# ITERATION 17: BRCA2 × olaparib stratified detailed analysis with confounder adjustment
# (Doubling-down on a known prior)
# ==============================================================================
print("\n=== Iteration 17: BRCA2 × olaparib detail ===")
it17 = []
for subgroup_var, subgroup_val in [("brca2_mutation", 1), ("brca2_mutation", 0)]:
    sub = df[df[subgroup_var] == subgroup_val]
    a = sub.loc[sub["treatment_olaparib"] == 1, "pfs_months"]
    b = sub.loc[sub["treatment_olaparib"] == 0, "pfs_months"]
    if len(a) > 5 and len(b) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        r = {
            "name": f"olaparib effect | {subgroup_var}={subgroup_val}",
            "effect": float(a.mean() - b.mean()),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "n_tx": int(len(a)),
            "n_no_tx": int(len(b)),
            "mean_tx": float(a.mean()),
            "mean_no_tx": float(b.mean()),
        }
        print(f"  {subgroup_var}={subgroup_val}: olaparib eff={r['effect']:.3f} p={r['p_value']:.3g}")
        it17.append(r)
results["iter17_brca_olaparib_detail"] = it17

# ==============================================================================
# ITERATION 18: AR-V7 × hormonal therapy detailed analysis
# ==============================================================================
print("\n=== Iteration 18: AR-V7 × androgen-receptor-targeted therapy detail ===")
it18 = []
for tx in ["treatment_enzalutamide", "treatment_abiraterone"]:
    for arv7 in [1, 0]:
        sub = df[df["ar_v7_positive"] == arv7]
        a = sub.loc[sub[tx] == 1, "pfs_months"]
        b = sub.loc[sub[tx] == 0, "pfs_months"]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        r = {
            "name": f"{tx} effect | ar_v7={arv7}",
            "effect": float(a.mean() - b.mean()),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "n_tx": int(len(a)),
            "n_no_tx": int(len(b)),
            "mean_tx": float(a.mean()),
            "mean_no_tx": float(b.mean()),
        }
        print(f"  ar_v7={arv7}, {tx}: eff={r['effect']:.3f} p={r['p_value']:.3g}")
        it18.append(r)
results["iter18_arv7_detail"] = it18

# ==============================================================================
# ITERATION 19: PSMA-high × Lu-177 PSMA detail
# ==============================================================================
print("\n=== Iteration 19: PSMA × Lu-177 detail ===")
it19 = []
for psma in [1, 0]:
    sub = df[df["psma_high"] == psma]
    a = sub.loc[sub["treatment_lu177_psma"] == 1, "pfs_months"]
    b = sub.loc[sub["treatment_lu177_psma"] == 0, "pfs_months"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    r = {
        "name": f"lu177 effect | psma_high={psma}",
        "effect": float(a.mean() - b.mean()),
        "p_value": float(p),
        "significant": bool(p < 0.05),
        "n_tx": int(len(a)),
        "n_no_tx": int(len(b)),
        "mean_tx": float(a.mean()),
        "mean_no_tx": float(b.mean()),
    }
    print(f"  psma_high={psma}: lu177 eff={r['effect']:.3f} p={r['p_value']:.3g}")
    it19.append(r)
results["iter19_psma_lu177_detail"] = it19

# ==============================================================================
# ITERATION 20: MSI-high × pembrolizumab detail
# ==============================================================================
print("\n=== Iteration 20: MSI × pembro detail ===")
it20 = []
for msi in [1, 0]:
    sub = df[df["msi_high"] == msi]
    a = sub.loc[sub["treatment_pembrolizumab"] == 1, "pfs_months"]
    b = sub.loc[sub["treatment_pembrolizumab"] == 0, "pfs_months"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    r = {
        "name": f"pembro effect | msi_high={msi}",
        "effect": float(a.mean() - b.mean()),
        "p_value": float(p),
        "significant": bool(p < 0.05),
        "n_tx": int(len(a)),
        "n_no_tx": int(len(b)),
        "mean_tx": float(a.mean()),
        "mean_no_tx": float(b.mean()),
    }
    print(f"  msi_high={msi}: pembro eff={r['effect']:.3f} p={r['p_value']:.3g}")
    it20.append(r)
results["iter20_msi_pembro_detail"] = it20

# ==============================================================================
# ITERATION 21: Vital sign and lab biomarker tests
# ==============================================================================
print("\n=== Iteration 21: Vital signs and labs ===")
it21 = []
for var in ["bmi", "systolic_bp_mmhg", "diastolic_bp_mmhg", "heart_rate_bpm",
            "spo2_pct", "creatinine_mg_dl", "bun_mg_dl", "sodium_meq_l",
            "potassium_meq_l", "calcium_mg_dl", "platelets_k_ul",
            "wbc_k_ul", "anc_k_ul", "alc_k_ul"]:
    r = linreg(f"PFS ~ {var}", f"pfs_months ~ {var}", var)
    print(f"  {var}: beta={r['effect']:.4f} p={r['p_value']:.3g}")
    it21.append(r)
results["iter21_vitals_labs"] = it21

# ==============================================================================
# ITERATION 22: Demographic/SES — rural residence, smoking, education
# ==============================================================================
print("\n=== Iteration 22: SES / demographics ===")
it22 = []
r = ttest_groups("PFS rural vs urban", df["rural_residence"] == 1, df["rural_residence"] == 0,
                 "rural=1", "rural=0")
print(f"  rural_residence: eff={r['effect']:.3f} p={r['p_value']:.3g}")
it22.append(r)
r = linreg("PFS ~ smoking_pack_years", "pfs_months ~ smoking_pack_years", "smoking_pack_years")
print(f"  smoking_pack_years: beta={r['effect']:.4f} p={r['p_value']:.3g}")
it22.append(r)
r = linreg("PFS ~ education_years", "pfs_months ~ education_years", "education_years")
print(f"  education_years: beta={r['effect']:.4f} p={r['p_value']:.3g}")
it22.append(r)
results["iter22_demographics"] = it22

# ==============================================================================
# ITERATION 23: Combination / multiplicity — does docetaxel + abiraterone differ from
# either alone? (CHAARTED/STAMPEDE prior)
# ==============================================================================
print("\n=== Iteration 23: Treatment combinations ===")
it23 = []
for combo_a, combo_b in [
    ("treatment_docetaxel", "treatment_abiraterone"),
    ("treatment_enzalutamide", "treatment_abiraterone"),
    ("treatment_docetaxel", "treatment_enzalutamide"),
]:
    r = interaction_test(f"{combo_a} × {combo_b}", combo_a, combo_b)
    if r:
        print(f"  {combo_a} × {combo_b}: inter_beta={r['effect']:.4f} p={r['p_value']:.3g}")
        it23.append(r)
results["iter23_combos"] = it23

# ==============================================================================
# ITERATION 24: Full multivariable model with all key covariates and the most
# important interactions
# ==============================================================================
print("\n=== Iteration 24: Full multivariable model ===")
formula_full = (
    "pfs_months ~ "
    "age_years + ecog_ps + mcrpc + visceral_mets + bone_mets + liver_mets + "
    "psa_ng_ml + gleason_score + albumin_g_dl + ldh_u_l + hemoglobin_g_dl + "
    "alkaline_phosphatase_u_l + crp_mg_l + nlr + weight_loss_pct_6mo + "
    "brca2_mutation + ar_v7_positive + msi_high + psma_high + tp53_mutation + "
    "pten_loss + treatment_enzalutamide + treatment_abiraterone + "
    "treatment_docetaxel + treatment_olaparib + treatment_lu177_psma + "
    "treatment_pembrolizumab + "
    "brca2_mutation:treatment_olaparib + ar_v7_positive:treatment_enzalutamide + "
    "ar_v7_positive:treatment_abiraterone + msi_high:treatment_pembrolizumab + "
    "psma_high:treatment_lu177_psma"
)
m_full = smf.ols(formula_full, data=df).fit()
print(f"  R² = {m_full.rsquared:.4f}")
it24 = []
for term in m_full.params.index:
    it24.append({
        "term": term,
        "effect": float(m_full.params[term]),
        "p_value": float(m_full.pvalues[term]),
        "significant": bool(m_full.pvalues[term] < 0.05),
    })
print("  Top 10 terms by significance:")
for r in sorted(it24, key=lambda x: x["p_value"])[:10]:
    print(f"    {r['term']:50s} eff={r['effect']:.4f} p={r['p_value']:.3g}")
results["iter24_full_model"] = {"rsquared": float(m_full.rsquared),
                                  "n": int(m_full.nobs),
                                  "terms": it24}

# ==============================================================================
# ITERATION 25: Robustness — race × treatment interactions; do biomarker effects
# vary by race?
# ==============================================================================
print("\n=== Iteration 25: Race × treatment interactions ===")
it25 = []
df["white"] = (df["race_ethnicity"] == "white").astype(int)
df["black"] = (df["race_ethnicity"] == "black").astype(int)
for tx in ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    f = f"pfs_months ~ black * {tx}"
    m = smf.ols(f, data=df).fit()
    inter = f"black:{tx}"
    r = {
        "name": f"black × {tx}",
        "effect": float(m.params[inter]),
        "p_value": float(m.pvalues[inter]),
        "significant": bool(m.pvalues[inter] < 0.05),
    }
    print(f"  black × {tx}: inter_beta={r['effect']:.4f} p={r['p_value']:.3g}")
    it25.append(r)
# Insurance × treatment interactions, e.g., uninsured vs private effect on
# olaparib (an expensive targeted drug)
df["uninsured"] = (df["insurance_type"] == "uninsured").astype(int)
for tx in ["treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    f = f"pfs_months ~ uninsured * {tx}"
    m = smf.ols(f, data=df).fit()
    inter = f"uninsured:{tx}"
    r = {
        "name": f"uninsured × {tx}",
        "effect": float(m.params[inter]),
        "p_value": float(m.pvalues[inter]),
        "significant": bool(m.pvalues[inter] < 0.05),
    }
    print(f"  uninsured × {tx}: inter_beta={r['effect']:.4f} p={r['p_value']:.3g}")
    it25.append(r)
results["iter25_equity"] = it25

# Save
with open("my_analysis_results.json", "w") as fh:
    json.dump(results, fh, indent=2, default=str)
print("\nSaved results to my_analysis_results.json")
