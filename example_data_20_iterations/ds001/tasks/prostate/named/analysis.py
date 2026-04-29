"""Iterative hypothesis-driven analysis of ds001_prostate.

We probe relationships between disease features, biomarkers, treatments and
progression-free survival (pfs_months). Outputs a list of (iter, hypotheses,
analyses) tuples consumed downstream to build transcript.json.
"""

import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
print("Shape:", df.shape, "PFS mean:", df["pfs_months"].mean().round(3))


# helper functions ----------------------------------------------------------
def ttest(group1, group0, label1, label0, outcome="pfs_months"):
    """Welch t-test on outcome between two groups (1 vs 0). Effect = mean1 - mean0."""
    a = group1[outcome].values
    b = group0[outcome].values
    res = stats.ttest_ind(a, b, equal_var=False)
    eff = float(np.mean(a) - np.mean(b))
    return {
        "result_summary": (
            f"Mean {outcome}: {np.mean(a):.3f} ({label1}, n={len(a)}) vs "
            f"{np.mean(b):.3f} ({label0}, n={len(b)}); diff={eff:.3f}, t={res.statistic:.2f}, p={res.pvalue:.3g}"
        ),
        "p_value": float(res.pvalue),
        "effect_estimate": eff,
        "significant": bool(res.pvalue < 0.05),
    }


def binary_split(col, outcome="pfs_months"):
    g1 = df[df[col] == 1]
    g0 = df[df[col] == 0]
    return ttest(g1, g0, f"{col}=1", f"{col}=0", outcome)


def ols_effect(formula, term, outcome="pfs_months"):
    model = smf.ols(formula, data=df).fit()
    coef = float(model.params.get(term, np.nan))
    pval = float(model.pvalues.get(term, np.nan))
    return {
        "result_summary": (
            f"OLS {formula}: coef[{term}]={coef:.4f}, p={pval:.3g}, n={int(model.nobs)}, R^2={model.rsquared:.4f}"
        ),
        "p_value": pval,
        "effect_estimate": coef,
        "significant": bool(pval < 0.05),
    }


def correlation(x, y="pfs_months"):
    r, p = stats.pearsonr(df[x], df[y])
    return {
        "result_summary": f"Pearson r({x}, {y})={r:.4f}, p={p:.3g}, n={len(df)}",
        "p_value": float(p),
        "effect_estimate": float(r),
        "significant": bool(p < 0.05),
    }


def subgroup_effect(treatment_col, subgroup_col, sub_val=1, outcome="pfs_months"):
    """Effect of treatment within subgroup (subgroup_col == sub_val)."""
    sub = df[df[subgroup_col] == sub_val]
    g1 = sub[sub[treatment_col] == 1]
    g0 = sub[sub[treatment_col] == 0]
    if len(g1) < 5 or len(g0) < 5:
        return {
            "result_summary": f"Insufficient sample within {subgroup_col}={sub_val}",
            "p_value": None,
            "effect_estimate": None,
            "significant": False,
        }
    res = stats.ttest_ind(g1[outcome], g0[outcome], equal_var=False)
    eff = float(g1[outcome].mean() - g0[outcome].mean())
    return {
        "result_summary": (
            f"Within {subgroup_col}={sub_val}: {outcome} {g1[outcome].mean():.3f} on {treatment_col}=1 "
            f"(n={len(g1)}) vs {g0[outcome].mean():.3f} off (n={len(g0)}); diff={eff:.3f}, p={res.pvalue:.3g}"
        ),
        "p_value": float(res.pvalue),
        "effect_estimate": eff,
        "significant": bool(res.pvalue < 0.05),
    }


def interaction_term(treatment_col, biomarker_col, outcome="pfs_months"):
    """Interaction term coefficient: pfs ~ T + B + T:B."""
    formula = f"{outcome} ~ {treatment_col} + {biomarker_col} + {treatment_col}:{biomarker_col}"
    m = smf.ols(formula, data=df).fit()
    term = f"{treatment_col}:{biomarker_col}"
    coef = float(m.params.get(term, np.nan))
    pval = float(m.pvalues.get(term, np.nan))
    return {
        "result_summary": (
            f"Interaction {term}: coef={coef:.4f}, p={pval:.3g}; "
            f"main {treatment_col}={m.params[treatment_col]:.3f} (p={m.pvalues[treatment_col]:.3g}), "
            f"main {biomarker_col}={m.params[biomarker_col]:.3f} (p={m.pvalues[biomarker_col]:.3g})"
        ),
        "p_value": pval,
        "effect_estimate": coef,
        "significant": bool(pval < 0.05),
    }


# Containers ----------------------------------------------------------------
results = []


def add_iteration(idx, hypotheses, analyses):
    results.append({"index": idx, "proposed_hypotheses": hypotheses, "analyses": analyses})


# === ITERATION 1: Disease severity main effects ===========================
hyps = [
    {"id": "h1.1", "text": "Patients with mCRPC (mcrpc=1) have shorter pfs_months than those without (mcrpc=0).", "kind": "novel"},
    {"id": "h1.2", "text": "Patients with visceral metastases (visceral_mets=1) have shorter pfs_months than those without.", "kind": "novel"},
    {"id": "h1.3", "text": "Higher ECOG performance status (ecog_ps) is associated with shorter pfs_months.", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h1.1"], "code": "ttest pfs by mcrpc", **binary_split("mcrpc")},
    {"hypothesis_ids": ["h1.2"], "code": "ttest pfs by visceral_mets", **binary_split("visceral_mets")},
    {"hypothesis_ids": ["h1.3"], "code": "OLS pfs ~ ecog_ps", **ols_effect("pfs_months ~ ecog_ps", "ecog_ps")},
]
add_iteration(1, hyps, ana)

# === ITERATION 2: Metastatic site burden ===================================
hyps = [
    {"id": "h2.1", "text": "Bone metastases (bone_mets=1) are associated with shorter pfs_months than no bone mets.", "kind": "novel"},
    {"id": "h2.2", "text": "Liver metastases (liver_mets=1) are associated with shorter pfs_months than no liver mets.", "kind": "novel"},
    {"id": "h2.3", "text": "A higher count of metastatic sites (bone+visceral+liver+adrenal) is associated with shorter pfs_months.", "kind": "novel"},
]
df["met_count"] = df[["bone_mets","visceral_mets","liver_mets","adrenal_mets"]].sum(axis=1)
ana = [
    {"hypothesis_ids": ["h2.1"], "code": "ttest pfs by bone_mets", **binary_split("bone_mets")},
    {"hypothesis_ids": ["h2.2"], "code": "ttest pfs by liver_mets", **binary_split("liver_mets")},
    {"hypothesis_ids": ["h2.3"], "code": "OLS pfs ~ met_count", **ols_effect("pfs_months ~ met_count", "met_count")},
]
add_iteration(2, hyps, ana)

# === ITERATION 3: Tumor markers ============================================
hyps = [
    {"id": "h3.1", "text": "Higher baseline psa_ng_ml is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h3.2", "text": "Higher gleason_score is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h3.3", "text": "Log-transformed PSA shows a stronger linear association with pfs_months than raw PSA.", "kind": "novel"},
]
df["log_psa"] = np.log1p(df["psa_ng_ml"])
ana = [
    {"hypothesis_ids": ["h3.1"], "code": "Pearson psa vs pfs", **correlation("psa_ng_ml")},
    {"hypothesis_ids": ["h3.2"], "code": "OLS pfs ~ gleason_score", **ols_effect("pfs_months ~ gleason_score", "gleason_score")},
    {"hypothesis_ids": ["h3.3"], "code": "Pearson log_psa vs pfs", **correlation("log_psa")},
]
add_iteration(3, hyps, ana)

# === ITERATION 4: Lab markers ==============================================
hyps = [
    {"id": "h4.1", "text": "Lower albumin_g_dl is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h4.2", "text": "Higher ldh_u_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h4.3", "text": "Higher alkaline_phosphatase_u_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h4.4", "text": "Lower hemoglobin_g_dl is associated with shorter pfs_months.", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h4.1"], "code": "Pearson albumin vs pfs", **correlation("albumin_g_dl")},
    {"hypothesis_ids": ["h4.2"], "code": "Pearson ldh vs pfs", **correlation("ldh_u_l")},
    {"hypothesis_ids": ["h4.3"], "code": "Pearson alk_phos vs pfs", **correlation("alkaline_phosphatase_u_l")},
    {"hypothesis_ids": ["h4.4"], "code": "Pearson hemoglobin vs pfs", **correlation("hemoglobin_g_dl")},
]
add_iteration(4, hyps, ana)

# === ITERATION 5: Inflammation / cachexia ==================================
hyps = [
    {"id": "h5.1", "text": "Higher crp_mg_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h5.2", "text": "Higher neutrophil-to-lymphocyte ratio (nlr) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h5.3", "text": "Greater weight_loss_pct_6mo is associated with shorter pfs_months.", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h5.1"], "code": "Pearson crp vs pfs", **correlation("crp_mg_l")},
    {"hypothesis_ids": ["h5.2"], "code": "Pearson nlr vs pfs", **correlation("nlr")},
    {"hypothesis_ids": ["h5.3"], "code": "Pearson weight_loss vs pfs", **correlation("weight_loss_pct_6mo")},
]
add_iteration(5, hyps, ana)

# === ITERATION 6: Symptom burden ===========================================
hyps = [
    {"id": "h6.1", "text": "Higher pain_nrs is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h6.2", "text": "Higher fatigue_grade is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h6.3", "text": "Higher appetite_loss_grade is associated with shorter pfs_months.", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h6.1"], "code": "Pearson pain vs pfs", **correlation("pain_nrs")},
    {"hypothesis_ids": ["h6.2"], "code": "Pearson fatigue vs pfs", **correlation("fatigue_grade")},
    {"hypothesis_ids": ["h6.3"], "code": "Pearson appetite_loss vs pfs", **correlation("appetite_loss_grade")},
]
add_iteration(6, hyps, ana)

# === ITERATION 7: Treatment main effects ===================================
hyps = [
    {"id": "h7.1", "text": "Patients receiving treatment_enzalutamide have longer pfs_months than those who do not.", "kind": "novel"},
    {"id": "h7.2", "text": "Patients receiving treatment_abiraterone have longer pfs_months than those who do not.", "kind": "novel"},
    {"id": "h7.3", "text": "Patients receiving treatment_docetaxel have longer pfs_months than those who do not.", "kind": "novel"},
    {"id": "h7.4", "text": "Patients receiving treatment_olaparib have longer pfs_months than those who do not.", "kind": "novel"},
    {"id": "h7.5", "text": "Patients receiving treatment_lu177_psma have longer pfs_months than those who do not.", "kind": "novel"},
    {"id": "h7.6", "text": "Patients receiving treatment_pembrolizumab have longer pfs_months than those who do not.", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h7.1"], "code": "ttest pfs by enzalutamide", **binary_split("treatment_enzalutamide")},
    {"hypothesis_ids": ["h7.2"], "code": "ttest pfs by abiraterone",  **binary_split("treatment_abiraterone")},
    {"hypothesis_ids": ["h7.3"], "code": "ttest pfs by docetaxel",    **binary_split("treatment_docetaxel")},
    {"hypothesis_ids": ["h7.4"], "code": "ttest pfs by olaparib",     **binary_split("treatment_olaparib")},
    {"hypothesis_ids": ["h7.5"], "code": "ttest pfs by lu177_psma",   **binary_split("treatment_lu177_psma")},
    {"hypothesis_ids": ["h7.6"], "code": "ttest pfs by pembrolizumab",**binary_split("treatment_pembrolizumab")},
]
add_iteration(7, hyps, ana)

# === ITERATION 8: Biomarker main effects on PFS ============================
hyps = [
    {"id": "h8.1", "text": "BRCA2 mutation carriers (brca2_mutation=1) have different pfs_months than non-carriers.", "kind": "novel"},
    {"id": "h8.2", "text": "AR-V7 positive patients (ar_v7_positive=1) have shorter pfs_months than AR-V7 negative.", "kind": "novel"},
    {"id": "h8.3", "text": "MSI-high patients (msi_high=1) have different pfs_months than MSI-stable.", "kind": "novel"},
    {"id": "h8.4", "text": "PSMA-high patients (psma_high=1) have different pfs_months than PSMA-low.", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h8.1"], "code": "ttest pfs by brca2_mutation", **binary_split("brca2_mutation")},
    {"hypothesis_ids": ["h8.2"], "code": "ttest pfs by ar_v7_positive", **binary_split("ar_v7_positive")},
    {"hypothesis_ids": ["h8.3"], "code": "ttest pfs by msi_high",        **binary_split("msi_high")},
    {"hypothesis_ids": ["h8.4"], "code": "ttest pfs by psma_high",       **binary_split("psma_high")},
]
add_iteration(8, hyps, ana)

# === ITERATION 9: Olaparib × BRCA2 (biomarker-targeted PARP inhibitor) =====
hyps = [
    {"id": "h9.1", "text": "Within BRCA2 mutation carriers, treatment_olaparib is associated with longer pfs_months than no olaparib.", "kind": "novel"},
    {"id": "h9.2", "text": "Within BRCA2 wild-type patients, treatment_olaparib has minimal/no benefit on pfs_months.", "kind": "novel"},
    {"id": "h9.3", "text": "There is a positive interaction between treatment_olaparib and brca2_mutation on pfs_months (olaparib benefit larger in BRCA2 carriers).", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h9.1"], "code": "subgroup olaparib in brca2=1", **subgroup_effect("treatment_olaparib","brca2_mutation",1)},
    {"hypothesis_ids": ["h9.2"], "code": "subgroup olaparib in brca2=0", **subgroup_effect("treatment_olaparib","brca2_mutation",0)},
    {"hypothesis_ids": ["h9.3"], "code": "interaction olaparib:brca2", **interaction_term("treatment_olaparib","brca2_mutation")},
]
add_iteration(9, hyps, ana)

# === ITERATION 10: Pembrolizumab × MSI-high ================================
hyps = [
    {"id": "h10.1", "text": "Within MSI-high patients, treatment_pembrolizumab is associated with longer pfs_months than no pembrolizumab.", "kind": "novel"},
    {"id": "h10.2", "text": "Within MSI-stable patients, treatment_pembrolizumab does not improve pfs_months.", "kind": "novel"},
    {"id": "h10.3", "text": "There is a positive interaction between treatment_pembrolizumab and msi_high on pfs_months.", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h10.1"], "code": "subgroup pembro in msi_high=1", **subgroup_effect("treatment_pembrolizumab","msi_high",1)},
    {"hypothesis_ids": ["h10.2"], "code": "subgroup pembro in msi_high=0", **subgroup_effect("treatment_pembrolizumab","msi_high",0)},
    {"hypothesis_ids": ["h10.3"], "code": "interaction pembro:msi_high", **interaction_term("treatment_pembrolizumab","msi_high")},
]
add_iteration(10, hyps, ana)

# === ITERATION 11: Lu177-PSMA × PSMA-high ==================================
hyps = [
    {"id": "h11.1", "text": "Within PSMA-high patients, treatment_lu177_psma is associated with longer pfs_months than no Lu177-PSMA.", "kind": "novel"},
    {"id": "h11.2", "text": "Within PSMA-low patients, treatment_lu177_psma does not improve pfs_months.", "kind": "novel"},
    {"id": "h11.3", "text": "There is a positive interaction between treatment_lu177_psma and psma_high on pfs_months.", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h11.1"], "code": "subgroup lu177 in psma_high=1", **subgroup_effect("treatment_lu177_psma","psma_high",1)},
    {"hypothesis_ids": ["h11.2"], "code": "subgroup lu177 in psma_high=0", **subgroup_effect("treatment_lu177_psma","psma_high",0)},
    {"hypothesis_ids": ["h11.3"], "code": "interaction lu177:psma_high", **interaction_term("treatment_lu177_psma","psma_high")},
]
add_iteration(11, hyps, ana)

# === ITERATION 12: AR-V7 resistance to AR-targeted therapies ==============
hyps = [
    {"id": "h12.1", "text": "Within AR-V7 positive patients, treatment_enzalutamide is associated with shorter or equal pfs_months compared to no enzalutamide (i.e., reduced benefit relative to AR-V7 negative).", "kind": "novel"},
    {"id": "h12.2", "text": "Within AR-V7 negative patients, treatment_enzalutamide is associated with longer pfs_months than no enzalutamide.", "kind": "novel"},
    {"id": "h12.3", "text": "There is a negative interaction between treatment_enzalutamide and ar_v7_positive on pfs_months (less benefit in AR-V7 positive).", "kind": "novel"},
    {"id": "h12.4", "text": "There is a negative interaction between treatment_abiraterone and ar_v7_positive on pfs_months.", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h12.1"], "code": "subgroup enza in ar_v7=1", **subgroup_effect("treatment_enzalutamide","ar_v7_positive",1)},
    {"hypothesis_ids": ["h12.2"], "code": "subgroup enza in ar_v7=0", **subgroup_effect("treatment_enzalutamide","ar_v7_positive",0)},
    {"hypothesis_ids": ["h12.3"], "code": "interaction enza:ar_v7", **interaction_term("treatment_enzalutamide","ar_v7_positive")},
    {"hypothesis_ids": ["h12.4"], "code": "interaction abi:ar_v7",  **interaction_term("treatment_abiraterone","ar_v7_positive")},
]
add_iteration(12, hyps, ana)

# === ITERATION 13: Docetaxel × visceral mets ================================
hyps = [
    {"id": "h13.1", "text": "There is a different pfs_months effect of treatment_docetaxel between patients with and without visceral_mets (interaction).", "kind": "novel"},
    {"id": "h13.2", "text": "Within mCRPC patients, treatment_docetaxel is associated with longer pfs_months than no docetaxel.", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h13.1"], "code": "interaction docetaxel:visceral_mets", **interaction_term("treatment_docetaxel","visceral_mets")},
    {"hypothesis_ids": ["h13.2"], "code": "subgroup docetaxel in mcrpc=1", **subgroup_effect("treatment_docetaxel","mcrpc",1)},
]
add_iteration(13, hyps, ana)

# === ITERATION 14: Demographics ============================================
hyps = [
    {"id": "h14.1", "text": "Older age (age_years) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h14.2", "text": "pfs_months differs across race_ethnicity categories.", "kind": "novel"},
    {"id": "h14.3", "text": "pfs_months differs across insurance_type categories (e.g., medicaid/uninsured worse than private/medicare).", "kind": "novel"},
    {"id": "h14.4", "text": "Patients with rural_residence=1 have shorter pfs_months than urban (rural_residence=0).", "kind": "novel"},
]
def anova_by(col):
    groups = [g["pfs_months"].values for _, g in df.groupby(col)]
    res = stats.f_oneway(*groups)
    means = df.groupby(col)["pfs_months"].mean().to_dict()
    eff = float(max(means.values()) - min(means.values()))
    return {
        "result_summary": f"ANOVA pfs by {col}: F={res.statistic:.3f}, p={res.pvalue:.3g}, group means={ {k:round(v,3) for k,v in means.items()} }",
        "p_value": float(res.pvalue),
        "effect_estimate": eff,
        "significant": bool(res.pvalue < 0.05),
    }
ana = [
    {"hypothesis_ids": ["h14.1"], "code": "Pearson age vs pfs", **correlation("age_years")},
    {"hypothesis_ids": ["h14.2"], "code": "ANOVA pfs by race", **anova_by("race_ethnicity")},
    {"hypothesis_ids": ["h14.3"], "code": "ANOVA pfs by insurance", **anova_by("insurance_type")},
    {"hypothesis_ids": ["h14.4"], "code": "ttest pfs by rural", **binary_split("rural_residence")},
]
add_iteration(14, hyps, ana)

# === ITERATION 15: Comorbidities ===========================================
hyps = [
    {"id": "h15.1", "text": "Patients with chronic_kidney_disease=1 have shorter pfs_months than without.", "kind": "novel"},
    {"id": "h15.2", "text": "Patients with heart_failure=1 have shorter pfs_months than without.", "kind": "novel"},
    {"id": "h15.3", "text": "Patients with diabetes_mellitus=1 have shorter pfs_months than without.", "kind": "novel"},
    {"id": "h15.4", "text": "Patients with prior_malignancy=1 have shorter pfs_months than without.", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h15.1"], "code": "ttest pfs by ckd",         **binary_split("chronic_kidney_disease")},
    {"hypothesis_ids": ["h15.2"], "code": "ttest pfs by hf",          **binary_split("heart_failure")},
    {"hypothesis_ids": ["h15.3"], "code": "ttest pfs by dm",          **binary_split("diabetes_mellitus")},
    {"hypothesis_ids": ["h15.4"], "code": "ttest pfs by prior_malig", **binary_split("prior_malignancy")},
]
add_iteration(15, hyps, ana)

# === ITERATION 16: Prior therapy =========================================
hyps = [
    {"id": "h16.1", "text": "Higher prior_lines_of_therapy is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h16.2", "text": "Patients with prior_chemotherapy=1 have shorter pfs_months than chemo-naive.", "kind": "novel"},
    {"id": "h16.3", "text": "Longer years_since_diagnosis is associated with shorter pfs_months (more advanced/treated disease).", "kind": "novel"},
]
ana = [
    {"hypothesis_ids": ["h16.1"], "code": "Pearson prior_lines vs pfs", **correlation("prior_lines_of_therapy")},
    {"hypothesis_ids": ["h16.2"], "code": "ttest pfs by prior_chemo",   **binary_split("prior_chemotherapy")},
    {"hypothesis_ids": ["h16.3"], "code": "Pearson years_since_dx vs pfs", **correlation("years_since_diagnosis")},
]
add_iteration(16, hyps, ana)

# === ITERATION 17: Multivariable adjusted treatment effects ===============
# After learning main effects, control for major prognostic factors.
hyps = [
    {"id": "h17.1", "text": "Adjusted for ECOG, mCRPC, visceral_mets, log_psa, ldh, albumin, the coefficient for treatment_olaparib remains positive (longer pfs_months).", "kind": "refined"},
    {"id": "h17.2", "text": "Adjusted for the same prognostic covariates, the coefficient for treatment_lu177_psma remains positive.", "kind": "refined"},
    {"id": "h17.3", "text": "Adjusted for the same prognostic covariates, the coefficient for treatment_pembrolizumab is small/non-significant overall (effect concentrated in MSI-high).", "kind": "refined"},
]
adj = "ecog_ps + mcrpc + visceral_mets + log_psa + ldh_u_l + albumin_g_dl + age_years"
ana = [
    {"hypothesis_ids": ["h17.1"], "code": f"OLS pfs ~ olaparib + {adj}", **ols_effect(f"pfs_months ~ treatment_olaparib + {adj}", "treatment_olaparib")},
    {"hypothesis_ids": ["h17.2"], "code": f"OLS pfs ~ lu177 + {adj}",    **ols_effect(f"pfs_months ~ treatment_lu177_psma + {adj}", "treatment_lu177_psma")},
    {"hypothesis_ids": ["h17.3"], "code": f"OLS pfs ~ pembro + {adj}",   **ols_effect(f"pfs_months ~ treatment_pembrolizumab + {adj}", "treatment_pembrolizumab")},
]
add_iteration(17, hyps, ana)

# === ITERATION 18: Olaparib×BRCA2 adjusted interaction ====================
hyps = [
    {"id": "h18.1", "text": "After adjusting for ecog_ps, mcrpc, visceral_mets, ldh, albumin, the interaction treatment_olaparib:brca2_mutation remains positive and significant on pfs_months.", "kind": "refined"},
    {"id": "h18.2", "text": "After adjustment, the interaction treatment_pembrolizumab:msi_high remains positive and significant on pfs_months.", "kind": "refined"},
    {"id": "h18.3", "text": "After adjustment, the interaction treatment_lu177_psma:psma_high remains positive and significant on pfs_months.", "kind": "refined"},
]
def adj_interaction(t,b):
    f = (f"pfs_months ~ {t} + {b} + {t}:{b} + ecog_ps + mcrpc + visceral_mets + "
         "log_psa + ldh_u_l + albumin_g_dl + age_years")
    return ols_effect(f, f"{t}:{b}")
ana = [
    {"hypothesis_ids": ["h18.1"], "code": "adj OLS olaparib:brca2", **adj_interaction("treatment_olaparib","brca2_mutation")},
    {"hypothesis_ids": ["h18.2"], "code": "adj OLS pembro:msi_high", **adj_interaction("treatment_pembrolizumab","msi_high")},
    {"hypothesis_ids": ["h18.3"], "code": "adj OLS lu177:psma_high", **adj_interaction("treatment_lu177_psma","psma_high")},
]
add_iteration(18, hyps, ana)

# === ITERATION 19: AR-V7 negative interaction (adjusted) ==================
hyps = [
    {"id": "h19.1", "text": "After adjustment, the interaction treatment_enzalutamide:ar_v7_positive on pfs_months is negative and significant (smaller benefit in AR-V7 positive).", "kind": "refined"},
    {"id": "h19.2", "text": "After adjustment, the interaction treatment_abiraterone:ar_v7_positive on pfs_months is negative and significant.", "kind": "refined"},
]
ana = [
    {"hypothesis_ids": ["h19.1"], "code": "adj OLS enza:ar_v7", **adj_interaction("treatment_enzalutamide","ar_v7_positive")},
    {"hypothesis_ids": ["h19.2"], "code": "adj OLS abi:ar_v7",  **adj_interaction("treatment_abiraterone","ar_v7_positive")},
]
add_iteration(19, hyps, ana)

# === ITERATION 20: Race-stratified treatment access =======================
# Are subgroup-targeted treatments delivered equitably?
hyps = [
    {"id": "h20.1", "text": "The proportion of BRCA2-mutated patients receiving treatment_olaparib differs across race_ethnicity categories.", "kind": "novel"},
    {"id": "h20.2", "text": "The proportion of MSI-high patients receiving treatment_pembrolizumab differs across race_ethnicity categories.", "kind": "novel"},
    {"id": "h20.3", "text": "After adjusting for major prognostic factors, race_ethnicity remains associated with pfs_months.", "kind": "novel"},
]
def chi_within(sub_col, treat_col, group_col):
    sub = df[df[sub_col]==1]
    ct = pd.crosstab(sub[group_col], sub[treat_col])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    rates = sub.groupby(group_col)[treat_col].mean().to_dict()
    eff = float(max(rates.values()) - min(rates.values()))
    return {
        "result_summary": f"Within {sub_col}=1, {treat_col} rate by {group_col}: { {k:round(v,3) for k,v in rates.items()} }; chi2={chi2:.2f}, p={p:.3g}",
        "p_value": float(p),
        "effect_estimate": eff,
        "significant": bool(p<0.05),
    }
def race_adjusted_pfs():
    f = ("pfs_months ~ C(race_ethnicity) + ecog_ps + mcrpc + visceral_mets + "
         "log_psa + ldh_u_l + albumin_g_dl + age_years")
    m = smf.ols(f, data=df).fit()
    # get min p among race terms and report range of coefs
    race_terms = [t for t in m.params.index if t.startswith("C(race_ethnicity)")]
    coefs = {t:(float(m.params[t]), float(m.pvalues[t])) for t in race_terms}
    min_p = min(v[1] for v in coefs.values())
    eff_range = max(v[0] for v in coefs.values()) - min(v[0] for v in coefs.values())
    return {
        "result_summary": f"Adjusted OLS race terms: {coefs}, min p={min_p:.3g}",
        "p_value": float(min_p),
        "effect_estimate": float(eff_range),
        "significant": bool(min_p<0.05),
    }
ana = [
    {"hypothesis_ids": ["h20.1"], "code": "olaparib rate among brca2 by race", **chi_within("brca2_mutation","treatment_olaparib","race_ethnicity")},
    {"hypothesis_ids": ["h20.2"], "code": "pembro rate among msi_high by race", **chi_within("msi_high","treatment_pembrolizumab","race_ethnicity")},
    {"hypothesis_ids": ["h20.3"], "code": "adj OLS pfs ~ race + covars", **race_adjusted_pfs()},
]
add_iteration(20, hyps, ana)

# === ITERATION 21: Insurance disparities ==================================
hyps = [
    {"id": "h21.1", "text": "Receipt of treatment_lu177_psma among PSMA-high patients differs by insurance_type.", "kind": "novel"},
    {"id": "h21.2", "text": "Receipt of treatment_olaparib among BRCA2 carriers differs by insurance_type.", "kind": "novel"},
    {"id": "h21.3", "text": "After adjusting for prognostic factors, insurance_type is associated with pfs_months.", "kind": "novel"},
]
def insurance_adjusted_pfs():
    f = ("pfs_months ~ C(insurance_type) + ecog_ps + mcrpc + visceral_mets + "
         "log_psa + ldh_u_l + albumin_g_dl + age_years")
    m = smf.ols(f, data=df).fit()
    terms = [t for t in m.params.index if t.startswith("C(insurance_type)")]
    coefs = {t:(float(m.params[t]), float(m.pvalues[t])) for t in terms}
    min_p = min(v[1] for v in coefs.values())
    eff_range = max(v[0] for v in coefs.values()) - min(v[0] for v in coefs.values())
    return {
        "result_summary": f"Adjusted OLS insurance terms: {coefs}, min p={min_p:.3g}",
        "p_value": float(min_p),
        "effect_estimate": float(eff_range),
        "significant": bool(min_p<0.05),
    }
ana = [
    {"hypothesis_ids": ["h21.1"], "code": "lu177 rate in psma_high by insurance", **chi_within("psma_high","treatment_lu177_psma","insurance_type")},
    {"hypothesis_ids": ["h21.2"], "code": "olaparib rate in brca2 by insurance",   **chi_within("brca2_mutation","treatment_olaparib","insurance_type")},
    {"hypothesis_ids": ["h21.3"], "code": "adj OLS pfs ~ insurance + covars",     **insurance_adjusted_pfs()},
]
add_iteration(21, hyps, ana)

# === ITERATION 22: SNP main effects (illustrative null) ====================
hyps = [
    {"id": "h22.1", "text": "snp_rs1045642 is associated with pfs_months.", "kind": "novel"},
    {"id": "h22.2", "text": "snp_rs429358 (APOE) is associated with pfs_months.", "kind": "novel"},
    {"id": "h22.3", "text": "Across many tested SNPs, the proportion of significant associations approximates the 5% nominal type-I rate (most SNPs are noise).", "kind": "novel"},
]
def snp_screen():
    snp_cols = [c for c in df.columns if c.startswith("snp_")]
    pvals=[]
    for c in snp_cols:
        m = smf.ols(f"pfs_months ~ {c}", data=df).fit()
        pvals.append(m.pvalues[c])
    arr = np.array(pvals)
    sig_rate = float((arr < 0.05).mean())
    return {
        "result_summary": f"Tested {len(arr)} SNPs as continuous predictors of pfs; fraction p<0.05 = {sig_rate:.3f} (expected ~0.05 under null).",
        "p_value": float(stats.binomtest(int((arr<0.05).sum()), len(arr), 0.05).pvalue),
        "effect_estimate": sig_rate,
        "significant": bool(stats.binomtest(int((arr<0.05).sum()), len(arr), 0.05).pvalue < 0.05),
    }
ana = [
    {"hypothesis_ids": ["h22.1"], "code": "OLS pfs ~ snp_rs1045642", **ols_effect("pfs_months ~ snp_rs1045642","snp_rs1045642")},
    {"hypothesis_ids": ["h22.2"], "code": "OLS pfs ~ snp_rs429358",  **ols_effect("pfs_months ~ snp_rs429358","snp_rs429358")},
    {"hypothesis_ids": ["h22.3"], "code": "Screen all SNPs", **snp_screen()},
]
add_iteration(22, hyps, ana)

# === ITERATION 23: Magnitude of biomarker-targeted benefit ================
# Quantify clinical magnitude: how large is the differential benefit?
hyps = [
    {"id": "h23.1", "text": "The differential benefit of olaparib in BRCA2 carriers vs non-carriers (difference in difference) is at least 1 month of pfs_months.", "kind": "refined"},
    {"id": "h23.2", "text": "The differential benefit of pembrolizumab in MSI-high vs MSI-stable patients is at least 1 month of pfs_months.", "kind": "refined"},
    {"id": "h23.3", "text": "The differential benefit of lu177_psma in PSMA-high vs PSMA-low patients is at least 1 month of pfs_months.", "kind": "refined"},
]
def did(t,b):
    a11 = df[(df[t]==1)&(df[b]==1)]["pfs_months"].mean()
    a01 = df[(df[t]==0)&(df[b]==1)]["pfs_months"].mean()
    a10 = df[(df[t]==1)&(df[b]==0)]["pfs_months"].mean()
    a00 = df[(df[t]==0)&(df[b]==0)]["pfs_months"].mean()
    diff_b1 = a11 - a01
    diff_b0 = a10 - a00
    did_eff = diff_b1 - diff_b0
    # interaction p-value
    m = smf.ols(f"pfs_months ~ {t} + {b} + {t}:{b}", data=df).fit()
    pval = float(m.pvalues[f"{t}:{b}"])
    return {
        "result_summary": (
            f"DiD on pfs (treat={t}, biomarker={b}): "
            f"{b}=1 effect={diff_b1:.3f}; {b}=0 effect={diff_b0:.3f}; DiD={did_eff:.3f}, interaction p={pval:.3g}"
        ),
        "p_value": pval,
        "effect_estimate": float(did_eff),
        "significant": bool(pval<0.05),
    }
ana = [
    {"hypothesis_ids": ["h23.1"], "code": "DiD olaparib×brca2", **did("treatment_olaparib","brca2_mutation")},
    {"hypothesis_ids": ["h23.2"], "code": "DiD pembro×msi_high", **did("treatment_pembrolizumab","msi_high")},
    {"hypothesis_ids": ["h23.3"], "code": "DiD lu177×psma_high", **did("treatment_lu177_psma","psma_high")},
]
add_iteration(23, hyps, ana)

# === ITERATION 24: Combined biomarker matching ============================
hyps = [
    {"id": "h24.1", "text": "Patients receiving any 'biomarker-matched' therapy (olaparib if BRCA2+, pembro if MSI-high, lu177-PSMA if PSMA-high) have longer pfs_months than patients who received those therapies without the corresponding biomarker.", "kind": "novel"},
    {"id": "h24.2", "text": "Within patients receiving treatment_olaparib, BRCA2 carriers have longer pfs_months than BRCA2 wild-type.", "kind": "refined"},
    {"id": "h24.3", "text": "Within patients receiving treatment_pembrolizumab, MSI-high patients have longer pfs_months than MSI-stable.", "kind": "refined"},
    {"id": "h24.4", "text": "Within patients receiving treatment_lu177_psma, PSMA-high patients have longer pfs_months than PSMA-low.", "kind": "refined"},
]
df["matched_olaparib"]   = (df["treatment_olaparib"]==1)   & (df["brca2_mutation"]==1)
df["unmatched_olaparib"] = (df["treatment_olaparib"]==1)   & (df["brca2_mutation"]==0)
df["matched_pembro"]     = (df["treatment_pembrolizumab"]==1) & (df["msi_high"]==1)
df["unmatched_pembro"]   = (df["treatment_pembrolizumab"]==1) & (df["msi_high"]==0)
df["matched_lu177"]      = (df["treatment_lu177_psma"]==1) & (df["psma_high"]==1)
df["unmatched_lu177"]    = (df["treatment_lu177_psma"]==1) & (df["psma_high"]==0)
df["any_matched"]   = df[["matched_olaparib","matched_pembro","matched_lu177"]].any(axis=1)
df["any_unmatched"] = df[["unmatched_olaparib","unmatched_pembro","unmatched_lu177"]].any(axis=1)
matched_subset = df[df["any_matched"] | df["any_unmatched"]]
res_match = stats.ttest_ind(
    matched_subset.loc[matched_subset["any_matched"]==True,"pfs_months"],
    matched_subset.loc[matched_subset["any_unmatched"]==True,"pfs_months"],
    equal_var=False)
eff_match = float(matched_subset.loc[matched_subset["any_matched"]==True,"pfs_months"].mean()
                  - matched_subset.loc[matched_subset["any_unmatched"]==True,"pfs_months"].mean())
ana = [
    {"hypothesis_ids": ["h24.1"], "code": "ttest matched vs unmatched receipt",
     "result_summary": f"Among recipients of olaparib/pembro/lu177, biomarker-matched n={int(matched_subset['any_matched'].sum())} mean pfs={matched_subset.loc[matched_subset['any_matched'],'pfs_months'].mean():.3f} vs unmatched n={int(matched_subset['any_unmatched'].sum())} mean pfs={matched_subset.loc[matched_subset['any_unmatched'],'pfs_months'].mean():.3f}; diff={eff_match:.3f}, p={res_match.pvalue:.3g}",
     "p_value": float(res_match.pvalue), "effect_estimate": eff_match, "significant": bool(res_match.pvalue<0.05)},
    {"hypothesis_ids": ["h24.2"], "code": "ttest pfs by brca2 within olaparib",
     **ttest(df[(df['treatment_olaparib']==1)&(df['brca2_mutation']==1)],
             df[(df['treatment_olaparib']==1)&(df['brca2_mutation']==0)],
             "olaparib+ brca2+", "olaparib+ brca2-")},
    {"hypothesis_ids": ["h24.3"], "code": "ttest pfs by msi_high within pembro",
     **ttest(df[(df['treatment_pembrolizumab']==1)&(df['msi_high']==1)],
             df[(df['treatment_pembrolizumab']==1)&(df['msi_high']==0)],
             "pembro+ msi_high+", "pembro+ msi_high-")},
    {"hypothesis_ids": ["h24.4"], "code": "ttest pfs by psma_high within lu177",
     **ttest(df[(df['treatment_lu177_psma']==1)&(df['psma_high']==1)],
             df[(df['treatment_lu177_psma']==1)&(df['psma_high']==0)],
             "lu177+ psma+", "lu177+ psma-")},
]
add_iteration(24, hyps, ana)

# === ITERATION 25: Final integrated multivariable model ===================
hyps = [
    {"id": "h25.1", "text": "In a multivariable OLS model including disease severity (ecog_ps, mcrpc, visceral_mets, log_psa, ldh, albumin), all six treatments, and the three biomarker×targeted-treatment interactions (olaparib×BRCA2, pembro×MSI-high, lu177×PSMA-high), each interaction term remains positive and significant on pfs_months while the AR-V7×enzalutamide interaction remains negative.", "kind": "refined"},
    {"id": "h25.2", "text": "In the same multivariable model, ecog_ps coefficient is negative and significant; albumin coefficient is positive and significant.", "kind": "refined"},
]
big = (
    "pfs_months ~ ecog_ps + mcrpc + visceral_mets + bone_mets + liver_mets + log_psa + "
    "ldh_u_l + albumin_g_dl + age_years + "
    "treatment_enzalutamide + treatment_abiraterone + treatment_docetaxel + treatment_olaparib + "
    "treatment_lu177_psma + treatment_pembrolizumab + brca2_mutation + msi_high + psma_high + ar_v7_positive + "
    "treatment_olaparib:brca2_mutation + treatment_pembrolizumab:msi_high + treatment_lu177_psma:psma_high + "
    "treatment_enzalutamide:ar_v7_positive"
)
big_model = smf.ols(big, data=df).fit()
def grab(term):
    return {
        "result_summary": (
            f"In integrated model, {term}: coef={big_model.params[term]:.4f}, "
            f"p={big_model.pvalues[term]:.3g} (R^2={big_model.rsquared:.4f}, n={int(big_model.nobs)})"
        ),
        "p_value": float(big_model.pvalues[term]),
        "effect_estimate": float(big_model.params[term]),
        "significant": bool(big_model.pvalues[term]<0.05),
    }
ana = [
    {"hypothesis_ids":["h25.1"], "code":"integrated OLS olaparib:brca2",  **grab("treatment_olaparib:brca2_mutation")},
    {"hypothesis_ids":["h25.1"], "code":"integrated OLS pembro:msi_high", **grab("treatment_pembrolizumab:msi_high")},
    {"hypothesis_ids":["h25.1"], "code":"integrated OLS lu177:psma_high", **grab("treatment_lu177_psma:psma_high")},
    {"hypothesis_ids":["h25.1"], "code":"integrated OLS enza:ar_v7",      **grab("treatment_enzalutamide:ar_v7_positive")},
    {"hypothesis_ids":["h25.2"], "code":"integrated OLS ecog_ps",         **grab("ecog_ps")},
    {"hypothesis_ids":["h25.2"], "code":"integrated OLS albumin",         **grab("albumin_g_dl")},
]
add_iteration(25, hyps, ana)

# Print summary -------------------------------------------------------------
for it in results:
    print(f"--- Iteration {it['index']} ---")
    for a in it["analyses"]:
        print(" ", a["result_summary"])

# Save raw structure
with open("_iterations.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\nDone. iterations:", len(results))
