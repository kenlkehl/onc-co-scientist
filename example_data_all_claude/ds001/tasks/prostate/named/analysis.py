"""Iterative statistical analysis of ds001_prostate."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import logit
import itertools

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")

TREATMENTS = [
    "treatment_enzalutamide",
    "treatment_abiraterone",
    "treatment_docetaxel",
    "treatment_olaparib",
    "treatment_lu177_psma",
    "treatment_pembrolizumab",
]
BIOMARKERS = ["brca2_mutation", "ar_v7_positive", "msi_high", "psma_high"]
BINARY_FEATS = ["mcrpc", "visceral_mets"] + BIOMARKERS
CONT_FEATS = [
    "age_years",
    "ecog_ps",
    "psa_ng_ml",
    "gleason_score",
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
ALL_FEATS = BINARY_FEATS + CONT_FEATS
OUTCOME = "objective_response"


def rd_chi2(d, group_col, outcome=OUTCOME):
    a = d.loc[d[group_col] == 1, outcome]
    b = d.loc[d[group_col] == 0, outcome]
    if len(a) < 5 or len(b) < 5:
        return None
    p_a, p_b = a.mean(), b.mean()
    table = pd.crosstab(d[group_col], d[outcome])
    chi2, p, _, _ = stats.chi2_contingency(table)
    return {
        "n_pos": int(len(a)),
        "n_neg": int(len(b)),
        "rate_pos": float(p_a),
        "rate_neg": float(p_b),
        "risk_diff": float(p_a - p_b),
        "p_value": float(p),
    }


def cont_logreg(d, feat, outcome=OUTCOME):
    X = sm.add_constant(d[[feat]])
    y = d[outcome]
    try:
        m = sm.Logit(y, X).fit(disp=False)
        return {
            "coef": float(m.params[feat]),
            "or": float(np.exp(m.params[feat])),
            "p_value": float(m.pvalues[feat]),
            "n": int(len(d)),
        }
    except Exception as e:
        return {"error": str(e)}


def interaction_test(d, treat, modifier, outcome=OUTCOME):
    """Logistic regression: outcome ~ treat + modifier + treat:modifier."""
    df_ = d[[treat, modifier, outcome]].copy()
    df_["t_x_m"] = df_[treat] * df_[modifier]
    X = sm.add_constant(df_[[treat, modifier, "t_x_m"]])
    y = df_[outcome]
    try:
        m = sm.Logit(y, X).fit(disp=False)
        return {
            "coef_treat": float(m.params[treat]),
            "coef_mod": float(m.params[modifier]),
            "coef_int": float(m.params["t_x_m"]),
            "p_int": float(m.pvalues["t_x_m"]),
        }
    except Exception as e:
        return {"error": str(e)}


def stratified_te(d, treat, group_col, group_val=1, outcome=OUTCOME):
    """Risk difference of treatment within a group (group_col == group_val)."""
    sub = d[d[group_col] == group_val]
    on = sub.loc[sub[treat] == 1, outcome]
    off = sub.loc[sub[treat] == 0, outcome]
    if len(on) < 5 or len(off) < 5:
        return None
    table = pd.crosstab(sub[treat], sub[outcome])
    if table.shape != (2, 2):
        return None
    chi2, p, _, _ = stats.chi2_contingency(table)
    return {
        "n": int(len(sub)),
        "n_on": int(len(on)),
        "n_off": int(len(off)),
        "rate_on": float(on.mean()),
        "rate_off": float(off.mean()),
        "risk_diff": float(on.mean() - off.mean()),
        "p_value": float(p),
    }


def stratified_te_continuous(d, treat, feat, threshold, direction=">=", outcome=OUTCOME):
    if direction == ">=":
        sub = d[d[feat] >= threshold]
    elif direction == "<=":
        sub = d[d[feat] <= threshold]
    else:
        raise ValueError("direction must be >= or <=")
    on = sub.loc[sub[treat] == 1, outcome]
    off = sub.loc[sub[treat] == 0, outcome]
    if len(on) < 5 or len(off) < 5:
        return None
    table = pd.crosstab(sub[treat], sub[outcome])
    if table.shape != (2, 2):
        return None
    chi2, p, _, _ = stats.chi2_contingency(table)
    return {
        "n": int(len(sub)),
        "n_on": int(len(on)),
        "n_off": int(len(off)),
        "rate_on": float(on.mean()),
        "rate_off": float(off.mean()),
        "risk_diff": float(on.mean() - off.mean()),
        "p_value": float(p),
    }


def adjusted_treatment_effect(d, treat, covariates, outcome=OUTCOME):
    """Logistic regression with treatment + covariates."""
    cols = [treat] + covariates
    X = sm.add_constant(d[cols])
    y = d[outcome]
    try:
        m = sm.Logit(y, X).fit(disp=False)
        return {
            "coef_treat": float(m.params[treat]),
            "or_treat": float(np.exp(m.params[treat])),
            "p_treat": float(m.pvalues[treat]),
            "n": int(len(d)),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("Loaded dataset:", df.shape)
    print("Outcome rate:", df[OUTCOME].mean())
