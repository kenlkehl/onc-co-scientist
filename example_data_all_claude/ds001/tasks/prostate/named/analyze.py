"""Comprehensive iterative analysis of ds001_prostate.

Runs many statistical tests and writes a JSON results bag that the
transcript builder consumes. Each test returns (label, n, effect, p, sig, summary).
"""
from __future__ import annotations
import json
import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

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
OUTCOME = "objective_response"

results: dict[str, dict] = {}


def store(key, **kw):
    results[key] = kw


def logistic(formula_y, formula_X):
    X = sm.add_constant(formula_X.astype(float), has_constant="add")
    model = sm.Logit(formula_y.astype(int), X).fit(disp=0, maxiter=200)
    return model


def diff_in_rates(mask):
    a = df.loc[mask, OUTCOME]
    b = df.loc[~mask, OUTCOME]
    rate_a = a.mean()
    rate_b = b.mean()
    diff = rate_a - rate_b
    counts = np.array([a.sum(), b.sum()])
    nobs = np.array([len(a), len(b)])
    if min(nobs) == 0:
        return None
    z, p = proportions_ztest(counts, nobs)
    return {
        "rate_a": float(rate_a),
        "rate_b": float(rate_b),
        "diff": float(diff),
        "z": float(z),
        "p": float(p),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "events_a": int(a.sum()),
        "events_b": int(b.sum()),
    }


# ===================================================================
# ITERATION 1 — overall response rate baselines: each treatment alone
# ===================================================================
for t in TREATMENTS:
    r = diff_in_rates(df[t] == 1)
    store(f"main_{t}", **r)

# ===================================================================
# ITERATION 2 — biomarker main effects on outcome
# ===================================================================
for b in BIOMARKERS:
    r = diff_in_rates(df[b] == 1)
    store(f"main_{b}", **r)

# ===================================================================
# ITERATION 3 — clinical/prognostic features on outcome
# ===================================================================
# ECOG (0/1/2) — compare 0 vs 1 vs 2
for e in [0, 1, 2]:
    r = diff_in_rates(df["ecog_ps"] == e)
    store(f"ecog_{e}_vs_rest", **r)

# Visceral mets and mcRPC
store("main_visceral_mets", **diff_in_rates(df["visceral_mets"] == 1))
store("main_mcrpc", **diff_in_rates(df["mcrpc"] == 1))

# Continuous covariates: logistic univariate
for col in [
    "age_years",
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
]:
    x = df[[col]].astype(float).copy()
    # use log transform on heavy-tailed labs
    if col in ("psa_ng_ml", "ldh_u_l", "crp_mg_l", "nlr", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl"):
        x[col] = np.log1p(x[col])
    try:
        m = logistic(df[OUTCOME], x)
        coef = float(m.params.iloc[1])
        p = float(m.pvalues.iloc[1])
        store(
            f"univ_{col}",
            coef=coef,
            p=p,
            log_transformed=col in (
                "psa_ng_ml",
                "ldh_u_l",
                "crp_mg_l",
                "nlr",
                "alkaline_phosphatase_u_l",
                "ast_u_l",
                "alt_u_l",
                "total_bilirubin_mg_dl",
            ),
        )
    except Exception as exc:
        store(f"univ_{col}", error=str(exc))

# ===================================================================
# ITERATION 4 — biomarker × matched-treatment interactions
# (the prostate-cancer canon: BRCA2↔olaparib, MSI↔pembro, PSMA↔Lu177,
#  AR-V7↔resistance to enza/abi)
# ===================================================================

def interact(treatment, biomarker, label=None):
    """Logistic with treatment, biomarker, treatment×biomarker."""
    X = pd.DataFrame({
        "T": df[treatment].astype(float),
        "B": df[biomarker].astype(float),
        "TB": df[treatment].astype(float) * df[biomarker].astype(float),
    })
    m = logistic(df[OUTCOME], X)
    coef_T = float(m.params["T"])
    p_T = float(m.pvalues["T"])
    coef_B = float(m.params["B"])
    p_B = float(m.pvalues["B"])
    coef_TB = float(m.params["TB"])
    p_TB = float(m.pvalues["TB"])
    # Stratified rates
    sub = df[df[biomarker] == 1]
    nosub = df[df[biomarker] == 0]
    sub_t = sub.loc[sub[treatment] == 1, OUTCOME].mean()
    sub_nt = sub.loc[sub[treatment] == 0, OUTCOME].mean()
    nosub_t = nosub.loc[nosub[treatment] == 1, OUTCOME].mean()
    nosub_nt = nosub.loc[nosub[treatment] == 0, OUTCOME].mean()
    return {
        "treatment": treatment,
        "biomarker": biomarker,
        "coef_T": coef_T,
        "p_T": p_T,
        "coef_B": coef_B,
        "p_B": p_B,
        "coef_TB": coef_TB,
        "p_TB": p_TB,
        "rate_b1_t1": float(sub_t) if not np.isnan(sub_t) else None,
        "rate_b1_t0": float(sub_nt) if not np.isnan(sub_nt) else None,
        "rate_b0_t1": float(nosub_t) if not np.isnan(nosub_t) else None,
        "rate_b0_t0": float(nosub_nt) if not np.isnan(nosub_nt) else None,
        "n_b1_t1": int((sub[treatment] == 1).sum()),
        "n_b1_t0": int((sub[treatment] == 0).sum()),
    }


PRIOR = [
    ("treatment_olaparib", "brca2_mutation"),
    ("treatment_pembrolizumab", "msi_high"),
    ("treatment_lu177_psma", "psma_high"),
    ("treatment_enzalutamide", "ar_v7_positive"),
    ("treatment_abiraterone", "ar_v7_positive"),
]
for t, b in PRIOR:
    store(f"interact_{t}_{b}", **interact(t, b))

# ===================================================================
# ITERATION 5 — full screen of every treatment × every biomarker
# ===================================================================
for t in TREATMENTS:
    for b in BIOMARKERS:
        key = f"screen_{t}_{b}"
        if key in results:
            continue
        store(key, **interact(t, b))

# ===================================================================
# ITERATION 6 — multivariable logistic with all features (response model)
# ===================================================================
features_cont = [
    "age_years",
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
features_bin = [
    "ecog_ps",
    "mcrpc",
    "visceral_mets",
    "brca2_mutation",
    "ar_v7_positive",
    "msi_high",
    "psma_high",
] + TREATMENTS

X_full = df[features_cont].astype(float).copy()
for c in ("psa_ng_ml", "ldh_u_l", "crp_mg_l", "nlr", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl"):
    X_full[c] = np.log1p(X_full[c])
X_full = pd.concat([X_full, df[features_bin].astype(float)], axis=1)

m_full = logistic(df[OUTCOME], X_full)
results["multivar_full"] = {
    "coefs": {k: float(v) for k, v in m_full.params.items()},
    "pvals": {k: float(v) for k, v in m_full.pvalues.items()},
    "loglik": float(m_full.llf),
    "n": int(m_full.nobs),
}

# ===================================================================
# ITERATION 7 — interaction in multivariable model: olaparib × BRCA2
# Goal: confirm interaction holds adjusting for prognostic factors
# ===================================================================

def adjusted_interaction(treatment, biomarker):
    base_cols = ["age_years", "ecog_ps", "albumin_g_dl", "psa_ng_ml", "ldh_u_l", "visceral_mets", "mcrpc", "gleason_score", "weight_loss_pct_6mo", "hemoglobin_g_dl", "nlr"]
    X = df[base_cols].astype(float).copy()
    for c in ("psa_ng_ml", "ldh_u_l", "nlr"):
        X[c] = np.log1p(X[c])
    # add other treatments as adjusters
    for t in TREATMENTS:
        if t != treatment:
            X[t] = df[t].astype(float)
    # add other biomarkers
    for b in BIOMARKERS:
        if b != biomarker:
            X[b] = df[b].astype(float)
    X["T"] = df[treatment].astype(float)
    X["B"] = df[biomarker].astype(float)
    X["TB"] = X["T"] * X["B"]
    m = logistic(df[OUTCOME], X)
    return {
        "coef_T": float(m.params["T"]),
        "p_T": float(m.pvalues["T"]),
        "coef_B": float(m.params["B"]),
        "p_B": float(m.pvalues["B"]),
        "coef_TB": float(m.params["TB"]),
        "p_TB": float(m.pvalues["TB"]),
    }


for t, b in PRIOR:
    store(f"adj_interact_{t}_{b}", **adjusted_interaction(t, b))

# ===================================================================
# ITERATION 8 — heterogeneity screen: each treatment × every biomarker AND
# every continuous-covariate split (above/below median)
# ===================================================================

# Median-split continuous covariates: capture sign of treatment effect by half
for t in TREATMENTS:
    for col in [
        "age_years",
        "ecog_ps",
        "albumin_g_dl",
        "ldh_u_l",
        "psa_ng_ml",
        "gleason_score",
        "alkaline_phosphatase_u_l",
        "hemoglobin_g_dl",
        "weight_loss_pct_6mo",
        "crp_mg_l",
        "nlr",
    ]:
        if col == "ecog_ps":
            high = df["ecog_ps"] >= 1
        else:
            med = df[col].median()
            high = df[col] > med
        # treatment effect in high vs low subgroup
        for label, mask in (("high", high), ("low", ~high)):
            sub = df[mask]
            if (sub[t] == 1).sum() < 20 or (sub[t] == 0).sum() < 20:
                continue
            r1 = sub.loc[sub[t] == 1, OUTCOME].mean()
            r0 = sub.loc[sub[t] == 0, OUTCOME].mean()
            counts = np.array([(sub.loc[sub[t] == 1, OUTCOME]).sum(), (sub.loc[sub[t] == 0, OUTCOME]).sum()])
            nobs = np.array([(sub[t] == 1).sum(), (sub[t] == 0).sum()])
            try:
                _, p = proportions_ztest(counts, nobs)
            except Exception:
                p = None
            store(
                f"het_{t}_{col}_{label}",
                rate_t=float(r1),
                rate_c=float(r0),
                diff=float(r1 - r0),
                p=float(p) if p is not None else None,
                n_t=int(nobs[0]),
                n_c=int(nobs[1]),
                covariate=col,
                level=label,
            )

# ===================================================================
# ITERATION 9 — multi-feature subgroup definitions for each canonical
# treatment-biomarker pair
# ===================================================================

# For olaparib + BRCA2: effect within ECOG=0/1 vs 2; visceral mets yes/no;
# albumin low/high; etc.
def stratified_effect(treatment, biomarker, modifier_mask, label):
    sub = df[(df[biomarker] == 1) & modifier_mask]
    if (sub[treatment] == 1).sum() < 10 or (sub[treatment] == 0).sum() < 10:
        return None
    a = sub.loc[sub[treatment] == 1, OUTCOME]
    b = sub.loc[sub[treatment] == 0, OUTCOME]
    r1 = a.mean()
    r0 = b.mean()
    counts = np.array([a.sum(), b.sum()])
    nobs = np.array([len(a), len(b)])
    try:
        _, p = proportions_ztest(counts, nobs)
    except Exception:
        p = None
    return {
        "rate_t": float(r1),
        "rate_c": float(r0),
        "diff": float(r1 - r0),
        "p": float(p) if p is not None else None,
        "n_t": int(nobs[0]),
        "n_c": int(nobs[1]),
        "label": label,
    }


def explore_modifiers(treatment, biomarker, prefix):
    modifier_specs = {
        "ecog0_1": df["ecog_ps"] <= 1,
        "ecog2": df["ecog_ps"] == 2,
        "visceral_no": df["visceral_mets"] == 0,
        "visceral_yes": df["visceral_mets"] == 1,
        "albumin_high": df["albumin_g_dl"] >= df["albumin_g_dl"].median(),
        "albumin_low": df["albumin_g_dl"] < df["albumin_g_dl"].median(),
        "ldh_low": df["ldh_u_l"] <= df["ldh_u_l"].median(),
        "ldh_high": df["ldh_u_l"] > df["ldh_u_l"].median(),
        "all_b1": pd.Series(True, index=df.index),
    }
    for k, mm in modifier_specs.items():
        r = stratified_effect(treatment, biomarker, mm, k)
        if r is not None:
            store(f"{prefix}_{k}", **r)


explore_modifiers("treatment_olaparib", "brca2_mutation", "subgroup_olaparib_brca2")
explore_modifiers("treatment_pembrolizumab", "msi_high", "subgroup_pembro_msi")
explore_modifiers("treatment_lu177_psma", "psma_high", "subgroup_lu177_psma")

# Also: non-matched cases where biomarker SIGNAL is absent
for t, b in [("treatment_olaparib", "brca2_mutation"), ("treatment_pembrolizumab", "msi_high"), ("treatment_lu177_psma", "psma_high")]:
    sub = df[df[b] == 0]
    a = sub.loc[sub[t] == 1, OUTCOME]
    c = sub.loc[sub[t] == 0, OUTCOME]
    counts = np.array([a.sum(), c.sum()])
    nobs = np.array([len(a), len(c)])
    _, p = proportions_ztest(counts, nobs)
    store(
        f"nobm_{t}_{b}",
        rate_t=float(a.mean()),
        rate_c=float(c.mean()),
        diff=float(a.mean() - c.mean()),
        p=float(p),
        n_t=int(nobs[0]),
        n_c=int(nobs[1]),
    )

# ===================================================================
# ITERATION 10 — AR-V7 and AR-pathway treatments (enza, abi):
# resistance hypothesis
# ===================================================================
for t in ["treatment_enzalutamide", "treatment_abiraterone"]:
    explore_modifiers(t, "ar_v7_positive", f"subgroup_{t.split('_')[1]}_arv7_pos")
    explore_modifiers(t, "ar_v7_positive", f"subgroup_{t.split('_')[1]}_arv7_pos2")
    # AR-V7 negative
    sub = df[df["ar_v7_positive"] == 0]
    a = sub.loc[sub[t] == 1, OUTCOME]
    c = sub.loc[sub[t] == 0, OUTCOME]
    counts = np.array([a.sum(), c.sum()])
    nobs = np.array([len(a), len(c)])
    _, p = proportions_ztest(counts, nobs)
    store(
        f"arv7neg_{t}",
        rate_t=float(a.mean()),
        rate_c=float(c.mean()),
        diff=float(a.mean() - c.mean()),
        p=float(p),
        n_t=int(nobs[0]),
        n_c=int(nobs[1]),
    )

# ===================================================================
# ITERATION 11 — Tree-based heterogeneity for olaparib ARM (BRCA2+)
# Use logistic with all feature × T interactions on the BRCA2+ subgroup
# to find multi-feature modifiers
# ===================================================================

def all_feature_interactions(treatment, biomarker_filter, base_pred=None):
    sub = df[df[biomarker_filter] == 1].copy()
    if sub.shape[0] < 200:
        return None
    res = {}
    for col in [
        "ecog_ps",
        "visceral_mets",
        "mcrpc",
        "albumin_g_dl",
        "ldh_u_l",
        "psa_ng_ml",
        "gleason_score",
        "alkaline_phosphatase_u_l",
        "hemoglobin_g_dl",
        "weight_loss_pct_6mo",
        "crp_mg_l",
        "nlr",
        "age_years",
    ]:
        x = sub[col].astype(float).copy()
        if col in ("psa_ng_ml", "ldh_u_l", "crp_mg_l", "nlr", "alkaline_phosphatase_u_l"):
            x = np.log1p(x)
        X = pd.DataFrame({
            "T": sub[treatment].astype(float).values,
            "X": x.values,
            "TX": sub[treatment].astype(float).values * x.values,
        })
        try:
            m = logistic(sub[OUTCOME], X)
            res[col] = {
                "coef_T": float(m.params["T"]),
                "p_T": float(m.pvalues["T"]),
                "coef_X": float(m.params["X"]),
                "p_X": float(m.pvalues["X"]),
                "coef_TX": float(m.params["TX"]),
                "p_TX": float(m.pvalues["TX"]),
            }
        except Exception as exc:
            res[col] = {"error": str(exc)}
    return res


results["tree_olaparib_brca2"] = all_feature_interactions("treatment_olaparib", "brca2_mutation")
results["tree_pembro_msi"] = all_feature_interactions("treatment_pembrolizumab", "msi_high")
results["tree_lu177_psma_pos"] = all_feature_interactions("treatment_lu177_psma", "psma_high")

# ===================================================================
# ITERATION 12 — joint/three-way subgroups for pembro × msi
# ===================================================================
def joint_subgroup(treatment, modifiers: dict):
    mask = pd.Series(True, index=df.index)
    label_parts = []
    for col, expr in modifiers.items():
        mask = mask & expr
        label_parts.append(col)
    sub = df[mask]
    if (sub[treatment] == 1).sum() < 5 or (sub[treatment] == 0).sum() < 5:
        return None
    a = sub.loc[sub[treatment] == 1, OUTCOME]
    b = sub.loc[sub[treatment] == 0, OUTCOME]
    counts = np.array([a.sum(), b.sum()])
    nobs = np.array([len(a), len(b)])
    try:
        _, p = proportions_ztest(counts, nobs)
    except Exception:
        p = None
    return {
        "rate_t": float(a.mean()),
        "rate_c": float(b.mean()),
        "diff": float(a.mean() - b.mean()),
        "p": float(p) if p is not None else None,
        "n_t": int(nobs[0]),
        "n_c": int(nobs[1]),
        "label": "+".join(label_parts),
    }


# Pembro × MSI × ECOG
results["joint_pembro_msi_ecog0_1"] = joint_subgroup(
    "treatment_pembrolizumab",
    {"msi_high=1": df["msi_high"] == 1, "ecog<=1": df["ecog_ps"] <= 1},
)
results["joint_pembro_msi_ecog2"] = joint_subgroup(
    "treatment_pembrolizumab",
    {"msi_high=1": df["msi_high"] == 1, "ecog=2": df["ecog_ps"] == 2},
)
# Olaparib × BRCA2 × visceral
results["joint_olaparib_brca2_visceral_no"] = joint_subgroup(
    "treatment_olaparib",
    {"brca2=1": df["brca2_mutation"] == 1, "visceral=0": df["visceral_mets"] == 0},
)
results["joint_olaparib_brca2_visceral_yes"] = joint_subgroup(
    "treatment_olaparib",
    {"brca2=1": df["brca2_mutation"] == 1, "visceral=1": df["visceral_mets"] == 1},
)
# Lu177 × PSMA × visceral
results["joint_lu177_psmahi_visceral_no"] = joint_subgroup(
    "treatment_lu177_psma",
    {"psma=1": df["psma_high"] == 1, "visceral=0": df["visceral_mets"] == 0},
)
results["joint_lu177_psmahi_visceral_yes"] = joint_subgroup(
    "treatment_lu177_psma",
    {"psma=1": df["psma_high"] == 1, "visceral=1": df["visceral_mets"] == 1},
)
# Lu177 × PSMA × LDH (worse if LDH high)
results["joint_lu177_psmahi_ldh_low"] = joint_subgroup(
    "treatment_lu177_psma",
    {"psma=1": df["psma_high"] == 1, "ldh_low": df["ldh_u_l"] <= df["ldh_u_l"].median()},
)
results["joint_lu177_psmahi_ldh_high"] = joint_subgroup(
    "treatment_lu177_psma",
    {"psma=1": df["psma_high"] == 1, "ldh_high": df["ldh_u_l"] > df["ldh_u_l"].median()},
)

# Olaparib BRCA2 ECOG
results["joint_olaparib_brca2_ecog0"] = joint_subgroup(
    "treatment_olaparib",
    {"brca2=1": df["brca2_mutation"] == 1, "ecog=0": df["ecog_ps"] == 0},
)
results["joint_olaparib_brca2_ecog2"] = joint_subgroup(
    "treatment_olaparib",
    {"brca2=1": df["brca2_mutation"] == 1, "ecog=2": df["ecog_ps"] == 2},
)

# ===================================================================
# ITERATION 13 — pembro effect outside MSI
# Should be ~null
# ===================================================================
for sub_name, sub_mask in [
    ("msi_low_full", df["msi_high"] == 0),
    ("msi_low_ecog0_1", (df["msi_high"] == 0) & (df["ecog_ps"] <= 1)),
]:
    sub = df[sub_mask]
    a = sub.loc[sub["treatment_pembrolizumab"] == 1, OUTCOME]
    b = sub.loc[sub["treatment_pembrolizumab"] == 0, OUTCOME]
    counts = np.array([a.sum(), b.sum()])
    nobs = np.array([len(a), len(b)])
    _, p = proportions_ztest(counts, nobs)
    store(
        f"pembro_in_{sub_name}",
        rate_t=float(a.mean()),
        rate_c=float(b.mean()),
        diff=float(a.mean() - b.mean()),
        p=float(p),
        n_t=int(nobs[0]),
        n_c=int(nobs[1]),
    )

# ===================================================================
# ITERATION 14 — PSMA-low + Lu177? Should be lower
# ===================================================================
sub = df[df["psma_high"] == 0]
a = sub.loc[sub["treatment_lu177_psma"] == 1, OUTCOME]
b = sub.loc[sub["treatment_lu177_psma"] == 0, OUTCOME]
counts = np.array([a.sum(), b.sum()])
nobs = np.array([len(a), len(b)])
_, p = proportions_ztest(counts, nobs)
store(
    "lu177_psmalow",
    rate_t=float(a.mean()),
    rate_c=float(b.mean()),
    diff=float(a.mean() - b.mean()),
    p=float(p),
    n_t=int(nobs[0]),
    n_c=int(nobs[1]),
)

# ===================================================================
# ITERATION 15 — interactions among biomarkers (e.g., BRCA2 × PSMA)
# ===================================================================
def biomarker_interaction(b1, b2):
    X = pd.DataFrame({
        "B1": df[b1].astype(float),
        "B2": df[b2].astype(float),
        "B12": df[b1].astype(float) * df[b2].astype(float),
    })
    m = logistic(df[OUTCOME], X)
    return {
        "coef_B1": float(m.params["B1"]),
        "p_B1": float(m.pvalues["B1"]),
        "coef_B2": float(m.params["B2"]),
        "p_B2": float(m.pvalues["B2"]),
        "coef_B12": float(m.params["B12"]),
        "p_B12": float(m.pvalues["B12"]),
    }


for b1, b2 in [
    ("brca2_mutation", "msi_high"),
    ("brca2_mutation", "psma_high"),
    ("ar_v7_positive", "psma_high"),
    ("msi_high", "psma_high"),
]:
    store(f"bxbi_{b1}_{b2}", **biomarker_interaction(b1, b2))

# ===================================================================
# ITERATION 16 — final multivariable model with key interactions
# (olaparib*BRCA2, pembro*MSI, lu177*PSMA, enza*ARV7, abi*ARV7)
# ===================================================================
base_cols = [
    "age_years",
    "ecog_ps",
    "albumin_g_dl",
    "psa_ng_ml",
    "ldh_u_l",
    "visceral_mets",
    "mcrpc",
    "gleason_score",
    "weight_loss_pct_6mo",
    "hemoglobin_g_dl",
    "nlr",
    "alkaline_phosphatase_u_l",
    "calcium_mg_dl",
]
X = df[base_cols].astype(float).copy()
for c in ("psa_ng_ml", "ldh_u_l", "nlr", "alkaline_phosphatase_u_l"):
    X[c] = np.log1p(X[c])
for t in TREATMENTS:
    X[t] = df[t].astype(float)
for b in BIOMARKERS:
    X[b] = df[b].astype(float)
# Key interactions
X["olap_x_brca2"] = df["treatment_olaparib"].astype(float) * df["brca2_mutation"].astype(float)
X["pembro_x_msi"] = df["treatment_pembrolizumab"].astype(float) * df["msi_high"].astype(float)
X["lu177_x_psma"] = df["treatment_lu177_psma"].astype(float) * df["psma_high"].astype(float)
X["enza_x_arv7"] = df["treatment_enzalutamide"].astype(float) * df["ar_v7_positive"].astype(float)
X["abi_x_arv7"] = df["treatment_abiraterone"].astype(float) * df["ar_v7_positive"].astype(float)

m_int = logistic(df[OUTCOME], X)
results["multivar_with_interactions"] = {
    "coefs": {k: float(v) for k, v in m_int.params.items()},
    "pvals": {k: float(v) for k, v in m_int.pvalues.items()},
    "loglik": float(m_int.llf),
}

# ===================================================================
# ITERATION 17 — heterogeneity for AR pathway treatments
# ===================================================================
# ar_v7_pos vs neg, for enza and abi
for t in ["treatment_enzalutamide", "treatment_abiraterone"]:
    for label, mask in (("arv7_pos", df["ar_v7_positive"] == 1), ("arv7_neg", df["ar_v7_positive"] == 0)):
        sub = df[mask]
        a = sub.loc[sub[t] == 1, OUTCOME]
        b = sub.loc[sub[t] == 0, OUTCOME]
        counts = np.array([a.sum(), b.sum()])
        nobs = np.array([len(a), len(b)])
        try:
            _, p = proportions_ztest(counts, nobs)
        except Exception:
            p = None
        store(
            f"strat_{t}_{label}",
            rate_t=float(a.mean()),
            rate_c=float(b.mean()),
            diff=float(a.mean() - b.mean()),
            p=float(p) if p is not None else None,
            n_t=int(nobs[0]),
            n_c=int(nobs[1]),
        )

# ===================================================================
# ITERATION 18 — Docetaxel (chemo) heterogeneity
# ===================================================================
for label, mask in (
    ("ecog0_1", df["ecog_ps"] <= 1),
    ("ecog2", df["ecog_ps"] == 2),
    ("visceral_yes", df["visceral_mets"] == 1),
    ("visceral_no", df["visceral_mets"] == 0),
    ("ldh_high", df["ldh_u_l"] > df["ldh_u_l"].median()),
    ("ldh_low", df["ldh_u_l"] <= df["ldh_u_l"].median()),
    ("alb_high", df["albumin_g_dl"] >= df["albumin_g_dl"].median()),
    ("alb_low", df["albumin_g_dl"] < df["albumin_g_dl"].median()),
):
    sub = df[mask]
    a = sub.loc[sub["treatment_docetaxel"] == 1, OUTCOME]
    b = sub.loc[sub["treatment_docetaxel"] == 0, OUTCOME]
    counts = np.array([a.sum(), b.sum()])
    nobs = np.array([len(a), len(b)])
    _, p = proportions_ztest(counts, nobs)
    store(
        f"strat_docetaxel_{label}",
        rate_t=float(a.mean()),
        rate_c=float(b.mean()),
        diff=float(a.mean() - b.mean()),
        p=float(p),
        n_t=int(nobs[0]),
        n_c=int(nobs[1]),
    )

# ===================================================================
# ITERATION 19 — Olaparib effect: only for BRCA2+, but does ECOG modify?
# Joint subgroup with multiple modifiers
# ===================================================================
results["joint_olaparib_brca2_ecog0_1_visceral0"] = joint_subgroup(
    "treatment_olaparib",
    {
        "brca2=1": df["brca2_mutation"] == 1,
        "ecog<=1": df["ecog_ps"] <= 1,
        "visceral=0": df["visceral_mets"] == 0,
    },
)
results["joint_olaparib_brca2_ecog2_or_visceral1"] = joint_subgroup(
    "treatment_olaparib",
    {
        "brca2=1": df["brca2_mutation"] == 1,
        "ecog2_or_visceral": (df["ecog_ps"] == 2) | (df["visceral_mets"] == 1),
    },
)

# Pembro joint (msi+ecog<=1 already done above).
results["joint_pembro_msi_albhigh"] = joint_subgroup(
    "treatment_pembrolizumab",
    {"msi_high=1": df["msi_high"] == 1, "alb_high": df["albumin_g_dl"] >= df["albumin_g_dl"].median()},
)
results["joint_pembro_msi_alblow"] = joint_subgroup(
    "treatment_pembrolizumab",
    {"msi_high=1": df["msi_high"] == 1, "alb_low": df["albumin_g_dl"] < df["albumin_g_dl"].median()},
)

# Lu177 + PSMA-high + low LDH + good ECOG
results["joint_lu177_psma_ldhlow_ecog01"] = joint_subgroup(
    "treatment_lu177_psma",
    {
        "psma=1": df["psma_high"] == 1,
        "ldh_low": df["ldh_u_l"] <= df["ldh_u_l"].median(),
        "ecog<=1": df["ecog_ps"] <= 1,
    },
)
results["joint_lu177_psma_ldhhigh_or_ecog2"] = joint_subgroup(
    "treatment_lu177_psma",
    {
        "psma=1": df["psma_high"] == 1,
        "bad": (df["ldh_u_l"] > df["ldh_u_l"].median()) | (df["ecog_ps"] == 2),
    },
)

# ===================================================================
# ITERATION 20 — Compare baseline rate by ECOG within targeted subgroups
# (does ECOG=2 alone wipe out otherwise large treatment effects?)
# ===================================================================
for t, b in [("treatment_olaparib", "brca2_mutation"), ("treatment_pembrolizumab", "msi_high"), ("treatment_lu177_psma", "psma_high")]:
    sub = df[(df[b] == 1) & (df["ecog_ps"] == 2)]
    if (sub[t] == 1).sum() >= 3 and (sub[t] == 0).sum() >= 3:
        a = sub.loc[sub[t] == 1, OUTCOME]
        c = sub.loc[sub[t] == 0, OUTCOME]
        counts = np.array([a.sum(), c.sum()])
        nobs = np.array([len(a), len(c)])
        try:
            _, p = proportions_ztest(counts, nobs)
        except Exception:
            p = None
        store(
            f"ecog2_in_b1_{t}_{b}",
            rate_t=float(a.mean()),
            rate_c=float(c.mean()),
            diff=float(a.mean() - c.mean()),
            p=float(p) if p is not None else None,
            n_t=int(nobs[0]),
            n_c=int(nobs[1]),
        )

# ===================================================================
# Save results
# ===================================================================
def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    return o


with open("results_full.json", "w") as f:
    json.dump(jsonable(results), f, indent=2)

print("Saved", len(results), "result entries to results_full.json")
print()
# Highlights
for k in [
    "main_treatment_olaparib",
    "main_treatment_pembrolizumab",
    "main_treatment_lu177_psma",
    "main_treatment_enzalutamide",
    "main_treatment_abiraterone",
    "main_treatment_docetaxel",
    "main_brca2_mutation",
    "main_msi_high",
    "main_ar_v7_positive",
    "main_psma_high",
    "interact_treatment_olaparib_brca2_mutation",
    "interact_treatment_pembrolizumab_msi_high",
    "interact_treatment_lu177_psma_psma_high",
    "interact_treatment_enzalutamide_ar_v7_positive",
    "interact_treatment_abiraterone_ar_v7_positive",
]:
    print(k, "->", results.get(k))
