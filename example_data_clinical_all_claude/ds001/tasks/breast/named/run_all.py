"""
Iterative analysis of ds001_breast.

Outcome: pfs_months (continuous).
Approach: 25 iterations of propose-test-refine, ending in heterogeneity searches.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from itertools import combinations

df = pd.read_parquet("dataset.parquet")
N = len(df)
print(f"Loaded {N} rows, {df.shape[1]} columns")

OUTCOME = "pfs_months"
TREATMENTS = [
    "treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
    "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
]
BINARY_FEATS = [
    "sex_female", "stage_iv", "has_brain_mets", "node_positive", "postmenopausal",
    "er_positive", "pr_positive", "her2_positive", "her2_low",
    "brca1_mutation", "brca2_mutation", "pik3ca_mutation",
]
CONT_FEATS = [
    "age_years", "ecog_ps", "ki67_pct", "tumor_size_cm", "albumin_g_dl", "ldh_u_l",
    "weight_loss_pct_6mo", "crp_mg_l", "nlr", "hemoglobin_g_dl",
    "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl",
    "creatinine_mg_dl", "bun_mg_dl", "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl",
]

iterations = []  # accumulate iteration records


def add_iter(idx, hypotheses, analyses):
    iterations.append({
        "index": idx,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses,
    })


def mean_diff_test(group1, group0, label):
    """Welch's t-test; returns effect (g1 - g0), p, sig, summary."""
    eff = float(group1.mean() - group0.mean())
    t, p = stats.ttest_ind(group1, group0, equal_var=False)
    sig = bool(p < 0.05)
    summary = (f"Mean {label}: {group1.mean():.3f} (n={len(group1)}) vs "
               f"{group0.mean():.3f} (n={len(group0)}); diff={eff:+.3f}, "
               f"Welch t={t:.2f}, p={p:.3g}")
    return eff, float(p), sig, summary


def linreg_effect(x, y, label_x, label_y):
    """OLS y ~ x. Returns slope, p, sig, summary."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = float(model.params.iloc[1])
    p = float(model.pvalues.iloc[1])
    sig = bool(p < 0.05)
    summary = f"OLS {label_y} ~ {label_x}: slope={slope:+.4g}/unit, p={p:.3g}, n={len(y)}"
    return slope, p, sig, summary


def adjusted_treatment_effect(df_, tx, covars, outcome=OUTCOME, label=None):
    """Linear model: outcome ~ tx + covars. Returns coef on tx."""
    cols = [tx] + covars
    X = df_[cols].copy()
    X = sm.add_constant(X)
    y = df_[outcome]
    model = sm.OLS(y, X).fit()
    coef = float(model.params[tx])
    p = float(model.pvalues[tx])
    sig = bool(p < 0.05)
    summary = (f"Adjusted {outcome} ~ {tx} + {len(covars)} covars: "
               f"coef={coef:+.4f} months, p={p:.3g}, n={len(df_)}")
    return coef, p, sig, summary


def interaction_effect(df_, tx, mod, outcome=OUTCOME):
    """outcome ~ tx + mod + tx:mod. Returns interaction coef, p, sig."""
    sub = df_[[outcome, tx, mod]].copy()
    sub["tx_mod"] = sub[tx] * sub[mod]
    X = sm.add_constant(sub[[tx, mod, "tx_mod"]])
    model = sm.OLS(sub[outcome], X).fit()
    coef = float(model.params["tx_mod"])
    p = float(model.pvalues["tx_mod"])
    sig = bool(p < 0.05)
    return coef, p, sig, model


# =========================================================
# Iteration 1: Treatment main effects (univariate)
# =========================================================
hypotheses = []
analyses = []
for tx in TREATMENTS:
    h_id = f"h1_{tx}"
    hypotheses.append({
        "id": h_id,
        "text": f"Patients receiving {tx} have a different mean pfs_months than those not receiving {tx}.",
        "kind": "novel",
    })
    g1 = df.loc[df[tx] == 1, OUTCOME]
    g0 = df.loc[df[tx] == 0, OUTCOME]
    eff, p, sig, summ = mean_diff_test(g1, g0, OUTCOME)
    analyses.append({
        "hypothesis_ids": [h_id],
        "code": f"df.loc[df['{tx}']==1,'{OUTCOME}'] vs df.loc[df['{tx}']==0,'{OUTCOME}']; Welch t-test",
        "result_summary": summ,
        "p_value": p,
        "effect_estimate": eff,
        "significant": sig,
    })
add_iter(1, hypotheses, analyses)

# =========================================================
# Iteration 2: Demographic & disease severity main effects
# =========================================================
hypotheses = []
analyses = []
for feat in ["age_years", "ecog_ps", "stage_iv", "has_brain_mets", "node_positive"]:
    h_id = f"h2_{feat}"
    hypotheses.append({
        "id": h_id,
        "text": f"Higher {feat} is associated with shorter pfs_months.",
        "kind": "novel",
    })
    if feat in BINARY_FEATS or feat == "stage_iv" or feat == "has_brain_mets" or feat == "node_positive":
        g1 = df.loc[df[feat] == 1, OUTCOME]
        g0 = df.loc[df[feat] == 0, OUTCOME]
        eff, p, sig, summ = mean_diff_test(g1, g0, OUTCOME)
        code = f"Welch t-test of {OUTCOME} by {feat}"
    else:
        eff, p, sig, summ = linreg_effect(df[feat], df[OUTCOME], feat, OUTCOME)
        code = f"OLS {OUTCOME} ~ {feat}"
    analyses.append({
        "hypothesis_ids": [h_id],
        "code": code,
        "result_summary": summ,
        "p_value": p,
        "effect_estimate": eff,
        "significant": sig,
    })
add_iter(2, hypotheses, analyses)

# =========================================================
# Iteration 3: Tumor biology / hormone receptor / mutation main effects
# =========================================================
hypotheses = []
analyses = []
bio_feats = ["er_positive", "pr_positive", "her2_positive", "her2_low",
             "brca1_mutation", "brca2_mutation", "pik3ca_mutation",
             "postmenopausal"]
for feat in bio_feats:
    h_id = f"h3_{feat}"
    hypotheses.append({
        "id": h_id,
        "text": f"{feat}=1 is associated with different mean pfs_months than {feat}=0.",
        "kind": "novel",
    })
    g1 = df.loc[df[feat] == 1, OUTCOME]
    g0 = df.loc[df[feat] == 0, OUTCOME]
    eff, p, sig, summ = mean_diff_test(g1, g0, OUTCOME)
    analyses.append({
        "hypothesis_ids": [h_id],
        "code": f"Welch t-test of {OUTCOME} by {feat}",
        "result_summary": summ,
        "p_value": p,
        "effect_estimate": eff,
        "significant": sig,
    })
add_iter(3, hypotheses, analyses)

# =========================================================
# Iteration 4: Continuous tumor / inflammation markers
# =========================================================
hypotheses = []
analyses = []
for feat in ["ki67_pct", "tumor_size_cm", "albumin_g_dl", "ldh_u_l",
             "weight_loss_pct_6mo", "crp_mg_l", "nlr"]:
    h_id = f"h4_{feat}"
    direction = ("lower" if feat in ("albumin_g_dl",) else "higher")
    hypotheses.append({
        "id": h_id,
        "text": f"Higher {feat} is associated with shorter pfs_months "
                f"(adverse prognostic marker; expected slope <0).",
        "kind": "novel",
    })
    eff, p, sig, summ = linreg_effect(df[feat], df[OUTCOME], feat, OUTCOME)
    analyses.append({
        "hypothesis_ids": [h_id],
        "code": f"OLS {OUTCOME} ~ {feat}",
        "result_summary": summ,
        "p_value": p,
        "effect_estimate": eff,
        "significant": sig,
    })
add_iter(4, hypotheses, analyses)

# =========================================================
# Iteration 5: Lab chemistry main effects
# =========================================================
hypotheses = []
analyses = []
for feat in ["hemoglobin_g_dl", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
             "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
             "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl"]:
    h_id = f"h5_{feat}"
    hypotheses.append({
        "id": h_id,
        "text": f"{feat} is associated with pfs_months in this cohort.",
        "kind": "novel",
    })
    eff, p, sig, summ = linreg_effect(df[feat], df[OUTCOME], feat, OUTCOME)
    analyses.append({
        "hypothesis_ids": [h_id],
        "code": f"OLS {OUTCOME} ~ {feat}",
        "result_summary": summ,
        "p_value": p,
        "effect_estimate": eff,
        "significant": sig,
    })
add_iter(5, hypotheses, analyses)

print("Iterations 1-5 done")
# Save intermediate
import pickle
with open("_iters_1_5.pkl", "wb") as f:
    pickle.dump(iterations, f)
print(f"Iterations so far: {len(iterations)}")
