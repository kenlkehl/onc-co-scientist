"""Comprehensive analysis of ds001_nsclc.

Executes the propose-test-update protocol across iterations and writes
intermediate results to my_results.json. The final transcript is built
separately.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DATA_PATH = Path("dataset.parquet")
OUT_PATH = Path("my_results.json")

TREATMENTS = [
    "treatment_pembrolizumab",
    "treatment_sotorasib",
    "treatment_olaparib",
    "treatment_osimertinib",
]
BIOMARKERS_BIN = [
    "egfr_mutation",
    "kras_g12c",
    "alk_fusion",
    "stk11_mutation",
    "brca2_mutation",
    "tmb_high",
]
CLIN_BIN = ["sex_female", "stage_iv", "has_brain_mets"]
LABS_CONT = [
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
    "pdl1_tps",
]
DEMOG = ["age_years"]
ALL_BIN = CLIN_BIN + BIOMARKERS_BIN + ["tmb_high"]
COVARS_FOR_ADJ = (
    DEMOG
    + CLIN_BIN
    + ["ecog_ps"]
    + BIOMARKERS_BIN
    + LABS_CONT
)


def fit_ols(df, formula):
    return smf.ols(formula, data=df).fit()


def main():
    df = pd.read_parquet(DATA_PATH)
    # Encode categoricals
    df["smoke_current"] = (df["smoking_status"] == "current").astype(int)
    df["smoke_former"] = (df["smoking_status"] == "former").astype(int)
    df["smoke_never"] = (df["smoking_status"] == "never").astype(int)
    df["hist_adeno"] = (df["histology"] == "adenocarcinoma").astype(int)
    # ecog as ordinal int already

    results = {}

    # =====================================================================
    # Iter 1: main effect of clinical/demographic features on PFS (unadj)
    # =====================================================================
    iter1 = []
    # age
    r = stats.pearsonr(df["age_years"], df["pfs_months"])
    iter1.append({"feat": "age_years", "test": "pearson_r", "r": r.statistic, "p": r.pvalue})
    # sex_female (t-test)
    g0 = df.loc[df["sex_female"] == 0, "pfs_months"]
    g1 = df.loc[df["sex_female"] == 1, "pfs_months"]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    iter1.append({
        "feat": "sex_female",
        "test": "welch_t",
        "mean_1": g1.mean(),
        "mean_0": g0.mean(),
        "diff": g1.mean() - g0.mean(),
        "p": t.pvalue,
    })
    # ecog_ps via OLS slope
    m = fit_ols(df, "pfs_months ~ ecog_ps")
    iter1.append({
        "feat": "ecog_ps",
        "test": "ols_slope",
        "beta": m.params["ecog_ps"],
        "p": m.pvalues["ecog_ps"],
    })
    # stage_iv
    g0 = df.loc[df["stage_iv"] == 0, "pfs_months"]
    g1 = df.loc[df["stage_iv"] == 1, "pfs_months"]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    iter1.append({
        "feat": "stage_iv",
        "test": "welch_t",
        "diff": g1.mean() - g0.mean(),
        "mean_1": g1.mean(),
        "mean_0": g0.mean(),
        "p": t.pvalue,
    })
    # has_brain_mets
    g0 = df.loc[df["has_brain_mets"] == 0, "pfs_months"]
    g1 = df.loc[df["has_brain_mets"] == 1, "pfs_months"]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    iter1.append({
        "feat": "has_brain_mets",
        "test": "welch_t",
        "diff": g1.mean() - g0.mean(),
        "p": t.pvalue,
    })
    # smoking_status (one-way ANOVA)
    groups = [df.loc[df["smoking_status"] == s, "pfs_months"] for s in ["never", "former", "current"]]
    f = stats.f_oneway(*groups)
    iter1.append({
        "feat": "smoking_status",
        "test": "anova",
        "F": f.statistic,
        "p": f.pvalue,
        "means": {s: g.mean() for s, g in zip(["never", "former", "current"], groups)},
    })
    # histology
    g0 = df.loc[df["histology"] == "squamous", "pfs_months"]
    g1 = df.loc[df["histology"] == "adenocarcinoma", "pfs_months"]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    iter1.append({
        "feat": "histology(adeno-sq)",
        "test": "welch_t",
        "diff": g1.mean() - g0.mean(),
        "mean_adeno": g1.mean(),
        "mean_sq": g0.mean(),
        "p": t.pvalue,
    })
    results["iter1_clinical_main"] = iter1

    # =====================================================================
    # Iter 2: main effect of biomarkers on PFS (unadj)
    # =====================================================================
    iter2 = []
    for b in BIOMARKERS_BIN:
        g0 = df.loc[df[b] == 0, "pfs_months"]
        g1 = df.loc[df[b] == 1, "pfs_months"]
        t = stats.ttest_ind(g1, g0, equal_var=False)
        iter2.append({
            "feat": b,
            "test": "welch_t",
            "mean_1": g1.mean(),
            "mean_0": g0.mean(),
            "diff": g1.mean() - g0.mean(),
            "p": t.pvalue,
        })
    # pdl1_tps continuous
    r = stats.pearsonr(df["pdl1_tps"], df["pfs_months"])
    iter2.append({"feat": "pdl1_tps", "test": "pearson_r", "r": r.statistic, "p": r.pvalue})
    results["iter2_biomarker_main"] = iter2

    # =====================================================================
    # Iter 3: main effect of labs on PFS (unadj)
    # =====================================================================
    iter3 = []
    for lab in LABS_CONT:
        r = stats.pearsonr(df[lab], df["pfs_months"])
        iter3.append({"feat": lab, "test": "pearson_r", "r": r.statistic, "p": r.pvalue})
    results["iter3_labs_main"] = iter3

    # =====================================================================
    # Iter 4: unadjusted treatment main effects
    # =====================================================================
    iter4 = []
    for tx in TREATMENTS:
        g0 = df.loc[df[tx] == 0, "pfs_months"]
        g1 = df.loc[df[tx] == 1, "pfs_months"]
        t = stats.ttest_ind(g1, g0, equal_var=False)
        iter4.append({
            "feat": tx,
            "test": "welch_t",
            "mean_1": g1.mean(),
            "mean_0": g0.mean(),
            "diff": g1.mean() - g0.mean(),
            "p": t.pvalue,
        })
    results["iter4_tx_unadj"] = iter4

    # =====================================================================
    # Iter 5: treatment effects adjusted for full covariate set
    # =====================================================================
    formula_base = (
        "pfs_months ~ "
        + " + ".join(
            DEMOG
            + CLIN_BIN
            + ["ecog_ps", "smoke_current", "smoke_former", "hist_adeno"]
            + BIOMARKERS_BIN
            + LABS_CONT
            + TREATMENTS
        )
    )
    m = fit_ols(df, formula_base)
    adj = []
    for tx in TREATMENTS:
        adj.append({
            "feat": tx,
            "beta": m.params[tx],
            "p": m.pvalues[tx],
            "se": m.bse[tx],
        })
    # also pull other notable adjusted coefficients
    other = {}
    for v in (
        DEMOG
        + CLIN_BIN
        + ["ecog_ps", "smoke_current", "smoke_former", "hist_adeno"]
        + BIOMARKERS_BIN
        + LABS_CONT
    ):
        other[v] = {"beta": m.params[v], "p": m.pvalues[v]}
    results["iter5_tx_adj"] = {"treatments": adj, "covariates": other, "rsq": m.rsquared}

    # =====================================================================
    # Iter 6-9: tx-by-feature interaction screens
    # =====================================================================
    interaction_features = (
        BIOMARKERS_BIN
        + CLIN_BIN
        + ["ecog_ps", "age_years", "pdl1_tps"]
        + LABS_CONT
        + ["smoke_current", "smoke_former", "hist_adeno"]
    )

    def screen_interactions(tx):
        rows = []
        for f in interaction_features:
            # OLS with main effects of tx, f, and all other treatments + key confounders
            other_tx = [t for t in TREATMENTS if t != tx]
            base_covars = [
                "age_years",
                "ecog_ps",
                "stage_iv",
                "has_brain_mets",
                "albumin_g_dl",
                "ldh_u_l",
                "nlr",
            ]
            covars = list({c for c in base_covars + other_tx if c != f})
            covar_str = " + ".join(covars) if covars else ""
            formula = f"pfs_months ~ {tx} * {f}" + (f" + {covar_str}" if covar_str else "")
            try:
                m = fit_ols(df, formula)
                int_term = f"{tx}:{f}"
                if int_term in m.params.index:
                    rows.append({
                        "feat": f,
                        "interaction_beta": m.params[int_term],
                        "interaction_p": m.pvalues[int_term],
                        "tx_main_beta": m.params[tx],
                        "tx_main_p": m.pvalues[tx],
                    })
            except Exception as exc:
                rows.append({"feat": f, "error": str(exc)})
        rows.sort(key=lambda r: r.get("interaction_p", 1.0))
        return rows

    for i, tx in enumerate(TREATMENTS, start=6):
        results[f"iter{i}_interactions_{tx}"] = screen_interactions(tx)

    # =====================================================================
    # Iter 10: stratified treatment effects by binary feature
    # =====================================================================
    strat = {}
    for tx in TREATMENTS:
        strat[tx] = []
        for f in BIOMARKERS_BIN + CLIN_BIN + ["smoke_current", "smoke_former", "hist_adeno"]:
            for v in [0, 1]:
                sub = df[df[f] == v]
                if sub[tx].sum() < 50 or (sub[tx] == 0).sum() < 50:
                    continue
                g0 = sub.loc[sub[tx] == 0, "pfs_months"]
                g1 = sub.loc[sub[tx] == 1, "pfs_months"]
                t = stats.ttest_ind(g1, g0, equal_var=False)
                strat[tx].append({
                    "feat": f,
                    "level": v,
                    "n": int(len(sub)),
                    "n_tx": int(sub[tx].sum()),
                    "diff": float(g1.mean() - g0.mean()),
                    "mean_1": float(g1.mean()),
                    "mean_0": float(g0.mean()),
                    "p": float(t.pvalue),
                })
    results["iter10_stratified_tx"] = strat

    # =====================================================================
    # Iter 11: 3-way interactions: tx x feat1 x feat2 (top from screens)
    # =====================================================================
    # Discover candidate combos: top 3 features per treatment by interaction p
    top_combo = {}
    for tx in TREATMENTS:
        rows = sorted(
            [r for r in results[f"iter6_interactions_{tx}"] if "interaction_p" in r]
            if tx == "treatment_pembrolizumab"
            else [r for r in results[f"iter{6 + TREATMENTS.index(tx)}_interactions_{tx}"] if "interaction_p" in r],
            key=lambda r: r["interaction_p"],
        )
        top = [r["feat"] for r in rows[:5]]
        top_combo[tx] = []
        for a, b in combinations(top, 2):
            other_tx = [t for t in TREATMENTS if t != tx]
            covars = ["age_years", "ecog_ps", "stage_iv", "has_brain_mets", "albumin_g_dl", "ldh_u_l", "nlr"] + other_tx
            covars = [c for c in covars if c not in {a, b}]
            covar_str = " + ".join(covars)
            # Three-way interaction
            formula = f"pfs_months ~ {tx} * {a} * {b} + {covar_str}"
            try:
                m = fit_ols(df, formula)
                three_way_term = next((p for p in m.params.index if p.count(":") == 2 and tx in p and a in p and b in p), None)
                two_way_a = f"{tx}:{a}"
                two_way_b = f"{tx}:{b}"
                row = {"feat_a": a, "feat_b": b}
                if three_way_term:
                    row["three_way_beta"] = m.params[three_way_term]
                    row["three_way_p"] = m.pvalues[three_way_term]
                if two_way_a in m.params.index:
                    row["tx_x_a_beta"] = m.params[two_way_a]
                    row["tx_x_a_p"] = m.pvalues[two_way_a]
                if two_way_b in m.params.index:
                    row["tx_x_b_beta"] = m.params[two_way_b]
                    row["tx_x_b_p"] = m.pvalues[two_way_b]
                top_combo[tx].append(row)
            except Exception as exc:
                top_combo[tx].append({"feat_a": a, "feat_b": b, "error": str(exc)})
    results["iter11_three_way"] = top_combo

    Path("my_results.json").write_text(json.dumps(results, indent=2, default=float))
    print("Wrote my_results.json")


if __name__ == "__main__":
    main()
