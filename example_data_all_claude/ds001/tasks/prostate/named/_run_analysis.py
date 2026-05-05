"""Iterative analysis script for ds001_prostate.

Loads the parquet, runs the full battery of analyses, and stores results in
a structured dict that the transcript-build step turns into transcript.json.
"""
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DF = pd.read_parquet(HERE / "dataset.parquet")

TREATMENTS = [
    "treatment_enzalutamide",
    "treatment_abiraterone",
    "treatment_docetaxel",
    "treatment_olaparib",
    "treatment_lu177_psma",
    "treatment_pembrolizumab",
]

BIOMARKERS = ["brca2_mutation", "ar_v7_positive", "msi_high", "psma_high"]
CLIN_BIN = ["mcrpc", "visceral_mets"]
CONT = [
    "age_years", "psa_ng_ml", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo",
    "crp_mg_l", "nlr", "hemoglobin_g_dl", "alkaline_phosphatase_u_l",
    "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl", "creatinine_mg_dl",
    "bun_mg_dl", "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl",
]


def two_prop(df, group_col, group_val_a=1, group_val_b=0, outcome="objective_response"):
    """Compare outcome rates between two groups; return (rate_a, rate_b, diff, p)."""
    a = df.loc[df[group_col] == group_val_a, outcome]
    b = df.loc[df[group_col] == group_val_b, outcome]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan, np.nan, np.nan, len(a), len(b))
    rate_a, rate_b = a.mean(), b.mean()
    # Z test for two proportions
    p1, p2 = rate_a, rate_b
    n1, n2 = len(a), len(b)
    p_pool = (a.sum() + b.sum()) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (rate_a, rate_b, rate_a - rate_b, 1.0, n1, n2)
    z = (p1 - p2) / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return (rate_a, rate_b, rate_a - rate_b, pval, n1, n2)


def logreg_or(df, predictors, outcome="objective_response"):
    """Fit a logistic regression and return coef table (beta, OR, p)."""
    import statsmodels.api as sm
    X = df[predictors].astype(float).copy()
    X = sm.add_constant(X)
    y = df[outcome].astype(int)
    res = sm.Logit(y, X).fit(disp=0, maxiter=200)
    out = {}
    for name in predictors:
        b = res.params[name]
        p = res.pvalues[name]
        out[name] = {"beta": float(b), "or": float(math.exp(b)), "p": float(p)}
    return out, res


def logreg_with_interaction(df, treat, modifier, outcome="objective_response", covariates=None):
    """Fit logistic regression with treat*modifier interaction. Return interaction beta and p."""
    import statsmodels.api as sm
    cols = [treat, modifier]
    if covariates:
        cols = cols + [c for c in covariates if c not in cols]
    X = df[cols].astype(float).copy()
    X["__inter"] = X[treat] * X[modifier]
    X = sm.add_constant(X)
    y = df[outcome].astype(int)
    res = sm.Logit(y, X).fit(disp=0, maxiter=200)
    return {
        "beta_treat": float(res.params[treat]),
        "p_treat": float(res.pvalues[treat]),
        "beta_modifier": float(res.params[modifier]),
        "p_modifier": float(res.pvalues[modifier]),
        "beta_inter": float(res.params["__inter"]),
        "p_inter": float(res.pvalues["__inter"]),
    }


def stratified_treatment_effect(df, treat, modifier, outcome="objective_response"):
    """Within each stratum of modifier (0/1), compute the treat effect on outcome."""
    out = {}
    for v in [0, 1]:
        sub = df[df[modifier] == v]
        if len(sub) == 0:
            continue
        ra, rb, diff, p, na, nb = two_prop(sub, treat, 1, 0, outcome)
        out[v] = {"rate_treat": ra, "rate_ctrl": rb, "diff": diff, "p": p, "n_treat": na, "n_ctrl": nb}
    return out


# Container for results
RESULTS = {}


def run_iter1():
    """Treatment main effects on objective_response (univariable)."""
    out = []
    for t in TREATMENTS:
        ra, rb, diff, p, na, nb = two_prop(DF, t)
        out.append({
            "treatment": t,
            "rate_treat": ra, "rate_ctrl": rb, "diff": diff, "p": p,
            "n_treat": na, "n_ctrl": nb,
            "significant": p < 0.05,
        })
    return out


def run_iter2():
    """Biomarker main effects on objective_response (univariable)."""
    out = []
    for b in BIOMARKERS:
        ra, rb, diff, p, na, nb = two_prop(DF, b)
        out.append({
            "biomarker": b,
            "rate_pos": ra, "rate_neg": rb, "diff": diff, "p": p,
            "n_pos": na, "n_neg": nb, "significant": p < 0.05,
        })
    return out


def run_iter3_clinical():
    """ECOG, mCRPC, visceral_mets, age, gleason main effects."""
    out = {}
    # ECOG
    rates = DF.groupby("ecog_ps")["objective_response"].agg(["mean", "count"])
    out["ecog_table"] = rates.to_dict()
    # chi-square
    ct = pd.crosstab(DF["ecog_ps"], DF["objective_response"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    out["ecog_chi2_p"] = float(p)
    # mCRPC
    out["mcrpc"] = two_prop(DF, "mcrpc")
    # visceral_mets
    out["visceral_mets"] = two_prop(DF, "visceral_mets")
    # gleason as continuous predictor via t-test grouping or correlation
    # Compare top-vs-bottom gleason
    gh = DF[DF["gleason_score"] >= 9]["objective_response"].mean()
    gl = DF[DF["gleason_score"] <= 7]["objective_response"].mean()
    ct_g = pd.crosstab(DF["gleason_score"], DF["objective_response"])
    chi2g, pg, _, _ = stats.chi2_contingency(ct_g)
    out["gleason_high_rate"] = float(gh)
    out["gleason_low_rate"] = float(gl)
    out["gleason_chi2_p"] = float(pg)
    # Age - logistic with age
    age_lr, _ = logreg_or(DF, ["age_years"])
    out["age_logit"] = age_lr["age_years"]
    return out


def run_iter4_labs():
    """Univariable logistic regression of each continuous lab vs response."""
    out = {}
    for c in CONT:
        try:
            res, _ = logreg_or(DF, [c])
            out[c] = res[c]
        except Exception as e:
            out[c] = {"error": str(e)}
    return out


def run_iter5_treat_x_biomarker():
    """Each treatment x each biomarker: stratified rates + interaction p in logistic."""
    out = {}
    for t in TREATMENTS:
        for b in BIOMARKERS:
            strat = stratified_treatment_effect(DF, t, b)
            try:
                inter = logreg_with_interaction(DF, t, b)
            except Exception as e:
                inter = {"error": str(e)}
            out[f"{t}__x__{b}"] = {"strat": strat, "inter": inter}
    return out


def run_iter6_treat_x_clinical():
    """Treatment x clinical/categorical modifier interactions: ECOG (binary highest), mCRPC, visceral, gleason high, age>=70."""
    df = DF.copy()
    df["ecog_high"] = (df["ecog_ps"] >= 2).astype(int)
    df["gleason_high"] = (df["gleason_score"] >= 9).astype(int)
    df["age_ge70"] = (df["age_years"] >= 70).astype(int)
    modifiers = ["ecog_high", "mcrpc", "visceral_mets", "gleason_high", "age_ge70"]
    out = {}
    for t in TREATMENTS:
        for m in modifiers:
            strat = stratified_treatment_effect(df, t, m)
            try:
                inter = logreg_with_interaction(df, t, m)
            except Exception as e:
                inter = {"error": str(e)}
            out[f"{t}__x__{m}"] = {"strat": strat, "inter": inter}
    return out


def run_iter7_treat_x_lab():
    """Treatment x continuous-lab interactions in logistic."""
    out = {}
    for t in TREATMENTS:
        for c in CONT:
            try:
                inter = logreg_with_interaction(DF, t, c)
            except Exception as e:
                inter = {"error": str(e)}
            out[f"{t}__x__{c}"] = inter
    return out


def run_iter8_multivariable():
    """Single multivariable logistic with all features (no interactions) to confirm independent effects."""
    df = DF.copy()
    df["ecog_high"] = (df["ecog_ps"] >= 2).astype(int)
    predictors = (
        ["age_years", "ecog_ps", "mcrpc", "visceral_mets", "gleason_score"]
        + BIOMARKERS + TREATMENTS + CONT
    )
    # remove duplicates and drop age_years duplicates
    seen = []
    for p in predictors:
        if p not in seen:
            seen.append(p)
    res, mod = logreg_or(df, seen)
    return {"coef": res, "n": len(df), "llf": float(mod.llf)}


def run_iter9_subgroup_definitions():
    """For each treatment, find the strongest (subgroup-defining) biomarker/clinical predicate by combining
    largest stratified diff and lowest interaction p from prior runs.

    Then test 2-feature combined subgroups for the top treatment-modifier pair to find suppressors.
    """
    df = DF.copy()
    df["ecog_high"] = (df["ecog_ps"] >= 2).astype(int)
    df["ecog_low"] = (df["ecog_ps"] <= 1).astype(int)  # the favorable level
    df["gleason_high"] = (df["gleason_score"] >= 9).astype(int)
    df["age_ge70"] = (df["age_years"] >= 70).astype(int)
    # candidate modifiers
    mods_bin = BIOMARKERS + ["mcrpc", "visceral_mets", "gleason_high", "ecog_high", "age_ge70", "ecog_low"]
    # compute lab-derived high/low binaries from medians
    for c in CONT:
        df[f"{c}_hi"] = (df[c] > df[c].median()).astype(int)
        mods_bin.append(f"{c}_hi")
    out = {}
    for t in TREATMENTS:
        rows = []
        for m in mods_bin:
            strat = stratified_treatment_effect(df, t, m)
            d1 = strat.get(1, {}).get("diff")
            d0 = strat.get(0, {}).get("diff")
            p1 = strat.get(1, {}).get("p")
            p0 = strat.get(0, {}).get("p")
            try:
                inter = logreg_with_interaction(df, t, m)
                inter_p = inter["p_inter"]
                inter_b = inter["beta_inter"]
            except Exception:
                inter_p = None
                inter_b = None
            rows.append({
                "modifier": m,
                "diff_when_mod1": d1, "p_when_mod1": p1,
                "diff_when_mod0": d0, "p_when_mod0": p0,
                "inter_beta": inter_b, "inter_p": inter_p,
                "delta_diff": (d1 - d0) if (d1 is not None and d0 is not None) else None,
            })
        out[t] = rows
    return out


def run_iter10_two_feature_subgroups():
    """For each treatment whose effect appears subgroup-concentrated, test definitions of the form
    A=1 AND B=1 (or =0 for unfavorable suppression) and report the within-subgroup treatment diff.
    Restrict to top candidate modifiers identified in iter9 to keep this tractable.
    """
    df = DF.copy()
    df["ecog_high"] = (df["ecog_ps"] >= 2).astype(int)
    df["ecog_le1"] = (df["ecog_ps"] <= 1).astype(int)
    df["gleason_high"] = (df["gleason_score"] >= 9).astype(int)
    df["age_ge70"] = (df["age_years"] >= 70).astype(int)
    candidates = {
        "treatment_olaparib": ["brca2_mutation", "ar_v7_positive", "msi_high", "psma_high",
                               "mcrpc", "visceral_mets", "ecog_high", "ecog_le1", "gleason_high"],
        "treatment_pembrolizumab": ["msi_high", "brca2_mutation", "ar_v7_positive", "psma_high",
                                    "mcrpc", "visceral_mets", "ecog_high", "ecog_le1", "gleason_high"],
        "treatment_lu177_psma": ["psma_high", "brca2_mutation", "ar_v7_positive", "msi_high",
                                 "mcrpc", "visceral_mets", "ecog_high", "ecog_le1", "gleason_high"],
        "treatment_enzalutamide": ["ar_v7_positive", "brca2_mutation", "msi_high", "psma_high",
                                   "mcrpc", "visceral_mets", "ecog_high", "ecog_le1", "gleason_high"],
        "treatment_abiraterone": ["ar_v7_positive", "brca2_mutation", "msi_high", "psma_high",
                                  "mcrpc", "visceral_mets", "ecog_high", "ecog_le1", "gleason_high"],
        "treatment_docetaxel": ["visceral_mets", "mcrpc", "brca2_mutation", "ar_v7_positive",
                                "msi_high", "psma_high", "ecog_high", "ecog_le1", "gleason_high"],
    }
    out = {}
    for t, mods in candidates.items():
        rows = []
        for m1 in mods:
            for m2 in mods:
                if m1 == m2:
                    continue
                # Test A=1 and B=1 subgroup
                for v1, v2 in [(1, 1), (1, 0), (0, 1)]:
                    sub = df[(df[m1] == v1) & (df[m2] == v2)]
                    if len(sub) < 50 or sub[t].sum() < 10 or (1 - sub[t]).sum() < 10:
                        continue
                    ra, rb, diff, p, na, nb = two_prop(sub, t)
                    rows.append({
                        "subgroup": f"{m1}={v1} & {m2}={v2}",
                        "n": len(sub),
                        "n_treat": na, "n_ctrl": nb,
                        "rate_treat": ra, "rate_ctrl": rb,
                        "diff": diff, "p": p,
                    })
        # Sort by largest absolute diff with p<0.01
        rows_sorted = sorted([r for r in rows if r["p"] is not None],
                             key=lambda r: -abs(r["diff"] or 0))
        out[t] = rows_sorted[:25]
    return out


def run_iter11_lab_quartiles_treat():
    """For top continuous-lab interaction signals, look at within-quartile treatment effect."""
    out = {}
    df = DF.copy()
    pairs = [
        ("treatment_olaparib", "albumin_g_dl"),
        ("treatment_olaparib", "ldh_u_l"),
        ("treatment_olaparib", "alkaline_phosphatase_u_l"),
        ("treatment_pembrolizumab", "albumin_g_dl"),
        ("treatment_pembrolizumab", "ldh_u_l"),
        ("treatment_pembrolizumab", "nlr"),
        ("treatment_lu177_psma", "psa_ng_ml"),
        ("treatment_lu177_psma", "albumin_g_dl"),
        ("treatment_lu177_psma", "ldh_u_l"),
        ("treatment_lu177_psma", "alkaline_phosphatase_u_l"),
        ("treatment_enzalutamide", "albumin_g_dl"),
        ("treatment_enzalutamide", "ldh_u_l"),
        ("treatment_abiraterone", "albumin_g_dl"),
        ("treatment_abiraterone", "ldh_u_l"),
        ("treatment_docetaxel", "albumin_g_dl"),
        ("treatment_docetaxel", "ldh_u_l"),
    ]
    for t, lab in pairs:
        try:
            qs = pd.qcut(df[lab], 4, labels=False, duplicates="drop")
        except Exception:
            continue
        rows = []
        for q in sorted(qs.dropna().unique()):
            sub = df[qs == q]
            if len(sub) < 100:
                continue
            ra, rb, diff, p, na, nb = two_prop(sub, t)
            rows.append({"q": int(q), "n": len(sub), "rate_treat": ra,
                         "rate_ctrl": rb, "diff": diff, "p": p,
                         "n_treat": na, "n_ctrl": nb})
        out[f"{t}__{lab}"] = rows
    return out


def run_iter12_final_subgroup_for_each_treatment():
    """For each treatment, identify final best-supported subgroup definition combining
    all suggestive modifiers. We test 'positive marker AND favorable clinical baseline' definitions.
    """
    df = DF.copy()
    df["ecog_le1"] = (df["ecog_ps"] <= 1).astype(int)
    df["albumin_ok"] = (df["albumin_g_dl"] >= df["albumin_g_dl"].median()).astype(int)
    df["ldh_low"] = (df["ldh_u_l"] <= df["ldh_u_l"].median()).astype(int)
    df["alp_low"] = (df["alkaline_phosphatase_u_l"] <= df["alkaline_phosphatase_u_l"].median()).astype(int)
    df["no_visceral"] = (df["visceral_mets"] == 0).astype(int)

    out = {}
    # We'll test a set of candidate subgroup definitions per treatment
    defs = {
        "treatment_olaparib": [
            ["brca2_mutation"],
            ["brca2_mutation", "ecog_le1"],
            ["brca2_mutation", "albumin_ok"],
            ["brca2_mutation", "ecog_le1", "albumin_ok"],
            ["brca2_mutation", "no_visceral"],
        ],
        "treatment_pembrolizumab": [
            ["msi_high"],
            ["msi_high", "ecog_le1"],
            ["msi_high", "albumin_ok"],
            ["msi_high", "ecog_le1", "albumin_ok"],
            ["msi_high", "no_visceral"],
        ],
        "treatment_lu177_psma": [
            ["psma_high"],
            ["psma_high", "ecog_le1"],
            ["psma_high", "albumin_ok"],
            ["psma_high", "ecog_le1", "albumin_ok"],
            ["psma_high", "no_visceral"],
        ],
        "treatment_enzalutamide": [
            [],
            ["ecog_le1"],
            ["albumin_ok"],
            ["no_visceral"],
        ],
        "treatment_abiraterone": [
            [],
            ["ecog_le1"],
            ["albumin_ok"],
            ["no_visceral"],
        ],
        "treatment_docetaxel": [
            [],
            ["ecog_le1"],
            ["no_visceral"],
            ["albumin_ok"],
        ],
    }
    for t, dlist in defs.items():
        rows = []
        for predicates in dlist:
            mask = pd.Series(True, index=df.index)
            for p in predicates:
                mask &= df[p] == 1
            sub = df[mask]
            if len(sub) < 30 or sub[t].sum() < 5 or (1 - sub[t]).sum() < 5:
                rows.append({"predicates": predicates, "n": int(len(sub)), "skipped": True})
                continue
            ra, rb, diff, p, na, nb = two_prop(sub, t)
            rows.append({
                "predicates": predicates, "n": int(len(sub)),
                "n_treat": int(na), "n_ctrl": int(nb),
                "rate_treat": float(ra), "rate_ctrl": float(rb),
                "diff": float(diff), "p": float(p),
            })
        out[t] = rows
    return out


def run_iter13_negation_outside_subgroup():
    """For the strongest subgroup definition per treatment, also report the treatment effect OUTSIDE the subgroup."""
    df = DF.copy()
    df["ecog_le1"] = (df["ecog_ps"] <= 1).astype(int)
    df["albumin_ok"] = (df["albumin_g_dl"] >= df["albumin_g_dl"].median()).astype(int)

    candidates = [
        ("treatment_olaparib", ["brca2_mutation"]),
        ("treatment_pembrolizumab", ["msi_high"]),
        ("treatment_lu177_psma", ["psma_high"]),
    ]
    out = {}
    for t, predicates in candidates:
        mask = pd.Series(True, index=df.index)
        for p in predicates:
            mask &= df[p] == 1
        in_sub = df[mask]
        out_sub = df[~mask]
        ra_i, rb_i, diff_i, p_i, na_i, nb_i = two_prop(in_sub, t)
        ra_o, rb_o, diff_o, p_o, na_o, nb_o = two_prop(out_sub, t)
        out[t] = {
            "predicates": predicates,
            "in_subgroup": {"n": int(len(in_sub)), "rate_treat": ra_i, "rate_ctrl": rb_i,
                            "diff": diff_i, "p": p_i, "n_treat": na_i, "n_ctrl": nb_i},
            "out_subgroup": {"n": int(len(out_sub)), "rate_treat": ra_o, "rate_ctrl": rb_o,
                             "diff": diff_o, "p": p_o, "n_treat": na_o, "n_ctrl": nb_o},
        }
    return out


def run_iter14_three_way():
    """Test whether biomarker-gated treatment effects persist after adjusting for the strongest baseline correlates."""
    import statsmodels.api as sm
    df = DF.copy()
    df["ecog_le1"] = (df["ecog_ps"] <= 1).astype(int)
    out = {}
    for t, b in [("treatment_olaparib", "brca2_mutation"),
                 ("treatment_pembrolizumab", "msi_high"),
                 ("treatment_lu177_psma", "psma_high")]:
        cov = ["age_years", "ecog_ps", "mcrpc", "visceral_mets", "gleason_score",
               "albumin_g_dl", "ldh_u_l", "alkaline_phosphatase_u_l", "hemoglobin_g_dl",
               "psa_ng_ml", "nlr", "crp_mg_l"]
        X = df[[t, b] + cov].astype(float).copy()
        X["__inter"] = X[t] * X[b]
        X = sm.add_constant(X)
        y = df["objective_response"].astype(int)
        res = sm.Logit(y, X).fit(disp=0, maxiter=300)
        out[f"{t}__x__{b}"] = {
            "beta_treat": float(res.params[t]),
            "p_treat": float(res.pvalues[t]),
            "beta_modifier": float(res.params[b]),
            "p_modifier": float(res.pvalues[b]),
            "beta_inter": float(res.params["__inter"]),
            "p_inter": float(res.pvalues["__inter"]),
        }
    return out


def run_iter15_treatment_combos():
    """Treatment combination patterns: do patients on multiple treatments respond differently?"""
    df = DF.copy()
    df["n_tx"] = df[TREATMENTS].sum(axis=1)
    out = {"by_n_tx": df.groupby("n_tx")["objective_response"].agg(["mean", "count"]).to_dict()}
    # Specific common doublets
    pairs_to_test = [
        ("treatment_enzalutamide", "treatment_abiraterone"),
        ("treatment_enzalutamide", "treatment_docetaxel"),
        ("treatment_abiraterone", "treatment_docetaxel"),
        ("treatment_olaparib", "treatment_enzalutamide"),
    ]
    pair_results = []
    for a, b in pairs_to_test:
        both = df[(df[a] == 1) & (df[b] == 1)]
        only_a = df[(df[a] == 1) & (df[b] == 0)]
        only_b = df[(df[a] == 0) & (df[b] == 1)]
        neither = df[(df[a] == 0) & (df[b] == 0)]
        pair_results.append({
            "a": a, "b": b,
            "both_n": len(both), "both_rr": float(both["objective_response"].mean()) if len(both) else None,
            "only_a_n": len(only_a), "only_a_rr": float(only_a["objective_response"].mean()) if len(only_a) else None,
            "only_b_n": len(only_b), "only_b_rr": float(only_b["objective_response"].mean()) if len(only_b) else None,
            "neither_n": len(neither), "neither_rr": float(neither["objective_response"].mean()) if len(neither) else None,
        })
    out["pairs"] = pair_results
    return out


def main():
    RESULTS["iter1_treatment_main"] = run_iter1()
    RESULTS["iter2_biomarker_main"] = run_iter2()
    RESULTS["iter3_clinical_main"] = run_iter3_clinical()
    RESULTS["iter4_lab_main"] = run_iter4_labs()
    RESULTS["iter5_treat_x_biomarker"] = run_iter5_treat_x_biomarker()
    RESULTS["iter6_treat_x_clinical"] = run_iter6_treat_x_clinical()
    RESULTS["iter7_treat_x_lab"] = run_iter7_treat_x_lab()
    RESULTS["iter8_multivariable"] = run_iter8_multivariable()
    RESULTS["iter9_subgroup_screen"] = run_iter9_subgroup_definitions()
    RESULTS["iter10_two_feature_subgroups"] = run_iter10_two_feature_subgroups()
    RESULTS["iter11_lab_quartiles"] = run_iter11_lab_quartiles_treat()
    RESULTS["iter12_final_subgroups"] = run_iter12_final_subgroup_for_each_treatment()
    RESULTS["iter13_negation"] = run_iter13_negation_outside_subgroup()
    RESULTS["iter14_three_way_adjusted"] = run_iter14_three_way()
    RESULTS["iter15_combos"] = run_iter15_treatment_combos()

    with open(HERE / "_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print("Done. Wrote _results.json")


if __name__ == "__main__":
    main()
