"""Iterative analysis of ds001_aml dataset.

Each section corresponds to one iteration. Results are accumulated in
RESULTS for later transcript construction.
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
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
print(f"Overall response rate: {df['objective_response'].mean():.4f}")

OUT = []  # list of (label, dict) results


def log(label, **kwargs):
    rec = dict(kwargs)
    rec["__label__"] = label
    OUT.append(rec)
    keep = {k: v for k, v in kwargs.items() if k in {"effect", "p", "n", "or_", "ci"}}
    print(f"[{label}] {keep}")


# ------------------------------------------------------------------
# Iteration 1: marginal treatment-response associations
# ------------------------------------------------------------------
print("\n=== Iteration 1: marginal treatment effects ===")
treatments = [
    "treatment_midostaurin",
    "treatment_gilteritinib",
    "treatment_ivosidenib",
    "treatment_enasidenib",
    "treatment_venetoclax_azacitidine",
    "treatment_7plus3",
]
for t in treatments:
    a = df.loc[df[t] == 1, "objective_response"]
    b = df.loc[df[t] == 0, "objective_response"]
    diff = a.mean() - b.mean()
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[t], df["objective_response"]))
    log(
        f"marginal_{t}",
        treatment=t,
        rate_on=float(a.mean()),
        rate_off=float(b.mean()),
        n_on=int(a.shape[0]),
        n_off=int(b.shape[0]),
        effect=float(diff),
        p=float(p),
    )

# ------------------------------------------------------------------
# Iteration 2: marginal mutation/feature -> response
# ------------------------------------------------------------------
print("\n=== Iteration 2: marginal biomarker effects ===")
binary_feats = [
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
for f in binary_feats:
    a = df.loc[df[f] == 1, "objective_response"]
    b = df.loc[df[f] == 0, "objective_response"]
    diff = a.mean() - b.mean()
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[f], df["objective_response"]))
    log(
        f"marginal_{f}",
        feature=f,
        rate_pos=float(a.mean()),
        rate_neg=float(b.mean()),
        effect=float(diff),
        p=float(p),
    )

# Continuous feature univariates via logistic regression
print("\n=== Iteration 2b: continuous feature univariate logistic ===")
cont_feats = [
    "age_years",
    "ecog_ps",
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
for f in cont_feats:
    X = sm.add_constant(df[[f]])
    try:
        res = sm.Logit(df["objective_response"], X).fit(disp=0)
        beta = float(res.params[f])
        p = float(res.pvalues[f])
        log(f"univ_{f}", feature=f, effect=beta, p=p)
    except Exception as e:
        log(f"univ_{f}", feature=f, error=str(e))


# ------------------------------------------------------------------
# Iteration 3: targeted-therapy interactions with mutations
# ------------------------------------------------------------------
print("\n=== Iteration 3: targeted therapy by mutation interactions ===")


def interact(treatment, biomarker):
    sub = df[[treatment, biomarker, "objective_response"]].copy()
    sub.columns = ["t", "m", "y"]
    formula = "y ~ t * m"
    try:
        res = smf.logit(formula, data=sub).fit(disp=0)
        beta_int = float(res.params["t:m"])
        p_int = float(res.pvalues["t:m"])
    except Exception:
        beta_int, p_int = float("nan"), float("nan")
    rate_t1m1 = float(sub.loc[(sub.t == 1) & (sub.m == 1), "y"].mean())
    rate_t0m1 = float(sub.loc[(sub.t == 0) & (sub.m == 1), "y"].mean())
    rate_t1m0 = float(sub.loc[(sub.t == 1) & (sub.m == 0), "y"].mean())
    rate_t0m0 = float(sub.loc[(sub.t == 0) & (sub.m == 0), "y"].mean())
    n_pos = int(((sub.t == 1) & (sub.m == 1)).sum())
    log(
        f"int_{treatment}_x_{biomarker}",
        treatment=treatment,
        biomarker=biomarker,
        effect=beta_int,
        p=p_int,
        rate_t1m1=rate_t1m1,
        rate_t0m1=rate_t0m1,
        rate_t1m0=rate_t1m0,
        rate_t0m0=rate_t0m0,
        n_t1m1=n_pos,
    )


# FLT3 ITD with midostaurin and gilteritinib
interact("treatment_midostaurin", "flt3_itd")
interact("treatment_midostaurin", "flt3_tkd")
interact("treatment_gilteritinib", "flt3_itd")
interact("treatment_gilteritinib", "flt3_tkd")
# IDH-targeted
interact("treatment_ivosidenib", "idh1_mutation")
interact("treatment_enasidenib", "idh2_mutation")
# Cross-checks: targeted in non-target subgroup (already captured above)
# Venetoclax/aza by unfit
interact("treatment_venetoclax_azacitidine", "unfit_for_intensive")
# 7+3 by unfit
interact("treatment_7plus3", "unfit_for_intensive")
# 7+3 by complex karyotype
interact("treatment_7plus3", "complex_karyotype")
# 7+3 by tp53
interact("treatment_7plus3", "tp53_mutation")
# Ven/aza by tp53
interact("treatment_venetoclax_azacitidine", "tp53_mutation")
# Ven/aza by npm1
interact("treatment_venetoclax_azacitidine", "npm1_mutation")


# ------------------------------------------------------------------
# Iteration 4: stratified rates inside vs outside biomarker subgroup
# ------------------------------------------------------------------
print("\n=== Iteration 4: within-subgroup treatment effects ===")


def strat(treatment, biomarker, value):
    sub = df[df[biomarker] == value]
    a = sub.loc[sub[treatment] == 1, "objective_response"]
    b = sub.loc[sub[treatment] == 0, "objective_response"]
    if len(a) < 5 or len(b) < 5:
        return
    diff = a.mean() - b.mean()
    table = pd.crosstab(sub[treatment], sub["objective_response"])
    if table.shape == (2, 2):
        try:
            _, p = stats.fisher_exact(table.values)
        except Exception:
            chi2, p, _, _ = stats.chi2_contingency(table)
    else:
        p = float("nan")
    log(
        f"strat_{treatment}_{biomarker}={value}",
        treatment=treatment,
        biomarker=biomarker,
        value=value,
        rate_on=float(a.mean()),
        rate_off=float(b.mean()),
        n_on=int(len(a)),
        n_off=int(len(b)),
        effect=float(diff),
        p=float(p),
    )


for tx, bm in [
    ("treatment_midostaurin", "flt3_itd"),
    ("treatment_midostaurin", "flt3_tkd"),
    ("treatment_gilteritinib", "flt3_itd"),
    ("treatment_gilteritinib", "flt3_tkd"),
    ("treatment_ivosidenib", "idh1_mutation"),
    ("treatment_enasidenib", "idh2_mutation"),
    ("treatment_venetoclax_azacitidine", "unfit_for_intensive"),
    ("treatment_7plus3", "unfit_for_intensive"),
]:
    strat(tx, bm, 1)
    strat(tx, bm, 0)


# ------------------------------------------------------------------
# Iteration 5: TP53 and complex karyotype overall and by treatment
# ------------------------------------------------------------------
print("\n=== Iteration 5: TP53 / complex karyotype heterogeneity ===")
for t in treatments:
    for bm in ["tp53_mutation", "complex_karyotype", "secondary_aml"]:
        sub = df[[t, bm, "objective_response"]].copy()
        sub.columns = ["t", "m", "y"]
        try:
            res = smf.logit("y ~ t * m", data=sub).fit(disp=0)
            log(
                f"int2_{t}_x_{bm}",
                treatment=t,
                biomarker=bm,
                effect=float(res.params["t:m"]),
                p=float(res.pvalues["t:m"]),
            )
        except Exception:
            pass


# ------------------------------------------------------------------
# Iteration 6: NPM1 / FLT3-ITD interplay
# ------------------------------------------------------------------
print("\n=== Iteration 6: NPM1 + FLT3-ITD ===")
g = df.groupby(["npm1_mutation", "flt3_itd"])["objective_response"].agg(["mean", "size"])
print(g)
for npm in (0, 1):
    for itd in (0, 1):
        sub = df[(df["npm1_mutation"] == npm) & (df["flt3_itd"] == itd)]
        log(
            f"subgroup_npm1={npm}_itd={itd}",
            n=int(len(sub)),
            rate=float(sub["objective_response"].mean()),
        )


# ------------------------------------------------------------------
# Iteration 7: Multivariable logistic regression - main effects
# ------------------------------------------------------------------
print("\n=== Iteration 7: multivariable logistic regression ===")
formula_main = (
    "objective_response ~ age_years + sex_female + ecog_ps + secondary_aml "
    "+ unfit_for_intensive + complex_karyotype + flt3_itd + flt3_tkd "
    "+ idh1_mutation + idh2_mutation + npm1_mutation + tp53_mutation "
    "+ wbc_k_per_ul + blast_pct_marrow + albumin_g_dl + ldh_u_l "
    "+ weight_loss_pct_6mo + crp_mg_l + nlr "
    "+ treatment_midostaurin + treatment_gilteritinib + treatment_ivosidenib "
    "+ treatment_enasidenib + treatment_venetoclax_azacitidine + treatment_7plus3 "
    "+ hemoglobin_g_dl + alkaline_phosphatase_u_l + ast_u_l + alt_u_l "
    "+ total_bilirubin_mg_dl + creatinine_mg_dl + bun_mg_dl + sodium_meq_l "
    "+ potassium_meq_l + calcium_mg_dl"
)
res_main = smf.logit(formula_main, data=df).fit(disp=0)
print(res_main.summary().tables[1])
mv_params = res_main.params.to_dict()
mv_p = res_main.pvalues.to_dict()
for k in mv_params:
    log(f"mv_{k}", feature=k, effect=float(mv_params[k]), p=float(mv_p[k]))


# ------------------------------------------------------------------
# Iteration 8: Multivariable with key targeted-therapy x biomarker interactions
# ------------------------------------------------------------------
print("\n=== Iteration 8: multivariable with interactions ===")
formula_int = (
    formula_main
    + " + treatment_midostaurin:flt3_itd + treatment_gilteritinib:flt3_itd"
    + " + treatment_ivosidenib:idh1_mutation + treatment_enasidenib:idh2_mutation"
    + " + treatment_venetoclax_azacitidine:unfit_for_intensive"
    + " + treatment_7plus3:unfit_for_intensive"
    + " + treatment_7plus3:tp53_mutation + treatment_7plus3:complex_karyotype"
)
res_int = smf.logit(formula_int, data=df).fit(disp=0)
print(res_int.summary().tables[1])
for k in res_int.params.index:
    if ":" in k:
        log(
            f"mvint_{k}",
            term=k,
            effect=float(res_int.params[k]),
            p=float(res_int.pvalues[k]),
        )


# ------------------------------------------------------------------
# Iteration 9: Treatment x biomarker exhaustive interaction screen
# (controls for age, ecog, key prognostics)
# ------------------------------------------------------------------
print("\n=== Iteration 9: exhaustive TxB interaction screen (adjusted) ===")
covars = ["age_years", "sex_female", "ecog_ps", "secondary_aml", "unfit_for_intensive",
          "complex_karyotype", "tp53_mutation", "albumin_g_dl"]
binary_modifiers = [
    "sex_female", "secondary_aml", "unfit_for_intensive", "complex_karyotype",
    "flt3_itd", "flt3_tkd", "idh1_mutation", "idh2_mutation",
    "npm1_mutation", "tp53_mutation",
]
screen_rows = []
for t in treatments:
    for m in binary_modifiers:
        cov_terms = [c for c in covars if c != m]
        rhs = " + ".join(cov_terms) + f" + {t} + {m} + {t}:{m}"
        formula = f"objective_response ~ {rhs}"
        try:
            r = smf.logit(formula, data=df).fit(disp=0)
            term = f"{t}:{m}"
            if term in r.params.index:
                screen_rows.append((t, m, float(r.params[term]), float(r.pvalues[term])))
        except Exception:
            pass

screen_df = pd.DataFrame(screen_rows, columns=["treatment", "modifier", "beta_int", "p"])
screen_df["abs_beta"] = screen_df["beta_int"].abs()
screen_df = screen_df.sort_values("p")
print(screen_df.head(30).to_string())
for _, row in screen_df.iterrows():
    log(
        f"screen_{row['treatment']}_x_{row['modifier']}",
        treatment=row["treatment"],
        modifier=row["modifier"],
        effect=float(row["beta_int"]),
        p=float(row["p"]),
    )


# ------------------------------------------------------------------
# Iteration 10: refined subgroup checks for top hits
# ------------------------------------------------------------------
print("\n=== Iteration 10: refined subgroup checks ===")
# For each treatment, examine response among target subgroup vs target subgroup
# without unfavorable modifiers (TP53, complex karyotype, secondary AML)


def subgroup_response(treatment, predicates, label):
    mask = pd.Series(True, index=df.index)
    for col, val in predicates:
        mask &= df[col] == val
    sub = df[mask]
    if len(sub) < 30:
        log(f"sub_{label}", n=int(len(sub)), insufficient=True)
        return
    a = sub.loc[sub[treatment] == 1, "objective_response"]
    b = sub.loc[sub[treatment] == 0, "objective_response"]
    if len(a) < 5 or len(b) < 5:
        log(f"sub_{label}", n_on=int(len(a)), n_off=int(len(b)), insufficient=True)
        return
    diff = a.mean() - b.mean()
    table = pd.crosstab(sub[treatment], sub["objective_response"])
    try:
        _, p = stats.fisher_exact(table.values)
    except Exception:
        chi2, p, _, _ = stats.chi2_contingency(table)
    log(
        f"sub_{label}",
        treatment=treatment,
        n_on=int(len(a)),
        n_off=int(len(b)),
        rate_on=float(a.mean()),
        rate_off=float(b.mean()),
        effect=float(diff),
        p=float(p),
    )


# Midostaurin in FLT3-ITD with various unfavorable modifiers stripped
subgroup_response("treatment_midostaurin", [("flt3_itd", 1)], "mido_in_flt3itd")
subgroup_response("treatment_midostaurin", [("flt3_itd", 1), ("tp53_mutation", 0)], "mido_in_flt3itd_tp53wt")
subgroup_response("treatment_midostaurin", [("flt3_itd", 1), ("complex_karyotype", 0)], "mido_in_flt3itd_normalkaryo")
subgroup_response("treatment_midostaurin", [("flt3_itd", 1), ("tp53_mutation", 0), ("complex_karyotype", 0)], "mido_in_flt3itd_tp53wt_normalkaryo")

subgroup_response("treatment_gilteritinib", [("flt3_itd", 1)], "gilt_in_flt3itd")
subgroup_response("treatment_gilteritinib", [("flt3_itd", 1), ("tp53_mutation", 0)], "gilt_in_flt3itd_tp53wt")
subgroup_response("treatment_gilteritinib", [("flt3_tkd", 1)], "gilt_in_flt3tkd")

subgroup_response("treatment_ivosidenib", [("idh1_mutation", 1)], "ivo_in_idh1")
subgroup_response("treatment_ivosidenib", [("idh1_mutation", 1), ("tp53_mutation", 0)], "ivo_in_idh1_tp53wt")
subgroup_response("treatment_ivosidenib", [("idh1_mutation", 1), ("complex_karyotype", 0)], "ivo_in_idh1_normalkaryo")

subgroup_response("treatment_enasidenib", [("idh2_mutation", 1)], "ena_in_idh2")
subgroup_response("treatment_enasidenib", [("idh2_mutation", 1), ("tp53_mutation", 0)], "ena_in_idh2_tp53wt")

subgroup_response("treatment_venetoclax_azacitidine", [("unfit_for_intensive", 1)], "venaza_in_unfit")
subgroup_response("treatment_venetoclax_azacitidine", [("unfit_for_intensive", 1), ("tp53_mutation", 0)], "venaza_in_unfit_tp53wt")
subgroup_response("treatment_venetoclax_azacitidine", [("unfit_for_intensive", 0)], "venaza_in_fit")

subgroup_response("treatment_7plus3", [("unfit_for_intensive", 0)], "sevenplus3_in_fit")
subgroup_response("treatment_7plus3", [("unfit_for_intensive", 0), ("tp53_mutation", 0)], "sevenplus3_in_fit_tp53wt")
subgroup_response("treatment_7plus3", [("unfit_for_intensive", 0), ("tp53_mutation", 0), ("complex_karyotype", 0)], "sevenplus3_in_fit_tp53wt_normalkaryo")
subgroup_response("treatment_7plus3", [("unfit_for_intensive", 0), ("complex_karyotype", 0)], "sevenplus3_in_fit_normalkaryo")
subgroup_response("treatment_7plus3", [("tp53_mutation", 1)], "sevenplus3_in_tp53")


# ------------------------------------------------------------------
# Iteration 11: continuous modifier interactions (age) by treatment
# ------------------------------------------------------------------
print("\n=== Iteration 11: continuous modifier interactions ===")
for t in treatments:
    for c in ["age_years", "ecog_ps", "albumin_g_dl", "wbc_k_per_ul", "ldh_u_l"]:
        sub = df[[t, c, "objective_response"]].copy()
        sub.columns = ["t", "c", "y"]
        sub["c_z"] = (sub["c"] - sub["c"].mean()) / sub["c"].std()
        try:
            r = smf.logit("y ~ t * c_z", data=sub).fit(disp=0)
            log(
                f"contint_{t}_x_{c}",
                treatment=t,
                modifier=c,
                effect=float(r.params["t:c_z"]),
                p=float(r.pvalues["t:c_z"]),
            )
        except Exception:
            pass


# ------------------------------------------------------------------
# Iteration 12: Joint best-modifier model for each treatment
# ------------------------------------------------------------------
print("\n=== Iteration 12: joint top-modifier models ===")
# For each treatment, look at which combination of biomarkers best modifies effect
joint_specs = {
    "treatment_midostaurin": [("flt3_itd", 1), ("tp53_mutation", 0), ("complex_karyotype", 0), ("secondary_aml", 0)],
    "treatment_gilteritinib": [("flt3_itd", 1), ("tp53_mutation", 0), ("complex_karyotype", 0)],
    "treatment_ivosidenib": [("idh1_mutation", 1), ("tp53_mutation", 0), ("complex_karyotype", 0)],
    "treatment_enasidenib": [("idh2_mutation", 1), ("tp53_mutation", 0), ("complex_karyotype", 0)],
    "treatment_venetoclax_azacitidine": [("unfit_for_intensive", 1), ("tp53_mutation", 0), ("complex_karyotype", 0)],
    "treatment_7plus3": [("unfit_for_intensive", 0), ("tp53_mutation", 0), ("complex_karyotype", 0), ("secondary_aml", 0)],
}
for tx, preds in joint_specs.items():
    label = f"joint_{tx}_all"
    subgroup_response(tx, preds, label)
    # And progressively
    for k in range(1, len(preds) + 1):
        subgroup_response(tx, preds[:k], f"joint_{tx}_step{k}")


# ------------------------------------------------------------------
# Iteration 13: Tree-based heterogeneity (CART for each treatment)
# ------------------------------------------------------------------
print("\n=== Iteration 13: tree-based subgroup search ===")
try:
    from sklearn.tree import DecisionTreeClassifier, export_text

    feats = [c for c in df.columns if c not in ("patient_id", "objective_response")]
    for t in treatments:
        # Within those receiving treatment t, look for response predictors
        on = df[df[t] == 1]
        off = df[df[t] == 0]
        if len(on) < 200:
            continue
        # Estimate per-leaf treatment effect by training on full data with t as feature
        # Use a small CART on all features + t to find heterogeneity
        X = df[feats].copy()
        y = df["objective_response"].astype(int).values
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, random_state=0)
        clf.fit(X, y)
        # leaf assignments
        leaves = clf.apply(X)
        df_temp = df.copy()
        df_temp["_leaf"] = leaves
        agg = (
            df_temp.groupby(["_leaf", t])["objective_response"]
            .agg(["mean", "size"])
            .unstack(t)
        )
        log(f"tree_{t}_summary", treatment=t, leaves=int(df_temp["_leaf"].nunique()))
        # Find leaf with biggest treatment effect
        try:
            agg.columns = ["_".join(map(str, c)) for c in agg.columns]
            if "mean_1" in agg.columns and "mean_0" in agg.columns:
                agg["effect"] = agg["mean_1"] - agg["mean_0"]
                top = agg.dropna(subset=["effect"]).sort_values("effect", ascending=False).head(2)
                for leaf, row in top.iterrows():
                    log(
                        f"tree_{t}_leaf{leaf}_top",
                        treatment=t,
                        leaf=int(leaf),
                        effect=float(row["effect"]),
                        rate_on=float(row.get("mean_1", float("nan"))),
                        rate_off=float(row.get("mean_0", float("nan"))),
                    )
        except Exception as e:
            print("agg failed", t, e)
except Exception as e:
    print("Tree analysis failed:", e)


# ------------------------------------------------------------------
# Iteration 14: directly estimate effect within candidate subgroup
# Final refined definitions — test with formal logistic regression
# (treatment effect within subgroup) and outside subgroup.
# ------------------------------------------------------------------
print("\n=== Iteration 14: final subgroup tests with logistic models ===")


def final_subgroup_test(treatment, predicates, label):
    sub_in = pd.Series(True, index=df.index)
    for col, val in predicates:
        sub_in &= df[col] == val
    inside = df[sub_in]
    outside = df[~sub_in]
    out = {"sub_label": label, "treatment": treatment, "predicates": str(predicates)}
    for name, dfx in [("inside", inside), ("outside", outside)]:
        if len(dfx) > 50 and dfx[treatment].sum() > 5 and (1 - dfx[treatment]).sum() > 5:
            try:
                r = smf.logit(f"objective_response ~ {treatment}", data=dfx).fit(disp=0)
                out[f"{name}_beta"] = float(r.params[treatment])
                out[f"{name}_p"] = float(r.pvalues[treatment])
                out[f"{name}_n"] = int(len(dfx))
                out[f"{name}_rate_on"] = float(dfx.loc[dfx[treatment] == 1, "objective_response"].mean())
                out[f"{name}_rate_off"] = float(dfx.loc[dfx[treatment] == 0, "objective_response"].mean())
                # rate-difference effect
                out[f"{name}_effect"] = out[f"{name}_rate_on"] - out[f"{name}_rate_off"]
            except Exception:
                pass
    log(f"final_{label}", effect=out.get("inside_effect"), p=out.get("inside_p"), **out)


final_subgroup_test("treatment_midostaurin", [("flt3_itd", 1)], "mido_flt3itd")
final_subgroup_test("treatment_midostaurin", [("flt3_itd", 1), ("tp53_mutation", 0), ("complex_karyotype", 0)], "mido_flt3itd_no_tp53_no_complex")
final_subgroup_test("treatment_gilteritinib", [("flt3_itd", 1)], "gilt_flt3itd")
final_subgroup_test("treatment_gilteritinib", [("flt3_itd", 1), ("tp53_mutation", 0)], "gilt_flt3itd_tp53wt")
final_subgroup_test("treatment_gilteritinib", [("flt3_tkd", 1)], "gilt_flt3tkd")
final_subgroup_test("treatment_ivosidenib", [("idh1_mutation", 1)], "ivo_idh1")
final_subgroup_test("treatment_enasidenib", [("idh2_mutation", 1)], "ena_idh2")
final_subgroup_test("treatment_venetoclax_azacitidine", [("unfit_for_intensive", 1)], "venaza_unfit")
final_subgroup_test("treatment_venetoclax_azacitidine", [("unfit_for_intensive", 1), ("tp53_mutation", 0)], "venaza_unfit_tp53wt")
final_subgroup_test("treatment_venetoclax_azacitidine", [("unfit_for_intensive", 1), ("npm1_mutation", 1)], "venaza_unfit_npm1mut")
final_subgroup_test("treatment_venetoclax_azacitidine", [("unfit_for_intensive", 1), ("tp53_mutation", 0), ("complex_karyotype", 0)], "venaza_unfit_tp53wt_normalkaryo")
final_subgroup_test("treatment_venetoclax_azacitidine", [("unfit_for_intensive", 1), ("tp53_mutation", 0), ("complex_karyotype", 0), ("npm1_mutation", 1)], "venaza_unfit_tp53wt_normalkaryo_npm1")
final_subgroup_test("treatment_venetoclax_azacitidine", [("npm1_mutation", 1)], "venaza_npm1")
final_subgroup_test("treatment_venetoclax_azacitidine", [("unfit_for_intensive", 0)], "venaza_fit")
final_subgroup_test("treatment_7plus3", [("unfit_for_intensive", 0)], "sevenp3_fit")
final_subgroup_test("treatment_7plus3", [("complex_karyotype", 0)], "sevenp3_normalkaryo")
final_subgroup_test("treatment_7plus3", [("complex_karyotype", 1)], "sevenp3_complex")
final_subgroup_test("treatment_7plus3", [("unfit_for_intensive", 0), ("tp53_mutation", 0), ("complex_karyotype", 0)], "sevenp3_fit_tp53wt_normalkaryo")

# Test ven/aza x npm1 within unfit
print("\n=== Iteration 15: NPM1 within ven/aza unfit subgroup ===")
unfit = df[df["unfit_for_intensive"] == 1]
res = smf.logit("objective_response ~ treatment_venetoclax_azacitidine * npm1_mutation", data=unfit).fit(disp=0)
print(res.summary().tables[1])
log(
    "venaza_npm1_within_unfit",
    treatment="treatment_venetoclax_azacitidine",
    biomarker="npm1_mutation_within_unfit",
    effect=float(res.params["treatment_venetoclax_azacitidine:npm1_mutation"]),
    p=float(res.pvalues["treatment_venetoclax_azacitidine:npm1_mutation"]),
)
# Stratified rates
for npm in (0, 1):
    for tp in (0, 1):
        sub = unfit[(unfit["npm1_mutation"] == npm) & (unfit["tp53_mutation"] == tp)]
        a = sub.loc[sub["treatment_venetoclax_azacitidine"] == 1, "objective_response"]
        b = sub.loc[sub["treatment_venetoclax_azacitidine"] == 0, "objective_response"]
        if len(a) > 5 and len(b) > 5:
            log(
                f"venaza_unfit_npm{npm}_tp53{tp}",
                n_on=int(len(a)), n_off=int(len(b)),
                rate_on=float(a.mean()), rate_off=float(b.mean()),
                effect=float(a.mean() - b.mean()),
                p=float(stats.fisher_exact(pd.crosstab(sub["treatment_venetoclax_azacitidine"], sub["objective_response"]).values)[1])
                if pd.crosstab(sub["treatment_venetoclax_azacitidine"], sub["objective_response"]).shape == (2,2) else None,
            )


# Save raw results
import os
os.makedirs("scratch", exist_ok=True)
with open("scratch/results.json", "w") as fh:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o) if isinstance(o, np.floating) else int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    json.dump(OUT, fh, default=default, indent=2)
print(f"\nWrote {len(OUT)} result records.")
