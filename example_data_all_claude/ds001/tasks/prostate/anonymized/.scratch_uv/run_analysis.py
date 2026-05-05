"""Single-run analysis script. Iterates through propose-test-update steps
and records every analysis result to results_full.json for transcript construction.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations, product
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

warnings.filterwarnings("ignore")

DF = pd.read_parquet("../dataset.parquet")
OUT = "objective_response"
features = [c for c in DF.columns if c.startswith("feature_")]

# Drop constant features early
constants = [c for c in features if DF[c].nunique() <= 1]
features_active = [c for c in features if c not in constants]

# Classify
binary = [c for c in features_active if set(DF[c].dropna().unique()) <= {0, 1}]
small_int = [c for c in features_active if c not in binary and DF[c].nunique() <= 6 and DF[c].dtype.kind in "iu"]
continuous = [c for c in features_active if c not in binary and c not in small_int]

results = {"data_setup": {
    "n": len(DF), "outcome_mean": float(DF[OUT].mean()),
    "constants": constants, "binary": binary, "small_int": small_int, "continuous": continuous,
}}


def chi2_2x2(col):
    tab = pd.crosstab(DF[col], DF[OUT])
    if tab.shape == (2, 2):
        chi2, p, _, _ = stats.chi2_contingency(tab)
        r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
        r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
        return {"col": col, "rate_1": float(r1), "rate_0": float(r0),
                "diff": float(r1 - r0), "p": float(p),
                "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
    return None


# ============= ITER 1: Univariate binary screen =============
iter1 = []
for c in binary:
    r = chi2_2x2(c)
    if r:
        iter1.append(r)
iter1 = sorted(iter1, key=lambda x: x["p"])
results["iter1_binary_univariate"] = iter1


# ============= ITER 2: Univariate small-int and continuous screen =============
iter2_smallint = []
for c in small_int:
    tab = pd.crosstab(DF[c], DF[OUT])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    rates = (tab.iloc[:, 1] / tab.sum(axis=1)).to_dict()
    iter2_smallint.append({"col": c, "rates": {int(k): float(v) for k, v in rates.items()},
                           "p": float(p), "chi2": float(chi2)})
iter2_smallint = sorted(iter2_smallint, key=lambda x: x["p"])

iter2_cont = []
for c in continuous:
    a = DF.loc[DF[OUT] == 1, c].dropna()
    b = DF.loc[DF[OUT] == 0, c].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    iter2_cont.append({"col": c, "mean_resp": float(a.mean()), "mean_nonresp": float(b.mean()),
                       "diff": float(a.mean() - b.mean()), "t": float(t), "p": float(p)})
iter2_cont = sorted(iter2_cont, key=lambda x: x["p"])
results["iter2_smallint"] = iter2_smallint
results["iter2_continuous"] = iter2_cont


# ============= ITER 3: Multivariable logistic regression =============
def build_design(df, cols, std_continuous=True):
    X = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c].astype(float)
        if c in continuous and std_continuous:
            # log1p for skewed positives
            if s.min() >= 0 and s.skew() > 1.5:
                s = np.log1p(s)
            s = (s - s.mean()) / s.std()
        X[c] = s
    X = sm.add_constant(X)
    return X


X3 = build_design(DF, [c for c in features_active])
y = DF[OUT].astype(int)
m3 = sm.Logit(y, X3).fit(disp=0, maxiter=200)
iter3 = {"params": m3.params.to_dict(), "pvalues": m3.pvalues.to_dict(),
         "llf": float(m3.llf), "aic": float(m3.aic),
         "summary_table": [(name, float(m3.params[name]), float(m3.pvalues[name]))
                           for name in m3.params.index]}
results["iter3_multivariable"] = iter3


# ============= ITER 4: 2x2 tables among top features =============
# Focus on interaction between the two strongest binary effects
def cross_2x2x2(c1, c2):
    cells = {}
    for v1, v2 in product([0, 1], [0, 1]):
        m = (DF[c1] == v1) & (DF[c2] == v2)
        n = int(m.sum())
        rate = float(DF.loc[m, OUT].mean()) if n else None
        cells[f"{c1}={v1},{c2}={v2}"] = {"n": n, "rate": rate}
    return cells


def interaction_logit(c1, c2):
    sub = DF[[c1, c2, OUT]].copy()
    sub["x1x2"] = sub[c1] * sub[c2]
    X = sm.add_constant(sub[[c1, c2, "x1x2"]].astype(float))
    m = sm.Logit(sub[OUT], X).fit(disp=0, maxiter=200)
    return {"main_c1": float(m.params[c1]), "main_c2": float(m.params[c2]),
            "interaction": float(m.params["x1x2"]),
            "p_interaction": float(m.pvalues["x1x2"])}


iter4 = {
    "f013_x_f008_cells": cross_2x2x2("feature_013", "feature_008"),
    "f013_x_f008_logit": interaction_logit("feature_013", "feature_008"),
}
results["iter4_f013_f008_interaction"] = iter4


# ============= ITER 5: Stratify candidate "treatments" by f013 and f008 =============
candidates = ["feature_015", "feature_021", "feature_027", "feature_001"]
stratifiers = ["feature_013", "feature_008"]
iter5 = {}
for cand in candidates:
    for strat in stratifiers:
        for sv in sorted(DF[strat].unique()):
            sub = DF[DF[strat] == sv]
            if cand == "feature_001":
                # ordinal — compute mean at each level
                rates = sub.groupby(cand)[OUT].mean().to_dict()
                ns = sub.groupby(cand)[OUT].count().to_dict()
                iter5[f"{cand}|{strat}={sv}"] = {"rates_by_level": {int(k): float(v) for k, v in rates.items()},
                                                  "ns_by_level": {int(k): int(v) for k, v in ns.items()}}
            else:
                if sub[cand].nunique() < 2:
                    continue
                tab = pd.crosstab(sub[cand], sub[OUT])
                chi2, p, _, _ = stats.chi2_contingency(tab)
                r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
                r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
                iter5[f"{cand}|{strat}={sv}"] = {"rate_1": float(r1), "rate_0": float(r0),
                                                  "diff": float(r1 - r0), "p": float(p),
                                                  "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
results["iter5_stratified_candidates"] = iter5


# ============= ITER 6: Joint subgroup — restrict to (f013=0 AND f008=1) =============
sub_strict = DF[(DF["feature_013"] == 0) & (DF["feature_008"] == 1)].copy()
results["iter6_strict_subgroup"] = {
    "n": int(len(sub_strict)),
    "outcome_mean": float(sub_strict[OUT].mean()),
}
strict_effects = {}
for cand in ["feature_015", "feature_021", "feature_027", "feature_001"]:
    if cand == "feature_001":
        rates = sub_strict.groupby(cand)[OUT].mean().to_dict()
        ns = sub_strict.groupby(cand)[OUT].count().to_dict()
        strict_effects[cand] = {"rates_by_level": {int(k): float(v) for k, v in rates.items()},
                                 "ns_by_level": {int(k): int(v) for k, v in ns.items()}}
    else:
        if sub_strict[cand].nunique() < 2:
            continue
        tab = pd.crosstab(sub_strict[cand], sub_strict[OUT])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
        r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
        strict_effects[cand] = {"rate_1": float(r1), "rate_0": float(r0),
                                 "diff": float(r1 - r0), "p": float(p),
                                 "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
results["iter6_strict_subgroup_effects"] = strict_effects


# ============= ITER 7: Balance check — are candidate treatments distributed independently of f013/f008? =============
balance = {}
for cand in ["feature_015", "feature_021", "feature_027"]:
    for strat in ["feature_013", "feature_008"]:
        tab = pd.crosstab(DF[strat], DF[cand])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        balance[f"{cand}_x_{strat}"] = {"p": float(p), "table": tab.values.tolist()}
results["iter7_treatment_balance"] = balance


# ============= ITER 8: Three-way interaction f015 × f013 × f008 =============
iter8 = {}
for v013 in [0, 1]:
    for v008 in [0, 1]:
        sub = DF[(DF["feature_013"] == v013) & (DF["feature_008"] == v008)]
        if sub["feature_015"].nunique() < 2:
            continue
        tab = pd.crosstab(sub["feature_015"], sub[OUT])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
        r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
        iter8[f"f013={v013},f008={v008}"] = {"diff": float(r1 - r0), "p": float(p),
                                              "rate_0": float(r0), "rate_1": float(r1),
                                              "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
results["iter8_threeway_f015"] = iter8

# Same for feature_021 and feature_027
for cand in ["feature_021", "feature_027"]:
    out = {}
    for v013 in [0, 1]:
        for v008 in [0, 1]:
            sub = DF[(DF["feature_013"] == v013) & (DF["feature_008"] == v008)]
            if sub[cand].nunique() < 2:
                continue
            tab = pd.crosstab(sub[cand], sub[OUT])
            chi2, p, _, _ = stats.chi2_contingency(tab)
            r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
            r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
            out[f"f013={v013},f008={v008}"] = {"diff": float(r1 - r0), "p": float(p),
                                                "rate_0": float(r0), "rate_1": float(r1),
                                                "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
    results[f"iter8_threeway_{cand}"] = out


# ============= ITER 9: Continuous modifier screen within strict subgroup =============
# Within (f013=0, f008=1), do continuous features modify f015 effect?
sub_str = DF[(DF["feature_013"] == 0) & (DF["feature_008"] == 1)].copy()

iter9 = {}
for c in continuous:
    s = sub_str[c]
    med = s.median()
    high = sub_str[s > med]
    low = sub_str[s <= med]
    diffs = []
    for half, label in [(low, "low"), (high, "high")]:
        if half["feature_015"].nunique() < 2:
            continue
        tab = pd.crosstab(half["feature_015"], half[OUT])
        if tab.shape != (2, 2):
            continue
        chi2, p, _, _ = stats.chi2_contingency(tab)
        r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
        r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
        diffs.append({"half": label, "diff": float(r1 - r0), "p": float(p),
                      "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())})
    if len(diffs) == 2:
        iter9[c] = diffs
results["iter9_continuous_modifiers_f015"] = iter9


# ============= ITER 10: feature_001 as additional necessary condition? =============
# Check f015 effect within (f013=0, f008=1) by f001 level
iter10 = {}
for v in [0, 1, 2]:
    sub = DF[(DF["feature_013"] == 0) & (DF["feature_008"] == 1) & (DF["feature_001"] == v)]
    if len(sub) < 50 or sub["feature_015"].nunique() < 2:
        continue
    tab = pd.crosstab(sub["feature_015"], sub[OUT])
    if tab.shape != (2, 2):
        continue
    chi2, p, _, _ = stats.chi2_contingency(tab)
    r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
    r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
    iter10[f"f001={v}"] = {"diff": float(r1 - r0), "p": float(p),
                            "rate_0": float(r0), "rate_1": float(r1),
                            "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
results["iter10_f015_by_f001_within_strict"] = iter10


# ============= ITER 11: f021 by 4-cell, full subgroup necessary conditions =============
iter11 = {}
for v013 in [0, 1]:
    for v008 in [0, 1]:
        sub = DF[(DF["feature_013"] == v013) & (DF["feature_008"] == v008)]
        for cand in ["feature_015", "feature_021", "feature_027"]:
            if sub[cand].nunique() < 2:
                continue
            tab = pd.crosstab(sub[cand], sub[OUT])
            if tab.shape != (2, 2):
                continue
            chi2, p, _, _ = stats.chi2_contingency(tab)
            r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
            r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
            iter11[f"{cand}|f013={v013},f008={v008}"] = {"diff": float(r1 - r0), "p": float(p),
                                                           "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
results["iter11_full_subgroup_check"] = iter11


# ============= ITER 12: Joint subgroup (best subgroup definition) – check whether removing any condition matters =============
# best subgroup: f013=0 AND f008=1; treatment = feature_015
# Verify: in the broader populations (relax one condition), is f015 effect smaller?
iter12 = {}
# (a) no conditioning
tab = pd.crosstab(DF["feature_015"], DF[OUT])
chi2, p, _, _ = stats.chi2_contingency(tab)
iter12["overall"] = {"diff": float(tab.iloc[1, 1] / tab.iloc[1].sum() - tab.iloc[0, 1] / tab.iloc[0].sum()),
                       "p": float(p)}
# (b) only f013=0
sub = DF[DF["feature_013"] == 0]
tab = pd.crosstab(sub["feature_015"], sub[OUT])
chi2, p, _, _ = stats.chi2_contingency(tab)
iter12["f013=0_only"] = {"diff": float(tab.iloc[1, 1] / tab.iloc[1].sum() - tab.iloc[0, 1] / tab.iloc[0].sum()),
                            "p": float(p)}
# (c) only f008=1
sub = DF[DF["feature_008"] == 1]
tab = pd.crosstab(sub["feature_015"], sub[OUT])
chi2, p, _, _ = stats.chi2_contingency(tab)
iter12["f008=1_only"] = {"diff": float(tab.iloc[1, 1] / tab.iloc[1].sum() - tab.iloc[0, 1] / tab.iloc[0].sum()),
                            "p": float(p)}
# (d) joint
sub = DF[(DF["feature_013"] == 0) & (DF["feature_008"] == 1)]
tab = pd.crosstab(sub["feature_015"], sub[OUT])
chi2, p, _, _ = stats.chi2_contingency(tab)
iter12["joint_f013=0,f008=1"] = {"diff": float(tab.iloc[1, 1] / tab.iloc[1].sum() - tab.iloc[0, 1] / tab.iloc[0].sum()),
                                    "p": float(p)}
# (e) opposite joint
sub = DF[(DF["feature_013"] == 1) & (DF["feature_008"] == 0)]
tab = pd.crosstab(sub["feature_015"], sub[OUT])
if tab.shape == (2, 2):
    chi2, p, _, _ = stats.chi2_contingency(tab)
    iter12["joint_f013=1,f008=0"] = {"diff": float(tab.iloc[1, 1] / tab.iloc[1].sum() - tab.iloc[0, 1] / tab.iloc[0].sum()),
                                        "p": float(p)}
results["iter12_subgroup_necessity"] = iter12


# ============= ITER 13: Three-way interactions with continuous and other candidate "treatments" =============
# Test if any other binary feature shows similar interaction with f013 or f008
iter13 = {}
for cand in binary:
    if cand in ["feature_013", "feature_008"]:
        continue
    out = {}
    for strat in ["feature_013", "feature_008"]:
        for v in [0, 1]:
            sub = DF[DF[strat] == v]
            if sub[cand].nunique() < 2:
                continue
            tab = pd.crosstab(sub[cand], sub[OUT])
            if tab.shape != (2, 2):
                continue
            chi2, p, _, _ = stats.chi2_contingency(tab)
            r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
            r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
            out[f"{strat}={v}"] = {"diff": float(r1 - r0), "p": float(p)}
    iter13[cand] = out
results["iter13_all_binary_stratified"] = iter13


# ============= ITER 14: Continuous modifier screen for each treatment effect within strict subgroup =============
# For f015, f021, f027 within (f013=0, f008=1) — does a continuous feature wipe out the effect?
sub_str = DF[(DF["feature_013"] == 0) & (DF["feature_008"] == 1)].copy()
iter14 = {}
for cand in ["feature_015", "feature_021", "feature_027"]:
    for c in continuous + small_int:
        if cand == c:
            continue
        s = sub_str[c]
        med = s.median()
        for op, half_label in [(s > med, "high"), (s <= med, "low")]:
            half = sub_str[op]
            if half[cand].nunique() < 2 or len(half) < 50:
                continue
            tab = pd.crosstab(half[cand], half[OUT])
            if tab.shape != (2, 2):
                continue
            chi2, p, _, _ = stats.chi2_contingency(tab)
            r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
            r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
            iter14[f"{cand}|{c}_{half_label}"] = {"diff": float(r1 - r0), "p": float(p),
                                                    "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
results["iter14_continuous_subgroup_treatment"] = iter14


# ============= ITER 15: Logistic regression with treatment*feature interactions inside strict subgroup =============
iter15 = {}
for cand in ["feature_015", "feature_021", "feature_027"]:
    sub = sub_str.copy()
    if sub[cand].nunique() < 2:
        continue
    for c in [c for c in features_active if c not in [cand, "feature_013", "feature_008"]]:
        try:
            sub2 = sub[[cand, c, OUT]].copy()
            sub2["x"] = sub2[cand] * sub2[c]
            X = sm.add_constant(sub2[[cand, c, "x"]].astype(float))
            m = sm.Logit(sub2[OUT], X).fit(disp=0, maxiter=100)
            iter15[f"{cand}*{c}"] = {"int_coef": float(m.params["x"]),
                                       "p": float(m.pvalues["x"]),
                                       "main_treat": float(m.params[cand]),
                                       "p_main_treat": float(m.pvalues[cand])}
        except Exception as e:
            iter15[f"{cand}*{c}"] = {"error": str(e)}
results["iter15_strict_subgroup_interactions"] = iter15


# ============= ITER 16: Look at f021 and f027 in same way as f015 =============
# Best subgroup for each treatment
iter16 = {}
for cand in ["feature_021", "feature_027"]:
    # In strict subgroup
    sub = sub_str.copy()
    if sub[cand].nunique() < 2:
        continue
    tab = pd.crosstab(sub[cand], sub[OUT])
    if tab.shape != (2, 2):
        continue
    chi2, p, _, _ = stats.chi2_contingency(tab)
    r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
    r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
    iter16[f"{cand}_in_strict"] = {"diff": float(r1 - r0), "p": float(p),
                                     "rate_0": float(r0), "rate_1": float(r1),
                                     "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
results["iter16_other_treatments_strict"] = iter16


# ============= ITER 17: Does the (f013=0,f008=1) subgroup have a third necessary condition for f015? =============
# Take strict subgroup, look for binary features whose value modifies the f015 effect strongly
iter17 = {}
for c in binary + small_int:
    if c in ["feature_013", "feature_008", "feature_015"]:
        continue
    if sub_str[c].nunique() < 2:
        continue
    for v in sorted(sub_str[c].unique()):
        sub = sub_str[sub_str[c] == v]
        if sub["feature_015"].nunique() < 2 or len(sub) < 100:
            continue
        tab = pd.crosstab(sub["feature_015"], sub[OUT])
        if tab.shape != (2, 2):
            continue
        chi2, p, _, _ = stats.chi2_contingency(tab)
        r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
        r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
        iter17[f"{c}={v}"] = {"diff": float(r1 - r0), "p": float(p),
                                "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
results["iter17_third_modifier_search"] = iter17


# ============= ITER 18: Same for f021 =============
iter18 = {}
for c in binary + small_int:
    if c in ["feature_013", "feature_008", "feature_021"]:
        continue
    if sub_str[c].nunique() < 2:
        continue
    for v in sorted(sub_str[c].unique()):
        sub = sub_str[sub_str[c] == v]
        if sub["feature_021"].nunique() < 2 or len(sub) < 100:
            continue
        tab = pd.crosstab(sub["feature_021"], sub[OUT])
        if tab.shape != (2, 2):
            continue
        chi2, p, _, _ = stats.chi2_contingency(tab)
        r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
        r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
        iter18[f"{c}={v}"] = {"diff": float(r1 - r0), "p": float(p),
                                "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
results["iter18_f021_third_modifier"] = iter18


# ============= ITER 19: Same for f027 =============
iter19 = {}
for c in binary + small_int:
    if c in ["feature_013", "feature_008", "feature_027"]:
        continue
    if sub_str[c].nunique() < 2:
        continue
    for v in sorted(sub_str[c].unique()):
        sub = sub_str[sub_str[c] == v]
        if sub["feature_027"].nunique() < 2 or len(sub) < 50:
            continue
        tab = pd.crosstab(sub["feature_027"], sub[OUT])
        if tab.shape != (2, 2):
            continue
        chi2, p, _, _ = stats.chi2_contingency(tab)
        r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
        r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
        iter19[f"{c}={v}"] = {"diff": float(r1 - r0), "p": float(p),
                                "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
results["iter19_f027_third_modifier"] = iter19


# ============= ITER 20: Continuous-feature interactions (centered) in strict subgroup =============
iter20 = {}
for cand in ["feature_015", "feature_021"]:
    for c in continuous:
        try:
            sub = sub_str[[cand, c, OUT]].copy()
            sub["c_std"] = (sub[c] - sub[c].mean()) / sub[c].std()
            sub["x"] = sub[cand] * sub["c_std"]
            X = sm.add_constant(sub[[cand, "c_std", "x"]].astype(float))
            m = sm.Logit(sub[OUT], X).fit(disp=0, maxiter=100)
            iter20[f"{cand}*{c}"] = {"int_coef": float(m.params["x"]),
                                       "p": float(m.pvalues["x"])}
        except Exception:
            pass
results["iter20_continuous_x_treatment_strict"] = iter20


# ============= ITER 21: Re-confirm overall effect of best subgroup definition =============
# For each candidate "treatment", find the subgroup where it has the largest absolute effect
# Test best subgroup: f015 in (f013=0, f008=1)
# Refine: does f001 modify it within strict subgroup?
iter21 = {}
sub_str = DF[(DF["feature_013"] == 0) & (DF["feature_008"] == 1)]
for v in sorted(DF["feature_001"].unique()):
    sub = sub_str[sub_str["feature_001"] == v]
    if sub["feature_015"].nunique() < 2:
        continue
    tab = pd.crosstab(sub["feature_015"], sub[OUT])
    if tab.shape != (2, 2):
        continue
    chi2, p, _, _ = stats.chi2_contingency(tab)
    r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
    r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
    iter21[f"f001={int(v)}"] = {"diff": float(r1 - r0), "p": float(p),
                                  "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum()),
                                  "rate_0": float(r0), "rate_1": float(r1)}
results["iter21_f015_by_f001_in_strict"] = iter21


# ============= ITER 22: feature_002 (continuous) as effect modifier =============
# Within strict, split f002 at median
iter22 = {}
for c in ["feature_002", "feature_022", "feature_020", "feature_024"]:
    s = sub_str[c]
    med = s.median()
    for op, label in [(s > med, "high"), (s <= med, "low")]:
        half = sub_str[op]
        if half["feature_015"].nunique() < 2:
            continue
        tab = pd.crosstab(half["feature_015"], half[OUT])
        if tab.shape != (2, 2):
            continue
        chi2, p, _, _ = stats.chi2_contingency(tab)
        r1 = tab.iloc[1, 1] / tab.iloc[1].sum()
        r0 = tab.iloc[0, 1] / tab.iloc[0].sum()
        iter22[f"{c}_{label}"] = {"diff": float(r1 - r0), "p": float(p),
                                    "n0": int(tab.iloc[0].sum()), "n1": int(tab.iloc[1].sum())}
results["iter22_continuous_modifiers_f015_strict"] = iter22


# ============= ITER 23: Joint logit with all known signals within strict subgroup =============
iter23 = None
try:
    cols = ["feature_015", "feature_021", "feature_027", "feature_001", "feature_002",
            "feature_020", "feature_022", "feature_024"]
    Xs = build_design(sub_str, cols)
    ms = sm.Logit(sub_str[OUT], Xs).fit(disp=0, maxiter=200)
    iter23 = {"params": ms.params.to_dict(), "pvalues": ms.pvalues.to_dict()}
except Exception as e:
    iter23 = {"error": str(e)}
results["iter23_strict_multivariable"] = iter23


# ============= ITER 24: Verify no confounding through balance of f015 by f001/f002/f022/f020/f024 in strict =============
iter24 = {}
for c in ["feature_001", "feature_002", "feature_020", "feature_022", "feature_024"]:
    by = sub_str.groupby("feature_015")[c].agg(["mean", "median"]).to_dict()
    iter24[c] = {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in by.items()}
results["iter24_balance_in_strict"] = iter24


# ============= ITER 25: Final subgroup confirmation =============
# Test "treatment": feature_015=1; subgroup definition: feature_013=0 AND feature_008=1
# Versus controls: same conditions but feature_015=0 in same subgroup
final = {}
sub = DF[(DF["feature_013"] == 0) & (DF["feature_008"] == 1)]
treated = sub[sub["feature_015"] == 1]
control = sub[sub["feature_015"] == 0]
final["n_subgroup"] = int(len(sub))
final["n_treated"] = int(len(treated))
final["n_control"] = int(len(control))
final["rate_treated"] = float(treated[OUT].mean())
final["rate_control"] = float(control[OUT].mean())
final["abs_diff"] = float(treated[OUT].mean() - control[OUT].mean())
tab = pd.crosstab(sub["feature_015"], sub[OUT])
chi2, p, _, _ = stats.chi2_contingency(tab)
final["p"] = float(p)
results["iter25_final_confirmation"] = final


# Save everything
with open("results_full.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("done")
print(json.dumps({"keys": list(results.keys())}, indent=2))
