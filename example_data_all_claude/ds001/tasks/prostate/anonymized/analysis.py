"""Comprehensive analysis for ds001_prostate.
Captures numbers needed to populate transcript.json across 25 iterations.
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
y = df["objective_response"].astype(int)
features = [c for c in df.columns if c.startswith("feature_")]
binary_feats = [c for c in features if df[c].nunique() == 2]
constant_feats = [c for c in features if df[c].nunique() == 1]
multilvl_feats = [c for c in features if 2 < df[c].nunique() <= 5]
continuous_feats = [c for c in features if df[c].nunique() > 5]

results = {}


def logreg(X, y, name=None):
    Xc = sm.add_constant(X, has_constant="add")
    m = sm.Logit(y, Xc).fit(disp=False, maxiter=200)
    return m


# ============================================================
# Iteration 1: Univariate associations of every feature with outcome.
# ============================================================
uni_rows = []
for f in features:
    s = df[f]
    if s.nunique() == 1:
        uni_rows.append({"feature": f, "type": "constant", "p": np.nan, "effect": 0.0})
        continue
    if s.nunique() == 2:
        # logistic regression slope (log-odds)
        m = logreg(s.astype(float), y)
        beta = m.params.iloc[1]
        p = m.pvalues.iloc[1]
        # also report OR and rate diff
        rate1 = y[s == 1].mean()
        rate0 = y[s == 0].mean()
        uni_rows.append({"feature": f, "type": "binary", "p": p, "effect": beta,
                         "rate0": rate0, "rate1": rate1, "n0": (s == 0).sum(), "n1": (s == 1).sum()})
    else:
        # continuous: logistic with z-scored
        z = (s - s.mean()) / s.std()
        m = logreg(z.astype(float), y)
        beta = m.params.iloc[1]
        p = m.pvalues.iloc[1]
        # also Mann-Whitney
        try:
            mw = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided")
            mwp = mw.pvalue
        except Exception:
            mwp = np.nan
        uni_rows.append({"feature": f, "type": "continuous", "p": p, "effect": beta,
                         "mean_y0": s[y == 0].mean(), "mean_y1": s[y == 1].mean(), "mw_p": mwp})

uni_df = pd.DataFrame(uni_rows).sort_values("p")
results["univariate"] = uni_df
print("=== Iter1: Univariate ===")
print(uni_df.to_string())

# ============================================================
# Iteration 2: Multivariable logistic regression with all non-constant features.
# Continuous features z-scored; binary as-is; multi-level treated as numeric.
# ============================================================
X_cols = [c for c in features if c not in constant_feats]
X = df[X_cols].copy().astype(float)
for c in continuous_feats:
    X[c] = (X[c] - X[c].mean()) / X[c].std()
mvm = logreg(X, y)
mv_summary = pd.DataFrame({
    "feature": ["const"] + X_cols,
    "beta": mvm.params.values,
    "p": mvm.pvalues.values,
    "or": np.exp(mvm.params.values),
}).sort_values("p")
results["multivariable"] = mv_summary
print("\n=== Iter2: Multivariable logistic ===")
print(mv_summary.to_string())
print("LL:", mvm.llf, "df:", mvm.df_resid)

# ============================================================
# Iteration 3: Identify candidate "treatments" — binary features with sizeable
# main effects. Examine balance of those vs covariates.
# ============================================================
# Pick the top ~5 binary features by absolute effect for treatment candidacy.
bin_uni = uni_df[uni_df["type"] == "binary"].copy()
bin_uni["abs_effect"] = bin_uni["effect"].abs()
bin_uni = bin_uni.sort_values("abs_effect", ascending=False)
print("\n=== Iter3: Binary feature effects (sorted by |effect|) ===")
print(bin_uni.to_string())

# Quickly pick treatment candidates: those with the largest effects.
treatment_candidates = bin_uni.head(6)["feature"].tolist()
print("Top treatment candidates:", treatment_candidates)

# ============================================================
# Iteration 4-6: For each candidate treatment, screen interactions with all
# other features (binary and continuous) on outcome.
# ============================================================
def interaction_test(t_col, m_col, y, df):
    """Return (p_int, beta_int, beta_t_at_m0, beta_t_at_m1)."""
    t = df[t_col].astype(float)
    if df[m_col].nunique() == 2:
        m = df[m_col].astype(float)
    else:
        s = df[m_col]
        m = ((s - s.mean()) / s.std()).astype(float)
    inter = t * m
    X = pd.DataFrame({"t": t, "m": m, "tm": inter})
    fit = logreg(X, y)
    return fit.pvalues["tm"], fit.params["tm"], fit.params["t"], fit.params["m"]


inter_rows = []
non_const_feats = [c for c in features if c not in constant_feats]
for t in treatment_candidates:
    for m in non_const_feats:
        if m == t:
            continue
        try:
            p_int, b_int, b_t, b_m = interaction_test(t, m, y, df)
            inter_rows.append({"treatment": t, "modifier": m, "p_int": p_int,
                               "beta_int": b_int, "beta_t": b_t, "beta_m": b_m})
        except Exception as e:
            inter_rows.append({"treatment": t, "modifier": m, "p_int": np.nan,
                               "beta_int": np.nan, "error": str(e)})

inter_df = pd.DataFrame(inter_rows)
inter_df = inter_df.sort_values("p_int")
results["interactions"] = inter_df
print("\n=== Iter4-6: Top 30 treatment-by-modifier interactions ===")
print(inter_df.head(30).to_string())

# ============================================================
# Iteration 7: Pick the strongest treatment + modifier combinations and
# characterize stratified effects (rate differences).
# ============================================================
top_inter = inter_df.head(15).copy()
strat_rows = []
for _, r in top_inter.iterrows():
    t, m = r["treatment"], r["modifier"]
    # build modifier strata
    if df[m].nunique() == 2:
        for mv in [0, 1]:
            sub = df[df[m] == mv]
            r1 = sub.loc[sub[t] == 1, "objective_response"].mean()
            r0 = sub.loc[sub[t] == 0, "objective_response"].mean()
            n1 = (sub[t] == 1).sum()
            n0 = (sub[t] == 0).sum()
            strat_rows.append({"treatment": t, "modifier": m, "modifier_val": mv,
                               "n_t1": n1, "n_t0": n0, "rate_t1": r1, "rate_t0": r0,
                               "rate_diff": r1 - r0, "p_int": r["p_int"]})
    else:
        med = df[m].median()
        for label, mask in [("low", df[m] <= med), ("high", df[m] > med)]:
            sub = df[mask]
            r1 = sub.loc[sub[t] == 1, "objective_response"].mean()
            r0 = sub.loc[sub[t] == 0, "objective_response"].mean()
            n1 = (sub[t] == 1).sum()
            n0 = (sub[t] == 0).sum()
            strat_rows.append({"treatment": t, "modifier": m, "modifier_val": label,
                               "n_t1": n1, "n_t0": n0, "rate_t1": r1, "rate_t0": r0,
                               "rate_diff": r1 - r0, "p_int": r["p_int"]})

strat_df = pd.DataFrame(strat_rows)
results["stratified"] = strat_df
print("\n=== Iter7: Stratified rates for top interactions ===")
print(strat_df.to_string())

# ============================================================
# Iteration 8: Joint subgroup hypothesis. For the strongest interaction,
# build a 3-feature subgroup and test the combined effect.
# ============================================================
# Take top treatment overall by main effect:
top_treatment = treatment_candidates[0]
print(f"\n=== Iter8: Joint subgroup discovery for treatment={top_treatment} ===")

# For top_treatment, find the top 5 interaction modifiers
mods_for_t = inter_df[inter_df["treatment"] == top_treatment].sort_values("p_int").head(8)
print(mods_for_t.to_string())

# Build 2-feature subgroups: dichotomize each modifier and test treatment effect
# in each combination of (top mod 1, top mod 2).
def dichotomize(col):
    if df[col].nunique() == 2:
        return df[col].astype(int)
    return (df[col] > df[col].median()).astype(int)


m1 = mods_for_t.iloc[0]["modifier"]
m2 = mods_for_t.iloc[1]["modifier"]
g1 = dichotomize(m1)
g2 = dichotomize(m2)
print(f"Dichotomized modifiers: {m1}, {m2}")

joint_rows = []
for v1 in [0, 1]:
    for v2 in [0, 1]:
        mask = (g1 == v1) & (g2 == v2)
        sub = df[mask]
        n1 = (sub[top_treatment] == 1).sum()
        n0 = (sub[top_treatment] == 0).sum()
        if n1 < 20 or n0 < 20:
            joint_rows.append({"m1_val": v1, "m2_val": v2, "n_t1": n1, "n_t0": n0,
                               "rate_t1": np.nan, "rate_t0": np.nan, "rd": np.nan, "p": np.nan})
            continue
        r1 = sub.loc[sub[top_treatment] == 1, "objective_response"].mean()
        r0 = sub.loc[sub[top_treatment] == 0, "objective_response"].mean()
        # 2x2 chi-square
        ct = pd.crosstab(sub[top_treatment], sub["objective_response"])
        if ct.shape == (2, 2):
            chi2, pp, _, _ = stats.chi2_contingency(ct.values)
        else:
            pp = np.nan
        joint_rows.append({"m1_val": v1, "m2_val": v2, "n_t1": n1, "n_t0": n0,
                           "rate_t1": r1, "rate_t0": r0, "rd": r1 - r0, "p": pp})

joint_df = pd.DataFrame(joint_rows)
results["joint"] = joint_df
print(joint_df.to_string())

# ============================================================
# Iteration 9: Try other treatments (the next candidate) and run interaction screen.
# ============================================================
print(f"\n=== Iter9: Repeat for second-best treatment candidate ===")
t2 = treatment_candidates[1]
mods_for_t2 = inter_df[inter_df["treatment"] == t2].sort_values("p_int").head(8)
print(mods_for_t2.to_string())

# ============================================================
# Iteration 10: Decision-tree-driven subgroup discovery for interaction with top treatment.
# Use sklearn DecisionTreeClassifier on (treatment, features) → outcome with depth 3.
# ============================================================
from sklearn.tree import DecisionTreeRegressor

# Causal-forest-lite: regress residualized outcome on residualized treatment per leaf.
# Alternative: fit a tree to predict outcome separately in T=1 and T=0, compare.
print("\n=== Iter10: Tree-based subgroup search (depth-3 trees per arm) ===")
arm1_mask = df[top_treatment] == 1
X_features = [c for c in non_const_feats if c != top_treatment]
Xall = df[X_features].astype(float).values
y_arr = y.values

t1_tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=300, random_state=0).fit(
    Xall[arm1_mask.values], y_arr[arm1_mask.values])
t0_tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=300, random_state=0).fit(
    Xall[~arm1_mask.values], y_arr[~arm1_mask.values])
pred1 = t1_tree.predict(Xall)
pred0 = t0_tree.predict(Xall)
ite = pred1 - pred0
print("ITE distribution:")
print(pd.Series(ite).describe())
# Find leaves with most positive vs most negative ITE
top_pos = np.argsort(-ite)[:1000]
top_neg = np.argsort(ite)[:1000]
print(f"Top 1000 ITE+ avg: {ite[top_pos].mean():.4f}")
print(f"Top 1000 ITE- avg: {ite[top_neg].mean():.4f}")

# Inspect the strongest "effect-modifier" combinations: within the tree leaves
# of the difference. Build a single tree on ITE.
ite_tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, random_state=0).fit(Xall, ite)
from sklearn.tree import export_text
print("Tree on ITE estimates:")
print(export_text(ite_tree, feature_names=X_features, max_depth=4))

# ============================================================
# Iteration 11: Robustness — refit logistic regression on subgroup definitions
# from the ITE tree.
# ============================================================
# We'll record leaf membership and treatment effect per leaf
leaf = ite_tree.apply(Xall)
leaf_tbl = []
for L in np.unique(leaf):
    mask = leaf == L
    sub_t = df.loc[mask, top_treatment]
    sub_y = y[mask]
    n1 = (sub_t == 1).sum()
    n0 = (sub_t == 0).sum()
    if n1 < 20 or n0 < 20:
        continue
    r1 = sub_y[sub_t == 1].mean()
    r0 = sub_y[sub_t == 0].mean()
    ct = pd.crosstab(sub_t, sub_y)
    chi2, pp, _, _ = stats.chi2_contingency(ct.values) if ct.shape == (2, 2) else (np.nan, np.nan, None, None)
    leaf_tbl.append({"leaf": L, "n": mask.sum(), "n_t1": n1, "n_t0": n0,
                     "rate_t1": r1, "rate_t0": r0, "rd": r1 - r0, "p_chi2": pp,
                     "ite_mean": ite[mask].mean()})

leaf_df = pd.DataFrame(leaf_tbl).sort_values("rd", ascending=False)
results["leaves"] = leaf_df
print("\n=== Iter11: Per-leaf treatment effect ===")
print(leaf_df.to_string())

# ============================================================
# Iteration 12: Test a clean subgroup hypothesis with a 3-way interaction.
# Build a logistic regression with t * m1 * m2 on the full data.
# ============================================================
print("\n=== Iter12: 3-way interaction model ===")
t = df[top_treatment].astype(float)
m1v = dichotomize(m1).astype(float)
m2v = dichotomize(m2).astype(float)
X = pd.DataFrame({
    "t": t, "m1": m1v, "m2": m2v,
    "t_m1": t * m1v, "t_m2": t * m2v, "m1_m2": m1v * m2v,
    "t_m1_m2": t * m1v * m2v,
})
fit3 = logreg(X, y)
print(fit3.summary())
results["3way"] = fit3

# ============================================================
# Iteration 13: Look at OTHER continuous modifiers for top_treatment beyond
# the top ones — specifically continuous features whose interaction effects
# point to therapy-resistance subgroups.
# ============================================================
cont_inters = inter_df[(inter_df["treatment"] == top_treatment) & inter_df["modifier"].isin(continuous_feats)]
cont_inters = cont_inters.sort_values("p_int")
print("\n=== Iter13: Continuous modifiers of top treatment ===")
print(cont_inters.head(15).to_string())

# ============================================================
# Iteration 14: For each of the other binary treatment-like features, test for
# heterogeneity in the leading modifier.
# ============================================================
print("\n=== Iter14: Other binary treatments tested for modification by leading modifier ===")
# Take top modifier for top_treatment
leading_mod = mods_for_t.iloc[0]["modifier"]
print(f"Leading modifier: {leading_mod}")
other_tx_rows = []
for tx in binary_feats:
    if tx == top_treatment or tx == leading_mod:
        continue
    p_int, b_int, b_t, _ = interaction_test(tx, leading_mod, y, df)
    # main effect of tx (univariate)
    main = bin_uni[bin_uni["feature"] == tx].iloc[0]["effect"]
    other_tx_rows.append({"tx": tx, "leading_mod": leading_mod, "p_int": p_int,
                          "beta_int": b_int, "main_effect": main})
other_tx_df = pd.DataFrame(other_tx_rows).sort_values("p_int")
print(other_tx_df.head(15).to_string())

# ============================================================
# Iteration 15: Validate top subgroup definition with a direct
# t-by-subgroup-indicator interaction.
# ============================================================
print("\n=== Iter15: Final subgroup definition test ===")
# Define subgroup S = {m1==v1 AND m2==v2 with largest treatment benefit}.
best_row = joint_df.dropna(subset=["rd"]).sort_values("rd", ascending=False).iloc[0]
v1, v2 = int(best_row["m1_val"]), int(best_row["m2_val"])
print(f"Best subgroup: {m1}={v1} AND {m2}={v2}")
S = ((g1 == v1) & (g2 == v2)).astype(float)
t = df[top_treatment].astype(float)
X = pd.DataFrame({"t": t, "S": S, "tS": t * S})
fitS = logreg(X, y)
print(fitS.summary())
results["subgroup_fit"] = fitS

# Also test subgroup with worst (most negative rate diff) and within-subgroup
# main effect of treatment.
worst_row = joint_df.dropna(subset=["rd"]).sort_values("rd").iloc[0]
print(f"Worst subgroup: {m1}={int(worst_row['m1_val'])} AND {m2}={int(worst_row['m2_val'])}")

# ============================================================
# Iteration 16: Multivariable adjustment of subgroup interaction.
# ============================================================
print("\n=== Iter16: Adjusted t*S interaction ===")
adj_cols = [c for c in non_const_feats if c not in {top_treatment, m1, m2}]
Xa = df[adj_cols].copy().astype(float)
for c in continuous_feats:
    if c in Xa.columns:
        Xa[c] = (Xa[c] - Xa[c].mean()) / Xa[c].std()
Xa["t"] = t
Xa["S"] = S
Xa["tS"] = t * S
fitSa = logreg(Xa, y)
print(fitSa.params.loc[["t", "S", "tS"]])
print(fitSa.pvalues.loc[["t", "S", "tS"]])
results["subgroup_adj_fit"] = fitSa

# ============================================================
# Iteration 17: Cross-validate by random split — does subgroup effect replicate?
# ============================================================
print("\n=== Iter17: Random-split replication ===")
rng = np.random.default_rng(42)
idx = np.arange(len(df))
rng.shuffle(idx)
half = len(df) // 2
splits = {"train": idx[:half], "test": idx[half:]}
rep_rows = []
for name, ix in splits.items():
    sub = df.iloc[ix]
    sub_S = S.iloc[ix]
    sub_t = t.iloc[ix]
    # within-S
    inS = sub[sub_S == 1]
    out = sub[sub_S == 0]
    rep_rows.append({"split": name, "where": "S=1",
                     "rate_t1": inS.loc[inS[top_treatment] == 1, "objective_response"].mean(),
                     "rate_t0": inS.loc[inS[top_treatment] == 0, "objective_response"].mean(),
                     "n": len(inS)})
    rep_rows.append({"split": name, "where": "S=0",
                     "rate_t1": out.loc[out[top_treatment] == 1, "objective_response"].mean(),
                     "rate_t0": out.loc[out[top_treatment] == 0, "objective_response"].mean(),
                     "n": len(out)})
rep_df = pd.DataFrame(rep_rows)
rep_df["rd"] = rep_df["rate_t1"] - rep_df["rate_t0"]
print(rep_df.to_string())
results["replication"] = rep_df

# ============================================================
# Iteration 18: Test top continuous feature interaction at a clean cutpoint.
# Find best cutpoint for the top continuous modifier.
# ============================================================
print("\n=== Iter18: Best cutpoint for top continuous modifier ===")
top_cont_mod = cont_inters.iloc[0]["modifier"]
print(f"Top continuous modifier: {top_cont_mod}")
quantiles = np.linspace(0.1, 0.9, 17)
cut_rows = []
for q in quantiles:
    c = df[top_cont_mod].quantile(q)
    above = (df[top_cont_mod] > c).astype(int)
    p_int, b_int, b_t, _ = interaction_test(top_treatment, top_cont_mod, y,
                                             pd.DataFrame({top_treatment: df[top_treatment],
                                                           top_cont_mod: above}))
    cut_rows.append({"q": q, "cut": c, "p_int": p_int, "beta_int": b_int})
cut_df = pd.DataFrame(cut_rows)
print(cut_df.to_string())
best_cut_row = cut_df.iloc[cut_df["p_int"].idxmin()]
print(f"Best q={best_cut_row['q']:.2f}, cut={best_cut_row['cut']:.3f}")

# ============================================================
# Iteration 19: Reassess subgroup with continuous modifier at best cutpoint.
# ============================================================
above = (df[top_cont_mod] > best_cut_row["cut"]).astype(int)
print(f"\n=== Iter19: 2x2 by {top_cont_mod}>{best_cut_row['cut']:.3f} and {m1}={v1} ===")
g_mA = above
g_mB = (g1 == v1).astype(int)
sub_rows = []
for vA in [0, 1]:
    for vB in [0, 1]:
        mask = (g_mA == vA) & (g_mB == vB)
        sub = df[mask]
        n1 = (sub[top_treatment] == 1).sum()
        n0 = (sub[top_treatment] == 0).sum()
        if n1 < 20 or n0 < 20:
            continue
        r1 = sub.loc[sub[top_treatment] == 1, "objective_response"].mean()
        r0 = sub.loc[sub[top_treatment] == 0, "objective_response"].mean()
        ct = pd.crosstab(sub[top_treatment], sub["objective_response"])
        chi2, pp, _, _ = stats.chi2_contingency(ct.values)
        sub_rows.append({"cont_above": vA, m1 + "_eq_" + str(v1): vB,
                         "n_t1": n1, "n_t0": n0, "rate_t1": r1, "rate_t0": r0,
                         "rd": r1 - r0, "p": pp})
sub_df_v2 = pd.DataFrame(sub_rows)
print(sub_df_v2.to_string())
results["subgroup_v2"] = sub_df_v2

# ============================================================
# Iteration 20: Composite 3-feature subgroup definition test
# ============================================================
# Take row in sub_df_v2 with biggest positive rd; combine all 3 conditions.
print("\n=== Iter20: Composite 3-feature subgroup ===")
best3 = sub_df_v2.dropna(subset=["rd"]).sort_values("rd", ascending=False).iloc[0]
print(best3)
cont_cond_value = int(best3["cont_above"])
m1_cond_value = int(best3[m1 + "_eq_" + str(v1)])
S3 = ((above == cont_cond_value) & ((g1 == v1).astype(int) == m1_cond_value)).astype(float)
print(f"Subgroup definition: ({top_cont_mod} > {best_cut_row['cut']:.3f}) is {bool(cont_cond_value)} "
      f"AND ({m1}=={v1}) is {bool(m1_cond_value)}")
print(f"Subgroup size: {S3.sum():.0f} of {len(S3)}")
X3 = pd.DataFrame({"t": t, "S": S3, "tS": t * S3})
fitS3 = logreg(X3, y)
print(fitS3.summary())
results["composite_fit"] = fitS3

# ============================================================
# Iteration 21: Treatment effect on outcome scale (rate differences) inside vs outside S3
# ============================================================
print("\n=== Iter21: Rate differences inside vs outside composite subgroup ===")
inS = df[S3 == 1]
out = df[S3 == 0]
for label, sub in [("inside_S", inS), ("outside_S", out)]:
    r1 = sub.loc[sub[top_treatment] == 1, "objective_response"].mean()
    r0 = sub.loc[sub[top_treatment] == 0, "objective_response"].mean()
    n1 = (sub[top_treatment] == 1).sum()
    n0 = (sub[top_treatment] == 0).sum()
    ct = pd.crosstab(sub[top_treatment], sub["objective_response"])
    chi2, pp, _, _ = stats.chi2_contingency(ct.values)
    print(f"{label}: n_t1={n1}, n_t0={n0}, rate_t1={r1:.4f}, rate_t0={r0:.4f}, "
          f"rd={r1-r0:.4f}, chi2_p={pp:.3e}")

# ============================================================
# Iteration 22: Propose third treatment heterogeneity for second-best treatment
# ============================================================
print("\n=== Iter22: Heterogeneity for second-best treatment ===")
top_t2 = treatment_candidates[1]
print(f"Treatment: {top_t2}")
mods_for_t2 = inter_df[inter_df["treatment"] == top_t2].sort_values("p_int").head(5)
print(mods_for_t2.to_string())

# ============================================================
# Iteration 23: For each binary feature, run adjusted main effect (controlling
# for everything else) — sanity check vs univariate.
# ============================================================
print("\n=== Iter23: Adjusted vs univariate main effects of binary features ===")
adj_results = []
for f in binary_feats:
    uni = bin_uni[bin_uni["feature"] == f].iloc[0]
    mv_row = mv_summary[mv_summary["feature"] == f]
    if len(mv_row):
        adj_results.append({"feature": f, "uni_beta": uni["effect"], "uni_p": uni["p"],
                            "adj_beta": mv_row.iloc[0]["beta"], "adj_p": mv_row.iloc[0]["p"]})
adj_df = pd.DataFrame(adj_results)
print(adj_df.to_string())

# ============================================================
# Iteration 24: Check whether the composite subgroup's heterogeneity holds
# under multivariable adjustment.
# ============================================================
print("\n=== Iter24: Composite-subgroup interaction adjusted ===")
adj_cols24 = [c for c in non_const_feats if c not in {top_treatment, m1, m2, top_cont_mod}]
X24 = df[adj_cols24].copy().astype(float)
for c in continuous_feats:
    if c in X24.columns:
        X24[c] = (X24[c] - X24[c].mean()) / X24[c].std()
X24["t"] = t
X24["S3"] = S3
X24["tS3"] = t * S3
fit24 = logreg(X24, y)
print(fit24.params.loc[["t", "S3", "tS3"]])
print(fit24.pvalues.loc[["t", "S3", "tS3"]])

# ============================================================
# Iteration 25: Final summary stats — write everything out.
# ============================================================
out = {
    "univariate": uni_df.to_dict(orient="records"),
    "multivariable": mv_summary.to_dict(orient="records"),
    "interactions_top": inter_df.head(50).to_dict(orient="records"),
    "stratified": strat_df.to_dict(orient="records"),
    "joint": joint_df.to_dict(orient="records"),
    "leaves": leaf_df.to_dict(orient="records"),
    "subgroup_v2": sub_df_v2.to_dict(orient="records"),
    "replication": rep_df.to_dict(orient="records"),
    "top_treatment": top_treatment,
    "treatment_candidates": treatment_candidates,
    "m1": m1, "m2": m2, "top_cont_mod": top_cont_mod,
    "best_cut": float(best_cut_row["cut"]),
    "best_subgroup_v1_v2": [int(v1), int(v2)],
    "composite_S3_fit_t": float(fitS3.params["t"]),
    "composite_S3_fit_tS_p": float(fitS3.pvalues["tS"]),
    "composite_S3_fit_tS": float(fitS3.params["tS"]),
    "subgroup_S2_fit_t": float(fitS.params["t"]),
    "subgroup_S2_fit_tS": float(fitS.params["tS"]),
    "subgroup_S2_fit_tS_p": float(fitS.pvalues["tS"]),
    "subgroup_adj_t": float(fitSa.params["t"]),
    "subgroup_adj_tS": float(fitSa.params["tS"]),
    "subgroup_adj_tS_p": float(fitSa.pvalues["tS"]),
    "composite_adj_t": float(fit24.params["t"]),
    "composite_adj_tS3": float(fit24.params["tS3"]),
    "composite_adj_tS3_p": float(fit24.pvalues["tS3"]),
    "best3_inside_outside": None,
}

with open("results.json", "w") as f:
    json.dump(out, f, default=str, indent=2)

print("\n\nResults saved to results.json")
