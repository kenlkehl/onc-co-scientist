"""25-iteration analysis of ds001_crc -> pfs_months.

Emits transcript.json (schema-conformant) and analysis_summary.txt.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DF = pd.read_parquet("dataset.parquet")
Y = DF["pfs_months"].astype(float)

# ---------- helpers ----------
def ttest(col, val1=1, val0=0):
    a = Y[DF[col] == val1].values
    b = Y[DF[col] == val0].values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"n1": int(len(a)), "n0": int(len(b)),
            "mean1": float(a.mean()), "mean0": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "p": float(p)}

def pearson(col):
    r, p = stats.pearsonr(DF[col].astype(float), Y)
    return {"r": float(r), "p": float(p),
            "slope": float(np.polyfit(DF[col].astype(float), Y, 1)[0])}

def anova_groups(col):
    levels = sorted(DF[col].unique().tolist())
    groups = [Y[DF[col] == v].values for v in levels]
    F, p = stats.f_oneway(*groups)
    means = {str(v): float(g.mean()) for v, g in zip(levels, groups)}
    ns = {str(v): int(len(g)) for v, g in zip(levels, groups)}
    return {"F": float(F), "p": float(p), "means": means, "ns": ns,
            "diff_max_min": float(max(means.values()) - min(means.values()))}

def ols_summary(formula, data=None):
    d = data if data is not None else DF
    m = smf.ols(formula, data=d).fit()
    return m

def interaction_test(formula_main, formula_int, data=None):
    """LRT-equivalent F-test by comparing nested OLS models."""
    d = data if data is not None else DF
    m0 = smf.ols(formula_main, data=d).fit()
    m1 = smf.ols(formula_int, data=d).fit()
    # F-test for added interaction terms
    df_diff = int(m0.df_resid - m1.df_resid)
    F = ((m0.ssr - m1.ssr) / df_diff) / (m1.ssr / m1.df_resid)
    p = 1 - stats.f.cdf(F, df_diff, m1.df_resid)
    return {"F": float(F), "df_diff": df_diff, "df_resid": float(m1.df_resid),
            "p": float(p), "params": dict(m1.params), "main_R2": float(m0.rsquared),
            "int_R2": float(m1.rsquared)}

# ---------- Iterations container ----------
ITERS = []

def add(idx, hyps, analyses):
    ITERS.append({"index": idx, "proposed_hypotheses": hyps, "analyses": analyses})

# ============================================================
# Iteration 1: Outcome distribution & key main-effect screening
# ============================================================
hyps = [
    {"id": "h1", "kind": "novel",
     "text": "feature_078 (continuous, range 30-90, mean 65) is positively associated with pfs_months (Pearson r > 0)."},
    {"id": "h2", "kind": "novel",
     "text": "feature_057 (3-level ordinal, values 0/1/2) is associated with pfs_months: higher levels predict shorter pfs_months (linear trend, slope < 0)."},
    {"id": "h3", "kind": "novel",
     "text": "feature_051 (binary) is associated with shorter pfs_months when present (mean diff < 0 versus absent)."},
]
a = []
r = pearson("feature_078")
a.append({"hypothesis_ids": ["h1"], "code": "stats.pearsonr(df.feature_078, df.pfs_months)",
          "result_summary": f"Pearson r={r['r']:+.3f}, slope={r['slope']:+.4f} months per unit feature_078, p={r['p']:.2e}.",
          "p_value": r["p"], "effect_estimate": r["slope"], "significant": r["p"] < 0.05})

# linear trend on feature_057 (ordinal 0/1/2)
m = smf.ols("pfs_months ~ feature_057", data=DF).fit()
slope = float(m.params["feature_057"]); pp = float(m.pvalues["feature_057"])
a.append({"hypothesis_ids": ["h2"], "code": "ols('pfs_months ~ feature_057')",
          "result_summary": f"Linear trend across feature_057 levels (0,1,2): slope={slope:+.3f} months/level, p={pp:.2e}. Group means: {anova_groups('feature_057')['means']}.",
          "p_value": pp, "effect_estimate": slope, "significant": pp < 0.05})

t = ttest("feature_051")
a.append({"hypothesis_ids": ["h3"], "code": "ttest_ind(pfs[feature_051==1], pfs[feature_051==0])",
          "result_summary": f"Mean pfs_months: feature_051=1: {t['mean1']:.3f} (n={t['n1']}); =0: {t['mean0']:.3f} (n={t['n0']}); diff={t['diff']:+.3f}, p={t['p']:.2e}.",
          "p_value": t["p"], "effect_estimate": t["diff"], "significant": t["p"] < 0.05})
add(1, hyps, a)

# ============================================================
# Iteration 2: More binary main effects
# ============================================================
hyps = [
    {"id": "h4", "kind": "novel",
     "text": "feature_038 (binary) is associated with longer pfs_months when present (mean diff > 0 versus absent)."},
    {"id": "h5", "kind": "novel",
     "text": "feature_013 (binary) is associated with shorter pfs_months when present (mean diff < 0)."},
    {"id": "h6", "kind": "novel",
     "text": "feature_043 (binary) is associated with shorter pfs_months when present (mean diff < 0)."},
    {"id": "h7", "kind": "novel",
     "text": "feature_109 (binary) is associated with shorter pfs_months when present (mean diff < 0)."},
    {"id": "h8", "kind": "novel",
     "text": "feature_067 (binary) is associated with longer pfs_months when present (mean diff > 0)."},
]
a = []
for hid, col, expected in [("h4","feature_038","+"),("h5","feature_013","-"),
                           ("h6","feature_043","-"),("h7","feature_109","-"),("h8","feature_067","+")]:
    t = ttest(col)
    a.append({"hypothesis_ids": [hid],
              "code": f"ttest_ind(pfs[{col}==1], pfs[{col}==0])",
              "result_summary": f"{col}=1 mean={t['mean1']:.3f} (n={t['n1']}); =0 mean={t['mean0']:.3f} (n={t['n0']}); diff={t['diff']:+.3f}, p={t['p']:.2e}.",
              "p_value": t["p"], "effect_estimate": t["diff"], "significant": t["p"] < 0.05})
add(2, hyps, a)

# ============================================================
# Iteration 3: Continuous feature correlations
# ============================================================
hyps = [
    {"id": "h9", "kind": "novel",
     "text": "feature_099 (continuous, range 0-24.6) is negatively correlated with pfs_months (r < 0)."},
    {"id": "h10", "kind": "novel",
     "text": "feature_092 (continuous, mean 3.8, range 1.5-5.5) is positively correlated with pfs_months (r > 0)."},
    {"id": "h11", "kind": "novel",
     "text": "feature_009 (continuous, range 0.02-777) is negatively correlated with pfs_months (r < 0)."},
]
a = []
for hid, col in [("h9","feature_099"),("h10","feature_092"),("h11","feature_009")]:
    r = pearson(col)
    a.append({"hypothesis_ids": [hid],
              "code": f"stats.pearsonr(df.{col}, df.pfs_months)",
              "result_summary": f"{col}: Pearson r={r['r']:+.4f}, slope={r['slope']:+.4f} months/unit, p={r['p']:.2e}.",
              "p_value": r["p"], "effect_estimate": r["slope"], "significant": r["p"] < 0.05})
add(3, hyps, a)

# ============================================================
# Iteration 4: Race / insurance / multi-cat main effects
# ============================================================
hyps = [
    {"id": "h12", "kind": "novel",
     "text": "Mean pfs_months differs across feature_064 race categories (white/hispanic/asian/black/other) (ANOVA p < 0.05)."},
    {"id": "h13", "kind": "novel",
     "text": "Mean pfs_months differs across feature_018 insurance categories (medicare/private/uninsured/medicaid) (ANOVA p < 0.05)."},
    {"id": "h14", "kind": "novel",
     "text": "feature_025 (5-level ordinal) shows a difference in mean pfs_months across levels (ANOVA p < 0.05)."},
]
a = []
for hid, col in [("h12","feature_064"),("h13","feature_018"),("h14","feature_025")]:
    g = anova_groups(col)
    a.append({"hypothesis_ids": [hid],
              "code": f"f_oneway over levels of {col}",
              "result_summary": f"{col}: F={g['F']:.2f}, p={g['p']:.2e}; means={g['means']}; range across groups = {g['diff_max_min']:.3f}.",
              "p_value": g["p"], "effect_estimate": g["diff_max_min"], "significant": g["p"] < 0.05})
add(4, hyps, a)

# ============================================================
# Iteration 5: Quartile / dose-response of feature_078
# ============================================================
hyps = [
    {"id": "h15", "kind": "novel",
     "text": "Mean pfs_months increases monotonically across quartiles of feature_078 (Q4 - Q1 > 0)."},
]
a = []
DF["_f078q"] = pd.qcut(DF["feature_078"], 4, labels=["Q1","Q2","Q3","Q4"])
g = DF.groupby("_f078q", observed=True)["pfs_months"].agg(["mean","count"])
diff = float(g.loc["Q4","mean"] - g.loc["Q1","mean"])
F, p = stats.f_oneway(*[Y[DF["_f078q"]==q].values for q in ["Q1","Q2","Q3","Q4"]])
a.append({"hypothesis_ids": ["h15"],
          "code": "qcut(feature_078,4); ANOVA + Q4-Q1 difference",
          "result_summary": f"Quartile means feature_078: {dict(g['mean'].round(3))}; n={dict(g['count'])}. Q4-Q1 diff={diff:+.3f} months; ANOVA F={F:.2f}, p={p:.2e}.",
          "p_value": float(p), "effect_estimate": diff, "significant": p < 0.05})
add(5, hyps, a)

# ============================================================
# Iteration 6: Adjusted effect of feature_051 controlling for feature_078
# ============================================================
hyps = [
    {"id": "h16", "kind": "refined",
     "text": "feature_051's negative association with pfs_months persists after adjusting for feature_078 (continuous), with adjusted coefficient < 0."},
]
a = []
m = smf.ols("pfs_months ~ feature_051 + feature_078", data=DF).fit()
b = float(m.params["feature_051"]); pp = float(m.pvalues["feature_051"])
a.append({"hypothesis_ids": ["h16"],
          "code": "ols('pfs_months ~ feature_051 + feature_078')",
          "result_summary": f"Adjusted for feature_078: feature_051 coef={b:+.3f} months, p={pp:.2e}. Model R^2={m.rsquared:.3f}.",
          "p_value": pp, "effect_estimate": b, "significant": pp < 0.05})
add(6, hyps, a)

# ============================================================
# Iteration 7: Adjusted effect of feature_038 (the positive binary)
# ============================================================
hyps = [
    {"id": "h17", "kind": "refined",
     "text": "feature_038's positive association with pfs_months persists after adjusting for feature_078, feature_057, and feature_051 (adjusted coef > 0)."},
]
a = []
m = smf.ols("pfs_months ~ feature_038 + feature_078 + feature_057 + feature_051", data=DF).fit()
b = float(m.params["feature_038"]); pp = float(m.pvalues["feature_038"])
a.append({"hypothesis_ids": ["h17"],
          "code": "ols('pfs_months ~ feature_038 + feature_078 + feature_057 + feature_051')",
          "result_summary": f"Adjusted: feature_038 coef={b:+.3f} months, p={pp:.2e}. Full coef table: { {k: round(float(v),4) for k,v in m.params.items()} }. R^2={m.rsquared:.3f}.",
          "p_value": pp, "effect_estimate": b, "significant": pp < 0.05})
add(7, hyps, a)

# ============================================================
# Iteration 8: Interaction feature_038 x feature_057
# ============================================================
hyps = [
    {"id": "h18", "kind": "novel",
     "text": "The benefit of feature_038 (effect on pfs_months) varies across feature_057 levels (interaction p < 0.05); we expect the largest absolute benefit at the most-advanced level (feature_057=2)."},
]
a = []
res = interaction_test("pfs_months ~ feature_038 + C(feature_057) + feature_078",
                       "pfs_months ~ feature_038 * C(feature_057) + feature_078")
# Stratified means
strata = []
for v in [0,1,2]:
    sub = DF[DF["feature_057"]==v]
    m1 = sub.loc[sub["feature_038"]==1, "pfs_months"].mean()
    m0 = sub.loc[sub["feature_038"]==0, "pfs_months"].mean()
    strata.append((v, float(m1-m0), int((sub["feature_038"]==1).sum()), int((sub["feature_038"]==0).sum())))
sign_eff = strata[2][1]
a.append({"hypothesis_ids": ["h18"],
          "code": "F-test ols(pfs ~ f038 + C(f057) + f078) vs ols(pfs ~ f038*C(f057) + f078)",
          "result_summary": f"Interaction F={res['F']:.2f} on df={res['df_diff']}, p={res['p']:.2e}. Stratified feature_038 effect (m1-m0): " + "; ".join(f"f057={v}: diff={d:+.3f} (n1={n1},n0={n0})" for v,d,n1,n0 in strata) + ".",
          "p_value": res["p"], "effect_estimate": sign_eff, "significant": res["p"] < 0.05})
add(8, hyps, a)

# ============================================================
# Iteration 9: Interaction feature_051 x feature_057
# ============================================================
hyps = [
    {"id": "h19", "kind": "novel",
     "text": "The negative effect of feature_051 on pfs_months differs in magnitude across feature_057 levels (interaction p < 0.05); we expect a more negative absolute effect at higher feature_057 levels."},
]
a = []
res = interaction_test("pfs_months ~ feature_051 + C(feature_057) + feature_078",
                       "pfs_months ~ feature_051 * C(feature_057) + feature_078")
strata = []
for v in [0,1,2]:
    sub = DF[DF["feature_057"]==v]
    m1 = sub.loc[sub["feature_051"]==1, "pfs_months"].mean()
    m0 = sub.loc[sub["feature_051"]==0, "pfs_months"].mean()
    strata.append((v, float(m1-m0), int((sub["feature_051"]==1).sum()), int((sub["feature_051"]==0).sum())))
sign_eff = strata[2][1]
a.append({"hypothesis_ids": ["h19"],
          "code": "F-test ols(pfs ~ f051 + C(f057) + f078) vs ols(pfs ~ f051*C(f057) + f078)",
          "result_summary": f"Interaction F={res['F']:.2f} on df={res['df_diff']}, p={res['p']:.2e}. Stratified diff: " + "; ".join(f"f057={v}: {d:+.3f} (n1={n1},n0={n0})" for v,d,n1,n0 in strata) + ".",
          "p_value": res["p"], "effect_estimate": sign_eff, "significant": res["p"] < 0.05})
add(9, hyps, a)

# ============================================================
# Iteration 10: Interaction feature_038 x feature_051 (treatment by comorbidity-like)
# ============================================================
hyps = [
    {"id": "h20", "kind": "novel",
     "text": "The benefit of feature_038 on pfs_months is attenuated in patients with feature_051=1 versus feature_051=0 (positive feature_038*feature_051 interaction term magnitude)."},
]
a = []
res = interaction_test("pfs_months ~ feature_038 + feature_051 + feature_078 + C(feature_057)",
                       "pfs_months ~ feature_038 * feature_051 + feature_078 + C(feature_057)")
# Stratified
m_a = DF.loc[(DF.feature_051==0) & (DF.feature_038==1),"pfs_months"].mean()
m_b = DF.loc[(DF.feature_051==0) & (DF.feature_038==0),"pfs_months"].mean()
m_c = DF.loc[(DF.feature_051==1) & (DF.feature_038==1),"pfs_months"].mean()
m_d = DF.loc[(DF.feature_051==1) & (DF.feature_038==0),"pfs_months"].mean()
diff_when0 = float(m_a - m_b); diff_when1 = float(m_c - m_d)
a.append({"hypothesis_ids": ["h20"],
          "code": "F-test for f038:f051 interaction adjusting for f078, C(f057)",
          "result_summary": f"Interaction F={res['F']:.2f} df={res['df_diff']}, p={res['p']:.2e}. f038 effect when f051=0: {diff_when0:+.3f}; when f051=1: {diff_when1:+.3f}.",
          "p_value": res["p"], "effect_estimate": float(diff_when1 - diff_when0), "significant": res["p"] < 0.05})
add(10, hyps, a)

# ============================================================
# Iteration 11: Race subgroup heterogeneity for feature_038
# ============================================================
hyps = [
    {"id": "h21", "kind": "novel",
     "text": "The effect of feature_038 on pfs_months differs across feature_064 race categories (interaction p < 0.05)."},
]
a = []
res = interaction_test("pfs_months ~ feature_038 + C(feature_064) + feature_078 + C(feature_057)",
                       "pfs_months ~ feature_038 * C(feature_064) + feature_078 + C(feature_057)")
strata = []
for v in DF["feature_064"].unique():
    sub = DF[DF["feature_064"]==v]
    m1 = sub.loc[sub["feature_038"]==1,"pfs_months"].mean()
    m0 = sub.loc[sub["feature_038"]==0,"pfs_months"].mean()
    strata.append((v, float(m1-m0), int((sub["feature_038"]==1).sum())))
diffs = [s[1] for s in strata]
a.append({"hypothesis_ids": ["h21"],
          "code": "F-test for f038:C(f064) interaction adjusted for f078,C(f057)",
          "result_summary": f"Interaction F={res['F']:.2f} df={res['df_diff']}, p={res['p']:.2e}. Stratified f038 diff: " + "; ".join(f"{v}: {d:+.3f} (n1={n})" for v,d,n in strata) + f". Range: {max(diffs)-min(diffs):.3f}.",
          "p_value": res["p"], "effect_estimate": float(max(diffs)-min(diffs)), "significant": res["p"] < 0.05})
add(11, hyps, a)

# ============================================================
# Iteration 12: Insurance subgroup heterogeneity for feature_038
# ============================================================
hyps = [
    {"id": "h22", "kind": "novel",
     "text": "The effect of feature_038 on pfs_months differs across feature_018 insurance categories (interaction p < 0.05)."},
]
a = []
res = interaction_test("pfs_months ~ feature_038 + C(feature_018) + feature_078 + C(feature_057)",
                       "pfs_months ~ feature_038 * C(feature_018) + feature_078 + C(feature_057)")
strata = []
for v in DF["feature_018"].unique():
    sub = DF[DF["feature_018"]==v]
    m1 = sub.loc[sub["feature_038"]==1,"pfs_months"].mean()
    m0 = sub.loc[sub["feature_038"]==0,"pfs_months"].mean()
    strata.append((v, float(m1-m0), int((sub["feature_038"]==1).sum())))
diffs = [s[1] for s in strata]
a.append({"hypothesis_ids": ["h22"],
          "code": "F-test for f038:C(f018) interaction adjusted for f078,C(f057)",
          "result_summary": f"Interaction F={res['F']:.2f} df={res['df_diff']}, p={res['p']:.2e}. f038 diff by insurance: " + "; ".join(f"{v}: {d:+.3f} (n1={n})" for v,d,n in strata) + f". Range: {max(diffs)-min(diffs):.3f}.",
          "p_value": res["p"], "effect_estimate": float(max(diffs)-min(diffs)), "significant": res["p"] < 0.05})
add(12, hyps, a)

# ============================================================
# Iteration 13: Continuous biomarker feature_099 effect modification
# ============================================================
hyps = [
    {"id": "h23", "kind": "novel",
     "text": "The negative association between feature_099 and pfs_months is steeper (more negative slope) within higher feature_057 stage levels (interaction p < 0.05)."},
]
a = []
res = interaction_test("pfs_months ~ feature_099 + C(feature_057) + feature_078",
                       "pfs_months ~ feature_099 * C(feature_057) + feature_078")
slopes = []
for v in [0,1,2]:
    sub = DF[DF["feature_057"]==v]
    s = float(np.polyfit(sub["feature_099"], sub["pfs_months"],1)[0])
    slopes.append((v, s, int(len(sub))))
a.append({"hypothesis_ids": ["h23"],
          "code": "F-test for f099:C(f057) interaction adjusted for f078",
          "result_summary": f"Interaction F={res['F']:.2f} df={res['df_diff']}, p={res['p']:.2e}. f099-on-pfs slope by f057: " + "; ".join(f"f057={v}: {s:+.4f} (n={n})" for v,s,n in slopes) + ".",
          "p_value": res["p"], "effect_estimate": float(slopes[2][1] - slopes[0][1]), "significant": res["p"] < 0.05})
add(13, hyps, a)

# ============================================================
# Iteration 14: feature_092 (presumed albumin-like) interaction with feature_038
# ============================================================
hyps = [
    {"id": "h24", "kind": "novel",
     "text": "feature_038 benefit on pfs_months is larger in patients with higher feature_092 values (positive feature_038*feature_092 interaction)."},
]
a = []
res = interaction_test("pfs_months ~ feature_038 + feature_092 + feature_078 + C(feature_057)",
                       "pfs_months ~ feature_038 * feature_092 + feature_078 + C(feature_057)")
# Median split
med = DF["feature_092"].median()
hi = DF[DF["feature_092"] >= med]
lo = DF[DF["feature_092"] < med]
diff_hi = float(hi.loc[hi.feature_038==1,"pfs_months"].mean() - hi.loc[hi.feature_038==0,"pfs_months"].mean())
diff_lo = float(lo.loc[lo.feature_038==1,"pfs_months"].mean() - lo.loc[lo.feature_038==0,"pfs_months"].mean())
a.append({"hypothesis_ids": ["h24"],
          "code": "F-test f038:f092 interaction; median-split diff",
          "result_summary": f"Interaction F={res['F']:.2f} df={res['df_diff']}, p={res['p']:.2e}. f038 effect on pfs_months: in high-f092 (>= median {med:.2f}) {diff_hi:+.3f}; in low-f092 {diff_lo:+.3f}.",
          "p_value": res["p"], "effect_estimate": float(diff_hi - diff_lo), "significant": res["p"] < 0.05})
add(14, hyps, a)

# ============================================================
# Iteration 15: feature_078 quartile x feature_038 (age x treatment-like)
# ============================================================
hyps = [
    {"id": "h25", "kind": "novel",
     "text": "feature_038 benefit on pfs_months differs across feature_078 quartiles (interaction p < 0.05)."},
]
a = []
DF["_f078q"] = pd.qcut(DF["feature_078"], 4, labels=False)
res = interaction_test("pfs_months ~ feature_038 + C(_f078q) + C(feature_057)",
                       "pfs_months ~ feature_038 * C(_f078q) + C(feature_057)", data=DF)
strata = []
for q in [0,1,2,3]:
    sub = DF[DF["_f078q"]==q]
    d = float(sub.loc[sub.feature_038==1,"pfs_months"].mean() - sub.loc[sub.feature_038==0,"pfs_months"].mean())
    strata.append((q, d, int(sub.shape[0])))
diffs = [s[1] for s in strata]
a.append({"hypothesis_ids": ["h25"],
          "code": "F-test f038:C(_f078q) adjusted for C(f057)",
          "result_summary": f"Interaction F={res['F']:.2f} df={res['df_diff']}, p={res['p']:.2e}. f038 effect by f078 quartile (Q1..Q4): " + "; ".join(f"Q{q+1}: {d:+.3f} (n={n})" for q,d,n in strata) + ".",
          "p_value": res["p"], "effect_estimate": float(diffs[3]-diffs[0]), "significant": res["p"] < 0.05})
add(15, hyps, a)

# ============================================================
# Iteration 16: Sex / binary feature_007 effect
# ============================================================
hyps = [
    {"id": "h26", "kind": "novel",
     "text": "feature_007 (binary, ~50/50 split, plausibly sex) is associated with pfs_months (mean diff != 0)."},
    {"id": "h27", "kind": "novel",
     "text": "The effect of feature_038 on pfs_months differs by feature_007 (interaction p < 0.05)."},
]
a = []
t = ttest("feature_007")
a.append({"hypothesis_ids": ["h26"],
          "code": "ttest_ind(pfs[feature_007==1], pfs[feature_007==0])",
          "result_summary": f"feature_007=1 mean={t['mean1']:.3f} (n={t['n1']}); =0 mean={t['mean0']:.3f} (n={t['n0']}); diff={t['diff']:+.3f}, p={t['p']:.2e}.",
          "p_value": t["p"], "effect_estimate": t["diff"], "significant": t["p"] < 0.05})
res = interaction_test("pfs_months ~ feature_038 + feature_007 + feature_078 + C(feature_057)",
                       "pfs_months ~ feature_038 * feature_007 + feature_078 + C(feature_057)")
diff_s1 = float(DF.loc[(DF.feature_007==1)&(DF.feature_038==1),"pfs_months"].mean()
                - DF.loc[(DF.feature_007==1)&(DF.feature_038==0),"pfs_months"].mean())
diff_s0 = float(DF.loc[(DF.feature_007==0)&(DF.feature_038==1),"pfs_months"].mean()
                - DF.loc[(DF.feature_007==0)&(DF.feature_038==0),"pfs_months"].mean())
a.append({"hypothesis_ids": ["h27"],
          "code": "F-test f038:f007 interaction adjusted for f078,C(f057)",
          "result_summary": f"Interaction F={res['F']:.2f} df={res['df_diff']}, p={res['p']:.2e}. f038 effect when f007=1: {diff_s1:+.3f}; when f007=0: {diff_s0:+.3f}.",
          "p_value": res["p"], "effect_estimate": float(diff_s1 - diff_s0), "significant": res["p"] < 0.05})
add(16, hyps, a)

# ============================================================
# Iteration 17: feature_013 / feature_043 / feature_109 effect after adjustment
# ============================================================
hyps = [
    {"id": "h28", "kind": "refined",
     "text": "feature_013 retains a negative association with pfs_months after adjusting for feature_078, feature_057, feature_051 (adjusted coef < 0)."},
    {"id": "h29", "kind": "refined",
     "text": "feature_043 retains a negative association with pfs_months after adjusting for feature_078, feature_057, feature_051 (adjusted coef < 0)."},
    {"id": "h30", "kind": "refined",
     "text": "feature_109 retains a negative association with pfs_months after adjusting for feature_078, feature_057, feature_051 (adjusted coef < 0)."},
]
a = []
m = smf.ols("pfs_months ~ feature_013 + feature_043 + feature_109 + feature_078 + C(feature_057) + feature_051", data=DF).fit()
for hid, col in [("h28","feature_013"),("h29","feature_043"),("h30","feature_109")]:
    b = float(m.params[col]); pp = float(m.pvalues[col])
    a.append({"hypothesis_ids": [hid],
              "code": "ols('pfs ~ f013 + f043 + f109 + f078 + C(f057) + f051')",
              "result_summary": f"{col} adj coef={b:+.3f} months, p={pp:.2e}. Model R^2={m.rsquared:.3f}.",
              "p_value": pp, "effect_estimate": b, "significant": pp < 0.05})
add(17, hyps, a)

# ============================================================
# Iteration 18: feature_067 adjusted (positive binary)
# ============================================================
hyps = [
    {"id": "h31", "kind": "refined",
     "text": "feature_067's positive association with pfs_months persists after adjustment for feature_078, feature_057, and feature_051 (adjusted coef > 0)."},
]
a = []
m = smf.ols("pfs_months ~ feature_067 + feature_078 + C(feature_057) + feature_051", data=DF).fit()
b = float(m.params["feature_067"]); pp = float(m.pvalues["feature_067"])
a.append({"hypothesis_ids": ["h31"],
          "code": "ols('pfs ~ f067 + f078 + C(f057) + f051')",
          "result_summary": f"feature_067 adj coef={b:+.3f} months, p={pp:.2e}. R^2={m.rsquared:.3f}.",
          "p_value": pp, "effect_estimate": b, "significant": pp < 0.05})
add(18, hyps, a)

# ============================================================
# Iteration 19: Three-way interaction: f038 x f057 x f051
# ============================================================
hyps = [
    {"id": "h32", "kind": "novel",
     "text": "The interaction between feature_038 and feature_057 on pfs_months is itself modified by feature_051 (three-way interaction p < 0.05)."},
]
a = []
res = interaction_test("pfs_months ~ feature_038 * C(feature_057) + feature_038 * feature_051 + C(feature_057) * feature_051 + feature_078",
                       "pfs_months ~ feature_038 * C(feature_057) * feature_051 + feature_078")
a.append({"hypothesis_ids": ["h32"],
          "code": "F-test 3-way f038*C(f057)*f051 vs all 2-way",
          "result_summary": f"Three-way interaction F={res['F']:.2f} df={res['df_diff']}, p={res['p']:.2e}.",
          "p_value": res["p"], "effect_estimate": float(res['F']), "significant": res["p"] < 0.05})
add(19, hyps, a)

# ============================================================
# Iteration 20: Combined biomarker signature: f099 + f092 jointly with f057
# ============================================================
hyps = [
    {"id": "h33", "kind": "novel",
     "text": "Combined model with feature_099 (negative) and feature_092 (positive) and feature_057 explains substantially more variance in pfs_months than feature_057 alone (Delta R^2 > 0.01)."},
]
a = []
m_base = smf.ols("pfs_months ~ C(feature_057)", data=DF).fit()
m_full = smf.ols("pfs_months ~ feature_099 + feature_092 + C(feature_057)", data=DF).fit()
dR2 = float(m_full.rsquared - m_base.rsquared)
df_diff = m_base.df_resid - m_full.df_resid
F = ((m_base.ssr - m_full.ssr) / df_diff) / (m_full.ssr / m_full.df_resid)
p = 1 - stats.f.cdf(F, df_diff, m_full.df_resid)
a.append({"hypothesis_ids": ["h33"],
          "code": "ols base C(f057) vs +f099+f092; F-test",
          "result_summary": f"R^2 base={m_base.rsquared:.4f}, full={m_full.rsquared:.4f}, ΔR^2={dR2:.4f}. F={F:.2f}, p={p:.2e}. Coefs: f099={float(m_full.params['feature_099']):+.4f} (p={float(m_full.pvalues['feature_099']):.2e}); f092={float(m_full.params['feature_092']):+.4f} (p={float(m_full.pvalues['feature_092']):.2e}).",
          "p_value": float(p), "effect_estimate": dR2, "significant": p < 0.05})
add(20, hyps, a)

# ============================================================
# Iteration 21: Feature_078 squared term (nonlinear)
# ============================================================
hyps = [
    {"id": "h34", "kind": "novel",
     "text": "There is a nonlinear (quadratic) component to the feature_078 - pfs_months relationship beyond the linear term (quadratic term coefficient p < 0.05)."},
]
a = []
DF["_f078sq"] = DF["feature_078"]**2
m_lin = smf.ols("pfs_months ~ feature_078", data=DF).fit()
m_q = smf.ols("pfs_months ~ feature_078 + _f078sq", data=DF).fit()
b_q = float(m_q.params["_f078sq"]); p_q = float(m_q.pvalues["_f078sq"])
a.append({"hypothesis_ids": ["h34"],
          "code": "ols('pfs ~ f078 + f078^2'); test quadratic term",
          "result_summary": f"Quadratic term coef={b_q:+.6f}, p={p_q:.2e}. Linear-only R^2={m_lin.rsquared:.4f}; quadratic R^2={m_q.rsquared:.4f}.",
          "p_value": p_q, "effect_estimate": b_q, "significant": p_q < 0.05})
add(21, hyps, a)

# ============================================================
# Iteration 22: Are there other binary features with strong adjusted effects?
# ============================================================
hyps = [
    {"id": "h35", "kind": "novel",
     "text": "Among the remaining binary features, at least one has an adjusted effect on pfs_months whose magnitude exceeds 0.05 months and is statistically significant after Bonferroni correction across the binary feature set."},
]
a = []
binary_cols = [c for c in DF.columns if c not in ("patient_id","pfs_months") and DF[c].dtype != "object" and DF[c].nunique()==2]
already = {"feature_051","feature_038","feature_013","feature_043","feature_109","feature_067","feature_007"}
remaining = [c for c in binary_cols if c not in already]
adj_results = []
covariates = "feature_078 + C(feature_057) + feature_051"
for c in remaining:
    m = smf.ols(f"pfs_months ~ {c} + {covariates}", data=DF).fit()
    adj_results.append((c, float(m.params[c]), float(m.pvalues[c])))
adj_results.sort(key=lambda x: x[2])
bon = 0.05 / len(remaining)
sig_after = [r for r in adj_results if r[2] < bon]
top = adj_results[:5]
top_str = "; ".join(f"{c}: b={b:+.3f} p={p:.2e}" for c,b,p in top)
strongest = adj_results[0]
a.append({"hypothesis_ids": ["h35"],
          "code": "loop ols('pfs ~ <feat> + f078 + C(f057) + f051') for each remaining binary; Bonferroni α=0.05/k",
          "result_summary": f"Tested {len(remaining)} remaining binaries. Bonferroni α={bon:.2e}. {len(sig_after)} survived. Top 5 by p: {top_str}.",
          "p_value": strongest[2], "effect_estimate": strongest[1], "significant": strongest[2] < bon})
add(22, hyps, a)

# ============================================================
# Iteration 23: feature_057 stage interaction with feature_078 (age-stage)
# ============================================================
hyps = [
    {"id": "h36", "kind": "novel",
     "text": "The slope of pfs_months on feature_078 differs across feature_057 levels (interaction p < 0.05)."},
]
a = []
res = interaction_test("pfs_months ~ feature_078 + C(feature_057)",
                       "pfs_months ~ feature_078 * C(feature_057)")
slopes = []
for v in [0,1,2]:
    sub = DF[DF["feature_057"]==v]
    s = float(np.polyfit(sub["feature_078"], sub["pfs_months"],1)[0])
    slopes.append((v, s, int(len(sub))))
a.append({"hypothesis_ids": ["h36"],
          "code": "F-test ols(pfs ~ f078 + C(f057)) vs ols(pfs ~ f078*C(f057))",
          "result_summary": f"Interaction F={res['F']:.2f} df={res['df_diff']}, p={res['p']:.2e}. f078-on-pfs slope by f057: " + "; ".join(f"{v}: {s:+.4f} (n={n})" for v,s,n in slopes) + ".",
          "p_value": res["p"], "effect_estimate": float(slopes[2][1] - slopes[0][1]), "significant": res["p"] < 0.05})
add(23, hyps, a)

# ============================================================
# Iteration 24: Race x feature_057 (stage) interaction on pfs_months
# ============================================================
hyps = [
    {"id": "h37", "kind": "novel",
     "text": "Mean pfs_months across feature_064 race categories differs by feature_057 level (interaction p < 0.05)."},
]
a = []
res = interaction_test("pfs_months ~ C(feature_064) + C(feature_057) + feature_078",
                       "pfs_months ~ C(feature_064) * C(feature_057) + feature_078")
a.append({"hypothesis_ids": ["h37"],
          "code": "F-test C(f064):C(f057) interaction adjusted for f078",
          "result_summary": f"Interaction F={res['F']:.2f} df={res['df_diff']}, p={res['p']:.2e}.",
          "p_value": res["p"], "effect_estimate": float(res['F']), "significant": res["p"] < 0.05})
add(24, hyps, a)

# ============================================================
# Iteration 25: Final consolidated multivariable model
# ============================================================
hyps = [
    {"id": "h38", "kind": "refined",
     "text": "A consolidated multivariable model containing feature_078, C(feature_057), feature_051, feature_038, feature_013, feature_043, feature_109, feature_067, feature_099, feature_092 explains substantially more variance in pfs_months than feature_078 + C(feature_057) alone (Delta R^2 > 0.005)."},
]
a = []
m_simple = smf.ols("pfs_months ~ feature_078 + C(feature_057)", data=DF).fit()
m_final = smf.ols("pfs_months ~ feature_078 + C(feature_057) + feature_051 + feature_038 + feature_013 + feature_043 + feature_109 + feature_067 + feature_099 + feature_092", data=DF).fit()
dR2 = float(m_final.rsquared - m_simple.rsquared)
df_diff = m_simple.df_resid - m_final.df_resid
F = ((m_simple.ssr - m_final.ssr) / df_diff) / (m_final.ssr / m_final.df_resid)
p = 1 - stats.f.cdf(F, df_diff, m_final.df_resid)
coefs_print = {k: f"{float(v):+.4f} (p={float(m_final.pvalues[k]):.2e})" for k,v in m_final.params.items()}
a.append({"hypothesis_ids": ["h38"],
          "code": "ols simple vs final 10-feature model; nested F-test",
          "result_summary": f"Simple R^2={m_simple.rsquared:.4f}; final R^2={m_final.rsquared:.4f}; ΔR^2={dR2:.4f}. F={F:.2f}, p={p:.2e}. Coefs: {coefs_print}.",
          "p_value": float(p), "effect_estimate": dR2, "significant": p < 0.05})
add(25, hyps, a)

# ============================================================
# Write transcript and run also collect simpler structured-summary fodder
# ============================================================
transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@manual-iter25",
    "max_iterations": 25,
    "iterations": ITERS,
}
with open("transcript.json","w") as f:
    json.dump(transcript, f, indent=1, default=float)
print("transcript.json written: iterations =", len(ITERS))

# Build summary
summary_records = []
for it in ITERS:
    for an in it["analyses"]:
        summary_records.append({
            "iter": it["index"],
            "hyp_ids": an["hypothesis_ids"],
            "summary": an["result_summary"],
            "p": an.get("p_value"),
            "eff": an.get("effect_estimate"),
            "sig": an.get("significant"),
        })

with open("_run_summary_records.json","w") as f:
    json.dump(summary_records, f, indent=1, default=float)
print("done")
