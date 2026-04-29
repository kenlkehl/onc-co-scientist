"""Comprehensive iterative analysis for ds001_crc.

Runs 25 iterations of hypothesize-test-refine and emits a JSON dump of all
results in `_my_results.json`. The transcript and summary are then assembled
from this JSON.
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
y = df["pfs_months"]

results = {"iterations": []}


def add_iter(idx, hyps, analyses):
    results["iterations"].append(
        {"index": idx, "proposed_hypotheses": hyps, "analyses": analyses}
    )


def ttest(col, val1=1, val0=0):
    s = df[col]
    a = y[s == val1]
    b = y[s == val0]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(p), float(a.mean()), float(b.mean()), len(a), len(b)


def pearson(col):
    r, p = stats.pearsonr(df[col], y)
    return float(r), float(p)


def lin_reg(formula):
    m = smf.ols(formula, data=df).fit()
    return m


# ---------- Iteration 1: top-hit confirmations (continuous) ----------
r78, p78 = pearson("feature_078")
r57, p57 = pearson("feature_057")
ttest51 = ttest("feature_051")
add_iter(
    1,
    [
        {"id": "h1_1", "text": "feature_078 (continuous, range 30-90) is positively correlated with pfs_months: higher values associated with longer pfs_months.", "kind": "novel"},
        {"id": "h1_2", "text": "feature_057 (ordinal 0/1/2) is negatively associated with pfs_months: higher values associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h1_3", "text": "Patients with feature_051 = 1 have shorter mean pfs_months than patients with feature_051 = 0.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h1_1"], "code": "scipy.stats.pearsonr(df.feature_078, df.pfs_months)",
         "result_summary": f"Pearson r={r78:.4f}, p={p78:.3g}. Strong positive correlation: each additional unit of feature_078 associates with substantially longer pfs_months.",
         "p_value": p78, "effect_estimate": r78, "significant": p78 < 0.05},
        {"hypothesis_ids": ["h1_2"], "code": "scipy.stats.pearsonr(df.feature_057, df.pfs_months)",
         "result_summary": f"Pearson r={r57:.4f}, p={p57:.3g}. Higher feature_057 strongly associated with shorter pfs_months.",
         "p_value": p57, "effect_estimate": r57, "significant": p57 < 0.05},
        {"hypothesis_ids": ["h1_3"], "code": "Welch t-test: pfs_months[feature_051==1] vs pfs_months[feature_051==0]",
         "result_summary": f"Mean pfs_months: {ttest51[2]:.3f} (feature_051=1, n={ttest51[4]}) vs {ttest51[3]:.3f} (feature_051=0, n={ttest51[5]}). Difference {ttest51[0]:+.3f} months, p={ttest51[1]:.3g}.",
         "p_value": ttest51[1], "effect_estimate": ttest51[0], "significant": ttest51[1] < 0.05},
    ],
)

# ---------- Iteration 2: more top hits ----------
ttest38 = ttest("feature_038")
r99, p99 = pearson("feature_099")
r92, p92 = pearson("feature_092")
add_iter(
    2,
    [
        {"id": "h2_1", "text": "Patients with feature_038 = 1 have longer mean pfs_months than patients with feature_038 = 0.", "kind": "novel"},
        {"id": "h2_2", "text": "feature_099 (continuous, range 0-24.6) is negatively correlated with pfs_months.", "kind": "novel"},
        {"id": "h2_3", "text": "feature_092 (continuous, range 1.5-5.5) is positively correlated with pfs_months.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h2_1"], "code": "Welch t-test on feature_038",
         "result_summary": f"Mean pfs_months: {ttest38[2]:.3f} (feature_038=1, n={ttest38[4]}) vs {ttest38[3]:.3f} (feature_038=0, n={ttest38[5]}). Difference {ttest38[0]:+.3f} months, p={ttest38[1]:.3g}.",
         "p_value": ttest38[1], "effect_estimate": ttest38[0], "significant": ttest38[1] < 0.05},
        {"hypothesis_ids": ["h2_2"], "code": "scipy.stats.pearsonr(df.feature_099, df.pfs_months)",
         "result_summary": f"Pearson r={r99:.4f}, p={p99:.3g}. Higher feature_099 associated with shorter pfs_months.",
         "p_value": p99, "effect_estimate": r99, "significant": p99 < 0.05},
        {"hypothesis_ids": ["h2_3"], "code": "scipy.stats.pearsonr(df.feature_092, df.pfs_months)",
         "result_summary": f"Pearson r={r92:.4f}, p={p92:.3g}. Higher feature_092 associated with longer pfs_months.",
         "p_value": p92, "effect_estimate": r92, "significant": p92 < 0.05},
    ],
)

# ---------- Iteration 3 ----------
ttest13 = ttest("feature_013")
ttest43 = ttest("feature_043")
r9, p9 = pearson("feature_009")
add_iter(
    3,
    [
        {"id": "h3_1", "text": "Patients with feature_013 = 1 have shorter mean pfs_months than feature_013 = 0.", "kind": "novel"},
        {"id": "h3_2", "text": "Patients with feature_043 = 1 have shorter mean pfs_months than feature_043 = 0.", "kind": "novel"},
        {"id": "h3_3", "text": "feature_009 (continuous, range 0.02-777, heavy-tailed) is negatively correlated with pfs_months.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h3_1"],
         "result_summary": f"feature_013=1 mean {ttest13[2]:.3f} vs 0 mean {ttest13[3]:.3f}, diff {ttest13[0]:+.3f} months, p={ttest13[1]:.3g}.",
         "p_value": ttest13[1], "effect_estimate": ttest13[0], "significant": ttest13[1] < 0.05},
        {"hypothesis_ids": ["h3_2"],
         "result_summary": f"feature_043=1 mean {ttest43[2]:.3f} vs 0 mean {ttest43[3]:.3f}, diff {ttest43[0]:+.3f} months, p={ttest43[1]:.3g}.",
         "p_value": ttest43[1], "effect_estimate": ttest43[0], "significant": ttest43[1] < 0.05},
        {"hypothesis_ids": ["h3_3"],
         "result_summary": f"Pearson r={r9:.4f}, p={p9:.3g}. Heavy-tailed; small but significant negative correlation.",
         "p_value": p9, "effect_estimate": r9, "significant": p9 < 0.05},
    ],
)

# ---------- Iteration 4 ----------
ttest109 = ttest("feature_109")
ttest67 = ttest("feature_067")
r6, p6 = pearson("feature_006")
add_iter(
    4,
    [
        {"id": "h4_1", "text": "Patients with feature_109 = 1 have shorter mean pfs_months than feature_109 = 0.", "kind": "novel"},
        {"id": "h4_2", "text": "Patients with feature_067 = 1 have longer mean pfs_months than feature_067 = 0.", "kind": "novel"},
        {"id": "h4_3", "text": "feature_006 (continuous, range 48-830) is negatively associated with pfs_months.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h4_1"],
         "result_summary": f"feature_109=1 mean {ttest109[2]:.3f} vs 0 mean {ttest109[3]:.3f}, diff {ttest109[0]:+.3f}, p={ttest109[1]:.3g}.",
         "p_value": ttest109[1], "effect_estimate": ttest109[0], "significant": ttest109[1] < 0.05},
        {"hypothesis_ids": ["h4_2"],
         "result_summary": f"feature_067=1 mean {ttest67[2]:.3f} vs 0 mean {ttest67[3]:.3f}, diff {ttest67[0]:+.3f}, p={ttest67[1]:.3g}.",
         "p_value": ttest67[1], "effect_estimate": ttest67[0], "significant": ttest67[1] < 0.05},
        {"hypothesis_ids": ["h4_3"],
         "result_summary": f"Pearson r={r6:.4f}, p={p6:.3g}. Marginal negative correlation.",
         "p_value": p6, "effect_estimate": r6, "significant": p6 < 0.05},
    ],
)

# ---------- Iteration 5: multivariable model with top features ----------
m5 = lin_reg("pfs_months ~ feature_078 + C(feature_057) + feature_051 + feature_038 + feature_099 + feature_092 + feature_013 + feature_043 + feature_009 + feature_109 + feature_067 + feature_006")
m5p = m5.pvalues.to_dict()
m5c = m5.params.to_dict()
add_iter(
    5,
    [
        {"id": "h5_1", "text": "feature_078 retains a positive association with pfs_months in a multivariable linear model adjusted for feature_057, feature_051, feature_038, feature_099, feature_092, feature_013, feature_043, feature_009, feature_109, feature_067 and feature_006.", "kind": "refined"},
        {"id": "h5_2", "text": "feature_051 retains a negative association with pfs_months after multivariable adjustment.", "kind": "refined"},
        {"id": "h5_3", "text": "feature_038 retains a positive association with pfs_months after multivariable adjustment.", "kind": "refined"},
    ],
    [
        {"hypothesis_ids": ["h5_1"], "code": "smf.ols(...).fit()",
         "result_summary": f"Adjusted coef for feature_078 = {m5c.get('feature_078', float('nan')):+.4f} months per unit, p={m5p.get('feature_078', float('nan')):.3g}.",
         "p_value": float(m5p.get("feature_078", float("nan"))), "effect_estimate": float(m5c.get("feature_078", float("nan"))), "significant": m5p.get("feature_078", 1) < 0.05},
        {"hypothesis_ids": ["h5_2"],
         "result_summary": f"Adjusted coef for feature_051 = {m5c.get('feature_051', float('nan')):+.4f} months, p={m5p.get('feature_051', float('nan')):.3g}.",
         "p_value": float(m5p.get("feature_051", float("nan"))), "effect_estimate": float(m5c.get("feature_051", float("nan"))), "significant": m5p.get("feature_051", 1) < 0.05},
        {"hypothesis_ids": ["h5_3"],
         "result_summary": f"Adjusted coef for feature_038 = {m5c.get('feature_038', float('nan')):+.4f} months, p={m5p.get('feature_038', float('nan')):.3g}.",
         "p_value": float(m5p.get("feature_038", float("nan"))), "effect_estimate": float(m5c.get("feature_038", float("nan"))), "significant": m5p.get("feature_038", 1) < 0.05},
    ],
)
print("M5 R^2:", m5.rsquared, "n=", int(m5.nobs))
print("M5 params:")
for k, v in m5c.items():
    print(f"  {k}: coef={v:+.4f}, p={m5p[k]:.3g}")

# ---------- Iteration 6: Race (feature_064) ----------
race_groups = {v: y[df["feature_064"] == v].values for v in df["feature_064"].unique()}
fanova, panova = stats.f_oneway(*race_groups.values())
race_means = {k: float(v.mean()) for k, v in race_groups.items()}
# Pairwise: white vs black
mw = race_means["white"]
mb = race_means["black"]
mh = race_means["hispanic"]
ma = race_means["asian"]
twb, pwb = stats.ttest_ind(race_groups["white"], race_groups["black"], equal_var=False)
twh, pwh = stats.ttest_ind(race_groups["white"], race_groups["hispanic"], equal_var=False)
twa, pwa = stats.ttest_ind(race_groups["white"], race_groups["asian"], equal_var=False)
add_iter(
    6,
    [
        {"id": "h6_1", "text": "Mean pfs_months differs across feature_064 categories (white, hispanic, black, asian, other).", "kind": "novel"},
        {"id": "h6_2", "text": "Mean pfs_months is shorter for feature_064 = black compared with feature_064 = white.", "kind": "novel"},
        {"id": "h6_3", "text": "Mean pfs_months differs between feature_064 = hispanic and feature_064 = white.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h6_1"], "code": "stats.f_oneway across feature_064",
         "result_summary": f"ANOVA F={fanova:.3f}, p={panova:.3g}. Group means: " + ", ".join(f"{k}={v:.3f}" for k, v in race_means.items()),
         "p_value": float(panova), "effect_estimate": float(max(race_means.values()) - min(race_means.values())), "significant": panova < 0.05},
        {"hypothesis_ids": ["h6_2"],
         "result_summary": f"black mean {mb:.3f} vs white mean {mw:.3f}, diff {mb - mw:+.3f}, p={pwb:.3g}.",
         "p_value": float(pwb), "effect_estimate": float(mb - mw), "significant": pwb < 0.05},
        {"hypothesis_ids": ["h6_3"],
         "result_summary": f"hispanic mean {mh:.3f} vs white mean {mw:.3f}, diff {mh - mw:+.3f}, p={pwh:.3g}.",
         "p_value": float(pwh), "effect_estimate": float(mh - mw), "significant": pwh < 0.05},
    ],
)

# ---------- Iteration 7: Insurance (feature_018) ----------
ins_groups = {v: y[df["feature_018"] == v].values for v in df["feature_018"].unique()}
fanova18, panova18 = stats.f_oneway(*ins_groups.values())
ins_means = {k: float(v.mean()) for k, v in ins_groups.items()}
mp = ins_means["private"]
mu = ins_means["uninsured"]
mc = ins_means["medicaid"]
mm = ins_means["medicare"]
tpu, ppu = stats.ttest_ind(ins_groups["private"], ins_groups["uninsured"], equal_var=False)
tpc, ppc = stats.ttest_ind(ins_groups["private"], ins_groups["medicaid"], equal_var=False)
tpm, ppm = stats.ttest_ind(ins_groups["private"], ins_groups["medicare"], equal_var=False)
add_iter(
    7,
    [
        {"id": "h7_1", "text": "Mean pfs_months differs across feature_018 categories (medicare, private, medicaid, uninsured).", "kind": "novel"},
        {"id": "h7_2", "text": "Patients with feature_018 = uninsured have shorter mean pfs_months than feature_018 = private.", "kind": "novel"},
        {"id": "h7_3", "text": "Patients with feature_018 = medicaid have shorter mean pfs_months than feature_018 = private.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h7_1"], "code": "stats.f_oneway across feature_018",
         "result_summary": f"ANOVA F={fanova18:.3f}, p={panova18:.3g}. Means: " + ", ".join(f"{k}={v:.3f}" for k, v in ins_means.items()),
         "p_value": float(panova18), "effect_estimate": float(max(ins_means.values()) - min(ins_means.values())), "significant": panova18 < 0.05},
        {"hypothesis_ids": ["h7_2"],
         "result_summary": f"uninsured mean {mu:.3f} vs private mean {mp:.3f}, diff {mu - mp:+.3f}, p={ppu:.3g}.",
         "p_value": float(ppu), "effect_estimate": float(mu - mp), "significant": ppu < 0.05},
        {"hypothesis_ids": ["h7_3"],
         "result_summary": f"medicaid mean {mc:.3f} vs private mean {mp:.3f}, diff {mc - mp:+.3f}, p={ppc:.3g}.",
         "p_value": float(ppc), "effect_estimate": float(mc - mp), "significant": ppc < 0.05},
    ],
)

# ---------- Iteration 8: feature_051 x feature_038 interaction ----------
m8 = lin_reg("pfs_months ~ feature_051 * feature_038")
b8 = m8.params.to_dict()
p8 = m8.pvalues.to_dict()
sub_means = {
    (a, b): float(y[(df["feature_051"] == a) & (df["feature_038"] == b)].mean())
    for a in (0, 1) for b in (0, 1)
}
add_iter(
    8,
    [
        {"id": "h8_1", "text": "The interaction term feature_051 x feature_038 is statistically significant: the negative effect of feature_051 on pfs_months differs depending on feature_038.", "kind": "novel"},
        {"id": "h8_2", "text": "Patients with feature_051 = 1 and feature_038 = 0 have the shortest mean pfs_months among the four feature_051 x feature_038 cells.", "kind": "novel"},
        {"id": "h8_3", "text": "Patients with feature_051 = 0 and feature_038 = 1 have the longest mean pfs_months among the four feature_051 x feature_038 cells.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h8_1"], "code": "smf.ols('pfs_months ~ feature_051 * feature_038', data=df).fit()",
         "result_summary": f"Interaction coef = {b8.get('feature_051:feature_038', float('nan')):+.4f}, p={p8.get('feature_051:feature_038', float('nan')):.3g}. Cell means: " + ", ".join(f"({a},{b})={v:.3f}" for (a, b), v in sub_means.items()),
         "p_value": float(p8.get("feature_051:feature_038", float("nan"))), "effect_estimate": float(b8.get("feature_051:feature_038", float("nan"))), "significant": p8.get("feature_051:feature_038", 1) < 0.05},
        {"hypothesis_ids": ["h8_2"],
         "result_summary": f"Mean pfs_months for feature_051=1, feature_038=0 = {sub_means[(1,0)]:.3f}; min across cells = {min(sub_means.values()):.3f}.",
         "p_value": None, "effect_estimate": float(sub_means[(1, 0)] - max(sub_means.values())), "significant": sub_means[(1, 0)] == min(sub_means.values())},
        {"hypothesis_ids": ["h8_3"],
         "result_summary": f"Mean pfs_months for feature_051=0, feature_038=1 = {sub_means[(0,1)]:.3f}; max across cells = {max(sub_means.values()):.3f}.",
         "p_value": None, "effect_estimate": float(sub_means[(0, 1)] - min(sub_means.values())), "significant": sub_means[(0, 1)] == max(sub_means.values())},
    ],
)

# ---------- Iteration 9: feature_051 x feature_057 (treatment x stage-like) ----------
m9 = lin_reg("pfs_months ~ feature_051 * feature_057")
b9 = m9.params.to_dict()
p9m = m9.pvalues.to_dict()
sub51_57 = {(a, b): float(y[(df["feature_051"] == a) & (df["feature_057"] == b)].mean()) for a in (0, 1) for b in (0, 1, 2)}
# stratum-specific effect of feature_051
strat_eff = {}
for lvl in (0, 1, 2):
    a = y[(df["feature_051"] == 1) & (df["feature_057"] == lvl)]
    b = y[(df["feature_051"] == 0) & (df["feature_057"] == lvl)]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    strat_eff[lvl] = (float(a.mean() - b.mean()), float(p), len(a), len(b))
add_iter(
    9,
    [
        {"id": "h9_1", "text": "The interaction feature_051 x feature_057 is significant: the (negative) effect of feature_051 on pfs_months varies across feature_057 levels (0/1/2).", "kind": "novel"},
        {"id": "h9_2", "text": "Within feature_057 = 2 (highest level), feature_051 = 1 still produces shorter pfs_months than feature_051 = 0.", "kind": "novel"},
        {"id": "h9_3", "text": "The negative pfs_months effect of feature_051 is largest in absolute magnitude in feature_057 = 0 patients (lowest stage-like level).", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h9_1"], "code": "smf.ols('pfs_months ~ feature_051 * feature_057', data=df).fit()",
         "result_summary": f"Interaction coef = {b9.get('feature_051:feature_057', float('nan')):+.4f}, p={p9m.get('feature_051:feature_057', float('nan')):.3g}. Stratum means feature_051={{0,1}} x feature_057={{0,1,2}}: " + ", ".join(f"({a},{b})={v:.3f}" for (a, b), v in sub51_57.items()),
         "p_value": float(p9m.get("feature_051:feature_057", float("nan"))), "effect_estimate": float(b9.get("feature_051:feature_057", float("nan"))), "significant": p9m.get("feature_051:feature_057", 1) < 0.05},
        {"hypothesis_ids": ["h9_2"],
         "result_summary": f"Within feature_057=2: feature_051 effect = {strat_eff[2][0]:+.3f}, p={strat_eff[2][1]:.3g} (n1={strat_eff[2][2]}, n0={strat_eff[2][3]}).",
         "p_value": strat_eff[2][1], "effect_estimate": strat_eff[2][0], "significant": strat_eff[2][1] < 0.05},
        {"hypothesis_ids": ["h9_3"],
         "result_summary": "Stratum-specific feature_051 effect: " + ", ".join(f"feature_057={k}: {v[0]:+.3f} (p={v[1]:.3g})" for k, v in strat_eff.items()),
         "p_value": None, "effect_estimate": strat_eff[0][0], "significant": abs(strat_eff[0][0]) >= max(abs(strat_eff[1][0]), abs(strat_eff[2][0]))},
    ],
)

# ---------- Iteration 10: feature_038 x feature_057 interaction ----------
m10 = lin_reg("pfs_months ~ feature_038 * feature_057")
b10 = m10.params.to_dict()
p10 = m10.pvalues.to_dict()
strat_eff_38 = {}
for lvl in (0, 1, 2):
    a = y[(df["feature_038"] == 1) & (df["feature_057"] == lvl)]
    b = y[(df["feature_038"] == 0) & (df["feature_057"] == lvl)]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    strat_eff_38[lvl] = (float(a.mean() - b.mean()), float(p), len(a), len(b))
add_iter(
    10,
    [
        {"id": "h10_1", "text": "The interaction feature_038 x feature_057 is significant: the positive effect of feature_038 on pfs_months differs across feature_057 levels.", "kind": "novel"},
        {"id": "h10_2", "text": "The benefit of feature_038 = 1 over feature_038 = 0 (in pfs_months) is largest in feature_057 = 2 patients.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h10_1"],
         "result_summary": f"Interaction coef = {b10.get('feature_038:feature_057', float('nan')):+.4f}, p={p10.get('feature_038:feature_057', float('nan')):.3g}.",
         "p_value": float(p10.get("feature_038:feature_057", float("nan"))), "effect_estimate": float(b10.get("feature_038:feature_057", float("nan"))), "significant": p10.get("feature_038:feature_057", 1) < 0.05},
        {"hypothesis_ids": ["h10_2"],
         "result_summary": "Stratum-specific feature_038 effect: " + ", ".join(f"feature_057={k}: {v[0]:+.3f} (p={v[1]:.3g})" for k, v in strat_eff_38.items()),
         "p_value": None,
         "effect_estimate": strat_eff_38[2][0],
         "significant": strat_eff_38[2][0] >= max(strat_eff_38[0][0], strat_eff_38[1][0])},
    ],
)

# ---------- Iteration 11: feature_078 x feature_051 (does treatment effect vary by age-like) ----------
m11 = lin_reg("pfs_months ~ feature_078 * feature_051")
b11 = m11.params.to_dict()
p11 = m11.pvalues.to_dict()
# By tertile of feature_078
df["_f78_tert"] = pd.qcut(df["feature_078"], 3, labels=False)
strat_51_by_age = {}
for t in (0, 1, 2):
    a = y[(df["feature_051"] == 1) & (df["_f78_tert"] == t)]
    b = y[(df["feature_051"] == 0) & (df["_f78_tert"] == t)]
    tt, pp = stats.ttest_ind(a, b, equal_var=False)
    strat_51_by_age[t] = (float(a.mean() - b.mean()), float(pp))
add_iter(
    11,
    [
        {"id": "h11_1", "text": "The interaction feature_078 x feature_051 is significant: the (negative) effect of feature_051 on pfs_months varies by feature_078 level.", "kind": "novel"},
        {"id": "h11_2", "text": "Within the lowest tertile of feature_078, the negative effect of feature_051 = 1 on pfs_months is more pronounced than in the highest tertile.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h11_1"],
         "result_summary": f"Interaction coef = {b11.get('feature_078:feature_051', float('nan')):+.5f}, p={p11.get('feature_078:feature_051', float('nan')):.3g}.",
         "p_value": float(p11.get("feature_078:feature_051", float("nan"))), "effect_estimate": float(b11.get("feature_078:feature_051", float("nan"))), "significant": p11.get("feature_078:feature_051", 1) < 0.05},
        {"hypothesis_ids": ["h11_2"],
         "result_summary": "By feature_078 tertile, feature_051 effect: " + ", ".join(f"T{k}: {v[0]:+.3f} (p={v[1]:.3g})" for k, v in strat_51_by_age.items()),
         "p_value": None,
         "effect_estimate": strat_51_by_age[0][0] - strat_51_by_age[2][0],
         "significant": strat_51_by_age[0][0] < strat_51_by_age[2][0]},
    ],
)

# ---------- Iteration 12: race x feature_051 interaction (disparity in treatment effect) ----------
m12 = lin_reg("pfs_months ~ feature_051 * C(feature_064)")
p12 = m12.pvalues.to_dict()
b12 = m12.params.to_dict()
# stratum-specific effect of feature_051 by race
race_levels = ["white", "black", "hispanic", "asian", "other"]
strat_51_race = {}
for r in race_levels:
    a = y[(df["feature_051"] == 1) & (df["feature_064"] == r)]
    b = y[(df["feature_051"] == 0) & (df["feature_064"] == r)]
    if len(a) > 5 and len(b) > 5:
        tt, pp = stats.ttest_ind(a, b, equal_var=False)
        strat_51_race[r] = (float(a.mean() - b.mean()), float(pp), len(a), len(b))
# Joint test: F-test on all interaction terms
inter_terms = [k for k in p12.keys() if "feature_051:" in k]
joint = m12.f_test(" = ".join(inter_terms) + " = 0") if inter_terms else None
add_iter(
    12,
    [
        {"id": "h12_1", "text": "The interaction feature_051 x feature_064 is jointly significant: the effect of feature_051 on pfs_months differs by feature_064 (race).", "kind": "novel"},
        {"id": "h12_2", "text": "Within feature_064 = black, feature_051 = 1 produces a larger negative pfs_months effect than within feature_064 = white.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h12_1"], "code": "joint F-test on feature_051:C(feature_064) terms",
         "result_summary": f"Joint F = {float(joint.fvalue) if joint is not None else float('nan'):.3f}, p={float(joint.pvalue) if joint is not None else float('nan'):.3g}.",
         "p_value": float(joint.pvalue) if joint is not None else None,
         "effect_estimate": float(joint.fvalue) if joint is not None else None,
         "significant": (float(joint.pvalue) if joint is not None else 1) < 0.05},
        {"hypothesis_ids": ["h12_2"],
         "result_summary": "Stratum-specific feature_051 effect by race: " + ", ".join(f"{k}: {v[0]:+.3f} (p={v[1]:.3g}, n1={v[2]}, n0={v[3]})" for k, v in strat_51_race.items()),
         "p_value": None,
         "effect_estimate": strat_51_race.get("black", (None,))[0],
         "significant": (strat_51_race.get("black", (0, 1))[0] < strat_51_race.get("white", (0, 1))[0])},
    ],
)

# ---------- Iteration 13: insurance x feature_078 (age) ----------
m13 = lin_reg("pfs_months ~ feature_078 * C(feature_018)")
p13 = m13.pvalues.to_dict()
inter_terms_13 = [k for k in p13.keys() if "feature_078:" in k]
joint13 = m13.f_test(" = ".join(inter_terms_13) + " = 0") if inter_terms_13 else None
# slope of feature_078 by insurance
ins_slope = {}
for ins in df["feature_018"].unique():
    sub = df[df["feature_018"] == ins]
    r, p = stats.pearsonr(sub["feature_078"], sub["pfs_months"])
    ins_slope[ins] = (float(r), float(p), len(sub))
add_iter(
    13,
    [
        {"id": "h13_1", "text": "The interaction feature_078 x feature_018 is jointly significant: the slope of pfs_months on feature_078 differs by insurance group.", "kind": "novel"},
        {"id": "h13_2", "text": "Within feature_018 = uninsured, the positive correlation between feature_078 and pfs_months is weaker than within feature_018 = private.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h13_1"], "code": "joint F-test on feature_078:C(feature_018) terms",
         "result_summary": f"Joint F = {float(joint13.fvalue) if joint13 is not None else float('nan'):.3f}, p={float(joint13.pvalue) if joint13 is not None else float('nan'):.3g}.",
         "p_value": float(joint13.pvalue) if joint13 is not None else None,
         "effect_estimate": float(joint13.fvalue) if joint13 is not None else None,
         "significant": (float(joint13.pvalue) if joint13 is not None else 1) < 0.05},
        {"hypothesis_ids": ["h13_2"],
         "result_summary": "Pearson r(feature_078, pfs_months) by insurance: " + ", ".join(f"{k}: r={v[0]:.3f}, p={v[1]:.3g}, n={v[2]}" for k, v in ins_slope.items()),
         "p_value": None,
         "effect_estimate": float(ins_slope.get("uninsured", (0,))[0] - ins_slope.get("private", (0,))[0]),
         "significant": ins_slope.get("uninsured", (0,))[0] < ins_slope.get("private", (0,))[0]},
    ],
)

# ---------- Iteration 14: feature_092 (lab) x feature_051 ----------
df["_f92_low"] = (df["feature_092"] < 3.5).astype(int)
m14 = lin_reg("pfs_months ~ feature_051 * _f92_low")
p14 = m14.pvalues.to_dict()
b14 = m14.params.to_dict()
sub_92 = {(a, b): float(y[(df["feature_051"] == a) & (df["_f92_low"] == b)].mean()) for a in (0, 1) for b in (0, 1)}
add_iter(
    14,
    [
        {"id": "h14_1", "text": "feature_051 effect on pfs_months differs in patients with feature_092 < 3.5 vs feature_092 >= 3.5 (interaction term significant).", "kind": "novel"},
        {"id": "h14_2", "text": "Among patients with feature_092 < 3.5, the negative effect of feature_051 is larger in absolute magnitude than among patients with feature_092 >= 3.5.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h14_1"],
         "result_summary": f"Interaction coef = {b14.get('feature_051:_f92_low', float('nan')):+.4f}, p={p14.get('feature_051:_f92_low', float('nan')):.3g}. Cell means: " + ", ".join(f"({a},{b})={v:.3f}" for (a, b), v in sub_92.items()),
         "p_value": float(p14.get("feature_051:_f92_low", float("nan"))),
         "effect_estimate": float(b14.get("feature_051:_f92_low", float("nan"))),
         "significant": p14.get("feature_051:_f92_low", 1) < 0.05},
        {"hypothesis_ids": ["h14_2"],
         "result_summary": f"feature_051 effect among feature_092<3.5: {sub_92[(1,1)] - sub_92[(0,1)]:+.3f}; among >=3.5: {sub_92[(1,0)] - sub_92[(0,0)]:+.3f}.",
         "p_value": None,
         "effect_estimate": (sub_92[(1, 1)] - sub_92[(0, 1)]) - (sub_92[(1, 0)] - sub_92[(0, 0)]),
         "significant": abs(sub_92[(1, 1)] - sub_92[(0, 1)]) > abs(sub_92[(1, 0)] - sub_92[(0, 0)])},
    ],
)

# ---------- Iteration 15: feature_099 x feature_038 ----------
df["_f99_high"] = (df["feature_099"] > df["feature_099"].median()).astype(int)
m15 = lin_reg("pfs_months ~ feature_038 * _f99_high")
p15 = m15.pvalues.to_dict()
b15 = m15.params.to_dict()
sub_99 = {(a, b): float(y[(df["feature_038"] == a) & (df["_f99_high"] == b)].mean()) for a in (0, 1) for b in (0, 1)}
add_iter(
    15,
    [
        {"id": "h15_1", "text": "The benefit of feature_038 = 1 on pfs_months differs in patients with feature_099 above vs below its median (interaction significant).", "kind": "novel"},
        {"id": "h15_2", "text": "Among patients with feature_099 above the median, feature_038 = 1 still confers a positive pfs_months benefit.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h15_1"],
         "result_summary": f"Interaction coef = {b15.get('feature_038:_f99_high', float('nan')):+.4f}, p={p15.get('feature_038:_f99_high', float('nan')):.3g}. Cell means: " + ", ".join(f"({a},{b})={v:.3f}" for (a, b), v in sub_99.items()),
         "p_value": float(p15.get("feature_038:_f99_high", float("nan"))),
         "effect_estimate": float(b15.get("feature_038:_f99_high", float("nan"))),
         "significant": p15.get("feature_038:_f99_high", 1) < 0.05},
        {"hypothesis_ids": ["h15_2"],
         "result_summary": f"Within feature_099_high: feature_038 effect = {sub_99[(1,1)] - sub_99[(0,1)]:+.3f}.",
         "p_value": None,
         "effect_estimate": float(sub_99[(1, 1)] - sub_99[(0, 1)]),
         "significant": (sub_99[(1, 1)] - sub_99[(0, 1)]) > 0},
    ],
)

# ---------- Iteration 16: race x feature_038 (treatment 2) ----------
m16 = lin_reg("pfs_months ~ feature_038 * C(feature_064)")
p16 = m16.pvalues.to_dict()
inter_terms_16 = [k for k in p16.keys() if "feature_038:" in k]
joint16 = m16.f_test(" = ".join(inter_terms_16) + " = 0") if inter_terms_16 else None
strat_38_race = {}
for r in race_levels:
    a = y[(df["feature_038"] == 1) & (df["feature_064"] == r)]
    b = y[(df["feature_038"] == 0) & (df["feature_064"] == r)]
    if len(a) > 5 and len(b) > 5:
        tt, pp = stats.ttest_ind(a, b, equal_var=False)
        strat_38_race[r] = (float(a.mean() - b.mean()), float(pp), len(a), len(b))
add_iter(
    16,
    [
        {"id": "h16_1", "text": "The interaction feature_038 x feature_064 is jointly significant: the pfs_months benefit of feature_038 differs across racial groups.", "kind": "novel"},
        {"id": "h16_2", "text": "Within every feature_064 group with sufficient sample size, the effect of feature_038 = 1 vs 0 on pfs_months remains positive.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h16_1"], "code": "joint F-test on feature_038:C(feature_064) terms",
         "result_summary": f"Joint F = {float(joint16.fvalue) if joint16 is not None else float('nan'):.3f}, p={float(joint16.pvalue) if joint16 is not None else float('nan'):.3g}.",
         "p_value": float(joint16.pvalue) if joint16 is not None else None,
         "effect_estimate": float(joint16.fvalue) if joint16 is not None else None,
         "significant": (float(joint16.pvalue) if joint16 is not None else 1) < 0.05},
        {"hypothesis_ids": ["h16_2"],
         "result_summary": "Stratum-specific feature_038 effect by race: " + ", ".join(f"{k}: {v[0]:+.3f} (p={v[1]:.3g})" for k, v in strat_38_race.items()),
         "p_value": None,
         "effect_estimate": min(v[0] for v in strat_38_race.values()),
         "significant": all(v[0] > 0 for v in strat_38_race.values())},
    ],
)

# ---------- Iteration 17: insurance x feature_051 ----------
m17 = lin_reg("pfs_months ~ feature_051 * C(feature_018)")
p17 = m17.pvalues.to_dict()
inter_terms_17 = [k for k in p17.keys() if "feature_051:" in k]
joint17 = m17.f_test(" = ".join(inter_terms_17) + " = 0") if inter_terms_17 else None
strat_51_ins = {}
for ins in df["feature_018"].unique():
    a = y[(df["feature_051"] == 1) & (df["feature_018"] == ins)]
    b = y[(df["feature_051"] == 0) & (df["feature_018"] == ins)]
    if len(a) > 5 and len(b) > 5:
        tt, pp = stats.ttest_ind(a, b, equal_var=False)
        strat_51_ins[ins] = (float(a.mean() - b.mean()), float(pp), len(a), len(b))
add_iter(
    17,
    [
        {"id": "h17_1", "text": "The interaction feature_051 x feature_018 is jointly significant: the harmful effect of feature_051 on pfs_months differs by insurance.", "kind": "novel"},
        {"id": "h17_2", "text": "Within feature_018 = uninsured, the negative effect of feature_051 = 1 on pfs_months is larger in magnitude than within feature_018 = private.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h17_1"], "code": "joint F-test on feature_051:C(feature_018) terms",
         "result_summary": f"Joint F = {float(joint17.fvalue) if joint17 is not None else float('nan'):.3f}, p={float(joint17.pvalue) if joint17 is not None else float('nan'):.3g}.",
         "p_value": float(joint17.pvalue) if joint17 is not None else None,
         "effect_estimate": float(joint17.fvalue) if joint17 is not None else None,
         "significant": (float(joint17.pvalue) if joint17 is not None else 1) < 0.05},
        {"hypothesis_ids": ["h17_2"],
         "result_summary": "Stratum-specific feature_051 effect by insurance: " + ", ".join(f"{k}: {v[0]:+.3f} (p={v[1]:.3g})" for k, v in strat_51_ins.items()),
         "p_value": None,
         "effect_estimate": float(strat_51_ins.get("uninsured", (0,))[0] - strat_51_ins.get("private", (0,))[0]),
         "significant": strat_51_ins.get("uninsured", (0,))[0] < strat_51_ins.get("private", (0,))[0]},
    ],
)

# ---------- Iteration 18: race x feature_057 (stage-like) ----------
m18 = lin_reg("pfs_months ~ feature_057 * C(feature_064)")
p18 = m18.pvalues.to_dict()
inter_terms_18 = [k for k in p18.keys() if "feature_057:" in k]
joint18 = m18.f_test(" = ".join(inter_terms_18) + " = 0") if inter_terms_18 else None
slope57_race = {}
for r in race_levels:
    sub = df[df["feature_064"] == r]
    rr, pp = stats.pearsonr(sub["feature_057"], sub["pfs_months"])
    slope57_race[r] = (float(rr), float(pp), len(sub))
add_iter(
    18,
    [
        {"id": "h18_1", "text": "The interaction feature_057 x feature_064 is jointly significant: the negative slope of pfs_months on feature_057 differs across racial groups.", "kind": "novel"},
        {"id": "h18_2", "text": "Within every feature_064 group, higher feature_057 levels are associated with shorter pfs_months.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h18_1"], "code": "joint F-test on feature_057:C(feature_064) terms",
         "result_summary": f"Joint F = {float(joint18.fvalue) if joint18 is not None else float('nan'):.3f}, p={float(joint18.pvalue) if joint18 is not None else float('nan'):.3g}.",
         "p_value": float(joint18.pvalue) if joint18 is not None else None,
         "effect_estimate": float(joint18.fvalue) if joint18 is not None else None,
         "significant": (float(joint18.pvalue) if joint18 is not None else 1) < 0.05},
        {"hypothesis_ids": ["h18_2"],
         "result_summary": "Pearson r(feature_057, pfs_months) by race: " + ", ".join(f"{k}: r={v[0]:.3f}, p={v[1]:.3g}" for k, v in slope57_race.items()),
         "p_value": None,
         "effect_estimate": max(v[0] for v in slope57_race.values()),
         "significant": all(v[0] < 0 for v in slope57_race.values())},
    ],
)

# ---------- Iteration 19: feature_078 x feature_038 ----------
m19 = lin_reg("pfs_months ~ feature_078 * feature_038")
b19 = m19.params.to_dict()
p19 = m19.pvalues.to_dict()
strat_38_age = {}
for t in (0, 1, 2):
    a = y[(df["feature_038"] == 1) & (df["_f78_tert"] == t)]
    b = y[(df["feature_038"] == 0) & (df["_f78_tert"] == t)]
    tt, pp = stats.ttest_ind(a, b, equal_var=False)
    strat_38_age[t] = (float(a.mean() - b.mean()), float(pp))
add_iter(
    19,
    [
        {"id": "h19_1", "text": "The interaction feature_078 x feature_038 is significant: the pfs_months benefit of feature_038 = 1 varies with feature_078.", "kind": "novel"},
        {"id": "h19_2", "text": "The pfs_months benefit of feature_038 = 1 (vs 0) is larger in the highest tertile of feature_078 than in the lowest.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h19_1"],
         "result_summary": f"Interaction coef = {b19.get('feature_078:feature_038', float('nan')):+.5f}, p={p19.get('feature_078:feature_038', float('nan')):.3g}.",
         "p_value": float(p19.get("feature_078:feature_038", float("nan"))),
         "effect_estimate": float(b19.get("feature_078:feature_038", float("nan"))),
         "significant": p19.get("feature_078:feature_038", 1) < 0.05},
        {"hypothesis_ids": ["h19_2"],
         "result_summary": "By feature_078 tertile, feature_038 effect: " + ", ".join(f"T{k}: {v[0]:+.3f} (p={v[1]:.3g})" for k, v in strat_38_age.items()),
         "p_value": None,
         "effect_estimate": float(strat_38_age[2][0] - strat_38_age[0][0]),
         "significant": strat_38_age[2][0] > strat_38_age[0][0]},
    ],
)

# ---------- Iteration 20: feature_013 + feature_043 combined index ----------
df["_burden"] = df["feature_013"] + df["feature_043"]
burden_means = {k: float(y[df["_burden"] == k].mean()) for k in sorted(df["_burden"].unique())}
m20 = lin_reg("pfs_months ~ _burden")
p20 = float(m20.pvalues.get("_burden", float("nan")))
b20 = float(m20.params.get("_burden", float("nan")))
# Sub-cells
ttest_burden_2_0 = stats.ttest_ind(y[df["_burden"] == 2], y[df["_burden"] == 0], equal_var=False)
add_iter(
    20,
    [
        {"id": "h20_1", "text": "A combined burden score (feature_013 + feature_043) is monotonically negatively associated with pfs_months: more positive flags associate with shorter pfs_months.", "kind": "novel"},
        {"id": "h20_2", "text": "Patients with feature_013 = 1 AND feature_043 = 1 have shorter mean pfs_months than patients with feature_013 = 0 AND feature_043 = 0.", "kind": "refined"},
    ],
    [
        {"hypothesis_ids": ["h20_1"],
         "result_summary": f"Linear slope on burden score = {b20:+.3f} months per +1 flag, p={p20:.3g}. Group means: " + ", ".join(f"burden={k}: {v:.3f}" for k, v in burden_means.items()),
         "p_value": p20, "effect_estimate": b20, "significant": p20 < 0.05},
        {"hypothesis_ids": ["h20_2"],
         "result_summary": f"burden=2 mean {burden_means[2]:.3f} vs burden=0 mean {burden_means[0]:.3f}, diff {burden_means[2] - burden_means[0]:+.3f}, p={float(ttest_burden_2_0.pvalue):.3g}.",
         "p_value": float(ttest_burden_2_0.pvalue),
         "effect_estimate": float(burden_means[2] - burden_means[0]),
         "significant": float(ttest_burden_2_0.pvalue) < 0.05},
    ],
)

# ---------- Iteration 21: low-albumin-like (feature_092) and high-LDH-like (feature_099) combined ----------
df["_bad_labs"] = ((df["feature_092"] < 3.5) & (df["feature_099"] > df["feature_099"].median())).astype(int)
ttest_badlabs = ttest("_bad_labs")
# Adjusted in multivariable
m21 = lin_reg("pfs_months ~ _bad_labs + feature_078 + feature_051 + feature_038 + C(feature_057)")
p21 = float(m21.pvalues.get("_bad_labs", float("nan")))
b21 = float(m21.params.get("_bad_labs", float("nan")))
add_iter(
    21,
    [
        {"id": "h21_1", "text": "Patients in the 'bad labs' subgroup (feature_092 < 3.5 AND feature_099 above median) have shorter mean pfs_months than the rest of the cohort.", "kind": "novel"},
        {"id": "h21_2", "text": "The 'bad labs' indicator remains a negative independent predictor of pfs_months after adjustment for feature_078, feature_051, feature_038, and feature_057.", "kind": "refined"},
    ],
    [
        {"hypothesis_ids": ["h21_1"],
         "result_summary": f"_bad_labs=1 mean {ttest_badlabs[2]:.3f} (n={ttest_badlabs[4]}) vs 0 mean {ttest_badlabs[3]:.3f} (n={ttest_badlabs[5]}); diff {ttest_badlabs[0]:+.3f}, p={ttest_badlabs[1]:.3g}.",
         "p_value": ttest_badlabs[1], "effect_estimate": ttest_badlabs[0], "significant": ttest_badlabs[1] < 0.05},
        {"hypothesis_ids": ["h21_2"],
         "result_summary": f"Adjusted coef for _bad_labs = {b21:+.4f}, p={p21:.3g}.",
         "p_value": p21, "effect_estimate": b21, "significant": p21 < 0.05},
    ],
)

# ---------- Iteration 22: feature_007 (binary) sex-like — main + interaction with feature_051 ----------
ttest7 = ttest("feature_007")
m22 = lin_reg("pfs_months ~ feature_007 * feature_051")
p22 = float(m22.pvalues.get("feature_007:feature_051", float("nan")))
b22 = float(m22.params.get("feature_007:feature_051", float("nan")))
add_iter(
    22,
    [
        {"id": "h22_1", "text": "Mean pfs_months differs between feature_007 = 1 and feature_007 = 0 patients in the overall cohort.", "kind": "novel"},
        {"id": "h22_2", "text": "The effect of feature_051 = 1 (vs 0) on pfs_months differs between feature_007 = 1 and feature_007 = 0 patients (interaction term significant).", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h22_1"],
         "result_summary": f"feature_007=1 mean {ttest7[2]:.3f} vs 0 mean {ttest7[3]:.3f}; diff {ttest7[0]:+.3f}, p={ttest7[1]:.3g}.",
         "p_value": ttest7[1], "effect_estimate": ttest7[0], "significant": ttest7[1] < 0.05},
        {"hypothesis_ids": ["h22_2"],
         "result_summary": f"Interaction feature_007:feature_051 coef = {b22:+.4f}, p={p22:.3g}.",
         "p_value": p22, "effect_estimate": b22, "significant": p22 < 0.05},
    ],
)

# ---------- Iteration 23: race x insurance interaction ----------
m23 = lin_reg("pfs_months ~ C(feature_064) * C(feature_018)")
p23_terms = [k for k in m23.pvalues.index if ":" in k and "feature_064" in k and "feature_018" in k]
joint23 = m23.f_test(" = ".join(p23_terms) + " = 0") if p23_terms else None
# Black uninsured cell
ru_mean = float(y[(df["feature_064"] == "black") & (df["feature_018"] == "uninsured")].mean()) if ((df["feature_064"] == "black") & (df["feature_018"] == "uninsured")).sum() > 5 else None
wp_mean = float(y[(df["feature_064"] == "white") & (df["feature_018"] == "private")].mean())
add_iter(
    23,
    [
        {"id": "h23_1", "text": "The interaction feature_064 x feature_018 is jointly significant: race-related differences in pfs_months are not uniform across insurance categories.", "kind": "novel"},
        {"id": "h23_2", "text": "The mean pfs_months for feature_064 = black AND feature_018 = uninsured is shorter than for feature_064 = white AND feature_018 = private.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h23_1"], "code": "joint F-test on feature_064:feature_018 interaction terms",
         "result_summary": f"Joint F = {float(joint23.fvalue) if joint23 is not None else float('nan'):.3f}, p={float(joint23.pvalue) if joint23 is not None else float('nan'):.3g}.",
         "p_value": float(joint23.pvalue) if joint23 is not None else None,
         "effect_estimate": float(joint23.fvalue) if joint23 is not None else None,
         "significant": (float(joint23.pvalue) if joint23 is not None else 1) < 0.05},
        {"hypothesis_ids": ["h23_2"],
         "result_summary": f"black/uninsured mean {ru_mean if ru_mean is not None else 'NA'}; white/private mean {wp_mean:.3f}.",
         "p_value": None,
         "effect_estimate": (ru_mean - wp_mean) if ru_mean is not None else None,
         "significant": (ru_mean is not None) and (ru_mean < wp_mean)},
    ],
)

# ---------- Iteration 24: high-dimensional multivariable model with all 'top' predictors and key categories ----------
formula24 = ("pfs_months ~ feature_078 + C(feature_057) + feature_051 + feature_038 + feature_013 + feature_043 "
             "+ feature_109 + feature_067 + feature_092 + feature_099 + feature_009 + feature_006 "
             "+ C(feature_064) + C(feature_018)")
m24 = lin_reg(formula24)
b24 = m24.params.to_dict()
p24 = m24.pvalues.to_dict()
add_iter(
    24,
    [
        {"id": "h24_1", "text": "After adjusting for the top biomarker/treatment features (feature_078, feature_057, feature_051, feature_038, feature_013, feature_043, feature_109, feature_067, feature_092, feature_099, feature_009, feature_006) AND for feature_064 (race) and feature_018 (insurance), feature_064 = black is still associated with shorter pfs_months than feature_064 = white.", "kind": "refined"},
        {"id": "h24_2", "text": "After the same multivariable adjustment, feature_018 = uninsured is still associated with shorter pfs_months than feature_018 = private.", "kind": "refined"},
        {"id": "h24_3", "text": "feature_051 retains a clinically meaningful negative coefficient (more than -0.3 months) and significant p-value in this fully adjusted model.", "kind": "refined"},
    ],
    [
        {"hypothesis_ids": ["h24_1"],
         "result_summary": f"Adjusted coef C(feature_064)[T.black] = {b24.get('C(feature_064)[T.black]', float('nan')):+.4f}, p={p24.get('C(feature_064)[T.black]', float('nan')):.3g}.",
         "p_value": float(p24.get("C(feature_064)[T.black]", float("nan"))),
         "effect_estimate": float(b24.get("C(feature_064)[T.black]", float("nan"))),
         "significant": p24.get("C(feature_064)[T.black]", 1) < 0.05},
        {"hypothesis_ids": ["h24_2"],
         "result_summary": f"Adjusted coef C(feature_018)[T.uninsured] = {b24.get('C(feature_018)[T.uninsured]', float('nan')):+.4f}, p={p24.get('C(feature_018)[T.uninsured]', float('nan')):.3g}.",
         "p_value": float(p24.get("C(feature_018)[T.uninsured]", float("nan"))),
         "effect_estimate": float(b24.get("C(feature_018)[T.uninsured]", float("nan"))),
         "significant": p24.get("C(feature_018)[T.uninsured]", 1) < 0.05},
        {"hypothesis_ids": ["h24_3"],
         "result_summary": f"Adjusted coef feature_051 = {b24.get('feature_051', float('nan')):+.4f}, p={p24.get('feature_051', float('nan')):.3g}.",
         "p_value": float(p24.get("feature_051", float("nan"))),
         "effect_estimate": float(b24.get("feature_051", float("nan"))),
         "significant": p24.get("feature_051", 1) < 0.05},
    ],
)
print("M24 R^2:", m24.rsquared, "n=", int(m24.nobs))

# ---------- Iteration 25: secondary screen findings — feature_089, feature_102, feature_112, feature_028 ----------
ttest89 = ttest("feature_089")
ttest102 = ttest("feature_102")
ttest112 = ttest("feature_112")
r28, p28 = pearson("feature_028")
add_iter(
    25,
    [
        {"id": "h25_1", "text": "Patients with feature_089 = 1 have longer mean pfs_months than feature_089 = 0.", "kind": "novel"},
        {"id": "h25_2", "text": "Patients with feature_102 = 1 have shorter mean pfs_months than feature_102 = 0.", "kind": "novel"},
        {"id": "h25_3", "text": "Patients with feature_112 = 1 have shorter mean pfs_months than feature_112 = 0.", "kind": "novel"},
        {"id": "h25_4", "text": "feature_028 (continuous, range 0.09-3.53) is negatively correlated with pfs_months.", "kind": "novel"},
    ],
    [
        {"hypothesis_ids": ["h25_1"],
         "result_summary": f"feature_089=1 mean {ttest89[2]:.3f} vs 0 mean {ttest89[3]:.3f}; diff {ttest89[0]:+.3f}, p={ttest89[1]:.3g}.",
         "p_value": ttest89[1], "effect_estimate": ttest89[0], "significant": ttest89[1] < 0.05},
        {"hypothesis_ids": ["h25_2"],
         "result_summary": f"feature_102=1 mean {ttest102[2]:.3f} vs 0 mean {ttest102[3]:.3f}; diff {ttest102[0]:+.3f}, p={ttest102[1]:.3g}.",
         "p_value": ttest102[1], "effect_estimate": ttest102[0], "significant": ttest102[1] < 0.05},
        {"hypothesis_ids": ["h25_3"],
         "result_summary": f"feature_112=1 mean {ttest112[2]:.3f} vs 0 mean {ttest112[3]:.3f}; diff {ttest112[0]:+.3f}, p={ttest112[1]:.3g}.",
         "p_value": ttest112[1], "effect_estimate": ttest112[0], "significant": ttest112[1] < 0.05},
        {"hypothesis_ids": ["h25_4"],
         "result_summary": f"Pearson r={r28:.4f}, p={p28:.3g}. Small but significant negative correlation.",
         "p_value": p28, "effect_estimate": r28, "significant": p28 < 0.05},
    ],
)

def _coerce(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"not serializable: {type(o)}")


with open("_my_results.json", "w") as f:
    json.dump(results, f, indent=2, default=_coerce)

print("Total iterations:", len(results["iterations"]))
print("Total hypotheses:", sum(len(it["proposed_hypotheses"]) for it in results["iterations"]))
print("Total analyses:", sum(len(it["analyses"]) for it in results["iterations"]))
