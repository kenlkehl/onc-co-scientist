"""Multi-iteration hypothesis testing on ds001_crc.

Runs a series of statistical analyses, captures each as an
(hypothesis, analysis) record, and writes transcript.json.
"""
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DF = pd.read_parquet(HERE / "dataset.parquet")
OUTCOME = "pfs_months"

INTS = [c for c in DF.columns if DF[c].dtype == "int64" and c != "patient_id"]
BINARIES = [c for c in INTS if DF[c].nunique() == 2 and DF[c].min() == 0 and DF[c].max() == 1]
ORDINALS = [c for c in INTS if c not in BINARIES]
FLOATS = [c for c in DF.columns if DF[c].dtype == "float64" and c != OUTCOME]
CATS = ["feature_064", "feature_018"]


# ---------------------------------------------------------------------------
# Helpers


def fit_ols(formula: str):
    """Fit and return (params, pvalues, model)."""
    model = smf.ols(formula, data=DF).fit()
    return model


def two_group_ttest(col: str):
    """Welch's t-test of pfs_months by binary col. Effect = mean(1) - mean(0)."""
    a = DF.loc[DF[col] == 1, OUTCOME]
    b = DF.loc[DF[col] == 0, OUTCOME]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(p), float(a.mean()), float(b.mean()), int(a.size), int(b.size)


def pearson(col: str):
    r, p = stats.pearsonr(DF[col], DF[OUTCOME])
    return float(r), float(p)


def spearman(col: str):
    r, p = stats.spearmanr(DF[col], DF[OUTCOME])
    return float(r), float(p)


def linreg_slope(col: str):
    """Slope of pfs_months ~ col (per unit of col)."""
    x = sm.add_constant(DF[col].astype(float))
    res = sm.OLS(DF[OUTCOME], x).fit()
    return float(res.params[col]), float(res.pvalues[col])


# ---------------------------------------------------------------------------
# Build iterations

iterations = []


def add_iter(idx, hyps, analyses):
    iterations.append({"index": idx, "proposed_hypotheses": hyps, "analyses": analyses})


# Iteration 1: univariate associations of every binary feature with pfs_months
# Each binary gets its own per-feature hypothesis (self-contained, names specific column).
hyps = []
analyses = []
binary_results = {}
for i, c in enumerate(BINARIES, start=1):
    diff, p, m1, m0, n1, n0 = two_group_ttest(c)
    binary_results[c] = (diff, p, m1, m0, n1, n0)
    hid = f"h1_{c}"
    hyps.append({
        "id": hid,
        "text": (
            f"Mean pfs_months differs between patients with {c}=1 and patients "
            f"with {c}=0 (two-sided test of association of binary {c} with progression-free survival)."
        ),
        "kind": "novel",
    })
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"stats.ttest_ind(df.loc[df['{c}']==1,'pfs_months'], df.loc[df['{c}']==0,'pfs_months'], equal_var=False)",
        "result_summary": (
            f"{c}: mean pfs_months = {m1:.3f} (n={n1}) when {c}=1 vs "
            f"{m0:.3f} (n={n0}) when {c}=0; diff = {diff:+.3f} months, p = {p:.3g}."
        ),
        "p_value": p,
        "effect_estimate": diff,
        "significant": bool(p < 0.05),
    })
add_iter(1, hyps, analyses)

# Iteration 2: continuous (float) features — Pearson correlation with pfs_months
# Per-feature hypothesis for each float column.
hyps = []
analyses = []
for c in FLOATS:
    r, p = pearson(c)
    hid = f"h2_{c}"
    hyps.append({
        "id": hid,
        "text": (
            f"Continuous feature {c} is linearly correlated with pfs_months "
            "(Pearson r ≠ 0). The signed correlation will show whether higher values "
            "predict longer (positive r) or shorter (negative r) progression-free survival."
        ),
        "kind": "novel",
    })
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"stats.pearsonr(df['{c}'], df['pfs_months'])",
        "result_summary": f"Pearson r({c}, pfs_months) = {r:+.4f}, p = {p:.3g}.",
        "p_value": p,
        "effect_estimate": r,
        "significant": bool(p < 0.05),
    })
add_iter(2, hyps, analyses)

# Iteration 3: ordinal small-int features — Spearman correlation
hyps = []
analyses = []
for c in ORDINALS:
    r, p = spearman(c)
    hid = f"h3_{c}"
    hyps.append({
        "id": hid,
        "text": (
            f"Ordinal feature {c} is monotonically associated with pfs_months "
            "(Spearman rho ≠ 0); the sign indicates whether higher levels of the ordinal "
            "predict longer (positive) or shorter (negative) progression-free survival."
        ),
        "kind": "novel",
    })
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"stats.spearmanr(df['{c}'], df['pfs_months'])",
        "result_summary": f"Spearman rho({c}, pfs_months) = {r:+.4f}, p = {p:.3g}.",
        "p_value": p,
        "effect_estimate": r,
        "significant": bool(p < 0.05),
    })
add_iter(3, hyps, analyses)

# Iteration 4: categorical features — ANOVA on pfs_months by race / insurance
hyps = [
    {
        "id": "h4a",
        "text": (
            "Unadjusted mean pfs_months differs across racial groups in feature_064 "
            "(levels: white, hispanic, black, asian, other); a one-way ANOVA will reject "
            "equality of group means."
        ),
        "kind": "novel",
    },
    {
        "id": "h4b",
        "text": (
            "Unadjusted mean pfs_months differs across insurance categories in feature_018 "
            "(levels: medicare, private, medicaid, uninsured); a one-way ANOVA will reject "
            "equality of group means."
        ),
        "kind": "novel",
    },
]
analyses = []
for c, hid in zip(CATS, ["h4a", "h4b"]):
    groups = [g[OUTCOME].values for _, g in DF.groupby(c)]
    f, p = stats.f_oneway(*groups)
    means = DF.groupby(c)[OUTCOME].mean().to_dict()
    overall = DF[OUTCOME].mean()
    spread = max(means.values()) - min(means.values())
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"stats.f_oneway(*[g['pfs_months'].values for _,g in df.groupby('{c}')])",
        "result_summary": (
            f"One-way ANOVA on pfs_months across {c}: F={f:.3f}, p={p:.3g}. "
            f"Group means: {means}. Range across groups = {spread:.3f}."
        ),
        "p_value": float(p),
        "effect_estimate": float(spread),
        "significant": bool(p < 0.05),
    })
add_iter(4, hyps, analyses)

# Identify top binary features by signed effect (and significance) for follow-up
sig_binaries = sorted(
    [(c, *binary_results[c]) for c in BINARIES if binary_results[c][1] < 0.05],
    key=lambda t: abs(t[1]),
    reverse=True,
)
TOP_BINARIES = [t[0] for t in sig_binaries[:15]]

# Iteration 5: top binary effects re-tested with Mann-Whitney to confirm robustness.
# Each top-15 binary gets a directional refined hypothesis (using sign from iteration 1).
hyps = []
analyses = []
for c in TOP_BINARIES:
    diff_iter1 = binary_results[c][0]
    direction = "shorter" if diff_iter1 < 0 else "longer"
    hid = f"h5_{c}"
    hyps.append({
        "id": hid,
        "text": (
            f"Patients with {c}=1 have {direction} pfs_months than patients with {c}=0; "
            f"this signed direction (from iter-1 t-test) is confirmed by a non-parametric "
            f"Mann-Whitney rank-sum test."
        ),
        "kind": "refined",
    })
    a = DF.loc[DF[c] == 1, OUTCOME]
    b = DF.loc[DF[c] == 0, OUTCOME]
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    diff = float(a.median() - b.median())
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"stats.mannwhitneyu(df.loc[df['{c}']==1,'pfs_months'], df.loc[df['{c}']==0,'pfs_months'])",
        "result_summary": (
            f"{c}: median PFS = {a.median():.3f} when {c}=1 vs {b.median():.3f} "
            f"when {c}=0; Mann-Whitney U={u:.0f}, p={p:.3g}."
        ),
        "p_value": float(p),
        "effect_estimate": diff,
        "significant": bool(p < 0.05),
    })
add_iter(5, hyps, analyses)

# Iteration 6: multivariable OLS regression with all binary features.
# One refined hypothesis per binary (does it retain signed effect after mutual adjustment?).
formula = OUTCOME + " ~ " + " + ".join(BINARIES)
m = fit_ols(formula)
hyps = []
analyses = []
for c in BINARIES:
    coef = float(m.params[c])
    p = float(m.pvalues[c])
    unadj_diff = binary_results[c][0]
    pred_dir = "negative" if unadj_diff < 0 else "positive"
    hid = f"h6_{c}"
    hyps.append({
        "id": hid,
        "text": (
            f"In a multivariable OLS of pfs_months on all 80 binary features, the adjusted "
            f"coefficient for {c} is non-zero and has the same sign as its unadjusted "
            f"mean-difference ({pred_dir}); i.e., {c}'s effect on pfs_months is not fully "
            "explained by confounding from other binaries."
        ),
        "kind": "refined",
    })
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ <all 80 binaries>', df).fit().params['{c}']",
        "result_summary": (
            f"Adjusted coef for {c}: beta = {coef:+.4f} months, p = {p:.3g}."
        ),
        "p_value": p,
        "effect_estimate": coef,
        "significant": bool(p < 0.05),
    })
add_iter(6, hyps, analyses)

# Track which binaries are significant in adjusted model — these are likely "real"
adj_sig = [c for c in BINARIES if m.pvalues[c] < 0.05]
adj_signs = {c: float(m.params[c]) for c in BINARIES}

# Iteration 7: are continuous features still associated after adjustment for binaries?
# One refined hypothesis per float feature.
formula7 = OUTCOME + " ~ " + " + ".join(BINARIES + FLOATS)
m7 = fit_ols(formula7)
hyps = []
analyses = []
for c in FLOATS:
    coef = float(m7.params[c])
    p = float(m7.pvalues[c])
    iter2_r, _ = pearson(c)
    pred_dir = "negative" if iter2_r < 0 else "positive"
    hid = f"h7_{c}"
    hyps.append({
        "id": hid,
        "text": (
            f"In a multivariable OLS of pfs_months on all 80 binaries plus all 35 "
            f"continuous features, the adjusted slope for {c} is non-zero and has the "
            f"same sign as its univariate Pearson correlation ({pred_dir}); i.e., {c}'s "
            "effect on pfs_months is independent of the binary covariate set."
        ),
        "kind": "refined",
    })
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ <80 binaries + 35 floats>', df).fit().params['{c}']",
        "result_summary": f"Adjusted slope per 1-unit increase in {c}: {coef:+.5f} months, p={p:.3g}.",
        "p_value": p,
        "effect_estimate": coef,
        "significant": bool(p < 0.05),
    })
add_iter(7, hyps, analyses)

# Persist iterations 1-7 partial results to a sidecar JSON for use by later iterations
inter = {
    "binary_results": binary_results,
    "adj_signs": adj_signs,
    "adj_sig": adj_sig,
}

# Iteration 8: Age (feature_078) effect, with quadratic term
hyps = [{
    "id": "h8a",
    "text": "feature_078 (likely age, range 30-90) is negatively associated with pfs_months: older patients have shorter PFS.",
    "kind": "novel",
}, {
    "id": "h8b",
    "text": "The age-PFS relationship is non-linear: a quadratic term in feature_078 will be statistically significant.",
    "kind": "novel",
}]
m8 = smf.ols("pfs_months ~ feature_078 + I(feature_078**2)", data=DF).fit()
analyses = [
    {
        "hypothesis_ids": ["h8a"],
        "code": "smf.ols('pfs_months ~ feature_078', df).fit()",
        "result_summary": (
            f"Linear: beta(feature_078) = {smf.ols('pfs_months ~ feature_078', DF).fit().params['feature_078']:+.5f} months/year, "
            f"p={smf.ols('pfs_months ~ feature_078', DF).fit().pvalues['feature_078']:.3g}."
        ),
        "p_value": float(smf.ols("pfs_months ~ feature_078", DF).fit().pvalues["feature_078"]),
        "effect_estimate": float(smf.ols("pfs_months ~ feature_078", DF).fit().params["feature_078"]),
        "significant": bool(smf.ols("pfs_months ~ feature_078", DF).fit().pvalues["feature_078"] < 0.05),
    },
    {
        "hypothesis_ids": ["h8b"],
        "code": "smf.ols('pfs_months ~ feature_078 + I(feature_078**2)', df).fit()",
        "result_summary": (
            f"Quadratic term beta = {m8.params['I(feature_078 ** 2)']:+.6f}, p={m8.pvalues['I(feature_078 ** 2)']:.3g}; "
            f"linear term beta = {m8.params['feature_078']:+.5f}, p={m8.pvalues['feature_078']:.3g}."
        ),
        "p_value": float(m8.pvalues["I(feature_078 ** 2)"]),
        "effect_estimate": float(m8.params["I(feature_078 ** 2)"]),
        "significant": bool(m8.pvalues["I(feature_078 ** 2)"] < 0.05),
    },
]
add_iter(8, hyps, analyses)

# Save what we have so far in case of interruption.
with open(HERE / "_partial_iters_1_8.json", "w") as f:
    json.dump(iterations, f, default=float, indent=2)
print(f"Iterations 1-8 done. Total iterations so far: {len(iterations)}")

# Identify the strongest two adjusted-significant binary features for interaction
ranked_adj = sorted(
    [(c, m.params[c], m.pvalues[c]) for c in BINARIES if m.pvalues[c] < 0.05],
    key=lambda t: abs(t[1]),
    reverse=True,
)
TOP2 = [t[0] for t in ranked_adj[:2]] if len(ranked_adj) >= 2 else BINARIES[:2]
print("Top adjusted binaries by |beta|:", ranked_adj[:5])

# Iteration 9: do the top binaries interact?
hyps = [{
    "id": "h9",
    "text": (
        f"The two strongest adjusted-significant binary predictors ({TOP2[0]} and {TOP2[1]}) "
        "interact: their joint effect on pfs_months differs from the sum of their main effects."
    ),
    "kind": "novel",
}]
formula9 = f"pfs_months ~ {TOP2[0]} * {TOP2[1]}"
m9 = smf.ols(formula9, data=DF).fit()
inter_term = f"{TOP2[0]}:{TOP2[1]}"
analyses = [{
    "hypothesis_ids": ["h9"],
    "code": f"smf.ols('{formula9}', df).fit()",
    "result_summary": (
        f"Interaction beta({inter_term}) = {m9.params[inter_term]:+.4f} months, p={m9.pvalues[inter_term]:.3g}. "
        f"Main effects: {TOP2[0]}={m9.params[TOP2[0]]:+.4f} (p={m9.pvalues[TOP2[0]]:.3g}); "
        f"{TOP2[1]}={m9.params[TOP2[1]]:+.4f} (p={m9.pvalues[TOP2[1]]:.3g})."
    ),
    "p_value": float(m9.pvalues[inter_term]),
    "effect_estimate": float(m9.params[inter_term]),
    "significant": bool(m9.pvalues[inter_term] < 0.05),
}]
add_iter(9, hyps, analyses)

# Iteration 10: sex/race subgroup heterogeneity — does any binary's effect vary by feature_064?
# Pick the single strongest adjusted binary as a probe
PROBE = TOP2[0]
hyps = [{
    "id": "h10",
    "text": (
        f"The effect of {PROBE} on pfs_months varies across racial groups (feature_064): "
        "the {PROBE} x feature_064 interaction will be statistically significant."
    ).format(PROBE=PROBE),
    "kind": "novel",
}]
formula10 = f"pfs_months ~ C(feature_064) * {PROBE}"
m10 = smf.ols(formula10, data=DF).fit()
# anova for the interaction set
anova10 = sm.stats.anova_lm(smf.ols(f"pfs_months ~ C(feature_064) + {PROBE}", DF).fit(), m10)
p10 = float(anova10["Pr(>F)"].iloc[1])
# Use the largest-magnitude interaction beta as the effect estimate
inter_terms = [k for k in m10.params.index if ":" in k]
biggest = max(inter_terms, key=lambda k: abs(m10.params[k]))
analyses = [{
    "hypothesis_ids": ["h10"],
    "code": f"sm.stats.anova_lm(smf.ols('pfs_months ~ C(feature_064)+{PROBE}',df).fit(), smf.ols('{formula10}',df).fit())",
    "result_summary": (
        f"Likelihood-ratio-style ANOVA p for the C(feature_064):{PROBE} interaction = {p10:.3g}. "
        f"Largest interaction term: {biggest} = {m10.params[biggest]:+.4f}."
    ),
    "p_value": p10,
    "effect_estimate": float(m10.params[biggest]),
    "significant": bool(p10 < 0.05),
}]
add_iter(10, hyps, analyses)

# Save state
with open(HERE / "_partial_iters_1_10.json", "w") as f:
    json.dump(iterations, f, default=float, indent=2)
print("Iterations 1-10 done.")

# Iteration 11: Insurance subgroup heterogeneity for PROBE
hyps = [{
    "id": "h11",
    "text": (
        f"The effect of {PROBE} on pfs_months differs by insurance type (feature_018): "
        "patients on medicaid or uninsured may show different treatment-outcome patterns "
        "than private-insurance or medicare patients."
    ),
    "kind": "novel",
}]
formula11 = f"pfs_months ~ C(feature_018) * {PROBE}"
m11 = smf.ols(formula11, data=DF).fit()
anova11 = sm.stats.anova_lm(smf.ols(f"pfs_months ~ C(feature_018) + {PROBE}", DF).fit(), m11)
p11 = float(anova11["Pr(>F)"].iloc[1])
inter_terms = [k for k in m11.params.index if ":" in k]
biggest = max(inter_terms, key=lambda k: abs(m11.params[k]))
analyses = [{
    "hypothesis_ids": ["h11"],
    "code": f"smf.ols('{formula11}', df).fit()",
    "result_summary": (
        f"ANOVA p for C(feature_018):{PROBE} interaction = {p11:.3g}. "
        f"Largest interaction term: {biggest} = {m11.params[biggest]:+.4f}."
    ),
    "p_value": p11,
    "effect_estimate": float(m11.params[biggest]),
    "significant": bool(p11 < 0.05),
}]
add_iter(11, hyps, analyses)

# Iteration 12: does PROBE's effect depend on age (feature_078)?
hyps = [{
    "id": "h12",
    "text": (
        f"The effect of {PROBE} on pfs_months is modified by age (feature_078): "
        f"the {PROBE} x feature_078 interaction term will be non-zero."
    ),
    "kind": "novel",
}]
formula12 = f"pfs_months ~ {PROBE} * feature_078"
m12 = smf.ols(formula12, data=DF).fit()
inter_term = f"{PROBE}:feature_078"
analyses = [{
    "hypothesis_ids": ["h12"],
    "code": f"smf.ols('{formula12}', df).fit()",
    "result_summary": (
        f"Interaction beta({inter_term}) = {m12.params[inter_term]:+.5f} months/year, "
        f"p={m12.pvalues[inter_term]:.3g}. Main {PROBE} beta={m12.params[PROBE]:+.4f}."
    ),
    "p_value": float(m12.pvalues[inter_term]),
    "effect_estimate": float(m12.params[inter_term]),
    "significant": bool(m12.pvalues[inter_term] < 0.05),
}]
add_iter(12, hyps, analyses)

# Iteration 13: ordinal feature_025 — strong probable performance-status / stage signal
hyps = [{
    "id": "h13",
    "text": (
        "Ordinal feature_025 (values 0-4) shows a monotone negative association with pfs_months: "
        "higher levels predict shorter PFS, with mean PFS decreasing roughly linearly across levels."
    ),
    "kind": "refined",
}]
group_means = DF.groupby("feature_025")[OUTCOME].agg(["mean", "count"])
m13 = smf.ols("pfs_months ~ feature_025", DF).fit()
analyses = [{
    "hypothesis_ids": ["h13"],
    "code": "df.groupby('feature_025')['pfs_months'].mean(); smf.ols('pfs_months ~ feature_025', df).fit()",
    "result_summary": (
        f"Per-level mean PFS: {group_means['mean'].round(3).to_dict()}; counts: {group_means['count'].to_dict()}. "
        f"Linear slope = {m13.params['feature_025']:+.4f} months/level, p={m13.pvalues['feature_025']:.3g}."
    ),
    "p_value": float(m13.pvalues["feature_025"]),
    "effect_estimate": float(m13.params["feature_025"]),
    "significant": bool(m13.pvalues["feature_025"] < 0.05),
}]
add_iter(13, hyps, analyses)

# Iteration 14: feature_014 (likely hemoglobin) — patients with low hemoglobin (<10) vs >=10
hyps = [{
    "id": "h14",
    "text": (
        "Patients with feature_014 < 10 (low-hemoglobin-like) have shorter mean pfs_months "
        "than patients with feature_014 >= 10."
    ),
    "kind": "novel",
}]
low = DF[DF["feature_014"] < 10][OUTCOME]
high = DF[DF["feature_014"] >= 10][OUTCOME]
t14, p14 = stats.ttest_ind(low, high, equal_var=False)
analyses = [{
    "hypothesis_ids": ["h14"],
    "code": "stats.ttest_ind(df[df.feature_014<10]['pfs_months'], df[df.feature_014>=10]['pfs_months'])",
    "result_summary": (
        f"Mean PFS: low-feature_014 (<10, n={low.size}) = {low.mean():.3f} vs high (>=10, n={high.size}) = {high.mean():.3f}. "
        f"Diff = {low.mean()-high.mean():+.3f} months, p={p14:.3g}."
    ),
    "p_value": float(p14),
    "effect_estimate": float(low.mean() - high.mean()),
    "significant": bool(p14 < 0.05),
}]
add_iter(14, hyps, analyses)

# Iteration 15: feature_092 (likely albumin) — low albumin (<3.5) vs >=3.5
hyps = [{
    "id": "h15",
    "text": (
        "Patients with feature_092 < 3.5 (low-albumin-like) have shorter mean pfs_months than patients with feature_092 >= 3.5."
    ),
    "kind": "novel",
}]
low = DF[DF["feature_092"] < 3.5][OUTCOME]
high = DF[DF["feature_092"] >= 3.5][OUTCOME]
t15, p15 = stats.ttest_ind(low, high, equal_var=False)
analyses = [{
    "hypothesis_ids": ["h15"],
    "code": "stats.ttest_ind(df[df.feature_092<3.5]['pfs_months'], df[df.feature_092>=3.5]['pfs_months'])",
    "result_summary": (
        f"Mean PFS: low-feature_092 (<3.5, n={low.size}) = {low.mean():.3f} vs high (>=3.5, n={high.size}) = {high.mean():.3f}. "
        f"Diff = {low.mean()-high.mean():+.3f} months, p={p15:.3g}."
    ),
    "p_value": float(p15),
    "effect_estimate": float(low.mean() - high.mean()),
    "significant": bool(p15 < 0.05),
}]
add_iter(15, hyps, analyses)

# Save state
with open(HERE / "_partial_iters_1_15.json", "w") as f:
    json.dump(iterations, f, default=float, indent=2)
print("Iterations 1-15 done.")

# Iteration 16: Among the 35 floats, which retain independent association after Bonferroni adjustment?
hyps = [{
    "id": "h16",
    "text": (
        "After Bonferroni correction for 35 simultaneous tests of continuous features, a "
        "subset (>=5 features) retain genome-wide-style significance (p < 0.05/35 ≈ 0.00143) "
        "for association with pfs_months in univariate Pearson tests."
    ),
    "kind": "refined",
}]
threshold = 0.05 / len(FLOATS)
analyses = []
sig_count = 0
sig_list = []
for c in FLOATS:
    r, p = pearson(c)
    if p < threshold:
        sig_count += 1
        sig_list.append((c, r, p))
analyses.append({
    "hypothesis_ids": ["h16"],
    "code": "for c in FLOATS: stats.pearsonr(df[c], df['pfs_months']); compare to 0.05/35",
    "result_summary": (
        f"Bonferroni threshold = {threshold:.4g}. {sig_count} of {len(FLOATS)} float features "
        f"clear it. Top 10 (signed r): "
        + ", ".join([f"{c}={r:+.3f} (p={p:.2g})" for c, r, p in sorted(sig_list, key=lambda x: abs(x[1]), reverse=True)[:10]])
    ),
    "p_value": float(threshold),  # threshold itself
    "effect_estimate": float(sig_count),
    "significant": bool(sig_count >= 5),
})
add_iter(16, hyps, analyses)

# Iteration 17: Race main effect on PFS, adjusted for all binaries+floats
hyps = [{
    "id": "h17",
    "text": (
        "After adjusting for all 80 binary features and 35 continuous features, mean "
        "pfs_months still differs across racial groups (feature_064)."
    ),
    "kind": "refined",
}]
formula17_full = f"pfs_months ~ C(feature_064) + " + " + ".join(BINARIES + FLOATS)
formula17_red = f"pfs_months ~ " + " + ".join(BINARIES + FLOATS)
m17 = smf.ols(formula17_full, DF).fit()
m17r = smf.ols(formula17_red, DF).fit()
anova17 = sm.stats.anova_lm(m17r, m17)
p17 = float(anova17["Pr(>F)"].iloc[1])
race_coefs = {k: float(m17.params[k]) for k in m17.params.index if k.startswith("C(feature_064)")}
biggest_race = max(race_coefs, key=lambda k: abs(race_coefs[k])) if race_coefs else None
analyses = [{
    "hypothesis_ids": ["h17"],
    "code": "compare anova_lm(<full + race> vs <full - race>)",
    "result_summary": (
        f"ANOVA p for race after full adjustment = {p17:.3g}. Adjusted race coefs: "
        + ", ".join([f"{k.split('.')[-1].rstrip(']')}={v:+.4f}" for k, v in race_coefs.items()])
    ),
    "p_value": p17,
    "effect_estimate": float(race_coefs[biggest_race]) if biggest_race else 0.0,
    "significant": bool(p17 < 0.05),
}]
add_iter(17, hyps, analyses)

# Iteration 18: insurance main effect adjusted (parallel to iter 17)
hyps = [{
    "id": "h18",
    "text": (
        "After adjusting for all 80 binary features and 35 continuous features, mean "
        "pfs_months still differs across insurance categories (feature_018)."
    ),
    "kind": "refined",
}]
formula18_full = f"pfs_months ~ C(feature_018) + " + " + ".join(BINARIES + FLOATS)
m18 = smf.ols(formula18_full, DF).fit()
m18r = smf.ols(formula17_red, DF).fit()  # same reduced model
anova18 = sm.stats.anova_lm(m18r, m18)
p18 = float(anova18["Pr(>F)"].iloc[1])
ins_coefs = {k: float(m18.params[k]) for k in m18.params.index if k.startswith("C(feature_018)")}
biggest_ins = max(ins_coefs, key=lambda k: abs(ins_coefs[k])) if ins_coefs else None
analyses = [{
    "hypothesis_ids": ["h18"],
    "code": "compare anova_lm(<full + insurance> vs <full>)",
    "result_summary": (
        f"ANOVA p for insurance after full adjustment = {p18:.3g}. Adjusted insurance coefs: "
        + ", ".join([f"{k.split('.')[-1].rstrip(']')}={v:+.4f}" for k, v in ins_coefs.items()])
    ),
    "p_value": p18,
    "effect_estimate": float(ins_coefs[biggest_ins]) if biggest_ins else 0.0,
    "significant": bool(p18 < 0.05),
}]
add_iter(18, hyps, analyses)

with open(HERE / "_partial_iters_1_18.json", "w") as f:
    json.dump(iterations, f, default=float, indent=2)
print("Iterations 1-18 done.")

# Iteration 19: prognostic score — sum of unfavorable indicators predicts pfs_months
hyps = [{
    "id": "h19",
    "text": (
        "A simple integer count of 'unfavorable' adjusted-significant binary features "
        "(those with negative beta in iter-6 multivariable model) is itself negatively "
        "associated with pfs_months in a linear regression."
    ),
    "kind": "novel",
}]
neg_binaries = [c for c in adj_sig if adj_signs[c] < 0]
DF["_burden"] = DF[neg_binaries].sum(axis=1)
m19 = smf.ols("pfs_months ~ _burden", DF).fit()
analyses = [{
    "hypothesis_ids": ["h19"],
    "code": "df['_burden'] = df[neg_adj_binaries].sum(1); smf.ols('pfs_months ~ _burden', df).fit()",
    "result_summary": (
        f"Burden score built from {len(neg_binaries)} 'unfavorable' binaries. "
        f"Slope = {m19.params['_burden']:+.4f} months per +1 unfavorable feature, "
        f"p = {m19.pvalues['_burden']:.3g}; R² = {m19.rsquared:.4f}."
    ),
    "p_value": float(m19.pvalues["_burden"]),
    "effect_estimate": float(m19.params["_burden"]),
    "significant": bool(m19.pvalues["_burden"] < 0.05),
}]
add_iter(19, hyps, analyses)

# Iteration 20: best k=10 binaries' joint R² compared to noise-level baseline
hyps = [{
    "id": "h20",
    "text": (
        "The top 10 binary features by adjusted-effect magnitude (from iter 6) jointly "
        "explain a non-trivial fraction of variance in pfs_months (R² > 0.05)."
    ),
    "kind": "refined",
}]
ranked = sorted([(c, abs(m.params[c]), m.params[c], m.pvalues[c]) for c in BINARIES],
                key=lambda t: t[1], reverse=True)
top10 = [t[0] for t in ranked[:10]]
m20 = smf.ols("pfs_months ~ " + " + ".join(top10), DF).fit()
analyses = [{
    "hypothesis_ids": ["h20"],
    "code": f"smf.ols('pfs_months ~ {' + '.join(top10)}', df).fit().rsquared",
    "result_summary": (
        f"Top-10 binaries: {top10}. Joint OLS R² = {m20.rsquared:.4f}, F-test p = {m20.f_pvalue:.3g}."
    ),
    "p_value": float(m20.f_pvalue),
    "effect_estimate": float(m20.rsquared),
    "significant": bool(m20.rsquared > 0.05 and m20.f_pvalue < 0.05),
}]
add_iter(20, hyps, analyses)

# Iteration 21: probe interaction of top continuous (feature_092 ~ albumin-like) and top binary PROBE
top_float = max(FLOATS, key=lambda c: abs(stats.pearsonr(DF[c], DF[OUTCOME])[0]))
hyps = [{
    "id": "h21",
    "text": (
        f"The effect of binary {PROBE} on pfs_months is modified by the most-correlated "
        f"continuous feature ({top_float}): a {PROBE} x {top_float} interaction term is significant."
    ),
    "kind": "novel",
}]
formula21 = f"pfs_months ~ {PROBE} * {top_float}"
m21 = smf.ols(formula21, DF).fit()
inter_term21 = f"{PROBE}:{top_float}"
analyses = [{
    "hypothesis_ids": ["h21"],
    "code": f"smf.ols('{formula21}', df).fit()",
    "result_summary": (
        f"Interaction beta({inter_term21}) = {m21.params[inter_term21]:+.5f}, "
        f"p = {m21.pvalues[inter_term21]:.3g}. Main {PROBE} beta = {m21.params[PROBE]:+.4f}; "
        f"main {top_float} slope = {m21.params[top_float]:+.5f}."
    ),
    "p_value": float(m21.pvalues[inter_term21]),
    "effect_estimate": float(m21.params[inter_term21]),
    "significant": bool(m21.pvalues[inter_term21] < 0.05),
}]
add_iter(21, hyps, analyses)

# Iteration 22: subgroup PFS — does feature_025 (probable performance status / stage) modify many binaries' effects?
hyps = [{
    "id": "h22",
    "text": (
        f"The effect of {PROBE} on pfs_months differs across levels of feature_025 "
        "(ordinal 0-4, likely a stage / performance indicator)."
    ),
    "kind": "novel",
}]
formula22 = f"pfs_months ~ C(feature_025) * {PROBE}"
m22 = smf.ols(formula22, DF).fit()
m22r = smf.ols(f"pfs_months ~ C(feature_025) + {PROBE}", DF).fit()
anova22 = sm.stats.anova_lm(m22r, m22)
p22 = float(anova22["Pr(>F)"].iloc[1])
inter_terms = [k for k in m22.params.index if ":" in k]
biggest = max(inter_terms, key=lambda k: abs(m22.params[k]))
analyses = [{
    "hypothesis_ids": ["h22"],
    "code": f"sm.stats.anova_lm(<additive>, smf.ols('{formula22}',df).fit())",
    "result_summary": (
        f"ANOVA p for C(feature_025):{PROBE} interaction = {p22:.3g}. "
        f"Largest interaction term: {biggest} = {m22.params[biggest]:+.4f}."
    ),
    "p_value": p22,
    "effect_estimate": float(m22.params[biggest]),
    "significant": bool(p22 < 0.05),
}]
add_iter(22, hyps, analyses)

# Iteration 23: top continuous interaction with feature_025
hyps = [{
    "id": "h23",
    "text": (
        f"The slope of pfs_months on {top_float} differs across levels of feature_025 "
        "(stage/performance-like ordinal)."
    ),
    "kind": "novel",
}]
formula23 = f"pfs_months ~ C(feature_025) * {top_float}"
m23 = smf.ols(formula23, DF).fit()
m23r = smf.ols(f"pfs_months ~ C(feature_025) + {top_float}", DF).fit()
anova23 = sm.stats.anova_lm(m23r, m23)
p23 = float(anova23["Pr(>F)"].iloc[1])
inter_terms = [k for k in m23.params.index if ":" in k]
biggest = max(inter_terms, key=lambda k: abs(m23.params[k]))
analyses = [{
    "hypothesis_ids": ["h23"],
    "code": f"sm.stats.anova_lm(<additive>, smf.ols('{formula23}',df).fit())",
    "result_summary": (
        f"ANOVA p for C(feature_025):{top_float} interaction = {p23:.3g}. "
        f"Largest interaction term: {biggest} = {m23.params[biggest]:+.5f}."
    ),
    "p_value": p23,
    "effect_estimate": float(m23.params[biggest]),
    "significant": bool(p23 < 0.05),
}]
add_iter(23, hyps, analyses)

# Iteration 24: race-stratified mean PFS — does any racial group have markedly shorter PFS even after adjustment?
hyps = [{
    "id": "h24",
    "text": (
        "Black or hispanic patients (feature_064) have shorter mean unadjusted pfs_months "
        "than white patients, consistent with disparities in real-world cohorts; the gap "
        "shrinks but does not disappear after adjustment for the binaries+floats covariate set."
    ),
    "kind": "refined",
}]
unadj_means = DF.groupby("feature_064")[OUTCOME].mean().to_dict()
unadj_white = unadj_means.get("white", float("nan"))
unadj_diff_b = unadj_means.get("black", float("nan")) - unadj_white
unadj_diff_h = unadj_means.get("hispanic", float("nan")) - unadj_white
# Race coefs from iter-17 adjusted model (m17): coefs are vs reference (alphabetical → asian is ref usually)
# Recompute with explicit treatment(reference="white") to compare cleanly.
formula24 = "pfs_months ~ C(feature_064, Treatment(reference='white')) + " + " + ".join(BINARIES + FLOATS)
m24 = smf.ols(formula24, DF).fit()
adj_coefs = {k: float(m24.params[k]) for k in m24.params.index if "feature_064" in k}
adj_b_key = next((k for k in adj_coefs if "black" in k), None)
adj_h_key = next((k for k in adj_coefs if "hispanic" in k), None)
adj_b = float(adj_coefs[adj_b_key]) if adj_b_key else float("nan")
adj_h = float(adj_coefs[adj_h_key]) if adj_h_key else float("nan")
adj_b_p = float(m24.pvalues[adj_b_key]) if adj_b_key else float("nan")
adj_h_p = float(m24.pvalues[adj_h_key]) if adj_h_key else float("nan")
analyses = [{
    "hypothesis_ids": ["h24"],
    "code": "smf.ols('pfs_months ~ C(feature_064, Treatment(reference=white)) + <covariates>', df).fit()",
    "result_summary": (
        f"Unadjusted means: {unadj_means}. Unadjusted black-white = {unadj_diff_b:+.3f}, "
        f"hispanic-white = {unadj_diff_h:+.3f}. "
        f"Adjusted (vs white): black = {adj_b:+.4f} (p={adj_b_p:.3g}); "
        f"hispanic = {adj_h:+.4f} (p={adj_h_p:.3g})."
    ),
    "p_value": adj_b_p,
    "effect_estimate": float(adj_b),
    "significant": bool(adj_b_p < 0.05) if not math.isnan(adj_b_p) else None,
}]
add_iter(24, hyps, analyses)

# Iteration 25: synthesis — overall full model R² + how many binaries still significant
hyps = [{
    "id": "h25",
    "text": (
        "A full additive multivariable OLS model (80 binaries + 35 continuous + race + "
        "insurance + 6 ordinals) explains a modest but non-trivial fraction (>5%) of "
        "variance in pfs_months, with at least 20 features retaining adjusted significance, "
        "supporting a multifactorial determination of progression-free survival in this cohort."
    ),
    "kind": "refined",
}]
formula25 = (
    "pfs_months ~ C(feature_064) + C(feature_018) + "
    + " + ".join(BINARIES + FLOATS + ORDINALS)
)
m25 = smf.ols(formula25, DF).fit()
n_sig_total = int((m25.pvalues < 0.05).sum())
analyses = [{
    "hypothesis_ids": ["h25"],
    "code": "smf.ols('pfs_months ~ C(feature_064) + C(feature_018) + <80 bins> + <35 floats> + <6 ordinals>', df).fit()",
    "result_summary": (
        f"Full additive model: n = {int(m25.nobs)}, R² = {m25.rsquared:.4f}, adj-R² = {m25.rsquared_adj:.4f}, "
        f"F-test p = {m25.f_pvalue:.3g}. Number of coefficients with p<0.05 = {n_sig_total} "
        f"(out of {len(m25.params)} including intercept and dummy levels)."
    ),
    "p_value": float(m25.f_pvalue),
    "effect_estimate": float(m25.rsquared),
    "significant": bool(m25.rsquared > 0.05 and m25.f_pvalue < 0.05),
}]
add_iter(25, hyps, analyses)


# ---------------------------------------------------------------------------
# Build transcript

transcript = {
    "dataset_id": "ds001",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-manual@1",
    "max_iterations": 25,
    "iterations": iterations,
}

with open(HERE / "transcript.json", "w") as f:
    json.dump(transcript, f, default=float, indent=2)
print(f"Wrote transcript.json with {len(iterations)} iterations.")

# ---------------------------------------------------------------------------
# Summary stats for analysis_summary.txt

summary = {
    "binary_results": binary_results,
    "adj_sig": adj_sig,
    "adj_signs": adj_signs,
    "TOP2": TOP2,
    "PROBE": PROBE,
    "top_float": top_float,
    "iter6_R2": float(m.rsquared),
    "iter7_R2": float(m7.rsquared),
    "iter25_R2": float(m25.rsquared),
    "iter25_R2_adj": float(m25.rsquared_adj),
    "iter25_n_sig": n_sig_total,
    "iter25_n_params": int(len(m25.params)),
    "iter17_p_race": p17,
    "iter18_p_ins": p18,
    "iter9_inter_p": float(m9.pvalues[f"{TOP2[0]}:{TOP2[1]}"]),
    "iter9_inter_beta": float(m9.params[f"{TOP2[0]}:{TOP2[1]}"]),
    "iter10_p_race_inter": p10,
    "iter11_p_ins_inter": p11,
    "iter12_age_inter_p": float(m12.pvalues[f"{PROBE}:feature_078"]),
    "iter13_f025_slope": float(m13.params["feature_025"]),
    "iter13_f025_p": float(m13.pvalues["feature_025"]),
    "iter14_low_hb_diff": float(low.mean() - high.mean()) if False else None,  # overwritten below
    "iter19_burden_slope": float(m19.params["_burden"]),
    "iter19_burden_p": float(m19.pvalues["_burden"]),
    "iter19_burden_R2": float(m19.rsquared),
    "iter20_top10": top10,
    "iter20_R2": float(m20.rsquared),
    "iter21_inter_p": float(m21.pvalues[f"{PROBE}:{top_float}"]),
    "iter21_inter_beta": float(m21.params[f"{PROBE}:{top_float}"]),
    "iter22_p": p22,
    "iter23_p": p23,
    "iter24_unadj_means": {str(k): float(v) for k, v in unadj_means.items()},
    "iter24_adj_b": float(adj_b),
    "iter24_adj_h": float(adj_h),
}
with open(HERE / "_summary_stats.json", "w") as f:
    json.dump(summary, f, default=float, indent=2)

print("Done.")
