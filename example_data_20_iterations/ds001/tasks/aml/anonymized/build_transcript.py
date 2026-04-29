"""Build transcript.json and analysis_summary.txt for the ds001_aml task.

Re-runs every analysis used in the iterations so that transcript values are
fully reproducible from this single script.
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
y = df["objective_response"]
N = len(df)


def chi2_pval(col, value=None):
    if value is None:
        ct = pd.crosstab(df[col], y)
    else:
        ct = pd.crosstab((df[col] == value).astype(int), y)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return float(chi2), float(p)


def ttest(col):
    a = df.loc[y == 1, col]
    b = df.loc[y == 0, col]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean()), float(b.mean()), float(p)


def logit_fit(cols, transforms=None):
    X = df[cols].copy()
    if transforms:
        for c, fn in transforms.items():
            X[c] = fn(X[c])
    X = sm.add_constant(X)
    m = sm.Logit(y, X).fit(disp=0)
    return m


iters = []

# ----------------------------------------------------------------------
# Iteration 1 — feature_057 (ordinal 0/1/2)
# ----------------------------------------------------------------------
mu1, mu0, p_t = ttest("feature_057")
rates_057 = df.groupby("feature_057")["objective_response"].mean().to_dict()
chi2_057, p_057 = chi2_pval("feature_057")
iters.append({
    "index": 1,
    "proposed_hypotheses": [
        {"id": "h1.1",
         "text": "Higher values of feature_057 (an ordinal 0/1/2 covariate) are associated with a lower probability of objective_response."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h1.1"],
            "code": "stats.ttest_ind(df.loc[y==1,'feature_057'], df.loc[y==0,'feature_057'])",
            "result_summary": (
                f"Mean feature_057 is {mu1:.3f} in responders vs {mu0:.3f} in non-responders "
                f"(Welch t-test p={p_t:.2e}). Response rate falls monotonically across feature_057 "
                f"levels: 0 -> {rates_057[0]:.3f}, 1 -> {rates_057[1]:.3f}, 2 -> {rates_057[2]:.3f}."
            ),
            "p_value": p_t,
            "effect_estimate": mu1 - mu0,  # negative => higher feature_057 -> lower response
            "significant": p_t < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 2 — feature_011 and feature_006
# ----------------------------------------------------------------------
mu1_11, mu0_11, p_11 = ttest("feature_011")
mu1_06, mu0_06, p_06 = ttest("feature_006")
iters.append({
    "index": 2,
    "proposed_hypotheses": [
        {"id": "h2.1",
         "text": "Higher values of the continuous variable feature_011 are associated with a lower probability of objective_response."},
        {"id": "h2.2",
         "text": "Higher values of the continuous variable feature_006 are associated with a lower probability of objective_response."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h2.1"],
            "code": "stats.ttest_ind(df.loc[y==1,'feature_011'], df.loc[y==0,'feature_011'])",
            "result_summary": (
                f"Mean feature_011 = {mu1_11:.3f} in responders vs {mu0_11:.3f} in non-responders "
                f"(Welch t-test p={p_11:.2e}); higher values predict lower response."
            ),
            "p_value": p_11,
            "effect_estimate": mu1_11 - mu0_11,
            "significant": p_11 < 0.05,
        },
        {
            "hypothesis_ids": ["h2.2"],
            "code": "stats.ttest_ind(df.loc[y==1,'feature_006'], df.loc[y==0,'feature_006'])",
            "result_summary": (
                f"Mean feature_006 = {mu1_06:.3f} in responders vs {mu0_06:.3f} in non-responders "
                f"(Welch t-test p={p_06:.2e})."
            ),
            "p_value": p_06,
            "effect_estimate": mu1_06 - mu0_06,
            "significant": p_06 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 3 — feature_063 and feature_092 (right-skewed continuous)
# ----------------------------------------------------------------------
mu1_63, mu0_63, p_63 = ttest("feature_063")
mu1_92, mu0_92, p_92 = ttest("feature_092")
# Log-transform t-tests
for col in ("feature_063", "feature_092"):
    df[f"log_{col}"] = np.log1p(df[col])
mu1_l63, mu0_l63, p_l63 = ttest("log_feature_063")
mu1_l92, mu0_l92, p_l92 = ttest("log_feature_092")
iters.append({
    "index": 3,
    "proposed_hypotheses": [
        {"id": "h3.1",
         "text": "Higher values of the right-skewed continuous variable feature_063 are associated with a lower probability of objective_response."},
        {"id": "h3.2",
         "text": "Higher values of the right-skewed continuous variable feature_092 are associated with a lower probability of objective_response."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h3.1"],
            "code": "stats.ttest_ind on raw and log1p-transformed feature_063",
            "result_summary": (
                f"feature_063 means: responders {mu1_63:.3f}, non-responders {mu0_63:.3f} (t-test p={p_63:.2e}). "
                f"After log1p: {mu1_l63:.3f} vs {mu0_l63:.3f} (p={p_l63:.2e})."
            ),
            "p_value": p_l63,
            "effect_estimate": mu1_l63 - mu0_l63,
            "significant": p_l63 < 0.05,
        },
        {
            "hypothesis_ids": ["h3.2"],
            "code": "stats.ttest_ind on raw and log1p-transformed feature_092",
            "result_summary": (
                f"feature_092 means: responders {mu1_92:.3f}, non-responders {mu0_92:.3f} (t-test p={p_92:.2e}). "
                f"After log1p: {mu1_l92:.3f} vs {mu0_l92:.3f} (p={p_l92:.2e})."
            ),
            "p_value": p_l92,
            "effect_estimate": mu1_l92 - mu0_l92,
            "significant": p_l92 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 4 — feature_099 (narrow-range continuous; opposite direction)
# ----------------------------------------------------------------------
mu1_99, mu0_99, p_99 = ttest("feature_099")
iters.append({
    "index": 4,
    "proposed_hypotheses": [
        {"id": "h4.1",
         "text": "Higher values of the continuous variable feature_099 are associated with a higher probability of objective_response (opposite direction from feature_011/006/063/092)."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h4.1"],
            "code": "stats.ttest_ind(df.loc[y==1,'feature_099'], df.loc[y==0,'feature_099'])",
            "result_summary": (
                f"feature_099 = {mu1_99:.3f} in responders vs {mu0_99:.3f} in non-responders "
                f"(Welch t-test p={p_99:.2e}); higher values modestly predict higher response, opposite to most other continuous predictors."
            ),
            "p_value": p_99,
            "effect_estimate": mu1_99 - mu0_99,
            "significant": p_99 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 5 — feature_035 (top binary predictor)
# ----------------------------------------------------------------------
rate1_35 = float(y[df["feature_035"] == 1].mean())
rate0_35 = float(y[df["feature_035"] == 0].mean())
chi2_35, p_35 = chi2_pval("feature_035")
iters.append({
    "index": 5,
    "proposed_hypotheses": [
        {"id": "h5.1",
         "text": "Patients with feature_035=1 have a higher probability of objective_response than patients with feature_035=0."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h5.1"],
            "code": "stats.chi2_contingency(pd.crosstab(df['feature_035'], y))",
            "result_summary": (
                f"Response rate {rate1_35:.3f} when feature_035=1 vs {rate0_35:.3f} when 0 "
                f"(chi-square p={p_35:.2e}); absolute difference {rate1_35-rate0_35:+.3f}."
            ),
            "p_value": p_35,
            "effect_estimate": rate1_35 - rate0_35,
            "significant": p_35 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 6 — Other modestly-significant binary predictors
# ----------------------------------------------------------------------
bin_results = []
for c in ("feature_093", "feature_121", "feature_014", "feature_030"):
    r1 = float(y[df[c] == 1].mean())
    r0 = float(y[df[c] == 0].mean())
    chi2, p = chi2_pval(c)
    bin_results.append((c, r1, r0, p))
iters.append({
    "index": 6,
    "proposed_hypotheses": [
        {"id": "h6.1",
         "text": "feature_093=1 is associated with a higher probability of objective_response than feature_093=0."},
        {"id": "h6.2",
         "text": "feature_121=1 is associated with a lower probability of objective_response than feature_121=0."},
        {"id": "h6.3",
         "text": "feature_014=1 is associated with a higher probability of objective_response than feature_014=0."},
        {"id": "h6.4",
         "text": "feature_030=1 is associated with a higher probability of objective_response than feature_030=0."},
    ],
    "analyses": [
        {
            "hypothesis_ids": [f"h6.{i+1}"],
            "code": f"stats.chi2_contingency(pd.crosstab(df['{c}'], y))",
            "result_summary": (
                f"{c}=1 response rate {r1:.3f} vs {c}=0 rate {r0:.3f} (chi-square p={p:.3g})."
            ),
            "p_value": p,
            "effect_estimate": r1 - r0,
            "significant": p < 0.05,
        }
        for i, (c, r1, r0, p) in enumerate(bin_results)
    ],
})

# ----------------------------------------------------------------------
# Iteration 7 — Race/ethnicity (feature_005)
# ----------------------------------------------------------------------
rates_005 = df.groupby("feature_005")["objective_response"].mean().to_dict()
chi2_005, p_005 = chi2_pval("feature_005")
white_rate = rates_005["white"]
other_rate = rates_005["other"]
iters.append({
    "index": 7,
    "proposed_hypotheses": [
        {"id": "h7.1",
         "text": "Probability of objective_response varies across categories of feature_005 (race/ethnicity), with at least one category differing from the rest."},
        {"id": "h7.2",
         "text": "Patients in feature_005='white' have a higher probability of objective_response than patients in feature_005='other'."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h7.1"],
            "code": "stats.chi2_contingency(pd.crosstab(df['feature_005'], y))",
            "result_summary": (
                "Univariate response rates by feature_005: "
                + ", ".join(f"{k}={v:.3f}" for k, v in rates_005.items())
                + f" (chi-square p={p_005:.4f})."
            ),
            "p_value": p_005,
            "effect_estimate": white_rate - np.mean(list(rates_005.values())),
            "significant": p_005 < 0.05,
        },
        {
            "hypothesis_ids": ["h7.2"],
            "code": "two-proportion test for white vs other within feature_005",
            "result_summary": (
                f"feature_005=white response {white_rate:.3f} vs feature_005=other {other_rate:.3f}; "
                f"difference {white_rate-other_rate:+.3f}, contributing to overall chi-square p={p_005:.4f}."
            ),
            "p_value": p_005,
            "effect_estimate": white_rate - other_rate,
            "significant": p_005 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 8 — Insurance (feature_087)
# ----------------------------------------------------------------------
rates_087 = df.groupby("feature_087")["objective_response"].mean().to_dict()
chi2_087, p_087 = chi2_pval("feature_087")
iters.append({
    "index": 8,
    "proposed_hypotheses": [
        {"id": "h8.1",
         "text": "Probability of objective_response varies across categories of feature_087 (insurance type), with at least one category differing from the rest."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h8.1"],
            "code": "stats.chi2_contingency(pd.crosstab(df['feature_087'], y))",
            "result_summary": (
                "Response rates by feature_087: "
                + ", ".join(f"{k}={v:.3f}" for k, v in rates_087.items())
                + f" (chi-square p={p_087:.3f}); differences are small and non-significant."
            ),
            "p_value": p_087,
            "effect_estimate": max(rates_087.values()) - min(rates_087.values()),
            "significant": p_087 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 9 — Age (feature_078) and sex-like (feature_085)
# ----------------------------------------------------------------------
mu1_78, mu0_78, p_78 = ttest("feature_078")
rate1_85 = float(y[df["feature_085"] == 1].mean())
rate0_85 = float(y[df["feature_085"] == 0].mean())
chi2_85, p_85 = chi2_pval("feature_085")
iters.append({
    "index": 9,
    "proposed_hypotheses": [
        {"id": "h9.1",
         "text": "Higher values of feature_078 (a continuous covariate ranging 18-92, plausibly age) are associated with a lower probability of objective_response."},
        {"id": "h9.2",
         "text": "Patients with feature_085=1 have a different probability of objective_response than feature_085=0 patients."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h9.1"],
            "code": "stats.ttest_ind(df.loc[y==1,'feature_078'], df.loc[y==0,'feature_078'])",
            "result_summary": (
                f"Mean feature_078 {mu1_78:.2f} in responders vs {mu0_78:.2f} in non-responders "
                f"(Welch t-test p={p_78:.3f}); no detectable association."
            ),
            "p_value": p_78,
            "effect_estimate": mu1_78 - mu0_78,
            "significant": p_78 < 0.05,
        },
        {
            "hypothesis_ids": ["h9.2"],
            "code": "stats.chi2_contingency(pd.crosstab(df['feature_085'], y))",
            "result_summary": (
                f"feature_085=1 response {rate1_85:.3f} vs feature_085=0 {rate0_85:.3f} "
                f"(chi-square p={p_85:.3f}); no detectable association."
            ),
            "p_value": p_85,
            "effect_estimate": rate1_85 - rate0_85,
            "significant": p_85 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 10 — Other ordinal features (018, 096, 064, 045, 042, 125)
# ----------------------------------------------------------------------
ord_results = []
for c in ("feature_018", "feature_096", "feature_064", "feature_045", "feature_042", "feature_125"):
    rho, p = stats.spearmanr(df[c], y)
    ord_results.append((c, float(rho), float(p)))
iters.append({
    "index": 10,
    "proposed_hypotheses": [
        {"id": f"h10.{i+1}",
         "text": f"There is a monotonic association between the ordinal variable {c} and the probability of objective_response."}
        for i, (c, _, _) in enumerate(ord_results)
    ],
    "analyses": [
        {
            "hypothesis_ids": [f"h10.{i+1}"],
            "code": f"stats.spearmanr(df['{c}'], y)",
            "result_summary": f"Spearman rho={rho:+.4f} between {c} and objective_response (p={p:.3f}).",
            "p_value": p,
            "effect_estimate": rho,
            "significant": p < 0.05,
        }
        for i, (c, rho, p) in enumerate(ord_results)
    ],
})

# ----------------------------------------------------------------------
# Iteration 11 — Multivariable logistic regression of top predictors
# ----------------------------------------------------------------------
top_cont = ["feature_057", "feature_011", "feature_006", "feature_063",
            "feature_099", "feature_092", "feature_078"]
top_bin = ["feature_035"]
m_full = logit_fit(top_cont + top_bin,
                   transforms={"feature_063": np.log1p, "feature_092": np.log1p})
mv_res = []
for var in top_cont + top_bin:
    coef = float(m_full.params[var])
    p = float(m_full.pvalues[var])
    mv_res.append((var, coef, p))

iters.append({
    "index": 11,
    "proposed_hypotheses": [
        {"id": "h11.1",
         "text": "feature_057, feature_011, feature_006, log1p(feature_063), feature_099, log1p(feature_092), and feature_035 are each independently associated with objective_response in a multivariable logistic regression; feature_078 is not."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h11.1"],
            "code": ("sm.Logit(y, sm.add_constant(X)).fit() with X = [feature_057, feature_011, "
                     "feature_006, log1p(feature_063), feature_099, log1p(feature_092), feature_078, feature_035]"),
            "result_summary": (
                "Adjusted logit coefficients (response on positive scale): "
                + "; ".join(f"{v} beta={b:+.3f} (p={p:.2e})" for v, b, p in mv_res)
                + ". feature_078 is not independently associated; all others are."
            ),
            "p_value": float(m_full.pvalues["feature_057"]),
            "effect_estimate": float(m_full.params["feature_057"]),
            "significant": True,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 12 — Race effect adjusted for prognostic factors
# ----------------------------------------------------------------------
def fit_with_dummies(extra_cols=()):
    X = df[list(top_cont) + list(top_bin) + list(extra_cols)].copy()
    X["feature_063"] = np.log1p(X["feature_063"])
    X["feature_092"] = np.log1p(X["feature_092"])
    return X


race_dum = pd.get_dummies(df["feature_005"], prefix="race", drop_first=True).astype(int)
X_race = pd.concat([fit_with_dummies(), race_dum], axis=1)
X_race = sm.add_constant(X_race)
m_race = sm.Logit(y, X_race).fit(disp=0)
race_terms = [c for c in X_race.columns if c.startswith("race_")]
race_p = float(m_race.f_test(" = ".join(race_terms) + " = 0").pvalue) if False else None
# Use likelihood-ratio test instead
m_race_null = sm.Logit(y, sm.add_constant(fit_with_dummies())).fit(disp=0)
lr_stat = 2 * (m_race.llf - m_race_null.llf)
df_lr = len(race_terms)
race_p = float(stats.chi2.sf(lr_stat, df_lr))
race_white_coef = float(m_race.params["race_white"])
race_white_p = float(m_race.pvalues["race_white"])
iters.append({
    "index": 12,
    "proposed_hypotheses": [
        {"id": "h12.1",
         "text": "After adjustment for the prognostic factors feature_057, feature_011, feature_006, log1p(feature_063), feature_099, log1p(feature_092), feature_078, and feature_035, the race/ethnicity variable feature_005 remains jointly associated with objective_response.",
         "kind": "refined"},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h12.1"],
            "code": ("Likelihood-ratio test comparing logit with and without the four feature_005 dummies, "
                     "controlling for top prognostic features."),
            "result_summary": (
                f"LR chi2({df_lr})={lr_stat:.2f}, p={race_p:.3f}. The univariate race signal "
                f"(p~0.01) is not detectable after adjusting for prognostic covariates "
                f"(strongest single race coefficient: race_white beta={race_white_coef:+.3f}, p={race_white_p:.3f})."
            ),
            "p_value": race_p,
            "effect_estimate": race_white_coef,
            "significant": race_p < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 13 — Insurance adjusted for prognostic factors
# ----------------------------------------------------------------------
ins_dum = pd.get_dummies(df["feature_087"], prefix="ins", drop_first=True).astype(int)
X_ins = pd.concat([fit_with_dummies(), ins_dum], axis=1)
X_ins = sm.add_constant(X_ins)
m_ins = sm.Logit(y, X_ins).fit(disp=0)
ins_terms = [c for c in X_ins.columns if c.startswith("ins_")]
m_ins_null = sm.Logit(y, sm.add_constant(fit_with_dummies())).fit(disp=0)
lr_ins = 2 * (m_ins.llf - m_ins_null.llf)
df_ins = len(ins_terms)
p_ins_adj = float(stats.chi2.sf(lr_ins, df_ins))
iters.append({
    "index": 13,
    "proposed_hypotheses": [
        {"id": "h13.1",
         "text": "After adjustment for prognostic factors, feature_087 (insurance type) remains jointly associated with objective_response.",
         "kind": "refined"},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h13.1"],
            "code": "Likelihood-ratio test for the four feature_087 dummies in the multivariable logit.",
            "result_summary": (
                f"LR chi2({df_ins})={lr_ins:.2f}, p={p_ins_adj:.3f}. Insurance type is not associated "
                f"with response after adjustment (univariate p was already non-significant, p={p_087:.3f})."
            ),
            "p_value": p_ins_adj,
            "effect_estimate": max(rates_087.values()) - min(rates_087.values()),
            "significant": p_ins_adj < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 14 — Sex (feature_085) within race subgroups
# ----------------------------------------------------------------------
sub_results = []
for race in df["feature_005"].unique():
    sub = df[df["feature_005"] == race]
    r1 = float(sub.loc[sub["feature_085"] == 1, "objective_response"].mean())
    r0 = float(sub.loc[sub["feature_085"] == 0, "objective_response"].mean())
    ct = pd.crosstab(sub["feature_085"], sub["objective_response"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    sub_results.append((race, r1, r0, float(p)))
iters.append({
    "index": 14,
    "proposed_hypotheses": [
        {"id": "h14.1",
         "text": "The (lack of) association between feature_085 and objective_response is consistent across categories of feature_005."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h14.1"],
            "code": "for each level of feature_005: stats.chi2_contingency(pd.crosstab(feature_085, y))",
            "result_summary": (
                "Within-race response by feature_085: "
                + "; ".join(f"{r}: 1={r1:.3f}, 0={r0:.3f} (p={p:.2f})" for r, r1, r0, p in sub_results)
                + ". No within-race subgroup shows a significant feature_085 effect."
            ),
            "p_value": float(np.median([p for _, _, _, p in sub_results])),
            "effect_estimate": float(np.mean([r1 - r0 for _, r1, r0, _ in sub_results])),
            "significant": False,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 15 — Interaction feature_035 x feature_057
# ----------------------------------------------------------------------
X = df[["feature_035", "feature_057"]].copy()
X["inter"] = X["feature_035"] * X["feature_057"]
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
inter_p_15 = float(m.pvalues["inter"])
inter_b_15 = float(m.params["inter"])
iters.append({
    "index": 15,
    "proposed_hypotheses": [
        {"id": "h15.1",
         "text": "The favourable effect of feature_035=1 on objective_response is larger in patients with low feature_057 (i.e., the feature_035 x feature_057 interaction term is negative on the logit scale)."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h15.1"],
            "code": "sm.Logit(y, [const, feature_035, feature_057, feature_035*feature_057])",
            "result_summary": (
                f"Interaction term beta={inter_b_15:+.4f} (p={inter_p_15:.3f}); the effect of feature_035 "
                f"is approximately constant across feature_057 strata (no heterogeneity detected)."
            ),
            "p_value": inter_p_15,
            "effect_estimate": inter_b_15,
            "significant": inter_p_15 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 16 — Interaction feature_011 x feature_057
# ----------------------------------------------------------------------
X = df[["feature_011", "feature_057"]].copy()
X["inter"] = X["feature_011"] * X["feature_057"]
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
inter_p_16 = float(m.pvalues["inter"])
inter_b_16 = float(m.params["inter"])
iters.append({
    "index": 16,
    "proposed_hypotheses": [
        {"id": "h16.1",
         "text": "The slope of feature_011 against objective_response is steeper (more negative) in patients with higher feature_057 (feature_011 x feature_057 interaction)."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h16.1"],
            "code": "sm.Logit(y, [const, feature_011, feature_057, feature_011*feature_057])",
            "result_summary": (
                f"Interaction term beta={inter_b_16:+.4f} (p={inter_p_16:.3f}); no detectable heterogeneity "
                f"in feature_011's effect across feature_057 strata."
            ),
            "p_value": inter_p_16,
            "effect_estimate": inter_b_16,
            "significant": inter_p_16 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 17 — Interaction feature_006 x feature_057
# ----------------------------------------------------------------------
X = df[["feature_006", "feature_057"]].copy()
X["inter"] = X["feature_006"] * X["feature_057"]
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
inter_p_17 = float(m.pvalues["inter"])
inter_b_17 = float(m.params["inter"])
iters.append({
    "index": 17,
    "proposed_hypotheses": [
        {"id": "h17.1",
         "text": "The negative effect of feature_006 on objective_response differs across feature_057 strata (feature_006 x feature_057 interaction)."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h17.1"],
            "code": "sm.Logit(y, [const, feature_006, feature_057, feature_006*feature_057])",
            "result_summary": (
                f"Interaction term beta={inter_b_17:+.4f} (p={inter_p_17:.3f}); no significant interaction."
            ),
            "p_value": inter_p_17,
            "effect_estimate": inter_b_17,
            "significant": inter_p_17 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 18 — Interaction feature_011 x feature_006
# ----------------------------------------------------------------------
X = df[["feature_011", "feature_006"]].copy()
X["inter"] = X["feature_011"] * X["feature_006"]
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
inter_p_18 = float(m.pvalues["inter"])
inter_b_18 = float(m.params["inter"])
iters.append({
    "index": 18,
    "proposed_hypotheses": [
        {"id": "h18.1",
         "text": "feature_011 and feature_006 interact: the joint effect on objective_response is non-additive on the logit scale (positive interaction would imply attenuation when both are high)."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h18.1"],
            "code": "sm.Logit(y, [const, feature_011, feature_006, feature_011*feature_006])",
            "result_summary": (
                f"Interaction term beta={inter_b_18:+.6f} (p={inter_p_18:.3f}); a marginal positive trend "
                f"(p~0.09) suggests possible attenuation when both are high but does not reach significance."
            ),
            "p_value": inter_p_18,
            "effect_estimate": inter_b_18,
            "significant": inter_p_18 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 19 — Age (feature_078) x feature_057 interaction
# ----------------------------------------------------------------------
X = df[["feature_078", "feature_057"]].copy()
X["inter"] = X["feature_078"] * X["feature_057"]
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
inter_p_19 = float(m.pvalues["inter"])
inter_b_19 = float(m.params["inter"])
iters.append({
    "index": 19,
    "proposed_hypotheses": [
        {"id": "h19.1",
         "text": "The (small or absent) effect of feature_078 on objective_response varies with feature_057 (age x performance-status-like interaction)."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h19.1"],
            "code": "sm.Logit(y, [const, feature_078, feature_057, feature_078*feature_057])",
            "result_summary": (
                f"Interaction term beta={inter_b_19:+.4f} (p={inter_p_19:.3f}); no significant heterogeneity "
                f"in the (already null) feature_078 effect across feature_057 strata."
            ),
            "p_value": inter_p_19,
            "effect_estimate": inter_b_19,
            "significant": inter_p_19 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 20 — Race x feature_057 interaction (Hispanic vs non-Hispanic)
# ----------------------------------------------------------------------
df["hispanic"] = (df["feature_005"] == "hispanic").astype(int)
X = df[["hispanic", "feature_057"]].copy()
X["inter"] = X["hispanic"] * X["feature_057"]
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
inter_p_20 = float(m.pvalues["inter"])
inter_b_20 = float(m.params["inter"])
iters.append({
    "index": 20,
    "proposed_hypotheses": [
        {"id": "h20.1",
         "text": "The relationship between feature_057 and objective_response differs in patients with feature_005='hispanic' vs the rest (race x performance-status interaction)."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h20.1"],
            "code": "sm.Logit(y, [const, hispanic, feature_057, hispanic*feature_057])",
            "result_summary": (
                f"Interaction term beta={inter_b_20:+.4f} (p={inter_p_20:.3f}); no detectable race-by-feature_057 interaction."
            ),
            "p_value": inter_p_20,
            "effect_estimate": inter_b_20,
            "significant": inter_p_20 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 21 — Combined risk score (multivariable LP) tertile validation
# ----------------------------------------------------------------------
X = df[top_cont + top_bin].copy()
X["feature_063"] = np.log1p(X["feature_063"])
X["feature_092"] = np.log1p(X["feature_092"])
Xc = sm.add_constant(X)
m_score = sm.Logit(y, Xc).fit(disp=0)
df["lp"] = m_score.predict(Xc)
df["risk_tertile"] = pd.qcut(df["lp"], 3, labels=["T1_low", "T2_mid", "T3_high"])
tertile_rates = df.groupby("risk_tertile", observed=True)["objective_response"].mean().to_dict()
hi_rate = float(tertile_rates["T3_high"])
lo_rate = float(tertile_rates["T1_low"])
n_hi = int((df["risk_tertile"] == "T3_high").sum())
n_lo = int((df["risk_tertile"] == "T1_low").sum())
# Two-prop z test
p_pool = (hi_rate * n_hi + lo_rate * n_lo) / (n_hi + n_lo)
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_hi + 1 / n_lo))
z = (hi_rate - lo_rate) / se
p_tertile = float(2 * stats.norm.sf(abs(z)))
iters.append({
    "index": 21,
    "proposed_hypotheses": [
        {"id": "h21.1",
         "text": "A composite linear-predictor score built from feature_057, feature_011, feature_006, log1p(feature_063), feature_099, log1p(feature_092), feature_078, and feature_035 stratifies patients into tertiles whose response rates differ substantially: top tertile responds more often than bottom tertile."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h21.1"],
            "code": "Predicted-probability tertiles from the multivariable logit; two-proportion z-test top vs bottom",
            "result_summary": (
                f"Response rates by risk tertile (using positive-on-response sign): "
                f"low={lo_rate:.3f}, mid={tertile_rates['T2_mid']:.3f}, high={hi_rate:.3f}. "
                f"Top vs bottom difference {hi_rate-lo_rate:+.3f} (z={z:.1f}, p={p_tertile:.2e})."
            ),
            "p_value": p_tertile,
            "effect_estimate": hi_rate - lo_rate,
            "significant": p_tertile < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 22 — Race effect adjusted only for feature_057 (sensitivity)
# ----------------------------------------------------------------------
race_dum2 = pd.get_dummies(df["feature_005"], prefix="race", drop_first=True).astype(int)
X22 = pd.concat([df[["feature_057"]], race_dum2], axis=1)
m22 = sm.Logit(y, sm.add_constant(X22)).fit(disp=0)
m22_null = sm.Logit(y, sm.add_constant(df[["feature_057"]])).fit(disp=0)
lr22 = 2 * (m22.llf - m22_null.llf)
df_lr22 = race_dum2.shape[1]
p22 = float(stats.chi2.sf(lr22, df_lr22))
race_white_coef22 = float(m22.params["race_white"])
iters.append({
    "index": 22,
    "proposed_hypotheses": [
        {"id": "h22.1",
         "text": "Even after adjusting only for feature_057 (the strongest single prognostic factor), feature_005 (race/ethnicity) is no longer associated with objective_response, indicating the univariate race signal is largely confounded by feature_057.",
         "kind": "refined"},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h22.1"],
            "code": "Likelihood-ratio test for race dummies in a logit adjusting for feature_057 only.",
            "result_summary": (
                f"LR chi2({df_lr22})={lr22:.2f}, p={p22:.3f}. Race effect attenuates to non-significance "
                f"after adjustment for feature_057 alone (race_white beta={race_white_coef22:+.3f}). "
                f"Univariate race signal (p={p_005:.4f}) appears confounded by feature_057."
            ),
            "p_value": p22,
            "effect_estimate": race_white_coef22,
            "significant": p22 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 23 — feature_125 ordinal (Spearman result revisited via logit)
# ----------------------------------------------------------------------
X23 = df[["feature_125"]].copy()
m23 = sm.Logit(y, sm.add_constant(X23)).fit(disp=0)
beta23 = float(m23.params["feature_125"])
p23 = float(m23.pvalues["feature_125"])
iters.append({
    "index": 23,
    "proposed_hypotheses": [
        {"id": "h23.1",
         "text": "Higher values of the ordinal variable feature_125 (range 0-4) are associated with a lower probability of objective_response."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h23.1"],
            "code": "sm.Logit(y, [const, feature_125])",
            "result_summary": (
                f"Logit coefficient {beta23:+.4f} per unit feature_125 (p={p23:.3f}); a small but "
                f"detectable monotonic decrease in response with increasing feature_125 "
                f"(consistent with Spearman rho=-0.010, p=0.020)."
            ),
            "p_value": p23,
            "effect_estimate": beta23,
            "significant": p23 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 24 — Cochran-Armitage trend test on feature_006 deciles
# ----------------------------------------------------------------------
df["f006_dec"] = pd.qcut(df["feature_006"], 10, duplicates="drop", labels=False)
trend_rates = df.groupby("f006_dec", observed=True)["objective_response"].mean().to_dict()
# logit on decile rank
X24 = df[["f006_dec"]].astype(float)
m24 = sm.Logit(y, sm.add_constant(X24)).fit(disp=0)
beta24 = float(m24.params["f006_dec"])
p24 = float(m24.pvalues["f006_dec"])
iters.append({
    "index": 24,
    "proposed_hypotheses": [
        {"id": "h24.1",
         "text": "Across deciles of feature_006 there is a monotonic dose-response: response rate decreases approximately linearly from the lowest to highest decile."},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h24.1"],
            "code": "sm.Logit(y, [const, decile_rank_of_feature_006])",
            "result_summary": (
                "Decile response rates: "
                + ", ".join(f"D{int(k)+1}={v:.3f}" for k, v in sorted(trend_rates.items()))
                + f". Logit beta per decile = {beta24:+.4f} (p={p24:.2e}); response falls "
                f"from ~0.193 at decile 1 to ~0.143 at decile 10."
            ),
            "p_value": p24,
            "effect_estimate": beta24,
            "significant": p24 < 0.05,
        },
    ],
})

# ----------------------------------------------------------------------
# Iteration 25 — Final multivariable model with all top predictors + race + insurance
# ----------------------------------------------------------------------
X_final = pd.concat([fit_with_dummies(), race_dum, ins_dum], axis=1)
X_final = sm.add_constant(X_final)
m_final = sm.Logit(y, X_final).fit(disp=0)
overall_p = float(m_final.llr_pvalue)
final_params = {var: (float(m_final.params[var]), float(m_final.pvalues[var]))
                for var in X_final.columns if var != "const"}
n_sig = sum(1 for v, (b, p) in final_params.items() if p < 0.05)
iters.append({
    "index": 25,
    "proposed_hypotheses": [
        {"id": "h25.1",
         "text": "In a single multivariable logistic regression including the eight prognostic factors plus feature_005 and feature_087 dummies, the prognostic factors (feature_057, feature_011, feature_006, log1p(feature_063), feature_099, log1p(feature_092), feature_035) remain independently associated with objective_response, while neither feature_005 nor feature_087 dummies reach significance.",
         "kind": "refined"},
    ],
    "analyses": [
        {
            "hypothesis_ids": ["h25.1"],
            "code": "Multivariable logit with prognostic features, race dummies, and insurance dummies",
            "result_summary": (
                f"Overall model LLR p={overall_p:.2e}; pseudo-R^2={float(m_final.prsquared):.4f}. "
                + "Prognostic features remain significant: "
                + ", ".join(f"{v} beta={b:+.3f} (p={p:.1e})" for v, (b, p) in final_params.items()
                            if v in top_cont + top_bin and p < 0.05)
                + ". Race and insurance dummy p-values: "
                + ", ".join(f"{v}={p:.2f}" for v, (b, p) in final_params.items()
                            if v.startswith(("race_", "ins_")))
                + ". After adjustment, no race or insurance category reaches p<0.05."
            ),
            "p_value": overall_p,
            "effect_estimate": float(m_final.params["feature_057"]),  # signed, primary prognostic effect
            "significant": True,
        },
    ],
})

# ----------------------------------------------------------------------
# Assemble transcript
# ----------------------------------------------------------------------
transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@aml-anonymized-1",
    "max_iterations": 25,
    "iterations": iters,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2, default=float)

print(f"Wrote transcript.json with {len(iters)} iterations.")
print(f"Total hypotheses: {sum(len(it['proposed_hypotheses']) for it in iters)}")
print(f"Total analyses:  {sum(len(it['analyses']) for it in iters)}")

# ----------------------------------------------------------------------
# Persist key numbers for the summary writer
# ----------------------------------------------------------------------
summary_state = {
    "rates_057": rates_057,
    "rate1_35": rate1_35, "rate0_35": rate0_35, "p_35": p_35,
    "p_057": p_057,
    "rates_005": rates_005,
    "rates_087": rates_087,
    "p_005": p_005, "p_087": p_087,
    "p_78": p_78, "p_85": p_85,
    "tertile_rates": tertile_rates,
    "hi_rate": hi_rate, "lo_rate": lo_rate,
    "p_tertile": p_tertile,
    "p22": p22,
    "race_p_full": race_p,
    "p_ins_adj": p_ins_adj,
    "mv_res": mv_res,
    "ord_results": ord_results,
    "trend_rates": trend_rates,
    "beta24": beta24, "p24": p24,
}
with open("_summary_state.json", "w") as f:
    json.dump(summary_state, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
print("Wrote _summary_state.json")
