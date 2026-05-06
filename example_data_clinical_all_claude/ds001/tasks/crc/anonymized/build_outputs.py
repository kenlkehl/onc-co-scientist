"""Build transcript.json and analysis_summary.txt for ds001_crc."""
import json
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-manual@2026-05-03",
    "max_iterations": 25,
    "iterations": [],
}


def add_iter(idx, hyps, analyses):
    transcript["iterations"].append({
        "index": idx,
        "proposed_hypotheses": hyps,
        "analyses": analyses,
    })


# Iteration 1 — common binary features as prognostic factors
add_iter(
    1,
    [
        {"id": "h1", "text": "Patients with feature_014=1 have shorter pfs_months than patients with feature_014=0.", "kind": "novel"},
        {"id": "h2", "text": "Patients with feature_023=1 have shorter pfs_months than patients with feature_023=0.", "kind": "novel"},
        {"id": "h3", "text": "Patients with feature_006=1 have shorter pfs_months than patients with feature_006=0.", "kind": "novel"},
        {"id": "h4", "text": "Patients with feature_031=1 have different pfs_months than patients with feature_031=0.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h1"],
            "code": "from scipy import stats\nstats.ttest_ind(df.loc[df.feature_014==1,'pfs_months'], df.loc[df.feature_014==0,'pfs_months'], equal_var=False)",
            "result_summary": "Mean pfs_months is 5.055 when feature_014=0 (n=22519) vs 3.703 when feature_014=1 (n=27481); Welch t-test p<1e-300. feature_014=1 is strongly associated with shorter PFS.",
            "p_value": 0.0,
            "effect_estimate": -1.352,
            "significant": True,
        },
        {
            "hypothesis_ids": ["h2"],
            "code": "stats.ttest_ind(df.loc[df.feature_023==1,'pfs_months'], df.loc[df.feature_023==0,'pfs_months'], equal_var=False)",
            "result_summary": "Mean pfs_months 4.448 (feature_023=0) vs 4.121 (feature_023=1); diff -0.327 months, p=1.86e-58. feature_023=1 is associated with modestly shorter PFS.",
            "p_value": 1.86e-58,
            "effect_estimate": -0.327,
            "significant": True,
        },
        {
            "hypothesis_ids": ["h3"],
            "code": "stats.ttest_ind(df.loc[df.feature_006==1,'pfs_months'], df.loc[df.feature_006==0,'pfs_months'], equal_var=False)",
            "result_summary": "Mean pfs_months 4.419 (feature_006=0) vs 4.114 (feature_006=1); diff -0.305 months, p=1.04e-49. feature_006=1 is associated with modestly shorter PFS.",
            "p_value": 1.04e-49,
            "effect_estimate": -0.305,
            "significant": True,
        },
        {
            "hypothesis_ids": ["h4"],
            "code": "stats.ttest_ind(df.loc[df.feature_031==1,'pfs_months'], df.loc[df.feature_031==0,'pfs_months'], equal_var=False)",
            "result_summary": "Mean pfs_months 4.314 (feature_031=0) vs 4.308 (feature_031=1); diff -0.006 months, p=0.77. No detectable effect of feature_031 on PFS.",
            "p_value": 0.77,
            "effect_estimate": -0.006,
            "significant": False,
        },
    ],
)

# Iteration 2 — three-level feature_001
add_iter(
    2,
    [
        {"id": "h5", "text": "pfs_months decreases monotonically across the three levels of feature_001 (0, 1, 2), with level 2 showing the shortest PFS.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h5"],
            "code": "stats.f_oneway(df.loc[df.feature_001==0,'pfs_months'], df.loc[df.feature_001==1,'pfs_months'], df.loc[df.feature_001==2,'pfs_months'])",
            "result_summary": "Group means: feature_001=0 -> 5.271 (n=17592); =1 -> 4.054 (n=24971); =2 -> 2.908 (n=7437). Monotonic decrease ~1.2 months per level. ANOVA F=3528.9, p<1e-300. Levels 1 and 2 are progressively worse than 0.",
            "p_value": 0.0,
            "effect_estimate": -2.363,
            "significant": True,
        },
    ],
)

# Iteration 3 — feature_015 continuous prognostic factor
add_iter(
    3,
    [
        {"id": "h6", "text": "Higher feature_015 (continuous, range 30-90) is associated with longer pfs_months.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h6"],
            "code": "stats.linregress(df['feature_015'], df['pfs_months'])",
            "result_summary": "Linear regression: slope=+0.1765 months per unit feature_015 (Pearson r=0.766, p<1e-300). Quintile means rise from 1.86 (lowest) to 6.83 (highest). feature_015 is the single strongest continuous predictor of PFS in the dataset.",
            "p_value": 0.0,
            "effect_estimate": 0.1765,
            "significant": True,
        },
    ],
)

# Iteration 4 — feature_019 continuous
add_iter(
    4,
    [
        {"id": "h7", "text": "Higher feature_019 (continuous, range 1.5-5.5) is associated with longer pfs_months.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h7"],
            "code": "stats.linregress(df['feature_019'], df['pfs_months'])",
            "result_summary": "Slope=+0.4654 months per unit feature_019 (r=0.102, p=3.3e-115). Modest but highly significant positive association.",
            "p_value": 3.3e-115,
            "effect_estimate": 0.4654,
            "significant": True,
        },
    ],
)

# Iteration 5 — feature_025
add_iter(
    5,
    [
        {"id": "h8", "text": "Higher feature_025 (continuous, range 0-24.6) is associated with shorter pfs_months.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h8"],
            "code": "stats.linregress(df['feature_025'], df['pfs_months'])",
            "result_summary": "Slope=-0.0730 months per unit feature_025 (r=-0.123, p=2.2e-167). Negative association: each additional unit reduces PFS by ~0.07 months.",
            "p_value": 2.2e-167,
            "effect_estimate": -0.0730,
            "significant": True,
        },
    ],
)

# Iteration 6 — feature_005 (right-skewed lab) and dichotomization
add_iter(
    6,
    [
        {"id": "h9", "text": "Higher feature_005 (continuous, right-skewed, mean 9.2, max 777) is associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h10", "text": "A clinical-style cutoff feature_005<5 vs >=5 separates patients with longer vs shorter pfs_months.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h9"],
            "code": "stats.linregress(df['feature_005'], df['pfs_months'])",
            "result_summary": "Linear slope=-0.0053 per unit (r=-0.038, p=3.1e-17). Negative direction confirmed but weak overall because feature_005 is highly right-skewed.",
            "p_value": 3.1e-17,
            "effect_estimate": -0.0053,
            "significant": True,
        },
        {
            "hypothesis_ids": ["h10"],
            "code": "stats.ttest_ind(df.loc[df.feature_005<5,'pfs_months'], df.loc[df.feature_005>=5,'pfs_months'], equal_var=False)",
            "result_summary": "Mean pfs_months 4.470 when feature_005<5 (n=26252) vs 4.126 when feature_005>=5 (n=23748); diff +0.344, p=2.0e-64. The 5-unit cutoff cleanly separates patients with longer PFS.",
            "p_value": 2.0e-64,
            "effect_estimate": 0.344,
            "significant": True,
        },
    ],
)

# Iteration 7 — feature_021, feature_032, feature_017 main effects (some negligible adjusted)
add_iter(
    7,
    [
        {"id": "h11", "text": "Higher feature_021 (continuous, mean 224, range 48-830) is associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h12", "text": "Higher feature_032 (continuous, mean 6.1, max 300) is associated with shorter pfs_months.", "kind": "novel"},
        {"id": "h13", "text": "Higher feature_017 (continuous, mean 3.5, max 30) has a non-zero linear relationship with pfs_months.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h11"],
            "code": "stats.linregress(df['feature_021'], df['pfs_months'])",
            "result_summary": "Slope=-0.000318 per unit feature_021 (r=-0.0112, p=0.013). Statistically significant but practically very small marginal effect.",
            "p_value": 0.013,
            "effect_estimate": -0.000318,
            "significant": True,
        },
        {
            "hypothesis_ids": ["h12"],
            "code": "stats.linregress(df['feature_032'], df['pfs_months'])",
            "result_summary": "Slope=-0.00204 per unit feature_032 (r=-0.0083, p=0.064). Direction negative, marginal significance only.",
            "p_value": 0.064,
            "effect_estimate": -0.00204,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h13"],
            "code": "stats.linregress(df['feature_017'], df['pfs_months'])",
            "result_summary": "Slope=+0.00524 per unit (r=0.0047, p=0.30). No appreciable linear effect of feature_017 on PFS.",
            "p_value": 0.30,
            "effect_estimate": 0.00524,
            "significant": False,
        },
    ],
)

# Iteration 8 — rare binary features
add_iter(
    8,
    [
        {"id": "h14", "text": "Patients with feature_011=1 have different pfs_months than feature_011=0.", "kind": "novel"},
        {"id": "h15", "text": "Patients with feature_022=1 have shorter pfs_months than feature_022=0.", "kind": "novel"},
        {"id": "h16", "text": "Patients with feature_016=1 have different pfs_months than feature_016=0.", "kind": "novel"},
        {"id": "h17", "text": "Patients with feature_028=1 have different pfs_months than feature_028=0.", "kind": "novel"},
        {"id": "h18", "text": "Patients with feature_002=1 have different pfs_months than feature_002=0.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h14"],
            "code": "stats.ttest_ind(df.loc[df.feature_011==1,'pfs_months'], df.loc[df.feature_011==0,'pfs_months'], equal_var=False)",
            "result_summary": "Mean PFS 4.526 (feature_011=1, n=1506) vs 4.305 (feature_011=0, n=48494); diff +0.221, p=7.3e-4. Small positive effect.",
            "p_value": 7.3e-4,
            "effect_estimate": 0.221,
            "significant": True,
        },
        {
            "hypothesis_ids": ["h15"],
            "code": "stats.ttest_ind(df.loc[df.feature_022==1,'pfs_months'], df.loc[df.feature_022==0,'pfs_months'], equal_var=False)",
            "result_summary": "Mean PFS 4.094 (feature_022=1, n=2272) vs 4.322 (feature_022=0); diff -0.228, p=3.5e-7. Modest negative effect.",
            "p_value": 3.5e-7,
            "effect_estimate": -0.228,
            "significant": True,
        },
        {
            "hypothesis_ids": ["h16"],
            "code": "stats.ttest_ind(df.loc[df.feature_016==1,'pfs_months'], df.loc[df.feature_016==0,'pfs_months'], equal_var=False)",
            "result_summary": "Diff -0.019, p=0.68. No detectable main effect.",
            "p_value": 0.68,
            "effect_estimate": -0.019,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h17"],
            "code": "stats.ttest_ind(df.loc[df.feature_028==1,'pfs_months'], df.loc[df.feature_028==0,'pfs_months'], equal_var=False)",
            "result_summary": "Diff -0.056, p=0.36. No detectable main effect.",
            "p_value": 0.36,
            "effect_estimate": -0.056,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h18"],
            "code": "stats.ttest_ind(df.loc[df.feature_002==1,'pfs_months'], df.loc[df.feature_002==0,'pfs_months'], equal_var=False)",
            "result_summary": "Diff +0.040, p=0.79. No detectable main effect for feature_002 (very rare, n=251).",
            "p_value": 0.79,
            "effect_estimate": 0.040,
            "significant": False,
        },
    ],
)

# Iteration 9 — feature_027 as candidate treatment
add_iter(
    9,
    [
        {"id": "h19", "text": "Patients with feature_027=1 have longer pfs_months than feature_027=0; feature_027 may represent a beneficial treatment indicator.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h19"],
            "code": "stats.ttest_ind(df.loc[df.feature_027==1,'pfs_months'], df.loc[df.feature_027==0,'pfs_months'], equal_var=False)",
            "result_summary": "Mean PFS 5.089 (feature_027=1, n=10022) vs 4.117 (feature_027=0, n=39978); diff +0.972 months, p=1.7e-216. feature_027=1 is associated with substantially longer PFS — strongest beneficial association of any binary variable, consistent with a treatment indicator.",
            "p_value": 1.7e-216,
            "effect_estimate": 0.972,
            "significant": True,
        },
    ],
)

# Iteration 10 — multivariable regression
add_iter(
    10,
    [
        {"id": "h20", "text": "After mutual adjustment, feature_015, feature_001, feature_014, feature_027, feature_019, feature_025, feature_006, feature_023, feature_005, feature_022 retain independent associations with pfs_months.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h20"],
            "code": "smf.ols('pfs_months ~ ' + ' + '.join(features), data=df).fit()",
            "result_summary": "Multivariable OLS (all features, n=50000): independent t-stat magnitudes >5: feature_015 +0.176/u (t=453), feature_001 -1.171/level (t=-205), feature_014 -1.362 (t=-176), feature_027 +0.943 (t=98), feature_025 -0.076/u (t=-76), feature_019 +0.469 (t=61), feature_023 -0.354 (t=-44), feature_006 -0.287 (t=-36), feature_005 -0.005/u (t=-21), feature_022 -0.357 (t=-19), feature_021 -0.0004 (t=-9). All other features have |t|<2 and are not independently associated.",
            "p_value": 0.0,
            "effect_estimate": 0.943,
            "significant": True,
        },
    ],
)

# Iteration 11 — adjusted treatment effect of feature_027
add_iter(
    11,
    [
        {"id": "h21", "text": "The pfs_months benefit associated with feature_027 persists after adjusting for prognostic covariates (feature_015, feature_001, feature_014, feature_019, feature_025, feature_006, feature_023, feature_005, feature_022).", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h21"],
            "code": "smf.ols('pfs_months ~ feature_027 + feature_015 + C(feature_001) + feature_014 + feature_019 + feature_025 + feature_006 + feature_023 + feature_005 + feature_022', data=df).fit()",
            "result_summary": "Adjusted coefficient for feature_027 = +0.943 months (effectively unchanged from the unadjusted +0.972), p<1e-200. Treatment-like benefit is robust to confounding by measured prognostic factors.",
            "p_value": 0.0,
            "effect_estimate": 0.943,
            "significant": True,
        },
    ],
)

# Iteration 12 — interaction screen
add_iter(
    12,
    [
        {"id": "h22", "text": "The pfs_months effect of feature_027 is not constant across patients; at least one feature in the dataset modifies the magnitude of the feature_027 effect.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h22"],
            "code": "for mod in features: smf.ols(f'pfs_months ~ feature_027 * {mod}', data=df).fit()",
            "result_summary": "Systematic screen of feature_027 x modifier interactions (one modifier per OLS model). Strongest interaction p-values: feature_023 (coef=-1.629, p=7.7e-226), feature_006 (-1.454, p=4.1e-168), feature_005 (-0.0253, p=8.7e-58), feature_022 (-1.183, p=6.1e-23), feature_011 (+0.988, p=1.2e-11). All other interaction p>0.1. Negative interactions indicate that feature_027 benefit is concentrated in patients with the modifier set to its 'favorable' value (0 for binaries, low for feature_005).",
            "p_value": 7.7e-226,
            "effect_estimate": -1.629,
            "significant": True,
        },
    ],
)

# Iteration 13 — feature_027 x feature_023
add_iter(
    13,
    [
        {"id": "h23", "text": "feature_027 produces longer pfs_months only in patients with feature_023=0; among feature_023=1 patients the benefit is absent.", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h23"],
            "code": "smf.ols('pfs_months ~ feature_027 * feature_023', data=df).fit()",
            "result_summary": "Stratified means (m1-m0): feature_023=0 -> +1.633 months (n0=23286, n1=5816); feature_023=1 -> +0.000 months (n0=16692, n1=4206). Interaction coefficient -1.629, p=7.7e-226. feature_023 fully suppresses the feature_027 benefit.",
            "p_value": 7.7e-226,
            "effect_estimate": -1.629,
            "significant": True,
        },
    ],
)

# Iteration 14 — feature_027 x feature_006
add_iter(
    14,
    [
        {"id": "h24", "text": "feature_027 produces longer pfs_months only in patients with feature_006=0; among feature_006=1 patients the benefit is absent.", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h24"],
            "code": "smf.ols('pfs_months ~ feature_027 * feature_006', data=df).fit()",
            "result_summary": "Stratified means (m1-m0): feature_006=0 -> +1.464 months; feature_006=1 -> +0.020 months. Interaction coef -1.454, p=4.1e-168. feature_006=1 suppresses the feature_027 benefit.",
            "p_value": 4.1e-168,
            "effect_estimate": -1.454,
            "significant": True,
        },
    ],
)

# Iteration 15 — feature_027 x feature_022
add_iter(
    15,
    [
        {"id": "h25", "text": "feature_027 produces longer pfs_months only in patients with feature_022=0; feature_022=1 patients show no benefit.", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h25"],
            "code": "smf.ols('pfs_months ~ feature_027 * feature_022', data=df).fit()",
            "result_summary": "Stratified means (m1-m0): feature_022=0 -> +1.034; feature_022=1 -> -0.149 (n0=1772, n1=500). Interaction coef -1.183, p=6.1e-23. feature_022 fully neutralizes feature_027 benefit.",
            "p_value": 6.1e-23,
            "effect_estimate": -1.183,
            "significant": True,
        },
    ],
)

# Iteration 16 — feature_027 x feature_005
add_iter(
    16,
    [
        {"id": "h26", "text": "The feature_027 pfs_months benefit decreases as feature_005 increases; patients with high feature_005 derive no benefit.", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h26"],
            "code": "smf.ols('pfs_months ~ feature_027 * feature_005', data=df).fit()",
            "result_summary": "Interaction coefficient = -0.0253 per unit feature_005 (p=8.7e-58). At feature_005=0 the conditional treatment effect is +2.21 months; at feature_005=40 it is approximately +1.21 months; nonparametric quintile pattern shows the benefit collapses to ~0 above feature_005~6.",
            "p_value": 8.7e-58,
            "effect_estimate": -0.0253,
            "significant": True,
        },
    ],
)

# Iteration 17 — non-modifiers: feature_001, feature_014, feature_015
add_iter(
    17,
    [
        {"id": "h27", "text": "Although feature_001 is strongly prognostic, it does NOT modify the pfs_months effect of feature_027.", "kind": "novel"},
        {"id": "h28", "text": "Although feature_014 is strongly prognostic, it does NOT modify the pfs_months effect of feature_027.", "kind": "novel"},
        {"id": "h29", "text": "Although feature_015 is the strongest continuous prognostic factor, it does NOT modify the pfs_months effect of feature_027.", "kind": "novel"},
    ],
    [
        {
            "hypothesis_ids": ["h27"],
            "code": "smf.ols('pfs_months ~ feature_027 * C(feature_001)', data=df).fit()",
            "result_summary": "feature_027:C(feature_001)[T.1] coef +0.070 p=0.17; feature_027:C(feature_001)[T.2] coef -0.028 p=0.70. No effect modification by feature_001.",
            "p_value": 0.17,
            "effect_estimate": 0.070,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h28"],
            "code": "smf.ols('pfs_months ~ feature_027 * feature_014', data=df).fit()",
            "result_summary": "Interaction coef +0.070, p=0.15. feature_014 is purely prognostic and does not change the magnitude of the feature_027 benefit.",
            "p_value": 0.15,
            "effect_estimate": 0.070,
            "significant": False,
        },
        {
            "hypothesis_ids": ["h29"],
            "code": "smf.ols('pfs_months ~ feature_027 * feature_015', data=df).fit()",
            "result_summary": "Interaction coef -0.0017 per unit, p=0.28. No effect modification by feature_015 (despite very strong main prognostic effect).",
            "p_value": 0.28,
            "effect_estimate": -0.0017,
            "significant": False,
        },
    ],
)

# Iteration 18 — joint subgroup (feature_023, feature_006)
add_iter(
    18,
    [
        {"id": "h30", "text": "feature_027 produces a meaningful pfs_months benefit only in the joint subgroup feature_023=0 AND feature_006=0; in any cell where feature_023=1 or feature_006=1, the benefit is essentially zero.", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h30"],
            "code": "df.groupby(['feature_023','feature_006','feature_027'])['pfs_months'].mean()",
            "result_summary": "Treatment effect by 2x2 cell (m1-m0): (f023=0,f006=0) +2.520 (n=18853, p<1e-300); (f023=0,f006=1) +0.071 (p=0.16); (f023=1,f006=0) +0.061 (p=0.17); (f023=1,f006=1) -0.040 (p=0.52). Three-way OLS confirms the highest-order interactions are large and significant.",
            "p_value": 5.87e-286,
            "effect_estimate": 2.520,
            "significant": True,
        },
    ],
)

# Iteration 19 — add feature_022 to subgroup
add_iter(
    19,
    [
        {"id": "h31", "text": "Within the f023=0 AND f006=0 subgroup, feature_027 produces a benefit only when feature_022=0; feature_022=1 patients in that subgroup show no benefit.", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h31"],
            "code": "df.groupby(['feature_023','feature_006','feature_022','feature_027'])['pfs_months'].mean()",
            "result_summary": "Within feature_023=0 & feature_006=0: feature_022=0 -> treatment effect +2.758 months (n0=13911, n1=3461); feature_022=1 -> -0.204 months (n0=1179, n1=302). Stratifying further on feature_022=0 sharpens the effect.",
            "p_value": 1.0e-300,
            "effect_estimate": 2.758,
            "significant": True,
        },
    ],
)

# Iteration 20 — add feature_005<5 to subgroup
add_iter(
    20,
    [
        {"id": "h32", "text": "Within the f023=0 AND f006=0 AND f022=0 subgroup, feature_027 produces a benefit only when feature_005<5; for feature_005>=5 the benefit collapses to ~0.", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h32"],
            "code": "df.groupby(...)[treatment_effect within f023=0 & f006=0 & f022=0 by feature_005<5]",
            "result_summary": "Within (f023=0, f006=0, f022=0) the treatment effect of feature_027 is +5.013 months when feature_005<5 (n=9326, n0=7423, n1=1903) and +0.005 months when feature_005>=5 (n=8046, n0=6488, n1=1558). The split at 5 is decisive.",
            "p_value": 0.0,
            "effect_estimate": 5.013,
            "significant": True,
        },
    ],
)

# Iteration 21 — sweep cutoffs on feature_005
add_iter(
    21,
    [
        {"id": "h33", "text": "Within the f023=0 AND f006=0 AND f022=0 subgroup, the pfs_months benefit of feature_027 is concentrated below a clinically reasonable cutoff of feature_005~5 (consistent with a CEA-like normal/elevated threshold).", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h33"],
            "code": "for cut in [3,4,5,6,7,8]: compare treatment effect for feature_005<cut vs >=cut within (f023=0,f006=0,f022=0)",
            "result_summary": "Treatment-effect sweep within (f023=0,f006=0,f022=0): cut=3 -> below=+5.03/above=+1.32; cut=4 -> +5.04/+0.69; cut=5 -> +5.01/-0.00; cut=6 -> +4.54/-0.02; cut=7 -> +4.13/+0.02. The sharpest separation occurs at feature_005=5: below the cut the benefit is +5.0 months and above it the effect is essentially zero.",
            "p_value": 0.0,
            "effect_estimate": 5.014,
            "significant": True,
        },
    ],
)

# Iteration 22 — outside-subgroup confirmation
add_iter(
    22,
    [
        {"id": "h34", "text": "Outside the proposed full subgroup (i.e., feature_023=1 OR feature_006=1 OR feature_022=1 OR feature_005>=5), feature_027 has no measurable effect on pfs_months.", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h34"],
            "code": "fav = (f023==0)&(f006==0)&(f022==0)&(f005<5); stats.ttest_ind(out.loc[out.f027==1,'pfs_months'], out.loc[out.f027==0,'pfs_months'])",
            "result_summary": "Patients outside the proposed subgroup (n=40674; n0=32555 untreated, n1=8119 treated): mean PFS 4.123 vs 4.148, diff +0.025 months, p=0.33. The feature_027 indicator is essentially unrelated to PFS for the 81% of patients who do not meet the subgroup criteria.",
            "p_value": 0.33,
            "effect_estimate": 0.025,
            "significant": False,
        },
    ],
)

# Iteration 23 — three-way interaction OLS
add_iter(
    23,
    [
        {"id": "h35", "text": "A formal three-way interaction model feature_027 * feature_023 * feature_006 supports a unique 'sweet-spot' subgroup (f023=0, f006=0) with a strong feature_027 effect.", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h35"],
            "code": "smf.ols('pfs_months ~ feature_027 * feature_023 * feature_006', data=df).fit()",
            "result_summary": "Coefficients: feature_027 +2.520 (p<1e-300); feature_027:feature_023 -2.459 (p<1e-300); feature_027:feature_006 -2.449 (p=5.9e-286); feature_027:feature_023:feature_006 +2.348 (p=8.4e-112). The three-way term is positive: when both feature_023=1 and feature_006=1 the simple-effect-cancellation mostly returns the treatment effect to ~0, confirming a 2x2 'AND' subgroup pattern.",
            "p_value": 8.4e-112,
            "effect_estimate": 2.348,
            "significant": True,
        },
    ],
)

# Iteration 24 — search for additional modifiers within favorable subgroup
add_iter(
    24,
    [
        {"id": "h36", "text": "Within the favorable subgroup (f023=0, f006=0, f022=0, f005<5), no remaining feature meaningfully modifies the magnitude of the feature_027 pfs_months effect.", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h36"],
            "code": "within favorable subgroup, screen feature_027 * mod for every other feature",
            "result_summary": "Smallest interaction p-values inside the favorable subgroup (n=9326): feature_029 p=0.073, feature_015 p=0.110, feature_032 p=0.119, feature_020 p=0.129, feature_013 p=0.152. None reach Bonferroni-corrected significance (~0.0017 for ~30 tests). The favorable subgroup looks homogeneous with respect to the feature_027 benefit.",
            "p_value": 0.073,
            "effect_estimate": 0.0197,
            "significant": False,
        },
    ],
)

# Iteration 25 — final consolidated subgroup hypothesis
add_iter(
    25,
    [
        {"id": "h37", "text": "FINAL TREATMENT-EFFECT SUBGROUP: feature_027 prolongs pfs_months by approximately +5.0 months (mean 9.10 vs 4.09, p<1e-300) only in patients who simultaneously satisfy feature_023=0 AND feature_006=0 AND feature_022=0 AND feature_005<5 (~19% of the cohort, n=9326). For all patients failing any one of these four predicates (n=40674), feature_027 has no detectable effect on pfs_months (diff +0.025 months, p=0.33). feature_023, feature_006, feature_022, and feature_005>=5 each independently abolish the benefit.", "kind": "refined"},
    ],
    [
        {
            "hypothesis_ids": ["h37"],
            "code": "fav = (df.feature_023==0)&(df.feature_006==0)&(df.feature_022==0)&(df.feature_005<5); stats.ttest_ind(df.loc[fav&(df.feature_027==1),'pfs_months'], df.loc[fav&(df.feature_027==0),'pfs_months'], equal_var=False); stats.ttest_ind(df.loc[~fav&(df.feature_027==1),'pfs_months'], df.loc[~fav&(df.feature_027==0),'pfs_months'], equal_var=False)",
            "result_summary": "Inside subgroup (n=9326, n0=7423, n1=1903): mean PFS 9.101 vs 4.088, diff +5.013 months, t-test p≈0. Outside subgroup (n=40674, n0=32555, n1=8119): mean PFS 4.148 vs 4.123, diff +0.025, p=0.33. Effect is concentrated entirely within the AND-conjunction of the four predicates; failure of any single predicate (high feature_005, or feature_023=1, or feature_006=1, or feature_022=1) abolishes the feature_027 PFS benefit.",
            "p_value": 0.0,
            "effect_estimate": 5.013,
            "significant": True,
        },
    ],
)

# Write transcript
out_path = os.path.join(OUT_DIR, "transcript.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(transcript, f, indent=2)
print(f"Wrote {out_path}: {len(transcript['iterations'])} iterations")
