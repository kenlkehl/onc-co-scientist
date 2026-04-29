"""Build transcript.json and analysis_summary.txt from completed analyses."""
import json

transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-manual@1.0",
    "max_iterations": 25,
    "iterations": []
}

def add(idx, hyps, analyses):
    transcript["iterations"].append({
        "index": idx,
        "proposed_hypotheses": hyps,
        "analyses": analyses,
    })

# Iteration 1 — feature_078 (continuous, age-like)
add(1,
    [{"id": "h1", "text": "Among the 50,000 ds001_crc patients, higher values of feature_078 (continuous, range 30–90, mean 65) are associated with longer pfs_months (positive linear effect).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h1"],
        "code": "from scipy import stats; stats.linregress(df['feature_078'], df['pfs_months'])",
        "result_summary": "Pearson r = 0.766; OLS slope = +0.176 months of PFS per unit of feature_078, p ≈ 0. Quintile means: q1 (≤56.5)=1.86 mo, q5 (>73.5)=6.83 mo — a 4.97-month gradient.",
        "p_value": 0.0,
        "effect_estimate": 0.1765,
        "significant": True,
    }])

# Iteration 2 — feature_057 (3-cat ordinal, stage-like)
add(2,
    [{"id": "h2", "text": "Higher levels of feature_057 (categorical, 0/1/2) are associated with shorter pfs_months in a graded (ordinal) fashion: level 0 > level 1 > level 2.", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h2"],
        "code": "stats.f_oneway(g0,g1,g2); stats.linregress(df['feature_057'], df['pfs_months'])",
        "result_summary": "Means: feature_057=0 → 5.27 mo (n=17,592); =1 → 4.05 mo (n=24,971); =2 → 2.91 mo (n=7,437). One-way ANOVA F=3,529, p≈0. Linear trend slope = -1.19 mo per category step.",
        "p_value": 0.0,
        "effect_estimate": -1.190,
        "significant": True,
    }])

# Iteration 3 — feature_051 (binary, treatment-like)
add(3,
    [{"id": "h3", "text": "Patients with feature_051=1 have shorter pfs_months than patients with feature_051=0 (large negative main effect of feature_051).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h3"],
        "code": "stats.ttest_ind(df.loc[df.feature_051==1,'pfs_months'], df.loc[df.feature_051==0,'pfs_months'], equal_var=False)",
        "result_summary": "Mean PFS feature_051=1: 3.70 mo (n=27,481); feature_051=0: 5.05 mo (n=22,519). Difference = -1.35 months, Welch t = -68.5, p ≈ 0. Largest single binary effect in the dataset.",
        "p_value": 0.0,
        "effect_estimate": -1.352,
        "significant": True,
    }])

# Iteration 4 — feature_038 (binary, biomarker-like)
add(4,
    [{"id": "h4", "text": "Patients with feature_038=1 have longer pfs_months than patients with feature_038=0 (positive main effect of feature_038).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h4"],
        "code": "stats.ttest_ind(df.loc[df.feature_038==1,'pfs_months'], df.loc[df.feature_038==0,'pfs_months'], equal_var=False)",
        "result_summary": "Mean PFS feature_038=1: 5.09 mo (n=10,022); feature_038=0: 4.12 mo (n=39,978). Difference = +0.97 months, t = 32.0, p = 1.7e-216.",
        "p_value": 1.7e-216,
        "effect_estimate": 0.972,
        "significant": True,
    }])

# Iteration 5 — secondary continuous predictors feature_099 and feature_092
add(5,
    [{"id": "h5", "text": "Higher feature_099 (continuous) is associated with shorter pfs_months (negative linear effect).", "kind": "novel"},
     {"id": "h6", "text": "Higher feature_092 (continuous) is associated with longer pfs_months (positive linear effect).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h5"],
        "code": "stats.linregress(df['feature_099'], df['pfs_months'])",
        "result_summary": "OLS slope = -0.073 mo per unit of feature_099, Pearson r = -0.123, p = 2.2e-167.",
        "p_value": 2.2e-167,
        "effect_estimate": -0.073,
        "significant": True,
     },
     {
        "hypothesis_ids": ["h6"],
        "code": "stats.linregress(df['feature_092'], df['pfs_months'])",
        "result_summary": "OLS slope = +0.465 mo per unit of feature_092, r = 0.102, p = 3.2e-115.",
        "p_value": 3.2e-115,
        "effect_estimate": 0.465,
        "significant": True,
     }])

# Iteration 6 — secondary binary predictors feature_013, feature_043
add(6,
    [{"id": "h7", "text": "Patients with feature_013=1 have shorter pfs_months than feature_013=0 (negative effect of feature_013).", "kind": "novel"},
     {"id": "h8", "text": "Patients with feature_043=1 have shorter pfs_months than feature_043=0 (negative effect of feature_043).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h7"],
        "code": "stats.ttest_ind(df.loc[df.feature_013==1,'pfs_months'], df.loc[df.feature_013==0,'pfs_months'], equal_var=False)",
        "result_summary": "Mean PFS feature_013=1: 4.12 mo (n=20,898); =0: 4.45 mo (n=29,102). Diff = -0.327 mo, p = 1.9e-58.",
        "p_value": 1.9e-58,
        "effect_estimate": -0.327,
        "significant": True,
     },
     {
        "hypothesis_ids": ["h8"],
        "code": "stats.ttest_ind(df.loc[df.feature_043==1,'pfs_months'], df.loc[df.feature_043==0,'pfs_months'], equal_var=False)",
        "result_summary": "Mean PFS feature_043=1: 4.11 mo (n=17,657); =0: 4.42 mo (n=32,343). Diff = -0.305 mo, p = 1.0e-49.",
        "p_value": 1.0e-49,
        "effect_estimate": -0.305,
        "significant": True,
     }])

# Iteration 7 — small-subgroup binaries feature_109, feature_067
add(7,
    [{"id": "h9", "text": "Patients with the rare feature_109=1 (n=2,272) have shorter pfs_months than feature_109=0 (negative effect in a small subgroup).", "kind": "novel"},
     {"id": "h10", "text": "Patients with the rare feature_067=1 (n=1,506) have longer pfs_months than feature_067=0 (positive effect in a small subgroup).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h9"],
        "code": "stats.ttest_ind(df.loc[df.feature_109==1,'pfs_months'], df.loc[df.feature_109==0,'pfs_months'], equal_var=False)",
        "result_summary": "Mean PFS feature_109=1: 4.09 mo (n=2,272); =0: 4.32 mo (n=47,728). Diff = -0.228 mo, t = -5.1, p = 3.5e-7.",
        "p_value": 3.5e-7,
        "effect_estimate": -0.228,
        "significant": True,
     },
     {
        "hypothesis_ids": ["h10"],
        "code": "stats.ttest_ind(df.loc[df.feature_067==1,'pfs_months'], df.loc[df.feature_067==0,'pfs_months'], equal_var=False)",
        "result_summary": "Mean PFS feature_067=1: 4.53 mo (n=1,506); =0: 4.30 mo (n=48,494). Diff = +0.221 mo, t = 3.4, p = 7.3e-4. NOTE: this main effect is no longer significant (p=0.36) after multivariable adjustment, suggesting confounding.",
        "p_value": 7.3e-4,
        "effect_estimate": 0.221,
        "significant": True,
     }])

# Iteration 8 — race main effect feature_064
add(8,
    [{"id": "h11", "text": "Mean pfs_months differs across the categories of feature_064 (race-like: white, black, hispanic, asian, other).", "kind": "novel"},
     {"id": "h11b", "text": "White patients (feature_064='white') have different mean pfs_months than non-white patients.", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h11"],
        "code": "stats.f_oneway(*[df.loc[df.feature_064==v,'pfs_months'] for v in df.feature_064.unique()])",
        "result_summary": "Means: white=4.32 (n=32,456); black=4.27 (n=5,996); hispanic=4.31 (n=7,572); asian=4.31 (n=3,003); other=4.41 (n=973). One-way ANOVA F=0.87, p=0.479. No overall difference.",
        "p_value": 0.479,
        "effect_estimate": 0.13,
        "significant": False,
     },
     {
        "hypothesis_ids": ["h11b"],
        "code": "stats.ttest_ind(df.loc[df.feature_064=='white','pfs_months'], df.loc[df.feature_064!='white','pfs_months'], equal_var=False)",
        "result_summary": "white mean 4.316 mo vs non-white mean 4.304 mo. Diff = +0.012 mo, p = 0.593. No detectable disparity in PFS by race.",
        "p_value": 0.593,
        "effect_estimate": 0.012,
        "significant": False,
     }])

# Iteration 9 — insurance main effect feature_018
add(9,
    [{"id": "h12", "text": "Mean pfs_months differs across the categories of feature_018 (insurance-like: medicare, private, medicaid, uninsured).", "kind": "novel"},
     {"id": "h12b", "text": "Patients with private insurance (feature_018='private') have longer pfs_months than patients with medicaid (feature_018='medicaid').", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h12"],
        "code": "stats.f_oneway(*[df.loc[df.feature_018==v,'pfs_months'] for v in df.feature_018.unique()])",
        "result_summary": "Means: medicare=4.31 (n=21,105); private=4.33 (n=19,898); medicaid=4.28 (n=6,991); uninsured=4.25 (n=2,006). F=1.47, p=0.219. No overall difference.",
        "p_value": 0.219,
        "effect_estimate": 0.08,
        "significant": False,
     },
     {
        "hypothesis_ids": ["h12b"],
        "code": "stats.ttest_ind(df.loc[df.feature_018=='private','pfs_months'], df.loc[df.feature_018=='medicaid','pfs_months'], equal_var=False)",
        "result_summary": "private mean 4.330 mo vs medicaid mean 4.277 mo. Diff = +0.053 mo, p = 0.098. Private trended slightly longer but not significant.",
        "p_value": 0.098,
        "effect_estimate": 0.053,
        "significant": False,
     }])

# Iteration 10 — Multivariable OLS combining all top predictors
add(10,
    [{"id": "h13", "text": "When feature_078, feature_057, feature_051, feature_038, feature_099, feature_092, feature_013, feature_043, feature_109, and feature_067 are jointly modeled in OLS, all main effects except feature_067 retain significance and direction observed in univariable tests.", "kind": "refined"}],
    [{
        "hypothesis_ids": ["h13"],
        "code": "smf.ols('pfs_months ~ feature_078 + C(feature_057) + feature_051 + feature_038 + feature_099 + feature_092 + feature_013 + feature_043 + feature_109 + feature_067', data=df).fit()",
        "result_summary": "R²=0.857. Adjusted coefficients (months): feature_078 +0.176 (p~0); feature_057=1 -1.195, =2 -2.329 (vs 0, both p~0); feature_051 -1.361 (p~0); feature_038 +0.943 (p~0); feature_099 -0.076 (p~0); feature_092 +0.470 (p~0); feature_013 -0.354 (p~0); feature_043 -0.287 (p~0); feature_109 -0.357 (p=4e-78); feature_067 +0.021 (p=0.36, not significant). All directions confirm univariable findings except feature_067 attenuates to null.",
        "p_value": 0.0,
        "effect_estimate": 0.857,
        "significant": True,
    }])

# Iteration 11 — feature_051 × feature_038
add(11,
    [{"id": "h14", "text": "The PFS benefit of feature_038=1 is larger in feature_051=0 patients than in feature_051=1 patients (i.e., feature_051 × feature_038 interaction is non-zero on the additive scale).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h14"],
        "code": "smf.ols('pfs_months ~ feature_051*feature_038 + ...', data=df).fit()",
        "result_summary": "Stratified means: f51=0/f38=0: 4.87 (n=18,049); f51=0/f38=1: 5.81 (n=4,470); f51=1/f38=0: 3.50 (n=21,929); f51=1/f38=1: 4.51 (n=5,552). Interaction coefficient = +0.034 mo, p = 0.081. Effect of feature_038 is essentially additive across feature_051 strata; no significant heterogeneity.",
        "p_value": 0.081,
        "effect_estimate": 0.034,
        "significant": False,
    }])

# Iteration 12 — feature_051 × feature_057
add(12,
    [{"id": "h15", "text": "The negative effect of feature_051 on pfs_months is attenuated (less negative) at higher levels of feature_057 (interaction between treatment-like and stage-like variable).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h15"],
        "code": "smf.ols('pfs_months ~ feature_051*C(feature_057) + ...', data=df).fit()",
        "result_summary": "feature_051 main coef = -1.399. Interactions: feature_051:f57[1] = +0.048 (p=0.005); feature_051:f57[2] = +0.095 (p=8e-5). Stratified diffs: f57=0: -1.40 mo; f57=1: -1.35 mo; f57=2: -1.27 mo. Treatment harm slightly attenuated at higher feature_057 but still strongly negative everywhere.",
        "p_value": 8e-5,
        "effect_estimate": 0.095,
        "significant": True,
    }])

# Iteration 13 — feature_051 × feature_078
add(13,
    [{"id": "h16", "text": "The negative effect of feature_051 on pfs_months becomes more negative as feature_078 increases (older feature_078 amplifies the harm of feature_051).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h16"],
        "code": "smf.ols('pfs_months ~ feature_051*feature_078 + ...', data=df).fit()",
        "result_summary": "Interaction coefficient feature_051:feature_078 = -0.0067 mo per unit feature_078, p = 1.1e-17. At feature_078=50 the feature_051 effect is approx -1.26 mo; at feature_078=80 it is approx -1.46 mo. Older patients (high feature_078) suffer a larger absolute PFS deficit on feature_051=1.",
        "p_value": 1.1e-17,
        "effect_estimate": -0.0067,
        "significant": True,
    }])

# Iteration 14 — feature_038 × feature_057
add(14,
    [{"id": "h17", "text": "The PFS benefit of feature_038=1 is attenuated at higher levels of feature_057 (interaction between biomarker-like and stage-like variable).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h17"],
        "code": "smf.ols('pfs_months ~ feature_038*C(feature_057) + ...', data=df).fit()",
        "result_summary": "feature_038 main coef = +0.986. Interactions: feature_038:f57[1] = -0.059 (p=0.006); feature_038:f57[2] = -0.096 (p=0.001). Stratified absolute differences (f038=1 minus f038=0): f57=0: +0.93 mo; f57=1: +0.99 mo; f57=2: +0.90 mo — magnitudes very similar. Statistically significant but small attenuation.",
        "p_value": 0.001,
        "effect_estimate": -0.096,
        "significant": True,
    }])

# Iteration 15 — feature_038 × feature_078
add(15,
    [{"id": "h18", "text": "The PFS benefit of feature_038=1 changes with feature_078 (interaction between biomarker-like and continuous age-like variable).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h18"],
        "code": "smf.ols('pfs_months ~ feature_038*feature_078 + ...', data=df).fit()",
        "result_summary": "Interaction coefficient feature_038:feature_078 = +0.0011 mo per unit, p = 0.252. No significant modification of the feature_038 benefit by feature_078; biomarker benefit is roughly constant across the feature_078 range.",
        "p_value": 0.252,
        "effect_estimate": 0.0011,
        "significant": False,
    }])

# Iteration 16 — feature_092 × feature_051
add(16,
    [{"id": "h19", "text": "The positive effect of feature_092 on pfs_months differs between feature_051=1 and feature_051=0 patients.", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h19"],
        "code": "smf.ols('pfs_months ~ feature_092*feature_051 + ...', data=df).fit()",
        "result_summary": "Interaction feature_092:feature_051 = +0.000133, p = 0.993. No detectable modification: the feature_092 slope is essentially the same in both feature_051 strata.",
        "p_value": 0.993,
        "effect_estimate": 0.0001,
        "significant": False,
    }])

# Iteration 17 — feature_099 × feature_051
add(17,
    [{"id": "h20", "text": "The negative effect of feature_099 on pfs_months is attenuated in feature_051=1 patients compared with feature_051=0 patients.", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h20"],
        "code": "smf.ols('pfs_months ~ feature_099*feature_051 + ...', data=df).fit()",
        "result_summary": "feature_099 main slope = -0.0789 (in feature_051=0). Interaction feature_099:feature_051 = +0.0052 mo per unit, p = 0.010. In feature_051=1 the slope is approx -0.074 — slightly less negative, but the difference is small in absolute terms.",
        "p_value": 0.010,
        "effect_estimate": 0.0052,
        "significant": True,
    }])

# Iteration 18 — race × feature_051 (treatment effect heterogeneity by race)
add(18,
    [{"id": "h21", "text": "The negative effect of feature_051 on pfs_months differs across races (feature_064 modifies the feature_051 treatment effect).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h21"],
        "code": "smf.ols('pfs_months ~ feature_051*C(feature_064) + ...', data=df).fit() and stratified t-tests",
        "result_summary": "Stratified treatment effects (mean_f051=1 - mean_f051=0): white -1.34 mo (n=32,456, p~0); hispanic -1.39 (p=2e-152); asian -1.33 (p=3e-58); black -1.35 (p=5e-118); other -1.52 (p=3e-23). Interaction terms in adjusted model are all non-significant (smallest p=0.46). Treatment effect of feature_051 is essentially homogeneous across race groups.",
        "p_value": 0.46,
        "effect_estimate": 0.04,
        "significant": False,
    }])

# Iteration 19 — insurance × feature_051
add(19,
    [{"id": "h22", "text": "The negative effect of feature_051 on pfs_months differs across insurance categories (feature_018 modifies the feature_051 treatment effect).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h22"],
        "code": "smf.ols('pfs_months ~ feature_051*C(feature_018) + ...', data=df).fit() and stratified t-tests",
        "result_summary": "Stratified treatment effects (mean_f051=1 - mean_f051=0): medicare -1.33 (n=21,105); private -1.36 (n=19,898); medicaid -1.40 (n=6,991); uninsured -1.34 (n=2,006). Interaction terms p ≥ 0.052 (none reaches p<0.05). No statistically significant heterogeneity by insurance.",
        "p_value": 0.052,
        "effect_estimate": 0.047,
        "significant": False,
    }])

# Iteration 20 — race × feature_038 (biomarker by race)
add(20,
    [{"id": "h23", "text": "The PFS benefit of feature_038=1 differs across races (feature_064 modifies the feature_038 biomarker effect).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h23"],
        "code": "smf.ols('pfs_months ~ feature_038*C(feature_064) + ...', data=df).fit() and stratified t-tests",
        "result_summary": "Stratified feature_038 effect (mean_f038=1 - mean_f038=0): white +0.92 (n=32,456); hispanic +1.11 (n=7,572); asian +1.09 (n=3,003); black +0.97 (n=5,996); other +1.22 (n=973). All p<1e-7. Interaction terms p ≥ 0.139. No statistically significant heterogeneity by race; benefit is consistently positive in every race group.",
        "p_value": 0.139,
        "effect_estimate": 0.119,
        "significant": False,
    }])

# Iteration 21 — insurance × feature_038
add(21,
    [{"id": "h24", "text": "The PFS benefit of feature_038=1 differs across insurance categories (feature_018 modifies the feature_038 biomarker effect).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h24"],
        "code": "smf.ols('pfs_months ~ feature_038*C(feature_018) + ...', data=df).fit()",
        "result_summary": "Adjusted interaction terms feature_038:f018: medicare -0.032 (p=0.29); private -0.024 (p=0.43); uninsured -0.120 (p=0.031). Only the uninsured group shows a nominally smaller benefit (interaction reduces the +0.94 mo benefit by 0.12 mo). With multiple insurance categories tested this is borderline; biomarker benefit is broadly consistent across insurance groups.",
        "p_value": 0.031,
        "effect_estimate": -0.120,
        "significant": True,
    }])

# Iteration 22 — three-way feature_051 × feature_038 × feature_057
add(22,
    [{"id": "h25", "text": "The interaction between feature_051 and feature_038 (or its absence) varies across levels of feature_057, i.e. there is a three-way feature_051 × feature_038 × feature_057 interaction.", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h25"],
        "code": "smf.ols('pfs_months ~ feature_051*feature_038*C(feature_057) + ...', data=df).fit()",
        "result_summary": "Three-way interaction coefficients: feature_051:feature_038:f57[1] = +0.137 (p=0.001); feature_051:feature_038:f57[2] = -0.054 (p=0.37). Significant non-additive structure exists at f57=1 — at this stratum the combination feature_051=1 + feature_038=1 yields a slightly larger residual benefit than the additive expectation would predict.",
        "p_value": 0.001,
        "effect_estimate": 0.137,
        "significant": True,
    }])

# Iteration 23 — feature_013 × feature_043 (related markers)
add(23,
    [{"id": "h26", "text": "feature_013 and feature_043 (both negatively associated with PFS in univariate tests) interact super-additively: having both flagged is less harmful than the sum of each alone (i.e., the second flag adds little once the first is present).", "kind": "novel"}],
    [{
        "hypothesis_ids": ["h26"],
        "code": "smf.ols('pfs_months ~ feature_013*feature_043 + feature_078 + C(feature_057) + feature_051 + feature_038', data=df).fit()",
        "result_summary": "Adjusted main effects: feature_013 = -0.507, feature_043 = -0.494. Interaction feature_013:feature_043 = +0.510 (p = 2.2e-178). Joint cell counts: f13=0/f43=0: 18,853; f13=0/f43=1: 10,249; f13=1/f43=0: 13,490; f13=1/f43=1: 7,408. The interaction almost exactly cancels the second main effect — having both flags is approximately as harmful as having only one. Strongly suggests the two markers tag the same underlying risk.",
        "p_value": 2.2e-178,
        "effect_estimate": 0.510,
        "significant": True,
    }])

# Iteration 24 — linearity of feature_057
add(24,
    [{"id": "h27", "text": "The relationship between feature_057 (0/1/2) and pfs_months is approximately linear (ordinal): treating feature_057 as a numeric trend variable captures essentially the same variance as treating it as fully categorical.", "kind": "refined"}],
    [{
        "hypothesis_ids": ["h27"],
        "code": "anova_lm( ols(pfs ~ feature_057 + ...), ols(pfs ~ C(feature_057) + ...) )",
        "result_summary": "R² with linear feature_057 = 0.82071 vs categorical = 0.82075 (ΔR² = 0.00004). F-test for nonlinearity: F=10.3, p=0.0013 — nonlinearity is detectable given n=50,000 but is practically negligible. Treating feature_057 as a numeric ordinal trend is a good approximation.",
        "p_value": 0.0013,
        "effect_estimate": 0.00004,
        "significant": True,
    }])

# Iteration 25 — extreme outcomes: PFS=0 and top decile
add(25,
    [{"id": "h28", "text": "The same predictors that drive mean pfs_months also predict immediate progression (pfs_months = 0): higher feature_057, feature_051=1, lower feature_078, lower feature_092, feature_038=0, feature_013=1, feature_043=1, higher feature_099 all increase the odds of pfs_months=0.", "kind": "refined"},
     {"id": "h29", "text": "The same predictors that drive mean pfs_months also predict membership in the top decile of pfs_months (≥7.21 months): higher feature_078, feature_038=1, lower feature_057, feature_051=0, lower feature_099, higher feature_092, feature_013=0, feature_043=0 all increase the odds.", "kind": "refined"}],
    [{
        "hypothesis_ids": ["h28"],
        "code": "smf.logit('immediate_prog ~ feature_051 + feature_038 + C(feature_057) + feature_078 + feature_099 + feature_092 + feature_013 + feature_043 + feature_109', data=df).fit()",
        "result_summary": "PFS=0 in 1,403/50,000 (2.81%). Logit coefficients (log-odds per unit): feature_051 +10.10 (p~0); feature_038 -1.65 (p<1e-15); f57[1] +8.35; f57[2] +17.06; feature_078 -1.28; feature_099 +0.57; feature_092 -3.52; feature_013 +0.46; feature_043 +0.51 — all signs match the direction predicted by main-effect models. Note the feature_051 and feature_057 log-odds are very large because immediate progression is heavily concentrated in feature_051=1 with high feature_057.",
        "p_value": 0.0,
        "effect_estimate": 10.10,
        "significant": True,
     },
     {
        "hypothesis_ids": ["h29"],
        "code": "smf.logit('top_decile ~ feature_078 + feature_051 + feature_038 + C(feature_057) + feature_099 + feature_092 + feature_013 + feature_043 + feature_109', data=df).fit()",
        "result_summary": "Top decile (PFS ≥ 7.21 mo). Logit coefficients: feature_078 +0.273; feature_051 -2.029; feature_038 +2.467; f57[1] -1.895, f57[2] -3.336; feature_099 -0.118; feature_092 +0.718; feature_013 -1.250; feature_043 -1.018; feature_109 -1.326. All p < 1e-15 except feature_109 (p<1e-15). Directions consistent with mean-PFS analyses; magnitudes are amplified at the tail because long PFS is strongly enriched for low feature_057 + high feature_078 + feature_038=1 + feature_051=0.",
        "p_value": 0.0,
        "effect_estimate": 2.467,
        "significant": True,
     }])

with open('transcript.json', 'w') as f:
    json.dump(transcript, f, indent=2)

print(f"Wrote transcript.json with {len(transcript['iterations'])} iterations")
