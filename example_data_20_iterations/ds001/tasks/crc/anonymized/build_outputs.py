import json

with open('analysis_results.json') as f:
    R = json.load(f)

iterations = []

# Iter 1: feature_078
r = R['iter1_feature_078']
iterations.append({
    "index": 1,
    "proposed_hypotheses": [
        {"id": "h1", "kind": "novel",
         "text": "Higher values of feature_078 (a continuous covariate ranging from 30 to 90, distributed like patient age in years) are associated with longer pfs_months."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h1"],
         "code": "scipy.stats.pearsonr(df['feature_078'], df['pfs_months']); statsmodels OLS pfs_months ~ feature_078",
         "result_summary": f"Strong positive linear association: Pearson r={r['pearson_r']:.3f}, p<1e-300. OLS slope = +{r['ols_slope']:.3f} months per unit increase in feature_078 (intercept {r['intercept']:.3f}). PFS rises monotonically across feature_078 deciles from ~0.3 mo at value 30 to ~7.8 mo at value 80.",
         "p_value": 0.0, "effect_estimate": r['ols_slope'], "significant": True}
    ]
})

# Iter 2: feature_057
r = R['iter2_feature_057']
iterations.append({
    "index": 2,
    "proposed_hypotheses": [
        {"id": "h2", "kind": "novel",
         "text": "Higher levels of the 3-level ordinal feature_057 (values 0/1/2) are associated with shorter pfs_months in a monotonic dose-response pattern."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h2"],
         "code": "OLS pfs_months ~ feature_057; group means per level",
         "result_summary": f"Strong monotonic negative association. Mean pfs_months by level: 0={r['means']['0']:.2f} (n={r['ns']['0']}), 1={r['means']['1']:.2f} (n={r['ns']['1']}), 2={r['means']['2']:.2f} (n={r['ns']['2']}). OLS slope per level = {r['slope']:.3f} months, p<1e-300. Pattern resembles a performance-status / stage indicator.",
         "p_value": 0.0, "effect_estimate": r['slope'], "significant": True}
    ]
})

# Iter 3: feature_051
r = R['iter3_feature_051']
iterations.append({
    "index": 3,
    "proposed_hypotheses": [
        {"id": "h3", "kind": "novel",
         "text": "Patients with feature_051=1 have shorter pfs_months than patients with feature_051=0."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h3"],
         "code": "ttest_ind(df.pfs_months by feature_051)",
         "result_summary": f"Strong negative effect. Mean PFS = {r['mean_0']:.3f} (feature_051=0, n={r['n_0']}) vs {r['mean_1']:.3f} (feature_051=1, n={r['n_1']}); difference {r['diff_1_minus_0']:.3f} months, p<1e-300. feature_051 is one of the largest single binary risk factors in the cohort.",
         "p_value": 0.0, "effect_estimate": r['diff_1_minus_0'], "significant": True}
    ]
})

# Iter 4: feature_038
r = R['iter4_feature_038']
iterations.append({
    "index": 4,
    "proposed_hypotheses": [
        {"id": "h4", "kind": "novel",
         "text": "Patients with feature_038=1 have longer pfs_months than patients with feature_038=0 (potentially a treatment or favorable biomarker indicator)."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h4"],
         "code": "ttest_ind(df.pfs_months by feature_038)",
         "result_summary": f"Strong positive effect. Mean PFS = {r['mean_0']:.3f} (feature_038=0, n={r['n_0']}) vs {r['mean_1']:.3f} (feature_038=1, n={r['n_1']}); difference +{r['diff_1_minus_0']:.3f} months, p<1e-300. feature_038 has prevalence ~20% and is the largest favorable binary marker for PFS.",
         "p_value": 0.0, "effect_estimate": r['diff_1_minus_0'], "significant": True}
    ]
})

# Iter 5: feature_013, feature_043
r = R['iter5_more_binaries']
iterations.append({
    "index": 5,
    "proposed_hypotheses": [
        {"id": "h5a", "kind": "novel",
         "text": "feature_013=1 is associated with shorter pfs_months than feature_013=0."},
        {"id": "h5b", "kind": "novel",
         "text": "feature_043=1 is associated with shorter pfs_months than feature_043=0."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h5a"],
         "code": "ttest_ind by feature_013",
         "result_summary": f"Mean PFS {r['feature_013']['mean_0']:.3f} (=0) vs {r['feature_013']['mean_1']:.3f} (=1); diff {r['feature_013']['diff']:.3f}, p={r['feature_013']['p']:.2e}. Modest but highly significant negative effect.",
         "p_value": r['feature_013']['p'], "effect_estimate": r['feature_013']['diff'], "significant": True},
        {"hypothesis_ids": ["h5b"],
         "code": "ttest_ind by feature_043",
         "result_summary": f"Mean PFS {r['feature_043']['mean_0']:.3f} (=0) vs {r['feature_043']['mean_1']:.3f} (=1); diff {r['feature_043']['diff']:.3f}, p={r['feature_043']['p']:.2e}. Similar magnitude negative effect to feature_013.",
         "p_value": r['feature_043']['p'], "effect_estimate": r['feature_043']['diff'], "significant": True}
    ]
})

# Iter 6: feature_092, feature_099 continuous
r = R['iter6_continuous']
iterations.append({
    "index": 6,
    "proposed_hypotheses": [
        {"id": "h6a", "kind": "novel",
         "text": "feature_092 (continuous, range ~1.5-5.5) is positively correlated with pfs_months."},
        {"id": "h6b", "kind": "novel",
         "text": "feature_099 (continuous, right-skewed with many zeros) is negatively correlated with pfs_months."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h6a"],
         "code": "pearsonr(df.feature_092, df.pfs_months)",
         "result_summary": f"Pearson r = +{r['feature_092']['r']:.4f}, p={r['feature_092']['p']:.2e}. Modest positive correlation; consistent with a favorable lab/biomarker (e.g., serum protein-like).",
         "p_value": r['feature_092']['p'], "effect_estimate": r['feature_092']['r'], "significant": True},
        {"hypothesis_ids": ["h6b"],
         "code": "pearsonr(df.feature_099, df.pfs_months)",
         "result_summary": f"Pearson r = {r['feature_099']['r']:.4f}, p={r['feature_099']['p']:.2e}. Modest negative correlation; consistent with disease-burden indicator.",
         "p_value": r['feature_099']['p'], "effect_estimate": r['feature_099']['r'], "significant": True}
    ]
})

# Iter 7: race feature_064
r = R['iter7_race']
iterations.append({
    "index": 7,
    "proposed_hypotheses": [
        {"id": "h7", "kind": "novel",
         "text": "Mean pfs_months differs across categories of feature_064 (race: white, hispanic, black, asian, other)."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h7"],
         "code": "f_oneway across feature_064 categories; pairwise vs white",
         "result_summary": f"No overall difference. ANOVA F={r['overall_F']:.3f}, p={r['overall_p']:.3f}. Group means: white {r['means']['white']:.3f}, hispanic {r['means']['hispanic']:.3f}, black {r['means']['black']:.3f}, asian {r['means']['asian']:.3f}, other {r['means']['other']:.3f}. Largest pairwise gap: black vs white = {r['pairwise_vs_white']['black']['diff_vs_white']:.3f} mo (p={r['pairwise_vs_white']['black']['p']:.3f}).",
         "p_value": r['overall_p'], "effect_estimate": r['pairwise_vs_white']['black']['diff_vs_white'], "significant": False}
    ]
})

# Iter 8: insurance
r = R['iter8_insurance']
iterations.append({
    "index": 8,
    "proposed_hypotheses": [
        {"id": "h8", "kind": "novel",
         "text": "Mean pfs_months differs across categories of feature_018 (insurance: medicare, private, medicaid, uninsured)."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h8"],
         "code": "f_oneway across feature_018",
         "result_summary": f"No overall difference. ANOVA F={r['overall_F']:.3f}, p={r['overall_p']:.3f}. Means: medicare {r['means']['medicare']:.3f}, private {r['means']['private']:.3f}, medicaid {r['means']['medicaid']:.3f}, uninsured {r['means']['uninsured']:.3f}. Spread is < 0.1 months across all groups.",
         "p_value": r['overall_p'], "effect_estimate": r['means']['private'] - r['means']['uninsured'], "significant": False}
    ]
})

# Iter 9: multivariable
r = R['iter9_mvm']
iterations.append({
    "index": 9,
    "proposed_hypotheses": [
        {"id": "h9", "kind": "novel",
         "text": "After adjustment for feature_078, feature_057, feature_051, and feature_038, the binary risk markers feature_013 and feature_043 retain independent negative associations with pfs_months."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h9"],
         "code": "OLS pfs_months ~ feature_078 + feature_057 + feature_051 + feature_038 + feature_013 + feature_043 + feature_099 + feature_092",
         "result_summary": f"R^2={r['r2']:.4f}. Adjusted coefficients (months): feature_078 +{r['params']['feature_078']:.3f} per unit, feature_057 {r['params']['feature_057']:.3f} per level, feature_051 {r['params']['feature_051']:.3f}, feature_038 +{r['params']['feature_038']:.3f}, feature_013 {r['params']['feature_013']:.3f}, feature_043 {r['params']['feature_043']:.3f}, feature_099 {r['params']['feature_099']:.4f}, feature_092 +{r['params']['feature_092']:.3f}. All p<1e-200 — every effect remains independently significant; no major confounding among them.",
         "p_value": 0.0, "effect_estimate": r['params']['feature_013'], "significant": True}
    ]
})

# Iter 10: 038 x 051
r = R['iter10_int_038x051']
iterations.append({
    "index": 10,
    "proposed_hypotheses": [
        {"id": "h10", "kind": "novel",
         "text": "The PFS benefit of feature_038=1 is modified by feature_051 status (interaction): the gain from feature_038 differs between feature_051=0 and feature_051=1 patients."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h10"],
         "code": "OLS pfs_months ~ feature_038 * feature_051 + feature_078 + feature_057",
         "result_summary": f"No multiplicative interaction. Coefficient on feature_038:feature_051 = {r['interaction_coef']:.4f}, p={r['interaction_p']:.3f}. Effects of feature_038 (+{r['main_038']:.3f} mo) and feature_051 ({r['main_051']:.3f} mo) appear additive, not synergistic.",
         "p_value": r['interaction_p'], "effect_estimate": r['interaction_coef'], "significant": False}
    ]
})

# Iter 11: 038 by race
r = R['iter11_038_by_race']
iterations.append({
    "index": 11,
    "proposed_hypotheses": [
        {"id": "h11", "kind": "novel",
         "text": "The benefit of feature_038=1 on pfs_months is heterogeneous across feature_064 race categories (potential treatment-effect modification)."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h11"],
         "code": "Stratified ttest by race; F-test of feature_038 * C(feature_064) interaction",
         "result_summary": f"Borderline significant interaction (F={r['_interaction_F']:.3f}, p={r['_interaction_p']:.3f}). Stratified diffs: white +{r['white']['diff']:.3f} (p={r['white']['p']:.1e}), hispanic +{r['hispanic']['diff']:.3f}, black +{r['black']['diff']:.3f}, asian +{r['asian']['diff']:.3f}, other +{r['other']['diff']:.3f}. All groups benefit substantially; hispanic/asian/other show numerically larger gains but overlapping CIs.",
         "p_value": r['_interaction_p'], "effect_estimate": r['hispanic']['diff'] - r['white']['diff'], "significant": True}
    ]
})

# Iter 12: 051 by race
r = R['iter12_051_by_race']
iterations.append({
    "index": 12,
    "proposed_hypotheses": [
        {"id": "h12", "kind": "novel",
         "text": "The negative effect of feature_051=1 on pfs_months differs across feature_064 race categories."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h12"],
         "code": "Stratified ttest by race; F-test of feature_051 * C(feature_064) interaction",
         "result_summary": f"No heterogeneity. F={r['_interaction_F']:.3f}, p={r['_interaction_p']:.3f}. Stratified diffs all clustered tightly around -1.35 mo: white {r['white']['diff']:.3f}, hispanic {r['hispanic']['diff']:.3f}, black {r['black']['diff']:.3f}, asian {r['asian']['diff']:.3f}, other {r['other']['diff']:.3f}. feature_051 acts uniformly across racial groups.",
         "p_value": r['_interaction_p'], "effect_estimate": r['white']['diff'], "significant": False}
    ]
})

# Iter 13: 038 by insurance
r = R['iter13_038_by_insurance']
iterations.append({
    "index": 13,
    "proposed_hypotheses": [
        {"id": "h13", "kind": "novel",
         "text": "The benefit of feature_038=1 on pfs_months is heterogeneous across feature_018 insurance categories."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h13"],
         "code": "Stratified ttest by insurance; F-test interaction",
         "result_summary": f"No significant heterogeneity. F={r['_interaction_F']:.3f}, p={r['_interaction_p']:.3f}. Diffs: medicare +{r['medicare']['diff']:.3f}, private +{r['private']['diff']:.3f}, medicaid +{r['medicaid']['diff']:.3f}, uninsured +{r['uninsured']['diff']:.3f}. Medicaid trends toward larger benefit but not significantly so.",
         "p_value": r['_interaction_p'], "effect_estimate": r['medicaid']['diff'] - r['private']['diff'], "significant": False}
    ]
})

# Iter 14: 038 x 057
r = R['iter14_int_038x057']
iterations.append({
    "index": 14,
    "proposed_hypotheses": [
        {"id": "h14", "kind": "refined",
         "text": "The benefit of feature_038=1 on pfs_months is attenuated as feature_057 (3-level ordinal disease severity) increases (negative interaction)."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h14"],
         "code": "OLS pfs_months ~ feature_038 * feature_057 + feature_078 + feature_051; stratified means",
         "result_summary": f"Small but significant negative interaction. Interaction coef = {r['interaction_coef']:.4f} per level, p={r['interaction_p']:.2e}. Stratified diffs: feature_057=0 +{r['stratified']['0']['diff']:.3f}, =1 +{r['stratified']['1']['diff']:.3f}, =2 +{r['stratified']['2']['diff']:.3f}. The benefit shrinks modestly at higher feature_057 but remains highly significant in every stratum.",
         "p_value": r['interaction_p'], "effect_estimate": r['interaction_coef'], "significant": True}
    ]
})

# Iter 15: feature_078 x feature_038
r = R['iter15_int_age038']
iterations.append({
    "index": 15,
    "proposed_hypotheses": [
        {"id": "h15", "kind": "novel",
         "text": "The PFS benefit of feature_038=1 differs by feature_078 (age-like covariate): older patients gain more (or less) than younger patients."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h15"],
         "code": "OLS pfs_months ~ feature_078 * feature_038 + feature_057 + feature_051",
         "result_summary": f"No interaction with feature_078. Coef = {r['interaction_coef']:.4f} per unit, p={r['interaction_p']:.3f}. The feature_038 benefit is constant across the entire feature_078 range.",
         "p_value": r['interaction_p'], "effect_estimate": r['interaction_coef'], "significant": False}
    ]
})

# Iter 16: lab adjusted
r = R['iter16_lab_adjusted']
iterations.append({
    "index": 16,
    "proposed_hypotheses": [
        {"id": "h16", "kind": "refined",
         "text": "feature_092 and feature_099 retain independent associations with pfs_months after adjusting for feature_078 and feature_057."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h16"],
         "code": "OLS pfs_months ~ feature_099 + feature_092 + feature_078 + feature_057",
         "result_summary": f"Both retain strong independent effects. feature_099 coef = {r['feature_099_coef']:.4f} mo per unit (p<1e-200); feature_092 coef = +{r['feature_092_coef']:.3f} mo per unit (p<1e-200). The continuous lab-like markers are not confounded by age or severity.",
         "p_value": 0.0, "effect_estimate": r['feature_092_coef'], "significant": True}
    ]
})

# Iter 17: feature_071
r = R['iter17_feature_071']
iterations.append({
    "index": 17,
    "proposed_hypotheses": [
        {"id": "h17", "kind": "novel",
         "text": "The 11-level ordinal feature_071 is associated with pfs_months either monotonically or in a U-shape."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h17"],
         "code": "pearsonr(feature_071, pfs_months); OLS adjusted",
         "result_summary": f"No association. Pearson r={r['pearson_r']:.4f}, p={r['pearson_p']:.3f}. Adjusted slope {r['adj_coef']:.4f} mo per level, p={r['adj_p']:.3f}. feature_071 is not predictive.",
         "p_value": r['pearson_p'], "effect_estimate": r['pearson_r'], "significant": False}
    ]
})

# Iter 18: 5-level ordinals
r = R['iter18_5level_ords']
iterations.append({
    "index": 18,
    "proposed_hypotheses": [
        {"id": "h18", "kind": "novel",
         "text": "The 5-level ordinals feature_025, feature_075, feature_026, feature_096, and feature_033 each correlate with pfs_months."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h18"],
         "code": "pearsonr for each ordinal vs pfs_months",
         "result_summary": "All five ordinals are uninformative for PFS. " + "; ".join([f"{c} r={r[c]['r']:.4f} p={r[c]['p']:.3f}" for c in r]) + ". None reach significance; all |r|<0.01.",
         "p_value": min(r[c]['p'] for c in r), "effect_estimate": max((abs(r[c]['r']) for c in r)), "significant": False}
    ]
})

# Iter 19: 092 x 038
r = R['iter19_int_092x038']
iterations.append({
    "index": 19,
    "proposed_hypotheses": [
        {"id": "h19", "kind": "refined",
         "text": "The PFS benefit of feature_038=1 is smaller when feature_092 is high (negative interaction between feature_092 and feature_038)."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h19"],
         "code": "OLS pfs_months ~ feature_092 * feature_038 + feature_078 + feature_057 + feature_051",
         "result_summary": f"Significant negative interaction. Coef = {r['interaction_coef']:.4f} mo per (unit feature_092 * feature_038), p={r['interaction_p']:.2e}. Patients with low feature_092 gain a larger absolute PFS benefit from feature_038 than patients with already-high feature_092 (consistent with a ceiling effect on a favorable host marker).",
         "p_value": r['interaction_p'], "effect_estimate": r['interaction_coef'], "significant": True}
    ]
})

# Iter 20: 051 x 057
r = R['iter20_int_051x057']
iterations.append({
    "index": 20,
    "proposed_hypotheses": [
        {"id": "h20", "kind": "refined",
         "text": "The harm associated with feature_051=1 is attenuated at higher feature_057 levels (positive interaction; diminishing marginal harm in worse-prognosis strata)."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h20"],
         "code": "OLS pfs_months ~ feature_051 * feature_057 + feature_078 + feature_038; stratified diffs",
         "result_summary": f"Small positive interaction (coef +{r['interaction_coef']:.4f} per level, p={r['interaction_p']:.3e}). Stratified diffs: feature_057=0 {r['stratified']['0']['diff']:.3f}, =1 {r['stratified']['1']['diff']:.3f}, =2 {r['stratified']['2']['diff']:.3f}. The feature_051 hit is largest at feature_057=0 and slightly smaller at higher severity, consistent with floor effects.",
         "p_value": r['interaction_p'], "effect_estimate": r['interaction_coef'], "significant": True}
    ]
})

# Iter 21: feature_078 by race
r = R['iter21_age_by_race']
iterations.append({
    "index": 21,
    "proposed_hypotheses": [
        {"id": "h21", "kind": "novel",
         "text": "The strong positive correlation between feature_078 and pfs_months is consistent across feature_064 race subgroups."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h21"],
         "code": "pearsonr(feature_078, pfs_months) within each race stratum",
         "result_summary": f"Highly consistent. Stratum r values: white {r['white']['r']:.3f}, hispanic {r['hispanic']['r']:.3f}, asian {r['asian']['r']:.3f}, black {r['black']['r']:.3f}, other {r['other']['r']:.3f}. All p<1e-200. The feature_078 -> PFS relationship is essentially the same in every group.",
         "p_value": 0.0, "effect_estimate": r['white']['r'], "significant": True}
    ]
})

# Iter 22: other binaries
r = R['iter22_other_binaries']
iterations.append({
    "index": 22,
    "proposed_hypotheses": [
        {"id": "h22", "kind": "novel",
         "text": "Among prevalent (>20%) binaries not yet tested (feature_106, feature_005, feature_122, feature_088, feature_007), at least one is associated with pfs_months."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h22"],
         "code": "ttest_ind for each binary vs pfs_months",
         "result_summary": "All five are null. " + "; ".join([f"{c} diff={r[c]['diff']:.4f} p={r[c]['p']:.2f}" for c in r]) + ". None reaches significance — in this cohort the prognostic structure is concentrated in a handful of variables.",
         "p_value": min(r[c]['p'] for c in r), "effect_estimate": max(r[c]['diff'] for c in r), "significant": False}
    ]
})

# Iter 23: 3-way 038x051 by race
r = R['iter23_3way']
iterations.append({
    "index": 23,
    "proposed_hypotheses": [
        {"id": "h23", "kind": "refined",
         "text": "The (already-null) feature_038 x feature_051 interaction differs across feature_064 race categories (3-way modification)."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h23"],
         "code": "OLS feature_038*feature_051 fit within each race stratum",
         "result_summary": "No 3-way modification detected. Stratum interaction coefs: " + "; ".join([f"{c}: {r[c]['int_coef']:.3f} (p={r[c]['int_p']:.2f}, n={r[c]['n']})" for c in r]) + ". No stratum reaches significance and signs differ — consistent with no genuine 3-way effect.",
         "p_value": min(r[c]['int_p'] for c in r), "effect_estimate": r['asian']['int_coef'], "significant": False}
    ]
})

# Iter 24: comprehensive multivariable
r = R['iter24_full_mvm']
iterations.append({
    "index": 24,
    "proposed_hypotheses": [
        {"id": "h24", "kind": "refined",
         "text": "A comprehensive multivariable OLS that adjusts for feature_078, feature_057, feature_051, feature_038, feature_013, feature_043, feature_099, feature_092, feature_109, feature_067, race (feature_064), and insurance (feature_018) explains the majority of pfs_months variance, with race and insurance having no independent effect."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h24"],
         "code": "OLS pfs_months ~ all listed covariates",
         "result_summary": f"R^2 = {r['r2']:.4f} (adj {r['r2_adj']:.4f}). All clinical covariates remain independently significant (p<1e-50) with similar magnitudes to univariate fits: feature_078 +{r['params']['feature_078']:.3f}, feature_057 {r['params']['feature_057']:.3f}, feature_051 {r['params']['feature_051']:.3f}, feature_038 +{r['params']['feature_038']:.3f}, feature_013 {r['params']['feature_013']:.3f}, feature_043 {r['params']['feature_043']:.3f}, feature_099 {r['params']['feature_099']:.4f}, feature_092 +{r['params']['feature_092']:.3f}, feature_109 {r['params']['feature_109']:.3f}, feature_067 +{r['params']['feature_067']:.3f}. Every level of feature_064 (race) and feature_018 (insurance) has p>0.2 — these are unrelated to PFS once clinical features are controlled for.",
         "p_value": 0.0, "effect_estimate": r['r2'], "significant": True}
    ]
})

# Iter 25: feature_078 x feature_057
r = R['iter25_int_age_ps']
iterations.append({
    "index": 25,
    "proposed_hypotheses": [
        {"id": "h25", "kind": "refined",
         "text": "The positive feature_078 - PFS slope flattens at higher levels of feature_057 (interaction: severity reduces the per-unit gain from feature_078)."}
    ],
    "analyses": [
        {"hypothesis_ids": ["h25"],
         "code": "OLS pfs_months ~ feature_078 * feature_057 + feature_051 + feature_038",
         "result_summary": f"Significant negative interaction. Coef = {r['interaction_coef']:.4f} mo per (unit feature_078 * level feature_057), p={r['interaction_p']:.2e}. Per-unit gain in PFS from feature_078 declines slightly as feature_057 rises — the largest PFS values are observed in patients with high feature_078 AND low feature_057.",
         "p_value": r['interaction_p'], "effect_estimate": r['interaction_coef'], "significant": True}
    ]
})

transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-interactive@manual-2026-04-28",
    "max_iterations": 25,
    "iterations": iterations
}

with open('transcript.json', 'w') as f:
    json.dump(transcript, f, indent=2)

print(f"Wrote transcript.json with {len(iterations)} iterations")
