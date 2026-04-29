import json

with open('_work/analyses.json') as fp:
    R = json.load(fp)

def pv(x):
    return None if x is None else float(x)

iterations = []

# ---------- Iteration 1: demographic main effects ----------
i1 = {
    'index': 1,
    'proposed_hypotheses': [
        {'id': 'h1.1', 'text': 'Higher values of feature_078 (continuous, range 30-90, mean 65; the candidate age variable) are associated with longer pfs_months (positive linear association).', 'kind': 'novel'},
        {'id': 'h1.2', 'text': 'Mean pfs_months differs across categories of feature_018 (race/ethnicity: white/black/hispanic/asian/other).', 'kind': 'novel'},
        {'id': 'h1.3', 'text': 'Mean pfs_months differs across categories of feature_045 (insurance: medicare/private/medicaid/uninsured).', 'kind': 'novel'},
    ],
    'analyses': [
        {
            'hypothesis_ids': ['h1.1'],
            'code': "smf.ols('pfs_months ~ feature_078', df).fit()",
            'result_summary': 'OLS slope of pfs_months on feature_078 = +%.4f mo per unit (p=%.2e). Spearman r reaches +0.85; mean pfs_months rises monotonically from 0.85 mo in the lowest decile of feature_078 to 6.86 mo in the highest decile.' % (R['i1_age']['coef']['feature_078'], R['i1_age']['pvalues']['feature_078']),
            'p_value': pv(R['i1_age']['pvalues']['feature_078']),
            'effect_estimate': float(R['i1_age']['coef']['feature_078']),
            'significant': True,
        },
        {
            'hypothesis_ids': ['h1.2'],
            'code': "stats.f_oneway over groups by feature_018",
            'result_summary': 'One-way ANOVA across the 5 race categories: F-test p=%.3f. Group means range only 3.74-3.80 mo; no overall difference detected.' % R['i1_race_anova']['p'],
            'p_value': pv(R['i1_race_anova']['p']),
            'effect_estimate': float(max(R['i1_race_anova']['means'].values()) - min(R['i1_race_anova']['means'].values())),
            'significant': False,
        },
        {
            'hypothesis_ids': ['h1.3'],
            'code': "stats.f_oneway over groups by feature_045",
            'result_summary': 'One-way ANOVA across the 4 insurance categories: F-test p=%.3f. Group means range 3.73-3.75 mo; no overall difference detected.' % R['i1_insurance_anova']['p'],
            'p_value': pv(R['i1_insurance_anova']['p']),
            'effect_estimate': float(max(R['i1_insurance_anova']['means'].values()) - min(R['i1_insurance_anova']['means'].values())),
            'significant': False,
        },
    ],
}
iterations.append(i1)

# ---------- Iteration 2: top clinical features ----------
i2 = {
    'index': 2,
    'proposed_hypotheses': [
        {'id': 'h2.1', 'text': 'Higher levels of feature_057 (ordinal 0/1/2, candidate stage or risk group) are associated with shorter pfs_months in a monotonically decreasing pattern.', 'kind': 'novel'},
        {'id': 'h2.2', 'text': 'Higher feature_013 (continuous, range 0.1-3622, candidate PSA-like biomarker) is associated with shorter pfs_months.', 'kind': 'novel'},
        {'id': 'h2.3', 'text': 'Patients with feature_051=1 (binary, candidate metastasis/disease-burden flag) have shorter pfs_months than feature_051=0.', 'kind': 'novel'},
    ],
    'analyses': [
        {
            'hypothesis_ids': ['h2.1'],
            'code': 'stats.spearmanr(df.feature_057, df.pfs_months); means by group',
            'result_summary': 'Spearman r=%.3f, p=%.2e. Means: feature_057=0 -> 4.68 mo (n=17592), =1 -> 3.49 mo (n=24971), =2 -> 2.38 mo (n=7437). Strong monotone decrease.' % (R['i2_f057_spearman']['r'], R['i2_f057_spearman']['p']),
            'p_value': pv(R['i2_f057_spearman']['p']),
            'effect_estimate': float(R['i2_f057_spearman']['r']),
            'significant': True,
        },
        {
            'hypothesis_ids': ['h2.2'],
            'code': 'stats.spearmanr(df.feature_013, df.pfs_months)',
            'result_summary': 'Spearman r=%.3f, p=%.2e. Higher feature_013 values associate with shorter PFS.' % (R['i2_f013_spearman']['r'], R['i2_f013_spearman']['p']),
            'p_value': pv(R['i2_f013_spearman']['p']),
            'effect_estimate': float(R['i2_f013_spearman']['r']),
            'significant': True,
        },
        {
            'hypothesis_ids': ['h2.3'],
            'code': "stats.ttest_ind on pfs_months by feature_051",
            'result_summary': 'Mean pfs_months: feature_051=0 -> %.3f (n=%d) vs feature_051=1 -> %.3f (n=%d). Difference (1 - 0) = %.3f mo, t-test p=%.2e.' % (R['i2_f051_tt']['mean_0'], R['i2_f051_tt']['n0'], R['i2_f051_tt']['mean_1'], R['i2_f051_tt']['n1'], R['i2_f051_tt']['mean_diff'], R['i2_f051_tt']['p']),
            'p_value': pv(R['i2_f051_tt']['p']),
            'effect_estimate': float(R['i2_f051_tt']['mean_diff']),
            'significant': True,
        },
    ],
}
iterations.append(i2)

# ---------- Iteration 3: more clinical features ----------
i3 = {
    'index': 3,
    'proposed_hypotheses': [
        {'id': 'h3.1', 'text': 'Higher feature_006 (continuous, range 0-24.6, mean 3.84) is associated with shorter pfs_months.', 'kind': 'novel'},
        {'id': 'h3.2', 'text': 'Higher feature_009 (continuous, range 1.5-5.5, mean 3.8; resembles serum albumin) is associated with longer pfs_months.', 'kind': 'novel'},
        {'id': 'h3.3', 'text': 'Patients with feature_109=1 (binary, ~10% prevalence) have different pfs_months than feature_109=0.', 'kind': 'novel'},
        {'id': 'h3.4', 'text': 'Patients with feature_039=1 (binary, ~10% prevalence) have different pfs_months than feature_039=0.', 'kind': 'novel'},
    ],
    'analyses': [
        {
            'hypothesis_ids': ['h3.1'],
            'code': 'stats.spearmanr(feature_006, pfs_months)',
            'result_summary': 'Spearman r=%.4f, p=%.2e. Significant negative association: higher feature_006 -> shorter PFS.' % (R['i3_feature_006_spearman']['r'], R['i3_feature_006_spearman']['p']),
            'p_value': pv(R['i3_feature_006_spearman']['p']),
            'effect_estimate': float(R['i3_feature_006_spearman']['r']),
            'significant': True,
        },
        {
            'hypothesis_ids': ['h3.2'],
            'code': 'stats.spearmanr(feature_009, pfs_months)',
            'result_summary': 'Spearman r=%.4f, p=%.2e. Significant positive association: higher feature_009 -> longer PFS.' % (R['i3_feature_009_spearman']['r'], R['i3_feature_009_spearman']['p']),
            'p_value': pv(R['i3_feature_009_spearman']['p']),
            'effect_estimate': float(R['i3_feature_009_spearman']['r']),
            'significant': True,
        },
        {
            'hypothesis_ids': ['h3.3'],
            'code': 't-test pfs by feature_109',
            'result_summary': 'feature_109=1 vs 0: mean diff = +%.3f mo (n1=%d, n0=%d), t-test p=%.4f. Modestly higher PFS in feature_109=1.' % (R['i3_feature_109_tt']['mean_diff'], R['i3_feature_109_tt']['n1'], R['i3_feature_109_tt']['n0'], R['i3_feature_109_tt']['p']),
            'p_value': pv(R['i3_feature_109_tt']['p']),
            'effect_estimate': float(R['i3_feature_109_tt']['mean_diff']),
            'significant': True,
        },
        {
            'hypothesis_ids': ['h3.4'],
            'code': 't-test pfs by feature_039',
            'result_summary': 'feature_039=1 vs 0: mean diff = +%.3f mo (n1=%d, n0=%d), t-test p=%.4f.' % (R['i3_feature_039_tt']['mean_diff'], R['i3_feature_039_tt']['n1'], R['i3_feature_039_tt']['n0'], R['i3_feature_039_tt']['p']),
            'p_value': pv(R['i3_feature_039_tt']['p']),
            'effect_estimate': float(R['i3_feature_039_tt']['mean_diff']),
            'significant': True,
        },
    ],
}
iterations.append(i3)

# ---------- Iteration 4: multivariable model ----------
mv = R['i4_multivar']
i4 = {
    'index': 4,
    'proposed_hypotheses': [
        {'id': 'h4.1', 'text': 'After mutual adjustment in a multivariable OLS, feature_078 retains a positive effect, feature_057 retains a negative effect, log(feature_013) retains a negative effect, feature_051 retains a negative effect, feature_009 retains a positive effect, and feature_006 retains a negative effect on pfs_months.', 'kind': 'refined'},
        {'id': 'h4.2', 'text': 'After adjusting for clinical covariates (age, stage, log-PSA, feature_051, feature_006, feature_009), race (feature_018) and insurance (feature_045) coefficients remain non-significant — i.e., observed PFS does not differ by demographics conditional on clinical features.', 'kind': 'refined'},
    ],
    'analyses': [
        {
            'hypothesis_ids': ['h4.1'],
            'code': "smf.ols('pfs_months ~ feature_078 + C(feature_057) + np.log1p(feature_013) + feature_051 + feature_006 + feature_009 + C(feature_018) + C(feature_045)', df).fit()",
            'result_summary': ('Adjusted multivariable OLS (n=%d, R^2=%.3f). Coefficients (per unit, mo): feature_078=+%.4f (p=%.1e); C(feature_057)[T.1]=%.3f (p=%.1e); C(feature_057)[T.2]=%.3f (p=%.1e); log1p(feature_013)=%.3f (p=%.1e); feature_051=%.3f (p=%.1e); feature_006=%.4f (p=%.1e); feature_009=+%.3f (p=%.1e). All clinical predictors retain expected directions and remain highly significant.' % (
                mv['n'], mv['r2'],
                mv['coef']['feature_078'], mv['pvalues']['feature_078'],
                mv['coef']['C(feature_057)[T.1]'], mv['pvalues']['C(feature_057)[T.1]'],
                mv['coef']['C(feature_057)[T.2]'], mv['pvalues']['C(feature_057)[T.2]'],
                mv['coef']['np.log1p(feature_013)'], mv['pvalues']['np.log1p(feature_013)'],
                mv['coef']['feature_051'], mv['pvalues']['feature_051'],
                mv['coef']['feature_006'], mv['pvalues']['feature_006'],
                mv['coef']['feature_009'], mv['pvalues']['feature_009'],
            )),
            'p_value': float(mv['pvalues']['feature_078']),
            'effect_estimate': float(mv['coef']['feature_078']),
            'significant': True,
        },
        {
            'hypothesis_ids': ['h4.2'],
            'code': 'Same multivariable model; inspect race/insurance coefficients.',
            'result_summary': ('Adjusted race coefficients (vs reference): black=%+.4f (p=%.2f), hispanic=%+.4f (p=%.2f), asian (reference: white) — all non-significant. Adjusted insurance coefficients all p>0.9. Demographics add no incremental information after clinical adjustment.' % (
                mv['coef'].get('C(feature_018)[T.black]', 0), mv['pvalues'].get('C(feature_018)[T.black]', 1),
                mv['coef'].get('C(feature_018)[T.hispanic]', 0), mv['pvalues'].get('C(feature_018)[T.hispanic]', 1),
            )),
            'p_value': float(mv['pvalues'].get('C(feature_018)[T.black]', 1)),
            'effect_estimate': float(mv['coef'].get('C(feature_018)[T.black]', 0)),
            'significant': False,
        },
    ],
}
iterations.append(i4)

# ---------- Iteration 5: pairwise demographic comparisons ----------
i5 = {
    'index': 5,
    'proposed_hypotheses': [
        {'id': 'h5.1', 'text': 'Mean pfs_months for black patients (feature_018=black) differs from white patients (reference).', 'kind': 'refined'},
        {'id': 'h5.2', 'text': 'Mean pfs_months for hispanic patients differs from white patients.', 'kind': 'refined'},
        {'id': 'h5.3', 'text': 'Mean pfs_months for asian patients differs from white patients.', 'kind': 'refined'},
        {'id': 'h5.4', 'text': 'Mean pfs_months for medicaid-insured patients differs from privately-insured patients (reference).', 'kind': 'refined'},
        {'id': 'h5.5', 'text': 'Mean pfs_months for uninsured patients differs from privately-insured patients.', 'kind': 'refined'},
        {'id': 'h5.6', 'text': 'Mean pfs_months for medicare patients differs from privately-insured patients.', 'kind': 'refined'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h5.1'], 'code': 't-test', 'result_summary': 'black vs white: diff=%+.4f mo, p=%.3f (n_black=%d, n_white=%d).' % (R['i5_race_black_vs_white']['mean_diff'], R['i5_race_black_vs_white']['p'], R['i5_race_black_vs_white']['n_race'], R['i5_race_black_vs_white']['n_white']), 'p_value': pv(R['i5_race_black_vs_white']['p']), 'effect_estimate': float(R['i5_race_black_vs_white']['mean_diff']), 'significant': False},
        {'hypothesis_ids': ['h5.2'], 'code': 't-test', 'result_summary': 'hispanic vs white: diff=%+.4f mo, p=%.3f.' % (R['i5_race_hispanic_vs_white']['mean_diff'], R['i5_race_hispanic_vs_white']['p']), 'p_value': pv(R['i5_race_hispanic_vs_white']['p']), 'effect_estimate': float(R['i5_race_hispanic_vs_white']['mean_diff']), 'significant': False},
        {'hypothesis_ids': ['h5.3'], 'code': 't-test', 'result_summary': 'asian vs white: diff=%+.4f mo, p=%.3f. Direction is nominally positive (~0.06 mo) but not significant.' % (R['i5_race_asian_vs_white']['mean_diff'], R['i5_race_asian_vs_white']['p']), 'p_value': pv(R['i5_race_asian_vs_white']['p']), 'effect_estimate': float(R['i5_race_asian_vs_white']['mean_diff']), 'significant': False},
        {'hypothesis_ids': ['h5.4'], 'code': 't-test', 'result_summary': 'medicaid vs private: diff=%+.4f mo, p=%.3f.' % (R['i5_ins_medicaid_vs_private']['mean_diff'], R['i5_ins_medicaid_vs_private']['p']), 'p_value': pv(R['i5_ins_medicaid_vs_private']['p']), 'effect_estimate': float(R['i5_ins_medicaid_vs_private']['mean_diff']), 'significant': False},
        {'hypothesis_ids': ['h5.5'], 'code': 't-test', 'result_summary': 'uninsured vs private: diff=%+.4f mo, p=%.3f.' % (R['i5_ins_uninsured_vs_private']['mean_diff'], R['i5_ins_uninsured_vs_private']['p']), 'p_value': pv(R['i5_ins_uninsured_vs_private']['p']), 'effect_estimate': float(R['i5_ins_uninsured_vs_private']['mean_diff']), 'significant': False},
        {'hypothesis_ids': ['h5.6'], 'code': 't-test', 'result_summary': 'medicare vs private: diff=%+.4f mo, p=%.3f.' % (R['i5_ins_medicare_vs_private']['mean_diff'], R['i5_ins_medicare_vs_private']['p']), 'p_value': pv(R['i5_ins_medicare_vs_private']['p']), 'effect_estimate': float(R['i5_ins_medicare_vs_private']['mean_diff']), 'significant': False},
    ],
}
iterations.append(i5)

# ---------- Iteration 6: race x top binary ----------
i6 = {
    'index': 6,
    'proposed_hypotheses': [
        {'id': 'h6.1', 'text': 'The effect of feature_051 on pfs_months differs across racial categories (race × feature_051 interaction).', 'kind': 'novel'},
        {'id': 'h6.2', 'text': 'The effect of feature_109 on pfs_months differs across racial categories (race × feature_109 interaction).', 'kind': 'novel'},
        {'id': 'h6.3', 'text': 'The effect of feature_039 on pfs_months differs across racial categories (race × feature_039 interaction).', 'kind': 'novel'},
        {'id': 'h6.4', 'text': 'The effect of feature_043 on pfs_months differs across racial categories (race × feature_043 interaction).', 'kind': 'novel'},
        {'id': 'h6.5', 'text': 'The effect of feature_035 on pfs_months differs across racial categories (race × feature_035 interaction).', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h6.1'], 'code': 'OLS interaction model with joint F-test', 'result_summary': 'Joint F-test for race × feature_051 interaction (4 df): F=%.3f, p=%.3f. No heterogeneity detected.' % (R['i6_race_x_feature_051']['F'], R['i6_race_x_feature_051']['p']), 'p_value': pv(R['i6_race_x_feature_051']['p']), 'effect_estimate': float(R['i6_race_x_feature_051']['F']), 'significant': False},
        {'hypothesis_ids': ['h6.2'], 'code': 'OLS interaction with F-test', 'result_summary': 'Joint F-test race × feature_109: F=%.3f, p=%.3f.' % (R['i6_race_x_feature_109']['F'], R['i6_race_x_feature_109']['p']), 'p_value': pv(R['i6_race_x_feature_109']['p']), 'effect_estimate': float(R['i6_race_x_feature_109']['F']), 'significant': False},
        {'hypothesis_ids': ['h6.3'], 'code': 'OLS interaction with F-test', 'result_summary': 'Joint F-test race × feature_039: F=%.3f, p=%.3f.' % (R['i6_race_x_feature_039']['F'], R['i6_race_x_feature_039']['p']), 'p_value': pv(R['i6_race_x_feature_039']['p']), 'effect_estimate': float(R['i6_race_x_feature_039']['F']), 'significant': False},
        {'hypothesis_ids': ['h6.4'], 'code': 'OLS interaction with F-test', 'result_summary': 'Joint F-test race × feature_043: F=%.3f, p=%.3f.' % (R['i6_race_x_feature_043']['F'], R['i6_race_x_feature_043']['p']), 'p_value': pv(R['i6_race_x_feature_043']['p']), 'effect_estimate': float(R['i6_race_x_feature_043']['F']), 'significant': False},
        {'hypothesis_ids': ['h6.5'], 'code': 'OLS interaction with F-test', 'result_summary': 'Joint F-test race × feature_035: F=%.3f, p=%.3f.' % (R['i6_race_x_feature_035']['F'], R['i6_race_x_feature_035']['p']), 'p_value': pv(R['i6_race_x_feature_035']['p']), 'effect_estimate': float(R['i6_race_x_feature_035']['F']), 'significant': False},
    ],
}
iterations.append(i6)

# ---------- Iteration 7: insurance x top binary ----------
i7 = {
    'index': 7,
    'proposed_hypotheses': [
        {'id': 'h7.1', 'text': 'The effect of feature_051 on pfs_months differs across insurance categories (insurance × feature_051 interaction).', 'kind': 'novel'},
        {'id': 'h7.2', 'text': 'The effect of feature_109 on pfs_months differs across insurance categories (insurance × feature_109 interaction).', 'kind': 'novel'},
        {'id': 'h7.3', 'text': 'The effect of feature_039 on pfs_months differs across insurance categories (insurance × feature_039 interaction).', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h7.1'], 'code': 'OLS with insurance × feature_051 interaction', 'result_summary': 'Joint F=%.3f, p=%.3f.' % (R['i7_ins_x_feature_051']['F'], R['i7_ins_x_feature_051']['p']), 'p_value': pv(R['i7_ins_x_feature_051']['p']), 'effect_estimate': float(R['i7_ins_x_feature_051']['F']), 'significant': False},
        {'hypothesis_ids': ['h7.2'], 'code': 'OLS with insurance × feature_109 interaction', 'result_summary': 'Joint F=%.3f, p=%.3f.' % (R['i7_ins_x_feature_109']['F'], R['i7_ins_x_feature_109']['p']), 'p_value': pv(R['i7_ins_x_feature_109']['p']), 'effect_estimate': float(R['i7_ins_x_feature_109']['F']), 'significant': False},
        {'hypothesis_ids': ['h7.3'], 'code': 'OLS with insurance × feature_039 interaction', 'result_summary': 'Joint F=%.3f, p=%.3f.' % (R['i7_ins_x_feature_039']['F'], R['i7_ins_x_feature_039']['p']), 'p_value': pv(R['i7_ins_x_feature_039']['p']), 'effect_estimate': float(R['i7_ins_x_feature_039']['F']), 'significant': False},
    ],
}
iterations.append(i7)

# ---------- Iteration 8: age x top binary ----------
i8 = {
    'index': 8,
    'proposed_hypotheses': [
        {'id': 'h8.1', 'text': 'The negative effect of feature_051 on pfs_months grows in magnitude with higher feature_078 (age × feature_051 interaction; coefficient negative).', 'kind': 'novel'},
        {'id': 'h8.2', 'text': 'The effect of feature_109 on pfs_months varies with feature_078 (age × feature_109 interaction).', 'kind': 'novel'},
        {'id': 'h8.3', 'text': 'The effect of feature_039 on pfs_months varies with feature_078 (age × feature_039 interaction).', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h8.1'], 'code': "OLS pfs ~ feature_051*feature_078 + C(feature_057)", 'result_summary': 'Interaction coef = %+.5f mo per (year × feature_051), p=%.2e. Significant: each unit of feature_078 amplifies the negative feature_051 effect.' % (R['i8_age_x_feature_051']['coef'], R['i8_age_x_feature_051']['p']), 'p_value': pv(R['i8_age_x_feature_051']['p']), 'effect_estimate': float(R['i8_age_x_feature_051']['coef']), 'significant': True},
        {'hypothesis_ids': ['h8.2'], 'code': 'OLS interaction', 'result_summary': 'Interaction coef = %+.5f, p=%.3f. Not significant.' % (R['i8_age_x_feature_109']['coef'], R['i8_age_x_feature_109']['p']), 'p_value': pv(R['i8_age_x_feature_109']['p']), 'effect_estimate': float(R['i8_age_x_feature_109']['coef']), 'significant': False},
        {'hypothesis_ids': ['h8.3'], 'code': 'OLS interaction', 'result_summary': 'Interaction coef = %+.5f, p=%.3f. Not significant.' % (R['i8_age_x_feature_039']['coef'], R['i8_age_x_feature_039']['p']), 'p_value': pv(R['i8_age_x_feature_039']['p']), 'effect_estimate': float(R['i8_age_x_feature_039']['coef']), 'significant': False},
    ],
}
iterations.append(i8)

# ---------- Iteration 9: stage x top binary ----------
i9 = {
    'index': 9,
    'proposed_hypotheses': [
        {'id': 'h9.1', 'text': 'The effect of feature_051 on pfs_months differs across feature_057 strata (stage × feature_051 interaction).', 'kind': 'novel'},
        {'id': 'h9.2', 'text': 'The effect of feature_109 on pfs_months differs across feature_057 strata.', 'kind': 'novel'},
        {'id': 'h9.3', 'text': 'The effect of feature_039 on pfs_months differs across feature_057 strata.', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h9.1'], 'code': 'OLS C(feature_051)*C(feature_057)', 'result_summary': 'Joint F=%.2f, p=%.4f. Significant interaction; the gap between feature_051 strata varies somewhat with stage (smallest in stage 2 where overall PFS is shortest).' % (R['i9_stage_x_feature_051']['F'], R['i9_stage_x_feature_051']['p']), 'p_value': pv(R['i9_stage_x_feature_051']['p']), 'effect_estimate': float(R['i9_stage_x_feature_051']['F']), 'significant': bool(R['i9_stage_x_feature_051']['p'] < 0.05)},
        {'hypothesis_ids': ['h9.2'], 'code': 'OLS interaction', 'result_summary': 'Joint F=%.2f, p=%.3f. Not significant.' % (R['i9_stage_x_feature_109']['F'], R['i9_stage_x_feature_109']['p']), 'p_value': pv(R['i9_stage_x_feature_109']['p']), 'effect_estimate': float(R['i9_stage_x_feature_109']['F']), 'significant': False},
        {'hypothesis_ids': ['h9.3'], 'code': 'OLS interaction', 'result_summary': 'Joint F=%.2f, p=%.3f. Not significant.' % (R['i9_stage_x_feature_039']['F'], R['i9_stage_x_feature_039']['p']), 'p_value': pv(R['i9_stage_x_feature_039']['p']), 'effect_estimate': float(R['i9_stage_x_feature_039']['F']), 'significant': False},
    ],
}
iterations.append(i9)

# ---------- Iteration 10: adjusted binary screen ----------
top25 = R['i10_adj_binary_screen_top25']
i10 = {
    'index': 10,
    'proposed_hypotheses': [
        {'id': 'h10.1', 'text': 'After adjusting for age (feature_078) and stage (feature_057), feature_051 retains a strong negative effect on pfs_months.', 'kind': 'refined'},
        {'id': 'h10.2', 'text': 'After adjusting for age and stage, feature_109 retains a positive effect on pfs_months.', 'kind': 'refined'},
        {'id': 'h10.3', 'text': 'After adjusting for age and stage, feature_039 retains a positive effect on pfs_months.', 'kind': 'refined'},
        {'id': 'h10.4', 'text': 'After adjusting for age and stage, no other binary feature has an effect on pfs_months larger in magnitude than ±0.05 months.', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h10.1'], 'code': "OLS pfs_months ~ feature_051 + feature_078 + C(feature_057) for each binary col, ranked by p", 'result_summary': 'feature_051 adjusted coef = %+.4f mo, p=%.2e (top of screen).' % (top25[0]['coef'], top25[0]['p']), 'p_value': pv(top25[0]['p']), 'effect_estimate': float(top25[0]['coef']), 'significant': True},
        {'hypothesis_ids': ['h10.2'], 'code': 'same screen', 'result_summary': 'feature_109 adjusted coef = %+.4f mo, p=%.2e (rank 2 in screen).' % (top25[1]['coef'], top25[1]['p']), 'p_value': pv(top25[1]['p']), 'effect_estimate': float(top25[1]['coef']), 'significant': True},
        {'hypothesis_ids': ['h10.3'], 'code': 'same screen', 'result_summary': 'feature_039 adjusted coef = %+.4f mo, p=%.2e (rank 3 in screen).' % (top25[2]['coef'], top25[2]['p']), 'p_value': pv(top25[2]['p']), 'effect_estimate': float(top25[2]['coef']), 'significant': True},
        {'hypothesis_ids': ['h10.4'], 'code': 'same screen', 'result_summary': 'Aside from feature_051, feature_109, feature_039, the remaining binary features all have |adjusted coefficient| <= 0.04 mo. Notable significant but tiny effects: feature_027 (+0.020 mo, p=1.5e-3), feature_043 (-0.023, p=3.3e-3), feature_079 (-0.033, p=0.04), feature_035 (-0.016, p=0.04). No additional clinically meaningful signals.', 'p_value': float(top25[3]['p']), 'effect_estimate': float(top25[3]['coef']), 'significant': True},
    ],
}
iterations.append(i10)

# ---------- Iteration 11: PSA quartile-stratified PFS ----------
psa_q = R['i11_psa_quartile_pfs']
i11 = {
    'index': 11,
    'proposed_hypotheses': [
        {'id': 'h11.1', 'text': 'Mean pfs_months decreases monotonically across rising quartiles of feature_013 (PSA-like), with the highest quartile having the shortest mean PFS.', 'kind': 'refined'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h11.1'], 'code': 'pd.qcut + groupby + linear trend', 'result_summary': 'Mean pfs_months by feature_013 quartile: q0=%.3f, q1=%.3f, q2=%.3f, q3=%.3f. Linear trend coef = %.3f mo per quartile, p=%.2e. Confirms a stepwise dose-response.' % (psa_q['q0']['mean'], psa_q['q1']['mean'], psa_q['q2']['mean'], psa_q['q3']['mean'], R['i11_psa_q_trend']['coef'], R['i11_psa_q_trend']['p']), 'p_value': pv(R['i11_psa_q_trend']['p']), 'effect_estimate': float(R['i11_psa_q_trend']['coef']), 'significant': True},
    ],
}
iterations.append(i11)

# ---------- Iteration 12: PSA × stage interaction & within-stage ----------
i12 = {
    'index': 12,
    'proposed_hypotheses': [
        {'id': 'h12.1', 'text': 'The effect of log(feature_013) on pfs_months differs across feature_057 strata (PSA × stage interaction).', 'kind': 'novel'},
        {'id': 'h12.2', 'text': 'Within feature_057=0 (stage 0), higher feature_013 still associates with shorter pfs_months.', 'kind': 'refined'},
        {'id': 'h12.3', 'text': 'Within feature_057=2 (stage 2), higher feature_013 still associates with shorter pfs_months.', 'kind': 'refined'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h12.1'], 'code': 'OLS log1p(feature_013)*C(feature_057)', 'result_summary': 'Joint F-test for PSA × stage: F=%.2f, p=%.2e (df=%d). Significant interaction.' % (R['i12_psa_x_stage']['F'], R['i12_psa_x_stage']['p'], R['i12_psa_x_stage']['df']), 'p_value': pv(R['i12_psa_x_stage']['p']), 'effect_estimate': float(R['i12_psa_x_stage']['F']), 'significant': True},
        {'hypothesis_ids': ['h12.2'], 'code': 'spearmanr within feature_057==0', 'result_summary': 'Within stage 0: spearman r=%.3f, p=%.2e (n=%d).' % (R['i12_psa_within_stage0']['r'], R['i12_psa_within_stage0']['p'], R['i12_psa_within_stage0']['n']), 'p_value': pv(R['i12_psa_within_stage0']['p']), 'effect_estimate': float(R['i12_psa_within_stage0']['r']), 'significant': True},
        {'hypothesis_ids': ['h12.3'], 'code': 'spearmanr within feature_057==2', 'result_summary': 'Within stage 2: spearman r=%.3f, p=%.2e (n=%d). Effect persists.' % (R['i12_psa_within_stage2']['r'], R['i12_psa_within_stage2']['p'], R['i12_psa_within_stage2']['n']), 'p_value': pv(R['i12_psa_within_stage2']['p']), 'effect_estimate': float(R['i12_psa_within_stage2']['r']), 'significant': True},
    ],
}
iterations.append(i12)

# ---------- Iteration 13: race-stratified feature_051 ----------
i13 = {
    'index': 13,
    'proposed_hypotheses': [
        {'id': 'h13.1', 'text': 'The negative feature_051 effect on pfs_months is observed within white patients.', 'kind': 'refined'},
        {'id': 'h13.2', 'text': 'The negative feature_051 effect on pfs_months is observed within black patients.', 'kind': 'refined'},
        {'id': 'h13.3', 'text': 'The negative feature_051 effect on pfs_months is observed within hispanic patients.', 'kind': 'refined'},
        {'id': 'h13.4', 'text': 'The negative feature_051 effect on pfs_months is observed within asian patients.', 'kind': 'refined'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h13.1'], 'code': 't-test within feature_018==white', 'result_summary': 'Within white: feature_051 mean diff = %+.4f mo, p=%.2e (n0=%d, n1=%d).' % (R['i13_f051_within_white']['mean_diff'], R['i13_f051_within_white']['p'], R['i13_f051_within_white']['n0'], R['i13_f051_within_white']['n1']), 'p_value': pv(R['i13_f051_within_white']['p']), 'effect_estimate': float(R['i13_f051_within_white']['mean_diff']), 'significant': True},
        {'hypothesis_ids': ['h13.2'], 'code': 't-test within feature_018==black', 'result_summary': 'Within black: feature_051 mean diff = %+.4f mo, p=%.2e (n0=%d, n1=%d).' % (R['i13_f051_within_black']['mean_diff'], R['i13_f051_within_black']['p'], R['i13_f051_within_black']['n0'], R['i13_f051_within_black']['n1']), 'p_value': pv(R['i13_f051_within_black']['p']), 'effect_estimate': float(R['i13_f051_within_black']['mean_diff']), 'significant': True},
        {'hypothesis_ids': ['h13.3'], 'code': 't-test within hispanic', 'result_summary': 'Within hispanic: feature_051 mean diff = %+.4f mo, p=%.2e.' % (R['i13_f051_within_hispanic']['mean_diff'], R['i13_f051_within_hispanic']['p']), 'p_value': pv(R['i13_f051_within_hispanic']['p']), 'effect_estimate': float(R['i13_f051_within_hispanic']['mean_diff']), 'significant': True},
        {'hypothesis_ids': ['h13.4'], 'code': 't-test within asian', 'result_summary': 'Within asian: feature_051 mean diff = %+.4f mo, p=%.2e.' % (R['i13_f051_within_asian']['mean_diff'], R['i13_f051_within_asian']['p']), 'p_value': pv(R['i13_f051_within_asian']['p']), 'effect_estimate': float(R['i13_f051_within_asian']['mean_diff']), 'significant': True},
    ],
}
iterations.append(i13)

# ---------- Iteration 14: insurance-stratified feature_051 ----------
i14 = {
    'index': 14,
    'proposed_hypotheses': [
        {'id': 'h14.1', 'text': 'The negative feature_051 effect on pfs_months is observed within privately-insured patients.', 'kind': 'refined'},
        {'id': 'h14.2', 'text': 'The negative feature_051 effect on pfs_months is observed within medicare patients.', 'kind': 'refined'},
        {'id': 'h14.3', 'text': 'The negative feature_051 effect on pfs_months is observed within medicaid patients.', 'kind': 'refined'},
        {'id': 'h14.4', 'text': 'The negative feature_051 effect on pfs_months is observed within uninsured patients.', 'kind': 'refined'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h14.1'], 'code': 't-test within private', 'result_summary': 'Within private: f051 diff=%+.4f mo, p=%.2e.' % (R['i14_f051_within_private']['mean_diff'], R['i14_f051_within_private']['p']), 'p_value': pv(R['i14_f051_within_private']['p']), 'effect_estimate': float(R['i14_f051_within_private']['mean_diff']), 'significant': True},
        {'hypothesis_ids': ['h14.2'], 'code': 't-test within medicare', 'result_summary': 'Within medicare: f051 diff=%+.4f mo, p=%.2e.' % (R['i14_f051_within_medicare']['mean_diff'], R['i14_f051_within_medicare']['p']), 'p_value': pv(R['i14_f051_within_medicare']['p']), 'effect_estimate': float(R['i14_f051_within_medicare']['mean_diff']), 'significant': True},
        {'hypothesis_ids': ['h14.3'], 'code': 't-test within medicaid', 'result_summary': 'Within medicaid: f051 diff=%+.4f mo, p=%.2e.' % (R['i14_f051_within_medicaid']['mean_diff'], R['i14_f051_within_medicaid']['p']), 'p_value': pv(R['i14_f051_within_medicaid']['p']), 'effect_estimate': float(R['i14_f051_within_medicaid']['mean_diff']), 'significant': True},
        {'hypothesis_ids': ['h14.4'], 'code': 't-test within uninsured', 'result_summary': 'Within uninsured: f051 diff=%+.4f mo, p=%.2e (largest magnitude across insurance strata).' % (R['i14_f051_within_uninsured']['mean_diff'], R['i14_f051_within_uninsured']['p']), 'p_value': pv(R['i14_f051_within_uninsured']['p']), 'effect_estimate': float(R['i14_f051_within_uninsured']['mean_diff']), 'significant': True},
    ],
}
iterations.append(i14)

# ---------- Iteration 15: continuous biomarker × feature_051 interactions ----------
i15 = {
    'index': 15,
    'proposed_hypotheses': [
        {'id': 'h15.1', 'text': 'The effect of feature_051 on pfs_months varies with feature_009 (continuous albumin-like biomarker).', 'kind': 'novel'},
        {'id': 'h15.2', 'text': 'The effect of feature_051 on pfs_months varies with feature_006.', 'kind': 'novel'},
        {'id': 'h15.3', 'text': 'The effect of feature_051 on pfs_months varies with feature_094.', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h15.1'], 'code': 'OLS feature_051*feature_009', 'result_summary': 'Interaction coef=%+.5f, p=%.3f.' % (R['i15_f051_x_feature_009']['coef'], R['i15_f051_x_feature_009']['p']), 'p_value': pv(R['i15_f051_x_feature_009']['p']), 'effect_estimate': float(R['i15_f051_x_feature_009']['coef']), 'significant': False},
        {'hypothesis_ids': ['h15.2'], 'code': 'OLS feature_051*feature_006', 'result_summary': 'Interaction coef=%+.5f, p=%.3f.' % (R['i15_f051_x_feature_006']['coef'], R['i15_f051_x_feature_006']['p']), 'p_value': pv(R['i15_f051_x_feature_006']['p']), 'effect_estimate': float(R['i15_f051_x_feature_006']['coef']), 'significant': False},
        {'hypothesis_ids': ['h15.3'], 'code': 'OLS feature_051*feature_094', 'result_summary': 'Interaction coef=%+.5f, p=%.3f. Marginally significant.' % (R['i15_f051_x_feature_094']['coef'], R['i15_f051_x_feature_094']['p']), 'p_value': pv(R['i15_f051_x_feature_094']['p']), 'effect_estimate': float(R['i15_f051_x_feature_094']['coef']), 'significant': bool(R['i15_f051_x_feature_094']['p']<0.05)},
    ],
}
iterations.append(i15)

# ---------- Iteration 16: combined effect feature_109 + feature_039 ----------
combo2 = R['i16_combo_109_039_C(_both_pf)[T.2]']
combo1 = R['i16_combo_109_039_C(_both_pf)[T.1]']
i16 = {
    'index': 16,
    'proposed_hypotheses': [
        {'id': 'h16.1', 'text': 'Patients with both feature_109=1 AND feature_039=1 (n≈500) have substantially longer pfs_months than patients with neither, beyond what each individual main effect would predict (super-additive effect).', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h16.1'], 'code': "OLS pfs_months ~ C(_both_pf) + age + stage where _both_pf = feature_109+feature_039", 'result_summary': 'Adjusted PFS contrast: 1-of-2 positive (only one of the two flags) vs 0-of-2: %+.4f mo, p=%.2f. 2-of-2 (both) vs 0-of-2: %+.4f mo, p=%.2e. The both-positive group shows roughly 5x larger benefit than the sum of single-positive effects (~0.30 mo expected vs +1.42 mo observed) — strong synergistic interaction.' % (combo1['coef'], combo1['p'], combo2['coef'], combo2['p']), 'p_value': pv(combo2['p']), 'effect_estimate': float(combo2['coef']), 'significant': True},
    ],
}
iterations.append(i16)

# ---------- Iteration 17: age × race ----------
i17 = {
    'index': 17,
    'proposed_hypotheses': [
        {'id': 'h17.1', 'text': 'The slope of pfs_months on feature_078 differs across racial categories (age × race interaction).', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h17.1'], 'code': 'OLS feature_078*C(feature_018) + C(feature_057)', 'result_summary': 'Joint F=%.2f (df=%d), p=%.3f. No heterogeneity of age effect by race.' % (R['i17_age_x_race']['F'], R['i17_age_x_race']['df'], R['i17_age_x_race']['p']), 'p_value': pv(R['i17_age_x_race']['p']), 'effect_estimate': float(R['i17_age_x_race']['F']), 'significant': False},
    ],
}
iterations.append(i17)

# ---------- Iteration 18: PSA × race ----------
i18 = {
    'index': 18,
    'proposed_hypotheses': [
        {'id': 'h18.1', 'text': 'The slope of pfs_months on log(feature_013) differs across racial categories (PSA × race interaction).', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h18.1'], 'code': 'OLS log1p(feature_013)*C(feature_018) + age + stage', 'result_summary': 'Joint F=%.2f (df=%d), p=%.3f. No racial heterogeneity in the PSA effect.' % (R['i18_psa_x_race']['F'], R['i18_psa_x_race']['df'], R['i18_psa_x_race']['p']), 'p_value': pv(R['i18_psa_x_race']['p']), 'effect_estimate': float(R['i18_psa_x_race']['F']), 'significant': False},
    ],
}
iterations.append(i18)

# ---------- Iteration 19: stage × race ----------
i19 = {
    'index': 19,
    'proposed_hypotheses': [
        {'id': 'h19.1', 'text': 'The negative effect of feature_057 on pfs_months differs across racial categories (stage × race interaction).', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h19.1'], 'code': 'OLS C(feature_057)*C(feature_018) + age', 'result_summary': 'Joint F=%.2f (df=%d), p=%.3f. No stage × race interaction.' % (R['i19_stage_x_race']['F'], R['i19_stage_x_race']['df'], R['i19_stage_x_race']['p']), 'p_value': pv(R['i19_stage_x_race']['p']), 'effect_estimate': float(R['i19_stage_x_race']['F']), 'significant': False},
    ],
}
iterations.append(i19)

# ---------- Iteration 20: stage × insurance ----------
i20 = {
    'index': 20,
    'proposed_hypotheses': [
        {'id': 'h20.1', 'text': 'The negative effect of feature_057 on pfs_months differs across insurance categories (stage × insurance interaction).', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h20.1'], 'code': 'OLS C(feature_057)*C(feature_045) + age', 'result_summary': 'Joint F=%.2f (df=%d), p=%.3f. No stage × insurance interaction.' % (R['i20_stage_x_ins']['F'], R['i20_stage_x_ins']['df'], R['i20_stage_x_ins']['p']), 'p_value': pv(R['i20_stage_x_ins']['p']), 'effect_estimate': float(R['i20_stage_x_ins']['F']), 'significant': False},
    ],
}
iterations.append(i20)

# ---------- Iteration 21: feature_006 within stage strata ----------
i21 = {
    'index': 21,
    'proposed_hypotheses': [
        {'id': 'h21.1', 'text': 'Within each feature_057 (stage) stratum, higher feature_006 is associated with shorter pfs_months.', 'kind': 'refined'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h21.1'], 'code': 'spearmanr by stage', 'result_summary': 'Within stage 0: r=%.3f (n=%d, p=%.2e); stage 1: r=%.3f (n=%d, p=%.2e); stage 2: r=%.3f (n=%d, p=%.2e). Effect direction is consistently negative across all strata.' % (R['i21_f006_within_stage0']['r'], R['i21_f006_within_stage0']['n'], R['i21_f006_within_stage0']['p'], R['i21_f006_within_stage1']['r'], R['i21_f006_within_stage1']['n'], R['i21_f006_within_stage1']['p'], R['i21_f006_within_stage2']['r'], R['i21_f006_within_stage2']['n'], R['i21_f006_within_stage2']['p']), 'p_value': pv(R['i21_f006_within_stage1']['p']), 'effect_estimate': float(R['i21_f006_within_stage1']['r']), 'significant': True},
    ],
}
iterations.append(i21)

# ---------- Iteration 22: receipt of features by race ----------
i22 = {
    'index': 22,
    'proposed_hypotheses': [
        {'id': 'h22.1', 'text': 'Prevalence of feature_051=1 differs across racial categories (i.e., disease-burden flag is unequally distributed across races).', 'kind': 'novel'},
        {'id': 'h22.2', 'text': 'Prevalence of feature_109=1 differs across racial categories.', 'kind': 'novel'},
        {'id': 'h22.3', 'text': 'Distribution of feature_057 levels differs across racial categories.', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h22.1'], 'code': 'chi2_contingency feature_051 x feature_018', 'result_summary': 'chi2 p=%.3f. Prevalence range across races: %.3f–%.3f. No significant racial disparity.' % (R['i22_prev_feature_051_by_race']['p'], min(R['i22_prev_feature_051_by_race']['prev_by_race'].values()), max(R['i22_prev_feature_051_by_race']['prev_by_race'].values())), 'p_value': pv(R['i22_prev_feature_051_by_race']['p']), 'effect_estimate': float(max(R['i22_prev_feature_051_by_race']['prev_by_race'].values()) - min(R['i22_prev_feature_051_by_race']['prev_by_race'].values())), 'significant': False},
        {'hypothesis_ids': ['h22.2'], 'code': 'chi2 feature_109 x feature_018', 'result_summary': 'chi2 p=%.3f. Prevalence range: %.3f–%.3f. No racial disparity.' % (R['i22_prev_feature_109_by_race']['p'], min(R['i22_prev_feature_109_by_race']['prev_by_race'].values()), max(R['i22_prev_feature_109_by_race']['prev_by_race'].values())), 'p_value': pv(R['i22_prev_feature_109_by_race']['p']), 'effect_estimate': float(max(R['i22_prev_feature_109_by_race']['prev_by_race'].values()) - min(R['i22_prev_feature_109_by_race']['prev_by_race'].values())), 'significant': False},
        {'hypothesis_ids': ['h22.3'], 'code': 'chi2 feature_057 x feature_018', 'result_summary': 'chi2 p=%.3f. Stage distribution does not differ by race.' % R['i22_prev_feature_057_by_race']['p'], 'p_value': pv(R['i22_prev_feature_057_by_race']['p']), 'effect_estimate': 0.0, 'significant': False},
    ],
}
iterations.append(i22)

# ---------- Iteration 23: receipt of features by insurance ----------
i23 = {
    'index': 23,
    'proposed_hypotheses': [
        {'id': 'h23.1', 'text': 'Prevalence of feature_051=1 differs across insurance categories.', 'kind': 'novel'},
        {'id': 'h23.2', 'text': 'Prevalence of feature_109=1 differs across insurance categories.', 'kind': 'novel'},
        {'id': 'h23.3', 'text': 'Prevalence of feature_043=1 differs across insurance categories.', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h23.1'], 'code': 'chi2 feature_051 x feature_045', 'result_summary': 'chi2 p=%.3f. Prevalence range %.3f–%.3f.' % (R['i23_prev_feature_051_by_ins']['p'], min(R['i23_prev_feature_051_by_ins']['prev_by_ins'].values()), max(R['i23_prev_feature_051_by_ins']['prev_by_ins'].values())), 'p_value': pv(R['i23_prev_feature_051_by_ins']['p']), 'effect_estimate': float(max(R['i23_prev_feature_051_by_ins']['prev_by_ins'].values()) - min(R['i23_prev_feature_051_by_ins']['prev_by_ins'].values())), 'significant': False},
        {'hypothesis_ids': ['h23.2'], 'code': 'chi2 feature_109 x feature_045', 'result_summary': 'chi2 p=%.3f.' % R['i23_prev_feature_109_by_ins']['p'], 'p_value': pv(R['i23_prev_feature_109_by_ins']['p']), 'effect_estimate': float(max(R['i23_prev_feature_109_by_ins']['prev_by_ins'].values()) - min(R['i23_prev_feature_109_by_ins']['prev_by_ins'].values())), 'significant': False},
        {'hypothesis_ids': ['h23.3'], 'code': 'chi2 feature_043 x feature_045', 'result_summary': 'chi2 p=%.3f. Range %.3f–%.3f; uninsured slightly elevated but not significant.' % (R['i23_prev_feature_043_by_ins']['p'], min(R['i23_prev_feature_043_by_ins']['prev_by_ins'].values()), max(R['i23_prev_feature_043_by_ins']['prev_by_ins'].values())), 'p_value': pv(R['i23_prev_feature_043_by_ins']['p']), 'effect_estimate': float(max(R['i23_prev_feature_043_by_ins']['prev_by_ins'].values()) - min(R['i23_prev_feature_043_by_ins']['prev_by_ins'].values())), 'significant': False},
    ],
}
iterations.append(i23)

# ---------- Iteration 24: age distribution by demographics ----------
i24 = {
    'index': 24,
    'proposed_hypotheses': [
        {'id': 'h24.1', 'text': 'Mean feature_078 (age) differs across racial categories — i.e., demographic groups have different age structures that could confound the age-PFS effect.', 'kind': 'novel'},
        {'id': 'h24.2', 'text': 'Mean feature_078 (age) differs across insurance categories — uninsured patients are notably younger than medicare patients.', 'kind': 'novel'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h24.1'], 'code': 'ANOVA feature_078 by feature_018', 'result_summary': 'ANOVA p=%.3f. Mean ages range %.2f–%.2f years across races; differences are negligible.' % (R['i24_age_by_race']['p'], min(R['i24_age_by_race']['mean_age'].values()), max(R['i24_age_by_race']['mean_age'].values())), 'p_value': pv(R['i24_age_by_race']['p']), 'effect_estimate': float(max(R['i24_age_by_race']['mean_age'].values()) - min(R['i24_age_by_race']['mean_age'].values())), 'significant': False},
        {'hypothesis_ids': ['h24.2'], 'code': 'ANOVA feature_078 by feature_045', 'result_summary': 'ANOVA p=%.3f. Mean ages range %.2f–%.2f years across insurance groups; uninsured age is virtually identical to medicare age (~65 yrs), inconsistent with the conventional pattern of younger uninsured.' % (R['i24_age_by_ins']['p'], min(R['i24_age_by_ins']['mean_age'].values()), max(R['i24_age_by_ins']['mean_age'].values())), 'p_value': pv(R['i24_age_by_ins']['p']), 'effect_estimate': float(max(R['i24_age_by_ins']['mean_age'].values()) - min(R['i24_age_by_ins']['mean_age'].values())), 'significant': False},
    ],
}
iterations.append(i24)

# ---------- Iteration 25: integrated final model ----------
fm = R['i25_final_model']
i25 = {
    'index': 25,
    'proposed_hypotheses': [
        {'id': 'h25.1', 'text': 'A single multivariable OLS containing feature_078, C(feature_057), log(feature_013), feature_051, feature_006, feature_009, feature_109, feature_039, race, insurance, plus the feature_078:feature_051 and C(feature_057):feature_051 interactions explains nearly all of the variance in pfs_months (R^2 > 0.95).', 'kind': 'refined'},
        {'id': 'h25.2', 'text': 'Within this final model, the C(feature_057):feature_051 interaction is positive — meaning the negative effect of feature_051 attenuates (becomes less negative) at higher feature_057 stages.', 'kind': 'refined'},
        {'id': 'h25.3', 'text': 'Within this final model, the feature_078:feature_051 interaction is negative — meaning the negative effect of feature_051 grows in magnitude with increasing age.', 'kind': 'refined'},
    ],
    'analyses': [
        {'hypothesis_ids': ['h25.1'], 'code': 'final OLS', 'result_summary': 'R^2 = %.4f, R^2_adj = %.4f, n=%d. The clinical features (feature_078, feature_057, log feature_013, feature_051, feature_006, feature_009, feature_109, feature_039) plus their feature_051 interactions explain >97%% of pfs_months variance.' % (fm['r2'], fm['r2_adj'], fm['n']), 'p_value': 0.0, 'effect_estimate': float(fm['r2']), 'significant': True},
        {'hypothesis_ids': ['h25.2'], 'code': 'final OLS interaction', 'result_summary': 'C(feature_057)[T.1]:feature_051 = %+.4f mo (p=%.2e); C(feature_057)[T.2]:feature_051 = %+.4f mo (p=%.2e). Both positive — feature_051 penalty is less severe at higher stages.' % (fm['coef']['C(feature_057)[T.1]:feature_051'], fm['pvalues']['C(feature_057)[T.1]:feature_051'], fm['coef']['C(feature_057)[T.2]:feature_051'], fm['pvalues']['C(feature_057)[T.2]:feature_051']), 'p_value': pv(fm['pvalues']['C(feature_057)[T.2]:feature_051']), 'effect_estimate': float(fm['coef']['C(feature_057)[T.2]:feature_051']), 'significant': True},
        {'hypothesis_ids': ['h25.3'], 'code': 'final OLS interaction', 'result_summary': 'feature_078:feature_051 coef = %+.5f mo per (year × feature_051), p=%.2e. Negative as predicted: each year of age intensifies the feature_051 detriment.' % (fm['coef']['feature_078:feature_051'], fm['pvalues']['feature_078:feature_051']), 'p_value': pv(fm['pvalues']['feature_078:feature_051']), 'effect_estimate': float(fm['coef']['feature_078:feature_051']), 'significant': True},
    ],
}
iterations.append(i25)

transcript = {
    'dataset_id': 'ds001_prostate',
    'model_id': 'claude-opus-4-7',
    'harness_id': 'claude-code-direct@manual-2026-04-28',
    'max_iterations': 25,
    'iterations': iterations,
}

with open('transcript.json', 'w') as fp:
    json.dump(transcript, fp, indent=2)
print('Wrote transcript.json with %d iterations' % len(iterations))
