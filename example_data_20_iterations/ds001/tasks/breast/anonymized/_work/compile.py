"""Compile final transcript.json and analysis_summary.txt."""
import json

iterations = json.load(open('_work/iters_partial.json'))

transcript = {
    'dataset_id': 'ds001_breast',
    'model_id': 'claude-opus-4-7',
    'harness_id': 'claude-code-manual@2026-04-28',
    'max_iterations': 25,
    'iterations': iterations,
}

with open('transcript.json', 'w') as f:
    json.dump(transcript, f, indent=2, default=str)

# Build narrative summary
summary = """ds001_breast oncology cohort analysis — narrative summary
=========================================================
Dataset: 50,000 patients, 127 features (89 binary or low-cardinality integer, 37 continuous, 2 categorical strings — race-like feature_011 and insurance-like feature_089), and outcome pfs_months (mean 4.69, sd 2.50, range 0–16.66).

Workflow: 25 iterations of propose-test-refine. Each iteration listed in transcript.json. Below is the synthesized story.

------------------------------------------------------------
Iteration 1 — top-effect binary main effects
------------------------------------------------------------
A univariate screen surfaced the four strongest binary predictors. All four were confirmed as expected:
- feature_042 = 1 (n=17,428, 35%): +1.10 mo PFS vs. feature_042 = 0; Welch p ≈ 0.
- feature_056 = 1 (n=14,969, 30%): −1.54 mo PFS; Welch p ≈ 0.
- feature_048 = 1 (n=5,090, 10%): −1.05 mo PFS; Welch p = 1.5e-178.
- feature_111 = 1 (n=34,886, 70%): +0.57 mo PFS; Welch p = 3.7e-143.
All four were highly significant with the predicted direction.

------------------------------------------------------------
Iteration 2 — ordinal feature_063 and three more binary effects
------------------------------------------------------------
- feature_063 (ordinal 0/1/2; ~half of patients are level 1, with the remainder roughly 35% level 0 and 15% level 2) shows a strong negative monotonic relationship with PFS: OLS slope −1.18 mo per unit (p ≈ 0). Mean PFS by level: 0→5.64, 1→4.44, 2→3.29. Looks like an ECOG- or stage-like prognostic indicator.
- feature_015 = 1 (35%): −0.56 mo (p = 6e-144).
- feature_040 = 1 (18%): −0.46 mo (p = 6e-70).
- feature_039 = 1 (64%): +0.36 mo (p = 6e-59).
All four supported.

------------------------------------------------------------
Iteration 3 — top continuous predictors
------------------------------------------------------------
- feature_080 (continuous 30–90, mean 65, sd 10): the single dominant predictor. Spearman ρ = +0.726, OLS slope +0.176 mo per unit. The range and effect direction (older = longer PFS in this cohort) suggests this is an age- or staging-like feature where higher values indicate slower-progressing disease.
- feature_019 (continuous 1.5–5.5, mean 3.8): +0.50 mo per unit (p = 9e-112). Range/mean fit albumin (g/dL).
- feature_067 (continuous 0–24.6, mean 3.8): −0.075 mo per unit (p = 4e-148).
- feature_101 (continuous 1–100, mean 15.5; possibly tumor burden or count): −0.019 mo per unit (p = 3e-93).
All four supported in their predicted directions.

------------------------------------------------------------
Iteration 4 — shape of the feature_080 relationship
------------------------------------------------------------
- Decile means rose monotonically across all ten deciles of feature_080: 1.70 → 2.86 → 3.46 → 3.97 → 4.44 → 4.90 → 5.38 → 5.89 → 6.56 → 7.84 mo. No inverted-U; the effect is monotonic across the full range.
- Adding a quadratic term (feature_080^2) was statistically significant (p = 6e-25) but the marginal R² gain was tiny: linear R² = 0.492 vs. quadratic R² = 0.493. The relationship is essentially linear with a very mild upward curvature.

------------------------------------------------------------
Iteration 5 — race and insurance main effects (no disparity at the population level)
------------------------------------------------------------
- One-way ANOVA over feature_011 (race, 5 levels): F = 0.85, p = 0.49. Group means (white 4.68, asian 4.74, black 4.72, hispanic 4.71, other 4.76) span 0.08 mo — clinically negligible.
- One-way ANOVA over feature_089 (insurance, 4 levels): F = 0.31, p = 0.82. Means (medicaid 4.71, private 4.68, medicare 4.69, uninsured 4.73) span 0.05 mo.
No detectable raw difference in PFS by race or insurance.

------------------------------------------------------------
Iteration 6 — multivariable model
------------------------------------------------------------
The 12 strong predictors were entered jointly in OLS:
- All 12 individually significant (smallest p < 1e-30).
- Coefficients essentially identical to univariate values: feature_080 +0.176, feature_063 −1.17, feature_042 +1.10, feature_056 −1.55, feature_048 −0.98, feature_111 +0.53, feature_015 −0.58, feature_040 −0.39, feature_039 +0.30, feature_019 +0.43, feature_067 −0.07, feature_101 −0.018.
- Multivariable R² = 0.792.

This stability indicates the strong predictors operate through largely independent channels — there is no significant confounding among them.

------------------------------------------------------------
Iteration 7 — interactions among strong binary features (negative)
------------------------------------------------------------
None of the three pairwise interactions tested were significant:
- feature_042 × feature_056: β = +0.006, p = 0.90.
- feature_042 × feature_048: β = −0.041, p = 0.59.
- feature_056 × feature_111: β = −0.007, p = 0.89.
The strong binary predictors act additively on PFS.

------------------------------------------------------------
Iteration 8 — feature_080 × strong-binary interactions
------------------------------------------------------------
Small but statistically detectable interactions:
- feature_080 × feature_056: interaction β = −0.011 mo per (unit × indicator), p = 3e-12. Slope of feature_080 = +0.180 in feature_056 = 0 vs. +0.168 in feature_056 = 1. Essentially indistinguishable clinically.
- feature_080 × feature_042: β = −0.0032, p = 0.04. Slopes 0.177 vs. 0.174.
- feature_080 × feature_063: β = −0.006, p = 6e-8. Slopes 0.179 (level 0), 0.176 (level 1), 0.166 (level 2).
Statistically significant due to n = 50,000, but magnitudes are tiny — the feature_080 slope is virtually constant across these strata.

------------------------------------------------------------
Iteration 9 — feature_042 and feature_056 effects across race
------------------------------------------------------------
- feature_042 × race interaction: joint F-test p = 0.92. Race-stratified effect of feature_042: white +1.11, asian +1.06, black +1.06, hispanic +1.13, other +1.18 mo. Highly consistent.
- feature_056 × race interaction: joint F-test p = 0.13. Stratified: white −1.55, asian −1.34, black −1.63, hispanic −1.45, other −1.65 mo. Statistically homogeneous.
The protective benefit of feature_042 and the harm of feature_056 do not differ across racial groups.

------------------------------------------------------------
Iteration 10 — feature_042 and feature_056 effects across insurance
------------------------------------------------------------
- feature_042 × insurance interaction: joint F-test p = 0.97. Effects: medicaid +1.08, private +1.09, medicare +1.11, uninsured +1.21 mo.
- feature_056 × insurance: p = 0.84. Effects: medicaid −1.50, private −1.56, medicare −1.53, uninsured −1.74 mo.
No insurance-based heterogeneity in either effect.

------------------------------------------------------------
Iteration 11 — common lab / vital features
------------------------------------------------------------
The remaining continuous lab-like features showed no meaningful association with PFS:
- feature_035 (range 6–18, ~hemoglobin g/dL): ρ = −0.004, p = 0.39.
- feature_007 (128–152, ~sodium): ρ = −0.001, p = 0.82.
- feature_006 (7.3–11.8, ~calcium): ρ = −0.013, p = 0.004 (statistically detectable, slope −0.064 mo/unit, but very small magnitude).
- feature_079 (14–47.7, ~BMI): ρ = +0.007, p = 0.14.
- feature_096 (0.3–2.1, ~creatinine): ρ = −0.005, p = 0.28.

------------------------------------------------------------
Iteration 12 — tumor-marker-like features
------------------------------------------------------------
None of the candidate tumor-marker features showed a meaningful PFS association on the raw scale:
- feature_059 (0.3–897): ρ = +0.006, p = 0.21.
- feature_027 (0.02–516): ρ = −0.001, p = 0.75.
- feature_046 (0.02–116): ρ = +0.005, p = 0.28.
- feature_064 (0.7–150): ρ = −0.003, p = 0.55.
- feature_053 (48–830, ~LDH): ρ = −0.013, p = 0.005 (slope −0.0005 mo per unit, trivial).
None practically meaningful.

------------------------------------------------------------
Iteration 13 — log transforms of skewed markers
------------------------------------------------------------
Log-transforming features 059 and 027 did not unmask any hidden association:
- log1p(feature_059): slope +0.017 mo/unit, p = 0.17. R² ≈ 0.
- log1p(feature_027): p = 0.998.

------------------------------------------------------------
Iteration 14 — other ordinal features
------------------------------------------------------------
None of the additional ordinal-coded features (021, 024, 127, 001, 073, 016) had a non-zero linear association with PFS (all p ≥ 0.30, slopes ≤ 0.011 mo/unit). They look like they encode patient or disease subtypes that are not prognostic in this cohort.

------------------------------------------------------------
Iteration 15 — rare binary features
------------------------------------------------------------
None of the rare binary features (feature_017, 094, 058, 118, 126; prevalences 0.5–2.6%) reached significance. Effect sizes were small (≤ 0.22 mo) with the lowest p = 0.08 (feature_094). They are not prognostic, or signal too small to detect with their prevalences.

------------------------------------------------------------
Iteration 16 — confounding by feature_080
------------------------------------------------------------
Despite feature_080's massive predictive power, adjustment for it does not change the binary treatment-like effects:
- feature_042: unadjusted β = +1.10, adjusted = +1.09 (almost identical).
- feature_056: unadjusted β = −1.54, adjusted = −1.55.
- Mean feature_080 by feature_042 status: 65.01 vs. 64.98 (diff = +0.03, p = 0.71).
- Mean feature_080 by feature_056 status: 65.04 vs. 64.96 (diff = +0.08, p = 0.42).
The strong predictors are independent of one another, so adjustment changes nothing and there is no case-mix confounding.

------------------------------------------------------------
Iteration 17 — adjusted disparity test + treatment access
------------------------------------------------------------
- After adjusting for feature_080, feature_063, feature_042, and feature_056, race coefficients in OLS are tiny (all ~−0.02) and the joint F-test p = 0.92. No residual race disparity.
- Same model with insurance: joint F p = 0.76. No residual insurance disparity.
- Chi-square of feature_042 by race: chi² = 1.25, p = 0.87. Proportions 0.337–0.351 across all five racial groups. Treatment-like access is uniform.
- Chi-square of feature_042 by insurance: chi² = 0.25, p = 0.97. Proportions 0.346–0.350. Uniform.

------------------------------------------------------------
Iteration 18 — variable importance via drop-1 R²
------------------------------------------------------------
Dropping each predictor in turn from the 12-feature multivariable model and measuring the loss of R²:
- feature_080: ΔR² = 0.491 (alone almost the entire model)
- feature_063: 0.101
- feature_056: 0.081
- feature_042: 0.044
- feature_048: 0.014
- feature_067: 0.014
- feature_015: 0.012
- feature_019: 0.009
- feature_101: 0.009
- feature_040: 0.005
- feature_111: 0.005
- feature_039: 0.000
feature_080 dominates — every other feature has < 1/5 of its importance.

------------------------------------------------------------
Iteration 19 — feature_063 × treatment-like interactions
------------------------------------------------------------
- feature_063 × feature_042: β = +0.009, p = 0.77. The +1.10-mo benefit of feature_042 is essentially identical at every feature_063 level (1.11 → 1.09 → 1.15 mo).
- feature_063 × feature_056: β = +0.067, p = 0.04. A statistically borderline interaction; harm of feature_056 ranges −1.60 (level 0) to −1.48 (level 2) — small.

------------------------------------------------------------
Iteration 20 — feature_080 × race / insurance
------------------------------------------------------------
- feature_080 × race: joint F p = 0.48. Stratified slopes 0.173–0.178 — flat.
- feature_080 × insurance: p = 0.63. Slopes 0.174–0.177.
No subgroup heterogeneity in the dominant continuous predictor.

------------------------------------------------------------
Iteration 21 — extreme-outcome contrasts (long PFS > 10 mo vs. short < 2 mo)
------------------------------------------------------------
- Mean feature_080: 75.5 (long, n=1554) vs. 52.2 (short, n=6688); diff +23 (p ≈ 0).
- feature_042 prevalence: 0.92 vs. 0.29 (chi² = 2089, p ≈ 0).
- feature_056 prevalence: 0.14 vs. 0.57 (chi² = 937, p ≈ 1e-205).
The extreme tails of PFS are differentiated almost entirely by feature_080, feature_042, and feature_056.

------------------------------------------------------------
Iteration 22 — additional weak binaries
------------------------------------------------------------
A handful of binary features had weak but statistically detectable effects:
- feature_086: −0.07 mo, p = 0.018.
- feature_077: +0.07 mo, p = 0.005.
- feature_031: −0.06 mo, p = 0.029.
- feature_005: −0.05 mo, p = 0.025.
- feature_034: +0.11 mo, p = 1e-6.
Likely real but clinically negligible.

------------------------------------------------------------
Iteration 23 — stratified treatment effect in extreme age strata
------------------------------------------------------------
- Elderly (feature_080 ≥ 75): feature_042 = +1.06 mo, feature_056 = −1.65 mo (p < 1e-87 for both).
- Young (feature_080 < 50): feature_042 = +1.10 mo, feature_056 = −1.02 mo (p < 1e-50 for both).
The protective and harmful effects persist in both extremes; the harm of feature_056 is somewhat larger in the elderly (−1.65 vs. −1.02), which echoes the small feature_080 × feature_056 interaction from iteration 8.

------------------------------------------------------------
Iteration 24 — residual disparity after adjustment
------------------------------------------------------------
After conditioning on the 12 strong predictors:
- Residual ANOVA across race: F = 0.35, p = 0.85. Mean residuals span 0.027 mo across all five groups.
- Residual ANOVA across insurance: F = 0.20, p = 0.89. Spans 0.015 mo.
There is no residual disparity in PFS by race or insurance after adjusting for clinical case-mix — and there was no raw disparity to begin with.

------------------------------------------------------------
Iteration 25 — overall predictive accuracy
------------------------------------------------------------
- 5-fold cross-validated OLS using all 127 features (one-hot encoded for the 2 categoricals): mean out-of-sample R² = 0.791 (sd 0.002 across folds). The model generalises essentially as well as it fits.
- feature_080 alone: R² = 0.492. So ~62% of the explainable variance is captured by this single feature, and the other 60% relative gain (0.49 → 0.79) comes mostly from feature_063, feature_056, feature_042, and feature_048.

============================================================
OVERALL CONCLUSIONS
============================================================
1. The cohort's PFS is dominated by a single continuous feature (feature_080) that monotonically increases mean PFS from ~1.7 mo (lowest decile) to ~7.8 mo (highest decile) and accounts for ~49% of variance alone. Direction (positive) and range (30–90) are most consistent with an age-like or stage-coded prognostic variable.
2. Two strong binary predictors operate roughly as treatment effects: feature_042 (+1.10 mo, prevalence 35%) and feature_111 (+0.57 mo, prevalence 70%) are protective; feature_056 (−1.54 mo, 30%), feature_048 (−1.05 mo, 10%), feature_015 (−0.56 mo), and feature_040 (−0.46 mo) are harmful. feature_039 (+0.36 mo) is mildly protective.
3. feature_063 (ordinal 0–2) is a strong prognostic axis, costing roughly 1.18 mo of PFS per level — consistent with a performance-status or stage-like ordinal score.
4. feature_019 (continuous, ~albumin range) is positively associated (+0.50 mo per unit). Other lab-like continuous features (sodium, calcium, creatinine, hemoglobin, BMI) have no meaningful relation with PFS.
5. Tumor-marker-like continuous features (feature_059, 027, 046, 064, 053) show no association with PFS, even after log transformation.
6. Effects are essentially additive: pairwise interactions among the strong binary predictors are not significant. The few statistically significant interactions involving feature_080 (× feature_063, feature_056, feature_042) are clinically negligible (slope shifts ~0.01).
7. Race (feature_011) and insurance (feature_089) show no main-effect difference in PFS (raw or adjusted), no heterogeneity in the effect of any treatment-like feature, and no difference in access to feature_042. The cohort shows no detectable race- or insurance-related disparity in PFS.
8. The best linear model with all 127 features achieves out-of-sample R² ≈ 0.79; a model with the 12 strongest predictors achieves R² ≈ 0.79 as well, and feature_080 alone reaches R² = 0.49. The dataset is highly explainable and dominated by a small handful of predictors.
"""

with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("Wrote transcript.json and analysis_summary.txt")
print("transcript size:", len(json.dumps(transcript)))
print("summary lines:", len(summary.splitlines()))
