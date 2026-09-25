Investigate diverse, scientifically meaningful comparisons and use
previous results to allocate further investigation. Broad exploration and focused follow-up are
both useful throughout the run. Assess discoveries in this supplied dataset and use independent
validation when useful. Biological interpretation and generalizability may remain uncertain.

Register comparisons, anticipated directions, and initial assessments before requesting analyses.
The controller assigns claim references. For a changed claim, propose a new comparison and link
refinements to the earlier parent claim.
Clinical log_pfs_months is natural log progression-free survival in months, fully observed without
censoring. Dependency scores are continuous; more negative values mean stronger dependency.
Research signatures (feature_038, feature_018, feature_036) are constructed standardized assays in arbitrary units. Research markers (feature_007, feature_058, feature_015) are constructed binary assays with no assigned gene, pathway, or clinical role.
A mean_difference is exposed minus comparator within eligibility and subgroup. An interaction
subtracts that comparison in the subgroup complement in the same eligible population.
Use your scientific judgment to assess the size, uncertainty, and importance of each effect
and decide which conclusions the evidence supports. Explain your reasoning in the narrative.
Returned estimates and intervals are ALREADY SIGNED to the indicated hypothesis direction. Do not
multiply them by direction again. For example, raw -0.4 with direction -1 is reported as +0.4.
Discovery analyses use 95% Welch intervals and need at least 20 observations in every
comparison cell.

Each iteration has explore, analyze, appraise, and synthesize stages. Every stage may register
hypotheses for later analysis. Analyze selects up to 12 previously registered hypotheses.
Appraise assesses every new discovery result, records active/deferred/closed investigation, and may
request validation after considering those results. At most one new voluntary validation request
is allowed per iteration and ten per run. Cached requests reuse the same evidence. The evaluator
may also deliver scheduled validation during the run. Independent intervals account for the maximum
number of validation comparisons in the run. Results arrive before synthesis. Assess newly released
validation immediately in synthesis and explicitly reassess the original claim when its response
deadline appears in the context, even if you also investigate a refinement. The controller maintains
the accepted claim set from your assessments; you do not need to repeat that list. Explanations are
retained for inspection. Acceptance does not require prior independent validation.

The endpoint is a composite dependency score; it does not represent a single gene's knockout measurement.

Predictor names and text categorical values use opaque labels. Levels are nominal labels, not ordered numbers. Numeric values, outcome scales, and missingness are preserved.
