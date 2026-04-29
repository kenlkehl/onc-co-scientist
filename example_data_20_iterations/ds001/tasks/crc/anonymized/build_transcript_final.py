"""Assemble transcript.json from screen + iteration results."""
import json

R = json.load(open("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/crc/anonymized/iter_results.json"))


def hyp(hid, text, kind="novel"):
    return {"id": hid, "text": text, "kind": kind}


def ana(ids, summary, est, p, sig=None, code=None):
    rec = {
        "hypothesis_ids": ids if isinstance(ids, list) else [ids],
        "result_summary": summary,
        "effect_estimate": est,
        "p_value": p,
    }
    if sig is not None:
        rec["significant"] = sig
    if code:
        rec["code"] = code
    return rec


iterations = []

# ---------------- Iteration 1: top main effects (continuous + first binary)
r051 = R["h1_feature_051"]; r038 = R["h1_feature_038"]
r078 = R["h1_feature_078"]; r057 = R["h1_feature_057"]
iterations.append({
    "index": 1,
    "proposed_hypotheses": [
        hyp("h1_078", "Higher values of feature_078 (a continuous variable in the range ~30–90) are associated with longer pfs_months in this CRC cohort, with a positive linear slope."),
        hyp("h1_051", "Patients with feature_051 == 1 have shorter pfs_months than patients with feature_051 == 0."),
        hyp("h1_038", "Patients with feature_038 == 1 have longer pfs_months than patients with feature_038 == 0."),
        hyp("h1_057", "pfs_months decreases monotonically as the ordinal feature_057 (levels 0, 1, 2) increases."),
    ],
    "analyses": [
        ana("h1_078",
            f"Pearson/OLS slope: pfs_months increases by {r078['est']:.3f} per unit feature_078; very large effect (R^2 in univariate model implies r=+0.77).",
            est=r078["est"], p=r078["p"], sig=True,
            code="smf.ols('pfs_months ~ feature_078', data=df).fit()"),
        ana("h1_051",
            f"Welch t-test: mean pfs_months {r051['mean1']:.3f} vs {r051['mean0']:.3f}; difference {r051['est']:+.3f} months (n1={r051['n1']}, n0={r051['n0']}).",
            est=r051["est"], p=r051["p"], sig=True,
            code="stats.ttest_ind(df.loc[df.feature_051==1,'pfs_months'], df.loc[df.feature_051==0,'pfs_months'], equal_var=False)"),
        ana("h1_038",
            f"Welch t-test: feature_038==1 patients have +{r038['est']:.3f}-month longer pfs_months (n1={r038['n1']}, n0={r038['n0']}).",
            est=r038["est"], p=r038["p"], sig=True,
            code="stats.ttest_ind(df.loc[df.feature_038==1,'pfs_months'], df.loc[df.feature_038==0,'pfs_months'], equal_var=False)"),
        ana("h1_057",
            "Linear regression with feature_057 as numeric ordinal: slope {:+.3f} months per level; group means 5.27 (n=17592) -> 4.05 (n=24971) -> 2.91 (n=7437).".format(r057["est"]),
            est=r057["est"], p=r057["p"], sig=True,
            code="smf.ols('pfs_months ~ feature_057', data=df).fit()"),
    ],
})

# ---------------- Iteration 2: secondary main effects
r013 = R["h2_feature_013"]; r043 = R["h2_feature_043"]
r109 = R["h2_feature_109"]; r067 = R["h2_feature_067"]
r099 = R["h2_feature_099"]; r092 = R["h2_feature_092"]
iterations.append({
    "index": 2,
    "proposed_hypotheses": [
        hyp("h2_013", "Patients with feature_013 == 1 have shorter pfs_months than feature_013 == 0."),
        hyp("h2_043", "Patients with feature_043 == 1 have shorter pfs_months than feature_043 == 0."),
        hyp("h2_109", "Patients with the rare flag feature_109 == 1 have shorter pfs_months than feature_109 == 0."),
        hyp("h2_067", "Patients with the rare flag feature_067 == 1 have longer pfs_months than feature_067 == 0."),
        hyp("h2_099", "Higher values of the continuous variable feature_099 (range 0–~25) are associated with shorter pfs_months."),
        hyp("h2_092", "Higher values of the continuous variable feature_092 (range ~1.5–5.5) are associated with longer pfs_months."),
    ],
    "analyses": [
        ana("h2_013",
            f"Welch t-test: difference {r013['est']:+.3f} months (n1={r013['n1']}, n0={r013['n0']}).",
            est=r013["est"], p=r013["p"], sig=True),
        ana("h2_043",
            f"Welch t-test: difference {r043['est']:+.3f} months (n1={r043['n1']}, n0={r043['n0']}).",
            est=r043["est"], p=r043["p"], sig=True),
        ana("h2_109",
            f"Welch t-test on a 4.5%-prevalence flag: difference {r109['est']:+.3f} months (n1={r109['n1']}, n0={r109['n0']}).",
            est=r109["est"], p=r109["p"], sig=True),
        ana("h2_067",
            f"Welch t-test on a 3%-prevalence flag: difference {r067['est']:+.3f} months (n1={r067['n1']}, n0={r067['n0']}); rare positive predictor.",
            est=r067["est"], p=r067["p"], sig=True),
        ana("h2_099",
            f"OLS slope on feature_099: {r099['est']:+.4f} months per unit; small but very precisely estimated negative effect.",
            est=r099["est"], p=r099["p"], sig=True),
        ana("h2_092",
            f"OLS slope on feature_092: {r092['est']:+.3f} months per unit; clear positive effect over a narrow numeric range.",
            est=r092["est"], p=r092["p"], sig=True),
    ],
})

# ---------------- Iteration 3: feature_057 linear vs categorical
r3 = R["h3_feature_057_nonlinearity"]
iterations.append({
    "index": 3,
    "proposed_hypotheses": [
        hyp("h3_057_lin",
            "The relationship between feature_057 (levels 0, 1, 2) and pfs_months departs from a strict linear trend; group means are not equally spaced.",
            kind="refined"),
    ],
    "analyses": [
        ana("h3_057_lin",
            "Nested F test of categorical vs linear feature_057. Linear-model means decline 5.27 -> 4.05 -> 2.91 (steps of -1.22 and -1.14, very close). F = {:.2f}, p = {:.3f} — no significant departure from linearity at alpha=0.05.".format(r3["F_nonlin"], r3["p"]),
            est=r3["F_nonlin"], p=r3["p"], sig=False,
            code="anova_lm(smf.ols('pfs_months ~ feature_057'), smf.ols('pfs_months ~ C(feature_057)'))"),
    ],
})

# ---------------- Iteration 4: feature_078 quadratic
r4 = R["h4_feature_078_quadratic"]
iterations.append({
    "index": 4,
    "proposed_hypotheses": [
        hyp("h4_078_quad",
            "The relationship between feature_078 and pfs_months has a non-zero quadratic component (i.e., the slope curves rather than being strictly linear).",
            kind="refined"),
    ],
    "analyses": [
        ana("h4_078_quad",
            f"Adding I(feature_078**2) to the OLS model yields a quadratic coefficient of {r4['est']:+.5f} (p={r4['p']:.2e}); statistically significant but tiny — over a 60-unit span this curvature changes the slope by ~0.07 month/unit, so practical effect is small.",
            est=r4["est"], p=r4["p"], sig=True),
    ],
})

# ---------------- Iteration 5: multivariable main-effects model
r5 = R["h5_multivariable"]
iterations.append({
    "index": 5,
    "proposed_hypotheses": [
        hyp("h5_mv",
            "When feature_078, feature_051, feature_038, feature_057, feature_099, feature_092, feature_013, and feature_043 are entered jointly, every coefficient retains its univariate sign and remains highly significant (i.e., these effects are largely additive rather than confounded by each other).",
            kind="refined"),
    ],
    "analyses": [
        ana("h5_mv",
            "8-predictor OLS model. R^2 = 0.856. Coefficients (sign, p): feature_078 +0.176 (p≈0); feature_051 -1.36 (p≈0); feature_038 +0.94 (p≈0); feature_057 -1.17 (p≈0); feature_099 -0.076 (p≈0); feature_092 +0.47 (p≈0); feature_013 -0.33 (p≈0); feature_043 -0.29 (p≈0). All eight effects are independently significant with the expected signs.",
            est=r5["r2"], p=None, sig=True,
            code="smf.ols('pfs_months ~ feature_078 + feature_051 + feature_038 + feature_057 + feature_099 + feature_092 + feature_013 + feature_043').fit()"),
    ],
})

# ---------------- Iteration 6: feature_078 x feature_051 interaction
r6 = R["h6_feature_078_x_feature_051"]
iterations.append({
    "index": 6,
    "proposed_hypotheses": [
        hyp("h6_078x051",
            "The slope of pfs_months on feature_078 differs between feature_051 strata: specifically, the positive feature_078 slope is attenuated in patients with feature_051 == 1 (negative interaction)."),
    ],
    "analyses": [
        ana("h6_078x051",
            f"Interaction term feature_078:feature_051 = {r6['est']:+.4f} (p={r6['p']:.2e}); negative as predicted, meaning the per-unit feature_078 slope is ~{r6['est']:+.3f} smaller when feature_051==1.",
            est=r6["est"], p=r6["p"], sig=True),
    ],
})

# ---------------- Iteration 7: feature_078 x feature_057 interaction
r7 = R["h7_feature_078_x_feature_057"]
iterations.append({
    "index": 7,
    "proposed_hypotheses": [
        hyp("h7_078x057",
            "The positive feature_078 slope shrinks as feature_057 increases (negative feature_078 × feature_057 interaction); equivalently, feature_057's harmful effect is muted at high feature_078 values."),
    ],
    "analyses": [
        ana("h7_078x057",
            f"Interaction feature_078:feature_057 = {r7['est']:+.4f} (p={r7['p']:.2e}); negative direction supports the hypothesis.",
            est=r7["est"], p=r7["p"], sig=True),
    ],
})

# ---------------- Iteration 8: feature_038 x feature_051 interaction
r8 = R["h8_feature_038_x_feature_051"]
sub8 = R["h8_subgroups"]["sub"]
iterations.append({
    "index": 8,
    "proposed_hypotheses": [
        hyp("h8_038x051",
            "The benefit of feature_038 == 1 (longer pfs_months) is modified by feature_051 status; specifically, the feature_038 effect differs between feature_051 strata."),
    ],
    "analyses": [
        ana("h8_038x051",
            "Interaction feature_038:feature_051 = {:+.3f} (p={:.3f}). Subgroup means (pfs_months): f038=0,f051=0: {:.2f} (n={}); f038=1,f051=0: {:.2f} (n={}); f038=0,f051=1: {:.2f} (n={}); f038=1,f051=1: {:.2f} (n={}). The two main effects appear largely additive — interaction not significant.".format(
                r8["est"], r8["p"],
                sub8["f038=0,f051=0"]["mean"], sub8["f038=0,f051=0"]["n"],
                sub8["f038=1,f051=0"]["mean"], sub8["f038=1,f051=0"]["n"],
                sub8["f038=0,f051=1"]["mean"], sub8["f038=0,f051=1"]["n"],
                sub8["f038=1,f051=1"]["mean"], sub8["f038=1,f051=1"]["n"],
            ),
            est=r8["est"], p=r8["p"], sig=False),
    ],
})

# ---------------- Iteration 9: feature_038 x feature_057 interaction
r9 = R["h9_feature_038_x_feature_057"]
sub9 = R["h9_subgroups"]["sub"]
iterations.append({
    "index": 9,
    "proposed_hypotheses": [
        hyp("h9_038x057",
            "The benefit of feature_038 == 1 is similar across feature_057 levels (no significant feature_038 × feature_057 interaction)."),
    ],
    "analyses": [
        ana("h9_038x057",
            "Interaction feature_038:feature_057 = {:+.3f} (p={:.2f}). Subgroup means: f057=0 -> {:.2f}/{:.2f} (off/on f038), f057=1 -> {:.2f}/{:.2f}, f057=2 -> {:.2f}/{:.2f}. The +1-month feature_038 benefit is consistent across strata.".format(
                r9["est"], r9["p"],
                sub9["f038=0,f057=0"]["mean"], sub9["f038=1,f057=0"]["mean"],
                sub9["f038=0,f057=1"]["mean"], sub9["f038=1,f057=1"]["mean"],
                sub9["f038=0,f057=2"]["mean"], sub9["f038=1,f057=2"]["mean"],
            ),
            est=r9["est"], p=r9["p"], sig=False),
    ],
})

# ---------------- Iteration 10: feature_051 x feature_057 interaction
r10 = R["h10_feature_051_x_feature_057"]
iterations.append({
    "index": 10,
    "proposed_hypotheses": [
        hyp("h10_051x057",
            "The feature_051 effect on pfs_months differs across feature_057 levels — specifically, the gap between feature_051==1 and feature_051==0 narrows as feature_057 increases (positive interaction in regression scale)."),
    ],
    "analyses": [
        ana("h10_051x057",
            f"Interaction feature_051:feature_057 = {r10['est']:+.3f} (p={r10['p']:.3f}); modestly significant positive interaction, indicating slight attenuation of feature_051's negative effect at higher feature_057.",
            est=r10["est"], p=r10["p"], sig=(r10["p"] < 0.05)),
    ],
})

# ---------------- Iteration 11: feature_013 x feature_043 interaction
r11 = R["h11_feature_013_x_feature_043"]
iterations.append({
    "index": 11,
    "proposed_hypotheses": [
        hyp("h11_013x043",
            "Patients with both feature_013==1 AND feature_043==1 have pfs_months that exceeds the additive prediction from each effect alone (positive interaction term, i.e., the two negative main effects sub-add)."),
    ],
    "analyses": [
        ana("h11_013x043",
            f"Interaction feature_013:feature_043 = {r11['est']:+.3f} (p={r11['p']:.2e}); strongly positive — the two negative main effects do not fully stack on top of each other.",
            est=r11["est"], p=r11["p"], sig=True),
    ],
})

# ---------------- Iteration 12: race
r12 = R["h12_feature_064_race"]
r12a = R["h12_feature_064_race_adjusted"]
race_means = r12["means"]; race_counts = r12["counts"]
iterations.append({
    "index": 12,
    "proposed_hypotheses": [
        hyp("h12_race",
            "Mean pfs_months differs across feature_064 (race/ethnicity-coded) groups in this cohort, both unadjusted and after adjusting for the strongest predictors (feature_078, feature_051, feature_057)."),
    ],
    "analyses": [
        ana("h12_race",
            "Unadjusted ANOVA across feature_064 levels (white={:.2f} n={}, black={:.2f} n={}, hispanic={:.2f} n={}, asian={:.2f} n={}, other={:.2f} n={}); F-test p={:.3f}. After adjusting for feature_078, feature_051, feature_057, joint F-test p={:.3f}. No detectable race effect on pfs_months in this cohort.".format(
                race_means.get("white", 0), race_counts.get("white", 0),
                race_means.get("black", 0), race_counts.get("black", 0),
                race_means.get("hispanic", 0), race_counts.get("hispanic", 0),
                race_means.get("asian", 0), race_counts.get("asian", 0),
                race_means.get("other", 0), race_counts.get("other", 0),
                r12["p"], r12a["p"]),
            est=max(race_means.values()) - min(race_means.values()),
            p=r12a["p"], sig=False),
    ],
})

# ---------------- Iteration 13: insurance
r13 = R["h13_feature_018_insurance"]
r13a = R["h13_feature_018_insurance_adjusted"]
ins_means = r13["means"]; ins_counts = r13["counts"]
iterations.append({
    "index": 13,
    "proposed_hypotheses": [
        hyp("h13_ins",
            "Mean pfs_months differs across feature_018 (insurance type) categories — patients on private insurance have longer pfs_months than uninsured/medicaid patients, both unadjusted and after adjusting for the strongest predictors."),
    ],
    "analyses": [
        ana("h13_ins",
            "Unadjusted means: medicare={:.2f} (n={}), private={:.2f} (n={}), uninsured={:.2f} (n={}), medicaid={:.2f} (n={}); ANOVA p={:.2f}. Adjusted joint F-test p={:.2f}. Insurance type does not significantly predict pfs_months in this cohort.".format(
                ins_means.get("medicare", 0), ins_counts.get("medicare", 0),
                ins_means.get("private", 0), ins_counts.get("private", 0),
                ins_means.get("uninsured", 0), ins_counts.get("uninsured", 0),
                ins_means.get("medicaid", 0), ins_counts.get("medicaid", 0),
                r13["p"], r13a["p"]),
            est=max(ins_means.values()) - min(ins_means.values()),
            p=r13a["p"], sig=False),
    ],
})

# ---------------- Iteration 14: feature_078 × feature_038 interaction
r14 = R["h14_feature_078_x_feature_038"]
iterations.append({
    "index": 14,
    "proposed_hypotheses": [
        hyp("h14_078x038",
            "The benefit of feature_038==1 (longer pfs_months) is amplified at higher values of feature_078 (positive feature_078 × feature_038 interaction)."),
    ],
    "analyses": [
        ana("h14_078x038",
            f"Interaction feature_078:feature_038 = {r14['est']:+.4f} (p={r14['p']:.3f}); the modest negative point estimate is non-significant. The feature_038 benefit appears roughly constant across feature_078.",
            est=r14["est"], p=r14["p"], sig=False),
    ],
})

# ---------------- Iteration 15: feature_099 nonlinearity & feature_099 × feature_051
r15q = R["h15_feature_099_quadratic"]
r15i = R["h15_feature_099_x_feature_051"]
iterations.append({
    "index": 15,
    "proposed_hypotheses": [
        hyp("h15_099_quad",
            "The relationship between feature_099 and pfs_months has a non-zero quadratic component (curving rather than strictly linear).",
            kind="refined"),
        hyp("h15_099x051",
            "The negative feature_099 slope is steeper among feature_051==1 patients (negative feature_099 × feature_051 interaction)."),
    ],
    "analyses": [
        ana("h15_099_quad",
            f"Quadratic term I(feature_099**2) = {r15q['est']:+.5f} (p={r15q['p']:.3f}); not significant. Linear approximation suffices for feature_099.",
            est=r15q["est"], p=r15q["p"], sig=False),
        ana("h15_099x051",
            f"Interaction feature_099:feature_051 = {r15i['est']:+.4f} (p={r15i['p']:.3f}); not significant — feature_099's negative effect is similar across feature_051 strata.",
            est=r15i["est"], p=r15i["p"], sig=False),
    ],
})

# ---------------- Iteration 16: feature_092 × feature_057
r16 = R["h16_feature_092_x_feature_057"]
iterations.append({
    "index": 16,
    "proposed_hypotheses": [
        hyp("h16_092x057",
            "The positive feature_092 slope is attenuated at higher feature_057 levels (negative feature_092 × feature_057 interaction)."),
    ],
    "analyses": [
        ana("h16_092x057",
            f"Interaction feature_092:feature_057 = {r16['est']:+.4f} (p={r16['p']:.2f}); not significant. feature_092 benefit is similar across feature_057.",
            est=r16["est"], p=r16["p"], sig=False),
    ],
})

# ---------------- Iteration 17: heteroscedasticity by feature_057
r17 = R["h17_levene_feature_057"]
iterations.append({
    "index": 17,
    "proposed_hypotheses": [
        hyp("h17_var",
            "Residual variance of pfs_months (after main-effect adjustment) differs across feature_057 levels — specifically the worst-prognosis stratum (feature_057==2) shows different variance than feature_057==0."),
    ],
    "analyses": [
        ana("h17_var",
            f"Levene test on residuals from a model with feature_078 + feature_051 + feature_038, grouped by feature_057: residual SDs 0={r17['sd0']:.3f}, 1={r17['sd1']:.3f}, 2={r17['sd2']:.3f}; W={r17['W']:.2f}, p={r17['p']:.3f}. Mild but significant variance heterogeneity.",
            est=r17["sd2"] - r17["sd0"], p=r17["p"], sig=True),
    ],
})

# ---------------- Iteration 18: 3-way interaction
r18 = R["h18_3way"]
iterations.append({
    "index": 18,
    "proposed_hypotheses": [
        hyp("h18_3way",
            "There is a positive 3-way interaction feature_038 × feature_051 × feature_057 — the joint pattern of these three predictors yields larger pfs_months than additive expectation in some cells."),
    ],
    "analyses": [
        ana("h18_3way",
            f"3-way interaction term = {r18['est']:+.3f} (p={r18['p']:.2f}); not significant. The 3-way structure does not exceed what 2-way and main effects already capture.",
            est=r18["est"], p=r18["p"], sig=False),
    ],
})

# ---------------- Iteration 19: full model with selected interactions
r19 = R["h19_full_model"]
iterations.append({
    "index": 19,
    "proposed_hypotheses": [
        hyp("h19_full",
            "A multivariable model containing main effects of feature_078, feature_051, feature_038, feature_057, feature_099, feature_092, feature_013, feature_043, feature_109, feature_067 plus the interactions feature_038×feature_051, feature_038×feature_057, feature_078×feature_051 explains substantially more variance than the simpler 8-predictor main-effects model.",
            kind="refined"),
    ],
    "analyses": [
        ana("h19_full",
            "Full model R^2 = {:.4f} (vs simpler model R^2 = 0.8561 in iteration 5); incremental R^2 ≈ 0.0013. Significant interactions retained: feature_078:feature_051 ({:+.4f}, p={:.2e}), feature_038:feature_057 ({:+.3f}, p={:.4f}). feature_038:feature_051 was non-significant at p={:.3f}.".format(
                r19["r2"],
                r19["params"]["feature_078:feature_051"]["est"], r19["params"]["feature_078:feature_051"]["p"],
                r19["params"]["feature_038:feature_057"]["est"], r19["params"]["feature_038:feature_057"]["p"],
                r19["params"]["feature_038:feature_051"]["p"],
            ),
            est=r19["r2"], p=None, sig=True),
    ],
})

# ---------------- Iteration 20: feature_109 adjusted
r20 = R["h20_feature_109_adjusted"]
iterations.append({
    "index": 20,
    "proposed_hypotheses": [
        hyp("h20_109_adj",
            "The negative effect of feature_109 on pfs_months persists after adjusting for feature_078, feature_051, and feature_057 (i.e., it is not explained away by these strong covariates)."),
    ],
    "analyses": [
        ana("h20_109_adj",
            f"Adjusted feature_109 coefficient = {r20['est']:+.3f} (p={r20['p']:.2e}); effect is ~13% smaller than unadjusted but remains highly significant.",
            est=r20["est"], p=r20["p"], sig=True),
    ],
})

# ---------------- Iteration 21: feature_067 adjusted
r21 = R["h21_feature_067_adjusted"]
iterations.append({
    "index": 21,
    "proposed_hypotheses": [
        hyp("h21_067_adj",
            "The positive effect of feature_067 on pfs_months persists after adjusting for feature_078, feature_051, and feature_057."),
    ],
    "analyses": [
        ana("h21_067_adj",
            f"Adjusted feature_067 coefficient = {r21['est']:+.3f} (p={r21['p']:.2e}); positive direction confirmed and significant in the adjusted model.",
            est=r21["est"], p=r21["p"], sig=True),
    ],
})

# ---------------- Iteration 22: feature_038 within feature_051 strata (subgroup)
r22 = R["h22_feature_038_within_feature_051"]; r22s = r22["sub"]
iterations.append({
    "index": 22,
    "proposed_hypotheses": [
        hyp("h22_038_in_051",
            "The pfs_months benefit of feature_038==1 vs ==0 is significant in BOTH feature_051==0 and feature_051==1 strata, with similar magnitude (~+0.9 to +1.0 month)."),
    ],
    "analyses": [
        ana("h22_038_in_051",
            "Stratified Welch t-tests: feature_051==0 -> diff {:+.3f} (n1={}, n0={}, p={:.2e}); feature_051==1 -> diff {:+.3f} (n1={}, n0={}, p={:.2e}). Effect is consistently positive and similar in size in both strata.".format(
                r22s["f051=0"]["diff"], r22s["f051=0"]["n1"], r22s["f051=0"]["n0"], r22s["f051=0"]["p"],
                r22s["f051=1"]["diff"], r22s["f051=1"]["n1"], r22s["f051=1"]["n0"], r22s["f051=1"]["p"],
            ),
            est=r22s["f051=1"]["diff"] - r22s["f051=0"]["diff"], p=None, sig=True),
    ],
})

# ---------------- Iteration 23: feature_038 within feature_057 strata
r23 = R["h23_feature_038_within_feature_057"]; r23s = r23["sub"]
iterations.append({
    "index": 23,
    "proposed_hypotheses": [
        hyp("h23_038_in_057",
            "The feature_038 benefit is preserved across all three feature_057 levels (0, 1, 2), with similar magnitude (~+0.9 month) — i.e., the effect is not concentrated in any one stratum."),
    ],
    "analyses": [
        ana("h23_038_in_057",
            "Stratified Welch t-tests: f057=0 -> diff {:+.3f} (n1={}, n0={}, p={:.2e}); f057=1 -> diff {:+.3f} (n1={}, n0={}, p={:.2e}); f057=2 -> diff {:+.3f} (n1={}, n0={}, p={:.2e}). Effect is consistent across strata.".format(
                r23s["f057=0"]["diff"], r23s["f057=0"]["n1"], r23s["f057=0"]["n0"], r23s["f057=0"]["p"],
                r23s["f057=1"]["diff"], r23s["f057=1"]["n1"], r23s["f057=1"]["n0"], r23s["f057=1"]["p"],
                r23s["f057=2"]["diff"], r23s["f057=2"]["n1"], r23s["f057=2"]["n0"], r23s["f057=2"]["p"],
            ),
            est=r23s["f057=2"]["diff"] - r23s["f057=0"]["diff"], p=None, sig=True),
    ],
})

# ---------------- Iteration 24: minor binary predictors adjusted
r24 = R["h24_minor_binary_adjusted"]["results"]
iterations.append({
    "index": 24,
    "proposed_hypotheses": [
        hyp("h24_089",
            "feature_089 (univariate p≈0.017) is not an independent predictor of pfs_months once feature_078, feature_051, feature_038, and feature_057 are adjusted for."),
        hyp("h24_other",
            "feature_102, feature_112, and feature_079 (all weak univariate signals) are similarly not independent predictors after adjustment."),
    ],
    "analyses": [
        ana("h24_089",
            f"Adjusted coefficient for feature_089 = {r24['feature_089']['est']:+.4f} (p={r24['feature_089']['p']:.3f}); only borderline independent signal.",
            est=r24["feature_089"]["est"], p=r24["feature_089"]["p"], sig=(r24["feature_089"]["p"] < 0.05)),
        ana("h24_other",
            "Adjusted coefficients: feature_102 = {:+.4f} (p={:.2f}); feature_112 = {:+.4f} (p={:.2f}); feature_079 = {:+.4f} (p={:.2f}). All three lose significance after adjustment.".format(
                r24["feature_102"]["est"], r24["feature_102"]["p"],
                r24["feature_112"]["est"], r24["feature_112"]["p"],
                r24["feature_079"]["est"], r24["feature_079"]["p"],
            ),
            est=r24["feature_102"]["est"], p=r24["feature_102"]["p"], sig=False),
    ],
})

# ---------------- Iteration 25: feature_078 binned means (linearity check)
r25m = R["h25_feature_078_bin_means"]["means"]
r25c = R["h25_feature_078_bin_means"]["counts"]
iterations.append({
    "index": 25,
    "proposed_hypotheses": [
        hyp("h25_078_lin",
            "Mean pfs_months across coarse bins of feature_078 (30–50, 50–60, 60–70, 70–80, 80–90) increases monotonically with roughly equal step size, consistent with a near-linear relationship.",
            kind="refined"),
    ],
    "analyses": [
        ana("h25_078_lin",
            "Bin means: 30-50 -> {:.2f} (n={}); 50-60 -> {:.2f} (n={}); 60-70 -> {:.2f} (n={}); 70-80 -> {:.2f} (n={}); 80-90 -> {:.2f} (n={}). Steps between successive bins: +1.57, +1.64, +1.66, +1.83 — strikingly linear (slope ≈ 0.17/unit, matching the global OLS slope from iteration 1).".format(
                r25m["30-50"], r25c["30-50"],
                r25m["50-60"], r25c["50-60"],
                r25m["60-70"], r25c["60-70"],
                r25m["70-80"], r25c["70-80"],
                r25m["80-90"], r25c["80-90"],
            ),
            est=r25m["80-90"] - r25m["30-50"], p=None, sig=True),
    ],
})

transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-manual@1.0",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/crc/anonymized/transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

# Validate against schema
import jsonschema
schema = json.load(open("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/crc/anonymized/transcript_schema.json"))
jsonschema.validate(transcript, schema)
print("transcript.json written and validated OK.")
print("iterations:", len(transcript["iterations"]))
print("total hypotheses:", sum(len(it["proposed_hypotheses"]) for it in iterations))
print("total analyses:", sum(len(it["analyses"]) for it in iterations))
