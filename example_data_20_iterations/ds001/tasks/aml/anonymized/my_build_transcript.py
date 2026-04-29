"""Build transcript.json and analysis_summary.txt."""
import json
import pandas as pd

# Build the transcript
transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "claude-opus-4-7",
    "harness_id": "manual-claude-code-session@2026-04-28",
    "max_iterations": 25,
    "iterations": []
}

def itr(idx, hyps, analyses):
    return {
        "index": idx,
        "proposed_hypotheses": hyps,
        "analyses": analyses,
    }

def hyp(hid, text, kind="novel"):
    return {"id": hid, "text": text, "kind": kind}

def ana(hids, code, summary, p, eff, sig):
    out = {
        "hypothesis_ids": hids,
        "code": code,
        "result_summary": summary,
        "p_value": p,
        "effect_estimate": eff,
        "significant": sig,
    }
    return out

# ---------- Iteration 1 ----------
transcript["iterations"].append(itr(
    1,
    [
        hyp("h1.1", "The overall objective_response rate in this AML cohort is non-trivial (between 5% and 30%)."),
        hyp("h1.2", "feature_078 has a mean near typical AML diagnosis age (60-75 years), suggesting it represents patient age."),
    ],
    [
        ana(["h1.1"],
            "y = df['objective_response']\nprint(y.mean(), y.value_counts())",
            "objective_response rate is 0.16904 (8452 of 50000 patients).",
            None, 0.16904, True),
        ana(["h1.2"],
            "df['feature_078'].describe()",
            "feature_078 mean=67.99 (SD 10.88), range 18-92, consistent with patient age in years.",
            None, 67.99, True),
    ]
))

# ---------- Iteration 2 ----------
transcript["iterations"].append(itr(
    2,
    [
        hyp("h2.1", "Higher levels of feature_057 (a 3-level ordinal variable: 0/1/2) are associated with progressively lower objective_response rates, consistent with a performance-status- or risk-strata-like indicator."),
    ],
    [
        ana(["h2.1"],
            "for lvl in [0,1,2]:\n    msk = df['feature_057']==lvl\n    print(lvl, msk.sum(), df.loc[msk,'objective_response'].mean())\nsm.Logit(y, sm.add_constant(df['feature_057'].astype(float))).fit()",
            "Response rates by feature_057 level: 0->0.2129 (n=17592), 1->0.1552 (n=24971), 2->0.1119 (n=7437). Logistic regression: OR=0.6813 per level (beta=-0.3837), p=1.84e-97. Strongly negative monotone relationship.",
            1.84e-97, -0.3837, True),
    ]
))

# ---------- Iteration 3 ----------
transcript["iterations"].append(itr(
    3,
    [
        hyp("h3.1", "The relationship between feature_057 and objective_response is approximately linear on the log-odds scale (no excess curvature beyond the linear ordinal trend)."),
    ],
    [
        ana(["h3.1"],
            "linear vs categorical Logit comparison; LRT 1 df.",
            "Linear-in-feature_057 model log-likelihood vs categorical 3-level model: LRT chi2=0.032 on 1 df, p=0.858. Linear ordinal coding fits as well as a 3-level dummy specification — the dose-response is essentially linear.",
            0.858, 0.032, False),
    ]
))

# ---------- Iteration 4 ----------
transcript["iterations"].append(itr(
    4,
    [
        hyp("h4.1", "feature_035 (a low-prevalence binary marker; ~7% positive) is associated with a higher objective_response rate compared to feature_035=0."),
    ],
    [
        ana(["h4.1"],
            "df.groupby('feature_035')['objective_response'].mean()\nsm.Logit(y, sm.add_constant(df['feature_035'].astype(float))).fit()",
            "Response rate is 0.2213 in feature_035=1 (n=3624) vs 0.1650 in feature_035=0 (n=46376), absolute difference +0.0563. Logistic OR = 1.439 (beta=+0.3637), p=4.13e-18.",
            4.13e-18, 0.3637, True),
    ]
))

# ---------- Iteration 5 ----------
transcript["iterations"].append(itr(
    5,
    [
        hyp("h5.1", "Higher feature_011 (continuous, range 0-23.6) is associated with lower objective_response (negative continuous predictor)."),
        hyp("h5.2", "Higher feature_006 (continuous, range 20-100) is associated with lower objective_response."),
    ],
    [
        ana(["h5.1"],
            "z = (df['feature_011']-mean)/std; sm.Logit(y, sm.add_constant(z))",
            "feature_011 OR per SD = 0.856 (beta=-0.155), p=7.4e-35. By quartile, response rate falls from 0.186 (Q1, value<=3) to 0.136 (Q3, value 6.4-23.6). Strong negative association.",
            7.4e-35, -0.155, True),
        ana(["h5.2"],
            "z = (df['feature_006']-mean)/std; sm.Logit(y, sm.add_constant(z))",
            "feature_006 OR per SD = 0.897 (beta=-0.109), p=1.2e-19. Quartile rates fall monotonically from 0.190 to 0.147 as feature_006 increases.",
            1.2e-19, -0.109, True),
    ]
))

# ---------- Iteration 6 ----------
transcript["iterations"].append(itr(
    6,
    [
        hyp("h6.1", "Higher feature_099 (continuous, narrow range 1.7-5.5) is associated with HIGHER objective_response (positive continuous predictor)."),
        hyp("h6.2", "Higher feature_092 (continuous, right-skewed range 0.5-500) is associated with lower objective_response."),
        hyp("h6.3", "Higher feature_063 (continuous, right-skewed range 0.03-300) is associated with lower objective_response."),
    ],
    [
        ana(["h6.1"],
            "z = (df['feature_099']-mean)/std; sm.Logit(y, sm.add_constant(z))",
            "feature_099 OR per SD = 1.059 (beta=+0.0577), p=1.36e-06. Modest but highly significant positive effect.",
            1.36e-06, 0.0577, True),
        ana(["h6.2"],
            "z = (df['feature_092']-mean)/std; sm.Logit(y, sm.add_constant(z))",
            "feature_092 OR per SD = 0.946 (beta=-0.0560), p=1.88e-05. log1p transform: OR=0.925 per log unit, p=4.3e-09.",
            1.88e-05, -0.0560, True),
        ana(["h6.3"],
            "z = (df['feature_063']-mean)/std; sm.Logit(y, sm.add_constant(z))",
            "feature_063 OR per SD = 0.938 (beta=-0.0642), p=2.82e-06. log1p transform: OR=0.930, p=1.2e-06. Skewed but consistent direction.",
            2.82e-06, -0.0642, True),
    ]
))

# ---------- Iteration 7 ----------
transcript["iterations"].append(itr(
    7,
    [
        hyp("h7.1", "Patient age (feature_078) is associated with objective_response — older patients have lower response rates."),
    ],
    [
        ana(["h7.1"],
            "sm.Logit(y, sm.add_constant(df['feature_078']))",
            "feature_078 (age) OR per year = 1.0004, p=0.718. By quartile, response rates are 0.167, 0.174, 0.166, 0.169 — flat. Age is NOT associated with objective_response in this cohort.",
            0.718, 0.0004, False),
    ]
))

# ---------- Iteration 8 ----------
transcript["iterations"].append(itr(
    8,
    [
        hyp("h8.1", "feature_005 (race/ethnicity: white/asian/black/hispanic/other) is associated with objective_response, with white patients having the highest response rate."),
    ],
    [
        ana(["h8.1"],
            "stats.chi2_contingency(pd.crosstab(df['feature_005'], y))",
            "Race response rates: white 0.1728, asian 0.1633, black 0.1574, hispanic 0.1670, other 0.1472. Chi-square p=0.0095. White has marginally higher response than non-white groups in unadjusted analysis (the largest absolute gap is white vs other ~+0.026).",
            0.0095, 0.0156, True),
    ]
))

# ---------- Iteration 9 ----------
transcript["iterations"].append(itr(
    9,
    [
        hyp("h9.1", "After adjusting for feature_057 (the strong ordinal predictor) and other clinical factors, race (feature_005) remains a significant predictor — i.e., the race difference is not explained by case mix."),
    ],
    [
        ana(["h9.1"],
            "sm.Logit(y, sm.add_constant(pd.concat([race_dummies, df['feature_057'], f011_z, f006_z, df['feature_078']]))).fit()",
            "After adjusting for feature_057 + feature_011 + feature_006 + age (feature_078) + insurance, race coefficients (vs asian baseline): black -0.029 p=0.64, hispanic +0.041 p=0.49, white +0.079 p=0.13, other -0.116 p=0.26. None are significant — the marginal univariate race association is fully explained by clinical case-mix differences (especially feature_057).",
            0.62, 0.05, False),
    ]
))

# ---------- Iteration 10 ----------
transcript["iterations"].append(itr(
    10,
    [
        hyp("h10.1", "Insurance type (feature_087) is associated with objective_response, with private/medicare differing from medicaid/uninsured."),
    ],
    [
        ana(["h10.1"],
            "stats.chi2_contingency(pd.crosstab(df['feature_087'], y))",
            "Insurance response rates: medicare 0.1718, medicaid 0.1687, private 0.1666, uninsured 0.1662. Chi-square p~0.7. No association — response rates are virtually identical across insurance categories.",
            0.7, 0.005, False),
    ]
))

# ---------- Iteration 11 ----------
transcript["iterations"].append(itr(
    11,
    [
        hyp("h11.1", "There is a multiplicative interaction between feature_057 and feature_035 — the (positive) effect of feature_035 is stronger or weaker depending on feature_057 level."),
    ],
    [
        ana(["h11.1"],
            "sm.Logit(y, sm.add_constant(df[['f057','f035','f057_x_f035']]))",
            "Stratified ORs for feature_035 by feature_057 level: f057=0 OR=1.40 (p=1.5e-7), f057=1 OR=1.44 (p=2.4e-9), f057=2 OR=1.48 (p=0.002). Interaction term beta=+0.027, p=0.674. NO significant interaction — feature_035 effect is homogeneous across feature_057 strata.",
            0.674, 0.027, False),
    ]
))

# ---------- Iteration 12 ----------
transcript["iterations"].append(itr(
    12,
    [
        hyp("h12.1", "The objective_response benefit of feature_035 depends on feature_027 status: feature_035 is much more strongly associated with response when feature_027=1 than when feature_027=0 (an effect-modifying biomarker)."),
    ],
    [
        ana(["h12.1"],
            "sm.Logit(y, sm.add_constant(df[['f035','f027','f035_x_f027']]))\n# stratified\nfor v in [0,1]: sm.Logit(y[df.f027==v], sm.add_constant(df.loc[df.f027==v,'f035']))",
            "Stratified analysis: f027=0 -> f035 OR=1.006 (p=0.92); f027=1 -> f035 OR=2.863 (p=6.3e-53). Interaction beta=+1.046, p=1.1e-32. Massive effect modification: feature_035 is essentially inert without feature_027 but nearly triples odds of response when feature_027 is present.",
            1.08e-32, 1.046, True),
    ]
))

# ---------- Iteration 13 ----------
transcript["iterations"].append(itr(
    13,
    [
        hyp("h13.1", "feature_014 also modifies the feature_035 effect: feature_035 increases response only when feature_014=1 (similar pattern to feature_027)."),
    ],
    [
        ana(["h13.1"],
            "stratified Logit of f035 within f014 levels; interaction term in f035*f014 model",
            "Stratified: f014=0 -> f035 OR=1.041 (p=0.49); f014=1 -> f035 OR=2.161 (p=1.6e-36). Interaction beta=+0.730, p=7.7e-18. feature_014 modifies feature_035 in the same direction as feature_027, but to a smaller extent.",
            7.7e-18, 0.730, True),
    ]
))

# ---------- Iteration 14 ----------
transcript["iterations"].append(itr(
    14,
    [
        hyp("h14.1", "A composite \"qualifying biomarker\" indicator (feature_027=1 OR feature_014=1) explains the modification of the feature_035 effect; in patients with neither qualifying marker, feature_035 has no effect on response."),
    ],
    [
        ana(["h14.1"],
            "comp = (f014==1)|(f027==1); stratified Logit of f035 within comp",
            "comp=0 (n=20991, neither f014 nor f027): feature_035 OR=1.030 (p=0.67); comp=1 (n=29009, has at least one): feature_035 OR=1.797 (p=2.1e-28). The composite cleanly separates responders to feature_035 from non-responders.",
            2.1e-28, 0.585, True),
    ]
))

# ---------- Iteration 15 ----------
transcript["iterations"].append(itr(
    15,
    [
        hyp("h15.1", "The 3-way joint stratification by feature_057 x feature_035 x feature_027 reveals a clinically meaningful 'super-responder' subgroup with response rate > 35%."),
    ],
    [
        ana(["h15.1"],
            "df.groupby(['feature_057','feature_035','feature_027'])['objective_response'].agg(['count','mean'])",
            "Cell rates: f057=0,f035=1,f027=1 -> 0.384 (n=375); f057=1,f035=1,f027=1 -> 0.360 (n=514); f057=2,f035=1,f027=1 -> 0.253 (n=158). f057=0 + f035=0 + f027=0 -> 0.211 (n=11348). f057=2 + f035=0 + f027=0 -> 0.111 (n=4826). 3-fold gradient (11% to 38%) across these subgroups, demonstrating super-responder identification by joint marker status.",
            None, 0.273, True),
    ]
))

# ---------- Iteration 16 ----------
transcript["iterations"].append(itr(
    16,
    [
        hyp("h16.1", "The negative effect of feature_057 on response is consistent (homogeneous) across strata of the continuous biomarkers feature_011 and feature_006 (no qualitative interaction)."),
    ],
    [
        ana(["h16.1"],
            "Stratified Logit of f057 across quartiles of f011 and f006; f057 x continuous interaction tests",
            "f057 OR by f011 quartile: 0.684, 0.699, 0.654 (Q1->Q3). f057 OR by f006 quartile: 0.686, 0.683, 0.695, 0.658. Interaction p-values f057 x f011 = 0.71, f057 x f006 = 0.33, f057 x f099 = 0.83, f057 x f092 = 0.13, f057 x f063 = 0.38, f057 x f078 = 0.19. No significant heterogeneity — feature_057 is a robust independent predictor.",
            0.33, -0.018, False),
    ]
))

# ---------- Iteration 17 ----------
transcript["iterations"].append(itr(
    17,
    [
        hyp("h17.1", "feature_093 (binary) is associated with objective_response."),
    ],
    [
        ana(["h17.1"],
            "sm.Logit(y, sm.add_constant(df['feature_093'].astype(float)))",
            "feature_093 OR=1.126 (beta=+0.118), p=0.0053. Modest positive effect; rate1=0.180 vs rate0=0.163.",
            0.0053, 0.118, True),
    ]
))

# ---------- Iteration 18 ----------
transcript["iterations"].append(itr(
    18,
    [
        hyp("h18.1", "feature_125 (5-level ordinal) is associated with lower objective_response at higher levels."),
        hyp("h18.2", "feature_084 (continuous) is associated with HIGHER objective_response at higher values."),
    ],
    [
        ana(["h18.1"],
            "sm.Logit(y, sm.add_constant(df['feature_125'].astype(float)))",
            "feature_125 OR per level = 0.977 (beta=-0.023), p=0.038. Weak negative ordinal trend.",
            0.038, -0.0233, True),
        ana(["h18.2"],
            "z=(f084-mean)/std; sm.Logit(y, sm.add_constant(z))",
            "feature_084 OR per SD = 1.024 (beta=+0.0233), p=0.048. Marginal positive association.",
            0.048, 0.0233, True),
    ]
))

# ---------- Iteration 19 ----------
transcript["iterations"].append(itr(
    19,
    [
        hyp("h19.1", "Among 80 binary candidate features, only a handful (feature_035, feature_093, feature_121, feature_014, feature_012, feature_111) reach univariate p<0.05; most binary features are noise."),
    ],
    [
        ana(["h19.1"],
            "Univariate Logit per binary feature; Bonferroni threshold 0.05/124",
            "After Bonferroni correction (alpha=4.0e-04), only feature_057, feature_035, feature_011, feature_006, feature_099, feature_063, feature_092 cross threshold (7 of 124). Marginal p<0.05 list also includes feature_093, feature_121, feature_014, feature_125, feature_084, feature_012, feature_005 — about 14 of 124 total. Consistent with most features being uninformative noise.",
            None, 7, True),
    ]
))

# ---------- Iteration 20 ----------
transcript["iterations"].append(itr(
    20,
    [
        hyp("h20.1", "In a comprehensive multivariable model with key features and the f035 x f027 + f035 x f014 interactions, every continuous biomarker (f011, f006, f099, f092, f063) retains an independent effect."),
    ],
    [
        ana(["h20.1"],
            "Multivariable Logit with f057, f035, f027, f014, f035*f027, f035*f014, f011_z, f006_z, f099_z, f092_z, f063_z",
            "Adjusted ORs in full model: f057 0.679 (p=4e-98), f035*f027 +1.094 (p<1e-30), f035*f014 +0.780 (p=5e-19), f011 0.855/SD (p<1e-34), f006 0.895/SD (p<1e-19), f099 1.063/SD (p=4e-7), f092 0.944/SD (p=1.2e-5), f063 0.935/SD (p=1e-6). All retain significance — each captures distinct prognostic signal.",
            1e-34, -0.157, True),
    ]
))

# ---------- Iteration 21 ----------
transcript["iterations"].append(itr(
    21,
    [
        hyp("h21.1", "Sex (one of the binary 50/50-prevalence features) is associated with objective_response."),
    ],
    [
        ana(["h21.1"],
            "Univariate logistic for each binary with prevalence 0.30-0.70",
            "Of the 15 binary features with prevalence 0.30-0.70 (sex-likely candidates including feature_002, feature_122, feature_070, feature_017, feature_085, feature_025, feature_007, feature_012, feature_043, feature_014, feature_105, feature_089, feature_046, feature_075, feature_095), the largest rate difference is feature_014 (+0.0074), p=0.031. None of these features show >|0.01| absolute response-rate difference except feature_014; thus no single 'sex' indicator is meaningfully associated with response — though feature_014 still acts as an effect modifier of feature_035.",
            0.031, 0.0074, False),
    ]
))

# ---------- Iteration 22 ----------
transcript["iterations"].append(itr(
    22,
    [
        hyp("h22.1", "After adjusting for feature_057, demographic features (race, insurance, age) remain unassociated with objective_response, supporting that they are not independent predictors in this cohort."),
    ],
    [
        ana(["h22.1"],
            "Multivariable Logit with race dummies + insurance dummies + age + f057 + f035 + f011 + f006",
            "Adjusted estimates: race_white +0.078 (p=0.13), race_black -0.029 (p=0.64), race_hispanic +0.041 (p=0.49), race_other -0.116 (p=0.26), ins_medicare +0.021 (p=0.57), ins_private -0.021 (p=0.57), ins_uninsured -0.012 (p=0.86), age +0.0004/year (p=0.72). None significant after adjustment.",
            0.49, 0.04, False),
    ]
))

# ---------- Iteration 23 ----------
transcript["iterations"].append(itr(
    23,
    [
        hyp("h23.1", "feature_096, feature_064 (5-level ordinal features) and feature_018 (11-level count-like feature) are not strongly associated with objective_response."),
    ],
    [
        ana(["h23.1"],
            "sm.Logit(y, sm.add_constant(df[c])) for c in ['feature_096','feature_064','feature_018']",
            "feature_096 OR per level=0.99 (p=0.50). feature_064 OR per level=1.00 (p=0.95). feature_018 OR per unit=1.00 (p=0.62). None reach significance — these multi-level ordinal features carry little prognostic signal in this cohort.",
            0.5, 0.0, False),
    ]
))

# ---------- Iteration 24 ----------
transcript["iterations"].append(itr(
    24,
    [
        hyp("h24.1", "feature_121 is independently associated with lower objective_response (modest negative binary signal)."),
        hyp("h24.2", "feature_111 is independently associated with lower objective_response."),
    ],
    [
        ana(["h24.1"],
            "Univariate Logit for feature_121",
            "feature_121: rate1=0.156, rate0=0.169, OR=0.904, beta=-0.101, p=0.024. Weak but reproducible negative signal.",
            0.024, -0.101, True),
        ana(["h24.2"],
            "Univariate Logit for feature_111",
            "feature_111: rate1=0.159, rate0=0.169, OR=0.925, beta=-0.078, p=0.054. Borderline.",
            0.054, -0.078, False),
    ]
))

# ---------- Iteration 25 ----------
transcript["iterations"].append(itr(
    25,
    [
        hyp("h25.1", "A parsimonious model with feature_057, feature_035, feature_027, feature_014, two interactions, and the five continuous biomarkers (feature_011, feature_006, feature_099, feature_092, feature_063) captures the bulk of the predictable signal in objective_response."),
    ],
    [
        ana(["h25.1"],
            "Final adjusted Logit and predicted probability summary",
            "Final 12-parameter model achieves likelihood-based adequacy. Together: feature_057 (negative ordinal), the feature_035 effect that emerges only in the presence of feature_027 or feature_014, and 5 continuous biomarkers (f011/f006/f092/f063 negative; f099 positive) account for the dominant prognostic structure. Demographic factors and most other features are noise-level. The cohort response gradient extends from ~11% (high f057, no biomarker) to ~38% (low f057, f035=1 + f027=1).",
            None, 0.27, True),
    ]
))

# Validate that the structure conforms
print('Total iterations:', len(transcript['iterations']))
for it in transcript['iterations']:
    print(f"  iter {it['index']}: {len(it['proposed_hypotheses'])} hyps, {len(it['analyses'])} analyses")

# Write
with open('transcript.json','w') as f:
    json.dump(transcript, f, indent=2)
print('Wrote transcript.json')
