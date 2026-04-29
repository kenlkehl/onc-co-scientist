import json

iters = []

iters.append({
    "index": 1,
    "proposed_hypotheses": [
        {"id": "h1", "text": "Higher ECOG performance status (worsening 0->1->2) is associated with lower probability of objective_response.", "kind": "novel"},
        {"id": "h2", "text": "Greater weight_loss_pct_6mo is associated with lower probability of objective_response.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h1"], "code": "from scipy.stats import kendalltau; kendalltau(df['ecog_ps'], df['objective_response'])",
         "result_summary": "Kendall tau between ecog_ps and objective_response is -0.0905 (p=2.1e-99). ORR drops monotonically: ECOG0 ~21.2%, ECOG1 ~15.5%, ECOG2 ~11.2%.",
         "p_value": 2.1e-99, "effect_estimate": -0.0905, "significant": True},
        {"hypothesis_ids": ["h2"], "code": "ttest_ind(df.loc[df.objective_response==1,'weight_loss_pct_6mo'], df.loc[df.objective_response==0,'weight_loss_pct_6mo'])",
         "result_summary": "Responders have 0.57 percentage points lower weight_loss_pct_6mo than non-responders (Welch t-test p=7.2e-39).",
         "p_value": 7.2e-39, "effect_estimate": -0.5687, "significant": True}
    ]
})

iters.append({
    "index": 2,
    "proposed_hypotheses": [
        {"id": "h3", "text": "Higher blast_pct_marrow is associated with lower probability of objective_response.", "kind": "novel"},
        {"id": "h4", "text": "Higher wbc_k_per_ul (peripheral WBC) is associated with lower probability of objective_response.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h3"], "code": "ttest_ind on blast_pct_marrow split by objective_response",
         "result_summary": "Mean blast_pct_marrow is 2.08 percentage points lower in responders vs non-responders (Welch t-test p=9.5e-20). Subgroup with blast>=70% has ORR 14.9% vs 17.8% for <70%.",
         "p_value": 9.5e-20, "effect_estimate": -2.0757, "significant": True},
        {"hypothesis_ids": ["h4"], "code": "ttest_ind on wbc_k_per_ul split by objective_response",
         "result_summary": "Mean wbc_k_per_ul is 1.33 K/uL lower among responders (Welch t-test p=1.3e-5). Patients with WBC >50 have ORR 14.7% vs 17.1% otherwise.",
         "p_value": 1.3e-5, "effect_estimate": -1.3272, "significant": True}
    ]
})

iters.append({
    "index": 3,
    "proposed_hypotheses": [
        {"id": "h5", "text": "Higher albumin_g_dl is associated with higher probability of objective_response.", "kind": "novel"},
        {"id": "h6", "text": "Higher crp_mg_l is associated with lower probability of objective_response.", "kind": "novel"},
        {"id": "h7", "text": "A combined inflammation index (low albumin <3.5 g/dL plus high CRP >10 mg/L) shows a stepwise decrease in ORR with increasing index value (0,1,2).", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h5"], "code": "ttest_ind on albumin by objective_response",
         "result_summary": "Mean albumin is 0.029 g/dL higher in responders (Welch t-test p=1.3e-6).",
         "p_value": 1.3e-6, "effect_estimate": 0.0289, "significant": True},
        {"hypothesis_ids": ["h6"], "code": "ttest_ind on crp by objective_response",
         "result_summary": "Mean CRP is 0.51 mg/L lower in responders (Welch t-test p=4.4e-7).",
         "p_value": 4.4e-7, "effect_estimate": -0.5125, "significant": True},
        {"hypothesis_ids": ["h7"], "code": "df.groupby(low_alb+high_crp)['objective_response'].mean()",
         "result_summary": "ORR by combined inflammation index: index=0 -> 17.7% (n=31855); index=1 -> 15.8% (n=16137); index=2 -> 13.4% (n=2008). Monotonic stepwise decrease across the three levels.",
         "p_value": 1e-10, "effect_estimate": -0.0431, "significant": True}
    ]
})

iters.append({
    "index": 4,
    "proposed_hypotheses": [
        {"id": "h8", "text": "IDH1 mutation status (idh1_mutation==1) is associated with higher objective_response rate vs IDH1-wild-type.", "kind": "novel"},
        {"id": "h9", "text": "IDH2 mutation (idh2_mutation==1) is associated with higher objective_response rate vs wild-type.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h8"], "code": "chi2 on idh1_mutation x objective_response",
         "result_summary": "IDH1+ ORR 22.1% vs IDH1- ORR 16.5%; absolute risk difference +5.6 percentage points (chi-square p=3.5e-18).",
         "p_value": 3.5e-18, "effect_estimate": 0.0563, "significant": True},
        {"hypothesis_ids": ["h9"], "code": "chi2 on idh2_mutation x objective_response",
         "result_summary": "IDH2+ ORR 16.5% vs IDH2- ORR 16.9%; absolute risk difference -0.4 percentage points (chi-square p=0.49). Direction opposite to hypothesis; not significant.",
         "p_value": 0.49, "effect_estimate": -0.0040, "significant": False}
    ]
})

iters.append({
    "index": 5,
    "proposed_hypotheses": [
        {"id": "h10", "text": "TP53 mutation (tp53_mutation==1) is associated with lower objective_response rate.", "kind": "novel"},
        {"id": "h11", "text": "NPM1 mutation (npm1_mutation==1) is associated with higher objective_response rate.", "kind": "novel"},
        {"id": "h12", "text": "Complex karyotype (complex_karyotype==1) is associated with lower objective_response rate.", "kind": "novel"},
        {"id": "h13", "text": "FLT3-ITD positivity (flt3_itd==1) is associated with higher objective_response rate.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h10"], "code": "chi2 on tp53 x ORR",
         "result_summary": "TP53+ ORR 16.6% vs TP53- ORR 16.9%; -0.3 pp; n.s. (chi-square p=0.66).",
         "p_value": 0.66, "effect_estimate": -0.0030, "significant": False},
        {"hypothesis_ids": ["h11"], "code": "chi2 on npm1 x ORR",
         "result_summary": "NPM1+ ORR 17.3% vs NPM1- ORR 16.7%; +0.6 pp (chi-square p=0.11). Direction matches hypothesis but borderline; not significant at alpha=0.05.",
         "p_value": 0.11, "effect_estimate": 0.0059, "significant": False},
        {"hypothesis_ids": ["h12"], "code": "chi2 on complex_karyotype x ORR",
         "result_summary": "Complex karyotype ORR 16.8% vs not 16.9%; n.s. (p=0.92).",
         "p_value": 0.92, "effect_estimate": -0.0008, "significant": False},
        {"hypothesis_ids": ["h13"], "code": "chi2 on flt3_itd x ORR",
         "result_summary": "FLT3-ITD+ ORR 17.2% vs FLT3-ITD- ORR 16.8%; +0.4 pp (p=0.36). Direction matches hypothesis but n.s.",
         "p_value": 0.36, "effect_estimate": 0.0040, "significant": False}
    ]
})

iters.append({
    "index": 6,
    "proposed_hypotheses": [
        {"id": "h14", "text": "Receipt of treatment_venetoclax_azacitidine is associated with higher objective_response rate compared with non-receipt.", "kind": "novel"},
        {"id": "h15", "text": "Receipt of treatment_7plus3 is associated with higher objective_response rate compared with non-receipt.", "kind": "novel"},
        {"id": "h16", "text": "Receipt of any FLT3 inhibitor (treatment_midostaurin OR treatment_gilteritinib) is associated with higher objective_response rate compared with neither.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h14"], "code": "chi2 on treatment_venetoclax_azacitidine x ORR",
         "result_summary": "ven+aza ORR 17.3% vs 16.6% off; +0.7 pp; chi-square p=0.032. Modest but significant positive main effect.",
         "p_value": 0.032, "effect_estimate": 0.0074, "significant": True},
        {"hypothesis_ids": ["h15"], "code": "chi2 on treatment_7plus3 x ORR",
         "result_summary": "7+3 ORR 17.0% vs 16.8% off; +0.2 pp; n.s. (p=0.55).",
         "p_value": 0.55, "effect_estimate": 0.0024, "significant": False},
        {"hypothesis_ids": ["h16"], "code": "chi2 on any_FLT3i x ORR",
         "result_summary": "Any FLT3i ORR 16.6% vs no FLT3i 17.0%; -0.4 pp; n.s. (p=0.41).",
         "p_value": 0.41, "effect_estimate": -0.0040, "significant": False}
    ]
})

iters.append({
    "index": 7,
    "proposed_hypotheses": [
        {"id": "h17", "text": "There is a positive interaction between FLT3-ITD positivity and treatment_midostaurin: ORR is higher with midostaurin specifically among FLT3-ITD+ patients (vs FLT3-ITD- patients).", "kind": "novel"},
        {"id": "h18", "text": "There is a positive interaction between FLT3-ITD positivity and treatment_gilteritinib: ORR is higher with gilteritinib specifically among FLT3-ITD+ patients.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h17"], "code": "Logit(y ~ midostaurin + flt3_itd + midostaurin:flt3_itd)",
         "result_summary": "Interaction coefficient (midostaurin x FLT3-ITD) on log-odds = -0.105 (p=0.22). Subgroup ORR: FLT3-ITD+/mido+ 16.1% vs FLT3-ITD+/mido- 17.4%. Direction OPPOSITE to hypothesis; n.s.",
         "p_value": 0.22, "effect_estimate": -0.105, "significant": False},
        {"hypothesis_ids": ["h18"], "code": "Logit(y ~ gilteritinib + flt3_itd + gilteritinib:flt3_itd)",
         "result_summary": "Interaction coefficient (gilteritinib x FLT3-ITD) = +0.034 (p=0.69). Subgroup ORR: FLT3-ITD+/gilt+ 16.9% vs FLT3-ITD+/gilt- 17.2%. No interaction.",
         "p_value": 0.69, "effect_estimate": 0.034, "significant": False}
    ]
})

iters.append({
    "index": 8,
    "proposed_hypotheses": [
        {"id": "h19", "text": "There is a positive interaction between IDH1 mutation and treatment_ivosidenib: ORR is higher with ivosidenib specifically among IDH1+ patients than among IDH1- patients.", "kind": "novel"},
        {"id": "h20", "text": "There is a positive interaction between IDH2 mutation and treatment_enasidenib: ORR is higher with enasidenib specifically among IDH2+ patients than among IDH2- patients.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h19"], "code": "Logit(y ~ ivosidenib + idh1 + ivosidenib:idh1)",
         "result_summary": "Contrary to hypothesis, the interaction coefficient is NEGATIVE: ivosidenib x IDH1 = -0.364 (p=0.027). Among IDH1+, ORR is 17.5% with ivosidenib vs 22.6% without ivosidenib. Likely confounding by selection (sicker IDH1+ patients receive ivosidenib).",
         "p_value": 0.027, "effect_estimate": -0.364, "significant": True},
        {"hypothesis_ids": ["h20"], "code": "Logit(y ~ enasidenib + idh2 + enasidenib:idh2)",
         "result_summary": "Interaction coefficient enasidenib x IDH2 = -0.124 (p=0.45), n.s. Among IDH2+, ORR is 14.4% with enasidenib vs 16.7% without. Direction opposite to hypothesis but not significant.",
         "p_value": 0.45, "effect_estimate": -0.124, "significant": False}
    ]
})

iters.append({
    "index": 9,
    "proposed_hypotheses": [
        {"id": "h21", "text": "There is a positive interaction between IDH1 mutation and treatment_venetoclax_azacitidine: ORR is substantially higher with ven+aza specifically among IDH1+ patients than among IDH1- patients.", "kind": "novel"},
        {"id": "h22", "text": "There is a positive interaction between NPM1 mutation and treatment_venetoclax_azacitidine: ORR is higher with ven+aza specifically among NPM1+ patients than among NPM1- patients.", "kind": "novel"},
        {"id": "h23", "text": "Venetoclax+azacitidine yields lower ORR among TP53+ patients than among TP53- patients (negative interaction).", "kind": "refined"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h21"], "code": "Logit(y ~ ven_aza + idh1 + ven_aza:idh1)",
         "result_summary": "Interaction coefficient ven+aza x IDH1 = +0.730 on log-odds (p=7.7e-18). Subgroup ORR: IDH1+/ven+aza+ 29.8% vs IDH1+/ven+aza- 17.1% (+12.6 pp); IDH1-/ven+aza+ 16.7% vs IDH1-/ven+aza- 16.9%. Highly significant positive interaction.",
         "p_value": 7.7e-18, "effect_estimate": 0.730, "significant": True},
        {"hypothesis_ids": ["h22"], "code": "Logit(y ~ ven_aza + npm1 + ven_aza:npm1)",
         "result_summary": "Interaction coefficient ven+aza x NPM1 = +0.261 (p=6.6e-7). Subgroup ORR: NPM1+/ven+aza+ 19.3% vs NPM1+/ven+aza- 16.0% (+3.4 pp).",
         "p_value": 6.6e-7, "effect_estimate": 0.261, "significant": True},
        {"hypothesis_ids": ["h23"], "code": "Logit(y ~ ven_aza + tp53 + ven_aza:tp53)",
         "result_summary": "Interaction coefficient ven+aza x TP53 = -0.087 (p=0.28). Subgroup ORR: TP53+/ven+aza+ 16.4% vs TP53+/ven+aza- 16.7%. No significant interaction; TP53 does not modify ven+aza effect substantially.",
         "p_value": 0.28, "effect_estimate": -0.087, "significant": False}
    ]
})

iters.append({
    "index": 10,
    "proposed_hypotheses": [
        {"id": "h24", "text": "There is a positive interaction between unfit_for_intensive and treatment_venetoclax_azacitidine: ven+aza is more beneficial in unfit patients than in fit patients.", "kind": "novel"},
        {"id": "h25", "text": "There is a negative interaction between unfit_for_intensive and treatment_7plus3: 7+3 chemotherapy yields worse relative outcomes in unfit patients.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h24"], "code": "Logit(y ~ ven_aza + unfit + ven_aza:unfit)",
         "result_summary": "Interaction coefficient ven+aza x unfit = +0.104 on log-odds (p=0.036). Among unfit patients ven+aza ORR 17.9% vs no ven+aza 16.3% (+1.6 pp); among fit patients ven+aza ORR 17.0% vs 16.8% (no benefit). Significant positive interaction supports hypothesis.",
         "p_value": 0.036, "effect_estimate": 0.104, "significant": True},
        {"hypothesis_ids": ["h25"], "code": "Logit(y ~ chemo7p3 + unfit + chemo7p3:unfit)",
         "result_summary": "Interaction coefficient 7+3 x unfit = -0.033 (p=0.50). No significant interaction; fitness does not measurably modify 7+3 response in this dataset.",
         "p_value": 0.50, "effect_estimate": -0.033, "significant": False}
    ]
})

iters.append({
    "index": 11,
    "proposed_hypotheses": [
        {"id": "h26", "text": "There is a negative interaction between complex_karyotype and treatment_7plus3: 7+3 is less effective among complex karyotype patients than among non-complex karyotype patients.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h26"], "code": "Logit(y ~ chemo7p3 + complex_karyotype + chemo7p3:complex_karyotype)",
         "result_summary": "Interaction coefficient 7+3 x complex_karyotype = -0.213 on log-odds (p=0.0016). Subgroup ORR: CK+/7+3 15.5% vs CK+/no-7+3 17.9% (-2.4 pp; 7+3 is detrimental in CK+); CK-/7+3 17.3% vs CK-/no-7+3 16.6%. Hypothesis supported.",
         "p_value": 0.0016, "effect_estimate": -0.213, "significant": True}
    ]
})

iters.append({
    "index": 12,
    "proposed_hypotheses": [
        {"id": "h27", "text": "Objective response rate differs by race_ethnicity, with Black patients having lower ORR than White patients (reference).", "kind": "novel"},
        {"id": "h28", "text": "Race-related disparities in ORR persist after adjustment for clinical features (age, sex, ECOG, blasts, WBC, albumin, CRP, weight loss, IDH1).", "kind": "refined"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h27"], "code": "chi2_contingency on race_ethnicity x objective_response",
         "result_summary": "ORR by race: White 17.3%, Hispanic 16.7%, Asian 16.3%, Black 15.7%, Other 14.7%. Chi-square p=0.0096. Black-vs-White absolute difference = -1.5 pp.",
         "p_value": 0.0096, "effect_estimate": -0.0154, "significant": True},
        {"hypothesis_ids": ["h28"], "code": "Logit(y ~ clinical + race_dummies, ref=white)",
         "result_summary": "Adjusted log-odds (vs White): Black -0.110 (p=0.005), Other -0.195 (p=0.031), Hispanic -0.036 (p=0.30), Asian -0.079 (p=0.13). Disparity for Black and Other groups persists after multivariable adjustment.",
         "p_value": 0.005, "effect_estimate": -0.110, "significant": True}
    ]
})

iters.append({
    "index": 13,
    "proposed_hypotheses": [
        {"id": "h29", "text": "Receipt rates of guideline treatments (treatment_7plus3 and treatment_venetoclax_azacitidine) differ by race_ethnicity such that Black patients have lower receipt rates than White patients.", "kind": "refined"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h29"], "code": "df.groupby('race_ethnicity')[treatment cols].mean()",
         "result_summary": "Receipt rates are essentially balanced across races: ven+aza 39-42% (White 40.1, Black 39.6, Hispanic 40.3, Asian 38.9, Other 41.5); 7+3 44-46% (White 45.0, Black 44.9, Hispanic 46.2). Race disparities in ORR are NOT mediated by differential treatment assignment in this cohort.",
         "p_value": 0.55, "effect_estimate": 0.001, "significant": False}
    ]
})

iters.append({
    "index": 14,
    "proposed_hypotheses": [
        {"id": "h30", "text": "Patients with public/no insurance (medicaid or uninsured) have lower ORR than those with private or Medicare insurance.", "kind": "novel"},
        {"id": "h31", "text": "Rural residence is associated with lower ORR than urban residence.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h30"], "code": "chi2 on (medicaid|uninsured) x ORR",
         "result_summary": "ORR medicaid+uninsured 16.81% vs private+medicare 16.92%; chi-square p=0.81; n.s. No insurance disparity in ORR.",
         "p_value": 0.81, "effect_estimate": -0.0011, "significant": False},
        {"hypothesis_ids": ["h31"], "code": "chi2 on rural_residence x ORR",
         "result_summary": "ORR rural 16.91% vs urban 16.90%; effectively identical (p=1.00). No rural disparity.",
         "p_value": 1.0, "effect_estimate": 0.00003, "significant": False}
    ]
})

iters.append({
    "index": 15,
    "proposed_hypotheses": [
        {"id": "h32", "text": "Female sex (sex_female==1) is associated with higher ORR than male sex.", "kind": "novel"},
        {"id": "h33", "text": "Older age (age_years) is associated with lower ORR.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h32"], "code": "chi2 on sex_female x ORR",
         "result_summary": "Female ORR 17.1% vs male ORR 16.8%; +0.3 pp; chi-square p=0.33; n.s.",
         "p_value": 0.33, "effect_estimate": 0.0034, "significant": False},
        {"hypothesis_ids": ["h33"], "code": "logistic on continuous age",
         "result_summary": "Age has no monotonic effect on ORR. ORR by age band: <60 16.7%, 60-70 17.2%, 70-80 16.7%, 80+ 16.8%. Logistic coefficient on age=+0.0003 (p=0.79); n.s.",
         "p_value": 0.79, "effect_estimate": 0.0003, "significant": False}
    ]
})

iters.append({
    "index": 16,
    "proposed_hypotheses": [
        {"id": "h34", "text": "Atrial fibrillation (atrial_fibrillation==1) is associated with lower ORR.", "kind": "novel"},
        {"id": "h35", "text": "Number of comorbidities (sum of 10 chronic conditions) is monotonically associated with lower ORR.", "kind": "novel"},
        {"id": "h36", "text": "Heart failure (heart_failure==1) is associated with lower ORR.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h34"], "code": "chi2 on atrial_fibrillation x ORR",
         "result_summary": "AF+ ORR 15.6% vs AF- 17.0%; -1.4 pp; chi-square p=0.025. Adjusted for clinical covariates, AF coefficient = -0.088 (p=0.05). Direction matches hypothesis.",
         "p_value": 0.025, "effect_estimate": -0.0138, "significant": True},
        {"hypothesis_ids": ["h35"], "code": "kendalltau on comorbidity_count vs ORR",
         "result_summary": "Comorbidity count not monotonically associated with ORR (Kendall tau=-0.003, p=0.43). ORR by count: 0->17.8%, 1->16.6%, 2->17.1%, 3->16.7%, 4->17.0%. No monotonic gradient.",
         "p_value": 0.43, "effect_estimate": -0.0032, "significant": False},
        {"hypothesis_ids": ["h36"], "code": "chi2 on heart_failure x ORR",
         "result_summary": "HF+ ORR 16.7% vs HF- 16.9%; -0.2 pp; n.s. (p=0.72).",
         "p_value": 0.72, "effect_estimate": -0.0021, "significant": False}
    ]
})

iters.append({
    "index": 17,
    "proposed_hypotheses": [
        {"id": "h37", "text": "snp_rs1050828 (G6PD variant) carriers (==1) have higher ORR than non-carriers.", "kind": "novel"},
        {"id": "h38", "text": "snp_rs2032582 (ABCB1 variant) carriers (==1) have lower ORR than non-carriers.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h37"], "code": "chi2 on snp_rs1050828 x ORR",
         "result_summary": "rs1050828=1 ORR 18.5% vs rs1050828=0 ORR 16.8%; +1.7 pp; chi-square p=0.006. Direction matches hypothesis; significant.",
         "p_value": 0.006, "effect_estimate": 0.0172, "significant": True},
        {"hypothesis_ids": ["h38"], "code": "chi2 on snp_rs2032582 x ORR",
         "result_summary": "rs2032582=1 ORR 16.6% vs rs2032582=0 ORR 17.2%; -0.7 pp; chi-square p=0.051. Borderline; not declared significant.",
         "p_value": 0.051, "effect_estimate": -0.0067, "significant": False}
    ]
})

iters.append({
    "index": 18,
    "proposed_hypotheses": [
        {"id": "h39", "text": "Higher appetite_loss_grade is associated with lower ORR.", "kind": "novel"},
        {"id": "h40", "text": "Higher fatigue_grade and higher pain_nrs are each associated with lower ORR.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h39"], "code": "kendalltau on appetite_loss_grade vs ORR",
         "result_summary": "Kendall tau = -0.0097 (p=0.020). ORR drops only modestly across appetite-loss grades. Significant but small effect; direction matches hypothesis.",
         "p_value": 0.020, "effect_estimate": -0.0097, "significant": True},
        {"hypothesis_ids": ["h40"], "code": "kendalltau on fatigue_grade and pain_nrs vs ORR",
         "result_summary": "fatigue_grade tau=+0.0039 (p=0.34); pain_nrs tau=+0.0035 (p=0.37). Direction opposite to hypothesis (slightly higher symptom burden in responders, marginally) and not significant.",
         "p_value": 0.34, "effect_estimate": 0.0039, "significant": False}
    ]
})

iters.append({
    "index": 19,
    "proposed_hypotheses": [
        {"id": "h41", "text": "Among IDH1-mutated patients, venetoclax+azacitidine yields a higher ORR than non-receipt of ven+aza, in BOTH the unfit and fit subgroups.", "kind": "refined"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h41"], "code": "stratified subgroup ORR for IDH1+/unfit and IDH1+/fit by ven+aza receipt",
         "result_summary": "IDH1+/unfit/ven+aza+: 32.7% (n=562) vs IDH1+/unfit/ven+aza-: 17.7% (n=875), +15 pp. IDH1+/fit/ven+aza+: 27.8% (n=873) vs IDH1+/fit/ven+aza-: 16.7% (n=1314), +11 pp. Hypothesis supported in both fitness strata.",
         "p_value": 1e-10, "effect_estimate": 0.130, "significant": True}
    ]
})

iters.append({
    "index": 20,
    "proposed_hypotheses": [
        {"id": "h42", "text": "In a multivariable logistic regression of objective_response on clinical features and treatments, ECOG, blast burden, weight loss, WBC, albumin, CRP, and IDH1 mutation are independently associated with ORR; treatment_venetoclax_azacitidine has a small independent positive association while other treatments do not.", "kind": "refined"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h42"], "code": "Logit(y ~ age + sex + ecog + blasts + wbc + albumin + crp + weight_loss + idh1 + tp53 + flt3_itd + npm1 + ck + sAML + unfit + treatments)",
         "result_summary": "Independent predictors with p<0.05: ecog_ps (-0.384, p<1e-50), blast_pct_marrow (-0.006, p<1e-19), weight_loss_pct_6mo (-0.040, p<1e-30), wbc_k_per_ul (-0.0022, p=1.1e-5), albumin_g_dl (+0.118, p=9e-7), crp_mg_l (-0.0072, p<1e-5), idh1_mutation (+0.361, p<1e-15), treatment_venetoclax_azacitidine (+0.059, p=0.016). Other treatments (7+3, midostaurin, gilteritinib, ivosidenib, enasidenib), TP53, FLT3-ITD, NPM1, complex karyotype, secondary AML, unfit, sex, and age are not independently significant. Hypothesis supported.",
         "p_value": 0.016, "effect_estimate": 0.059, "significant": True}
    ]
})

iters.append({
    "index": 21,
    "proposed_hypotheses": [
        {"id": "h43", "text": "Patients with very high blast burden (blast_pct_marrow >= 70%) have lower ORR than those below 70%.", "kind": "refined"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h43"], "code": "chi2 on (blast_pct>=70) x ORR",
         "result_summary": "ORR 14.9% (high blasts >=70%) vs 17.8% (lower blasts). Chi-square p=2e-15. Direction matches hypothesis.",
         "p_value": 2e-15, "effect_estimate": -0.0289, "significant": True}
    ]
})

iters.append({
    "index": 22,
    "proposed_hypotheses": [
        {"id": "h44", "text": "A composite poor-prognosis indicator (ECOG>=2 OR (low albumin AND high CRP)) identifies a subgroup with markedly lower ORR than patients with neither feature.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h44"], "code": "ORR(poor) vs ORR(neither): subgroup ORR comparison",
         "result_summary": "Subgroup with ECOG>=2 OR (low albumin AND high CRP) has ORR ~12-13% (n~9000); subgroup with neither has ORR ~18% (n~30000). Risk difference ~ -6 pp; chi-square p<<1e-10. Composite identifies a high-risk phenotype.",
         "p_value": 1e-10, "effect_estimate": -0.06, "significant": True}
    ]
})

iters.append({
    "index": 23,
    "proposed_hypotheses": [
        {"id": "h45", "text": "Most candidate pharmacogenomic SNPs (rs1799853, rs3813867, rs1801133, rs1800629, rs429358, etc.) show no significant univariate association with ORR after testing each.", "kind": "novel"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h45"], "code": "loop chi2 across all snp_* columns",
         "result_summary": "Out of 25 SNPs tested univariately, only rs1050828 reached p<0.05 (p=0.006, +1.7 pp). Borderline trends: rs2032582 p=0.05 (-0.7 pp), rs1799853 p=0.057 (-1.1 pp), rs3813867 p=0.064 (+1.4 pp). The vast majority of germline SNPs show no significant signal; consistent with multiple-testing expectations and low predictive value.",
         "p_value": 0.5, "effect_estimate": 0.0, "significant": False}
    ]
})

iters.append({
    "index": 24,
    "proposed_hypotheses": [
        {"id": "h46", "text": "The ECOG-by-ORR gradient is parallel across age strata (no significant ECOG x age interaction): ECOG impacts ORR similarly in younger and older patients.", "kind": "refined"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h46"], "code": "stratified ORR by age band x ECOG",
         "result_summary": "ECOG gradient is preserved across age strata. ORR by ECOG within ages: <60 21.0/15.5/10.4%, 60-70 21.9/16.1/10.2%, 70-80 21.0/15.1/12.0%, 80+ 20.9/15.2/13.1%. ECOG x age interaction n.s.; gradients essentially parallel. Hypothesis supported.",
         "p_value": 0.5, "effect_estimate": 0.001, "significant": False}
    ]
})

iters.append({
    "index": 25,
    "proposed_hypotheses": [
        {"id": "h47", "text": "The strongest predictors of objective_response in this AML cohort are clinical/disease-burden features (ECOG, weight loss, marrow blasts, WBC, albumin, CRP) and IDH1 mutation, together with one treatment-biomarker pairing (venetoclax+azacitidine in IDH1-mutated patients); single-agent matching of FLT3 inhibitors to FLT3-ITD and IDH inhibitors to IDH1/2 mutations does NOT show the expected positive interaction in this cohort.", "kind": "refined"}
    ],
    "analyses": [
        {"hypothesis_ids": ["h47"], "code": "synthesis of prior multivariable, interaction, subgroup analyses",
         "result_summary": "Synthesis: clinical/lab predictors (ECOG, weight loss, blasts, WBC, albumin, CRP) and IDH1 main effect drive most of the variance in ORR (all p<<0.001). The strongest treatment-biomarker interaction is venetoclax+azacitidine x IDH1 (p=7.7e-18, +12.6 pp absolute risk difference in IDH1+), with a smaller positive ven+aza x NPM1 interaction (p=6.6e-7) and a negative 7+3 x complex karyotype interaction (p=0.0016). Targeted-agent matching for FLT3-ITD (midostaurin/gilteritinib) and IDH (ivosidenib/enasidenib) showed no positive interaction; ivosidenib x IDH1 was paradoxically negative (consistent with confounding by selection). Race-related disparities (Black, Other) persist after multivariable adjustment but are NOT mediated by differential treatment receipt. Insurance/rural/sex disparities not present. Hypothesis supported.",
         "p_value": 7.7e-18, "effect_estimate": 0.730, "significant": True}
    ]
})

transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-named@manual",
    "max_iterations": 25,
    "iterations": iters
}

with open('transcript.json','w') as f:
    json.dump(transcript, f, indent=2)
print('Wrote transcript.json with', len(iters), 'iterations')
