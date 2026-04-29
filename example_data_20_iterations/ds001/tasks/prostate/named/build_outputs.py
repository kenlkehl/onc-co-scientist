"""Build transcript.json and analysis_summary.txt from _agent_records.json."""
import json
import math
from collections import defaultdict, OrderedDict

with open("_agent_records.json") as f:
    R = json.load(f)

# Group by iteration
by_iter = defaultdict(list)
for r in R:
    by_iter[r["iter"]].append(r)

iterations = []
for it in sorted(by_iter):
    rs = by_iter[it]
    proposed = []
    analyses = []
    seen_hids = set()
    for r in rs:
        if r["hid"] not in seen_hids:
            proposed.append({
                "id": r["hid"],
                "text": r["text"],
                "kind": r.get("kind", "novel"),
            })
            seen_hids.add(r["hid"])
        ar = {
            "hypothesis_ids": [r["hid"]],
            "code": r["code"],
            "result_summary": r["summary"],
        }
        if r["p"] is not None and not (isinstance(r["p"], float) and math.isnan(r["p"])):
            ar["p_value"] = r["p"]
        if r["eff"] is not None and not (isinstance(r["eff"], float) and math.isnan(r["eff"])):
            ar["effect_estimate"] = r["eff"]
        ar["significant"] = bool(r["sig"])
        analyses.append(ar)
    iterations.append({
        "index": it,
        "proposed_hypotheses": proposed,
        "analyses": analyses,
    })

transcript = {
    "dataset_id": "ds001_prostate",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-agent@manual-iter-2026-04-28",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

print(f"Wrote transcript.json: {len(iterations)} iterations, "
      f"{sum(len(it['proposed_hypotheses']) for it in iterations)} hypotheses, "
      f"{sum(len(it['analyses']) for it in iterations)} analyses")

# ====================== Build analysis_summary.txt ======================

# Helper: find a record by hid
RH = {r["hid"]: r for r in R}

def fmt(r):
    return f"effect={r['eff']:+.4g}, p={r['p']:.3g}{' (sig)' if r['sig'] else ''}"

lines = []
lines.append("ds001_prostate — iterative hypothesis-testing analysis (25 iterations)")
lines.append("=" * 78)
lines.append("")
lines.append("Cohort: 50,000 patients (all male; sex_female=0). Outcome: pfs_months "
             "(progression-free survival, mean=3.74, sd=2.02). The cohort has the typical "
             "advanced/metastatic prostate cancer mix: 55% mCRPC, 60% PSMA-high, 20% "
             "AR-V7+, 10% BRCA2-mutated, 3% MSI-high. Six treatments are recorded "
             "(enzalutamide, abiraterone, docetaxel, olaparib, lu177-PSMA, pembrolizumab).")
lines.append("")
lines.append("ITERATION-BY-ITERATION SYNTHESIS")
lines.append("-" * 78)
lines.append("")

# ---- Iter 1 ----
lines.append("Iteration 1 — Treatment main effects on PFS (unadjusted).")
for tx in ["enzalutamide", "abiraterone", "docetaxel", "olaparib", "lu177_psma", "pembrolizumab"]:
    r = RH[f"h1_{tx}"]
    lines.append(f"  • treatment_{tx}: {fmt(r)}.")
lines.append("  Only treatment_olaparib showed a positive crude effect (~+0.09 months, p=0.003); "
             "the others were small and non-significant. This already hints at "
             "biomarker-driven heterogeneity rather than uniform benefit.")
lines.append("")

# ---- Iter 2 ----
lines.append("Iteration 2 — Pre-specified predictive biomarker × treatment interactions.")
lines.append("  Hypotheses encoded standard prostate oncology pharmacology: olaparib should help "
             "BRCA2+; pembrolizumab should help MSI-high; lu177-PSMA should help PSMA-high; "
             "AR-V7 should confer resistance to enzalutamide/abiraterone.")
for hid in ["h2_olaparib_brca2", "h2_pembro_msi", "h2_lu177_psma",
            "h2_enza_arv7", "h2_abi_arv7"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  Only the olaparib × BRCA2 interaction was strongly supported "
             "(coef=+1.62, p~4e-61). The MSI/PSMA/AR-V7 hypotheses were NOT supported in this "
             "cohort despite biological plausibility.")
lines.append("")

# ---- Iter 3 ----
lines.append("Iteration 3 — Marker / disease-burden main effects.")
for hid in ["h3_brca2", "h3_arv7", "h3_msi", "h3_psma", "h3_mcrpc",
            "h3_vismets", "h3_livmets", "h3_bonemets", "h3_admets",
            "h3_pleff", "h3_pereff"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  mcrpc=1 was the dominant adverse marker (-0.52 months, p~1e-181). "
             "BRCA2+ was paradoxically associated with LONGER PFS (+0.10 months, p=0.002), "
             "almost certainly because BRCA2+ patients differentially benefit from olaparib. "
             "Visceral metastases were modestly adverse. Liver/bone/adrenal metastasis flags "
             "and effusions did not show isolated effects in this aggregate cohort.")
lines.append("")

# ---- Iter 4 ----
lines.append("Iteration 4 — Continuous prognostic main effects (univariate OLS).")
key4 = ["h4_age", "h4_ecog", "h4_psa", "h4_gleason", "h4_alb", "h4_ldh",
        "h4_hgb", "h4_alp", "h4_crp", "h4_nlr", "h4_wtloss", "h4_pain",
        "h4_fatigue", "h4_appetite", "h4_dyspnea", "h4_cea", "h4_ca", "h4_creat"]
for hid in key4:
    r = RH[hid]
    lines.append(f"  • {hid} ({r['hid'][3:]}): {fmt(r)}.")
lines.append("  Strongly supported and clinically expected: ECOG (β=-1.16/unit), "
             "albumin (+0.49/g/dL), PSA (β=-0.003/ng/mL), weight-loss (β=-0.075/%), "
             "LDH (β=-4e-04/U/L) and ALP (β=-5e-04/U/L). Age was associated with LONGER PFS "
             "(β=+0.17/yr, p~0); this is plausibly because older patients in this cohort have "
             "more indolent disease or shorter expected exposure horizons, and the effect "
             "reverses when adjusting for ECOG/albumin (see iteration 14). Symptom grades "
             "(pain, fatigue, dyspnea, appetite-loss) were not associated with PFS univariately, "
             "suggesting they are not independent prognostic markers in this dataset.")
lines.append("")

# ---- Iter 5 ----
lines.append("Iteration 5 — Comorbidity main effects.")
for hid in ["h5_dm", "h5_htn", "h5_ckd", "h5_hf", "h5_cad", "h5_copd",
            "h5_afib", "h5_vte", "h5_autoimmune", "h5_hcv", "h5_hiv",
            "h5_pm", "h5_depanx", "h5_ild"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  No comorbidity flag (DM, HTN, CKD, HF, CAD, COPD, AF, VTE, autoimmune, HCV, HIV, "
             "prior malignancy, depression/anxiety, ILD) showed a univariate association with PFS. "
             "In this synthetic cohort, comorbidity is not driving the outcome.")
lines.append("")

# ---- Iter 6 ----
lines.append("Iteration 6 — SDOH / demographics.")
for hid in ["h6_race_black", "h6_race_hispanic", "h6_race_asian", "h6_race_other",
            "h6_ins_medicare", "h6_ins_medicaid", "h6_ins_uninsured",
            "h6_rural", "h6_smoking", "h6_education"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  No association of race_ethnicity, insurance_type, rural_residence, smoking, "
             "or education with PFS — i.e., no measurable disparity signal in this cohort.")
lines.append("")

# ---- Iter 7 ----
lines.append("Iteration 7 — Prior treatment exposure.")
for hid in ["h7_lines", "h7_ysd", "h7_prior_chemo", "h7_prior_rad",
            "h7_prior_surg", "h7_prior_io", "h7_prior_tt"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  Prior-line counts and prior-modality flags were not significantly associated with "
             "PFS univariately, suggesting that within-cohort treatment burden is balanced.")
lines.append("")

# ---- Iter 8 ----
lines.append("Iteration 8 — SNP main-effect screen (25 SNPs).")
sig_snps = [r for r in R if r["iter"]==8 and r["sig"]]
lines.append(f"  Of 25 candidate SNPs scanned, {len(sig_snps)} were significant at p<0.05 "
             "(i.e., consistent with chance at α=0.05 with no multiple-testing correction). "
             f"The single nominally-significant SNP was: ")
for r in sig_snps:
    lines.append(f"  • {r['hid']}: {fmt(r)}.")
lines.append("  No SNP had a strong (p<1e-3) effect on PFS, so we did not pursue SNP × treatment "
             "interactions in this cohort.")
lines.append("")

# ---- Iter 9 ----
lines.append("Iteration 9 — Tumor co-mutation main effects (lung-cancer-style genes that may "
             "appear in this prostate cohort).")
for hid in ["h9_tp53", "h9_pten", "h9_cdkn2a", "h9_pik3ca", "h9_her2",
            "h9_braf", "h9_fgfr", "h9_met", "h9_ret", "h9_ros1",
            "h9_ntrk", "h9_nrg1", "h9_keap1"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  None of the alternative driver alterations (tp53, pten, cdkn2a, pik3ca, her2, "
             "braf, fgfr, met-ex14, ret, ros1, ntrk, nrg1, keap1) showed a univariate association "
             "with PFS, confirming this is a prostate-specific cohort where prostate biomarkers "
             "(BRCA2, AR-V7, MSI, PSMA, mCRPC) carry the signal.")
lines.append("")

# ---- Iter 10 ----
lines.append("Iteration 10 — Secondary labs / vitals.")
sig10 = [r for r in R if r["iter"]==10 and r["sig"]]
lines.append("  Significant univariate associations: " +
             ", ".join(f"{r['hid'][4:]} ({fmt(r)})" for r in sig10) + ".")
lines.append("  Most secondary labs and vitals (AST/ALT, BUN, Na, K, platelets, WBC/ANC/ALC, "
             "CA-125, TSH, INR, BMI, BP, HR, SpO2) were not associated with PFS, so the "
             "prognostic signal concentrates in the iter-4 set (ECOG, albumin, PSA, LDH, weight-loss, ALP).")
lines.append("")

# ---- Iter 11 ----
lines.append("Iteration 11 — ECOG × treatment interactions.")
for hid in ["h11_enzalutamide_ecog", "h11_abiraterone_ecog", "h11_docetaxel_ecog",
            "h11_olaparib_ecog", "h11_lu177_psma_ecog", "h11_pembrolizumab_ecog"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  No treatment showed an ECOG-modified PFS effect — including docetaxel, where "
             "we hypothesized that less-fit patients might tolerate chemo worse. ECOG is "
             "prognostic but not predictive in this cohort.")
lines.append("")

# ---- Iter 12 ----
lines.append("Iteration 12 — visceral_mets × treatment interactions.")
for hid in ["h12_enzalutamide_visceral", "h12_abiraterone_visceral", "h12_docetaxel_visceral",
            "h12_olaparib_visceral", "h12_lu177_psma_visceral", "h12_pembrolizumab_visceral"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  Only olaparib × visceral_mets reached p=0.017 (β=-0.18); marginal. No clear "
             "evidence that any treatment is more or less effective in visceral disease.")
lines.append("")

# ---- Iter 13 ----
lines.append("Iteration 13 — mcrpc × treatment interactions.")
for hid in ["h13_enzalutamide_mcrpc", "h13_abiraterone_mcrpc", "h13_docetaxel_mcrpc",
            "h13_olaparib_mcrpc", "h13_lu177_psma_mcrpc", "h13_pembrolizumab_mcrpc"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  Marginal docetaxel × mcrpc (β=+0.08, p=0.04). No strong predictive signal by mCRPC.")
lines.append("")

# ---- Iter 14 ----
lines.append("Iteration 14 — Adjusted multivariable model "
             "(prognostic + biomarker + treatment terms; R²≈0.95).")
sig14 = [r for r in R if r["iter"]==14 and r["sig"]]
for r in sig14:
    lines.append(f"  • {r['hid']}: {fmt(r)}.")
lines.append("  In the adjusted model: ECOG (β=-1.15), albumin (+0.48), LDH, PSA, "
             "weight-loss, mCRPC, visceral_mets, and CRP (small +ve coef) remain independently "
             "prognostic. brca2_mutation (+0.15) and treatment_olaparib (+0.15) are independently "
             "associated with longer PFS, consistent with a strong predictive interaction (see "
             "iteration 15). After adjustment, the AR-V7, MSI, PSMA-high markers and the "
             "non-olaparib treatments did not retain independent associations with PFS.")
lines.append("")

# ---- Iter 15 ----
lines.append("Iteration 15 — Refined: predictive interactions in adjusted models.")
for hid in ["h15_olaparib_brca2_adj", "h15_pembro_msi_adj", "h15_lu177_psma_adj",
            "h15_enza_arv7_adj", "h15_abi_arv7_adj"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  After multivariable adjustment, ONLY the olaparib × BRCA2 interaction remained "
             "(β=+1.46, p~0). The MSI/pembrolizumab, PSMA/lu177-PSMA, and AR-V7/AR-pathway "
             "predictive hypotheses were definitively NOT supported.")
lines.append("")

# ---- Iter 16 ----
lines.append("Iteration 16 — Secondary predictive interactions (exploratory).")
for hid in ["h16_lu177_psa", "h16_doce_visceral", "h16_olap_tp53",
            "h16_pembro_tp53", "h16_enza_tp53", "h16_pembro_arv7"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  No additional predictive interactions emerged — confirms that BRCA2 × olaparib "
             "is the dominant heterogeneous-treatment-effect signal.")
lines.append("")

# ---- Iter 17 ----
lines.append("Iteration 17 — Race × treatment interactions (disparities check).")
for hid in ["h17_enzalutamide_black", "h17_abiraterone_black", "h17_docetaxel_black",
            "h17_olaparib_black", "h17_lu177_psma_black", "h17_pembrolizumab_black"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  No treatment effect is modified by black-race vs other (no evidence of "
             "race-based differential efficacy in this cohort).")
lines.append("")

# ---- Iter 18 ----
lines.append("Iteration 18 — Insurance × treatment interactions.")
for hid in ["h18_olaparib_medicaid", "h18_lu177_psma_medicaid", "h18_pembrolizumab_medicaid"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  No interaction with medicaid status — no evidence of access-related differential "
             "efficacy.")
lines.append("")

# ---- Iter 19 ----
lines.append("Iteration 19 — Stratified subgroup confirmation of predictive markers.")
for hid in ["h19_olap_brca2_1", "h19_olap_brca2_0",
            "h19_pembro_msi_1", "h19_pembro_msi_0",
            "h19_lu177_psma_1", "h19_lu177_psma_0",
            "h19_enza_arv7_1", "h19_enza_arv7_0"]:
    if hid in RH:
        r = RH[hid]
        lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  Olaparib added ~+1.55 mo PFS in BRCA2+ patients (p~4e-27) but had a small "
             "NEGATIVE effect (-0.07 mo, p=0.02) in BRCA2- patients — strongly confirming the "
             "predictive interaction. Pembrolizumab/MSI, lu177/PSMA, enzalutamide/AR-V7 "
             "stratified analyses showed no subgroup benefit.")
lines.append("")

# ---- Iter 20 ----
lines.append("Iteration 20 — Symptom burden composite.")
r = RH["h20_sympburden"]
lines.append(f"  • {r['hid']}: {fmt(r)}.")
lines.append("  No association of composite symptom burden with PFS, consistent with "
             "iter 4 individual symptom-grade results.")
lines.append("")

# ---- Iter 21 ----
lines.append("Iteration 21 — Inflammation × treatment interactions.")
for hid in ["h21_pembro_nlr", "h21_doce_crp"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  Neither high NLR (a known marker of poor immunotherapy response) nor high CRP "
             "modified treatment effect in this cohort.")
lines.append("")

# ---- Iter 22 ----
lines.append("Iteration 22 — Age × treatment interactions.")
for hid in ["h22_docetaxel_age", "h22_enzalutamide_age", "h22_abiraterone_age",
            "h22_olaparib_age", "h22_lu177_psma_age", "h22_pembrolizumab_age"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  No age-modified treatment effect — including docetaxel, where we expected "
             "older patients to derive less benefit (or experience harm).")
lines.append("")

# ---- Iter 23 ----
lines.append("Iteration 23 — Albumin × treatment interactions.")
for hid in ["h23_docetaxel_alb", "h23_olaparib_alb", "h23_lu177_psma_alb",
            "h23_pembrolizumab_alb"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  Pembrolizumab × albumin reached p=0.045 (β=+0.17): higher albumin associated "
             "with greater pembrolizumab benefit. Marginal and exploratory.")
lines.append("")

# ---- Iter 24 ----
lines.append("Iteration 24 — Co-mutation interactions on PFS.")
for hid in ["h24_brca2_tp53", "h24_tp53_pten", "h24_arv7_tp53"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  No multiplicative interaction among BRCA2/TP53/PTEN/AR-V7 — these mutations "
             "act independently (or with no detectable synergistic effect).")
lines.append("")

# ---- Iter 25 ----
lines.append("Iteration 25 — Final adjusted model with all four predictive interactions "
             "simultaneously (R²≈0.95).")
for hid in ["h25_olap_brca2_final", "h25_pembro_msi_final", "h25_lu177_psma_final",
            "h25_enza_arv7_final", "h25_abi_arv7_final"]:
    r = RH[hid]
    lines.append(f"  • {hid}: {fmt(r)}.")
lines.append("  In the omnibus model: olaparib × BRCA2 retains its strong positive interaction "
             "(β=+1.44, p~0). The pembrolizumab × MSI, lu177 × PSMA, and the two AR-pathway × "
             "AR-V7 interactions are all small and non-significant — i.e., the dataset does NOT "
             "encode the textbook MSI/PSMA/AR-V7 predictive relationships, even though it does "
             "encode the BRCA2/olaparib relationship.")
lines.append("")

# ============== Overall conclusions ==============
lines.append("OVERALL CONCLUSIONS")
lines.append("-" * 78)
lines.append("")
lines.append("1) Strongly supported predictive biomarker–treatment relationship:")
lines.append("     • BRCA2 mutation predicts a large benefit from olaparib (~+1.5 mo PFS in "
             "BRCA2+; near-zero or slightly negative effect in BRCA2-). Adjusted interaction "
             "β≈+1.44 mo (p~0). This recapitulates the expected PARP-inhibitor pharmacology.")
lines.append("")
lines.append("2) Refuted predictive biomarker–treatment relationships in this cohort:")
lines.append("     • pembrolizumab × MSI-high — no benefit (interaction n.s.).")
lines.append("     • lu177-PSMA × PSMA-high — no benefit (interaction n.s.).")
lines.append("     • enzalutamide × AR-V7 — no resistance signal.")
lines.append("     • abiraterone × AR-V7 — no resistance signal.")
lines.append("   Despite the textbook biology suggesting these interactions should exist, "
             "they are not present in this dataset — only the olaparib/BRCA2 axis is.")
lines.append("")
lines.append("3) Strong independent prognostic factors (multivariable R²≈0.95):")
lines.append("     • ecog_ps (β=-1.15/unit, p~0)")
lines.append("     • albumin_g_dl (β=+0.48/g/dL, p~0)")
lines.append("     • mcrpc (β=-0.38, p~0)")
lines.append("     • psa_ng_ml (β=-0.003/ng/mL, p~0)")
lines.append("     • weight_loss_pct_6mo (β=-0.077/%, p~0)")
lines.append("     • ldh_u_l (β=-5e-04/U/L, p~1e-82)")
lines.append("     • visceral_mets (β=-0.02, p~4e-05)")
lines.append("   These align with the established mCRPC prognostic literature (Halabi nomogram-"
             "type variables).")
lines.append("")
lines.append("4) Null / not-supported in this cohort:")
lines.append("     • Race, insurance, rural-residence, smoking, education — no PFS associations.")
lines.append("     • Comorbidities (DM, HTN, CKD, HF, CAD, COPD, AF, VTE, autoimmune, HCV, HIV, "
             "prior_malignancy, depression/anxiety, ILD) — none associated with PFS.")
lines.append("     • Most laboratory variables outside the prognostic set (AST/ALT, "
             "platelets, WBC, BP, HR, SpO2, BMI, INR) — no PFS association.")
lines.append("     • All non-prostate driver genes (TP53, PTEN, CDKN2A, PIK3CA, HER2, BRAF, "
             "FGFR, MET ex14, RET/ROS1/NTRK/NRG1, KEAP1) — no PFS association.")
lines.append("     • SNP screen — only 1 of 25 SNPs hit p<0.05 (consistent with type-I noise "
             "with no multiple-testing correction).")
lines.append("     • Symptom grades, age × treatment, ECOG × treatment, race/insurance × "
             "treatment, NLR/CRP × treatment, co-mutation interactions — all null.")
lines.append("")
lines.append("5) Direction reversals worth noting:")
lines.append("     • Univariate age effect on PFS was POSITIVE (older → longer PFS, "
             "β=+0.17/yr); the sign flips when adjusting for ECOG/albumin/disease burden, "
             "suggesting age is largely a proxy for selection / disease tempo here.")
lines.append("     • BRCA2+ patients had longer mean PFS in the cohort (univariate +0.10 mo) "
             "because their olaparib responsiveness drags up the subgroup mean — the "
             "interaction analysis (iter 2/15/19/25) is the correct frame.")
lines.append("")
lines.append("6) Practical takeaway:")
lines.append("   In ds001_prostate, treatment heterogeneity is concentrated entirely in the "
             "BRCA2 / olaparib axis. Beyond that, prognosis is driven by performance status, "
             "burden-of-disease labs (PSA, LDH, ALP), nutritional/cachexia markers (albumin, "
             "weight-loss), and the mCRPC / visceral-mets phenotype. Treatments other than "
             "olaparib do not show measurable PFS benefit in their textbook biomarker-defined "
             "subgroups in this dataset.")

with open("analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote analysis_summary.txt: {len(lines)} lines")
