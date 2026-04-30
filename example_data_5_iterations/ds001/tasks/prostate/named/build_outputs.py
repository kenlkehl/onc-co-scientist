"""Assemble transcript.json and analysis_summary.txt from all_results.json."""
import json
from pathlib import Path

with open("all_results.json") as f:
    data = json.load(f)

# Re-run analysis.py first to refresh all_results.json with corrected text.
# (Caller is expected to have run analysis.py before this script.)

transcript = {
    "dataset_id": "ds001_prostate",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-manual@2026-04-26",
    "max_iterations": 10,
    "iterations": data["iterations"],
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

print("Wrote transcript.json")

# Build the narrative summary.
lines = []
lines.append("# ds001_prostate analysis summary\n")
lines.append("Cohort: 50,000 patient records from a commercial EHR-derived oncology dataset")
lines.append("focused on prostate cancer.  All patients are male (sex_female=0 throughout).")
lines.append("Outcome of interest: pfs_months (continuous, mean 3.74, SD 2.02, range 0-14.6).")
lines.append("Across 10 iterations I proposed and tested 51 hypotheses spanning treatment main")
lines.append("effects, classical prognostic factors, biomarkers, treatment x biomarker")
lines.append("interactions, lab values, symptom burden, comorbidities, social determinants,")
lines.append("germline SNPs, and a final multivariable model.\n")

lines.append("## Iteration 1 - Treatment main effects on pfs_months")
lines.append("Of the six treatments tested in unadjusted Welch t-tests, only treatment_olaparib")
lines.append("showed a statistically significant effect: mean pfs_months 3.83 on olaparib vs 3.74")
lines.append("off (diff +0.094 months, p=0.003). treatment_enzalutamide, treatment_abiraterone,")
lines.append("treatment_docetaxel, treatment_lu177_psma, and treatment_pembrolizumab all showed")
lines.append("trivial mean differences (|diff| <= 0.06 months) with p>0.13 - no overall main effect.\n")

lines.append("## Iteration 2 - Classical prognostic factors")
lines.append("Strongest signals in the entire analysis came from this iteration:")
lines.append("  - ecog_ps: beta = -1.16 months per ECOG point (p ~ 0). Largest single effect.")
lines.append("  - mcrpc=1 vs 0: -0.52 months (p = 1.6e-181).")
lines.append("  - log(1+psa_ng_ml): beta = -0.38 months per log-unit (p ~ 0).")
lines.append("  - visceral_mets=1: -0.05 months (p = 0.034) - significant but very small.")
lines.append("  - age_years: beta = +0.174 months per year (p ~ 0).  Direction OPPOSITE")
lines.append("    to my prior; in this cohort older patients have longer recorded PFS")
lines.append("    (likely a synthetic data construct or selection effect).")
lines.append("  - liver_mets, bone_mets, gleason_score: no significant association with pfs_months.\n")

lines.append("## Iteration 3 - Biomarker main effects")
lines.append("Most biomarkers showed no main effect on PFS:")
lines.append("  - brca2_mutation=1: +0.100 months vs wild-type (p = 0.002). Direction OPPOSITE")
lines.append("    to my prior expectation that BRCA2 mutation is a poor prognostic feature;")
lines.append("    here it is associated with longer PFS.")
lines.append("  - ar_v7_positive, msi_high, psma_high, tp53_mutation, pten_loss: all NS.\n")

lines.append("## Iteration 4 - Treatment x biomarker (precision-oncology) interactions")
lines.append("This iteration produced the single most striking interaction in the dataset:")
lines.append("  - treatment_olaparib x brca2_mutation: interaction beta = +1.62 (p = 4.3e-61).")
lines.append("    Olaparib effect on PFS is +1.55 months in BRCA2-mutant patients but -0.07")
lines.append("    months in BRCA2-wild-type patients - a textbook synthetic-lethality pattern.")
lines.append("  - treatment_enzalutamide x ar_v7_positive: NS (p=0.96).")
lines.append("  - treatment_abiraterone x ar_v7_positive: NS (p=0.33).")
lines.append("  - treatment_pembrolizumab x msi_high: NS (p=0.15) and trended in the OPPOSITE")
lines.append("    direction (pembro effect -0.27 mo in MSI-high vs +0.075 in MSI-low). The")
lines.append("    expected immunotherapy/MSI signal is not present in this cohort.")
lines.append("  - treatment_lu177_psma x psma_high: NS (p=0.74). The expected theranostic")
lines.append("    interaction is not present.\n")

lines.append("## Iteration 5 - Lab-based prognostic factors")
lines.append("  - albumin_g_dl: beta = +0.495 mo per g/dl (p = 1.5e-166). Strongly prognostic.")
lines.append("  - ldh_u_l: beta = -4.2e-4 per U/L (p = 0.00018). Significant but very small per-unit.")
lines.append("  - alkaline_phosphatase_u_l: beta = -5.0e-4 per U/L (p = 0.0066).")
lines.append("  - hemoglobin_g_dl: beta = -0.012 mo per g/dl (p = 0.023). Direction OPPOSITE")
lines.append("    to my prior - significant but tiny in magnitude.")
lines.append("  - nlr, crp_mg_l: NS, contrary to the usual inflammation-prognosis literature.\n")

lines.append("## Iteration 6 - Symptom burden and weight loss")
lines.append("  - weight_loss_pct_6mo: beta = -0.075 mo per pct point (p = 9.9e-226).")
lines.append("    Among the strongest prognostic signals after ECOG.")
lines.append("  - fatigue_grade, pain_nrs, dyspnea_grade, appetite_loss_grade: all NS.\n")

lines.append("## Iteration 7 - Comorbidities and prior therapies")
lines.append("All tested factors were NS: chronic_kidney_disease, heart_failure,")
lines.append("diabetes_mellitus, prior_chemotherapy, prior_lines_of_therapy,")
lines.append("years_since_diagnosis. Comorbidity burden does not appear to drive PFS in this cohort.\n")

lines.append("## Iteration 8 - Demographics and social determinants")
lines.append("Race-ethnicity (5 groups), insurance type (4 groups), rural_residence,")
lines.append("smoking_pack_years, education_years, and uninsured-vs-private comparison were")
lines.append("ALL NS. There is no detectable disparity by demographic / socioeconomic factors")
lines.append("in this dataset - notable given how often such effects are observed in real-world")
lines.append("oncology data.\n")

lines.append("## Iteration 9 - Germline SNP screen")
lines.append("Of the 25 SNPs in the dataset, 1 reached uncorrected p<0.05 (snp_rs4986893,")
lines.append("p=0.0145, beta=-0.092). 0 survived Bonferroni correction (threshold ~0.002).")
lines.append("Specific candidates rs1045642 (ABCB1), rs429358 (APOE), and rs1800629 (TNF)")
lines.append("were all NS. No germline-PFS signal in this cohort.\n")

lines.append("## Iteration 10 - Multivariable confirmation and refinements")
lines.append("Joint OLS adjusting for ECOG, mCRPC, visceral mets, log PSA, albumin, hemoglobin,")
lines.append("LDH, NLR, and all six treatments (R^2 = 0.220):")
lines.append("  - ecog_ps remains independently strongly negative (beta=-1.16, p~0). CONFIRMED.")
lines.append("  - albumin_g_dl remains independently positive (beta=+0.49, p=1.7e-205). CONFIRMED.")
lines.append("  - The olaparib x brca2_mutation interaction strengthens after adjustment for")
lines.append("    ECOG and mCRPC (beta=+1.62, p=1.9e-73). CONFIRMED.")
lines.append("  - The pembrolizumab x msi_high interaction is NOT significant after adjustment")
lines.append("    (p=0.14). REFUTED.")
lines.append("  - The unadjusted nominal docetaxel signal is gone after adjustment (p=0.44).")
lines.append("    REFUTED - any apparent docetaxel effect is explained by other variables.")
lines.append("  - Uninsured-vs-private contrast remains NS after disease-state adjustment")
lines.append("    (p=0.72). The dataset shows no insurance disparity in PFS.\n")

lines.append("## Overall conclusions")
lines.append("1. Performance status (ECOG) is by far the dominant prognostic factor for PFS in")
lines.append("   this cohort, followed by serum albumin, mCRPC status, log PSA, weight loss,")
lines.append("   and (paradoxically positive) age.")
lines.append("2. The single robust treatment-biomarker interaction is olaparib x BRCA2 mutation:")
lines.append("   olaparib provides ~1.5 months of PFS benefit in BRCA2-mutant patients and no")
lines.append("   benefit in BRCA2 wild-type. This is highly significant (p < 1e-60), survives")
lines.append("   multivariable adjustment, and matches the well-established mechanism of PARP")
lines.append("   inhibition in homologous-recombination-deficient tumors.")
lines.append("3. Other anticipated precision-oncology interactions (enzalutamide/abiraterone")
lines.append("   x AR-V7, pembrolizumab x MSI-high, Lu177-PSMA x PSMA-high) are NOT supported")
lines.append("   in this cohort.")
lines.append("4. Most laboratory inflammation markers (NLR, CRP), comorbidities, symptom grades")
lines.append("   (fatigue, pain, dyspnea, appetite loss), demographic / socioeconomic factors,")
lines.append("   and germline SNPs show no detectable association with PFS - these are likely")
lines.append("   pure noise variables in this synthetic cohort.")
lines.append("5. Several main effects went in the opposite direction from my clinical prior")
lines.append("   (older age associated with LONGER PFS; BRCA2 mutation main effect POSITIVE;")
lines.append("   higher hemoglobin slightly NEGATIVE) - probably reflecting how the synthetic")
lines.append("   data were generated rather than real biology, but flagged here for completeness.")

with open("analysis_summary.txt","w") as f:
    f.write("\n".join(lines) + "\n")
print("Wrote analysis_summary.txt")
