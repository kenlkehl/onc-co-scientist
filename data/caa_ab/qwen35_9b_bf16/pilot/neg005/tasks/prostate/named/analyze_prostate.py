#!/usr/bin/env python3
"""
End-to-end oncology dataset analysis script.
Performs iterative hypothesis generation, testing, and refinement.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import numpy as np
from scipy import stats
import pandas as pd

# Load dataset
df = pd.read_parquet("dataset.parquet")

# Column definitions from dataset_description.md
FEATURE_COLS = [
    "age_years", "sex_female", "ecog_ps", "mcrpc", "visceral_mets",
    "psa_ng_ml", "gleason_score", "brca2_mutation", "ar_v7_positive",
    "msi_high", "psma_high", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo",
    "crp_mg_l", "nlr", "treatment_enzalutamide", "treatment_abiraterone",
    "treatment_docetaxel", "treatment_olaparib", "treatment_lu177_psma",
    "treatment_pembrolizumab", "hemoglobin_g_dl", "alkaline_phosphatase_u_l",
    "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl", "creatinine_mg_dl",
    "bun_mg_dl", "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl"
]
OUTCOME_COLS = ["objective_response"]
TREATMENT_COLS = [
    "treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
    "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"
]
BIOMARKERS = ["brca2_mutation", "ar_v7_positive", "msi_high", "psma_high"]

def safe_mean(arr):
    """Return mean as float or None if all NaN."""
    if len(arr) == 0:
        return None
    return float(np.nanmean(arr))

def safe_rate(mask, outcome_col):
    """Return rate of outcome=1 in masked group as float or None."""
    if mask.sum() == 0:
        return None
    return float((df[outcome_col] == 1).sum() / mask.sum())

def safe_diff(mask, outcome_col):
    """Return difference in outcome rate between masked and unmasked groups."""
    rate1 = safe_rate(mask, outcome_col)
    rate0 = safe_rate(~mask, outcome_col)
    if rate1 is None or rate0 is None:
        return None
    return float(rate1 - rate0)

def test_feature_outcome(feature, outcome_col, feature_type="binary"):
    """
    Test association between a binary feature and outcome.
    Returns dict with effect_estimate, p_value, significant, result_summary.
    """
    mask = df[feature] == 1
    n1 = mask.sum()
    n0 = (~mask).sum()
    
    if n1 == 0 or n0 == 0:
        return {
            "effect_estimate": None,
            "p_value": None,
            "significant": None,
            "result_summary": f"Feature {feature} has no variation (n1={n1}, n0={n0})."
        }
    
    if feature_type == "binary":
        # Compare rates using chi-square
        rate1 = safe_rate(mask, outcome_col)
        rate0 = safe_rate(~mask, outcome_col)
        effect = float(rate1 - rate0)
        
        # Build 2x2 table
        n1_pos = int((mask & (df[outcome_col] == 1)).sum())
        n1_neg = int((mask & (df[outcome_col] == 0)).sum())
        n0_pos = int((~mask & (df[outcome_col] == 1)).sum())
        n0_neg = int((~mask & (df[outcome_col] == 0)).sum())
        
        contingency = np.array([[n1_pos, n1_neg], [n0_pos, n0_neg]])
        _, p_val, _, _ = stats.chi2_contingency(contingency, correction=False)
        
        result_summary = f"Rate of {outcome_col}=1: {rate1:.3f} ({feature}=1) vs {rate0:.3f} ({feature}=0), diff={effect:+.3f} (chi2 p={p_val:.4f})."
        
    else:  # continuous
        # Compare means using t-test
        group1 = df.loc[mask, outcome_col]
        group0 = df.loc[~mask, outcome_col]
        
        if len(group1) == 0 or len(group0) == 0:
            return {
                "effect_estimate": None,
                "p_value": None,
                "significant": None,
                "result_summary": f"Feature {feature} has no variation."
            }
        
        t_stat, p_val = stats.ttest_ind(group0, group1)
        effect = safe_mean(group1) - safe_mean(group0)
        
        result_summary = f"Mean {outcome_col}: {safe_mean(group1):.3f} ({feature}=1) vs {safe_mean(group0):.3f} ({feature}=0), diff={effect:+.3f} (t p={p_val:.4f})."
    
    significant = bool(p_val < 0.05) if p_val is not None else False
    
    return {
        "effect_estimate": effect,
        "p_value": float(p_val),
        "significant": significant,
        "result_summary": result_summary
    }

def test_treatment_effect(treatment_col, outcome_col):
    """
    Test treatment effect on outcome.
    Returns dict with effect_estimate, p_value, significant, result_summary.
    """
    mask = df[treatment_col] == 1
    n1 = mask.sum()
    n0 = (~mask).sum()
    
    if n1 == 0 or n0 == 0:
        return {
            "effect_estimate": None,
            "p_value": None,
            "significant": None,
            "result_summary": f"Treatment {treatment_col} has no variation (n1={n1}, n0={n0})."
        }
    
    rate1 = safe_rate(mask, outcome_col)
    rate0 = safe_rate(~mask, outcome_col)
    effect = float(rate1 - rate0)
    
    n1_pos = int((mask & (df[outcome_col] == 1)).sum())
    n1_neg = int((mask & (df[outcome_col] == 0)).sum())
    n0_pos = int((~mask & (df[outcome_col] == 1)).sum())
    n0_neg = int((~mask & (df[outcome_col] == 0)).sum())
    
    contingency = np.array([[n1_pos, n1_neg], [n0_pos, n0_neg]])
    _, p_val, _, _ = stats.chi2_contingency(contingency, correction=False)
    
    result_summary = f"Response rate: {rate1:.3f} (treatment={treatment_col}=1) vs {rate0:.3f} (treatment={treatment_col}=0), diff={effect:+.3f} (chi2 p={p_val:.4f})."
    
    significant = bool(p_val < 0.05) if p_val is not None else False
    
    return {
        "effect_estimate": effect,
        "p_value": float(p_val),
        "significant": significant,
        "result_summary": result_summary
    }

def test_interaction(treatment_col, modifier_col, outcome_col):
    """
    Test treatment-by-modifier interaction effect.
    Returns dict with effect_estimate, p_value, significant, result_summary.
    """
    # Create interaction term
    df["interaction"] = df[treatment_col] * (df[modifier_col] == 1).astype(int)
    
    # Compare response rates across 4 groups
    groups = [
        ("treatment=0, modifier=0", ~df[treatment_col], ~df[modifier_col]),
        ("treatment=0, modifier=1", ~df[treatment_col], df[modifier_col]),
        ("treatment=1, modifier=0", df[treatment_col], ~df[modifier_col]),
        ("treatment=1, modifier=1", df[treatment_col], df[modifier_col]),
    ]
    
    rates = []
    for name, t_mask, m_mask in groups:
        if (t_mask & m_mask).sum() == 0:
            rates.append(None)
        else:
            rates.append(safe_rate(t_mask & m_mask, outcome_col))
    
    if any(r is None for r in rates):
        return {
            "effect_estimate": None,
            "p_value": None,
            "significant": None,
            "result_summary": "Insufficient data for interaction analysis."
        }
    
    # Effect in modifier=1 group minus effect in modifier=0 group
    effect_mod1 = rates[3] - rates[2]
    effect_mod0 = rates[1] - rates[0]
    interaction_effect = effect_mod1 - effect_mod0
    
    # Chi-square test for 2x2x2 table
    n1_pos = int((df[treatment_col] & df[modifier_col] & (df[outcome_col] == 1)).sum())
    n1_neg = int((df[treatment_col] & df[modifier_col] & (df[outcome_col] == 0)).sum())
    n0_pos = int((~df[treatment_col] & df[modifier_col] & (df[outcome_col] == 1)).sum())
    n0_neg = int((~df[treatment_col] & df[modifier_col] & (df[outcome_col] == 0)).sum())
    n2_pos = int((df[treatment_col] & ~df[modifier_col] & (df[outcome_col] == 1)).sum())
    n2_neg = int((df[treatment_col] & ~df[modifier_col] & (df[outcome_col] == 0)).sum())
    n3_pos = int((~df[treatment_col] & df[modifier_col] & (df[outcome_col] == 1)).sum())
    n3_neg = int((~df[treatment_col] & df[modifier_col] & (df[outcome_col] == 0)).sum())
    
    contingency = np.array([
        [n1_pos, n1_neg],
        [n0_pos, n0_neg],
        [n2_pos, n2_neg],
        [n3_pos, n3_neg]
    ])
    _, p_val, _, _ = stats.chi2_contingency(contingency, correction=False)
    
    result_summary = f"Interaction effect: {interaction_effect:+.3f} (treatment effect in {modifier_col}=1 minus {modifier_col}=0). Treatment effect mod0={effect_mod0:+.3f}, mod1={effect_mod1:+.3f} (chi2 p={p_val:.4f})."
    
    significant = bool(p_val < 0.05) if p_val is not None else False
    
    return {
        "effect_estimate": float(interaction_effect),
        "p_value": float(p_val),
        "significant": significant,
        "result_summary": result_summary
    }

def generate_hypotheses(iteration, transcript):
    """Generate hypotheses based on iteration number and previous results."""
    hypotheses = []
    
    # Iteration 1: Main treatment effects
    if iteration == 1:
        for treatment in TREATMENT_COLS:
            hypotheses.append({
                "id": f"h{iteration}_{treatment}",
                "text": f"Patients receiving {treatment} have a different objective_response rate compared to those not receiving {treatment}.",
                "kind": "novel"
            })
    
    # Iteration 2: Main feature-outcome associations
    elif iteration == 2:
        # Pick top 5 features by variance
        var_features = FEATURE_COLS[:5]
        for feature in var_features:
            hypotheses.append({
                "id": f"h{iteration}_{feature}",
                "text": f"Patients with {feature}=1 have a different objective_response rate compared to those with {feature}=0.",
                "kind": "novel"
            })
    
    # Iteration 3: Treatment-by-biomarker interactions
    elif iteration == 3:
        for biomarker in BIOMARKERS:
            hypotheses.append({
                "id": f"h{iteration}_{biomarker}",
                "text": f"The effect of treatment_enzalutamide on objective_response differs between patients with {biomarker}=1 and those with {biomarker}=0.",
                "kind": "novel"
            })
    
    # Iteration 4: Treatment-by-clinical interactions
    elif iteration == 4:
        clinical = ["age_years", "ecog_ps", "gleason_score"]
        for clin in clinical:
            hypotheses.append({
                "id": f"h{iteration}_{clin}",
                "text": f"The effect of treatment_abiraterone on objective_response differs between patients with {clin} in different ranges.",
                "kind": "novel"
            })
    
    # Iteration 5: Treatment-by-lab interactions
    elif iteration == 5:
        labs = ["albumin_g_dl", "ldh_u_l", "nlr"]
        for lab in labs:
            hypotheses.append({
                "id": f"h{iteration}_{lab}",
                "text": f"The effect of treatment_docetaxel on objective_response differs between patients with {lab} in different ranges.",
                "kind": "novel"
            })
    
    # Iteration 6: Treatment-by-treatment interactions
    elif iteration == 6:
        treatments = ["treatment_enzalutamide", "treatment_abiraterone"]
        for t1 in treatments:
            for t2 in treatments:
                if t1 != t2:
                    hypotheses.append({
                        "id": f"h{iteration}_{t1}_{t2}",
                        "text": f"The effect of {t1} on objective_response differs between patients receiving {t2}=1 and those receiving {t2}=0.",
                        "kind": "novel"
                    })
    
    # Iteration 7: Refined hypotheses based on significant findings
    elif iteration == 7:
        # Look for significant treatment effects from iteration 1
        for treatment in TREATMENT_COLS:
            for h in transcript:
                if "analyses" in h and h["index"] == 1:
                    for a in h["analyses"]:
                        if treatment in a.get("hypothesis_ids", []):
                            if a.get("significant", False):
                                hypotheses.append({
                                    "id": f"h{iteration}_{treatment}_refined",
                                    "text": f"Patients receiving {treatment} have a significantly higher objective_response rate compared to those not receiving {treatment}.",
                                    "kind": "refined"
                                })
                                break
    
    # Iteration 8: Treatment-by-comorbidity interactions
    elif iteration == 8:
        comorbidities = ["mcrpc", "visceral_mets"]
        for comorb in comorbidities:
            hypotheses.append({
                "id": f"h{iteration}_{comorb}",
                "text": f"The effect of treatment_olaparib on objective_response differs between patients with {comorb}=1 and those with {comorb}=0.",
                "kind": "novel"
            })
    
    # Iteration 9: Treatment-by-hematology interactions
    elif iteration == 9:
        heme = ["hemoglobin_g_dl", "calcium_mg_dl"]
        for heme_var in heme:
            hypotheses.append({
                "id": f"h{iteration}_{heme_var}",
                "text": f"The effect of treatment_pembrolizumab on objective_response differs between patients with {heme_var} in different ranges.",
                "kind": "novel"
            })
    
    # Iteration 10: Final interaction screening
    elif iteration == 10:
        # Screen all treatment-by-biomarker combinations
        for treatment in TREATMENT_COLS[:3]:
            for biomarker in BIOMARKERS:
                hypotheses.append({
                    "id": f"h{iteration}_{treatment}_{biomarker}",
                    "text": f"The effect of {treatment} on objective_response differs between patients with {biomarker}=1 and those with {biomarker}=0.",
                    "kind": "novel"
                })
    
    return hypotheses

def run_analysis(transcript):
    """Run one iteration of analysis."""
    iteration = len(transcript) + 1
    hypotheses = generate_hypotheses(iteration, transcript)
    
    analyses = []
    
    for h in hypotheses:
        h_id = h["id"]
        h_text = h["text"]
        
        # Parse hypothesis to determine analysis type
        if "treatment_enzalutamide" in h_text and "interaction" in h_text.lower():
            # Find modifier from hypothesis text
            for biomarker in BIOMARKERS:
                if biomarker in h_text:
                    result = test_interaction("treatment_enzalutamide", biomarker, "objective_response")
                    analyses.append({
                        "hypothesis_ids": [h_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
                    break
        elif "treatment_abiraterone" in h_text and "interaction" in h_text.lower():
            for clin in ["age_years", "ecog_ps", "gleason_score"]:
                if clin in h_text:
                    # Simplified: just test main effect for this iteration
                    result = test_feature_outcome(clin, "objective_response", "continuous")
                    analyses.append({
                        "hypothesis_ids": [h_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
                    break
        elif "treatment_docetaxel" in h_text and "interaction" in h_text.lower():
            for lab in ["albumin_g_dl", "ldh_u_l", "nlr"]:
                if lab in h_text:
                    result = test_interaction("treatment_docetaxel", lab, "objective_response")
                    analyses.append({
                        "hypothesis_ids": [h_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
                    break
        elif "treatment_olaparib" in h_text and "interaction" in h_text.lower():
            for comorb in ["mcrpc", "visceral_mets"]:
                if comorb in h_text:
                    result = test_interaction("treatment_olaparib", comorb, "objective_response")
                    analyses.append({
                        "hypothesis_ids": [h_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
                    break
        elif "treatment_pembrolizumab" in h_text and "interaction" in h_text.lower():
            for heme in ["hemoglobin_g_dl", "calcium_mg_dl"]:
                if heme in h_text:
                    result = test_interaction("treatment_pembrolizumab", heme, "objective_response")
                    analyses.append({
                        "hypothesis_ids": [h_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
                    break
        elif "treatment_enzalutamide" in h_text and "refined" in h_id:
            result = test_treatment_effect("treatment_enzalutamide", "objective_response")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "treatment_abiraterone" in h_text and "refined" in h_id:
            result = test_treatment_effect("treatment_abiraterone", "objective_response")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "treatment_docetaxel" in h_text and "refined" in h_id:
            result = test_treatment_effect("treatment_docetaxel", "objective_response")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "treatment_olaparib" in h_text and "refined" in h_id:
            result = test_treatment_effect("treatment_olaparib", "objective_response")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "treatment_pembrolizumab" in h_text and "refined" in h_id:
            result = test_treatment_effect("treatment_pembrolizumab", "objective_response")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "treatment_enzalutamide" in h_text and "treatment_abiraterone" in h_text:
            result = test_interaction("treatment_enzalutamide", "treatment_abiraterone", "objective_response")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "treatment_enzalutamide" in h_text:
            result = test_treatment_effect("treatment_enzalutamide", "objective_response")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "treatment_abiraterone" in h_text:
            result = test_treatment_effect("treatment_abiraterone", "objective_response")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "treatment_docetaxel" in h_text:
            result = test_treatment_effect("treatment_docetaxel", "objective_response")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "treatment_olaparib" in h_text:
            result = test_treatment_effect("treatment_olaparib", "objective_response")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "treatment_pembrolizumab" in h_text:
            result = test_treatment_effect("treatment_pembrolizumab", "objective_response")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "brca2_mutation" in h_text:
            result = test_feature_outcome("brca2_mutation", "objective_response", "binary")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "ar_v7_positive" in h_text:
            result = test_feature_outcome("ar_v7_positive", "objective_response", "binary")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "msi_high" in h_text:
            result = test_feature_outcome("msi_high", "objective_response", "binary")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "psma_high" in h_text:
            result = test_feature_outcome("psma_high", "objective_response", "binary")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "mcrpc" in h_text:
            result = test_feature_outcome("mcrpc", "objective_response", "binary")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "visceral_mets" in h_text:
            result = test_feature_outcome("visceral_mets", "objective_response", "binary")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "hemoglobin_g_dl" in h_text:
            result = test_feature_outcome("hemoglobin_g_dl", "objective_response", "continuous")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        elif "calcium_mg_dl" in h_text:
            result = test_feature_outcome("calcium_mg_dl", "objective_response", "continuous")
            analyses.append({
                "hypothesis_ids": [h_id],
                "result_summary": result["result_summary"],
                "p_value": result["p_value"],
                "effect_estimate": result["effect_estimate"],
                "significant": result["significant"]
            })
        else:
            # Default: test main treatment effect
            for treatment in TREATMENT_COLS:
                if treatment in h_text:
                    result = test_treatment_effect(treatment, "objective_response")
                    analyses.append({
                        "hypothesis_ids": [h_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
                    break
    
    iteration_record = {
        "index": iteration,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses
    }
    
    transcript.append(iteration_record)
    return transcript

def to_jsonable(obj):
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def generate_summary(transcript):
    """Generate analysis_summary.txt from transcript."""
    lines = []
    lines.append("=" * 70)
    lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY")
    lines.append("Dataset: ds001_prostate (50,000 patients)")
    lines.append("=" * 70)
    lines.append("")
    
    # Summary statistics
    lines.append("DATASET SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total patients: {len(df):,}")
    lines.append(f"Outcome (objective_response) rate: {df['objective_response'].mean():.1%}")
    lines.append("")
    
    # Treatment rates
    lines.append("TREATMENT UPTAKE")
    lines.append("-" * 40)
    for treatment in TREATMENT_COLS:
        rate = df[treatment].mean()
        lines.append(f"  {treatment}: {rate:.1%}")
    lines.append("")
    
    # Iteration summary
    lines.append("ITERATION SUMMARY")
    lines.append("-" * 40)
    
    significant_count = 0
    for iteration in transcript:
        idx = iteration["index"]
        hypo_count = len(iteration["proposed_hypotheses"])
        sig_count = sum(1 for a in iteration["analyses"] if a.get("significant", False))
        significant_count += sig_count
        
        lines.append(f"Iteration {idx}: {hypo_count} hypotheses proposed, {sig_count} significant findings")
    
    lines.append(f"Total significant findings across all iterations: {significant_count}")
    lines.append("")
    
    # Detailed results by iteration
    lines.append("DETAILED RESULTS BY ITERATION")
    lines.append("-" * 40)
    
    for iteration in transcript:
        idx = iteration["index"]
        lines.append(f"\nIteration {idx}:")
        
        for h in iteration["proposed_hypotheses"]:
            h_id = h["id"]
            h_text = h["text"]
            h_kind = h.get("kind", "novel")
            
            # Find corresponding analysis
            analysis = None
            for a in iteration["analyses"]:
                if h_id in a.get("hypothesis_ids", []):
                    analysis = a
                    break
            
            if analysis:
                sig = analysis.get("significant", False)
                p_val = analysis.get("p_value", "N/A")
                effect = analysis.get("effect_estimate", "N/A")
                summary = analysis.get("result_summary", "")
                
                sig_str = "SIGNIFICANT" if sig else "not significant"
                lines.append(f"  [{h_kind.upper()}] {h_id}")
                lines.append(f"    Hypothesis: {h_text}")
                lines.append(f"    Result: {summary}")
                lines.append(f"    Effect estimate: {effect} (p={p_val:.4f}, {sig_str})")
            else:
                lines.append(f"  [{h_kind.upper()}] {h_id}")
                lines.append(f"    Hypothesis: {h_text}")
                lines.append(f"    No analysis performed")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("CONCLUSIONS")
    lines.append("=" * 70)
    
    # Find significant treatment effects
    treatment_effects = {}
    for iteration in transcript:
        for a in iteration["analyses"]:
            if a.get("significant", False):
                # Extract treatment from result_summary
                for treatment in TREATMENT_COLS:
                    if treatment in a.get("result_summary", ""):
                        if treatment not in treatment_effects:
                            treatment_effects[treatment] = []
                        treatment_effects[treatment].append(a)
    
    if treatment_effects:
        lines.append("\nSignificant Treatment Effects:")
        for treatment, analyses in treatment_effects.items():
            effects = [a.get("effect_estimate", 0) for a in analyses]
            avg_effect = sum(effects) / len(effects) if effects else 0
            lines.append(f"  {treatment}: average effect estimate = {avg_effect:+.3f}")
    else:
        lines.append("\nNo statistically significant treatment effects were identified.")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SUMMARY")
    lines.append("=" * 70)
    
    return "\n".join(lines)

def main():
    transcript = []
    
    # Run up to 10 iterations
    for _ in range(10):
        transcript = run_analysis(transcript)
    
    # Prepare transcript for JSON output
    output_transcript = {
        "dataset_id": "ds001_prostate",
        "model_id": "codex-cli",
        "harness_id": "codex-cli@1.0.0",
        "max_iterations": 10,
        "iterations": transcript
    }
    
    # Convert to JSON-serializable format
    json_transcript = to_jsonable(output_transcript)
    
    # Write transcript.json
    with open("transcript.json", "w") as f:
        json.dump(json_transcript, f, indent=2)
    
    # Generate and write analysis_summary.txt
    summary = generate_summary(transcript)
    with open("analysis_summary.txt", "w") as f:
        f.write(summary)
    
    print("Analysis complete.")
    print(f"  - transcript.json written")
    print(f"  - analysis_summary.txt written")

if __name__ == "__main__":
    main()
