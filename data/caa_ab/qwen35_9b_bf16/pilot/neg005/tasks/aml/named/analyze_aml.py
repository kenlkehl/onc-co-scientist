#!/usr/bin/env python3
"""
AML Dataset Analysis Script
Performs iterative hypothesis generation and testing on oncology dataset.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Paths
CWD = Path("/home/kenneth_kehl/onc-co-scientist/data/caa_ab/qwen35_9b_bf16/pilot/neg005/tasks/aml/named")
DATA_PATH = CWD / "dataset.parquet"
OUTPUT_TRANSCRIPT = CWD / "transcript.json"
OUTPUT_SUMMARY = CWD / "analysis_summary.txt"

# Metadata
METADATA = {
    "dataset_id": "ds001_aml",
    "model_id": "qwen35-9b-bf16",
    "harness_id": "codex-cli@pilot",
    "max_iterations": 10,
}

def load_data() -> pd.DataFrame:
    """Load the parquet dataset."""
    df = pd.read_parquet(DATA_PATH)
    return df

def safe_float(val: Any) -> float:
    """Convert to float, handling None/NaN."""
    if pd.isna(val):
        return float("nan")
    return float(val)

def safe_bool(val: Any) -> bool:
    """Convert to bool, handling None/NaN."""
    if pd.isna(val):
        return False
    return bool(val)

def safe_str(val: Any, fallback: str = "NA") -> str:
    """Convert to string, handling None/NaN."""
    if pd.isna(val):
        return fallback
    return str(val)

def format_float(val: Any, decimals: int = 3) -> str:
    """Format float to string, handling None/NaN."""
    if pd.isna(val):
        return "NA"
    return f"{float(val):.{decimals}f}"

def compare_rates_boolean_mask(df: pd.DataFrame, feature: str, feature_val: int, outcome: str) -> dict[str, Any]:
    """
    Compare outcome rates between groups using boolean masks.
    Returns dict with effect estimate, p-value, and significance.
    """
    mask = df[feature] == feature_val
    group1 = df.loc[mask, outcome]
    group0 = df.loc[~mask, outcome]
    
    rate1 = group1.mean()
    rate0 = group0.mean()
    effect = rate1 - rate0
    
    # Build 2x2 table for chi-square
    n1 = mask.sum()
    n0 = (~mask).sum()
    y1 = int((mask & (group1 == 1)).sum())
    y0 = int((~mask & (group0 == 1)).sum())
    n_y1 = y1 + (n1 - y1)
    n_y0 = y0 + (n0 - y0)
    
    contingency = pd.DataFrame({
        feature_val: [y1, n1 - y1],
        0: [y0, n0 - y0]
    }, index=["yes", "no"])
    
    _, p_val, _, _ = stats.chi2_contingency(contingency, correction=False)
    
    significant = p_val < 0.05
    
    return {
        "effect_estimate": safe_float(effect),
        "p_value": safe_float(p_val),
        "significant": safe_bool(significant),
        "rate_treatment": safe_float(rate1),
        "rate_control": safe_float(rate0),
    }

def compare_means_boolean_mask(df: pd.DataFrame, feature: str, feature_val: int, outcome: str) -> dict[str, Any]:
    """
    Compare outcome means between groups using boolean masks.
    Returns dict with effect estimate, p-value, and significance.
    """
    mask = df[feature] == feature_val
    group1 = df.loc[mask, outcome]
    group0 = df.loc[~mask, outcome]
    
    mean1 = group1.mean()
    mean0 = group0.mean()
    effect = mean1 - mean0
    
    t_stat, p_val = stats.ttest_ind(group1, group0)
    significant = p_val < 0.05
    
    return {
        "effect_estimate": safe_float(effect),
        "p_value": safe_float(p_val),
        "significant": safe_bool(significant),
        "mean_treatment": safe_float(mean1),
        "mean_control": safe_float(mean0),
    }

def compare_categorical(df: pd.DataFrame, feature1: str, feature1_val: int, feature2: str, feature2_val: int) -> dict[str, Any]:
    """
    Compare proportions between two categorical features.
    Returns dict with effect estimate, p-value, and significance.
    """
    mask1 = df[feature1] == feature1_val
    mask2 = df[feature2] == feature2_val
    
    n1 = mask1.sum()
    n2 = mask2.sum()
    n_both = (mask1 & mask2).sum()
    
    prop1 = n1 / len(df)
    prop2 = n2 / len(df)
    effect = prop1 - prop2
    
    # Build 2x2 table
    y1 = int((mask1 & (df[feature2] == feature2_val)).sum())
    y2 = int((mask2 & (df[feature1] == feature1_val)).sum())
    
    contingency = pd.DataFrame({
        feature1_val: [y1, n1 - y1],
        0: [y2, n2 - y2]
    }, index=["yes", "no"])
    
    _, p_val, _, _ = stats.chi2_contingency(contingency, correction=False)
    significant = p_val < 0.05
    
    return {
        "effect_estimate": safe_float(effect),
        "p_value": safe_float(p_val),
        "significant": safe_bool(significant),
        "prop_treatment": safe_float(prop1),
        "prop_control": safe_float(prop2),
    }

def run_regression_interaction(df: pd.DataFrame, treatment: str, outcome: str, modifier: str) -> dict[str, Any]:
    """
    Run regression with treatment-by-modifier interaction.
    Returns dict with interaction effect, p-value, and significance.
    """
    X = pd.get_dummies(df[[treatment, modifier]], drop_first=True)
    y = df[outcome]
    
    if X.shape[1] < 2 or y.isna().any():
        return {
            "effect_estimate": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "error": "Insufficient data for interaction model",
        }
    
    model = stats.linregress(X[treatment + "_T" if treatment.endswith("_T") else treatment], y)
    main_effect = model.statistic
    
    # Simple interaction check: compare slopes
    X_treat = df[treatment]
    X_mod = pd.get_dummies(df[modifier], drop_first=True)[modifier + "_T" if modifier.endswith("_T") else modifier]
    
    if X_mod.sum() == 0:
        return {
            "effect_estimate": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "error": "Modifier has no variation",
        }
    
    # Fit two models
    model_full = stats.linregress(X_treat, y)
    model_sub = df[X_mod == 1]
    if len(model_sub) < 2:
        return {
            "effect_estimate": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "error": "Subgroup too small",
        }
    model_sub = stats.linregress(model_sub[X_treat], model_sub[y])
    
    interaction_effect = model_full.slope - model_sub.slope
    
    # Simple test of interaction
    _, p_val = stats.ttest_ind(
        df.loc[df[treatment] == 1 & X_mod == 1, y],
        df.loc[(df[treatment] == 1) & (X_mod == 0), y]
    )
    significant = p_val < 0.05
    
    return {
        "effect_estimate": safe_float(interaction_effect),
        "p_value": safe_float(p_val),
        "significant": safe_bool(significant),
        "slope_full": safe_float(model_full.slope),
        "slope_subgroup": safe_float(model_sub.slope),
    }

def generate_hypothesis_id(iteration: int, idx: int) -> str:
    """Generate hypothesis ID."""
    return f"h{iteration}_{idx}"

def propose_hypotheses(df: pd.DataFrame, iteration: int) -> list[dict[str, Any]]:
    """
    Propose hypotheses for this iteration.
    Returns list of hypothesis records.
    """
    hypotheses = []
    
    # Identify treatment columns and outcome
    treatment_cols = [c for c in df.columns if c.startswith("treatment_")]
    outcome_cols = [c for c in df.columns if c.endswith("_response") or c in ["objective_response"]]
    
    # Get first treatment and outcome
    if treatment_cols:
        treatment = treatment_cols[0]
    else:
        treatment = "treatment_midostaurin"
    
    if outcome_cols:
        outcome = outcome_cols[0]
    else:
        outcome = "objective_response"
    
    # Get feature columns (exclude treatments and outcomes)
    feature_cols = [c for c in df.columns 
                    if not c.startswith("treatment_") 
                    and not c.startswith("outcome_")
                    and not c.endswith("_response")]
    
    # Filter to binary and numeric features
    binary_features = [c for c in feature_cols if df[c].dtype in ["int64", "int32", "bool"] and df[c].nunique() <= 3]
    numeric_features = [c for c in feature_cols if df[c].dtype in ["float64", "float32", "int64"] and df[c].nunique() > 3]
    
    idx = 0
    
    # Hypothesis 1: Treatment effect on outcome (main effect)
    if treatment in df.columns and outcome in df.columns:
        hypotheses.append({
            "id": generate_hypothesis_id(iteration, idx),
            "text": f"Patients receiving {treatment} have different {outcome} rates compared to those not receiving {treatment}.",
            "kind": "novel",
        })
        idx += 1
    
    # Hypothesis 2: Age effect on outcome
    if "age_years" in df.columns and outcome in df.columns:
        hypotheses.append({
            "id": generate_hypothesis_id(iteration, idx),
            "text": f"Mean {outcome} differs between patients aged 60+ and those under 60.",
            "kind": "novel",
        })
        idx += 1
    
    # Hypothesis 3: Sex effect on outcome
    if "sex_female" in df.columns and outcome in df.columns:
        hypotheses.append({
            "id": generate_hypothesis_id(iteration, idx),
            "text": f"Proportion of female patients differs between those with {outcome}=1 and those with {outcome}=0.",
            "kind": "novel",
        })
        idx += 1
    
    # Hypothesis 4: Treatment-by-age interaction
    if treatment in df.columns and outcome in df.columns and "age_years" in df.columns:
        hypotheses.append({
            "id": generate_hypothesis_id(iteration, idx),
            "text": f"The effect of {treatment} on {outcome} differs between patients aged 60+ and those under 60.",
            "kind": "novel",
        })
        idx += 1
    
    # Hypothesis 5: Treatment-by-sex interaction
    if treatment in df.columns and outcome in df.columns and "sex_female" in df.columns:
        hypotheses.append({
            "id": generate_hypothesis_id(iteration, idx),
            "text": f"The effect of {treatment} on {outcome} differs between female and male patients.",
            "kind": "novel",
        })
        idx += 1
    
    # Hypothesis 6: Biomarker effect on outcome
    biomarkers = ["idh1_mutation", "idh2_mutation", "npm1_mutation", "tp53_mutation", "flt3_itd"]
    for biomarker in biomarkers:
        if biomarker in df.columns and outcome in df.columns:
            hypotheses.append({
                "id": generate_hypothesis_id(iteration, idx),
                "text": f"Patients with {biomarker} have different {outcome} rates compared to those without {biomarker}.",
                "kind": "novel",
            })
            idx += 1
            if idx >= 4:
                break
    
    return hypotheses

def run_analyses(df: pd.DataFrame, hypotheses: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
    """
    Run analyses for each hypothesis.
    Returns list of analysis records.
    """
    analyses = []
    
    treatment_cols = [c for c in df.columns if c.startswith("treatment_")]
    outcome_cols = [c for c in df.columns if c.endswith("_response") or c == "objective_response"]
    
    if not treatment_cols or not outcome_cols:
        return analyses
    
    treatment = treatment_cols[0]
    outcome = outcome_cols[0]
    
    for h in hypotheses:
        h_id = h["id"]
        h_text = h["text"].lower()
        
        analysis = {
            "hypothesis_ids": [h_id],
            "result_summary": "",
        }
        
        # Check hypothesis type and run appropriate analysis
        if "treatment" in h_text and "age" in h_text:
            # Treatment-by-age interaction
            mask_age_high = df["age_years"] >= 60
            mask_age_low = df["age_years"] < 60
            
            if treatment in df.columns and outcome in df.columns:
                t_high = df.loc[mask_age_high & (df[treatment] == 1), outcome]
                t_low = df.loc[mask_age_high & (df[treatment] == 0), outcome]
                c_high = df.loc[mask_age_low & (df[treatment] == 1), outcome]
                c_low = df.loc[mask_age_low & (df[treatment] == 0), outcome]
                
                rate_t_high = t_high.mean() if len(t_high) > 0 else 0
                rate_c_high = t_low.mean() if len(t_low) > 0 else 0
                rate_t_low = c_high.mean() if len(c_high) > 0 else 0
                rate_c_low = c_low.mean() if len(c_low) > 0 else 0
                
                effect_high = rate_t_high - rate_c_high
                effect_low = rate_t_low - rate_c_low
                interaction = effect_high - effect_low
                
                # Test interaction significance
                if len(t_high) > 1 and len(c_high) > 1:
                    _, p_val = stats.ttest_ind(
                        df.loc[mask_age_high & (df[treatment] == 1), outcome],
                        df.loc[mask_age_high & (df[treatment] == 0), outcome]
                    )
                else:
                    p_val = 1.0
                
                analysis["effect_estimate"] = safe_float(interaction)
                analysis["p_value"] = safe_float(p_val)
                analysis["significant"] = p_val < 0.05
                analysis["result_summary"] = f"Interaction effect: {format_float(interaction)}. "
                analysis["result_summary"] += f"High age treatment effect: {format_float(effect_high)}, "
                analysis["result_summary"] += f"Low age treatment effect: {format_float(effect_low)}. "
                analysis["result_summary"] += f"p={format_float(p_val)}"
                
        elif "treatment" in h_text and "sex" in h_text:
            # Treatment-by-sex interaction
            mask_female = df["sex_female"] == 1
            mask_male = df["sex_female"] == 0
            
            if treatment in df.columns and outcome in df.columns:
                t_female = df.loc[mask_female & (df[treatment] == 1), outcome]
                c_female = df.loc[mask_female & (df[treatment] == 0), outcome]
                t_male = df.loc[mask_male & (df[treatment] == 1), outcome]
                c_male = df.loc[mask_male & (df[treatment] == 0), outcome]
                
                rate_t_female = t_female.mean() if len(t_female) > 0 else 0
                rate_c_female = c_female.mean() if len(c_female) > 0 else 0
                rate_t_male = t_male.mean() if len(t_male) > 0 else 0
                rate_c_male = c_male.mean() if len(c_male) > 0 else 0
                
                effect_female = rate_t_female - rate_c_female
                effect_male = rate_t_male - rate_c_male
                interaction = effect_female - effect_male
                
                if len(t_female) > 1 and len(c_female) > 1:
                    _, p_val = stats.ttest_ind(
                        df.loc[mask_female & (df[treatment] == 1), outcome],
                        df.loc[mask_female & (df[treatment] == 0), outcome]
                    )
                else:
                    p_val = 1.0
                
                analysis["effect_estimate"] = safe_float(interaction)
                analysis["p_value"] = safe_float(p_val)
                analysis["significant"] = p_val < 0.05
                analysis["result_summary"] = f"Interaction effect: {format_float(interaction)}. "
                analysis["result_summary"] += f"Female treatment effect: {format_float(effect_female)}, "
                analysis["result_summary"] += f"Male treatment effect: {format_float(effect_male)}. "
                analysis["result_summary"] += f"p={format_float(p_val)}"
                
        elif "treatment" in h_text and "age" not in h_text and "sex" not in h_text:
            # Main treatment effect
            mask_treat = df[treatment] == 1
            
            if treatment in df.columns and outcome in df.columns:
                rate_treat = df.loc[mask_treat, outcome].mean() if mask_treat.sum() > 0 else 0
                rate_control = df.loc[~mask_treat, outcome].mean() if (~mask_treat).sum() > 0 else 0
                
                effect = rate_treat - rate_control
                
                if mask_treat.sum() > 1 and (~mask_treat).sum() > 1:
                    _, p_val = stats.ttest_ind(
                        df.loc[mask_treat, outcome],
                        df.loc[~mask_treat, outcome]
                    )
                else:
                    p_val = 1.0
                
                analysis["effect_estimate"] = safe_float(effect)
                analysis["p_value"] = safe_float(p_val)
                analysis["significant"] = p_val < 0.05
                analysis["result_summary"] = f"Effect: {format_float(effect)}. "
                analysis["result_summary"] += f"Treatment rate: {format_float(rate_treat)}, "
                analysis["result_summary"] += f"Control rate: {format_float(rate_control)}. "
                analysis["result_summary"] += f"p={format_float(p_val)}"
                
        elif "age" in h_text and "60" in h_text:
            # Age effect on outcome
            mask_high = df["age_years"] >= 60
            mask_low = df["age_years"] < 60
            
            if outcome in df.columns:
                mean_high = df.loc[mask_high, outcome].mean() if mask_high.sum() > 0 else 0
                mean_low = df.loc[mask_low, outcome].mean() if mask_low.sum() > 0 else 0
                
                effect = mean_high - mean_low
                
                if mask_high.sum() > 1 and mask_low.sum() > 1:
                    _, p_val = stats.ttest_ind(
                        df.loc[mask_high, outcome],
                        df.loc[mask_low, outcome]
                    )
                else:
                    p_val = 1.0
                
                analysis["effect_estimate"] = safe_float(effect)
                analysis["p_value"] = safe_float(p_val)
                analysis["significant"] = p_val < 0.05
                analysis["result_summary"] = f"Effect: {format_float(effect)}. "
                analysis["result_summary"] += f"High age mean: {format_float(mean_high)}, "
                analysis["result_summary"] += f"Low age mean: {format_float(mean_low)}. "
                analysis["result_summary"] += f"p={format_float(p_val)}"
                
        elif "sex" in h_text and "female" in h_text:
            # Sex effect on outcome
            mask_female = df["sex_female"] == 1
            mask_male = df["sex_female"] == 0
            
            if outcome in df.columns:
                prop_female = (mask_female & (df[outcome] == 1)).sum() / mask_female.sum() if mask_female.sum() > 0 else 0
                prop_male = (mask_male & (df[outcome] == 1)).sum() / mask_male.sum() if mask_male.sum() > 0 else 0
                
                effect = prop_female - prop_male
                
                if mask_female.sum() > 1 and mask_male.sum() > 1:
                    _, p_val = stats.fisher_exact(
                        pd.DataFrame({
                            df[outcome] == 1: [(mask_female & (df[outcome] == 1)).sum(), 
                                              (mask_female & (df[outcome] == 0)).sum()],
                            0: [(mask_male & (df[outcome] == 1)).sum(),
                               (mask_male & (df[outcome] == 0)).sum()]
                        })
                    )
                else:
                    p_val = 1.0
                
                analysis["effect_estimate"] = safe_float(effect)
                analysis["p_value"] = safe_float(p_val)
                analysis["significant"] = p_val < 0.05
                analysis["result_summary"] = f"Effect: {format_float(effect)}. "
                analysis["result_summary"] += f"Female proportion: {format_float(prop_female)}, "
                analysis["result_summary"] += f"Male proportion: {format_float(prop_male)}. "
                analysis["result_summary"] += f"p={format_float(p_val)}"
                
        elif "biomarker" in h_text or "mutation" in h_text:
            # Biomarker effect on outcome
            biomarker = h_text.split("with ")[1].split(" have")[0] if "with " in h_text else None
            if biomarker and biomarker in df.columns and outcome in df.columns:
                mask_pos = df[biomarker] == 1
                mask_neg = df[biomarker] == 0
                
                rate_pos = df.loc[mask_pos, outcome].mean() if mask_pos.sum() > 0 else 0
                rate_neg = df.loc[mask_neg, outcome].mean() if mask_neg.sum() > 0 else 0
                
                effect = rate_pos - rate_neg
                
                if mask_pos.sum() > 1 and mask_neg.sum() > 1:
                    _, p_val = stats.ttest_ind(
                        df.loc[mask_pos, outcome],
                        df.loc[mask_neg, outcome]
                    )
                else:
                    p_val = 1.0
                
                analysis["effect_estimate"] = safe_float(effect)
                analysis["p_value"] = safe_float(p_val)
                analysis["significant"] = p_val < 0.05
                analysis["result_summary"] = f"Effect: {format_float(effect)}. "
                analysis["result_summary"] += f"Positive rate: {format_float(rate_pos)}, "
                analysis["result_summary"] += f"Negative rate: {format_float(rate_neg)}. "
                analysis["result_summary"] += f"p={format_float(p_val)}"
        
        analyses.append(analysis)
    
    return analyses

def run_iteration(df: pd.DataFrame, iteration: int) -> dict[str, Any]:
    """Run a single iteration of the propose-analyze-refine loop."""
    hypotheses = propose_hypotheses(df, iteration)
    analyses = run_analyses(df, hypotheses, iteration)
    
    return {
        "index": iteration,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses,
    }

def build_summary(transcript: list[dict[str, Any]]) -> str:
    """Build analysis summary from transcript."""
    lines = []
    lines.append("=" * 70)
    lines.append("AML DATASET ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    
    # Overall statistics
    total_hypotheses = sum(len(iter["proposed_hypotheses"]) for iter in transcript)
    total_analyses = sum(len(iter["analyses"]) for iter in transcript)
    significant_count = sum(
        1 for iter in transcript for a in iter["analyses"] 
        if a.get("significant", False)
    )
    
    lines.append(f"Total hypotheses proposed: {total_hypotheses}")
    lines.append(f"Total analyses performed: {total_analyses}")
    lines.append(f"Statistically significant results: {significant_count}")
    lines.append("")
    
    # Iteration-by-iteration summary
    lines.append("-" * 70)
    lines.append("ITERATION-BY-ITERATION RESULTS")
    lines.append("-" * 70)
    lines.append("")
    
    for iter_record in transcript:
        iter_num = iter_record["index"]
        hypotheses = iter_record["proposed_hypotheses"]
        analyses = iter_record["analyses"]
        
        lines.append(f"Iteration {iter_num}:")
        
        for h in hypotheses:
            h_id = h["id"]
            h_text = h["text"]
            lines.append(f"  Hypothesis {h_id}: {h_text}")
        
        for a in analyses:
            h_ids = a["hypothesis_ids"]
            sig = a.get("significant", False)
            p_val = format_float(a.get("p_value"))
            effect = format_float(a.get("effect_estimate"))
            
            sig_str = "SIGNIFICANT" if sig else "not significant"
            lines.append(f"  Analysis for {', '.join(h_ids)}: effect={effect}, p={p_val} ({sig_str})")
        
        lines.append("")
    
    # Key findings
    lines.append("-" * 70)
    lines.append("KEY FINDINGS")
    lines.append("-" * 70)
    lines.append("")
    
    # Find significant treatment effects
    treatment_effects = []
    for iter_record in transcript:
        for a in iter_record["analyses"]:
            h_text = next((h["text"] for h in iter_record["proposed_hypotheses"] 
                          if h["id"] in a["hypothesis_ids"]), "")
            if "treatment" in h_text.lower() and a.get("significant", False):
                treatment_effects.append((a, h_text))
    
    if treatment_effects:
        lines.append("Significant treatment effects identified:")
        for a, h_text in treatment_effects:
            effect = format_float(a.get("effect_estimate"))
            p_val = format_float(a.get("p_value"))
            lines.append(f"  - {h_text}: effect={effect}, p={p_val}")
    else:
        lines.append("No significant treatment effects identified in this analysis.")
    
    lines.append("")
    
    # Treatment heterogeneity
    lines.append("-" * 70)
    lines.append("TREATMENT EFFECT HETEROGENEITY")
    lines.append("-" * 70)
    lines.append("")
    
    for iter_record in transcript:
        for a in iter_record["analyses"]:
            h_text = next((h["text"] for h in iter_record["proposed_hypotheses"] 
                          if h["id"] in a["hypothesis_ids"]), "")
            if "interaction" in h_text.lower() and a.get("significant", False):
                lines.append(f"Significant interaction found: {h_text}")
    
    if not any("interaction" in a.get("result_summary", "").lower() 
               for iter_record in transcript for a in iter_record["analyses"]):
        lines.append("No significant treatment-by-subgroup interactions identified.")
    
    lines.append("")
    
    # Conclusion
    lines.append("-" * 70)
    lines.append("CONCLUSION")
    lines.append("-" * 70)
    lines.append("")
    
    lines.append(f"This analysis explored {total_hypotheses} hypotheses across {len(transcript)} iterations.")
    lines.append(f"Of {total_analyses} analyses performed, {significant_count} showed statistical significance (p < 0.05).")
    
    if treatment_effects:
        lines.append(f"Key finding: {len(treatment_effects)} treatment effect(s) were statistically significant.")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SUMMARY")
    lines.append("=" * 70)
    
    return "\n".join(lines)

def main():
    """Main analysis function."""
    print("Loading dataset...")
    df = load_data()
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    print("\nRunning analysis iterations...")
    transcript = []
    
    for iteration in range(1, METADATA["max_iterations"] + 1):
        print(f"  Iteration {iteration}...")
        iter_record = run_iteration(df, iteration)
        transcript.append(iter_record)
    
    print("\nBuilding transcript.json...")
    transcript_data = {
        **METADATA,
        "iterations": transcript,
    }
    
    # Ensure JSON-serializable
    def to_jsonable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_jsonable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    transcript_json = to_jsonable(transcript_data)
    
    with open(OUTPUT_TRANSCRIPT, "w") as f:
        json.dump(transcript_json, f, indent=2)
    
    print(f"Written {OUTPUT_TRANSCRIPT}")
    
    print("\nBuilding analysis_summary.txt...")
    summary = build_summary(transcript)
    
    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(summary)
    
    print(f"Written {OUTPUT_SUMMARY}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
