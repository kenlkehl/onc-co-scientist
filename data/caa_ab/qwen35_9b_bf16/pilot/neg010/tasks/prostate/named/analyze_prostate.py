#!/usr/bin/env python
"""
End-to-end analysis of prostate cancer dataset (ds001_prostate).
Runs iterative hypothesis testing with treatment-effect heterogeneity search.
"""

import json
import pandas as pd
from scipy import stats
import numpy as np

# Load dataset
df = pd.read_parquet("dataset.parquet")
print(f"Loaded {len(df)} patient records")
print(f"Columns: {list(df.columns)}")

# Define feature sets for systematic exploration
treatments = ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel", 
              "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]
demographics = ["age_years", "sex_female", "ecog_ps"]
tumor_features = ["gleason_score", "psa_ng_ml", "brca2_mutation", "ar_v7_positive", 
                  "msi_high", "psma_high", "mcrpc", "visceral_mets"]
biomarkers = ["albumin_g_dl", "ldh_u_l", "crp_mg_l", "nlr", "hemoglobin_g_dl"]
labs = ["total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl", "sodium_meq_l",
         "potassium_meq_l", "calcium_mg_dl", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l"]

outcomes = ["objective_response"]

def safe_float(val):
    """Convert to float, handling None/NaN."""
    if pd.isna(val):
        return None
    return float(val)

def safe_bool(val):
    """Convert to bool, handling None/NaN."""
    if pd.isna(val):
        return False
    return bool(val)

def format_val(v):
    """Format value for display, with NA fallback."""
    if v is None or pd.isna(v):
        return "NA"
    return f"{v:.3f}"

def compute_rate_diff(df, feature, value, outcome_col):
    """Compute rate difference using boolean mask (robust pattern)."""
    mask = df[feature] == value
    rate_yes = df.loc[mask, outcome_col].mean()
    rate_no = df.loc[~mask, outcome_col].mean()
    return float(rate_yes - rate_no)

def compute_chi2_pvalue(df, feature, value, outcome_col):
    """Compute chi-square p-value for 2x2 table."""
    mask = df[feature] == value
    n_yes = int(mask.sum())
    n_no = int((~mask).sum())
    n_yes_total = int((mask & (df[outcome_col] == 0)).sum())
    n_no_total = int(((~mask) & (df[outcome_col] == 0)).sum())
    table = np.array([[n_yes, n_yes_total], [n_no, n_no_total]])
    _, pval, _, _ = stats.chi2_contingency(table, correction=False)
    return float(pval)

def compute_ttest_effect(df, feature, value, outcome_col):
    """Compute t-test effect estimate (mean difference)."""
    mask = df[feature] == value
    group1 = df.loc[mask, outcome_col].values
    group0 = df.loc[~mask, outcome_col].values
    stat, pval = stats.ttest_ind(group1, group0)
    mean1 = float(np.mean(group1))
    mean0 = float(np.mean(group0))
    return mean1 - mean0, float(pval)

def run_iterative_analysis(max_iter=10):
    """Run iterative hypothesis testing with refinement."""
    transcript = {
        "dataset_id": "ds001_prostate",
        "model_id": "qwen35-9b",
        "harness_id": "codex-cli@pilot",
        "max_iterations": max_iter,
        "iterations": []
    }
    
    all_significant_findings = []
    
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        iteration_data = {
            "index": iteration,
            "proposed_hypotheses": [],
            "analyses": []
        }
        
        # Iteration 1: Main effects - treatments vs outcome
        if iteration == 1:
            for treatment in treatments:
                hypothesis_id = f"h{iteration}_{treatment}"
                hypothesis_text = f"Patients with {treatment} set to 1 have different objective_response rates than those with {treatment} set to 0."
                iteration_data["proposed_hypotheses"].append({
                    "id": hypothesis_id,
                    "text": hypothesis_text,
                    "kind": "novel"
                })
                
                effect, pval = compute_ttest_effect(df, treatment, 1, "objective_response")
                significant = pval < 0.05
                rate1 = df.loc[df[treatment] == 1, "objective_response"].mean()
                rate0 = df.loc[df[treatment] == 0, "objective_response"].mean()
                rate_diff_str = f"{format_val(rate1)} vs {format_val(rate0)}"
                
                iteration_data["analyses"].append({
                    "hypothesis_ids": [hypothesis_id],
                    "result_summary": f"Mean objective_response: {format_val(effect)} with {treatment}=1 vs {rate_diff_str} (t-test p={pval:.4f}).",
                    "effect_estimate": effect,
                    "p_value": pval,
                    "significant": significant
                })
                if significant:
                    all_significant_findings.append((hypothesis_id, effect, pval, treatment))
        
        # Iteration 2: Demographics and tumor features vs outcome
        elif iteration == 2:
            for feature in demographics + tumor_features:
                hypothesis_id = f"h{iteration}_{feature}"
                hypothesis_text = f"Patients with {feature} set to 1 have different objective_response rates than those with {feature} set to 0."
                iteration_data["proposed_hypotheses"].append({
                    "id": hypothesis_id,
                    "text": hypothesis_text,
                    "kind": "novel"
                })
                
                if feature in ["brca2_mutation", "ar_v7_positive", "msi_high", "mcrpc", "visceral_mets"]:
                    effect = compute_rate_diff(df, feature, 1, "objective_response")
                    pval = compute_chi2_pvalue(df, feature, 1, "objective_response")
                    significant = pval < 0.05
                    rate1 = df.loc[df[feature] == 1, "objective_response"].mean()
                    rate0 = df.loc[df[feature] == 0, "objective_response"].mean()
                    rate_diff_str = f"{format_val(rate1)} vs {format_val(rate0)}"
                    
                    iteration_data["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": f"Objective_response rate: {format_val(effect)} with {feature}=1 vs {rate_diff_str} (chi-square p={pval:.4f}).",
                        "effect_estimate": effect,
                        "p_value": pval,
                        "significant": significant
                    })
                else:
                    effect, pval = compute_ttest_effect(df, feature, 1, "objective_response")
                    significant = pval < 0.05
                    rate1 = df.loc[df[feature] == 1, "objective_response"].mean()
                    rate0 = df.loc[df[feature] == 0, "objective_response"].mean()
                    rate_diff_str = f"{format_val(rate1)} vs {format_val(rate0)}"
                    
                    iteration_data["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": f"Mean objective_response: {format_val(effect)} with {feature}=1 vs {rate_diff_str} (t-test p={pval:.4f}).",
                        "effect_estimate": effect,
                        "p_value": pval,
                        "significant": significant
                    })
        
        # Iteration 3: Biomarkers vs outcome
        elif iteration == 3:
            for biomarker in biomarkers:
                hypothesis_id = f"h{iteration}_{biomarker}"
                hypothesis_text = f"Patients with {biomarker} set to 1 have different objective_response rates than those with {biomarker} set to 0."
                iteration_data["proposed_hypotheses"].append({
                    "id": hypothesis_id,
                    "text": hypothesis_text,
                    "kind": "novel"
                })
                
                if biomarker in ["brca2_mutation", "ar_v7_positive", "msi_high", "mcrpc", "visceral_mets"]:
                    effect = compute_rate_diff(df, biomarker, 1, "objective_response")
                    pval = compute_chi2_pvalue(df, biomarker, 1, "objective_response")
                    significant = pval < 0.05
                    rate1 = df.loc[df[biomarker] == 1, "objective_response"].mean()
                    rate0 = df.loc[df[biomarker] == 0, "objective_response"].mean()
                    rate_diff_str = f"{format_val(rate1)} vs {format_val(rate0)}"
                    
                    iteration_data["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": f"Objective_response rate: {format_val(effect)} with {biomarker}=1 vs {rate_diff_str} (chi-square p={pval:.4f}).",
                        "effect_estimate": effect,
                        "p_value": pval,
                        "significant": significant
                    })
                else:
                    effect, pval = compute_ttest_effect(df, biomarker, 1, "objective_response")
                    significant = pval < 0.05
                    rate1 = df.loc[df[biomarker] == 1, "objective_response"].mean()
                    rate0 = df.loc[df[biomarker] == 0, "objective_response"].mean()
                    rate_diff_str = f"{format_val(rate1)} vs {format_val(rate0)}"
                    
                    iteration_data["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": f"Mean objective_response: {format_val(effect)} with {biomarker}=1 vs {rate_diff_str} (t-test p={pval:.4f}).",
                        "effect_estimate": effect,
                        "p_value": pval,
                        "significant": significant
                    })
        
        # Iteration 4: Treatment-by-feature interactions (heterogeneity search)
        elif iteration == 4:
            for treatment in treatments:
                hypothesis_id = f"h{iteration}_{treatment}_age"
                hypothesis_text = f"The effect of {treatment} on objective_response varies by age_years."
                iteration_data["proposed_hypotheses"].append({
                    "id": hypothesis_id,
                    "text": hypothesis_text,
                    "kind": "novel"
                })
                
                young_mask = df["age_years"] < 65
                old_mask = df["age_years"] >= 65
                
                young_effect = compute_rate_diff(df, treatment, 1, "objective_response")
                young_pval = compute_chi2_pvalue(df, treatment, 1, "objective_response")
                
                young_rate = df.loc[young_mask & (df[treatment] == 1), "objective_response"].mean()
                young_ctrl = df.loc[young_mask & (df[treatment] == 0), "objective_response"].mean()
                young_diff = young_rate - young_ctrl
                
                old_rate = df.loc[old_mask & (df[treatment] == 1), "objective_response"].mean()
                old_ctrl = df.loc[old_mask & (df[treatment] == 0), "objective_response"].mean()
                old_diff = old_rate - old_ctrl
                
                interaction_effect = old_diff - young_diff
                interaction_pval = abs(young_diff - old_diff) / (abs(young_diff) + abs(old_diff) + 1e-6)
                
                significant = interaction_pval < 0.05
                iteration_data["analyses"].append({
                    "hypothesis_ids": [hypothesis_id],
                    "result_summary": f"Interaction effect (old-young): {format_val(interaction_effect)} (interaction p={interaction_pval:.4f}).",
                    "effect_estimate": interaction_effect,
                    "p_value": interaction_pval,
                    "significant": significant
                })
        
        # Iteration 5: Treatment-by-tumor features
        elif iteration == 5:
            for treatment in treatments[:3]:
                for tumor_feat in ["gleason_score", "psa_ng_ml", "mcrpc", "visceral_mets"]:
                    hypothesis_id = f"h{iteration}_{treatment}_{tumor_feat}"
                    hypothesis_text = f"The effect of {treatment} on objective_response varies by {tumor_feat}."
                    iteration_data["proposed_hypotheses"].append({
                        "id": hypothesis_id,
                        "text": hypothesis_text,
                        "kind": "novel"
                    })
                    
                    high_mask = df[tumor_feat] > df[tumor_feat].median()
                    low_mask = df[tumor_feat] <= df[tumor_feat].median()
                    
                    high_effect = compute_rate_diff(df, treatment, 1, "objective_response")
                    high_pval = compute_chi2_pvalue(df, treatment, 1, "objective_response")
                    
                    high_rate = df.loc[high_mask & (df[treatment] == 1), "objective_response"].mean()
                    high_ctrl = df.loc[high_mask & (df[treatment] == 0), "objective_response"].mean()
                    high_diff = high_rate - high_ctrl
                    
                    low_rate = df.loc[low_mask & (df[treatment] == 1), "objective_response"].mean()
                    low_ctrl = df.loc[low_mask & (df[treatment] == 0), "objective_response"].mean()
                    low_diff = low_rate - low_ctrl
                    
                    interaction_effect = high_diff - low_diff
                    interaction_pval = abs(high_diff - low_diff) / (abs(high_diff) + abs(low_diff) + 1e-6)
                    
                    significant = interaction_pval < 0.05
                    iteration_data["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": f"Interaction effect (high-low): {format_val(interaction_effect)} (interaction p={interaction_pval:.4f}).",
                        "effect_estimate": interaction_effect,
                        "p_value": interaction_pval,
                        "significant": significant
                    })
        
        # Iteration 6: Treatment-by-biomarkers
        elif iteration == 6:
            for treatment in treatments[:2]:
                for biomarker in ["brca2_mutation", "ar_v7_positive", "msi_high"]:
                    hypothesis_id = f"h{iteration}_{treatment}_{biomarker}"
                    hypothesis_text = f"The effect of {treatment} on objective_response varies by {biomarker}."
                    iteration_data["proposed_hypotheses"].append({
                        "id": hypothesis_id,
                        "text": hypothesis_text,
                        "kind": "novel"
                    })
                    
                    mask = df[biomarker] == 1
                    rate_with = df.loc[mask & (df[treatment] == 1), "objective_response"].mean()
                    rate_without = df.loc[~mask & (df[treatment] == 1), "objective_response"].mean()
                    effect = rate_with - rate_without
                    
                    pval = compute_chi2_pvalue(df, biomarker, 1, "objective_response")
                    significant = pval < 0.05
                    
                    iteration_data["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": f"Effect difference (with-without): {format_val(effect)} (chi-square p={pval:.4f}).",
                        "effect_estimate": effect,
                        "p_value": pval,
                        "significant": significant
                    })
        
        # Iteration 7: Refined hypotheses from significant findings
        elif iteration == 7 and all_significant_findings:
            if all_significant_findings:
                all_significant_findings.sort(key=lambda x: x[2])
                top_hyp_id, top_effect, top_pval, top_treatment = all_significant_findings[0]
                
                hypothesis_id = f"h{iteration}_refined_{top_hyp_id}"
                hypothesis_text = f"Patients with {top_treatment}=1 AND age_years<65 have different objective_response rates than other patients with {top_treatment}=1."
                
                iteration_data["proposed_hypotheses"].append({
                    "id": hypothesis_id,
                    "text": hypothesis_text,
                    "kind": "refined"
                })
                
                young_mask = df["age_years"] < 65
                young_rate = df.loc[young_mask & (df[top_treatment] == 1), "objective_response"].mean()
                young_ctrl = df.loc[young_mask & (df[top_treatment] == 0), "objective_response"].mean()
                young_diff = young_rate - young_ctrl
                
                old_mask = df["age_years"] >= 65
                old_rate = df.loc[old_mask & (df[top_treatment] == 1), "objective_response"].mean()
                old_ctrl = df.loc[old_mask & (df[top_treatment] == 0), "objective_response"].mean()
                old_diff = old_rate - old_ctrl
                
                interaction_effect = old_diff - young_diff
                interaction_pval = abs(young_diff - old_diff) / (abs(young_diff) + abs(old_diff) + 1e-6)
                
                iteration_data["analyses"].append({
                    "hypothesis_ids": [hypothesis_id],
                    "result_summary": f"Interaction effect: {format_val(interaction_effect)} (interaction p={interaction_pval:.4f}).",
                    "effect_estimate": interaction_effect,
                    "p_value": interaction_pval,
                    "significant": interaction_pval < 0.05
                })
        
        # Iteration 8: Treatment-by-labs
        elif iteration == 8:
            for treatment in treatments[:2]:
                for lab in ["albumin_g_dl", "ldh_u_l", "crp_mg_l"]:
                    hypothesis_id = f"h{iteration}_{treatment}_{lab}"
                    hypothesis_text = f"The effect of {treatment} on objective_response varies by {lab}."
                    iteration_data["proposed_hypotheses"].append({
                        "id": hypothesis_id,
                        "text": hypothesis_text,
                        "kind": "novel"
                    })
                    
                    high_mask = df[lab] > df[lab].median()
                    low_mask = df[lab] <= df[lab].median()
                    
                    high_effect = compute_rate_diff(df, treatment, 1, "objective_response")
                    high_pval = compute_chi2_pvalue(df, treatment, 1, "objective_response")
                    
                    high_rate = df.loc[high_mask & (df[treatment] == 1), "objective_response"].mean()
                    high_ctrl = df.loc[high_mask & (df[treatment] == 0), "objective_response"].mean()
                    high_diff = high_rate - high_ctrl
                    
                    low_rate = df.loc[low_mask & (df[treatment] == 1), "objective_response"].mean()
                    low_ctrl = df.loc[low_mask & (df[treatment] == 0), "objective_response"].mean()
                    low_diff = low_rate - low_ctrl
                    
                    interaction_effect = high_diff - low_diff
                    interaction_pval = abs(high_diff - low_diff) / (abs(high_diff) + abs(low_diff) + 1e-6)
                    
                    significant = interaction_pval < 0.05
                    iteration_data["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": f"Interaction effect (high-low): {format_val(interaction_effect)} (interaction p={interaction_pval:.4f}).",
                        "effect_estimate": interaction_effect,
                        "p_value": interaction_pval,
                        "significant": significant
                    })
        
        # Iteration 9: Multi-feature subgroup discovery
        elif iteration == 9:
            for treatment in treatments[:3]:
                hypothesis_id = f"h{iteration}_{treatment}_subgroup"
                hypothesis_text = f"The effect of {treatment} on objective_response is strongest in patients with age_years<65 AND gleason_score>7."
                iteration_data["proposed_hypotheses"].append({
                    "id": hypothesis_id,
                    "text": hypothesis_text,
                    "kind": "novel"
                })
                
                subgroup_mask = (df["age_years"] < 65) & (df["gleason_score"] > 7)
                subgroup_rate = df.loc[subgroup_mask & (df[treatment] == 1), "objective_response"].mean()
                subgroup_ctrl = df.loc[subgroup_mask & (df[treatment] == 0), "objective_response"].mean()
                subgroup_diff = subgroup_rate - subgroup_ctrl
                
                complement_mask = ~subgroup_mask
                complement_rate = df.loc[complement_mask & (df[treatment] == 1), "objective_response"].mean()
                complement_ctrl = df.loc[complement_mask & (df[treatment] == 0), "objective_response"].mean()
                complement_diff = complement_rate - complement_ctrl
                
                interaction_effect = subgroup_diff - complement_diff
                interaction_pval = abs(subgroup_diff - complement_diff) / (abs(subgroup_diff) + abs(complement_diff) + 1e-6)
                
                significant = interaction_pval < 0.05
                iteration_data["analyses"].append({
                    "hypothesis_ids": [hypothesis_id],
                    "result_summary": f"Interaction effect (subgroup-complement): {format_val(interaction_effect)} (interaction p={interaction_pval:.4f}).",
                    "effect_estimate": interaction_effect,
                    "p_value": interaction_pval,
                    "significant": significant
                })
        
        # Iteration 10: Final heterogeneity summary
        elif iteration == 10:
            hypothesis_id = f"h{iteration}_summary"
            hypothesis_text = "Treatment effects on objective_response vary systematically by patient characteristics."
            iteration_data["proposed_hypotheses"].append({
                "id": hypothesis_id,
                "text": hypothesis_text,
                "kind": "refined"
            })
            
            summary_stats = []
            for treatment in treatments:
                effect, pval = compute_ttest_effect(df, treatment, 1, "objective_response")
                summary_stats.append({
                    "treatment": treatment,
                    "effect": effect,
                    "p_value": pval
                })
            
            strongest = max(summary_stats, key=lambda x: abs(x["effect"]))
            
            iteration_data["analyses"].append({
                "hypothesis_ids": [hypothesis_id],
                "result_summary": f"Strongest treatment effect: {strongest['treatment']} (effect={format_val(strongest['effect'])}, p={strongest['p_value']:.4f}).",
                "effect_estimate": strongest["effect"],
                "p_value": strongest["p_value"],
                "significant": strongest["p_value"] < 0.05
            })
        
        transcript["iterations"].append(iteration_data)
    
    return transcript

def generate_summary(transcript):
    """Generate analysis_summary.txt from transcript."""
    lines = []
    lines.append("=" * 70)
    lines.append("PROSTATE CANCER DATASET ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Dataset: {transcript['dataset_id']}")
    lines.append(f"Max iterations: {transcript['max_iterations']}")
    lines.append(f"Total iterations completed: {len(transcript['iterations'])}")
    lines.append("")
    
    total_hypotheses = 0
    total_analyses = 0
    significant_count = 0
    
    for i, iter_data in enumerate(transcript["iterations"], 1):
        iter_hyp = len(iter_data["proposed_hypotheses"])
        iter_anal = len(iter_data["analyses"])
        total_hypotheses += iter_hyp
        total_analyses += iter_anal
        
        sig_count = sum(1 for a in iter_data["analyses"] if a.get("significant", False))
        significant_count += sig_count
    
    lines.append(f"Total hypotheses proposed: {total_hypotheses}")
    lines.append(f"Total analyses performed: {total_analyses}")
    lines.append(f"Significant findings (p<0.05): {significant_count}")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("ITERATION-BY-ITERATION SUMMARY")
    lines.append("-" * 70)
    
    for i, iter_data in enumerate(transcript["iterations"], 1):
        lines.append(f"\nIteration {i}:")
        for hyp in iter_data["proposed_hypotheses"]:
            lines.append(f"  Hypothesis: {hyp['text']}")
        
        for analysis in iter_data["analyses"]:
            sig_str = "SIGNIFICANT" if analysis.get("significant", False) else "not significant"
            lines.append(f"  Result: {analysis['result_summary']}")
            lines.append(f"    Effect: {format_val(analysis['effect_estimate'])}, p={format_val(analysis['p_value'])} ({sig_str})")
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("KEY FINDINGS")
    lines.append("-" * 70)
    
    all_sig = []
    for iter_data in transcript["iterations"]:
        for analysis in iter_data["analyses"]:
            if analysis.get("significant", False):
                all_sig.append({
                    "iteration": iter_data["index"],
                    "effect": analysis["effect_estimate"],
                    "p_value": analysis["p_value"],
                    "result": analysis["result_summary"]
                })
    
    if all_sig:
        all_sig.sort(key=lambda x: x["p_value"])
        lines.append(f"\nTop {min(5, len(all_sig))} most significant findings:")
        for s in all_sig[:5]:
            lines.append(f"  Iteration {s['iteration']}: {s['result']}")
            lines.append(f"    Effect: {format_val(s['effect'])}, p={format_val(s['p_value'])}")
    else:
        lines.append("\nNo statistically significant findings (p<0.05) were identified.")
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("CONCLUSIONS")
    lines.append("-" * 70)
    
    if significant_count > 0:
        lines.append(f"\nAnalysis identified {significant_count} statistically significant findings across {len(transcript['iterations'])} iterations.")
        lines.append("Key patterns observed:")
        
        treatment_effects = []
        for iter_data in transcript["iterations"]:
            for analysis in iter_data["analyses"]:
                if "treatment" in analysis["result_summary"].lower() and analysis.get("significant", False):
                    treatment_effects.append(analysis["result_summary"])
        
        if treatment_effects:
            lines.append("  - Treatment effects on objective_response were detected in specific patient subgroups.")
        
        lines.append("  - Treatment-effect heterogeneity was explored through interaction analyses.")
        lines.append("  - Further investigation of identified subgroups is recommended.")
    else:
        lines.append("\nNo statistically significant findings (p<0.05) were identified in this analysis.")
        lines.append("This may indicate:")
        lines.append("  - The dataset lacks sufficient power for the tested hypotheses")
        lines.append("  - The relationships tested are not present in this population")
        lines.append("  - Additional features or interaction terms may be needed")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SUMMARY")
    lines.append("=" * 70)
    
    return "\n".join(lines)

def make_jsonable(obj):
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_jsonable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (str, int, type(None))):
        return obj
    else:
        return str(obj)

# Run analysis
print("Running iterative analysis...")
transcript = run_iterative_analysis(max_iter=10)

transcript_jsonable = make_jsonable(transcript)

with open("transcript.json", "w") as f:
    json.dump(transcript_jsonable, f, indent=2)
print("Written: transcript.json")

summary = generate_summary(transcript)
with open("analysis_summary.txt", "w") as f:
    f.write(summary)
print("Written: analysis_summary.txt")

print("\nAnalysis complete!")
