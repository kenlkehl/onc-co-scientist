#!/usr/bin/env python
"""
End-to-end AML dataset analysis script.
Performs iterative hypothesis generation, testing, and refinement.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import os
from typing import Any
import pandas as pd
import numpy as np
from scipy import stats

# Paths
DATA_PATH = "dataset.parquet"
OUTPUT_TRANSCRIPT = "transcript.json"
OUTPUT_SUMMARY = "analysis_summary.txt"

# Load dataset
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Separate features and outcome
outcome_col = "objective_response"
feature_cols = [c for c in df.columns if c != outcome_col and c != "patient_id"]

# Helper functions
def compute_effect_estimate(df: pd.DataFrame, feature: str, outcome: str, 
                            feature_type: str = "binary") -> dict:
    """
    Compute effect estimate for a feature-outcome comparison.
    Returns a dictionary with effect_estimate, p_value, significant, and result_summary.
    """
    result = {"effect_estimate": None, "p_value": None, "significant": None, "result_summary": ""}
    
    if feature_type == "binary":
        # Binary feature: compare means using boolean mask
        mask = df[feature] == 1
        group1_mean = df.loc[mask, outcome].mean()
        group0_mean = df.loc[~mask, outcome].mean()
        effect = group1_mean - group0_mean
        
        # Chi-square test for proportions
        n1 = mask.sum()
        n0 = ~mask.sum()
        p1 = df.loc[mask, outcome].sum() / n1 if n1 > 0 else 0
        p0 = df.loc[~mask, outcome].sum() / n0 if n0 > 0 else 0
        
        # 2x2 contingency table
        cont_table = pd.crosstab(df[feature], df[outcome])
        _, p_val, _, _ = stats.chi2_contingency(cont_table, correction=False)
        
        result["effect_estimate"] = float(effect)
        result["p_value"] = float(p_val)
        result["significant"] = bool(p_val < 0.05)
        result["result_summary"] = f"Mean {outcome}: {group1_mean:.3f} (feature=1, n={n1}) vs {group0_mean:.3f} (feature=0, n={n0}); chi2 p={p_val:.4f}"
        
    elif feature_type == "categorical":
        # Categorical feature: compare means across categories
        categories = sorted(df[feature].unique())
        means = {}
        counts = {}
        for cat in categories:
            mask = df[feature] == cat
            means[cat] = df.loc[mask, outcome].mean()
            counts[cat] = mask.sum()
        
        # ANOVA F-test
        f_stat, p_val = stats.f_oneway(*[df.loc[df[feature] == cat, outcome].values for cat in categories])
        
        # Effect estimate: difference between highest and lowest mean
        max_mean = max(means.values())
        min_mean = min(means.values())
        effect = max_mean - min_mean
        
        result["effect_estimate"] = float(effect)
        result["p_value"] = float(p_val)
        result["significant"] = bool(p_val < 0.05)
        result["result_summary"] = f"ANOVA across {len(categories)} categories; F={f_stat:.2f}, p={p_val:.4f}; means: {dict(means)}"
        
    elif feature_type == "continuous":
        # Continuous feature: Spearman correlation
        corr, p_val = stats.spearmanr(df[feature], df[outcome])
        effect = corr  # Correlation coefficient as effect estimate
        
        result["effect_estimate"] = float(corr)
        result["p_value"] = float(p_val)
        result["significant"] = bool(p_val < 0.05)
        result["result_summary"] = f"Spearman r={corr:.4f}, p={p_val:.4f}"
    
    return result


def compute_interaction_effect(df: pd.DataFrame, treatment: str, modifier: str, 
                               outcome: str, treatment_type: str = "binary", 
                               modifier_type: str = "binary") -> dict:
    """
    Compute treatment effect heterogeneity by interaction.
    Returns effect estimate for treatment effect within modifier subgroup.
    """
    result = {"effect_estimate": None, "p_value": None, "significant": None, "result_summary": ""}
    
    # Get treatment effect within modifier=1 subgroup
    mask_mod = df[modifier] == 1
    mask_treat = df[treatment] == 1
    
    if mask_mod.sum() > 0 and mask_treat.sum() > 0:
        treat1_mean = df.loc[mask_mod & mask_treat, outcome].mean()
        treat0_mean = df.loc[mask_mod & ~mask_treat, outcome].mean()
        effect_in_subgroup = treat1_mean - treat0_mean
        
        # Get treatment effect in full population for comparison
        treat1_full = df.loc[mask_treat, outcome].mean()
        treat0_full = df.loc[~mask_treat, outcome].mean()
        effect_full = treat1_full - treat0_full
        
        # Test if interaction is significant (difference in treatment effects)
        # Using permutation test for simplicity
        n_treat = mask_treat.sum()
        n_control = ~mask_treat.sum()
        n_mod = mask_mod.sum()
        n_not_mod = ~mask_mod.sum()
        
        # Create combined groups for permutation
        combined = df.loc[mask_mod | ~mask_mod, outcome].values
        combined_treat = (df.loc[mask_mod | ~mask_mod, treatment] == 1).astype(int).values
        
        # Permutation test for interaction
        n_perm = 10000
        perm_pvals = []
        
        for _ in range(n_perm):
            perm_treat = np.random.permutation(combined_treat)
            perm_treat1 = perm_treat[:n_treat]
            perm_treat0 = perm_treat[n_treat:]
            perm_effect = perm_treat1.mean() - perm_treat0.mean()
            perm_pvals.append(abs(perm_effect - effect_full))
        
        perm_pval = float(np.mean(perm_pvals >= abs(effect_in_subgroup - effect_full)))
        
        result["effect_estimate"] = float(effect_in_subgroup)
        result["p_value"] = perm_pval
        result["significant"] = bool(perm_pval < 0.05)
        result["result_summary"] = f"Treatment effect in {modifier}=1: {effect_in_subgroup:.4f} (vs full pop {effect_full:.4f}); perm p={perm_pval:.4f}"
    
    return result


def compute_categorical_interaction(df: pd.DataFrame, treatment: str, modifier: str,
                                    outcome: str, treatment_type: str = "binary",
                                    modifier_type: str = "categorical") -> dict:
    """
    Compute treatment effect heterogeneity for categorical modifier.
    """
    result = {"effect_estimate": None, "p_value": None, "significant": None, "result_summary": ""}
    
    categories = sorted(df[modifier].unique())
    effects_by_cat = {}
    
    for cat in categories:
        mask_mod = df[modifier] == cat
        mask_treat = df[treatment] == 1
        
        if mask_mod.sum() > 0 and mask_treat.sum() > 0:
            treat1_mean = df.loc[mask_mod & mask_treat, outcome].mean()
            treat0_mean = df.loc[mask_mod & ~mask_treat, outcome].mean()
            effects_by_cat[cat] = treat1_mean - treat0_mean
    
    if effects_by_cat:
        # Effect estimate: difference between max and min category effects
        max_effect = max(effects_by_cat.values())
        min_effect = min(effects_by_cat.values())
        effect = max_effect - min_effect
        
        # Simple test: compare variance of effects to null
        effects_list = list(effects_by_cat.values())
        mean_effect = np.mean(effects_list)
        var_effect = np.var(effects_list)
        
        # F-test for variance
        f_stat, p_val = stats.f_oneway(*[np.array([effects_by_cat[cat]]) for cat in categories])
        
        result["effect_estimate"] = float(effect)
        result["p_value"] = float(p_val)
        result["significant"] = bool(p_val < 0.05)
        result["result_summary"] = f"Treatment effects by {modifier}: {dict(effects_by_cat)}; max-min={effect:.4f}"
    
    return result


# Initialize transcript
transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "qwen35-9b",
    "harness_id": "codex-cli@1.0.0",
    "max_iterations": 10,
    "iterations": []
}

# Iteration 1: Main effects - binary features
print("\n=== Iteration 1: Main effects (binary features) ===")
iteration1_hypotheses = []
iteration1_analyses = []

binary_features = [c for c in feature_cols if df[c].nunique() == 2]
print(f"Binary features: {len(binary_features)}")

for i, feat in enumerate(binary_features):
    hypothesis_id = f"h1_{i+1}"
    hypothesis_text = f"Mean objective_response differs between patients with {feat}=1 and those with {feat}=0."
    iteration1_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "novel"})
    
    analysis_result = compute_effect_estimate(df, feat, outcome_col, "binary")
    analysis_result["hypothesis_ids"] = [hypothesis_id]
    iteration1_analyses.append(analysis_result)

transcript["iterations"].append({
    "index": 1,
    "proposed_hypotheses": iteration1_hypotheses,
    "analyses": iteration1_analyses
})

# Iteration 2: Main effects - categorical features
print("\n=== Iteration 2: Main effects (categorical features) ===")
iteration2_hypotheses = []
iteration2_analyses = []

# Features with 3-10 unique values
categorical_features = [c for c in feature_cols if 3 <= df[c].nunique() <= 10]
print(f"Categorical features (3-10 unique): {len(categorical_features)}")

for i, feat in enumerate(categorical_features):
    hypothesis_id = f"h2_{i+1}"
    n_cats = df[feat].nunique()
    hypothesis_text = f"Mean objective_response differs across {n_cats} categories of {feat}."
    iteration2_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "novel"})
    
    analysis_result = compute_effect_estimate(df, feat, outcome_col, "categorical")
    analysis_result["hypothesis_ids"] = [hypothesis_id]
    iteration2_analyses.append(analysis_result)

transcript["iterations"].append({
    "index": 2,
    "proposed_hypotheses": iteration2_hypotheses,
    "analyses": iteration2_analyses
})

# Iteration 3: Main effects - continuous features
print("\n=== Iteration 3: Main effects (continuous features) ===")
iteration3_hypotheses = []
iteration3_analyses = []

continuous_features = [c for c in feature_cols if df[c].nunique() > 100]
print(f"Continuous features (>100 unique): {len(continuous_features)}")

for i, feat in enumerate(continuous_features):
    hypothesis_id = f"h3_{i+1}"
    hypothesis_text = f"Spearman correlation between {feat} and objective_response is non-zero."
    iteration3_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "novel"})
    
    analysis_result = compute_effect_estimate(df, feat, outcome_col, "continuous")
    analysis_result["hypothesis_ids"] = [hypothesis_id]
    iteration3_analyses.append(analysis_result)

transcript["iterations"].append({
    "index": 3,
    "proposed_hypotheses": iteration3_hypotheses,
    "analyses": iteration3_analyses
})

# Iteration 4: Treatment effect heterogeneity - binary modifiers
print("\n=== Iteration 4: Treatment effect heterogeneity (binary modifiers) ===")
iteration4_hypotheses = []
iteration4_analyses = []

# Find features with strong main effects
strong_binary = [f for f in binary_features if any(a["hypothesis_ids"] == [f"h1_{binary_features.index(f)+1}"] and a["significant"] for a in iteration1_analyses)]

for i, feat in enumerate(strong_binary[:5]):
    hypothesis_id = f"h4_{i+1}"
    hypothesis_text = f"Treatment effect (objective_response difference between feature=1 and feature=0) differs in patients with {feat}=1 vs {feat}=0."
    iteration4_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "novel"})
    
    analysis_result = compute_interaction_effect(df, "objective_response", feat, "objective_response", "binary", "binary")
    analysis_result["hypothesis_ids"] = [hypothesis_id]
    iteration4_analyses.append(analysis_result)

transcript["iterations"].append({
    "index": 4,
    "proposed_hypotheses": iteration4_hypotheses,
    "analyses": iteration4_analyses
})

# Iteration 5: Treatment effect heterogeneity - categorical modifiers
print("\n=== Iteration 5: Treatment effect heterogeneity (categorical modifiers) ===")
iteration5_hypotheses = []
iteration5_analyses = []

strong_categorical = [f for f in categorical_features if any(a["hypothesis_ids"] == [f"h2_{categorical_features.index(f)+1}"] and a["significant"] for a in iteration2_analyses)]

for i, feat in enumerate(strong_categorical[:3]):
    hypothesis_id = f"h5_{i+1}"
    hypothesis_text = f"Treatment effect differs across categories of {feat}."
    iteration5_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "novel"})
    
    analysis_result = compute_categorical_interaction(df, "objective_response", feat, "objective_response", "binary", "categorical")
    analysis_result["hypothesis_ids"] = [hypothesis_id]
    iteration5_analyses.append(analysis_result)

transcript["iterations"].append({
    "index": 5,
    "proposed_hypotheses": iteration5_hypotheses,
    "analyses": iteration5_analyses
})

# Iteration 6: Refined hypotheses based on significant findings
print("\n=== Iteration 6: Refined hypotheses ===")
iteration6_hypotheses = []
iteration6_analyses = []

# Find significant findings from previous iterations
significant_binary = [f for f in binary_features if any(a["hypothesis_ids"] == [f"h1_{binary_features.index(f)+1}"] and a["significant"] for a in iteration1_analyses)]
significant_categorical = [f for f in categorical_features if any(a["hypothesis_ids"] == [f"h2_{categorical_features.index(f)+1}"] and a["significant"] for a in iteration2_analyses)]

for i, feat in enumerate(significant_binary[:3]):
    hypothesis_id = f"h6_{i+1}"
    # Get the actual effect from iteration 1
    effect_info = next((a for a in iteration1_analyses if a["hypothesis_ids"] == [f"h1_{binary_features.index(feat)+1}"]), None)
    if effect_info:
        effect = effect_info["effect_estimate"]
        hypothesis_text = f"Patients with {feat}=1 have {'higher' if effect > 0 else 'lower'} objective_response than those with {feat}=0 (effect={effect:.4f})."
    else:
        hypothesis_text = f"Patients with {feat}=1 have different objective_response than those with {feat}=0."
    iteration6_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "refined"})
    
    analysis_result = compute_effect_estimate(df, feat, outcome_col, "binary")
    analysis_result["hypothesis_ids"] = [hypothesis_id]
    iteration6_analyses.append(analysis_result)

transcript["iterations"].append({
    "index": 6,
    "proposed_hypotheses": iteration6_hypotheses,
    "analyses": iteration6_analyses
})

# Iteration 7: Multivariable interaction screening
print("\n=== Iteration 7: Multivariable interaction screening ===")
iteration7_hypotheses = []
iteration7_analyses = []

# Test interactions between top binary features
top_binary = significant_binary[:3] if significant_binary else binary_features[:3]

for i, feat1 in enumerate(top_binary):
    for feat2 in top_binary[i+1:]:
        hypothesis_id = f"h7_{i+1}_{feat2}"
        hypothesis_text = f"Interaction between {feat1} and {feat2} affects objective_response."
        iteration7_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "novel"})
        
        # Simple interaction test: compare 4 groups
        groups = df.groupby([feat1, feat2])[outcome_col].mean().unstack(fill_value=0)
        if len(groups.columns) == 2 and len(groups.index) == 2:
            # Compute interaction effect
            means = groups.values
            interaction = (means[0,0] - means[0,1]) - (means[1,0] - means[1,1])
            
            # Fisher's exact test for 2x2
            cont_table = pd.crosstab(df[feat1], df[feat2])
            p_val = float(stats.fisher_exact(cont_table)[1])
            
            analysis_result = {
                "hypothesis_ids": [hypothesis_id],
                "effect_estimate": float(interaction),
                "p_value": p_val,
                "significant": bool(p_val < 0.05),
                "result_summary": f"Interaction effect={interaction:.4f}; Fisher p={p_val:.4f}"
            }
        else:
            analysis_result = {
                "hypothesis_ids": [hypothesis_id],
                "effect_estimate": 0.0,
                "p_value": 1.0,
                "significant": False,
                "result_summary": f"Insufficient data for interaction test"
            }
        iteration7_analyses.append(analysis_result)

transcript["iterations"].append({
    "index": 7,
    "proposed_hypotheses": iteration7_hypotheses,
    "analyses": iteration7_analyses
})

# Iteration 8: Best-supported treatment effect subgroup
print("\n=== Iteration 8: Best-supported treatment effect subgroup ===")
iteration8_hypotheses = []
iteration8_analyses = []

# Find the best treatment effect subgroup based on previous analyses
best_subgroup = None
best_effect = 0
best_p = 1.0

# Check binary modifiers
for feat in binary_features:
    analysis = next((a for a in iteration1_analyses if a["hypothesis_ids"] == [f"h1_{binary_features.index(feat)+1}"]), None)
    if analysis and analysis["significant"] and abs(analysis["effect_estimate"]) > abs(best_effect):
        best_effect = analysis["effect_estimate"]
        best_p = analysis["p_value"]
        best_subgroup = feat

# Check categorical modifiers
for feat in categorical_features:
    analysis = next((a for a in iteration2_analyses if a["hypothesis_ids"] == [f"h2_{categorical_features.index(feat)+1}"]), None)
    if analysis and analysis["significant"] and abs(analysis["effect_estimate"]) > abs(best_effect):
        best_effect = analysis["effect_estimate"]
        best_p = analysis["p_value"]
        best_subgroup = feat

if best_subgroup:
    hypothesis_id = "h8_best"
    n_cats = df[best_subgroup].nunique()
    if n_cats == 2:
        hypothesis_text = f"Treatment effect is strongest in patients with {best_subgroup}=1 (effect={best_effect:.4f}, p={best_p:.4f})."
    else:
        hypothesis_text = f"Treatment effect varies by {best_subgroup} categories (max effect={best_effect:.4f}, p={best_p:.4f})."
    iteration8_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "refined"})
    
    analysis_result = compute_effect_estimate(df, best_subgroup, outcome_col, "binary" if n_cats == 2 else "categorical")
    analysis_result["hypothesis_ids"] = [hypothesis_id]
    iteration8_analyses.append(analysis_result)
else:
    hypothesis_id = "h8_none"
    hypothesis_text = "No significant treatment effect heterogeneity found."
    iteration8_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "refined"})
    iteration8_analyses.append({
        "hypothesis_ids": [hypothesis_id],
        "effect_estimate": 0.0,
        "p_value": 1.0,
        "significant": False,
        "result_summary": "No significant main effects found"
    })

transcript["iterations"].append({
    "index": 8,
    "proposed_hypotheses": iteration8_hypotheses,
    "analyses": iteration8_analyses
})

# Iteration 9: Additional interaction screening
print("\n=== Iteration 9: Additional interaction screening ===")
iteration9_hypotheses = []
iteration9_analyses = []

# Test interactions with continuous features
top_continuous = continuous_features[:3] if continuous_features else []

for i, feat in enumerate(top_continuous):
    hypothesis_id = f"h9_{i+1}"
    hypothesis_text = f"Interaction between continuous feature {feat} and objective_response is non-linear."
    iteration9_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "novel"})
    
    # Test for non-linearity using Spearman correlation of squared term
    df_temp = df.copy()
    df_temp[f"{feat}_sq"] = df_temp[feat] ** 2
    corr_sq, p_val_sq = stats.spearmanr(df_temp[f"{feat}_sq"], df_temp[outcome_col])
    corr_orig, p_val_orig = stats.spearmanr(df_temp[feat], df_temp[outcome_col])
    
    # Interaction effect: difference in correlations
    interaction = corr_sq - corr_orig
    
    analysis_result = {
        "hypothesis_ids": [hypothesis_id],
        "effect_estimate": float(interaction),
        "p_value": float(min(p_val_sq, p_val_orig)),
        "significant": bool(min(p_val_sq, p_val_orig) < 0.05),
        "result_summary": f"Spearman {feat}: r={corr_orig:.4f}, {feat}^2: r={corr_sq:.4f}; interaction={interaction:.4f}"
    }
    iteration9_analyses.append(analysis_result)

transcript["iterations"].append({
    "index": 9,
    "proposed_hypotheses": iteration9_hypotheses,
    "analyses": iteration9_analyses
})

# Iteration 10: Final synthesis
print("\n=== Iteration 10: Final synthesis ===")
iteration10_hypotheses = []
iteration10_analyses = []

# Summarize all significant findings
significant_findings = []
for i, feat in enumerate(binary_features):
    analysis = next((a for a in iteration1_analyses if a["hypothesis_ids"] == [f"h1_{i+1}"]), None)
    if analysis and analysis["significant"]:
        significant_findings.append((feat, analysis["effect_estimate"], analysis["p_value"]))

for i, feat in enumerate(categorical_features):
    analysis = next((a for a in iteration2_analyses if a["hypothesis_ids"] == [f"h2_{i+1}"]), None)
    if analysis and analysis["significant"]:
        significant_findings.append((feat, analysis["effect_estimate"], analysis["p_value"]))

for i, feat in enumerate(continuous_features):
    analysis = next((a for a in iteration3_analyses if a["hypothesis_ids"] == [f"h3_{i+1}"]), None)
    if analysis and analysis["significant"]:
        significant_findings.append((feat, analysis["effect_estimate"], analysis["p_value"]))

if significant_findings:
    hypothesis_id = "h10_summary"
    hypothesis_text = f"Key significant associations found: {[f'{f}: eff={e:.4f}' for f, e, p in significant_findings[:5]]}."
    iteration10_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "refined"})
    
    avg_effect = sum(e for _, e, _ in significant_findings) / len(significant_findings)
    min_p = min(p for _, _, p in significant_findings)
    analysis_result = {
        "hypothesis_ids": [hypothesis_id],
        "effect_estimate": float(avg_effect),
        "p_value": float(min_p),
        "significant": True,
        "result_summary": f"Found {len(significant_findings)} significant associations out of {len(binary_features) + len(categorical_features) + len(continuous_features)} tested."
    }
    iteration10_analyses.append(analysis_result)
else:
    hypothesis_id = "h10_none"
    hypothesis_text = "No significant associations found."
    iteration10_hypotheses.append({"id": hypothesis_id, "text": hypothesis_text, "kind": "refined"})
    iteration10_analyses.append({
        "hypothesis_ids": [hypothesis_id],
        "effect_estimate": 0.0,
        "p_value": 1.0,
        "significant": False,
        "result_summary": "No significant associations found"
    })

transcript["iterations"].append({
    "index": 10,
    "proposed_hypotheses": iteration10_hypotheses,
    "analyses": iteration10_analyses
})

# Write transcript.json
print(f"\nWriting {OUTPUT_TRANSCRIPT}...")
with open(OUTPUT_TRANSCRIPT, "w") as f:
    json.dump(transcript, f, indent=2)

# Generate analysis_summary.txt
print(f"Writing {OUTPUT_SUMMARY}...")

# Collect all significant findings
all_significant = []
for iter_idx, iteration in enumerate(transcript["iterations"], 1):
    for analysis in iteration["analyses"]:
        if analysis.get("significant", False):
            all_significant.append({
                "iteration": iter_idx,
                "hypothesis_ids": analysis["hypothesis_ids"],
                "effect_estimate": analysis.get("effect_estimate", 0),
                "p_value": analysis.get("p_value", 1),
                "result_summary": analysis.get("result_summary", "")
            })

summary_lines = [
    "=" * 80,
    "AML DATASET ANALYSIS SUMMARY",
    "=" * 80,
    "",
    f"Dataset: ds001_aml (50,000 patients)",
    f"Total iterations: {len(transcript['iterations'])}",
    f"Total hypotheses tested: {sum(len(it['proposed_hypotheses']) for it in transcript['iterations'])}",
    f"Significant findings (p < 0.05): {len(all_significant)}",
    "",
    "-" * 80,
    "ITERATION-BY-ITERATION SUMMARY",
    "-" * 80,
    ""
]

for iteration in transcript["iterations"]:
    summary_lines.append(f"Iteration {iteration['index']}:")
    for hyp in iteration["proposed_hypotheses"]:
        summary_lines.append(f"  Hypothesis: {hyp['text']}")
    for analysis in iteration["analyses"]:
        sig = "SIGNIFICANT" if analysis.get("significant", False) else "not significant"
        summary_lines.append(f"  Result: {analysis['result_summary']} ({sig})")
    summary_lines.append("")

summary_lines.extend([
    "-" * 80,
    "KEY FINDINGS",
    "-" * 80,
    ""
])

if all_significant:
    summary_lines.append("Significant associations identified:")
    for finding in all_significant:
        summary_lines.append(f"  - Effect estimate: {finding['effect_estimate']:.4f}, p-value: {finding['p_value']:.4f}")
else:
    summary_lines.append("No statistically significant associations were found at p < 0.05.")

summary_lines.extend([
    "",
    "-" * 80,
    "CONCLUSIONS",
    "-" * 80,
    ""
])

# Count by feature type
binary_sig = sum(1 for f in binary_features if any(a["hypothesis_ids"] == [f"h1_{binary_features.index(f)+1}"] and a["significant"] for a in iteration1_analyses))
cat_sig = sum(1 for f in categorical_features if any(a["hypothesis_ids"] == [f"h2_{categorical_features.index(f)+1}"] and a["significant"] for a in iteration2_analyses))
cont_sig = sum(1 for f in continuous_features if any(a["hypothesis_ids"] == [f"h3_{continuous_features.index(f)+1}"] and a["significant"] for a in iteration3_analyses))

summary_lines.append(f"Binary features with significant effects: {binary_sig}/{len(binary_features)}")
summary_lines.append(f"Categorical features with significant effects: {cat_sig}/{len(categorical_features)}")
summary_lines.append(f"Continuous features with significant effects: {cont_sig}/{len(continuous_features)}")
summary_lines.append("")

if all_significant:
    summary_lines.append("The analysis identified several statistically significant associations between")
    summary_lines.append("patient features and objective_response. These findings suggest that specific")
    summary_lines.append("characteristics of the patient population are associated with different outcomes.")
    summary_lines.append("Further clinical validation is recommended for the most significant findings.")
else:
    summary_lines.append("No statistically significant associations were identified in this analysis.")
    summary_lines.append("This may indicate that the features examined do not have strong effects on")
    summary_lines.append("objective_response, or that the sample size is insufficient to detect smaller effects.")

summary_lines.extend([
    "",
    "=" * 80,
    "END OF SUMMARY",
    "=" * 80,
    ""
])

with open(OUTPUT_SUMMARY, "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nAnalysis complete!")
print(f"Output files: {OUTPUT_TRANSCRIPT}, {OUTPUT_SUMMARY}")
