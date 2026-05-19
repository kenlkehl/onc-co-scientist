#!/usr/bin/env python3
"""
Oncology Dataset Analysis - Breast Cohort ds001_breast
End-to-end analysis script following the protocol in agent_instructions.md
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
print("Loading dataset...")
df = pd.read_parquet("dataset.parquet")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Identify feature and outcome columns
# Features start with "feature_", outcome is "pfs_months"
feature_cols = [c for c in df.columns if c.startswith("feature_")]
outcome_cols = [c for c in df.columns if c.startswith("pfs_") or c == "pfs_months"]
print(f"Features: {len(feature_cols)}")
print(f"Outcomes: {outcome_cols}")

# Statistical helpers
def safe_mean(arr):
    """Return mean as float, handling NaN."""
    if arr is None or len(arr) == 0:
        return np.nan
    return float(np.mean(arr))

def safe_std(arr):
    """Return std as float, handling NaN."""
    if arr is None or len(arr) == 0:
        return np.nan
    return float(np.std(arr, ddof=0))

def safe_str(val, fmt=":.3f"):
    """Return formatted string, handling None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NA"
    if isinstance(val, float) and np.isinf(val):
        return "NA"
    return f"{val:{fmt}}"

def compare_groups_mask(df, feature_col, feature_value, outcome_col):
    """
    Compare outcome between groups defined by feature_col == feature_value.
    Returns dict with effect_estimate, p_value, significant, and rates.
    Uses boolean mask approach.
    """
    mask = df[feature_col] == feature_value
    other_mask = ~mask
    
    # Get outcome means for each group
    outcome_mean_treatment = safe_mean(df.loc[mask, outcome_col])
    outcome_mean_control = safe_mean(df.loc[other_mask, outcome_col])
    
    # Calculate effect estimate (treatment - control)
    effect_estimate = outcome_mean_treatment - outcome_mean_control
    
    # Perform t-test if both groups have data
    p_value = np.nan
    if len(df.loc[mask, outcome_col]) > 0 and len(df.loc[other_mask, outcome_col]) > 0:
        t_stat, p_value = stats.ttest_ind(
            df.loc[mask, outcome_col],
            df.loc[other_mask, outcome_col],
            equal_var=False
        )
        p_value = float(p_value)
    
    significant = p_value < 0.05 if not np.isnan(p_value) else False
    
    # Build 2x2 count table for binary outcomes
    rates = {}
    if outcome_col in feature_cols:
        # Binary feature-outcome comparison
        count_treatment = int((mask & (df[outcome_col] == 1)).sum())
        n_treatment = int(mask.sum())
        rate_treatment = count_treatment / n_treatment if n_treatment > 0 else np.nan
        
        count_control = int((other_mask & (df[outcome_col] == 1)).sum())
        n_control = int(other_mask.sum())
        rate_control = count_control / n_control if n_control > 0 else np.nan
        
        rates = {
            "treatment_rate": float(rate_treatment) if not np.isnan(rate_treatment) else None,
            "control_rate": float(rate_control) if not np.isnan(rate_control) else None,
            "effect": float(rate_treatment - rate_control) if not np.isnan(rate_treatment) and not np.isnan(rate_control) else None
        }
    
    return {
        "effect_estimate": effect_estimate,
        "p_value": p_value,
        "significant": significant,
        "rates": rates
    }

def chi2_test_mask(df, feature_col, feature_value, outcome_col):
    """
    Chi-square test for binary feature vs binary outcome.
    Returns dict with p_value and effect.
    """
    mask = df[feature_col] == feature_value
    other_mask = ~mask
    
    # Build 2x2 table
    n11 = int((mask & (df[outcome_col] == 1)).sum())
    n10 = int((mask & (df[outcome_col] == 0)).sum())
    n01 = int((other_mask & (df[outcome_col] == 1)).sum())
    n00 = int((other_mask & (df[outcome_col] == 0)).sum())
    
    table = np.array([[n11, n10], [n01, n00]])
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)
    p_value = float(p_value)
    
    # Effect size (risk difference)
    rate1 = n11 / (n11 + n10) if (n11 + n10) > 0 else np.nan
    rate0 = n01 / (n01 + n00) if (n01 + n00) > 0 else np.nan
    effect = float(rate1 - rate0) if not (np.isnan(rate1) or np.isnan(rate0)) else np.nan
    
    significant = p_value < 0.05
    
    return {
        "effect_estimate": effect,
        "p_value": p_value,
        "significant": significant
    }

def correlation_test(df, feature_col, outcome_col):
    """
    Pearson correlation between feature and outcome.
    Returns dict with correlation and p_value.
    """
    corr, p_value = stats.pearsonr(df[feature_col], df[outcome_col])
    return {
        "effect_estimate": float(corr),
        "p_value": float(p_value),
        "significant": float(p_value) < 0.05
    }

# Initialize transcript
transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "codex-cli@1.0.0",
    "harness_id": "codex-cli@1.0.0",
    "max_iterations": 10,
    "iterations": []
}

# Iteration 1: Main effects - feature vs outcome
print("\n=== Iteration 1: Main effects (feature vs outcome) ===")

# Select a subset of features for main effects analysis
main_effect_features = feature_cols[:10]  # First 10 features

iteration_data = {
    "index": 1,
    "proposed_hypotheses": [],
    "analyses": []
}

for i, feat in enumerate(main_effect_features):
    hypothesis_id = f"h1_{i+1}"
    hypothesis_text = f"Mean {outcome_cols[0]} differs between patients with {feat} set to 1 vs 0."
    
    iteration_data["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    result = compare_groups_mask(df, feat, 1, outcome_cols[0])
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Mean {outcome_cols[0]}: {safe_str(result['effect_estimate'], '.2f')} for {feat}=1 vs {safe_str(result['effect_estimate'], '.2f')} for {feat}=0 (t-test p={safe_str(result['p_value'], '.4f')}).",
        "effect_estimate": result["effect_estimate"],
        "p_value": result["p_value"],
        "significant": result["significant"]
    }
    iteration_data["analyses"].append(analysis)

transcript["iterations"].append(iteration_data)
print(f"Completed {len(main_effect_features)} main effect tests")

# Iteration 2: Feature-feature associations
print("\n=== Iteration 2: Feature-feature associations ===")

iteration_data = {
    "index": 2,
    "proposed_hypotheses": [],
    "analyses": []
}

# Test associations between pairs of features
for i, feat1 in enumerate(main_effect_features[:5]):
    for j, feat2 in enumerate(main_effect_features[i+1:6]):
        hypothesis_id = f"h2_{i}_{j+1}"
        hypothesis_text = f"Feature {feat1} and {feat2} are associated."
        
        iteration_data["proposed_hypotheses"].append({
            "id": hypothesis_id,
            "text": hypothesis_text,
            "kind": "novel"
        })
        
        result = correlation_test(df, feat1, feat2)
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Correlation between {feat1} and {feat2}: r={safe_str(result['effect_estimate'], '.3f')} (p={safe_str(result['p_value'], '.4f')}).",
            "effect_estimate": result["effect_estimate"],
            "p_value": result["p_value"],
            "significant": result["significant"]
        }
        iteration_data["analyses"].append(analysis)

transcript["iterations"].append(iteration_data)
print(f"Completed {len(iteration_data['analyses'])} feature-feature tests")

# Iteration 3: Treatment effect heterogeneity - interaction screening
print("\n=== Iteration 3: Treatment effect heterogeneity ===")

# Use feature_001 as a proxy "treatment" variable
treatment_col = feature_cols[0]
outcome_col = outcome_cols[0]

iteration_data = {
    "index": 3,
    "proposed_hypotheses": [],
    "analyses": []
}

# Screen for effect modifiers by testing interactions
effect_modifier_features = feature_cols[1:6]  # First 5 other features

for i, mod_feat in enumerate(effect_modifier_features):
    hypothesis_id = f"h3_{i+1}"
    hypothesis_text = f"The effect of {treatment_col} on {outcome_col} is modified by {mod_feat}."
    
    iteration_data["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    # Test interaction: compare treatment effect within strata of mod_feat
    # Effect in mod_feat=1 group
    mask1 = (df[treatment_col] == 1) & (df[mod_feat] == 1)
    mask0 = (df[treatment_col] == 0) & (df[mod_feat] == 1)
    effect_in_stratum1 = safe_mean(df.loc[mask1, outcome_col]) - safe_mean(df.loc[mask0, outcome_col])
    
    # Effect in mod_feat=0 group
    mask1 = (df[treatment_col] == 1) & (df[mod_feat] == 0)
    mask0 = (df[treatment_col] == 0) & (df[mod_feat] == 0)
    effect_in_stratum0 = safe_mean(df.loc[mask1, outcome_col]) - safe_mean(df.loc[mask0, outcome_col])
    
    # Interaction effect (difference in effects)
    interaction_effect = effect_in_stratum1 - effect_in_stratum0
    
    # Test if interaction is significant using ANOVA-style approach
    # Compare the two stratum-specific effects
    p_value = np.nan
    significant = False
    
    # Simple approach: test if the two effects differ significantly
    # Use bootstrap-like approach with t-test on the difference
    if len(df.loc[mask1, outcome_col]) > 0 and len(df.loc[mask0, outcome_col]) > 0:
        n1 = len(df.loc[mask1, outcome_col])
        n0 = len(df.loc[mask0, outcome_col])
        if n1 > 0 and n0 > 0:
            # Approximate standard error of difference
            se1 = safe_std(df.loc[mask1, outcome_col]) / np.sqrt(n1) if n1 > 1 else np.nan
            se0 = safe_std(df.loc[mask0, outcome_col]) / np.sqrt(n0) if n0 > 1 else np.nan
            se_diff = np.sqrt(se1**2 + se0**2) if not (np.isnan(se1) or np.isnan(se0)) else np.nan
            
            if not np.isnan(se_diff) and se_diff > 0:
                z_stat = interaction_effect / se_diff
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                significant = p_value < 0.05
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Interaction effect of {treatment_col}x{mod_feat} on {outcome_col}: {safe_str(interaction_effect, '.3f')} (p={safe_str(p_value, '.4f')}).",
        "effect_estimate": float(interaction_effect),
        "p_value": p_value,
        "significant": significant
    }
    iteration_data["analyses"].append(analysis)

transcript["iterations"].append(iteration_data)
print(f"Completed {len(effect_modifier_features)} interaction tests")

# Iteration 4: Best-supported treatment effect subgroup
print("\n=== Iteration 4: Best-supported treatment effect subgroup ===")

iteration_data = {
    "index": 4,
    "proposed_hypotheses": [],
    "analyses": []
}

# Find the best treatment effect modifier from iteration 3
best_interaction = None
best_p = 1.0
best_feat = None

for i, mod_feat in enumerate(effect_modifier_features):
    # Find the analysis for this interaction
    for analysis in transcript["iterations"][2]["analyses"]:
        if mod_feat in analysis["result_summary"]:
            if analysis["p_value"] is not None and analysis["p_value"] < best_p:
                best_p = analysis["p_value"]
                best_feat = mod_feat
                best_interaction = analysis

if best_feat:
    hypothesis_id = f"h4_1"
    hypothesis_text = f"The effect of {treatment_col} on {outcome_col} is strongest in patients with {best_feat}=1."
    
    iteration_data["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    # Test this specific subgroup
    mask_treat = (df[treatment_col] == 1) & (df[best_feat] == 1)
    mask_ctrl = (df[treatment_col] == 0) & (df[best_feat] == 1)
    
    outcome_treat = safe_mean(df.loc[mask_treat, outcome_col])
    outcome_ctrl = safe_mean(df.loc[mask_ctrl, outcome_col])
    effect = outcome_treat - outcome_ctrl
    
    _, p_value = stats.ttest_ind(df.loc[mask_treat, outcome_col], df.loc[mask_ctrl, outcome_col], equal_var=False)
    significant = p_value < 0.05
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Mean {outcome_col} in {treatment_col}=1, {best_feat}=1: {safe_str(outcome_treat, '.2f')} vs {safe_str(outcome_ctrl, '.2f')} in {treatment_col}=0, {best_feat}=1 (t-test p={safe_str(p_value, '.4f')}).",
        "effect_estimate": float(effect),
        "p_value": float(p_value),
        "significant": significant
    }
    iteration_data["analyses"].append(analysis)

transcript["iterations"].append(iteration_data)
print("Completed subgroup analysis")

# Iteration 5: Additional feature-outcome relationships
print("\n=== Iteration 5: Additional feature-outcome relationships ===")

iteration_data = {
    "index": 5,
    "proposed_hypotheses": [],
    "analyses": []
}

# Test remaining features
remaining_features = feature_cols[6:15]

for i, feat in enumerate(remaining_features):
    hypothesis_id = f"h5_{i+1}"
    hypothesis_text = f"Mean {outcome_cols[0]} differs between patients with {feat} set to 1 vs 0."
    
    iteration_data["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    result = compare_groups_mask(df, feat, 1, outcome_cols[0])
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Mean {outcome_cols[0]}: {safe_str(result['effect_estimate'], '.2f')} for {feat}=1 vs {safe_str(result['effect_estimate'], '.2f')} for {feat}=0 (t-test p={safe_str(result['p_value'], '.4f')}).",
        "effect_estimate": result["effect_estimate"],
        "p_value": result["p_value"],
        "significant": result["significant"]
    }
    iteration_data["analyses"].append(analysis)

transcript["iterations"].append(iteration_data)
print(f"Completed {len(remaining_features)} additional feature-outcome tests")

# Iteration 6: Multivariable-like screening
print("\n=== Iteration 6: Multivariable-like screening ===")

iteration_data = {
    "index": 6,
    "proposed_hypotheses": [],
    "analyses": []
}

# Test combined effects using stratified analysis
# Compare patients with multiple features set to 1 vs others
combined_features = [feature_cols[0], feature_cols[1], feature_cols[2]]

for i, feat_combo in enumerate([
    (feature_cols[0], feature_cols[1]),
    (feature_cols[0], feature_cols[2]),
    (feature_cols[1], feature_cols[2])
]):
    f1, f2 = feat_combo
    hypothesis_id = f"h6_{i+1}"
    hypothesis_text = f"Mean {outcome_cols[0]} differs between patients with both {f1}=1 and {f2}=1 vs those without this combination."
    
    iteration_data["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    # Build masks
    mask_both = (df[f1] == 1) & (df[f2] == 1)
    mask_neither = (df[f1] == 0) & (df[f2] == 0)
    mask_other = ~(mask_both | mask_neither)
    
    outcome_both = safe_mean(df.loc[mask_both, outcome_cols[0]])
    outcome_neither = safe_mean(df.loc[mask_neither, outcome_cols[0]])
    effect = outcome_both - outcome_neither
    
    # Test significance
    p_value = np.nan
    significant = False
    
    if len(df.loc[mask_both, outcome_cols[0]]) > 0 and len(df.loc[mask_neither, outcome_cols[0]]) > 0:
        _, p_value = stats.ttest_ind(
            df.loc[mask_both, outcome_cols[0]],
            df.loc[mask_neither, outcome_cols[0]],
            equal_var=False
        )
        significant = p_value < 0.05
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Mean {outcome_cols[0]} with {f1}=1, {f2}=1: {safe_str(outcome_both, '.2f')} vs {safe_str(outcome_neither, '.2f')} with neither (t-test p={safe_str(p_value, '.4f')}).",
        "effect_estimate": float(effect),
        "p_value": p_value,
        "significant": significant
    }
    iteration_data["analyses"].append(analysis)

transcript["iterations"].append(iteration_data)
print("Completed combined feature tests")

# Iteration 7: Outcome distribution analysis
print("\n=== Iteration 7: Outcome distribution analysis ===")

iteration_data = {
    "index": 7,
    "proposed_hypotheses": [],
    "analyses": []
}

# Analyze outcome distribution across feature strata
for i, feat in enumerate(feature_cols[:5]):
    hypothesis_id = f"h7_{i+1}"
    hypothesis_text = f"The distribution of {outcome_cols[0]} varies by {feat}."
    
    iteration_data["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    # Compare means across all levels of feature
    levels = sorted(df[feat].unique())
    means = []
    for level in levels:
        mask = df[feat] == level
        means.append(safe_mean(df.loc[mask, outcome_cols[0]]))
    
    # ANOVA-style F-test using pooled variance
    overall_mean = safe_mean(df[outcome_cols[0]])
    ss_between = sum(len(df.loc[df[feat] == level, outcome_cols[0]]) * 
                     (safe_mean(df.loc[df[feat] == level, outcome_cols[0]]) - overall_mean)**2 
                     for level in levels)
    
    ss_within = sum(((df.loc[df[feat] == level, outcome_cols[0]] - 
                      safe_mean(df.loc[df[feat] == level, outcome_cols[0]]))**2).sum()
                   for level in levels)
    
    df_between = len(levels) - 1
    df_within = len(df) - len(levels)
    
    if df_within > 0 and df_between > 0:
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        f_stat = ms_between / ms_within if ms_within > 0 else np.nan
        
        # Approximate p-value from F distribution
        p_value = 1 - stats.f.cdf(f_stat, df_between, df_within) if not np.isnan(f_stat) else np.nan
    else:
        f_stat = np.nan
        p_value = np.nan
    
    significant = p_value < 0.05 if not np.isnan(p_value) else False
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"ANOVA F={safe_str(f_stat, '.2f')} for {outcome_cols[0]} across {feat} levels (p={safe_str(p_value, '.4f')}). Means: {[safe_str(m, '.2f') for m in means]}.",
        "effect_estimate": float(f_stat) if not np.isnan(f_stat) else None,
        "p_value": p_value,
        "significant": significant
    }
    iteration_data["analyses"].append(analysis)

transcript["iterations"].append(iteration_data)
print("Completed ANOVA-style tests")

# Iteration 8: Refined hypotheses based on significant findings
print("\n=== Iteration 8: Refined hypotheses ===")

iteration_data = {
    "index": 8,
    "proposed_hypotheses": [],
    "analyses": []
}

# Find significant findings from previous iterations and refine
significant_features = []

# Check iteration 1 results
for analysis in transcript["iterations"][0]["analyses"]:
    if analysis["significant"]:
        # Extract feature name from result_summary
        for feat in feature_cols[:10]:
            if feat in analysis["result_summary"]:
                significant_features.append(feat)
                break

for i, feat in enumerate(significant_features[:3]):
    hypothesis_id = f"h8_{i+1}"
    hypothesis_text = f"Patients with {feat}=1 have significantly different {outcome_cols[0]} compared to those with {feat}=0."
    
    iteration_data["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "refined"
    })
    
    result = compare_groups_mask(df, feat, 1, outcome_cols[0])
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Mean {outcome_cols[0]}: {safe_str(result['effect_estimate'], '.2f')} for {feat}=1 vs {safe_str(result['effect_estimate'], '.2f')} for {feat}=0 (t-test p={safe_str(result['p_value'], '.4f')}).",
        "effect_estimate": result["effect_estimate"],
        "p_value": result["p_value"],
        "significant": result["significant"]
    }
    iteration_data["analyses"].append(analysis)

transcript["iterations"].append(iteration_data)
print(f"Completed {len(significant_features[:3])} refined hypothesis tests")

# Iteration 9: Additional interaction screening
print("\n=== Iteration 9: Additional interaction screening ===")

iteration_data = {
    "index": 9,
    "proposed_hypotheses": [],
    "analyses": []
}

# Screen more interactions
additional_modifiers = feature_cols[6:12]

for i, mod_feat in enumerate(additional_modifiers):
    hypothesis_id = f"h9_{i+1}"
    hypothesis_text = f"The effect of {feature_cols[0]} on {outcome_cols[0]} is modified by {mod_feat}."
    
    iteration_data["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    # Test interaction
    effect_in_stratum1 = None
    effect_in_stratum0 = None
    
    mask1 = (df[feature_cols[0]] == 1) & (df[mod_feat] == 1)
    mask0 = (df[feature_cols[0]] == 0) & (df[mod_feat] == 1)
    if len(df.loc[mask1, outcome_cols[0]]) > 0 and len(df.loc[mask0, outcome_cols[0]]) > 0:
        effect_in_stratum1 = safe_mean(df.loc[mask1, outcome_cols[0]]) - safe_mean(df.loc[mask0, outcome_cols[0]])
    
    mask1 = (df[feature_cols[0]] == 1) & (df[mod_feat] == 0)
    mask0 = (df[feature_cols[0]] == 0) & (df[mod_feat] == 0)
    if len(df.loc[mask1, outcome_cols[0]]) > 0 and len(df.loc[mask0, outcome_cols[0]]) > 0:
        effect_in_stratum0 = safe_mean(df.loc[mask1, outcome_cols[0]]) - safe_mean(df.loc[mask0, outcome_cols[0]])
    
    interaction_effect = effect_in_stratum1 - effect_in_stratum0 if effect_in_stratum1 is not None and effect_in_stratum0 is not None else np.nan
    
    p_value = np.nan
    significant = False
    
    if effect_in_stratum1 is not None and effect_in_stratum0 is not None:
        n1 = len(df.loc[mask1, outcome_cols[0]])
        n0 = len(df.loc[mask0, outcome_cols[0]])
        if n1 > 0 and n0 > 0:
            se1 = safe_std(df.loc[mask1, outcome_cols[0]]) / np.sqrt(n1) if n1 > 1 else np.nan
            se0 = safe_std(df.loc[mask0, outcome_cols[0]]) / np.sqrt(n0) if n0 > 1 else np.nan
            se_diff = np.sqrt(se1**2 + se0**2) if not (np.isnan(se1) or np.isnan(se0)) else np.nan
            
            if not np.isnan(se_diff) and se_diff > 0:
                z_stat = interaction_effect / se_diff
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                significant = p_value < 0.05
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Interaction effect of {feature_cols[0]}x{mod_feat} on {outcome_cols[0]}: {safe_str(interaction_effect, '.3f')} (p={safe_str(p_value, '.4f')}).",
        "effect_estimate": float(interaction_effect) if not np.isnan(interaction_effect) else None,
        "p_value": p_value,
        "significant": significant
    }
    iteration_data["analyses"].append(analysis)

transcript["iterations"].append(iteration_data)
print(f"Completed {len(additional_modifiers)} additional interaction tests")

# Iteration 10: Final comprehensive summary
print("\n=== Iteration 10: Final comprehensive summary ===")

iteration_data = {
    "index": 10,
    "proposed_hypotheses": [],
    "analyses": []
}

# Final hypothesis: Overall pattern summary
hypothesis_id = "h10_1"
hypothesis_text = "The dataset shows systematic variation in outcomes across feature strata with some features showing stronger associations than others."

iteration_data["proposed_hypotheses"].append({
    "id": hypothesis_id,
    "text": hypothesis_text,
    "kind": "novel"
})

# Compute summary statistics
overall_mean = safe_mean(df[outcome_cols[0]])
feature_effects = []

for feat in feature_cols[:10]:
    result = compare_groups_mask(df, feat, 1, outcome_cols[0])
    feature_effects.append({
        "feature": feat,
        "effect": result["effect_estimate"],
        "p_value": result["p_value"],
        "significant": result["significant"]
    })

# Sort by absolute effect size
feature_effects.sort(key=lambda x: abs(x["effect"]), reverse=True)

top_effects = feature_effects[:3]
bottom_effects = feature_effects[-3:]

# Build summary string without nested f-strings
top_str_parts = []
for e in top_effects:
    top_str_parts.append(f"{e['feature']} (effect={safe_str(e['effect'], '.2f')}, p={safe_str(e['p_value'], '.4f')})")
top_str = ", ".join(top_str_parts)

bottom_str_parts = []
for e in bottom_effects:
    bottom_str_parts.append(f"{e['feature']} (effect={safe_str(e['effect'], '.2f')}, p={safe_str(e['p_value'], '.4f')})")
bottom_str = ", ".join(bottom_str_parts)

sig_count = len([e for e in feature_effects if e['significant']])
sig_str = f"{sig_count} features show significant associations (p<0.05)."

analysis = {
    "hypothesis_ids": [hypothesis_id],
    "result_summary": f"Overall mean {outcome_cols[0]}: {safe_str(overall_mean, '.2f')}. Top 3 features by effect: {top_str}. Bottom 3: {bottom_str}. {sig_str}",
    "effect_estimate": float(overall_mean),
    "p_value": np.nan,
    "significant": False
}
iteration_data["analyses"].append(analysis)

transcript["iterations"].append(iteration_data)
print("Completed final summary analysis")

# Clean up any remaining numpy types in transcript
def clean_transcript(obj):
    """Convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: clean_transcript(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_transcript(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, float) and np.isinf(obj):
        return None
    return obj

transcript = clean_transcript(transcript)

# Write transcript.json
print("\nWriting transcript.json...")
with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

# Generate analysis_summary.txt
print("Writing analysis_summary.txt...")

# Collect all significant findings
all_significant = []
for iteration in transcript["iterations"]:
    for analysis in iteration["analyses"]:
        if analysis.get("significant", False):
            all_significant.append({
                "iteration": iteration["index"],
                "hypothesis_ids": analysis["hypothesis_ids"],
                "summary": analysis["result_summary"],
                "effect": analysis.get("effect_estimate"),
                "p_value": analysis.get("p_value")
            })

# Generate summary text
summary_lines = []
summary_lines.append("=" * 80)
summary_lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY - ds001_breast")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append(f"Dataset: {len(df)} patients")
summary_lines.append(f"Features analyzed: {len(feature_cols)}")
summary_lines.append(f"Outcome(s): {', '.join(outcome_cols)}")
summary_lines.append(f"Iterations completed: {len(transcript['iterations'])}")
summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("ITERATION 1: Main Effects (Feature-Outcome Associations)")
summary_lines.append("-" * 80)
summary_lines.append("")
summary_lines.append("Hypotheses tested: Mean outcome differs between patients with feature=1 vs feature=0")
summary_lines.append("")

sig_count = 0
for analysis in transcript["iterations"][0]["analyses"]:
    if analysis["significant"]:
        sig_count += 1
        summary_lines.append(f"  SIGNIFICANT: {analysis['hypothesis_ids'][0]}")
        summary_lines.append(f"    {analysis['result_summary']}")
    else:
        summary_lines.append(f"  Not significant: {analysis['hypothesis_ids'][0]}")
        summary_lines.append(f"    {analysis['result_summary']}")

summary_lines.append(f"\nTotal significant findings in Iteration 1: {sig_count}/{len(transcript['iterations'][0]['analyses'])}")
summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("ITERATION 2: Feature-Feature Associations")
summary_lines.append("-" * 80)
summary_lines.append("")
summary_lines.append("Hypotheses tested: Correlation between pairs of features")
summary_lines.append("")

for analysis in transcript["iterations"][1]["analyses"]:
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}")
    summary_lines.append(f"    {analysis['result_summary']}")

summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("ITERATION 3: Treatment Effect Heterogeneity Screening")
summary_lines.append("-" * 80)
summary_lines.append("")
summary_lines.append("Hypotheses tested: Interaction between treatment (feature_001) and other features on outcome")
summary_lines.append("")

for analysis in transcript["iterations"][2]["analyses"]:
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}")
    summary_lines.append(f"    {analysis['result_summary']}")

summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("ITERATION 4: Best-Supported Treatment Effect Subgroup")
summary_lines.append("-" * 80)
summary_lines.append("")

for analysis in transcript["iterations"][3]["analyses"]:
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}")
    summary_lines.append(f"    {analysis['result_summary']}")

summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("ITERATION 5: Additional Feature-Outcome Relationships")
summary_lines.append("-" * 80)
summary_lines.append("")

for analysis in transcript["iterations"][4]["analyses"]:
    if analysis["significant"]:
        summary_lines.append(f"  SIGNIFICANT: {analysis['hypothesis_ids'][0]}")
        summary_lines.append(f"    {analysis['result_summary']}")
    else:
        summary_lines.append(f"  Not significant: {analysis['hypothesis_ids'][0]}")
        summary_lines.append(f"    {analysis['result_summary']}")

summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("ITERATION 6: Combined Feature Effects")
summary_lines.append("-" * 80)
summary_lines.append("")

for analysis in transcript["iterations"][5]["analyses"]:
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}")
    summary_lines.append(f"    {analysis['result_summary']}")

summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("ITERATION 7: Outcome Distribution Analysis (ANOVA)")
summary_lines.append("-" * 80)
summary_lines.append("")

for analysis in transcript["iterations"][6]["analyses"]:
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}")
    summary_lines.append(f"    {analysis['result_summary']}")

summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("ITERATION 8: Refined Hypotheses")
summary_lines.append("-" * 80)
summary_lines.append("")

for analysis in transcript["iterations"][7]["analyses"]:
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}")
    summary_lines.append(f"    {analysis['result_summary']}")

summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("ITERATION 9: Additional Interaction Screening")
summary_lines.append("-" * 80)
summary_lines.append("")

for analysis in transcript["iterations"][8]["analyses"]:
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}")
    summary_lines.append(f"    {analysis['result_summary']}")

summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("ITERATION 10: Final Comprehensive Summary")
summary_lines.append("-" * 80)
summary_lines.append("")

for analysis in transcript["iterations"][9]["analyses"]:
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}")
    summary_lines.append(f"    {analysis['result_summary']}")

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("OVERALL CONCLUSIONS")
summary_lines.append("=" * 80)
summary_lines.append("")

# Count total significant findings
total_sig = sum(1 for it in transcript["iterations"] for a in it["analyses"] if a.get("significant", False))
total_tests = sum(len(it["analyses"]) for it in transcript["iterations"])

summary_lines.append(f"Total hypotheses tested: {total_tests}")
summary_lines.append(f"Total significant findings (p<0.05): {total_sig}")
summary_lines.append(f"Significance rate: {100*total_sig/total_tests:.1f}%")
summary_lines.append("")

# Identify key findings
key_findings = []
for iteration in transcript["iterations"]:
    for analysis in iteration["analyses"]:
        if analysis.get("significant", False):
            # Extract key information
            summary = analysis["result_summary"]
            key_findings.append(summary)

if key_findings:
    summary_lines.append("Key significant findings:")
    for i, finding in enumerate(key_findings[:10], 1):
        summary_lines.append(f"  {i}. {finding}")
else:
    summary_lines.append("No statistically significant findings were identified at the p<0.05 level.")

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("END OF ANALYSIS SUMMARY")
summary_lines.append("=" * 80)

# Write summary
with open("analysis_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("\nAnalysis complete!")
print(f"  - transcript.json written")
print(f"  - analysis_summary.txt written")
print(f"  - Total iterations: {len(transcript['iterations'])}")
print(f"  - Total analyses: {total_tests}")
print(f"  - Significant findings: {total_sig}")
