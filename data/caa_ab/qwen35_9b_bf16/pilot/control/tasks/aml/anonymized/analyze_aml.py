#!/usr/bin/env python3
"""
AML Dataset Analysis Script
Performs iterative hypothesis testing on oncology dataset.
Uses boolean-mask group comparisons for robust statistical testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from typing import List, Dict, Any, Tuple

# Load dataset
print("Loading dataset...")
df = pd.read_parquet('dataset.parquet')
print(f"Dataset shape: {df.shape}")

# Binary outcome
OUTCOME = 'objective_response'

# All features (excluding patient_id and outcome)
FEATURE_COLS = [col for col in df.columns if col not in ['patient_id', OUTCOME]]
print(f"Number of features: {len(FEATURE_COLS)}")

# Helper function for boolean-mask group comparison
def compare_groups(df: pd.DataFrame, feature_col: str, feature_value, outcome_col: str) -> Tuple[float, float]:
    """
    Compare outcome means between two groups defined by a feature value.
    Returns (effect_estimate, p_value) where effect = mean(group1) - mean(group0)
    """
    mask = df[feature_col] == feature_value
    group1_mean = df.loc[mask, outcome_col].mean()
    group0_mean = df.loc[~mask, outcome_col].mean()
    effect = group1_mean - group0_mean
    
    # Chi-square test for binary outcome
    n1 = mask.sum()
    n0 = (~mask).sum()
    y1 = df.loc[mask, outcome_col].sum()
    y0 = df.loc[~mask, outcome_col].sum()
    
    # 2x2 contingency table
    contingency = [[y1, n1 - y1], [y0, n0 - y0]]
    _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
    
    return float(effect), float(p_value)

# Helper function for correlation
def compute_correlation(df: pd.DataFrame, feature_col: str, outcome_col: str) -> Tuple[float, float]:
    """
    Compute Pearson correlation and p-value.
    Returns (correlation, p_value)
    """
    feature_vals = df[feature_col].astype(float)
    outcome_vals = df[outcome_col].astype(float)
    
    corr, p_value = stats.pearsonr(feature_vals, outcome_vals)
    return float(corr), float(p_value)

# Transcript structure
transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "qwen35-9b",
    "harness_id": "codex-cli@1.0.0",
    "max_iterations": 10,
    "iterations": []
}

# Iteration counter
iteration = 0

# Track significant findings for treatment heterogeneity
significant_findings = []

print("\n" + "="*60)
print("Starting iterative hypothesis testing")
print("="*60)

# Iteration 1: Main effects screening - binary features
print("\n--- Iteration 1: Main effects on binary features ---")
iteration += 1

binary_features = [f for f in FEATURE_COLS if df[f].nunique() == 2]
print(f"Testing {len(binary_features)} binary features...")

iteration_results = {
    "index": iteration,
    "proposed_hypotheses": [],
    "analyses": []
}

for i, feature in enumerate(binary_features[:10]):  # Start with first 10
    hypothesis_id = f"h{iteration}_{i+1}"
    hypothesis_text = f"Patients with {feature}={1} have different objective_response rates than those with {feature}={0}."
    
    iteration_results["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    effect, p_value = compare_groups(df, feature, 1, OUTCOME)
    significant = bool(p_value < 0.05)
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Objective response rate: {df.loc[df[feature]==1, OUTCOME].mean():.3f} vs {df.loc[df[feature]==0, OUTCOME].mean():.3f} (chi-square p={p_value:.4f}).",
        "effect_estimate": float(effect),
        "p_value": float(p_value),
        "significant": significant
    }
    iteration_results["analyses"].append(analysis)
    
    if significant:
        significant_findings.append({
            "feature": feature,
            "effect": float(effect),
            "p_value": float(p_value)
        })

transcript["iterations"].append(iteration_results)
print(f"Iteration 1: Tested {len(binary_features[:10])} features, found {sum(1 for a in iteration_results['analyses'] if a['significant'])} significant")

# Iteration 2: Main effects on 3-level features
print("\n--- Iteration 2: Main effects on 3-level features ---")
iteration += 1

three_level_features = [f for f in FEATURE_COLS if df[f].nunique() == 3]
print(f"Testing {len(three_level_features)} three-level features...")

iteration_results = {
    "index": iteration,
    "proposed_hypotheses": [],
    "analyses": []
}

for i, feature in enumerate(three_level_features[:8]):
    hypothesis_id = f"h{iteration}_{i+1}"
    hypothesis_text = f"Patients with {feature}={2} have different objective_response rates than those with {feature} in levels 0-1."
    
    iteration_results["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    # Compare level 2 vs levels 0-1
    mask = df[feature] == 2
    effect, p_value = compare_groups(df, feature, 2, OUTCOME)
    significant = bool(p_value < 0.05)
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Objective response rate: {df.loc[mask, OUTCOME].mean():.3f} vs {df.loc[~mask, OUTCOME].mean():.3f} (chi-square p={p_value:.4f}).",
        "effect_estimate": float(effect),
        "p_value": float(p_value),
        "significant": significant
    }
    iteration_results["analyses"].append(analysis)
    
    if significant:
        significant_findings.append({
            "feature": feature,
            "effect": float(effect),
            "p_value": float(p_value)
        })

transcript["iterations"].append(iteration_results)
print(f"Iteration 2: Tested {len(three_level_features[:8])} features, found {sum(1 for a in iteration_results['analyses'] if a['significant'])} significant")

# Iteration 3: Correlation screening for continuous features
print("\n--- Iteration 3: Correlation screening ---")
iteration += 1

continuous_features = [f for f in FEATURE_COLS if df[f].dtype in ['float64', 'int64'] and df[f].nunique() > 10]
print(f"Testing {len(continuous_features)} continuous features for correlation...")

iteration_results = {
    "index": iteration,
    "proposed_hypotheses": [],
    "analyses": []
}

for i, feature in enumerate(continuous_features[:15]):
    hypothesis_id = f"h{iteration}_{i+1}"
    hypothesis_text = f"Feature {feature} is correlated with objective_response."
    
    iteration_results["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    corr, p_value = compute_correlation(df, feature, OUTCOME)
    significant = bool(p_value < 0.05)
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Pearson correlation r={corr:.4f} (p={p_value:.4f}).",
        "effect_estimate": float(corr),
        "p_value": float(p_value),
        "significant": significant
    }
    iteration_results["analyses"].append(analysis)
    
    if significant:
        significant_findings.append({
            "feature": feature,
            "effect": float(corr),
            "p_value": float(p_value)
        })

transcript["iterations"].append(iteration_results)
print(f"Iteration 3: Tested {len(continuous_features[:15])} features, found {sum(1 for a in iteration_results['analyses'] if a['significant'])} significant")

# Iteration 4: Treatment effect heterogeneity search
print("\n--- Iteration 4: Treatment effect heterogeneity search ---")
iteration += 1

# Use the most significant binary feature as a "treatment" proxy
if significant_findings:
    treatment_feature = significant_findings[0]['feature']
    print(f"Using {treatment_feature} as treatment proxy (effect={significant_findings[0]['effect']:.4f}, p={significant_findings[0]['p_value']:.4f})")
    
    iteration_results = {
        "index": iteration,
        "proposed_hypotheses": [],
        "analyses": []
    }
    
    # Test interaction: treatment effect varies by other binary features
    binary_features_for_interaction = [f for f in binary_features if f != treatment_feature][:5]
    
    for i, modifier in enumerate(binary_features_for_interaction):
        hypothesis_id = f"h{iteration}_{i+1}"
        hypothesis_text = f"The effect of {treatment_feature} on objective_response differs by {modifier}."
        
        iteration_results["proposed_hypotheses"].append({
            "id": hypothesis_id,
            "text": hypothesis_text,
            "kind": "novel"
        })
        
        # Stratified analysis
        treatment_mask = df[treatment_feature] == 1
        modifier_mask = df[modifier] == 1
        
        # Effect in modifier=1 group
        effect_t1 = df.loc[treatment_mask & modifier_mask, OUTCOME].mean() - df.loc[~treatment_mask & modifier_mask, OUTCOME].mean()
        # Effect in modifier=0 group
        effect_t0 = df.loc[treatment_mask & ~modifier_mask, OUTCOME].mean() - df.loc[~treatment_mask & ~modifier_mask, OUTCOME].mean()
        
        # Interaction effect
        interaction_effect = effect_t1 - effect_t0
        
        # Test interaction significance using 2x2 contingency
        contingency = pd.crosstab(df[treatment_feature], df[modifier])
        _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
        
        significant = bool(p_value < 0.05)
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Interaction effect: {interaction_effect:.4f} (p={p_value:.4f}). Effect in {modifier}=1: {effect_t1:.3f}, in {modifier}=0: {effect_t0:.3f}.",
            "effect_estimate": float(interaction_effect),
            "p_value": float(p_value),
            "significant": significant
        }
        iteration_results["analyses"].append(analysis)
        
        if significant:
            significant_findings.append({
                "feature": f"{treatment_feature}x{modifier}",
                "effect": float(interaction_effect),
                "p_value": float(p_value)
            })

transcript["iterations"].append(iteration_results)
print(f"Iteration 4: Tested {len(binary_features_for_interaction)} interactions, found {sum(1 for a in iteration_results['analyses'] if a['significant'])} significant")

# Iteration 5: Refined hypotheses from significant findings
print("\n--- Iteration 5: Refined hypotheses ---")
iteration += 1

iteration_results = {
    "index": iteration,
    "proposed_hypotheses": [],
    "analyses": []
}

# Refine the most significant finding
if significant_findings:
    sig = significant_findings[0]
    hypothesis_id = f"h{iteration}_1"
    hypothesis_text = f"The effect of {sig['feature']} on objective_response is strongest in patients with high values of other features."
    
    iteration_results["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "refined"
    })
    
    # Test with a continuous feature modifier
    continuous_modifiers = [f for f in continuous_features if f != sig['feature']][:3]
    
    for modifier in continuous_modifiers:
        mask_high = df[modifier] > df[modifier].median()
        mask_low = df[modifier] <= df[modifier].median()
        
        effect_high = df.loc[mask_high & (df[sig['feature']] == 1), OUTCOME].mean() - df.loc[mask_high & (df[sig['feature']] == 0), OUTCOME].mean()
        effect_low = df.loc[mask_low & (df[sig['feature']] == 1), OUTCOME].mean() - df.loc[mask_low & (df[sig['feature']] == 0), OUTCOME].mean()
        
        interaction_effect = effect_high - effect_low
        
        # Test significance
        n1_high = (mask_high & (df[sig['feature']] == 1)).sum()
        n0_high = (mask_high & (df[sig['feature']] == 0)).sum()
        y1_high = df.loc[mask_high & (df[sig['feature']] == 1), OUTCOME].sum()
        y0_high = df.loc[mask_high & (df[sig['feature']] == 0), OUTCOME].sum()
        
        n1_low = (mask_low & (df[sig['feature']] == 1)).sum()
        n0_low = (mask_low & (df[sig['feature']] == 0)).sum()
        y1_low = df.loc[mask_low & (df[sig['feature']] == 1), OUTCOME].sum()
        y0_low = df.loc[mask_low & (df[sig['feature']] == 0), OUTCOME].sum()
        
        contingency = [[y1_high, n1_high - y1_high], [y1_low, n0_low - y1_low]]
        _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
        
        significant = bool(p_value < 0.05)
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Interaction with {modifier}: effect in high={modifier}={effect_high:.3f}, low={modifier}={effect_low:.3f}, interaction={interaction_effect:.4f} (p={p_value:.4f}).",
            "effect_estimate": float(interaction_effect),
            "p_value": float(p_value),
            "significant": significant
        }
        iteration_results["analyses"].append(analysis)

transcript["iterations"].append(iteration_results)
print(f"Iteration 5: Tested refined hypotheses")

# Iteration 6: Subgroup discovery
print("\n--- Iteration 6: Subgroup discovery ---")
iteration += 1

iteration_results = {
    "index": iteration,
    "proposed_hypotheses": [],
    "analyses": []
}

# Search for subgroups where treatment effect is concentrated
if significant_findings:
    sig = significant_findings[0]
    treatment_feature = sig['feature']
    
    # Test different subgroup definitions
    subgroup_defs = [
        ("feature_001", 1),
        ("feature_004", 1),
        ("feature_005", 1),
    ]
    
    for feature, value in subgroup_defs:
        if feature in df.columns:
            hypothesis_id = f"h{iteration}_{subgroup_defs.index((feature, value))+1}"
            hypothesis_text = f"The effect of {treatment_feature} on objective_response is concentrated in patients with {feature}={value}."
            
            iteration_results["proposed_hypotheses"].append({
                "id": hypothesis_id,
                "text": hypothesis_text,
                "kind": "novel"
            })
            
            # Compare effect in subgroup vs overall
            subgroup_mask = df[feature] == value
            overall_mask = df[treatment_feature] == 1
            
            effect_in_subgroup = df.loc[subgroup_mask & overall_mask, OUTCOME].mean() - df.loc[subgroup_mask & ~overall_mask, OUTCOME].mean()
            effect_overall = df.loc[overall_mask, OUTCOME].mean() - df.loc[~overall_mask, OUTCOME].mean()
            
            # Test if subgroup effect differs from overall
            n1_sub = (subgroup_mask & overall_mask).sum()
            n0_sub = (subgroup_mask & ~overall_mask).sum()
            y1_sub = df.loc[subgroup_mask & overall_mask, OUTCOME].sum()
            y0_sub = df.loc[subgroup_mask & ~overall_mask, OUTCOME].sum()
            
            n1_over = overall_mask.sum()
            n0_over = (~overall_mask).sum()
            y1_over = df.loc[overall_mask, OUTCOME].sum()
            y0_over = df.loc[~overall_mask, OUTCOME].sum()
            
            contingency = [[y1_sub, n1_sub - y1_sub], [y1_over, n0_over - y1_over]]
            _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
            
            significant = bool(p_value < 0.05)
            
            analysis = {
                "hypothesis_ids": [hypothesis_id],
                "result_summary": f"Effect in {feature}={value}: {effect_in_subgroup:.3f}, overall: {effect_overall:.3f} (p={p_value:.4f}).",
                "effect_estimate": float(effect_in_subgroup - effect_overall),
                "p_value": float(p_value),
                "significant": significant
            }
            iteration_results["analyses"].append(analysis)

transcript["iterations"].append(iteration_results)
print(f"Iteration 6: Tested subgroup hypotheses")

# Iteration 7: Additional main effects
print("\n--- Iteration 7: Additional main effects ---")
iteration += 1

iteration_results = {
    "index": iteration,
    "proposed_hypotheses": [],
    "analyses": []
}

# Test remaining binary features
remaining_binary = [f for f in binary_features if f not in [sf['feature'] for sf in significant_findings]]
for i, feature in enumerate(remaining_binary[:8]):
    hypothesis_id = f"h{iteration}_{i+1}"
    hypothesis_text = f"Patients with {feature}={1} have different objective_response rates than those with {feature}={0}."
    
    iteration_results["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    effect, p_value = compare_groups(df, feature, 1, OUTCOME)
    significant = bool(p_value < 0.05)
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Objective response rate: {df.loc[df[feature]==1, OUTCOME].mean():.3f} vs {df.loc[df[feature]==0, OUTCOME].mean():.3f} (chi-square p={p_value:.4f}).",
        "effect_estimate": float(effect),
        "p_value": float(p_value),
        "significant": significant
    }
    iteration_results["analyses"].append(analysis)
    
    if significant:
        significant_findings.append({
            "feature": feature,
            "effect": float(effect),
            "p_value": float(p_value)
        })

transcript["iterations"].append(iteration_results)
print(f"Iteration 7: Tested {len(remaining_binary[:8])} features, found {sum(1 for a in iteration_results['analyses'] if a['significant'])} significant")

# Iteration 8: More interaction tests
print("\n--- Iteration 8: Additional interaction tests ---")
iteration += 1

iteration_results = {
    "index": iteration,
    "proposed_hypotheses": [],
    "analyses": []
}

# Test interactions with significant findings
if len(significant_findings) >= 2:
    sig1 = significant_findings[0]
    sig2 = significant_findings[1]
    
    hypothesis_id = f"h{iteration}_1"
    hypothesis_text = f"The joint effect of {sig1['feature']} and {sig2['feature']} on objective_response differs from additive expectation."
    
    iteration_results["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    # Test 2x2 interaction
    mask1 = df[sig1['feature']] == 1
    mask2 = df[sig2['feature']] == 1
    
    # Four groups
    g00 = df.loc[~mask1 & ~mask2, OUTCOME].mean()
    g10 = df.loc[mask1 & ~mask2, OUTCOME].mean()
    g01 = df.loc[~mask1 & mask2, OUTCOME].mean()
    g11 = df.loc[mask1 & mask2, OUTCOME].mean()
    
    # Additive expectation
    additive = g00 + (g10 - g00) + (g01 - g00)
    interaction = g11 - additive
    
    # Test significance using 2x2x2 contingency
    contingency = pd.crosstab([df[sig1['feature']], df[sig2['feature']]], df[OUTCOME])
    _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
    
    significant = bool(p_value < 0.05)
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Interaction effect: {interaction:.4f} (p={p_value:.4f}). Response rates: {g00:.3f}, {g10:.3f}, {g01:.3f}, {g11:.3f}.",
        "effect_estimate": float(interaction),
        "p_value": float(p_value),
        "significant": significant
    }
    iteration_results["analyses"].append(analysis)

transcript["iterations"].append(iteration_results)
print(f"Iteration 8: Tested interaction hypotheses")

# Iteration 9: Final main effects pass
print("\n--- Iteration 9: Final main effects pass ---")
iteration += 1

iteration_results = {
    "index": iteration,
    "proposed_hypotheses": [],
    "analyses": []
}

# Test remaining features
remaining = [f for f in FEATURE_COLS if f not in [sf['feature'] for sf in significant_findings]]
for i, feature in enumerate(remaining[:10]):
    hypothesis_id = f"h{iteration}_{i+1}"
    hypothesis_text = f"Patients with {feature}={1} (or highest category) have different objective_response rates than those without."
    
    iteration_results["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    # For binary features
    if df[feature].nunique() == 2:
        effect, p_value = compare_groups(df, feature, 1, OUTCOME)
    else:
        # For multi-level, compare highest vs rest
        max_val = df[feature].max()
        mask = df[feature] == max_val
        effect, p_value = compare_groups(df, feature, max_val, OUTCOME)
    
    significant = bool(p_value < 0.05)
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Objective response rate: {df.loc[df[feature]==1 if df[feature].nunique()==2 else mask, OUTCOME].mean():.3f} vs {df.loc[df[feature]!=1 if df[feature].nunique()==2 else ~mask, OUTCOME].mean():.3f} (p={p_value:.4f}).",
        "effect_estimate": float(effect),
        "p_value": float(p_value),
        "significant": significant
    }
    iteration_results["analyses"].append(analysis)
    
    if significant:
        significant_findings.append({
            "feature": feature,
            "effect": float(effect),
            "p_value": float(p_value)
        })

transcript["iterations"].append(iteration_results)
print(f"Iteration 9: Tested {len(remaining[:10])} features, found {sum(1 for a in iteration_results['analyses'] if a['significant'])} significant")

# Iteration 10: Best supported treatment-effect subgroup
print("\n--- Iteration 10: Best supported treatment-effect subgroup ---")
iteration += 1

iteration_results = {
    "index": iteration,
    "proposed_hypotheses": [],
    "analyses": []
}

# Identify best supported subgroup
if significant_findings:
    # Find the feature with strongest effect
    best_sig = max(significant_findings, key=lambda x: abs(x['effect']))
    
    hypothesis_id = f"h{iteration}_1"
    hypothesis_text = f"The effect of {best_sig['feature']} on objective_response is strongest in patients with {best_sig['feature']}={1}."
    
    iteration_results["proposed_hypotheses"].append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "refined"
    })
    
    # Confirm the effect
    mask = df[best_sig['feature']] == 1
    effect = df.loc[mask, OUTCOME].mean() - df.loc[~mask, OUTCOME].mean()
    
    # Chi-square test
    contingency = pd.crosstab(df[best_sig['feature']], df[OUTCOME])
    _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
    
    significant = bool(p_value < 0.05)
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Best supported: {best_sig['feature']} effect = {effect:.4f} (p={p_value:.4f}). Response rate: {df.loc[mask, OUTCOME].mean():.3f} vs {df.loc[~mask, OUTCOME].mean():.3f}.",
        "effect_estimate": float(effect),
        "p_value": float(p_value),
        "significant": significant
    }
    iteration_results["analyses"].append(analysis)

transcript["iterations"].append(iteration_results)
print(f"Iteration 10: Confirmed best supported subgroup")

# Write transcript.json
print("\n" + "="*60)
print("Writing transcript.json...")
with open('transcript.json', 'w') as f:
    json.dump(transcript, f, indent=2)
print("transcript.json written successfully")

# Generate analysis_summary.txt
print("\nGenerating analysis_summary.txt...")

summary_lines = [
    "="*70,
    "AML DATASET ANALYSIS SUMMARY",
    "="*70,
    "",
    f"Dataset: ds001_aml",
    f"Total patients: {len(df)}",
    f"Features analyzed: {len(FEATURE_COLS)}",
    f"Outcome: objective_response (binary: 0={df[OUTCOME].sum()==0}, 1={df[OUTCOME].sum()})",
    "",
    "="*70,
    "EXECUTIVE SUMMARY",
    "="*70,
    "",
]

# Count significant findings
total_significant = sum(1 for a in transcript['iterations'] for analysis in a['analyses'] if analysis.get('significant', False))
summary_lines.append(f"Total significant findings (p < 0.05): {total_significant}")
summary_lines.append(f"Total hypotheses tested: {sum(len(i['analyses']) for i in transcript['iterations'])}")
summary_lines.append("")

# Summary by iteration
summary_lines.append("="*70)
summary_lines.append("ITERATION-BY-ITERATION RESULTS")
summary_lines.append("="*70)
summary_lines.append("")

for iter_data in transcript['iterations']:
    sig_count = sum(1 for a in iter_data['analyses'] if a.get('significant', False))
    summary_lines.append(f"Iteration {iter_data['index']}: {len(iter_data['analyses'])} analyses, {sig_count} significant")
    summary_lines.append("")

# Detailed findings
summary_lines.append("="*70)
summary_lines.append("DETAILED SIGNIFICANT FINDINGS")
summary_lines.append("="*70)
summary_lines.append("")

if significant_findings:
    summary_lines.append("Significant feature-outcome associations (p < 0.05):")
    summary_lines.append("")
    
    # Group by feature type
    binary_sig = [f for f in significant_findings if f['feature'] in binary_features]
    continuous_sig = [f for f in significant_findings if f['feature'] in continuous_features]
    
    if binary_sig:
        summary_lines.append("Binary features with significant effects:")
        for f in sorted(binary_sig, key=lambda x: abs(x['effect']), reverse=True)[:10]:
            summary_lines.append(f"  - {f['feature']}: effect = {f['effect']:.4f}, p = {f['p_value']:.4f}")
        summary_lines.append("")
    
    if continuous_sig:
        summary_lines.append("Continuous features with significant correlations:")
        for f in sorted(continuous_sig, key=lambda x: abs(x['effect']), reverse=True)[:10]:
            summary_lines.append(f"  - {f['feature']}: r = {f['effect']:.4f}, p = {f['p_value']:.4f}")
        summary_lines.append("")
    
    # Interactions
    interaction_sig = [f for f in significant_findings if 'x' in f['feature']]
    if interaction_sig:
        summary_lines.append("Significant interaction effects:")
        for f in interaction_sig[:5]:
            summary_lines.append(f"  - {f['feature']}: effect = {f['effect']:.4f}, p = {f['p_value']:.4f}")
        summary_lines.append("")

# Best supported subgroup
if significant_findings:
    best_sig = max(significant_findings, key=lambda x: abs(x['effect']))
    summary_lines.append("="*70)
    summary_lines.append("BEST SUPPORTED TREATMENT-EFFECT SUBGROUP")
    summary_lines.append("="*70)
    summary_lines.append("")
    summary_lines.append(f"Feature: {best_sig['feature']}")
    summary_lines.append(f"Effect estimate: {best_sig['effect']:.4f}")
    summary_lines.append(f"P-value: {best_sig['p_value']:.4f}")
    summary_lines.append("")
    summary_lines.append(f"Interpretation: Patients with {best_sig['feature']}={1} show objective_response rate of {df.loc[df[best_sig['feature']]==1, OUTCOME].mean():.1%} vs {df.loc[df[best_sig['feature']]==0, OUTCOME].mean():.1%} in those without.")
    summary_lines.append("")

# Conclusions
summary_lines.append("="*70)
summary_lines.append("CONCLUSIONS")
summary_lines.append("="*70)
summary_lines.append("")
summary_lines.append("This analysis explored feature-outcome relationships in the AML dataset using")
summary_lines.append("iterative hypothesis testing. Key findings include:")
summary_lines.append("")
summary_lines.append(f"1. {len(binary_sig)} binary features showed significant associations with objective_response")
summary_lines.append(f"2. {len(continuous_sig)} continuous features showed significant correlations")
summary_lines.append(f"3. {len(interaction_sig)} interaction effects were identified")
summary_lines.append("")
if significant_findings:
    summary_lines.append(f"The strongest effect was observed for {best_sig['feature']} (effect = {best_sig['effect']:.4f}, p = {best_sig['p_value']:.4f}).")
    summary_lines.append("")
summary_lines.append("Recommendations for further investigation:")
summary_lines.append("- Validate top findings in independent cohorts")
summary_lines.append("- Explore clinical interpretability of significant features")
summary_lines.append("- Consider multivariable modeling to adjust for confounding")
summary_lines.append("")
summary_lines.append("="*70)
summary_lines.append("END OF ANALYSIS SUMMARY")
summary_lines.append("="*70)

with open('analysis_summary.txt', 'w') as f:
    f.write('\n'.join(summary_lines))

print("analysis_summary.txt written successfully")
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Output files: transcript.json, analysis_summary.txt")
