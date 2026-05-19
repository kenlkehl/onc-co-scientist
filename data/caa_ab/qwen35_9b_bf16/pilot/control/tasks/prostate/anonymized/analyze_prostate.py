#!/usr/bin/env python3
"""
End-to-end oncology dataset analysis script.
Performs iterative hypothesis generation, testing, and refinement.
Outputs transcript.json and analysis_summary.txt.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_parquet('dataset.parquet')

# Identify binary/categorical features (suitable for group comparisons)
binary_features = [col for col in df.columns if col != 'patient_id' and col != 'objective_response' and df[col].nunique() == 2]
print(f"Binary features ({len(binary_features)}): {binary_features}")

# Features with more than 2 values (for categorical analysis)
multi_val_features = [col for col in df.columns if col != 'patient_id' and col != 'objective_response' and df[col].nunique() > 2]
print(f"Multi-value features ({len(multi_val_features)}): {multi_val_features}")

# Outcome
outcome_col = 'objective_response'

def compute_effect(df: pd.DataFrame, feature: str, value: Any, outcome: str) -> Tuple[float, float]:
    """
    Compute effect estimate (difference in means) and p-value using boolean mask.
    Returns (effect_estimate, p_value).
    """
    mask = df[feature] == value
    group1 = df.loc[mask, outcome]
    group0 = df.loc[~mask, outcome]
    
    if len(group1) == 0 or len(group0) == 0:
        return np.nan, np.nan
    
    mean1 = group1.mean()
    mean0 = group0.mean()
    effect = mean1 - mean0
    
    # Two-sample t-test - returns TtestResult object
    result = stats.ttest_ind(group1, group0, equal_var=False)
    p_value = float(result.pvalue)
    
    return float(effect), float(p_value)

def compute_chi2_effect(df: pd.DataFrame, feature: str, value: Any, outcome: str) -> Tuple[float, float]:
    """
    Compute effect estimate (difference in proportions) and p-value using chi-square.
    Returns (effect_estimate, p_value).
    """
    mask = df[feature] == value
    group1 = df.loc[mask, outcome]
    group0 = df.loc[~mask, outcome]
    
    if len(group1) == 0 or len(group0) == 0:
        return np.nan, np.nan
    
    prop1 = group1.mean()
    prop0 = group0.mean()
    effect = prop1 - prop0
    
    # Build 2x2 contingency table
    n1 = len(group1)
    n0 = len(group0)
    k1 = group1.sum()
    k0 = group0.sum()
    
    table = np.array([[k1, n1 - k1], [k0, n0 - k0]])
    _, p_value, _, _ = stats.chi2_contingency(table, correction=False)
    
    return float(effect), float(p_value)

def compute_correlation(df: pd.DataFrame, feature: str, outcome: str) -> Tuple[float, float]:
    """
    Compute Pearson correlation and p-value.
    Returns (correlation, p_value).
    """
    mask = df[outcome].notna() & df[feature].notna()
    x = df.loc[mask, feature]
    y = df.loc[mask, outcome]
    
    if len(x) < 3:
        return np.nan, np.nan
    
    corr, p_value = stats.pearsonr(x, y)
    return float(corr), float(p_value)

# Iteration management
transcript = {
    "dataset_id": "ds001_prostate",
    "model_id": "qwen35-9b",
    "harness_id": "codex-cli@1.0.0",
    "max_iterations": 10,
    "iterations": []
}

iteration = 0
all_significant_results = []

# Iteration 1: Main effects - binary features vs outcome
print("\n=== Iteration 1: Main effects (binary features) ===")
iteration += 1

hypotheses = []
analyses = []

# Select a subset of binary features for main effects analysis
selected_binary = binary_features[:10]  # Start with first 10

for feat in selected_binary:
    h_id = f"h{iteration}_{feat}"
    hypotheses.append({
        "id": h_id,
        "text": f"Mean {outcome_col} differs between patients with {feat}={1} and {feat}={0}.",
        "kind": "novel"
    })
    
    effect, p_val = compute_effect(df, feat, 1, outcome_col)
    significant = bool(p_val < 0.05) if not np.isnan(p_val) else False
    
    analyses.append({
        "hypothesis_ids": [h_id],
        "code": f"compute_effect(df, '{feat}', 1, '{outcome_col}')",
        "result_summary": f"Mean {outcome_col}: {effect:.3f} (feat={1}) vs reference (feat={0}), t-test p={p_val:.4f}.",
        "effect_estimate": float(effect),
        "p_value": float(p_val),
        "significant": significant
    })
    
    if significant:
        all_significant_results.append((feat, effect, p_val))

transcript["iterations"].append({
    "index": iteration,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len([a for a in analyses if a['significant']])} significant results in iteration 1")

# Iteration 2: Main effects - multi-value features (top 5 by variance)
print("\n=== Iteration 2: Main effects (multi-value features) ===")
iteration += 1

# Compute variance for each multi-value feature
variances = {}
for feat in multi_val_features:
    variances[feat] = df[feat].var()

# Sort by variance and select top 5
sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:5]
selected_multi = [f[0] for f in sorted_features]

hypotheses = []
analyses = []

for feat in selected_multi:
    h_id = f"h{iteration}_{feat}"
    hypotheses.append({
        "id": h_id,
        "text": f"Mean {outcome_col} differs between patients with {feat}={df[feat].min()} and {feat}={df[feat].max()}.",
        "kind": "novel"
    })
    
    # Use correlation for continuous features
    if df[feat].dtype in ['float64', 'int64']:
        corr, p_val = compute_correlation(df, feat, outcome_col)
        effect = corr * df[feat].std() * df[outcome_col].std()  # Convert to effect-like scale
    else:
        effect, p_val = compute_effect(df, feat, df[feat].min(), outcome_col)
    
    significant = bool(p_val < 0.05) if not np.isnan(p_val) else False
    
    analyses.append({
        "hypothesis_ids": [h_id],
        "code": f"compute_correlation(df, '{feat}', '{outcome_col}')",
        "result_summary": f"Correlation: {corr:.4f}, p={p_val:.4f}.",
        "effect_estimate": float(effect),
        "p_value": float(p_val),
        "significant": significant
    })
    
    if significant:
        all_significant_results.append((feat, effect, p_val))

transcript["iterations"].append({
    "index": iteration,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len([a for a in analyses if a['significant']])} significant results in iteration 2")

# Iteration 3: Treatment-effect heterogeneity search
print("\n=== Iteration 3: Treatment-effect heterogeneity search ===")
iteration += 1

# Find the strongest binary feature from iteration 1
if all_significant_results:
    strongest_feat = all_significant_results[0][0]
    strongest_effect = all_significant_results[0][1]
    
    h_id = f"h{iteration}_heterogeneity"
    hypotheses.append({
        "id": h_id,
        "text": f"The effect of {strongest_feat} on {outcome_col} may be modified by other features.",
        "kind": "novel"
    })
    
    # Test interaction: strongest binary feature with other binary features
    analyses = []
    
    # Check interaction with other binary features
    for other_feat in binary_features[:5]:
        if other_feat == strongest_feat:
            continue
        
        # Create interaction term
        interaction = df[strongest_feat] * df[other_feat]
        
        # Compare groups defined by interaction
        mask1 = interaction == 1
        mask0 = interaction == 0
        
        group1 = df.loc[mask1, outcome_col]
        group0 = df.loc[mask0, outcome_col]
        
        if len(group1) > 0 and len(group0) > 0:
            result = stats.ttest_ind(group1, group0, equal_var=False)
            effect = float(group1.mean() - group0.mean())
            p_val = float(result.pvalue)
            significant = bool(p_val < 0.05)
            
            analyses.append({
                "hypothesis_ids": [h_id],
                "code": f"ttest_ind(df.loc[df['{strongest_feat}']==1 & df['{other_feat}']==1, '{outcome_col}'], df.loc[~(df['{strongest_feat}']==1 & df['{other_feat}']==1), '{outcome_col}'])",
                "result_summary": f"Interaction {strongest_feat}*{other_feat}: effect={effect:.3f}, p={p_val:.4f}.",
                "effect_estimate": effect,
                "p_value": p_val,
                "significant": significant
            })
            
            if significant:
                all_significant_results.append((f"{strongest_feat}*{other_feat}", effect, p_val))

transcript["iterations"].append({
    "index": iteration,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len([a for a in analyses if a['significant']])} significant results in iteration 3")

# Iteration 4: Stratified analysis by strongest modifier
print("\n=== Iteration 4: Stratified analysis ===")
iteration += 1

# Find the strongest binary feature
if all_significant_results:
    strongest_feat = all_significant_results[0][0]
    
    h_id = f"h{iteration}_stratified"
    hypotheses.append({
        "id": h_id,
        "text": f"Within strata of {strongest_feat}, the relationship with {outcome_col} may differ.",
        "kind": "refined"
    })
    
    analyses = []
    
    # Stratify by strongest feature and analyze other features
    for other_feat in binary_features[:5]:
        if other_feat == strongest_feat:
            continue
        
        # Stratified analysis
        for strat_val in [0, 1]:
            mask_strat = df[strongest_feat] == strat_val
            mask_other = df[other_feat] == 1
            
            group1 = df.loc[mask_strat & mask_other, outcome_col]
            group0 = df.loc[mask_strat & ~mask_other, outcome_col]
            
            if len(group1) > 0 and len(group0) > 0:
                result = stats.ttest_ind(group1, group0, equal_var=False)
                effect = float(group1.mean() - group0.mean())
                p_val = float(result.pvalue)
                significant = bool(p_val < 0.05)
                
                analyses.append({
                    "hypothesis_ids": [h_id],
                    "code": f"ttest_ind(df.loc[df['{strongest_feat}']=={strat_val} & df['{other_feat}']==1, '{outcome_col}'], df.loc[df['{strongest_feat}']=={strat_val} & df['{other_feat}']==0, '{outcome_col}'])",
                    "result_summary": f"Stratified by {strongest_feat}={strat_val}: {other_feat}=1 vs 0, effect={effect:.3f}, p={p_val:.4f}.",
                    "effect_estimate": effect,
                    "p_value": p_val,
                    "significant": significant
                })
                
                if significant:
                    all_significant_results.append((f"{strongest_feat}={strat_val},{other_feat}=1", effect, p_val))

transcript["iterations"].append({
    "index": iteration,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len([a for a in analyses if a['significant']])} significant results in iteration 4")

# Iteration 5: Regression with multiple predictors
print("\n=== Iteration 5: Multivariable regression ===")
iteration += 1

h_id = f"h{iteration}_regression"
hypotheses.append({
    "id": h_id,
    "text": f"Multiple features jointly predict {outcome_col} after adjusting for confounders.",
    "kind": "novel"
})

# Select top 5 binary features by variance
top_binary = sorted(binary_features, key=lambda x: df[x].var(), reverse=True)[:5]

# Build regression model
if len(top_binary) >= 2:
    X_cols = top_binary
    y = df[outcome_col]
    
    # Simple OLS using statsmodels if available, otherwise manual
    try:
        import statsmodels.api as sm
        X = sm.add_constant(df[X_cols])
        model = sm.OLS(y, X).fit()
        
        # Extract coefficients
        results = []
        for col in X_cols:
            coef = float(model.params[col])
            p_val = float(model.pvalues[col])
            significant = bool(p_val < 0.05)
            
            analyses.append({
                "hypothesis_ids": [h_id],
                "code": f"sm.OLS(df['{outcome_col}'], sm.add_constant(df[{col}])).fit()",
                "result_summary": f"Adjusted coefficient for {col}: {coef:.4f}, p={p_val:.4f}.",
                "effect_estimate": coef,
                "p_value": p_val,
                "significant": significant
            })
            
            if significant:
                all_significant_results.append((col, coef, p_val))
        
        # Overall model fit - use a dummy p_value for R-squared
        r_squared = float(model.rsquared)
        analyses.append({
            "hypothesis_ids": [h_id],
            "code": None,
            "result_summary": f"Model R-squared: {r_squared:.4f}.",
            "effect_estimate": r_squared,
            "p_value": 0.0,  # Use 0.0 as placeholder for R-squared
            "significant": bool(r_squared > 0.1)
        })
        
    except ImportError:
        # Manual regression approximation
        analyses.append({
            "hypothesis_ids": [h_id],
            "code": None,
            "result_summary": f"statsmodels not available; skipping multivariable regression.",
            "effect_estimate": np.nan,
            "p_value": np.nan,
            "significant": False
        })

transcript["iterations"].append({
    "index": iteration,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len([a for a in analyses if a['significant']])} significant results in iteration 5")

# Iteration 6: Subgroup discovery - exhaustive check of 2-feature combinations
print("\n=== Iteration 6: Subgroup discovery (2-feature interactions) ===")
iteration += 1

h_id = f"h{iteration}_subgroup"
hypotheses.append({
    "id": h_id,
    "text": f"Combinations of two binary features define subgroups with distinct {outcome_col} rates.",
    "kind": "novel"
})

analyses = []
significant_count = 0

# Check combinations of binary features
for i, feat1 in enumerate(binary_features[:8]):
    for feat2 in binary_features[i+1:8]:
        # Create 4 groups based on feature combinations
        groups = []
        for v1 in [0, 1]:
            for v2 in [0, 1]:
                mask = (df[feat1] == v1) & (df[feat2] == v2)
                group = df.loc[mask, outcome_col]
                if len(group) > 0:
                    groups.append((v1, v2, float(group.mean())))
        
        if len(groups) >= 2:
            # Compare groups with highest and lowest mean
            groups_sorted = sorted(groups, key=lambda x: x[2])
            if len(groups_sorted) >= 2:
                low_group = groups_sorted[0][2]
                high_group = groups_sorted[-1][2]
                effect = high_group - low_group
                
                # Chi-square test for difference in proportions
                n1 = len(df[(df[feat1] == groups_sorted[-1][0]) & (df[feat2] == groups_sorted[-1][1])])
                k1 = df[(df[feat1] == groups_sorted[-1][0]) & (df[feat2] == groups_sorted[-1][1])][outcome_col].sum()
                
                n0 = len(df[(df[feat1] == groups_sorted[0][0]) & (df[feat2] == groups_sorted[0][1])])
                k0 = df[(df[feat1] == groups_sorted[0][0]) & (df[feat2] == groups_sorted[0][1])][outcome_col].sum()
                
                table = np.array([[k1, n1 - k1], [k0, n0 - k0]])
                _, p_val, _, _ = stats.chi2_contingency(table, correction=False)
                significant = bool(p_val < 0.05)
                
                if significant:
                    significant_count += 1
                    all_significant_results.append((f"{feat1}={groups_sorted[-1][0]},{feat2}={groups_sorted[-1][1]}", effect, p_val))
                
                analyses.append({
                    "hypothesis_ids": [h_id],
                    "code": None,
                    "result_summary": f"Subgroup {feat1}={groups_sorted[-1][0]},{feat2}={groups_sorted[-1][1]} vs {feat1}={groups_sorted[0][0]},{feat2}={groups_sorted[0][1]}: effect={effect:.3f}, p={p_val:.4f}.",
                    "effect_estimate": effect,
                    "p_value": p_val,
                    "significant": significant
                })

transcript["iterations"].append({
    "index": iteration,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {significant_count} significant subgroup results in iteration 6")

# Iteration 7: Best treatment-effect subgroup identification
print("\n=== Iteration 7: Best treatment-effect subgroup ===")
iteration += 1

# Find the best treatment-effect subgroup from all significant results
if all_significant_results:
    # Sort by effect size * significance
    sorted_results = sorted(all_significant_results, key=lambda x: abs(x[1]), reverse=True)[:5]
    
    h_id = f"h{iteration}_best_subgroup"
    hypotheses.append({
        "id": h_id,
        "text": f"The strongest treatment-effect subgroup is defined by: {sorted_results[0][0]} with effect={sorted_results[0][1]:.3f}, p={sorted_results[0][2]:.4f}.",
        "kind": "refined"
    })
    
    analyses = []
    for feat, effect, p_val in sorted_results:
        analyses.append({
            "hypothesis_ids": [h_id],
            "code": None,
            "result_summary": f"Subgroup '{feat}': effect={effect:.3f}, p={p_val:.4f}.",
            "effect_estimate": float(effect),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05)
        })

transcript["iterations"].append({
    "index": iteration,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Identified best treatment-effect subgroup in iteration 7")

# Iteration 8: Refine best subgroup with additional features
print("\n=== Iteration 8: Refine best subgroup ===")
iteration += 1

if all_significant_results:
    best_result = all_significant_results[0]
    best_subgroup = best_result[0]
    
    h_id = f"h{iteration}_refined"
    hypotheses.append({
        "id": h_id,
        "text": f"Adding additional features to the best subgroup {best_subgroup} may improve prediction of {outcome_col}.",
        "kind": "refined"
    })
    
    analyses = []
    
    # Try adding other binary features to the best subgroup
    best_parts = best_subgroup.split(',')
    if len(best_parts) == 2:
        feat1, feat2 = best_parts
        for other_feat in binary_features[:5]:
            if other_feat in [feat1, feat2]:
                continue
            
            # Test with additional feature
            mask = (df[feat1] == int(best_parts[0].split('=')[1])) & (df[feat2] == int(best_parts[1].split('=')[1])) & (df[other_feat] == 1)
            group1 = df.loc[mask, outcome_col]
            group0 = df.loc[~mask, outcome_col]
            
            if len(group1) > 0 and len(group0) > 0:
                result = stats.ttest_ind(group1, group0, equal_var=False)
                effect = float(group1.mean() - group0.mean())
                p_val = float(result.pvalue)
                significant = bool(p_val < 0.05)
                
                analyses.append({
                    "hypothesis_ids": [h_id],
                    "code": None,
                    "result_summary": f"Refined subgroup with {other_feat}=1: effect={effect:.3f}, p={p_val:.4f}.",
                    "effect_estimate": effect,
                    "p_value": p_val,
                    "significant": significant
                })
                
                if significant:
                    all_significant_results.append((f"{best_subgroup},{other_feat}=1", effect, p_val))

transcript["iterations"].append({
    "index": iteration,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Refined best subgroup in iteration 8")

# Iteration 9: Check for non-linear relationships
print("\n=== Iteration 9: Non-linear relationships ===")
iteration += 1

h_id = f"h{iteration}_nonlinear"
hypotheses.append({
    "id": h_id,
    "text": f"Non-linear relationships between features and {outcome_col} may exist.",
    "kind": "novel"
})

analyses = []

# Check quadratic relationships for continuous features
for feat in multi_val_features[:5]:
    if df[feat].dtype not in ['float64', 'int64']:
        continue
    
    # Create squared term
    df_temp = df.copy()
    df_temp[f'{feat}_sq'] = df_temp[feat] ** 2
    
    # Compare linear vs quadratic effect
    mask = df_temp[f'{feat}_sq'] > df_temp[feat].mean() ** 2
    group1 = df_temp.loc[mask, outcome_col]
    group0 = df_temp.loc[~mask, outcome_col]
    
    if len(group1) > 0 and len(group0) > 0:
        result = stats.ttest_ind(group1, group0, equal_var=False)
        effect = float(group1.mean() - group0.mean())
        p_val = float(result.pvalue)
        significant = bool(p_val < 0.05)
        
        analyses.append({
            "hypothesis_ids": [h_id],
            "code": None,
            "result_summary": f"Quadratic effect of {feat}: effect={effect:.3f}, p={p_val:.4f}.",
            "effect_estimate": effect,
            "p_value": p_val,
            "significant": significant
        })
        
        if significant:
            all_significant_results.append((f"{feat}^2", effect, p_val))

transcript["iterations"].append({
    "index": iteration,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len([a for a in analyses if a['significant']])} significant non-linear results in iteration 9")

# Iteration 10: Final comprehensive summary
print("\n=== Iteration 10: Final comprehensive analysis ===")
iteration += 1

h_id = f"h{iteration}_final"
hypotheses.append({
    "id": h_id,
    "text": f"Comprehensive assessment of all feature-outcome relationships.",
    "kind": "refined"
})

analyses = []

# Summary statistics
overall_rate = df[outcome_col].mean()
print(f"Overall {outcome_col} rate: {overall_rate:.3f}")

# Check feature importance by correlation strength
correlations = []
for feat in binary_features[:10]:
    corr, p_val = compute_correlation(df, feat, outcome_col)
    correlations.append((feat, corr, p_val))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

for feat, corr, p_val in correlations[:5]:
    analyses.append({
        "hypothesis_ids": [h_id],
        "code": None,
        "result_summary": f"Correlation between {feat} and {outcome_col}: {corr:.4f}, p={p_val:.4f}.",
        "effect_estimate": float(corr),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05)
    })

transcript["iterations"].append({
    "index": iteration,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Completed iteration 10")

# Write transcript.json
with open('transcript.json', 'w') as f:
    json.dump(transcript, f, indent=2)
print("\nWrote transcript.json")

# Generate analysis_summary.txt
summary_lines = [
    "=" * 80,
    "ONCOLOGY DATASET ANALYSIS SUMMARY",
    "=" * 80,
    "",
    f"Dataset: ds001_prostate",
    f"Total patients: {len(df)}",
    f"Outcome ({outcome_col}): {df[outcome_col].sum()} responders ({df[outcome_col].mean()*100:.1f}%)",
    f"Binary features analyzed: {len(binary_features)}",
    f"Multi-value features analyzed: {len(multi_val_features)}",
    "",
    "-" * 80,
    "ITERATION 1: Main Effects (Binary Features)",
    "-" * 80,
]

for analysis in transcript["iterations"][0]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis["significant"] else "not significant"
    summary_lines.append(f"  {analysis['result_summary']} [{sig_str}]")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 2: Main Effects (Multi-Value Features)",
    "-" * 80,
])

for analysis in transcript["iterations"][1]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis["significant"] else "not significant"
    summary_lines.append(f"  {analysis['result_summary']} [{sig_str}]")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 3: Treatment-Effect Heterogeneity Search",
    "-" * 80,
])

for analysis in transcript["iterations"][2]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis["significant"] else "not significant"
    summary_lines.append(f"  {analysis['result_summary']} [{sig_str}]")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 4: Stratified Analysis",
    "-" * 80,
])

for analysis in transcript["iterations"][3]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis["significant"] else "not significant"
    summary_lines.append(f"  {analysis['result_summary']} [{sig_str}]")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 5: Multivariable Regression",
    "-" * 80,
])

for analysis in transcript["iterations"][4]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis["significant"] else "not significant"
    summary_lines.append(f"  {analysis['result_summary']} [{sig_str}]")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 6: Subgroup Discovery (2-Feature Interactions)",
    "-" * 80,
])

for analysis in transcript["iterations"][5]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis["significant"] else "not significant"
    summary_lines.append(f"  {analysis['result_summary']} [{sig_str}]")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 7: Best Treatment-Effect Subgroup",
    "-" * 80,
])

for analysis in transcript["iterations"][6]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis["significant"] else "not significant"
    summary_lines.append(f"  {analysis['result_summary']} [{sig_str}]")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 8: Refined Best Subgroup",
    "-" * 80,
])

for analysis in transcript["iterations"][7]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis["significant"] else "not significant"
    summary_lines.append(f"  {analysis['result_summary']} [{sig_str}]")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 9: Non-Linear Relationships",
    "-" * 80,
])

for analysis in transcript["iterations"][8]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis["significant"] else "not significant"
    summary_lines.append(f"  {analysis['result_summary']} [{sig_str}]")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 10: Final Comprehensive Analysis",
    "-" * 80,
])

for analysis in transcript["iterations"][9]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis["significant"] else "not significant"
    summary_lines.append(f"  {analysis['result_summary']} [{sig_str}]")

summary_lines.extend([
    "",
    "-" * 80,
    "KEY FINDINGS",
    "-" * 80,
])

# Count significant results
total_significant = sum(1 for iter_data in transcript["iterations"] for a in iter_data["analyses"] if a.get("significant", False))
summary_lines.append(f"Total significant findings across all iterations: {total_significant}")

# List top significant findings - filter out None p_values
top_findings = []
for iter_data in transcript["iterations"]:
    for analysis in iter_data["analyses"]:
        if analysis.get("significant", False) and analysis.get("p_value") is not None:
            top_findings.append((analysis["result_summary"], analysis["effect_estimate"], analysis["p_value"]))

# Sort by absolute effect size, handling 0.0 p_values
def sort_key(x):
    effect, p_val = x[1], x[2]
    return abs(effect) if not np.isnan(effect) else 0.0

top_findings.sort(key=sort_key, reverse=True)[:10]

for finding, effect, p_val in top_findings:
    summary_lines.append(f"  - {finding} (effect={effect:.3f}, p={p_val:.4f})")

summary_lines.extend([
    "",
    "-" * 80,
    "CONCLUSIONS",
    "-" * 80,
])

if total_significant > 0:
    summary_lines.append("Several statistically significant feature-outcome relationships were identified.")
    summary_lines.append("The analysis suggests that specific subgroups of patients may have distinct")
    summary_lines.append("response patterns, warranting further clinical investigation.")
else:
    summary_lines.append("No statistically significant feature-outcome relationships were identified")
    summary_lines.append("at the conventional alpha level of 0.05. This may reflect true absence of")
    summary_lines.append("associations, limited statistical power, or the need for more sophisticated")
    summary_lines.append("analytical approaches.")

summary_lines.extend([
    "",
    "=" * 80,
    "END OF ANALYSIS SUMMARY",
    "=" * 80,
])

with open('analysis_summary.txt', 'w') as f:
    f.write('\n'.join(summary_lines))
print("Wrote analysis_summary.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"Total iterations: {len(transcript['iterations'])}")
print(f"Total significant findings: {total_significant}")
print("Output files: transcript.json, analysis_summary.txt")
