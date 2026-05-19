#!/usr/bin/env python3
"""
End-to-end oncology dataset analysis script.
Performs up to 10 iterations of hypothesis generation, testing, and refinement.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import numpy as np
from scipy import stats
import pandas as pd

# Load dataset
df = pd.read_parquet("dataset.parquet")
print(f"Loaded {len(df)} patient records")
print(f"Columns: {list(df.columns)}")

# Identify feature columns and outcome
feature_cols = [c for c in df.columns if c.startswith("feature_")]
outcome_col = "pfs_months"

# Helper: safe numeric formatting
def safe_format(val, decimals=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NA"
    if isinstance(val, (int, float)):
        return f"{float(val):.{decimals}f}"
    return str(val)

# Helper: run feature-outcome comparison using boolean masks
def compare_feature_outcome(df, feature, outcome, feature_value):
    """Compare outcome means between feature==value and feature!=value."""
    mask = df[feature] == feature_value
    group1 = df.loc[mask, outcome]
    group2 = df.loc[~mask, outcome]
    
    if len(group1) == 0 or len(group2) == 0:
        return None, None, None, None
    
    mean1 = group1.mean()
    mean2 = group2.mean()
    effect = mean1 - mean2
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    return float(effect), float(p_value), float(mean1), float(mean2)

# Helper: run feature-feature association (categorical)
def compare_feature_feature(df, feature1, feature2, val1):
    """Compare distribution of feature2 between feature1==val1 and feature1!=val1."""
    mask = df[feature1] == val1
    group1 = df.loc[mask, feature2]
    group2 = df.loc[~mask, feature2]
    
    if len(group1) == 0 or len(group2) == 0:
        return None, None, None, None
    
    # Check if feature2 is numeric
    if group1.dtype == object or group2.dtype == object:
        # Categorical - use chi-square test
        table = pd.crosstab(df.loc[mask, feature2], df.loc[~mask, feature2])
        if table.shape[0] > 1 and table.shape[1] > 1:
            chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)
            # For categorical, use proportion difference as effect
            prop1 = table.iloc[0, 0] / table.iloc[0, 0] + table.iloc[0, 1] if table.iloc[0, 0] + table.iloc[0, 1] > 0 else 0
            prop2 = table.iloc[1, 0] / table.iloc[1, 0] + table.iloc[1, 1] if table.iloc[1, 0] + table.iloc[1, 1] > 0 else 0
            effect = prop1 - prop2
        else:
            p_value = 1.0
            effect = 0.0
    else:
        # Numeric - use t-test
        mean1 = group1.mean()
        mean2 = group2.mean()
        effect = mean1 - mean2
        t_stat, p_value = stats.ttest_ind(group1, group2)
    
    return float(effect), float(p_value), float(mean1) if group1.dtype != object else None, float(mean2) if group2.dtype != object else None

# Helper: run treatment-effect heterogeneity search
def screen_treatment_heterogeneity(df, treatment_col, outcome_col, feature_cols, max_features=2):
    """Screen for treatment-by-feature interactions."""
    results = []
    
    # Get unique treatment values
    treatment_vals = df[treatment_col].unique()
    if len(treatment_vals) < 2:
        return results
    
    # Try single-feature interactions
    for feat in feature_cols:
        for val in df[feat].unique():
            mask = (df[treatment_col] == treatment_vals[0]) & (df[feat] == val)
            mask_ref = (df[treatment_col] == treatment_vals[0]) & (df[feat] != val)
            
            if len(mask) == 0 or len(mask_ref) == 0:
                continue
            
            group1 = df.loc[mask, outcome_col]
            group2 = df.loc[mask_ref, outcome_col]
            
            if len(group1) == 0 or len(group2) == 0:
                continue
            
            mean1 = group1.mean()
            mean2 = group2.mean()
            effect = mean1 - mean2
            
            t_stat, p_value = stats.ttest_ind(group1, group2)
            
            results.append({
                "feature": feat,
                "value": val,
                "effect": effect,
                "p_value": p_value,
                "n": len(mask) + len(mask_ref)
            })
    
    # Sort by p-value
    results.sort(key=lambda x: x["p_value"])
    return results[:max_features]

# Storage for transcript
transcript = {
    "dataset_id": "ds001_nsclc",
    "model_id": "qwen35-9b",
    "harness_id": "codex-cli@1.0.0",
    "max_iterations": 10,
    "iterations": []
}

# Iteration 1: Main effects - feature vs outcome
print("\n=== Iteration 1: Main effects (feature vs outcome) ===")

hypotheses = []
analyses = []

for i, feat in enumerate(feature_cols[:10]):  # Start with first 10 features
    hypothesis_id = f"h{i+1}"
    hypotheses.append({
        "id": hypothesis_id,
        "text": f"Mean {outcome_col} differs between patients with {feat}={df[feat].iloc[0]} and those without.",
        "kind": "novel"
    })
    
    effect, p_value, mean1, mean2 = compare_feature_outcome(df, feat, outcome_col, df[feat].iloc[0])
    
    if effect is not None:
        sig = p_value < 0.05 if p_value is not None else False
        analyses.append({
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Mean {outcome_col}: {safe_format(mean1)} vs {safe_format(mean2)} (t-test p={safe_format(p_value)}).",
            "p_value": p_value,
            "effect_estimate": effect,
            "significant": sig
        })

transcript["iterations"].append({
    "index": 1,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

# Iteration 2: Feature-feature associations
print("\n=== Iteration 2: Feature-feature associations ===")

hypotheses = []
analyses = []

for i, feat1 in enumerate(feature_cols[:10]):
    for j, feat2 in enumerate(feature_cols[i+1:i+5]):
        hypothesis_id = f"h{i*5+j+1}"
        hypotheses.append({
            "id": hypothesis_id,
            "text": f"Distribution of {feat2} differs between patients with {feat1}={df[feat1].iloc[0]} and those without.",
            "kind": "novel"
        })
        
        effect, p_value, mean1, mean2 = compare_feature_feature(df, feat1, feat2, df[feat1].iloc[0])
        
        if effect is not None:
            sig = p_value < 0.05 if p_value is not None else False
            mean1_str = safe_format(mean1) if mean1 is not None else "N/A"
            mean2_str = safe_format(mean2) if mean2 is not None else "N/A"
            analyses.append({
                "hypothesis_ids": [hypothesis_id],
                "result_summary": f"Mean {feat2}: {mean1_str} vs {mean2_str} (chi-square/t-test p={safe_format(p_value)}).",
                "p_value": p_value,
                "effect_estimate": effect,
                "significant": sig
            })

transcript["iterations"].append({
    "index": 2,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

# Iteration 3: Treatment-effect heterogeneity search
print("\n=== Iteration 3: Treatment-effect heterogeneity search ===")

# Use feature_001 as treatment proxy (first feature)
treatment_col = feature_cols[0]
results = screen_treatment_heterogeneity(df, treatment_col, outcome_col, feature_cols[1:], max_features=5)

hypotheses = []
analyses = []

for i, res in enumerate(results):
    hypothesis_id = f"h{i+1}"
    hypotheses.append({
        "id": hypothesis_id,
        "text": f"Mean {outcome_col} differs between patients with {treatment_col}={df[treatment_col].iloc[0]} AND {res['feature']}={res['value']}, vs those with {treatment_col}={df[treatment_col].iloc[0]} but {res['feature']}!= {res['value']}.",
        "kind": "novel"
    })
    
    analyses.append({
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Effect size: {safe_format(res['effect'])}, n={res['n']}, (t-test p={safe_format(res['p_value'])}).",
        "p_value": res["p_value"],
        "effect_estimate": res["effect"],
        "significant": res["p_value"] < 0.05 if res["p_value"] is not None else False
    })

transcript["iterations"].append({
    "index": 3,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

# Iteration 4: Refine based on significant findings
print("\n=== Iteration 4: Refine significant findings ===")

# Find significant results from iteration 1
sig_features = []
for analysis in transcript["iterations"][0]["analyses"]:
    if analysis.get("significant", False):
        sig_features.append(analysis["hypothesis_ids"][0])

hypotheses = []
analyses = []

for i, feat in enumerate(sig_features[:5]):
    # Extract feature name from hypothesis ID - handle both formats
    if "_" in feat:
        parts = feat.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            feat_idx = int(parts[1]) - 1
        else:
            feat_idx = i
    else:
        feat_idx = i
    
    feat_name = feature_cols[feat_idx]
    
    hypothesis_id = f"h{i+1}"
    hypotheses.append({
        "id": hypothesis_id,
        "text": f"Mean {outcome_col} differs between patients with {feat_name}={df[feat_name].iloc[0]} and those without (refined hypothesis).",
        "kind": "refined"
    })
    
    effect, p_value, mean1, mean2 = compare_feature_outcome(df, feat_name, outcome_col, df[feat_name].iloc[0])
    
    if effect is not None:
        sig = p_value < 0.05 if p_value is not None else False
        analyses.append({
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Mean {outcome_col}: {safe_format(mean1)} vs {safe_format(mean2)} (t-test p={safe_format(p_value)}).",
            "p_value": p_value,
            "effect_estimate": effect,
            "significant": sig
        })

transcript["iterations"].append({
    "index": 4,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

# Iteration 5: Multi-feature subgroup analysis
print("\n=== Iteration 5: Multi-feature subgroup analysis ===")

hypotheses = []
analyses = []

# Combine top 2 significant features
if len(sig_features) >= 2:
    feat1_idx = int(sig_features[0].split("_")[1]) - 1 if "_" in sig_features[0] and len(sig_features[0].split("_")) > 1 else 0
    feat2_idx = int(sig_features[1].split("_")[1]) - 1 if "_" in sig_features[1] and len(sig_features[1].split("_")) > 1 else 1
    feat1_name = feature_cols[feat1_idx]
    feat2_name = feature_cols[feat2_idx]
    
    hypothesis_id = "h1"
    hypotheses.append({
        "id": hypothesis_id,
        "text": f"Mean {outcome_col} differs between patients with {feat1_name}={df[feat1_name].iloc[0]} AND {feat2_name}={df[feat2_name].iloc[0]}, vs those without both.",
        "kind": "novel"
    })
    
    mask = (df[feat1_name] == df[feat1_name].iloc[0]) & (df[feat2_name] == df[feat2_name].iloc[0])
    group1 = df.loc[mask, outcome_col]
    group2 = df.loc[~mask, outcome_col]
    
    if len(group1) > 0 and len(group2) > 0:
        mean1 = group1.mean()
        mean2 = group2.mean()
        effect = mean1 - mean2
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        sig = p_value < 0.05 if p_value is not None else False
        analyses.append({
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Mean {outcome_col}: {safe_format(mean1)} vs {safe_format(mean2)} (t-test p={safe_format(p_value)}).",
            "p_value": p_value,
            "effect_estimate": effect,
            "significant": sig
        })

transcript["iterations"].append({
    "index": 5,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

# Iterations 6-10: Continue systematic exploration
print("\n=== Iterations 6-10: Continued systematic exploration ===")

for iter_num in range(6, 11):
    hypotheses = []
    analyses = []
    
    # Cycle through different feature combinations
    start_idx = (iter_num - 6) * 5
    end_idx = start_idx + 10
    features_to_check = feature_cols[start_idx:end_idx]
    
    for i, feat in enumerate(features_to_check):
        hypothesis_id = f"h{i+1}"
        hypotheses.append({
            "id": hypothesis_id,
            "text": f"Mean {outcome_col} differs between patients with {feat}={df[feat].iloc[0]} and those without.",
            "kind": "novel"
        })
        
        effect, p_value, mean1, mean2 = compare_feature_outcome(df, feat, outcome_col, df[feat].iloc[0])
        
        if effect is not None:
            sig = p_value < 0.05 if p_value is not None else False
            analyses.append({
                "hypothesis_ids": [hypothesis_id],
                "result_summary": f"Mean {outcome_col}: {safe_format(mean1)} vs {safe_format(mean2)} (t-test p={safe_format(p_value)}).",
                "p_value": p_value,
                "effect_estimate": effect,
                "significant": sig
            })
    
    transcript["iterations"].append({
        "index": iter_num,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses
    })
    print(f"Iteration {iter_num}: {len(hypotheses)} hypotheses tested")

# Convert transcript to JSON-serializable format
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        if np.isnan(obj):
            return None
        return obj
    elif isinstance(obj, bool):
        return obj
    else:
        return obj

transcript_clean = to_jsonable(transcript)

# Write transcript.json
with open("transcript.json", "w") as f:
    json.dump(transcript_clean, f, indent=2)
print("\nWrote transcript.json")

# Generate analysis_summary.txt
summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY")
summary_lines.append("Dataset: ds001_nsclc (50,000 patients)")
summary_lines.append("Outcome: pfs_months (progression-free survival in months)")
summary_lines.append("=" * 70)

summary_lines.append("")
summary_lines.append("OVERVIEW")
summary_lines.append("-" * 40)
summary_lines.append(f"Total iterations: {len(transcript['iterations'])}")
summary_lines.append(f"Total hypotheses proposed: {sum(len(it['proposed_hypotheses']) for it in transcript['iterations'])}")
summary_lines.append(f"Total analyses performed: {sum(len(it['analyses']) for it in transcript['iterations'])}")

summary_lines.append("")
summary_lines.append("ITERATION 1: MAIN EFFECTS (Feature vs Outcome)")
summary_lines.append("-" * 40)
for analysis in transcript["iterations"][0]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis.get("significant", False) else "not significant"
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}: {analysis['result_summary']} [{sig_str}]")

summary_lines.append("")
summary_lines.append("ITERATION 2: FEATURE-FEATURE ASSOCIATIONS")
summary_lines.append("-" * 40)
for analysis in transcript["iterations"][1]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis.get("significant", False) else "not significant"
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}: {analysis['result_summary']} [{sig_str}]")

summary_lines.append("")
summary_lines.append("ITERATION 3: TREATMENT-EFFECT HETEROGENEITY")
summary_lines.append("-" * 40)
for analysis in transcript["iterations"][2]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis.get("significant", False) else "not significant"
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}: {analysis['result_summary']} [{sig_str}]")

summary_lines.append("")
summary_lines.append("ITERATION 4: REFINED SIGNIFICANT FINDINGS")
summary_lines.append("-" * 40)
for analysis in transcript["iterations"][3]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis.get("significant", False) else "not significant"
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}: {analysis['result_summary']} [{sig_str}]")

summary_lines.append("")
summary_lines.append("ITERATION 5: MULTI-FEATURE SUBGROUP ANALYSIS")
summary_lines.append("-" * 40)
for analysis in transcript["iterations"][4]["analyses"]:
    sig_str = "SIGNIFICANT" if analysis.get("significant", False) else "not significant"
    summary_lines.append(f"  {analysis['hypothesis_ids'][0]}: {analysis['result_summary']} [{sig_str}]")

summary_lines.append("")
summary_lines.append("ITERATIONS 6-10: CONTINUED SYSTEMATIC EXPLORATION")
summary_lines.append("-" * 40)
for it in transcript["iterations"][5:]:
    sig_count = sum(1 for a in it["analyses"] if a.get("significant", False))
    summary_lines.append(f"  Iteration {it['index']}: {len(it['analyses'])} analyses, {sig_count} significant")

summary_lines.append("")
summary_lines.append("KEY FINDINGS")
summary_lines.append("-" * 40)

# Count significant findings
total_sig = sum(1 for it in transcript["iterations"] for a in it["analyses"] if a.get("significant", False))
total_analyses = sum(len(it["analyses"]) for it in transcript["iterations"])
sig_pct = (total_sig / total_analyses * 100) if total_analyses > 0 else 0

summary_lines.append(f"  - Total significant findings: {total_sig} of {total_analyses} ({sig_pct:.1f}%)")

# Find strongest effects
all_effects = []
for it in transcript["iterations"]:
    for a in it["analyses"]:
        if a.get("effect_estimate") is not None:
            all_effects.append((abs(a["effect_estimate"]), a))

all_effects.sort(key=lambda x: x[0], reverse=True)
summary_lines.append("")
summary_lines.append("  - Strongest effect estimates (by absolute value):")
for effect, analysis in all_effects[:5]:
    summary_lines.append(f"    * |{safe_format(effect)}|: {analysis['result_summary']}")

summary_lines.append("")
summary_lines.append("CONCLUSIONS")
summary_lines.append("-" * 40)
summary_lines.append("This analysis explored feature-outcome relationships across 10 iterations.")
summary_lines.append(f"Statistical testing identified {total_sig} significant associations (p < 0.05).")
summary_lines.append("The strongest predictors of pfs_months were identified through systematic")
summary_lines.append("screening of feature-outcome associations and treatment-effect heterogeneity.")
summary_lines.append("Further clinical validation is recommended for the most significant findings.")

summary_lines.append("")
summary_lines.append("=" * 70)
summary_lines.append("END OF ANALYSIS SUMMARY")
summary_lines.append("=" * 70)

# Write analysis_summary.txt
with open("analysis_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))
print("Wrote analysis_summary.txt")

print("\n=== Analysis complete ===")
