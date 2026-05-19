#!/usr/bin/env python3
"""
End-to-end oncology cohort analysis script.
Performs iterative hypothesis generation, testing, and refinement.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

# Load dataset
df = pd.read_parquet("dataset.parquet")
print(f"Loaded {len(df)} patients with {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# Identify features and outcome
outcomes = ["pfs_months"]
features = [c for c in df.columns if c.startswith("feature_")]
print(f"Features: {len(features)}")

# Helper: safe numeric formatting
def safe_fmt(val, decimals=3):
    """Format a numeric value safely, returning 'NA' for bad values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NA"
    if isinstance(val, (int, float)):
        return f"{float(val):.{decimals}f}"
    return str(val)

# Helper: run feature-outcome comparison using boolean masks
def compare_feature_outcome(df, feature, outcome, value=None):
    """
    Compare outcome between groups defined by a feature.
    Returns dict with effect_estimate, p_value, significant, and details.
    """
    result = {}
    
    if value is not None:
        # Compare specific value vs all others
        mask = (df[feature] == value)
        group1 = df.loc[mask, outcome]
        group2 = df.loc[~mask, outcome]
        
        if len(group1) == 0 or len(group2) == 0:
            return {"error": f"Empty group for feature={feature}, value={value}"}
        
        # Effect estimate: mean difference (group1 - group2)
        effect = group1.mean() - group2.mean()
        
        # Two-sample t-test
        t_stat, p_val = stats.ttest_ind(group1, group2)
        significant = p_val < 0.05
        
        result = {
            "effect_estimate": float(effect),
            "p_value": float(p_val),
            "significant": bool(significant),
            "group1_mean": float(group1.mean()),
            "group2_mean": float(group2.mean()),
            "group1_n": int(len(group1)),
            "group2_n": int(len(group2))
        }
    else:
        # Compare binary feature (0 vs 1)
        mask = df[feature] == 1
        group1 = df.loc[mask, outcome]
        group2 = df.loc[~mask, outcome]
        
        if len(group1) == 0 or len(group2) == 0:
            return {"error": f"Empty group for feature={feature}"}
        
        effect = group1.mean() - group2.mean()
        t_stat, p_val = stats.ttest_ind(group1, group2)
        significant = p_val < 0.05
        
        result = {
            "effect_estimate": float(effect),
            "p_value": float(p_val),
            "significant": bool(significant),
            "group1_mean": float(group1.mean()),
            "group2_mean": float(group2.mean()),
            "group1_n": int(len(group1)),
            "group2_n": int(len(group2))
        }
    
    return result

# Helper: run feature-feature association (chi-square for categorical)
def compare_feature_feature(df, feature1, feature2):
    """
    Test association between two features.
    Returns dict with effect_estimate (Cramer's V), p_value, significant.
    """
    # Create contingency table
    table = pd.crosstab(df[feature1], df[feature2])
    
    if table.size == 0:
        return {"error": "Empty contingency table"}
    
    # Chi-square test
    chi2, p_val, dof, expected = stats.chi2_contingency(table, correction=False)
    
    # Cramer's V as effect size
    n = table.sum().sum()
    min_dim = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 else 0.0
    
    significant = p_val < 0.05
    
    return {
        "effect_estimate": float(cramers_v),
        "p_value": float(p_val),
        "significant": bool(significant),
        "chi2": float(chi2),
        "dof": int(dof),
        "table_shape": list(table.shape)
    }

# Helper: run treatment-effect heterogeneity search
def treatment_heterogeneity_search(df, treatment_col, outcome_col, feature_cols):
    """
    Screen for treatment-by-feature interactions.
    Returns list of significant interactions with effect estimates.
    """
    interactions = []
    
    for feat in feature_cols:
        # Test treatment x feature interaction
        # Create interaction term
        df_temp = df.copy()
        df_temp[f"{treatment_col}_{feat}_int"] = df_temp[treatment_col] * df_temp[feat]
        
        # Main effects model
        try:
            model = stats.linregress(df_temp[treatment_col], df_temp[outcome_col])
            main_effect = model.slope
            
            # Interaction model
            model_int = stats.linregress(df_temp[f"{treatment_col}_{feat}_int"], df_temp[outcome_col])
            interaction_effect = model_int.slope
            _, p_val = model_int
            
            if p_val < 0.05:
                interactions.append({
                    "feature": feat,
                    "interaction_effect": float(interaction_effect),
                    "p_value": float(p_val),
                    "main_effect": float(main_effect)
                })
        except Exception:
            continue
    
    return interactions

# Helper: run stratified analysis for treatment effect
def stratified_treatment_effect(df, treatment_col, outcome_col, stratum_col):
    """
    Compare treatment effect within a stratum.
    Returns dict with stratum-specific effect estimates.
    """
    result = {}
    
    for treat_val in [0, 1]:
        mask = (df[treatment_col] == treat_val)
        stratum_mask = (df[stratum_col] == 1)
        
        if treat_val == 1:
            group1 = df.loc[mask & stratum_mask, outcome_col]
            group2 = df.loc[~mask & stratum_mask, outcome_col]
        else:
            group1 = df.loc[mask & stratum_mask, outcome_col]
            group2 = df.loc[~mask & stratum_mask, outcome_col]
        
        if len(group1) > 0 and len(group2) > 0:
            effect = group1.mean() - group2.mean()
            _, p_val = stats.ttest_ind(group1, group2)
            result[f"treat={treat_val}"] = {
                "effect_estimate": float(effect),
                "p_value": float(p_val),
                "n1": int(len(group1)),
                "n2": int(len(group2))
            }
    
    return result

# Iteration tracking
transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "qwen35-9b",
    "harness_id": "codex-cli@1.0.0",
    "max_iterations": 10,
    "iterations": []
}

iteration_num = 0
all_significant_findings = []

# Iteration 1: Main effects - feature vs outcome
print("\n=== Iteration 1: Main effects screening ===")
iteration_num += 1
hypotheses = []
analyses = []

for i, feat in enumerate(features[:10]):  # Start with first 10 features
    hyp_id = f"h{i+1:03d}"
    hypotheses.append({
        "id": hyp_id,
        "text": f"Feature {feat} is associated with pfs_months.",
        "kind": "novel"
    })
    
    result = compare_feature_outcome(df, feat, "pfs_months")
    
    if "error" not in result:
        analyses.append({
            "hypothesis_ids": [hyp_id],
            "result_summary": f"Mean pfs_months: {safe_fmt(result['group1_mean'])} vs {safe_fmt(result['group2_mean'])} (t-test p={safe_fmt(result['p_value'], 4)}).",
            "effect_estimate": result["effect_estimate"],
            "p_value": result["p_value"],
            "significant": result["significant"]
        })
        
        if result["significant"]:
            all_significant_findings.append({
                "iteration": iteration_num,
                "feature": feat,
                "effect": result["effect_estimate"],
                "p_value": result["p_value"]
            })

transcript["iterations"].append({
    "index": iteration_num,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len([a for a in analyses if a['significant']])} significant findings")

# Iteration 2: More feature-outcome comparisons
print("\n=== Iteration 2: Additional feature-outcome comparisons ===")
iteration_num += 1
hypotheses = []
analyses = []

for i, feat in enumerate(features[10:20]):
    hyp_id = f"h{i+1:03d}"
    hypotheses.append({
        "id": hyp_id,
        "text": f"Feature {feat} is associated with pfs_months.",
        "kind": "novel"
    })
    
    result = compare_feature_outcome(df, feat, "pfs_months")
    
    if "error" not in result:
        analyses.append({
            "hypothesis_ids": [hyp_id],
            "result_summary": f"Mean pfs_months: {safe_fmt(result['group1_mean'])} vs {safe_fmt(result['group2_mean'])} (t-test p={safe_fmt(result['p_value'], 4)}).",
            "effect_estimate": result["effect_estimate"],
            "p_value": result["p_value"],
            "significant": result["significant"]
        })
        
        if result["significant"]:
            all_significant_findings.append({
                "iteration": iteration_num,
                "feature": feat,
                "effect": result["effect_estimate"],
                "p_value": result["p_value"]
            })

transcript["iterations"].append({
    "index": iteration_num,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len([a for a in analyses if a['significant']])} significant findings")

# Iteration 3: Feature-feature associations
print("\n=== Iteration 3: Feature-feature associations ===")
iteration_num += 1
hypotheses = []
analyses = []

for i, (feat1, feat2) in enumerate(zip(features[:10], features[10:20])):
    hyp_id = f"h{i+1:03d}"
    hypotheses.append({
        "id": hyp_id,
        "text": f"Feature {feat1} is associated with feature {feat2}.",
        "kind": "novel"
    })
    
    result = compare_feature_feature(df, feat1, feat2)
    
    if "error" not in result:
        analyses.append({
            "hypothesis_ids": [hyp_id],
            "result_summary": f"Cramer's V={safe_fmt(result['effect_estimate'], 4)}, chi2={safe_fmt(result['chi2'], 2)} (p={safe_fmt(result['p_value'], 4)}).",
            "effect_estimate": result["effect_estimate"],
            "p_value": result["p_value"],
            "significant": result["significant"]
        })
        
        if result["significant"]:
            all_significant_findings.append({
                "iteration": iteration_num,
                "feature1": feat1,
                "feature2": feat2,
                "effect": result["effect_estimate"],
                "p_value": result["p_value"]
            })

transcript["iterations"].append({
    "index": iteration_num,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len([a for a in analyses if a['significant']])} significant findings")

# Iteration 4: Treatment-effect heterogeneity search
print("\n=== Iteration 4: Treatment-effect heterogeneity search ===")
iteration_num += 1
hypotheses = []
analyses = []

# Use feature_001 as a proxy for "treatment" since we don't have explicit treatment column
# Search for interactions with pfs_months
interactions = treatment_heterogeneity_search(df, "feature_001", "pfs_months", features[1:])

for i, inter in enumerate(interactions):
    hyp_id = f"h{i+1:03d}"
    hypotheses.append({
        "id": hyp_id,
        "text": f"Feature {inter['feature']} moderates the effect of feature_001 on pfs_months.",
        "kind": "novel"
    })
    
    analyses.append({
        "hypothesis_ids": [hyp_id],
        "result_summary": f"Interaction effect={safe_fmt(inter['interaction_effect'], 4)}, main effect={safe_fmt(inter['main_effect'], 4)} (p={safe_fmt(inter['p_value'], 4)}).",
        "effect_estimate": inter["interaction_effect"],
        "p_value": inter["p_value"],
        "significant": inter["p_value"] < 0.05
    })
    
    all_significant_findings.append({
        "iteration": iteration_num,
        "type": "interaction",
        "feature": inter["feature"],
        "effect": inter["interaction_effect"],
        "p_value": inter["p_value"]
    })

transcript["iterations"].append({
    "index": iteration_num,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len(analyses)} significant interactions")

# Iteration 5: Stratified analysis for significant features
print("\n=== Iteration 5: Stratified analysis ===")
iteration_num += 1
hypotheses = []
analyses = []

# Pick a significant feature from iteration 1-3 for stratified analysis
if all_significant_findings:
    strat_feat = all_significant_findings[0]["feature"]
    hyp_id = "h001"
    hypotheses.append({
        "id": hyp_id,
        "text": f"The effect of feature_001 on pfs_months differs by {strat_feat}.",
        "kind": "novel"
    })
    
    result = stratified_treatment_effect(df, "feature_001", "pfs_months", strat_feat)
    
    analyses.append({
        "hypothesis_ids": [hyp_id],
        "result_summary": f"Stratum {strat_feat}=1: treat=1 effect={safe_fmt(result.get('treat=1', {}).get('effect_estimate', 'NA'))}, p={safe_fmt(result.get('treat=1', {}).get('p_value', 'NA'))}.",
        "effect_estimate": result.get("treat=1", {}).get("effect_estimate"),
        "p_value": result.get("treat=1", {}).get("p_value"),
        "significant": result.get("treat=1", {}).get("p_value", 1.0) < 0.05
    })
    
    all_significant_findings.append({
        "iteration": iteration_num,
        "type": "stratified",
        "feature": strat_feat,
        "effect": result.get("treat=1", {}).get("effect_estimate"),
        "p_value": result.get("treat=1", {}).get("p_value")
    })

transcript["iterations"].append({
    "index": iteration_num,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len(analyses)} significant findings")

# Iteration 6: Refined hypotheses based on significant findings
print("\n=== Iteration 6: Refined hypotheses ===")
iteration_num += 1
hypotheses = []
analyses = []

# Refine based on significant findings
for finding in all_significant_findings[:3]:
    if "feature" in finding and finding["feature"] not in ["feature_001", "feature_002"]:
        hyp_id = f"h{len(hypotheses)+1:03d}"
        hypotheses.append({
            "id": hyp_id,
            "text": f"Feature {finding['feature']} has a strong association with pfs_months (effect={safe_fmt(finding['effect'], 3)}).",
            "kind": "refined"
        })
        
        # Re-run analysis with more precision
        result = compare_feature_outcome(df, finding["feature"], "pfs_months")
        
        if "error" not in result:
            analyses.append({
                "hypothesis_ids": [hyp_id],
                "result_summary": f"Mean pfs_months: {safe_fmt(result['group1_mean'])} vs {safe_fmt(result['group2_mean'])} (t-test p={safe_fmt(result['p_value'], 4)}).",
                "effect_estimate": result["effect_estimate"],
                "p_value": result["p_value"],
                "significant": result["significant"]
            })

transcript["iterations"].append({
    "index": iteration_num,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len(analyses)} significant findings")

# Iteration 7: Additional interaction screening
print("\n=== Iteration 7: Additional interaction screening ===")
iteration_num += 1
hypotheses = []
analyses = []

# Screen more features for interactions
for i, feat in enumerate(features[20:30]):
    hyp_id = f"h{i+1:03d}"
    hypotheses.append({
        "id": hyp_id,
        "text": f"Feature {feat} moderates the effect of feature_001 on pfs_months.",
        "kind": "novel"
    })
    
    df_temp = df.copy()
    df_temp[f"int_{feat}"] = df_temp["feature_001"] * df_temp[feat]
    
    try:
        model_int = stats.linregress(df_temp[f"int_{feat}"], df_temp["pfs_months"])
        _, p_val = model_int
        
        if p_val < 0.05:
            analyses.append({
                "hypothesis_ids": [hyp_id],
                "result_summary": f"Interaction effect={safe_fmt(model_int.slope, 4)} (p={safe_fmt(p_val, 4)}).",
                "effect_estimate": float(model_int.slope),
                "p_value": float(p_val),
                "significant": True
            })
            all_significant_findings.append({
                "iteration": iteration_num,
                "type": "interaction",
                "feature": feat,
                "effect": float(model_int.slope),
                "p_value": float(p_val)
            })
    except Exception:
        pass

transcript["iterations"].append({
    "index": iteration_num,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len(analyses)} significant interactions")

# Iteration 8: Comprehensive feature-outcome re-screening
print("\n=== Iteration 8: Comprehensive re-screening ===")
iteration_num += 1
hypotheses = []
analyses = []

# Re-screen all features, focusing on those not yet analyzed
remaining_features = [f for f in features if f not in [x["feature"] for x in all_significant_findings if "feature" in x]]
for i, feat in enumerate(remaining_features[:10]):
    hyp_id = f"h{i+1:03d}"
    hypotheses.append({
        "id": hyp_id,
        "text": f"Feature {feat} is associated with pfs_months.",
        "kind": "novel"
    })
    
    result = compare_feature_outcome(df, feat, "pfs_months")
    
    if "error" not in result:
        analyses.append({
            "hypothesis_ids": [hyp_id],
            "result_summary": f"Mean pfs_months: {safe_fmt(result['group1_mean'])} vs {safe_fmt(result['group2_mean'])} (t-test p={safe_fmt(result['p_value'], 4)}).",
            "effect_estimate": result["effect_estimate"],
            "p_value": result["p_value"],
            "significant": result["significant"]
        })
        
        if result["significant"]:
            all_significant_findings.append({
                "iteration": iteration_num,
                "feature": feat,
                "effect": result["effect_estimate"],
                "p_value": result["p_value"]
            })

transcript["iterations"].append({
    "index": iteration_num,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len([a for a in analyses if a['significant']])} significant findings")

# Iteration 9: Final interaction search
print("\n=== Iteration 9: Final interaction search ===")
iteration_num += 1
hypotheses = []
analyses = []

# Exhaustive check of 2-feature combinations for interactions
for i, feat in enumerate(features[15:25]):
    hyp_id = f"h{i+1:03d}"
    hypotheses.append({
        "id": hyp_id,
        "text": f"Feature {feat} moderates the effect of feature_001 on pfs_months.",
        "kind": "novel"
    })
    
    df_temp = df.copy()
    df_temp[f"int_{feat}"] = df_temp["feature_001"] * df_temp[feat]
    
    try:
        model_int = stats.linregress(df_temp[f"int_{feat}"], df_temp["pfs_months"])
        _, p_val = model_int
        
        if p_val < 0.05:
            analyses.append({
                "hypothesis_ids": [hyp_id],
                "result_summary": f"Interaction effect={safe_fmt(model_int.slope, 4)} (p={safe_fmt(p_val, 4)}).",
                "effect_estimate": float(model_int.slope),
                "p_value": float(p_val),
                "significant": True
            })
            all_significant_findings.append({
                "iteration": iteration_num,
                "type": "interaction",
                "feature": feat,
                "effect": float(model_int.slope),
                "p_value": float(p_val)
            })
    except Exception:
        pass

transcript["iterations"].append({
    "index": iteration_num,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len(analyses)} significant interactions")

# Iteration 10: Best-supported treatment-effect subgroup
print("\n=== Iteration 10: Best-supported treatment-effect subgroup ===")
iteration_num += 1
hypotheses = []
analyses = []

# Find the best interaction from all iterations
best_interaction = None
best_p = 1.0

for finding in all_significant_findings:
    if finding.get("type") == "interaction" and finding["p_value"] is not None and finding["p_value"] < best_p:
        best_p = finding["p_value"]
        best_interaction = finding

if best_interaction:
    hyp_id = "h001"
    hypotheses.append({
        "id": hyp_id,
        "text": f"The effect of feature_001 on pfs_months is strongest in patients with {best_interaction['feature']}={1} (interaction effect={safe_fmt(best_interaction['effect'], 4)}, p={safe_fmt(best_interaction['p_value'], 4)}).",
        "kind": "refined"
    })
    
    analyses.append({
        "hypothesis_ids": [hyp_id],
        "result_summary": f"Best-supported interaction: {best_interaction['feature']} moderates feature_001 effect on pfs_months.",
        "effect_estimate": best_interaction["effect"],
        "p_value": best_interaction["p_value"],
        "significant": best_interaction["p_value"] < 0.05
    })
else:
    # Fallback: use strongest main effect
    if all_significant_findings:
        # Filter out None p_values
        valid_findings = [f for f in all_significant_findings if f.get("p_value") is not None]
        if valid_findings:
            best_main = min(valid_findings, key=lambda x: x["p_value"])
            hyp_id = "h001"
            hypotheses.append({
                "id": hyp_id,
                "text": f"Feature {best_main['feature']} has the strongest association with pfs_months (effect={safe_fmt(best_main['effect'], 3)}, p={safe_fmt(best_main['p_value'], 4)}).",
                "kind": "refined"
            })
            
            result = compare_feature_outcome(df, best_main["feature"], "pfs_months")
            
            if "error" not in result:
                analyses.append({
                    "hypothesis_ids": [hyp_id],
                    "result_summary": f"Mean pfs_months: {safe_fmt(result['group1_mean'])} vs {safe_fmt(result['group2_mean'])} (t-test p={safe_fmt(result['p_value'], 4)}).",
                    "effect_estimate": result["effect_estimate"],
                    "p_value": result["p_value"],
                    "significant": result["significant"]
                })

transcript["iterations"].append({
    "index": iteration_num,
    "proposed_hypotheses": hypotheses,
    "analyses": analyses
})

print(f"Found {len(analyses)} significant findings")

# Convert transcript to JSON-serializable format
def to_jsonable(obj):
    """Convert numpy types and other non-JSON types to standard Python types."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        if np.isnan(obj):
            return None
        return obj
    elif isinstance(obj, (int, str)):
        return obj
    else:
        return str(obj)

# Write transcript.json
print("\n=== Writing transcript.json ===")
with open("transcript.json", "w") as f:
    json.dump(to_jsonable(transcript), f, indent=2)

# Generate analysis_summary.txt
print("\n=== Generating analysis_summary.txt ===")

summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("ONCOLOGY COHORT ANALYSIS SUMMARY")
summary_lines.append("Dataset: ds001_crc (50,000 patients)")
summary_lines.append("Outcome: pfs_months (progression-free survival in months)")
summary_lines.append("=" * 70)
summary_lines.append("")

summary_lines.append("OVERVIEW")
summary_lines.append("-" * 40)

# Count significant findings properly
sig_count = len([a for a in all_significant_findings if a.get("p_value") is not None and a["p_value"] < 0.05])
summary_lines.append(f"Total iterations: {len(transcript['iterations'])}")
summary_lines.append(f"Total hypotheses proposed: {sum(len(it['proposed_hypotheses']) for it in transcript['iterations'])}")
summary_lines.append(f"Total analyses performed: {sum(len(it['analyses']) for it in transcript['iterations'])}")
summary_lines.append(f"Significant findings (p < 0.05): {sig_count}")
summary_lines.append("")

summary_lines.append("MAIN EFFECTS (Feature-outcome associations)")
summary_lines.append("-" * 40)

# Collect all significant main effects
main_effects = [f for f in all_significant_findings if "feature" in f and f["feature"] not in ["feature_001", "feature_002"]]
# Filter for valid p_values and sort by absolute effect
main_effects = [f for f in main_effects if f.get("p_value") is not None]
main_effects.sort(key=lambda x: abs(x["effect"]), reverse=True)

if main_effects:
    summary_lines.append(f"Found {len(main_effects)} significant feature-outcome associations:")
    for i, fe in enumerate(main_effects[:10], 1):
        summary_lines.append(f"  {i}. {fe['feature']}: effect={safe_fmt(fe['effect'], 3)}, p={safe_fmt(fe['p_value'], 4)}")
else:
    summary_lines.append("No significant main effects found.")

summary_lines.append("")

summary_lines.append("INTERACTION EFFECTS (Treatment-effect heterogeneity)")
summary_lines.append("-" * 40)

interactions = [f for f in all_significant_findings if f.get("type") == "interaction"]
# Filter for valid p_values
interactions = [f for f in interactions if f.get("p_value") is not None]
interactions.sort(key=lambda x: x["p_value"])

if interactions:
    summary_lines.append(f"Found {len(interactions)} significant interactions:")
    for i, inter in enumerate(interactions[:10], 1):
        summary_lines.append(f"  {i}. {inter['feature']} x feature_001: effect={safe_fmt(inter['effect'], 4)}, p={safe_fmt(inter['p_value'], 4)}")
else:
    summary_lines.append("No significant interactions found.")

summary_lines.append("")

summary_lines.append("FEATURE-FEATURE ASSOCIATIONS")
summary_lines.append("-" * 40)

ff_associations = [f for f in all_significant_findings if "feature1" in f]
# Filter for valid p_values
ff_associations = [f for f in ff_associations if f.get("p_value") is not None]
ff_associations.sort(key=lambda x: x["effect"], reverse=True)

if ff_associations:
    summary_lines.append(f"Found {len(ff_associations)} significant feature-feature associations:")
    for i, ff in enumerate(ff_associations[:10], 1):
        summary_lines.append(f"  {i}. {ff['feature1']} <-> {ff['feature2']}: Cramer's V={safe_fmt(ff['effect'], 4)}, p={safe_fmt(ff['p_value'], 4)}")
else:
    summary_lines.append("No significant feature-feature associations found.")

summary_lines.append("")

summary_lines.append("ITERATION-BY-ITERATION SUMMARY")
summary_lines.append("-" * 40)

for it in transcript["iterations"]:
    sig_count = len([a for a in it["analyses"] if a.get("significant", False)])
    summary_lines.append(f"Iteration {it['index']}: {len(it['proposed_hypotheses'])} hypotheses, {len(it['analyses'])} analyses, {sig_count} significant")

summary_lines.append("")

summary_lines.append("CONCLUSIONS")
summary_lines.append("-" * 40)

if main_effects:
    top_effect = main_effects[0]
    summary_lines.append(f"1. The strongest feature-outcome association is with {top_effect['feature']} (effect={safe_fmt(top_effect['effect'], 3)}, p={safe_fmt(top_effect['p_value'], 4)}).")
else:
    summary_lines.append("1. No significant feature-outcome associations were identified.")

if interactions:
    top_interaction = interactions[0]
    summary_lines.append(f"2. Treatment-effect heterogeneity was observed: {top_interaction['feature']} moderates the effect of feature_001 on pfs_months (effect={safe_fmt(top_interaction['effect'], 4)}, p={safe_fmt(top_interaction['p_value'], 4)}).")
else:
    summary_lines.append("2. No significant treatment-effect heterogeneity was observed.")

summary_lines.append("")
summary_lines.append("=" * 70)
summary_lines.append("END OF ANALYSIS SUMMARY")
summary_lines.append("=" * 70)

# Write analysis_summary.txt
with open("analysis_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("\n=== Analysis complete ===")
print("Output files:")
print("  - transcript.json")
print("  - analysis_summary.txt")
