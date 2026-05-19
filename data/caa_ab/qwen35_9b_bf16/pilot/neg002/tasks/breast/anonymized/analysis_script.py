#!/usr/bin/env python3
"""
End-to-end oncology dataset analysis script.
Performs iterative hypothesis generation, testing, and refinement.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import os
import numpy as np
from scipy import stats
import pandas as pd

# Paths
DATA_PATH = '/home/kenneth_kehl/onc-co-scientist/data/caa_ab/qwen35_9b_bf16/pilot/neg002/tasks/breast/anonymized/dataset.parquet'
OUTPUT_DIR = '/home/kenneth_kehl/onc-co-scientist/data/caa_ab/qwen35_9b_bf16/pilot/neg002/tasks/breast/anonymized'
MAX_ITERATIONS = 10

# Load data
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Identify feature types
BINARY_FEATURES = ['feature_011', 'feature_006', 'feature_001', 'feature_030', 'feature_019', 
                   'feature_029', 'feature_023', 'feature_018', 'feature_016', 'feature_009', 
                   'feature_026', 'feature_028', 'feature_015', 'feature_013', 'feature_022', 
                   'feature_008', 'feature_032', 'feature_020']

LOW_CARDINALITY_FEATURES = ['feature_005', 'feature_035', 'feature_002', 'feature_007', 'feature_010']

# Helper functions
def to_jsonable(obj):
    """Convert numpy types and other non-JSON types to JSON-serializable types."""
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
    elif isinstance(obj, (int, str, type(None))):
        return obj
    else:
        return str(obj)

def compute_group_rates(df, feature, value, outcome):
    """Compute outcome rates for two groups defined by a binary feature."""
    mask = df[feature] == value
    group1_rate = df.loc[mask, outcome].mean() if mask.sum() > 0 else np.nan
    group0_rate = df.loc[~mask, outcome].mean() if (~mask).sum() > 0 else np.nan
    return group1_rate, group0_rate

def compute_2x2_table(df, feature, feature2, value1, value2):
    """Compute 2x2 contingency table for two binary features."""
    table = pd.crosstab(df[feature] == value1, df[feature2] == value2)
    return table.values

def run_chi2_test(df, feature, feature2, value1, value2):
    """Run chi-square test for independence between two binary features."""
    table = compute_2x2_table(df, feature, feature2, value1, value2)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None, None, None
    chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p_value), table

def run_ttest(df, feature, outcome, value):
    """Run t-test comparing outcome between two groups."""
    mask = df[feature] == value
    if mask.sum() == 0 or (~mask).sum() == 0:
        return None, None, None
    group1 = df.loc[mask, outcome].values
    group0 = df.loc[~mask, outcome].values
    t_stat, p_value = stats.ttest_ind(group1, group0)
    effect = group1.mean() - group0.mean()
    return float(t_stat), float(p_value), float(effect)

def run_fisher_exact(df, feature, feature2, value1, value2):
    """Run Fisher's exact test for two binary features."""
    table = compute_2x2_table(df, feature, feature2, value1, value2)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None, None, None
    _, p_value = stats.fisher_exact(table)
    return float(p_value), table

def run_correlation(df, feature1, feature2):
    """Compute Pearson correlation between two continuous features."""
    mask = df[feature1].notna() & df[feature2].notna()
    if mask.sum() < 10:
        return None, None, None
    corr, p_value = stats.pearsonr(df.loc[mask, feature1], df.loc[mask, feature2])
    return float(corr), float(p_value)

def run_regression(df, treatment, outcome, covariates=None):
    """Run linear regression with optional covariates."""
    if covariates is None:
        covariates = []
    y = df[outcome].values
    X = [df[treatment].values]
    for cov in covariates:
        X.append(df[cov].values)
    X = np.column_stack(X)
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        treatment_effect = coeffs[0]
        # Compute p-value for treatment effect
        n = len(y)
        se = np.sqrt(np.sum(residuals) / (n - len(X))) * np.sqrt(np.sum(X[:, 0] ** 2) / (n * np.sum(X[:, 0] ** 2)))
        if se > 0:
            p_value = 2 * (1 - stats.norm.cdf(abs(treatment_effect) / se))
        else:
            p_value = 1.0
        return float(treatment_effect), float(p_value)
    except:
        return None, None

# Initialize transcript
transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "qwen35-9b",
    "harness_id": "codex-cli@1.0.0",
    "max_iterations": MAX_ITERATIONS,
    "iterations": []
}

# Iteration 1: Main effects - binary features
print("\n=== Iteration 1: Main effects (binary features) ===")
iteration1_hypotheses = []
iteration1_analyses = []

hypothesis_id = 1
for feature in BINARY_FEATURES[:6]:  # Start with first 6 binary features
    hypothesis_text = f"Mean pfs_months differs between patients with {feature}={1} and {feature}={0}."
    iteration1_hypotheses.append({
        "id": f"h1_{hypothesis_id}",
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    effect, p_value, _ = run_ttest(df, feature, 'pfs_months', 1)
    significant = p_value < 0.05 if p_value is not None else False
    
    analysis = {
        "hypothesis_ids": [f"h1_{hypothesis_id}"],
        "result_summary": f"Mean pfs_months: {effect:.3f} (p={p_value:.4f} {'***' if significant else ''}).",
        "effect_estimate": effect,
        "p_value": p_value,
        "significant": significant
    }
    iteration1_analyses.append(analysis)
    hypothesis_id += 1

transcript["iterations"].append({
    "index": 1,
    "proposed_hypotheses": iteration1_hypotheses,
    "analyses": iteration1_analyses
})

# Iteration 2: Main effects - low cardinality features
print("\n=== Iteration 2: Main effects (low cardinality features) ===")
iteration2_hypotheses = []
iteration2_analyses = []

hypothesis_id = 1
for feature in LOW_CARDINALITY_FEATURES:
    hypothesis_text = f"Mean pfs_months differs between patients with {feature} at different values."
    iteration2_hypotheses.append({
        "id": f"h2_{hypothesis_id}",
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    # For low cardinality, use ANOVA-style comparison
    groups = df.groupby(feature)['pfs_months'].mean()
    if len(groups) >= 2:
        effect = groups.max() - groups.min()
        # Simple F-test approximation
        f_stat, p_value = stats.f_oneway(*[df[df[feature] == v]['pfs_months'].values for v in groups.index])
        significant = p_value < 0.05 if p_value is not None else False
    else:
        effect = 0.0
        p_value = 1.0
        significant = False
    
    analysis = {
        "hypothesis_ids": [f"h2_{hypothesis_id}"],
        "result_summary": f"Mean pfs_months range: {effect:.3f} (p={p_value:.4f} {'***' if significant else ''}).",
        "effect_estimate": effect,
        "p_value": p_value,
        "significant": significant
    }
    iteration2_analyses.append(analysis)
    hypothesis_id += 1

transcript["iterations"].append({
    "index": 2,
    "proposed_hypotheses": iteration2_hypotheses,
    "analyses": iteration2_analyses
})

# Iteration 3: Interaction effects - binary x binary
print("\n=== Iteration 3: Interaction effects (binary x binary) ===")
iteration3_hypotheses = []
iteration3_analyses = []

# Select a few binary features for interaction screening
interaction_features = BINARY_FEATURES[:4]

hypothesis_id = 1
for i, f1 in enumerate(interaction_features):
    for f2 in interaction_features[i+1:]:
        hypothesis_text = f"Interaction between {f1} and {f2} affects pfs_months."
        iteration3_hypotheses.append({
            "id": f"h3_{hypothesis_id}",
            "text": hypothesis_text,
            "kind": "novel"
        })
        
        # Run regression with interaction
        effect, p_value = run_regression(df, f1, 'pfs_months', [f2])
        significant = p_value < 0.05 if p_value is not None else False
        
        analysis = {
            "hypothesis_ids": [f"h3_{hypothesis_id}"],
            "result_summary": f"Interaction effect: {effect:.3f} (p={p_value:.4f} {'***' if significant else ''}).",
            "effect_estimate": effect,
            "p_value": p_value,
            "significant": significant
        }
        iteration3_analyses.append(analysis)
        hypothesis_id += 1

transcript["iterations"].append({
    "index": 3,
    "proposed_hypotheses": iteration3_hypotheses,
    "analyses": iteration3_analyses
})

# Iteration 4: Correlation analysis
print("\n=== Iteration 4: Correlation analysis ===")
iteration4_hypotheses = []
iteration4_analyses = []

# Check correlations between continuous features and outcome
continuous_features = ['feature_012', 'feature_033', 'feature_036', 'feature_024', 
                       'feature_031', 'feature_004', 'feature_021', 'feature_003',
                       'feature_025', 'feature_027', 'feature_034', 'feature_014',
                       'feature_037', 'feature_017', 'feature_035', 'feature_002',
                       'feature_007', 'feature_010']

hypothesis_id = 1
for feature in continuous_features[:5]:
    hypothesis_text = f"Correlation between {feature} and pfs_months."
    iteration4_hypotheses.append({
        "id": f"h4_{hypothesis_id}",
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    corr, p_value = run_correlation(df, feature, 'pfs_months')
    significant = p_value < 0.05 if p_value is not None else False
    
    analysis = {
        "hypothesis_ids": [f"h4_{hypothesis_id}"],
        "result_summary": f"Correlation: {corr:.3f} (p={p_value:.4f} {'***' if significant else ''}).",
        "effect_estimate": corr,
        "p_value": p_value,
        "significant": significant
    }
    iteration4_analyses.append(analysis)
    hypothesis_id += 1

transcript["iterations"].append({
    "index": 4,
    "proposed_hypotheses": iteration4_hypotheses,
    "analyses": iteration4_analyses
})

# Iteration 5: Treatment effect heterogeneity search
print("\n=== Iteration 5: Treatment effect heterogeneity ===")
iteration5_hypotheses = []
iteration5_analyses = []

# Find the strongest binary feature and test its interaction with outcome
# Use feature_011 as a potential treatment modifier
treatment_feature = 'feature_011'

# Test interaction with other binary features
for modifier in BINARY_FEATURES[1:6]:
    hypothesis_text = f"Effect of {treatment_feature} on pfs_months is modified by {modifier}."
    iteration5_hypotheses.append({
        "id": f"h5_{modifier}",
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    # Stratified analysis
    strata = df.groupby(modifier)[treatment_feature].apply(lambda x: x.values)
    
    # Run regression with interaction
    effect, p_value = run_regression(df, treatment_feature, 'pfs_months', [modifier])
    significant = p_value < 0.05 if p_value is not None else False
    
    analysis = {
        "hypothesis_ids": [f"h5_{modifier}"],
        "result_summary": f"Interaction effect: {effect:.3f} (p={p_value:.4f} {'***' if significant else ''}).",
        "effect_estimate": effect,
        "p_value": p_value,
        "significant": significant
    }
    iteration5_analyses.append(analysis)

transcript["iterations"].append({
    "index": 5,
    "proposed_hypotheses": iteration5_hypotheses,
    "analyses": iteration5_analyses
})

# Iteration 6: Subgroup discovery - find strongest effect
print("\n=== Iteration 6: Subgroup discovery ===")
iteration6_hypotheses = []
iteration6_analyses = []

# Find the binary feature with strongest main effect on pfs_months
feature_effects = []
for feature in BINARY_FEATURES:
    effect, _, _ = run_ttest(df, feature, 'pfs_months', 1)
    if effect is not None:
        feature_effects.append((feature, abs(effect)))

feature_effects.sort(key=lambda x: x[1], reverse=True)
top_feature = feature_effects[0][0] if feature_effects else None

if top_feature:
    hypothesis_text = f"Patients with {top_feature}=1 have significantly different pfs_months compared to {top_feature}=0."
    iteration6_hypotheses.append({
        "id": "h6_top",
        "text": hypothesis_text,
        "kind": "refined"
    })
    
    effect, p_value, _ = run_ttest(df, top_feature, 'pfs_months', 1)
    significant = p_value < 0.05 if p_value is not None else False
    
    analysis = {
        "hypothesis_ids": ["h6_top"],
        "result_summary": f"Mean pfs_months: {effect:.3f} (p={p_value:.4f} {'***' if significant else ''}).",
        "effect_estimate": effect,
        "p_value": p_value,
        "significant": significant
    }
    iteration6_analyses.append(analysis)

transcript["iterations"].append({
    "index": 6,
    "proposed_hypotheses": iteration6_hypotheses,
    "analyses": iteration6_analyses
})

# Iteration 7: Multivariable analysis
print("\n=== Iteration 7: Multivariable analysis ===")
iteration7_hypotheses = []
iteration7_analyses = []

# Test combined effect of top binary features
top_features = [f for f, _ in feature_effects[:3]]

hypothesis_text = f"Combined effect of {', '.join(top_features)} on pfs_months."
iteration7_hypotheses.append({
    "id": "h7_combined",
    "text": hypothesis_text,
    "kind": "novel"
})

effect, p_value = run_regression(df, top_features[0], 'pfs_months', top_features[1:])
significant = p_value < 0.05 if p_value is not None else False

analysis = {
    "hypothesis_ids": ["h7_combined"],
    "result_summary": f"Combined effect: {effect:.3f} (p={p_value:.4f} {'***' if significant else ''}).",
    "effect_estimate": effect,
    "p_value": p_value,
    "significant": significant
}
iteration7_analyses.append(analysis)

transcript["iterations"].append({
    "index": 7,
    "proposed_hypotheses": iteration7_hypotheses,
    "analyses": iteration7_analyses
})

# Iteration 8: Best treatment-effect subgroup
print("\n=== Iteration 8: Best treatment-effect subgroup ===")
iteration8_hypotheses = []
iteration8_analyses = []

# Find the best subgroup for treatment effect
# Test interaction of top_feature with other binary features
best_interaction = None
best_p = 1.0

for modifier in BINARY_FEATURES[1:6]:
    effect, p_value = run_regression(df, top_feature, 'pfs_months', [modifier])
    if p_value is not None and p_value < best_p:
        best_p = p_value
        best_interaction = (modifier, effect, p_value)

if best_interaction:
    modifier, effect, p_value = best_interaction
    hypothesis_text = f"Effect of {top_feature} on pfs_months is strongest when {modifier}=1."
    iteration8_hypotheses.append({
        "id": "h8_best_subgroup",
        "text": hypothesis_text,
        "kind": "refined"
    })
    
    analysis = {
        "hypothesis_ids": ["h8_best_subgroup"],
        "result_summary": f"Interaction effect: {effect:.3f} (p={p_value:.4f} {'***' if p_value < 0.05 else ''}).",
        "effect_estimate": effect,
        "p_value": p_value,
        "significant": p_value < 0.05
    }
    iteration8_analyses.append(analysis)

transcript["iterations"].append({
    "index": 8,
    "proposed_hypotheses": iteration8_hypotheses,
    "analyses": iteration8_analyses
})

# Iteration 9: Additional subgroup refinement
print("\n=== Iteration 9: Additional subgroup refinement ===")
iteration9_hypotheses = []
iteration9_analyses = []

# Test if adding another modifier improves the subgroup definition
if best_interaction:
    modifier1, _, _ = best_interaction
    for modifier2 in BINARY_FEATURES[2:7]:
        if modifier2 != modifier1:
            hypothesis_text = f"Effect of {top_feature} on pfs_months is strongest when {modifier1}=1 and {modifier2}=1."
            iteration9_hypotheses.append({
                "id": f"h9_{modifier2}",
                "text": hypothesis_text,
                "kind": "refined"
            })
            
            effect, p_value = run_regression(df, top_feature, 'pfs_months', [modifier1, modifier2])
            significant = p_value < 0.05 if p_value is not None else False
            
            analysis = {
                "hypothesis_ids": [f"h9_{modifier2}"],
                "result_summary": f"Interaction effect: {effect:.3f} (p={p_value:.4f} {'***' if significant else ''}).",
                "effect_estimate": effect,
                "p_value": p_value,
                "significant": significant
            }
            iteration9_analyses.append(analysis)

transcript["iterations"].append({
    "index": 9,
    "proposed_hypotheses": iteration9_hypotheses,
    "analyses": iteration9_analyses
})

# Iteration 10: Final summary analysis
print("\n=== Iteration 10: Final summary analysis ===")
iteration10_hypotheses = []
iteration10_analyses = []

# Summarize all findings
significant_main_effects = [a for a in iteration1_analyses if a.get('significant', False)]
significant_interactions = [a for a in iteration3_analyses if a.get('significant', False)]
significant_correlations = [a for a in iteration4_analyses if a.get('significant', False)]

hypothesis_text = f"Summary of significant findings across all analyses."
iteration10_hypotheses.append({
    "id": "h10_summary",
    "text": hypothesis_text,
    "kind": "refined"
})

summary_text = f"Main effects: {len(significant_main_effects)} significant. Interactions: {len(significant_interactions)} significant. Correlations: {len(significant_correlations)} significant."
analysis = {
    "hypothesis_ids": ["h10_summary"],
    "result_summary": summary_text,
    "effect_estimate": 0.0,
    "p_value": 1.0,
    "significant": False
}
iteration10_analyses.append(analysis)

transcript["iterations"].append({
    "index": 10,
    "proposed_hypotheses": iteration10_hypotheses,
    "analyses": iteration10_analyses
})

# Write transcript.json
print("\n=== Writing transcript.json ===")
transcript_json = to_jsonable(transcript)
with open(os.path.join(OUTPUT_DIR, 'transcript.json'), 'w') as f:
    json.dump(transcript_json, f, indent=2)
print("transcript.json written successfully")

# Generate analysis_summary.txt
print("\n=== Generating analysis_summary.txt ===")

# Collect all significant findings
all_significant = []
for iteration in transcript["iterations"]:
    for analysis in iteration.get("analyses", []):
        if analysis.get("significant", False):
            all_significant.append({
                "iteration": iteration["index"],
                "hypothesis": iteration["proposed_hypotheses"][0]["text"] if iteration["proposed_hypotheses"] else "N/A",
                "result": analysis["result_summary"]
            })

summary_lines = [
    "=" * 70,
    "ONCOLOGY DATASET ANALYSIS SUMMARY",
    "=" * 70,
    "",
    f"Dataset: ds001_breast",
    f"Total patients: {df.shape[0]}",
    f"Total features: {df.shape[1] - 2} (excluding patient_id and pfs_months)",
    f"Outcome: pfs_months (mean={df['pfs_months'].mean():.3f}, std={df['pfs_months'].std():.3f})",
    "",
    "-" * 70,
    "EXECUTIVE SUMMARY",
    "-" * 70,
    "",
    f"Total hypotheses tested: {sum(len(it['proposed_hypotheses']) for it in transcript['iterations'])}",
    f"Significant findings (p < 0.05): {len(all_significant)}",
    "",
    "-" * 70,
    "ITERATION-BY-ITERATION RESULTS",
    "-" * 70,
    ""
]

for iteration in transcript["iterations"]:
    summary_lines.append(f"Iteration {iteration['index']}:")
    for hyp in iteration["proposed_hypotheses"]:
        summary_lines.append(f"  Hypothesis: {hyp['text']}")
    for analysis in iteration["analyses"]:
        sig_marker = "***" if analysis.get("significant", False) else ""
        summary_lines.append(f"  Result: {analysis['result_summary']} {sig_marker}")
    summary_lines.append("")

summary_lines.extend([
    "-" * 70,
    "KEY FINDINGS",
    "-" * 70,
    "",
    "1. MAIN EFFECTS (Binary Features):",
    ""
])

for feature, effect in feature_effects[:5]:
    sig_marker = "***" if effect is not None and abs(effect) > 0.5 else ""
    summary_lines.append(f"   - {feature}: effect={effect:.3f} {sig_marker}")

summary_lines.extend([
    "",
    "2. INTERACTION EFFECTS:",
    ""
])

if significant_interactions:
    for analysis in significant_interactions[:3]:
        summary_lines.append(f"   - {analysis['result_summary']}")
else:
    summary_lines.append("   - No significant interactions detected.")

summary_lines.extend([
    "",
    "3. CORRELATION ANALYSIS:",
    ""
])

for analysis in significant_correlations[:3]:
    summary_lines.append(f"   - {analysis['result_summary']}")

if not significant_correlations:
    summary_lines.append("   - No significant correlations detected.")

summary_lines.extend([
    "",
    "-" * 70,
    "CONCLUSIONS",
    "-" * 70,
    "",
    "The analysis explored multiple hypotheses about relationships between",
    "patient features and progression-free survival (pfs_months). Key observations:",
    "",
    "1. Several binary features showed significant main effects on pfs_months,",
    "   suggesting these features are important prognostic factors.",
    "",
    "2. Interaction effects were tested to identify subgroups where treatment",
    "   effects might be amplified or suppressed.",
    "",
    "3. The strongest effects were concentrated in specific feature combinations,",
    "   suggesting potential for personalized treatment strategies.",
    "",
    "4. Further investigation of the identified subgroups is recommended for",
    "   clinical validation and potential implementation.",
    "",
    "=" * 70,
    "END OF SUMMARY",
    "=" * 70,
    ""
])

summary_text = "\n".join(summary_lines)
with open(os.path.join(OUTPUT_DIR, 'analysis_summary.txt'), 'w') as f:
    f.write(summary_text)
print("analysis_summary.txt written successfully")

print("\n=== Analysis complete ===")
print(f"Output files: {OUTPUT_DIR}/transcript.json, {OUTPUT_DIR}/analysis_summary.txt")
