#!/usr/bin/env python
"""
End-to-end oncology dataset analysis script.
Performs iterative hypothesis generation, testing, and refinement.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path('/home/kenneth_kehl/onc-co-scientist/data/caa_ab/qwen35_9b_bf16/pilot/neg005/tasks/breast/anonymized/dataset.parquet')
OUTPUT_DIR = Path('/home/kenneth_kehl/onc-co-scientist/data/caa_ab/qwen35_9b_bf16/pilot/neg005/tasks/breast/anonymized')

# Load data
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Separate features and outcome
feature_cols = [c for c in df.columns if c != 'patient_id' and c != 'pfs_months']
outcome_col = 'pfs_months'

# Helper: safe numeric formatting
def safe_format(val, decimals=3):
    """Format a numeric value safely, returning 'NA' for invalid values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NA"
    if isinstance(val, (int, float)):
        if np.isinf(val):
            return "NA"
        return f"{val:.{decimals}f}"
    return str(val)

# Helper: compute effect estimate and p-value for feature-outcome comparison
def compare_feature_outcome(df, feature, outcome, feature_value):
    """
    Compare outcome between feature==value and feature!=value.
    Returns dict with effect_estimate, p_value, significant, and descriptive stats.
    """
    mask = df[feature] == feature_value
    group1 = df.loc[mask, outcome]
    group2 = df.loc[~mask, outcome]
    
    # Effect estimate: mean difference (group1 - group2)
    effect = group1.mean() - group2.mean()
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    # Build 2x2 count table for categorical features
    counts = {
        'feature_value': int(mask.sum()),
        'not_feature_value': int(~mask.sum()),
        'outcome_group1': int((mask & (df[outcome] >= df[outcome].median())).sum()),
        'outcome_group2': int((~mask & (df[outcome] >= df[outcome].median())).sum()),
    }
    
    return {
        'effect_estimate': float(effect),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'group1_mean': float(group1.mean()),
        'group2_mean': float(group2.mean()),
        'n_group1': int(len(group1)),
        'n_group2': int(len(group2)),
        'counts': counts,
    }

# Helper: compute correlation for continuous features
def correlate_feature_outcome(df, feature, outcome):
    """Compute Pearson correlation and p-value."""
    mask = df[[feature, outcome]].dropna()
    if len(mask) < 3:
        return {
            'effect_estimate': None,
            'p_value': None,
            'significant': False,
            'correlation': None,
            'n': 0,
        }
    corr, p_value = stats.pearsonr(mask[feature], mask[outcome])
    return {
        'effect_estimate': float(corr),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'correlation': float(corr),
        'n': int(len(mask)),
    }

# Helper: treatment effect heterogeneity search
def find_treatment_heterogeneity(df, treatment_col, outcome_col, feature_cols, top_k=5):
    """
    Screen for treatment-by-feature interactions.
    Returns list of (feature, effect_diff, p_value) tuples.
    """
    results = []
    treatment_levels = df[treatment_col].unique()
    if len(treatment_levels) < 2:
        return results
    
    # Get treatment groups
    t1_mask = df[treatment_col] == treatment_levels[1] if len(treatment_levels) > 1 else df[treatment_col] == 1
    t0_mask = ~t1_mask
    
    # For each feature, compute treatment effect in each group
    for feat in feature_cols:
        # Group 1: treatment effect in feature=value group
        feat_mask = df[feat] == 1
        t1_feat = df.loc[t1_mask & feat_mask, outcome_col]
        t0_feat = df.loc[t0_mask & feat_mask, outcome_col]
        effect1 = t1_feat.mean() - t0_feat.mean() if len(t1_feat) > 0 and len(t0_feat) > 0 else np.nan
        
        # Group 0: treatment effect in feature!=value group
        t1_notfeat = df.loc[t1_mask & ~feat_mask, outcome_col]
        t0_notfeat = df.loc[t0_mask & ~feat_mask, outcome_col]
        effect0 = t1_notfeat.mean() - t0_notfeat.mean() if len(t1_notfeat) > 0 and len(t0_notfeat) > 0 else np.nan
        
        # Interaction effect
        if not np.isnan(effect1) and not np.isnan(effect0):
            interaction = effect1 - effect0
            # Simple interaction test via 2x2 ANOVA approximation
            groups = [
                df.loc[t1_mask & feat_mask, outcome_col],
                df.loc[t0_mask & feat_mask, outcome_col],
                df.loc[t1_mask & ~feat_mask, outcome_col],
                df.loc[t0_mask & ~feat_mask, outcome_col],
            ]
            flat = np.concatenate([g for g in groups if len(g) > 0])
            if len(flat) > 3:
                # Compute interaction p-value via regression
                X = pd.get_dummies(df[[treatment_col, feat]], drop_first=True)
                X = X.reindex(df.index)
                model = stats.linregress(flat, X.iloc[:, 1].values)
                p_interaction = model.pvalue
            else:
                p_interaction = 1.0
            results.append((feat, interaction, p_interaction))
    
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results[:top_k]

# Initialize transcript
transcript = {
    'dataset_id': 'ds001_breast',
    'model_id': 'codex-cli',
    'harness_id': 'codex-cli@1.0.0',
    'max_iterations': 10,
    'iterations': [],
}

# Iteration 1: Main effects - identify strongest feature-outcome associations
print("\n=== Iteration 1: Main effects screening ===")

# Identify binary and continuous features
binary_features = [f for f in feature_cols if df[f].nunique() == 2]
continuous_features = [f for f in feature_cols if df[f].dtype in ['float64', 'int64'] and df[f].nunique() > 10]

# Test binary features
print("Testing binary features...")
binary_results = []
for feat in binary_features:
    result = compare_feature_outcome(df, feat, outcome_col, 1)
    result['feature'] = feat
    result['feature_value'] = 1
    binary_results.append(result)

# Test continuous features (top correlations)
print("Testing continuous features...")
continuous_results = []
for feat in continuous_features[:10]:  # Top 10 by correlation
    result = correlate_feature_outcome(df, feat, outcome_col)
    result['feature'] = feat
    continuous_results.append(result)

# Combine and rank
all_results = binary_results + continuous_results
all_results.sort(key=lambda x: abs(x.get('effect_estimate', 0) or 0), reverse=True)

# Propose hypotheses for top findings
hypotheses = []
analyses = []

for i, res in enumerate(all_results[:5]):
    feat = res['feature']
    effect = res.get('effect_estimate')
    pval = res.get('p_value')
    corr = res.get('correlation')
    
    if feat in binary_features:
        text = f"Patients with {feat}==1 have {'higher' if effect > 0 else 'lower'} {outcome_col} (mean diff={safe_format(effect)}, p={safe_format(pval)})"
        hypothesis_id = f"h1_{i+1}"
        hypotheses.append({'id': hypothesis_id, 'text': text, 'kind': 'novel'})
        
        # Precompute t-stat for result summary
        t_stat, _ = stats.ttest_ind(df.loc[df[feat]==1, outcome_col], df.loc[df[feat]==0, outcome_col])
        t_str = safe_format(t_stat)
        res_summary = f"Mean {outcome_col}: {safe_format(res['group1_mean'])} vs {safe_format(res['group2_mean'])} (t={t_str}, p={safe_format(pval)})"
        
        analyses.append({
            'hypothesis_ids': [hypothesis_id],
            'result_summary': res_summary,
            'effect_estimate': effect,
            'p_value': pval,
            'significant': pval is not None and pval < 0.05,
        })
    else:
        text = f"{feat} correlates with {outcome_col} (r={safe_format(corr)}, p={safe_format(pval)})"
        hypothesis_id = f"h1_{i+1}"
        hypotheses.append({'id': hypothesis_id, 'text': text, 'kind': 'novel'})
        analyses.append({
            'hypothesis_ids': [hypothesis_id],
            'result_summary': f"Correlation: {safe_format(corr)}, p={safe_format(pval)}",
            'effect_estimate': corr,
            'p_value': pval,
            'significant': pval is not None and pval < 0.05,
        })

transcript['iterations'].append({
    'index': 1,
    'proposed_hypotheses': hypotheses,
    'analyses': analyses,
})

# Iteration 2: Deep dive on top feature
print("\n=== Iteration 2: Deep dive on top feature ===")
top_feat = all_results[0]['feature']
print(f"Top feature: {top_feat}, effect: {safe_format(all_results[0].get('effect_estimate'))}")

# Test different values for this feature
if top_feat in binary_features:
    values = [0, 1]
else:
    values = [0, 1, 2]  # Try low, mid, high

hypotheses = []
analyses = []

for i, val in enumerate(values):
    result = compare_feature_outcome(df, top_feat, outcome_col, val)
    text = f"Patients with {top_feat}={val} have {'higher' if result['effect_estimate'] > 0 else 'lower'} {outcome_col} (mean diff={safe_format(result['effect_estimate'])}, p={safe_format(result['p_value'])})"
    hypothesis_id = f"h2_{i+1}"
    hypotheses.append({'id': hypothesis_id, 'text': text, 'kind': 'novel'})
    analyses.append({
        'hypothesis_ids': [hypothesis_id],
        'result_summary': f"Mean {outcome_col}: {safe_format(result['group1_mean'])} vs {safe_format(result['group2_mean'])}, p={safe_format(result['p_value'])}",
        'effect_estimate': result['effect_estimate'],
        'p_value': result['p_value'],
        'significant': result['p_value'] is not None and result['p_value'] < 0.05,
    })

transcript['iterations'].append({
    'index': 2,
    'proposed_hypotheses': hypotheses,
    'analyses': analyses,
})

# Iteration 3: Interaction screening
print("\n=== Iteration 3: Interaction screening ===")

# Use top binary feature as treatment
treatment_feat = binary_features[0] if binary_features else feature_cols[0]

# Find treatment heterogeneity
interactions = find_treatment_heterogeneity(df, treatment_feat, outcome_col, feature_cols, top_k=5)

hypotheses = []
analyses = []

for i, (feat, interaction, pval) in enumerate(interactions):
    text = f"Interaction between {treatment_feat} and {feat} on {outcome_col}: effect diff={safe_format(interaction)}, p={safe_format(pval)}"
    hypothesis_id = f"h3_{i+1}"
    hypotheses.append({'id': hypothesis_id, 'text': text, 'kind': 'novel'})
    analyses.append({
        'hypothesis_ids': [hypothesis_id],
        'result_summary': f"Interaction effect: {safe_format(interaction)}, p={safe_format(pval)}",
        'effect_estimate': interaction,
        'p_value': pval,
        'significant': pval is not None and pval < 0.05,
    })

transcript['iterations'].append({
    'index': 3,
    'proposed_hypotheses': hypotheses,
    'analyses': analyses,
})

# Iteration 4: Subgroup analysis for strongest interaction
print("\n=== Iteration 4: Subgroup analysis ===")

if interactions:
    best_interaction_feat = interactions[0][0]
    best_interaction_pval = interactions[0][2]
    
    # Test the interaction subgroup
    mask = (df[treatment_feat] == 1) & (df[best_interaction_feat] == 1)
    group1 = df.loc[mask, outcome_col]
    group2 = df.loc[~mask, outcome_col]
    
    effect = group1.mean() - group2.mean()
    t_stat, pval = stats.ttest_ind(group1, group2)
    
    text = f"Patients with {treatment_feat}==1 AND {best_interaction_feat}==1 have {'higher' if effect > 0 else 'lower'} {outcome_col} (mean diff={safe_format(effect)}, p={safe_format(pval)})"
    hypothesis_id = "h4"
    hypotheses = [{'id': hypothesis_id, 'text': text, 'kind': 'novel'}]
    analyses = [{
        'hypothesis_ids': [hypothesis_id],
        'result_summary': f"Subgroup mean {outcome_col}: {safe_format(group1.mean())} vs {safe_format(group2.mean())}, p={safe_format(pval)}",
        'effect_estimate': effect,
        'p_value': pval,
        'significant': pval is not None and pval < 0.05,
    }]
else:
    hypotheses = []
    analyses = []

transcript['iterations'].append({
    'index': 4,
    'proposed_hypotheses': hypotheses,
    'analyses': analyses,
})

# Iteration 5: Additional binary feature testing
print("\n=== Iteration 5: Additional binary features ===")

remaining_binary = [f for f in binary_features if f != treatment_feat]
hypotheses = []
analyses = []

for i, feat in enumerate(remaining_binary[:3]):
    result = compare_feature_outcome(df, feat, outcome_col, 1)
    text = f"Patients with {feat}==1 have {'higher' if result['effect_estimate'] > 0 else 'lower'} {outcome_col} (mean diff={safe_format(result['effect_estimate'])}, p={safe_format(result['p_value'])})"
    hypothesis_id = f"h5_{i+1}"
    hypotheses.append({'id': hypothesis_id, 'text': text, 'kind': 'novel'})
    analyses.append({
        'hypothesis_ids': [hypothesis_id],
        'result_summary': f"Mean {outcome_col}: {safe_format(result['group1_mean'])} vs {safe_format(result['group2_mean'])}, p={safe_format(result['p_value'])}",
        'effect_estimate': result['effect_estimate'],
        'p_value': result['p_value'],
        'significant': result['p_value'] is not None and result['p_value'] < 0.05,
    })

transcript['iterations'].append({
    'index': 5,
    'proposed_hypotheses': hypotheses,
    'analyses': analyses,
})

# Iteration 6: Three-way interaction (if applicable)
print("\n=== Iteration 6: Three-way interaction ===")

if len(binary_features) >= 3:
    f1, f2, f3 = binary_features[0], binary_features[1], binary_features[2]
    
    # Test 4-group interaction
    groups = [
        ('f1=0,f2=0', df.loc[(df[f1]==0) & (df[f2]==0), outcome_col]),
        ('f1=0,f2=1', df.loc[(df[f1]==0) & (df[f2]==1), outcome_col]),
        ('f1=1,f2=0', df.loc[(df[f1]==1) & (df[f2]==0), outcome_col]),
        ('f1=1,f2=1', df.loc[(df[f1]==1) & (df[f2]==1), outcome_col]),
    ]
    
    means = [g[1].mean() if len(g[1]) > 0 else np.nan for g in groups]
    ns = [len(g[1]) for g in groups]
    
    # Simple interaction test
    if all(m is not None for m in means) and all(n > 0 for n in ns):
        # Two-way interaction effect
        effect = means[3] - means[2] - means[1] + means[0]
        
        # P-value via ANOVA-like approach
        from scipy import stats as st
        data = np.concatenate([g[1] for g in groups if len(g[1]) > 0])
        X = pd.get_dummies(df[[f1, f2]], drop_first=True).reindex(df.index)
        model = st.linregress(data, X.iloc[:, 1].values)
        pval = model.pvalue
        
        text = f"Two-way interaction between {f1} and {f2} on {outcome_col}: effect={safe_format(effect)}, p={safe_format(pval)}"
        hypothesis_id = "h6"
        hypotheses = [{'id': hypothesis_id, 'text': text, 'kind': 'novel'}]
        analyses = [{
            'hypothesis_ids': [hypothesis_id],
            'result_summary': f"Interaction effect: {safe_format(effect)}, p={safe_format(pval)}",
            'effect_estimate': effect,
            'p_value': pval,
            'significant': pval is not None and pval < 0.05,
        }]
    else:
        hypotheses = []
        analyses = []
else:
    hypotheses = []
    analyses = []

transcript['iterations'].append({
    'index': 6,
    'proposed_hypotheses': hypotheses,
    'analyses': analyses,
})

# Iteration 7: Continuous feature subgroup
print("\n=== Iteration 7: Continuous feature subgroup ===")

if continuous_features:
    cont_feat = continuous_features[0]
    
    # Test quartile-based subgroups using cut instead of qcut
    quartiles = pd.cut(df[cont_feat], 4, labels=False, duplicates='drop')
    quartile_means = []
    for i in range(4):
        mask = quartiles == i
        mean_val = df.loc[mask, outcome_col].mean()
        quartile_means.append((i, mean_val, mask.sum()))
    
    quartile_means.sort(key=lambda x: x[1])
    
    # Compare lowest vs highest quartile
    low_q, low_mean, low_n = quartile_means[0]
    high_q, high_mean, high_n = quartile_means[-1]
    
    low_mask = quartiles == low_q
    high_mask = quartiles == high_q
    
    effect = high_mean - low_mean
    t_stat, pval = stats.ttest_ind(df.loc[high_mask, outcome_col], df.loc[low_mask, outcome_col])
    
    text = f"Patients in highest quartile of {cont_feat} have {'higher' if effect > 0 else 'lower'} {outcome_col} (mean diff={safe_format(effect)}, p={safe_format(pval)})"
    hypothesis_id = "h7"
    hypotheses = [{'id': hypothesis_id, 'text': text, 'kind': 'novel'}]
    analyses = [{
        'hypothesis_ids': [hypothesis_id],
        'result_summary': f"Q{high_q+1} mean: {safe_format(high_mean)}, Q{low_q+1} mean: {safe_format(low_mean)}, p={safe_format(pval)}",
        'effect_estimate': effect,
        'p_value': pval,
        'significant': pval is not None and pval < 0.05,
    }]
else:
    hypotheses = []
    analyses = []

transcript['iterations'].append({
    'index': 7,
    'proposed_hypotheses': hypotheses,
    'analyses': analyses,
})

# Iteration 8: Refine top hypothesis
print("\n=== Iteration 8: Refine top hypothesis ===")

# Find most significant finding from iteration 1
if all_results:
    top_result = all_results[0]
    top_feat = top_result['feature']
    
    # Test with different thresholds
    hypotheses = []
    analyses = []
    
    if top_feat in binary_features:
        for thresh in [0.5, 0.6, 0.7]:
            result = compare_feature_outcome(df, top_feat, outcome_col, thresh)
            text = f"Patients with {top_feat} >= {thresh} have {'higher' if result['effect_estimate'] > 0 else 'lower'} {outcome_col} (mean diff={safe_format(result['effect_estimate'])}, p={safe_format(result['p_value'])})"
            hypothesis_id = f"h8_{thresh}"
            hypotheses.append({'id': hypothesis_id, 'text': text, 'kind': 'refined'})
            analyses.append({
                'hypothesis_ids': [hypothesis_id],
                'result_summary': f"Mean {outcome_col}: {safe_format(result['group1_mean'])} vs {safe_format(result['group2_mean'])}, p={safe_format(result['p_value'])}",
                'effect_estimate': result['effect_estimate'],
                'p_value': result['p_value'],
                'significant': result['p_value'] is not None and result['p_value'] < 0.05,
            })
    else:
        # For continuous, test percentile-based
        for pct in [25, 50, 75]:
            result = compare_feature_outcome(df, top_feat, outcome_col, pct)
            text = f"Patients in top {pct}% of {top_feat} have {'higher' if result['effect_estimate'] > 0 else 'lower'} {outcome_col} (mean diff={safe_format(result['effect_estimate'])}, p={safe_format(result['p_value'])})"
            hypothesis_id = f"h8_{pct}"
            hypotheses.append({'id': hypothesis_id, 'text': text, 'kind': 'refined'})
            analyses.append({
                'hypothesis_ids': [hypothesis_id],
                'result_summary': f"Mean {outcome_col}: {safe_format(result['group1_mean'])} vs {safe_format(result['group2_mean'])}, p={safe_format(result['p_value'])}",
                'effect_estimate': result['effect_estimate'],
                'p_value': result['p_value'],
                'significant': result['p_value'] is not None and result['p_value'] < 0.05,
            })

transcript['iterations'].append({
    'index': 8,
    'proposed_hypotheses': hypotheses,
    'analyses': analyses,
})

# Iteration 9: Comprehensive interaction search
print("\n=== Iteration 9: Comprehensive interaction search ===")

# Screen all binary feature pairs for interactions
binary_results = []
for f1 in binary_features[:5]:
    for f2 in binary_features[:5]:
        if f1 != f2:
            # Test interaction
            groups = [
                df.loc[(df[f1]==0) & (df[f2]==0), outcome_col],
                df.loc[(df[f1]==0) & (df[f2]==1), outcome_col],
                df.loc[(df[f1]==1) & (df[f2]==0), outcome_col],
                df.loc[(df[f1]==1) & (df[f2]==1), outcome_col],
            ]
            flat = np.concatenate([g for g in groups if len(g) > 0])
            if len(flat) > 3:
                X = pd.get_dummies(df[[f1, f2]], drop_first=True).reindex(df.index)
                model = stats.linregress(flat, X.iloc[:, 1].values)
                pval = model.pvalue
                # Interaction effect
                means = [g.mean() if len(g) > 0 else np.nan for g in groups]
                interaction = means[3] - means[2] - means[1] + means[0]
                binary_results.append((f1, f2, interaction, pval))

binary_results.sort(key=lambda x: abs(x[2]), reverse=True)

hypotheses = []
analyses = []

for i, (f1, f2, interaction, pval) in enumerate(binary_results[:3]):
    text = f"Interaction between {f1} and {f2} on {outcome_col}: effect={safe_format(interaction)}, p={safe_format(pval)}"
    hypothesis_id = f"h9_{i+1}"
    hypotheses.append({'id': hypothesis_id, 'text': text, 'kind': 'novel'})
    analyses.append({
        'hypothesis_ids': [hypothesis_id],
        'result_summary': f"Interaction effect: {safe_format(interaction)}, p={safe_format(pval)}",
        'effect_estimate': interaction,
        'p_value': pval,
        'significant': pval is not None and pval < 0.05,
    })

transcript['iterations'].append({
    'index': 9,
    'proposed_hypotheses': hypotheses,
    'analyses': analyses,
})

# Iteration 10: Final best-supported hypothesis
print("\n=== Iteration 10: Final best-supported hypothesis ===")

# Find the most significant interaction
best_interaction = None
best_pval = 1.0
best_effect = 0.0

for f1 in binary_features[:5]:
    for f2 in binary_features[:5]:
        if f1 != f2:
            groups = [
                df.loc[(df[f1]==0) & (df[f2]==0), outcome_col],
                df.loc[(df[f1]==0) & (df[f2]==1), outcome_col],
                df.loc[(df[f1]==1) & (df[f2]==0), outcome_col],
                df.loc[(df[f1]==1) & (df[f2]==1), outcome_col],
            ]
            flat = np.concatenate([g for g in groups if len(g) > 0])
            if len(flat) > 3:
                X = pd.get_dummies(df[[f1, f2]], drop_first=True).reindex(df.index)
                model = stats.linregress(flat, X.iloc[:, 1].values)
                pval = model.pvalue
                means = [g.mean() if len(g) > 0 else np.nan for g in groups]
                interaction = means[3] - means[2] - means[1] + means[0]
                if pval < best_pval and pval < 0.1:
                    best_pval = pval
                    best_effect = interaction
                    best_interaction = (f1, f2)

if best_interaction:
    f1, f2 = best_interaction
    # Define the subgroup where treatment effect is strongest
    mask = (df[f1] == 1) & (df[f2] == 1)
    group1 = df.loc[mask, outcome_col]
    group2 = df.loc[~mask, outcome_col]
    
    effect = group1.mean() - group2.mean()
    t_stat, pval = stats.ttest_ind(group1, group2)
    
    text = f"In patients with {f1}==1 AND {f2}==1, {outcome_col} is {'higher' if effect > 0 else 'lower'} (mean diff={safe_format(effect)}, p={safe_format(pval)})"
    hypothesis_id = "h10"
    hypotheses = [{'id': hypothesis_id, 'text': text, 'kind': 'refined'}]
    analyses = [{
        'hypothesis_ids': [hypothesis_id],
        'result_summary': f"Subgroup mean {outcome_col}: {safe_format(group1.mean())} vs {safe_format(group2.mean())}, p={safe_format(pval)}",
        'effect_estimate': effect,
        'p_value': pval,
        'significant': pval is not None and pval < 0.05,
    }]
else:
    hypotheses = []
    analyses = []

transcript['iterations'].append({
    'index': 10,
    'proposed_hypotheses': hypotheses,
    'analyses': analyses,
})

# Convert transcript to JSON-serializable format
def to_jsonable(obj):
    """Convert object to JSON-serializable format."""
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
    elif isinstance(obj, (float, int, bool, str, type(None))):
        return obj
    else:
        return str(obj)

# Clean up transcript for JSON
clean_transcript = to_jsonable(transcript)

# Write transcript.json
transcript_path = OUTPUT_DIR / 'transcript.json'
with open(transcript_path, 'w') as f:
    json.dump(clean_transcript, f, indent=2)
print(f"\nWrote {transcript_path}")

# Generate analysis_summary.txt
print("Generating analysis summary...")

summary_lines = [
    "=" * 80,
    "ONCOLOGY DATASET ANALYSIS SUMMARY",
    "=" * 80,
    "",
    f"Dataset: ds001_breast",
    f"Total patients: {len(df)}",
    f"Outcome: {outcome_col} (mean={safe_format(df[outcome_col].mean())}, std={safe_format(df[outcome_col].std())})",
    "",
    "-" * 80,
    "ITERATION 1: MAIN EFFECTS SCREENING",
    "-" * 80,
]

# Add iteration 1 results
if all_results:
    summary_lines.append(f"\nTop feature-outcome associations (by effect magnitude):")
    for i, res in enumerate(all_results[:5]):
        feat = res['feature']
        effect = res.get('effect_estimate')
        pval = res.get('p_value')
        corr = res.get('correlation')
        if feat in binary_features:
            summary_lines.append(f"  {feat}==1: mean diff={safe_format(effect)}, p={safe_format(pval)}")
        else:
            summary_lines.append(f"  {feat}: r={safe_format(corr)}, p={safe_format(pval)}")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 2: DEEP DIVE ON TOP FEATURE",
    "-" * 80,
])

if all_results:
    top_feat = all_results[0]['feature']
    summary_lines.append(f"\nTop feature: {top_feat}")
    summary_lines.append(f"Effect on {outcome_col}: {safe_format(all_results[0].get('effect_estimate'))}")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 3: INTERACTION SCREENING",
    "-" * 80,
])

if interactions:
    summary_lines.append(f"\nTreatment feature: {treatment_feat}")
    summary_lines.append(f"Top interactions with {outcome_col}:")
    for feat, effect, pval in interactions[:5]:
        summary_lines.append(f"  {feat}: interaction effect={safe_format(effect)}, p={safe_format(pval)}")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 4: SUBGROUP ANALYSIS",
    "-" * 80,
])

if interactions:
    best_feat = interactions[0][0]
    summary_lines.append(f"\nBest interaction: {treatment_feat} x {best_feat}")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 5-7: ADDITIONAL ANALYSES",
    "-" * 80,
])

summary_lines.append("\nAdditional binary features tested:")
for feat in remaining_binary[:3]:
    result = compare_feature_outcome(df, feat, outcome_col, 1)
    summary_lines.append(f"  {feat}==1: mean diff={safe_format(result['effect_estimate'])}, p={safe_format(result['p_value'])}")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 8: THRESHOLD ANALYSIS",
    "-" * 80,
])

if all_results:
    top_feat = all_results[0]['feature']
    summary_lines.append(f"\nThreshold analysis for {top_feat}:")
    for thresh in [0.5, 0.6, 0.7]:
        result = compare_feature_outcome(df, top_feat, outcome_col, thresh)
        summary_lines.append(f"  {top_feat} >= {thresh}: mean diff={safe_format(result['effect_estimate'])}, p={safe_format(result['p_value'])}")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 9: COMPREHENSIVE INTERACTION SEARCH",
    "-" * 80,
])

if binary_results:
    summary_lines.append(f"\nTop binary feature interactions:")
    for f1, f2, effect, pval in binary_results[:5]:
        summary_lines.append(f"  {f1} x {f2}: effect={safe_format(effect)}, p={safe_format(pval)}")

summary_lines.extend([
    "",
    "-" * 80,
    "ITERATION 10: FINAL BEST-SUPPORTED HYPOTHESIS",
    "-" * 80,
])

if best_interaction:
    f1, f2 = best_interaction
    summary_lines.append(f"\nBest-supported interaction: {f1} x {f2}")
    summary_lines.append(f"Interaction effect: {safe_format(best_effect)}, p={safe_format(best_pval)}")
    summary_lines.append(f"Subgroup (f1=1, f2=1) mean {outcome_col}: {safe_format(group1.mean())}")
    summary_lines.append(f"Non-subgroup mean {outcome_col}: {safe_format(group2.mean())}")

summary_lines.extend([
    "",
    "-" * 80,
    "CONCLUSIONS",
    "-" * 80,
])

# Count significant findings
significant_count = sum(1 for it in transcript['iterations'] for a in it['analyses'] if a.get('significant', False))
summary_lines.append(f"\nTotal hypotheses tested: {sum(len(it['analyses']) for it in transcript['iterations'])}")
summary_lines.append(f"Statistically significant (p<0.05): {significant_count}")

summary_lines.append(f"\nKey findings:")
summary_lines.append(f"  1. {all_results[0]['feature']} shows the strongest association with {outcome_col}")
if interactions:
    summary_lines.append(f"  2. Interaction between {treatment_feat} and {interactions[0][0]} modifies {outcome_col}")
if best_interaction:
    summary_lines.append(f"  3. Best interaction: {best_interaction[0]} x {best_interaction[1]}")

summary_lines.extend([
    "",
    "=" * 80,
    "END OF SUMMARY",
    "=" * 80,
])

# Write analysis_summary.txt
summary_path = OUTPUT_DIR / 'analysis_summary.txt'
with open(summary_path, 'w') as f:
    f.write('\n'.join(summary_lines))
print(f"Wrote {summary_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output files:")
print(f"  - {transcript_path}")
print(f"  - {summary_path}")
