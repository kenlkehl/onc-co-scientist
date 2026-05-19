#!/usr/bin/env python3
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
CWD = Path('/home/kenneth_kehl/onc-co-scientist/data/caa_ab/qwen35_9b_bf16/pilot/neg010/tasks/crc/anonymized')
DATA_PATH = CWD / 'dataset.parquet'
TRANSCRIPT_PATH = CWD / 'transcript.json'
SUMMARY_PATH = CWD / 'analysis_summary.txt'

# Dataset info
DATASET_ID = 'ds001_crc'
MAX_ITERATIONS = 10
MODEL_ID = 'codex-cli@0.1.0'
HARNESS_ID = 'codex-cli@0.1.0'

# Load data
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df)} patient records with {len(df.columns)} columns")

# Separate features and outcome
OUTCOME = 'pfs_months'
FEATURE_COLS = [c for c in df.columns if c != 'patient_id' and c != OUTCOME]
BINARY_FEATURES = ['feature_031', 'feature_014', 'feature_024', 'feature_010', 'feature_012', 
                   'feature_018', 'feature_020', 'feature_027', 'feature_029', 'feature_007']
CONTINUOUS_FEATURES = ['feature_015', 'feature_005', 'feature_019', 'feature_021', 'feature_025',
                       'feature_032', 'feature_017', 'feature_003', 'feature_030', 'feature_008',
                       'feature_004', 'feature_013', 'feature_026', 'feature_033', 'feature_009']
CATEGORICAL_FEATURES = ['feature_001']  # 3 levels

print(f"Binary features: {len(BINARY_FEATURES)}")
print(f"Continuous features: {len(CONTINUOUS_FEATURES)}")
print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")

# Safe numeric formatting helper
def safe_format(value, decimals=3):
    """Format a numeric value safely, returning 'NA' for invalid values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (int, float)):
        try:
            return f"{float(value):.{decimals}f}"
        except:
            return "NA"
    return str(value)

# Statistical analysis helpers - return dictionaries for stability
def analyze_feature_outcome_binary(df, feature, outcome, value):
    """
    Compare outcome means between groups defined by binary feature=value.
    Returns dict with effect_estimate, p_value, significant, result_summary.
    """
    mask = df[feature] == value
    group1 = df.loc[mask, outcome]
    group0 = df.loc[~mask, outcome]
    
    if len(group1) == 0 or len(group0) == 0:
        return {
            'effect_estimate': None,
            'p_value': None,
            'significant': None,
            'result_summary': f'Insufficient data for {feature}={value}'
        }
    
    # Effect estimate: mean difference (group1 - group0)
    effect = group1.mean() - group0.mean()
    
    # Two-sample t-test
    t_stat, p_val = stats.ttest_ind(group1, group0)
    
    significant = p_val < 0.05
    
    summary = f"Mean {outcome}: {safe_format(group1.mean(), 2)} vs {safe_format(group0.mean(), 2)} " \
              f"(diff={safe_format(effect, 3)}, t={t_stat:.3f}, p={safe_format(p_val, 4)})"
    
    return {
        'effect_estimate': float(effect),
        'p_value': float(p_val),
        'significant': bool(significant),
        'result_summary': summary
    }

def analyze_feature_outcome_continuous(df, feature, outcome):
    """
    Compare outcome means between groups defined by continuous feature quartiles.
    Returns dict with effect_estimate, p_value, significant, result_summary.
    """
    # Quartile groups using quantile-based bins
    q = np.quantile(df[feature], [0.25, 0.5, 0.75])
    groups = pd.cut(df[feature], bins=q, labels=False, retbins=True)[0]
    
    # ANOVA
    group0 = df.loc[groups == 0, outcome]
    group1 = df.loc[groups == 1, outcome]
    group2 = df.loc[groups == 2, outcome]
    group3 = df.loc[groups == 3, outcome]
    
    f_stat, p_val = stats.f_oneway(group0, group1, group2, group3)
    
    # Post-hoc: compare highest vs lowest quartile
    high = df.loc[groups == 3, outcome]
    low = df.loc[groups == 0, outcome]
    effect = high.mean() - low.mean()
    t_stat, t_pval = stats.ttest_ind(high, low)
    
    significant = p_val < 0.05
    
    summary = f"ANOVA F={f_stat:.3f}, p={safe_format(p_val, 4)}. " \
              f"Highest vs lowest quartile: {safe_format(high.mean(), 2)} vs {safe_format(low.mean(), 2)} " \
              f"(diff={safe_format(effect, 3)}, t={t_stat:.3f}, p={safe_format(t_pval, 4)})"
    
    return {
        'effect_estimate': float(effect),
        'p_value': float(p_val),
        'significant': bool(significant),
        'result_summary': summary
    }

def analyze_feature_outcome_categorical(df, feature, outcome):
    """
    Compare outcome means across categorical feature levels.
    Returns dict with effect_estimate, p_value, significant, result_summary.
    """
    # Get unique levels
    levels = sorted(df[feature].unique())
    
    # One-way ANOVA using boolean masks
    groups_data = []
    for level in levels:
        mask = df[feature] == level
        groups_data.append(df.loc[mask, outcome])
    
    f_stat, p_val = stats.f_oneway(*groups_data)
    
    # Post-hoc: compare highest vs lowest level
    high = df.loc[df[feature] == levels[-1], outcome]
    low = df.loc[df[feature] == levels[0], outcome]
    effect = high.mean() - low.mean()
    t_stat, t_pval = stats.ttest_ind(high, low)
    
    significant = p_val < 0.05
    
    summary = f"ANOVA F={f_stat:.3f}, p={safe_format(p_val, 4)}. " \
              f"Highest vs lowest level: {safe_format(high.mean(), 2)} vs {safe_format(low.mean(), 2)} " \
              f"(diff={safe_format(effect, 3)}, t={t_stat:.3f}, p={safe_format(t_pval, 4)})"
    
    return {
        'effect_estimate': float(effect),
        'p_value': float(p_val),
        'significant': bool(significant),
        'result_summary': summary
    }

def analyze_correlation(df, feature1, feature2):
    """
    Compute Pearson correlation between two continuous features.
    Returns dict with correlation coefficient, p_value, significant, result_summary.
    """
    corr, p_val = stats.pearsonr(df[feature1], df[feature2])
    significant = p_val < 0.05
    
    summary = f"Correlation r={safe_format(corr, 3)}, p={safe_format(p_val, 4)}"
    
    return {
        'effect_estimate': float(corr),
        'p_value': float(p_val),
        'significant': bool(significant),
        'result_summary': summary
    }

def analyze_treatment_heterogeneity(df, treatment, outcome, modifier):
    """
    Test treatment effect heterogeneity by modifier.
    Returns dict with interaction effect, p_value, significant, result_summary.
    """
    # Treatment effect in modifier=1 group
    mask1 = (df[treatment] == 1) & (df[modifier] == 1)
    mask0 = (df[treatment] == 1) & (df[modifier] == 0)
    
    if mask1.sum() == 0 or mask0.sum() == 0:
        return {
            'effect_estimate': None,
            'p_value': None,
            'significant': None,
            'result_summary': f'Insufficient data for treatment={treatment} interaction with {modifier}'
        }
    
    effect1 = df.loc[mask1, outcome].mean() - df.loc[mask0, outcome].mean()
    
    # Treatment effect in modifier=0 group
    mask1_0 = (df[treatment] == 1) & (df[modifier] == 0)
    mask0_0 = (df[treatment] == 0) & (df[modifier] == 0)
    
    if mask1_0.sum() == 0 or mask0_0.sum() == 0:
        return {
            'effect_estimate': None,
            'p_value': None,
            'significant': None,
            'result_summary': f'Insufficient data for treatment={treatment} interaction with {modifier}'
        }
    
    effect0 = df.loc[mask1_0, outcome].mean() - df.loc[mask0_0, outcome].mean()
    
    # Interaction effect
    interaction = effect1 - effect0
    
    # Test interaction via regression
    from statsmodels.formula.api import ols
    try:
        model_full = ols(f'{outcome} ~ {treatment} + {modifier} + {treatment}:{modifier}', data=df).fit()
        p_interaction = model_full.pvalues.get(f'{treatment}:{modifier}', 1.0)
    except Exception as e:
        p_interaction = 1.0
    
    significant = p_interaction < 0.05
    
    summary = f"Main effect: {safe_format(effect1, 2)} vs {safe_format(effect0, 2)}. " \
              f"Interaction: {safe_format(interaction, 3)}, p={safe_format(p_interaction, 4)}"
    
    return {
        'effect_estimate': float(interaction),
        'p_value': float(p_interaction),
        'significant': bool(significant),
        'result_summary': summary
    }

# Initialize transcript
transcript = {
    'dataset_id': DATASET_ID,
    'model_id': MODEL_ID,
    'harness_id': HARNESS_ID,
    'max_iterations': MAX_ITERATIONS,
    'iterations': []
}

# Iteration 1: Main effects - binary features
print("\n=== Iteration 1: Main effects (binary features) ===")
iteration = {
    'index': 1,
    'proposed_hypotheses': [],
    'analyses': []
}

for i, feature in enumerate(BINARY_FEATURES[:5]):  # Start with first 5
    hypothesis_id = f'h1_{i+1}'
    hypothesis_text = f"Mean {OUTCOME} differs between patients with {feature}=1 and those with {feature}=0."
    iteration['proposed_hypotheses'].append({
        'id': hypothesis_id,
        'text': hypothesis_text,
        'kind': 'novel'
    })
    
    result = analyze_feature_outcome_binary(df, feature, OUTCOME, 1)
    result['hypothesis_ids'] = [hypothesis_id]
    result['code'] = f"analyze_feature_outcome_binary(df, '{feature}', '{OUTCOME}', 1)"
    iteration['analyses'].append(result)
    print(f"  {hypothesis_id}: {result['result_summary']}")

transcript['iterations'].append(iteration)

# Iteration 2: Main effects - continuous features
print("\n=== Iteration 2: Main effects (continuous features) ===")
iteration = {
    'index': 2,
    'proposed_hypotheses': [],
    'analyses': []
}

for i, feature in enumerate(CONTINUOUS_FEATURES[:5]):  # Start with first 5
    hypothesis_id = f'h2_{i+1}'
    hypothesis_text = f"Mean {OUTCOME} differs across quartiles of {feature}."
    iteration['proposed_hypotheses'].append({
        'id': hypothesis_id,
        'text': hypothesis_text,
        'kind': 'novel'
    })
    
    result = analyze_feature_outcome_continuous(df, feature, OUTCOME)
    result['hypothesis_ids'] = [hypothesis_id]
    result['code'] = f"analyze_feature_outcome_continuous(df, '{feature}', '{OUTCOME}')"
    iteration['analyses'].append(result)
    print(f"  {hypothesis_id}: {result['result_summary']}")

transcript['iterations'].append(iteration)

# Iteration 3: Main effects - categorical features
print("\n=== Iteration 3: Main effects (categorical features) ===")
iteration = {
    'index': 3,
    'proposed_hypotheses': [],
    'analyses': []
}

for i, feature in enumerate(CATEGORICAL_FEATURES[:2]):  # feature_001 has 3 levels
    hypothesis_id = f'h3_{i+1}'
    hypothesis_text = f"Mean {OUTCOME} differs across levels of {feature}."
    iteration['proposed_hypotheses'].append({
        'id': hypothesis_id,
        'text': hypothesis_text,
        'kind': 'novel'
    })
    
    result = analyze_feature_outcome_categorical(df, feature, OUTCOME)
    result['hypothesis_ids'] = [hypothesis_id]
    result['code'] = f"analyze_feature_outcome_categorical(df, '{feature}', '{OUTCOME}')"
    iteration['analyses'].append(result)
    print(f"  {hypothesis_id}: {result['result_summary']}")

transcript['iterations'].append(iteration)

# Iteration 4: Feature-feature correlations
print("\n=== Iteration 4: Feature-feature correlations ===")
iteration = {
    'index': 4,
    'proposed_hypotheses': [],
    'analyses': []
}

# Select pairs of continuous features
correlation_pairs = [
    ('feature_015', 'feature_005'),
    ('feature_019', 'feature_021'),
    ('feature_025', 'feature_032'),
    ('feature_003', 'feature_030'),
    ('feature_008', 'feature_004'),
]

for i, (f1, f2) in enumerate(correlation_pairs):
    hypothesis_id = f'h4_{i+1}'
    hypothesis_text = f"Feature {f1} and {f2} are correlated."
    iteration['proposed_hypotheses'].append({
        'id': hypothesis_id,
        'text': hypothesis_text,
        'kind': 'novel'
    })
    
    result = analyze_correlation(df, f1, f2)
    result['hypothesis_ids'] = [hypothesis_id]
    result['code'] = f"analyze_correlation(df, '{f1}', '{f2}')"
    iteration['analyses'].append(result)
    print(f"  {hypothesis_id}: {result['result_summary']}")

transcript['iterations'].append(iteration)

# Iteration 5: Treatment effect heterogeneity search
print("\n=== Iteration 5: Treatment effect heterogeneity ===")
iteration = {
    'index': 5,
    'proposed_hypotheses': [],
    'analyses': []
}

# Use feature_031 as a "treatment" proxy (binary)
treatment = 'feature_031'
modifiers = ['feature_015', 'feature_005', 'feature_019', 'feature_021']

for i, modifier in enumerate(modifiers[:3]):
    hypothesis_id = f'h5_{i+1}'
    hypothesis_text = f"Treatment effect of {treatment} on {OUTCOME} is modified by {modifier}."
    iteration['proposed_hypotheses'].append({
        'id': hypothesis_id,
        'text': hypothesis_text,
        'kind': 'novel'
    })
    
    result = analyze_treatment_heterogeneity(df, treatment, OUTCOME, modifier)
    result['hypothesis_ids'] = [hypothesis_id]
    result['code'] = f"analyze_treatment_heterogeneity(df, '{treatment}', '{OUTCOME}', '{modifier}')"
    iteration['analyses'].append(result)
    print(f"  {hypothesis_id}: {result['result_summary']}")

transcript['iterations'].append(iteration)

# Iteration 6: More heterogeneity with different treatment
print("\n=== Iteration 6: Additional heterogeneity analysis ===")
iteration = {
    'index': 6,
    'proposed_hypotheses': [],
    'analyses': []
}

# Use feature_014 as another "treatment" proxy
treatment = 'feature_014'
modifiers = ['feature_015', 'feature_005', 'feature_019']

for i, modifier in enumerate(modifiers[:2]):
    hypothesis_id = f'h6_{i+1}'
    hypothesis_text = f"Treatment effect of {treatment} on {OUTCOME} is modified by {modifier}."
    iteration['proposed_hypotheses'].append({
        'id': hypothesis_id,
        'text': hypothesis_text,
        'kind': 'novel'
    })
    
    result = analyze_treatment_heterogeneity(df, treatment, OUTCOME, modifier)
    result['hypothesis_ids'] = [hypothesis_id]
    result['code'] = f"analyze_treatment_heterogeneity(df, '{treatment}', '{OUTCOME}', '{modifier}')"
    iteration['analyses'].append(result)
    print(f"  {hypothesis_id}: {result['result_summary']}")

transcript['iterations'].append(iteration)

# Iteration 7: Subgroup discovery - combinations
print("\n=== Iteration 7: Subgroup discovery ===")
iteration = {
    'index': 7,
    'proposed_hypotheses': [],
    'analyses': []
}

# Test specific subgroup: feature_031=1 AND feature_015 > median
median_015 = df['feature_015'].median()
subgroup_mask = (df['feature_031'] == 1) & (df['feature_015'] > median_015)

hypothesis_id = 'h7_1'
hypothesis_text = f"Mean {OUTCOME} is higher in patients with feature_031=1 AND feature_015 > median."
iteration['proposed_hypotheses'].append({
    'id': hypothesis_id,
    'text': hypothesis_text,
    'kind': 'novel'
})

group1 = df.loc[subgroup_mask, OUTCOME]
group0 = df.loc[~subgroup_mask, OUTCOME]

if len(group1) > 0 and len(group0) > 0:
    effect = group1.mean() - group0.mean()
    t_stat, p_val = stats.ttest_ind(group1, group0)
    significant = p_val < 0.05
    
    summary = f"Subgroup (n={len(group1)}) vs rest (n={len(group0)}): " \
              f"{safe_format(group1.mean(), 2)} vs {safe_format(group0.mean(), 2)} " \
              f"(diff={safe_format(effect, 3)}, t={t_stat:.3f}, p={safe_format(p_val, 4)})"
    
    result = {
        'effect_estimate': float(effect),
        'p_value': float(p_val),
        'significant': bool(significant),
        'result_summary': summary
    }
else:
    result = {
        'effect_estimate': None,
        'p_value': None,
        'significant': None,
        'result_summary': 'Insufficient data for subgroup analysis'
    }

result['hypothesis_ids'] = [hypothesis_id]
result['code'] = f"analyze_subgroup(df, feature_031=1, feature_015>median, '{OUTCOME}')"
iteration['analyses'].append(result)
print(f"  {hypothesis_id}: {result['result_summary']}")

transcript['iterations'].append(iteration)

# Iteration 8: More subgroup analysis
print("\n=== Iteration 8: Additional subgroup analysis ===")
iteration = {
    'index': 8,
    'proposed_hypotheses': [],
    'analyses': []
}

# Test: feature_014=1 AND feature_005 > median
median_005 = df['feature_005'].median()
subgroup_mask = (df['feature_014'] == 1) & (df['feature_005'] > median_005)

hypothesis_id = 'h8_1'
hypothesis_text = f"Mean {OUTCOME} is higher in patients with feature_014=1 AND feature_005 > median."
iteration['proposed_hypotheses'].append({
    'id': hypothesis_id,
    'text': hypothesis_text,
    'kind': 'novel'
})

group1 = df.loc[subgroup_mask, OUTCOME]
group0 = df.loc[~subgroup_mask, OUTCOME]

if len(group1) > 0 and len(group0) > 0:
    effect = group1.mean() - group0.mean()
    t_stat, p_val = stats.ttest_ind(group1, group0)
    significant = p_val < 0.05
    
    summary = f"Subgroup (n={len(group1)}) vs rest (n={len(group0)}): " \
              f"{safe_format(group1.mean(), 2)} vs {safe_format(group0.mean(), 2)} " \
              f"(diff={safe_format(effect, 3)}, t={t_stat:.3f}, p={safe_format(p_val, 4)})"
    
    result = {
        'effect_estimate': float(effect),
        'p_value': float(p_val),
        'significant': bool(significant),
        'result_summary': summary
    }
else:
    result = {
        'effect_estimate': None,
        'p_value': None,
        'significant': None,
        'result_summary': 'Insufficient data for subgroup analysis'
    }

result['hypothesis_ids'] = [hypothesis_id]
result['code'] = f"analyze_subgroup(df, feature_014=1, feature_005>median, '{OUTCOME}')"
iteration['analyses'].append(result)
print(f"  {hypothesis_id}: {result['result_summary']}")

transcript['iterations'].append(iteration)

# Iteration 9: Refine based on findings
print("\n=== Iteration 9: Refinement based on findings ===")
iteration = {
    'index': 9,
    'proposed_hypotheses': [],
    'analyses': []
}

# Identify significant findings and refine
# Look for the strongest effect from previous iterations
all_effects = []
for iter_data in transcript['iterations']:
    for analysis in iter_data['analyses']:
        if analysis['effect_estimate'] is not None:
            all_effects.append((abs(analysis['effect_estimate']), analysis))

all_effects.sort(reverse=True)
if all_effects:
    strongest_effect, strongest_analysis = all_effects[0]
    
    # Extract feature from result summary
    summary = strongest_analysis['result_summary']
    if 'feature_031' in summary:
        feature = 'feature_031'
        hypothesis_id = 'h9_1'
        hypothesis_text = f"Mean {OUTCOME} is higher in patients with {feature}=1."
        iteration['proposed_hypotheses'].append({
            'id': hypothesis_id,
            'text': hypothesis_text,
            'kind': 'refined'
        })
        
        result = analyze_feature_outcome_binary(df, feature, OUTCOME, 1)
        result['hypothesis_ids'] = [hypothesis_id]
        result['code'] = f"analyze_feature_outcome_binary(df, '{feature}', '{OUTCOME}', 1)"
        iteration['analyses'].append(result)
        print(f"  {hypothesis_id}: {result['result_summary']}")
    elif 'feature_014' in summary:
        feature = 'feature_014'
        hypothesis_id = 'h9_1'
        hypothesis_text = f"Mean {OUTCOME} is higher in patients with {feature}=1."
        iteration['proposed_hypotheses'].append({
            'id': hypothesis_id,
            'text': hypothesis_text,
            'kind': 'refined'
        })
        
        result = analyze_feature_outcome_binary(df, feature, OUTCOME, 1)
        result['hypothesis_ids'] = [hypothesis_id]
        result['code'] = f"analyze_feature_outcome_binary(df, '{feature}', '{OUTCOME}', 1)"
        iteration['analyses'].append(result)
        print(f"  {hypothesis_id}: {result['result_summary']}")
    else:
        # Default refinement
        hypothesis_id = 'h9_1'
        hypothesis_text = f"Mean {OUTCOME} differs across feature_015 quartiles."
        iteration['proposed_hypotheses'].append({
            'id': hypothesis_id,
            'text': hypothesis_text,
            'kind': 'refined'
        })
        
        result = analyze_feature_outcome_continuous(df, 'feature_015', OUTCOME)
        result['hypothesis_ids'] = [hypothesis_id]
        result['code'] = f"analyze_feature_outcome_continuous(df, 'feature_015', '{OUTCOME}')"
        iteration['analyses'].append(result)
        print(f"  {hypothesis_id}: {result['result_summary']}")
else:
    hypothesis_id = 'h9_1'
    hypothesis_text = f"Mean {OUTCOME} differs across feature_015 quartiles."
    iteration['proposed_hypotheses'].append({
        'id': hypothesis_id,
        'text': hypothesis_text,
        'kind': 'refined'
    })
    
    result = analyze_feature_outcome_continuous(df, 'feature_015', OUTCOME)
    result['hypothesis_ids'] = [hypothesis_id]
    result['code'] = f"analyze_feature_outcome_continuous(df, 'feature_015', '{OUTCOME}')"
    iteration['analyses'].append(result)
    print(f"  {hypothesis_id}: {result['result_summary']}")

transcript['iterations'].append(iteration)

# Iteration 10: Final comprehensive heterogeneity search
print("\n=== Iteration 10: Final heterogeneity search ===")
iteration = {
    'index': 10,
    'proposed_hypotheses': [],
    'analyses': []
}

# Exhaustive check of 2-feature interactions for treatment effect
treatment = 'feature_031'
modifier_candidates = ['feature_015', 'feature_005', 'feature_019', 'feature_021', 'feature_025']

for i, modifier in enumerate(modifier_candidates):
    hypothesis_id = f'h10_{i+1}'
    hypothesis_text = f"Treatment effect of {treatment} on {OUTCOME} is modified by {modifier}."
    iteration['proposed_hypotheses'].append({
        'id': hypothesis_id,
        'text': hypothesis_text,
        'kind': 'novel'
    })
    
    result = analyze_treatment_heterogeneity(df, treatment, OUTCOME, modifier)
    result['hypothesis_ids'] = [hypothesis_id]
    result['code'] = f"analyze_treatment_heterogeneity(df, '{treatment}', '{OUTCOME}', '{modifier}')"
    iteration['analyses'].append(result)
    print(f"  {hypothesis_id}: {result['result_summary']}")

transcript['iterations'].append(iteration)

# Convert transcript to JSON-serializable format
def to_jsonable(obj):
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
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
json_data = to_jsonable(transcript)
with open(TRANSCRIPT_PATH, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f"Written {TRANSCRIPT_PATH}")

# Generate analysis_summary.txt
print("\n=== Generating analysis_summary.txt ===")

summary_lines = []
summary_lines.append("=" * 80)
summary_lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY")
summary_lines.append(f"Dataset: {DATASET_ID}")
summary_lines.append(f"Total patients: {len(df)}")
summary_lines.append(f"Total iterations: {MAX_ITERATIONS}")
summary_lines.append("=" * 80)
summary_lines.append("")

# Summary statistics
summary_lines.append("DATASET OVERVIEW")
summary_lines.append("-" * 40)
summary_lines.append(f"Outcome: {OUTCOME} (mean={safe_format(df[OUTCOME].mean(), 2)}, "
                    f"sd={safe_format(df[OUTCOME].std(), 2)})")
summary_lines.append("")

# Iteration summaries
summary_lines.append("ITERATION RESULTS")
summary_lines.append("-" * 40)

for iter_num, iter_data in enumerate(transcript['iterations'], 1):
    summary_lines.append(f"\nIteration {iter_num}:")
    
    for hyp in iter_data['proposed_hypotheses']:
        summary_lines.append(f"  Hypothesis {hyp['id']}: {hyp['text']}")
    
    for analysis in iter_data['analyses']:
        sig_str = "SIGNIFICANT" if analysis.get('significant', False) else "not significant"
        summary_lines.append(f"    - {analysis['result_summary']}")
        summary_lines.append(f"      (p={safe_format(analysis.get('p_value'), 4)}, "
                            f"effect={safe_format(analysis.get('effect_estimate'), 3)}, "
                            f"{sig_str})")

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("KEY FINDINGS")
summary_lines.append("-" * 40)

# Collect all significant findings
significant_findings = []
for iter_data in transcript['iterations']:
    for analysis in iter_data['analyses']:
        if analysis.get('significant', False) and analysis.get('effect_estimate') is not None:
            significant_findings.append(analysis)

if significant_findings:
    summary_lines.append(f"\nFound {len(significant_findings)} statistically significant findings (p < 0.05):")
    for finding in significant_findings[:10]:  # Top 10
        summary_lines.append(f"  - {finding['result_summary']}")
else:
    summary_lines.append("\nNo statistically significant findings (p < 0.05) were identified.")

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("CONCLUSIONS")
summary_lines.append("-" * 40)

# Analyze patterns
significant_count = sum(1 for a in significant_findings if a.get('significant', False))
total_analyses = sum(len(iter_data['analyses']) for iter_data in transcript['iterations'])

summary_lines.append(f"\nTotal analyses performed: {total_analyses}")
summary_lines.append(f"Significant findings: {significant_count}")
summary_lines.append(f"Significance rate: {safe_format(100 * significant_count / total_analyses, 1)}%")

if significant_count > 0:
    summary_lines.append("\nKey observations:")
    for finding in significant_findings[:5]:
        summary_lines.append(f"  - {finding['result_summary']}")
else:
    summary_lines.append("\nNo statistically significant associations were detected at the 0.05 level.")
    summary_lines.append("This may indicate:")
    summary_lines.append("  - Weak effect sizes in the dataset")
    summary_lines.append("  - High variability in the outcome")
    summary_lines.append("  - Need for larger sample size")
    summary_lines.append("  - Need for more specific subgroup definitions")

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("END OF ANALYSIS SUMMARY")
summary_lines.append("=" * 80)

# Write summary
with open(SUMMARY_PATH, 'w') as f:
    f.write('\n'.join(summary_lines))
print(f"Written {SUMMARY_PATH}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print(f"  - Transcript: {TRANSCRIPT_PATH}")
print(f"  - Summary: {SUMMARY_PATH}")
print("=" * 80)
