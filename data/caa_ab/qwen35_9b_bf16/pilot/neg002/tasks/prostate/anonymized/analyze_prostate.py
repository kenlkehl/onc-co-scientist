#!/usr/bin/env python3
"""
End-to-end oncology dataset analysis script.
Performs iterative hypothesis testing on prostate cancer dataset.
Outputs: transcript.json and analysis_summary.txt
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional

# Load dataset
df = pd.read_parquet('dataset.parquet')

# Binary outcome
OUTCOME = 'objective_response'

# Features to analyze (binary and categorical)
BINARY_FEATURES = [
    'feature_006', 'feature_008', 'feature_011', 'feature_015', 
    'feature_017', 'feature_019', 'feature_021', 'feature_023', 
    'feature_027'
]

# Multi-level categorical features
CAT_FEATURES = {
    'feature_001': [0, 1, 2],
    'feature_010': [6, 7, 8, 9, 10],
    'feature_028': list(range(3, 37))  # 3-36
}

# Continuous features for correlation
CONTINUOUS_FEATURES = [
    'feature_016', 'feature_002', 'feature_018', 'feature_020', 
    'feature_024', 'feature_031', 'feature_009', 'feature_029', 
    'feature_003', 'feature_012', 'feature_025', 'feature_007', 
    'feature_014', 'feature_032'
]

def safe_mean(arr: np.ndarray) -> float:
    """Return mean or 0.0 for empty arrays."""
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr))

def safe_std(arr: np.ndarray) -> float:
    """Return std or 0.0 for empty arrays."""
    if len(arr) == 0:
        return 0.0
    return float(np.std(arr))

def compute_rate_masked(df: pd.DataFrame, feature: str, value: Any, outcome: str) -> Tuple[float, float, int, int]:
    """
    Compute outcome rate for group with feature=value vs complement.
    Returns: (rate_with, rate_without, n_with, n_without)
    """
    mask = df[feature] == value
    n_with = int(mask.sum())
    n_without = int(~mask.sum())
    rate_with = safe_mean(df.loc[mask, outcome].values)
    rate_without = safe_mean(df.loc[~mask, outcome].values)
    return rate_with, rate_without, n_with, n_without

def compute_2x2_table(df: pd.DataFrame, feature: str, value: Any, outcome: str) -> np.ndarray:
    """
    Build 2x2 contingency table for feature=value vs outcome=1.
    [[a, b], [c, d]] where a=outcome=1&feature=value, etc.
    """
    feature_eq = df[feature] == value
    feature_ne = df[feature] != value
    outcome_eq = df[outcome] == 1
    outcome_ne = df[outcome] == 0
    
    a = int(feature_eq & outcome_eq).sum()
    b = int(feature_eq & outcome_ne).sum()
    c = int(feature_ne & outcome_eq).sum()
    d = int(feature_ne & outcome_ne).sum()
    return np.array([[a, b], [c, d]], dtype=float)

def chi2_test(table: np.ndarray, correction: bool = True) -> Tuple[float, float]:
    """
    Perform chi-square test.
    Returns: (statistic, p_value)
    """
    try:
        stat, p, _, _ = stats.chi2_contingency(table, correction=correction)
        return float(stat), float(p)
    except:
        return 0.0, 1.0

def fisher_exact_test(table: np.ndarray) -> Tuple[float, float]:
    """
    Perform Fisher's exact test.
    Returns: (odds_ratio, p_value)
    """
    try:
        _, p = stats.fisher_exact(table, alternative='two-sided')
        return float(p)
    except:
        return 1.0

def ttest_ind_groups(df: pd.DataFrame, feature: str, value: Any, outcome: str) -> Tuple[float, float]:
    """
    Perform t-test comparing outcome means between feature=value and feature!=value.
    Returns: (effect_estimate, p_value)
    """
    mask = df[feature] == value
    group1 = df.loc[mask, outcome].values
    group2 = df.loc[~mask, outcome].values
    
    if len(group1) == 0 or len(group2) == 0:
        return 0.0, 1.0
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    effect = safe_mean(group1) - safe_mean(group2)
    return effect, float(p_value)

def pearson_corr(df: pd.DataFrame, feature: str, outcome: str) -> Tuple[float, float]:
    """
    Compute Pearson correlation between feature and outcome.
    Returns: (correlation, p_value)
    """
    try:
        corr, p = stats.pearsonr(df[feature], df[outcome])
        return float(corr), float(p)
    except:
        return 0.0, 1.0

def compute_effect_categorical(df: pd.DataFrame, feature: str, level: Any, outcome: str) -> Tuple[float, float]:
    """
    Compute effect estimate for categorical feature level vs reference (lowest level).
    Returns: (effect_estimate, p_value)
    """
    levels = sorted(CAT_FEATURES.get(feature, [level]))
    ref_level = levels[0]
    
    mask = df[feature] == level
    ref_mask = df[feature] == ref_level
    
    if mask.sum() == 0 or ref_mask.sum() == 0:
        return 0.0, 1.0
    
    rate_level = safe_mean(df.loc[mask, outcome].values)
    rate_ref = safe_mean(df.loc[ref_mask, outcome].values)
    effect = rate_level - rate_ref
    
    # Chi-square test
    table = compute_2x2_table(df, feature, level, outcome)
    _, p = chi2_test(table)
    
    return effect, p

def compute_effect_continuous(df: pd.DataFrame, feature: str, outcome: str) -> Tuple[float, float]:
    """
    Compute effect estimate for continuous feature (per unit increase).
    Returns: (effect_estimate, p_value)
    """
    try:
        # Linear regression
        model = stats.linregress(df[feature], df[outcome])
        effect = model.slope
        p_value = model.pvalue
        return float(effect), float(p_value)
    except:
        return 0.0, 1.0

def run_feature_outcome_analysis(df: pd.DataFrame, feature: str, outcome: str, feature_type: str) -> Dict[str, Any]:
    """
    Run analysis for a feature-outcome pair.
    Returns analysis result dictionary.
    """
    result = {
        'feature': feature,
        'outcome': outcome,
        'feature_type': feature_type,
        'effect_estimate': None,
        'p_value': None,
        'significant': False,
        'result_summary': ''
    }
    
    if feature_type == 'binary':
        rate_with, rate_without, n_with, n_without = compute_rate_masked(df, feature, 1, outcome)
        effect = rate_with - rate_without
        table = compute_2x2_table(df, feature, 1, outcome)
        _, p = chi2_test(table)
        
        result['effect_estimate'] = effect
        result['p_value'] = p
        result['significant'] = p < 0.05
        result['result_summary'] = f"Rate with feature=1: {rate_with:.3f}, without: {rate_without:.3f} (chi2 p={p:.4f})"
        
    elif feature_type == 'categorical':
        levels = CAT_FEATURES.get(feature, [1])
        ref_level = min(levels)
        
        # Compare highest level vs reference
        max_level = max(levels)
        effect, p = compute_effect_categorical(df, feature, max_level, outcome)
        
        result['effect_estimate'] = effect
        result['p_value'] = p
        result['significant'] = p < 0.05
        result['result_summary'] = f"Level {max_level} vs ref {ref_level}: effect={effect:.4f}, p={p:.4f}"
        
    elif feature_type == 'continuous':
        effect, p = compute_effect_continuous(df, feature, outcome)
        
        result['effect_estimate'] = effect
        result['p_value'] = p
        result['significant'] = p < 0.05
        result['result_summary'] = f"Per-unit effect: {effect:.4f}, p={p:.4f}"
    
    return result

def run_treatment_heterogeneity(df: pd.DataFrame, treatment: str, outcome: str, modifier: str) -> Dict[str, Any]:
    """
    Test treatment effect heterogeneity by modifier.
    Returns analysis result dictionary.
    """
    result = {
        'treatment': treatment,
        'outcome': outcome,
        'modifier': modifier,
        'effect_estimate': None,
        'p_value': None,
        'significant': False,
        'result_summary': ''
    }
    
    # Get treatment groups
    treatment_mask = df[treatment] == 1
    control_mask = df[treatment] == 0
    
    if treatment_mask.sum() == 0 or control_mask.sum() == 0:
        return result
    
    # Get modifier groups
    modifier_mask = df[modifier] == 1
    
    # Treatment effect in modifier=1 group
    t1_mask = treatment_mask & modifier_mask
    c1_mask = control_mask & modifier_mask
    
    if t1_mask.sum() == 0 or c1_mask.sum() == 0:
        return result
    
    rate_t1 = safe_mean(df.loc[t1_mask, outcome].values)
    rate_c1 = safe_mean(df.loc[c1_mask, outcome].values)
    effect_t1 = rate_t1 - rate_c1
    
    # Treatment effect in modifier=0 group
    t0_mask = treatment_mask & ~modifier_mask
    c0_mask = control_mask & ~modifier_mask
    
    if t0_mask.sum() == 0 or c0_mask.sum() == 0:
        return result
    
    rate_t0 = safe_mean(df.loc[t0_mask, outcome].values)
    rate_c0 = safe_mean(df.loc[c0_mask, outcome].values)
    effect_t0 = rate_t0 - rate_c0
    
    # Interaction effect
    interaction = effect_t1 - effect_t0
    
    # Test if interaction is significant
    # Use 2x2x2 table for interaction test
    a = int(t1_mask & modifier_mask & (df[outcome] == 1)).sum()
    b = int(t1_mask & modifier_mask & (df[outcome] == 0)).sum()
    c = int(c1_mask & modifier_mask & (df[outcome] == 1)).sum()
    d = int(c1_mask & modifier_mask & (df[outcome] == 0)).sum()
    e = int(t0_mask & modifier_mask & (df[outcome] == 1)).sum()
    f = int(t0_mask & modifier_mask & (df[outcome] == 0)).sum()
    g = int(t1_mask & ~modifier_mask & (df[outcome] == 1)).sum()
    h = int(t1_mask & ~modifier_mask & (df[outcome] == 0)).sum()
    i = int(c0_mask & ~modifier_mask & (df[outcome] == 1)).sum()
    j = int(c0_mask & ~modifier_mask & (df[outcome] == 0)).sum()
    
    table = np.array([[a, b, e, f], [c, d, g, h], [i, j, 0, 0]], dtype=float)
    _, p = chi2_test(table)
    
    result['effect_estimate'] = interaction
    result['p_value'] = p
    result['significant'] = p < 0.05
    result['result_summary'] = f"Interaction effect: {interaction:.4f}, p={p:.4f}"
    
    return result

def run_correlation_analysis(df: pd.DataFrame, feature: str, outcome: str) -> Dict[str, Any]:
    """
    Run correlation analysis for continuous feature-outcome pair.
    Returns analysis result dictionary.
    """
    result = {
        'feature': feature,
        'outcome': outcome,
        'effect_estimate': None,
        'p_value': None,
        'significant': False,
        'result_summary': ''
    }
    
    corr, p = pearson_corr(df, feature, outcome)
    
    result['effect_estimate'] = corr
    result['p_value'] = p
    result['significant'] = p < 0.05
    result['result_summary'] = f"Correlation: {corr:.4f}, p={p:.4f}"
    
    return result

def to_jsonable(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj):
            return None
        return obj
    return obj

def generate_summary(transcript: List[Dict]) -> str:
    """Generate analysis summary text from transcript."""
    lines = []
    lines.append("=" * 70)
    lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    
    # Overall statistics
    total_hypotheses = sum(len(iteration['proposed_hypotheses']) for iteration in transcript)
    total_analyses = sum(len(iteration['analyses']) for iteration in transcript)
    significant_count = sum(1 for iter in transcript for a in iter['analyses'] if a.get('significant', False))
    
    lines.append(f"Total hypotheses proposed: {total_hypotheses}")
    lines.append(f"Total analyses performed: {total_analyses}")
    lines.append(f"Statistically significant findings (p<0.05): {significant_count}")
    lines.append("")
    
    # Iteration-by-iteration summary
    lines.append("-" * 70)
    lines.append("ITERATION-BY-ITERATION RESULTS")
    lines.append("-" * 70)
    lines.append("")
    
    for iteration in transcript:
        iter_num = iteration['index']
        lines.append(f"Iteration {iter_num}:")
        
        for hyp in iteration['proposed_hypotheses']:
            lines.append(f"  Hypothesis: {hyp['text']}")
        
        for analysis in iteration['analyses']:
            sig_marker = " [SIGNIFICANT]" if analysis.get('significant', False) else ""
            lines.append(f"  Analysis: {analysis['result_summary']}{sig_marker}")
        
        lines.append("")
    
    # Key findings summary
    lines.append("-" * 70)
    lines.append("KEY FINDINGS")
    lines.append("-" * 70)
    lines.append("")
    
    # Collect all significant findings
    significant_findings = []
    for iteration in transcript:
        for analysis in iteration['analyses']:
            if analysis.get('significant', False):
                significant_findings.append(analysis)
    
    if significant_findings:
        lines.append("Significant findings identified:")
        for finding in significant_findings[:10]:  # Top 10
            lines.append(f"  - {finding['result_summary']}")
    else:
        lines.append("No statistically significant findings (p<0.05) were identified.")
    
    lines.append("")
    
    # Treatment heterogeneity findings
    lines.append("-" * 70)
    lines.append("TREATMENT EFFECT HETEROGENEITY")
    lines.append("-" * 70)
    lines.append("")
    
    hetero_findings = [a for a in transcript if 'treatment' in a and 'modifier' in a]
    if hetero_findings:
        lines.append("Treatment-by-modifier interactions tested:")
        for finding in hetero_findings:
            sig_marker = " [SIGNIFICANT]" if finding.get('significant', False) else ""
            lines.append(f"  - {finding['result_summary']}{sig_marker}")
    else:
        lines.append("No treatment effect heterogeneity analyses were performed.")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SUMMARY")
    lines.append("=" * 70)
    
    return '\n'.join(lines)

def main():
    """Main analysis loop."""
    transcript = []
    iteration_num = 0
    
    # Iteration 1: Feature-outcome associations (binary features)
    iteration_num += 1
    hypotheses = []
    analyses = []
    
    for feature in BINARY_FEATURES:
        hyp_id = f"h{iteration_num}_{feature}"
        hypotheses.append({
            'id': hyp_id,
            'text': f"Patients with {feature}=1 have different objective_response rates than those with {feature}=0.",
            'kind': 'novel'
        })
    
    for feature in BINARY_FEATURES:
        hyp_id = f"h{iteration_num}_{feature}"
        result = run_feature_outcome_analysis(df, feature, OUTCOME, 'binary')
        analyses.append({
            'hypothesis_ids': [hyp_id],
            'result_summary': result['result_summary'],
            'p_value': result['p_value'],
            'effect_estimate': result['effect_estimate'],
            'significant': result['significant']
        })
    
    transcript.append({
        'index': iteration_num,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses
    })
    
    # Iteration 2: Categorical feature-outcome associations
    iteration_num += 1
    hypotheses = []
    analyses = []
    
    for feature, levels in CAT_FEATURES.items():
        hyp_id = f"h{iteration_num}_{feature}"
        hypotheses.append({
            'id': hyp_id,
            'text': f"Patients with {feature} at higher levels have different objective_response rates than those at lower levels.",
            'kind': 'novel'
        })
    
    for feature, levels in CAT_FEATURES.items():
        hyp_id = f"h{iteration_num}_{feature}"
        result = run_feature_outcome_analysis(df, feature, OUTCOME, 'categorical')
        analyses.append({
            'hypothesis_ids': [hyp_id],
            'result_summary': result['result_summary'],
            'p_value': result['p_value'],
            'effect_estimate': result['effect_estimate'],
            'significant': result['significant']
        })
    
    transcript.append({
        'index': iteration_num,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses
    })
    
    # Iteration 3: Continuous feature-outcome correlations
    iteration_num += 1
    hypotheses = []
    analyses = []
    
    for feature in CONTINUOUS_FEATURES:
        hyp_id = f"h{iteration_num}_{feature}"
        hypotheses.append({
            'id': hyp_id,
            'text': f"Higher values of {feature} are associated with different objective_response rates.",
            'kind': 'novel'
        })
    
    for feature in CONTINUOUS_FEATURES:
        hyp_id = f"h{iteration_num}_{feature}"
        result = run_correlation_analysis(df, feature, OUTCOME)
        analyses.append({
            'hypothesis_ids': [hyp_id],
            'result_summary': result['result_summary'],
            'p_value': result['p_value'],
            'effect_estimate': result['effect_estimate'],
            'significant': result['significant']
        })
    
    transcript.append({
        'index': iteration_num,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses
    })
    
    # Iteration 4: Treatment effect heterogeneity search
    # Use feature_006 as treatment (binary, good balance)
    iteration_num += 1
    hypotheses = []
    analyses = []
    
    treatment = 'feature_006'
    modifier_candidates = BINARY_FEATURES.copy()
    
    for modifier in modifier_candidates:
        if modifier == treatment:
            continue
        hyp_id = f"h{iteration_num}_{treatment}_by_{modifier}"
        hypotheses.append({
            'id': hyp_id,
            'text': f"The effect of {treatment}=1 on objective_response differs between patients with {modifier}=1 vs {modifier}=0.",
            'kind': 'novel'
        })
    
    for modifier in modifier_candidates:
        if modifier == treatment:
            continue
        hyp_id = f"h{iteration_num}_{treatment}_by_{modifier}"
        result = run_treatment_heterogeneity(df, treatment, OUTCOME, modifier)
        analyses.append({
            'hypothesis_ids': [hyp_id],
            'result_summary': result['result_summary'],
            'p_value': result['p_value'],
            'effect_estimate': result['effect_estimate'],
            'significant': result['significant']
        })
    
    transcript.append({
        'index': iteration_num,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses
    })
    
    # Iteration 5: Additional binary feature-outcome analysis
    # Focus on features not yet analyzed in depth
    iteration_num += 1
    hypotheses = []
    analyses = []
    
    # Analyze feature_001 (3 levels) more thoroughly
    hyp_id = f"h{iteration_num}_feature_001"
    hypotheses.append({
        'id': hyp_id,
        'text': "Patients with feature_001=2 have different objective_response rates than those with feature_001=0.",
        'kind': 'novel'
    })
    
    result = run_feature_outcome_analysis(df, 'feature_001', OUTCOME, 'categorical')
    analyses.append({
        'hypothesis_ids': [hyp_id],
        'result_summary': result['result_summary'],
        'p_value': result['p_value'],
        'effect_estimate': result['effect_estimate'],
        'significant': result['significant']
    })
    
    # Analyze feature_010 (5 levels)
    hyp_id = f"h{iteration_num}_feature_010"
    hypotheses.append({
        'id': hyp_id,
        'text': "Patients with feature_010=10 have different objective_response rates than those with feature_010=6.",
        'kind': 'novel'
    })
    
    result = run_feature_outcome_analysis(df, 'feature_010', OUTCOME, 'categorical')
    analyses.append({
        'hypothesis_ids': [hyp_id],
        'result_summary': result['result_summary'],
        'p_value': result['p_value'],
        'effect_estimate': result['effect_estimate'],
        'significant': result['significant']
    })
    
    transcript.append({
        'index': iteration_num,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses
    })
    
    # Iteration 6: Treatment heterogeneity with feature_008
    iteration_num += 1
    hypotheses = []
    analyses = []
    
    treatment = 'feature_008'
    modifier_candidates = BINARY_FEATURES.copy()
    
    for modifier in modifier_candidates:
        if modifier == treatment:
            continue
        hyp_id = f"h{iteration_num}_{treatment}_by_{modifier}"
        hypotheses.append({
            'id': hyp_id,
            'text': f"The effect of {treatment}=1 on objective_response differs between patients with {modifier}=1 vs {modifier}=0.",
            'kind': 'novel'
        })
    
    for modifier in modifier_candidates:
        if modifier == treatment:
            continue
        hyp_id = f"h{iteration_num}_{treatment}_by_{modifier}"
        result = run_treatment_heterogeneity(df, treatment, OUTCOME, modifier)
        analyses.append({
            'hypothesis_ids': [hyp_id],
            'result_summary': result['result_summary'],
            'p_value': result['p_value'],
            'effect_estimate': result['effect_estimate'],
            'significant': result['significant']
        })
    
    transcript.append({
        'index': iteration_num,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses
    })
    
    # Iteration 7: Correlation analysis for remaining continuous features
    iteration_num += 1
    hypotheses = []
    analyses = []
    
    for feature in CONTINUOUS_FEATURES:
        hyp_id = f"h{iteration_num}_{feature}"
        hypotheses.append({
            'id': hyp_id,
            'text': f"Higher values of {feature} are associated with different objective_response rates.",
            'kind': 'novel'
        })
    
    for feature in CONTINUOUS_FEATURES:
        hyp_id = f"h{iteration_num}_{feature}"
        result = run_correlation_analysis(df, feature, OUTCOME)
        analyses.append({
            'hypothesis_ids': [hyp_id],
            'result_summary': result['result_summary'],
            'p_value': result['p_value'],
            'effect_estimate': result['effect_estimate'],
            'significant': result['significant']
        })
    
    transcript.append({
        'index': iteration_num,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses
    })
    
    # Iteration 8: Treatment heterogeneity with feature_011
    iteration_num += 1
    hypotheses = []
    analyses = []
    
    treatment = 'feature_011'
    modifier_candidates = BINARY_FEATURES.copy()
    
    for modifier in modifier_candidates:
        if modifier == treatment:
            continue
        hyp_id = f"h{iteration_num}_{treatment}_by_{modifier}"
        hypotheses.append({
            'id': hyp_id,
            'text': f"The effect of {treatment}=1 on objective_response differs between patients with {modifier}=1 vs {modifier}=0.",
            'kind': 'novel'
        })
    
    for modifier in modifier_candidates:
        if modifier == treatment:
            continue
        hyp_id = f"h{iteration_num}_{treatment}_by_{modifier}"
        result = run_treatment_heterogeneity(df, treatment, OUTCOME, modifier)
        analyses.append({
            'hypothesis_ids': [hyp_id],
            'result_summary': result['result_summary'],
            'p_value': result['p_value'],
            'effect_estimate': result['effect_estimate'],
            'significant': result['significant']
        })
    
    transcript.append({
        'index': iteration_num,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses
    })
    
    # Iteration 9: Additional categorical analysis
    iteration_num += 1
    hypotheses = []
    analyses = []
    
    # feature_028 (many levels)
    hyp_id = f"h{iteration_num}_feature_028"
    hypotheses.append({
        'id': hyp_id,
        'text': "Patients with feature_028 at higher levels have different objective_response rates than those at lower levels.",
        'kind': 'novel'
    })
    
    result = run_feature_outcome_analysis(df, 'feature_028', OUTCOME, 'categorical')
    analyses.append({
        'hypothesis_ids': [hyp_id],
        'result_summary': result['result_summary'],
        'p_value': result['p_value'],
        'effect_estimate': result['effect_estimate'],
        'significant': result['significant']
    })
    
    transcript.append({
        'index': iteration_num,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses
    })
    
    # Iteration 10: Final treatment heterogeneity search
    iteration_num += 1
    hypotheses = []
    analyses = []
    
    treatment = 'feature_015'
    modifier_candidates = BINARY_FEATURES.copy()
    
    for modifier in modifier_candidates:
        if modifier == treatment:
            continue
        hyp_id = f"h{iteration_num}_{treatment}_by_{modifier}"
        hypotheses.append({
            'id': hyp_id,
            'text': f"The effect of {treatment}=1 on objective_response differs between patients with {modifier}=1 vs {modifier}=0.",
            'kind': 'novel'
        })
    
    for modifier in modifier_candidates:
        if modifier == treatment:
            continue
        hyp_id = f"h{iteration_num}_{treatment}_by_{modifier}"
        result = run_treatment_heterogeneity(df, treatment, OUTCOME, modifier)
        analyses.append({
            'hypothesis_ids': [hyp_id],
            'result_summary': result['result_summary'],
            'p_value': result['p_value'],
            'effect_estimate': result['effect_estimate'],
            'significant': result['significant']
        })
    
    transcript.append({
        'index': iteration_num,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses
    })
    
    # Write transcript.json
    transcript_json = to_jsonable(transcript)
    with open('transcript.json', 'w') as f:
        json.dump(transcript_json, f, indent=2)
    
    # Generate and write analysis_summary.txt
    summary = generate_summary(transcript)
    with open('analysis_summary.txt', 'w') as f:
        f.write(summary)
    
    print("Analysis complete!")
    print(f"Generated transcript.json with {len(transcript)} iterations")
    print(f"Generated analysis_summary.txt")

if __name__ == '__main__':
    main()
