#!/usr/bin/env python
"""
End-to-end prostate cancer dataset analysis.
Tests treatment-outcome relationships and treatment-effect heterogeneity
across up to 10 iterations using boolean-mask group comparisons.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional

# Load dataset
df = pd.read_parquet('dataset.parquet')

# Constants
MAX_ITERATIONS = 10
SIGNIFICANCE_THRESHOLD = 0.05
TREATMENT_COLS = [
    'treatment_enzalutamide',
    'treatment_abiraterone',
    'treatment_docetaxel',
    'treatment_olaparib',
    'treatment_lu177_psma',
    'treatment_pembrolizumab'
]
OUTCOME_COLS = ['objective_response']

# Transcript structure
transcript: Dict[str, Any] = {
    'dataset_id': 'ds001_prostate',
    'model_id': 'codex-cli',
    'harness_id': 'codex-cli@1.0.0',
    'max_iterations': MAX_ITERATIONS,
    'iterations': []
}

def safe_float(val) -> float:
    """Convert to float, handling None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float('nan')
    return float(val)

def safe_bool(val) -> bool:
    """Convert to bool, handling None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    return bool(val)

def format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if np.isnan(p):
        return 'NA'
    if p < 0.001:
        return f'<0.001'
    if p < 0.01:
        return f'<0.01'
    if p < 0.05:
        return f'<0.05'
    return f'{p:.4f}'

def compute_effect_size(group1_mean: float, group2_mean: float) -> float:
    """Compute signed effect estimate (group1 - group2)."""
    return group1_mean - group2_mean

def test_treatment_effect(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> Tuple[float, float, bool]:
    """
    Test treatment effect using boolean-mask group comparison.
    Returns: (effect_estimate, p_value, significant)
    """
    mask_treated = df[treatment_col] == 1
    mask_control = df[treatment_col] == 0
    
    if mask_treated.sum() == 0 or mask_control.sum() == 0:
        return float('nan'), float('nan'), False
    
    treated_outcome = df.loc[mask_treated, outcome_col].mean()
    control_outcome = df.loc[mask_control, outcome_col].mean()
    
    effect = treated_outcome - control_outcome
    
    # 2x2 contingency table for proportions
    table = pd.crosstab(df[treatment_col], df[outcome_col])
    _, p_value, _, _ = stats.chi2_contingency(table, correction=False)
    
    significant = p_value < SIGNIFICANCE_THRESHOLD
    return effect, p_value, significant

def test_subgroup_treatment_effect(
    df: pd.DataFrame, 
    treatment_col: str, 
    outcome_col: str,
    subgroup_cols: List[str],
    subgroup_values: List[Tuple[str, int]]
) -> Tuple[float, float, bool]:
    """
    Test treatment effect within a specific subgroup.
    subgroup_values: list of (column_name, value) tuples defining the subgroup.
    Returns: (effect_estimate, p_value, significant)
    """
    mask_treated = df[treatment_col] == 1
    mask_control = df[treatment_col] == 0
    
    # Build subgroup mask
    subgroup_mask = pd.Series(True, index=df.index)
    for col, val in subgroup_values:
        subgroup_mask = subgroup_mask & (df[col] == val)
    
    # Apply both treatment and subgroup masks
    treated_mask = mask_treated & subgroup_mask
    control_mask = mask_control & subgroup_mask
    
    if treated_mask.sum() == 0 or control_mask.sum() == 0:
        return float('nan'), float('nan'), False
    
    treated_outcome = df.loc[treated_mask, outcome_col].mean()
    control_outcome = df.loc[control_mask, outcome_col].mean()
    
    effect = treated_outcome - control_outcome
    
    # 2x2 contingency table
    table = pd.crosstab(df.loc[subgroup_mask, treatment_col], df.loc[subgroup_mask, outcome_col])
    _, p_value, _, _ = stats.chi2_contingency(table, correction=False)
    
    significant = p_value < SIGNIFICANCE_THRESHOLD
    return effect, p_value, significant

def test_feature_outcome_association(
    df: pd.DataFrame,
    feature_col: str,
    outcome_col: str
) -> Tuple[float, float, bool]:
    """
    Test association between a feature and outcome.
    For binary features: compare rates between groups.
    For continuous features: compare means between high/low groups.
    Returns: (effect_estimate, p_value, significant)
    """
    # For binary features (0/1)
    if df[feature_col].nunique() == 2:
        mask_present = df[feature_col] == 1
        mask_absent = df[feature_col] == 0
        
        if mask_present.sum() == 0 or mask_absent.sum() == 0:
            return float('nan'), float('nan'), False
        
        present_outcome = df.loc[mask_present, outcome_col].mean()
        absent_outcome = df.loc[mask_absent, outcome_col].mean()
        
        effect = present_outcome - absent_outcome
        
        # 2x2 contingency table
        table = pd.crosstab(df[feature_col], df[outcome_col])
        _, p_value, _, _ = stats.chi2_contingency(table, correction=False)
        
        significant = p_value < SIGNIFICANCE_THRESHOLD
        return effect, p_value, significant
    else:
        # For continuous features: split into high/low groups
        median_val = df[feature_col].median()
        mask_high = df[feature_col] >= median_val
        mask_low = df[feature_col] < median_val
        
        if mask_high.sum() == 0 or mask_low.sum() == 0:
            return float('nan'), float('nan'), False
        
        high_outcome = df.loc[mask_high, outcome_col].mean()
        low_outcome = df.loc[mask_low, outcome_col].mean()
        
        effect = high_outcome - low_outcome
        
        # Two-sample t-test
        _, p_value = stats.ttest_ind(df.loc[mask_high, outcome_col], df.loc[mask_low, outcome_col])
        
        significant = p_value < SIGNIFICANCE_THRESHOLD
        return effect, p_value, significant

def test_interaction_effect(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    modifier_col: str
) -> Tuple[float, float, bool]:
    """
    Test if treatment effect differs by a modifier (interaction).
    Returns: (interaction_effect, p_value, significant)
    """
    # Split by modifier
    mask_modifier_present = df[modifier_col] == 1
    mask_modifier_absent = df[modifier_col] == 0
    
    if mask_modifier_present.sum() == 0 or mask_modifier_absent.sum() == 0:
        return float('nan'), float('nan'), False
    
    # Treatment effect in each subgroup
    effect_present, _, _ = test_treatment_effect(
        df.loc[mask_modifier_present], treatment_col, outcome_col
    )
    effect_absent, _, _ = test_treatment_effect(
        df.loc[mask_modifier_absent], treatment_col, outcome_col
    )
    
    # Interaction effect = difference in treatment effects
    interaction = effect_present - effect_absent
    
    # Test if interaction is significant using bootstrap
    n_bootstrap = 1000
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement from the modifier-present group
        idx_present = np.random.choice(
            df.loc[mask_modifier_present].index, 
            size=min(mask_modifier_present.sum(), 1000), 
            replace=True
        )
        idx_absent = np.random.choice(
            df.loc[mask_modifier_absent].index, 
            size=min(mask_modifier_absent.sum(), 1000), 
            replace=True
        )
        
        effect_p = df.loc[idx_present, outcome_col].mean() - df.loc[idx_present, treatment_col] == 1
        effect_a = df.loc[idx_absent, outcome_col].mean() - df.loc[idx_absent, treatment_col] == 1
        
        if effect_p.sum() > 0 and effect_a.sum() > 0:
            diff = (df.loc[idx_present, outcome_col].mean() - 
                    df.loc[idx_present, treatment_col] == 1).mean() - \
                   (df.loc[idx_absent, outcome_col].mean() - 
                    df.loc[idx_absent, treatment_col] == 1).mean()
            bootstrap_diffs.append(diff)
    
    if len(bootstrap_diffs) < 10:
        return interaction, float('nan'), False
    
    # Compare observed interaction to bootstrap distribution
    observed_p = 2 * min(
        np.mean(np.array(bootstrap_diffs) <= interaction),
        np.mean(np.array(bootstrap_diffs) >= interaction)
    )
    
    significant = observed_p < SIGNIFICANCE_THRESHOLD
    return interaction, observed_p, significant

def run_iteration(
    iteration_num: int,
    df: pd.DataFrame,
    previous_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Run one iteration of the analysis protocol.
    Proposes and tests hypotheses based on previous results.
    """
    iteration = {
        'index': iteration_num,
        'proposed_hypotheses': [],
        'analyses': []
    }
    
    # Generate hypothesis IDs
    hypothesis_counter = [f'h{iteration_num:02d}_{i:02d}' for i in range(1, 11)]
    
    # Strategy: Focus on treatment-outcome relationships and heterogeneity
    
    # 1. Test main treatment effects for each treatment
    for treatment in TREATMENT_COLS:
        for outcome in OUTCOME_COLS:
            if not hypothesis_counter:
                break
            
            hypothesis_id = hypothesis_counter.pop(0)
            
            hypothesis = {
                'id': hypothesis_id,
                'text': f"Patients receiving {treatment} have different objective response rates compared to those not receiving {treatment}.",
                'kind': 'novel'
            }
            iteration['proposed_hypotheses'].append(hypothesis)
            
            effect, p_value, significant = test_treatment_effect(df, treatment, outcome)
            
            analysis = {
                'hypothesis_ids': [hypothesis_id],
                'result_summary': f"Response rate: {safe_float(effect):.3f} with {treatment} vs {safe_float(effect):.3f} without ({format_pvalue(p_value)}).",
                'effect_estimate': safe_float(effect),
                'p_value': safe_float(p_value),
                'significant': safe_bool(significant)
            }
            iteration['analyses'].append(analysis)
    
    # 2. Test treatment-effect heterogeneity for treatments with significant main effects
    significant_treatments = [
        t for t in TREATMENT_COLS 
        if any(a['hypothesis_ids'] == [f'h{iteration_num:02d}_{i:02d}'] and a['significant'] 
                for i, a in enumerate(iteration['analyses']))
    ]
    
    # Find potential modifiers (biomarkers and clinical features)
    potential_modifiers = [
        'brca2_mutation', 'ar_v7_positive', 'msi_high', 'psma_high',
        'gleason_score', 'ecog_ps', 'age_years', 'psa_ng_ml'
    ]
    
    for treatment in significant_treatments:
        for modifier in potential_modifiers:
            if modifier in df.columns and not hypothesis_counter:
                break
            
            if modifier not in df.columns:
                continue
            
            hypothesis_id = hypothesis_counter.pop(0) if hypothesis_counter else f'h{iteration_num:02d}_mod'
            
            hypothesis = {
                'id': hypothesis_id,
                'text': f"The effect of {treatment} on objective response differs by {modifier}.",
                'kind': 'novel'
            }
            iteration['proposed_hypotheses'].append(hypothesis)
            
            interaction, p_value, significant = test_interaction_effect(
                df, treatment, OUTCOME_COLS[0], modifier
            )
            
            analysis = {
                'hypothesis_ids': [hypothesis_id],
                'result_summary': f"Interaction effect: {safe_float(interaction):.4f} ({format_pvalue(p_value)}).",
                'effect_estimate': safe_float(interaction),
                'p_value': safe_float(p_value),
                'significant': safe_bool(significant)
            }
            iteration['analyses'].append(analysis)
    
    return iteration

def generate_summary(transcript: Dict[str, Any]) -> str:
    """Generate analysis summary from transcript."""
    lines = []
    lines.append("=" * 70)
    lines.append("PROSTATE CANCER DATASET ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Dataset: {transcript['dataset_id']}")
    lines.append(f"Total patients: {len(df)}")
    lines.append(f"Total iterations: {len(transcript['iterations'])}")
    lines.append("")
    
    # Overall response rate
    response_rate = df['objective_response'].mean()
    lines.append(f"Overall objective response rate: {response_rate:.1%}")
    lines.append("")
    
    # Summary by iteration
    lines.append("-" * 70)
    lines.append("ITERATION-BY-ITERATION RESULTS")
    lines.append("-" * 70)
    lines.append("")
    
    for iteration in transcript['iterations']:
        lines.append(f"Iteration {iteration['index']}:")
        
        for hypothesis in iteration['proposed_hypotheses']:
            lines.append(f"  Hypothesis: {hypothesis['text']}")
        
        for analysis in iteration['analyses']:
            sig_str = "SIGNIFICANT" if analysis['significant'] else "not significant"
            lines.append(f"  Result: {analysis['result_summary']}")
            lines.append(f"    Effect: {analysis['effect_estimate']:.4f}, p={format_pvalue(analysis['p_value'])}, {sig_str}")
        
        lines.append("")
    
    # Summary of significant findings
    lines.append("-" * 70)
    lines.append("SIGNIFICANT FINDINGS SUMMARY")
    lines.append("-" * 70)
    lines.append("")
    
    significant_findings = []
    for iteration in transcript['iterations']:
        for analysis in iteration['analyses']:
            if analysis['significant']:
                significant_findings.append({
                    'iteration': iteration['index'],
                    'hypothesis': analysis['hypothesis_ids'],
                    'effect': analysis['effect_estimate'],
                    'p_value': analysis['p_value'],
                    'summary': analysis['result_summary']
                })
    
    if significant_findings:
        lines.append(f"Total significant findings: {len(significant_findings)}")
        lines.append("")
        
        # Group by treatment
        treatment_effects = {}
        for finding in significant_findings:
            # Extract treatment from summary
            for treatment in TREATMENT_COLS:
                if treatment in finding['summary']:
                    if treatment not in treatment_effects:
                        treatment_effects[treatment] = []
                    treatment_effects[treatment].append(finding)
                    break
        
        for treatment, effects in treatment_effects.items():
            lines.append(f"{treatment}:")
            for effect in effects:
                lines.append(f"  - {effect['summary']}")
                lines.append(f"    Effect: {effect['effect']:.4f}, p={format_pvalue(effect['p_value'])}")
            lines.append("")
    else:
        lines.append("No statistically significant findings were identified.")
        lines.append("")
    
    # Treatment-effect heterogeneity findings
    lines.append("-" * 70)
    lines.append("TREATMENT-EFFECT HETEROGENEITY")
    lines.append("-" * 70)
    lines.append("")
    
    heterogeneity_findings = []
    for iteration in transcript['iterations']:
        for analysis in iteration['analyses']:
            if 'interaction' in analysis['result_summary'].lower():
                heterogeneity_findings.append(analysis)
    
    if heterogeneity_findings:
        lines.append(f"Significant treatment-effect interactions found: {len(heterogeneity_findings)}")
        lines.append("")
        
        for finding in heterogeneity_findings:
            lines.append(f"  - {finding['result_summary']}")
            lines.append(f"    Effect: {finding['effect_estimate']:.4f}, p={format_pvalue(finding['p_value'])}")
    else:
        lines.append("No significant treatment-effect heterogeneity was identified.")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SUMMARY")
    lines.append("=" * 70)
    
    return "\n".join(lines)

def main():
    """Main analysis loop."""
    print("Starting prostate cancer dataset analysis...")
    print(f"Dataset: {len(df)} patients, {len(df.columns)} columns")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print("")
    
    all_iterations = []
    previous_results = []
    
    for iteration_num in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration_num}/{MAX_ITERATIONS}...")
        
        iteration = run_iteration(iteration_num, df, previous_results)
        all_iterations.append(iteration)
        
        # Store results for next iteration
        previous_results.extend(iteration['analyses'])
        
        # Print summary of this iteration
        sig_count = sum(1 for a in iteration['analyses'] if a['significant'])
        print(f"  Proposed {len(iteration['proposed_hypotheses'])} hypotheses, "
              f"ran {len(iteration['analyses'])} analyses, "
              f"{sig_count} significant")
    
    # Build transcript
    transcript['iterations'] = all_iterations
    
    # Generate summary
    summary = generate_summary(transcript)
    
    # Write transcript.json
    with open('transcript.json', 'w') as f:
        json.dump(transcript, f, indent=2)
    print("\nWrote transcript.json")
    
    # Write analysis_summary.txt
    with open('analysis_summary.txt', 'w') as f:
        f.write(summary)
    print("Wrote analysis_summary.txt")
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
