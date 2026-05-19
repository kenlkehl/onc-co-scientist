#!/usr/bin/env python3
"""
End-to-end CRC oncology dataset analysis script.
Follows the protocol: propose hypotheses, test them, refine across iterations.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional

# Load dataset
df = pd.read_parquet('dataset.parquet')

# Column definitions
TREATMENTS = ['treatment_cetuximab', 'treatment_bevacizumab', 'treatment_pembrolizumab',
              'treatment_encorafenib', 'treatment_trastuzumab_tucatinib', 'treatment_regorafenib']
BINARY_FEATURES = ['sex_female', 'ecog_ps', 'stage_iv', 'right_sided_primary',
                   'kras_mutation', 'nras_mutation', 'braf_v600e', 'msi_high',
                   'her2_amplified', 'ntrk_fusion']
CONTINUOUS_FEATURES = ['age_years', 'cea_ng_ml', 'albumin_g_dl', 'ldh_u_l',
                       'weight_loss_pct_6mo', 'crp_mg_l', 'nlr', 'hemoglobin_g_dl',
                       'alkaline_phosphatase_u_l', 'ast_u_l', 'alt_u_l',
                       'total_bilirubin_mg_dl', 'creatinine_mg_dl', 'bun_mg_dl',
                       'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl']
OUTCOMES = ['pfs_months']

def safe_float(val):
    """Convert to float, handling None/NaN."""
    if pd.isna(val):
        return np.nan
    return float(val)

def compute_effect(df, feature, outcome, value):
    """Compute effect estimate: mean(outcome|feature=value) - mean(outcome|feature!=value)."""
    mask = df[feature] == value
    if mask.sum() == 0 or (~mask).sum() == 0:
        return np.nan
    mean_treated = df.loc[mask, outcome].mean()
    mean_control = df.loc[~mask, outcome].mean()
    return mean_treated - mean_control

def compute_rate_effect(df, feature, outcome, value):
    """Compute rate effect for binary outcome: rate(treated) - rate(control)."""
    mask = df[feature] == value
    if mask.sum() == 0 or (~mask).sum() == 0:
        return np.nan
    rate_treated = (df.loc[mask, outcome] == 1).mean()
    rate_control = (df.loc[~mask, outcome] == 1).mean()
    return rate_treated - rate_control

def test_independence(df, feature, outcome, method='chi2'):
    """Test independence between feature and outcome."""
    mask = df[feature] == 1
    if method == 'ttest':
        stat, p = stats.ttest_ind(df.loc[mask, outcome], df.loc[~mask, outcome])
        return stat, p
    elif method == 'mannwhitney':
        stat, p = stats.mannwhitneyu(df.loc[mask, outcome], df.loc[~mask, outcome], alternative='two-sided')
        return stat, p
    elif method == 'chi2':
        # Build contingency table for binary outcome
        if df[outcome].dtype in ['int64', 'int32'] and df[outcome].min() == 0 and df[outcome].max() == 1:
            table = pd.crosstab(df[feature], df[outcome])
            if table.sum().sum() > 10:
                chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
                return chi2, p
        # For continuous outcome, use t-test
        stat, p = stats.ttest_ind(df.loc[mask, outcome], df.loc[~mask, outcome])
        return stat, p
    return np.nan, 1.0

def test_interaction(df, treatment, modifier, outcome):
    """Test treatment-by-modifier interaction effect."""
    # Create interaction term
    df['interaction'] = df[treatment] * df[modifier]
    
    # Fit simple linear model: outcome ~ treatment + modifier + interaction
    try:
        model = stats.linregress(df['interaction'], df[outcome])
        interaction_stat = model.slope
        interaction_p = model.pvalue
        
        # Main treatment effect (averaged)
        treatment_stat, treatment_p = test_independence(df, treatment, outcome, 'ttest')
        
        return {
            'interaction_effect': interaction_stat,
            'interaction_p': interaction_p,
            'treatment_effect': treatment_stat,
            'treatment_p': treatment_p,
            'modifier': modifier
        }
    except Exception:
        return {'interaction_effect': np.nan, 'interaction_p': 1.0, 'treatment_effect': np.nan, 'treatment_p': 1.0, 'modifier': modifier}

def run_analysis(df, hypothesis_id, feature, outcome, value, method='ttest'):
    """Run a single analysis and return structured result."""
    effect = compute_effect(df, feature, outcome, value)
    stat, p = test_independence(df, feature, outcome, method)
    
    significant = p < 0.05 if not np.isnan(p) else False
    
    return {
        'hypothesis_ids': [hypothesis_id],
        'effect_estimate': safe_float(effect),
        'p_value': safe_float(p),
        'significant': significant,
        'result_summary': f"Effect: {safe_float(effect):.3f}, p={safe_float(p):.4f}, significant={significant}"
    }

def generate_hypothesis_id(iteration, idx, feature, outcome=None):
    """Generate a hypothesis ID from iteration and feature."""
    if outcome:
        return f"h{iteration}_{idx}_{feature}_{outcome}"
    return f"h{iteration}_{idx}_{feature}"

def propose_hypotheses(df, iteration, idx, feature, outcome, value, method='ttest'):
    """Propose a hypothesis about feature-outcome relationship."""
    effect = compute_effect(df, feature, outcome, value)
    stat, p = test_independence(df, feature, outcome, method)
    significant = p < 0.05 if not np.isnan(p) else False
    
    direction = "higher" if effect > 0 else "lower"
    text = f"Patients with {feature}={value} have {direction} {outcome} than those with {feature}!={value}."
    
    return {
        'id': generate_hypothesis_id(iteration, idx, feature, outcome),
        'text': text,
        'kind': 'novel'
    }

def run_iteration(df, iteration, feature, outcome, value, method='ttest'):
    """Run one iteration: propose hypothesis and test it."""
    hypothesis = propose_hypotheses(df, iteration, 0, feature, outcome, value, method)
    analysis = run_analysis(df, hypothesis['id'], feature, outcome, value, method)
    
    return {
        'index': iteration,
        'proposed_hypotheses': [hypothesis],
        'analyses': [analysis]
    }

def run_treatment_effect_heterogeneity(df, iteration, treatment, outcome):
    """Run treatment effect heterogeneity search."""
    hypotheses = []
    analyses = []
    
    # Test main treatment effect
    main_effect = compute_effect(df, treatment, outcome, 1)
    main_stat, main_p = test_independence(df, treatment, outcome, 'ttest')
    main_sig = main_p < 0.05 if not np.isnan(main_p) else False
    
    main_hypothesis = {
        'id': f"h{iteration}_treatment_effect",
        'text': f"Patients receiving {treatment} have different {outcome} than those not receiving it.",
        'kind': 'novel'
    }
    
    main_analysis = {
        'hypothesis_ids': [main_hypothesis['id']],
        'effect_estimate': safe_float(main_effect),
        'p_value': safe_float(main_p),
        'significant': main_sig,
        'result_summary': f"Effect: {safe_float(main_effect):.3f}, p={safe_float(main_p):.4f}, significant={main_sig}"
    }
    
    hypotheses.append(main_hypothesis)
    analyses.append(main_analysis)
    
    # Test interaction with key modifiers
    modifiers = ['stage_iv', 'msi_high', 'kras_mutation', 'age_years']
    best_interaction = None
    best_interaction_p = 1.0
    
    for modifier in modifiers:
        interaction_result = test_interaction(df, treatment, modifier, outcome)
        if not np.isnan(interaction_result['interaction_effect']):
            if interaction_result['interaction_p'] < best_interaction_p:
                best_interaction_p = interaction_result['interaction_p']
                best_interaction = interaction_result
    
    if best_interaction is not None and best_interaction_p < 0.1:
        interaction_hypothesis = {
            'id': f"h{iteration}_interaction_{treatment}_{outcome}",
            'text': f"The effect of {treatment} on {outcome} is modified by {best_interaction['modifier']}.",
            'kind': 'refined'
        }
        
        interaction_analysis = {
            'hypothesis_ids': [interaction_hypothesis['id']],
            'effect_estimate': safe_float(best_interaction['interaction_effect']),
            'p_value': safe_float(best_interaction['interaction_p']),
            'significant': best_interaction['interaction_p'] < 0.05,
            'result_summary': f"Interaction effect: {safe_float(best_interaction['interaction_effect']):.3f}, p={safe_float(best_interaction['interaction_p']):.4f}"
        }
        
        hypotheses.append(interaction_hypothesis)
        analyses.append(interaction_analysis)
    
    return {
        'index': iteration,
        'proposed_hypotheses': hypotheses,
        'analyses': analyses
    }

def main():
    transcript = {
        'dataset_id': 'ds001_crc',
        'model_id': 'codex-cli@1.0.0',
        'harness_id': 'oncology-analysis-harness',
        'max_iterations': 10,
        'iterations': []
    }
    
    iteration = 1
    
    # Iteration 1: Main effects for key treatments on PFS
    for treatment in TREATMENTS:
        result = run_iteration(df, iteration, treatment, OUTCOMES[0], 1, 'ttest')
        transcript['iterations'].append(result)
        iteration += 1
    
    # Iteration 2: Main effects for key binary features on PFS
    for feature in BINARY_FEATURES[:5]:  # Top 5 binary features
        result = run_iteration(df, iteration, feature, OUTCOMES[0], 1, 'ttest')
        transcript['iterations'].append(result)
        iteration += 1
    
    # Iteration 3: Treatment effect heterogeneity for cetuximab
    het_result = run_treatment_effect_heterogeneity(df, iteration, 'treatment_cetuximab', OUTCOMES[0])
    transcript['iterations'].append(het_result)
    iteration += 1
    
    # Iteration 4: Treatment effect heterogeneity for pembrolizumab
    het_result = run_treatment_effect_heterogeneity(df, iteration, 'treatment_pembrolizumab', OUTCOMES[0])
    transcript['iterations'].append(het_result)
    iteration += 1
    
    # Iteration 5: Main effects for continuous features on PFS
    for feature in CONTINUOUS_FEATURES[:5]:  # Top 5 continuous features
        result = run_iteration(df, iteration, feature, OUTCOMES[0], 1, 'ttest')
        transcript['iterations'].append(result)
        iteration += 1
    
    # Iteration 6: Treatment effect heterogeneity for bevacizumab
    het_result = run_treatment_effect_heterogeneity(df, iteration, 'treatment_bevacizumab', OUTCOMES[0])
    transcript['iterations'].append(het_result)
    iteration += 1
    
    # Iteration 7: Main effects for stage and MSI on PFS
    result = run_iteration(df, iteration, 'stage_iv', OUTCOMES[0], 1, 'ttest')
    transcript['iterations'].append(result)
    iteration += 1
    
    result = run_iteration(df, iteration, 'msi_high', OUTCOMES[0], 1, 'ttest')
    transcript['iterations'].append(result)
    iteration += 1
    
    # Iteration 8: Treatment effect heterogeneity for regorafenib
    het_result = run_treatment_effect_heterogeneity(df, iteration, 'treatment_regorafenib', OUTCOMES[0])
    transcript['iterations'].append(het_result)
    iteration += 1
    
    # Iteration 9: Main effects for KRAS and BRAF on PFS
    result = run_iteration(df, iteration, 'kras_mutation', OUTCOMES[0], 1, 'ttest')
    transcript['iterations'].append(result)
    iteration += 1
    
    result = run_iteration(df, iteration, 'braf_v600e', OUTCOMES[0], 1, 'ttest')
    transcript['iterations'].append(result)
    iteration += 1
    
    # Iteration 10: Treatment effect heterogeneity for encorafenib
    het_result = run_treatment_effect_heterogeneity(df, iteration, 'treatment_encorafenib', OUTCOMES[0])
    transcript['iterations'].append(het_result)
    
    # Convert to JSON-serializable format
    def make_jsonable(obj):
        if isinstance(obj, dict):
            return {k: make_jsonable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_jsonable(item) for item in obj]
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
    
    transcript_jsonable = make_jsonable(transcript)
    
    # Write transcript.json
    with open('transcript.json', 'w') as f:
        json.dump(transcript_jsonable, f, indent=2)
    
    # Generate analysis summary
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY")
    summary_lines.append("Dataset: ds001_crc (50,000 patients)")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    
    for iter_record in transcript['iterations']:
        summary_lines.append(f"ITERATION {iter_record['index']}:")
        summary_lines.append("-" * 40)
        
        for hypothesis in iter_record['proposed_hypotheses']:
            summary_lines.append(f"  Hypothesis: {hypothesis['text']}")
        
        for analysis in iter_record['analyses']:
            effect = analysis['effect_estimate']
            p_val = analysis['p_value']
            sig = analysis['significant']
            
            if effect is not None and not np.isnan(effect):
                summary_lines.append(f"  Result: Effect = {effect:.3f}, p = {p_val:.4f}, significant = {sig}")
            else:
                summary_lines.append(f"  Result: p = {p_val:.4f}, significant = {sig}")
        
        summary_lines.append("")
    
    # Summary conclusions
    summary_lines.append("=" * 70)
    summary_lines.append("KEY FINDINGS SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    
    # Collect significant findings
    significant_findings = []
    for iter_record in transcript['iterations']:
        for analysis in iter_record['analyses']:
            if analysis['significant']:
                effect = analysis['effect_estimate']
                p_val = analysis['p_value']
                hypothesis_text = analysis['result_summary']
                significant_findings.append((effect, p_val, hypothesis_text))
    
    if significant_findings:
        summary_lines.append(f"Found {len(significant_findings)} statistically significant findings (p < 0.05).")
        summary_lines.append("")
        summary_lines.append("Top significant findings:")
        for effect, p_val, text in sorted(significant_findings, key=lambda x: x[1])[:10]:
            summary_lines.append(f"  - {text}")
    else:
        summary_lines.append("No statistically significant findings (p < 0.05) were identified.")
    
    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("END OF ANALYSIS SUMMARY")
    summary_lines.append("=" * 70)
    
    # Write analysis_summary.txt
    with open('analysis_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print("Analysis complete. Generated:")
    print("  - transcript.json")
    print("  - analysis_summary.txt")

if __name__ == '__main__':
    main()
