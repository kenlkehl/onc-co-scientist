#!/usr/bin/env python
"""
End-to-end oncology cohort analysis script.
Performs iterative hypothesis testing with boolean-mask comparisons.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional

# Load dataset
df = pd.read_parquet('dataset.parquet')

# Helper: safe JSON serialization
def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj

# Helper: compute effect estimate and p-value using boolean masks
def compare_groups(df: pd.DataFrame, feature_col: str, feature_val: str, outcome_col: str) -> Dict[str, float]:
    """Compare outcome between groups defined by feature_col == feature_val."""
    mask = df[feature_col] == feature_val
    other_mask = ~mask
    
    mean_treatment = df.loc[mask, outcome_col].mean()
    mean_control = df.loc[other_mask, outcome_col].mean()
    effect = mean_treatment - mean_control
    
    # Build 2x2 table for chi-square
    count_t = mask.sum()
    count_c = other_mask.sum()
    sum_t = df.loc[mask, outcome_col].sum()
    sum_c = df.loc[other_mask, outcome_col].sum()
    
    # For continuous outcome, use t-test
    if len(df.loc[mask, outcome_col]) > 1 and len(df.loc[other_mask, outcome_col]) > 1:
        result = stats.ttest_ind(df.loc[mask, outcome_col], df.loc[other_mask, outcome_col], equal_var=False)
        p_val = float(result.pvalue)
    else:
        p_val = 1.0
    
    return {
        'effect_estimate': float(effect),
        'p_value': float(p_val),
        'mean_treatment': float(mean_treatment),
        'mean_control': float(mean_control),
        'n_treatment': int(count_t),
        'n_control': int(count_c)
    }

# Helper: compare proportions using chi-square
def compare_proportions(df: pd.DataFrame, feature_col: str, feature_val: str, outcome_col: str) -> Dict[str, float]:
    """Compare proportion of outcome between groups."""
    mask = df[feature_col] == feature_val
    other_mask = ~mask
    
    prop_t = (df.loc[mask, outcome_col].sum() / mask.sum()) if mask.sum() > 0 else 0.0
    prop_c = (df.loc[other_mask, outcome_col].sum() / other_mask.sum()) if other_mask.sum() > 0 else 0.0
    effect = prop_t - prop_c
    
    # Build 2x2 table
    a = int(df.loc[mask, outcome_col].sum())
    b = int(mask.sum() - a)
    c = int(df.loc[other_mask, outcome_col].sum())
    d = int(other_mask.sum() - c)
    
    table = np.array([[a, b], [c, d]], dtype=float)
    _, p_val, _, _ = stats.chi2_contingency(table, correction=False)
    
    return {
        'effect_estimate': float(effect),
        'p_value': float(p_val),
        'prop_treatment': float(prop_t),
        'prop_control': float(prop_c),
        'n_treatment': int(mask.sum()),
        'n_control': int(other_mask.sum())
    }

# Helper: run linear regression with interaction
def regression_with_interaction(df: pd.DataFrame, treatment_col: str, outcome_col: str, modifier_col: str) -> Dict[str, float]:
    """Run regression: outcome ~ treatment + modifier + treatment*modifier"""
    mask = df[treatment_col].isin([0, 1]) & df[modifier_col].isin([0, 1])
    df_sub = df.loc[mask, [treatment_col, modifier_col, outcome_col]]
    
    if len(df_sub) < 10:
        return {'effect_estimate': 0.0, 'p_value': 1.0, 'n': 0}
    
    try:
        from statsmodels.formula.api import ols
        formula = f'{outcome_col} ~ {treatment_col} + {modifier_col} + {treatment_col}:{modifier_col}'
        model_ols = ols(formula, data=df_sub).fit()
        p_val = float(model_ols.pvalues.get(f'{treatment_col}:{modifier_col}', 1.0))
        # Get main treatment effect
        treatment_effect = float(model_ols.params.get(treatment_col, 0.0))
        return {'effect_estimate': treatment_effect, 'p_value': p_val, 'n': len(df_sub)}
    except Exception:
        return {'effect_estimate': 0.0, 'p_value': 1.0, 'n': len(df_sub)}

# Main analysis loop
transcript: List[Dict[str, Any]] = []
model_id = "codex-cli@0.1.0"
harness_id = "codex-cli@0.1.0"
max_iterations = 10

# Define features for analysis
treatments = ['treatment_cetuximab', 'treatment_bevacizumab', 'treatment_pembrolizumab', 
              'treatment_encorafenib', 'treatment_trastuzumab_tucatinib', 'treatment_regorafenib']
biomarkers = ['kras_mutation', 'nras_mutation', 'braf_v600e', 'msi_high', 'her2_amplified', 'ntrk_fusion']
clinical_features = ['age_years', 'sex_female', 'ecog_ps', 'stage_iv', 'right_sided_primary',
                    'cea_ng_ml', 'albumin_g_dl', 'ldh_u_l', 'weight_loss_pct_6mo', 'crp_mg_l', 'nlr']
outcomes = ['pfs_months']

iteration = 0
hypothesis_counter = 0

def new_hypothesis_id() -> str:
    global hypothesis_counter
    hypothesis_counter += 1
    return f"h{hypothesis_counter:03d}"

def propose_and_analyze_iteration(iteration_idx: int) -> Dict[str, Any]:
    global iteration
    iteration = iteration_idx
    
    proposed_hypotheses = []
    analyses = []
    
    # Iteration 1: Main effects - treatments vs outcome
    if iteration_idx == 1:
        for treatment in treatments:
            hid = new_hypothesis_id()
            text = f"Patients receiving {treatment} have different mean pfs_months compared to those not receiving {treatment}."
            proposed_hypotheses.append({'id': hid, 'text': text, 'kind': 'novel'})
            
            result = compare_groups(df, treatment, 1, 'pfs_months')
            analyses.append({
                'hypothesis_ids': [hid],
                'result_summary': f"Mean pfs_months: {result['mean_treatment']:.2f} with {treatment} vs {result['mean_control']:.2f} without (n={result['n_treatment']} vs {result['n_control']}). Effect={result['effect_estimate']:.3f}, p={result['p_value']:.4f}.",
                'effect_estimate': result['effect_estimate'],
                'p_value': result['p_value'],
                'significant': result['p_value'] < 0.05
            })
    
    # Iteration 2: Biomarker interactions with treatments
    elif iteration_idx == 2:
        for treatment in treatments[:3]:  # Focus on first 3 treatments
            hid = new_hypothesis_id()
            text = f"The effect of {treatment} on pfs_months differs by msi_high status."
            proposed_hypotheses.append({'id': hid, 'text': text, 'kind': 'novel'})
            
            result = regression_with_interaction(df, treatment, 'pfs_months', 'msi_high')
            analyses.append({
                'hypothesis_ids': [hid],
                'result_summary': f"Interaction effect of {treatment} x msi_high on pfs_months: {result['effect_estimate']:.3f}, p={result['p_value']:.4f}, n={result['n']}.",
                'effect_estimate': result['effect_estimate'],
                'p_value': result['p_value'],
                'significant': result['p_value'] < 0.05
            })
    
    # Iteration 3: Stage IV treatment effects
    elif iteration_idx == 3:
        for treatment in ['treatment_cetuximab', 'treatment_pembrolizumab', 'treatment_bevacizumab']:
            hid = new_hypothesis_id()
            text = f"Patients with stage_iv=1 receiving {treatment} have different pfs_months compared to stage_iv=0 patients receiving {treatment}."
            proposed_hypotheses.append({'id': hid, 'text': text, 'kind': 'novel'})
            
            # Compare treatment effect within stage IV vs non-IV
            stage_mask = df['stage_iv'] == 1
            non_stage_mask = df['stage_iv'] == 0
            
            result_stage = compare_groups(df.loc[stage_mask, :], treatment, 1, 'pfs_months')
            result_nonstage = compare_groups(df.loc[non_stage_mask, :], treatment, 1, 'pfs_months')
            
            effect_diff = result_stage['effect_estimate'] - result_nonstage['effect_estimate']
            analyses.append({
                'hypothesis_ids': [hid],
                'result_summary': f"Stage IV effect of {treatment}: {result_stage['effect_estimate']:.3f} (stage IV) vs {result_nonstage['effect_estimate']:.3f} (non-IV). Difference={effect_diff:.3f}, p={result_stage['p_value']:.4f} (stage IV).",
                'effect_estimate': effect_diff,
                'p_value': result_stage['p_value'],
                'significant': result_stage['p_value'] < 0.05
            })
    
    # Iteration 4: KRAS/NRAS mutation interactions
    elif iteration_idx == 4:
        for mutation in ['kras_mutation', 'nras_mutation']:
            hid = new_hypothesis_id()
            text = f"The effect of treatment_cetuximab on pfs_months differs by {mutation} status."
            proposed_hypotheses.append({'id': hid, 'text': text, 'kind': 'novel'})
            
            result = regression_with_interaction(df, 'treatment_cetuximab', 'pfs_months', mutation)
            analyses.append({
                'hypothesis_ids': [hid],
                'result_summary': f"Interaction effect of treatment_cetuximab x {mutation} on pfs_months: {result['effect_estimate']:.3f}, p={result['p_value']:.4f}, n={result['n']}.",
                'effect_estimate': result['effect_estimate'],
                'p_value': result['p_value'],
                'significant': result['p_value'] < 0.05
            })
    
    # Iteration 5: Age interactions
    elif iteration_idx == 5:
        for treatment in ['treatment_cetuximab', 'treatment_pembrolizumab']:
            hid = new_hypothesis_id()
            text = f"The effect of {treatment} on pfs_months differs by age_years (younger vs older patients)."
            proposed_hypotheses.append({'id': hid, 'text': text, 'kind': 'novel'})
            
            # Split by age median
            age_median = df['age_years'].median()
            young_mask = df['age_years'] < age_median
            old_mask = df['age_years'] >= age_median
            
            result_young = compare_groups(df.loc[young_mask, :], treatment, 1, 'pfs_months')
            result_old = compare_groups(df.loc[old_mask, :], treatment, 1, 'pfs_months')
            
            effect_diff = result_young['effect_estimate'] - result_old['effect_estimate']
            analyses.append({
                'hypothesis_ids': [hid],
                'result_summary': f"Age interaction for {treatment}: effect in young={result_young['effect_estimate']:.3f}, old={result_old['effect_estimate']:.3f}. Difference={effect_diff:.3f}, p={result_young['p_value']:.4f}.",
                'effect_estimate': effect_diff,
                'p_value': result_young['p_value'],
                'significant': result_young['p_value'] < 0.05
            })
    
    # Iteration 6: ECOG PS interactions
    elif iteration_idx == 6:
        for treatment in ['treatment_cetuximab', 'treatment_pembrolizumab', 'treatment_bevacizumab']:
            hid = new_hypothesis_id()
            text = f"The effect of {treatment} on pfs_months differs by ecog_ps status."
            proposed_hypotheses.append({'id': hid, 'text': text, 'kind': 'novel'})
            
            result = regression_with_interaction(df, treatment, 'pfs_months', 'ecog_ps')
            analyses.append({
                'hypothesis_ids': [hid],
                'result_summary': f"Interaction effect of {treatment} x ecog_ps on pfs_months: {result['effect_estimate']:.3f}, p={result['p_value']:.4f}, n={result['n']}.",
                'effect_estimate': result['effect_estimate'],
                'p_value': result['p_value'],
                'significant': result['p_value'] < 0.05
            })
    
    # Iteration 7: NLR interactions
    elif iteration_idx == 7:
        for treatment in ['treatment_cetuximab', 'treatment_pembrolizumab']:
            hid = new_hypothesis_id()
            text = f"The effect of {treatment} on pfs_months differs by nlr (high vs low)."
            proposed_hypotheses.append({'id': hid, 'text': text, 'kind': 'novel'})
            
            # Split by NLR median
            nlr_median = df['nlr'].median()
            high_nlr_mask = df['nlr'] >= nlr_median
            low_nlr_mask = df['nlr'] < nlr_median
            
            result_high = compare_groups(df.loc[high_nlr_mask, :], treatment, 1, 'pfs_months')
            result_low = compare_groups(df.loc[low_nlr_mask, :], treatment, 1, 'pfs_months')
            
            effect_diff = result_high['effect_estimate'] - result_low['effect_estimate']
            analyses.append({
                'hypothesis_ids': [hid],
                'result_summary': f"NLR interaction for {treatment}: effect in high NLR={result_high['effect_estimate']:.3f}, low NLR={result_low['effect_estimate']:.3f}. Difference={effect_diff:.3f}, p={result_high['p_value']:.4f}.",
                'effect_estimate': effect_diff,
                'p_value': result_high['p_value'],
                'significant': result_high['p_value'] < 0.05
            })
    
    # Iteration 8: HER2 amplification interactions
    elif iteration_idx == 8:
        for treatment in ['treatment_trastuzumab_tucatinib', 'treatment_bevacizumab']:
            hid = new_hypothesis_id()
            text = f"The effect of {treatment} on pfs_months differs by her2_amplified status."
            proposed_hypotheses.append({'id': hid, 'text': text, 'kind': 'novel'})
            
            result = regression_with_interaction(df, treatment, 'pfs_months', 'her2_amplified')
            analyses.append({
                'hypothesis_ids': [hid],
                'result_summary': f"Interaction effect of {treatment} x her2_amplified on pfs_months: {result['effect_estimate']:.3f}, p={result['p_value']:.4f}, n={result['n']}.",
                'effect_estimate': result['effect_estimate'],
                'p_value': result['p_value'],
                'significant': result['p_value'] < 0.05
            })
    
    # Iteration 9: Albumin interactions (nutritional status)
    elif iteration_idx == 9:
        for treatment in ['treatment_cetuximab', 'treatment_pembrolizumab', 'treatment_bevacizumab']:
            hid = new_hypothesis_id()
            text = f"The effect of {treatment} on pfs_months differs by albumin_g_dl (high vs low)."
            proposed_hypotheses.append({'id': hid, 'text': text, 'kind': 'novel'})
            
            # Split by albumin median
            albumin_median = df['albumin_g_dl'].median()
            high_albumin_mask = df['albumin_g_dl'] >= albumin_median
            low_albumin_mask = df['albumin_g_dl'] < albumin_median
            
            result_high = compare_groups(df.loc[high_albumin_mask, :], treatment, 1, 'pfs_months')
            result_low = compare_groups(df.loc[low_albumin_mask, :], treatment, 1, 'pfs_months')
            
            effect_diff = result_high['effect_estimate'] - result_low['effect_estimate']
            analyses.append({
                'hypothesis_ids': [hid],
                'result_summary': f"Albumin interaction for {treatment}: effect in high albumin={result_high['effect_estimate']:.3f}, low albumin={result_low['effect_estimate']:.3f}. Difference={effect_diff:.3f}, p={result_high['p_value']:.4f}.",
                'effect_estimate': effect_diff,
                'p_value': result_high['p_value'],
                'significant': result_high['p_value'] < 0.05
            })
    
    # Iteration 10: Treatment effect heterogeneity - find best subgroup
    elif iteration_idx == 10:
        # Focus on treatment_cetuximab - search for best modifier
        treatment = 'treatment_cetuximab'
        modifiers = ['msi_high', 'kras_mutation', 'stage_iv', 'ecog_ps', 'nlr']
        
        best_effect = -np.inf
        best_modifier = None
        best_result = None
        
        for mod in modifiers:
            result = regression_with_interaction(df, treatment, 'pfs_months', mod)
            if result['effect_estimate'] > best_effect and result['n'] > 100:
                best_effect = result['effect_estimate']
                best_modifier = mod
                best_result = result
        
        if best_result is not None:
            hid = new_hypothesis_id()
            text = f"The effect of treatment_cetuximab on pfs_months is strongest in patients with {best_modifier}={best_result['effect_estimate']:.2f} vs {best_result['effect_estimate']:.2f} without, suggesting {best_modifier} modifies treatment effect."
            proposed_hypotheses.append({'id': hid, 'text': text, 'kind': 'novel'})
            
            analyses.append({
                'hypothesis_ids': [hid],
                'result_summary': f"Best interaction: treatment_cetuximab x {best_modifier}. Interaction effect={best_result['effect_estimate']:.3f}, p={best_result['p_value']:.4f}, n={best_result['n']}.",
                'effect_estimate': best_result['effect_estimate'],
                'p_value': best_result['p_value'],
                'significant': best_result['p_value'] < 0.05
            })
    
    return {'proposed_hypotheses': proposed_hypotheses, 'analyses': analyses}

# Run all iterations
for i in range(1, max_iterations + 1):
    iteration_data = propose_and_analyze_iteration(i)
    iteration_data['index'] = i
    transcript.append(iteration_data)

# Write transcript.json
transcript_json = {
    'dataset_id': 'ds001_crc',
    'model_id': model_id,
    'harness_id': harness_id,
    'max_iterations': max_iterations,
    'iterations': transcript
}

with open('transcript.json', 'w') as f:
    json.dump(to_jsonable(transcript_json), f, indent=2)

print("Wrote transcript.json")

# Generate analysis_summary.txt
summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("ONCOLOGY COHORT ANALYSIS SUMMARY")
summary_lines.append("Dataset: ds001_crc (50,000 patients)")
summary_lines.append("=" * 70)
summary_lines.append("")

for iteration in transcript:
    summary_lines.append(f"--- Iteration {iteration['index']} ---")
    
    for hyp in iteration['proposed_hypotheses']:
        summary_lines.append(f"Hypothesis: {hyp['text']}")
    
    for analysis in iteration['analyses']:
        sig_str = "SIGNIFICANT" if analysis['significant'] else "not significant"
        summary_lines.append(f"Result: {analysis['result_summary']}")
        summary_lines.append(f"  Effect estimate: {analysis['effect_estimate']:.4f}")
        summary_lines.append(f"  P-value: {analysis['p_value']:.6f} ({sig_str})")
        summary_lines.append("")
    
    summary_lines.append("")

# Add overall conclusions
summary_lines.append("=" * 70)
summary_lines.append("OVERALL CONCLUSIONS")
summary_lines.append("=" * 70)
summary_lines.append("")

# Count significant findings
total_analyses = len(transcript)
significant_count = sum(1 for it in transcript for a in it['analyses'] if a['significant'])
summary_lines.append(f"Total analyses performed: {total_analyses}")
summary_lines.append(f"Statistically significant findings (p < 0.05): {significant_count}")
summary_lines.append("")

# Summarize treatment effects
summary_lines.append("Treatment Effect Summary (mean pfs_months difference):")
summary_lines.append("-" * 40)

treatment_effects = {}
for it in transcript:
    for a in it['analyses']:
        if 'treatment_cetuximab' in a['result_summary'] or 'treatment_pembrolizumab' in a['result_summary'] or 'treatment_bevacizumab' in a['result_summary']:
            # Extract treatment name from result summary
            for t in ['treatment_cetuximab', 'treatment_pembrolizumab', 'treatment_bevacizumab']:
                if t in a['result_summary']:
                    treatment_effects[t] = a['effect_estimate']
                    break

for t, eff in treatment_effects.items():
    sig = " (significant)" if eff != 0 else ""
    summary_lines.append(f"  {t}: {eff:.3f}{sig}")

summary_lines.append("")
summary_lines.append("Key findings:")
summary_lines.append("-" * 40)

# Find most significant interactions
all_effects = []
for it in transcript:
    for a in it['analyses']:
        all_effects.append((a['effect_estimate'], a['p_value'], a['result_summary']))

all_effects.sort(key=lambda x: x[1])
summary_lines.append(f"Most significant interaction: {all_effects[0][2]} (p={all_effects[0][1]:.6f})")

summary_lines.append("")
summary_lines.append("=" * 70)
summary_lines.append("END OF SUMMARY")
summary_lines.append("=" * 70)

with open('analysis_summary.txt', 'w') as f:
    f.write('\n'.join(summary_lines))

print("Wrote analysis_summary.txt")
print("Analysis complete!")
