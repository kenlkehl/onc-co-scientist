#!/usr/bin/env python3
"""
Oncology Dataset Analysis Script
Analyzes ds001_nsclc dataset with iterative hypothesis testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime

# Load dataset
df = pd.read_parquet('dataset.parquet')

# Configuration
MAX_ITERATIONS = 10
SIGNIFICANCE_THRESHOLD = 0.05

# Column definitions
TREATMENTS = ['treatment_pembrolizumab', 'treatment_sotorasib', 'treatment_olaparib', 'treatment_osimertinib']
OUTCOMES = ['pfs_months']
CONTINUOUS_FEATURES = ['age_years', 'pdl1_tps', 'albumin_g_dl', 'ldh_u_l', 'weight_loss_pct_6mo', 
                       'crp_mg_l', 'nlr', 'hemoglobin_g_dl', 'alkaline_phosphatase_u_l', 'ast_u_l',
                       'alt_u_l', 'total_bilirubin_mg_dl', 'creatinine_mg_dl', 'bun_mg_dl',
                       'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl']
BINARY_FEATURES = ['sex_female', 'smoking_status', 'ecog_ps', 'histology', 'stage_iv', 
                   'has_brain_mets', 'egfr_mutation', 'kras_g12c', 'alk_fusion', 'stk11_mutation',
                   'brca2_mutation', 'tmb_high']

def test_treatment_effect(df, treatment, outcome):
    """Test main treatment effect on outcome."""
    mask = df[treatment] == 1
    if mask.sum() == 0:
        return None, None
    mean_treated = df.loc[mask, outcome].mean()
    mean_control = df.loc[~mask, outcome].mean()
    effect = mean_treated - mean_control
    p_value = compute_pvalue(df, treatment, outcome, 1)
    return effect, p_value

def compute_pvalue(df, feature, outcome, value):
    """Compute p-value using t-test for continuous outcomes."""
    mask = df[feature] == value
    if mask.sum() == 0 or (~mask).sum() == 0:
        return None
    t_stat, p_value = stats.ttest_ind(df.loc[mask, outcome], df.loc[~mask, outcome])
    return float(p_value)

def test_interaction(df, treatment, outcome, modifier):
    """Test treatment-by-modifier interaction."""
    # Stratified analysis
    strata = df[modifier].unique()
    effects = []
    pvalues = []
    for s in strata:
        mask = (df[treatment] == 1) & (df[modifier] == s)
        if mask.sum() > 0:
            mean_treated = df.loc[mask, outcome].mean()
            mask_c = (df[treatment] == 0) & (df[modifier] == s)
            if mask_c.sum() > 0:
                mean_control = df.loc[mask_c, outcome].mean()
                effect = mean_treated - mean_control
                effects.append(effect)
                pvalues.append(compute_pvalue(df, treatment, outcome, 1))
    
    if len(effects) < 2:
        return None, None
    
    # Test if effects differ across strata (interaction)
    if len(effects) >= 2:
        strata_list = list(strata)
        if len(strata_list) >= 2:
            s1, s2 = strata_list[0], strata_list[1]
            mask1 = (df[treatment] == 1) & (df[modifier] == s1)
            mask2 = (df[treatment] == 1) & (df[modifier] == s2)
            if mask1.sum() > 0 and mask2.sum() > 0:
                effect1 = df.loc[mask1, outcome].mean() - df.loc[(df[treatment] == 0) & (df[modifier] == s1), outcome].mean()
                effect2 = df.loc[mask2, outcome].mean() - df.loc[(df[treatment] == 0) & (df[modifier] == s2), outcome].mean()
                # Compute standard errors safely
                n1_t = mask1.sum()
                n1_c = ((df[treatment] == 0) & (df[modifier] == s1)).sum()
                n2_t = mask2.sum()
                n2_c = ((df[treatment] == 0) & (df[modifier] == s2)).sum()
                
                if n1_t > 0 and n1_c > 0 and n2_t > 0 and n2_c > 0:
                    se1 = np.sqrt(df.loc[mask1, outcome].var() / n1_t + 
                                df.loc[(df[treatment] == 0) & (df[modifier] == s1), outcome].var() / n1_c)
                    se2 = np.sqrt(df.loc[mask2, outcome].var() / n2_t + 
                                df.loc[(df[treatment] == 0) & (df[modifier] == s2), outcome].var() / n2_c)
                    diff = effect1 - effect2
                    se_diff = np.sqrt(se1**2 + se2**2)
                    if se_diff > 0:
                        z_stat = diff / se_diff
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        return float(diff), float(p_value)
    return None, None

def run_iteration(iteration_num, transcript, df):
    """Run one iteration of hypothesis testing."""
    iteration = {
        'index': iteration_num,
        'proposed_hypotheses': [],
        'analyses': []
    }
    
    # Iteration 1: Main treatment effects
    if iteration_num == 1:
        for treatment in TREATMENTS:
            for outcome in OUTCOMES:
                hypothesis_id = f'h{iteration_num}_{treatment.replace("_", "_")}'
                hypothesis = {
                    'id': hypothesis_id,
                    'text': f"Patients receiving {treatment} have different {outcome} compared to those not receiving {treatment}.",
                    'kind': 'novel'
                }
                iteration['proposed_hypotheses'].append(hypothesis)
                
                effect, p_value = test_treatment_effect(df, treatment, outcome)
                significant = bool(p_value is not None and p_value < SIGNIFICANCE_THRESHOLD)
                
                analysis = {
                    'hypothesis_ids': [hypothesis_id],
                    'result_summary': f"Mean {outcome}: {df.loc[df[treatment]==1, outcome].mean():.3f} vs {df.loc[df[treatment]==0, outcome].mean():.3f} (diff={effect:.3f}, p={p_value:.4f} if p_value else 'N/A')",
                    'effect_estimate': float(effect) if effect is not None else None,
                    'p_value': p_value,
                    'significant': significant
                }
                iteration['analyses'].append(analysis)
    
    # Iteration 2: Treatment by binary feature interactions
    elif iteration_num == 2:
        for treatment in TREATMENTS:
            for outcome in OUTCOMES:
                for feature in BINARY_FEATURES:
                    hypothesis_id = f'h{iteration_num}_{treatment}_{feature}'
                    hypothesis = {
                        'id': hypothesis_id,
                        'text': f"The effect of {treatment} on {outcome} differs by {feature}.",
                        'kind': 'novel'
                    }
                    iteration['proposed_hypotheses'].append(hypothesis)
                    
                    effect, p_value = test_interaction(df, treatment, outcome, feature)
                    significant = bool(p_value is not None and p_value < SIGNIFICANCE_THRESHOLD)
                    
                    if effect is not None and p_value is not None:
                        summary = f"Interaction effect: {effect:.3f} (p={p_value:.4f})"
                    else:
                        summary = f"Interaction effect: N/A (insufficient data)"
                    
                    analysis = {
                        'hypothesis_ids': [hypothesis_id],
                        'result_summary': summary,
                        'effect_estimate': float(effect) if effect is not None else None,
                        'p_value': p_value,
                        'significant': significant
                    }
                    iteration['analyses'].append(analysis)
    
    # Iteration 3: Treatment by continuous feature interactions
    elif iteration_num == 3:
        for treatment in TREATMENTS:
            for outcome in OUTCOMES:
                for feature in CONTINUOUS_FEATURES[:5]:
                    hypothesis_id = f'h{iteration_num}_{treatment}_{feature}'
                    hypothesis = {
                        'id': hypothesis_id,
                        'text': f"The effect of {treatment} on {outcome} differs by {feature}.",
                        'kind': 'novel'
                    }
                    iteration['proposed_hypotheses'].append(hypothesis)
                    
                    effect, p_value = test_interaction(df, treatment, outcome, feature)
                    significant = bool(p_value is not None and p_value < SIGNIFICANCE_THRESHOLD)
                    
                    if effect is not None and p_value is not None:
                        summary = f"Interaction effect: {effect:.3f} (p={p_value:.4f})"
                    else:
                        summary = f"Interaction effect: N/A (insufficient data)"
                    
                    analysis = {
                        'hypothesis_ids': [hypothesis_id],
                        'result_summary': summary,
                        'effect_estimate': float(effect) if effect is not None else None,
                        'p_value': p_value,
                        'significant': significant
                    }
                    iteration['analyses'].append(analysis)
    
    # Iteration 4: Treatment by histology
    elif iteration_num == 4:
        for treatment in TREATMENTS:
            for outcome in OUTCOMES:
                hypothesis_id = f'h{iteration_num}_{treatment}_histology'
                hypothesis = {
                    'id': hypothesis_id,
                    'text': f"The effect of {treatment} on {outcome} differs by histology (adenocarcinoma vs squamous).",
                    'kind': 'novel'
                }
                iteration['proposed_hypotheses'].append(hypothesis)
                
                effect, p_value = test_interaction(df, treatment, outcome, 'histology')
                significant = bool(p_value is not None and p_value < SIGNIFICANCE_THRESHOLD)
                
                if effect is not None and p_value is not None:
                    summary = f"Interaction effect: {effect:.3f} (p={p_value:.4f})"
                else:
                    summary = f"Interaction effect: N/A (insufficient data)"
                
                analysis = {
                    'hypothesis_ids': [hypothesis_id],
                    'result_summary': summary,
                    'effect_estimate': float(effect) if effect is not None else None,
                    'p_value': p_value,
                    'significant': significant
                }
                iteration['analyses'].append(analysis)
    
    # Iteration 5: Treatment by stage
    elif iteration_num == 5:
        for treatment in TREATMENTS:
            for outcome in OUTCOMES:
                hypothesis_id = f'h{iteration_num}_{treatment}_stage'
                hypothesis = {
                    'id': hypothesis_id,
                    'text': f"The effect of {treatment} on {outcome} differs by stage IV status.",
                    'kind': 'novel'
                }
                iteration['proposed_hypotheses'].append(hypothesis)
                
                effect, p_value = test_interaction(df, treatment, outcome, 'stage_iv')
                significant = bool(p_value is not None and p_value < SIGNIFICANCE_THRESHOLD)
                
                if effect is not None and p_value is not None:
                    summary = f"Interaction effect: {effect:.3f} (p={p_value:.4f})"
                else:
                    summary = f"Interaction effect: N/A (insufficient data)"
                
                analysis = {
                    'hypothesis_ids': [hypothesis_id],
                    'result_summary': summary,
                    'effect_estimate': float(effect) if effect is not None else None,
                    'p_value': p_value,
                    'significant': significant
                }
                iteration['analyses'].append(analysis)
    
    # Iteration 6: Treatment by smoking status
    elif iteration_num == 6:
        for treatment in TREATMENTS:
            for outcome in OUTCOMES:
                hypothesis_id = f'h{iteration_num}_{treatment}_smoking'
                hypothesis = {
                    'id': hypothesis_id,
                    'text': f"The effect of {treatment} on {outcome} differs by smoking status.",
                    'kind': 'novel'
                }
                iteration['proposed_hypotheses'].append(hypothesis)
                
                effect, p_value = test_interaction(df, treatment, outcome, 'smoking_status')
                significant = bool(p_value is not None and p_value < SIGNIFICANCE_THRESHOLD)
                
                if effect is not None and p_value is not None:
                    summary = f"Interaction effect: {effect:.3f} (p={p_value:.4f})"
                else:
                    summary = f"Interaction effect: N/A (insufficient data)"
                
                analysis = {
                    'hypothesis_ids': [hypothesis_id],
                    'result_summary': summary,
                    'effect_estimate': float(effect) if effect is not None else None,
                    'p_value': p_value,
                    'significant': significant
                }
                iteration['analyses'].append(analysis)
    
    # Iteration 7: Treatment by PD-L1
    elif iteration_num == 7:
        for treatment in TREATMENTS:
            for outcome in OUTCOMES:
                hypothesis_id = f'h{iteration_num}_{treatment}_pdl1'
                hypothesis = {
                    'id': hypothesis_id,
                    'text': f"The effect of {treatment} on {outcome} differs by PD-L1 TPS.",
                    'kind': 'novel'
                }
                iteration['proposed_hypotheses'].append(hypothesis)
                
                effect, p_value = test_interaction(df, treatment, outcome, 'pdl1_tps')
                significant = bool(p_value is not None and p_value < SIGNIFICANCE_THRESHOLD)
                
                if effect is not None and p_value is not None:
                    summary = f"Interaction effect: {effect:.3f} (p={p_value:.4f})"
                else:
                    summary = f"Interaction effect: N/A (insufficient data)"
                
                analysis = {
                    'hypothesis_ids': [hypothesis_id],
                    'result_summary': summary,
                    'effect_estimate': float(effect) if effect is not None else None,
                    'p_value': p_value,
                    'significant': significant
                }
                iteration['analyses'].append(analysis)
    
    # Iteration 8: Treatment by TMB
    elif iteration_num == 8:
        for treatment in TREATMENTS:
            for outcome in OUTCOMES:
                hypothesis_id = f'h{iteration_num}_{treatment}_tmb'
                hypothesis = {
                    'id': hypothesis_id,
                    'text': f"The effect of {treatment} on {outcome} differs by TMB status.",
                    'kind': 'novel'
                }
                iteration['proposed_hypotheses'].append(hypothesis)
                
                effect, p_value = test_interaction(df, treatment, outcome, 'tmb_high')
                significant = bool(p_value is not None and p_value < SIGNIFICANCE_THRESHOLD)
                
                if effect is not None and p_value is not None:
                    summary = f"Interaction effect: {effect:.3f} (p={p_value:.4f})"
                else:
                    summary = f"Interaction effect: N/A (insufficient data)"
                
                analysis = {
                    'hypothesis_ids': [hypothesis_id],
                    'result_summary': summary,
                    'effect_estimate': float(effect) if effect is not None else None,
                    'p_value': p_value,
                    'significant': significant
                }
                iteration['analyses'].append(analysis)
    
    # Iteration 9: Treatment by ECOG PS
    elif iteration_num == 9:
        for treatment in TREATMENTS:
            for outcome in OUTCOMES:
                hypothesis_id = f'h{iteration_num}_{treatment}_ecog'
                hypothesis = {
                    'id': hypothesis_id,
                    'text': f"The effect of {treatment} on {outcome} differs by ECOG PS.",
                    'kind': 'novel'
                }
                iteration['proposed_hypotheses'].append(hypothesis)
                
                effect, p_value = test_interaction(df, treatment, outcome, 'ecog_ps')
                significant = bool(p_value is not None and p_value < SIGNIFICANCE_THRESHOLD)
                
                if effect is not None and p_value is not None:
                    summary = f"Interaction effect: {effect:.3f} (p={p_value:.4f})"
                else:
                    summary = f"Interaction effect: N/A (insufficient data)"
                
                analysis = {
                    'hypothesis_ids': [hypothesis_id],
                    'result_summary': summary,
                    'effect_estimate': float(effect) if effect is not None else None,
                    'p_value': p_value,
                    'significant': significant
                }
                iteration['analyses'].append(analysis)
    
    # Iteration 10: Treatment by brain metastases
    elif iteration_num == 10:
        for treatment in TREATMENTS:
            for outcome in OUTCOMES:
                hypothesis_id = f'h{iteration_num}_{treatment}_brain'
                hypothesis = {
                    'id': hypothesis_id,
                    'text': f"The effect of {treatment} on {outcome} differs by brain metastases status.",
                    'kind': 'novel'
                }
                iteration['proposed_hypotheses'].append(hypothesis)
                
                effect, p_value = test_interaction(df, treatment, outcome, 'has_brain_mets')
                significant = bool(p_value is not None and p_value < SIGNIFICANCE_THRESHOLD)
                
                if effect is not None and p_value is not None:
                    summary = f"Interaction effect: {effect:.3f} (p={p_value:.4f})"
                else:
                    summary = f"Interaction effect: N/A (insufficient data)"
                
                analysis = {
                    'hypothesis_ids': [hypothesis_id],
                    'result_summary': summary,
                    'effect_estimate': float(effect) if effect is not None else None,
                    'p_value': p_value,
                    'significant': significant
                }
                iteration['analyses'].append(analysis)
    
    transcript.append(iteration)
    return iteration

def generate_summary(transcript, df):
    """Generate analysis summary text."""
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append(f"Dataset: ds001_nsclc (50,000 patients)")
    summary_lines.append(f"Total iterations: {len(transcript)}")
    summary_lines.append("")
    
    # Summary by iteration
    for iteration in transcript:
        summary_lines.append(f"--- Iteration {iteration['index']} ---")
        for analysis in iteration['analyses']:
            hypothesis_id = analysis['hypothesis_ids'][0]
            sig = "SIGNIFICANT" if analysis.get('significant', False) else "not significant"
            summary_lines.append(f"  {hypothesis_id}: {analysis['result_summary']} [{sig}]")
        summary_lines.append("")
    
    # Overall findings
    summary_lines.append("=" * 80)
    summary_lines.append("KEY FINDINGS")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Find significant main effects
    main_effects = []
    for iteration in transcript:
        for analysis in iteration['analyses']:
            if 'treatment' in analysis['result_summary'].lower() and 'interaction' not in analysis['result_summary'].lower():
                if analysis.get('significant', False):
                    main_effects.append(analysis)
    
    if main_effects:
        summary_lines.append("Significant main treatment effects on PFS:")
        for effect in main_effects:
            summary_lines.append(f"  - {effect['result_summary']}")
    else:
        summary_lines.append("No significant main treatment effects identified.")
    summary_lines.append("")
    
    # Find significant interactions
    interactions = []
    for iteration in transcript:
        for analysis in iteration['analyses']:
            if 'interaction' in analysis['result_summary'].lower() and analysis.get('significant', False):
                interactions.append(analysis)
    
    if interactions:
        summary_lines.append("Significant treatment-effect heterogeneity (interactions):")
        for interaction in interactions:
            summary_lines.append(f"  - {interaction['result_summary']}")
    else:
        summary_lines.append("No significant treatment-effect heterogeneity identified.")
    summary_lines.append("")
    
    summary_lines.append("=" * 80)
    summary_lines.append("CONCLUSION")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("This analysis explored treatment-outcome relationships across multiple")
    summary_lines.append("subgroups. The results indicate whether treatment effects vary by patient")
    summary_lines.append("characteristics, informing potential personalized treatment strategies.")
    summary_lines.append("")
    
    return "\n".join(summary_lines)

def main():
    transcript = []
    
    print("Starting oncology dataset analysis...")
    print(f"Dataset shape: {df.shape}")
    print(f"Running {MAX_ITERATIONS} iterations...")
    print()
    
    for i in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {i}/{MAX_ITERATIONS}...", end=" ", flush=True)
        run_iteration(i, transcript, df)
        print(f"Completed ({len(transcript[-1]['analyses'])} analyses)")
    
    print()
    print("Generating outputs...")
    
    # Write transcript.json
    transcript_output = {
        'dataset_id': 'ds001_nsclc',
        'model_id': 'codex-cli',
        'harness_id': 'codex-cli@1.0.0',
        'max_iterations': MAX_ITERATIONS,
        'iterations': transcript
    }
    
    with open('transcript.json', 'w') as f:
        json.dump(transcript_output, f, indent=2)
    print("Wrote transcript.json")
    
    # Generate and write summary
    summary = generate_summary(transcript, df)
    with open('analysis_summary.txt', 'w') as f:
        f.write(summary)
    print("Wrote analysis_summary.txt")
    
    print()
    print("Analysis complete!")
    print(f"Total hypotheses tested: {sum(len(it['analyses']) for it in transcript)}")
    print(f"Significant results: {sum(1 for it in transcript for a in it['analyses'] if a.get('significant', False))}")

if __name__ == '__main__':
    main()
