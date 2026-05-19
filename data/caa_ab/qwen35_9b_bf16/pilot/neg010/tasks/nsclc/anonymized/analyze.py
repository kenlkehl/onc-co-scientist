#!/usr/bin/env python
"""
End-to-end oncology dataset analysis script.
Performs up to 10 iterations of hypothesis generation, testing, and refinement.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import os
from pathlib import Path
from scipy import stats
import pandas as pd
import numpy as np

# Paths
CWD = Path("/home/kenneth_kehl/onc-co-scientist/data/caa_ab/qwen35_9b_bf16/pilot/neg010/tasks/nsclc/anonymized")
DATA_FILE = CWD / "dataset.parquet"
TRANSCRIPT_FILE = CWD / "transcript.json"
SUMMARY_FILE = CWD / "analysis_summary.txt"

# Dataset info
DATASET_ID = "ds001_nsclc"
MAX_ITERATIONS = 10

def load_data():
    """Load the parquet dataset."""
    df = pd.read_parquet(DATA_FILE)
    return df

def safe_float(val, default=np.nan):
    """Convert to float safely."""
    if pd.isna(val):
        return default
    return float(val)

def safe_bool(val, default=False):
    """Convert to bool safely."""
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return default
    return bool(val)

def compute_rate(df, mask, outcome_col):
    """Compute rate (mean) of outcome in masked group."""
    return df.loc[mask, outcome_col].mean()

def compute_effect(df, feature_col, outcome_col, feature_value):
    """
    Compute effect estimate for binary feature.
    Returns (effect_estimate, p_value, significant, result_summary)
    """
    mask = df[feature_col] == feature_value
    other_mask = df[feature_col] != feature_value
    
    rate_treated = compute_rate(df, mask, outcome_col)
    rate_control = compute_rate(df, other_mask, outcome_col)
    
    effect_estimate = rate_treated - rate_control
    
    # Build 2x2 table for chi-square
    n_treated = mask.sum()
    n_control = other_mask.sum()
    count_treated = int((mask & (df[outcome_col] == 1)).sum())
    count_control = int((other_mask & (df[outcome_col] == 1)).sum())
    
    contingency = np.array([[count_treated, n_treated - count_treated],
                           [count_control, n_control - count_control]])
    
    # Use fisher_exact for small expected values
    try:
        _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
    except ValueError:
        # Fall back to fisher's exact test
        _, p_value = stats.fisher_exact(contingency)
    
    significant = p_value < 0.05
    
    result_summary = f"Rate: {rate_treated:.3f} vs {rate_control:.3f} (p={p_value:.4f})."
    
    return effect_estimate, p_value, significant, result_summary

def compute_correlation(df, feature_col, outcome_col):
    """
    Compute Pearson correlation for continuous feature-outcome.
    Returns (effect_estimate, p_value, significant, result_summary)
    """
    feature_vals = df[feature_col].values.astype(float)
    outcome_vals = df[outcome_col].values.astype(float)
    
    corr, p_value = stats.pearsonr(feature_vals, outcome_vals)
    
    # Effect estimate as correlation coefficient
    effect_estimate = float(corr)
    
    significant = p_value < 0.05
    
    result_summary = f"Correlation: {corr:.4f} (p={p_value:.4f})."
    
    return effect_estimate, p_value, significant, result_summary

def compute_regression_effect(df, feature_col, outcome_col):
    """
    Compute regression coefficient for feature on outcome.
    Returns (effect_estimate, p_value, significant, result_summary)
    """
    X = pd.to_numeric(df[feature_col], errors='coerce').values.astype(float)
    y = df[outcome_col].values.astype(float)
    
    # Remove NaN values
    mask = ~np.isnan(X) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        return 0.0, 1.0, False, "No valid data."
    
    # Simple linear regression
    X_mean = X.mean()
    y_mean = y.mean()
    
    numerator = ((X - X_mean) * (y - y_mean)).sum()
    denominator = ((X - X_mean) ** 2).sum()
    
    if denominator == 0:
        effect_estimate = 0.0
    else:
        effect_estimate = numerator / denominator
    
    # Standard error and t-statistic
    residuals = y - (effect_estimate * (X - X_mean) + y_mean)
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    
    if ss_tot == 0:
        p_value = 1.0
    else:
        r_squared = 1 - ss_res / ss_tot
        n = len(y)
        mse = ss_res / (n - 2)
        se = np.sqrt(mse / ((X - X_mean) ** 2).sum())
        t_stat = effect_estimate / se if se != 0 else 0.0
        
        # Two-tailed p-value from t-distribution
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    
    significant = p_value < 0.05
    
    result_summary = f"Regression coef: {effect_estimate:.4f} (p={p_value:.4f})."
    
    return effect_estimate, p_value, significant, result_summary

def propose_hypotheses(df, iteration, iteration_results):
    """
    Propose hypotheses based on previous iteration results.
    Returns list of (hypothesis_id, hypothesis_text) tuples.
    """
    hypotheses = []
    
    # Get all feature columns
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    # Only pfs_months is the outcome (patient_id is an identifier)
    outcome_cols = ['pfs_months']
    
    # Strategy: In each iteration, focus on different feature-outcome pairs
    # Use iteration number to select which features to examine
    
    # Get features examined in previous iterations
    examined_features = set()
    for prev_iter in iteration_results:
        for analysis in prev_iter.get('analyses', []):
            # Extract feature from result_summary or hypothesis text
            for h in prev_iter.get('proposed_hypotheses', []):
                examined_features.add(h['text'].split()[0] if h['text'] else None)
    
    # Select features not yet examined (or re-examine with different angle)
    unexamined_features = [f for f in feature_cols if f not in examined_features]
    
    # If we've examined all features, start cycling back
    if not unexamined_features:
        unexamined_features = feature_cols[:3]  # Re-examine first 3
    
    # Select outcome to analyze
    if outcome_cols:
        outcome = outcome_cols[0]  # pfs_months
    else:
        outcome = None
    
    # Propose hypotheses for selected features
    for i, feature in enumerate(unexamined_features[:2]):  # 1-2 hypotheses per iteration
        hypothesis_id = f"h{iteration}_{i+1}"
        
        if outcome:
            # Feature-outcome hypothesis
            text = f"Feature {feature} is associated with {outcome}."
            hypotheses.append((hypothesis_id, text))
        else:
            # Feature-feature hypothesis (correlation)
            other_feature = unexamined_features[i+1] if i+1 < len(unexamined_features) else feature_cols[0]
            text = f"Feature {feature} is correlated with feature {other_feature}."
            hypotheses.append((hypothesis_id, text))
    
    return hypotheses

def analyze_hypothesis(df, hypothesis_id, hypothesis_text, iteration):
    """
    Analyze a hypothesis and return analysis record.
    """
    analyses = []
    
    # Parse hypothesis to determine analysis type
    # Look for feature names in hypothesis text
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    outcome_cols = ['pfs_months']
    
    # Extract feature from hypothesis text
    feature_match = None
    outcome_match = None
    
    for feat in feature_cols:
        if feat in hypothesis_text:
            feature_match = feat
            break
    
    if not feature_match and feature_cols:
        feature_match = feature_cols[0]
    
    for outcome in outcome_cols:
        if outcome in hypothesis_text:
            outcome_match = outcome
            break
    
    if not outcome_match and outcome_cols:
        outcome_match = outcome_cols[0]
    
    if not feature_match or not outcome_match:
        # Default to first feature and outcome
        feature_match = feature_cols[0] if feature_cols else None
        outcome_match = outcome_cols[0] if outcome_cols else None
    
    if not feature_match or not outcome_match:
        return analyses
    
    # Determine analysis type based on feature type
    # Check if feature is binary (two unique values)
    unique_vals = df[feature_match].nunique()
    
    if unique_vals == 2:
        # Binary feature - use chi-square for outcome
        # Test both values
        for val in sorted(df[feature_match].unique()):
            effect, p_val, sig, summary = compute_effect(df, feature_match, outcome_match, val)
            
            analysis = {
                "hypothesis_ids": [hypothesis_id],
                "result_summary": summary,
                "p_value": float(p_val),
                "effect_estimate": float(effect),
                "significant": bool(sig)
            }
            analyses.append(analysis)
    elif unique_vals > 2:
        # Categorical feature - use regression
        effect, p_val, sig, summary = compute_regression_effect(df, feature_match, outcome_match)
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": summary,
            "p_value": float(p_val),
            "effect_estimate": float(effect),
            "significant": bool(sig)
        }
        analyses.append(analysis)
    else:
        # Continuous feature - use correlation
        effect, p_val, sig, summary = compute_correlation(df, feature_match, outcome_match)
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": summary,
            "p_value": float(p_val),
            "effect_estimate": float(effect),
            "significant": bool(sig)
        }
        analyses.append(analysis)
    
    return analyses

def run_analysis():
    """Main analysis loop."""
    # Load data
    df = load_data()
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize transcript
    transcript = {
        "dataset_id": DATASET_ID,
        "model_id": "codex-cli",
        "harness_id": "codex-cli@1.0.0",
        "max_iterations": MAX_ITERATIONS,
        "iterations": []
    }
    
    # Track examined features for hypothesis generation
    examined_features = set()
    
    # Run iterations
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n=== Iteration {iteration} ===")
        
        # Propose hypotheses
        hypotheses = propose_hypotheses(df, iteration, transcript["iterations"])
        
        proposed_hypotheses = []
        for h_id, h_text in hypotheses:
            proposed_hypotheses.append({
                "id": h_id,
                "text": h_text,
                "kind": "novel"
            })
        
        # Analyze hypotheses
        analyses = []
        for h_id, h_text in hypotheses:
            iteration_results = [i for i in transcript["iterations"]]
            analyses.extend(analyze_hypothesis(df, h_id, h_text, iteration))
        
        # Add iteration to transcript
        iteration_record = {
            "index": iteration,
            "proposed_hypotheses": proposed_hypotheses,
            "analyses": analyses
        }
        transcript["iterations"].append(iteration_record)
        
        # Track examined features
        for analysis in analyses:
            for h in proposed_hypotheses:
                examined_features.add(h['text'])
        
        print(f"Proposed {len(proposed_hypotheses)} hypotheses, ran {len(analyses)} analyses")
        
        # Print summary of significant findings
        for analysis in analyses:
            if analysis.get('significant', False):
                print(f"  Significant: {analysis['result_summary']}")
    
    return transcript, df

def generate_summary(transcript, df):
    """Generate analysis_summary.txt from transcript."""
    lines = []
    lines.append("=" * 70)
    lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Dataset: {transcript['dataset_id']}")
    lines.append(f"Total patients: {len(df)}")
    lines.append(f"Total iterations: {len(transcript['iterations'])}")
    lines.append("")
    
    # Summary statistics
    lines.append("-" * 70)
    lines.append("DATASET OVERVIEW")
    lines.append("-" * 70)
    
    # Feature types
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    outcome_cols = ['pfs_months']
    
    lines.append(f"Features: {len(feature_cols)}")
    lines.append(f"Outcomes: {len(outcome_cols)}")
    lines.append("")
    
    # Outcome statistics
    for outcome in outcome_cols:
        lines.append(f"  {outcome}: mean={df[outcome].mean():.3f}, std={df[outcome].std():.3f}")
    lines.append("")
    
    # Iteration summary
    lines.append("-" * 70)
    lines.append("ITERATION SUMMARY")
    lines.append("-" * 70)
    
    significant_count = 0
    total_analyses = 0
    
    for iteration_record in transcript['iterations']:
        idx = iteration_record['index']
        hypo_count = len(iteration_record['proposed_hypotheses'])
        anal_count = len(iteration_record['analyses'])
        sig_count = sum(1 for a in iteration_record['analyses'] if a.get('significant', False))
        
        total_analyses += anal_count
        significant_count += sig_count
        
        lines.append(f"Iteration {idx}: {hypo_count} hypotheses, {anal_count} analyses, {sig_count} significant")
    
    lines.append(f"\nTotal analyses: {total_analyses}")
    lines.append(f"Significant findings: {significant_count}")
    lines.append("")
    
    # Detailed findings by iteration
    lines.append("-" * 70)
    lines.append("DETAILED FINDINGS")
    lines.append("-" * 70)
    
    for iteration_record in transcript['iterations']:
        idx = iteration_record['index']
        lines.append(f"\n### Iteration {idx}")
        
        for hypo in iteration_record['proposed_hypotheses']:
            lines.append(f"\nHypothesis: {hypo['text']}")
            
            for analysis in iteration_record['analyses']:
                if hypo['id'] in analysis['hypothesis_ids']:
                    lines.append(f"  Effect estimate: {analysis['effect_estimate']:.4f}")
                    lines.append(f"  P-value: {analysis['p_value']:.4f}")
                    lines.append(f"  Significant: {analysis['significant']}")
                    lines.append(f"  Summary: {analysis['result_summary']}")
    
    # Conclusion
    lines.append("")
    lines.append("-" * 70)
    lines.append("CONCLUSIONS")
    lines.append("-" * 70)
    
    # Find significant feature-outcome relationships
    significant_findings = []
    for iteration_record in transcript['iterations']:
        for analysis in iteration_record['analyses']:
            if analysis.get('significant', False):
                significant_findings.append(analysis)
    
    if significant_findings:
        lines.append(f"Found {len(significant_findings)} statistically significant associations.")
        lines.append("")
        for finding in significant_findings[:10]:  # Top 10
            lines.append(f"  - {finding['result_summary']}")
    else:
        lines.append("No statistically significant associations were found at alpha=0.05.")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SUMMARY")
    lines.append("=" * 70)
    
    return "\n".join(lines)

def main():
    """Main entry point."""
    print("Starting oncology dataset analysis...")
    
    # Run analysis
    transcript, df = run_analysis()
    
    # Generate summary
    summary = generate_summary(transcript, df)
    
    # Write outputs
    print(f"\nWriting transcript.json to {TRANSCRIPT_FILE}...")
    with open(TRANSCRIPT_FILE, 'w') as f:
        json.dump(transcript, f, indent=2)
    
    print(f"Writing analysis_summary.txt to {SUMMARY_FILE}...")
    with open(SUMMARY_FILE, 'w') as f:
        f.write(summary)
    
    print("\nAnalysis complete!")
    print(f"  - {TRANSCRIPT_FILE}")
    print(f"  - {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
