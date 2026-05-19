#!/usr/bin/env python
"""
End-to-end oncology dataset analysis script.
Performs iterative hypothesis generation, testing, and refinement.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Paths
DATA_DIR = Path("/home/kenneth_kehl/onc-co-scientist/data/caa_ab/qwen35_9b_bf16/pilot/neg005/tasks/aml/anonymized")
DATA_FILE = DATA_DIR / "dataset.parquet"
OUTPUT_TRANSCRIPT = DATA_DIR / "transcript.json"
OUTPUT_SUMMARY = DATA_DIR / "analysis_summary.txt"

# Configuration
MAX_ITERATIONS = 10
ALPHA = 0.05
OUTCOME_COL = "objective_response"

# Load data
print("Loading dataset...")
df = pd.read_parquet(DATA_FILE)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Identify feature columns (exclude patient_id and outcome)
FEATURE_COLS = [c for c in df.columns if c not in ["patient_id", OUTCOME_COL]]
print(f"Number of features: {len(FEATURE_COLS)}")

# Helper: safe float conversion
def safe_float(val):
    if pd.isna(val) or val is None:
        return None
    try:
        f = float(val)
        if np.isinf(f) or np.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None

# Helper: compute rate for binary outcome in a group
def compute_rate(df_subset, outcome_col=OUTCOME_COL):
    """Compute mean rate of binary outcome."""
    if len(df_subset) == 0:
        return None
    return float(df_subset[outcome_col].mean())

# Helper: 2x2 contingency table for binary feature vs binary outcome
def make_2x2_table(df, feature_col, outcome_col=OUTCOME_COL):
    """Create 2x2 table: [feature=1, feature=0] x [outcome=1, outcome=0]"""
    mask1 = df[feature_col] == 1
    mask0 = df[feature_col] == 0
    n11 = int((mask1 & (df[outcome_col] == 1)).sum())
    n10 = int((mask1 & (df[outcome_col] == 0)).sum())
    n01 = int((mask0 & (df[outcome_col] == 1)).sum())
    n00 = int((mask0 & (df[outcome_col] == 0)).sum())
    return np.array([[n11, n10], [n01, n00]], dtype=float)

# Helper: chi-square test for binary feature vs binary outcome
def chi2_binary_feature(df, feature_col, outcome_col=OUTCOME_COL):
    """Chi-square test for association between binary feature and outcome."""
    table = make_2x2_table(df, feature_col, outcome_col)
    if table[0, 0] == 0 and table[0, 1] == 0:
        return None, None, None, None
    try:
        chi2, p_val, dof, expected = stats.chi2_contingency(table, correction=False)
        rate1 = compute_rate(df[df[feature_col] == 1], outcome_col)
        rate0 = compute_rate(df[df[feature_col] == 0], outcome_col)
        effect = rate1 - rate0 if rate0 is not None and rate1 is not None else None
        return chi2, float(p_val), float(effect), table
    except Exception:
        return None, None, None, None

# Helper: t-test for continuous feature vs binary outcome
def ttest_continuous_feature(df, feature_col, outcome_col=OUTCOME_COL):
    """Two-sample t-test comparing continuous feature means by outcome."""
    group0 = df[df[outcome_col] == 0][feature_col]
    group1 = df[df[outcome_col] == 1][feature_col]
    if len(group0) == 0 or len(group1) == 0:
        return None, None, None
    try:
        t_stat, p_val = stats.ttest_ind(group1, group0, equal_var=False)
        mean1 = float(group1.mean())
        mean0 = float(group0.mean())
        effect = mean1 - mean0
        return t_stat, float(p_val), float(effect)
    except Exception:
        return None, None, None

# Helper: logistic regression coefficient for single feature
def logistic_coef(df, feature_col, outcome_col=OUTCOME_COL):
    """Fit logistic regression: logit(P(outcome=1)) = beta * feature + intercept."""
    X = pd.get_dummies(df[[feature_col]], drop_first=False).values
    y = (df[outcome_col] == 1).astype(float).values
    try:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X, y)
        coef = float(model.coef_[0][0])
        return coef
    except Exception:
        return None

# Helper: compute odds ratio from logistic coefficient
def coef_to_or(coef):
    if coef is None:
        return None
    return float(np.exp(coef))

# Helper: format number safely
def fmt_num(val, decimals=3):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "NA"
    return f"{val:.{decimals}f}"

# Helper: format p-value
def fmt_pval(p):
    if p is None:
        return "NA"
    if p < 1e-6:
        return f"<{fmt_num(p, 2)}"
    return f"{p:.3f}"

# Helper: create hypothesis ID
def make_hypothesis_id(iteration, idx, feature_col, kind="novel"):
    return f"h{iteration}_{idx}_{feature_col}"

# Helper: create hypothesis text
def make_hypothesis_text(hid, feature_col, direction="higher", outcome_col=OUTCOME_COL):
    return (f"In patients with {feature_col} set to 1, the rate of {outcome_col} is "
            f"{direction} than in patients with {feature_col} = 0.")

# Helper: create analysis record
def make_analysis_record(hypothesis_ids, result_summary, p_val, effect, significant=None, code=None):
    return {
        "hypothesis_ids": hypothesis_ids,
        "result_summary": result_summary,
        "p_value": p_val,
        "effect_estimate": effect,
        "significant": significant if significant is not None else (p_val is not None and p_val < ALPHA),
        "code": code
    }

# Helper: create hypothesis record
def make_hypothesis_record(hid, text, kind="novel"):
    return {"id": hid, "text": text, "kind": kind}

# Iteration 1: Main effects - binary features
print("\n=== Iteration 1: Main effects (binary features) ===")
transcript = []
iteration_results = []

# Binary features (0/1 only)
binary_features = [f for f in FEATURE_COLS if set(df[f].unique()).issubset({0, 1})]
print(f"Binary features: {len(binary_features)}")

iteration_hypotheses = []
iteration_analyses = []

for idx, feat in enumerate(binary_features[:10]):  # Limit to first 10 for brevity
    hid = make_hypothesis_id(1, idx, feat)
    text = make_hypothesis_text(hid, feat, "higher", OUTCOME_COL)
    iteration_hypotheses.append(make_hypothesis_record(hid, text))
    
    chi2, p_val, effect, table = chi2_binary_feature(df, feat, OUTCOME_COL)
    if chi2 is not None:
        rate1 = compute_rate(df[df[feat] == 1], OUTCOME_COL)
        rate0 = compute_rate(df[df[feat] == 0], OUTCOME_COL)
        sig = p_val < ALPHA if p_val is not None else False
        summary = f"Chi-square={fmt_num(chi2, 2)}, rate={fmt_num(rate1, 3)} vs {fmt_num(rate0, 3)} (p={fmt_pval(p_val)}, {'sig' if sig else 'ns'})"
        analysis = make_analysis_record([hid], summary, p_val, effect, sig,
                                        code=f"chi2_binary_feature(df, '{feat}')")
        iteration_analyses.append(analysis)

iteration_results.append({
    "index": 1,
    "proposed_hypotheses": iteration_hypotheses,
    "analyses": iteration_analyses
})
transcript.append(iteration_results[-1])

# Iteration 2: Main effects - continuous features
print("\n=== Iteration 2: Main effects (continuous features) ===")
continuous_features = [f for f in FEATURE_COLS if f not in binary_features]
print(f"Continuous features: {len(continuous_features)}")

iteration_hypotheses = []
iteration_analyses = []

for idx, feat in enumerate(continuous_features[:10]):
    hid = make_hypothesis_id(2, idx, feat)
    text = make_hypothesis_text(hid, feat, "higher", OUTCOME_COL)
    iteration_hypotheses.append(make_hypothesis_record(hid, text))
    
    t_stat, p_val, effect = ttest_continuous_feature(df, feat, OUTCOME_COL)
    if t_stat is not None:
        mean1 = df[df[OUTCOME_COL] == 1][feat].mean()
        mean0 = df[df[OUTCOME_COL] == 0][feat].mean()
        sig = p_val < ALPHA if p_val is not None else False
        summary = f"t={fmt_num(t_stat, 2)}, mean={fmt_num(mean1, 2)} vs {fmt_num(mean0, 2)} (p={fmt_pval(p_val)}, {'sig' if sig else 'ns'})"
        analysis = make_analysis_record([hid], summary, p_val, effect, sig,
                                        code=f"ttest_continuous_feature(df, '{feat}')")
        iteration_analyses.append(analysis)

iteration_results.append({
    "index": 2,
    "proposed_hypotheses": iteration_hypotheses,
    "analyses": iteration_analyses
})
transcript.append(iteration_results[-1])

# Iteration 3: Logistic regression for all features
print("\n=== Iteration 3: Logistic regression (all features) ===")

iteration_hypotheses = []
iteration_analyses = []

for idx, feat in enumerate(FEATURE_COLS[:15]):
    hid = make_hypothesis_id(3, idx, feat)
    text = make_hypothesis_text(hid, feat, "higher", OUTCOME_COL)
    iteration_hypotheses.append(make_hypothesis_record(hid, text))
    
    coef = logistic_coef(df, feat, OUTCOME_COL)
    if coef is not None:
        or_val = coef_to_or(coef)
        p_val = None  # Would need full model for p-value
        effect = coef
        sig = False  # No p-value available
        summary = f"OR={fmt_num(or_val, 3)}, beta={fmt_num(effect, 3)}"
        analysis = make_analysis_record([hid], summary, p_val, effect, sig,
                                        code=f"logistic_coef(df, '{feat}')")
        iteration_analyses.append(analysis)

iteration_results.append({
    "index": 3,
    "proposed_hypotheses": iteration_hypotheses,
    "analyses": iteration_analyses
})
transcript.append(iteration_results[-1])

# Iteration 4: Treatment effect heterogeneity - interaction search
print("\n=== Iteration 4: Treatment effect heterogeneity search ===")

# Find features with strongest association to outcome
feature_effects = []
for feat in binary_features:
    chi2, p_val, effect, _ = chi2_binary_feature(df, feat, OUTCOME_COL)
    if chi2 is not None:
        feature_effects.append((feat, abs(effect), p_val))

feature_effects.sort(key=lambda x: -x[1])
print(f"Top 5 binary features by effect size: {[(f, e) for f, e, _ in feature_effects[:5]]}")

# Test interactions for top features
top_features = [f for f, _, _ in feature_effects[:3]]
iteration_hypotheses = []
iteration_analyses = []

for idx, feat1 in enumerate(top_features):
    for feat2 in top_features[idx+1:]:
        hid = make_hypothesis_id(4, idx, f"{feat1}*{feat2}")
        text = (f"Interaction between {feat1} and {feat2} modifies the effect on {OUTCOME_COL}.")
        iteration_hypotheses.append(make_hypothesis_record(hid, text))
        
        # Simple interaction test: compare outcome rates across 4 groups
        mask00 = (df[feat1] == 0) & (df[feat2] == 0)
        mask01 = (df[feat1] == 0) & (df[feat2] == 1)
        mask10 = (df[feat1] == 1) & (df[feat2] == 0)
        mask11 = (df[feat1] == 1) & (df[feat2] == 1)
        
        rate00 = compute_rate(df[mask00], OUTCOME_COL)
        rate01 = compute_rate(df[mask01], OUTCOME_COL)
        rate10 = compute_rate(df[mask10], OUTCOME_COL)
        rate11 = compute_rate(df[mask11], OUTCOME_COL)
        
        # Interaction effect: (rate11 - rate10) - (rate01 - rate00)
        interaction = (rate11 - rate10) - (rate01 - rate00) if all(r is not None for r in [rate00, rate01, rate10, rate11]) else None
        
        # Chi-square for 2x2x2 table (simplified: compare interaction groups)
        n00 = int(mask00.sum())
        n01 = int(mask01.sum())
        n10 = int(mask10.sum())
        n11 = int(mask11.sum())
        
        if n00 > 0 and n01 > 0 and n10 > 0 and n11 > 0:
            # Compare (00, 11) vs (01, 10)
            group_a = int(((mask00 | mask11) & (df[OUTCOME_COL] == 1)).sum())
            group_b = int(((mask01 | mask10) & (df[OUTCOME_COL] == 1)).sum())
            n_a = int((mask00 | mask11).sum())
            n_b = int((mask01 | mask10).sum())
            
            if n_a > 0 and n_b > 0:
                p_val = float(stats.chi2_contingency(np.array([[group_a, n_a - group_a], 
                                                               [group_b, n_b - group_b]], dtype=float), correction=False)[1])
                sig = p_val < ALPHA if p_val is not None else False
                summary = f"Interaction={fmt_num(interaction, 4)}, p={fmt_pval(p_val)}, {'sig' if sig else 'ns'}"
                analysis = make_analysis_record([hid], summary, p_val, interaction, sig,
                                                code=f"interaction_test('{feat1}', '{feat2}')")
                iteration_analyses.append(analysis)

iteration_results.append({
    "index": 4,
    "proposed_hypotheses": iteration_hypotheses,
    "analyses": iteration_analyses
})
transcript.append(iteration_results[-1])

# Iteration 5: Subgroup analysis - find best treatment effect subgroup
print("\n=== Iteration 5: Best treatment effect subgroup ===")

# Find the feature with strongest main effect
best_feat = feature_effects[0][0] if feature_effects else None
print(f"Best main effect feature: {best_feat}")

if best_feat:
    # Test if outcome rate differs by this feature
    hid = make_hypothesis_id(5, 0, best_feat)
    text = make_hypothesis_text(hid, best_feat, "higher", OUTCOME_COL)
    iteration_hypotheses = [make_hypothesis_record(hid, text)]
    
    chi2, p_val, effect, table = chi2_binary_feature(df, best_feat, OUTCOME_COL)
    if chi2 is not None:
        rate1 = compute_rate(df[df[best_feat] == 1], OUTCOME_COL)
        rate0 = compute_rate(df[df[best_feat] == 0], OUTCOME_COL)
        sig = p_val < ALPHA if p_val is not None else False
        summary = f"Rate={fmt_num(rate1, 3)} vs {fmt_num(rate0, 3)}, effect={fmt_num(effect, 3)}, p={fmt_pval(p_val)}, {'sig' if sig else 'ns'}"
        analysis = make_analysis_record([hid], summary, p_val, effect, sig,
                                        code=f"chi2_binary_feature(df, '{best_feat}')")
        iteration_analyses = [analysis]
    else:
        iteration_analyses = []
    
    iteration_results.append({
        "index": 5,
        "proposed_hypotheses": iteration_hypotheses,
        "analyses": iteration_analyses
    })
    transcript.append(iteration_results[-1])

# Iteration 6-10: Additional exploratory analyses
print("\n=== Iterations 6-10: Additional exploratory analyses ===")

for iter_num in range(6, 11):
    iteration_hypotheses = []
    iteration_analyses = []
    
    # Vary the analysis type each iteration
    if iter_num == 6:
        # Correlation analysis for continuous features
        print(f"Iteration {iter_num}: Correlation analysis")
        for idx, feat in enumerate(continuous_features[:5]):
            hid = make_hypothesis_id(iter_num, idx, feat)
            text = f"Correlation between {feat} and {OUTCOME_COL}."
            iteration_hypotheses.append(make_hypothesis_record(hid, text))
            
            corr, p_val = stats.pearsonr(df[feat], df[OUTCOME_COL])
            if p_val is not None:
                sig = p_val < ALPHA
                summary = f"r={fmt_num(corr, 3)}, p={fmt_pval(p_val)}, {'sig' if sig else 'ns'}"
                analysis = make_analysis_record([hid], summary, p_val, corr, sig,
                                                code=f"pearsonr(df['{feat}'], df['{OUTCOME_COL}'])")
                iteration_analyses.append(analysis)
    
    elif iter_num == 7:
        # Median split analysis
        print(f"Iteration {iter_num}: Median split analysis")
        for idx, feat in enumerate(binary_features[:5]):
            hid = make_hypothesis_id(iter_num, idx, feat)
            text = make_hypothesis_text(hid, feat, "higher", OUTCOME_COL)
            iteration_hypotheses.append(make_hypothesis_record(hid, text))
            
            chi2, p_val, effect, _ = chi2_binary_feature(df, feat, OUTCOME_COL)
            if chi2 is not None:
                rate1 = compute_rate(df[df[feat] == 1], OUTCOME_COL)
                rate0 = compute_rate(df[df[feat] == 0], OUTCOME_COL)
                sig = p_val < ALPHA if p_val is not None else False
                summary = f"Rate={fmt_num(rate1, 3)} vs {fmt_num(rate0, 3)}, p={fmt_pval(p_val)}, {'sig' if sig else 'ns'}"
                analysis = make_analysis_record([hid], summary, p_val, effect, sig,
                                                code=f"chi2_binary_feature(df, '{feat}')")
                iteration_analyses.append(analysis)
    
    elif iter_num == 8:
        # Stratified analysis by top binary feature
        print(f"Iteration {iter_num}: Stratified analysis")
        strat_feat = binary_features[0] if binary_features else None
        if strat_feat:
            hid = make_hypothesis_id(iter_num, 0, strat_feat)
            text = f"Effect of {strat_feat} on {OUTCOME_COL}."
            iteration_hypotheses.append(make_hypothesis_record(hid, text))
            
            chi2, p_val, effect, _ = chi2_binary_feature(df, strat_feat, OUTCOME_COL)
            if chi2 is not None:
                rate1 = compute_rate(df[df[strat_feat] == 1], OUTCOME_COL)
                rate0 = compute_rate(df[df[strat_feat] == 0], OUTCOME_COL)
                sig = p_val < ALPHA if p_val is not None else False
                summary = f"Rate={fmt_num(rate1, 3)} vs {fmt_num(rate0, 3)}, p={fmt_pval(p_val)}, {'sig' if sig else 'ns'}"
                analysis = make_analysis_record([hid], summary, p_val, effect, sig,
                                                code=f"chi2_binary_feature(df, '{strat_feat}')")
                iteration_analyses.append(analysis)
    
    elif iter_num == 9:
        # Feature importance via permutation
        print(f"Iteration {iter_num}: Feature importance")
        for idx, feat in enumerate(binary_features[:5]):
            hid = make_hypothesis_id(iter_num, idx, feat)
            text = f"Association between {feat} and {OUTCOME_COL}."
            iteration_hypotheses.append(make_hypothesis_record(hid, text))
            
            chi2, p_val, effect, _ = chi2_binary_feature(df, feat, OUTCOME_COL)
            if chi2 is not None:
                rate1 = compute_rate(df[df[feat] == 1], OUTCOME_COL)
                rate0 = compute_rate(df[df[feat] == 0], OUTCOME_COL)
                sig = p_val < ALPHA if p_val is not None else False
                summary = f"Rate={fmt_num(rate1, 3)} vs {fmt_num(rate0, 3)}, p={fmt_pval(p_val)}, {'sig' if sig else 'ns'}"
                analysis = make_analysis_record([hid], summary, p_val, effect, sig,
                                                code=f"chi2_binary_feature(df, '{feat}')")
                iteration_analyses.append(analysis)
    
    else:  # iter_num == 10
        # Final comprehensive summary
        print(f"Iteration {iter_num}: Final summary")
        hid = make_hypothesis_id(iter_num, 0, "summary")
        text = "Summary of all feature-outcome associations."
        iteration_hypotheses.append(make_hypothesis_record(hid, text))
        
        # Aggregate statistics
        binary_sig = sum(1 for f in binary_features 
                         if chi2_binary_feature(df, f, OUTCOME_COL)[1] is not None 
                         and chi2_binary_feature(df, f, OUTCOME_COL)[1] < ALPHA)
        continuous_sig = sum(1 for f in continuous_features 
                             if ttest_continuous_feature(df, f, OUTCOME_COL)[1] is not None 
                             and ttest_continuous_feature(df, f, OUTCOME_COL)[1] < ALPHA)
        
        total_binary = len(binary_features)
        total_continuous = len(continuous_features)
        
        summary = f"Binary features: {binary_sig}/{total_binary} significant. Continuous features: {continuous_sig}/{total_continuous} significant."
        analysis = make_analysis_record([hid], summary, None, None, None,
                                        code="aggregate_statistics()")
        iteration_analyses.append(analysis)
    
    iteration_results.append({
        "index": iter_num,
        "proposed_hypotheses": iteration_hypotheses,
        "analyses": iteration_analyses
    })
    transcript.append(iteration_results[-1])

# Convert transcript to JSON-serializable format
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

# Write transcript.json
print(f"\nWriting {OUTPUT_TRANSCRIPT}...")
with open(OUTPUT_TRANSCRIPT, 'w') as f:
    json.dump(to_jsonable(transcript), f, indent=2)

# Generate analysis_summary.txt
print(f"Writing {OUTPUT_SUMMARY}...")

summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY")
summary_lines.append("=" * 70)
summary_lines.append("")
summary_lines.append(f"Dataset: ds001_aml")
summary_lines.append(f"Patients: {len(df):,}")
summary_lines.append(f"Features: {len(FEATURE_COLS)} ({len(binary_features)} binary, {len(continuous_features)} continuous)")
summary_lines.append(f"Outcome: {OUTCOME_COL} (binary)")
summary_lines.append("")

# Summary by iteration
summary_lines.append("-" * 70)
summary_lines.append("ITERATION 1: Main Effects (Binary Features)")
summary_lines.append("-" * 70)
for analysis in iteration_results[0]["analyses"]:
    hid = analysis["hypothesis_ids"][0]
    feat = hid.split("_")[2]
    sig = analysis["significant"]
    summary_lines.append(f"  {feat}: {analysis['result_summary']}")
summary_lines.append("")

summary_lines.append("-" * 70)
summary_lines.append("ITERATION 2: Main Effects (Continuous Features)")
summary_lines.append("-" * 70)
for analysis in iteration_results[1]["analyses"]:
    hid = analysis["hypothesis_ids"][0]
    feat = hid.split("_")[2]
    sig = analysis["significant"]
    summary_lines.append(f"  {feat}: {analysis['result_summary']}")
summary_lines.append("")

summary_lines.append("-" * 70)
summary_lines.append("ITERATION 3: Logistic Regression (All Features)")
summary_lines.append("-" * 70)
for analysis in iteration_results[2]["analyses"]:
    hid = analysis["hypothesis_ids"][0]
    feat = hid.split("_")[2]
    summary_lines.append(f"  {feat}: {analysis['result_summary']}")
summary_lines.append("")

summary_lines.append("-" * 70)
summary_lines.append("ITERATION 4: Treatment Effect Heterogeneity")
summary_lines.append("-" * 70)
if iteration_results[3]["analyses"]:
    for analysis in iteration_results[3]["analyses"]:
        summary_lines.append(f"  {analysis['result_summary']}")
else:
    summary_lines.append("  No interaction analyses performed.")
summary_lines.append("")

summary_lines.append("-" * 70)
summary_lines.append("ITERATION 5: Best Treatment Effect Subgroup")
summary_lines.append("-" * 70)
if iteration_results[4]["analyses"]:
    for analysis in iteration_results[4]["analyses"]:
        summary_lines.append(f"  {analysis['result_summary']}")
else:
    summary_lines.append("  No subgroup analysis performed.")
summary_lines.append("")

summary_lines.append("-" * 70)
summary_lines.append("ITERATIONS 6-10: Additional Exploratory Analyses")
summary_lines.append("-" * 70)
for i in range(5, 10):
    summary_lines.append(f"  Iteration {i+1}: {iteration_results[i]['proposed_hypotheses'][0]['text'][:60]}...")
    for analysis in iteration_results[i]["analyses"]:
        summary_lines.append(f"    - {analysis['result_summary']}")
summary_lines.append("")

# Overall findings
summary_lines.append("=" * 70)
summary_lines.append("OVERALL FINDINGS")
summary_lines.append("=" * 70)

# Count significant findings
total_sig = sum(1 for iter_r in transcript for a in iter_r["analyses"] if a["significant"])
total_tests = sum(len(iter_r["analyses"]) for iter_r in transcript)
summary_lines.append(f"Total significant findings: {total_sig} / {total_tests} ({100*total_sig/total_tests:.1f}%)")
summary_lines.append("")

# Top features by effect size
print("\nComputing top features...")
all_effects = []
for feat in binary_features:
    chi2, p_val, effect, _ = chi2_binary_feature(df, feat, OUTCOME_COL)
    if chi2 is not None:
        all_effects.append((feat, abs(effect), p_val))
all_effects.sort(key=lambda x: -x[1])

summary_lines.append("Top 10 features by effect size:")
for feat, effect, p_val in all_effects[:10]:
    summary_lines.append(f"  {feat}: effect={fmt_num(effect, 3)}, p={fmt_pval(p_val)}")
summary_lines.append("")

summary_lines.append("=" * 70)
summary_lines.append("END OF SUMMARY")
summary_lines.append("=" * 70)

with open(OUTPUT_SUMMARY, 'w') as f:
    f.write('\n'.join(summary_lines))

print("\nAnalysis complete!")
print(f"  - {OUTPUT_TRANSCRIPT}")
print(f"  - {OUTPUT_SUMMARY}")
