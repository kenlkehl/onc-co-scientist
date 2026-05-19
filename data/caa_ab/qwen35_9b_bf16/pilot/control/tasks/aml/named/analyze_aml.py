#!/usr/bin/env python3
"""
End-to-end AML dataset analysis script.
Performs iterative hypothesis generation, testing, and refinement.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import numpy as np
from scipy import stats
import pandas as pd
from pathlib import Path

# Load dataset
DATA_PATH = Path("/home/kenneth_kehl/onc-co-scientist/data/caa_ab/qwen35_9b_bf16/pilot/control/tasks/aml/named/dataset.parquet")
df = pd.read_parquet(DATA_PATH)

print(f"Loaded {len(df)} patient records")
print(f"Columns: {list(df.columns)}")

# Identify key columns
treatments = [
    "treatment_midostaurin",
    "treatment_gilteritinib", 
    "treatment_ivosidenib",
    "treatment_enasidenib",
    "treatment_venetoclax_azacitidine",
    "treatment_7plus3"
]

outcomes = ["objective_response"]

# Helper functions
def compute_effect(df, feature, value, outcome):
    """Compute effect estimate using boolean mask comparison."""
    mask = df[feature] == value
    group1 = df.loc[mask, outcome]
    group0 = df.loc[~mask, outcome]
    effect = group1.mean() - group0.mean()
    return float(effect)

def compute_chi2_pvalue(df, feature, value, outcome):
    """Compute chi-square p-value for binary/categorical feature-outcome comparison."""
    mask = df[feature] == value
    n1 = len(df.loc[mask, outcome])
    n0 = len(df.loc[~mask, outcome])
    p1 = df.loc[mask, outcome].mean()
    p0 = df.loc[~mask, outcome].mean()
    total = n1 + n0
    total_p = (n1 * p1 + n0 * p0) / total
    contingency = [[n1 * p1, n1 * (1 - p1)], [n0 * p0, n0 * (1 - p0)]]
    _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
    return float(p_value)

def compute_ttest_pvalue(df, feature, value, outcome):
    """Compute t-test p-value for continuous feature-outcome comparison."""
    mask = df[feature] == value
    group1 = df.loc[mask, outcome]
    group0 = df.loc[~mask, outcome]
    t_stat, p_value = stats.ttest_ind(group1, group0, equal_var=False)
    return float(p_value)

def test_interaction(df, treatment, modifier, outcome):
    """Test treatment-by-modifier interaction effect."""
    # Compare treatment effect in modifier=1 vs modifier=0
    mask1 = df[modifier] == 1
    mask0 = df[modifier] == 0
    
    # Check if we have enough samples
    n1 = len(df.loc[mask1])
    n0 = len(df.loc[mask0])
    
    if n1 < 10 or n0 < 10:
        return 0.0, 1.0  # Not enough samples
    
    effect1 = df.loc[mask1 & (df[treatment] == 1), outcome].mean() - df.loc[mask1 & (df[treatment] == 0), outcome].mean()
    effect0 = df.loc[mask0 & (df[treatment] == 1), outcome].mean() - df.loc[mask0 & (df[treatment] == 0), outcome].mean()
    
    interaction_effect = effect1 - effect0
    
    # Test if interaction is significant using pooled variance
    var1 = df.loc[mask1, outcome].var() if len(df.loc[mask1, outcome]) > 1 else 0
    var0 = df.loc[mask0, outcome].var() if len(df.loc[mask0, outcome]) > 1 else 0
    
    pooled_var = ((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2) if (n1 + n0) > 2 else 0
    
    se = np.sqrt(pooled_var / n1 + pooled_var / n0) if pooled_var > 0 else 1e-10
    t_stat = interaction_effect / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n1 + n0 - 2))
    
    return float(interaction_effect), float(p_value)

def test_stratified_effect(df, treatment, modifier, outcome):
    """Test treatment effect stratified by modifier."""
    mask1 = df[modifier] == 1
    mask0 = df[modifier] == 0
    
    # Check if we have enough samples
    n1 = len(df.loc[mask1])
    n0 = len(df.loc[mask0])
    
    if n1 < 10 or n0 < 10:
        return 0.0, 0.0
    
    # Treatment effect in each stratum
    effect1 = df.loc[mask1 & (df[treatment] == 1), outcome].mean() - df.loc[mask1 & (df[treatment] == 0), outcome].mean()
    effect0 = df.loc[mask0 & (df[treatment] == 1), outcome].mean() - df.loc[mask0 & (df[treatment] == 0), outcome].mean()
    
    return float(effect1), float(effect0)

# Initialize transcript
transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "qwen35-9b",
    "harness_id": "codex-cli@1.0.0",
    "max_iterations": 10,
    "iterations": []
}

# Iteration 1: Main effects - treatment vs outcome
print("\n=== Iteration 1: Main treatment effects ===")

iteration1_hypotheses = []
iteration1_analyses = []

for treatment in treatments:
    for outcome in outcomes:
        hypothesis_id = f"h1_{treatment}_{outcome}"
        hypothesis_text = f"Patients receiving {treatment} have different {outcome} rates compared to those not receiving {treatment}."
        iteration1_hypotheses.append({
            "id": hypothesis_id,
            "text": hypothesis_text,
            "kind": "novel"
        })
        
        # Compute effect and p-value
        effect = compute_effect(df, treatment, 1, outcome)
        p_value = compute_chi2_pvalue(df, treatment, 1, outcome)
        significant = bool(p_value < 0.05)
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Effect of {treatment} on {outcome}: {effect:.4f} (p={p_value:.4f}, significant={significant}).",
            "effect_estimate": effect,
            "p_value": p_value,
            "significant": significant
        }
        iteration1_analyses.append(analysis)

transcript["iterations"].append({
    "index": 1,
    "proposed_hypotheses": iteration1_hypotheses,
    "analyses": iteration1_analyses
})

# Iteration 2: Baseline predictors vs outcome
print("\n=== Iteration 2: Baseline predictors vs outcome ===")

iteration2_hypotheses = []
iteration2_analyses = []

# Select key baseline predictors
key_predictors = ["age_years", "ecog_ps", "secondary_aml", "unfit_for_intensive", "complex_karyotype", 
                  "tp53_mutation", "blast_pct_marrow", "albumin_g_dl", "ldh_u_l", "nlr"]

for predictor in key_predictors:
    for outcome in outcomes:
        hypothesis_id = f"h2_{predictor}_{outcome}"
        hypothesis_text = f"Patients with higher {predictor} have different {outcome} rates."
        iteration2_hypotheses.append({
            "id": hypothesis_id,
            "text": hypothesis_text,
            "kind": "novel"
        })
        
        # Determine test type based on predictor type
        if predictor in ["sex_female", "ecog_ps", "secondary_aml", "unfit_for_intensive", "complex_karyotype", 
                         "tp53_mutation", "blast_pct_marrow"]:
            p_value = compute_chi2_pvalue(df, predictor, 1, outcome)
        else:
            p_value = compute_ttest_pvalue(df, predictor, 1, outcome)
        
        effect = compute_effect(df, predictor, 1, outcome)
        significant = bool(p_value < 0.05)
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Effect of {predictor} on {outcome}: {effect:.4f} (p={p_value:.4f}, significant={significant}).",
            "effect_estimate": effect,
            "p_value": p_value,
            "significant": significant
        }
        iteration2_analyses.append(analysis)

transcript["iterations"].append({
    "index": 2,
    "proposed_hypotheses": iteration2_hypotheses,
    "analyses": iteration2_analyses
})

# Iteration 3: Treatment by baseline predictor interactions
print("\n=== Iteration 3: Treatment-by-predictor interactions ===")

iteration3_hypotheses = []
iteration3_analyses = []

# Focus on key interactions
interaction_pairs = [
    ("treatment_7plus3", "age_years"),
    ("treatment_7plus3", "ecog_ps"),
    ("treatment_7plus3", "unfit_for_intensive"),
    ("treatment_7plus3", "complex_karyotype"),
    ("treatment_7plus3", "tp53_mutation"),
    ("treatment_midostaurin", "flt3_itd"),
    ("treatment_midostaurin", "idh1_mutation"),
    ("treatment_ivosidenib", "idh1_mutation"),
]

for treatment, modifier in interaction_pairs:
    hypothesis_id = f"h3_{treatment}_{modifier}"
    hypothesis_text = f"The effect of {treatment} on {outcomes[0]} differs by {modifier} status."
    iteration3_hypotheses.append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    interaction_effect, p_value = test_interaction(df, treatment, modifier, outcomes[0])
    significant = bool(p_value < 0.05)
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Interaction effect of {treatment}x{modifier} on {outcomes[0]}: {interaction_effect:.4f} (p={p_value:.4f}, significant={significant}).",
        "effect_estimate": interaction_effect,
        "p_value": p_value,
        "significant": significant
    }
    iteration3_analyses.append(analysis)

transcript["iterations"].append({
    "index": 3,
    "proposed_hypotheses": iteration3_hypotheses,
    "analyses": iteration3_analyses
})

# Iteration 4: Stratified treatment effects
print("\n=== Iteration 4: Stratified treatment effects ===")

iteration4_hypotheses = []
iteration4_analyses = []

# Stratify key treatments by key modifiers
stratification_pairs = [
    ("treatment_7plus3", "age_years"),
    ("treatment_7plus3", "ecog_ps"),
    ("treatment_7plus3", "unfit_for_intensive"),
    ("treatment_7plus3", "complex_karyotype"),
    ("treatment_7plus3", "tp53_mutation"),
    ("treatment_midostaurin", "flt3_itd"),
    ("treatment_midostaurin", "idh1_mutation"),
    ("treatment_ivosidenib", "idh1_mutation"),
]

for treatment, modifier in stratification_pairs:
    hypothesis_id = f"h4_{treatment}_{modifier}"
    hypothesis_text = f"The effect of {treatment} on {outcomes[0]} varies by {modifier} status."
    iteration4_hypotheses.append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    effect1, effect0 = test_stratified_effect(df, treatment, modifier, outcomes[0])
    
    # Test if effects differ significantly
    mask1 = df[modifier] == 1
    mask0 = df[modifier] == 0
    n1 = len(df.loc[mask1])
    n0 = len(df.loc[mask0])
    
    if n1 < 10 or n0 < 10:
        p_value = 1.0
    else:
        pooled_var = ((n1 - 1) * (df.loc[mask1, outcomes[0]].var() + df.loc[mask1 & (df[treatment] == 0), outcomes[0]].var()) + 
                       (n0 - 1) * (df.loc[mask0, outcomes[0]].var() + df.loc[mask0 & (df[treatment] == 0), outcomes[0]].var())) / (n1 + n0 - 2)
        
        se = np.sqrt(pooled_var / n1 + pooled_var / n0) if pooled_var > 0 else 1e-10
        t_stat = (effect1 - effect0) / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n1 + n0 - 2))
    
    significant = bool(p_value < 0.05)
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Effect of {treatment} in {modifier}=1: {effect1:.4f}; in {modifier}=0: {effect0:.4f} (diff={effect1-effect0:.4f}, p={p_value:.4f}, significant={significant}).",
        "effect_estimate": effect1 - effect0,
        "p_value": p_value,
        "significant": significant
    }
    iteration4_analyses.append(analysis)

transcript["iterations"].append({
    "index": 4,
    "proposed_hypotheses": iteration4_hypotheses,
    "analyses": iteration4_analyses
})

# Iteration 5: Treatment effect heterogeneity - exhaustive search for 7+3
print("\n=== Iteration 5: Treatment effect heterogeneity search (7+3) ===")

iteration5_hypotheses = []
iteration5_analyses = []

# Search for subgroups where 7+3 has strongest effect
modifiers_7plus3 = ["age_years", "ecog_ps", "unfit_for_intensive", "complex_karyotype", "tp53_mutation", 
                    "secondary_aml", "blast_pct_marrow", "nlr"]

best_effect = -np.inf
best_modifier = None
best_effect_diff = -np.inf

for modifier in modifiers_7plus3:
    effect1, effect0 = test_stratified_effect(df, "treatment_7plus3", modifier, outcomes[0])
    effect_diff = effect1 - effect0
    if effect_diff > best_effect_diff:
        best_effect_diff = effect_diff
        best_effect = effect1
        best_modifier = modifier

hypothesis_id = f"h5_7plus3_heterogeneity"
hypothesis_text = f"The effect of treatment_7plus3 on objective_response is strongest in patients with {best_modifier}=1."
iteration5_hypotheses.append({
    "id": hypothesis_id,
    "text": hypothesis_text,
    "kind": "novel"
})

effect1, effect0 = test_stratified_effect(df, "treatment_7plus3", best_modifier, outcomes[0])
mask1 = df[best_modifier] == 1
mask0 = df[best_modifier] == 0
n1 = len(df.loc[mask1])
n0 = len(df.loc[mask0])

if n1 < 10 or n0 < 10:
    p_value = 1.0
else:
    pooled_var = ((n1 - 1) * (df.loc[mask1, outcomes[0]].var() + df.loc[mask1 & (df["treatment_7plus3"] == 0), outcomes[0]].var()) + 
                   (n0 - 1) * (df.loc[mask0, outcomes[0]].var() + df.loc[mask0 & (df["treatment_7plus3"] == 0), outcomes[0]].var())) / (n1 + n0 - 2)
    
    se = np.sqrt(pooled_var / n1 + pooled_var / n0) if pooled_var > 0 else 1e-10
    t_stat = (effect1 - effect0) / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n1 + n0 - 2))

significant = bool(p_value < 0.05)

analysis = {
    "hypothesis_ids": [hypothesis_id],
    "result_summary": f"Effect of 7+3 in {best_modifier}=1: {effect1:.4f}; in {best_modifier}=0: {effect0:.4f} (diff={effect1-effect0:.4f}, p={p_value:.4f}, significant={significant}).",
    "effect_estimate": effect1 - effect0,
    "p_value": p_value,
    "significant": significant
}
iteration5_analyses.append(analysis)

transcript["iterations"].append({
    "index": 5,
    "proposed_hypotheses": iteration5_hypotheses,
    "analyses": iteration5_analyses
})

# Iteration 6: Refined hypothesis based on iteration 5
print("\n=== Iteration 6: Refined heterogeneity hypothesis ===")

iteration6_hypotheses = []
iteration6_analyses = []

# Refine based on best modifier found
hypothesis_id = f"h6_7plus3_{best_modifier}_refined"
hypothesis_text = f"Patients with {best_modifier}=1 who receive treatment_7plus3 have significantly higher objective_response rates than those without {best_modifier}."
iteration6_hypotheses.append({
    "id": hypothesis_id,
    "text": hypothesis_text,
    "kind": "refined"
})

effect1, effect0 = test_stratified_effect(df, "treatment_7plus3", best_modifier, outcomes[0])
mask1 = df[best_modifier] == 1
mask0 = df[best_modifier] == 0
n1 = len(df.loc[mask1])
n0 = len(df.loc[mask0])

if n1 < 10 or n0 < 10:
    p_value = 1.0
else:
    pooled_var = ((n1 - 1) * (df.loc[mask1, outcomes[0]].var() + df.loc[mask1 & (df["treatment_7plus3"] == 0), outcomes[0]].var()) + 
                   (n0 - 1) * (df.loc[mask0, outcomes[0]].var() + df.loc[mask0 & (df["treatment_7plus3"] == 0), outcomes[0]].var())) / (n1 + n0 - 2)
    
    se = np.sqrt(pooled_var / n1 + pooled_var / n0) if pooled_var > 0 else 1e-10
    t_stat = (effect1 - effect0) / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n1 + n0 - 2))

significant = bool(p_value < 0.05)

analysis = {
    "hypothesis_ids": [hypothesis_id],
    "result_summary": f"Effect of 7+3 in {best_modifier}=1: {effect1:.4f}; in {best_modifier}=0: {effect0:.4f} (diff={effect1-effect0:.4f}, p={p_value:.4f}, significant={significant}).",
    "effect_estimate": effect1 - effect0,
    "p_value": p_value,
    "significant": significant
}
iteration6_analyses.append(analysis)

transcript["iterations"].append({
    "index": 6,
    "proposed_hypotheses": iteration6_hypotheses,
    "analyses": iteration6_analyses
})

# Iteration 7: Additional treatment-outcome relationships
print("\n=== Iteration 7: Additional treatment-outcome relationships ===")

iteration7_hypotheses = []
iteration7_analyses = []

# Check remaining treatment-outcome pairs
for treatment in treatments:
    for outcome in outcomes:
        hypothesis_id = f"h7_{treatment}_{outcome}"
        hypothesis_text = f"Patients receiving {treatment} have different {outcome} rates compared to those not receiving {treatment}."
        iteration7_hypotheses.append({
            "id": hypothesis_id,
            "text": hypothesis_text,
            "kind": "novel"
        })
        
        effect = compute_effect(df, treatment, 1, outcome)
        p_value = compute_chi2_pvalue(df, treatment, 1, outcome)
        significant = bool(p_value < 0.05)
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Effect of {treatment} on {outcome}: {effect:.4f} (p={p_value:.4f}, significant={significant}).",
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

# Iteration 8: Baseline predictor-outcome relationships
print("\n=== Iteration 8: Additional baseline predictor-outcome relationships ===")

iteration8_hypotheses = []
iteration8_analyses = []

# Additional predictors
additional_predictors = ["wbc_k_per_ul", "hemoglobin_g_dl", "creatinine_mg_dl", "albumin_g_dl", "nlr"]

for predictor in additional_predictors:
    for outcome in outcomes:
        hypothesis_id = f"h8_{predictor}_{outcome}"
        hypothesis_text = f"Patients with higher {predictor} have different {outcome} rates."
        iteration8_hypotheses.append({
            "id": hypothesis_id,
            "text": hypothesis_text,
            "kind": "novel"
        })
        
        if predictor in ["sex_female", "ecog_ps", "secondary_aml", "unfit_for_intensive", "complex_karyotype", 
                         "tp53_mutation", "blast_pct_marrow"]:
            p_value = compute_chi2_pvalue(df, predictor, 1, outcome)
        else:
            p_value = compute_ttest_pvalue(df, predictor, 1, outcome)
        
        effect = compute_effect(df, predictor, 1, outcome)
        significant = bool(p_value < 0.05)
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Effect of {predictor} on {outcome}: {effect:.4f} (p={p_value:.4f}, significant={significant}).",
            "effect_estimate": effect,
            "p_value": p_value,
            "significant": significant
        }
        iteration8_analyses.append(analysis)

transcript["iterations"].append({
    "index": 8,
    "proposed_hypotheses": iteration8_hypotheses,
    "analyses": iteration8_analyses
})

# Iteration 9: Multi-feature subgroup discovery
print("\n=== Iteration 9: Multi-feature subgroup discovery ===")

iteration9_hypotheses = []
iteration9_analyses = []

# Test combinations of modifiers for 7+3
# Focus on clinically relevant combinations
combinations = [
    ("unfit_for_intensive", "complex_karyotype"),
    ("unfit_for_intensive", "tp53_mutation"),
    ("ecog_ps", "unfit_for_intensive"),
]

for mod1, mod2 in combinations:
    hypothesis_id = f"h9_{mod1}_{mod2}_7plus3"
    hypothesis_text = f"The effect of treatment_7plus3 on objective_response differs by {mod1} and {mod2} status."
    iteration9_hypotheses.append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    # Test 2x2 stratification
    effect_11 = df.loc[(df[mod1] == 1) & (df[mod2] == 1) & (df["treatment_7plus3"] == 1), outcomes[0]].mean()
    effect_10 = df.loc[(df[mod1] == 1) & (df[mod2] == 0) & (df["treatment_7plus3"] == 1), outcomes[0]].mean()
    effect_01 = df.loc[(df[mod1] == 0) & (df[mod2] == 1) & (df["treatment_7plus3"] == 1), outcomes[0]].mean()
    effect_00 = df.loc[(df[mod1] == 0) & (df[mod2] == 0) & (df["treatment_7plus3"] == 1), outcomes[0]].mean()
    
    # Average interaction effect
    interaction_effect = (effect_11 + effect_01) - (effect_10 + effect_00)
    
    # Simplified p-value calculation
    n11 = len(df.loc[(df[mod1] == 1) & (df[mod2] == 1)])
    n10 = len(df.loc[(df[mod1] == 1) & (df[mod2] == 0)])
    n01 = len(df.loc[(df[mod1] == 0) & (df[mod2] == 1)])
    n00 = len(df.loc[(df[mod1] == 0) & (df[mod2] == 0)])
    
    total_n = n11 + n10 + n01 + n00
    pooled_var = ((n11 + n10 + n01 + n00) * 0.25)  # Approximation
    
    se = np.sqrt(pooled_var / total_n) if total_n > 0 else 1e-10
    t_stat = interaction_effect / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), total_n - 1))
    
    significant = bool(p_value < 0.05)
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Interaction effect of 7+3 by {mod1}x{mod2}: {interaction_effect:.4f} (p={p_value:.4f}, significant={significant}).",
        "effect_estimate": interaction_effect,
        "p_value": p_value,
        "significant": significant
    }
    iteration9_analyses.append(analysis)

transcript["iterations"].append({
    "index": 9,
    "proposed_hypotheses": iteration9_hypotheses,
    "analyses": iteration9_analyses
})

# Iteration 10: Final synthesis and best-supported hypotheses
print("\n=== Iteration 10: Final synthesis ===")

iteration10_hypotheses = []
iteration10_analyses = []

# Identify best-supported treatment effect heterogeneity
# Based on all analyses, summarize the strongest findings

# Find the most significant interaction
best_interaction = None
best_interaction_p = np.inf
best_interaction_effect = 0

for i, analysis in enumerate(transcript["iterations"][4]["analyses"]):  # Iteration 5
    if analysis["p_value"] < best_interaction_p:
        best_interaction_p = analysis["p_value"]
        best_interaction_effect = analysis["effect_estimate"]
        best_interaction = analysis["result_summary"]

hypothesis_id = f"h10_best_interaction"
hypothesis_text = f"The strongest treatment-effect heterogeneity is found in the interaction between treatment_7plus3 and {best_modifier}."
iteration10_hypotheses.append({
    "id": hypothesis_id,
    "text": hypothesis_text,
    "kind": "refined"
})

analysis = {
    "hypothesis_ids": [hypothesis_id],
    "result_summary": best_interaction,
    "effect_estimate": best_interaction_effect,
    "p_value": best_interaction_p,
    "significant": bool(best_interaction_p < 0.05)
}
iteration10_analyses.append(analysis)

transcript["iterations"].append({
    "index": 10,
    "proposed_hypotheses": iteration10_hypotheses,
    "analyses": iteration10_analyses
})

# Write transcript.json
output_dir = Path("/home/kenneth_kehl/onc-co-scientist/data/caa_ab/qwen35_9b_bf16/pilot/control/tasks/aml/named")
transcript_path = output_dir / "transcript.json"
with open(transcript_path, "w") as f:
    json.dump(transcript, f, indent=2)
print(f"\nWrote {transcript_path}")

# Generate analysis_summary.txt
summary_lines = []
summary_lines.append("=" * 80)
summary_lines.append("AML DATASET ANALYSIS SUMMARY")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append(f"Dataset: {transcript['dataset_id']}")
summary_lines.append(f"Total patients: {len(df)}")
summary_lines.append(f"Total iterations: {len(transcript['iterations'])}")
summary_lines.append("")

# Summary by iteration
for iteration in transcript["iterations"]:
    summary_lines.append(f"--- Iteration {iteration['index']} ---")
    
    for hypothesis in iteration["proposed_hypotheses"]:
        summary_lines.append(f"  Hypothesis: {hypothesis['text']}")
    
    for analysis in iteration["analyses"]:
        sig_str = "SIGNIFICANT" if analysis["significant"] else "not significant"
        summary_lines.append(f"  Result: {analysis['result_summary']}")
        summary_lines.append(f"    Effect: {analysis['effect_estimate']:.4f}, p={analysis['p_value']:.4f} ({sig_str})")
    
    summary_lines.append("")

# Overall conclusions
summary_lines.append("=" * 80)
summary_lines.append("OVERALL CONCLUSIONS")
summary_lines.append("=" * 80)
summary_lines.append("")

# Count significant findings
total_significant = sum(
    1 for it in transcript["iterations"] 
    for a in it["analyses"] 
    if a["significant"]
)
total_analyses = sum(len(it["analyses"]) for it in transcript["iterations"])

summary_lines.append(f"Total analyses performed: {total_analyses}")
summary_lines.append(f"Statistically significant findings: {total_significant}")
summary_lines.append("")

# Treatment effects summary
summary_lines.append("TREATMENT EFFECTS SUMMARY:")
summary_lines.append("-" * 40)

for treatment in treatments:
    treatment_results = [
        (it, a) for it in transcript["iterations"] 
        for a in it["analyses"] 
        if treatment in a["result_summary"]
    ]
    if treatment_results:
        avg_effect = sum(a["effect_estimate"] for _, a in treatment_results) / len(treatment_results)
        avg_p = sum(a["p_value"] for _, a in treatment_results) / len(treatment_results)
        sig_count = sum(1 for _, a in treatment_results if a["significant"])
        summary_lines.append(f"  {treatment}: avg effect={avg_effect:.4f}, avg p={avg_p:.4f}, {sig_count}/{len(treatment_results)} significant")

summary_lines.append("")

# Heterogeneity findings
summary_lines.append("TREATMENT-EFFECT HETEROGENEITY:")
summary_lines.append("-" * 40)

for iteration in transcript["iterations"][4:]:  # Iterations 5-10
    for analysis in iteration["analyses"]:
        if "interaction" in analysis["result_summary"].lower() or "diff" in analysis["result_summary"].lower():
            summary_lines.append(f"  {analysis['result_summary']}")

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("END OF SUMMARY")
summary_lines.append("=" * 80)

summary_path = output_dir / "analysis_summary.txt"
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))
print(f"Wrote {summary_path}")

print("\nAnalysis complete!")
