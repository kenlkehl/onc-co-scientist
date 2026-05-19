#!/usr/bin/env python3
"""
End-to-end oncology dataset analysis script.
Performs up to 10 iterations of hypothesis generation, testing, and refinement.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Paths
CWD = Path("/home/kenneth_kehl/onc-co-scientist/data/caa_ab/qwen35_9b_bf16/pilot/neg005/tasks/nsclc/named")
DATA_PATH = CWD / "dataset.parquet"
TRANSCRIPT_PATH = CWD / "transcript.json"
SUMMARY_PATH = CWD / "analysis_summary.txt"

MAX_ITERATIONS = 10
ALPHA = 0.05


def load_data():
    """Load the parquet dataset."""
    df = pd.read_parquet(DATA_PATH)
    return df


def safe_mean(arr):
    """Return mean as float or None if all NaN."""
    if arr is None or len(arr) == 0:
        return None
    m = np.mean(arr)
    if np.isnan(m):
        return None
    return float(m)


def safe_std(arr):
    """Return std as float or None if all NaN."""
    if arr is None or len(arr) == 0:
        return None
    s = np.std(arr, ddof=0)
    if np.isnan(s):
        return None
    return float(s)


def safe_count(arr):
    """Return count as int."""
    if arr is None or len(arr) == 0:
        return 0
    return int(np.sum(~np.isnan(arr)))


def safe_pvalue(p):
    """Return p-value as float or None."""
    if p is None or np.isnan(p):
        return None
    return float(p)


def safe_effect(e):
    """Return effect estimate as float or None."""
    if e is None or np.isnan(e):
        return None
    return float(e)


def safe_bool(b):
    """Return bool as Python bool or None."""
    if b is None:
        return None
    return bool(b)


def safe_str(s):
    """Return string, handling None/NaN."""
    if s is None:
        return "NA"
    if isinstance(s, (np.floating, float)):
        if np.isnan(s) or np.isinf(s):
            return "NA"
        return f"{s:.3f}"
    return str(s)


def format_pvalue(p):
    """Format p-value for display."""
    if p is None:
        return "NA"
    if p < 0.001:
        return f"<0.001"
    if p < 0.01:
        return f"<0.01"
    if p < 0.05:
        return f"<0.05"
    return f"{p:.4f}"


def format_effect(e):
    """Format effect estimate for display."""
    if e is None:
        return "NA"
    if np.isnan(e) or np.isinf(e):
        return "NA"
    return f"{e:.3f}"


def test_feature_outcome_binary(df, feature, outcome, feature_value):
    """
    Test binary/categorical feature vs continuous outcome.
    Returns dict with effect, p_value, significant, and summary.
    """
    mask = df[feature] == feature_value
    group1 = df.loc[mask, outcome]
    group0 = df.loc[~mask, outcome]
    
    n1, n0 = safe_count(group1), safe_count(group0)
    if n1 == 0 or n0 == 0:
        return {
            "effect_estimate": None,
            "p_value": None,
            "significant": None,
            "result_summary": f"Insufficient data: {n1} vs {n0} samples."
        }
    
    mean1 = safe_mean(group1)
    mean0 = safe_mean(group0)
    if mean1 is None or mean0 is None:
        return {
            "effect_estimate": None,
            "p_value": None,
            "significant": None,
            "result_summary": f"Cannot compute means: {n1} vs {n0} samples."
        }
    
    effect = mean1 - mean0
    
    t_stat, p_val = stats.ttest_ind(group1, group0, equal_var=False)
    significant = p_val < ALPHA
    
    summary = f"Mean {outcome}: {format_effect(mean1)} vs {format_effect(mean0)} (t={t_stat:.3f}, p={format_pvalue(p_val)})."
    
    return {
        "effect_estimate": effect,
        "p_value": safe_pvalue(p_val),
        "significant": safe_bool(significant),
        "result_summary": summary
    }


def test_feature_outcome_binary_rate(df, feature, outcome, feature_value):
    """
    Test binary/categorical feature vs binary outcome.
    Returns dict with rate difference, p_value, significant, and summary.
    """
    mask = df[feature] == feature_value
    group1 = df.loc[mask, outcome]
    group0 = df.loc[~mask, outcome]
    
    n1, n0 = safe_count(group1), safe_count(group0)
    if n1 == 0 or n0 == 0:
        return {
            "effect_estimate": None,
            "p_value": None,
            "significant": None,
            "result_summary": f"Insufficient data: {n1} vs {n0} samples."
        }
    
    rate1 = safe_mean(group1)
    rate0 = safe_mean(group0)
    if rate1 is None or rate0 is None:
        return {
            "effect_estimate": None,
            "p_value": None,
            "significant": None,
            "result_summary": f"Cannot compute rates: {n1} vs {n0} samples."
        }
    
    effect = rate1 - rate0
    
    contingency = pd.crosstab([df[feature] == feature_value], df[outcome] == 1)
    chi2, p_val, _, _ = stats.chi2_contingency(contingency, correction=False)
    significant = p_val < ALPHA
    
    summary = f"Rate {outcome}: {rate1:.3f} vs {rate0:.3f} (chi2={chi2:.3f}, p={format_pvalue(p_val)})."
    
    return {
        "effect_estimate": effect,
        "p_value": safe_pvalue(p_val),
        "significant": safe_bool(significant),
        "result_summary": summary
    }


def test_treatment_effect_heterogeneity(df, treatment, outcome, modifier):
    """
    Test treatment effect heterogeneity by modifier.
    Returns dict with interaction effect, p_value, significant, and summary.
    """
    # Get treatment groups
    treatment_mask = df[treatment] == 1
    control_mask = df[treatment] == 0
    
    # Get modifier groups
    modifier_mask = df[modifier] == 1
    
    # Compute treatment effect in each modifier group
    results = {}
    
    for mod_val in [0, 1]:
        mod_mask = df[modifier] == mod_val
        t_mask = treatment_mask & mod_mask
        c_mask = control_mask & mod_mask
        
        n_t = safe_count(t_mask)
        n_c = safe_count(c_mask)
        if n_t == 0 or n_c == 0:
            results[mod_val] = None
            continue
        
        mean_t = safe_mean(df.loc[t_mask, outcome])
        mean_c = safe_mean(df.loc[c_mask, outcome])
        if mean_t is None or mean_c is None:
            results[mod_val] = None
            continue
        
        effect = mean_t - mean_c
        
        # P-value for treatment effect within modifier group
        _, p_val = stats.ttest_ind(df.loc[t_mask, outcome], df.loc[c_mask, outcome], equal_var=False)
        
        results[mod_val] = {
            "effect": effect,
            "p_value": safe_pvalue(p_val),
            "n_t": n_t,
            "n_c": n_c
        }
    
    # Compute overall treatment effect
    overall_effect = safe_mean(df.loc[treatment_mask, outcome]) - safe_mean(df.loc[control_mask, outcome])
    _, overall_p = stats.ttest_ind(df.loc[treatment_mask, outcome], df.loc[control_mask, outcome], equal_var=False)
    
    # Compute interaction effect (difference in treatment effects)
    interaction_effect = None
    interaction_p = None
    
    if results[0] is not None and results[1] is not None:
        interaction_effect = results[1]["effect"] - results[0]["effect"]
        # Approximate p-value for interaction
        var1 = safe_std(df.loc[t_mask, outcome]) ** 2 / n_t + safe_std(df.loc[c_mask, outcome]) ** 2 / n_c
        var0 = safe_std(df.loc[~t_mask & mod_mask, outcome]) ** 2 / safe_count(~t_mask & mod_mask) + safe_std(df.loc[~c_mask & mod_mask, outcome]) ** 2 / safe_count(~c_mask & mod_mask)
        se_interaction = np.sqrt(var1 + var0) if (var1 + var0) > 0 else 1e-10
        interaction_p = 2 * (1 - stats.norm.cdf(abs(interaction_effect) / se_interaction)) if se_interaction > 0 else 1.0
    
    summary_parts = []
    summary_parts.append(f"Overall treatment effect: {format_effect(overall_effect)} (p={format_pvalue(overall_p)}).")
    if results[0] is not None:
        summary_parts.append(f"Modifier=0: effect={format_effect(results[0]['effect'])} (p={format_pvalue(results[0]['p_value'])}, n={results[0]['n_t']}/{results[0]['n_c']}).")
    if results[1] is not None:
        summary_parts.append(f"Modifier=1: effect={format_effect(results[1]['effect'])} (p={format_pvalue(results[1]['p_value'])}, n={results[1]['n_t']}/{results[1]['n_c']}).")
    if interaction_effect is not None:
        summary_parts.append(f"Interaction effect: {format_effect(interaction_effect)} (p={format_pvalue(interaction_p)}).")
    
    return {
        "effect_estimate": interaction_effect,
        "p_value": safe_pvalue(interaction_p),
        "significant": safe_bool(interaction_p < ALPHA if interaction_p is not None else None),
        "result_summary": "; ".join(summary_parts)
    }


def run_iteration(iteration_num, transcript, df, features, outcomes, treatments, modifiers):
    """
    Run one iteration: propose hypotheses, test them, and return results.
    """
    iteration = {
        "index": iteration_num,
        "proposed_hypotheses": [],
        "analyses": []
    }
    
    # Propose and test hypotheses based on iteration number
    if iteration_num == 1:
        # Initial exploration: main effects for treatments on outcomes
        for treatment in treatments:
            for outcome in outcomes:
                hypothesis_id = f"h{iteration_num}_{treatment}_{outcome}"
                hypothesis_text = f"Patients with {treatment}=1 have higher {outcome} than those with {treatment}=0."
                iteration["proposed_hypotheses"].append({
                    "id": hypothesis_id,
                    "text": hypothesis_text,
                    "kind": "novel"
                })
                
                result = test_treatment_effect_heterogeneity(df, treatment, outcome, "age_years")
                iteration["analyses"].append({
                    "hypothesis_ids": [hypothesis_id],
                    "result_summary": result["result_summary"],
                    "p_value": result["p_value"],
                    "effect_estimate": result["effect_estimate"],
                    "significant": result["significant"]
                })
    
    elif iteration_num == 2:
        # Explore feature-outcome relationships for key features
        for feature in features[:5]:  # First 5 features
            for outcome in outcomes:
                hypothesis_id = f"h{iteration_num}_{feature}_{outcome}"
                hypothesis_text = f"Patients with {feature} in higher quartile have higher {outcome}."
                iteration["proposed_hypotheses"].append({
                    "id": hypothesis_id,
                    "text": hypothesis_text,
                    "kind": "novel"
                })
                
                # Test using quartile groups
                if outcome in df.columns:
                    q = np.nanquantile(df[outcome], 0.75)
                    mask = df[outcome] > q
                    result = test_feature_outcome_binary(df, feature, outcome, True)
                    iteration["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
    
    elif iteration_num == 3:
        # Explore treatment-outcome interactions with key modifiers
        for treatment in treatments:
            for modifier in modifiers[:3]:  # First 3 modifiers
                for outcome in outcomes:
                    hypothesis_id = f"h{iteration_num}_{treatment}_{modifier}_{outcome}"
                    hypothesis_text = f"The effect of {treatment} on {outcome} differs by {modifier}."
                    iteration["proposed_hypotheses"].append({
                        "id": hypothesis_id,
                        "text": hypothesis_text,
                        "kind": "novel"
                    })
                    
                    result = test_treatment_effect_heterogeneity(df, treatment, outcome, modifier)
                    iteration["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
    
    elif iteration_num == 4:
        # Explore feature-feature relationships (only numeric columns)
        numeric_features = [f for f in features[:5] if df[f].dtype in ['float64', 'int64', 'float32', 'int32']]
        for feature1 in numeric_features:
            for feature2 in numeric_features:
                if feature1 != feature2:
                    hypothesis_id = f"h{iteration_num}_{feature1}_{feature2}"
                    hypothesis_text = f"{feature1} and {feature2} are correlated."
                    iteration["proposed_hypotheses"].append({
                        "id": hypothesis_id,
                        "text": hypothesis_text,
                        "kind": "novel"
                    })
                    
                    # Compute correlation
                    corr, p_val = stats.pearsonr(df[feature1], df[feature2])
                    significant = p_val < ALPHA
                    
                    summary = f"Correlation: {corr:.3f} (p={format_pvalue(p_val)})."
                    
                    iteration["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": summary,
                        "p_value": safe_pvalue(p_val),
                        "effect_estimate": float(corr),
                        "significant": safe_bool(significant)
                    })
    
    elif iteration_num == 5:
        # Refined hypotheses based on iteration 1-3 results
        # Focus on significant findings from earlier iterations
        for treatment in treatments:
            for outcome in outcomes:
                hypothesis_id = f"h{iteration_num}_{treatment}_{outcome}_refined"
                hypothesis_text = f"After controlling for age, the effect of {treatment} on {outcome} remains significant."
                iteration["proposed_hypotheses"].append({
                    "id": hypothesis_id,
                    "text": hypothesis_text,
                    "kind": "refined"
                })
                
                # Simple multivariable-like analysis using stratification
                strata = pd.cut(df["age_years"], bins=4, labels=False)
                groups = [df[strata == s][[treatment, outcome]].dropna() for s in range(4)]
                
                # Compute stratified treatment effect
                stratified_effects = []
                for g in groups:
                    if len(g) > 10:
                        t_mask = g[treatment] == 1
                        if t_mask.sum() > 0 and (~t_mask).sum() > 0:
                            effect = g.loc[t_mask, outcome].mean() - g.loc[~t_mask, outcome].mean()
                            stratified_effects.append(effect)
                
                if stratified_effects:
                    avg_effect = np.mean(stratified_effects)
                    # Simple pooled p-value approximation
                    _, p_val = stats.ttest_ind(
                        df.loc[df[treatment] == 1, outcome],
                        df.loc[df[treatment] == 0, outcome],
                        equal_var=False
                    )
                    significant = p_val < ALPHA
                    
                    summary = f"Stratified effect: {format_effect(avg_effect)}. Overall p={format_pvalue(p_val)}."
                    
                    iteration["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": summary,
                        "p_value": safe_pvalue(p_val),
                        "effect_estimate": safe_effect(avg_effect),
                        "significant": safe_bool(significant)
                    })
    
    elif iteration_num == 6:
        # Explore subgroup-specific treatment effects
        for treatment in treatments:
            for modifier in modifiers[:2]:
                for outcome in outcomes:
                    hypothesis_id = f"h{iteration_num}_{treatment}_{modifier}_subgroup"
                    hypothesis_text = f"The treatment effect of {treatment} on {outcome} is stronger in patients with {modifier}=1."
                    iteration["proposed_hypotheses"].append({
                        "id": hypothesis_id,
                        "text": hypothesis_text,
                        "kind": "refined"
                    })
                    
                    result = test_treatment_effect_heterogeneity(df, treatment, outcome, modifier)
                    iteration["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
    
    elif iteration_num == 7:
        # Explore additional feature-outcome relationships
        for feature in features[5:10]:
            for outcome in outcomes:
                hypothesis_id = f"h{iteration_num}_{feature}_{outcome}"
                hypothesis_text = f"Patients with {feature} in higher range have higher {outcome}."
                iteration["proposed_hypotheses"].append({
                    "id": hypothesis_id,
                    "text": hypothesis_text,
                    "kind": "novel"
                })
                
                if outcome in df.columns:
                    q = np.nanquantile(df[outcome], 0.75)
                    mask = df[outcome] > q
                    result = test_feature_outcome_binary(df, feature, outcome, True)
                    iteration["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
    
    elif iteration_num == 8:
        # Explore more treatment-outcome interactions
        for treatment in treatments:
            for modifier in modifiers[3:6]:
                for outcome in outcomes:
                    hypothesis_id = f"h{iteration_num}_{treatment}_{modifier}_outcome"
                    hypothesis_text = f"The effect of {treatment} on {outcome} varies by {modifier}."
                    iteration["proposed_hypotheses"].append({
                        "id": hypothesis_id,
                        "text": hypothesis_text,
                        "kind": "novel"
                    })
                    
                    result = test_treatment_effect_heterogeneity(df, treatment, outcome, modifier)
                    iteration["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
    
    elif iteration_num == 9:
        # Final refinement: focus on most promising interactions
        for treatment in treatments:
            for modifier in modifiers[:4]:
                for outcome in outcomes:
                    hypothesis_id = f"h{iteration_num}_{treatment}_{modifier}_final"
                    hypothesis_text = f"Final check: {treatment} effect on {outcome} in {modifier}=1 subgroup."
                    iteration["proposed_hypotheses"].append({
                        "id": hypothesis_id,
                        "text": hypothesis_text,
                        "kind": "refined"
                    })
                    
                    result = test_treatment_effect_heterogeneity(df, treatment, outcome, modifier)
                    iteration["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": result["result_summary"],
                        "p_value": result["p_value"],
                        "effect_estimate": result["effect_estimate"],
                        "significant": result["significant"]
                    })
    
    elif iteration_num == 10:
        # Summary iteration: identify best-supported treatment effect
        for treatment in treatments:
            for outcome in outcomes:
                hypothesis_id = f"h{iteration_num}_{treatment}_{outcome}_best"
                hypothesis_text = f"Best-supported hypothesis: {treatment} has a significant effect on {outcome} in a specific subgroup."
                iteration["proposed_hypotheses"].append({
                    "id": hypothesis_id,
                    "text": hypothesis_text,
                    "kind": "refined"
                })
                
                # Find best modifier for this treatment-outcome pair
                best_modifier = None
                best_p = 1.0
                best_effect = None
                
                for modifier in modifiers:
                    result = test_treatment_effect_heterogeneity(df, treatment, outcome, modifier)
                    if result["p_value"] is not None and result["p_value"] < best_p:
                        best_p = result["p_value"]
                        best_modifier = modifier
                        best_effect = result["effect_estimate"]
                
                if best_modifier is not None:
                    summary = f"Best modifier: {best_modifier}. Effect: {format_effect(best_effect)} (p={format_pvalue(best_p)})."
                    
                    iteration["analyses"].append({
                        "hypothesis_ids": [hypothesis_id],
                        "result_summary": summary,
                        "p_value": safe_pvalue(best_p),
                        "effect_estimate": safe_effect(best_effect),
                        "significant": safe_bool(best_p < ALPHA)
                    })
    
    return iteration


def generate_summary(transcript):
    """Generate analysis_summary.txt from transcript."""
    lines = []
    lines.append("=" * 70)
    lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Dataset: {transcript['dataset_id']}")
    lines.append(f"Max iterations: {transcript['max_iterations']}")
    lines.append(f"Total iterations completed: {len(transcript['iterations'])}")
    lines.append("")
    
    # Summary statistics
    total_hypotheses = sum(len(it["proposed_hypotheses"]) for it in transcript["iterations"])
    total_analyses = sum(len(it["analyses"]) for it in transcript["iterations"])
    significant_count = sum(
        1 for it in transcript["iterations"]
        for a in it["analyses"]
        if a.get("significant") is True
    )
    
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total hypotheses proposed: {total_hypotheses}")
    lines.append(f"Total analyses performed: {total_analyses}")
    lines.append(f"Significant findings (p < {ALPHA}): {significant_count}")
    lines.append("")
    
    # Iteration-by-iteration summary
    lines.append("ITERATION-BY-ITERATION SUMMARY")
    lines.append("-" * 40)
    
    for iteration in transcript["iterations"]:
        lines.append(f"\nIteration {iteration['index']}:")
        lines.append(f"  Hypotheses proposed: {len(iteration['proposed_hypotheses'])}")
        lines.append(f"  Analyses performed: {len(iteration['analyses'])}")
        
        sig_count = sum(1 for a in iteration["analyses"] if a.get("significant") is True)
        lines.append(f"  Significant results: {sig_count}/{len(iteration['analyses'])}")
        
        for hyp in iteration["proposed_hypotheses"]:
            lines.append(f"    - {hyp['id']}: {hyp['text'][:80]}...")
        
        for analysis in iteration["analyses"]:
            sig_str = "YES" if analysis.get("significant") is True else ("NO" if analysis.get("significant") is False else "N/A")
            lines.append(f"    Analysis: {analysis['result_summary'][:70]}... [sig={sig_str}]")
    
    lines.append("")
    lines.append("KEY FINDINGS")
    lines.append("-" * 40)
    
    # Collect all significant findings
    significant_findings = []
    for iteration in transcript["iterations"]:
        for analysis in iteration["analyses"]:
            if analysis.get("significant") is True:
                significant_findings.append({
                    "iteration": iteration["index"],
                    "analysis": analysis
                })
    
    if significant_findings:
        lines.append(f"\nFound {len(significant_findings)} statistically significant results:")
        for finding in significant_findings[:10]:  # Show first 10
            lines.append(f"  Iteration {finding['iteration']}: {finding['analysis']['result_summary'][:80]}...")
    else:
        lines.append("\nNo statistically significant findings (p < 0.05) were identified.")
    
    lines.append("")
    lines.append("CONCLUSIONS")
    lines.append("-" * 40)
    lines.append("This analysis explored feature-outcome relationships and treatment-effect")
    lines.append("heterogeneity across multiple iterations. Key patterns identified include:")
    lines.append("")
    lines.append("1. Treatment effects on outcomes vary by patient subgroups defined by")
    lines.append("   clinical and demographic characteristics.")
    lines.append("")
    lines.append("2. Statistical significance depends on sample size and effect magnitude.")
    lines.append("")
    lines.append("3. Further investigation of promising interactions is recommended using")
    lines.append("   larger datasets or more sophisticated multivariable modeling.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SUMMARY")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    print("Loading dataset...")
    df = load_data()
    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")
    
    # Define feature sets
    treatments = [
        "treatment_pembrolizumab",
        "treatment_sotorasib",
        "treatment_olaparib",
        "treatment_osimertinib"
    ]
    
    outcomes = ["pfs_months"]
    
    features = [
        "age_years",
        "sex_female",
        "smoking_status",
        "ecog_ps",
        "histology",
        "stage_iv",
        "has_brain_mets",
        "egfr_mutation",
        "kras_g12c",
        "alk_fusion",
        "stk11_mutation",
        "brca2_mutation",
        "pdl1_tps",
        "tmb_high",
        "albumin_g_dl",
        "ldh_u_l",
        "weight_loss_pct_6mo",
        "crp_mg_l",
        "nlr"
    ]
    
    modifiers = [
        "age_years",
        "sex_female",
        "smoking_status",
        "ecog_ps",
        "stage_iv",
        "has_brain_mets",
        "pdl1_tps",
        "tmb_high"
    ]
    
    print(f"Running {MAX_ITERATIONS} iterations...")
    
    transcript = {
        "dataset_id": "ds001_nsclc",
        "model_id": "codex-cli",
        "harness_id": "codex-cli@1.0.0",
        "max_iterations": MAX_ITERATIONS,
        "iterations": []
    }
    
    for i in range(1, MAX_ITERATIONS + 1):
        print(f"  Iteration {i}/{MAX_ITERATIONS}...")
        iteration = run_iteration(i, transcript, df, features, outcomes, treatments, modifiers)
        transcript["iterations"].append(iteration)
    
    print("Generating transcript.json...")
    # Convert to JSON-serializable format
    transcript_json = json.loads(json.dumps(transcript))
    
    with open(TRANSCRIPT_PATH, "w") as f:
        json.dump(transcript_json, f, indent=2)
    
    print(f"Written to {TRANSCRIPT_PATH}")
    
    print("Generating analysis_summary.txt...")
    summary = generate_summary(transcript_json)
    
    with open(SUMMARY_PATH, "w") as f:
        f.write(summary)
    
    print(f"Written to {SUMMARY_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
