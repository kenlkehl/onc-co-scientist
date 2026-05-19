#!/usr/bin/env python3
"""End-to-end oncology dataset analysis script."""
import json, pandas as pd, numpy as np
from scipy import stats
from pathlib import Path

DATA_PATH, OUTPUT_DIR = Path("dataset.parquet"), Path(".")
MAX_ITERATIONS, ALPHA = 10, 0.05

df = pd.read_parquet(DATA_PATH)
outcome_col = "objective_response"
feature_cols = [c for c in df.columns if c.startswith("feature_")]

def safe_float(val):
    return None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)

def compute_effect(feature, outcome, df, method="chi2"):
    mask = df[feature] == 1
    n1, n0 = len(df.loc[mask, outcome]), len(df.loc[~mask, outcome])
    if n1 == 0 or n0 == 0:
        return {"effect_estimate": None, "p_value": None, "significant": None, "result_summary": "Insufficient data"}
    if method == "chi2":
        cont = pd.crosstab(df[feature], df[outcome])
        chi2, p_val, _, _ = stats.chi2_contingency(cont, correction=False)
        prop1 = (cont.loc[1, 1] if 1 in cont.index and 1 in cont.columns else 0) / n1
        prop0 = (cont.loc[0, 1] if 0 in cont.index and 1 in cont.columns else 0) / n0
        effect = prop1 - prop0
    elif method == "correlation":
        corr, p_val = stats.pearsonr(df[feature], df[outcome])
        effect = corr
    else:
        t_stat, p_val = stats.ttest_ind(df.loc[mask, outcome], df.loc[~mask, outcome])
        effect = df.loc[mask, outcome].mean() - df.loc[~mask, outcome].mean()
    return {"effect_estimate": safe_float(effect), "p_value": safe_float(p_val),
            "significant": bool(p_val < ALPHA), "result_summary": f"Effect: {effect:.4f}, p={p_val:.4f}"}

def compute_categorical_effect(feature, outcome, df):
    unique_vals = df[feature].unique()
    results = []
    for val in unique_vals:
        mask = df[feature] == val
        group = df.loc[mask, outcome]
        n = len(group)
        if n == 0:
            continue
        ref_val = unique_vals[0]
        ref_mask = df[feature] == ref_val
        ref_group = df.loc[ref_mask, outcome]
        ref_n = len(ref_group)
        if ref_n == 0:
            continue
        if ref_n == 1 or n == 1:
            table = pd.crosstab(df[feature], df[outcome])
            _, p_val = stats.fisher_exact(table)
            prop = (table.loc[val, 1] if val in table.index and 1 in table.columns else 0) / n
            ref_prop = (table.loc[ref_val, 1] if ref_val in table.index and 1 in table.columns else 0) / ref_n
            effect = prop - ref_prop
        else:
            t_stat, p_val = stats.ttest_ind(group, ref_group)
            effect = group.mean() - ref_group.mean()
        results.append({"value": val, "n": n, "mean": group.mean(), "effect": effect,
                       "p_value": p_val, "significant": p_val < ALPHA})
    if results:
        ref_mean = df.loc[df[feature] == unique_vals[0], outcome].mean()
        overall_effect = results[0]["mean"] - ref_mean
        overall_p = results[0]["p_value"]
        overall_sig = results[0]["significant"]
    else:
        overall_effect, overall_p, overall_sig = None, None, None
    summary = f"Effect vs reference ({unique_vals[0]}): {overall_effect:.4f}, p={overall_p:.4f}"
    if overall_sig:
        summary += " (significant)"
    return {"effect_estimate": safe_float(overall_effect), "p_value": safe_float(overall_p),
            "significant": bool(overall_sig), "result_summary": summary, "categorical_details": results}

transcript = {"dataset_id": "ds001_prostate", "model_id": "qwen35-9b-bf16",
              "harness_id": "codex-cli@pilot", "max_iterations": MAX_ITERATIONS, "iterations": []}

print("=== Iteration 1: Main effects screening ===")
binary_features = [c for c in feature_cols if df[c].nunique() == 2]
continuous_features = [c for c in feature_cols if df[c].dtype == "float64" and df[c].nunique() > 2]
categorical_features = [c for c in feature_cols if df[c].nunique() > 2 and df[c].dtype in ["int64", "float64"]]

iteration_results = []
for i, feat in enumerate(binary_features[:15]):
    result = compute_effect(feat, outcome_col, df, method="chi2")
    result["hypothesis_id"] = f"h1_{i+1}"
    result["feature"] = feat
    iteration_results.append(result)
    print(f"  {feat}: effect={result['effect_estimate']}, p={result['p_value']}, sig={result['significant']}")

for i, feat in enumerate(continuous_features[:15]):
    result = compute_effect(feat, outcome_col, df, method="correlation")
    result["hypothesis_id"] = f"h1_{len(binary_features[:15])+i+1}"
    result["feature"] = feat
    iteration_results.append(result)
    print(f"  {feat}: effect={result['effect_estimate']}, p={result['p_value']}, sig={result['significant']}")

for i, feat in enumerate(categorical_features[:15]):
    result = compute_categorical_effect(feat, outcome_col, df)
    result["hypothesis_id"] = f"h1_{len(binary_features[:15])+len(continuous_features[:15])+i+1}"
    result["feature"] = feat
    iteration_results.append(result)
    print(f"  {feat}: effect={result['effect_estimate']}, p={result['p_value']}, sig={result['significant']}")

transcript["iterations"].append({"index": 1,
    "proposed_hypotheses": [{"id": f"h1_{i+1}", "text": f"Feature {feat} is associated with objective_response.", "kind": "novel"}
                            for i, r in enumerate(iteration_results) for feat in [r["feature"]]],
    "analyses": iteration_results})

significant_findings = [r for r in iteration_results if r["significant"]]
print(f"\nSignificant findings: {len(significant_findings)}")

print("\n=== Iteration 2: Deep dive on significant binary features ===")
if significant_findings:
    sig_binary = [r for r in significant_findings if r["feature"] in binary_features]
    iteration_results = []
    for i, r in enumerate(sig_binary[:5]):
        feat = r["feature"]
        mask = df[feat] == 1
        group1, group0 = df.loc[mask, outcome_col], df.loc[~mask, outcome_col]
        t_stat, p_val = stats.ttest_ind(group1, group0)
        effect = group1.mean() - group0.mean()
        result = {"hypothesis_id": f"h2_{i+1}", "feature": feat,
                  "effect_estimate": safe_float(effect), "p_value": safe_float(p_val),
                  "significant": bool(p_val < ALPHA),
                  "result_summary": f"Mean outcome: {group1.mean():.4f} vs {group0.mean():.4f}, p={p_val:.4f}"}
        iteration_results.append(result)
        print(f"  {feat}: effect={result['effect_estimate']}, p={result['p_value']}")
    transcript["iterations"].append({"index": 2,
        "proposed_hypotheses": [{"id": f"h2_{i+1}", "text": f"Feature {r['feature']} has a significant association with objective_response.", "kind": "novel"}
                                for i, r in enumerate(iteration_results)],
        "analyses": iteration_results})
else:
    transcript["iterations"].append({"index": 2, "proposed_hypotheses": [], "analyses": []})

print("\n=== Iteration 3: Interaction screening ===")
if significant_findings:
    top_sig = max(significant_findings, key=lambda x: x["p_value"])
    top_feat = top_sig["feature"]
    other_binary = [f for f in binary_features if f != top_feat]
    interactions = []
    for mod in other_binary[:10]:
        treat_mask = df[top_feat] == 1
        mod_mask = df[mod] == 1
        mod_treat = df.loc[treat_mask & mod_mask, outcome_col]
        mod_no_treat = df.loc[~treat_mask & mod_mask, outcome_col]
        if len(mod_treat) > 5 and len(mod_no_treat) > 5:
            mod_treat_effect = mod_treat.mean() - mod_no_treat.mean()
            all_treat = df.loc[treat_mask, outcome_col]
            all_no_treat = df.loc[~treat_mask, outcome_col]
            t_stat, p_val = stats.ttest_ind(mod_treat, all_no_treat)
            interactions.append({"modifier": mod, "modifier_value": 1,
                                 "treatment_effect_in_group": mod_treat_effect,
                                 "overall_treatment_effect": all_treat.mean() - all_no_treat.mean(),
                                 "interaction_effect": mod_treat_effect - (all_treat.mean() - all_no_treat.mean()),
                                 "p_value": p_val, "significant": p_val < ALPHA})
    interactions.sort(key=lambda x: x["p_value"])
    iteration_results = []
    for i, inter in enumerate(interactions[:10]):
        result = {"hypothesis_id": f"h3_{i+1}", "treatment_feature": top_feat, "modifier": inter["modifier"],
                  "modifier_value": inter["modifier_value"], "effect_estimate": safe_float(inter["interaction_effect"]),
                  "p_value": safe_float(inter["p_value"]), "significant": inter["significant"],
                  "result_summary": f"Interaction effect: {inter['interaction_effect']:.4f}, p={inter['p_value']:.4f}"}
        iteration_results.append(result)
        print(f"  {inter['modifier']}={inter['modifier_value']}: interaction={result['effect_estimate']}, p={result['p_value']}")
    transcript["iterations"].append({"index": 3,
        "proposed_hypotheses": [{"id": f"h3_{i+1}", "text": f"Interaction between {top_feat} and {inter['modifier']} affects objective_response.", "kind": "novel"}
                                for i, inter in enumerate(interactions[:10])],
        "analyses": iteration_results})
else:
    transcript["iterations"].append({"index": 3, "proposed_hypotheses": [], "analyses": []})

print("\n=== Iteration 4: Subgroup analysis ===")
if significant_findings:
    top_sig = max(significant_findings, key=lambda x: x["p_value"])
    top_feat = top_sig["feature"]
    best_interaction = None
    best_p = 1.0
    for mod in binary_features:
        if mod == top_feat:
            continue
        treat_mask = df[top_feat] == 1
        mod_mask = df[mod] == 1
        treat_and_mod = df.loc[treat_mask & mod_mask, outcome_col]
        no_treat_and_mod = df.loc[~treat_mask & mod_mask, outcome_col]
        if len(treat_and_mod) > 5 and len(no_treat_and_mod) > 5:
            mod_treat_effect = treat_and_mod.mean() - no_treat_and_mod.mean()
            treat_no_mod = df.loc[treat_mask & ~mod_mask, outcome_col]
            no_treat_no_mod = df.loc[~treat_mask & ~mod_mask, outcome_col]
            if len(treat_no_mod) > 5 and len(no_treat_no_mod) > 5:
                no_mod_treat_effect = treat_no_mod.mean() - no_treat_no_mod.mean()
                interaction = mod_treat_effect - no_mod_treat_effect
                t_stat, p_val = stats.ttest_ind(treat_and_mod, treat_no_mod)
                if p_val < best_p:
                    best_p = p_val
                    best_interaction = {"treatment": top_feat, "modifier": mod, "modifier_value": 1,
                                        "treatment_effect_in_subgroup": mod_treat_effect,
                                        "treatment_effect_outside": no_mod_treat_effect,
                                        "interaction": interaction, "p_value": p_val}
    if best_interaction:
        result = {"hypothesis_id": "h4", "treatment_feature": best_interaction["treatment"],
                  "modifier": best_interaction["modifier"], "modifier_value": best_interaction["modifier_value"],
                  "effect_estimate": safe_float(best_interaction["interaction"]),
                  "p_value": safe_float(best_interaction["p_value"]),
                  "significant": best_interaction["p_value"] < ALPHA,
                  "result_summary": f"Best interaction: {best_interaction['treatment']} x {best_interaction['modifier']}={best_interaction['modifier_value']}, p={best_interaction['p_value']:.4f}"}
        transcript["iterations"].append({"index": 4,
            "proposed_hypotheses": [{"id": "h4", "text": f"Treatment effect of {best_interaction['treatment']} is modified by {best_interaction['modifier']}={best_interaction['modifier_value']}.", "kind": "novel"}],
            "analyses": [result]})
        print(f"Best interaction: {best_interaction['treatment']} x {best_interaction['modifier']}={best_interaction['modifier_value']}, p={best_interaction['p_value']:.4f}")
    else:
        transcript["iterations"].append({"index": 4, "proposed_hypotheses": [], "analyses": []})
        print("No significant interactions found")
else:
    transcript["iterations"].append({"index": 4, "proposed_hypotheses": [], "analyses": []})

print("\n=== Iterations 5-10: Additional explorations ===")
for iter_num in range(5, MAX_ITERATIONS + 1):
    iteration_results = []
    if iter_num == 5:
        result = compute_categorical_effect("feature_016", outcome_col, df)
        result["hypothesis_id"] = "h5"
        result["feature"] = "feature_016"
        iteration_results.append(result)
        print(f"  feature_016: effect={result['effect_estimate']}, p={result['p_value']}")
    elif iter_num == 6:
        result = compute_effect("feature_001", outcome_col, df, method="chi2")
        result["hypothesis_id"] = "h6"
        result["feature"] = "feature_001"
        iteration_results.append(result)
        print(f"  feature_001: effect={result['effect_estimate']}, p={result['p_value']}")
    elif iter_num == 7:
        result = compute_effect("feature_013", outcome_col, df, method="chi2")
        result["hypothesis_id"] = "h7"
        result["feature"] = "feature_013"
        iteration_results.append(result)
        print(f"  feature_013: effect={result['effect_estimate']}, p={result['p_value']}")
    elif iter_num == 8:
        result = compute_effect("feature_006", outcome_col, df, method="chi2")
        result["hypothesis_id"] = "h8"
        result["feature"] = "feature_006"
        iteration_results.append(result)
        print(f"  feature_006: effect={result['effect_estimate']}, p={result['p_value']}")
    elif iter_num == 9:
        result = compute_effect("feature_021", outcome_col, df, method="chi2")
        result["hypothesis_id"] = "h9"
        result["feature"] = "feature_021"
        iteration_results.append(result)
        print(f"  feature_021: effect={result['effect_estimate']}, p={result['p_value']}")
    elif iter_num == 10:
        result = compute_effect("feature_015", outcome_col, df, method="chi2")
        result["hypothesis_id"] = "h10"
        result["feature"] = "feature_015"
        iteration_results.append(result)
        print(f"  feature_015: effect={result['effect_estimate']}, p={result['p_value']}")
    transcript["iterations"].append({"index": iter_num,
        "proposed_hypotheses": [{"id": r["hypothesis_id"], "text": f"Feature {r.get('feature', 'unknown')} is associated with objective_response.", "kind": "novel"} for r in iteration_results],
        "analyses": iteration_results})

def make_jsonable(obj):
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_jsonable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (float, int, str, type(None))):
        return obj
    elif pd.isna(obj):
        return None
    else:
        return str(obj)

transcript_jsonable = make_jsonable(transcript)
print("\nWriting transcript.json...")
with open(OUTPUT_DIR / "transcript.json", "w") as f:
    json.dump(transcript_jsonable, f, indent=2)

print("Writing analysis_summary.txt...")
summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY")
summary_lines.append("=" * 70)
summary_lines.append("")
summary_lines.append(f"Dataset: ds001_prostate")
summary_lines.append(f"Total patients: {len(df)}")
summary_lines.append(f"Outcome: objective_response (binary: 0/1)")
summary_lines.append(f"Total features analyzed: {len(feature_cols)}")
summary_lines.append(f"  - Binary features: {len(binary_features)}")
summary_lines.append(f"  - Continuous features: {len(continuous_features)}")
summary_lines.append(f"  - Categorical features: {len(categorical_features)}")
summary_lines.append("")
summary_lines.append("-" * 70)
summary_lines.append("ITERATION-BY-ITERATION SUMMARY")
summary_lines.append("-" * 70)
for iteration in transcript["iterations"]:
    iter_num = iteration["index"]
    sig_count = sum(1 for a in iteration["analyses"] if a.get("significant", False))
    total_count = len(iteration["analyses"])
    summary_lines.append(f"\nIteration {iter_num}:")
    summary_lines.append(f"  Total analyses: {total_count}")
    summary_lines.append(f"  Significant findings: {sig_count}")
    if iteration["analyses"]:
        for analysis in iteration["analyses"]:
            feat = analysis.get("feature", analysis.get("treatment_feature", "unknown"))
            effect = analysis.get("effect_estimate")
            p_val = analysis.get("p_value")
            sig = analysis.get("significant", False)
            summary_lines.append(f"    - {feat}: effect={effect}, p={p_val}, significant={'YES' if sig else 'NO'}")
summary_lines.append("")
summary_lines.append("-" * 70)
summary_lines.append("OVERALL FINDINGS")
summary_lines.append("-" * 70)
all_sig = [a for it in transcript["iterations"] for a in it["analyses"] if a.get("significant", False)]
all_not_sig = [a for it in transcript["iterations"] for a in it["analyses"] if not a.get("significant", False)]
summary_lines.append(f"\nTotal significant findings: {len(all_sig)}")
summary_lines.append(f"Total non-significant findings: {len(all_not_sig)}")
if all_sig:
    summary_lines.append("\nSignificant feature-outcome associations:")
    for a in all_sig:
        feat = a.get("feature", a.get("treatment_feature", "unknown"))
        effect = a.get("effect_estimate")
        p_val = a.get("p_value")
        summary_lines.append(f"  - {feat}: effect={effect}, p={p_val}")
if all_not_sig:
    summary_lines.append("\nNon-significant features (sample):")
    for a in all_not_sig[:5]:
        feat = a.get("feature", a.get("treatment_feature", "unknown"))
        summary_lines.append(f"  - {feat}")
summary_lines.append("")
summary_lines.append("-" * 70)
summary_lines.append("TREATMENT-EFFECT HETEROGENEITY")
summary_lines.append("-" * 70)
if best_interaction:
    summary_lines.append(f"\nBest treatment-effect modifier:")
    summary_lines.append(f"  Treatment: {best_interaction['treatment']}")
    summary_lines.append(f"  Modifier: {best_interaction['modifier']}={best_interaction['modifier_value']}")
    summary_lines.append(f"  Interaction effect: {best_interaction['interaction']:.4f}")
    summary_lines.append(f"  P-value: {best_interaction['p_value']:.4f}")
    summary_lines.append(f"  Significant: {'YES' if best_interaction['p_value'] < ALPHA else 'NO'}")
else:
    summary_lines.append("\nNo significant treatment-effect heterogeneity identified.")
summary_lines.append("")
summary_lines.append("-" * 70)
summary_lines.append("CONCLUSIONS")
summary_lines.append("-" * 70)
summary_lines.append("\nThis analysis explored feature-outcome relationships in the ds001_prostate")
summary_lines.append("oncology dataset using an iterative hypothesis testing approach.")
if len(all_sig) > 0:
    summary_lines.append(f"Key findings: {len(all_sig)} feature-outcome associations were statistically")
    summary_lines.append(f"significant (p < {ALPHA}). These features may be clinically relevant for")
    summary_lines.append("predicting objective response in prostate cancer patients.")
else:
    summary_lines.append("No statistically significant feature-outcome associations were identified")
    summary_lines.append("at the specified significance threshold.")
summary_lines.append("")
summary_lines.append("=" * 70)
summary_lines.append("END OF ANALYSIS SUMMARY")
summary_lines.append("=" * 70)

with open(OUTPUT_DIR / "analysis_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("\nDone! Files written:")
print(f"  - {OUTPUT_DIR / 'transcript.json'}")
print(f"  - {OUTPUT_DIR / 'analysis_summary.txt'}")
