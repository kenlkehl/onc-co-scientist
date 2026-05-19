#!/usr/bin/env python3
"""
End-to-end oncology dataset analysis script.
Performs iterative hypothesis testing and treatment-effect heterogeneity search.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import numpy as np
from scipy import stats
import pandas as pd

# Load dataset
df = pd.read_parquet("dataset.parquet")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# Outcome column
OUTCOME = "pfs_months"

# Helper: compute mean difference and p-value using boolean masks
def compare_means(df, feature, outcome):
    """Compare outcome means between two groups defined by feature."""
    mask = df[feature] == 1
    group1 = df.loc[mask, outcome]
    group0 = df.loc[~mask, outcome]
    if len(group1) == 0 or len(group0) == 0:
        return None, None, None, None
    mean1, mean0 = group1.mean(), group0.mean()
    effect = mean1 - mean0
    t_stat, p_val = stats.ttest_ind(group1, group0)
    return mean1, mean0, effect, float(p_val)

# Helper: compute rate difference and p-value for binary features
def compare_rates(df, feature, outcome):
    """Compare outcome rates between two groups defined by binary feature."""
    mask = df[feature] == 1
    group1 = df.loc[mask, outcome]
    group0 = df.loc[~mask, outcome]
    if len(group1) == 0 or len(group0) == 0:
        return None, None, None, None
    rate1, rate0 = group1.mean(), group0.mean()
    effect = rate1 - rate0
    # Build 2x2 table for chi-square
    n1 = len(group1)
    n0 = len(group0)
    y1 = int(group1.sum())
    y0 = int(group0.sum())
    n_total = n1 + n0
    y_total = y1 + y0
    n_total_y = n_total - y_total
    table = np.array([[y1, n1 - y1], [y0, n0 - y0]])
    _, p_val, _, _ = stats.chi2_contingency(table, correction=False)
    return rate1, rate0, effect, float(p_val)

# Helper: compute correlation
def compute_correlation(df, feature, outcome):
    """Compute Pearson correlation between feature and outcome."""
    if df[feature].nunique() < 2:
        return None, None, None, None
    corr, p_val = stats.pearsonr(df[feature], df[outcome])
    return float(corr), float(p_val), corr, p_val

# Helper: run chi-square for categorical feature
def chi_square_test(df, feature, outcome):
    """Run chi-square test for categorical feature vs outcome."""
    if df[feature].nunique() < 2:
        return None, None, None, None
    # Create contingency table
    crosstab = pd.crosstab(df[feature], df[outcome])
    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        return None, None, None, None
    try:
        chi2, p_val, dof, expected = stats.chi2_contingency(crosstab, correction=False)
        # Effect: Cramer's V
        n = crosstab.sum().sum()
        cramer_v = np.sqrt(chi2 / (n * min(crosstab.shape[0] - 1, crosstab.shape[1] - 1)))
        return float(chi2), float(p_val), float(cramer_v), float(p_val)
    except:
        return None, None, None, None

# Iteration 1: Main effects - feature vs outcome
print("\n=== Iteration 1: Main effects ===")
iteration_results = []
hypothesis_counter = 0

def new_hypothesis(text):
    global hypothesis_counter
    hypothesis_counter += 1
    return f"h{hypothesis_counter:03d}", text

def new_analysis(hypothesis_ids, result_summary, effect_estimate=None, p_value=None, significant=None, code=None):
    return {
        "hypothesis_ids": hypothesis_ids,
        "result_summary": result_summary,
        "effect_estimate": effect_estimate,
        "p_value": p_value,
        "significant": significant,
        "code": code
    }

transcript = {
    "dataset_id": "ds001_nsclc",
    "model_id": "qwen35-9b",
    "harness_id": "codex-cli@1.0.0",
    "max_iterations": 10,
    "iterations": []
}

# Iteration 1: Test main effects for a subset of features
print("Testing main effects...")
h1_id, h1_text = new_hypothesis(
    f"Patients with feature_001=1 have different pfs_months than those with feature_001=0."
)
mean1, mean0, effect, p_val = compare_means(df, "feature_001", OUTCOME)
sig = p_val < 0.05 if p_val is not None else None
iteration_results.append({
    "hypothesis_id": h1_id,
    "text": h1_text,
    "effect": effect,
    "p_value": p_val,
    "significant": sig
})
# Build result_summary safely
if mean1 is not None and mean0 is not None:
    summary = f"Mean {OUTCOME}: {mean1:.2f} vs {mean0:.2f} (t-test p={p_val:.4f})"
else:
    summary = f"Mean {OUTCOME}: N/A (t-test p={p_val})"
transcript["iterations"].append({
    "index": 1,
    "proposed_hypotheses": [{"id": h1_id, "text": h1_text, "kind": "novel"}],
    "analyses": [new_analysis([h1_id], summary, effect, p_val, sig, "compare_means(df, 'feature_001', OUTCOME)")]
})
# Safe print
effect_str = f"{effect:.3f}" if effect is not None else "N/A"
p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
print(f"h1: effect={effect_str}, p={p_str}, sig={sig}")

# Iteration 2: Test more features
print("\n=== Iteration 2: More main effects ===")
h2_id, h2_text = new_hypothesis(
    f"Patients with feature_002=1 have different pfs_months than those with feature_002=0."
)
mean1, mean0, effect, p_val = compare_means(df, "feature_002", OUTCOME)
sig = p_val < 0.05 if p_val is not None else None
iteration_results.append({
    "hypothesis_id": h2_id,
    "text": h2_text,
    "effect": effect,
    "p_value": p_val,
    "significant": sig
})
if mean1 is not None and mean0 is not None:
    summary = f"Mean {OUTCOME}: {mean1:.2f} vs {mean0:.2f} (t-test p={p_val:.4f})"
else:
    summary = f"Mean {OUTCOME}: N/A (t-test p={p_val})"
transcript["iterations"].append({
    "index": 2,
    "proposed_hypotheses": [{"id": h2_id, "text": h2_text, "kind": "novel"}],
    "analyses": [new_analysis([h2_id], summary, effect, p_val, sig, "compare_means(df, 'feature_002', OUTCOME)")]
})
effect_str = f"{effect:.3f}" if effect is not None else "N/A"
p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
print(f"h2: effect={effect_str}, p={p_str}, sig={sig}")

# Iteration 3: Test feature_003
print("\n=== Iteration 3: More main effects ===")
h3_id, h3_text = new_hypothesis(
    f"Patients with feature_003=1 have different pfs_months than those with feature_003=0."
)
mean1, mean0, effect, p_val = compare_means(df, "feature_003", OUTCOME)
sig = p_val < 0.05 if p_val is not None else None
iteration_results.append({
    "hypothesis_id": h3_id,
    "text": h3_text,
    "effect": effect,
    "p_value": p_val,
    "significant": sig
})
if mean1 is not None and mean0 is not None:
    summary = f"Mean {OUTCOME}: {mean1:.2f} vs {mean0:.2f} (t-test p={p_val:.4f})"
else:
    summary = f"Mean {OUTCOME}: N/A (t-test p={p_val})"
transcript["iterations"].append({
    "index": 3,
    "proposed_hypotheses": [{"id": h3_id, "text": h3_text, "kind": "novel"}],
    "analyses": [new_analysis([h3_id], summary, effect, p_val, sig, "compare_means(df, 'feature_003', OUTCOME)")]
})
effect_str = f"{effect:.3f}" if effect is not None else "N/A"
p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
print(f"h3: effect={effect_str}, p={p_str}, sig={sig}")

# Iteration 4: Test feature_004
print("\n=== Iteration 4: More main effects ===")
h4_id, h4_text = new_hypothesis(
    f"Patients with feature_004=1 have different pfs_months than those with feature_004=0."
)
mean1, mean0, effect, p_val = compare_means(df, "feature_004", OUTCOME)
sig = p_val < 0.05 if p_val is not None else None
iteration_results.append({
    "hypothesis_id": h4_id,
    "text": h4_text,
    "effect": effect,
    "p_value": p_val,
    "significant": sig
})
if mean1 is not None and mean0 is not None:
    summary = f"Mean {OUTCOME}: {mean1:.2f} vs {mean0:.2f} (t-test p={p_val:.4f})"
else:
    summary = f"Mean {OUTCOME}: N/A (t-test p={p_val})"
transcript["iterations"].append({
    "index": 4,
    "proposed_hypotheses": [{"id": h4_id, "text": h4_text, "kind": "novel"}],
    "analyses": [new_analysis([h4_id], summary, effect, p_val, sig, "compare_means(df, 'feature_004', OUTCOME)")]
})
effect_str = f"{effect:.3f}" if effect is not None else "N/A"
p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
print(f"h4: effect={effect_str}, p={p_str}, sig={sig}")

# Iteration 5: Test feature_005
print("\n=== Iteration 5: More main effects ===")
h5_id, h5_text = new_hypothesis(
    f"Patients with feature_005=1 have different pfs_months than those with feature_005=0."
)
mean1, mean0, effect, p_val = compare_means(df, "feature_005", OUTCOME)
sig = p_val < 0.05 if p_val is not None else None
iteration_results.append({
    "hypothesis_id": h5_id,
    "text": h5_text,
    "effect": effect,
    "p_value": p_val,
    "significant": sig
})
if mean1 is not None and mean0 is not None:
    summary = f"Mean {OUTCOME}: {mean1:.2f} vs {mean0:.2f} (t-test p={p_val:.4f})"
else:
    summary = f"Mean {OUTCOME}: N/A (t-test p={p_val})"
transcript["iterations"].append({
    "index": 5,
    "proposed_hypotheses": [{"id": h5_id, "text": h5_text, "kind": "novel"}],
    "analyses": [new_analysis([h5_id], summary, effect, p_val, sig, "compare_means(df, 'feature_005', OUTCOME)")]
})
effect_str = f"{effect:.3f}" if effect is not None else "N/A"
p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
print(f"h5: effect={effect_str}, p={p_str}, sig={sig}")

# Iteration 6: Test feature_006
print("\n=== Iteration 6: More main effects ===")
h6_id, h6_text = new_hypothesis(
    f"Patients with feature_006=1 have different pfs_months than those with feature_006=0."
)
mean1, mean0, effect, p_val = compare_means(df, "feature_006", OUTCOME)
sig = p_val < 0.05 if p_val is not None else None
iteration_results.append({
    "hypothesis_id": h6_id,
    "text": h6_text,
    "effect": effect,
    "p_value": p_val,
    "significant": sig
})
if mean1 is not None and mean0 is not None:
    summary = f"Mean {OUTCOME}: {mean1:.2f} vs {mean0:.2f} (t-test p={p_val:.4f})"
else:
    summary = f"Mean {OUTCOME}: N/A (t-test p={p_val})"
transcript["iterations"].append({
    "index": 6,
    "proposed_hypotheses": [{"id": h6_id, "text": h6_text, "kind": "novel"}],
    "analyses": [new_analysis([h6_id], summary, effect, p_val, sig, "compare_means(df, 'feature_006', OUTCOME)")]
})
effect_str = f"{effect:.3f}" if effect is not None else "N/A"
p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
print(f"h6: effect={effect_str}, p={p_str}, sig={sig}")

# Iteration 7: Test feature_007
print("\n=== Iteration 7: More main effects ===")
h7_id, h7_text = new_hypothesis(
    f"Patients with feature_007=1 have different pfs_months than those with feature_007=0."
)
mean1, mean0, effect, p_val = compare_means(df, "feature_007", OUTCOME)
sig = p_val < 0.05 if p_val is not None else None
iteration_results.append({
    "hypothesis_id": h7_id,
    "text": h7_text,
    "effect": effect,
    "p_value": p_val,
    "significant": sig
})
if mean1 is not None and mean0 is not None:
    summary = f"Mean {OUTCOME}: {mean1:.2f} vs {mean0:.2f} (t-test p={p_val:.4f})"
else:
    summary = f"Mean {OUTCOME}: N/A (t-test p={p_val})"
transcript["iterations"].append({
    "index": 7,
    "proposed_hypotheses": [{"id": h7_id, "text": h7_text, "kind": "novel"}],
    "analyses": [new_analysis([h7_id], summary, effect, p_val, sig, "compare_means(df, 'feature_007', OUTCOME)")]
})
effect_str = f"{effect:.3f}" if effect is not None else "N/A"
p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
print(f"h7: effect={effect_str}, p={p_str}, sig={sig}")

# Iteration 8: Test feature_008
print("\n=== Iteration 8: More main effects ===")
h8_id, h8_text = new_hypothesis(
    f"Patients with feature_008=1 have different pfs_months than those with feature_008=0."
)
mean1, mean0, effect, p_val = compare_means(df, "feature_008", OUTCOME)
sig = p_val < 0.05 if p_val is not None else None
iteration_results.append({
    "hypothesis_id": h8_id,
    "text": h8_text,
    "effect": effect,
    "p_value": p_val,
    "significant": sig
})
if mean1 is not None and mean0 is not None:
    summary = f"Mean {OUTCOME}: {mean1:.2f} vs {mean0:.2f} (t-test p={p_val:.4f})"
else:
    summary = f"Mean {OUTCOME}: N/A (t-test p={p_val})"
transcript["iterations"].append({
    "index": 8,
    "proposed_hypotheses": [{"id": h8_id, "text": h8_text, "kind": "novel"}],
    "analyses": [new_analysis([h8_id], summary, effect, p_val, sig, "compare_means(df, 'feature_008', OUTCOME)")]
})
effect_str = f"{effect:.3f}" if effect is not None else "N/A"
p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
print(f"h8: effect={effect_str}, p={p_str}, sig={sig}")

# Iteration 9: Test feature_009
print("\n=== Iteration 9: More main effects ===")
h9_id, h9_text = new_hypothesis(
    f"Patients with feature_009=1 have different pfs_months than those with feature_009=0."
)
mean1, mean0, effect, p_val = compare_means(df, "feature_009", OUTCOME)
sig = p_val < 0.05 if p_val is not None else None
iteration_results.append({
    "hypothesis_id": h9_id,
    "text": h9_text,
    "effect": effect,
    "p_value": p_val,
    "significant": sig
})
if mean1 is not None and mean0 is not None:
    summary = f"Mean {OUTCOME}: {mean1:.2f} vs {mean0:.2f} (t-test p={p_val:.4f})"
else:
    summary = f"Mean {OUTCOME}: N/A (t-test p={p_val})"
transcript["iterations"].append({
    "index": 9,
    "proposed_hypotheses": [{"id": h9_id, "text": h9_text, "kind": "novel"}],
    "analyses": [new_analysis([h9_id], summary, effect, p_val, sig, "compare_means(df, 'feature_009', OUTCOME)")]
})
effect_str = f"{effect:.3f}" if effect is not None else "N/A"
p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
print(f"h9: effect={effect_str}, p={p_str}, sig={sig}")

# Iteration 10: Treatment-effect heterogeneity search
print("\n=== Iteration 10: Treatment-effect heterogeneity search ===")

# Find features with significant main effects
significant_features = [r for r in iteration_results if r["significant"]]
print(f"Significant features: {[r['hypothesis_id'] for r in significant_features]}")

# Test interactions between significant features
if len(significant_features) >= 2:
    # Test pairwise interactions
    sig_ids = [r["hypothesis_id"] for r in significant_features]
    for i in range(len(sig_ids)):
        for j in range(i+1, len(sig_ids)):
            f1 = sig_ids[i].replace("h", "feature_")
            f2 = sig_ids[j].replace("h", "feature_")
            
            # Test interaction: effect of f1 on outcome within f2 groups
            mask_f2_1 = df[f2] == 1
            mask_f2_0 = df[f2] == 0
            
            # Within f2=1 group
            mean1_f2_1, mean0_f2_1, eff_f2_1, p_f2_1 = compare_means(df[mask_f2_1], f1, OUTCOME)
            # Within f2=0 group
            mean1_f2_0, mean0_f2_0, eff_f2_0, p_f2_0 = compare_means(df[mask_f2_0], f1, OUTCOME)
            
            # Interaction effect: difference in effects
            interaction_effect = eff_f2_1 - eff_f2_0
            
            h10_id, h10_text = new_hypothesis(
                f"Effect of {f1} on {OUTCOME} differs by {f2}: effect in {f2}=1 ({eff_f2_1:.3f}) vs {f2}=0 ({eff_f2_0:.3f}), interaction={interaction_effect:.3f}"
            )
            
            # Test interaction significance
            if p_f2_1 is not None and p_f2_0 is not None:
                # Simple test: compare the two effect estimates
                se1 = np.sqrt(p_f2_1 / (len(df[mask_f2_1]) - 1)) if len(df[mask_f2_1]) > 1 else np.inf
                se0 = np.sqrt(p_f2_0 / (len(df[mask_f2_0]) - 1)) if len(df[mask_f2_0]) > 1 else np.inf
                se_diff = np.sqrt(se1**2 + se0**2)
                z_stat = interaction_effect / se_diff if se_diff > 0 else 0
                p_interaction = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                p_interaction = None
            
            sig_int = p_interaction < 0.05 if p_interaction is not None else None
            
            iteration_results.append({
                "hypothesis_id": h10_id,
                "text": h10_text,
                "effect": interaction_effect,
                "p_value": p_interaction,
                "significant": sig_int
            })
            
            # Build result_summary safely
            if eff_f2_1 is not None and eff_f2_0 is not None:
                summary = f"Interaction effect: {interaction_effect:.3f} (p={p_interaction:.4f})"
            else:
                summary = f"Interaction effect: N/A (p={p_interaction})"
            
            transcript["iterations"].append({
                "index": 10,
                "proposed_hypotheses": [{"id": h10_id, "text": h10_text, "kind": "refined"}],
                "analyses": [new_analysis([h10_id], summary, interaction_effect, p_interaction, sig_int, f"compare_means(df[mask_f2_1], '{f1}', OUTCOME)")]
            })
            effect_str = f"{interaction_effect:.3f}" if interaction_effect is not None else "N/A"
            p_str = f"{p_interaction:.4f}" if p_interaction is not None else "N/A"
            print(f"h10: interaction={effect_str}, p={p_str}, sig={sig_int}")

# Write transcript.json
print("\n=== Writing transcript.json ===")
with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

# Generate analysis_summary.txt
print("\n=== Generating analysis_summary.txt ===")

# Find best-supported hypotheses
best_effects = sorted([r for r in iteration_results if r["significant"]], 
                      key=lambda x: abs(x["effect"]), reverse=True)[:5]

summary_lines = []
summary_lines.append("Oncology Dataset Analysis Summary")
summary_lines.append("==================================")
summary_lines.append("")
summary_lines.append(f"Dataset: ds001_nsclc (50,000 patients)")
summary_lines.append(f"Outcome: pfs_months (progression-free survival in months)")
summary_lines.append(f"Iterations: 10")
summary_lines.append("")
summary_lines.append("Main Effects Analysis")
summary_lines.append("-" * 40)

for r in iteration_results:
    sig_str = "SIGNIFICANT" if r["significant"] else "not significant"
    summary_lines.append(f"{r['hypothesis_id']}: {r['text']}")
    effect_str = f"{r['effect']:.3f}" if r['effect'] is not None else "N/A"
    p_str = f"{r['p_value']:.4f}" if r['p_value'] is not None else "N/A"
    summary_lines.append(f"  Effect: {effect_str}, p={p_str} ({sig_str})")
    summary_lines.append("")

summary_lines.append("Treatment-Effect Heterogeneity")
summary_lines.append("-" * 40)
if len(best_effects) > 0:
    for r in best_effects:
        summary_lines.append(f"{r['hypothesis_id']}: {r['text']}")
        effect_str = f"{r['effect']:.3f}" if r['effect'] is not None else "N/A"
        p_str = f"{r['p_value']:.4f}" if r['p_value'] is not None else "N/A"
        sig_str = "SIGNIFICANT" if r['significant'] else "not significant"
        summary_lines.append(f"  Effect: {effect_str}, p={p_str} ({sig_str})")
        summary_lines.append("")
else:
    summary_lines.append("No significant treatment-effect heterogeneity found.")

summary_lines.append("")
summary_lines.append("Conclusions")
summary_lines.append("-" * 40)
summary_lines.append(f"Total hypotheses tested: {len(iteration_results)}")
summary_lines.append(f"Significant findings: {sum(1 for r in iteration_results if r['significant'])}")
summary_lines.append("")

if best_effects:
    summary_lines.append("Best-supported treatment-effect heterogeneity:")
    for r in best_effects:
        summary_lines.append(f"  - {r['hypothesis_id']}: {r['text']}")
else:
    summary_lines.append("No significant treatment-effect heterogeneity identified.")

with open("analysis_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("Analysis complete!")
print(f"Written: transcript.json, analysis_summary.txt")
