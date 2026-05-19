#!/usr/bin/env python3
"""
End-to-end oncology dataset analysis for ds001_breast.
Performs iterative hypothesis testing with treatment-effect heterogeneity searches.
Outputs transcript.json and analysis_summary.txt.
"""

import json
import numpy as np
from scipy import stats
import pandas as pd

# Load dataset
df = pd.read_parquet("dataset.parquet")
print(f"Loaded {len(df)} records with columns: {list(df.columns)}")

# Column definitions from dataset_description.md
FEATURES = [
    "age_years", "sex_female", "ecog_ps", "stage_iv", "has_brain_mets",
    "node_positive", "postmenopausal", "er_positive", "pr_positive", "her2_positive",
    "her2_low", "brca1_mutation", "brca2_mutation", "pik3ca_mutation", "ki67_pct",
    "tumor_size_cm", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l",
    "nlr", "treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
    "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
    "hemoglobin_g_dl", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
    "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl", "sodium_meq_l",
    "potassium_meq_l", "calcium_mg_dl"
]
OUTCOMES = ["pfs_months"]
TREATMENTS = ["treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
              "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab"]

def to_jsonable(obj):
    """Convert numpy types and other non-JSON types to plain Python types."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        if np.isnan(obj):
            return None
        return obj
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)

def run_ttest(df, feature, value, outcome):
    """Run t-test comparing outcome means between two groups."""
    mask = df[feature] == value
    group1 = df.loc[mask, outcome].values
    group0 = df.loc[~mask, outcome].values
    if len(group1) < 2 or len(group0) < 2:
        return None, None, None, None
    t_stat, p_value = stats.ttest_ind(group1, group0, equal_var=False)
    effect = group1.mean() - group0.mean()
    return effect, p_value, t_stat, len(group1), len(group0)

def run_correlation(df, feature1, feature2, outcome):
    """Run correlation analysis between two features and outcome."""
    mask = df[feature1].notna() & df[feature2].notna() & df[outcome].notna()
    f1 = df.loc[mask, feature1].values
    f2 = df.loc[mask, feature2].values
    o = df.loc[mask, outcome].values
    if len(f1) < 3:
        return None, None, None, None
    corr1, p1 = stats.pearsonr(f1, o)
    corr2, p2 = stats.pearsonr(f2, o)
    return corr1, p1, corr2, p2

def run_stratified_comparison(df, treatment, feature, value, outcome):
    """Run stratified treatment comparison within feature=value subgroup."""
    mask = df[feature] == value
    treatment_col = df.loc[mask, treatment].values
    outcome_col = df.loc[mask, outcome].values
    if len(np.unique(treatment_col)) < 2:
        return None, None, None, None
    treatment_1 = outcome_col[treatment_col == 1]
    treatment_0 = outcome_col[treatment_col == 0]
    if len(treatment_1) < 2 or len(treatment_0) < 2:
        return None, None, None, None
    t_stat, p_value = stats.ttest_ind(treatment_1, treatment_0, equal_var=False)
    effect = treatment_1.mean() - treatment_0.mean()
    return effect, p_value, t_stat, len(treatment_1), len(treatment_0)

def run_interaction_regression(df, treatment, feature, value, outcome):
    """Run regression with treatment-by-feature interaction."""
    mask = df[feature] == value
    treatment_col = df.loc[mask, treatment].values
    feature_col = np.ones(len(treatment_col))
    outcome_col = df.loc[mask, outcome].values
    if len(np.unique(treatment_col)) < 2:
        return None, None, None, None
    interaction = treatment_col * feature_col
    X = np.column_stack([np.ones(len(treatment_col)), treatment_col, interaction])
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, outcome_col, rcond=None)
        interaction_effect = coeffs[2]
        ss_res = np.sum((outcome_col - X @ coeffs) ** 2)
        ss_tot = np.sum((outcome_col - outcome_col.mean()) ** 2)
        ss_reg = ss_tot - ss_res
        f_stat = (ss_reg / 2) / (ss_res / (len(outcome_col) - 3))
        p_value = 1 - stats.f.cdf(f_stat, 2, len(outcome_col) - 3)
        return interaction_effect, p_value, f_stat, len(np.unique(treatment_col))
    except:
        return None, None, None, None

def run_subgroup_discovery(df, treatment, outcome, max_subgroups=20):
    """Discover subgroups where treatment effect is strongest."""
    treatment_col = df[treatment].values
    outcome_col = df[outcome].values
    if len(np.unique(treatment_col)) < 2:
        return []
    treatment_1 = outcome_col[treatment_col == 1]
    treatment_0 = outcome_col[treatment_col == 0]
    if len(treatment_1) < 2 or len(treatment_0) < 2:
        return []
    modifiers = []
    for feature in FEATURES:
        if feature in [treatment, outcome]:
            continue
        mask = df[feature].notna()
        f = df.loc[mask, feature].values
        o1 = df.loc[mask & (df[treatment] == 1), outcome].values
        o0 = df.loc[mask & (df[treatment] == 0), outcome].values
        if len(np.unique(f)) < 2 or len(np.unique(o1)) < 2 or len(np.unique(o0)) < 2:
            continue
        effects = []
        for val in np.unique(f):
            m = f == val
            if np.sum(m) < 2:
                continue
            e1 = o1[m].mean()
            e0 = o0[m].mean()
            effects.append(e1 - e0)
        if len(effects) < 2:
            continue
        corr, p = stats.pearsonr(f, effects)
        modifiers.append((feature, corr, p))
    modifiers.sort(key=lambda x: -abs(x[1]))
    return modifiers[:max_subgroups]

def run_treatment_effect_heterogeneity(df, treatment, outcome):
    """Systematic search for treatment effect heterogeneity."""
    treatment_col = df[treatment].values
    outcome_col = df[outcome].values
    if len(np.unique(treatment_col)) < 2:
        return []
    treatment_1 = outcome_col[treatment_col == 1]
    treatment_0 = outcome_col[treatment_col == 0]
    if len(treatment_1) < 2 or len(treatment_0) < 2:
        return []
    overall_effect = treatment_1.mean() - treatment_0.mean()
    overall_p = stats.ttest_ind(treatment_1, treatment_0, equal_var=False).pvalue
    modifiers = run_subgroup_discovery(df, treatment, outcome, max_subgroups=15)
    results = []
    for feature, corr, p in modifiers:
        effect, p_val, _, _, _ = run_stratified_comparison(df, treatment, feature, None, outcome)
        if effect is not None:
            results.append({
                "feature": feature,
                "correlation_with_effect": corr,
                "p_correlation": p,
                "stratified_effect": effect,
                "stratified_p": p_val
            })
    interaction_results = []
    for feature, corr, p in modifiers[:5]:
        effect, p_val, _, _ = run_interaction_regression(df, treatment, feature, None, outcome)
        if effect is not None:
            interaction_results.append({
                "feature": feature,
                "interaction_effect": effect,
                "interaction_p": p_val
            })
    return results, interaction_results, overall_effect, overall_p

def run_main_effect_analysis(df, feature, outcome):
    """Run main effect analysis for a feature-outcome pair."""
    if feature in OUTCOMES:
        return None, None, None, None
    if feature in TREATMENTS:
        treatment = feature
        treatment_col = df[treatment].values
        outcome_col = df[outcome].values
        if len(np.unique(treatment_col)) < 2:
            return None, None, None, None
        treatment_1 = outcome_col[treatment_col == 1]
        treatment_0 = outcome_col[treatment_col == 0]
        if len(treatment_1) < 2 or len(treatment_0) < 2:
            return None, None, None, None
        t_stat, p_value = stats.ttest_ind(treatment_1, treatment_0, equal_var=False)
        effect = treatment_1.mean() - treatment_0.mean()
        return effect, p_value, t_stat, len(treatment_1), len(treatment_0)
    else:
        return run_ttest(df, feature, None, outcome)

def run_categorical_comparison(df, feature, outcome):
    """Run comparison for categorical features."""
    if feature in OUTCOMES:
        return None, None, None, None
    if feature in TREATMENTS:
        treatment = feature
        treatment_col = df[treatment].values
        outcome_col = df[outcome].values
        if len(np.unique(treatment_col)) < 2:
            return None, None, None, None
        treatment_1 = outcome_col[treatment_col == 1]
        treatment_0 = outcome_col[treatment_col == 0]
        if len(treatment_1) < 2 or len(treatment_0) < 2:
            return None, None, None, None
        median_outcome = treatment_0.median()
        table = pd.crosstab(
            [df[treatment] == 1, df[outcome] > median_outcome],
            rownames=["treatment=1", "treatment=0"],
            colnames=[">median", "<=median"]
        )
        chi2, p_value, dof, expected = stats.chi2_contingency(table.values, correction=False)
        return chi2, p_value, dof, expected
    else:
        mask = df[feature].notna()
        f = df.loc[mask, feature].values
        o = df.loc[mask, outcome].values
        if len(np.unique(f)) < 2:
            return None, None, None, None
        groups = [o[f == v] for v in np.unique(f)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            return None, None, None, None
        f_stat, p_value = stats.f_oneway(*groups)
        return f_stat, p_value, len(groups), groups

def run_mediation_analysis_simple(df, treatment, mediator, outcome):
    """Simple mediation analysis."""
    treatment_col = df[treatment].values
    mediator_col = df[mediator].values
    outcome_col = df[outcome].values
    if len(np.unique(treatment_col)) < 2:
        return None, None, None, None
    X1 = np.column_stack([np.ones(len(treatment_col)), treatment_col])
    try:
        coeffs1, _, _, _ = np.linalg.lstsq(X1, mediator_col, rcond=None)
        a_path = coeffs1[1]
        X2 = np.column_stack([np.ones(len(mediator_col)), mediator_col, treatment_col])
        coeffs2, _, _, _ = np.linalg.lstsq(X2, outcome_col, rcond=None)
        b_path = coeffs2[1]
        X3 = np.column_stack([np.ones(len(treatment_col)), treatment_col])
        coeffs3, _, _, _ = np.linalg.lstsq(X3, outcome_col, rcond=None)
        c_path = coeffs3[1]
        c_prime = c_path - a_path * b_path
        return a_path, b_path, c_path, c_prime
    except:
        return None, None, None, None

# Initialize transcript
transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "qwen35-9b-caa-l24-neg002",
    "harness_id": "codex-cli@2026-05-19",
    "max_iterations": 10,
    "iterations": []
}

# Iteration 1: Main effects for treatments
print("\n=== Iteration 1: Main effects for treatments ===")
iteration1_hypotheses = []
iteration1_analyses = []

for treatment in TREATMENTS:
    hypothesis_id = f"h1_{treatment}"
    hypothesis_text = f"Patients assigned to {treatment} have different pfs_months compared to those not assigned to {treatment}."
    iteration1_hypotheses.append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    effect, p_value, _, _, _ = run_main_effect_analysis(df, treatment, OUTCOMES[0])
    significant = p_value < 0.05 if p_value is not None else False
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Mean pfs_months: {treatment}=1: {effect:.2f} vs {treatment}=0: {effect:.2f} (t-test p={p_value:.4f}).",
        "effect_estimate": float(effect) if effect is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "significant": significant
    }
    iteration1_analyses.append(analysis)

transcript["iterations"].append({
    "index": 1,
    "proposed_hypotheses": iteration1_hypotheses,
    "analyses": iteration1_analyses
})

# Iteration 2: Feature-outcome correlations
print("\n=== Iteration 2: Feature-outcome correlations ===")
iteration2_hypotheses = []
iteration2_analyses = []

key_features = ["age_years", "ecog_ps", "stage_iv", "node_positive", "tumor_size_cm",
                "ki67_pct", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "nlr"]

for feature in key_features:
    hypothesis_id = f"h2_{feature}"
    hypothesis_text = f"{feature} is correlated with pfs_months."
    iteration2_hypotheses.append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    corr, p_value, _, _ = run_correlation(df, feature, None, OUTCOMES[0])
    significant = p_value < 0.05 if p_value is not None else False
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Correlation between {feature} and pfs_months: r={corr:.4f} (p={p_value:.4f}).",
        "effect_estimate": float(corr) if corr is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "significant": significant
    }
    iteration2_analyses.append(analysis)

transcript["iterations"].append({
    "index": 2,
    "proposed_hypotheses": iteration2_hypotheses,
    "analyses": iteration2_analyses
})

# Iteration 3: Treatment-by-feature interactions
print("\n=== Iteration 3: Treatment-by-feature interactions ===")
iteration3_hypotheses = []
iteration3_analyses = []

for treatment in TREATMENTS:
    modifiers = run_subgroup_discovery(df, treatment, OUTCOMES[0], max_subgroups=5)
    
    for feature, corr, p in modifiers[:3]:
        hypothesis_id = f"h3_{treatment}_{feature}"
        hypothesis_text = f"The effect of {treatment} on pfs_months is modified by {feature} (correlation with effect: {corr:.4f}, p={p:.4f})."
        iteration3_hypotheses.append({
            "id": hypothesis_id,
            "text": hypothesis_text,
            "kind": "novel"
        })
        
        effect, p_val, _, _ = run_interaction_regression(df, treatment, feature, None, OUTCOMES[0])
        significant = p_val < 0.05 if p_val is not None else False
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Interaction between {treatment} and {feature}: effect={effect:.4f} (p={p_val:.4f}).",
            "effect_estimate": float(effect) if effect is not None else None,
            "p_value": float(p_val) if p_val is not None else None,
            "significant": significant
        }
        iteration3_analyses.append(analysis)

transcript["iterations"].append({
    "index": 3,
    "proposed_hypotheses": iteration3_hypotheses,
    "analyses": iteration3_analyses
})

# Iteration 4: Stratified treatment comparisons
print("\n=== Iteration 4: Stratified treatment comparisons ===")
iteration4_hypotheses = []
iteration4_analyses = []

stratification_features = ["stage_iv", "node_positive", "er_positive", "pr_positive", "her2_positive"]

for feature in stratification_features:
    hypothesis_id = f"h4_{feature}"
    hypothesis_text = f"The effect of treatments on pfs_months differs by {feature}."
    iteration4_hypotheses.append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    for treatment in TREATMENTS:
        effect, p_val, _, _, _ = run_stratified_comparison(df, treatment, feature, None, OUTCOMES[0])
        if effect is not None:
            significant = p_val < 0.05 if p_val is not None else False
            analysis = {
                "hypothesis_ids": [f"h4_{feature}_{treatment}"],
                "result_summary": f"Stratified {treatment} effect within {feature}: effect={effect:.4f} (p={p_val:.4f}).",
                "effect_estimate": float(effect),
                "p_value": float(p_val),
                "significant": significant
            }
            iteration4_analyses.append(analysis)

transcript["iterations"].append({
    "index": 4,
    "proposed_hypotheses": iteration4_hypotheses,
    "analyses": iteration4_analyses
})

# Iteration 5: Treatment-effect heterogeneity search
print("\n=== Iteration 5: Treatment-effect heterogeneity search ===")
iteration5_hypotheses = []
iteration5_analyses = []

for treatment in TREATMENTS:
    hypothesis_id = f"h5_{treatment}"
    hypothesis_text = f"Systematic search for treatment-effect heterogeneity for {treatment} on pfs_months."
    iteration5_hypotheses.append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    modifiers, interaction_results, overall_effect, overall_p = run_treatment_effect_heterogeneity(df, treatment, OUTCOMES[0])
    
    for modifier in modifiers[:3]:
        hypothesis_id_mod = f"h5_{treatment}_{modifier['feature']}"
        hypothesis_text_mod = f"The effect of {treatment} on pfs_months is modified by {modifier['feature']} (correlation={modifier['correlation_with_effect']:.4f}, p={modifier['p_correlation']:.4f})."
        iteration5_hypotheses.append({
            "id": hypothesis_id_mod,
            "text": hypothesis_text_mod,
            "kind": "refined"
        })
        
        effect = modifier.get('stratified_effect')
        p_val = modifier.get('stratified_p')
        significant = p_val < 0.05 if p_val is not None else False
        
        analysis = {
            "hypothesis_ids": [hypothesis_id_mod],
            "result_summary": f"Stratified {treatment} effect within {modifier['feature']}: effect={effect:.4f} (p={p_val:.4f}).",
            "effect_estimate": float(effect) if effect is not None else None,
            "p_value": float(p_val) if p_val is not None else None,
            "significant": significant
        }
        iteration5_analyses.append(analysis)
    
    for interaction in interaction_results[:2]:
        hypothesis_id_int = f"h5_{treatment}_{interaction['feature']}_interaction"
        hypothesis_text_int = f"Interaction between {treatment} and {interaction['feature']} modifies pfs_months (effect={interaction['interaction_effect']:.4f}, p={interaction['interaction_p']:.4f})."
        iteration5_hypotheses.append({
            "id": hypothesis_id_int,
            "text": hypothesis_text_int,
            "kind": "refined"
        })
        
        analysis = {
            "hypothesis_ids": [hypothesis_id_int],
            "result_summary": f"Interaction effect: {interaction['interaction_effect']:.4f} (p={interaction['interaction_p']:.4f}).",
            "effect_estimate": float(interaction['interaction_effect']),
            "p_value": float(interaction['interaction_p']),
            "significant": interaction['interaction_p'] < 0.05
        }
        iteration5_analyses.append(analysis)
    
    analysis_overall = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Overall {treatment} effect: {overall_effect:.4f} (p={overall_p:.4f}).",
        "effect_estimate": float(overall_effect),
        "p_value": float(overall_p),
        "significant": overall_p < 0.05
    }
    iteration5_analyses.insert(0, analysis_overall)

transcript["iterations"].append({
    "index": 5,
    "proposed_hypotheses": iteration5_hypotheses,
    "analyses": iteration5_analyses
})

# Iteration 6: Mediation analysis
print("\n=== Iteration 6: Mediation analysis ===")
iteration6_hypotheses = []
iteration6_analyses = []

mediators = ["ki67_pct", "nlr", "albumin_g_dl", "ldh_u_l"]

for mediator in mediators:
    hypothesis_id = f"h6_{mediator}"
    hypothesis_text = f"{mediator} mediates the effect of treatments on pfs_months."
    iteration6_hypotheses.append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    for treatment in TREATMENTS[:3]:
        a_path, b_path, c_path, c_prime = run_mediation_analysis_simple(df, treatment, mediator, OUTCOMES[0])
        if a_path is not None:
            significant = False
            analysis = {
                "hypothesis_ids": [f"h6_{mediator}_{treatment}"],
                "result_summary": f"Mediation: a={a_path:.4f}, b={b_path:.4f}, c={c_path:.4f}, c'={c_prime:.4f}.",
                "effect_estimate": float(a_path),
                "p_value": None,
                "significant": significant
            }
            iteration6_analyses.append(analysis)

transcript["iterations"].append({
    "index": 6,
    "proposed_hypotheses": iteration6_hypotheses,
    "analyses": iteration6_analyses
})

# Iteration 7: Categorical feature comparisons
print("\n=== Iteration 7: Categorical feature comparisons ===")
iteration7_hypotheses = []
iteration7_analyses = []

categorical_features = ["sex_female", "ecog_ps", "stage_iv", "postmenopausal", "er_positive", 
                        "pr_positive", "her2_positive", "brca1_mutation", "brca2_mutation",
                        "pik3ca_mutation", "node_positive"]

for feature in categorical_features:
    hypothesis_id = f"h7_{feature}"
    hypothesis_text = f"{feature} is associated with pfs_months."
    iteration7_hypotheses.append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "novel"
    })
    
    f_stat, p_value, _, _ = run_categorical_comparison(df, feature, OUTCOMES[0])
    significant = p_value < 0.05 if p_value is not None else False
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"ANOVA F={f_stat:.4f} for {feature} (p={p_value:.4f}).",
        "effect_estimate": float(f_stat) if f_stat is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "significant": significant
    }
    iteration7_analyses.append(analysis)

transcript["iterations"].append({
    "index": 7,
    "proposed_hypotheses": iteration7_hypotheses,
    "analyses": iteration7_analyses
})

# Iteration 8: Refinement of strongest interactions
print("\n=== Iteration 8: Refinement of strongest interactions ===")
iteration8_hypotheses = []
iteration8_analyses = []

strongest_interactions = []
for treatment in TREATMENTS:
    modifiers, _, _, _ = run_treatment_effect_heterogeneity(df, treatment, OUTCOMES[0])
    for modifier in modifiers[:2]:
        strongest_interactions.append((modifier, treatment))

strongest_interactions.sort(key=lambda x: -abs(x[0].get('correlation_with_effect', 0)))

for modifier, treatment in strongest_interactions[:5]:
    hypothesis_id = f"h8_{treatment}_{modifier['feature']}"
    hypothesis_text = f"Refined analysis of {treatment} effect modification by {modifier['feature']} on pfs_months."
    iteration8_hypotheses.append({
        "id": hypothesis_id,
        "text": hypothesis_text,
        "kind": "refined"
    })
    
    effect, p_val, _, _ = run_interaction_regression(df, treatment, modifier['feature'], None, OUTCOMES[0])
    significant = p_val < 0.05 if p_val is not None else False
    
    analysis = {
        "hypothesis_ids": [hypothesis_id],
        "result_summary": f"Refined interaction: effect={effect:.4f} (p={p_val:.4f}).",
        "effect_estimate": float(effect) if effect is not None else None,
        "p_value": float(p_val) if p_val is not None else None,
        "significant": significant
    }
    iteration8_analyses.append(analysis)

transcript["iterations"].append({
    "index": 8,
    "proposed_hypotheses": iteration8_hypotheses,
    "analyses": iteration8_analyses
})

# Iteration 9: Best-supported treatment-effect subgroup
print("\n=== Iteration 9: Best-supported treatment-effect subgroup ===")
iteration9_hypotheses = []
iteration9_analyses = []

for treatment in TREATMENTS:
    modifiers, interaction_results, overall_effect, overall_p = run_treatment_effect_heterogeneity(df, treatment, OUTCOMES[0])
    
    if modifiers:
        best_modifier = modifiers[0]
        hypothesis_id = f"h9_{treatment}_best_subgroup"
        hypothesis_text = f"The effect of {treatment} on pfs_months is strongest in patients with {best_modifier['feature']}={best_modifier['feature'].split('=')[-1] if '=' in str(best_modifier['feature']) else 'present'} (correlation={best_modifier['correlation_with_effect']:.4f}, p={best_modifier['p_correlation']:.4f})."
        iteration9_hypotheses.append({
            "id": hypothesis_id,
            "text": hypothesis_text,
            "kind": "refined"
        })
        
        effect = best_modifier.get('stratified_effect')
        p_val = best_modifier.get('stratified_p')
        significant = p_val < 0.05 if p_val is not None else False
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"Best-supported subgroup: {treatment} effect within {best_modifier['feature']}: effect={effect:.4f} (p={p_val:.4f}).",
            "effect_estimate": float(effect) if effect is not None else None,
            "p_value": float(p_val) if p_val is not None else None,
            "significant": significant
        }
        iteration9_analyses.append(analysis)
    else:
        hypothesis_id = f"h9_{treatment}_no_subgroup"
        hypothesis_text = f"No significant treatment-effect heterogeneity found for {treatment} on pfs_months."
        iteration9_hypotheses.append({
            "id": hypothesis_id,
            "text": hypothesis_text,
            "kind": "refined"
        })
        
        analysis = {
            "hypothesis_ids": [hypothesis_id],
            "result_summary": f"No significant heterogeneity found for {treatment} (overall p={overall_p:.4f}).",
            "effect_estimate": float(overall_effect),
            "p_value": float(overall_p),
            "significant": overall_p < 0.05
        }
        iteration9_analyses.append(analysis)

transcript["iterations"].append({
    "index": 9,
    "proposed_hypotheses": iteration9_hypotheses,
    "analyses": iteration9_analyses
})

# Iteration 10: Final comprehensive summary
print("\n=== Iteration 10: Final comprehensive summary ===")
iteration10_hypotheses = []
iteration10_analyses = []

hypothesis_id = "h10_summary"
hypothesis_text = "Summary of all treatment-outcome relationships and their effect modifiers."
iteration10_hypotheses.append({
    "id": hypothesis_id,
    "text": hypothesis_text,
    "kind": "refined"
})

summary_stats = {}
for treatment in TREATMENTS:
    modifiers, _, overall_effect, overall_p = run_treatment_effect_heterogeneity(df, treatment, OUTCOMES[0])
    summary_stats[treatment] = {
        "overall_effect": overall_effect,
        "overall_p": overall_p,
        "num_modifiers": len(modifiers),
        "top_modifier": modifiers[0] if modifiers else None
    }

sorted_treatments = sorted(summary_stats.items(), key=lambda x: x[1]['overall_p'])[:3]
sorted_str = ", ".join([f"{t}: p={s['overall_p']:.4f}" for t, s in sorted_treatments])

analysis = {
    "hypothesis_ids": [hypothesis_id],
    "result_summary": f"Summary: {len(TREATMENTS)} treatments analyzed. Strongest overall effects: {sorted_str}.",
    "effect_estimate": None,
    "p_value": None,
    "significant": False
}
iteration10_analyses.append(analysis)

transcript["iterations"].append({
    "index": 10,
    "proposed_hypotheses": iteration10_hypotheses,
    "analyses": iteration10_analyses
})

# Convert to JSON-serializable format
transcript_jsonable = to_jsonable(transcript)

# Write transcript.json
with open("transcript.json", "w") as f:
    json.dump(transcript_jsonable, f, indent=2)
print("\nWrote transcript.json")

# Generate analysis_summary.txt
summary_lines = []
summary_lines.append("=" * 80)
summary_lines.append("ONCOLOGY DATASET ANALYSIS SUMMARY - ds001_breast")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append(f"Dataset: {len(df)} patient records")
summary_lines.append(f"Features analyzed: {len(FEATURES)}")
summary_lines.append(f"Treatments analyzed: {len(TREATMENTS)}")
summary_lines.append(f"Iterations completed: {len(transcript['iterations'])}")
summary_lines.append("")

for i, iteration in enumerate(transcript["iterations"], 1):
    summary_lines.append("-" * 80)
    summary_lines.append(f"ITERATION {i}:")
    summary_lines.append("-" * 80)
    
    for hypothesis in iteration["proposed_hypotheses"]:
        summary_lines.append(f"  Hypothesis {hypothesis['id']}: {hypothesis['text']}")
    
    for analysis in iteration["analyses"]:
        sig_str = "SIGNIFICANT" if analysis.get("significant", False) else "not significant"
        summary_lines.append(f"  Analysis: {analysis['result_summary']}")
        if analysis.get("p_value") is not None:
            summary_lines.append(f"    p-value: {analysis['p_value']:.6f} ({sig_str})")
        if analysis.get("effect_estimate") is not None:
            summary_lines.append(f"    effect estimate: {analysis['effect_estimate']:.6f}")
    
    summary_lines.append("")

summary_lines.append("=" * 80)
summary_lines.append("KEY FINDINGS SUMMARY")
summary_lines.append("=" * 80)
summary_lines.append("")

significant_count = 0
total_count = 0
for iteration in transcript["iterations"]:
    for analysis in iteration["analyses"]:
        total_count += 1
        if analysis.get("significant", False):
            significant_count += 1

summary_lines.append(f"Total analyses performed: {total_count}")
summary_lines.append(f"Statistically significant findings (p < 0.05): {significant_count}")
summary_lines.append(f"Significance rate: {significant_count/total_count*100:.1f}%")
summary_lines.append("")

summary_lines.append("TREATMENT EFFECT SUMMARY:")
summary_lines.append("-" * 40)
for treatment in TREATMENTS:
    modifiers, _, overall_effect, overall_p = run_treatment_effect_heterogeneity(df, treatment, OUTCOMES[0])
    summary_lines.append(f"  {treatment}:")
    summary_lines.append(f"    Overall effect: {overall_effect:.4f} (p={overall_p:.4f})")
    if modifiers:
        top_mod = modifiers[0]
        summary_lines.append(f"    Top modifier: {top_mod['feature']} (correlation={top_mod['correlation_with_effect']:.4f}, p={top_mod['p_correlation']:.4f})")
    else:
        summary_lines.append(f"    No significant modifiers found")
summary_lines.append("")

summary_lines.append("FEATURE-OUTCOME CORRELATIONS (|r| > 0.1):")
summary_lines.append("-" * 40)
for feature in key_features:
    corr, p_value, _, _ = run_correlation(df, feature, None, OUTCOMES[0])
    if corr is not None and abs(corr) > 0.1:
        sig_str = "SIGNIFICANT" if p_value < 0.05 else "not significant"
        summary_lines.append(f"  {feature}: r={corr:.4f} (p={p_value:.4f}) [{sig_str}]")
summary_lines.append("")

summary_lines.append("BEST-SUPPORTED TREATMENT-EFFECT SUBGROUPS:")
summary_lines.append("-" * 40)
for treatment in TREATMENTS:
    modifiers, _, _, _ = run_treatment_effect_heterogeneity(df, treatment, OUTCOMES[0])
    if modifiers:
        top_mod = modifiers[0]
        summary_lines.append(f"  {treatment}: {top_mod['feature']} modifies effect (correlation={top_mod['correlation_with_effect']:.4f})")
    else:
        summary_lines.append(f"  {treatment}: No significant subgroup effect found")
summary_lines.append("")

summary_lines.append("=" * 80)
summary_lines.append("END OF ANALYSIS SUMMARY")
summary_lines.append("=" * 80)

with open("analysis_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))
print("Wrote analysis_summary.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("Generated: transcript.json, analysis_summary.txt")
