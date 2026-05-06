"""Run all analyses for ds001_breast and write intermediate results.

This file produces results.json with structured findings for each iteration,
which is then used to build transcript.json and analysis_summary.txt.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")
print("Loaded:", DF.shape)

OUT = {"iterations": []}

TREATMENTS = [
    "treatment_tamoxifen",
    "treatment_palbociclib",
    "treatment_trastuzumab",
    "treatment_olaparib",
    "treatment_sacituzumab_govitecan",
    "treatment_pembrolizumab",
]

BIOMARKERS = [
    "er_positive", "pr_positive", "her2_positive", "her2_low",
    "brca1_mutation", "brca2_mutation", "pik3ca_mutation",
    "node_positive", "stage_iv", "has_brain_mets", "postmenopausal",
]

CONT_FEATURES = [
    "age_years", "ecog_ps", "ki67_pct", "tumor_size_cm",
    "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr",
    "hemoglobin_g_dl", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
    "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
    "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl",
]


def add_iter(idx, hypotheses, analyses):
    OUT["iterations"].append(
        {"index": idx, "proposed_hypotheses": hypotheses, "analyses": analyses}
    )


def mean_diff(treat_col, outcome="pfs_months"):
    a = DF.loc[DF[treat_col] == 1, outcome]
    b = DF.loc[DF[treat_col] == 0, outcome]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(p), float(a.mean()), float(b.mean()), len(a), len(b)


def ols_effect(formula):
    m = ols(formula, data=DF).fit()
    return m


def interaction_effect(treat, modifier, outcome="pfs_months"):
    """Difference-in-differences: treatment effect when modifier=1 minus when modifier=0."""
    sub11 = DF.loc[(DF[treat] == 1) & (DF[modifier] == 1), outcome]
    sub10 = DF.loc[(DF[treat] == 0) & (DF[modifier] == 1), outcome]
    sub01 = DF.loc[(DF[treat] == 1) & (DF[modifier] == 0), outcome]
    sub00 = DF.loc[(DF[treat] == 0) & (DF[modifier] == 0), outcome]
    eff_in = sub11.mean() - sub10.mean()
    eff_out = sub01.mean() - sub00.mean()
    diff = eff_in - eff_out
    f = f"{outcome} ~ {treat} * {modifier}"
    m = ols(f, data=DF).fit()
    p = float(m.pvalues[f"{treat}:{modifier}"])
    return {
        "eff_in_subgroup": float(eff_in),
        "eff_outside_subgroup": float(eff_out),
        "interaction_diff": float(diff),
        "p_value": p,
        "n_in_treat": int(len(sub11)),
        "n_in_ctrl": int(len(sub10)),
        "n_out_treat": int(len(sub01)),
        "n_out_ctrl": int(len(sub00)),
    }


# =========================================
# ITERATION 1: main effects of each treatment on PFS
# =========================================
hyp1 = []
an1 = []
for i, t in enumerate(TREATMENTS, start=1):
    hid = f"h1_{i}"
    hyp1.append(
        {
            "id": hid,
            "text": f"Patients receiving {t}=1 have a different mean pfs_months than patients with {t}=0.",
            "kind": "novel",
        }
    )
    diff, p, m1, m0, n1, n0 = mean_diff(t)
    an1.append(
        {
            "hypothesis_ids": [hid],
            "code": f"stats.ttest_ind(df.loc[df['{t}']==1,'pfs_months'], df.loc[df['{t}']==0,'pfs_months'])",
            "result_summary": f"Mean pfs_months: {m1:.3f} on {t} (n={n1}) vs {m0:.3f} off (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
            "p_value": p,
            "effect_estimate": diff,
            "significant": bool(p < 0.05),
        }
    )
add_iter(1, hyp1, an1)
print("Iter 1 done")

# =========================================
# ITERATION 2: main effects of binary biomarkers on PFS
# =========================================
hyp2 = []
an2 = []
for i, b in enumerate(BIOMARKERS, start=1):
    hid = f"h2_{i}"
    hyp2.append(
        {
            "id": hid,
            "text": f"Patients with {b}=1 have a different mean pfs_months than those with {b}=0.",
            "kind": "novel",
        }
    )
    diff, p, m1, m0, n1, n0 = mean_diff(b)
    an2.append(
        {
            "hypothesis_ids": [hid],
            "code": f"stats.ttest_ind(df.loc[df['{b}']==1,'pfs_months'], df.loc[df['{b}']==0,'pfs_months'])",
            "result_summary": f"Mean pfs_months: {m1:.3f} when {b}=1 (n={n1}) vs {m0:.3f} when 0 (n={n0}); diff={diff:+.3f}, p={p:.3g}.",
            "p_value": p,
            "effect_estimate": diff,
            "significant": bool(p < 0.05),
        }
    )
add_iter(2, hyp2, an2)
print("Iter 2 done")

# =========================================
# ITERATION 3: continuous-feature main effects via OLS slope on PFS
# =========================================
hyp3 = []
an3 = []
for i, c in enumerate(CONT_FEATURES, start=1):
    hid = f"h3_{i}"
    hyp3.append(
        {
            "id": hid,
            "text": f"Higher {c} is associated with a different pfs_months (linear association).",
            "kind": "novel",
        }
    )
    m = ols_effect(f"pfs_months ~ {c}")
    b = float(m.params[c])
    p = float(m.pvalues[c])
    an3.append(
        {
            "hypothesis_ids": [hid],
            "code": f"ols('pfs_months ~ {c}', data=df).fit()",
            "result_summary": f"OLS slope of pfs_months on {c}: beta={b:+.5f} per unit, p={p:.3g}.",
            "p_value": p,
            "effect_estimate": b,
            "significant": bool(p < 0.05),
        }
    )
add_iter(3, hyp3, an3)
print("Iter 3 done")

# =========================================
# ITERATION 4: HER2-targeted treatments by HER2 status
# (trastuzumab and sacituzumab govitecan)
# =========================================
hyp4 = []
an4 = []
for i, (treat, mod) in enumerate(
    [
        ("treatment_trastuzumab", "her2_positive"),
        ("treatment_trastuzumab", "her2_low"),
        ("treatment_sacituzumab_govitecan", "her2_low"),
        ("treatment_sacituzumab_govitecan", "her2_positive"),
    ],
    start=1,
):
    hid = f"h4_{i}"
    hyp4.append(
        {
            "id": hid,
            "text": f"The effect of {treat} on pfs_months is larger (more positive) in patients with {mod}=1 than {mod}=0.",
            "kind": "novel",
        }
    )
    r = interaction_effect(treat, mod)
    an4.append(
        {
            "hypothesis_ids": [hid],
            "code": f"ols('pfs_months ~ {treat} * {mod}', data=df).fit()",
            "result_summary": (
                f"Treatment effect of {treat} = {r['eff_in_subgroup']:+.3f} when {mod}=1 vs "
                f"{r['eff_outside_subgroup']:+.3f} when {mod}=0; interaction diff = {r['interaction_diff']:+.3f}, p={r['p_value']:.3g}."
            ),
            "p_value": r["p_value"],
            "effect_estimate": r["interaction_diff"],
            "significant": bool(r["p_value"] < 0.05),
        }
    )
add_iter(4, hyp4, an4)
print("Iter 4 done")

# =========================================
# ITERATION 5: Tamoxifen by ER/PR/postmenopausal
# =========================================
hyp5 = []
an5 = []
for i, (treat, mod) in enumerate(
    [
        ("treatment_tamoxifen", "er_positive"),
        ("treatment_tamoxifen", "pr_positive"),
        ("treatment_tamoxifen", "postmenopausal"),
    ],
    start=1,
):
    hid = f"h5_{i}"
    hyp5.append(
        {
            "id": hid,
            "text": f"The effect of {treat} on pfs_months is larger (more positive) in patients with {mod}=1 than {mod}=0.",
            "kind": "novel",
        }
    )
    r = interaction_effect(treat, mod)
    an5.append(
        {
            "hypothesis_ids": [hid],
            "code": f"ols('pfs_months ~ {treat} * {mod}', data=df).fit()",
            "result_summary": (
                f"Treatment effect of {treat} = {r['eff_in_subgroup']:+.3f} when {mod}=1 vs "
                f"{r['eff_outside_subgroup']:+.3f} when {mod}=0; interaction diff = {r['interaction_diff']:+.3f}, p={r['p_value']:.3g}."
            ),
            "p_value": r["p_value"],
            "effect_estimate": r["interaction_diff"],
            "significant": bool(r["p_value"] < 0.05),
        }
    )
add_iter(5, hyp5, an5)
print("Iter 5 done")

# =========================================
# ITERATION 6: Palbociclib by ER+/postmenopausal/PIK3CA
# =========================================
hyp6 = []
an6 = []
for i, (treat, mod) in enumerate(
    [
        ("treatment_palbociclib", "er_positive"),
        ("treatment_palbociclib", "postmenopausal"),
        ("treatment_palbociclib", "pik3ca_mutation"),
        ("treatment_palbociclib", "her2_positive"),
    ],
    start=1,
):
    hid = f"h6_{i}"
    hyp6.append(
        {
            "id": hid,
            "text": f"The effect of {treat} on pfs_months is larger in patients with {mod}=1 than {mod}=0.",
            "kind": "novel",
        }
    )
    r = interaction_effect(treat, mod)
    an6.append(
        {
            "hypothesis_ids": [hid],
            "code": f"ols('pfs_months ~ {treat} * {mod}', data=df).fit()",
            "result_summary": (
                f"Treatment effect of {treat} = {r['eff_in_subgroup']:+.3f} when {mod}=1 vs "
                f"{r['eff_outside_subgroup']:+.3f} when {mod}=0; interaction diff = {r['interaction_diff']:+.3f}, p={r['p_value']:.3g}."
            ),
            "p_value": r["p_value"],
            "effect_estimate": r["interaction_diff"],
            "significant": bool(r["p_value"] < 0.05),
        }
    )
add_iter(6, hyp6, an6)
print("Iter 6 done")

# =========================================
# ITERATION 7: Olaparib by BRCA1/BRCA2/either-BRCA
# =========================================
DF["brca_any"] = ((DF["brca1_mutation"] == 1) | (DF["brca2_mutation"] == 1)).astype(int)

hyp7 = []
an7 = []
for i, (treat, mod) in enumerate(
    [
        ("treatment_olaparib", "brca1_mutation"),
        ("treatment_olaparib", "brca2_mutation"),
        ("treatment_olaparib", "brca_any"),
    ],
    start=1,
):
    hid = f"h7_{i}"
    hyp7.append(
        {
            "id": hid,
            "text": f"The effect of {treat} on pfs_months is larger in patients with {mod}=1 than {mod}=0.",
            "kind": "novel",
        }
    )
    r = interaction_effect(treat, mod)
    an7.append(
        {
            "hypothesis_ids": [hid],
            "code": f"ols('pfs_months ~ {treat} * {mod}', data=df).fit()",
            "result_summary": (
                f"Treatment effect of {treat} = {r['eff_in_subgroup']:+.3f} when {mod}=1 vs "
                f"{r['eff_outside_subgroup']:+.3f} when {mod}=0; interaction diff = {r['interaction_diff']:+.3f}, p={r['p_value']:.3g}."
            ),
            "p_value": r["p_value"],
            "effect_estimate": r["interaction_diff"],
            "significant": bool(r["p_value"] < 0.05),
        }
    )
add_iter(7, hyp7, an7)
print("Iter 7 done")

# Save partial
with open("results.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("Saved partial results after iter 7")

# =========================================
# ITERATION 8: Pembrolizumab by various features
# =========================================
hyp8 = []
an8 = []
for i, (treat, mod) in enumerate(
    [
        ("treatment_pembrolizumab", "er_positive"),
        ("treatment_pembrolizumab", "her2_positive"),
        ("treatment_pembrolizumab", "stage_iv"),
        ("treatment_pembrolizumab", "node_positive"),
        ("treatment_pembrolizumab", "has_brain_mets"),
    ],
    start=1,
):
    hid = f"h8_{i}"
    hyp8.append(
        {
            "id": hid,
            "text": f"The effect of {treat} on pfs_months differs in patients with {mod}=1 vs {mod}=0.",
            "kind": "novel",
        }
    )
    r = interaction_effect(treat, mod)
    an8.append(
        {
            "hypothesis_ids": [hid],
            "code": f"ols('pfs_months ~ {treat} * {mod}', data=df).fit()",
            "result_summary": (
                f"Treatment effect of {treat} = {r['eff_in_subgroup']:+.3f} when {mod}=1 vs "
                f"{r['eff_outside_subgroup']:+.3f} when {mod}=0; interaction diff = {r['interaction_diff']:+.3f}, p={r['p_value']:.3g}."
            ),
            "p_value": r["p_value"],
            "effect_estimate": r["interaction_diff"],
            "significant": bool(r["p_value"] < 0.05),
        }
    )
add_iter(8, hyp8, an8)
print("Iter 8 done")

# =========================================
# ITERATION 9: Multivariable model — adjusted treatment effects
# =========================================
hyp9 = []
an9 = []
adj_form_base = (
    "pfs_months ~ "
    + " + ".join(TREATMENTS)
    + " + "
    + " + ".join(BIOMARKERS)
    + " + age_years + ecog_ps + ki67_pct + tumor_size_cm + albumin_g_dl + ldh_u_l "
    + "+ weight_loss_pct_6mo + crp_mg_l + nlr"
)
m_adj = ols_effect(adj_form_base)
for i, t in enumerate(TREATMENTS, start=1):
    hid = f"h9_{i}"
    hyp9.append(
        {
            "id": hid,
            "text": f"After adjusting for biomarkers and clinical features, {t} retains an independent association with pfs_months.",
            "kind": "refined",
        }
    )
    b = float(m_adj.params[t])
    p = float(m_adj.pvalues[t])
    an9.append(
        {
            "hypothesis_ids": [hid],
            "code": "OLS pfs_months ~ all treatments + biomarkers + clin features",
            "result_summary": f"Adjusted beta for {t}: {b:+.4f} (p={p:.3g}).",
            "p_value": p,
            "effect_estimate": b,
            "significant": bool(p < 0.05),
        }
    )
add_iter(9, hyp9, an9)
print("Iter 9 done")

# =========================================
# ITERATION 10: ECOG x treatment interactions (frailty modifier)
# =========================================
hyp10 = []
an10 = []
for i, t in enumerate(TREATMENTS, start=1):
    hid = f"h10_{i}"
    hyp10.append(
        {
            "id": hid,
            "text": f"The effect of {t} on pfs_months differs by ecog_ps (functional status modifies treatment efficacy).",
            "kind": "novel",
        }
    )
    f = f"pfs_months ~ {t} * ecog_ps"
    m = ols(f, data=DF).fit()
    coef = f"{t}:ecog_ps"
    b = float(m.params[coef])
    p = float(m.pvalues[coef])
    an10.append(
        {
            "hypothesis_ids": [hid],
            "code": f"ols('{f}', data=df).fit()",
            "result_summary": f"Interaction beta {coef}: {b:+.4f}, p={p:.3g}.",
            "p_value": p,
            "effect_estimate": b,
            "significant": bool(p < 0.05),
        }
    )
add_iter(10, hyp10, an10)
print("Iter 10 done")

# =========================================
# ITERATION 11: Albumin x treatment (nutritional/inflammatory modifier)
# =========================================
hyp11 = []
an11 = []
for i, t in enumerate(TREATMENTS, start=1):
    hid = f"h11_{i}"
    hyp11.append(
        {
            "id": hid,
            "text": f"The effect of {t} on pfs_months is modified by albumin_g_dl (low albumin attenuates benefit).",
            "kind": "novel",
        }
    )
    f = f"pfs_months ~ {t} * albumin_g_dl"
    m = ols(f, data=DF).fit()
    coef = f"{t}:albumin_g_dl"
    b = float(m.params[coef])
    p = float(m.pvalues[coef])
    an11.append(
        {
            "hypothesis_ids": [hid],
            "code": f"ols('{f}', data=df).fit()",
            "result_summary": f"Interaction beta {coef}: {b:+.4f}, p={p:.3g}.",
            "p_value": p,
            "effect_estimate": b,
            "significant": bool(p < 0.05),
        }
    )
add_iter(11, hyp11, an11)
print("Iter 11 done")

with open("results.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("Saved partial results after iter 11")

# =========================================
# ITERATION 12: Stage IV / brain mets x each treatment
# =========================================
hyp12 = []
an12 = []
for i, (t, mod) in enumerate(
    [(t, m) for t in TREATMENTS for m in ["stage_iv", "has_brain_mets"]],
    start=1,
):
    hid = f"h12_{i}"
    hyp12.append(
        {
            "id": hid,
            "text": f"The effect of {t} on pfs_months differs by {mod} status.",
            "kind": "novel",
        }
    )
    r = interaction_effect(t, mod)
    an12.append(
        {
            "hypothesis_ids": [hid],
            "code": f"ols('pfs_months ~ {t} * {mod}', data=df).fit()",
            "result_summary": (
                f"Treatment effect of {t} = {r['eff_in_subgroup']:+.3f} when {mod}=1 vs "
                f"{r['eff_outside_subgroup']:+.3f} when {mod}=0; interaction diff = {r['interaction_diff']:+.3f}, p={r['p_value']:.3g}."
            ),
            "p_value": r["p_value"],
            "effect_estimate": r["interaction_diff"],
            "significant": bool(r["p_value"] < 0.05),
        }
    )
add_iter(12, hyp12, an12)
print("Iter 12 done")

# =========================================
# ITERATION 13: Systematic treatment-by-binary-biomarker interaction screen
# (every treatment x every binary biomarker)
# =========================================
hyp13 = []
an13 = []
counter = 0
screen_results = []
for t in TREATMENTS:
    for mod in BIOMARKERS + ["brca_any"]:
        counter += 1
        hid = f"h13_{counter}"
        hyp13.append(
            {
                "id": hid,
                "text": f"The treatment effect of {t} on pfs_months differs by {mod}.",
                "kind": "novel",
            }
        )
        r = interaction_effect(t, mod)
        screen_results.append(
            {"treat": t, "mod": mod, "diff": r["interaction_diff"], "p": r["p_value"], "in": r["eff_in_subgroup"], "out": r["eff_outside_subgroup"]}
        )
        an13.append(
            {
                "hypothesis_ids": [hid],
                "code": f"ols('pfs_months ~ {t} * {mod}', data=df).fit()",
                "result_summary": (
                    f"Effect of {t}: {r['eff_in_subgroup']:+.3f} when {mod}=1 vs "
                    f"{r['eff_outside_subgroup']:+.3f} when {mod}=0; diff={r['interaction_diff']:+.3f}, p={r['p_value']:.3g}."
                ),
                "p_value": r["p_value"],
                "effect_estimate": r["interaction_diff"],
                "significant": bool(r["p_value"] < 0.05),
            }
        )
add_iter(13, hyp13, an13)

# Sort screen by largest absolute interaction
screen_sorted = sorted(screen_results, key=lambda x: -abs(x["diff"]))
print("Top interactions:")
for r in screen_sorted[:20]:
    print(f"  {r['treat']} x {r['mod']}: diff={r['diff']:+.3f}, p={r['p']:.3g}, in={r['in']:+.3f}, out={r['out']:+.3f}")
print("Iter 13 done")

# Save the top results for later
OUT["_top_interactions"] = screen_sorted[:30]

with open("results.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("Saved partial results after iter 13")

# =========================================
# ITERATION 14: Continuous-modifier interaction screening (every treatment x every continuous feature)
# =========================================
hyp14 = []
an14 = []
counter = 0
cont_screen = []
for t in TREATMENTS:
    for c in CONT_FEATURES:
        counter += 1
        hid = f"h14_{counter}"
        hyp14.append(
            {
                "id": hid,
                "text": f"The treatment effect of {t} on pfs_months is modified by {c} (continuous).",
                "kind": "novel",
            }
        )
        f = f"pfs_months ~ {t} * {c}"
        m = ols(f, data=DF).fit()
        coef = f"{t}:{c}"
        b = float(m.params[coef])
        p = float(m.pvalues[coef])
        cont_screen.append({"treat": t, "mod": c, "beta": b, "p": p})
        an14.append(
            {
                "hypothesis_ids": [hid],
                "code": f"ols('{f}', data=df).fit()",
                "result_summary": f"Interaction {coef}: beta={b:+.5f}, p={p:.3g}.",
                "p_value": p,
                "effect_estimate": b,
                "significant": bool(p < 0.05),
            }
        )
add_iter(14, hyp14, an14)
cont_sorted = sorted(cont_screen, key=lambda x: x["p"])
print("Top continuous interactions:")
for r in cont_sorted[:15]:
    print(f"  {r['treat']} x {r['mod']}: beta={r['beta']:+.5f}, p={r['p']:.3g}")
print("Iter 14 done")
OUT["_top_cont_interactions"] = cont_sorted[:25]

with open("results.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("Saved partial results after iter 14")

# Persist top results to prepare for refined iterations
print("=== Best binary subgroup x treatment effects (top 20) ===")
for r in screen_sorted[:20]:
    print(r)

print("=== Best continuous interactions (top 15) ===")
for r in cont_sorted[:15]:
    print(r)

print("Done with first batch.")
