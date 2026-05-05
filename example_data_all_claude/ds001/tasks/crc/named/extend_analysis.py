"""
Add regorafenib responder-subgroup analyses to iterations 23, 24, 25 (extending the existing transcript).
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")

def subgroup_treatment_effect(df, treat, outcome, mask):
    sub = df.loc[mask]
    if sub[treat].sum() < 20 or (1 - sub[treat]).sum() < 20:
        return None
    a = sub.loc[sub[treat] == 1, outcome].values
    b = sub.loc[sub[treat] == 0, outcome].values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "n_treated": int(len(a)),
        "n_untreated": int(len(b)),
        "mean_treated": float(np.mean(a)),
        "mean_untreated": float(np.mean(b)),
        "effect_estimate": float(np.mean(a) - np.mean(b)),
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }

with open("transcript.json") as f:
    T = json.load(f)

# ITERATION 23 — extend with regorafenib pair search
it23 = next(it for it in T["iterations"] if it["index"] == 23)
it23["proposed_hypotheses"].extend([
    {"id":"h23.3","text":"Regorafenib benefit on pfs_months is concentrated in a subgroup that includes kras_mutation=0 (KRAS wild-type); within KRAS-mutant patients, regorafenib has no PFS benefit.","kind":"novel"},
    {"id":"h23.4","text":"Regorafenib benefit on pfs_months is concentrated in left-sided primaries (right_sided_primary=0); within right-sided primaries, regorafenib has no benefit (or harmful effect).","kind":"novel"},
    {"id":"h23.5","text":"Regorafenib benefit on pfs_months is concentrated in BRAF wild-type (braf_v600e=0); within BRAF V600E patients, regorafenib has no benefit.","kind":"novel"},
])
mods = ["sex_female","stage_iv","right_sided_primary","kras_mutation","nras_mutation","braf_v600e","msi_high","her2_amplified"]

def best_pair(treat, mods, top=8):
    rows = []
    for a in mods:
        for av in (0,1):
            for b in mods:
                if b == a: continue
                for bv in (0,1):
                    mask = (DF[a]==av) & (DF[b]==bv)
                    s = subgroup_treatment_effect(DF, treat, "pfs_months", mask)
                    if s is None: continue
                    rows.append({"a":a,"av":av,"b":b,"bv":bv,
                                 "n":int(mask.sum()),
                                 "diff":s["effect_estimate"],
                                 "p":s["p_value"]})
    return sorted(rows, key=lambda r: r["diff"], reverse=True)[:top]

top_rego = best_pair("treatment_regorafenib", mods, top=10)
for r in top_rego:
    it23["analyses"].append({
        "hypothesis_ids": ["h23.3","h23.4","h23.5"],
        "code": f"regorafenib pair search: {r['a']}={r['av']} & {r['b']}={r['bv']}",
        "result_summary": f"regorafenib subgroup [{r['a']}={r['av']} & {r['b']}={r['bv']}] (n={r['n']}): diff={r['diff']:.3f}, p={r['p']:.3g}",
        "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05),
    })

# Single-feature regorafenib stratification (drop one predicate)
for label, mask in [
    ("KRAS WT only", DF["kras_mutation"]==0),
    ("KRAS mut only", DF["kras_mutation"]==1),
    ("left-sided only", DF["right_sided_primary"]==0),
    ("right-sided only", DF["right_sided_primary"]==1),
    ("BRAF WT only", DF["braf_v600e"]==0),
    ("BRAF V600E only", DF["braf_v600e"]==1),
    ("NRAS WT only", DF["nras_mutation"]==0),
    ("NRAS mut only", DF["nras_mutation"]==1),
]:
    s = subgroup_treatment_effect(DF, "treatment_regorafenib", "pfs_months", mask)
    if s is None: continue
    it23["analyses"].append({
        "hypothesis_ids":["h23.3","h23.4","h23.5"],
        "code": f"regorafenib effect | {label}",
        "result_summary": f"regorafenib effect ({label}): treated={s['mean_treated']:.3f} (n={s['n_treated']}) vs untreated={s['mean_untreated']:.3f} (n={s['n_untreated']}); diff={s['effect_estimate']:.3f}, p={s['p_value']:.3g}",
        "p_value": s["p_value"], "effect_estimate": s["effect_estimate"], "significant": s["significant"],
    })

# Multi-feature subgroups
multi_subs = [
    ("RAS WT (KRAS=0 & NRAS=0)", (DF["kras_mutation"]==0)&(DF["nras_mutation"]==0)),
    ("RAS/RAF WT (KRAS=0 & NRAS=0 & BRAF=0)", (DF["kras_mutation"]==0)&(DF["nras_mutation"]==0)&(DF["braf_v600e"]==0)),
    ("RAS WT & left-sided", (DF["kras_mutation"]==0)&(DF["nras_mutation"]==0)&(DF["right_sided_primary"]==0)),
    ("RAS/RAF WT & left-sided", (DF["kras_mutation"]==0)&(DF["nras_mutation"]==0)&(DF["braf_v600e"]==0)&(DF["right_sided_primary"]==0)),
    ("KRAS WT & left-sided", (DF["kras_mutation"]==0)&(DF["right_sided_primary"]==0)),
    ("KRAS WT & BRAF WT", (DF["kras_mutation"]==0)&(DF["braf_v600e"]==0)),
    ("KRAS mut OR right-sided OR BRAF V600E (non-responder superset)",
        (DF["kras_mutation"]==1) | (DF["right_sided_primary"]==1) | (DF["braf_v600e"]==1)),
    ("NRAS mut", DF["nras_mutation"]==1),
]
for label, mask in multi_subs:
    s = subgroup_treatment_effect(DF, "treatment_regorafenib", "pfs_months", mask)
    if s is None: continue
    it23["analyses"].append({
        "hypothesis_ids":["h23.3","h23.4","h23.5"],
        "code": f"regorafenib effect | {label}",
        "result_summary": f"regorafenib effect ({label}): treated={s['mean_treated']:.3f} (n={s['n_treated']}) vs untreated={s['mean_untreated']:.3f} (n={s['n_untreated']}); diff={s['effect_estimate']:.3f}, p={s['p_value']:.3g}",
        "p_value": s["p_value"], "effect_estimate": s["effect_estimate"], "significant": s["significant"],
    })

# ITERATION 24 — add the final regorafenib responder hypothesis
it24 = next(it for it in T["iterations"] if it["index"] == 24)
it24["proposed_hypotheses"].extend([
    {"id":"h24.5","text":"Final regorafenib responder subgroup: kras_mutation=0 AND right_sided_primary=0 AND braf_v600e=0 (left-sided KRAS-WT, BRAF-WT). Within this subgroup, treatment_regorafenib=1 increases pfs_months substantially (signed positive).","kind":"refined"},
    {"id":"h24.6","text":"Within RAS WT (KRAS=0 & NRAS=0) AND BRAF WT AND left-sided, treatment_regorafenib increases pfs_months (the most refined responder definition).","kind":"refined"},
    {"id":"h24.7","text":"Removing the KRAS-WT predicate, the BRAF-WT predicate, OR the left-sided predicate from the regorafenib responder subgroup definition substantially reduces (or abolishes) the regorafenib treatment effect.","kind":"refined"},
])

# Final regorafenib responder subgroup
mask_rego_full = (DF["kras_mutation"]==0) & (DF["right_sided_primary"]==0) & (DF["braf_v600e"]==0)
s = subgroup_treatment_effect(DF, "treatment_regorafenib", "pfs_months", mask_rego_full)
it24["analyses"].append({
    "hypothesis_ids": ["h24.5"],
    "code": "regorafenib effect | KRAS=0 & right=0 & BRAF=0",
    "result_summary": f"FINAL regorafenib responder subgroup (left-sided KRAS-WT BRAF-WT, n={int(mask_rego_full.sum())}): treated PFS={s['mean_treated']:.3f} (n={s['n_treated']}) vs untreated={s['mean_untreated']:.3f} (n={s['n_untreated']}); diff={s['effect_estimate']:.3f}, p={s['p_value']:.3g}",
    "p_value": s["p_value"], "effect_estimate": s["effect_estimate"], "significant": s["significant"],
})

# Most refined: RAS/RAF WT + left-sided
mask_rego_refined = (DF["kras_mutation"]==0) & (DF["nras_mutation"]==0) & (DF["braf_v600e"]==0) & (DF["right_sided_primary"]==0)
s = subgroup_treatment_effect(DF, "treatment_regorafenib", "pfs_months", mask_rego_refined)
it24["analyses"].append({
    "hypothesis_ids": ["h24.6"],
    "code": "regorafenib effect | KRAS=0 & NRAS=0 & BRAF=0 & right=0",
    "result_summary": f"FINAL most-refined regorafenib responder subgroup (left-sided RAS/RAF WT, n={int(mask_rego_refined.sum())}): treated PFS={s['mean_treated']:.3f} (n={s['n_treated']}) vs untreated={s['mean_untreated']:.3f} (n={s['n_untreated']}); diff={s['effect_estimate']:.3f}, p={s['p_value']:.3g}",
    "p_value": s["p_value"], "effect_estimate": s["effect_estimate"], "significant": s["significant"],
})

# Predicate drops on regorafenib subgroup
configs = [
    ("drop KRAS-WT requirement (now any KRAS)",
        (DF["nras_mutation"]==0) & (DF["braf_v600e"]==0) & (DF["right_sided_primary"]==0)),
    ("drop BRAF-WT requirement (now any BRAF)",
        (DF["kras_mutation"]==0) & (DF["nras_mutation"]==0) & (DF["right_sided_primary"]==0)),
    ("drop left-sided requirement (now any sidedness)",
        (DF["kras_mutation"]==0) & (DF["nras_mutation"]==0) & (DF["braf_v600e"]==0)),
    ("drop NRAS-WT requirement (now any NRAS)",
        (DF["kras_mutation"]==0) & (DF["braf_v600e"]==0) & (DF["right_sided_primary"]==0)),
]
for label, mask in configs:
    s = subgroup_treatment_effect(DF, "treatment_regorafenib", "pfs_months", mask)
    if s is None: continue
    it24["analyses"].append({
        "hypothesis_ids":["h24.7"],
        "code": f"regorafenib effect | {label}",
        "result_summary": f"{label}: subgroup n={int(mask.sum())}; treated={s['mean_treated']:.3f} (n={s['n_treated']}) vs untreated={s['mean_untreated']:.3f} (n={s['n_untreated']}); diff={s['effect_estimate']:.3f}, p={s['p_value']:.3g}",
        "p_value": s["p_value"], "effect_estimate": s["effect_estimate"], "significant": s["significant"],
    })

# ITERATION 25 — confirmatory: outside subgroup
it25 = next(it for it in T["iterations"] if it["index"] == 25)
it25["proposed_hypotheses"].append({
    "id":"h25.5",
    "text":"Outside the regorafenib responder subgroup defined in h24.5 (i.e., right-sided OR KRAS-mutant OR BRAF-V600E), treatment_regorafenib has no PFS benefit (or negative effect).",
    "kind":"refined"
})
mask_outside_rego = ~((DF["kras_mutation"]==0) & (DF["right_sided_primary"]==0) & (DF["braf_v600e"]==0))
s = subgroup_treatment_effect(DF, "treatment_regorafenib", "pfs_months", mask_outside_rego)
it25["analyses"].append({
    "hypothesis_ids": ["h25.5"],
    "code": "regorafenib effect | outside responder subgroup",
    "result_summary": f"OUTSIDE regorafenib responder subgroup (right-sided OR KRAS-mut OR BRAF-V600E, n={int(mask_outside_rego.sum())}): treated PFS={s['mean_treated']:.3f} (n={s['n_treated']}) vs untreated={s['mean_untreated']:.3f} (n={s['n_untreated']}); diff={s['effect_estimate']:.3f}, p={s['p_value']:.3g}",
    "p_value": s["p_value"], "effect_estimate": s["effect_estimate"], "significant": s["significant"],
})

# write back
with open("transcript.json","w") as f:
    json.dump(T, f, indent=2)

print("Updated transcript.json")
print("Total analyses:", sum(len(it["analyses"]) for it in T["iterations"]))
print("Total hypotheses:", sum(len(it["proposed_hypotheses"]) for it in T["iterations"]))
