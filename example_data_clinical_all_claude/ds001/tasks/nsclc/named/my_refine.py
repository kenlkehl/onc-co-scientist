"""Refinement of subgroup hypotheses (iters 12-21)."""
from __future__ import annotations
import json
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
df["smoke_current"] = (df["smoking_status"] == "current").astype(int)
df["smoke_former"] = (df["smoking_status"] == "former").astype(int)
df["smoke_never"] = (df["smoking_status"] == "never").astype(int)
df["hist_adeno"] = (df["histology"] == "adenocarcinoma").astype(int)


def adj_diff(sub, tx, covars):
    """Adjusted treatment effect within a subgroup using OLS on PFS."""
    use = covars + [tx]
    use = [c for c in use if sub[c].nunique() > 1]
    if tx not in use:
        return None
    f = "pfs_months ~ " + " + ".join(use)
    m = smf.ols(f, data=sub).fit()
    return {"beta": float(m.params[tx]), "p": float(m.pvalues[tx]), "n": int(len(sub))}


COVARS = [
    "age_years",
    "ecog_ps",
    "stage_iv",
    "has_brain_mets",
    "albumin_g_dl",
    "ldh_u_l",
    "nlr",
    "weight_loss_pct_6mo",
    "sex_female",
    "smoke_current",
    "smoke_former",
    "hist_adeno",
    "egfr_mutation",
    "kras_g12c",
    "alk_fusion",
    "stk11_mutation",
    "brca2_mutation",
    "tmb_high",
    "pdl1_tps",
]
OTHER_TX = {
    "treatment_pembrolizumab": ["treatment_sotorasib", "treatment_olaparib", "treatment_osimertinib"],
    "treatment_sotorasib": ["treatment_pembrolizumab", "treatment_olaparib", "treatment_osimertinib"],
    "treatment_olaparib": ["treatment_pembrolizumab", "treatment_sotorasib", "treatment_osimertinib"],
    "treatment_osimertinib": ["treatment_pembrolizumab", "treatment_sotorasib", "treatment_olaparib"],
}

results = {}

# ---------------------------------------------------------------
# 12: Osimertinib × EGFR specifically (canonical biology check)
# ---------------------------------------------------------------
out = []
for v in [0, 1]:
    sub = df[df["egfr_mutation"] == v].copy()
    out.append({"egfr_mutation": v, **adj_diff(sub, "treatment_osimertinib", COVARS + OTHER_TX["treatment_osimertinib"])})
# Also formal interaction test
formula = (
    "pfs_months ~ treatment_osimertinib * egfr_mutation + "
    + " + ".join([c for c in COVARS if c != "egfr_mutation"] + OTHER_TX["treatment_osimertinib"])
)
m = smf.ols(formula, data=df).fit()
out.append({
    "interaction_test": "osimertinib × egfr_mutation",
    "beta": float(m.params["treatment_osimertinib:egfr_mutation"]),
    "p": float(m.pvalues["treatment_osimertinib:egfr_mutation"]),
    "tx_main_beta": float(m.params["treatment_osimertinib"]),
    "tx_main_p": float(m.pvalues["treatment_osimertinib"]),
})
results["iter12_osi_x_egfr"] = out

# ---------------------------------------------------------------
# 13: Pembrolizumab — multiple candidate modifiers (pdl1, tmb, smoke, hist)
# ---------------------------------------------------------------
out = []
for f in ["pdl1_tps", "tmb_high", "smoke_current", "smoke_former", "hist_adeno", "egfr_mutation", "alk_fusion", "stk11_mutation"]:
    formula = (
        f"pfs_months ~ treatment_pembrolizumab * {f} + "
        + " + ".join([c for c in COVARS if c != f] + OTHER_TX["treatment_pembrolizumab"])
    )
    m = smf.ols(formula, data=df).fit()
    int_term = f"treatment_pembrolizumab:{f}"
    out.append({
        "feat": f,
        "interaction_beta": float(m.params[int_term]),
        "interaction_p": float(m.pvalues[int_term]),
        "tx_main_beta": float(m.params["treatment_pembrolizumab"]),
        "tx_main_p": float(m.pvalues["treatment_pembrolizumab"]),
    })
results["iter13_pembro_modifiers"] = out

# ---------------------------------------------------------------
# 14: Olaparib × BRCA2 (canonical) and STK11 / others
# ---------------------------------------------------------------
out = []
for f in ["brca2_mutation", "stk11_mutation", "egfr_mutation", "alk_fusion", "tmb_high", "kras_g12c"]:
    formula = (
        f"pfs_months ~ treatment_olaparib * {f} + "
        + " + ".join([c for c in COVARS if c != f] + OTHER_TX["treatment_olaparib"])
    )
    m = smf.ols(formula, data=df).fit()
    int_term = f"treatment_olaparib:{f}"
    out.append({
        "feat": f,
        "interaction_beta": float(m.params[int_term]),
        "interaction_p": float(m.pvalues[int_term]),
        "tx_main_beta": float(m.params["treatment_olaparib"]),
        "tx_main_p": float(m.pvalues["treatment_olaparib"]),
    })
results["iter14_olaparib_modifiers"] = out

# ---------------------------------------------------------------
# 15: Sotorasib subgroup refinement — within kras_g12c=1 by other features
# ---------------------------------------------------------------
sub_kras = df[df["kras_g12c"] == 1].copy()
out = []
# stratify within kras_g12c+ by sex, smoke, hist, stk11, brca2, alk_fusion
for f in ["sex_female", "smoke_current", "smoke_former", "hist_adeno", "stk11_mutation", "brca2_mutation", "alk_fusion", "egfr_mutation", "tmb_high", "stage_iv", "has_brain_mets"]:
    for v in [0, 1]:
        s = sub_kras[sub_kras[f] == v]
        if (s["treatment_sotorasib"] == 1).sum() < 20 or (s["treatment_sotorasib"] == 0).sum() < 20:
            out.append({"feat": f, "level": v, "n": int(len(s)), "skip": True})
            continue
        # within-subgroup t-test
        g0 = s.loc[s["treatment_sotorasib"] == 0, "pfs_months"]
        g1 = s.loc[s["treatment_sotorasib"] == 1, "pfs_months"]
        t = stats.ttest_ind(g1, g0, equal_var=False)
        # adjusted
        adj = adj_diff(
            s,
            "treatment_sotorasib",
            [c for c in COVARS if c != f and c != "kras_g12c"] + OTHER_TX["treatment_sotorasib"],
        )
        out.append({
            "feat": f,
            "level": v,
            "n": int(len(s)),
            "n_tx": int((s["treatment_sotorasib"] == 1).sum()),
            "unadj_diff": float(g1.mean() - g0.mean()),
            "unadj_p": float(t.pvalue),
            "adj_beta": adj["beta"] if adj else None,
            "adj_p": adj["p"] if adj else None,
        })
results["iter15_sotorasib_refine_within_kras"] = out

# ---------------------------------------------------------------
# 16: Three-way interactions for sotorasib: kras × <feat>
# ---------------------------------------------------------------
out = []
for f in ["sex_female", "smoke_current", "smoke_former", "hist_adeno", "stk11_mutation", "brca2_mutation", "alk_fusion", "egfr_mutation", "tmb_high"]:
    formula = (
        f"pfs_months ~ treatment_sotorasib * kras_g12c * {f} + "
        + " + ".join([c for c in COVARS if c not in {"kras_g12c", f}] + OTHER_TX["treatment_sotorasib"])
    )
    m = smf.ols(formula, data=df).fit()
    three_way = f"treatment_sotorasib:kras_g12c:{f}"
    out.append({
        "feat": f,
        "three_way_beta": float(m.params[three_way]) if three_way in m.params.index else None,
        "three_way_p": float(m.pvalues[three_way]) if three_way in m.params.index else None,
        "two_way_tx_kras_beta": float(m.params["treatment_sotorasib:kras_g12c"]),
        "two_way_tx_kras_p": float(m.pvalues["treatment_sotorasib:kras_g12c"]),
    })
results["iter16_sotorasib_threeway"] = out

# ---------------------------------------------------------------
# 17: Joint subgroup model — final candidate sotorasib subgroup
# Test: kras_g12c=1 AND sex_female=0 vs others
# ---------------------------------------------------------------
df["soto_subgroup_strict"] = (
    (df["kras_g12c"] == 1) & (df["sex_female"] == 0)
).astype(int)
df["soto_subgroup_kras_only"] = (df["kras_g12c"] == 1).astype(int)

# Compare effect of sotorasib × subgroup_strict vs subgroup_kras_only
out = []
for sg in ["soto_subgroup_kras_only", "soto_subgroup_strict"]:
    formula = (
        f"pfs_months ~ treatment_sotorasib * {sg} + "
        + " + ".join([c for c in COVARS if c not in {sg.replace('soto_subgroup_', ''), 'kras_g12c', 'sex_female'}] + OTHER_TX["treatment_sotorasib"])
    )
    m = smf.ols(formula, data=df).fit()
    int_term = f"treatment_sotorasib:{sg}"
    out.append({
        "subgroup": sg,
        "interaction_beta": float(m.params[int_term]),
        "interaction_p": float(m.pvalues[int_term]),
        "tx_main_beta": float(m.params["treatment_sotorasib"]),
        "tx_main_p": float(m.pvalues["treatment_sotorasib"]),
    })
# Direct: effect of sotorasib in subgroups (unadjusted)
for sg, label in [("kras_g12c", "kras_g12c==1"), ("soto_subgroup_strict", "kras+ AND male")]:
    s = df[df[sg] == 1]
    g0 = s.loc[s["treatment_sotorasib"] == 0, "pfs_months"]
    g1 = s.loc[s["treatment_sotorasib"] == 1, "pfs_months"]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    out.append({
        "subgroup": f"in {label}",
        "n": int(len(s)),
        "n_tx": int((s["treatment_sotorasib"] == 1).sum()),
        "unadj_diff": float(g1.mean() - g0.mean()),
        "unadj_p": float(t.pvalue),
    })
results["iter17_soto_joint_subgroup"] = out

# ---------------------------------------------------------------
# 18: 4-treatment heterogeneity check — pembro/olaparib/osimertinib re-examined
# Check if effects appear in the canonical biological subgroup that's small.
# Pembro × pdl1_tps high (>=0.5)
# ---------------------------------------------------------------
df["pdl1_high"] = (df["pdl1_tps"] >= 0.5).astype(int)
out = []
for f in ["pdl1_high", "tmb_high"]:
    for v in [0, 1]:
        s = df[df[f] == v]
        g0 = s.loc[s["treatment_pembrolizumab"] == 0, "pfs_months"]
        g1 = s.loc[s["treatment_pembrolizumab"] == 1, "pfs_months"]
        t = stats.ttest_ind(g1, g0, equal_var=False)
        out.append({
            "feat": f, "level": v, "n": int(len(s)),
            "diff": float(g1.mean() - g0.mean()),
            "p": float(t.pvalue),
        })
# also pdl1 continuous interaction
formula = (
    "pfs_months ~ treatment_pembrolizumab * pdl1_high + "
    + " + ".join([c for c in COVARS if c != "pdl1_tps"] + OTHER_TX["treatment_pembrolizumab"])
)
m = smf.ols(formula, data=df).fit()
out.append({
    "feat": "pdl1_high (>=0.5) interaction",
    "beta": float(m.params["treatment_pembrolizumab:pdl1_high"]),
    "p": float(m.pvalues["treatment_pembrolizumab:pdl1_high"]),
})
results["iter18_pembro_pdl1"] = out

# ---------------------------------------------------------------
# 19: Olaparib × brca2 mutation refinement
# ---------------------------------------------------------------
out = []
for v in [0, 1]:
    s = df[df["brca2_mutation"] == v]
    g0 = s.loc[s["treatment_olaparib"] == 0, "pfs_months"]
    g1 = s.loc[s["treatment_olaparib"] == 1, "pfs_months"]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    out.append({
        "brca2_mutation": v,
        "n": int(len(s)),
        "n_tx": int((s["treatment_olaparib"] == 1).sum()),
        "diff": float(g1.mean() - g0.mean()),
        "p": float(t.pvalue),
    })
# Also try refined: brca2+ AND additional features
formula = (
    "pfs_months ~ treatment_olaparib * brca2_mutation + "
    + " + ".join([c for c in COVARS if c != "brca2_mutation"] + OTHER_TX["treatment_olaparib"])
)
m = smf.ols(formula, data=df).fit()
out.append({
    "interaction": "olaparib × brca2",
    "beta": float(m.params["treatment_olaparib:brca2_mutation"]),
    "p": float(m.pvalues["treatment_olaparib:brca2_mutation"]),
})
results["iter19_olaparib_brca"] = out

# ---------------------------------------------------------------
# 20: Tree/decision-rule heterogeneity exploration via causal subgrouping.
# For each treatment, brute-force search all single-feature splits and
# pairs of binary features for the best subgroup with significant effect.
# ---------------------------------------------------------------
BIN_FEATS = [
    "sex_female",
    "stage_iv",
    "has_brain_mets",
    "egfr_mutation",
    "kras_g12c",
    "alk_fusion",
    "stk11_mutation",
    "brca2_mutation",
    "tmb_high",
    "smoke_current",
    "smoke_former",
    "hist_adeno",
    "pdl1_high",
]
out = {}
for tx in ["treatment_pembrolizumab", "treatment_sotorasib", "treatment_olaparib", "treatment_osimertinib"]:
    rows = []
    # single feature
    for f in BIN_FEATS:
        for v in [0, 1]:
            s = df[df[f] == v]
            n_tx = int((s[tx] == 1).sum())
            n_ctrl = int((s[tx] == 0).sum())
            if n_tx < 100 or n_ctrl < 100:
                continue
            g0 = s.loc[s[tx] == 0, "pfs_months"]
            g1 = s.loc[s[tx] == 1, "pfs_months"]
            t = stats.ttest_ind(g1, g0, equal_var=False)
            rows.append({
                "rule": f"{f}={v}",
                "n": int(len(s)),
                "diff": float(g1.mean() - g0.mean()),
                "p": float(t.pvalue),
            })
    # pairs
    for (f1, v1), (f2, v2) in combinations([(f, v) for f in BIN_FEATS for v in [0, 1]], 2):
        if f1 == f2:
            continue
        s = df[(df[f1] == v1) & (df[f2] == v2)]
        n_tx = int((s[tx] == 1).sum())
        n_ctrl = int((s[tx] == 0).sum())
        if n_tx < 100 or n_ctrl < 100:
            continue
        g0 = s.loc[s[tx] == 0, "pfs_months"]
        g1 = s.loc[s[tx] == 1, "pfs_months"]
        t = stats.ttest_ind(g1, g0, equal_var=False)
        rows.append({
            "rule": f"{f1}={v1} & {f2}={v2}",
            "n": int(len(s)),
            "diff": float(g1.mean() - g0.mean()),
            "p": float(t.pvalue),
        })
    rows.sort(key=lambda r: -abs(r["diff"]) if r["p"] < 0.001 else 0)
    out[tx] = rows[:25]
results["iter20_subgroup_search"] = out

Path("my_refine.json").write_text(json.dumps(results, indent=2, default=float))
print("Wrote my_refine.json")
