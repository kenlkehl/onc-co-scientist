"""Iterations 21-25: final refinement of the sotorasib subgroup, plus
sensitivity checks for the null treatments."""
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
df["pdl1_high"] = (df["pdl1_tps"] >= 0.5).astype(int)

OTHER_TX_SOTO = ["treatment_pembrolizumab", "treatment_olaparib", "treatment_osimertinib"]
COVARS = [
    "age_years", "ecog_ps", "stage_iv", "has_brain_mets",
    "albumin_g_dl", "ldh_u_l", "nlr", "weight_loss_pct_6mo",
    "smoke_current", "smoke_former", "hist_adeno",
    "egfr_mutation", "alk_fusion", "stk11_mutation", "brca2_mutation",
    "tmb_high", "pdl1_tps",
]

results = {}

# ---------------------------------------------------------------
# 21: Within kras+ AND male: stratify by brca2 and alk_fusion
# ---------------------------------------------------------------
sub = df[(df["kras_g12c"] == 1) & (df["sex_female"] == 0)].copy()
out = []
for f in ["brca2_mutation", "alk_fusion", "stk11_mutation", "egfr_mutation",
          "tmb_high", "smoke_current", "smoke_former", "hist_adeno",
          "stage_iv", "has_brain_mets"]:
    for v in [0, 1]:
        s = sub[sub[f] == v]
        n_tx = (s["treatment_sotorasib"] == 1).sum()
        n_ctrl = (s["treatment_sotorasib"] == 0).sum()
        if n_tx < 20 or n_ctrl < 20:
            out.append({"feat": f, "level": v, "n": int(len(s)), "skip": True})
            continue
        g0 = s.loc[s["treatment_sotorasib"] == 0, "pfs_months"]
        g1 = s.loc[s["treatment_sotorasib"] == 1, "pfs_months"]
        t = stats.ttest_ind(g1, g0, equal_var=False)
        out.append({
            "feat": f, "level": v, "n": int(len(s)),
            "n_tx": int(n_tx),
            "diff": float(g1.mean() - g0.mean()),
            "p": float(t.pvalue),
        })
results["iter21_in_krasplus_male"] = out

# ---------------------------------------------------------------
# 22: Final candidate subgroup: kras+ & male & brca2- & alk_fusion-
# ---------------------------------------------------------------
strict = (
    (df["kras_g12c"] == 1)
    & (df["sex_female"] == 0)
    & (df["brca2_mutation"] == 0)
    & (df["alk_fusion"] == 0)
).astype(int)
df["strict_grp"] = strict
out = []
# Effect inside the strict group
s = df[df["strict_grp"] == 1]
g0 = s.loc[s["treatment_sotorasib"] == 0, "pfs_months"]
g1 = s.loc[s["treatment_sotorasib"] == 1, "pfs_months"]
t = stats.ttest_ind(g1, g0, equal_var=False)
out.append({
    "subgroup": "kras+ & male & brca2- & alk_fusion-",
    "n": int(len(s)),
    "n_tx": int((s["treatment_sotorasib"] == 1).sum()),
    "unadj_diff": float(g1.mean() - g0.mean()),
    "unadj_p": float(t.pvalue),
})
# Adjusted within
formula = (
    "pfs_months ~ treatment_sotorasib + "
    + " + ".join(COVARS + OTHER_TX_SOTO)
)
m = smf.ols(formula, data=s).fit()
out.append({
    "subgroup": "kras+ & male & brca2- & alk_fusion- (adj)",
    "beta": float(m.params["treatment_sotorasib"]),
    "p": float(m.pvalues["treatment_sotorasib"]),
})
# Effect outside strict group
s2 = df[df["strict_grp"] == 0]
g0 = s2.loc[s2["treatment_sotorasib"] == 0, "pfs_months"]
g1 = s2.loc[s2["treatment_sotorasib"] == 1, "pfs_months"]
t = stats.ttest_ind(g1, g0, equal_var=False)
out.append({
    "subgroup": "complement",
    "n": int(len(s2)),
    "diff": float(g1.mean() - g0.mean()),
    "p": float(t.pvalue),
})
# Compare strict vs kras+&male only
s_male = df[(df["kras_g12c"] == 1) & (df["sex_female"] == 0)]
g0 = s_male.loc[s_male["treatment_sotorasib"] == 0, "pfs_months"]
g1 = s_male.loc[s_male["treatment_sotorasib"] == 1, "pfs_months"]
t = stats.ttest_ind(g1, g0, equal_var=False)
out.append({
    "subgroup": "kras+ & male only",
    "n": int(len(s_male)),
    "diff": float(g1.mean() - g0.mean()),
    "p": float(t.pvalue),
})
# Interaction with strict group
formula = (
    "pfs_months ~ treatment_sotorasib * strict_grp + "
    + " + ".join([c for c in COVARS] + OTHER_TX_SOTO)
)
m = smf.ols(formula, data=df).fit()
out.append({
    "interaction": "treatment_sotorasib × strict_grp",
    "beta": float(m.params["treatment_sotorasib:strict_grp"]),
    "p": float(m.pvalues["treatment_sotorasib:strict_grp"]),
    "tx_main_beta": float(m.params["treatment_sotorasib"]),
    "tx_main_p": float(m.pvalues["treatment_sotorasib"]),
})
results["iter22_strict_subgroup"] = out

# ---------------------------------------------------------------
# 23: Confirm null-treatment findings with broader search:
# any 3-way interaction with two binary features that produces a
# significant interaction term for pembro/olaparib/osimertinib?
# ---------------------------------------------------------------
BIN_FEATS = [
    "sex_female", "stage_iv", "has_brain_mets",
    "egfr_mutation", "kras_g12c", "alk_fusion", "stk11_mutation",
    "brca2_mutation", "tmb_high", "smoke_current", "smoke_former",
    "hist_adeno", "pdl1_high",
]
out_null = {}
for tx in ["treatment_pembrolizumab", "treatment_olaparib", "treatment_osimertinib"]:
    rows = []
    for (f1, v1), (f2, v2) in combinations([(f, v) for f in BIN_FEATS for v in [0, 1]], 2):
        if f1 == f2:
            continue
        s = df[(df[f1] == v1) & (df[f2] == v2)]
        if (s[tx] == 1).sum() < 50 or (s[tx] == 0).sum() < 50:
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
    rows.sort(key=lambda r: r["p"])
    out_null[tx] = rows[:15]
results["iter23_null_tx_subgroup_search"] = out_null

# ---------------------------------------------------------------
# 24: Multiplicative robustness: within kras+ & male, full OLS w/ tx
# ---------------------------------------------------------------
out = []
sub = df[(df["kras_g12c"] == 1) & (df["sex_female"] == 0)].copy()
formula = (
    "pfs_months ~ treatment_sotorasib + "
    + " + ".join([c for c in COVARS if c != "kras_g12c"] + OTHER_TX_SOTO)
)
m = smf.ols(formula, data=sub).fit()
out.append({
    "subgroup": "kras+ & male, full OLS",
    "n": int(len(sub)),
    "tx_beta": float(m.params["treatment_sotorasib"]),
    "tx_p": float(m.pvalues["treatment_sotorasib"]),
})
# Within kras+ & female
sub = df[(df["kras_g12c"] == 1) & (df["sex_female"] == 1)].copy()
m = smf.ols(formula, data=sub).fit()
out.append({
    "subgroup": "kras+ & female, full OLS",
    "n": int(len(sub)),
    "tx_beta": float(m.params["treatment_sotorasib"]),
    "tx_p": float(m.pvalues["treatment_sotorasib"]),
})
results["iter24_robustness"] = out

# ---------------------------------------------------------------
# 25: Symmetry — does sex modify other treatments? (sanity)
# ---------------------------------------------------------------
out = []
for tx in ["treatment_pembrolizumab", "treatment_olaparib", "treatment_osimertinib", "treatment_sotorasib"]:
    formula = (
        f"pfs_months ~ {tx} * sex_female + "
        + " + ".join([c for c in COVARS] + [t for t in ["treatment_pembrolizumab","treatment_olaparib","treatment_osimertinib","treatment_sotorasib"] if t != tx])
    )
    m = smf.ols(formula, data=df).fit()
    int_term = f"{tx}:sex_female"
    out.append({
        "tx": tx,
        "interaction_beta": float(m.params[int_term]),
        "interaction_p": float(m.pvalues[int_term]),
        "tx_main_beta": float(m.params[tx]),
        "tx_main_p": float(m.pvalues[tx]),
    })
results["iter25_sex_symmetry"] = out

Path("my_refine2.json").write_text(json.dumps(results, indent=2, default=float))
print("Wrote my_refine2.json")
