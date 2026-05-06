"""Fresh, self-contained analysis for ds001_crc.

Runs main effects, treatment-biomarker interactions, and exhaustive
heterogeneity searches.  Writes structured results that the transcript
builder will consume.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).with_name("dataset.parquet")
OUT_PATH = Path(__file__).with_name("fresh_results.json")

df = pd.read_parquet(DATA_PATH)
print("Loaded", df.shape)

biomarkers = [
    "kras_mutation",
    "nras_mutation",
    "braf_v600e",
    "msi_high",
    "her2_amplified",
    "ntrk_fusion",
]
treatments = [
    "cetuximab",
    "bevacizumab",
    "pembrolizumab",
    "encorafenib",
    "trastuzumab_tucatinib",
    "regorafenib",
]
binary_features = biomarkers + ["sex_female", "stage_iv", "right_sided_primary"]
continuous_features = [
    "age_years",
    "ecog_ps",
    "cea_ng_ml",
    "albumin_g_dl",
    "ldh_u_l",
    "weight_loss_pct_6mo",
    "crp_mg_l",
    "nlr",
    "hemoglobin_g_dl",
    "alkaline_phosphatase_u_l",
    "ast_u_l",
    "alt_u_l",
    "total_bilirubin_mg_dl",
    "creatinine_mg_dl",
    "bun_mg_dl",
    "sodium_meq_l",
    "potassium_meq_l",
    "calcium_mg_dl",
]

results = {}


def ttest(g1, g0):
    g1 = np.asarray(g1)
    g0 = np.asarray(g0)
    if len(g1) < 2 or len(g0) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    return float(t), float(p)


def mean_diff(g1, g0):
    return float(np.mean(g1) - np.mean(g0))


# 1. Main effect of every binary feature on PFS
main_binary = {}
for b in binary_features + [f"treatment_{t}" for t in treatments]:
    g1 = df.loc[df[b] == 1, "pfs_months"]
    g0 = df.loc[df[b] == 0, "pfs_months"]
    t, p = ttest(g1, g0)
    main_binary[b] = {
        "n_pos": int((df[b] == 1).sum()),
        "n_neg": int((df[b] == 0).sum()),
        "mean_pos": float(g1.mean()),
        "mean_neg": float(g0.mean()),
        "diff": mean_diff(g1, g0),
        "t": t,
        "p": p,
    }
results["main_binary"] = main_binary

# 2. Main effect of every continuous feature on PFS via linear regression
main_continuous = {}
for c in continuous_features:
    X = sm.add_constant(df[[c]])
    res = sm.OLS(df["pfs_months"], X).fit()
    main_continuous[c] = {
        "beta": float(res.params[c]),
        "se": float(res.bse[c]),
        "p": float(res.pvalues[c]),
        "spearman_r": float(stats.spearmanr(df[c], df["pfs_months"]).statistic),
    }
results["main_continuous"] = main_continuous

# 3. Treatment x biomarker interactions
tb_inter = {}
for t in treatments:
    tcol = f"treatment_{t}"
    for b in biomarkers + ["right_sided_primary", "stage_iv", "sex_female"]:
        # Subset means
        groups = {}
        for tv in (0, 1):
            for bv in (0, 1):
                sub = df[(df[tcol] == tv) & (df[b] == bv)]
                groups[(tv, bv)] = {
                    "n": int(len(sub)),
                    "mean": float(sub["pfs_months"].mean()) if len(sub) else float("nan"),
                }
        # Interaction model
        sub = df[[tcol, b, "pfs_months"]].copy()
        sub["int"] = sub[tcol] * sub[b]
        X = sm.add_constant(sub[[tcol, b, "int"]])
        res = sm.OLS(sub["pfs_months"], X).fit()
        tb_inter[f"{t}__{b}"] = {
            "treatment": t,
            "modifier": b,
            "groups": {f"{k[0]}{k[1]}": v for k, v in groups.items()},
            "beta_t": float(res.params[tcol]),
            "beta_b": float(res.params[b]),
            "beta_int": float(res.params["int"]),
            "p_int": float(res.pvalues["int"]),
            # Effect of treatment in biomarker-positive vs negative
            "te_in_bpos": groups[(1, 1)]["mean"] - groups[(0, 1)]["mean"],
            "te_in_bneg": groups[(1, 0)]["mean"] - groups[(0, 0)]["mean"],
        }
results["tx_biomarker_interactions"] = tb_inter

# 4. Treatment x continuous-feature interactions (median-split)
tc_inter = {}
for t in treatments:
    tcol = f"treatment_{t}"
    for c in continuous_features:
        med = df[c].median()
        high = df[c] >= med
        # Two-way interaction model with continuous
        X = sm.add_constant(pd.DataFrame({
            "t": df[tcol].astype(float),
            "c": df[c].astype(float),
            "tc": df[tcol].astype(float) * df[c].astype(float),
        }))
        res = sm.OLS(df["pfs_months"], X).fit()
        # Stratified treatment effect
        te_high = (
            df.loc[(df[tcol] == 1) & high, "pfs_months"].mean()
            - df.loc[(df[tcol] == 0) & high, "pfs_months"].mean()
        )
        te_low = (
            df.loc[(df[tcol] == 1) & ~high, "pfs_months"].mean()
            - df.loc[(df[tcol] == 0) & ~high, "pfs_months"].mean()
        )
        tc_inter[f"{t}__{c}"] = {
            "treatment": t,
            "modifier": c,
            "beta_t": float(res.params["t"]),
            "beta_c": float(res.params["c"]),
            "beta_int": float(res.params["tc"]),
            "p_int": float(res.pvalues["tc"]),
            "te_high": float(te_high),
            "te_low": float(te_low),
        }
results["tx_continuous_interactions"] = tc_inter

# 5. Two-way subgroup search for each treatment within biomarker-positive cohorts
two_way = {}
for t in treatments:
    tcol = f"treatment_{t}"
    for b1 in biomarkers + ["right_sided_primary", "stage_iv"]:
        sub = df[df[b1] == 1]
        if len(sub) < 200:
            continue
        if (sub[tcol] == 1).sum() < 50 or (sub[tcol] == 0).sum() < 50:
            continue
        # Within b1+, look for second modifier
        for b2 in biomarkers + ["right_sided_primary", "stage_iv", "sex_female"]:
            if b2 == b1:
                continue
            for v2 in (0, 1):
                sg = sub[sub[b2] == v2]
                if (sg[tcol] == 1).sum() < 30 or (sg[tcol] == 0).sum() < 30:
                    continue
                te = sg.loc[sg[tcol] == 1, "pfs_months"].mean() - sg.loc[sg[tcol] == 0, "pfs_months"].mean()
                t_, p_ = ttest(
                    sg.loc[sg[tcol] == 1, "pfs_months"],
                    sg.loc[sg[tcol] == 0, "pfs_months"],
                )
                two_way[f"{t}|{b1}=1&{b2}={v2}"] = {
                    "treatment": t,
                    "subgroup": f"{b1}=1 & {b2}={v2}",
                    "n_treated": int((sg[tcol] == 1).sum()),
                    "n_untreated": int((sg[tcol] == 0).sum()),
                    "te": float(te),
                    "p": p_,
                }
results["two_way_subgroups"] = two_way

# 6. Within biomarker-positive, screen continuous modifiers
cont_in_bm = {}
for t, b_target in [
    ("cetuximab", None),  # all
    ("bevacizumab", None),
    ("pembrolizumab", "msi_high"),
    ("encorafenib", "braf_v600e"),
    ("trastuzumab_tucatinib", "her2_amplified"),
    ("regorafenib", None),
]:
    tcol = f"treatment_{t}"
    sub = df if b_target is None else df[df[b_target] == 1]
    if len(sub) < 100:
        continue
    for c in continuous_features:
        med = sub[c].median()
        sg_high = sub[sub[c] >= med]
        sg_low = sub[sub[c] < med]
        te_high = (
            sg_high.loc[sg_high[tcol] == 1, "pfs_months"].mean()
            - sg_high.loc[sg_high[tcol] == 0, "pfs_months"].mean()
        )
        te_low = (
            sg_low.loc[sg_low[tcol] == 1, "pfs_months"].mean()
            - sg_low.loc[sg_low[tcol] == 0, "pfs_months"].mean()
        )
        # Interaction p in this subgroup
        try:
            X = sm.add_constant(pd.DataFrame({
                "t": sub[tcol].astype(float),
                "c": sub[c].astype(float),
                "tc": sub[tcol].astype(float) * sub[c].astype(float),
            }))
            res = sm.OLS(sub["pfs_months"], X).fit()
            p_int = float(res.pvalues["tc"])
            beta_int = float(res.params["tc"])
        except Exception:
            p_int = float("nan")
            beta_int = float("nan")
        cont_in_bm[f"{t}|{b_target}|{c}"] = {
            "treatment": t,
            "biomarker_anchor": b_target,
            "modifier": c,
            "te_high": float(te_high) if not np.isnan(te_high) else None,
            "te_low": float(te_low) if not np.isnan(te_low) else None,
            "p_int": p_int,
            "beta_int": beta_int,
        }
results["continuous_within_biomarker"] = cont_in_bm

# 7. Specifically look for the "best" two-feature subgroup definition for each treatment
#    that maximises treatment effect.  Use a grid over biomarker-positive anchors and
#    binary modifiers (including median-split continuous).
best_subgroups = {}
for t in treatments:
    tcol = f"treatment_{t}"
    candidates = []
    # Anchor on each biomarker (pos and the absence of all biomarkers)
    anchor_options = [(b, 1) for b in biomarkers]
    for a_name, a_val in anchor_options + [("__all__", None)]:
        if a_val is None:
            sub_anchor = df
            a_label = "all_patients"
        else:
            sub_anchor = df[df[a_name] == a_val]
            a_label = f"{a_name}={a_val}"
        if len(sub_anchor) < 200:
            continue
        if (sub_anchor[tcol] == 1).sum() < 50 or (sub_anchor[tcol] == 0).sum() < 50:
            continue
        # Treatment effect in anchor
        te_anchor = (
            sub_anchor.loc[sub_anchor[tcol] == 1, "pfs_months"].mean()
            - sub_anchor.loc[sub_anchor[tcol] == 0, "pfs_months"].mean()
        )
        candidates.append({
            "subgroup": a_label,
            "n_treated": int((sub_anchor[tcol] == 1).sum()),
            "n_untreated": int((sub_anchor[tcol] == 0).sum()),
            "te": float(te_anchor),
        })
        # Add a second binary predicate (presence of biomarker, or sex/right/stage/ecog>=1)
        modifiers = (
            [(b, 1) for b in biomarkers if b != a_name]
            + [(b, 0) for b in biomarkers if b != a_name]
            + [("right_sided_primary", 1), ("right_sided_primary", 0)]
            + [("stage_iv", 1), ("stage_iv", 0)]
            + [("sex_female", 1), ("sex_female", 0)]
        )
        for m_name, m_val in modifiers:
            sg = sub_anchor[sub_anchor[m_name] == m_val]
            if (sg[tcol] == 1).sum() < 30 or (sg[tcol] == 0).sum() < 30:
                continue
            te = (
                sg.loc[sg[tcol] == 1, "pfs_months"].mean()
                - sg.loc[sg[tcol] == 0, "pfs_months"].mean()
            )
            t_, p_ = ttest(
                sg.loc[sg[tcol] == 1, "pfs_months"],
                sg.loc[sg[tcol] == 0, "pfs_months"],
            )
            candidates.append({
                "subgroup": f"{a_label} & {m_name}={m_val}",
                "n_treated": int((sg[tcol] == 1).sum()),
                "n_untreated": int((sg[tcol] == 0).sum()),
                "te": float(te),
                "p": p_,
            })
    # Sort by TE
    candidates_sorted = sorted(candidates, key=lambda x: x["te"], reverse=True)
    best_subgroups[t] = candidates_sorted[:15]
results["best_subgroups"] = best_subgroups

# 8. Treatment-biomarker pairs known mechanism: detail
mechanism_pairs = [
    ("cetuximab", "kras_mutation"),
    ("cetuximab", "nras_mutation"),
    ("cetuximab", "braf_v600e"),
    ("cetuximab", "right_sided_primary"),
    ("pembrolizumab", "msi_high"),
    ("encorafenib", "braf_v600e"),
    ("trastuzumab_tucatinib", "her2_amplified"),
    ("regorafenib", "kras_mutation"),
    ("bevacizumab", "kras_mutation"),
]
mech = {}
for t, b in mechanism_pairs:
    tcol = f"treatment_{t}"
    grid = {}
    for tv in (0, 1):
        for bv in (0, 1):
            sub = df[(df[tcol] == tv) & (df[b] == bv)]
            grid[(tv, bv)] = (int(len(sub)), float(sub["pfs_months"].mean()) if len(sub) else float("nan"))
    te_bpos = grid[(1, 1)][1] - grid[(0, 1)][1]
    te_bneg = grid[(1, 0)][1] - grid[(0, 0)][1]
    sub = df[[tcol, b, "pfs_months"]].copy()
    sub["int"] = sub[tcol] * sub[b]
    X = sm.add_constant(sub[[tcol, b, "int"]])
    res = sm.OLS(sub["pfs_months"], X).fit()
    mech[f"{t}__{b}"] = {
        "treatment": t,
        "modifier": b,
        "grid": {f"t{k[0]}_b{k[1]}": {"n": v[0], "mean": v[1]} for k, v in grid.items()},
        "te_bpos": te_bpos,
        "te_bneg": te_bneg,
        "p_int": float(res.pvalues["int"]),
        "beta_int": float(res.params["int"]),
    }
results["mechanism_pairs"] = mech

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("Wrote", OUT_PATH)
