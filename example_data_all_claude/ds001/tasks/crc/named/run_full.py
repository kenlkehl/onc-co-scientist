"""Run all analyses for the CRC ds001 task and dump results to JSON.

Iterations are organized to follow the propose -> test -> refine pattern.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
N = len(df)

results = {}


def fmt(x, p=4):
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    return float(round(float(x), p))


def ttest_means(y, mask):
    a = y[mask]
    b = y[~mask]
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "n_pos": int(mask.sum()),
        "n_neg": int((~mask).sum()),
        "mean_pos": fmt(a.mean()),
        "mean_neg": fmt(b.mean()),
        "diff": fmt(a.mean() - b.mean()),
        "p_value": fmt(p, 6),
        "t": fmt(t),
    }


def ols_with(y_col, x_cols):
    X = df[x_cols].copy()
    X = sm.add_constant(X)
    y = df[y_col]
    m = sm.OLS(y, X).fit()
    out = {"r2": fmt(m.rsquared)}
    for c in m.params.index:
        out[c] = {
            "coef": fmt(m.params[c]),
            "p": fmt(m.pvalues[c], 6),
        }
    return out


# ===== Iteration 1: Treatment main effects (univariate) =====
treatments = [
    "treatment_cetuximab",
    "treatment_bevacizumab",
    "treatment_pembrolizumab",
    "treatment_encorafenib",
    "treatment_trastuzumab_tucatinib",
    "treatment_regorafenib",
]
results["it1_treatment_main"] = {}
for t in treatments:
    results["it1_treatment_main"][t] = ttest_means(df["pfs_months"], df[t] == 1)

# ===== Iteration 2: Demographic / clinical main effects =====
results["it2_clinical"] = {}
for f in ["sex_female", "stage_iv", "right_sided_primary"]:
    results["it2_clinical"][f] = ttest_means(df["pfs_months"], df[f] == 1)
mod = smf.ols("pfs_months ~ C(ecog_ps)", data=df).fit()
results["it2_clinical"]["ecog_ps_anova"] = {
    "f": fmt(mod.fvalue),
    "p": fmt(mod.f_pvalue, 8),
    "ecog0_mean": fmt(df.loc[df.ecog_ps == 0, "pfs_months"].mean()),
    "ecog1_mean": fmt(df.loc[df.ecog_ps == 1, "pfs_months"].mean()),
    "ecog2_mean": fmt(df.loc[df.ecog_ps == 2, "pfs_months"].mean()),
}
for f in ["age_years", "weight_loss_pct_6mo"]:
    r, p = stats.pearsonr(df[f], df["pfs_months"])
    results["it2_clinical"][f] = {"r": fmt(r), "p": fmt(p, 8)}

# ===== Iteration 3: Biomarker main effects =====
results["it3_biomarkers"] = {}
for b in [
    "kras_mutation",
    "nras_mutation",
    "braf_v600e",
    "msi_high",
    "her2_amplified",
    "ntrk_fusion",
]:
    results["it3_biomarkers"][b] = ttest_means(df["pfs_months"], df[b] == 1)

# ===== Iteration 4: Lab main effects =====
results["it4_labs"] = {}
for f in [
    "cea_ng_ml",
    "albumin_g_dl",
    "ldh_u_l",
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
]:
    r, p = stats.pearsonr(df[f], df["pfs_months"])
    results["it4_labs"][f] = {"r": fmt(r), "p": fmt(p, 8)}

# ===== Iteration 5: Multivariable model with all main effects =====
mv_cols = [
    "age_years",
    "sex_female",
    "ecog_ps",
    "stage_iv",
    "right_sided_primary",
    "kras_mutation",
    "nras_mutation",
    "braf_v600e",
    "msi_high",
    "her2_amplified",
    "ntrk_fusion",
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
] + treatments
results["it5_multivariable"] = ols_with("pfs_months", mv_cols)


# ===== Helper for interactions =====
def interact(treatment, modifier, label):
    sub = df.copy()
    sub["t"] = sub[treatment]
    sub["m"] = sub[modifier]
    sub["tm"] = sub["t"] * sub["m"]
    m = sm.OLS(sub["pfs_months"], sm.add_constant(sub[["t", "m", "tm"]])).fit()
    cells = {}
    for tv in [0, 1]:
        for mv in [0, 1]:
            mask = (sub["t"] == tv) & (sub["m"] == mv)
            cells[f"t{tv}_m{mv}"] = {
                "n": int(mask.sum()),
                "mean": fmt(sub.loc[mask, "pfs_months"].mean()),
            }
    eff_pos = (
        cells["t1_m1"]["mean"] - cells["t0_m1"]["mean"]
        if cells["t1_m1"]["mean"] is not None and cells["t0_m1"]["mean"] is not None
        else None
    )
    eff_neg = (
        cells["t1_m0"]["mean"] - cells["t0_m0"]["mean"]
        if cells["t1_m0"]["mean"] is not None and cells["t0_m0"]["mean"] is not None
        else None
    )
    return {
        "label": label,
        "treatment": treatment,
        "modifier": modifier,
        "main_t": {"coef": fmt(m.params["t"]), "p": fmt(m.pvalues["t"], 6)},
        "main_m": {"coef": fmt(m.params["m"]), "p": fmt(m.pvalues["m"], 6)},
        "interaction": {"coef": fmt(m.params["tm"]), "p": fmt(m.pvalues["tm"], 6)},
        "cells": cells,
        "effect_in_pos": fmt(eff_pos) if eff_pos is not None else None,
        "effect_in_neg": fmt(eff_neg) if eff_neg is not None else None,
    }


# ===== Iteration 6: Cetuximab interactions =====
results["it6_cetux_kras"] = interact("treatment_cetuximab", "kras_mutation", "Cetuximab x KRAS")
results["it6_cetux_nras"] = interact("treatment_cetuximab", "nras_mutation", "Cetuximab x NRAS")
results["it6_cetux_braf"] = interact("treatment_cetuximab", "braf_v600e", "Cetuximab x BRAF V600E")

# ===== Iteration 7: Pembrolizumab x MSI =====
results["it7_pembro_msi"] = interact(
    "treatment_pembrolizumab", "msi_high", "Pembrolizumab x MSI-H"
)
results["it7_pembro_kras"] = interact(
    "treatment_pembrolizumab", "kras_mutation", "Pembrolizumab x KRAS"
)

# ===== Iteration 8: Encorafenib x BRAF V600E =====
results["it8_encora_braf"] = interact(
    "treatment_encorafenib", "braf_v600e", "Encorafenib x BRAF V600E"
)
results["it8_encora_kras"] = interact(
    "treatment_encorafenib", "kras_mutation", "Encorafenib x KRAS"
)

# ===== Iteration 9: HER2 =====
results["it9_her2_combo"] = interact(
    "treatment_trastuzumab_tucatinib", "her2_amplified", "Trastuzumab/tucatinib x HER2"
)

# ===== Iteration 10: Bevacizumab =====
results["it10_bev_kras"] = interact("treatment_bevacizumab", "kras_mutation", "Bev x KRAS")
results["it10_bev_braf"] = interact("treatment_bevacizumab", "braf_v600e", "Bev x BRAF")
results["it10_bev_right"] = interact(
    "treatment_bevacizumab", "right_sided_primary", "Bev x right-sided"
)

# ===== Iteration 11: Regorafenib =====
results["it11_rego_main"] = ttest_means(df["pfs_months"], df["treatment_regorafenib"] == 1)
for b in ["kras_mutation", "braf_v600e", "msi_high", "her2_amplified"]:
    results[f"it11_rego_{b}"] = interact("treatment_regorafenib", b, f"Rego x {b}")

# ===== Iteration 12: NTRK =====
results["it12_ntrk"] = ttest_means(df["pfs_months"], df["ntrk_fusion"] == 1)


# ===== Subgroup helper =====
def subgroup_ttest(predicate, treatment, label):
    sub = df[predicate]
    if sub[treatment].sum() < 10 or (sub[treatment] == 0).sum() < 10:
        return {
            "label": label,
            "n_total": int(len(sub)),
            "n_treated": int(sub[treatment].sum()) if len(sub) else 0,
            "note": "insufficient data",
        }
    a = sub.loc[sub[treatment] == 1, "pfs_months"]
    b = sub.loc[sub[treatment] == 0, "pfs_months"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "label": label,
        "treatment": treatment,
        "n_total": int(len(sub)),
        "n_treated": int(sub[treatment].sum()),
        "mean_treated": fmt(a.mean()),
        "mean_untreated": fmt(b.mean()),
        "diff": fmt(a.mean() - b.mean()),
        "p_value": fmt(p, 8),
    }


# ===== Iteration 13: Cetuximab subgroups =====
pred_panwt = (
    (df["kras_mutation"] == 0) & (df["nras_mutation"] == 0) & (df["braf_v600e"] == 0)
)
results["it13_cetux_panwt"] = subgroup_ttest(
    pred_panwt, "treatment_cetuximab", "Cetuximab in KRAS/NRAS/BRAF all wild-type"
)
results["it13_cetux_kras_only"] = subgroup_ttest(
    df["kras_mutation"] == 0, "treatment_cetuximab", "Cetuximab in KRAS wild-type"
)
results["it13_cetux_kras_mut"] = subgroup_ttest(
    df["kras_mutation"] == 1, "treatment_cetuximab", "Cetuximab in KRAS mutant"
)
results["it13_cetux_nras"] = subgroup_ttest(
    df["nras_mutation"] == 1, "treatment_cetuximab", "Cetuximab in NRAS mutant"
)
results["it13_cetux_braf"] = subgroup_ttest(
    df["braf_v600e"] == 1, "treatment_cetuximab", "Cetuximab in BRAF V600E mutant"
)
results["it13_cetux_panwt_left"] = subgroup_ttest(
    pred_panwt & (df["right_sided_primary"] == 0),
    "treatment_cetuximab",
    "Cetuximab in panRAS/BRAF wt + LEFT-sided",
)
results["it13_cetux_panwt_right"] = subgroup_ttest(
    pred_panwt & (df["right_sided_primary"] == 1),
    "treatment_cetuximab",
    "Cetuximab in panRAS/BRAF wt + RIGHT-sided",
)

# ===== Iteration 14: Pembrolizumab subgroup =====
results["it14_pembro_msi_pos"] = subgroup_ttest(
    df["msi_high"] == 1, "treatment_pembrolizumab", "Pembrolizumab in MSI-high"
)
results["it14_pembro_msi_neg"] = subgroup_ttest(
    df["msi_high"] == 0, "treatment_pembrolizumab", "Pembrolizumab in MSI-low"
)

# ===== Iteration 15: Encorafenib subgroup =====
results["it15_encora_braf_pos"] = subgroup_ttest(
    df["braf_v600e"] == 1, "treatment_encorafenib", "Encorafenib in BRAF V600E"
)
results["it15_encora_braf_neg"] = subgroup_ttest(
    df["braf_v600e"] == 0, "treatment_encorafenib", "Encorafenib in BRAF wt"
)
results["it15_encora_braf_kraswt"] = subgroup_ttest(
    (df["braf_v600e"] == 1) & (df["kras_mutation"] == 0),
    "treatment_encorafenib",
    "Encorafenib in BRAF V600E + KRAS wt",
)

# ===== Iteration 16: HER2 =====
results["it16_her2_pos"] = subgroup_ttest(
    df["her2_amplified"] == 1,
    "treatment_trastuzumab_tucatinib",
    "Trastuzumab/tucatinib in HER2+",
)
results["it16_her2_neg"] = subgroup_ttest(
    df["her2_amplified"] == 0,
    "treatment_trastuzumab_tucatinib",
    "Trastuzumab/tucatinib in HER2-",
)
results["it16_her2_pos_kraswt"] = subgroup_ttest(
    (df["her2_amplified"] == 1) & (df["kras_mutation"] == 0),
    "treatment_trastuzumab_tucatinib",
    "Trastuzumab/tucatinib in HER2+ & KRAS wt",
)
results["it16_her2_pos_krasmut"] = subgroup_ttest(
    (df["her2_amplified"] == 1) & (df["kras_mutation"] == 1),
    "treatment_trastuzumab_tucatinib",
    "Trastuzumab/tucatinib in HER2+ & KRAS mut",
)

# ===== Iteration 17: Systematic binary interaction screen =====
binary_feats = [
    "sex_female",
    "stage_iv",
    "right_sided_primary",
    "kras_mutation",
    "nras_mutation",
    "braf_v600e",
    "msi_high",
    "her2_amplified",
    "ntrk_fusion",
]
screen = []
for t in treatments:
    for b in binary_feats:
        if df[b].sum() < 50:
            continue
        sub = df.copy()
        sub["t"] = sub[t]
        sub["m"] = sub[b]
        sub["tm"] = sub["t"] * sub["m"]
        m = sm.OLS(
            sub["pfs_months"], sm.add_constant(sub[["t", "m", "tm"]])
        ).fit()
        screen.append(
            {
                "treatment": t,
                "modifier": b,
                "interaction_coef": fmt(m.params["tm"]),
                "interaction_p": fmt(m.pvalues["tm"], 8),
                "main_t_coef": fmt(m.params["t"]),
                "main_t_p": fmt(m.pvalues["t"], 8),
            }
        )
results["it17_screen_binary"] = sorted(
    screen, key=lambda r: r["interaction_p"] if r["interaction_p"] is not None else 1
)

# ===== Iteration 18: Continuous modifier screen =====
cont_feats = [
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
cont_screen = []
for t in treatments:
    for f in cont_feats:
        sub = df.copy()
        sub["t"] = sub[t]
        sub["f"] = (sub[f] - sub[f].mean()) / sub[f].std()
        sub["tf"] = sub["t"] * sub["f"]
        m = sm.OLS(
            sub["pfs_months"], sm.add_constant(sub[["t", "f", "tf"]])
        ).fit()
        cont_screen.append(
            {
                "treatment": t,
                "modifier": f,
                "interaction_coef": fmt(m.params["tf"]),
                "interaction_p": fmt(m.pvalues["tf"], 8),
            }
        )
results["it18_screen_continuous"] = sorted(
    cont_screen,
    key=lambda r: r["interaction_p"] if r["interaction_p"] is not None else 1,
)

# ===== Iteration 19: Dichotomize top continuous interactions =====
top_cont = [r for r in results["it18_screen_continuous"] if r["interaction_p"] is not None][:15]
dicho = []
for r in top_cont:
    t = r["treatment"]
    f = r["modifier"]
    med = df[f].median()
    high_mask = df[f] >= med
    low_mask = ~high_mask
    res_high = subgroup_ttest(high_mask, t, f"{t} when {f} >= median ({med:.3g})")
    res_low = subgroup_ttest(low_mask, t, f"{t} when {f} < median ({med:.3g})")
    dicho.append({"high": res_high, "low": res_low})
results["it19_dichotomize_top"] = dicho

# ===== Iteration 20: Regorafenib subgroup search =====
rego = {}
for f in binary_feats:
    rego[f"pos_{f}"] = subgroup_ttest(df[f] == 1, "treatment_regorafenib", f"Rego in {f}=1")
    rego[f"neg_{f}"] = subgroup_ttest(df[f] == 0, "treatment_regorafenib", f"Rego in {f}=0")
results["it20_rego_subgroups"] = rego

# ===== Iteration 21: Bevacizumab subgroup search =====
bev = {}
for f in binary_feats:
    bev[f"pos_{f}"] = subgroup_ttest(df[f] == 1, "treatment_bevacizumab", f"Bev in {f}=1")
    bev[f"neg_{f}"] = subgroup_ttest(df[f] == 0, "treatment_bevacizumab", f"Bev in {f}=0")
results["it21_bev_subgroups"] = bev

# ===== Iteration 22: Multivariable with key interactions =====
formula = (
    "pfs_months ~ age_years + sex_female + ecog_ps + stage_iv + right_sided_primary + "
    "kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified + ntrk_fusion + "
    "cea_ng_ml + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr + "
    "hemoglobin_g_dl + alkaline_phosphatase_u_l + ast_u_l + alt_u_l + total_bilirubin_mg_dl + "
    "creatinine_mg_dl + bun_mg_dl + sodium_meq_l + potassium_meq_l + calcium_mg_dl + "
    "treatment_cetuximab*kras_mutation + treatment_cetuximab*nras_mutation + "
    "treatment_cetuximab*braf_v600e + "
    "treatment_pembrolizumab*msi_high + "
    "treatment_encorafenib*braf_v600e + "
    "treatment_trastuzumab_tucatinib*her2_amplified + "
    "treatment_bevacizumab + treatment_regorafenib"
)
m22 = smf.ols(formula, data=df).fit()
results["it22_full_interaction_model"] = {
    "r2": fmt(m22.rsquared),
    "n": int(m22.nobs),
    "params": {
        c: {"coef": fmt(m22.params[c]), "p": fmt(m22.pvalues[c], 6)}
        for c in m22.params.index
        if (
            "treatment" in c
            or ":" in c
            or c
            in [
                "kras_mutation",
                "msi_high",
                "braf_v600e",
                "her2_amplified",
                "right_sided_primary",
            ]
        )
    },
}

# ===== Iteration 23: Pembrolizumab refined =====
results["it23_pembro_msi_byecog"] = {}
for ecog in [0, 1, 2]:
    pred = (df["msi_high"] == 1) & (df["ecog_ps"] == ecog)
    results["it23_pembro_msi_byecog"][f"ecog{ecog}"] = subgroup_ttest(
        pred, "treatment_pembrolizumab", f"Pembrolizumab in MSI-H + ECOG={ecog}"
    )
for stg in [0, 1]:
    pred = (df["msi_high"] == 1) & (df["stage_iv"] == stg)
    results[f"it23_pembro_msi_stage{stg}"] = subgroup_ttest(
        pred, "treatment_pembrolizumab", f"Pembrolizumab in MSI-H + stage_iv={stg}"
    )

# ===== Iteration 24: Cetuximab in panwt by side and ECOG =====
results["it24_cetux_panwt_ecog"] = {}
for ecog in [0, 1, 2]:
    pred = pred_panwt & (df["ecog_ps"] == ecog)
    results["it24_cetux_panwt_ecog"][f"ecog{ecog}"] = subgroup_ttest(
        pred, "treatment_cetuximab", f"Cetuximab in panwt + ECOG={ecog}"
    )
for ecog in [0, 1, 2]:
    pred = pred_panwt & (df["right_sided_primary"] == 0) & (df["ecog_ps"] == ecog)
    results[f"it24_cetux_panwt_left_ecog{ecog}"] = subgroup_ttest(
        pred, "treatment_cetuximab", f"Cetuximab in panwt + LEFT + ECOG={ecog}"
    )

# ===== Iteration 25: Final integrated subgroups =====
results["it25_pembro_msi_sex"] = {}
for s in [0, 1]:
    pred = (df["msi_high"] == 1) & (df["sex_female"] == s)
    results["it25_pembro_msi_sex"][f"sex_female{s}"] = subgroup_ttest(
        pred, "treatment_pembrolizumab", f"Pembrolizumab in MSI-H + sex_female={s}"
    )

results["it25_encora_braf_side"] = {}
for side in [0, 1]:
    pred = (df["braf_v600e"] == 1) & (df["right_sided_primary"] == side)
    results["it25_encora_braf_side"][f"right{side}"] = subgroup_ttest(
        pred, "treatment_encorafenib", f"Encorafenib in BRAF V600E + right={side}"
    )

pred_her2_wt = (
    (df["her2_amplified"] == 1)
    & (df["kras_mutation"] == 0)
    & (df["nras_mutation"] == 0)
    & (df["braf_v600e"] == 0)
)
results["it25_her2_panwt"] = subgroup_ttest(
    pred_her2_wt,
    "treatment_trastuzumab_tucatinib",
    "Trastuzumab/tucatinib in HER2+ & KRAS/NRAS/BRAF all wt",
)

pred_encora_eligible = (
    (df["braf_v600e"] == 1) & (df["kras_mutation"] == 0) & (df["nras_mutation"] == 0)
)
results["it25_encora_panwt"] = subgroup_ttest(
    pred_encora_eligible,
    "treatment_encorafenib",
    "Encorafenib in BRAF V600E + KRAS/NRAS wt",
)

results["it25_cetux_panwt_left"] = subgroup_ttest(
    pred_panwt & (df["right_sided_primary"] == 0),
    "treatment_cetuximab",
    "Cetuximab in KRAS/NRAS/BRAF wt + LEFT-sided",
)

# Save
with open("results_main.json", "w") as fh:
    json.dump(results, fh, indent=2, default=str)

print("Done. Results saved.")
print("\nTreatment main effects (mean PFS treated - untreated):")
for k, v in results["it1_treatment_main"].items():
    print(f"  {k}: diff={v['diff']}, p={v['p_value']}")
print("\nTop 10 binary interactions (by p):")
for r in results["it17_screen_binary"][:10]:
    print(
        f"  {r['treatment']} x {r['modifier']}: coef={r['interaction_coef']}, p={r['interaction_p']}"
    )
print("\nTop 10 continuous interactions:")
for r in results["it18_screen_continuous"][:10]:
    print(
        f"  {r['treatment']} x {r['modifier']}: coef={r['interaction_coef']}, p={r['interaction_p']}"
    )
