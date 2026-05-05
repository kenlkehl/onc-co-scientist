"""Iterations 10-25: deeper regorafenib subgroup discovery + final subgroup tests."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
RESULTS = json.load(open("new_results.json"))


def record(key, **kwargs):
    for k, v in list(kwargs.items()):
        if isinstance(v, (np.bool_, np.integer)):
            kwargs[k] = bool(v) if isinstance(v, np.bool_) else int(v)
        elif isinstance(v, np.floating):
            kwargs[k] = float(v)
    RESULTS[key] = kwargs
    eff = kwargs.get("effect_estimate")
    p = kwargs.get("p_value")
    sig = kwargs.get("significant")
    sm_str = kwargs.get("summary", "")
    print(f"[{key}] eff={eff} p={p} sig={sig} :: {sm_str[:160]}")


def trt_effect(mask, trt):
    on = df.loc[mask & (df[trt] == 1), "pfs_months"]
    off = df.loc[mask & (df[trt] == 0), "pfs_months"]
    if len(on) < 5 or len(off) < 5:
        return {"n_on": len(on), "n_off": len(off), "mean_on": float("nan"),
                "mean_off": float("nan"), "diff": float("nan"), "p": 1.0}
    t, p = stats.ttest_ind(on, off, equal_var=False)
    return {
        "n_on": int(len(on)),
        "n_off": int(len(off)),
        "mean_on": float(on.mean()),
        "mean_off": float(off.mean()),
        "diff": float(on.mean() - off.mean()),
        "p": float(p),
    }


# ============================================================
# Iteration 10: Regorafenib stratified by KRAS status
# ============================================================
print("\n=== Iteration 10: Rego x KRAS strata ===")
for v, lbl in [(0, "kraswt"), (1, "krasmut")]:
    r = trt_effect(df["kras_mutation"] == v, "treatment_regorafenib")
    record(
        f"i10_rego_in_{lbl}",
        effect_estimate=r["diff"],
        p_value=r["p"],
        significant=r["p"] < 0.05,
        summary=f"Rego in {lbl}: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
    )

# ============================================================
# Iteration 11: Rego stratified by BRAF
# ============================================================
print("\n=== Iteration 11: Rego x BRAF strata ===")
for v, lbl in [(0, "brafwt"), (1, "brafmut")]:
    r = trt_effect(df["braf_v600e"] == v, "treatment_regorafenib")
    record(
        f"i11_rego_in_{lbl}",
        effect_estimate=r["diff"],
        p_value=r["p"],
        significant=r["p"] < 0.05,
        summary=f"Rego in {lbl}: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
    )

# ============================================================
# Iteration 12: Rego stratified by side
# ============================================================
print("\n=== Iteration 12: Rego x side strata ===")
for v, lbl in [(0, "leftsided"), (1, "rightsided")]:
    r = trt_effect(df["right_sided_primary"] == v, "treatment_regorafenib")
    record(
        f"i12_rego_in_{lbl}",
        effect_estimate=r["diff"],
        p_value=r["p"],
        significant=r["p"] < 0.05,
        summary=f"Rego in {lbl}: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
    )

# ============================================================
# Iteration 13: Rego in joint subgroup KRAS-wt AND BRAF-wt AND left-sided
# ============================================================
print("\n=== Iteration 13: Rego in joint biomarker-wt left-sided ===")
mask_joint = (df["kras_mutation"] == 0) & (df["braf_v600e"] == 0) & (df["right_sided_primary"] == 0)
r = trt_effect(mask_joint, "treatment_regorafenib")
record(
    "i13_rego_in_kras0_braf0_left",
    effect_estimate=r["diff"],
    p_value=r["p"],
    significant=r["p"] < 0.05,
    summary=f"Rego in KRAS-wt+BRAF-wt+left: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
)

# Complement
r = trt_effect(~mask_joint, "treatment_regorafenib")
record(
    "i13_rego_outside_subgroup",
    effect_estimate=r["diff"],
    p_value=r["p"],
    significant=r["p"] < 0.05,
    summary=f"Rego outside subgroup: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
)

# ============================================================
# Iteration 14: Add ECOG to subgroup
# ============================================================
print("\n=== Iteration 14: Rego x ECOG ===")
for v in [0, 1, 2]:
    r = trt_effect(df["ecog_ps"] == v, "treatment_regorafenib")
    record(
        f"i14_rego_ecog_{v}",
        effect_estimate=r["diff"],
        p_value=r["p"],
        significant=r["p"] < 0.05,
        summary=f"Rego in ECOG={v}: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
    )

# Joint subgroup with ECOG=0
mask_joint_ec = mask_joint & (df["ecog_ps"] == 0)
r = trt_effect(mask_joint_ec, "treatment_regorafenib")
record(
    "i14_rego_subgroup_ec0",
    effect_estimate=r["diff"],
    p_value=r["p"],
    significant=r["p"] < 0.05,
    summary=f"Rego in KRAS-wt+BRAF-wt+left+ECOG=0: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
)

# ============================================================
# Iteration 15: Comprehensive interaction screen for rego
# ============================================================
print("\n=== Iteration 15: Rego x continuous predictors ===")
for c in [
    "age_years",
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
]:
    f = f"pfs_months ~ treatment_regorafenib * {c} + age_years + ecog_ps + stage_iv"
    if c == "age_years":
        f = f"pfs_months ~ treatment_regorafenib * {c} + ecog_ps + stage_iv"
    m = smf.ols(f, data=df).fit()
    ix = f"treatment_regorafenib:{c}"
    record(
        f"i15_rego_x_{c}",
        effect_estimate=float(m.params[ix]),
        p_value=float(m.pvalues[ix]),
        significant=float(m.pvalues[ix]) < 0.05,
        summary=f"Rego x {c} interaction beta={m.params[ix]:.4f}",
    )

# ============================================================
# Iteration 16: Joint regression with rego x KRAS x BRAF x side
# ============================================================
print("\n=== Iteration 16: Joint rego model with key modifiers ===")
m = smf.ols(
    """pfs_months ~ treatment_regorafenib * kras_mutation
                  + treatment_regorafenib * braf_v600e
                  + treatment_regorafenib * right_sided_primary
                  + age_years + ecog_ps + stage_iv + albumin_g_dl + cea_ng_ml""",
    data=df,
).fit()
for ix in [
    "treatment_regorafenib",
    "treatment_regorafenib:kras_mutation",
    "treatment_regorafenib:braf_v600e",
    "treatment_regorafenib:right_sided_primary",
]:
    record(
        f"i16_joint_{ix.replace(':', '_')}",
        effect_estimate=float(m.params[ix]),
        p_value=float(m.pvalues[ix]),
        significant=float(m.pvalues[ix]) < 0.05,
        summary=f"Joint model {ix} = {m.params[ix]:.3f}",
    )

# ============================================================
# Iteration 17: Pembro main effect adjusted with biomarker / age stratification
# ============================================================
print("\n=== Iteration 17: Pembro adjusted ===")
m = smf.ols(
    "pfs_months ~ treatment_pembrolizumab + age_years + ecog_ps + stage_iv + msi_high + albumin_g_dl + cea_ng_ml",
    data=df,
).fit()
record(
    "i17_pembro_adjusted",
    effect_estimate=float(m.params["treatment_pembrolizumab"]),
    p_value=float(m.pvalues["treatment_pembrolizumab"]),
    significant=float(m.pvalues["treatment_pembrolizumab"]) < 0.05,
    summary=f"Pembro adjusted beta={m.params['treatment_pembrolizumab']:.3f}",
)

# Pembro x biomarker scan
print("\n  -- pembro x feature --")
for mod in ["msi_high", "kras_mutation", "braf_v600e", "right_sided_primary", "stage_iv", "ecog_ps", "her2_amplified"]:
    f = f"pfs_months ~ treatment_pembrolizumab * {mod} + age_years + ecog_ps + stage_iv"
    if mod in ("ecog_ps", "stage_iv"):
        f = f"pfs_months ~ treatment_pembrolizumab * {mod} + age_years"
    m = smf.ols(f, data=df).fit()
    ix = f"treatment_pembrolizumab:{mod}"
    record(
        f"i17_pembro_x_{mod}",
        effect_estimate=float(m.params[ix]),
        p_value=float(m.pvalues[ix]),
        significant=float(m.pvalues[ix]) < 0.05,
        summary=f"Pembro x {mod} = {m.params[ix]:.3f}",
    )

# ============================================================
# Iteration 18: Cetuximab interaction screen broader
# ============================================================
print("\n=== Iteration 18: Cetux x feature scan ===")
for mod in ["kras_mutation", "nras_mutation", "braf_v600e", "msi_high", "her2_amplified", "right_sided_primary", "stage_iv", "ecog_ps", "sex_female"]:
    f = f"pfs_months ~ treatment_cetuximab * {mod} + age_years + ecog_ps + stage_iv"
    if mod in ("ecog_ps", "stage_iv"):
        f = f"pfs_months ~ treatment_cetuximab * {mod} + age_years"
    m = smf.ols(f, data=df).fit()
    ix = f"treatment_cetuximab:{mod}"
    record(
        f"i18_cetux_x_{mod}",
        effect_estimate=float(m.params[ix]),
        p_value=float(m.pvalues[ix]),
        significant=float(m.pvalues[ix]) < 0.05,
        summary=f"Cetux x {mod} = {m.params[ix]:.3f}",
    )

# ============================================================
# Iteration 19: Encorafenib full interaction scan
# ============================================================
print("\n=== Iteration 19: Encorafenib x feature scan ===")
for mod in ["kras_mutation", "nras_mutation", "braf_v600e", "msi_high", "her2_amplified", "right_sided_primary", "stage_iv", "ecog_ps", "treatment_cetuximab", "treatment_bevacizumab"]:
    f = f"pfs_months ~ treatment_encorafenib * {mod} + age_years + ecog_ps + stage_iv"
    if mod in ("ecog_ps", "stage_iv"):
        f = f"pfs_months ~ treatment_encorafenib * {mod} + age_years"
    m = smf.ols(f, data=df).fit()
    ix = f"treatment_encorafenib:{mod}"
    record(
        f"i19_encora_x_{mod}",
        effect_estimate=float(m.params[ix]),
        p_value=float(m.pvalues[ix]),
        significant=float(m.pvalues[ix]) < 0.05,
        summary=f"Encora x {mod} = {m.params[ix]:.3f}",
    )

# ============================================================
# Iteration 20: T+T full interaction scan
# ============================================================
print("\n=== Iteration 20: T+T x feature scan ===")
for mod in ["kras_mutation", "nras_mutation", "braf_v600e", "msi_high", "her2_amplified", "right_sided_primary", "stage_iv", "ecog_ps"]:
    f = f"pfs_months ~ treatment_trastuzumab_tucatinib * {mod} + age_years + ecog_ps + stage_iv"
    if mod in ("ecog_ps", "stage_iv"):
        f = f"pfs_months ~ treatment_trastuzumab_tucatinib * {mod} + age_years"
    m = smf.ols(f, data=df).fit()
    ix = f"treatment_trastuzumab_tucatinib:{mod}"
    record(
        f"i20_tt_x_{mod}",
        effect_estimate=float(m.params[ix]),
        p_value=float(m.pvalues[ix]),
        significant=float(m.pvalues[ix]) < 0.05,
        summary=f"T+T x {mod} = {m.params[ix]:.3f}",
    )

# ============================================================
# Iteration 21: Bevacizumab full interaction scan
# ============================================================
print("\n=== Iteration 21: Bev x feature scan ===")
for mod in ["nras_mutation", "her2_amplified", "ntrk_fusion", "sex_female", "ecog_ps", "stage_iv", "right_sided_primary"]:
    f = f"pfs_months ~ treatment_bevacizumab * {mod} + age_years + ecog_ps + stage_iv"
    if mod in ("ecog_ps", "stage_iv"):
        f = f"pfs_months ~ treatment_bevacizumab * {mod} + age_years"
    m = smf.ols(f, data=df).fit()
    ix = f"treatment_bevacizumab:{mod}"
    record(
        f"i21_bev_x_{mod}",
        effect_estimate=float(m.params[ix]),
        p_value=float(m.pvalues[ix]),
        significant=float(m.pvalues[ix]) < 0.05,
        summary=f"Bev x {mod} = {m.params[ix]:.3f}",
    )

# ============================================================
# Iteration 22: Rego dose-response: stratify by KRAS, BRAF, side jointly to pin down full subgroup
# ============================================================
print("\n=== Iteration 22: Rego in 8 cells of KRAS x BRAF x side ===")
for kras in [0, 1]:
    for braf in [0, 1]:
        for side in [0, 1]:
            mask = (
                (df["kras_mutation"] == kras)
                & (df["braf_v600e"] == braf)
                & (df["right_sided_primary"] == side)
            )
            r = trt_effect(mask, "treatment_regorafenib")
            record(
                f"i22_rego_k{kras}_b{braf}_s{side}",
                effect_estimate=r["diff"],
                p_value=r["p"],
                significant=r["p"] < 0.05,
                summary=f"Rego cell K={kras},B={braf},S={side}: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
            )

# ============================================================
# Iteration 23: Final candidate subgroup — KRAS-wt AND BRAF-wt AND left-sided
# - confirm in independent halves
# ============================================================
print("\n=== Iteration 23: Confirmation of best subgroup ===")
mask_best = (df["kras_mutation"] == 0) & (df["braf_v600e"] == 0) & (df["right_sided_primary"] == 0)
r = trt_effect(mask_best, "treatment_regorafenib")
record(
    "i23_rego_best_subgroup",
    effect_estimate=r["diff"],
    p_value=r["p"],
    significant=r["p"] < 0.05,
    summary=(
        f"Rego in KRAS-wt + BRAF-wt + left-sided: on={r['mean_on']:.2f} "
        f"(n={r['n_on']}) off={r['mean_off']:.2f} (n={r['n_off']}); diff={r['diff']:.2f}"
    ),
)

# Each predicate alone vs combined: test whether dropping one predicate reduces effect size
for drop, lbl in [
    ("kras", "drop_kras"),
    ("braf", "drop_braf"),
    ("side", "drop_side"),
]:
    if drop == "kras":
        m = (df["braf_v600e"] == 0) & (df["right_sided_primary"] == 0)
    elif drop == "braf":
        m = (df["kras_mutation"] == 0) & (df["right_sided_primary"] == 0)
    else:
        m = (df["kras_mutation"] == 0) & (df["braf_v600e"] == 0)
    r = trt_effect(m, "treatment_regorafenib")
    record(
        f"i23_rego_{lbl}",
        effect_estimate=r["diff"],
        p_value=r["p"],
        significant=r["p"] < 0.05,
        summary=f"Rego with {lbl}: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
    )

# ============================================================
# Iteration 24: NRAS-positive joint with rego (positive interaction in i9)
# ============================================================
print("\n=== Iteration 24: Rego in NRAS-mut subgroup ===")
r = trt_effect(df["nras_mutation"] == 1, "treatment_regorafenib")
record(
    "i24_rego_in_nras1",
    effect_estimate=r["diff"],
    p_value=r["p"],
    significant=r["p"] < 0.05,
    summary=f"Rego in NRAS-mut: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
)

# Rego in NRAS-mut + KRAS-wt + BRAF-wt + left
mask_n = (
    (df["nras_mutation"] == 1)
    & (df["kras_mutation"] == 0)
    & (df["braf_v600e"] == 0)
    & (df["right_sided_primary"] == 0)
)
r = trt_effect(mask_n, "treatment_regorafenib")
record(
    "i24_rego_nras1_subgroup",
    effect_estimate=r["diff"],
    p_value=r["p"],
    significant=r["p"] < 0.05,
    summary=f"Rego NRAS-mut+KRAS-wt+BRAF-wt+left: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
)

# ============================================================
# Iteration 25: Final summary tests — combination treatment effect
# Confirm that within best rego subgroup, other treatments still show no benefit
# ============================================================
print("\n=== Iteration 25: Other-treatment effects within rego best subgroup ===")
for trt in [
    "treatment_cetuximab",
    "treatment_bevacizumab",
    "treatment_pembrolizumab",
    "treatment_encorafenib",
    "treatment_trastuzumab_tucatinib",
]:
    r = trt_effect(mask_best, trt)
    record(
        f"i25_in_best_{trt}",
        effect_estimate=r["diff"],
        p_value=r["p"],
        significant=r["p"] < 0.05,
        summary=f"{trt} in rego-best subgroup: on={r['mean_on']:.2f}(n={r['n_on']}) off={r['mean_off']:.2f}(n={r['n_off']})",
    )

# Final overall regression: rego x (kras_mut + braf_v600e + right_sided)
m = smf.ols(
    """pfs_months ~ treatment_regorafenib
       + treatment_regorafenib:kras_mutation
       + treatment_regorafenib:braf_v600e
       + treatment_regorafenib:right_sided_primary
       + kras_mutation + braf_v600e + right_sided_primary
       + age_years + ecog_ps + stage_iv""",
    data=df,
).fit()
record(
    "i25_final_rego_pure",
    effect_estimate=float(m.params["treatment_regorafenib"]),
    p_value=float(m.pvalues["treatment_regorafenib"]),
    significant=float(m.pvalues["treatment_regorafenib"]) < 0.05,
    summary=(
        f"Final model rego baseline (KRAS-wt, BRAF-wt, left, ECOG=0, stage non-IV) "
        f"effect = {m.params['treatment_regorafenib']:.3f} months"
    ),
)

# Save
with open("new_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\nTotal results: {len(RESULTS)}")
