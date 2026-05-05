"""Fresh structured analysis for ds001_crc — iterative hypothesis testing on pfs_months."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
RESULTS = {}


def record(key, **kwargs):
    # Coerce numpy types to native Python so JSON round-trips cleanly.
    for k, v in list(kwargs.items()):
        if isinstance(v, (np.bool_, np.integer)):
            kwargs[k] = bool(v) if isinstance(v, np.bool_) else int(v)
        elif isinstance(v, np.floating):
            kwargs[k] = float(v)
    RESULTS[key] = kwargs
    # Print short summary
    eff = kwargs.get("effect_estimate")
    p = kwargs.get("p_value")
    sig = kwargs.get("significant")
    sm_str = kwargs.get("summary", "")
    print(f"[{key}] eff={eff} p={p} sig={sig} :: {sm_str[:160]}")


def lin_reg(y, X, label):
    X1 = sm.add_constant(X)
    m = sm.OLS(y, X1).fit()
    coefs = m.params.to_dict()
    pvs = m.pvalues.to_dict()
    return m, coefs, pvs


def t_test(group_a, group_b, label):
    a = df.loc[group_a, "pfs_months"].values
    b = df.loc[group_b, "pfs_months"].values
    if len(a) < 5 or len(b) < 5:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(a.mean() - b.mean()),
        "p": float(p),
    }


# ============================================================
# Iteration 1: Univariate main effects of treatments on PFS
# ============================================================
print("\n=== Iteration 1: Treatment main effects ===")
for t in [
    "treatment_cetuximab",
    "treatment_bevacizumab",
    "treatment_pembrolizumab",
    "treatment_encorafenib",
    "treatment_trastuzumab_tucatinib",
    "treatment_regorafenib",
]:
    r = t_test(df[t] == 1, df[t] == 0, t)
    record(
        f"i1_{t}_main",
        effect_estimate=r["diff"],
        p_value=r["p"],
        significant=r["p"] < 0.05,
        summary=f"PFS on {t} mean={r['mean_a']:.2f} vs off mean={r['mean_b']:.2f} (n_on={r['n_a']}, n_off={r['n_b']})",
    )

# ============================================================
# Iteration 2: Biomarker / clinical feature main effects on PFS
# ============================================================
print("\n=== Iteration 2: Feature main effects ===")
for b in [
    "kras_mutation",
    "nras_mutation",
    "braf_v600e",
    "msi_high",
    "her2_amplified",
    "ntrk_fusion",
    "stage_iv",
    "right_sided_primary",
    "sex_female",
]:
    r = t_test(df[b] == 1, df[b] == 0, b)
    record(
        f"i2_{b}_main",
        effect_estimate=r["diff"],
        p_value=r["p"],
        significant=r["p"] < 0.05,
        summary=f"PFS with {b}=1 mean={r['mean_a']:.2f} vs =0 mean={r['mean_b']:.2f} (n_1={r['n_a']}, n_0={r['n_b']})",
    )

# ECOG by category
for ec in [0, 1, 2]:
    r = t_test(df["ecog_ps"] == ec, df["ecog_ps"] != ec, f"ecog_{ec}")
    record(
        f"i2_ecog_{ec}_main",
        effect_estimate=r["diff"],
        p_value=r["p"],
        significant=r["p"] < 0.05,
        summary=f"PFS with ECOG={ec} mean={r['mean_a']:.2f} vs not={ec} mean={r['mean_b']:.2f}",
    )

# Continuous predictors via linear regression
print("\n  -- continuous predictors --")
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
    m, coefs, pvs = lin_reg(df["pfs_months"], df[[c]], c)
    record(
        f"i2_{c}_main",
        effect_estimate=float(coefs[c]),
        p_value=float(pvs[c]),
        significant=pvs[c] < 0.05,
        summary=f"OLS pfs_months ~ {c}: beta={coefs[c]:.4f}",
    )


# ============================================================
# Iteration 3: Multivariable regression of clinical features
# ============================================================
print("\n=== Iteration 3: Multivariable clinical model ===")
preds = [
    "age_years",
    "sex_female",
    "ecog_ps",
    "stage_iv",
    "right_sided_primary",
    "albumin_g_dl",
    "ldh_u_l",
    "cea_ng_ml",
    "weight_loss_pct_6mo",
    "crp_mg_l",
    "nlr",
    "hemoglobin_g_dl",
]
m, coefs, pvs = lin_reg(df["pfs_months"], df[preds], "multi")
for p in preds:
    record(
        f"i3_multi_{p}",
        effect_estimate=float(coefs[p]),
        p_value=float(pvs[p]),
        significant=pvs[p] < 0.05,
        summary=f"Multivariable beta {p}={coefs[p]:.4f}",
    )

# ============================================================
# Iteration 4: Pembrolizumab x MSI-high interaction
# ============================================================
print("\n=== Iteration 4: Pembro x MSI ===")
sub_msi = df[df["msi_high"] == 1]
sub_nomsi = df[df["msi_high"] == 0]
for sub, lbl in [(sub_msi, "msi1"), (sub_nomsi, "msi0")]:
    on = sub.loc[sub["treatment_pembrolizumab"] == 1, "pfs_months"]
    off = sub.loc[sub["treatment_pembrolizumab"] == 0, "pfs_months"]
    if len(on) > 5 and len(off) > 5:
        t, p = stats.ttest_ind(on, off, equal_var=False)
        record(
            f"i4_pembro_in_{lbl}",
            effect_estimate=float(on.mean() - off.mean()),
            p_value=float(p),
            significant=p < 0.05,
            summary=f"Pembro effect in {lbl}: on={on.mean():.2f} (n={len(on)}) off={off.mean():.2f} (n={len(off)})",
        )

# Interaction test
m = smf.ols(
    "pfs_months ~ treatment_pembrolizumab * msi_high + age_years + ecog_ps + stage_iv",
    data=df,
).fit()
ix = "treatment_pembrolizumab:msi_high"
record(
    "i4_pembro_msi_interaction",
    effect_estimate=float(m.params[ix]),
    p_value=float(m.pvalues[ix]),
    significant=float(m.pvalues[ix]) < 0.05,
    summary=f"Interaction pembro:msi_high coef={m.params[ix]:.3f}",
)

# ============================================================
# Iteration 5: Cetuximab x KRAS/NRAS/BRAF (RAS/RAF wild-type hypothesis)
# ============================================================
print("\n=== Iteration 5: Cetuximab x KRAS/NRAS/BRAF ===")
df["ras_raf_wt"] = ((df["kras_mutation"] == 0) & (df["nras_mutation"] == 0) & (df["braf_v600e"] == 0)).astype(int)
for sub_val, lbl in [(1, "ras_raf_wt"), (0, "ras_raf_mut")]:
    sub = df[df["ras_raf_wt"] == sub_val]
    on = sub.loc[sub["treatment_cetuximab"] == 1, "pfs_months"]
    off = sub.loc[sub["treatment_cetuximab"] == 0, "pfs_months"]
    t, p = stats.ttest_ind(on, off, equal_var=False)
    record(
        f"i5_cetux_in_{lbl}",
        effect_estimate=float(on.mean() - off.mean()),
        p_value=float(p),
        significant=p < 0.05,
        summary=f"Cetux effect in {lbl}: on={on.mean():.2f} (n={len(on)}) off={off.mean():.2f} (n={len(off)})",
    )

m = smf.ols(
    "pfs_months ~ treatment_cetuximab * ras_raf_wt + age_years + ecog_ps + stage_iv + right_sided_primary",
    data=df,
).fit()
ix = "treatment_cetuximab:ras_raf_wt"
record(
    "i5_cetux_rasrafwt_interaction",
    effect_estimate=float(m.params[ix]),
    p_value=float(m.pvalues[ix]),
    significant=float(m.pvalues[ix]) < 0.05,
    summary=f"Interaction cetux:ras_raf_wt coef={m.params[ix]:.3f}",
)

# Stratify by side
for side, lbl in [(0, "left"), (1, "right")]:
    sub = df[(df["right_sided_primary"] == side) & (df["ras_raf_wt"] == 1)]
    on = sub.loc[sub["treatment_cetuximab"] == 1, "pfs_months"]
    off = sub.loc[sub["treatment_cetuximab"] == 0, "pfs_months"]
    if len(on) > 5 and len(off) > 5:
        t, p = stats.ttest_ind(on, off, equal_var=False)
        record(
            f"i5_cetux_rasrafwt_{lbl}",
            effect_estimate=float(on.mean() - off.mean()),
            p_value=float(p),
            significant=p < 0.05,
            summary=f"Cetux in RAS/RAF-wt + {lbl}-sided: on={on.mean():.2f} (n={len(on)}) off={off.mean():.2f} (n={len(off)})",
        )

# ============================================================
# Iteration 6: Encorafenib x BRAF V600E interaction
# ============================================================
print("\n=== Iteration 6: Encorafenib x BRAF V600E ===")
for v, lbl in [(1, "braf1"), (0, "braf0")]:
    sub = df[df["braf_v600e"] == v]
    on = sub.loc[sub["treatment_encorafenib"] == 1, "pfs_months"]
    off = sub.loc[sub["treatment_encorafenib"] == 0, "pfs_months"]
    if len(on) > 5 and len(off) > 5:
        t, p = stats.ttest_ind(on, off, equal_var=False)
        record(
            f"i6_encora_in_{lbl}",
            effect_estimate=float(on.mean() - off.mean()),
            p_value=float(p),
            significant=p < 0.05,
            summary=f"Encora in {lbl}: on={on.mean():.2f} (n={len(on)}) off={off.mean():.2f} (n={len(off)})",
        )

m = smf.ols(
    "pfs_months ~ treatment_encorafenib * braf_v600e + age_years + ecog_ps + stage_iv",
    data=df,
).fit()
ix = "treatment_encorafenib:braf_v600e"
record(
    "i6_encora_braf_interaction",
    effect_estimate=float(m.params[ix]),
    p_value=float(m.pvalues[ix]),
    significant=float(m.pvalues[ix]) < 0.05,
    summary=f"Interaction encora:braf_v600e coef={m.params[ix]:.3f}",
)

# ============================================================
# Iteration 7: Trastuzumab+Tucatinib x HER2-amplified
# ============================================================
print("\n=== Iteration 7: T+T x HER2 ===")
for v, lbl in [(1, "her21"), (0, "her20")]:
    sub = df[df["her2_amplified"] == v]
    on = sub.loc[sub["treatment_trastuzumab_tucatinib"] == 1, "pfs_months"]
    off = sub.loc[sub["treatment_trastuzumab_tucatinib"] == 0, "pfs_months"]
    if len(on) > 5 and len(off) > 5:
        t, p = stats.ttest_ind(on, off, equal_var=False)
        record(
            f"i7_tt_in_{lbl}",
            effect_estimate=float(on.mean() - off.mean()),
            p_value=float(p),
            significant=p < 0.05,
            summary=f"T+T in {lbl}: on={on.mean():.2f} (n={len(on)}) off={off.mean():.2f} (n={len(off)})",
        )

m = smf.ols(
    "pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified + age_years + ecog_ps + stage_iv",
    data=df,
).fit()
ix = "treatment_trastuzumab_tucatinib:her2_amplified"
record(
    "i7_tt_her2_interaction",
    effect_estimate=float(m.params[ix]),
    p_value=float(m.pvalues[ix]),
    significant=float(m.pvalues[ix]) < 0.05,
    summary=f"Interaction T+T:her2 coef={m.params[ix]:.3f}",
)

# Within HER2: KRAS modifier?
print("\n  -- T+T x HER2 with KRAS status --")
for k, lbl in [(0, "kraswt"), (1, "krasmut")]:
    sub = df[(df["her2_amplified"] == 1) & (df["kras_mutation"] == k)]
    on = sub.loc[sub["treatment_trastuzumab_tucatinib"] == 1, "pfs_months"]
    off = sub.loc[sub["treatment_trastuzumab_tucatinib"] == 0, "pfs_months"]
    if len(on) > 5 and len(off) > 5:
        t, p = stats.ttest_ind(on, off, equal_var=False)
        record(
            f"i7_tt_her2_{lbl}",
            effect_estimate=float(on.mean() - off.mean()),
            p_value=float(p),
            significant=p < 0.05,
            summary=f"T+T in HER2+ & {lbl}: on={on.mean():.2f} (n={len(on)}) off={off.mean():.2f} (n={len(off)})",
        )

# ============================================================
# Iteration 8: Bevacizumab broad effect with adjustment
# ============================================================
print("\n=== Iteration 8: Bevacizumab adjusted ===")
m = smf.ols(
    "pfs_months ~ treatment_bevacizumab + age_years + ecog_ps + stage_iv + right_sided_primary + albumin_g_dl + ldh_u_l + cea_ng_ml",
    data=df,
).fit()
record(
    "i8_bev_adjusted",
    effect_estimate=float(m.params["treatment_bevacizumab"]),
    p_value=float(m.pvalues["treatment_bevacizumab"]),
    significant=float(m.pvalues["treatment_bevacizumab"]) < 0.05,
    summary=f"Bev adjusted beta={m.params['treatment_bevacizumab']:.3f}",
)

# Bev x interactions
print("\n  -- bev interactions --")
for mod in ["kras_mutation", "right_sided_primary", "msi_high", "stage_iv", "ecog_ps", "braf_v600e"]:
    f = f"pfs_months ~ treatment_bevacizumab * {mod} + age_years + ecog_ps + stage_iv"
    if mod == "ecog_ps" or mod == "stage_iv":
        f = f"pfs_months ~ treatment_bevacizumab * {mod} + age_years"
    m = smf.ols(f, data=df).fit()
    ix = f"treatment_bevacizumab:{mod}"
    record(
        f"i8_bev_x_{mod}",
        effect_estimate=float(m.params[ix]),
        p_value=float(m.pvalues[ix]),
        significant=float(m.pvalues[ix]) < 0.05,
        summary=f"Interaction bev:{mod} coef={m.params[ix]:.3f}",
    )

# ============================================================
# Iteration 9: Regorafenib effect screen
# ============================================================
print("\n=== Iteration 9: Regorafenib ===")
m = smf.ols(
    "pfs_months ~ treatment_regorafenib + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + cea_ng_ml",
    data=df,
).fit()
record(
    "i9_rego_adjusted",
    effect_estimate=float(m.params["treatment_regorafenib"]),
    p_value=float(m.pvalues["treatment_regorafenib"]),
    significant=float(m.pvalues["treatment_regorafenib"]) < 0.05,
    summary=f"Rego adjusted beta={m.params['treatment_regorafenib']:.3f}",
)

# Rego heterogeneity screen — interactions
print("\n  -- rego x feature --")
for mod in [
    "kras_mutation",
    "nras_mutation",
    "braf_v600e",
    "msi_high",
    "her2_amplified",
    "ntrk_fusion",
    "stage_iv",
    "right_sided_primary",
    "sex_female",
    "ecog_ps",
]:
    f = f"pfs_months ~ treatment_regorafenib * {mod} + age_years + ecog_ps + stage_iv"
    if mod in ("ecog_ps", "stage_iv"):
        f = f"pfs_months ~ treatment_regorafenib * {mod} + age_years"
    m = smf.ols(f, data=df).fit()
    ix = f"treatment_regorafenib:{mod}"
    record(
        f"i9_rego_x_{mod}",
        effect_estimate=float(m.params[ix]),
        p_value=float(m.pvalues[ix]),
        significant=float(m.pvalues[ix]) < 0.05,
        summary=f"Interaction rego:{mod} coef={m.params[ix]:.3f}",
    )

# ============================================================
# Save
# ============================================================
with open("new_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print(f"\nSaved {len(RESULTS)} results.")
