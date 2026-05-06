"""Comprehensive ds001_crc analysis.

Runs main-effect, interaction, and subgroup analyses. Emits a JSON dictionary
keyed by analysis_id which is later assembled into transcript.json.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

df = pd.read_parquet("dataset.parquet")
N = len(df)

results = {}


def t_test(group1, group2, name):
    g1 = group1.dropna().values
    g2 = group2.dropna().values
    if len(g1) < 2 or len(g2) < 2:
        return None
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    return {
        "n_treated": int(len(g1)),
        "n_control": int(len(g2)),
        "mean_treated": float(np.mean(g1)),
        "mean_control": float(np.mean(g2)),
        "diff": float(np.mean(g1) - np.mean(g2)),
        "p_value": float(p),
        "significant": bool(p < 0.05),
        "result": (
            f"{name}: mean PFS {np.mean(g1):.3f} vs {np.mean(g2):.3f} "
            f"(diff {np.mean(g1)-np.mean(g2):+.3f} mo, t-test p={p:.2e}, "
            f"n={len(g1)}/{len(g2)})"
        ),
    }


def store(key, val):
    results[key] = val
    if val is not None:
        print(f"[{key}] {val.get('result', val)}")


# =====================================================================
# Iteration 1: Demographics & disease severity main effects on PFS
# =====================================================================
print("\n=== Iteration 1: demographics & disease severity ===")

# Pearson correlation: age vs PFS
r, p = stats.pearsonr(df["age_years"], df["pfs_months"])
store("age_pfs", {"effect": float(r), "p_value": float(p),
                  "significant": bool(p < 0.05),
                  "result": f"Pearson r(age, PFS)={r:.4f}, p={p:.2e}"})

store("sex_pfs", t_test(df.loc[df.sex_female == 1, "pfs_months"],
                        df.loc[df.sex_female == 0, "pfs_months"],
                        "PFS female vs male"))

# ECOG: ANOVA across 0/1/2
groups = [df.loc[df.ecog_ps == k, "pfs_months"].values for k in [0, 1, 2]]
f, p = stats.f_oneway(*groups)
means = [float(g.mean()) for g in groups]
store("ecog_pfs", {"effect": float(means[0] - means[2]),
                   "p_value": float(p), "significant": bool(p < 0.05),
                   "result": f"PFS by ECOG 0/1/2: {means[0]:.3f}/{means[1]:.3f}/{means[2]:.3f} (ANOVA p={p:.2e})"})

store("stage_iv_pfs", t_test(df.loc[df.stage_iv == 1, "pfs_months"],
                             df.loc[df.stage_iv == 0, "pfs_months"],
                             "PFS stage IV vs not"))

# =====================================================================
# Iteration 2: Tumor biology main effects
# =====================================================================
print("\n=== Iteration 2: tumor biology main effects ===")
for bm in ["right_sided_primary", "kras_mutation", "nras_mutation",
           "braf_v600e", "msi_high", "her2_amplified", "ntrk_fusion"]:
    store(f"{bm}_pfs",
          t_test(df.loc[df[bm] == 1, "pfs_months"],
                 df.loc[df[bm] == 0, "pfs_months"],
                 f"PFS {bm}+ vs -"))

# =====================================================================
# Iteration 3: Lab values main effects on PFS (continuous)
# =====================================================================
print("\n=== Iteration 3: lab continuous correlations ===")
for lab in ["cea_ng_ml", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo",
            "crp_mg_l", "nlr", "hemoglobin_g_dl", "alkaline_phosphatase_u_l",
            "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl", "creatinine_mg_dl",
            "bun_mg_dl", "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl"]:
    r, p = stats.spearmanr(df[lab], df["pfs_months"])
    store(f"{lab}_pfs",
          {"effect": float(r), "p_value": float(p),
           "significant": bool(p < 0.05),
           "result": f"Spearman r({lab}, PFS)={r:.4f}, p={p:.2e}"})

# =====================================================================
# Iteration 4: Treatment main effects (raw, ignoring biomarker)
# =====================================================================
print("\n=== Iteration 4: treatment main effects ===")
trts = ["treatment_cetuximab", "treatment_bevacizumab",
        "treatment_pembrolizumab", "treatment_encorafenib",
        "treatment_trastuzumab_tucatinib", "treatment_regorafenib"]
for t in trts:
    store(f"{t}_main",
          t_test(df.loc[df[t] == 1, "pfs_months"],
                 df.loc[df[t] == 0, "pfs_months"],
                 f"PFS {t} on vs off"))

# =====================================================================
# Iteration 5: Cetuximab × RAS/RAF/right-sided
# =====================================================================
print("\n=== Iteration 5: cetuximab interactions ===")
# Cetuximab in RAS/BRAF wildtype
for predicate, name in [
    ((df.kras_mutation == 0) & (df.nras_mutation == 0) & (df.braf_v600e == 0),
     "RAS/BRAF wildtype"),
    ((df.kras_mutation == 0), "KRAS wildtype"),
    ((df.nras_mutation == 0), "NRAS wildtype"),
    ((df.braf_v600e == 0), "BRAF wildtype"),
    ((df.right_sided_primary == 0), "left-sided"),
    ((df.right_sided_primary == 1), "right-sided"),
    ((df.kras_mutation == 1), "KRAS mutant"),
    ((df.nras_mutation == 1), "NRAS mutant"),
    ((df.braf_v600e == 1), "BRAF V600E mutant"),
]:
    sub = df.loc[predicate]
    res = t_test(sub.loc[sub.treatment_cetuximab == 1, "pfs_months"],
                 sub.loc[sub.treatment_cetuximab == 0, "pfs_months"],
                 f"cetuximab effect in {name}")
    store(f"cetux_in_{name.replace(' ','_').replace('/','_')}", res)

# Formal interaction test: pfs ~ cetux + bm + cetux:bm
for bm in ["kras_mutation", "nras_mutation", "braf_v600e", "right_sided_primary"]:
    m = smf.ols(f"pfs_months ~ treatment_cetuximab*{bm}", data=df).fit()
    coef = m.params[f"treatment_cetuximab:{bm}"]
    p = m.pvalues[f"treatment_cetuximab:{bm}"]
    store(f"cetux_x_{bm}_interaction",
          {"effect": float(coef), "p_value": float(p),
           "significant": bool(p < 0.05),
           "result": f"cetux × {bm} interaction coef={coef:+.3f}, p={p:.2e}"})

# =====================================================================
# Iteration 6: Pembrolizumab × MSI-high (and other biomarkers)
# =====================================================================
print("\n=== Iteration 6: pembrolizumab interactions ===")
for predicate, name in [
    ((df.msi_high == 1), "MSI-high"),
    ((df.msi_high == 0), "MSS"),
    ((df.right_sided_primary == 1), "right-sided"),
    ((df.right_sided_primary == 0), "left-sided"),
]:
    sub = df.loc[predicate]
    res = t_test(sub.loc[sub.treatment_pembrolizumab == 1, "pfs_months"],
                 sub.loc[sub.treatment_pembrolizumab == 0, "pfs_months"],
                 f"pembro effect in {name}")
    store(f"pembro_in_{name.replace(' ','_').replace('-','_')}", res)

for bm in ["msi_high", "right_sided_primary", "braf_v600e"]:
    m = smf.ols(f"pfs_months ~ treatment_pembrolizumab*{bm}", data=df).fit()
    coef = m.params[f"treatment_pembrolizumab:{bm}"]
    p = m.pvalues[f"treatment_pembrolizumab:{bm}"]
    store(f"pembro_x_{bm}_interaction",
          {"effect": float(coef), "p_value": float(p),
           "significant": bool(p < 0.05),
           "result": f"pembro × {bm} interaction coef={coef:+.3f}, p={p:.2e}"})

# =====================================================================
# Iteration 7: Encorafenib × BRAF V600E, with cetuximab combo
# =====================================================================
print("\n=== Iteration 7: encorafenib interactions ===")
for predicate, name in [
    ((df.braf_v600e == 1), "BRAF V600E mutant"),
    ((df.braf_v600e == 0), "BRAF wildtype"),
]:
    sub = df.loc[predicate]
    res = t_test(sub.loc[sub.treatment_encorafenib == 1, "pfs_months"],
                 sub.loc[sub.treatment_encorafenib == 0, "pfs_months"],
                 f"encorafenib effect in {name}")
    store(f"encor_in_{name.replace(' ','_')}", res)

# encorafenib + cetuximab combo within BRAF V600E mutant
sub = df.loc[df.braf_v600e == 1]
for predicate, name in [
    ((sub.treatment_cetuximab == 1) & (sub.treatment_encorafenib == 1),
     "encor+cetux"),
    ((sub.treatment_cetuximab == 1) & (sub.treatment_encorafenib == 0),
     "cetux alone"),
    ((sub.treatment_cetuximab == 0) & (sub.treatment_encorafenib == 1),
     "encor alone"),
    ((sub.treatment_cetuximab == 0) & (sub.treatment_encorafenib == 0),
     "neither"),
]:
    n = int(predicate.sum())
    m = float(sub.loc[predicate, "pfs_months"].mean())
    print(f"BRAF V600E mut, {name}: n={n}, mean PFS={m:.3f}")

# Interaction term in BRAF V600E
m = smf.ols("pfs_months ~ treatment_encorafenib*braf_v600e", data=df).fit()
coef = m.params["treatment_encorafenib:braf_v600e"]
p = m.pvalues["treatment_encorafenib:braf_v600e"]
store("encor_x_braf_interaction",
      {"effect": float(coef), "p_value": float(p),
       "significant": bool(p < 0.05),
       "result": f"encor × BRAF V600E coef={coef:+.3f}, p={p:.2e}"})

# Three-way: encor, cetux, BRAF
m = smf.ols(
    "pfs_months ~ treatment_encorafenib*treatment_cetuximab*braf_v600e",
    data=df,
).fit()
key = "treatment_encorafenib:treatment_cetuximab:braf_v600e"
store("encor_cetux_braf_3way",
      {"effect": float(m.params[key]), "p_value": float(m.pvalues[key]),
       "significant": bool(m.pvalues[key] < 0.05),
       "result": f"encor × cetux × BRAF V600E coef={m.params[key]:+.3f}, p={m.pvalues[key]:.2e}"})

# =====================================================================
# Iteration 8: Trastuzumab+tucatinib × HER2 amplification
# =====================================================================
print("\n=== Iteration 8: trastuzumab+tucatinib interactions ===")
for predicate, name in [
    ((df.her2_amplified == 1), "HER2-amplified"),
    ((df.her2_amplified == 0), "HER2-not-amplified"),
    ((df.her2_amplified == 1) & (df.kras_mutation == 0) &
     (df.nras_mutation == 0) & (df.braf_v600e == 0),
     "HER2+ RAS/BRAF wildtype"),
    ((df.her2_amplified == 1) & (df.kras_mutation == 1),
     "HER2+ KRAS mutant"),
]:
    sub = df.loc[predicate]
    res = t_test(sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"],
                 sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"],
                 f"trast+tuc effect in {name}")
    store(f"trasttuc_in_{name.replace(' ','_').replace('+','p').replace('/','_')}", res)

m = smf.ols(
    "pfs_months ~ treatment_trastuzumab_tucatinib*her2_amplified", data=df,
).fit()
key = "treatment_trastuzumab_tucatinib:her2_amplified"
store("trasttuc_x_her2_interaction",
      {"effect": float(m.params[key]), "p_value": float(m.pvalues[key]),
       "significant": bool(m.pvalues[key] < 0.05),
       "result": f"trast+tuc × HER2 amp coef={m.params[key]:+.3f}, p={m.pvalues[key]:.2e}"})

# =====================================================================
# Iteration 9: Regorafenib effects (refractory salvage)
# =====================================================================
print("\n=== Iteration 9: regorafenib interactions ===")
for predicate, name in [
    ((df.ecog_ps == 0), "ECOG 0"),
    ((df.ecog_ps == 1), "ECOG 1"),
    ((df.ecog_ps == 2), "ECOG 2"),
    ((df.stage_iv == 1), "stage IV"),
    ((df.stage_iv == 0), "stage <IV"),
]:
    sub = df.loc[predicate]
    res = t_test(sub.loc[sub.treatment_regorafenib == 1, "pfs_months"],
                 sub.loc[sub.treatment_regorafenib == 0, "pfs_months"],
                 f"regorafenib effect in {name}")
    store(f"rego_in_{name.replace(' ','_').replace('<','lt')}", res)

m = smf.ols("pfs_months ~ treatment_regorafenib*ecog_ps", data=df).fit()
store("rego_x_ecog_interaction",
      {"effect": float(m.params["treatment_regorafenib:ecog_ps"]),
       "p_value": float(m.pvalues["treatment_regorafenib:ecog_ps"]),
       "significant": bool(m.pvalues["treatment_regorafenib:ecog_ps"] < 0.05),
       "result": f"rego × ECOG coef={m.params['treatment_regorafenib:ecog_ps']:+.3f}, p={m.pvalues['treatment_regorafenib:ecog_ps']:.2e}"})

# =====================================================================
# Iteration 10: Bevacizumab interactions
# =====================================================================
print("\n=== Iteration 10: bevacizumab interactions ===")
for predicate, name in [
    ((df.right_sided_primary == 1), "right-sided"),
    ((df.right_sided_primary == 0), "left-sided"),
    ((df.kras_mutation == 1), "KRAS mutant"),
    ((df.kras_mutation == 0), "KRAS wildtype"),
    ((df.stage_iv == 1), "stage IV"),
]:
    sub = df.loc[predicate]
    res = t_test(sub.loc[sub.treatment_bevacizumab == 1, "pfs_months"],
                 sub.loc[sub.treatment_bevacizumab == 0, "pfs_months"],
                 f"bev effect in {name}")
    store(f"bev_in_{name.replace(' ','_')}", res)

# =====================================================================
# Iteration 11: Multivariable adjusted model
# =====================================================================
print("\n=== Iteration 11: multivariable PFS model ===")
covars = ["age_years", "sex_female", "ecog_ps", "stage_iv",
          "right_sided_primary", "kras_mutation", "nras_mutation",
          "braf_v600e", "msi_high", "her2_amplified", "ntrk_fusion",
          "albumin_g_dl", "ldh_u_l", "cea_ng_ml", "weight_loss_pct_6mo",
          "crp_mg_l", "nlr", "hemoglobin_g_dl", "alkaline_phosphatase_u_l"] + trts
m = smf.ols("pfs_months ~ " + " + ".join(covars), data=df).fit()
mvar_summary = {
    var: {"coef": float(m.params[var]), "p": float(m.pvalues[var])}
    for var in covars
}
results["mvar_pfs"] = {
    "result": "Multivariable OLS PFS — coefficients listed below",
    "coefs": mvar_summary,
    "r2": float(m.rsquared),
    "n": int(m.nobs),
}
print("R^2 =", m.rsquared)
for v, d in mvar_summary.items():
    print(f"  {v}: {d['coef']:+.3f}  p={d['p']:.2e}")

# =====================================================================
# Iteration 12: Build performance/lab "high risk" composite,
# test interaction with treatments
# =====================================================================
print("\n=== Iteration 12: composite risk score ===")
# Standardize key prognostic labs and average — higher = worse
zalb = -(df.albumin_g_dl - df.albumin_g_dl.mean()) / df.albumin_g_dl.std()
zldh = (df.ldh_u_l - df.ldh_u_l.mean()) / df.ldh_u_l.std()
znlr = (df.nlr - df.nlr.mean()) / df.nlr.std()
zcrp = (df.crp_mg_l - df.crp_mg_l.mean()) / df.crp_mg_l.std()
zwl = (df.weight_loss_pct_6mo - df.weight_loss_pct_6mo.mean()) / df.weight_loss_pct_6mo.std()
df["risk_score"] = (zalb + zldh + znlr + zcrp + zwl) / 5

r, p = stats.pearsonr(df.risk_score, df.pfs_months)
store("risk_score_pfs",
      {"effect": float(r), "p_value": float(p),
       "significant": bool(p < 0.05),
       "result": f"composite risk score vs PFS Pearson r={r:.4f}, p={p:.2e}"})

# Quartile means
df["risk_q"] = pd.qcut(df.risk_score, 4, labels=False)
qmeans = df.groupby("risk_q").pfs_months.mean()
print("Risk quartile mean PFS:", qmeans.to_dict())

# =====================================================================
# Iteration 13: Comprehensive treatment × biomarker interaction screen
# =====================================================================
print("\n=== Iteration 13: full interaction screen ===")
biomarkers = ["right_sided_primary", "kras_mutation", "nras_mutation",
              "braf_v600e", "msi_high", "her2_amplified", "ntrk_fusion",
              "ecog_ps", "stage_iv", "sex_female"]
screen = []
for t in trts:
    for bm in biomarkers:
        try:
            m = smf.ols(f"pfs_months ~ {t}*{bm}", data=df).fit()
            key = f"{t}:{bm}"
            screen.append({
                "treatment": t, "biomarker": bm,
                "interaction_coef": float(m.params[key]),
                "interaction_p": float(m.pvalues[key]),
            })
        except Exception as e:
            screen.append({"treatment": t, "biomarker": bm, "error": str(e)})
screen_sorted = sorted(screen, key=lambda r: r.get("interaction_p", 1.0))
results["interaction_screen"] = {
    "result": "treatment × biomarker interaction screen on PFS, sorted by p",
    "rows": screen_sorted,
}
print("Top interactions:")
for row in screen_sorted[:15]:
    print(f"  {row['treatment']} × {row['biomarker']}: "
          f"coef={row.get('interaction_coef',0):+.3f}, "
          f"p={row.get('interaction_p',1):.2e}")

# =====================================================================
# Iteration 14: Refine cetuximab subgroup — RAS/BRAF wildtype + left-sided
# =====================================================================
print("\n=== Iteration 14: refine cetuximab subgroup ===")
combos = [
    (df.kras_mutation == 0, "KRASwt"),
    ((df.kras_mutation == 0) & (df.nras_mutation == 0), "RASwt"),
    ((df.kras_mutation == 0) & (df.nras_mutation == 0) & (df.braf_v600e == 0),
     "RAS/BRAFwt"),
    ((df.kras_mutation == 0) & (df.nras_mutation == 0) & (df.braf_v600e == 0) &
     (df.right_sided_primary == 0), "RAS/BRAFwt + left-sided"),
    ((df.kras_mutation == 0) & (df.nras_mutation == 0) & (df.braf_v600e == 0) &
     (df.right_sided_primary == 1), "RAS/BRAFwt + right-sided"),
]
for predicate, name in combos:
    sub = df.loc[predicate]
    res = t_test(sub.loc[sub.treatment_cetuximab == 1, "pfs_months"],
                 sub.loc[sub.treatment_cetuximab == 0, "pfs_months"],
                 f"cetux effect in {name}")
    store(f"cetux_refine_{name.replace(' ','_').replace('/','_').replace('+','p')}", res)

# =====================================================================
# Iteration 15: Refine pembrolizumab — MSI-high subgroup details
# =====================================================================
print("\n=== Iteration 15: refine pembrolizumab subgroup ===")
combos = [
    (df.msi_high == 1, "MSI-high"),
    ((df.msi_high == 1) & (df.right_sided_primary == 1), "MSI-high right-sided"),
    ((df.msi_high == 1) & (df.right_sided_primary == 0), "MSI-high left-sided"),
    ((df.msi_high == 1) & (df.braf_v600e == 1), "MSI-high BRAFmut"),
    ((df.msi_high == 1) & (df.braf_v600e == 0), "MSI-high BRAFwt"),
    ((df.msi_high == 1) & (df.ecog_ps <= 1), "MSI-high ECOG<=1"),
    ((df.msi_high == 1) & (df.stage_iv == 1), "MSI-high stage IV"),
]
for predicate, name in combos:
    sub = df.loc[predicate]
    res = t_test(sub.loc[sub.treatment_pembrolizumab == 1, "pfs_months"],
                 sub.loc[sub.treatment_pembrolizumab == 0, "pfs_months"],
                 f"pembro effect in {name}")
    store(f"pembro_refine_{name.replace(' ','_').replace('-','_').replace('<=','le').replace('+','p')}", res)

# =====================================================================
# Iteration 16: Refine encorafenib — BRAF V600E with/without combo
# =====================================================================
print("\n=== Iteration 16: encorafenib in BRAF V600E ===")
combos = [
    (df.braf_v600e == 1, "BRAFmut"),
    ((df.braf_v600e == 1) & (df.treatment_cetuximab == 1),
     "BRAFmut+cetux"),
    ((df.braf_v600e == 1) & (df.treatment_cetuximab == 0),
     "BRAFmut no cetux"),
    ((df.braf_v600e == 1) & (df.right_sided_primary == 1),
     "BRAFmut right"),
    ((df.braf_v600e == 1) & (df.right_sided_primary == 0),
     "BRAFmut left"),
]
for predicate, name in combos:
    sub = df.loc[predicate]
    res = t_test(sub.loc[sub.treatment_encorafenib == 1, "pfs_months"],
                 sub.loc[sub.treatment_encorafenib == 0, "pfs_months"],
                 f"encor effect in {name}")
    store(f"encor_refine_{name.replace(' ','_').replace('+','p')}", res)

# =====================================================================
# Iteration 17: Refine trastuzumab+tucatinib — HER2+ refinement
# =====================================================================
print("\n=== Iteration 17: trast+tuc in HER2+ ===")
combos = [
    (df.her2_amplified == 1, "HER2+"),
    ((df.her2_amplified == 1) & (df.kras_mutation == 0) &
     (df.nras_mutation == 0) & (df.braf_v600e == 0), "HER2+ RAS/BRAFwt"),
    ((df.her2_amplified == 1) & (df.right_sided_primary == 0), "HER2+ left"),
    ((df.her2_amplified == 1) & (df.right_sided_primary == 1), "HER2+ right"),
    ((df.her2_amplified == 1) & (df.ecog_ps <= 1), "HER2+ ECOG<=1"),
]
for predicate, name in combos:
    sub = df.loc[predicate]
    res = t_test(sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"],
                 sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"],
                 f"trast+tuc effect in {name}")
    store(f"trasttuc_refine_{name.replace(' ','_').replace('+','p').replace('/','_').replace('<=','le')}", res)

# =====================================================================
# Iteration 18: Regorafenib refinement
# =====================================================================
print("\n=== Iteration 18: regorafenib refinement ===")
combos = [
    (df.ecog_ps == 0, "ECOG0"),
    ((df.ecog_ps == 0) & (df.albumin_g_dl >= 3.5), "ECOG0 albumin>=3.5"),
    ((df.ecog_ps == 0) & (df.albumin_g_dl < 3.5), "ECOG0 albumin<3.5"),
    ((df.ecog_ps <= 1), "ECOG<=1"),
    ((df.ecog_ps >= 1), "ECOG>=1"),
]
for predicate, name in combos:
    sub = df.loc[predicate]
    res = t_test(sub.loc[sub.treatment_regorafenib == 1, "pfs_months"],
                 sub.loc[sub.treatment_regorafenib == 0, "pfs_months"],
                 f"rego effect in {name}")
    store(f"rego_refine_{name.replace(' ','_').replace('<=','le').replace('>=','ge').replace('<','lt').replace('+','p')}", res)

# =====================================================================
# Iteration 19: NTRK fusion / rare biomarker subgroup
# =====================================================================
print("\n=== Iteration 19: rare biomarkers ===")
sub = df.loc[df.ntrk_fusion == 1]
print(f"NTRK fusion n={len(sub)}, mean PFS={sub.pfs_months.mean():.3f}")
for t in trts:
    res = t_test(sub.loc[sub[t] == 1, "pfs_months"],
                 sub.loc[sub[t] == 0, "pfs_months"],
                 f"{t} in NTRK+")
    store(f"ntrk_{t}", res)

# =====================================================================
# Iteration 20: Three-way: cetuximab × KRAS-wt × right-sided / left-sided
# =====================================================================
print("\n=== Iteration 20: cetuximab three-way ===")
m = smf.ols(
    "pfs_months ~ treatment_cetuximab*kras_mutation*right_sided_primary",
    data=df,
).fit()
for k in ["treatment_cetuximab", "treatment_cetuximab:kras_mutation",
          "treatment_cetuximab:right_sided_primary",
          "treatment_cetuximab:kras_mutation:right_sided_primary"]:
    if k in m.params.index:
        store(f"cetux_3way_{k.replace(':','_x_')}",
              {"effect": float(m.params[k]), "p_value": float(m.pvalues[k]),
               "significant": bool(m.pvalues[k] < 0.05),
               "result": f"{k}: coef={m.params[k]:+.3f}, p={m.pvalues[k]:.2e}"})

# =====================================================================
# Iteration 21: Test treatment combos within RAS/BRAF wt
# =====================================================================
print("\n=== Iteration 21: combos in RAS/BRAFwt ===")
sub = df.loc[(df.kras_mutation == 0) & (df.nras_mutation == 0) &
             (df.braf_v600e == 0)]
for t in trts:
    res = t_test(sub.loc[sub[t] == 1, "pfs_months"],
                 sub.loc[sub[t] == 0, "pfs_months"],
                 f"{t} in RAS/BRAFwt")
    store(f"rasBRAFwt_{t}", res)

# =====================================================================
# Iteration 22: Test all treatments within MSI-high
# =====================================================================
print("\n=== Iteration 22: treatments in MSI-high ===")
sub = df.loc[df.msi_high == 1]
for t in trts:
    res = t_test(sub.loc[sub[t] == 1, "pfs_months"],
                 sub.loc[sub[t] == 0, "pfs_months"],
                 f"{t} in MSI-high")
    store(f"msi_{t}", res)

# =====================================================================
# Iteration 23: Sensitivity analyses with multivariable adjustment of
# best subgroup hits
# =====================================================================
print("\n=== Iteration 23: multivariable confirm subgroup effects ===")
# Cetuximab effect within RAS/BRAFwt + left-sided, adjusted
sub = df.loc[(df.kras_mutation == 0) & (df.nras_mutation == 0) &
             (df.braf_v600e == 0) & (df.right_sided_primary == 0)].copy()
m = smf.ols(
    "pfs_months ~ treatment_cetuximab + age_years + sex_female + "
    "ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + cea_ng_ml + "
    "treatment_bevacizumab + treatment_pembrolizumab + treatment_encorafenib + "
    "treatment_trastuzumab_tucatinib + treatment_regorafenib", data=sub,
).fit()
key = "treatment_cetuximab"
store("cetux_adj_RASBRAFwt_left",
      {"effect": float(m.params[key]), "p_value": float(m.pvalues[key]),
       "significant": bool(m.pvalues[key] < 0.05),
       "result": f"cetux adjusted in RAS/BRAFwt+left coef={m.params[key]:+.3f}, p={m.pvalues[key]:.2e}, n={int(m.nobs)}"})

# Pembrolizumab adjusted within MSI-high
sub = df.loc[df.msi_high == 1].copy()
m = smf.ols(
    "pfs_months ~ treatment_pembrolizumab + age_years + sex_female + "
    "ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + cea_ng_ml + "
    "treatment_bevacizumab + treatment_cetuximab + treatment_encorafenib + "
    "treatment_trastuzumab_tucatinib + treatment_regorafenib", data=sub,
).fit()
key = "treatment_pembrolizumab"
store("pembro_adj_MSIhigh",
      {"effect": float(m.params[key]), "p_value": float(m.pvalues[key]),
       "significant": bool(m.pvalues[key] < 0.05),
       "result": f"pembro adjusted in MSI-high coef={m.params[key]:+.3f}, p={m.pvalues[key]:.2e}, n={int(m.nobs)}"})

# Encorafenib adjusted within BRAF V600E
sub = df.loc[df.braf_v600e == 1].copy()
m = smf.ols(
    "pfs_months ~ treatment_encorafenib + age_years + sex_female + "
    "ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + cea_ng_ml + "
    "treatment_bevacizumab + treatment_cetuximab + treatment_pembrolizumab + "
    "treatment_trastuzumab_tucatinib + treatment_regorafenib", data=sub,
).fit()
key = "treatment_encorafenib"
store("encor_adj_BRAF",
      {"effect": float(m.params[key]), "p_value": float(m.pvalues[key]),
       "significant": bool(m.pvalues[key] < 0.05),
       "result": f"encor adjusted in BRAF V600E coef={m.params[key]:+.3f}, p={m.pvalues[key]:.2e}, n={int(m.nobs)}"})

# Trast+tuc adjusted within HER2+
sub = df.loc[df.her2_amplified == 1].copy()
m = smf.ols(
    "pfs_months ~ treatment_trastuzumab_tucatinib + age_years + sex_female + "
    "ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + cea_ng_ml + "
    "treatment_bevacizumab + treatment_cetuximab + treatment_pembrolizumab + "
    "treatment_encorafenib + treatment_regorafenib", data=sub,
).fit()
key = "treatment_trastuzumab_tucatinib"
store("trasttuc_adj_HER2",
      {"effect": float(m.params[key]), "p_value": float(m.pvalues[key]),
       "significant": bool(m.pvalues[key] < 0.05),
       "result": f"trast+tuc adjusted in HER2+ coef={m.params[key]:+.3f}, p={m.pvalues[key]:.2e}, n={int(m.nobs)}"})

# Regorafenib adjusted overall
m = smf.ols(
    "pfs_months ~ treatment_regorafenib + age_years + sex_female + "
    "ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + cea_ng_ml + "
    "treatment_bevacizumab + treatment_cetuximab + treatment_pembrolizumab + "
    "treatment_encorafenib + treatment_trastuzumab_tucatinib", data=df,
).fit()
key = "treatment_regorafenib"
store("rego_adj_full",
      {"effect": float(m.params[key]), "p_value": float(m.pvalues[key]),
       "significant": bool(m.pvalues[key] < 0.05),
       "result": f"rego adjusted overall coef={m.params[key]:+.3f}, p={m.pvalues[key]:.2e}"})

# Bevacizumab adjusted overall
m = smf.ols(
    "pfs_months ~ treatment_bevacizumab + age_years + sex_female + "
    "ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + cea_ng_ml + "
    "treatment_regorafenib + treatment_cetuximab + treatment_pembrolizumab + "
    "treatment_encorafenib + treatment_trastuzumab_tucatinib", data=df,
).fit()
key = "treatment_bevacizumab"
store("bev_adj_full",
      {"effect": float(m.params[key]), "p_value": float(m.pvalues[key]),
       "significant": bool(m.pvalues[key] < 0.05),
       "result": f"bev adjusted overall coef={m.params[key]:+.3f}, p={m.pvalues[key]:.2e}"})

# =====================================================================
# Iteration 24: Tree-based subgroup discovery
# Use a small decision tree on residuals to find heterogeneous subgroups
# =====================================================================
print("\n=== Iteration 24: tree-based heterogeneity ===")
from sklearn.tree import DecisionTreeRegressor, export_text

# Compute treatment effect heterogeneity by fitting a tree to residuals
# of pfs ~ treatment for each treatment, then check splits
tree_results = {}
for t in trts:
    other = df.copy()
    # Naive: predict PFS difference via T-learner using a small tree
    m = smf.ols(f"pfs_months ~ {t}", data=df).fit()
    other["resid"] = df["pfs_months"] - m.predict(df)
    # Subgroup-by-subgroup: examine top features that split treated vs untreated
    treat_mask = df[t] == 1
    feats = ["age_years", "sex_female", "ecog_ps", "stage_iv",
             "right_sided_primary", "kras_mutation", "nras_mutation",
             "braf_v600e", "msi_high", "her2_amplified", "ntrk_fusion",
             "albumin_g_dl", "ldh_u_l", "cea_ng_ml", "weight_loss_pct_6mo",
             "crp_mg_l", "nlr"]
    # T-learner: fit a tree to predict pfs in treated and in untreated, then
    # estimate CATE = pfs_treated_pred - pfs_untreated_pred
    t1 = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200,
                               random_state=0)
    t0 = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200,
                               random_state=0)
    t1.fit(df.loc[treat_mask, feats], df.loc[treat_mask, "pfs_months"])
    t0.fit(df.loc[~treat_mask, feats], df.loc[~treat_mask, "pfs_months"])
    cate = t1.predict(df[feats]) - t0.predict(df[feats])
    cate_mean = float(cate.mean())
    cate_q90 = float(np.quantile(cate, 0.9))
    cate_q10 = float(np.quantile(cate, 0.1))
    tree_results[t] = {"mean": cate_mean, "q10": cate_q10, "q90": cate_q90}
    print(f"{t}: CATE mean={cate_mean:+.3f}, q10={cate_q10:+.3f}, q90={cate_q90:+.3f}")

results["cate_summary"] = {"result": "T-learner CATE summary",
                          "rows": tree_results}

# =====================================================================
# Iteration 25: Final consolidated subgroup definitions
# Confirm with full conditioning what the "best" subgroup for each
# treatment looks like.
# =====================================================================
print("\n=== Iteration 25: final consolidated subgroups ===")

final_subgroups = {}

# Cetuximab: RAS/BRAF wildtype, possibly left-sided
sub = df.loc[(df.kras_mutation == 0) & (df.nras_mutation == 0) &
             (df.braf_v600e == 0) & (df.right_sided_primary == 0)]
res = t_test(sub.loc[sub.treatment_cetuximab == 1, "pfs_months"],
             sub.loc[sub.treatment_cetuximab == 0, "pfs_months"],
             "cetuximab in RAS/BRAFwt + left-sided")
final_subgroups["cetuximab"] = res
store("final_cetux", res)

# Pembrolizumab: MSI-high
sub = df.loc[df.msi_high == 1]
res = t_test(sub.loc[sub.treatment_pembrolizumab == 1, "pfs_months"],
             sub.loc[sub.treatment_pembrolizumab == 0, "pfs_months"],
             "pembro in MSI-high")
final_subgroups["pembrolizumab"] = res
store("final_pembro", res)

# Encorafenib: BRAF V600E
sub = df.loc[df.braf_v600e == 1]
res = t_test(sub.loc[sub.treatment_encorafenib == 1, "pfs_months"],
             sub.loc[sub.treatment_encorafenib == 0, "pfs_months"],
             "encor in BRAF V600E")
final_subgroups["encorafenib"] = res
store("final_encor", res)

# Trast+tuc: HER2 amplified, RAS/BRAF wildtype
sub = df.loc[(df.her2_amplified == 1) & (df.kras_mutation == 0) &
             (df.nras_mutation == 0) & (df.braf_v600e == 0)]
res = t_test(sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"],
             sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"],
             "trast+tuc in HER2+ RAS/BRAFwt")
final_subgroups["trast_tuc"] = res
store("final_trasttuc", res)

# Save all
with open("results_main.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print("\nSaved results_main.json")
