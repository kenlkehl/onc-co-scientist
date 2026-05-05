"""Comprehensive iterative analysis of ds001_crc dataset.
Captures effect estimates and p-values for each hypothesis in every iteration.
Output is dumped to results.json for later assembly into transcript.json.
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
print("Loaded", df.shape)

results = {}

def record(key, summary, effect=None, p=None, sig=None):
    eff_v = float(effect) if effect is not None and np.isfinite(effect) else None
    p_v = float(p) if p is not None and np.isfinite(p) else None
    if sig is None:
        sig_v = bool(p_v is not None and p_v < 0.05)
    else:
        sig_v = bool(sig)
    results[key] = {
        "result_summary": summary,
        "effect_estimate": eff_v,
        "p_value": p_v,
        "significant": sig_v,
    }
    print(f"[{key}] eff={eff_v} p={p_v} sig={sig_v}")

# Treatment labels
TX = [
    "treatment_cetuximab",
    "treatment_bevacizumab",
    "treatment_pembrolizumab",
    "treatment_encorafenib",
    "treatment_trastuzumab_tucatinib",
    "treatment_regorafenib",
]

# ---------------- Iteration 1: Outcome distribution & baseline associations ---------------- #
# h1: PFS is positively associated with albumin
m = smf.ols("pfs_months ~ albumin_g_dl", data=df).fit()
record("h1", f"OLS pfs ~ albumin: beta={m.params['albumin_g_dl']:.3f} per g/dL, p={m.pvalues['albumin_g_dl']:.2e}",
       effect=m.params["albumin_g_dl"], p=m.pvalues["albumin_g_dl"])

# h2: PFS is negatively associated with ECOG performance status
m = smf.ols("pfs_months ~ ecog_ps", data=df).fit()
record("h2", f"OLS pfs ~ ecog_ps: beta={m.params['ecog_ps']:.3f} months per ECOG point, p={m.pvalues['ecog_ps']:.2e}",
       effect=m.params["ecog_ps"], p=m.pvalues["ecog_ps"])

# h3: PFS is shorter in stage IV vs not
mean_iv = df.loc[df.stage_iv == 1, "pfs_months"].mean()
mean_not = df.loc[df.stage_iv == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(df.loc[df.stage_iv == 1, "pfs_months"], df.loc[df.stage_iv == 0, "pfs_months"])
record("h3", f"Mean PFS stage_iv=1: {mean_iv:.3f} vs stage_iv=0: {mean_not:.3f} (t-test p={p:.2e})",
       effect=mean_iv - mean_not, p=p)

# ---------------- Iteration 2: More baseline labs ---------------- #
# h4: PFS negatively associated with CEA (log-transformed)
df["log_cea"] = np.log1p(df.cea_ng_ml)
m = smf.ols("pfs_months ~ log_cea", data=df).fit()
record("h4", f"OLS pfs ~ log(1+CEA): beta={m.params['log_cea']:.3f}, p={m.pvalues['log_cea']:.2e}",
       effect=m.params["log_cea"], p=m.pvalues["log_cea"])

# h5: PFS negatively associated with LDH
m = smf.ols("pfs_months ~ ldh_u_l", data=df).fit()
record("h5", f"OLS pfs ~ LDH: beta={m.params['ldh_u_l']:.4f} per U/L, p={m.pvalues['ldh_u_l']:.2e}",
       effect=m.params["ldh_u_l"], p=m.pvalues["ldh_u_l"])

# h6: PFS negatively associated with NLR
m = smf.ols("pfs_months ~ nlr", data=df).fit()
record("h6", f"OLS pfs ~ NLR: beta={m.params['nlr']:.3f}, p={m.pvalues['nlr']:.2e}",
       effect=m.params["nlr"], p=m.pvalues["nlr"])

# h7: PFS negatively associated with CRP
m = smf.ols("pfs_months ~ crp_mg_l", data=df).fit()
record("h7", f"OLS pfs ~ CRP: beta={m.params['crp_mg_l']:.4f} per mg/L, p={m.pvalues['crp_mg_l']:.2e}",
       effect=m.params["crp_mg_l"], p=m.pvalues["crp_mg_l"])

# h8: PFS negatively associated with weight loss
m = smf.ols("pfs_months ~ weight_loss_pct_6mo", data=df).fit()
record("h8", f"OLS pfs ~ weight loss %: beta={m.params['weight_loss_pct_6mo']:.4f}, p={m.pvalues['weight_loss_pct_6mo']:.2e}",
       effect=m.params["weight_loss_pct_6mo"], p=m.pvalues["weight_loss_pct_6mo"])

# ---------------- Iteration 3: Demographics ---------------- #
# h9: PFS varies with age
m = smf.ols("pfs_months ~ age_years", data=df).fit()
record("h9", f"OLS pfs ~ age: beta={m.params['age_years']:.4f} per year, p={m.pvalues['age_years']:.2e}",
       effect=m.params["age_years"], p=m.pvalues["age_years"])

# h10: PFS differs by sex
mean_f = df.loc[df.sex_female == 1, "pfs_months"].mean()
mean_m = df.loc[df.sex_female == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(df.loc[df.sex_female == 1, "pfs_months"], df.loc[df.sex_female == 0, "pfs_months"])
record("h10", f"Mean PFS female: {mean_f:.3f} vs male: {mean_m:.3f} (t-test p={p:.2e})",
       effect=mean_f - mean_m, p=p)

# h11: PFS differs by tumor sidedness
mean_r = df.loc[df.right_sided_primary == 1, "pfs_months"].mean()
mean_l = df.loc[df.right_sided_primary == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(df.loc[df.right_sided_primary == 1, "pfs_months"], df.loc[df.right_sided_primary == 0, "pfs_months"])
record("h11", f"Mean PFS right-sided: {mean_r:.3f} vs left-sided: {mean_l:.3f} (t-test p={p:.2e})",
       effect=mean_r - mean_l, p=p)

# ---------------- Iteration 4: Treatment main effects ---------------- #
for tx in TX:
    on = df.loc[df[tx] == 1, "pfs_months"].mean()
    off = df.loc[df[tx] == 0, "pfs_months"].mean()
    t, p = stats.ttest_ind(df.loc[df[tx] == 1, "pfs_months"], df.loc[df[tx] == 0, "pfs_months"])
    key = f"main_{tx}"
    record(key, f"Mean PFS {tx}=1: {on:.3f} vs =0: {off:.3f} (t-test p={p:.2e})",
           effect=on - off, p=p)

# ---------------- Iteration 5: Cetuximab x KRAS ---------------- #
# h_cetux_kras: cetuximab benefits patients with KRAS WT, harms/no benefit if KRAS mut
for kras in [0, 1]:
    sub = df[df.kras_mutation == kras]
    on = sub.loc[sub.treatment_cetuximab == 1, "pfs_months"].mean()
    off = sub.loc[sub.treatment_cetuximab == 0, "pfs_months"].mean()
    t, p = stats.ttest_ind(sub.loc[sub.treatment_cetuximab == 1, "pfs_months"], sub.loc[sub.treatment_cetuximab == 0, "pfs_months"])
    record(f"cetux_kras{kras}", f"Within KRAS={kras}: cetuximab effect = {on-off:+.3f} months (n_on={int(sub.treatment_cetuximab.sum())}, p={p:.2e})",
           effect=on - off, p=p)

# Interaction test
m = smf.ols("pfs_months ~ treatment_cetuximab * kras_mutation", data=df).fit()
key_b = "treatment_cetuximab:kras_mutation"
record("cetux_kras_interaction", f"OLS interaction beta={m.params[key_b]:.3f}, p={m.pvalues[key_b]:.2e}",
       effect=m.params[key_b], p=m.pvalues[key_b])

# ---------------- Iteration 6: Cetuximab x NRAS, BRAF, sidedness ---------------- #
for bm in ["nras_mutation", "braf_v600e", "right_sided_primary"]:
    for v in [0, 1]:
        sub = df[df[bm] == v]
        on = sub.loc[sub.treatment_cetuximab == 1, "pfs_months"].mean()
        off = sub.loc[sub.treatment_cetuximab == 0, "pfs_months"].mean()
        if sub.treatment_cetuximab.sum() > 1 and (sub.treatment_cetuximab == 0).sum() > 1:
            t, p = stats.ttest_ind(sub.loc[sub.treatment_cetuximab == 1, "pfs_months"], sub.loc[sub.treatment_cetuximab == 0, "pfs_months"])
        else:
            p = np.nan
        record(f"cetux_{bm}{v}", f"Within {bm}={v}: cetuximab effect = {on-off:+.3f} months (p={p:.2e})",
               effect=on - off, p=p)
    m = smf.ols(f"pfs_months ~ treatment_cetuximab * {bm}", data=df).fit()
    key_b = f"treatment_cetuximab:{bm}"
    record(f"cetux_{bm}_interaction", f"OLS interaction beta={m.params[key_b]:.3f}, p={m.pvalues[key_b]:.2e}",
           effect=m.params[key_b], p=m.pvalues[key_b])

# ---------------- Iteration 7: Cetuximab in fully wild-type, left-sided ---------------- #
mask_wt = (df.kras_mutation == 0) & (df.nras_mutation == 0) & (df.braf_v600e == 0)
sub = df[mask_wt]
on = sub.loc[sub.treatment_cetuximab == 1, "pfs_months"].mean()
off = sub.loc[sub.treatment_cetuximab == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(sub.loc[sub.treatment_cetuximab == 1, "pfs_months"], sub.loc[sub.treatment_cetuximab == 0, "pfs_months"])
record("cetux_full_wt", f"In RAS/RAF wild-type (n={len(sub)}): cetuximab PFS effect = {on-off:+.3f} months (p={p:.2e})",
       effect=on - off, p=p)

mask_wt_left = mask_wt & (df.right_sided_primary == 0)
sub = df[mask_wt_left]
on = sub.loc[sub.treatment_cetuximab == 1, "pfs_months"].mean()
off = sub.loc[sub.treatment_cetuximab == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(sub.loc[sub.treatment_cetuximab == 1, "pfs_months"], sub.loc[sub.treatment_cetuximab == 0, "pfs_months"])
record("cetux_full_wt_left", f"In RAS/RAF WT, left-sided (n={len(sub)}): cetuximab PFS effect = {on-off:+.3f} months (p={p:.2e})",
       effect=on - off, p=p)

# Same in mutant or right-sided (complement)
mask_alt = (~mask_wt_left) & (df.treatment_cetuximab.isin([0, 1]))
sub = df[~mask_wt_left]
on = sub.loc[sub.treatment_cetuximab == 1, "pfs_months"].mean()
off = sub.loc[sub.treatment_cetuximab == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(sub.loc[sub.treatment_cetuximab == 1, "pfs_months"], sub.loc[sub.treatment_cetuximab == 0, "pfs_months"])
record("cetux_NOT_full_wt_left", f"Outside RAS/RAF-WT-left (n={len(sub)}): cetuximab PFS effect = {on-off:+.3f} months (p={p:.2e})",
       effect=on - off, p=p)

# ---------------- Iteration 8: Pembrolizumab x MSI-high ---------------- #
for v in [0, 1]:
    sub = df[df.msi_high == v]
    on = sub.loc[sub.treatment_pembrolizumab == 1, "pfs_months"].mean()
    off = sub.loc[sub.treatment_pembrolizumab == 0, "pfs_months"].mean()
    if sub.treatment_pembrolizumab.sum() > 1:
        t, p = stats.ttest_ind(sub.loc[sub.treatment_pembrolizumab == 1, "pfs_months"], sub.loc[sub.treatment_pembrolizumab == 0, "pfs_months"])
    else:
        p = np.nan
    record(f"pembro_msi{v}", f"Within msi_high={v}: pembrolizumab effect = {on-off:+.3f} months (p={p:.2e})",
           effect=on - off, p=p)
m = smf.ols("pfs_months ~ treatment_pembrolizumab * msi_high", data=df).fit()
key_b = "treatment_pembrolizumab:msi_high"
record("pembro_msi_interaction", f"OLS interaction beta={m.params[key_b]:.3f}, p={m.pvalues[key_b]:.2e}",
       effect=m.params[key_b], p=m.pvalues[key_b])

# ---------------- Iteration 9: Encorafenib x BRAF V600E ---------------- #
for v in [0, 1]:
    sub = df[df.braf_v600e == v]
    on = sub.loc[sub.treatment_encorafenib == 1, "pfs_months"].mean()
    off = sub.loc[sub.treatment_encorafenib == 0, "pfs_months"].mean()
    if sub.treatment_encorafenib.sum() > 1 and (sub.treatment_encorafenib == 0).sum() > 1:
        t, p = stats.ttest_ind(sub.loc[sub.treatment_encorafenib == 1, "pfs_months"], sub.loc[sub.treatment_encorafenib == 0, "pfs_months"])
    else:
        p = np.nan
    record(f"enco_braf{v}", f"Within braf_v600e={v}: encorafenib effect = {on-off:+.3f} months (p={p:.2e})",
           effect=on - off, p=p)
m = smf.ols("pfs_months ~ treatment_encorafenib * braf_v600e", data=df).fit()
key_b = "treatment_encorafenib:braf_v600e"
record("enco_braf_interaction", f"OLS interaction beta={m.params[key_b]:.3f}, p={m.pvalues[key_b]:.2e}",
       effect=m.params[key_b], p=m.pvalues[key_b])

# ---------------- Iteration 10: Trastuzumab/tucatinib x HER2 ---------------- #
for v in [0, 1]:
    sub = df[df.her2_amplified == v]
    on = sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"].mean()
    off = sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"].mean()
    if sub.treatment_trastuzumab_tucatinib.sum() > 1 and (sub.treatment_trastuzumab_tucatinib == 0).sum() > 1:
        t, p = stats.ttest_ind(sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"], sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"])
    else:
        p = np.nan
    record(f"her2tx_her2{v}", f"Within her2_amplified={v}: trastuzumab/tucatinib effect = {on-off:+.3f} months (p={p:.2e})",
           effect=on - off, p=p)
m = smf.ols("pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified", data=df).fit()
key_b = "treatment_trastuzumab_tucatinib:her2_amplified"
record("her2tx_her2_interaction", f"OLS interaction beta={m.params[key_b]:.3f}, p={m.pvalues[key_b]:.2e}",
       effect=m.params[key_b], p=m.pvalues[key_b])

# Maybe HER2 tx requires RAS WT also
mask = (df.her2_amplified == 1) & (df.kras_mutation == 0) & (df.nras_mutation == 0)
sub = df[mask]
on = sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"].mean()
off = sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"].mean()
if sub.treatment_trastuzumab_tucatinib.sum() > 1 and (sub.treatment_trastuzumab_tucatinib == 0).sum() > 1:
    t, p = stats.ttest_ind(sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"], sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"])
else:
    p = np.nan
record("her2tx_her2_rasWT", f"In HER2+ AND KRAS WT AND NRAS WT (n={len(sub)}): trastuzumab/tucatinib PFS effect = {on-off:+.3f} months (p={p:.2e})",
       effect=on - off, p=p)

mask_alt = (df.her2_amplified == 1) & ((df.kras_mutation == 1) | (df.nras_mutation == 1))
sub = df[mask_alt]
on = sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"].mean()
off = sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"].mean()
if sub.treatment_trastuzumab_tucatinib.sum() > 1 and (sub.treatment_trastuzumab_tucatinib == 0).sum() > 1:
    t, p = stats.ttest_ind(sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"], sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"])
else:
    p = np.nan
record("her2tx_her2_rasMUT", f"In HER2+ AND (KRAS or NRAS mut) (n={len(sub)}): trastuzumab/tucatinib PFS effect = {on-off:+.3f} months (p={p:.2e})",
       effect=on - off, p=p)

# ---------------- Iteration 11: Bevacizumab subgroups ---------------- #
# Test bevacizumab effect across multiple subgroups
m = smf.ols("pfs_months ~ treatment_bevacizumab * kras_mutation", data=df).fit()
key_b = "treatment_bevacizumab:kras_mutation"
record("bev_kras_interaction", f"OLS interaction bev*kras beta={m.params[key_b]:.3f}, p={m.pvalues[key_b]:.2e}",
       effect=m.params[key_b], p=m.pvalues[key_b])

m = smf.ols("pfs_months ~ treatment_bevacizumab * right_sided_primary", data=df).fit()
key_b = "treatment_bevacizumab:right_sided_primary"
record("bev_side_interaction", f"OLS interaction bev*right_sided beta={m.params[key_b]:.3f}, p={m.pvalues[key_b]:.2e}",
       effect=m.params[key_b], p=m.pvalues[key_b])

# ---------------- Iteration 12: Regorafenib subgroups ---------------- #
# Hypothesis: regorafenib benefit may depend on a biomarker (often associated with later-line use; test multiple)
for bm in ["kras_mutation", "right_sided_primary", "msi_high", "stage_iv", "her2_amplified", "braf_v600e"]:
    m = smf.ols(f"pfs_months ~ treatment_regorafenib * {bm}", data=df).fit()
    key_b = f"treatment_regorafenib:{bm}"
    record(f"rego_{bm}_interaction", f"OLS interaction rego*{bm} beta={m.params[key_b]:.3f}, p={m.pvalues[key_b]:.2e}",
           effect=m.params[key_b], p=m.pvalues[key_b])

# Continuous modifiers - ECOG, age, CEA, albumin
for bm in ["ecog_ps", "age_years", "log_cea", "albumin_g_dl", "ldh_u_l", "nlr"]:
    m = smf.ols(f"pfs_months ~ treatment_regorafenib * {bm}", data=df).fit()
    key_b = f"treatment_regorafenib:{bm}"
    record(f"rego_{bm}_interaction", f"OLS interaction rego*{bm} beta={m.params[key_b]:.4f}, p={m.pvalues[key_b]:.2e}",
           effect=m.params[key_b], p=m.pvalues[key_b])

# ---------------- Iteration 13: Multivariable Cox-style OLS / adjusted treatment effects ---------------- #
covars = ["age_years", "sex_female", "ecog_ps", "stage_iv", "right_sided_primary",
          "kras_mutation", "nras_mutation", "braf_v600e", "msi_high", "her2_amplified",
          "log_cea", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr",
          "hemoglobin_g_dl"]
formula = "pfs_months ~ " + " + ".join(covars + TX)
m_full = smf.ols(formula, data=df).fit()
for tx in TX:
    record(f"adj_{tx}", f"Adjusted OLS effect of {tx} on PFS: beta={m_full.params[tx]:.3f} months, p={m_full.pvalues[tx]:.2e}",
           effect=m_full.params[tx], p=m_full.pvalues[tx])

# ---------------- Iteration 14: Treatment-by-feature interaction screen for each treatment ---------------- #
# For each treatment, test interactions with every feature (a la heterogeneity screen)
features = ["age_years", "sex_female", "ecog_ps", "stage_iv", "right_sided_primary",
            "kras_mutation", "nras_mutation", "braf_v600e", "msi_high", "her2_amplified",
            "ntrk_fusion", "log_cea", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo",
            "crp_mg_l", "nlr", "hemoglobin_g_dl"]

screen_top = {}
for tx in TX:
    out = []
    for f in features:
        try:
            m = smf.ols(f"pfs_months ~ {tx} * {f}", data=df).fit()
            key_b = f"{tx}:{f}"
            out.append((f, m.params[key_b], m.pvalues[key_b]))
        except Exception as e:
            pass
    out.sort(key=lambda x: x[2])
    screen_top[tx] = out[:3]
    s = "; ".join([f"{f}: beta={b:+.3f} p={p:.2e}" for f, b, p in out[:3]])
    record(f"screen_top3_{tx}", f"Top 3 modifier candidates for {tx}: {s}",
           effect=out[0][1], p=out[0][2])

# ---------------- Iteration 15: Refined top subgroups for each treatment ---------------- #
# Build "best" subgroup hypothesis for each treatment using top 1-2 modifiers
# Cetuximab: KRAS WT AND NRAS WT AND BRAF WT AND left-sided
mask = (df.kras_mutation == 0) & (df.nras_mutation == 0) & (df.braf_v600e == 0) & (df.right_sided_primary == 0)
sub = df[mask]
on = sub.loc[sub.treatment_cetuximab == 1, "pfs_months"].mean()
off = sub.loc[sub.treatment_cetuximab == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(sub.loc[sub.treatment_cetuximab == 1, "pfs_months"], sub.loc[sub.treatment_cetuximab == 0, "pfs_months"])
record("final_cetux_subgroup", f"Cetuximab in KRAS WT, NRAS WT, BRAF WT, left-sided (n={len(sub)}): PFS effect={on-off:+.3f} months, p={p:.2e}",
       effect=on - off, p=p)

# Pembrolizumab: MSI-high
mask = df.msi_high == 1
sub = df[mask]
on = sub.loc[sub.treatment_pembrolizumab == 1, "pfs_months"].mean()
off = sub.loc[sub.treatment_pembrolizumab == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(sub.loc[sub.treatment_pembrolizumab == 1, "pfs_months"], sub.loc[sub.treatment_pembrolizumab == 0, "pfs_months"])
record("final_pembro_subgroup", f"Pembrolizumab in MSI-high (n={len(sub)}): PFS effect={on-off:+.3f} months, p={p:.2e}",
       effect=on - off, p=p)

# Encorafenib: BRAF V600E
mask = df.braf_v600e == 1
sub = df[mask]
on = sub.loc[sub.treatment_encorafenib == 1, "pfs_months"].mean()
off = sub.loc[sub.treatment_encorafenib == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(sub.loc[sub.treatment_encorafenib == 1, "pfs_months"], sub.loc[sub.treatment_encorafenib == 0, "pfs_months"])
record("final_enco_subgroup", f"Encorafenib in BRAF V600E (n={len(sub)}): PFS effect={on-off:+.3f} months, p={p:.2e}",
       effect=on - off, p=p)

# HER2 tx: HER2 amplified (and possibly RAS WT)
mask = (df.her2_amplified == 1) & (df.kras_mutation == 0) & (df.nras_mutation == 0)
sub = df[mask]
on = sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"].mean()
off = sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"], sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"])
record("final_her2tx_subgroup", f"Trastuzumab/tucatinib in HER2+ AND RAS WT (n={len(sub)}): PFS effect={on-off:+.3f} months, p={p:.2e}",
       effect=on - off, p=p)

# Bevacizumab subgroup (try most common: any patient, or right-sided)
m_full2 = smf.ols("pfs_months ~ treatment_bevacizumab + kras_mutation + nras_mutation + braf_v600e + right_sided_primary + ecog_ps + stage_iv + age_years + albumin_g_dl + log_cea + ldh_u_l + nlr + crp_mg_l + weight_loss_pct_6mo", data=df).fit()
record("final_bev_adjusted", f"Adjusted OLS bevacizumab main effect: beta={m_full2.params['treatment_bevacizumab']:.3f}, p={m_full2.pvalues['treatment_bevacizumab']:.2e}",
       effect=m_full2.params["treatment_bevacizumab"], p=m_full2.pvalues["treatment_bevacizumab"])

# Regorafenib in all and in good ECOG (ecog 0)
sub = df[df.ecog_ps == 0]
on = sub.loc[sub.treatment_regorafenib == 1, "pfs_months"].mean()
off = sub.loc[sub.treatment_regorafenib == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(sub.loc[sub.treatment_regorafenib == 1, "pfs_months"], sub.loc[sub.treatment_regorafenib == 0, "pfs_months"])
record("final_rego_ecog0", f"Regorafenib in ECOG 0 (n={len(sub)}): PFS effect={on-off:+.3f} months, p={p:.2e}",
       effect=on - off, p=p)

# ---------------- Iteration 16: Three-way interactions for cetuximab ---------------- #
# Test if cetuximab needs both biomarkers (e.g., KRAS WT + left-sided)
# Stratified across the 4 cells
for k in [0, 1]:
    for r in [0, 1]:
        sub = df[(df.kras_mutation == k) & (df.right_sided_primary == r)]
        on = sub.loc[sub.treatment_cetuximab == 1, "pfs_months"].mean()
        off = sub.loc[sub.treatment_cetuximab == 0, "pfs_months"].mean()
        if sub.treatment_cetuximab.sum() > 1 and (sub.treatment_cetuximab == 0).sum() > 1:
            t, p = stats.ttest_ind(sub.loc[sub.treatment_cetuximab == 1, "pfs_months"], sub.loc[sub.treatment_cetuximab == 0, "pfs_months"])
        else:
            p = np.nan
        record(f"cetux_kras{k}_side{r}", f"Cetuximab effect in KRAS={k}, right_sided={r} (n={len(sub)}): {on-off:+.3f} months, p={p:.2e}",
               effect=on - off, p=p)

# ---------------- Iteration 17: BRAF V600E impact on cetuximab (often resistance) ---------------- #
# In the full RAS WT subgroup, examine BRAF effect on cetuximab
sub = df[(df.kras_mutation == 0) & (df.nras_mutation == 0)]
m = smf.ols("pfs_months ~ treatment_cetuximab * braf_v600e", data=sub).fit()
record("cetux_braf_in_rasWT", f"In RAS WT: cetuximab*BRAF interaction beta={m.params['treatment_cetuximab:braf_v600e']:.3f}, p={m.pvalues['treatment_cetuximab:braf_v600e']:.2e}",
       effect=m.params["treatment_cetuximab:braf_v600e"], p=m.pvalues["treatment_cetuximab:braf_v600e"])

# ---------------- Iteration 18: Encorafenib subgroup refinement ---------------- #
# Check if encorafenib effect in BRAF V600E is uniform across RAS, sidedness
for k in [0, 1]:
    sub = df[(df.braf_v600e == 1) & (df.kras_mutation == k)]
    on = sub.loc[sub.treatment_encorafenib == 1, "pfs_months"].mean()
    off = sub.loc[sub.treatment_encorafenib == 0, "pfs_months"].mean()
    if sub.treatment_encorafenib.sum() > 1 and (sub.treatment_encorafenib == 0).sum() > 1:
        t, p = stats.ttest_ind(sub.loc[sub.treatment_encorafenib == 1, "pfs_months"], sub.loc[sub.treatment_encorafenib == 0, "pfs_months"])
    else:
        p = np.nan
    record(f"enco_braf1_kras{k}", f"Encorafenib in BRAF+ AND KRAS={k} (n={len(sub)}): {on-off:+.3f} months, p={p:.2e}",
           effect=on - off, p=p)

# ---------------- Iteration 19: Pembrolizumab heterogeneity (within MSI-high test other modifiers) ---------------- #
sub_msi = df[df.msi_high == 1]
for bm in ["right_sided_primary", "kras_mutation", "stage_iv", "ecog_ps"]:
    try:
        m = smf.ols(f"pfs_months ~ treatment_pembrolizumab * {bm}", data=sub_msi).fit()
        key_b = f"treatment_pembrolizumab:{bm}"
        record(f"pembro_{bm}_in_msi", f"In MSI-high: pembro*{bm} interaction beta={m.params[key_b]:.3f}, p={m.pvalues[key_b]:.2e}",
               effect=m.params[key_b], p=m.pvalues[key_b])
    except Exception:
        pass

# ---------------- Iteration 20: HER2 tx subgroup refinement ---------------- #
# In HER2+ check if also need BRAF WT
mask = (df.her2_amplified == 1) & (df.kras_mutation == 0) & (df.nras_mutation == 0) & (df.braf_v600e == 0)
sub = df[mask]
on = sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"].mean()
off = sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"].mean()
t, p = stats.ttest_ind(sub.loc[sub.treatment_trastuzumab_tucatinib == 1, "pfs_months"], sub.loc[sub.treatment_trastuzumab_tucatinib == 0, "pfs_months"])
record("final_her2tx_full_wt", f"Trastuzumab/tucatinib in HER2+ AND KRAS WT AND NRAS WT AND BRAF WT (n={len(sub)}): PFS effect={on-off:+.3f} months, p={p:.2e}",
       effect=on - off, p=p)

# ---------------- Iteration 21: Continuous lab thresholds and treatment interactions ---------------- #
# Bevacizumab x continuous
for bm in ["log_cea", "albumin_g_dl", "ldh_u_l", "ecog_ps", "age_years"]:
    m = smf.ols(f"pfs_months ~ treatment_bevacizumab * {bm}", data=df).fit()
    key_b = f"treatment_bevacizumab:{bm}"
    record(f"bev_{bm}_interaction", f"OLS bev*{bm} interaction beta={m.params[key_b]:.4f}, p={m.pvalues[key_b]:.2e}",
           effect=m.params[key_b], p=m.pvalues[key_b])

# ---------------- Iteration 22: Regorafenib in mutated/right-sided/poor prognosis (third+ line context) ---------------- #
# Test in BRAF MUT, KRAS MUT
for k in [0, 1]:
    sub = df[df.kras_mutation == k]
    on = sub.loc[sub.treatment_regorafenib == 1, "pfs_months"].mean()
    off = sub.loc[sub.treatment_regorafenib == 0, "pfs_months"].mean()
    t, p = stats.ttest_ind(sub.loc[sub.treatment_regorafenib == 1, "pfs_months"], sub.loc[sub.treatment_regorafenib == 0, "pfs_months"])
    record(f"rego_kras{k}", f"Regorafenib in KRAS={k} (n={len(sub)}): {on-off:+.3f} months, p={p:.2e}",
           effect=on - off, p=p)

# ---------------- Iteration 23: NTRK fusion ---------------- #
# Check if NTRK fusion alters anything (no NTRK treatment in dataset, but maybe pembrolizumab works?)
sub = df[df.ntrk_fusion == 1]
print("NTRK fusion n=", len(sub))
record("ntrk_count", f"Patients with NTRK fusion: n={len(sub)}; mean PFS={sub.pfs_months.mean():.3f} vs no NTRK={df.loc[df.ntrk_fusion==0,'pfs_months'].mean():.3f}",
       effect=sub.pfs_months.mean() - df.loc[df.ntrk_fusion == 0, "pfs_months"].mean(),
       p=stats.ttest_ind(sub.pfs_months, df.loc[df.ntrk_fusion == 0, "pfs_months"]).pvalue)

# ---------------- Iteration 24: Adjusted treatment-by-biomarker interactions in full multivariable model ---------------- #
# Confirm key interactions in adjusted model
formula2 = ("pfs_months ~ age_years + sex_female + ecog_ps + stage_iv + right_sided_primary "
            "+ kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified "
            "+ log_cea + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr + hemoglobin_g_dl "
            "+ treatment_cetuximab*kras_mutation "
            "+ treatment_cetuximab*nras_mutation "
            "+ treatment_cetuximab*braf_v600e "
            "+ treatment_cetuximab*right_sided_primary "
            "+ treatment_pembrolizumab*msi_high "
            "+ treatment_encorafenib*braf_v600e "
            "+ treatment_trastuzumab_tucatinib*her2_amplified "
            "+ treatment_bevacizumab + treatment_regorafenib")
m_adj = smf.ols(formula2, data=df).fit()
for k in ["treatment_cetuximab:kras_mutation", "treatment_cetuximab:nras_mutation",
          "treatment_cetuximab:braf_v600e", "treatment_cetuximab:right_sided_primary",
          "treatment_pembrolizumab:msi_high", "treatment_encorafenib:braf_v600e",
          "treatment_trastuzumab_tucatinib:her2_amplified"]:
    record(f"adj_{k.replace(':','_x_')}",
           f"Adjusted interaction {k}: beta={m_adj.params[k]:.3f}, p={m_adj.pvalues[k]:.2e}",
           effect=m_adj.params[k], p=m_adj.pvalues[k])

# ---------------- Iteration 25: Final consolidated subgroup summaries ---------------- #
# Best subgroup definitions for each treatment, and complement to demonstrate suppression
defs = {
    "cetuximab": dict(
        on=(df.kras_mutation == 0) & (df.nras_mutation == 0) & (df.braf_v600e == 0) & (df.right_sided_primary == 0),
        off=(df.kras_mutation == 0) & (df.nras_mutation == 0) & (df.braf_v600e == 0) & (df.right_sided_primary == 1),
        tx="treatment_cetuximab",
    ),
}
for name, d in defs.items():
    sub = df[d["on"]]
    on_eff = sub.loc[sub[d["tx"]] == 1, "pfs_months"].mean() - sub.loc[sub[d["tx"]] == 0, "pfs_months"].mean()
    sub2 = df[d["off"]]
    off_eff = sub2.loc[sub2[d["tx"]] == 1, "pfs_months"].mean() - sub2.loc[sub2[d["tx"]] == 0, "pfs_months"].mean()
    record(f"complement_{name}_left", f"Cetuximab in RAS/RAF WT LEFT-sided n={len(sub)}: effect={on_eff:+.3f}", effect=on_eff)
    record(f"complement_{name}_right", f"Cetuximab in RAS/RAF WT RIGHT-sided n={len(sub2)}: effect={off_eff:+.3f}", effect=off_eff)

# Save all results
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to results.json")
