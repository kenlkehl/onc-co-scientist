"""Run all analyses for the CRC dataset and persist results to results.json.

Uses ordinary regression / t-tests / chi-square — fast and explicit.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_parquet("dataset.parquet")
Y = "pfs_months"

results = {}


def store(key, *, p, eff, summary, sig=None):
    if sig is None and p is not None:
        sig = bool(p < 0.05)
    results[key] = {
        "p_value": None if p is None else float(p),
        "effect_estimate": None if eff is None else float(eff),
        "result_summary": summary,
        "significant": sig,
    }


def tt(col_bin, label_on="1", label_off="0"):
    a = df.loc[df[col_bin] == 1, Y].values
    b = df.loc[df[col_bin] == 0, Y].values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    eff = a.mean() - b.mean()
    return p, eff, f"PFS mean {a.mean():.2f} (n={len(a)}) vs {b.mean():.2f} (n={len(b)}); diff={eff:+.3f} mo, t-test p={p:.3g}"


def lr_main(col):
    X = sm.add_constant(df[col].astype(float))
    m = sm.OLS(df[Y], X).fit()
    eff = float(m.params[col])
    p = float(m.pvalues[col])
    return p, eff, f"OLS PFS ~ {col}: beta={eff:+.4f} per unit, p={p:.3g}"


def lr_inter(treat, marker):
    """OLS PFS ~ treat + marker + treat:marker. Return interaction term."""
    sub = df[[Y, treat, marker]].copy()
    sub["x"] = sub[treat] * sub[marker]
    X = sm.add_constant(sub[[treat, marker, "x"]].astype(float))
    m = sm.OLS(sub[Y], X).fit()
    p = float(m.pvalues["x"])
    eff = float(m.params["x"])
    # Also compute conditional means for transparency
    mm = sub.groupby([marker, treat])[Y].agg(["mean", "count"])
    return p, eff, f"PFS ~ {treat} + {marker} + interaction: interaction beta={eff:+.4f}, p={p:.3g}\nGroup means:\n{mm.to_string()}"


# === Iteration 1: prognostic main effects ===
for col in ["stage_iv", "liver_mets", "bone_mets", "ecog_ps"]:
    if col == "ecog_ps":
        p, e, s = lr_main(col)
    else:
        p, e, s = tt(col)
    store(f"i1_{col}", p=p, eff=e, summary=s)

# === Iteration 2: lab-based prognostic features ===
for col in ["albumin_g_dl", "ldh_u_l", "crp_mg_l", "nlr", "weight_loss_pct_6mo", "cea_ng_ml", "hemoglobin_g_dl"]:
    p, e, s = lr_main(col)
    store(f"i2_{col}", p=p, eff=e, summary=s)

# === Iteration 3: biomarker prevalence and main effects ===
for col in ["kras_mutation", "nras_mutation", "braf_v600e", "msi_high", "her2_amplified", "ntrk_fusion", "right_sided_primary"]:
    p, e, s = tt(col)
    store(f"i3_{col}", p=p, eff=e, summary=s)

# === Iteration 4: treatment main effects (raw, unadjusted) ===
for t in ["treatment_cetuximab", "treatment_bevacizumab", "treatment_pembrolizumab",
          "treatment_encorafenib", "treatment_trastuzumab_tucatinib", "treatment_regorafenib"]:
    p, e, s = tt(t)
    store(f"i4_{t}", p=p, eff=e, summary=s)

# === Iteration 5: classic CRC biomarker x treatment interactions ===
# 5a Cetuximab benefits RAS-wild-type only
p, e, s = lr_inter("treatment_cetuximab", "kras_mutation")
store("i5_cetux_x_kras", p=p, eff=e, summary=s)
p, e, s = lr_inter("treatment_cetuximab", "nras_mutation")
store("i5_cetux_x_nras", p=p, eff=e, summary=s)
p, e, s = lr_inter("treatment_cetuximab", "braf_v600e")
store("i5_cetux_x_braf", p=p, eff=e, summary=s)

# 5b Pembrolizumab in MSI-high
p, e, s = lr_inter("treatment_pembrolizumab", "msi_high")
store("i5_pembro_x_msi", p=p, eff=e, summary=s)

# 5c Encorafenib in BRAF V600E
p, e, s = lr_inter("treatment_encorafenib", "braf_v600e")
store("i5_enco_x_braf", p=p, eff=e, summary=s)

# 5d Trastuzumab+tucatinib in HER2 amplified
p, e, s = lr_inter("treatment_trastuzumab_tucatinib", "her2_amplified")
store("i5_tt_x_her2", p=p, eff=e, summary=s)

# === Iteration 6: stratified PFS in indication-relevant subgroups ===
def strat(treat, marker, value):
    sub = df[df[marker] == value]
    a = sub.loc[sub[treat] == 1, Y]
    b = sub.loc[sub[treat] == 0, Y]
    if len(a) < 5 or len(b) < 5:
        return None, None, f"too few patients to compare (n_treated={len(a)}, n_untreated={len(b)})"
    t, p = stats.ttest_ind(a, b, equal_var=False)
    eff = a.mean() - b.mean()
    return float(p), float(eff), (
        f"In {marker}={value} (n={len(sub)}): PFS {a.mean():.2f} on {treat} (n={len(a)}) "
        f"vs {b.mean():.2f} off (n={len(b)}); diff={eff:+.3f} mo, p={p:.3g}"
    )

for m, v, t in [
    ("kras_mutation", 0, "treatment_cetuximab"),
    ("kras_mutation", 1, "treatment_cetuximab"),
    ("nras_mutation", 0, "treatment_cetuximab"),
    ("nras_mutation", 1, "treatment_cetuximab"),
    ("braf_v600e", 0, "treatment_cetuximab"),
    ("braf_v600e", 1, "treatment_cetuximab"),
    ("msi_high", 1, "treatment_pembrolizumab"),
    ("msi_high", 0, "treatment_pembrolizumab"),
    ("braf_v600e", 1, "treatment_encorafenib"),
    ("braf_v600e", 0, "treatment_encorafenib"),
    ("her2_amplified", 1, "treatment_trastuzumab_tucatinib"),
    ("her2_amplified", 0, "treatment_trastuzumab_tucatinib"),
]:
    p, e, s = strat(t, m, v)
    store(f"i6_{t}_in_{m}={v}", p=p, eff=e, summary=s)

# === Iteration 7: comorbidity / symptom main effects ===
for col in ["fatigue_grade", "pain_nrs", "dyspnea_grade", "appetite_loss_grade", "cough_grade",
            "diabetes_mellitus", "hypertension", "copd", "chronic_kidney_disease",
            "heart_failure", "coronary_artery_disease", "atrial_fibrillation", "venous_thromboembolism_history"]:
    if df[col].nunique() <= 2:
        p, e, s = tt(col)
    else:
        p, e, s = lr_main(col)
    store(f"i7_{col}", p=p, eff=e, summary=s)

# === Iteration 8: vitals & demographics ===
for col in ["age_years", "sex_female", "rural_residence", "smoking_pack_years",
            "bmi", "systolic_bp_mmhg", "heart_rate_bpm", "spo2_pct",
            "education_years"]:
    if df[col].nunique() <= 2:
        p, e, s = tt(col)
    else:
        p, e, s = lr_main(col)
    store(f"i8_{col}", p=p, eff=e, summary=s)

# === Iteration 9: race/ethnicity & insurance — ANOVA ===
for col in ["race_ethnicity", "insurance_type"]:
    groups = [df.loc[df[col] == g, Y].values for g in df[col].unique()]
    f, p = stats.f_oneway(*groups)
    means = df.groupby(col)[Y].mean().to_dict()
    eff_max = max(means.values()) - min(means.values())  # range of group means
    store(f"i9_{col}", p=float(p), eff=float(eff_max),
          summary=f"ANOVA across {col}: F={f:.2f}, p={p:.3g}; group means={means}")

# === Iteration 10: prior treatment / disease history ===
for col in ["prior_chemotherapy", "prior_radiation", "prior_surgery", "prior_immunotherapy",
            "prior_targeted_therapy", "prior_lines_of_therapy", "years_since_diagnosis",
            "prior_malignancy"]:
    if df[col].nunique() <= 2:
        p, e, s = tt(col)
    else:
        p, e, s = lr_main(col)
    store(f"i10_{col}", p=p, eff=e, summary=s)

# === Iteration 11: covariate-adjusted treatment effects ===
# Adjust for age, ECOG, stage, albumin, LDH; check whether main effects survive
adj_cov = ["age_years", "ecog_ps", "stage_iv", "albumin_g_dl", "ldh_u_l", "liver_mets"]

def adj_lr(treat):
    X = df[[treat] + adj_cov].astype(float)
    X = sm.add_constant(X)
    m = sm.OLS(df[Y], X).fit()
    return float(m.pvalues[treat]), float(m.params[treat]), \
        f"OLS PFS ~ {treat} + age + ECOG + stage_iv + albumin + LDH + liver_mets: beta={m.params[treat]:+.4f}, p={m.pvalues[treat]:.3g}"

for t in ["treatment_cetuximab", "treatment_bevacizumab", "treatment_pembrolizumab",
          "treatment_encorafenib", "treatment_trastuzumab_tucatinib", "treatment_regorafenib"]:
    p, e, s = adj_lr(t)
    store(f"i11_adj_{t}", p=p, eff=e, summary=s)

# === Iteration 12: laterality x cetuximab interaction ===
p, e, s = lr_inter("treatment_cetuximab", "right_sided_primary")
store("i12_cetux_x_rightsided", p=p, eff=e, summary=s)

# Also: in right-sided primary, KRAS-WT subgroup, does cetuximab still help?
sub = df[(df["right_sided_primary"] == 1) & (df["kras_mutation"] == 0) & (df["nras_mutation"] == 0)]
a = sub.loc[sub["treatment_cetuximab"] == 1, Y]
b = sub.loc[sub["treatment_cetuximab"] == 0, Y]
if len(a) >= 5 and len(b) >= 5:
    t, p = stats.ttest_ind(a, b, equal_var=False)
    store("i12_cetux_in_rightRASwt",
          p=float(p), eff=float(a.mean() - b.mean()),
          summary=f"Right-sided RAS-WT: PFS {a.mean():.2f} on cetux (n={len(a)}) vs {b.mean():.2f} off (n={len(b)}), diff={a.mean()-b.mean():+.3f}, p={p:.3g}")

sub = df[(df["right_sided_primary"] == 0) & (df["kras_mutation"] == 0) & (df["nras_mutation"] == 0)]
a = sub.loc[sub["treatment_cetuximab"] == 1, Y]
b = sub.loc[sub["treatment_cetuximab"] == 0, Y]
if len(a) >= 5 and len(b) >= 5:
    t, p = stats.ttest_ind(a, b, equal_var=False)
    store("i12_cetux_in_leftRASwt",
          p=float(p), eff=float(a.mean() - b.mean()),
          summary=f"Left-sided RAS-WT: PFS {a.mean():.2f} on cetux (n={len(a)}) vs {b.mean():.2f} off (n={len(b)}), diff={a.mean()-b.mean():+.3f}, p={p:.3g}")

# === Iteration 13: triple-marker subgroup analyses ===
# Cetuximab in KRAS+NRAS+BRAF wild-type only
sub = df[(df["kras_mutation"] == 0) & (df["nras_mutation"] == 0) & (df["braf_v600e"] == 0)]
a = sub.loc[sub["treatment_cetuximab"] == 1, Y]
b = sub.loc[sub["treatment_cetuximab"] == 0, Y]
t, p = stats.ttest_ind(a, b, equal_var=False)
store("i13_cetux_in_tripleWT",
      p=float(p), eff=float(a.mean() - b.mean()),
      summary=f"In KRAS/NRAS/BRAF-WT (n={len(sub)}): PFS {a.mean():.2f} on cetux (n={len(a)}) vs {b.mean():.2f} off (n={len(b)}); diff={a.mean()-b.mean():+.3f}, p={p:.3g}")

# === Iteration 14: SNP scan — main effects ===
snp_cols = [c for c in df.columns if c.startswith("snp_")]
snp_results = []
for c in snp_cols:
    p, e, s = lr_main(c)
    snp_results.append((c, p, e))
    store(f"i14_{c}", p=p, eff=e, summary=s)

# === Iteration 15: BRAF + cetuximab + encorafenib triple-test ===
# Encorafenib alone in BRAF V600E vs cetuximab+encorafenib combo
sub = df[df["braf_v600e"] == 1]
# group by enc/cetux combinations
groups = sub.groupby(["treatment_encorafenib", "treatment_cetuximab"])[Y].agg(["mean", "count"])
store("i15_braf_combo_table",
      p=None, eff=None,
      summary=f"BRAF V600E PFS by treatment combo:\n{groups.to_string()}")

# Three-way interaction enc × cetux in BRAF V600E
sub2 = sub.copy()
sub2["enc_cetux"] = sub2["treatment_encorafenib"] * sub2["treatment_cetuximab"]
X = sm.add_constant(sub2[["treatment_encorafenib", "treatment_cetuximab", "enc_cetux"]].astype(float))
m = sm.OLS(sub2[Y], X).fit()
store("i15_enc_x_cetux_in_braf",
      p=float(m.pvalues["enc_cetux"]),
      eff=float(m.params["enc_cetux"]),
      summary=f"In BRAF V600E (n={len(sub)}): interaction beta={m.params['enc_cetux']:+.3f}, p={m.pvalues['enc_cetux']:.3g}")

# === Iteration 16: bevacizumab effect by laterality and KRAS ===
p, e, s = lr_inter("treatment_bevacizumab", "right_sided_primary")
store("i16_bev_x_rightsided", p=p, eff=e, summary=s)
p, e, s = lr_inter("treatment_bevacizumab", "kras_mutation")
store("i16_bev_x_kras", p=p, eff=e, summary=s)

# === Iteration 17: regorafenib effects (typically late-line; check by prior_lines_of_therapy) ===
p, e, s = lr_inter("treatment_regorafenib", "prior_lines_of_therapy")
store("i17_rego_x_priorlines", p=p, eff=e, summary=s)

# === Iteration 18: pembrolizumab in MSI-high subgroup, also msi-low ===
# Already have i6 results; check pembrolizumab x msi_high more deeply with adjustments
sub = df[df["msi_high"] == 1]
X = sub[["treatment_pembrolizumab", "age_years", "ecog_ps", "stage_iv", "albumin_g_dl", "ldh_u_l"]].astype(float)
X = sm.add_constant(X)
m = sm.OLS(sub[Y], X).fit()
store("i18_pembro_in_msi_adj",
      p=float(m.pvalues["treatment_pembrolizumab"]),
      eff=float(m.params["treatment_pembrolizumab"]),
      summary=f"Adjusted PFS ~ pembro + covariates in MSI-high (n={len(sub)}): beta={m.params['treatment_pembrolizumab']:+.3f}, p={m.pvalues['treatment_pembrolizumab']:.3g}")

# === Iteration 19: HER2-amp subgroup with trastuzumab+tucatinib, adjusted ===
sub = df[df["her2_amplified"] == 1]
X = sub[["treatment_trastuzumab_tucatinib", "age_years", "ecog_ps", "stage_iv", "albumin_g_dl", "ldh_u_l"]].astype(float)
X = sm.add_constant(X)
m = sm.OLS(sub[Y], X).fit()
store("i19_tt_in_her2_adj",
      p=float(m.pvalues["treatment_trastuzumab_tucatinib"]),
      eff=float(m.params["treatment_trastuzumab_tucatinib"]),
      summary=f"Adjusted PFS ~ trastuzumab_tucatinib + covariates in HER2-amplified (n={len(sub)}): beta={m.params['treatment_trastuzumab_tucatinib']:+.3f}, p={m.pvalues['treatment_trastuzumab_tucatinib']:.3g}")

# === Iteration 20: race/ethnicity & insurance effects on access to targeted Tx ===
# Compare proportion of pembrolizumab use among MSI-high by insurance_type
sub = df[df["msi_high"] == 1]
ct = pd.crosstab(sub["insurance_type"], sub["treatment_pembrolizumab"])
ch2, p, dof, exp = stats.chi2_contingency(ct)
prop = ct.apply(lambda r: r[1] / r.sum(), axis=1).to_dict()
store("i20_pembro_access_by_insurance",
      p=float(p), eff=float(max(prop.values()) - min(prop.values())),
      summary=f"Among MSI-high (n={len(sub)}), pembrolizumab use by insurance: {prop}; chi2 p={p:.3g}")

# === Iteration 21: BRAF subgroup analysis of encorafenib + cetuximab combo more carefully ===
sub = df[df["braf_v600e"] == 1]
combo = (sub["treatment_encorafenib"] == 1) & (sub["treatment_cetuximab"] == 1)
mono_enc = (sub["treatment_encorafenib"] == 1) & (sub["treatment_cetuximab"] == 0)
neither = (sub["treatment_encorafenib"] == 0) & (sub["treatment_cetuximab"] == 0)
res = {
    "combo_mean": float(sub.loc[combo, Y].mean()), "combo_n": int(combo.sum()),
    "mono_enc_mean": float(sub.loc[mono_enc, Y].mean()), "mono_enc_n": int(mono_enc.sum()),
    "neither_mean": float(sub.loc[neither, Y].mean()), "neither_n": int(neither.sum()),
}
# t-test combo vs neither
a = sub.loc[combo, Y]
b = sub.loc[neither, Y]
t, p = stats.ttest_ind(a, b, equal_var=False)
store("i21_braf_combo_vs_neither",
      p=float(p), eff=float(a.mean() - b.mean()),
      summary=f"BRAF V600E: combo encora+cetux PFS {a.mean():.2f} (n={len(a)}) vs neither {b.mean():.2f} (n={len(b)}); diff={a.mean()-b.mean():+.3f}, p={p:.3g}; full table: {res}")

# === Iteration 22: ECOG x stage interaction ===
p, e, s = lr_inter("ecog_ps", "stage_iv")
store("i22_ecog_x_stageiv", p=p, eff=e, summary=s)

# === Iteration 23: composite prognostic model ===
covs = ["age_years", "sex_female", "ecog_ps", "stage_iv", "right_sided_primary",
        "liver_mets", "bone_mets", "albumin_g_dl", "ldh_u_l", "crp_mg_l", "nlr",
        "weight_loss_pct_6mo", "kras_mutation", "nras_mutation", "braf_v600e",
        "msi_high", "her2_amplified", "ntrk_fusion",
        "treatment_cetuximab", "treatment_bevacizumab", "treatment_pembrolizumab",
        "treatment_encorafenib", "treatment_trastuzumab_tucatinib", "treatment_regorafenib"]
X = df[covs].astype(float)
X = sm.add_constant(X)
m = sm.OLS(df[Y], X).fit()
top = m.pvalues.sort_values().head(20).to_dict()
betas = m.params[list(top.keys())].to_dict()
results["i23_full_OLS"] = {
    "p_value": None, "effect_estimate": None, "significant": None,
    "result_summary": "Multivariable OLS PFS ~ comprehensive covariates. Top significant terms (sorted by p):\n" +
        "\n".join([f"  {k}: beta={betas[k]:+.4f}, p={top[k]:.3g}" for k in top]) +
        f"\n  R^2 = {m.rsquared:.4f}, n={int(m.nobs)}",
}

# === Iteration 24: targeted-treatment delivery audit ===
# Are biomarker-targeted treatments preferentially given to biomarker-positive patients?
def access_audit(treat, marker):
    p_marker_pos = df.loc[df[marker] == 1, treat].mean()
    p_marker_neg = df.loc[df[marker] == 0, treat].mean()
    ct = pd.crosstab(df[marker], df[treat])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return p_marker_pos, p_marker_neg, p, chi2

for treat, marker in [
    ("treatment_pembrolizumab", "msi_high"),
    ("treatment_encorafenib", "braf_v600e"),
    ("treatment_trastuzumab_tucatinib", "her2_amplified"),
    ("treatment_cetuximab", "kras_mutation"),
]:
    pp, pn, p, chi2 = access_audit(treat, marker)
    store(f"i24_audit_{treat}_in_{marker}",
          p=float(p), eff=float(pp - pn),
          summary=f"{treat} use: {pp*100:.1f}% in {marker}=1 vs {pn*100:.1f}% in {marker}=0; chi2 p={p:.3g}")

# === Iteration 25: treatment x msi_high triple-cell, pembro_x_msi adjusted further ===
p, e, s = lr_inter("treatment_pembrolizumab", "msi_high")
store("i25_pembro_msi_recheck", p=p, eff=e, summary=s)

# Prior immunotherapy effect on pembrolizumab response in MSI-high
sub = df[df["msi_high"] == 1]
X = sub[["treatment_pembrolizumab", "prior_immunotherapy"]].astype(float)
X["x"] = X["treatment_pembrolizumab"] * X["prior_immunotherapy"]
X = sm.add_constant(X)
m = sm.OLS(sub[Y], X).fit()
store("i25_pembro_x_priorimmuno_in_msi",
      p=float(m.pvalues["x"]),
      eff=float(m.params["x"]),
      summary=f"In MSI-high: pembro × prior_immuno interaction beta={m.params['x']:+.3f}, p={m.pvalues['x']:.3g}")

# Save
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Wrote", len(results), "results to results.json")
print("Significant (p<0.05):", sum(1 for r in results.values() if r["significant"]))
