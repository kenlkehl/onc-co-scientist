"""Iterative analysis of ds001_crc dataset."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet("dataset.parquet")
y = df["pfs_months"].values

OUT = {}

def add(name, payload):
    OUT[name] = payload
    print(f"== {name} ==")
    print(payload)
    print()

# ---- Iter 1: Universal screen ----
screen = []
for col in df.columns:
    if col in ("patient_id", "pfs_months"):
        continue
    s = df[col]
    if s.dtype == object or s.nunique() <= 6:
        groups = [df.loc[s == v, "pfs_months"].values for v in s.unique() if (s == v).sum() > 5]
        if len(groups) < 2:
            continue
        f, p = stats.f_oneway(*groups)
        means = {str(v): float(df.loc[s == v, "pfs_months"].mean()) for v in s.unique()}
        screen.append({"col": col, "kind": "cat", "stat": float(f), "p": float(p), "means": means, "nunique": int(s.nunique())})
    else:
        r, p = stats.pearsonr(s, df["pfs_months"])
        screen.append({"col": col, "kind": "cont", "stat": float(r), "p": float(p), "nunique": int(s.nunique())})
screen.sort(key=lambda r: r["p"])
add("screen_top20", screen[:20])

# ---- Iter 2: feature_078 (age) ----
df["q078"] = pd.qcut(df["feature_078"], 4, labels=False)
mod78 = smf.ols("pfs_months ~ feature_078", data=df).fit()
add("feature_078_ols", {"slope": float(mod78.params["feature_078"]),
                         "p": float(mod78.pvalues["feature_078"]),
                         "r2": float(mod78.rsquared),
                         "by_quartile": df.groupby("q078")["pfs_months"].mean().to_dict()})

# ---- Iter 3: feature_057 ordinal ----
mod57 = smf.ols("pfs_months ~ C(feature_057)", data=df).fit()
gr57 = df.groupby("feature_057")["pfs_months"].agg(["count", "mean"]).to_dict()
# Linear trend via codes
mod57tr = smf.ols("pfs_months ~ feature_057", data=df).fit()
add("feature_057_anova", {"by_level": gr57,
                            "trend_slope": float(mod57tr.params["feature_057"]),
                            "trend_p": float(mod57tr.pvalues["feature_057"]),
                            "f_test_p": float(mod57.f_pvalue)})

# ---- Iter 4: top binary effects (with t-test direction) ----
top_bin = ["feature_051", "feature_038", "feature_013", "feature_043", "feature_109", "feature_067"]
bin_results = {}
for c in top_bin:
    g0 = df.loc[df[c] == 0, "pfs_months"]
    g1 = df.loc[df[c] == 1, "pfs_months"]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    bin_results[c] = {"mean_1": float(g1.mean()), "mean_0": float(g0.mean()),
                       "diff_1_minus_0": float(g1.mean() - g0.mean()),
                       "p": float(p), "n1": int(len(g1)), "n0": int(len(g0))}
add("top_binary_effects", bin_results)

# ---- Iter 5: continuous predictors after age ----
cont_cands = ["feature_099", "feature_092", "feature_009", "feature_055", "feature_006", "feature_070"]
cont_adj = {}
for c in cont_cands:
    m = smf.ols(f"pfs_months ~ feature_078 + {c}", data=df).fit()
    cont_adj[c] = {"slope_adj": float(m.params[c]), "p_adj": float(m.pvalues[c]),
                    "slope_unadj": float(stats.pearsonr(df[c], df["pfs_months"])[0])}
add("continuous_adjusted_for_age", cont_adj)

# ---- Iter 6: race & insurance ----
race_grp = df.groupby("feature_064")["pfs_months"].agg(["count", "mean", "std"]).to_dict()
ins_grp = df.groupby("feature_018")["pfs_months"].agg(["count", "mean", "std"]).to_dict()
mod_race = smf.ols("pfs_months ~ C(feature_064, Treatment(reference='white'))", data=df).fit()
mod_ins = smf.ols("pfs_months ~ C(feature_018, Treatment(reference='private'))", data=df).fit()
race_adj = smf.ols("pfs_months ~ C(feature_064, Treatment(reference='white')) + feature_078 + C(feature_057) + feature_051 + feature_038", data=df).fit()
ins_adj = smf.ols("pfs_months ~ C(feature_018, Treatment(reference='private')) + feature_078 + C(feature_057) + feature_051 + feature_038", data=df).fit()
add("race_insurance", {
    "race_unadj_means": race_grp["mean"], "race_unadj_p": float(mod_race.f_pvalue),
    "ins_unadj_means": ins_grp["mean"], "ins_unadj_p": float(mod_ins.f_pvalue),
    "race_adj_p": float(race_adj.f_test([f"C(feature_064, Treatment(reference='white'))[T.{r}] = 0" for r in ['asian','black','hispanic','other']]).pvalue),
    "ins_adj_p": float(ins_adj.f_test([f"C(feature_018, Treatment(reference='private'))[T.{r}] = 0" for r in ['medicaid','medicare','uninsured']]).pvalue),
    "race_adj_coefs": {k: float(v) for k, v in race_adj.params.items() if 'feature_064' in k},
    "ins_adj_coefs": {k: float(v) for k, v in ins_adj.params.items() if 'feature_018' in k},
})

# ---- Iter 7: multivariable model w/ top predictors ----
mv = smf.ols(
    "pfs_months ~ feature_078 + C(feature_057) + feature_051 + feature_038 + feature_013 + feature_043 "
    "+ feature_109 + feature_067 + feature_099 + feature_092 + feature_009",
    data=df,
).fit()
add("multivariable", {"r2": float(mv.rsquared),
                        "params": {k: float(v) for k, v in mv.params.items()},
                        "pvalues": {k: float(v) for k, v in mv.pvalues.items()}})

# ---- Iter 8: identify clusters (mutually-exclusive treatment indicators?) ----
# Look for binary cols with prevalence and pairwise overlap
bin_cols = [c for c in df.columns if df[c].dtype in (np.int64, int) and df[c].nunique() == 2 and c != "patient_id"]
prev = pd.Series({c: df[c].mean() for c in bin_cols})
mid_prev = prev[(prev > 0.05) & (prev < 0.5)].sort_values(ascending=False)
# Find features that look mutually exclusive (e.g., regimen indicators)
# Compute correlation among top mid-prevalence binaries
from itertools import combinations
mid = mid_prev.index.tolist()[:25]
corr = df[mid].corr()
neg_pairs = []
for a, b in combinations(mid, 2):
    if corr.loc[a, b] < -0.1:
        neg_pairs.append((a, b, float(corr.loc[a, b])))
neg_pairs.sort(key=lambda x: x[2])
add("mutually_exclusive_pairs_top", neg_pairs[:15])

# ---- Iter 9: Are feature_051, feature_038, feature_013, feature_043 etc related to feature_057 ----
# Look at distribution of feature_038 and feature_051 across feature_057 levels
strat = df.groupby("feature_057")[["feature_038", "feature_051", "feature_013", "feature_043", "feature_109", "feature_067"]].mean()
add("binary_by_feature_057", strat.to_dict())

# ---- Iter 10: interaction tests ----
# Test interaction: feature_038 (likely beneficial Tx) x feature_051 (poor prognostic)
inter1 = smf.ols("pfs_months ~ feature_038 * feature_051 + feature_078 + C(feature_057)", data=df).fit()
inter2 = smf.ols("pfs_months ~ feature_038 * feature_013 + feature_078 + C(feature_057)", data=df).fit()
inter3 = smf.ols("pfs_months ~ feature_038 * feature_043 + feature_078 + C(feature_057)", data=df).fit()
inter4 = smf.ols("pfs_months ~ feature_038 * feature_067 + feature_078 + C(feature_057)", data=df).fit()
inter5 = smf.ols("pfs_months ~ feature_051 * feature_013 + feature_078 + C(feature_057)", data=df).fit()
add("interaction_tests", {
    "f038_x_f051": {"coef": float(inter1.params["feature_038:feature_051"]),
                     "p": float(inter1.pvalues["feature_038:feature_051"])},
    "f038_x_f013": {"coef": float(inter2.params["feature_038:feature_013"]),
                     "p": float(inter2.pvalues["feature_038:feature_013"])},
    "f038_x_f043": {"coef": float(inter3.params["feature_038:feature_043"]),
                     "p": float(inter3.pvalues["feature_038:feature_043"])},
    "f038_x_f067": {"coef": float(inter4.params["feature_038:feature_067"]),
                     "p": float(inter4.pvalues["feature_038:feature_067"])},
    "f051_x_f013": {"coef": float(inter5.params["feature_051:feature_013"]),
                     "p": float(inter5.pvalues["feature_051:feature_013"])},
})

# ---- Iter 11: Subgroup PFS for feature_038 by feature_057 level ----
subg = df.groupby(["feature_057", "feature_038"])["pfs_months"].mean().unstack()
add("f038_by_f057", subg.to_dict())

# Also test stratified
strata_038 = {}
for lvl in [0, 1, 2]:
    s = df[df["feature_057"] == lvl]
    g0 = s.loc[s["feature_038"] == 0, "pfs_months"]
    g1 = s.loc[s["feature_038"] == 1, "pfs_months"]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    strata_038[lvl] = {"mean_1": float(g1.mean()), "mean_0": float(g0.mean()),
                        "diff": float(g1.mean() - g0.mean()), "p": float(p),
                        "n1": int(len(g1)), "n0": int(len(g0))}
add("f038_stratified_by_f057", strata_038)

# ---- Iter 12: Age x feature_038 interaction ----
inter_age038 = smf.ols("pfs_months ~ feature_078 * feature_038 + C(feature_057)", data=df).fit()
add("age_x_f038", {"interaction_coef": float(inter_age038.params["feature_078:feature_038"]),
                     "interaction_p": float(inter_age038.pvalues["feature_078:feature_038"])})

# ---- Iter 13: Insurance disparities adjusted ----
# Already done above. Let me check uninsured specifically
unins_un = smf.ols("pfs_months ~ C(feature_018, Treatment(reference='private'))", data=df).fit()
unins_adj = smf.ols("pfs_months ~ C(feature_018, Treatment(reference='private')) + feature_078 + C(feature_057) + feature_051 + feature_038 + feature_013 + feature_043 + feature_099 + feature_092", data=df).fit()
add("insurance_uninsured", {
    "unadj": {k: float(v) for k, v in unins_un.params.items() if 'uninsured' in k},
    "unadj_p": {k: float(v) for k, v in unins_un.pvalues.items() if 'uninsured' in k},
    "adj": {k: float(v) for k, v in unins_adj.params.items() if 'feature_018' in k},
    "adj_p": {k: float(v) for k, v in unins_adj.pvalues.items() if 'feature_018' in k},
})

# ---- Iter 14: Other ordinal features (feature_025, _075, _071, _026, _096, _033) ----
ord_feat_results = {}
for c in ["feature_025", "feature_075", "feature_071", "feature_026", "feature_096", "feature_033"]:
    by = df.groupby(c)["pfs_months"].mean().to_dict()
    f, p = stats.f_oneway(*[df.loc[df[c] == v, "pfs_months"].values for v in df[c].unique() if (df[c] == v).sum() > 5])
    ord_feat_results[c] = {"by_level": {str(k): float(v) for k, v in by.items()}, "anova_p": float(p)}
add("other_ordinal_features", ord_feat_results)

# ---- Iter 15: PFS = 0 (early progression / immediate failure) ----
n0 = int((df["pfs_months"] == 0).sum())
prop0 = float((df["pfs_months"] == 0).mean())
# What predicts PFS=0?
df["pfs0"] = (df["pfs_months"] == 0).astype(int)
import statsmodels.formula.api as smf2
pf0_mod = smf2.logit("pfs0 ~ feature_078 + C(feature_057) + feature_051 + feature_038 + feature_013 + feature_043", data=df).fit(disp=0)
add("pfs_zero_predictors", {
    "n_pfs0": n0, "prop_pfs0": prop0,
    "params": {k: float(v) for k, v in pf0_mod.params.items()},
    "p": {k: float(v) for k, v in pf0_mod.pvalues.items()},
})

# ---- Iter 16: feature_099, feature_092, feature_009 ----
# Range of these:
desc_99 = df["feature_099"].describe()
desc_92 = df["feature_092"].describe()
desc_09 = df["feature_009"].describe()
# Log transform highly skewed variables and re-test
df["log_f009"] = np.log1p(df["feature_009"])
df["log_f099"] = np.log1p(df["feature_099"])
mod_log = smf.ols("pfs_months ~ feature_078 + C(feature_057) + log_f009 + log_f099 + feature_092", data=df).fit()
add("continuous_log_transformed", {
    "f099_describe": desc_99.to_dict(),
    "f092_describe": desc_92.to_dict(),
    "f009_describe": desc_09.to_dict(),
    "log_model_params": {k: float(v) for k, v in mod_log.params.items() if 'log' in k or 'feature_092' in k},
    "log_model_p": {k: float(v) for k, v in mod_log.pvalues.items() if 'log' in k or 'feature_092' in k},
})

# ---- Iter 17: Race-by-treatment interaction ----
race_tx = smf.ols("pfs_months ~ C(feature_064, Treatment(reference='white')) * feature_038 + feature_078 + C(feature_057)", data=df).fit()
race_tx_p = race_tx.f_test([f"C(feature_064, Treatment(reference='white'))[T.{r}]:feature_038 = 0" for r in ['asian', 'black', 'hispanic', 'other']]).pvalue
add("race_x_treatment", {"joint_p": float(race_tx_p),
                           "coefs": {k: float(v) for k, v in race_tx.params.items() if ':feature_038' in k},
                           "p_each": {k: float(v) for k, v in race_tx.pvalues.items() if ':feature_038' in k}})

# ---- Iter 18: Insurance-by-treatment interaction ----
ins_tx = smf.ols("pfs_months ~ C(feature_018, Treatment(reference='private')) * feature_038 + feature_078 + C(feature_057)", data=df).fit()
ins_tx_p = ins_tx.f_test([f"C(feature_018, Treatment(reference='private'))[T.{r}]:feature_038 = 0" for r in ['medicaid', 'medicare', 'uninsured']]).pvalue
add("insurance_x_treatment", {"joint_p": float(ins_tx_p),
                                "coefs": {k: float(v) for k, v in ins_tx.params.items() if ':feature_038' in k},
                                "p_each": {k: float(v) for k, v in ins_tx.pvalues.items() if ':feature_038' in k}})

# ---- Iter 19: Top 5 features by absolute multivariable contribution ----
# Standardize and refit
df_z = df.copy()
for c in ["feature_078", "feature_099", "feature_092", "feature_009"]:
    df_z[c + "_z"] = (df_z[c] - df_z[c].mean()) / df_z[c].std()
big_mv = smf.ols(
    "pfs_months ~ feature_078_z + C(feature_057) + feature_051 + feature_038 + feature_013 + feature_043 "
    "+ feature_109 + feature_067 + feature_099_z + feature_092_z + feature_009_z + "
    "feature_006 + feature_055 + feature_070 + feature_028 + feature_065",
    data=df_z,
).fit()
add("standardized_multivariable", {
    "r2": float(big_mv.rsquared),
    "n_obs": int(big_mv.nobs),
    "params": {k: float(v) for k, v in big_mv.params.items()},
    "p": {k: float(v) for k, v in big_mv.pvalues.items()},
})

# ---- Iter 20: PFS-related extreme analyses ----
# Patients with very long PFS (>10 mo)
long_pfs = (df["pfs_months"] > 10).astype(int)
df["long_pfs"] = long_pfs
n_long = int(long_pfs.sum())
# Predictors of long PFS
lp_mod = smf2.logit("long_pfs ~ feature_078 + C(feature_057) + feature_051 + feature_038 + feature_013 + feature_043 + feature_099 + feature_092", data=df).fit(disp=0)
add("long_pfs_predictors", {
    "n_long_pfs": n_long, "prop": float(n_long / len(df)),
    "params": {k: float(v) for k, v in lp_mod.params.items()},
    "p": {k: float(v) for k, v in lp_mod.pvalues.items()},
})

# ---- Iter 21: Three-way interaction f057 x f038 x f078 (age) ----
df["f057_high"] = (df["feature_057"] >= 1).astype(int)
three_way = smf.ols("pfs_months ~ feature_038 * f057_high * feature_078", data=df).fit()
add("three_way_038_057_age", {
    "params": {k: float(v) for k, v in three_way.params.items()},
    "p": {k: float(v) for k, v in three_way.pvalues.items()},
})

# ---- Iter 22: Identify what may be sex (~50% prevalence binary, near-null PFS effect) ----
# Already screened. feature_005 (45%), feature_088 (45%), feature_012 (48%), feature_106 (55%), feature_051 (55%) are top mid-prev
# Sex is usually around 50/50. Check feature_088 effect
sex_cands = ["feature_088", "feature_005", "feature_012", "feature_106"]
sex_results = {}
for c in sex_cands:
    g0 = df.loc[df[c] == 0, "pfs_months"]
    g1 = df.loc[df[c] == 1, "pfs_months"]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    sex_results[c] = {"prev_1": float(df[c].mean()), "diff": float(g1.mean() - g0.mean()), "p": float(p)}
add("sex_candidates", sex_results)

# ---- Iter 23: Check site-of-mets-like or stage features ----
# Test relationship of feature_071 (11 levels, 0-10) - could be metastatic site count or follow-up time
f71 = df.groupby("feature_071")["pfs_months"].agg(["count", "mean"]).to_dict()
f71_corr = float(stats.pearsonr(df["feature_071"], df["pfs_months"])[0])
add("feature_071_eleven_levels", {"by_level": f71, "linear_corr": f71_corr})

# ---- Iter 24: Variables that strongly correlate with feature_078 (age) ----
# Look at which features are different across feature_057 to understand it
# Check what features track feature_057
f57_corr = {}
for c in df.columns:
    if c in ("patient_id", "pfs_months", "feature_057", "q078"):
        continue
    if df[c].dtype in (np.int64, int):
        try:
            r, p = stats.pearsonr(df[c], df["feature_057"])
            f57_corr[c] = (float(r), float(p))
        except:
            pass
top_57 = sorted(f57_corr.items(), key=lambda x: abs(x[1][0]), reverse=True)[:10]
add("features_correlated_with_f057", top_57)

# ---- Iter 25: Final integrative race-disparities investigation ----
# Compare race-stratified rates of feature_038 (the apparent beneficial treatment)
race_tx_rate = df.groupby("feature_064")["feature_038"].mean().to_dict()
ins_tx_rate = df.groupby("feature_018")["feature_038"].mean().to_dict()
# And rate of poor prognostic feature_051
race_pp_rate = df.groupby("feature_064")["feature_051"].mean().to_dict()
ins_pp_rate = df.groupby("feature_018")["feature_051"].mean().to_dict()
add("race_insurance_treatment_rates", {
    "race_f038_rate": race_tx_rate, "ins_f038_rate": ins_tx_rate,
    "race_f051_rate": race_pp_rate, "ins_f051_rate": ins_pp_rate,
})

# Save all
with open("analysis_results.json", "w") as f:
    json.dump(OUT, f, default=str, indent=2)
print("WROTE analysis_results.json")
