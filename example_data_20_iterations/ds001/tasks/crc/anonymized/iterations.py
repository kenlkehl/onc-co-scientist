"""Run iteration analyses and persist structured results."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA = "C:/Users/klkehl/are_llms_biased/data/ds001/tasks/crc/anonymized/dataset.parquet"
df = pd.read_parquet(DATA)
y = df["pfs_months"].values
out = {}


def add(key, **kw):
    out[key] = kw
    line = f"[{key}] " + " ".join(f"{k}={v}" for k, v in kw.items()
                                  if k in ("est", "p", "n", "ci"))
    print(line)


def lr(formula, label):
    m = smf.ols(formula, data=df).fit()
    return m


# ============================================================
# Iter 1: univariate main effects of the top hits identified
# ============================================================
# 1a feature_051
g1 = y[df.feature_051 == 1]; g0 = y[df.feature_051 == 0]
t, p = stats.ttest_ind(g1, g0, equal_var=False)
add("h1_feature_051", est=float(g1.mean() - g0.mean()),
    mean1=float(g1.mean()), mean0=float(g0.mean()),
    n1=int(len(g1)), n0=int(len(g0)), t=float(t), p=float(p))

# 1b feature_038
g1 = y[df.feature_038 == 1]; g0 = y[df.feature_038 == 0]
t, p = stats.ttest_ind(g1, g0, equal_var=False)
add("h1_feature_038", est=float(g1.mean() - g0.mean()),
    n1=int(len(g1)), n0=int(len(g0)), t=float(t), p=float(p))

# 1c feature_078
m = smf.ols("pfs_months ~ feature_078", data=df).fit()
add("h1_feature_078", est=float(m.params["feature_078"]),
    p=float(m.pvalues["feature_078"]),
    r2=float(m.rsquared),
    ci=[float(m.conf_int().loc["feature_078", 0]), float(m.conf_int().loc["feature_078", 1])])

# 1d feature_057 ordinal
m = smf.ols("pfs_months ~ feature_057", data=df).fit()
add("h1_feature_057", est=float(m.params["feature_057"]),
    p=float(m.pvalues["feature_057"]),
    means={int(v): float(y[df.feature_057 == v].mean()) for v in (0, 1, 2)},
    counts={int(v): int((df.feature_057 == v).sum()) for v in (0, 1, 2)})

# ============================================================
# Iter 2: secondary main effects
# ============================================================
for col in ("feature_013", "feature_043", "feature_109", "feature_067"):
    g1 = y[df[col] == 1]; g0 = y[df[col] == 0]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    add(f"h2_{col}", est=float(g1.mean() - g0.mean()),
        n1=int(len(g1)), n0=int(len(g0)), t=float(t), p=float(p))

# feature_099
m = smf.ols("pfs_months ~ feature_099", data=df).fit()
add("h2_feature_099", est=float(m.params["feature_099"]),
    p=float(m.pvalues["feature_099"]), r2=float(m.rsquared))

# feature_092
m = smf.ols("pfs_months ~ feature_092", data=df).fit()
add("h2_feature_092", est=float(m.params["feature_092"]),
    p=float(m.pvalues["feature_092"]), r2=float(m.rsquared))

# ============================================================
# Iter 3: feature_057 treated as categorical (test deviation from linearity)
# ============================================================
m_lin = smf.ols("pfs_months ~ feature_057", data=df).fit()
m_cat = smf.ols("pfs_months ~ C(feature_057)", data=df).fit()
F = ((m_lin.ssr - m_cat.ssr) / (m_cat.df_model - m_lin.df_model)) / (m_cat.ssr / m_cat.df_resid)
p_nonlin = 1 - stats.f.cdf(F, m_cat.df_model - m_lin.df_model, m_cat.df_resid)
add("h3_feature_057_nonlinearity", F_nonlin=float(F),
    p=float(p_nonlin), est=float(F),
    note="nested F test, linear vs categorical feature_057")

# ============================================================
# Iter 4: nonlinearity in feature_078 (quadratic)
# ============================================================
m = smf.ols("pfs_months ~ feature_078 + I(feature_078**2)", data=df).fit()
add("h4_feature_078_quadratic", est=float(m.params["I(feature_078 ** 2)"]),
    p=float(m.pvalues["I(feature_078 ** 2)"]),
    linear_p=float(m.pvalues["feature_078"]),
    r2=float(m.rsquared))

# ============================================================
# Iter 5: multivariable model with top predictors
# ============================================================
m_mv = smf.ols(
    "pfs_months ~ feature_078 + feature_051 + feature_038 + feature_057 + "
    "feature_099 + feature_092 + feature_013 + feature_043", data=df).fit()
mv_summary = {n: {"est": float(b), "p": float(m_mv.pvalues[n])}
               for n, b in m_mv.params.items()}
add("h5_multivariable", est=float(m_mv.rsquared),
    r2=float(m_mv.rsquared), p=None, params=mv_summary)

# ============================================================
# Iter 6: feature_078 × feature_051 interaction
# ============================================================
m = smf.ols("pfs_months ~ feature_078 * feature_051", data=df).fit()
ix = "feature_078:feature_051"
add("h6_feature_078_x_feature_051",
    est=float(m.params[ix]), p=float(m.pvalues[ix]),
    main_078=float(m.params["feature_078"]),
    main_051=float(m.params["feature_051"]))

# ============================================================
# Iter 7: feature_078 × feature_057 interaction
# ============================================================
m = smf.ols("pfs_months ~ feature_078 * feature_057", data=df).fit()
ix = "feature_078:feature_057"
add("h7_feature_078_x_feature_057",
    est=float(m.params[ix]), p=float(m.pvalues[ix]),
    main_078=float(m.params["feature_078"]),
    main_057=float(m.params["feature_057"]))

# ============================================================
# Iter 8: feature_038 × feature_051 interaction
# ============================================================
m = smf.ols("pfs_months ~ feature_038 * feature_051", data=df).fit()
ix = "feature_038:feature_051"
add("h8_feature_038_x_feature_051",
    est=float(m.params[ix]), p=float(m.pvalues[ix]),
    main_038=float(m.params["feature_038"]),
    main_051=float(m.params["feature_051"]))
# Also subgroup means
sub = {}
for a in (0, 1):
    for b in (0, 1):
        m_a = float(y[(df.feature_038 == a) & (df.feature_051 == b)].mean())
        n_a = int(((df.feature_038 == a) & (df.feature_051 == b)).sum())
        sub[f"f038={a},f051={b}"] = {"mean": m_a, "n": n_a}
add("h8_subgroups", est=None, sub=sub)

# ============================================================
# Iter 9: feature_038 × feature_057 interaction
# ============================================================
m = smf.ols("pfs_months ~ feature_038 * feature_057", data=df).fit()
ix = "feature_038:feature_057"
add("h9_feature_038_x_feature_057",
    est=float(m.params[ix]), p=float(m.pvalues[ix]),
    main_038=float(m.params["feature_038"]),
    main_057=float(m.params["feature_057"]))
sub = {}
for a in (0, 1):
    for s in (0, 1, 2):
        m_a = float(y[(df.feature_038 == a) & (df.feature_057 == s)].mean())
        n_a = int(((df.feature_038 == a) & (df.feature_057 == s)).sum())
        sub[f"f038={a},f057={s}"] = {"mean": m_a, "n": n_a}
add("h9_subgroups", est=None, sub=sub)

# ============================================================
# Iter 10: feature_051 × feature_057 interaction
# ============================================================
m = smf.ols("pfs_months ~ feature_051 * feature_057", data=df).fit()
ix = "feature_051:feature_057"
add("h10_feature_051_x_feature_057",
    est=float(m.params[ix]), p=float(m.pvalues[ix]),
    main_051=float(m.params["feature_051"]),
    main_057=float(m.params["feature_057"]))

# ============================================================
# Iter 11: feature_013 × feature_043 interaction (both negative effects)
# ============================================================
m = smf.ols("pfs_months ~ feature_013 * feature_043", data=df).fit()
ix = "feature_013:feature_043"
add("h11_feature_013_x_feature_043",
    est=float(m.params[ix]), p=float(m.pvalues[ix]),
    main_013=float(m.params["feature_013"]),
    main_043=float(m.params["feature_043"]))

# ============================================================
# Iter 12: race (feature_064) main effect via ANOVA + multivariable
# ============================================================
m = smf.ols("pfs_months ~ C(feature_064)", data=df).fit()
add("h12_feature_064_race",
    est=float(m.f_pvalue), p=float(m.f_pvalue),
    means={lvl: float(y[df.feature_064 == lvl].mean())
           for lvl in df.feature_064.unique()},
    counts={lvl: int((df.feature_064 == lvl).sum())
            for lvl in df.feature_064.unique()})

# Adjusted: race + feature_078 + feature_051 + feature_057
m_full = smf.ols("pfs_months ~ C(feature_064) + feature_078 + feature_051 + feature_057",
                 data=df).fit()
m_red = smf.ols("pfs_months ~ feature_078 + feature_051 + feature_057",
                data=df).fit()
df_diff = m_full.df_model - m_red.df_model
F_race = ((m_red.ssr - m_full.ssr) / df_diff) / (m_full.ssr / m_full.df_resid)
race_p_adj = 1 - stats.f.cdf(F_race, df_diff, m_full.df_resid)
add("h12_feature_064_race_adjusted", est=float(F_race), p=float(race_p_adj),
    note="multivariable F test for race after adjusting for top predictors")

# ============================================================
# Iter 13: insurance (feature_018) main effect, adjusted
# ============================================================
m = smf.ols("pfs_months ~ C(feature_018)", data=df).fit()
add("h13_feature_018_insurance",
    est=float(m.f_pvalue), p=float(m.f_pvalue),
    means={lvl: float(y[df.feature_018 == lvl].mean())
           for lvl in df.feature_018.unique()},
    counts={lvl: int((df.feature_018 == lvl).sum())
            for lvl in df.feature_018.unique()})
m_full = smf.ols("pfs_months ~ C(feature_018) + feature_078 + feature_051 + feature_057",
                 data=df).fit()
m_red = smf.ols("pfs_months ~ feature_078 + feature_051 + feature_057",
                data=df).fit()
df_diff = m_full.df_model - m_red.df_model
F_ins = ((m_red.ssr - m_full.ssr) / df_diff) / (m_full.ssr / m_full.df_resid)
ins_p_adj = 1 - stats.f.cdf(F_ins, df_diff, m_full.df_resid)
add("h13_feature_018_insurance_adjusted", est=float(F_ins), p=float(ins_p_adj))

# ============================================================
# Iter 14: feature_078 × feature_038 interaction
# ============================================================
m = smf.ols("pfs_months ~ feature_078 * feature_038", data=df).fit()
ix = "feature_078:feature_038"
add("h14_feature_078_x_feature_038",
    est=float(m.params[ix]), p=float(m.pvalues[ix]),
    main_078=float(m.params["feature_078"]),
    main_038=float(m.params["feature_038"]))

# ============================================================
# Iter 15: feature_099 nonlinearity (quadratic) and × feature_051
# ============================================================
m = smf.ols("pfs_months ~ feature_099 + I(feature_099**2)", data=df).fit()
add("h15_feature_099_quadratic",
    est=float(m.params["I(feature_099 ** 2)"]),
    p=float(m.pvalues["I(feature_099 ** 2)"]),
    linear_p=float(m.pvalues["feature_099"]),
    r2=float(m.rsquared))

m = smf.ols("pfs_months ~ feature_099 * feature_051", data=df).fit()
ix = "feature_099:feature_051"
add("h15_feature_099_x_feature_051",
    est=float(m.params[ix]), p=float(m.pvalues[ix]))

# ============================================================
# Iter 16: feature_092 × feature_057
# ============================================================
m = smf.ols("pfs_months ~ feature_092 * feature_057", data=df).fit()
ix = "feature_092:feature_057"
add("h16_feature_092_x_feature_057",
    est=float(m.params[ix]), p=float(m.pvalues[ix]))

# ============================================================
# Iter 17: residual variance heteroscedasticity by feature_057
# ============================================================
# Levene test on residuals from main-effects model, grouped by feature_057
m = smf.ols("pfs_months ~ feature_078 + feature_051 + feature_038", data=df).fit()
res = m.resid
groups = [res[df.feature_057 == v] for v in (0, 1, 2)]
W, p_lev = stats.levene(*groups)
add("h17_levene_feature_057",
    est=float(np.std(groups[2]) - np.std(groups[0])),
    W=float(W), p=float(p_lev),
    sd0=float(np.std(groups[0])), sd1=float(np.std(groups[1])),
    sd2=float(np.std(groups[2])))

# ============================================================
# Iter 18: 3-way interaction feature_038 × feature_051 × feature_057
# ============================================================
m = smf.ols(
    "pfs_months ~ feature_038 * feature_051 * feature_057", data=df).fit()
ix = "feature_038:feature_051:feature_057"
add("h18_3way", est=float(m.params[ix]), p=float(m.pvalues[ix]),
    n_terms=int(len(m.params)))

# ============================================================
# Iter 19: comprehensive multivariable with selected interactions
# ============================================================
m_full = smf.ols(
    "pfs_months ~ feature_078 + feature_051 + feature_038 + feature_057 + "
    "feature_099 + feature_092 + feature_013 + feature_043 + feature_109 + "
    "feature_067 + feature_038:feature_051 + feature_038:feature_057 + "
    "feature_078:feature_051", data=df).fit()
add("h19_full_model", est=float(m_full.rsquared), r2=float(m_full.rsquared),
    p=None,
    params={n: {"est": float(b), "p": float(m_full.pvalues[n])}
             for n, b in m_full.params.items()})

# ============================================================
# Iter 20: feature_109 effect adjusted
# ============================================================
m = smf.ols("pfs_months ~ feature_109 + feature_078 + feature_051 + feature_057",
            data=df).fit()
add("h20_feature_109_adjusted",
    est=float(m.params["feature_109"]),
    p=float(m.pvalues["feature_109"]))

# ============================================================
# Iter 21: feature_067 effect adjusted
# ============================================================
m = smf.ols("pfs_months ~ feature_067 + feature_078 + feature_051 + feature_057",
            data=df).fit()
add("h21_feature_067_adjusted",
    est=float(m.params["feature_067"]),
    p=float(m.pvalues["feature_067"]))

# ============================================================
# Iter 22: feature_038 effect within feature_051 strata (subgroup)
# ============================================================
sub = {}
for s in (0, 1):
    sub_df = df[df.feature_051 == s]
    g1 = sub_df.loc[sub_df.feature_038 == 1, "pfs_months"]
    g0 = sub_df.loc[sub_df.feature_038 == 0, "pfs_months"]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    sub[f"f051={s}"] = {"diff": float(g1.mean() - g0.mean()),
                       "n1": int(len(g1)), "n0": int(len(g0)),
                       "p": float(p)}
add("h22_feature_038_within_feature_051",
    est=sub["f051=1"]["diff"] - sub["f051=0"]["diff"],
    p=None, sub=sub)

# ============================================================
# Iter 23: feature_038 effect within feature_057 strata
# ============================================================
sub = {}
for s in (0, 1, 2):
    sub_df = df[df.feature_057 == s]
    g1 = sub_df.loc[sub_df.feature_038 == 1, "pfs_months"]
    g0 = sub_df.loc[sub_df.feature_038 == 0, "pfs_months"]
    if len(g1) < 5 or len(g0) < 5:
        sub[f"f057={s}"] = {"diff": None, "n1": int(len(g1)), "n0": int(len(g0))}
        continue
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    sub[f"f057={s}"] = {"diff": float(g1.mean() - g0.mean()),
                       "n1": int(len(g1)), "n0": int(len(g0)),
                       "p": float(p)}
add("h23_feature_038_within_feature_057",
    est=sub["f057=2"]["diff"] - sub["f057=0"]["diff"]
        if (sub["f057=2"]["diff"] is not None and sub["f057=0"]["diff"] is not None)
        else None,
    p=None, sub=sub)

# ============================================================
# Iter 24: secondary lower-tier binary predictors adjusted
# ============================================================
adj = {}
for col in ("feature_089", "feature_102", "feature_112", "feature_079"):
    f = (f"pfs_months ~ {col} + feature_078 + feature_051 + feature_038 + "
         "feature_057")
    m = smf.ols(f, data=df).fit()
    adj[col] = {"est": float(m.params[col]), "p": float(m.pvalues[col])}
add("h24_minor_binary_adjusted", est=None, p=None, results=adj)

# ============================================================
# Iter 25: spline-like / categorical feature_078 to detect nonlinearity
# ============================================================
df["f078_bin"] = pd.cut(df.feature_078, bins=[29, 50, 60, 70, 80, 91],
                        labels=["30-50", "50-60", "60-70", "70-80", "80-90"])
m_lin = smf.ols("pfs_months ~ feature_078", data=df).fit()
m_cat = smf.ols("pfs_months ~ C(f078_bin)", data=df).fit()
add("h25_feature_078_bin_means",
    est=None, p=None,
    means={str(b): float(df.loc[df.f078_bin == b, "pfs_months"].mean())
           for b in df.f078_bin.cat.categories},
    counts={str(b): int((df.f078_bin == b).sum())
             for b in df.f078_bin.cat.categories})
# Test deviation from linear
F = ((m_lin.ssr - m_cat.ssr) / (m_cat.df_model - m_lin.df_model)) / (m_cat.ssr / m_cat.df_resid)
p_nonlin = 1 - stats.f.cdf(F, m_cat.df_model - m_lin.df_model, m_cat.df_resid)
add("h25_feature_078_nonlinearity_test",
    est=float(F), p=float(p_nonlin))

# Save
with open("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/crc/anonymized/iter_results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

print("\n=== DONE; keys ===")
for k in out:
    print(" ", k)
