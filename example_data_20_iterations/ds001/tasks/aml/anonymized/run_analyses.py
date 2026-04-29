"""Run all hypothesis tests for the 25-iteration analysis protocol."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
y = df["objective_response"]
N = len(df)
print(f"N={N}, response_rate={y.mean():.4f}")

# Helpers ------------------------------------------------------------------

def chi2_2x2(col, val_a=1, val_b=0):
    a = df.loc[df[col] == val_a, "objective_response"]
    b = df.loc[df[col] == val_b, "objective_response"]
    rr_a, rr_b = a.mean(), b.mean()
    tab = pd.crosstab(df[col], y)
    chi2, p, _, _ = stats.chi2_contingency(tab)
    return rr_a, rr_b, rr_a - rr_b, p, len(a), len(b)

def ttest_continuous(col):
    a = df.loc[y == 1, col]
    b = df.loc[y == 0, col]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return a.mean(), b.mean(), a.mean() - b.mean(), p

def logit_or(formula, data=None):
    if data is None:
        data = df
    m = smf.logit(formula, data=data).fit(disp=0)
    return m

results = {}

# Iter 1: feature_035 main effect
print("\n=== Iter 1: feature_035 ===")
rr1, rr0, diff, p, n1, n0 = chi2_2x2("feature_035")
print(f"feature_035=1: rr={rr1:.4f} (n={n1}); =0: rr={rr0:.4f} (n={n0}); diff={diff:.4f}, p={p:.3e}")
results["i1"] = dict(rr1=rr1, rr0=rr0, diff=diff, p=p, n1=n1, n0=n0)

# Iter 2: feature_057 main effect (3-level)
print("\n=== Iter 2: feature_057 (3-level) ===")
tab = pd.crosstab(df["feature_057"], y)
chi2, p_chi, _, _ = stats.chi2_contingency(tab)
rr_lvl = df.groupby("feature_057")["objective_response"].mean()
print(rr_lvl, "chi2 p=", p_chi)
m = smf.logit("objective_response ~ feature_057", data=df).fit(disp=0)
print("logit slope per level:", m.params["feature_057"], "p=", m.pvalues["feature_057"])
results["i2"] = dict(rr_by_level=rr_lvl.to_dict(), chi2_p=p_chi, slope=m.params["feature_057"], slope_p=m.pvalues["feature_057"])

# Iter 3: feature_011 (continuous lab)
print("\n=== Iter 3: feature_011 ===")
mr, mn, diff, p = ttest_continuous("feature_011")
m = smf.logit("objective_response ~ feature_011", data=df).fit(disp=0)
print(f"resp mean={mr:.3f}, nonresp={mn:.3f}, diff={diff:.3f}, p={p:.3e}; logit beta={m.params['feature_011']:.4f}, p={m.pvalues['feature_011']:.3e}")
results["i3"] = dict(mr=mr, mn=mn, diff=diff, p=p, beta=m.params["feature_011"], beta_p=m.pvalues["feature_011"])

# Iter 4: feature_006 (continuous)
print("\n=== Iter 4: feature_006 ===")
mr, mn, diff, p = ttest_continuous("feature_006")
m = smf.logit("objective_response ~ feature_006", data=df).fit(disp=0)
print(f"resp mean={mr:.3f}, nonresp={mn:.3f}, diff={diff:.3f}, p={p:.3e}; logit beta={m.params['feature_006']:.4f}, p={m.pvalues['feature_006']:.3e}")
results["i4"] = dict(mr=mr, mn=mn, diff=diff, p=p, beta=m.params["feature_006"], beta_p=m.pvalues["feature_006"])

# Iter 5: age (feature_078) main effect
print("\n=== Iter 5: feature_078 (age) ===")
mr, mn, diff, p = ttest_continuous("feature_078")
m = smf.logit("objective_response ~ feature_078", data=df).fit(disp=0)
print(f"resp mean={mr:.3f}, nonresp={mn:.3f}, diff={diff:.3f}, p={p:.3e}; logit beta={m.params['feature_078']:.5f}, p={m.pvalues['feature_078']:.3e}")
results["i5"] = dict(mr=mr, mn=mn, diff=diff, p=p, beta=m.params["feature_078"], beta_p=m.pvalues["feature_078"])

# Iter 6: panel of other continuous markers
print("\n=== Iter 6: continuous panel ===")
for c in ["feature_063", "feature_092", "feature_099", "feature_084", "feature_055"]:
    mr, mn, diff, p = ttest_continuous(c)
    m = smf.logit(f"objective_response ~ {c}", data=df).fit(disp=0)
    print(f"  {c}: resp={mr:.3f} nonresp={mn:.3f} diff={diff:.3f} p={p:.3e} beta={m.params[c]:.4f} beta_p={m.pvalues[c]:.3e}")
    results.setdefault("i6", {})[c] = dict(mr=mr, mn=mn, diff=diff, p=p, beta=m.params[c], beta_p=m.pvalues[c])

# Iter 7: race (feature_005)
print("\n=== Iter 7: feature_005 (race) ===")
tab = pd.crosstab(df["feature_005"], y)
chi2, p_race, _, _ = stats.chi2_contingency(tab)
rr_race = df.groupby("feature_005")["objective_response"].mean()
print(rr_race, "chi2 p=", p_race)
# white vs nonwhite
df["race_white"] = (df["feature_005"] == "white").astype(int)
m = smf.logit("objective_response ~ race_white", data=df).fit(disp=0)
print("white vs nonwhite OR=", np.exp(m.params["race_white"]), "p=", m.pvalues["race_white"])
results["i7"] = dict(rr_race=rr_race.to_dict(), chi2_p=p_race,
                     beta_white=m.params["race_white"], p_white=m.pvalues["race_white"])

# Iter 8: insurance (feature_087)
print("\n=== Iter 8: feature_087 (insurance) ===")
tab = pd.crosstab(df["feature_087"], y)
chi2, p_ins, _, _ = stats.chi2_contingency(tab)
rr_ins = df.groupby("feature_087")["objective_response"].mean()
print(rr_ins, "chi2 p=", p_ins)
results["i8"] = dict(rr=rr_ins.to_dict(), chi2_p=p_ins)

# Iter 9: multivariable logistic with the strongest predictors
print("\n=== Iter 9: multivariable logistic ===")
m = smf.logit(
    "objective_response ~ feature_035 + feature_057 + feature_011 + feature_006 + "
    "feature_063 + feature_092 + feature_099 + feature_078 + race_white",
    data=df,
).fit(disp=0)
print(m.summary().tables[1])
results["i9"] = dict(coefs=m.params.to_dict(), pvals=m.pvalues.to_dict(),
                     llf=m.llf, prsquared=m.prsquared)

# Iter 10: interaction feature_035 x feature_057
print("\n=== Iter 10: feature_035 x feature_057 ===")
m = smf.logit("objective_response ~ feature_035 * feature_057", data=df).fit(disp=0)
print(m.summary().tables[1])
# stratified
for lvl in [0, 1, 2]:
    sub = df[df["feature_057"] == lvl]
    rr1 = sub.loc[sub.feature_035 == 1, "objective_response"].mean()
    rr0 = sub.loc[sub.feature_035 == 0, "objective_response"].mean()
    print(f"  feature_057={lvl}: feature_035=1 rr={rr1:.4f} (n={(sub.feature_035==1).sum()}), =0 rr={rr0:.4f} (n={(sub.feature_035==0).sum()}), diff={rr1-rr0:.4f}")
results["i10"] = dict(coefs=m.params.to_dict(), pvals=m.pvalues.to_dict(),
                      interaction_p=m.pvalues.get("feature_035:feature_057"))

# Iter 11: interaction feature_035 x race_white
print("\n=== Iter 11: feature_035 x race_white ===")
m = smf.logit("objective_response ~ feature_035 * race_white", data=df).fit(disp=0)
print(m.summary().tables[1])
for r in [0, 1]:
    sub = df[df.race_white == r]
    rr1 = sub.loc[sub.feature_035 == 1, "objective_response"].mean()
    rr0 = sub.loc[sub.feature_035 == 0, "objective_response"].mean()
    print(f"  race_white={r}: f035=1 rr={rr1:.4f} (n={(sub.feature_035==1).sum()}), =0 rr={rr0:.4f} (n={(sub.feature_035==0).sum()}), diff={rr1-rr0:.4f}")
results["i11"] = dict(coefs=m.params.to_dict(), pvals=m.pvalues.to_dict(),
                      interaction_p=m.pvalues.get("feature_035:race_white"))

# Iter 12: interaction feature_035 x feature_011
print("\n=== Iter 12: feature_035 x feature_011 ===")
m = smf.logit("objective_response ~ feature_035 * feature_011", data=df).fit(disp=0)
print(m.summary().tables[1])
results["i12"] = dict(coefs=m.params.to_dict(), pvals=m.pvalues.to_dict(),
                      interaction_p=m.pvalues.get("feature_035:feature_011"))

# Iter 13: interaction feature_035 x feature_078 (age)
print("\n=== Iter 13: feature_035 x feature_078 (age) ===")
m = smf.logit("objective_response ~ feature_035 * feature_078", data=df).fit(disp=0)
print(m.summary().tables[1])
results["i13"] = dict(coefs=m.params.to_dict(), pvals=m.pvalues.to_dict(),
                      interaction_p=m.pvalues.get("feature_035:feature_078"))

# Iter 14: feature_057 effect adjusted for feature_035 - check robustness
print("\n=== Iter 14: feature_057 adjusted for feature_035 ===")
m = smf.logit("objective_response ~ feature_057 + feature_035", data=df).fit(disp=0)
print(m.summary().tables[1])
results["i14"] = dict(coefs=m.params.to_dict(), pvals=m.pvalues.to_dict())

# Iter 15: feature_096, feature_064 (5-level scores) main effect
print("\n=== Iter 15: feature_096, feature_064 ===")
for c in ["feature_096", "feature_064"]:
    m = smf.logit(f"objective_response ~ {c}", data=df).fit(disp=0)
    rr = df.groupby(c)["objective_response"].mean()
    print(f"  {c}: rr by level={rr.to_dict()} slope_beta={m.params[c]:.4f} p={m.pvalues[c]:.3e}")
    results.setdefault("i15", {})[c] = dict(rr_by_level=rr.to_dict(), beta=m.params[c], p=m.pvalues[c])

# Iter 16: feature_018 (comorbidity-like, 11 levels)
print("\n=== Iter 16: feature_018 ===")
m = smf.logit("objective_response ~ feature_018", data=df).fit(disp=0)
rr = df.groupby("feature_018")["objective_response"].mean()
print(rr.to_dict(), "beta=", m.params["feature_018"], "p=", m.pvalues["feature_018"])
results["i16"] = dict(rr_by_level=rr.to_dict(), beta=m.params["feature_018"], p=m.pvalues["feature_018"])

# Iter 17: feature_035 within highest-risk stratum (feature_057==2)
print("\n=== Iter 17: feature_035 effect within feature_057==2 ===")
sub = df[df["feature_057"] == 2]
m = smf.logit("objective_response ~ feature_035", data=sub).fit(disp=0)
rr1 = sub.loc[sub.feature_035 == 1, "objective_response"].mean()
rr0 = sub.loc[sub.feature_035 == 0, "objective_response"].mean()
print(f"feature_057=2: f035=1 rr={rr1:.4f} (n={(sub.feature_035==1).sum()}), =0 rr={rr0:.4f} (n={(sub.feature_035==0).sum()}), diff={rr1-rr0:.4f}, beta_p={m.pvalues['feature_035']:.3e}")
results["i17"] = dict(rr1=rr1, rr0=rr0, diff=rr1-rr0, beta=m.params["feature_035"], p=m.pvalues["feature_035"])

# Iter 18: feature_093 main effect (next strongest binary)
print("\n=== Iter 18: feature_093 ===")
rr1, rr0, diff, p, n1, n0 = chi2_2x2("feature_093")
print(f"feature_093=1: rr={rr1:.4f} (n={n1}); =0: rr={rr0:.4f} (n={n0}); diff={diff:.4f}, p={p:.3e}")
results["i18"] = dict(rr1=rr1, rr0=rr0, diff=diff, p=p)

# Iter 19: 4 ordinal features (042, 045, 125) - already shown weak; do logistic
print("\n=== Iter 19: ordinal features 042,045,125 ===")
for c in ["feature_042", "feature_045", "feature_125"]:
    m = smf.logit(f"objective_response ~ {c}", data=df).fit(disp=0)
    print(f"  {c}: beta={m.params[c]:.4f} p={m.pvalues[c]:.3e}")
    results.setdefault("i19", {})[c] = dict(beta=m.params[c], p=m.pvalues[c])

# Iter 20: comprehensive multivariable model
print("\n=== Iter 20: full multivariable ===")
formula = (
    "objective_response ~ feature_035 + feature_057 + feature_011 + feature_006 + "
    "feature_063 + feature_092 + feature_099 + feature_078 + race_white + "
    "feature_093 + feature_018 + feature_096 + feature_125"
)
m = smf.logit(formula, data=df).fit(disp=0)
print(m.summary().tables[1])
results["i20"] = dict(coefs=m.params.to_dict(), pvals=m.pvalues.to_dict(),
                      llf=m.llf, prsquared=m.prsquared)

# Iter 21: scan for additional binary heterogeneity (interaction with feature_035)
print("\n=== Iter 21: scan for feature_035 x binary interactions ===")
bin_cols = [c for c in df.columns if c not in ["patient_id","objective_response","feature_035"] and df[c].nunique()==2]
inter_results = []
for c in bin_cols:
    try:
        m = smf.logit(f"objective_response ~ feature_035 * {c}", data=df).fit(disp=0)
        key = f"feature_035:{c}"
        if key in m.pvalues.index:
            inter_results.append((c, m.params[key], m.pvalues[key]))
    except Exception:
        pass
inter_df = pd.DataFrame(inter_results, columns=["feature","int_coef","int_p"]).sort_values("int_p")
print(inter_df.head(10).to_string())
results["i21"] = dict(top=inter_df.head(10).to_dict(orient="records"))

# Iter 22: feature_035 effect in each race
print("\n=== Iter 22: feature_035 by race ===")
race_eff = {}
for race in df.feature_005.unique():
    sub = df[df.feature_005==race]
    rr1 = sub.loc[sub.feature_035==1, "objective_response"].mean()
    rr0 = sub.loc[sub.feature_035==0, "objective_response"].mean()
    n1 = (sub.feature_035==1).sum(); n0 = (sub.feature_035==0).sum()
    if n1>5 and n0>5:
        try:
            m = smf.logit("objective_response ~ feature_035", data=sub).fit(disp=0)
            print(f"  {race}: rr1={rr1:.4f}(n={n1}), rr0={rr0:.4f}(n={n0}), diff={rr1-rr0:+.4f}, beta={m.params['feature_035']:.3f}, p={m.pvalues['feature_035']:.3e}")
            race_eff[race] = dict(rr1=rr1, rr0=rr0, diff=rr1-rr0, beta=m.params["feature_035"], p=m.pvalues["feature_035"])
        except Exception:
            pass
results["i22"] = race_eff

# Iter 23: dichotomize feature_011 at median
print("\n=== Iter 23: feature_011 dichotomized ===")
med = df.feature_011.median()
df["f011_high"] = (df.feature_011 > med).astype(int)
rr1, rr0, diff, p, n1, n0 = chi2_2x2("f011_high")
print(f"feature_011>{med:.2f}: rr={rr1:.4f}(n={n1}), <=: rr={rr0:.4f}(n={n0}), diff={diff:.4f}, p={p:.3e}")
results["i23"] = dict(median=med, rr_high=rr1, rr_low=rr0, diff=diff, p=p)

# Iter 24: combined risk score & evaluate predicted vs observed
print("\n=== Iter 24: combined model performance ===")
formula24 = "objective_response ~ feature_035 + feature_057 + feature_011 + feature_006 + feature_063 + feature_092 + feature_099 + race_white"
m = smf.logit(formula24, data=df).fit(disp=0)
df["pred"] = m.predict(df)
deciles = pd.qcut(df["pred"], 10, labels=False, duplicates="drop")
rr_dec = df.groupby(deciles)["objective_response"].mean()
print("Response rate by predicted-prob decile:")
print(rr_dec.to_dict())
# AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(df["objective_response"], df["pred"])
print(f"AUC={auc:.4f}")
results["i24"] = dict(rr_by_decile=rr_dec.to_dict(), auc=auc)

# Iter 25: race effect adjusted for clinical predictors (test bias)
print("\n=== Iter 25: race effect adjusted for clinical ===")
m = smf.logit(
    "objective_response ~ race_white + feature_035 + feature_057 + feature_011 + feature_006 + feature_063 + feature_092 + feature_099 + feature_078",
    data=df,
).fit(disp=0)
print(m.summary().tables[1])
results["i25"] = dict(coefs=m.params.to_dict(), pvals=m.pvalues.to_dict())
# also race_white x feature_035 with adjustment
m2 = smf.logit(
    "objective_response ~ race_white * feature_035 + feature_057 + feature_011 + feature_006 + feature_063 + feature_092 + feature_099",
    data=df,
).fit(disp=0)
print("\nrace_white*feature_035 with adjustment:")
print(m2.summary().tables[1])
results["i25b"] = dict(coefs=m2.params.to_dict(), pvals=m2.pvalues.to_dict())

# dump
with open("analysis_results.json", "w") as f:
    json.dump({k: (v if not isinstance(v, dict) else {kk: (str(vv) if isinstance(vv, (np.integer, np.floating, np.bool_)) else vv) for kk, vv in v.items()}) for k, v in results.items()}, f, indent=2, default=str)
print("\nDone.")
