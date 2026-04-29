"""Final round: a few extra targeted analyses."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")

extras = {}

# Trend test for feature_057 (ordinal): use it as numeric
m = smf.logit("objective_response ~ feature_057", data=df).fit(disp=0)
extras["f057_trend"] = {
    "coef_per_level": float(m.params["feature_057"]),
    "p": float(m.pvalues["feature_057"]),
    "OR_per_level": float(np.exp(m.params["feature_057"]))
}

# Trend test for ordinal features 045, 096, 064, 042, 125
for c in ["feature_045", "feature_096", "feature_064", "feature_042", "feature_125"]:
    m = smf.logit(f"objective_response ~ {c}", data=df).fit(disp=0)
    extras[f"{c}_trend"] = {
        "coef_per_level": float(m.params[c]),
        "p": float(m.pvalues[c]),
        "OR_per_level": float(np.exp(m.params[c]))
    }

# feature_035 effect within race strata
race_strat = []
df_std = df.copy()
for c in ["feature_011","feature_006","feature_099","feature_063","feature_092"]:
    df_std[c] = (df[c]-df[c].mean())/df[c].std()
for race in df["feature_005"].unique():
    sub = df_std[df_std["feature_005"]==race]
    if sub["feature_035"].nunique() < 2 or len(sub) < 100:
        continue
    m = smf.logit("objective_response ~ feature_035", data=sub).fit(disp=0)
    rate1 = float(sub.loc[sub["feature_035"]==1, "objective_response"].mean())
    rate0 = float(sub.loc[sub["feature_035"]==0, "objective_response"].mean())
    race_strat.append({
        "race": race, "n": len(sub),
        "rate_x1": rate1, "rate_x0": rate0,
        "log_or": float(m.params["feature_035"]),
        "p": float(m.pvalues["feature_035"]),
        "OR": float(np.exp(m.params["feature_035"]))
    })
extras["f035_by_race"] = race_strat

# feature_035 effect within feature_006 (age) tertiles
df_std["age_tert"] = pd.qcut(df_std["feature_006"], 3, labels=False)
age_strat = []
for t in [0,1,2]:
    sub = df_std[df_std["age_tert"]==t]
    rng = (sub["feature_006"].min()*df["feature_006"].std()+df["feature_006"].mean(),
           sub["feature_006"].max()*df["feature_006"].std()+df["feature_006"].mean())
    if sub["feature_035"].nunique() < 2:
        continue
    m = smf.logit("objective_response ~ feature_035", data=sub).fit(disp=0)
    rate1 = float(sub.loc[sub["feature_035"]==1, "objective_response"].mean())
    rate0 = float(sub.loc[sub["feature_035"]==0, "objective_response"].mean())
    age_strat.append({
        "tertile": int(t), "n": int(len(sub)),
        "rate_x1": rate1, "rate_x0": rate0,
        "log_or": float(m.params["feature_035"]),
        "p": float(m.pvalues["feature_035"]),
        "OR": float(np.exp(m.params["feature_035"]))
    })
extras["f035_by_age_tertile"] = age_strat

# Top binary features beyond f035: f093, f121, f014
print("=== Validating other binary features ===")
for c in ["feature_093","feature_121","feature_014"]:
    rate1 = float(df.loc[df[c]==1, "objective_response"].mean())
    rate0 = float(df.loc[df[c]==0, "objective_response"].mean())
    m = smf.logit(f"objective_response ~ {c}", data=df).fit(disp=0)
    print(f"  {c}: rate1={rate1:.4f}, rate0={rate0:.4f}, OR={np.exp(m.params[c]):.3f}, p={m.pvalues[c]:.4g}")

# Insurance unadjusted
print("\n=== Insurance unadjusted ===")
m = smf.logit("objective_response ~ C(feature_087)", data=df).fit(disp=0)
m0 = smf.logit("objective_response ~ 1", data=df).fit(disp=0)
lr = 2*(m.llf - m0.llf)
ddf = int(m.df_model)
p = float(stats.chi2.sf(lr, ddf))
extras["insurance_unadjusted_p"] = p
print(f"  Insurance LR p = {p:.4g}")
ins_rates = df.groupby("feature_087")["objective_response"].agg(["mean","count"]).to_dict("index")
extras["ins_rates"] = {k: {"rate": float(v["mean"]), "n": int(v["count"])} for k,v in ins_rates.items()}
print(f"  Rates: {extras['ins_rates']}")

# Test feature_035 x feature_005 (race) interaction more carefully
m_int = smf.logit("objective_response ~ feature_035 * C(feature_005)", data=df).fit(disp=0)
m_main = smf.logit("objective_response ~ feature_035 + C(feature_005)", data=df).fit(disp=0)
lr = 2*(m_int.llf - m_main.llf)
ddf = int(m_int.df_model - m_main.df_model)
p_race_tx = float(stats.chi2.sf(lr, ddf))
extras["f035_x_race_LR_p"] = p_race_tx
print(f"\nfeature_035 x race LR p = {p_race_tx:.4g}")

# Test interaction f035 x feature_087 (insurance)
m_int = smf.logit("objective_response ~ feature_035 * C(feature_087)", data=df).fit(disp=0)
m_main = smf.logit("objective_response ~ feature_035 + C(feature_087)", data=df).fit(disp=0)
lr = 2*(m_int.llf - m_main.llf)
ddf = int(m_int.df_model - m_main.df_model)
p_ins_tx = float(stats.chi2.sf(lr, ddf))
extras["f035_x_insurance_LR_p"] = p_ins_tx
print(f"feature_035 x insurance LR p = {p_ins_tx:.4g}")

# Cumulative top binary scan: count how many binary features have p < 0.05
import json as _json
with open("results.json") as f:
    r = _json.load(f)
b = pd.DataFrame(r["binary_univariate"])
n_sig = int((b["p"] < 0.05).sum())
extras["binary_n_sig_at_05"] = n_sig
extras["binary_n_total"] = int(len(b))
print(f"\nBinary features with p<0.05: {n_sig}/{len(b)}")

with open("results3.json","w") as f:
    json.dump(extras, f, indent=2, default=str)
print("\nWrote results3.json")
