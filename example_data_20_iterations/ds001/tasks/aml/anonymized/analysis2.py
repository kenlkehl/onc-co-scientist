"""Multivariable, interaction, and subgroup analyses."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
y = df["objective_response"].astype(int)

# Top features identified from univariate
top_continuous = ["feature_011", "feature_006", "feature_099", "feature_063", "feature_092", "feature_084"]
top_binary = ["feature_035", "feature_093", "feature_121", "feature_014", "feature_012"]
top_cat = ["feature_057", "feature_005"]

# Standardize continuous
df_std = df.copy()
for c in top_continuous:
    df_std[c] = (df[c] - df[c].mean()) / df[c].std()

# ============== MULTIVARIABLE MODEL ==============
print("=== Multivariable model ===")
formula = ("objective_response ~ " +
           " + ".join(top_continuous) + " + " +
           " + ".join(top_binary) + " + " +
           "C(feature_057) + C(feature_005)")
m = smf.logit(formula, data=df_std).fit(disp=0, maxiter=200)
print(m.summary().tables[1])
mv = []
for name, coef, se, pval in zip(m.params.index, m.params.values, m.bse.values, m.pvalues.values):
    mv.append({"term": name, "coef": float(coef), "se": float(se), "p": float(pval),
               "OR": float(np.exp(coef))})

# ============== INTERACTIONS ==============
print("\n=== Interactions: feature_035 (Tx?) x continuous ===")
interactions = []

# Treatment-like x age-like
for a in ["feature_035"]:  # main "treatment"-like
    for b in ["feature_006", "feature_011", "feature_063", "feature_092", "feature_099"]:
        f = f"objective_response ~ {a} * {b}"
        m_int = smf.logit(f, data=df_std).fit(disp=0, maxiter=200)
        # interaction term
        ix_name = f"{a}:{b}"
        if ix_name in m_int.params.index:
            interactions.append({
                "a": a, "b": b, "interaction_coef": float(m_int.params[ix_name]),
                "interaction_p": float(m_int.pvalues[ix_name]),
                "main_a_coef": float(m_int.params[a]),
                "main_b_coef": float(m_int.params[b])
            })
            print(f"  {a} x {b}: int_coef={m_int.params[ix_name]:.4f} p={m_int.pvalues[ix_name]:.4g}")

# feature_035 x feature_057 (ordinal severity)
for a in ["feature_035"]:
    for b in ["feature_057"]:
        f = f"objective_response ~ C({a}) * C({b})"
        m_int = smf.logit(f, data=df_std).fit(disp=0, maxiter=200)
        m_main = smf.logit(f"objective_response ~ C({a}) + C({b})", data=df_std).fit(disp=0, maxiter=200)
        lr = 2*(m_int.llf - m_main.llf)
        ddf = int(m_int.df_model - m_main.df_model)
        p = float(stats.chi2.sf(lr, ddf))
        interactions.append({"a": a, "b": b, "type": "categorical_lr",
                             "lr_p": p, "df": ddf})
        print(f"  {a} x C({b}) LR p={p:.4g}")

# feature_057 x continuous severity x age
for a in ["feature_057"]:
    for b in ["feature_006", "feature_011"]:
        f = f"objective_response ~ C({a}) * {b}"
        m_int = smf.logit(f, data=df_std).fit(disp=0, maxiter=200)
        m_main = smf.logit(f"objective_response ~ C({a}) + {b}", data=df_std).fit(disp=0, maxiter=200)
        lr = 2*(m_int.llf - m_main.llf)
        ddf = int(m_int.df_model - m_main.df_model)
        p = float(stats.chi2.sf(lr, ddf))
        interactions.append({"a": a, "b": b, "type": "cat_x_cont",
                             "lr_p": p, "df": ddf})
        print(f"  C({a}) x {b} LR p={p:.4g}")

# Continuous x continuous interactions among top features
for a, b in [("feature_006", "feature_011"), ("feature_006", "feature_063"),
             ("feature_011", "feature_092"), ("feature_011", "feature_099"),
             ("feature_006", "feature_099"), ("feature_063", "feature_092")]:
    f = f"objective_response ~ {a} * {b}"
    m_int = smf.logit(f, data=df_std).fit(disp=0, maxiter=200)
    ix_name = f"{a}:{b}"
    interactions.append({
        "a": a, "b": b, "type": "cont_x_cont",
        "interaction_coef": float(m_int.params[ix_name]),
        "interaction_p": float(m_int.pvalues[ix_name])
    })
    print(f"  {a} x {b}: int_coef={m_int.params[ix_name]:.4f} p={m_int.pvalues[ix_name]:.4g}")

# Race x feature_035 (does race modulate treatment effect?)
f = "objective_response ~ feature_035 * C(feature_005)"
m_int = smf.logit(f, data=df_std).fit(disp=0, maxiter=200)
m_main = smf.logit("objective_response ~ feature_035 + C(feature_005)", data=df_std).fit(disp=0, maxiter=200)
lr = 2*(m_int.llf - m_main.llf)
ddf = int(m_int.df_model - m_main.df_model)
p_race035 = float(stats.chi2.sf(lr, ddf))
interactions.append({"a": "feature_035", "b": "feature_005", "type": "race_x_tx",
                     "lr_p": p_race035, "df": ddf})
print(f"  feature_035 x C(feature_005) LR p={p_race035:.4g}")

# ============== SUBGROUPS ==============
print("\n=== Subgroup: response by feature_057 level ===")
subgroups = []
# response by feature_057
for lev in [0, 1, 2]:
    sub = df[df["feature_057"] == lev]
    rate = sub["objective_response"].mean()
    subgroups.append({"feature": "feature_057", "level": lev, "n": len(sub),
                      "rate": float(rate)})

# feature_035 effect within each feature_057 level
print("\n=== feature_035 effect stratified by feature_057 ===")
strat = []
for lev in [0, 1, 2]:
    sub = df_std[df_std["feature_057"] == lev]
    if sub["feature_035"].nunique() < 2:
        continue
    m_sub = smf.logit("objective_response ~ feature_035", data=sub).fit(disp=0)
    rate1 = sub.loc[sub["feature_035"]==1, "objective_response"].mean()
    rate0 = sub.loc[sub["feature_035"]==0, "objective_response"].mean()
    strat.append({
        "stratum": f"feature_057={lev}",
        "n": len(sub),
        "rate_x1": float(rate1),
        "rate_x0": float(rate0),
        "log_or": float(m_sub.params["feature_035"]),
        "p": float(m_sub.pvalues["feature_035"]),
        "OR": float(np.exp(m_sub.params["feature_035"]))
    })
    print(f"  feature_057={lev}: n={len(sub)}, rate_f035=1: {rate1:.3f}, rate_f035=0: {rate0:.3f}, OR={np.exp(m_sub.params['feature_035']):.3f} p={m_sub.pvalues['feature_035']:.4g}")

# Continuous by quartile (feature_006 = age-like)
print("\n=== response by feature_006 quartile ===")
df_q = df.copy()
df_q["f006_q"] = pd.qcut(df_q["feature_006"], 4, labels=False)
quartile_rates = []
for q in [0,1,2,3]:
    sub = df_q[df_q["f006_q"]==q]
    rng = (sub["feature_006"].min(), sub["feature_006"].max())
    rate = sub["objective_response"].mean()
    quartile_rates.append({"feature": "feature_006", "quartile": int(q),
                            "min": float(rng[0]), "max": float(rng[1]),
                            "n": len(sub), "rate": float(rate)})
    print(f"  Q{q+1} ({rng[0]:.1f}-{rng[1]:.1f}): n={len(sub)} rate={rate:.4f}")

# response by feature_011 quartile
print("\n=== response by feature_011 quartile ===")
df_q["f011_q"] = pd.qcut(df_q["feature_011"], 4, labels=False, duplicates="drop")
for q in sorted(df_q["f011_q"].dropna().unique()):
    sub = df_q[df_q["f011_q"]==q]
    rng = (sub["feature_011"].min(), sub["feature_011"].max())
    rate = sub["objective_response"].mean()
    quartile_rates.append({"feature": "feature_011", "quartile": int(q),
                            "min": float(rng[0]), "max": float(rng[1]),
                            "n": len(sub), "rate": float(rate)})
    print(f"  Q{q+1} ({rng[0]:.2f}-{rng[1]:.2f}): n={len(sub)} rate={rate:.4f}")

# Race-stratified response rates
print("\n=== response by race (feature_005) ===")
race_rates = []
for race in df["feature_005"].unique():
    sub = df[df["feature_005"]==race]
    rate = sub["objective_response"].mean()
    race_rates.append({"race": race, "n": len(sub), "rate": float(rate)})

# Insurance-stratified
ins_rates = []
for ins in df["feature_087"].unique():
    sub = df[df["feature_087"]==ins]
    rate = sub["objective_response"].mean()
    ins_rates.append({"insurance": ins, "n": len(sub), "rate": float(rate)})

# Adjusted race effect (after controlling for top covariates)
print("\n=== Adjusted race effect (in multivariable) ===")
f_adj = ("objective_response ~ feature_011 + feature_006 + feature_099 + feature_063 + feature_092 + "
         "feature_035 + C(feature_057) + C(feature_005)")
m_adj = smf.logit(f_adj, data=df_std).fit(disp=0, maxiter=200)
race_adj = []
for term in m_adj.params.index:
    if "feature_005" in term:
        race_adj.append({"term": term,
                         "coef": float(m_adj.params[term]),
                         "p": float(m_adj.pvalues[term]),
                         "OR": float(np.exp(m_adj.params[term]))})
        print(f"  {term}: OR={np.exp(m_adj.params[term]):.3f} p={m_adj.pvalues[term]:.4g}")

# Test race effect via LRT after adjustment
m_adj_norace = smf.logit(("objective_response ~ feature_011 + feature_006 + feature_099 + feature_063 + feature_092 + "
                          "feature_035 + C(feature_057)"), data=df_std).fit(disp=0, maxiter=200)
lr_race_adj = 2*(m_adj.llf - m_adj_norace.llf)
df_race = int(m_adj.df_model - m_adj_norace.df_model)
p_race_adj = float(stats.chi2.sf(lr_race_adj, df_race))
print(f"  Adjusted race LR p = {p_race_adj:.4g}")

# Test insurance after adjustment
f_ins = f_adj + " + C(feature_087)"
m_ins = smf.logit(f_ins, data=df_std).fit(disp=0, maxiter=200)
lr_ins = 2*(m_ins.llf - m_adj.llf)
df_ins = int(m_ins.df_model - m_adj.df_model)
p_ins = float(stats.chi2.sf(lr_ins, df_ins))
print(f"  Adjusted insurance LR p = {p_ins:.4g}")
ins_adj_terms = []
for term in m_ins.params.index:
    if "feature_087" in term:
        ins_adj_terms.append({"term": term,
                               "coef": float(m_ins.params[term]),
                               "p": float(m_ins.pvalues[term]),
                               "OR": float(np.exp(m_ins.params[term]))})

out = {
    "multivariable": mv,
    "interactions": interactions,
    "subgroups_f057": subgroups,
    "stratified_f035_by_f057": strat,
    "quartiles": quartile_rates,
    "race_rates": race_rates,
    "insurance_rates": ins_rates,
    "race_adjusted": race_adj,
    "race_adj_LR_p": p_race_adj,
    "insurance_adjusted": ins_adj_terms,
    "insurance_adj_LR_p": p_ins
}
with open("results2.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nDone. Wrote results2.json")
