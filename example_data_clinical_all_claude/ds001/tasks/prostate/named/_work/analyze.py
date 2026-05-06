"""Comprehensive iterative analysis of ds001_prostate.

Runs every analysis we want to record in transcript.json and prints a JSON-friendly
result for each so the wrapper script can collect them.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeClassifier
import json
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
N = len(df)

results = {}

def chi2_or(df, var, outcome="objective_response"):
    """Risk difference + chi-square for binary var vs binary outcome."""
    a = df.loc[df[var] == 1, outcome]
    b = df.loc[df[var] == 0, outcome]
    p1, p0 = a.mean(), b.mean()
    rd = p1 - p0
    ct = pd.crosstab(df[var], df[outcome])
    if ct.shape == (2, 2):
        chi2, p, *_ = stats.chi2_contingency(ct, correction=False)
    else:
        chi2, p = float("nan"), float("nan")
    n1, n0 = (df[var] == 1).sum(), (df[var] == 0).sum()
    return dict(p_on=p1, p_off=p0, rd=rd, p_value=p, n_on=int(n1), n_off=int(n0))

def logit_main(df, var, outcome="objective_response"):
    X = sm.add_constant(df[[var]].astype(float))
    m = sm.Logit(df[outcome], X).fit(disp=0)
    return dict(coef=float(m.params[var]), p=float(m.pvalues[var]),
                or_=float(np.exp(m.params[var])))

def logit_inter(df, treat, modifier, outcome="objective_response"):
    """objective_response ~ treat * modifier (modifier numeric)."""
    d = df[[treat, modifier, outcome]].copy()
    d["interaction"] = d[treat] * d[modifier]
    X = sm.add_constant(d[[treat, modifier, "interaction"]].astype(float))
    m = sm.Logit(d[outcome], X).fit(disp=0)
    return dict(
        treat_coef=float(m.params[treat]),
        treat_p=float(m.pvalues[treat]),
        mod_coef=float(m.params[modifier]),
        mod_p=float(m.pvalues[modifier]),
        inter_coef=float(m.params["interaction"]),
        inter_p=float(m.pvalues["interaction"]),
    )

def stratified_rd(df, treat, modifier, outcome="objective_response"):
    """Risk difference for treat within modifier=1 vs modifier=0."""
    out = {}
    for v in (0, 1):
        d = df[df[modifier] == v]
        on = d.loc[d[treat] == 1, outcome]
        off = d.loc[d[treat] == 0, outcome]
        out[f"mod={v}"] = dict(
            n_on=int(len(on)), n_off=int(len(off)),
            p_on=float(on.mean()) if len(on) else float("nan"),
            p_off=float(off.mean()) if len(off) else float("nan"),
            rd=float(on.mean() - off.mean()) if (len(on) and len(off)) else float("nan"),
        )
    # difference in differences via interaction
    try:
        out["interaction"] = logit_inter(df, treat, modifier, outcome)
    except Exception as e:
        out["interaction"] = {"error": str(e)}
    return out

def cont_compare(df, var, outcome="objective_response"):
    a = df.loc[df[outcome] == 1, var]
    b = df.loc[df[outcome] == 0, var]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return dict(mean_resp=float(a.mean()), mean_no=float(b.mean()),
                diff=float(a.mean() - b.mean()), p_value=float(p))


# ---------- Iter 1: baseline descriptives ----------
results["it1"] = {
    "overall_response": float(df["objective_response"].mean()),
    "n": int(len(df)),
    "ecog_dist": df["ecog_ps"].value_counts().sort_index().to_dict(),
    "mcrpc_pct": float(df["mcrpc"].mean()),
    "visceral_pct": float(df["visceral_mets"].mean()),
    "psma_high_pct": float(df["psma_high"].mean()),
    "brca2_pct": float(df["brca2_mutation"].mean()),
    "ar_v7_pct": float(df["ar_v7_positive"].mean()),
    "msi_high_pct": float(df["msi_high"].mean()),
}

# ---------- Iter 2: biomarker univariate ----------
results["it2"] = {b: chi2_or(df, b) for b in
                  ["brca2_mutation", "ar_v7_positive", "msi_high", "psma_high"]}

# ---------- Iter 3: clinical features ----------
results["it3"] = {
    "ecog_logit": logit_main(df, "ecog_ps"),
    "mcrpc_chi2": chi2_or(df, "mcrpc"),
    "visceral_chi2": chi2_or(df, "visceral_mets"),
    "gleason_logit": logit_main(df, "gleason_score"),
    "age_cont": cont_compare(df, "age_years"),
    "psa_cont": cont_compare(df, "psa_ng_ml"),
}

# ---------- Iter 4: lab univariate ----------
results["it4"] = {
    v: cont_compare(df, v) for v in
    ["albumin_g_dl","ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr",
     "hemoglobin_g_dl","alkaline_phosphatase_u_l","ast_u_l","alt_u_l",
     "total_bilirubin_mg_dl","creatinine_mg_dl","bun_mg_dl",
     "sodium_meq_l","potassium_meq_l","calcium_mg_dl"]
}

# ---------- Iter 5: treatment main effects ----------
trts = ["treatment_enzalutamide","treatment_abiraterone","treatment_docetaxel",
        "treatment_olaparib","treatment_lu177_psma","treatment_pembrolizumab"]
results["it5"] = {t: chi2_or(df, t) for t in trts}

# ---------- Iter 6: olaparib x BRCA2 ----------
results["it6"] = stratified_rd(df, "treatment_olaparib", "brca2_mutation")

# ---------- Iter 7: pembro x MSI-high ----------
results["it7"] = stratified_rd(df, "treatment_pembrolizumab", "msi_high")

# ---------- Iter 8: lu177 x PSMA-high ----------
results["it8"] = stratified_rd(df, "treatment_lu177_psma", "psma_high")

# ---------- Iter 9: enzalutamide / abiraterone x AR-V7 ----------
results["it9_enza"] = stratified_rd(df, "treatment_enzalutamide", "ar_v7_positive")
results["it9_abi"] = stratified_rd(df, "treatment_abiraterone", "ar_v7_positive")

# ---------- Iter 10: ECOG x treatment heterogeneity ----------
results["it10"] = {}
for t in trts:
    inter = logit_inter(df, t, "ecog_ps")
    rd_by = {}
    for e in sorted(df["ecog_ps"].unique()):
        d = df[df["ecog_ps"] == e]
        on = d.loc[d[t] == 1, "objective_response"]
        off = d.loc[d[t] == 0, "objective_response"]
        if len(on) and len(off):
            rd_by[int(e)] = dict(n_on=int(len(on)), n_off=int(len(off)),
                                  p_on=float(on.mean()), p_off=float(off.mean()),
                                  rd=float(on.mean()-off.mean()))
    results["it10"][t] = {"inter": inter, "by_ecog": rd_by}

# ---------- Iter 11: mCRPC x treatment ----------
results["it11"] = {t: stratified_rd(df, t, "mcrpc") for t in trts}

# ---------- Iter 12: visceral x treatment ----------
results["it12"] = {t: stratified_rd(df, t, "visceral_mets") for t in trts}

# ---------- Iter 13: full multivariable logistic ----------
features = (["age_years","ecog_ps","mcrpc","visceral_mets","psa_ng_ml","gleason_score",
             "brca2_mutation","ar_v7_positive","msi_high","psma_high",
             "albumin_g_dl","ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr",
             "hemoglobin_g_dl","alkaline_phosphatase_u_l","total_bilirubin_mg_dl",
             "creatinine_mg_dl","calcium_mg_dl"] + trts)
X = sm.add_constant(df[features].astype(float))
m13 = sm.Logit(df["objective_response"], X).fit(disp=0, maxiter=200)
results["it13"] = {f: dict(coef=float(m13.params[f]), p=float(m13.pvalues[f]),
                            or_=float(np.exp(m13.params[f]))) for f in features}

# ---------- Iter 14: treatment-by-feature interaction screen ----------
binary_mods = ["mcrpc","visceral_mets","brca2_mutation","ar_v7_positive",
               "msi_high","psma_high"]
cont_mods = ["ecog_ps","age_years","albumin_g_dl","ldh_u_l","crp_mg_l","nlr",
             "hemoglobin_g_dl","alkaline_phosphatase_u_l","psa_ng_ml",
             "weight_loss_pct_6mo","gleason_score","calcium_mg_dl"]
screen = []
for t in trts:
    for mod in binary_mods + cont_mods:
        try:
            inter = logit_inter(df, t, mod)
            screen.append(dict(treat=t, mod=mod, **inter))
        except Exception as e:
            screen.append(dict(treat=t, mod=mod, error=str(e)))
screen = sorted(screen, key=lambda r: r.get("inter_p", 1))
results["it14"] = screen[:30]  # top 30 interactions by p-value

# ---------- Iter 15: olaparib + BRCA2 + ECOG/visceral nested subgroup ----------
results["it15"] = {}
for ec in [0,1,2]:
    sub = df[(df["brca2_mutation"]==1) & (df["ecog_ps"]==ec)]
    on = sub.loc[sub["treatment_olaparib"]==1, "objective_response"]
    off = sub.loc[sub["treatment_olaparib"]==0, "objective_response"]
    if len(on) and len(off):
        results["it15"][f"brca2+ecog={ec}"] = dict(
            n_on=int(len(on)), n_off=int(len(off)),
            p_on=float(on.mean()), p_off=float(off.mean()),
            rd=float(on.mean()-off.mean()))
for v in [0,1]:
    sub = df[(df["brca2_mutation"]==1) & (df["visceral_mets"]==v)]
    on = sub.loc[sub["treatment_olaparib"]==1, "objective_response"]
    off = sub.loc[sub["treatment_olaparib"]==0, "objective_response"]
    if len(on) and len(off):
        results["it15"][f"brca2+visc={v}"] = dict(
            n_on=int(len(on)), n_off=int(len(off)),
            p_on=float(on.mean()), p_off=float(off.mean()),
            rd=float(on.mean()-off.mean()))

# ---------- Iter 16: pembro + MSI-high refinement (ECOG, visceral) ----------
results["it16"] = {}
for ec in [0,1,2]:
    sub = df[(df["msi_high"]==1) & (df["ecog_ps"]==ec)]
    on = sub.loc[sub["treatment_pembrolizumab"]==1, "objective_response"]
    off = sub.loc[sub["treatment_pembrolizumab"]==0, "objective_response"]
    if len(on) and len(off):
        results["it16"][f"msi+ecog={ec}"] = dict(
            n_on=int(len(on)), n_off=int(len(off)),
            p_on=float(on.mean()), p_off=float(off.mean()),
            rd=float(on.mean()-off.mean()))
for v in [0,1]:
    sub = df[(df["msi_high"]==1) & (df["visceral_mets"]==v)]
    on = sub.loc[sub["treatment_pembrolizumab"]==1, "objective_response"]
    off = sub.loc[sub["treatment_pembrolizumab"]==0, "objective_response"]
    if len(on) and len(off):
        results["it16"][f"msi+visc={v}"] = dict(
            n_on=int(len(on)), n_off=int(len(off)),
            p_on=float(on.mean()), p_off=float(off.mean()),
            rd=float(on.mean()-off.mean()))

# ---------- Iter 17: lu177 + PSMA-high subgroup refinement ----------
results["it17"] = {}
for ec in [0,1,2]:
    sub = df[(df["psma_high"]==1) & (df["ecog_ps"]==ec)]
    on = sub.loc[sub["treatment_lu177_psma"]==1, "objective_response"]
    off = sub.loc[sub["treatment_lu177_psma"]==0, "objective_response"]
    if len(on) and len(off):
        results["it17"][f"psma+ecog={ec}"] = dict(
            n_on=int(len(on)), n_off=int(len(off)),
            p_on=float(on.mean()), p_off=float(off.mean()),
            rd=float(on.mean()-off.mean()))
for v in [0,1]:
    sub = df[(df["psma_high"]==1) & (df["visceral_mets"]==v)]
    on = sub.loc[sub["treatment_lu177_psma"]==1, "objective_response"]
    off = sub.loc[sub["treatment_lu177_psma"]==0, "objective_response"]
    if len(on) and len(off):
        results["it17"][f"psma+visc={v}"] = dict(
            n_on=int(len(on)), n_off=int(len(off)),
            p_on=float(on.mean()), p_off=float(off.mean()),
            rd=float(on.mean()-off.mean()))
for m in [0,1]:
    sub = df[(df["psma_high"]==1) & (df["mcrpc"]==m)]
    on = sub.loc[sub["treatment_lu177_psma"]==1, "objective_response"]
    off = sub.loc[sub["treatment_lu177_psma"]==0, "objective_response"]
    if len(on) and len(off):
        results["it17"][f"psma+mcrpc={m}"] = dict(
            n_on=int(len(on)), n_off=int(len(off)),
            p_on=float(on.mean()), p_off=float(off.mean()),
            rd=float(on.mean()-off.mean()))

# ---------- Iter 18: enzalutamide AR-V7 negative subgroup refinement ----------
# Enzalutamide overall benefit is huge; check if ar_v7_positive eliminates it.
results["it18"] = {}
for arv in [0,1]:
    sub = df[df["ar_v7_positive"]==arv]
    on = sub.loc[sub["treatment_enzalutamide"]==1, "objective_response"]
    off = sub.loc[sub["treatment_enzalutamide"]==0, "objective_response"]
    results["it18"][f"ar_v7={arv}"] = dict(
        n_on=int(len(on)), n_off=int(len(off)),
        p_on=float(on.mean()), p_off=float(off.mean()),
        rd=float(on.mean()-off.mean()))
# Then examine within ar_v7=0 by ECOG / visceral / mCRPC
for ec in [0,1,2]:
    sub = df[(df["ar_v7_positive"]==0) & (df["ecog_ps"]==ec)]
    on = sub.loc[sub["treatment_enzalutamide"]==1, "objective_response"]
    off = sub.loc[sub["treatment_enzalutamide"]==0, "objective_response"]
    results["it18"][f"arv7=0+ecog={ec}"] = dict(
        n_on=int(len(on)), n_off=int(len(off)),
        p_on=float(on.mean()), p_off=float(off.mean()),
        rd=float(on.mean()-off.mean()))

# ---------- Iter 19: olaparib + BRCA2 three-way (ECOG*visceral) ----------
results["it19"] = {}
sub = df[(df["brca2_mutation"]==1) & (df["ecog_ps"]<=1) & (df["visceral_mets"]==0)]
on = sub.loc[sub["treatment_olaparib"]==1, "objective_response"]
off = sub.loc[sub["treatment_olaparib"]==0, "objective_response"]
results["it19"]["brca2+ecog<=1+no_visc"] = dict(
    n_on=int(len(on)), n_off=int(len(off)),
    p_on=float(on.mean()) if len(on) else None,
    p_off=float(off.mean()) if len(off) else None,
    rd=float(on.mean()-off.mean()) if (len(on) and len(off)) else None)
sub = df[(df["brca2_mutation"]==1) & (df["ecog_ps"]>=2)]
on = sub.loc[sub["treatment_olaparib"]==1, "objective_response"]
off = sub.loc[sub["treatment_olaparib"]==0, "objective_response"]
results["it19"]["brca2+ecog>=2"] = dict(
    n_on=int(len(on)), n_off=int(len(off)),
    p_on=float(on.mean()) if len(on) else None,
    p_off=float(off.mean()) if len(off) else None,
    rd=float(on.mean()-off.mean()) if (len(on) and len(off)) else None)

# ---------- Iter 20: enzalutamide x mCRPC interaction ----------
results["it20"] = {
    "enza_x_mcrpc": stratified_rd(df, "treatment_enzalutamide", "mcrpc"),
    "enza_x_visceral": stratified_rd(df, "treatment_enzalutamide", "visceral_mets"),
}

# ---------- Iter 21: docetaxel/abiraterone heterogeneity by mCRPC, visceral, ECOG ----------
results["it21"] = {}
for t in ["treatment_docetaxel","treatment_abiraterone"]:
    results["it21"][f"{t}_x_mcrpc"] = stratified_rd(df, t, "mcrpc")
    results["it21"][f"{t}_x_visceral"] = stratified_rd(df, t, "visceral_mets")
    inter = logit_inter(df, t, "ecog_ps")
    results["it21"][f"{t}_x_ecog"] = inter

# ---------- Iter 22: lu177 + PSMA-high + ECOG + visceral combined ----------
results["it22"] = {}
for ec in [0,1,2]:
    for v in [0,1]:
        sub = df[(df["psma_high"]==1) & (df["ecog_ps"]==ec) & (df["visceral_mets"]==v)]
        on = sub.loc[sub["treatment_lu177_psma"]==1, "objective_response"]
        off = sub.loc[sub["treatment_lu177_psma"]==0, "objective_response"]
        if len(on) and len(off):
            results["it22"][f"psma+ecog={ec}+visc={v}"] = dict(
                n_on=int(len(on)), n_off=int(len(off)),
                p_on=float(on.mean()), p_off=float(off.mean()),
                rd=float(on.mean()-off.mean()))

# ---------- Iter 23: pembrolizumab + MSI-high + ECOG + visceral combined ----------
results["it23"] = {}
for ec in [0,1,2]:
    for v in [0,1]:
        sub = df[(df["msi_high"]==1) & (df["ecog_ps"]==ec) & (df["visceral_mets"]==v)]
        on = sub.loc[sub["treatment_pembrolizumab"]==1, "objective_response"]
        off = sub.loc[sub["treatment_pembrolizumab"]==0, "objective_response"]
        if len(on) and len(off):
            results["it23"][f"msi+ecog={ec}+visc={v}"] = dict(
                n_on=int(len(on)), n_off=int(len(off)),
                p_on=float(on.mean()), p_off=float(off.mean()),
                rd=float(on.mean()-off.mean()))

# ---------- Iter 24: olaparib + BRCA2 + ECOG + visceral combined ----------
results["it24"] = {}
for ec in [0,1,2]:
    for v in [0,1]:
        sub = df[(df["brca2_mutation"]==1) & (df["ecog_ps"]==ec) & (df["visceral_mets"]==v)]
        on = sub.loc[sub["treatment_olaparib"]==1, "objective_response"]
        off = sub.loc[sub["treatment_olaparib"]==0, "objective_response"]
        if len(on) and len(off):
            results["it24"][f"brca2+ecog={ec}+visc={v}"] = dict(
                n_on=int(len(on)), n_off=int(len(off)),
                p_on=float(on.mean()), p_off=float(off.mean()),
                rd=float(on.mean()-off.mean()))

# ---------- Iter 25: tree-based subgroup discovery for each treatment ----------
results["it25"] = {}
for t in trts:
    feat_cols = [c for c in df.columns if c not in ("patient_id","objective_response") + tuple(trts)]
    X = df[feat_cols].astype(float).copy()
    X["__t"] = df[t].astype(float)
    # Estimate treatment-specific response rate by features via interactions in shallow tree
    # Use response as target on treated subset and untreated subset, then find largest gap subgroup.
    treated = df[df[t]==1]
    untreated = df[df[t]==0]
    if len(treated) < 50:
        continue
    feat_only = [c for c in feat_cols]
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=0)
    tree.fit(treated[feat_only], treated["objective_response"])
    treated_leaf_pred = tree.predict_proba(treated[feat_only])[:,1]
    untreated_leaf_pred = tree.predict_proba(untreated[feat_only])[:,1]
    leaf_t = tree.apply(treated[feat_only])
    leaf_u = tree.apply(untreated[feat_only])
    leaves = sorted(set(leaf_t))
    rows = []
    for L in leaves:
        on = treated.loc[leaf_t==L, "objective_response"]
        off = untreated.loc[leaf_u==L, "objective_response"]
        if len(off) >= 50 and len(on) >= 50:
            rows.append(dict(leaf=int(L), n_on=int(len(on)), n_off=int(len(off)),
                             p_on=float(on.mean()), p_off=float(off.mean()),
                             rd=float(on.mean()-off.mean())))
    rows = sorted(rows, key=lambda r: -r["rd"])
    results["it25"][t] = rows[:5]

print(json.dumps(results, indent=1, default=str))
