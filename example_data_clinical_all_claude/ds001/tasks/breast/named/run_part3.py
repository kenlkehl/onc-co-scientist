"""Iterations 16-25: heterogeneity screening and final subgroup hypotheses."""
import json
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

df = pd.read_parquet("dataset.parquet")
OUTCOME = "pfs_months"

with open("_iters_1_15.pkl", "rb") as f:
    iterations = pickle.load(f)


def add_iter(idx, hypotheses, analyses):
    iterations.append({"index": idx, "proposed_hypotheses": hypotheses, "analyses": analyses})


def fit_with(df_, formula_cols, outcome=OUTCOME):
    X = sm.add_constant(df_[formula_cols])
    return sm.OLS(df_[outcome], X).fit()


TREATMENTS = [
    "treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
    "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
]
BINARY_MODS = [
    "sex_female", "stage_iv", "has_brain_mets", "node_positive", "postmenopausal",
    "er_positive", "pr_positive", "her2_positive", "her2_low",
    "brca1_mutation", "brca2_mutation", "pik3ca_mutation",
]
CONT_MODS = [
    "age_years", "ecog_ps", "ki67_pct", "tumor_size_cm", "albumin_g_dl",
    "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr",
    "hemoglobin_g_dl", "alkaline_phosphatase_u_l",
]


def screen_interactions(tx, mods_binary, mods_cont):
    """For one treatment, scan single-modifier interactions."""
    rows = []
    for m_ in mods_binary + mods_cont:
        df_ = df.copy()
        df_["i"] = df_[tx] * df_[m_]
        try:
            mod = fit_with(df_, [tx, m_, "i"])
            coef = float(mod.params["i"]); p = float(mod.pvalues["i"])
            rows.append((m_, coef, p))
        except Exception as e:
            rows.append((m_, float("nan"), 1.0))
    return rows


# Iter 16: Palbociclib triple interaction (ER+ × PIK3CA wt × palbo)
hypotheses = []; analyses = []
hypotheses.append({"id":"h16_palbo_triple","text":"The pfs benefit of treatment_palbociclib is largest in patients who are er_positive==1 AND pik3ca_mutation==0 (canonical palbo-sensitive subgroup), reflecting a positive triple interaction palbo×er_positive×(1-pik3ca).","kind":"refined"})
df_ = df.copy()
df_["er_pos"] = df_["er_positive"]
df_["pik_wt"] = (1 - df_["pik3ca_mutation"]).astype(int)
df_["t_e"] = df_["treatment_palbociclib"] * df_["er_pos"]
df_["t_p"] = df_["treatment_palbociclib"] * df_["pik_wt"]
df_["e_p"] = df_["er_pos"] * df_["pik_wt"]
df_["t_e_p"] = df_["treatment_palbociclib"] * df_["er_pos"] * df_["pik_wt"]
m = fit_with(df_, ["treatment_palbociclib","er_pos","pik_wt","t_e","t_p","e_p","t_e_p"])
coef = float(m.params["t_e_p"]); p = float(m.pvalues["t_e_p"])
analyses.append({"hypothesis_ids":["h16_palbo_triple"],"code":"OLS pfs ~ palbo + er_pos + pik_wt + 2-way + 3-way (palbo:er_pos:pik_wt)","result_summary":f"Triple interaction palbo×er_positive×pik3ca_wt = {coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})

# direct subgroup: ER+ AND PIK3CA wt
df_["sub_palbo"] = ((df_["er_positive"] == 1) & (df_["pik3ca_mutation"] == 0)).astype(int)
sub = df_[df_["sub_palbo"] == 1]
g1, g0 = sub.loc[sub["treatment_palbociclib"] == 1, OUTCOME], sub.loc[sub["treatment_palbociclib"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
hypotheses.append({"id":"h16_sub_palbo","text":"In the ER+/PIK3CA-wt subgroup, treatment_palbociclib produces a substantially larger pfs_months benefit than in the complement.","kind":"refined"})
analyses.append({"hypothesis_ids":["h16_sub_palbo"],"code":"Welch t-test pfs by palbo within er_positive==1 AND pik3ca_mutation==0","result_summary":f"ER+/PIK3CA-wt: palbo pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

sub = df_[df_["sub_palbo"] == 0]
g1, g0 = sub.loc[sub["treatment_palbociclib"] == 1, OUTCOME], sub.loc[sub["treatment_palbociclib"] == 0, OUTCOME]
eff2 = float(g1.mean() - g0.mean()); t, p2 = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h16_sub_palbo"],"code":"Welch t-test pfs by palbo OUTSIDE ER+/PIK3CA-wt","result_summary":f"Outside subgroup: palbo pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff2:+.3f}, p={p2:.3g}","p_value":float(p2),"effect_estimate":eff2,"significant":bool(p2<0.05)})
add_iter(16, hypotheses, analyses)


# Iter 17: Palbociclib full canonical subgroup ER+/HER2-/PIK3CA-wt
hypotheses = []; analyses = []
hypotheses.append({"id":"h17_palbo_canonical","text":"In ER+/HER2-/PIK3CA-wt patients, treatment_palbociclib produces the largest pfs_months benefit; outside this subgroup the benefit is small or absent.","kind":"refined"})
df_ = df.copy()
df_["palbo_sub"] = ((df_["er_positive"] == 1) & (df_["her2_positive"] == 0) & (df_["pik3ca_mutation"] == 0)).astype(int)
sub = df_[df_["palbo_sub"] == 1]
g1, g0 = sub.loc[sub["treatment_palbociclib"] == 1, OUTCOME], sub.loc[sub["treatment_palbociclib"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h17_palbo_canonical"],"code":"Welch t-test pfs by palbo within er_positive==1 AND her2_positive==0 AND pik3ca_mutation==0","result_summary":f"ER+/HER2-/PIK3CA-wt: palbo pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

sub = df_[df_["palbo_sub"] == 0]
g1, g0 = sub.loc[sub["treatment_palbociclib"] == 1, OUTCOME], sub.loc[sub["treatment_palbociclib"] == 0, OUTCOME]
eff2 = float(g1.mean() - g0.mean()); t, p2 = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h17_palbo_canonical"],"code":"Welch t-test pfs by palbo OUTSIDE ER+/HER2-/PIK3CA-wt","result_summary":f"Outside ER+/HER2-/PIK3CA-wt: palbo pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff2:+.3f}, p={p2:.3g}","p_value":float(p2),"effect_estimate":eff2,"significant":bool(p2<0.05)})

# Subgroup membership × treatment interaction
df_["i"] = df_["treatment_palbociclib"] * df_["palbo_sub"]
m = fit_with(df_, ["treatment_palbociclib","palbo_sub","i"])
coef = float(m.params["i"]); p = float(m.pvalues["i"])
analyses.append({"hypothesis_ids":["h17_palbo_canonical"],"code":"OLS pfs ~ palbo + sub + palbo:sub","result_summary":f"Interaction palbo×(ER+/HER2-/PIK3CA-wt) = {coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})
add_iter(17, hypotheses, analyses)


# Iter 18: Olaparib heterogeneity screen
hypotheses = []; analyses = []
tx = "treatment_olaparib"
hypotheses.append({"id":"h18_olap_screen","text":"treatment_olaparib effect on pfs_months is modified by patient features; we screen all candidate modifiers for interactions.","kind":"novel"})
rows = screen_interactions(tx, BINARY_MODS, CONT_MODS)
sig_rows = [r for r in rows if r[2] < 0.05]
top = sorted(rows, key=lambda x: x[2])[:5]
analyses.append({"hypothesis_ids":["h18_olap_screen"],"code":f"For each modifier m: OLS pfs ~ {tx} + m + {tx}:m; report all p<0.05","result_summary":f"Olaparib interaction screen: significant modifiers ({len(sig_rows)}/{len(rows)}): " + "; ".join(f"{m_}: coef={c:+.4f},p={p:.3g}" for m_,c,p in sig_rows) + " | top 5 by p: " + "; ".join(f"{m_}:p={p:.3g}" for m_,c,p in top),"p_value":(top[0][2] if top else None),"effect_estimate":(top[0][1] if top else None),"significant":bool(top and top[0][2]<0.05)})

# Final olaparib subgroup hypothesis: BRCA mutated
df_ = df.copy(); df_["any_brca"] = ((df_["brca1_mutation"] == 1) | (df_["brca2_mutation"] == 1)).astype(int)
sub = df_[df_["any_brca"] == 1]
g1, g0 = sub.loc[sub["treatment_olaparib"] == 1, OUTCOME], sub.loc[sub["treatment_olaparib"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
hypotheses.append({"id":"h18_olap_brca_final","text":"treatment_olaparib's pfs benefit is concentrated in patients with brca1_mutation==1 OR brca2_mutation==1.","kind":"refined"})
analyses.append({"hypothesis_ids":["h18_olap_brca_final"],"code":"Welch t-test pfs by olaparib within any_brca==1","result_summary":f"BRCA-mutated subgroup: olaparib pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})
add_iter(18, hypotheses, analyses)


# Iter 19: Pembrolizumab heterogeneity screen
hypotheses = []; analyses = []
tx = "treatment_pembrolizumab"
hypotheses.append({"id":"h19_pemb_screen","text":"treatment_pembrolizumab effect on pfs is modified by patient features; we screen all candidates.","kind":"novel"})
rows = screen_interactions(tx, BINARY_MODS, CONT_MODS)
sig_rows = [r for r in rows if r[2] < 0.05]
top = sorted(rows, key=lambda x: x[2])[:5]
analyses.append({"hypothesis_ids":["h19_pemb_screen"],"code":f"For each modifier m: OLS pfs ~ {tx} + m + {tx}:m","result_summary":f"Pembro interaction screen: significant modifiers ({len(sig_rows)}/{len(rows)}): " + "; ".join(f"{m_}: coef={c:+.4f},p={p:.3g}" for m_,c,p in sig_rows) + " | top 5 by p: " + "; ".join(f"{m_}:p={p:.3g}" for m_,c,p in top),"p_value":(top[0][2] if top else None),"effect_estimate":(top[0][1] if top else None),"significant":bool(top and top[0][2]<0.05)})
add_iter(19, hypotheses, analyses)


# Iter 20: Trastuzumab heterogeneity screen
hypotheses = []; analyses = []
tx = "treatment_trastuzumab"
hypotheses.append({"id":"h20_t_screen","text":"treatment_trastuzumab effect on pfs is modified by patient features; we screen all candidates.","kind":"novel"})
rows = screen_interactions(tx, BINARY_MODS, CONT_MODS)
sig_rows = [r for r in rows if r[2] < 0.05]
top = sorted(rows, key=lambda x: x[2])[:5]
analyses.append({"hypothesis_ids":["h20_t_screen"],"code":f"For each modifier m: OLS pfs ~ {tx} + m + {tx}:m","result_summary":f"Trastuzumab interaction screen: significant modifiers ({len(sig_rows)}/{len(rows)}): " + "; ".join(f"{m_}: coef={c:+.4f},p={p:.3g}" for m_,c,p in sig_rows) + " | top 5 by p: " + "; ".join(f"{m_}:p={p:.3g}" for m_,c,p in top),"p_value":(top[0][2] if top else None),"effect_estimate":(top[0][1] if top else None),"significant":bool(top and top[0][2]<0.05)})
add_iter(20, hypotheses, analyses)


# Iter 21: Sacituzumab heterogeneity screen
hypotheses = []; analyses = []
tx = "treatment_sacituzumab_govitecan"
hypotheses.append({"id":"h21_saci_screen","text":"treatment_sacituzumab_govitecan effect on pfs is modified by patient features; we screen all candidates.","kind":"novel"})
rows = screen_interactions(tx, BINARY_MODS, CONT_MODS)
sig_rows = [r for r in rows if r[2] < 0.05]
top = sorted(rows, key=lambda x: x[2])[:5]
analyses.append({"hypothesis_ids":["h21_saci_screen"],"code":f"For each modifier m: OLS pfs ~ {tx} + m + {tx}:m","result_summary":f"Sacituzumab interaction screen: significant modifiers ({len(sig_rows)}/{len(rows)}): " + "; ".join(f"{m_}: coef={c:+.4f},p={p:.3g}" for m_,c,p in sig_rows) + " | top 5 by p: " + "; ".join(f"{m_}:p={p:.3g}" for m_,c,p in top),"p_value":(top[0][2] if top else None),"effect_estimate":(top[0][1] if top else None),"significant":bool(top and top[0][2]<0.05)})
add_iter(21, hypotheses, analyses)


# Iter 22: Tamoxifen heterogeneity screen
hypotheses = []; analyses = []
tx = "treatment_tamoxifen"
hypotheses.append({"id":"h22_tam_screen","text":"treatment_tamoxifen effect on pfs is modified by patient features; we screen all candidates.","kind":"novel"})
rows = screen_interactions(tx, BINARY_MODS, CONT_MODS)
sig_rows = [r for r in rows if r[2] < 0.05]
top = sorted(rows, key=lambda x: x[2])[:5]
analyses.append({"hypothesis_ids":["h22_tam_screen"],"code":f"For each modifier m: OLS pfs ~ {tx} + m + {tx}:m","result_summary":f"Tamoxifen interaction screen: significant modifiers ({len(sig_rows)}/{len(rows)}): " + "; ".join(f"{m_}: coef={c:+.4f},p={p:.3g}" for m_,c,p in sig_rows) + " | top 5 by p: " + "; ".join(f"{m_}:p={p:.3g}" for m_,c,p in top),"p_value":(top[0][2] if top else None),"effect_estimate":(top[0][1] if top else None),"significant":bool(top and top[0][2]<0.05)})
add_iter(22, hypotheses, analyses)


# Iter 23: Joint model with key palbociclib interactions
hypotheses = []; analyses = []
hypotheses.append({"id":"h23_joint_palbo","text":"In a joint model containing palbociclib×er_positive and palbociclib×pik3ca_mutation simultaneously, both interactions remain large and significant: palbo benefits ER+ patients, and PIK3CA mutation suppresses the palbo benefit.","kind":"refined"})
df_ = df.copy()
df_["t_e"] = df_["treatment_palbociclib"] * df_["er_positive"]
df_["t_p"] = df_["treatment_palbociclib"] * df_["pik3ca_mutation"]
m = fit_with(df_, ["treatment_palbociclib","er_positive","pik3ca_mutation","t_e","t_p"])
coef_e = float(m.params["t_e"]); p_e = float(m.pvalues["t_e"])
coef_p = float(m.params["t_p"]); p_p = float(m.pvalues["t_p"])
coef_main = float(m.params["treatment_palbociclib"]); p_main = float(m.pvalues["treatment_palbociclib"])
analyses.append({"hypothesis_ids":["h23_joint_palbo"],"code":"OLS pfs ~ palbo + er_pos + pik3ca + palbo:er_pos + palbo:pik3ca","result_summary":f"Joint model: palbo main={coef_main:+.3f} (p={p_main:.3g}); palbo×er_positive={coef_e:+.3f} (p={p_e:.3g}); palbo×pik3ca={coef_p:+.3f} (p={p_p:.3g})","p_value":max(p_e, p_p),"effect_estimate":coef_e,"significant":bool(p_e<0.05 and p_p<0.05)})
add_iter(23, hypotheses, analyses)


# Iter 24: Final palbociclib subgroup — quantify benefit & sub-subgroup arithmetic
hypotheses = []; analyses = []
hypotheses.append({"id":"h24_palbo_final","text":"The maximally enriched palbociclib-benefit subgroup is er_positive==1 AND her2_positive==0 AND pik3ca_mutation==0; within this subgroup the pfs benefit of treatment_palbociclib exceeds 2 months, while in the complement the benefit is approximately zero.","kind":"refined"})
df_ = df.copy()
df_["sub"] = ((df_["er_positive"] == 1) & (df_["her2_positive"] == 0) & (df_["pik3ca_mutation"] == 0)).astype(int)

# Inside subgroup
sub = df_[df_["sub"] == 1]
g1, g0 = sub.loc[sub["treatment_palbociclib"] == 1, OUTCOME], sub.loc[sub["treatment_palbociclib"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h24_palbo_final"],"code":"Welch t-test pfs by palbo within ER+/HER2-/PIK3CA-wt","result_summary":f"ER+/HER2-/PIK3CA-wt (n={len(sub)}): palbo pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

# Outside subgroup
sub2 = df_[df_["sub"] == 0]
g1, g0 = sub2.loc[sub2["treatment_palbociclib"] == 1, OUTCOME], sub2.loc[sub2["treatment_palbociclib"] == 0, OUTCOME]
eff2 = float(g1.mean() - g0.mean()); t, p2 = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h24_palbo_final"],"code":"Welch t-test pfs by palbo OUTSIDE ER+/HER2-/PIK3CA-wt","result_summary":f"Outside subgroup (n={len(sub2)}): palbo pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff2:+.3f}, p={p2:.3g}","p_value":float(p2),"effect_estimate":eff2,"significant":bool(p2<0.05)})

# Single-component drop tests: confirm each predicate matters
for drop in ["er_positive_drop", "her2_positive_drop", "pik3ca_drop"]:
    if drop == "er_positive_drop":
        # require ER+ to be 0 instead
        mask = (df_["er_positive"] == 0) & (df_["her2_positive"] == 0) & (df_["pik3ca_mutation"] == 0)
        label = "ER-/HER2-/PIK3CA-wt"
    elif drop == "her2_positive_drop":
        mask = (df_["er_positive"] == 1) & (df_["her2_positive"] == 1) & (df_["pik3ca_mutation"] == 0)
        label = "ER+/HER2+/PIK3CA-wt"
    else:
        mask = (df_["er_positive"] == 1) & (df_["her2_positive"] == 0) & (df_["pik3ca_mutation"] == 1)
        label = "ER+/HER2-/PIK3CA-mut"
    sub_d = df_[mask]
    g1, g0 = sub_d.loc[sub_d["treatment_palbociclib"] == 1, OUTCOME], sub_d.loc[sub_d["treatment_palbociclib"] == 0, OUTCOME]
    if len(g1) > 5 and len(g0) > 5:
        eff_d = float(g1.mean() - g0.mean()); t, p_d = stats.ttest_ind(g1, g0, equal_var=False)
        analyses.append({"hypothesis_ids":["h24_palbo_final"],"code":f"Welch t-test pfs by palbo within {label}","result_summary":f"{label} (n={len(sub_d)}): palbo pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff_d:+.3f}, p={p_d:.3g}","p_value":float(p_d),"effect_estimate":eff_d,"significant":bool(p_d<0.05)})
add_iter(24, hypotheses, analyses)


# Iter 25: Final per-treatment best subgroup hypotheses
hypotheses = []; analyses = []

# (a) Palbociclib already covered — restated
hypotheses.append({"id":"h25_palbo","text":"FINAL: treatment_palbociclib prolongs pfs_months substantially in patients with er_positive==1 AND her2_positive==0 AND pik3ca_mutation==0; outside this subgroup the effect is null. The PIK3CA-mutated state suppresses palbociclib benefit.","kind":"refined"})

# (b) Olaparib
hypotheses.append({"id":"h25_olap","text":"FINAL: treatment_olaparib prolongs pfs_months in BRCA1- or BRCA2-mutated patients (any_brca==1); the effect is absent in BRCA wild-type.","kind":"refined"})

# (c) Pembrolizumab — check if benefit concentrates in TNBC-like with high PD-L1 surrogates (we don't have PD-L1; fall back on TNBC + something)
# Try interactions with other features within TNBC
df_ = df.copy(); df_["tnbc"] = ((df_["er_positive"] == 0) & (df_["her2_positive"] == 0)).astype(int)
sub = df_[df_["tnbc"] == 1]
# screen modifiers within TNBC
in_rows = []
for m_ in BINARY_MODS:
    if m_ in ("er_positive","her2_positive"):
        continue
    s = sub.copy(); s["i"] = s["treatment_pembrolizumab"] * s[m_]
    try:
        Xs = sm.add_constant(s[["treatment_pembrolizumab", m_, "i"]])
        mod = sm.OLS(s[OUTCOME], Xs).fit()
        in_rows.append((m_, float(mod.params["i"]), float(mod.pvalues["i"])))
    except Exception:
        pass
in_rows.sort(key=lambda x: x[2])
top_in = in_rows[:3]
hypotheses.append({"id":"h25_pemb","text":"FINAL: treatment_pembrolizumab prolongs pfs_months modestly in the TNBC-like subgroup (er_positive==0 AND her2_positive==0); the effect is absent or null elsewhere.","kind":"refined"})
analyses.append({"hypothesis_ids":["h25_pemb"],"code":"Within TNBC: screen pembro × each modifier","result_summary":f"Within-TNBC pembro × modifier top hits: " + "; ".join(f"{m_}:coef={c:+.3f},p={p:.3g}" for m_,c,p in top_in),"p_value":(top_in[0][2] if top_in else None),"effect_estimate":(top_in[0][1] if top_in else None),"significant":bool(top_in and top_in[0][2]<0.05)})

# Direct test: pembro in TNBC again, with adjusted model
sub = df_[df_["tnbc"] == 1]
g1, g0 = sub.loc[sub["treatment_pembrolizumab"] == 1, OUTCOME], sub.loc[sub["treatment_pembrolizumab"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h25_pemb"],"code":"Welch t-test pfs by pembro within TNBC (final subgroup test)","result_summary":f"TNBC subgroup (n={len(sub)}): pembro pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

# Olaparib subgroup test (recap, marked final)
df_b = df.copy(); df_b["any_brca"] = ((df_b["brca1_mutation"] == 1) | (df_b["brca2_mutation"] == 1)).astype(int)
sub_b = df_b[df_b["any_brca"] == 1]
g1, g0 = sub_b.loc[sub_b["treatment_olaparib"] == 1, OUTCOME], sub_b.loc[sub_b["treatment_olaparib"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h25_olap"],"code":"Welch t-test pfs by olaparib within any_brca==1","result_summary":f"BRCA-mutated subgroup (n={len(sub_b)}): olaparib pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

# Palbo subgroup test (recap, marked final)
df_p = df.copy()
df_p["sub"] = ((df_p["er_positive"] == 1) & (df_p["her2_positive"] == 0) & (df_p["pik3ca_mutation"] == 0)).astype(int)
sub_p = df_p[df_p["sub"] == 1]
g1, g0 = sub_p.loc[sub_p["treatment_palbociclib"] == 1, OUTCOME], sub_p.loc[sub_p["treatment_palbociclib"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h25_palbo"],"code":"Welch t-test pfs by palbo within ER+/HER2-/PIK3CA-wt (final subgroup test)","result_summary":f"ER+/HER2-/PIK3CA-wt (n={len(sub_p)}): palbo pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

# Tamoxifen: no convincing subgroup → conservative "null" final hypothesis
hypotheses.append({"id":"h25_tam","text":"FINAL: treatment_tamoxifen does not produce a clinically meaningful pfs_months benefit in any tested subgroup of this cohort, including ER-positive patients.","kind":"refined"})
sub = df[df["er_positive"] == 1]
g1, g0 = sub.loc[sub["treatment_tamoxifen"] == 1, OUTCOME], sub.loc[sub["treatment_tamoxifen"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h25_tam"],"code":"Welch t-test pfs by tamoxifen within er_positive==1 (final)","result_summary":f"ER+: tamoxifen pfs diff={eff:+.3f}, p={p:.3g} (n={len(sub)})","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

# Trastuzumab: no convincing subgroup → null
hypotheses.append({"id":"h25_t","text":"FINAL: treatment_trastuzumab does not produce a measurable pfs_months benefit in any tested subgroup, including HER2-positive patients (which is unexpected biologically and likely reflects this dataset's structure).","kind":"refined"})
sub = df[df["her2_positive"] == 1]
g1, g0 = sub.loc[sub["treatment_trastuzumab"] == 1, OUTCOME], sub.loc[sub["treatment_trastuzumab"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h25_t"],"code":"Welch t-test pfs by trastuzumab within her2_positive==1 (final)","result_summary":f"HER2+: trastuzumab pfs diff={eff:+.3f}, p={p:.3g} (n={len(sub)})","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

# Sacituzumab: no convincing subgroup → null
hypotheses.append({"id":"h25_saci","text":"FINAL: treatment_sacituzumab_govitecan does not produce a measurable pfs_months benefit in tested subgroups including her2_low==1 patients.","kind":"refined"})
sub = df[df["her2_low"] == 1]
g1, g0 = sub.loc[sub["treatment_sacituzumab_govitecan"] == 1, OUTCOME], sub.loc[sub["treatment_sacituzumab_govitecan"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h25_saci"],"code":"Welch t-test pfs by sacituzumab within her2_low==1 (final)","result_summary":f"HER2-low: saci pfs diff={eff:+.3f}, p={p:.3g} (n={len(sub)})","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

add_iter(25, hypotheses, analyses)


with open("_iters_1_25.pkl", "wb") as f:
    pickle.dump(iterations, f)

print(f"All iterations done: {len(iterations)}")
for it in iterations[15:]:
    print(f"-- Iter {it['index']} --")
    for a in it['analyses']:
        print(" ", a['result_summary'][:260])
