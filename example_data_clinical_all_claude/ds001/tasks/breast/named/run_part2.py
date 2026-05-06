"""Iterations 6-15: adjusted models and biomarker-treatment alignment tests."""
import json
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

df = pd.read_parquet("dataset.parquet")
OUTCOME = "pfs_months"

with open("_iters_1_5.pkl", "rb") as f:
    iterations = pickle.load(f)


def add_iter(idx, hypotheses, analyses):
    iterations.append({"index": idx, "proposed_hypotheses": hypotheses, "analyses": analyses})


CONF = ["age_years", "ecog_ps", "stage_iv", "has_brain_mets",
        "albumin_g_dl", "weight_loss_pct_6mo", "ki67_pct",
        "er_positive", "pr_positive", "her2_positive", "her2_low",
        "brca1_mutation", "brca2_mutation", "pik3ca_mutation"]

TREATMENTS = [
    "treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
    "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
]


def fit_with(df_, formula_cols, outcome=OUTCOME):
    X = sm.add_constant(df_[formula_cols])
    return sm.OLS(df_[outcome], X).fit()


# Iter 6: Adjusted treatment effects
hypotheses = []
analyses = []
for tx in TREATMENTS:
    h_id = f"h6_{tx}"
    hypotheses.append({
        "id": h_id,
        "text": f"After adjustment for stage, ECOG, age, key biomarkers, and labs, "
                f"{tx} remains associated with pfs_months.",
        "kind": "refined",
    })
    cols = [tx] + CONF
    model = fit_with(df, cols)
    coef = float(model.params[tx])
    p = float(model.pvalues[tx])
    sig = bool(p < 0.05)
    summ = (f"Adjusted OLS pfs_months ~ {tx} + {len(CONF)} covariates "
            f"(age, ecog, stage_iv, brain_mets, albumin, weight_loss, ki67, ER/PR/HER2/HER2-low, BRCA1/2, PIK3CA): "
            f"coef on {tx} = {coef:+.4f} months, p={p:.3g}")
    analyses.append({
        "hypothesis_ids": [h_id],
        "code": f"OLS {OUTCOME} ~ {tx} + {len(CONF)} confounders",
        "result_summary": summ,
        "p_value": p,
        "effect_estimate": coef,
        "significant": sig,
    })
add_iter(6, hypotheses, analyses)


# Iter 7: Tamoxifen × ER
hypotheses = []
analyses = []
hypotheses.append({"id":"h7_tam_er_in_ER","text":"Among ER-positive patients, treatment_tamoxifen prolongs pfs_months.","kind":"novel"})
hypotheses.append({"id":"h7_tam_er_in_ERneg","text":"Among ER-negative patients, treatment_tamoxifen does not prolong pfs_months.","kind":"novel"})
hypotheses.append({"id":"h7_tam_er_intx","text":"There is a positive treatment_tamoxifen × er_positive interaction on pfs_months.","kind":"novel"})

sub = df[df["er_positive"] == 1]
g1, g0 = sub.loc[sub["treatment_tamoxifen"] == 1, OUTCOME], sub.loc[sub["treatment_tamoxifen"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h7_tam_er_in_ER"],"code":"Welch t-test pfs by tamoxifen within er_positive==1","result_summary":f"In ER+: tamoxifen pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

sub = df[df["er_positive"] == 0]
g1, g0 = sub.loc[sub["treatment_tamoxifen"] == 1, OUTCOME], sub.loc[sub["treatment_tamoxifen"] == 0, OUTCOME]
eff2 = float(g1.mean() - g0.mean()); t, p2 = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h7_tam_er_in_ERneg"],"code":"Welch t-test pfs by tamoxifen within er_positive==0","result_summary":f"In ER-: tamoxifen pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff2:+.3f}, p={p2:.3g}","p_value":float(p2),"effect_estimate":eff2,"significant":bool(p2<0.05)})

df_ = df.copy(); df_["tx_er"] = df_["treatment_tamoxifen"] * df_["er_positive"]
m = fit_with(df_, ["treatment_tamoxifen", "er_positive", "tx_er"])
coef = float(m.params["tx_er"]); p = float(m.pvalues["tx_er"])
analyses.append({"hypothesis_ids":["h7_tam_er_intx"],"code":"OLS pfs ~ tamoxifen + er_positive + tamoxifen:er_positive","result_summary":f"Interaction tamoxifen×er_positive = {coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})
add_iter(7, hypotheses, analyses)


# Iter 8: Trastuzumab × HER2
hypotheses = []; analyses = []
hypotheses.append({"id":"h8_t_her2_in_pos","text":"Among HER2-positive patients, treatment_trastuzumab prolongs pfs_months.","kind":"novel"})
hypotheses.append({"id":"h8_t_her2_in_neg","text":"Among HER2-negative patients, treatment_trastuzumab does not prolong pfs_months.","kind":"novel"})
hypotheses.append({"id":"h8_t_her2_intx","text":"treatment_trastuzumab × her2_positive interaction on pfs is positive.","kind":"novel"})

sub = df[df["her2_positive"] == 1]
g1, g0 = sub.loc[sub["treatment_trastuzumab"] == 1, OUTCOME], sub.loc[sub["treatment_trastuzumab"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h8_t_her2_in_pos"],"code":"Welch t-test pfs by trastuzumab in HER2+","result_summary":f"HER2+: trastuzumab pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

sub = df[df["her2_positive"] == 0]
g1, g0 = sub.loc[sub["treatment_trastuzumab"] == 1, OUTCOME], sub.loc[sub["treatment_trastuzumab"] == 0, OUTCOME]
eff2 = float(g1.mean() - g0.mean()); t, p2 = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h8_t_her2_in_neg"],"code":"Welch t-test pfs by trastuzumab in HER2-","result_summary":f"HER2-: trastuzumab pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff2:+.3f}, p={p2:.3g}","p_value":float(p2),"effect_estimate":eff2,"significant":bool(p2<0.05)})

df_ = df.copy(); df_["tx_h2"] = df_["treatment_trastuzumab"] * df_["her2_positive"]
m = fit_with(df_, ["treatment_trastuzumab", "her2_positive", "tx_h2"])
coef = float(m.params["tx_h2"]); p = float(m.pvalues["tx_h2"])
analyses.append({"hypothesis_ids":["h8_t_her2_intx"],"code":"OLS pfs ~ trastuzumab + her2_positive + trastuzumab:her2_positive","result_summary":f"Interaction trastuzumab×her2_positive={coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})
add_iter(8, hypotheses, analyses)


# Iter 9: Olaparib × any-BRCA
hypotheses = []; analyses = []
hypotheses.append({"id":"h9_olap_in_brca","text":"Among patients with brca1_mutation==1 OR brca2_mutation==1, treatment_olaparib prolongs pfs_months.","kind":"novel"})
hypotheses.append({"id":"h9_olap_in_nobrca","text":"Among patients with neither BRCA mutation, treatment_olaparib does not prolong pfs_months.","kind":"novel"})
hypotheses.append({"id":"h9_olap_intx","text":"treatment_olaparib × any_brca interaction on pfs is positive.","kind":"novel"})

df_ = df.copy()
df_["any_brca"] = ((df_["brca1_mutation"] == 1) | (df_["brca2_mutation"] == 1)).astype(int)
sub = df_[df_["any_brca"] == 1]
g1, g0 = sub.loc[sub["treatment_olaparib"] == 1, OUTCOME], sub.loc[sub["treatment_olaparib"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h9_olap_in_brca"],"code":"Welch t-test pfs by olaparib in BRCA1/2","result_summary":f"BRCA+: olaparib pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

sub = df_[df_["any_brca"] == 0]
g1, g0 = sub.loc[sub["treatment_olaparib"] == 1, OUTCOME], sub.loc[sub["treatment_olaparib"] == 0, OUTCOME]
eff2 = float(g1.mean() - g0.mean()); t, p2 = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h9_olap_in_nobrca"],"code":"Welch t-test pfs by olaparib in BRCA wt","result_summary":f"BRCA wt: olaparib pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff2:+.3f}, p={p2:.3g}","p_value":float(p2),"effect_estimate":eff2,"significant":bool(p2<0.05)})

df_["tx_b"] = df_["treatment_olaparib"] * df_["any_brca"]
m = fit_with(df_, ["treatment_olaparib", "any_brca", "tx_b"])
coef = float(m.params["tx_b"]); p = float(m.pvalues["tx_b"])
analyses.append({"hypothesis_ids":["h9_olap_intx"],"code":"OLS pfs ~ olaparib + any_brca + olaparib:any_brca","result_summary":f"Interaction olaparib×any_brca={coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})
add_iter(9, hypotheses, analyses)


# Iter 10: Sacituzumab × HER2-low; Pembrolizumab × TNBC
hypotheses = []; analyses = []
hypotheses.append({"id":"h10_sac_low","text":"Among her2_low==1 patients, treatment_sacituzumab_govitecan prolongs pfs_months.","kind":"novel"})
hypotheses.append({"id":"h10_sac_intx","text":"treatment_sacituzumab_govitecan × her2_low interaction on pfs is positive.","kind":"novel"})
hypotheses.append({"id":"h10_pemb_tnbc","text":"Among TNBC-like (er_positive==0 AND her2_positive==0) patients, treatment_pembrolizumab prolongs pfs_months.","kind":"novel"})
hypotheses.append({"id":"h10_pemb_tnbc_intx","text":"treatment_pembrolizumab × TNBC interaction on pfs is positive.","kind":"novel"})

sub = df[df["her2_low"] == 1]
g1, g0 = sub.loc[sub["treatment_sacituzumab_govitecan"] == 1, OUTCOME], sub.loc[sub["treatment_sacituzumab_govitecan"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h10_sac_low"],"code":"Welch t-test pfs by sacituzumab in her2_low==1","result_summary":f"HER2-low: saci pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

df_ = df.copy(); df_["tx_l"] = df_["treatment_sacituzumab_govitecan"] * df_["her2_low"]
m = fit_with(df_, ["treatment_sacituzumab_govitecan", "her2_low", "tx_l"])
coef = float(m.params["tx_l"]); p = float(m.pvalues["tx_l"])
analyses.append({"hypothesis_ids":["h10_sac_intx"],"code":"OLS pfs ~ saci + her2_low + saci:her2_low","result_summary":f"Interaction saci×her2_low={coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})

df_ = df.copy()
df_["tnbc"] = ((df_["er_positive"] == 0) & (df_["her2_positive"] == 0)).astype(int)
sub = df_[df_["tnbc"] == 1]
g1, g0 = sub.loc[sub["treatment_pembrolizumab"] == 1, OUTCOME], sub.loc[sub["treatment_pembrolizumab"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h10_pemb_tnbc"],"code":"Welch t-test pfs by pembro in TNBC","result_summary":f"TNBC: pembro pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

df_["tx_t"] = df_["treatment_pembrolizumab"] * df_["tnbc"]
m = fit_with(df_, ["treatment_pembrolizumab", "tnbc", "tx_t"])
coef = float(m.params["tx_t"]); p = float(m.pvalues["tx_t"])
analyses.append({"hypothesis_ids":["h10_pemb_tnbc_intx"],"code":"OLS pfs ~ pembro + tnbc + pembro:tnbc","result_summary":f"Interaction pembro×TNBC={coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})
add_iter(10, hypotheses, analyses)


# Iter 11: Palbociclib × ER; palbociclib × postmenopausal
hypotheses = []; analyses = []
hypotheses.append({"id":"h11_palbo_er_in_er","text":"Among ER-positive patients, treatment_palbociclib prolongs pfs_months more than in ER-negative patients.","kind":"novel"})
hypotheses.append({"id":"h11_palbo_er_intx","text":"treatment_palbociclib × er_positive interaction on pfs is positive.","kind":"novel"})
hypotheses.append({"id":"h11_palbo_pmp","text":"treatment_palbociclib × postmenopausal interaction on pfs is positive.","kind":"novel"})

sub = df[df["er_positive"] == 1]
g1, g0 = sub.loc[sub["treatment_palbociclib"] == 1, OUTCOME], sub.loc[sub["treatment_palbociclib"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h11_palbo_er_in_er"],"code":"Welch t-test pfs by palbo in ER+","result_summary":f"ER+: palbo pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})
sub = df[df["er_positive"] == 0]
g1, g0 = sub.loc[sub["treatment_palbociclib"] == 1, OUTCOME], sub.loc[sub["treatment_palbociclib"] == 0, OUTCOME]
eff_n = float(g1.mean() - g0.mean()); t, p_n = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h11_palbo_er_in_er"],"code":"Welch t-test pfs by palbo in ER-","result_summary":f"ER-: palbo pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff_n:+.3f}, p={p_n:.3g}","p_value":float(p_n),"effect_estimate":eff_n,"significant":bool(p_n<0.05)})

df_ = df.copy(); df_["tx_e"] = df_["treatment_palbociclib"] * df_["er_positive"]
m = fit_with(df_, ["treatment_palbociclib", "er_positive", "tx_e"])
coef = float(m.params["tx_e"]); p = float(m.pvalues["tx_e"])
analyses.append({"hypothesis_ids":["h11_palbo_er_intx"],"code":"OLS pfs ~ palbo + er_positive + palbo:er_positive","result_summary":f"Interaction palbo×er_positive={coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})

df_["tx_p"] = df_["treatment_palbociclib"] * df_["postmenopausal"]
m = fit_with(df_, ["treatment_palbociclib", "postmenopausal", "tx_p"])
coef = float(m.params["tx_p"]); p = float(m.pvalues["tx_p"])
analyses.append({"hypothesis_ids":["h11_palbo_pmp"],"code":"OLS pfs ~ palbo + postmenopausal + palbo:postmenopausal","result_summary":f"Interaction palbo×postmenopausal={coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})
add_iter(11, hypotheses, analyses)


# Iter 12: Palbociclib × PIK3CA; palbociclib × ER+/HER2-
hypotheses = []; analyses = []
hypotheses.append({"id":"h12_palbo_pik","text":"treatment_palbociclib effect on pfs differs by pik3ca_mutation (interaction).","kind":"novel"})
hypotheses.append({"id":"h12_palbo_erh2","text":"In ER+/HER2- subgroup, treatment_palbociclib produces a larger pfs benefit than in other subgroups.","kind":"novel"})

df_ = df.copy(); df_["tx_pk"] = df_["treatment_palbociclib"] * df_["pik3ca_mutation"]
m = fit_with(df_, ["treatment_palbociclib", "pik3ca_mutation", "tx_pk"])
coef = float(m.params["tx_pk"]); p = float(m.pvalues["tx_pk"])
analyses.append({"hypothesis_ids":["h12_palbo_pik"],"code":"OLS pfs ~ palbo + pik3ca + palbo:pik3ca","result_summary":f"Interaction palbo×pik3ca={coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})

df_["erh2"] = ((df_["er_positive"] == 1) & (df_["her2_positive"] == 0)).astype(int)
sub = df_[df_["erh2"] == 1]
g1, g0 = sub.loc[sub["treatment_palbociclib"] == 1, OUTCOME], sub.loc[sub["treatment_palbociclib"] == 0, OUTCOME]
eff = float(g1.mean() - g0.mean()); t, p = stats.ttest_ind(g1, g0, equal_var=False)
analyses.append({"hypothesis_ids":["h12_palbo_erh2"],"code":"Welch t-test pfs by palbo in ER+/HER2-","result_summary":f"ER+/HER2-: palbo pfs={g1.mean():.3f} (n={len(g1)}) vs {g0.mean():.3f} (n={len(g0)}), diff={eff:+.3f}, p={p:.3g}","p_value":float(p),"effect_estimate":eff,"significant":bool(p<0.05)})

df_["tx_eh2"] = df_["treatment_palbociclib"] * df_["erh2"]
m = fit_with(df_, ["treatment_palbociclib", "erh2", "tx_eh2"])
coef = float(m.params["tx_eh2"]); p = float(m.pvalues["tx_eh2"])
analyses.append({"hypothesis_ids":["h12_palbo_erh2"],"code":"OLS pfs ~ palbo + erh2 + palbo:erh2","result_summary":f"Interaction palbo×(ER+/HER2-)={coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})
add_iter(12, hypotheses, analyses)


# Iter 13: Treatment × stage_iv interactions
hypotheses = []; analyses = []
for tx in TREATMENTS:
    h_id = f"h13_{tx}_stage4"
    hypotheses.append({"id":h_id,"text":f"{tx} effect on pfs differs between stage_iv==1 and stage_iv==0 (interaction).","kind":"novel"})
    df_ = df.copy(); df_["i"] = df_[tx] * df_["stage_iv"]
    m = fit_with(df_, [tx, "stage_iv", "i"])
    coef = float(m.params["i"]); p = float(m.pvalues["i"])
    analyses.append({"hypothesis_ids":[h_id],"code":f"OLS pfs ~ {tx} + stage_iv + {tx}:stage_iv","result_summary":f"Interaction {tx}×stage_iv={coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})
add_iter(13, hypotheses, analyses)


# Iter 14: Treatment × ECOG interactions
hypotheses = []; analyses = []
for tx in TREATMENTS:
    h_id = f"h14_{tx}_ecog"
    hypotheses.append({"id":h_id,"text":f"{tx} effect on pfs is attenuated as ecog_ps increases (negative interaction).","kind":"novel"})
    df_ = df.copy(); df_["i"] = df_[tx] * df_["ecog_ps"]
    m = fit_with(df_, [tx, "ecog_ps", "i"])
    coef = float(m.params["i"]); p = float(m.pvalues["i"])
    analyses.append({"hypothesis_ids":[h_id],"code":f"OLS pfs ~ {tx} + ecog_ps + {tx}:ecog_ps","result_summary":f"Interaction {tx}×ecog_ps={coef:+.4f}, p={p:.3g}","p_value":p,"effect_estimate":coef,"significant":bool(p<0.05)})
add_iter(14, hypotheses, analyses)


# Iter 15: Multivariable model
hypotheses = []; analyses = []
hypotheses.append({"id":"h15_full","text":"In a multivariable OLS with all features and treatments simultaneously, treatment_palbociclib remains the dominant positive treatment effect on pfs_months.","kind":"refined"})
all_cols = TREATMENTS + ["age_years","ecog_ps","stage_iv","has_brain_mets","node_positive","postmenopausal","er_positive","pr_positive","her2_positive","her2_low","brca1_mutation","brca2_mutation","pik3ca_mutation","ki67_pct","tumor_size_cm","albumin_g_dl","ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr","hemoglobin_g_dl","alkaline_phosphatase_u_l","ast_u_l","alt_u_l","total_bilirubin_mg_dl","creatinine_mg_dl","bun_mg_dl","sodium_meq_l","potassium_meq_l","calcium_mg_dl","sex_female"]
m = fit_with(df, all_cols)
tx_results = []
for tx in TREATMENTS:
    tx_results.append((tx, float(m.params[tx]), float(m.pvalues[tx])))
tx_results.sort(key=lambda x: -x[1])
summ_lines = [f"{t}: {c:+.4f} (p={p:.3g})" for t,c,p in tx_results]
biggest = tx_results[0]
analyses.append({"hypothesis_ids":["h15_full"],"code":"OLS pfs ~ all 6 treatments + 31 covariates","result_summary":"Full multivariable model treatment coefs (months): " + "; ".join(summ_lines) + f". R²={m.rsquared:.3f}","p_value":biggest[2],"effect_estimate":biggest[1],"significant":bool(biggest[2]<0.05)})
add_iter(15, hypotheses, analyses)


with open("_iters_1_15.pkl", "wb") as f:
    pickle.dump(iterations, f)

print(f"Iterations done: {len(iterations)}")
for it in iterations[5:]:
    print(f"-- Iter {it['index']} --")
    for a in it['analyses']:
        print(" ", a['result_summary'][:230])
