"""Comprehensive prostate cancer analysis. Saves results to my_results.json."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
RESULTS = {}


def logreg(formula, data=None):
    data = df if data is None else data
    model = smf.logit(formula, data=data).fit(disp=0, maxiter=200)
    return model


def two_prop(group_mask, outcome="objective_response"):
    g1 = df.loc[group_mask, outcome]
    g0 = df.loc[~group_mask, outcome]
    p1, p0 = g1.mean(), g0.mean()
    table = np.array([[g1.sum(), len(g1) - g1.sum()],
                      [g0.sum(), len(g0) - g0.sum()]])
    chi2, p, _, _ = stats.chi2_contingency(table)
    return {
        "n_pos": int(g1.sum()), "n_pos_total": int(len(g1)),
        "n_neg": int(g0.sum()), "n_neg_total": int(len(g0)),
        "rr_pos": float(p1), "rr_neg": float(p0),
        "diff": float(p1 - p0), "chi2": float(chi2), "p_value": float(p),
    }


# ---------- ITER 1: marginal treatment effects ----------
treatments = ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
              "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]

iter1 = {}
for t in treatments:
    iter1[t] = two_prop(df[t].astype(bool))
RESULTS["iter1_marginal_treatment"] = iter1

# ---------- ITER 2: marginal biomarker effects ----------
biomarkers = ["mcrpc", "visceral_mets", "brca2_mutation", "ar_v7_positive", "msi_high", "psma_high"]
iter2 = {}
for b in biomarkers:
    iter2[b] = two_prop(df[b].astype(bool))
RESULTS["iter2_marginal_biomarker"] = iter2

# ---------- ITER 3: ECOG and Gleason ----------
iter3 = {}
m = smf.logit("objective_response ~ ecog_ps", data=df).fit(disp=0)
iter3["ecog_ps_logit"] = {
    "coef": float(m.params["ecog_ps"]),
    "p_value": float(m.pvalues["ecog_ps"]),
    "or": float(np.exp(m.params["ecog_ps"])),
}
m = smf.logit("objective_response ~ gleason_score", data=df).fit(disp=0)
iter3["gleason_logit"] = {
    "coef": float(m.params["gleason_score"]),
    "p_value": float(m.pvalues["gleason_score"]),
    "or": float(np.exp(m.params["gleason_score"])),
}
m = smf.logit("objective_response ~ age_years", data=df).fit(disp=0)
iter3["age_logit"] = {
    "coef": float(m.params["age_years"]),
    "p_value": float(m.pvalues["age_years"]),
    "or": float(np.exp(m.params["age_years"])),
}
RESULTS["iter3_clinical_features"] = iter3

# ---------- ITER 4: continuous lab features (univariable logistic on standardized) ----------
labs = ["psa_ng_ml", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr",
        "hemoglobin_g_dl", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
        "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
        "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl"]
iter4 = {}
for lab in labs:
    x = df[lab]
    z = (x - x.mean()) / x.std()
    sub = pd.DataFrame({"y": df["objective_response"], "z": z})
    m = smf.logit("y ~ z", data=sub).fit(disp=0)
    iter4[lab] = {
        "coef_per_sd": float(m.params["z"]),
        "or_per_sd": float(np.exp(m.params["z"])),
        "p_value": float(m.pvalues["z"]),
    }
RESULTS["iter4_labs_per_sd"] = iter4

# ---------- ITER 5: multivariable logistic with all features (no interactions) ----------
features = (treatments + biomarkers +
            ["age_years", "ecog_ps", "gleason_score"] + labs)
formula = "objective_response ~ " + " + ".join(features)
m = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
iter5 = {}
for f in features:
    iter5[f] = {
        "coef": float(m.params[f]),
        "or": float(np.exp(m.params[f])),
        "p_value": float(m.pvalues[f]),
    }
RESULTS["iter5_multivariable"] = iter5

# ---------- ITER 6: treatment x biomarker interaction screen (adjusted) ----------
# Use a small adjusted model: treatment + biomarker + interaction + key covariates.
covars = "ecog_ps + age_years + albumin_g_dl + ldh_u_l + hemoglobin_g_dl + visceral_mets + mcrpc"
iter6 = {}
treat_biomarker_pairs = [(t, b) for t in treatments for b in biomarkers if b != "mcrpc" and b != "visceral_mets"]
# also include mcrpc/visceral_mets but they're in covars; keep biomarker side limited
for t, b in treat_biomarker_pairs:
    f = f"objective_response ~ {t} * {b} + {covars}"
    try:
        m = smf.logit(f, data=df).fit(disp=0, maxiter=200)
        ix = f"{t}:{b}"
        iter6[ix] = {
            "interaction_coef": float(m.params[ix]),
            "interaction_or": float(np.exp(m.params[ix])),
            "interaction_p": float(m.pvalues[ix]),
            "treatment_main_coef": float(m.params[t]),
            "biomarker_main_coef": float(m.params[b]),
        }
    except Exception as e:
        iter6[f"{t}:{b}"] = {"error": str(e)}
RESULTS["iter6_interaction_screen"] = iter6

# ---------- ITER 7: stratified response rates within biomarker subgroups ----------
iter7 = {}
candidate = [
    ("treatment_olaparib", "brca2_mutation"),
    ("treatment_pembrolizumab", "msi_high"),
    ("treatment_lu177_psma", "psma_high"),
    ("treatment_enzalutamide", "ar_v7_positive"),
    ("treatment_abiraterone", "ar_v7_positive"),
    ("treatment_docetaxel", "visceral_mets"),
]
for t, b in candidate:
    pos = two_prop(df[t].astype(bool) & df[b].astype(bool))
    pos["subgroup"] = f"{t}=1 & {b}=1 vs others"
    # within-biomarker comparison
    sub = df[df[b] == 1]
    if len(sub) > 50:
        rr_t = sub.loc[sub[t] == 1, "objective_response"].mean()
        rr_n = sub.loc[sub[t] == 0, "objective_response"].mean()
        n_t = int((sub[t] == 1).sum())
        n_n = int((sub[t] == 0).sum())
        table = pd.crosstab(sub[t], sub["objective_response"])
        try:
            chi2, p, _, _ = stats.chi2_contingency(table.values)
        except Exception:
            chi2, p = float("nan"), float("nan")
        within_pos = {"rr_treated": float(rr_t), "rr_untreated": float(rr_n),
                      "diff": float(rr_t - rr_n), "n_treated": n_t,
                      "n_untreated": n_n, "p_value": float(p)}
    else:
        within_pos = {}
    sub2 = df[df[b] == 0]
    if len(sub2) > 50:
        rr_t = sub2.loc[sub2[t] == 1, "objective_response"].mean()
        rr_n = sub2.loc[sub2[t] == 0, "objective_response"].mean()
        n_t = int((sub2[t] == 1).sum())
        n_n = int((sub2[t] == 0).sum())
        table = pd.crosstab(sub2[t], sub2["objective_response"])
        try:
            chi2, p, _, _ = stats.chi2_contingency(table.values)
        except Exception:
            chi2, p = float("nan"), float("nan")
        within_neg = {"rr_treated": float(rr_t), "rr_untreated": float(rr_n),
                      "diff": float(rr_t - rr_n), "n_treated": n_t,
                      "n_untreated": n_n, "p_value": float(p)}
    else:
        within_neg = {}
    iter7[f"{t}__by__{b}"] = {"within_positive": within_pos, "within_negative": within_neg}
RESULTS["iter7_stratified"] = iter7

# ---------- ITER 8: triple-interaction probes for top heterogeneity signals ----------
# Look for whether enzalutamide x ar_v7 effect depends on ECOG; olaparib x brca2 by visceral mets, etc.
iter8 = {}
triples = [
    ("treatment_enzalutamide", "ar_v7_positive", "ecog_ps"),
    ("treatment_olaparib", "brca2_mutation", "visceral_mets"),
    ("treatment_pembrolizumab", "msi_high", "ldh_u_l"),
    ("treatment_lu177_psma", "psma_high", "visceral_mets"),
    ("treatment_abiraterone", "ar_v7_positive", "mcrpc"),
]
for t, b, m_var in triples:
    f = f"objective_response ~ {t} * {b} * {m_var} + ecog_ps + age_years + albumin_g_dl + hemoglobin_g_dl + ldh_u_l"
    try:
        mod = smf.logit(f, data=df).fit(disp=0, maxiter=300)
        key = f"{t}:{b}:{m_var}"
        iter8[key] = {
            "triple_coef": float(mod.params[key]),
            "triple_p": float(mod.pvalues[key]),
        }
    except Exception as e:
        iter8[f"{t}:{b}:{m_var}"] = {"error": str(e)}
RESULTS["iter8_triples"] = iter8

# ---------- ITER 9: subgroup-defined treatment effects (for transcript completeness) ----------
# Best-supported subgroup hypotheses with effect sizes for each treatment.
iter9 = {}
def subgroup_effect(treat, mask, label):
    sub = df[mask]
    if len(sub) < 50:
        return None
    rr_t = sub.loc[sub[treat] == 1, "objective_response"].mean()
    rr_n = sub.loc[sub[treat] == 0, "objective_response"].mean()
    n_t = int((sub[treat] == 1).sum())
    n_n = int((sub[treat] == 0).sum())
    table = pd.crosstab(sub[treat], sub["objective_response"])
    try:
        chi2, p, _, _ = stats.chi2_contingency(table.values)
    except Exception:
        p = float("nan")
    return {"label": label, "n_treated": n_t, "n_untreated": n_n,
            "rr_treated": float(rr_t), "rr_untreated": float(rr_n),
            "diff": float(rr_t - rr_n), "p_value": float(p)}

iter9["olaparib_brca2"] = subgroup_effect("treatment_olaparib",
                                          df["brca2_mutation"] == 1,
                                          "BRCA2-mutant patients")
iter9["pembrolizumab_msi"] = subgroup_effect("treatment_pembrolizumab",
                                             df["msi_high"] == 1,
                                             "MSI-high patients")
iter9["lu177_psmahigh"] = subgroup_effect("treatment_lu177_psma",
                                          df["psma_high"] == 1,
                                          "PSMA-high patients")
iter9["enzalutamide_arv7neg"] = subgroup_effect("treatment_enzalutamide",
                                                df["ar_v7_positive"] == 0,
                                                "AR-V7 negative patients")
iter9["enzalutamide_arv7pos"] = subgroup_effect("treatment_enzalutamide",
                                                df["ar_v7_positive"] == 1,
                                                "AR-V7 positive patients")
iter9["abiraterone_arv7neg"] = subgroup_effect("treatment_abiraterone",
                                               df["ar_v7_positive"] == 0,
                                               "AR-V7 negative patients")
iter9["docetaxel_visceral"] = subgroup_effect("treatment_docetaxel",
                                              df["visceral_mets"] == 1,
                                              "visceral mets patients")
iter9["docetaxel_novisceral"] = subgroup_effect("treatment_docetaxel",
                                                df["visceral_mets"] == 0,
                                                "no visceral mets")
RESULTS["iter9_subgroup_effects"] = iter9

# ---------- ITER 10: combined subgroup definitions (e.g., olaparib in BRCA2+ AND not visceral) ----------
iter10 = {}
def subgroup_effect_mask(treat, mask, label):
    return subgroup_effect(treat, mask, label)

# olaparib: BRCA2+ stratified by ecog
m1 = (df["brca2_mutation"] == 1) & (df["ecog_ps"] <= 1)
iter10["olaparib_brca2_ecog01"] = subgroup_effect_mask(
    "treatment_olaparib", m1, "BRCA2+ AND ECOG<=1")
m2 = (df["brca2_mutation"] == 1) & (df["ecog_ps"] == 2)
iter10["olaparib_brca2_ecog2"] = subgroup_effect_mask(
    "treatment_olaparib", m2, "BRCA2+ AND ECOG=2")

# pembrolizumab: MSI-high by ECOG
m1 = (df["msi_high"] == 1) & (df["ecog_ps"] <= 1)
iter10["pembrolizumab_msi_ecog01"] = subgroup_effect_mask(
    "treatment_pembrolizumab", m1, "MSI-high AND ECOG<=1")
m2 = (df["msi_high"] == 1) & (df["ecog_ps"] == 2)
iter10["pembrolizumab_msi_ecog2"] = subgroup_effect_mask(
    "treatment_pembrolizumab", m2, "MSI-high AND ECOG=2")

# lu177-psma: PSMA-high by visceral
m1 = (df["psma_high"] == 1) & (df["visceral_mets"] == 0)
iter10["lu177_psmahigh_novisceral"] = subgroup_effect_mask(
    "treatment_lu177_psma", m1, "PSMA-high AND no visceral mets")
m2 = (df["psma_high"] == 1) & (df["visceral_mets"] == 1)
iter10["lu177_psmahigh_visceral"] = subgroup_effect_mask(
    "treatment_lu177_psma", m2, "PSMA-high AND visceral mets")

# enzalutamide AR-V7 neg: by mcrpc
m1 = (df["ar_v7_positive"] == 0) & (df["mcrpc"] == 1)
iter10["enz_arv7neg_mcrpc"] = subgroup_effect_mask(
    "treatment_enzalutamide", m1, "AR-V7 neg AND mCRPC")
m2 = (df["ar_v7_positive"] == 0) & (df["mcrpc"] == 0)
iter10["enz_arv7neg_nomcrpc"] = subgroup_effect_mask(
    "treatment_enzalutamide", m2, "AR-V7 neg AND not mCRPC")
RESULTS["iter10_combined_subgroup"] = iter10

# ---------- ITER 11: tree-based heterogeneity discovery for each treatment ----------
# Use a logistic regression with treatment + treatment*feature interactions for each treatment,
# screening over all features.
from sklearn.preprocessing import StandardScaler
iter11 = {}
all_features = ["age_years", "ecog_ps", "gleason_score", "mcrpc", "visceral_mets",
                "brca2_mutation", "ar_v7_positive", "msi_high", "psma_high",
                "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr",
                "hemoglobin_g_dl", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
                "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
                "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl"]
for t in treatments:
    out = {}
    for feat in all_features:
        if df[feat].nunique() <= 5:
            x = df[feat].astype(float)
        else:
            mu, sd = df[feat].mean(), df[feat].std()
            x = (df[feat] - mu) / sd
        sub = pd.DataFrame({"y": df["objective_response"],
                            "t": df[t], "x": x})
        f = "y ~ t * x"
        try:
            mod = smf.logit(f, data=sub).fit(disp=0, maxiter=200)
            out[feat] = {
                "interaction_coef": float(mod.params["t:x"]),
                "interaction_p": float(mod.pvalues["t:x"]),
                "treat_main_coef": float(mod.params["t"]),
                "treat_main_p": float(mod.pvalues["t"]),
            }
        except Exception as e:
            out[feat] = {"error": str(e)}
    iter11[t] = out
RESULTS["iter11_het_screen"] = iter11

# ---------- ITER 12: joint model — confirm top heterogeneity per treatment ----------
iter12 = {}
# Pull top-significant interaction per treatment from iter11 then re-fit joint model
for t, results in iter11.items():
    sig = [(feat, r["interaction_p"], r["interaction_coef"])
           for feat, r in results.items() if "interaction_p" in r]
    sig.sort(key=lambda x: x[1])
    top3 = [s[0] for s in sig[:3]]
    if not top3:
        continue
    parts = []
    for feat in top3:
        if df[feat].nunique() <= 5:
            parts.append(f"{t} * C({feat})")
        else:
            parts.append(f"{t} * {feat}")
    f = (f"objective_response ~ " + " + ".join(parts) +
         " + ecog_ps + age_years + albumin_g_dl + ldh_u_l + hemoglobin_g_dl")
    try:
        mod = smf.logit(f, data=df).fit(disp=0, maxiter=300)
        rec = {"top3_features": top3}
        for name in mod.params.index:
            if t in name and ":" in name:
                rec[name] = {
                    "coef": float(mod.params[name]),
                    "p_value": float(mod.pvalues[name]),
                }
        iter12[t] = rec
    except Exception as e:
        iter12[t] = {"top3_features": top3, "error": str(e)}
RESULTS["iter12_joint_top_modifiers"] = iter12


# ---------- Write out everything ----------
def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [to_jsonable(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o

with open("my_results.json", "w") as f:
    json.dump(to_jsonable(RESULTS), f, indent=2, default=str)

print("Wrote my_results.json")
print("Total result blocks:", len(RESULTS))
