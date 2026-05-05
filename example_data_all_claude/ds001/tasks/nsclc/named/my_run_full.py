"""
Comprehensive NSCLC dataset analysis.
Outcome: pfs_months
Treatments: pembrolizumab, sotorasib, olaparib, osimertinib
Saves results to results_my.json for use in transcript building.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
np.random.seed(42)

df = pd.read_parquet("dataset.parquet")

# Build derived columns we'll use repeatedly
df["smoking_current"] = (df["smoking_status"] == "current").astype(int)
df["smoking_former"] = (df["smoking_status"] == "former").astype(int)
df["smoking_never"] = (df["smoking_status"] == "never").astype(int)
df["adeno"] = (df["histology"] == "adenocarcinoma").astype(int)
df["pdl1_high"] = (df["pdl1_tps"] >= 0.5).astype(int)
df["pdl1_low"] = (df["pdl1_tps"] < 0.01).astype(int)

ALL_FEATURES = [
    "age_years", "sex_female",
    "smoking_current", "smoking_former",
    "ecog_ps", "adeno", "stage_iv", "has_brain_mets",
    "egfr_mutation", "kras_g12c", "alk_fusion",
    "stk11_mutation", "brca2_mutation",
    "pdl1_tps", "tmb_high",
    "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo",
    "crp_mg_l", "nlr",
    "hemoglobin_g_dl", "alkaline_phosphatase_u_l",
    "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl",
    "creatinine_mg_dl", "bun_mg_dl",
    "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl",
]

TREATMENTS = [
    "treatment_pembrolizumab",
    "treatment_sotorasib",
    "treatment_olaparib",
    "treatment_osimertinib",
]

OUT = {}


def ols_summary(model, term):
    coef = float(model.params.get(term, np.nan))
    pv = float(model.pvalues.get(term, np.nan))
    se = float(model.bse.get(term, np.nan)) if term in model.bse.index else np.nan
    return {"coef": coef, "p": pv, "se": se}


# ---- Iter 1: univariate associations of features with PFS ----
print("=== Iteration 1: feature-PFS univariate associations ===")
iter1 = {}
for f in ALL_FEATURES:
    formula = f"pfs_months ~ {f}"
    m = smf.ols(formula, data=df).fit()
    iter1[f] = ols_summary(m, f)
    iter1[f]["n"] = int(m.nobs)
    print(f"  {f:30s}  beta={iter1[f]['coef']:+.4f}  p={iter1[f]['p']:.2e}")
OUT["iter1_feature_assoc"] = iter1


# ---- Iter 2: main treatment effects on PFS (unadjusted and adjusted) ----
print("\n=== Iteration 2: main treatment effects on PFS ===")
iter2 = {}
for t in TREATMENTS:
    m1 = smf.ols(f"pfs_months ~ {t}", data=df).fit()
    others = [x for x in TREATMENTS if x != t]
    formula_adj = f"pfs_months ~ {t} + " + " + ".join(others) + " + age_years + sex_female + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + ldh_u_l + nlr + crp_mg_l + weight_loss_pct_6mo + adeno + smoking_current + smoking_former + egfr_mutation + kras_g12c + alk_fusion + stk11_mutation + brca2_mutation + pdl1_tps + tmb_high"
    m2 = smf.ols(formula_adj, data=df).fit()
    iter2[t] = {
        "unadjusted": ols_summary(m1, t),
        "adjusted":   ols_summary(m2, t),
        "mean_on":  float(df.loc[df[t]==1, "pfs_months"].mean()),
        "mean_off": float(df.loc[df[t]==0, "pfs_months"].mean()),
        "n_on":  int(df[t].sum()),
        "n_off": int((df[t]==0).sum()),
    }
    print(f"  {t}: mean_on={iter2[t]['mean_on']:.3f} mean_off={iter2[t]['mean_off']:.3f} unadj_beta={iter2[t]['unadjusted']['coef']:+.3f} (p={iter2[t]['unadjusted']['p']:.2e})  adj_beta={iter2[t]['adjusted']['coef']:+.3f} (p={iter2[t]['adjusted']['p']:.2e})")
OUT["iter2_treatment_main"] = iter2


# ---- Iter 3: Biomarker-targeted treatment hypotheses ----
print("\n=== Iteration 3: biomarker-targeted treatment effects ===")
def subgroup_effect(treat, sub_mask, label):
    sub = df.loc[sub_mask]
    if sub[treat].nunique() < 2 or sub.shape[0] < 50:
        return None
    m = smf.ols(f"pfs_months ~ {treat}", data=sub).fit()
    return {
        "subgroup": label,
        "n": int(sub.shape[0]),
        "n_on": int(sub[treat].sum()),
        "mean_on":  float(sub.loc[sub[treat]==1, "pfs_months"].mean()),
        "mean_off": float(sub.loc[sub[treat]==0, "pfs_months"].mean()),
        **ols_summary(m, treat),
    }

iter3 = {}
# pembrolizumab x PDL1
for label, mask in [
    ("pdl1_tps>=0.5", df["pdl1_tps"]>=0.5),
    ("pdl1_tps<0.5",  df["pdl1_tps"]<0.5),
    ("pdl1_tps<0.01", df["pdl1_tps"]<0.01),
    ("tmb_high==1",   df["tmb_high"]==1),
    ("tmb_high==0",   df["tmb_high"]==0),
]:
    res = subgroup_effect("treatment_pembrolizumab", mask, label)
    if res: iter3.setdefault("pembrolizumab", []).append(res)
    print(f"  pembro | {label}: {res}")

# sotorasib x KRAS-G12C
for label, mask in [
    ("kras_g12c==1", df["kras_g12c"]==1),
    ("kras_g12c==0", df["kras_g12c"]==0),
]:
    res = subgroup_effect("treatment_sotorasib", mask, label)
    if res: iter3.setdefault("sotorasib", []).append(res)
    print(f"  soto   | {label}: {res}")

# olaparib x BRCA2
for label, mask in [
    ("brca2_mutation==1", df["brca2_mutation"]==1),
    ("brca2_mutation==0", df["brca2_mutation"]==0),
]:
    res = subgroup_effect("treatment_olaparib", mask, label)
    if res: iter3.setdefault("olaparib", []).append(res)
    print(f"  ola    | {label}: {res}")

# osimertinib x EGFR
for label, mask in [
    ("egfr_mutation==1", df["egfr_mutation"]==1),
    ("egfr_mutation==0", df["egfr_mutation"]==0),
]:
    res = subgroup_effect("treatment_osimertinib", mask, label)
    if res: iter3.setdefault("osimertinib", []).append(res)
    print(f"  osi    | {label}: {res}")

OUT["iter3_biomarker_targeted"] = iter3


# ---- Iter 4: formal interaction tests (treatment x main biomarker) ----
print("\n=== Iteration 4: formal interaction tests (treatment x intended biomarker) ===")
iter4 = {}
specs = [
    ("treatment_osimertinib", "egfr_mutation"),
    ("treatment_sotorasib",   "kras_g12c"),
    ("treatment_olaparib",    "brca2_mutation"),
    ("treatment_pembrolizumab","pdl1_high"),
    ("treatment_pembrolizumab","tmb_high"),
    ("treatment_pembrolizumab","pdl1_low"),
]
for t, b in specs:
    formula = f"pfs_months ~ {t}*{b}"
    m = smf.ols(formula, data=df).fit()
    iter4[f"{t}__x__{b}"] = {
        "main_t":  ols_summary(m, t),
        "main_b":  ols_summary(m, b),
        "intxn":   ols_summary(m, f"{t}:{b}"),
    }
    print(f"  {t} x {b}: intxn beta={iter4[f'{t}__x__{b}']['intxn']['coef']:+.3f} p={iter4[f'{t}__x__{b}']['intxn']['p']:.2e}")
OUT["iter4_targeted_intxn"] = iter4


# ---- Iter 5: full treatment-by-feature interaction screen for each treatment ----
print("\n=== Iteration 5: treatment x feature interaction screen ===")
SCREEN_FEATURES = [
    "age_years", "sex_female",
    "smoking_current", "smoking_former",
    "ecog_ps", "adeno", "stage_iv", "has_brain_mets",
    "egfr_mutation", "kras_g12c", "alk_fusion",
    "stk11_mutation", "brca2_mutation",
    "pdl1_tps", "pdl1_high", "pdl1_low", "tmb_high",
    "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo",
    "crp_mg_l", "nlr",
    "hemoglobin_g_dl", "alkaline_phosphatase_u_l",
    "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl",
    "creatinine_mg_dl", "bun_mg_dl",
    "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl",
]
iter5 = {}
for t in TREATMENTS:
    iter5[t] = {}
    for f in SCREEN_FEATURES:
        if f == t: continue
        try:
            m = smf.ols(f"pfs_months ~ {t}*{f}", data=df).fit()
            iter5[t][f] = ols_summary(m, f"{t}:{f}")
        except Exception as e:
            iter5[t][f] = {"err": str(e)}
    # report top significant
    sig = [(k, v) for k,v in iter5[t].items() if isinstance(v, dict) and "p" in v and v["p"] < 0.01]
    sig.sort(key=lambda kv: kv[1]["p"])
    print(f"\n  {t} top intxn (p<0.01):")
    for k, v in sig[:8]:
        print(f"    {k:30s}  beta={v['coef']:+.4f}  p={v['p']:.2e}")
OUT["iter5_full_intxn_screen"] = iter5


# ---- Iter 6: refined subgroup search (intersection of strong modifiers) ----
print("\n=== Iteration 6: targeted subgroup refinements ===")
iter6 = {}

def grid_subgroup(treat, masks):
    """masks: list of (label, mask). Compute joint subgroup."""
    label = " AND ".join(m[0] for m in masks)
    if not masks:
        sub_mask = pd.Series(True, index=df.index)
    else:
        sub_mask = masks[0][1]
        for m in masks[1:]:
            sub_mask = sub_mask & m[1]
    return subgroup_effect(treat, sub_mask, label)

# osimertinib: EGFR + ECOG + Stage variants
for ecog_lab, ecog_mask in [("ecog<=1", df["ecog_ps"]<=1), ("ecog==0", df["ecog_ps"]==0)]:
    for stage_lab, stage_mask in [("stageIV==1", df["stage_iv"]==1), ("stageIV==0", df["stage_iv"]==0), ("any-stage", pd.Series(True,index=df.index))]:
        res = grid_subgroup("treatment_osimertinib", [
            ("egfr_mutation==1", df["egfr_mutation"]==1),
            (ecog_lab, ecog_mask),
            (stage_lab, stage_mask),
        ])
        if res: iter6.setdefault("osimertinib", []).append(res)
        print(f"  osi: {res}")

# sotorasib: KRAS + ECOG/Stage
for ecog_lab, ecog_mask in [("ecog<=1", df["ecog_ps"]<=1), ("ecog==0", df["ecog_ps"]==0)]:
    res = grid_subgroup("treatment_sotorasib", [
        ("kras_g12c==1", df["kras_g12c"]==1),
        (ecog_lab, ecog_mask),
    ])
    if res: iter6.setdefault("sotorasib", []).append(res)
    print(f"  soto: {res}")

# pembrolizumab: PDL1 + TMB + ECOG
for pdl_lab, pdl_mask in [("pdl1_tps>=0.5", df["pdl1_tps"]>=0.5)]:
    for tmb_lab, tmb_mask in [("tmb_high==1", df["tmb_high"]==1), ("tmb_high==0", df["tmb_high"]==0)]:
        for ecog_lab, ecog_mask in [("ecog<=1", df["ecog_ps"]<=1), ("ecog==0", df["ecog_ps"]==0)]:
            res = grid_subgroup("treatment_pembrolizumab", [
                (pdl_lab, pdl_mask),
                (tmb_lab, tmb_mask),
                (ecog_lab, ecog_mask),
            ])
            if res: iter6.setdefault("pembrolizumab", []).append(res)
            print(f"  pem: {res}")

# olaparib: BRCA2 + ECOG
for ecog_lab, ecog_mask in [("ecog<=1", df["ecog_ps"]<=1), ("any-ecog", pd.Series(True,index=df.index))]:
    res = grid_subgroup("treatment_olaparib", [
        ("brca2_mutation==1", df["brca2_mutation"]==1),
        (ecog_lab, ecog_mask),
    ])
    if res: iter6.setdefault("olaparib", []).append(res)
    print(f"  ola: {res}")

OUT["iter6_subgroup_refine"] = iter6


# ---- Iter 7: STK11 modifier of pembrolizumab ----
print("\n=== Iteration 7: STK11 + immunotherapy ===")
iter7 = {}
# pembro x stk11 in PDL1-high
specs = [
    ("pdl1_tps>=0.5 & stk11==1", (df["pdl1_tps"]>=0.5) & (df["stk11_mutation"]==1)),
    ("pdl1_tps>=0.5 & stk11==0", (df["pdl1_tps"]>=0.5) & (df["stk11_mutation"]==0)),
    ("pdl1_tps<0.5  & stk11==1", (df["pdl1_tps"]<0.5)  & (df["stk11_mutation"]==1)),
    ("pdl1_tps<0.5  & stk11==0", (df["pdl1_tps"]<0.5)  & (df["stk11_mutation"]==0)),
    ("stk11==1",                  df["stk11_mutation"]==1),
    ("stk11==0",                  df["stk11_mutation"]==0),
    ("stk11==1 & tmb_high==1",   (df["stk11_mutation"]==1) & (df["tmb_high"]==1)),
    ("stk11==0 & tmb_high==1",   (df["stk11_mutation"]==0) & (df["tmb_high"]==1)),
    ("stk11==1 & tmb_high==0",   (df["stk11_mutation"]==1) & (df["tmb_high"]==0)),
    ("stk11==0 & tmb_high==0",   (df["stk11_mutation"]==0) & (df["tmb_high"]==0)),
]
for lab, mask in specs:
    res = subgroup_effect("treatment_pembrolizumab", mask, lab)
    if res: iter7.setdefault("pembrolizumab", []).append(res)
    print(f"  pem | {lab}: n={res['n'] if res else None} beta={res['coef'] if res else None}")
OUT["iter7_stk11_pembro"] = iter7


# ---- Iter 8: dosing combinations and contamination check ----
print("\n=== Iteration 8: combination effects (more than one treatment) ===")
iter8 = {}
for t in TREATMENTS:
    others = [x for x in TREATMENTS if x != t]
    df["__on_other"] = df[others].sum(axis=1) > 0
    sub_mono = df[~df["__on_other"]]
    sub_mono = sub_mono[sub_mono[t].isin([0,1])]
    if sub_mono[t].sum() > 50 and (sub_mono[t]==0).sum() > 50:
        m = smf.ols(f"pfs_months ~ {t}", data=sub_mono).fit()
        iter8[t] = ols_summary(m, t)
        iter8[t]["n_total"] = int(sub_mono.shape[0])
        iter8[t]["n_on"] = int(sub_mono[t].sum())
        print(f"  monotherapy {t}: beta={iter8[t]['coef']:+.3f}  p={iter8[t]['p']:.2e}  n={iter8[t]['n_total']}")
del df["__on_other"]
OUT["iter8_monotherapy"] = iter8


# ---- Iter 9: full subgroup grid for each targeted treatment ----
print("\n=== Iteration 9: refined grid - pembro PDL1 high + ECOG + STK11 + TMB ===")
iter9 = {}
mask_pem = (df["pdl1_tps"]>=0.5) & (df["ecog_ps"]<=1) & (df["stk11_mutation"]==0)
res = subgroup_effect("treatment_pembrolizumab", mask_pem, "pdl1_high & ecog<=1 & stk11==0")
if res: iter9["pembro_best"] = res
print(f"  pem best: {res}")

mask_pem2 = (df["pdl1_tps"]>=0.5) & (df["ecog_ps"]==0) & (df["stk11_mutation"]==0) & (df["tmb_high"]==1)
res = subgroup_effect("treatment_pembrolizumab", mask_pem2, "pdl1_high & ecog==0 & stk11==0 & tmb_high==1")
if res: iter9["pembro_strict"] = res
print(f"  pem strict: {res}")

mask_osi = (df["egfr_mutation"]==1)
res = subgroup_effect("treatment_osimertinib", mask_osi, "egfr_mutation==1")
if res: iter9["osi_egfr"] = res
print(f"  osi egfr: {res}")

mask_osi2 = (df["egfr_mutation"]==1) & (df["ecog_ps"]<=1)
res = subgroup_effect("treatment_osimertinib", mask_osi2, "egfr_mutation==1 & ecog<=1")
if res: iter9["osi_egfr_ecog"] = res
print(f"  osi egfr ecog: {res}")

mask_soto = (df["kras_g12c"]==1)
res = subgroup_effect("treatment_sotorasib", mask_soto, "kras_g12c==1")
if res: iter9["soto_kras"] = res
print(f"  soto kras: {res}")

mask_ola = (df["brca2_mutation"]==1)
res = subgroup_effect("treatment_olaparib", mask_ola, "brca2_mutation==1")
if res: iter9["ola_brca"] = res
print(f"  ola brca: {res}")

mask_ola2 = (df["brca2_mutation"]==1) & (df["ecog_ps"]<=1)
res = subgroup_effect("treatment_olaparib", mask_ola2, "brca2_mutation==1 & ecog<=1")
if res: iter9["ola_brca_ecog"] = res
print(f"  ola brca ecog: {res}")

OUT["iter9_targeted_grids"] = iter9


# ---- Iter 10: 3-way interactions for treatment effect modifiers ----
print("\n=== Iteration 10: feature x feature interactions among modifiers ===")
iter10 = {}
# Test in EGFR+ patients: does ecog modify osimertinib effect?
sub = df[df["egfr_mutation"]==1]
for f in ["ecog_ps", "stage_iv", "has_brain_mets", "albumin_g_dl", "ldh_u_l", "stk11_mutation", "weight_loss_pct_6mo", "nlr", "crp_mg_l"]:
    try:
        m = smf.ols(f"pfs_months ~ treatment_osimertinib*{f}", data=sub).fit()
        iter10.setdefault("osi_in_egfr", {})[f] = ols_summary(m, f"treatment_osimertinib:{f}")
    except Exception as e:
        iter10.setdefault("osi_in_egfr", {})[f] = {"err": str(e)}
print("  osi_in_egfr keys:", list(iter10.get("osi_in_egfr", {}).keys()))
for k, v in iter10.get("osi_in_egfr", {}).items():
    print(f"    {k}: {v}")

# In KRAS+ patients: modifiers of sotorasib effect
sub = df[df["kras_g12c"]==1]
for f in ["ecog_ps", "stage_iv", "has_brain_mets", "albumin_g_dl", "ldh_u_l", "stk11_mutation"]:
    try:
        m = smf.ols(f"pfs_months ~ treatment_sotorasib*{f}", data=sub).fit()
        iter10.setdefault("soto_in_kras", {})[f] = ols_summary(m, f"treatment_sotorasib:{f}")
    except Exception:
        pass
for k, v in iter10.get("soto_in_kras", {}).items():
    print(f"  soto_in_kras {k}: {v}")

# In BRCA2+ patients: modifiers of olaparib effect
sub = df[df["brca2_mutation"]==1]
for f in ["ecog_ps", "stage_iv", "has_brain_mets", "albumin_g_dl", "ldh_u_l", "stk11_mutation"]:
    try:
        m = smf.ols(f"pfs_months ~ treatment_olaparib*{f}", data=sub).fit()
        iter10.setdefault("ola_in_brca", {})[f] = ols_summary(m, f"treatment_olaparib:{f}")
    except Exception:
        pass
for k, v in iter10.get("ola_in_brca", {}).items():
    print(f"  ola_in_brca {k}: {v}")

# In PDL1-high: modifiers of pembrolizumab effect
sub = df[df["pdl1_tps"]>=0.5]
for f in ["ecog_ps", "stage_iv", "has_brain_mets", "albumin_g_dl", "ldh_u_l", "stk11_mutation", "tmb_high"]:
    try:
        m = smf.ols(f"pfs_months ~ treatment_pembrolizumab*{f}", data=sub).fit()
        iter10.setdefault("pem_in_pdl1high", {})[f] = ols_summary(m, f"treatment_pembrolizumab:{f}")
    except Exception:
        pass
for k, v in iter10.get("pem_in_pdl1high", {}).items():
    print(f"  pem_in_pdl1high {k}: {v}")

OUT["iter10_3way"] = iter10


# ---- Iter 11: regression-tree subgroup discovery for each treatment ----
print("\n=== Iteration 11: tree-based subgroup discovery ===")
from sklearn.tree import DecisionTreeRegressor, _tree

iter11 = {}
def tree_subgroups(treat, max_depth=3):
    """Fit tree on treated vs untreated diff; report leaves."""
    feats = [c for c in [
        "age_years", "sex_female", "smoking_current", "smoking_former",
        "ecog_ps", "adeno", "stage_iv", "has_brain_mets",
        "egfr_mutation", "kras_g12c", "alk_fusion",
        "stk11_mutation", "brca2_mutation",
        "pdl1_tps", "tmb_high",
        "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo",
        "crp_mg_l", "nlr",
        "hemoglobin_g_dl", "alkaline_phosphatase_u_l",
        "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl",
        "creatinine_mg_dl", "bun_mg_dl",
        "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl",
    ] if c in df.columns]
    X = df[feats].values
    y = df["pfs_months"].values
    t = df[treat].values
    # T-learner
    tr_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=200, random_state=0).fit(X[t==1], y[t==1])
    co_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=200, random_state=0).fit(X[t==0], y[t==0])
    pred_treated = tr_model.predict(X)
    pred_control = co_model.predict(X)
    cate = pred_treated - pred_control
    # Find top region with largest CATE
    rank = np.argsort(-cate)
    top = rank[:int(0.10*len(cate))]
    # Within top decile, fit OLS for treatment effect
    sub_mask = np.zeros(len(df), dtype=bool)
    sub_mask[top] = True
    sub = df[sub_mask]
    res = subgroup_effect(treat, pd.Series(sub_mask, index=df.index), f"top10pct_predicted_CATE_for_{treat}")
    return {"top10pct_subgroup_effect": res, "mean_cate_top10pct": float(cate[top].mean()), "mean_cate_overall": float(cate.mean())}

for t in TREATMENTS:
    res = tree_subgroups(t, max_depth=3)
    iter11[t] = res
    print(f"  {t}: mean_cate={res['mean_cate_overall']:+.3f}, top10pct cate={res['mean_cate_top10pct']:+.3f}, top10pct effect={res['top10pct_subgroup_effect']}")
OUT["iter11_tree"] = iter11


# ---- Iter 12: continuous biomarker thresholds ----
print("\n=== Iteration 12: continuous biomarker effects on PFS ===")
iter12 = {}
for c in ["age_years", "albumin_g_dl", "ldh_u_l", "nlr", "crp_mg_l", "weight_loss_pct_6mo", "hemoglobin_g_dl", "alkaline_phosphatase_u_l", "calcium_mg_dl", "sodium_meq_l", "creatinine_mg_dl", "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl", "bun_mg_dl", "potassium_meq_l", "pdl1_tps"]:
    formula = f"pfs_months ~ {c} + age_years + sex_female + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + ldh_u_l + nlr + treatment_pembrolizumab + treatment_sotorasib + treatment_olaparib + treatment_osimertinib"
    try:
        m = smf.ols(formula, data=df).fit()
        iter12[c] = ols_summary(m, c)
    except Exception as e:
        iter12[c] = {"err": str(e)}
    if c in iter12 and "p" in iter12[c]:
        print(f"  {c}: beta={iter12[c]['coef']:+.4f}  p={iter12[c]['p']:.2e}")
OUT["iter12_continuous"] = iter12


# ---- Iter 13: sex/age/smoking interactions with each treatment ----
print("\n=== Iteration 13: demographic modifiers ===")
iter13 = {}
for t in TREATMENTS:
    iter13[t] = {}
    for f in ["sex_female", "age_years", "smoking_current", "smoking_never", "adeno"]:
        try:
            m = smf.ols(f"pfs_months ~ {t}*{f}", data=df).fit()
            iter13[t][f] = ols_summary(m, f"{t}:{f}")
        except Exception:
            pass
        if f in iter13[t]:
            print(f"  {t} x {f}: beta={iter13[t][f]['coef']:+.4f}  p={iter13[t][f]['p']:.2e}")
OUT["iter13_demographics"] = iter13


# ---- Iter 14: final subgroup hypotheses with robust CIs ----
print("\n=== Iteration 14: final best-supported subgroups (robust) ===")
def report_subgroup(treat, mask, label):
    sub = df[mask]
    if sub[treat].nunique() < 2:
        return {"label": label, "n": int(sub.shape[0]), "n_on": int(sub[treat].sum()), "issue":"no variation"}
    m = smf.ols(f"pfs_months ~ {treat}", data=sub).fit()
    p = float(m.pvalues[treat])
    coef = float(m.params[treat])
    ci = m.conf_int().loc[treat].tolist()
    return {"label": label, "n": int(sub.shape[0]), "n_on": int(sub[treat].sum()),
            "n_off": int((sub[treat]==0).sum()),
            "mean_on":  float(sub.loc[sub[treat]==1, "pfs_months"].mean()),
            "mean_off": float(sub.loc[sub[treat]==0, "pfs_months"].mean()),
            "coef": coef, "p": p, "ci_low": float(ci[0]), "ci_high": float(ci[1])}

iter14 = {}
iter14["osimertinib_egfr"] = report_subgroup("treatment_osimertinib", df["egfr_mutation"]==1, "egfr_mutation==1")
iter14["osimertinib_egfr_off"] = report_subgroup("treatment_osimertinib", df["egfr_mutation"]==0, "egfr_mutation==0")
iter14["sotorasib_kras"] = report_subgroup("treatment_sotorasib", df["kras_g12c"]==1, "kras_g12c==1")
iter14["sotorasib_kras_off"] = report_subgroup("treatment_sotorasib", df["kras_g12c"]==0, "kras_g12c==0")
iter14["olaparib_brca"] = report_subgroup("treatment_olaparib", df["brca2_mutation"]==1, "brca2_mutation==1")
iter14["olaparib_brca_off"] = report_subgroup("treatment_olaparib", df["brca2_mutation"]==0, "brca2_mutation==0")
iter14["pembro_pdl1high"] = report_subgroup("treatment_pembrolizumab", df["pdl1_tps"]>=0.5, "pdl1_tps>=0.5")
iter14["pembro_pdl1low"] = report_subgroup("treatment_pembrolizumab", df["pdl1_tps"]<0.5, "pdl1_tps<0.5")
iter14["pembro_pdl1high_stk11neg"] = report_subgroup(
    "treatment_pembrolizumab",
    (df["pdl1_tps"]>=0.5) & (df["stk11_mutation"]==0),
    "pdl1_tps>=0.5 & stk11==0"
)
iter14["pembro_pdl1high_stk11pos"] = report_subgroup(
    "treatment_pembrolizumab",
    (df["pdl1_tps"]>=0.5) & (df["stk11_mutation"]==1),
    "pdl1_tps>=0.5 & stk11==1"
)
iter14["pembro_tmbhigh_stk11neg"] = report_subgroup(
    "treatment_pembrolizumab",
    (df["tmb_high"]==1) & (df["stk11_mutation"]==0),
    "tmb_high==1 & stk11==0"
)
iter14["pembro_pdl1high_tmbhigh_stk11neg"] = report_subgroup(
    "treatment_pembrolizumab",
    (df["pdl1_tps"]>=0.5) & (df["tmb_high"]==1) & (df["stk11_mutation"]==0),
    "pdl1_tps>=0.5 & tmb_high==1 & stk11==0"
)

for k, v in iter14.items():
    print(f"  {k}: {v}")
OUT["iter14_final_subgroups"] = iter14


# ---- Iter 15: marker-treatment "mismatch" sanity checks ----
print("\n=== Iteration 15: marker-treatment mismatch checks ===")
iter15 = {}
checks = [
    ("treatment_osimertinib", "egfr_mutation==0", df["egfr_mutation"]==0),
    ("treatment_sotorasib",   "kras_g12c==0",    df["kras_g12c"]==0),
    ("treatment_olaparib",    "brca2_mutation==0", df["brca2_mutation"]==0),
    ("treatment_pembrolizumab","pdl1_tps<0.01",   df["pdl1_tps"]<0.01),
]
for t, lab, mask in checks:
    res = subgroup_effect(t, mask, lab)
    iter15[f"{t}__{lab}"] = res
    print(f"  {t} | {lab}: {res}")
OUT["iter15_mismatch"] = iter15


# Save raw results
with open("results_my.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("\nSaved to results_my.json")
