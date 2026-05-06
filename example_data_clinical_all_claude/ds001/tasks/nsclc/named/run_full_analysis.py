"""
Comprehensive analysis of ds001_nsclc.
Runs all tests needed for the 25-iteration transcript and saves results to results_full.json.
"""
import json
import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")
print(f"Loaded n={len(DF)}")

RESULTS = {}

def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return float(x)
    except Exception:
        return None

def record(key, **kwargs):
    RESULTS[key] = {k: (safe_float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                    for k, v in kwargs.items()}
    print(f"[{key}]", kwargs.get("summary", ""))

# ----- Helpers -----
def ttest(a, b):
    a = np.asarray(a); b = np.asarray(b)
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(np.mean(a) - np.mean(b)), float(res.pvalue), float(np.mean(a)), float(np.mean(b))

def linreg(y, x):
    """OLS y ~ x; returns slope, p, R^2."""
    X = sm.add_constant(np.asarray(x, dtype=float))
    m = sm.OLS(np.asarray(y, dtype=float), X).fit()
    return float(m.params[1]), float(m.pvalues[1]), float(m.rsquared)

def chi2_or(t):
    chi2, p, _, _ = stats.chi2_contingency(t)
    a, b = t[0,0], t[0,1]
    c, d = t[1,0], t[1,1]
    or_ = (a*d) / max(1e-9, (b*c))
    return float(chi2), float(p), float(or_)

def fit_ols(formula, data):
    return smf.ols(formula, data=data).fit()

# ===== ITERATION 1: cohort baseline + main outcome distribution =====
print("\n=== Cohort overview ===")
y = DF["pfs_months"].values
record("cohort_pfs",
       mean=float(np.mean(y)), median=float(np.median(y)), sd=float(np.std(y)),
       n=len(y),
       summary=f"PFS mean={np.mean(y):.2f} median={np.median(y):.2f} sd={np.std(y):.2f}")

# ===== ITERATION 2: demographics ~ PFS =====
print("\n=== Demographics vs PFS ===")
slope, p, r2 = linreg(DF["pfs_months"], DF["age_years"])
record("age_pfs", slope=slope, p=p, r2=r2,
       summary=f"slope={slope:.4f} months/yr p={p:.3g}")

slope, p, r2 = linreg(DF.loc[DF["sex_female"]==1,"pfs_months"].tolist()+DF.loc[DF["sex_female"]==0,"pfs_months"].tolist(),
                      [1]*int(DF["sex_female"].sum())+[0]*int((1-DF["sex_female"]).sum()))
diff, pp, ma, mb = ttest(DF.loc[DF["sex_female"]==1,"pfs_months"], DF.loc[DF["sex_female"]==0,"pfs_months"])
record("sex_pfs", diff_female_minus_male=diff, p=pp, mean_female=ma, mean_male=mb,
       summary=f"female-male diff={diff:.3f} p={pp:.3g}")

# Smoking status (3-level): use ANOVA / OLS encoded
m = fit_ols("pfs_months ~ C(smoking_status)", DF)
record("smoking_pfs_anova", f_p=float(m.f_pvalue),
       coefs={k: float(v) for k,v in m.params.items()},
       summary=f"smoking_status anova p={m.f_pvalue:.3g}")

# ECOG PS continuous-ish
slope, p, r2 = linreg(DF["pfs_months"], DF["ecog_ps"])
record("ecog_pfs", slope=slope, p=p, r2=r2,
       summary=f"ECOG slope={slope:.3f} months per unit p={p:.3g}")

# Histology
diff, pp, ma, mb = ttest(DF.loc[DF["histology"]=="adenocarcinoma","pfs_months"],
                         DF.loc[DF["histology"]!="adenocarcinoma","pfs_months"])
record("histology_pfs", diff_adeno_minus_other=diff, p=pp,
       summary=f"adeno-other diff={diff:.3f} p={pp:.3g}")

# Stage IV
diff, pp, ma, mb = ttest(DF.loc[DF["stage_iv"]==1,"pfs_months"], DF.loc[DF["stage_iv"]==0,"pfs_months"])
record("stage_iv_pfs", diff_iv_minus_lower=diff, p=pp, mean_iv=ma, mean_lower=mb,
       summary=f"stage IV-not diff={diff:.3f} p={pp:.3g}")

# Brain mets
diff, pp, ma, mb = ttest(DF.loc[DF["has_brain_mets"]==1,"pfs_months"], DF.loc[DF["has_brain_mets"]==0,"pfs_months"])
record("brain_mets_pfs", diff_yes_minus_no=diff, p=pp, mean_yes=ma, mean_no=mb,
       summary=f"brainmets-no diff={diff:.3f} p={pp:.3g}")

# ===== Mutations =====
print("\n=== Mutations vs PFS ===")
for mut in ["egfr_mutation","kras_g12c","alk_fusion","stk11_mutation","brca2_mutation","tmb_high"]:
    diff, pp, ma, mb = ttest(DF.loc[DF[mut]==1,"pfs_months"], DF.loc[DF[mut]==0,"pfs_months"])
    record(f"{mut}_pfs", diff_pos_minus_neg=diff, p=pp, mean_pos=ma, mean_neg=mb,
           summary=f"{mut}+ vs - diff={diff:.3f} p={pp:.3g}")

# PD-L1 continuous
slope, p, r2 = linreg(DF["pfs_months"], DF["pdl1_tps"])
record("pdl1_pfs", slope=slope, p=p, r2=r2,
       summary=f"PD-L1 TPS slope={slope:.3f} months per unit p={p:.3g}")

# ===== Labs continuous =====
print("\n=== Labs vs PFS ===")
labs = ["albumin_g_dl","ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr","hemoglobin_g_dl",
        "alkaline_phosphatase_u_l","ast_u_l","alt_u_l","total_bilirubin_mg_dl","creatinine_mg_dl",
        "bun_mg_dl","sodium_meq_l","potassium_meq_l","calcium_mg_dl"]
for v in labs:
    slope, p, r2 = linreg(DF["pfs_months"], DF[v])
    record(f"{v}_pfs", slope=slope, p=p, r2=r2,
           summary=f"{v} slope={slope:.4f} p={p:.3g}")

# ===== Treatment main effects =====
print("\n=== Treatment main effects (univariate) ===")
for tx in ["treatment_pembrolizumab","treatment_sotorasib","treatment_olaparib","treatment_osimertinib"]:
    diff, pp, ma, mb = ttest(DF.loc[DF[tx]==1,"pfs_months"], DF.loc[DF[tx]==0,"pfs_months"])
    record(f"{tx}_pfs", diff_on_minus_off=diff, p=pp, mean_on=ma, mean_off=mb,
           summary=f"{tx} on-off diff={diff:.3f} p={pp:.3g}")

# ===== Treatments adjusted for each other and key prognostic factors =====
print("\n=== Treatments adjusted (multivariable OLS) ===")
formula = ("pfs_months ~ treatment_pembrolizumab + treatment_sotorasib + treatment_olaparib + treatment_osimertinib"
           " + age_years + sex_female + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + ldh_u_l + nlr + crp_mg_l"
           " + weight_loss_pct_6mo + hemoglobin_g_dl + pdl1_tps"
           " + egfr_mutation + kras_g12c + alk_fusion + stk11_mutation + brca2_mutation + tmb_high")
m_full = fit_ols(formula, DF)
adj = {}
for k in m_full.params.index:
    adj[k] = {"coef": float(m_full.params[k]), "p": float(m_full.pvalues[k])}
record("multivariable_main", coefs=adj, r2=float(m_full.rsquared),
       summary=f"multivariable model R^2={m_full.rsquared:.3f}")

# ===== Treatment x biomarker interactions =====
print("\n=== Treatment x biomarker interactions ===")

# Pembrolizumab x PD-L1 TPS continuous
m = fit_ols("pfs_months ~ treatment_pembrolizumab*pdl1_tps + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + nlr", DF)
record("pembro_x_pdl1_tps",
       coef=float(m.params["treatment_pembrolizumab:pdl1_tps"]),
       p=float(m.pvalues["treatment_pembrolizumab:pdl1_tps"]),
       summary=f"pembro:pdl1_tps coef={m.params['treatment_pembrolizumab:pdl1_tps']:.3f} p={m.pvalues['treatment_pembrolizumab:pdl1_tps']:.3g}")

# Pembrolizumab x TMB
m = fit_ols("pfs_months ~ treatment_pembrolizumab*tmb_high + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + nlr", DF)
record("pembro_x_tmb_high",
       coef=float(m.params["treatment_pembrolizumab:tmb_high"]),
       p=float(m.pvalues["treatment_pembrolizumab:tmb_high"]),
       summary=f"pembro:tmb coef={m.params['treatment_pembrolizumab:tmb_high']:.3f} p={m.pvalues['treatment_pembrolizumab:tmb_high']:.3g}")

# Pembrolizumab x STK11 (known antagonistic biomarker in real-world)
m = fit_ols("pfs_months ~ treatment_pembrolizumab*stk11_mutation + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + nlr", DF)
record("pembro_x_stk11",
       coef=float(m.params["treatment_pembrolizumab:stk11_mutation"]),
       p=float(m.pvalues["treatment_pembrolizumab:stk11_mutation"]),
       summary=f"pembro:stk11 coef={m.params['treatment_pembrolizumab:stk11_mutation']:.3f} p={m.pvalues['treatment_pembrolizumab:stk11_mutation']:.3g}")

# Pembrolizumab x histology
m = fit_ols("pfs_months ~ treatment_pembrolizumab*C(histology) + ecog_ps + stage_iv + albumin_g_dl", DF)
for term in m.params.index:
    if "treatment_pembrolizumab:" in term:
        record(f"pembro_x_histology_{term}",
               coef=float(m.params[term]), p=float(m.pvalues[term]),
               summary=f"{term} coef={m.params[term]:.3f} p={m.pvalues[term]:.3g}")

# Sotorasib x KRAS G12C
m = fit_ols("pfs_months ~ treatment_sotorasib*kras_g12c + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + nlr", DF)
record("sotorasib_x_krasg12c",
       coef=float(m.params["treatment_sotorasib:kras_g12c"]),
       p=float(m.pvalues["treatment_sotorasib:kras_g12c"]),
       main_sotorasib=float(m.params["treatment_sotorasib"]),
       p_main=float(m.pvalues["treatment_sotorasib"]),
       main_kras=float(m.params["kras_g12c"]),
       p_kras=float(m.pvalues["kras_g12c"]),
       summary=f"soto:krasg12c coef={m.params['treatment_sotorasib:kras_g12c']:.3f} p={m.pvalues['treatment_sotorasib:kras_g12c']:.3g}")

# Stratified: in KRAS G12C-positive only, sotorasib effect
sub = DF[DF["kras_g12c"]==1]
diff, pp, ma, mb = ttest(sub.loc[sub["treatment_sotorasib"]==1,"pfs_months"], sub.loc[sub["treatment_sotorasib"]==0,"pfs_months"])
record("sotorasib_in_krasg12c_pos", diff=diff, p=pp, n_on=int((sub["treatment_sotorasib"]==1).sum()), n_off=int((sub["treatment_sotorasib"]==0).sum()),
       summary=f"sotorasib in KRAS+ diff={diff:.3f} p={pp:.3g}")
# In KRAS G12C-negative only
sub = DF[DF["kras_g12c"]==0]
diff, pp, ma, mb = ttest(sub.loc[sub["treatment_sotorasib"]==1,"pfs_months"], sub.loc[sub["treatment_sotorasib"]==0,"pfs_months"])
record("sotorasib_in_krasg12c_neg", diff=diff, p=pp,
       summary=f"sotorasib in KRAS- diff={diff:.3f} p={pp:.3g}")

# Osimertinib x EGFR
m = fit_ols("pfs_months ~ treatment_osimertinib*egfr_mutation + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + nlr", DF)
record("osi_x_egfr",
       coef=float(m.params["treatment_osimertinib:egfr_mutation"]),
       p=float(m.pvalues["treatment_osimertinib:egfr_mutation"]),
       main_osi=float(m.params["treatment_osimertinib"]),
       p_main=float(m.pvalues["treatment_osimertinib"]),
       main_egfr=float(m.params["egfr_mutation"]),
       p_egfr=float(m.pvalues["egfr_mutation"]),
       summary=f"osi:egfr coef={m.params['treatment_osimertinib:egfr_mutation']:.3f} p={m.pvalues['treatment_osimertinib:egfr_mutation']:.3g}")
sub = DF[DF["egfr_mutation"]==1]
diff, pp, ma, mb = ttest(sub.loc[sub["treatment_osimertinib"]==1,"pfs_months"], sub.loc[sub["treatment_osimertinib"]==0,"pfs_months"])
record("osi_in_egfr_pos", diff=diff, p=pp,
       summary=f"osi in EGFR+ diff={diff:.3f} p={pp:.3g}")
sub = DF[DF["egfr_mutation"]==0]
diff, pp, ma, mb = ttest(sub.loc[sub["treatment_osimertinib"]==1,"pfs_months"], sub.loc[sub["treatment_osimertinib"]==0,"pfs_months"])
record("osi_in_egfr_neg", diff=diff, p=pp,
       summary=f"osi in EGFR- diff={diff:.3f} p={pp:.3g}")

# Olaparib x BRCA2
m = fit_ols("pfs_months ~ treatment_olaparib*brca2_mutation + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + nlr", DF)
record("olap_x_brca2",
       coef=float(m.params["treatment_olaparib:brca2_mutation"]),
       p=float(m.pvalues["treatment_olaparib:brca2_mutation"]),
       main_olap=float(m.params["treatment_olaparib"]),
       p_main=float(m.pvalues["treatment_olaparib"]),
       main_brca2=float(m.params["brca2_mutation"]),
       p_brca2=float(m.pvalues["brca2_mutation"]),
       summary=f"olap:brca2 coef={m.params['treatment_olaparib:brca2_mutation']:.3f} p={m.pvalues['treatment_olaparib:brca2_mutation']:.3g}")
sub = DF[DF["brca2_mutation"]==1]
diff, pp, ma, mb = ttest(sub.loc[sub["treatment_olaparib"]==1,"pfs_months"], sub.loc[sub["treatment_olaparib"]==0,"pfs_months"])
record("olap_in_brca2_pos", diff=diff, p=pp, n_on=int((sub["treatment_olaparib"]==1).sum()), n_off=int((sub["treatment_olaparib"]==0).sum()),
       summary=f"olap in BRCA2+ diff={diff:.3f} p={pp:.3g}")
sub = DF[DF["brca2_mutation"]==0]
diff, pp, ma, mb = ttest(sub.loc[sub["treatment_olaparib"]==1,"pfs_months"], sub.loc[sub["treatment_olaparib"]==0,"pfs_months"])
record("olap_in_brca2_neg", diff=diff, p=pp,
       summary=f"olap in BRCA2- diff={diff:.3f} p={pp:.3g}")

# ===== Systematic treatment-by-feature interaction screening =====
print("\n=== Interaction screening per treatment ===")
candidates = ["age_years","sex_female","ecog_ps","stage_iv","has_brain_mets",
              "egfr_mutation","kras_g12c","alk_fusion","stk11_mutation","brca2_mutation",
              "tmb_high","pdl1_tps","albumin_g_dl","ldh_u_l","weight_loss_pct_6mo",
              "crp_mg_l","nlr","hemoglobin_g_dl","alkaline_phosphatase_u_l",
              "ast_u_l","alt_u_l","total_bilirubin_mg_dl","creatinine_mg_dl",
              "bun_mg_dl","sodium_meq_l","potassium_meq_l","calcium_mg_dl"]
# Add histology and smoking_status by encoding
DF2 = DF.copy()
DF2["adeno"] = (DF2["histology"]=="adenocarcinoma").astype(int)
DF2["smk_current"] = (DF2["smoking_status"]=="current").astype(int)
DF2["smk_former"] = (DF2["smoking_status"]=="former").astype(int)
DF2["smk_never"] = (DF2["smoking_status"]=="never").astype(int)
candidates += ["adeno","smk_current","smk_former","smk_never"]

scan = {}
for tx in ["treatment_pembrolizumab","treatment_sotorasib","treatment_olaparib","treatment_osimertinib"]:
    scan[tx] = []
    for c in candidates:
        try:
            m = fit_ols(f"pfs_months ~ {tx}*{c}", DF2)
            term = f"{tx}:{c}"
            if term in m.params:
                scan[tx].append({
                    "feature": c,
                    "interaction_coef": float(m.params[term]),
                    "interaction_p": float(m.pvalues[term]),
                    "main_tx_coef": float(m.params[tx]),
                    "main_tx_p": float(m.pvalues[tx]),
                })
        except Exception as e:
            pass
    scan[tx].sort(key=lambda r: r["interaction_p"])
RESULTS["interaction_scan"] = scan

# Print top 5 per treatment
for tx, items in scan.items():
    print(f"\nTop interactions for {tx}:")
    for r in items[:6]:
        print(f"  {r['feature']:25s}  coef={r['interaction_coef']:.4f}  p={r['interaction_p']:.4g}")

# ===== Joint model of strongest modifiers per treatment =====
print("\n=== Joint models of top modifiers ===")
def joint(tx, modifiers, df):
    inter = " + ".join([f"{tx}:{m}" for m in modifiers])
    main = " + ".join([tx] + modifiers)
    f = f"pfs_months ~ {main} + {inter} + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + nlr"
    return fit_ols(f, df)

# Pembrolizumab: top 3 from scan
pembro_top = [r["feature"] for r in scan["treatment_pembrolizumab"][:3]]
m = joint("treatment_pembrolizumab", pembro_top, DF2)
RESULTS["joint_pembro"] = {
    "modifiers": pembro_top,
    "coefs": {k:{"coef":float(v),"p":float(m.pvalues[k])} for k,v in m.params.items() if "treatment_pembrolizumab" in k},
}
print(f"Pembro joint top: {pembro_top}")

soto_top = [r["feature"] for r in scan["treatment_sotorasib"][:3]]
m = joint("treatment_sotorasib", soto_top, DF2)
RESULTS["joint_soto"] = {
    "modifiers": soto_top,
    "coefs": {k:{"coef":float(v),"p":float(m.pvalues[k])} for k,v in m.params.items() if "treatment_sotorasib" in k},
}
print(f"Soto joint top: {soto_top}")

osi_top = [r["feature"] for r in scan["treatment_osimertinib"][:3]]
m = joint("treatment_osimertinib", osi_top, DF2)
RESULTS["joint_osi"] = {
    "modifiers": osi_top,
    "coefs": {k:{"coef":float(v),"p":float(m.pvalues[k])} for k,v in m.params.items() if "treatment_osimertinib" in k},
}
print(f"Osi joint top: {osi_top}")

olap_top = [r["feature"] for r in scan["treatment_olaparib"][:3]]
m = joint("treatment_olaparib", olap_top, DF2)
RESULTS["joint_olap"] = {
    "modifiers": olap_top,
    "coefs": {k:{"coef":float(v),"p":float(m.pvalues[k])} for k,v in m.params.items() if "treatment_olaparib" in k},
}
print(f"Olap joint top: {olap_top}")

# ===== Subgroup discovery: candidate populations defined by combinations =====
print("\n=== Subgroup discovery (refined) ===")
def subgroup_effect(name, mask, tx, df):
    sub = df[mask]
    if len(sub) < 50:
        return None
    on = sub.loc[sub[tx]==1,"pfs_months"]; off = sub.loc[sub[tx]==0,"pfs_months"]
    if len(on)<20 or len(off)<20:
        return None
    diff = float(on.mean() - off.mean())
    p = float(stats.ttest_ind(on, off, equal_var=False).pvalue)
    rec = {"name":name,"n":int(len(sub)),"n_on":int(len(on)),"n_off":int(len(off)),
           "diff":diff,"p":p,"mean_on":float(on.mean()),"mean_off":float(off.mean())}
    return rec

subgroup_results = {}

# Sotorasib KRAS G12C+ refined by other features
candidates_subgroup = []
candidates_subgroup.append(("kras_g12c==1", DF2["kras_g12c"]==1))
candidates_subgroup.append(("kras_g12c==1 & stk11==0", (DF2["kras_g12c"]==1) & (DF2["stk11_mutation"]==0)))
candidates_subgroup.append(("kras_g12c==1 & stk11==1", (DF2["kras_g12c"]==1) & (DF2["stk11_mutation"]==1)))
candidates_subgroup.append(("kras_g12c==1 & ecog==0", (DF2["kras_g12c"]==1) & (DF2["ecog_ps"]==0)))
candidates_subgroup.append(("kras_g12c==1 & adeno==1", (DF2["kras_g12c"]==1) & (DF2["adeno"]==1)))
candidates_subgroup.append(("kras_g12c==1 & ldh<median", (DF2["kras_g12c"]==1) & (DF2["ldh_u_l"]<DF2["ldh_u_l"].median())))
subgroup_results["sotorasib"] = []
for name, mask in candidates_subgroup:
    r = subgroup_effect(name, mask, "treatment_sotorasib", DF2)
    if r: subgroup_results["sotorasib"].append(r)

# Osimertinib EGFR+
egfr_subs = [
    ("egfr==1", DF2["egfr_mutation"]==1),
    ("egfr==1 & adeno==1", (DF2["egfr_mutation"]==1) & (DF2["adeno"]==1)),
    ("egfr==1 & ecog==0", (DF2["egfr_mutation"]==1) & (DF2["ecog_ps"]==0)),
    ("egfr==1 & sex_female==1", (DF2["egfr_mutation"]==1) & (DF2["sex_female"]==1)),
    ("egfr==1 & has_brain_mets==0", (DF2["egfr_mutation"]==1) & (DF2["has_brain_mets"]==0)),
    ("egfr==1 & smk_never==1", (DF2["egfr_mutation"]==1) & (DF2["smk_never"]==1)),
    ("egfr==1 & ldh<median", (DF2["egfr_mutation"]==1) & (DF2["ldh_u_l"]<DF2["ldh_u_l"].median())),
]
subgroup_results["osimertinib"] = []
for name, mask in egfr_subs:
    r = subgroup_effect(name, mask, "treatment_osimertinib", DF2)
    if r: subgroup_results["osimertinib"].append(r)

# Olaparib BRCA2+
brca_subs = [
    ("brca2==1", DF2["brca2_mutation"]==1),
    ("brca2==1 & ecog==0", (DF2["brca2_mutation"]==1) & (DF2["ecog_ps"]==0)),
    ("brca2==1 & ecog<2", (DF2["brca2_mutation"]==1) & (DF2["ecog_ps"]<2)),
    ("brca2==1 & adeno==1", (DF2["brca2_mutation"]==1) & (DF2["adeno"]==1)),
    ("brca2==1 & albumin>=median", (DF2["brca2_mutation"]==1) & (DF2["albumin_g_dl"]>=DF2["albumin_g_dl"].median())),
]
subgroup_results["olaparib"] = []
for name, mask in brca_subs:
    r = subgroup_effect(name, mask, "treatment_olaparib", DF2)
    if r: subgroup_results["olaparib"].append(r)

# Pembrolizumab — explore PD-L1 high, TMB high, adeno, no STK11, etc.
pembro_subs = [
    ("pdl1>=0.5", DF2["pdl1_tps"]>=0.5),
    ("tmb_high==1", DF2["tmb_high"]==1),
    ("pdl1>=0.5 & tmb_high==1", (DF2["pdl1_tps"]>=0.5) & (DF2["tmb_high"]==1)),
    ("pdl1>=0.5 & stk11==0", (DF2["pdl1_tps"]>=0.5) & (DF2["stk11_mutation"]==0)),
    ("pdl1>=0.5 & stk11==1", (DF2["pdl1_tps"]>=0.5) & (DF2["stk11_mutation"]==1)),
    ("pdl1>=0.5 & adeno==1", (DF2["pdl1_tps"]>=0.5) & (DF2["adeno"]==1)),
    ("pdl1>=0.5 & ecog==0", (DF2["pdl1_tps"]>=0.5) & (DF2["ecog_ps"]==0)),
    ("tmb_high==1 & stk11==0", (DF2["tmb_high"]==1) & (DF2["stk11_mutation"]==0)),
    ("pdl1>=0.5 & tmb_high==1 & stk11==0", (DF2["pdl1_tps"]>=0.5) & (DF2["tmb_high"]==1) & (DF2["stk11_mutation"]==0)),
    ("pdl1>=0.5 & tmb_high==1 & stk11==0 & ecog==0", (DF2["pdl1_tps"]>=0.5) & (DF2["tmb_high"]==1) & (DF2["stk11_mutation"]==0) & (DF2["ecog_ps"]==0)),
]
subgroup_results["pembrolizumab"] = []
for name, mask in pembro_subs:
    r = subgroup_effect(name, mask, "treatment_pembrolizumab", DF2)
    if r: subgroup_results["pembrolizumab"].append(r)

RESULTS["subgroup_results"] = subgroup_results

for tx, recs in subgroup_results.items():
    print(f"\n{tx}:")
    for r in recs:
        print(f"  {r['name']:50s}  n={r['n']:5d}  diff={r['diff']:+.3f}  p={r['p']:.3g}")

# ===== Stratification grids: 3-way for each treatment-biomarker subgroup =====
print("\n=== Three-way subgroup checks ===")
# pembro sub-subgroups inside pdl1>=0.5
g = DF2[DF2["pdl1_tps"]>=0.5]
threeway = []
for col in ["stk11_mutation","tmb_high","ecog_ps","stage_iv","has_brain_mets"]:
    for v in sorted(g[col].unique()):
        m = g[g[col]==v]
        on = m.loc[m["treatment_pembrolizumab"]==1,"pfs_months"]
        off = m.loc[m["treatment_pembrolizumab"]==0,"pfs_months"]
        if len(on)>20 and len(off)>20:
            d = float(on.mean()-off.mean())
            p = float(stats.ttest_ind(on,off,equal_var=False).pvalue)
            threeway.append({"stratum":f"pdl1>=0.5 & {col}={v}","n":int(len(m)),"diff":d,"p":p})
RESULTS["pembro_pdl1_threeway"] = threeway

# Save
with open("results_full.json","w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\nSaved results_full.json")
