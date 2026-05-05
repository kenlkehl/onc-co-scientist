"""
Analysis script for ds001_nsclc.
Runs all analyses needed for the iteration transcript and prints structured results.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json, itertools, warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
np.random.seed(0)

results = {}

def record(name, **kw):
    results[name] = kw
    print(f"--- {name} ---")
    for k, v in kw.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4g}")
        else:
            print(f"  {k}: {v}")

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def ttest(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(np.mean(a) - np.mean(b)), float(p)

def ols_coef(formula, data=df):
    m = smf.ols(formula, data=data).fit()
    return m

def trt_effect(trt, sub=None, data=df):
    d = data if sub is None else data.loc[sub]
    a = d.loc[d[trt]==1, "pfs_months"]
    b = d.loc[d[trt]==0, "pfs_months"]
    if len(a) < 5 or len(b) < 5:
        return None
    diff, p = ttest(a, b)
    return {"n_treated": int(len(a)), "n_control": int(len(b)),
            "mean_treated": float(np.mean(a)), "mean_control": float(np.mean(b)),
            "diff": diff, "p_value": p}

# ----------------------------------------------------------
# ITER 1 — main effect of demographic / clinical features on PFS
# ----------------------------------------------------------
# age: continuous OLS
m = smf.ols("pfs_months ~ age_years", data=df).fit()
record("age_main",
       beta=float(m.params["age_years"]), p=float(m.pvalues["age_years"]))

# sex
diff, p = ttest(df.loc[df.sex_female==1,"pfs_months"], df.loc[df.sex_female==0,"pfs_months"])
record("sex_female_main", diff=diff, p=p,
       mean_f=float(df.loc[df.sex_female==1,"pfs_months"].mean()),
       mean_m=float(df.loc[df.sex_female==0,"pfs_months"].mean()))

# smoking
groups = [df.loc[df.smoking_status==s,"pfs_months"] for s in ["never","former","current"]]
F, p = stats.f_oneway(*groups)
record("smoking_anova", F=float(F), p=float(p),
       mean_never=float(groups[0].mean()), mean_former=float(groups[1].mean()),
       mean_current=float(groups[2].mean()))

# ECOG
m = smf.ols("pfs_months ~ ecog_ps", data=df).fit()
record("ecog_main", beta=float(m.params["ecog_ps"]), p=float(m.pvalues["ecog_ps"]))

# Histology
diff, p = ttest(df.loc[df.histology=="adenocarcinoma","pfs_months"],
                df.loc[df.histology=="squamous","pfs_months"])
record("histology_main", diff_adeno_minus_squamous=diff, p=p)

# Stage IV
diff, p = ttest(df.loc[df.stage_iv==1,"pfs_months"], df.loc[df.stage_iv==0,"pfs_months"])
record("stage_iv_main", diff=diff, p=p)

# Brain mets
diff, p = ttest(df.loc[df.has_brain_mets==1,"pfs_months"], df.loc[df.has_brain_mets==0,"pfs_months"])
record("brain_mets_main", diff=diff, p=p)

# ----------------------------------------------------------
# ITER 2 — main effect of mutations / biomarkers
# ----------------------------------------------------------
for c in ["egfr_mutation","kras_g12c","alk_fusion","stk11_mutation","brca2_mutation","tmb_high"]:
    diff, p = ttest(df.loc[df[c]==1,"pfs_months"], df.loc[df[c]==0,"pfs_months"])
    record(f"{c}_main", diff=diff, p=p,
           n_pos=int((df[c]==1).sum()), mean_pos=float(df.loc[df[c]==1,"pfs_months"].mean()),
           mean_neg=float(df.loc[df[c]==0,"pfs_months"].mean()))

# pdl1_tps (continuous)
m = smf.ols("pfs_months ~ pdl1_tps", data=df).fit()
record("pdl1_tps_main", beta=float(m.params["pdl1_tps"]), p=float(m.pvalues["pdl1_tps"]))

# ----------------------------------------------------------
# ITER 3 — main effect of labs / inflammation markers
# ----------------------------------------------------------
for c in ["albumin_g_dl","ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr",
          "hemoglobin_g_dl","alkaline_phosphatase_u_l","ast_u_l","alt_u_l",
          "total_bilirubin_mg_dl","creatinine_mg_dl","bun_mg_dl",
          "sodium_meq_l","potassium_meq_l","calcium_mg_dl"]:
    m = smf.ols(f"pfs_months ~ {c}", data=df).fit()
    record(f"{c}_main", beta=float(m.params[c]), p=float(m.pvalues[c]))

# ----------------------------------------------------------
# ITER 4 — overall main effect of each treatment on PFS
# ----------------------------------------------------------
for t in ["treatment_pembrolizumab","treatment_sotorasib","treatment_olaparib","treatment_osimertinib"]:
    eff = trt_effect(t)
    record(f"{t}_main", **eff)

# ----------------------------------------------------------
# ITER 5 — pembrolizumab × pdl1_tps interaction (continuous)
# ----------------------------------------------------------
m = smf.ols("pfs_months ~ treatment_pembrolizumab * pdl1_tps", data=df).fit()
record("pembro_x_pdl1",
       beta_main=float(m.params["treatment_pembrolizumab"]),
       beta_inter=float(m.params["treatment_pembrolizumab:pdl1_tps"]),
       p_main=float(m.pvalues["treatment_pembrolizumab"]),
       p_inter=float(m.pvalues["treatment_pembrolizumab:pdl1_tps"]))

# Subgroup: high pdl1 (>=0.5) vs low
df["pdl1_high"] = (df["pdl1_tps"]>=0.5).astype(int)
eff_hi = trt_effect("treatment_pembrolizumab", df["pdl1_high"]==1)
eff_lo = trt_effect("treatment_pembrolizumab", df["pdl1_high"]==0)
record("pembro_pdl1_subgroup", high_pdl1=eff_hi, low_pdl1=eff_lo)

# ----------------------------------------------------------
# ITER 6 — pembrolizumab × TMB
# ----------------------------------------------------------
m = smf.ols("pfs_months ~ treatment_pembrolizumab * tmb_high", data=df).fit()
record("pembro_x_tmb",
       beta_main=float(m.params["treatment_pembrolizumab"]),
       beta_inter=float(m.params["treatment_pembrolizumab:tmb_high"]),
       p_main=float(m.pvalues["treatment_pembrolizumab"]),
       p_inter=float(m.pvalues["treatment_pembrolizumab:tmb_high"]))
eff_tmb = trt_effect("treatment_pembrolizumab", df["tmb_high"]==1)
eff_no  = trt_effect("treatment_pembrolizumab", df["tmb_high"]==0)
record("pembro_tmb_subgroup", tmb_high=eff_tmb, tmb_low=eff_no)

# ----------------------------------------------------------
# ITER 7 — pembrolizumab × STK11
# ----------------------------------------------------------
m = smf.ols("pfs_months ~ treatment_pembrolizumab * stk11_mutation", data=df).fit()
record("pembro_x_stk11",
       beta_main=float(m.params["treatment_pembrolizumab"]),
       beta_inter=float(m.params["treatment_pembrolizumab:stk11_mutation"]),
       p_main=float(m.pvalues["treatment_pembrolizumab"]),
       p_inter=float(m.pvalues["treatment_pembrolizumab:stk11_mutation"]))
eff_stk = trt_effect("treatment_pembrolizumab", df["stk11_mutation"]==1)
eff_no  = trt_effect("treatment_pembrolizumab", df["stk11_mutation"]==0)
record("pembro_stk11_subgroup", stk11_mut=eff_stk, stk11_wt=eff_no)

# ----------------------------------------------------------
# ITER 8 — sotorasib × kras_g12c
# ----------------------------------------------------------
m = smf.ols("pfs_months ~ treatment_sotorasib * kras_g12c", data=df).fit()
record("sotorasib_x_krasg12c",
       beta_main=float(m.params["treatment_sotorasib"]),
       beta_inter=float(m.params["treatment_sotorasib:kras_g12c"]),
       p_main=float(m.pvalues["treatment_sotorasib"]),
       p_inter=float(m.pvalues["treatment_sotorasib:kras_g12c"]))
eff_pos = trt_effect("treatment_sotorasib", df["kras_g12c"]==1)
eff_neg = trt_effect("treatment_sotorasib", df["kras_g12c"]==0)
record("sotorasib_kras_subgroup", kras_pos=eff_pos, kras_neg=eff_neg)

# ----------------------------------------------------------
# ITER 9 — olaparib × brca2
# ----------------------------------------------------------
m = smf.ols("pfs_months ~ treatment_olaparib * brca2_mutation", data=df).fit()
record("olaparib_x_brca2",
       beta_main=float(m.params["treatment_olaparib"]),
       beta_inter=float(m.params["treatment_olaparib:brca2_mutation"]),
       p_main=float(m.pvalues["treatment_olaparib"]),
       p_inter=float(m.pvalues["treatment_olaparib:brca2_mutation"]))
eff_pos = trt_effect("treatment_olaparib", df["brca2_mutation"]==1)
eff_neg = trt_effect("treatment_olaparib", df["brca2_mutation"]==0)
record("olaparib_brca2_subgroup", brca2_pos=eff_pos, brca2_neg=eff_neg)

# ----------------------------------------------------------
# ITER 10 — osimertinib × egfr_mutation
# ----------------------------------------------------------
m = smf.ols("pfs_months ~ treatment_osimertinib * egfr_mutation", data=df).fit()
record("osimertinib_x_egfr",
       beta_main=float(m.params["treatment_osimertinib"]),
       beta_inter=float(m.params["treatment_osimertinib:egfr_mutation"]),
       p_main=float(m.pvalues["treatment_osimertinib"]),
       p_inter=float(m.pvalues["treatment_osimertinib:egfr_mutation"]))
eff_pos = trt_effect("treatment_osimertinib", df["egfr_mutation"]==1)
eff_neg = trt_effect("treatment_osimertinib", df["egfr_mutation"]==0)
record("osimertinib_egfr_subgroup", egfr_pos=eff_pos, egfr_neg=eff_neg)

# ----------------------------------------------------------
# ITER 11 — multivariable model of PFS on all features (no treatments)
# ----------------------------------------------------------
features = ["age_years","sex_female","C(smoking_status)","ecog_ps","C(histology)",
            "stage_iv","has_brain_mets","egfr_mutation","kras_g12c","alk_fusion",
            "stk11_mutation","brca2_mutation","pdl1_tps","tmb_high","albumin_g_dl",
            "ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr","hemoglobin_g_dl",
            "alkaline_phosphatase_u_l","ast_u_l","alt_u_l","total_bilirubin_mg_dl",
            "creatinine_mg_dl","bun_mg_dl","sodium_meq_l","potassium_meq_l","calcium_mg_dl"]
formula = "pfs_months ~ " + " + ".join(features)
m_clin = smf.ols(formula, data=df).fit()
clin_coefs = {k: (float(m_clin.params[k]), float(m_clin.pvalues[k])) for k in m_clin.params.index}
record("multivar_clinical_R2", R2=float(m_clin.rsquared), n=int(m_clin.nobs))
print("Top features by |coef|:")
sorted_coefs = sorted(clin_coefs.items(), key=lambda kv: abs(kv[1][0]), reverse=True)
for k,(b,p) in sorted_coefs[:15]:
    print(f"  {k}: beta={b:.4f}, p={p:.3g}")

# ----------------------------------------------------------
# ITER 12 — multivariable model adding treatments and key interactions
# ----------------------------------------------------------
formula_t = (formula
             + " + treatment_pembrolizumab*pdl1_tps + treatment_pembrolizumab*tmb_high"
             + " + treatment_pembrolizumab*stk11_mutation"
             + " + treatment_sotorasib*kras_g12c"
             + " + treatment_olaparib*brca2_mutation"
             + " + treatment_osimertinib*egfr_mutation"
             + " + treatment_osimertinib*has_brain_mets")
m_full = smf.ols(formula_t, data=df).fit()
record("multivar_full_R2", R2=float(m_full.rsquared), n=int(m_full.nobs))
keys_of_interest = ["treatment_pembrolizumab","treatment_pembrolizumab:pdl1_tps",
                    "treatment_pembrolizumab:tmb_high","treatment_pembrolizumab:stk11_mutation",
                    "treatment_sotorasib","treatment_sotorasib:kras_g12c",
                    "treatment_olaparib","treatment_olaparib:brca2_mutation",
                    "treatment_osimertinib","treatment_osimertinib:egfr_mutation",
                    "treatment_osimertinib:has_brain_mets"]
for k in keys_of_interest:
    if k in m_full.params:
        record(f"full_{k}",
               beta=float(m_full.params[k]), p=float(m_full.pvalues[k]))

# ----------------------------------------------------------
# ITER 13–16 — heterogeneity scan: pembro/sotorasib/olaparib/osimertinib × every modifier
# ----------------------------------------------------------
modifiers = ["sex_female","ecog_ps","stage_iv","has_brain_mets","egfr_mutation","kras_g12c",
             "alk_fusion","stk11_mutation","brca2_mutation","tmb_high","pdl1_high",
             "smoking_status","histology"]

def het_scan(trt):
    rows = []
    for mod in modifiers:
        if mod in ("smoking_status","histology"):
            f = f"pfs_months ~ {trt} * C({mod})"
        else:
            f = f"pfs_months ~ {trt} * {mod}"
        try:
            mm = smf.ols(f, data=df).fit()
            # find interaction terms
            inter_terms = [n for n in mm.params.index if n.startswith(f"{trt}:")]
            best = None
            for it in inter_terms:
                rows.append({"trt":trt,"mod":mod,"term":it,
                             "beta":float(mm.params[it]),"p":float(mm.pvalues[it])})
        except Exception as e:
            rows.append({"trt":trt,"mod":mod,"term":"err","beta":np.nan,"p":np.nan,"err":str(e)})
    return rows

het_rows = []
for trt in ["treatment_pembrolizumab","treatment_sotorasib","treatment_olaparib","treatment_osimertinib"]:
    het_rows.extend(het_scan(trt))
het = pd.DataFrame(het_rows)
print("\nFull heterogeneity scan (all interaction terms):")
print(het.sort_values("p").head(40).to_string())
results["heterogeneity_scan"] = het.to_dict(orient="records")

# ----------------------------------------------------------
# ITER 17 — pembro within PD-L1 high & TMB high & STK11 wt (compound)
# ----------------------------------------------------------
sub = (df["pdl1_high"]==1) & (df["tmb_high"]==1) & (df["stk11_mutation"]==0)
eff = trt_effect("treatment_pembrolizumab", sub)
record("pembro_pdl1hi_tmbhi_stk11wt", **(eff or {}))
sub2 = (df["pdl1_high"]==1) & (df["tmb_high"]==1) & (df["stk11_mutation"]==1)
eff2 = trt_effect("treatment_pembrolizumab", sub2)
record("pembro_pdl1hi_tmbhi_stk11mut", **(eff2 or {}))

# Also report each axis individually within the compound
sub3 = (df["pdl1_high"]==1) & (df["tmb_high"]==1)
record("pembro_pdl1hi_tmbhi_any_stk11", **(trt_effect("treatment_pembrolizumab", sub3) or {}))
sub4 = (df["pdl1_high"]==1) & (df["stk11_mutation"]==0)
record("pembro_pdl1hi_stk11wt", **(trt_effect("treatment_pembrolizumab", sub4) or {}))
sub5 = (df["tmb_high"]==1) & (df["stk11_mutation"]==0)
record("pembro_tmbhi_stk11wt", **(trt_effect("treatment_pembrolizumab", sub5) or {}))

# Also: triple-interaction model in the high-pdl1 group
m_pdl1hi = smf.ols("pfs_months ~ treatment_pembrolizumab*tmb_high*stk11_mutation",
                   data=df.loc[df["pdl1_high"]==1]).fit()
print("\n3-way model (pembro × tmb_high × stk11) within PD-L1 high:")
for k in m_pdl1hi.params.index:
    if "treatment" in k:
        print(f"  {k}: beta={m_pdl1hi.params[k]:.3f}, p={m_pdl1hi.pvalues[k]:.3g}")
results["pembro_3way_in_pdl1hi"] = {k: (float(m_pdl1hi.params[k]), float(m_pdl1hi.pvalues[k]))
                                    for k in m_pdl1hi.params.index if "treatment" in k}

# ----------------------------------------------------------
# ITER 18 — osimertinib × egfr × has_brain_mets
# ----------------------------------------------------------
sub_a = (df["egfr_mutation"]==1) & (df["has_brain_mets"]==1)
sub_b = (df["egfr_mutation"]==1) & (df["has_brain_mets"]==0)
sub_c = (df["egfr_mutation"]==0) & (df["has_brain_mets"]==1)
sub_d = (df["egfr_mutation"]==0) & (df["has_brain_mets"]==0)
record("osi_egfrpos_brainpos", **(trt_effect("treatment_osimertinib", sub_a) or {}))
record("osi_egfrpos_brainneg", **(trt_effect("treatment_osimertinib", sub_b) or {}))
record("osi_egfrneg_brainpos", **(trt_effect("treatment_osimertinib", sub_c) or {}))
record("osi_egfrneg_brainneg", **(trt_effect("treatment_osimertinib", sub_d) or {}))
m3 = smf.ols("pfs_months ~ treatment_osimertinib*egfr_mutation*has_brain_mets", data=df).fit()
print("\nOsimertinib 3-way interaction (egfr × brain_mets):")
for k in m3.params.index:
    if "treatment" in k:
        print(f"  {k}: beta={m3.params[k]:.3f}, p={m3.pvalues[k]:.3g}")
results["osi_3way"] = {k: (float(m3.params[k]), float(m3.pvalues[k]))
                       for k in m3.params.index if "treatment" in k}

# ----------------------------------------------------------
# ITER 19 — sotorasib × kras × stk11 (STK11 known to suppress sotorasib too?)
# ----------------------------------------------------------
m_so = smf.ols("pfs_months ~ treatment_sotorasib*kras_g12c*stk11_mutation", data=df).fit()
for k in m_so.params.index:
    if "treatment" in k:
        print(f"  {k}: beta={m_so.params[k]:.3f}, p={m_so.pvalues[k]:.3g}")
results["sotorasib_3way"] = {k: (float(m_so.params[k]), float(m_so.pvalues[k]))
                             for k in m_so.params.index if "treatment" in k}
sub = (df["kras_g12c"]==1)&(df["stk11_mutation"]==0)
record("sotorasib_krasg12c_stk11wt", **(trt_effect("treatment_sotorasib", sub) or {}))
sub = (df["kras_g12c"]==1)&(df["stk11_mutation"]==1)
record("sotorasib_krasg12c_stk11mut", **(trt_effect("treatment_sotorasib", sub) or {}))

# ----------------------------------------------------------
# ITER 20 — olaparib in BRCA2+ further refined
# ----------------------------------------------------------
m_ol = smf.ols("pfs_months ~ treatment_olaparib*brca2_mutation*histology", data=df).fit()
for k in m_ol.params.index:
    if "treatment" in k:
        print(f"  {k}: beta={m_ol.params[k]:.3f}, p={m_ol.pvalues[k]:.3g}")

# ----------------------------------------------------------
# ITER 21 — ECOG modifies response (clinical fitness)
# ----------------------------------------------------------
for trt in ["treatment_pembrolizumab","treatment_sotorasib","treatment_olaparib","treatment_osimertinib"]:
    for ecog in [0,1,2]:
        eff = trt_effect(trt, df["ecog_ps"]==ecog)
        if eff:
            record(f"{trt}_ecog{ecog}", **eff)

# ----------------------------------------------------------
# ITER 22 — Smoking status modifies pembro?
# ----------------------------------------------------------
for s in ["never","former","current"]:
    eff = trt_effect("treatment_pembrolizumab", df["smoking_status"]==s)
    record(f"pembro_smoke_{s}", **(eff or {}))

# ----------------------------------------------------------
# ITER 23 — sex × pembrolizumab
# ----------------------------------------------------------
for sx in [0,1]:
    eff = trt_effect("treatment_pembrolizumab", df["sex_female"]==sx)
    record(f"pembro_sex_{sx}", **(eff or {}))

# ----------------------------------------------------------
# ITER 24 — Final compound subgroup definitions for each treatment
# ----------------------------------------------------------
# Pembrolizumab: best subgroup PD-L1≥0.5 AND TMB high AND STK11 wt (and never/former smoker?)
sub = (df["pdl1_high"]==1) & (df["tmb_high"]==1) & (df["stk11_mutation"]==0)
record("FINAL_pembro_subgroup", **(trt_effect("treatment_pembrolizumab", sub) or {}))
# Sotorasib: KRAS G12C+ (and STK11 wt?)
sub = (df["kras_g12c"]==1)
record("FINAL_sotorasib_subgroup", **(trt_effect("treatment_sotorasib", sub) or {}))
sub = (df["kras_g12c"]==1)&(df["stk11_mutation"]==0)
record("FINAL_sotorasib_subgroup_strict", **(trt_effect("treatment_sotorasib", sub) or {}))
# Olaparib: BRCA2+
sub = (df["brca2_mutation"]==1)
record("FINAL_olaparib_subgroup", **(trt_effect("treatment_olaparib", sub) or {}))
# Osimertinib: EGFR+
sub = (df["egfr_mutation"]==1)
record("FINAL_osimertinib_subgroup", **(trt_effect("treatment_osimertinib", sub) or {}))

# Save raw results
with open("raw_analysis_results.json","w") as f:
    json.dump(results, f, default=str, indent=2)
print("\nSaved raw_analysis_results.json")
