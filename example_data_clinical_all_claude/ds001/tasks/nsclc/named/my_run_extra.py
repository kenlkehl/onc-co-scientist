"""Additional focused analyses on heterogeneity within positive subgroups."""
import json, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
df["smoking_current"] = (df["smoking_status"]=="current").astype(int)
df["smoking_former"]  = (df["smoking_status"]=="former").astype(int)
df["smoking_never"]   = (df["smoking_status"]=="never").astype(int)
df["adeno"] = (df["histology"]=="adenocarcinoma").astype(int)
df["pdl1_high"] = (df["pdl1_tps"]>=0.5).astype(int)

OUT = {}

def subg(treat, mask, label):
    sub = df[mask]
    if sub.empty or sub[treat].nunique()<2:
        return {"label": label, "n": int(sub.shape[0]), "issue":"no var"}
    m = smf.ols(f"pfs_months ~ {treat}", data=sub).fit()
    coef = float(m.params[treat]); p = float(m.pvalues[treat])
    return {"label":label,"n":int(sub.shape[0]),
            "n_on":int(sub[treat].sum()),"n_off":int((sub[treat]==0).sum()),
            "mean_on":float(sub.loc[sub[treat]==1,"pfs_months"].mean()),
            "mean_off":float(sub.loc[sub[treat]==0,"pfs_months"].mean()),
            "coef":coef,"p":p}

# Sotorasib heterogeneity within KRAS G12C+ patients
print("=== Sotorasib heterogeneity within KRAS G12C+ ===")
kras = df["kras_g12c"]==1
heter = {}
specs = [
    ("kras+ & sex_female==0",      kras & (df["sex_female"]==0)),
    ("kras+ & sex_female==1",      kras & (df["sex_female"]==1)),
    ("kras+ & smoking_current==1", kras & (df["smoking_current"]==1)),
    ("kras+ & smoking_current==0", kras & (df["smoking_current"]==0)),
    ("kras+ & smoking_never==1",   kras & (df["smoking_never"]==1)),
    ("kras+ & smoking_never==0",   kras & (df["smoking_never"]==0)),
    ("kras+ & adeno==1",           kras & (df["adeno"]==1)),
    ("kras+ & adeno==0",           kras & (df["adeno"]==0)),
    ("kras+ & ecog<=1",            kras & (df["ecog_ps"]<=1)),
    ("kras+ & ecog==2",            kras & (df["ecog_ps"]==2)),
    ("kras+ & stage_iv==0",        kras & (df["stage_iv"]==0)),
    ("kras+ & stage_iv==1",        kras & (df["stage_iv"]==1)),
    ("kras+ & has_brain_mets==0",  kras & (df["has_brain_mets"]==0)),
    ("kras+ & has_brain_mets==1",  kras & (df["has_brain_mets"]==1)),
    ("kras+ & stk11==1",           kras & (df["stk11_mutation"]==1)),
    ("kras+ & stk11==0",           kras & (df["stk11_mutation"]==0)),
    ("kras+ & alk_fusion==0",      kras & (df["alk_fusion"]==0)),
    ("kras+ & egfr==0",            kras & (df["egfr_mutation"]==0)),
    ("kras+ & albumin>=median",    kras & (df["albumin_g_dl"]>=df["albumin_g_dl"].median())),
    ("kras+ & albumin<median",     kras & (df["albumin_g_dl"]<df["albumin_g_dl"].median())),
    ("kras+ & ldh>=median",        kras & (df["ldh_u_l"]>=df["ldh_u_l"].median())),
    ("kras+ & ldh<median",         kras & (df["ldh_u_l"]<df["ldh_u_l"].median())),
]
for lab, mask in specs:
    r = subg("treatment_sotorasib", mask, lab)
    heter[lab]=r
    print(f"  {lab:35s}: n={r.get('n')} on={r.get('n_on')} mean_on={r.get('mean_on'):.3f} mean_off={r.get('mean_off'):.3f} beta={r.get('coef'):+.3f} p={r.get('p'):.2e}")
OUT["sotorasib_kras_heter"] = heter

# Formal interactions within KRAS G12C+
print("\n=== Formal sex/smoking interactions within KRAS G12C+ ===")
sub = df[kras]
intxn = {}
for f in ["sex_female","smoking_current","smoking_never","adeno","ecog_ps","stage_iv","has_brain_mets","stk11_mutation","albumin_g_dl","ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr","tmb_high","brca2_mutation"]:
    m = smf.ols(f"pfs_months ~ treatment_sotorasib*{f}", data=sub).fit()
    intxn[f] = {"main_t": float(m.params["treatment_sotorasib"]),
                "main_t_p": float(m.pvalues["treatment_sotorasib"]),
                "intxn_coef": float(m.params[f"treatment_sotorasib:{f}"]),
                "intxn_p":   float(m.pvalues[f"treatment_sotorasib:{f}"])}
    print(f"  {f:25s}: intxn beta={intxn[f]['intxn_coef']:+.4f} p={intxn[f]['intxn_p']:.2e}")
OUT["sotorasib_in_kras_intxn"] = intxn

# Three-way: sotorasib x kras x sex
print("\n=== 3-way sotorasib x kras x sex ===")
m = smf.ols("pfs_months ~ treatment_sotorasib*kras_g12c*sex_female", data=df).fit()
print(m.summary().tables[1])
OUT["sotorasib_kras_sex_3way"] = {k:{"coef":float(m.params[k]),"p":float(m.pvalues[k])} for k in m.params.index}

# Joint subgroup definitions: female KRAS+ vs male KRAS+ for sotorasib
print("\n=== KRAS+ male vs KRAS+ female sotorasib effect ===")
for lab, mask in [
    ("kras+ & male",   kras & (df["sex_female"]==0)),
    ("kras+ & female", kras & (df["sex_female"]==1)),
]:
    r = subg("treatment_sotorasib", mask, lab)
    print(f"  {lab}: {r}")

# Pembrolizumab heterogeneity within PDL1-high
print("\n=== Pembrolizumab heterogeneity within PDL1-high ===")
ph = df["pdl1_tps"]>=0.5
pem_h = {}
for lab, mask in [
    ("pdl1high & stk11==0", ph & (df["stk11_mutation"]==0)),
    ("pdl1high & stk11==1", ph & (df["stk11_mutation"]==1)),
    ("pdl1high & stk11==0 & ecog<=1", ph & (df["stk11_mutation"]==0) & (df["ecog_ps"]<=1)),
    ("pdl1high & stk11==0 & ecog==0", ph & (df["stk11_mutation"]==0) & (df["ecog_ps"]==0)),
    ("pdl1high & stk11==0 & weight_loss<5", ph & (df["stk11_mutation"]==0) & (df["weight_loss_pct_6mo"]<5)),
    ("pdl1high & stk11==0 & ldh<median",   ph & (df["stk11_mutation"]==0) & (df["ldh_u_l"]<df["ldh_u_l"].median())),
    ("pdl1high & ecog<=1 & weight_loss<5", ph & (df["ecog_ps"]<=1) & (df["weight_loss_pct_6mo"]<5)),
    ("pdl1high & smoking_never==0", ph & (df["smoking_never"]==0)),
    ("pdl1high & smoking_never==1", ph & (df["smoking_never"]==1)),
]:
    r = subg("treatment_pembrolizumab", mask, lab)
    pem_h[lab]=r
    print(f"  {lab:50s}: {r}")
OUT["pembro_pdl1_high_heter"] = pem_h

# Pembrolizumab x weight_loss interaction
print("\n=== Pembro x weight_loss x stage_iv ===")
m = smf.ols("pfs_months ~ treatment_pembrolizumab*weight_loss_pct_6mo", data=df).fit()
OUT["pem_x_wl"] = {k:{"coef":float(m.params[k]),"p":float(m.pvalues[k])} for k in m.params.index}
print(m.summary().tables[1])

m = smf.ols("pfs_months ~ treatment_pembrolizumab*stage_iv", data=df).fit()
OUT["pem_x_stage"] = {k:{"coef":float(m.params[k]),"p":float(m.pvalues[k])} for k in m.params.index}
print(m.summary().tables[1])

# Olaparib among brca2+
print("\n=== Olaparib heterogeneity within BRCA2+ ===")
br = df["brca2_mutation"]==1
ola_h = {}
for lab, mask in [
    ("brca2+ & ecog==0", br & (df["ecog_ps"]==0)),
    ("brca2+ & sex_female==1", br & (df["sex_female"]==1)),
    ("brca2+ & sex_female==0", br & (df["sex_female"]==0)),
    ("brca2+ & adeno==1", br & (df["adeno"]==1)),
    ("brca2+ & stage_iv==0", br & (df["stage_iv"]==0)),
    ("brca2+ & has_brain_mets==0", br & (df["has_brain_mets"]==0)),
    ("brca2+ & albumin>=median", br & (df["albumin_g_dl"]>=df["albumin_g_dl"].median())),
]:
    r = subg("treatment_olaparib", mask, lab)
    ola_h[lab]=r
    print(f"  {lab:40s}: {r}")
OUT["olaparib_brca_heter"] = ola_h

# Osimertinib among egfr+
print("\n=== Osimertinib heterogeneity within EGFR+ ===")
eg = df["egfr_mutation"]==1
osi_h = {}
for lab, mask in [
    ("egfr+ & ecog==0", eg & (df["ecog_ps"]==0)),
    ("egfr+ & sex_female==1", eg & (df["sex_female"]==1)),
    ("egfr+ & sex_female==0", eg & (df["sex_female"]==0)),
    ("egfr+ & smoking_never==1", eg & (df["smoking_never"]==1)),
    ("egfr+ & smoking_never==0", eg & (df["smoking_never"]==0)),
    ("egfr+ & adeno==1", eg & (df["adeno"]==1)),
    ("egfr+ & has_brain_mets==0", eg & (df["has_brain_mets"]==0)),
    ("egfr+ & has_brain_mets==1", eg & (df["has_brain_mets"]==1)),
    ("egfr+ & stage_iv==0", eg & (df["stage_iv"]==0)),
    ("egfr+ & stage_iv==1", eg & (df["stage_iv"]==1)),
]:
    r = subg("treatment_osimertinib", mask, lab)
    osi_h[lab]=r
    print(f"  {lab:40s}: {r}")
OUT["osimertinib_egfr_heter"] = osi_h

with open("results_extra.json","w") as f:
    json.dump(OUT,f,indent=2,default=str)
print("\nSaved to results_extra.json")
