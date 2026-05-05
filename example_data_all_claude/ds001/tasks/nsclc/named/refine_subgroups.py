"""Refine sotorasib subgroup using interaction-screen modifiers."""
import json, numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")
DF["adeno"] = (DF["histology"]=="adenocarcinoma").astype(int)
DF["smk_current"] = (DF["smoking_status"]=="current").astype(int)
DF["smk_former"] = (DF["smoking_status"]=="former").astype(int)
DF["smk_never"] = (DF["smoking_status"]=="never").astype(int)

OUT = {}

def stratum(name, mask, tx="treatment_sotorasib"):
    s = DF[mask]
    on = s.loc[s[tx]==1,"pfs_months"]; off=s.loc[s[tx]==0,"pfs_months"]
    if len(on)<5 or len(off)<5: return None
    d=float(on.mean()-off.mean()); p=float(stats.ttest_ind(on,off,equal_var=False).pvalue)
    return {"name":name,"n":int(len(s)),"n_on":int(len(on)),"n_off":int(len(off)),
            "mean_on":float(on.mean()),"mean_off":float(off.mean()),"diff":d,"p":p}

print("=== Refining sotorasib subgroup ===")
# In KRAS+, examine effect by suppressors
m_pos = DF["kras_g12c"]==1
res = []
res.append(stratum("KRAS+ all", m_pos))
res.append(stratum("KRAS+ & sex_female==0", m_pos & (DF["sex_female"]==0)))
res.append(stratum("KRAS+ & sex_female==1", m_pos & (DF["sex_female"]==1)))
res.append(stratum("KRAS+ & smk_never==0", m_pos & (DF["smk_never"]==0)))
res.append(stratum("KRAS+ & smk_never==1", m_pos & (DF["smk_never"]==1)))
res.append(stratum("KRAS+ & alk_fusion==0", m_pos & (DF["alk_fusion"]==0)))
res.append(stratum("KRAS+ & alk_fusion==1", m_pos & (DF["alk_fusion"]==1)))
res.append(stratum("KRAS+ & egfr_mutation==0", m_pos & (DF["egfr_mutation"]==0)))
res.append(stratum("KRAS+ & egfr_mutation==1", m_pos & (DF["egfr_mutation"]==1)))
# Combined "favorable" subgroup
combined_mask = m_pos & (DF["sex_female"]==0) & (DF["smk_never"]==0) & (DF["alk_fusion"]==0) & (DF["egfr_mutation"]==0)
res.append(stratum("KRAS+ & male & ever-smoker & ALK- & EGFR-", combined_mask))
unfav_mask = m_pos & ((DF["sex_female"]==1) | (DF["smk_never"]==1) | (DF["alk_fusion"]==1) | (DF["egfr_mutation"]==1))
res.append(stratum("KRAS+ but any unfavorable modifier", unfav_mask))
for r in res:
    if r: print(f"  {r['name']:50s}  n={r['n']:5d}  diff={r['diff']:+.3f}  p={r['p']:.3g}")
OUT["sotorasib_refined"] = res

# Test multi-way interaction within KRAS+
m = smf.ols("pfs_months ~ treatment_sotorasib*sex_female + treatment_sotorasib*smk_never + treatment_sotorasib*alk_fusion + treatment_sotorasib*egfr_mutation + ecog_ps + stage_iv + albumin_g_dl",
            data=DF[m_pos]).fit()
inter_coefs = {}
for k in m.params.index:
    if "treatment_sotorasib:" in k:
        inter_coefs[k] = {"coef":float(m.params[k]),"p":float(m.pvalues[k])}
inter_coefs["main_treatment_sotorasib"] = {"coef":float(m.params["treatment_sotorasib"]),"p":float(m.pvalues["treatment_sotorasib"])}
print("\nWithin-KRAS+ interaction model:")
for k,v in inter_coefs.items():
    print(f"  {k:60s}  coef={v['coef']:+.3f}  p={v['p']:.3g}")
OUT["sotorasib_within_krasg12c_interactions"] = inter_coefs

# Pembro subgroup: refine by weight loss tertiles & stage_iv
print("\n=== Refining pembro subgroup ===")
DF["wl_tertile"] = pd.qcut(DF["weight_loss_pct_6mo"], 3, labels=["low","mid","high"], duplicates="drop")
res = []
for tier in ["low","mid","high"]:
    res.append(stratum(f"weight_loss tertile {tier}", DF["wl_tertile"]==tier, "treatment_pembrolizumab"))
res.append(stratum("stage_iv==1", DF["stage_iv"]==1, "treatment_pembrolizumab"))
res.append(stratum("stage_iv==0", DF["stage_iv"]==0, "treatment_pembrolizumab"))
res.append(stratum("weight_loss>=6 & stage_iv==1", (DF["weight_loss_pct_6mo"]>=6) & (DF["stage_iv"]==1), "treatment_pembrolizumab"))
res.append(stratum("weight_loss<6 & stage_iv==0", (DF["weight_loss_pct_6mo"]<6) & (DF["stage_iv"]==0), "treatment_pembrolizumab"))
for r in res:
    if r: print(f"  {r['name']:50s}  n={r['n']:5d}  diff={r['diff']:+.3f}  p={r['p']:.3g}")
OUT["pembro_refined"] = res

# Olaparib BRCA2+ refined with bun_mg_dl, crp_mg_l etc.
print("\n=== Refining olaparib subgroup ===")
res = []
m_brca = DF["brca2_mutation"]==1
res.append(stratum("BRCA2+ all", m_brca, "treatment_olaparib"))
res.append(stratum("BRCA2+ & bun>=median", m_brca & (DF["bun_mg_dl"]>=DF["bun_mg_dl"].median()), "treatment_olaparib"))
res.append(stratum("BRCA2+ & bun<median", m_brca & (DF["bun_mg_dl"]<DF["bun_mg_dl"].median()), "treatment_olaparib"))
res.append(stratum("BRCA2+ & ecog==0", m_brca & (DF["ecog_ps"]==0), "treatment_olaparib"))
for r in res:
    if r: print(f"  {r['name']:50s}  n={r['n']:5d}  diff={r['diff']:+.3f}  p={r['p']:.3g}")
OUT["olaparib_refined"] = res

# Look at osimertinib subgroups more carefully
print("\n=== Osimertinib explorations ===")
m_egfr = DF["egfr_mutation"]==1
res = []
res.append(stratum("EGFR+ & alk_fusion==0", m_egfr & (DF["alk_fusion"]==0), "treatment_osimertinib"))
res.append(stratum("EGFR+ & adeno==1 & smk_never==1", m_egfr & (DF["adeno"]==1) & (DF["smk_never"]==1), "treatment_osimertinib"))
res.append(stratum("EGFR+ & adeno==1 & ecog==0 & has_brain_mets==0",
                   m_egfr & (DF["adeno"]==1) & (DF["ecog_ps"]==0) & (DF["has_brain_mets"]==0), "treatment_osimertinib"))
for r in res:
    if r: print(f"  {r['name']:50s}  n={r['n']:5d}  diff={r['diff']:+.3f}  p={r['p']:.3g}")
OUT["osimertinib_refined"] = res

with open("results_refine.json","w") as f:
    json.dump(OUT,f,indent=2,default=str)
print("\nSaved results_refine.json")
