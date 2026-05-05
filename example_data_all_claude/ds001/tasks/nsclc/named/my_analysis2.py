"""
Refinement and verification analyses.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")
RES = {}


def add(key, **kw):
    RES[key] = kw
    print(f"\n[{key}]")
    for k, v in kw.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")


def linreg(formula, data=None):
    if data is None:
        data = DF
    return smf.ols(formula, data=data).fit()


def coef(m, name):
    return float(m.params.get(name, np.nan)), float(m.pvalues.get(name, np.nan))


# 1) Verify sotorasib × KRAS G12C interaction with full adjustment
print("=" * 60)
print("SOTORASIB × KRAS G12C — full adjusted interaction model")
print("=" * 60)
adj = ("C(ecog_ps) + stage_iv + has_brain_mets + albumin_g_dl + ldh_u_l + "
       "weight_loss_pct_6mo + age_years + sex_female + C(smoking_status) + "
       "C(histology) + alk_fusion + egfr_mutation + stk11_mutation + "
       "brca2_mutation + tmb_high")
m = linreg(f"pfs_months ~ treatment_sotorasib * kras_g12c + {adj}")
e, p = coef(m, "treatment_sotorasib:kras_g12c")
e_main, p_main = coef(m, "treatment_sotorasib")
add("soto_kras_adjusted", interaction_coef=e, interaction_p=p,
    soto_main_coef=e_main, soto_main_p=p_main)

# Within KRAS+ adjusted model
m = linreg(f"pfs_months ~ treatment_sotorasib + {adj}",
           data=DF[DF.kras_g12c == 1])
e, p = coef(m, "treatment_sotorasib")
add("soto_in_kras_adjusted", coef=e, p=p)

# Within KRAS− adjusted model
m = linreg(f"pfs_months ~ treatment_sotorasib + {adj}",
           data=DF[DF.kras_g12c == 0])
e, p = coef(m, "treatment_sotorasib")
add("soto_in_no_kras_adjusted", coef=e, p=p)

# 2) Investigate pembro × egfr inverse (worse outcomes in EGFR− patients)
print("\n" + "=" * 60)
print("PEMBRO × EGFR — adjusted")
print("=" * 60)
m = linreg(f"pfs_months ~ treatment_pembrolizumab * egfr_mutation + {adj}")
e, p = coef(m, "treatment_pembrolizumab:egfr_mutation")
e_main, p_main = coef(m, "treatment_pembrolizumab")
add("pembro_egfr_adjusted", interaction_coef=e, interaction_p=p,
    pembro_main_coef=e_main, pembro_main_p=p_main)

m = linreg(f"pfs_months ~ treatment_pembrolizumab + {adj}",
           data=DF[DF.egfr_mutation == 0])
e, p = coef(m, "treatment_pembrolizumab")
add("pembro_in_no_egfr_adjusted", coef=e, p=p)

m = linreg(f"pfs_months ~ treatment_pembrolizumab + {adj}",
           data=DF[DF.egfr_mutation == 1])
e, p = coef(m, "treatment_pembrolizumab")
add("pembro_in_egfr_adjusted", coef=e, p=p)

# 3) Look at pembro × ecog
m = linreg(f"pfs_months ~ treatment_pembrolizumab * C(ecog_ps) + albumin_g_dl + "
          f"weight_loss_pct_6mo + stage_iv + has_brain_mets")
print(m.summary().tables[1])
RES["pembro_ecog_interaction"] = {
    name: (float(m.params[name]), float(m.pvalues[name]))
    for name in m.params.index
}

# 4) Verify osi × egfr is truly null (try unadjusted and adjusted within strata)
print("\n" + "=" * 60)
print("OSIMERTINIB × EGFR — verify null")
print("=" * 60)
m = linreg(f"pfs_months ~ treatment_osimertinib * egfr_mutation + {adj}")
e, p = coef(m, "treatment_osimertinib:egfr_mutation")
e_main, p_main = coef(m, "treatment_osimertinib")
add("osi_egfr_adjusted", interaction_coef=e, interaction_p=p,
    osi_main_coef=e_main, osi_main_p=p_main)

# 5) Verify olaparib × brca2 is truly null
print("\n" + "=" * 60)
print("OLAPARIB × BRCA2 — verify null")
print("=" * 60)
m = linreg(f"pfs_months ~ treatment_olaparib * brca2_mutation + {adj}")
e, p = coef(m, "treatment_olaparib:brca2_mutation")
e_main, p_main = coef(m, "treatment_olaparib")
add("ola_brca2_adjusted", interaction_coef=e, interaction_p=p,
    ola_main_coef=e_main, ola_main_p=p_main)

# 6) Investigate why TMB high has WORSE main effect on PFS (counterintuitive)
print("\n" + "=" * 60)
print("TMB high prognostic — investigate")
print("=" * 60)
# adjusted main effect
m = linreg(f"pfs_months ~ tmb_high + {adj.replace('+ tmb_high','')}")
print("Note: tmb_high adjusted coef:", m.params.get("tmb_high"),
      "p:", m.pvalues.get("tmb_high"))
RES["tmb_main_adjusted"] = (float(m.params.get("tmb_high")),
                             float(m.pvalues.get("tmb_high")))

# 7) Check whether EGFR is associated with treatment_osimertinib (treatment imbalance check)
print("\n" + "=" * 60)
print("Treatment vs biomarker assoc")
print("=" * 60)
ct = pd.crosstab(DF.egfr_mutation, DF.treatment_osimertinib)
print("egfr × osi:\n", ct)
print(stats.chi2_contingency(ct))
RES["assoc_egfr_osi"] = float(stats.chi2_contingency(ct)[1])

ct = pd.crosstab(DF.kras_g12c, DF.treatment_sotorasib)
print("\nkras × soto:\n", ct)
print(stats.chi2_contingency(ct))
RES["assoc_kras_soto"] = float(stats.chi2_contingency(ct)[1])

ct = pd.crosstab(DF.brca2_mutation, DF.treatment_olaparib)
print("\nbrca2 × ola:\n", ct)
print(stats.chi2_contingency(ct))
RES["assoc_brca2_ola"] = float(stats.chi2_contingency(ct)[1])

ct = pd.crosstab(DF.pdl1_tps >= 0.5, DF.treatment_pembrolizumab)
print("\npdl1>=0.5 × pembro:\n", ct)
print(stats.chi2_contingency(ct))
RES["assoc_pdl1_pembro"] = float(stats.chi2_contingency(ct)[1])

# 8) Ultimate joint model with KEY interactions only and proper adjustment
print("\n" + "=" * 60)
print("FINAL JOINT MODEL")
print("=" * 60)
final_form = (
    "pfs_months ~ treatment_sotorasib * kras_g12c + "
    "treatment_pembrolizumab + treatment_olaparib + treatment_osimertinib + "
    "C(ecog_ps) + stage_iv + has_brain_mets + albumin_g_dl + ldh_u_l + "
    "weight_loss_pct_6mo + age_years + sex_female + C(smoking_status) + "
    "C(histology) + egfr_mutation + alk_fusion + stk11_mutation + "
    "brca2_mutation + tmb_high + pdl1_tps + crp_mg_l + nlr + "
    "hemoglobin_g_dl + alkaline_phosphatase_u_l + ast_u_l + alt_u_l + "
    "total_bilirubin_mg_dl + creatinine_mg_dl + bun_mg_dl + sodium_meq_l + "
    "potassium_meq_l + calcium_mg_dl"
)
m = linreg(final_form)
print(m.summary().tables[1])
RES["final_joint"] = {name: (float(m.params[name]), float(m.pvalues[name]))
                      for name in m.params.index}
print("R^2:", m.rsquared, "adj R^2:", m.rsquared_adj)

# 9) Check pembro × histology (for squamous)
m = linreg(f"pfs_months ~ treatment_pembrolizumab * C(histology) + {adj}")
e, p = coef(m, "treatment_pembrolizumab:C(histology)[T.squamous]")
e_main, p_main = coef(m, "treatment_pembrolizumab")
add("pembro_squamous_adjusted", interaction_coef=e, interaction_p=p,
    pembro_main_coef=e_main, pembro_main_p=p_main)

# 10) Check whether the pembro_in_pdl1_high effect is real after adjustment
m = linreg(f"pfs_months ~ treatment_pembrolizumab + {adj}",
           data=DF[DF.pdl1_tps >= 0.5])
e, p = coef(m, "treatment_pembrolizumab")
add("pembro_in_pdl1_high_adjusted", coef=e, p=p)

m = linreg(f"pfs_months ~ treatment_pembrolizumab + {adj}",
           data=DF[DF.pdl1_tps < 0.5])
e, p = coef(m, "treatment_pembrolizumab")
add("pembro_in_pdl1_low_adjusted", coef=e, p=p)

# 11) Final final: pembro in PDL1>=0.5 & STK11=0 (best subgroup) — adjusted
m_pembro_best = ((DF.pdl1_tps >= 0.5) & (DF.stk11_mutation == 0))
m = linreg(f"pfs_months ~ treatment_pembrolizumab + {adj}",
           data=DF[m_pembro_best])
e, p = coef(m, "treatment_pembrolizumab")
add("pembro_pdl1high_stk11neg_adjusted", coef=e, p=p)

# 12) Test if there's a "true" pembrolizumab benefit subgroup we missed —
# look at pembro × every binary feature × pdl1_tps
print("\n" + "=" * 60)
print("Pembro 3-way interactions search")
print("=" * 60)
binary_feats = ["sex_female", "stage_iv", "has_brain_mets", "egfr_mutation",
                "kras_g12c", "alk_fusion", "stk11_mutation", "brca2_mutation",
                "tmb_high"]
three_way = {}
for f in binary_feats:
    try:
        m = linreg(f"pfs_months ~ treatment_pembrolizumab * pdl1_tps * {f}")
        ix = f"treatment_pembrolizumab:pdl1_tps:{f}"
        e, p = coef(m, ix)
        three_way[f] = (e, p)
    except Exception:
        three_way[f] = (None, None)
for f, (e, p) in sorted(three_way.items(), key=lambda x: x[1][1] if x[1][1] is not None else 1):
    print(f"  pembro:pdl1_tps:{f}: coef={e:.4f}  p={p:.3g}")
RES["pembro_3way_pdl1"] = {f: list(v) for f, v in three_way.items()}

# Save
with open("my_raw_results2.json", "w") as fh:
    json.dump(RES, fh, indent=2, default=str)
print("\nDONE — wrote my_raw_results2.json")
