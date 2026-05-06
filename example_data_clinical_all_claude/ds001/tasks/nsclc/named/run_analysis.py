"""Run a comprehensive systematic analysis of ds001_nsclc and dump JSON results.

Each analysis produces a key in results dict with:
  - effect estimate (signed, on natural scale)
  - p-value
  - significance (p<0.05)
  - description
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
N = len(df)

# Encode categoricals
df["adeno"] = (df["histology"] == "adenocarcinoma").astype(int)
df["smoke_current"] = (df["smoking_status"] == "current").astype(int)
df["smoke_former"] = (df["smoking_status"] == "former").astype(int)
df["smoke_never"] = (df["smoking_status"] == "never").astype(int)

results = {}


def add(name, *, est, p, desc, sig=None):
    if sig is None and p is not None:
        sig = bool(p < 0.05)
    results[name] = {
        "effect_estimate": float(est) if est is not None else None,
        "p_value": float(p) if p is not None else None,
        "significant": sig,
        "result_summary": desc,
    }


def ttest_pfs(name, mask, label_high, label_low, desc_template):
    a = df.loc[mask, "pfs_months"]
    b = df.loc[~mask, "pfs_months"]
    if len(a) < 5 or len(b) < 5:
        return
    t = stats.ttest_ind(a, b, equal_var=False)
    diff = float(a.mean() - b.mean())
    desc = desc_template.format(
        n_a=len(a), m_a=a.mean(), n_b=len(b), m_b=b.mean(), diff=diff, p=t.pvalue
    )
    add(name, est=diff, p=float(t.pvalue), desc=desc)


def ols_effect(name, formula, term, desc_template, dataset=None, **fmt):
    d = df if dataset is None else dataset
    m = smf.ols(formula, data=d).fit()
    coef = float(m.params[term])
    pv = float(m.pvalues[term])
    se = float(m.bse[term])
    desc = desc_template.format(coef=coef, se=se, p=pv, n=len(d), **fmt)
    add(name, est=coef, p=pv, desc=desc)
    return m


# =========================================================================
# Iteration 1 — Outcome distribution + simple prognostic factors
# =========================================================================

# H1: Older patients have shorter PFS (continuous age vs. PFS)
m = smf.ols("pfs_months ~ age_years", data=df).fit()
add(
    "age_pfs",
    est=float(m.params["age_years"]),
    p=float(m.pvalues["age_years"]),
    desc=f"Linear regression PFS ~ age_years (n=50000): slope={m.params['age_years']:.4f} months/year (SE={m.bse['age_years']:.4f}), p={m.pvalues['age_years']:.3g}.",
)

# H2: Female patients have longer PFS than male patients
ttest_pfs(
    "female_pfs",
    df["sex_female"] == 1,
    "female",
    "male",
    "Female (n={n_a}) mean PFS={m_a:.3f} mo; male (n={n_b}) mean PFS={m_b:.3f} mo; diff={diff:.3f}, t-test p={p:.3g}.",
)

# H3: ECOG PS is inversely associated with PFS (higher ECOG = shorter PFS)
m = smf.ols("pfs_months ~ ecog_ps", data=df).fit()
add(
    "ecog_pfs",
    est=float(m.params["ecog_ps"]),
    p=float(m.pvalues["ecog_ps"]),
    desc=f"PFS ~ ecog_ps (continuous 0/1/2): slope={m.params['ecog_ps']:.4f} mo/level (SE={m.bse['ecog_ps']:.4f}), p={m.pvalues['ecog_ps']:.3g}; mean PFS by ECOG: 0={df.loc[df['ecog_ps']==0,'pfs_months'].mean():.2f}, 1={df.loc[df['ecog_ps']==1,'pfs_months'].mean():.2f}, 2={df.loc[df['ecog_ps']==2,'pfs_months'].mean():.2f}.",
)

# H4: Stage IV vs. earlier stage — shorter PFS
ttest_pfs(
    "stage_iv_pfs",
    df["stage_iv"] == 1,
    "stage IV",
    "non-stage IV",
    "Stage IV (n={n_a}) PFS={m_a:.3f} mo; non-stage-IV (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# H5: Brain mets — shorter PFS
ttest_pfs(
    "brain_mets_pfs",
    df["has_brain_mets"] == 1,
    "brain mets",
    "no brain mets",
    "Brain mets (n={n_a}) PFS={m_a:.3f} mo; none (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# =========================================================================
# Iteration 2 — Histology, smoking, mutations as prognostic factors
# =========================================================================

# H6: Adenocarcinoma vs. squamous PFS
ttest_pfs(
    "adeno_pfs",
    df["histology"] == "adenocarcinoma",
    "adeno",
    "squamous",
    "Adenocarcinoma (n={n_a}) PFS={m_a:.3f} mo; squamous (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# H7-H9: Smoking status (3-level) effect on PFS: ANOVA
groups = [df.loc[df["smoking_status"] == s, "pfs_months"] for s in ("never", "former", "current")]
F, p = stats.f_oneway(*groups)
add(
    "smoking_pfs_anova",
    est=float(groups[2].mean() - groups[0].mean()),
    p=float(p),
    desc=f"ANOVA PFS by smoking_status: never (n={len(groups[0])}, mean={groups[0].mean():.3f}), former (n={len(groups[1])}, mean={groups[1].mean():.3f}), current (n={len(groups[2])}, mean={groups[2].mean():.3f}); F={F:.3f}, p={p:.3g}. Effect estimate is current minus never.",
)

# H10: EGFR mutation — main effect on PFS (could go either way; in untreated cohorts EGFR+ often live longer due to TKI)
for mut in ["egfr_mutation", "kras_g12c", "alk_fusion", "stk11_mutation", "brca2_mutation", "tmb_high"]:
    ttest_pfs(
        f"{mut}_pfs",
        df[mut] == 1,
        mut,
        f"no {mut}",
        f"{mut}=1 (n={{n_a}}) PFS={{m_a:.3f}} mo; {mut}=0 (n={{n_b}}) PFS={{m_b:.3f}} mo; diff={{diff:.3f}}, p={{p:.3g}}.",
    )

# =========================================================================
# Iteration 3 — Continuous lab features prognostic effects (univariate OLS slopes)
# =========================================================================

cont_labs = [
    "pdl1_tps", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr",
    "hemoglobin_g_dl", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
    "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
    "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl",
]

for col in cont_labs:
    m = smf.ols(f"pfs_months ~ {col}", data=df).fit()
    coef = float(m.params[col]); pv = float(m.pvalues[col]); se = float(m.bse[col])
    add(
        f"{col}_pfs",
        est=coef, p=pv,
        desc=f"OLS PFS ~ {col} (univariate, n=50000): slope={coef:.5f} mo per unit (SE={se:.5f}), p={pv:.3g}.",
    )

# =========================================================================
# Iteration 4 — Main treatment effects (univariate)
# =========================================================================

for tx in ["treatment_pembrolizumab", "treatment_sotorasib", "treatment_olaparib", "treatment_osimertinib"]:
    ttest_pfs(
        f"{tx}_main",
        df[tx] == 1,
        tx, f"no {tx}",
        f"{tx}=1 (n={{n_a}}) PFS={{m_a:.3f}} mo vs {tx}=0 (n={{n_b}}) PFS={{m_b:.3f}} mo; diff={{diff:.3f}}, t-test p={{p:.3g}}.",
    )

# =========================================================================
# Iteration 5 — Treatment x EGFR (osimertinib's expected biomarker)
# =========================================================================

# Test in EGFR+ subgroup
sub = df[df["egfr_mutation"] == 1]
ttest_pfs(
    "osi_in_egfr_pos",
    (sub["treatment_osimertinib"] == 1).reindex(df.index, fill_value=False),
    "osi in EGFR+", "no-osi in EGFR+",
    "Among EGFR+ patients only: osimertinib (n={n_a}) PFS={m_a:.3f} mo vs no-osimertinib (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# Properly compute within EGFR+ subgroup
def subgroup_treatment_effect(name, sub, tx, desc_template):
    a = sub.loc[sub[tx] == 1, "pfs_months"]
    b = sub.loc[sub[tx] == 0, "pfs_months"]
    if len(a) < 5 or len(b) < 5:
        add(name, est=None, p=None, desc=f"Insufficient data for {name}", sig=None)
        return
    t = stats.ttest_ind(a, b, equal_var=False)
    diff = float(a.mean() - b.mean())
    desc = desc_template.format(
        n_a=len(a), m_a=a.mean(), n_b=len(b), m_b=b.mean(), diff=diff, p=t.pvalue
    )
    add(name, est=diff, p=float(t.pvalue), desc=desc)
    return diff, t.pvalue

# Re-do with correct subgroup logic
del results["osi_in_egfr_pos"]
subgroup_treatment_effect(
    "osi_in_egfr_pos",
    df[df["egfr_mutation"] == 1],
    "treatment_osimertinib",
    "EGFR+ subgroup: osimertinib (n={n_a}) PFS={m_a:.3f} mo vs no-osimertinib (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, t-test p={p:.3g}.",
)
subgroup_treatment_effect(
    "osi_in_egfr_neg",
    df[df["egfr_mutation"] == 0],
    "treatment_osimertinib",
    "EGFR- subgroup: osimertinib (n={n_a}) PFS={m_a:.3f} mo vs no-osimertinib (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# Interaction term in OLS
m = smf.ols("pfs_months ~ treatment_osimertinib * egfr_mutation", data=df).fit()
add(
    "osi_x_egfr_interaction",
    est=float(m.params["treatment_osimertinib:egfr_mutation"]),
    p=float(m.pvalues["treatment_osimertinib:egfr_mutation"]),
    desc=f"OLS PFS ~ treatment_osimertinib * egfr_mutation: interaction coef={m.params['treatment_osimertinib:egfr_mutation']:.4f} (SE={m.bse['treatment_osimertinib:egfr_mutation']:.4f}), p={m.pvalues['treatment_osimertinib:egfr_mutation']:.3g}; main osi coef={m.params['treatment_osimertinib']:.4f}, main EGFR coef={m.params['egfr_mutation']:.4f}.",
)

# =========================================================================
# Iteration 6 — Treatment x KRAS G12C (sotorasib)
# =========================================================================

subgroup_treatment_effect(
    "soto_in_kras_pos",
    df[df["kras_g12c"] == 1],
    "treatment_sotorasib",
    "KRAS G12C+ subgroup: sotorasib (n={n_a}) PFS={m_a:.3f} mo vs no-sotorasib (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
subgroup_treatment_effect(
    "soto_in_kras_neg",
    df[df["kras_g12c"] == 0],
    "treatment_sotorasib",
    "KRAS G12C- subgroup: sotorasib (n={n_a}) PFS={m_a:.3f} mo vs no-sotorasib (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
m = smf.ols("pfs_months ~ treatment_sotorasib * kras_g12c", data=df).fit()
add(
    "soto_x_kras_interaction",
    est=float(m.params["treatment_sotorasib:kras_g12c"]),
    p=float(m.pvalues["treatment_sotorasib:kras_g12c"]),
    desc=f"OLS PFS ~ treatment_sotorasib * kras_g12c: interaction coef={m.params['treatment_sotorasib:kras_g12c']:.4f} (SE={m.bse['treatment_sotorasib:kras_g12c']:.4f}), p={m.pvalues['treatment_sotorasib:kras_g12c']:.3g}; main soto coef={m.params['treatment_sotorasib']:.4f}, main KRAS coef={m.params['kras_g12c']:.4f}.",
)

# =========================================================================
# Iteration 7 — Treatment x PDL1/TMB (pembrolizumab)
# =========================================================================

# TMB-high
subgroup_treatment_effect(
    "pembro_in_tmb_high",
    df[df["tmb_high"] == 1],
    "treatment_pembrolizumab",
    "TMB-high subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
subgroup_treatment_effect(
    "pembro_in_tmb_low",
    df[df["tmb_high"] == 0],
    "treatment_pembrolizumab",
    "TMB-low subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
m = smf.ols("pfs_months ~ treatment_pembrolizumab * tmb_high", data=df).fit()
add(
    "pembro_x_tmb_interaction",
    est=float(m.params["treatment_pembrolizumab:tmb_high"]),
    p=float(m.pvalues["treatment_pembrolizumab:tmb_high"]),
    desc=f"OLS PFS ~ treatment_pembrolizumab * tmb_high: interaction coef={m.params['treatment_pembrolizumab:tmb_high']:.4f} (SE={m.bse['treatment_pembrolizumab:tmb_high']:.4f}), p={m.pvalues['treatment_pembrolizumab:tmb_high']:.3g}.",
)

# PDL1 binary >= 0.5
df["pdl1_high"] = (df["pdl1_tps"] >= 0.5).astype(int)
subgroup_treatment_effect(
    "pembro_in_pdl1_high",
    df[df["pdl1_high"] == 1],
    "treatment_pembrolizumab",
    "PDL1>=0.5 subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
subgroup_treatment_effect(
    "pembro_in_pdl1_low",
    df[df["pdl1_high"] == 0],
    "treatment_pembrolizumab",
    "PDL1<0.5 subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
m = smf.ols("pfs_months ~ treatment_pembrolizumab * pdl1_high", data=df).fit()
add(
    "pembro_x_pdl1_interaction",
    est=float(m.params["treatment_pembrolizumab:pdl1_high"]),
    p=float(m.pvalues["treatment_pembrolizumab:pdl1_high"]),
    desc=f"OLS PFS ~ treatment_pembrolizumab * pdl1_high (>=0.5): interaction coef={m.params['treatment_pembrolizumab:pdl1_high']:.4f} (SE={m.bse['treatment_pembrolizumab:pdl1_high']:.4f}), p={m.pvalues['treatment_pembrolizumab:pdl1_high']:.3g}.",
)

# Continuous PDL1 interaction
m = smf.ols("pfs_months ~ treatment_pembrolizumab * pdl1_tps", data=df).fit()
add(
    "pembro_x_pdl1cont_interaction",
    est=float(m.params["treatment_pembrolizumab:pdl1_tps"]),
    p=float(m.pvalues["treatment_pembrolizumab:pdl1_tps"]),
    desc=f"OLS PFS ~ pembro * pdl1_tps (continuous): interaction coef={m.params['treatment_pembrolizumab:pdl1_tps']:.4f} (SE={m.bse['treatment_pembrolizumab:pdl1_tps']:.4f}), p={m.pvalues['treatment_pembrolizumab:pdl1_tps']:.3g}.",
)

# =========================================================================
# Iteration 8 — Treatment x BRCA2 (olaparib)
# =========================================================================

subgroup_treatment_effect(
    "olap_in_brca2_pos",
    df[df["brca2_mutation"] == 1],
    "treatment_olaparib",
    "BRCA2+ subgroup: olaparib (n={n_a}) PFS={m_a:.3f} mo vs no-olaparib (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
subgroup_treatment_effect(
    "olap_in_brca2_neg",
    df[df["brca2_mutation"] == 0],
    "treatment_olaparib",
    "BRCA2- subgroup: olaparib (n={n_a}) PFS={m_a:.3f} mo vs no-olaparib (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
m = smf.ols("pfs_months ~ treatment_olaparib * brca2_mutation", data=df).fit()
add(
    "olap_x_brca2_interaction",
    est=float(m.params["treatment_olaparib:brca2_mutation"]),
    p=float(m.pvalues["treatment_olaparib:brca2_mutation"]),
    desc=f"OLS PFS ~ olaparib * brca2: interaction coef={m.params['treatment_olaparib:brca2_mutation']:.4f} (SE={m.bse['treatment_olaparib:brca2_mutation']:.4f}), p={m.pvalues['treatment_olaparib:brca2_mutation']:.3g}.",
)

# =========================================================================
# Iteration 9 — STK11 mutation as a negative modifier (real-world: STK11 mutation suppresses pembro response)
# =========================================================================

subgroup_treatment_effect(
    "pembro_in_stk11_mut",
    df[df["stk11_mutation"] == 1],
    "treatment_pembrolizumab",
    "STK11-mutant subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
subgroup_treatment_effect(
    "pembro_in_stk11_wt",
    df[df["stk11_mutation"] == 0],
    "treatment_pembrolizumab",
    "STK11-WT subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
m = smf.ols("pfs_months ~ treatment_pembrolizumab * stk11_mutation", data=df).fit()
add(
    "pembro_x_stk11_interaction",
    est=float(m.params["treatment_pembrolizumab:stk11_mutation"]),
    p=float(m.pvalues["treatment_pembrolizumab:stk11_mutation"]),
    desc=f"OLS PFS ~ pembro * stk11: interaction coef={m.params['treatment_pembrolizumab:stk11_mutation']:.4f} (SE={m.bse['treatment_pembrolizumab:stk11_mutation']:.4f}), p={m.pvalues['treatment_pembrolizumab:stk11_mutation']:.3g}.",
)

# =========================================================================
# Iteration 10 — Adjusted treatment effects in matched biomarker subgroups
# =========================================================================

# Adjust for prognostic factors
adj_formula = (
    "pfs_months ~ treatment_pembrolizumab + treatment_sotorasib + treatment_olaparib + treatment_osimertinib + "
    "age_years + sex_female + ecog_ps + stage_iv + has_brain_mets + adeno + smoke_current + smoke_former + "
    "egfr_mutation + kras_g12c + alk_fusion + stk11_mutation + brca2_mutation + tmb_high + pdl1_tps + "
    "albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr + hemoglobin_g_dl"
)
m = smf.ols(adj_formula, data=df).fit()
for tx in ["treatment_pembrolizumab", "treatment_sotorasib", "treatment_olaparib", "treatment_osimertinib"]:
    add(
        f"{tx}_adj_main",
        est=float(m.params[tx]),
        p=float(m.pvalues[tx]),
        desc=f"Adjusted OLS coef for {tx} (controlling for age, sex, ECOG, stage, brain mets, histology, smoking, all biomarkers, and key labs): coef={m.params[tx]:.4f} mo (SE={m.bse[tx]:.4f}), p={m.pvalues[tx]:.3g}.",
    )

# Adjusted treatment-x-biomarker interactions (control for prognostic factors)
covars = "age_years + sex_female + ecog_ps + stage_iv + has_brain_mets + adeno + smoke_current + smoke_former + albumin_g_dl + ldh_u_l + nlr"

# Adjusted osi x egfr
m = smf.ols(f"pfs_months ~ treatment_osimertinib * egfr_mutation + {covars}", data=df).fit()
add(
    "osi_x_egfr_adj",
    est=float(m.params["treatment_osimertinib:egfr_mutation"]),
    p=float(m.pvalues["treatment_osimertinib:egfr_mutation"]),
    desc=f"Adjusted OLS interaction coef for treatment_osimertinib × egfr_mutation = {m.params['treatment_osimertinib:egfr_mutation']:.4f} (SE={m.bse['treatment_osimertinib:egfr_mutation']:.4f}), p={m.pvalues['treatment_osimertinib:egfr_mutation']:.3g}.",
)

# Adjusted soto x kras
m = smf.ols(f"pfs_months ~ treatment_sotorasib * kras_g12c + {covars}", data=df).fit()
add(
    "soto_x_kras_adj",
    est=float(m.params["treatment_sotorasib:kras_g12c"]),
    p=float(m.pvalues["treatment_sotorasib:kras_g12c"]),
    desc=f"Adjusted OLS interaction coef for treatment_sotorasib × kras_g12c = {m.params['treatment_sotorasib:kras_g12c']:.4f} (SE={m.bse['treatment_sotorasib:kras_g12c']:.4f}), p={m.pvalues['treatment_sotorasib:kras_g12c']:.3g}.",
)

# Adjusted pembro x tmb
m = smf.ols(f"pfs_months ~ treatment_pembrolizumab * tmb_high + {covars}", data=df).fit()
add(
    "pembro_x_tmb_adj",
    est=float(m.params["treatment_pembrolizumab:tmb_high"]),
    p=float(m.pvalues["treatment_pembrolizumab:tmb_high"]),
    desc=f"Adjusted OLS interaction coef for treatment_pembrolizumab × tmb_high = {m.params['treatment_pembrolizumab:tmb_high']:.4f} (SE={m.bse['treatment_pembrolizumab:tmb_high']:.4f}), p={m.pvalues['treatment_pembrolizumab:tmb_high']:.3g}.",
)

# Adjusted olap x brca2
m = smf.ols(f"pfs_months ~ treatment_olaparib * brca2_mutation + {covars}", data=df).fit()
add(
    "olap_x_brca2_adj",
    est=float(m.params["treatment_olaparib:brca2_mutation"]),
    p=float(m.pvalues["treatment_olaparib:brca2_mutation"]),
    desc=f"Adjusted OLS interaction coef for treatment_olaparib × brca2_mutation = {m.params['treatment_olaparib:brca2_mutation']:.4f} (SE={m.bse['treatment_olaparib:brca2_mutation']:.4f}), p={m.pvalues['treatment_olaparib:brca2_mutation']:.3g}.",
)

# =========================================================================
# Iteration 11 — Systematic treatment x feature interaction screen for each treatment
# =========================================================================

binary_features = [
    "sex_female", "stage_iv", "has_brain_mets", "egfr_mutation", "kras_g12c",
    "alk_fusion", "stk11_mutation", "brca2_mutation", "tmb_high", "adeno", "smoke_current"
]
cont_features = ["age_years", "ecog_ps", "pdl1_tps", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr", "hemoglobin_g_dl"]

screens = {}
for tx in ["treatment_pembrolizumab", "treatment_sotorasib", "treatment_olaparib", "treatment_osimertinib"]:
    rows = []
    for f in binary_features + cont_features:
        try:
            mm = smf.ols(f"pfs_months ~ {tx} * {f}", data=df).fit()
            interaction_term = f"{tx}:{f}"
            if interaction_term in mm.params:
                rows.append({
                    "feature": f,
                    "coef": float(mm.params[interaction_term]),
                    "se": float(mm.bse[interaction_term]),
                    "p": float(mm.pvalues[interaction_term]),
                })
        except Exception as e:
            rows.append({"feature": f, "error": str(e)})
    rows.sort(key=lambda r: r.get("p", 1.0))
    screens[tx] = rows
    # Save top-3 hits as analyses
    for i, r in enumerate(rows[:5]):
        if "p" not in r:
            continue
        add(
            f"{tx}_screen_{r['feature']}",
            est=r["coef"], p=r["p"],
            desc=f"Treatment-x-feature interaction screen: {tx} × {r['feature']} interaction coef={r['coef']:.4f} (SE={r['se']:.4f}), p={r['p']:.3g} in unadjusted OLS for PFS.",
        )

# =========================================================================
# Iteration 12 — Multi-feature subgroup search per treatment
# Look for biomarker-positive AND not-suppressor subgroups
# =========================================================================

# For each treatment, given its known driver, look for additional features that further separate responders
# We test treatment effect within the biomarker-positive subset and search for which secondary feature increases the effect

def best_dual_subgroup(tx, primary_mask_col, primary_mask_val, secondary_features, label, name):
    """Within rows where primary mask holds, find a binary secondary feature whose presence (or absence)
    maximally enhances treatment effect."""
    rows_out = []
    primary_sub = df[df[primary_mask_col] == primary_mask_val]
    for sec in secondary_features:
        for sec_val in [0, 1]:
            mask = (primary_sub[sec] == sec_val)
            if mask.sum() < 50:
                continue
            sub = primary_sub[mask]
            a = sub.loc[sub[tx] == 1, "pfs_months"]
            b = sub.loc[sub[tx] == 0, "pfs_months"]
            if len(a) < 25 or len(b) < 25:
                continue
            t = stats.ttest_ind(a, b, equal_var=False)
            rows_out.append({
                "secondary": sec, "value": sec_val,
                "n_tx": int(len(a)), "n_ctrl": int(len(b)),
                "diff": float(a.mean() - b.mean()),
                "p": float(t.pvalue),
            })
    rows_out.sort(key=lambda r: -r["diff"])  # largest positive treatment benefit
    return rows_out

# Osimertinib in EGFR+: search for additional modifier
secondary_bins = [
    "stage_iv", "has_brain_mets", "stk11_mutation", "tmb_high", "alk_fusion",
    "kras_g12c", "brca2_mutation", "sex_female", "adeno"
]
osi_search = best_dual_subgroup("treatment_osimertinib", "egfr_mutation", 1, secondary_bins, "EGFR+", "osi_dual")
soto_search = best_dual_subgroup("treatment_sotorasib", "kras_g12c", 1, [s for s in secondary_bins if s != "kras_g12c"], "KRAS+", "soto_dual")
pembro_search = best_dual_subgroup("treatment_pembrolizumab", "tmb_high", 1, [s for s in secondary_bins if s != "tmb_high"], "TMB-high", "pembro_dual")
olap_search = best_dual_subgroup("treatment_olaparib", "brca2_mutation", 1, [s for s in secondary_bins if s != "brca2_mutation"], "BRCA2+", "olap_dual")

# Save top 3 dual subgroups for each
for tx_label, search in [("osi", osi_search), ("soto", soto_search), ("pembro", pembro_search), ("olap", olap_search)]:
    for i, r in enumerate(search[:5]):
        add(
            f"{tx_label}_dual_{r['secondary']}_{r['value']}",
            est=r["diff"], p=r["p"],
            desc=f"Dual subgroup: in primary biomarker-positive {tx_label} cohort with {r['secondary']}={r['value']} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}), treatment difference in PFS = {r['diff']:.3f} mo, p={r['p']:.3g}.",
        )

# =========================================================================
# Iteration 13 — STK11 as suppressor of pembro within TMB-high
# =========================================================================

# Pembro in TMB-high AND STK11-WT (the "clean" responders)
sub = df[(df["tmb_high"] == 1) & (df["stk11_mutation"] == 0)]
subgroup_treatment_effect(
    "pembro_in_tmbhigh_stk11wt",
    sub, "treatment_pembrolizumab",
    "TMB-high AND STK11-WT subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
sub = df[(df["tmb_high"] == 1) & (df["stk11_mutation"] == 1)]
subgroup_treatment_effect(
    "pembro_in_tmbhigh_stk11mut",
    sub, "treatment_pembrolizumab",
    "TMB-high AND STK11-MUT subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# 3-way interaction
m = smf.ols("pfs_months ~ treatment_pembrolizumab * tmb_high * stk11_mutation", data=df).fit()
term = "treatment_pembrolizumab:tmb_high:stk11_mutation"
add(
    "pembro_x_tmb_x_stk11",
    est=float(m.params[term]),
    p=float(m.pvalues[term]),
    desc=f"OLS PFS ~ pembro * tmb_high * stk11: 3-way interaction coef={m.params[term]:.4f} (SE={m.bse[term]:.4f}), p={m.pvalues[term]:.3g}; pembro×tmb coef={m.params['treatment_pembrolizumab:tmb_high']:.4f}, pembro×stk11 coef={m.params['treatment_pembrolizumab:stk11_mutation']:.4f}.",
)

# =========================================================================
# Iteration 14 — PDL1-high AND TMB-high pembro effect; ECOG modifiers
# =========================================================================

sub = df[(df["pdl1_high"] == 1) & (df["tmb_high"] == 1)]
subgroup_treatment_effect(
    "pembro_pdl1hi_tmbhi",
    sub, "treatment_pembrolizumab",
    "PDL1>=0.5 AND TMB-high subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

sub = df[(df["pdl1_high"] == 1) & (df["tmb_high"] == 1) & (df["stk11_mutation"] == 0)]
subgroup_treatment_effect(
    "pembro_pdl1hi_tmbhi_stk11wt",
    sub, "treatment_pembrolizumab",
    "PDL1-high AND TMB-high AND STK11-WT subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# Pembro x ECOG (poor PS may suppress benefit)
sub = df[df["ecog_ps"] <= 1]
subgroup_treatment_effect(
    "pembro_in_ecog_low",
    sub, "treatment_pembrolizumab",
    "ECOG 0-1 subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
sub = df[df["ecog_ps"] >= 2]
subgroup_treatment_effect(
    "pembro_in_ecog_high",
    sub, "treatment_pembrolizumab",
    "ECOG 2 subgroup: pembro (n={n_a}) PFS={m_a:.3f} mo vs no-pembro (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# =========================================================================
# Iteration 15 — Osimertinib refined: EGFR+ x brain mets, EGFR+ x stage IV
# =========================================================================

sub = df[(df["egfr_mutation"] == 1) & (df["has_brain_mets"] == 0)]
subgroup_treatment_effect(
    "osi_egfr_nobrain",
    sub, "treatment_osimertinib",
    "EGFR+ AND no brain mets subgroup: osi (n={n_a}) PFS={m_a:.3f} mo vs no-osi (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
sub = df[(df["egfr_mutation"] == 1) & (df["has_brain_mets"] == 1)]
subgroup_treatment_effect(
    "osi_egfr_brain",
    sub, "treatment_osimertinib",
    "EGFR+ AND brain mets subgroup: osi (n={n_a}) PFS={m_a:.3f} mo vs no-osi (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# =========================================================================
# Iteration 16 — Sotorasib refined: KRAS+ x stk11 (often co-mutated and modifies benefit)
# =========================================================================

sub = df[(df["kras_g12c"] == 1) & (df["stk11_mutation"] == 0)]
subgroup_treatment_effect(
    "soto_kras_stk11wt",
    sub, "treatment_sotorasib",
    "KRAS G12C+ AND STK11-WT subgroup: soto (n={n_a}) PFS={m_a:.3f} mo vs no-soto (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
sub = df[(df["kras_g12c"] == 1) & (df["stk11_mutation"] == 1)]
subgroup_treatment_effect(
    "soto_kras_stk11mut",
    sub, "treatment_sotorasib",
    "KRAS G12C+ AND STK11-MUT subgroup: soto (n={n_a}) PFS={m_a:.3f} mo vs no-soto (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# =========================================================================
# Iteration 17 — Treatment-treatment interactions (do combinations matter?)
# =========================================================================

# Pembro x sotorasib (in KRAS+)
sub = df[df["kras_g12c"] == 1]
m = smf.ols("pfs_months ~ treatment_sotorasib * treatment_pembrolizumab", data=sub).fit()
term = "treatment_sotorasib:treatment_pembrolizumab"
add(
    "soto_x_pembro_in_kras",
    est=float(m.params[term]),
    p=float(m.pvalues[term]),
    desc=f"In KRAS G12C+ patients (n={len(sub)}), OLS PFS ~ sotorasib * pembro: interaction coef={m.params[term]:.4f} (SE={m.bse[term]:.4f}), p={m.pvalues[term]:.3g}.",
)

# Osimertinib x pembro (in EGFR+)
sub = df[df["egfr_mutation"] == 1]
m = smf.ols("pfs_months ~ treatment_osimertinib * treatment_pembrolizumab", data=sub).fit()
term = "treatment_osimertinib:treatment_pembrolizumab"
add(
    "osi_x_pembro_in_egfr",
    est=float(m.params[term]),
    p=float(m.pvalues[term]),
    desc=f"In EGFR+ patients (n={len(sub)}), OLS PFS ~ osi * pembro: interaction coef={m.params[term]:.4f} (SE={m.bse[term]:.4f}), p={m.pvalues[term]:.3g}.",
)

# =========================================================================
# Save full results, plus screens
# =========================================================================

with open("analysis_results.json", "w") as f:
    json.dump({"results": results, "screens": screens}, f, indent=2)

print(f"Wrote {len(results)} analyses.")
print()
print("Sample results:")
for k in list(results.keys())[:5]:
    print(k, results[k]["effect_estimate"], results[k]["p_value"], results[k]["significant"])
