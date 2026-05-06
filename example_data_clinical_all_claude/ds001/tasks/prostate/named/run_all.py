"""Iterative analysis of ds001_prostate. Saves results to all_results.json."""
import json, math
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
N = len(df)
RESULTS = []  # list of dicts: {iter, hyp_id, hyp_text, kind, code, summary, p, eff, sig}

def add(it, hid, htext, code, summary, p, eff, sig=None, kind="novel"):
    if sig is None and p is not None and not (isinstance(p, float) and math.isnan(p)):
        sig = bool(p < 0.05)
    RESULTS.append({
        "iter": it, "hyp_id": hid, "hyp_text": htext, "kind": kind,
        "code": code, "summary": summary,
        "p": (None if p is None or (isinstance(p, float) and math.isnan(p)) else float(p)),
        "eff": (None if eff is None or (isinstance(eff, float) and math.isnan(eff)) else float(eff)),
        "sig": sig
    })

def logreg(formula, data=None):
    d = df if data is None else data
    return smf.logit(formula, data=d).fit(disp=0)

def logreg_summary(model, term):
    coef = model.params[term]
    p = model.pvalues[term]
    return coef, p

def two_prop(x, y):
    """Returns (rate_y1, rate_y0, diff, p_chi2)."""
    a = df.loc[x==1, y].mean()
    b = df.loc[x==0, y].mean()
    ct = pd.crosstab(x, df[y])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return a, b, a-b, p

# ============================================================
# Iteration 1: Treatment main effects on objective_response
# ============================================================
it = 1
for tcol in ['treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel',
             'treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab']:
    a = df.loc[df[tcol]==1, 'objective_response'].mean()
    b = df.loc[df[tcol]==0, 'objective_response'].mean()
    ct = pd.crosstab(df[tcol], df['objective_response'])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    hid = f"i1_{tcol}"
    htext = f"Patients receiving {tcol} have a higher objective_response rate than those not receiving it."
    summary = f"ORR {tcol}=1: {a:.3f} (n={int(df[tcol].sum())}) vs {tcol}=0: {b:.3f} (n={int((1-df[tcol]).sum())}); diff={a-b:+.3f}, chi2 p={p:.3g}."
    add(it, hid, htext, f"chi2_contingency on {tcol} x objective_response", summary, p, a-b)

# ============================================================
# Iteration 2: ECOG, mCRPC, visceral mets main effects on response
# ============================================================
it = 2
# ECOG (ordered)
m = logreg("objective_response ~ ecog_ps", df)
coef, p = logreg_summary(m, "ecog_ps")
rates = df.groupby("ecog_ps")["objective_response"].mean().to_dict()
add(it, "i2_ecog",
    "Higher ECOG performance status (worse) is associated with lower objective_response rate.",
    "logit(objective_response ~ ecog_ps)",
    f"Logistic ecog_ps coef={coef:.3f} (per unit), p={p:.3g}. ORR by ECOG: {rates}.",
    p, coef)
# mCRPC
a,b,d,p = two_prop(df['mcrpc'], 'objective_response')
add(it, "i2_mcrpc",
    "Patients with mCRPC have a different objective_response rate than non-mCRPC.",
    "chi2 on mcrpc x objective_response",
    f"ORR mCRPC=1: {a:.3f} vs mCRPC=0: {b:.3f}; diff={d:+.3f}, p={p:.3g}.", p, d)
# Visceral mets
a,b,d,p = two_prop(df['visceral_mets'], 'objective_response')
add(it, "i2_visc",
    "Patients with visceral metastases have a lower objective_response rate.",
    "chi2 on visceral_mets x objective_response",
    f"ORR visceral=1: {a:.3f} vs 0: {b:.3f}; diff={d:+.3f}, p={p:.3g}.", p, d)
# Gleason
m = logreg("objective_response ~ gleason_score", df)
coef, p = logreg_summary(m, "gleason_score")
add(it, "i2_gleason",
    "Higher Gleason score is associated with lower objective_response rate.",
    "logit(objective_response ~ gleason_score)",
    f"Gleason coef={coef:.3f} per point, p={p:.3g}. ORR by Gleason: {df.groupby('gleason_score')['objective_response'].mean().to_dict()}.",
    p, coef)

# ============================================================
# Iteration 3: Continuous lab/biomarker associations with response
# ============================================================
it = 3
for col, hyp_dir, hid_short in [
    ('psa_ng_ml','higher PSA -> lower ORR','psa'),
    ('albumin_g_dl','lower albumin -> lower ORR (so positive coef)','alb'),
    ('ldh_u_l','higher LDH -> lower ORR','ldh'),
    ('crp_mg_l','higher CRP -> lower ORR','crp'),
    ('nlr','higher NLR -> lower ORR','nlr'),
    ('hemoglobin_g_dl','higher Hb -> higher ORR','hb'),
    ('alkaline_phosphatase_u_l','higher ALP -> lower ORR','alp'),
    ('weight_loss_pct_6mo','more weight loss -> lower ORR','wt'),
    ('age_years','older age -> lower ORR','age'),
    ('calcium_mg_dl','higher Ca -> lower ORR','ca'),
    ('creatinine_mg_dl','higher Cr -> lower ORR','cr'),
    ('total_bilirubin_mg_dl','higher bilirubin -> lower ORR','bili'),
    ('ast_u_l','higher AST -> lower ORR','ast'),
    ('alt_u_l','higher ALT -> lower ORR','alt'),
    ('bun_mg_dl','higher BUN -> lower ORR','bun'),
    ('sodium_meq_l','higher Na -> higher ORR','na'),
    ('potassium_meq_l','higher K -> ?','k'),
]:
    m = logreg(f"objective_response ~ {col}", df)
    coef, p = logreg_summary(m, col)
    add(it, f"i3_{hid_short}",
        f"{col} is associated with objective_response ({hyp_dir}).",
        f"logit(objective_response ~ {col})",
        f"{col} coef={coef:.4g} per unit, p={p:.3g}.", p, coef)

# ============================================================
# Iteration 4: Key biomarker x treatment interactions (THE BIG ONES)
# ============================================================
it = 4
def interaction_test(treat, biom, hid, htext_extra=""):
    f = f"objective_response ~ {treat} * {biom}"
    m = logreg(f, df)
    inter = f"{treat}:{biom}"
    coef = m.params[inter]; p = m.pvalues[inter]
    # Also stratified rates
    g = df.groupby([biom, treat])['objective_response'].agg(['mean','count']).reset_index()
    rate_b1_t1 = df.loc[(df[biom]==1)&(df[treat]==1),'objective_response'].mean()
    rate_b1_t0 = df.loc[(df[biom]==1)&(df[treat]==0),'objective_response'].mean()
    rate_b0_t1 = df.loc[(df[biom]==0)&(df[treat]==1),'objective_response'].mean()
    rate_b0_t0 = df.loc[(df[biom]==0)&(df[treat]==0),'objective_response'].mean()
    diff_in_b1 = rate_b1_t1 - rate_b1_t0
    diff_in_b0 = rate_b0_t1 - rate_b0_t0
    summary = (f"In {biom}=1: ORR {treat}+={rate_b1_t1:.3f} vs -={rate_b1_t0:.3f} (diff={diff_in_b1:+.3f}). "
               f"In {biom}=0: ORR {treat}+={rate_b0_t1:.3f} vs -={rate_b0_t0:.3f} (diff={diff_in_b0:+.3f}). "
               f"Interaction coef={coef:.3f}, p={p:.3g}.")
    return coef, p, summary, diff_in_b1, diff_in_b0

# olaparib x brca2
c, p, s, d1, d0 = interaction_test('treatment_olaparib','brca2_mutation','i4_ola_brca')
add(it, "i4_ola_brca",
    "treatment_olaparib increases objective_response specifically in brca2_mutation=1 patients (positive interaction; little/no effect in BRCA2 wild-type).",
    "logit(objective_response ~ treatment_olaparib * brca2_mutation)", s, p, c)

# pembrolizumab x msi_high
c, p, s, d1, d0 = interaction_test('treatment_pembrolizumab','msi_high','i4_pem_msi')
add(it, "i4_pem_msi",
    "treatment_pembrolizumab increases objective_response specifically in msi_high=1 patients (positive interaction).",
    "logit(objective_response ~ treatment_pembrolizumab * msi_high)", s, p, c)

# lu177 x psma_high
c, p, s, d1, d0 = interaction_test('treatment_lu177_psma','psma_high','i4_lu_psma')
add(it, "i4_lu_psma",
    "treatment_lu177_psma increases objective_response specifically in psma_high=1 patients (positive interaction).",
    "logit(objective_response ~ treatment_lu177_psma * psma_high)", s, p, c)

# enzalutamide x ar_v7
c, p, s, d1, d0 = interaction_test('treatment_enzalutamide','ar_v7_positive','i4_enza_arv7')
add(it, "i4_enza_arv7",
    "treatment_enzalutamide effect on objective_response is reduced in ar_v7_positive=1 patients (negative interaction; AR-V7 mediated resistance).",
    "logit(objective_response ~ treatment_enzalutamide * ar_v7_positive)", s, p, c)

# abiraterone x ar_v7
c, p, s, d1, d0 = interaction_test('treatment_abiraterone','ar_v7_positive','i4_abi_arv7')
add(it, "i4_abi_arv7",
    "treatment_abiraterone effect on objective_response is reduced in ar_v7_positive=1 patients (negative interaction).",
    "logit(objective_response ~ treatment_abiraterone * ar_v7_positive)", s, p, c)

# docetaxel x ar_v7 (chemo effective regardless)
c, p, s, d1, d0 = interaction_test('treatment_docetaxel','ar_v7_positive','i4_doce_arv7')
add(it, "i4_doce_arv7",
    "treatment_docetaxel response does NOT depend on ar_v7_positive (no interaction expected; chemo bypasses AR-V7).",
    "logit(objective_response ~ treatment_docetaxel * ar_v7_positive)", s, p, c)

# Save partial
with open("all_results.json","w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"After iter 4, {len(RESULTS)} analyses recorded.")
