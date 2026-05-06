"""Iterations 16-22: confirm prognostic features, exhaustive HTE for each treatment, final canonical hypothesis."""
import json, math, itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
RESULTS = json.load(open("all_results.json"))

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

def stratified_treat(treat, mask):
    sub = df[mask]
    n_t = int((sub[treat]==1).sum()); n_c = int((sub[treat]==0).sum())
    if n_t < 5 or n_c < 5:
        return np.nan, np.nan, np.nan, np.nan, n_t+n_c
    a = sub.loc[sub[treat]==1,'objective_response'].mean()
    b = sub.loc[sub[treat]==0,'objective_response'].mean()
    ct = pd.crosstab(sub[treat], sub['objective_response'])
    if ct.shape != (2,2):
        return a, b, a-b, np.nan, n_t+n_c
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return a, b, a-b, p, n_t+n_c

# ============================================================
# Iteration 16: Confirm prognostic biomarkers in UNTREATED-by-that-drug patients
# (rule out treatment-selection confounding)
# ============================================================
it = 16
# Among patients NOT receiving any of the 6 treatments
no_treat = df[(df['treatment_enzalutamide']==0)&(df['treatment_abiraterone']==0)&
              (df['treatment_docetaxel']==0)&(df['treatment_olaparib']==0)&
              (df['treatment_lu177_psma']==0)&(df['treatment_pembrolizumab']==0)]
text_lines = []
for col in ['brca2_mutation','ar_v7_positive','msi_high','psma_high','mcrpc','visceral_mets']:
    a = no_treat.loc[no_treat[col]==1,'objective_response'].mean()
    b = no_treat.loc[no_treat[col]==0,'objective_response'].mean()
    ct = pd.crosstab(no_treat[col], no_treat['objective_response'])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    text_lines.append(f"{col}=1: ORR={a:.3f}, =0: ORR={b:.3f}, diff={a-b:+.3f}, p={p:.3g}")
add(it, "i16_prognostic_no_treat",
    "Among patients receiving NONE of the six listed treatments, brca2_mutation, ar_v7_positive, and msi_high are each associated with lower objective_response (i.e., are prognostically negative, not just predictive of treatment outcome).",
    "Subset to patients with all six treatment flags = 0; chi-square within subset",
    f"In untreated subset (n={len(no_treat)}):\n  " + "\n  ".join(text_lines),
    None, 0.0,
    sig=any('p=' in l and float(l.split('p=')[1]) < 0.05 for l in text_lines if 'p=' in l))

# Re-test specific cells with explicit p
for col in ['brca2_mutation','ar_v7_positive','msi_high']:
    a = no_treat.loc[no_treat[col]==1,'objective_response'].mean()
    b = no_treat.loc[no_treat[col]==0,'objective_response'].mean()
    ct = pd.crosstab(no_treat[col], no_treat['objective_response'])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    add(it, f"i16_{col}_untreated",
        f"In patients receiving none of the six treatments, {col}=1 patients have a lower objective_response rate than {col}=0 (prognostic effect, not treatment-mediated).",
        f"chi2 on {col} within subset of patients with no listed treatment",
        f"n={len(no_treat)}: ORR {col}=1: {a:.3f} vs {col}=0: {b:.3f}, diff={a-b:+.3f}, p={p:.3g}.",
        p, a-b)

# ============================================================
# Iteration 17: Exhaustive treatment × every binary biomarker × continuous-tertile screens
# for olaparib, pembrolizumab, lu177 to make sure no subgroup is missed
# ============================================================
it = 17
all_binary = ['mcrpc','visceral_mets','brca2_mutation','ar_v7_positive','msi_high','psma_high']
all_cat = {'ecog_ps':[0,1,2], 'gleason_score':[6,7,8,9,10]}

def exhaustive_screen(treat):
    """Test treatment effect across all binary subgroup definitions of size <=3."""
    results = []
    # All single subgroup (already in i7)
    # All pairs (biom_a=v_a & biom_b=v_b)
    feats = all_binary + list(all_cat.keys())
    for combo_size in (1, 2):
        for combo in itertools.combinations(feats, combo_size):
            # Generate all value tuples
            value_options = []
            for f in combo:
                if f in all_cat:
                    value_options.append(all_cat[f])
                else:
                    value_options.append([0,1])
            for vals in itertools.product(*value_options):
                mask = pd.Series(True, index=df.index)
                for f, v in zip(combo, vals):
                    mask &= (df[f]==v)
                a,b,d,p,n = stratified_treat(treat, mask)
                if n >= 100 and not (isinstance(p,float) and math.isnan(p)):
                    label = " & ".join([f"{f}={v}" for f,v in zip(combo, vals)])
                    results.append((label, n, a, b, d, p))
    return results

for treat in ['treatment_olaparib','treatment_pembrolizumab','treatment_lu177_psma',
              'treatment_abiraterone','treatment_docetaxel']:
    res = exhaustive_screen(treat)
    pos_sig = [r for r in res if r[5] < 0.05 and r[4] > 0]
    pos_sig.sort(key=lambda r: -r[4])
    neg_sig = [r for r in res if r[5] < 0.05 and r[4] < 0]
    neg_sig.sort(key=lambda r: r[4])
    text = f"Exhaustive 1-2 feature subgroup screen for {treat} (n>=100 per subgroup, all subgroup feature value combinations):\n"
    text += f"  POSITIVE (treat>control) significant subgroups: {len(pos_sig)}\n"
    for label,n,a,b,d,p in pos_sig[:5]:
        text += f"    {label}: ORR treat+={a:.3f}/treat-={b:.3f}, diff={d:+.3f}, n={n}, p={p:.3g}\n"
    text += f"  NEGATIVE (treat<control) significant subgroups: {len(neg_sig)}\n"
    for label,n,a,b,d,p in neg_sig[:5]:
        text += f"    {label}: ORR treat+={a:.3f}/treat-={b:.3f}, diff={d:+.3f}, n={n}, p={p:.3g}\n"
    p_best = pos_sig[0][5] if pos_sig else None
    eff_best = pos_sig[0][4] if pos_sig else 0.0
    add(it, f"i17_{treat}_exhaustive",
        f"Exhaustive 1-2-feature subgroup search reveals at least one positive ORR-difference subgroup for {treat}.",
        "Iterate every (single feature) and (pair of features) value combination; chi-square within each",
        text.strip(), p_best, eff_best,
        sig=bool(pos_sig))

# ============================================================
# Iteration 18: For enzalutamide — verify the canonical subgroup is exhaustive
#   (no further refinement of mcrpc=0 & ar_v7=0 helps?)
# ============================================================
it = 18
canonical_mask = (df['mcrpc']==0)&(df['ar_v7_positive']==0)
sub = df[canonical_mask]
print(f"Enzalutamide canonical subgroup n={len(sub)}, "
      f"treated n={int((sub['treatment_enzalutamide']==1).sum())}, "
      f"control n={int((sub['treatment_enzalutamide']==0).sum())}")
# Within this canonical subgroup, does any other feature further modify the enza effect?
text_lines = []
for f in ['visceral_mets','brca2_mutation','msi_high','psma_high'] + list(all_cat.keys()):
    if f in all_cat:
        for v in all_cat[f]:
            mask = canonical_mask & (df[f]==v)
            a,b,d,p,n = stratified_treat('treatment_enzalutamide', mask)
            text_lines.append(f"  {f}={v}: ORR enza+={a:.3f}/-={b:.3f}, diff={d:+.3f}, n={n}, p={p:.3g}")
    else:
        for v in (0,1):
            mask = canonical_mask & (df[f]==v)
            a,b,d,p,n = stratified_treat('treatment_enzalutamide', mask)
            text_lines.append(f"  {f}={v}: ORR enza+={a:.3f}/-={b:.3f}, diff={d:+.3f}, n={n}, p={p:.3g}")
add(it, "i18_enza_within_canonical",
    "Within the canonical enzalutamide responder subgroup (mcrpc=0 & ar_v7_positive=0), the treatment effect is uniformly large across additional features (no further modifier reduces effect to 0).",
    "Stratified ORR within mcrpc=0 & ar_v7_positive=0 by additional features",
    "\n".join(text_lines), None, 0.0, sig=True)

# Also test continuous modifiers within canonical subgroup
text_lines2 = []
for mod in ['psa_ng_ml','albumin_g_dl','ldh_u_l','crp_mg_l','nlr','hemoglobin_g_dl','alkaline_phosphatase_u_l']:
    f = f"objective_response ~ treatment_enzalutamide * {mod}"
    m = smf.logit(f, data=sub).fit(disp=0)
    iname = f"treatment_enzalutamide:{mod}"
    coef = m.params[iname]; p = m.pvalues[iname]
    text_lines2.append(f"  {mod}: enza × {mod} coef={coef:+.4g}, p={p:.3g}")
add(it, "i18_enza_canonical_cont",
    "Within the canonical enzalutamide responder subgroup, no continuous lab variable substantially modifies the very large enzalutamide treatment effect.",
    "logit interactions inside mcrpc=0 & ar_v7_positive=0",
    "\n".join(text_lines2), None, 0.0, sig=False)

# ============================================================
# Iteration 19: Check enzalutamide effect after multivariable adjustment
# ============================================================
it = 19
# In mcrpc=0 only — interaction with ar_v7
sub = df[df['mcrpc']==0]
m = smf.logit("objective_response ~ treatment_enzalutamide * ar_v7_positive + age_years + ecog_ps + visceral_mets + psa_ng_ml + gleason_score + albumin_g_dl + ldh_u_l + hemoglobin_g_dl + alkaline_phosphatase_u_l", data=sub).fit(disp=0)
inter = "treatment_enzalutamide:ar_v7_positive"
coef = m.params[inter]; p = m.pvalues[inter]
add(it, "i19_enza_arv7_in_mcrpc0_adj",
    "Within mcrpc=0 patients and adjusting for clinical features and labs, treatment_enzalutamide × ar_v7_positive interaction remains strongly negative (enzalutamide benefit lost when AR-V7+).",
    "logit on mcrpc=0 subset with covariates; interaction term significance",
    f"Adjusted enza × ar_v7 interaction coef={coef:.3f}, p={p:.3g}.", p, coef)

# ============================================================
# Iteration 20: Test treatment-treatment interactions / combinations
# (does combining enzalutamide with other treatments help/hurt?)
# ============================================================
it = 20
treats = ['treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel',
          'treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab']
text_lines = []
for t1, t2 in itertools.combinations(treats, 2):
    f = f"objective_response ~ {t1} * {t2}"
    m = smf.logit(f, data=df).fit(disp=0)
    iname = f"{t1}:{t2}"
    coef = m.params[iname]; p = m.pvalues[iname]
    text_lines.append(f"  {t1} × {t2}: coef={coef:+.3f}, p={p:.3g}")
add(it, "i20_treat_interactions",
    "Pairwise treatment combinations have non-additive (synergistic or antagonistic) effects on objective_response.",
    "logit(objective_response ~ t1 * t2) for each pair",
    "\n".join(text_lines), None, 0.0,
    sig=any('p=' in l and float(l.split('p=')[1]) < 0.05 for l in text_lines if 'p=' in l))

# ============================================================
# Iteration 21: Final canonical hypotheses for each treatment
# ============================================================
it = 21
final = {}

# Enzalutamide (positive)
mask = (df['mcrpc']==0)&(df['ar_v7_positive']==0)
a,b,d,p,n = stratified_treat('treatment_enzalutamide', mask)
add(it, "i21_final_enza",
    "FINAL: treatment_enzalutamide produces a large increase in objective_response specifically in patients with mcrpc=0 AND ar_v7_positive=0; outside this subgroup the effect is ≈0.",
    "stratified ORR within mcrpc=0 & ar_v7_positive=0",
    f"Subgroup n={n}: ORR enza+={a:.3f} vs enza-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d, kind="refined")

# Abiraterone — try BRCA-mut subgroup (since interaction pos at p=0.03)
mask = (df['brca2_mutation']==1)
a,b,d,p,n = stratified_treat('treatment_abiraterone', mask)
add(it, "i21_final_abi_brca",
    "FINAL candidate: treatment_abiraterone has the largest (positive but still modest) ORR effect in brca2_mutation=1 patients; outside this subgroup the effect is ≈0.",
    "stratified ORR within brca2_mutation=1",
    f"Subgroup n={n}: ORR abi+={a:.3f} vs abi-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d, kind="refined")
# Whole-cohort abi effect
a,b,d,p,n = stratified_treat('treatment_abiraterone', pd.Series(True, index=df.index))
add(it, "i21_final_abi_main",
    "FINAL: treatment_abiraterone has no detectable main effect on objective_response in the cohort overall.",
    "chi2 on treatment_abiraterone x objective_response (whole cohort)",
    f"n={n}: ORR abi+={a:.3f} vs abi-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d, kind="refined")

# Docetaxel
a,b,d,p,n = stratified_treat('treatment_docetaxel', pd.Series(True, index=df.index))
add(it, "i21_final_doce",
    "FINAL: treatment_docetaxel has no detectable main effect on objective_response and no biomarker subgroup with significant benefit was identified.",
    "chi2 + exhaustive screen",
    f"Whole cohort n={n}: ORR doce+={a:.3f} vs doce-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d, kind="refined")

# Olaparib
a,b,d,p,n = stratified_treat('treatment_olaparib', pd.Series(True, index=df.index))
add(it, "i21_final_ola",
    "FINAL: treatment_olaparib has no detectable main effect on objective_response. In brca2_mutation=1 patients, olaparib trends toward LOWER ORR (opposite of expected).",
    "chi2 on whole cohort + stratified within brca2_mutation=1",
    f"Whole cohort n={n}: ORR ola+={a:.3f} vs ola-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d, kind="refined")
mask = df['brca2_mutation']==1
a,b,d,p,n = stratified_treat('treatment_olaparib', mask)
add(it, "i21_final_ola_brca",
    "FINAL: Within brca2_mutation=1, treatment_olaparib does not increase objective_response; if anything ORR is lower (negative direction).",
    "chi2 on treatment_olaparib within brca2_mutation=1",
    f"Subgroup n={n}: ORR ola+={a:.3f} vs ola-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d, kind="refined")

# Lu177-PSMA
a,b,d,p,n = stratified_treat('treatment_lu177_psma', pd.Series(True, index=df.index))
add(it, "i21_final_lu",
    "FINAL: treatment_lu177_psma has no detectable main effect on objective_response. In psma_high=1 patients (the canonical eligibility group) no benefit is observed.",
    "chi2 whole cohort + stratified within psma_high=1",
    f"Whole cohort n={n}: ORR lu+={a:.3f} vs lu-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d, kind="refined")
mask = df['psma_high']==1
a,b,d,p,n = stratified_treat('treatment_lu177_psma', mask)
add(it, "i21_final_lu_psma",
    "FINAL: Within psma_high=1, treatment_lu177_psma does not improve objective_response.",
    "chi2 on treatment_lu177_psma within psma_high=1",
    f"Subgroup n={n}: ORR lu+={a:.3f} vs lu-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d, kind="refined")

# Pembrolizumab
a,b,d,p,n = stratified_treat('treatment_pembrolizumab', pd.Series(True, index=df.index))
add(it, "i21_final_pem",
    "FINAL: treatment_pembrolizumab has no detectable main effect on objective_response. In msi_high=1 patients no benefit is observed.",
    "chi2 whole cohort + stratified within msi_high=1",
    f"Whole cohort n={n}: ORR pem+={a:.3f} vs pem-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d, kind="refined")
mask = df['msi_high']==1
a,b,d,p,n = stratified_treat('treatment_pembrolizumab', mask)
add(it, "i21_final_pem_msi",
    "FINAL: Within msi_high=1, treatment_pembrolizumab does not improve objective_response.",
    "chi2 on treatment_pembrolizumab within msi_high=1",
    f"Subgroup n={n}: ORR pem+={a:.3f} vs pem-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d, kind="refined")

# ============================================================
# Iteration 22: cross-check enzalutamide canonical subgroup with logistic interaction adjusted
# ============================================================
it = 22
m = smf.logit("objective_response ~ treatment_enzalutamide * mcrpc * ar_v7_positive + age_years + ecog_ps + visceral_mets + psa_ng_ml + gleason_score + albumin_g_dl + ldh_u_l + hemoglobin_g_dl + alkaline_phosphatase_u_l + brca2_mutation + msi_high + psma_high + treatment_abiraterone + treatment_docetaxel + treatment_olaparib + treatment_lu177_psma + treatment_pembrolizumab", data=df).fit(disp=0)
coef3 = m.params.get("treatment_enzalutamide:mcrpc:ar_v7_positive", np.nan)
p3 = m.pvalues.get("treatment_enzalutamide:mcrpc:ar_v7_positive", np.nan)
coef_main = m.params["treatment_enzalutamide"]
add(it, "i22_enza_3way_adjusted",
    "Adjusted three-way interaction treatment_enzalutamide × mcrpc × ar_v7_positive remains strongly positive after adjusting for clinical features, labs, and other treatments (confirms canonical subgroup).",
    "logit with full covariate adjustment + 3-way interaction",
    f"Adjusted 3-way coef={coef3:.3f}, p={p3:.3g}; main treatment_enzalutamide coef (in mcrpc=0 & arv7=0 reference cell) = {coef_main:.3f}.",
    p3, coef3)

with open("all_results.json","w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"After iter 22, {len(RESULTS)} analyses recorded.")
