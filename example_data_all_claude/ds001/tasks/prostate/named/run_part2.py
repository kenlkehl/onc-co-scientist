"""Iterations 5-12: deeper interactions, 3-way, biomarker prognostics."""
import json, math, itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
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

def logit(formula, data=None):
    d = df if data is None else data
    return smf.logit(formula, data=d).fit(disp=0)

def stratified_treat(treat, mask, label):
    """Returns (rate_treated, rate_untreated, diff, p, n_total)."""
    sub = df[mask]
    a = sub.loc[sub[treat]==1, 'objective_response'].mean()
    b = sub.loc[sub[treat]==0, 'objective_response'].mean()
    n_t = int((sub[treat]==1).sum()); n_c = int((sub[treat]==0).sum())
    if n_t < 5 or n_c < 5:
        return a, b, a-b, np.nan, n_t+n_c
    ct = pd.crosstab(sub[treat], sub['objective_response'])
    if ct.shape != (2,2):
        return a, b, a-b, np.nan, n_t+n_c
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return a, b, a-b, p, n_t+n_c

# ============================================================
# Iteration 5: Biomarker main effects (prognostic only — no treatment)
# ============================================================
it = 5
for col in ['brca2_mutation','ar_v7_positive','msi_high','psma_high']:
    a = df.loc[df[col]==1,'objective_response'].mean()
    b = df.loc[df[col]==0,'objective_response'].mean()
    ct = pd.crosstab(df[col], df['objective_response'])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    add(it, f"i5_{col}",
        f"{col}=1 patients have a different overall objective_response rate than {col}=0.",
        f"chi2 on {col}",
        f"ORR {col}=1: {a:.3f} (n={int(df[col].sum())}) vs {col}=0: {b:.3f}; diff={a-b:+.3f}, p={p:.3g}.",
        p, a-b)

# ============================================================
# Iteration 6: 3-way interactions for the targeted-therapy hypotheses
#   - olaparib effect requires BRCA2 AND mCRPC
#   - pembrolizumab effect requires MSI AND mCRPC
#   - lu177 effect requires PSMA-high AND mCRPC
# ============================================================
it = 6
def threeway(treat, biom, hid, htext):
    # Stratified rates in each cell
    rows = []
    for biom_v in (0,1):
        for mc in (0,1):
            mask = (df[biom]==biom_v)&(df['mcrpc']==mc)
            a, b, d, p, n = stratified_treat(treat, mask, "")
            rows.append((biom_v, mc, a, b, d, p, n))
    # Test the three-way interaction
    f = f"objective_response ~ {treat} * {biom} * mcrpc"
    m = logit(f)
    iname = f"{treat}:{biom}:mcrpc"
    coef = m.params.get(iname, np.nan)
    p3 = m.pvalues.get(iname, np.nan)
    # Specifically test treatment effect within {biom=1, mcrpc=1}
    mask = (df[biom]==1)&(df['mcrpc']==1)
    a,b,d,p_in_target,n = stratified_treat(treat, mask, f"{biom}=1 & mcrpc=1")
    s = (f"3-way interaction {treat}*{biom}*mcrpc: coef={coef:.3f}, p={p3:.3g}. "
         f"Within {biom}=1 & mcrpc=1 (n={n}): ORR {treat}+={a:.3f} vs -={b:.3f}, diff={d:+.3f}, p={p_in_target:.3g}. "
         f"Cell rates (biom,mcrpc,treat+,treat-,diff): " +
         "; ".join([f"({b_v},{m_v}): +={ai:.3f}/-={bi:.3f}/d={di:+.3f}" for (b_v,m_v,ai,bi,di,_,_) in rows]))
    add(it, hid, htext, f"logit({f})", s, p_in_target, d)

threeway('treatment_olaparib','brca2_mutation','i6_ola_brca_mcrpc',
         "treatment_olaparib improves objective_response specifically in patients with brca2_mutation=1 AND mcrpc=1.")
threeway('treatment_pembrolizumab','msi_high','i6_pem_msi_mcrpc',
         "treatment_pembrolizumab improves objective_response specifically in patients with msi_high=1 AND mcrpc=1.")
threeway('treatment_lu177_psma','psma_high','i6_lu_psma_mcrpc',
         "treatment_lu177_psma improves objective_response specifically in patients with psma_high=1 AND mcrpc=1.")

# ============================================================
# Iteration 7: Stratified treatment effects across many subgroups
# (search) - look for hidden modifiers for olaparib/pembro/lu177
# ============================================================
it = 7

def screen(treat, label):
    """For each binary or quartile-grouped feature, compute stratified treatment effect."""
    feats_bin = ['mcrpc','visceral_mets','brca2_mutation','ar_v7_positive','msi_high','psma_high']
    feats_cat = {'ecog_ps':[0,1,2], 'gleason_score':[6,7,8,9,10]}
    feats_cont = ['psa_ng_ml','albumin_g_dl','ldh_u_l','crp_mg_l','nlr','hemoglobin_g_dl','alkaline_phosphatase_u_l']
    out = []
    for fcol in feats_bin:
        for v in (0,1):
            mask = df[fcol]==v
            a,b,d,p,n = stratified_treat(treat, mask, f"{fcol}={v}")
            out.append((f"{fcol}={v}", n, a, b, d, p))
    for fcol, vals in feats_cat.items():
        for v in vals:
            mask = df[fcol]==v
            a,b,d,p,n = stratified_treat(treat, mask, f"{fcol}={v}")
            out.append((f"{fcol}={v}", n, a, b, d, p))
    for fcol in feats_cont:
        med = df[fcol].median()
        for op,v_label in [('<=', f'<={med:.2f}'), ('>', f'>{med:.2f}')]:
            mask = (df[fcol]<=med) if op=='<=' else (df[fcol]>med)
            a,b,d,p,n = stratified_treat(treat, mask, f"{fcol}{v_label}")
            out.append((f"{fcol}{v_label}", n, a, b, d, p))
    return out

for treat in ['treatment_olaparib','treatment_pembrolizumab','treatment_lu177_psma',
              'treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel']:
    out = screen(treat, treat)
    # find biggest positive diff (treated > untreated) by significant ones
    sig = [r for r in out if r[5] is not None and not (isinstance(r[5],float) and math.isnan(r[5])) and r[5] < 0.05]
    sig.sort(key=lambda r: r[4], reverse=True)
    top = sig[:5]
    text = "Significant subgroups (p<0.05) where {} treatment effect on ORR is largest:\n".format(treat)
    if not sig:
        text += "  (none)"
    else:
        for label, n, a, b, d, p in top:
            text += f"  {label}: ORR {a:.3f} (treated) vs {b:.3f} (control), diff={d:+.3f}, n={n}, p={p:.3g}\n"
    # Use top diff as a hypothesis test result if any
    if sig:
        label, n, a, b, d, p = sig[0]
        add(it, f"i7_{treat}_top",
            f"In screening across single-feature subgroups, {treat} has its largest significant ORR benefit in subgroup '{label}'.",
            f"stratified ORR comparison across binary, categorical, median-split features",
            text.strip(), p, d)
    else:
        add(it, f"i7_{treat}_top",
            f"No single-feature subgroup shows a statistically significant ORR benefit for {treat}.",
            "stratified ORR comparison across binary, categorical, median-split features",
            text.strip(), None, 0.0, sig=False)

# ============================================================
# Iteration 8: Double-subgroup screen for olaparib, pembrolizumab, lu177
#   (since main effects are absent, look for biomarker pairs that
#    define the responder population)
# ============================================================
it = 8
def double_screen(treat, biom_options):
    """Test treatment effect in every pairwise intersection of two binary features=1."""
    pairs = list(itertools.combinations(biom_options, 2))
    out = []
    for f1, f2 in pairs:
        mask = (df[f1]==1)&(df[f2]==1)
        a,b,d,p,n = stratified_treat(treat, mask, f"{f1}=1 & {f2}=1")
        out.append((f"{f1}=1 & {f2}=1", n, a, b, d, p))
    return out

biom_options = ['mcrpc','visceral_mets','brca2_mutation','ar_v7_positive','msi_high','psma_high']
for treat in ['treatment_olaparib','treatment_pembrolizumab','treatment_lu177_psma']:
    out = double_screen(treat, biom_options)
    sig = [r for r in out if r[5] is not None and not (isinstance(r[5],float) and math.isnan(r[5])) and r[5] < 0.05 and r[4] > 0]
    sig.sort(key=lambda r: r[4], reverse=True)
    text = f"Pairwise two-biomarker subgroups (both =1) with significant positive {treat} ORR diffs:\n"
    if not sig:
        text += "  (none)"
    else:
        for label, n, a, b, d, p in sig[:8]:
            text += f"  {label}: ORR {a:.3f} (treated) vs {b:.3f} (control), diff={d:+.3f}, n={n}, p={p:.3g}\n"
    add(it, f"i8_{treat}_pairs",
        f"For {treat}, the largest ORR benefit appears in the joint subgroup defined by the top biomarker pair from a pairwise screen.",
        "stratified ORR comparison in every pair of biomarker=1 intersections",
        text.strip(),
        sig[0][5] if sig else None,
        sig[0][4] if sig else 0.0)

# ============================================================
# Iteration 9: Continuous-modifier interactions for enzalutamide
#   - is enza efficacy modulated by PSA, ECOG, etc on top of AR-V7?
# ============================================================
it = 9
for mod in ['ecog_ps','psa_ng_ml','albumin_g_dl','ldh_u_l','crp_mg_l','nlr','hemoglobin_g_dl','alkaline_phosphatase_u_l','gleason_score']:
    f = f"objective_response ~ treatment_enzalutamide * {mod}"
    m = logit(f)
    iname = f"treatment_enzalutamide:{mod}"
    coef = m.params[iname]; p = m.pvalues[iname]
    add(it, f"i9_enza_{mod}",
        f"The enzalutamide treatment effect on objective_response is modified by {mod} (interaction).",
        f"logit({f})",
        f"Interaction coef enza:{mod} = {coef:.4g}, p={p:.3g}.", p, coef)

# Also enzalutamide × ar_v7 within ECOG strata to confirm
for ecog in (0,1,2):
    mask = df['ecog_ps']==ecog
    f = f"objective_response ~ treatment_enzalutamide * ar_v7_positive"
    m = smf.logit(f, data=df[mask]).fit(disp=0)
    iname = "treatment_enzalutamide:ar_v7_positive"
    coef = m.params[iname]; p = m.pvalues[iname]
    sub = df[mask]
    a = sub.loc[(sub['ar_v7_positive']==0)&(sub['treatment_enzalutamide']==1),'objective_response'].mean()
    b = sub.loc[(sub['ar_v7_positive']==0)&(sub['treatment_enzalutamide']==0),'objective_response'].mean()
    add(it, f"i9_enza_arv7_ecog{ecog}",
        f"Among ecog_ps={ecog} patients, treatment_enzalutamide × ar_v7_positive interaction (negative interaction expected) holds.",
        f"logit({f}) on ecog_ps={ecog} subset",
        f"ECOG={ecog}: AR-V7- ORR enza+={a:.3f} vs enza-={b:.3f}; interaction coef={coef:.3f}, p={p:.3g}.", p, coef)

# ============================================================
# Iteration 10: Multivariable response model (adjust for treatments + features)
# ============================================================
it = 10
features = ['age_years','ecog_ps','mcrpc','visceral_mets','psa_ng_ml','gleason_score',
            'brca2_mutation','ar_v7_positive','msi_high','psma_high',
            'albumin_g_dl','ldh_u_l','weight_loss_pct_6mo','crp_mg_l','nlr',
            'hemoglobin_g_dl','alkaline_phosphatase_u_l','ast_u_l','alt_u_l',
            'total_bilirubin_mg_dl','creatinine_mg_dl','bun_mg_dl',
            'sodium_meq_l','potassium_meq_l','calcium_mg_dl',
            'treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel',
            'treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab']
formula = "objective_response ~ " + " + ".join(features)
m = logit(formula)
# Report effects with significance
for feat in features:
    coef = m.params[feat]; p = m.pvalues[feat]
    add(it, f"i10_{feat}",
        f"In a multivariable logistic model for objective_response, {feat} is associated with response (signed coefficient).",
        f"logit({formula})",
        f"Adj coef {feat}={coef:.4g}, p={p:.3g}.", p, coef)

with open("all_results.json","w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"After iter 10, {len(RESULTS)} analyses recorded.")
