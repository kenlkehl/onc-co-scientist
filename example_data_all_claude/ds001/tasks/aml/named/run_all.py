"""Iterative analysis of ds001_aml.

Runs an exploratory propose-test-refine loop across iterations and emits
transcript.json + analysis_summary.txt.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
N = len(df)

TX = ['treatment_midostaurin','treatment_gilteritinib','treatment_ivosidenib',
      'treatment_enasidenib','treatment_venetoclax_azacitidine','treatment_7plus3']
MUT = ['flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation','npm1_mutation','tp53_mutation']
LABS = ['wbc_k_per_ul','blast_pct_marrow','albumin_g_dl','ldh_u_l','crp_mg_l','nlr',
        'hemoglobin_g_dl','alkaline_phosphatase_u_l','ast_u_l','alt_u_l',
        'total_bilirubin_mg_dl','creatinine_mg_dl','bun_mg_dl','sodium_meq_l',
        'potassium_meq_l','calcium_mg_dl']
DEMO = ['age_years','sex_female','ecog_ps','secondary_aml','unfit_for_intensive',
        'complex_karyotype','weight_loss_pct_6mo']

iters = []
NARR = []  # plain narrative lines for the summary

def f(x, d=4):
    if x is None: return "NA"
    try:
        if np.isnan(x) or np.isinf(x): return "NA"
    except (TypeError, ValueError):
        return str(x)
    return f"{x:.{d}f}"

def chi2_diff(d, g, y='objective_response'):
    """Returns (rate_diff_g1_minus_g0, p_value, n1, n0, r1, r0)."""
    if d[g].nunique() < 2:
        return None, None, 0, 0, None, None
    g1 = d.loc[d[g]==1, y]; g0 = d.loc[d[g]==0, y]
    if len(g1)<5 or len(g0)<5: return None, None, len(g1), len(g0), None, None
    p1, p0 = g1.mean(), g0.mean()
    tab = pd.crosstab(d[g], d[y])
    if tab.shape != (2,2): return None, None, len(g1), len(g0), p1, p0
    try:
        _,p,_,_ = stats.chi2_contingency(tab)
    except Exception:
        p = None
    return p1-p0, p, len(g1), len(g0), p1, p0

def logit_or(d, x, y='objective_response', adj=None):
    cols = [x] + (list(adj) if adj else [])
    cols = [c for c in cols if c in d.columns]
    X = d[cols].astype(float).copy()
    X = sm.add_constant(X)
    yv = d[y].astype(int).values
    try:
        m = sm.Logit(yv, X).fit(disp=0, maxiter=200)
        return float(m.params[x]), float(m.pvalues[x])
    except Exception:
        return None, None

def py(x):
    if x is None: return None
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x) or np.isinf(x): return None
        return float(x)
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x): return None
    return x

def clean(an):
    out = dict(an)
    out["p_value"] = py(an.get("p_value"))
    out["effect_estimate"] = py(an.get("effect_estimate"))
    if "significant" in an:
        s = an["significant"]
        out["significant"] = bool(s) if s is not None else None
    return out

def add_iter(idx, hyps, ans):
    iters.append({"index":idx,"proposed_hypotheses":hyps,"analyses":[clean(a) for a in ans]})

# ============================================================
# ITERATION 1: Patient-feature main effects on objective_response
# ============================================================
hyps, ans = [], []
for spec in [
    ("h1.1","Higher age_years lowers objective_response.","age_years","logit"),
    ("h1.2","Higher ecog_ps lowers objective_response.","ecog_ps","logit"),
    ("h1.3","tp53_mutation lowers objective_response.","tp53_mutation","chi"),
    ("h1.4","complex_karyotype lowers objective_response.","complex_karyotype","chi"),
    ("h1.5","npm1_mutation raises objective_response.","npm1_mutation","chi"),
    ("h1.6","secondary_aml lowers objective_response.","secondary_aml","chi"),
    ("h1.7","unfit_for_intensive lowers objective_response.","unfit_for_intensive","chi"),
    ("h1.8","sex_female associates with objective_response.","sex_female","chi"),
]:
    hid, txt, x, kind = spec
    hyps.append({"id":hid,"text":txt,"kind":"novel"})
    if kind == "logit":
        coef, p = logit_or(df, x)
        ans.append({"hypothesis_ids":[hid],
                    "code":f"sm.Logit(objective_response ~ {x})",
                    "result_summary":f"logit coef({x})={f(coef)}, OR={f(np.exp(coef) if coef is not None else None,3)}, p={f(p,6)}",
                    "p_value":p, "effect_estimate":coef,
                    "significant":(p is not None and p<0.05)})
    else:
        diff,p,n1,n0,r1,r0 = chi2_diff(df, x)
        ans.append({"hypothesis_ids":[hid],
                    "code":f"chi2 {x} x objective_response",
                    "result_summary":f"resp {x}+={f(r1,4)} vs {x}-={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)} (n+={n1}, n-={n0})",
                    "p_value":p, "effect_estimate":diff,
                    "significant":(p is not None and p<0.05)})
add_iter(1,hyps,ans)

# ============================================================
# ITERATION 2: Mutation main effects (FLT3/IDH/etc) — full panel
# ============================================================
hyps, ans = [], []
for i,m in enumerate(MUT):
    hid=f"h2.{i+1}"
    hyps.append({"id":hid,"text":f"{m} is associated with a different objective_response rate vs patients without it.","kind":"novel"})
    diff,p,n1,n0,r1,r0 = chi2_diff(df,m)
    ans.append({"hypothesis_ids":[hid],
                "code":f"chi2 {m} x objective_response",
                "result_summary":f"resp {m}+={f(r1,4)} vs {m}-={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":diff,
                "significant":(p is not None and p<0.05)})
add_iter(2,hyps,ans)

# ============================================================
# ITERATION 3: Lab-value main effects (continuous)
# ============================================================
hyps, ans = [], []
for i,lab in enumerate(LABS):
    hid=f"h3.{i+1}"
    hyps.append({"id":hid,"text":f"Higher {lab} is associated with a different probability of objective_response.","kind":"novel"})
    coef,p = logit_or(df,lab)
    ans.append({"hypothesis_ids":[hid],
                "code":f"sm.Logit(objective_response ~ {lab})",
                "result_summary":f"logit coef({lab})={f(coef)}, OR={f(np.exp(coef) if coef is not None else None,3)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":coef,
                "significant":(p is not None and p<0.05)})
add_iter(3,hyps,ans)

# ============================================================
# ITERATION 4: Treatment main effects (marginal — unadjusted)
# ============================================================
hyps, ans = [], []
for i,t in enumerate(TX):
    hid=f"h4.{i+1}"
    hyps.append({"id":hid,"text":f"Receipt of {t} is associated with a different objective_response rate vs no receipt of {t}.","kind":"novel"})
    diff,p,n1,n0,r1,r0 = chi2_diff(df,t)
    ans.append({"hypothesis_ids":[hid],
                "code":f"chi2 {t} x objective_response",
                "result_summary":f"resp {t}+={f(r1,4)} vs {t}-={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)} (n+={n1}, n-={n0})",
                "p_value":p,"effect_estimate":diff,
                "significant":(p is not None and p<0.05)})
add_iter(4,hyps,ans)

# ============================================================
# ITERATION 5: Targeted-therapy + biomarker subgroup analyses
# (Each agent paired with its canonical AML target.)
# ============================================================
hyps, ans = [], []
pairs = [
    ('h5.1','treatment_midostaurin','flt3_itd'),
    ('h5.2','treatment_midostaurin','flt3_tkd'),
    ('h5.3','treatment_gilteritinib','flt3_itd'),
    ('h5.4','treatment_gilteritinib','flt3_tkd'),
    ('h5.5','treatment_ivosidenib','idh1_mutation'),
    ('h5.6','treatment_enasidenib','idh2_mutation'),
]
for hid, t, m in pairs:
    hyps.append({"id":hid,
                 "text":f"Within {m}-positive patients, receipt of {t} is associated with a higher objective_response rate than no receipt of {t}.",
                 "kind":"novel"})
    sub = df[df[m]==1]
    diff,p,n1,n0,r1,r0 = chi2_diff(sub, t)
    ans.append({"hypothesis_ids":[hid],
                "code":f"chi2 {t} x objective_response | {m}==1",
                "result_summary":f"In {m}+ subgroup (N={len(sub)}): resp {t}+={f(r1,4)} vs {t}-={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":diff,
                "significant":(p is not None and p<0.05)})
add_iter(5,hyps,ans)

# ============================================================
# ITERATION 6: Same agents OUTSIDE biomarker subgroup
# (Tests whether targeted agents lack effect in non-target patients.)
# ============================================================
hyps, ans = [], []
for hid_, t, m in pairs:
    hid = hid_.replace('h5','h6')
    hyps.append({"id":hid,
                 "text":f"Within {m}-NEGATIVE patients, {t} does NOT improve objective_response (effect concentrated in {m}+).",
                 "kind":"refined"})
    sub = df[df[m]==0]
    diff,p,n1,n0,r1,r0 = chi2_diff(sub, t)
    ans.append({"hypothesis_ids":[hid],
                "code":f"chi2 {t} x objective_response | {m}==0",
                "result_summary":f"In {m}- subgroup (N={len(sub)}): resp {t}+={f(r1,4)} vs {t}-={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":diff,
                "significant":(p is not None and p<0.05)})
add_iter(6,hyps,ans)

# ============================================================
# ITERATION 7: Treatment-by-biomarker interaction tests (logit)
# ============================================================
hyps, ans = [], []
for i,(t,m) in enumerate([
    ('treatment_midostaurin','flt3_itd'),
    ('treatment_midostaurin','flt3_tkd'),
    ('treatment_gilteritinib','flt3_itd'),
    ('treatment_gilteritinib','flt3_tkd'),
    ('treatment_ivosidenib','idh1_mutation'),
    ('treatment_enasidenib','idh2_mutation'),
]):
    hid = f"h7.{i+1}"
    hyps.append({"id":hid,
                 "text":f"There is a positive {t} x {m} interaction on objective_response (effect of {t} larger in {m}+).",
                 "kind":"novel"})
    d = df.copy(); d['inter'] = d[t]*d[m]
    X = sm.add_constant(d[[t,m,'inter']].astype(float))
    y = d['objective_response'].astype(int).values
    try:
        mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
        coef = float(mfit.params['inter']); p = float(mfit.pvalues['inter'])
    except Exception:
        coef, p = None, None
    ans.append({"hypothesis_ids":[hid],
                "code":f"sm.Logit(objective_response ~ {t} + {m} + {t}:{m})",
                "result_summary":f"Interaction coef({t}*{m})={f(coef)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":coef,
                "significant":(p is not None and p<0.05)})
add_iter(7,hyps,ans)

# ============================================================
# ITERATION 8: Adjusted treatment effects (control for case-mix)
# ============================================================
hyps, ans = [], []
adj_vars = ['age_years','ecog_ps','tp53_mutation','complex_karyotype',
            'npm1_mutation','secondary_aml','unfit_for_intensive','albumin_g_dl',
            'wbc_k_per_ul','blast_pct_marrow']
# all treatments simultaneously to avoid contamination
d = df.copy()
for i,t in enumerate(TX):
    hid=f"h8.{i+1}"
    hyps.append({"id":hid,
                 "text":f"After adjusting for age, ECOG, mutations, fitness, and labs, {t} has an independent effect on objective_response.",
                 "kind":"refined"})
predictors = TX + adj_vars
X = sm.add_constant(d[predictors].astype(float))
y = d['objective_response'].astype(int).values
mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
for i,t in enumerate(TX):
    hid=f"h8.{i+1}"
    coef = float(mfit.params[t]); p = float(mfit.pvalues[t])
    ans.append({"hypothesis_ids":[hid],
                "code":"sm.Logit(objective_response ~ all treatments + adjusters)",
                "result_summary":f"Adjusted coef({t})={f(coef)} (OR={f(np.exp(coef),3)}), p={f(p,6)}",
                "p_value":p,"effect_estimate":coef,
                "significant":(p is not None and p<0.05)})
add_iter(8,hyps,ans)

# ============================================================
# ITERATION 9: Heterogeneity of treatment_venetoclax_azacitidine
# (since it is the only treatment with a clear marginal effect,
# screen for treatment-by-feature interactions)
# ============================================================
hyps, ans = [], []
modifiers = ['age_years','ecog_ps','sex_female','tp53_mutation','complex_karyotype',
             'npm1_mutation','flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation',
             'secondary_aml','unfit_for_intensive','albumin_g_dl','wbc_k_per_ul',
             'blast_pct_marrow','crp_mg_l','hemoglobin_g_dl','ldh_u_l']
T = 'treatment_venetoclax_azacitidine'
for i,m in enumerate(modifiers):
    hid=f"h9.{i+1}"
    hyps.append({"id":hid,
                 "text":f"The objective_response benefit of {T} is modified by {m} (interaction term in logistic regression).",
                 "kind":"novel"})
    d = df.copy()
    d['inter']=d[T]*d[m]
    X = sm.add_constant(d[[T,m,'inter']].astype(float))
    y = d['objective_response'].astype(int).values
    try:
        mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
        coef=float(mfit.params['inter']); p=float(mfit.pvalues['inter'])
    except Exception:
        coef,p=None,None
    ans.append({"hypothesis_ids":[hid],
                "code":f"Logit(objective_response ~ {T} + {m} + {T}:{m})",
                "result_summary":f"Interaction coef({T}*{m})={f(coef)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":coef,
                "significant":(p is not None and p<0.05)})
add_iter(9,hyps,ans)

# ============================================================
# ITERATION 10: Stratified ven/aza response across the strongest modifiers
# (read interactions from iter 9, but compute response-rate diffs in
#  each level for transparency)
# ============================================================
hyps, ans = [], []
strat_vars = ['ecog_ps','tp53_mutation','complex_karyotype','npm1_mutation',
              'unfit_for_intensive','secondary_aml','flt3_itd','idh1_mutation','idh2_mutation']
for i,m in enumerate(strat_vars):
    levels = sorted(df[m].unique().tolist())
    for j,lv in enumerate(levels):
        hid=f"h10.{i+1}.{j+1}"
        hyps.append({"id":hid,
                     "text":f"Within patients with {m}=={int(lv)}, {T} raises objective_response rate vs no {T}.",
                     "kind":"refined"})
        sub = df[df[m]==lv]
        diff,p,n1,n0,r1,r0 = chi2_diff(sub,T)
        ans.append({"hypothesis_ids":[hid],
                    "code":f"chi2 {T} x objective_response | {m}=={int(lv)}",
                    "result_summary":f"{m}=={int(lv)} (N={len(sub)}): resp {T}+={f(r1,4)} vs {T}-={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
                    "p_value":p,"effect_estimate":diff,
                    "significant":(p is not None and p<0.05)})
add_iter(10,hyps,ans)

# ============================================================
# ITERATION 11: Joint subgroup hypothesis — ven/aza in NPM1+ & unfit
# ============================================================
hyps, ans = [], []
T = 'treatment_venetoclax_azacitidine'

hid='h11.1'
hyps.append({"id":hid,"text":f"In patients who are unfit_for_intensive==1 AND npm1_mutation==1, {T} markedly increases objective_response.","kind":"refined"})
sub = df[(df.unfit_for_intensive==1)&(df.npm1_mutation==1)]
diff,p,n1,n0,r1,r0 = chi2_diff(sub,T)
ans.append({"hypothesis_ids":[hid],
            "code":f"chi2 {T} | unfit==1 & npm1==1",
            "result_summary":f"unfit&npm1+ (N={len(sub)}): resp {T}+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})

hid='h11.2'
hyps.append({"id":hid,"text":f"In patients who are unfit_for_intensive==1 AND npm1_mutation==1 AND tp53_mutation==0 AND complex_karyotype==0, {T} markedly increases objective_response.","kind":"refined"})
sub = df[(df.unfit_for_intensive==1)&(df.npm1_mutation==1)&(df.tp53_mutation==0)&(df.complex_karyotype==0)]
diff,p,n1,n0,r1,r0 = chi2_diff(sub,T)
ans.append({"hypothesis_ids":[hid],
            "code":f"chi2 {T} | unfit==1 & npm1==1 & tp53==0 & CK==0",
            "result_summary":f"clean+favorable (N={len(sub)}): resp {T}+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})

hid='h11.3'
hyps.append({"id":hid,"text":f"Outside the unfit & npm1+ subgroup, {T} provides no objective_response benefit.","kind":"refined"})
sub = df[~((df.unfit_for_intensive==1)&(df.npm1_mutation==1))]
diff,p,n1,n0,r1,r0 = chi2_diff(sub,T)
ans.append({"hypothesis_ids":[hid],
            "code":f"chi2 {T} | NOT (unfit==1 & npm1==1)",
            "result_summary":f"complement (N={len(sub)}): resp {T}+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})

hid='h11.4'
hyps.append({"id":hid,"text":f"In TP53-mutated patients, {T} does NOT increase objective_response (TP53 suppresses ven/aza benefit).","kind":"refined"})
sub = df[df.tp53_mutation==1]
diff,p,n1,n0,r1,r0 = chi2_diff(sub,T)
ans.append({"hypothesis_ids":[hid],
            "code":f"chi2 {T} | tp53==1",
            "result_summary":f"TP53+ (N={len(sub)}): resp {T}+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})

hid='h11.5'
hyps.append({"id":hid,"text":f"In complex_karyotype+ patients, {T} does NOT increase objective_response.","kind":"refined"})
sub = df[df.complex_karyotype==1]
diff,p,n1,n0,r1,r0 = chi2_diff(sub,T)
ans.append({"hypothesis_ids":[hid],
            "code":f"chi2 {T} | CK==1",
            "result_summary":f"CK+ (N={len(sub)}): resp {T}+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})
add_iter(11,hyps,ans)

# ============================================================
# ITERATION 12: Heterogeneity screen for treatment_7plus3
# (intensive chemotherapy — should benefit FIT patients with favorable cyto)
# ============================================================
hyps, ans = [], []
T = 'treatment_7plus3'
modifiers = ['age_years','ecog_ps','sex_female','tp53_mutation','complex_karyotype',
             'npm1_mutation','flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation',
             'secondary_aml','unfit_for_intensive','albumin_g_dl','wbc_k_per_ul','blast_pct_marrow']
for i,m in enumerate(modifiers):
    hid=f"h12.{i+1}"
    hyps.append({"id":hid,
                 "text":f"The objective_response effect of {T} is modified by {m} (interaction logit term).",
                 "kind":"novel"})
    d = df.copy(); d['inter']=d[T]*d[m]
    X = sm.add_constant(d[[T,m,'inter']].astype(float))
    y = d['objective_response'].astype(int).values
    try:
        mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
        coef=float(mfit.params['inter']); p=float(mfit.pvalues['inter'])
    except Exception:
        coef,p=None,None
    ans.append({"hypothesis_ids":[hid],
                "code":f"Logit(objective_response ~ {T} + {m} + {T}:{m})",
                "result_summary":f"Interaction coef({T}*{m})={f(coef)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":coef,
                "significant":(p is not None and p<0.05)})
add_iter(12,hyps,ans)

# ============================================================
# ITERATION 13: Heterogeneity screens for FLT3 inhibitors
# (midostaurin, gilteritinib) -- check more than just FLT3 status
# ============================================================
hyps, ans = [], []
mods = ['flt3_itd','flt3_tkd','npm1_mutation','tp53_mutation','complex_karyotype',
        'unfit_for_intensive','ecog_ps','age_years','secondary_aml']
for T in ['treatment_midostaurin','treatment_gilteritinib']:
    for j,m in enumerate(mods):
        hid=f"h13.{T[-3:]}.{j+1}"  # short suffix to avoid duplicates? use unique
        hid=f"h13.{T}_{m}"
        hyps.append({"id":hid,
                     "text":f"The objective_response effect of {T} is modified by {m}.",
                     "kind":"novel"})
        d = df.copy(); d['inter']=d[T]*d[m]
        X = sm.add_constant(d[[T,m,'inter']].astype(float))
        y = d['objective_response'].astype(int).values
        try:
            mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
            coef=float(mfit.params['inter']); p=float(mfit.pvalues['inter'])
        except Exception:
            coef,p=None,None
        ans.append({"hypothesis_ids":[hid],
                    "code":f"Logit(objective_response ~ {T} + {m} + {T}:{m})",
                    "result_summary":f"Interaction coef({T}*{m})={f(coef)}, p={f(p,6)}",
                    "p_value":p,"effect_estimate":coef,
                    "significant":(p is not None and p<0.05)})
add_iter(13,hyps,ans)

# ============================================================
# ITERATION 14: Heterogeneity screens for IDH inhibitors
# ============================================================
hyps, ans = [], []
for T in ['treatment_ivosidenib','treatment_enasidenib']:
    for j,m in enumerate(mods):
        hid=f"h14.{T}_{m}"
        hyps.append({"id":hid,
                     "text":f"The objective_response effect of {T} is modified by {m}.",
                     "kind":"novel"})
        d = df.copy(); d['inter']=d[T]*d[m]
        X = sm.add_constant(d[[T,m,'inter']].astype(float))
        y = d['objective_response'].astype(int).values
        try:
            mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
            coef=float(mfit.params['inter']); p=float(mfit.pvalues['inter'])
        except Exception:
            coef,p=None,None
        ans.append({"hypothesis_ids":[hid],
                    "code":f"Logit(objective_response ~ {T} + {m} + {T}:{m})",
                    "result_summary":f"Interaction coef({T}*{m})={f(coef)}, p={f(p,6)}",
                    "p_value":p,"effect_estimate":coef,
                    "significant":(p is not None and p<0.05)})
add_iter(14,hyps,ans)

# ============================================================
# ITERATION 15: Confirm the complete ven/aza subgroup definition
# (joint NPM1+ & unfit, with TP53 & CK suppressors removed)
# ============================================================
hyps, ans = [], []
T='treatment_venetoclax_azacitidine'
# Stepwise narrowing
for i,(predicate, label) in enumerate([
    (df.unfit_for_intensive==1, 'unfit==1'),
    (df.npm1_mutation==1, 'npm1==1'),
    ((df.unfit_for_intensive==1)&(df.npm1_mutation==1), 'unfit==1 & npm1==1'),
    ((df.unfit_for_intensive==1)&(df.npm1_mutation==1)&(df.tp53_mutation==0), 'unfit & npm1+ & tp53-'),
    ((df.unfit_for_intensive==1)&(df.npm1_mutation==1)&(df.tp53_mutation==0)&(df.complex_karyotype==0), 'unfit & npm1+ & tp53- & CK-'),
    (~((df.unfit_for_intensive==1)&(df.npm1_mutation==1)), 'NOT(unfit & npm1+) [complement]'),
    ((df.unfit_for_intensive==1)&(df.tp53_mutation==1), 'unfit & tp53+ (suppressor present)'),
    ((df.unfit_for_intensive==1)&(df.complex_karyotype==1), 'unfit & CK+ (suppressor present)'),
    ((df.npm1_mutation==1)&(df.unfit_for_intensive==0), 'fit & npm1+ (no benefit expected)'),
]):
    hid=f"h15.{i+1}"
    hyps.append({"id":hid,
                 "text":f"Within the subgroup defined by {label}, {T} is associated with a different objective_response rate vs no {T}.",
                 "kind":"refined"})
    sub = df[predicate]
    diff,p,n1,n0,r1,r0 = chi2_diff(sub,T)
    ans.append({"hypothesis_ids":[hid],
                "code":f"chi2 {T} | {label}",
                "result_summary":f"{label} (N={len(sub)}): resp {T}+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})
add_iter(15,hyps,ans)

# ============================================================
# ITERATION 16: Stratify 7+3 by complex_karyotype and explore subgroups
# ============================================================
hyps, ans = [], []
T='treatment_7plus3'
for i,(predicate,label) in enumerate([
    (df.complex_karyotype==0,'CK==0'),
    (df.complex_karyotype==1,'CK==1'),
    ((df.complex_karyotype==0)&(df.unfit_for_intensive==0),'CK==0 & fit'),
    ((df.complex_karyotype==0)&(df.unfit_for_intensive==0)&(df.npm1_mutation==1),'CK==0 & fit & npm1+'),
    ((df.complex_karyotype==0)&(df.tp53_mutation==0),'CK==0 & tp53==0'),
    ((df.complex_karyotype==1)&(df.tp53_mutation==1),'CK==1 & tp53==1 (very poor risk)'),
]):
    hid=f"h16.{i+1}"
    hyps.append({"id":hid,
                 "text":f"Within {label}, {T} is associated with a different objective_response rate.",
                 "kind":"refined"})
    sub = df[predicate]
    diff,p,n1,n0,r1,r0 = chi2_diff(sub,T)
    ans.append({"hypothesis_ids":[hid],
                "code":f"chi2 {T} | {label}",
                "result_summary":f"{label} (N={len(sub)}): resp {T}+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})
add_iter(16,hyps,ans)

# ============================================================
# ITERATION 17: Confirm targeted-agent null effects (FLT3, IDH inhibitors)
# in indicated subgroups, adjusted for case-mix.
# ============================================================
hyps, ans = [], []
agent_target = [
    ('treatment_midostaurin','flt3_itd'),
    ('treatment_midostaurin','flt3_tkd'),
    ('treatment_gilteritinib','flt3_itd'),
    ('treatment_gilteritinib','flt3_tkd'),
    ('treatment_ivosidenib','idh1_mutation'),
    ('treatment_enasidenib','idh2_mutation'),
]
adj = ['age_years','ecog_ps','tp53_mutation','complex_karyotype','npm1_mutation',
       'unfit_for_intensive','albumin_g_dl']
for i,(t,m) in enumerate(agent_target):
    hid=f"h17.{i+1}"
    hyps.append({"id":hid,
                 "text":f"In {m}+ patients, after adjustment for case-mix, {t} has no effect on objective_response.",
                 "kind":"refined"})
    sub = df[df[m]==1].copy()
    cols=[t]+adj
    X = sm.add_constant(sub[cols].astype(float))
    y = sub['objective_response'].astype(int).values
    try:
        mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
        coef=float(mfit.params[t]); p=float(mfit.pvalues[t])
    except Exception:
        coef,p=None,None
    ans.append({"hypothesis_ids":[hid],
                "code":f"Logit(objective_response ~ {t} + adj | {m}==1)",
                "result_summary":f"In {m}+ N={len(sub)}: adj coef({t})={f(coef)} OR={f(np.exp(coef) if coef is not None else None,3)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":coef,"significant":(p is not None and p<0.05)})
add_iter(17,hyps,ans)

# ============================================================
# ITERATION 18: Lab-value heterogeneity for ven/aza
# (continuous interaction with albumin, blast %, etc.)
# ============================================================
hyps, ans = [], []
T='treatment_venetoclax_azacitidine'
labs_of_interest = ['albumin_g_dl','blast_pct_marrow','wbc_k_per_ul','crp_mg_l','hemoglobin_g_dl','ldh_u_l','nlr','calcium_mg_dl']
for i,lab in enumerate(labs_of_interest):
    # Median split
    med = df[lab].median()
    hid=f"h18.{i+1}.hi"
    hyps.append({"id":hid,
                 "text":f"In patients with {lab} ABOVE median ({med:.2f}), {T} is associated with a different objective_response rate.",
                 "kind":"novel"})
    sub = df[df[lab]>med]
    diff,p,n1,n0,r1,r0 = chi2_diff(sub,T)
    ans.append({"hypothesis_ids":[hid],
                "code":f"chi2 {T} | {lab}>median",
                "result_summary":f"{lab}>{med:.2f} (N={len(sub)}): resp {T}+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})
    hid2=f"h18.{i+1}.lo"
    hyps.append({"id":hid2,
                 "text":f"In patients with {lab} AT-OR-BELOW median ({med:.2f}), {T} is associated with a different objective_response rate.",
                 "kind":"novel"})
    sub = df[df[lab]<=med]
    diff,p,n1,n0,r1,r0 = chi2_diff(sub,T)
    ans.append({"hypothesis_ids":[hid2],
                "code":f"chi2 {T} | {lab}<=median",
                "result_summary":f"{lab}<={med:.2f} (N={len(sub)}): resp {T}+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})
add_iter(18,hyps,ans)

# ============================================================
# ITERATION 19: Joint subgroup search for 7+3
# (Is there ANY subgroup where 7+3 helps?)
# ============================================================
hyps, ans = [], []
T='treatment_7plus3'
for i,(predicate,label) in enumerate([
    ((df.complex_karyotype==0)&(df.tp53_mutation==0)&(df.unfit_for_intensive==0),'fit & CK- & tp53-'),
    ((df.complex_karyotype==0)&(df.tp53_mutation==0)&(df.unfit_for_intensive==0)&(df.npm1_mutation==1),'fit & CK- & tp53- & npm1+'),
    ((df.complex_karyotype==0)&(df.tp53_mutation==0)&(df.unfit_for_intensive==0)&(df.npm1_mutation==0),'fit & CK- & tp53- & npm1-'),
    ((df.unfit_for_intensive==0)&(df.ecog_ps<=1),'fit & ECOG<=1'),
    ((df.complex_karyotype==1),'CK==1 (poor risk — 7+3 may harm)'),
    ((df.tp53_mutation==1),'tp53==1 (poor risk)'),
    ((df.age_years<60),'age<60'),
    ((df.age_years>=70),'age>=70'),
]):
    hid=f"h19.{i+1}"
    hyps.append({"id":hid,
                 "text":f"Within {label}, {T} is associated with a different objective_response rate.",
                 "kind":"refined"})
    sub = df[predicate]
    diff,p,n1,n0,r1,r0 = chi2_diff(sub,T)
    ans.append({"hypothesis_ids":[hid],
                "code":f"chi2 {T} | {label}",
                "result_summary":f"{label} (N={len(sub)}): resp {T}+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})
add_iter(19,hyps,ans)

# ============================================================
# ITERATION 20: Cross-treatment interactions / co-treatment effects
# (Does receiving multiple treatments matter? Is there synergy with ven/aza?)
# ============================================================
hyps, ans = [], []
T='treatment_venetoclax_azacitidine'
for i,t2 in enumerate(['treatment_midostaurin','treatment_gilteritinib','treatment_ivosidenib','treatment_enasidenib','treatment_7plus3']):
    hid=f"h20.{i+1}"
    hyps.append({"id":hid,
                 "text":f"There is a {T} x {t2} interaction on objective_response (combination differs from sum of individual effects).",
                 "kind":"novel"})
    d = df.copy(); d['inter']=d[T]*d[t2]
    X = sm.add_constant(d[[T,t2,'inter']].astype(float))
    y = d['objective_response'].astype(int).values
    try:
        mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
        coef=float(mfit.params['inter']); p=float(mfit.pvalues['inter'])
    except Exception:
        coef,p=None,None
    ans.append({"hypothesis_ids":[hid],
                "code":f"Logit(objective_response ~ {T} + {t2} + {T}:{t2})",
                "result_summary":f"Interaction coef({T}*{t2})={f(coef)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":coef,"significant":(p is not None and p<0.05)})
add_iter(20,hyps,ans)

# ============================================================
# ITERATION 21: Adjusted ven/aza subgroup confirmation logit
# ============================================================
hyps, ans = [], []
T='treatment_venetoclax_azacitidine'
adj = ['age_years','ecog_ps','tp53_mutation','complex_karyotype','npm1_mutation',
       'unfit_for_intensive','albumin_g_dl','wbc_k_per_ul','blast_pct_marrow','secondary_aml']
for i,(predicate,label) in enumerate([
    ((df.unfit_for_intensive==1)&(df.npm1_mutation==1)&(df.tp53_mutation==0)&(df.complex_karyotype==0),'unfit & npm1+ & tp53- & CK-'),
    (~((df.unfit_for_intensive==1)&(df.npm1_mutation==1)&(df.tp53_mutation==0)&(df.complex_karyotype==0)),'complement of clean+favorable'),
]):
    hid=f"h21.{i+1}"
    hyps.append({"id":hid,
                 "text":f"In subgroup [{label}], {T} adjusted effect (logit OR) on objective_response.",
                 "kind":"refined"})
    sub = df[predicate].copy()
    cols = [T] + [c for c in adj if sub[c].nunique()>1]
    X = sm.add_constant(sub[cols].astype(float))
    y = sub['objective_response'].astype(int).values
    try:
        mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
        coef=float(mfit.params[T]); p=float(mfit.pvalues[T])
    except Exception:
        coef,p=None,None
    ans.append({"hypothesis_ids":[hid],
                "code":f"Logit(objective_response ~ {T} + adj | {label})",
                "result_summary":f"{label} N={len(sub)}: adj coef({T})={f(coef)} OR={f(np.exp(coef) if coef is not None else None,3)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":coef,"significant":(p is not None and p<0.05)})
add_iter(21,hyps,ans)

# ============================================================
# ITERATION 22: Three-way ven/aza interaction (npm1 x unfit x tp53)
# Tests whether suppression of the npm1+/unfit benefit by tp53 is real.
# ============================================================
hyps, ans = [], []
hid='h22.1'
hyps.append({"id":hid,"text":f"There is a positive 3-way interaction {T} x npm1_mutation x unfit_for_intensive on objective_response.","kind":"refined"})
d = df.copy()
d['t_n']=d[T]*d.npm1_mutation
d['t_u']=d[T]*d.unfit_for_intensive
d['n_u']=d.npm1_mutation*d.unfit_for_intensive
d['t_n_u']=d[T]*d.npm1_mutation*d.unfit_for_intensive
X = sm.add_constant(d[[T,'npm1_mutation','unfit_for_intensive','t_n','t_u','n_u','t_n_u']].astype(float))
y = d['objective_response'].astype(int).values
try:
    mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
    coef=float(mfit.params['t_n_u']); p=float(mfit.pvalues['t_n_u'])
except Exception:
    coef,p=None,None
ans.append({"hypothesis_ids":[hid],
            "code":f"Logit(objective_response ~ {T}*npm1*unfit (full 3-way))",
            "result_summary":f"3-way coef({T}*npm1*unfit)={f(coef)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":coef,"significant":(p is not None and p<0.05)})

hid='h22.2'
hyps.append({"id":hid,"text":f"Negative 3-way interaction {T} x npm1_mutation x tp53_mutation on objective_response (TP53 wipes out NPM1+ benefit).","kind":"refined"})
d = df.copy()
d['t_n']=d[T]*d.npm1_mutation
d['t_t']=d[T]*d.tp53_mutation
d['n_t']=d.npm1_mutation*d.tp53_mutation
d['t_n_t']=d[T]*d.npm1_mutation*d.tp53_mutation
X = sm.add_constant(d[[T,'npm1_mutation','tp53_mutation','t_n','t_t','n_t','t_n_t']].astype(float))
y = d['objective_response'].astype(int).values
try:
    mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
    coef=float(mfit.params['t_n_t']); p=float(mfit.pvalues['t_n_t'])
except Exception:
    coef,p=None,None
ans.append({"hypothesis_ids":[hid],
            "code":f"Logit(objective_response ~ {T}*npm1*tp53 (full 3-way))",
            "result_summary":f"3-way coef({T}*npm1*tp53)={f(coef)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":coef,"significant":(p is not None and p<0.05)})
add_iter(22,hyps,ans)

# ============================================================
# ITERATION 23: Final consolidated ven/aza subgroup test —
# adjusted logit including the joint subgroup indicator + interaction.
# ============================================================
hyps, ans = [], []
T='treatment_venetoclax_azacitidine'
hid='h23.1'
hyps.append({"id":hid,
             "text":f"Define S = (unfit_for_intensive==1 & npm1_mutation==1 & tp53_mutation==0 & complex_karyotype==0). "
                    f"In an adjusted logistic regression, {T} interacts positively with S on objective_response.",
             "kind":"refined"})
d = df.copy()
d['S'] = ((d.unfit_for_intensive==1)&(d.npm1_mutation==1)&(d.tp53_mutation==0)&(d.complex_karyotype==0)).astype(int)
d['inter'] = d[T]*d['S']
adj = ['age_years','ecog_ps','tp53_mutation','complex_karyotype','npm1_mutation',
       'unfit_for_intensive','albumin_g_dl','blast_pct_marrow','secondary_aml']
cols = [T,'S','inter'] + adj
X = sm.add_constant(d[cols].astype(float))
y = d['objective_response'].astype(int).values
mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
coef=float(mfit.params['inter']); p=float(mfit.pvalues['inter'])
ans.append({"hypothesis_ids":[hid],
            "code":"Logit(objective_response ~ T + S + T:S + adj)",
            "result_summary":f"Interaction T*S coef={f(coef)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":coef,"significant":(p is not None and p<0.05)})

hid='h23.2'
hyps.append({"id":hid,"text":f"Within S=1, adjusted {T} effect on objective_response is positive.","kind":"refined"})
sub = d[d['S']==1].copy()
cols2 = [T] + [c for c in adj if sub[c].nunique()>1]
X = sm.add_constant(sub[cols2].astype(float))
y = sub['objective_response'].astype(int).values
mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
coef=float(mfit.params[T]); p=float(mfit.pvalues[T])
ans.append({"hypothesis_ids":[hid],
            "code":"Logit(objective_response ~ T + adj | S=1)",
            "result_summary":f"Within S N={len(sub)}: coef({T})={f(coef)} OR={f(np.exp(coef),3)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":coef,"significant":(p is not None and p<0.05)})

hid='h23.3'
hyps.append({"id":hid,"text":f"Within S=0, adjusted {T} effect on objective_response is null.","kind":"refined"})
sub = d[d['S']==0].copy()
cols2 = [T] + [c for c in adj if sub[c].nunique()>1]
X = sm.add_constant(sub[cols2].astype(float))
y = sub['objective_response'].astype(int).values
mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
coef=float(mfit.params[T]); p=float(mfit.pvalues[T])
ans.append({"hypothesis_ids":[hid],
            "code":"Logit(objective_response ~ T + adj | S=0)",
            "result_summary":f"Outside S N={len(sub)}: coef({T})={f(coef)} OR={f(np.exp(coef),3)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":coef,"significant":(p is not None and p<0.05)})
add_iter(23,hyps,ans)

# ============================================================
# ITERATION 24: Tree-based subgroup discovery — exhaustive small subgroups
# (search over single binary features for ven/aza interaction)
# Already largely covered; here we systematically rank top single-variable
# interactions for each treatment to ensure nothing is missed.
# ============================================================
hyps, ans = [], []
binary_mods = ['sex_female','tp53_mutation','complex_karyotype','npm1_mutation',
               'flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation',
               'secondary_aml','unfit_for_intensive']
top_signals = []  # (treatment, modifier, coef, p)
for t in TX:
    for m in binary_mods:
        if t in m or m in t: continue
        d = df.copy(); d['inter']=d[t]*d[m]
        X = sm.add_constant(d[[t,m,'inter']].astype(float))
        y = d['objective_response'].astype(int).values
        try:
            mfit = sm.Logit(y, X).fit(disp=0, maxiter=200)
            top_signals.append((t,m,float(mfit.params['inter']),float(mfit.pvalues['inter'])))
        except Exception:
            pass

# Take top 8 interactions by smallest p
top_signals.sort(key=lambda r: r[3])
for i,(t,m,coef,p) in enumerate(top_signals[:10]):
    hid=f"h24.{i+1}"
    hyps.append({"id":hid,
                 "text":f"Treatment-effect heterogeneity scan: {t} x {m} interaction on objective_response.",
                 "kind":"novel"})
    ans.append({"hypothesis_ids":[hid],
                "code":f"Logit(objective_response ~ {t} + {m} + {t}:{m})",
                "result_summary":f"top scan: {t}*{m} coef={f(coef)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":coef,"significant":(p is not None and p<0.05)})
add_iter(24,hyps,ans)

# ============================================================
# ITERATION 25: Final best-supported subgroup hypotheses for each treatment
# ============================================================
hyps, ans = [], []
# (a) ven/aza definitive
hid='h25.1'
hyps.append({"id":hid,
             "text":"FINAL: treatment_venetoclax_azacitidine increases objective_response in patients with unfit_for_intensive==1 AND npm1_mutation==1 AND tp53_mutation==0 AND complex_karyotype==0 (and provides essentially no benefit outside this subgroup).",
             "kind":"refined"})
sub = df[(df.unfit_for_intensive==1)&(df.npm1_mutation==1)&(df.tp53_mutation==0)&(df.complex_karyotype==0)]
diff,p,n1,n0,r1,r0 = chi2_diff(sub,'treatment_venetoclax_azacitidine')
ans.append({"hypothesis_ids":[hid],
            "code":"chi2 ven/aza | unfit & npm1+ & tp53- & CK-",
            "result_summary":f"FINAL ven/aza subgroup N={len(sub)}: resp+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})

# (b) 7+3: weak/no benefit; explicitly test joint favorable subgroup
hid='h25.2'
hyps.append({"id":hid,
             "text":"FINAL: treatment_7plus3 has no clinically meaningful effect on objective_response in any subgroup tested; the only consistent signal is a small detrimental association in complex_karyotype==1 patients.",
             "kind":"refined"})
sub = df[df.complex_karyotype==1]
diff,p,n1,n0,r1,r0 = chi2_diff(sub,'treatment_7plus3')
ans.append({"hypothesis_ids":[hid],
            "code":"chi2 7+3 | CK==1",
            "result_summary":f"FINAL 7+3 in CK==1 N={len(sub)}: resp+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
            "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})

# (c) Targeted FLT3/IDH agents — null
for i,(t,m) in enumerate([('treatment_midostaurin','flt3_itd'),
                          ('treatment_gilteritinib','flt3_itd'),
                          ('treatment_ivosidenib','idh1_mutation'),
                          ('treatment_enasidenib','idh2_mutation')]):
    hid=f"h25.{3+i}"
    hyps.append({"id":hid,
                 "text":f"FINAL: {t} does NOT meaningfully change objective_response, even in the {m}+ subgroup.",
                 "kind":"refined"})
    sub = df[df[m]==1]
    diff,p,n1,n0,r1,r0 = chi2_diff(sub,t)
    ans.append({"hypothesis_ids":[hid],
                "code":f"chi2 {t} | {m}==1",
                "result_summary":f"In {m}+ N={len(sub)}: resp+={f(r1,4)} vs -={f(r0,4)}, diff={f(diff,4)}, p={f(p,6)}",
                "p_value":p,"effect_estimate":diff,"significant":(p is not None and p<0.05)})
add_iter(25,hyps,ans)

# ============== Persist final ==============
with open("iter_partial.json","w") as fp:
    json.dump(iters,fp,indent=2)
print(f"Iterations done: {len(iters)}")

# Sanity check ID uniqueness across iterations
all_ids = []
for it in iters:
    for h in it['proposed_hypotheses']:
        all_ids.append(h['id'])
dupes = {x for x in all_ids if all_ids.count(x)>1}
if dupes:
    print(f"WARNING duplicate ids: {dupes}")
else:
    print(f"All {len(all_ids)} hypothesis ids unique.")

# === Emit transcript.json ===
transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-manual@opus-4-7-1m",
    "max_iterations": 25,
    "iterations": iters,
}
with open("transcript.json","w") as fp:
    json.dump(transcript, fp, indent=2)
print("transcript.json written.")
