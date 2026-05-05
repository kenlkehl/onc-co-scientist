"""Iteration 11: Final confirmatory tests of the proposed subgroup hypotheses.

For each treatment, test the best-supported subgroup definition with:
- Subgroup-restricted chi-square test
- Logistic regression with treatment, subgroup indicator, interaction, adjusted
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

binary = ['mcrpc','visceral_mets','brca2_mutation','ar_v7_positive','msi_high','psma_high']
contin = ['age_years','ecog_ps','psa_ng_ml','gleason_score','albumin_g_dl','ldh_u_l',
          'weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
          'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
          'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']
treatments = ['treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel',
              'treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab']

# ---- ENZA SUBGROUP ----
print("=== ENZALUTAMIDE responsive subgroup: mcrpc=0 & ar_v7=0 & brca2=0 & msi=0 ===")
enza_sg = ((df['mcrpc']==0)&(df['ar_v7_positive']==0)&(df['brca2_mutation']==0)&(df['msi_high']==0))
print(f"n(in)={enza_sg.sum()}, n(out)={(~enza_sg).sum()}")
for label, mask in [('IN', enza_sg), ('OUT', ~enza_sg)]:
    s = df[mask]
    on = s.loc[s['treatment_enzalutamide']==1,'objective_response']
    off = s.loc[s['treatment_enzalutamide']==0,'objective_response']
    chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(s['treatment_enzalutamide'], s['objective_response']))
    print(f"  {label}: n={len(s)}, on rate={on.mean():.4f}  off rate={off.mean():.4f}  diff={on.mean()-off.mean():+.4f}  p={p:.3g}")

# Adjusted: t × subgroup interaction
others = [c for c in treatments if c != 'treatment_enzalutamide']
feats = binary + contin + others + ['treatment_enzalutamide']
d = df[feats + ['objective_response']].copy()
d['enza_sg'] = enza_sg.astype(int).values
d['t_x_sg'] = d['treatment_enzalutamide']*d['enza_sg']
X = sm.add_constant(d[feats + ['enza_sg','t_x_sg']])
res = sm.Logit(d['objective_response'], X).fit(disp=False, maxiter=200)
print(f"  Adjusted: t_main={res.params['treatment_enzalutamide']:+.4f} (p={res.pvalues['treatment_enzalutamide']:.3g})")
print(f"  enza_sg: {res.params['enza_sg']:+.4f} (p={res.pvalues['enza_sg']:.3g})")
print(f"  t × enza_sg: {res.params['t_x_sg']:+.4f} (p={res.pvalues['t_x_sg']:.3g})")
print(f"  t effect when enza_sg=1: {res.params['treatment_enzalutamide']+res.params['t_x_sg']:+.4f}")

# Test that each of the 4 conditions is necessary: drop one at a time and see if effect persists
print("\n  Necessity check: drop each predicate and see if effect remains in 'almost responsive' subgroup")
preds = [('mcrpc',0),('ar_v7_positive',0),('brca2_mutation',0),('msi_high',0)]
for drop_idx in range(4):
    keep = [(p,v) for i,(p,v) in enumerate(preds) if i != drop_idx]
    drop = preds[drop_idx]
    # Patients matching all OTHER conditions but failing the dropped one
    mask = np.ones(len(df), bool)
    for (p,v) in keep:
        mask &= (df[p].values==v)
    mask &= (df[drop[0]].values != drop[1])  # dropped condition violated
    s = df[mask]
    on = s.loc[s['treatment_enzalutamide']==1,'objective_response']
    off = s.loc[s['treatment_enzalutamide']==0,'objective_response']
    if len(on)>=10 and len(off)>=10:
        diff = on.mean()-off.mean()
        print(f"    Drop {drop[0]}={drop[1]}, others-pass: n={len(s)}, on rate={on.mean():.3f}, off rate={off.mean():.3f}, diff={diff:+.4f}")

# ---- PEMBRO SUBGROUP ----
print("\n=== PEMBROLIZUMAB responsive subgroup: albumin>=median(3.8) & crp<=median(3.34) ===")
pembro_sg = (df['albumin_g_dl']>=df['albumin_g_dl'].median()) & (df['crp_mg_l']<=df['crp_mg_l'].median())
print(f"n(in)={pembro_sg.sum()}, n(out)={(~pembro_sg).sum()}")
for label, mask in [('IN', pembro_sg), ('OUT', ~pembro_sg)]:
    s = df[mask]
    on = s.loc[s['treatment_pembrolizumab']==1,'objective_response']
    off = s.loc[s['treatment_pembrolizumab']==0,'objective_response']
    chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(s['treatment_pembrolizumab'], s['objective_response']))
    print(f"  {label}: n={len(s)}, on rate={on.mean():.4f}  off rate={off.mean():.4f}  diff={on.mean()-off.mean():+.4f}  p={p:.3g}")

others = [c for c in treatments if c != 'treatment_pembrolizumab']
feats = binary + contin + others + ['treatment_pembrolizumab']
d = df[feats + ['objective_response']].copy()
d['pembro_sg'] = pembro_sg.astype(int).values
d['t_x_sg'] = d['treatment_pembrolizumab']*d['pembro_sg']
X = sm.add_constant(d[feats + ['pembro_sg','t_x_sg']])
res = sm.Logit(d['objective_response'], X).fit(disp=False, maxiter=200)
print(f"  Adjusted: t_main={res.params['treatment_pembrolizumab']:+.4f} (p={res.pvalues['treatment_pembrolizumab']:.3g})")
print(f"  pembro_sg: {res.params['pembro_sg']:+.4f} (p={res.pvalues['pembro_sg']:.3g})")
print(f"  t × pembro_sg: {res.params['t_x_sg']:+.4f} (p={res.pvalues['t_x_sg']:.3g})")
print(f"  t effect when pembro_sg=1: {res.params['treatment_pembrolizumab']+res.params['t_x_sg']:+.4f}")

# Necessity check for pembro
print("\n  Necessity check: pembro effect by single-marker subgroups")
for mname, mdef in [('alb_high_only', (df['albumin_g_dl']>=df['albumin_g_dl'].median())&(df['crp_mg_l']>df['crp_mg_l'].median())),
                    ('crp_low_only', (df['albumin_g_dl']<df['albumin_g_dl'].median())&(df['crp_mg_l']<=df['crp_mg_l'].median())),
                    ('both_bad', (df['albumin_g_dl']<df['albumin_g_dl'].median())&(df['crp_mg_l']>df['crp_mg_l'].median()))]:
    s = df[mdef]
    on = s.loc[s['treatment_pembrolizumab']==1,'objective_response']
    off = s.loc[s['treatment_pembrolizumab']==0,'objective_response']
    if len(on)>=10 and len(off)>=10:
        diff = on.mean()-off.mean()
        print(f"    {mname}: n={len(s)}, on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={diff:+.4f}")

# ---- Confirm null hypotheses for other treatments ----
print("\n=== Confirm null treatment effects (overall and adjusted) ===")
for t in ['treatment_abiraterone','treatment_docetaxel']:
    others = [c for c in treatments if c != t]
    feats = binary + contin + others + [t]
    d = df[feats + ['objective_response']].copy()
    X = sm.add_constant(d[feats])
    res = sm.Logit(d['objective_response'], X).fit(disp=False, maxiter=200)
    print(f"  {t}: adj logit coef={res.params[t]:+.4f}  p={res.pvalues[t]:.3g}")

# ---- Olaparib × brca2 final ----
print("\n=== OLAPARIB × BRCA2 (negative interaction surprise) ===")
sub = df[df['brca2_mutation']==1]
on = sub.loc[sub['treatment_olaparib']==1,'objective_response']
off = sub.loc[sub['treatment_olaparib']==0,'objective_response']
chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(sub['treatment_olaparib'], sub['objective_response']))
print(f"  brca2=1: n={len(sub)}, olaparib on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.4f}  p={p:.3g}")

# Lu177 × visceral_mets
print("\n=== LU177-PSMA × VISCERAL METS ===")
for v in [0,1]:
    s = df[df['visceral_mets']==v]
    on = s.loc[s['treatment_lu177_psma']==1,'objective_response']
    off = s.loc[s['treatment_lu177_psma']==0,'objective_response']
    chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(s['treatment_lu177_psma'], s['objective_response']))
    print(f"  visceral_mets={v}: n={len(s)}, lu177 on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.4f}  p={p:.3g}")

# Abiraterone × brca2 (positive interaction)
print("\n=== ABIRATERONE × BRCA2 ===")
for v in [0,1]:
    s = df[df['brca2_mutation']==v]
    on = s.loc[s['treatment_abiraterone']==1,'objective_response']
    off = s.loc[s['treatment_abiraterone']==0,'objective_response']
    chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(s['treatment_abiraterone'], s['objective_response']))
    print(f"  brca2={v}: n={len(s)}, abi on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.4f}  p={p:.3g}")
