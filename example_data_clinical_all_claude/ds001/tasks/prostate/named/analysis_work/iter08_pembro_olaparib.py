"""Iteration 8: Refine pembro and olaparib subgroups; verify with adjusted models."""
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

print("=== Pembrolizumab × continuous-feature interaction (adjusted) ===")
y = df['objective_response'].values
others = [c for c in treatments if c != 'treatment_pembrolizumab']
for f in contin:
    feats = binary + contin + others + ['treatment_pembrolizumab']
    d = df[feats + ['objective_response']].copy()
    d['t_x_f'] = d['treatment_pembrolizumab']*d[f]
    X = sm.add_constant(d[feats + ['t_x_f']])
    try:
        res = sm.Logit(d['objective_response'], X).fit(disp=False, maxiter=200)
        tcoef = res.params['treatment_pembrolizumab']; tp = res.pvalues['treatment_pembrolizumab']
        ix = res.params['t_x_f']; ip = res.pvalues['t_x_f']
        print(f"  pembro x {f}: tMain={tcoef:+.3f} (p={tp:.3g})  intx={ix:+.5f} (p={ip:.3g})")
    except Exception as e:
        print(f"  pembro x {f}: ERROR")

print("\n=== Pembrolizumab effect within fine-grained inflammation subgroups ===")
df['inflam_bad'] = ((df['crp_mg_l']>df['crp_mg_l'].median()) | (df['albumin_g_dl']<df['albumin_g_dl'].median())).astype(int)
df['inflam_good'] = ((df['crp_mg_l']<=df['crp_mg_l'].median()) & (df['albumin_g_dl']>=df['albumin_g_dl'].median())).astype(int)
for v in [0,1]:
    sub = df[df['inflam_good']==v]
    on = sub.loc[sub['treatment_pembrolizumab']==1,'objective_response']
    off = sub.loc[sub['treatment_pembrolizumab']==0,'objective_response']
    if len(on) and len(off):
        try:
            tab = pd.crosstab(sub['treatment_pembrolizumab'], sub['objective_response'])
            chi2,p,_,_ = stats.chi2_contingency(tab)
        except: p = float('nan')
        print(f"  inflam_good={v}: n={len(sub)}, on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}  chi2_p={p:.3g}")

print("\n=== Adjusted: pembro × inflam_good interaction ===")
others = [c for c in treatments if c != 'treatment_pembrolizumab']
feats = binary + contin + others + ['treatment_pembrolizumab','inflam_good']
d = df[feats + ['objective_response']].copy()
d['t_x_g'] = d['treatment_pembrolizumab']*d['inflam_good']
X = sm.add_constant(d[feats + ['t_x_g']])
res = sm.Logit(d['objective_response'], X).fit(disp=False, maxiter=200)
print(f"  pembro main: {res.params['treatment_pembrolizumab']:+.4f} (p={res.pvalues['treatment_pembrolizumab']:.3g})")
print(f"  inflam_good main: {res.params['inflam_good']:+.4f} (p={res.pvalues['inflam_good']:.3g})")
print(f"  interaction: {res.params['t_x_g']:+.4f} (p={res.pvalues['t_x_g']:.3g})")
print(f"  pembro effect when inflam_good=1: {res.params['treatment_pembrolizumab']+res.params['t_x_g']:+.4f}")

print("\n=== Olaparib × brca2 (extra check): adjusted with all features ===")
others = [c for c in treatments if c != 'treatment_olaparib']
feats = binary + contin + others + ['treatment_olaparib']
d = df[feats + ['objective_response']].copy()
d['t_x_b'] = d['treatment_olaparib']*d['brca2_mutation']
X = sm.add_constant(d[feats + ['t_x_b']])
res = sm.Logit(d['objective_response'], X).fit(disp=False, maxiter=200)
print(f"  olaparib main: {res.params['treatment_olaparib']:+.4f} (p={res.pvalues['treatment_olaparib']:.3g})")
print(f"  brca2 main: {res.params['brca2_mutation']:+.4f} (p={res.pvalues['brca2_mutation']:.3g})")
print(f"  interaction: {res.params['t_x_b']:+.4f} (p={res.pvalues['t_x_b']:.3g})")
print(f"  olaparib effect when brca2=1: {res.params['treatment_olaparib']+res.params['t_x_b']:+.4f}")

print("\n=== Within msi_high=1 subgroup, pembro effect (n only ~1500) ===")
sub = df[df['msi_high']==1]
on = sub.loc[sub['treatment_pembrolizumab']==1,'objective_response']
off = sub.loc[sub['treatment_pembrolizumab']==0,'objective_response']
print(f"  n={len(sub)}, on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}")

# Within combined msi_high or brca2 (any DDR/immune-relevant marker)
for f in ['msi_high','brca2_mutation','ar_v7_positive','psma_high','visceral_mets']:
    for v in [0,1]:
        sub = df[df[f]==v]
        on = sub.loc[sub['treatment_pembrolizumab']==1,'objective_response']
        off = sub.loc[sub['treatment_pembrolizumab']==0,'objective_response']
        if len(on) >= 30 and len(off) >= 30:
            try:
                tab = pd.crosstab(sub['treatment_pembrolizumab'], sub['objective_response'])
                chi2,p,_,_ = stats.chi2_contingency(tab)
            except: p = float('nan')
            print(f"  pembro within {f}={v}: n={len(sub)}, on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}  p={p:.3g}")

print("\n=== Within enzalutamide-responsive subgroup (mcrpc=0 & ar_v7=0 & brca2=0 & msi=0): does pembro/other treatments ever help? ===")
sub = df[(df['mcrpc']==0)&(df['ar_v7_positive']==0)&(df['brca2_mutation']==0)&(df['msi_high']==0)].copy()
print(f"n={len(sub)}")
for t in treatments:
    on = sub.loc[sub[t]==1,'objective_response']
    off = sub.loc[sub[t]==0,'objective_response']
    print(f"  {t}: on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}")
