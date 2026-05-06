"""Iteration 9: Refine pembro subgroup further. Test (i) albumin alone, (ii) crp alone,
(iii) AND/OR variants, (iv) other binary modifiers."""
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

T = 'treatment_pembrolizumab'
print("=== Pembrolizumab × {single-marker subgroup} effects ===")
defns = [
    ('alb_high',         df['albumin_g_dl']>=df['albumin_g_dl'].median()),
    ('crp_low',          df['crp_mg_l']<=df['crp_mg_l'].median()),
    ('alb_high_AND_crp_low',
                         (df['albumin_g_dl']>=df['albumin_g_dl'].median())&(df['crp_mg_l']<=df['crp_mg_l'].median())),
    ('alb_high_OR_crp_low',
                         (df['albumin_g_dl']>=df['albumin_g_dl'].median())|(df['crp_mg_l']<=df['crp_mg_l'].median())),
    ('alb_top25',        df['albumin_g_dl']>=df['albumin_g_dl'].quantile(0.75)),
    ('crp_bot25',        df['crp_mg_l']<=df['crp_mg_l'].quantile(0.25)),
    ('alb_top25_AND_crp_bot25',
                         (df['albumin_g_dl']>=df['albumin_g_dl'].quantile(0.75))&(df['crp_mg_l']<=df['crp_mg_l'].quantile(0.25))),
]
for nm, mask in defns:
    on = df.loc[(df[T]==1)&mask,'objective_response']
    off = df.loc[(df[T]==0)&mask,'objective_response']
    on_c = df.loc[(df[T]==1)&~mask,'objective_response']
    off_c = df.loc[(df[T]==0)&~mask,'objective_response']
    if len(on)<10 or len(off)<10: continue
    print(f"\n  {nm} (n={mask.sum()}):  in: on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}")
    if len(on_c) and len(off_c):
        print(f"  {nm}=NO  (n={(~mask).sum()}):  on={len(on_c)} ({on_c.mean():.3f})  off={len(off_c)} ({off_c.mean():.3f})  diff={on_c.mean()-off_c.mean():+.3f}")

print("\n=== Within inflam_good (alb>=med & crp<=med), test pembro × additional binary modifiers ===")
inflam_good = (df['albumin_g_dl']>=df['albumin_g_dl'].median()) & (df['crp_mg_l']<=df['crp_mg_l'].median())
sub = df[inflam_good]
print(f"  inflam_good n={len(sub)}, pembro on rate={sub.loc[sub[T]==1,'objective_response'].mean():.3f}, off rate={sub.loc[sub[T]==0,'objective_response'].mean():.3f}")
for f in binary + ['ecog_ps']:
    for v in sorted(sub[f].unique()):
        s = sub[sub[f]==v]
        on = s.loc[s[T]==1,'objective_response']
        off = s.loc[s[T]==0,'objective_response']
        if len(on) >= 20 and len(off) >= 20:
            print(f"    {f}={v}: n={len(s)} on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}")

# Test if removing ar_v7_positive (which seems negative for pembro overall) refines things
print("\n=== inflam_good AND ar_v7=0 ===")
mask = inflam_good & (df['ar_v7_positive']==0)
on = df.loc[(df[T]==1)&mask,'objective_response']
off = df.loc[(df[T]==0)&mask,'objective_response']
print(f"  n={mask.sum()}, on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}")
chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(df.loc[mask,T], df.loc[mask,'objective_response']))
print(f"  chi2 p={p:.3g}")

print("\n=== inflam_good AND mcrpc=0 ===")
mask = inflam_good & (df['mcrpc']==0)
on = df.loc[(df[T]==1)&mask,'objective_response']
off = df.loc[(df[T]==0)&mask,'objective_response']
print(f"  n={mask.sum()}, on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}")

# Check if it's specifically related to crp / albumin or something else
print("\n=== Adjusted multivariable: pembro + interactions with continuous variables ===")
others = [c for c in treatments if c != T]
feats = binary + contin + others + [T]
d = df[feats + ['objective_response']].copy()
# Add inflam_good and interaction
d['inflam_good'] = inflam_good.astype(int).values
d['t_x_inflam'] = d[T]*d['inflam_good']
d['t_x_alb'] = d[T]*d['albumin_g_dl']
d['t_x_crp'] = d[T]*d['crp_mg_l']
X = sm.add_constant(d[feats + ['inflam_good','t_x_inflam','t_x_alb','t_x_crp']])
res = sm.Logit(d['objective_response'], X).fit(disp=False, maxiter=200)
print(f"  pembro main: {res.params[T]:+.4f} (p={res.pvalues[T]:.3g})")
print(f"  inflam_good: {res.params['inflam_good']:+.4f} (p={res.pvalues['inflam_good']:.3g})")
print(f"  t × inflam:  {res.params['t_x_inflam']:+.4f} (p={res.pvalues['t_x_inflam']:.3g})")
print(f"  t × albumin: {res.params['t_x_alb']:+.4f} (p={res.pvalues['t_x_alb']:.3g})")
print(f"  t × crp:     {res.params['t_x_crp']:+.4f} (p={res.pvalues['t_x_crp']:.3g})")

# Also test if pembrolizumab effect is confined to ECOG=0 or low ECOG
print("\n=== Pembro effect by ECOG ===")
for v in [0,1,2]:
    s = df[df['ecog_ps']==v]
    on = s.loc[s[T]==1,'objective_response']
    off = s.loc[s[T]==0,'objective_response']
    if len(on)>=10:
        print(f"  ecog_ps={v}: n={len(s)} on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}")

# inflam_good AND ecog=0
print("\n=== inflam_good AND ecog=0 ===")
mask = inflam_good & (df['ecog_ps']==0)
on = df.loc[(df[T]==1)&mask,'objective_response']
off = df.loc[(df[T]==0)&mask,'objective_response']
print(f"  n={mask.sum()}, on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}")
