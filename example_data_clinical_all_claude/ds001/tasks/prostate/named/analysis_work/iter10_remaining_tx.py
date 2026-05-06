"""Iteration 10: Final search for hidden subgroup effects in docetaxel, lu177_psma, abiraterone."""
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

# Verify with two-stage tests: first run all interaction tests, then test specific predicate
def adj_interact(t, b_col_or_def, name):
    others = [c for c in treatments if c != t]
    feats = binary + contin + others + [t]
    d = df[feats + ['objective_response']].copy()
    if isinstance(b_col_or_def, str):
        d[name] = d[b_col_or_def]
    else:
        d[name] = b_col_or_def.astype(int).values
    d['t_x_b'] = d[t] * d[name]
    if name not in feats:
        cols = feats + [name, 't_x_b']
    else:
        cols = feats + ['t_x_b']
    X = sm.add_constant(d[cols])
    res = sm.Logit(d['objective_response'], X).fit(disp=False, maxiter=200)
    print(f"  {t} × {name}:  t_main={res.params[t]:+.4f} (p={res.pvalues[t]:.3g})  "
          f"intx={res.params['t_x_b']:+.4f} (p={res.pvalues['t_x_b']:.3g})  "
          f"t_in_b1={res.params[t]+res.params['t_x_b']:+.4f}")

print("=== Docetaxel: test of various candidate modifiers (adjusted) ===")
adj_interact('treatment_docetaxel', 'visceral_mets', 'visceral_mets')
adj_interact('treatment_docetaxel', 'mcrpc', 'mcrpc')
adj_interact('treatment_docetaxel', (df['ecog_ps']==2), 'ecog_2')
adj_interact('treatment_docetaxel', (df['ecog_ps']==0), 'ecog_0')
adj_interact('treatment_docetaxel', (df['psa_ng_ml']>df['psa_ng_ml'].median()), 'psa_high_med')
adj_interact('treatment_docetaxel', (df['ldh_u_l']>df['ldh_u_l'].median()), 'ldh_high')
adj_interact('treatment_docetaxel', (df['alkaline_phosphatase_u_l']>df['alkaline_phosphatase_u_l'].median()), 'alp_high')
adj_interact('treatment_docetaxel', (df['gleason_score']>=8), 'gleason_high')

print("\n=== Lu177-PSMA: candidate modifiers (adjusted) ===")
adj_interact('treatment_lu177_psma', 'psma_high', 'psma_high')
adj_interact('treatment_lu177_psma', 'visceral_mets', 'visceral_mets')
adj_interact('treatment_lu177_psma', 'mcrpc', 'mcrpc')
adj_interact('treatment_lu177_psma', (df['ecog_ps']==0), 'ecog_0')
adj_interact('treatment_lu177_psma', (df['psa_ng_ml']>df['psa_ng_ml'].median()), 'psa_high_med')
adj_interact('treatment_lu177_psma', (df['gleason_score']>=8), 'gleason_high')

print("\n=== Abiraterone: candidate modifiers (adjusted) ===")
adj_interact('treatment_abiraterone', 'mcrpc', 'mcrpc')
adj_interact('treatment_abiraterone', 'ar_v7_positive', 'ar_v7_positive')
adj_interact('treatment_abiraterone', 'brca2_mutation', 'brca2_mutation')
adj_interact('treatment_abiraterone', (df['psa_ng_ml']>df['psa_ng_ml'].median()), 'psa_high_med')
adj_interact('treatment_abiraterone', (df['ecog_ps']==0), 'ecog_0')

print("\n=== Olaparib: candidate modifiers (adjusted) ===")
adj_interact('treatment_olaparib', 'brca2_mutation', 'brca2_mutation')
adj_interact('treatment_olaparib', (df['psa_ng_ml']>df['psa_ng_ml'].median()), 'psa_high_med')
adj_interact('treatment_olaparib', (df['ecog_ps']==0), 'ecog_0')
adj_interact('treatment_olaparib', 'mcrpc', 'mcrpc')

print("\n=== Within enzalutamide non-responsive subgroup (mcrpc=1 OR ar_v7=1 OR brca2=1 OR msi=1), is anything active? ===")
non_resp = (df['mcrpc']==1)|(df['ar_v7_positive']==1)|(df['brca2_mutation']==1)|(df['msi_high']==1)
sub = df[non_resp]
print(f"n(non-responsive subgroup) = {len(sub)}")
for t in treatments:
    on = sub.loc[sub[t]==1,'objective_response']
    off = sub.loc[sub[t]==0,'objective_response']
    print(f"  {t}: on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}")

# Now test pembrolizumab × inflam_good within enzalutamide non-responsive
inflam_good = (sub['albumin_g_dl']>=df['albumin_g_dl'].median()) & (sub['crp_mg_l']<=df['crp_mg_l'].median())
print(f"\n=== Within non-resp + inflam_good=1 (n={inflam_good.sum()}) ===")
on = sub.loc[(sub['treatment_pembrolizumab']==1)&inflam_good,'objective_response']
off = sub.loc[(sub['treatment_pembrolizumab']==0)&inflam_good,'objective_response']
print(f"  pembro on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}")

# Pembro effect: also test within mcrpc=0
inflam_good_full = (df['albumin_g_dl']>=df['albumin_g_dl'].median()) & (df['crp_mg_l']<=df['crp_mg_l'].median())
print("\n=== Pembro effect by mcrpc within inflam_good ===")
for v in [0,1]:
    mask = inflam_good_full & (df['mcrpc']==v)
    on = df.loc[(df['treatment_pembrolizumab']==1)&mask,'objective_response']
    off = df.loc[(df['treatment_pembrolizumab']==0)&mask,'objective_response']
    print(f"  inflam_good=1 & mcrpc={v}: n={mask.sum()}, on={len(on)} ({on.mean():.3f})  off={len(off)} ({off.mean():.3f})  diff={on.mean()-off.mean():+.3f}")
