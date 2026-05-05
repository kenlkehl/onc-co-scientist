"""Iteration 6: Tree-based subgroup discovery for non-enzalutamide treatments.

For each non-enzalutamide treatment, fit a small decision tree on:
  X = features (no other treatments)  predicting (response | t=1) - (response | t=0)
Use a HTE proxy: fit logistic with treatment + treatment:feature splits.

Approach:
- For each treatment t, fit a tree predicting objective_response on subsets t==1 and t==0
  separately in the SAME leaves (T-learner style). Compare leaf rates.
- Also try a virtual-twin / S-learner: gradient boosting on (X, T) -> response,
  then estimate CATE = E[Y|X,T=1] - E[Y|X,T=0] for each patient.
- Find the leaf with largest positive CATE estimate and test the implied subgroup.
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
features = binary + contin
treatments = ['treatment_abiraterone','treatment_docetaxel',
              'treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab']

print("=== T-learner style with sklearn GBM, find max-CATE subgroup for each treatment ===")
try:
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError:
    print("sklearn not available")
    raise

X = df[features].values
y = df['objective_response'].values

for t in treatments:
    T = df[t].values
    # S-learner
    Xs = np.column_stack([X, T])
    m = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0)
    m.fit(Xs, y)
    p1 = m.predict_proba(np.column_stack([X, np.ones_like(T)]))[:,1]
    p0 = m.predict_proba(np.column_stack([X, np.zeros_like(T)]))[:,1]
    cate = p1 - p0
    print(f"\n--- {t} ---")
    print(f"  CATE distribution:  min={cate.min():.3f}  q25={np.quantile(cate,.25):.3f}  med={np.median(cate):.3f}  q75={np.quantile(cate,.75):.3f}  max={cate.max():.3f}  mean={cate.mean():.3f}")
    # Identify high-CATE subgroup (top 10%)
    thresh = np.quantile(cate, 0.90)
    mask = cate >= thresh
    sub = df[mask]
    on = sub.loc[sub[t]==1,'objective_response']
    off = sub.loc[sub[t]==0,'objective_response']
    if len(on) and len(off):
        print(f"  Top-10% CATE subgroup: n={mask.sum()}, on rate={on.mean():.3f} (n={len(on)})  off rate={off.mean():.3f} (n={len(off)})  diff={on.mean()-off.mean():+.3f}")
        # Mean of features in this group (binary) and median for continuous
        print("  Subgroup feature profile (binary prevalences):")
        for c in binary:
            full = df[c].mean(); sub_p = sub[c].mean()
            print(f"    {c}: full={full:.2f}  sub={sub_p:.2f}  ratio={sub_p/max(full,1e-9):.2f}")
    # Bottom 10% CATE
    thresh = np.quantile(cate, 0.10)
    mask = cate <= thresh
    sub = df[mask]
    on = sub.loc[sub[t]==1,'objective_response']
    off = sub.loc[sub[t]==0,'objective_response']
    if len(on) and len(off):
        print(f"  Bottom-10% CATE subgroup: n={mask.sum()}, on rate={on.mean():.3f} (n={len(on)})  off rate={off.mean():.3f} (n={len(off)})  diff={on.mean()-off.mean():+.3f}")
