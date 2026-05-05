"""Iteration 2: main effects of features on objective_response (univariate + adjusted)."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
y = df['objective_response'].values

binary = ['mcrpc','visceral_mets','brca2_mutation','ar_v7_positive','msi_high','psma_high']
contin = ['age_years','ecog_ps','psa_ng_ml','gleason_score','albumin_g_dl','ldh_u_l',
          'weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
          'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
          'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']
treatments = [c for c in df.columns if c.startswith('treatment_')]

print("=== Univariate (binary feature -> response) chi-square ===")
for c in binary:
    a = df.loc[df[c]==1,'objective_response'].mean()
    b = df.loc[df[c]==0,'objective_response'].mean()
    t = pd.crosstab(df[c], df['objective_response'])
    chi2, p, _, _ = stats.chi2_contingency(t)
    print(f"  {c}: on={a:.3f} off={b:.3f} diff={a-b:+.3f}  p={p:.3g}")

print("\n=== Univariate (continuous feature -> response) point-biserial / logistic ===")
for c in contin:
    # Logistic univariate
    X = sm.add_constant(df[[c]])
    res = sm.Logit(y, X).fit(disp=False)
    coef = res.params[c]; p = res.pvalues[c]
    # rates by quartile
    q = pd.qcut(df[c], 4, duplicates='drop')
    by = df.groupby(q, observed=True)['objective_response'].mean()
    print(f"  {c}: logit coef={coef:+.4f}  p={p:.3g}   quartile rates={by.round(3).tolist()}")

print("\n=== Adjusted: full logistic model (features + treatments) ===")
feats = binary + contin + treatments
X = sm.add_constant(df[feats])
res = sm.Logit(y, X).fit(disp=False, maxiter=200)
print(res.summary().tables[1])
