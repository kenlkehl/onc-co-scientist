"""Iteration 3: systematic treatment x feature interaction screen.

For each treatment t and each feature f, fit logistic regression:
  response ~ const + t + f + t:f  (and adjusted version including baseline covariates)
Report treatment main effect, feature main effect, interaction coefficient + p-value.
"""
import pandas as pd
import numpy as np
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
all_features = binary + contin
treatments = ['treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel',
              'treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab']

results = []

for t in treatments:
    for f in all_features:
        # Unadjusted t x f interaction
        d = df[[t, f, 'objective_response']].copy()
        d['t_x_f'] = d[t] * d[f]
        X = sm.add_constant(d[[t, f, 't_x_f']])
        try:
            res = sm.Logit(d['objective_response'], X).fit(disp=False, maxiter=200)
            t_main = res.params[t]; tp = res.pvalues[t]
            f_main = res.params[f]; fp = res.pvalues[f]
            ix = res.params['t_x_f']; ip = res.pvalues['t_x_f']
            results.append({'treatment': t, 'feature': f,
                           't_main': t_main, 't_p': tp,
                           'f_main': f_main, 'f_p': fp,
                           'interaction': ix, 'i_p': ip})
        except Exception as e:
            results.append({'treatment': t, 'feature': f,
                           't_main': None, 't_p': None,
                           'f_main': None, 'f_p': None,
                           'interaction': None, 'i_p': None})

R = pd.DataFrame(results)

print("=== Top interactions by p-value (across all treatments x features) ===")
top = R.sort_values('i_p').head(40)
for _, row in top.iterrows():
    print(f"  {row.treatment:25s} x {row.feature:25s}  beta_int={row.interaction:+8.4f} p={row.i_p:.3g}  | tMain={row.t_main:+.3f} (p={row.t_p:.2g})")

print("\n=== Per-treatment top-3 interactions (smallest p) ===")
for t in treatments:
    sub = R[R.treatment==t].sort_values('i_p').head(5)
    print(f"\n  {t}:")
    for _, row in sub.iterrows():
        print(f"    x {row.feature:25s}  beta_int={row.interaction:+8.4f} p={row.i_p:.3g}  tMain={row.t_main:+.3f} (p={row.t_p:.2g})")

# Save full table
R.to_csv('analysis_work/iter03_interactions.csv', index=False)
print(f"\nFull table written to analysis_work/iter03_interactions.csv (n={len(R)})")
