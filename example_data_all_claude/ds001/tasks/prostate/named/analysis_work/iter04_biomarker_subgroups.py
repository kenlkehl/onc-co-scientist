"""Iteration 4: Test classic biomarker-treatment matchups directly.

For each (treatment, biomarker) pair, compute response rates in the four cells:
- biomarker+, treatment+
- biomarker+, treatment-
- biomarker-, treatment+
- biomarker-, treatment-
And the within-biomarker treatment effect (rate difference + chi2 p).
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

pairs = [
    ('treatment_pembrolizumab','msi_high'),
    ('treatment_olaparib','brca2_mutation'),
    ('treatment_lu177_psma','psma_high'),
    ('treatment_enzalutamide','ar_v7_positive'),
    ('treatment_abiraterone','ar_v7_positive'),
    ('treatment_enzalutamide','mcrpc'),
    ('treatment_abiraterone','mcrpc'),
    ('treatment_docetaxel','visceral_mets'),
    ('treatment_docetaxel','mcrpc'),
    ('treatment_pembrolizumab','brca2_mutation'),
]

print("=== Treatment effect within biomarker subgroups ===\n")
for t,b in pairs:
    print(f"\n--- {t}  vs  {b} ---")
    for bv in [1, 0]:
        sub = df[df[b]==bv]
        on = sub.loc[sub[t]==1, 'objective_response']
        off = sub.loc[sub[t]==0, 'objective_response']
        rate_on = on.mean() if len(on) else float('nan')
        rate_off = off.mean() if len(off) else float('nan')
        diff = rate_on - rate_off
        # chi2
        try:
            tab = pd.crosstab(sub[t], sub['objective_response'])
            chi2, p, _, _ = stats.chi2_contingency(tab)
        except Exception:
            p = float('nan')
        print(f"  {b}={bv}: n={len(sub)},  on(n={len(on)}) rate={rate_on:.3f}  |  off(n={len(off)}) rate={rate_off:.3f}  diff={diff:+.3f}  chi2_p={p:.3g}")

print("\n=== Same, but adjusting for the full set of features (logistic with treatment*biomarker) ===")
binary = ['mcrpc','visceral_mets','brca2_mutation','ar_v7_positive','msi_high','psma_high']
contin = ['age_years','ecog_ps','psa_ng_ml','gleason_score','albumin_g_dl','ldh_u_l',
          'weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
          'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
          'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']
treatments = ['treatment_enzalutamide','treatment_abiraterone','treatment_docetaxel',
              'treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab']
for t,b in pairs:
    others = [c for c in treatments if c != t]
    feats = binary + contin + others + [t]
    d = df[feats + ['objective_response']].copy()
    d['t_x_b'] = d[t]*d[b]
    X = sm.add_constant(d[feats + ['t_x_b']])
    try:
        res = sm.Logit(d['objective_response'], X).fit(disp=False, maxiter=200)
        tcoef = res.params[t]; tp = res.pvalues[t]
        ix = res.params['t_x_b']; ip = res.pvalues['t_x_b']
        # treatment effect within b=1: tcoef + ix
        ix_term = tcoef + ix
        print(f"  {t} x {b}: t_main={tcoef:+.4f} (p={tp:.3g})  interaction={ix:+.4f} (p={ip:.3g})  t_in_b1={ix_term:+.4f}")
    except Exception as e:
        print(f"  {t} x {b}: ERROR {e}")
