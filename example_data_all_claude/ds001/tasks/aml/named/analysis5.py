import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

bin_features = ['sex_female','secondary_aml','unfit_for_intensive','complex_karyotype',
                'flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation','npm1_mutation','tp53_mutation']
tx_cols = ['treatment_midostaurin','treatment_gilteritinib','treatment_ivosidenib',
           'treatment_enasidenib','treatment_venetoclax_azacitidine','treatment_7plus3']

print("=== Systematic interaction screen: each treatment x each binary feature ===")
print(f"{'treatment':<35} {'modifier':<25} {'beta_int':<10} {'p_int':<10}")
results = []
for tx in tx_cols:
    for mod in bin_features:
        try:
            m = smf.logit(f"objective_response ~ {tx} * {mod}", data=df).fit(disp=0)
            term = f"{tx}:{mod}"
            if term in m.params.index:
                beta = m.params[term]
                p = m.pvalues[term]
            else:
                beta = float('nan'); p = float('nan')
            results.append((tx, mod, beta, p))
            print(f"{tx:<35} {mod:<25} {beta:+.4f}    {p:.4g}")
        except Exception as e:
            print(f"{tx} x {mod}: error {e}")

# Also screen ECOG (treated as ordinal)
print("\n=== Treatment x ecog_ps interaction ===")
for tx in tx_cols:
    m = smf.logit(f"objective_response ~ {tx} * ecog_ps", data=df).fit(disp=0)
    term = f"{tx}:ecog_ps"
    print(f"{tx} x ecog_ps: beta={m.params[term]:+.4f} p={m.pvalues[term]:.4g}")

# Continuous features - z-score each then interaction
print("\n=== Treatment x continuous-feature interaction (z-scored) ===")
cont_features = ['age_years','wbc_k_per_ul','blast_pct_marrow','albumin_g_dl','ldh_u_l',
                 'weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
                 'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
                 'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']
df2 = df.copy()
for c in cont_features:
    df2[c+'_z'] = (df2[c]-df2[c].mean())/df2[c].std()
for tx in tx_cols:
    for c in cont_features:
        m = smf.logit(f"objective_response ~ {tx} * {c}_z", data=df2).fit(disp=0)
        term = f"{tx}:{c}_z"
        beta = m.params[term]; p = m.pvalues[term]
        if p < 0.05:
            print(f"{tx} x {c}_z: beta={beta:+.4f} p={p:.4g} *** SIG ***")
