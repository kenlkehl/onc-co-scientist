import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
print("N=", len(df), "ORR overall=", df['objective_response'].mean().round(4))

# 1) Univariable response rates by binary feature
bin_features = ['sex_female','secondary_aml','unfit_for_intensive','complex_karyotype',
                'flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation','npm1_mutation','tp53_mutation']
print("\n=== Binary feature ORR ===")
for c in bin_features:
    a = df.loc[df[c]==1,'objective_response']
    b = df.loc[df[c]==0,'objective_response']
    odds = (a.mean()/(1-a.mean())) / (b.mean()/(1-b.mean())) if a.mean()>0 and b.mean()>0 and a.mean()<1 and b.mean()<1 else float('nan')
    chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(df[c], df['objective_response']))
    print(f"{c}: ORR1={a.mean():.4f} (n={len(a)}) ORR0={b.mean():.4f} (n={len(b)}) diff={a.mean()-b.mean():+.4f} OR={odds:.3f} p={p:.4g}")

# 2) ECOG levels
print("\n=== ECOG ===")
for v in sorted(df['ecog_ps'].unique()):
    sub = df.loc[df['ecog_ps']==v,'objective_response']
    print(f"ecog={v}: ORR={sub.mean():.4f} (n={len(sub)})")
chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(df['ecog_ps'], df['objective_response']))
print(f"chi2 p={p:.4g}")

# 3) Treatment ORR
print("\n=== Treatments ===")
tx_cols = ['treatment_midostaurin','treatment_gilteritinib','treatment_ivosidenib',
           'treatment_enasidenib','treatment_venetoclax_azacitidine','treatment_7plus3']
for c in tx_cols:
    a = df.loc[df[c]==1,'objective_response']
    b = df.loc[df[c]==0,'objective_response']
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[c], df['objective_response']))
    odds = (a.mean()/(1-a.mean())) / (b.mean()/(1-b.mean())) if a.mean()>0 and b.mean()>0 and a.mean()<1 and b.mean()<1 else float('nan')
    print(f"{c}: ORR1={a.mean():.4f} (n={len(a)}) ORR0={b.mean():.4f} (n={len(b)}) diff={a.mean()-b.mean():+.4f} OR={odds:.3f} p={p:.4g}")

# 4) Continuous features t-test (response vs not)
cont_features = ['age_years','wbc_k_per_ul','blast_pct_marrow','albumin_g_dl','ldh_u_l',
                 'weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
                 'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
                 'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']
print("\n=== Continuous feature mean by response ===")
for c in cont_features:
    a = df.loc[df['objective_response']==1, c]
    b = df.loc[df['objective_response']==0, c]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"{c}: resp_mean={a.mean():.3f} nonresp_mean={b.mean():.3f} diff={a.mean()-b.mean():+.3f} p={p:.4g}")
