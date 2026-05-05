import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

# enasidenib x sex stratified
print("=== enasidenib x sex_female ===")
for s in [0,1]:
    sub = df[df['sex_female']==s]
    a = sub.loc[sub['treatment_enasidenib']==1,'objective_response']
    b = sub.loc[sub['treatment_enasidenib']==0,'objective_response']
    chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(sub['treatment_enasidenib'], sub['objective_response']))
    print(f"sex_female={s}: tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")

# Among sex_female=1 with idh2+ does enasidenib work?
print("\n=== enasidenib in idh2+ x sex ===")
for s in [0,1]:
    sub = df[(df['sex_female']==s)&(df['idh2_mutation']==1)]
    a = sub.loc[sub['treatment_enasidenib']==1,'objective_response']
    b = sub.loc[sub['treatment_enasidenib']==0,'objective_response']
    if len(a)>=10 and len(b)>=10:
        chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(sub['treatment_enasidenib'], sub['objective_response']))
        print(f"sex_female={s}, idh2+: tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")

# ivosidenib x npm1
print("\n=== ivosidenib x npm1 ===")
for n in [0,1]:
    sub = df[df['npm1_mutation']==n]
    a = sub.loc[sub['treatment_ivosidenib']==1,'objective_response']
    b = sub.loc[sub['treatment_ivosidenib']==0,'objective_response']
    chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(sub['treatment_ivosidenib'], sub['objective_response']))
    print(f"npm1={n}: tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")

# Confirm ven+aza has NO effect outside the unfit&npm1+&tp53-&ck- core
print("\n=== ven+aza in non-(unfit&npm1+&tp53-&ck-) ===")
mask = ~((df['unfit_for_intensive']==1)&(df['npm1_mutation']==1)&(df['tp53_mutation']==0)&(df['complex_karyotype']==0))
sub = df[mask]
a = sub.loc[sub['treatment_venetoclax_azacitidine']==1,'objective_response']
b = sub.loc[sub['treatment_venetoclax_azacitidine']==0,'objective_response']
chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(sub['treatment_venetoclax_azacitidine'], sub['objective_response']))
print(f"complement: n={len(sub)} tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")

# Final 4-way logit
print("\n=== Final 4-way model ===")
formula = ("objective_response ~ treatment_venetoclax_azacitidine * unfit_for_intensive * npm1_mutation * (tp53_mutation + complex_karyotype)")
m = smf.logit(formula, data=df).fit(disp=0)
print(m.summary().tables[1])

# Sanity check: in unfit&npm1+&tp53-&ck- core: by ECOG?
print("\n=== Within ven+aza responder core, by ECOG ===")
core = df[(df['unfit_for_intensive']==1)&(df['npm1_mutation']==1)&(df['tp53_mutation']==0)&(df['complex_karyotype']==0)]
for v in sorted(core['ecog_ps'].unique()):
    s2 = core[core['ecog_ps']==v]
    a = s2.loc[s2['treatment_venetoclax_azacitidine']==1,'objective_response']
    b = s2.loc[s2['treatment_venetoclax_azacitidine']==0,'objective_response']
    if len(a)>=10 and len(b)>=10:
        chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(s2['treatment_venetoclax_azacitidine'], s2['objective_response']))
        print(f"ecog={v}: tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")

# Also confirm: in unfit_for_intensive=0 but npm1+, no effect
print("\n=== ven+aza in fit & npm1+ ===")
sub = df[(df['unfit_for_intensive']==0)&(df['npm1_mutation']==1)]
a = sub.loc[sub['treatment_venetoclax_azacitidine']==1,'objective_response']
b = sub.loc[sub['treatment_venetoclax_azacitidine']==0,'objective_response']
chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(sub['treatment_venetoclax_azacitidine'], sub['objective_response']))
print(f"fit&npm1+: n={len(sub)} tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")
