import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

print("=== ven+aza effect in: unfit & npm1+, by complex_karyotype, tp53 ===")
def cell(mask, name):
    sub = df[mask]
    a = sub.loc[sub['treatment_venetoclax_azacitidine']==1,'objective_response']
    b = sub.loc[sub['treatment_venetoclax_azacitidine']==0,'objective_response']
    if len(a)<10 or len(b)<10:
        return f"{name} too small (n_tx={len(a)}, n_ctrl={len(b)})"
    chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(sub['treatment_venetoclax_azacitidine'], sub['objective_response']))
    return f"{name}: tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}"

for tp in [0,1]:
    for ck in [0,1]:
        for npm in [0,1]:
            for u in [0,1]:
                mask = (df['unfit_for_intensive']==u)&(df['npm1_mutation']==npm)&(df['tp53_mutation']==tp)&(df['complex_karyotype']==ck)
                print(cell(mask, f"unfit={u}, npm1={npm}, tp53={tp}, ck={ck}"))

# Multivariable interaction model
print("\n=== Logit: ven+aza x npm1 x unfit x (tp53,ck) ===")
formula = ("objective_response ~ treatment_venetoclax_azacitidine * unfit_for_intensive * npm1_mutation"
           " + tp53_mutation + complex_karyotype + ecog_ps + age_years + secondary_aml")
m = smf.logit(formula, data=df).fit(disp=0)
print(m.summary().tables[1])

# Subgroup test in npm1+ unfit only by TP53 and CK
print("\n=== ven+aza in (unfit & npm1+) by tp53 ===")
sub = df[(df['unfit_for_intensive']==1)&(df['npm1_mutation']==1)]
print("Total subgroup n =", len(sub))
for tp in [0,1]:
    s2 = sub[sub['tp53_mutation']==tp]
    a = s2.loc[s2['treatment_venetoclax_azacitidine']==1,'objective_response']
    b = s2.loc[s2['treatment_venetoclax_azacitidine']==0,'objective_response']
    if len(a)>=10 and len(b)>=10:
        chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(s2['treatment_venetoclax_azacitidine'], s2['objective_response']))
        print(f"  tp53={tp}: tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")

print("\n=== ven+aza in (unfit & npm1+) by complex_karyotype ===")
for ck in [0,1]:
    s2 = sub[sub['complex_karyotype']==ck]
    a = s2.loc[s2['treatment_venetoclax_azacitidine']==1,'objective_response']
    b = s2.loc[s2['treatment_venetoclax_azacitidine']==0,'objective_response']
    if len(a)>=10 and len(b)>=10:
        chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(s2['treatment_venetoclax_azacitidine'], s2['objective_response']))
        print(f"  ck={ck}: tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")

# Final: cleanest subgroup = unfit & npm1+ & tp53- & ck-
print("\n=== Cleanest 'best response' subgroup ===")
mask = (df['unfit_for_intensive']==1)&(df['npm1_mutation']==1)&(df['tp53_mutation']==0)&(df['complex_karyotype']==0)
sub = df[mask]
a = sub.loc[sub['treatment_venetoclax_azacitidine']==1,'objective_response']
b = sub.loc[sub['treatment_venetoclax_azacitidine']==0,'objective_response']
chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(sub['treatment_venetoclax_azacitidine'], sub['objective_response']))
print(f"unfit&npm1+&tp53-&ck-: n={len(sub)} tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")

# The complement: in this subgroup but with tp53 or ck, do we lose the effect?
mask2 = (df['unfit_for_intensive']==1)&(df['npm1_mutation']==1)&((df['tp53_mutation']==1)|(df['complex_karyotype']==1))
sub = df[mask2]
a = sub.loc[sub['treatment_venetoclax_azacitidine']==1,'objective_response']
b = sub.loc[sub['treatment_venetoclax_azacitidine']==0,'objective_response']
if len(a)>=10 and len(b)>=10:
    chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(sub['treatment_venetoclax_azacitidine'], sub['objective_response']))
    print(f"unfit&npm1+&(tp53 or ck): n={len(sub)} tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")
