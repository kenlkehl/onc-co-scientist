import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

# Multivariable adjusted main effects (control for confounders)
covars = "age_years + sex_female + ecog_ps + secondary_aml + unfit_for_intensive + complex_karyotype + flt3_itd + flt3_tkd + idh1_mutation + idh2_mutation + npm1_mutation + tp53_mutation + wbc_k_per_ul + blast_pct_marrow + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr"
tx_cols = ['treatment_midostaurin','treatment_gilteritinib','treatment_ivosidenib',
           'treatment_enasidenib','treatment_venetoclax_azacitidine','treatment_7plus3']

print("=== Multivariable model ===")
formula = "objective_response ~ " + " + ".join(tx_cols) + " + " + covars
m = smf.logit(formula, data=df).fit(disp=0)
print(m.summary().tables[1])

# Cofactors of unfit
print("\n=== Cofactors of unfit_for_intensive ===")
sub_unfit = df[df['unfit_for_intensive']==1]
print(f"unfit n={len(sub_unfit)}")
print("Mean age in unfit:", sub_unfit['age_years'].mean(), "vs fit:", df[df['unfit_for_intensive']==0]['age_years'].mean())
# ECOG distribution
print("ECOG in unfit:", sub_unfit['ecog_ps'].value_counts(normalize=True).sort_index().to_dict())
print("ECOG in fit:", df[df['unfit_for_intensive']==0]['ecog_ps'].value_counts(normalize=True).sort_index().to_dict())

# In unfit, look at ven+aza effect by various subgroups
print("\n=== Ven+aza effect within unfit, stratified by other features ===")
def strat_effect(modifier, sub=sub_unfit):
    rows = []
    for v in sorted(sub[modifier].unique()):
        s2 = sub[sub[modifier]==v]
        a = s2.loc[s2['treatment_venetoclax_azacitidine']==1,'objective_response']
        b = s2.loc[s2['treatment_venetoclax_azacitidine']==0,'objective_response']
        if len(a)<10 or len(b)<10:
            continue
        chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(s2['treatment_venetoclax_azacitidine'], s2['objective_response']))
        rows.append((v, len(a), a.mean(), len(b), b.mean(), a.mean()-b.mean(), p))
    return rows

for mod in ['ecog_ps','tp53_mutation','complex_karyotype','npm1_mutation','flt3_itd',
            'idh1_mutation','idh2_mutation','secondary_aml','sex_female']:
    print(f"\n--- modifier={mod} ---")
    for v,na,oa,nb,ob,d,p in strat_effect(mod):
        print(f"  {mod}={v}: tx_n={na} ORRtx={oa:.4f} ctrl_n={nb} ORRctrl={ob:.4f} diff={d:+.4f} p={p:.4g}")

# Three-way: ven+aza x unfit x tp53
print("\n=== 3-way: ven+aza x unfit_for_intensive (within tp53 strata) ===")
for tp in [0,1]:
    print(f"\n--- tp53={tp} ---")
    sub = df[df['tp53_mutation']==tp]
    for u in [0,1]:
        s2 = sub[sub['unfit_for_intensive']==u]
        a = s2.loc[s2['treatment_venetoclax_azacitidine']==1,'objective_response']
        b = s2.loc[s2['treatment_venetoclax_azacitidine']==0,'objective_response']
        if len(a)<10 or len(b)<10:
            continue
        chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(s2['treatment_venetoclax_azacitidine'], s2['objective_response']))
        print(f"  unfit={u}: tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")

print("\n=== 3-way: ven+aza x unfit x complex_karyotype ===")
for ck in [0,1]:
    print(f"\n--- complex_karyotype={ck} ---")
    sub = df[df['complex_karyotype']==ck]
    for u in [0,1]:
        s2 = sub[sub['unfit_for_intensive']==u]
        a = s2.loc[s2['treatment_venetoclax_azacitidine']==1,'objective_response']
        b = s2.loc[s2['treatment_venetoclax_azacitidine']==0,'objective_response']
        if len(a)<10 or len(b)<10:
            continue
        chi2,p,_,_ = stats.chi2_contingency(pd.crosstab(s2['treatment_venetoclax_azacitidine'], s2['objective_response']))
        print(f"  unfit={u}: tx_n={len(a)} ORRtx={a.mean():.4f} ctrl_n={len(b)} ORRctrl={b.mean():.4f} diff={a.mean()-b.mean():+.4f} p={p:.4g}")
