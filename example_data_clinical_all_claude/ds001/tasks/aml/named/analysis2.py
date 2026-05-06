import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

def stratified(treat, modifier, df=df):
    rows = []
    for m in [0,1]:
        sub = df[df[modifier]==m]
        a = sub.loc[sub[treat]==1, 'objective_response']
        b = sub.loc[sub[treat]==0, 'objective_response']
        if len(a)<10 or len(b)<10:
            rows.append((m, len(a), a.mean() if len(a) else np.nan, len(b), b.mean() if len(b) else np.nan, np.nan, np.nan))
            continue
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(sub[treat], sub['objective_response']))
        diff = a.mean()-b.mean()
        rows.append((m, len(a), a.mean(), len(b), b.mean(), diff, p))
    return rows

def logit_inter(treat, modifier, df=df, extra=None):
    formula = f"objective_response ~ {treat} * {modifier}"
    if extra:
        formula = formula + " + " + " + ".join(extra)
    m = smf.logit(formula, data=df).fit(disp=0)
    return m

print("=== FLT3 (itd) x midostaurin ===")
for m, na, oa, nb, ob, d, p in stratified('treatment_midostaurin','flt3_itd'):
    print(f"flt3_itd={m}: tx_n={na} ORRtx={oa:.4f} ctrl_n={nb} ORRctrl={ob:.4f} diff={d:+.4f} p={p:.4g}")
m = logit_inter('treatment_midostaurin','flt3_itd')
print(m.summary().tables[1])

print("\n=== FLT3 (tkd) x midostaurin ===")
for k, na, oa, nb, ob, d, p in stratified('treatment_midostaurin','flt3_tkd'):
    print(f"flt3_tkd={k}: tx_n={na} ORRtx={oa:.4f} ctrl_n={nb} ORRctrl={ob:.4f} diff={d:+.4f} p={p:.4g}")
m = logit_inter('treatment_midostaurin','flt3_tkd')
print(m.summary().tables[1])

print("\n=== FLT3 (itd) x gilteritinib ===")
for k, na, oa, nb, ob, d, p in stratified('treatment_gilteritinib','flt3_itd'):
    print(f"flt3_itd={k}: tx_n={na} ORRtx={oa:.4f} ctrl_n={nb} ORRctrl={ob:.4f} diff={d:+.4f} p={p:.4g}")
m = logit_inter('treatment_gilteritinib','flt3_itd')
print(m.summary().tables[1])

print("\n=== FLT3 (tkd) x gilteritinib ===")
for k, na, oa, nb, ob, d, p in stratified('treatment_gilteritinib','flt3_tkd'):
    print(f"flt3_tkd={k}: tx_n={na} ORRtx={oa:.4f} ctrl_n={nb} ORRctrl={ob:.4f} diff={d:+.4f} p={p:.4g}")
m = logit_inter('treatment_gilteritinib','flt3_tkd')
print(m.summary().tables[1])

print("\n=== IDH1 x ivosidenib ===")
for k, na, oa, nb, ob, d, p in stratified('treatment_ivosidenib','idh1_mutation'):
    print(f"idh1={k}: tx_n={na} ORRtx={oa:.4f} ctrl_n={nb} ORRctrl={ob:.4f} diff={d:+.4f} p={p:.4g}")
m = logit_inter('treatment_ivosidenib','idh1_mutation')
print(m.summary().tables[1])

print("\n=== IDH2 x enasidenib ===")
for k, na, oa, nb, ob, d, p in stratified('treatment_enasidenib','idh2_mutation'):
    print(f"idh2={k}: tx_n={na} ORRtx={oa:.4f} ctrl_n={nb} ORRctrl={ob:.4f} diff={d:+.4f} p={p:.4g}")
m = logit_inter('treatment_enasidenib','idh2_mutation')
print(m.summary().tables[1])

print("\n=== unfit_for_intensive x venetoclax_azacitidine ===")
for k, na, oa, nb, ob, d, p in stratified('treatment_venetoclax_azacitidine','unfit_for_intensive'):
    print(f"unfit={k}: tx_n={na} ORRtx={oa:.4f} ctrl_n={nb} ORRctrl={ob:.4f} diff={d:+.4f} p={p:.4g}")
m = logit_inter('treatment_venetoclax_azacitidine','unfit_for_intensive')
print(m.summary().tables[1])

print("\n=== unfit_for_intensive x 7+3 ===")
for k, na, oa, nb, ob, d, p in stratified('treatment_7plus3','unfit_for_intensive'):
    print(f"unfit={k}: tx_n={na} ORRtx={oa:.4f} ctrl_n={nb} ORRctrl={ob:.4f} diff={d:+.4f} p={p:.4g}")
m = logit_inter('treatment_7plus3','unfit_for_intensive')
print(m.summary().tables[1])
