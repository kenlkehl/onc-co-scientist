import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response']

# Final subgroup hypothesis: feature_008 effect maximal where 013=0, 015=0, 021=0, 027=0, low PSA
# Try adding 027 to the "good prognosis" mask
print("=== Add feature_027 to suppressor list ===")
for combo in [['feature_013','feature_015','feature_021'],
              ['feature_013','feature_015','feature_021','feature_027']]:
    mask = (df[combo].sum(axis=1) == 0)
    sub = df.loc[mask]
    rr0 = sub.loc[sub['feature_008']==0,'objective_response'].mean()
    rr1 = sub.loc[sub['feature_008']==1,'objective_response'].mean()
    n0 = (sub['feature_008']==0).sum()
    n1 = (sub['feature_008']==1).sum()
    ct = pd.crosstab(sub['feature_008'], sub['objective_response'])
    chi2,p,_,_ = stats.chi2_contingency(ct)
    print(f"  {combo} all zero: n={mask.sum()}, RR(8=0)={rr0:.3f} (n={n0}), RR(8=1)={rr1:.3f} (n={n1}), diff={rr1-rr0:+.3f}, p={p:.2e}")

# Now also condition on low PSA (< median)
print("\n=== Best subgroup with PSA cuts ===")
for psa_thr in [df['feature_022'].quantile(q) for q in [0.25,0.5,0.75]]:
    mask = (df[['feature_013','feature_015','feature_021','feature_027']].sum(axis=1) == 0) & (df['feature_022'] < psa_thr)
    sub = df.loc[mask]
    if len(sub) < 50: continue
    rr0 = sub.loc[sub['feature_008']==0,'objective_response'].mean()
    rr1 = sub.loc[sub['feature_008']==1,'objective_response'].mean()
    n0 = (sub['feature_008']==0).sum()
    n1 = (sub['feature_008']==1).sum()
    print(f"  + feature_022<{psa_thr:.1f}: n={mask.sum()}, RR(8=0)={rr0:.3f} (n={n0}), RR(8=1)={rr1:.3f} (n={n1}), diff={rr1-rr0:+.3f}")

# Now the converse: in the BAD subgroup (any of 013/015/021/027 == 1), does treatment have ANY effect?
print("\n=== Treatment effect within BAD subgroups ===")
for f in ['feature_013','feature_015','feature_021','feature_027']:
    sub = df.loc[df[f] == 1]
    rr0 = sub.loc[sub['feature_008']==0,'objective_response'].mean()
    rr1 = sub.loc[sub['feature_008']==1,'objective_response'].mean()
    n0 = (sub['feature_008']==0).sum()
    n1 = (sub['feature_008']==1).sum()
    diff = rr1-rr0
    if n0 > 0 and n1 > 0:
        ct = pd.crosstab(sub['feature_008'], sub['objective_response'])
        chi2,p,_,_ = stats.chi2_contingency(ct)
        print(f"  {f}=1: n={len(sub)}, RR(8=0)={rr0:.3f} (n={n0}), RR(8=1)={rr1:.3f} (n={n1}), diff={diff:+.3f}, p={p:.2e}")

# Joint logistic with all interactions
print("\n=== Joint logistic with all 4 binary interactions ===")
df2 = df.copy()
for f in ['feature_013','feature_015','feature_021','feature_027']:
    df2[f'i_{f}'] = df2['feature_008'] * df2[f]
cols = ['feature_008','feature_013','feature_015','feature_021','feature_027',
        'i_feature_013','i_feature_015','i_feature_021','i_feature_027',
        'feature_022','feature_001','feature_020','feature_002','feature_024']
Xj = sm.add_constant(df2[cols])
mj = sm.Logit(y, Xj).fit(disp=0)
print(mj.summary().tables[1])

# Confirm direction: feature_002 (albumin?) higher → better
print("\n=== feature_002 quartiles vs response ===")
df2['alb_q'] = pd.qcut(df['feature_002'], 4)
print(df2.groupby('alb_q')['objective_response'].agg(['mean','count']))

# feature_020 vs response
print("\n=== feature_020 (zero vs >0) ===")
print(df2.groupby(df2['feature_020'] > 0)['objective_response'].agg(['mean','count']))

# feature_022 (PSA) interaction with feature_008 — formal continuous interaction
print("\n=== Continuous PSA × feature_008 interaction ===")
sub = df.copy()
sub['log_psa'] = np.log1p(sub['feature_022'])
sub['inter'] = sub['feature_008'] * sub['log_psa']
Xc = sm.add_constant(sub[['feature_008','log_psa','inter']])
mc = sm.Logit(y, Xc).fit(disp=0)
print(mc.summary().tables[1])

# Test: does adding low-PSA to subgroup further improve? (sensitivity)
print("\n=== Final 'good' subgroup definition ===")
good = (df['feature_013']==0) & (df['feature_015']==0) & (df['feature_021']==0) & (df['feature_027']==0)
g = df.loc[good]
print(f"  Good subgroup n={good.sum()} ({good.mean()*100:.1f}%)")
print(f"  Within good: RR(8=0)={g.loc[g['feature_008']==0,'objective_response'].mean():.3f} (n={(g['feature_008']==0).sum()})")
print(f"  Within good: RR(8=1)={g.loc[g['feature_008']==1,'objective_response'].mean():.3f} (n={(g['feature_008']==1).sum()})")
bad = ~good
b = df.loc[bad]
print(f"  Bad subgroup n={bad.sum()} ({bad.mean()*100:.1f}%)")
print(f"  Within bad: RR(8=0)={b.loc[b['feature_008']==0,'objective_response'].mean():.3f}")
print(f"  Within bad: RR(8=1)={b.loc[b['feature_008']==1,'objective_response'].mean():.3f}")

# One last check: do feature_020, feature_024 also modify the treatment?
print("\n=== feature_020 and feature_024 interactions, restricted to good subgroup ===")
for f in ['feature_020','feature_024','feature_022']:
    g2 = df.loc[good].copy()
    g2['inter'] = g2['feature_008'] * g2[f]
    Xg = sm.add_constant(g2[['feature_008',f,'inter']])
    mg = sm.Logit(g2['objective_response'], Xg).fit(disp=0)
    print(f"  in good subgroup, feature_008 × {f}: beta_int={mg.params['inter']:.4f}, p={mg.pvalues['inter']:.2e}")
