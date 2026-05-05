import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response']

bin_feats = ['feature_006','feature_013','feature_021','feature_015','feature_027','feature_005',
             'feature_008','feature_023','feature_011','feature_017','feature_019','feature_004']
cont_feats = ['feature_016','feature_022','feature_002','feature_018','feature_020','feature_024','feature_031',
              'feature_026','feature_009','feature_029','feature_003','feature_012','feature_025','feature_028',
              'feature_007','feature_014','feature_032']
ord_feats = ['feature_001','feature_010']

# 1) Multivariable logistic — main effects
allf = bin_feats + cont_feats + ord_feats
X = sm.add_constant(df[allf])
m = sm.Logit(y, X).fit(disp=0)
print("=== Main-effects logistic regression ===")
print(m.summary().tables[1])

# 2) Treatment-by-feature interaction screen with feature_008 as treatment
print("\n=== Interaction screen: feature_008 × X (logistic with main effects + product) ===")
results = []
for f in [c for c in allf if c != 'feature_008']:
    cols = ['feature_008', f]
    sub = df[cols].copy()
    sub['inter'] = sub['feature_008'] * sub[f]
    Xi = sm.add_constant(sub[['feature_008', f, 'inter']])
    try:
        mi = sm.Logit(y, Xi).fit(disp=0)
        beta_int = mi.params['inter']
        p_int = mi.pvalues['inter']
        # also subgroup response
        results.append((f, beta_int, p_int))
    except Exception as e:
        results.append((f, None, None))

results.sort(key=lambda r: (r[2] if r[2] is not None else 1))
for r in results:
    print(f"  feat={r[0]}: beta_int={r[1]:.4f}, p_int={r[2]:.3e}")

# 3) For all pairs of binary features, check whether feature_008 effect persists in subgroups
print("\n=== Subgroup response rates: stratify by combinations of negative-prognostic binaries ===")
for combo in [['feature_013'],['feature_015'],['feature_021'],
              ['feature_013','feature_015'],['feature_013','feature_021'],
              ['feature_015','feature_021'],['feature_013','feature_015','feature_021']]:
    print(f"\n  Subgroup defined by ALL of {combo} == 0:")
    mask = (df[combo].sum(axis=1) == 0)
    sub = df.loc[mask]
    rr0 = sub.loc[sub['feature_008']==0,'objective_response'].mean()
    rr1 = sub.loc[sub['feature_008']==1,'objective_response'].mean()
    n0 = (sub['feature_008']==0).sum()
    n1 = (sub['feature_008']==1).sum()
    diff = rr1-rr0
    # Fisher / chi-square
    ct = pd.crosstab(sub['feature_008'], sub['objective_response'])
    chi2,p,_,_ = stats.chi2_contingency(ct)
    print(f"    n={mask.sum()}, RR(8=0)={rr0:.3f} (n={n0}), RR(8=1)={rr1:.3f} (n={n1}), diff={diff:+.3f}, chi2 p={p:.2e}")

# 4) PSA (feature_022) cuts: does effect depend on PSA?
print("\n=== feature_008 effect by feature_022 quartile ===")
df['psa_q'] = pd.qcut(df['feature_022'], 4, labels=['Q1','Q2','Q3','Q4'])
print(df.groupby(['psa_q','feature_008'])['objective_response'].agg(['mean','count']))

# 5) Check if other "treatment-like" binaries have heterogeneous effects
print("\n=== Other binaries: marginal and within feature_008==1, check effects ===")
for t in ['feature_005','feature_023','feature_011','feature_017','feature_019','feature_004','feature_006']:
    overall_diff = df.loc[df[t]==1,'objective_response'].mean() - df.loc[df[t]==0,'objective_response'].mean()
    # within feature_008==1
    s = df.loc[df['feature_008']==1]
    diff_in_treat = s.loc[s[t]==1,'objective_response'].mean() - s.loc[s[t]==0,'objective_response'].mean()
    print(f"  {t}: overall diff={overall_diff:+.4f}, within feature_008==1 diff={diff_in_treat:+.4f}")
