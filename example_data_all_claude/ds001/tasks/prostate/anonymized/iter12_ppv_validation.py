"""Iteration 12-15: Recoded 'any modifier ON' check, half-sample validation,
NNT/PPV calculation, and stress test of the joint subgroup definition.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
T = 'feature_008'
modifiers = ['feature_013', 'feature_015', 'feature_021', 'feature_027']

# ===== Recoded indicator: ANY modifier ON =====
df['any_modifier'] = (df[modifiers].sum(axis=1) > 0).astype(int)
print('=== ANY modifier ON: prevalence and ORR ===')
print(f'Prevalence of any_modifier=1: {df["any_modifier"].mean():.4f}')
ct = pd.crosstab(df['any_modifier'], df['objective_response'])
print(ct)
print()

# Single-modifier interaction model
print('=== Logistic: T x any_modifier ===')
X = df[[T, 'any_modifier']].copy()
X['Tx'] = X[T] * X['any_modifier']
X = sm.add_constant(X.astype(float))
m = sm.Logit(df['objective_response'].astype(int), X).fit(disp=False)
or_table = pd.DataFrame({'beta': m.params, 'OR': np.exp(m.params),
                         'OR_LCL': np.exp(m.conf_int()[0]),
                         'OR_UCL': np.exp(m.conf_int()[1]),
                         'p': m.pvalues}).round(4)
print(or_table)
print()
# OR(T) when any_modifier=0
or_t_off = np.exp(m.params[T])
or_t_on = np.exp(m.params[T] + m.params['Tx'])
print(f'OR(T) | any_modifier=0: {or_t_off:.3f}')
print(f'OR(T) | any_modifier=1: {or_t_on:.3f}')
print()

# Compare model fit: 4-modifier saturated vs single any_modifier
print('=== Comparing models: T*any_modifier vs T*4 individual modifiers ===')
# Already have m_red = T + 4 modifiers + 4 T*modifier
X_full = df[[T] + modifiers].copy()
for mm in modifiers:
    X_full[f'T_x_{mm}'] = X_full[T] * X_full[mm]
X_full = sm.add_constant(X_full.astype(float))
m_full = sm.Logit(df['objective_response'].astype(int), X_full).fit(disp=False)
print(f'  T + 4 modifiers + 4 T*mod (df=9): LL={m_full.llf:.2f}')
print(f'  T + any_modifier + T*any (df=3): LL={m.llf:.2f}')
lrt = 2*(m_full.llf - m.llf)
print(f'  LRT (additive 4-modifier vs any-modifier collapsed): chi2={lrt:.2f}, df=6, p={stats.chi2.sf(lrt, 6):.3e}')
print()

# ===== Half-sample validation =====
print('=== Half-sample validation ===')
rng = np.random.default_rng(123)
idx = np.arange(len(df))
rng.shuffle(idx)
half = idx[:len(df)//2]; rest = idx[len(df)//2:]
for label, ii in [('half A', half), ('half B', rest)]:
    sub = df.iloc[ii]
    fav_sub = ((sub['feature_013']==0)&(sub['feature_015']==0)&
               (sub['feature_021']==0)&(sub['feature_027']==0))
    ssub = sub[fav_sub]
    p0 = ssub.loc[ssub[T]==0,'objective_response'].mean()
    p1 = ssub.loc[ssub[T]==1,'objective_response'].mean()
    print(f'{label}: n_responsive_subgroup={fav_sub.sum()}, ORR T0={p0:.4f}, T1={p1:.4f}, RD=+{p1-p0:.4f}')
    # And in the unfavorable
    sub_unfav = sub[~fav_sub]
    p0u = sub_unfav.loc[sub_unfav[T]==0,'objective_response'].mean()
    p1u = sub_unfav.loc[sub_unfav[T]==1,'objective_response'].mean()
    print(f'         non-responsive: n={(~fav_sub).sum()}, ORR T0={p0u:.4f}, T1={p1u:.4f}, RD=+{p1u-p0u:+.4f}')
print()

# ===== NNT, PPV =====
print('=== Cohort-level NNT and PPV using "all 4 modifiers off" rule ===')
fav = ((df['feature_013']==0)&(df['feature_015']==0)&
       (df['feature_021']==0)&(df['feature_027']==0))
sub = df[fav]
p0 = sub.loc[sub[T]==0,'objective_response'].mean()
p1 = sub.loc[sub[T]==1,'objective_response'].mean()
nnt = 1 / (p1 - p0)
print(f'In responsive subgroup: NNT to gain one ORR = {nnt:.2f}')
sub_unfav = df[~fav]
p0u = sub_unfav.loc[sub_unfav[T]==0,'objective_response'].mean()
p1u = sub_unfav.loc[sub_unfav[T]==1,'objective_response'].mean()
print(f'In non-responsive subgroup: T0 ORR={p0u:.4f}, T1 ORR={p1u:.4f}, RD={p1u-p0u:+.4f}')

# Confusion-style stats: among treated patients, the rule is "treat only if all 4 modifiers off".
treated = df[df[T]==1].copy()
treated['rule_says_treat'] = ((treated['feature_013']==0)&(treated['feature_015']==0)&
                              (treated['feature_021']==0)&(treated['feature_027']==0))
print(f'\nAmong treated (n={len(treated)}):')
print(f'  Predicted to benefit (all 4 OFF): n={treated["rule_says_treat"].sum()} ({treated["rule_says_treat"].mean()*100:.1f}%)')
print(f'    ORR among predicted-benefit: {treated.loc[treated["rule_says_treat"],"objective_response"].mean():.4f}')
print(f'    ORR among predicted-no-benefit: {treated.loc[~treated["rule_says_treat"],"objective_response"].mean():.4f}')
