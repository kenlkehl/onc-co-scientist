"""Additional checks: confirm features that look like 'no benefit' patterns."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].astype(int).values
T = df['feature_008'].astype(int).values
fav = ((df['feature_013']==0)&(df['feature_015']==0)&(df['feature_021']==0)&(df['feature_027']==0)).values

# Conditional model: log(odds(y)) = a + b1*T + b2*fav + b3*T*fav + adj covariates
# This nests within multivariable framework
print('=== Adjusted treatment-effect heterogeneity model:')
feature_cols = [c for c in df.columns if c.startswith('feature_') and c not in ['feature_030','feature_008','feature_013','feature_015','feature_021','feature_027']]
X = df[feature_cols].astype(float).copy()
for c in feature_cols:
    if X[c].nunique() > 5:
        X[c] = (X[c]-X[c].mean())/X[c].std()
X['T'] = T
X['fav'] = fav.astype(int)
X['T_fav'] = T*fav.astype(int)
Xc = sm.add_constant(X)
m = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
print(f'  T (in non-fav):       coef={m.params["T"]:+.4f} p={m.pvalues["T"]:.3e}  -> OR={np.exp(m.params["T"]):.3f}')
print(f'  fav main:             coef={m.params["fav"]:+.4f} p={m.pvalues["fav"]:.3e}')
print(f'  T*fav interaction:    coef={m.params["T_fav"]:+.4f} p={m.pvalues["T_fav"]:.3e}')
print(f'  T-effect in fav:      coef={m.params["T"]+m.params["T_fav"]:+.4f}  -> OR={np.exp(m.params["T"]+m.params["T_fav"]):.3f}')

# Marginal effects
print('\n=== Marginal counts:')
print(f'  Overall: T=0 N={(T==0).sum()} resp%={y[T==0].mean()*100:.1f}; T=1 N={(T==1).sum()} resp%={y[T==1].mean()*100:.1f}')
print(f'  Favorable: T=0 N={(fav&(T==0)).sum()} resp%={y[fav&(T==0)].mean()*100:.1f}; T=1 N={(fav&(T==1)).sum()} resp%={y[fav&(T==1)].mean()*100:.1f}')
print(f'  Non-fav: T=0 N={((~fav)&(T==0)).sum()} resp%={y[(~fav)&(T==0)].mean()*100:.1f}; T=1 N={((~fav)&(T==1)).sum()} resp%={y[(~fav)&(T==1)].mean()*100:.1f}')

# Distribution of treatment by fav
print(f'\n  Treatment prevalence in favorable: {T[fav].mean():.3f}')
print(f'  Treatment prevalence in non-favorable: {T[~fav].mean():.3f}')
# Confirm balance (no confounding by indication apparent)

# Look at feature_022 effect within favorable, T=0 and T=1 separately
print('\n=== Effect of feature_022 within favorable by treatment:')
for grp_name, grp in [('fav T=0', fav & (T==0)), ('fav T=1', fav & (T==1))]:
    yy = y[grp]; xx = df.loc[grp,'feature_022'].values
    Xc = sm.add_constant((xx-xx.mean())/xx.std())
    m = sm.Logit(yy, Xc).fit(disp=0)
    print(f'  {grp_name}: f022 coef (per SD)={m.params[1]:+.4f} p={m.pvalues[1]:.3e}')

# Look at feature_001 effect within favorable by treatment
print('\n=== Effect of feature_001 (stage-like) within favorable by treatment:')
for grp_name, grp in [('fav T=0', fav & (T==0)), ('fav T=1', fav & (T==1))]:
    yy = y[grp]; xx = df.loc[grp,'feature_001'].astype(float).values
    Xc = sm.add_constant(xx)
    m = sm.Logit(yy, Xc).fit(disp=0)
    print(f'  {grp_name}: f001 coef={m.params[1]:+.4f} p={m.pvalues[1]:.3e}')
