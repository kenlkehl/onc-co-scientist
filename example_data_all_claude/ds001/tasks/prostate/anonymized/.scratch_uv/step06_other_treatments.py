"""Test other binary features for treatment-like heterogeneity in subgroups."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].astype(int).values
feature_cols = [c for c in df.columns if c.startswith('feature_') and c != 'feature_030']
binary = [c for c in feature_cols if df[c].nunique()==2 and set(df[c].unique()).issubset({0,1})]
print(f'Binary features: {binary}')

print('\nFor each binary feature, examine main effect and interaction with the same set of modifiers (013,015,021,027,022):')
mods = ['feature_013','feature_015','feature_021','feature_027','feature_022']
for cand in binary:
    if cand in mods:
        continue
    # Main effect
    x = df[cand].astype(float).values
    # Build model with just main effect and adjusted for the modifiers
    cov_cols = [c for c in feature_cols if c not in [cand]]
    X = df[cov_cols].astype(float).copy()
    for c in cov_cols:
        if X[c].nunique() > 5:
            X[c] = (X[c]-X[c].mean())/X[c].std()
    X.insert(0, cand, x)
    Xc = sm.add_constant(X)
    m = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
    print(f'\n  {cand}: prevalence={df[cand].mean():.3f}, adj coef={m.params[cand]:+.4f} p={m.pvalues[cand]:.3e}')
    # Test interactions
    for mod in mods:
        xm = df[mod].astype(float).values
        if df[mod].nunique() > 5:
            xm = (xm - xm.mean())/xm.std()
        t = df[cand].astype(float).values
        Xint = np.column_stack([np.ones(len(t)), t, xm, t*xm])
        try:
            mi = sm.Logit(y, Xint).fit(disp=0, maxiter=100)
            print(f'    interaction with {mod}: int coef={mi.params[3]:+.4f} p={mi.pvalues[3]:.3e}')
        except Exception:
            pass
