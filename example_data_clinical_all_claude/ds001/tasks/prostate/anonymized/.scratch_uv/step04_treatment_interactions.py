"""Test feature_008 (strongest positive predictor, candidate treatment)
for heterogeneity of effect by every other feature."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].astype(int).values
feature_cols = [c for c in df.columns if c.startswith('feature_') and c != 'feature_030']

T = 'feature_008'
print(f'Treatment candidate: {T}')
print(f'  prevalence={df[T].mean():.3f}')
print(f'  response | T=1: {y[df[T]==1].mean():.3f}; T=0: {y[df[T]==0].mean():.3f}')
print(f'  marginal lift: {y[df[T]==1].mean() - y[df[T]==0].mean():.4f}')

print()
print('Treatment-by-feature interactions (logistic with main effects + interaction):')
print(f'{"feat":<14}{"main coef":>12}{"int coef":>12}{"int p":>14}{"T-eff f=0":>12}{"T-eff f=1":>12}')
results = []
for c in feature_cols:
    if c == T:
        continue
    x = df[c].astype(float).values.copy()
    if df[c].nunique() > 5:
        x = (x - x.mean()) / x.std()
    t = df[T].astype(float).values
    X = np.column_stack([np.ones(len(t)), t, x, t*x])
    try:
        m = sm.Logit(y, X).fit(disp=0, maxiter=100)
        b_T, b_X, b_TX = m.params[1], m.params[2], m.params[3]
        p_int = m.pvalues[3]
        # Treatment effect at f=0 vs f=1 (for binary feats)
        if df[c].nunique() == 2:
            te0 = b_T  # at x=0
            te1 = b_T + b_TX  # at x=1
        else:
            te0 = b_T - b_TX  # at -1 SD
            te1 = b_T + b_TX  # at +1 SD
        results.append((c, b_X, b_TX, p_int, te0, te1))
    except Exception as e:
        results.append((c, np.nan, np.nan, np.nan, np.nan, np.nan))
results.sort(key=lambda r: r[3] if not np.isnan(r[3]) else 1.0)
for c, bX, bTX, p, te0, te1 in results:
    print(f'{c:<14}{bX:+12.4f}{bTX:+12.4f}{p:>14.3e}{te0:+12.4f}{te1:+12.4f}')
