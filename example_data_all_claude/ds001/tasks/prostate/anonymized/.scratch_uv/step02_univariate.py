import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].astype(int).values
print(f'Outcome rate overall: {y.mean():.4f} (n={len(y)})')

# Identify feature columns; drop constant and id
feature_cols = [c for c in df.columns if c.startswith('feature_') and c != 'feature_030']
binary_feats = [c for c in feature_cols if df[c].nunique() == 2 and set(df[c].unique()).issubset({0,1})]
multilevel_feats = [c for c in feature_cols if df[c].nunique() in (3,4,5)]
continuous_feats = [c for c in feature_cols if df[c].nunique() > 5]

print(f'Binary: {len(binary_feats)} -> {binary_feats}')
print(f'Multilevel (3-5): {len(multilevel_feats)} -> {multilevel_feats}')
print(f'Continuous: {len(continuous_feats)} -> {continuous_feats}')
print()

# Univariate logistic regression for each feature
print('Univariate logistic regression on objective_response:')
print(f'{"feat":<14}{"coef":>10}{"OR":>8}{"p":>12}{"n_pos":>8}')
results = []
for c in feature_cols:
    x = df[c].astype(float).values
    X = sm.add_constant(x)
    try:
        m = sm.Logit(y, X).fit(disp=0, maxiter=100)
        coef = m.params[1]
        p = m.pvalues[1]
        OR = np.exp(coef)
        results.append((c, coef, OR, p))
    except Exception as e:
        results.append((c, np.nan, np.nan, np.nan))
results.sort(key=lambda r: r[3] if not np.isnan(r[3]) else 1.0)
for c, coef, OR, p in results:
    print(f'{c:<14}{coef:>10.4f}{OR:>8.3f}{p:>12.3e}')
