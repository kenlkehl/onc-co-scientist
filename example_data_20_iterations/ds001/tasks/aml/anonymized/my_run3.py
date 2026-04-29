"""More targeted hypothesis testing — focus on top continuous features and possible interactions."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
y = df['objective_response'].astype(int)

# Look at distributions of key continuous features
key_cont = ['feature_078','feature_011','feature_006','feature_099','feature_092','feature_063','feature_055','feature_044','feature_084','feature_028','feature_065','feature_119','feature_056','feature_103','feature_094','feature_123','feature_019','feature_059','feature_003','feature_010','feature_101','feature_020','feature_090','feature_120','feature_054','feature_062','feature_031','feature_082','feature_097','feature_077','feature_106','feature_086','feature_033','feature_048','feature_037']

print('Looking at continuous-feature ranges to guess identities:')
for c in key_cont[:8]:
    print(f'{c}: range={df[c].min()}-{df[c].max()}, mean={df[c].mean():.2f}, std={df[c].std():.2f}')

# 4-quartile response rate for top continuous predictors
print('\n=== feature_011 quartile analysis ===')
df['f011_q'] = pd.qcut(df['feature_011'], 4, duplicates='drop')
print(df.groupby('f011_q', observed=True)['objective_response'].agg(['count','mean']))

print('\n=== feature_006 quartile analysis ===')
df['f006_q'] = pd.qcut(df['feature_006'], 4, duplicates='drop')
print(df.groupby('f006_q', observed=True)['objective_response'].agg(['count','mean']))

print('\n=== feature_078 (age?) quartile analysis ===')
df['f078_q'] = pd.qcut(df['feature_078'], 4, duplicates='drop')
print(df.groupby('f078_q', observed=True)['objective_response'].agg(['count','mean']))

# Logistic for f078
m = sm.Logit(y, sm.add_constant(df['feature_078'])).fit(disp=0)
print(f'feature_078: OR per unit = {np.exp(m.params.iloc[1]):.4f}, p={m.pvalues.iloc[1]:.3g}')

# Test interaction f057 x continuous biomarkers
print('\n=== f057 x continuous-biomarker interactions ===')
for cv in ['feature_011','feature_006','feature_099','feature_092','feature_063','feature_078','feature_044','feature_084']:
    z = (df[cv]-df[cv].mean())/df[cv].std()
    X = pd.DataFrame({
        'f057': df['feature_057'].astype(float),
        'cv': z,
        'f057_x_cv': df['feature_057'].astype(float) * z,
    })
    mm = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    pi = mm.pvalues['f057_x_cv']
    bi = mm.params['f057_x_cv']
    print(f'  f057 x {cv}: interaction beta={bi:.4f}, p={pi:.4g}')

# Test interaction f035 x continuous biomarkers
print('\n=== f035 x continuous-biomarker interactions ===')
for cv in ['feature_011','feature_006','feature_099','feature_092','feature_063','feature_078']:
    z = (df[cv]-df[cv].mean())/df[cv].std()
    X = pd.DataFrame({
        'f035': df['feature_035'].astype(float),
        'cv': z,
        'f035_x_cv': df['feature_035'].astype(float) * z,
    })
    mm = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    pi = mm.pvalues['f035_x_cv']
    bi = mm.params['f035_x_cv']
    print(f'  f035 x {cv}: interaction beta={bi:.4f}, p={pi:.4g}')

# Comprehensive multivariable model with all features
print('\n=== Comprehensive multivariable logistic ===')
feature_cols = [c for c in df.columns if c.startswith('feature_')]
binary_cols = [c for c in feature_cols if df[c].dtype != object and df[c].nunique() == 2]
multi_cols = [c for c in feature_cols if df[c].dtype.kind in 'iu' and 2 < df[c].nunique() <= 11]
cont_cols = [c for c in feature_cols if df[c].dtype.kind == 'f']
cat_cols = [c for c in feature_cols if df[c].dtype == object]

X_parts = [df[binary_cols].astype(float)]
X_parts.append(df[multi_cols].astype(float))
# Standardize continuous
for c in cont_cols:
    X_parts.append(((df[c]-df[c].mean())/df[c].std()).rename(c))
# Categorical
for c in cat_cols:
    d = pd.get_dummies(df[c], prefix=c, drop_first=True).astype(float)
    X_parts.append(d)

X = pd.concat(X_parts, axis=1)
X = sm.add_constant(X)
print('Model has', X.shape[1], 'parameters')
m = sm.Logit(y, X).fit(disp=0, maxiter=200)
res = pd.DataFrame({'coef': m.params, 'OR': np.exp(m.params), 'p': m.pvalues}).sort_values('p')
res.to_csv('multiv_my.csv')
print('Top 25 multivariable signals:')
print(res.head(25).to_string())
print('\nFeatures significant at 0.05 (multivariable, 124 features):', (res['p']<0.05).sum())
