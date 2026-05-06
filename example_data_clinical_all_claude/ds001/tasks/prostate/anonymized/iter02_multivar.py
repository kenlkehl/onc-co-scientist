"""Iteration 2: Multivariable logistic regression and correlations among predictors."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

# ===== Correlations among the strongest predictors =====
strong = ['feature_008', 'feature_013', 'feature_015', 'feature_021', 'feature_027',
          'feature_001', 'feature_022', 'feature_020', 'feature_024', 'feature_002']
print('=== Pearson correlations among strong predictors ===')
print(df[strong].corr().round(3))
print()

# ===== Co-occurrence of the binary "treatment-like" features =====
binary_strong = ['feature_008', 'feature_013', 'feature_015', 'feature_021', 'feature_027']
print('=== Crosstabs among binary strong predictors ===')
for i, a in enumerate(binary_strong):
    for b in binary_strong[i+1:]:
        ct = pd.crosstab(df[a], df[b])
        n11 = ct.iloc[1,1]
        ja = (df[a]==1).sum(); jb = (df[b]==1).sum()
        expected_n11 = ja*jb/len(df)
        print(f'{a} & {b}: both=1 n={n11} (expected {expected_n11:.0f})')
print()

# ===== Multivariable logistic with Box-Cox-like transforms for skewed =====
# log-transform the right-skewed continuous features that mattered
df['log_f022'] = np.log1p(df['feature_022'])
df['log_f020'] = np.log1p(df['feature_020'])
df['log_f024'] = np.log1p(df['feature_024'])

predictors = ['feature_008', 'feature_013', 'feature_015', 'feature_021', 'feature_027',
              'feature_001', 'log_f022', 'log_f020', 'log_f024', 'feature_002',
              'feature_016', 'feature_010']
X = sm.add_constant(df[predictors].astype(float))
y = df['objective_response'].astype(int)
print('=== Multivariable logistic regression (objective_response) ===')
model = sm.Logit(y, X).fit(disp=False)
print(model.summary())
print()

# Print odds ratios for clarity
print('=== Adjusted odds ratios ===')
or_table = pd.DataFrame({
    'beta': model.params,
    'OR': np.exp(model.params),
    'OR_LCL': np.exp(model.conf_int()[0]),
    'OR_UCL': np.exp(model.conf_int()[1]),
    'p': model.pvalues
})
print(or_table.round(4))
