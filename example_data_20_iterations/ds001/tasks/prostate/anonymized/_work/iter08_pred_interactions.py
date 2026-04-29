"""Iteration 8: pairwise interactions among top continuous predictors, plus quadratic terms."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf

df = pd.read_parquet('../dataset.parquet')
df['log_feature_013'] = np.log1p(df['feature_013'])

base_terms = ['feature_078','log_feature_013','feature_006','feature_009','feature_092',
              'feature_109','feature_039','feature_027','feature_043']
key_pairs = [
    ('feature_078','log_feature_013'),
    ('feature_078','feature_006'),
    ('feature_078','feature_009'),
    ('log_feature_013','feature_006'),
    ('log_feature_013','feature_009'),
    ('feature_006','feature_009'),
    ('feature_078','feature_109'),
    ('feature_078','feature_039'),
    ('log_feature_013','feature_109'),
]
base = 'pfs_months ~ ' + ' + '.join(base_terms)
m_base = smf.ols(base, data=df).fit()
print(f'Base model R2={m_base.rsquared:.4f}')

print('\n=== Pairwise interactions ===')
for a, b in key_pairs:
    formula = base + f' + {a}:{b}'
    m = smf.ols(formula, data=df).fit()
    coef = m.params[f'{a}:{b}']
    p = m.pvalues[f'{a}:{b}']
    delta_r2 = m.rsquared - m_base.rsquared
    print(f'  {a} x {b}: coef={coef:+.5f}, p={p:.3e}, dR2={delta_r2:+.5f}')

print('\n=== Quadratic terms ===')
for v in ['feature_078','log_feature_013','feature_006','feature_009','feature_092']:
    formula = base + f' + I({v}**2)'
    m = smf.ols(formula, data=df).fit()
    key = f'I({v} ** 2)'
    if key not in m.pvalues:
        # statsmodels prints with spaces sometimes
        key = [k for k in m.pvalues.index if v in k and '**' in k][0]
    coef = m.params[key]
    p = m.pvalues[key]
    delta_r2 = m.rsquared - m_base.rsquared
    print(f'  {v}^2: coef={coef:+.5g}, p={p:.3e}, dR2={delta_r2:+.5f}')
