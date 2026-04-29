"""Iteration 7: do effects of top predictors differ by race / insurance?
We test interaction terms after adjusting for the main multivariable model."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf

df = pd.read_parquet('../dataset.parquet')
df['log_feature_013'] = np.log1p(df['feature_013'])

base = ('pfs_months ~ feature_078 + log_feature_013 + feature_006 + feature_009 '
        '+ feature_092 + feature_109 + feature_039 + feature_027 + feature_043 '
        '+ C(feature_018, Treatment(reference="white")) '
        '+ C(feature_045, Treatment(reference="private"))')

# Test interactions one predictor at a time
key_preds = ['feature_078','log_feature_013','feature_006','feature_009','feature_109','feature_039']

for kp in key_preds:
    for demo, ref in [('feature_018','white'),('feature_045','private')]:
        formula = base + f' + {kp}:C({demo}, Treatment(reference="{ref}"))'
        m = smf.ols(formula, data=df).fit()
        # Extract interaction p-values
        interaction_terms = [n for n in m.params.index if (kp in n and demo in n)]
        # Joint F-test
        if interaction_terms:
            try:
                wald = m.f_test(' = '.join(interaction_terms) + ' = 0' if False else
                                ', '.join([f'{t} = 0' for t in interaction_terms]))
                print(f'{kp} x {demo}: joint F={wald.fvalue:.3f}, p={wald.pvalue:.3e}')
            except Exception as e:
                print(f'{kp} x {demo}: error {e}')
            for t in interaction_terms:
                print(f'    {t}: coef={m.params[t]:+.5f}, p={m.pvalues[t]:.3e}')
        print()
