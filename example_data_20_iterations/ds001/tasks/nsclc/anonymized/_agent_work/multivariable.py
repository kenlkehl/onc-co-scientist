"""Multivariable logistic regression on top features."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('../dataset.parquet')

# Top features identified by univariate screen
top_features = [
    'feature_051',  # ordinal 0-2 (ECOG-like)
    'feature_013',  # binary, ~65%
    'feature_011',  # continuous 0-28
    'feature_067',  # binary, ~25%
    'feature_006',  # binary, ~28%, positive
    'feature_099',  # continuous 1.7-5.5
    'feature_007',  # binary
    'feature_063',  # continuous (right-skewed: 0.03-296.88)
    'feature_092',  # continuous 0-0.8
    'feature_039',  # binary
    'feature_076',  # binary 4066
    'feature_112',  # binary 3879
    'feature_033',  # ordinal
    'feature_123',  # categorical (race)
    'feature_021',  # binary
    'feature_043',  # categorical histology
    'feature_057',  # smoking status
    'feature_078',  # age (continuous 30-90)
]

# Build design matrix
formula_terms = []
for c in top_features:
    if df[c].dtype == 'object':
        formula_terms.append(f"C({c})")
    else:
        formula_terms.append(c)
formula = "objective_response ~ " + " + ".join(formula_terms)
print('Formula:', formula)

mod = smf.logit(formula, data=df).fit(disp=0)
print(mod.summary())

# save
with open('multivariable_summary.txt', 'w') as f:
    f.write(str(mod.summary()))

# coefficients with OR
coef_df = pd.DataFrame({
    'coef': mod.params,
    'OR': np.exp(mod.params),
    'p': mod.pvalues,
    'se': mod.bse,
})
coef_df = coef_df.sort_values('p')
coef_df.to_csv('multivariable_coefs.csv')
print()
print(coef_df.to_string())
