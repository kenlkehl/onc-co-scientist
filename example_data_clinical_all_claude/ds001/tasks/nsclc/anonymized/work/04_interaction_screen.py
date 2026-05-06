"""Iteration 4: full treatment-by-feature interaction screen.

For each candidate treatment T in {feature_012, feature_018, feature_020, feature_027},
fit OLS: pfs_months ~ T + X + T:X with appropriate encoding for X. Report the interaction
coefficient and p-value.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet').copy()
df['hist_squam'] = (df['feature_006'] == 'squamous').astype(int)
df['smk_current'] = (df['feature_001'] == 'current').astype(int)
df['smk_former'] = (df['feature_001'] == 'former').astype(int)

binary_feats = ['feature_031','feature_023','feature_011','feature_022','feature_016',
                'feature_028','feature_002','feature_005','feature_021','feature_018',
                'feature_020','feature_027','feature_012','hist_squam','smk_current','smk_former']
ord_feats = ['feature_014']
cont_feats = ['feature_015','feature_019','feature_025','feature_032','feature_017',
              'feature_024','feature_010','feature_003','feature_030','feature_008',
              'feature_004','feature_013','feature_026','feature_029','feature_007',
              'feature_033','feature_009']

treatments = ['feature_012','feature_018','feature_020','feature_027']

results = []
for T in treatments:
    others = [f for f in binary_feats + ord_feats + cont_feats if f != T]
    for X in others:
        # Standardize continuous X for clean interpretation
        if X in cont_feats:
            xv = (df[X] - df[X].mean()) / df[X].std()
        else:
            xv = df[X].astype(float)
        tv = df[T].astype(float)
        d = pd.DataFrame({'y': df['pfs_months'].values, 'T': tv.values, 'X': xv.values})
        m = smf.ols("y ~ T*X", data=d).fit()
        results.append({
            'T': T,
            'X': X,
            'beta_T': m.params.get('T', np.nan),
            'beta_X': m.params.get('X', np.nan),
            'beta_TX': m.params.get('T:X', np.nan),
            'p_TX': m.pvalues.get('T:X', np.nan),
            'r2': m.rsquared
        })

R = pd.DataFrame(results)
R = R.sort_values('p_TX')
print("=== Top 30 interactions by p-value (T:X) ===")
print(R.head(30).to_string(index=False))

R.to_csv('work/04_interaction_results.csv', index=False)

# Print top interactions per treatment
for T in treatments:
    print(f"\n=== Top interactions for T={T} ===")
    Rt = R[R['T'] == T].head(10)
    print(Rt.to_string(index=False))
