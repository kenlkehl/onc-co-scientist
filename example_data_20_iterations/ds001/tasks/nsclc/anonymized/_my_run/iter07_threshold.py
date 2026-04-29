"""Iteration 7: explore threshold behavior of feature_092 -- is the interaction with 006/007/039
limited to a high-biomarker subset? Test feature_092 binarized at quartiles and standard
PD-L1 thresholds (0.50)."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values

# Different thresholds for feature_092
for thresh, label in [(0.5, '>=0.50 (PD-L1 high-like)'),
                      (0.25, '>=0.25'),
                      (0.01, '>0 (positive vs negative)'),
                      (df['feature_092'].quantile(0.75), 'top-quartile (~0.47)')]:
    print(f'\n=== feature_092 threshold {label} ({thresh:.3f}) ===')
    high = (df['feature_092'] >= thresh).astype(int)
    for trt in ['feature_006', 'feature_007', 'feature_039']:
        tab = df.groupby([high.rename('high'), trt])['objective_response'].agg(['mean','count']).unstack(trt)
        # Compute interaction term
        x_high = high.values.astype(float)
        x_trt = df[trt].values.astype(float)
        X = np.column_stack([np.ones(len(df)), x_high, x_trt, x_high*x_trt])
        try:
            base = sm.Logit(y, X[:, :3]).fit(disp=False, maxiter=100).llf
            full = sm.Logit(y, X).fit(disp=False, maxiter=100)
            lr = 2*(full.llf - base)
            p_int = 1 - chi2_dist.cdf(lr, df=1)
            inter_coef = full.params[3]
            print(f'  {trt}: interaction coef={inter_coef:.3f}, p={p_int:.3g}')
            print(f'    high=0: trt0 ORR={tab[("mean",0)].iloc[0]:.3f}  trt1 ORR={tab[("mean",1)].iloc[0]:.3f}')
            print(f'    high=1: trt0 ORR={tab[("mean",0)].iloc[1]:.3f}  trt1 ORR={tab[("mean",1)].iloc[1]:.3f}')
        except Exception as e:
            print('  err', e)

# Now also check feature_006/007/039 jointly: interaction with feature_092 in a single model
print('\n=== Three-treatment model with biomarker interaction ===')
data = pd.DataFrame({
    'f092': (df['feature_092'] - df['feature_092'].mean()) / df['feature_092'].std(),
    'f006': df['feature_006'].astype(float),
    'f007': df['feature_007'].astype(float),
    'f039': df['feature_039'].astype(float),
})
data['f092_x_f006'] = data['f092'] * data['f006']
data['f092_x_f007'] = data['f092'] * data['f007']
data['f092_x_f039'] = data['f092'] * data['f039']
X = sm.add_constant(data).astype(float)
res = sm.Logit(y, X).fit(disp=False, maxiter=200)
print(res.summary())
