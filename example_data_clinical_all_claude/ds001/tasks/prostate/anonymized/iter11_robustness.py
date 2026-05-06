"""Iteration 11: Bootstrap CIs for the responsive subgroup ORR estimates,
and continuous-feature prognostic model within untreated patients.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
T = 'feature_008'
modifiers = ['feature_013', 'feature_015', 'feature_021', 'feature_027']

fav = ((df['feature_013']==0)&(df['feature_015']==0)&
       (df['feature_021']==0)&(df['feature_027']==0))
sub = df[fav].copy()

# Bootstrap risk difference and OR
rng = np.random.default_rng(42)
B = 2000
rds = []; ors = []
n = len(sub)
y_arr = sub['objective_response'].values
T_arr = sub[T].values
for _ in range(B):
    idx = rng.integers(0, n, n)
    yb = y_arr[idx]; Tb = T_arr[idx]
    p0 = yb[Tb==0].mean(); p1 = yb[Tb==1].mean()
    rds.append(p1 - p0)
    a = (Tb==1).sum() * p1
    b = (Tb==1).sum() * (1-p1)
    c = (Tb==0).sum() * p0
    d = (Tb==0).sum() * (1-p0)
    ors.append((a*d)/(b*c))
print(f'Responsive subgroup (n={n}):')
print(f'  Risk difference: {np.mean(rds):.4f}  95% CI [{np.percentile(rds, 2.5):.4f}, {np.percentile(rds, 97.5):.4f}]')
print(f'  Odds ratio:      {np.mean(ors):.3f}  95% CI [{np.percentile(ors, 2.5):.3f}, {np.percentile(ors, 97.5):.3f}]')
print()

# ===== Continuous-feature prognostic model in untreated patients =====
print('=== Prognostic model within T=0 (untreated): logistic regression ===')
sub_t0 = df[df[T]==0].copy()
y_t0 = sub_t0['objective_response'].astype(int)
sub_t0['log_f022'] = np.log1p(sub_t0['feature_022'])
sub_t0['log_f020'] = np.log1p(sub_t0['feature_020'])
sub_t0['log_f024'] = np.log1p(sub_t0['feature_024'])

predictors = ['feature_013', 'feature_015', 'feature_021', 'feature_027',
              'feature_001', 'log_f022', 'log_f020', 'log_f024', 'feature_002',
              'feature_016', 'feature_010', 'feature_018', 'feature_026',
              'feature_009', 'feature_029', 'feature_003', 'feature_012',
              'feature_025', 'feature_028', 'feature_007', 'feature_014', 'feature_032',
              'feature_031', 'feature_006', 'feature_005', 'feature_023', 'feature_011',
              'feature_017', 'feature_019', 'feature_004']
X = sm.add_constant(sub_t0[predictors].astype(float))
model = sm.Logit(y_t0, X).fit(disp=False)
or_table = pd.DataFrame({
    'beta': model.params,
    'OR': np.exp(model.params),
    'p': model.pvalues
}).round(4)
print(or_table.sort_values('p').to_string())
print()

# ===== Same prognostic model in treated patients =====
print('=== Within T=1 (treated): predictive model — what predicts response on treatment? ===')
sub_t1 = df[df[T]==1].copy()
y_t1 = sub_t1['objective_response'].astype(int)
sub_t1['log_f022'] = np.log1p(sub_t1['feature_022'])
sub_t1['log_f020'] = np.log1p(sub_t1['feature_020'])
sub_t1['log_f024'] = np.log1p(sub_t1['feature_024'])
X = sm.add_constant(sub_t1[predictors].astype(float))
model = sm.Logit(y_t1, X).fit(disp=False)
or_table = pd.DataFrame({
    'beta': model.params,
    'OR': np.exp(model.params),
    'p': model.pvalues
}).round(4)
print(or_table.sort_values('p').to_string())
