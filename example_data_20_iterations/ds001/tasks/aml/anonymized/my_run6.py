"""Final refining iterations: 3-way interactions, full multivariable, demographic adjustment."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
y = df['objective_response'].astype(int)

# 1. Three-way interaction f057 x f035 x f027
print('=== f057 x f035 x f027 cells ===')
df['cell'] = df['feature_057'].astype(str) + '_' + df['feature_035'].astype(str) + '_' + df['feature_027'].astype(str)
ct = df.groupby('cell')['objective_response'].agg(['count','mean']).round(4)
print(ct)

# 2. Adjusted f035 effect: in f027=1 vs f027=0
# is f027=1 + f035=1 the highest response group?
print('\nResponse rates by f027 + f035 combo:')
print(df.groupby(['feature_027','feature_035'])['objective_response'].agg(['count','mean']).round(4))

# 3. Same for f014 + f035
print('\nResponse rates by f014 + f035 combo:')
print(df.groupby(['feature_014','feature_035'])['objective_response'].agg(['count','mean']).round(4))

# 4. Composite biomarker f027 OR f014
df['comp_marker'] = ((df['feature_014']==1) | (df['feature_027']==1)).astype(int)
print('\n=== Composite (f014 OR f027) x f035 ===')
print(df.groupby(['comp_marker','feature_035'])['objective_response'].agg(['count','mean']).round(4))
# stratified
for cm in [0,1]:
    sub = df[df['comp_marker']==cm]
    yy = sub['objective_response']
    xx = sub['feature_035'].astype(float)
    mm = sm.Logit(yy, sm.add_constant(xx)).fit(disp=0)
    print(f'  comp={cm}: f035 OR={np.exp(mm.params.iloc[1]):.3f}, p={mm.pvalues.iloc[1]:.3g}, n={len(sub)}')

# 5. Final adjusted multivariable model — top features
print('\n=== Final adjusted multivariable (with key interactions) ===')
X = pd.DataFrame({
    'f057': df['feature_057'].astype(float),
    'f035': df['feature_035'].astype(float),
    'f027': df['feature_027'].astype(float),
    'f014': df['feature_014'].astype(float),
    'f035_x_f027': df['feature_035'].astype(float) * df['feature_027'].astype(float),
    'f035_x_f014': df['feature_035'].astype(float) * df['feature_014'].astype(float),
    'f011_z': (df['feature_011']-df['feature_011'].mean())/df['feature_011'].std(),
    'f006_z': (df['feature_006']-df['feature_006'].mean())/df['feature_006'].std(),
    'f099_z': (df['feature_099']-df['feature_099'].mean())/df['feature_099'].std(),
    'f092_z': (df['feature_092']-df['feature_092'].mean())/df['feature_092'].std(),
    'f063_z': (df['feature_063']-df['feature_063'].mean())/df['feature_063'].std(),
})
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
print(m.summary().tables[1])

# 6. Demographic adjustment (sex if any binary near 50/50, race, insurance)
# Looking at balanced binaries — none meaningfully predict response on univariate
# Let's add race adjustment
print('\n=== Adjusted for race + insurance ===')
race_d = pd.get_dummies(df['feature_005'], prefix='race', drop_first=True).astype(float)
ins_d = pd.get_dummies(df['feature_087'], prefix='ins', drop_first=True).astype(float)
X = pd.concat([
    df['feature_057'].astype(float).rename('f057'),
    df['feature_035'].astype(float).rename('f035'),
    race_d,
    ins_d,
    ((df['feature_011']-df['feature_011'].mean())/df['feature_011'].std()).rename('f011'),
    ((df['feature_006']-df['feature_006'].mean())/df['feature_006'].std()).rename('f006'),
    df['feature_078'].rename('f078'),  # age
], axis=1)
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
print(m.summary().tables[1])

# 7. ROC-style discrimination
from sklearn.metrics import roc_auc_score
preds = m.predict(sm.add_constant(X))
auc = roc_auc_score(y, preds)
print(f'\nMultivariable model AUC: {auc:.4f}')

# 8. Summary table for top features
print('\n=== Final summary of effects ===')
all_summ = []
def lrf(name, code, x_label):
    """Run a logit and return summary."""
    return code

# f057 effect numerical summary
summ_lines = []
for c in ['feature_057', 'feature_035', 'feature_011', 'feature_006', 'feature_099',
          'feature_092', 'feature_063', 'feature_093', 'feature_014', 'feature_027']:
    if df[c].dtype.kind == 'f':
        z = (df[c]-df[c].mean())/df[c].std()
        m = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
        summ_lines.append(f'{c} (per SD): OR={np.exp(m.params.iloc[1]):.3f}, p={m.pvalues.iloc[1]:.3g}')
    else:
        m = sm.Logit(y, sm.add_constant(df[c].astype(float))).fit(disp=0)
        summ_lines.append(f'{c}: OR={np.exp(m.params.iloc[1]):.3f}, p={m.pvalues.iloc[1]:.3g}')
print('\n'.join(summ_lines))

# 9. f027 main effect within f035=1 — is this just bookkeeping
# Note: f027 main effect stratified by f035 was already computed.
# Save key data
import json
with open('my_summary.json','w') as f:
    json.dump({
        'response_rate': float(y.mean()),
        'n': int(len(df)),
        'auc_multivariable': float(auc),
        'summaries': summ_lines,
    }, f, indent=2)
print('\nDone.')
