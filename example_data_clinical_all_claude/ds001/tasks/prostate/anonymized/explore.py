"""Initial exploration of ds001_prostate."""
import pandas as pd
import numpy as np

df = pd.read_parquet('dataset.parquet')
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print()
print('Dtypes:')
print(df.dtypes)
print()
print('Describe (numeric):')
print(df.describe(include=[np.number]).T)
print()
print('Describe (object):')
print(df.describe(include=[object, 'category']).T if any(df.dtypes == object) or any(df.dtypes.astype(str).str.contains('category')) else 'no object columns')
print()
print('NA counts (top):')
print(df.isna().sum().sort_values(ascending=False).head(10))
print()
print('Unique values per column:')
for c in df.columns:
    nu = df[c].nunique(dropna=False)
    print(f'  {c}: nunique={nu}, dtype={df[c].dtype}')
