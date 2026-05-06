import pandas as pd
import numpy as np

df = pd.read_parquet('../dataset.parquet')
print('Shape:', df.shape)
print('\nDtypes:')
print(df.dtypes.to_string())
print('\nColumn nunique:')
for c in df.columns:
    print(f'  {c}: nunique={df[c].nunique()}, dtype={df[c].dtype}')
print('\nDescribe (numeric):')
print(df.describe().T.to_string())
print('\nDescribe (object):')
print(df.describe(include=['object', 'category', 'bool']).T.to_string() if (df.select_dtypes(include=['object','category','bool']).shape[1]>0) else 'no object cols')
print('\nValue counts for categorical/binary cols:')
for c in df.columns:
    if df[c].nunique() <= 10:
        print(f'\n{c}:')
        print(df[c].value_counts(dropna=False).to_string())
