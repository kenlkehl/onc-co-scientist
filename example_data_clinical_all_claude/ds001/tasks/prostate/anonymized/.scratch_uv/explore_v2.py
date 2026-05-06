import pandas as pd
import numpy as np

df = pd.read_parquet('../dataset.parquet')
print('shape:', df.shape)
print()
print('--- per-column summary ---')
for c in df.columns:
    s = df[c]
    is_numeric = pd.api.types.is_numeric_dtype(s)
    if not is_numeric:
        vc = s.value_counts(dropna=False)
        print(f'{c}: {s.dtype} | nuniq={s.nunique()} | top={vc.head(8).to_dict()}')
    else:
        print(f'{c}: {s.dtype} | nuniq={s.nunique()} | min={s.min()} max={s.max()} mean={s.mean():.3f} std={s.std():.3f}')
        if s.nunique() <= 12:
            print('    vc:', s.value_counts(dropna=False).sort_index().to_dict())
print()
print('outcome objective_response:')
print(df['objective_response'].value_counts(dropna=False))
print('overall response rate:', df['objective_response'].mean())
