import pandas as pd
import numpy as np

df = pd.read_parquet("dataset.parquet")
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nDescribe:\n", df.describe().T)
print("\nBinary col means:")
for c in df.columns:
    if df[c].dropna().isin([0,1]).all() and df[c].nunique() <= 2:
        print(f"  {c}: mean={df[c].mean():.4f}")
