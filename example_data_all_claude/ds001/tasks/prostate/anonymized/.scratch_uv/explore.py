import pandas as pd
import numpy as np

df = pd.read_parquet('../dataset.parquet')
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nOutcome distribution:")
print(df['objective_response'].value_counts())
print("Response rate:", df['objective_response'].mean())

print("\nDescribe all features:")
desc = df.drop(columns=['patient_id']).describe().T
print(desc.to_string())

print("\nUnique value counts (top 10 cardinality features):")
for c in df.columns:
    if c == 'patient_id':
        continue
    n_uniq = df[c].nunique()
    print(f"  {c}: dtype={df[c].dtype}, n_unique={n_uniq}, sample={sorted(df[c].dropna().unique())[:6] if n_uniq < 20 else 'continuous'}")
