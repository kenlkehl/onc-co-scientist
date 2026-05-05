"""Iteration 1: dataset overview, treatment overlap, response rate, baseline contingency."""
import pandas as pd
import numpy as np

df = pd.read_parquet('dataset.parquet')

print("=== Treatments per patient (overlap) ===")
tcols = [c for c in df.columns if c.startswith('treatment_')]
print(tcols)
df['n_treatments'] = df[tcols].sum(axis=1)
print(df['n_treatments'].value_counts().sort_index())

print("\n=== Treatment prevalence ===")
for c in tcols:
    print(f"  {c}: n={df[c].sum()}  ({df[c].mean()*100:.2f}%)")

print("\n=== Overall objective_response rate ===")
print(f"  rate = {df['objective_response'].mean():.4f}  (n_resp={df['objective_response'].sum()})")

print("\n=== Treatment x response rate (unadjusted) ===")
for c in tcols:
    on = df.loc[df[c]==1, 'objective_response'].mean()
    off = df.loc[df[c]==0, 'objective_response'].mean()
    print(f"  {c}: on={on:.3f}  off={off:.3f}  diff={on-off:+.3f}")

print("\n=== Treatment co-occurrence matrix (Jaccard like — count both==1) ===")
import itertools
for a,b in itertools.combinations(tcols,2):
    both = ((df[a]==1)&(df[b]==1)).sum()
    print(f"  {a} & {b}: both={both}")

print("\n=== Categorical/binary feature prevalences ===")
binary_cols = ['mcrpc','visceral_mets','brca2_mutation','ar_v7_positive','msi_high','psma_high']
for c in binary_cols:
    print(f"  {c}: prev={df[c].mean():.3f}")

print("\n=== ECOG distribution ===")
print(df['ecog_ps'].value_counts().sort_index())
print("\n=== Gleason distribution ===")
print(df['gleason_score'].value_counts().sort_index())
