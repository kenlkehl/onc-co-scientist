"""Try simpler subgroup definitions and confirm necessity formally."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].astype(int).values
T = df['feature_008'].astype(int).values
mods = ['feature_013','feature_015','feature_021','feature_027']

print('Test all subset combinations of the 4 modifiers:')
print(f'{"def":<60}{"n_fav":>10}{"T-eff fav":>12}{"T-eff !fav":>14}')
for r in range(1,5):
    for combo in combinations(mods, r):
        fav = pd.Series([True]*len(df))
        for m in combo:
            fav = fav & (df[m]==0)
        fav = fav.values
        if fav.sum()<10 or (~fav).sum()<10:
            continue
        eff_fav = y[fav & (T==1)].mean() - y[fav & (T==0)].mean()
        eff_nfav = y[(~fav) & (T==1)].mean() - y[(~fav) & (T==0)].mean()
        print(f'{"&".join([m+"=0" for m in combo]):<60}{fav.sum():>10}{eff_fav:>12.3f}{eff_nfav:>14.3f}')

# Confirm formal interaction test: treatment * subgroup_indicator
print('\n=== Formal logistic: response ~ T * fav (where fav = all 4 mods=0)')
fav = ((df['feature_013']==0)&(df['feature_015']==0)&(df['feature_021']==0)&(df['feature_027']==0)).astype(int).values
X = np.column_stack([np.ones(len(T)), T, fav, T*fav])
m = sm.Logit(y, X).fit(disp=0)
print(m.summary())

# Now within non-favorable, look for sub-subgroup with effect
print('\n=== Within NON-favorable, look for any treatment effect heterogeneity:')
nfav = ~((df['feature_013']==0)&(df['feature_015']==0)&(df['feature_021']==0)&(df['feature_027']==0)).values
yn = y[nfav]; Tn = T[nfav]
print(f'  Overall T-effect in non-favorable: {yn[Tn==1].mean()-yn[Tn==0].mean():.4f}')
# Test interactions with all features inside non-favorable
feature_cols = [c for c in df.columns if c.startswith('feature_') and c not in ['feature_030','feature_008']]
print('  Top interactions inside non-favorable (treatment x feature):')
res = []
for c in feature_cols:
    x = df.loc[nfav, c].astype(float).values
    if df[c].nunique() > 5:
        x = (x - x.mean())/x.std()
    Xint = np.column_stack([np.ones(len(Tn)), Tn, x, Tn*x])
    try:
        mi = sm.Logit(yn, Xint).fit(disp=0, maxiter=100)
        res.append((c, mi.params[3], mi.pvalues[3]))
    except Exception:
        pass
res.sort(key=lambda r: r[2])
for c, b, p in res[:10]:
    print(f'    {c}: int={b:+.4f} p={p:.3e}')
