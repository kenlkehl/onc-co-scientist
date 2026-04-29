import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('dataset.parquet')
y = df['pfs_months']
results = []
for c in df.columns:
    if c in ('patient_id', 'pfs_months'):
        continue
    s = df[c]
    if s.dtype == 'object':
        # ANOVA across categories
        groups = [y[s == lvl].values for lvl in s.unique()]
        try:
            f, p = stats.f_oneway(*groups)
        except Exception:
            f, p = np.nan, np.nan
        means = {str(lvl): float(y[s == lvl].mean()) for lvl in s.unique()}
        results.append({'col': c, 'kind': 'cat', 'n_lvl': s.nunique(), 'stat': f, 'p': p,
                        'effect': max(means.values()) - min(means.values()),
                        'detail': means})
    elif s.nunique() == 2:
        a = y[s == 1].values; b = y[s == 0].values
        if len(a) < 5 or len(b) < 5:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results.append({'col': c, 'kind': 'bin', 'n1': len(a), 'n0': len(b),
                        'stat': t, 'p': p, 'effect': float(a.mean() - b.mean()),
                        'mean1': float(a.mean()), 'mean0': float(b.mean())})
    elif s.nunique() <= 11:
        # ordinal
        r, p = stats.spearmanr(s, y)
        # also linear trend
        slope, intercept, rval, plin, _ = stats.linregress(s, y)
        results.append({'col': c, 'kind': 'ord', 'n_lvl': s.nunique(),
                        'spearman_rho': r, 'p': p, 'lin_slope': slope, 'lin_p': plin,
                        'effect': slope})
    else:
        # continuous - spearman
        r, p = stats.spearmanr(s, y)
        slope, intercept, rval, plin, _ = stats.linregress(s, y)
        results.append({'col': c, 'kind': 'cont',
                        'spearman_rho': r, 'p': p, 'lin_slope': slope, 'lin_p': plin,
                        'effect': slope})

dfres = pd.DataFrame(results)
dfres['absp'] = dfres['p'].apply(lambda v: -np.log10(v) if v and v > 0 else np.nan)
dfres = dfres.sort_values('p', na_position='last')
print(dfres.head(40).to_string())
print()
print('--- Top by absolute effect (binary) ---')
binmask = dfres['kind'] == 'bin'
print(dfres[binmask].assign(abseff=lambda d: d['effect'].abs()).sort_values('abseff', ascending=False).head(25).to_string())
print()
print('--- Categorical results ---')
print(dfres[dfres['kind'] == 'cat'].to_string())
print()
print('--- Ordinal results ---')
print(dfres[dfres['kind'] == 'ord'].to_string())
dfres.to_csv('_work/screen_results.csv', index=False)
