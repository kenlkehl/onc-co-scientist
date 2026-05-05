"""Iter 21-25: exhaustive 2- and 3-feature subgroup search for treatments with weak/null main effects."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from itertools import combinations

df = pd.read_parquet('dataset.parquet')
y = df['pfs_months'].values

with open('results_v2.json') as f:
    results = json.load(f)

# Define candidate binary modifiers for subgroup definition
modifiers = {
    'er_pos': df['er_positive']==1, 'er_neg': df['er_positive']==0,
    'pr_pos': df['pr_positive']==1, 'pr_neg': df['pr_positive']==0,
    'her2_pos': df['her2_positive']==1, 'her2_neg': df['her2_positive']==0,
    'her2_low': df['her2_low']==1,
    'brca1': df['brca1_mutation']==1, 'brca2': df['brca2_mutation']==1,
    'pik3ca': df['pik3ca_mutation']==1, 'pik3ca_wt': df['pik3ca_mutation']==0,
    'stage_iv': df['stage_iv']==1, 'stage_lt4': df['stage_iv']==0,
    'brain_mets': df['has_brain_mets']==1, 'no_brain': df['has_brain_mets']==0,
    'node_pos': df['node_positive']==1,
    'postmeno': df['postmenopausal']==1, 'premeno': df['postmenopausal']==0,
    'ecog0': df['ecog_ps']==0, 'ecog2': df['ecog_ps']==2,
    'high_ki67': df['ki67_pct']>=df['ki67_pct'].median(),
    'low_ki67': df['ki67_pct']<df['ki67_pct'].median(),
    'high_alb': df['albumin_g_dl']>=df['albumin_g_dl'].median(),
    'low_alb': df['albumin_g_dl']<df['albumin_g_dl'].median(),
    'old': df['age_years']>=df['age_years'].median(),
    'young': df['age_years']<df['age_years'].median(),
}

def search(treat, max_combo=2, top_n=15, min_n=200):
    rows = []
    keys = list(modifiers.keys())
    for k in range(1, max_combo+1):
        for combo in combinations(keys, k):
            mask = np.ones(len(df), dtype=bool)
            for m in combo:
                mask &= modifiers[m].values
            n_t = int(((df[treat]==1).values & mask).sum())
            n_u = int(((df[treat]==0).values & mask).sum())
            if n_t < min_n or n_u < min_n:
                continue
            yt = y[mask & (df[treat]==1).values]
            yu = y[mask & (df[treat]==0).values]
            diff = float(np.mean(yt) - np.mean(yu))
            t, p = stats.ttest_ind(yt, yu, equal_var=False)
            rows.append((combo, diff, float(p), n_t, n_u))
    # sort by signed diff (positive interesting)
    rows.sort(key=lambda r: -abs(r[1]))
    return rows[:top_n]

results['exhaustive_subgroups'] = {}
for t in ['treatment_trastuzumab','treatment_olaparib','treatment_sacituzumab_govitecan',
          'treatment_pembrolizumab','treatment_tamoxifen','treatment_palbociclib']:
    print(f'\n=== {t} top subgroups (|diff| sorted) ===')
    rows = search(t, max_combo=3, top_n=20, min_n=150)
    out = []
    for combo, diff, p, n_t, n_u in rows:
        print(f'  {combo}: diff={diff:+.3f}, p={p:.2e}, n_t={n_t}, n_u={n_u}')
        out.append({'subgroup': list(combo), 'diff': diff, 'p': p, 'n_t': n_t, 'n_u': n_u})
    results['exhaustive_subgroups'][t] = out

with open('results_v2.json', 'w') as fp:
    json.dump(results, fp, indent=2)
