"""Iteration 7: Exhaustive 2-way binary subgroup search for non-enzalutamide treatments.

For each treatment t, for each pair (b1, b2) of binary features, evaluate the
within-cell treatment effect for all 4 combinations (b1∈{0,1} x b2∈{0,1}).
Report top cells by abs(diff) with reasonable n.

Then a 3-way exhaustive over binary features for each treatment.
"""
import pandas as pd
import numpy as np
from scipy import stats
import itertools
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
binary = ['mcrpc','visceral_mets','brca2_mutation','ar_v7_positive','msi_high','psma_high']
# Add ECOG categories and gleason categories as binarized splits for richer search
df['ecog_high'] = (df['ecog_ps']>=2).astype(int)
df['ecog_zero'] = (df['ecog_ps']==0).astype(int)
df['gleason_high'] = (df['gleason_score']>=8).astype(int)
df['psa_high'] = (df['psa_ng_ml']>=df['psa_ng_ml'].median()).astype(int)
df['ldh_high'] = (df['ldh_u_l']>df['ldh_u_l'].median()).astype(int)
df['alb_low'] = (df['albumin_g_dl']<df['albumin_g_dl'].median()).astype(int)
df['nlr_high'] = (df['nlr']>df['nlr'].median()).astype(int)
df['crp_high'] = (df['crp_mg_l']>df['crp_mg_l'].median()).astype(int)

bin_features = binary + ['ecog_high','ecog_zero','gleason_high','psa_high','ldh_high','alb_low','nlr_high','crp_high']
treatments = ['treatment_abiraterone','treatment_docetaxel',
              'treatment_olaparib','treatment_lu177_psma','treatment_pembrolizumab']

print("=== 2-way subgroup search: top cells with biggest treatment effect (n_on>=20, n_off>=20) ===")
for t in treatments:
    rows = []
    for b1, b2 in itertools.combinations(bin_features, 2):
        for v1, v2 in itertools.product([0,1],[0,1]):
            sub = df[(df[b1]==v1) & (df[b2]==v2)]
            if len(sub) < 50: continue
            on = sub.loc[sub[t]==1, 'objective_response']
            off = sub.loc[sub[t]==0, 'objective_response']
            if len(on) < 20 or len(off) < 20: continue
            d = on.mean() - off.mean()
            try:
                tab = pd.crosstab(sub[t], sub['objective_response'])
                chi2, p, _, _ = stats.chi2_contingency(tab)
            except:
                p = 1.0
            rows.append({'b1':f"{b1}={v1}", 'b2':f"{b2}={v2}", 'n':len(sub),
                        'n_on':len(on), 'n_off':len(off),
                        'rate_on':on.mean(), 'rate_off':off.mean(), 'diff':d, 'p':p})
    R = pd.DataFrame(rows).sort_values('p').head(8)
    print(f"\n--- {t}: top 8 cells by p ---")
    for _, r in R.iterrows():
        print(f"  {r.b1} & {r.b2}: n={r.n}, on={r.n_on} ({r.rate_on:.3f})  off={r.n_off} ({r.rate_off:.3f})  diff={r['diff']:+.3f}  p={r.p:.3g}")

# Also run a 3-way restricted: hold-back to looking for any subgroup defined by 3 binary features
# where the treatment difference is large.
print("\n=== 3-way subgroup search (looser): for each treatment, top 5 by abs(diff) where n_on>=30, n_off>=30 ===")
for t in treatments:
    rows = []
    for combo in itertools.combinations(bin_features, 3):
        for vals in itertools.product([0,1],[0,1],[0,1]):
            mask = np.ones(len(df), bool)
            for c,v in zip(combo, vals):
                mask &= (df[c].values==v)
            sub = df[mask]
            if len(sub) < 100: continue
            on = sub.loc[sub[t]==1, 'objective_response']
            off = sub.loc[sub[t]==0, 'objective_response']
            if len(on) < 30 or len(off) < 30: continue
            d = on.mean() - off.mean()
            try:
                tab = pd.crosstab(sub[t], sub['objective_response'])
                chi2, p, _, _ = stats.chi2_contingency(tab)
            except:
                p = 1.0
            rows.append({'predicate': "  &  ".join(f"{c}={v}" for c,v in zip(combo,vals)),
                        'n':len(sub), 'rate_on':on.mean(), 'rate_off':off.mean(), 'diff':d, 'p':p})
    R = pd.DataFrame(rows).sort_values('p').head(5)
    print(f"\n--- {t}: top 5 ---")
    for _, r in R.iterrows():
        print(f"  {r.predicate}  | n={r.n}, on rate={r.rate_on:.3f}  off rate={r.rate_off:.3f}  diff={r['diff']:+.3f}  p={r.p:.3g}")
