"""Iterative analysis of ds001_nsclc dataset for transcript generation."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
print('Loaded:', df.shape)

# Encode categoricals
df['smoke_current'] = (df['smoking_status'] == 'current').astype(int)
df['smoke_never'] = (df['smoking_status'] == 'never').astype(int)
df['histology_squamous'] = (df['histology'] == 'squamous').astype(int)
df['ecog_ge2'] = (df['ecog_ps'] >= 2).astype(int)
df['pdl1_high'] = (df['pdl1_tps'] >= 0.5).astype(int)

OUTCOME = 'pfs_months'
TREATMENTS = ['treatment_pembrolizumab', 'treatment_sotorasib',
              'treatment_olaparib', 'treatment_osimertinib']
BIN_FEATURES = ['sex_female', 'stage_iv', 'has_brain_mets',
                'egfr_mutation', 'kras_g12c', 'alk_fusion',
                'stk11_mutation', 'brca2_mutation', 'tmb_high',
                'smoke_current', 'smoke_never', 'histology_squamous',
                'ecog_ge2', 'pdl1_high']
CONT_FEATURES = ['age_years', 'pdl1_tps', 'albumin_g_dl', 'ldh_u_l',
                 'weight_loss_pct_6mo', 'crp_mg_l', 'nlr',
                 'hemoglobin_g_dl', 'alkaline_phosphatase_u_l', 'ast_u_l',
                 'alt_u_l', 'total_bilirubin_mg_dl', 'creatinine_mg_dl',
                 'bun_mg_dl', 'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl']

results = {}


def ttest_bin(feat, y='pfs_months'):
    a = df.loc[df[feat] == 1, y]
    b = df.loc[df[feat] == 0, y]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    eff = a.mean() - b.mean()
    return eff, p, a.mean(), b.mean(), len(a), len(b)


def corr_cont(feat, y='pfs_months'):
    r, p = stats.pearsonr(df[feat], df[y])
    return r, p


def ols_treatment(t, controls=None, sub=None):
    d = df if sub is None else df[sub]
    X_cols = [t]
    if controls:
        X_cols += controls
    X = sm.add_constant(d[X_cols])
    y = d[OUTCOME]
    m = sm.OLS(y, X).fit()
    return m.params[t], m.pvalues[t], m


def interaction(t, mod, sub=None):
    d = df if sub is None else df[sub]
    formula = f'{OUTCOME} ~ {t} * {mod}'
    m = smf.ols(formula, data=d).fit()
    name = f'{t}:{mod}'
    return m.params[name], m.pvalues[name], m


# =========================================================
# Iteration 1: Main effects of treatments
# =========================================================
print('\n=== Iteration 1: treatment main effects ===')
it1 = {'index': 1, 'proposed_hypotheses': [], 'analyses': []}
for i, t in enumerate(TREATMENTS, 1):
    eff, p, ma, mb, na, nb = ttest_bin(t)
    print(f'{t}: eff={eff:+.4f}, p={p:.3g}, on={ma:.2f}, off={mb:.2f}')
    hid = f'h1_{i}'
    it1['proposed_hypotheses'].append({
        'id': hid,
        'text': f'Patients receiving {t} have higher mean pfs_months than those not receiving it.',
        'kind': 'novel',
    })
    it1['analyses'].append({
        'hypothesis_ids': [hid],
        'code': f'stats.ttest_ind(df.loc[df["{t}"]==1,"pfs_months"], df.loc[df["{t}"]==0,"pfs_months"])',
        'result_summary': f'Mean pfs_months on {t}={ma:.3f} (n={na}) vs off={mb:.3f} (n={nb}); Welch t-test p={p:.3g}.',
        'p_value': float(p),
        'effect_estimate': float(eff),
        'significant': bool(p < 0.05),
    })
results['it1'] = it1

# =========================================================
# Iteration 2: Patient feature main effects on PFS
# =========================================================
print('\n=== Iteration 2: feature main effects ===')
it2 = {'index': 2, 'proposed_hypotheses': [], 'analyses': []}
for i, f in enumerate(BIN_FEATURES, 1):
    eff, p, ma, mb, na, nb = ttest_bin(f)
    print(f'{f}: eff={eff:+.4f}, p={p:.3g}')
    hid = f'h2_{i}'
    direction = 'higher' if eff > 0 else 'lower'
    it2['proposed_hypotheses'].append({
        'id': hid,
        'text': f'Patients with {f}=1 have {direction} mean pfs_months than those with {f}=0.',
        'kind': 'novel',
    })
    it2['analyses'].append({
        'hypothesis_ids': [hid],
        'code': f'stats.ttest_ind(df[df["{f}"]==1]["pfs_months"], df[df["{f}"]==0]["pfs_months"])',
        'result_summary': f'pfs_months for {f}=1: {ma:.3f} (n={na}); =0: {mb:.3f} (n={nb}); diff={eff:+.3f}; p={p:.3g}.',
        'p_value': float(p),
        'effect_estimate': float(eff),
        'significant': bool(p < 0.05),
    })
results['it2'] = it2

# =========================================================
# Iteration 3: Continuous feature correlations
# =========================================================
print('\n=== Iteration 3: continuous feature correlations ===')
it3 = {'index': 3, 'proposed_hypotheses': [], 'analyses': []}
for i, f in enumerate(CONT_FEATURES, 1):
    r, p = corr_cont(f)
    print(f'{f}: r={r:+.4f}, p={p:.3g}')
    hid = f'h3_{i}'
    direction = 'positively' if r > 0 else 'negatively'
    it3['proposed_hypotheses'].append({
        'id': hid,
        'text': f'{f} is {direction} associated with pfs_months.',
        'kind': 'novel',
    })
    it3['analyses'].append({
        'hypothesis_ids': [hid],
        'code': f'stats.pearsonr(df["{f}"], df["pfs_months"])',
        'result_summary': f'Pearson r({f}, pfs_months)={r:+.4f}, p={p:.3g}.',
        'p_value': float(p),
        'effect_estimate': float(r),
        'significant': bool(p < 0.05),
    })
results['it3'] = it3

with open('analysis_state.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('Saved iterations 1-3 to analysis_state.json')
