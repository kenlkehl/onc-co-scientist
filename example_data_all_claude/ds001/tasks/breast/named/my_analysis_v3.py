"""Iter 4-10: treatment x feature interactions screen."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet('dataset.parquet')
y = df['pfs_months'].values

with open('results_v2.json') as f:
    results = json.load(f)

binary_features = [
    'sex_female','stage_iv','has_brain_mets','node_positive','postmenopausal',
    'er_positive','pr_positive','her2_positive','her2_low',
    'brca1_mutation','brca2_mutation','pik3ca_mutation',
]
treatments = [
    'treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
    'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab',
]
ordinal_features = ['ecog_ps']
continuous_features = [
    'age_years','ki67_pct','tumor_size_cm','albumin_g_dl','ldh_u_l',
    'weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
    'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
    'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl',
]
covariates = binary_features + ordinal_features + continuous_features  # excluded other treatments

def interaction_test(treat, mod, kind):
    """Fit y ~ treat + mod + treat*mod (+covariates) and return interaction coef and p."""
    if kind == 'binary':
        # subgroup means
        m1_t1 = float(np.mean(y[(df[treat]==1) & (df[mod]==1)]))
        m1_t0 = float(np.mean(y[(df[treat]==0) & (df[mod]==1)]))
        m0_t1 = float(np.mean(y[(df[treat]==1) & (df[mod]==0)]))
        m0_t0 = float(np.mean(y[(df[treat]==0) & (df[mod]==0)]))
        eff_in_pos = m1_t1 - m1_t0
        eff_in_neg = m0_t1 - m0_t0
        # Fit OLS with covariates and the interaction
        cov_cols = [c for c in covariates if c != mod] + [t for t in treatments if t != treat]
        X = pd.DataFrame({
            'treat': df[treat].astype(float),
            'mod': df[mod].astype(float),
            'inter': (df[treat]*df[mod]).astype(float),
        })
        for c in cov_cols:
            X[c] = df[c].astype(float)
        Xv = sm.add_constant(X.values)
        m = sm.OLS(y, Xv).fit()
        # treat=col1, mod=col2, inter=col3
        return {
            'inter_coef': float(m.params[3]),
            'inter_p': float(m.pvalues[3]),
            'treat_coef': float(m.params[1]),  # effect when mod=0
            'eff_in_pos': eff_in_pos,
            'eff_in_neg': eff_in_neg,
            'n_pos_treated': int(((df[treat]==1)&(df[mod]==1)).sum()),
            'n_pos_untreated': int(((df[treat]==0)&(df[mod]==1)).sum()),
            'n_neg_treated': int(((df[treat]==1)&(df[mod]==0)).sum()),
            'n_neg_untreated': int(((df[treat]==0)&(df[mod]==0)).sum()),
        }
    elif kind == 'continuous':
        # Standardize modifier for stable interaction coef
        mvals = df[mod].astype(float).values
        mz = (mvals - mvals.mean()) / mvals.std()
        cov_cols = [c for c in covariates if c != mod] + [t for t in treatments if t != treat]
        X = pd.DataFrame({
            'treat': df[treat].astype(float),
            'mod_z': mz,
            'inter': df[treat].astype(float)*mz,
        })
        for c in cov_cols:
            X[c] = df[c].astype(float)
        Xv = sm.add_constant(X.values)
        m = sm.OLS(y, Xv).fit()
        # tertile-stratified treatment effect for direction sanity
        try:
            q = pd.qcut(mvals, 3, labels=['low','mid','high'], duplicates='drop')
            levels = ['low','mid','high']
        except ValueError:
            # Few unique values; group by raw level
            q = pd.Series(mvals).astype(int).astype(str)
            levels = sorted(q.unique())
        eff_by_t = {}
        for lvl in levels:
            mask = (q == lvl)
            t1 = y[mask & (df[treat]==1)]
            t0 = y[mask & (df[treat]==0)]
            if len(t1) > 0 and len(t0) > 0:
                eff_by_t[lvl] = float(np.mean(t1) - np.mean(t0))
            else:
                eff_by_t[lvl] = None
        return {
            'inter_coef_per_sd': float(m.params[3]),
            'inter_p': float(m.pvalues[3]),
            'treat_coef_at_mean': float(m.params[1]),
            'tertile_treatment_effects': eff_by_t,
        }
    elif kind == 'ordinal':
        return interaction_test(treat, mod, 'continuous')

results['interactions'] = {}
for t in treatments:
    results['interactions'][t] = {}
    for f in binary_features:
        results['interactions'][t][f] = interaction_test(t, f, 'binary')
    for f in ordinal_features:
        results['interactions'][t][f] = interaction_test(t, f, 'ordinal')
    for f in continuous_features:
        results['interactions'][t][f] = interaction_test(t, f, 'continuous')

# Save
with open('results_v2.json', 'w') as fp:
    json.dump(results, fp, indent=2)

# Print top interactions for each treatment
for t in treatments:
    print(f'\n=== {t} ===')
    rows = []
    for f, v in results['interactions'][t].items():
        p = v.get('inter_p')
        coef = v.get('inter_coef', v.get('inter_coef_per_sd'))
        rows.append((f, coef, p))
    rows.sort(key=lambda r: r[2])
    for f, coef, p in rows[:8]:
        print(f'  {f}: coef={coef:+.4f}, p={p:.3e}')
