"""Iterations 23-25: final consolidated subgroup hypotheses + comprehensive testing."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

df = pd.read_parquet('dataset.parquet')

results = {}


def subgroup_effect(df, treat, mask, label):
    sub = df[mask]
    on = sub.loc[sub[treat] == 1, 'pfs_months']
    off = sub.loc[sub[treat] == 0, 'pfs_months']
    out = {'label': label, 'n_on': int(len(on)), 'n_off': int(len(off))}
    if len(on) > 5 and len(off) > 5:
        tt = stats.ttest_ind(on, off, equal_var=False)
        out['mean_on'] = float(on.mean())
        out['mean_off'] = float(off.mean())
        out['diff'] = float(on.mean() - off.mean())
        out['p_value'] = float(tt.pvalue)
    return out


# --- Iter 23: Final palbociclib quad-marker hypothesis with several Ki67 thresholds ---
treat = 'treatment_palbociclib'
iter23 = {}
for ki67_th in [12, 13, 14, 15]:
    base = (df['er_positive'] == 1) & (df['her2_positive'] == 0) & (df['pik3ca_mutation'] == 0) & (df['ki67_pct'] < ki67_th)
    iter23[f'ER+/HER2-/PIK3CA-WT/Ki67<{ki67_th}'] = subgroup_effect(df, treat, base, f'quad_ki67<{ki67_th}')

# Compute fully adjusted interaction model with ki67<14 quad
sub = df.copy()
sub['quad'] = ((sub['er_positive'] == 1) & (sub['her2_positive'] == 0) &
               (sub['pik3ca_mutation'] == 0) & (sub['ki67_pct'] < 14)).astype(int)
sub['palbo_quad'] = sub[treat] * sub['quad']
adjust = ['age_years', 'ecog_ps', 'stage_iv', 'has_brain_mets', 'albumin_g_dl',
          'weight_loss_pct_6mo', 'ldh_u_l', 'pr_positive', 'her2_low',
          'brca1_mutation', 'brca2_mutation', 'postmenopausal', 'tumor_size_cm', 'crp_mg_l']
X = sm.add_constant(sub[[treat, 'quad', 'palbo_quad'] + adjust])
m = sm.OLS(sub['pfs_months'], X).fit()
iter23['adjusted_quad_ki67_lt_14'] = {
    'palbo_main_outside_quad': float(m.params[treat]),
    'palbo_main_p': float(m.pvalues[treat]),
    'inter_coef': float(m.params['palbo_quad']),
    'inter_p': float(m.pvalues['palbo_quad']),
    'effect_in_quad': float(m.params[treat] + m.params['palbo_quad']),
    'r_squared': float(m.rsquared),
    'n_quad': int(sub['quad'].sum()),
}

# Direct OLS in quad subgroup vs outside (sanity)
quad_mask = sub['quad'] == 1
on_q = sub.loc[quad_mask & (sub[treat] == 1), 'pfs_months'].values
off_q = sub.loc[quad_mask & (sub[treat] == 0), 'pfs_months'].values
on_oq = sub.loc[~quad_mask & (sub[treat] == 1), 'pfs_months'].values
off_oq = sub.loc[~quad_mask & (sub[treat] == 0), 'pfs_months'].values
iter23['raw_diff_quad_vs_outside'] = {
    'quad_mean_on': float(on_q.mean()), 'quad_mean_off': float(off_q.mean()),
    'quad_diff': float(on_q.mean() - off_q.mean()),
    'quad_p': float(stats.ttest_ind(on_q, off_q, equal_var=False).pvalue),
    'outside_mean_on': float(on_oq.mean()), 'outside_mean_off': float(off_oq.mean()),
    'outside_diff': float(on_oq.mean() - off_oq.mean()),
    'outside_p': float(stats.ttest_ind(on_oq, off_oq, equal_var=False).pvalue),
}
results['iter23_palbociclib_final_quad'] = iter23

# --- Iter 24: Olaparib in TNBC + BRCA subgroup, with adjustments and PR-null ---
treat = 'treatment_olaparib'
iter24 = {}
combos = [
    ('BRCA either AND ER- AND HER2- (TNBC+BRCA)',
        ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['er_positive'] == 0) & (df['her2_positive'] == 0)),
    ('BRCA either AND ER- AND HER2- AND PR-',
        ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['er_positive'] == 0) & (df['her2_positive'] == 0) & (df['pr_positive'] == 0)),
    ('BRCA either AND ER- AND HER2- AND no brain mets',
        ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['er_positive'] == 0) & (df['her2_positive'] == 0) & (df['has_brain_mets'] == 0)),
    ('BRCA1 only AND ER- AND HER2-',
        (df['brca1_mutation'] == 1) & (df['er_positive'] == 0) & (df['her2_positive'] == 0)),
    ('BRCA2 only AND ER- AND HER2-',
        (df['brca2_mutation'] == 1) & (df['er_positive'] == 0) & (df['her2_positive'] == 0)),
    ('BRCA either control: not BRCA',
        (df['brca1_mutation'] == 0) & (df['brca2_mutation'] == 0) & (df['er_positive'] == 0) & (df['her2_positive'] == 0)),
]
for label, mask in combos:
    iter24[label] = subgroup_effect(df, treat, mask, label)

# Adjusted interaction in TNBC subset only
tnbc = df[(df['er_positive'] == 0) & (df['her2_positive'] == 0)].copy()
tnbc['brca_any'] = ((tnbc['brca1_mutation'] == 1) | (tnbc['brca2_mutation'] == 1)).astype(int)
tnbc['olap_brca'] = tnbc[treat] * tnbc['brca_any']
adjust = ['age_years', 'ecog_ps', 'stage_iv', 'has_brain_mets', 'albumin_g_dl',
          'weight_loss_pct_6mo', 'ldh_u_l', 'pik3ca_mutation', 'ki67_pct', 'pr_positive']
X = sm.add_constant(tnbc[[treat, 'brca_any', 'olap_brca'] + adjust])
m = sm.OLS(tnbc['pfs_months'], X).fit()
iter24['adjusted_interaction_in_TNBC'] = {
    'inter_coef': float(m.params['olap_brca']),
    'inter_p': float(m.pvalues['olap_brca']),
    'olap_main_in_brca_neg_tnbc': float(m.params[treat]),
    'effect_in_brca_pos_tnbc': float(m.params[treat] + m.params['olap_brca']),
    'n_tnbc': int(len(tnbc)),
}
results['iter24_olaparib_final_subgroup'] = iter24

# --- Iter 25: Final null-confirmation tests for non-effective treatments and exhaustive 4-way for palbociclib ---
iter25 = {}
# Confirm null adjusted effects for trastuzumab/pembro/sacit/tamoxifen across populations
for treat, restrict_label, restrict_mask in [
    ('treatment_trastuzumab', 'all', np.ones(len(df), bool)),
    ('treatment_trastuzumab', 'HER2+', df['her2_positive'] == 1),
    ('treatment_pembrolizumab', 'all', np.ones(len(df), bool)),
    ('treatment_pembrolizumab', 'TNBC', (df['er_positive'] == 0) & (df['her2_positive'] == 0)),
    ('treatment_sacituzumab_govitecan', 'all', np.ones(len(df), bool)),
    ('treatment_sacituzumab_govitecan', 'TNBC', (df['er_positive'] == 0) & (df['her2_positive'] == 0)),
    ('treatment_sacituzumab_govitecan', 'HER2-low', df['her2_low'] == 1),
    ('treatment_tamoxifen', 'all', np.ones(len(df), bool)),
    ('treatment_tamoxifen', 'ER+', df['er_positive'] == 1),
]:
    sub = df[restrict_mask].copy()
    adjust = ['age_years', 'ecog_ps', 'stage_iv', 'has_brain_mets', 'albumin_g_dl',
              'weight_loss_pct_6mo', 'ldh_u_l', 'er_positive', 'her2_positive',
              'pik3ca_mutation', 'ki67_pct', 'brca1_mutation', 'brca2_mutation',
              'her2_low', 'postmenopausal', 'pr_positive']
    keep = [a for a in adjust if sub[a].nunique() > 1]
    X = sm.add_constant(sub[[treat] + keep])
    mm = sm.OLS(sub['pfs_months'], X).fit()
    iter25[f'{treat}_in_{restrict_label}'] = {
        'coef': float(mm.params[treat]),
        'p': float(mm.pvalues[treat]),
        'se': float(mm.bse[treat]),
        'n': int(mm.nobs),
    }

# Exhaustive 4-way subgroup search (only for palbociclib) to verify quad is the best
treat = 'treatment_palbociclib'
binary_modifiers = ['sex_female', 'stage_iv', 'has_brain_mets', 'node_positive',
                    'postmenopausal', 'er_positive', 'pr_positive', 'her2_positive',
                    'her2_low', 'brca1_mutation', 'brca2_mutation', 'pik3ca_mutation']
# Add ki67<14 as boolean
df['ki67_low14'] = (df['ki67_pct'] < 14).astype(int)
binary_modifiers_aug = binary_modifiers + ['ki67_low14']
pos_modifiers = [(m, df[m]) for m in binary_modifiers_aug]
neg_modifiers = [(f'NOT_{m}', 1 - df[m]) for m in binary_modifiers_aug]
all_modifiers = pos_modifiers + neg_modifiers

import itertools as it
top4 = []
for combo in it.combinations(all_modifiers, 4):
    names = [c[0] for c in combo]
    mask = np.ones(len(df), bool)
    for _, vals in combo:
        mask &= (vals == 1)
    sub = df[mask]
    on = sub.loc[sub[treat] == 1, 'pfs_months']
    off = sub.loc[sub[treat] == 0, 'pfs_months']
    if len(on) < 200 or len(off) < 400:
        continue
    diff = on.mean() - off.mean()
    tt = stats.ttest_ind(on, off, equal_var=False)
    top4.append({
        'subgroup': ' AND '.join(names),
        'n_on': int(len(on)), 'n_off': int(len(off)),
        'diff': float(diff), 'p': float(tt.pvalue), 't': float(tt.statistic),
    })
top4.sort(key=lambda r: -r['diff'])
iter25['palbociclib_top10_4way_subgroups'] = top4[:10]
iter25['palbociclib_count_searched_subgroups'] = len(top4)
results['iter25_final_consolidated'] = iter25

with open('iters_23_25_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("=== Iter 23: palbociclib final quad ===")
print(json.dumps(iter23, indent=2))

print("\n=== Iter 24: olaparib final subgroup ===")
print(json.dumps(iter24, indent=2))

print("\n=== Iter 25: null confirmations + exhaustive 4-way for palbociclib ===")
print("\n-- Adjusted main effects (null tests) --")
for k, v in iter25.items():
    if k.startswith('treatment'):
        print(f"  {k}: coef={v['coef']:+.4f} (se {v['se']:.4f}) p={v['p']:.3e} n={v['n']}")
print(f"\n-- Top 10 4-way palbociclib subgroups (out of {iter25['palbociclib_count_searched_subgroups']} searched) --")
for r in iter25['palbociclib_top10_4way_subgroups']:
    print(f"  [{r['subgroup']}] n_on={r['n_on']} n_off={r['n_off']} diff={r['diff']:+.3f} p={r['p']:.2e}")
