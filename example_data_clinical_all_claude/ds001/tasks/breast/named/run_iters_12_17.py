"""Iterations 12-17: joint subgroup models + exhaustive 2-/3-way subgroup discovery."""
import json
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

df = pd.read_parquet('dataset.parquet')

results = {}
ki67_med = df['ki67_pct'].median()


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


# --- Iter 12: Palbociclib joint subgroup test ---
treat = 'treatment_palbociclib'
iter12 = []
combos = [
    ('all', np.ones(len(df), bool)),
    ('ER+', df['er_positive'] == 1),
    ('PIK3CA-WT', df['pik3ca_mutation'] == 0),
    ('HER2-', df['her2_positive'] == 0),
    ('ER+ AND HER2-', (df['er_positive'] == 1) & (df['her2_positive'] == 0)),
    ('ER+ AND PIK3CA-WT', (df['er_positive'] == 1) & (df['pik3ca_mutation'] == 0)),
    ('ER+ AND HER2- AND PIK3CA-WT',
        (df['er_positive'] == 1) & (df['her2_positive'] == 0) & (df['pik3ca_mutation'] == 0)),
    ('ER+ AND HER2- AND PIK3CA-WT AND ki67<med',
        (df['er_positive'] == 1) & (df['her2_positive'] == 0) & (df['pik3ca_mutation'] == 0) & (df['ki67_pct'] < ki67_med)),
    ('ER+ AND HER2- AND PIK3CA-WT AND ki67>=med',
        (df['er_positive'] == 1) & (df['her2_positive'] == 0) & (df['pik3ca_mutation'] == 0) & (df['ki67_pct'] >= ki67_med)),
    ('ER- (negative control)', df['er_positive'] == 0),
    ('PIK3CA-mut (negative control)', df['pik3ca_mutation'] == 1),
    ('HER2+ (negative control)', df['her2_positive'] == 1),
]
for label, mask in combos:
    iter12.append(subgroup_effect(df, treat, mask, label))
results['iter12_palbociclib_joint_subgroups'] = iter12

# --- Iter 13: Three-way and four-way interaction models for palbociclib ---
sub = df[[treat, 'er_positive', 'her2_positive', 'pik3ca_mutation', 'ki67_pct', 'pfs_months']].copy()
sub['her2_neg'] = 1 - sub['her2_positive']
sub['pik3ca_wt'] = 1 - sub['pik3ca_mutation']
sub['lowKi67'] = (sub['ki67_pct'] < ki67_med).astype(int)
sub['triple_marker'] = sub['er_positive'] * sub['her2_neg'] * sub['pik3ca_wt']
sub['quad_marker'] = sub['triple_marker'] * sub['lowKi67']
sub['inter_triple'] = sub[treat] * sub['triple_marker']
sub['inter_quad'] = sub[treat] * sub['quad_marker']
X = sm.add_constant(sub[[treat, 'triple_marker', 'inter_triple']])
m = sm.OLS(sub['pfs_months'], X).fit()
iter13_triple = {
    'inter_coef': float(m.params['inter_triple']),
    'inter_p': float(m.pvalues['inter_triple']),
    'treat_main_outside': float(m.params[treat]),
    'treat_main_p_outside': float(m.pvalues[treat]),
    'effect_in_triple': float(m.params[treat] + m.params['inter_triple']),
}
X2 = sm.add_constant(sub[[treat, 'quad_marker', 'inter_quad']])
m2 = sm.OLS(sub['pfs_months'], X2).fit()
iter13_quad = {
    'inter_coef': float(m2.params['inter_quad']),
    'inter_p': float(m2.pvalues['inter_quad']),
    'treat_main_outside': float(m2.params[treat]),
    'treat_main_p_outside': float(m2.pvalues[treat]),
    'effect_in_quad': float(m2.params[treat] + m2.params['inter_quad']),
}
results['iter13_palbociclib_joint_interaction_models'] = {
    'triple_ER+_HER2-_PIK3CA-WT': iter13_triple,
    'quad_ER+_HER2-_PIK3CA-WT_lowKi67': iter13_quad,
}

# --- Iter 14: Exhaustive pairwise+triple subgroup discovery for each treatment ---
# For each treatment, search all combinations of 1, 2, 3 binary modifiers (positive direction)
# defining a subgroup, and rank by interaction t-statistic.
binary_modifiers = ['sex_female', 'stage_iv', 'has_brain_mets', 'node_positive',
                    'postmenopausal', 'er_positive', 'pr_positive', 'her2_positive',
                    'her2_low', 'brca1_mutation', 'brca2_mutation', 'pik3ca_mutation']
# Augment with negations (e.g., HER2- = not her2_positive)
neg_modifiers = [(f'NOT_{m}', 1 - df[m]) for m in binary_modifiers]
pos_modifiers = [(m, df[m]) for m in binary_modifiers]
all_modifiers = pos_modifiers + neg_modifiers


def search_subgroups(df, treat, modifiers, sizes=(1, 2, 3), min_n_per_arm=50):
    out = []
    for size in sizes:
        for combo in itertools.combinations(modifiers, size):
            names = [c[0] for c in combo]
            mask = np.ones(len(df), bool)
            for _, vals in combo:
                mask &= (vals == 1)
            sub = df[mask]
            on = sub.loc[sub[treat] == 1, 'pfs_months']
            off = sub.loc[sub[treat] == 0, 'pfs_months']
            if len(on) < min_n_per_arm or len(off) < min_n_per_arm:
                continue
            diff = on.mean() - off.mean()
            tt = stats.ttest_ind(on, off, equal_var=False)
            out.append({
                'subgroup': ' AND '.join(names), 'size': size,
                'n_on': int(len(on)), 'n_off': int(len(off)),
                'mean_on': float(on.mean()), 'mean_off': float(off.mean()),
                'diff': float(diff), 'p_value': float(tt.pvalue),
                't': float(tt.statistic),
            })
    return out


iter14 = {}
for t in ['treatment_palbociclib', 'treatment_olaparib', 'treatment_sacituzumab_govitecan',
          'treatment_pembrolizumab', 'treatment_trastuzumab', 'treatment_tamoxifen']:
    res = search_subgroups(df, t, all_modifiers, sizes=(1, 2, 3), min_n_per_arm=50)
    res.sort(key=lambda r: -r['t'])
    iter14[t] = {
        'top_positive_uplift': res[:8],
        'top_negative_uplift': sorted(res, key=lambda r: r['t'])[:8],
    }
results['iter14_exhaustive_subgroup_search'] = iter14

# --- Iter 15: Olaparib subgroup deep-dive ---
treat = 'treatment_olaparib'
iter15 = []
combos15 = [
    ('all', np.ones(len(df), bool)),
    ('BRCA1+', df['brca1_mutation'] == 1),
    ('BRCA2+', df['brca2_mutation'] == 1),
    ('BRCA1+ or BRCA2+', (df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)),
    ('BRCA1- AND BRCA2-', (df['brca1_mutation'] == 0) & (df['brca2_mutation'] == 0)),
    ('BRCA either AND ER+', ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['er_positive'] == 1)),
    ('BRCA either AND HER2-', ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['her2_positive'] == 0)),
    ('BRCA either AND TNBC',
        ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['er_positive'] == 0) & (df['her2_positive'] == 0)),
    ('BRCA either AND no brain mets',
        ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['has_brain_mets'] == 0)),
]
for label, mask in combos15:
    iter15.append(subgroup_effect(df, treat, mask, label))

df['brca_any'] = ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)).astype(int)
sub = df[[treat, 'brca_any', 'pfs_months']].copy()
sub['inter'] = sub[treat] * sub['brca_any']
X = sm.add_constant(sub[[treat, 'brca_any', 'inter']])
m = sm.OLS(sub['pfs_months'], X).fit()
iter15_inter = {
    'brca_any_inter_coef': float(m.params['inter']),
    'brca_any_inter_p': float(m.pvalues['inter']),
    'olaparib_main_in_brca_neg': float(m.params[treat]),
    'olaparib_main_in_brca_neg_p': float(m.pvalues[treat]),
    'olaparib_effect_in_brca_pos': float(m.params[treat] + m.params['inter']),
}
results['iter15_olaparib_brca_subgroups'] = {'subgroup_means': iter15, 'interaction_test': iter15_inter}

# --- Iter 16: Trastuzumab subgroup analyses (HER2+ only) ---
treat = 'treatment_trastuzumab'
iter16 = []
combos16 = [
    ('all', np.ones(len(df), bool)),
    ('HER2+', df['her2_positive'] == 1),
    ('HER2- (negative control)', df['her2_positive'] == 0),
    ('HER2+ AND ER+', (df['her2_positive'] == 1) & (df['er_positive'] == 1)),
    ('HER2+ AND ER-', (df['her2_positive'] == 1) & (df['er_positive'] == 0)),
    ('HER2+ AND postmenopausal', (df['her2_positive'] == 1) & (df['postmenopausal'] == 1)),
    ('HER2+ AND stage_iv', (df['her2_positive'] == 1) & (df['stage_iv'] == 1)),
    ('HER2+ AND no brain mets', (df['her2_positive'] == 1) & (df['has_brain_mets'] == 0)),
    ('HER2+ AND node positive', (df['her2_positive'] == 1) & (df['node_positive'] == 1)),
    ('HER2+ AND PIK3CA-WT', (df['her2_positive'] == 1) & (df['pik3ca_mutation'] == 0)),
]
for label, mask in combos16:
    iter16.append(subgroup_effect(df, treat, mask, label))
results['iter16_trastuzumab_subgroups'] = iter16

# --- Iter 17: Pembrolizumab and sacituzumab subgroups ---
iter17 = {}
for treat in ['treatment_pembrolizumab', 'treatment_sacituzumab_govitecan']:
    rows = []
    combos17 = [
        ('all', np.ones(len(df), bool)),
        ('TNBC: ER- AND HER2-', (df['er_positive'] == 0) & (df['her2_positive'] == 0)),
        ('TNBC AND ki67_high', (df['er_positive'] == 0) & (df['her2_positive'] == 0) & (df['ki67_pct'] >= ki67_med)),
        ('TNBC AND PIK3CA-WT',
            (df['er_positive'] == 0) & (df['her2_positive'] == 0) & (df['pik3ca_mutation'] == 0)),
        ('HER2 low', df['her2_low'] == 1),
        ('HER2 low AND ER-', (df['her2_low'] == 1) & (df['er_positive'] == 0)),
        ('PD-like: high nlr', df['nlr'] >= df['nlr'].median()),
        ('Visceral: brain mets', df['has_brain_mets'] == 1),
        ('Stage IV', df['stage_iv'] == 1),
        ('Stage IV AND TNBC', (df['stage_iv'] == 1) & (df['er_positive'] == 0) & (df['her2_positive'] == 0)),
        ('PIK3CA mutation', df['pik3ca_mutation'] == 1),
    ]
    for label, mask in combos17:
        rows.append(subgroup_effect(df, treat, mask, label))
    iter17[treat] = rows
results['iter17_pembro_sacituzumab_subgroups'] = iter17

with open('iters_12_17_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Print key findings
print("=== Iter 12 palbociclib joint subgroups ===")
for r in results['iter12_palbociclib_joint_subgroups']:
    if 'diff' in r:
        print(f"  {r['label']}: n_on={r['n_on']} n_off={r['n_off']} diff={r['diff']:+.3f} p={r['p_value']:.2e}")

print("\n=== Iter 13 joint interaction models ===")
print(json.dumps(results['iter13_palbociclib_joint_interaction_models'], indent=2))

print("\n=== Iter 14 exhaustive subgroup search (top 5 positive per treatment) ===")
for t, info in results['iter14_exhaustive_subgroup_search'].items():
    print(f"\n{t}:")
    print("  TOP POSITIVE (largest treatment - control PFS difference):")
    for r in info['top_positive_uplift'][:5]:
        print(f"    [{r['subgroup']}] n_on={r['n_on']} n_off={r['n_off']} diff={r['diff']:+.3f} p={r['p_value']:.2e}")
    print("  TOP NEGATIVE:")
    for r in info['top_negative_uplift'][:3]:
        print(f"    [{r['subgroup']}] n_on={r['n_on']} n_off={r['n_off']} diff={r['diff']:+.3f} p={r['p_value']:.2e}")

print("\n=== Iter 15 olaparib BRCA subgroups ===")
for r in results['iter15_olaparib_brca_subgroups']['subgroup_means']:
    if 'diff' in r:
        print(f"  {r['label']}: n_on={r['n_on']} n_off={r['n_off']} diff={r['diff']:+.3f} p={r['p_value']:.2e}")
print("Interaction test:", results['iter15_olaparib_brca_subgroups']['interaction_test'])

print("\n=== Iter 16 trastuzumab subgroups ===")
for r in results['iter16_trastuzumab_subgroups']:
    if 'diff' in r:
        print(f"  {r['label']}: n_on={r['n_on']} n_off={r['n_off']} diff={r['diff']:+.3f} p={r['p_value']:.2e}")

print("\n=== Iter 17 pembro/sacituzumab ===")
for treat, rows in results['iter17_pembro_sacituzumab_subgroups'].items():
    print(f"\n  {treat}:")
    for r in rows:
        if 'diff' in r:
            print(f"    {r['label']}: n_on={r['n_on']} n_off={r['n_off']} diff={r['diff']:+.3f} p={r['p_value']:.2e}")
