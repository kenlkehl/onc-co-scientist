"""Iter 13-20: systematic treatment-effect heterogeneity search."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
out = 'pfs_months'
df['brca_any'] = ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)).astype(int)
df['triple_neg'] = ((df['er_positive'] == 0) & (df['pr_positive'] == 0) & (df['her2_positive'] == 0)).astype(int)
df['ki67_high'] = (df['ki67_pct'] >= 20).astype(int)
df['age_ge65'] = (df['age_years'] >= 65).astype(int)
df['albumin_low'] = (df['albumin_g_dl'] < 3.5).astype(int)
df['ldh_high'] = (df['ldh_u_l'] > 240).astype(int)
df['weight_loss_5p'] = (df['weight_loss_pct_6mo'] >= 5).astype(int)

binary_modifiers = [
    'age_ge65', 'sex_female', 'stage_iv', 'has_brain_mets', 'node_positive',
    'postmenopausal', 'er_positive', 'pr_positive', 'her2_positive', 'her2_low',
    'brca1_mutation', 'brca2_mutation', 'brca_any', 'pik3ca_mutation',
    'triple_neg', 'ki67_high', 'albumin_low', 'ldh_high', 'weight_loss_5p',
]
ecog_levels = sorted(df['ecog_ps'].unique())  # 0,1,2

treatments = ['treatment_tamoxifen', 'treatment_palbociclib', 'treatment_trastuzumab',
              'treatment_olaparib', 'treatment_sacituzumab_govitecan', 'treatment_pembrolizumab']

results = {}


def stratified_diff(treat, mod_col, mod_val):
    sub = df[df[mod_col] == mod_val]
    if len(sub) < 50:
        return None
    a = sub.loc[sub[treat] == 1, out]
    b = sub.loc[sub[treat] == 0, out]
    if len(a) < 20 or len(b) < 20:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
        'diff': float(a.mean() - b.mean()),
        'n_tx': int(len(a)), 'n_ctrl': int(len(b)),
        'p': float(p),
    }


def interaction_test(treat, mod_col):
    f = f'{out} ~ {treat} * {mod_col}'
    m = smf.ols(f, data=df).fit()
    inter = f'{treat}:{mod_col}'
    return {
        'main_treat_coef': float(m.params[treat]),
        'main_treat_p': float(m.pvalues[treat]),
        'main_mod_coef': float(m.params[mod_col]),
        'main_mod_p': float(m.pvalues[mod_col]),
        'inter_coef': float(m.params[inter]),
        'inter_p': float(m.pvalues[inter]),
    }


print('=== ITER 13: Heterogeneity scan — treatment x each binary modifier ===')
het = {}
for t in treatments:
    het[t] = {}
    for mod in binary_modifiers:
        try:
            it = interaction_test(t, mod)
            het[t][mod] = it
        except Exception as e:
            het[t][mod] = {'error': str(e)}
results['iter13_heterogeneity_scan'] = het

# print top interactions
print('\nTop interactions by p-value:')
flat = []
for t, mods in het.items():
    for mod, it in mods.items():
        if 'inter_p' in it:
            flat.append((t, mod, it['inter_coef'], it['inter_p']))
flat.sort(key=lambda x: x[3])
for row in flat[:30]:
    print(f'  {row[0]:35s} x {row[1]:20s} coef={row[2]:+.3f}  p={row[3]:.3g}')

print('\n=== ITER 14: Continuous-modifier interactions (treatment x ki67, x albumin, x ldh) ===')
cont = {}
for t in treatments:
    cont[t] = {}
    for mod in ['ki67_pct', 'albumin_g_dl', 'ldh_u_l', 'weight_loss_pct_6mo', 'tumor_size_cm', 'age_years']:
        try:
            f = f'{out} ~ {t} * {mod}'
            m = smf.ols(f, data=df).fit()
            inter = f'{t}:{mod}'
            cont[t][mod] = {
                'inter_coef': float(m.params[inter]),
                'inter_p': float(m.pvalues[inter]),
            }
        except Exception as e:
            cont[t][mod] = {'error': str(e)}
results['iter14_cont_modifiers'] = cont
flat2 = []
for t, mods in cont.items():
    for mod, it in mods.items():
        if 'inter_p' in it:
            flat2.append((t, mod, it['inter_coef'], it['inter_p']))
flat2.sort(key=lambda x: x[3])
for row in flat2[:25]:
    print(f'  {row[0]:35s} x {row[1]:25s} coef={row[2]:+.4f}  p={row[3]:.3g}')

print('\n=== ITER 15: Drill into palbociclib heterogeneity (best driver) ===')
# Palbociclib is the main beneficial drug; find what modulates it.
# Combined ER+/HER2-/postmeno already showed +1.87. Try further refinements.
def detail(treat, conds, label):
    mask = np.ones(len(df), dtype=bool)
    for c, v in conds.items():
        mask &= (df[c] == v)
    sub = df[mask]
    a = sub.loc[sub[treat] == 1, out]
    b = sub.loc[sub[treat] == 0, out]
    if len(a) < 5 or len(b) < 5:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        'label': label,
        'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
        'diff': float(a.mean() - b.mean()),
        'n_tx': int(len(a)), 'n_ctrl': int(len(b)),
        'p': float(p),
    }


palbo_subgroups = [
    detail('treatment_palbociclib', {'er_positive': 1}, 'ER+'),
    detail('treatment_palbociclib', {'er_positive': 0}, 'ER-'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0}, 'ER+/HER2-'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 1}, 'ER+/HER2-/postmeno'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 1, 'pik3ca_mutation': 0}, 'ER+/HER2-/postmeno/PIK3CA-'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 1, 'pik3ca_mutation': 1}, 'ER+/HER2-/postmeno/PIK3CA+'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 0}, 'ER+/HER2-/premeno'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 1}, 'ER+/HER2+'),
    detail('treatment_palbociclib', {'her2_positive': 1}, 'HER2+'),
    detail('treatment_palbociclib', {'pik3ca_mutation': 1}, 'PIK3CA+'),
    detail('treatment_palbociclib', {'pik3ca_mutation': 0}, 'PIK3CA-'),
    detail('treatment_palbociclib', {'er_positive': 1, 'pik3ca_mutation': 0}, 'ER+/PIK3CA-'),
    detail('treatment_palbociclib', {'ecog_ps': 0}, 'ECOG 0'),
    detail('treatment_palbociclib', {'ecog_ps': 1}, 'ECOG 1'),
    detail('treatment_palbociclib', {'ecog_ps': 2}, 'ECOG 2'),
    detail('treatment_palbociclib', {'stage_iv': 1}, 'stage IV'),
    detail('treatment_palbociclib', {'stage_iv': 0}, 'not stage IV'),
]
results['iter15_palbo_subgroups'] = palbo_subgroups
for r in palbo_subgroups:
    if r:
        print(f"  palbo in {r['label']:40s}: diff={r['diff']:+.3f} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}, p={r['p']:.3g})")

print('\n=== ITER 16: Drill into pembrolizumab x triple_neg and other modifiers ===')
pembro_subgroups = [
    detail('treatment_pembrolizumab', {'triple_neg': 1}, 'TNBC'),
    detail('treatment_pembrolizumab', {'triple_neg': 0}, 'not TNBC'),
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'ki67_high': 1}, 'TNBC/ki67-high'),
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'pik3ca_mutation': 0}, 'TNBC/PIK3CA-'),
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'pik3ca_mutation': 1}, 'TNBC/PIK3CA+'),
    detail('treatment_pembrolizumab', {'er_positive': 0, 'her2_positive': 0}, 'ER-/HER2-'),
    detail('treatment_pembrolizumab', {'er_positive': 0}, 'ER-'),
    detail('treatment_pembrolizumab', {'er_positive': 1}, 'ER+'),
    detail('treatment_pembrolizumab', {'her2_positive': 1}, 'HER2+'),
    detail('treatment_pembrolizumab', {'her2_positive': 0}, 'HER2-'),
    detail('treatment_pembrolizumab', {'ki67_high': 1}, 'ki67_high'),
    detail('treatment_pembrolizumab', {'ki67_high': 0}, 'ki67_low'),
    detail('treatment_pembrolizumab', {'stage_iv': 1}, 'stage IV'),
    detail('treatment_pembrolizumab', {'has_brain_mets': 1}, 'brain mets'),
]
results['iter16_pembro_subgroups'] = pembro_subgroups
for r in pembro_subgroups:
    if r:
        print(f"  pembro in {r['label']:35s}: diff={r['diff']:+.3f} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}, p={r['p']:.3g})")

print('\n=== ITER 17: Drill into olaparib x BRCA & friends ===')
olap_subgroups = [
    detail('treatment_olaparib', {'brca1_mutation': 1}, 'BRCA1+'),
    detail('treatment_olaparib', {'brca2_mutation': 1}, 'BRCA2+'),
    detail('treatment_olaparib', {'brca_any': 1}, 'BRCA1/2+'),
    detail('treatment_olaparib', {'brca_any': 0}, 'BRCA wild-type'),
    detail('treatment_olaparib', {'brca_any': 1, 'her2_positive': 0}, 'BRCA+/HER2-'),
    detail('treatment_olaparib', {'brca_any': 1, 'er_positive': 1}, 'BRCA+/ER+'),
    detail('treatment_olaparib', {'brca_any': 1, 'er_positive': 0}, 'BRCA+/ER-'),
    detail('treatment_olaparib', {'brca_any': 1, 'triple_neg': 1}, 'BRCA+/TNBC'),
    detail('treatment_olaparib', {'pik3ca_mutation': 1}, 'PIK3CA+'),
]
results['iter17_olap_subgroups'] = olap_subgroups
for r in olap_subgroups:
    if r:
        print(f"  olaparib in {r['label']:25s}: diff={r['diff']:+.3f} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}, p={r['p']:.3g})")

print('\n=== ITER 18: Drill into trastuzumab x HER2 / ER ===')
trast_subgroups = [
    detail('treatment_trastuzumab', {'her2_positive': 1}, 'HER2+'),
    detail('treatment_trastuzumab', {'her2_positive': 0}, 'HER2-'),
    detail('treatment_trastuzumab', {'her2_positive': 1, 'er_positive': 1}, 'HER2+/ER+'),
    detail('treatment_trastuzumab', {'her2_positive': 1, 'er_positive': 0}, 'HER2+/ER-'),
    detail('treatment_trastuzumab', {'her2_positive': 1, 'stage_iv': 0}, 'HER2+/non-stage IV'),
    detail('treatment_trastuzumab', {'her2_positive': 1, 'stage_iv': 1}, 'HER2+/stage IV'),
    detail('treatment_trastuzumab', {'her2_low': 1}, 'HER2-low'),
]
results['iter18_trast_subgroups'] = trast_subgroups
for r in trast_subgroups:
    if r:
        print(f"  trast in {r['label']:30s}: diff={r['diff']:+.3f} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}, p={r['p']:.3g})")

print('\n=== ITER 19: Tamoxifen and sacituzumab subgroup drill ===')
tamox_subgroups = [
    detail('treatment_tamoxifen', {'er_positive': 1}, 'ER+'),
    detail('treatment_tamoxifen', {'er_positive': 1, 'postmenopausal': 1}, 'ER+/postmeno'),
    detail('treatment_tamoxifen', {'er_positive': 1, 'postmenopausal': 0}, 'ER+/premeno'),
    detail('treatment_tamoxifen', {'er_positive': 1, 'her2_positive': 0}, 'ER+/HER2-'),
    detail('treatment_tamoxifen', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 0}, 'ER+/HER2-/premeno'),
    detail('treatment_tamoxifen', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 1}, 'ER+/HER2-/postmeno'),
    detail('treatment_tamoxifen', {'er_positive': 1, 'pr_positive': 1}, 'ER+/PR+'),
    detail('treatment_tamoxifen', {'pr_positive': 1}, 'PR+'),
]
results['iter19_tamox_subgroups'] = tamox_subgroups
for r in tamox_subgroups:
    if r:
        print(f"  tamox in {r['label']:30s}: diff={r['diff']:+.3f} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}, p={r['p']:.3g})")

sacit_subgroups = [
    detail('treatment_sacituzumab_govitecan', {'triple_neg': 1}, 'TNBC'),
    detail('treatment_sacituzumab_govitecan', {'her2_low': 1}, 'HER2-low'),
    detail('treatment_sacituzumab_govitecan', {'her2_low': 1, 'er_positive': 1}, 'HER2-low/ER+'),
    detail('treatment_sacituzumab_govitecan', {'her2_low': 1, 'triple_neg': 1}, 'HER2-low/TNBC'),
    detail('treatment_sacituzumab_govitecan', {'er_positive': 0, 'her2_positive': 0}, 'ER-/HER2-'),
    detail('treatment_sacituzumab_govitecan', {'her2_positive': 0}, 'HER2-'),
]
results['iter19_sacit_subgroups'] = sacit_subgroups
for r in sacit_subgroups:
    if r:
        print(f"  sacit in {r['label']:30s}: diff={r['diff']:+.3f} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}, p={r['p']:.3g})")

print('\n=== ITER 20: Top 3-way subgroups for palbociclib (further refinement) ===')
# Already saw ER+/HER2-/postmeno is best. Now narrow further.
extras = [
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 1, 'stage_iv': 0}, 'ER+/HER2-/postmeno/non-stage-IV'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 1, 'stage_iv': 1}, 'ER+/HER2-/postmeno/stage-IV'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 1, 'has_brain_mets': 0}, 'ER+/HER2-/postmeno/no-brain-mets'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 1, 'ecog_ps': 0}, 'ER+/HER2-/postmeno/ECOG=0'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 1, 'ecog_ps': 1}, 'ER+/HER2-/postmeno/ECOG=1'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'postmenopausal': 1, 'ecog_ps': 2}, 'ER+/HER2-/postmeno/ECOG=2'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'pik3ca_mutation': 0}, 'ER+/HER2-/PIK3CA-'),
    detail('treatment_palbociclib', {'er_positive': 1, 'her2_positive': 0, 'pik3ca_mutation': 1}, 'ER+/HER2-/PIK3CA+'),
]
for r in extras:
    if r:
        print(f"  palbo in {r['label']:45s}: diff={r['diff']:+.3f} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}, p={r['p']:.3g})")
results['iter20_palbo_extras'] = extras

with open('analysis_part3.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('Saved analysis_part3.json')
