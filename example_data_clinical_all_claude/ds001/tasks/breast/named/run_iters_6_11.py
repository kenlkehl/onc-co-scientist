"""Iterations 6-11: per-treatment interaction screening.
For each of 6 treatments, screen interactions with each candidate modifier.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_parquet('dataset.parquet')

treatments = [
    'treatment_tamoxifen', 'treatment_palbociclib', 'treatment_trastuzumab',
    'treatment_olaparib', 'treatment_sacituzumab_govitecan', 'treatment_pembrolizumab'
]
binary_modifiers = ['sex_female', 'stage_iv', 'has_brain_mets', 'node_positive',
                    'postmenopausal', 'er_positive', 'pr_positive', 'her2_positive',
                    'her2_low', 'brca1_mutation', 'brca2_mutation', 'pik3ca_mutation']

cont_modifiers = ['age_years', 'ecog_ps', 'ki67_pct', 'tumor_size_cm', 'albumin_g_dl',
                  'ldh_u_l', 'weight_loss_pct_6mo', 'crp_mg_l', 'nlr', 'hemoglobin_g_dl',
                  'alkaline_phosphatase_u_l', 'ast_u_l', 'alt_u_l',
                  'total_bilirubin_mg_dl', 'creatinine_mg_dl', 'bun_mg_dl',
                  'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl']


def interaction_test(df, treat, modifier, modifier_is_binary):
    """Test treat x modifier interaction with main effect adjustment."""
    sub = df[[treat, modifier, 'pfs_months']].copy()
    sub['inter'] = sub[treat] * sub[modifier]
    X = sm.add_constant(sub[[treat, modifier, 'inter']])
    m = sm.OLS(sub['pfs_months'], X).fit()
    out = {
        'inter_coef': float(m.params['inter']),
        'inter_p': float(m.pvalues['inter']),
        'treat_main': float(m.params[treat]),
        'treat_main_p': float(m.pvalues[treat]),
    }
    if modifier_is_binary:
        eff_pos = sub.loc[sub[modifier] == 1].assign(d=lambda d: d.pfs_months).groupby(treat)['d'].mean()
        eff_neg = sub.loc[sub[modifier] == 0].assign(d=lambda d: d.pfs_months).groupby(treat)['d'].mean()
        out['eff_in_pos'] = float(eff_pos.get(1, np.nan) - eff_pos.get(0, np.nan))
        out['eff_in_neg'] = float(eff_neg.get(1, np.nan) - eff_neg.get(0, np.nan))
        out['n_pos'] = int((sub[modifier] == 1).sum())
        out['n_neg'] = int((sub[modifier] == 0).sum())
    else:
        # split at median
        med = sub[modifier].median()
        eff_hi = sub.loc[sub[modifier] >= med].groupby(treat)['pfs_months'].mean()
        eff_lo = sub.loc[sub[modifier] < med].groupby(treat)['pfs_months'].mean()
        out['median'] = float(med)
        out['eff_above_med'] = float(eff_hi.get(1, np.nan) - eff_hi.get(0, np.nan))
        out['eff_below_med'] = float(eff_lo.get(1, np.nan) - eff_lo.get(0, np.nan))
    return out


# Iter 6: tamoxifen
# Iter 7: palbociclib
# Iter 8: trastuzumab
# Iter 9: olaparib
# Iter 10: sacituzumab_govitecan
# Iter 11: pembrolizumab
results = {}
for i, t in enumerate(treatments, start=6):
    res_t = {}
    for m in binary_modifiers:
        res_t[m] = interaction_test(df, t, m, True)
    for m in cont_modifiers:
        res_t[m] = interaction_test(df, t, m, False)
    results[f'iter{i}_{t}'] = res_t

with open('iters_6_11_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Print top significant interactions per treatment
print("=== Top interactions per treatment (sorted by p) ===")
for key, res_t in results.items():
    print(f"\n--- {key} ---")
    rows = [(m, v['inter_p'], v['inter_coef'], v) for m, v in res_t.items()]
    rows.sort(key=lambda x: x[1])
    for m, p, c, v in rows[:8]:
        if 'eff_in_pos' in v:
            print(f"  {m}: inter_coef={c:+.3f} p={p:.2e} (eff_pos={v['eff_in_pos']:+.3f}, eff_neg={v['eff_in_neg']:+.3f})")
        else:
            print(f"  {m}: inter_coef={c:+.4f} p={p:.2e} (eff_above_med={v['eff_above_med']:+.3f}, eff_below_med={v['eff_below_med']:+.3f})")
