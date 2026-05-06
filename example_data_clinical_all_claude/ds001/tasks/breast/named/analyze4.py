"""Iter 17-25: Verification, ki67 cutoff search, multivariable models, exhaustive subgroup checks."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_parquet("dataset.parquet")
OUT = {}

# === Iter 17: Find optimal Ki67 cutoff for palbo benefit (in ER+ HER2- PIK3CA-wt) ===
print("\n=== ITER 17: Ki67 cutoff search in ER+ HER2- PIK3CA-wt ===")
core = df[(df['er_positive']==1) & (df['her2_positive']==0) & (df['pik3ca_mutation']==0)].copy()
print(f"  N in ER+ HER2- PIK3CA-wt subgroup: {len(core)}")
ki_results = []
for cut in [10, 12, 14, 15, 17, 20, 25, 30]:
    low = core[core['ki67_pct'] < cut]
    high = core[core['ki67_pct'] >= cut]
    a_l = low.loc[low['treatment_palbociclib']==1, 'pfs_months']
    b_l = low.loc[low['treatment_palbociclib']==0, 'pfs_months']
    a_h = high.loc[high['treatment_palbociclib']==1, 'pfs_months']
    b_h = high.loc[high['treatment_palbociclib']==0, 'pfs_months']
    if len(a_l)>5 and len(a_h)>5:
        d_l = a_l.mean()-b_l.mean()
        d_h = a_h.mean()-b_h.mean()
        _, p_l = stats.ttest_ind(a_l, b_l, equal_var=False)
        _, p_h = stats.ttest_ind(a_h, b_h, equal_var=False)
        ki_results.append({'cut':cut,'diff_low':float(d_l),'diff_high':float(d_h),'n_low_tx':len(a_l),'n_high_tx':len(a_h),
                           'p_low':float(p_l),'p_high':float(p_h)})
        print(f"  Ki67 cutoff={cut:3d}:  ki67<{cut} diff={d_l:+.3f} (n_tx={len(a_l)}, p={p_l:.1e}) | ki67>={cut} diff={d_h:+.3f} (n_tx={len(a_h)}, p={p_h:.1e})")
OUT['ki67_cutoff_search'] = ki_results

# === Iter 18: Continuous palbo x ki67 interaction within ER+ HER2- PIK3CA-wt ===
print("\n=== ITER 18: Continuous palbo x ki67 interaction in ER+ HER2- PIK3CA-wt ===")
X = pd.DataFrame({'tx':core['treatment_palbociclib'],'ki67':core['ki67_pct'],
                  'inter':core['treatment_palbociclib']*core['ki67_pct']})
X = sm.add_constant(X)
fit = sm.OLS(core['pfs_months'], X).fit()
print(f"  tx coef = {fit.params['tx']:+.3f}")
print(f"  ki67 coef = {fit.params['ki67']:+.4f}")
print(f"  tx*ki67 coef = {fit.params['inter']:+.4f}, p={fit.pvalues['inter']:.2e}")
print(f"  Implied palbo effect = {fit.params['tx']:+.3f} + {fit.params['inter']:+.4f} * ki67")
print(f"  Implied effect at ki67=10: {fit.params['tx'] + 10*fit.params['inter']:+.3f}")
print(f"  Implied effect at ki67=20: {fit.params['tx'] + 20*fit.params['inter']:+.3f}")
print(f"  Implied effect at ki67=30: {fit.params['tx'] + 30*fit.params['inter']:+.3f}")
OUT['palbo_ki67_continuous'] = {
    'tx_coef': float(fit.params['tx']), 'ki67_coef': float(fit.params['ki67']),
    'tx_x_ki67_coef': float(fit.params['inter']), 'tx_x_ki67_p': float(fit.pvalues['inter']),
}

# === Iter 19: Multivariable model with all key biomarkers (entire dataset) ===
print("\n=== ITER 19: Multivariable model — palbo x all key modifiers ===")
mv_cols = ['treatment_palbociclib','er_positive','her2_positive','pik3ca_mutation','ki67_pct',
           'age_years','ecog_ps','stage_iv','has_brain_mets','albumin_g_dl','weight_loss_pct_6mo']
X = df[mv_cols].copy()
X['palbo_x_er'] = df['treatment_palbociclib']*df['er_positive']
X['palbo_x_her2'] = df['treatment_palbociclib']*df['her2_positive']
X['palbo_x_pik3ca'] = df['treatment_palbociclib']*df['pik3ca_mutation']
X['palbo_x_ki67'] = df['treatment_palbociclib']*df['ki67_pct']
X = sm.add_constant(X)
fit = sm.OLS(df['pfs_months'], X).fit()
print(fit.summary().tables[1])
OUT['palbo_mv_model'] = {'coef': fit.params.to_dict(), 'p': fit.pvalues.to_dict()}

# === Iter 20: 4-way subgroup: optimal palbo subgroup mean effect ===
print("\n=== ITER 20: Final palbo subgroup definition test ===")
# Test ER+ AND HER2- AND PIK3CA-wt AND Ki67<20
sub_mask = ((df['er_positive']==1) & (df['her2_positive']==0) &
            (df['pik3ca_mutation']==0) & (df['ki67_pct']<20))
sub = df[sub_mask]
out = df[~sub_mask]
a = sub.loc[sub['treatment_palbociclib']==1, 'pfs_months']
b = sub.loc[sub['treatment_palbociclib']==0, 'pfs_months']
c = out.loc[out['treatment_palbociclib']==1, 'pfs_months']
d = out.loc[out['treatment_palbociclib']==0, 'pfs_months']
diff_in = a.mean() - b.mean()
diff_out = c.mean() - d.mean()
_, p_in = stats.ttest_ind(a, b, equal_var=False)
_, p_out = stats.ttest_ind(c, d, equal_var=False)
print(f"  In ER+/HER2-/PIK3CA-wt/Ki67<20 ({len(sub)} pts, n_tx={len(a)}): palbo diff = {diff_in:+.3f}, p={p_in:.2e}")
print(f"  Outside subgroup ({len(out)} pts, n_tx={len(c)}): palbo diff = {diff_out:+.3f}, p={p_out:.2e}")

# Formal interaction in single regression
df['_subgroup'] = sub_mask.astype(int)
X = pd.DataFrame({'tx':df['treatment_palbociclib'], 'sub':df['_subgroup'],
                  'inter':df['treatment_palbociclib']*df['_subgroup']})
X = sm.add_constant(X)
fit = sm.OLS(df['pfs_months'], X).fit()
print(f"  Formal palbo x subgroup interaction coef = {fit.params['inter']:+.3f}, p = {fit.pvalues['inter']:.2e}")
OUT['palbo_final_subgroup'] = {
    'subgroup_definition': 'er_positive==1 AND her2_positive==0 AND pik3ca_mutation==0 AND ki67_pct<20',
    'n_subgroup': int(len(sub)), 'n_outside': int(len(out)),
    'diff_in_subgroup': float(diff_in), 'p_in_subgroup': float(p_in),
    'diff_outside_subgroup': float(diff_out), 'p_outside_subgroup': float(p_out),
    'formal_interaction_coef': float(fit.params['inter']),
    'formal_interaction_p': float(fit.pvalues['inter']),
}

# === Iter 21: Olaparib BRCA-mut verification with covariate-adjusted regression ===
print("\n=== ITER 21: Olaparib x BRCA covariate-adjusted ===")
df['brca_any'] = ((df['brca1_mutation']==1) | (df['brca2_mutation']==1)).astype(int)
covs = ['age_years','ecog_ps','stage_iv','has_brain_mets','albumin_g_dl']
X = df[['treatment_olaparib','brca_any'] + covs].copy()
X['inter'] = df['treatment_olaparib']*df['brca_any']
X = sm.add_constant(X)
fit = sm.OLS(df['pfs_months'], X).fit()
print(f"  olaparib coef={fit.params['treatment_olaparib']:+.3f} p={fit.pvalues['treatment_olaparib']:.2e}")
print(f"  brca_any coef={fit.params['brca_any']:+.3f} p={fit.pvalues['brca_any']:.2e}")
print(f"  olaparib x brca_any coef={fit.params['inter']:+.3f} p={fit.pvalues['inter']:.2e}")
OUT['olaparib_adjusted'] = {
    'olaparib_main_coef': float(fit.params['treatment_olaparib']),
    'olaparib_main_p': float(fit.pvalues['treatment_olaparib']),
    'brca_main_coef': float(fit.params['brca_any']),
    'brca_main_p': float(fit.pvalues['brca_any']),
    'olaparib_x_brca_coef': float(fit.params['inter']),
    'olaparib_x_brca_p': float(fit.pvalues['inter']),
}

# === Iter 22: Treatment combination check — palbo + tamox? palbo + trastu? ===
print("\n=== ITER 22: Treatment combinations ===")
# palbo alone vs palbo+tamox in ER+
er_pos = df[df['er_positive']==1]
groups = [
    ('no palbo no tamox', (er_pos['treatment_palbociclib']==0) & (er_pos['treatment_tamoxifen']==0)),
    ('palbo only',        (er_pos['treatment_palbociclib']==1) & (er_pos['treatment_tamoxifen']==0)),
    ('tamox only',        (er_pos['treatment_palbociclib']==0) & (er_pos['treatment_tamoxifen']==1)),
    ('palbo + tamox',     (er_pos['treatment_palbociclib']==1) & (er_pos['treatment_tamoxifen']==1)),
]
combo_results = []
for lab, m in groups:
    v = er_pos.loc[m, 'pfs_months']
    combo_results.append({'group':lab,'n':int(m.sum()),'mean':float(v.mean())})
    print(f"  {lab:25s}: n={m.sum()}  mean PFS={v.mean():.3f}")
OUT['palbo_tamox_combo'] = combo_results

# === Iter 23: Exhaustive 2-feature subgroup search for palbo (top heterogeneity) ===
print("\n=== ITER 23: Exhaustive 2-feature subgroup heterogeneity for palbo ===")
mods = ['er_positive','pr_positive','her2_positive','her2_low','pik3ca_mutation','postmenopausal',
        'stage_iv','has_brain_mets','node_positive','brca1_mutation','brca2_mutation']
def palbo_diff(mask):
    a = df.loc[mask & (df['treatment_palbociclib']==1), 'pfs_months']
    b = df.loc[mask & (df['treatment_palbociclib']==0), 'pfs_months']
    if len(a) < 50 or len(b) < 50:
        return None
    return float(a.mean()-b.mean()), len(a), len(b)

import itertools
exhaust = []
for m1, m2 in itertools.combinations(mods, 2):
    for v1 in [0,1]:
        for v2 in [0,1]:
            mask = (df[m1]==v1) & (df[m2]==v2)
            r = palbo_diff(mask)
            if r is None: continue
            d, na, nb = r
            exhaust.append({'def':f"{m1}={v1} & {m2}={v2}", 'diff':d, 'n_tx':na, 'n_no':nb})
exhaust = sorted(exhaust, key=lambda x:-x['diff'])
print("Top 12 by palbo diff:")
for r in exhaust[:12]:
    print(f"  {r['def']:55s} diff={r['diff']:+.3f}  n_tx={r['n_tx']}  n_no={r['n_no']}")
print("Bottom 8 by palbo diff:")
for r in exhaust[-8:]:
    print(f"  {r['def']:55s} diff={r['diff']:+.3f}  n_tx={r['n_tx']}  n_no={r['n_no']}")
OUT['palbo_exhaustive_2way'] = exhaust[:30] + exhaust[-10:]

# === Iter 24: Confirm: Are olaparib effects modified by anything else, given borderline BRCA result? ===
print("\n=== ITER 24: Olaparib BRCA + HER2/ER cross ===")
ola_def = [
    ('BRCA-mut + ER-', (df['brca_any']==1) & (df['er_positive']==0)),
    ('BRCA-mut + ER+', (df['brca_any']==1) & (df['er_positive']==1)),
    ('BRCA-mut + HER2-', (df['brca_any']==1) & (df['her2_positive']==0)),
    ('BRCA-mut + TNBC-like', (df['brca_any']==1) & (df['er_positive']==0) & (df['pr_positive']==0) & (df['her2_positive']==0)),
    ('BRCA-mut + HR+', (df['brca_any']==1) & ((df['er_positive']==1) | (df['pr_positive']==1))),
]
ola2 = []
for lab, m in ola_def:
    a = df.loc[m & (df['treatment_olaparib']==1), 'pfs_months']
    b = df.loc[m & (df['treatment_olaparib']==0), 'pfs_months']
    if len(a)>5 and len(b)>5:
        _, p = stats.ttest_ind(a,b,equal_var=False)
        d = a.mean()-b.mean()
        ola2.append({'def':lab,'n_tx':len(a),'n_no':len(b),'diff':float(d),'p':float(p)})
        print(f"  Olaparib in {lab:25s}: n_tx={len(a):4d} n_no={len(b):4d} diff={d:+.3f} p={p:.2e}")
OUT['olaparib_extra_subgroups'] = ola2

# === Iter 25: Check palbo subgroup robustness with covariate adjustment ===
print("\n=== ITER 25: Adjusted palbo effect within subgroup ===")
sub = df[(df['er_positive']==1) & (df['her2_positive']==0) &
         (df['pik3ca_mutation']==0) & (df['ki67_pct']<20)]
covs = ['age_years','ecog_ps','stage_iv','has_brain_mets','albumin_g_dl','weight_loss_pct_6mo','tumor_size_cm']
X = sub[['treatment_palbociclib'] + covs].copy()
X = sm.add_constant(X)
fit = sm.OLS(sub['pfs_months'], X).fit()
print(f"  Adjusted palbo effect within subgroup = {fit.params['treatment_palbociclib']:+.3f}, p={fit.pvalues['treatment_palbociclib']:.2e}")
print(f"  N = {len(sub)}, R^2={fit.rsquared:.3f}")
OUT['palbo_subgroup_adjusted'] = {
    'coef': float(fit.params['treatment_palbociclib']),
    'p': float(fit.pvalues['treatment_palbociclib']),
    'n': int(len(sub)),
    'r2': float(fit.rsquared),
}

with open("results_iter17_25.json","w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("\nSaved results_iter17_25.json")
