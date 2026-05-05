"""Run statistical analyses on ds001_crc dataset and save results."""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings('ignore')

DIR = Path(__file__).parent
df = pd.read_parquet(DIR / 'dataset.parquet')

results = {}

def linreg(x, y):
    """Univariate linear regression, returns slope, p-value, intercept."""
    res = stats.linregress(x, y)
    return float(res.slope), float(res.pvalue), float(res.intercept)

def ttest(g1, g2):
    """Welch's t-test, returns mean diff, p-value."""
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    return float(g1.mean() - g2.mean()), float(p)

def ols_interaction(df, t_col, b_col, y_col='pfs_months'):
    """OLS y ~ T + B + T*B; return interaction coef and p."""
    import statsmodels.api as sm
    X = pd.DataFrame({
        't': df[t_col],
        'b': df[b_col],
        'tb': df[t_col] * df[b_col],
    })
    X = sm.add_constant(X)
    y = df[y_col]
    model = sm.OLS(y, X).fit()
    return float(model.params['tb']), float(model.pvalues['tb']), float(model.params['t']), float(model.params['b'])


# === Iteration 1: prognostic main effects of demographics and stage ===
print("=== Iteration 1 ===")
slope, p, _ = linreg(df['ecog_ps'], df['pfs_months'])
results['it1_ecog'] = (slope, p)
print(f"ecog_ps slope={slope:.4f}, p={p:.3g}")

slope, p, _ = linreg(df['stage_iv'], df['pfs_months'])
results['it1_stage'] = (slope, p)
print(f"stage_iv slope={slope:.4f}, p={p:.3g}")

slope, p, _ = linreg(df['sex_female'], df['pfs_months'])
results['it1_sex'] = (slope, p)
print(f"sex_female slope={slope:.4f}, p={p:.3g}")

# === Iteration 2: age and primary tumor location ===
print("\n=== Iteration 2 ===")
slope, p, _ = linreg(df['age_years'], df['pfs_months'])
results['it2_age'] = (slope, p)
print(f"age_years slope={slope:.4f}, p={p:.3g}")

slope, p, _ = linreg(df['right_sided_primary'], df['pfs_months'])
results['it2_right'] = (slope, p)
print(f"right_sided slope={slope:.4f}, p={p:.3g}")

# === Iteration 3: oncogenic driver mutations ===
print("\n=== Iteration 3 ===")
for col in ['kras_mutation', 'braf_v600e', 'nras_mutation', 'msi_high']:
    slope, p, _ = linreg(df[col], df['pfs_months'])
    results[f'it3_{col}'] = (slope, p)
    print(f"{col} slope={slope:.4f}, p={p:.3g}")

# === Iteration 4: lab biomarkers ===
print("\n=== Iteration 4 ===")
for col in ['albumin_g_dl', 'cea_ng_ml', 'weight_loss_pct_6mo', 'ldh_u_l']:
    slope, p, _ = linreg(df[col], df['pfs_months'])
    results[f'it4_{col}'] = (slope, p)
    print(f"{col} slope={slope:.4f}, p={p:.3g}")

# === Iteration 5: inflammation and rare biomarkers ===
print("\n=== Iteration 5 ===")
for col in ['crp_mg_l', 'nlr', 'her2_amplified', 'ntrk_fusion']:
    slope, p, _ = linreg(df[col], df['pfs_months'])
    results[f'it5_{col}'] = (slope, p)
    print(f"{col} slope={slope:.4f}, p={p:.3g}")

# === Iteration 6: treatment main effects ===
print("\n=== Iteration 6 ===")
for tcol in ['treatment_cetuximab', 'treatment_bevacizumab', 'treatment_pembrolizumab',
             'treatment_encorafenib', 'treatment_trastuzumab_tucatinib', 'treatment_regorafenib']:
    on = df.loc[df[tcol] == 1, 'pfs_months']
    off = df.loc[df[tcol] == 0, 'pfs_months']
    diff, p = ttest(on, off)
    results[f'it6_{tcol}'] = (diff, p, float(on.mean()), float(off.mean()))
    print(f"{tcol}: on={on.mean():.4f} off={off.mean():.4f} diff={diff:.4f} p={p:.3g}")

# === Iteration 7: pembrolizumab x msi_high ===
print("\n=== Iteration 7 ===")
inter, p_inter, _, _ = ols_interaction(df, 'treatment_pembrolizumab', 'msi_high')
sub = df[df['msi_high'] == 1]
diff, p_sub = ttest(sub.loc[sub['treatment_pembrolizumab'] == 1, 'pfs_months'],
                    sub.loc[sub['treatment_pembrolizumab'] == 0, 'pfs_months'])
results['it7_pembro_msi'] = (inter, p_inter, diff, p_sub, len(sub))
print(f"interaction={inter:.4f}, p={p_inter:.3g}; in MSI-H (n={len(sub)}): diff={diff:.4f}, p={p_sub:.3g}")

# === Iteration 8: encorafenib x braf_v600e ===
print("\n=== Iteration 8 ===")
inter, p_inter, _, _ = ols_interaction(df, 'treatment_encorafenib', 'braf_v600e')
sub = df[df['braf_v600e'] == 1]
diff, p_sub = ttest(sub.loc[sub['treatment_encorafenib'] == 1, 'pfs_months'],
                    sub.loc[sub['treatment_encorafenib'] == 0, 'pfs_months'])
results['it8_enc_braf'] = (inter, p_inter, diff, p_sub, len(sub))
print(f"interaction={inter:.4f}, p={p_inter:.3g}; in BRAF V600E (n={len(sub)}): diff={diff:.4f}, p={p_sub:.3g}")

# Encorafenib + cetuximab combo within BRAF V600E
braf_sub = df[df['braf_v600e'] == 1]
combo_mask = (braf_sub['treatment_encorafenib'] == 1) & (braf_sub['treatment_cetuximab'] == 1)
diff_combo, p_combo = ttest(braf_sub.loc[combo_mask, 'pfs_months'],
                             braf_sub.loc[~combo_mask, 'pfs_months'])
results['it8_combo'] = (diff_combo, p_combo, int(combo_mask.sum()), int((~combo_mask).sum()))
print(f"BRAF combo n={int(combo_mask.sum())}: diff={diff_combo:.4f}, p={p_combo:.3g}")

# === Iteration 9: trastuzumab+tucatinib x her2_amplified ===
print("\n=== Iteration 9 ===")
inter, p_inter, _, _ = ols_interaction(df, 'treatment_trastuzumab_tucatinib', 'her2_amplified')
sub = df[df['her2_amplified'] == 1]
diff, p_sub = ttest(sub.loc[sub['treatment_trastuzumab_tucatinib'] == 1, 'pfs_months'],
                    sub.loc[sub['treatment_trastuzumab_tucatinib'] == 0, 'pfs_months'])
results['it9_tras_her2'] = (inter, p_inter, diff, p_sub, len(sub))
print(f"interaction={inter:.4f}, p={p_inter:.3g}; in HER2 amp (n={len(sub)}): diff={diff:.4f}, p={p_sub:.3g}")

# === Iteration 10: cetuximab x RAS/BRAF wild-type, left-sided ===
print("\n=== Iteration 10 ===")
mask = (df['kras_mutation'] == 0) & (df['nras_mutation'] == 0) & (df['braf_v600e'] == 0) & (df['right_sided_primary'] == 0)
sub = df[mask]
diff, p = ttest(sub.loc[sub['treatment_cetuximab'] == 1, 'pfs_months'],
                sub.loc[sub['treatment_cetuximab'] == 0, 'pfs_months'])
results['it10_cetux_canon'] = (diff, p, len(sub))
print(f"RAS/BRAF-WT left-sided (n={len(sub)}): diff={diff:.4f}, p={p:.3g}")

# === Iteration 11: regorafenib heterogeneity screen ===
print("\n=== Iteration 11 ===")
for mod in ['kras_mutation', 'braf_v600e', 'right_sided_primary']:
    inter, p_inter, _, _ = ols_interaction(df, 'treatment_regorafenib', mod)
    sub_pos = df[df[mod] == 1]
    sub_neg = df[df[mod] == 0]
    diff_pos, p_pos = ttest(sub_pos.loc[sub_pos['treatment_regorafenib'] == 1, 'pfs_months'],
                             sub_pos.loc[sub_pos['treatment_regorafenib'] == 0, 'pfs_months'])
    diff_neg, p_neg = ttest(sub_neg.loc[sub_neg['treatment_regorafenib'] == 1, 'pfs_months'],
                             sub_neg.loc[sub_neg['treatment_regorafenib'] == 0, 'pfs_months'])
    results[f'it11_rego_{mod}'] = (inter, p_inter, diff_pos, p_pos, diff_neg, p_neg)
    print(f"rego x {mod}: inter={inter:.4f}, p={p_inter:.3g}; in mod=1: diff={diff_pos:.4f} p={p_pos:.3g}; in mod=0: diff={diff_neg:.4f} p={p_neg:.3g}")

# === Iteration 12: joint regorafenib subgroup ===
print("\n=== Iteration 12 ===")
mask = (df['kras_mutation'] == 0) & (df['braf_v600e'] == 0) & (df['right_sided_primary'] == 0)
sub = df[mask]
out = df[~mask]
diff_in, p_in = ttest(sub.loc[sub['treatment_regorafenib'] == 1, 'pfs_months'],
                      sub.loc[sub['treatment_regorafenib'] == 0, 'pfs_months'])
diff_out, p_out = ttest(out.loc[out['treatment_regorafenib'] == 1, 'pfs_months'],
                        out.loc[out['treatment_regorafenib'] == 0, 'pfs_months'])
results['it12_rego_joint'] = (diff_in, p_in, diff_out, p_out, len(sub), len(out))
print(f"In subgroup (n={len(sub)}): diff={diff_in:.4f}, p={p_in:.3g}")
print(f"Out subgroup (n={len(out)}): diff={diff_out:.4f}, p={p_out:.3g}")

# === Iteration 13: continuous modifiers of regorafenib ===
print("\n=== Iteration 13 ===")
import statsmodels.api as sm
# Full sample T:CEA
X = pd.DataFrame({
    't': df['treatment_regorafenib'],
    'cea': df['cea_ng_ml'],
    'tcea': df['treatment_regorafenib'] * df['cea_ng_ml'],
})
X = sm.add_constant(X)
m = sm.OLS(df['pfs_months'], X).fit()
results['it13_rego_cea_full'] = (float(m.params['tcea']), float(m.pvalues['tcea']))
print(f"rego x CEA full: inter={m.params['tcea']:.4f}, p={m.pvalues['tcea']:.3g}")

# Within subgroup
sub_mask = (df['kras_mutation'] == 0) & (df['braf_v600e'] == 0) & (df['right_sided_primary'] == 0)
sub_df = df[sub_mask]
X = pd.DataFrame({
    't': sub_df['treatment_regorafenib'],
    'cea': sub_df['cea_ng_ml'],
    'tcea': sub_df['treatment_regorafenib'] * sub_df['cea_ng_ml'],
})
X = sm.add_constant(X)
m = sm.OLS(sub_df['pfs_months'], X).fit()
results['it13_rego_cea_sub'] = (float(m.params['tcea']), float(m.pvalues['tcea']))
print(f"rego x CEA within k0/b0/r0: inter={m.params['tcea']:.4f}, p={m.pvalues['tcea']:.3g}")

# ECOG strata
ecog_diffs = []
for ec in [0, 1, 2]:
    s = df[df['ecog_ps'] == ec]
    if len(s) == 0:
        continue
    d, p = ttest(s.loc[s['treatment_regorafenib'] == 1, 'pfs_months'],
                 s.loc[s['treatment_regorafenib'] == 0, 'pfs_months'])
    ecog_diffs.append((ec, d, p, len(s)))
results['it13_rego_ecog'] = ecog_diffs
for ec, d, p, n in ecog_diffs:
    print(f"  ECOG={ec} (n={n}): diff={d:.4f} p={p:.3g}")

# === Iteration 14: CEA threshold scan within subgroup ===
print("\n=== Iteration 14 ===")
for thresh in [3, 4, 5, 6, 7, 8, 10]:
    sub_low = sub_df[sub_df['cea_ng_ml'] < thresh]
    sub_high = sub_df[sub_df['cea_ng_ml'] >= thresh]
    if len(sub_low) == 0 or len(sub_high) == 0:
        continue
    d_low, p_low = ttest(sub_low.loc[sub_low['treatment_regorafenib'] == 1, 'pfs_months'],
                          sub_low.loc[sub_low['treatment_regorafenib'] == 0, 'pfs_months'])
    d_high, p_high = ttest(sub_high.loc[sub_high['treatment_regorafenib'] == 1, 'pfs_months'],
                            sub_high.loc[sub_high['treatment_regorafenib'] == 0, 'pfs_months'])
    print(f"thresh={thresh}: low n={len(sub_low)} d={d_low:.4f} p={p_low:.3g}; high n={len(sub_high)} d={d_high:.4f} p={p_high:.3g}")
    results[f'it14_thresh_{thresh}'] = (d_low, p_low, d_high, p_high, len(sub_low), len(sub_high))

# === Iteration 15: full 4-predicate subgroup ===
print("\n=== Iteration 15 ===")
mask = (df['kras_mutation'] == 0) & (df['braf_v600e'] == 0) & (df['right_sided_primary'] == 0) & (df['cea_ng_ml'] < 5)
sub = df[mask]
out = df[~mask]
diff_in, p_in = ttest(sub.loc[sub['treatment_regorafenib'] == 1, 'pfs_months'],
                      sub.loc[sub['treatment_regorafenib'] == 0, 'pfs_months'])
diff_out, p_out = ttest(out.loc[out['treatment_regorafenib'] == 1, 'pfs_months'],
                        out.loc[out['treatment_regorafenib'] == 0, 'pfs_months'])
mean_in_on = float(sub.loc[sub['treatment_regorafenib'] == 1, 'pfs_months'].mean())
mean_in_off = float(sub.loc[sub['treatment_regorafenib'] == 0, 'pfs_months'].mean())
results['it15_rego_4pred'] = (diff_in, p_in, diff_out, p_out, len(sub), len(out), mean_in_on, mean_in_off)
print(f"Inside (n={len(sub)}): on={mean_in_on:.4f} off={mean_in_off:.4f} diff={diff_in:.4f}, p={p_in:.3g}")
print(f"Outside (n={len(out)}): diff={diff_out:.4f}, p={p_out:.3g}")

# === Iteration 16: NRAS as candidate modifier ===
print("\n=== Iteration 16 ===")
sub_mask = (df['kras_mutation'] == 0) & (df['braf_v600e'] == 0) & (df['right_sided_primary'] == 0)
for nras_val in [0, 1]:
    s = df[sub_mask & (df['nras_mutation'] == nras_val)]
    d, p = ttest(s.loc[s['treatment_regorafenib'] == 1, 'pfs_months'],
                 s.loc[s['treatment_regorafenib'] == 0, 'pfs_months'])
    print(f"NRAS={nras_val} within k0/b0/r0 (n={len(s)}): diff={d:.4f} p={p:.3g}")
    results[f'it16_nras_{nras_val}'] = (d, p, len(s))

# === Iteration 17: T x continuous-feature scan ===
print("\n=== Iteration 17 ===")
treatments = ['treatment_cetuximab', 'treatment_bevacizumab', 'treatment_pembrolizumab',
              'treatment_encorafenib', 'treatment_trastuzumab_tucatinib']
cont_features = ['age_years', 'cea_ng_ml', 'albumin_g_dl', 'ldh_u_l', 'weight_loss_pct_6mo',
                 'crp_mg_l', 'nlr', 'hemoglobin_g_dl', 'alkaline_phosphatase_u_l']
min_p = 1.0
strongest = None
all_strong = []
for t in treatments:
    for f in cont_features:
        X = pd.DataFrame({
            't': df[t], 'f': df[f], 'tf': df[t] * df[f],
        })
        X = sm.add_constant(X)
        m = sm.OLS(df['pfs_months'], X).fit()
        p = float(m.pvalues['tf'])
        b = float(m.params['tf'])
        if p < min_p:
            min_p = p
            strongest = (t, f, b, p)
        if p < 0.01:
            all_strong.append((t, f, b, p))
results['it17_scan'] = (strongest, all_strong)
print(f"Strongest T:F (continuous): {strongest}")
print(f"All p<0.01: {all_strong}")

# === Iteration 18: T x binary-feature scan + tree CATE ===
print("\n=== Iteration 18 ===")
bin_features = ['kras_mutation', 'nras_mutation', 'braf_v600e', 'msi_high', 'her2_amplified',
                'ntrk_fusion', 'right_sided_primary', 'stage_iv']
strong_bin = []
min_p_bin = 1.0
strongest_bin = None
for t in treatments:
    for b in bin_features:
        try:
            inter, p_inter, _, _ = ols_interaction(df, t, b)
            if p_inter < min_p_bin:
                min_p_bin = p_inter
                strongest_bin = (t, b, inter, p_inter)
            if p_inter < 0.05:
                strong_bin.append((t, b, inter, p_inter))
        except Exception:
            pass
results['it18_bin_scan'] = (strongest_bin, strong_bin)
print(f"Strongest T:Binary: {strongest_bin}")
print(f"All p<0.05: {strong_bin}")

# T-learner CATE for non-regorafenib
print("CATE std for non-regorafenib treatments:")
features_for_model = [c for c in df.columns if c not in ['patient_id', 'pfs_months'] + treatments + ['treatment_regorafenib']]
cate_stds = {}
for t in treatments:
    train_features = [c for c in df.columns if c not in ['patient_id', 'pfs_months'] + treatments + ['treatment_regorafenib']]
    df_t = df[df[t] == 1]
    df_c = df[df[t] == 0]
    if len(df_t) < 100:
        continue
    m_t = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0).fit(df_t[train_features], df_t['pfs_months'])
    m_c = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0).fit(df_c[train_features], df_c['pfs_months'])
    cate = m_t.predict(df[train_features]) - m_c.predict(df[train_features])
    cate_stds[t] = float(np.std(cate))
    print(f"  {t}: CATE std={cate_stds[t]:.4f}")
results['it18_cate'] = cate_stds

# === Iteration 19: prognostic feature interactions ===
print("\n=== Iteration 19 ===")
# right x kras
inter, p_inter, _, _ = ols_interaction(df, 'right_sided_primary', 'kras_mutation')
results['it19_right_kras'] = (inter, p_inter)
print(f"right x kras: inter={inter:.4f}, p={p_inter:.3g}")
# albumin x stage
X = pd.DataFrame({
    'a': df['albumin_g_dl'], 's': df['stage_iv'], 'as_': df['albumin_g_dl']*df['stage_iv'],
})
X = sm.add_constant(X)
m = sm.OLS(df['pfs_months'], X).fit()
results['it19_alb_stage'] = (float(m.params['as_']), float(m.pvalues['as_']))
print(f"alb x stage: inter={m.params['as_']:.4f}, p={m.pvalues['as_']:.3g}")
# weight x ecog
X = pd.DataFrame({
    'w': df['weight_loss_pct_6mo'], 'e': df['ecog_ps'], 'we': df['weight_loss_pct_6mo']*df['ecog_ps'],
})
X = sm.add_constant(X)
m = sm.OLS(df['pfs_months'], X).fit()
results['it19_w_ecog'] = (float(m.params['we']), float(m.pvalues['we']))
print(f"w x ecog: inter={m.params['we']:.4f}, p={m.pvalues['we']:.3g}")

# === Iteration 20: full multivariable model ===
print("\n=== Iteration 20 ===")
feature_cols = [c for c in df.columns if c not in ['patient_id', 'pfs_months']]
X = sm.add_constant(df[feature_cols])
m_full = sm.OLS(df['pfs_months'], X).fit()
print(f"R^2 = {m_full.rsquared:.4f}")
mv_treatment = {}
for t in ['treatment_cetuximab', 'treatment_bevacizumab', 'treatment_pembrolizumab',
          'treatment_encorafenib', 'treatment_trastuzumab_tucatinib', 'treatment_regorafenib']:
    mv_treatment[t] = (float(m_full.params[t]), float(m_full.pvalues[t]))
    print(f"  {t}: beta={m_full.params[t]:.4f}, p={m_full.pvalues[t]:.3g}")
mv_age = (float(m_full.params['age_years']), float(m_full.pvalues['age_years']))
print(f"age: beta={mv_age[0]:.4f}, p={mv_age[1]:.3g}")
results['it20_mv'] = (m_full.rsquared, mv_treatment, mv_age)

# Save additional adjusted coefficients for it25_3
print("Other key coefficients:")
key_feats = {}
for f in ['ecog_ps', 'stage_iv', 'albumin_g_dl', 'kras_mutation', 'braf_v600e',
         'right_sided_primary', 'age_years', 'weight_loss_pct_6mo', 'cea_ng_ml']:
    key_feats[f] = (float(m_full.params[f]), float(m_full.pvalues[f]))
    print(f"  {f}: beta={m_full.params[f]:.4f}, p={m_full.pvalues[f]:.3g}")
results['it20_key'] = key_feats

# === Iteration 21: ECOG x stage joint ===
print("\n=== Iteration 21 ===")
inter, p_inter, _, _ = ols_interaction(df, 'stage_iv', 'ecog_ps')
results['it21_se_inter'] = (inter, p_inter)
print(f"stage x ecog interaction: {inter:.4f}, p={p_inter:.3g}")
g00 = df[(df['stage_iv']==0) & (df['ecog_ps']==0)]['pfs_months'].mean()
g1ge1 = df[(df['stage_iv']==1) & (df['ecog_ps']>=1)]['pfs_months'].mean()
results['it21_groups'] = (float(g00), float(g1ge1))
print(f"stage=0 ecog=0: {g00:.4f}")
print(f"stage=1 ecog>=1: {g1ge1:.4f}")

# === Iteration 22: regorafenib x other treatments ===
print("\n=== Iteration 22 ===")
all_t = ['treatment_cetuximab', 'treatment_bevacizumab', 'treatment_pembrolizumab',
         'treatment_encorafenib', 'treatment_trastuzumab_tucatinib', 'treatment_regorafenib']
strongest_pair = None
min_pp = 1.0
strong_pairs = []
for i, t1 in enumerate(all_t):
    for t2 in all_t[i+1:]:
        try:
            inter, p_inter, _, _ = ols_interaction(df, t1, t2)
            if p_inter < min_pp:
                min_pp = p_inter
                strongest_pair = (t1, t2, inter, p_inter)
            if p_inter < 0.05:
                strong_pairs.append((t1, t2, inter, p_inter))
        except Exception:
            pass
results['it22_pairs'] = (strongest_pair, strong_pairs)
print(f"Strongest pair: {strongest_pair}")
print(f"p<0.05 pairs: {strong_pairs}")

# === Iteration 23: T-learner CATE for regorafenib ===
print("\n=== Iteration 23 ===")
features_for_model = [c for c in df.columns if c not in ['patient_id', 'pfs_months'] + all_t]
df_t = df[df['treatment_regorafenib'] == 1]
df_c = df[df['treatment_regorafenib'] == 0]
m_t = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(df_t[features_for_model], df_t['pfs_months'])
m_c = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(df_c[features_for_model], df_c['pfs_months'])
cate = m_t.predict(df[features_for_model]) - m_c.predict(df[features_for_model])
print(f"Regorafenib CATE: mean={cate.mean():.4f}, std={cate.std():.4f}")
# Tree on CATE
tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=200, random_state=0).fit(df[features_for_model], cate)
imp = pd.Series(tree.feature_importances_, index=features_for_model).sort_values(ascending=False)
top_modifiers = imp.head(6).to_dict()
print(f"Top tree feature importances on CATE:")
for k, v in top_modifiers.items():
    print(f"  {k}: {v:.4f}")
# Within joint subgroup
mask_in = (df['kras_mutation'] == 0) & (df['braf_v600e'] == 0) & (df['right_sided_primary'] == 0) & (df['cea_ng_ml'] < 5)
cate_in = cate[mask_in].mean()
cate_out = cate[~mask_in].mean()
print(f"CATE inside 4-pred: {cate_in:.4f}")
print(f"CATE outside 4-pred: {cate_out:.4f}")
results['it23'] = (float(cate.std()), top_modifiers, float(cate_in), float(cate_out))

# === Iteration 24: pembrolizumab in MSI-H subgroups ===
print("\n=== Iteration 24 ===")
msi = df[df['msi_high'] == 1]
# Left-sided MSI-H
left = msi[msi['right_sided_primary'] == 0]
right = msi[msi['right_sided_primary'] == 1]
d_left, p_left = ttest(left.loc[left['treatment_pembrolizumab'] == 1, 'pfs_months'],
                       left.loc[left['treatment_pembrolizumab'] == 0, 'pfs_months'])
d_right, p_right = ttest(right.loc[right['treatment_pembrolizumab'] == 1, 'pfs_months'],
                         right.loc[right['treatment_pembrolizumab'] == 0, 'pfs_months'])
print(f"MSI-H left (n={len(left)}): diff={d_left:.4f} p={p_left:.3g}")
print(f"MSI-H right (n={len(right)}): diff={d_right:.4f} p={p_right:.3g}")
results['it24_msi_left_right'] = (d_left, p_left, len(left), d_right, p_right, len(right))
# MSI-H stage IV
msi_s4 = msi[msi['stage_iv'] == 1]
d_s, p_s = ttest(msi_s4.loc[msi_s4['treatment_pembrolizumab'] == 1, 'pfs_months'],
                 msi_s4.loc[msi_s4['treatment_pembrolizumab'] == 0, 'pfs_months'])
print(f"MSI-H stage IV (n={len(msi_s4)}): diff={d_s:.4f} p={p_s:.3g}")
results['it24_msi_s4'] = (d_s, p_s, len(msi_s4))

# Save all results
with open(DIR / 'analysis_results.json', 'w') as f:
    # Convert tuples to lists for JSON serialization
    def convert(o):
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    json.dump(convert(results), f, indent=2)
print("\nSaved analysis_results.json")
