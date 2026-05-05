"""Additional analyses to refine subgroup hypotheses."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
df['log_pfs'] = np.log(df['pfs_months'] + 0.1)
df['pdl1_high'] = (df['pdl1_tps'] >= 0.5).astype(int)
df['ever_smoker'] = (df['smoking_status'] != 'never').astype(int)

with open('results.json', 'r') as f:
    results = json.load(f)


def record(key, *, summary, p, eff, sig=None, code=""):
    if sig is None and p is not None:
        sig = bool(p < 0.05)
    results[key] = {
        "result_summary": summary,
        "p_value": None if p is None else float(p),
        "effect_estimate": None if eff is None else float(eff),
        "significant": sig,
        "code": code,
    }
    print(f"[{key}] eff={eff!r} p={p!r}")
    print(f"   {summary}")


def diff_means(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(np.mean(a) - np.mean(b)), float(p)


# =============== Sotorasib x sex within KRAS+ ===============
print("\n=== Sotorasib by sex within KRAS+ ===")
for sex in [0, 1]:
    sub = df.loc[(df.kras_g12c == 1) & (df.sex_female == sex)]
    on = sub.loc[sub.treatment_sotorasib == 1, 'pfs_months']
    off = sub.loc[sub.treatment_sotorasib == 0, 'pfs_months']
    eff, p = diff_means(on, off)
    record(f"h_sot_kras_sex{sex}",
           summary=f"Sotorasib in KRAS+ AND sex_female={sex}: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
           p=p, eff=eff)

# =============== Osimertinib in EGFR+ stratified by smoking, since EGFR is enriched in never-smokers ===============
print("\n=== Osimertinib in EGFR+ by smoking ===")
for sm in ['never', 'former', 'current']:
    sub = df.loc[(df.egfr_mutation == 1) & (df.smoking_status == sm)]
    on = sub.loc[sub.treatment_osimertinib == 1, 'pfs_months']
    off = sub.loc[sub.treatment_osimertinib == 0, 'pfs_months']
    if len(on) > 5 and len(off) > 5:
        eff, p = diff_means(on, off)
        record(f"h_osi_egfr_smoke_{sm}",
               summary=f"Osimertinib in EGFR+ AND smoking={sm}: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
               p=p, eff=eff)

# Cross-tabulation: how many EGFR+ patients are also each smoking type
print("\nEGFR by smoking:")
print(pd.crosstab(df.egfr_mutation, df.smoking_status))

# =============== Pembro vs chemo subgroup search using lab cuts ===============
print("\n=== Pembrolizumab in low vs high inflammatory markers ===")
df['nlr_high'] = (df['nlr'] > df['nlr'].median()).astype(int)
df['crp_high'] = (df['crp_mg_l'] > df['crp_mg_l'].median()).astype(int)
df['alb_low'] = (df['albumin_g_dl'] < df['albumin_g_dl'].median()).astype(int)

for var in ['nlr_high', 'crp_high', 'alb_low']:
    for v in [0, 1]:
        sub = df.loc[df[var] == v]
        on = sub.loc[sub.treatment_pembrolizumab == 1, 'pfs_months']
        off = sub.loc[sub.treatment_pembrolizumab == 0, 'pfs_months']
        eff, p = diff_means(on, off)
        record(f"h_pem_{var}{v}",
               summary=f"Pembro in {var}={v}: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
               p=p, eff=eff)

# =============== Confirm age positive effect, check on log scale ===============
print("\n=== Age effect ===")
m = smf.ols('log_pfs ~ age_years', data=df).fit()
record("h_age_log_pfs",
       summary=f"Age coefficient on log_pfs (univariate): coef={m.params['age_years']:.4f}, p={m.pvalues['age_years']:.2e}",
       p=float(m.pvalues['age_years']), eff=float(m.params['age_years']))

# Quartile means
df['age_quartile'] = pd.qcut(df['age_years'], 4, labels=False)
for q in range(4):
    mn = df.loc[df.age_quartile == q, 'pfs_months'].mean()
    age_range = (df.loc[df.age_quartile == q, 'age_years'].min(), df.loc[df.age_quartile == q, 'age_years'].max())
    print(f"  Age quartile {q} ({age_range[0]:.1f}-{age_range[1]:.1f}): mean PFS = {mn:.2f}")

# =============== Final treatment heterogeneity: tree-based subgroup ===============
print("\n=== Tree-based subgroup discovery (CART on residuals) ===")
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# For each treatment: fit DT on the difference between treated PFS and predicted PFS without treatment
# Use a baseline OLS without treatment, then look at treatment effect heterogeneity
features = ['age_years', 'sex_female', 'ecog_ps', 'stage_iv', 'has_brain_mets',
            'egfr_mutation', 'kras_g12c', 'alk_fusion', 'stk11_mutation',
            'brca2_mutation', 'pdl1_tps', 'tmb_high', 'albumin_g_dl', 'ldh_u_l',
            'weight_loss_pct_6mo', 'crp_mg_l', 'nlr', 'hemoglobin_g_dl',
            'ever_smoker']

# Fit a model on control patients, predict counterfactual untreated PFS
def estimate_treatment_effect(treatment):
    print(f"\n  Treatment: {treatment}")
    control = df.loc[df[treatment] == 0]
    treated = df.loc[df[treatment] == 1]
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=0)
    rf.fit(control[features], control['pfs_months'])
    treated_pred = rf.predict(treated[features])
    treated_effect = treated['pfs_months'].values - treated_pred
    print(f"  Mean estimated treatment effect: {treated_effect.mean():.3f} mo")
    # Fit a tree on treatment effect to find subgroups
    dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=0)
    dt.fit(treated[features], treated_effect)
    leaves = dt.apply(treated[features])
    leaf_data = []
    for leaf in np.unique(leaves):
        mask = leaves == leaf
        leaf_data.append({
            "leaf": int(leaf),
            "n": int(mask.sum()),
            "effect": float(treated_effect[mask].mean())
        })
    leaf_data.sort(key=lambda r: -r['effect'])
    print(f"  Top leaves by effect:")
    for ld in leaf_data[:5]:
        print(f"    leaf {ld['leaf']}: n={ld['n']}, est_effect={ld['effect']:.3f} mo")
    print(f"  Bottom leaves by effect:")
    for ld in leaf_data[-5:]:
        print(f"    leaf {ld['leaf']}: n={ld['n']}, est_effect={ld['effect']:.3f} mo")
    return dt, treated, treated_effect, leaves

for tx in ['treatment_pembrolizumab', 'treatment_sotorasib',
           'treatment_olaparib', 'treatment_osimertinib']:
    dt, tr, eff, leaves = estimate_treatment_effect(tx)
    # Top leaf
    leaf_means = pd.Series(eff).groupby(leaves).mean().sort_values(ascending=False)
    top_leaf = leaf_means.index[0]
    bot_leaf = leaf_means.index[-1]
    print(f"  {tx} top leaf={top_leaf}, mean_effect={leaf_means.iloc[0]:.3f} mo, n={(leaves==top_leaf).sum()}")
    print(f"  {tx} bot leaf={bot_leaf}, mean_effect={leaf_means.iloc[-1]:.3f} mo, n={(leaves==bot_leaf).sum()}")
    record(f"h_tree_top_{tx}",
           summary=f"CART subgroup with largest estimated {tx} effect: est_effect={leaf_means.iloc[0]:.3f} mo, n={(leaves==top_leaf).sum()}",
           p=None, eff=float(leaf_means.iloc[0]), sig=None)
    record(f"h_tree_bot_{tx}",
           summary=f"CART subgroup with smallest/most-negative estimated {tx} effect: est_effect={leaf_means.iloc[-1]:.3f} mo, n={(leaves==bot_leaf).sum()}",
           p=None, eff=float(leaf_means.iloc[-1]), sig=None)

# =============== Pembrolizumab grid by clinically meaningful subset combos including labs ===============
print("\n=== Pembrolizumab among high-PD-L1, never-smoker, low-NLR, etc. ===")
sub = df.loc[(df.pdl1_high == 1) & (df.nlr < df.nlr.median())]
on = sub.loc[sub.treatment_pembrolizumab == 1, 'pfs_months']
off = sub.loc[sub.treatment_pembrolizumab == 0, 'pfs_months']
eff, p = diff_means(on, off)
record("h_pem_pdl1_nlrlow",
       summary=f"Pembro in PD-L1 high AND NLR low: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

sub = df.loc[(df.pdl1_high == 1) & (df.smoking_status != 'never') & (df.stk11_mutation == 0) & (df.ecog_ps <= 1)]
on = sub.loc[sub.treatment_pembrolizumab == 1, 'pfs_months']
off = sub.loc[sub.treatment_pembrolizumab == 0, 'pfs_months']
eff, p = diff_means(on, off)
record("h_pem_pdl1_smoker_stk11neg_goodecog",
       summary=f"Pembro in PD-L1 high AND ever-smoker AND STK11- AND ECOG<=1: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# =============== Sotorasib in KRAS+ across modifiers ===============
print("\n=== Sotorasib in KRAS+ across modifiers ===")
sub = df.loc[(df.kras_g12c == 1)]
m = smf.ols('log_pfs ~ treatment_sotorasib * (sex_female + age_years + ecog_ps + has_brain_mets + stk11_mutation + albumin_g_dl + ldh_u_l)', data=sub).fit()
print(m.summary())
for var in ['sex_female', 'age_years', 'ecog_ps', 'has_brain_mets', 'stk11_mutation', 'albumin_g_dl', 'ldh_u_l']:
    key = f'treatment_sotorasib:{var}'
    if key in m.params:
        record(f"h_sot_kras_int_{var}",
               summary=f"In KRAS+ (n={len(sub)}): interaction sotorasib*{var} on log_pfs: coef={m.params[key]:.4f}, p={m.pvalues[key]:.2e}",
               p=float(m.pvalues[key]), eff=float(m.params[key]))

# =============== Save ===============
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nTotal records: {len(results)}")
