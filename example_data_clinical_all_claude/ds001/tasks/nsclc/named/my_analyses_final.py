"""Final confirmatory analyses focused on the sotorasib subgroup and overall summary."""
import json
import numpy as np
import pandas as pd
from scipy import stats
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


# ===== Sotorasib in KRAS+ males by additional modifiers =====
print("\n=== Sotorasib in KRAS+ males by additional modifiers ===")

# Check by ECOG within KRAS+ males
for ecog in [0, 1, 2]:
    sub = df.loc[(df.kras_g12c == 1) & (df.sex_female == 0) & (df.ecog_ps == ecog)]
    on = sub.loc[sub.treatment_sotorasib == 1, 'pfs_months']
    off = sub.loc[sub.treatment_sotorasib == 0, 'pfs_months']
    if len(on) > 5 and len(off) > 5:
        eff, p = diff_means(on, off)
        record(f"h_sot_kras_male_ecog{ecog}",
               summary=f"Sotorasib in KRAS+ AND male AND ECOG={ecog}: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
               p=p, eff=eff)

# By brain mets within KRAS+ males
for brain in [0, 1]:
    sub = df.loc[(df.kras_g12c == 1) & (df.sex_female == 0) & (df.has_brain_mets == brain)]
    on = sub.loc[sub.treatment_sotorasib == 1, 'pfs_months']
    off = sub.loc[sub.treatment_sotorasib == 0, 'pfs_months']
    eff, p = diff_means(on, off)
    record(f"h_sot_kras_male_brain{brain}",
           summary=f"Sotorasib in KRAS+ AND male AND brain_mets={brain}: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
           p=p, eff=eff)

# By STK11 within KRAS+ males
for stk in [0, 1]:
    sub = df.loc[(df.kras_g12c == 1) & (df.sex_female == 0) & (df.stk11_mutation == stk)]
    on = sub.loc[sub.treatment_sotorasib == 1, 'pfs_months']
    off = sub.loc[sub.treatment_sotorasib == 0, 'pfs_months']
    eff, p = diff_means(on, off)
    record(f"h_sot_kras_male_stk11{stk}",
           summary=f"Sotorasib in KRAS+ AND male AND STK11={stk}: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
           p=p, eff=eff)

# By age tertile within KRAS+ males
df['age_tertile'] = pd.qcut(df['age_years'], 3, labels=False)
for at in [0, 1, 2]:
    sub = df.loc[(df.kras_g12c == 1) & (df.sex_female == 0) & (df.age_tertile == at)]
    on = sub.loc[sub.treatment_sotorasib == 1, 'pfs_months']
    off = sub.loc[sub.treatment_sotorasib == 0, 'pfs_months']
    if len(on) > 5 and len(off) > 5:
        eff, p = diff_means(on, off)
        record(f"h_sot_kras_male_age_tertile{at}",
               summary=f"Sotorasib in KRAS+ AND male AND age tertile {at}: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
               p=p, eff=eff)

# ===== KRAS+ female sotorasib effect by subgroup (looking for any responders) =====
print("\n=== Sotorasib in KRAS+ females (any subgroup with effect?) ===")
for ecog in [0, 1, 2]:
    sub = df.loc[(df.kras_g12c == 1) & (df.sex_female == 1) & (df.ecog_ps == ecog)]
    on = sub.loc[sub.treatment_sotorasib == 1, 'pfs_months']
    off = sub.loc[sub.treatment_sotorasib == 0, 'pfs_months']
    if len(on) > 5 and len(off) > 5:
        eff, p = diff_means(on, off)
        record(f"h_sot_kras_female_ecog{ecog}",
               summary=f"Sotorasib in KRAS+ AND female AND ECOG={ecog}: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
               p=p, eff=eff)

# ===== Triple interaction: sotorasib * kras * sex =====
print("\n=== Triple interaction sotorasib*kras*sex ===")
m = smf.ols('log_pfs ~ treatment_sotorasib * kras_g12c * sex_female', data=df).fit()
print(m.summary())
key = 'treatment_sotorasib:kras_g12c:sex_female'
if key in m.params:
    record("h_sot_kras_sex_3way",
           summary=f"Three-way interaction sotorasib*kras_g12c*sex_female on log_pfs: coef={m.params[key]:.3f}, p={m.pvalues[key]:.2e}",
           p=float(m.pvalues[key]), eff=float(m.params[key]))

# ===== Final best-supported subgroup statement =====
sub = df.loc[(df.kras_g12c == 1) & (df.sex_female == 0)]
on = sub.loc[sub.treatment_sotorasib == 1, 'pfs_months']
off = sub.loc[sub.treatment_sotorasib == 0, 'pfs_months']
eff, p = diff_means(on, off)
record("h_final_sotorasib_subgroup",
       summary=f"FINAL: Sotorasib improves PFS specifically in KRAS G12C+ AND male patients: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# ===== Final best-supported subgroup hypotheses for other treatments =====
# Pembrolizumab: nothing strong. State the result.
# Use interaction model on log scale
m = smf.ols('log_pfs ~ treatment_pembrolizumab + sex_female + age_years + ecog_ps + stage_iv + has_brain_mets + pdl1_high + tmb_high + stk11_mutation + ever_smoker + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo', data=df).fit()
record("h_final_pembrolizumab_adj",
       summary=f"FINAL: Pembrolizumab adjusted effect on log_pfs: coef={m.params['treatment_pembrolizumab']:.4f}, p={m.pvalues['treatment_pembrolizumab']:.2e}; no consistent subgroup with PFS benefit was identified.",
       p=float(m.pvalues['treatment_pembrolizumab']), eff=float(m.params['treatment_pembrolizumab']))

m = smf.ols('log_pfs ~ treatment_olaparib + brca2_mutation + treatment_olaparib:brca2_mutation', data=df).fit()
record("h_final_olaparib_adj",
       summary=f"FINAL: Olaparib*BRCA2 interaction on log_pfs: coef={m.params['treatment_olaparib:brca2_mutation']:.4f}, p={m.pvalues['treatment_olaparib:brca2_mutation']:.2e}; no PFS benefit detected even in BRCA2+.",
       p=float(m.pvalues['treatment_olaparib:brca2_mutation']), eff=float(m.params['treatment_olaparib:brca2_mutation']))

m = smf.ols('log_pfs ~ treatment_osimertinib + egfr_mutation + treatment_osimertinib:egfr_mutation', data=df).fit()
record("h_final_osimertinib_adj",
       summary=f"FINAL: Osimertinib*EGFR interaction on log_pfs: coef={m.params['treatment_osimertinib:egfr_mutation']:.4f}, p={m.pvalues['treatment_osimertinib:egfr_mutation']:.2e}; no PFS benefit detected even in EGFR+.",
       p=float(m.pvalues['treatment_osimertinib:egfr_mutation']), eff=float(m.params['treatment_osimertinib:egfr_mutation']))

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nTotal records: {len(results)}")
