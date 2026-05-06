"""
NSCLC dataset analysis - iterative hypothesis testing.
Outputs a results dictionary saved to results.json for assembly into the transcript.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')

# Use log(PFS+0.1) for parametric analyses (PFS is right-skewed and >=0)
df['log_pfs'] = np.log(df['pfs_months'] + 0.1)

results = {}


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
    print(f"[{key}] eff={eff!r}  p={p!r}  sig={sig!r}")
    print(f"   {summary}")


def diff_means(a, b):
    """Welch t-test on two arrays. Returns (mean_a - mean_b, p)."""
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(np.mean(a) - np.mean(b)), float(p)


# ---------------- Iteration 1: distributions + main effects ----------------
print("\n=== Iteration 1: descriptive + simple main effects ===")

# Frequencies
print("\nFrequencies:")
for c in ['smoking_status', 'histology', 'stage_iv', 'has_brain_mets',
          'egfr_mutation', 'kras_g12c', 'alk_fusion', 'stk11_mutation',
          'brca2_mutation', 'tmb_high', 'treatment_pembrolizumab',
          'treatment_sotorasib', 'treatment_olaparib', 'treatment_osimertinib',
          'ecog_ps']:
    print(f"  {c}: {df[c].value_counts().to_dict()}")

# h1: ECOG PS associated with shorter PFS
m_ecog = smf.ols('pfs_months ~ C(ecog_ps)', data=df).fit()
mean_pfs_by_ecog = df.groupby('ecog_ps')['pfs_months'].mean().to_dict()
# Compare ECOG=2 vs ECOG=0
e0 = df.loc[df.ecog_ps == 0, 'pfs_months']
e2 = df.loc[df.ecog_ps == 2, 'pfs_months']
eff, p = diff_means(e2, e0)
record("h1_ecog2_vs_ecog0",
       summary=f"PFS by ECOG: mean PFS ECOG0={mean_pfs_by_ecog.get(0):.2f}, ECOG1={mean_pfs_by_ecog.get(1):.2f}, ECOG2={mean_pfs_by_ecog.get(2):.2f}; ECOG2 - ECOG0 = {eff:.2f} months, Welch p={p:.2e}",
       p=p, eff=eff,
       code="diff means PFS for ecog_ps==2 vs ecog_ps==0")

# h2: stage IV associated with shorter PFS
eff, p = diff_means(df.loc[df.stage_iv == 1, 'pfs_months'], df.loc[df.stage_iv == 0, 'pfs_months'])
record("h2_stageiv",
       summary=f"PFS in stage IV ({df.loc[df.stage_iv==1,'pfs_months'].mean():.2f}) vs non-stage IV ({df.loc[df.stage_iv==0,'pfs_months'].mean():.2f}); diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# h3: brain mets associated with shorter PFS
eff, p = diff_means(df.loc[df.has_brain_mets == 1, 'pfs_months'], df.loc[df.has_brain_mets == 0, 'pfs_months'])
record("h3_brainmets",
       summary=f"PFS with brain mets ({df.loc[df.has_brain_mets==1,'pfs_months'].mean():.2f}) vs without ({df.loc[df.has_brain_mets==0,'pfs_months'].mean():.2f}); diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# h4: female vs male PFS
eff, p = diff_means(df.loc[df.sex_female == 1, 'pfs_months'], df.loc[df.sex_female == 0, 'pfs_months'])
record("h4_female",
       summary=f"PFS in females vs males: diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# ---------------- Iteration 2: treatment main effects on PFS ----------------
print("\n=== Iteration 2: treatment main effects ===")
for tx in ['treatment_pembrolizumab', 'treatment_sotorasib', 'treatment_olaparib', 'treatment_osimertinib']:
    eff, p = diff_means(df.loc[df[tx] == 1, 'pfs_months'], df.loc[df[tx] == 0, 'pfs_months'])
    record(f"h_main_{tx}",
           summary=f"PFS on {tx}={df.loc[df[tx]==1,'pfs_months'].mean():.2f} vs off={df.loc[df[tx]==0,'pfs_months'].mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
           p=p, eff=eff)

# ---------------- Iteration 3: biomarker / lab main effects ----------------
print("\n=== Iteration 3: biomarker + lab main effects ===")
for biomarker in ['egfr_mutation', 'kras_g12c', 'alk_fusion', 'stk11_mutation', 'brca2_mutation', 'tmb_high']:
    eff, p = diff_means(df.loc[df[biomarker] == 1, 'pfs_months'], df.loc[df[biomarker] == 0, 'pfs_months'])
    record(f"h_main_{biomarker}",
           summary=f"PFS in {biomarker}+ ({df.loc[df[biomarker]==1,'pfs_months'].mean():.2f}) vs - ({df.loc[df[biomarker]==0,'pfs_months'].mean():.2f}); diff={eff:.2f} mo, p={p:.2e}",
           p=p, eff=eff)

# Continuous labs vs PFS (Pearson correlation)
for lab in ['albumin_g_dl', 'ldh_u_l', 'crp_mg_l', 'nlr', 'weight_loss_pct_6mo',
            'pdl1_tps', 'hemoglobin_g_dl', 'alkaline_phosphatase_u_l', 'ast_u_l',
            'alt_u_l', 'total_bilirubin_mg_dl', 'creatinine_mg_dl', 'bun_mg_dl',
            'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl', 'age_years']:
    r, p = stats.pearsonr(df[lab], df['pfs_months'])
    record(f"h_corr_{lab}",
           summary=f"Pearson r({lab}, pfs_months) = {r:.3f}, p={p:.2e}",
           p=float(p), eff=float(r))

# ---------------- Iteration 4: smoking and histology ----------------
print("\n=== Iteration 4: categorical features ===")
# Smoking groups
for s in df['smoking_status'].unique():
    sub = df.loc[df['smoking_status'] == s, 'pfs_months']
    print(f"  smoking={s}: n={len(sub)}, mean PFS={sub.mean():.2f}")
# never vs current
nev = df.loc[df.smoking_status == 'never', 'pfs_months']
cur = df.loc[df.smoking_status == 'current', 'pfs_months']
eff, p = diff_means(nev, cur)
record("h_smoking_never_vs_current",
       summary=f"PFS never={nev.mean():.2f} vs current={cur.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# Histology
for h in df['histology'].unique():
    sub = df.loc[df['histology'] == h, 'pfs_months']
    print(f"  hist={h}: n={len(sub)}, mean PFS={sub.mean():.2f}")
adeno = df.loc[df.histology == 'adenocarcinoma', 'pfs_months']
sq = df.loc[df.histology == 'squamous', 'pfs_months']
eff, p = diff_means(adeno, sq)
record("h_hist_adeno_vs_sq",
       summary=f"PFS adeno={adeno.mean():.2f} vs squamous={sq.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# ---------------- Iteration 5: targeted-treatment x biomarker interactions ----------------
print("\n=== Iteration 5: matched targeted treatment x biomarker ===")

def interaction_test(treatment, biomarker, key_prefix):
    # Stratified means
    on_b = df.loc[df[biomarker] == 1]
    off_b = df.loc[df[biomarker] == 0]
    eff_on, p_on = diff_means(on_b.loc[on_b[treatment] == 1, 'pfs_months'],
                              on_b.loc[on_b[treatment] == 0, 'pfs_months'])
    eff_off, p_off = diff_means(off_b.loc[off_b[treatment] == 1, 'pfs_months'],
                                off_b.loc[off_b[treatment] == 0, 'pfs_months'])
    # OLS interaction on log_pfs
    model = smf.ols(f'log_pfs ~ {treatment} * {biomarker}', data=df).fit()
    interaction_coef = model.params.get(f'{treatment}:{biomarker}', np.nan)
    interaction_p = model.pvalues.get(f'{treatment}:{biomarker}', np.nan)
    record(f"{key_prefix}_in_{biomarker}pos",
           summary=f"In {biomarker}+ (n={len(on_b)}): {treatment} on={on_b.loc[on_b[treatment]==1,'pfs_months'].mean():.2f} vs off={on_b.loc[on_b[treatment]==0,'pfs_months'].mean():.2f}, diff={eff_on:.2f} mo, p={p_on:.2e}",
           p=p_on, eff=eff_on)
    record(f"{key_prefix}_in_{biomarker}neg",
           summary=f"In {biomarker}- (n={len(off_b)}): {treatment} on={off_b.loc[off_b[treatment]==1,'pfs_months'].mean():.2f} vs off={off_b.loc[off_b[treatment]==0,'pfs_months'].mean():.2f}, diff={eff_off:.2f} mo, p={p_off:.2e}",
           p=p_off, eff=eff_off)
    record(f"{key_prefix}_interaction_{biomarker}",
           summary=f"Interaction {treatment}*{biomarker} on log_pfs: coef={interaction_coef:.3f}, p={interaction_p:.2e}",
           p=float(interaction_p), eff=float(interaction_coef))

# Osimertinib x EGFR
interaction_test('treatment_osimertinib', 'egfr_mutation', 'h_osi_egfr')
# Sotorasib x KRAS G12C
interaction_test('treatment_sotorasib', 'kras_g12c', 'h_sot_kras')
# Olaparib x BRCA2
interaction_test('treatment_olaparib', 'brca2_mutation', 'h_ola_brca2')

# ---------------- Iteration 6: pembrolizumab x PD-L1, TMB, smoking ----------------
print("\n=== Iteration 6: pembrolizumab heterogeneity ===")
# PD-L1 (continuous): split at 0.5 (50% TPS) per clinical convention
df['pdl1_high'] = (df['pdl1_tps'] >= 0.5).astype(int)
print(f"  pdl1_high prevalence: {df['pdl1_high'].mean():.3f}")

for biomarker in ['pdl1_high', 'tmb_high', 'stk11_mutation']:
    on_b = df.loc[df[biomarker] == 1]
    off_b = df.loc[df[biomarker] == 0]
    eff_on, p_on = diff_means(on_b.loc[on_b['treatment_pembrolizumab'] == 1, 'pfs_months'],
                              on_b.loc[on_b['treatment_pembrolizumab'] == 0, 'pfs_months'])
    eff_off, p_off = diff_means(off_b.loc[off_b['treatment_pembrolizumab'] == 1, 'pfs_months'],
                                off_b.loc[off_b['treatment_pembrolizumab'] == 0, 'pfs_months'])
    model = smf.ols(f'log_pfs ~ treatment_pembrolizumab * {biomarker}', data=df).fit()
    interaction_coef = model.params.get(f'treatment_pembrolizumab:{biomarker}', np.nan)
    interaction_p = model.pvalues.get(f'treatment_pembrolizumab:{biomarker}', np.nan)
    record(f"h_pem_in_{biomarker}pos",
           summary=f"Pembrolizumab in {biomarker}+ (n={len(on_b)}): on={on_b.loc[on_b['treatment_pembrolizumab']==1,'pfs_months'].mean():.2f} vs off={on_b.loc[on_b['treatment_pembrolizumab']==0,'pfs_months'].mean():.2f}; diff={eff_on:.2f} mo, p={p_on:.2e}",
           p=p_on, eff=eff_on)
    record(f"h_pem_in_{biomarker}neg",
           summary=f"Pembrolizumab in {biomarker}- (n={len(off_b)}): on={off_b.loc[off_b['treatment_pembrolizumab']==1,'pfs_months'].mean():.2f} vs off={off_b.loc[off_b['treatment_pembrolizumab']==0,'pfs_months'].mean():.2f}; diff={eff_off:.2f} mo, p={p_off:.2e}",
           p=p_off, eff=eff_off)
    record(f"h_pem_interaction_{biomarker}",
           summary=f"Interaction pembro*{biomarker} on log_pfs: coef={interaction_coef:.3f}, p={interaction_p:.2e}",
           p=float(interaction_p), eff=float(interaction_coef))

# Pembro x smoker (current/former vs never)
df['ever_smoker'] = (df['smoking_status'] != 'never').astype(int)
on = df.loc[df.ever_smoker == 1]
off = df.loc[df.ever_smoker == 0]
eff_on, p_on = diff_means(on.loc[on['treatment_pembrolizumab'] == 1, 'pfs_months'],
                          on.loc[on['treatment_pembrolizumab'] == 0, 'pfs_months'])
eff_off, p_off = diff_means(off.loc[off['treatment_pembrolizumab'] == 1, 'pfs_months'],
                            off.loc[off['treatment_pembrolizumab'] == 0, 'pfs_months'])
model = smf.ols('log_pfs ~ treatment_pembrolizumab * ever_smoker', data=df).fit()
ic = model.params.get('treatment_pembrolizumab:ever_smoker', np.nan)
ip_ = model.pvalues.get('treatment_pembrolizumab:ever_smoker', np.nan)
record("h_pem_in_eversmoker",
       summary=f"Pembrolizumab in ever-smokers: diff={eff_on:.2f} mo, p={p_on:.2e}",
       p=p_on, eff=eff_on)
record("h_pem_in_neversmoker",
       summary=f"Pembrolizumab in never-smokers: diff={eff_off:.2f} mo, p={p_off:.2e}",
       p=p_off, eff=eff_off)
record("h_pem_interaction_eversmoker",
       summary=f"Interaction pembro*ever_smoker on log_pfs: coef={ic:.3f}, p={ip_:.2e}",
       p=float(ip_), eff=float(ic))

# ---------------- Iteration 7: multivariable model on log_pfs ----------------
print("\n=== Iteration 7: multivariable model on log_pfs ===")
formula = ('log_pfs ~ age_years + sex_female + C(smoking_status) + ecog_ps + C(histology) '
           '+ stage_iv + has_brain_mets + egfr_mutation + kras_g12c + alk_fusion '
           '+ stk11_mutation + brca2_mutation + pdl1_tps + tmb_high + albumin_g_dl '
           '+ ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr + treatment_pembrolizumab '
           '+ treatment_sotorasib + treatment_olaparib + treatment_osimertinib '
           '+ hemoglobin_g_dl + alkaline_phosphatase_u_l + ast_u_l + alt_u_l '
           '+ total_bilirubin_mg_dl + creatinine_mg_dl + bun_mg_dl + sodium_meq_l '
           '+ potassium_meq_l + calcium_mg_dl')
mvm = smf.ols(formula, data=df).fit()
print(mvm.summary())
# Save adjusted treatment effects
for tx in ['treatment_pembrolizumab', 'treatment_sotorasib', 'treatment_olaparib', 'treatment_osimertinib']:
    coef = mvm.params[tx]
    p = mvm.pvalues[tx]
    record(f"h_adj_{tx}",
           summary=f"Adjusted {tx} effect on log_pfs in full multivariable model: coef={coef:.3f}, p={p:.2e}",
           p=float(p), eff=float(coef))

# Save key adjusted feature effects
for feat in ['ecog_ps', 'stage_iv', 'has_brain_mets', 'pdl1_tps', 'tmb_high',
             'albumin_g_dl', 'ldh_u_l', 'nlr', 'crp_mg_l', 'weight_loss_pct_6mo',
             'egfr_mutation', 'kras_g12c', 'alk_fusion', 'stk11_mutation', 'brca2_mutation']:
    coef = mvm.params[feat]
    p = mvm.pvalues[feat]
    record(f"h_adj_{feat}",
           summary=f"Adjusted {feat} effect on log_pfs (multivariable): coef={coef:.3f}, p={p:.2e}",
           p=float(p), eff=float(coef))

# ---------------- Iteration 8: STK11 modifies pembrolizumab? joint model ----------------
print("\n=== Iteration 8: triple/joint subgroup search for pembrolizumab ===")

# Pembrolizumab in PDL1-high AND STK11-negative
sub = df.loc[(df.pdl1_high == 1) & (df.stk11_mutation == 0)]
on = sub.loc[sub.treatment_pembrolizumab == 1, 'pfs_months']
off = sub.loc[sub.treatment_pembrolizumab == 0, 'pfs_months']
eff, p = diff_means(on, off)
record("h_pem_pdl1high_stk11neg",
       summary=f"Pembrolizumab in PD-L1 high AND STK11-: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# Pembrolizumab in PDL1-high AND STK11-positive
sub = df.loc[(df.pdl1_high == 1) & (df.stk11_mutation == 1)]
on = sub.loc[sub.treatment_pembrolizumab == 1, 'pfs_months']
off = sub.loc[sub.treatment_pembrolizumab == 0, 'pfs_months']
eff, p = diff_means(on, off)
record("h_pem_pdl1high_stk11pos",
       summary=f"Pembrolizumab in PD-L1 high AND STK11+: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# Pembrolizumab in PDL1-high AND TMB-high
sub = df.loc[(df.pdl1_high == 1) & (df.tmb_high == 1)]
on = sub.loc[sub.treatment_pembrolizumab == 1, 'pfs_months']
off = sub.loc[sub.treatment_pembrolizumab == 0, 'pfs_months']
eff, p = diff_means(on, off)
record("h_pem_pdl1high_tmbhigh",
       summary=f"Pembrolizumab in PD-L1 high AND TMB high: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# Pembrolizumab in PDL1-high AND TMB-high AND STK11-negative
sub = df.loc[(df.pdl1_high == 1) & (df.tmb_high == 1) & (df.stk11_mutation == 0)]
on = sub.loc[sub.treatment_pembrolizumab == 1, 'pfs_months']
off = sub.loc[sub.treatment_pembrolizumab == 0, 'pfs_months']
eff, p = diff_means(on, off)
record("h_pem_pdl1high_tmbhigh_stk11neg",
       summary=f"Pembrolizumab in PD-L1 high AND TMB high AND STK11-: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# ---------------- Iteration 9: comprehensive treatment-effect heterogeneity screen ----------------
print("\n=== Iteration 9: treatment*feature interaction screen on log_pfs ===")

modifiers = ['age_years', 'sex_female', 'ecog_ps', 'stage_iv', 'has_brain_mets',
             'egfr_mutation', 'kras_g12c', 'alk_fusion', 'stk11_mutation',
             'brca2_mutation', 'pdl1_tps', 'tmb_high', 'albumin_g_dl', 'ldh_u_l',
             'weight_loss_pct_6mo', 'crp_mg_l', 'nlr', 'hemoglobin_g_dl',
             'alkaline_phosphatase_u_l', 'ast_u_l', 'alt_u_l', 'total_bilirubin_mg_dl',
             'creatinine_mg_dl', 'bun_mg_dl', 'sodium_meq_l', 'potassium_meq_l',
             'calcium_mg_dl', 'ever_smoker', 'pdl1_high']
treatments = ['treatment_pembrolizumab', 'treatment_sotorasib',
              'treatment_olaparib', 'treatment_osimertinib']

interaction_screen = []
for tx in treatments:
    for mod in modifiers:
        if mod == tx:
            continue
        try:
            m = smf.ols(f'log_pfs ~ {tx} * {mod}', data=df).fit()
            ikey = f'{tx}:{mod}'
            coef = m.params.get(ikey, np.nan)
            p = m.pvalues.get(ikey, np.nan)
            interaction_screen.append({"treatment": tx, "modifier": mod,
                                       "coef": float(coef), "p": float(p)})
        except Exception as e:
            print(f"  failed {tx} x {mod}: {e}")

# Sort and record top interactions per treatment
interaction_screen_sorted = sorted(interaction_screen, key=lambda x: x['p'])
print("\nTop interactions overall (lowest p):")
for row in interaction_screen_sorted[:30]:
    print(f"  {row['treatment']} x {row['modifier']}: coef={row['coef']:.3f}, p={row['p']:.2e}")

# Save the full screen
with open('interaction_screen.json', 'w') as f:
    json.dump(interaction_screen_sorted, f, indent=2)

# Record top per treatment
for tx in treatments:
    tx_rows = [r for r in interaction_screen_sorted if r['treatment'] == tx][:8]
    for i, r in enumerate(tx_rows):
        record(f"h_screen_{tx}_top{i+1}_{r['modifier']}",
               summary=f"Interaction {tx} * {r['modifier']} (log_pfs): coef={r['coef']:.3f}, p={r['p']:.2e}",
               p=r['p'], eff=r['coef'])

# ---------------- Iteration 10: subgroup definitions and tests ----------------
print("\n=== Iteration 10: refined subgroup tests ===")

# Osimertinib in EGFR+ AND no brain mets
sub = df.loc[(df.egfr_mutation == 1) & (df.has_brain_mets == 0)]
on = sub.loc[sub.treatment_osimertinib == 1, 'pfs_months']
off = sub.loc[sub.treatment_osimertinib == 0, 'pfs_months']
eff, p = diff_means(on, off)
record("h_osi_egfr_nobrainmets",
       summary=f"Osimertinib in EGFR+ AND no brain mets: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# Osimertinib in EGFR+ AND brain mets
sub = df.loc[(df.egfr_mutation == 1) & (df.has_brain_mets == 1)]
on = sub.loc[sub.treatment_osimertinib == 1, 'pfs_months']
off = sub.loc[sub.treatment_osimertinib == 0, 'pfs_months']
eff, p = diff_means(on, off)
record("h_osi_egfr_brainmets",
       summary=f"Osimertinib in EGFR+ AND brain mets: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

# Sotorasib in KRAS+ stratified by STK11 (STK11 known immunotherapy resistance, also affects KRAS)
for stk in [0, 1]:
    sub = df.loc[(df.kras_g12c == 1) & (df.stk11_mutation == stk)]
    on = sub.loc[sub.treatment_sotorasib == 1, 'pfs_months']
    off = sub.loc[sub.treatment_sotorasib == 0, 'pfs_months']
    eff, p = diff_means(on, off)
    record(f"h_sot_kras_stk11{stk}",
           summary=f"Sotorasib in KRAS+ AND STK11={stk}: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
           p=p, eff=eff)

# Sotorasib by ECOG within KRAS+
for ecog in [0, 1, 2]:
    sub = df.loc[(df.kras_g12c == 1) & (df.ecog_ps == ecog)]
    on = sub.loc[sub.treatment_sotorasib == 1, 'pfs_months']
    off = sub.loc[sub.treatment_sotorasib == 0, 'pfs_months']
    if len(on) > 0 and len(off) > 0:
        eff, p = diff_means(on, off)
        record(f"h_sot_kras_ecog{ecog}",
               summary=f"Sotorasib in KRAS+ AND ECOG={ecog}: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
               p=p, eff=eff)

# Olaparib in BRCA2+ — by stage / ECOG / PSA-equivalent biomarkers
sub = df.loc[df.brca2_mutation == 1]
on = sub.loc[sub.treatment_olaparib == 1, 'pfs_months']
off = sub.loc[sub.treatment_olaparib == 0, 'pfs_months']
eff, p = diff_means(on, off)
record("h_ola_brca2only",
       summary=f"Olaparib in BRCA2+: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
       p=p, eff=eff)

for ecog in [0, 1, 2]:
    sub = df.loc[(df.brca2_mutation == 1) & (df.ecog_ps == ecog)]
    on = sub.loc[sub.treatment_olaparib == 1, 'pfs_months']
    off = sub.loc[sub.treatment_olaparib == 0, 'pfs_months']
    if len(on) > 5 and len(off) > 5:
        eff, p = diff_means(on, off)
        record(f"h_ola_brca2_ecog{ecog}",
               summary=f"Olaparib in BRCA2+ AND ECOG={ecog}: n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
               p=p, eff=eff)

# Pembrolizumab joint subgroups, exhaustive small subgroups
print("\n=== Pembrolizumab subgroup grid ===")
pem_subs = []
for pdl in [0, 1]:
    for tmb in [0, 1]:
        for stk in [0, 1]:
            for sm in ['never', 'former', 'current']:
                sub = df.loc[(df.pdl1_high == pdl) & (df.tmb_high == tmb) &
                             (df.stk11_mutation == stk) & (df.smoking_status == sm)]
                on = sub.loc[sub.treatment_pembrolizumab == 1, 'pfs_months']
                off = sub.loc[sub.treatment_pembrolizumab == 0, 'pfs_months']
                if len(on) > 30 and len(off) > 30:
                    eff, p = diff_means(on, off)
                    pem_subs.append({
                        "pdl1_high": pdl, "tmb_high": tmb, "stk11_mutation": stk,
                        "smoking_status": sm, "n": len(sub), "diff": eff, "p": p,
                        "on_mean": float(on.mean()), "off_mean": float(off.mean())
                    })
pem_subs.sort(key=lambda r: -r['diff'])
print("\nTop pembro subgroups by effect size:")
for r in pem_subs[:10]:
    print(f"  pdl1={r['pdl1_high']} tmb={r['tmb_high']} stk11={r['stk11_mutation']} smoke={r['smoking_status']}: n={r['n']}, diff={r['diff']:.2f}, p={r['p']:.2e}")
print("\nBottom pembro subgroups by effect size:")
for r in pem_subs[-10:]:
    print(f"  pdl1={r['pdl1_high']} tmb={r['tmb_high']} stk11={r['stk11_mutation']} smoke={r['smoking_status']}: n={r['n']}, diff={r['diff']:.2f}, p={r['p']:.2e}")

with open('pem_subgroup_grid.json', 'w') as f:
    json.dump(pem_subs, f, indent=2)

# Record best/worst pembro subgroups
if pem_subs:
    best = pem_subs[0]
    record("h_pem_best_subgroup",
           summary=f"Best pembrolizumab subgroup: pdl1_high={best['pdl1_high']}, tmb_high={best['tmb_high']}, stk11={best['stk11_mutation']}, smoking={best['smoking_status']}; n={best['n']}; diff={best['diff']:.2f} mo, p={best['p']:.2e}",
           p=best['p'], eff=best['diff'])
    worst = pem_subs[-1]
    record("h_pem_worst_subgroup",
           summary=f"Worst pembrolizumab subgroup: pdl1_high={worst['pdl1_high']}, tmb_high={worst['tmb_high']}, stk11={worst['stk11_mutation']}, smoking={worst['smoking_status']}; n={worst['n']}; diff={worst['diff']:.2f} mo, p={worst['p']:.2e}",
           p=worst['p'], eff=worst['diff'])

# Similar grid for sotorasib
print("\n=== Sotorasib subgroup grid (KRAS+) ===")
sot_subs = []
for stk in [0, 1]:
    for ecog in [0, 1, 2]:
        for brain in [0, 1]:
            sub = df.loc[(df.kras_g12c == 1) & (df.stk11_mutation == stk) &
                         (df.ecog_ps == ecog) & (df.has_brain_mets == brain)]
            on = sub.loc[sub.treatment_sotorasib == 1, 'pfs_months']
            off = sub.loc[sub.treatment_sotorasib == 0, 'pfs_months']
            if len(on) > 10 and len(off) > 10:
                eff, p = diff_means(on, off)
                sot_subs.append({
                    "stk11": stk, "ecog": ecog, "brain": brain,
                    "n": len(sub), "diff": eff, "p": p
                })
sot_subs.sort(key=lambda r: -r['diff'])
print("Top KRAS+ sotorasib subgroups:")
for r in sot_subs[:10]:
    print(f"  stk11={r['stk11']} ecog={r['ecog']} brain={r['brain']}: n={r['n']}, diff={r['diff']:.2f}, p={r['p']:.2e}")
with open('sot_subgroup_grid.json', 'w') as f:
    json.dump(sot_subs, f, indent=2)

if sot_subs:
    best = sot_subs[0]
    record("h_sot_best_subgroup",
           summary=f"Best sotorasib subgroup (KRAS+): stk11={best['stk11']}, ecog={best['ecog']}, brain={best['brain']}; n={best['n']}; diff={best['diff']:.2f} mo, p={best['p']:.2e}",
           p=best['p'], eff=best['diff'])

# Olaparib grid (BRCA2+)
print("\n=== Olaparib grid (BRCA2+) ===")
ola_subs = []
for ecog in [0, 1, 2]:
    for brain in [0, 1]:
        for stage in [0, 1]:
            sub = df.loc[(df.brca2_mutation == 1) & (df.ecog_ps == ecog) &
                         (df.has_brain_mets == brain) & (df.stage_iv == stage)]
            on = sub.loc[sub.treatment_olaparib == 1, 'pfs_months']
            off = sub.loc[sub.treatment_olaparib == 0, 'pfs_months']
            if len(on) > 5 and len(off) > 5:
                eff, p = diff_means(on, off)
                ola_subs.append({
                    "ecog": ecog, "brain": brain, "stage_iv": stage,
                    "n": len(sub), "diff": eff, "p": p
                })
ola_subs.sort(key=lambda r: -r['diff'])
print("Top BRCA2+ olaparib subgroups:")
for r in ola_subs[:10]:
    print(f"  ecog={r['ecog']} brain={r['brain']} stage4={r['stage_iv']}: n={r['n']}, diff={r['diff']:.2f}, p={r['p']:.2e}")
with open('ola_subgroup_grid.json', 'w') as f:
    json.dump(ola_subs, f, indent=2)

if ola_subs:
    best = ola_subs[0]
    record("h_ola_best_subgroup",
           summary=f"Best olaparib subgroup (BRCA2+): ecog={best['ecog']}, brain={best['brain']}, stage_iv={best['stage_iv']}; n={best['n']}; diff={best['diff']:.2f} mo, p={best['p']:.2e}",
           p=best['p'], eff=best['diff'])

# Osimertinib grid (EGFR+)
print("\n=== Osimertinib grid (EGFR+) ===")
osi_subs = []
for ecog in [0, 1, 2]:
    for brain in [0, 1]:
        for stage in [0, 1]:
            sub = df.loc[(df.egfr_mutation == 1) & (df.ecog_ps == ecog) &
                         (df.has_brain_mets == brain) & (df.stage_iv == stage)]
            on = sub.loc[sub.treatment_osimertinib == 1, 'pfs_months']
            off = sub.loc[sub.treatment_osimertinib == 0, 'pfs_months']
            if len(on) > 5 and len(off) > 5:
                eff, p = diff_means(on, off)
                osi_subs.append({
                    "ecog": ecog, "brain": brain, "stage_iv": stage,
                    "n": len(sub), "diff": eff, "p": p
                })
osi_subs.sort(key=lambda r: -r['diff'])
print("Top EGFR+ osimertinib subgroups:")
for r in osi_subs[:10]:
    print(f"  ecog={r['ecog']} brain={r['brain']} stage4={r['stage_iv']}: n={r['n']}, diff={r['diff']:.2f}, p={r['p']:.2e}")
with open('osi_subgroup_grid.json', 'w') as f:
    json.dump(osi_subs, f, indent=2)

if osi_subs:
    best = osi_subs[0]
    record("h_osi_best_subgroup",
           summary=f"Best osimertinib subgroup (EGFR+): ecog={best['ecog']}, brain={best['brain']}, stage_iv={best['stage_iv']}; n={best['n']}; diff={best['diff']:.2f} mo, p={best['p']:.2e}",
           p=best['p'], eff=best['diff'])

# ---------------- Iteration 11: ALK fusion explorations ----------------
print("\n=== Iteration 11: ALK fusion ===")
sub = df.loc[df.alk_fusion == 1]
print(f"  ALK+ n={len(sub)}, mean PFS = {sub['pfs_months'].mean():.2f}")
# No targeted therapy for ALK in dataset; check pembrolizumab in ALK+
on = sub.loc[sub.treatment_pembrolizumab == 1, 'pfs_months']
off = sub.loc[sub.treatment_pembrolizumab == 0, 'pfs_months']
if len(on) > 5 and len(off) > 5:
    eff, p = diff_means(on, off)
    record("h_pem_in_alk",
           summary=f"Pembrolizumab in ALK+ (no targeted therapy in dataset): n={len(sub)}; on={on.mean():.2f} vs off={off.mean():.2f}; diff={eff:.2f} mo, p={p:.2e}",
           p=p, eff=eff)

# ---------------- Iteration 12: confirm final per-treatment best subgroups using interaction tests ----------------
print("\n=== Iteration 12: confirmatory tests ===")

# Confirm osimertinib in EGFR+ — 3-way interaction with brain mets and ecog
m = smf.ols('log_pfs ~ treatment_osimertinib * egfr_mutation', data=df).fit()
record("h_osi_egfr_log_interaction",
       summary=f"Confirmatory interaction osimertinib*egfr_mutation on log_pfs: coef={m.params['treatment_osimertinib:egfr_mutation']:.3f}, p={m.pvalues['treatment_osimertinib:egfr_mutation']:.2e}",
       p=float(m.pvalues['treatment_osimertinib:egfr_mutation']),
       eff=float(m.params['treatment_osimertinib:egfr_mutation']))

m = smf.ols('log_pfs ~ treatment_sotorasib * kras_g12c', data=df).fit()
record("h_sot_kras_log_interaction",
       summary=f"Confirmatory interaction sotorasib*kras_g12c on log_pfs: coef={m.params['treatment_sotorasib:kras_g12c']:.3f}, p={m.pvalues['treatment_sotorasib:kras_g12c']:.2e}",
       p=float(m.pvalues['treatment_sotorasib:kras_g12c']),
       eff=float(m.params['treatment_sotorasib:kras_g12c']))

m = smf.ols('log_pfs ~ treatment_olaparib * brca2_mutation', data=df).fit()
record("h_ola_brca_log_interaction",
       summary=f"Confirmatory interaction olaparib*brca2 on log_pfs: coef={m.params['treatment_olaparib:brca2_mutation']:.3f}, p={m.pvalues['treatment_olaparib:brca2_mutation']:.2e}",
       p=float(m.pvalues['treatment_olaparib:brca2_mutation']),
       eff=float(m.params['treatment_olaparib:brca2_mutation']))

m = smf.ols('log_pfs ~ treatment_pembrolizumab * pdl1_high', data=df).fit()
record("h_pem_pdl1_log_interaction",
       summary=f"Confirmatory interaction pembro*pdl1_high on log_pfs: coef={m.params['treatment_pembrolizumab:pdl1_high']:.3f}, p={m.pvalues['treatment_pembrolizumab:pdl1_high']:.2e}",
       p=float(m.pvalues['treatment_pembrolizumab:pdl1_high']),
       eff=float(m.params['treatment_pembrolizumab:pdl1_high']))

m = smf.ols('log_pfs ~ treatment_pembrolizumab * stk11_mutation', data=df).fit()
record("h_pem_stk11_log_interaction",
       summary=f"Confirmatory interaction pembro*stk11 on log_pfs: coef={m.params['treatment_pembrolizumab:stk11_mutation']:.3f}, p={m.pvalues['treatment_pembrolizumab:stk11_mutation']:.2e}",
       p=float(m.pvalues['treatment_pembrolizumab:stk11_mutation']),
       eff=float(m.params['treatment_pembrolizumab:stk11_mutation']))

# 3-way: pembro * pdl1_high * stk11
m = smf.ols('log_pfs ~ treatment_pembrolizumab * pdl1_high * stk11_mutation', data=df).fit()
key = 'treatment_pembrolizumab:pdl1_high:stk11_mutation'
if key in m.params:
    record("h_pem_pdl1_stk11_3way",
           summary=f"Three-way pembro * pdl1_high * stk11 on log_pfs: coef={m.params[key]:.3f}, p={m.pvalues[key]:.2e}",
           p=float(m.pvalues[key]), eff=float(m.params[key]))

# Save
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nTotal records: {len(results)}")
print("Saved results.json")
