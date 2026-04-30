"""Iterative analysis of ds001_breast. Run once, prints all results to stdout."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
print(f"N={len(df)} cols={df.shape[1]}")
print()


def lr(formula, label=""):
    """Run an OLS, return coef table summary."""
    m = smf.ols(formula, data=df).fit()
    if label:
        print(f"--- {label} ---")
    print(f"Formula: {formula}")
    # Compact
    for name, coef, p in zip(m.params.index, m.params.values, m.pvalues.values):
        print(f"  {name:50s} coef={coef:+.4f}  p={p:.3g}")
    print()
    return m


def two_group(col_y, col_g, label=""):
    g1 = df.loc[df[col_g] == 1, col_y]
    g0 = df.loc[df[col_g] == 0, col_y]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    diff = g1.mean() - g0.mean()
    print(f"{label or (col_y+'~'+col_g)}: mean({col_g}=1)={g1.mean():.3f} (n={len(g1)}), mean({col_g}=0)={g0.mean():.3f} (n={len(g0)}), diff={diff:+.4f}, p={p:.3g}")
    return diff, p


# ====================================================================
print("=" * 80)
print("ITERATION 1: Main effects of each treatment on PFS")
print("=" * 80)
for t in ['treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
          'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab']:
    two_group('pfs_months', t, label=f"PFS by {t}")
print()

# Adjusted main effects
print("Adjusted main effects (mutually adjusted + key prognostics):")
m = lr("pfs_months ~ treatment_tamoxifen + treatment_palbociclib + treatment_trastuzumab "
       "+ treatment_olaparib + treatment_sacituzumab_govitecan + treatment_pembrolizumab "
       "+ age_years + ecog_ps + stage_iv + has_brain_mets + tumor_size_cm + ki67_pct",
       label="PFS adjusted main effects")

# ====================================================================
print("=" * 80)
print("ITERATION 2: Biomarker main effects + ER/PR/HER2/BRCA/PIK3CA on PFS")
print("=" * 80)
for b in ['er_positive','pr_positive','her2_positive','her2_low','brca1_mutation',
          'brca2_mutation','pik3ca_mutation','her2_amplification','tp53_mutation',
          'pten_loss','msi_high','node_positive','postmenopausal']:
    two_group('pfs_months', b, label=f"PFS by {b}")
print()

m = lr("pfs_months ~ er_positive + pr_positive + her2_positive + her2_low "
       "+ brca1_mutation + brca2_mutation + pik3ca_mutation + tp53_mutation + pten_loss + msi_high "
       "+ node_positive + postmenopausal + age_years + ecog_ps + stage_iv + has_brain_mets",
       label="PFS adjusted biomarkers")

# ====================================================================
print("=" * 80)
print("ITERATION 3: Treatment x biomarker interactions (canonical breast pairings)")
print("=" * 80)
# Tamoxifen x ER
m = lr("pfs_months ~ treatment_tamoxifen * er_positive + age_years + ecog_ps + stage_iv",
       label="Tamoxifen x ER+")
# Trastuzumab x HER2
m = lr("pfs_months ~ treatment_trastuzumab * her2_positive + age_years + ecog_ps + stage_iv",
       label="Trastuzumab x HER2+")
# Olaparib x BRCA
df['brca_any'] = ((df['brca1_mutation']==1) | (df['brca2_mutation']==1)).astype(int)
m = lr("pfs_months ~ treatment_olaparib * brca_any + age_years + ecog_ps + stage_iv",
       label="Olaparib x BRCA1/2 mutation")
# Palbociclib x ER+/HER2-
df['er_pos_her2_neg'] = ((df['er_positive']==1) & (df['her2_positive']==0)).astype(int)
m = lr("pfs_months ~ treatment_palbociclib * er_pos_her2_neg + age_years + ecog_ps + stage_iv",
       label="Palbociclib x ER+/HER2-")
# Pembrolizumab x MSI-H
m = lr("pfs_months ~ treatment_pembrolizumab * msi_high + age_years + ecog_ps + stage_iv",
       label="Pembrolizumab x MSI-H")
# Sacituzumab x triple negative (ER-/PR-/HER2-)
df['triple_neg'] = ((df['er_positive']==0) & (df['pr_positive']==0) & (df['her2_positive']==0)).astype(int)
print(f"Triple-negative prevalence: {df['triple_neg'].mean():.3f}")
m = lr("pfs_months ~ treatment_sacituzumab_govitecan * triple_neg + age_years + ecog_ps + stage_iv",
       label="Sacituzumab x triple-neg")

# ====================================================================
print("=" * 80)
print("ITERATION 4: Prognostic continuous variables (labs) on PFS")
print("=" * 80)
labs = ['albumin_g_dl','ldh_u_l','crp_mg_l','nlr','hemoglobin_g_dl','alkaline_phosphatase_u_l',
        'ast_u_l','alt_u_l','total_bilirubin_mg_dl','creatinine_mg_dl','calcium_mg_dl',
        'platelets_k_ul','wbc_k_ul','anc_k_ul','alc_k_ul','weight_loss_pct_6mo',
        'ca_125_u_ml','cea_ng_ml']
rows = []
for v in labs:
    r, p = stats.pearsonr(df[v], df['pfs_months'])
    rows.append((v, r, p))
rows.sort(key=lambda x: x[2])
for v, r, p in rows:
    print(f"  {v:35s} pearson r={r:+.4f}  p={p:.3g}")
print()

# ====================================================================
print("=" * 80)
print("ITERATION 5: ECOG / stage / brain mets / weight loss as prognostics")
print("=" * 80)
m = lr("pfs_months ~ ecog_ps + stage_iv + has_brain_mets + liver_mets + bone_mets "
       "+ weight_loss_pct_6mo + albumin_g_dl + ldh_u_l + nlr + age_years",
       label="Multivariable prognostic")

# ECOG by group
for ec in [0,1,2]:
    print(f"  ECOG={ec}: PFS mean={df.loc[df['ecog_ps']==ec,'pfs_months'].mean():.3f} (n={(df['ecog_ps']==ec).sum()})")
print()

# ====================================================================
print("=" * 80)
print("ITERATION 6: Demographics (race, insurance, rural) on PFS")
print("=" * 80)
print("PFS by race:")
for r, g in df.groupby('race_ethnicity'):
    print(f"  {r:10s} mean={g['pfs_months'].mean():.3f} n={len(g)}")
# ANOVA
groups = [g['pfs_months'].values for _, g in df.groupby('race_ethnicity')]
f, p = stats.f_oneway(*groups)
print(f"  ANOVA F={f:.3f} p={p:.3g}")
print()

print("PFS by insurance:")
for r, g in df.groupby('insurance_type'):
    print(f"  {r:10s} mean={g['pfs_months'].mean():.3f} n={len(g)}")
groups = [g['pfs_months'].values for _, g in df.groupby('insurance_type')]
f, p = stats.f_oneway(*groups)
print(f"  ANOVA F={f:.3f} p={p:.3g}")
print()

two_group('pfs_months','rural_residence', label='PFS by rural residence')

# Adjusted model with demographics
df_dum = pd.get_dummies(df[['race_ethnicity','insurance_type']], drop_first=True).astype(int)
df2 = pd.concat([df, df_dum], axis=1)
race_cols = [c for c in df_dum.columns if c.startswith('race_ethnicity_')]
ins_cols  = [c for c in df_dum.columns if c.startswith('insurance_type_')]
formula = ("pfs_months ~ " + " + ".join(race_cols + ins_cols)
           + " + rural_residence + age_years + ecog_ps + stage_iv + has_brain_mets")
m = smf.ols(formula, data=df2).fit()
print(f"Adjusted demographics: {formula}")
for name, coef, p in zip(m.params.index, m.params.values, m.pvalues.values):
    print(f"  {name:50s} coef={coef:+.4f}  p={p:.3g}")
print()

# ====================================================================
print("=" * 80)
print("ITERATION 7: SNP main effects on PFS (screen all SNPs)")
print("=" * 80)
snp_cols = [c for c in df.columns if c.startswith('snp_')]
print(f"# SNPs: {len(snp_cols)}")
snp_results = []
for s in snp_cols:
    # treat as ordinal 0/1/2 if numeric
    r, p = stats.pearsonr(df[s], df['pfs_months'])
    snp_results.append((s, r, p))
snp_results.sort(key=lambda x: x[2])
print("Top 10 SNPs by p-value:")
for s, r, p in snp_results[:10]:
    print(f"  {s:25s} r={r:+.4f}  p={p:.3g}")
# Bonferroni cutoff
n_tests = len(snp_cols)
bonf = 0.05 / n_tests
print(f"\nBonferroni cutoff = {bonf:.3g}; # significant: {sum(1 for _,_,p in snp_results if p<bonf)}")

# ====================================================================
print("=" * 80)
print("ITERATION 8: Race x treatment interactions (equity check)")
print("=" * 80)
df2['white'] = (df2['race_ethnicity']=='white').astype(int)
df2['black'] = (df2['race_ethnicity']=='black').astype(int)
for t in ['treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
          'treatment_pembrolizumab','treatment_sacituzumab_govitecan','treatment_olaparib']:
    f = f"pfs_months ~ {t} * black + age_years + ecog_ps + stage_iv"
    m = smf.ols(f, data=df2).fit()
    inter = f"{t}:black"
    if inter in m.params.index:
        print(f"  {t} x black-race interaction: coef={m.params[inter]:+.4f} p={m.pvalues[inter]:.3g}")
print()

# Race differences in treatment receipt
print("Treatment receipt by race (proportion):")
for t in ['treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
          'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab']:
    print(f"  {t}:")
    for r, g in df.groupby('race_ethnicity'):
        print(f"    {r:10s} {g[t].mean():.4f}")

# ====================================================================
print("=" * 80)
print("ITERATION 9: Comorbidities and symptoms on PFS")
print("=" * 80)
comorbs = ['diabetes_mellitus','hypertension','copd','chronic_kidney_disease','heart_failure',
           'coronary_artery_disease','atrial_fibrillation','venous_thromboembolism_history',
           'autoimmune_disease','prior_malignancy','depression_anxiety_diagnosis',
           'interstitial_lung_disease_history']
for c in comorbs:
    two_group('pfs_months', c, label=f"PFS by {c}")
print()

print("Symptom grades (Pearson r vs PFS):")
for s in ['fatigue_grade','pain_nrs','dyspnea_grade','cough_grade','appetite_loss_grade']:
    r, p = stats.pearsonr(df[s], df['pfs_months'])
    print(f"  {s:25s} r={r:+.4f} p={p:.3g}")

# Comprehensive multivariable
print()
m = lr("pfs_months ~ diabetes_mellitus + hypertension + copd + chronic_kidney_disease "
       "+ heart_failure + venous_thromboembolism_history + autoimmune_disease "
       "+ fatigue_grade + pain_nrs + dyspnea_grade + appetite_loss_grade "
       "+ age_years + ecog_ps + stage_iv",
       label="Comorbidities + symptoms adjusted")

# ====================================================================
print("=" * 80)
print("ITERATION 10: Triple-checks and additional interactions")
print("=" * 80)
# PIK3CA x palbociclib (BYLieve / SOLAR-1 era thinking)
m = lr("pfs_months ~ treatment_palbociclib * pik3ca_mutation + er_positive + age_years + ecog_ps",
       label="Palbociclib x PIK3CA")
# Pembrolizumab x triple-neg
m = lr("pfs_months ~ treatment_pembrolizumab * triple_neg + age_years + ecog_ps + stage_iv",
       label="Pembro x triple-negative")
# Olaparib x HRD-like (BRCA already done) - also try in non-BRCA
df['no_brca'] = (df['brca_any']==0).astype(int)
m = lr("pfs_months ~ treatment_olaparib * no_brca + age_years + ecog_ps",
       label="Olaparib effect when no BRCA mutation")
# Trastuzumab x HER2-low
m = lr("pfs_months ~ treatment_trastuzumab * her2_low + age_years + ecog_ps",
       label="Trastuzumab x HER2-low")
# Postmenopausal x palbociclib
m = lr("pfs_months ~ treatment_palbociclib * postmenopausal + er_positive + age_years",
       label="Palbociclib x postmenopausal")
# Tamoxifen x postmenopausal
m = lr("pfs_months ~ treatment_tamoxifen * postmenopausal + er_positive + age_years",
       label="Tamoxifen x postmenopausal")
# Brain mets x pembrolizumab
m = lr("pfs_months ~ treatment_pembrolizumab * has_brain_mets + age_years + ecog_ps + stage_iv",
       label="Pembro x brain mets")

# Treatment receipt by sex (this dataset is mixed-sex despite breast label)
print("\nSex distribution: female=", df['sex_female'].mean())
two_group('pfs_months','sex_female', label='PFS by sex_female')
print("Treatments by sex_female:")
for t in ['treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
          'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab']:
    f1 = df.loc[df['sex_female']==1, t].mean()
    f0 = df.loc[df['sex_female']==0, t].mean()
    print(f"  {t}: F={f1:.3f} M={f0:.3f}")
