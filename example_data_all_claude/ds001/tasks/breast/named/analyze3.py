"""Iter 11-18: Drill into palbociclib subgroups, joint biomarker effects."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_parquet("dataset.parquet")
OUT = {}

def palbo_effect_in(mask, label):
    sub = df[mask]
    a = sub.loc[sub['treatment_palbociclib']==1, 'pfs_months']
    b = sub.loc[sub['treatment_palbociclib']==0, 'pfs_months']
    if len(a)<5 or len(b)<5:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        'label': label, 'n_tx': len(a), 'n_no': len(b),
        'mean_tx': float(a.mean()), 'mean_no': float(b.mean()),
        'diff': float(a.mean()-b.mean()), 'p': float(p)
    }

# === Iter 11: palbociclib in joint biomarker strata ===
print("\n=== ITER 11: Palbociclib effect across joint biomarker subgroups ===")
results = []
strata = [
    (df['er_positive']==1, 'ER+'),
    (df['er_positive']==0, 'ER-'),
    ((df['er_positive']==1) & (df['her2_positive']==0), 'ER+ HER2-'),
    ((df['er_positive']==1) & (df['her2_positive']==1), 'ER+ HER2+'),
    ((df['er_positive']==1) & (df['her2_positive']==0) & (df['pik3ca_mutation']==0), 'ER+ HER2- PIK3CA-wt'),
    ((df['er_positive']==1) & (df['her2_positive']==0) & (df['pik3ca_mutation']==1), 'ER+ HER2- PIK3CA-mut'),
    ((df['er_positive']==1) & (df['her2_positive']==1) & (df['pik3ca_mutation']==0), 'ER+ HER2+ PIK3CA-wt'),
    ((df['pik3ca_mutation']==0), 'PIK3CA-wt all'),
    ((df['pik3ca_mutation']==1), 'PIK3CA-mut all'),
    ((df['er_positive']==1) & (df['pik3ca_mutation']==0), 'ER+ PIK3CA-wt'),
    ((df['er_positive']==1) & (df['pik3ca_mutation']==1), 'ER+ PIK3CA-mut'),
    ((df['er_positive']==1) & (df['her2_positive']==0) & (df['pik3ca_mutation']==0) & (df['ki67_pct'] < 20),
     'ER+ HER2- PIK3CA-wt ki67<20'),
    ((df['er_positive']==1) & (df['her2_positive']==0) & (df['pik3ca_mutation']==0) & (df['ki67_pct'] >= 20),
     'ER+ HER2- PIK3CA-wt ki67>=20'),
    ((df['er_positive']==1) & (df['her2_positive']==0) & (df['pr_positive']==1) & (df['pik3ca_mutation']==0),
     'ER+ PR+ HER2- PIK3CA-wt'),
    ((df['er_positive']==1) & (df['her2_positive']==0) & (df['pr_positive']==0) & (df['pik3ca_mutation']==0),
     'ER+ PR- HER2- PIK3CA-wt'),
    ((df['er_positive']==1) & (df['her2_positive']==0) & (df['her2_low']==1) & (df['pik3ca_mutation']==0),
     'ER+ HER2- HER2low PIK3CA-wt'),
    ((df['er_positive']==1) & (df['her2_positive']==0) & (df['her2_low']==0) & (df['pik3ca_mutation']==0),
     'ER+ HER2- HER2-zero PIK3CA-wt'),
]
for mask, lab in strata:
    r = palbo_effect_in(mask, lab)
    if r:
        results.append(r)
        print(f"  Palbo in {lab:50s}: n_tx={r['n_tx']:5d} n_no={r['n_no']:5d} diff={r['diff']:+.3f} p={r['p']:.2e}")
OUT['palbo_subgroups'] = results

# === Iter 12: Olaparib x BRCA (HRD) joint ===
print("\n=== ITER 12: Olaparib in BRCA-mut subgroups ===")
ola_strata = [
    (df['brca1_mutation']==1, 'BRCA1-mut'),
    (df['brca2_mutation']==1, 'BRCA2-mut'),
    ((df['brca1_mutation']==1) | (df['brca2_mutation']==1), 'BRCA1 or BRCA2-mut'),
    ((df['brca1_mutation']==0) & (df['brca2_mutation']==0), 'BRCA1/2 wild-type'),
]
ola_results = []
for mask, lab in ola_strata:
    sub = df[mask]
    a = sub.loc[sub['treatment_olaparib']==1, 'pfs_months']
    b = sub.loc[sub['treatment_olaparib']==0, 'pfs_months']
    if len(a)>5 and len(b)>5:
        t,p = stats.ttest_ind(a,b,equal_var=False)
        r = {'label':lab,'n_tx':len(a),'n_no':len(b),'mean_tx':float(a.mean()),'mean_no':float(b.mean()),
             'diff':float(a.mean()-b.mean()),'p':float(p)}
        ola_results.append(r)
        print(f"  Olaparib in {lab:30s}: n_tx={r['n_tx']:4d} n_no={r['n_no']:5d} diff={r['diff']:+.3f} p={r['p']:.2e}")
# formal interaction olaparib x (brca1+brca2)
df['brca_any'] = ((df['brca1_mutation']==1) | (df['brca2_mutation']==1)).astype(int)
X = pd.DataFrame({'tx':df['treatment_olaparib'],'mod':df['brca_any'],
                  'inter':df['treatment_olaparib']*df['brca_any']})
X = sm.add_constant(X)
fit = sm.OLS(df['pfs_months'], X).fit()
ola_int = {'tx':'olaparib','mod':'brca_any',
           'coef':float(fit.params['inter']),'p':float(fit.pvalues['inter'])}
print(f"  Olaparib x brca_any interaction coef={ola_int['coef']:+.3f} p={ola_int['p']:.2e}")
OUT['olaparib_subgroups'] = ola_results
OUT['olaparib_brca_interaction'] = ola_int

# === Iter 13: Trastuzumab x HER2-positive ===
print("\n=== ITER 13: Trastuzumab in HER2-positive vs negative ===")
trastu_strata = [
    (df['her2_positive']==1, 'HER2+'),
    (df['her2_positive']==0, 'HER2-'),
    (df['her2_low']==1, 'HER2-low'),
    ((df['her2_positive']==0) & (df['her2_low']==0), 'HER2-zero'),
]
trastu_results = []
for mask, lab in trastu_strata:
    sub = df[mask]
    a = sub.loc[sub['treatment_trastuzumab']==1, 'pfs_months']
    b = sub.loc[sub['treatment_trastuzumab']==0, 'pfs_months']
    if len(a)>5 and len(b)>5:
        t,p = stats.ttest_ind(a,b,equal_var=False)
        r = {'label':lab,'n_tx':len(a),'n_no':len(b),'mean_tx':float(a.mean()),'mean_no':float(b.mean()),
             'diff':float(a.mean()-b.mean()),'p':float(p)}
        trastu_results.append(r)
        print(f"  Trastuzumab in {lab:15s}: n_tx={r['n_tx']:4d} n_no={r['n_no']:5d} diff={r['diff']:+.3f} p={r['p']:.2e}")
OUT['trastu_subgroups'] = trastu_results

# === Iter 14: Pembrolizumab subgroups (TNBC pattern?) ===
print("\n=== ITER 14: Pembrolizumab in subgroups ===")
df['tnbc_like'] = ((df['er_positive']==0) & (df['pr_positive']==0) & (df['her2_positive']==0)).astype(int)
pembro_strata = [
    (df['tnbc_like']==1, 'TNBC-like (ER-,PR-,HER2-)'),
    (df['tnbc_like']==0, 'Non-TNBC'),
    ((df['er_positive']==0), 'ER-'),
    ((df['er_positive']==1), 'ER+'),
    ((df['her2_positive']==0) & (df['er_positive']==0), 'ER- HER2-'),
    ((df['ki67_pct']>=20), 'High Ki67 (>=20%)'),
    ((df['ki67_pct']<20), 'Low Ki67 (<20%)'),
    ((df['her2_low']==1) & (df['er_positive']==0), 'HER2-low ER-'),
]
pembro_results = []
for mask, lab in pembro_strata:
    sub = df[mask]
    a = sub.loc[sub['treatment_pembrolizumab']==1, 'pfs_months']
    b = sub.loc[sub['treatment_pembrolizumab']==0, 'pfs_months']
    if len(a)>5 and len(b)>5:
        t,p = stats.ttest_ind(a,b,equal_var=False)
        r = {'label':lab,'n_tx':len(a),'n_no':len(b),'mean_tx':float(a.mean()),'mean_no':float(b.mean()),
             'diff':float(a.mean()-b.mean()),'p':float(p)}
        pembro_results.append(r)
        print(f"  Pembro in {lab:30s}: n_tx={r['n_tx']:4d} n_no={r['n_no']:5d} diff={r['diff']:+.3f} p={r['p']:.2e}")
OUT['pembro_subgroups'] = pembro_results

# === Iter 15: Sacituzumab subgroups (HER2-low? TNBC?) ===
print("\n=== ITER 15: Sacituzumab govitecan in subgroups ===")
saci_strata = [
    (df['tnbc_like']==1, 'TNBC-like'),
    (df['her2_low']==1, 'HER2-low'),
    ((df['her2_low']==1) & (df['tnbc_like']==1), 'TNBC HER2-low'),
    ((df['her2_low']==1) & (df['er_positive']==1), 'ER+ HER2-low'),
    (df['er_positive']==1, 'ER+'),
]
saci_results = []
for mask, lab in saci_strata:
    sub = df[mask]
    a = sub.loc[sub['treatment_sacituzumab_govitecan']==1, 'pfs_months']
    b = sub.loc[sub['treatment_sacituzumab_govitecan']==0, 'pfs_months']
    if len(a)>5 and len(b)>5:
        t,p = stats.ttest_ind(a,b,equal_var=False)
        r = {'label':lab,'n_tx':len(a),'n_no':len(b),'mean_tx':float(a.mean()),'mean_no':float(b.mean()),
             'diff':float(a.mean()-b.mean()),'p':float(p)}
        saci_results.append(r)
        print(f"  Saci in {lab:30s}: n_tx={r['n_tx']:4d} n_no={r['n_no']:5d} diff={r['diff']:+.3f} p={r['p']:.2e}")
OUT['saci_subgroups'] = saci_results

# === Iter 16: Tamoxifen subgroups ===
print("\n=== ITER 16: Tamoxifen subgroups ===")
tam_strata = [
    (df['er_positive']==1, 'ER+'),
    (df['er_positive']==0, 'ER-'),
    ((df['er_positive']==1) & (df['postmenopausal']==1), 'ER+ post-meno'),
    ((df['er_positive']==1) & (df['postmenopausal']==0), 'ER+ pre-meno'),
]
tam_results = []
for mask, lab in tam_strata:
    sub = df[mask]
    a = sub.loc[sub['treatment_tamoxifen']==1, 'pfs_months']
    b = sub.loc[sub['treatment_tamoxifen']==0, 'pfs_months']
    if len(a)>5 and len(b)>5:
        t,p = stats.ttest_ind(a,b,equal_var=False)
        r = {'label':lab,'n_tx':len(a),'n_no':len(b),'mean_tx':float(a.mean()),'mean_no':float(b.mean()),
             'diff':float(a.mean()-b.mean()),'p':float(p)}
        tam_results.append(r)
        print(f"  Tamox in {lab:25s}: n_tx={r['n_tx']:4d} n_no={r['n_no']:5d} diff={r['diff']:+.3f} p={r['p']:.2e}")
OUT['tamox_subgroups'] = tam_results

with open("results_iter11_16.json","w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("\nSaved results_iter11_16.json")
