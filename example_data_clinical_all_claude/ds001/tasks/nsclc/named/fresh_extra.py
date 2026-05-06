import json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
df['hist_sq'] = (df['histology']=='squamous').astype(int)
df['smk_never'] = (df['smoking_status']=='never').astype(int)
df['smk_current'] = (df['smoking_status']=='current').astype(int)

OUT = {}

# 1) sotorasib effect: kras x sex stratified
res = {}
for kras in [0,1]:
    for sex in [0,1]:
        sub = df[(df['kras_g12c']==kras) & (df['sex_female']==sex)]
        a = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
        n1 = int((sub['treatment_sotorasib']==1).sum())
        n0 = int((sub['treatment_sotorasib']==0).sum())
        if n1>0 and n0>0:
            t = stats.ttest_ind(sub.loc[sub['treatment_sotorasib']==1,'pfs_months'], sub.loc[sub['treatment_sotorasib']==0,'pfs_months'], equal_var=False)
            res[f'kras={kras}_sex_female={sex}'] = {'effect': float(a), 'n_total': int(sub.shape[0]), 'n_on': n1, 'n_off': n0, 'p': float(t.pvalue)}
m = smf.ols('pfs_months ~ treatment_sotorasib * kras_g12c * sex_female', data=df).fit()
for t in m.params.index:
    if 'treatment_sotorasib' in t:
        res[t] = {'beta': float(m.params[t]), 'p': float(m.pvalues[t])}
OUT['sot_kras_sex'] = res
print('sot_kras_sex', res)

# 2) Confirm: sotorasib in KRAS+ males vs KRAS+ females
res = {}
sub = df[(df['kras_g12c']==1) & (df['sex_female']==0)]
m1 = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean()
m0 = sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
res['kras1_male_means'] = {'on': float(m1), 'off': float(m0), 'diff': float(m1-m0), 'n_on': int((sub['treatment_sotorasib']==1).sum()), 'n_off': int((sub['treatment_sotorasib']==0).sum())}
sub = df[(df['kras_g12c']==1) & (df['sex_female']==1)]
m1 = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean()
m0 = sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
res['kras1_female_means'] = {'on': float(m1), 'off': float(m0), 'diff': float(m1-m0), 'n_on': int((sub['treatment_sotorasib']==1).sum()), 'n_off': int((sub['treatment_sotorasib']==0).sum())}
OUT['sot_kras_sex_means'] = res
print(res)

# 3) Smoking pattern: sotorasib effect in kras1 by smoking
res = {}
for smk in ['current','former','never']:
    sub = df[(df['kras_g12c']==1) & (df['smoking_status']==smk)]
    a = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
    n1 = int((sub['treatment_sotorasib']==1).sum()); n0 = int((sub['treatment_sotorasib']==0).sum())
    res[f'kras1_smk={smk}'] = {'effect': float(a), 'n': int(sub.shape[0]), 'n_on': n1, 'n_off': n0}
OUT['sot_kras_smoking'] = res
print(res)

# 4) Osimertinib + ALK confirmation
res = {}
for alk in [0,1]:
    for egfr in [0,1]:
        sub = df[(df['alk_fusion']==alk) & (df['egfr_mutation']==egfr)]
        if sub.shape[0]<50: continue
        a = sub.loc[sub['treatment_osimertinib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_osimertinib']==0,'pfs_months'].mean()
        n1 = int((sub['treatment_osimertinib']==1).sum()); n0 = int((sub['treatment_osimertinib']==0).sum())
        if n1>0 and n0>0:
            t = stats.ttest_ind(sub.loc[sub['treatment_osimertinib']==1,'pfs_months'], sub.loc[sub['treatment_osimertinib']==0,'pfs_months'], equal_var=False)
            res[f'alk={alk}_egfr={egfr}'] = {'effect': float(a), 'n': int(sub.shape[0]), 'p': float(t.pvalue)}
OUT['osi_alk_egfr'] = res
print(res)

# 5) Pembrolizumab additional probes: KRAS G12C as suppressor?
res = {}
m = smf.ols('pfs_months ~ treatment_pembrolizumab * kras_g12c', data=df).fit()
for t in m.params.index:
    if 'treatment_pembrolizumab' in t:
        res[t] = {'beta': float(m.params[t]), 'p': float(m.pvalues[t])}
# pembro effect in pdl1>=0.5 AND stk11=0 AND kras_g12c=0
sub = df[(df['pdl1_tps']>=0.5) & (df['stk11_mutation']==0) & (df['kras_g12c']==0)]
a = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months'].mean()
res['pembro_pdl1ge0.5_stk11=0_kras=0'] = {'effect': float(a), 'n': int(sub.shape[0])}
sub = df[(df['pdl1_tps']>=0.5) & (df['stk11_mutation']==0) & (df['kras_g12c']==1)]
a = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months'].mean()
res['pembro_pdl1ge0.5_stk11=0_kras=1'] = {'effect': float(a), 'n': int(sub.shape[0])}
OUT['pembro_kras'] = res
print(res)

# 6) Olaparib brute: search 3-feature subgroups
import itertools
binary_mods = ['sex_female','stage_iv','has_brain_mets','egfr_mutation','kras_g12c','alk_fusion',
               'stk11_mutation','brca2_mutation','tmb_high','hist_sq','smk_never','smk_current']
def best_3(tx):
    rows = []
    for f1,f2,f3 in itertools.combinations(binary_mods,3):
        for v1 in [0,1]:
            for v2 in [0,1]:
                for v3 in [0,1]:
                    sub = df[(df[f1]==v1) & (df[f2]==v2) & (df[f3]==v3)]
                    if sub.shape[0] < 200: continue
                    n1 = int((sub[tx]==1).sum()); n0 = int((sub[tx]==0).sum())
                    if n1<30 or n0<30: continue
                    a = sub.loc[sub[tx]==1,'pfs_months'].mean() - sub.loc[sub[tx]==0,'pfs_months'].mean()
                    rows.append({'sub': f'{f1}={v1}, {f2}={v2}, {f3}={v3}', 'n': int(sub.shape[0]), 'effect': float(a)})
    rows.sort(key=lambda r: r['effect'], reverse=True)
    return rows[:10]
res = {}
for tx in ['treatment_pembrolizumab','treatment_olaparib','treatment_osimertinib']:
    res[tx] = best_3(tx)
OUT['brute_3'] = res
print('brute3 done')

# 7) Sotorasib in kras+ male: confirm with regression
res = {}
sub = df[(df['kras_g12c']==1) & (df['sex_female']==0)]
m = smf.ols('pfs_months ~ treatment_sotorasib + ecog_ps + albumin_g_dl + has_brain_mets + ldh_u_l', data=sub).fit()
res['sot_kras1_male_adj'] = {'beta_sot': float(m.params['treatment_sotorasib']), 'p': float(m.pvalues['treatment_sotorasib']), 'n': int(sub.shape[0])}
sub = df[(df['kras_g12c']==1) & (df['sex_female']==1)]
m = smf.ols('pfs_months ~ treatment_sotorasib + ecog_ps + albumin_g_dl + has_brain_mets + ldh_u_l', data=sub).fit()
res['sot_kras1_female_adj'] = {'beta_sot': float(m.params['treatment_sotorasib']), 'p': float(m.pvalues['treatment_sotorasib']), 'n': int(sub.shape[0])}
OUT['sot_adj'] = res
print(res)

with open('fresh_extra.json','w') as f:
    json.dump(OUT, f, indent=2)
