"""Treatment x biomarker interactions on pfs_months."""
import pandas as pd, numpy as np, json
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
out = {}

# Helper: simulated treatment effect within subgroup
def tx_effect(tx, mask, label):
    sub = df[mask]
    a = sub.loc[sub[tx]==1,'pfs_months']
    b = sub.loc[sub[tx]==0,'pfs_months']
    if len(a)<5 or len(b)<5: return None
    t,p = stats.ttest_ind(a,b,equal_var=False)
    return {
        'subgroup':label,'tx':tx,
        'n_on':int(len(a)),'n_off':int(len(b)),
        'mean_on':float(a.mean()),'mean_off':float(b.mean()),
        'diff':float(a.mean()-b.mean()),'p':float(p)
    }

# Tamoxifen × ER status
out['tam_ER+'] = tx_effect('treatment_tamoxifen', df['er_positive']==1, 'er_positive')
out['tam_ER-'] = tx_effect('treatment_tamoxifen', df['er_positive']==0, 'er_negative')

# Tamoxifen × postmenopausal × ER
out['tam_ER+_postmeno'] = tx_effect('treatment_tamoxifen', (df['er_positive']==1)&(df['postmenopausal']==1), 'er+_postmeno')
out['tam_ER+_premeno'] = tx_effect('treatment_tamoxifen', (df['er_positive']==1)&(df['postmenopausal']==0), 'er+_premeno')

# Palbociclib × ER × postmenopausal
out['palbo_ER+'] = tx_effect('treatment_palbociclib', df['er_positive']==1, 'er_pos')
out['palbo_ER-'] = tx_effect('treatment_palbociclib', df['er_positive']==0, 'er_neg')
out['palbo_ER+postmeno'] = tx_effect('treatment_palbociclib',(df['er_positive']==1)&(df['postmenopausal']==1),'er+_postmeno')
out['palbo_ER+premeno'] = tx_effect('treatment_palbociclib',(df['er_positive']==1)&(df['postmenopausal']==0),'er+_premeno')
out['palbo_ER-postmeno'] = tx_effect('treatment_palbociclib',(df['er_positive']==0)&(df['postmenopausal']==1),'er-_postmeno')

# Palbociclib × PIK3CA mutation
out['palbo_pik3ca+'] = tx_effect('treatment_palbociclib',(df['pik3ca_mutation']==1),'pik3ca_mut')
out['palbo_pik3ca-'] = tx_effect('treatment_palbociclib',(df['pik3ca_mutation']==0),'pik3ca_wt')

# Palbociclib × ER × PIK3CA
for er in [0,1]:
    for pi in [0,1]:
        m = (df['er_positive']==er)&(df['pik3ca_mutation']==pi)
        out[f'palbo_ER{er}_PIK{pi}'] = tx_effect('treatment_palbociclib', m, f'er{er}_pik{pi}')

# Palbociclib × HER2
out['palbo_her2+'] = tx_effect('treatment_palbociclib', df['her2_positive']==1, 'her2_pos')
out['palbo_her2-'] = tx_effect('treatment_palbociclib', df['her2_positive']==0, 'her2_neg')

# Trastuzumab × HER2
out['tras_HER2+'] = tx_effect('treatment_trastuzumab', df['her2_positive']==1, 'her2_pos')
out['tras_HER2-'] = tx_effect('treatment_trastuzumab', df['her2_positive']==0, 'her2_neg')

# Trastuzumab × HER2 × ER
for h in [0,1]:
    for e in [0,1]:
        m = (df['her2_positive']==h)&(df['er_positive']==e)
        out[f'tras_HER2{h}_ER{e}'] = tx_effect('treatment_trastuzumab', m, f'her2{h}_er{e}')

# Olaparib × BRCA1, × BRCA2, × any BRCA
out['ola_brca1+'] = tx_effect('treatment_olaparib', df['brca1_mutation']==1, 'brca1_mut')
out['ola_brca1-'] = tx_effect('treatment_olaparib', df['brca1_mutation']==0, 'brca1_wt')
out['ola_brca2+'] = tx_effect('treatment_olaparib', df['brca2_mutation']==1, 'brca2_mut')
out['ola_brca2-'] = tx_effect('treatment_olaparib', df['brca2_mutation']==0, 'brca2_wt')
out['ola_anyBRCA+'] = tx_effect('treatment_olaparib', (df['brca1_mutation']==1)|(df['brca2_mutation']==1), 'any_brca_mut')
out['ola_anyBRCA-'] = tx_effect('treatment_olaparib', (df['brca1_mutation']==0)&(df['brca2_mutation']==0), 'no_brca')

# Sacituzumab govitecan: typically benefits TNBC and HER2-low. ER-/PR-/HER2-
tnbc = (df['er_positive']==0)&(df['pr_positive']==0)&(df['her2_positive']==0)
out['saci_TNBC'] = tx_effect('treatment_sacituzumab_govitecan', tnbc, 'tnbc')
out['saci_nonTNBC'] = tx_effect('treatment_sacituzumab_govitecan', ~tnbc, 'non_tnbc')
out['saci_her2low'] = tx_effect('treatment_sacituzumab_govitecan', df['her2_low']==1, 'her2_low')
out['saci_her2low_TNBC'] = tx_effect('treatment_sacituzumab_govitecan', tnbc & (df['her2_low']==1), 'tnbc_her2low')
out['saci_her2low_HRpos'] = tx_effect('treatment_sacituzumab_govitecan', (df['her2_low']==1)&(df['er_positive']==1)&(df['her2_positive']==0), 'hr+_her2low')

# Pembrolizumab: usually TNBC + PD-L1 surrogate (high CRP / high LDH / ki67)
out['pembro_TNBC'] = tx_effect('treatment_pembrolizumab', tnbc, 'tnbc')
out['pembro_nonTNBC'] = tx_effect('treatment_pembrolizumab', ~tnbc, 'non_tnbc')
out['pembro_stage4'] = tx_effect('treatment_pembrolizumab', df['stage_iv']==1, 'stage4')
out['pembro_brain_mets'] = tx_effect('treatment_pembrolizumab', df['has_brain_mets']==1, 'brain_mets')

# Pembrolizumab × highKi67 / highCRP / highLDH (surrogates)
out['pembro_highki67'] = tx_effect('treatment_pembrolizumab', df['ki67_pct']>=20, 'ki67>=20')
out['pembro_highCRP'] = tx_effect('treatment_pembrolizumab', df['crp_mg_l']>=10, 'crp>=10')
out['pembro_highLDH'] = tx_effect('treatment_pembrolizumab', df['ldh_u_l']>=300, 'ldh>=300')
out['pembro_highNLR'] = tx_effect('treatment_pembrolizumab', df['nlr']>=4, 'nlr>=4')

# OLS interaction tests - main treatments
fmla_results = {}
for tx in ['treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
          'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab']:
    for bm in ['er_positive','pr_positive','her2_positive','her2_low',
               'brca1_mutation','brca2_mutation','pik3ca_mutation',
               'postmenopausal','stage_iv','has_brain_mets']:
        m = smf.ols(f'pfs_months ~ {tx}*{bm}', data=df).fit()
        intkey = f'{tx}:{bm}'
        if intkey in m.params.index:
            fmla_results[f'{tx}__{bm}'] = {
                'tx_main': float(m.params[tx]),
                'bm_main': float(m.params[bm]),
                'interaction': float(m.params[intkey]),
                'p_interaction': float(m.pvalues[intkey])
            }
out['interactions_OLS'] = fmla_results

with open('out_interactions.json','w') as fp:
    json.dump(out, fp, indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
