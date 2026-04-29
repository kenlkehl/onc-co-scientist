import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2
import warnings, json
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
y = df['objective_response']
N = len(df)
overall = y.mean()
print(f'N={N}, overall ORR={overall:.4f}')

def chi2_test(col):
    ct = pd.crosstab(df[col], y)
    chi2v, p, _, _ = stats.chi2_contingency(ct)
    return ct, chi2v, p

def or_logit_binary(col, covars=None):
    x = df[col].astype(int)
    if covars is None:
        X = sm.add_constant(x)
    else:
        X = pd.concat([x.rename(col), covars], axis=1)
        X = sm.add_constant(X)
    m = sm.Logit(y, X).fit(disp=0)
    return m

def or_logit_cont(col, covars=None):
    x = df[col]
    xz = (x - x.mean())/x.std()
    if covars is None:
        X = sm.add_constant(xz.rename(col))
    else:
        X = pd.concat([xz.rename(col), covars], axis=1)
        X = sm.add_constant(X)
    m = sm.Logit(y, X).fit(disp=0)
    return m

def lr_test(m_full, m_red, df_diff):
    lr = 2*(m_full.llf - m_red.llf)
    return lr, chi2.sf(lr, df_diff)

results = {}

m = or_logit_cont('feature_057')
results['A1'] = {'feature':'feature_057',
    'orr_by_level':{int(k):float(v) for k,v in df.groupby('feature_057')['objective_response'].mean().items()},
    'coef':float(m.params['feature_057']),'p':float(m.pvalues['feature_057']),
    'OR_per_level':float(np.exp(m.params['feature_057']))}

m = or_logit_binary('feature_035')
results['A2'] = {'feature':'feature_035',
    'orr_pos': float(y[df['feature_035']==1].mean()),
    'orr_neg': float(y[df['feature_035']==0].mean()),
    'risk_diff': float(y[df['feature_035']==1].mean() - y[df['feature_035']==0].mean()),
    'coef':float(m.params['feature_035']),'OR':float(np.exp(m.params['feature_035'])),
    'p':float(m.pvalues['feature_035'])}

m = or_logit_cont('feature_011')
results['A3'] = {'feature':'feature_011',
    'mean_resp': float(df.loc[y==1,'feature_011'].mean()),
    'mean_noresp': float(df.loc[y==0,'feature_011'].mean()),
    'coef_per_sd': float(m.params['feature_011']),'p':float(m.pvalues['feature_011'])}

m = or_logit_cont('feature_006')
results['A4'] = {'feature':'feature_006',
    'mean_resp': float(df.loc[y==1,'feature_006'].mean()),
    'mean_noresp': float(df.loc[y==0,'feature_006'].mean()),
    'coef_per_sd': float(m.params['feature_006']),'p':float(m.pvalues['feature_006'])}

m = or_logit_cont('feature_099')
results['A5'] = {'feature':'feature_099',
    'mean_resp': float(df.loc[y==1,'feature_099'].mean()),
    'mean_noresp': float(df.loc[y==0,'feature_099'].mean()),
    'coef_per_sd': float(m.params['feature_099']),'p':float(m.pvalues['feature_099'])}

m = or_logit_cont('feature_063')
results['A6'] = {'feature':'feature_063',
    'mean_resp': float(df.loc[y==1,'feature_063'].mean()),
    'mean_noresp': float(df.loc[y==0,'feature_063'].mean()),
    'coef_per_sd': float(m.params['feature_063']),'p':float(m.pvalues['feature_063'])}

m = or_logit_cont('feature_092')
results['A7'] = {'feature':'feature_092',
    'coef_per_sd': float(m.params['feature_092']),'p':float(m.pvalues['feature_092'])}

m = or_logit_binary('feature_093')
results['A8'] = {'feature':'feature_093',
    'orr_pos': float(y[df['feature_093']==1].mean()),
    'orr_neg': float(y[df['feature_093']==0].mean()),
    'risk_diff': float(y[df['feature_093']==1].mean() - y[df['feature_093']==0].mean()),
    'coef':float(m.params['feature_093']),'p':float(m.pvalues['feature_093'])}

m = or_logit_binary('feature_121')
results['A9'] = {'feature':'feature_121',
    'orr_pos': float(y[df['feature_121']==1].mean()),
    'orr_neg': float(y[df['feature_121']==0].mean()),
    'risk_diff': float(y[df['feature_121']==1].mean() - y[df['feature_121']==0].mean()),
    'coef':float(m.params['feature_121']),'p':float(m.pvalues['feature_121'])}

ct, chi2v, p = chi2_test('feature_005')
results['A10'] = {'feature':'feature_005',
    'orr_by_race': {k:float(v) for k,v in df.groupby('feature_005')['objective_response'].mean().items()},
    'chi2':float(chi2v),'p':float(p)}

ct, chi2v, p = chi2_test('feature_087')
results['A11'] = {'feature':'feature_087',
    'orr_by_ins': {k:float(v) for k,v in df.groupby('feature_087')['objective_response'].mean().items()},
    'chi2':float(chi2v),'p':float(p)}

m = or_logit_binary('feature_014')
results['A12'] = {'feature':'feature_014',
    'orr_pos': float(y[df['feature_014']==1].mean()),
    'orr_neg': float(y[df['feature_014']==0].mean()),
    'risk_diff': float(y[df['feature_014']==1].mean() - y[df['feature_014']==0].mean()),
    'coef':float(m.params['feature_014']),'p':float(m.pvalues['feature_014'])}

m = or_logit_cont('feature_018')
results['A13'] = {'feature':'feature_018',
    'coef':float(m.params['feature_018']),'p':float(m.pvalues['feature_018'])}

m = or_logit_cont('feature_044')
results['A14'] = {'feature':'feature_044',
    'mean_resp': float(df.loc[y==1,'feature_044'].mean()),
    'mean_noresp': float(df.loc[y==0,'feature_044'].mean()),
    'coef_per_sd': float(m.params['feature_044']),'p':float(m.pvalues['feature_044'])}

m = or_logit_cont('feature_078')
results['A15'] = {'feature':'feature_078',
    'mean_resp': float(df.loc[y==1,'feature_078'].mean()),
    'mean_noresp': float(df.loc[y==0,'feature_078'].mean()),
    'coef_per_sd': float(m.params['feature_078']),'p':float(m.pvalues['feature_078'])}

# A16 multivariable
top_cont = ['feature_011','feature_006','feature_099','feature_063','feature_092']
top_cont_z = df[top_cont].apply(lambda x: (x-x.mean())/x.std())
top_bin = ['feature_035','feature_093']
race_d = pd.get_dummies(df['feature_005'], prefix='race', drop_first=True).astype(int)
ins_d = pd.get_dummies(df['feature_087'], prefix='ins', drop_first=True).astype(int)
X = pd.concat([top_cont_z, df[top_bin].astype(int), df['feature_057'].astype(int).rename('feature_057'), race_d, ins_d], axis=1)
X = sm.add_constant(X)
m_adj = sm.Logit(y, X).fit(disp=0)
results['A16'] = {
    'coefs': {k:float(v) for k,v in m_adj.params.items()},
    'pvalues': {k:float(v) for k,v in m_adj.pvalues.items()},
    'OR': {k:float(np.exp(v)) for k,v in m_adj.params.items()},
}

# A17 nonwhite vs white adjusted
df['nonwhite'] = (df['feature_005']!='white').astype(int)
X2 = pd.concat([top_cont_z, df[top_bin].astype(int), df['feature_057'].astype(int).rename('feature_057'), df['nonwhite']], axis=1)
X2 = sm.add_constant(X2)
m_nw = sm.Logit(y, X2).fit(disp=0)
results['A17'] = {
    'nonwhite_coef': float(m_nw.params['nonwhite']),
    'nonwhite_OR': float(np.exp(m_nw.params['nonwhite'])),
    'p': float(m_nw.pvalues['nonwhite'])}

# A18 f035 x f057
f35 = df['feature_035'].astype(int)
f57 = df['feature_057'].astype(int)
Xf = pd.concat([f35.rename('f35'), f57.rename('f57'), (f35*f57).rename('f35_f57')], axis=1)
Xf = sm.add_constant(Xf)
mf = sm.Logit(y, Xf).fit(disp=0)
results['A18'] = {
    'inter_coef': float(mf.params['f35_f57']),
    'p': float(mf.pvalues['f35_f57']),
    'orr_by_subgroup': {f'f57={s}_f35={t}': float(y[(df['feature_057']==s)&(df['feature_035']==t)].mean()) for s in [0,1,2] for t in [0,1]},
    'risk_diff_by_f57': {int(s): float(y[(df['feature_057']==s)&(df['feature_035']==1)].mean() - y[(df['feature_057']==s)&(df['feature_035']==0)].mean()) for s in [0,1,2]},
}

# A19 f035 x race
inter = race_d.multiply(df['feature_035'], axis=0)
inter.columns = [c+'_x_f035' for c in inter.columns]
Xf = pd.concat([f35.rename('f035'), race_d, inter], axis=1)
Xf = sm.add_constant(Xf)
mfu = sm.Logit(y, Xf).fit(disp=0)
Xr = pd.concat([f35.rename('f035'), race_d], axis=1)
Xr = sm.add_constant(Xr)
mfr = sm.Logit(y, Xr).fit(disp=0)
lr, plr = lr_test(mfu, mfr, 4)
results['A19'] = {'lr_chi2': float(lr), 'p_lr': float(plr),
    'rd_by_race':{r: float(y[(df['feature_005']==r)&(df['feature_035']==1)].mean() - y[(df['feature_005']==r)&(df['feature_035']==0)].mean()) for r in df['feature_005'].unique()}}

# A20 f011 x f057
f11z = (df['feature_011']-df['feature_011'].mean())/df['feature_011'].std()
Xf = pd.concat([f11z.rename('f11'), f57.rename('f57'), (f11z*f57).rename('f11_f57')], axis=1)
Xf = sm.add_constant(Xf)
m20 = sm.Logit(y, Xf).fit(disp=0)
results['A20'] = {'inter_coef': float(m20.params['f11_f57']),'p': float(m20.pvalues['f11_f57'])}

# A21 f011 x f121
f121 = df['feature_121'].astype(int)
Xf = pd.concat([f11z.rename('f11'), f121.rename('f121'), (f11z*f121).rename('f11_f121')], axis=1)
Xf = sm.add_constant(Xf)
m21 = sm.Logit(y, Xf).fit(disp=0)
results['A21'] = {'inter_coef': float(m21.params['f11_f121']),'p': float(m21.pvalues['f11_f121'])}

# A22 f099 x f035
f99z = (df['feature_099']-df['feature_099'].mean())/df['feature_099'].std()
Xf = pd.concat([f99z.rename('f99'), f35.rename('f35'), (f99z*f35).rename('f99_f35')], axis=1)
Xf = sm.add_constant(Xf)
m22 = sm.Logit(y, Xf).fit(disp=0)
results['A22'] = {'inter_coef': float(m22.params['f99_f35']),'p': float(m22.pvalues['f99_f35'])}

# A23 f006 x f092
f6z = (df['feature_006']-df['feature_006'].mean())/df['feature_006'].std()
f92z = (df['feature_092']-df['feature_092'].mean())/df['feature_092'].std()
Xf = pd.concat([f6z.rename('f6'), f92z.rename('f92'), (f6z*f92z).rename('f6_f92')], axis=1)
Xf = sm.add_constant(Xf)
m23 = sm.Logit(y, Xf).fit(disp=0)
results['A23'] = {'inter_coef': float(m23.params['f6_f92']),'p': float(m23.pvalues['f6_f92'])}

# A24 race x insurance interaction LR
ins_d2 = pd.get_dummies(df['feature_087'], prefix='ins', drop_first=True).astype(int)
race_d2 = pd.get_dummies(df['feature_005'], prefix='race', drop_first=True).astype(int)
inters = []
for r in race_d2.columns:
    for i in ins_d2.columns:
        inters.append((race_d2[r]*ins_d2[i]).rename(f'{r}_x_{i}'))
inter_df = pd.concat(inters, axis=1)
X_full = pd.concat([race_d2, ins_d2, inter_df], axis=1)
X_full = sm.add_constant(X_full)
m_full = sm.Logit(y, X_full).fit(disp=0)
X_red = pd.concat([race_d2, ins_d2], axis=1)
X_red = sm.add_constant(X_red)
m_red = sm.Logit(y, X_red).fit(disp=0)
lr, plr = lr_test(m_full, m_red, inter_df.shape[1])
results['A24'] = {'lr_chi2': float(lr),'df': int(inter_df.shape[1]),'p_lr': float(plr)}

# A25 final consolidated full model summary
print('Top adjusted ORs:')
for k in ['feature_011','feature_006','feature_099','feature_063','feature_092','feature_035','feature_093','feature_057']:
    if k in m_adj.params:
        print(f'  {k}: OR={np.exp(m_adj.params[k]):.3f}, p={m_adj.pvalues[k]:.4g}')

# Also test feature_007 (the largest binary group), feature_017 etc null findings
for k in ['feature_007','feature_017','feature_002','feature_122','feature_025','feature_070','feature_085']:
    m = or_logit_binary(k)
    results.setdefault('null_binaries', {})[k] = {
        'orr_pos': float(y[df[k]==1].mean()),
        'orr_neg': float(y[df[k]==0].mean()),
        'p': float(m.pvalues[k])
    }

with open('_results_cache.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print('Saved _results_cache.json')
