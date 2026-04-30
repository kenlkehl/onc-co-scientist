"""Comprehensive 10-iteration analysis of ds001_nsclc."""
import pandas as pd, numpy as np, json
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
y = df['objective_response'].astype(int)

cols = [c for c in df.columns if c not in ('objective_response', 'patient_id')]
binary_cols, multi_cols, cont_cols = [], [], []
for c in cols:
    if df[c].dtype == 'object':
        multi_cols.append(c)
    else:
        u = df[c].nunique()
        if u == 2:
            binary_cols.append(c)
        elif u <= 5:
            multi_cols.append(c)
        else:
            cont_cols.append(c)

results_log = []

def log(section, payload):
    results_log.append({'section': section, 'payload': payload})
    print(f'[{section}] {payload}')

# ---------- Iteration 1: univariate associations ----------
print('\n=== Iteration 1: univariate ===')
uni = []
for c in binary_cols:
    a = y[df[c] == 1]; b = y[df[c] == 0]
    if len(a) < 10 or len(b) < 10: continue
    diff = a.mean() - b.mean()
    # logistic
    try:
        m = sm.Logit(y, sm.add_constant(df[c].astype(float))).fit(disp=0)
        beta = m.params.iloc[1]; p = m.pvalues.iloc[1]
    except Exception:
        beta, p = np.nan, np.nan
    uni.append((c, 'binary', diff, beta, p, len(a), len(b), a.mean(), b.mean()))

for c in cont_cols:
    a = df.loc[y == 1, c]; b = df.loc[y == 0, c]
    diff = a.mean() - b.mean()
    try:
        x = (df[c] - df[c].mean()) / df[c].std()
        m = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
        beta = m.params.iloc[1]; p = m.pvalues.iloc[1]
    except Exception:
        beta, p = np.nan, np.nan
    uni.append((c, 'cont', diff, beta, p, len(a), len(b), a.mean(), b.mean()))

for c in multi_cols:
    # chi-square
    ct = pd.crosstab(df[c], y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    rates = y.groupby(df[c]).mean().to_dict()
    uni.append((c, 'multi', None, chi2, p, None, None, str(rates), None))

uni_df = pd.DataFrame(uni, columns=['feature','type','diff','effect','p','n_pos','n_neg','mean_pos','mean_neg'])
uni_df = uni_df.sort_values('p')
uni_df.to_csv('uni_my.csv', index=False)
print(uni_df.head(30).to_string())

top_features = uni_df.head(15)['feature'].tolist()
log('iter1_top', top_features)

# ---------- Iteration 2: multivariable model ----------
print('\n=== Iteration 2: multivariable ===')
# build design with top features
X_parts = []
labels = []
for f in top_features:
    if f in binary_cols:
        X_parts.append(df[[f]].astype(float)); labels.append(f)
    elif f in cont_cols:
        z = (df[f] - df[f].mean()) / df[f].std()
        X_parts.append(z.to_frame(f)); labels.append(f)
    else:
        # multi-cat dummies
        d = pd.get_dummies(df[f], prefix=f, drop_first=True).astype(float)
        X_parts.append(d); labels.extend(d.columns)
X = pd.concat(X_parts, axis=1)
X = X.loc[:, ~X.columns.duplicated()]
m_full = sm.Logit(y, sm.add_constant(X.astype(float))).fit(disp=0, maxiter=200)
print(m_full.summary())
mv_results = pd.DataFrame({'coef': m_full.params, 'p': m_full.pvalues, 'or': np.exp(m_full.params)})
mv_results.to_csv('mv_my.csv')
log('iter2_mv', mv_results.to_dict())

# ---------- Iteration 3: subgroup heterogeneity ----------
print('\n=== Iteration 3: subgroups by smoking, sex, race, insurance, histology ===')
sub_results = {}
for sub_var in ['feature_057', 'feature_123', 'feature_005', 'feature_043']:
    sub_results[sub_var] = {}
    for grp, idx in df.groupby(sub_var).groups.items():
        rate = y.loc[idx].mean()
        sub_results[sub_var][grp] = (rate, len(idx))
    print(f'  {sub_var}: {sub_results[sub_var]}')
# stat test of differences
for sub_var in ['feature_057', 'feature_123', 'feature_005', 'feature_043']:
    ct = pd.crosstab(df[sub_var], y)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    print(f'  chi2 {sub_var}: chi2={chi2:.3f}, p={p:.3g}')
    log(f'iter3_{sub_var}', {'chi2': chi2, 'p': p, 'rates': str(sub_results[sub_var])})

# also check key binary demographics that may not be in multi-cat
for f in ['feature_007', 'feature_013', 'feature_067']:
    if f in df.columns:
        a = y[df[f] == 1]; b = y[df[f] == 0]
        diff = a.mean() - b.mean()
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[f], y))
        print(f'  {f}: rate1={a.mean():.4f} (n={len(a)}), rate0={b.mean():.4f} (n={len(b)}), diff={diff:.4f}, p={p:.3g}')
        log(f'iter3_{f}', {'diff': diff, 'p': p, 'rate1': a.mean(), 'rate0': b.mean()})

# ---------- Iteration 4: interactions ----------
print('\n=== Iteration 4: interactions among top features ===')
# Use top continuous + top binary
top_bin = [f for f in top_features if f in binary_cols][:5]
top_cont = [f for f in top_features if f in cont_cols][:3]

interaction_results = []
for f1 in top_bin + top_cont:
    for f2 in top_bin + top_cont:
        if f1 >= f2: continue
        try:
            d = df[[f1, f2]].copy().astype(float)
            d[f1+'_z'] = (d[f1] - d[f1].mean())/d[f1].std()
            d[f2+'_z'] = (d[f2] - d[f2].mean())/d[f2].std()
            d['inter'] = d[f1+'_z'] * d[f2+'_z']
            X = sm.add_constant(d[[f1+'_z', f2+'_z', 'inter']])
            m = sm.Logit(y, X).fit(disp=0, maxiter=100)
            interaction_results.append((f1, f2, m.params['inter'], m.pvalues['inter']))
        except Exception as e:
            pass

# Test interaction with smoking, sex, etc.
key_subs = {'feature_057_never': (df['feature_057']=='never').astype(int),
            'feature_057_current': (df['feature_057']=='current').astype(int),
            'feature_043_squamous': (df['feature_043']=='squamous').astype(int),
            'feature_123_white': (df['feature_123']=='white').astype(int)}
for sub_name, sub_ind in key_subs.items():
    for f in top_bin + top_cont:
        try:
            v = df[f].astype(float)
            if f in cont_cols:
                v = (v - v.mean())/v.std()
            X = pd.DataFrame({'a': v, 'b': sub_ind, 'ab': v * sub_ind})
            m = sm.Logit(y, sm.add_constant(X)).fit(disp=0, maxiter=100)
            interaction_results.append((f, sub_name, m.params['ab'], m.pvalues['ab']))
        except Exception:
            pass

ir_df = pd.DataFrame(interaction_results, columns=['f1','f2','beta_inter','p']).sort_values('p')
ir_df.to_csv('inter_my.csv', index=False)
print(ir_df.head(20).to_string())
log('iter4_top_interactions', ir_df.head(10).to_dict('records'))

# ---------- Iteration 5: continuous biomarker dose-response ----------
print('\n=== Iteration 5: continuous dose-response (top continuous features) ===')
top_cont_all = uni_df[uni_df['type']=='cont'].head(8)['feature'].tolist()
dose_results = []
for f in top_cont_all:
    qs = pd.qcut(df[f], 4, labels=False, duplicates='drop')
    rates = y.groupby(qs).mean()
    print(f'  {f} quartile rates: {rates.tolist()}')
    # test trend
    z = (df[f] - df[f].mean())/df[f].std()
    m = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
    dose_results.append({'feature': f, 'q1': rates.iloc[0], 'q4': rates.iloc[-1], 'beta': m.params.iloc[1], 'p': m.pvalues.iloc[1]})
log('iter5_dose', dose_results)

# ---------- Iteration 6: ordinal effects (stage-like, ECOG-like, PD-L1-like) ----------
print('\n=== Iteration 6: ordinal multi-cat ===')
ordinal_cols = ['feature_051', 'feature_026', 'feature_033', 'feature_018', 'feature_045', 'feature_042']
for f in ordinal_cols:
    # treat as numeric
    v = df[f].astype(int)
    rates = y.groupby(v).mean().to_dict()
    z = (v - v.mean())/v.std()
    m = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
    print(f'  {f}: rates={rates}, beta={m.params.iloc[1]:.4f}, p={m.pvalues.iloc[1]:.3g}')
    log(f'iter6_{f}', {'rates': str(rates), 'beta': float(m.params.iloc[1]), 'p': float(m.pvalues.iloc[1])})

# ---------- Iteration 7: smoking x PD-L1 interaction (immunotherapy hypothesis) ----------
print('\n=== Iteration 7: smoking x PD-L1, histology x PD-L1, etc. ===')
df['pdl1'] = df['feature_051'].astype(int)
df['ever_smoker'] = (df['feature_057'] != 'never').astype(int)
df['squamous'] = (df['feature_043'] == 'squamous').astype(int)
df['female'] = df['feature_013']  # guess
# check pdl1 x ever_smoker
m = smf.logit('objective_response ~ pdl1 * ever_smoker', df).fit(disp=0)
print(m.summary())
log('iter7_pdl1_smoke', {'beta_inter': float(m.params['pdl1:ever_smoker']), 'p': float(m.pvalues['pdl1:ever_smoker'])})
m2 = smf.logit('objective_response ~ pdl1 * squamous', df).fit(disp=0)
log('iter7_pdl1_squamous', {'beta_inter': float(m2.params['pdl1:squamous']), 'p': float(m2.pvalues['pdl1:squamous'])})
# check race x smoking
m3 = smf.logit('objective_response ~ C(feature_123) * ever_smoker', df).fit(disp=0)
print('Race x smoking:')
for k, v in m3.pvalues.items():
    if 'ever_smoker' in k and ':' in k:
        print(f'  {k}: beta={m3.params[k]:.3f}, p={v:.3g}')

# ---------- Iteration 8: lab marker patterns / composite ----------
print('\n=== Iteration 8: composite biomarker score ===')
# build composite from top continuous predictors
top_cont_features = uni_df[uni_df['type']=='cont'].head(5)['feature'].tolist()
print('Top cont:', top_cont_features)
# Z-score, sign-correct each
score = np.zeros(len(df))
weights = {}
for f in top_cont_features:
    z = (df[f] - df[f].mean())/df[f].std()
    m = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
    sign = np.sign(m.params.iloc[1])
    score += sign * z
    weights[f] = float(sign)
df['composite_score'] = score
qs = pd.qcut(score, 4, labels=False)
rates = y.groupby(qs).mean()
print(f'  composite quartile rates: {rates.tolist()}')
m_comp = sm.Logit(y, sm.add_constant((score-score.mean())/score.std())).fit(disp=0)
print(f'  composite beta={m_comp.params.iloc[1]:.4f}, p={m_comp.pvalues.iloc[1]:.3g}')
log('iter8_composite', {'weights': weights, 'q1': float(rates.iloc[0]), 'q4': float(rates.iloc[-1]),
                       'beta': float(m_comp.params.iloc[1]), 'p': float(m_comp.pvalues.iloc[1])})

# ---------- Iteration 9: refined multivariable with key signals + selected interactions ----------
print('\n=== Iteration 9: refined model with selected interactions ===')
# use top features and add key interactions
df['pdl1_x_smoker'] = df['pdl1'] * df['ever_smoker']
formula = "objective_response ~ pdl1 + ever_smoker + squamous + C(feature_123) + C(feature_005) + feature_011"
m9 = smf.logit(formula, df).fit(disp=0, maxiter=200)
print(m9.summary())

# include feature_007, feature_013, feature_067
formula2 = "objective_response ~ pdl1 + ever_smoker + feature_011 + feature_013 + feature_067 + feature_007 + feature_006 + feature_039 + feature_092 + feature_099"
m92 = smf.logit(formula2, df).fit(disp=0, maxiter=200)
print(m92.summary())
log('iter9_mv2', {k: (float(v), float(m92.pvalues[k])) for k, v in m92.params.items()})

# ---------- Iteration 10: race / insurance disparities adjusted ----------
print('\n=== Iteration 10: disparities adjusted for clinical features ===')
# Adjust race and insurance for top clinical features
formula = "objective_response ~ C(feature_123, Treatment(reference='white')) + pdl1 + ever_smoker + feature_011 + feature_013 + feature_067 + feature_006"
m10 = smf.logit(formula, df).fit(disp=0, maxiter=200)
print(m10.summary())
log('iter10_race_adj', {k: (float(v), float(m10.pvalues[k])) for k, v in m10.params.items()})

formula_ins = "objective_response ~ C(feature_005, Treatment(reference='private')) + pdl1 + ever_smoker + feature_011 + feature_013 + feature_067 + feature_006"
m10b = smf.logit(formula_ins, df).fit(disp=0, maxiter=200)
print(m10b.summary())
log('iter10_ins_adj', {k: (float(v), float(m10b.pvalues[k])) for k, v in m10b.params.items()})

# Save log
with open('analysis_log.json', 'w') as f:
    json.dump(results_log, f, indent=2, default=str)
print('\nDone.')
