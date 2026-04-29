"""Subgroup and interaction analyses on key features."""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_parquet('../dataset.parquet')

# Significant main-effect predictors:
core = ['feature_051', 'feature_013', 'feature_011', 'feature_067', 'feature_006',
        'feature_099', 'feature_007', 'feature_063', 'feature_092', 'feature_039',
        'feature_076', 'feature_112', 'feature_033']

# Candidate interaction targets - feature_006 (positive predictor, ~28%) suggests treatment-like;
# feature_092 (continuous 0-0.8) plausible biomarker.
# Test interactions of feature_006 with prognostic vars; and feature_092 with feature_006.
out = []

base = "objective_response ~ feature_051 + feature_013 + feature_011 + feature_067 + feature_006 + feature_099 + feature_007 + feature_063 + feature_092 + feature_039 + feature_076 + feature_112 + feature_033"

# 1. interactions of feature_006 with each other predictor
for other in core:
    if other == 'feature_006':
        continue
    f = base + f" + feature_006:{other}"
    try:
        m = smf.logit(f, data=df).fit(disp=0)
        term = f"feature_006:{other}"
        if term in m.params.index:
            out.append({'pair': term, 'coef': m.params[term], 'p': m.pvalues[term], 'OR': np.exp(m.params[term])})
    except Exception as e:
        pass

# 2. interactions of feature_092 with each other predictor
for other in core:
    if other == 'feature_092':
        continue
    f = base + f" + feature_092:{other}"
    try:
        m = smf.logit(f, data=df).fit(disp=0)
        term = f"feature_092:{other}"
        if term in m.params.index:
            out.append({'pair': term, 'coef': m.params[term], 'p': m.pvalues[term], 'OR': np.exp(m.params[term])})
    except Exception:
        pass

# 3. interactions of feature_013 with each other predictor (could also be a treatment marker - 65% prevalence)
for other in core:
    if other == 'feature_013':
        continue
    f = base + f" + feature_013:{other}"
    try:
        m = smf.logit(f, data=df).fit(disp=0)
        term = f"feature_013:{other}"
        if term in m.params.index:
            out.append({'pair': term, 'coef': m.params[term], 'p': m.pvalues[term], 'OR': np.exp(m.params[term])})
    except Exception:
        pass

# 4. histology x feature_006 (categorical)
m = smf.logit(base + " + C(feature_043) + feature_006:C(feature_043)", data=df).fit(disp=0)
for k in m.params.index:
    if 'feature_006:C(feature_043)' in k or k.startswith('feature_006:'):
        out.append({'pair': 'hist x f006: ' + k, 'coef': m.params[k], 'p': m.pvalues[k], 'OR': np.exp(m.params[k])})

# 5. smoking x feature_006
m = smf.logit(base + " + C(feature_057) + feature_006:C(feature_057)", data=df).fit(disp=0)
for k in m.params.index:
    if 'feature_006:C(feature_057)' in k:
        out.append({'pair': 'smoke x f006: ' + k, 'coef': m.params[k], 'p': m.pvalues[k], 'OR': np.exp(m.params[k])})

res = pd.DataFrame(out).sort_values('p')
res.to_csv('interactions.csv', index=False)
print(res.head(40).to_string())
