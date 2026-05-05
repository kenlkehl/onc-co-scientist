"""Iteration 3: Treatment-candidate balance and pairwise heterogeneity.
- Is feature_008 balanced w.r.t. other features (treatment-like)?
- Test feature_008 x other binary features for ORR interactions.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
df['log_f022'] = np.log1p(df['feature_022'])
df['log_f020'] = np.log1p(df['feature_020'])
df['log_f024'] = np.log1p(df['feature_024'])

T = 'feature_008'  # candidate treatment
print(f'=== Balance check: {T} prevalence and feature distributions by {T} ===')
print(f'{T} prevalence: {(df[T]==1).mean():.4f} ({(df[T]==1).sum()}/{len(df)})')
print()

binary_other = [c for c in df.columns if c.startswith('feature_') and df[c].nunique()==2 and c != T]
multi_other = ['feature_001', 'feature_010']
cont_other = [c for c in df.columns if c.startswith('feature_') and df[c].nunique() > 10]

print(f'{"feature":<14}{"prev|T=0":>12}{"prev|T=1":>12}{"chi2 p":>12}')
for c in binary_other:
    p0 = df.loc[df[T]==0, c].mean()
    p1 = df.loc[df[T]==1, c].mean()
    ct = pd.crosstab(df[T], df[c])
    _, p, _, _ = stats.chi2_contingency(ct)
    print(f'{c:<14}{p0:>12.4f}{p1:>12.4f}{p:>12.3e}')
print()

for c in multi_other:
    print(f'{c} distribution by {T}:')
    print(pd.crosstab(df[T], df[c], normalize='index').round(4))
    ct = pd.crosstab(df[T], df[c])
    _, p, _, _ = stats.chi2_contingency(ct)
    print(f'  chi2 p = {p:.3e}')
    print()

print(f'{"feature":<14}{"mean|T=0":>14}{"mean|T=1":>14}{"diff":>14}{"p":>12}')
for c in cont_other:
    g0 = df.loc[df[T]==0, c].values
    g1 = df.loc[df[T]==1, c].values
    t, p = stats.ttest_ind(g0, g1, equal_var=False)
    print(f'{c:<14}{g0.mean():>14.4f}{g1.mean():>14.4f}{g1.mean()-g0.mean():>+14.4f}{p:>12.3e}')
print()

# ===== Treatment x feature interactions on ORR =====
print(f'=== {T} interaction screen against every other feature on ORR (LRT p) ===')
print(f'{"modifier":<14}{"main_OR_T@mod=0":>18}{"main_OR_T@mod=1or+1":>22}{"interaction p":>16}')
y = df['objective_response'].astype(int)
results = []
for c in binary_other + multi_other:
    Xb = df[[T, c]].copy()
    Xb['Tx'] = Xb[T] * Xb[c]
    Xb_full = sm.add_constant(Xb.astype(float))
    Xb_red = sm.add_constant(df[[T, c]].astype(float))
    try:
        m_full = sm.Logit(y, Xb_full).fit(disp=False)
        m_red = sm.Logit(y, Xb_red).fit(disp=False)
        lrt = 2 * (m_full.llf - m_red.llf)
        p_int = stats.chi2.sf(lrt, 1)
        # OR of T at modifier=0 and modifier=1 (or modifier=mean+1 for multi)
        or_t_at_0 = np.exp(m_full.params[T])
        or_t_at_1 = np.exp(m_full.params[T] + m_full.params['Tx'])
        results.append((c, or_t_at_0, or_t_at_1, p_int))
        print(f'{c:<14}{or_t_at_0:>18.3f}{or_t_at_1:>22.3f}{p_int:>16.3e}')
    except Exception as e:
        print(f'{c:<14}  FAILED: {e}')

# Continuous modifiers, scaled
print()
for c in cont_other:
    s = (df[c] - df[c].mean()) / df[c].std()
    Xb = pd.DataFrame({'T': df[T], 'mod': s, 'Tx': df[T]*s})
    Xb_full = sm.add_constant(Xb.astype(float))
    Xb_red = sm.add_constant(Xb[['T','mod']].astype(float))
    try:
        m_full = sm.Logit(y, Xb_full).fit(disp=False)
        m_red = sm.Logit(y, Xb_red).fit(disp=False)
        lrt = 2 * (m_full.llf - m_red.llf)
        p_int = stats.chi2.sf(lrt, 1)
        # OR of T at mod = -1 SD vs +1 SD
        or_t_at_low = np.exp(m_full.params['T'] - m_full.params['Tx'])
        or_t_at_high = np.exp(m_full.params['T'] + m_full.params['Tx'])
        print(f'{c:<14}OR(T)@-1SD={or_t_at_low:.3f}  OR(T)@+1SD={or_t_at_high:.3f}  int_p={p_int:.3e}')
    except Exception as e:
        print(f'{c:<14}FAILED: {e}')
