import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response']

# 1) Within "good" subgroup, screen all features for further interaction with feature_008
good = (df['feature_013']==0) & (df['feature_015']==0) & (df['feature_021']==0) & (df['feature_027']==0)
g = df.loc[good].copy()
print(f"Good subgroup n={len(g)}")

allf = ['feature_006','feature_005','feature_023','feature_011','feature_017','feature_019','feature_004',
        'feature_016','feature_022','feature_002','feature_018','feature_020','feature_024','feature_031',
        'feature_026','feature_009','feature_029','feature_003','feature_012','feature_025','feature_028',
        'feature_007','feature_014','feature_032','feature_001','feature_010']

print("\nWithin good subgroup, feature_008 × X interactions:")
res = []
for f in allf:
    g2 = g.copy()
    g2['inter'] = g2['feature_008'] * g2[f]
    Xi = sm.add_constant(g2[['feature_008',f,'inter']])
    try:
        mi = sm.Logit(g2['objective_response'], Xi).fit(disp=0)
        res.append((f, mi.params['inter'], mi.pvalues['inter']))
    except Exception as e:
        res.append((f, None, None))
res.sort(key=lambda r: r[2])
for r in res[:15]:
    print(f"  {r[0]}: beta_int={r[1]:.4f}, p={r[2]:.2e}")

# 2) Sensitivity: maybe there are further negative-prog binaries we missed
print("\n\nFurther screen: are there more rare binaries that suppress treatment?")
for f in ['feature_006','feature_005','feature_011','feature_017','feature_019','feature_004','feature_023']:
    s = df.loc[df[f] == 1]
    rr0 = s.loc[s['feature_008']==0,'objective_response'].mean() if (s['feature_008']==0).sum() > 0 else np.nan
    rr1 = s.loc[s['feature_008']==1,'objective_response'].mean() if (s['feature_008']==1).sum() > 0 else np.nan
    print(f"  {f}=1 (n={len(s)}): RR(8=0)={rr0:.3f}, RR(8=1)={rr1:.3f}, diff={rr1-rr0:+.3f}")

# 3) Check whether feature_001 levels modify treatment effect within "good"
print("\nfeature_001 in good subgroup × feature_008:")
print(g.groupby(['feature_001','feature_008'])['objective_response'].agg(['mean','count']))

# 4) Check feature_022 (PSA) interactions only WITHIN bad subgroup — does treatment ever help anyone bad?
print("\nWithin bad subgroup (any of 013/015/021/027 == 1):")
bad = ~good
b = df.loc[bad].copy()
print(f"  bad subgroup n={len(b)}")
# Maybe in bad subgroup, low-PSA still benefits?
for psa_thr in [df['feature_022'].quantile(q) for q in [0.10,0.25]]:
    m = bad & (df['feature_022'] < psa_thr)
    s = df.loc[m]
    rr0 = s.loc[s['feature_008']==0,'objective_response'].mean()
    rr1 = s.loc[s['feature_008']==1,'objective_response'].mean()
    n0 = (s['feature_008']==0).sum()
    n1 = (s['feature_008']==1).sum()
    print(f"  PSA<{psa_thr:.2f}: n={m.sum()}, RR(8=0)={rr0:.3f} (n={n0}), RR(8=1)={rr1:.3f} (n={n1}), diff={rr1-rr0:+.3f}")

# 5) Are the 4 suppressors independently related, or correlated?
print("\nCorrelations among suppressors:")
print(df[['feature_013','feature_015','feature_021','feature_027']].corr())

# 6) Confirm overall AUC of best model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X = df[['feature_008','feature_013','feature_015','feature_021','feature_027','feature_022',
        'feature_001','feature_020','feature_002','feature_024']].copy()
X['i13'] = X['feature_008']*X['feature_013']
X['i15'] = X['feature_008']*X['feature_015']
X['i21'] = X['feature_008']*X['feature_021']
X['i27'] = X['feature_008']*X['feature_027']
lr = LogisticRegression(max_iter=2000)
lr.fit(X, y)
auc = roc_auc_score(y, lr.predict_proba(X)[:,1])
print(f"\nAUC of joint model (with key interactions): {auc:.4f}")

# 7) Quick decision-tree subgroup discovery
from sklearn.tree import DecisionTreeClassifier, export_text
features_for_tree = ['feature_008','feature_013','feature_015','feature_021','feature_027','feature_022',
                     'feature_001','feature_020','feature_002','feature_024','feature_005','feature_023','feature_011']
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=300, random_state=0)
dt.fit(df[features_for_tree], y)
print("\nDecision tree:")
print(export_text(dt, feature_names=features_for_tree, max_depth=5))
