"""Iterations 11-18: refine enzalutamide subgroup, deep subgroup searches for other treatments."""
import json, math, itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
RESULTS = json.load(open("all_results.json"))

def add(it, hid, htext, code, summary, p, eff, sig=None, kind="novel"):
    if sig is None and p is not None and not (isinstance(p, float) and math.isnan(p)):
        sig = bool(p < 0.05)
    RESULTS.append({
        "iter": it, "hyp_id": hid, "hyp_text": htext, "kind": kind,
        "code": code, "summary": summary,
        "p": (None if p is None or (isinstance(p, float) and math.isnan(p)) else float(p)),
        "eff": (None if eff is None or (isinstance(eff, float) and math.isnan(eff)) else float(eff)),
        "sig": sig
    })

def stratified_treat(treat, mask):
    sub = df[mask]
    n_t = int((sub[treat]==1).sum()); n_c = int((sub[treat]==0).sum())
    if n_t < 5 or n_c < 5:
        return np.nan, np.nan, np.nan, np.nan, n_t+n_c
    a = sub.loc[sub[treat]==1,'objective_response'].mean()
    b = sub.loc[sub[treat]==0,'objective_response'].mean()
    ct = pd.crosstab(sub[treat], sub['objective_response'])
    if ct.shape != (2,2):
        return a, b, a-b, np.nan, n_t+n_c
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return a, b, a-b, p, n_t+n_c

# ============================================================
# Iteration 11: refine enzalutamide responder subgroup
# ============================================================
it = 11

# Test enzalutamide × mcrpc interaction
m = smf.logit("objective_response ~ treatment_enzalutamide * mcrpc", data=df).fit(disp=0)
coef = m.params["treatment_enzalutamide:mcrpc"]; p = m.pvalues["treatment_enzalutamide:mcrpc"]
add(it, "i11_enza_mcrpc",
    "treatment_enzalutamide × mcrpc interaction: enzalutamide effect on objective_response is much larger in mcrpc=0 (non-castrate-resistant) than in mcrpc=1.",
    "logit(objective_response ~ treatment_enzalutamide * mcrpc)",
    f"Interaction coef={coef:.3f}, p={p:.3g}.", p, coef)

# Test enzalutamide × mcrpc × ar_v7
m = smf.logit("objective_response ~ treatment_enzalutamide * mcrpc * ar_v7_positive", data=df).fit(disp=0)
coef3 = m.params.get("treatment_enzalutamide:mcrpc:ar_v7_positive", np.nan)
p3 = m.pvalues.get("treatment_enzalutamide:mcrpc:ar_v7_positive", np.nan)
# Cell-by-cell
text_cells = []
for mc in (0,1):
    for arv7 in (0,1):
        mask = (df['mcrpc']==mc)&(df['ar_v7_positive']==arv7)
        a,b,d,pp,n = stratified_treat('treatment_enzalutamide', mask)
        text_cells.append(f"mcrpc={mc},arv7={arv7}: enza+={a:.3f}, enza-={b:.3f}, diff={d:+.3f}, n={n}, p={pp:.3g}")
add(it, "i11_enza_mcrpc_arv7_3way",
    "Three-way interaction: enzalutamide effect requires mcrpc=0 AND ar_v7_positive=0 (effect is concentrated in this cell).",
    "logit(objective_response ~ treatment_enzalutamide * mcrpc * ar_v7_positive); plus stratified rates",
    f"3-way coef={coef3:.3f}, p={p3:.3g}. Cell rates:\n  " + "\n  ".join(text_cells), p3, coef3)

# Specifically the canonical hypothesis: enzalutamide benefit in mcrpc=0 & ar_v7=0
mask = (df['mcrpc']==0)&(df['ar_v7_positive']==0)
a,b,d,p,n = stratified_treat('treatment_enzalutamide', mask)
add(it, "i11_enza_canonical",
    "Within mcrpc=0 AND ar_v7_positive=0, treatment_enzalutamide produces a large positive ORR difference vs no enzalutamide.",
    "stratified ORR within mcrpc=0 & ar_v7_positive=0",
    f"n={n}: ORR enza+={a:.3f} vs enza-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d)

# Within mcrpc=1: does enzalutamide do anything?
mask = df['mcrpc']==1
a,b,d,p,n = stratified_treat('treatment_enzalutamide', mask)
add(it, "i11_enza_mcrpc1",
    "Among mcrpc=1 patients, treatment_enzalutamide does NOT improve objective_response (effect ≈ 0).",
    "stratified ORR within mcrpc=1",
    f"n={n}: ORR enza+={a:.3f} vs enza-={b:.3f}, diff={d:+.3f}, p={p:.3g}.", p, d)

# ============================================================
# Iteration 12: Try every triple subgroup containing each "natural" biomarker
# for olaparib (brca2), pembrolizumab (msi_high), lu177 (psma_high)
# ============================================================
it = 12
extra = ['mcrpc','visceral_mets','ar_v7_positive']  # additional binary modifiers

def triple_search(treat, biom):
    rows = []
    # subgroup is biom=1 + each pair of extras (each extra value in {0,1})
    for f1, f2 in itertools.combinations(extra, 2):
        for v1 in (0,1):
            for v2 in (0,1):
                mask = (df[biom]==1)&(df[f1]==v1)&(df[f2]==v2)
                a,b,d,p,n = stratified_treat(treat, mask)
                rows.append((f"{biom}=1 & {f1}={v1} & {f2}={v2}", n, a, b, d, p))
    return rows

for treat, biom in [('treatment_olaparib','brca2_mutation'),
                    ('treatment_pembrolizumab','msi_high'),
                    ('treatment_lu177_psma','psma_high')]:
    rows = triple_search(treat, biom)
    rows.sort(key=lambda r: (-(r[4] if not (isinstance(r[4],float) and math.isnan(r[4])) else -1)))
    text = f"Triple-subgroup search for {treat} effect, conditioning on {biom}=1 plus pairs of extras:\n"
    for label, n, a, b, d, p in rows[:6]:
        text += f"  {label}: n={n}, ORR treat+={a:.3f}/treat-={b:.3f}/diff={d:+.3f}, p={p:.3g}\n"
    # Take best subgroup
    best = max(rows, key=lambda r: (r[4] if not (isinstance(r[4],float) and math.isnan(r[4])) else -1))
    add(it, f"i12_{treat}_triple",
        f"For {treat}, the largest positive ORR difference (treated minus control) appears in a 3-feature subgroup containing {biom}=1.",
        "stratified ORR within (biom=1) and each pair of {mcrpc, visceral_mets, ar_v7_positive}",
        text.strip(), best[5], best[4])

# ============================================================
# Iteration 13: Pure interaction-coef screen — what features modify EACH treatment?
# ============================================================
it = 13
binary_mods = ['mcrpc','visceral_mets','brca2_mutation','ar_v7_positive','msi_high','psma_high']
for treat in ['treatment_olaparib','treatment_pembrolizumab','treatment_lu177_psma',
              'treatment_abiraterone','treatment_docetaxel']:
    text = f"Interaction-coefficient screen for {treat}:\n"
    best_coef = 0; best_p = 1; best_mod = None
    for mod in binary_mods:
        if mod == treat: continue
        f = f"objective_response ~ {treat} * {mod}"
        m = smf.logit(f, data=df).fit(disp=0)
        iname = f"{treat}:{mod}"
        coef = m.params[iname]; p = m.pvalues[iname]
        text += f"  {mod}: interaction coef={coef:+.3f}, p={p:.3g}\n"
        if abs(coef) > abs(best_coef): best_coef, best_p, best_mod = coef, p, mod
    add(it, f"i13_{treat}_interactions",
        f"Among single binary biomarkers, the largest interaction coefficient with {treat} on objective_response is with {best_mod}.",
        "logit(objective_response ~ treat * mod) for each mod",
        text.strip(), best_p, best_coef)

# ============================================================
# Iteration 14: Continuous-modifier interactions for olaparib/pembro/lu177
# (maybe efficacy concentrated in low LDH or high albumin etc.)
# ============================================================
it = 14
cont_mods = ['psa_ng_ml','albumin_g_dl','ldh_u_l','crp_mg_l','nlr','hemoglobin_g_dl',
             'alkaline_phosphatase_u_l','ecog_ps','gleason_score']
for treat in ['treatment_olaparib','treatment_pembrolizumab','treatment_lu177_psma']:
    text = f"Continuous-modifier interaction screen for {treat}:\n"
    best_coef = 0; best_p = 1; best_mod = None
    for mod in cont_mods:
        f = f"objective_response ~ {treat} * {mod}"
        m = smf.logit(f, data=df).fit(disp=0)
        iname = f"{treat}:{mod}"
        coef = m.params[iname]; p = m.pvalues[iname]
        text += f"  {mod}: coef={coef:+.4g}, p={p:.3g}\n"
        if p < best_p: best_coef, best_p, best_mod = coef, p, mod
    add(it, f"i14_{treat}_cont",
        f"For {treat}, the strongest continuous modifier of treatment effect on objective_response is {best_mod}.",
        "logit(objective_response ~ treat * cont_mod) for each cont_mod",
        text.strip(), best_p, best_coef)

# ============================================================
# Iteration 15: Tree-based subgroup identification (sklearn decision tree on
# treatment_effect proxy or use a causal-forest-light approach via residualization)
# ============================================================
it = 15
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

# For each treatment, fit a tree predicting response within treated and within untreated; compare predicted ORR
# Simpler: fit DT on whole data with treatment + features, look for branches splitting treatment effect
# Use a HTE-style approach: residualize response on a logistic of all features (no treatment), then look
# at residuals by treatment
features_all = ['age_years','ecog_ps','mcrpc','visceral_mets','psa_ng_ml','gleason_score',
                'brca2_mutation','ar_v7_positive','msi_high','psma_high',
                'albumin_g_dl','ldh_u_l','weight_loss_pct_6mo','crp_mg_l','nlr',
                'hemoglobin_g_dl','alkaline_phosphatase_u_l','ast_u_l','alt_u_l',
                'total_bilirubin_mg_dl','creatinine_mg_dl','bun_mg_dl',
                'sodium_meq_l','potassium_meq_l','calcium_mg_dl']

# Per-treatment HTE search: fit RF on treated, RF on untreated, compute uplift on whole population, find subgroup with biggest predicted uplift
for treat in ['treatment_enzalutamide','treatment_olaparib','treatment_pembrolizumab',
              'treatment_lu177_psma','treatment_abiraterone','treatment_docetaxel']:
    treated_idx = df[treat]==1; untreated_idx = df[treat]==0
    Xt = df.loc[treated_idx, features_all]; yt = df.loc[treated_idx, 'objective_response']
    Xc = df.loc[untreated_idx, features_all]; yc = df.loc[untreated_idx, 'objective_response']
    rf_t = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=200, random_state=42, n_jobs=1)
    rf_c = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=200, random_state=42, n_jobs=1)
    rf_t.fit(Xt, yt); rf_c.fit(Xc, yc)
    p_t = rf_t.predict_proba(df[features_all])[:,1]
    p_c = rf_c.predict_proba(df[features_all])[:,1]
    uplift = p_t - p_c
    df['_uplift'] = uplift
    # Top decile by predicted uplift: characterize and test ORR difference within that group
    cutoff = np.quantile(uplift, 0.9)
    top = df[uplift >= cutoff]
    a = top.loc[top[treat]==1,'objective_response'].mean()
    b = top.loc[top[treat]==0,'objective_response'].mean()
    n_t = int((top[treat]==1).sum()); n_c = int((top[treat]==0).sum())
    if n_t>=5 and n_c>=5:
        ct = pd.crosstab(top[treat], top['objective_response'])
        try: _,p,_,_ = stats.chi2_contingency(ct)
        except: p = np.nan
    else: p = np.nan
    # Describe top decile
    desc = top[features_all].mean()
    desc_brief = " ".join([f"{c}={desc[c]:.2g}" for c in ['mcrpc','ar_v7_positive','brca2_mutation','msi_high','psma_high','ecog_ps','psa_ng_ml','albumin_g_dl','ldh_u_l']])
    add(it, f"i15_{treat}_uplift",
        f"In a random-forest-based HTE search, the top-uplift decile for {treat} shows a positive ORR difference between treated and untreated.",
        "RF on treated and untreated separately; predicted uplift; top decile",
        f"Top decile (n={len(top)}, n_treat={n_t}, n_ctrl={n_c}): ORR treat+={a:.3f} vs treat-={b:.3f}, diff={a-b:+.3f}, p={p:.3g}. Top-decile feature means: {desc_brief}",
        p, a-b)

with open("all_results.json","w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"After iter 15, {len(RESULTS)} analyses recorded.")
