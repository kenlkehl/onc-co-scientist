"""Deep dive into regorafenib subgroup definition."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
df["log_cea"] = np.log1p(df.cea_ng_ml)

with open("results.json", "r") as f:
    results = json.load(f)

def record(key, summary, effect=None, p=None, sig=None):
    eff_v = float(effect) if effect is not None and np.isfinite(effect) else None
    p_v = float(p) if p is not None and np.isfinite(p) else None
    if sig is None:
        sig_v = bool(p_v is not None and p_v < 0.05)
    else:
        sig_v = bool(sig)
    results[key] = {
        "result_summary": summary,
        "effect_estimate": eff_v,
        "p_value": p_v,
        "significant": sig_v,
    }
    print(f"[{key}] eff={eff_v} p={p_v} sig={sig_v}")

def tt(sub, tx="treatment_regorafenib"):
    on = sub.loc[sub[tx] == 1, "pfs_months"]
    off = sub.loc[sub[tx] == 0, "pfs_months"]
    if len(on) > 1 and len(off) > 1:
        t, p = stats.ttest_ind(on, off)
        return on.mean() - off.mean(), p, len(on), len(off)
    else:
        return float(np.nan), float(np.nan), len(on), len(off)

# 1) Stratified across KRAS, NRAS, BRAF, sidedness, MSI - test rego in each
print("\n=== Rego stratified by combinations of binary biomarkers ===")
# Two-way: KRAS x BRAF
for k in [0, 1]:
    for b in [0, 1]:
        sub = df[(df.kras_mutation == k) & (df.braf_v600e == b)]
        eff, p, n1, n0 = tt(sub)
        record(f"rego_kras{k}_braf{b}", f"Rego in KRAS={k}, BRAF={b} (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
               effect=eff, p=p)

# KRAS x sidedness
for k in [0, 1]:
    for r in [0, 1]:
        sub = df[(df.kras_mutation == k) & (df.right_sided_primary == r)]
        eff, p, n1, n0 = tt(sub)
        record(f"rego_kras{k}_side{r}", f"Rego in KRAS={k}, right_sided={r} (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
               effect=eff, p=p)

# Triple: KRAS x BRAF x sidedness
for k in [0, 1]:
    for b in [0, 1]:
        for r in [0, 1]:
            sub = df[(df.kras_mutation == k) & (df.braf_v600e == b) & (df.right_sided_primary == r)]
            eff, p, n1, n0 = tt(sub)
            record(f"rego_kras{k}_braf{b}_side{r}", f"Rego in KRAS={k}, BRAF={b}, right={r} (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
                   effect=eff, p=p)

# Add NRAS too (full RAS+RAF WT, left-sided)
sub = df[(df.kras_mutation == 0) & (df.nras_mutation == 0) & (df.braf_v600e == 0) & (df.right_sided_primary == 0)]
eff, p, n1, n0 = tt(sub)
record("rego_full_wt_left", f"Rego in KRAS+NRAS+BRAF WT, left-sided (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
       effect=eff, p=p)

# 2) CEA modifier - effect by CEA quartile
print("\n=== Rego effect by CEA quartile ===")
df["cea_q"] = pd.qcut(df.cea_ng_ml, 4, labels=False)
for q in range(4):
    sub = df[df.cea_q == q]
    eff, p, n1, n0 = tt(sub)
    record(f"rego_cea_q{q}", f"Rego in CEA quartile {q} (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
           effect=eff, p=p)

# 3) Combine CEA threshold with KRAS-WT, left, BRAF WT
# Try CEA < median in KRAS-WT, BRAF-WT, left-sided
cea_med = df.cea_ng_ml.median()
print(f"CEA median = {cea_med:.2f}")
mask = (df.kras_mutation == 0) & (df.braf_v600e == 0) & (df.right_sided_primary == 0) & (df.cea_ng_ml < cea_med)
sub = df[mask]
eff, p, n1, n0 = tt(sub)
record("rego_kras0_braf0_left_loCEA", f"Rego in KRAS WT, BRAF WT, left-sided, CEA<{cea_med:.2f} (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
       effect=eff, p=p)

mask = (df.kras_mutation == 0) & (df.braf_v600e == 0) & (df.right_sided_primary == 0) & (df.cea_ng_ml >= cea_med)
sub = df[mask]
eff, p, n1, n0 = tt(sub)
record("rego_kras0_braf0_left_hiCEA", f"Rego in KRAS WT, BRAF WT, left-sided, CEA>={cea_med:.2f} (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
       effect=eff, p=p)

# 4) Continuous CEA: split CEA at meaningful thresholds (5 ng/mL is upper limit of normal)
mask = (df.kras_mutation == 0) & (df.braf_v600e == 0) & (df.right_sided_primary == 0) & (df.cea_ng_ml < 5)
sub = df[mask]
eff, p, n1, n0 = tt(sub)
record("rego_kras0_braf0_left_cea_lt5", f"Rego in KRAS WT, BRAF WT, left-sided, CEA<5 (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
       effect=eff, p=p)

mask = (df.kras_mutation == 0) & (df.braf_v600e == 0) & (df.right_sided_primary == 0) & (df.cea_ng_ml >= 5)
sub = df[mask]
eff, p, n1, n0 = tt(sub)
record("rego_kras0_braf0_left_cea_ge5", f"Rego in KRAS WT, BRAF WT, left-sided, CEA>=5 (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
       effect=eff, p=p)

# 5) Final overall best subgroup: KRAS WT
sub = df[df.kras_mutation == 0]
eff, p, n1, n0 = tt(sub)
record("rego_best_kras0", f"Rego in KRAS WT alone (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
       effect=eff, p=p)

# 6) Test in nras-mut and braf-mut subgroups separately
sub = df[df.braf_v600e == 1]
eff, p, n1, n0 = tt(sub)
record("rego_braf1", f"Rego in BRAF V600E (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
       effect=eff, p=p)

sub = df[df.nras_mutation == 1]
eff, p, n1, n0 = tt(sub)
record("rego_nras1", f"Rego in NRAS mutant (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
       effect=eff, p=p)

sub = df[df.right_sided_primary == 1]
eff, p, n1, n0 = tt(sub)
record("rego_right", f"Rego in right-sided (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
       effect=eff, p=p)

# 7) Multivariable: rego + key three-way interactions
m = smf.ols("pfs_months ~ treatment_regorafenib * (kras_mutation + nras_mutation + braf_v600e + right_sided_primary + log_cea)", data=df).fit()
for key_b in ["treatment_regorafenib:kras_mutation", "treatment_regorafenib:nras_mutation",
              "treatment_regorafenib:braf_v600e", "treatment_regorafenib:right_sided_primary",
              "treatment_regorafenib:log_cea"]:
    record(f"rego_multi_{key_b.replace(':', '_x_')}",
           f"Multi-interaction model {key_b}: beta={m.params[key_b]:+.3f}, p={m.pvalues[key_b]:.2e}",
           effect=m.params[key_b], p=m.pvalues[key_b])
record("rego_multi_main",
       f"Multi-interaction model regorafenib main effect: beta={m.params['treatment_regorafenib']:+.3f}, p={m.pvalues['treatment_regorafenib']:.2e}",
       effect=m.params["treatment_regorafenib"], p=m.pvalues["treatment_regorafenib"])

# 8) Final winning combined subgroup: KRAS WT, NRAS WT, BRAF WT, left-sided, low CEA
mask_ALL = (df.kras_mutation == 0) & (df.nras_mutation == 0) & (df.braf_v600e == 0) & (df.right_sided_primary == 0) & (df.cea_ng_ml < 5)
sub = df[mask_ALL]
eff, p, n1, n0 = tt(sub)
record("rego_FINAL_subgroup", f"Rego in KRAS WT + NRAS WT + BRAF WT + left-sided + CEA<5 (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
       effect=eff, p=p)

# Check complement (anyone who fails one or more conditions)
sub = df[~mask_ALL]
eff, p, n1, n0 = tt(sub)
record("rego_FINAL_complement", f"Rego in COMPLEMENT (any unfavorable feature) (n_on={n1},n_off={n0}): {eff:+.3f} months, p={p:.2e}",
       effect=eff, p=p)

# Save
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nTotal results: {len(results)}")
