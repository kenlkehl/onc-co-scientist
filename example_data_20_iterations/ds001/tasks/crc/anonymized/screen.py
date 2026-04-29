"""Univariate screen and structural inspection of the dataset."""
import json
import numpy as np
import pandas as pd
from scipy import stats

DATA = "C:/Users/klkehl/are_llms_biased/data/ds001/tasks/crc/anonymized/dataset.parquet"
df = pd.read_parquet(DATA)

results = {}

# Categorize columns
binary_cols, cont_cols, multi_int, obj_cols = [], [], [], []
for c in df.columns:
    if c in ("pfs_months", "patient_id"):
        continue
    if df[c].dtype == "object":
        obj_cols.append(c)
    elif df[c].dtype == "int64":
        nu = df[c].nunique()
        if nu == 2:
            binary_cols.append(c)
        else:
            multi_int.append(c)
    else:
        cont_cols.append(c)

results["binary_cols"] = binary_cols
results["cont_cols"] = cont_cols
results["multi_int_cols"] = multi_int
results["obj_cols"] = obj_cols

# Binary feature: t-test of pfs_months by group
binary_tests = []
y = df["pfs_months"].values
for c in binary_cols:
    x = df[c].values
    g1 = y[x == 1]
    g0 = y[x == 0]
    if len(g1) < 20 or len(g0) < 20:
        continue
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    binary_tests.append({
        "col": c, "n1": int(len(g1)), "n0": int(len(g0)),
        "mean1": float(g1.mean()), "mean0": float(g0.mean()),
        "diff": float(g1.mean() - g0.mean()),
        "t": float(t), "p": float(p),
    })
binary_tests.sort(key=lambda r: r["p"])
results["binary_tests"] = binary_tests

# Continuous feature: pearson correlation
cont_tests = []
for c in cont_cols:
    x = df[c].values
    r, p = stats.pearsonr(x, y)
    cont_tests.append({
        "col": c, "r": float(r), "p": float(p),
        "mean": float(x.mean()), "std": float(x.std()),
        "min": float(x.min()), "max": float(x.max()),
        "nunique": int(df[c].nunique()),
    })
cont_tests.sort(key=lambda r: r["p"])
results["cont_tests"] = cont_tests

# Multi-int (ordinal): kendall tau and ANOVA
multi_int_tests = []
for c in multi_int:
    x = df[c].values
    # rank correlation
    tau, p_tau = stats.kendalltau(x, y)
    # ANOVA across categories
    groups = [y[x == v] for v in sorted(np.unique(x))]
    F, p_anova = stats.f_oneway(*groups)
    means = {int(v): float(y[x == v].mean()) for v in sorted(np.unique(x))}
    counts = {int(v): int((x == v).sum()) for v in sorted(np.unique(x))}
    multi_int_tests.append({
        "col": c, "tau": float(tau), "p_tau": float(p_tau),
        "F": float(F), "p_anova": float(p_anova),
        "means": means, "counts": counts,
    })
multi_int_tests.sort(key=lambda r: r["p_anova"])
results["multi_int_tests"] = multi_int_tests

# Object (categorical) features: ANOVA
obj_tests = []
for c in obj_cols:
    levels = df[c].unique().tolist()
    groups = [y[df[c] == v] for v in levels]
    F, p_anova = stats.f_oneway(*groups)
    means = {str(v): float(y[df[c] == v].mean()) for v in levels}
    counts = {str(v): int((df[c] == v).sum()) for v in levels}
    obj_tests.append({"col": c, "F": float(F), "p": float(p_anova),
                      "means": means, "counts": counts, "levels": [str(v) for v in levels]})
obj_tests.sort(key=lambda r: r["p"])
results["obj_tests"] = obj_tests

# Save
with open("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/crc/anonymized/screen_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Print top hits
print("=== TOP BINARY ===")
for r in binary_tests[:25]:
    print(f"  {r['col']}: diff={r['diff']:+.3f} (n1={r['n1']}, n0={r['n0']}), p={r['p']:.2e}")
print("=== TOP CONT ===")
for r in cont_tests[:25]:
    print(f"  {r['col']}: r={r['r']:+.4f}, p={r['p']:.2e} (range {r['min']:.2f}..{r['max']:.2f})")
print("=== MULTI-INT ===")
for r in multi_int_tests:
    print(f"  {r['col']}: tau={r['tau']:+.4f}, p_tau={r['p_tau']:.2e}, p_anova={r['p_anova']:.2e}, means={r['means']}")
print("=== OBJECT ===")
for r in obj_tests:
    print(f"  {r['col']}: F={r['F']:.2f}, p={r['p']:.2e}, means={r['means']}")
