"""Final subgroup characterization with multiple modifiers."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
y = df["objective_response"].astype(int)


def logreg(X, y):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.Logit(y, Xc).fit(disp=False, maxiter=200)


# --- The "platinum" subgroup: f_013=0 AND f_015=0 AND f_021=0 AND f_022 <= 22.65 ---
T = df["feature_008"].astype(float)
m13 = (df["feature_013"] == 0).astype(float)
m15 = (df["feature_015"] == 0).astype(float)
m21 = (df["feature_021"] == 0).astype(float)
m22_low = (df["feature_022"] <= 22.65).astype(float)
m27 = (df["feature_027"] == 0).astype(float)

S = (m13 * m15 * m21 * m22_low * m27).astype(float)
print(f"Subgroup S size: {S.sum()} of {len(S)} ({100*S.mean():.1f}%)")

# in-subgroup vs out-subgroup treatment effect
for label, mask in [("inS", S == 1), ("outS", S == 0)]:
    sub = df[mask]
    n_t1 = (sub["feature_008"] == 1).sum()
    n_t0 = (sub["feature_008"] == 0).sum()
    r1 = sub.loc[sub["feature_008"] == 1, "objective_response"].mean()
    r0 = sub.loc[sub["feature_008"] == 0, "objective_response"].mean()
    print(f"  {label}: n_t1={n_t1}, n_t0={n_t0}, rate_t1={r1:.4f}, rate_t0={r0:.4f}, rd={r1-r0:.4f}")

# Interaction test
X = pd.DataFrame({"t": T, "S": S, "tS": T * S})
fit = logreg(X, y)
print(fit.summary())

# Build incremental subgroup hierarchy
print("\n--- Incremental subgroup composition ---")
defs = [
    ("S1: f013=0", df["feature_013"] == 0),
    ("S2: + f015=0", (df["feature_013"] == 0) & (df["feature_015"] == 0)),
    ("S3: + f021=0", (df["feature_013"] == 0) & (df["feature_015"] == 0) & (df["feature_021"] == 0)),
    ("S4: + f027=0", (df["feature_013"] == 0) & (df["feature_015"] == 0) & (df["feature_021"] == 0) & (df["feature_027"] == 0)),
    ("S5: + f022<=22.65", (df["feature_013"] == 0) & (df["feature_015"] == 0) & (df["feature_021"] == 0) & (df["feature_027"] == 0) & (df["feature_022"] <= 22.65)),
]
hier_rows = []
for name, mask in defs:
    sub = df[mask]
    n = mask.sum()
    n_t1 = (sub["feature_008"] == 1).sum()
    n_t0 = (sub["feature_008"] == 0).sum()
    r1 = sub.loc[sub["feature_008"] == 1, "objective_response"].mean()
    r0 = sub.loc[sub["feature_008"] == 0, "objective_response"].mean()
    rd = r1 - r0
    print(f"  {name}: n={n}, n_t1={n_t1}, n_t0={n_t0}, rate_t1={r1:.4f}, rate_t0={r0:.4f}, rd={rd:.4f}")
    hier_rows.append({"def": name, "n": int(n), "n_t1": int(n_t1), "n_t0": int(n_t0),
                      "rate_t1": float(r1), "rate_t0": float(r0), "rd": float(rd)})

# Test t * (full subgroup) interaction adjusted for everything else
non_const = [c for c in df.columns if c.startswith("feature_") and df[c].nunique() > 1]
adj_cols = [c for c in non_const if c not in {"feature_008", "feature_013", "feature_015",
                                                "feature_021", "feature_022", "feature_027"}]
Xa = df[adj_cols].copy().astype(float)
for c in adj_cols:
    if df[c].nunique() > 5:
        Xa[c] = (Xa[c] - Xa[c].mean()) / Xa[c].std()
Xa["t"] = T
Xa["S"] = S
Xa["tS"] = T * S
fitA = logreg(Xa, y)
print(f"\nAdjusted t={fitA.params['t']:.4f} (p={fitA.pvalues['t']:.3e})")
print(f"Adjusted S={fitA.params['S']:.4f} (p={fitA.pvalues['S']:.3e})")
print(f"Adjusted tS={fitA.params['tS']:.4f} (p={fitA.pvalues['tS']:.3e})")

# --- Negative-control check: are continuous features (other than f022) really nulls? ---
print("\n--- Quick continuous interaction screen with t adjusted for top mods ---")
for cont in ["feature_002", "feature_018", "feature_020", "feature_024", "feature_031",
             "feature_026", "feature_009", "feature_029", "feature_003", "feature_012",
             "feature_025", "feature_028", "feature_007", "feature_014", "feature_032",
             "feature_016"]:
    z = (df[cont] - df[cont].mean()) / df[cont].std()
    X = pd.DataFrame({"t": T, "z": z, "tz": T * z, "f013": df["feature_013"].astype(float),
                      "f015": df["feature_015"].astype(float), "f021": df["feature_021"].astype(float)})
    fit = logreg(X, y)
    print(f"  {cont}: tz beta={fit.params['tz']:.4f}, p={fit.pvalues['tz']:.3e}")

# Save final summary
out = {
    "incremental_hierarchy": hier_rows,
    "adj_t": float(fitA.params["t"]),
    "adj_tS": float(fitA.params["tS"]),
    "adj_tS_p": float(fitA.pvalues["tS"]),
    "S_size": int(S.sum()),
}
with open("results2.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved results2.json")
