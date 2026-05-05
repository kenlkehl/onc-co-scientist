"""Adds final results that I want in the transcript: feature_008 treatment-effect heterogeneity."""
import json
import pandas as pd
from scipy import stats

df = pd.read_parquet("../dataset.parquet")

with open("results_full.json") as f:
    R = json.load(f)


def eff(mask, label, treat="feature_008"):
    s = df[mask]
    if s[treat].nunique() < 2:
        return None
    t = pd.crosstab(s[treat], s["objective_response"])
    if t.shape != (2, 2):
        return None
    chi2, p, _, _ = stats.chi2_contingency(t)
    r1 = t.iloc[1, 1] / t.iloc[1].sum()
    r0 = t.iloc[0, 1] / t.iloc[0].sum()
    return {"label": label, "n_treated": int(t.iloc[1].sum()),
            "n_control": int(t.iloc[0].sum()),
            "rate_treated": float(r1), "rate_control": float(r0),
            "diff": float(r1 - r0), "p": float(p)}


nested = []
nested.append(eff(pd.Series(True, index=df.index), "overall"))
nested.append(eff(df["feature_013"] == 0, "f013=0"))
nested.append(eff((df["feature_013"] == 0) & (df["feature_015"] == 0), "f013=0,f015=0"))
nested.append(eff((df["feature_013"] == 0) & (df["feature_015"] == 0) & (df["feature_021"] == 0),
                   "f013=0,f015=0,f021=0"))
nested.append(eff((df["feature_013"] == 0) & (df["feature_015"] == 0) & (df["feature_021"] == 0) & (df["feature_027"] == 0),
                   "f013=0,f015=0,f021=0,f027=0"))
nested.append(eff((df["feature_013"] == 0) & (df["feature_015"] == 0) & (df["feature_021"] == 0) & (df["feature_027"] == 0) & (df["feature_001"] == 0),
                   "f013=0,f015=0,f021=0,f027=0,f001=0"))
nested.append(eff(df["feature_013"] == 1, "f013=1"))
nested.append(eff((df["feature_013"] == 0) & (df["feature_015"] == 1), "f013=0,f015=1"))
nested.append(eff((df["feature_013"] == 0) & (df["feature_021"] == 1), "f013=0,f021=1"))
nested.append(eff((df["feature_013"] == 0) & (df["feature_027"] == 1), "f013=0,f027=1"))
R["iter_extra_f008_nested"] = nested

# also test reverse: each of f015, f021, f027 effect within (f013=0, f008=1, f015=0/etc) — ablation
combo_eff = {}

# In the otherwise-responsive cohort minus the treatment of interest
def ee(mask, label, treat):
    s = df[mask]
    if s[treat].nunique() < 2:
        return None
    t = pd.crosstab(s[treat], s["objective_response"])
    if t.shape != (2, 2):
        return None
    chi2, p, _, _ = stats.chi2_contingency(t)
    r1 = t.iloc[1, 1] / t.iloc[1].sum()
    r0 = t.iloc[0, 1] / t.iloc[0].sum()
    return {"label": label, "n_treated": int(t.iloc[1].sum()),
            "n_control": int(t.iloc[0].sum()),
            "rate_treated": float(r1), "rate_control": float(r0),
            "diff": float(r1 - r0), "p": float(p)}


# f015 effect when f021=0, f027=0, f008=1, f013=0
combo_eff["f015|strict,no_other_treats"] = ee(
    (df["feature_013"] == 0) & (df["feature_008"] == 1) & (df["feature_021"] == 0) & (df["feature_027"] == 0),
    "f013=0,f008=1,f021=0,f027=0", "feature_015")
combo_eff["f021|strict,no_other_treats"] = ee(
    (df["feature_013"] == 0) & (df["feature_008"] == 1) & (df["feature_015"] == 0) & (df["feature_027"] == 0),
    "f013=0,f008=1,f015=0,f027=0", "feature_021")
combo_eff["f027|strict,no_other_treats"] = ee(
    (df["feature_013"] == 0) & (df["feature_008"] == 1) & (df["feature_015"] == 0) & (df["feature_021"] == 0),
    "f013=0,f008=1,f015=0,f021=0", "feature_027")
R["iter_extra_clean_treatment_effects"] = combo_eff

# Final: very pure responder cell (no resistance, all favorable)
pure = df[(df["feature_013"] == 0) & (df["feature_008"] == 1) &
          (df["feature_015"] == 0) & (df["feature_021"] == 0) & (df["feature_027"] == 0)]
R["iter_extra_pure_responder"] = {
    "n": int(len(pure)), "rate": float(pure["objective_response"].mean()),
}

# Save
with open("results_full.json", "w") as f:
    json.dump(R, f, indent=2, default=str)

print(json.dumps(R["iter_extra_f008_nested"], indent=2))
print()
print(json.dumps(R["iter_extra_clean_treatment_effects"], indent=2))
print()
print(json.dumps(R["iter_extra_pure_responder"], indent=2))
