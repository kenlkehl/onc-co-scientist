"""Follow-up: characterize the enzalutamide responder subgroup and confirm other treatments."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
df = pd.read_parquet("dataset.parquet")

OUT = {}

def rr_in_subgroup(treat, mask, label):
    sub = df[mask]
    if len(sub) < 30:
        return None
    rr_t = float(sub.loc[sub[treat] == 1, "objective_response"].mean())
    rr_n = float(sub.loc[sub[treat] == 0, "objective_response"].mean())
    n_t = int((sub[treat] == 1).sum())
    n_n = int((sub[treat] == 0).sum())
    table = pd.crosstab(sub[treat], sub["objective_response"])
    try:
        chi2, p, _, _ = stats.chi2_contingency(table.values)
    except Exception:
        p = float("nan")
    return {"label": label, "n_treated": n_t, "n_untreated": n_n,
            "rr_treated": rr_t, "rr_untreated": rr_n,
            "diff": rr_t - rr_n, "p_value": float(p)}

# Build progressively more restricted enzalutamide-responsive subgroup
ENZ = "treatment_enzalutamide"
masks = [
    ("All patients", pd.Series(True, index=df.index)),
    ("Non-mCRPC", df["mcrpc"] == 0),
    ("AR-V7 negative", df["ar_v7_positive"] == 0),
    ("BRCA2 negative", df["brca2_mutation"] == 0),
    ("MSI low", df["msi_high"] == 0),
    ("Non-mCRPC + AR-V7 neg", (df["mcrpc"] == 0) & (df["ar_v7_positive"] == 0)),
    ("Non-mCRPC + AR-V7 neg + BRCA2 neg",
     (df["mcrpc"] == 0) & (df["ar_v7_positive"] == 0) & (df["brca2_mutation"] == 0)),
    ("Non-mCRPC + AR-V7 neg + BRCA2 neg + MSI low",
     (df["mcrpc"] == 0) & (df["ar_v7_positive"] == 0) & (df["brca2_mutation"] == 0) &
     (df["msi_high"] == 0)),
    ("mCRPC", df["mcrpc"] == 1),
    ("AR-V7 positive", df["ar_v7_positive"] == 1),
    ("BRCA2 positive", df["brca2_mutation"] == 1),
    ("MSI high", df["msi_high"] == 1),
    ("mCRPC OR AR-V7+ OR BRCA2+ OR MSI-high",
     (df["mcrpc"] == 1) | (df["ar_v7_positive"] == 1) | (df["brca2_mutation"] == 1) |
     (df["msi_high"] == 1)),
]

OUT["enzalutamide_subgroups"] = []
for label, mask in masks:
    r = rr_in_subgroup(ENZ, mask, label)
    if r:
        OUT["enzalutamide_subgroups"].append(r)

# Also test joint model with all four modifiers as interactions
formula = ("objective_response ~ treatment_enzalutamide * (mcrpc + ar_v7_positive + "
           "brca2_mutation + msi_high) + ecog_ps + age_years + albumin_g_dl + "
           "ldh_u_l + hemoglobin_g_dl + visceral_mets + weight_loss_pct_6mo + crp_mg_l")
m = smf.logit(formula, data=df).fit(disp=0, maxiter=300)
OUT["enzalutamide_joint_model"] = {}
for name in m.params.index:
    OUT["enzalutamide_joint_model"][name] = {
        "coef": float(m.params[name]),
        "p_value": float(m.pvalues[name]),
        "or": float(np.exp(m.params[name])),
    }

# For each other treatment, try the same logic — does any combination unmask an effect?
def explore_treatment(treat):
    out = []
    masks = [
        ("All", pd.Series(True, index=df.index)),
        ("Non-mCRPC", df["mcrpc"] == 0),
        ("mCRPC", df["mcrpc"] == 1),
        ("Visceral mets", df["visceral_mets"] == 1),
        ("No visceral", df["visceral_mets"] == 0),
        ("BRCA2+", df["brca2_mutation"] == 1),
        ("BRCA2 negative", df["brca2_mutation"] == 0),
        ("AR-V7+", df["ar_v7_positive"] == 1),
        ("AR-V7 neg", df["ar_v7_positive"] == 0),
        ("MSI-high", df["msi_high"] == 1),
        ("MSI low", df["msi_high"] == 0),
        ("PSMA-high", df["psma_high"] == 1),
        ("PSMA low", df["psma_high"] == 0),
        ("ECOG 0-1", df["ecog_ps"] <= 1),
        ("ECOG 2", df["ecog_ps"] == 2),
    ]
    for label, mask in masks:
        r = rr_in_subgroup(treat, mask, label)
        if r:
            out.append(r)
    return out

for t in ["treatment_abiraterone", "treatment_docetaxel", "treatment_olaparib",
          "treatment_lu177_psma", "treatment_pembrolizumab"]:
    OUT[f"{t}_subgroups"] = explore_treatment(t)

# Also: does enzalutamide effect depend on the OTHER treatments (combination)?
# Each combination of [enzalutamide, abiraterone, docetaxel] etc.
combo_results = []
for treat in ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
              "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    for other in ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
                  "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
        if other == treat:
            continue
        # within other == 0
        sub = df[df[other] == 0]
        if len(sub) > 200:
            r = rr_in_subgroup(treat, df[other] == 0, f"{treat} effect within {other}=0")
            r2 = rr_in_subgroup(treat, df[other] == 1, f"{treat} effect within {other}=1")
            if r and r2:
                combo_results.append({"treat": treat, "other": other,
                                      "within_other_neg": r, "within_other_pos": r2})
OUT["treatment_combo_effects"] = combo_results

# Save
def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [jsonable(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o

with open("my_followup.json", "w") as f:
    json.dump(jsonable(OUT), f, indent=2, default=str)
print("Wrote my_followup.json")
