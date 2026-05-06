"""Iteration 16+: refine the ven/aza subgroup further; check 7+3 subgroup."""
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
df = pd.read_parquet("dataset.parquet")

OUT = []

def log(label, **kw):
    rec = dict(kw); rec["__label__"] = label
    OUT.append(rec)
    print(f"[{label}] {kw}")


# Stratified ven/aza by 8 cells (unfit x npm1 x tp53)
print("=== Iteration 16: 8-cell ven/aza stratification ===")
for unfit in (0, 1):
    for npm in (0, 1):
        for tp in (0, 1):
            sub = df[
                (df["unfit_for_intensive"] == unfit)
                & (df["npm1_mutation"] == npm)
                & (df["tp53_mutation"] == tp)
            ]
            a = sub.loc[sub["treatment_venetoclax_azacitidine"] == 1, "objective_response"]
            b = sub.loc[sub["treatment_venetoclax_azacitidine"] == 0, "objective_response"]
            if len(a) > 5 and len(b) > 5:
                tab = pd.crosstab(sub["treatment_venetoclax_azacitidine"], sub["objective_response"])
                p = stats.fisher_exact(tab.values)[1] if tab.shape == (2, 2) else None
                log(
                    f"venaza_cell_unfit{unfit}_npm{npm}_tp53{tp}",
                    n_on=int(len(a)), n_off=int(len(b)),
                    rate_on=float(a.mean()), rate_off=float(b.mean()),
                    effect=float(a.mean() - b.mean()), p=p,
                )

# Same with complex_karyotype as fourth modifier inside (unfit, npm1=1, tp53=0)
print("\n=== Iteration 17: ven/aza in unfit+NPM1+ +TP53wt by complex karyotype ===")
for ck in (0, 1):
    sub = df[
        (df["unfit_for_intensive"] == 1)
        & (df["npm1_mutation"] == 1)
        & (df["tp53_mutation"] == 0)
        & (df["complex_karyotype"] == ck)
    ]
    a = sub.loc[sub["treatment_venetoclax_azacitidine"] == 1, "objective_response"]
    b = sub.loc[sub["treatment_venetoclax_azacitidine"] == 0, "objective_response"]
    if len(a) > 5 and len(b) > 5:
        tab = pd.crosstab(sub["treatment_venetoclax_azacitidine"], sub["objective_response"])
        p = stats.fisher_exact(tab.values)[1] if tab.shape == (2, 2) else None
        log(
            f"venaza_unfit_npm1_tp53wt_ck{ck}",
            n_on=int(len(a)), n_off=int(len(b)),
            rate_on=float(a.mean()), rate_off=float(b.mean()),
            effect=float(a.mean() - b.mean()), p=p,
        )

# Check ven/aza effect in fit + NPM1+ + TP53wt
print("\n=== Iteration 18: ven/aza in fit + NPM1+ ===")
sub = df[
    (df["unfit_for_intensive"] == 0)
    & (df["npm1_mutation"] == 1)
    & (df["tp53_mutation"] == 0)
]
a = sub.loc[sub["treatment_venetoclax_azacitidine"] == 1, "objective_response"]
b = sub.loc[sub["treatment_venetoclax_azacitidine"] == 0, "objective_response"]
if len(a) > 5 and len(b) > 5:
    tab = pd.crosstab(sub["treatment_venetoclax_azacitidine"], sub["objective_response"])
    p = stats.fisher_exact(tab.values)[1] if tab.shape == (2, 2) else None
    log("venaza_fit_npm1_tp53wt", n_on=int(len(a)), n_off=int(len(b)),
        rate_on=float(a.mean()), rate_off=float(b.mean()),
        effect=float(a.mean() - b.mean()), p=p)


# Iteration 19: 7+3 subgroup search
print("\n=== Iteration 19: 7+3 stratified by complex karyotype + tp53 ===")
for ck in (0, 1):
    for tp in (0, 1):
        sub = df[
            (df["complex_karyotype"] == ck)
            & (df["tp53_mutation"] == tp)
        ]
        a = sub.loc[sub["treatment_7plus3"] == 1, "objective_response"]
        b = sub.loc[sub["treatment_7plus3"] == 0, "objective_response"]
        if len(a) > 5 and len(b) > 5:
            tab = pd.crosstab(sub["treatment_7plus3"], sub["objective_response"])
            p = stats.fisher_exact(tab.values)[1] if tab.shape == (2, 2) else None
            log(
                f"sevenplus3_ck{ck}_tp53{tp}",
                n_on=int(len(a)), n_off=int(len(b)),
                rate_on=float(a.mean()), rate_off=float(b.mean()),
                effect=float(a.mean() - b.mean()), p=p,
            )

# Iteration 20: more granular treatment by NPM1 status x ECOG
print("\n=== Iteration 20: ven/aza by NPM1+/TP53wt x ecog_ps ===")
for ecog in (0, 1, 2):
    sub = df[
        (df["npm1_mutation"] == 1)
        & (df["tp53_mutation"] == 0)
        & (df["ecog_ps"] == ecog)
    ]
    a = sub.loc[sub["treatment_venetoclax_azacitidine"] == 1, "objective_response"]
    b = sub.loc[sub["treatment_venetoclax_azacitidine"] == 0, "objective_response"]
    if len(a) > 5 and len(b) > 5:
        tab = pd.crosstab(sub["treatment_venetoclax_azacitidine"], sub["objective_response"])
        p = stats.fisher_exact(tab.values)[1] if tab.shape == (2, 2) else None
        log(
            f"venaza_npm1_tp53wt_ecog{ecog}",
            n_on=int(len(a)), n_off=int(len(b)),
            rate_on=float(a.mean()), rate_off=float(b.mean()),
            effect=float(a.mean() - b.mean()), p=p,
        )

# Iteration 21: triple interaction model for venetoclax: with key modifiers
print("\n=== Iteration 21: ven/aza joint logistic with all main modifiers ===")
res = smf.logit(
    "objective_response ~ treatment_venetoclax_azacitidine * unfit_for_intensive "
    "+ treatment_venetoclax_azacitidine * npm1_mutation "
    "+ treatment_venetoclax_azacitidine * tp53_mutation "
    "+ treatment_venetoclax_azacitidine * complex_karyotype "
    "+ ecog_ps + age_years + albumin_g_dl + secondary_aml + flt3_itd",
    data=df,
).fit(disp=0)
print(res.summary().tables[1])
for k, v in res.params.items():
    if "treatment" in k:
        log(f"venaza_joint_{k}", term=k, effect=float(v), p=float(res.pvalues[k]))


# Iteration 22: predicted treatment effect contrast — for each row predict y if venaza=0 vs 1
print("\n=== Iteration 22: model-based effect by NPM1/TP53/unfit cell ===")
# Use a flexible logistic with all relevant interactions
formula = (
    "objective_response ~ treatment_venetoclax_azacitidine"
    " * (unfit_for_intensive + npm1_mutation + tp53_mutation + complex_karyotype)"
    " + age_years + sex_female + ecog_ps + secondary_aml"
    " + flt3_itd + flt3_tkd + idh1_mutation + idh2_mutation"
    " + wbc_k_per_ul + blast_pct_marrow + albumin_g_dl"
)
m = smf.logit(formula, data=df).fit(disp=0)
print(m.summary().tables[1])
# Also test 7+3 in a similar joint model
m2 = smf.logit(
    "objective_response ~ treatment_7plus3 * (complex_karyotype + tp53_mutation + unfit_for_intensive) "
    "+ age_years + sex_female + ecog_ps + secondary_aml + npm1_mutation + flt3_itd "
    "+ wbc_k_per_ul + blast_pct_marrow + albumin_g_dl",
    data=df,
).fit(disp=0)
print(m2.summary().tables[1])
for k in m2.params.index:
    if "treatment" in k:
        log(f"sevenp3_joint_{k}", term=k, effect=float(m2.params[k]), p=float(m2.pvalues[k]))


# Iteration 23: refine 7+3 effect by complex karyotype subset
print("\n=== Iteration 23: 7+3 in normal-karyotype, all-comers ===")
for predicate_name, predicate in [
    ("normal_karyo", df["complex_karyotype"] == 0),
    ("normal_karyo+tp53wt", (df["complex_karyotype"] == 0) & (df["tp53_mutation"] == 0)),
    ("complex_karyo", df["complex_karyotype"] == 1),
    ("tp53_mut", df["tp53_mutation"] == 1),
]:
    sub = df[predicate]
    a = sub.loc[sub["treatment_7plus3"] == 1, "objective_response"]
    b = sub.loc[sub["treatment_7plus3"] == 0, "objective_response"]
    tab = pd.crosstab(sub["treatment_7plus3"], sub["objective_response"])
    p = stats.fisher_exact(tab.values)[1] if tab.shape == (2, 2) else None
    log(
        f"sevenp3_{predicate_name}",
        n_on=int(len(a)), n_off=int(len(b)),
        rate_on=float(a.mean()), rate_off=float(b.mean()),
        effect=float(a.mean() - b.mean()), p=p,
    )


# Save
import os
os.makedirs("scratch", exist_ok=True)
with open("scratch/results2.json", "w") as fh:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o) if isinstance(o, np.floating) else int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    json.dump(OUT, fh, default=default, indent=2)
print(f"\nWrote {len(OUT)} additional result records.")
