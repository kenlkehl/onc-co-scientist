"""Comprehensive analysis of AML cohort ds001_aml.

Runs all statistical tests needed for the transcript and writes results to
a JSON file used to build transcript.json + analysis_summary.txt.
"""

from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

import statsmodels.api as sm
import statsmodels.formula.api as smf

DF = pd.read_parquet("dataset.parquet")
RESULTS: list[dict[str, Any]] = []


def record(
    iteration: int,
    hyp_ids: list[str],
    summary: str,
    p_value: float | None,
    effect: float | None,
    significant: bool | None = None,
    code: str | None = None,
) -> None:
    if significant is None and p_value is not None:
        significant = bool(p_value < 0.05)
    RESULTS.append(
        {
            "iteration": iteration,
            "hypothesis_ids": hyp_ids,
            "result_summary": summary,
            "p_value": None if p_value is None else float(p_value),
            "effect_estimate": None if effect is None else float(effect),
            "significant": significant,
            "code": code,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chi2_or(df: pd.DataFrame, exposure: str, outcome: str = "objective_response"):
    tab = pd.crosstab(df[exposure], df[outcome])
    if tab.shape != (2, 2):
        return None, None, None, None
    chi2, p, _, _ = stats.chi2_contingency(tab)
    a, b = tab.iloc[1, 1], tab.iloc[1, 0]
    c, d = tab.iloc[0, 1], tab.iloc[0, 0]
    or_ = (a * d) / (b * c) if b * c else float("inf")
    rd = a / (a + b) - c / (c + d)
    return chi2, p, or_, rd


def logit_subgroup(df: pd.DataFrame, predictor: str, adjust: list[str] | None = None,
                   outcome: str = "objective_response"):
    cols = [predictor] + (adjust or [])
    X = sm.add_constant(df[cols].astype(float))
    y = df[outcome].astype(int)
    res = sm.Logit(y, X).fit(disp=False)
    coef = res.params[predictor]
    p = res.pvalues[predictor]
    or_ = float(np.exp(coef))
    return coef, p, or_, res


def run_interaction(df: pd.DataFrame, treatment: str, modifier: str,
                    outcome: str = "objective_response"):
    """Logistic regression with treatment*modifier interaction."""
    sub = df[[treatment, modifier, outcome]].copy().astype(float)
    sub["int"] = sub[treatment] * sub[modifier]
    X = sm.add_constant(sub[[treatment, modifier, "int"]])
    y = sub[outcome].astype(int)
    res = sm.Logit(y, X).fit(disp=False)
    coef = res.params["int"]
    p = res.pvalues["int"]
    return coef, p, res


def diff_in_diff_response(df: pd.DataFrame, treatment: str, modifier: str,
                          outcome: str = "objective_response"):
    """Effect of treatment within modifier=1 vs modifier=0."""
    out = {}
    for m in (0, 1):
        sub = df[df[modifier] == m]
        on = sub.loc[sub[treatment] == 1, outcome].mean()
        off = sub.loc[sub[treatment] == 0, outcome].mean()
        n_on = int(sub[treatment].sum())
        n_off = int(len(sub) - n_on)
        out[m] = {"on": on, "off": off, "diff": on - off, "n_on": n_on, "n_off": n_off}
    return out


# ---------------------------------------------------------------------------
# Iteration 1: Baseline associations of demographics/clinical with outcome
# ---------------------------------------------------------------------------
print("=== Iteration 1: baseline patient-feature associations ===")

# h1: older age associated with lower response
age_on = DF[DF["objective_response"] == 1]["age_years"]
age_off = DF[DF["objective_response"] == 0]["age_years"]
t, p = stats.ttest_ind(age_on, age_off, equal_var=False)
record(
    1, ["h1"],
    f"Mean age: {age_on.mean():.1f} responders vs {age_off.mean():.1f} non-responders (Welch t-test).",
    p, float(age_on.mean() - age_off.mean()),
    code="stats.ttest_ind(df.loc[df.objective_response==1,'age_years'], df.loc[df.objective_response==0,'age_years'], equal_var=False)",
)

# h2: ECOG PS associations
res = smf.logit("objective_response ~ ecog_ps", data=DF).fit(disp=False)
record(
    1, ["h2"],
    f"Logistic: 1-unit increase in ECOG PS multiplies odds of response by {np.exp(res.params['ecog_ps']):.3f}.",
    float(res.pvalues["ecog_ps"]),
    float(res.params["ecog_ps"]),
    code="smf.logit('objective_response ~ ecog_ps', data=df)",
)

# h3: female sex
chi2, p, or_, rd = chi2_or(DF, "sex_female")
record(
    1, ["h3"],
    f"Response rate female {DF.loc[DF.sex_female==1,'objective_response'].mean():.3f} vs male {DF.loc[DF.sex_female==0,'objective_response'].mean():.3f} (chi2 OR={or_:.3f}).",
    p, float(rd),
)

# h4: secondary AML lower response
chi2, p, or_, rd = chi2_or(DF, "secondary_aml")
record(
    1, ["h4"],
    f"Response: secondary AML {DF.loc[DF.secondary_aml==1,'objective_response'].mean():.3f} vs de novo {DF.loc[DF.secondary_aml==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)

# h5: unfit_for_intensive lower response
chi2, p, or_, rd = chi2_or(DF, "unfit_for_intensive")
record(
    1, ["h5"],
    f"Response: unfit {DF.loc[DF.unfit_for_intensive==1,'objective_response'].mean():.3f} vs fit {DF.loc[DF.unfit_for_intensive==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)


# ---------------------------------------------------------------------------
# Iteration 2: Cytogenetic/molecular markers
# ---------------------------------------------------------------------------
print("=== Iteration 2: cytogenetic/molecular markers ===")

for hid, col in [
    ("h6", "complex_karyotype"),
    ("h7", "tp53_mutation"),
    ("h8", "npm1_mutation"),
    ("h9", "flt3_itd"),
    ("h10", "flt3_tkd"),
    ("h11", "idh1_mutation"),
    ("h12", "idh2_mutation"),
]:
    chi2, p, or_, rd = chi2_or(DF, col)
    on = DF.loc[DF[col] == 1, "objective_response"].mean()
    off = DF.loc[DF[col] == 0, "objective_response"].mean()
    record(
        2, [hid],
        f"{col}=1 response {on:.3f} vs =0 {off:.3f} (chi2 OR={or_:.3f}).",
        p, float(rd),
    )


# ---------------------------------------------------------------------------
# Iteration 3: Treatment main effects (unadjusted)
# ---------------------------------------------------------------------------
print("=== Iteration 3: treatment main effects (unadjusted) ===")

for hid, col in [
    ("h13", "treatment_midostaurin"),
    ("h14", "treatment_gilteritinib"),
    ("h15", "treatment_ivosidenib"),
    ("h16", "treatment_enasidenib"),
    ("h17", "treatment_venetoclax_azacitidine"),
    ("h18", "treatment_7plus3"),
]:
    chi2, p, or_, rd = chi2_or(DF, col)
    on = DF.loc[DF[col] == 1, "objective_response"].mean()
    off = DF.loc[DF[col] == 0, "objective_response"].mean()
    record(
        3, [hid],
        f"{col}=1 response {on:.3f} (n={int(DF[col].sum())}) vs =0 {off:.3f} (chi2 OR={or_:.3f}).",
        p, float(rd),
    )


# ---------------------------------------------------------------------------
# Iteration 4: Lab/continuous features vs response
# ---------------------------------------------------------------------------
print("=== Iteration 4: lab/continuous features ===")

cont_cols = [
    "wbc_k_per_ul", "blast_pct_marrow", "albumin_g_dl", "ldh_u_l",
    "weight_loss_pct_6mo", "crp_mg_l", "nlr", "hemoglobin_g_dl",
    "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl",
    "creatinine_mg_dl", "bun_mg_dl", "sodium_meq_l", "potassium_meq_l",
    "calcium_mg_dl",
]
for i, col in enumerate(cont_cols, start=19):
    hid = f"h{i}"
    on = DF.loc[DF["objective_response"] == 1, col]
    off = DF.loc[DF["objective_response"] == 0, col]
    t, p = stats.ttest_ind(on, off, equal_var=False)
    record(
        4, [hid],
        f"Mean {col}: responders {on.mean():.3f} vs non-responders {off.mean():.3f} (Welch t-test).",
        float(p), float(on.mean() - off.mean()),
    )


# ---------------------------------------------------------------------------
# Iteration 5: Multivariable adjusted treatment effects
# ---------------------------------------------------------------------------
print("=== Iteration 5: adjusted treatment effects ===")

base_adj = (
    "age_years + sex_female + ecog_ps + secondary_aml + unfit_for_intensive + "
    "complex_karyotype + flt3_itd + flt3_tkd + idh1_mutation + idh2_mutation + "
    "npm1_mutation + tp53_mutation + wbc_k_per_ul + blast_pct_marrow + albumin_g_dl"
)

for hid, treat in [
    ("h36", "treatment_midostaurin"),
    ("h37", "treatment_gilteritinib"),
    ("h38", "treatment_ivosidenib"),
    ("h39", "treatment_enasidenib"),
    ("h40", "treatment_venetoclax_azacitidine"),
    ("h41", "treatment_7plus3"),
]:
    formula = f"objective_response ~ {treat} + {base_adj}"
    res = smf.logit(formula, data=DF).fit(disp=False)
    coef = res.params[treat]
    p = res.pvalues[treat]
    record(
        5, [hid],
        f"Adjusted logistic OR for {treat}: {np.exp(coef):.3f} (p={p:.3g}).",
        float(p), float(coef),
        code=f"smf.logit('objective_response ~ {treat} + ...covariates', data=df)",
    )


# ---------------------------------------------------------------------------
# Iteration 6: FLT3-targeted therapy x FLT3 mutation interactions
# ---------------------------------------------------------------------------
print("=== Iteration 6: FLT3 inhibitors x FLT3 ===")

for hid_int, hid_sub, treat, mod in [
    ("h42", "h43", "treatment_midostaurin", "flt3_itd"),
    ("h44", "h45", "treatment_midostaurin", "flt3_tkd"),
    ("h46", "h47", "treatment_gilteritinib", "flt3_itd"),
    ("h48", "h49", "treatment_gilteritinib", "flt3_tkd"),
]:
    coef, p, _ = run_interaction(DF, treat, mod)
    record(
        6, [hid_int],
        f"Logit interaction {treat}*{mod}: coef={coef:.3f} (OR={np.exp(coef):.3f}).",
        float(p), float(coef),
    )
    sub = diff_in_diff_response(DF, treat, mod)
    txt = (
        f"Stratified response: {mod}=0: on={sub[0]['on']:.3f} off={sub[0]['off']:.3f} (Δ={sub[0]['diff']:+.3f}); "
        f"{mod}=1: on={sub[1]['on']:.3f} off={sub[1]['off']:.3f} (Δ={sub[1]['diff']:+.3f})."
    )
    # test diff in diff via chi2 within each subgroup
    sub1 = DF[DF[mod] == 1]
    chi2, p2, or_, rd = chi2_or(sub1, treat)
    record(
        6, [hid_sub],
        txt + f" Within {mod}=1: chi2 OR={or_:.3f}, p={p2:.3g}.",
        float(p2), float(rd),
    )


# ---------------------------------------------------------------------------
# Iteration 7: IDH inhibitors x IDH mutation interactions
# ---------------------------------------------------------------------------
print("=== Iteration 7: IDH inhibitors x IDH ===")

for hid_int, hid_sub, treat, mod in [
    ("h50", "h51", "treatment_ivosidenib", "idh1_mutation"),
    ("h52", "h53", "treatment_ivosidenib", "idh2_mutation"),
    ("h54", "h55", "treatment_enasidenib", "idh2_mutation"),
    ("h56", "h57", "treatment_enasidenib", "idh1_mutation"),
]:
    coef, p, _ = run_interaction(DF, treat, mod)
    record(
        7, [hid_int],
        f"Logit interaction {treat}*{mod}: coef={coef:.3f} (OR={np.exp(coef):.3f}).",
        float(p), float(coef),
    )
    sub = diff_in_diff_response(DF, treat, mod)
    txt = (
        f"Stratified: {mod}=0 Δ={sub[0]['diff']:+.3f} (n_on={sub[0]['n_on']}); "
        f"{mod}=1 Δ={sub[1]['diff']:+.3f} (n_on={sub[1]['n_on']})."
    )
    sub1 = DF[DF[mod] == 1]
    chi2, p2, or_, rd = chi2_or(sub1, treat)
    record(
        7, [hid_sub],
        txt + f" Within {mod}=1: OR={or_:.3f}, p={p2:.3g}.",
        float(p2), float(rd),
    )


# ---------------------------------------------------------------------------
# Iteration 8: TP53 / complex karyotype interactions with treatment
# ---------------------------------------------------------------------------
print("=== Iteration 8: TP53 / complex karyotype interactions ===")

for hid, treat, mod in [
    ("h58", "treatment_7plus3", "tp53_mutation"),
    ("h59", "treatment_7plus3", "complex_karyotype"),
    ("h60", "treatment_venetoclax_azacitidine", "tp53_mutation"),
    ("h61", "treatment_venetoclax_azacitidine", "complex_karyotype"),
]:
    coef, p, _ = run_interaction(DF, treat, mod)
    sub = diff_in_diff_response(DF, treat, mod)
    record(
        8, [hid],
        f"{treat}*{mod}: int_coef={coef:.3f} (p={p:.3g}); strat: {mod}=0 Δ={sub[0]['diff']:+.3f}, {mod}=1 Δ={sub[1]['diff']:+.3f}.",
        float(p), float(coef),
    )


# ---------------------------------------------------------------------------
# Iteration 9: Fitness x intensive therapy
# ---------------------------------------------------------------------------
print("=== Iteration 9: fitness x intensive therapy ===")

for hid, treat, mod in [
    ("h62", "treatment_7plus3", "unfit_for_intensive"),
    ("h63", "treatment_venetoclax_azacitidine", "unfit_for_intensive"),
]:
    coef, p, _ = run_interaction(DF, treat, mod)
    sub = diff_in_diff_response(DF, treat, mod)
    record(
        9, [hid],
        f"{treat}*{mod}: int_coef={coef:.3f} (p={p:.3g}); fit Δ={sub[0]['diff']:+.3f}, unfit Δ={sub[1]['diff']:+.3f}.",
        float(p), float(coef),
    )


# ---------------------------------------------------------------------------
# Iteration 10: Age-based heterogeneity
# ---------------------------------------------------------------------------
print("=== Iteration 10: age-based heterogeneity ===")

DF["age_ge75"] = (DF["age_years"] >= 75).astype(int)
for hid, treat in [
    ("h64", "treatment_7plus3"),
    ("h65", "treatment_venetoclax_azacitidine"),
]:
    coef, p, _ = run_interaction(DF, treat, "age_ge75")
    sub = diff_in_diff_response(DF, treat, "age_ge75")
    record(
        10, [hid],
        f"{treat}*age>=75: int_coef={coef:.3f} (p={p:.3g}); <75 Δ={sub[0]['diff']:+.3f}, >=75 Δ={sub[1]['diff']:+.3f}.",
        float(p), float(coef),
    )


# ---------------------------------------------------------------------------
# Iteration 11: ECOG-based heterogeneity
# ---------------------------------------------------------------------------
print("=== Iteration 11: ECOG-based heterogeneity ===")

DF["ecog_ge2"] = (DF["ecog_ps"] >= 2).astype(int)
for hid, treat in [
    ("h66", "treatment_7plus3"),
    ("h67", "treatment_venetoclax_azacitidine"),
    ("h68", "treatment_midostaurin"),
    ("h69", "treatment_gilteritinib"),
]:
    coef, p, _ = run_interaction(DF, treat, "ecog_ge2")
    sub = diff_in_diff_response(DF, treat, "ecog_ge2")
    record(
        11, [hid],
        f"{treat}*ecog>=2: int_coef={coef:.3f} (p={p:.3g}); ecog<2 Δ={sub[0]['diff']:+.3f}, ecog>=2 Δ={sub[1]['diff']:+.3f}.",
        float(p), float(coef),
    )


# ---------------------------------------------------------------------------
# Iteration 12: Joint subgroup - FLT3 inhibitor effect within FLT3-mutated, fit, ECOG<2
# ---------------------------------------------------------------------------
print("=== Iteration 12: joint subgroup checks ===")

# Gilteritinib in FLT3-ITD or FLT3-TKD (FLT3-mut), stratified by ECOG and fitness
DF["flt3_any"] = ((DF["flt3_itd"] == 1) | (DF["flt3_tkd"] == 1)).astype(int)

# h70: gilteritinib in FLT3-mut overall
sub = DF[DF["flt3_any"] == 1]
chi2, p, or_, rd = chi2_or(sub, "treatment_gilteritinib")
record(
    12, ["h70"],
    f"Within FLT3-mut (n={len(sub)}): gilteritinib resp {sub.loc[sub.treatment_gilteritinib==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_gilteritinib==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)
# h71: gilteritinib in FLT3-mut + fit
sub = DF[(DF["flt3_any"] == 1) & (DF["unfit_for_intensive"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_gilteritinib")
record(
    12, ["h71"],
    f"Within FLT3-mut & fit (n={len(sub)}): gilteritinib resp {sub.loc[sub.treatment_gilteritinib==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_gilteritinib==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)
# h72: midostaurin in FLT3-mut + fit
sub = DF[(DF["flt3_any"] == 1) & (DF["unfit_for_intensive"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_midostaurin")
record(
    12, ["h72"],
    f"Within FLT3-mut & fit (n={len(sub)}): midostaurin resp {sub.loc[sub.treatment_midostaurin==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_midostaurin==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)
# h73: midostaurin within FLT3-ITD specifically + fit
sub = DF[(DF["flt3_itd"] == 1) & (DF["unfit_for_intensive"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_midostaurin")
record(
    12, ["h73"],
    f"Within FLT3-ITD & fit (n={len(sub)}): midostaurin resp {sub.loc[sub.treatment_midostaurin==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_midostaurin==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)


# ---------------------------------------------------------------------------
# Iteration 13: IDH inhibitor joint subgroup checks
# ---------------------------------------------------------------------------
print("=== Iteration 13: IDH joint subgroup ===")

# h74: ivosidenib in idh1+ overall
sub = DF[DF["idh1_mutation"] == 1]
chi2, p, or_, rd = chi2_or(sub, "treatment_ivosidenib")
record(
    13, ["h74"],
    f"Within IDH1-mut (n={len(sub)}): ivosidenib resp {sub.loc[sub.treatment_ivosidenib==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_ivosidenib==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)
# h75: enasidenib in idh2+ overall
sub = DF[DF["idh2_mutation"] == 1]
chi2, p, or_, rd = chi2_or(sub, "treatment_enasidenib")
record(
    13, ["h75"],
    f"Within IDH2-mut (n={len(sub)}): enasidenib resp {sub.loc[sub.treatment_enasidenib==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_enasidenib==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)
# h76: ivosidenib in idh1+, no TP53
sub = DF[(DF["idh1_mutation"] == 1) & (DF["tp53_mutation"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_ivosidenib")
record(
    13, ["h76"],
    f"Within IDH1-mut & TP53-wt (n={len(sub)}): ivosidenib resp {sub.loc[sub.treatment_ivosidenib==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_ivosidenib==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)
# h77: enasidenib in idh2+, no TP53
sub = DF[(DF["idh2_mutation"] == 1) & (DF["tp53_mutation"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_enasidenib")
record(
    13, ["h77"],
    f"Within IDH2-mut & TP53-wt (n={len(sub)}): enasidenib resp {sub.loc[sub.treatment_enasidenib==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_enasidenib==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)


# ---------------------------------------------------------------------------
# Iteration 14: Venetoclax-azacitidine effect heterogeneity
# ---------------------------------------------------------------------------
print("=== Iteration 14: ven-aza heterogeneity ===")

# h78: ven-aza in unfit
sub = DF[DF["unfit_for_intensive"] == 1]
chi2, p, or_, rd = chi2_or(sub, "treatment_venetoclax_azacitidine")
record(
    14, ["h78"],
    f"Within unfit (n={len(sub)}): ven-aza resp {sub.loc[sub.treatment_venetoclax_azacitidine==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_venetoclax_azacitidine==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)
# h79: ven-aza in unfit, no TP53
sub = DF[(DF["unfit_for_intensive"] == 1) & (DF["tp53_mutation"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_venetoclax_azacitidine")
record(
    14, ["h79"],
    f"Within unfit & TP53-wt (n={len(sub)}): ven-aza resp {sub.loc[sub.treatment_venetoclax_azacitidine==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_venetoclax_azacitidine==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)
# h80: ven-aza in unfit, no complex karyotype
sub = DF[(DF["unfit_for_intensive"] == 1) & (DF["complex_karyotype"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_venetoclax_azacitidine")
record(
    14, ["h80"],
    f"Within unfit & non-complex (n={len(sub)}): ven-aza resp {sub.loc[sub.treatment_venetoclax_azacitidine==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_venetoclax_azacitidine==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)
# h81: ven-aza in unfit & TP53-wt & non-complex
sub = DF[
    (DF["unfit_for_intensive"] == 1)
    & (DF["tp53_mutation"] == 0)
    & (DF["complex_karyotype"] == 0)
]
chi2, p, or_, rd = chi2_or(sub, "treatment_venetoclax_azacitidine")
record(
    14, ["h81"],
    f"Within unfit & TP53-wt & non-complex (n={len(sub)}): ven-aza resp {sub.loc[sub.treatment_venetoclax_azacitidine==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_venetoclax_azacitidine==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)


# ---------------------------------------------------------------------------
# Iteration 15: 7+3 effect heterogeneity (fit, NPM1+, FLT3 status, etc.)
# ---------------------------------------------------------------------------
print("=== Iteration 15: 7+3 heterogeneity ===")

# h82: 7+3 in fit
sub = DF[DF["unfit_for_intensive"] == 0]
chi2, p, or_, rd = chi2_or(sub, "treatment_7plus3")
record(
    15, ["h82"],
    f"Within fit (n={len(sub)}): 7+3 resp {sub.loc[sub.treatment_7plus3==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_7plus3==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)
# h83: 7+3 in fit & TP53-wt
sub = DF[(DF["unfit_for_intensive"] == 0) & (DF["tp53_mutation"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_7plus3")
record(
    15, ["h83"],
    f"Within fit & TP53-wt (n={len(sub)}): 7+3 resp {sub.loc[sub.treatment_7plus3==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_7plus3==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)
# h84: 7+3 in fit & TP53-wt & non-complex
sub = DF[
    (DF["unfit_for_intensive"] == 0)
    & (DF["tp53_mutation"] == 0)
    & (DF["complex_karyotype"] == 0)
]
chi2, p, or_, rd = chi2_or(sub, "treatment_7plus3")
record(
    15, ["h84"],
    f"Within fit & TP53-wt & non-complex (n={len(sub)}): 7+3 resp {sub.loc[sub.treatment_7plus3==1,'objective_response'].mean():.3f} vs not {sub.loc[sub.treatment_7plus3==0,'objective_response'].mean():.3f} (OR={or_:.3f}).",
    p, float(rd),
)


# ---------------------------------------------------------------------------
# Iteration 16: Systematic treatment-by-feature interaction screens
# ---------------------------------------------------------------------------
print("=== Iteration 16: systematic interaction screens ===")

mod_cols = [
    "sex_female", "secondary_aml", "unfit_for_intensive", "complex_karyotype",
    "flt3_itd", "flt3_tkd", "idh1_mutation", "idh2_mutation",
    "npm1_mutation", "tp53_mutation", "age_ge75", "ecog_ge2",
]
treat_cols = [
    "treatment_midostaurin", "treatment_gilteritinib", "treatment_ivosidenib",
    "treatment_enasidenib", "treatment_venetoclax_azacitidine", "treatment_7plus3",
]

screen_rows = []
for treat in treat_cols:
    for mod in mod_cols:
        try:
            coef, p, _ = run_interaction(DF, treat, mod)
        except Exception:
            continue
        screen_rows.append({"treatment": treat, "modifier": mod, "coef": coef, "p": p})
screen = pd.DataFrame(screen_rows).sort_values("p")
screen.to_csv("interaction_screen.csv", index=False)

# Record top hits
top = screen.head(15)
hid_counter = 85
for _, row in top.iterrows():
    hid = f"h{hid_counter}"
    hid_counter += 1
    record(
        16, [hid],
        f"Interaction screen hit: {row['treatment']}*{row['modifier']}: coef={row['coef']:.3f}, p={row['p']:.3g}.",
        float(row["p"]), float(row["coef"]),
    )


# ---------------------------------------------------------------------------
# Iteration 17: Joint multi-modifier subgroup search for each treatment
# Identify combinations where Δ(response | on - off) is largest within
# subgroup defined by 1-3 binary features.
# ---------------------------------------------------------------------------
print("=== Iteration 17: joint multi-feature subgroup search ===")

binary_mods = [
    "secondary_aml", "unfit_for_intensive", "complex_karyotype",
    "flt3_itd", "flt3_tkd", "idh1_mutation", "idh2_mutation",
    "npm1_mutation", "tp53_mutation", "age_ge75", "ecog_ge2",
]

def best_subgroup(df: pd.DataFrame, treat: str, max_features: int = 3, min_n: int = 200):
    """Search combinations of up to max_features binary modifiers (each =0 or =1)
    and find the subgroup where the treatment vs no-treatment response delta
    is largest (positive)."""
    best = None
    from itertools import combinations, product
    for k in range(1, max_features + 1):
        for combo in combinations(binary_mods, k):
            for vals in product((0, 1), repeat=k):
                mask = np.ones(len(df), dtype=bool)
                for c, v in zip(combo, vals):
                    mask &= (df[c].values == v)
                sub = df[mask]
                if len(sub) < min_n:
                    continue
                n_on = int(sub[treat].sum())
                n_off = int(len(sub) - n_on)
                if n_on < 30 or n_off < 30:
                    continue
                on_rate = sub.loc[sub[treat] == 1, "objective_response"].mean()
                off_rate = sub.loc[sub[treat] == 0, "objective_response"].mean()
                delta = on_rate - off_rate
                # require statistical significance for ranking
                tab = pd.crosstab(sub[treat], sub["objective_response"])
                if tab.shape != (2, 2):
                    continue
                _, p, _, _ = stats.chi2_contingency(tab)
                key = (delta, -p)
                if best is None or key > best[0]:
                    best = (key, {
                        "combo": combo, "vals": vals, "n": len(sub),
                        "n_on": n_on, "n_off": n_off,
                        "on_rate": on_rate, "off_rate": off_rate,
                        "delta": delta, "p": p,
                    })
    return best[1] if best else None


hid_counter = 100
for treat in treat_cols:
    res = best_subgroup(DF, treat)
    if res is None:
        continue
    hid = f"h{hid_counter}"
    hid_counter += 1
    desc = " & ".join(f"{c}={v}" for c, v in zip(res["combo"], res["vals"]))
    record(
        17, [hid],
        f"Best Δ-response subgroup for {treat}: [{desc}] n={res['n']}, on={res['on_rate']:.3f} off={res['off_rate']:.3f} Δ={res['delta']:+.3f} chi2 p={res['p']:.3g}.",
        float(res["p"]), float(res["delta"]),
    )


# ---------------------------------------------------------------------------
# Iteration 18: Refined hypotheses around mutation-targeted subgroups
# Final treatment-effect subgroup hypotheses for each treatment.
# ---------------------------------------------------------------------------
print("=== Iteration 18: refined / final subgroup hypotheses ===")

# h120: Gilteritinib benefit is concentrated in FLT3-ITD patients (main signal)
sub = DF[DF["flt3_itd"] == 1]
chi2, p, or_, rd = chi2_or(sub, "treatment_gilteritinib")
record(
    18, ["h120"],
    f"Final: gilteritinib in FLT3-ITD (n={len(sub)}): on={sub.loc[sub.treatment_gilteritinib==1,'objective_response'].mean():.3f}, off={sub.loc[sub.treatment_gilteritinib==0,'objective_response'].mean():.3f}, Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)

# h121: Midostaurin benefit in FLT3-ITD + fit
sub = DF[(DF["flt3_itd"] == 1) & (DF["unfit_for_intensive"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_midostaurin")
record(
    18, ["h121"],
    f"Final: midostaurin in FLT3-ITD & fit (n={len(sub)}): on={sub.loc[sub.treatment_midostaurin==1,'objective_response'].mean():.3f}, off={sub.loc[sub.treatment_midostaurin==0,'objective_response'].mean():.3f}, Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)

# h122: Ivosidenib in IDH1-mut, TP53-wt, non-complex
sub = DF[(DF["idh1_mutation"] == 1) & (DF["tp53_mutation"] == 0) & (DF["complex_karyotype"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_ivosidenib")
record(
    18, ["h122"],
    f"Final: ivosidenib in IDH1-mut & TP53-wt & non-complex (n={len(sub)}): on={sub.loc[sub.treatment_ivosidenib==1,'objective_response'].mean():.3f}, off={sub.loc[sub.treatment_ivosidenib==0,'objective_response'].mean():.3f}, Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)

# h123: Enasidenib in IDH2-mut & TP53-wt
sub = DF[(DF["idh2_mutation"] == 1) & (DF["tp53_mutation"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_enasidenib")
record(
    18, ["h123"],
    f"Final: enasidenib in IDH2-mut & TP53-wt (n={len(sub)}): on={sub.loc[sub.treatment_enasidenib==1,'objective_response'].mean():.3f}, off={sub.loc[sub.treatment_enasidenib==0,'objective_response'].mean():.3f}, Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)

# h124: Ven-aza in unfit & TP53-wt & non-complex
sub = DF[
    (DF["unfit_for_intensive"] == 1)
    & (DF["tp53_mutation"] == 0)
    & (DF["complex_karyotype"] == 0)
]
chi2, p, or_, rd = chi2_or(sub, "treatment_venetoclax_azacitidine")
record(
    18, ["h124"],
    f"Final: ven-aza in unfit & TP53-wt & non-complex (n={len(sub)}): on={sub.loc[sub.treatment_venetoclax_azacitidine==1,'objective_response'].mean():.3f}, off={sub.loc[sub.treatment_venetoclax_azacitidine==0,'objective_response'].mean():.3f}, Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)

# h125: 7+3 in fit & TP53-wt & non-complex
sub = DF[
    (DF["unfit_for_intensive"] == 0)
    & (DF["tp53_mutation"] == 0)
    & (DF["complex_karyotype"] == 0)
]
chi2, p, or_, rd = chi2_or(sub, "treatment_7plus3")
record(
    18, ["h125"],
    f"Final: 7+3 in fit & TP53-wt & non-complex (n={len(sub)}): on={sub.loc[sub.treatment_7plus3==1,'objective_response'].mean():.3f}, off={sub.loc[sub.treatment_7plus3==0,'objective_response'].mean():.3f}, Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)


# ---------------------------------------------------------------------------
# Iteration 19: Adjusted treatment effects within key subgroups (sanity check)
# ---------------------------------------------------------------------------
print("=== Iteration 19: adjusted subgroup treatment effects ===")

def adj_logit_within(df, treat, covars):
    formula = f"objective_response ~ {treat} + " + " + ".join(covars)
    res = smf.logit(formula, data=df).fit(disp=False)
    return float(res.params[treat]), float(res.pvalues[treat])

covars = [
    "age_years", "sex_female", "ecog_ps", "wbc_k_per_ul", "blast_pct_marrow",
    "albumin_g_dl", "ldh_u_l",
]

for hid, treat, mask, label in [
    ("h126", "treatment_gilteritinib", DF["flt3_itd"] == 1, "FLT3-ITD"),
    ("h127", "treatment_midostaurin", (DF["flt3_itd"] == 1) & (DF["unfit_for_intensive"] == 0), "FLT3-ITD & fit"),
    ("h128", "treatment_ivosidenib", DF["idh1_mutation"] == 1, "IDH1-mut"),
    ("h129", "treatment_enasidenib", DF["idh2_mutation"] == 1, "IDH2-mut"),
    ("h130", "treatment_venetoclax_azacitidine", DF["unfit_for_intensive"] == 1, "unfit"),
    ("h131", "treatment_7plus3", DF["unfit_for_intensive"] == 0, "fit"),
]:
    sub = DF[mask]
    coef, p = adj_logit_within(sub, treat, covars)
    record(
        19, [hid],
        f"Adjusted logistic: {treat} within {label} (n={len(sub)}): logOR={coef:.3f} (OR={np.exp(coef):.3f}), p={p:.3g}.",
        p, coef,
    )


# ---------------------------------------------------------------------------
# Iteration 20: Negative controls / no-target subgroups
# ---------------------------------------------------------------------------
print("=== Iteration 20: negative-control subgroups ===")

# h132: gilteritinib in FLT3-wt should not show clear benefit
sub = DF[DF["flt3_any"] == 0]
chi2, p, or_, rd = chi2_or(sub, "treatment_gilteritinib")
record(
    20, ["h132"],
    f"Gilteritinib in FLT3-wt (n={len(sub)}): on={sub.loc[sub.treatment_gilteritinib==1,'objective_response'].mean():.3f}, off={sub.loc[sub.treatment_gilteritinib==0,'objective_response'].mean():.3f}, Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)
# h133: midostaurin in FLT3-wt should not show benefit
sub = DF[DF["flt3_any"] == 0]
chi2, p, or_, rd = chi2_or(sub, "treatment_midostaurin")
record(
    20, ["h133"],
    f"Midostaurin in FLT3-wt (n={len(sub)}): on={sub.loc[sub.treatment_midostaurin==1,'objective_response'].mean():.3f}, off={sub.loc[sub.treatment_midostaurin==0,'objective_response'].mean():.3f}, Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)
# h134: ivosidenib in IDH1-wt
sub = DF[DF["idh1_mutation"] == 0]
chi2, p, or_, rd = chi2_or(sub, "treatment_ivosidenib")
record(
    20, ["h134"],
    f"Ivosidenib in IDH1-wt (n={len(sub)}): on={sub.loc[sub.treatment_ivosidenib==1,'objective_response'].mean():.3f}, off={sub.loc[sub.treatment_ivosidenib==0,'objective_response'].mean():.3f}, Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)
# h135: enasidenib in IDH2-wt
sub = DF[DF["idh2_mutation"] == 0]
chi2, p, or_, rd = chi2_or(sub, "treatment_enasidenib")
record(
    20, ["h135"],
    f"Enasidenib in IDH2-wt (n={len(sub)}): on={sub.loc[sub.treatment_enasidenib==1,'objective_response'].mean():.3f}, off={sub.loc[sub.treatment_enasidenib==0,'objective_response'].mean():.3f}, Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)


# ---------------------------------------------------------------------------
# Iteration 21: Key 3-way: NPM1 status modifier of intensive therapy
# ---------------------------------------------------------------------------
print("=== Iteration 21: NPM1 modifier of 7+3 ===")

sub = DF[(DF["npm1_mutation"] == 1) & (DF["unfit_for_intensive"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_7plus3")
record(
    21, ["h136"],
    f"7+3 in NPM1-mut & fit (n={len(sub)}): on={sub.loc[sub.treatment_7plus3==1,'objective_response'].mean():.3f} off={sub.loc[sub.treatment_7plus3==0,'objective_response'].mean():.3f} Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)
sub = DF[(DF["npm1_mutation"] == 0) & (DF["unfit_for_intensive"] == 0)]
chi2, p, or_, rd = chi2_or(sub, "treatment_7plus3")
record(
    21, ["h137"],
    f"7+3 in NPM1-wt & fit (n={len(sub)}): on={sub.loc[sub.treatment_7plus3==1,'objective_response'].mean():.3f} off={sub.loc[sub.treatment_7plus3==0,'objective_response'].mean():.3f} Δ={rd:+.3f} OR={or_:.3f}.",
    p, float(rd),
)


# ---------------------------------------------------------------------------
# Iteration 22: Treatment-by-treatment co-administration check
# ---------------------------------------------------------------------------
print("=== Iteration 22: treatment co-administration ===")

# Midostaurin commonly combined with 7+3 - test interaction
for hid, t1, t2 in [
    ("h138", "treatment_midostaurin", "treatment_7plus3"),
    ("h139", "treatment_gilteritinib", "treatment_7plus3"),
    ("h140", "treatment_ivosidenib", "treatment_venetoclax_azacitidine"),
]:
    coef, p, _ = run_interaction(DF, t1, t2)
    record(
        22, [hid],
        f"{t1}*{t2}: int_coef={coef:.3f} (p={p:.3g}).",
        float(p), float(coef),
    )


# ---------------------------------------------------------------------------
# Iteration 23: Multivariable model with all interactions of interest
# ---------------------------------------------------------------------------
print("=== Iteration 23: full interaction model ===")

formula_full = (
    "objective_response ~ age_years + sex_female + ecog_ps + secondary_aml + "
    "unfit_for_intensive + complex_karyotype + tp53_mutation + npm1_mutation + "
    "wbc_k_per_ul + blast_pct_marrow + albumin_g_dl + "
    "treatment_midostaurin*flt3_itd + treatment_midostaurin*flt3_tkd + "
    "treatment_gilteritinib*flt3_itd + treatment_gilteritinib*flt3_tkd + "
    "treatment_ivosidenib*idh1_mutation + treatment_enasidenib*idh2_mutation + "
    "treatment_venetoclax_azacitidine + treatment_7plus3"
)
res_full = smf.logit(formula_full, data=DF).fit(disp=False)
for hid, term in [
    ("h141", "treatment_midostaurin:flt3_itd"),
    ("h142", "treatment_gilteritinib:flt3_itd"),
    ("h143", "treatment_gilteritinib:flt3_tkd"),
    ("h144", "treatment_ivosidenib:idh1_mutation"),
    ("h145", "treatment_enasidenib:idh2_mutation"),
]:
    coef = float(res_full.params.get(term, np.nan))
    p = float(res_full.pvalues.get(term, np.nan))
    record(
        23, [hid],
        f"Full multivariable interaction term {term}: coef={coef:.3f} (OR={np.exp(coef):.3f}), p={p:.3g}.",
        p, coef,
    )


# ---------------------------------------------------------------------------
# Iteration 24: Sensitivity - lab covariates as response predictors
# ---------------------------------------------------------------------------
print("=== Iteration 24: lab covariates adjusted ===")

formula_labs = (
    "objective_response ~ age_years + sex_female + ecog_ps + albumin_g_dl + "
    "ldh_u_l + crp_mg_l + nlr + hemoglobin_g_dl + wbc_k_per_ul + "
    "blast_pct_marrow + weight_loss_pct_6mo"
)
res_lab = smf.logit(formula_labs, data=DF).fit(disp=False)
for hid, term in [
    ("h146", "albumin_g_dl"),
    ("h147", "ldh_u_l"),
    ("h148", "crp_mg_l"),
    ("h149", "nlr"),
    ("h150", "weight_loss_pct_6mo"),
    ("h151", "hemoglobin_g_dl"),
]:
    coef = float(res_lab.params.get(term, np.nan))
    p = float(res_lab.pvalues.get(term, np.nan))
    record(
        24, [hid],
        f"Adjusted logit coefficient for {term}: {coef:.4f} (OR per +1={np.exp(coef):.3f}), p={p:.3g}.",
        p, coef,
    )


# ---------------------------------------------------------------------------
# Iteration 25: Final pinned subgroup hypotheses (one per treatment) using
# the complete subgroup definitions including unfavorable suppressors.
# ---------------------------------------------------------------------------
print("=== Iteration 25: final pinned subgroup hypotheses ===")

# Final h152..h157
final_pins = [
    ("h152", "treatment_gilteritinib",
     {"flt3_itd": 1, "tp53_mutation": 0, "complex_karyotype": 0},
     "Gilteritinib increases response in FLT3-ITD+ patients without TP53 mutation or complex karyotype."),
    ("h153", "treatment_midostaurin",
     {"flt3_itd": 1, "unfit_for_intensive": 0, "tp53_mutation": 0, "complex_karyotype": 0},
     "Midostaurin increases response in FLT3-ITD+, fit, TP53-wt, non-complex-karyotype patients."),
    ("h154", "treatment_ivosidenib",
     {"idh1_mutation": 1, "tp53_mutation": 0, "complex_karyotype": 0},
     "Ivosidenib increases response in IDH1-mutated patients without TP53 mutation or complex karyotype."),
    ("h155", "treatment_enasidenib",
     {"idh2_mutation": 1, "tp53_mutation": 0, "complex_karyotype": 0},
     "Enasidenib increases response in IDH2-mutated patients without TP53 mutation or complex karyotype."),
    ("h156", "treatment_venetoclax_azacitidine",
     {"unfit_for_intensive": 1, "tp53_mutation": 0, "complex_karyotype": 0},
     "Venetoclax-azacitidine increases response in unfit patients without TP53 mutation or complex karyotype."),
    ("h157", "treatment_7plus3",
     {"unfit_for_intensive": 0, "tp53_mutation": 0, "complex_karyotype": 0},
     "7+3 increases response in fit patients without TP53 mutation or complex karyotype."),
]

for hid, treat, conds, _label in final_pins:
    mask = np.ones(len(DF), dtype=bool)
    for c, v in conds.items():
        mask &= (DF[c].values == v)
    sub = DF[mask]
    if len(sub) < 30:
        record(25, [hid], f"Subgroup too small (n={len(sub)}).", None, None, significant=False)
        continue
    chi2, p, or_, rd = chi2_or(sub, treat)
    on = sub.loc[sub[treat] == 1, "objective_response"].mean()
    off = sub.loc[sub[treat] == 0, "objective_response"].mean()
    desc = " & ".join(f"{c}={v}" for c, v in conds.items())
    record(
        25, [hid],
        f"Final pinned: {treat} within [{desc}] (n={len(sub)}, on={int(sub[treat].sum())}): on={on:.3f}, off={off:.3f}, Δ={rd:+.3f}, OR={or_:.3f}.",
        p, float(rd),
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open("analysis_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"Saved {len(RESULTS)} analysis records.")
