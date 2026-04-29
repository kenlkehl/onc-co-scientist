"""Run the iterative analysis loop for ds001_nsclc and emit transcript + summary."""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DF = pd.read_parquet(ROOT / "dataset.parquet")
N = len(DF)

ITERATIONS: list[dict[str, Any]] = []


def add_iter(index, hypotheses, analyses):
    ITERATIONS.append(
        {"index": index, "proposed_hypotheses": hypotheses, "analyses": analyses}
    )


def logit_or(df, formula):
    """Fit a logit, return coef, p, OR for the first non-intercept term."""
    res = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
    # The first non-intercept named term
    names = [n for n in res.params.index if n != "Intercept"]
    name = names[0]
    coef = float(res.params[name])
    p = float(res.pvalues[name])
    return coef, p, res


def chi2_or(df, group_col, outcome_col="objective_response"):
    tab = pd.crosstab(df[group_col], df[outcome_col])
    chi2, p, _, _ = stats.chi2_contingency(tab.values)
    # Risk difference for binary group_col
    if set(df[group_col].unique()) <= {0, 1}:
        rr1 = df.loc[df[group_col] == 1, outcome_col].mean()
        rr0 = df.loc[df[group_col] == 0, outcome_col].mean()
        return rr1 - rr0, p
    return None, p


def diff_means(df, group_col, value_col):
    a = df.loc[df[group_col] == 1, value_col]
    b = df.loc[df[group_col] == 0, value_col]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(p)


# ============================================================================
# Iteration 1: Treatment main effects on objective response
# ============================================================================
analyses = []
hyps = []
for tr in [
    "treatment_pembrolizumab",
    "treatment_sotorasib",
    "treatment_olaparib",
    "treatment_osimertinib",
]:
    hid = f"h1_{tr}"
    hyps.append(
        {
            "id": hid,
            "text": (
                f"Patients receiving {tr} have a different overall objective_response "
                f"rate than patients not receiving {tr}; specifically, response is higher on treatment."
            ),
            "kind": "novel",
        }
    )
    rd, p = chi2_or(DF, tr)
    or_, plog, _ = logit_or(DF, f"objective_response ~ {tr}")
    analyses.append(
        {
            "hypothesis_ids": [hid],
            "code": f"chi2 + logit objective_response ~ {tr}",
            "result_summary": (
                f"ORR on {tr}=1: {DF.loc[DF[tr]==1,'objective_response'].mean():.4f}; "
                f"ORR on {tr}=0: {DF.loc[DF[tr]==0,'objective_response'].mean():.4f}; "
                f"risk diff={rd:.4f}, chi2 p={p:.3g}; logit beta={or_:.3f}, p={plog:.3g}"
            ),
            "p_value": float(plog),
            "effect_estimate": float(rd),
            "significant": bool(plog < 0.05),
        }
    )
add_iter(1, hyps, analyses)


# ============================================================================
# Iteration 2: Biomarker main effects (unadjusted)
# ============================================================================
hyps = []
analyses = []
for bm in ["egfr_mutation", "kras_g12c", "alk_fusion", "tmb_high"]:
    hid = f"h2_{bm}"
    hyps.append(
        {
            "id": hid,
            "text": (
                f"Patients positive for {bm} have a different objective_response rate "
                f"than patients negative for {bm}."
            ),
            "kind": "novel",
        }
    )
    rd, p = chi2_or(DF, bm)
    analyses.append(
        {
            "hypothesis_ids": [hid],
            "code": f"chi2 objective_response ~ {bm}",
            "result_summary": (
                f"ORR {bm}=1: {DF.loc[DF[bm]==1,'objective_response'].mean():.4f}; "
                f"ORR {bm}=0: {DF.loc[DF[bm]==0,'objective_response'].mean():.4f}; "
                f"risk diff={rd:.4f}, chi2 p={p:.3g}"
            ),
            "p_value": float(p),
            "effect_estimate": float(rd),
            "significant": bool(p < 0.05),
        }
    )

# Continuous PD-L1
hid = "h2_pdl1"
hyps.append(
    {
        "id": hid,
        "text": (
            "Higher pdl1_tps (continuous) is associated with a higher objective_response rate."
        ),
        "kind": "novel",
    }
)
beta, p, _ = logit_or(DF, "objective_response ~ pdl1_tps")
analyses.append(
    {
        "hypothesis_ids": [hid],
        "code": "logit objective_response ~ pdl1_tps",
        "result_summary": f"Logit beta(pdl1_tps)={beta:.3f}, p={p:.3g}",
        "p_value": float(p),
        "effect_estimate": float(beta),
        "significant": bool(p < 0.05),
    }
)
add_iter(2, hyps, analyses)


# ============================================================================
# Iteration 3: Pembrolizumab × PD-L1 interaction
# ============================================================================
hyps = [
    {
        "id": "h3_pembro_pdl1",
        "text": (
            "The objective_response benefit of treatment_pembrolizumab is greater "
            "in patients with higher pdl1_tps; i.e., the interaction "
            "pdl1_tps x treatment_pembrolizumab has a positive coefficient."
        ),
        "kind": "novel",
    },
    {
        "id": "h3_pembro_pdl1_high",
        "text": (
            "Among patients with pdl1_tps >= 0.5, treatment_pembrolizumab is "
            "associated with a higher objective_response rate; among patients with "
            "pdl1_tps < 0.5, the effect of treatment_pembrolizumab is smaller or null."
        ),
        "kind": "refined",
    },
]
res = smf.logit(
    "objective_response ~ treatment_pembrolizumab * pdl1_tps", data=DF
).fit(disp=0, maxiter=200)
inter = "treatment_pembrolizumab:pdl1_tps"
beta_int = float(res.params[inter])
p_int = float(res.pvalues[inter])

# Stratified
high = DF[DF["pdl1_tps"] >= 0.5]
low = DF[DF["pdl1_tps"] < 0.5]
rd_high, p_high = chi2_or(high, "treatment_pembrolizumab")
rd_low, p_low = chi2_or(low, "treatment_pembrolizumab")

analyses = [
    {
        "hypothesis_ids": ["h3_pembro_pdl1"],
        "code": "logit objective_response ~ treatment_pembrolizumab * pdl1_tps",
        "result_summary": (
            f"Interaction beta(treatment_pembrolizumab:pdl1_tps)={beta_int:.3f}, p={p_int:.3g}. "
            f"Main pembro beta={float(res.params['treatment_pembrolizumab']):.3f}, "
            f"pdl1 beta={float(res.params['pdl1_tps']):.3f}."
        ),
        "p_value": float(p_int),
        "effect_estimate": float(beta_int),
        "significant": bool(p_int < 0.05),
    },
    {
        "hypothesis_ids": ["h3_pembro_pdl1_high"],
        "code": "stratified ORR by pdl1_tps>=0.5 vs <0.5",
        "result_summary": (
            f"PDL1 >=50%: ORR pembro=1 {high.loc[high['treatment_pembrolizumab']==1,'objective_response'].mean():.3f} "
            f"vs pembro=0 {high.loc[high['treatment_pembrolizumab']==0,'objective_response'].mean():.3f}, "
            f"diff={rd_high:.3f}, p={p_high:.3g}. "
            f"PDL1 <50%: pembro=1 {low.loc[low['treatment_pembrolizumab']==1,'objective_response'].mean():.3f} "
            f"vs pembro=0 {low.loc[low['treatment_pembrolizumab']==0,'objective_response'].mean():.3f}, "
            f"diff={rd_low:.3f}, p={p_low:.3g}."
        ),
        "p_value": float(p_high),
        "effect_estimate": float(rd_high),
        "significant": bool(p_high < 0.05),
    },
]
add_iter(3, hyps, analyses)


# ============================================================================
# Iteration 4: Pembrolizumab × TMB-high
# ============================================================================
hyps = [
    {
        "id": "h4_pembro_tmb",
        "text": (
            "The objective_response benefit of treatment_pembrolizumab is greater in "
            "tmb_high=1 patients; i.e., the interaction "
            "tmb_high x treatment_pembrolizumab has a positive coefficient."
        ),
        "kind": "novel",
    }
]
res = smf.logit(
    "objective_response ~ treatment_pembrolizumab * tmb_high", data=DF
).fit(disp=0, maxiter=200)
inter = "treatment_pembrolizumab:tmb_high"
beta_int = float(res.params[inter])
p_int = float(res.pvalues[inter])
rd_h, p_h = chi2_or(DF[DF["tmb_high"] == 1], "treatment_pembrolizumab")
rd_l, p_l = chi2_or(DF[DF["tmb_high"] == 0], "treatment_pembrolizumab")
analyses = [
    {
        "hypothesis_ids": ["h4_pembro_tmb"],
        "code": "logit objective_response ~ treatment_pembrolizumab * tmb_high; stratified",
        "result_summary": (
            f"Interaction beta={beta_int:.3f}, p={p_int:.3g}. "
            f"TMB-high: pembro RD={rd_h:.3f}, p={p_h:.3g}; "
            f"TMB-low: pembro RD={rd_l:.3f}, p={p_l:.3g}."
        ),
        "p_value": float(p_int),
        "effect_estimate": float(beta_int),
        "significant": bool(p_int < 0.05),
    }
]
add_iter(4, hyps, analyses)


# ============================================================================
# Iteration 5: Sotorasib × KRAS G12C
# ============================================================================
hyps = [
    {
        "id": "h5_soto_krasg12c",
        "text": (
            "The objective_response benefit of treatment_sotorasib is concentrated in "
            "kras_g12c=1 patients; i.e., the interaction "
            "kras_g12c x treatment_sotorasib has a positive coefficient."
        ),
        "kind": "novel",
    }
]
res = smf.logit(
    "objective_response ~ treatment_sotorasib * kras_g12c", data=DF
).fit(disp=0, maxiter=200)
inter = "treatment_sotorasib:kras_g12c"
beta_int = float(res.params[inter])
p_int = float(res.pvalues[inter])
rd_h, p_h = chi2_or(DF[DF["kras_g12c"] == 1], "treatment_sotorasib")
rd_l, p_l = chi2_or(DF[DF["kras_g12c"] == 0], "treatment_sotorasib")
analyses = [
    {
        "hypothesis_ids": ["h5_soto_krasg12c"],
        "code": "logit objective_response ~ treatment_sotorasib * kras_g12c; stratified",
        "result_summary": (
            f"Interaction beta={beta_int:.3f}, p={p_int:.3g}. "
            f"KRAS G12C+: soto RD={rd_h:.3f}, p={p_h:.3g} "
            f"(ORR soto=1 {DF[(DF['kras_g12c']==1)&(DF['treatment_sotorasib']==1)]['objective_response'].mean():.3f} "
            f"vs soto=0 {DF[(DF['kras_g12c']==1)&(DF['treatment_sotorasib']==0)]['objective_response'].mean():.3f}); "
            f"KRAS G12C-: soto RD={rd_l:.3f}, p={p_l:.3g}."
        ),
        "p_value": float(p_int),
        "effect_estimate": float(beta_int),
        "significant": bool(p_int < 0.05),
    }
]
add_iter(5, hyps, analyses)


# ============================================================================
# Iteration 6: Osimertinib × EGFR mutation
# ============================================================================
hyps = [
    {
        "id": "h6_osi_egfr",
        "text": (
            "The objective_response benefit of treatment_osimertinib is concentrated "
            "in egfr_mutation=1 patients; i.e., the interaction "
            "egfr_mutation x treatment_osimertinib has a positive coefficient."
        ),
        "kind": "novel",
    }
]
res = smf.logit(
    "objective_response ~ treatment_osimertinib * egfr_mutation", data=DF
).fit(disp=0, maxiter=200)
inter = "treatment_osimertinib:egfr_mutation"
beta_int = float(res.params[inter])
p_int = float(res.pvalues[inter])
rd_h, p_h = chi2_or(DF[DF["egfr_mutation"] == 1], "treatment_osimertinib")
rd_l, p_l = chi2_or(DF[DF["egfr_mutation"] == 0], "treatment_osimertinib")
analyses = [
    {
        "hypothesis_ids": ["h6_osi_egfr"],
        "code": "logit objective_response ~ treatment_osimertinib * egfr_mutation",
        "result_summary": (
            f"Interaction beta={beta_int:.3f}, p={p_int:.3g}. "
            f"EGFR+: osi RD={rd_h:.3f}, p={p_h:.3g} "
            f"(ORR osi=1 {DF[(DF['egfr_mutation']==1)&(DF['treatment_osimertinib']==1)]['objective_response'].mean():.3f} "
            f"vs osi=0 {DF[(DF['egfr_mutation']==1)&(DF['treatment_osimertinib']==0)]['objective_response'].mean():.3f}); "
            f"EGFR-: osi RD={rd_l:.3f}, p={p_l:.3g}."
        ),
        "p_value": float(p_int),
        "effect_estimate": float(beta_int),
        "significant": bool(p_int < 0.05),
    }
]
add_iter(6, hyps, analyses)


# ============================================================================
# Iteration 7: Olaparib × BRCA2
# ============================================================================
hyps = [
    {
        "id": "h7_olap_brca2",
        "text": (
            "The objective_response benefit of treatment_olaparib is concentrated in "
            "brca2_mutation=1 patients; i.e., the interaction "
            "brca2_mutation x treatment_olaparib has a positive coefficient."
        ),
        "kind": "novel",
    }
]
res = smf.logit(
    "objective_response ~ treatment_olaparib * brca2_mutation", data=DF
).fit(disp=0, maxiter=200)
inter = "treatment_olaparib:brca2_mutation"
beta_int = float(res.params[inter])
p_int = float(res.pvalues[inter])
rd_h, p_h = chi2_or(DF[DF["brca2_mutation"] == 1], "treatment_olaparib")
rd_l, p_l = chi2_or(DF[DF["brca2_mutation"] == 0], "treatment_olaparib")
analyses = [
    {
        "hypothesis_ids": ["h7_olap_brca2"],
        "code": "logit objective_response ~ treatment_olaparib * brca2_mutation",
        "result_summary": (
            f"Interaction beta={beta_int:.3f}, p={p_int:.3g}. "
            f"BRCA2+: olap RD={rd_h:.3f}, p={p_h:.3g} "
            f"(n={(DF['brca2_mutation']==1).sum()}); "
            f"BRCA2-: olap RD={rd_l:.3f}, p={p_l:.3g}."
        ),
        "p_value": float(p_int),
        "effect_estimate": float(beta_int),
        "significant": bool(p_int < 0.05),
    }
]
add_iter(7, hyps, analyses)


# ============================================================================
# Iteration 8: Pembrolizumab × STK11 (negative interaction expected)
# ============================================================================
hyps = [
    {
        "id": "h8_pembro_stk11",
        "text": (
            "The objective_response benefit of treatment_pembrolizumab is reduced in "
            "stk11_mutation=1 patients; i.e., the interaction "
            "stk11_mutation x treatment_pembrolizumab has a negative coefficient."
        ),
        "kind": "novel",
    }
]
res = smf.logit(
    "objective_response ~ treatment_pembrolizumab * stk11_mutation", data=DF
).fit(disp=0, maxiter=200)
inter = "treatment_pembrolizumab:stk11_mutation"
beta_int = float(res.params[inter])
p_int = float(res.pvalues[inter])
rd_h, p_h = chi2_or(DF[DF["stk11_mutation"] == 1], "treatment_pembrolizumab")
rd_l, p_l = chi2_or(DF[DF["stk11_mutation"] == 0], "treatment_pembrolizumab")
analyses = [
    {
        "hypothesis_ids": ["h8_pembro_stk11"],
        "code": "logit objective_response ~ treatment_pembrolizumab * stk11_mutation",
        "result_summary": (
            f"Interaction beta={beta_int:.3f}, p={p_int:.3g}. "
            f"STK11+: pembro RD={rd_h:.3f}, p={p_h:.3g}; "
            f"STK11-: pembro RD={rd_l:.3f}, p={p_l:.3g}."
        ),
        "p_value": float(p_int),
        "effect_estimate": float(beta_int),
        "significant": bool(p_int < 0.05),
    }
]
add_iter(8, hyps, analyses)


# ============================================================================
# Iteration 9: Pembrolizumab × KEAP1 (negative interaction expected)
# ============================================================================
hyps = [
    {
        "id": "h9_pembro_keap1",
        "text": (
            "The objective_response benefit of treatment_pembrolizumab is reduced in "
            "keap1_mutation=1 patients; i.e., the interaction "
            "keap1_mutation x treatment_pembrolizumab has a negative coefficient."
        ),
        "kind": "novel",
    }
]
res = smf.logit(
    "objective_response ~ treatment_pembrolizumab * keap1_mutation", data=DF
).fit(disp=0, maxiter=200)
inter = "treatment_pembrolizumab:keap1_mutation"
beta_int = float(res.params[inter])
p_int = float(res.pvalues[inter])
rd_h, p_h = chi2_or(DF[DF["keap1_mutation"] == 1], "treatment_pembrolizumab")
rd_l, p_l = chi2_or(DF[DF["keap1_mutation"] == 0], "treatment_pembrolizumab")
analyses = [
    {
        "hypothesis_ids": ["h9_pembro_keap1"],
        "code": "logit objective_response ~ treatment_pembrolizumab * keap1_mutation",
        "result_summary": (
            f"Interaction beta={beta_int:.3f}, p={p_int:.3g}. "
            f"KEAP1+: pembro RD={rd_h:.3f}, p={p_h:.3g}; "
            f"KEAP1-: pembro RD={rd_l:.3f}, p={p_l:.3g}."
        ),
        "p_value": float(p_int),
        "effect_estimate": float(beta_int),
        "significant": bool(p_int < 0.05),
    }
]
add_iter(9, hyps, analyses)


# ============================================================================
# Iteration 10: Pembrolizumab × EGFR (immunotherapy underperforms in EGFR+)
# ============================================================================
hyps = [
    {
        "id": "h10_pembro_egfr",
        "text": (
            "treatment_pembrolizumab provides less objective_response benefit in "
            "egfr_mutation=1 patients than in egfr_mutation=0 patients; i.e., the "
            "interaction egfr_mutation x treatment_pembrolizumab has a negative coefficient."
        ),
        "kind": "novel",
    }
]
res = smf.logit(
    "objective_response ~ treatment_pembrolizumab * egfr_mutation", data=DF
).fit(disp=0, maxiter=200)
inter = "treatment_pembrolizumab:egfr_mutation"
beta_int = float(res.params[inter])
p_int = float(res.pvalues[inter])
rd_h, p_h = chi2_or(DF[DF["egfr_mutation"] == 1], "treatment_pembrolizumab")
rd_l, p_l = chi2_or(DF[DF["egfr_mutation"] == 0], "treatment_pembrolizumab")
analyses = [
    {
        "hypothesis_ids": ["h10_pembro_egfr"],
        "code": "logit objective_response ~ treatment_pembrolizumab * egfr_mutation",
        "result_summary": (
            f"Interaction beta={beta_int:.3f}, p={p_int:.3g}. "
            f"EGFR+: pembro RD={rd_h:.3f}, p={p_h:.3g}; "
            f"EGFR-: pembro RD={rd_l:.3f}, p={p_l:.3g}."
        ),
        "p_value": float(p_int),
        "effect_estimate": float(beta_int),
        "significant": bool(p_int < 0.05),
    }
]
add_iter(10, hyps, analyses)


# ============================================================================
# Iteration 11: Performance status (ECOG) main effect
# ============================================================================
hyps = [
    {
        "id": "h11_ecog",
        "text": (
            "Higher ecog_ps (worse performance status) is associated with a lower "
            "objective_response rate."
        ),
        "kind": "novel",
    }
]
beta, p, _ = logit_or(DF, "objective_response ~ ecog_ps")
analyses = [
    {
        "hypothesis_ids": ["h11_ecog"],
        "code": "logit objective_response ~ ecog_ps",
        "result_summary": (
            f"Logit beta(ecog_ps)={beta:.3f}, p={p:.3g}. "
            f"ORR by ecog_ps: "
            + ", ".join(
                f"{int(k)}={v:.3f}"
                for k, v in DF.groupby("ecog_ps")["objective_response"].mean().items()
            )
        ),
        "p_value": float(p),
        "effect_estimate": float(beta),
        "significant": bool(p < 0.05),
    }
]
add_iter(11, hyps, analyses)


# ============================================================================
# Iteration 12: Stage IV, brain mets, liver mets, bone mets
# ============================================================================
hyps = []
analyses = []
for col in ["stage_iv", "has_brain_mets", "liver_mets", "bone_mets", "adrenal_mets"]:
    hid = f"h12_{col}"
    hyps.append(
        {
            "id": hid,
            "text": f"Patients with {col}=1 have a lower objective_response rate than patients with {col}=0.",
            "kind": "novel",
        }
    )
    rd, p = chi2_or(DF, col)
    analyses.append(
        {
            "hypothesis_ids": [hid],
            "code": f"chi2 objective_response ~ {col}",
            "result_summary": (
                f"ORR {col}=1: {DF.loc[DF[col]==1,'objective_response'].mean():.4f}; "
                f"ORR {col}=0: {DF.loc[DF[col]==0,'objective_response'].mean():.4f}; "
                f"risk diff={rd:.4f}, chi2 p={p:.3g}"
            ),
            "p_value": float(p),
            "effect_estimate": float(rd),
            "significant": bool(p < 0.05),
        }
    )
add_iter(12, hyps, analyses)


# ============================================================================
# Iteration 13: Lab markers — albumin, LDH, NLR, CRP, hemoglobin
# ============================================================================
hyps = []
analyses = []
for col, direction in [
    ("albumin_g_dl", "higher → higher response (better prognosis)"),
    ("ldh_u_l", "higher → lower response"),
    ("nlr", "higher → lower response"),
    ("crp_mg_l", "higher → lower response"),
    ("hemoglobin_g_dl", "higher → higher response"),
]:
    hid = f"h13_{col}"
    hyps.append(
        {
            "id": hid,
            "text": f"{col} as a continuous predictor is associated with objective_response ({direction}).",
            "kind": "novel",
        }
    )
    beta, p, _ = logit_or(DF, f"objective_response ~ {col}")
    analyses.append(
        {
            "hypothesis_ids": [hid],
            "code": f"logit objective_response ~ {col}",
            "result_summary": f"Logit beta({col})={beta:.4f}, p={p:.3g}",
            "p_value": float(p),
            "effect_estimate": float(beta),
            "significant": bool(p < 0.05),
        }
    )
add_iter(13, hyps, analyses)


# ============================================================================
# Iteration 14: Demographics — age, sex, smoking
# ============================================================================
hyps = [
    {
        "id": "h14_age",
        "text": "Older age_years is associated with lower objective_response.",
        "kind": "novel",
    },
    {
        "id": "h14_sex",
        "text": "Female patients (sex_female=1) have a different objective_response rate than male patients.",
        "kind": "novel",
    },
    {
        "id": "h14_smoking",
        "text": (
            "Patients with smoking_status='never' have a different objective_response rate "
            "than current/former smokers (and may be enriched for EGFR-driven disease)."
        ),
        "kind": "novel",
    },
]
analyses = []
beta, p, _ = logit_or(DF, "objective_response ~ age_years")
analyses.append(
    {
        "hypothesis_ids": ["h14_age"],
        "code": "logit objective_response ~ age_years",
        "result_summary": f"Logit beta(age_years)={beta:.4f}, p={p:.3g}",
        "p_value": float(p),
        "effect_estimate": float(beta),
        "significant": bool(p < 0.05),
    }
)
rd_sex, p_sex = chi2_or(DF, "sex_female")
analyses.append(
    {
        "hypothesis_ids": ["h14_sex"],
        "code": "chi2 objective_response ~ sex_female",
        "result_summary": (
            f"ORR female: {DF.loc[DF['sex_female']==1,'objective_response'].mean():.4f}; "
            f"male: {DF.loc[DF['sex_female']==0,'objective_response'].mean():.4f}; "
            f"risk diff={rd_sex:.4f}, p={p_sex:.3g}"
        ),
        "p_value": float(p_sex),
        "effect_estimate": float(rd_sex),
        "significant": bool(p_sex < 0.05),
    }
)
DF["_never"] = (DF["smoking_status"] == "never").astype(int)
rd_n, p_n = chi2_or(DF, "_never")
analyses.append(
    {
        "hypothesis_ids": ["h14_smoking"],
        "code": "chi2 objective_response ~ (smoking_status==never)",
        "result_summary": (
            f"ORR never: {DF.loc[DF['_never']==1,'objective_response'].mean():.4f}; "
            f"ever: {DF.loc[DF['_never']==0,'objective_response'].mean():.4f}; "
            f"risk diff={rd_n:.4f}, p={p_n:.3g}"
        ),
        "p_value": float(p_n),
        "effect_estimate": float(rd_n),
        "significant": bool(p_n < 0.05),
    }
)
add_iter(14, hyps, analyses)


# ============================================================================
# Iteration 15: Histology and squamous-specific patterns
# ============================================================================
hyps = [
    {
        "id": "h15_squamous",
        "text": (
            "Patients with histology='squamous' have a different objective_response rate "
            "than 'adenocarcinoma' patients."
        ),
        "kind": "novel",
    },
    {
        "id": "h15_squamous_pembro",
        "text": (
            "Among patients with histology='squamous', treatment_pembrolizumab effect "
            "on objective_response differs from the effect among adenocarcinoma patients."
        ),
        "kind": "refined",
    },
]
DF["_sq"] = (DF["histology"] == "squamous").astype(int)
rd, p = chi2_or(DF, "_sq")
res = smf.logit(
    "objective_response ~ treatment_pembrolizumab * _sq", data=DF
).fit(disp=0, maxiter=200)
inter = "treatment_pembrolizumab:_sq"
beta_int = float(res.params[inter])
p_int = float(res.pvalues[inter])
analyses = [
    {
        "hypothesis_ids": ["h15_squamous"],
        "code": "chi2 objective_response ~ histology(squamous=1)",
        "result_summary": (
            f"ORR squamous: {DF.loc[DF['_sq']==1,'objective_response'].mean():.4f}; "
            f"adeno: {DF.loc[DF['_sq']==0,'objective_response'].mean():.4f}; "
            f"diff={rd:.4f}, p={p:.3g}"
        ),
        "p_value": float(p),
        "effect_estimate": float(rd),
        "significant": bool(p < 0.05),
    },
    {
        "hypothesis_ids": ["h15_squamous_pembro"],
        "code": "logit objective_response ~ treatment_pembrolizumab * histology_squamous",
        "result_summary": f"Interaction beta(pembro:squamous)={beta_int:.3f}, p={p_int:.3g}",
        "p_value": float(p_int),
        "effect_estimate": float(beta_int),
        "significant": bool(p_int < 0.05),
    },
]
add_iter(15, hyps, analyses)


# ============================================================================
# Iteration 16: Other targetable alterations (rare but high effect)
# ============================================================================
hyps = []
analyses = []
for col in [
    "alk_fusion",
    "ros1_fusion",
    "ret_fusion",
    "braf_v600e",
    "met_exon14_skipping",
    "ntrk_fusion",
    "her2_amplification",
    "nrg1_fusion",
]:
    hid = f"h16_{col}"
    hyps.append(
        {
            "id": hid,
            "text": (
                f"{col}=1 is associated with a different objective_response rate than {col}=0 "
                f"(targetable oncogenic driver may correlate with response patterns)."
            ),
            "kind": "novel",
        }
    )
    rd, p = chi2_or(DF, col)
    analyses.append(
        {
            "hypothesis_ids": [hid],
            "code": f"chi2 objective_response ~ {col}",
            "result_summary": (
                f"ORR {col}=1: {DF.loc[DF[col]==1,'objective_response'].mean():.4f} "
                f"(n={(DF[col]==1).sum()}); "
                f"{col}=0: {DF.loc[DF[col]==0,'objective_response'].mean():.4f}; "
                f"diff={rd:.4f}, p={p:.3g}"
            ),
            "p_value": float(p),
            "effect_estimate": float(rd),
            "significant": bool(p < 0.05),
        }
    )
add_iter(16, hyps, analyses)


# ============================================================================
# Iteration 17: Symptom grades — fatigue, pain, dyspnea, cough, appetite
# ============================================================================
hyps = []
analyses = []
for col in ["fatigue_grade", "pain_nrs", "dyspnea_grade", "cough_grade", "appetite_loss_grade"]:
    hid = f"h17_{col}"
    hyps.append(
        {
            "id": hid,
            "text": f"Higher {col} is associated with a lower objective_response rate (worse symptom burden, worse outcomes).",
            "kind": "novel",
        }
    )
    beta, p, _ = logit_or(DF, f"objective_response ~ {col}")
    analyses.append(
        {
            "hypothesis_ids": [hid],
            "code": f"logit objective_response ~ {col}",
            "result_summary": f"Logit beta({col})={beta:.4f}, p={p:.3g}",
            "p_value": float(p),
            "effect_estimate": float(beta),
            "significant": bool(p < 0.05),
        }
    )
add_iter(17, hyps, analyses)


# ============================================================================
# Iteration 18: Socioeconomic — insurance, rural, education
# ============================================================================
hyps = [
    {
        "id": "h18_uninsured",
        "text": (
            "Uninsured patients (insurance_type='uninsured') have a different "
            "objective_response rate than insured patients."
        ),
        "kind": "novel",
    },
    {
        "id": "h18_rural",
        "text": "Rural residence is associated with a different objective_response rate.",
        "kind": "novel",
    },
    {
        "id": "h18_edu",
        "text": "Higher education_years is associated with higher objective_response rate.",
        "kind": "novel",
    },
]
DF["_uninsured"] = (DF["insurance_type"] == "uninsured").astype(int)
rd_u, p_u = chi2_or(DF, "_uninsured")
rd_r, p_r = chi2_or(DF, "rural_residence")
beta_e, p_e, _ = logit_or(DF, "objective_response ~ education_years")
analyses = [
    {
        "hypothesis_ids": ["h18_uninsured"],
        "code": "chi2 objective_response ~ uninsured",
        "result_summary": f"ORR uninsured RD={rd_u:.4f}, p={p_u:.3g}",
        "p_value": float(p_u),
        "effect_estimate": float(rd_u),
        "significant": bool(p_u < 0.05),
    },
    {
        "hypothesis_ids": ["h18_rural"],
        "code": "chi2 objective_response ~ rural_residence",
        "result_summary": f"ORR rural RD={rd_r:.4f}, p={p_r:.3g}",
        "p_value": float(p_r),
        "effect_estimate": float(rd_r),
        "significant": bool(p_r < 0.05),
    },
    {
        "hypothesis_ids": ["h18_edu"],
        "code": "logit objective_response ~ education_years",
        "result_summary": f"Logit beta(education_years)={beta_e:.4f}, p={p_e:.3g}",
        "p_value": float(p_e),
        "effect_estimate": float(beta_e),
        "significant": bool(p_e < 0.05),
    },
]
add_iter(18, hyps, analyses)


# ============================================================================
# Iteration 19: Race/ethnicity heterogeneity
# ============================================================================
hyps = [
    {
        "id": "h19_race_overall",
        "text": (
            "Objective_response rates differ across race_ethnicity categories "
            "(omnibus chi-square test of independence)."
        ),
        "kind": "novel",
    },
    {
        "id": "h19_asian_egfr",
        "text": (
            "race_ethnicity='asian' patients have a higher prevalence of egfr_mutation "
            "than non-asian patients."
        ),
        "kind": "novel",
    },
]
tab = pd.crosstab(DF["race_ethnicity"], DF["objective_response"])
chi2, p_race, _, _ = stats.chi2_contingency(tab.values)
DF["_asian"] = (DF["race_ethnicity"] == "asian").astype(int)
rd_a, p_a = chi2_or(DF, "_asian", outcome_col="egfr_mutation")
analyses = [
    {
        "hypothesis_ids": ["h19_race_overall"],
        "code": "chi2 objective_response ~ race_ethnicity (omnibus)",
        "result_summary": (
            f"Omnibus chi2={chi2:.2f}, p={p_race:.3g}. ORR by race: "
            + ", ".join(
                f"{k}={v:.3f}"
                for k, v in DF.groupby("race_ethnicity")["objective_response"].mean().items()
            )
        ),
        "p_value": float(p_race),
        "effect_estimate": float(chi2),
        "significant": bool(p_race < 0.05),
    },
    {
        "hypothesis_ids": ["h19_asian_egfr"],
        "code": "chi2 egfr_mutation ~ asian",
        "result_summary": (
            f"EGFR prevalence asian: {DF.loc[DF['_asian']==1,'egfr_mutation'].mean():.3f}; "
            f"non-asian: {DF.loc[DF['_asian']==0,'egfr_mutation'].mean():.3f}; "
            f"diff={rd_a:.3f}, p={p_a:.3g}"
        ),
        "p_value": float(p_a),
        "effect_estimate": float(rd_a),
        "significant": bool(p_a < 0.05),
    },
]
add_iter(19, hyps, analyses)


# ============================================================================
# Iteration 20: SNP panel screen (likely all null)
# ============================================================================
SNP_COLS = [c for c in DF.columns if c.startswith("snp_")]
hyps = [
    {
        "id": "h20_snp_panel",
        "text": (
            "At least one of the genotyped SNPs (snp_*) shows a Bonferroni-significant "
            "association with objective_response."
        ),
        "kind": "novel",
    }
]
analyses = []
sig_snps = []
min_p = 1.0
min_snp = None
min_beta = 0.0
for snp in SNP_COLS:
    try:
        beta, p, _ = logit_or(DF, f"objective_response ~ {snp}")
        if p < min_p:
            min_p = p
            min_snp = snp
            min_beta = beta
        if p < 0.05 / len(SNP_COLS):
            sig_snps.append((snp, beta, p))
    except Exception:
        continue
analyses.append(
    {
        "hypothesis_ids": ["h20_snp_panel"],
        "code": f"logit objective_response ~ each of {len(SNP_COLS)} SNPs; Bonferroni alpha={0.05/len(SNP_COLS):.3g}",
        "result_summary": (
            f"Tested {len(SNP_COLS)} SNPs. Bonferroni-significant: "
            f"{len(sig_snps)} ({sig_snps if sig_snps else 'none'}). "
            f"Smallest unadjusted p among SNPs: {min_snp} beta={min_beta:.3f}, p={min_p:.3g}."
        ),
        "p_value": float(min_p),
        "effect_estimate": float(min_beta),
        "significant": bool(len(sig_snps) > 0),
    }
)
add_iter(20, hyps, analyses)


# ============================================================================
# Iteration 21: Comorbidities — autoimmune disease, ILD, CKD, HF
# ============================================================================
hyps = []
analyses = []
for col in [
    "autoimmune_disease",
    "interstitial_lung_disease_history",
    "chronic_kidney_disease",
    "heart_failure",
    "copd",
    "hiv_positive",
]:
    hid = f"h21_{col}"
    hyps.append(
        {
            "id": hid,
            "text": f"{col}=1 is associated with a different objective_response rate than {col}=0.",
            "kind": "novel",
        }
    )
    rd, p = chi2_or(DF, col)
    analyses.append(
        {
            "hypothesis_ids": [hid],
            "code": f"chi2 objective_response ~ {col}",
            "result_summary": (
                f"ORR {col}=1: {DF.loc[DF[col]==1,'objective_response'].mean():.4f} "
                f"(n={(DF[col]==1).sum()}); "
                f"{col}=0: {DF.loc[DF[col]==0,'objective_response'].mean():.4f}; "
                f"diff={rd:.4f}, p={p:.3g}"
            ),
            "p_value": float(p),
            "effect_estimate": float(rd),
            "significant": bool(p < 0.05),
        }
    )
add_iter(21, hyps, analyses)


# ============================================================================
# Iteration 22: Prior treatment / lines of therapy
# ============================================================================
hyps = [
    {
        "id": "h22_prior_lines",
        "text": "Greater prior_lines_of_therapy is associated with a lower objective_response rate.",
        "kind": "novel",
    },
    {
        "id": "h22_prior_io",
        "text": "Patients with prior_immunotherapy=1 have a lower objective_response rate than those without.",
        "kind": "novel",
    },
    {
        "id": "h22_prior_chemo",
        "text": "Patients with prior_chemotherapy=1 have a lower objective_response rate than those without.",
        "kind": "novel",
    },
]
beta_l, p_l, _ = logit_or(DF, "objective_response ~ prior_lines_of_therapy")
rd_io, p_io = chi2_or(DF, "prior_immunotherapy")
rd_ch, p_ch = chi2_or(DF, "prior_chemotherapy")
analyses = [
    {
        "hypothesis_ids": ["h22_prior_lines"],
        "code": "logit objective_response ~ prior_lines_of_therapy",
        "result_summary": f"Logit beta(prior_lines_of_therapy)={beta_l:.4f}, p={p_l:.3g}",
        "p_value": float(p_l),
        "effect_estimate": float(beta_l),
        "significant": bool(p_l < 0.05),
    },
    {
        "hypothesis_ids": ["h22_prior_io"],
        "code": "chi2 objective_response ~ prior_immunotherapy",
        "result_summary": f"RD prior_immunotherapy={rd_io:.4f}, p={p_io:.3g}",
        "p_value": float(p_io),
        "effect_estimate": float(rd_io),
        "significant": bool(p_io < 0.05),
    },
    {
        "hypothesis_ids": ["h22_prior_chemo"],
        "code": "chi2 objective_response ~ prior_chemotherapy",
        "result_summary": f"RD prior_chemotherapy={rd_ch:.4f}, p={p_ch:.3g}",
        "p_value": float(p_ch),
        "effect_estimate": float(rd_ch),
        "significant": bool(p_ch < 0.05),
    },
]
add_iter(22, hyps, analyses)


# ============================================================================
# Iteration 23: Multivariable model — treatment-biomarker interactions adjusting
# for ECOG, stage, age
# ============================================================================
hyps = [
    {
        "id": "h23_mv_model",
        "text": (
            "After adjusting for ecog_ps, stage_iv, age_years and albumin_g_dl, the four "
            "treatment-biomarker interactions remain significant: "
            "treatment_pembrolizumab x pdl1_tps (positive), treatment_sotorasib x kras_g12c "
            "(positive), treatment_osimertinib x egfr_mutation (positive), and "
            "treatment_olaparib x brca2_mutation (positive)."
        ),
        "kind": "refined",
    }
]
formula = (
    "objective_response ~ treatment_pembrolizumab * pdl1_tps "
    "+ treatment_sotorasib * kras_g12c "
    "+ treatment_osimertinib * egfr_mutation "
    "+ treatment_olaparib * brca2_mutation "
    "+ ecog_ps + stage_iv + age_years + albumin_g_dl + ldh_u_l + nlr"
)
res = smf.logit(formula, data=DF).fit(disp=0, maxiter=200)
key = {
    "treatment_pembrolizumab:pdl1_tps": "pembro x pdl1",
    "treatment_sotorasib:kras_g12c": "sotorasib x KRAS G12C",
    "treatment_osimertinib:egfr_mutation": "osimertinib x EGFR",
    "treatment_olaparib:brca2_mutation": "olaparib x BRCA2",
}
parts = []
worst_p = 0.0
worst_beta = 0.0
for term, label in key.items():
    b = float(res.params[term])
    p = float(res.pvalues[term])
    parts.append(f"{label}: beta={b:.3f}, p={p:.3g}")
    if p > worst_p:
        worst_p = p
        worst_beta = b
analyses = [
    {
        "hypothesis_ids": ["h23_mv_model"],
        "code": formula,
        "result_summary": (
            "Multivariable logit with adjustment. Key interactions: "
            + "; ".join(parts)
            + f". Pseudo-R2={res.prsquared:.4f}, n={int(res.nobs)}."
        ),
        "p_value": float(worst_p),
        "effect_estimate": float(worst_beta),
        "significant": bool(worst_p < 0.05),
    }
]
add_iter(23, hyps, analyses)


# ============================================================================
# Iteration 24: Pembrolizumab subgroup — high PDL1 AND TMB high (synergy)
# ============================================================================
hyps = [
    {
        "id": "h24_pdl1_tmb_combo",
        "text": (
            "The pembrolizumab response benefit is largest in the joint subgroup "
            "pdl1_tps>=0.5 AND tmb_high=1."
        ),
        "kind": "refined",
    }
]
mask = (DF["pdl1_tps"] >= 0.5) & (DF["tmb_high"] == 1)
sub = DF[mask]
neither = DF[~mask]
rd_sub, p_sub = chi2_or(sub, "treatment_pembrolizumab")
rd_else, p_else = chi2_or(neither, "treatment_pembrolizumab")
# Three-way interaction
res = smf.logit(
    "objective_response ~ treatment_pembrolizumab * pdl1_tps * tmb_high", data=DF
).fit(disp=0, maxiter=200)
three = "treatment_pembrolizumab:pdl1_tps:tmb_high"
beta3 = float(res.params.get(three, np.nan))
p3 = float(res.pvalues.get(three, np.nan))
analyses = [
    {
        "hypothesis_ids": ["h24_pdl1_tmb_combo"],
        "code": "logit ... * pdl1_tps * tmb_high; subgroup ORR",
        "result_summary": (
            f"PDL1>=50% AND TMB-high (n={len(sub)}): pembro RD={rd_sub:.3f}, p={p_sub:.3g}. "
            f"Other patients (n={len(neither)}): pembro RD={rd_else:.3f}, p={p_else:.3g}. "
            f"3-way interaction beta={beta3:.3f}, p={p3:.3g}."
        ),
        "p_value": float(p_sub),
        "effect_estimate": float(rd_sub),
        "significant": bool(p_sub < 0.05),
    }
]
add_iter(24, hyps, analyses)


# ============================================================================
# Iteration 25: Final integrated model — match-treatment-to-biomarker indicator
# ============================================================================
hyps = [
    {
        "id": "h25_match",
        "text": (
            "A composite indicator 'biomarker_matched_treatment' (any of: "
            "treatment_osimertinib & egfr_mutation; treatment_sotorasib & kras_g12c; "
            "treatment_olaparib & brca2_mutation; treatment_pembrolizumab & pdl1_tps>=0.5) "
            "is positively associated with objective_response above and beyond the "
            "individual treatment and biomarker main effects, after multivariable adjustment."
        ),
        "kind": "refined",
    }
]
DF["match_osi"] = ((DF["treatment_osimertinib"] == 1) & (DF["egfr_mutation"] == 1)).astype(int)
DF["match_soto"] = ((DF["treatment_sotorasib"] == 1) & (DF["kras_g12c"] == 1)).astype(int)
DF["match_olap"] = ((DF["treatment_olaparib"] == 1) & (DF["brca2_mutation"] == 1)).astype(int)
DF["match_pembro"] = ((DF["treatment_pembrolizumab"] == 1) & (DF["pdl1_tps"] >= 0.5)).astype(int)
DF["any_match"] = (
    (DF["match_osi"] + DF["match_soto"] + DF["match_olap"] + DF["match_pembro"]) > 0
).astype(int)

rd_m, p_m = chi2_or(DF, "any_match")

formula_full = (
    "objective_response ~ any_match "
    "+ treatment_pembrolizumab + treatment_sotorasib + treatment_osimertinib + treatment_olaparib "
    "+ pdl1_tps + tmb_high + egfr_mutation + kras_g12c + brca2_mutation + stk11_mutation + keap1_mutation "
    "+ ecog_ps + stage_iv + has_brain_mets + liver_mets + albumin_g_dl + ldh_u_l + nlr "
    "+ age_years + C(smoking_status) + C(histology)"
)
res_full = smf.logit(formula_full, data=DF).fit(disp=0, maxiter=300)
beta_match = float(res_full.params["any_match"])
p_match = float(res_full.pvalues["any_match"])

analyses = [
    {
        "hypothesis_ids": ["h25_match"],
        "code": "Composite any_match indicator; multivariable logit with full covariates",
        "result_summary": (
            f"Univariate any_match RD={rd_m:.3f}, p={p_m:.3g} "
            f"(ORR matched: {DF.loc[DF['any_match']==1,'objective_response'].mean():.3f} "
            f"vs unmatched: {DF.loc[DF['any_match']==0,'objective_response'].mean():.3f}). "
            f"Adjusted logit beta(any_match)={beta_match:.3f}, p={p_match:.3g}, "
            f"pseudo-R2={res_full.prsquared:.4f}, n={int(res_full.nobs)}."
        ),
        "p_value": float(p_match),
        "effect_estimate": float(beta_match),
        "significant": bool(p_match < 0.05),
    }
]
add_iter(25, hyps, analyses)


# ============================================================================
# Write transcript
# ============================================================================
transcript = {
    "dataset_id": "ds001_nsclc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-direct@1",
    "max_iterations": 25,
    "iterations": ITERATIONS,
}
with open(ROOT / "transcript.json", "w", encoding="utf-8") as f:
    json.dump(transcript, f, indent=2)

print("Transcript written.")
print(f"Iterations: {len(ITERATIONS)}")
total_h = sum(len(it["proposed_hypotheses"]) for it in ITERATIONS)
total_a = sum(len(it["analyses"]) for it in ITERATIONS)
print(f"Hypotheses: {total_h}, Analyses: {total_a}")
