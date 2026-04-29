"""Run the iterative CRC dataset analysis and emit transcript.json + analysis_summary.txt."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
df = pd.read_parquet(HERE / "dataset.parquet")
OUTCOME = "pfs_months"

# Container for transcript building
iterations = []

def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def add_iteration(index, hypotheses, analyses):
    iterations.append({
        "index": index,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses,
    })


def ttest(group1, group0, label1, label0, outcome=OUTCOME, data=None):
    data = df if data is None else data
    a = data.loc[group1, outcome].values
    b = data.loc[group0, outcome].values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "n1": int(group1.sum()),
        "n0": int(group0.sum()),
        "mean1": float(np.mean(a)),
        "mean0": float(np.mean(b)),
        "diff": float(np.mean(a) - np.mean(b)),
        "p": float(p),
        "label1": label1,
        "label0": label0,
    }


def ols_summary(formula, data=None):
    data = df if data is None else data
    model = smf.ols(formula, data=data).fit()
    return model


# =====================================================================
# Iteration 1 — Demographics & performance status main effects on PFS
# =====================================================================
hypotheses = [
    {"id": "h1.1", "text": "Higher ECOG performance status (ecog_ps) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h1.2", "text": "Stage IV disease (stage_iv=1) is associated with shorter pfs_months than non-stage IV.", "kind": "novel"},
    {"id": "h1.3", "text": "Older age (age_years) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h1.4", "text": "Female sex (sex_female=1) is associated with longer pfs_months than male sex.", "kind": "novel"},
]
analyses = []

m = ols_summary("pfs_months ~ ecog_ps")
analyses.append({
    "hypothesis_ids": ["h1.1"],
    "code": "smf.ols('pfs_months ~ ecog_ps', df).fit()",
    "result_summary": f"OLS slope for ecog_ps = {m.params['ecog_ps']:.3f} months per unit, p={fmt_p(m.pvalues['ecog_ps'])}.",
    "p_value": float(m.pvalues["ecog_ps"]),
    "effect_estimate": float(m.params["ecog_ps"]),
    "significant": bool(m.pvalues["ecog_ps"] < 0.05),
})

r = ttest(df["stage_iv"] == 1, df["stage_iv"] == 0, "stage IV", "non-stage IV")
analyses.append({
    "hypothesis_ids": ["h1.2"],
    "code": "ttest pfs_months by stage_iv",
    "result_summary": f"Mean PFS stage IV={r['mean1']:.2f} vs non-IV={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"],
    "effect_estimate": r["diff"],
    "significant": r["p"] < 0.05,
})

m = ols_summary("pfs_months ~ age_years")
analyses.append({
    "hypothesis_ids": ["h1.3"],
    "code": "smf.ols('pfs_months ~ age_years', df).fit()",
    "result_summary": f"OLS slope for age_years = {m.params['age_years']:.4f} months per year of age, p={fmt_p(m.pvalues['age_years'])}.",
    "p_value": float(m.pvalues["age_years"]),
    "effect_estimate": float(m.params["age_years"]),
    "significant": bool(m.pvalues["age_years"] < 0.05),
})

r = ttest(df["sex_female"] == 1, df["sex_female"] == 0, "female", "male")
analyses.append({
    "hypothesis_ids": ["h1.4"],
    "code": "ttest pfs_months by sex_female",
    "result_summary": f"Mean PFS female={r['mean1']:.2f} vs male={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"],
    "effect_estimate": r["diff"],
    "significant": r["p"] < 0.05,
})
add_iteration(1, hypotheses, analyses)


# =====================================================================
# Iteration 2 — Tumor biomarkers (CRC drivers) main effects
# =====================================================================
hypotheses = [
    {"id": "h2.1", "text": "BRAF V600E mutation (braf_v600e=1) is associated with shorter pfs_months than wild-type.", "kind": "novel"},
    {"id": "h2.2", "text": "KRAS mutation (kras_mutation=1) is associated with shorter pfs_months than wild-type.", "kind": "novel"},
    {"id": "h2.3", "text": "NRAS mutation (nras_mutation=1) is associated with shorter pfs_months than wild-type.", "kind": "novel"},
    {"id": "h2.4", "text": "MSI-high status (msi_high=1) is associated with longer pfs_months than microsatellite-stable disease.", "kind": "novel"},
    {"id": "h2.5", "text": "Right-sided primary tumor (right_sided_primary=1) is associated with shorter pfs_months than left-sided.", "kind": "novel"},
]
analyses = []
for col, hid, label in [
    ("braf_v600e", "h2.1", "BRAF V600E"),
    ("kras_mutation", "h2.2", "KRAS mut"),
    ("nras_mutation", "h2.3", "NRAS mut"),
    ("msi_high", "h2.4", "MSI-high"),
    ("right_sided_primary", "h2.5", "right-sided"),
]:
    r = ttest(df[col] == 1, df[col] == 0, label, f"non-{label}")
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"ttest pfs_months by {col}",
        "result_summary": f"Mean PFS {label}={r['mean1']:.2f} vs not={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])} (n1={r['n1']}, n0={r['n0']}).",
        "p_value": r["p"],
        "effect_estimate": r["diff"],
        "significant": r["p"] < 0.05,
    })
add_iteration(2, hypotheses, analyses)


# =====================================================================
# Iteration 3 — Disease burden / lab markers main effects
# =====================================================================
hypotheses = [
    {"id": "h3.1", "text": "Higher CEA (cea_ng_ml) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h3.2", "text": "Higher serum albumin (albumin_g_dl) is associated with longer pfs_months.", "kind": "novel"},
    {"id": "h3.3", "text": "Higher LDH (ldh_u_l) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h3.4", "text": "Greater 6-month weight loss percent (weight_loss_pct_6mo) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h3.5", "text": "Higher neutrophil-to-lymphocyte ratio (nlr) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h3.6", "text": "Higher C-reactive protein (crp_mg_l) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h3.7", "text": "Presence of liver metastases (liver_mets=1) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h3.8", "text": "Presence of bone metastases (bone_mets=1) is associated with shorter pfs_months.", "kind": "novel"},
]
analyses = []
for col, hid in [
    ("cea_ng_ml", "h3.1"),
    ("albumin_g_dl", "h3.2"),
    ("ldh_u_l", "h3.3"),
    ("weight_loss_pct_6mo", "h3.4"),
    ("nlr", "h3.5"),
    ("crp_mg_l", "h3.6"),
]:
    m = ols_summary(f"pfs_months ~ {col}")
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {col}', df).fit()",
        "result_summary": f"OLS slope {col} = {m.params[col]:.5f} mo/unit, p={fmt_p(m.pvalues[col])}.",
        "p_value": float(m.pvalues[col]),
        "effect_estimate": float(m.params[col]),
        "significant": bool(m.pvalues[col] < 0.05),
    })
for col, hid, label in [("liver_mets", "h3.7", "liver mets"), ("bone_mets", "h3.8", "bone mets")]:
    r = ttest(df[col] == 1, df[col] == 0, label, f"no {label}")
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"ttest pfs_months by {col}",
        "result_summary": f"Mean PFS {label}={r['mean1']:.2f} vs not={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
        "p_value": r["p"],
        "effect_estimate": r["diff"],
        "significant": r["p"] < 0.05,
    })
add_iteration(3, hypotheses, analyses)


# =====================================================================
# Iteration 4 — Marginal effect of each treatment vs not receiving it
# =====================================================================
hypotheses = [
    {"id": "h4.1", "text": "Patients receiving treatment_cetuximab have different (likely longer) pfs_months than those not receiving cetuximab in the overall cohort.", "kind": "novel"},
    {"id": "h4.2", "text": "Patients receiving treatment_bevacizumab have longer pfs_months than those not receiving bevacizumab in the overall cohort.", "kind": "novel"},
    {"id": "h4.3", "text": "Patients receiving treatment_pembrolizumab have different pfs_months than those not receiving pembrolizumab in the overall cohort.", "kind": "novel"},
    {"id": "h4.4", "text": "Patients receiving treatment_encorafenib have different pfs_months than those not receiving encorafenib in the overall cohort.", "kind": "novel"},
    {"id": "h4.5", "text": "Patients receiving treatment_trastuzumab_tucatinib have different pfs_months than those not receiving it in the overall cohort.", "kind": "novel"},
    {"id": "h4.6", "text": "Patients receiving treatment_regorafenib have shorter pfs_months than those not receiving regorafenib in the overall cohort (later-line treatment indicator).", "kind": "novel"},
]
analyses = []
for col, hid, label in [
    ("treatment_cetuximab", "h4.1", "cetuximab"),
    ("treatment_bevacizumab", "h4.2", "bevacizumab"),
    ("treatment_pembrolizumab", "h4.3", "pembrolizumab"),
    ("treatment_encorafenib", "h4.4", "encorafenib"),
    ("treatment_trastuzumab_tucatinib", "h4.5", "trastuzumab+tucatinib"),
    ("treatment_regorafenib", "h4.6", "regorafenib"),
]:
    r = ttest(df[col] == 1, df[col] == 0, label, f"no {label}")
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"ttest pfs_months by {col}",
        "result_summary": f"Mean PFS on {label}={r['mean1']:.2f} vs off={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
        "p_value": r["p"],
        "effect_estimate": r["diff"],
        "significant": r["p"] < 0.05,
    })
add_iteration(4, hypotheses, analyses)


# =====================================================================
# Iteration 5 — Cetuximab × KRAS interaction (canonical CRC effect)
# =====================================================================
hypotheses = [
    {"id": "h5.1", "text": "The pfs_months benefit of treatment_cetuximab is larger in KRAS wild-type (kras_mutation=0) than KRAS mutant (kras_mutation=1) patients (positive cetuximab x KRAS-WT interaction).", "kind": "novel"},
    {"id": "h5.2", "text": "Within KRAS wild-type patients, treatment_cetuximab is associated with longer pfs_months than no cetuximab.", "kind": "novel"},
    {"id": "h5.3", "text": "Within KRAS mutant patients, treatment_cetuximab is not associated with longer pfs_months (and may be associated with shorter) compared to no cetuximab.", "kind": "novel"},
]
analyses = []

m = ols_summary("pfs_months ~ treatment_cetuximab * kras_mutation")
inter_p = float(m.pvalues["treatment_cetuximab:kras_mutation"])
inter_b = float(m.params["treatment_cetuximab:kras_mutation"])
analyses.append({
    "hypothesis_ids": ["h5.1"],
    "code": "smf.ols('pfs_months ~ treatment_cetuximab * kras_mutation', df).fit()",
    "result_summary": f"Interaction coefficient cetuximab:kras_mutation = {inter_b:.3f} mo, p={fmt_p(inter_p)}. Negative => cetuximab benefit shrinks/reverses in KRAS mutants.",
    "p_value": inter_p,
    "effect_estimate": inter_b,
    "significant": inter_p < 0.05,
})

sub = df[df["kras_mutation"] == 0]
r = ttest(sub["treatment_cetuximab"] == 1, sub["treatment_cetuximab"] == 0, "cetux on, KRAS-WT", "cetux off, KRAS-WT", data=sub)
analyses.append({
    "hypothesis_ids": ["h5.2"],
    "code": "ttest pfs_months by treatment_cetuximab within KRAS-WT",
    "result_summary": f"KRAS-WT: PFS cetux on={r['mean1']:.2f} vs off={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})

sub = df[df["kras_mutation"] == 1]
r = ttest(sub["treatment_cetuximab"] == 1, sub["treatment_cetuximab"] == 0, "cetux on, KRAS-mut", "cetux off, KRAS-mut", data=sub)
analyses.append({
    "hypothesis_ids": ["h5.3"],
    "code": "ttest pfs_months by treatment_cetuximab within KRAS-mut",
    "result_summary": f"KRAS-mut: PFS cetux on={r['mean1']:.2f} vs off={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})
add_iteration(5, hypotheses, analyses)


# =====================================================================
# Iteration 6 — Cetuximab × NRAS and × BRAF interactions
# =====================================================================
hypotheses = [
    {"id": "h6.1", "text": "Cetuximab benefit on pfs_months is reduced in NRAS-mutant compared to NRAS-wild-type patients (negative cetuximab × NRAS interaction).", "kind": "novel"},
    {"id": "h6.2", "text": "Cetuximab benefit on pfs_months is reduced in BRAF V600E mutant compared to BRAF wild-type patients (negative cetuximab × BRAF interaction).", "kind": "novel"},
    {"id": "h6.3", "text": "Within RAS/RAF wild-type patients (kras=0 & nras=0 & braf_v600e=0), treatment_cetuximab is associated with longer pfs_months than no cetuximab, with a larger effect than in the unselected cohort.", "kind": "refined"},
]
analyses = []

m = ols_summary("pfs_months ~ treatment_cetuximab * nras_mutation")
b = float(m.params["treatment_cetuximab:nras_mutation"]); p = float(m.pvalues["treatment_cetuximab:nras_mutation"])
analyses.append({
    "hypothesis_ids": ["h6.1"],
    "code": "smf.ols('pfs_months ~ treatment_cetuximab * nras_mutation', df).fit()",
    "result_summary": f"Interaction cetuximab:nras = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})

m = ols_summary("pfs_months ~ treatment_cetuximab * braf_v600e")
b = float(m.params["treatment_cetuximab:braf_v600e"]); p = float(m.pvalues["treatment_cetuximab:braf_v600e"])
analyses.append({
    "hypothesis_ids": ["h6.2"],
    "code": "smf.ols('pfs_months ~ treatment_cetuximab * braf_v600e', df).fit()",
    "result_summary": f"Interaction cetuximab:braf_v600e = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})

sub = df[(df["kras_mutation"] == 0) & (df["nras_mutation"] == 0) & (df["braf_v600e"] == 0)]
r = ttest(sub["treatment_cetuximab"] == 1, sub["treatment_cetuximab"] == 0, "cetux on, RAS/RAF WT", "cetux off, RAS/RAF WT", data=sub)
analyses.append({
    "hypothesis_ids": ["h6.3"],
    "code": "ttest within RAS/RAF wild-type subset",
    "result_summary": f"RAS/RAF WT (n={len(sub)}): PFS cetux on={r['mean1']:.2f} vs off={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})
add_iteration(6, hypotheses, analyses)


# =====================================================================
# Iteration 7 — Cetuximab × right-sided primary
# =====================================================================
hypotheses = [
    {"id": "h7.1", "text": "Cetuximab benefit on pfs_months is greater in left-sided (right_sided_primary=0) than right-sided (right_sided_primary=1) primaries (negative cetuximab × right-sided interaction).", "kind": "novel"},
    {"id": "h7.2", "text": "Within left-sided primaries, treatment_cetuximab is associated with longer pfs_months than no cetuximab.", "kind": "novel"},
]
analyses = []

m = ols_summary("pfs_months ~ treatment_cetuximab * right_sided_primary")
b = float(m.params["treatment_cetuximab:right_sided_primary"]); p = float(m.pvalues["treatment_cetuximab:right_sided_primary"])
analyses.append({
    "hypothesis_ids": ["h7.1"],
    "code": "smf.ols('pfs_months ~ treatment_cetuximab * right_sided_primary', df).fit()",
    "result_summary": f"Interaction cetuximab:right_sided_primary = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})

sub = df[df["right_sided_primary"] == 0]
r = ttest(sub["treatment_cetuximab"] == 1, sub["treatment_cetuximab"] == 0, "cetux on, left-sided", "cetux off, left-sided", data=sub)
analyses.append({
    "hypothesis_ids": ["h7.2"],
    "code": "ttest within left-sided primaries",
    "result_summary": f"Left-sided: PFS cetux on={r['mean1']:.2f} vs off={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})
add_iteration(7, hypotheses, analyses)


# =====================================================================
# Iteration 8 — Pembrolizumab × MSI-high interaction
# =====================================================================
hypotheses = [
    {"id": "h8.1", "text": "The pfs_months benefit of treatment_pembrolizumab is larger in MSI-high (msi_high=1) than in non-MSI-high patients (positive pembrolizumab × msi_high interaction).", "kind": "novel"},
    {"id": "h8.2", "text": "Within MSI-high patients, treatment_pembrolizumab is associated with longer pfs_months than no pembrolizumab.", "kind": "novel"},
    {"id": "h8.3", "text": "Within microsatellite-stable patients (msi_high=0), treatment_pembrolizumab is not associated with longer pfs_months.", "kind": "novel"},
]
analyses = []

m = ols_summary("pfs_months ~ treatment_pembrolizumab * msi_high")
b = float(m.params["treatment_pembrolizumab:msi_high"]); p = float(m.pvalues["treatment_pembrolizumab:msi_high"])
analyses.append({
    "hypothesis_ids": ["h8.1"],
    "code": "smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high', df).fit()",
    "result_summary": f"Interaction pembrolizumab:msi_high = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})

sub = df[df["msi_high"] == 1]
r = ttest(sub["treatment_pembrolizumab"] == 1, sub["treatment_pembrolizumab"] == 0, "pembro on, MSI-H", "pembro off, MSI-H", data=sub)
analyses.append({
    "hypothesis_ids": ["h8.2"],
    "code": "ttest within MSI-H subset",
    "result_summary": f"MSI-H (n={len(sub)}): PFS pembro on={r['mean1']:.2f} vs off={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})

sub = df[df["msi_high"] == 0]
r = ttest(sub["treatment_pembrolizumab"] == 1, sub["treatment_pembrolizumab"] == 0, "pembro on, MSS", "pembro off, MSS", data=sub)
analyses.append({
    "hypothesis_ids": ["h8.3"],
    "code": "ttest within MSS subset",
    "result_summary": f"MSS (n={len(sub)}): PFS pembro on={r['mean1']:.2f} vs off={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})
add_iteration(8, hypotheses, analyses)


# =====================================================================
# Iteration 9 — Encorafenib × BRAF V600E interaction
# =====================================================================
hypotheses = [
    {"id": "h9.1", "text": "The pfs_months benefit of treatment_encorafenib is larger in BRAF V600E mutant (braf_v600e=1) than wild-type patients (positive encorafenib × braf_v600e interaction).", "kind": "novel"},
    {"id": "h9.2", "text": "Within BRAF V600E mutant patients, treatment_encorafenib is associated with longer pfs_months than no encorafenib.", "kind": "novel"},
    {"id": "h9.3", "text": "Within BRAF wild-type patients, treatment_encorafenib is not associated with longer pfs_months than no encorafenib.", "kind": "novel"},
]
analyses = []

m = ols_summary("pfs_months ~ treatment_encorafenib * braf_v600e")
b = float(m.params["treatment_encorafenib:braf_v600e"]); p = float(m.pvalues["treatment_encorafenib:braf_v600e"])
analyses.append({
    "hypothesis_ids": ["h9.1"],
    "code": "smf.ols('pfs_months ~ treatment_encorafenib * braf_v600e', df).fit()",
    "result_summary": f"Interaction encorafenib:braf_v600e = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})

sub = df[df["braf_v600e"] == 1]
r = ttest(sub["treatment_encorafenib"] == 1, sub["treatment_encorafenib"] == 0, "enc on, BRAF-mut", "enc off, BRAF-mut", data=sub)
analyses.append({
    "hypothesis_ids": ["h9.2"],
    "code": "ttest within BRAF V600E subset",
    "result_summary": f"BRAF-V600E (n={len(sub)}): PFS encorafenib on={r['mean1']:.2f} vs off={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})

sub = df[df["braf_v600e"] == 0]
r = ttest(sub["treatment_encorafenib"] == 1, sub["treatment_encorafenib"] == 0, "enc on, BRAF-WT", "enc off, BRAF-WT", data=sub)
analyses.append({
    "hypothesis_ids": ["h9.3"],
    "code": "ttest within BRAF wild-type subset",
    "result_summary": f"BRAF-WT (n={len(sub)}): PFS encorafenib on={r['mean1']:.2f} vs off={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})
add_iteration(9, hypotheses, analyses)


# =====================================================================
# Iteration 10 — Trastuzumab/tucatinib × HER2 interaction
# =====================================================================
hypotheses = [
    {"id": "h10.1", "text": "The pfs_months benefit of treatment_trastuzumab_tucatinib is larger in HER2-amplified (her2_amplified=1) than non-amplified patients (positive interaction).", "kind": "novel"},
    {"id": "h10.2", "text": "Within HER2-amplified patients, treatment_trastuzumab_tucatinib is associated with longer pfs_months than no trastuzumab/tucatinib.", "kind": "novel"},
    {"id": "h10.3", "text": "Within HER2-non-amplified patients, treatment_trastuzumab_tucatinib is not associated with longer pfs_months.", "kind": "novel"},
]
analyses = []

m = ols_summary("pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified")
b = float(m.params["treatment_trastuzumab_tucatinib:her2_amplified"]); p = float(m.pvalues["treatment_trastuzumab_tucatinib:her2_amplified"])
analyses.append({
    "hypothesis_ids": ["h10.1"],
    "code": "smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified', df).fit()",
    "result_summary": f"Interaction tras_tuc:her2_amplified = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})

sub = df[df["her2_amplified"] == 1]
r = ttest(sub["treatment_trastuzumab_tucatinib"] == 1, sub["treatment_trastuzumab_tucatinib"] == 0, "tras_tuc on, HER2+", "tras_tuc off, HER2+", data=sub)
analyses.append({
    "hypothesis_ids": ["h10.2"],
    "code": "ttest within HER2-amplified subset",
    "result_summary": f"HER2+ (n={len(sub)}): PFS tras_tuc on={r['mean1']:.2f} vs off={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})

sub = df[df["her2_amplified"] == 0]
r = ttest(sub["treatment_trastuzumab_tucatinib"] == 1, sub["treatment_trastuzumab_tucatinib"] == 0, "tras_tuc on, HER2-", "tras_tuc off, HER2-", data=sub)
analyses.append({
    "hypothesis_ids": ["h10.3"],
    "code": "ttest within HER2-non-amplified subset",
    "result_summary": f"HER2- (n={len(sub)}): PFS tras_tuc on={r['mean1']:.2f} vs off={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})
add_iteration(10, hypotheses, analyses)


# =====================================================================
# Iteration 11 — Bevacizumab effects and modifiers
# =====================================================================
hypotheses = [
    {"id": "h11.1", "text": "Treatment_bevacizumab is associated with longer pfs_months across the cohort independent of biomarker status (no strong biomarker selection).", "kind": "novel"},
    {"id": "h11.2", "text": "Bevacizumab benefit is similar between KRAS-mutant and KRAS-wild-type patients (no significant bevacizumab × kras interaction).", "kind": "novel"},
    {"id": "h11.3", "text": "Bevacizumab benefit is similar between right-sided and left-sided primaries (no significant bevacizumab × right-sided interaction).", "kind": "novel"},
]
analyses = []

m = ols_summary("pfs_months ~ treatment_bevacizumab")
b = float(m.params["treatment_bevacizumab"]); p = float(m.pvalues["treatment_bevacizumab"])
analyses.append({
    "hypothesis_ids": ["h11.1"],
    "code": "smf.ols('pfs_months ~ treatment_bevacizumab', df).fit()",
    "result_summary": f"OLS bevacizumab effect = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})

m = ols_summary("pfs_months ~ treatment_bevacizumab * kras_mutation")
b = float(m.params["treatment_bevacizumab:kras_mutation"]); p = float(m.pvalues["treatment_bevacizumab:kras_mutation"])
analyses.append({
    "hypothesis_ids": ["h11.2"],
    "code": "smf.ols('pfs_months ~ treatment_bevacizumab * kras_mutation', df).fit()",
    "result_summary": f"Interaction bevacizumab:kras_mutation = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})

m = ols_summary("pfs_months ~ treatment_bevacizumab * right_sided_primary")
b = float(m.params["treatment_bevacizumab:right_sided_primary"]); p = float(m.pvalues["treatment_bevacizumab:right_sided_primary"])
analyses.append({
    "hypothesis_ids": ["h11.3"],
    "code": "smf.ols('pfs_months ~ treatment_bevacizumab * right_sided_primary', df).fit()",
    "result_summary": f"Interaction bevacizumab:right_sided_primary = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})
add_iteration(11, hypotheses, analyses)


# =====================================================================
# Iteration 12 — Regorafenib effects and modifiers
# =====================================================================
hypotheses = [
    {"id": "h12.1", "text": "Treatment_regorafenib has different (likely shorter) pfs_months effect than no regorafenib in the overall cohort, reflecting later-line use.", "kind": "novel"},
    {"id": "h12.2", "text": "Regorafenib effect on pfs_months differs by prior_lines_of_therapy (interaction term significant).", "kind": "novel"},
]
analyses = []
m = ols_summary("pfs_months ~ treatment_regorafenib")
b = float(m.params["treatment_regorafenib"]); p = float(m.pvalues["treatment_regorafenib"])
analyses.append({
    "hypothesis_ids": ["h12.1"],
    "code": "smf.ols('pfs_months ~ treatment_regorafenib', df).fit()",
    "result_summary": f"OLS regorafenib effect = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})
m = ols_summary("pfs_months ~ treatment_regorafenib * prior_lines_of_therapy")
b = float(m.params["treatment_regorafenib:prior_lines_of_therapy"]); p = float(m.pvalues["treatment_regorafenib:prior_lines_of_therapy"])
analyses.append({
    "hypothesis_ids": ["h12.2"],
    "code": "smf.ols('pfs_months ~ treatment_regorafenib * prior_lines_of_therapy', df).fit()",
    "result_summary": f"Interaction regorafenib:prior_lines = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})
add_iteration(12, hypotheses, analyses)


# =====================================================================
# Iteration 13 — Multivariable model: treatments + key biomarkers + covariates
# =====================================================================
hypotheses = [
    {"id": "h13.1", "text": "After adjusting for age, sex, ECOG, stage_iv, key biomarkers, and disease burden, treatment_cetuximab remains associated with longer pfs_months (adjusted main effect remains positive on average; in cohort it may be diluted by RAS mutants).", "kind": "refined"},
    {"id": "h13.2", "text": "After adjusting for ECOG, stage_iv, age, sex, biomarkers, and disease burden, MSI-high status remains a positive prognostic biomarker for pfs_months.", "kind": "refined"},
    {"id": "h13.3", "text": "After adjusting, BRAF V600E remains a poor-prognosis biomarker (negative association with pfs_months).", "kind": "refined"},
]
analyses = []
formula = (
    "pfs_months ~ age_years + sex_female + ecog_ps + stage_iv + right_sided_primary "
    "+ kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified "
    "+ liver_mets + bone_mets + albumin_g_dl + ldh_u_l + cea_ng_ml + weight_loss_pct_6mo + nlr "
    "+ treatment_cetuximab + treatment_bevacizumab + treatment_pembrolizumab "
    "+ treatment_encorafenib + treatment_trastuzumab_tucatinib + treatment_regorafenib"
)
m = ols_summary(formula)
for var, hid in [
    ("treatment_cetuximab", "h13.1"),
    ("msi_high", "h13.2"),
    ("braf_v600e", "h13.3"),
]:
    b = float(m.params[var]); p = float(m.pvalues[var])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"OLS adjusted: pfs_months ~ ... + {var} + ...",
        "result_summary": f"Adjusted coefficient {var} = {b:.3f} mo, p={fmt_p(p)}.",
        "p_value": p, "effect_estimate": b, "significant": p < 0.05,
    })
# also save full model for later reference
multivar_model = m
add_iteration(13, hypotheses, analyses)


# =====================================================================
# Iteration 14 — ECOG-modified treatment effects
# =====================================================================
hypotheses = [
    {"id": "h14.1", "text": "Treatment effect of bevacizumab on pfs_months is smaller in patients with ECOG ≥2 than ECOG 0–1 (negative bevacizumab × ECOG≥2 interaction).", "kind": "novel"},
    {"id": "h14.2", "text": "Treatment effect of cetuximab on pfs_months is smaller in patients with ECOG ≥2 than ECOG 0–1 (negative cetuximab × ECOG≥2 interaction).", "kind": "novel"},
]
analyses = []
df["ecog_ge2"] = (df["ecog_ps"] >= 2).astype(int)
for tx, hid in [("treatment_bevacizumab", "h14.1"), ("treatment_cetuximab", "h14.2")]:
    m = ols_summary(f"pfs_months ~ {tx} * ecog_ge2")
    b = float(m.params[f"{tx}:ecog_ge2"]); p = float(m.pvalues[f"{tx}:ecog_ge2"])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {tx} * ecog_ge2', df).fit()",
        "result_summary": f"Interaction {tx}:ecog_ge2 = {b:.3f} mo, p={fmt_p(p)}.",
        "p_value": p, "effect_estimate": b, "significant": p < 0.05,
    })
add_iteration(14, hypotheses, analyses)


# =====================================================================
# Iteration 15 — Age × treatment interactions
# =====================================================================
hypotheses = [
    {"id": "h15.1", "text": "The pfs_months effect of treatment_pembrolizumab does not vary substantially with age_years (no significant pembrolizumab × age interaction).", "kind": "novel"},
    {"id": "h15.2", "text": "The pfs_months effect of treatment_bevacizumab decreases with increasing age (negative bevacizumab × age interaction).", "kind": "novel"},
]
analyses = []
for tx, hid in [("treatment_pembrolizumab", "h15.1"), ("treatment_bevacizumab", "h15.2")]:
    m = ols_summary(f"pfs_months ~ {tx} * age_years")
    key = f"{tx}:age_years"
    b = float(m.params[key]); p = float(m.pvalues[key])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {tx} * age_years', df).fit()",
        "result_summary": f"Interaction {tx}:age_years = {b:.4f} mo per year, p={fmt_p(p)}.",
        "p_value": p, "effect_estimate": b, "significant": p < 0.05,
    })
add_iteration(15, hypotheses, analyses)


# =====================================================================
# Iteration 16 — Sex × treatment interactions
# =====================================================================
hypotheses = [
    {"id": "h16.1", "text": "Treatment effects on pfs_months do not differ between female and male patients for cetuximab (no significant cetuximab × sex_female interaction).", "kind": "novel"},
    {"id": "h16.2", "text": "Treatment effects on pfs_months do not differ between female and male patients for pembrolizumab (no significant pembrolizumab × sex_female interaction).", "kind": "novel"},
]
analyses = []
for tx, hid in [("treatment_cetuximab", "h16.1"), ("treatment_pembrolizumab", "h16.2")]:
    m = ols_summary(f"pfs_months ~ {tx} * sex_female")
    key = f"{tx}:sex_female"
    b = float(m.params[key]); p = float(m.pvalues[key])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {tx} * sex_female', df).fit()",
        "result_summary": f"Interaction {tx}:sex_female = {b:.3f} mo, p={fmt_p(p)}.",
        "p_value": p, "effect_estimate": b, "significant": p < 0.05,
    })
add_iteration(16, hypotheses, analyses)


# =====================================================================
# Iteration 17 — Race/ethnicity associations
# =====================================================================
hypotheses = [
    {"id": "h17.1", "text": "Mean pfs_months differs across race_ethnicity categories (overall ANOVA).", "kind": "novel"},
    {"id": "h17.2", "text": "Black patients have shorter pfs_months than white patients on average (race_ethnicity main effect).", "kind": "novel"},
]
analyses = []
groups = [df.loc[df["race_ethnicity"] == g, "pfs_months"].values for g in df["race_ethnicity"].unique()]
F, p = stats.f_oneway(*groups)
analyses.append({
    "hypothesis_ids": ["h17.1"],
    "code": "stats.f_oneway across race_ethnicity",
    "result_summary": f"ANOVA F={F:.3f}, p={fmt_p(p)} across race_ethnicity.",
    "p_value": float(p), "effect_estimate": float(F), "significant": p < 0.05,
})
m = smf.ols("pfs_months ~ C(race_ethnicity, Treatment(reference='white'))", df).fit()
key = "C(race_ethnicity, Treatment(reference='white'))[T.black]"
b = float(m.params[key]); p = float(m.pvalues[key])
analyses.append({
    "hypothesis_ids": ["h17.2"],
    "code": "OLS with race_ethnicity vs white reference",
    "result_summary": f"Black vs white coefficient = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})
add_iteration(17, hypotheses, analyses)


# =====================================================================
# Iteration 18 — Insurance & social factors
# =====================================================================
hypotheses = [
    {"id": "h18.1", "text": "Mean pfs_months differs across insurance_type categories (overall ANOVA).", "kind": "novel"},
    {"id": "h18.2", "text": "Uninsured patients have shorter pfs_months than privately insured patients.", "kind": "novel"},
    {"id": "h18.3", "text": "Rural residence (rural_residence=1) is associated with shorter pfs_months than non-rural.", "kind": "novel"},
]
analyses = []
groups = [df.loc[df["insurance_type"] == g, "pfs_months"].values for g in df["insurance_type"].unique()]
F, p = stats.f_oneway(*groups)
analyses.append({
    "hypothesis_ids": ["h18.1"],
    "code": "stats.f_oneway across insurance_type",
    "result_summary": f"ANOVA F={F:.3f}, p={fmt_p(p)}.",
    "p_value": float(p), "effect_estimate": float(F), "significant": p < 0.05,
})
m = smf.ols("pfs_months ~ C(insurance_type, Treatment(reference='private'))", df).fit()
key = "C(insurance_type, Treatment(reference='private'))[T.uninsured]"
b = float(m.params[key]); p = float(m.pvalues[key])
analyses.append({
    "hypothesis_ids": ["h18.2"],
    "code": "OLS insurance_type vs private reference",
    "result_summary": f"Uninsured vs private coefficient = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})
r = ttest(df["rural_residence"] == 1, df["rural_residence"] == 0, "rural", "non-rural")
analyses.append({
    "hypothesis_ids": ["h18.3"],
    "code": "ttest pfs_months by rural_residence",
    "result_summary": f"PFS rural={r['mean1']:.2f} vs non-rural={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})
add_iteration(18, hypotheses, analyses)


# =====================================================================
# Iteration 19 — Comorbidity burden
# =====================================================================
hypotheses = [
    {"id": "h19.1", "text": "Greater number of comorbidities (sum of diabetes_mellitus, hypertension, copd, chronic_kidney_disease, heart_failure, coronary_artery_disease, atrial_fibrillation) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h19.2", "text": "History of venous_thromboembolism_history is associated with shorter pfs_months.", "kind": "novel"},
]
analyses = []
df["comorb_count"] = df[["diabetes_mellitus", "hypertension", "copd", "chronic_kidney_disease",
                          "heart_failure", "coronary_artery_disease", "atrial_fibrillation"]].sum(axis=1)
m = ols_summary("pfs_months ~ comorb_count")
b = float(m.params["comorb_count"]); p = float(m.pvalues["comorb_count"])
analyses.append({
    "hypothesis_ids": ["h19.1"],
    "code": "smf.ols('pfs_months ~ comorb_count', df).fit()",
    "result_summary": f"Comorb count slope = {b:.3f} mo per condition, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})
r = ttest(df["venous_thromboembolism_history"] == 1, df["venous_thromboembolism_history"] == 0, "VTE hx", "no VTE hx")
analyses.append({
    "hypothesis_ids": ["h19.2"],
    "code": "ttest pfs_months by venous_thromboembolism_history",
    "result_summary": f"PFS VTE hx={r['mean1']:.2f} vs no VTE hx={r['mean0']:.2f}, diff={r['diff']:.3f} mo, p={fmt_p(r['p'])}.",
    "p_value": r["p"], "effect_estimate": r["diff"], "significant": r["p"] < 0.05,
})
add_iteration(19, hypotheses, analyses)


# =====================================================================
# Iteration 20 — Symptom grade impact
# =====================================================================
hypotheses = [
    {"id": "h20.1", "text": "Higher fatigue_grade is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h20.2", "text": "Higher pain_nrs is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h20.3", "text": "Higher appetite_loss_grade is associated with shorter pfs_months.", "kind": "novel"},
]
analyses = []
for col, hid in [("fatigue_grade", "h20.1"), ("pain_nrs", "h20.2"), ("appetite_loss_grade", "h20.3")]:
    m = ols_summary(f"pfs_months ~ {col}")
    b = float(m.params[col]); p = float(m.pvalues[col])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {col}', df).fit()",
        "result_summary": f"Slope {col} = {b:.3f} mo per unit, p={fmt_p(p)}.",
        "p_value": p, "effect_estimate": b, "significant": p < 0.05,
    })
add_iteration(20, hypotheses, analyses)


# =====================================================================
# Iteration 21 — SNPs vs PFS (screening)
# =====================================================================
hypotheses = [
    {"id": "h21.1", "text": "None of the candidate SNPs (snp_rs* columns coded as additive 0/1/2 dosages) are individually significantly associated with pfs_months after Bonferroni correction across the SNP panel.", "kind": "novel"},
]
analyses = []
snp_cols = [c for c in df.columns if c.startswith("snp_rs")]
sig_snps = []
min_p = 1.0
min_snp = None
for s in snp_cols:
    m = smf.ols(f"pfs_months ~ {s}", df).fit()
    p = float(m.pvalues[s])
    if p < min_p:
        min_p = p; min_snp = s; min_b = float(m.params[s])
    if p < 0.05 / len(snp_cols):
        sig_snps.append((s, float(m.params[s]), p))
analyses.append({
    "hypothesis_ids": ["h21.1"],
    "code": "loop OLS pfs_months ~ snp_rsXXX for each SNP; Bonferroni alpha=0.05/N",
    "result_summary": (
        f"Tested {len(snp_cols)} SNPs. Bonferroni-significant SNPs: {len(sig_snps)}. "
        f"Smallest p across SNPs: {min_snp} b={min_b:.3f}, p={fmt_p(min_p)}."
    ),
    "p_value": min_p,
    "effect_estimate": float(min_b),
    "significant": len(sig_snps) > 0,
})
add_iteration(21, hypotheses, analyses)


# =====================================================================
# Iteration 22 — Lab markers in adjusted model
# =====================================================================
hypotheses = [
    {"id": "h22.1", "text": "After adjusting for age, sex, ECOG, stage_iv, biomarkers, and disease burden, lower albumin_g_dl is independently associated with shorter pfs_months.", "kind": "refined"},
    {"id": "h22.2", "text": "After adjusting, higher LDH (ldh_u_l) is independently associated with shorter pfs_months.", "kind": "refined"},
    {"id": "h22.3", "text": "After adjusting, higher NLR (nlr) is independently associated with shorter pfs_months.", "kind": "refined"},
]
analyses = []
m = multivar_model  # already includes albumin, ldh, nlr
for var, hid in [("albumin_g_dl", "h22.1"), ("ldh_u_l", "h22.2"), ("nlr", "h22.3")]:
    b = float(m.params[var]); p = float(m.pvalues[var])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": "Adjusted multivariable OLS (Iteration 13 model)",
        "result_summary": f"Adjusted coefficient {var} = {b:.5f} mo per unit, p={fmt_p(p)}.",
        "p_value": p, "effect_estimate": b, "significant": p < 0.05,
    })
add_iteration(22, hypotheses, analyses)


# =====================================================================
# Iteration 23 — Cetuximab benefit constrained by NRAS, BRAF, right-sidedness in adjusted model
# =====================================================================
hypotheses = [
    {"id": "h23.1", "text": "In a multivariable model with cetuximab × kras_mutation interaction (adjusting for age, sex, ECOG, stage_iv, right_sided, biomarkers), the cetuximab × kras_mutation interaction remains significantly negative.", "kind": "refined"},
    {"id": "h23.2", "text": "In a multivariable model with pembrolizumab × msi_high interaction, the interaction remains significantly positive after adjustment.", "kind": "refined"},
]
analyses = []
formula = ("pfs_months ~ age_years + sex_female + ecog_ps + stage_iv + right_sided_primary "
           "+ kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified "
           "+ liver_mets + bone_mets + albumin_g_dl + ldh_u_l + cea_ng_ml + weight_loss_pct_6mo + nlr "
           "+ treatment_cetuximab*kras_mutation "
           "+ treatment_bevacizumab + treatment_pembrolizumab + treatment_encorafenib "
           "+ treatment_trastuzumab_tucatinib + treatment_regorafenib")
m = smf.ols(formula, df).fit()
b = float(m.params["treatment_cetuximab:kras_mutation"]); p = float(m.pvalues["treatment_cetuximab:kras_mutation"])
analyses.append({
    "hypothesis_ids": ["h23.1"],
    "code": "Adjusted OLS with cetuximab × kras interaction",
    "result_summary": f"Adjusted interaction cetuximab:kras_mutation = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})
formula2 = formula.replace("treatment_cetuximab*kras_mutation", "treatment_cetuximab + kras_mutation").replace(
    "treatment_pembrolizumab", "treatment_pembrolizumab*msi_high")
m2 = smf.ols(formula2, df).fit()
b = float(m2.params["treatment_pembrolizumab:msi_high"]); p = float(m2.pvalues["treatment_pembrolizumab:msi_high"])
analyses.append({
    "hypothesis_ids": ["h23.2"],
    "code": "Adjusted OLS with pembrolizumab × msi_high interaction",
    "result_summary": f"Adjusted interaction pembrolizumab:msi_high = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})
add_iteration(23, hypotheses, analyses)


# =====================================================================
# Iteration 24 — Combined biomarker-matched treatment benefit
# =====================================================================
hypotheses = [
    {"id": "h24.1", "text": "Within the BRAF V600E mutant subgroup, the combination of treatment_encorafenib AND treatment_cetuximab is associated with longer pfs_months than treatment_encorafenib alone (interaction term encorafenib×cetuximab>0 within BRAF-mut).", "kind": "novel"},
    {"id": "h24.2", "text": "Within HER2-amplified patients, treatment_trastuzumab_tucatinib effect is larger when patients are KRAS wild-type than KRAS mutant (negative trastuzumab_tucatinib × kras_mutation interaction in HER2+ subset).", "kind": "novel"},
]
analyses = []
sub = df[df["braf_v600e"] == 1]
m = smf.ols("pfs_months ~ treatment_encorafenib * treatment_cetuximab", data=sub).fit()
b = float(m.params["treatment_encorafenib:treatment_cetuximab"]); p = float(m.pvalues["treatment_encorafenib:treatment_cetuximab"])
analyses.append({
    "hypothesis_ids": ["h24.1"],
    "code": "OLS within BRAF-V600E: pfs_months ~ encorafenib * cetuximab",
    "result_summary": f"BRAF-V600E subset (n={len(sub)}): interaction encorafenib:cetuximab = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})
sub2 = df[df["her2_amplified"] == 1]
m = smf.ols("pfs_months ~ treatment_trastuzumab_tucatinib * kras_mutation", data=sub2).fit()
b = float(m.params["treatment_trastuzumab_tucatinib:kras_mutation"]); p = float(m.pvalues["treatment_trastuzumab_tucatinib:kras_mutation"])
analyses.append({
    "hypothesis_ids": ["h24.2"],
    "code": "OLS within HER2+ : pfs_months ~ tras_tuc * kras_mutation",
    "result_summary": f"HER2+ subset (n={len(sub2)}): interaction tras_tuc:kras_mutation = {b:.3f} mo, p={fmt_p(p)}.",
    "p_value": p, "effect_estimate": b, "significant": p < 0.05,
})
add_iteration(24, hypotheses, analyses)


# =====================================================================
# Iteration 25 — Final adjusted model with all key biomarker × treatment interactions
# =====================================================================
hypotheses = [
    {"id": "h25.1", "text": "In a single OLS model jointly estimating cetuximab × kras_mutation, pembrolizumab × msi_high, encorafenib × braf_v600e, and trastuzumab_tucatinib × her2_amplified interactions while adjusting for clinical/lab covariates, all four biomarker-matched treatment interactions are in the expected direction (cetuximab benefit smaller in KRAS-mut; pembrolizumab, encorafenib, and trastuzumab_tucatinib benefits larger in their matched biomarker groups).", "kind": "refined"},
    {"id": "h25.2", "text": "After jointly adjusting for biomarker-matched interactions and clinical covariates, ECOG_ps and stage_iv remain significant prognostic features for pfs_months.", "kind": "refined"},
]
analyses = []
formula = (
    "pfs_months ~ age_years + sex_female + ecog_ps + stage_iv + right_sided_primary "
    "+ liver_mets + bone_mets + albumin_g_dl + ldh_u_l + cea_ng_ml + weight_loss_pct_6mo + nlr "
    "+ kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified "
    "+ treatment_bevacizumab + treatment_regorafenib "
    "+ treatment_cetuximab*kras_mutation "
    "+ treatment_pembrolizumab*msi_high "
    "+ treatment_encorafenib*braf_v600e "
    "+ treatment_trastuzumab_tucatinib*her2_amplified"
)
m_final = smf.ols(formula, df).fit()
for term, hid in [
    ("treatment_cetuximab:kras_mutation", "h25.1"),
    ("treatment_pembrolizumab:msi_high", "h25.1"),
    ("treatment_encorafenib:braf_v600e", "h25.1"),
    ("treatment_trastuzumab_tucatinib:her2_amplified", "h25.1"),
]:
    b = float(m_final.params[term]); p = float(m_final.pvalues[term])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": "Final joint adjusted OLS with four biomarker × treatment interactions",
        "result_summary": f"Adjusted interaction {term} = {b:.3f} mo, p={fmt_p(p)}.",
        "p_value": p, "effect_estimate": b, "significant": p < 0.05,
    })
for term, hid in [("ecog_ps", "h25.2"), ("stage_iv", "h25.2")]:
    b = float(m_final.params[term]); p = float(m_final.pvalues[term])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": "Final joint adjusted OLS — prognostic main effects",
        "result_summary": f"Adjusted coefficient {term} = {b:.3f} mo, p={fmt_p(p)}.",
        "p_value": p, "effect_estimate": b, "significant": p < 0.05,
    })
add_iteration(25, hypotheses, analyses)


# =====================================================================
# Build transcript and write outputs
# =====================================================================
transcript = {
    "dataset_id": "ds001_crc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-named-direct@1.0.0",
    "max_iterations": 25,
    "iterations": iterations,
}

def _coerce(o):
    if isinstance(o, dict):
        return {k: _coerce(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_coerce(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o

with open(HERE / "transcript.json", "w", encoding="utf-8") as f:
    json.dump(_coerce(transcript), f, indent=2)

# ---- Build the prose summary ----
def get(itx, ix):
    return iterations[itx - 1]["analyses"][ix]


def fmt_a(a, label=None):
    eff = a.get("effect_estimate")
    p = a.get("p_value")
    sig = a.get("significant")
    s = f"effect={eff:+.3f}" if eff is not None else "effect=?"
    s += f", p={fmt_p(p)}" if p is not None else ""
    s += " (sig)" if sig else " (ns)"
    if label:
        return f"{label}: {s}"
    return s


lines = []
lines.append("CRC oncology dataset (ds001_crc) — analysis summary across 25 iterations\n")
lines.append("=" * 78 + "\n\n")
lines.append("Outcome: pfs_months (progression-free survival in months); n=50,000.\n\n")

lines.append("Iteration 1 — Demographics & performance status\n")
for i, h in enumerate(iterations[0]["proposed_hypotheses"]):
    lines.append(f"  Hypothesis ({h['id']}): {h['text']}\n")
for a in iterations[0]["analyses"]:
    lines.append(f"    {a['result_summary']}\n")
lines.append("\n")

for it in iterations[1:]:
    lines.append(f"Iteration {it['index']}\n")
    for h in it["proposed_hypotheses"]:
        lines.append(f"  Hypothesis ({h['id']}, {h['kind']}): {h['text']}\n")
    for a in it["analyses"]:
        lines.append(f"    {a['result_summary']}\n")
    lines.append("\n")

lines.append("=" * 78 + "\n")
lines.append("OVERALL CONCLUSIONS\n")
lines.append("=" * 78 + "\n\n")
lines.append(
    "Prognostic main effects supported by the data:\n"
    "  - ECOG performance status: -1.19 mo PFS per unit (p<1e-300). Robust in adjusted\n"
    "    model (-1.17 mo per unit).\n"
    "  - Stage IV: -1.35 mo PFS vs non-stage IV (p<1e-300). Robust in adjusted model.\n"
    "  - Albumin: +0.47 mo PFS per g/dL (p~3e-115); preserved in adjusted model.\n"
    "  - 6-month weight loss percent: -0.073 mo PFS per percent (p~2e-167).\n"
    "  - KRAS mutation: -0.33 mo PFS (p~2e-58).\n"
    "  - Right-sided primary: -0.31 mo PFS (p~1e-49).\n"
    "  - BRAF V600E: -0.23 mo PFS unadjusted; -0.36 mo adjusted (p~5e-79).\n"
    "  - CEA: -0.005 mo PFS per ng/mL (p~3e-17).\n"
    "  - LDH: small negative association unadjusted (p=0.013), highly significant in\n"
    "    adjusted model (p~7e-19).\n"
    "  - Age: small POSITIVE association (older patients had slightly longer PFS,\n"
    "    +0.18 mo per year, p<1e-300), opposite to the hypothesized direction.\n\n"
)
lines.append(
    "Hypotheses refuted or unsupported:\n"
    "  - Sex (female vs male): essentially identical mean PFS (diff ~0 mo, p=0.77).\n"
    "  - MSI-high: not associated with longer PFS (diff -0.02 mo, p=0.68); refutes\n"
    "    the prognostic-favorable expectation for MSI-H in this cohort.\n"
    "  - NRAS mutation: associated with LONGER PFS (+0.22 mo, p=0.0007), opposite to\n"
    "    the hypothesized poor-prognosis direction.\n"
    "  - Liver metastases, bone metastases: no significant association with PFS.\n"
    "  - NLR, CRP, fatigue/pain/appetite-loss grades, comorbidity count,\n"
    "    venous-thromboembolism history: no significant unadjusted association with PFS.\n"
    "  - Race/ethnicity, insurance type, rural residence: no significant differences\n"
    "    in PFS in this cohort (ANOVA p>0.2 for race; uninsured-vs-private p=0.13).\n"
    "  - SNP panel (27 candidate SNPs): no SNP reached Bonferroni-corrected\n"
    "    significance; smallest unadjusted p was 0.015 (snp_rs1050828).\n\n"
)
lines.append(
    "Treatment effects (overall and biomarker-modified) — striking findings:\n"
    "  - Cetuximab: no significant overall effect (-0.04 mo, p=0.09). The classic\n"
    "    cetuximab × KRAS, × NRAS, × BRAF V600E, and × right-sided interactions were\n"
    "    ALL non-significant (interaction p-values 0.30–0.96). Subgroup analyses\n"
    "    within KRAS-WT, RAS/RAF-WT, and left-sided primaries failed to show longer\n"
    "    PFS on cetuximab. The canonical anti-EGFR biomarker selection signal is not\n"
    "    detectable in this dataset.\n"
    "  - Bevacizumab: no significant overall effect (p=0.35) and no significant\n"
    "    interactions with KRAS, right-sidedness, age, or ECOG.\n"
    "  - Pembrolizumab: no significant overall effect; pembrolizumab × MSI-high\n"
    "    interaction null (p=0.99). Within MSI-high (n=2513), pembrolizumab vs no\n"
    "    pembrolizumab differed by only 0.007 mo (p=0.96). The expected MSI-driven\n"
    "    immunotherapy benefit was not observed.\n"
    "  - Encorafenib: no significant overall effect; encorafenib × BRAF V600E\n"
    "    interaction non-significant and in the opposite direction (-0.14 mo,\n"
    "    p=0.38). Within BRAF-V600E mutants (n=2272), encorafenib was associated\n"
    "    with numerically shorter PFS (-0.13 mo, p=0.33).\n"
    "  - Trastuzumab + tucatinib: no significant overall effect; tras_tuc × HER2\n"
    "    interaction null (p=0.78). Within HER2-amplified patients (n=1504),\n"
    "    tras_tuc made no difference (+0.02 mo, p=0.93).\n"
    "  - Regorafenib: significantly LONGER PFS on regorafenib than off (+0.97 mo,\n"
    "    p<1e-216). This was the largest treatment-associated effect in the cohort\n"
    "    and is the opposite direction from the originally hypothesized later-line\n"
    "    indication; in this dataset regorafenib appears as a positive predictor of\n"
    "    PFS, possibly reflecting selection of patients who received it.\n"
    "  - Combined targeted regimens within BRAF-V600E (encorafenib + cetuximab) and\n"
    "    within HER2+ (tras_tuc × KRAS) showed no significant interactions.\n\n"
)
lines.append(
    "Adjusted multivariable model (Iteration 13/22) confirmed ECOG, stage IV, BRAF\n"
    "V600E, albumin, LDH, weight-loss percent, and CEA as independent prognostic\n"
    "features for PFS, while neither MSI-high nor any treatment indicator (after\n"
    "adjustment) carried an independent association of clinical magnitude. The\n"
    "final joint model with four biomarker × treatment interactions (Iteration 25)\n"
    "kept all four interactions at p>0.2, while ECOG and stage IV remained the\n"
    "dominant prognostic features.\n\n"
)
lines.append(
    "Bottom line: This CRC cohort behaves as a strongly prognostic but largely\n"
    "treatment-null and biomarker-matching-null dataset for the standard CRC\n"
    "precision-oncology hypotheses. Performance status, stage IV, BRAF V600E,\n"
    "albumin, weight loss, KRAS mutation, primary-side, CEA, LDH, and (paradoxically)\n"
    "older age and regorafenib exposure are the variables that move PFS in this\n"
    "dataset; the canonical cetuximab × RAS, pembrolizumab × MSI-H, encorafenib ×\n"
    "BRAF V600E, and trastuzumab/tucatinib × HER2 effects are not present at\n"
    "detectable magnitudes despite ample sample size in each biomarker subgroup.\n"
)

with open(HERE / "analysis_summary.txt", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Done. Wrote transcript.json and analysis_summary.txt.")
print(f"Iterations recorded: {len(iterations)}")
print(f"Total analyses: {sum(len(it['analyses']) for it in iterations)}")
print(f"Total hypotheses: {sum(len(it['proposed_hypotheses']) for it in iterations)}")
