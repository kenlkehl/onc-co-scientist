"""Fresh iterative analysis of ds001_crc.

Loads dataset, runs a series of statistical tests organized into iterations,
and dumps all numeric results to results.json. The transcript and summary
are built from that JSON in a separate step.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
np.random.seed(0)

DF = pd.read_parquet("dataset.parquet")
N = len(DF)
TX = [
    "treatment_cetuximab",
    "treatment_bevacizumab",
    "treatment_pembrolizumab",
    "treatment_encorafenib",
    "treatment_trastuzumab_tucatinib",
    "treatment_regorafenib",
]
BINARY = [
    "sex_female", "stage_iv", "right_sided_primary",
    "kras_mutation", "nras_mutation", "braf_v600e",
    "msi_high", "her2_amplified", "ntrk_fusion",
] + TX
CONT = [
    "age_years", "ecog_ps", "cea_ng_ml", "albumin_g_dl", "ldh_u_l",
    "weight_loss_pct_6mo", "crp_mg_l", "nlr", "hemoglobin_g_dl",
    "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
    "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
    "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl",
]

results = {}


def reg(formula, label):
    m = smf.ols(formula, data=DF).fit()
    out = {}
    for name in m.params.index:
        out[name] = {
            "coef": float(m.params[name]),
            "se": float(m.bse[name]),
            "p": float(m.pvalues[name]),
            "ci_low": float(m.conf_int().loc[name, 0]),
            "ci_high": float(m.conf_int().loc[name, 1]),
        }
    results[label] = out
    return m


def mean_diff(col, group_col, group_val=1):
    a = DF.loc[DF[group_col] == group_val, col]
    b = DF.loc[DF[group_col] != group_val, col]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    out = {
        "group_mean": float(a.mean()),
        "other_mean": float(b.mean()),
        "diff": float(a.mean() - b.mean()),
        "t": float(t),
        "p": float(p),
        "n_group": int(a.size),
        "n_other": int(b.size),
    }
    return out


def subgroup_effect(treatment, subgroup_mask, label):
    """OLS of pfs ~ treatment in the subgroup defined by subgroup_mask."""
    sub = DF.loc[subgroup_mask].copy()
    if sub[treatment].nunique() < 2 or len(sub) < 30:
        results[label] = {"n": int(len(sub)), "note": "insufficient variation"}
        return None
    m = smf.ols(f"pfs_months ~ {treatment}", data=sub).fit()
    results[label] = {
        "n": int(len(sub)),
        "n_treated": int(sub[treatment].sum()),
        "coef": float(m.params[treatment]),
        "se": float(m.bse[treatment]),
        "p": float(m.pvalues[treatment]),
        "mean_treated": float(sub.loc[sub[treatment] == 1, "pfs_months"].mean()),
        "mean_untreated": float(sub.loc[sub[treatment] == 0, "pfs_months"].mean()),
    }
    return m


# ---------- Iter 1: PFS distribution & demographics ----------
results["pfs_describe"] = DF["pfs_months"].describe().to_dict()
results["age_vs_pfs_corr"] = {
    "r": float(DF["age_years"].corr(DF["pfs_months"])),
    "n": N,
}
# spearman as well
r, p = stats.pearsonr(DF["age_years"], DF["pfs_months"])
results["age_vs_pfs_pearson"] = {"r": float(r), "p": float(p)}
r, p = stats.pearsonr(DF["sex_female"], DF["pfs_months"])
results["sex_vs_pfs_pearson"] = {"r": float(r), "p": float(p)}
results["pfs_by_sex"] = mean_diff("pfs_months", "sex_female", 1)

# ---------- Iter 2: Disease stage, ECOG, sidedness ----------
results["pfs_by_stage_iv"] = mean_diff("pfs_months", "stage_iv", 1)
results["pfs_by_right_sided"] = mean_diff("pfs_months", "right_sided_primary", 1)
# ECOG by group
ecog_groups = {int(v): float(DF.loc[DF["ecog_ps"] == v, "pfs_months"].mean()) for v in sorted(DF["ecog_ps"].unique())}
results["pfs_by_ecog"] = ecog_groups
# Linear regression with ecog
r, p = stats.pearsonr(DF["ecog_ps"], DF["pfs_months"])
results["ecog_vs_pfs_pearson"] = {"r": float(r), "p": float(p)}

# ---------- Iter 3: mutation main effects ----------
for mut in ["kras_mutation", "nras_mutation", "braf_v600e",
            "msi_high", "her2_amplified", "ntrk_fusion"]:
    results[f"pfs_by_{mut}"] = mean_diff("pfs_months", mut, 1)

# ---------- Iter 4: lab biomarker univariate associations ----------
lab_results = {}
for lab in ["cea_ng_ml", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo",
            "crp_mg_l", "nlr", "hemoglobin_g_dl",
            "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
            "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
            "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl"]:
    r, p = stats.pearsonr(DF[lab], DF["pfs_months"])
    lab_results[lab] = {"r": float(r), "p": float(p)}
results["lab_pfs_pearson"] = lab_results

# ---------- Iter 5: treatment main effects (univariate) ----------
for tx in TX:
    results[f"pfs_by_{tx}"] = mean_diff("pfs_months", tx, 1)

# ---------- Iter 6: multivariable model with all features (no interactions) ----------
covars = (
    ["age_years", "sex_female", "ecog_ps", "stage_iv", "right_sided_primary",
     "kras_mutation", "nras_mutation", "braf_v600e", "msi_high",
     "her2_amplified", "ntrk_fusion",
     "cea_ng_ml", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo",
     "crp_mg_l", "nlr", "hemoglobin_g_dl", "alkaline_phosphatase_u_l",
     "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl", "creatinine_mg_dl",
     "bun_mg_dl", "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl"]
    + TX
)
formula_full = "pfs_months ~ " + " + ".join(covars)
reg(formula_full, "mv_full")

# ---------- Iter 7: interaction screening — each treatment x each biomarker ----------
# We test the interaction coefficient
inter_results = {}
biomarkers = ["kras_mutation", "nras_mutation", "braf_v600e", "msi_high",
              "her2_amplified", "ntrk_fusion", "right_sided_primary",
              "stage_iv", "ecog_ps"]
for tx in TX:
    for bm in biomarkers:
        if DF[tx].nunique() < 2 or DF[bm].nunique() < 2:
            continue
        f = f"pfs_months ~ {tx} * {bm}"
        m = smf.ols(f, data=DF).fit()
        key = f"{tx}__x__{bm}"
        inter = f"{tx}:{bm}"
        if inter in m.params.index:
            inter_results[key] = {
                "coef": float(m.params[inter]),
                "p": float(m.pvalues[inter]),
                "tx_main": float(m.params[tx]),
                "tx_main_p": float(m.pvalues[tx]),
                "bm_main": float(m.params[bm]),
                "bm_main_p": float(m.pvalues[bm]),
            }
results["interaction_screen"] = inter_results

# ---------- Iter 8: subgroup analyses driven by canonical CRC biology ----------
# Cetuximab in RAS/BRAF wild-type
ras_wt_braf_wt = (DF["kras_mutation"] == 0) & (DF["nras_mutation"] == 0) & (DF["braf_v600e"] == 0)
subgroup_effect("treatment_cetuximab", ras_wt_braf_wt, "cetux_in_RASwt_BRAFwt")
subgroup_effect("treatment_cetuximab", DF["kras_mutation"] == 1, "cetux_in_KRASmut")
subgroup_effect("treatment_cetuximab", DF["nras_mutation"] == 1, "cetux_in_NRASmut")
subgroup_effect("treatment_cetuximab", DF["braf_v600e"] == 1, "cetux_in_BRAFmut")
# Add right vs left sidedness within RAS WT
subgroup_effect("treatment_cetuximab", ras_wt_braf_wt & (DF["right_sided_primary"] == 0),
                "cetux_in_RASwt_BRAFwt_LEFT")
subgroup_effect("treatment_cetuximab", ras_wt_braf_wt & (DF["right_sided_primary"] == 1),
                "cetux_in_RASwt_BRAFwt_RIGHT")

# Pembrolizumab in MSI-high vs MSS
subgroup_effect("treatment_pembrolizumab", DF["msi_high"] == 1, "pembro_in_MSI_high")
subgroup_effect("treatment_pembrolizumab", DF["msi_high"] == 0, "pembro_in_MSS")

# Encorafenib in BRAF V600E
subgroup_effect("treatment_encorafenib", DF["braf_v600e"] == 1, "encora_in_BRAFmut")
subgroup_effect("treatment_encorafenib", DF["braf_v600e"] == 0, "encora_in_BRAFwt")

# Trastuzumab/Tucatinib in HER2+
subgroup_effect("treatment_trastuzumab_tucatinib", DF["her2_amplified"] == 1, "trastu_in_HER2pos")
subgroup_effect("treatment_trastuzumab_tucatinib", DF["her2_amplified"] == 0, "trastu_in_HER2neg")

# Bevacizumab — generally any
subgroup_effect("treatment_bevacizumab", pd.Series(True, index=DF.index), "bev_overall")
subgroup_effect("treatment_bevacizumab", DF["right_sided_primary"] == 1, "bev_in_right_sided")
subgroup_effect("treatment_bevacizumab", DF["right_sided_primary"] == 0, "bev_in_left_sided")

# Regorafenib overall
subgroup_effect("treatment_regorafenib", pd.Series(True, index=DF.index), "rego_overall")

# ---------- Iter 9: multivariable model with the canonical biomarker interactions ----------
formula_inter = (
    "pfs_months ~ age_years + sex_female + ecog_ps + stage_iv + right_sided_primary "
    "+ kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified "
    "+ ntrk_fusion + cea_ng_ml + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo "
    "+ crp_mg_l + nlr + hemoglobin_g_dl + alkaline_phosphatase_u_l "
    "+ ast_u_l + alt_u_l + total_bilirubin_mg_dl + creatinine_mg_dl "
    "+ bun_mg_dl + sodium_meq_l + potassium_meq_l + calcium_mg_dl "
    "+ treatment_cetuximab*kras_mutation "
    "+ treatment_cetuximab*nras_mutation "
    "+ treatment_cetuximab*braf_v600e "
    "+ treatment_pembrolizumab*msi_high "
    "+ treatment_encorafenib*braf_v600e "
    "+ treatment_trastuzumab_tucatinib*her2_amplified "
    "+ treatment_bevacizumab + treatment_regorafenib"
)
reg(formula_inter, "mv_with_canonical_interactions")

# ---------- Iter 10: regorafenib heterogeneity sweep ----------
# regorafenib doesn't have an obvious biomarker target — sweep biomarkers
rego_subgroups = {}
for col in ["right_sided_primary", "stage_iv", "kras_mutation",
            "nras_mutation", "braf_v600e", "msi_high",
            "her2_amplified", "ntrk_fusion", "sex_female"]:
    for v in [0, 1]:
        mask = DF[col] == v
        m = smf.ols("pfs_months ~ treatment_regorafenib", data=DF.loc[mask]).fit()
        rego_subgroups[f"{col}={v}"] = {
            "n": int(mask.sum()),
            "coef": float(m.params["treatment_regorafenib"]),
            "p": float(m.pvalues["treatment_regorafenib"]),
        }
# continuous lab thresholds (tertiles)
for lab in ["cea_ng_ml", "albumin_g_dl", "ldh_u_l", "crp_mg_l", "nlr",
            "hemoglobin_g_dl", "weight_loss_pct_6mo", "ecog_ps"]:
    try:
        terts = pd.qcut(DF[lab], 3, labels=["low", "mid", "high"], duplicates="drop")
    except Exception:
        continue
    for level in terts.cat.categories:
        mask = terts == level
        if mask.sum() < 50:
            continue
        m = smf.ols("pfs_months ~ treatment_regorafenib", data=DF.loc[mask]).fit()
        if "treatment_regorafenib" in m.params.index:
            rego_subgroups[f"{lab}_{level}"] = {
                "n": int(mask.sum()),
                "coef": float(m.params["treatment_regorafenib"]),
                "p": float(m.pvalues["treatment_regorafenib"]),
            }
results["rego_subgroups"] = rego_subgroups

# ---------- Iter 11: bevacizumab heterogeneity sweep ----------
bev_subgroups = {}
for col in ["right_sided_primary", "stage_iv", "kras_mutation",
            "nras_mutation", "braf_v600e", "msi_high",
            "her2_amplified", "ntrk_fusion", "sex_female"]:
    for v in [0, 1]:
        mask = DF[col] == v
        m = smf.ols("pfs_months ~ treatment_bevacizumab", data=DF.loc[mask]).fit()
        bev_subgroups[f"{col}={v}"] = {
            "n": int(mask.sum()),
            "coef": float(m.params["treatment_bevacizumab"]),
            "p": float(m.pvalues["treatment_bevacizumab"]),
        }
results["bev_subgroups"] = bev_subgroups

# ---------- Iter 12: deeper subgroup definition for the strongest signals ----------
# Pembrolizumab in MSI high — try adding ECOG / sidedness
pembro_msi_left = subgroup_effect(
    "treatment_pembrolizumab",
    (DF["msi_high"] == 1) & (DF["right_sided_primary"] == 0),
    "pembro_MSIhigh_LEFT",
)
pembro_msi_right = subgroup_effect(
    "treatment_pembrolizumab",
    (DF["msi_high"] == 1) & (DF["right_sided_primary"] == 1),
    "pembro_MSIhigh_RIGHT",
)
pembro_msi_ecog0 = subgroup_effect(
    "treatment_pembrolizumab",
    (DF["msi_high"] == 1) & (DF["ecog_ps"] == 0),
    "pembro_MSIhigh_ECOG0",
)
pembro_msi_ecog_high = subgroup_effect(
    "treatment_pembrolizumab",
    (DF["msi_high"] == 1) & (DF["ecog_ps"] >= 1),
    "pembro_MSIhigh_ECOGge1",
)

# Encorafenib in BRAF — by sidedness, ECOG
subgroup_effect("treatment_encorafenib",
                (DF["braf_v600e"] == 1) & (DF["right_sided_primary"] == 1),
                "encora_BRAF_RIGHT")
subgroup_effect("treatment_encorafenib",
                (DF["braf_v600e"] == 1) & (DF["right_sided_primary"] == 0),
                "encora_BRAF_LEFT")

# Trastuzumab/tucatinib HER2 by RAS
subgroup_effect("treatment_trastuzumab_tucatinib",
                (DF["her2_amplified"] == 1) & (DF["kras_mutation"] == 0)
                & (DF["nras_mutation"] == 0) & (DF["braf_v600e"] == 0),
                "trastu_HER2_RASwt_BRAFwt")
subgroup_effect("treatment_trastuzumab_tucatinib",
                (DF["her2_amplified"] == 1) & (DF["kras_mutation"] == 1),
                "trastu_HER2_KRASmut")

# Cetuximab in RAS/BRAF WT, ECOG 0
subgroup_effect("treatment_cetuximab",
                ras_wt_braf_wt & (DF["ecog_ps"] == 0),
                "cetux_RASwt_BRAFwt_ECOG0")
subgroup_effect("treatment_cetuximab",
                ras_wt_braf_wt & (DF["ecog_ps"] >= 1),
                "cetux_RASwt_BRAFwt_ECOGge1")
subgroup_effect("treatment_cetuximab",
                ras_wt_braf_wt & (DF["right_sided_primary"] == 0) & (DF["ecog_ps"] == 0),
                "cetux_RASwt_BRAFwt_LEFT_ECOG0")

# ---------- Iter 13: final adjusted model in canonical responsive subgroups ----------
def adjusted_subgroup(mask, treatment, label):
    sub = DF.loc[mask].copy()
    if sub[treatment].nunique() < 2 or len(sub) < 50:
        results[label] = {"n": int(len(sub)), "note": "insufficient"}
        return
    f = (
        f"pfs_months ~ {treatment} + age_years + sex_female + ecog_ps + stage_iv "
        "+ cea_ng_ml + albumin_g_dl + ldh_u_l + nlr + crp_mg_l + hemoglobin_g_dl"
    )
    m = smf.ols(f, data=sub).fit()
    results[label] = {
        "n": int(len(sub)),
        "coef": float(m.params[treatment]),
        "se": float(m.bse[treatment]),
        "p": float(m.pvalues[treatment]),
    }

adjusted_subgroup(ras_wt_braf_wt, "treatment_cetuximab", "adj_cetux_RASwt_BRAFwt")
adjusted_subgroup(ras_wt_braf_wt & (DF["right_sided_primary"] == 0),
                  "treatment_cetuximab", "adj_cetux_RASwt_BRAFwt_LEFT")
adjusted_subgroup(DF["msi_high"] == 1, "treatment_pembrolizumab", "adj_pembro_MSIhigh")
adjusted_subgroup(DF["braf_v600e"] == 1, "treatment_encorafenib", "adj_encora_BRAFmut")
adjusted_subgroup(DF["her2_amplified"] == 1, "treatment_trastuzumab_tucatinib", "adj_trastu_HER2pos")

# ---------- Iter 14: secondary checks — features that should/shouldn't matter ----------
# baseline labs adjusted
formula_baseline = (
    "pfs_months ~ age_years + sex_female + ecog_ps + stage_iv + right_sided_primary "
    "+ cea_ng_ml + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr "
    "+ hemoglobin_g_dl + alkaline_phosphatase_u_l + calcium_mg_dl"
)
reg(formula_baseline, "mv_baseline_no_tx")

# Treatments adjusted for baseline (without interactions)
formula_tx_adj = formula_baseline + " + " + " + ".join(TX)
reg(formula_tx_adj, "mv_tx_adjusted_for_baseline")

# ---------- Iter 15: NTRK fusion + larotrectinib-like? we have no NTRK drug, but check NTRK effect alone on PFS ----------
results["pfs_by_ntrk_alone"] = mean_diff("pfs_months", "ntrk_fusion", 1)

# Save
with open("my_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("Wrote my_results.json with", len(results), "top-level keys")
print("Sample keys:", list(results.keys())[:10])
