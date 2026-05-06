"""Deeper analyses focusing on regorafenib heterogeneity and combined subgroups."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
DF = pd.read_parquet("dataset.parquet")

results = {}


def sub(mask, tx, label):
    s = DF.loc[mask]
    if s[tx].nunique() < 2 or len(s) < 30:
        results[label] = {"n": int(len(s)), "note": "insufficient"}
        return
    m = smf.ols(f"pfs_months ~ {tx}", data=s).fit()
    results[label] = {
        "n": int(len(s)),
        "n_treated": int(s[tx].sum()),
        "coef": float(m.params[tx]),
        "p": float(m.pvalues[tx]),
        "mean_treated": float(s.loc[s[tx] == 1, "pfs_months"].mean()),
        "mean_untreated": float(s.loc[s[tx] == 0, "pfs_months"].mean()),
    }


# --- regorafenib in deeper combinations ---
ras_wt = (DF["kras_mutation"] == 0) & (DF["nras_mutation"] == 0) & (DF["braf_v600e"] == 0)
left = DF["right_sided_primary"] == 0

sub(ras_wt, "treatment_regorafenib", "rego_RASwt_BRAFwt")
sub(ras_wt & left, "treatment_regorafenib", "rego_RASwt_BRAFwt_LEFT")
sub(left, "treatment_regorafenib", "rego_LEFT_only")
sub(left & (DF["kras_mutation"] == 0), "treatment_regorafenib", "rego_LEFT_KRASwt")
sub(left & (DF["kras_mutation"] == 0) & (DF["braf_v600e"] == 0),
    "treatment_regorafenib", "rego_LEFT_KRASwt_BRAFwt")

# split CEA within rego-responsive subgroup
cea_med = DF["cea_ng_ml"].median()
sub(ras_wt & left & (DF["cea_ng_ml"] <= cea_med),
    "treatment_regorafenib", "rego_RASwt_BRAFwt_LEFT_CEAlow")
sub(ras_wt & left & (DF["cea_ng_ml"] > cea_med),
    "treatment_regorafenib", "rego_RASwt_BRAFwt_LEFT_CEAhigh")

# CEA tertile high cutoff
ceat = pd.qcut(DF["cea_ng_ml"], 3, labels=["low", "mid", "high"], duplicates="drop")
sub(ras_wt & left & (ceat == "high"), "treatment_regorafenib", "rego_RASwt_BRAFwt_LEFT_CEAhigh_tertile")
sub(ras_wt & left & (ceat != "high"), "treatment_regorafenib", "rego_RASwt_BRAFwt_LEFT_CEAlowmid")
sub(ras_wt & left & (ceat == "low"), "treatment_regorafenib", "rego_RASwt_BRAFwt_LEFT_CEAlow_tertile")

# Look at regorafenib by CEA across the full cohort, since CEA-high seems to lose benefit
sub(DF["cea_ng_ml"] > DF["cea_ng_ml"].quantile(0.66),
    "treatment_regorafenib", "rego_overall_CEAhigh_tertile")
sub(DF["cea_ng_ml"] <= DF["cea_ng_ml"].quantile(0.66),
    "treatment_regorafenib", "rego_overall_CEAnonhigh")

# adjusted regorafenib effect within key subgroups
def adj_sub(mask, tx, label):
    s = DF.loc[mask].copy()
    if s[tx].nunique() < 2 or len(s) < 50:
        results[label] = {"n": int(len(s)), "note": "insufficient"}
        return
    f = (
        f"pfs_months ~ {tx} + age_years + sex_female + ecog_ps + stage_iv "
        "+ cea_ng_ml + albumin_g_dl + ldh_u_l + nlr + crp_mg_l + hemoglobin_g_dl"
    )
    m = smf.ols(f, data=s).fit()
    results[label] = {
        "n": int(len(s)),
        "coef": float(m.params[tx]),
        "p": float(m.pvalues[tx]),
    }

adj_sub(ras_wt & left, "treatment_regorafenib", "adj_rego_RASwt_BRAFwt_LEFT")
adj_sub(ras_wt & left & (ceat != "high"), "treatment_regorafenib",
        "adj_rego_RASwt_BRAFwt_LEFT_CEAnonhigh")

# 3-way interaction: regorafenib x KRAS x right
m = smf.ols("pfs_months ~ treatment_regorafenib * kras_mutation * right_sided_primary "
            "+ braf_v600e + age_years + ecog_ps + stage_iv + albumin_g_dl + cea_ng_ml",
            data=DF).fit()
threeway = {}
for k in m.params.index:
    threeway[k] = {"coef": float(m.params[k]), "p": float(m.pvalues[k])}
results["rego_threeway_KRAS_RIGHT"] = threeway

# joint interaction model with all the modifiers we found
m2 = smf.ols("pfs_months ~ treatment_regorafenib*kras_mutation "
             "+ treatment_regorafenib*braf_v600e "
             "+ treatment_regorafenib*right_sided_primary "
             "+ treatment_regorafenib*cea_ng_ml "
             "+ age_years + sex_female + ecog_ps + stage_iv + albumin_g_dl "
             "+ nras_mutation + msi_high + her2_amplified + ntrk_fusion",
             data=DF).fit()
joint = {}
for k in m2.params.index:
    joint[k] = {"coef": float(m2.params[k]), "p": float(m2.pvalues[k])}
results["rego_joint_modifier_model"] = joint

# Final candidate: regorafenib in KRAS-WT, BRAF-WT, left-sided, CEA non-high tertile
candidate_mask = (DF["kras_mutation"] == 0) & (DF["braf_v600e"] == 0) \
                 & (DF["right_sided_primary"] == 0) & (ceat != "high")
sub(candidate_mask, "treatment_regorafenib", "rego_FINAL_KRASwt_BRAFwt_LEFT_CEAnonhigh")
adj_sub(candidate_mask, "treatment_regorafenib",
        "adj_rego_FINAL_KRASwt_BRAFwt_LEFT_CEAnonhigh")

# Also: include NRAS — does NRAS modify too?
m3 = smf.ols("pfs_months ~ treatment_regorafenib * nras_mutation + kras_mutation + braf_v600e "
             "+ right_sided_primary", data=DF).fit()
results["rego_nras_check"] = {
    "interaction_coef": float(m3.params["treatment_regorafenib:nras_mutation"]),
    "interaction_p": float(m3.pvalues["treatment_regorafenib:nras_mutation"]),
}

# NRAS positive interaction with rego — explore
sub(DF["nras_mutation"] == 1, "treatment_regorafenib", "rego_NRASmut")
sub((DF["nras_mutation"] == 1) & (DF["kras_mutation"] == 0), "treatment_regorafenib",
    "rego_NRASmut_KRASwt")
sub((DF["nras_mutation"] == 0) & (DF["kras_mutation"] == 0) & (DF["braf_v600e"] == 0)
    & (DF["right_sided_primary"] == 0),
    "treatment_regorafenib", "rego_NRASwt_KRASwt_BRAFwt_LEFT")

# Check if cetuximab works in the same RAS WT, left-sided, CEA non-high group — to confirm
# that anti-EGFR truly has no effect even where it should
sub(ras_wt & left & (ceat != "high"), "treatment_cetuximab", "cetux_RASwt_BRAFwt_LEFT_CEAnonhigh")

# Save
with open("my_results2.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("Wrote my_results2.json")
