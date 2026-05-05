"""Analysis for ds001_crc — comprehensive PFS analysis."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
print(f"Loaded {len(df)} patients, {df.shape[1]} columns")

results = {}

def linreg_main(formula, label):
    """Fit OLS, return coef on first non-intercept term."""
    m = smf.ols(formula, data=df).fit()
    return m

def t_test(group_var, outcome="pfs_months"):
    a = df.loc[df[group_var] == 1, outcome]
    b = df.loc[df[group_var] == 0, outcome]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "mean_pos": float(a.mean()),
        "mean_neg": float(b.mean()),
        "diff": float(a.mean() - b.mean()),
        "n_pos": int(len(a)),
        "n_neg": int(len(b)),
        "t": float(t),
        "p": float(p),
    }

# =========================================================
# ITERATION 1: Main effects of demographics & ECOG
# =========================================================
print("\n=== ITERATION 1: demographics / ECOG ===")
it1 = {}
m = smf.ols("pfs_months ~ age_years", data=df).fit()
it1["age"] = {"coef": float(m.params["age_years"]), "p": float(m.pvalues["age_years"]),
              "ci": list(map(float, m.conf_int().loc["age_years"].tolist()))}
print("age coef:", it1["age"])

it1["sex_female"] = t_test("sex_female")
print("sex_female:", it1["sex_female"])

it1["stage_iv"] = t_test("stage_iv")
print("stage_iv:", it1["stage_iv"])

m = smf.ols("pfs_months ~ C(ecog_ps)", data=df).fit()
it1["ecog"] = {
    "coef_1_vs_0": float(m.params.get("C(ecog_ps)[T.1]", np.nan)),
    "p_1_vs_0": float(m.pvalues.get("C(ecog_ps)[T.1]", np.nan)),
    "coef_2_vs_0": float(m.params.get("C(ecog_ps)[T.2]", np.nan)),
    "p_2_vs_0": float(m.pvalues.get("C(ecog_ps)[T.2]", np.nan)),
    "f_p": float(m.f_pvalue),
}
print("ecog:", it1["ecog"])

results["iteration_1"] = it1

# =========================================================
# ITERATION 2: Lab values main effects (univariate)
# =========================================================
print("\n=== ITERATION 2: lab values ===")
it2 = {}
labs = ["cea_ng_ml","albumin_g_dl","ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr",
        "hemoglobin_g_dl","alkaline_phosphatase_u_l","ast_u_l","alt_u_l",
        "total_bilirubin_mg_dl","creatinine_mg_dl","bun_mg_dl",
        "sodium_meq_l","potassium_meq_l","calcium_mg_dl"]
for lab in labs:
    m = smf.ols(f"pfs_months ~ {lab}", data=df).fit()
    it2[lab] = {"coef": float(m.params[lab]), "p": float(m.pvalues[lab]),
                "n": int(m.nobs)}
for k, v in it2.items():
    print(f"  {k}: coef={v['coef']:.4f}, p={v['p']:.3g}")
results["iteration_2"] = it2

# =========================================================
# ITERATION 3: Biomarker / sidedness main effects
# =========================================================
print("\n=== ITERATION 3: biomarkers / sidedness ===")
it3 = {}
for bm in ["right_sided_primary","kras_mutation","nras_mutation","braf_v600e",
           "msi_high","her2_amplified","ntrk_fusion"]:
    it3[bm] = t_test(bm)
    print(f"  {bm}: diff={it3[bm]['diff']:+.3f} p={it3[bm]['p']:.3g}")
results["iteration_3"] = it3

# =========================================================
# ITERATION 4: Treatment main effects (unadjusted)
# =========================================================
print("\n=== ITERATION 4: treatment main effects ===")
it4 = {}
for tx in ["treatment_cetuximab","treatment_bevacizumab","treatment_pembrolizumab",
           "treatment_encorafenib","treatment_trastuzumab_tucatinib","treatment_regorafenib"]:
    it4[tx] = t_test(tx)
    print(f"  {tx}: diff={it4[tx]['diff']:+.3f} p={it4[tx]['p']:.3g}")
results["iteration_4"] = it4

# =========================================================
# ITERATION 5: Multivariable model — features only (no treatment)
# =========================================================
print("\n=== ITERATION 5: multivariable feature model ===")
features = ("age_years + sex_female + C(ecog_ps) + stage_iv + right_sided_primary + "
            "kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified + "
            "ntrk_fusion + cea_ng_ml + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + "
            "crp_mg_l + nlr + hemoglobin_g_dl + alkaline_phosphatase_u_l + ast_u_l + "
            "alt_u_l + total_bilirubin_mg_dl + creatinine_mg_dl + bun_mg_dl + "
            "sodium_meq_l + potassium_meq_l + calcium_mg_dl")
m = smf.ols(f"pfs_months ~ {features}", data=df).fit()
it5 = {"params": {k: float(v) for k, v in m.params.items()},
       "p": {k: float(v) for k, v in m.pvalues.items()},
       "rsquared": float(m.rsquared),
       "n": int(m.nobs)}
print(f"  R^2={it5['rsquared']:.4f}")
for k in m.params.index:
    if k == "Intercept": continue
    print(f"  {k}: coef={m.params[k]:+.4f}, p={m.pvalues[k]:.3g}")
results["iteration_5"] = it5

# =========================================================
# ITERATION 6: All treatments adjusted in single model
# =========================================================
print("\n=== ITERATION 6: treatments adjusted ===")
treats = ("treatment_cetuximab + treatment_bevacizumab + treatment_pembrolizumab + "
          "treatment_encorafenib + treatment_trastuzumab_tucatinib + treatment_regorafenib")
m = smf.ols(f"pfs_months ~ {features} + {treats}", data=df).fit()
it6 = {}
for tx in ["treatment_cetuximab","treatment_bevacizumab","treatment_pembrolizumab",
           "treatment_encorafenib","treatment_trastuzumab_tucatinib","treatment_regorafenib"]:
    it6[tx] = {"coef": float(m.params[tx]), "p": float(m.pvalues[tx]),
               "ci": list(map(float, m.conf_int().loc[tx].tolist()))}
    print(f"  {tx}: adj coef={m.params[tx]:+.4f}, p={m.pvalues[tx]:.3g}")
results["iteration_6"] = it6

# =========================================================
# ITERATION 7-12: Treatment × biomarker interactions (KEY)
# =========================================================
print("\n=== ITERATION 7-12: treatment × biomarker interactions ===")
it_interactions = {}

# Helper for interaction model
def interaction_test(tx, bm, covariate_set=features):
    formula = f"pfs_months ~ {covariate_set} + {tx} * {bm}"
    m = smf.ols(formula, data=df).fit()
    inter = f"{tx}:{bm}"
    res = {
        "main_tx": float(m.params[tx]),
        "main_tx_p": float(m.pvalues[tx]),
        "main_bm": float(m.params[bm]),
        "main_bm_p": float(m.pvalues[bm]),
        "interaction": float(m.params[inter]),
        "interaction_p": float(m.pvalues[inter]),
    }
    # Stratified subgroup effects
    for sub_val in [0, 1]:
        sub = df[df[bm] == sub_val]
        if len(sub) > 50 and sub[tx].sum() > 5:
            mm = smf.ols(f"pfs_months ~ {covariate_set} + {tx}", data=sub).fit()
            res[f"tx_in_{bm}_eq_{sub_val}"] = {
                "coef": float(mm.params[tx]),
                "p": float(mm.pvalues[tx]),
                "n": int(mm.nobs),
                "n_treated": int(sub[tx].sum()),
            }
    return res

# Cetuximab × KRAS, NRAS, BRAF, sidedness
print("\n-- cetuximab interactions --")
it_interactions["cetuximab_x_kras"] = interaction_test("treatment_cetuximab","kras_mutation")
print(f"  KRAS interaction p={it_interactions['cetuximab_x_kras']['interaction_p']:.3g}")
it_interactions["cetuximab_x_nras"] = interaction_test("treatment_cetuximab","nras_mutation")
print(f"  NRAS interaction p={it_interactions['cetuximab_x_nras']['interaction_p']:.3g}")
it_interactions["cetuximab_x_braf"] = interaction_test("treatment_cetuximab","braf_v600e")
print(f"  BRAF interaction p={it_interactions['cetuximab_x_braf']['interaction_p']:.3g}")
it_interactions["cetuximab_x_right"] = interaction_test("treatment_cetuximab","right_sided_primary")
print(f"  right-sided interaction p={it_interactions['cetuximab_x_right']['interaction_p']:.3g}")

# Pembrolizumab × MSI
print("\n-- pembrolizumab × MSI --")
it_interactions["pembro_x_msi"] = interaction_test("treatment_pembrolizumab","msi_high")
print(f"  MSI interaction p={it_interactions['pembro_x_msi']['interaction_p']:.3g}")

# Encorafenib × BRAF
print("\n-- encorafenib × BRAF --")
it_interactions["enco_x_braf"] = interaction_test("treatment_encorafenib","braf_v600e")
print(f"  BRAF interaction p={it_interactions['enco_x_braf']['interaction_p']:.3g}")

# Trastuzumab/tucatinib × HER2
print("\n-- trastuzumab/tucatinib × HER2 --")
it_interactions["tt_x_her2"] = interaction_test("treatment_trastuzumab_tucatinib","her2_amplified")
print(f"  HER2 interaction p={it_interactions['tt_x_her2']['interaction_p']:.3g}")

# Bevacizumab — no canonical biomarker, but check sidedness, KRAS
print("\n-- bevacizumab interactions --")
it_interactions["bev_x_kras"] = interaction_test("treatment_bevacizumab","kras_mutation")
print(f"  KRAS interaction p={it_interactions['bev_x_kras']['interaction_p']:.3g}")
it_interactions["bev_x_right"] = interaction_test("treatment_bevacizumab","right_sided_primary")
print(f"  right-sided interaction p={it_interactions['bev_x_right']['interaction_p']:.3g}")

# Regorafenib — broad
print("\n-- regorafenib interactions --")
it_interactions["rego_x_kras"] = interaction_test("treatment_regorafenib","kras_mutation")
print(f"  KRAS interaction p={it_interactions['rego_x_kras']['interaction_p']:.3g}")

results["interactions"] = it_interactions

# =========================================================
# Save partial results so far
# =========================================================
with open("analysis_results_full.json","w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nPartial results saved.")
