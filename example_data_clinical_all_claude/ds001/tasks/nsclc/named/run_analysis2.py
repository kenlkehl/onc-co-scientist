"""Phase-2 deep dive into sotorasib subgroup heterogeneity (the dominant signal),
plus systematic verification that pembro/osi/olap have no biomarker-interaction signal."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
df["adeno"] = (df["histology"] == "adenocarcinoma").astype(int)
df["smoke_current"] = (df["smoking_status"] == "current").astype(int)
df["smoke_former"] = (df["smoking_status"] == "former").astype(int)
df["smoke_never"] = (df["smoking_status"] == "never").astype(int)
df["pdl1_high"] = (df["pdl1_tps"] >= 0.5).astype(int)

results = {}

def add(name, *, est, p, desc, sig=None):
    if sig is None and p is not None:
        sig = bool(p < 0.05)
    results[name] = {
        "effect_estimate": float(est) if est is not None else None,
        "p_value": float(p) if p is not None else None,
        "significant": sig,
        "result_summary": desc,
    }

def subgroup_treatment_effect(name, sub, tx, desc_template):
    a = sub.loc[sub[tx] == 1, "pfs_months"]
    b = sub.loc[sub[tx] == 0, "pfs_months"]
    if len(a) < 5 or len(b) < 5:
        add(name, est=None, p=None, desc=f"Insufficient data for {name}", sig=None)
        return None
    t = stats.ttest_ind(a, b, equal_var=False)
    diff = float(a.mean() - b.mean())
    desc = desc_template.format(
        n_a=len(a), m_a=a.mean(), n_b=len(b), m_b=b.mean(), diff=diff, p=t.pvalue
    )
    add(name, est=diff, p=float(t.pvalue), desc=desc)
    return diff, t.pvalue

# ============================================================
# Soto x sex_female interaction (in KRAS+ subset)
# ============================================================
sub = df[df["kras_g12c"] == 1]
m = smf.ols("pfs_months ~ treatment_sotorasib * sex_female", data=sub).fit()
term = "treatment_sotorasib:sex_female"
add(
    "soto_x_sex_in_kras",
    est=float(m.params[term]),
    p=float(m.pvalues[term]),
    desc=(f"In KRAS G12C+ patients only (n={len(sub)}), OLS PFS ~ sotorasib * sex_female: "
          f"interaction coef={m.params[term]:.4f} (SE={m.bse[term]:.4f}), p={m.pvalues[term]:.3g}; "
          f"main soto coef={m.params['treatment_sotorasib']:.4f}, main sex_female coef={m.params['sex_female']:.4f}."),
)

# Soto effect within KRAS+ males vs KRAS+ females
sub_m = df[(df["kras_g12c"] == 1) & (df["sex_female"] == 0)]
subgroup_treatment_effect(
    "soto_kras_male",
    sub_m, "treatment_sotorasib",
    "KRAS G12C+ AND male subgroup: soto (n={n_a}) PFS={m_a:.3f} mo vs no-soto (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)
sub_f = df[(df["kras_g12c"] == 1) & (df["sex_female"] == 1)]
subgroup_treatment_effect(
    "soto_kras_female",
    sub_f, "treatment_sotorasib",
    "KRAS G12C+ AND female subgroup: soto (n={n_a}) PFS={m_a:.3f} mo vs no-soto (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# Soto in KRAS+ men vs KRAS+ men with smoking history
for smoke in ["current", "former", "never"]:
    sub = df[(df["kras_g12c"] == 1) & (df["sex_female"] == 0) & (df["smoking_status"] == smoke)]
    subgroup_treatment_effect(
        f"soto_kras_male_smoke_{smoke}",
        sub, "treatment_sotorasib",
        f"KRAS G12C+ AND male AND {smoke} smoker subgroup: soto (n={{n_a}}) PFS={{m_a:.3f}} mo vs no-soto (n={{n_b}}) PFS={{m_b:.3f}} mo; diff={{diff:.3f}}, p={{p:.3g}}.",
    )

# ============================================================
# Systematic interaction screen for sotorasib *within* KRAS+ subset
# ============================================================
sub = df[df["kras_g12c"] == 1]
features_to_test = [
    "sex_female", "stage_iv", "has_brain_mets", "stk11_mutation",
    "alk_fusion", "brca2_mutation", "tmb_high", "adeno", "smoke_current",
    "smoke_former", "egfr_mutation", "pdl1_high", "ecog_ps"
]
soto_kras_screen = []
for f in features_to_test:
    try:
        mm = smf.ols(f"pfs_months ~ treatment_sotorasib * {f}", data=sub).fit()
        term = f"treatment_sotorasib:{f}"
        if term in mm.params:
            soto_kras_screen.append({
                "feature": f,
                "coef": float(mm.params[term]),
                "se": float(mm.bse[term]),
                "p": float(mm.pvalues[term]),
            })
    except Exception:
        pass
soto_kras_screen.sort(key=lambda r: r["p"])
for r in soto_kras_screen[:8]:
    add(
        f"soto_x_{r['feature']}_in_kras",
        est=r["coef"], p=r["p"],
        desc=(f"Within KRAS G12C+ subset only (n={len(sub)}), OLS PFS ~ sotorasib * {r['feature']}: "
              f"interaction coef={r['coef']:.4f} (SE={r['se']:.4f}), p={r['p']:.3g}."),
    )

# ============================================================
# Best 2- and 3-feature subgroups for sotorasib (try to find the cleanest responder profile)
# ============================================================
sub = df[df["kras_g12c"] == 1]
best_3way = []
secondary = ["sex_female", "stage_iv", "has_brain_mets", "stk11_mutation",
             "alk_fusion", "brca2_mutation", "tmb_high", "adeno", "smoke_current",
             "egfr_mutation", "pdl1_high"]
# pairs of (feat1, val1, feat2, val2)
for i, f1 in enumerate(secondary):
    for v1 in [0, 1]:
        for f2 in secondary[i+1:]:
            for v2 in [0, 1]:
                mask = (sub[f1] == v1) & (sub[f2] == v2)
                if mask.sum() < 100:
                    continue
                ssub = sub[mask]
                a = ssub.loc[ssub["treatment_sotorasib"] == 1, "pfs_months"]
                b = ssub.loc[ssub["treatment_sotorasib"] == 0, "pfs_months"]
                if len(a) < 30 or len(b) < 30:
                    continue
                t = stats.ttest_ind(a, b, equal_var=False)
                best_3way.append({
                    "f1": f1, "v1": v1, "f2": f2, "v2": v2,
                    "n_tx": int(len(a)), "n_ctrl": int(len(b)),
                    "diff": float(a.mean() - b.mean()),
                    "p": float(t.pvalue),
                })
best_3way.sort(key=lambda r: -r["diff"])

# Save top 5
for i, r in enumerate(best_3way[:5]):
    add(
        f"soto_kras_{r['f1']}{r['v1']}_{r['f2']}{r['v2']}",
        est=r["diff"], p=r["p"],
        desc=(f"Sotorasib in KRAS G12C+ AND {r['f1']}={r['v1']} AND {r['f2']}={r['v2']} subgroup "
              f"(n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}): PFS difference = {r['diff']:.3f} mo, p={r['p']:.3g}."),
    )

# ============================================================
# Confirm null biomarker effects with continuous-PDL1 interactions for pembro
# ============================================================
# Also test pembro interactions in subsets
m = smf.ols("pfs_months ~ treatment_pembrolizumab * pdl1_tps + treatment_pembrolizumab * tmb_high", data=df).fit()
add(
    "pembro_pdl1_and_tmb_joint",
    est=float(m.params["treatment_pembrolizumab:pdl1_tps"]),
    p=float(m.pvalues["treatment_pembrolizumab:pdl1_tps"]),
    desc=(f"Joint OLS PFS ~ pembro*pdl1_tps + pembro*tmb_high: pembro×pdl1_tps coef={m.params['treatment_pembrolizumab:pdl1_tps']:.4f}, "
          f"p={m.pvalues['treatment_pembrolizumab:pdl1_tps']:.3g}; pembro×tmb_high coef={m.params['treatment_pembrolizumab:tmb_high']:.4f}, "
          f"p={m.pvalues['treatment_pembrolizumab:tmb_high']:.3g}. Effect estimate reported is pembro×pdl1_tps."),
)

# Verify osimertinib has NO benefit even in subgroups
sub = df[(df["egfr_mutation"] == 1) & (df["ecog_ps"] <= 1) & (df["stage_iv"] == 0)]
subgroup_treatment_effect(
    "osi_egfr_ecoglow_nostageiv",
    sub, "treatment_osimertinib",
    "EGFR+ AND ECOG 0-1 AND not stage IV subgroup: osi (n={n_a}) PFS={m_a:.3f} mo vs no-osi (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# Verify olaparib has NO benefit anywhere by 3-way subgroup search
olap_search = []
for f in ["brca2_mutation", "stk11_mutation", "tmb_high", "adeno", "stage_iv", "has_brain_mets", "ecog_ps", "egfr_mutation"]:
    for v in [0, 1, 2]:
        if f == "ecog_ps" and v not in [0, 1, 2]:
            continue
        if f != "ecog_ps" and v not in [0, 1]:
            continue
        sub = df[df[f] == v]
        if len(sub) < 100:
            continue
        a = sub.loc[sub["treatment_olaparib"] == 1, "pfs_months"]
        b = sub.loc[sub["treatment_olaparib"] == 0, "pfs_months"]
        if len(a) < 30 or len(b) < 30:
            continue
        t = stats.ttest_ind(a, b, equal_var=False)
        olap_search.append({"f": f, "v": v, "n_tx": len(a), "n_ctrl": len(b),
                            "diff": float(a.mean() - b.mean()), "p": float(t.pvalue)})
olap_search.sort(key=lambda r: r["p"])
for r in olap_search[:5]:
    add(
        f"olap_subgroup_{r['f']}_{r['v']}",
        est=r["diff"], p=r["p"],
        desc=(f"Olaparib subgroup search: in {r['f']}={r['v']} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}), "
              f"PFS difference = {r['diff']:.3f} mo, p={r['p']:.3g}."),
    )

# Same for osimertinib
osi_search = []
for f in ["egfr_mutation", "stk11_mutation", "tmb_high", "adeno", "stage_iv", "has_brain_mets", "alk_fusion"]:
    for v in [0, 1]:
        sub = df[df[f] == v]
        if len(sub) < 100:
            continue
        a = sub.loc[sub["treatment_osimertinib"] == 1, "pfs_months"]
        b = sub.loc[sub["treatment_osimertinib"] == 0, "pfs_months"]
        if len(a) < 30 or len(b) < 30:
            continue
        t = stats.ttest_ind(a, b, equal_var=False)
        osi_search.append({"f": f, "v": v, "n_tx": len(a), "n_ctrl": len(b),
                           "diff": float(a.mean() - b.mean()), "p": float(t.pvalue)})
osi_search.sort(key=lambda r: r["p"])
for r in osi_search[:5]:
    add(
        f"osi_subgroup_{r['f']}_{r['v']}",
        est=r["diff"], p=r["p"],
        desc=(f"Osimertinib subgroup search: in {r['f']}={r['v']} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}), "
              f"PFS difference = {r['diff']:.3f} mo, p={r['p']:.3g}."),
    )

# Same for pembro
pembro_search = []
for f in ["pdl1_high", "tmb_high", "stk11_mutation", "adeno", "stage_iv", "has_brain_mets",
          "egfr_mutation", "kras_g12c", "alk_fusion", "brca2_mutation", "smoke_current",
          "smoke_former", "smoke_never", "sex_female"]:
    for v in [0, 1]:
        sub = df[df[f] == v]
        if len(sub) < 100:
            continue
        a = sub.loc[sub["treatment_pembrolizumab"] == 1, "pfs_months"]
        b = sub.loc[sub["treatment_pembrolizumab"] == 0, "pfs_months"]
        if len(a) < 30 or len(b) < 30:
            continue
        t = stats.ttest_ind(a, b, equal_var=False)
        pembro_search.append({"f": f, "v": v, "n_tx": len(a), "n_ctrl": len(b),
                              "diff": float(a.mean() - b.mean()), "p": float(t.pvalue)})
pembro_search.sort(key=lambda r: r["p"])
for r in pembro_search[:5]:
    add(
        f"pembro_subgroup_{r['f']}_{r['v']}",
        est=r["diff"], p=r["p"],
        desc=(f"Pembrolizumab subgroup search: in {r['f']}={r['v']} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}), "
              f"PFS difference = {r['diff']:.3f} mo, p={r['p']:.3g}."),
    )

# ============================================================
# Final adjusted "best subgroup" sotorasib model
# ============================================================
# Adjusted soto effect within best subgroup
sub = df[(df["kras_g12c"] == 1) & (df["sex_female"] == 0)]
m = smf.ols(("pfs_months ~ treatment_sotorasib + age_years + ecog_ps + stage_iv + has_brain_mets + adeno + "
             "smoke_current + smoke_former + albumin_g_dl + ldh_u_l + nlr + stk11_mutation + alk_fusion"), data=sub).fit()
add(
    "soto_kras_male_adj",
    est=float(m.params["treatment_sotorasib"]),
    p=float(m.pvalues["treatment_sotorasib"]),
    desc=(f"Adjusted OLS in KRAS G12C+ AND male subgroup (n={len(sub)}): "
          f"sotorasib coef={m.params['treatment_sotorasib']:.4f} mo (SE={m.bse['treatment_sotorasib']:.4f}), p={m.pvalues['treatment_sotorasib']:.3g}, "
          f"controlling for age, ECOG, stage IV, brain mets, histology, smoking, albumin, LDH, NLR, STK11, ALK."),
)

# Three-way: KRAS+ AND male AND no-ALK-fusion
sub = df[(df["kras_g12c"] == 1) & (df["sex_female"] == 0) & (df["alk_fusion"] == 0)]
subgroup_treatment_effect(
    "soto_kras_male_noalk",
    sub, "treatment_sotorasib",
    "KRAS G12C+ AND male AND no ALK fusion subgroup: soto (n={n_a}) PFS={m_a:.3f} mo vs no-soto (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# Whole "expected vs unexpected" pattern check: how does soto perform in KRAS- male subgroup?
sub = df[(df["kras_g12c"] == 0) & (df["sex_female"] == 0)]
subgroup_treatment_effect(
    "soto_kraswt_male",
    sub, "treatment_sotorasib",
    "KRAS G12C-WT AND male subgroup: soto (n={n_a}) PFS={m_a:.3f} mo vs no-soto (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# Combine KRAS+ AND male AND no-ALK AND smoking_history (current/former)
sub = df[(df["kras_g12c"] == 1) & (df["sex_female"] == 0) & (df["alk_fusion"] == 0) & (df["smoking_status"] != "never")]
subgroup_treatment_effect(
    "soto_kras_male_noalk_smoker",
    sub, "treatment_sotorasib",
    "KRAS G12C+ AND male AND no ALK AND smoker (former or current): soto (n={n_a}) PFS={m_a:.3f} mo vs no-soto (n={n_b}) PFS={m_b:.3f} mo; diff={diff:.3f}, p={p:.3g}.",
)

# Save
out = {"results": results, "best_3way_top20": best_3way[:20], "soto_kras_screen": soto_kras_screen,
       "olap_search": olap_search, "osi_search": osi_search, "pembro_search": pembro_search}
with open("analysis_results2.json", "w") as f:
    json.dump(out, f, indent=2)

# Print summary
print(f"Wrote {len(results)} additional analyses.")
print()
print("=== Top 10 best soto subgroups (by largest +diff) ===")
for r in best_3way[:10]:
    print(f"  KRAS+ & {r['f1']}={r['v1']} & {r['f2']}={r['v2']}: n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}, diff={r['diff']:.3f}, p={r['p']:.3g}")

print()
print("=== Soto x feature interactions in KRAS+ subset ===")
for r in soto_kras_screen[:5]:
    print(f"  {r['feature']}: coef={r['coef']:.4f}, p={r['p']:.3g}")
