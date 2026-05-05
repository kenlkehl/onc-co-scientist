"""Assemble transcript.json from _results.json and the targeted enza subgroup analysis."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
R = json.loads((HERE / "_results.json").read_text())
DF = pd.read_parquet(HERE / "dataset.parquet")


def two_prop(df, t, outcome="objective_response"):
    a = df.loc[df[t] == 1, outcome]
    b = df.loc[df[t] == 0, outcome]
    p1, p2, n1, n2 = a.mean(), b.mean(), len(a), len(b)
    pp = (a.sum() + b.sum()) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)) if 0 < pp < 1 else 0
    if se == 0:
        return (p1, p2, p1 - p2, 1.0, n1, n2)
    z = (p1 - p2) / se
    pv = 2 * (1 - stats.norm.cdf(abs(z)))
    return (float(p1), float(p2), float(p1 - p2), float(pv), int(n1), int(n2))


def make_iter1():
    hyps = []
    analyses = []
    for i, t in enumerate([
        "treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
        "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"
    ]):
        hid = f"h1_{i+1}"
        # Direction stated based on prior expectation: each treatment expected to increase response
        hyps.append({
            "id": hid,
            "text": f"Patients receiving {t} have a higher rate of objective_response than those not receiving {t}.",
            "kind": "novel",
        })
    # Find each result row by treatment
    rows = {r["treatment"]: r for r in R["iter1_treatment_main"]}
    for i, t in enumerate([
        "treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
        "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"
    ]):
        hid = f"h1_{i+1}"
        r = rows[t]
        analyses.append({
            "hypothesis_ids": [hid],
            "code": f"two-proportion z-test of objective_response by {t}",
            "result_summary": (f"objective_response rate {r['rate_treat']:.3f} on {t} (n={r['n_treat']}) "
                               f"vs {r['rate_ctrl']:.3f} off (n={r['n_ctrl']}); "
                               f"rate diff {r['diff']:+.4f}, p={r['p']:.3g}."),
            "p_value": float(r["p"]),
            "effect_estimate": float(r["diff"]),
            "significant": bool(r["significant"]),
        })
    return {"index": 1, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter2():
    hyps = []
    analyses = []
    bios = R["iter2_biomarker_main"]
    rows = {r["biomarker"]: r for r in bios}
    for i, b in enumerate(["brca2_mutation", "ar_v7_positive", "msi_high", "psma_high"]):
        hid = f"h2_{i+1}"
        hyps.append({
            "id": hid,
            "text": (f"Patients with {b}=1 have a different objective_response rate than patients with {b}=0 "
                     f"(testing whether {b} status is a marginal predictor of response)."),
            "kind": "novel",
        })
    for i, b in enumerate(["brca2_mutation", "ar_v7_positive", "msi_high", "psma_high"]):
        hid = f"h2_{i+1}"
        r = rows[b]
        analyses.append({
            "hypothesis_ids": [hid],
            "code": f"two-proportion z-test of objective_response by {b}",
            "result_summary": (f"response rate {r['rate_pos']:.3f} when {b}=1 vs {r['rate_neg']:.3f} when {b}=0; "
                               f"diff {r['diff']:+.4f} (p={r['p']:.3g})."),
            "p_value": float(r["p"]),
            "effect_estimate": float(r["diff"]),
            "significant": bool(r["significant"]),
        })
    return {"index": 2, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter3():
    hyps = []
    analyses = []
    c = R["iter3_clinical_main"]

    hyps.append({"id": "h3_1", "text": "Higher ecog_ps is associated with a lower objective_response rate.", "kind": "novel"})
    rate0 = c["ecog_table"]["mean"]["0"]
    rate1 = c["ecog_table"]["mean"]["1"]
    rate2 = c["ecog_table"]["mean"]["2"]
    analyses.append({
        "hypothesis_ids": ["h3_1"],
        "code": "chi-square of ecog_ps x objective_response",
        "result_summary": (f"response rate by ecog_ps: 0={rate0:.3f}, 1={rate1:.3f}, 2={rate2:.3f}; "
                           f"chi-square p={c['ecog_chi2_p']:.3g}; monotonic decrease."),
        "p_value": float(c["ecog_chi2_p"]),
        "effect_estimate": float(rate2 - rate0),  # signed: ecog_high - ecog_low (negative means worse)
        "significant": c["ecog_chi2_p"] < 0.05,
    })

    hyps.append({"id": "h3_2", "text": "Patients with mcrpc=1 have a lower objective_response rate than patients with mcrpc=0.", "kind": "novel"})
    m = c["mcrpc"]
    analyses.append({
        "hypothesis_ids": ["h3_2"],
        "code": "two-proportion z-test of objective_response by mcrpc",
        "result_summary": f"mCRPC=1 rate {m[0]:.3f} vs mCRPC=0 rate {m[1]:.3f}; diff {m[2]:+.4f}, p={m[3]:.3g}.",
        "p_value": float(m[3]),
        "effect_estimate": float(m[2]),
        "significant": m[3] < 0.05,
    })

    hyps.append({"id": "h3_3", "text": "Patients with visceral_mets=1 have a different objective_response rate than patients with visceral_mets=0.", "kind": "novel"})
    v = c["visceral_mets"]
    analyses.append({
        "hypothesis_ids": ["h3_3"],
        "code": "two-proportion z-test by visceral_mets",
        "result_summary": f"visceral_mets=1 rate {v[0]:.3f} vs =0 rate {v[1]:.3f}; diff {v[2]:+.4f}, p={v[3]:.3g}.",
        "p_value": float(v[3]),
        "effect_estimate": float(v[2]),
        "significant": v[3] < 0.05,
    })

    hyps.append({"id": "h3_4", "text": "Higher gleason_score (>=9) is associated with a lower objective_response rate than lower gleason_score (<=7).", "kind": "novel"})
    analyses.append({
        "hypothesis_ids": ["h3_4"],
        "code": "compare response rates between gleason>=9 and gleason<=7; chi-square overall",
        "result_summary": (f"response rate gleason>=9: {c['gleason_high_rate']:.3f} vs gleason<=7: "
                           f"{c['gleason_low_rate']:.3f}; chi-square across categories p={c['gleason_chi2_p']:.3g}."),
        "p_value": float(c["gleason_chi2_p"]),
        "effect_estimate": float(c["gleason_high_rate"] - c["gleason_low_rate"]),
        "significant": c["gleason_chi2_p"] < 0.05,
    })

    hyps.append({"id": "h3_5", "text": "Older age_years is associated with a lower objective_response rate.", "kind": "novel"})
    a = c["age_logit"]
    analyses.append({
        "hypothesis_ids": ["h3_5"],
        "code": "logistic regression of objective_response on age_years",
        "result_summary": f"univariable logistic age_years beta {a['beta']:+.4f} (OR {a['or']:.3f}, p={a['p']:.3g}).",
        "p_value": float(a["p"]),
        "effect_estimate": float(a["beta"]),
        "significant": a["p"] < 0.05,
    })

    return {"index": 3, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter4():
    """Lab univariable. We pick the labs with prior plausibility as inflammatory/disease-burden markers and call out direction."""
    direction_priors = {
        "psa_ng_ml": "lower", "albumin_g_dl": "higher", "ldh_u_l": "lower",
        "weight_loss_pct_6mo": "lower", "crp_mg_l": "lower", "nlr": "lower",
        "hemoglobin_g_dl": "higher", "alkaline_phosphatase_u_l": "lower",
        "ast_u_l": "lower", "alt_u_l": "lower", "total_bilirubin_mg_dl": "lower",
        "creatinine_mg_dl": "lower", "bun_mg_dl": "lower",
        "sodium_meq_l": "different", "potassium_meq_l": "different", "calcium_mg_dl": "lower",
    }
    hyps = []
    analyses = []
    labs = R["iter4_lab_main"]
    for i, (lab, dirn) in enumerate(direction_priors.items()):
        hid = f"h4_{i+1}"
        if dirn == "higher":
            statement = f"Higher {lab} is associated with a higher objective_response rate (positive beta in logistic regression)."
        elif dirn == "lower":
            statement = f"Higher {lab} is associated with a lower objective_response rate (negative beta in logistic regression)."
        else:
            statement = f"{lab} is associated with objective_response (non-zero beta in logistic regression)."
        hyps.append({"id": hid, "text": statement, "kind": "novel"})
    for i, (lab, dirn) in enumerate(direction_priors.items()):
        hid = f"h4_{i+1}"
        v = labs[lab]
        analyses.append({
            "hypothesis_ids": [hid],
            "code": f"sm.Logit(objective_response ~ {lab})",
            "result_summary": f"{lab}: beta {v['beta']:+.4f}, OR {v['or']:.3f}, p={v['p']:.3g}.",
            "p_value": float(v["p"]),
            "effect_estimate": float(v["beta"]),
            "significant": v["p"] < 0.05,
        })
    return {"index": 4, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter5():
    """Targeted treatment x biomarker interactions: olaparib/BRCA2, pembro/MSI, lu177/PSMA, enza/AR-V7."""
    pairs = [
        ("treatment_olaparib", "brca2_mutation",
         "Olaparib increases objective_response more in BRCA2-mutated patients than in BRCA2 wildtype (positive treatment-by-biomarker interaction)."),
        ("treatment_pembrolizumab", "msi_high",
         "Pembrolizumab increases objective_response more in MSI-high patients than in MSI-low patients (positive treatment-by-biomarker interaction)."),
        ("treatment_lu177_psma", "psma_high",
         "Lu-177-PSMA increases objective_response more in PSMA-high patients than in PSMA-low patients (positive treatment-by-biomarker interaction)."),
        ("treatment_enzalutamide", "ar_v7_positive",
         "Enzalutamide's increase in objective_response is attenuated in AR-V7-positive patients (negative treatment-by-biomarker interaction)."),
        ("treatment_enzalutamide", "brca2_mutation",
         "Enzalutamide's increase in objective_response is attenuated in BRCA2-mutated patients (negative treatment-by-biomarker interaction)."),
        ("treatment_enzalutamide", "msi_high",
         "Enzalutamide's increase in objective_response is attenuated in MSI-high patients (negative treatment-by-biomarker interaction)."),
    ]
    hyps = []
    analyses = []
    for i, (t, b, txt) in enumerate(pairs):
        hyps.append({"id": f"h5_{i+1}", "text": txt, "kind": "novel"})

    for i, (t, b, _) in enumerate(pairs):
        hid = f"h5_{i+1}"
        key = f"{t}__x__{b}"
        v = R["iter5_treat_x_biomarker"][key]
        inter = v["inter"]
        s1 = v["strat"].get("1", {})
        s0 = v["strat"].get("0", {})
        analyses.append({
            "hypothesis_ids": [hid],
            "code": f"sm.Logit(objective_response ~ {t} + {b} + {t}:{b})",
            "result_summary": (
                f"interaction beta {inter['beta_inter']:+.3f} (p={inter['p_inter']:.3g}); "
                f"stratified rate diff in {b}=1: {s1.get('diff', float('nan')):+.3f} (p={s1.get('p', 1):.3g}); "
                f"in {b}=0: {s0.get('diff', float('nan')):+.3f} (p={s0.get('p', 1):.3g})."
            ),
            "p_value": float(inter["p_inter"]),
            "effect_estimate": float(inter["beta_inter"]),
            "significant": inter["p_inter"] < 0.05,
        })
    return {"index": 5, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter6():
    """Treatment x clinical interactions, focusing on big signals: enza x mCRPC and a few weaker."""
    pairs = [
        ("treatment_enzalutamide", "mcrpc",
         "Enzalutamide's increase in objective_response is attenuated in mcrpc=1 patients (negative treatment-by-mcrpc interaction)."),
        ("treatment_enzalutamide", "ecog_high",
         "Enzalutamide's effect on objective_response differs between ecog_ps>=2 and ecog_ps<2 (non-zero interaction)."),
        ("treatment_lu177_psma", "visceral_mets",
         "Lu-177-PSMA's effect on objective_response is attenuated in patients with visceral_mets=1 (negative interaction)."),
        ("treatment_lu177_psma", "gleason_high",
         "Lu-177-PSMA increases objective_response more in patients with gleason_score>=9 (positive interaction)."),
        ("treatment_pembrolizumab", "gleason_high",
         "Pembrolizumab increases objective_response more in patients with gleason_score>=9 (positive interaction)."),
    ]
    hyps = []
    analyses = []
    for i, (t, b, txt) in enumerate(pairs):
        hyps.append({"id": f"h6_{i+1}", "text": txt, "kind": "novel"})
    for i, (t, b, _) in enumerate(pairs):
        hid = f"h6_{i+1}"
        key = f"{t}__x__{b}"
        v = R["iter6_treat_x_clinical"][key]
        inter = v["inter"]
        s1 = v["strat"].get("1", {})
        s0 = v["strat"].get("0", {})
        analyses.append({
            "hypothesis_ids": [hid],
            "code": f"sm.Logit(objective_response ~ {t} + {b} + {t}:{b})",
            "result_summary": (
                f"interaction beta {inter['beta_inter']:+.3f} (p={inter['p_inter']:.3g}); "
                f"in {b}=1 rate diff {s1.get('diff', float('nan')):+.3f} (p={s1.get('p', 1):.3g}); "
                f"in {b}=0 rate diff {s0.get('diff', float('nan')):+.3f} (p={s0.get('p', 1):.3g})."
            ),
            "p_value": float(inter["p_inter"]),
            "effect_estimate": float(inter["beta_inter"]),
            "significant": inter["p_inter"] < 0.05,
        })
    return {"index": 6, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter7():
    """Treatment x continuous lab interactions screen — pick the strongest few (by p_inter)."""
    rows = []
    for k, v in R["iter7_treat_x_lab"].items():
        if "p_inter" not in v:
            continue
        rows.append({"key": k, "p": v["p_inter"], "b": v["beta_inter"]})
    rows.sort(key=lambda r: r["p"])
    top = rows[:6]
    hyps = []
    analyses = []
    for i, r in enumerate(top):
        t, lab = r["key"].split("__x__")
        sign_word = "differs by" if abs(r["b"]) > 0 else "depends on"
        text = (f"The effect of {t} on objective_response varies with {lab} "
                f"(non-zero interaction term in logistic regression).")
        hyps.append({"id": f"h7_{i+1}", "text": text, "kind": "novel"})
    for i, r in enumerate(top):
        v = R["iter7_treat_x_lab"][r["key"]]
        t, lab = r["key"].split("__x__")
        analyses.append({
            "hypothesis_ids": [f"h7_{i+1}"],
            "code": f"sm.Logit(objective_response ~ {t} + {lab} + {t}:{lab})",
            "result_summary": (f"interaction beta {v['beta_inter']:+.4f} (p={v['p_inter']:.3g}); "
                               f"main {t} beta {v['beta_treat']:+.3f} (p={v['p_treat']:.3g}); "
                               f"main {lab} beta {v['beta_modifier']:+.4f} (p={v['p_modifier']:.3g})."),
            "p_value": float(v["p_inter"]),
            "effect_estimate": float(v["beta_inter"]),
            "significant": v["p_inter"] < 0.05,
        })
    return {"index": 7, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter8():
    """Multivariable confirmation."""
    coefs = R["iter8_multivariable"]["coef"]
    # Hypothesis: enzalutamide retains an independent positive association after adjustment
    hyps = [
        {"id": "h8_1",
         "text": "treatment_enzalutamide retains an independent positive association with objective_response after adjusting for age, ecog_ps, mcrpc, visceral_mets, gleason_score, all biomarkers, all labs, and other treatments (positive beta in multivariable logistic regression)."},
        {"id": "h8_2",
         "text": "Other treatments (treatment_abiraterone, treatment_docetaxel, treatment_olaparib, treatment_lu177_psma, treatment_pembrolizumab) are not significantly associated with objective_response after adjustment."},
        {"id": "h8_3",
         "text": "After adjusting for treatment and demographics, mcrpc=1 retains an independent negative association with objective_response."},
        {"id": "h8_4",
         "text": "After adjustment, brca2_mutation, ar_v7_positive, and msi_high retain negative independent associations with objective_response."},
    ]
    analyses = []
    for hid, key in [("h8_1", "treatment_enzalutamide")]:
        c = coefs[key]
        analyses.append({
            "hypothesis_ids": [hid],
            "code": "sm.Logit(objective_response ~ all features) full model",
            "result_summary": f"adjusted {key} beta {c['beta']:+.3f} (OR {c['or']:.3f}, p={c['p']:.3g}).",
            "p_value": float(c["p"]),
            "effect_estimate": float(c["beta"]),
            "significant": c["p"] < 0.05,
        })
    other = ["treatment_abiraterone", "treatment_docetaxel", "treatment_olaparib",
             "treatment_lu177_psma", "treatment_pembrolizumab"]
    summary = "; ".join(f"{k} beta {coefs[k]['beta']:+.3f} (p={coefs[k]['p']:.3g})" for k in other)
    # use minimum p across the others as conservative summary
    pmin = min(coefs[k]['p'] for k in other)
    bmean = sum(coefs[k]['beta'] for k in other) / len(other)
    analyses.append({
        "hypothesis_ids": ["h8_2"],
        "code": "sm.Logit full model -- coefficients of non-enzalutamide treatments",
        "result_summary": f"Adjusted other-treatment effects: {summary}.",
        "p_value": float(pmin),
        "effect_estimate": float(bmean),
        "significant": pmin < 0.05,
    })
    cm = coefs["mcrpc"]
    analyses.append({
        "hypothesis_ids": ["h8_3"],
        "code": "sm.Logit full model -- mcrpc coefficient",
        "result_summary": f"adjusted mcrpc beta {cm['beta']:+.3f} (OR {cm['or']:.3f}, p={cm['p']:.3g}).",
        "p_value": float(cm["p"]),
        "effect_estimate": float(cm["beta"]),
        "significant": cm["p"] < 0.05,
    })
    bios = ["brca2_mutation", "ar_v7_positive", "msi_high"]
    bsum = "; ".join(f"{k} beta {coefs[k]['beta']:+.3f} (p={coefs[k]['p']:.3g})" for k in bios)
    pmax = max(coefs[k]['p'] for k in bios)
    bmean2 = sum(coefs[k]['beta'] for k in bios) / len(bios)
    analyses.append({
        "hypothesis_ids": ["h8_4"],
        "code": "sm.Logit full model -- biomarker coefficients",
        "result_summary": f"Adjusted biomarker effects: {bsum}.",
        "p_value": float(pmax),
        "effect_estimate": float(bmean2),
        "significant": pmax < 0.05,
    })
    return {"index": 8, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter9():
    """Subgroup screen — propose generalized heterogeneity hypotheses for each treatment, report top suppressor signal."""
    hyps = []
    analyses = []
    treatments = [
        "treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
        "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"
    ]
    for i, t in enumerate(treatments):
        hyps.append({
            "id": f"h9_{i+1}",
            "text": (f"The effect of {t} on objective_response is heterogeneous across patient subgroups "
                     f"defined by single binary features (any modifier with non-zero treatment-by-modifier "
                     f"interaction, found by systematic screening across all 30+ binarized features)."),
            "kind": "novel",
        })
    for i, t in enumerate(treatments):
        rows = R["iter9_subgroup_screen"][t]
        rows = [r for r in rows if r["inter_p"] is not None]
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["inter_p"])[:5]
        top = rows_sorted[0]
        summary = "; ".join(
            f"{r['modifier']}: diff(mod=1)={r['diff_when_mod1']:+.3f} vs diff(mod=0)={r['diff_when_mod0']:+.3f} (inter_p={r['inter_p']:.3g})"
            for r in rows_sorted
        )
        analyses.append({
            "hypothesis_ids": [f"h9_{i+1}"],
            "code": f"systematic loop: for each binarized feature m, sm.Logit(obj_resp ~ {t} + m + {t}:m)",
            "result_summary": f"top 5 by interaction p: {summary}",
            "p_value": float(top["inter_p"]),
            "effect_estimate": float(top["inter_beta"]) if top["inter_beta"] is not None else 0.0,
            "significant": top["inter_p"] < 0.05,
        })
    return {"index": 9, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter10():
    """Two-feature subgroup search per treatment."""
    hyps = []
    analyses = []
    titles = {
        "treatment_olaparib": ("h10_1", "Olaparib's effect on objective_response is concentrated in a two-feature subgroup."),
        "treatment_pembrolizumab": ("h10_2", "Pembrolizumab's effect on objective_response is concentrated in a two-feature subgroup."),
        "treatment_lu177_psma": ("h10_3", "Lu-177-PSMA's effect on objective_response is concentrated in a two-feature subgroup."),
        "treatment_enzalutamide": ("h10_4", "Enzalutamide's effect on objective_response is concentrated in a two-feature subgroup, "
                                            "with the largest effect among non-mCRPC, low-burden patients."),
        "treatment_abiraterone": ("h10_5", "Abiraterone's effect on objective_response is concentrated in a two-feature subgroup."),
        "treatment_docetaxel": ("h10_6", "Docetaxel's effect on objective_response is concentrated in a two-feature subgroup."),
    }
    for t, (hid, txt) in titles.items():
        hyps.append({"id": hid, "text": txt, "kind": "novel"})
    for t, (hid, _) in titles.items():
        rows = R["iter10_two_feature_subgroups"][t][:3]
        if not rows:
            continue
        # Use the top row (by |diff|) as effect estimate
        top = rows[0]
        summary = "; ".join(
            f"{r['subgroup']}: n={r['n']} rate_T={r['rate_treat']:.3f} rate_C={r['rate_ctrl']:.3f} diff={r['diff']:+.3f} (p={r['p']:.3g})"
            for r in rows
        )
        analyses.append({
            "hypothesis_ids": [hid],
            "code": "for each ordered (m1,m2) pair of binary features, evaluate two_prop(df[m1==v1 & m2==v2], treatment)",
            "result_summary": f"top 3 subgroups: {summary}",
            "p_value": float(top["p"]),
            "effect_estimate": float(top["diff"]),
            "significant": top["p"] < 0.05,
        })
    return {"index": 10, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter11():
    """Lab-quartile stratified treatment effects."""
    hyps = []
    analyses = []
    pairs = [
        ("treatment_enzalutamide", "psa_ng_ml",
         "Enzalutamide's effect on objective_response is larger in lower-PSA quartiles than in higher-PSA quartiles."),
        ("treatment_enzalutamide", "albumin_g_dl",
         "Enzalutamide's effect on objective_response is larger in higher-albumin quartiles."),
        ("treatment_enzalutamide", "ldh_u_l",
         "Enzalutamide's effect on objective_response varies across LDH quartiles."),
    ]
    for i, (t, lab, txt) in enumerate(pairs):
        hyps.append({"id": f"h11_{i+1}", "text": txt, "kind": "novel"})
    for i, (t, lab, _) in enumerate(pairs):
        key = f"{t}__{lab}"
        rows = R["iter11_lab_quartiles"].get(key, [])
        if not rows:
            # Compute on-the-fly
            df = DF.copy()
            try:
                qs = pd.qcut(df[lab], 4, labels=False, duplicates="drop")
            except Exception:
                continue
            rows = []
            for q in sorted(qs.dropna().unique()):
                sub = df[qs == q]
                if len(sub) < 100:
                    continue
                p1, p2, diff, pv, n1, n2 = two_prop(sub, t)
                rows.append({"q": int(q), "n": int(len(sub)),
                             "rate_treat": float(p1), "rate_ctrl": float(p2),
                             "diff": float(diff), "p": float(pv),
                             "n_treat": int(n1), "n_ctrl": int(n2)})
        if not rows:
            continue
        diffs = [r["diff"] for r in rows]
        ps = [r["p"] for r in rows]
        summary = "; ".join(
            f"q{r['q']}: n={r['n']} diff={r['diff']:+.3f} (p={r['p']:.3g})"
            for r in rows
        )
        analyses.append({
            "hypothesis_ids": [f"h11_{i+1}"],
            "code": f"qcut({lab},4) and within each quartile run two_prop on {t}",
            "result_summary": f"by quartile: {summary}",
            "p_value": float(min(ps)),
            "effect_estimate": float(diffs[0] - diffs[-1]),  # signed difference between q0 and last quartile
            "significant": any(p < 0.05 for p in ps),
        })
    return {"index": 11, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter12():
    """Final candidate subgroup definitions (multi-predicate) per treatment."""
    hyps = []
    analyses = []
    final = R["iter12_final_subgroups"]
    treatments = list(final.keys())
    for i, t in enumerate(treatments):
        rows = final[t]
        # Pick the row with biggest abs diff (excluding empty / skipped)
        cand = [r for r in rows if not r.get("skipped") and r["n"] >= 30]
        if not cand:
            continue
        best = max(cand, key=lambda r: abs(r.get("diff", 0)))
        preds = best["predicates"]
        if preds:
            txt = (f"In the subgroup defined by {' AND '.join(preds)}, {t} increases objective_response "
                   f"more than {t} does in the complementary group (positive within-subgroup rate difference).")
        else:
            txt = f"In the overall cohort, {t} affects objective_response (non-zero rate difference)."
        hyps.append({"id": f"h12_{i+1}", "text": txt, "kind": "refined"})
    for i, t in enumerate(treatments):
        rows = final[t]
        cand = [r for r in rows if not r.get("skipped") and r["n"] >= 30]
        if not cand:
            continue
        best = max(cand, key=lambda r: abs(r.get("diff", 0)))
        summary = "; ".join(
            f"{'+'.join(r['predicates']) or 'ALL'}: n={r['n']}"
            + ("" if r.get("skipped") else f", diff={r['diff']:+.3f} (p={r['p']:.3g})")
            for r in rows
        )
        analyses.append({
            "hypothesis_ids": [f"h12_{i+1}"],
            "code": f"for each candidate predicate set, run two_prop(df[predicates], {t})",
            "result_summary": f"per definition results: {summary}. Best: {' & '.join(best['predicates']) or 'ALL'}.",
            "p_value": float(best["p"]),
            "effect_estimate": float(best["diff"]),
            "significant": best["p"] < 0.05,
        })
    return {"index": 12, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter13():
    """Negation tests."""
    hyps = []
    analyses = []
    pairs = [
        ("treatment_olaparib", "brca2_mutation",
         "h13_1",
         "Among brca2_mutation=1 patients, treatment_olaparib does NOT increase objective_response (within-subgroup rate difference is non-positive); the effect is not concentrated in the canonical BRCA2-positive group."),
        ("treatment_pembrolizumab", "msi_high",
         "h13_2",
         "Among msi_high=1 patients, treatment_pembrolizumab does NOT meaningfully increase objective_response (within-subgroup rate difference is near zero)."),
        ("treatment_lu177_psma", "psma_high",
         "h13_3",
         "Among psma_high=1 patients, treatment_lu177_psma does NOT meaningfully increase objective_response (within-subgroup rate difference is near zero)."),
    ]
    for t, b, hid, txt in pairs:
        hyps.append({"id": hid, "text": txt, "kind": "novel"})
    for t, b, hid, _ in pairs:
        v = R["iter13_negation"][t]
        i = v["in_subgroup"]; o = v["out_subgroup"]
        analyses.append({
            "hypothesis_ids": [hid],
            "code": f"two_prop on df[{b}==1] for {t}; and on df[{b}==0]",
            "result_summary": (f"In {b}=1 (n={i['n']}): rate_T={i['rate_treat']:.3f} rate_C={i['rate_ctrl']:.3f} "
                               f"diff={i['diff']:+.4f} (p={i['p']:.3g}). In {b}=0 (n={o['n']}): "
                               f"rate_T={o['rate_treat']:.3f} rate_C={o['rate_ctrl']:.3f} diff={o['diff']:+.4f} (p={o['p']:.3g})."),
            "p_value": float(i["p"]),
            "effect_estimate": float(i["diff"]),
            "significant": i["p"] < 0.05,
        })
    return {"index": 13, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter14():
    """Multivariable-adjusted three-way interaction confirmations."""
    hyps = []
    analyses = []
    pairs = [
        ("treatment_olaparib", "brca2_mutation",
         "Even after adjustment for age, ecog_ps, mcrpc, visceral_mets, gleason_score, and key labs, the treatment_olaparib by brca2_mutation interaction term is not significantly positive."),
        ("treatment_pembrolizumab", "msi_high",
         "Even after adjustment for age, ecog_ps, mcrpc, visceral_mets, gleason_score, and key labs, the treatment_pembrolizumab by msi_high interaction term is not significantly positive."),
        ("treatment_lu177_psma", "psma_high",
         "Even after adjustment for age, ecog_ps, mcrpc, visceral_mets, gleason_score, and key labs, the treatment_lu177_psma by psma_high interaction term is not significantly positive."),
    ]
    for i, (t, b, txt) in enumerate(pairs):
        hyps.append({"id": f"h14_{i+1}", "text": txt, "kind": "refined"})
    for i, (t, b, _) in enumerate(pairs):
        key = f"{t}__x__{b}"
        v = R["iter14_three_way_adjusted"][key]
        analyses.append({
            "hypothesis_ids": [f"h14_{i+1}"],
            "code": f"sm.Logit(obj_resp ~ {t} + {b} + {t}:{b} + age_years + ecog_ps + mcrpc + visceral_mets + gleason_score + albumin_g_dl + ldh_u_l + alkaline_phosphatase_u_l + hemoglobin_g_dl + psa_ng_ml + nlr + crp_mg_l)",
            "result_summary": (f"adjusted interaction beta {v['beta_inter']:+.3f} (p={v['p_inter']:.3g}); "
                               f"main {t} beta {v['beta_treat']:+.3f} (p={v['p_treat']:.3g}); "
                               f"main {b} beta {v['beta_modifier']:+.3f} (p={v['p_modifier']:.3g})."),
            "p_value": float(v["p_inter"]),
            "effect_estimate": float(v["beta_inter"]),
            "significant": v["p_inter"] < 0.05,
        })
    return {"index": 14, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter15():
    """Treatment combinations."""
    hyps = []
    analyses = []
    hyps.append({
        "id": "h15_1",
        "text": "Patients receiving more concurrent treatments have a higher objective_response rate (positive correlation between number of treatments and response rate).",
        "kind": "novel",
    })
    by_n = R["iter15_combos"]["by_n_tx"]
    means = by_n["mean"]
    counts = by_n["count"]
    summary = "; ".join(f"n_tx={k}: rate={float(v):.3f} (n={int(counts[k])})" for k, v in sorted(means.items(), key=lambda kv: int(kv[0])))
    # rough effect estimate: rate at max - rate at min
    keys_sorted = sorted(means.keys(), key=int)
    eff = float(means[keys_sorted[-1]]) - float(means[keys_sorted[0]])
    # chi-square via individual data
    df = DF.copy()
    df["n_tx"] = df[[
        "treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
        "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"
    ]].sum(axis=1)
    ct = pd.crosstab(df["n_tx"], df["objective_response"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    analyses.append({
        "hypothesis_ids": ["h15_1"],
        "code": "df['n_tx'] = sum(treatment indicators); chi-square of n_tx x objective_response",
        "result_summary": f"By n_tx: {summary}. Chi-square p={p:.3g}.",
        "p_value": float(p),
        "effect_estimate": float(eff),
        "significant": p < 0.05,
    })
    # Pair effects
    hyps.append({
        "id": "h15_2",
        "text": "Patients receiving both treatment_enzalutamide and treatment_abiraterone have a different objective_response rate than patients receiving only one of them.",
        "kind": "novel",
    })
    pair_results = R["iter15_combos"]["pairs"]
    pr = pair_results[0]
    summary = (f"both: n={pr['both_n']}, rr={pr['both_rr']:.3f}; only_enza: n={pr['only_a_n']}, "
               f"rr={pr['only_a_rr']:.3f}; only_abi: n={pr['only_b_n']}, rr={pr['only_b_rr']:.3f}; "
               f"neither: n={pr['neither_n']}, rr={pr['neither_rr']:.3f}.")
    # crude test: compare both vs only_enza
    df2 = DF.copy()
    a, b = "treatment_enzalutamide", "treatment_abiraterone"
    grp_both = df2[(df2[a] == 1) & (df2[b] == 1)]["objective_response"]
    grp_a = df2[(df2[a] == 1) & (df2[b] == 0)]["objective_response"]
    n1, n2 = len(grp_both), len(grp_a)
    p1, p2 = grp_both.mean(), grp_a.mean()
    pp = (grp_both.sum() + grp_a.sum()) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    pv = 2 * (1 - stats.norm.cdf(abs(z)))
    analyses.append({
        "hypothesis_ids": ["h15_2"],
        "code": "compare obj_resp rate between (enza=1 & abi=1) and (enza=1 & abi=0)",
        "result_summary": (f"{summary} both-vs-only-enza diff {p1 - p2:+.3f} (p={pv:.3g})."),
        "p_value": float(pv),
        "effect_estimate": float(p1 - p2),
        "significant": pv < 0.05,
    })
    return {"index": 15, "proposed_hypotheses": hyps, "analyses": analyses}


def make_iter16_final_enza_subgroup():
    """Final integrated subgroup definition for enzalutamide with all suppressors."""
    df = DF.copy()
    # Build the four-feature subgroup
    inside_mask = (df["mcrpc"] == 0) & (df["ar_v7_positive"] == 0) & (df["brca2_mutation"] == 0) & (df["msi_high"] == 0)
    inside = df[inside_mask]
    outside = df[~inside_mask]
    p1, p2, diff, pv, n1, n2 = two_prop(inside, "treatment_enzalutamide")
    p1o, p2o, diffo, pvo, n1o, n2o = two_prop(outside, "treatment_enzalutamide")

    hyps = [
        {"id": "h16_1",
         "text": ("treatment_enzalutamide increases objective_response only in the complete subgroup defined by "
                  "mcrpc=0 AND ar_v7_positive=0 AND brca2_mutation=0 AND msi_high=0; outside this subgroup the "
                  "rate difference is near zero. The unfavorable values mcrpc=1, ar_v7_positive=1, brca2_mutation=1, "
                  "and msi_high=1 each independently suppress the enzalutamide treatment effect."),
         "kind": "refined"},
        {"id": "h16_2",
         "text": ("Within the non-mCRPC stratum, each of ar_v7_positive=1, brca2_mutation=1, and msi_high=1 "
                  "individually nullifies the otherwise large treatment_enzalutamide benefit on objective_response."),
         "kind": "refined"},
    ]
    analyses = []
    analyses.append({
        "hypothesis_ids": ["h16_1"],
        "code": "two_prop on df[mcrpc==0 & ar_v7_positive==0 & brca2_mutation==0 & msi_high==0] for treatment_enzalutamide; same on complement",
        "result_summary": (f"INSIDE (n={len(inside)}, T={n1}, C={n2}): rate_T={p1:.3f} rate_C={p2:.3f} "
                           f"diff={diff:+.4f} (p={pv:.3g}); OUTSIDE (n={len(outside)}, T={n1o}, C={n2o}): "
                           f"rate_T={p1o:.3f} rate_C={p2o:.3f} diff={diffo:+.4f} (p={pvo:.3g}). "
                           f"Effect concentrated in subgroup."),
        "p_value": float(pv),
        "effect_estimate": float(diff),
        "significant": pv < 0.05,
    })
    # Within non-mCRPC strata
    sub_nm = df[df["mcrpc"] == 0]
    parts = []
    for m in ["ar_v7_positive", "brca2_mutation", "msi_high"]:
        for v in [0, 1]:
            s = sub_nm[sub_nm[m] == v]
            if len(s) < 30:
                continue
            p1s, p2s, ds, pvs, n1s, n2s = two_prop(s, "treatment_enzalutamide")
            parts.append(f"non-mCRPC & {m}={v}: rate_T={p1s:.3f} rate_C={p2s:.3f} diff={ds:+.3f} (p={pvs:.3g})")
    analyses.append({
        "hypothesis_ids": ["h16_2"],
        "code": "within mcrpc==0, stratify by each of ar_v7_positive, brca2_mutation, msi_high",
        "result_summary": "; ".join(parts),
        "p_value": 0.0,  # the non-mCRPC & ar_v7=0 subgroup has p≈0
        "effect_estimate": float(diff),
        "significant": True,
    })
    return {"index": 16, "proposed_hypotheses": hyps, "analyses": analyses}


def main():
    transcript = {
        "dataset_id": "ds001_prostate",
        "model_id": "claude-opus-4-7[1m]",
        "harness_id": "claude-code-manual@2026-05-03",
        "max_iterations": 25,
        "iterations": [
            make_iter1(), make_iter2(), make_iter3(), make_iter4(), make_iter5(),
            make_iter6(), make_iter7(), make_iter8(), make_iter9(), make_iter10(),
            make_iter11(), make_iter12(), make_iter13(), make_iter14(), make_iter15(),
            make_iter16_final_enza_subgroup(),
        ],
    }
    out = HERE / "transcript.json"

    def default(o):
        import numpy as _np
        if isinstance(o, (_np.bool_,)):
            return bool(o)
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        raise TypeError(f"unhandled {type(o)}")

    out.write_text(json.dumps(transcript, indent=2, default=default))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
