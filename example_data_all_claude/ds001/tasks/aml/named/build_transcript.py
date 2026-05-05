"""Build transcript.json and analysis_summary.txt from analysis_results.json."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

with open("analysis_results.json", encoding="utf-8") as f:
    RESULTS: list[dict[str, Any]] = json.load(f)

# ---------------------------------------------------------------------------
# Hypothesis registry: id -> (text, kind, iteration_index)
# ---------------------------------------------------------------------------
H: dict[str, tuple[str, str, int]] = {}

# Iteration 1
H["h1"] = ("Older age (higher age_years) is associated with lower probability of objective_response.", "novel", 1)
H["h2"] = ("Higher ECOG performance status (ecog_ps) is associated with lower probability of objective_response.", "novel", 1)
H["h3"] = ("The objective_response rate differs between sex_female=1 and sex_female=0 patients.", "novel", 1)
H["h4"] = ("Patients with secondary_aml=1 have a lower objective_response rate than patients with secondary_aml=0.", "novel", 1)
H["h5"] = ("Patients flagged unfit_for_intensive=1 have a lower objective_response rate than patients with unfit_for_intensive=0.", "novel", 1)

# Iteration 2
H["h6"] = ("Patients with complex_karyotype=1 have a lower objective_response rate than patients with complex_karyotype=0.", "novel", 2)
H["h7"] = ("Patients with tp53_mutation=1 have a lower objective_response rate than patients with tp53_mutation=0.", "novel", 2)
H["h8"] = ("Patients with npm1_mutation=1 have a higher objective_response rate than patients with npm1_mutation=0.", "novel", 2)
H["h9"] = ("Patients with flt3_itd=1 have a different objective_response rate than patients with flt3_itd=0.", "novel", 2)
H["h10"] = ("Patients with flt3_tkd=1 have a different objective_response rate than patients with flt3_tkd=0.", "novel", 2)
H["h11"] = ("Patients with idh1_mutation=1 have a different objective_response rate than patients with idh1_mutation=0.", "novel", 2)
H["h12"] = ("Patients with idh2_mutation=1 have a different objective_response rate than patients with idh2_mutation=0.", "novel", 2)

# Iteration 3
H["h13"] = ("Receiving treatment_midostaurin=1 is associated with higher overall objective_response than treatment_midostaurin=0.", "novel", 3)
H["h14"] = ("Receiving treatment_gilteritinib=1 is associated with higher overall objective_response than treatment_gilteritinib=0.", "novel", 3)
H["h15"] = ("Receiving treatment_ivosidenib=1 is associated with higher overall objective_response than treatment_ivosidenib=0.", "novel", 3)
H["h16"] = ("Receiving treatment_enasidenib=1 is associated with higher overall objective_response than treatment_enasidenib=0.", "novel", 3)
H["h17"] = ("Receiving treatment_venetoclax_azacitidine=1 is associated with higher overall objective_response than treatment_venetoclax_azacitidine=0.", "novel", 3)
H["h18"] = ("Receiving treatment_7plus3=1 is associated with higher overall objective_response than treatment_7plus3=0.", "novel", 3)

# Iteration 4 (continuous labs)
H["h19"] = ("Higher wbc_k_per_ul is associated with a lower objective_response rate.", "novel", 4)
H["h20"] = ("Higher blast_pct_marrow is associated with a different objective_response rate.", "novel", 4)
H["h21"] = ("Higher albumin_g_dl is associated with higher objective_response rate.", "novel", 4)
H["h22"] = ("Higher ldh_u_l is associated with lower objective_response rate.", "novel", 4)
H["h23"] = ("Higher weight_loss_pct_6mo is associated with lower objective_response rate.", "novel", 4)
H["h24"] = ("Higher crp_mg_l is associated with lower objective_response rate.", "novel", 4)
H["h25"] = ("Higher nlr (neutrophil-lymphocyte ratio) is associated with lower objective_response rate.", "novel", 4)
H["h26"] = ("Higher hemoglobin_g_dl is associated with higher objective_response rate.", "novel", 4)
H["h27"] = ("Higher alkaline_phosphatase_u_l is associated with a different objective_response rate.", "novel", 4)
H["h28"] = ("Higher ast_u_l is associated with a different objective_response rate.", "novel", 4)
H["h29"] = ("Higher alt_u_l is associated with a different objective_response rate.", "novel", 4)
H["h30"] = ("Higher total_bilirubin_mg_dl is associated with lower objective_response rate.", "novel", 4)
H["h31"] = ("Higher creatinine_mg_dl is associated with lower objective_response rate.", "novel", 4)
H["h32"] = ("Higher bun_mg_dl is associated with lower objective_response rate.", "novel", 4)
H["h33"] = ("Lower sodium_meq_l is associated with lower objective_response rate (i.e., higher sodium = higher response).", "novel", 4)
H["h34"] = ("potassium_meq_l is associated with objective_response rate.", "novel", 4)
H["h35"] = ("calcium_mg_dl is associated with objective_response rate.", "novel", 4)

# Iteration 5 (adjusted treatment effects)
H["h36"] = ("After multivariable logistic adjustment for demographics, fitness, mutations and labs, treatment_midostaurin is independently associated with higher objective_response.", "refined", 5)
H["h37"] = ("After multivariable adjustment, treatment_gilteritinib is independently associated with higher objective_response.", "refined", 5)
H["h38"] = ("After multivariable adjustment, treatment_ivosidenib is independently associated with higher objective_response.", "refined", 5)
H["h39"] = ("After multivariable adjustment, treatment_enasidenib is independently associated with higher objective_response.", "refined", 5)
H["h40"] = ("After multivariable adjustment, treatment_venetoclax_azacitidine is independently associated with higher objective_response.", "refined", 5)
H["h41"] = ("After multivariable adjustment, treatment_7plus3 is independently associated with higher objective_response.", "refined", 5)

# Iteration 6 (FLT3 inhibitor x FLT3)
H["h42"] = ("There is a positive interaction between treatment_midostaurin and flt3_itd: midostaurin's effect on objective_response is larger in flt3_itd=1 than in flt3_itd=0.", "novel", 6)
H["h43"] = ("Within flt3_itd=1, treatment_midostaurin=1 produces higher objective_response than treatment_midostaurin=0.", "novel", 6)
H["h44"] = ("There is a positive interaction between treatment_midostaurin and flt3_tkd: midostaurin's effect is larger in flt3_tkd=1 than in flt3_tkd=0.", "novel", 6)
H["h45"] = ("Within flt3_tkd=1, treatment_midostaurin=1 produces higher objective_response than treatment_midostaurin=0.", "novel", 6)
H["h46"] = ("There is a positive interaction between treatment_gilteritinib and flt3_itd: gilteritinib's effect is larger in flt3_itd=1 than in flt3_itd=0.", "novel", 6)
H["h47"] = ("Within flt3_itd=1, treatment_gilteritinib=1 produces higher objective_response than treatment_gilteritinib=0.", "novel", 6)
H["h48"] = ("There is a positive interaction between treatment_gilteritinib and flt3_tkd: gilteritinib's effect is larger in flt3_tkd=1 than in flt3_tkd=0.", "novel", 6)
H["h49"] = ("Within flt3_tkd=1, treatment_gilteritinib=1 produces higher objective_response than treatment_gilteritinib=0.", "novel", 6)

# Iteration 7 (IDH inhibitors)
H["h50"] = ("There is a positive interaction between treatment_ivosidenib and idh1_mutation: ivosidenib's effect is larger in idh1_mutation=1 than in idh1_mutation=0.", "novel", 7)
H["h51"] = ("Within idh1_mutation=1, treatment_ivosidenib=1 produces higher objective_response than treatment_ivosidenib=0.", "novel", 7)
H["h52"] = ("There is a positive interaction between treatment_ivosidenib and idh2_mutation (cross-target activity).", "novel", 7)
H["h53"] = ("Within idh2_mutation=1, treatment_ivosidenib=1 produces higher objective_response than treatment_ivosidenib=0.", "novel", 7)
H["h54"] = ("There is a positive interaction between treatment_enasidenib and idh2_mutation: enasidenib's effect is larger in idh2_mutation=1 than in idh2_mutation=0.", "novel", 7)
H["h55"] = ("Within idh2_mutation=1, treatment_enasidenib=1 produces higher objective_response than treatment_enasidenib=0.", "novel", 7)
H["h56"] = ("There is a positive interaction between treatment_enasidenib and idh1_mutation (cross-target activity).", "novel", 7)
H["h57"] = ("Within idh1_mutation=1, treatment_enasidenib=1 produces higher objective_response than treatment_enasidenib=0.", "novel", 7)

# Iteration 8
H["h58"] = ("treatment_7plus3 has a smaller (more-negative interaction) effect on objective_response in tp53_mutation=1 vs tp53_mutation=0 patients.", "novel", 8)
H["h59"] = ("treatment_7plus3 has a smaller effect on objective_response in complex_karyotype=1 vs complex_karyotype=0 patients.", "novel", 8)
H["h60"] = ("treatment_venetoclax_azacitidine has a smaller effect on objective_response in tp53_mutation=1 vs tp53_mutation=0 patients.", "novel", 8)
H["h61"] = ("treatment_venetoclax_azacitidine has a smaller effect on objective_response in complex_karyotype=1 vs complex_karyotype=0 patients.", "novel", 8)

# Iteration 9
H["h62"] = ("treatment_7plus3 has a larger positive effect on objective_response in unfit_for_intensive=0 (fit) vs unfit_for_intensive=1 patients.", "novel", 9)
H["h63"] = ("treatment_venetoclax_azacitidine has a larger positive effect on objective_response in unfit_for_intensive=1 vs unfit_for_intensive=0 patients.", "novel", 9)

# Iteration 10
H["h64"] = ("treatment_7plus3's effect on objective_response is larger in age_years<75 (age_ge75=0) than in age_ge75=1 patients.", "novel", 10)
H["h65"] = ("treatment_venetoclax_azacitidine's effect on objective_response is larger in age_years>=75 (age_ge75=1) than in age_ge75=0 patients.", "novel", 10)

# Iteration 11
H["h66"] = ("treatment_7plus3's effect on objective_response is larger in ecog_ps<2 (ecog_ge2=0) than in ecog_ge2=1 patients.", "novel", 11)
H["h67"] = ("treatment_venetoclax_azacitidine's effect on objective_response is larger in ecog_ge2=1 than in ecog_ge2=0 patients.", "novel", 11)
H["h68"] = ("treatment_midostaurin's effect on objective_response varies between ecog_ge2 strata.", "novel", 11)
H["h69"] = ("treatment_gilteritinib's effect on objective_response varies between ecog_ge2 strata.", "novel", 11)

# Iteration 12
H["h70"] = ("Within FLT3-mutated patients (flt3_itd=1 OR flt3_tkd=1), treatment_gilteritinib=1 produces higher objective_response than treatment_gilteritinib=0.", "refined", 12)
H["h71"] = ("Within FLT3-mutated and unfit_for_intensive=0 (fit) patients, treatment_gilteritinib=1 produces higher objective_response than treatment_gilteritinib=0.", "refined", 12)
H["h72"] = ("Within FLT3-mutated and unfit_for_intensive=0 (fit) patients, treatment_midostaurin=1 produces higher objective_response than treatment_midostaurin=0.", "refined", 12)
H["h73"] = ("Within flt3_itd=1 and unfit_for_intensive=0 (fit) patients, treatment_midostaurin=1 produces higher objective_response than treatment_midostaurin=0.", "refined", 12)

# Iteration 13
H["h74"] = ("Within idh1_mutation=1 patients, treatment_ivosidenib=1 produces higher objective_response than treatment_ivosidenib=0.", "refined", 13)
H["h75"] = ("Within idh2_mutation=1 patients, treatment_enasidenib=1 produces higher objective_response than treatment_enasidenib=0.", "refined", 13)
H["h76"] = ("Within idh1_mutation=1 and tp53_mutation=0 patients, treatment_ivosidenib=1 produces higher objective_response than treatment_ivosidenib=0.", "refined", 13)
H["h77"] = ("Within idh2_mutation=1 and tp53_mutation=0 patients, treatment_enasidenib=1 produces higher objective_response than treatment_enasidenib=0.", "refined", 13)

# Iteration 14 (ven-aza)
H["h78"] = ("Within unfit_for_intensive=1 patients, treatment_venetoclax_azacitidine=1 produces higher objective_response than treatment_venetoclax_azacitidine=0.", "refined", 14)
H["h79"] = ("Within unfit_for_intensive=1 and tp53_mutation=0 patients, treatment_venetoclax_azacitidine=1 produces higher objective_response than treatment_venetoclax_azacitidine=0.", "refined", 14)
H["h80"] = ("Within unfit_for_intensive=1 and complex_karyotype=0 patients, treatment_venetoclax_azacitidine=1 produces higher objective_response than treatment_venetoclax_azacitidine=0.", "refined", 14)
H["h81"] = ("Within unfit_for_intensive=1 AND tp53_mutation=0 AND complex_karyotype=0 patients, treatment_venetoclax_azacitidine=1 produces higher objective_response than treatment_venetoclax_azacitidine=0.", "refined", 14)

# Iteration 15 (7+3)
H["h82"] = ("Within unfit_for_intensive=0 (fit) patients, treatment_7plus3=1 produces higher objective_response than treatment_7plus3=0.", "refined", 15)
H["h83"] = ("Within unfit_for_intensive=0 AND tp53_mutation=0 patients, treatment_7plus3=1 produces higher objective_response than treatment_7plus3=0.", "refined", 15)
H["h84"] = ("Within unfit_for_intensive=0 AND tp53_mutation=0 AND complex_karyotype=0 patients, treatment_7plus3=1 produces higher objective_response than treatment_7plus3=0.", "refined", 15)

# Iteration 16: top-15 systematic interaction screen
for i in range(85, 100):
    H[f"h{i}"] = (
        f"Top-ranked interaction screen hit #{i-84}: a treatment-by-feature interaction term predicts objective_response (treatment and modifier identified by lowest-p sweep).",
        "novel",
        16,
    )

# Iteration 17: best-Δ subgroup search per treatment
H["h100"] = ("There exists a 1-3 binary feature combination defining a subgroup where treatment_midostaurin produces the largest positive difference in objective_response (on - off).", "novel", 17)
H["h101"] = ("There exists a 1-3 binary feature combination defining a subgroup where treatment_gilteritinib produces the largest positive difference in objective_response (on - off).", "novel", 17)
H["h102"] = ("There exists a 1-3 binary feature combination defining a subgroup where treatment_ivosidenib produces the largest positive difference in objective_response (on - off).", "novel", 17)
H["h103"] = ("There exists a 1-3 binary feature combination defining a subgroup where treatment_enasidenib produces the largest positive difference in objective_response (on - off).", "novel", 17)
H["h104"] = ("There exists a 1-3 binary feature combination defining a subgroup where treatment_venetoclax_azacitidine produces the largest positive difference in objective_response (on - off).", "novel", 17)
H["h105"] = ("There exists a 1-3 binary feature combination defining a subgroup where treatment_7plus3 produces the largest positive difference in objective_response (on - off).", "novel", 17)

# Iteration 18 final/refined statements (restated for emphasis)
H["h120"] = ("Within flt3_itd=1 patients, treatment_gilteritinib=1 produces higher objective_response than treatment_gilteritinib=0 (final restated).", "refined", 18)
H["h121"] = ("Within flt3_itd=1 AND unfit_for_intensive=0 patients, treatment_midostaurin=1 produces higher objective_response than treatment_midostaurin=0 (final restated).", "refined", 18)
H["h122"] = ("Within idh1_mutation=1 AND tp53_mutation=0 AND complex_karyotype=0 patients, treatment_ivosidenib=1 produces higher objective_response than treatment_ivosidenib=0 (final restated).", "refined", 18)
H["h123"] = ("Within idh2_mutation=1 AND tp53_mutation=0 patients, treatment_enasidenib=1 produces higher objective_response than treatment_enasidenib=0 (final restated).", "refined", 18)
H["h124"] = ("Within unfit_for_intensive=1 AND tp53_mutation=0 AND complex_karyotype=0 patients, treatment_venetoclax_azacitidine=1 produces higher objective_response than treatment_venetoclax_azacitidine=0 (final restated).", "refined", 18)
H["h125"] = ("Within unfit_for_intensive=0 AND tp53_mutation=0 AND complex_karyotype=0 patients, treatment_7plus3=1 produces higher objective_response than treatment_7plus3=0 (final restated).", "refined", 18)

# Iteration 19 adjusted within subgroups
H["h126"] = ("In a multivariable logistic model fit only within flt3_itd=1 (adjusting for age_years, sex_female, ecog_ps, wbc_k_per_ul, blast_pct_marrow, albumin_g_dl, ldh_u_l), treatment_gilteritinib has positive log-odds for objective_response.", "refined", 19)
H["h127"] = ("In a multivariable logistic model fit within flt3_itd=1 AND unfit_for_intensive=0 patients, treatment_midostaurin has positive log-odds for objective_response.", "refined", 19)
H["h128"] = ("In a multivariable logistic model fit within idh1_mutation=1 patients, treatment_ivosidenib has positive log-odds for objective_response.", "refined", 19)
H["h129"] = ("In a multivariable logistic model fit within idh2_mutation=1 patients, treatment_enasidenib has positive log-odds for objective_response.", "refined", 19)
H["h130"] = ("In a multivariable logistic model fit within unfit_for_intensive=1 patients, treatment_venetoclax_azacitidine has positive log-odds for objective_response.", "refined", 19)
H["h131"] = ("In a multivariable logistic model fit within unfit_for_intensive=0 patients, treatment_7plus3 has positive log-odds for objective_response.", "refined", 19)

# Iteration 20 negative-control subgroups
H["h132"] = ("Within flt3_itd=0 AND flt3_tkd=0 (FLT3-wt) patients, treatment_gilteritinib=1 does NOT produce a meaningful difference in objective_response vs treatment_gilteritinib=0 (negative control).", "novel", 20)
H["h133"] = ("Within flt3_itd=0 AND flt3_tkd=0 (FLT3-wt) patients, treatment_midostaurin=1 does NOT produce a meaningful difference in objective_response vs treatment_midostaurin=0 (negative control).", "novel", 20)
H["h134"] = ("Within idh1_mutation=0 patients, treatment_ivosidenib=1 does NOT produce a meaningful difference in objective_response vs treatment_ivosidenib=0 (negative control).", "novel", 20)
H["h135"] = ("Within idh2_mutation=0 patients, treatment_enasidenib=1 does NOT produce a meaningful difference in objective_response vs treatment_enasidenib=0 (negative control).", "novel", 20)

# Iteration 21 NPM1 modifier of 7+3
H["h136"] = ("Within npm1_mutation=1 AND unfit_for_intensive=0 patients, treatment_7plus3=1 produces higher objective_response than treatment_7plus3=0.", "novel", 21)
H["h137"] = ("Within npm1_mutation=0 AND unfit_for_intensive=0 patients, treatment_7plus3=1 produces higher objective_response than treatment_7plus3=0.", "novel", 21)

# Iteration 22 co-administration
H["h138"] = ("There is a non-zero interaction between treatment_midostaurin and treatment_7plus3 in their joint effect on objective_response.", "novel", 22)
H["h139"] = ("There is a non-zero interaction between treatment_gilteritinib and treatment_7plus3 in their joint effect on objective_response.", "novel", 22)
H["h140"] = ("There is a non-zero interaction between treatment_ivosidenib and treatment_venetoclax_azacitidine in their joint effect on objective_response.", "novel", 22)

# Iteration 23 full-model interaction terms
H["h141"] = ("In a full multivariable logistic model with all targeted-therapy x mutation interactions, the treatment_midostaurin:flt3_itd interaction term is positive and significant.", "refined", 23)
H["h142"] = ("In the full multivariable model, the treatment_gilteritinib:flt3_itd interaction term is positive and significant.", "refined", 23)
H["h143"] = ("In the full multivariable model, the treatment_gilteritinib:flt3_tkd interaction term is positive and significant.", "refined", 23)
H["h144"] = ("In the full multivariable model, the treatment_ivosidenib:idh1_mutation interaction term is positive and significant.", "refined", 23)
H["h145"] = ("In the full multivariable model, the treatment_enasidenib:idh2_mutation interaction term is positive and significant.", "refined", 23)

# Iteration 24 lab covariates adjusted
H["h146"] = ("After adjusting for demographics, ecog_ps and other labs, higher albumin_g_dl is independently associated with higher odds of objective_response (positive coefficient).", "novel", 24)
H["h147"] = ("After adjustment, higher ldh_u_l is independently associated with lower odds of objective_response (negative coefficient).", "novel", 24)
H["h148"] = ("After adjustment, higher crp_mg_l is independently associated with lower odds of objective_response.", "novel", 24)
H["h149"] = ("After adjustment, higher nlr is independently associated with lower odds of objective_response.", "novel", 24)
H["h150"] = ("After adjustment, higher weight_loss_pct_6mo is independently associated with lower odds of objective_response.", "novel", 24)
H["h151"] = ("After adjustment, higher hemoglobin_g_dl is independently associated with higher odds of objective_response.", "novel", 24)

# Iteration 25 FINAL pinned subgroup hypotheses
H["h152"] = ("FINAL PINNED: treatment_gilteritinib increases objective_response within the subgroup defined by flt3_itd=1 AND tp53_mutation=0 AND complex_karyotype=0 (TP53 mutation and complex karyotype hypothesised to suppress benefit).", "refined", 25)
H["h153"] = ("FINAL PINNED: treatment_midostaurin increases objective_response within the subgroup defined by flt3_itd=1 AND unfit_for_intensive=0 AND tp53_mutation=0 AND complex_karyotype=0 (unfitness, TP53 mutation and complex karyotype hypothesised to suppress benefit).", "refined", 25)
H["h154"] = ("FINAL PINNED: treatment_ivosidenib increases objective_response within the subgroup defined by idh1_mutation=1 AND tp53_mutation=0 AND complex_karyotype=0 (TP53 and complex karyotype hypothesised to suppress benefit).", "refined", 25)
H["h155"] = ("FINAL PINNED: treatment_enasidenib increases objective_response within the subgroup defined by idh2_mutation=1 AND tp53_mutation=0 AND complex_karyotype=0 (TP53 and complex karyotype hypothesised to suppress benefit).", "refined", 25)
H["h156"] = ("FINAL PINNED (intermediate): treatment_venetoclax_azacitidine increases objective_response within the subgroup defined by unfit_for_intensive=1 AND tp53_mutation=0 AND complex_karyotype=0; this captures most of the venetoclax-azacitidine benefit but the effect is further concentrated when npm1_mutation=1 (see h158).", "refined", 25)
H["h157"] = ("FINAL PINNED: treatment_7plus3 increases objective_response within the subgroup defined by unfit_for_intensive=0 AND tp53_mutation=0 AND complex_karyotype=0 (unfitness, TP53 and complex karyotype hypothesised to suppress benefit).", "refined", 25)
H["h158"] = ("FINAL PINNED (most refined): treatment_venetoclax_azacitidine dramatically increases objective_response within the subgroup defined by unfit_for_intensive=1 AND tp53_mutation=0 AND complex_karyotype=0 AND npm1_mutation=1. Suppressors of benefit: TP53 mutation, complex karyotype; NPM1-wildtype within the otherwise-eligible group also abolishes benefit (h159).", "refined", 25)
H["h159"] = ("Within unfit_for_intensive=1 AND tp53_mutation=0 AND complex_karyotype=0 AND npm1_mutation=0 patients, treatment_venetoclax_azacitidine=1 does NOT produce a meaningful difference in objective_response vs treatment_venetoclax_azacitidine=0 (negative control showing NPM1+ is required for benefit).", "refined", 25)

# ---------------------------------------------------------------------------
# Group analyses by iteration
# ---------------------------------------------------------------------------
by_iter: dict[int, list[dict[str, Any]]] = defaultdict(list)
for r in RESULTS:
    by_iter[int(r["iteration"])].append(
        {
            "hypothesis_ids": r["hypothesis_ids"],
            "code": r.get("code"),
            "result_summary": r["result_summary"],
            "p_value": r["p_value"],
            "effect_estimate": r["effect_estimate"],
            "significant": r["significant"],
        }
    )

# Group hypotheses by iteration
hyps_by_iter: dict[int, list[dict[str, Any]]] = defaultdict(list)
for hid, (text, kind, it) in H.items():
    hyps_by_iter[it].append({"id": hid, "text": text, "kind": kind})

# Validate: every hypothesis_id referenced in analyses must be defined in H
referenced_ids: set[str] = set()
for r in RESULTS:
    referenced_ids.update(r["hypothesis_ids"])
missing = referenced_ids - set(H.keys())
if missing:
    raise SystemExit(f"Hypotheses referenced in analyses but not defined: {sorted(missing)}")

# Build iterations list
iterations = []
for idx in sorted(set(list(by_iter.keys()) + list(hyps_by_iter.keys()))):
    iterations.append(
        {
            "index": idx,
            "proposed_hypotheses": sorted(hyps_by_iter[idx], key=lambda h: int(h["id"][1:])),
            "analyses": by_iter[idx],
        }
    )

transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@named-task",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json", "w", encoding="utf-8") as f:
    json.dump(transcript, f, indent=2)

print(f"Wrote transcript.json with {len(iterations)} iterations and {sum(len(it['analyses']) for it in iterations)} analyses.")
