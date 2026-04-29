"""
Comprehensive iterative analysis of ds001_nsclc dataset.
Builds transcript.json and analysis_summary.txt via 25 propose-test-refine iterations.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
N = len(df)
print(f"Loaded {N} rows, {df.shape[1]} cols")

iterations = []
def fmt(x, k=4):
    if x is None: return "NA"
    if isinstance(x, (int, np.integer)): return str(int(x))
    return f"{x:.{k}g}"

# Helper for two-proportion z-test based on subgroup variable affecting ORR
def prop_compare(df, mask_a, mask_b, label_a, label_b, outcome="objective_response"):
    a = df.loc[mask_a, outcome]
    b = df.loc[mask_b, outcome]
    pa, pb = a.mean(), b.mean()
    diff = pa - pb
    # Use chi-square 2x2
    table = np.array([[a.sum(), len(a)-a.sum()], [b.sum(), len(b)-b.sum()]])
    chi2, p, _, _ = stats.chi2_contingency(table)
    return {
        "p_a": pa, "n_a": len(a), "p_b": pb, "n_b": len(b),
        "diff": diff, "chi2": chi2, "p_value": p,
        "summary": f"ORR {label_a} = {pa:.3f} (n={len(a)}) vs {label_b} = {pb:.3f} (n={len(b)}); diff = {diff:+.3f}; chi2 p = {p:.3g}",
    }

# Logistic regression - returns coef (log-odds) and p
def logit_fit(df, formula, outcome="objective_response"):
    f = f"{outcome} ~ {formula}"
    m = smf.logit(f, data=df).fit(disp=0, maxiter=200)
    return m

# === Iteration 1: Treatment main effects ===
it1 = {"index":1, "proposed_hypotheses":[], "analyses":[]}
it1["proposed_hypotheses"] = [
    {"id":"h1.1","text":"Patients receiving treatment_pembrolizumab have a higher objective_response rate than those not receiving treatment_pembrolizumab.","kind":"novel"},
    {"id":"h1.2","text":"Patients receiving treatment_sotorasib have a higher objective_response rate than those not receiving treatment_sotorasib.","kind":"novel"},
    {"id":"h1.3","text":"Patients receiving treatment_olaparib have a higher objective_response rate than those not receiving treatment_olaparib.","kind":"novel"},
    {"id":"h1.4","text":"Patients receiving treatment_osimertinib have a higher objective_response rate than those not receiving treatment_osimertinib.","kind":"novel"},
]
for hid, t in [("h1.1","treatment_pembrolizumab"),("h1.2","treatment_sotorasib"),("h1.3","treatment_olaparib"),("h1.4","treatment_osimertinib")]:
    r = prop_compare(df, df[t]==1, df[t]==0, f"{t}=1", f"{t}=0")
    it1["analyses"].append({
        "hypothesis_ids":[hid],
        "code": f"prop ORR by {t} (chi-square)",
        "result_summary": r["summary"],
        "p_value": r["p_value"],
        "effect_estimate": r["diff"],
        "significant": bool(r["p_value"]<0.05),
    })
iterations.append(it1)
print("Iter 1 done")

# === Iteration 2: Pembrolizumab × PD-L1 ===
it2 = {"index":2,"proposed_hypotheses":[
    {"id":"h2.1","text":"Higher pdl1_tps is associated with higher objective_response rate in patients receiving treatment_pembrolizumab (positive interaction).","kind":"novel"},
    {"id":"h2.2","text":"Among patients with pdl1_tps >= 0.5, treatment_pembrolizumab is associated with higher objective_response than no treatment_pembrolizumab.","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "treatment_pembrolizumab * pdl1_tps")
coef = m.params["treatment_pembrolizumab:pdl1_tps"]; pval = m.pvalues["treatment_pembrolizumab:pdl1_tps"]
it2["analyses"].append({"hypothesis_ids":["h2.1"],"code":"logit(orr ~ pembro * pdl1_tps)","result_summary":f"Interaction coef (log-odds) = {coef:+.3f}, p = {pval:.3g}; main pembro coef = {m.params['treatment_pembrolizumab']:+.3f}; pdl1 coef = {m.params['pdl1_tps']:+.3f}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
high = df[df["pdl1_tps"]>=0.5]
r = prop_compare(high, high["treatment_pembrolizumab"]==1, high["treatment_pembrolizumab"]==0, "pembro+/PDL1>=0.5", "pembro-/PDL1>=0.5")
it2["analyses"].append({"hypothesis_ids":["h2.2"],"code":"chi-square ORR by pembro within PDL1>=0.5","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
iterations.append(it2)
print("Iter 2 done")

# === Iteration 3: Pembrolizumab × TMB high ===
it3 = {"index":3,"proposed_hypotheses":[
    {"id":"h3.1","text":"tmb_high modifies the effect of treatment_pembrolizumab on objective_response: pembrolizumab response rate is higher in tmb_high=1 patients.","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "treatment_pembrolizumab * tmb_high")
coef = m.params["treatment_pembrolizumab:tmb_high"]; pval = m.pvalues["treatment_pembrolizumab:tmb_high"]
sub_high = df[df["tmb_high"]==1]
sub_low = df[df["tmb_high"]==0]
r_high = prop_compare(sub_high, sub_high["treatment_pembrolizumab"]==1, sub_high["treatment_pembrolizumab"]==0, "pembro+/tmb=1","pembro-/tmb=1")
r_low = prop_compare(sub_low, sub_low["treatment_pembrolizumab"]==1, sub_low["treatment_pembrolizumab"]==0, "pembro+/tmb=0","pembro-/tmb=0")
it3["analyses"].append({"hypothesis_ids":["h3.1"],"code":"logit(orr ~ pembro * tmb_high)","result_summary":f"Interaction coef = {coef:+.3f}, p = {pval:.3g}; in TMB-high pembro vs not: {r_high['summary']}; in TMB-low: {r_low['summary']}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
iterations.append(it3)
print("Iter 3 done")

# === Iteration 4: Osimertinib × EGFR ===
it4 = {"index":4,"proposed_hypotheses":[
    {"id":"h4.1","text":"Among patients with egfr_mutation=1, treatment_osimertinib is associated with higher objective_response rate than no osimertinib (matched targeted therapy).","kind":"novel"},
    {"id":"h4.2","text":"There is a positive interaction between treatment_osimertinib and egfr_mutation on objective_response.","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "treatment_osimertinib * egfr_mutation")
coef = m.params["treatment_osimertinib:egfr_mutation"]; pval = m.pvalues["treatment_osimertinib:egfr_mutation"]
sub = df[df["egfr_mutation"]==1]
r = prop_compare(sub, sub["treatment_osimertinib"]==1, sub["treatment_osimertinib"]==0, "osi+/EGFR+","osi-/EGFR+")
it4["analyses"].append({"hypothesis_ids":["h4.1"],"code":"chi-square ORR by osimertinib within EGFR+","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
it4["analyses"].append({"hypothesis_ids":["h4.2"],"code":"logit(orr ~ osimertinib * egfr_mutation)","result_summary":f"Interaction coef = {coef:+.3f}, p = {pval:.3g}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
iterations.append(it4)
print("Iter 4 done")

# === Iteration 5: Sotorasib × KRAS G12C ===
it5 = {"index":5,"proposed_hypotheses":[
    {"id":"h5.1","text":"Among patients with kras_g12c=1, treatment_sotorasib is associated with higher objective_response than no sotorasib (matched targeted therapy).","kind":"novel"},
    {"id":"h5.2","text":"There is a positive interaction between treatment_sotorasib and kras_g12c on objective_response.","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "treatment_sotorasib * kras_g12c")
coef = m.params["treatment_sotorasib:kras_g12c"]; pval = m.pvalues["treatment_sotorasib:kras_g12c"]
sub = df[df["kras_g12c"]==1]
r = prop_compare(sub, sub["treatment_sotorasib"]==1, sub["treatment_sotorasib"]==0, "soto+/KRAS+","soto-/KRAS+")
it5["analyses"].append({"hypothesis_ids":["h5.1"],"code":"chi-square ORR by sotorasib within KRAS_G12C+","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
it5["analyses"].append({"hypothesis_ids":["h5.2"],"code":"logit(orr ~ sotorasib * kras_g12c)","result_summary":f"Interaction coef = {coef:+.3f}, p = {pval:.3g}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
iterations.append(it5)
print("Iter 5 done")

# === Iteration 6: Olaparib × BRCA2 ===
it6 = {"index":6,"proposed_hypotheses":[
    {"id":"h6.1","text":"Among patients with brca2_mutation=1, treatment_olaparib is associated with higher objective_response than no olaparib (matched PARP inhibitor).","kind":"novel"},
    {"id":"h6.2","text":"There is a positive interaction between treatment_olaparib and brca2_mutation on objective_response.","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "treatment_olaparib * brca2_mutation")
coef = m.params["treatment_olaparib:brca2_mutation"]; pval = m.pvalues["treatment_olaparib:brca2_mutation"]
sub = df[df["brca2_mutation"]==1]
r = prop_compare(sub, sub["treatment_olaparib"]==1, sub["treatment_olaparib"]==0, "ola+/BRCA2+","ola-/BRCA2+")
it6["analyses"].append({"hypothesis_ids":["h6.1"],"code":"chi-square ORR by olaparib within BRCA2+","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
it6["analyses"].append({"hypothesis_ids":["h6.2"],"code":"logit(orr ~ olaparib * brca2_mutation)","result_summary":f"Interaction coef = {coef:+.3f}, p = {pval:.3g}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
iterations.append(it6)
print("Iter 6 done")

# === Iteration 7: STK11 negative interaction with pembrolizumab ===
it7 = {"index":7,"proposed_hypotheses":[
    {"id":"h7.1","text":"stk11_mutation modifies the effect of treatment_pembrolizumab on objective_response: pembrolizumab response is lower in stk11_mutation=1 patients (negative interaction).","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "treatment_pembrolizumab * stk11_mutation")
coef = m.params["treatment_pembrolizumab:stk11_mutation"]; pval = m.pvalues["treatment_pembrolizumab:stk11_mutation"]
sub_s1 = df[df["stk11_mutation"]==1]
sub_s0 = df[df["stk11_mutation"]==0]
r1 = prop_compare(sub_s1, sub_s1["treatment_pembrolizumab"]==1, sub_s1["treatment_pembrolizumab"]==0, "pembro+/STK11+","pembro-/STK11+")
r0 = prop_compare(sub_s0, sub_s0["treatment_pembrolizumab"]==1, sub_s0["treatment_pembrolizumab"]==0, "pembro+/STK11-","pembro-/STK11-")
it7["analyses"].append({"hypothesis_ids":["h7.1"],"code":"logit(orr ~ pembro * stk11)","result_summary":f"Interaction coef = {coef:+.3f}, p = {pval:.3g}; in STK11+ {r1['summary']}; in STK11- {r0['summary']}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
iterations.append(it7)
print("Iter 7 done")

# === Iteration 8: KEAP1 negative interaction with pembrolizumab ===
it8 = {"index":8,"proposed_hypotheses":[
    {"id":"h8.1","text":"keap1_mutation modifies the effect of treatment_pembrolizumab on objective_response: pembrolizumab response is lower in keap1_mutation=1 patients (negative interaction).","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "treatment_pembrolizumab * keap1_mutation")
coef = m.params["treatment_pembrolizumab:keap1_mutation"]; pval = m.pvalues["treatment_pembrolizumab:keap1_mutation"]
sub = df[df["keap1_mutation"]==1]
r = prop_compare(sub, sub["treatment_pembrolizumab"]==1, sub["treatment_pembrolizumab"]==0, "pembro+/KEAP1+","pembro-/KEAP1+")
it8["analyses"].append({"hypothesis_ids":["h8.1"],"code":"logit(orr ~ pembro * keap1)","result_summary":f"Interaction coef = {coef:+.3f}, p = {pval:.3g}; in KEAP1+: {r['summary']}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
iterations.append(it8)
print("Iter 8 done")

# === Iteration 9: ECOG PS main effect ===
it9 = {"index":9,"proposed_hypotheses":[
    {"id":"h9.1","text":"Higher ecog_ps is associated with lower objective_response (worse performance status reduces response).","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "ecog_ps")
coef = m.params["ecog_ps"]; pval = m.pvalues["ecog_ps"]
by_ecog = df.groupby("ecog_ps")["objective_response"].agg(["mean","count"])
it9["analyses"].append({"hypothesis_ids":["h9.1"],"code":"logit(orr ~ ecog_ps)","result_summary":f"ECOG PS coef = {coef:+.3f} log-odds per unit, p = {pval:.3g}; ORR by ECOG: {by_ecog.to_dict()}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
iterations.append(it9)
print("Iter 9 done")

# === Iteration 10: Albumin and other lab markers ===
it10 = {"index":10,"proposed_hypotheses":[
    {"id":"h10.1","text":"Higher albumin_g_dl is associated with higher objective_response.","kind":"novel"},
    {"id":"h10.2","text":"Higher ldh_u_l is associated with lower objective_response.","kind":"novel"},
    {"id":"h10.3","text":"Higher crp_mg_l is associated with lower objective_response.","kind":"novel"},
],"analyses":[]}
for hid, var in [("h10.1","albumin_g_dl"),("h10.2","ldh_u_l"),("h10.3","crp_mg_l")]:
    m = logit_fit(df, var)
    coef = m.params[var]; pval = m.pvalues[var]
    it10["analyses"].append({"hypothesis_ids":[hid],"code":f"logit(orr ~ {var})","result_summary":f"{var} coef = {coef:+.5f} log-odds per unit, p = {pval:.3g}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
iterations.append(it10)
print("Iter 10 done")

# === Iteration 11: NLR ===
it11 = {"index":11,"proposed_hypotheses":[
    {"id":"h11.1","text":"Higher nlr (neutrophil-to-lymphocyte ratio) is associated with lower objective_response.","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "nlr")
coef = m.params["nlr"]; pval = m.pvalues["nlr"]
it11["analyses"].append({"hypothesis_ids":["h11.1"],"code":"logit(orr ~ nlr)","result_summary":f"NLR coef = {coef:+.4f} log-odds per unit, p = {pval:.3g}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
iterations.append(it11)
print("Iter 11 done")

# === Iteration 12: Weight loss ===
it12 = {"index":12,"proposed_hypotheses":[
    {"id":"h12.1","text":"Higher weight_loss_pct_6mo is associated with lower objective_response.","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "weight_loss_pct_6mo")
coef = m.params["weight_loss_pct_6mo"]; pval = m.pvalues["weight_loss_pct_6mo"]
it12["analyses"].append({"hypothesis_ids":["h12.1"],"code":"logit(orr ~ weight_loss_pct_6mo)","result_summary":f"Weight loss coef = {coef:+.4f} per pct point, p = {pval:.3g}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
iterations.append(it12)
print("Iter 12 done")

# === Iteration 13: Brain mets / liver mets ===
it13 = {"index":13,"proposed_hypotheses":[
    {"id":"h13.1","text":"has_brain_mets=1 is associated with lower objective_response.","kind":"novel"},
    {"id":"h13.2","text":"liver_mets=1 is associated with lower objective_response.","kind":"novel"},
    {"id":"h13.3","text":"bone_mets=1 is associated with lower objective_response.","kind":"novel"},
],"analyses":[]}
for hid, v in [("h13.1","has_brain_mets"),("h13.2","liver_mets"),("h13.3","bone_mets")]:
    r = prop_compare(df, df[v]==1, df[v]==0, f"{v}=1", f"{v}=0")
    it13["analyses"].append({"hypothesis_ids":[hid],"code":f"chi-square orr by {v}","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
iterations.append(it13)
print("Iter 13 done")

# === Iteration 14: Smoking status ===
it14 = {"index":14,"proposed_hypotheses":[
    {"id":"h14.1","text":"Patients with smoking_status='current' have higher objective_response than 'never' (smoking-related immunogenicity).","kind":"novel"},
    {"id":"h14.2","text":"Within treatment_pembrolizumab patients, current/former smokers have higher objective_response than never-smokers.","kind":"novel"},
],"analyses":[]}
ct = df.groupby("smoking_status")["objective_response"].agg(["mean","count"]).to_dict()
table = pd.crosstab(df["smoking_status"], df["objective_response"])
chi2, p, _, _ = stats.chi2_contingency(table)
diff = df.loc[df["smoking_status"]=="current","objective_response"].mean() - df.loc[df["smoking_status"]=="never","objective_response"].mean()
it14["analyses"].append({"hypothesis_ids":["h14.1"],"code":"chi-square smoking_status × orr","result_summary":f"ORR by smoking: {ct}; chi2 p = {p:.3g}; current minus never = {diff:+.3f}.","p_value":float(p),"effect_estimate":float(diff),"significant":bool(p<0.05)})
sub = df[df["treatment_pembrolizumab"]==1]
r = prop_compare(sub, sub["smoking_status"].isin(["current","former"]), sub["smoking_status"]=="never", "pembro+ ever-smoker","pembro+ never-smoker")
it14["analyses"].append({"hypothesis_ids":["h14.2"],"code":"chi-square ever vs never smoker within pembro+","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
iterations.append(it14)
print("Iter 14 done")

# === Iteration 15: Histology ===
it15 = {"index":15,"proposed_hypotheses":[
    {"id":"h15.1","text":"Patients with histology='squamous' differ in objective_response rate compared to 'adenocarcinoma'.","kind":"novel"},
],"analyses":[]}
r = prop_compare(df, df["histology"]=="squamous", df["histology"]=="adenocarcinoma", "squamous","adeno")
it15["analyses"].append({"hypothesis_ids":["h15.1"],"code":"chi-square orr by histology","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
iterations.append(it15)
print("Iter 15 done")

# === Iteration 16: Sex × pembrolizumab ===
it16 = {"index":16,"proposed_hypotheses":[
    {"id":"h16.1","text":"sex_female modifies the effect of treatment_pembrolizumab on objective_response.","kind":"novel"},
    {"id":"h16.2","text":"sex_female has a main effect on objective_response.","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "treatment_pembrolizumab * sex_female")
coef = m.params["treatment_pembrolizumab:sex_female"]; pval = m.pvalues["treatment_pembrolizumab:sex_female"]
it16["analyses"].append({"hypothesis_ids":["h16.1"],"code":"logit(orr ~ pembro * sex_female)","result_summary":f"Interaction coef = {coef:+.3f}, p = {pval:.3g}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
r = prop_compare(df, df["sex_female"]==1, df["sex_female"]==0, "female","male")
it16["analyses"].append({"hypothesis_ids":["h16.2"],"code":"chi-square orr by sex","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
iterations.append(it16)
print("Iter 16 done")

# === Iteration 17: Age ===
it17 = {"index":17,"proposed_hypotheses":[
    {"id":"h17.1","text":"Higher age_years is associated with lower objective_response.","kind":"novel"},
],"analyses":[]}
m = logit_fit(df, "age_years")
coef = m.params["age_years"]; pval = m.pvalues["age_years"]
it17["analyses"].append({"hypothesis_ids":["h17.1"],"code":"logit(orr ~ age_years)","result_summary":f"Age coef = {coef:+.4f} per year, p = {pval:.3g}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
iterations.append(it17)
print("Iter 17 done")

# === Iteration 18: TP53 main effect ===
it18 = {"index":18,"proposed_hypotheses":[
    {"id":"h18.1","text":"tp53_mutation=1 is associated with lower objective_response.","kind":"novel"},
],"analyses":[]}
r = prop_compare(df, df["tp53_mutation"]==1, df["tp53_mutation"]==0, "TP53+","TP53-")
it18["analyses"].append({"hypothesis_ids":["h18.1"],"code":"chi-square orr by tp53_mutation","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
iterations.append(it18)
print("Iter 18 done")

# === Iteration 19: SNP scan (sanity check; expect mostly null) ===
it19 = {"index":19,"proposed_hypotheses":[
    {"id":"h19.1","text":"None of the surveyed SNPs (snp_rs1045642, snp_rs1065852, snp_rs1799853, snp_rs1800566, snp_rs2228001, snp_rs1801133, snp_rs429358, snp_rs7412, snp_rs662) are associated with objective_response after considering multiple comparisons.","kind":"novel"},
],"analyses":[]}
snp_cols = [c for c in df.columns if c.startswith("snp_")]
results = []
for snp in snp_cols:
    try:
        m = logit_fit(df, snp)
        c = m.params[snp]; p = m.pvalues[snp]
        results.append((snp, c, p))
    except Exception:
        pass
sig_snps = [(s,c,p) for s,c,p in results if p < 0.05]
min_p = min(p for _,_,p in results)
n_tests = len(results)
bonf_thresh = 0.05/n_tests
n_sig_bonf = sum(1 for _,_,p in results if p < bonf_thresh)
it19["analyses"].append({"hypothesis_ids":["h19.1"],"code":"per-SNP logit(orr ~ snp); count nominal and Bonferroni-significant","result_summary":f"Tested {n_tests} SNPs; {len(sig_snps)} nominally significant (p<0.05); min p = {min_p:.3g}; Bonferroni threshold {bonf_thresh:.3g}; {n_sig_bonf} significant after correction. Nominal hits: {sig_snps[:5]}.","p_value":float(min_p),"effect_estimate":float(0 if not results else max(abs(c) for _,c,_ in results)),"significant":bool(n_sig_bonf>0)})
iterations.append(it19)
print("Iter 19 done")

# === Iteration 20: Race / ethnicity ===
it20 = {"index":20,"proposed_hypotheses":[
    {"id":"h20.1","text":"objective_response rate differs by race_ethnicity category.","kind":"novel"},
],"analyses":[]}
ct = df.groupby("race_ethnicity")["objective_response"].agg(["mean","count"]).to_dict()
table = pd.crosstab(df["race_ethnicity"], df["objective_response"])
chi2, p, _, _ = stats.chi2_contingency(table)
diff = df.loc[df["race_ethnicity"]=="white","objective_response"].mean() - df.loc[df["race_ethnicity"]=="black","objective_response"].mean()
it20["analyses"].append({"hypothesis_ids":["h20.1"],"code":"chi-square race_ethnicity × orr","result_summary":f"ORR by race: {ct}; chi2 p = {p:.3g}; white minus black = {diff:+.3f}.","p_value":float(p),"effect_estimate":float(diff),"significant":bool(p<0.05)})
iterations.append(it20)
print("Iter 20 done")

# === Iteration 21: Insurance / SES ===
it21 = {"index":21,"proposed_hypotheses":[
    {"id":"h21.1","text":"objective_response rate differs by insurance_type.","kind":"novel"},
    {"id":"h21.2","text":"rural_residence=1 is associated with different objective_response than rural_residence=0.","kind":"novel"},
],"analyses":[]}
ct = df.groupby("insurance_type")["objective_response"].agg(["mean","count"]).to_dict()
table = pd.crosstab(df["insurance_type"], df["objective_response"])
chi2, p, _, _ = stats.chi2_contingency(table)
diff = df.loc[df["insurance_type"]=="private","objective_response"].mean() - df.loc[df["insurance_type"]=="uninsured","objective_response"].mean()
it21["analyses"].append({"hypothesis_ids":["h21.1"],"code":"chi-square insurance × orr","result_summary":f"ORR by insurance: {ct}; chi2 p = {p:.3g}; private minus uninsured = {diff:+.3f}.","p_value":float(p),"effect_estimate":float(diff),"significant":bool(p<0.05)})
r = prop_compare(df, df["rural_residence"]==1, df["rural_residence"]==0, "rural=1","rural=0")
it21["analyses"].append({"hypothesis_ids":["h21.2"],"code":"chi-square orr by rural_residence","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
iterations.append(it21)
print("Iter 21 done")

# === Iteration 22: ALK fusion main and pembro ===
it22 = {"index":22,"proposed_hypotheses":[
    {"id":"h22.1","text":"alk_fusion=1 is associated with lower objective_response in pembrolizumab-treated patients (ALK-rearranged NSCLC tends to be less responsive to PD-1 blockade).","kind":"novel"},
],"analyses":[]}
sub = df[df["treatment_pembrolizumab"]==1]
r = prop_compare(sub, sub["alk_fusion"]==1, sub["alk_fusion"]==0, "pembro+/ALK+","pembro+/ALK-")
it22["analyses"].append({"hypothesis_ids":["h22.1"],"code":"chi-square ORR by alk_fusion within pembro+","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
iterations.append(it22)
print("Iter 22 done")

# === Iteration 23: Multivariable model (key features) ===
it23 = {"index":23,"proposed_hypotheses":[
    {"id":"h23.1","text":"In a multivariable logistic regression including treatments, biomarkers, ECOG PS, albumin, NLR, brain/liver mets, age, and PDL1, the matched-targeted-therapy interactions (osimertinib×egfr, sotorasib×kras_g12c, olaparib×brca2) and pembrolizumab×pdl1_tps remain positively and significantly associated with objective_response.","kind":"refined"},
],"analyses":[]}
formula = ("treatment_pembrolizumab*pdl1_tps + treatment_pembrolizumab*tmb_high + "
           "treatment_pembrolizumab*stk11_mutation + treatment_pembrolizumab*keap1_mutation + "
           "treatment_osimertinib*egfr_mutation + treatment_sotorasib*kras_g12c + "
           "treatment_olaparib*brca2_mutation + ecog_ps + albumin_g_dl + ldh_u_l + crp_mg_l + nlr + "
           "weight_loss_pct_6mo + has_brain_mets + liver_mets + bone_mets + age_years + sex_female + "
           "C(smoking_status) + C(histology)")
m23 = logit_fit(df, formula)
key_terms = ["treatment_pembrolizumab:pdl1_tps","treatment_pembrolizumab:tmb_high","treatment_pembrolizumab:stk11_mutation","treatment_pembrolizumab:keap1_mutation","treatment_osimertinib:egfr_mutation","treatment_sotorasib:kras_g12c","treatment_olaparib:brca2_mutation","ecog_ps","albumin_g_dl","crp_mg_l","nlr","ldh_u_l","weight_loss_pct_6mo","has_brain_mets","liver_mets","age_years"]
summary_lines = []
for t in key_terms:
    if t in m23.params.index:
        summary_lines.append(f"{t}: {m23.params[t]:+.4f} (p={m23.pvalues[t]:.3g})")
it23["analyses"].append({"hypothesis_ids":["h23.1"],"code":"multivariable logit with treatment×biomarker interactions and prognostic covariates","result_summary":"Adjusted multivariable: " + "; ".join(summary_lines),
    "p_value": float(m23.pvalues.get("treatment_osimertinib:egfr_mutation", 1.0)),
    "effect_estimate": float(m23.params.get("treatment_osimertinib:egfr_mutation", 0.0)),
    "significant": bool(m23.pvalues.get("treatment_osimertinib:egfr_mutation", 1.0) < 0.05)})
# Save the full fit summary text
multivar_text = m23.summary().as_text()
iterations.append(it23)
print("Iter 23 done")

# === Iteration 24: Combined PDL1+TMB groups for pembro effect size ===
it24 = {"index":24,"proposed_hypotheses":[
    {"id":"h24.1","text":"Among patients with pdl1_tps>=0.5 AND tmb_high=1, treatment_pembrolizumab is associated with markedly higher objective_response than no pembrolizumab (combined biomarker enrichment).","kind":"refined"},
],"analyses":[]}
sub = df[(df["pdl1_tps"]>=0.5) & (df["tmb_high"]==1)]
r = prop_compare(sub, sub["treatment_pembrolizumab"]==1, sub["treatment_pembrolizumab"]==0, "pembro+/PDL1>=0.5+TMBhi","pembro-/PDL1>=0.5+TMBhi")
it24["analyses"].append({"hypothesis_ids":["h24.1"],"code":"chi-square ORR by pembro within PDL1>=0.5 & TMB-high","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
iterations.append(it24)
print("Iter 24 done")

# === Iteration 25: Stage IV; symptoms; final synthesis test ===
it25 = {"index":25,"proposed_hypotheses":[
    {"id":"h25.1","text":"stage_iv=1 is associated with lower objective_response.","kind":"novel"},
    {"id":"h25.2","text":"Higher fatigue_grade is associated with lower objective_response.","kind":"novel"},
    {"id":"h25.3","text":"In a final adjusted model, treatment_pembrolizumab interacts negatively with stk11_mutation AND with keap1_mutation, and positively with pdl1_tps; these interactions remain significant after adjustment for prognostic covariates.","kind":"refined"},
],"analyses":[]}
r = prop_compare(df, df["stage_iv"]==1, df["stage_iv"]==0, "stage IV","not stage IV")
it25["analyses"].append({"hypothesis_ids":["h25.1"],"code":"chi-square orr by stage_iv","result_summary":r["summary"],"p_value":r["p_value"],"effect_estimate":r["diff"],"significant":bool(r["p_value"]<0.05)})
m = logit_fit(df, "fatigue_grade")
coef = m.params["fatigue_grade"]; pval = m.pvalues["fatigue_grade"]
it25["analyses"].append({"hypothesis_ids":["h25.2"],"code":"logit(orr ~ fatigue_grade)","result_summary":f"fatigue coef = {coef:+.4f}, p = {pval:.3g}.","p_value":float(pval),"effect_estimate":float(coef),"significant":bool(pval<0.05)})
# Reuse adjusted multivariable model; report stk11 and keap1 interaction
stk11_c = m23.params["treatment_pembrolizumab:stk11_mutation"]; stk11_p = m23.pvalues["treatment_pembrolizumab:stk11_mutation"]
keap1_c = m23.params["treatment_pembrolizumab:keap1_mutation"]; keap1_p = m23.pvalues["treatment_pembrolizumab:keap1_mutation"]
pdl1_c = m23.params["treatment_pembrolizumab:pdl1_tps"]; pdl1_p = m23.pvalues["treatment_pembrolizumab:pdl1_tps"]
it25["analyses"].append({"hypothesis_ids":["h25.3"],"code":"adjusted multivariable interaction estimates from iteration 23","result_summary":f"Adjusted: pembro×STK11 = {stk11_c:+.3f} (p={stk11_p:.3g}); pembro×KEAP1 = {keap1_c:+.3f} (p={keap1_p:.3g}); pembro×PDL1 = {pdl1_c:+.3f} (p={pdl1_p:.3g}).","p_value":float(stk11_p),"effect_estimate":float(stk11_c),"significant":bool(stk11_p<0.05)})
iterations.append(it25)
print("Iter 25 done")

# === Build transcript ===
transcript = {
    "dataset_id": "ds001_nsclc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@manual-2026-04-28",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json","w") as f:
    json.dump(transcript, f, indent=2, default=lambda o: float(o) if isinstance(o,(np.floating,np.integer)) else str(o))

# === Save adjacent multivariable text for narrative ===
with open("multivariable_fit.txt","w") as f:
    f.write(multivar_text)

print("Transcript saved.")
