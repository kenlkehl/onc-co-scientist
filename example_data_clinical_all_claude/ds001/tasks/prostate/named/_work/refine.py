"""Refine — characterize tree leaves and confirm key subgroup hypotheses."""
import pandas as pd, numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
import json
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
trts = ["treatment_enzalutamide","treatment_abiraterone","treatment_docetaxel",
        "treatment_olaparib","treatment_lu177_psma","treatment_pembrolizumab"]

out = {}

# 1) Re-fit tree for enzalutamide and dump the tree structure
treated = df[df["treatment_enzalutamide"]==1]
untreated = df[df["treatment_enzalutamide"]==0]
feat_cols = [c for c in df.columns if c not in ("patient_id","objective_response") + tuple(trts)]
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=0)
tree.fit(treated[feat_cols], treated["objective_response"])
out["enza_tree_text"] = export_text(tree, feature_names=feat_cols, max_depth=3)

# 2) Characterize patients who fall into leaf 3 (the high-RD leaf)
leaf_t = tree.apply(treated[feat_cols])
leaf_u = tree.apply(untreated[feat_cols])
mask_t_l3 = (leaf_t == 3)
mask_u_l3 = (leaf_u == 3)
patients_l3 = pd.concat([treated[mask_t_l3], untreated[mask_u_l3]])
out["leaf3_n"] = int(len(patients_l3))
out["leaf3_summary"] = {
    "mcrpc_pct": float(patients_l3["mcrpc"].mean()),
    "ar_v7_pct": float(patients_l3["ar_v7_positive"].mean()),
    "ecog_dist": patients_l3["ecog_ps"].value_counts().sort_index().to_dict(),
    "visceral_pct": float(patients_l3["visceral_mets"].mean()),
    "brca2_pct": float(patients_l3["brca2_mutation"].mean()),
    "msi_high_pct": float(patients_l3["msi_high"].mean()),
    "psma_high_pct": float(patients_l3["psma_high"].mean()),
    "median_psa": float(patients_l3["psa_ng_ml"].median()),
}

# 3) Confirm the "natural" subgroup mCRPC=0 AND AR-V7-negative
sub = df[(df["mcrpc"]==0) & (df["ar_v7_positive"]==0)]
on = sub.loc[sub["treatment_enzalutamide"]==1, "objective_response"]
off = sub.loc[sub["treatment_enzalutamide"]==0, "objective_response"]
out["enza_mcrpc0_arv7neg"] = dict(
    n_on=int(len(on)), n_off=int(len(off)),
    p_on=float(on.mean()), p_off=float(off.mean()),
    rd=float(on.mean()-off.mean()))

# 4) Olaparib in BRCA2 split by other modifiers — confirm direction
sub = df[df["brca2_mutation"]==1]
on = sub.loc[sub["treatment_olaparib"]==1, "objective_response"]
off = sub.loc[sub["treatment_olaparib"]==0, "objective_response"]
out["olap_in_brca2"] = dict(
    n_on=int(len(on)), n_off=int(len(off)),
    p_on=float(on.mean()), p_off=float(off.mean()),
    rd=float(on.mean()-off.mean()))

# 5) Counter-intuitive direction check: olaparib + BRCA2 + non-mCRPC?
for m in [0,1]:
    sub = df[(df["brca2_mutation"]==1) & (df["mcrpc"]==m)]
    on = sub.loc[sub["treatment_olaparib"]==1, "objective_response"]
    off = sub.loc[sub["treatment_olaparib"]==0, "objective_response"]
    out[f"olap_brca2_mcrpc={m}"] = dict(
        n_on=int(len(on)), n_off=int(len(off)),
        p_on=float(on.mean()) if len(on) else None,
        p_off=float(off.mean()) if len(off) else None,
        rd=(float(on.mean()-off.mean()) if (len(on) and len(off)) else None))

# 6) lu177 in PSMA-high + mCRPC=0?
sub = df[(df["psma_high"]==1) & (df["mcrpc"]==0)]
on = sub.loc[sub["treatment_lu177_psma"]==1, "objective_response"]
off = sub.loc[sub["treatment_lu177_psma"]==0, "objective_response"]
out["lu177_psmahi_mcrpc0"] = dict(
    n_on=int(len(on)), n_off=int(len(off)),
    p_on=float(on.mean()), p_off=float(off.mean()),
    rd=float(on.mean()-off.mean()))

sub = df[(df["psma_high"]==1) & (df["mcrpc"]==1)]
on = sub.loc[sub["treatment_lu177_psma"]==1, "objective_response"]
off = sub.loc[sub["treatment_lu177_psma"]==0, "objective_response"]
out["lu177_psmahi_mcrpc1"] = dict(
    n_on=int(len(on)), n_off=int(len(off)),
    p_on=float(on.mean()), p_off=float(off.mean()),
    rd=float(on.mean()-off.mean()))

# 7) Pembro in MSI-high + mCRPC=0
sub = df[(df["msi_high"]==1) & (df["mcrpc"]==0)]
on = sub.loc[sub["treatment_pembrolizumab"]==1, "objective_response"]
off = sub.loc[sub["treatment_pembrolizumab"]==0, "objective_response"]
out["pembro_msihi_mcrpc0"] = dict(
    n_on=int(len(on)), n_off=int(len(off)),
    p_on=float(on.mean()), p_off=float(off.mean()),
    rd=float(on.mean()-off.mean()))

# 8) Final enzalutamide subgroup — full predicates: mCRPC=0 AND AR-V7-neg AND ECOG<=1?
for ec_max in [0,1,2]:
    sub = df[(df["mcrpc"]==0) & (df["ar_v7_positive"]==0) & (df["ecog_ps"]<=ec_max)]
    on = sub.loc[sub["treatment_enzalutamide"]==1, "objective_response"]
    off = sub.loc[sub["treatment_enzalutamide"]==0, "objective_response"]
    out[f"enza_mcrpc0_arv7neg_ecog<={ec_max}"] = dict(
        n_on=int(len(on)), n_off=int(len(off)),
        p_on=float(on.mean()), p_off=float(off.mean()),
        rd=float(on.mean()-off.mean()))

print(json.dumps(out, indent=1, default=str))
