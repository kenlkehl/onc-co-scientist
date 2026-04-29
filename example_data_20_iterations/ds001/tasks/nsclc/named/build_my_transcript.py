"""Convert my_results.json into a schema-conformant transcript.json."""
import json, math
from collections import defaultdict

with open("my_results.json") as f:
    rows = json.load(f)

iters = defaultdict(lambda: {"hypotheses": {}, "analyses": []})

for r in rows:
    idx = int(r["iter"])
    hid = r["hyp_id"]
    if hid not in iters[idx]["hypotheses"]:
        iters[idx]["hypotheses"][hid] = {
            "id": hid,
            "text": r["hyp_text"],
            "kind": r.get("kind", "novel"),
        }
    eff = r["effect_estimate"]
    pval = r["p_value"]
    try:
        eff = float(eff)
        if math.isnan(eff): eff = None
    except Exception:
        eff = None
    try:
        pval = float(pval)
        if math.isnan(pval): pval = None
    except Exception:
        pval = None
    iters[idx]["analyses"].append({
        "hypothesis_ids": [hid],
        "code": r.get("code"),
        "result_summary": r["result_summary"],
        "p_value": pval,
        "effect_estimate": eff,
        "significant": bool(r.get("significant", False)),
    })

iterations = []
for idx in sorted(iters):
    it = iters[idx]
    iterations.append({
        "index": idx,
        "proposed_hypotheses": list(it["hypotheses"].values()),
        "analyses": it["analyses"],
    })

transcript = {
    "dataset_id": "ds001_nsclc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@interactive",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

# Print quick summary
total_h = sum(len(it["proposed_hypotheses"]) for it in iterations)
total_a = sum(len(it["analyses"]) for it in iterations)
sig = sum(1 for it in iterations for a in it["analyses"] if a["significant"])
print(f"Iterations: {len(iterations)}")
print(f"Total hypotheses: {total_h}")
print(f"Total analyses: {total_a}")
print(f"Significant: {sig}")
