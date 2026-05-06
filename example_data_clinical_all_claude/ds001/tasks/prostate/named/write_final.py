"""Build final transcript.json and analysis_summary.txt from all_results.json."""
import json
from collections import defaultdict

RESULTS = json.load(open("all_results.json"))

# Group by iteration
iters = defaultdict(lambda: {"hyps": {}, "analyses": []})
for r in RESULTS:
    it = r["iter"]
    hid = r["hyp_id"]
    if hid not in iters[it]["hyps"]:
        iters[it]["hyps"][hid] = {
            "id": hid,
            "text": r["hyp_text"],
            "kind": r.get("kind","novel"),
        }
    a = {
        "hypothesis_ids": [hid],
        "code": r.get("code"),
        "result_summary": r["summary"],
    }
    if r["p"] is not None: a["p_value"] = r["p"]
    if r["eff"] is not None: a["effect_estimate"] = r["eff"]
    if r["sig"] is not None: a["significant"] = r["sig"]
    iters[it]["analyses"].append(a)

iter_records = []
for it in sorted(iters):
    iter_records.append({
        "index": it,
        "proposed_hypotheses": list(iters[it]["hyps"].values()),
        "analyses": iters[it]["analyses"],
    })

transcript = {
    "dataset_id": "ds001_prostate",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-opus-4-7@manual-iter",
    "max_iterations": 25,
    "iterations": iter_records,
}

with open("transcript.json","w") as f:
    json.dump(transcript, f, indent=2)
print("transcript.json written. iterations =", len(iter_records),
      "; hypotheses =", sum(len(it["proposed_hypotheses"]) for it in iter_records),
      "; analyses =", sum(len(it["analyses"]) for it in iter_records))
