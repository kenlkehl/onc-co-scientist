"""Build transcript.json from results_out.json."""
import json

with open("results_out.json") as f:
    out = json.load(f)

iterations = []
for it in out:
    iterations.append({
        "index": it["index"],
        "proposed_hypotheses": it["hypotheses"],
        "analyses": [
            {
                "hypothesis_ids": a["hids"],
                "code": a.get("code"),
                "result_summary": a["summary"],
                "p_value": a.get("p"),
                "effect_estimate": a.get("eff"),
                "significant": a.get("sig"),
            }
            for a in it["analyses"]
        ],
    })

transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-interactive@manual",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json", "w", encoding="utf-8") as f:
    json.dump(transcript, f, indent=2, ensure_ascii=False)
print("Wrote transcript.json")
