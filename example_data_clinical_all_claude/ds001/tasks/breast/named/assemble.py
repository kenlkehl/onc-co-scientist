"""Build transcript.json and analysis_summary.txt from results.json."""
import json

with open("results.json") as f:
    raw = json.load(f)

iters = raw["iterations"]

# Strip helper keys not in schema
for it in iters:
    pass
extra = {k: v for k, v in raw.items() if k.startswith("_")}

transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-custom@local",
    "max_iterations": 25,
    "iterations": iters,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

print(f"Wrote transcript.json with {len(iters)} iterations.")
total_h = sum(len(i["proposed_hypotheses"]) for i in iters)
total_a = sum(len(i["analyses"]) for i in iters)
print(f"Total hypotheses: {total_h}, total analyses: {total_a}")
