"""Build transcript.json and analysis_summary.txt from accumulated iterations."""
import json
import pickle

with open("_iters_1_25.pkl", "rb") as f:
    iterations = pickle.load(f)

# Sanity: verify each iteration record is well-formed
for it in iterations:
    assert "index" in it and "proposed_hypotheses" in it and "analyses" in it
    for h in it["proposed_hypotheses"]:
        assert "id" in h and "text" in h
    for a in it["analyses"]:
        assert "hypothesis_ids" in a and "result_summary" in a

transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-custom@2026-05-03",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

print(f"Wrote transcript.json with {len(iterations)} iterations")

# Count totals
n_h = sum(len(it["proposed_hypotheses"]) for it in iterations)
n_a = sum(len(it["analyses"]) for it in iterations)
print(f"Total hypotheses: {n_h}, total analyses: {n_a}")
