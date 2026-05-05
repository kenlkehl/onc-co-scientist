"""Assemble transcript.json (schema-conformant) and analysis_summary.txt from results.json."""
import json
from pathlib import Path

with open("results.json") as f:
    r = json.load(f)

transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "claude-opus-4-7[1m]",
    "harness_id": "claude-code-cli@manual",
    "max_iterations": 25,
    "iterations": r["iterations"],
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)
print("transcript.json written; iterations:", len(transcript["iterations"]))
