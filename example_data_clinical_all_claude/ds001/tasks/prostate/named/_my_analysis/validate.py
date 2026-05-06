import json
from pathlib import Path

ROOT = Path(r"C:\Users\klkehl\are_llms_biased\data\ds001\tasks\prostate\named")
schema = json.load(open(ROOT / "transcript_schema.json"))
data = json.load(open(ROOT / "transcript.json"))

# Required top-level
for f in schema["required"]:
    assert f in data, f"missing {f}"
assert isinstance(data["dataset_id"], str)
assert isinstance(data["model_id"], str)
assert isinstance(data["harness_id"], str)
assert isinstance(data["max_iterations"], int) and data["max_iterations"] >= 1
assert isinstance(data["iterations"], list)

# Iter validation
for it in data["iterations"]:
    assert "index" in it and isinstance(it["index"], int) and it["index"] >= 1
    assert "proposed_hypotheses" in it and isinstance(it["proposed_hypotheses"], list)
    for h in it["proposed_hypotheses"]:
        assert "id" in h and isinstance(h["id"], str)
        assert "text" in h and isinstance(h["text"], str)
        if "kind" in h:
            assert h["kind"] in ("novel", "refined")
    if "analyses" in it:
        for a in it["analyses"]:
            assert "hypothesis_ids" in a and isinstance(a["hypothesis_ids"], list)
            assert "result_summary" in a and isinstance(a["result_summary"], str)
            if "p_value" in a and a["p_value"] is not None:
                assert isinstance(a["p_value"], (int, float))
            if "effect_estimate" in a and a["effect_estimate"] is not None:
                assert isinstance(a["effect_estimate"], (int, float))
            if "significant" in a and a["significant"] is not None:
                assert isinstance(a["significant"], bool)

print("Schema validation passed.")
print(f"Iterations: {len(data['iterations'])}")
print(f"Top-level: dataset_id={data['dataset_id']}, model_id={data['model_id']}, harness_id={data['harness_id']}, max_iterations={data['max_iterations']}")
total_h = sum(len(it['proposed_hypotheses']) for it in data['iterations'])
total_a = sum(len(it.get('analyses', [])) for it in data['iterations'])
print(f"Total hypotheses: {total_h}, analyses: {total_a}")

# Spot-check analysis_summary.txt
text = open(ROOT / "analysis_summary.txt", encoding="utf-8").read()
print(f"\nanalysis_summary.txt: {len(text)} chars, {text.count(chr(10))} lines")
print("First 600 chars:")
print(text[:600])
