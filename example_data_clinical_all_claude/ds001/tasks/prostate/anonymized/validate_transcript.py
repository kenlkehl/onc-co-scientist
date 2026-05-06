"""Validate transcript.json against schema requirements and ID cross-refs."""
import json

d = json.load(open('transcript.json'))
# Check required top-level fields
required_top = ['dataset_id','model_id','harness_id','max_iterations','iterations']
for f in required_top:
    assert f in d, f'missing top: {f}'
for it in d['iterations']:
    assert 'index' in it and 'proposed_hypotheses' in it
    for h in it['proposed_hypotheses']:
        assert 'id' in h and 'text' in h
    for a in it.get('analyses', []):
        assert 'hypothesis_ids' in a and 'result_summary' in a
        for hid in a['hypothesis_ids']:
            assert isinstance(hid, str)
# Cross-iteration: every analysis's hypothesis_ids must exist in the same iteration's hypotheses
warns = 0
for it in d['iterations']:
    hids = {h['id'] for h in it['proposed_hypotheses']}
    for a in it.get('analyses', []):
        for hid in a['hypothesis_ids']:
            if hid not in hids:
                print(f"WARN iteration {it['index']}: analysis cites missing hypothesis id {hid}")
                warns += 1
print('Schema check passed.' if warns==0 else f'{warns} warnings')
print(f'Top-level fields: {list(d.keys())}')
print(f'max_iterations: {d["max_iterations"]}')
print(f'dataset_id: {d["dataset_id"]}')
print(f'model_id: {d["model_id"]}')
print(f'harness_id: {d["harness_id"]}')
print(f'iterations: {len(d["iterations"])}')
print(f'total hypotheses: {sum(len(it["proposed_hypotheses"]) for it in d["iterations"])}')
print(f'total analyses: {sum(len(it["analyses"]) for it in d["iterations"])}')
