"""Build transcript.json from all_my_results.json."""
import json
from collections import defaultdict, OrderedDict

results = json.load(open('all_my_results.json'))

iters = defaultdict(lambda: {'hypotheses': OrderedDict(), 'analyses': []})

for r in results:
    it = r['iter']
    if r['hid'] not in iters[it]['hypotheses']:
        iters[it]['hypotheses'][r['hid']] = {
            'id': r['hid'],
            'text': r['text'],
            'kind': r['kind'],
        }
    iters[it]['analyses'].append({
        'hypothesis_ids': [r['hid']],
        'code': r['code'],
        'result_summary': r['result_summary'],
        'p_value': r['p_value'],
        'effect_estimate': r['effect_estimate'],
        'significant': r['significant'],
    })

iterations = []
for idx in sorted(iters.keys()):
    iterations.append({
        'index': idx,
        'proposed_hypotheses': list(iters[idx]['hypotheses'].values()),
        'analyses': iters[idx]['analyses'],
    })

transcript = {
    'dataset_id': 'ds001_crc',
    'model_id': 'claude-opus-4-7',
    'harness_id': 'claude-code-manual@1.0',
    'max_iterations': 25,
    'iterations': iterations,
}

with open('transcript.json','w') as f:
    json.dump(transcript, f, indent=2)
print(f"Wrote transcript.json with {len(iterations)} iterations and {sum(len(it['analyses']) for it in iterations)} analyses.")
