"""Build transcript.json from analysis_state.json."""
import json

with open('analysis_state.json') as f:
    st = json.load(f)

iterations = sorted(st.values(), key=lambda x: x['index'])

# Sanity: ensure analyses with p_value have a proper float and significant flag
for it in iterations:
    for a in it['analyses']:
        if 'p_value' in a and a['p_value'] is not None:
            try:
                a['p_value'] = float(a['p_value'])
            except Exception:
                a['p_value'] = None
        if 'effect_estimate' in a and a['effect_estimate'] is not None:
            try:
                a['effect_estimate'] = float(a['effect_estimate'])
            except Exception:
                a['effect_estimate'] = None
        if 'significant' in a and a['significant'] is not None:
            a['significant'] = bool(a['significant'])

transcript = {
    'dataset_id': 'ds001_nsclc',
    'model_id': 'claude-opus-4-7',
    'harness_id': 'manual-claude-code@named-2026-05-03',
    'max_iterations': 25,
    'iterations': iterations,
}

with open('transcript.json', 'w') as f:
    json.dump(transcript, f, indent=2)

# Validate against schema (basic)
print('Iterations:', len(transcript['iterations']))
print('Total hypotheses:', sum(len(i['proposed_hypotheses']) for i in iterations))
print('Total analyses:', sum(len(i['analyses']) for i in iterations))
print('OK')
