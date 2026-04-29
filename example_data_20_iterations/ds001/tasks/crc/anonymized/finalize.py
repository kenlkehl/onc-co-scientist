"""Compose final transcript.json from iter_partial.json with required top-level fields."""
import json

with open('iter_partial.json') as f:
    body = json.load(f)

transcript = {
    'dataset_id': 'ds001_crc',
    'model_id': 'claude-opus-4-7',
    'harness_id': 'claude-code-manual@2026-04-28',
    'max_iterations': 25,
    'iterations': body['iterations'],
}
with open('transcript.json','w') as f:
    json.dump(transcript, f, indent=2)
print('Wrote transcript.json with', len(transcript['iterations']), 'iterations.')
