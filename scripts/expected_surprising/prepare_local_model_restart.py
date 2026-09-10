"""Freeze fresh local-model cells from either a masked or unmasked clinical grid."""
import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-grid', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source_grid.resolve(), args.out.resolve()
    repo = Path(__file__).resolve().parents[2]
    frozen = json.loads((source / 'frozen_manifest.json').read_text())
    for name, digest in {**frozen['input_hashes'], 'config.yaml': frozen['config_sha256']}.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    raw = yaml.safe_load((source / 'config.yaml').read_text())
    models = [m for m in raw['models'] if (m.get('provider_config') or {}).get('kind') == 'vllm_openai']
    if not models:
        raise ValueError('No local vLLM models in source grid')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(raw['expected_surprising']['root']), out / 'input_data')
    shutil.copytree(repo / 'src', out / 'source/src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(repo / 'scripts/expected_surprising/run_selected_cells.py', out / 'source/run_selected_cells.py')
    raw.update(experiment_id=out.name, output_root=str(out), models=models, max_parallel=24)
    raw['expected_surprising'].update(root=str(out / 'input_data'), peer_failure_policy='chair_with_available', stage_failure_policy='retain_scientific_scores')
    for m in models:
        m['provider_config'].update(sampling_profile='auto', json_object_output=True, disable_thinking_on_final_retry=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    names = {m['id'] for m in models}
    plans = [p for p in json.loads((source / 'plan.json').read_text()) if p['model_profile'] in names]
    ids = [p['run_id'] for p in plans]
    assert len(ids) == len(set(ids))
    (out / 'selection.json').write_text(json.dumps(ids, indent=2))
    (out / 'plan.json').write_text(json.dumps(plans, indent=2))
    def hashes(folder):
        return {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*')) if p.is_file()}
    manifest = dict(frozen_at=datetime.now(UTC).isoformat(), original_grid=str(source), source_hashes=hashes(out / 'source'), input_hashes=hashes(out / 'input_data'), config_sha256=hashlib.sha256((out / 'config.yaml').read_bytes()).hexdigest(), selection_sha256=hashlib.sha256((out / 'selection.json').read_bytes()).hexdigest(), selected_runs=len(ids), policy='Recommended sampling; explicit thinking; final fallback only after two consecutive truncations')
    (out / 'frozen_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(out, len(ids))


if __name__ == '__main__':
    main()
