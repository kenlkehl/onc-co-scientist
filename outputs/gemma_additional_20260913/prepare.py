"""Freeze an additional NSCLC campaign using the prior Gemma implementation."""
import argparse
from collections import Counter
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import yaml

repo = Path('/data1/ken/onc-co-scientist')
base = repo / 'data/expected_surprising_ledger/full_runs'
p = argparse.ArgumentParser()
p.add_argument('--per-condition', action='store_true')
args = p.parse_args()
root = base / '20260913_additional_gemma_8060'
sources = {'named': base / '20260910_vllm_recommended',
           'masked': base / '20260911_masked_gemma_8060'}
code_source = sources['masked']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for source in sources.values():
    frozen = json.loads((source / 'frozen_manifest.json').read_text())
    for name, expected in {**frozen['input_hashes'], 'config.yaml': frozen['config_sha256']}.items():
        assert sha(source / name) == expected, (source, name)
frozen = json.loads((code_source / 'frozen_manifest.json').read_text())
for name, expected in frozen['source_hashes'].items():
    assert sha(code_source / name) == expected, name
root.mkdir(exist_ok=False)
shutil.copytree(code_source / 'source', root / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copy2(Path(__file__).with_name('run.py'), root / 'source/run_additional.py')
shutil.copy2(__file__, root / 'source/prepare_additional.py')
sys.path.insert(0, str(root / 'source/src'))
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans
configs = {}
for label, source in sources.items():
    out = root / label
    out.mkdir()
    shutil.copytree(source / 'input_data', out / 'input_data')
    raw = yaml.safe_load((code_source / 'config.yaml').read_text())
    raw.update(experiment_id=f'{root.name}_{label}', output_root=str(out),
               description='Additional independent Gemma NSCLC runs; same frozen scientific protocol; no federation.',
               replicates=175, max_parallel=25, schedule_seed=20260913)
    raw['expected_surprising']['root'] = str(out / 'input_data')
    assert 'federation' not in raw
    (out / 'config.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    configs[label] = load_experiment_spec(out / 'config.yaml')
    assert all(not w.federated for w in configs[label].workflows)
plans = []
workflows = ['persistent', 'sequential', 'deliberative']
conditions = [('named', 'expected'), ('named', 'surprising'), ('masked', 'expected'), ('masked', 'surprising')]
for label, spec in configs.items():
    for plan in build_run_plans(spec):
        c = conditions.index((label, plan.task.semantic_condition))
        w = workflows.index(plan.workflow.id)
        count = 75 if args.per_condition else 18 + int((c-w) % 4 < 3)
        if 101 <= plan.replicate <= 100 + count:
            plans.append({'grid': label, **plan.public_dict()})
rng = random.Random(20260913)
rng.shuffle(plans)
plans.sort(key=lambda r: r['replicate'])
counts = Counter((r['grid'], r['semantic_condition'], r['workflow_id']) for r in plans)
assert len(plans) == (900 if args.per_condition else 225)
assert set(Counter(r['workflow_id'] for r in plans).values()) == {300 if args.per_condition else 75}
(root / 'plan.json').write_text(json.dumps(plans, indent=2))
manifest = dict(created_at=datetime.now(UTC).isoformat(), source_grids={k:str(v) for k,v in sources.items()},
    code_source=str(code_source), dataset_pair='es-v2-nsclc_clinical-42000',
    assigned_runs=len(plans), workers=25, iterations=25, federation=False,
    count_policy='75 per workflow per condition' if args.per_condition else '75 total per workflow, balanced across four conditions',
    replicate_policy='Fresh replicate IDs starting at 101; independent stochastic model requests',
    counts=[dict(grid=k[0],version=k[1],workflow=k[2],runs=v) for k,v in sorted(counts.items())],
    hashes={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
(root / 'frozen_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k != 'hashes'}, indent=2))
