from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,UTC
import yaml
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans
repo=Path(__file__).resolve().parents[2]
base=repo/'data/expected_surprising_ledger/full_runs'
parent=base/'20260910_clinical10pct_masked_workflows'
repaired=base/'20260910_vllm_repair'
root=base/'20260910_masked_vllm_repair'
assert not root.exists()
for folder in (parent,repaired):
 f=json.loads((folder/'frozen_manifest.json').read_text())
 for p,h in {**f['source_hashes'],**f['input_hashes'],'config.yaml':f['config_sha256']}.items():
  assert hashlib.sha256((folder/p).read_bytes()).hexdigest()==h,p
assert not json.loads((parent/'execution.json').read_text())['vllm_released']
root.mkdir()
shutil.copytree(parent/'input_data',root/'input_data')
shutil.copytree(repaired/'source',root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copyfile(Path(__file__).with_name('supervise.py'),root/'source/supervise_masked.py')
config=yaml.safe_load((repaired/'config.yaml').read_text())
config.update(experiment_id='clinical_10pct_masked_vllm_repair_20260910',description='120 masked Qwen/Gemma jobs with exactly the repaired unmasked implementation.',output_root=str(root),max_parallel=6)
config['expected_surprising']['root']=str(root/'input_data')
(root/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
rows=[p for p in json.loads((parent/'plan.json').read_text()) if p['model_profile'] in {'qwen_3_8_27b','gemma_4_31b'}]
ids=[p['run_id'] for p in rows]
assert len(ids)==len(set(ids))==120
assert set(ids)=={p.run_id for p in build_run_plans(load_experiment_spec(root/'config.yaml'))}
(root/'plan.json').write_text(json.dumps(rows,indent=2))
(root/'selection.json').write_text(json.dumps(ids,indent=2))
handoff=dict(created_at=datetime.now(UTC).isoformat(),codex_root=str(parent),repaired_unmasked_root=str(repaired),workers=6,concurrency_authorization='User explicitly requested masked jobs alongside unmasked jobs.',retired_gate=str(repo/'outputs/clinical_workflow_restart_20260909/vllm_continuation/execution.json'),policy='Leave retired gate unfinished. The original driver owns only 240 Codex identities; this runner owns the 120 never-started masked Qwen/Gemma identities. Retire the old scheduler only when Codex has no active or queued jobs.')
(root/'handoff.json').write_text(json.dumps(handoff,indent=2))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes(folder):return {str(p.relative_to(root)):digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}
frozen=dict(frozen_at=datetime.now(UTC).isoformat(),source_hashes=hashes(root/'source'),input_hashes=hashes(root/'input_data'),config_sha256=digest(root/'config.yaml'),selection_sha256=digest(root/'selection.json'),handoff_sha256=digest(root/'handoff.json'),selected_runs=120,repair_source=str(repaired))
(root/'frozen_manifest.json').write_text(json.dumps(frozen,indent=2))
(parent/'vllm_handoff.json').write_text(json.dumps({**handoff,'corrected_masked_root':str(root)},indent=2))
print(root)
