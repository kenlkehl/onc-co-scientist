from pathlib import Path
import shutil,json,hashlib,datetime,yaml
repo=Path(__file__).resolve().parents[2]
old=repo/'data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows'
new=repo/'data/expected_surprising_ledger/full_runs/20260909_codex_repair'
new.mkdir(exist_ok=False)
shutil.copytree(old/'input_data',new/'input_data')
shutil.copytree(repo/'src',new/'source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(repo/'scripts/expected_surprising/run_selected_cells.py',new/'source/run_selected_cells.py')
raw=yaml.safe_load((old/'config.yaml').read_text())
raw['experiment_id']='clinical_10pct_selective_codex_repair_20260909'
raw['description']='104 explicit replacement/unfinished Codex cells; use selection.json, never execute the entire implied 240-cell matrix.'
raw['output_root']=str(new)
raw['expected_surprising'].update(root=str(new/'input_data'),peer_failure_policy='chair_with_available',stage_failure_policy='retain_scientific_scores')
raw['models']=[m for m in raw['models'] if m['provider_config']['kind']=='codex_cli']
raw['max_parallel']=16
(new/'config.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
ids=json.loads((repo/'outputs/clinical_workflow_restart_20260909/codex_selection.json').read_text())
(new/'selection.json').write_text(json.dumps(ids,indent=2))
oldplans=json.loads((old/'plan.json').read_text())
(new/'plan.json').write_text(json.dumps([p for p in oldplans if p['run_id'] in ids],indent=2))
def hashes(folder):return {str(p.relative_to(new)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*')) if p.is_file()}
manifest={'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_grid':str(old),'source_hashes':hashes(new/'source'),'input_hashes':hashes(new/'input_data'),'config_sha256':hashlib.sha256((new/'config.yaml').read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256((new/'selection.json').read_bytes()).hexdigest(),'selected_runs':len(ids),'requested_reasoning':'medium','requested_service_tier':'default','retain_original_clean_terminal_runs':True}
(new/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2))
print(new)
