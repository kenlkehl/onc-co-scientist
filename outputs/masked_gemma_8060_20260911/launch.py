import json,os,signal,subprocess,time,shutil,hashlib
from pathlib import Path
from datetime import datetime,UTC
import yaml
repo=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent;base=repo/'data/expected_surprising_ledger/full_runs';old=base/'20260910_masked_vllm_recommended';new=base/'20260911_masked_gemma_8060';assert not (new/'frozen_manifest.json').exists() and not (out/'handoff.json').exists()
# Prepare identical frozen implementation and masked inputs before changing scheduling.
new.mkdir(exist_ok=True);shutil.copytree(old/'source',new/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True);shutil.copytree(old/'input_data',new/'input_data',dirs_exist_ok=True)
cfg=yaml.safe_load((old/'config.yaml').read_text());cfg.update(experiment_id=new.name,output_root=str(new),max_parallel=12);cfg['expected_surprising']['root']=str(new/'input_data');cfg['models']=[m for m in cfg['models'] if m['id']=='gemma_4_31b'];cfg['models'][0]['model_id']='gemma4-31b';cfg['models'][0]['provider_config'].update(base_url='http://camus:8060/v1',model_id='gemma4-31b');(new/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans
plans={p.run_id for p in build_run_plans(load_experiment_spec(new/'config.yaml'))}
control=old/'gemma_4_31b_control';pid=json.loads((control/'launch.json').read_text())['pid'];assert str(old/'source/run_selected_cells.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
os.kill(pid,signal.SIGSTOP)
children={}
for row in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines():
 p,pp=map(int,row.split());children.setdefault(pp,[]).append(p)
desc=[]
def stop_tree(p):
 for c in children.get(p,[]):
  try:os.kill(c,signal.SIGSTOP)
  except ProcessLookupError:pass
  stop_tree(c);desc.append(c)
stop_tree(pid)
ids=json.loads((control/'selection.json').read_text());queued=[x for x in ids if not (old/'runs'/x).exists()];active=[x for x in ids if (old/'runs'/x).exists() and not (old/'runs'/x/'run.json').exists()];assert set(queued)<=plans and not set(queued)&set(active) and len(queued)>=12
(new/'selection.json').write_text(json.dumps(queued,indent=2));rows=json.loads((old/'plan.json').read_text());(new/'plan.json').write_text(json.dumps([p for p in rows if p['run_id'] in queued],indent=2))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes(folder):return {str(p.relative_to(new)):digest(p) for p in folder.rglob('*') if p.is_file()}
frozen={'source_hashes':hashes(new/'source'),'input_hashes':hashes(new/'input_data'),'config_sha256':digest(new/'config.yaml'),'selection_sha256':digest(new/'selection.json'),'frozen_at':datetime.now(UTC).isoformat(),'original_grid':str(old)};(new/'frozen_manifest.json').write_text(json.dumps(frozen,indent=2))
record={'at':datetime.now(UTC).isoformat(),'old_pid':pid,'old_descendants':desc,'moved_queued_ids':queued,'retained_active_ids':active,'new_root':str(new),'endpoint':'http://camus:8060/v1','served_model':'gemma4-31b','underlying_model':'RedHatAI/Gemma-4-31B-IT-FP8-Dynamic'};(out/'handoff.json').write_text(json.dumps(record,indent=2))
for p in desc+[pid]:
 try:os.kill(p,signal.SIGKILL)
 except ProcessLookupError:pass
for _ in range(100):
 alive=[p for p in desc+[pid] if Path(f'/proc/{p}/stat').exists() and Path(f'/proc/{p}/stat').read_text().split()[2]!='Z']
 if not alive:break
 time.sleep(.1)
else:raise RuntimeError('Old process tree remains alive; no replacements started')
record['drivers']=[]
for root,selection,workers,name,resume in [(old,active,6,'gemma_existing_only_control',True),(new,queued,12,'control',False)]:
 c=root/name;c.mkdir();(c/'selection.json').write_text(json.dumps(selection,indent=2));env=dict(os.environ,PYTHONPATH=str(root/'source/src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');cmd=['/tmp/ocs-es-refactor-venv/bin/python',str(root/'source/run_selected_cells.py'),'--root',str(root),'--selection',str(c/'selection.json'),'--control',str(c),'--workers',str(workers)]+(['--resume'] if resume else [])
 with (c/'driver.log').open('ab') as log:p=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 entry={'pid':p.pid,'workers':workers,'run_n':len(selection),'command':cmd};record['drivers'].append(entry);(c/'launch.json').write_text(json.dumps(entry,indent=2))
(old/'routing_overrides.json').write_text(json.dumps({rid:str(new) for rid in queued},indent=2));(out/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record['drivers']),flush=True)
