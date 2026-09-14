import json,os,signal,subprocess,time
from pathlib import Path
from datetime import datetime,UTC
repo=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
root=repo/'data/expected_surprising_ledger/full_runs/20260910_masked_vllm_recommended'
assert not (out/'launch.json').exists()
old=json.loads((root/'control/launch.json').read_text())['pid']
assert str(root/'source/run_selected_cells.py').encode() in Path(f'/proc/{old}/cmdline').read_bytes().split(b'\0')
ids=json.loads((root/'selection.json').read_text());assert len(ids)==120
# Freeze the scheduler, then recursively stop its descendants before replacement.
os.kill(old,signal.SIGSTOP)
children={}
for row in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines():
 pid,ppid=map(int,row.split());children.setdefault(ppid,[]).append(pid)
desc=[]
def visit(pid):
 for child in children.get(pid,[]):
  try:os.kill(child,signal.SIGSTOP)
  except ProcessLookupError:pass
  visit(child);desc.append(child)
visit(old)
selections={model:[rid for rid in ids if '__'+model+'__' in rid and not (root/'runs'/rid/'run.json').exists()] for model in ('gemma_4_31b','qwen_3_8_27b')}
assert not set(selections['gemma_4_31b'])&set(selections['qwen_3_8_27b'])
record={'at':datetime.now(UTC).isoformat(),'retired_shared_pid':old,'retired_descendants':desc,'policy':'Disjoint model queues; six Gemma workers plus three Qwen workers. Frozen code and saved calls unchanged. Astra deliberative remains paused.','selections':selections}
(out/'handoff.json').write_text(json.dumps(record,indent=2))
for pid in desc+[old]:
 try:os.kill(pid,signal.SIGKILL)
 except ProcessLookupError:pass
for _ in range(100):
 alive=[]
 for pid in desc+[old]:
  p=Path(f'/proc/{pid}/stat')
  if p.exists() and p.read_text().split()[2]!='Z':alive.append(pid)
 if not alive:break
 time.sleep(.1)
else:raise RuntimeError(f'Old process tree still alive: {alive}; no new runners launched')
env=dict(os.environ,PYTHONPATH=str(root/'source/src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
record['drivers']={}
for model,workers in [('gemma_4_31b',6),('qwen_3_8_27b',3)]:
 control=root/(model+'_control');control.mkdir(exist_ok=False);selection=control/'selection.json';selection.write_text(json.dumps(selections[model],indent=2))
 cmd=['/tmp/ocs-es-refactor-venv/bin/python',str(root/'source/run_selected_cells.py'),'--root',str(root),'--selection',str(selection),'--control',str(control),'--workers',str(workers),'--resume']
 with (control/'driver.log').open('ab') as log:p=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 entry={'pid':p.pid,'workers':workers,'remaining_runs':len(selections[model]),'command':cmd};record['drivers'][model]=entry;(control/'launch.json').write_text(json.dumps(entry,indent=2))
(out/'launch.json').write_text(json.dumps(record,indent=2));(root/'split_workers.json').write_text(json.dumps(record,indent=2));print(json.dumps(record['drivers']),flush=True)
