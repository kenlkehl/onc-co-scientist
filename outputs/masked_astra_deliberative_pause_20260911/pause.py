import json,os,signal,subprocess,time
from pathlib import Path
from datetime import datetime,UTC
repo=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
root=repo/'data/expected_surprising_ledger/full_runs/20260910_clinical10pct_masked_workflows'
control=root/'codex_without_astra_deliberative';assert not (control/'launch.json').exists()
state=json.loads((root/'execution.json').read_text());pid=state['pid']
assert str(root/'source/run_masked_grid.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
plans=json.loads((root/'plan.json').read_text());codex=[p['run_id'] for p in plans if p['model_profile'] in {'astra_medium','terra_medium','sol_medium','luna_medium'}]
held=[rid for rid in codex if '__deliberative__astra_medium__' in rid];selected=[rid for rid in codex if rid not in held]
assert len(held)==20 and len(selected)==220
control.mkdir(exist_ok=True);(control/'selection.json').write_text(json.dumps(selected,indent=2))
(out/'held_selection.json').write_text(json.dumps(held,indent=2))
# Stop scheduler first, then freeze and retire only its descendant processes.
os.kill(pid,signal.SIGSTOP)
rows=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True)
children={}
for row in rows.splitlines():
 a,b=map(int,row.split());children.setdefault(b,[]).append(a)
subtree=[]
def visit(p):
 for c in children.get(p,[]):visit(c);subtree.append(c)
visit(pid)
for p in subtree:
 try:os.kill(p,signal.SIGSTOP)
 except ProcessLookupError:pass
record={'paused_at':datetime.now(UTC).isoformat(),'policy':'Astra deliberative stays paused until explicit user authorization, even after other runs complete.','held_ids':held,'held_unfinished_ids':[rid for rid in held if not (root/'runs'/rid/'run.json').exists()],'retired_driver_pid':pid,'retired_descendants':subtree,'continued_selection':str(control/'selection.json'),'codex_frozen_source_unchanged':True}
(root/'astra_deliberative_pause.json').write_text(json.dumps(record,indent=2))
for p in subtree+[pid]:
 try:os.kill(p,signal.SIGKILL)
 except ProcessLookupError:pass
for _ in range(100):
 remaining=[]
 for p in subtree+[pid]:
  f=Path(f'/proc/{p}/stat')
  if f.exists() and f.read_text().split()[2]!='Z':remaining.append(p)
 if not remaining:break
 time.sleep(.1)
else:raise RuntimeError(f'Old workers still alive: {remaining}')
env=dict(os.environ,PYTHONPATH=str(root/'source/src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['/tmp/ocs-es-refactor-venv/bin/python',str(out/'run_selected_cells.py'),'--root',str(root),'--selection',str(control/'selection.json'),'--control',str(control),'--workers','30','--resume']
with (control/'driver.log').open('ab') as log:p=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
record.update(new_driver_pid=p.pid,command=cmd)
(control/'launch.json').write_text(json.dumps(record,indent=2));(root/'astra_deliberative_pause.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'driver_pid':p.pid,'held_unfinished':len(record['held_unfinished_ids']),'held_finished':20-len(record['held_unfinished_ids'])}),flush=True)
time.sleep(3)
assert p.poll() is None,(control/'driver.log').read_text()[-4000:]
