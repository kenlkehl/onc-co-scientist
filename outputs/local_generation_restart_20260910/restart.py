"""User-authorized erasure of 120 old unmasked local-model cells, then fresh launch."""
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

repo=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
old=repo/'data/expected_surprising_ledger/full_runs/20260910_vllm_repair'
new=repo/'data/expected_surprising_ledger/full_runs/20260910_vllm_recommended'
assert '54 passed' in (out/'tests_final.log').read_text()
checks=json.loads((out/'live_results.json').read_text());assert len(checks)==4 and all(r['passed'] for r in checks)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo)==subprocess.check_output(['git','rev-parse','origin/expected-or-surprising'],cwd=repo)
subprocess.run([sys.executable,str(repo/'scripts/expected_surprising/prepare_local_model_restart.py'),'--source-grid',str(old),'--out',str(new)],check=True,cwd=repo,env=dict(os.environ,PYTHONPATH=str(repo/'src')))
ids=json.loads((new/'selection.json').read_text());assert len(ids)==120
snapshot=[]
for rid in ids:
 assert '__qwen_3_8_27b__' in rid or '__gemma_4_31b__' in rid
 folder=old/'runs'/rid;assert not folder.is_symlink()
 result=json.loads((folder/'run.json').read_text()) if (folder/'run.json').exists() else {}
 tokens=0;missing=0
 for call in (folder/'calls').glob('*.json'):
  u=json.loads(call.read_text())['result'].get('usage',{}).get('output_tokens');tokens+=u or 0;missing+=u is None
 snapshot.append({'run_id':rid,'status':result.get('status','paused_or_queued'),'observed_output_tokens':tokens,'missing_usage_calls':missing})
(out/'erased_runs.json').write_text(json.dumps(snapshot,indent=2))
for rid in ids:
 folder=old/'runs'/rid
 if folder.exists():shutil.rmtree(folder)
assert not any((old/'runs'/rid).exists() for rid in ids)
control=new/'control';control.mkdir()
env=dict(os.environ,PYTHONPATH=str(new/'source/src'),PYTHONNOUSERSITE='1')
cmd=[sys.executable,str(new/'source/run_selected_cells.py'),'--root',str(new),'--selection',str(new/'selection.json'),'--control',str(control),'--workers','24']
with (control/'driver.log').open('a') as log:p=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(control/'launch.json').write_text(json.dumps({'pid':p.pid,'command':cmd,'at':datetime.now(UTC).isoformat(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()},indent=2))
monitor=repo/'outputs/clinical_workflow_restart_20260909'
with (monitor/'monitor.log').open('a') as log:q=subprocess.Popen([sys.executable,str(monitor/'living_report.py')],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(monitor/'monitor_launch.json').write_text(json.dumps({'pid':q.pid}))
print('New driver',p.pid,'local report updater',q.pid,flush=True)
