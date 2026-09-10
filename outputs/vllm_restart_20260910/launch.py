import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
repo=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
root=repo/'data/expected_surprising_ledger/full_runs/20260910_vllm_repair'
checks={r['model']:r for name in ['live_smoke','live_smoke_retries'] for r in json.loads((out/name/'results.json').read_text()) if r['passed']}
assert set(checks)=={'qwen_3_8_27b','gemma_4_31b'}
(out/'accepted_smoke_checks.json').write_text(json.dumps(checks,indent=2))
assert json.loads((out/'thinking_fallback_smoke.json').read_text())['passed']
assert '49 passed' in (out/'tests.log').read_text()
control=root/'control';control.mkdir(exist_ok=False)
env=dict(os.environ,PYTHONPATH=str(root/'source/src'),PYTHONNOUSERSITE='1')
cmd=[sys.executable,str(root/'source/run_selected_cells.py'),'--root',str(root),'--selection',str(root/'selection.json'),'--control',str(control),'--workers','24']
with (control/'driver.log').open('a') as log:
 p=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(control/'launch.json').write_text(json.dumps({'pid':p.pid,'command':cmd,'started_at':datetime.now(timezone.utc).isoformat()},indent=2))
monitor=repo/'outputs/clinical_workflow_restart_20260909'
with (monitor/'monitor.log').open('a') as log:
 q=subprocess.Popen([sys.executable,str(monitor/'living_report.py')],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(monitor/'monitor_launch.json').write_text(json.dumps({'pid':q.pid,'started_at':datetime.now(timezone.utc).isoformat()},indent=2))
print('Driver',p.pid,'report monitor',q.pid,flush=True)
