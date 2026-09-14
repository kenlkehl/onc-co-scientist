import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
repo=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
root=repo/'data/expected_surprising_ledger/full_runs/20260910_vllm_repair'
ids=json.loads((root/'selection.json').read_text())
selected=[rid for rid in ids if '__gemma_4_31b__' in rid and not (root/'runs'/rid/'run.json').exists()]
(out/'gemma_selection.json').write_text(json.dumps(selected,indent=2))
(out/'pause.json').write_text(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'stopped_driver':2015538,'qwen_paused':True,'gemma_resuming':selected},indent=2))
control=out/'gemma_control';control.mkdir(exist_ok=False)
env=dict(os.environ,PYTHONPATH=str(root/'source/src'),PYTHONNOUSERSITE='1')
cmd=[sys.executable,str(root/'source/run_selected_cells.py'),'--root',str(root),'--selection',str(out/'gemma_selection.json'),'--control',str(control),'--workers','12','--resume']
with (control/'driver.log').open('a') as log:p=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(control/'launch.json').write_text(json.dumps({'pid':p.pid,'command':cmd},indent=2));print('Gemma driver',p.pid,'unfinished cells',len(selected))
