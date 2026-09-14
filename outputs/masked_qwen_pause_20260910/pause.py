import os,json,signal,subprocess,time
from pathlib import Path
from datetime import datetime,UTC
root=Path('/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260910_masked_vllm_repair')
control=root/'gemma_only_control'
assert not (control/'launch.json').exists(), 'Already launched; inspect first'
state=json.loads((root/'control/execution.json').read_text())
pid=state['pid']
args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert str(root/'source/run_selected_cells.py').encode() in args
ids=json.loads((root/'selection.json').read_text())
gemma=[s for s in ids if '__gemma_4_31b__' in s]
qwen=[s for s in ids if '__qwen_3_8_27b__' in s]
assert len(gemma)==len(qwen)==60
control.mkdir(exist_ok=True)
selection=control/'selection.json'
selection.write_text(json.dumps(gemma,indent=2))
record={'paused_at':datetime.now(UTC).isoformat(),'model':'qwen_3_8_27b','reason':'User requested pause for parser fixes in another task','old_shared_driver_pid':pid,'qwen_ids':qwen,'gemma_control':str(control),'note':'Original supervisor child exit is intentional. Gemma resumes cached calls in its separate runner; Codex unchanged.'}
(root/'qwen_pause.json').write_text(json.dumps(record,indent=2))
os.kill(pid,signal.SIGTERM)
for _ in range(100):
 p=Path(f'/proc/{pid}/stat')
 if not p.exists() or p.read_text().split()[2]=='Z':break
 time.sleep(.1)
else:raise RuntimeError('Shared driver did not exit; no replacement launched')
env=dict(os.environ,PYTHONPATH=str(root/'source/src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['/tmp/ocs-es-refactor-venv/bin/python',str(root/'source/run_selected_cells.py'),'--root',str(root),'--selection',str(selection),'--control',str(control),'--workers','5','--resume']
with (control/'driver.log').open('ab') as log:
 child=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record.update(gemma_driver_pid=child.pid,command=cmd)
(control/'launch.json').write_text(json.dumps(record,indent=2))
(root/'qwen_pause.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'stopped_shared_pid':pid,'gemma_pid':child.pid,'qwen_paused':60,'gemma_selection':60}))
time.sleep(2)
assert child.poll() is None,(control/'driver.log').read_text()[-4000:]
