import hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
from datetime import datetime,UTC
repo=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
base=repo/'data/expected_surprising_ledger/full_runs';old=base/'20260910_masked_vllm_repair';new=base/'20260910_masked_vllm_recommended';named=base/'20260910_vllm_recommended'
assert '6 passed' in (out/'tests.log').read_text()
assert not (new/'control').exists()
a=json.loads((new/'frozen_manifest.json').read_text());b=json.loads((named/'frozen_manifest.json').read_text())
assert a['source_hashes']==b['source_hashes'],'Must match running unmasked implementation'
assert a['input_hashes']==json.loads((old/'frozen_manifest.json').read_text())['input_hashes']
ids=json.loads((new/'selection.json').read_text());assert len(ids)==120
assert set(ids)==set(json.loads((old/'selection.json').read_text()))
for rid in ids:assert '__qwen_3_8_27b__' in rid or '__gemma_4_31b__' in rid
# Stop only the obsolete masked monitor and the suspended Gemma-only process.
for pid,expected in [(json.loads((old/'supervisor_launch.json').read_text())['pid'],old/'source/supervise_masked.py'),(json.loads((old/'gemma_only_control/launch.json').read_text())['gemma_driver_pid'],old/'source/run_selected_cells.py')]:
 p=Path(f'/proc/{pid}/cmdline')
 if p.exists():
  assert str(expected).encode() in p.read_bytes().split(b'\0')
  os.kill(pid,signal.SIGKILL) # suspended worker must not issue another request
  for _ in range(100):
   p=Path(f'/proc/{pid}/stat')
   if not p.exists() or p.read_text().split()[2]=='Z':break
   time.sleep(.05)
  else:raise RuntimeError('Old masked process remains alive')
assert not Path('/proc/3178568/cmdline').exists(),'Old shared driver unexpectedly alive'
snapshot=[]
for rid in ids:
 folder=old/'runs'/rid;assert not folder.is_symlink()
 result=json.loads((folder/'run.json').read_text()) if (folder/'run.json').exists() else {}
 snapshot.append({'run_id':rid,'existed':folder.exists(),'status':result.get('status','paused_or_queued'),'agent_calls':result.get('agent_calls')})
(out/'deleted_runs.json').write_text(json.dumps(snapshot,indent=2))
for rid in ids:
 folder=old/'runs'/rid
 if folder.exists():shutil.rmtree(folder)
assert not any((old/'runs'/rid).exists() for rid in ids)
control=new/'control';control.mkdir()
env=dict(os.environ,PYTHONPATH=str(new/'source/src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['/tmp/ocs-es-refactor-venv/bin/python',str(new/'source/run_selected_cells.py'),'--root',str(new),'--selection',str(new/'selection.json'),'--control',str(control),'--workers','6']
with (control/'driver.log').open('ab') as log:p=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
record={'pid':p.pid,'command':cmd,'at':datetime.now(UTC).isoformat(),'replaces':str(old),'codex_unchanged':True,'unmasked_unchanged':True}
(control/'launch.json').write_text(json.dumps(record,indent=2))
with (new/'monitor.log').open('ab') as log:q=subprocess.Popen(['/tmp/ocs-es-refactor-venv/bin/python',str(out/'monitor.py')],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(new/'monitor_launch.json').write_text(json.dumps({'pid':q.pid}))
(old/'REPLACED.md').write_text(f'Qwen/Gemma results deleted by user request. Fresh authoritative root: {new}\n')
(old/'LIVE_PROGRESS.md').write_text(f'# Replaced\n\n[Current masked progress]({new}/LIVE_PROGRESS.md)\n')
(base/'20260910_clinical10pct_masked_workflows/vllm_handoff.json').write_text(json.dumps({**record,'corrected_masked_root':str(new)},indent=2))
print(json.dumps(record),flush=True)
time.sleep(3)
assert p.poll() is None,(control/'driver.log').read_text()[-4000:]
assert q.poll() is None,(new/'monitor.log').read_text()[-4000:]
