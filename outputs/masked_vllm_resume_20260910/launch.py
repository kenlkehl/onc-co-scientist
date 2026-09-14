from pathlib import Path
import json,os,subprocess,sys,time
from datetime import datetime,UTC
repo=Path(__file__).resolve().parents[2]
root=repo/'data/expected_surprising_ledger/full_runs/20260910_masked_vllm_repair'
assert '6 passed' in (Path(__file__).with_name('tests.log')).read_text()
assert not (root/'launch.json').exists()
env=dict(os.environ,PYTHONPATH=str(root/'source/src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
command=[sys.executable,str(root/'source/supervise_masked.py'),'--root',str(root)]
with (root/'supervisor.log').open('ab') as log:
 process=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':process.pid,'started_at':datetime.now(UTC).isoformat(),'command':command}
(root/'supervisor_launch.json').write_text(json.dumps(record,indent=2))
for _ in range(30):
 if process.poll() is not None:raise RuntimeError((root/'supervisor.log').read_text()[-5000:])
 if (root/'launch.json').exists():break
 time.sleep(.5)
else:raise RuntimeError('Supervisor did not initialize; inspect log before retrying')
print((root/'launch.json').read_text(),flush=True)
