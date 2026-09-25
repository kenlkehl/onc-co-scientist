"""Sample only allocated GPUs until the supervised acceptance cell terminates."""
import datetime,json,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parent
status=root/'acceptance_runtime/status.json'
with (root/'gpu_samples.jsonl').open('a',buffering=1) as out:
    while True:
        sample={'time':datetime.datetime.now(datetime.UTC).isoformat()}
        p=subprocess.run(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
        sample['gpus']=[dict(zip(['index','memory_used_mib','memory_total_mib','utilization_pct'],map(int,row.split(',')))) for row in p.stdout.strip().splitlines()] if p.returncode==0 else []
        if p.returncode: sample['error']=p.stderr
        state=json.loads(status.read_text()) if status.exists() else {}
        sample['status']=state.get('status')
        out.write(json.dumps(sample)+'\n')
        if state.get('ended_at'): break
        time.sleep(15)
