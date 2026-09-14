import collections,json,os,time
from pathlib import Path
from datetime import datetime,UTC
from onc_co_scientist.harness.durable_io import atomic_write_json,atomic_write_text
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans
from onc_co_scientist.expected_surprising.experiment_report import write_report
repo=Path(__file__).resolve().parents[2]
base=repo/'data/expected_surprising_ledger/full_runs'
root=base/'20260910_masked_vllm_recommended';parent=base/'20260910_clinical10pct_masked_workflows'
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return None
spec=load_experiment_spec(parent/'config.yaml');plans=build_run_plans(spec)
local=set(read(root/'selection.json'));ownership={p.run_id:str(root if p.run_id in local else parent) for p in plans}
overrides=read(root/'routing_overrides.json') or {}
assert set(overrides)<=local
ownership.update(overrides)
assert len(ownership)==360 and len(local)==120
atomic_write_json(root/'selected_run_roots.json',ownership)
while True:
 pause=read(parent/'astra_deliberative_pause.json') or {}
 held=set(pause.get('held_unfinished_ids',[])) if not pause.get('resumed_at') else set()
 groups=collections.Counter();results=[]
 for p in plans:
  folder=Path(ownership[p.run_id])/'runs'/p.run_id;r=read(folder/'run.json')
  status=r['status'] if r and r.get('status') in ('completed','failed') else ('paused' if p.run_id in held else 'active' if folder.exists() else 'queued')
  groups[p.model.id,p.workflow.id,status]+=1
  if status in ('completed','failed'):results.append(r)
 stamp=datetime.now(UTC).isoformat()
 lines=['# Masked grid progress','',f'Updated {stamp}','','240 original Codex identities plus 120 fresh Qwen/Gemma identities using recommended generation settings. Earlier masked Qwen/Gemma results were deleted by request. Twelve Gemma workers on camus:8060 serve reassigned queued runs, six workers finish existing Gemma runs on the original endpoint, and three dedicated Qwen workers continue. Astra deliberative resumed with explicit user authorization using six workers and saved-call replay; other Codex identities remain in their separate selection.','','| Model | Workflow | Completed | Failed | Active | Queued | Paused |','|---|---|---:|---:|---:|---:|---:|']
 for m,w in sorted({(p.model.id,p.workflow.id) for p in plans}):lines.append('| '+ ' | '.join([m,w]+[str(groups[m,w,s]) for s in ('completed','failed','active','queued','paused')])+' |')
 for dest in (root/'LIVE_PROGRESS.md',parent/'COMBINED_PROGRESS.md'):atomic_write_text(dest,'\n'.join(lines)+'\n')
 atomic_write_json(root/'monitor.json',{'pid':os.getpid(),'updated_at':stamp,'terminal_runs':len(results),'codex_terminal':sum(r['run_id'] not in local for r in results)})
 if len(results)==360:
  write_report(spec,plans,results,root)
  atomic_write_json(root/'combined_summary.json',{'n_runs':360,'n_completed':sum(r['status']=='completed' for r in results),'n_failed':sum(r['status']=='failed' for r in results),'runs':results,'selected_run_roots':ownership})
  break
 time.sleep(30)
