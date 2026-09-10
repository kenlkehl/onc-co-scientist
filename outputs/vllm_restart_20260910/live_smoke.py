from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import pandas as pd
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans
from onc_co_scientist.expected_surprising.workflow import WorkflowController
from onc_co_scientist.expected_surprising.schemas import PairSpec,ValidationPolicy
from onc_co_scientist.expected_surprising.coordination import StageCoordinator
from onc_co_scientist.expected_surprising.prompting import build_prompt,translate
from onc_co_scientist.expected_surprising.rollout import json_response,stage_transaction
from onc_co_scientist.providers.registry import get_provider
repo=Path(__file__).resolve().parents[2]
root=repo/'data/expected_surprising_ledger/full_runs/20260910_vllm_repair'
out=Path(__file__).resolve().parent/'live_smoke_retries';out.mkdir(exist_ok=False)
spec=load_experiment_spec(root/'config.yaml')
plans=[p for p in build_run_plans(spec) if p.replicate==1 and p.workflow.id=='deliberative' and p.task.semantic_condition=='expected']
def run(plan):
 folder=out/plan.model.id;folder.mkdir()
 pair=PairSpec.model_validate_json(plan.task.private_evaluation_path.read_text())
 policy=ValidationPolicy.model_validate(json.loads((plan.task.private_evaluation_path.parent/'workflow.json').read_text())['policy'])
 public=plan.task.public_workspace;frame=pd.read_parquet(public/'dataset.parquet')
 c=WorkflowController(pair,'expected',frame,policy,'smoke-only')
 config=dict(plan.model.provider_config,audit_dir=str(folder/'provider_audit'))
 provider=get_provider(config)
 coord=StageCoordinator(provider,plan.workflow,spec.stages,spec.expected_surprising,spec.budget,folder/'calls')
 context={'instructions':(public/'instructions.md').read_text(),'outcomes':json.loads((public/'task.json').read_text())['outcomes'],'observations':len(frame),'variables':{k:v for k,v in json.loads(frame.describe(include='all').fillna('').to_json()).items() if k!='patient_id'}}
 prompt=build_prompt(c,context,1,25,'explore',1,{},None)
 for attempt in range(1,4):
  try:
   response=coord.respond(prompt,iteration=1,stage='explore',attempt=attempt)
   record,form,audit=translate(c,'explore',1,json_response(response.text))
   with stage_transaction([c.state,c.service.state]):c.apply(record)
   break
  except Exception as exc:
   coord.reject()
   if attempt==3:raise
   feedback={'error':str(exc),'instruction':'Return exactly one JSON object; escape newlines inside strings. Nothing was committed.'}
   prompt=build_prompt(c,context,1,25,'explore',attempt+1,{},feedback)
 coord.commit(); ca=coord.audit()
 result={'model':plan.model.id,'passed':True,'claims':len(record.hypotheses),'narrative':form.narrative,'audit':ca,'reasoning':config['reasoning_effort'],'tier':config['service_tier']}
 (folder/'result.json').write_text(json.dumps(result,indent=2));return result
results=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 futures={pool.submit(run,p):p.model.id for p in plans}
 for f in as_completed(futures):
  try:r=f.result()
  except Exception as e:r={'model':futures[f],'passed':False,'error':repr(e)}
  results.append(r);print(json.dumps({k:v for k,v in r.items() if k not in {'audit','narrative'}}),flush=True)
  (out/'results.json').write_text(json.dumps(results,indent=2))
assert all(r['passed'] for r in results)
