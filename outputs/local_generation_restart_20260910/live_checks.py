import json
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from onc_co_scientist.providers.registry import get_provider
from onc_co_scientist.providers.base import ChatMessage
from onc_co_scientist.expected_surprising.prompting import FORMS
out=Path(__file__).resolve().parent;repo=out.parents[1]
root=repo/'data/expected_surprising_ledger/full_runs/20260910_vllm_repair'
import yaml
models=yaml.safe_load((root/'config.yaml').read_text())['models']
def run(m,count):
 config=dict(m['provider_config'],sampling_profile='auto',json_object_output=True,disable_thinking_on_final_retry=True)
 provider=get_provider(config)
 p=next((root/'runs').glob('*'+m['id']+'*/calls/*appraise-agent-a1.json'))
 req=json.loads(p.read_text())['request']
 response=provider.chat_for_retry([ChatMessage(**x) for x in req['messages'][1:]],system=req['messages'][0]['content'],temperature=0,max_tokens=8192,final_retry=True,truncation_failures=count)
 assert not response.raw.get('adapter_error')
 FORMS['appraise'].model_validate(json.loads(response.text))
 key=m['id']+'_'+str(count)
 (out/(key+'.json')).write_text(json.dumps(response.raw,indent=2))
 result={'model':m['id'],'prior_truncations':count,'passed':True,'usage':response.raw['usage'],'options':response.raw['generation_options']}
 print(json.dumps(result),flush=True);return result
results=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(run,m,n) for m in models for n in (0,2)]
 for f in as_completed(futures):
  try:r=f.result()
  except Exception as e:r={'passed':False,'error':repr(e)};print(r,flush=True)
  results.append(r);(out/'live_results.json').write_text(json.dumps(results,indent=2))
assert len(results)==4 and all(r['passed'] for r in results)
