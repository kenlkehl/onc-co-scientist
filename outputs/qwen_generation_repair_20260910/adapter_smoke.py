import json
from pathlib import Path
from onc_co_scientist.providers.registry import get_provider
from onc_co_scientist.providers.base import ChatMessage
from onc_co_scientist.expected_surprising.prompting import FORMS
out=Path(__file__).resolve().parent;repo=out.parents[1]
config={'kind':'vllm_openai','model_id':'Inferact/Qwen3.8-27B-NVFP4','base_url':'http://sn4622130540:8000/v1','timeout_s':240,'reasoning_effort':'medium','service_tier':'default','sampling_profile':'qwen3_8','json_object_output':True,'disable_thinking_on_final_retry':True}
(out/'qwen_provider_candidate.json').write_text(json.dumps(config,indent=2))
p=get_provider(config)
req=json.loads((repo/'data/expected_surprising_ledger/full_runs/20260910_vllm_repair/runs/997d9404f1e89442__persistent__qwen_3_8_27b__r002/calls/i009-appraise-agent-a1.json').read_text())['request']
results=[]
for final in (False,True):
 r=p.chat_for_retry([ChatMessage(**m) for m in req['messages'][1:]],system=req['messages'][0]['content'],temperature=0,max_tokens=8192,final_retry=final)
 FORMS['appraise'].model_validate(json.loads(r.text));assert not r.raw.get('adapter_error')
 results.append({'final_retry':final,'passed':True,'usage':r.raw['usage'],'actual_generation_options':r.raw['generation_options'],'response':r.text})
 (out/'adapter_smoke_results.json').write_text(json.dumps(results,indent=2))
 print(final,r.raw['usage'],flush=True)
