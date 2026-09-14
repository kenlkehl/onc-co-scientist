"""Bounded diagnostics only; never commits scientific actions or resumes runs."""
import json,urllib.request,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
out=Path(__file__).resolve().parent;repo=out.parents[1]
root=repo/'data/expected_surprising_ledger/full_runs/20260910_vllm_repair/runs'
cases=[('997d9404f1e89442__sequential__qwen_3_8_27b__r002','i017-appraise-agent-a1'),('997d9404f1e89442__persistent__qwen_3_8_27b__r002','i009-appraise-agent-a1'),('0dfe6f786ad7dd9d__sequential__qwen_3_8_27b__r001','i002-synthesize-agent-a1')]
def run(case,thinking):
 rid,slot=case;prior=json.loads((root/rid/'calls'/f'{slot}.json').read_text())['request']
 key=rid+'__'+slot+('__thinking' if thinking else '__instruct')
 body={'model':prior['model'],'messages':prior['messages'],'temperature':1.0 if thinking else 0.7,'top_p':0.95 if thinking else 0.8,'top_k':20,'min_p':0.0,'presence_penalty':0.0 if thinking else 1.5,'repetition_penalty':1.0,'chat_template_kwargs':{'enable_thinking':thinking},'response_format':{'type':'json_object'},'reasoning_effort':'medium','max_tokens':8192,'seed':1701}
 (out/(key+'.request.json')).write_text(json.dumps(body))
 started=time.monotonic()
 try:
  request=urllib.request.Request('http://sn4622130540:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=240) as response:d=json.load(response)
  (out/(key+'.response.json')).write_text(json.dumps(d))
  choice=d['choices'][0];text=choice['message'].get('content') or '';obj=json.loads(text)
  result={'case':key,'stage':prior['stage'],'thinking':thinking,'valid_json':isinstance(obj,dict),'finish_reason':choice['finish_reason'],'usage':d.get('usage'),'duration':time.monotonic()-started}
 except Exception as e:result={'case':key,'error':str(e),'duration':time.monotonic()-started}
 print(json.dumps(result),flush=True);return result
results=[]
with ThreadPoolExecutor(max_workers=3) as pool:
 fs=[pool.submit(run,c,t) for c in cases for t in (True,False)]
 for f in as_completed(fs):
  results.append(f.result());(out/'replay_results.json').write_text(json.dumps(results,indent=2))
