import csv,json,statistics,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import defaultdict
RATES={'gpt-6-astra':(10,1,12.5,50),'gpt-5.6-sol':(4,.4,5,20),'gpt-5.6-terra':(2,.2,2.5,12),'gpt-5.6-luna':(.2,.02,.25,1.2),'claude-opus-5':(5,.5,6.25,25)}
rows=[r for r in csv.DictReader(open('outputs/presentation_results/run_metrics.csv')) if r['llm'] in RATES]
def inspect(r):
 p=Path(r['source']); ri,rc,rw,ro=RATES[r['llm']]
 d={k:r[k] for k in ['llm','workflow','view','version','run_id','status']}
 d.update(input_tokens=0,cached_input_tokens=0,cache_write_input_tokens=0,output_tokens=0,cost_usd=0,uncached_cost_usd=0,usage_records=0,max_input_tokens=0,long_context_records=0,missing_usage_records=0)
 if r['llm'].startswith('gpt-'):
  # Every saved provider invocation counts, including retries that incurred tokens.
  paths=list((p/'provider_audit').glob('call-*/metrics.json'))
  us=[json.loads(f.read_text())['cli_usage'] for f in paths]
 else:
  paths=list((p/'calls').glob('*.json'))
  us=[json.loads(f.read_text())['result'].get('usage',{}) for f in paths]
 for u in us:
  i,o=u.get('input_tokens'),u.get('output_tokens')
  if i is None or o is None:
   d['missing_usage_records']+=1;continue
  c=u.get('cached_input_tokens',0) or 0;w=u.get('cache_write_input_tokens',0) or 0
  assert 0<=c+w<=i,(r['run_id'],u)
  long=r['llm'].startswith('gpt-') and i>272000
  mi,mo=(2,1.5) if long else (1,1)
  d['cost_usd']+=((i-c-w)*ri*mi+c*rc*mi+w*rw*mi+o*ro*mo)/1e6
  d['uncached_cost_usd']+=(i*ri*mi+o*ro*mo)/1e6
  d['usage_records']+=1;d['input_tokens']+=i;d['cached_input_tokens']+=c
  d['cache_write_input_tokens']+=w;d['output_tokens']+=o
  d['max_input_tokens']=max(d['max_input_tokens'],i);d['long_context_records']+=long
 d['report_output_tokens']=int(r['output_tokens']) if r['output_tokens'] else None
 d['report_input_tokens']=int(r['input_tokens']) if r['input_tokens'] else None
 d['reported_missing_token_calls']=int(r['missing_token_calls'] or 0)
 d['unaccounted_infrastructure_attempts']=int(r['unknown_internal_attempts'] or 0)
 return d
results=[]
with ThreadPoolExecutor(max_workers=12) as pool:
 for f in as_completed([pool.submit(inspect,r) for r in rows]):
  results.append(f.result())
  if len(results)%60==0:print('processed',len(results),'/',len(rows),flush=True)
results.sort(key=lambda r:(list(RATES).index(r['llm']),r['workflow'],r['run_id']))
Path('/tmp/clinical_api_costs_runs.json').write_text(json.dumps(results,indent=2))
def group_summary(group):
 completed=[r for r in group if r['status']=='completed']
 return dict(runs=len(group),completed=len(completed),input_m=sum(r['input_tokens'] for r in group)/1e6,cached_m=sum(r['cached_input_tokens'] for r in group)/1e6,write_m=sum(r['cache_write_input_tokens'] for r in group)/1e6,output_m=sum(r['output_tokens'] for r in group)/1e6,cost=sum(r['cost_usd'] for r in group),uncached_cost=sum(r['uncached_cost_usd'] for r in group),mean_completed=statistics.mean(r['cost_usd'] for r in completed),median_completed=statistics.median(r['cost_usd'] for r in completed),max_input=max(r['max_input_tokens'] for r in group),long_calls=sum(r['long_context_records'] for r in group),records=sum(r['usage_records'] for r in group),report_output_delta=sum(r['output_tokens']-(r['report_output_tokens'] or 0) for r in group),input_mismatches=sum(r['input_tokens']!=r['report_input_tokens'] for r in group if r['report_input_tokens'] is not None),missing=sum(r['missing_usage_records'] for r in group),report_missing=sum(r['reported_missing_token_calls'] for r in group))
for model in RATES:
 selected=[r for r in results if r['llm']==model]
 print(model,json.dumps(group_summary(selected)),flush=True)
 for workflow in ['persistent','sequential','deliberative']:
  print(workflow,json.dumps(group_summary([r for r in selected if r['workflow']==workflow])),flush=True)
