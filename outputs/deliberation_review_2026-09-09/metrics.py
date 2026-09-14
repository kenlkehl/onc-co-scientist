import collections, concurrent.futures, json
from pathlib import Path

ROOT=Path('/data1/ken/onc-co-scientist')
GRID=ROOT/'data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows'
OUT=ROOT/'outputs/deliberation_review_2026-09-09'
snap=json.loads((ROOT/'outputs/interim_results_9-8-26/snapshot.json').read_text())
def load(run):
 r=json.loads((GRID/'runs'/run['run_id']/'report.json').read_text())
 focal=next(m for m in r['behavior']['milestones'] if m['focal'])
 row={'run_id':run['run_id'],'model':run['model_profile'],'workflow':run['workflow_id'],'version':run['semantic_condition'],'replicate':run['replicate'],'status':run['status'],'F1':r['scores']['D'],'diagnostic_F1':r['scores']['discovery']['exact']['diagnostic_D'],'R':r['scores']['discovery']['exact']['R'],'P':r['scores']['discovery']['exact']['Q'],'E':r['scores']['E'],'B':r['scores']['B'],'focal':r['confirmation']['primary_recovery'],'focal_milestone':focal,'calls':run['agent_calls'],'proposals':len(r['state']['registrations']),'analyses':len(r['state']['executions']),'unique_tests':len({e['comparison_key'] for e in r['state']['executions'] if e['result']['valid']}),'protocol_errors':r['protocol_errors'],'draft_errors':r['coordination']['draft_errors'],'provider_errors':r['coordination']['provider_error_calls'],'stages':r['successful_stages'],'category_recall':r['scores']['discovery']['exact']['R_c']}
 row['peer_veto_stages']=[e for e in row['protocol_errors'] if 'exhausted draft repairs' in e['error']]
 row['other_failed_stages']=[e for e in row['protocol_errors'] if 'exhausted draft repairs' not in e['error']]
 row['native_provider_failure']=[e for e in row['protocol_errors'] if 'Codex turn failed' in e['error']]
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(load,snap['runs']))
def balanced(rows,key):
 means=[]
 for v in ['expected','surprising']:
  xs=[r[key] for r in rows if r['version']==v and r[key] is not None]
  if xs:means.append(sum(xs)/len(xs))
 return sum(means)/len(means) if means else None
groups=[]
for model in sorted({r['model'] for r in rows}):
 for w in ['persistent','sequential','deliberative']:
  rs=[r for r in rows if r['model']==model and r['workflow']==w]
  groups.append({'model':model,'workflow':w,'n':len(rs),'failed':sum(r['status']=='failed' for r in rs),'peer_veto_runs':sum(bool(r['peer_veto_stages']) for r in rs),'peer_veto_stages':sum(len(r['peer_veto_stages']) for r in rs),'other_failed_stages':sum(len(r['other_failed_stages']) for r in rs),'provider_errors':sum(r['provider_errors'] for r in rs),**{k:balanced(rs,k) for k in ['F1','diagnostic_F1','R','P','E','proposals','unique_tests']}})
pairs=[]
for model in sorted({r['model'] for r in rows}):
 for baseline in ['persistent','sequential']:
  index={(r['version'],r['replicate']):r for r in rows if r['model']==model and r['workflow']==baseline}
  ds=[r for r in rows if r['model']==model and r['workflow']=='deliberative']
  for clean in [False,True]:
   selected=[]
   for d in ds:
    b=index.get((d['version'],d['replicate']))
    if b is None or (clean and (b['status']!='completed' or d['status']!='completed')):continue
    selected.append({'version':d['version'],'replicate':d['replicate'],'baseline':b['run_id'],'deliberative':d['run_id'],**{k:d[k]-b[k] if d[k] is not None and b[k] is not None else None for k in ['F1','diagnostic_F1','R','P','E','proposals','unique_tests','focal']}})
   pairs.append({'model':model,'baseline':baseline,'both_completed_only':clean,'n':len(selected),'versions':dict(collections.Counter(p['version'] for p in selected)),**{k:balanced(selected,k) for k in ['F1','diagnostic_F1','R','P','E','proposals','unique_tests','focal']},'pairs':selected})
result={'snapshot_at':snap['at'],'runs':rows,'groups':groups,'matched_differences_deliberative_minus_baseline':pairs}
(OUT/'metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'groups':groups,'contrasts':[{k:v for k,v in p.items() if k!='pairs'} for p in pairs]},indent=2),flush=True)
