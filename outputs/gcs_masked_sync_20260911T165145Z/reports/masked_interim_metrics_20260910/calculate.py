import csv,hashlib,json,statistics
from pathlib import Path
from datetime import datetime,UTC
repo=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
root=repo/'data/expected_surprising_ledger/full_runs/20260910_masked_vllm_recommended'
ownership=json.loads((root/'selected_run_roots.json').read_text());assert len(ownership)==360
rows=[];statuses=[]
for rid,owner in sorted(ownership.items()):
 folder=Path(owner)/'runs'/rid;p=folder/'run.json';model=rid.split('__')[2];workflow=rid.split('__')[1]
 if not p.exists():
  statuses.append(dict(run_id=rid,model=model,workflow=workflow,status='active' if folder.exists() else 'queued'));continue
 r=json.loads(p.read_text());status=r['status'];statuses.append(dict(run_id=rid,model=model,workflow=workflow,status=status))
 if status not in ('completed','failed'):continue
 path=Path(r['scientific_report']);raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['scientific_report_sha256']
 d=json.loads(raw);s=d['scores'];exact=s['discovery']['exact'];a=d['coordination'].get('output_token_accounting',{})
 rows.append(dict(run_id=rid,model=model,workflow=workflow,version=d['version'],replicate=rid.split('__')[-1],status=status,primary=d['confirmation']['primary_recovery'],R=exact['R'],P=exact['Q'],F1=s['D'],E=s['E'],B=s['B'],calls=r['agent_calls'],known_output_tokens=a.get('known_output_tokens'),missing_token_calls=a.get('missing_calls'),unaccounted_infrastructure_attempts=a.get('unaccounted_infrastructure_attempts'),report_path=str(path),report_sha256=r['scientific_report_sha256']))
def mean(rs,k):
 vals=[r[k] for r in rs if r[k] is not None];return statistics.mean(vals) if vals else None
completed=[r for r in rows if r['status']=='completed'];groups=[]
for model,workflow in sorted({(s['model'],s['workflow']) for s in statuses}):
 rs=[r for r in completed if (r['model'],r['workflow'])==(model,workflow)];ss=[s for s in statuses if (s['model'],s['workflow'])==(model,workflow)]
 g=dict(model=model,workflow=workflow,**{s:sum(r['status']==s for r in ss) for s in ('completed','failed','active','queued')})
 for v in ('expected','surprising'):
  vs=[r for r in rs if r['version']==v];g[v+'_n']=len(vs);g[v+'_recovered']=sum(r['primary'] for r in vs);g[v+'_recovery']=mean(vs,'primary')
 g.update({k:mean(rs,k) for k in ('R','P','F1','E','B','calls','known_output_tokens')});g['B_n']=sum(r['B'] is not None for r in rs);g['P_n']=sum(r['P'] is not None for r in rs)
 matched=[]
 for rep in sorted({r['replicate'] for r in rs}):
  pair={r['version']:r for r in rs if r['replicate']==rep}
  if set(pair)=={'expected','surprising'}:matched.append(pair['surprising']['primary']-pair['expected']['primary'])
 g['matched_pairs']=len(matched);g['matched_difference_pp']=100*statistics.mean(matched) if matched else None;groups.append(g)
for name,data in [('run_metrics',rows),('condition_metrics',groups),('run_status',statuses)]:
 with (out/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
stamp=datetime.now(UTC).isoformat();summary=dict(as_of=stamp,counts={s:sum(r['status']==s for r in statuses) for s in ('completed','failed','active','queued')},conditions=groups)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
def fmt(x,m=1):return '—' if x is None else f'{x*m:.1f}'
def rec(g,v):return '—' if not g[v+'_n'] else f"{g[v+'_recovered']}/{g[v+'_n']} ({100*g[v+'_recovery']:.0f}%)"
lines=['# Interim masked benchmark metrics','',f'Snapshot: {stamp}.','',str(summary['counts']),'','Completed runs only in the performance tables; failures are counted separately. Pending runs are never scored as failures. Previous deleted local-model runs are excluded. Means weight available completed runs equally; expected/surprising and workflow completion counts are uneven. These are descriptive interim results, not a final ranking. Only one base dataset pair is represented, so no across-dataset confidence intervals are estimated.','','## Focal recovery','','Recovery requires recovery of the focal planted finding and independent evaluator confirmation.','','| Model | Workflow | Complete | Failed | Expected recovered/n | Surprising recovered/n | Matched pairs | Matched S−E (pp) |','|---|---|---:|---:|---|---|---:|---:|']
for g in groups:lines.append('| '+' | '.join([g['model'],g['workflow'],str(g['completed']),str(g['failed']),rec(g,'expected'),rec(g,'surprising'),str(g['matched_pairs']),fmt(g['matched_difference_pp'])])+' |')
lines+=['','Matched differences use only replicates with both versions completed. Separate recovery columns use all completed runs in each version.','','## Supporting metrics','','R: category-balanced planted-finding recall; P: independently confirmed fraction of accepted claims; F1*: mean per-run harmonic discovery score; E: exploration coverage; B: evidence responsiveness (available runs only). R and P are percentages; F1*, E, and B are on 0–100 scales.','','| Model | Workflow | R % | P % | F1* | E | B (n) | Mean calls | Mean known output tokens |','|---|---|---:|---:|---:|---:|---|---:|---:|']
for g in groups:lines.append('| '+' | '.join([g['model'],g['workflow'],fmt(g['R'],100),fmt(g['P'],100),fmt(g['F1']),fmt(g['E']),f"{fmt(g['B'])} ({g['B_n']})",fmt(g['calls']),fmt(g['known_output_tokens'])])+' |')
lines+=['','Known token counts are lower bounds where provider or infrastructure usage is missing; missingness is recorded per run in run_metrics.csv. Tokens and calls here cover completed runs only. Report SHA-256 checks passed for every included terminal run.','']
(out/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(summary,indent=2))
