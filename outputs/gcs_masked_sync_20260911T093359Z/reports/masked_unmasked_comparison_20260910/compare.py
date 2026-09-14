import csv,json,hashlib,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,UTC
repo=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
masked=list(csv.DictReader((repo/'outputs/masked_interim_metrics_20260910/run_metrics.csv').open()))
named=json.loads((repo/'outputs/clinical_workflow_restart_20260909/living_metrics.json').read_text())
def key(r):return r['model'],r['workflow'],r['version'],r['run_id'].split('__')[-1]
lookup={key(r):r for r in named['runs']};pairs=[];excluded=[]
def inspect(m):
 n=lookup[key(m)]
 if m['status']!='completed' or n['state']!='completed':return None
 path=Path(n['folder']);r=json.loads((path/'run.json').read_text());raw=(path/'report.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==r['scientific_report_sha256'];d=json.loads(raw)
 row={k:m[k] for k in ('model','workflow','version','replicate')};row.update(masked_id=m['run_id'],unmasked_id=n['run_id'],unmasked_generation=n['generation'])
 for k,val in [('primary',d['confirmation']['primary_recovery']),('R',d['scores']['discovery']['exact']['R']),('P',d['scores']['discovery']['exact']['Q']),('F1',d['scores']['D']),('E',d['scores']['E'])]:
  row['masked_'+k]=float(m[k]) if m[k] else None;row['unmasked_'+k]=val
 return row
with ThreadPoolExecutor(max_workers=8) as pool:pairs=[x for x in pool.map(inspect,masked) if x]
def mean(rs,k):
 xs=[r[k] for r in rs if r[k] is not None];return statistics.mean(xs) if xs else None
def summarize(rs,label):
 d={'label':label,'n':len(rs)}
 for k in ('primary','R','P','F1','E'):
  for side in ('masked','unmasked'):d[side+'_'+k]=mean(rs,side+'_'+k)
  d['delta_'+k]=d['masked_'+k]-d['unmasked_'+k] if d['masked_'+k] is not None and d['unmasked_'+k] is not None else None
 for side in ('masked','unmasked'):d[side+'_recovered']=sum(r[side+'_primary'] for r in rs)
 return d
conditions=[summarize([r for r in pairs if (r['model'],r['workflow'])==(m,w)],m+' / '+w) for m,w in sorted({(r['model'],r['workflow']) for r in pairs})]
models=[summarize([r for r in pairs if r['model']==m],m) for m in sorted({r['model'] for r in pairs})]
versions=[summarize([r for r in pairs if r['version']==v],v) for v in ('expected','surprising')]
for filename,rows in [('paired_runs',pairs),('conditions',conditions),('models',models),('versions',versions)]:
 with (out/(filename+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
summary={'created_at':datetime.now(UTC).isoformat(),'unmasked_snapshot':named['at'],'masked_snapshot':'2026-09-10T23:33:55.604358+00:00','matched_completed_n':len(pairs),'masked_completed_unmatched':[{'id':m['run_id'],'unmasked_status':lookup[key(m)]['state']} for m in masked if m['status']=='completed' and lookup[key(m)]['state']!='completed'],'models':models,'conditions':conditions,'versions':versions}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Matched masked versus unmasked interim comparison','','Uses the prior 176-completed masked snapshot. Only runs completed on both sides are compared, matching model, workflow, expected/surprising version, and replicate. Deleted local-model results are excluded. Differences are masked minus unmasked. Means weight the matched runs equally.','','Unmasked snapshot: '+named['at']+'.',f"Matched completed runs: {len(pairs)}; masked completed runs excluded because their unmasked counterparts were not completed: {len(summary['masked_completed_unmatched'])}.",'','## By workflow','','| Model / workflow | Matched n | Unmasked focal recovery | Masked focal recovery | Unmasked F1* | Masked F1* | Δ F1* |','|---|---:|---:|---:|---:|---:|---:|']
for r in conditions:lines.append(f"| {r['label']} | {r['n']} | {r['unmasked_recovered']:g}/{r['n']} | {r['masked_recovered']:g}/{r['n']} | {r['unmasked_F1']:.1f} | {r['masked_F1']:.1f} | {r['delta_F1']:+.1f} |")
lines+=['','F1* is the existing 0–100 discovery score. Focal recovery requires independent confirmation. This completed-only analysis excludes failed runs on either side and can be affected by which runs finish first. Repeated runs share one base dataset pair; no across-dataset confidence intervals or causal masking-effect claim is justified. The unmasked Codex cohort mixes retained original and repaired implementations; masked Codex uses its frozen implementation. Qwen/Gemma use matching newly recommended caller settings on both sides. Per-pair generation provenance is retained in paired_runs.csv.','']
(out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(summary,indent=2))
