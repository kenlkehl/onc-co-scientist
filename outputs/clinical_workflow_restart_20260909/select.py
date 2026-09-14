import json,hashlib,re,collections
from pathlib import Path
out=Path(__file__).resolve().parent
snapshot=json.loads((out/'snapshot.json').read_text()); rows=[]
for row in snapshot['rows']:
 p=row['plan'];t=row['terminal']; rid=p['run_id']
 if p['model_profile'] not in {'luna_medium','terra_medium','sol_medium','astra_medium'}:continue
 evidence=[]; errors=t.get('call_failures',[]) if t else []
 for item in errors:
  message=item['error']
  if 'Codex turn failed; see ' not in message:continue
  path=Path(message.split('Codex turn failed; see ',1)[1].split('\n')[0])
  events=[json.loads(l) for l in (path/'events.jsonl').read_text().splitlines() if l.strip()]
  terminal=next((e for e in reversed(events) if e.get('type') in {'turn.started','turn.failed','turn.completed','error'}),{})
  if terminal.get('type')=='turn.completed' and any(e.get('type')=='error' for e in events) and (path/'final.txt').exists():
   evidence.append({'attempt':str(path),'events_sha256':hashlib.sha256((path/'events.jsonl').read_bytes()).hexdigest(),'output_tokens':terminal.get('usage',{}).get('output_tokens')})
 evidence=list({e['attempt']:e for e in evidence}.values())
 reasons=[]
 if t is None:reasons.append('unfinished_at_pause' if row['had_directory'] else 'not_started_at_pause')
 if evidence:reasons.append('adapter_discarded_completed_response')
 rows.append({'run_id':rid,'model_profile':p['model_profile'],'workflow_id':p['workflow_id'],'semantic_condition':p['semantic_condition'],'replicate':p['replicate'],'original_status':t['status'] if t else 'unfinished' if row['had_directory'] else 'queued','replace':bool(reasons),'reasons':reasons,'adapter_evidence':evidence})
(out/'selection_audit.json').write_text(json.dumps({'at':snapshot['at'],'rule':'Restart only Codex cells unfinished at pause or with native proof of an adapter-discarded completed response. Preserve other terminal cells, including non-adapter failures.','rows':rows},indent=2))
ids=[r['run_id'] for r in rows if r['replace']];(out/'codex_selection.json').write_text(json.dumps(ids,indent=2))
print(json.dumps({'selected':len(ids),'preserved':len(rows)-len(ids),'adapter_affected_terminal_runs':sum(bool(r['adapter_evidence']) for r in rows),'preserved_failures':[r['run_id'] for r in rows if not r['replace'] and r['original_status']=='failed'],'by_model':dict(collections.Counter(r['model_profile'] for r in rows if r['replace']))},indent=2))
