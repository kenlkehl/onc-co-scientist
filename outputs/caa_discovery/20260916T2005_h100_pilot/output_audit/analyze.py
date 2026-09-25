from pathlib import Path
import collections, csv, hashlib, json, re, statistics
ROOT=Path('/data1/ken/onc-co-scientist/data/caa_discovery/20260916T2005_h100_pilot')
OUT=Path(__file__).resolve().parent
rows=[]
excerpts=[]
for view in ['masked','unmasked']:
 for plan in json.loads((ROOT/view/'plan.json').read_text()):
  run=ROOT/view/'runs'/plan['run_id']
  report=json.loads((run/'report.json').read_text())
  terminal=json.loads((run/'run.json').read_text())
  transcript=sorted(run.glob('science-*/transcript.jsonl'))[-1]
  events=[json.loads(l) for l in transcript.read_text().splitlines()]
  stages=[e for e in events if e['kind']=='stage']
  assert len(stages)==100
  calls={p.stem:json.loads(p.read_text()) for p in (run/'calls').glob('*.json')}
  committed=[]
  for e in stages:
   s=e['record'];slot=f"i{s['iteration']:03d}-{s['stage']}-agent-a{e['attempt']}"
   committed.append(calls[slot])
  texts=[a['result']['text'] for a in committed]
  hypotheses=[x['hypothesis'] for x in report['state']['registrations']]
  outcome=dict(condition=plan['semantic_condition']+'_'+view,arm=plan['model_id'],source=str(run),
   calls=len(calls),repairs=len(terminal['call_failures']),
   output_tokens=terminal['usage']['output_tokens'],
   committed_output_tokens=sum(a['result']['usage']['output_tokens'] for a in committed),
   final_chars=sum(len(t) for t in texts),
   narrative_words=sum(len(re.findall(r"\b[\w'-]+\b",s['record'].get('narrative',''))) for s in stages),
   fenced_responses=sum(t.lstrip().startswith('```') for t in texts),
   last_analysis_iteration=max(s['record']['iteration'] for s in stages if s['record']['executed_ids']),
   active_analysis_stages=sum(bool(s['record']['executed_ids']) for s in stages),
   proposed=report['exploration'][-1]['cumulative_proposed'],tested=report['exploration'][-1]['cumulative_tested'],
   interaction_proposals=sum(h['contrast']=='interaction' for h in hypotheses),
   conditioned_proposals=sum(bool(h['eligibility'] or h['subgroup']) for h in hypotheses),
   voluntary_validations=report['validation']['voluntary_slots_used'],
   final_accepted=len(report['final_accepted_ids']),
   f1=report['scores']['D'],
   invalid_results=sum(not result['valid'] for e in stages for result in e['results']),
   first_prompt_sha256=hashlib.sha256(json.dumps(calls['i001-explore-agent-a1']['request']['messages'],sort_keys=True).encode()).hexdigest(),
   first_output_tokens=calls['i001-explore-agent-a1']['result']['usage']['output_tokens'],
   first_text_chars=len(calls['i001-explore-agent-a1']['result']['text']),
   response_classes=report['responsiveness']['components'],
   errors=[e['error'].split('\n')[0] for e in terminal['call_failures']])
  for stage in ['explore','analyze','appraise','synthesize']:
   group=[a for a in committed if a['request']['stage']==stage]
   outcome[stage+'_output_tokens']=sum(a['result']['usage']['output_tokens'] for a in group)
   outcome[stage+'_final_chars']=sum(len(a['result']['text']) for a in group)
  rows.append(outcome)
  excerpts.append('\n## '+outcome['condition']+' '+outcome['arm']+'\n')
  for a in committed:
   req=a['request']
   if (req['iteration'] in [1,2] and req['stage']=='appraise') or (req['iteration'] in [10,25] and req['stage']=='synthesize'):
    excerpts.append('\n### '+req['slot']+'\n'+a['result']['text'])
(OUT/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'excerpts.md').write_text('\n'.join(excerpts)+'\n')
cols=[k for k in rows[0] if k not in ['response_classes','errors']]
with (OUT/'metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
for r in rows:
 print(json.dumps({k:v for k,v in r.items() if k not in ['source','response_classes','errors','first_prompt_sha256'] and not k.endswith('_final_chars') and not (k.endswith('_output_tokens') and k.split('_')[0] in ['explore','analyze','appraise','synthesize'])}))
print('MATCHED FIRST PROMPTS',[(c,len({r['first_prompt_sha256'] for r in rows if r['condition']==c})) for c in sorted({r['condition'] for r in rows})])
print('TOTALS')
for arm in ['gemma4-caa','gemma4-control']:
 selected=[r for r in rows if r['arm']==arm]
 print(arm,{k:sum(r[k] for r in selected) for k in ['repairs','output_tokens','committed_output_tokens','final_chars','narrative_words','proposed','tested','interaction_proposals','voluntary_validations']})
 print('RESPONSIVENESS',{c:(sum(r['response_classes'][c]['numerator'] for r in selected),sum(r['response_classes'][c]['denominator'] for r in selected)) for c in ['supported','excluded','ambiguous']})
# Trace formal acceptance separately from narrative recognition, using only the
# final replay transcript and committed stage records.
timelines=[]
for row in rows:
 run=Path(row['source']);report=json.loads((run/'report.json').read_text())
 hypotheses={x['hypothesis']['id']:x['hypothesis'] for x in report['state']['registrations']}
 accepted=[hypotheses[i] for i in report['final_accepted_ids']]
 events=[json.loads(l) for l in sorted(run.glob('science-*/transcript.jsonl'))[-1].read_text().splitlines()]
 trace=[]
 for e in events:
  if e['kind']!='stage':continue
  stage=e['record']
  for a in stage['assessment_records']:
   h=hypotheses[a['hypothesis_id']]
   if a['status']=='accept' and h['exposure'] in ['ecog_ps_ge_2','nlr_ge_3','feature_025','feature_028'] and not h['eligibility'] and not h['subgroup'] and h['contrast']=='mean_difference':
    if not any(t['hypothesis_id']==h['id'] for t in trace):
     trace.append(dict(hypothesis_id=h['id'],exposure=h['exposure'],direction=h['direction'],iteration=stage['iteration'],stage=stage['stage'],evidence=a['result_ids']))
 timelines.append(dict(condition=row['condition'],arm=row['arm'],source=row['source'],
  accepted_conditioned=sum(bool(h['eligibility'] or h['subgroup']) for h in accepted),
  first_acceptances=trace,transcript=str(sorted(run.glob('science-*/transcript.jsonl'))[-1]),
  report_sha256=hashlib.sha256((run/'report.json').read_bytes()).hexdigest()))
(OUT/'acceptance_timelines.json').write_text(json.dumps(timelines,indent=2)+'\n')
