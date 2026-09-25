"""Post-hoc matched acceptance latencies, anchored at first numerical evidence."""
from pathlib import Path
import json, statistics, collections
from onc_co_scientist.expected_surprising.schemas import Hypothesis, EvidenceResult
from onc_co_scientist.expected_surprising.scoring import comparison_key, claim_key, evidence_alignment
ROOT=Path('/data1/ken/onc-co-scientist/data/caa_discovery/20260916T2005_h100_pilot')
OUT=Path(__file__).resolve().parent
runs={}
for view in ['masked','unmasked']:
 for plan in json.loads((ROOT/view/'plan.json').read_text()):
  report=json.loads((ROOT/view/'runs'/plan['run_id']/'report.json').read_text());state=report['state']
  condition=plan['semantic_condition']+'_'+view;arm=plan['model_id']
  first={}
  for e in state['events']:
   if e['source']=='discovery': first.setdefault(e['comparison_key'],e)
  items={}
  for key,e in first.items():
   r=EvidenceResult.model_validate(e['result']);h=Hypothesis.model_validate(e['hypothesis'])
   if not r.valid:continue
   # Restrict to evidence already beyond the meaningful-effect threshold in
   # either direction. A statistically nonzero but negligible effect is excluded.
   if r.lower>r.delta: target=h
   elif r.upper < -r.delta: target=h.model_copy(update={'direction':-h.direction})
   else:continue
   a=state['first_acceptances'].get(claim_key(target))
   items[key]=dict(exposure=h.exposure,contrast=h.contrast,conditioned=bool(h.eligibility or h.subgroup),
    evidence_iteration=e['iteration'],evidence_sequence=e['sequence'],hypothesis_id=h.id,
    expectation=evidence_alignment(e),pre_evidence=e['expectation']['pre_evidence'],
    claim_reversal=(target.direction!=h.direction),
    accepted_iteration=a['iteration'] if a else None,accepted_sequence=a['sequence'] if a else None,
    stage_delay=a['sequence']-e['sequence'] if a else None,
    iteration_delay=a['iteration']-e['iteration'] if a else None,
    expected_direction=e['anticipated_direction'],abs_estimate=abs(r.estimate),
    report_source=str(ROOT/view/'runs'/plan['run_id']/'report.json'))
  runs[(condition,arm)]=items
matched=[]
for condition in sorted({c for c,a in runs}):
 caa=runs[(condition,'gemma4-caa')];control=runs[(condition,'gemma4-control')]
 for key in sorted(caa.keys()&control.keys()):
  a,b=caa[key],control[key]
  group=a['expectation'] if a['expectation']==b['expectation'] else 'different_expectations'
  row=dict(condition=condition,comparison_key=key,exposure=a['exposure'],group=group,
   both_pre_evidence=a['pre_evidence'] and b['pre_evidence'],
   simple_main=not a['conditioned'] and a['contrast']=='mean_difference',caa=a,control=b)
  matched.append(row)
(OUT/'matched_acceptance_latency.json').write_text(json.dumps(matched,indent=2)+'\n')
print('MATCHED SIMPLE MAIN EFFECTS')
for r in matched:
 if not r['simple_main']:continue
 print(r['condition'],r['exposure'],r['group'],'pre',r['both_pre_evidence'],
  'evidence',[r[a]['evidence_iteration'] for a in ['caa','control']],
  'accepted',[r[a]['accepted_iteration'] for a in ['caa','control']],
  'delay',[r[a]['iteration_delay'] for a in ['caa','control']],
  'stages',[r[a]['stage_delay'] for a in ['caa','control']])
print('ALL MATCHED',len(matched))
for subset in ['simple','all']:
 for group in sorted({r['group'] for r in matched}):
  rr=[r for r in matched if r['group']==group and r['both_pre_evidence'] and (subset=='all' or r['simple_main'])]
  if not rr:continue
  print(subset,group,'n',len(rr))
  for arm in ['caa','control']:
   delays=[r[arm]['iteration_delay'] for r in rr if r[arm]['iteration_delay'] is not None]
   print(arm,'accepted',len(delays),'never',len(rr)-len(delays),'delays',delays,'mean',statistics.mean(delays) if delays else None)
