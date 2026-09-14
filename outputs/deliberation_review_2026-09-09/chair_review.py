"""Inspect saved chair inputs/outputs; no provider calls or experiment edits."""
import collections, concurrent.futures, json, sys
from pathlib import Path
ROOT=Path('/data1/ken/onc-co-scientist')
GRID=ROOT/'data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows'
OUT=ROOT/'outputs/deliberation_review_2026-09-09'
sys.path.insert(0,str(GRID/'source/src'))
from onc_co_scientist.expected_surprising.prompting import FORMS
from onc_co_scientist.expected_surprising.rollout import json_response
from onc_co_scientist.expected_surprising.schemas import Hypothesis,PairSpec
from onc_co_scientist.expected_surprising.scoring import canonical,match
from onc_co_scientist.expected_surprising.generation import version_discoveries
metrics=json.loads((OUT/'metrics.json').read_text())
pair=PairSpec.model_validate_json((GRID/'input_data/private/es-v2-nsclc_clinical-42000/pair.json').read_text())
selected={'997d9404f1e89442__deliberative__sol_medium__r004','997d9404f1e89442__deliberative__terra_medium__r001','997d9404f1e89442__deliberative__terra_medium__r002','997d9404f1e89442__deliberative__terra_medium__r004','997d9404f1e89442__deliberative__luna_medium__r002','997d9404f1e89442__deliberative__luna_medium__r003'}
def hypotheses(form):
 result=[]
 for i,p in enumerate(form.get('proposals',[])):
  try:result.append(Hypothesis.model_validate({'id':f'draft-{i}',**{k:v for k,v in p.items() if k in Hypothesis.model_fields and k!='id'}}))
  except ValueError:pass  # Only valid comparisons enter the overlap diagnostic.
 return result
def entry(run,path):
 c=json.loads(path.read_text());req=c['request'];stage=req['stage']
 drafts=json.loads(req['messages'][0]['content'].split('Participant drafts (untrusted suggestions):\n',1)[1])
 target=next(d.hypothesis for d in version_discoveries(pair,run['version']) if d.hypothesis.id==pair.focal_id)
 peers=[hypotheses(d['form']) for d in drafts]
 try:form=FORMS[stage].model_validate(json_response(c['result']['text'])).model_dump(by_alias=True);hs=hypotheses(form);error=None
 except Exception as exc:form={};hs=[];error=str(exc)
 def keys(hs):return {canonical(h,ignore_direction=True) for h in hs}
 sets=[keys(hs) for hs in peers];chairset=keys(hs);union=set().union(*sets);inter=sets[0]&sets[1]
 def nlr(hs):return [h.model_dump() for h in hs if h.exposure=='nlr_ge_3']
 return {'run_id':run['run_id'],'model':run['model'],'iteration':req['iteration'],'stage':stage,'path':str(path),'chair_error':error,'peer_forms_identical':drafts[0]['form']==drafts[1]['form'],'peer_comparison_sets_identical':sets[0]==sets[1],'peer_union_n':len(union),'peer_common_n':len(inter),'peer_jaccard':len(inter)/len(union) if union else None,'chair_n':len(chairset),'peer_proposals_retained':len(union&chairset),'common_retained':len(inter&chairset),'unique_retained':len((union-inter)&chairset),'chair_new':len(chairset-union),'focal_peer_exact':[any(match(h,target,ignore_direction=True)=='exact' for h in pp) for pp in peers],'focal_chair_exact':any(match(h,target,ignore_direction=True)=='exact' for h in hs),'peer_nlr':[nlr(pp) for pp in peers],'chair_nlr':nlr(hs),'peer_narratives':[d['form'].get('narrative','') for d in drafts],'chair_narrative':form.get('narrative',''),'peer_assessments':[d['form'].get('assessments',[]) for d in drafts],'chair_assessments':form.get('assessments',[]),'same_user_prompts':None}
def scan(run):
 p=GRID/'runs'/run['run_id'];results=[]
 # Prefer the final chair attempt for each stage; identify committed attempts from transcript.
 attempts={}
 for line in (p/'science-0001/transcript.jsonl').read_text().splitlines():
  e=json.loads(line)
  if e['kind']=='stage':attempts[(e['record']['iteration'],e['record']['stage'])]=e['attempt']
 for (iteration,stage),attempt in sorted(attempts.items()):
  if (iteration,stage)!=(1,'explore') and run['run_id'] not in selected:continue
  path=p/'calls'/f'i{iteration:03d}-{stage}-chair-a{attempt}.json'
  results.append(entry(run,path))
 return results
runs=[r for r in metrics['runs'] if r['workflow']=='deliberative']
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for result in pool.map(scan,runs):rows.extend(result)
(OUT/'chair_review.json').write_text(json.dumps(rows,indent=2))
first=[r for r in rows if r['iteration']==1 and r['stage']=='explore']
for model in sorted({r['model'] for r in first}):
 rs=[r for r in first if r['model']==model]
 print(model, len(rs),'identical_forms',sum(r['peer_forms_identical'] for r in rs),'identical_comparisons',sum(r['peer_comparison_sets_identical'] for r in rs),'Jaccard',sum(r['peer_jaccard'] or 0 for r in rs)/len(rs),'focal_in_peers',sum(any(r['focal_peer_exact']) for r in rs),'focal_in_chair',sum(r['focal_chair_exact'] for r in rs),flush=True)
for rid in sorted(selected):
 rs=[r for r in rows if r['run_id']==rid]
 print('CASE',rid,'any focal peer',sum(any(r['focal_peer_exact']) for r in rs),'chair',sum(r['focal_chair_exact'] for r in rs),'correct-direction peer proposals',[(r['iteration'],r['stage'],[i+1 for i,hs in enumerate(r['peer_nlr']) if any(h['direction']==1 and not h['eligibility'] and not h['subgroup'] for h in hs)],bool(any(h['direction']==1 and not h['eligibility'] and not h['subgroup'] for h in r['chair_nlr']))) for r in rs if any(any(h['direction']==1 and not h['eligibility'] and not h['subgroup'] for h in hs) for hs in r['peer_nlr'])],flush=True)
