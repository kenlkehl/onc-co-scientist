"""Recheck audit assertions corrected while the first read-only scan was running.

Provenance uses a canonical JSON digest, not the bytes of its pretty-printed file.
Peer repairs also zero the diagnostic first-attempt score. Neither change alters
an experimental artifact or its scientific scores.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/data1/ken/onc-co-scientist')
GRID=ROOT/'data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows'
OUT=ROOT/'outputs/interim_results_9-8-26'
sys.path.insert(0,str(GRID/'source/src'))
from onc_co_scientist.expected_surprising.schemas import PairSpec,Hypothesis
from onc_co_scientist.expected_surprising.generation import version_discoveries
from onc_co_scientist.expected_surprising.scoring import workflow_discovery

data=json.loads((OUT/'audit_results.json').read_text())
data.pop('total',None)
pair=PairSpec.model_validate_json((GRID/'input_data/private/es-v2-nsclc_clinical-42000/pair.json').read_text())

def verify(a):
    p=GRID/'runs'/a['run']['run_id']
    r=json.loads((p/'report.json').read_text())
    prov=json.loads((p/'provenance.json').read_text())
    assert hashlib.sha256(json.dumps(prov,sort_keys=True).encode()).hexdigest()==r['provenance_sha256']
    a['issues']=[s for s in a['issues'] if s!='provenance checksum']
    if 'recomputed discovery scores' in a['issues']:
        hs={x['hypothesis']['id']:Hypothesis.model_validate(x['hypothesis']) for x in r['state']['registrations']}
        tests={}
        for e in r['state']['executions']:
            if e['result']['valid']:tests.setdefault(e['comparison_key'],e)
        recalculated=workflow_discovery([hs[h] for h in r['final_accepted_ids']],version_discoveries(pair,r['version']),tests,r['confirmation'],failed=bool(r['protocol_errors']),repaired=bool(r['attempt_errors'] or r['coordination']['draft_errors']))
        assert recalculated==r['scores']['discovery'],a['run']['run_id']
        a['issues'].remove('recomputed discovery scores')
    a['B_denominators']={c:{k:v[k] for k in ('denominator','numerator','accuracy')} for c,v in r['responsiveness']['components'].items()}
    for rel in a['native_warning_examples']:
        path=GRID/rel
        events=[json.loads(line) for line in path.read_text().splitlines()]
        final=[e['type'] for e in events if e.get('type') in {'turn.started','turn.failed','turn.completed','error'}][-1]
        if final!='turn.completed' or not (path.parent/'final.txt').is_file():
            a['issues'].append('warning/completion requires manual review:'+rel)
    return a

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    data['runs']=list(pool.map(verify,data['runs']))
data['audit_check_note']='Canonical provenance digest and peer-repair first-attempt penalties rechecked with the corrected audit assertions; experiment files unchanged.'
(OUT/'audit_results.json').write_text(json.dumps(data,indent=2))
print('Remaining discrepancies:',[(a['run']['run_id'],a['issues']) for a in data['runs'] if a['issues']])
