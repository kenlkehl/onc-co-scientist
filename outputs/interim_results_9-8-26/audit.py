"""Read-only audit of a fixed terminal-run snapshot; never runs a provider."""
import collections
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/data1/ken/onc-co-scientist')
GRID = ROOT / 'data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows'
OUT = ROOT / 'outputs/interim_results_9-8-26'
sys.path.insert(0, str(GRID / 'source/src'))
from onc_co_scientist.expected_surprising.schemas import PairSpec, Hypothesis
from onc_co_scientist.expected_surprising.generation import version_discoveries
from onc_co_scientist.expected_surprising.scoring import workflow_discovery, exploration_coverage
from onc_co_scientist.expected_surprising.events import response_summary
from onc_co_scientist.expected_surprising.summary import _hierarchy

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

snapshot_path = OUT / 'snapshot.json'
if not snapshot_path.exists():
    plans = read(GRID / 'plan.json')
    runs = [read(p) for p in sorted((GRID / 'runs').glob('*/run.json'))]
    snapshot_path.write_text(json.dumps({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'plans': plans, 'runs': runs}, indent=2))
snapshot = read(snapshot_path)
plans = {p['run_id']: p for p in snapshot['plans']}
manifest = read(GRID / 'frozen_manifest.json')
hashes = {**manifest['source_hashes'], **manifest['input_hashes'], 'config.yaml': manifest['config_sha256']}
hash_errors = [p for p, expected in hashes.items() if sha(GRID / p) != expected]
pair = PairSpec.model_validate_json((GRID / 'input_data/private/es-v2-nsclc_clinical-42000/pair.json').read_text())

def audit(run):
    rid = run['run_id']
    folder = GRID / 'runs' / rid
    report = read(folder / 'report.json')
    issues = []
    def check(ok, label):
        if not ok:
            issues.append(label)
    check(sha(folder / 'report.json') == run['scientific_report_sha256'], 'report checksum')
    check(hashlib.sha256(json.dumps(read(folder / 'provenance.json'), sort_keys=True).encode()).hexdigest() == report['provenance_sha256'], 'provenance checksum')
    for key in ('model_profile', 'workflow_id', 'model_id', 'replicate', 'semantic_condition'):
        check(run[key] == plans[rid][key], 'plan identity:' + key)
    check(report['dataset_sha256'] == manifest['input_hashes'][f"input_data/public/{run['task_id']}/dataset.parquet"], 'dataset hash')
    check(report['iterations'] == 25 and report['expected_stages'] == 100, 'iteration budget')
    check(report['policy']['response_window'] == 2 and report['policy']['analyses_per_iteration'] == 12, 'policy')
    stages = report['state']['completed_stages']
    committed = report['coordination']['committed_stages']
    check([f"i{s['iteration']:03d}-{s['stage']}" for s in stages] == committed, 'committed stages')
    check(len(set(committed)) == len(committed) == report['successful_stages'], 'unique stages')
    check((run['status'] == 'completed') == (not report['protocol_errors']), 'completion status')
    if run['status'] == 'completed':
        check(run['iterations_completed'] == 25 and len(stages) == 100, 'completed full budget')
    transcript = [json.loads(line) for line in (folder / 'science-0001/transcript.jsonl').read_text().splitlines()]
    check(sum(e['kind'] == 'stage' for e in transcript) == len(stages), 'transcript stages')
    hypotheses = {x['hypothesis']['id']: Hypothesis.model_validate(x['hypothesis']) for x in report['state']['registrations']}
    accepted = [hypotheses[h] for h in report['final_accepted_ids']]
    tests = {}
    for e in report['state']['executions']:
        if e['result']['valid']:
            tests.setdefault(e['comparison_key'], e)
    discoveries = version_discoveries(pair, report['version'])
    recalc = workflow_discovery(accepted, discoveries, tests, report['confirmation'], failed=bool(report['protocol_errors']), repaired=bool(report['attempt_errors'] or report['coordination']['draft_errors']))
    check(recalc == report['scores']['discovery'], 'recomputed discovery scores')
    coverage = exploration_coverage(list(tests.values()), discoveries, 25)
    check(coverage == report['scores']['coverage'], 'recomputed coverage')
    responses = response_summary(report['state']['events'], stages, 25, discoveries)
    check(responses == report['responsiveness'], 'recomputed response-to-evidence')
    focal = int(any(m['discovery_id'] == pair.focal_id and m['match'] == 'exact' for m in recalc['confirmed_matches']))
    check(report['confirmation']['primary_recovery'] == (0 if report['protocol_errors'] else focal), 'focal recovery penalty')
    check(report['scores']['D'] == recalc['exact']['D'] and report['scores']['E'] == coverage['exact']['E'] and report['scores']['B'] == responses['B'], 'top-level scores')
    token = collections.Counter()
    history_counts = []
    call_kinds = collections.Counter()
    errors = []
    records = []
    for path, expected in report['coordination']['participant_artifact_sha256'].items():
        raw = (folder / path).read_bytes()
        check(hashlib.sha256(raw).hexdigest() == expected, 'call checksum:' + path)
        c = json.loads(raw)
        req, res = c['request'], c['result']
        records.append(c)
        check(req['model'] == run['model_id'] and req['max_tokens'] == 125000, 'request settings:' + path)
        check(req['authoritative_candidate'] == (req['kind'] != 'peer'), 'peer authority:' + path)
        call_kinds[req['kind']] += 1
        history_counts.append(len(req['messages']))
        usage = res.get('usage') or {}
        n = usage.get('output_tokens')
        token['original_known'] += n or 0
        token['original_missing_calls'] += n is None
        token['provider_errors'] += bool(res.get('error'))
        if res.get('error'):
            errors.append({'slot': req['slot'], 'error': res['error']})
    check(len(records) == run['agent_calls'] == report['coordination']['agent_calls'], 'call count')
    check(token['original_known'] == report['coordination']['output_token_accounting']['known_output_tokens'], 'original token sum')
    check(token['original_missing_calls'] == report['coordination']['output_token_accounting']['missing_calls'], 'missing token count')
    if run['workflow_id'] == 'deliberative':
        check(call_kinds['peer'] >= 200 and call_kinds['chair'] >= len(stages), 'deliberative participants')
    else:
        check(call_kinds['peer'] == call_kinds['chair'] == 0, 'single-agent participants')
    native_files = sorted((folder / 'provider_audit').glob('call-*/attempt-*/events.jsonl'))
    native_examples = []
    if '_medium' in run['model_profile']:
        native_calls = {p.parent.parent.name for p in native_files}
        check(len(native_calls) == len(records), 'native call count')
        for path in native_files:
            events = [json.loads(line) for line in path.read_text().splitlines()]
            completions = [e for e in events if e.get('type') == 'turn.completed']
            warnings = [e for e in events if e.get('type') == 'error']
            for e in completions:
                n = (e.get('usage') or {}).get('output_tokens')
                token['reconciled_observed'] += n or 0
                token['native_completion_missing_usage'] += n is None
            token['native_completions'] += len(completions)
            if not completions:
                token['native_attempts_without_completion'] += 1
                token['capacity_attempts'] += 'at capacity' in path.read_text().lower()
            if completions and warnings:
                token['completed_with_warning'] += 1
                native_examples.append(str(path.relative_to(GRID)))
            cmd = read(path.parent / 'command.json')
            check('model_reasoning_effort="medium"' in cmd and 'service_tier="default"' in cmd, 'native reasoning/tier')
        token['tokens_recovered'] = token['reconciled_observed'] - token['original_known']
        check(token['tokens_recovered'] >= 0, 'native token reconciliation')
    else:
        token['reconciled_observed'] = token['original_known']
    per_iter = collections.Counter(e['iteration'] for e in report['state']['executions'])
    check(max(per_iter.values(), default=0) <= 12, 'analysis budget')
    return {'run': run, 'report': report, 'issues': issues, 'tokens': dict(token), 'history_range': [min(history_counts), max(history_counts)], 'call_kinds': dict(call_kinds), 'analyses': len(report['state']['executions']), 'valid_tests': len(tests), 'accepted': len(accepted), 'confirmed': recalc['confirmed_tested_n'], 'errors': errors, 'native_warning_examples': native_examples}

audits = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for a in pool.map(audit, snapshot['runs']):
        audits.append(a)
        print(f"audited {len(audits)}/{len(snapshot['runs'])}: {a['run']['run_id']} ({len(a['issues'])} issues)", flush=True)

def summarize(rows):
    reports = [a['report'] for a in rows]
    if not reports:
        return {'n': 0}
    versions = collections.Counter(r['version'] for r in reports)
    components = {c: _hierarchy(reports, lambda r: r['responsiveness']['components'][c]['accuracy']) for c in ('supported', 'excluded', 'ambiguous')}
    # A partially observed condition is explicitly descriptive. No incomplete pair is invented.
    result = {'n': len(rows), 'versions': dict(versions), 'paired_repeats': len({a['run']['replicate'] for a in rows if a['report']['version']=='expected'} & {a['run']['replicate'] for a in rows if a['report']['version']=='surprising'}), 'failed': sum(a['run']['status']=='failed' for a in rows), 'R': _hierarchy(reports, lambda r:r['scores']['discovery']['exact']['R']), 'P': _hierarchy(reports, lambda r:r['scores']['discovery']['exact']['Q']), 'F1': _hierarchy(reports, lambda r:r['scores']['D']), 'E': _hierarchy(reports, lambda r:r['scores']['E']), 'B': None if None in components.values() else 100*sum(components.values())/3, 'B_components': components, 'P_available_n': sum(r['scores']['discovery']['exact']['Q'] is not None for r in reports), 'focal': {v: {'n': versions[v], 'recovered': sum(r['confirmation']['primary_recovery'] for r in reports if r['version']==v)} for v in ('expected','surprising')}, 'tokens': dict(sum((collections.Counter(a['tokens']) for a in rows), collections.Counter())), 'calls':sum(a['run']['agent_calls'] for a in rows), 'successful_stages':sum(r['successful_stages'] for r in reports), 'memory_trims':sum(len(r['coordination']['memory_trims']) for r in reports)}
    if len(versions) < 2:
        result['single_version_only'] = True
    return result

conditions = []
for model in ('luna_medium','terra_medium','sol_medium','astra_medium','qwen_3_8_27b','gemma_4_31b'):
    for workflow in ('persistent','sequential','deliberative'):
        rows = [a for a in audits if a['run']['model_profile']==model and a['run']['workflow_id']==workflow]
        conditions.append({'model':model, 'workflow':workflow, 'all_terminal':summarize(rows), 'completed_only':summarize([a for a in rows if a['run']['status']=='completed'])})

result = {'snapshot_at':snapshot['at'], 'frozen_files_checked':len(hashes), 'hash_errors':hash_errors, 'conditions':conditions, 'runs':[{k:v for k,v in a.items() if k!='report'} for a in audits]}
(OUT/'audit_results.json').write_text(json.dumps(result,indent=2))
print('SAVED', OUT/'audit_results.json', flush=True)
