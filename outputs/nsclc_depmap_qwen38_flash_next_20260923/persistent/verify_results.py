from pathlib import Path
from datetime import datetime, timezone
import collections
import hashlib
import json
import math
import sys

root = Path(sys.argv[1]).resolve()
phase = sys.argv[2]
out = root / phase
generated = json.loads((out / 'RESULTS.json').read_text())
rows = []
cells = []
for condition in ('unmasked', 'masked'):
    plans = json.loads((out / condition / 'plan.json').read_text())
    assert len(plans) == 20
    for plan in plans:
        run_dir = out / condition / 'runs' / plan['run_id']
        run = json.loads((run_dir / 'run.json').read_text())
        report_path = Path(run['scientific_report'])
        report_bytes = report_path.read_bytes()
        assert hashlib.sha256(report_bytes).hexdigest() == run['scientific_report_sha256']
        rep = json.loads(report_bytes)
        assert rep['run_id'] == run['run_id'] == plan['run_id']
        assert run['primary_recovery'] == rep['confirmation']['primary_recovery']
        calls = list((run_dir / 'calls').glob('*.json'))
        assert len(calls) == run['agent_calls']
        usage = run['usage']
        known = {k: usage.get(k) for k in ('input_tokens', 'output_tokens')}
        incomplete = any(v is None for v in known.values())
        unknown_records = 0
        if incomplete:
            known = {k: 0 for k in known}
            for p in calls:
                result = json.loads(p.read_text())['result']
                u = result.get('usage') or {}
                unknown_records += int(any(u.get(k) is None for k in known))
                for k in known:
                    known[k] += u.get(k) or 0
        row = dict(run_id=run['run_id'], version=run['semantic_condition'], masking=condition,
                   status=run['status'], exact=rep['scores']['discovery']['exact'],
                   focal_recovery=run['primary_recovery'], successful_stages=rep['successful_stages'],
                   expected_stages=rep['expected_stages'], attempts_with_errors=len(rep['attempt_errors']),
                   repaired_stages=len(rep['recovered_stages']), exhausted_stages=len(rep['protocol_errors']),
                   degraded_stages=len(run.get('degraded_stages', [])), calls=len(calls),
                   known_usage=known, run_usage_incomplete=incomplete, unknown_usage_call_records=unknown_records,
                   report_sha256=run['scientific_report_sha256'])
        rows.append(row)
    for version in ('expected', 'surprising'):
        selected = [r for r in rows if r['masking'] == condition and r['version'] == version]
        assert len(selected) == 10
        cell = dict(version=version, masking=condition, assigned=10,
                    completed=sum(r['status'] == 'completed' for r in selected),
                    failed=sum(r['status'] == 'failed' for r in selected),
                    focal_recovery=sum(r['focal_recovery'] for r in selected),
                    mean_exact_D=sum(r['exact']['D'] for r in selected)/10,
                    mean_exact_R=sum(r['exact']['R'] for r in selected)/10,
                    mean_Q=(sum(r['exact']['Q'] for r in selected if r['exact']['Q'] is not None)
                            / sum(r['exact']['Q'] is not None for r in selected))
                    if any(r['exact']['Q'] is not None for r in selected) else None,
                    Q_defined_runs=sum(r['exact']['Q'] is not None for r in selected))
        for key in ('successful_stages', 'expected_stages', 'attempts_with_errors', 'repaired_stages', 'exhausted_stages', 'degraded_stages', 'calls', 'unknown_usage_call_records'):
            cell[key] = sum(r[key] for r in selected)
        for key in ('input_tokens', 'output_tokens'):
            cell['known_' + key] = sum(r['known_usage'][key] for r in selected)
        old = next(c for c in generated['cells'] if c['masking'] == condition and c['version'] == version)
        for key in ('assigned', 'completed', 'failed', 'focal_recovery', 'calls', 'mean_exact_D'):
            assert math.isclose(old[key], cell[key], rel_tol=1e-12, abs_tol=1e-12), (key, old[key], cell[key])
        cell['generated_summary_known_output_tokens'] = old['known_output_tokens']
        cell['known_output_tokens_omitted_by_generated_summary'] = cell['known_output_tokens'] - old['known_output_tokens']
        progress = json.loads((out / 'progress.json').read_text())
        original_progress = next(c for c in progress['cells'] if c['masking'] == condition and c['version'] == version)
        assert original_progress['known_output_tokens'] == cell['known_output_tokens']
        cells.append(cell)
        print('VERIFIED_CELL', json.dumps(cell), flush=True)
notes = [
    'Scores and counts match the original RESULTS.json and all 40 individual scientific reports; report hashes verified. All assigned repeats remain in denominators, including the partially failed run.',
    'Known input/output tokens include saved calls from a run whose aggregate usage is null because one timeout has unknown usage. The original generated summary treats that null aggregate as zero and undercounts known output tokens; the corrected known totals are reported here without changing scientific scores or frozen artifacts.',
    'Known tokens are not complete physical usage: 70 interrupted unfinished requests, one recorded timeout, and any failed physical SDK retry attempts have unavailable usage.',
    'Repeats measure model variability on one dataset pair, not generalization across datasets.',
    'Original 16-outcome frozen prompts retain the masked row-identifier exposure, missing masked assay descriptions, and relative-versus-absolute threshold mismatch documented in the masking audit. The composite campaign uses the corrected protocol.'
]
payload = dict(verified_at=datetime.now(timezone.utc).isoformat(), workflow=phase, cells=cells, runs=rows,
               original_results_sha256=hashlib.sha256((out/'RESULTS.json').read_bytes()).hexdigest(),
               original_results_md_sha256=hashlib.sha256((out/'RESULTS.md').read_bytes()).hexdigest(), notes=notes)
(out / 'VERIFIED_RESULTS.json').write_text(json.dumps(payload, indent=2)+'\n')
lines = ['# Verified persistent results: 16-outcome NSCLC DepMap', '',
         '| Version | Masking | Exact D | Exact R | Q | Focal / 10 | Complete | Failed | Known output tokens |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for c in cells:
    lines.append(f"| {c['version']} | {c['masking']} | {c['mean_exact_D']:.3f} | {c['mean_exact_R']:.4f} | {c['mean_Q']:.4f} | {c['focal_recovery']} | {c['completed']} | {c['failed']} | {c['known_output_tokens']:,} |")
lines += ['', 'D (F1*) is on a 0–100 scale. R is category-balanced exact recovery/recall. Q is the machine-readable name for P, the fraction of final accepted claims confirmed on fresh evaluator data. R and Q are on a 0–1 scale. D and R average all ten assigned repeats; Q averages only runs with accepted claims, and its denominator is recorded as Q_defined_runs in VERIFIED_RESULTS.json.', '']
lines += ['- '+n for n in notes]
lines += ['', 'Execution and resource totals:', '']
for key in ('calls', 'successful_stages', 'expected_stages', 'attempts_with_errors', 'repaired_stages', 'exhausted_stages', 'degraded_stages', 'known_input_tokens', 'known_output_tokens'):
    lines.append(f"- {key}: {sum(c[key] for c in cells):,}")
(out / 'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('VERIFIED_SUMMARY', json.dumps({'cells': cells, 'notes': notes}), flush=True)
