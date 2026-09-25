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
                   E=rep['scores']['E'], response_components=rep['responsiveness']['components'],
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
        cell['E'] = sum(r['E'] for r in selected) / 10
        cell['response_components'] = {}
        for cls in ('supported', 'excluded', 'ambiguous'):
            values = [r['response_components'][cls] for r in selected]
            available = [v for v in values if v['accuracy'] is not None]
            cell['response_components'][cls] = dict(
                mean_accuracy=sum(v['accuracy'] for v in available)/len(available) if available else None,
                eligible_runs=len(available), events=sum(v['denominator'] for v in values))
        accuracies = [v['mean_accuracy'] for v in cell['response_components'].values()]
        cell['B'] = 100*sum(accuracies)/3 if all(v is not None for v in accuracies) else None
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
    'Discovery scores and run counts match RESULTS.json and all 40 individual scientific reports; report hashes verified. All assigned repeats remain in D/R/E denominators, including any partially failed runs. P averages runs with accepted claims; B averages eligible runs within each evidence class, then gives the three classes equal weight.',
    'Known input/output tokens include saved calls when a run aggregate is null. Any known output tokens omitted by the generated summary are quantified separately, without changing scientific scores or frozen artifacts.',
    'Provider-reported known tokens do not establish complete physical usage: unobserved failed SDK retry attempts can have unavailable usage. Explicit interruption and unknown-usage counts are reported separately.',
    'Repeats measure model variability on one dataset pair, not generalization across datasets.',
    'Original 16-outcome frozen prompts retain the masked row-identifier exposure, missing masked assay descriptions, and relative-versus-absolute threshold mismatch documented in the masking audit. The composite campaign uses the corrected protocol.'
]
interrupted = (70 if phase == 'persistent' else 40 if phase == 'sequential' else 0) if root.name == 'nsclc_depmap_qwen38_flash_next_20260923' else 0
notes.append(f'Known interrupted unfinished requests for this campaign/phase: {interrupted}; their partial usage is unavailable. Saved call records with unavailable usage: {sum(c["unknown_usage_call_records"] for c in cells)}.')
payload = dict(verified_at=datetime.now(timezone.utc).isoformat(), workflow=phase, cells=cells, runs=rows,
               original_results_sha256=hashlib.sha256((out/'RESULTS.json').read_bytes()).hexdigest(),
               original_results_md_sha256=hashlib.sha256((out/'RESULTS.md').read_bytes()).hexdigest(), notes=notes)
(out / 'VERIFIED_RESULTS.json').write_text(json.dumps(payload, indent=2)+'\n')
label = 'composite-outcome' if 'single_outcome' in root.name else '16-outcome'
lines = [f'# Verified {phase} results: {label} NSCLC DepMap', '',
         '| Version | Masking | Exact D | Exact R | P | E | B | Focal / 10 | Complete | Failed | Known output tokens |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
def fmt(value):
    return 'unavailable' if value is None else f'{value:.4f}'
for c in cells:
    lines.append(f"| {c['version']} | {c['masking']} | {c['mean_exact_D']:.3f} | {fmt(c['mean_exact_R'])} | {fmt(c['mean_Q'])} | {fmt(c['E'])} | {fmt(c['B'])} | {c['focal_recovery']} | {c['completed']} | {c['failed']} | {c['known_output_tokens']:,} |")
lines += ['', 'D (F1*) is on a 0–100 scale. R is category-balanced exact recovery/recall. Q is the machine-readable name for P, the fraction of final accepted claims confirmed on fresh evaluator data. R and Q are on a 0–1 scale. D and R average all ten assigned repeats; Q averages only runs with accepted claims, and its denominator is recorded as Q_defined_runs in VERIFIED_RESULTS.json.', '']
lines += ['E is exact-target exploration coverage over the full iteration budget, rewarding earlier testing. B is evidence responsiveness, balanced across supported, excluded, and ambiguous evidence. Both use a 0–100 scale. B evidence and eligible-run counts are recorded in the JSON; it is conditional on evidence encountered.', '']
lines += ['- '+n for n in notes]
lines += ['', 'Execution and resource totals:', '']
for key in ('calls', 'successful_stages', 'expected_stages', 'attempts_with_errors', 'repaired_stages', 'exhausted_stages', 'degraded_stages', 'known_input_tokens', 'known_output_tokens'):
    lines.append(f"- {key}: {sum(c[key] for c in cells):,}")
(out / 'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('VERIFIED_SUMMARY', json.dumps({'cells': cells, 'notes': notes}), flush=True)
