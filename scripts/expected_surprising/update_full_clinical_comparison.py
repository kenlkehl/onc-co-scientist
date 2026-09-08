"""Refresh the cross-batch report using only endpoints with all assigned runs finished."""
import ast
import json
import re
from pathlib import Path

from onc_co_scientist.expected_surprising.summary import paired_summary

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/expected_surprising_ledger/full_runs'
CODEX = BASE / '20260907T224450Z_codex'
BATCHES = [CODEX, BASE / '20260907T224450Z_vllm', BASE / '20260908T103000Z_terra']
OUT = BASE / 'full_clinical_comparison'


def read(path):
    return json.loads(path.read_text())


def absolute_links(text, base):
    return re.sub(r'\]\(([^)]+)\)', lambda m: '](' + (str((base / m[1]).resolve()) if not m[1].startswith(('/', 'https:', 'http:', '#')) else m[1]) + ')', text)


def main():
    OUT.mkdir(exist_ok=True)
    summaries, results, jobs, endpoints, sources, pending = {}, [], [], [], [], []
    for batch in BATCHES:
        if not (batch / 'manifest.json').exists():
            continue
        manifest = read(batch / 'manifest.json')
        finished = read(batch / 'summary.json') if (batch / 'summary.json').exists() else []
        existing = read(batch / 'paired_summaries.json') if (batch / 'paired_summaries.json').exists() else {}
        for endpoint in manifest['config']['endpoints']:
            label = endpoint['label']
            assigned = [j for j in manifest['jobs'] if j['endpoint'] == label]
            done = [r for r in finished if r['endpoint'] == label]
            if {r['run_id'] for r in done} != {j['run_id'] for j in assigned}:
                pending.append(f"{label}: {len(done)}/{len(assigned)} runs finished")
                continue
            paths = [batch / 'runs' / j['run_id'] / 'report.json' for j in assigned]
            if not all(p.exists() for p in paths):
                pending.append(f'{label}: finished with missing reports; review required')
                continue
            summary = existing.get(label)
            if summary is None:
                cache = OUT / f'{label}_paired_summary.json'
                if cache.exists():
                    summary = read(cache)
                else:
                    summary = paired_summary([read(p) for p in paths])
                    cache.write_text(json.dumps(summary, indent=2) + '\n')
            summaries[label] = summary
            results.extend(done)
            jobs.extend(assigned)
            endpoints.append(endpoint)
            sources.append({'label': label, 'batch': str(batch), 'manifest': str(batch / 'manifest.json'), 'execution': str(batch / 'execution.json')})
    assert len({j['run_id'] for j in jobs}) == len(jobs)
    manifest = {'jobs': jobs, 'config': {'endpoints': endpoints}, 'sources': sources}
    for filename, value in [('manifest.json', manifest), ('summary.json', results), ('paired_summaries.json', summaries), ('execution.json', {'source_batches': sources, 'pending': pending})]:
        (OUT / filename).write_text(json.dumps(value, indent=2) + '\n')
    diagnostic = CODEX / 'B_DIAGNOSTIC.md'
    if diagnostic.exists():
        (OUT / diagnostic.name).write_text(absolute_links(diagnostic.read_text(), CODEX))
    # Load only rendering code, avoiding provider initialization or run execution.
    script = ROOT / 'scripts/expected_surprising/run_replicates.py'
    tree = ast.parse(script.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'write_results')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(script), 'exec'), namespace)
    namespace['write_results'](OUT, manifest, summaries, results)
    report = (OUT / 'RESULTS.md').read_text()
    details = ['**Completion and model settings**', '', '| Model | Runs finished | All stages successful | Successful stages | Reasoning / requested tier |', '|---|---:|---:|---:|---|']
    for endpoint in endpoints:
        label, provider = endpoint['label'], endpoint['provider']
        done = [r for r in results if r['endpoint'] == label]
        settings = (provider.get('reasoning_effort', '') + ' / ' + provider.get('service_tier', 'default')) if provider['kind'] == 'codex_cli' else 'Server settings / vLLM'
        details.append(f"| {label} | {len(done)} | {sum(r.get('protocol_complete', False) for r in done)} | {sum(r.get('valid_stages', 0) for r in done)} / {sum(r.get('expected_stages', 100) for r in done)} | {settings} |")
    details += ['', 'Stage failures and their existing score penalties are retained. Finished does not mean every stage succeeded. Codex tiers shown are requested settings; Luna, Sol, and Astra requested Priority, while Terra requests normal (`default`) tier.']
    if pending:
        details += ['', 'Still pending; excluded from the score tables: ' + '; '.join(pending) + '.']
    details += ['', 'Source batches: ' + ' · '.join(f"[{s['label']}]({s['manifest']})" for s in sources), '']
    report = report.replace('Recall R is category-balanced', '\n'.join(details) + '\nRecall R is category-balanced', 1)
    (OUT / 'RESULTS.md').write_text(report)
    # Keep the previously shared report URL current, with unambiguous artifact links.
    (CODEX / 'RESULTS.md').write_text(absolute_links(report, OUT))
    (ROOT / 'benchmarks/expected_surprising/full_clinical_ledger_20260907/RESULTS.md').write_text(absolute_links(report, OUT))
    print(json.dumps({'report': str(OUT / 'RESULTS.md'), 'models': {k: {m: v.get(m) for m in ['E', 'B', 'D']} for k, v in summaries.items()}, 'pending': pending}))


if __name__ == '__main__':
    main()
