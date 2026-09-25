"""Run a fixed balanced subset through the unchanged frozen scientific harness."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import time
import traceback


def identity(row):
    return row['condition'], row['run_id']


def validate_selection(grid, selection):
    expected = [r for r in grid if r['replicate'] <= 3]
    if selection['target'] != expected:
        raise ValueError('Selection must preserve the exact frozen rows for repeats 1–3')
    if selection['core'] != [r for r in grid if r['replicate'] <= 2]:
        raise ValueError('Core must preserve the exact frozen rows for repeats 1–2')
    deferred = [r for r in grid if r['replicate'] > 3]
    if selection['deferred'] != deferred or len(expected) != 36 or len(deferred) != 84:
        raise ValueError('Require 36 selected and 84 deferred identities')
    if selection.get('execution_repeat_limit', 3) not in (2, 3):
        raise ValueError('Execution must retain a balanced two- or three-repeat cohort')
    return expected, deferred


def outside_execution_records(root, rows, driver):
    """Keep terminal outcomes visible when narrowing the execution cohort."""
    records = []
    for row in rows:
        folder = root / row['condition'] / 'runs' / row['run_id']
        prior = driver.read(folder / 'run.json', {})
        status = 'paused'
        if (prior.get('status') in ('completed', 'failed')
                and prior.get('scientific_report_sha256')):
            if driver.sha(folder / 'report.json') != prior['scientific_report_sha256']:
                raise ValueError('Outside-cohort terminal report changed: ' + row['run_id'])
            status = prior['status']
        records.append(dict(condition=row['condition'], run_id=row['run_id'], status=status,
            reason=('Terminal record retained outside the current execution cohort' if status != 'paused'
                    else 'Deferred for balanced September 24 preliminary cohort')))
    return records


def score_selected(root, selection, driver):
    """Use the frozen scorer; report explicit denominators for both nested cohorts."""
    from scripts.expected_surprising import build_presentation_results as scoring
    from scripts.expected_surprising.report_federation_initial30 import aggregate_sites, markdown_table
    out = root / 'analysis/deadline_20260924'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for plan in selection['target']:
        row = scoring.inspect_job((plan, 'unmasked' if plan['condition'] == 'named' else 'masked',
                                   root / plan['condition']))
        row.update(site_count=4, partition_id='random', harness='vLLM', replicate=plan['replicate'])
        row.update(input_tokens=None, successful_stages=None)
        for cls in scoring.CLASSES:
            row['response_' + cls + '_correct'] = None
            row['response_' + cls + '_eligible'] = None
        if row['report_available']:
            report = driver.read(Path(row['source']) / 'report.json')
            row['input_tokens'] = report['coordination']['usage'].get('input_tokens')
            row['successful_stages'] = report['successful_stages']
            for cls in scoring.CLASSES:
                component = report['responsiveness']['components'][cls]
                row['response_' + cls + '_correct'] = component['numerator']
                row['response_' + cls + '_eligible'] = component['denominator']
        rows.append(row)
    scoring.write_csv(out / 'run_metrics.csv', rows)
    lines = ['# Gemma federation: September 24 preliminary cohort', '',
             f'Generated {driver.now()}. Fixed selection: repeats 1–3 in all 12 cells (36 runs); '
             'nested core: repeats 1–2 (24 runs). All runs retain 25 iterations and four random sites.', '',
             'Saved FP8 responses are retained; subsequent responses use NVIDIA NVFP4 on GCP '
             'or sn4622130540. The local server uses vLLM 0.28.0; GCP uses 0.29.0. '
             'This is a mixed deployment continuation. Selection does not use scientific outcomes.', '',
             'Scores are 0–100. P: precision; R: recall; F1*: scientific F1; E: exploration; '
             'B: evidence responsiveness; Focal: focal recovery; C: weighted condition score.', '',
             '**Clean / planned** counts runs where all required stages succeeded. The clean-only tables '
             'score those runs; the all-terminal tables also score finished reports flagged for validation '
             'failures. A cell can therefore have scientific scores even when its clean count is zero. '
             'Missing evidence classes still leave affected metrics unavailable.']
    tables = {}
    handoff = driver.read(root / 'control/gcp-nvfp4/camus-handoff.json', {})
    if handoff.get('approved_by_user'):
        lines += ['', '**Camus continuation:** Remaining work moved to camus:8060 '
            '(RedHatAI Gemma 4 31B FP8 Dynamic, vLLM 0.27.1). Earlier FP8/NVFP4 responses '
            'are retained. Only the server and served model alias change; full prompts and sampling '
            'remain unchanged. Exact token IDs matched across servers on the checked long prompt. '
            'Per-attempt deployment and context-fit records identify the actual server, model, and wire request.']
    returned = driver.read(root / 'control/gcp-nvfp4/camus8060-return-handoff.json', {})
    if returned.get('approved_by_user'):
        lines += ['', '**Return to camus:8060:** On September 24 the user authorized '
                  'finishing the remaining work on the idle port8060 server. Existing '
                  'port8002 requests drain before saved-state recovery resumes on8060. '
                  'Both ports report the same model repository, vLLM version and context size; '
                  'long-prompt token counts match. Exact weight revisions remain unverified. '
                  'All saved responses are retained; attempt records identify the endpoint.']
    port_handoff = driver.read(root / 'control/gcp-nvfp4/camus8002-handoff.json', {})
    if port_handoff.get('approved_by_user'):
        lines += ['', '**Earlier Camus port migration:** Work moved to camus:8002 at the user\'s '
                  'request, with camus:8060 released after draining at that time. The replacement reports the same '
                  'model repository, vLLM version and context size; exact weight revisions remain '
                  'unverified. Saved answers and scientific inputs are retained. Per-attempt records '
                  'identify which endpoint served each response.']
    if selection.get('execution_repeat_limit', 3) == 2:
        lines += ['', '**Execution focus:** The predeclared 24-run core (repeats 1–2) is active '
                  'following the September 23 noon fallback. Existing third-repeat reports remain '
                  'included in the 36-run tables; unfinished third-repeat work is preserved and deferred.']
    for count, repeats in ((36, 3), (24, 2)):
        chosen = [r for r in rows if r['replicate'] <= repeats]
        for cohort in ('completed', 'finished'):
            groups, cells, cross = aggregate_sites(chosen, cohort)
            prefix = f'{count}_{cohort}'
            tables[prefix] = dict(groups=groups, conditions=cells, cross=cross)
            for name, data in (('grouped_metrics', groups), ('condition_metrics', cells),
                               ('cross_condition_scores', cross)):
                scoring.write_csv(out / f'{prefix}_{name}.csv', data)
            lookup = {(r['workflow'], r['condition'], r['metric']): r for r in cells}
            display = []
            for w in driver.WORKFLOWS:
                for c in driver.CONDITIONS:
                    cell = [r for r in chosen if r['workflow'] == w and r['version'] + '-' + r['view'] == c]
                    display.append([w, c, f"{sum(r['status'] == 'completed' for r in cell)}/{repeats}",
                        *[lookup.get((w, c, m), {}).get('value') for m in
                          ('precision', 'recall', 'f1', 'exploration', 'responsiveness', 'focal_recovery', 'condition_score')]])
            cohort_label = 'clean completions only' if cohort == 'completed' else 'all terminal reports'
            lines += ['', f'## {count}-run cohort: {cohort_label}', '',
                markdown_table(['Workflow', 'Condition', 'Clean / planned', 'P', 'R', 'F1*', 'E', 'B', 'Focal', 'C'], display), '',
                markdown_table(['Workflow', 'Metric', 'Value', 'Availability'],
                               [[r['workflow'], r['metric'], r['value'], r['value_status']] for r in cross])]
    progress = driver.read(root / 'progress.json', {})
    selected_sources = {r['source'] for r in rows}
    usage = Counter()
    for row in progress.get('runs', []):
        if row['source'] in selected_sources:
            usage.update({k: row.get(k, 0) for k in ('input_tokens', 'output_tokens', 'cached_input_tokens', 'responses')})
    lines += ['', f"Recorded selected-cohort receipts as of {progress.get('updated_at', 'unavailable')}: {dict(usage)}.", '',
        'C = 0.35 F1* + 0.25 Focal + 0.20 E + 0.20 B. S is the geometric mean of C across all four '
        'conditions. Missing conditions or evidence classes remain unavailable. CSV files include the usual '
        'whole-run bootstrap and Wilson intervals; three repeats provide limited precision. Unfinished '
        'runs remain in the reported completion denominators. Infrastructure costs are separate from token accounting.', '',
        f'[Run metrics]({out / "run_metrics.csv"}) · [All tables]({out / "metrics.json"}) · '
        f'[Live progress]({root / "CLOUD_LIVE_PROGRESS.md"})']
    if (out / 'VALIDATION_NOTES.md').exists():
        lines += ['', f'[Observed validation failures and interpretation]({out / "VALIDATION_NOTES.md"})']
    policy = driver.read(root / 'control/gcp-nvfp4/context-fit-policy.json', {})
    if policy.get('json_whitespace', {}).get('approved_by_user'):
        lines += ['', '**Approved JSON formatting continuation:** From September 24, new wire '
                  'requests remove only insignificant JSON whitespace. Every field, value, evidence '
                  'association, narrative string and ordering is retained. Nominal requests and '
                  'saved responses remain unchanged; per-request audits record both prompt hashes. '
                  'Results include this explicitly approved mixed-format continuation.']
    if policy.get('enabled') and policy.get('approved_by_user'):
        primary_floor = driver.read(root / 'transport_migration.json', {}).get('context_fit_policy', {}).get('min_output_tokens')
        floor_note = (f"Original workers retain their {primary_floor:,}-token minimum until handed to recovery. "
                      if primary_floor and primary_floor != policy['min_output_tokens'] else
                      'All current workers use this minimum. ')
        lines += ['', '**Approved output allowance amendment:** New requests are tokenized on their '
            'selected server. The output ceiling is min(65,536, 262,144 − input tokens − 512), '
            f"with a {policy['min_output_tokens']:,}-token minimum for current workers; requests below that floor pause. " +
            floor_note + 'Full prompts and sampling '
            'stay unchanged, truncated answers remain rejected, and every adjusted wire request is audited. '
            'Saved responses retain their original allowance. This is a mixed output-policy continuation.']
    driver.atomic_write_text(out / 'REPORT.md', '\n'.join(lines) + '\n')
    driver.atomic_write_json(out / 'metrics.json', dict(generated_at=driver.now(), runs=rows, **tables))
    driver.atomic_write_json(out / 'provenance.json', dict(
        frozen_manifest_sha256=driver.sha(root / 'frozen_manifest.json'),
        selection=selection, scorer_sha256=driver.sha(Path(scoring.__file__)), transport_handoff=handoff or None,
        report_hashes={r['source']: r['report_sha256'] for r in rows if r['report_available']}))


def terminal_status(root, row, driver):
    """Preserve completed and validation-failed reports across transport handoffs."""
    folder = root / row['condition'] / 'runs' / row['run_id']
    result = driver.read(folder / 'run.json', {})
    expected = result.get('scientific_report_sha256')
    if expected:
        if driver.sha(folder / 'report.json') != expected:
            raise ValueError('Terminal scientific report changed: ' + row['run_id'])
        if result.get('status') not in {'completed', 'failed'}:
            raise ValueError('Unexpected terminal scientific status: ' + row['run_id'])
        return result['status']
    return None


def run_selected(root, workers, selection, driver, pool):
    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import build_run_plans
    from onc_co_scientist.expected_surprising.experiment import run_cell
    from onc_co_scientist.expected_surprising.coordination import WorkflowInfrastructureError
    import onc_co_scientist.expected_surprising.federation as module
    if not Path(module.__file__).is_relative_to(root / 'source'):
        raise ValueError('Use frozen science on PYTHONPATH')
    lock = (root / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = driver.read(root / 'frozen_manifest.json')
    for name, expected in manifest['hashes'].items():
        if driver.sha(root / name) != expected:
            raise ValueError(f'Frozen file changed: {name}')
    grid = driver.read(root / 'grid.json')
    driver.verify_grid(grid, manifest['model'])
    selected, deferred = validate_selection(grid, selection)
    if selection.get('execution_repeat_limit', 3) == 2:
        selected = selection['core']
        deferred = [r for r in grid if r['replicate'] > 2]
    specs = {c: load_experiment_spec(root / c / 'config.yaml') for c in ('named', 'masked')}
    plans = {c: {p.run_id: p for p in build_run_plans(s)} for c, s in specs.items()}
    pending = list(selected)
    active, finished, cache = {}, [], {}
    for row in list(pending):
        status = terminal_status(root, row, driver)
        if status is not None:
            pending.remove(row)
            finished.append(dict(condition=row['condition'], run_id=row['run_id'], status=status))
    deferred_records = outside_execution_records(root, deferred, driver)
    state = dict(status='running', started_at=driver.now(), pid=os.getpid(), workers=workers,
                 selected_count=len(selected), core_count=24,
                 deferred_count=sum(r['status'] == 'paused' for r in deferred_records),
                 outside_execution_count=len(deferred),
                 terminal_outside_execution_count=sum(r['status'] != 'paused' for r in deferred_records),
                 deadline=selection['deadline'], report_snapshot_target=selection['report_snapshot_target'])

    def work(row):
        c, rid = identity(row)
        folder = root / c / 'runs' / rid
        for retry in range(5):
            try:
                return run_cell(specs[c], plans[c][rid], root / c, specs[c].fingerprint(), resume=True)
            except WorkflowInfrastructureError as exc:
                for path in (folder / 'calls').rglob('*.json'):
                    record = driver.read(path, {})
                    if record.get('result', {}).get('infrastructure_error'):
                        dest = folder / 'infrastructure_archive' / f'retry-{time.time_ns()}' / path.relative_to(folder)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        path.replace(dest)
                driver.atomic_write_json(folder / 'retry_status.json', dict(
                    status='retry_wait', retry=retry + 1, error=str(exc), updated_at=driver.now()))
                if 'CONTEXT_FIT_OUTPUT_FLOOR' in str(exc):
                    driver.atomic_write_json(folder / 'retry_status.json', dict(
                        status='paused_output_floor', retry=retry + 1, error=str(exc), updated_at=driver.now()))
                    raise  # Replaying the same oversized prompt cannot change this floor.
                if retry == 4:
                    raise
                time.sleep(min(60, 10 * 2 ** retry))

    def save():
        state.update(updated_at=driver.now(), active=list(active.values()), queued_runs=len(pending),
                     finished=finished + deferred_records, selected_finished=finished)
        driver.atomic_write_json(root / 'execution.json', state)
        driver.atomic_write_json(root / 'control/gcp-nvfp4/hybrid-runtime.json',
                                 dict(updated_at=driver.now(), **pool.snapshot()))

    # Slow historical receipt scans must never block admitting the next request,
    # collecting finished workers, or publishing accurate active/deferred counts.
    with ThreadPoolExecutor(max_workers=1) as reporting, ThreadPoolExecutor(max_workers=workers) as executor:
        report_future, last_refresh = None, 0.
        while pending or active:
            while pending and len(active) < workers and not (root / 'PAUSE').exists():
                row = pending.pop(0)
                active[executor.submit(work, row)] = row
            save()
            if report_future is None or (report_future.done() and time.monotonic() - last_refresh >= 300):
                if report_future is not None:
                    try:
                        report_future.result()
                    except Exception:
                        driver.atomic_write_json(root / 'control/gcp-nvfp4/refresh-error.json',
                            dict(updated_at=driver.now(), error=traceback.format_exc()))
                report_future = reporting.submit(driver.refresh, root, cache)
                last_refresh = time.monotonic()
            if not active:
                break
            done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                row = active.pop(future)
                try:
                    status = future.result()['status']
                except Exception as exc:
                    status = 'interrupted'
                    driver.atomic_write_json(root / row['condition'] / 'runs' / row['run_id'] / 'worker_error.json',
                        dict(error=str(exc), traceback=traceback.format_exc(), updated_at=driver.now()))
                finished.append(dict(condition=row['condition'], run_id=row['run_id'], status=status))
        state['status'] = 'paused' if pending else 'scoring'
        save()
    if not pending:
        try:
            driver.refresh(root, cache)
            score_selected(root, selection, driver)
            state['status'] = 'finished_selected_cohort'
        except Exception:
            state.update(status='scoring_error', scoring_error=traceback.format_exc())
        state['ended_at'] = driver.now()
        save()


if __name__ == '__main__':
    # Read-only scoring entry point for the preliminary deadline while workers
    # continue; report hashes record exactly which terminal artifacts were scored.
    import argparse
    import importlib.util
    parser = argparse.ArgumentParser(description='Score the fixed preliminary Gemma cohort')
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / 'source/scripts/expected_surprising/gemini_federation_grid.py'
    spec = importlib.util.spec_from_file_location('frozen_deadline_scorer', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    manifest = driver.read(root / 'frozen_manifest.json')
    for name, expected in manifest['hashes'].items():
        if driver.sha(root / name) != expected:
            raise ValueError(f'Frozen file changed: {name}')
    selection = driver.read(root / 'control/gcp-nvfp4/selection.json')
    validate_selection(driver.read(root / 'grid.json'), selection)
    score_selected(root, selection, driver)
    print(root / 'analysis/deadline_20260924/REPORT.md')
