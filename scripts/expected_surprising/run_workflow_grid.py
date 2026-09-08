"""Execute a frozen clinical workflow matrix with durable progress and final report."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import threading
import traceback

from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import run_experiment
from onc_co_scientist.harness.durable_io import atomic_write_json, atomic_write_text


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    frozen = json.loads((out / 'frozen_manifest.json').read_text())
    for path, expected in {**frozen['source_hashes'], **frozen['input_hashes'], 'config.yaml': frozen['config_sha256']}.items():
        assert hashlib.sha256((out / path).read_bytes()).hexdigest() == expected, path
    spec = load_experiment_spec(out / 'config.yaml')
    if args.dry_run:
        print(json.dumps(run_experiment(spec, dry_run=True)))
        return
    if (out / 'execution.json').exists():
        raise FileExistsError('This matrix has already been launched')
    import onc_co_scientist.expected_surprising.coordination as coordinator
    assert Path(coordinator.__file__).is_relative_to(out / 'source')
    execution = {'started_at': now(), 'pid': os.getpid(), 'source': coordinator.__file__, 'assigned_runs': 360}
    atomic_write_json(out / 'execution.json', execution)
    stop = threading.Event()

    def status():
        rows = collections.Counter()
        total_known = 0
        for path in (out / 'runs').glob('*/run.json'):
            try:
                run = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            rows[(run.get('model_profile', run.get('model_id', '?')), run.get('workflow_id', '?'), run.get('status', '?'))] += 1
            total_known += (run.get('usage') or {}).get('output_tokens') or 0
        text = ['# Clinical workflow grid progress', '', f'Updated {now()} · driver {os.getpid()}', '',
                '360 assigned runs: six models × three workflows × two paired versions × ten repeats. 25 iterations; 30 concurrent runs overall.', '',
                '| Model profile | Workflow | State | Runs |', '|---|---|---|---:|']
        for (model, workflow, state), count in sorted(rows.items()):
            text.append(f'| {model} | {workflow} | {state} | {count} |')
        text += ['', f'Known output tokens from terminal run summaries so far: {total_known:,}. Active-call usage remains in calls/*.json; this is not a complete total.', '',
                 '[Plan](plan.json) · [Configuration](config.yaml) · [Execution](execution.json) · [Final report, when ready](RESULTS.md)']
        atomic_write_text(out / 'STATUS.md', '\n'.join(text)+'\n')

    def watch():
        while not stop.wait(30):
            try:
                status()
            except Exception as error:
                print('Progress refresh:', repr(error), flush=True)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        status()
        summary = run_experiment(spec)
        report = (out / 'expected_surprising_report.md').read_text()
        report += '\n[Detailed scores and token totals](expected_surprising_summary.json) · [Run results](summary.json) · [Frozen grid](plan.json) · [Settings](config.yaml)\n'
        atomic_write_text(out / 'RESULTS.md', report)
        execution.update(completed_at=now(), status=summary['status'], n_runs=summary['n_runs'], n_completed=summary['n_completed'], n_failed=summary['n_failed'])
        atomic_write_json(out / 'execution.json', execution)
        print(json.dumps(execution), flush=True)
    except BaseException as error:
        execution.update(failed_at=now(), error=repr(error), traceback=traceback.format_exc())
        atomic_write_json(out / 'execution.json', execution)
        raise
    finally:
        stop.set()
        watcher.join()
        status()


if __name__ == '__main__':
    main()
