"""Supervise corrected masked vLLM jobs and consolidate their unchanged Codex siblings."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime

from onc_co_scientist.harness.durable_io import atomic_write_json, atomic_write_text
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans
from onc_co_scientist.expected_surprising.experiment_report import write_report


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def verified_signal(pid, expected_arg):
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    if str(expected_arg).encode() not in args:
        raise RuntimeError(f'Unexpected process identity for {pid}')
    os.kill(pid, signal.SIGTERM)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    handoff = read(root / 'handoff.json')
    parent = Path(handoff['codex_root'])
    frozen = read(root / 'frozen_manifest.json')
    for name, checksum in {**frozen['source_hashes'], **frozen['input_hashes'],
                           'config.yaml': frozen['config_sha256'],
                           'selection.json': frozen['selection_sha256'],
                           'handoff.json': frozen['handoff_sha256']}.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == checksum, name
    original_state = read(parent / 'execution.json')
    assert original_state and not original_state['vllm_released']
    old_gate = Path(original_state['wait_for_vllm_control'])
    assert not read(old_gate).get('completed_at'), 'Old driver gate must remain closed'
    assert not any((parent / 'runs').glob('*__qwen_3_8_27b__*'))
    assert not any((parent / 'runs').glob('*__gemma_4_31b__*'))
    control = root / 'control'
    control.mkdir(exist_ok=True)
    command = [sys.executable, str(root / 'source/run_selected_cells.py'), '--root', str(root),
               '--selection', str(root / 'selection.json'), '--control', str(control),
               '--workers', str(handoff['workers'])]
    if (control / 'execution.json').exists():
        raise FileExistsError('Existing child execution; inspect before restarting supervisor')
    with (control / 'driver.log').open('ab') as log:
        child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state = dict(started_at=now(), pid=os.getpid(), driver_pid=child.pid, command=command,
                 codex_root=str(parent), vllm_root=str(root), workers=handoff['workers'])
    atomic_write_json(root / 'launch.json', state)
    parent_spec = load_experiment_spec(parent / 'config.yaml')
    plans_by_id = {p.run_id: p for p in build_run_plans(parent_spec)}
    rows = read(parent / 'plan.json')
    plans = [plans_by_id[r['run_id']] for r in rows]
    vllm_ids = set(read(root / 'selection.json'))
    assert len(vllm_ids) == 120 and len(plans) == 360
    # Reports and journals are never merged; each identity has one authoritative root.
    ownership = {p.run_id: str(root if p.run_id in vllm_ids else parent) for p in plans}
    atomic_write_json(root / 'selected_run_roots.json', ownership)
    known_calls = set()
    tokens = collections.Counter()
    calls = collections.Counter()
    while True:
        results, groups = [], collections.Counter()
        codex_terminal = 0
        for plan in plans:
            run_root = Path(ownership[plan.run_id]) / 'runs' / plan.run_id
            result = read(run_root / 'run.json')
            if result is not None and result.get('status') in {'completed', 'failed'}:
                results.append(result)
                status = result['status']
                codex_terminal += plan.run_id not in vllm_ids
            else:
                status = 'active' if run_root.exists() else 'queued'
            key = (plan.model.id, plan.workflow.id)
            groups[(*key, status)] += 1
            for path in (run_root / 'calls').glob('*.json'):
                if path in known_calls:
                    continue
                record = read(path)
                if not record or 'result' not in record:
                    continue
                known_calls.add(path)
                calls[key] += 1
                tokens[key] += (record['result'].get('usage') or {}).get('output_tokens') or 0
        lines = ['# Masked grid: combined progress', '', f'Updated {now()}', '',
                 '360 identities: original frozen Codex runs plus 120 corrected Qwen/Gemma runs.',
                 'Qwen/Gemma run alongside the repaired unmasked experiment, with six additional workers. Codex execution is unchanged.', '',
                 '| Model | Workflow | Completed | Failed | Active | Queued | Calls | Known output tokens |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        for key in sorted({(p.model.id, p.workflow.id) for p in plans}):
            nums = [groups[(*key, status)] for status in ('completed', 'failed', 'active', 'queued')]
            lines.append(f'| {key[0]} | {key[1]} | ' + ' | '.join(map(str, [*nums, calls[key], tokens[key]])) + ' |')
        lines += ['', 'Known token totals exclude missing usage and separate smoke tests.',
                  '[Handoff](handoff.json) · [Corrected vLLM execution](control/execution.json) · [Final results](RESULTS.md)']
        text = '\n'.join(lines) + '\n'
        atomic_write_text(root / 'LIVE_PROGRESS.md', text)
        atomic_write_text(parent / 'COMBINED_PROGRESS.md', text)
        state.update(updated_at=now(), terminal_runs=len(results), codex_terminal=codex_terminal,
                     driver_exit_code=child.poll())
        # Retire the old scheduler only after all Codex work has finished and its
        # only remaining work is the never-released vLLM queue owned here.
        current = read(parent / 'execution.json')
        if (codex_terminal == 240 and current and current.get('active_runs') == 0
                and not current['vllm_released'] and not state.get('old_scheduler_retired_at')):
            verified_signal(current['pid'], parent / 'source/run_masked_grid.py')
            old_launch = read(parent / 'launch.json')
            if old_launch and old_launch.get('monitor_pid'):
                verified_signal(old_launch['monitor_pid'], parent / 'source/monitor_workflow_grid.py')
            state['old_scheduler_retired_at'] = now()
            atomic_write_json(parent / 'scheduler_handoff_complete.json', state)
        if len(results) == 360:
            write_report(parent_spec, plans, results, root)
            report = ('# Masked comparison with corrected vLLM execution\n\n'
                      'Codex uses the original masked snapshot; Qwen/Gemma use the September 10 parser and retry fixes. '
                      'Only Qwen disables thinking on its final retry. Concurrency differs during the overlap.\n\n'
                      + (root / 'expected_surprising_report.md').read_text())
            atomic_write_text(root / 'RESULTS.md', report)
            atomic_write_text(parent / 'RESULTS.md', report)
            summary = dict(n_runs=360, n_completed=sum(r['status'] == 'completed' for r in results),
                           n_failed=sum(r['status'] == 'failed' for r in results), runs=results,
                           selected_run_roots=ownership)
            atomic_write_json(root / 'combined_summary.json', summary)
            state['completed_at'] = now()
            atomic_write_json(root / 'supervisor.json', state)
            break
        if child.poll() not in (None, 0):
            state['driver_failure'] = 'Corrected vLLM driver exited unexpectedly; inspect control/driver.log'
        atomic_write_json(root / 'supervisor.json', state)
        time.sleep(30)


if __name__ == '__main__':
    main()
