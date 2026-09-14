"""Run both semantic grids with one global pool and durable status."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, UTC
import fcntl
import hashlib
import json
import os
from pathlib import Path
import traceback

from onc_co_scientist.harness.durable_io import atomic_write_json, atomic_write_text
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans
from onc_co_scientist.expected_surprising.experiment import run_cell
from onc_co_scientist.expected_surprising.experiment_report import write_report

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--resume', action='store_true')
p.add_argument('--validate-only', action='store_true')
p.add_argument('--workers', type=int, default=25)
args = p.parse_args()
if args.workers < 1:
    p.error('--workers must be positive')
root = args.root.resolve()
now = lambda: datetime.now(UTC).isoformat()
lock = (root / 'driver.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
manifest = json.loads((root / 'frozen_manifest.json').read_text())
for name, expected in manifest['hashes'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
import onc_co_scientist.expected_surprising.coordination as code
assert Path(code.__file__).is_relative_to(root / 'source')
specs = {label: load_experiment_spec(root / label / 'config.yaml') for label in ('named', 'masked')}
plans = {label: {p.run_id:p for p in build_run_plans(spec)} for label,spec in specs.items()}
rows = json.loads((root / 'plan.json').read_text())
assert len({(r['grid'],r['run_id']) for r in rows}) == len(rows) == manifest['assigned_runs']
for row in rows:
    assert row['run_id'] in plans[row['grid']]
if args.validate_only:
    print(f'Validated {len(rows)} runs, {args.workers} global workers, frozen code and all input hashes.', flush=True)
    raise SystemExit(0)
if (root / 'execution.json').exists() and not args.resume:
    raise FileExistsError('Already launched; use --resume')
state = dict(started_at=now(), pid=os.getpid(), assigned_runs=len(rows), workers=args.workers,
             resume=args.resume, status='running', finished=[], worker_errors=[])
active = {}
pending = list(rows)
results = {label: [] for label in specs}

def status():
    state.update(updated_at=now(), active_runs=len(active), queued_runs=len(pending),
                 terminal_runs=len(state['finished'])+len(state['worker_errors']),
                 active=[{'grid':r['grid'],'run_id':r['run_id']} for r in active.values()])
    atomic_write_json(root / 'execution.json', state)
    counts = Counter((r['grid'],r['workflow_id'],'queued') for r in pending)
    counts.update((r['grid'],r['workflow_id'],'running') for r in active.values())
    for grid, rr in results.items():
        counts.update((grid,r['workflow_id'],r['status']) for r in rr)
    text = ['# Additional Gemma NSCLC batch', '', f'Updated {now()}', '',
            f'{len(rows)} assigned runs · {args.workers} global workers · 25 iterations · no federation.',
            'Model: gemma4-31b at http://camus:8060/v1. Dataset: es-v2-nsclc_clinical-42000.', '',
            '| Semantics | Workflow | Status | Runs |', '|---|---|---|---:|']
    text += [f'| {g} | {w} | {s} | {n} |' for (g,w,s),n in sorted(counts.items())]
    text += ['', '[Execution](execution.json) · [Plan](plan.json) · [Frozen manifest](frozen_manifest.json)']
    atomic_write_text(root / 'STATUS.md', '\n'.join(text)+'\n')

try:
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while pending or active:
            while pending and len(active) < args.workers:
                row = pending.pop(0)
                label = row['grid']
                active[pool.submit(run_cell, specs[label], plans[label][row['run_id']],
                    root / label, specs[label].fingerprint(), resume=args.resume)] = row
            status()
            done,_ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
            for future in done:
                row = active.pop(future)
                try:
                    result = future.result()
                    state['finished'].append({'grid':row['grid'],'run_id':row['run_id'],'status':result['status']})
                except Exception as exc:
                    result = {**row, 'status':'failed', 'error':str(exc), 'agent_calls':0,
                              'usage':{}, 'stop_reason':'worker_error', 'traceback':traceback.format_exc()}
                    state['worker_errors'].append(result)
                    atomic_write_json(root / row['grid'] / 'runs' / row['run_id'] / 'run.json', result)
                results[row['grid']].append(result)
                print(f"{now()} {result['status']} {row['grid']} {row['run_id']}", flush=True)
    for label, rr in results.items():
        selected = [plans[label][r['run_id']] for r in rows if r['grid']==label]
        atomic_write_json(root / label / 'summary.json', {'n_runs':len(rr),'runs':rr})
        write_report(specs[label], selected, rr, root / label)
    state.update(completed_at=now(), status='completed_with_failures' if any(r['status']=='failed' for rr in results.values() for r in rr) else 'completed')
    status()
except BaseException:
    state.update(status='interrupted', error=traceback.format_exc())
    status()
    raise
