"""Resume only released context-floor failures; never restart primary workers."""
import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import UTC, datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback

from recovery_state import read, process_alive


def recovery_endpoint_url(control, camus):
    if not camus:
        return 'http://sn4622130540:8000/v1'
    returned = read(control / 'camus8060-return-handoff.json')
    if returned:
        if (returned.get('approved_by_user') is not True
                or not returned.get('authorization')
                or returned.get('source_url') != 'http://camus:8002/v1'
                or returned.get('destination_url') != 'http://camus:8060/v1'):
            raise ValueError('Invalid approved Camus return handoff')
        return returned['destination_url']
    handoff = read(control / 'camus8002-handoff.json')
    if not handoff:
        return 'http://camus:8060/v1'
    if (handoff.get('approved_by_user') is not True
            or not handoff.get('authorization')
            or handoff.get('source_url') != 'http://camus:8060/v1'
            or handoff.get('destination_url') != 'http://camus:8002/v1'):
        raise ValueError('Invalid approved Camus endpoint handoff')
    return handoff['destination_url']


def eligible(root, selection, primary, attempted, floor):
    active = {r['run_id'] for r in primary.get('active', [])}
    released = {r['run_id'] for r in primary.get('selected_finished', [])
                if r['status'] == 'interrupted'}
    rows = []
    for row in selection['target']:
        rid = row['run_id']
        if rid not in released or rid in active or rid in attempted:
            continue
        folder = root / row['condition'] / 'runs' / rid
        retry = read(folder / 'retry_status.json')
        minimum = re.search(r'minimum is (\d+)', retry.get('error', ''))
        available = re.search(r'only (-?\d+) output tokens fit', retry.get('error', ''))
        if (retry.get('status') != 'paused_output_floor' or minimum is None
                or floor >= int(minimum[1]) or (available and int(available[1]) < floor)):
            continue
        if read(folder / 'run.json').get('scientific_report_sha256'):
            continue
        rows.append(row)
    return sorted(rows, key=lambda r: (r['replicate'], r['run_id']))


def install_camus_admission(pool, control, directory, root, primary_pid, write):
    """Keep primary capacity during replay; split eight slots only at live work."""
    original_slot = pool.slot
    activation_lock = threading.Lock()
    activated = False

    @contextmanager
    def slot(priority=0):
        nonlocal activated
        with activation_lock:
            if not activated:
                primary = read(control / 'concurrency.json')
                limits = primary.get('endpoint_limits', {})
                if (any(v for k, v in limits.items() if k != 'local-camus')
                        or primary.get('requests_per_gpu') != 0):
                    raise ValueError('Recovery requires Camus-only routing')
                alive = process_alive(primary_pid, root, 'resume_on_pool.py')
                if alive:
                    limits['local-camus'] = 4
                    primary['context_recovery_slot_split_at'] = datetime.now(UTC).isoformat()
                    write(control / 'concurrency.json', primary)
                    time.sleep(1.1)  # Outlast the primary broker's one-second config cache.
                    # Finish existing primary requests normally before using the
                    # four released slots. No request or model server is stopped.
                    while process_alive(primary_pid, root, 'resume_on_pool.py'):
                        runtime = read(control / 'hybrid-runtime.json')
                        active = sum(e['active'] for e in runtime.get('endpoints', [])
                                     if e['id'] == 'local-camus')
                        if active <= 4:
                            break
                        time.sleep(1)
                alive = process_alive(primary_pid, root, 'resume_on_pool.py')
                recovery = read(directory / 'concurrency.json')
                recovery['endpoint_limits'] = {'local-camus': 4 if alive else 8}
                write(directory / 'concurrency.json', recovery)
                write(directory / 'admission-activation.json', dict(
                    activated_at=datetime.now(UTC).isoformat(), total_limit=8,
                    primary_limit=4 if alive else 0, recovery_limit=4 if alive else 8,
                    reason='Preserve primary throughput during saved replay, then share the eight Camus slots.'))
                activated = True
        with original_slot(priority) as index:
            yield index

    pool.slot = slot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--primary-pid', required=True, type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    control = root / 'control/gcp-nvfp4'
    directory = control / 'recovery-min-output'
    directory.mkdir(exist_ok=True)
    lock = (directory / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if read(directory / 'execution.json'):
        raise ValueError('Existing recovery records require explicit restart handling')
    manifest = read(root / 'frozen_manifest.json')
    for name, expected in manifest['hashes'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen scientific file changed: ' + name)
    spec = importlib.util.spec_from_file_location('frozen_gemma_recovery_driver',
        root / 'source/scripts/expected_surprising/gemini_federation_grid.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    from deadline_scheduler import validate_selection, score_selected
    from context_fit import ContextFit
    from hybrid_pool import HybridPool, install_routes
    from parallel_sites import install_parallel_sites
    from onc_co_scientist.providers.base import ProviderInfrastructureError
    from onc_co_scientist.providers.vllm_federation import VLLMFederationProvider
    from onc_co_scientist.expected_surprising import federation, coordination
    from onc_co_scientist.expected_surprising.experiment import run_cell
    from onc_co_scientist.expected_surprising.coordination import WorkflowInfrastructureError
    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import build_run_plans
    if not Path(federation.__file__).is_relative_to(root / 'source'):
        raise ValueError('Recovery requires frozen scientific source')
    selection = read(control / 'selection.json')
    validate_selection(read(root / 'grid.json'), selection)
    policy = read(control / 'context-fit-policy.json')
    fit = ContextFit(policy, ProviderInfrastructureError, driver.atomic_write_json)
    floor = fit.policy['min_output_tokens']
    if floor not in {1024, 8192, 16384}:
        raise ValueError('Recovery requires the approved lower output minimum')
    camus = bool(read(control / 'camus-handoff.json').get('approved_by_user'))
    endpoint_id = 'local-camus' if camus else 'local-sn4622130540'
    endpoints = [dict(e, limit=0 if camus else 2) for e in read(control / 'hybrid-endpoints.json')
                 if e['id'] == endpoint_id and e['kind'] == 'local']
    expected_url = recovery_endpoint_url(control, camus)
    if len(endpoints) != 1 or endpoints[0]['url'] != expected_url:
        raise ValueError('Recovery may use only the currently authorized local endpoint')
    pool = HybridPool(endpoints, directory / 'concurrency.json')
    if camus:
        install_camus_admission(pool, control, directory, root, args.primary_pid,
                                driver.atomic_write_json)
    cleanup = install_parallel_sites(federation, coordination, VLLMFederationProvider,
                                     site_workers=16)
    install_routes(VLLMFederationProvider, selection['target'], root, pool,
                   driver.atomic_write_json, context_fit=fit)
    specs = {c: load_experiment_spec(root / c / 'config.yaml') for c in ('masked', 'named')}
    plans = {c: {p.run_id: p for p in build_run_plans(s)} for c, s in specs.items()}
    state = dict(pid=os.getpid(), primary_pid=args.primary_pid, status='running',
                 started_at=driver.now(), context_fit_policy=policy,
                 frozen_hashes_verified=len(manifest['hashes']), active=[], finished=[])
    attempted, active = set(), {}
    primary_lock = None

    def save():
        state.update(updated_at=driver.now(), active=list(active.values()))
        driver.atomic_write_json(directory / 'execution.json', state)
        driver.atomic_write_json(directory / 'runtime.json',
                                 dict(updated_at=driver.now(), **pool.snapshot()))

    def work(row):
        # The original controller has already joined its site futures before
        # publishing interrupted. Its scheduler never resubmits finished rows.
        primary = read(root / 'execution.json')
        if row not in eligible(root, selection, primary, set(), floor):
            raise ValueError('Recovery identity is not released by the primary')
        folder = root / row['condition'] / 'runs' / row['run_id']
        claim = (folder / 'context-recovery.lock').open('a')
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for retry in range(5):
                try:
                    return run_cell(specs[row['condition']], plans[row['condition']][row['run_id']],
                        root / row['condition'], specs[row['condition']].fingerprint(), resume=True)
                except WorkflowInfrastructureError as exc:
                    for path in (folder / 'calls').rglob('*.json'):
                        if driver.read(path, {}).get('result', {}).get('infrastructure_error'):
                            dest = folder / 'infrastructure_archive' / f'recovery-{time.time_ns()}' / path.relative_to(folder)
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            path.replace(dest)
                    floor_error = 'CONTEXT_FIT_OUTPUT_FLOOR' in str(exc)
                    driver.atomic_write_json(folder / 'retry_status.json', dict(
                        status='paused_output_floor' if floor_error else 'retry_wait',
                        retry=retry + 1, error=str(exc), updated_at=driver.now(), recovery=True))
                    if floor_error or retry == 4:
                        raise
                    time.sleep(min(60, 10 * 2 ** retry))
        finally:
            claim.close()

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                primary = read(root / 'execution.json')
                if read(control / 'launch.json').get('pid') != args.primary_pid:
                    raise ValueError('Primary controller changed; stop new recovery admission')
                if not (root / 'PAUSE').exists():
                    for row in eligible(root, selection, primary, attempted, floor):
                        attempted.add(row['run_id'])
                        active[executor.submit(work, row)] = row
                save()
                alive = process_alive(args.primary_pid, root, 'resume_on_pool.py')
                if not alive and primary_lock is None:
                    primary_lock = (root / 'driver.lock').open('a')
                    fcntl.flock(primary_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if camus:
                        config = read(directory / 'concurrency.json')
                        config['endpoint_limits'] = {'local-camus': 8}
                        driver.atomic_write_json(directory / 'concurrency.json', config)
                if not active and not alive:
                    break
                if active:
                    done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                    for future in done:
                        row = active.pop(future)
                        try:
                            status = future.result()['status']
                        except Exception as exc:
                            status = 'interrupted'
                            driver.atomic_write_json(root / row['condition'] / 'runs' / row['run_id'] / 'worker_error.json',
                                dict(error=str(exc), traceback=traceback.format_exc(),
                                     updated_at=driver.now(), recovery=True))
                        state['finished'].append(dict(condition=row['condition'], run_id=row['run_id'], status=status))
                else:
                    time.sleep(5)
            state['status'] = 'scoring'
            save()
            driver.refresh(root)
            score_selected(root, selection, driver)
            state['status'] = ('paused' if any(r['status'] == 'interrupted' for r in state['finished'])
                               else 'finished_recovery')
            state['ended_at'] = driver.now()
            save()
    except BaseException:
        state.update(status='recovery_error', error=traceback.format_exc())
        save()
        raise
    finally:
        cleanup()


if __name__ == '__main__':
    main()
