"""Combine disjoint recovery workers with the untouched primary controller."""
from copy import deepcopy
import json
from pathlib import Path


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def process_alive(pid, root, script):
    try:
        command = Path(f'/proc/{int(pid)}/cmdline').read_bytes().decode()
        return str(root) in command and script in command
    except (OSError, ValueError, TypeError):
        return False


def merge_execution(primary, recovery, alive):
    state = deepcopy(primary)
    if not recovery:
        return state
    active = recovery.get('active', []) if alive else []
    finished = list(recovery.get('finished', []))
    if not alive:
        finished += [dict(condition=r['condition'], run_id=r['run_id'], status='interrupted')
                     for r in recovery.get('active', [])]
    owned = {(r['condition'], r['run_id']) for r in [*active, *finished]}
    # Recovery may supersede only identities released by the primary worker.
    if owned & {(r['condition'], r['run_id']) for r in state.get('active', [])}:
        raise ValueError('Primary and recovery workers overlap')
    state['active'] = state.get('active', []) + active
    for key in ('finished', 'selected_finished'):
        state[key] = [r for r in state.get(key, [])
                      if (r['condition'], r['run_id']) not in owned] + finished
    state['recovery'] = dict(pid=recovery.get('pid'), alive=alive,
                             status=recovery.get('status'))
    return state


def merge_runtime(primary, recovery, alive, primary_alive=True):
    result = deepcopy(primary)
    result.setdefault('endpoints', [])
    if not primary_alive:
        result['waiting'] = 0
        for endpoint in result['endpoints']:
            endpoint.update(active=0, limit=0)
    by_url = {e['url']: e for e in result['endpoints']}
    for endpoint in recovery.get('endpoints', []):
        target = by_url.get(endpoint['url'])
        if target is None:
            raise ValueError('Recovery endpoint outside original pool')
        for key in ('completed', 'failed'):
            target[key] += endpoint[key]
        if alive:
            for key in ('active', 'limit'):
                target[key] += endpoint[key]
    result['waiting'] = result.get('waiting', 0) + (recovery.get('waiting', 0) if alive else 0)
    return result


def combined(root, primary, runtime):
    directory = root / 'control/gcp-nvfp4/recovery-min-output'
    recovery = read(directory / 'execution.json')
    alive = process_alive(recovery.get('pid'), root, 'recover_paused.py')
    primary_alive = process_alive(primary.get('pid'), root, 'resume_on_pool.py')
    return (merge_execution(primary, recovery, alive),
            merge_runtime(runtime, read(directory / 'runtime.json'), alive,
                          primary_alive=primary_alive), alive)
