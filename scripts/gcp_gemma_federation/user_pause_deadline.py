"""Stop new experiment admission at a user deadline, retaining live worker state."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.deadline.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def terminal_count(root, selection):
    count = 0
    for row in selection['target']:
        path = root / row['condition'] / 'runs' / row['run_id'] / 'run.json'
        count += bool(path.exists() and read(path).get('scientific_report_sha256'))
    return count


def close_admission(root, control):
    marker = root / 'PAUSE'
    if not marker.exists():
        marker.write_text('User requested a 7pm Eastern pause on September24. '
                          'Explicit user release required before new requests.\n')
    for path in [control / 'concurrency.json',
                 control / 'recovery-min-output/concurrency.json']:
        if path.exists():
            config = read(path)
            config.update(local_requests=0, requests_per_gpu=0,
                          endpoint_limits={key: 0 for key in config.get('endpoint_limits', {})},
                          reason='User deadline reached; retain worker state and drain active requests.')
            write(path, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    control = root / 'control/gcp-nvfp4'
    lock = (control / 'user-pause-deadline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy_path = control / 'user-pause-deadline.json'
    status_path = control / 'user-pause-deadline-status.json'
    selection = read(control / 'selection.json')
    assert len(selection['target']) == 36
    while True:
        policy = read(policy_path)
        if not policy.get('enabled'):
            write(status_path, dict(status='disabled', updated_at=datetime.now(timezone.utc).isoformat()))
            return
        assert policy.get('approved_by_user') is True and policy.get('authorization')
        deadline = datetime.fromisoformat(policy['deadline'])
        assert deadline.tzinfo is not None
        now = datetime.now(timezone.utc)
        completed = terminal_count(root, selection)
        if completed == 36:
            write(status_path, dict(status='all_terminal', terminal_count=36, updated_at=now.isoformat()))
            return
        state = dict(deadline=deadline.isoformat(), updated_at=now.isoformat(), terminal_count=completed,
                     unfinished_count=36-completed)
        if now < deadline:
            write(status_path, dict(state, status='armed'))
            time.sleep(min(10, (deadline-now).total_seconds()))
            continue
        close_admission(root, control)
        runtime_path = control / 'recovery-min-output/runtime.json'
        runtime = read(runtime_path) if runtime_path.exists() else {}
        fresh = (runtime.get('updated_at') and
                 (now-datetime.fromisoformat(runtime['updated_at'])).total_seconds() < 30)
        requests = sum(endpoint['active'] for endpoint in runtime.get('endpoints', []))
        limits = sum(endpoint['limit'] for endpoint in runtime.get('endpoints', []))
        drained = bool(fresh and requests == limits == 0)
        state.update(status='paused_in_memory' if drained else 'draining', active_requests=requests,
                     workers_retained=True, model_servers_untouched=True)
        write(status_path, state)
        eta_path = control / 'deadline-eta.json'
        eta = read(eta_path) if eta_path.exists() else {}
        summary = ('The user\'s 7pm Eastern cutoff has been reached. New requests are paused. '
                   + ('All in-flight requests have drained; workers retain their state for a later authorized resume.'
                      if drained else 'Requests already in flight are finishing and saving their responses.')
                   + f' {completed}/36 selected experiments are terminal. The shared model server remains running.')
        eta.update(updated_at=now.isoformat(), status=state['status'], summary=summary,
                   planning_hours=None, planning_finish_local='Paused by user deadline; explicit release required.')
        write(eta_path, eta)
        (root / 'analysis/deadline_20260924/ETA.md').write_text('# Experiment pause\n\n'+summary+'\n')
        if drained:
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
