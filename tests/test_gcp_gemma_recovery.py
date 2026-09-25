import importlib.util
import json
from pathlib import Path
import sys

import pytest

directory = Path(__file__).parents[1] / 'scripts/gcp_gemma_federation'
sys.path.insert(0, str(directory))
from recovery_state import merge_execution, merge_runtime
from recover_paused import eligible
from recover_paused import install_camus_admission
from recover_paused import recovery_endpoint_url


def test_recovery_endpoint_requires_explicit_matching_handoff(tmp_path):
    assert recovery_endpoint_url(tmp_path, True) == 'http://camus:8060/v1'
    path = tmp_path / 'camus8002-handoff.json'
    handoff = dict(approved_by_user=True, authorization='User: move to camus:8002',
                   source_url='http://camus:8060/v1', destination_url='http://camus:8002/v1')
    path.write_text(json.dumps(handoff))
    assert recovery_endpoint_url(tmp_path, True) == 'http://camus:8002/v1'
    path.write_text(json.dumps(dict(handoff, approved_by_user=False)))
    with pytest.raises(ValueError, match='Invalid approved'):
        recovery_endpoint_url(tmp_path, True)
    path.write_text(json.dumps(dict(handoff, destination_url='http://other:8002/v1')))
    with pytest.raises(ValueError, match='Invalid approved'):
        recovery_endpoint_url(tmp_path, True)


def test_approved_return_to_8060_supersedes_historical_8002_handoff(tmp_path):
    (tmp_path / 'camus8002-handoff.json').write_text(json.dumps(dict(
        approved_by_user=True, authorization='Original8002move',
        source_url='http://camus:8060/v1', destination_url='http://camus:8002/v1')))
    path = tmp_path / 'camus8060-return-handoff.json'
    handoff = dict(approved_by_user=True, authorization='User: you can use camus:8060',
                  source_url='http://camus:8002/v1', destination_url='http://camus:8060/v1')
    path.write_text(json.dumps(handoff))
    assert recovery_endpoint_url(tmp_path, True) == 'http://camus:8060/v1'
    for change in [dict(approved_by_user=False), dict(authorization=''),
                   dict(destination_url='http://other:8060/v1')]:
        path.write_text(json.dumps(dict(handoff, **change)))
        with pytest.raises(ValueError, match='Invalid approved Camus return'):
            recovery_endpoint_url(tmp_path, True)


def test_recovery_excludes_active_deferred_and_terminal_identities(tmp_path):
    rows = [dict(condition='masked', run_id=str(i), replicate=2) for i in range(4)]
    for row in rows:
        folder = tmp_path / 'masked/runs' / row['run_id']; folder.mkdir(parents=True)
        (folder / 'retry_status.json').write_text(json.dumps(dict(
            status='paused_output_floor', error='minimum is 32768')))
    (tmp_path / 'masked/runs/2/run.json').write_text(json.dumps(dict(scientific_report_sha256='preserved')))
    primary = dict(active=[rows[0]], selected_finished=[dict(r, status='interrupted') for r in rows])
    assert eligible(tmp_path, dict(target=rows[:3]), primary, set(), 16384) == [rows[1]]
    assert eligible(tmp_path, dict(target=rows), primary, {'1', '3'}, 16384) == []


def test_reporting_merges_disjoint_workers_and_preserves_failed_reports():
    a = dict(condition='masked', run_id='a')
    b = dict(condition='masked', run_id='b')
    primary = dict(active=[a], finished=[dict(b, status='interrupted')],
                   selected_finished=[dict(b, status='interrupted')])
    recovery = dict(active=[b], finished=[], pid=123)
    state = merge_execution(primary, recovery, True)
    assert state['active'] == [a, b] and state['finished'] == []
    assert primary['finished'][0]['status'] == 'interrupted'
    recovery.update(active=[], finished=[dict(b, status='failed')])
    state = merge_execution(primary, recovery, False)
    assert state['finished'] == [dict(b, status='failed')]
    assert state['selected_finished'] == state['finished']


def test_reporting_rejects_overlapping_ownership_and_handles_recovery_crash():
    row = dict(condition='masked', run_id='a')
    with pytest.raises(ValueError, match='overlap'):
        merge_execution(dict(active=[row]), dict(active=[row]), True)
    result = merge_execution(dict(active=[]), dict(active=[row]), False)
    assert result['active'] == []
    assert result['finished'][0]['status'] == 'interrupted'


def test_counter_merge_keeps_cumulative_recovery_counts_after_exit():
    primary = dict(waiting=1, endpoints=[dict(url='local', active=2, limit=4, completed=20, failed=1)])
    recovery = dict(waiting=2, endpoints=[dict(url='local', active=1, limit=2, completed=3, failed=0)])
    result = merge_runtime(primary, recovery, True)
    assert result['endpoints'][0] == dict(url='local', active=3, limit=6, completed=23, failed=1)
    result = merge_runtime(primary, recovery, False)
    assert result['endpoints'][0] == dict(url='local', active=2, limit=4, completed=23, failed=1)
    assert result['waiting'] == 1


def test_new_recovery_handles_16384_floor_without_repeating_8192_failure(tmp_path):
    row = dict(condition='masked', run_id='paused', replicate=2)
    folder = tmp_path / 'masked/runs/paused'; folder.mkdir(parents=True)
    path = folder / 'retry_status.json'
    primary = dict(active=[], selected_finished=[dict(row, status='interrupted')])
    path.write_text(json.dumps(dict(status='paused_output_floor',
        error='CONTEXT_FIT_OUTPUT_FLOOR: only 12280 output tokens fit; minimum is 16384.')))
    assert eligible(tmp_path, dict(target=[row]), primary, set(), 8192) == [row]
    assert eligible(tmp_path, dict(target=[row]), primary, set(), 16384) == []
    path.write_text(json.dumps(dict(status='paused_output_floor',
        error='CONTEXT_FIT_OUTPUT_FLOOR: only 8000 output tokens fit; minimum is 8192.')))
    assert eligible(tmp_path, dict(target=[row]), primary, set(), 8192) == []


def test_camus_recovery_uses_primary_spare_slots_only_after_replay(tmp_path, monkeypatch):
    from hybrid_pool import HybridPool
    import recover_paused
    control = tmp_path / 'control'; control.mkdir()
    directory = control / 'recovery'; directory.mkdir()
    def write(path, value): path.write_text(json.dumps(value))
    write(control / 'concurrency.json', dict(requests_per_gpu=0,
        endpoint_limits={'GCP-0':0, 'local-sn4622130540':0, 'local-camus':8}))
    write(control / 'hybrid-runtime.json', dict(endpoints=[dict(id='local-camus', active=4)]))
    write(directory / 'concurrency.json', dict(local_requests=0, endpoint_limits={'local-camus':0}))
    pool = HybridPool([dict(id='local-camus', kind='local', url='camus', limit=0)],
                      directory / 'concurrency.json')
    monkeypatch.setattr(recover_paused, 'process_alive', lambda *a: True)
    monkeypatch.setattr(recover_paused.time, 'sleep', lambda *a: None)
    install_camus_admission(pool, control, directory, tmp_path, 123, write)
    assert json.loads((control / 'concurrency.json').read_text())['endpoint_limits']['local-camus'] == 8
    assert pool.snapshot()['endpoints'][0]['limit'] == 0
    with pool.slot():
        assert json.loads((control / 'concurrency.json').read_text())['endpoint_limits']['local-camus'] == 4
        assert pool.snapshot()['endpoints'][0]['limit'] == 4
    assert pool.completed == [1]


def test_dead_primary_does_not_add_phantom_recovery_capacity():
    primary = dict(waiting=2, endpoints=[dict(url='camus', active=2, limit=4, completed=60, failed=1)])
    recovery = dict(waiting=0, endpoints=[dict(url='camus', active=4, limit=8, completed=20, failed=0)])
    result = merge_runtime(primary, recovery, True, primary_alive=False)
    assert result['endpoints'][0] == dict(url='camus', active=4, limit=8, completed=80, failed=1)
    assert result['waiting'] == 0
