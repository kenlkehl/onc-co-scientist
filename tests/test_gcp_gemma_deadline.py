"""Balanced deadline scheduling and concurrent endpoint provenance regressions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import hashlib

import pytest


def load(name):
    path = Path(__file__).parents[1] / 'scripts/gcp_gemma_federation' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hybrid = load('hybrid_pool')
deadline = load('deadline_scheduler')
live = load('live_status')


def grid_and_selection():
    grid = [dict(condition=str(cell), run_id=f'{cell}-{repeat}', replicate=repeat,
                 iterations=25, workflow_id='persistent', semantic_condition='expected')
            for repeat in range(1, 11) for cell in range(12)]
    return grid, dict(target=[r for r in grid if r['replicate'] <= 3],
                     core=[r for r in grid if r['replicate'] <= 2],
                     deferred=[r for r in grid if r['replicate'] > 3])


def test_fixed_selection_keeps_full_iterations_and_excludes_later_repeats():
    grid, selection = grid_and_selection()
    chosen, deferred = deadline.validate_selection(grid, selection)
    assert len(chosen) == 36 and len(deferred) == 84
    assert all(r['iterations'] == 25 for r in chosen)
    selection['target'] = [dict(r) for r in selection['target']]
    selection['target'][0]['iterations'] = 10
    with pytest.raises(ValueError, match='exact frozen rows'):
        deadline.validate_selection(grid, selection)


def test_hybrid_concurrent_routes_do_not_mutate_shared_config_or_prior_attempts(tmp_path):
    @dataclass(frozen=True)
    class Config:
        base_url: str = 'original'

    barrier = threading.Barrier(2)
    received = []

    class Provider:
        def _send(self, body, call):
            before = self._config.base_url
            barrier.wait(timeout=3)
            time.sleep(.01)
            assert self._config.base_url == before
            assert body == {'messages': [{'role': 'user', 'content': 'unchanged'}], 'top_k': 64}
            (call / 'attempt-0002').mkdir()
            received.append(before)
            return before

    endpoints = [dict(id='cloud', kind='cloud', url='cloud-url', limit=1),
                 dict(id='local', kind='local', url='local-url', limit=1)]
    pool = hybrid.HybridPool(endpoints, tmp_path / 'config.json')
    rows = [dict(condition='named', run_id='same-run', replicate=1)]
    write = lambda path, value: path.write_text(json.dumps(value))
    hybrid.install_routes(Provider, rows, tmp_path, pool, write)
    p = Provider()
    p._config = Config()
    p.root = tmp_path / 'named/runs/same-run/provider_audit'
    calls = [p.root / f'call-{i}' for i in range(2)]
    for call in calls:
        (call / 'attempt-0001').mkdir(parents=True)
        (call / 'attempt-0001/response.json').write_text('old response')
    body = {'messages': [{'role': 'user', 'content': 'unchanged'}], 'top_k': 64}
    with ThreadPoolExecutor(2) as workers:
        results = list(workers.map(lambda call: p._send(body, call), calls))
    assert sorted(results) == ['cloud-url', 'local-url']
    assert p._config.base_url == 'original'
    for call, result in zip(calls, results):
        assert json.loads((call / 'attempt-0002/deployment.json').read_text())['endpoint'] == result
        assert (call / 'attempt-0001/response.json').read_text() == 'old response'
        assert not (call / 'attempt-0001/deployment.json').exists()
    assert sum(pool.completed) == 2 and sum(pool.active) == 0
    p.root = tmp_path / 'named/runs/deferred/provider_audit'
    with pytest.raises(KeyError):
        p._send(body, calls[0])


def test_core_takes_next_slot_before_waiting_third_repeat_and_recovers_on_error(tmp_path):
    pool = hybrid.HybridPool([dict(id='local', kind='local', url='local', limit=1)], tmp_path / 'config')
    order = []
    def work(priority):
        with pool.slot(priority):
            order.append(priority)
    with ThreadPoolExecutor(2) as executor:
        with pool.slot():
            third = executor.submit(work, 1)
            core = executor.submit(work, 0)
            deadline_at = time.monotonic() + 2
            while len(pool.waiting) < 2 and time.monotonic() < deadline_at:
                time.sleep(.01)
            assert len(pool.waiting) == 2
        third.result(timeout=2)
        core.result(timeout=2)
    assert order == [0, 1]
    with pytest.raises(RuntimeError):
        with pool.slot():
            raise RuntimeError('transport failure')
    assert pool.active == [0] and pool.failed == [1]
    assert pool.snapshot()['endpoints'][0]['cooling_down']


def test_live_cohort_counts_do_not_depend_on_stale_archive_snapshot(tmp_path):
    _, selection = grid_and_selection()
    folder = tmp_path / 'control/gcp-nvfp4'
    folder.mkdir(parents=True)
    (folder / 'selection.json').write_text(json.dumps(selection))
    first = selection['target'][0]
    state = dict(selected_count=36, active=selection['target'][1:],
                 selected_finished=[dict(first, status='completed')])
    text = '\n'.join(live.cohort_lines(tmp_path, state, {'runs': []}))
    assert '1/36 completed' in text and '1/24 completed' in text
    assert '35 active' in text and '84 deferred' in text


def test_core_fallback_retains_terminal_third_repeat_status_and_report(tmp_path):
    rows = [dict(condition='named', run_id=f'r{i}') for i in range(3)]
    for row, status in zip(rows, ('completed', 'failed', None)):
        folder = tmp_path / 'named/runs' / row['run_id']
        folder.mkdir(parents=True)
        if status:
            blob = json.dumps({'status': status, 'scientific_result': 'preserved'}).encode()
            (folder / 'report.json').write_bytes(blob)
            (folder / 'run.json').write_text(json.dumps(dict(status=status,
                scientific_report_sha256=hashlib.sha256(blob).hexdigest())))
    driver = SimpleNamespace(read=lambda path, default: json.loads(path.read_text()) if path.exists() else default,
                             sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    records = deadline.outside_execution_records(tmp_path, rows, driver)
    assert [r['status'] for r in records] == ['completed', 'failed', 'paused']
    (tmp_path / 'named/runs/r1/report.json').write_text('changed')
    with pytest.raises(ValueError, match='terminal report changed'):
        deadline.outside_execution_records(tmp_path, rows, driver)


def test_core_fallback_live_counts_include_terminal_third_repeats(tmp_path):
    _, selection = grid_and_selection()
    selection['execution_repeat_limit'] = 2
    control = tmp_path / 'control/gcp-nvfp4'
    control.mkdir(parents=True)
    (control / 'selection.json').write_text(json.dumps(selection))
    core = selection['core'][0]
    third = selection['target'][24]
    state = dict(selected_count=24, active=[], deferred_count=85,
                 selected_finished=[dict(core, status='completed')],
                 finished=[dict(core, status='completed'), dict(third, status='failed')])
    text = '\n'.join(live.cohort_lines(tmp_path, state, {'runs': []}))
    assert '1/36 completed' in text and '1/24 completed' in text
    assert '1 failed' in text and '85 deferred' in text
    assert 'noon fallback is active' in text
