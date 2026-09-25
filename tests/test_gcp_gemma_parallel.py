import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from onc_co_scientist.expected_surprising import coordination, federation
from onc_co_scientist.providers.base import ChatMessage, ChatResponse
from onc_co_scientist.providers.vllm_federation import VLLMFederationProvider, usage_receipt
from tests.test_vllm_federation import provider
from tests.test_federated_context_v2 import SYSTEM, payload, message

path = Path(__file__).parents[1] / 'scripts/gcp_gemma_federation/parallel_sites.py'
spec = importlib.util.spec_from_file_location('gemma_parallel', path)
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)


@pytest.fixture
def enabled():
    cleanup = parallel.install_parallel_sites(federation, coordination, VLLMFederationProvider,
                                              site_workers=8)
    yield
    cleanup()


def raw():
    return dict(model='gemma4-31b', choices=[dict(finish_reason='stop', message=dict(content='{}'))],
                usage=dict(prompt_tokens=100, completion_tokens=25))


def test_transport_parallel_identity_and_exact_replay(enabled, monkeypatch, tmp_path):
    p = provider(tmp_path)
    calls = []
    barrier = threading.Barrier(4)
    def send(body, call):
        calls.append((body, call))
        barrier.wait(timeout=5)
        return raw(), usage_receipt(raw()), 1
    monkeypatch.setattr(p, '_send', send)
    def invoke(identity):
        return p.chat_for_call([message(payload())], system=SYSTEM, call_identity=identity)
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(invoke, ['peer1', 'peer2', 'peer3', 'peer4']))
    assert [x.text for x in answers] == ['{}'] * 4
    assert len({str(x[1]) for x in calls}) == 4
    assert all(x[0] == calls[0][0] for x in calls)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(invoke, ['peer1'] * 4))
    assert len(calls) == 4


def test_duplicate_identity_is_sent_once(enabled, monkeypatch, tmp_path):
    p = provider(tmp_path)
    calls = []
    def send(body, call):
        calls.append(call)
        time.sleep(.03)
        return raw(), usage_receipt(raw()), 1
    monkeypatch.setattr(p, '_send', send)
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(lambda _: p.chat_for_call([message(payload())], system=SYSTEM,
                                                       call_identity='same'), range(4)))
    assert len(calls) == 1 and all(x.text == '{}' for x in answers)


def test_shared_budget_reserves_inflight_calls(enabled, tmp_path):
    barrier = threading.Barrier(2)
    class Provider:
        model_id = 'test'
        def chat(self, *args, **kwargs):
            barrier.wait(timeout=5)
            return ChatResponse(text='{}', model_id=self.model_id, raw={'usage': raw()['usage']})
    budget = {'calls': 0, '_parallel_lock': threading.RLock(), '_parallel_reserved': set()}
    coords = [coordination.StageCoordinator(Provider(), SimpleNamespace(mode='sequential'), [],
              SimpleNamespace(max_retries_per_stage=2, max_tokens_per_call=100),
              SimpleNamespace(max_agent_calls=2), tmp_path / f'site{i}', budget) for i in range(4)]
    def run(coord):
        try:
            coord._call('i001-explore-agent-a1', [ChatMessage('system', 's'), ChatMessage('user', 'u')],
                        session='s', authoritative=True, iteration=1, stage='explore', kind='linear')
            return 'ok'
        except ValueError as error:
            return str(error)
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(run, coords))
    assert result.count('ok') == 2
    assert result.count('Run exhausted max_agent_calls') == 2
    assert budget['calls'] == 2 and not budget['_parallel_reserved']


def test_sites_overlap_but_central_order_and_repairs_match(monkeypatch):
    monkeypatch.setattr(federation, 'translate', lambda c, s, i, data:
                        (None, SimpleNamespace(model_dump=lambda **kw: data), None))
    monkeypatch.setattr(federation, 'repair_message', lambda c, error: error)
    def make(barrier=None):
        obj = federation.FederatedCoordinator.__new__(federation.FederatedCoordinator)
        obj.source = SimpleNamespace(max_retries_per_stage=2, peer_failure_policy='chair_with_available')
        obj.prepared, obj.validation_summaries, obj.handoff_errors, obj.active = {}, {}, [], None
        obj.controller = SimpleNamespace(service=SimpleNamespace(state={'cache': {}}))
        obj.contexts = {f'site_{i}': {'site': i} for i in range(1, 5)}
        obj._site_prompt = lambda prompt, site, stage: prompt + site
        obj._committed_local_results = lambda site: {}
        obj._committed_validation_summaries = lambda: {}
        obj._write_handoff = lambda *args: None
        finished = set()
        class Site:
            def __init__(self, name): self.name = name
            def reject(self): pass
            def respond(self, prompt, *, attempt, **kwargs):
                if barrier is not None and attempt == 1: barrier.wait(timeout=5)
                if self.name in ('site_1', 'site_3') and attempt == 1:
                    raise ValueError('repair ' + self.name)
                finished.add(self.name)
                return ChatResponse(text=json.dumps({'site': self.name}), model_id='test')
        obj.sites = {name: Site(name) for name in obj.contexts}
        prompts = []
        def central(prompt, **kwargs):
            assert finished == set(obj.sites)
            prompts.append(prompt)
            return ChatResponse(text='{}', model_id='test')
        obj.central = SimpleNamespace(respond=central)
        return obj, prompts
    serial, serial_prompts = make()
    serial.respond('frozen research prompt', iteration=1, stage='explore', attempt=1)
    cleanup = parallel.install_parallel_sites(federation, coordination, VLLMFederationProvider,
                                              site_workers=4)
    try:
        concurrent, concurrent_prompts = make(threading.Barrier(4))
        concurrent.respond('frozen research prompt', iteration=1, stage='explore', attempt=1)
        assert concurrent_prompts == serial_prompts
        assert concurrent.prepared == serial.prepared
        assert concurrent.handoff_errors == serial.handoff_errors
    finally:
        cleanup()


def test_limiter_bounds_inflight_and_releases_after_failure(tmp_path):
    path = tmp_path / 'concurrency.json'
    path.write_text('{"requests_per_gpu": 2}')
    gate = parallel.PoolLimiter(path, default=2)
    lock = threading.Lock()
    current = maximum = 0
    def run(i):
        nonlocal current, maximum
        try:
            with gate.slot(0):
                with lock:
                    current += 1
                    maximum = max(maximum, current)
                time.sleep(.02)
                with lock: current -= 1
                if i == 1: raise RuntimeError('failure')
        except RuntimeError: pass
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))
    assert maximum == 2 and gate.active == [0] * 8 and not gate.waiting[0]
    path.write_text('{"requests_per_gpu": 1}')
    gate.last_read = 0
    with gate.slot(0): assert gate.limit == 1


def test_existing_scientific_equivalence_grid_with_parallel_sites(enabled, tmp_path, monkeypatch):
    from tests.test_expected_surprising_federation import (
        test_grid_n1_equivalence_stage_handoffs_tokens_and_shared_budget as check,
    )
    check(tmp_path, monkeypatch)
