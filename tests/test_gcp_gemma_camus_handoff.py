"""Server handoff must preserve nominal journals and terminal science."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/gcp_gemma_federation'))
from context_fit import ContextFit, digest
from hybrid_pool import HybridPool, install_routes
from deadline_scheduler import terminal_status
from live_status import active_local_endpoints, experiment_request_lines


def write(path, value):
    path.write_text(json.dumps(value))


def test_endpoint_limits_disable_old_servers_while_camus_is_active(tmp_path):
    endpoints = [dict(id=i, kind=k, url=i, limit=4) for i, k in
                 [('GCP-0', 'cloud'), ('sn', 'local'), ('camus', 'local')]]
    config = tmp_path / 'concurrency.json'
    write(config, dict(requests_per_gpu=0, local_requests=0,
                       endpoint_limits={'GCP-0': 0, 'sn': 0, 'camus': 4}))
    pool = HybridPool(endpoints, config)
    with pool.slot() as i:
        assert i == 2 and pool.limits == [0, 0, 4]
    assert pool.completed == [0, 0, 1]


@pytest.mark.parametrize('count', [100, 239000])
def test_alias_and_output_changes_audit_exact_wire_without_rewriting_nominal(tmp_path, monkeypatch, count):
    @dataclass
    class Config:
        base_url: str = 'http://old/v1'
        api_key: str = 'unused'
    row = dict(condition='masked', run_id='r1', replicate=1)
    audit = tmp_path / 'masked/runs/r1/provider_audit'; audit.mkdir(parents=True)
    call = audit / 'call-original'; call.mkdir()
    original = dict(model='gemma4-31b', max_tokens=65536,
                    messages=[dict(role='user', content='preserved')], temperature=1,
                    top_p=.95, top_k=64, chat_template_kwargs={'enable_thinking': True})
    write(call / 'request.json', original)
    class Provider:
        def _send(self, body, directory):
            assert self._config.base_url == 'http://camus:8060/v1'
            assert body['model'] == 'RedHatAI/Gemma-4-31B-IT-FP8-Dynamic'
            assert body['max_tokens'] == min(65536, 262144 - count - 512)
            assert body['messages'] == original['messages']
            (directory / 'attempt-0001').mkdir()
            return 'answer'
    p = Provider(); p.root = audit; p._config = Config()
    monkeypatch.setattr('requests.post', lambda url, **kwargs:
        SimpleNamespace(status_code=200, json=lambda: dict(count=count, max_model_len=262144)))
    policy = dict(enabled=True, approved_by_user=True, nominal_max_tokens=65536,
        safety_margin_tokens=512, context_window=262144, min_output_tokens=16384,
        policy_id='approved', approved_at='2026-09-23', floor_amendment=dict(
            approved_by_user=True, min_output_tokens=16384, authorization='user', approved_at='now'))
    endpoint = dict(id='camus', kind='local', url='http://camus:8060/v1', limit=4,
        served_model='RedHatAI/Gemma-4-31B-IT-FP8-Dynamic', quantization='FP8')
    pool = HybridPool([endpoint], tmp_path / 'unused')
    install_routes(Provider, [row], tmp_path, pool, write,
                   context_fit=ContextFit(policy, RuntimeError, write))
    assert p._send(original, call) == 'answer'
    assert json.loads((call / 'request.json').read_text()) == original
    assert p._config.base_url == 'http://old/v1'
    record = json.loads((call / 'attempt-0001/context-fit.json').read_text())
    wire = json.loads(Path(record['wire_request']).read_text())
    assert record['nominal_request_sha256'] == digest(original)
    assert record['effective_request_sha256'] == digest(wire)
    assert record['model_alias_adjusted'] is True
    assert json.loads((call / 'attempt-0001/deployment.json').read_text())['quantization'] == 'FP8'


@pytest.mark.parametrize('status', ['completed', 'failed'])
def test_verified_terminal_reports_are_not_replayed(tmp_path, status):
    folder = tmp_path / 'masked/runs/r1'; folder.mkdir(parents=True)
    report = folder / 'report.json'; report.write_text('{"scientific":"preserved"}')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    write(folder / 'run.json', dict(status=status, scientific_report_sha256=sha(report)))
    driver = SimpleNamespace(read=lambda p, default: json.loads(p.read_text()), sha=sha)
    row = dict(condition='masked', run_id='r1')
    assert terminal_status(tmp_path, row, driver) == status
    report.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        terminal_status(tmp_path, row, driver)


def test_live_report_names_camus_and_probes_only_enabled_endpoint(tmp_path):
    write(tmp_path / 'hybrid-endpoints.json', [dict(id=i, kind='local', url=i) for i in ['sn', 'camus']])
    write(tmp_path / 'concurrency.json', dict(local_requests=0, endpoint_limits={'camus':4}))
    assert [e['id'] for e in active_local_endpoints(tmp_path)] == ['camus']
    text = '\n'.join(experiment_request_lines(dict(endpoints=[dict(
        id='local-camus', kind='local', active=2, limit=4, completed=5)])))
    assert '| Local camus | 2 | 4 | 5 |' in text
