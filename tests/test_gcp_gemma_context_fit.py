"""Approved output amendment preserves prompts, replay, and truncation rejection."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onc_co_scientist.providers.base import ProviderInfrastructureError
from tests.test_vllm_federation import provider
from tests.test_federated_context_v2 import SYSTEM, payload, message


path = Path(__file__).parents[1] / 'scripts/gcp_gemma_federation/context_fit.py'
spec = importlib.util.spec_from_file_location('gemma_context_fit', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def policy():
    return dict(enabled=True, approved_by_user=True, nominal_max_tokens=65536,
                min_output_tokens=32768, safety_margin_tokens=512, context_window=262144,
                policy_id='test-approved', approved_at='2026-09-23T10:36:00+00:00')


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('count,effective', [(100, 65536), (197168, 64464),
                                           (201235, 60397), (228864, 32768)])
def test_exact_fit_audits_actual_wire_without_changing_nominal_or_prior_attempts(tmp_path, monkeypatch, count, effective):
    calls = []
    def post(url, **kwargs):
        assert url.endswith('/tokenize')
        calls.append(kwargs['json'])
        return SimpleNamespace(status_code=200, json=lambda: dict(count=count, max_model_len=262144))
    monkeypatch.setattr('requests.post', post)
    p = provider(tmp_path)
    body, _ = p.request_body([message(payload())], SYSTEM, 65536)
    original = json.dumps(body, sort_keys=True)
    call = tmp_path / 'call-original'; call.mkdir()
    write(call / 'request.json', body)
    (call / 'attempt-0001').mkdir()
    (call / 'attempt-0001/response.json').write_text('preserved')
    def send(self, wire, directory):
        assert wire == dict(body, max_tokens=effective)
        (directory / 'attempt-0002').mkdir()
        return 'answer'
    fit = module.ContextFit(policy(), ProviderInfrastructureError, write)
    assert fit.send(send, p, body, call) == 'answer'
    assert json.dumps(body, sort_keys=True) == original
    assert json.loads((call / 'request.json').read_text()) == body
    assert calls[0]['messages'] == body['messages']
    assert calls[0]['chat_template_kwargs'] == body['chat_template_kwargs']
    assert calls[0]['add_generation_prompt'] is True
    audit = json.loads((call / 'attempt-0002/context-fit.json').read_text())
    actual = json.loads(Path(audit['wire_request']).read_text())
    assert actual == dict(body, max_tokens=effective)
    assert audit['effective_request_sha256'] == module.digest(actual)
    assert audit['nominal_request_sha256'] == module.digest(body)
    assert (call / 'attempt-0001/response.json').read_text() == 'preserved'
    assert not (call / 'attempt-0001/context-fit.json').exists()


@pytest.mark.parametrize('count,window', [(228865, 262144), (-1, 262144), (True, 262144),
                                        (100, 131072), (100, '262144')])
def test_floor_and_bad_tokenization_send_no_generation(tmp_path, monkeypatch, count, window):
    monkeypatch.setattr('requests.post', lambda *a, **kw:
        SimpleNamespace(status_code=200, json=lambda: dict(count=count, max_model_len=window)))
    p = provider(tmp_path)
    body, _ = p.request_body([message(payload())], SYSTEM, 65536)
    call = tmp_path / 'call'; call.mkdir()
    def send(*args):
        pytest.fail('No generation is allowed when counting fails or the minimum cannot fit')
    with pytest.raises(ProviderInfrastructureError) as error:
        module.ContextFit(policy(), ProviderInfrastructureError, write).send(send, p, body, call)
    if count == 228865:
        assert module.FLOOR_MARKER in str(error.value)
        assert json.loads(next((call / 'context-fit').glob('send-*')).read_text())['status'] == 'paused_output_floor'


@pytest.mark.parametrize('finish', ['stop', 'length'])
def test_frozen_provider_replays_adjusted_result_and_rejects_truncation(tmp_path, monkeypatch, finish):
    p = provider(tmp_path)
    original_send = type(p)._send
    fit = module.ContextFit(policy(), ProviderInfrastructureError, write)
    monkeypatch.setattr(p, '_send', lambda body, call: fit.send(original_send, p, body, call))
    sent = []
    def post(url, **kwargs):
        sent.append(url)
        if url.endswith('/tokenize'):
            raw = dict(count=197168, max_model_len=262144)
        else:
            assert kwargs['json']['max_tokens'] == 64464
            raw = dict(model='gemma4-31b', choices=[dict(finish_reason=finish, message=dict(content='{}'))],
                       usage=dict(prompt_tokens=197168, completion_tokens=20))
        return SimpleNamespace(status_code=200, json=lambda: raw)
    monkeypatch.setattr('requests.post', post)
    kwargs = dict(system=SYSTEM, call_identity='same-nominal-call')
    first = p.chat_for_call([message(payload())], **kwargs)
    second = p.chat_for_call([message(payload())], **kwargs)
    assert first.raw == second.raw and len(sent) == 2
    assert bool(first.raw['adapter_error']) == (finish == 'length')
    if finish == 'length':
        assert 'incomplete generation rejected' in first.raw['adapter_error']
    assert len(list(tmp_path.glob('call-*'))) == 1


def test_tokenization_retries_temporary_failure_without_logging_credentials(tmp_path, monkeypatch):
    responses = iter([SimpleNamespace(status_code=503),
        SimpleNamespace(status_code=200, json=lambda: dict(count=100, max_model_len=262144))])
    monkeypatch.setattr('requests.post', lambda *a, **kw: next(responses))
    monkeypatch.setattr(module.time, 'sleep', lambda delay: None)
    p = provider(tmp_path)
    call = tmp_path / 'call'; call.mkdir()
    fit = module.ContextFit(policy(), ProviderInfrastructureError, write)
    assert fit.tokenize(p, dict(model='gemma4-31b', messages=[]), call)[:2] == (100, 262144)
    records = list((call / 'context-fit').glob('tokenize-*'))
    assert len(records) == 2
    assert 'Authorization' not in ''.join(p.read_text() for p in records)


def test_unapproved_policy_rejected():
    changed = dict(policy(), min_output_tokens=8192)
    with pytest.raises(ValueError, match='approved allowance'):
        module.ContextFit(changed, ProviderInfrastructureError, write)


@pytest.mark.parametrize('count,effective', [(233281, 28351), (228925, 32707), (245248, 16384)])
def test_approved_lower_floor_preserves_entire_request(tmp_path, monkeypatch, count, effective):
    changed = dict(policy(), min_output_tokens=16384, floor_amendment=dict(
        approved_by_user=True, min_output_tokens=16384,
        approved_at='2026-09-23', authorization='User approved reducing the floor'))
    monkeypatch.setattr('requests.post', lambda *a, **kw:
        SimpleNamespace(status_code=200, json=lambda: dict(count=count, max_model_len=262144)))
    p = provider(tmp_path)
    body, _ = p.request_body([message(payload())], SYSTEM, 65536)
    call = tmp_path / 'call'; call.mkdir()
    def send(self, wire, directory):
        assert wire == dict(body, max_tokens=effective)
        assert body['max_tokens'] == 65536
        return 'ok'
    fit = module.ContextFit(changed, ProviderInfrastructureError, write)
    assert fit.send(send, p, body, call) == 'ok'


def test_lower_floor_still_requires_explicit_matching_amendment():
    changed = dict(policy(), min_output_tokens=16384)
    with pytest.raises(ValueError):
        module.ContextFit(changed, ProviderInfrastructureError, write)


def test_approved_json_compaction_is_audited_without_changing_nominal_request(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(path.parent))
    changed = dict(policy(), json_whitespace=dict(approved_by_user=True,
        algorithm='json-whitespace-v1', approved_at='2026-09-24',
        authorization='User approved lossless JSON compaction'))
    p = provider(tmp_path)
    body = dict(model='gemma4-31b', max_tokens=65536,
        messages=[dict(role='user', content='Goal\n{ "evidence": [ "R1" ], "text": "keep  spaces" }')],
        temperature=1.0)
    original = json.dumps(body)
    compact = 'Goal\n{"evidence":["R1"],"text":"keep  spaces"}'
    def post(url, **kwargs):
        assert kwargs['json']['messages'][0]['content'] == compact
        return SimpleNamespace(status_code=200, json=lambda: dict(count=100, max_model_len=262144))
    monkeypatch.setattr('requests.post', post)
    call = tmp_path / 'call'; call.mkdir()
    write(call / 'request.json', body)
    def send(self, wire, directory):
        assert wire == dict(body, messages=[dict(role='user', content=compact)])
        return 'complete'
    assert module.ContextFit(changed, ProviderInfrastructureError, write).send(send, p, body, call) == 'complete'
    assert json.dumps(body) == original
    assert json.loads((call / 'request.json').read_text()) == body
    audit = json.loads(next((call / 'context-fit').glob('send-*')).read_text())
    assert audit['json_whitespace']['messages'][0]['removed_characters'] > 0
    assert audit['json_whitespace']['nominal_prompt_sha256'] != audit['json_whitespace']['effective_prompt_sha256']


def test_unapproved_json_compaction_rejected():
    with pytest.raises(ValueError, match='explicit approval'):
        module.ContextFit(dict(policy(), json_whitespace={'algorithm':'json-whitespace-v1'}),
                          ProviderInfrastructureError, write)


@pytest.mark.parametrize('floor,available', [(8192, 15861), (8192, 12280),
    (8192, 8192), (8192, 8191), (1024, 7913), (1024, 7807),
    (1024, 2021), (1024, 1024), (1024, 1023)])
def test_new_approved_floor_is_not_an_output_cap(tmp_path, monkeypatch, floor, available):
    changed = dict(policy(), min_output_tokens=floor, floor_amendment=dict(
        approved_by_user=True, min_output_tokens=floor,
        approved_at='2026-09-24', authorization='User: fix limit'))
    monkeypatch.setattr('requests.post', lambda *a, **kw: SimpleNamespace(
        status_code=200, json=lambda: dict(count=262144-512-available, max_model_len=262144)))
    p = provider(tmp_path)
    body, _ = p.request_body([message(payload())], SYSTEM, 65536)
    call = tmp_path / 'call'; call.mkdir()
    def send(self, wire, directory):
        assert available >= floor
        assert wire == dict(body, max_tokens=available)
        return 'complete answer'
    fit = module.ContextFit(changed, ProviderInfrastructureError, write)
    if available >= floor:
        assert fit.send(send, p, body, call) == 'complete answer'
    else:
        with pytest.raises(ProviderInfrastructureError, match=f'minimum is {floor}'):
            fit.send(send, p, body, call)
