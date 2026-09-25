import json
from types import SimpleNamespace

import pytest

from onc_co_scientist.providers.vllm_federation import VLLMFederationConfig,VLLMFederationProvider,usage_receipt
from onc_co_scientist.providers.base import ChatMessage,ProviderInfrastructureError
from tests.test_federated_context_v2 import SYSTEM,payload,message


def provider(tmp_path):
    return VLLMFederationProvider(VLLMFederationConfig(model_id='gemma4-31b',
        audit_dir=str(tmp_path),sampling_profile='gemma4',json_object_output=True,reasoning_effort='medium'))


def test_gemma_request_preserves_aggregate_evidence_and_sampling(tmp_path):
    p=provider(tmp_path)
    body,layout=p.request_body([message(payload())],SYSTEM,65536)
    assert body['model']=='gemma4-31b' and body['max_tokens']==65536
    assert body['temperature']==1.0 and body['top_p']==.95 and body['top_k']==64
    assert body['chat_template_kwargs']=={'enable_thinking':True}
    assert body['response_format']=={'type':'json_object'} and body['tool_choice']=='none'
    assert 'tools' not in body and layout['current_ledger_roundtrip']
    assert '-0.12345678901234' in json.dumps(body)
    fallback,_=p.request_body([message(payload())],SYSTEM,65536,final_retry=True,truncation_failures=2)
    assert fallback['chat_template_kwargs']=={'enable_thinking':False}


def test_coordinator_peers_independent_replay_safe_and_usage_retained(monkeypatch,tmp_path):
    from onc_co_scientist.expected_surprising.coordination import StageCoordinator
    p=provider(tmp_path/'audit');calls=[]
    raw=dict(model='gemma4-31b',choices=[dict(finish_reason='stop',message=dict(content='{}'))],
        usage=dict(prompt_tokens=100,completion_tokens=25,prompt_tokens_details=None))
    monkeypatch.setattr(p,'_send',lambda body,call:(calls.append((body,call)) or (raw,usage_receipt(raw),1)))
    coord=StageCoordinator(p,SimpleNamespace(mode='deliberative'),[],
        SimpleNamespace(max_retries_per_stage=2,max_tokens_per_call=65536),
        SimpleNamespace(max_agent_calls=100),tmp_path/'calls')
    messages=[ChatMessage('system',SYSTEM),message(payload())]
    for peer in (1,2,1,2):
        coord._call(f'i001-analyze-peer{peer}-r1-a1',messages,session=f'peer{peer}',
            authoritative=False,iteration=1,stage='analyze',kind='peer')
    assert len(calls)==2 and calls[0][1]!=calls[1][1]
    assert coord.replayed_calls==2
    assert coord.records[0]['result']['usage']['output_tokens']==25
    assert coord.records[0]['request']['generation_options']['temperature']==1.0


def test_vllm_retry_logs_each_attempt_without_auth_headers(monkeypatch,tmp_path):
    p=provider(tmp_path);sleeps=[]
    raw=dict(usage=dict(prompt_tokens=10,completion_tokens=3))
    responses=iter([SimpleNamespace(status_code=c,text='temporary',json=lambda:raw) for c in (503,200)])
    monkeypatch.setattr('requests.post',lambda *a,**k:next(responses))
    monkeypatch.setattr('time.sleep',sleeps.append)
    call=tmp_path/'call-test';call.mkdir()
    _,usage,attempts=p._send({},call)
    assert attempts==2 and sleeps==[1] and usage['input_tokens']==10
    assert len(list(call.glob('attempt-*/http.json')))==2
    assert 'Authorization' not in ''.join(x.read_text() for x in call.rglob('*.json'))


def test_federation_routing_keeps_single_site_provider(monkeypatch,tmp_path):
    from onc_co_scientist.expected_surprising import experiment
    config=dict(kind='vllm_openai',model_id='gemma4-31b')
    marker=object();monkeypatch.setattr(experiment,'get_provider',lambda cfg:marker)
    assert experiment._cell_provider(config,SimpleNamespace(federation=None)) is marker
    assert experiment._cell_provider(config,SimpleNamespace(federation=SimpleNamespace(sites=1,context_policy='legacy'))) is marker
    p=experiment._cell_provider({**config,'audit_dir':str(tmp_path)},
        SimpleNamespace(federation=SimpleNamespace(sites=4,context_policy='federated_v2')))
    assert isinstance(p,VLLMFederationProvider)
    with pytest.raises(ProviderInfrastructureError):usage_receipt({})
