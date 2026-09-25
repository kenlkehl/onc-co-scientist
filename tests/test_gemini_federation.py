import json
from types import SimpleNamespace

import pytest

from onc_co_scientist.providers.gemini_federation import (
    GeminiFederationConfig, GeminiFederationProvider, usage_receipt,
)
from onc_co_scientist.providers.base import ProviderInfrastructureError
from tests.test_federated_context_v2 import SYSTEM, payload, message


def provider(monkeypatch, tmp_path):
    monkeypatch.setattr('onc_co_scientist.providers.gemini_federation.GeminiVertexClient',
        lambda config: SimpleNamespace(base_url='https://example.invalid',credentials=SimpleNamespace(valid=True)))
    return GeminiFederationProvider(GeminiFederationConfig(audit_dir=str(tmp_path),max_retries=2))


def test_federated_request_is_tool_free_and_preserves_evidence(monkeypatch,tmp_path):
    p=provider(monkeypatch,tmp_path)
    body,meta=p.request_body([message(payload())],SYSTEM,125000)
    assert body['generationConfig']==dict(maxOutputTokens=65536,
        thinkingConfig=dict(thinkingLevel='MEDIUM'),responseMimeType='application/json')
    assert not {'tools','cachedContent'} & body.keys()
    assert meta['current_ledger_roundtrip']
    text=json.dumps(body)
    assert '-0.12345678901234' in text and 'R1' in text and 'H1' in text
    other=payload('site_2');other['claims'][0]['evidence'][0]['ref']='R9'
    next_body,_=p.request_body([message(other)],SYSTEM,65536)
    assert 'R9' in json.dumps(next_body)
    assert '"ref": "R9"' not in json.dumps(body)
    with pytest.raises(ValueError,match='single-site'):
        p.request_body([message(payload())],'ordinary scientist',100)


def test_durable_completion_replay_and_reasoning_accounting(monkeypatch,tmp_path):
    p=provider(monkeypatch,tmp_path);calls=[]
    raw=dict(modelVersion='gemini-3.8-flash',candidates=[dict(finishReason='STOP',content=dict(parts=[dict(text='{"ok":true}')]))],
        usageMetadata=dict(promptTokenCount=100,candidatesTokenCount=5,thoughtsTokenCount=20,cachedContentTokenCount=50))
    monkeypatch.setattr(p,'_send',lambda body,call:(calls.append(body) or (raw,usage_receipt(raw),1,.0001)))
    a=p.chat_for_call([message(payload())],system=SYSTEM,call_identity='peer1-stage1')
    b=p.chat_for_call([message(payload())],system=SYSTEM,call_identity='peer1-stage1')
    assert a==b and len(calls)==1
    assert a.raw['metrics']['usage']['output_tokens']==25
    assert a.raw['metrics']['usage']['cached_input_tokens']==50
    assert len(list(tmp_path.glob('call-*/request.json')))==1
    p.chat_for_call([message(payload())],system=SYSTEM,call_identity='peer2-stage1')
    assert len(calls)==2
    assert len(list(tmp_path.glob('call-*/request.json')))==2


def test_auth_expiration_and_rate_limit_retry(monkeypatch,tmp_path):
    p=provider(monkeypatch,tmp_path);refreshed=[]
    p.client.credentials=SimpleNamespace(valid=True,apply=lambda headers:None,
        refresh=lambda request:refreshed.append(True))
    raw=dict(usageMetadata=dict(promptTokenCount=12,candidatesTokenCount=5,thoughtsTokenCount=20))
    responses=iter([SimpleNamespace(status_code=c,headers={},json=lambda c=c:raw if c==200 else {'error':{'message':'retry'}}) for c in [401,429,200]])
    monkeypatch.setattr('requests.post',lambda *a,**k:next(responses))
    monkeypatch.setattr('time.sleep',lambda _:None)
    call=tmp_path/'call-test';call.mkdir()
    answer,u,attempts,cost=p._send({},call)
    assert refreshed==[True] and attempts==3 and u['output_tokens']==25
    assert cost==pytest.approx((12*.75+25*3.75)/1e6)
    assert len(list(call.glob('attempt-*/http.json')))==3
    assert 'Authorization' not in ''.join(x.read_text() for x in call.rglob('*.json'))


def test_grid_requires_every_cell_and_repeat():
    from scripts.expected_surprising.gemini_federation_grid import verify_grid,WORKFLOWS,MODEL
    rows=[dict(workflow_id=w,condition=c,semantic_condition=s,replicate=r,run_id=f'{w}-{c}-{s}-{r}',site_count=4,model_id=MODEL,iterations=25)
        for w in WORKFLOWS for c in ('named','masked') for s in ('expected','surprising') for r in range(1,11)]
    verify_grid(rows)
    with pytest.raises(ValueError):verify_grid(rows[:-1])
    rows[-1]['replicate']=9
    with pytest.raises(ValueError):verify_grid(rows)


def test_missing_usage_does_not_become_zero_cost():
    with pytest.raises(ProviderInfrastructureError):usage_receipt({})


def test_coordinator_identical_peer_prompts_sample_independently_then_replay(monkeypatch,tmp_path):
    from onc_co_scientist.expected_surprising.coordination import StageCoordinator
    from onc_co_scientist.providers.base import ChatMessage
    p=provider(monkeypatch,tmp_path/'audit');calls=[]
    raw=dict(candidates=[dict(finishReason='STOP',content=dict(parts=[dict(text='{}')]))],
        usageMetadata=dict(promptTokenCount=20,candidatesTokenCount=5))
    monkeypatch.setattr(p,'_send',lambda body,call:(calls.append(call) or (raw,usage_receipt(raw),1,.0001)))
    coord=StageCoordinator(p,SimpleNamespace(mode='deliberative'),[],
        SimpleNamespace(max_retries_per_stage=2,max_tokens_per_call=65536),
        SimpleNamespace(max_agent_calls=100),tmp_path/'calls')
    messages=[ChatMessage('system',SYSTEM),message(payload())]
    for peer in (1,2,1,2):
        coord._call(f'i001-analyze-peer{peer}-r1-a1',messages,
            session=f'i001-analyze-peer{peer}',authoritative=False,iteration=1,stage='analyze',kind='peer')
    assert len(calls)==2 and calls[0]!=calls[1]
    assert coord.replayed_calls==2
