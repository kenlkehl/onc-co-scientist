import pytest
from types import SimpleNamespace
from onc_co_scientist.expected_surprising.research import json_response
from onc_co_scientist.providers.vllm_openai import VLLMProvider, VLLMConfig

@pytest.mark.parametrize('tag', ['think','thinking'])
def test_explicit_reasoning_boundary(tag):
    assert json_response('{"draft":1}\n</'+tag+'>\n```json\n{"final":2}\n```') == {'final':2}

def test_does_not_guess_final_or_modify_string():
    for text in ['{"a":1}\n{"b":2}', '<think>unfinished {"a":1}', '{"a":1}\n</thinking>\n']:
        with pytest.raises(ValueError): json_response(text)
    assert json_response('{"narrative":"</thinking>"}') == {'narrative':'</thinking>'}

def test_final_retry_thinking_option(monkeypatch):
    requests=[]
    def create(**kw):
        requests.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
    monkeypatch.setattr(VLLMProvider,'_build_client',lambda _:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    p=VLLMProvider(VLLMConfig(model_id='test',reasoning_effort='medium',disable_thinking_on_final_retry=True))
    p.chat_for_retry([],final_retry=False)
    p.chat_for_retry([],final_retry=True)
    assert 'extra_body' not in requests[0]
    assert requests[1]['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    assert requests[1]['reasoning_effort']=='medium'

def test_coordinator_records_and_routes_final_retry(tmp_path):
    from onc_co_scientist.expected_surprising.coordination import StageCoordinator
    from onc_co_scientist.harness.experiment import ResourceBudget, WorkflowSpec, default_stages
    from onc_co_scientist.providers.base import ChatResponse
    flags=[]
    class Provider:
        model_id='test'
        def chat(self,*a,**k): raise AssertionError('use retry-aware path')
        def chat_for_retry(self,messages,*,final_retry,**kwargs):
            flags.append(final_retry)
            return ChatResponse(text='{}',model_id='test')
    source=SimpleNamespace(max_retries_per_stage=2,max_tokens_per_call=100,persistent_history_chars=10000)
    coord=StageCoordinator(Provider(),WorkflowSpec(id='sequential',mode='sequential'),default_stages(),source,ResourceBudget(),tmp_path/'calls')
    coord.respond('initial',iteration=1,stage='explore',attempt=1)
    coord.reject()
    coord.respond('error feedback',iteration=1,stage='explore',attempt=3)
    assert flags==[False,True]
    assert [r['request']['final_retry'] for r in coord.records]==[False,True]
