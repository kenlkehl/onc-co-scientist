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
    p.chat_for_retry([],final_retry=True, truncation_failures=2)
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


def test_qwen_recommended_sampling_and_json_constraint(monkeypatch):
    monkeypatch.setattr(VLLMProvider, '_build_client', lambda _: None)
    p=VLLMProvider(VLLMConfig(model_id='qwen', sampling_profile='qwen3_8', json_object_output=True, disable_thinking_on_final_retry=True))
    normal=p.generation_options()
    fallback=p.generation_options(final_retry=True, truncation_failures=2)
    assert (normal['temperature'],normal['top_p'],normal['presence_penalty'])==(1.0,0.95,0.0)
    assert (fallback['temperature'],fallback['top_p'],fallback['presence_penalty'])==(0.7,0.8,1.5)
    for opts in (normal,fallback):
        assert opts['extra_body']['top_k']==20
        assert opts['extra_body']['min_p']==0.0
        assert opts['extra_body']['repetition_penalty']==1.0
        assert opts['response_format']=={'type':'json_object'}
    assert fallback['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    assert VLLMProvider(VLLMConfig(model_id='gemma')).generation_options()=={}


def test_length_failure_preserves_usage(monkeypatch):
    class Response:
        choices=[SimpleNamespace(message=SimpleNamespace(content='{}'),finish_reason='length')]
        def model_dump(self):return {'usage':{'completion_tokens':100}}
    monkeypatch.setattr(VLLMProvider,'_build_client',lambda _:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:Response()))))
    r=VLLMProvider(VLLMConfig(model_id='test')).chat([])
    assert r.text==''
    assert r.raw['usage']['completion_tokens']==100
    assert 'limit reached' in r.raw['adapter_error']


def test_gemma_defaults_and_truncation_only_fallback(monkeypatch):
    monkeypatch.setattr(VLLMProvider, '_build_client', lambda _: None)
    p=VLLMProvider(VLLMConfig(model_id='RedHatAI/Gemma-4-31B-IT-FP8-Dynamic'))
    for count in (0,1,2):
        o=p.generation_options(final_retry=True,truncation_failures=count)
        assert (o['temperature'],o['top_p'],o['extra_body']['top_k'])==(1.0,0.95,64)
        assert o['extra_body']['chat_template_kwargs']['enable_thinking'] == (count<2)
    assert p.generation_options(final_retry=False,truncation_failures=2)['extra_body']['chat_template_kwargs']['enable_thinking']


def test_coordinator_truncations_are_scoped_and_preserve_usage(tmp_path,monkeypatch):
    from onc_co_scientist.expected_surprising.coordination import StageCoordinator
    from onc_co_scientist.harness.experiment import ResourceBudget, WorkflowSpec, default_stages
    from onc_co_scientist.providers.base import ChatResponse
    monkeypatch.setattr(VLLMProvider, '_build_client', lambda _: None)
    provider=VLLMProvider(VLLMConfig(model_id='Qwen3.8-27B'))
    flags=[]
    def reply(messages,*,final_retry=False,truncation_failures=0,**kwargs):
        flags.append(truncation_failures)
        return ChatResponse(text='',model_id=provider.model_id,raw={'adapter_error':'Output token limit reached; incomplete generation rejected','usage':{'completion_tokens':100}})
    monkeypatch.setattr(provider,'chat_for_retry',reply)
    source=SimpleNamespace(max_retries_per_stage=2,max_tokens_per_call=100,persistent_history_chars=10000)
    c=StageCoordinator(provider,WorkflowSpec(id='sequential',mode='sequential'),default_stages(),source,ResourceBudget(),tmp_path/'calls')
    for a in (1,2,3):
        with pytest.raises(ValueError,match='limit reached'):c.respond('prompt',iteration=1,stage='explore',attempt=a)
    assert flags==[0,1,2]
    assert c.records[-1]['request']['generation_options']['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    assert sum(r['result']['usage']['output_tokens'] for r in c.records)==300
    with pytest.raises(ValueError):c.respond('prompt',iteration=1,stage='analyze',attempt=3)
    assert flags[-1]==0
