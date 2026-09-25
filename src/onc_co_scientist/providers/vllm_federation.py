"""Audited local vLLM transport, opt-in for federated scientific workflows."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..harness.durable_io import atomic_write_json
from .base import ChatResponse, ProviderInfrastructureError
from .federated_prompt import FederatedPromptLayout
from .vllm_openai import VLLMConfig, VLLMProvider


def now():
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class VLLMFederationConfig(VLLMConfig):
    audit_dir: str = ""
    resume_audit: bool = False


def usage_receipt(raw):
    u=raw.get('usage',{})
    inp,out=u.get('prompt_tokens'),u.get('completion_tokens')
    cached=(u.get('prompt_tokens_details') or {}).get('cached_tokens',0)
    if not all(type(x) is int and x>=0 for x in (inp,out,cached)) or cached>inp:
        raise ProviderInfrastructureError('vLLM response omitted valid token usage')
    return dict(input_tokens=inp,output_tokens=out,cached_input_tokens=cached,
        cache_write_input_tokens=0,
        reasoning_output_tokens=(u.get('completion_tokens_details') or {}).get('reasoning_tokens',0) or 0)


class VLLMFederationProvider(VLLMProvider):
    def __init__(self,config):
        if not config.audit_dir:raise ValueError('Federated vLLM requires durable audit directory')
        self._config=config
        self.root=Path(config.audit_dir);self.root.mkdir(parents=True,exist_ok=True)
        self.layout=FederatedPromptLayout(str(self.root));self.lock=threading.Lock()

    def request_body(self,messages,system,max_tokens,*,final_retry=False,truncation_failures=0):
        wire,_,layout=self.layout.render(messages,system,
            'Follow the supplied scientific task and return its requested JSON record. '
            'All evidence is supplied in the messages; an external controller executes analyses. '
            'No external tools are available.')
        systems,content=[],[]
        for m in wire:
            value=m['content'];text=value if isinstance(value,str) else '\n'.join(p['text'] for p in value)
            if m['role'] in {'system','developer'}:systems.append(text)
            elif content and content[-1]['role']==m['role']:
                content[-1]['content']+='\n'+text
            else:content.append(dict(role=m['role'],content=text))
        options=self.generation_options(final_retry=final_retry,truncation_failures=truncation_failures)
        extra=options.pop('extra_body',{})
        body=dict(model=self.model_id,messages=[dict(role='system',content='\n\n'.join(systems)),*content],
            max_tokens=max_tokens,tool_choice='none',**options,**extra)
        if self._config.reasoning_effort:body['reasoning_effort']=self._config.reasoning_effort
        if self._config.service_tier:body['service_tier']=self._config.service_tier
        return body,layout

    def _send(self,body,call):
        import requests
        offset=max([int(p.name.split('-')[1]) for p in call.glob('attempt-*')] or [0])
        for index in range(self._config.max_retries+1):
            attempt=call/f'attempt-{offset+index+1:04d}';attempt.mkdir()
            atomic_write_json(call/'activity.json',dict(status='requesting',updated_at=now(),attempt=attempt.name))
            started=time.monotonic()
            try:
                response=requests.post(self._config.base_url.rstrip('/')+'/chat/completions',
                    json=body,headers={'Authorization':'Bearer '+self._config.api_key},timeout=(30,self._config.timeout_s))
                atomic_write_json(attempt/'http.json',dict(status=response.status_code,ended_at=now()))
                if response.status_code==200:
                    raw=response.json();atomic_write_json(attempt/'response.json',raw)
                    usage=usage_receipt(raw)
                    atomic_write_json(attempt/'metrics.json',dict(backend='vllm',usage=usage,
                        estimated_cost_usd=0,cost_basis='Local endpoint; infrastructure/electricity costs not estimated',
                        duration_seconds=time.monotonic()-started,ended_at=now()))
                    return raw,usage,offset+index+1
                error=f'vLLM HTTP {response.status_code}: {response.text[:1500]}'
                retryable=response.status_code in {408,429,500,502,503,504}
            except ProviderInfrastructureError:raise
            except Exception as exc:
                error=f'vLLM transport error: {type(exc).__name__}';retryable=True
            atomic_write_json(attempt/'error.json',dict(error=error,retryable=retryable,ended_at=now()))
            if not retryable or index==self._config.max_retries:
                atomic_write_json(call/'activity.json',dict(status='interrupted',error=error,updated_at=now()))
                raise ProviderInfrastructureError(error)
            delay=min(60,2**index)
            atomic_write_json(call/'activity.json',dict(status='retry_wait',error=error,retry_in_seconds=delay,updated_at=now()))
            time.sleep(delay)
        raise AssertionError('unreachable')

    def chat(self,messages,*,temperature=0.0,max_tokens=65536,system=None,**kwargs):
        return self.chat_for_call(messages,call_identity=uuid.uuid4().hex,
            temperature=temperature,max_tokens=max_tokens,system=system,**kwargs)

    def chat_for_call(self,messages,*,call_identity,temperature=0.0,max_tokens=65536,system=None,
                      final_retry=False,truncation_failures=0):
        with self.lock:
            body,layout=self.request_body(messages,system,max_tokens,
                final_retry=final_retry,truncation_failures=truncation_failures)
            digest=hashlib.sha256(json.dumps([call_identity,body],sort_keys=True).encode()).hexdigest()
            call=self.root/('call-'+digest);call.mkdir(exist_ok=True)
            p=call/'request.json'
            if p.exists() and json.loads(p.read_text())!=body:
                raise ProviderInfrastructureError('vLLM request identity collision')
            atomic_write_json(p,body)
            atomic_write_json(call/'identity.json',dict(participant_call_identity=call_identity))
            atomic_write_json(call/'layout.json',layout)
            cached=call/'completion.json'
            if cached.exists():
                value=json.loads(cached.read_text())
                return ChatResponse(text=value['text'],model_id=self.model_id,raw=value['raw'])
            raw,usage,attempts=self._send(body,call)
            choice=(raw.get('choices') or [{}])[0];message=choice.get('message',{})
            text=message.get('content') or '';error=None
            if message.get('tool_calls'):error='Unexpected tool call in tool-free vLLM response'
            elif choice.get('finish_reason')=='length':error='Output token limit reached; incomplete generation rejected'
            elif choice.get('finish_reason')!='stop' or not text:error='vLLM returned no usable answer'
            metrics=dict(backend='vllm',usage=usage,infrastructure_attempts=attempts,
                model_version=raw.get('model'),estimated_cost_usd=0)
            atomic_write_json(call/'metrics.json',metrics)
            result=dict(metrics=metrics,adapter_error=error)
            atomic_write_json(cached,dict(text=text,raw=result))
            atomic_write_json(call/'activity.json',dict(status='completed',updated_at=now(),error=error))
            return ChatResponse(text=text,model_id=self.model_id,raw=result)
