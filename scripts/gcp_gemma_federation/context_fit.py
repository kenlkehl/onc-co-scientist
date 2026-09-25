"""Explicitly approved output allowance amendment for the Gemma continuation.

Nominal requests and replay identities stay unchanged. Only a new request's
max_tokens may change, after exact tokenization on its selected server. The wire
request and both allowances are recorded separately before generation starts.
"""
from datetime import UTC, datetime
import hashlib
import json
import time
from urllib.parse import urlsplit, urlunsplit
import uuid


FLOOR_MARKER = 'CONTEXT_FIT_OUTPUT_FLOOR'


def digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class ContextFit:
    def __init__(self, policy, error_class, write_json):
        expected = dict(enabled=True, approved_by_user=True, nominal_max_tokens=65536,
                        safety_margin_tokens=512,
                        context_window=262144)
        if any(policy.get(k) != v for k, v in expected.items()):
            raise ValueError('Context-fit policy must match the approved allowance and floor')
        floor = policy.get('min_output_tokens')
        amendment = policy.get('floor_amendment', {})
        if floor != 32768 and not (
                floor in {1024, 8192, 16384} and amendment.get('approved_by_user') is True
                and amendment.get('min_output_tokens') == floor
                and amendment.get('authorization') and amendment.get('approved_at')):
            raise ValueError('Context-fit policy must match the approved allowance and floor')
        self.policy = dict(policy)
        compaction = policy.get('json_whitespace')
        if compaction and not (
                compaction.get('approved_by_user') is True
                and compaction.get('algorithm') == 'json-whitespace-v1'
                and compaction.get('authorization') and compaction.get('approved_at')):
            raise ValueError('JSON whitespace compaction requires explicit approval')
        self.error_class = error_class
        self.write = write_json

    def tokenize(self, provider, body, call):
        import requests
        base = urlsplit(provider._config.base_url)
        if not base.path.rstrip('/').endswith('/v1'):
            raise self.error_class('Context-fit requires an explicit vLLM /v1 endpoint')
        url = urlunsplit((base.scheme, base.netloc,
                         base.path.rstrip('/')[:-3] + '/tokenize', '', ''))
        request = {k: body[k] for k in ('model', 'messages', 'tools', 'chat_template_kwargs') if k in body}
        request['add_generation_prompt'] = True
        audit = call / 'context-fit'
        audit.mkdir(exist_ok=True)
        for retry in range(3):
            record = dict(endpoint=url, request_sha256=digest(request), retry=retry,
                          started_at=datetime.now(UTC).isoformat())
            error = None
            try:
                response = requests.post(url, json=request,
                    headers={'Authorization': 'Bearer ' + provider._config.api_key}, timeout=(30, 120))
                record['http_status'] = response.status_code
                retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                if response.status_code == 200:
                    raw = response.json()
                    count, window = raw.get('count'), raw.get('max_model_len')
                    if (type(count) is not int or count < 0 or type(window) is not int
                            or window != self.policy['context_window']):
                        error = 'Context-fit tokenization returned invalid count or context window'
                        raise self.error_class(error)
                    record.update(input_tokens=count, context_window=window)
                    return count, window, record
                error = f'Context-fit tokenization HTTP {response.status_code}'
            except requests.RequestException as exc:
                error = f'Context-fit tokenization transport error: {type(exc).__name__}'
                retryable = True
            except (ValueError, TypeError) as exc:
                error = f'Context-fit tokenization response invalid: {type(exc).__name__}'
                retryable = False
            finally:
                record.update(error=error, ended_at=datetime.now(UTC).isoformat())
                self.write(audit / ('tokenize-' + uuid.uuid4().hex + '.json'), record)
            if not retryable or retry == 2:
                raise self.error_class(error)
            time.sleep(2 ** retry)

    def send(self, original_send, provider, body, call, nominal_body=None):
        nominal_body = body if nominal_body is None else nominal_body
        if {k: v for k, v in body.items() if k != 'model'} != {k: v for k, v in nominal_body.items() if k != 'model'}:
            raise self.error_class('Transport may translate only the served model alias')
        if body.get('max_tokens') != self.policy['nominal_max_tokens']:
            raise self.error_class('Context-fit encountered an unapproved nominal output allowance')
        wire_body = body
        compaction_record = None
        if self.policy.get('json_whitespace'):
            from json_whitespace import compact_messages
            messages, records = compact_messages(body['messages'])
            wire_body = dict(body, messages=messages)
            compaction_record = dict(self.policy['json_whitespace'], messages=records,
                                    nominal_prompt_sha256=digest(body['messages']),
                                    effective_prompt_sha256=digest(messages))
        count, window, tokenization = self.tokenize(provider, wire_body, call)
        effective = min(body['max_tokens'], window - count - self.policy['safety_margin_tokens'])
        changed = effective != body['max_tokens']
        wire = dict(wire_body, max_tokens=effective) if changed else wire_body
        record = dict(policy_id=self.policy['policy_id'], approved_at=self.policy['approved_at'],
            endpoint=provider._config.base_url, input_tokens=count, context_window=window,
            nominal_max_tokens=body['max_tokens'], effective_max_tokens=effective,
            min_output_tokens=self.policy['min_output_tokens'],
            safety_margin_tokens=self.policy['safety_margin_tokens'], adjusted=changed,
            nominal_request_sha256=digest(nominal_body), effective_request_sha256=digest(wire),
            nominal_model=nominal_body.get('model'), served_model=body.get('model'),
            model_alias_adjusted=body.get('model') != nominal_body.get('model'),
            prompt_sha256=digest(body['messages']), tokenization=tokenization,
            recorded_at=datetime.now(UTC).isoformat())
        if compaction_record is not None:
            record['json_whitespace'] = compaction_record
        audit = call / 'context-fit'
        batch = uuid.uuid4().hex
        if effective < self.policy['min_output_tokens']:
            record['status'] = 'paused_output_floor'
            self.write(audit / ('send-' + batch + '.json'), record)
            self.write(call / 'activity.json', dict(status='interrupted',
                error=FLOOR_MARKER, updated_at=datetime.now(UTC).isoformat()))
            raise self.error_class(f'{FLOOR_MARKER}: only {effective} output tokens fit; '
                                   f'minimum is {self.policy["min_output_tokens"]}. No generation sent.')
        # Write the actual amended body before sending. Keep request.json, used
        # for nominal replay identities, intact; unchanged wires need no duplicate.
        wire_path = audit / ('wire-' + batch + '.json') if wire != nominal_body else call / 'request.json'
        if wire != nominal_body:
            self.write(wire_path, wire)
        record.update(status='sending', wire_request=str(wire_path))
        self.write(audit / ('send-' + batch + '.json'), record)
        previous = {p.name for p in call.glob('attempt-*')}
        try:
            # Frozen transport retries and frozen rejection of finish_reason=length
            # remain in force. Do not rewrite or accept a truncated answer here.
            result = original_send(provider, wire, call)
            record['status'] = 'transport_returned'
            return result
        except BaseException:
            record['status'] = 'transport_interrupted'
            raise
        finally:
            record['ended_at'] = datetime.now(UTC).isoformat()
            self.write(audit / ('send-' + batch + '.json'), record)
            for attempt in call.glob('attempt-*'):
                if attempt.name not in previous:
                    self.write(attempt / 'context-fit.json', record)
