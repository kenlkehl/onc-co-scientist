"""Check model identity, reasoning/JSON and automatic tool parsing on eight GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import requests


def probe(item):
    gpu, endpoint = item
    models = requests.get(endpoint + '/models', timeout=20)
    models.raise_for_status()
    assert any(x['id'] == 'gemma4-31b' for x in models.json()['data'])
    common = dict(model='gemma4-31b', temperature=1.0, top_p=.95, top_k=64,
                  max_tokens=2048, chat_template_kwargs={'enable_thinking': True})
    body = dict(common, messages=[dict(role='user', content='Return a JSON object with sum equal to 2 + 2.')],
                response_format={'type': 'json_object'}, tool_choice='none', reasoning_effort='medium')
    response = requests.post(endpoint + '/chat/completions', json=body, timeout=300)
    response.raise_for_status(); raw = response.json(); choice = raw['choices'][0]
    assert choice['finish_reason'] == 'stop', raw
    assert json.loads(choice['message']['content'])['sum'] == 4, raw
    assert choice['message'].get('reasoning') or choice['message'].get('reasoning_content'), raw
    tool = dict(type='function', function=dict(name='get_site_count', description='Get number of sites.',
                parameters=dict(type='object', properties={}, additionalProperties=False)))
    body = dict(common, messages=[dict(role='user', content='Call get_site_count now to retrieve the number of sites. Do not guess.')],
                tools=[tool], tool_choice='auto')
    response = requests.post(endpoint + '/chat/completions', json=body, timeout=300)
    response.raise_for_status(); tool_raw=response.json()
    calls=tool_raw['choices'][0]['message'].get('tool_calls') or []
    assert calls and calls[0]['function']['name'] == 'get_site_count', tool_raw
    assert isinstance(json.loads(calls[0]['function']['arguments']),dict)
    return dict(gpu=gpu, endpoint=endpoint, models=models.json(), json_reasoning=raw, auto_tool=tool_raw)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    deployment=json.loads(args.deployment.read_text())
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(probe,enumerate(deployment['endpoints'])))
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print('All eight endpoints passed model, reasoning/JSON and automatic tool parser checks.')
