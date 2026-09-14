"""Reproducible standard API and active-GPU electricity estimates for the report.

Run standalone or through build_presentation_results.py. Rates are a dated,
editable snapshot in presentation_costs.json; reruns never silently fetch rates.
Terminal API usage is cached against the complete saved run-inventory row.
Use --refresh-costs to reread audits after editing source files in place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_suffix('.json')
CACHE_VERSION = 1


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def fingerprint(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def validate_config(config):
    e = config['electricity']
    for field in ('usd_per_kwh', 'input_tokens_per_second', 'output_tokens_per_second',
                  'gpu_watts', 'gpu_count'):
        value = e[field]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'electricity.{field} must be finite and positive')
    if e['gpu_count'] != int(e['gpu_count']):
        raise ValueError('electricity.gpu_count must be an integer')
    if config['api_tier'] != 'standard':
        raise ValueError('This report requires standard API pricing')
    for rates in config['api_usd_per_million_tokens'].values():
        for field in ('input', 'cached_input', 'cache_write', 'output'):
            if not math.isfinite(rates[field]) or rates[field] < 0:
                raise ValueError(f'Invalid API rate: {field}')


def electricity_cost(input_tokens, output_tokens, assumptions):
    """Return modeled active hours, kWh, USD; add prefill and generation time."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError('Token counts cannot be negative')
    hours = (input_tokens / assumptions['input_tokens_per_second']
             + output_tokens / assumptions['output_tokens_per_second']) / 3600
    kwh = hours * assumptions['gpu_watts'] / 1000 * assumptions['gpu_count']
    return hours, kwh, kwh * assumptions['usd_per_kwh']


def api_cost(usage, model, config):
    """Price token buckets; long-context buckets contain whole qualifying requests."""
    rates = config['api_usd_per_million_tokens'][model]
    total = 0
    for prefix in ('', 'long_'):
        i, c, w, o = (usage.get(prefix + field, 0) for field in
                      ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
                       'output_tokens'))
        if min(i, c, w, o) < 0 or c + w > i:
            raise ValueError('Invalid token buckets')
        input_multiplier = config['openai_long_context_input_multiplier'] if prefix else 1
        output_multiplier = config['openai_long_context_output_multiplier'] if prefix else 1
        total += ((i - c - w) * rates['input'] + c * rates['cached_input']
                  + w * rates['cache_write']) * input_multiplier + o * rates['output'] * output_multiplier
    return total / 1e6


def empty_usage():
    u = {prefix + field: 0 for prefix in ('', 'long_') for field in
         ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens')}
    return dict(u, usage_records=0, missing_usage_records=0, max_input_tokens=0)


def audit_usage(row, threshold):
    """Read known provider invocations, retaining missing usage as a coverage gap."""
    root = Path(row['source'])
    u = empty_usage()
    if row['status'] == 'queued':
        return u
    openai = row['llm'].startswith('gpt-')
    pattern = 'provider_audit/call-*/metrics.json' if openai else 'calls/*.json'
    paths = sorted(root.glob(pattern))
    if not paths and row['status'] in ('completed', 'failed'):
        raise ValueError(f'No API usage audit available: {root}')
    for path in paths:
        data = json.loads(path.read_text())
        raw = data['cli_usage'] if openai else data['result'].get('usage', {})
        i, o = raw.get('input_tokens'), raw.get('output_tokens')
        if i is None or o is None:
            u['missing_usage_records'] += 1
            continue
        c = raw.get('cached_input_tokens', 0) or 0
        w = raw.get('cache_write_input_tokens', 0) or 0
        if min(i, c, w, o) < 0 or c + w > i:
            raise ValueError(f'Invalid token usage: {path}')
        prefix = 'long_' if openai and i > threshold else ''
        for field, value in zip(('input_tokens', 'cached_input_tokens',
                                 'cache_write_input_tokens', 'output_tokens'), (i, c, w, o)):
            u[prefix + field] += value
        u['usage_records'] += 1
        u['max_input_tokens'] = max(u['max_input_tokens'], i)
    return u


def summarize(rows):
    complete = [r for r in rows if r['status'] == 'completed' and r['estimated_cost_usd'] is not None]
    known = [r for r in rows if r['estimated_cost_usd'] is not None]
    return {
        'runs_attempted': sum(r['status'] != 'queued' for r in rows),
        'runs_completed': sum(r['status'] == 'completed' for r in rows),
        'completed_runs_with_cost': len(complete),
        'total_known_cost_usd': sum(r['estimated_cost_usd'] for r in known) if known else None,
        'mean_cost_per_completed_run_usd': statistics.mean(r['estimated_cost_usd'] for r in complete) if complete else None,
        'median_cost_per_completed_run_usd': statistics.median(r['estimated_cost_usd'] for r in complete) if complete else None,
        'input_tokens': sum(r['input_tokens'] for r in rows if r['input_tokens'] is not None),
        'output_tokens': sum(r['output_tokens'] for r in rows if r['output_tokens'] is not None),
        'cached_input_tokens': sum(r['cached_input_tokens'] for r in rows),
        'gpu_kwh': sum(r['gpu_kwh'] or 0 for r in rows),
        'reported_missing_token_calls': sum(r['reported_missing_token_calls'] for r in rows),
        'unaccounted_infrastructure_attempts': sum(r['unaccounted_infrastructure_attempts'] for r in rows),
    }


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(out, config_path=DEFAULT_CONFIG, refresh=False, workers=8):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    raw = (out / 'run_metrics.csv').read_bytes()
    rows = list(csv.DictReader(raw.decode().splitlines()))
    cache_path = out / 'cost_usage_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() and not refresh else {}
    if cache.get('version') != CACHE_VERSION or cache.get('threshold') != config['openai_long_context_threshold']:
        cache = {}
    cached = cache.get('runs', {})
    fresh_cache = {}

    def inspect(row):
        model = row['llm']
        record = {k: row[k] for k in ('llm', 'harness', 'workflow', 'view', 'version', 'run_id', 'status')}
        record.update(input_tokens=None, cached_input_tokens=0, output_tokens=None,
                      gpu_hours=None, gpu_kwh=None, estimated_cost_usd=None,
                      reported_missing_token_calls=int(row['missing_token_calls'] or 0),
                      unaccounted_infrastructure_attempts=int(row['unknown_internal_attempts'] or 0))
        if model in config['api_usd_per_million_tokens']:
            record['cost_basis'] = 'Standard API'
            key = fingerprint(row)
            u = cached.get(key)
            if u is None:
                u = audit_usage(row, config['openai_long_context_threshold'])
            if row['status'] in ('completed', 'failed'):
                fresh_cache[key] = u
            if u['usage_records']:
                record.update(input_tokens=u['input_tokens'] + u['long_input_tokens'],
                              cached_input_tokens=u['cached_input_tokens'] + u['long_cached_input_tokens'],
                              output_tokens=u['output_tokens'] + u['long_output_tokens'],
                              estimated_cost_usd=api_cost(u, model, config))
        elif model in config['electricity']['local_models']:
            record['cost_basis'] = 'GPU electricity'
            if row['input_tokens'] and row['output_tokens']:
                i, o = int(row['input_tokens']), int(row['output_tokens'])
                hours, kwh, usd = electricity_cost(i, o, config['electricity'])
                record.update(input_tokens=i, output_tokens=o, gpu_hours=hours,
                              gpu_kwh=kwh, estimated_cost_usd=usd)
        else:
            raise ValueError(f'No cost assumptions configured for {model}')
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(inspect, rows))
    write_json(cache_path, dict(version=CACHE_VERSION,
                               threshold=config['openai_long_context_threshold'], runs=fresh_cache))
    totals, workflows, conditions = [], [], []
    for model, harness in dict.fromkeys((r['llm'], r['harness']) for r in records):
        selected = [r for r in records if (r['llm'], r['harness']) == (model, harness)]
        base = dict(llm=model, harness=harness, cost_basis=selected[0]['cost_basis'])
        totals.append(dict(base, **summarize(selected)))
        for workflow in dict.fromkeys(r['workflow'] for r in selected):
            group = [r for r in selected if r['workflow'] == workflow]
            workflows.append(dict(base, workflow=workflow, **summarize(group)))
            for view, version in [('unmasked', 'expected'), ('masked', 'expected'),
                                  ('unmasked', 'surprising'), ('masked', 'surprising')]:
                subset = [r for r in group if (r['view'], r['version']) == (view, version)]
                conditions.append(dict(base, workflow=workflow, condition=f'{version}-{view}',
                                       **summarize(subset)))
    payload = dict(assumptions=config, config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                   cohort_sha256=hashlib.sha256(raw).hexdigest(), totals=totals, workflows=workflows,
                   conditions=conditions, known_cost_usd=sum(r['estimated_cost_usd'] or 0 for r in records))
    write_json(out / 'cost_estimates.json', payload)
    for name, data in [('run_costs', records), ('cost_totals', totals),
                       ('cost_by_workflow', workflows), ('cost_by_condition', conditions)]:
        write_csv(out / (name + '.csv'), data)
    print(f'Cost estimates: {len(records)} selected runs; ${payload["known_cost_usd"]:,.2f} known cost', flush=True)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('outputs/presentation_results'))
    parser.add_argument('--cost-assumptions', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--refresh-costs', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    build(args.out.resolve(), args.cost_assumptions, args.refresh_costs, args.workers)
