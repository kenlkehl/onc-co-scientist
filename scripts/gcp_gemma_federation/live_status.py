"""Publish cloud activity without rescanning the scientific response archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json
from pathlib import Path
import time
import urllib.request
try:
    from recovery_state import combined
except ModuleNotFoundError:
    from scripts.gcp_gemma_federation.recovery_state import combined


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write(path, content):
    temp = path.with_suffix('.tmp')
    temp.write_text(content)
    temp.replace(path)


def active_local_endpoints(control):
    config = read(control / 'concurrency.json')
    return [e for e in read(control / 'hybrid-endpoints.json') or []
            if e['kind'] == 'local' and int(config.get('endpoint_limits', {}).get(
                e['id'], config.get('local_requests', e.get('limit', 0)))) > 0]


def local_label(endpoint_id):
    return endpoint_id.replace('local-', 'Local ', 1)


def replay_position(folder, restarted_at):
    """Read a bounded tail of the new reconstruction transcript, not old calls."""
    paths = sorted(folder.glob('science-*/transcript.jsonl'), key=lambda p: p.parent.name)
    if not paths:
        return None
    path = paths[-1]
    try:
        with path.open('rb') as handle:
            size = path.stat().st_size
            handle.seek(max(0, size - 2 * 1024 * 1024))
            lines = handle.read().splitlines()
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue  # The first/last line can be partial during a write.
            if (record.get('timestamp', '') >= restarted_at and
                    isinstance(record.get('iteration'), int) and record.get('stage')):
                return dict(iteration=record['iteration'], stage=record['stage'],
                            updated_at=record['timestamp'])
    except OSError:
        pass
    return None


def spending_text(deployment, budget):
    if deployment.get('cloud_status') == 'stopped':
        policy = (f"**GCP VM stopped at {deployment['cloud_stopped_at']}.** "
                  'Selected work continues locally. VM and disk are retained; storage charges can continue.')
    elif deployment.get('cloud_status') == 'draining':
        policy = '**GCP shutdown requested:** finish already-admitted requests, then stop the VM. New work uses the local server.'
    elif not deployment.get('budget_guard_enabled', True):
        policy = '**Spending cap and runtime cutoff removed by user.** The VM will shut down when the run finishes.'
    else:
        policy = (f"**Spending cap: ${deployment['budget_cap_usd']:,.0f}.** "
                  f"Automatic stop deadline: {deployment['automatic_stop_deadline']}.")
    return (f"{policy} Conservative runtime estimate: "
            f"${budget.get('conservative_runtime_estimate_usd', 0):,.2f}. "
            f"This uses a ${deployment['conservative_hourly_ceiling_usd']:g}/hour envelope, "
            'not an actual Spot billing statement; ancillary charges are separate.')


def update_detailed_spending(root, deployment):
    # The existing controller can keep running with its original in-memory
    # report annotation. Correct that annotation without restarting workers.
    if deployment.get('budget_guard_enabled', True):
        return
    path = root / 'LIVE_PROGRESS.md'
    if path.exists():
        old = path.read_text()
        new = old.replace('Cloud spending cap: **$2,000**.',
            'Cloud spending cap and runtime cutoff: **removed by user**. '
            'The VM will shut down when the run finishes.')
        if deployment.get('cloud_status') in {'draining', 'stopped'}:
            locals_ = active_local_endpoints(root / 'control/gcp-nvfp4')
            label = ' + '.join(e['url'].removeprefix('http://').removesuffix('/v1') for e in locals_)
            label = label or 'local server (no new requests admitted)'
            new = new.replace('Endpoint: **GCP profile-notes · 8 × RTX PRO 6000 · NVIDIA NVFP4**',
                              f'Endpoint: **{label}**')
            if label in {'camus:8060', 'camus:8002'}:
                new = new.replace('Endpoint: **sn4622130540:8000 · NVIDIA NVFP4**',
                                  f'Endpoint: **{label} · RedHatAI FP8**')
                if label == 'camus:8002':
                    new = new.replace('Endpoint: **camus:8060 · RedHatAI FP8**',
                                      'Endpoint: **camus:8002 · RedHatAI FP8**')
            message = ('The GCP VM is stopped; selected work continues locally.'
                       if deployment['cloud_status'] == 'stopped' else
                       'Already-admitted GCP requests are draining before VM shutdown; new work uses the local server.')
            new = new.replace('The VM will shut down when the run finishes.', message)
            if deployment['cloud_status'] == 'stopped':
                new = new.replace('Already-admitted GCP requests are draining before VM shutdown; new work uses the local server.', message)
        if new != old:
            temp = path.with_suffix('.policy.tmp')
            temp.write_text(new)
            temp.replace(path)


def probe(item):
    gpu, endpoint = item
    try:
        url = endpoint.removesuffix('/v1') + '/metrics'
        with urllib.request.urlopen(url, timeout=8) as response:
            lines = response.read().decode().splitlines()
        values = {}
        for name in ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
                     'num_preemptions_total', 'generation_tokens_total', 'prompt_tokens_total',
                     'prompt_tokens_cached_total', 'request_success_total'):
            values[name] = sum(float(line.rsplit(' ', 1)[1]) for line in lines
                               if line.startswith('vllm:' + name + '{'))
        return dict(gpu=gpu, available=True, **values)
    except Exception as exc:
        return dict(gpu=gpu, available=False, error=str(exc))


def cohort_lines(root, state, progress):
    selection = read(root / 'control/gcp-nvfp4/selection.json')
    if not selection or not state.get('selected_count'):
        return []
    active = {(r['condition'], r['run_id']) for r in state.get('active', [])}
    finished = {(r['condition'], r['run_id']): r['status'] for r in state.get('selected_finished', [])}
    deferred_status = {(r['condition'], r['run_id']): r['status'] for r in state.get('finished', [])}
    snapshots = {r['source']: r for r in progress.get('runs', [])}
    checkpoint = read(root / 'control/gcp-nvfp4/deadline-work-checkpoint.json')
    journal_progress = {r['run_id']: r for r in checkpoint.get('runs', [])}
    handoff = read(root / 'control/gcp-nvfp4/camus-handoff.json')
    recovery = read(root / 'control/gcp-nvfp4/recovery-min-output/execution.json')
    recovery_ids = {r['run_id'] for r in recovery.get('active', [])}
    target_identities = {(r['condition'], r['run_id']) for r in selection['target']}
    target_status = {**deferred_status, **finished}
    completed = sum(target_status.get(key) == 'completed' for key in target_identities)
    failed = sum(target_status.get(key) == 'failed' for key in target_identities)
    interrupted = sum(target_status.get(key) == 'interrupted' for key in target_identities)
    core_done = sum(finished.get((r['condition'], r['run_id'])) == 'completed' for r in selection['core'])
    cloud_status = read(root / 'control/gcp-nvfp4/deployment.json').get('cloud_status')
    cloud_note = ('The GCP VM is stopped; selected work continues on the shared local server.'
                  if cloud_status == 'stopped' else
                  'Already-admitted cloud requests are draining before GCP shutdown; new work uses the local server.'
                  if cloud_status == 'draining' else
                  'The dedicated GCP VM shuts down when this selected cohort finishes; the shared local server stays running.')
    lines = ['', f'**Deadline cohort: {completed}/36 completed · priority core {core_done}/24 completed · '
             f'{failed} failed · {interrupted} interrupted · '
             f"{len(active)} active · {state.get('deferred_count', 84)} deferred with saved work retained.**", '',
             'Completed means all required stages succeeded. Failed runs can still finish all scheduled '
             'iterations and produce scientific reports; their validation failures remain in the results.', '',
             'Preliminary results needed **September 24, noon EDT**. Target scoring snapshot: '
             '**September 24, 9 a.m. EDT**; report target **11 a.m. EDT**. ' +
             ('The noon fallback is active: finish the fixed 24-run core and preserve unfinished third repeats.'
              if selection.get('execution_repeat_limit', 3) == 2 else
              'Keep the 36-run target with the balanced 24-run core prioritized; the proposed restart was canceled.'), '',
             'All selected runs keep four random sites, 25 iterations, original prompts and sampling. '
             'Core requests receive admission priority. ' + cloud_note, '',
             f'[Deadline plan]({root / "analysis/deadline_20260924/PLAN.md"}) · '
             f'[Selected-cohort scores]({root / "analysis/deadline_20260924/REPORT.md"})', '',
             '| Workflow | Condition | Repeat | Cohort | Status | Current replay / latest saved activity |',
             '|---|---|---:|---|---|---|']
    for row in selection['target']:
        ident = row['condition'], row['run_id']
        folder = root / row['condition'] / 'runs' / row['run_id']
        status = 'active' if ident in active else finished.get(ident, deferred_status.get(ident, 'queued'))
        condition = row['semantic_condition'] + '-' + ('unmasked' if row['condition'] == 'named' else 'masked')
        activity = snapshots.get(str(folder), {}).get('progress', 'replaying saved history')
        saved = journal_progress.get(row['run_id'])
        if status == 'active' and saved and saved.get('phase') == 'replaying_saved_history':
            status = 'replaying saved history'
        if saved and checkpoint.get('updated_at', '') > progress.get('updated_at', ''):
            activity = (f"i{saved['latest_central_iteration']:03d}-{saved['latest_central_stage']}; "
                        f"{saved['baseline_remaining']} baseline calls left (checkpoint {checkpoint['updated_at']})")
        if status == 'replaying saved history' and handoff.get('launched_at'):
            restarted_at = (recovery['started_at'] if row['run_id'] in recovery_ids
                            else handoff['launched_at'])
            position = replay_position(folder, restarted_at)
            if position:
                activity = (f"Rebuilding i{position['iteration']:03d}-{position['stage']}; "
                            f"saved through i{saved['latest_central_iteration']:03d}-{saved['latest_central_stage']}. "
                            f"Replay advanced {position['updated_at']}")
        lines.append(f"| {row['workflow_id']} | {condition} | {row['replicate']} | "
                     f"{'core' if row['replicate'] <= 2 else 'third repeat'} | {status} | {activity} |")
    return lines


def experiment_request_lines(runtime, alive=True):
    if not runtime.get('endpoints'):
        return []
    lines = ['**Model requests from this experiment only**', '',
             'Each experiment makes many model requests across its sites and stages. '
             'Requests can use different servers; these are not counts of experiments assigned to each server.', '',
             '| Server | Our in-flight model requests | Our request limit | Recorded responses (cumulative) |',
             '|---|---:|---:|---:|']
    for endpoint in runtime['endpoints']:
        label = local_label(endpoint['id']) if endpoint['kind'] == 'local' else endpoint['id']
        lines.append(f"| {label} | {endpoint['active'] if alive else 0} | "
                     f"{endpoint['limit']} | {endpoint['completed']} |")
    lines += ['', 'In-flight counts come from this experiment’s dispatcher and include requests awaiting '
              'a response. The server-wide health table below includes traffic from every client.', '']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--watch-paused', action='store_true',
                        help='Keep publishing after controllers exit; never restart experiment work.')
    args = parser.parse_args()
    root = args.root.resolve()
    control = root / 'control/gcp-nvfp4'
    deployment = read(control / 'deployment.json')
    previous = None
    previous_preemptions = None
    last_backoff = 0
    while True:
        now = datetime.now(UTC)
        deployment = read(control / 'deployment.json')
        state = read(root / 'execution.json')
        budget = read(control / 'budget-status.json')
        progress = read(root / 'progress.json')
        maintenance = read(control / 'budget-removal.json')
        try:
            command = Path(f'/proc/{args.pid}/cmdline').read_bytes().decode()
            alive = str(root) in command and 'resume_on_pool.py' in command
        except OSError:
            alive = False
        cloud_stopped = deployment.get('cloud_status') == 'stopped'
        local_only = cloud_stopped or deployment.get('cloud_status') == 'draining'
        if cloud_stopped:
            servers = [dict(gpu=i, available=False, intentionally_stopped=True)
                       for i, _ in enumerate(deployment['endpoints'])]
        else:
            with ThreadPoolExecutor(max_workers=8) as pool:
                servers = list(pool.map(probe, enumerate(deployment['endpoints'])))
        concurrency = read(control / 'concurrency.json')
        # This optional policy only reduces the authorized trial's admission rate.
        # Existing requests finish normally; no scientific call is cancelled.
        if all(s['available'] for s in servers):
            preemptions = sum(s['num_preemptions_total'] for s in servers)
            if (concurrency.get('automatic_backoff') and previous_preemptions is not None
                    and preemptions - previous_preemptions >= 3
                    and concurrency.get('requests_per_gpu', 15) > 15
                    and time.monotonic() - last_backoff > 60):
                concurrency['requests_per_gpu'] = max(15, concurrency['requests_per_gpu'] - 2)
                concurrency['last_backoff_at'] = now.isoformat()
                concurrency['last_backoff_reason'] = 'At least three GPU preemptions since the prior sample'
                write(control / 'concurrency.json', json.dumps(concurrency, indent=2) + '\n')
                last_backoff = time.monotonic()
            previous_preemptions = preemptions
        # Local counters must not trigger a GCP admission backoff. This endpoint
        # is shared and its counters can include work outside this experiment.
        hybrid_runtime = read(control / 'hybrid-runtime.json')
        state, hybrid_runtime, recovery_alive = combined(root, state, hybrid_runtime)
        alive = alive or recovery_alive
        local_endpoints = active_local_endpoints(control)
        for endpoint in local_endpoints:
            servers.append(probe((endpoint['id'], endpoint['url'])))
        expected_servers = [s for s in servers if not s.get('intentionally_stopped')]
        output = sum(s.get('generation_tokens_total', 0) for s in expected_servers)
        rate = None
        if previous and all(s['available'] for s in expected_servers) and output >= previous[1]:
            rate = (output - previous[1]) / (now.timestamp() - previous[0])
        previous = (now.timestamp(), output) if all(s['available'] for s in expected_servers) else None
        active = len(state.get('active', [])) if alive else 0
        requests_label = (f"{sum(e['active'] for e in hybrid_runtime['endpoints'])} experiment requests admitted"
                          if hybrid_runtime else
                          f"{sum(s.get('num_requests_running', 0) for s in servers):.0f} server requests currently running")
        controller_label = 'running' if alive else 'stopped'
        if alive and maintenance.get('status') in {
                'controller_suspended_for_vm_scheduling_update',
                'runtime_limit_removed_waiting_for_vm'}:
            controller_label = 'suspended for VM scheduling maintenance'
        local_names = ', '.join(e['url'].removeprefix('http://').removesuffix('/v1') for e in local_endpoints)
        deployment_label = (f'**Local-only execution:** {local_names or "admission draining"}. The eight-GPU GCP VM is stopped.'
                            if cloud_stopped else
                            '**GCP is draining for shutdown.** All new requests use sn4622130540:8000.'
                            if local_only else
                            'GCP `profile-notes` · Spot · eight RTX PRO 6000 GPUs · one NVIDIA Gemma 4 31B NVFP4 replica per GPU.')
        lines = ['# Gemma federation: live progress', '',
                 f'Updated {now.isoformat()} · refreshes every 30 seconds.', '',
                 f"**Controller: {controller_label} · {active}/{state.get('selected_count', 120)} experiment workers assigned · "
                 f"{requests_label}.**", '',
                 *experiment_request_lines(hybrid_runtime, alive),
                 deployment_label, '',
                 'Saved answers are replayed; each new attempt records its server and weight format. '
                 'Workers replaying history or processing site aggregates may have no model request in flight.', '',
                 spending_text(deployment, budget), '',
                 '**Server-wide health — all clients, including other workloads**', '',
                 '| Server | Health | All clients: running requests | All clients: queued requests | KV cache | All clients: returned requests | Output tokens | Preemptions |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        if concurrency:
            local_limit = sum(e['limit'] for e in hybrid_runtime.get('endpoints', []) if e['kind'] == 'local')
            lines[6:6] = ([f"**Local request allocation:** up to {local_limit} concurrent requests across the experiment workers. "
                          'Independent sites run concurrently; central steps wait for their handoffs. Cloud admission is zero.', '']
                         if local_only else [f"**Parallel-site trial:** up to {concurrency['requests_per_gpu']} requests per GPU; "
                          'independent sites run concurrently, then the central step waits for all handoffs. '
                          'Admission automatically backs off if GPU preemptions accumulate.', ''])
        for s in servers:
            label = f"GCP-{s['gpu']}" if isinstance(s['gpu'], int) else local_label(s['gpu'])
            if s['available']:
                lines.append(f"| {label} | healthy | {s['num_requests_running']:.0f} | "
                             f"{s['num_requests_waiting']:.0f} | {s['kv_cache_usage_perc']:.1%} | "
                             f"{s['request_success_total']:,.0f} | {s['generation_tokens_total']:,.0f} | "
                             f"{s['num_preemptions_total']:.0f} |")
            elif s.get('intentionally_stopped'):
                lines.append(f"| {label} | stopped by request | 0 | 0 | — | — | — | — |")
            else:
                lines.append(f"| {label} | unavailable | — | — | — | — | — | — |")
        lines += ['', 'Server counters include startup probes and reset if a replica restarts. '
                  'Finished requests are individual model calls, not completed experiments.']
        eta = read(control / 'deadline-eta.json')
        if eta.get('summary'):
            lines += ['', '**Deadline estimate**', '', eta['summary'], '',
                      f"[Timing assumptions and calculation]({root / 'analysis/deadline_20260924/ETA.md'})"]
        context_fit = read(control / 'context-fit-policy.json')
        if context_fit.get('json_whitespace', {}).get('approved_by_user'):
            lines += ['', '**Approved lossless JSON compaction:** New wire requests remove only '
                      'insignificant JSON whitespace. All fields, values, evidence IDs, narrative strings '
                      'and ordering are retained. Original requests and saved responses are preserved; '
                      'the formatting change and both prompt hashes are audited.']
        if context_fit.get('enabled') and context_fit.get('approved_by_user'):
            primary_floor = read(root / 'transport_migration.json').get('context_fit_policy', {}).get('min_output_tokens')
            floor_note = (f"The original workers retain their {primary_floor:,}-token minimum until handed to recovery. "
                          if primary_floor and primary_floor != context_fit['min_output_tokens'] else
                          'All current workers use this minimum. ')
            lines += ['', '**Approved context-fit output allowance:** Preserve full prompts and sampling; '
                      'reserve up to 65,536 output tokens, reduced only to fit the 262,144-token window '
                      f"with a 512-token margin. Newly launched workers pause below {context_fit['min_output_tokens']:,} available output tokens. " +
                      floor_note +
                      'Adjusted requests are audited; truncated answers remain rejected.']
        if local_endpoints:
            details = '; '.join(f"{e['url'].removeprefix('http://').removesuffix('/v1')} · {e.get('model_repository', 'Gemma 4 31B')} · vLLM {e.get('vllm_version', 'unknown')}" for e in local_endpoints)
            lines += ['', f"**{'Active' if local_only else 'Additional'} server:** {details} · "
                      f"up to {sum(e['limit'] for e in hybrid_runtime.get('endpoints', []) if e['kind'] == 'local')} concurrent experiment requests. "
                      'Saved FP8/NVFP4 responses are retained; the current weight format is recorded per attempt. '
                      'Local server counters may include other workloads.']
        if rate is not None:
            lines += ['', f'Combined server generation rate over the last update interval: **{rate:,.0f} tokens/second**.'
                      + (' This includes other work on the shared local server; it is not this experiment’s throughput.'
                         if local_endpoints else '')]
        archive_status = ('This archive scan covers the original 120 identities; use the selected-cohort counts below for current status. '
                          if state.get('selected_count') else f"Recorded statuses: {progress.get('counts', {})}. ")
        score_report = root / ('analysis/deadline_20260924/REPORT.md' if state.get('selected_count') else 'analysis/REPORT.md')
        lines += ['', f"Latest detailed scientific snapshot: {progress.get('updated_at', 'pending')}. "
                  + archive_status +
                  'The detailed snapshot scans the response archive and can lag this live server view.', '',
                  f"[Detailed experiment progress]({root / 'LIVE_PROGRESS.md'}) · "
                  f"[Scientific scores]({score_report})", '']
        lines += cohort_lines(root, state, progress)
        write(root / 'CLOUD_LIVE_PROGRESS.md', '\n'.join(lines))
        update_detailed_spending(root, deployment)
        snapshot = dict(updated_at=now.isoformat(),
              controller_alive=alive, assigned_workers=active, servers=servers,
              generation_tokens_per_second=rate, budget=budget, concurrency=concurrency)
        snapshot.update(selected_count=state.get('selected_count'),
                        selected_finished=state.get('selected_finished', []), hybrid=hybrid_runtime,
                        cloud_status=deployment.get('cloud_status', 'running'),
                        expected_endpoints_healthy=all(s['available'] for s in expected_servers))
        write(control / 'cloud-live-status.json', json.dumps(snapshot, indent=2) + '\n')
        with (control / 'cloud-live-history.jsonl').open('a') as history:
            history.write(json.dumps(snapshot) + '\n')
        if not alive and not args.watch_paused:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
