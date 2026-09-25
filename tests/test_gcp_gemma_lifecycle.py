"""Removing the spending cap must retain completion and explicit-pause shutdown."""
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace


def load(name):
    path = Path(__file__).parents[1] / 'scripts/gcp_gemma_federation' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load('budget_guard')
publisher = load('live_status')


def deployment():
    return dict(created_at='2026-09-01T00:00:00+00:00',
                automatic_stop_deadline='2026-09-03T00:00:00+00:00',
                budget_cap_usd=2000, conservative_hourly_ceiling_usd=40,
                project='profile-notes', instance='dedicated-gemma', zone='us-west1-a')


def test_removed_cap_ignores_old_deadline_but_honors_explicit_pause(tmp_path):
    d = deployment()
    d['budget_guard_enabled'] = False
    now = datetime(2026, 9, 21, tzinfo=UTC)
    status = guard.budget_status(d, now)
    assert status['spending_cap_usd'] is None and status['deadline'] is None
    assert status['conservative_runtime_estimate_usd'] > 2000
    assert guard.pause_reason(tmp_path, status, now) is None
    (tmp_path / 'PAUSE').touch()
    assert guard.pause_reason(tmp_path, status, now) == 'experiment pause requested'


def test_existing_budget_policy_still_stops_at_deadline(tmp_path):
    now = datetime(2026, 9, 21, tzinfo=UTC)
    assert guard.pause_reason(tmp_path, guard.budget_status(deployment(), now), now) == 'budget deadline'


def test_no_cap_still_shuts_down_exact_vm_when_controller_exits(tmp_path, monkeypatch):
    d = deployment()
    d.update(budget_guard_enabled=False, budget_cap_usd=None, automatic_stop_deadline=None)
    control = tmp_path / 'control/gcp-nvfp4'
    control.mkdir(parents=True)
    manifest = control / 'deployment.json'
    manifest.write_text(json.dumps(d))
    monkeypatch.setattr('sys.argv', ['guard', '--root', str(tmp_path), '--deployment', str(manifest),
                                     '--pid', '999999999'])
    stops = []
    monkeypatch.setattr(guard.subprocess, 'run', lambda command, **kwargs:
                        stops.append(command) or SimpleNamespace(returncode=0, stdout='', stderr=''))
    guard.main()
    assert stops == [['gcloud', 'compute', 'instances', 'stop', 'dedicated-gemma',
                      '--project=profile-notes', '--zone=us-west1-a', '--quiet']]
    assert not (tmp_path / 'PAUSE').exists()
    assert json.loads((control / 'guard-stop.json').read_text())['reason'] == 'controller exited'


def test_reports_follow_removed_policy_without_controller_restart(tmp_path):
    d = deployment()
    d.update(budget_guard_enabled=False, budget_cap_usd=None, automatic_stop_deadline=None)
    text = publisher.spending_text(d, {'conservative_runtime_estimate_usd': 340})
    assert '$2,000' not in text and 'removed by user' in text
    assert 'shut down when the run finishes' in text and '$340.00' in text
    path = tmp_path / 'LIVE_PROGRESS.md'
    path.write_text('Cloud spending cap: **$2,000**.\nScientific result retained.\n')
    publisher.update_detailed_spending(tmp_path, d)
    assert 'Scientific result retained.' in path.read_text()
    assert '$2,000' not in path.read_text()


def test_verified_shutdown_freezes_runtime_cost_without_budget_cap():
    d = deployment()
    d.update(budget_guard_enabled=False, cloud_status='stopped',
             cloud_stopped_at='2026-09-01T02:00:00+00:00')
    early = guard.budget_status(d, datetime(2026, 9, 2, tzinfo=UTC))
    later = guard.budget_status(d, datetime(2026, 9, 23, tzinfo=UTC))
    assert early['conservative_runtime_estimate_usd'] == later['conservative_runtime_estimate_usd'] == 80
    assert later['deadline'] is None and later['spending_cap_usd'] is None
    assert later['cloud_vm_status'] == 'stopped'


def test_local_only_publisher_excludes_stopped_cloud_from_probes(tmp_path, monkeypatch):
    d = deployment()
    d.update(budget_guard_enabled=False, cloud_status='stopped',
             cloud_stopped_at='2026-09-01T02:00:00+00:00',
             endpoints=[f'http://127.0.0.1:{18000+i}/v1' for i in range(8)])
    control = tmp_path / 'control/gcp-nvfp4'
    control.mkdir(parents=True)
    local = {'id': 'local-sn4622130540', 'kind': 'local', 'url': 'http://local:8000/v1'}
    for name, data in [('deployment.json', d), ('hybrid-endpoints.json', [local]),
                       ('concurrency.json', {'requests_per_gpu': 0, 'local_requests': 4})]:
        (control / name).write_text(json.dumps(data))
    calls = []
    def probe(item):
        calls.append(item)
        return dict(gpu=item[0], available=True, num_requests_running=4,
                    num_requests_waiting=0, kv_cache_usage_perc=.5,
                    num_preemptions_total=0, generation_tokens_total=100,
                    request_success_total=1)
    monkeypatch.setattr(publisher, 'probe', probe)
    monkeypatch.setattr('sys.argv', ['publisher', '--root', str(tmp_path), '--pid', '999999999'])
    publisher.main()
    assert calls == [(local['id'], local['url'])]
    state = json.loads((control / 'cloud-live-status.json').read_text())
    assert state['expected_endpoints_healthy'] and state['cloud_status'] == 'stopped'
    text = (tmp_path / 'CLOUD_LIVE_PROGRESS.md').read_text()
    assert text.count('stopped by request') == 8
    assert 'Local-only execution' in text and 'unavailable' not in text


def test_detailed_report_moves_from_drain_to_stopped_without_changing_results(tmp_path):
    d = deployment()
    d.update(budget_guard_enabled=False, cloud_status='draining')
    report = tmp_path / 'LIVE_PROGRESS.md'
    report.write_text('Endpoint: **GCP profile-notes · 8 × RTX PRO 6000 · NVIDIA NVFP4**\n'
                      'The VM will shut down when the run finishes.\nScientific result retained.\n')
    publisher.update_detailed_spending(tmp_path, d)
    assert 'draining' in report.read_text()
    d['cloud_status'] = 'stopped'
    publisher.update_detailed_spending(tmp_path, d)
    assert 'The GCP VM is stopped' in report.read_text()
    assert 'draining' not in report.read_text()
    assert 'Scientific result retained.' in report.read_text()
