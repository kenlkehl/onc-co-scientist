"""Resume frozen scientific journals using a separately audited transport migration.

By default only transport URLs and worker scheduling change. The optional approved
context-fit policy amends new output allowances with separate wire provenance.
Nominal call IDs, prompts, sampling, science code and saved responses stay intact.
The approved FP8-to-NVFP4 transition is recorded for every new provider attempt.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def install_routes(provider_class, grid, root, deployment, write_json, limiter=None):
    routes = {str((root / row['condition'] / 'runs' / row['run_id'] / 'provider_audit').resolve()): i % 8
              for i, row in enumerate(grid)}
    endpoints = deployment['endpoints']
    if len(endpoints) != 8 or len(set(endpoints)) != 8:
        raise ValueError('Require eight distinct GPU endpoints')
    original_send = provider_class._send

    def send(self, body, call):
        gpu = routes[str(self.root.resolve())]
        previous = {p.name for p in call.glob('attempt-*')}
        # URL is excluded from the request body and therefore does not invalidate
        # exact successful-call replay. Every new attempt gets explicit provenance.
        self._config = replace(self._config, base_url=endpoints[gpu])
        write_json(call / 'transport-migration.json', dict(deployment=deployment,
                   gpu=gpu, migrated_at=datetime.now(UTC).isoformat()))
        admitted_limit = None
        try:
            if limiter is not None:
                write_json(call / 'activity.json', dict(status='waiting_for_gpu_admission',
                           updated_at=datetime.now(UTC).isoformat(), gpu=gpu))
            with limiter.slot(gpu) if limiter is not None else nullcontext():
                admitted_limit = limiter.limit if limiter is not None else None
                return original_send(self, body, call)
        finally:
            for attempt in call.glob('attempt-*'):
                if attempt.name not in previous:
                    write_json(attempt / 'deployment.json', dict(
                        model_repository=deployment['model_repository'],
                        model_revision=deployment['model_revision'],
                        quantization='NVFP4', gpu=gpu, endpoint=endpoints[gpu],
                        instance=deployment['instance'], zone=deployment['zone'],
                        container_image=deployment['container_image'],
                        prior_attempts_preserved=True,
                        parallel_sites=limiter is not None,
                        requests_per_gpu_at_admission=admitted_limit))

    provider_class._send = send
    return routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=120)
    parser.add_argument('--stop-instance-on-exit', action='store_true')
    parser.add_argument('--parallel-sites', action='store_true')
    parser.add_argument('--requests-per-gpu', type=int, default=20)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--hybrid-endpoints', type=Path)
    parser.add_argument('--context-fit-policy', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / 'PAUSE').exists():
        raise ValueError('Experiment is paused; explicitly release before launch')
    if not 1 <= args.workers <= 120:
        raise ValueError('Workers must be between 1 and 120')
    if not 1 <= args.requests_per_gpu <= 64:
        raise ValueError('Request limit must be between 1 and 64 per GPU')
    manifest = json.loads((root / 'frozen_manifest.json').read_text())
    path = root / 'source/scripts/expected_surprising/gemini_federation_grid.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['hashes'][str(path.relative_to(root))]:
        raise ValueError('Frozen driver changed')
    spec = importlib.util.spec_from_file_location('frozen_gemma_driver', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    from onc_co_scientist.providers.vllm_federation import VLLMFederationProvider
    import onc_co_scientist.providers.vllm_federation as provider_module
    if not Path(provider_module.__file__).is_relative_to(root / 'source'):
        raise ValueError('Use frozen source on PYTHONPATH')
    deployment = json.loads(args.deployment.read_text())
    grid = json.loads((root / 'grid.json').read_text())
    selection = None
    if args.selection or args.hybrid_endpoints:
        if not (args.selection and args.hybrid_endpoints and args.parallel_sites):
            raise ValueError('Hybrid deadline mode requires selection, endpoints, and parallel sites')
        from deadline_scheduler import validate_selection
        selection = json.loads(args.selection.read_text())
        validate_selection(grid, selection)
    context_fit = None
    if args.context_fit_policy:
        if selection is None:
            raise ValueError('Context-fit is restricted to the selected hybrid cohort')
        from context_fit import ContextFit
        from onc_co_scientist.providers.base import ProviderInfrastructureError
        context_fit = ContextFit(json.loads(args.context_fit_policy.read_text()),
                                 ProviderInfrastructureError, driver.atomic_write_json)
    limiter = None
    cleanup = lambda: None
    if args.parallel_sites:
        from parallel_sites import PoolLimiter, install_parallel_sites
        import parallel_sites
        from onc_co_scientist.expected_surprising import federation, coordination
        path = root / 'control/gcp-nvfp4/concurrency.json'
        if not path.exists():
            driver.atomic_write_json(path, dict(requests_per_gpu=args.requests_per_gpu))
        limiter = PoolLimiter(path, default=args.requests_per_gpu)
        cleanup = install_parallel_sites(federation, coordination, VLLMFederationProvider,
                                         site_workers=args.workers * 4)
        driver.atomic_write_json(root / 'control/gcp-nvfp4/parallel-schedule.json', dict(
            activated_at=datetime.now(UTC).isoformat(), approved_by_user=True,
            independent_sites_parallel=True, central_waits_for_all_sites=True,
            site_handoff_order_preserved=True, peers_remain_sequential=True,
            scientific_files_unchanged=True, replay_saved_successes=True,
            shared_call_budget_synchronized=True,
            scheduler_sha256=hashlib.sha256(Path(parallel_sites.__file__).read_bytes()).hexdigest()))
    if selection:
        from hybrid_pool import HybridPool, install_routes as install_hybrid_routes
        hybrid = HybridPool(json.loads(args.hybrid_endpoints.read_text()),
                            root / 'control/gcp-nvfp4/concurrency.json')
        if context_fit:
            # The rollout drains requests before restarting. Preserve cumulative
            # counters so monitoring does not interpret the restart as lost work.
            prior_runtime = driver.read(root / 'control/gcp-nvfp4/hybrid-runtime.json', {})
            prior_endpoints = {e['id']: e for e in prior_runtime.get('endpoints', [])}
            if prior_endpoints:
                if (set(prior_endpoints) != {e['id'] for e in hybrid.endpoints}
                        or any(prior_endpoints[e['id']]['url'] != e['url'] for e in hybrid.endpoints)):
                    raise ValueError('Cannot restore counters across a different endpoint pool')
                hybrid.completed = [prior_endpoints[e['id']]['completed'] for e in hybrid.endpoints]
                hybrid.failed = [prior_endpoints[e['id']]['failed'] for e in hybrid.endpoints]
        admitted = [r for r in selection['target'] if r['replicate'] <= selection.get('execution_repeat_limit', 3)]
        routes = install_hybrid_routes(VLLMFederationProvider, admitted, root,
                                      hybrid, driver.atomic_write_json, context_fit=context_fit)
    else:
        routes = install_routes(VLLMFederationProvider, grid, root, deployment,
                                driver.atomic_write_json, limiter)
    driver.atomic_write_json(root / 'transport_migration.json', dict(
        approved_by_user=True, created_at=datetime.now(UTC).isoformat(),
        prior_model='RedHatAI/Gemma-4-31B-IT-FP8-Dynamic', deployment=deployment,
        scientific_source_unchanged=True, replay_saved_successes=True,
        cohort_note='Continuation: saved FP8/NVFP4 history; current per-attempt transport identifies server and weights',
        current_endpoints=hybrid.endpoints if selection else None,
        workers=args.workers, routes=routes,
        parallel_sites=args.parallel_sites,
        selection=str(args.selection) if args.selection else None,
        hybrid_endpoints=str(args.hybrid_endpoints) if args.hybrid_endpoints else None,
        context_fit_policy=context_fit.policy if context_fit else None,
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    # Preserve the original scorer/refresh, adding a visible deployment annotation.
    original_refresh = driver.refresh
    def refresh(root, cache=None):
        snapshot = original_refresh(root, cache)
        policy = json.loads(args.deployment.read_text())
        spending = (f"Cloud spending cap: **${policy['budget_cap_usd']:,.0f}**. "
                    if policy.get('budget_guard_enabled', True) else
                    'Cloud spending cap and runtime cutoff: **removed by user**. '
                    'The VM will shut down when the run finishes. ')
        file = root / 'LIVE_PROGRESS.md'
        text = file.read_text().replace('Endpoint: **camus:8060**',
            'Endpoint: **GCP profile-notes · 8 × RTX PRO 6000 · NVIDIA NVFP4**')
        text = text.replace('Local endpoint: no per-token API charge is assumed; infrastructure and electricity costs are not estimated.',
            spending + 'Infrastructure is billed by VM runtime; per-response $0 entries below are not total cost. '
            'This continues saved FP8 histories with NVFP4 responses; per-attempt deployment records identify the switch.')
        if selection:
            count = 12 * selection.get('execution_repeat_limit', 3)
            text = (f'> **September 24 cohort active:** {count} selected runs, '
                    f'24-run priority core (repeats 1–2), {120 - count} deferred. The archive tables below '
                    'retain the original 120-run denominators. '
                    f'[Current selected-cohort status]({root / "CLOUD_LIVE_PROGRESS.md"}).\n\n' + text)
        driver.atomic_write_text(file, text)
        return snapshot
    driver.refresh = refresh
    original_score = driver.score
    def score(root):
        original_score(root)
        report = root / 'analysis/REPORT.md'
        driver.atomic_write_text(report, report.read_text() + '\n\n'
            '**Model deployment:** This is a continuation cohort. Earlier saved responses '
            'used RedHatAI FP8 weights on camus; subsequent responses used NVIDIA NVFP4 '
            'on eight GCP RTX PRO 6000 replicas. It is not a uniform-NVFP4 comparison. '
            'See transport_migration.json and each provider attempt’s deployment.json. '
            'GPU infrastructure costs are separate from per-response token accounting.\n')
    driver.score = score
    try:
        if selection:
            from deadline_scheduler import run_selected
            run_selected(root, args.workers, selection, driver, hybrid)
        else:
            driver.run(root, args.workers)
    finally:
        cleanup()
        if args.stop_instance_on_exit:
            result = subprocess.run(['gcloud', 'compute', 'instances', 'stop', deployment['instance'],
                '--project='+deployment['project'], '--zone='+deployment['zone'], '--quiet'],
                capture_output=True, text=True, timeout=180)
            driver.atomic_write_json(root / 'control/gcp-nvfp4/stop-on-exit.json', dict(
                updated_at=datetime.now(UTC).isoformat(), returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr))


if __name__ == '__main__':
    main()
