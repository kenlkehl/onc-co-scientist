"""Transport-only admission across cloud replicas and an optional shared server."""
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import threading
import time


class HybridPool:
    def __init__(self, endpoints, config_path):
        if not endpoints or len({e['url'] for e in endpoints}) != len(endpoints):
            raise ValueError('Distinct endpoints required')
        self.endpoints = endpoints
        self.config_path = Path(config_path)
        self.condition = threading.Condition()
        self.active = [0] * len(endpoints)
        self.completed = [0] * len(endpoints)
        self.failed = [0] * len(endpoints)
        self.cooldown = [0.] * len(endpoints)
        self.waiting = []
        self.sequence = 0
        self.limits = [int(e['limit']) for e in endpoints]
        self.last_read = 0.

    def reload(self):
        if time.monotonic() - self.last_read < 1:
            return
        self.last_read = time.monotonic()
        try:
            config = json.loads(self.config_path.read_text())
            limits = [int(config.get('endpoint_limits', {}).get(e['id'],
                          config.get('local_requests', e['limit']) if e['kind'] == 'local'
                          else config.get('requests_per_gpu', e['limit']))) for e in self.endpoints]
            if all(0 <= value <= 64 for value in limits):
                self.limits = limits
        except (OSError, ValueError, TypeError):
            pass  # Retain the last valid admission limits during an atomic update.

    def snapshot(self):
        with self.condition:
            return dict(waiting=len(self.waiting), endpoints=[dict(
                id=e['id'], kind=e['kind'], url=e['url'], limit=self.limits[i],
                active=self.active[i], completed=self.completed[i], failed=self.failed[i],
                cooling_down=time.monotonic() < self.cooldown[i])
                for i, e in enumerate(self.endpoints)])

    @contextmanager
    def slot(self, priority=0):
        with self.condition:
            ticket = (priority, self.sequence)
            self.sequence += 1
            self.waiting.append(ticket)
            try:
                while True:
                    self.reload()
                    available = [i for i in range(len(self.endpoints))
                                 if self.active[i] < self.limits[i]
                                 and time.monotonic() >= self.cooldown[i]]
                    if available and ticket == min(self.waiting):
                        # Fill the additional server's explicitly bounded capacity,
                        # then balance cloud work by admitted load.
                        index = min(available, key=lambda i: (
                            self.endpoints[i]['kind'] != 'local',
                            self.active[i] / max(1, self.limits[i]), i))
                        self.waiting.remove(ticket)
                        self.active[index] += 1
                        self.condition.notify_all()
                        break
                    self.condition.wait(timeout=1)
            except BaseException:
                self.waiting.remove(ticket)
                self.condition.notify_all()
                raise
        failed = False
        try:
            yield index
        except Exception:
            failed = True
            raise
        finally:
            with self.condition:
                self.active[index] -= 1
                if failed:
                    self.failed[index] += 1
                    self.cooldown[index] = time.monotonic() + 60
                else:
                    self.completed[index] += 1
                self.condition.notify_all()


def install_routes(provider_class, selected, root, pool, write_json, context_fit=None):
    priorities = {str((root / r['condition'] / 'runs' / r['run_id'] / 'provider_audit').resolve()):
                  (0 if r['replicate'] <= 2 else 1) for r in selected}
    original_send = provider_class._send

    def send(self, body, call):
        # A deferred run reaching transport is a scheduling error, not permission
        # to spend on that run. Successful saved calls bypass transport entirely.
        priority = priorities[str(self.root.resolve())]
        write_json(call / 'activity.json', dict(status='waiting_for_hybrid_admission',
                   priority=priority, updated_at=datetime.now(UTC).isoformat()))
        with pool.slot(priority) as index:
            endpoint = pool.endpoints[index]
            previous = {p.name for p in call.glob('attempt-*')}
            provenance = dict(endpoint, endpoint=endpoint['url'],
                prior_attempts_preserved=True, parallel_sites=True,
                admitted_at=datetime.now(UTC).isoformat(),
                requests_at_admission=pool.limits[index])
            provenance['quantization'] = endpoint.get('quantization', 'NVFP4')
            write_json(call / 'transport-migration.json', provenance)
            # Sites share a provider. Never change its config while a different
            # site's request is in flight or sleeping before a retry.
            transport = copy(self)
            transport._config = replace(self._config, base_url=endpoint['url'])
            wire = dict(body, model=endpoint['served_model']) if endpoint.get('served_model') else body
            try:
                if context_fit is not None:
                    return context_fit.send(original_send, transport, wire, call, nominal_body=body)
                if wire != body:
                    raise ValueError('Model alias translation requires context-fit wire auditing')
                return original_send(transport, wire, call)
            finally:
                for attempt in call.glob('attempt-*'):
                    if attempt.name not in previous:
                        write_json(attempt / 'deployment.json', provenance)

    provider_class._send = send
    return priorities
