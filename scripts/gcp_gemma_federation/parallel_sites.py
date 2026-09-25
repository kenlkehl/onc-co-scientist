"""Opt-in scheduling transforms for the frozen Gemma cloud continuation.

Scientific method bodies are derived from the verified frozen source, rather than
forked. Only site scheduling, shared-budget synchronization and provider lock scope
change. Central calls, site-local peers, repairs and commits remain ordered.
"""
import ast
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import inspect
import json
import threading
import time
import textwrap


def parsed(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def compiled(original, node, additions):
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(original.__globals__, **additions)
    exec(compile(module, '<gemma-cloud-parallel-sites>', 'exec'), namespace)
    return namespace[node.name]


def install_parallel_sites(federation, coordination, provider_class, *, site_workers=480):
    """Return a cleanup callback; changes apply only in this opt-in process."""
    cls = federation.FederatedCoordinator
    original_init, original_respond = cls.__init__, cls.respond
    original_call = coordination.StageCoordinator._call
    original_chat = provider_class.chat_for_call
    original_request_body = provider_class.request_body
    pool = ThreadPoolExecutor(max_workers=site_workers, thread_name_prefix='federated-site')

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.shared_budget['_parallel_lock'] = threading.RLock()
        self.shared_budget['_parallel_reserved'] = set()

    def call_key(self, slot):
        return str(self.calls_dir), slot

    def reserve(self, slot):
        budget = self.shared_budget
        if budget is None or '_parallel_lock' not in budget:
            used = budget['calls'] if budget is not None else len(self.records)
            if used >= self.budget.max_agent_calls:
                raise ValueError('Run exhausted max_agent_calls')
            return
        with budget['_parallel_lock']:
            pending = budget['_parallel_reserved']
            if budget['calls'] + len(pending) >= self.budget.max_agent_calls:
                raise ValueError('Run exhausted max_agent_calls')
            pending.add(call_key(self, slot))

    def count(self, slot):
        budget = self.shared_budget
        if '_parallel_lock' not in budget:
            budget['calls'] += 1
            return
        with budget['_parallel_lock']:
            budget['calls'] += 1
            budget['_parallel_reserved'].discard(call_key(self, slot))

    # Reserve the shared call allowance atomically before network I/O. Cached calls
    # retain the original accounting and do not reserve another call.
    call_node = parsed(original_call)
    class BudgetTransform(ast.NodeTransformer):
        reservations = increments = 0
        def visit_Assign(self, node):
            if len(node.targets) == 1 and ast.unparse(node.targets[0]) == 'used':
                self.reservations += 1
                return ast.parse('_parallel_reserve(self, slot)').body[0]
            return self.generic_visit(node)
        def visit_If(self, node):
            if ast.unparse(node.test) == 'used >= self.budget.max_agent_calls':
                return None
            return self.generic_visit(node)
        def visit_AugAssign(self, node):
            if ast.unparse(node.target) == "self.shared_budget['calls']":
                self.increments += 1
                return ast.parse('_parallel_count(self, slot)').body[0]
            return self.generic_visit(node)
    transformer = BudgetTransform()
    call_node = transformer.visit(call_node)
    if (transformer.reservations, transformer.increments) != (1, 1):
        raise ValueError('Unexpected frozen budget method shape')
    budgeted_call = compiled(original_call, call_node, dict(
        _parallel_reserve=reserve, _parallel_count=count))

    def call(self, slot, *args, **kwargs):
        if self.shared_budget is None or '_parallel_lock' not in self.shared_budget:
            return original_call(self, slot, *args, **kwargs)
        try:
            return budgeted_call(self, slot, *args, **kwargs)
        finally:
            with self.shared_budget['_parallel_lock']:
                self.shared_budget['_parallel_reserved'].discard(call_key(self, slot))

    # Each site's complete repair loop runs independently. Gather in original site
    # order so the central prompt and error audit do not depend on finish order.
    respond_node = parsed(original_respond)
    loops = [n for n in respond_node.body if isinstance(n, ast.For)
             and ast.unparse(n.target) == '(site, coordinator)']
    if len(loops) != 1:
        raise ValueError('Unexpected frozen federation site loop')
    loop = loops[0]
    if (ast.unparse(loop.body[0].test) != "site in entry['handoffs']"
            or ast.unparse(loop.body[-2]) != "entry['handoffs'][site] = handoff"
            or ast.unparse(loop.body[-1]) != 'self._write_handoff(key, site, handoff)'):
        raise ValueError('Unexpected frozen handoff merge shape')
    body = loop.body[1:-2]
    class ErrorTransform(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if ast.unparse(node) == 'self.handoff_errors':
                return ast.Name(id='_site_errors', ctx=ast.Load())
            return self.generic_visit(node)
    body = [ErrorTransform().visit(n) for n in body]
    helper = ast.parse('def _consult(site, coordinator):\n    pass').body[0]
    helper.body = ast.parse('_site_errors = []').body + body + ast.parse(
        'return handoff, _site_errors').body
    replacement = [helper] + ast.parse('''
_site_jobs = [(site, _parallel_site_pool.submit(_consult, site, coordinator))
              for site, coordinator in self.sites.items() if site not in entry['handoffs']]
_site_results = []
_site_error = None
for site, future in _site_jobs:
    try:
        _site_results.append((site, future.result()))
    except Exception as error:
        if _site_error is None:
            _site_error = error
if _site_error is not None:
    raise _site_error
for site, (handoff, errors) in _site_results:
    self.handoff_errors.extend(errors)
    entry['handoffs'][site] = handoff
    self._write_handoff(key, site, handoff)
''').body
    index = respond_node.body.index(loop)
    respond_node.body[index:index+1] = replacement
    respond = compiled(original_respond, respond_node, dict(_parallel_site_pool=pool))

    # Keep mutable catalog rendering serialized, but only serialize a complete
    # request against another request with the SAME durable participant identity.
    def request_body(self, *args, **kwargs):
        with self.lock:
            return original_request_body(self, *args, **kwargs)

    def call_lock(self, identity):
        with self.lock:
            if not hasattr(self, '_parallel_call_locks'):
                self._parallel_call_locks = {}
            return self._parallel_call_locks.setdefault(identity, threading.Lock())

    chat_node = parsed(original_chat)
    if len(chat_node.body) != 1 or not isinstance(chat_node.body[0], ast.With):
        raise ValueError('Unexpected frozen provider lock shape')
    if ast.unparse(chat_node.body[0].items[0].context_expr) != 'self.lock':
        raise ValueError('Unexpected frozen provider lock')
    chat_node.body[0].items[0].context_expr = ast.parse(
        '_parallel_call_lock(self, call_identity)', mode='eval').body
    chat = compiled(original_chat, chat_node, dict(_parallel_call_lock=call_lock))

    cls.__init__, cls.respond = init, respond
    coordination.StageCoordinator._call = call
    provider_class.request_body, provider_class.chat_for_call = request_body, chat

    def cleanup():
        pool.shutdown(wait=True)
        cls.__init__, cls.respond = original_init, original_respond
        coordination.StageCoordinator._call = original_call
        provider_class.request_body, provider_class.chat_for_call = original_request_body, original_chat
    return cleanup


class PoolLimiter:
    """FIFO per-GPU admission; limits can change between requests without restart."""
    def __init__(self, config_path, *, default=20, gpu_count=8):
        self.path = config_path
        self.limit = default
        self.active = [0] * gpu_count
        self.waiting = [deque() for _ in range(gpu_count)]
        self.condition = threading.Condition()
        self.last_read = 0

    def reload(self):
        if time.monotonic() - self.last_read < 1:
            return
        self.last_read = time.monotonic()
        try:
            value = json.loads(self.path.read_text())['requests_per_gpu']
            if type(value) is int and 1 <= value <= 64:
                self.limit = value
        except (OSError, ValueError, KeyError):
            pass

    @contextmanager
    def slot(self, gpu):
        ticket = object()
        acquired = False
        try:
            with self.condition:
                self.waiting[gpu].append(ticket)
                while True:
                    self.reload()
                    if self.waiting[gpu][0] is ticket and self.active[gpu] < self.limit:
                        self.waiting[gpu].popleft()
                        self.active[gpu] += 1
                        acquired = True
                        self.condition.notify_all()
                        break
                    self.condition.wait(timeout=1)
            yield
        finally:
            with self.condition:
                if acquired:
                    self.active[gpu] -= 1
                elif ticket in self.waiting[gpu]:
                    self.waiting[gpu].remove(ticket)
                self.condition.notify_all()
