"""Retry-safe persistence helpers for experiment artifacts.

Experiment output is sometimes hosted on FUSE/SSHFS mounts.  Those mounts can
briefly return ``EPERM``/``EIO`` or disconnect while an otherwise successful
agent call is being persisted.  A model result must not be lost merely because
the output mount had a transient failure.

Writes are first materialized in a local spool, then published through a unique
temporary file and atomic replace.  Retryable storage failures share a small
process-local circuit breaker so concurrent workers back off together instead
of hammering an unhealthy mount.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCAL_SPOOL_ENV = "OCS_LOCAL_IO_SPOOL_ROOT"

_RETRYABLE_ERRNOS = frozenset(
    value
    for value in (
        errno.EACCES,
        errno.EPERM,
        errno.EBUSY,
        errno.EINTR,
        errno.EIO,
        getattr(errno, "ESTALE", None),
        errno.ETIMEDOUT,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ENOTCONN,
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.EHOSTDOWN,
        errno.EHOSTUNREACH,
    )
    if value is not None
)


class StorageUnavailable(RuntimeError):
    """A bounded storage operation exhausted its retry policy."""


@dataclass(frozen=True)
class StorageRetryPolicy:
    attempts: int = 12
    initial_delay_seconds: float = 0.25
    maximum_delay_seconds: float = 5.0
    deadline_seconds: float = 120.0
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: float = 15.0

    @classmethod
    def from_environment(cls) -> StorageRetryPolicy:
        return cls(
            attempts=int(os.environ.get("OCS_STORAGE_RETRY_ATTEMPTS", "12")),
            initial_delay_seconds=float(
                os.environ.get("OCS_STORAGE_RETRY_INITIAL_SECONDS", "0.25")
            ),
            maximum_delay_seconds=float(
                os.environ.get("OCS_STORAGE_RETRY_MAX_SECONDS", "5")
            ),
            deadline_seconds=float(
                os.environ.get("OCS_STORAGE_RETRY_DEADLINE_SECONDS", "120")
            ),
            circuit_failure_threshold=int(
                os.environ.get("OCS_STORAGE_CIRCUIT_FAILURE_THRESHOLD", "3")
            ),
            circuit_cooldown_seconds=float(
                os.environ.get("OCS_STORAGE_CIRCUIT_COOLDOWN_SECONDS", "15")
            ),
        )


@dataclass
class _CircuitState:
    failures: int = 0
    open_until: float = 0.0


_CIRCUIT_LOCK = threading.Lock()
_CIRCUITS: dict[str, _CircuitState] = {}


def _storage_key(path: Path) -> str:
    absolute = path.absolute()
    if len(absolute.parts) >= 2:
        return os.path.join(absolute.parts[0], absolute.parts[1])
    return str(absolute)


def _circuit_delay(path: Path) -> float:
    with _CIRCUIT_LOCK:
        state = _CIRCUITS.get(_storage_key(path))
        return max(0.0, state.open_until - time.monotonic()) if state else 0.0


def _record_storage_success(path: Path) -> None:
    with _CIRCUIT_LOCK:
        _CIRCUITS[_storage_key(path)] = _CircuitState()


def _record_storage_failure(path: Path, policy: StorageRetryPolicy) -> None:
    with _CIRCUIT_LOCK:
        state = _CIRCUITS.setdefault(_storage_key(path), _CircuitState())
        state.failures += 1
        if (
            policy.circuit_failure_threshold > 0
            and state.failures >= policy.circuit_failure_threshold
        ):
            state.open_until = max(
                state.open_until,
                time.monotonic() + policy.circuit_cooldown_seconds,
            )


def _retryable(exc: OSError) -> bool:
    return exc.errno in _RETRYABLE_ERRNOS or isinstance(exc, PermissionError)


def retry_storage_operation[T](
    operation: Callable[[], T],
    *,
    path: Path,
    operation_name: str,
    policy: StorageRetryPolicy | None = None,
    verify_after_error: Callable[[], bool] | None = None,
) -> T:
    """Run one storage operation with bounded backoff and shared circuit state."""

    selected = policy or StorageRetryPolicy.from_environment()
    if selected.attempts < 1 or selected.deadline_seconds <= 0:
        raise ValueError("Storage retry policy must allow at least one bounded attempt.")
    deadline = time.monotonic() + selected.deadline_seconds
    last_error: OSError | None = None
    attempts_made = 0
    for attempt in range(1, selected.attempts + 1):
        circuit_delay = _circuit_delay(path)
        if circuit_delay:
            time.sleep(min(circuit_delay, max(0.0, deadline - time.monotonic())))
        if time.monotonic() >= deadline and last_error is not None:
            break
        attempts_made += 1
        try:
            result = operation()
        except OSError as exc:
            last_error = exc
            if not _retryable(exc):
                raise
            if verify_after_error is not None:
                try:
                    if verify_after_error():
                        _record_storage_success(path)
                        return None  # type: ignore[return-value]
                except OSError:
                    pass
            _record_storage_failure(path, selected)
            remaining = deadline - time.monotonic()
            if attempt >= selected.attempts or remaining <= 0:
                break
            delay = min(
                selected.maximum_delay_seconds,
                selected.initial_delay_seconds * (2 ** (attempt - 1)),
                remaining,
            )
            if delay > 0:
                time.sleep(delay)
        else:
            _record_storage_success(path)
            return result
    assert last_error is not None
    raise StorageUnavailable(
        f"{operation_name} failed after {attempts_made} attempt(s) for {path}: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def durable_mkdir(path: Path, *, parents: bool = True, exist_ok: bool = True) -> None:
    def is_directory() -> bool:
        return stat.S_ISDIR(path.stat().st_mode)

    retry_storage_operation(
        lambda: path.mkdir(parents=parents, exist_ok=exist_ok),
        path=path,
        operation_name="mkdir",
        verify_after_error=is_directory,
    )


def local_spool_directory() -> Path:
    """Return the host-local spool used before publishing to artifact storage."""

    configured = os.environ.get(LOCAL_SPOOL_ENV, "").strip()
    root = Path(configured) if configured else Path(tempfile.gettempdir()) / "ocs-io-spool"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _digest_matches(path: Path, expected: bytes) -> bool:
    try:
        if path.stat().st_size != len(expected):
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.digest() == hashlib.sha256(expected).digest()
    except OSError:
        return False


def atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o666,
    policy: StorageRetryPolicy | None = None,
) -> None:
    """Publish bytes atomically after first materializing them on local storage."""

    durable_mkdir(path.parent)
    spool_root = local_spool_directory()
    local_fd, local_name = tempfile.mkstemp(prefix="write-", suffix=".spool", dir=spool_root)
    local_path = Path(local_name)
    try:
        with os.fdopen(local_fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        def publish() -> None:
            temporary = path.with_name(
                f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
            )
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                mode,
            )
            with local_path.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
                destination.flush()
            os.replace(temporary, path)

        retry_storage_operation(
            publish,
            path=path,
            operation_name="atomic write",
            policy=policy,
            verify_after_error=lambda: _digest_matches(path, payload),
        )
    finally:
        local_path.unlink(missing_ok=True)


def atomic_write_text(
    path: Path,
    payload: str,
    *,
    encoding: str = "utf-8",
    policy: StorageRetryPolicy | None = None,
) -> None:
    atomic_write_bytes(path, payload.encode(encoding), policy=policy)


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = False,
    policy: StorageRetryPolicy | None = None,
) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n",
        policy=policy,
    )


def durable_append_line(
    path: Path,
    line: str,
    *,
    encoding: str = "utf-8",
    policy: StorageRetryPolicy | None = None,
) -> None:
    """Append one line, checking for an ambiguous successful write before retrying."""

    durable_mkdir(path.parent)
    payload = (line + "\n").encode(encoding)

    def append() -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
        finally:
            os.close(descriptor)

    def already_appended() -> bool:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            if size < len(payload):
                return False
            stream.seek(-len(payload), os.SEEK_END)
            return stream.read() == payload

    retry_storage_operation(
        append,
        path=path,
        operation_name="append",
        policy=policy,
        verify_after_error=already_appended,
    )


def durable_unlink(path: Path, *, missing_ok: bool = True) -> None:
    def is_missing() -> bool:
        try:
            path.stat()
        except FileNotFoundError:
            return True
        return False

    retry_storage_operation(
        lambda: path.unlink(missing_ok=missing_ok),
        path=path,
        operation_name="unlink",
        verify_after_error=is_missing,
    )


def durable_replace(source: Path, destination: Path) -> None:
    def replaced() -> bool:
        return durable_exists(destination) and not durable_exists(source)

    retry_storage_operation(
        lambda: source.replace(destination),
        path=destination,
        operation_name="replace",
        verify_after_error=replaced,
    )


def durable_copy_file(source: Path, destination: Path) -> None:
    payload = durable_read_bytes(source)
    source_mode = retry_storage_operation(
        lambda: source.stat().st_mode & 0o777,
        path=source,
        operation_name="stat for copy",
    )
    atomic_write_bytes(destination, payload, mode=source_mode)


def durable_read_bytes(path: Path) -> bytes:
    return retry_storage_operation(
        path.read_bytes,
        path=path,
        operation_name="read",
    )


def durable_read_text(path: Path, *, encoding: str = "utf-8") -> str:
    return retry_storage_operation(
        lambda: path.read_text(encoding=encoding),
        path=path,
        operation_name="read",
    )


def durable_read_json(path: Path) -> Any:
    return json.loads(durable_read_text(path))


def durable_exists(path: Path) -> bool:
    def inspect() -> bool:
        try:
            path.stat()
        except FileNotFoundError:
            return False
        return True

    return retry_storage_operation(inspect, path=path, operation_name="stat")


def durable_is_file(path: Path) -> bool:
    def inspect() -> bool:
        try:
            return stat.S_ISREG(path.stat().st_mode)
        except FileNotFoundError:
            return False

    return retry_storage_operation(inspect, path=path, operation_name="stat")


def durable_is_dir(path: Path) -> bool:
    def inspect() -> bool:
        try:
            return stat.S_ISDIR(path.stat().st_mode)
        except FileNotFoundError:
            return False

    return retry_storage_operation(inspect, path=path, operation_name="stat")


def storage_health_probe(directory: Path) -> None:
    """Verify create/write/replace/read/unlink before spending a model call."""

    durable_mkdir(directory)
    path = directory / f".ocs-storage-probe-{os.getpid()}-{uuid.uuid4().hex}.json"
    payload = {"schema_version": 1, "pid": os.getpid(), "time": time.time()}
    atomic_write_json(path, payload, sort_keys=True)
    if durable_read_json(path) != payload:
        raise StorageUnavailable(f"Storage probe readback mismatch for {directory}.")
    durable_unlink(path)
