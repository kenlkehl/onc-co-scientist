from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from onc_co_scientist.harness.durable_io import (
    StorageRetryPolicy,
    atomic_write_text,
    durable_append_line,
    durable_is_file,
    retry_storage_operation,
    storage_health_probe,
)


def _fast_policy(*, attempts: int = 3) -> StorageRetryPolicy:
    return StorageRetryPolicy(
        attempts=attempts,
        initial_delay_seconds=0.0,
        maximum_delay_seconds=0.0,
        deadline_seconds=2.0,
        circuit_failure_threshold=0,
        circuit_cooldown_seconds=0.0,
    )


def test_retry_storage_operation_recovers_from_transient_permission_error(
    tmp_path: Path,
) -> None:
    calls = 0

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError(errno.EPERM, "transient FUSE rejection")
        return "accepted"

    assert retry_storage_operation(
        operation,
        path=tmp_path / "artifact.json",
        operation_name="test write",
        policy=_fast_policy(),
    ) == "accepted"
    assert calls == 2


def test_atomic_write_retries_publish_from_local_spool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "remote" / "artifact.json"
    spool = tmp_path / "local-spool"
    monkeypatch.setenv("OCS_LOCAL_IO_SPOOL_ROOT", str(spool))
    real_replace = os.replace
    replacements = 0

    def flaky_replace(source: Path | str, destination: Path | str) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 1:
            raise PermissionError(errno.EPERM, "transient FUSE rejection")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", flaky_replace)
    monkeypatch.setenv("OCS_STORAGE_RETRY_INITIAL_SECONDS", "0")
    monkeypatch.setenv("OCS_STORAGE_RETRY_MAX_SECONDS", "0")
    monkeypatch.setenv("OCS_STORAGE_CIRCUIT_FAILURE_THRESHOLD", "0")

    atomic_write_text(target, "complete\n")

    assert target.read_text(encoding="utf-8") == "complete\n"
    assert replacements == 2
    assert not list(spool.glob("write-*.spool"))


def test_durable_append_and_health_probe(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    durable_append_line(events, '{"event": 1}', policy=_fast_policy())
    durable_append_line(events, '{"event": 2}', policy=_fast_policy())

    assert events.read_text(encoding="utf-8").splitlines() == [
        '{"event": 1}',
        '{"event": 2}',
    ]
    assert durable_is_file(events)
    storage_health_probe(tmp_path / "probe")
    assert not list((tmp_path / "probe").glob(".ocs-storage-probe-*"))
