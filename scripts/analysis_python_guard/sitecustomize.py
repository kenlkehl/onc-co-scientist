"""Resource guard automatically loaded by controlled analysis interpreters."""

from __future__ import annotations

import os
import resource
import sys
from types import TracebackType
from typing import Any

_LIMIT_ENV = "OCS_ANALYSIS_MEMORY_LIMIT_MB"


def _configured_limit_bytes() -> int | None:
    raw = os.environ.get(_LIMIT_ENV, "").strip()
    if not raw:
        return None
    try:
        megabytes = int(raw)
    except ValueError:
        return None
    return megabytes * 1024 * 1024 if megabytes > 0 else None


def _install_address_space_limit(limit_bytes: int) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    finite = [value for value in (soft, hard) if value != resource.RLIM_INFINITY]
    effective = min([limit_bytes, *finite])
    resource.setrlimit(resource.RLIMIT_AS, (effective, effective))


def _install_memory_error_marker(limit_bytes: int) -> None:
    previous_hook = sys.excepthook

    def marked_excepthook(
        exception_type: type[BaseException],
        exception: BaseException,
        traceback: TracebackType | None,
    ) -> Any:
        if issubclass(exception_type, MemoryError):
            limit_mb = limit_bytes // (1024 * 1024)
            print(
                f"OCS_ANALYSIS_RESOURCE_LIMIT: memory ceiling {limit_mb} MiB exceeded; "
                "inspect array shapes and broadcasting before retrying.",
                file=sys.stderr,
            )
        return previous_hook(exception_type, exception, traceback)

    sys.excepthook = marked_excepthook


_limit_bytes = _configured_limit_bytes()
if _limit_bytes is not None:
    try:
        _install_address_space_limit(_limit_bytes)
    except (OSError, ValueError):
        # Some non-Linux Python builds expose RLIMIT_AS without implementing it.
        # The adapter still retains process-tree cleanup and crash retries.
        pass
    else:
        _install_memory_error_marker(_limit_bytes)
