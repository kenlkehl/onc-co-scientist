"""Reusable provenance checks for NSCLC experiment launch gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path


def require_codex_command_locks(
    command: object,
    *,
    context: Path,
    pinned_executable: str,
    model_id: str,
    reasoning_effort: str,
    service_tier: str,
    required_paths: Sequence[str] = (),
    forbidden_paths: Sequence[str] = (),
    required_environment: Mapping[str, str] | None = None,
) -> None:
    """Reject an audited Codex command that is missing required runtime locks."""
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(argument, str) for argument in command)
    ):
        raise RuntimeError(f"Missing audited command in {context}.")
    if command[0] != pinned_executable:
        raise RuntimeError(f"Unpinned Codex executable in {context}.")
    if f"model_reasoning_effort={reasoning_effort}" not in command:
        raise RuntimeError(f"Missing {reasoning_effort} reasoning lock in {context}.")
    if f'service_tier="{service_tier}"' not in command:
        raise RuntimeError(f"Missing {service_tier} service-tier lock in {context}.")
    try:
        model_flag = command.index("--model")
    except ValueError as exc:
        raise RuntimeError(f"Missing model flag in {context}.") from exc
    if model_flag + 1 >= len(command) or command[model_flag + 1] != model_id:
        raise RuntimeError(f"Incorrect model lock in {context}.")

    # Search the raw arguments. JSON-serializing first escapes quotes inside Codex's
    # inline configuration arguments and causes exact environment locks to be missed.
    command_text = "\n".join(command)
    for path in required_paths:
        if path not in command_text:
            raise RuntimeError(f"Missing required runtime path {path} in {context}.")
    for path in forbidden_paths:
        if path in command_text:
            raise RuntimeError(f"Forbidden runtime path {path} in {context}.")
    for name, value in (required_environment or {}).items():
        assignment = f'"{name}"="{value}"'
        if assignment not in command_text:
            raise RuntimeError(f"Missing environment lock {name}={value} in {context}.")
