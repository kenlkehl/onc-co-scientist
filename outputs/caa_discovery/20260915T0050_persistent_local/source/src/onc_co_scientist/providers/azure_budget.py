"""Durable, process-shared spending admission for opt-in Azure experiments."""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path


class ExperimentPaused(BaseException):
    """Cancellation, deliberately outside scientific protocol/format retry handlers."""


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def amount(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid monetary amount") from exc
    if not number.is_finite() or number < 0:
        raise ValueError("Amounts must be finite and nonnegative")
    return number


def dollars_to_micro(value):
    return int((amount(value) * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


def normalize_usage(usage):
    """Require all billable categories; missing usage is never free usage."""
    details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    result = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cached_input_tokens": details.get("cached_tokens"),
        "cache_write_input_tokens": details.get("cache_write_tokens"),
        "reasoning_output_tokens": output_details.get("reasoning_tokens", 0),
    }
    if any(type(v) is not int or v < 0 for v in result.values()):
        raise ValueError("Missing or invalid Azure token accounting")
    if (
        result["cached_input_tokens"] + result["cache_write_input_tokens"] > result["input_tokens"]
        or result["reasoning_output_tokens"] > result["output_tokens"]
    ):
        raise ValueError("Inconsistent Azure token accounting")
    return result


def usage_cost_micro(usage, rates):
    long = usage["input_tokens"] > rates["short_context_tokens"]
    tariff = rates["long" if long else "short"]
    cached, written = usage["cached_input_tokens"], usage["cache_write_input_tokens"]
    ordinary = usage["input_tokens"] - cached - written
    # USD / million tokens is numerically micro-USD / token.
    total = sum(
        amount(tariff[k]) * n
        for k, n in (
            ("input", ordinary),
            ("cached", cached),
            ("write", written),
            ("output", usage["output_tokens"]),
        )
    )
    return int(total.to_integral_value(rounding=ROUND_CEILING))


class AzureBudget:
    def __init__(self, policy_path):
        self.policy_path = Path(policy_path).resolve()
        self.state_path = self.policy_path.with_name("spend_state.json")

    @contextmanager
    def locked(self):
        try:
            with self.policy_path.with_name("spend.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.policy = json.loads(self.policy_path.read_text())
                self.state = json.loads(self.state_path.read_text())
                if (
                    type(self.state["spent_micro_usd"]) is not int
                    or self.state["spent_micro_usd"] < 0
                ):
                    raise ValueError("Invalid settled spending")
                for attempt in self.state["attempts"].values():
                    if (
                        type(attempt["reserved_micro_usd"]) is not int
                        or attempt["reserved_micro_usd"] < 0
                        or attempt["status"] not in {"reserved", "unknown", "settled", "rejected"}
                    ):
                        raise ValueError("Invalid spending reservation")
                yield
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ExperimentPaused("Azure spending ledger unavailable or invalid") from exc

    def save(self):
        self.state["updated_at"] = datetime.now(UTC).isoformat()
        atomic_json(self.state_path, self.state)

    def check(self, model):
        with self.locked():
            self._check(model)

    def _check(self, model):
        if self.policy.get("enabled") is not True or self.state.get("hold_reason"):
            raise ExperimentPaused(self.state.get("hold_reason") or "Azure spending gate is held")
        if datetime.now(UTC) >= datetime.fromisoformat(self.policy["rates_valid_until"]):
            raise ExperimentPaused("Azure price verification expired")
        release = json.loads(Path(self.policy["release_policy_path"]).read_text())
        profile = self.policy["model_profiles"][model]
        if profile not in release["released_models"]:
            raise ExperimentPaused(f"{profile} is held by the experiment release policy")
        # Validate every configured tariff before any inference.
        for band in ("short", "long"):
            for field in ("input", "cached", "write", "output"):
                if amount(self.policy["rates"][model][band][field]) <= 0:
                    raise ValueError("Azure tariff must be positive")
        if dollars_to_micro(self.policy["additional_budget_usd"]) <= self.state["spent_micro_usd"]:
            raise ExperimentPaused("Additional Azure spending budget is exhausted")
        for field in ("max_unknown_attempts", "max_reusable_prefix_misses"):
            if type(self.policy[field]) is not int or self.policy[field] < 1:
                raise ValueError("Spending circuit breakers must be positive integers")
        rates = self.policy["rates"][model]
        for field in ("input", "cached", "write", "output"):
            if amount(rates["long"][field]) < amount(rates["short"][field]):
                raise ValueError("Long-context reservation cannot be lower than short-context cost")

    def reserve(self, request_id, model, body):
        with self.locked():
            self._check(model)
            if request_id in self.state["attempts"]:
                raise ExperimentPaused("Request already reserved; refusing duplicate dispatch")
            rates = self.policy["rates"][model]
            # A deliberately conservative byte bound, including message framing/hidden context.
            # Reserve the output ceiling, never the adaptive throughput estimate.
            input_bound = len(json.dumps(body, ensure_ascii=False).encode()) + 32768
            if input_bound + body["max_output_tokens"] > rates["context_window"]:
                raise ExperimentPaused("Request exceeds conservative context reservation bound")
            tariff = rates["long"]
            worst_input_rate = max(amount(tariff[k]) for k in ("input", "write", "cached"))
            reserved = int(
                (
                    input_bound * worst_input_rate
                    + body["max_output_tokens"] * amount(tariff["output"])
                ).to_integral_value(rounding=ROUND_CEILING)
            )
            used = self.state["spent_micro_usd"] + sum(
                a["reserved_micro_usd"]
                for a in self.state["attempts"].values()
                if a["status"] in {"reserved", "unknown"}
            )
            if used + reserved > dollars_to_micro(self.policy["additional_budget_usd"]):
                # Do not latch a global hold: other in-flight requests may release reservations.
                raise ExperimentPaused("Additional Azure budget cannot cover this request")
            self.state["attempts"][request_id] = dict(
                model=model,
                status="reserved",
                reserved_micro_usd=reserved,
                input_bound=input_bound,
                output_bound=body["max_output_tokens"],
                rates=rates,
                at=datetime.now(UTC).isoformat(),
            )
            self.save()
            return reserved

    def rejected(self, request_id):
        """Only explicit non-inference rejections (401/403/429) release reservations."""
        with self.locked():
            attempt = self.state["attempts"][request_id]
            if attempt["status"] != "reserved":
                raise ExperimentPaused("Cannot release a settled request")
            attempt["status"] = "rejected"
            self.save()

    def unknown(self, request_id, reason):
        with self.locked():
            attempt = self.state["attempts"][request_id]
            if attempt["status"] == "reserved":
                attempt.update(status="unknown", reason=reason)
            unknown = sum(a["status"] == "unknown" for a in self.state["attempts"].values())
            if unknown >= self.policy["max_unknown_attempts"]:
                self.state["hold_reason"] = "Unknown Azure usage reached the configured limit"
            self.save()

    def settle(self, request_id, usage, cache_prefix):
        with self.locked():
            attempt = self.state["attempts"][request_id]
            if attempt["status"] != "reserved":
                raise ExperimentPaused("Request accounting already finalized")
            cost = usage_cost_micro(usage, attempt["rates"])
            attempt.update(status="settled", cost_micro_usd=cost, usage=usage)
            self.state["spent_micro_usd"] += cost
            if (
                cost > attempt["reserved_micro_usd"]
                or usage["input_tokens"] > attempt["input_bound"]
                or usage["output_tokens"] > attempt["output_bound"]
            ):
                self.state["hold_reason"] = "Azure usage exceeded the conservative reservation"
            if cache_prefix:
                key = attempt["model"] + ":" + cache_prefix
                history = self.state.setdefault("cache_prefixes", {}).setdefault(
                    key, {"seen": 0, "misses": 0}
                )
                if usage["cached_input_tokens"]:
                    history["misses"] = 0
                elif history["seen"]:
                    history["misses"] += 1
                history["seen"] += 1
                if history["misses"] >= self.policy["max_reusable_prefix_misses"]:
                    self.state["hold_reason"] = "Repeated reusable prefixes received no cache reads"
            self.save()
            # Return the received science result. The hold applies to the next dispatch.
            return cost
