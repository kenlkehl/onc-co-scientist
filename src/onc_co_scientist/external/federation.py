"""External orchestration of persistent, native Biomni scientists at disjoint sites.

Sites own native cognition, code and memory. Only the external orchestrator can
mutate the common benchmark ledger. Trusted statistics implement canonical tests.
"""

from __future__ import annotations

import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..expected_surprising.federation_spec import FederationCell
from ..expected_surprising.packaging import sha256
from ..expected_surprising.prompting import (
    COMMON,
    AnalyzeForm,
    AppraiseForm,
    Judgment,
    Proposal,
    ResearchForm,
    references,
)
from ..expected_surprising.research import json_response
from ..expected_surprising.schemas import StrictModel
from ..expected_surprising.site_statistics import (
    combine_statistics,
    partition_frame,
    public_result,
    public_site_context,
    sufficient_statistics,
)
from .config import ExternalSpec
from .exchanges import Exchange, ExchangeGateway
from .runner import BiomniRunner
from .transport import RunBroker, fingerprint, write_json


class NativeFederationSpec(ExternalSpec):
    schema_version: Literal["external-native-federation-1"] = "external-native-federation-1"
    execution_mode: Literal["external-native-federation"] = "external-native-federation"
    sites: Literal[2, 4] = 2
    partition_seed: int = 20260908
    max_requests: int = Field(default=4200, ge=1)
    max_central_actions_per_round: int = Field(default=40, ge=4)
    central_retries: int = Field(default=2, ge=0, le=5)


class SiteHandoff(StrictModel):
    proposals: list[Proposal] = Field(default_factory=list, max_length=12)
    assessments: list[Judgment] = Field(default_factory=list, max_length=100)
    recommended_analyses: list[str] = Field(default_factory=list, max_length=12)
    recommended_validation: str | None = None
    narrative: str = Field(default="", max_length=8000)


class SiteExchange(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    round: int = Field(ge=1)
    action: Literal["state", "handoff"]
    payload: dict = Field(default_factory=dict)


SITE_TOOL = """Exchange with your site coordinator. Supply request_id, round, action, payload.
Actions are state (payload {}) and handoff (the supplied SiteHandoff JSON schema).
Submit exactly one handoff for this dispatch, then finish with a native <solution>.
The orchestrator alone registers claims, selects canonical analyses, requests independent
validation, assesses the common claims and closes reporting rounds. Your handoff contains
recommendations, not authoritative evidence. Use only common H/R references for assessments.
Do not transmit patient rows, identifiers, raw arrays or file contents in any handoff.
Canonical numerical evidence comes from trusted local statistics, not narrative numbers.
"""


class SiteGateway:
    """No site endpoint can mutate the shared scientific ledger."""

    def __init__(self, round_number, context):
        self.round = round_number
        self.context = context
        self.handoff = None
        self.receipts = {}

    def exchange(self, request):
        req = SiteExchange.model_validate(request)
        canonical = req.model_dump()
        if req.request_id in self.receipts:
            old, response = self.receipts[req.request_id]
            if old != canonical:
                raise ValueError("request_id reused with different content")
            return response
        if req.round != self.round:
            raise ValueError("Wrong site reporting round")
        if req.action == "state":
            if req.payload:
                raise ValueError("state takes no payload")
            response = {"round": self.round, **self.context}
        else:
            if self.handoff is not None:
                raise ValueError("One handoff per dispatch")
            handoff = SiteHandoff.model_validate(req.payload).model_dump()
            self.handoff = handoff
            response = {"round": self.round, "handoff_received": True}
        self.receipts[req.request_id] = canonical, response
        return response


class NativeSite:
    """Dispatch native research, preserving its sandboxed namespace across rounds."""

    def __init__(self, spec, site, public, root, budget, *, runner=None, resume=False):
        self.spec, self.site = spec, site
        self.public, self.root = Path(public), Path(root)
        self.budget, self.resume = budget, resume
        self.runner = runner or BiomniRunner()
        self.scratch = self.root / "scratch"
        self.scratch.mkdir(parents=True, exist_ok=True)

    def dispatch(self, round_number, phase, context):
        dispatch_id = f"r{round_number:03d}-{phase}"
        folder = self.root / "dispatches" / dispatch_id
        folder.mkdir(parents=True, exist_ok=True)
        prompt = (
            f"Site {self.site}, reporting round {round_number}, {phase} handoff.\n"
            "Investigate only your local /public/dataset.parquet with native Biomni tools. "
            "Preserve your code, conversation and research memory in /work. Other sites' data "
            "are unavailable. The central instructions and shared ledger are below. "
            "Read state through benchmark_exchange if needed. Submit a structured handoff, "
            "then finish this dispatch; a later dispatch will continue your native workspace. "
            "Do not close or advance benchmark rounds yourself.\n"
            + SITE_TOOL
            + "\n"
            + json.dumps(context)
            + "\nSiteHandoff schema: "
            + json.dumps(SiteHandoff.model_json_schema())
        )
        request_hash = fingerprint({"prompt": prompt, "site": self.site})
        outcome = folder / "outcome.json"
        if outcome.exists():
            saved = json.loads(outcome.read_text())
            if not self.resume or saved["request_hash"] != request_hash:
                raise ValueError("Native dispatch already exists or inputs changed")
            if saved["handoff_sha256"] != fingerprint(saved["handoff"]):
                raise ValueError("Saved native handoff changed")
            return saved["handoff"]
        gateway = SiteGateway(round_number, context)
        broker = RunBroker(
            self.spec, gateway, folder, secrets.token_hex(32), shared_budget=self.budget
        )
        journal = folder / "exchanges.jsonl"
        for line in journal.read_text().splitlines() if journal.exists() else []:
            row = json.loads(line)
            if gateway.exchange(row["request"]) != row["response"]:
                raise ValueError("Native site exchange replay changed")
            broker.exchange_count += 1
        checkpoint = self.scratch / "native_checkpoint.json"
        continuation = False
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            namespace = self.scratch / "native_namespace.pkl"
            if (
                not saved.get("resumable")
                or not namespace.exists()
                or (saved.get("namespace_sha256") != sha256(namespace))
            ):
                raise ValueError("No safe native namespace checkpoint; code will not be replayed")
            continuation = saved.get("dispatch_id") != dispatch_id
            if continuation:
                previous = (
                    self.root / "dispatches" / saved.get("dispatch_id", "missing") / "outcome.json"
                )
                if not previous.exists() or broker.requests or broker.exchange_count:
                    raise ValueError("Cannot continue from an unfinished native dispatch")
            elif not self.resume or (
                saved["llm_requests"] != len(broker.requests)
                or saved["exchange_count"] != broker.exchange_count
            ):
                raise ValueError("No consistent native checkpoint for this dispatch")
        elif broker.requests or broker.exchange_count:
            raise ValueError("Missing native checkpoint; arbitrary code will not be replayed")
        config_path = folder / "worker_config.json"
        url = broker.start()
        write_json(
            config_path,
            dict(
                broker=url,
                secret=broker.secret,
                model=self.spec.model,
                max_tokens=self.spec.max_tokens,
                temperature=self.spec.temperature,
                request_timeout=self.spec.request_timeout,
                tool_timeout=self.spec.tool_timeout,
                prompt=prompt,
                run_id=dispatch_id,
                resume=checkpoint.exists(),
                continuation=continuation,
                require_namespace_hash=True,
                exchange_doc=SITE_TOOL,
            ),
        )
        try:
            code = self.runner.run(self.spec, self.public, self.scratch, config_path, broker)
        finally:
            broker.stop()
            config_path.unlink(missing_ok=True)
        if code or broker.fatal or gateway.handoff is None:
            raise RuntimeError(broker.fatal or f"native_site_exit_{code}_or_missing_handoff")
        saved = json.loads(checkpoint.read_text())
        if (
            not saved.get("resumable")
            or saved.get("dispatch_id") != dispatch_id
            or saved["llm_requests"] != len(broker.requests)
            or saved["exchange_count"] != broker.exchange_count
            or saved.get("namespace_sha256") != sha256(self.scratch / "native_namespace.pkl")
        ):
            raise ValueError("Native dispatch ended without a consistent checkpoint")
        write_json(
            outcome,
            dict(
                request_hash=request_hash,
                handoff=gateway.handoff,
                handoff_sha256=fingerprint(gateway.handoff),
            ),
        )
        return gateway.handoff


class CentralDirection(StrictModel):
    goal: str = Field(max_length=8000)
    site_instructions: dict[str, str] = Field(default_factory=dict)


class NativeFederation:
    def __init__(self, spec, controller, root, central, sites):
        self.spec, self.controller, self.root = spec, controller, Path(root)
        self.central, self.sites = central, sites
        self.gateway = ExchangeGateway(controller, spec.rounds)
        self.cell = FederationCell(sites=spec.sites, seed=spec.partition_seed)
        self.frames, membership = partition_frame(
            controller.frame, self.cell, controller.spec.pair_id, controller.service.replicate_id
        )
        if set(self.frames) != set(sites):
            raise ValueError("Site workers do not match partitions")
        self.contexts = {s: public_site_context(f, {}) for s, f in self.frames.items()}
        self.partition = dict(condition=self.cell.model_dump(), membership=membership)
        write_json(self.root / "partition.json", self.partition)
        controller.frame = controller.frame.iloc[:0].copy()
        controller.analysis_estimator = self.analysis
        controller.service.validation_estimator = self.validation
        self.local_evidence = {}
        self.allowed = set()
        self.memory = {"notes": "", "narratives": {}}

    def combine(self, frames, hypothesis, kwargs, source):
        statistics = {
            site: sufficient_statistics(frame, hypothesis) for site, frame in frames.items()
        }
        result = combine_statistics(list(statistics.values()), hypothesis, **kwargs)
        self.local_evidence[result.id] = dict(
            source=source,
            sites={
                s: public_result(combine_statistics([v], hypothesis, **kwargs))
                for s, v in statistics.items()
            },
            combined=public_result(result),
        )
        return result

    def analysis(self, _frame, hypothesis, **kwargs):
        if hypothesis.id not in self.allowed:
            raise ValueError("Canonical analysis was not approved by the orchestrator")
        return self.combine(self.frames, hypothesis, kwargs, "discovery")

    def validation(self, frame, hypothesis, **kwargs):
        frames, _ = partition_frame(
            frame, self.cell, self.controller.spec.pair_id, self.controller.service.replicate_id
        )
        return self.combine(frames, hypothesis, kwargs, "independent_validation")

    def context(self):
        return dict(
            ledger=self.gateway.public_state(),
            site_evidence=self.local_evidence,
            research_notes=self.memory["notes"],
            latest_exchange_narratives=self.memory["narratives"],
        )

    def ask(self, key, payload, schema):
        prompt = (
            COMMON + "\nYou are an external orchestrator coordinating native Biomni sites. "
            "Only you may mutate the shared ledger. Site handoffs are recommendations; "
            "only trusted canonical results are official numerical evidence. You have no raw rows. "
            "Choose how to weigh pooled evidence, replication and disagreement. Return JSON only.\n"
            + json.dumps(dict(**payload, schema=schema))
        )
        path = self.root / "central" / "decisions" / f"{key}.json"
        request_hash = fingerprint(prompt)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["request_hash"] != request_hash:
                raise ValueError("Central replay prompt changed")
            if saved["response_sha256"] != fingerprint(saved["response"]):
                raise ValueError("Central response changed")
            return json_response(saved["response"])
        result = self.central.complete({"messages": [{"role": "user", "content": prompt}]})
        response = result["choices"][0]["message"]["content"]
        write_json(
            path,
            dict(
                request_hash=request_hash, response=response, response_sha256=fingerprint(response)
            ),
        )
        return json_response(response)

    def dispatch(self, number, phase, direction):
        shared = self.gateway.public_state()

        def run(site):
            context = dict(
                goal=direction["goal"],
                instructions=direction["site_instructions"].get(site, ""),
                phase=phase,
                ledger=shared,
                site=self.contexts[site],
                site_evidence={
                    k: dict(source=v["source"], result=v["sites"][site])
                    for k, v in self.local_evidence.items()
                },
            )
            return self.sites[site].dispatch(number, phase, context)

        with ThreadPoolExecutor(max_workers=self.spec.sites) as pool:
            values = list(pool.map(run, sorted(self.sites)))
        return dict(zip(sorted(self.sites), values, strict=True))

    def operate(self, number, phase, handoffs):
        target = "prepare_close" if phase == "research" else "close"
        for step in range(self.spec.max_central_actions_per_round):
            error = None
            for attempt in range(self.spec.central_retries + 1):
                key = f"r{number:03d}-{phase}-{step:03d}-a{attempt}"
                try:
                    response = self.ask(
                        key,
                        dict(
                            **self.context(),
                            handoffs=handoffs,
                            instruction=f"Choose one exchange; end with {target}. "
                            "Assess new/due evidence before closing. Allowances: 12 analyses "
                            "and one new voluntary validation per round, ten voluntary total. "
                            "prepare_close releases scheduled evidence. Assess before close. "
                            "Do not use state as an action.",
                            previous_error=error,
                            action_payload_schemas={
                                "register": ResearchForm.model_json_schema(),
                                "analyze": AnalyzeForm.model_json_schema(),
                                "assess": ResearchForm.model_json_schema(),
                                "validate": AppraiseForm.model_json_schema(),
                                "prepare_close": {"type": "object", "maxProperties": 0},
                                "close": ResearchForm.model_json_schema(),
                            },
                        ),
                        Exchange.model_json_schema(),
                    )
                    request = Exchange.model_validate(response)
                    if request.round != number or request.action == "state":
                        raise ValueError("Wrong round or non-progressing state action")
                    if phase == "research" and request.action == "close":
                        raise ValueError("Site review must occur before closing the round")
                    if phase == "review" and request.action not in {"assess", "close"}:
                        raise ValueError("Review permits only assess and close")
                    request.request_id = key
                    hs, _ = references(self.controller)
                    self.allowed = (
                        {hs[h] for h in request.payload.get("run_analyses", [])}
                        if request.action == "analyze"
                        else set()
                    )
                    receipt = self.gateway.exchange(request.model_dump())
                    if request.payload.get("research_notes") is not None:
                        self.memory["notes"] = request.payload["research_notes"]
                    if "narrative" in request.payload:
                        self.memory["narratives"][request.action] = request.payload["narrative"]
                    write_json(self.root / "central" / "receipts" / f"{key}.json", receipt)
                    break
                except (ValueError, KeyError) as exc:
                    error = str(exc)
                    if attempt == self.spec.central_retries:
                        raise
            if request.action == target:
                return
        raise RuntimeError("Central action limit exceeded before round boundary")

    def run(self):
        for number in range(1, self.spec.rounds + 1):
            error = None
            for attempt in range(self.spec.central_retries + 1):
                try:
                    direction = CentralDirection.model_validate(
                        self.ask(
                            f"r{number:03d}-direction-a{attempt}",
                            dict(
                                **self.context(),
                                instruction="Set the common goal and optional site directions.",
                                sites=self.contexts,
                                previous_error=error,
                            ),
                            CentralDirection.model_json_schema(),
                        )
                    ).model_dump()
                    if set(direction["site_instructions"]) - self.sites.keys():
                        raise ValueError("Unknown site in central directions")
                    break
                except ValueError as exc:
                    error = str(exc)
                    if attempt == self.spec.central_retries:
                        raise
            handoffs = self.dispatch(number, "research", direction)
            self.operate(number, "research", handoffs)
            reviews = self.dispatch(number, "review", direction)
            self.operate(number, "review", reviews)
            write_json(
                self.root / "progress.json", dict(completed_rounds=number, rounds=self.spec.rounds)
            )
        return self.gateway.public_state()
