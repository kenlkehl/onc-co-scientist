"""Framework-independent public exchange API over the existing scientific ledger.

Stage names below are internal record types, not instructions to an agent. An
external scientist can register, test and reassess repeatedly in any order, subject
only to the reporting protocol. Private scoring state never crosses this API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..expected_surprising.prompting import (
    AnalyzeForm,
    AppraiseForm,
    ResearchForm,
    ledger,
    references,
    translate,
)
from ..expected_surprising.rollout import stage_transaction
from ..expected_surprising.schemas import StrictModel
from ..expected_surprising.scoring import comparison_key
from ..expected_surprising.workflow import WorkflowController, public_event


class Exchange(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    round: int = Field(ge=1)
    action: Literal["state", "register", "analyze", "assess", "validate", "prepare_close", "close"]
    payload: dict = Field(default_factory=dict)


class NativeController(WorkflowController):
    def required(self, iteration, stage):
        # The gateway enforces outstanding evidence obligations across native
        # exchanges, rather than requiring a named appraisal/synthesis stage.
        return []

    def _assess(self, record, sequence, available):
        super()._assess(record, sequence, available)
        for event in self.state["events"]:
            for assessment in self.state["assessments"]:
                if (
                    assessment["sequence"] != sequence
                    or assessment["hypothesis_id"] != event["hypothesis_id"]
                    or event["result"]["id"] not in assessment["result_ids"]
                ):
                    continue
                if event["iteration"] == record.iteration:
                    event["immediate"] = assessment
                if event["source"] != "discovery" and event["due_iteration"] == record.iteration:
                    event["delayed"] = assessment


class ExchangeGateway:
    """Transactional, idempotent operations; one instance per isolated replicate."""

    def __init__(self, controller, rounds):
        self.controller = controller
        self.rounds = rounds
        self.state = {
            "round": 1,
            "sealed": False,
            "analyses": 0,
            "validations": 0,
            "completed_rounds": 0,
            "receipts": {},
        }

    def pending(self):
        c, current = self.controller, self.state["round"]
        hs, rs = references(c)
        hr, rr = {v: k for k, v in hs.items()}, {v: k for k, v in rs.items()}
        needed = {}
        # Invalid analyses must also receive explicit appraisal, but never a score.
        for execution in c.state["executions"]:
            if execution["iteration"] == current:
                key = (execution["hypothesis"]["id"], execution["result"]["id"])
                if not any(
                    a["iteration"] == current
                    and a["hypothesis_id"] == key[0]
                    and key[1] in a["result_ids"]
                    for a in c.state["assessments"]
                ):
                    needed[key] = "new_result"
        for e in c.state["events"]:
            key = (e["hypothesis_id"], e["result"]["id"])
            if e["iteration"] == current and e["immediate"] is None:
                needed[key] = "new_result"
            if (
                e["source"] != "discovery"
                and e["due_iteration"] == current
                and e["delayed"] is None
            ):
                needed[key] = "response_deadline"
        return [
            {"claim": hr[h], "evidence": rr[r], "reason": reason}
            for (h, r), reason in needed.items()
        ]

    def public_state(self):
        c = self.controller
        return {
            "round": self.state["round"],
            "rounds": self.rounds,
            "completed_rounds": self.state["completed_rounds"],
            "sealed": self.state["sealed"],
            "finished": self.state["completed_rounds"] == self.rounds,
            "analyses_remaining": 12 - self.state["analyses"],
            "voluntary_requests_remaining": c.service.policy.voluntary_limit
            - len(c.service.state["voluntary_keys"]),
            "claims": ledger(c),
            "assessments_due": self.pending(),
        }

    def exchange(self, request):
        req = Exchange.model_validate(request)
        previous = self.state["receipts"].get(req.request_id)
        if previous:
            if previous["request"] != req.model_dump():
                raise ValueError("request_id reused with different content")
            return previous["response"]
        if req.round != self.state["round"]:
            raise ValueError("Wrong reporting round; request state with the current round")
        c = self.controller
        with stage_transaction([self.state, c.state, c.service.state]):
            event = self._apply(req)
            response = {"event": event, **self.public_state()}
            self.state["receipts"][req.request_id] = {
                "request": req.model_dump(),
                "response": response,
            }
        return response

    def _apply(self, req):
        c, s = self.controller, self.state
        if req.action == "state":
            if req.payload:
                raise ValueError("state takes no payload")
            return None
        if s["completed_rounds"] == self.rounds:
            raise ValueError("All reporting rounds are complete")
        if s["sealed"] and req.action not in {"assess", "close"}:
            raise ValueError("Round sealed; assess released evidence, then close")
        if req.action == "prepare_close":
            if req.payload:
                raise ValueError("prepare_close takes no payload")
            if self.pending():
                raise ValueError("Assess outstanding evidence before preparing to close")
            c.service.select(s["round"], c.state["tests"])
            deliveries = []
            for opportunity in c.service.due(s["round"]):
                h = c.state["hypotheses"][opportunity["hypothesis_id"]]
                seq = len(c.state["completed_stages"]) + 1
                result, fresh = c.service.deliver(h, s["round"], "automatic", sequence=seq)
                deliveries.append(c._deliver(h, result, s["round"], seq, "automatic", fresh))
            s["sealed"] = True
            return public_event({"results": deliveries})
        if req.action == "close" and not s["sealed"]:
            raise ValueError("Call prepare_close to release scheduled evidence first")
        stage = {
            "register": "explore",
            "analyze": "analyze",
            "assess": "explore",
            "validate": "appraise",
            "close": "synthesize",
        }[req.action]
        if req.action == "register" and req.payload.get("assessments"):
            raise ValueError("Use assess for subsequent assessments")
        if req.action in {"assess", "validate", "close"} and req.payload.get("proposals"):
            raise ValueError("Use register for proposals")
        if req.action == "analyze":
            form = AnalyzeForm.model_validate(req.payload)
            if form.proposals or form.assessments:
                raise ValueError("Register and assess in separate exchanges")
            if s["analyses"] + len(form.run_analyses) > 12:
                raise ValueError("At most 12 canonical analyses per reporting round")
            s["analyses"] += len(form.run_analyses)
        if req.action == "validate":
            form = AppraiseForm.model_validate(req.payload)
            if not form.validation_claim:
                raise ValueError("validate requires a claim")
            if self.pending():
                raise ValueError("Appraise new evidence before requesting validation")
            hs, _ = references(c)
            if form.validation_claim not in hs:
                raise ValueError("Unknown claim")
            key = comparison_key(c.state["hypotheses"][hs[form.validation_claim]])
            if key not in c.service.state["cache"]:
                if s["validations"] >= 1:
                    raise ValueError("At most one new voluntary validation per round")
                s["validations"] += 1
        if req.action == "close":
            ResearchForm.model_validate(req.payload)
        record, _, _ = translate(c, stage, s["round"], req.payload)
        event = c.apply(record, deliver_scheduled=False)
        # Public data inspection is unrestricted in this arm. Conservatively do
        # not characterize any registration as an unexposed prior expectation.
        for h in record.hypotheses:
            c.state["expectations"][h.id]["pre_evidence"] = False
            for registration in c.state["registrations"]:
                if registration["hypothesis"]["id"] == h.id:
                    registration["pre_evidence"] = False
        if req.action == "close":
            if self.pending():
                raise ValueError("Explicitly assess all new and due evidence before closing")
            s["completed_rounds"] += 1
            if s["completed_rounds"] < self.rounds:
                s.update(round=s["round"] + 1, sealed=False, analyses=0, validations=0)
        return public_event(event)
