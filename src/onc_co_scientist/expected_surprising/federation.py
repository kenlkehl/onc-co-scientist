"""Stage-wise hub-and-spoke science using a single authoritative budget and ledger.

The LLM orchestrator never receives rows. Site LLMs receive only their own aggregate
context, the shared aggregate ledger, and central directions. Trusted Python executes
all site queries; only the central selected form changes the common scientific record.
"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..harness.durable_io import atomic_write_json
from ..harness.experiment import WorkflowSpec
from ..providers.base import ChatMessage
from .coordination import StageCoordinator
from .prompting import FORMS, SCIENTIFIC_GUIDANCE, references
from .research import json_response
from .scoring import comparison_key
from .site_statistics import (
    combine_statistics,
    partition_frame,
    public_result,
    public_site_context,
    sufficient_statistics,
)
from .workflow import WorkflowInfrastructureError


class Direction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = ""
    site_instructions: dict[str, str] = Field(default_factory=dict)
    analyses: list[str] = Field(default_factory=list, max_length=12)


CENTRAL = """You are the federated orchestrator pursuing one unified research goal. Direct the
sites and interpret their aggregate evidence. You cannot access patient rows or private site
files; never request raw records or row-level exports. Site narratives are proposals, not
new evidence. Use only controller-produced numerical summaries for empirical claims.
Choose how to weigh combined evidence, replication and site disagreement; no rule forces
acceptance or consensus across sites. One shared budget applies to the entire federation:
at most 12 distinct comparisons per iteration and the single displayed validation allowance.
Each approved comparison is evaluated across all sites as one analysis request, not N requests.
Only your final stage form changes the shared ledger. Sites can recommend validation; only
you can spend the common validation allowance. The final independent evaluator is unchanged.
"""


class FederatedCoordinator:
    def __init__(
        self, provider, workflow, stages, source, budget, run_dir, cell, *, central_provider=None
    ):
        self.cell, self.workflow, self.source = cell, workflow, source
        self.root = Path(run_dir)
        self.shared_budget = {"calls": 0}
        self.central = StageCoordinator(
            central_provider or provider,
            WorkflowSpec(id="orchestrator", mode="persistent"),
            stages,
            source,
            budget,
            self.root / "calls" / "central",
            self.shared_budget,
        )
        self.sites = {
            f"site_{i + 1}": StageCoordinator(
                provider,
                workflow,
                stages,
                source,
                budget,
                self.root / "calls" / f"site_{i + 1}",
                self.shared_budget,
            )
            for i in range(cell.sites)
        }
        for coordinator in self.sites.values():
            coordinator.central_authority = False
        self.prepared = {}
        self.local_statistics = {}
        self.validation_summaries = {}
        self.site_memory = {site: {} for site in self.sites}
        self.site_discovery = {site: {} for site in self.sites}
        self.handoff_errors = []
        self.active = None

    def bind(self, controller, task_context):
        self.controller = controller
        self.frames, membership = partition_frame(
            controller.frame, self.cell, controller.spec.pair_id, controller.service.replicate_id
        )
        task = {k: v for k, v in task_context.items() if k != "variables"}
        self.contexts = {
            site: public_site_context(frame, task) for site, frame in self.frames.items()
        }
        # Central setup has schema only. Numerical results arrive in site handoffs.
        task_context["variables"] = {
            name: {"dtype": str(dtype)}
            for name, dtype in controller.frame.dtypes.items()
            if name not in {"patient_id", "cell_line_id"}
        }
        task_context["federation"] = {"sites": list(self.sites), "shared_goal": True}
        controller.frame = controller.frame.iloc[
            :0
        ].copy()  # schema only; no central raw-data fallback
        controller.analysis_estimator = self._analysis
        controller.service.validation_estimator = self._validation
        self.partition_audit = {
            "condition": self.cell.model_dump(),
            "membership": membership,
            "rows": sum(len(f) for f in self.frames.values()),
        }
        atomic_write_json(self.root / "partition.json", self.partition_audit)

    def _stats(self, hypothesis):
        key = comparison_key(hypothesis)
        # Keep directional sufficient statistics keyed to the claim, not just its comparison.
        cache_key = (key, hypothesis.direction, str(hypothesis.exposed), str(hypothesis.comparator))
        if cache_key not in self.local_statistics:
            self.local_statistics[cache_key] = {
                site: sufficient_statistics(frame, hypothesis)
                for site, frame in self.frames.items()
            }
        return self.local_statistics[cache_key]

    def _analysis(self, _frame, hypothesis, **kwargs):
        hs, _ = references(self.controller)
        allowed = self.prepared[self.active]["direction"]["analyses"]
        if hypothesis.id not in {hs[ref] for ref in allowed}:
            raise ValueError("The orchestrator cannot execute an analysis not sent to the sites")
        return combine_statistics(list(self._stats(hypothesis).values()), hypothesis, **kwargs)

    def _validation(self, frame, hypothesis, **kwargs):
        parts, _ = partition_frame(
            frame, self.cell, self.controller.spec.pair_id, self.controller.service.replicate_id
        )
        statistics = {site: sufficient_statistics(part, hypothesis) for site, part in parts.items()}
        result = combine_statistics(list(statistics.values()), hypothesis, **kwargs)
        self.validation_summaries[result.id] = {
            "source": "independent_validation",
            "sites": {
                site: public_result(combine_statistics([stats], hypothesis, **kwargs))
                for site, stats in statistics.items()
            },
            "combined": public_result(result),
        }
        return result

    def _direction(self, prompt, iteration, stage, attempt):
        marker = prompt.index('{"schema":')
        payload = json.loads(prompt[marker:])
        payload["schema"] = Direction.model_json_schema()
        request = (
            "Before this stage, direct the sites. Return only the Direction JSON form below. "
            "Use site_instructions for optional site-specific scientific directions. "
            "Only analyze may set analyses: choose existing H references, at most 12. "
            "All sites evaluate each selected comparison so coverage is comparable. "
            "Other stages must return analyses=[]. This planning call does not change the ledger.\n"
            + prompt.splitlines()[0]
            + "\n"
            + SCIENTIFIC_GUIDANCE
            + "\n"
            + json.dumps(payload, separators=(",", ":"))
        )
        response = self.central._call(
            f"i{iteration:03d}-{stage}-direction-a{attempt}",
            [ChatMessage("system", CENTRAL), ChatMessage("user", request)],
            session="orchestrator",
            authoritative=False,
            iteration=iteration,
            stage=stage,
            kind="direction",
        )
        direction = Direction.model_validate(json_response(response.text))
        hs, _ = references(self.controller)
        if set(direction.site_instructions) - self.sites.keys():
            raise ValueError("Unknown site in site_instructions")
        if (
            len(set(direction.analyses)) != len(direction.analyses)
            or set(direction.analyses) - hs.keys()
        ):
            raise ValueError("analyses must contain distinct existing H references")
        if stage != "analyze" and direction.analyses:
            raise ValueError("Only the analyze stage can direct analyses")
        return direction.model_dump()

    def _site_prompt(self, prompt, site, direction, stage):
        marker = prompt.index('{"schema":')
        payload = json.loads(prompt[marker:])
        payload["task"] = self.contexts[site]
        payload["research_notes"] = self.site_memory[site].get("notes", "")
        payload["latest_stage_narratives"] = self.site_memory[site].get("narratives", {})
        payload["federation"] = {
            "site": site,
            "goal": direction["goal"],
            "directions": direction["site_instructions"].get(site, ""),
            "approved_analyses": direction["analyses"],
            "local_analyses": self.site_discovery[site],
            "local_validation": {
                key: value["sites"][site] for key, value in self.validation_summaries.items()
            },
        }
        instructions = (
            "\nYou are a site team. Use only your own aggregate context and the central "
            "shared evidence ledger. Do not request raw records, contact other sites, "
            "or describe invented rows. Your form is a recommendation to the orchestrator; "
            "it alone registers claims and spends validation slots. "
            "Discuss your local numerical results in the narrative; use only shared ledger "
            "R references in the evidence field. "
        )
        if stage == "analyze":
            instructions += (
                "Set run_analyses to exactly approved_analyses. "
                "Trusted local execution follows your form. "
            )
        return prompt[:marker] + instructions + "\n" + json.dumps(payload, separators=(",", ":"))

    def respond(self, prompt, *, iteration, stage, attempt):
        key = f"i{iteration:03d}-{stage}"
        if self.active is not None and self.active != key:
            for coordinator in self.sites.values():
                coordinator.reject()
        self.active = key
        committed_validation_ids = {
            v["result"].id for v in self.controller.service.state["cache"].values()
        }
        self.validation_summaries = {
            k: v for k, v in self.validation_summaries.items() if k in committed_validation_ids
        }
        if key not in self.prepared:
            self.prepared[key] = {
                "direction": self._direction(prompt, iteration, stage, attempt),
                "handoffs": {},
                "prompt": prompt,
            }
        entry = self.prepared[key]
        for site, coordinator in self.sites.items():
            if site in entry["handoffs"]:
                continue
            site_prompt = self._site_prompt(entry["prompt"], site, entry["direction"], stage)
            for repair in range(1, self.source.max_retries_per_stage + 2):
                try:
                    response = coordinator.respond(
                        site_prompt, iteration=iteration, stage=stage, attempt=repair
                    )
                    form = FORMS[stage].model_validate(json_response(response.text))
                    if stage == "analyze" and set(form.run_analyses) != set(
                        entry["direction"]["analyses"]
                    ):
                        raise ValueError("Use exactly the approved_analyses references")
                    break
                except WorkflowInfrastructureError:
                    raise
                except Exception as error:
                    coordinator.reject()
                    self.handoff_errors.append(
                        {
                            "site": site,
                            "iteration": iteration,
                            "stage": stage,
                            "attempt": repair,
                            "error": str(error),
                        }
                    )
                    if repair > self.source.max_retries_per_stage:
                        raise ValueError(f"{site} exhausted stage retries: {error}") from error
                    site_prompt += "\nRepair the site form: " + str(error)
            handoff = {
                "site": site,
                "form": form.model_dump(by_alias=True),
                "site_summary": self.contexts[site],
                "analysis_results": {},
            }
            if stage == "analyze":
                hs, _ = references(self.controller)
                for ref in entry["direction"]["analyses"]:
                    h = self.controller.state["hypotheses"][hs[ref]]
                    kwargs = dict(
                        delta=self.controller.outcomes[h.outcome].delta,
                        alpha=0.05,
                        result_id="site-analysis-"
                        + hashlib.sha256((site + comparison_key(h)).encode()).hexdigest()[:16],
                    )
                    handoff["analysis_results"][ref] = public_result(
                        combine_statistics([self._stats(h)[site]], h, **kwargs)
                    )
            entry["handoffs"][site] = handoff
            atomic_write_json(self.root / "handoffs" / f"{key}-{site}.json", handoff)
        combined = {}
        if stage == "analyze":
            hs, _ = references(self.controller)
            for ref in entry["direction"]["analyses"]:
                h = self.controller.state["hypotheses"][hs[ref]]
                combined[ref] = public_result(
                    self._analysis(
                        None,
                        h,
                        delta=self.controller.outcomes[h.outcome].delta,
                        alpha=0.05,
                        result_id="analysis-"
                        + hashlib.sha256(comparison_key(h).encode()).hexdigest()[:16],
                    )
                )
        central_prompt = (
            CENTRAL
            + "\n"
            + prompt
            + "\nAll sites completed this stage. Their aggregate handoffs:\n"
            + json.dumps(list(entry["handoffs"].values()))
            + "\nCombined summaries computed from site sufficient statistics:\n"
            + json.dumps(combined)
            + "\nIndependent validation site summaries:\n"
            + json.dumps(self.validation_summaries)
            + "\nDecide how to integrate this evidence; explain site disagreement. "
            "Return the ordinary stage form. "
            "In analyze, run_analyses must exactly record the approved comparisons "
            "already executed at the sites: " + json.dumps(entry["direction"]["analyses"])
        )
        response = self.central.respond(
            central_prompt, iteration=iteration, stage=stage, attempt=attempt
        )
        if stage == "analyze":
            form = FORMS[stage].model_validate(json_response(response.text))
            if set(form.run_analyses) != set(entry["direction"]["analyses"]):
                raise ValueError(
                    "Central run_analyses must record exactly the site-executed comparisons"
                )
        return response

    def commit(self):
        self.central.commit()
        for site, coordinator in self.sites.items():
            coordinator.commit()
            handoff = self.prepared[self.active]["handoffs"][site]
            form = handoff["form"]
            self.site_discovery[site].update(handoff["analysis_results"])
            if form.get("research_notes") is not None:
                self.site_memory[site]["notes"] = form["research_notes"]
            stage = self.active.split("-", 1)[1]
            self.site_memory[site].setdefault("narratives", {})[stage] = form.get("narrative", "")

    def reject(self):
        self.central.reject()

    def audit(self):
        groups = {"central": self.central, **self.sites}
        audits = {scope: coordinator.audit() for scope, coordinator in groups.items()}
        central = audits["central"]
        artifacts, hashes = [], {}
        for scope, audit in audits.items():
            for artifact, digest in audit["participant_artifact_sha256"].items():
                path = "calls/" + scope + "/" + Path(artifact).name
                artifacts.append(path)
                hashes[path] = digest
        usage = {
            key: sum(a["usage"][key] for a in audits.values())
            if all(a["usage"][key] is not None for a in audits.values())
            else None
            for key in ("input_tokens", "output_tokens", "duration_seconds")
        }
        return {
            **central,
            "workflow": self.workflow.model_dump(),
            "federation": self.partition_audit,
            "resource_comparison": "shared scientific comparisons and validation; "
            "all site and central calls counted",
            "agent_calls": sum(a["agent_calls"] for a in audits.values()),
            "usage": usage,
            "usage_missing_calls": sum(a["usage_missing_calls"] for a in audits.values()),
            "provider_error_calls": sum(a["provider_error_calls"] for a in audits.values()),
            "provider_infrastructure_attempts": (
                sum(a["provider_infrastructure_attempts"] for a in audits.values())
                if all(a["provider_infrastructure_attempts"] is not None for a in audits.values())
                else None
            ),
            "draft_errors": self.handoff_errors
            + [dict(e, scope=scope) for scope, a in audits.items() for e in a["draft_errors"]],
            "output_token_accounting": {
                key: sum(a["output_token_accounting"][key] for a in audits.values())
                for key in central["output_token_accounting"]
            },
            "memory_trims": [
                dict(trim, scope=scope) for scope, a in audits.items() for trim in a["memory_trims"]
            ],
            "participant_artifacts": artifacts,
            "participant_artifact_sha256": hashes,
            "scope_audits": audits,
        }
