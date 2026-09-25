"""Stage-wise hub-and-spoke science using a single authoritative budget and ledger.

The LLM orchestrator never receives rows. Site LLMs receive only their own aggregate
context and the shared aggregate ledger. Trusted Python executes
all site queries; only the central selected form changes the common scientific record.
"""

import json
from pathlib import Path

from ..harness.durable_io import StorageUnavailable, atomic_write_json
from .coordination import StageCoordinator
from .federated_history import FederatedHistoryCoordinator
from .prompting import STAGE_INSTRUCTIONS, direct_results, references, repair_message, translate
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


class FederatedCoordinator:
    def __init__(
        self, provider, workflow, stages, source, budget, run_dir, cell, *, central_provider=None
    ):
        self.cell, self.workflow, self.source = cell, workflow, source
        self.root = Path(run_dir)
        self.shared_budget = {"calls": 0}
        coordinator_type = (
            FederatedHistoryCoordinator
            if cell.context_policy == "federated_v2"
            else StageCoordinator
        )
        self.central = coordinator_type(
            central_provider or provider,
            workflow,
            stages,
            source,
            budget,
            self.root / "calls" / "central",
            self.shared_budget,
        )
        self.central.federation_role = "orchestrator"
        self.sites = {
            f"site_{i + 1}": coordinator_type(
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
            coordinator.federation_role = "site"
        self.prepared = {}
        self.local_statistics = {}
        self.validation_summaries = {}
        self.site_memory = {site: {} for site in self.sites}
        self.site_discovery = {site: {} for site in self.sites}
        self.handoff_errors = []
        self.active = None

    def _write_handoff(self, key, site, handoff):
        try:
            atomic_write_json(self.root / "handoffs" / f"{key}-{site}.json", handoff)
        except (OSError, StorageUnavailable) as error:
            raise WorkflowInfrastructureError(
                f"Handoff persistence failed: {key}/{site}"
            ) from error

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
        allowed = self.prepared[self.active]["analyses"]
        if hypothesis.id not in {hs[ref] for ref in allowed}:
            raise ValueError("The orchestrator cannot execute an analysis not sent to the sites")
        statistics = self._stats(hypothesis)
        ref = next(ref for ref, hid in hs.items() if hid == hypothesis.id)
        for site, stats in statistics.items():
            self.prepared[self.active]["local_results"][site][ref] = public_result(
                combine_statistics([stats], hypothesis, **kwargs)
            )
        return combine_statistics(list(statistics.values()), hypothesis, **kwargs)

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

    def _committed_local_results(self, site):
        hs, rs = references(self.controller)
        reverse_r = {rid: ref for ref, rid in rs.items()}
        summaries = {}
        for claim, result in self.site_discovery[site].items():
            h = self.controller.state["hypotheses"][hs[claim]]
            refs = [
                reverse_r[rid]
                for rid in direct_results(self.controller, h)
                if rid.startswith("analysis-")
            ]
            if refs:
                summaries[claim] = {
                    "shared_evidence": refs,
                    **{k: v for k, v in result.items() if k not in {"id", "hypothesis_id"}},
                }
        return summaries

    def _committed_validation_summaries(self):
        _, rs = references(self.controller)
        reverse_r = {rid: ref for ref, rid in rs.items()}
        return {
            reverse_r[rid]: {
                "source": item["source"],
                "sites": {
                    site: {k: v for k, v in result.items() if k != "id"}
                    for site, result in item["sites"].items()
                },
                "combined": {k: v for k, v in item["combined"].items() if k != "id"},
            }
            for rid, item in self.validation_summaries.items()
            if rid in reverse_r
        }

    def _site_prompt(self, prompt, site, stage):
        marker = prompt.index('{"schema":')
        payload = json.loads(prompt[marker:])
        payload["task"] = self.contexts[site]
        shared_notes = payload["research_notes"]
        payload["research_notes"] = self.site_memory[site].get("notes", "")
        payload["latest_stage_narratives"] = self.site_memory[site].get("narratives", {})
        payload["federation"] = {
            "site": site,
            "shared_research_notes": shared_notes,
            "local_analyses": self._committed_local_results(site),
            "local_validation": {
                key: value["sites"][site]
                for key, value in self._committed_validation_summaries().items()
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
                "Recommend up to 12 comparisons in run_analyses. The central team selects "
                "the shared set after considering every site's recommendations. Trusted "
                "execution then tests that set at every site; assess the registered results "
                "in appraise. "
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
                "handoffs": {},
                "prompt": prompt,
            }
        entry = self.prepared[key]
        # A repaired central candidate may choose different comparisons. Keep all
        # provisional results private until the ordinary controller commits.
        entry["analyses"] = []
        entry["local_results"] = {site: {} for site in self.sites}
        for site, coordinator in self.sites.items():
            if site in entry["handoffs"]:
                continue
            site_prompt = self._site_prompt(entry["prompt"], site, stage)
            for repair in range(1, self.source.max_retries_per_stage + 2):
                try:
                    response = coordinator.respond(
                        site_prompt, iteration=iteration, stage=stage, attempt=repair
                    )
                    # Check shared H/R links without applying the recommendation or
                    # allocating IDs, changing assessments, or spending validation.
                    _, form, _ = translate(
                        self.controller, stage, iteration, json_response(response.text)
                    )
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
                        if getattr(self.source, "peer_failure_policy", "require_all") != (
                            "chair_with_available"
                        ):
                            raise ValueError(f"{site} exhausted stage retries: {error}") from error
                        form = None
                        break
                    site_prompt += "\nRepair the site form: " + repair_message(
                        self.controller, str(error)
                    )
            handoff = {
                "site": site,
                "form": form.model_dump(by_alias=True) if form is not None else None,
                "unavailable": form is None,
                "notice": (
                    "Site recommendation unavailable after repairs; do not infer agreement. "
                    "Trusted analysis still includes this site's aggregate statistics."
                    if form is None
                    else ""
                ),
                "site_summary": self.contexts[site],
                "analysis_results": {},
            }
            entry["handoffs"][site] = handoff
            self._write_handoff(key, site, handoff)
        # New numerical results cannot be assessed until controller.apply registers
        # them. Keep this stage's computed handoffs in the audit, but expose only
        # committed results to agents. Appraise sees the new R references and sites.
        visible_handoffs = [
            {**handoff, "analysis_results": self._committed_local_results(site)}
            for site, handoff in entry["handoffs"].items()
        ]
        if stage == "analyze":
            prompt = prompt.replace(
                STAGE_INSTRUCTIONS[stage],
                "Select up to 12 shared comparisons in run_analyses. "
                "Their new results will enter the ledger after this form commits; "
                "assess them in appraise. Use only already registered evidence now.",
                1,
            )
        central_prompt = (
            prompt + "\nSite consultation is complete. Available recommendations and previously "
            "committed local discovery summaries:\n"
            + json.dumps(visible_handoffs)
            + "\nIndependent validation site summaries:\n"
            + json.dumps(self._committed_validation_summaries())
            + "\nUse the registered shared ledger for assessments. Explain site disagreement "
            "using the committed site summaries. Their shared_evidence links and validation "
            "summary keys identify the corresponding shared ledger R references. "
            "Return the ordinary stage form. "
            + (
                "New results are withheld until this form commits; assess them in appraise, "
                "when they have registered R references. Select run_analyses using the "
                "shared ledger and site recommendations. The controller executes each "
                "selected comparison at every site, including sites with unavailable "
                "recommendations; this spends one shared comparison per selected claim."
                if stage == "analyze"
                else STAGE_INSTRUCTIONS[stage]
            )
        )
        response = self.central.respond(
            central_prompt, iteration=iteration, stage=stage, attempt=attempt
        )
        _, form, _ = translate(self.controller, stage, iteration, json_response(response.text))
        if stage == "analyze":
            entry["analyses"] = form.run_analyses
        return response

    def commit(self):
        self.central.commit()
        for site, coordinator in self.sites.items():
            entry = self.prepared[self.active]
            handoff = entry["handoffs"][site]
            handoff["analysis_results"] = entry["local_results"][site]
            self._write_handoff(self.active, site, handoff)
            self.site_discovery[site].update(handoff["analysis_results"])
            if handoff["form"] is None:
                continue
            coordinator.commit()
            form = handoff["form"]
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
            "federation_protocol": "workflow-preserving-2",
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
