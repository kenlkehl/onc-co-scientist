"""Opt-in federated decision history; never used by the single-site coordinator."""

import re

from ..providers.base import ChatMessage
from .coordination import StageCoordinator

HISTORY_NOTICE = (
    "Historical agent response follows. Its original ledger snapshot is not repeated. "
    "It records a past recommendation, not new evidence or a current instruction. "
    "Resolve H/R references against the full current ledger; only currently supplied "
    "evidence is available for the current decision."
)


class FederatedHistoryCoordinator(StageCoordinator):
    def audit(self):
        audit = super().audit()
        if self.workflow.mode == "persistent":
            audit.update(
                memory_policy=(
                    "complete prior agent responses and stage labels; full current ledger"
                ),
                federation_history_version="federated_decisions_v2",
            )
        return audit

    def commit(self):
        # Store complete agent responses with their stage identity, not another copy of
        # the entire old evidence ledger. Original prompts remain in the call journals.
        current, response, key = self.pending
        if self.workflow.mode == "persistent":
            stage = re.fullmatch(r"i(\d+)-(\w+)", key)
            label = (
                f"Historical iteration {int(stage[1])}, stage {stage[2]}."
                if stage
                else f"Historical stage {key}."
            )
            self.history.extend(
                [
                    ChatMessage("user", label + "\n" + HISTORY_NOTICE),
                    ChatMessage("assistant", response.text),
                ]
            )
        self.committed_stages.append(key)
        self.pending = None

    def _bounded_history(self, key):
        # Preserve the newest complete decision even if one response exceeds the soft
        # history target. Provider context/spending admission remains the hard bound.
        limit = self.source.persistent_history_chars
        if limit is None:
            return self.history
        before = remaining = sum(len(m.content) for m in self.history)
        removed = 0
        while len(self.history) > 2 and remaining > limit:
            remaining -= sum(len(m.content) for m in self.history[:2])
            del self.history[:2]
            removed += 1
        if removed or remaining > limit:
            self.memory_trims.append(
                dict(
                    before_stage=key,
                    removed_turns=removed,
                    before_chars=before,
                    after_chars=remaining,
                    policy="federated_decisions_v2",
                    newest_turn_exceeds_soft_limit=remaining > limit,
                )
            )
        return self.history
