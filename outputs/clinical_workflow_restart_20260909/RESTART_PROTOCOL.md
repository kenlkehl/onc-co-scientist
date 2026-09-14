# Selective Codex replacement protocol

The user authorized repairing the adapter, bookkeeping and failure policy, and restarting only adapter-affected or unfinished Codex cells. They separately approved briefly interrupting the shared driver so Qwen/Gemma could resume from durable calls under unchanged code.

The selection was frozen at 2026-09-09T13:50 UTC in snapshot.json and selection_audit.json. It selects 104 of 240 Codex cells: 22 terminal cells with native completed-with-warning evidence, plus 82 unfinished cells (active or queued). It preserves 136 terminal Codex cells, including four non-adapter failures. It does not select results by their scores. All 120 vLLM cells retain their original implementation; 59 were unfinished and were submitted for exact call replay/continuation. Interrupted in-flight vLLM requests can repeat with unknown original usage.

Replacement source and inputs are frozen under data/expected_surprising_ledger/full_runs/20260909_codex_repair. The unchanged old source remains under 20260908_clinical10pct_workflows. Replacement cells start with empty call journals and the same run/task/version/repeat identity; old partial and terminal outputs remain untouched. The replacement configuration implies 240 possible cells, but **only selection.json's 104 cells are scheduled**. Never launch that whole configuration with run_experiment.

All replacement models use medium reasoning and default service tier; 25 iterations, two retries, the original scientific data/seeds/analysis/validation budgets, persistent memory limit, and two peers/one chair remain. There are 16 parallel replacement runs and up to 24 continuing vLLM runs.

Changes:

- Previously fixed Codex transport accepts successful terminal completion after reconnect warnings and retains usage. All genuine terminal failure, missing final message, prohibited tool, and nonzero exit checks remain. Native transport attempts remain available for reconciliation.
- Duplicate canonical registrations preserve the existing belief and record the requested/effective status in bookkeeping. They do not reinitialize a belief or acceptance timestamp.
- Empty-evidence assessments are allowed only if belief status remains unchanged. Investigation edits are recorded separately from evidence-response assessments; they cannot satisfy evidence deadlines. Actual belief changes still require available evidence.
- `peer_failure_policy: chair_with_available` allows the chair to proceed after a peer's bounded retry budget is exhausted, with explicit unavailability notices. Missing peers are recorded by stage/round/peer. No hidden extra peer retries on chair repair; no implicit agreement; only validated chair actions commit. Corrupt journals/storage errors still stop execution. Default policy remains require_all for other configurations.
- `stage_failure_policy: retain_scientific_scores` preserves scientific scores from the realized trace while retaining failed-stage/run flags. The old zero penalty is saved separately. Defaults remain zero_run for other configurations. This is not an assertion that a failed trace equals a clean execution.

Validation: 79 targeted tests passed, covering Codex adapter, workflow, coordination, bookkeeping, peer fallback, prompting, and federation. Ruff checks passed. The live smoke uses one deliberative exploration stage per Codex model (two peer drafts and chair), real public clinical data context, medium/default, and the frozen repaired build. It validates the chair's stage form and applies it transactionally. Smoke outputs are excluded from the experiment's 360 cells; token costs are reported separately.

The living report's scientific F1* uses the already saved diagnostic (unpenalized) score for both generations. Its penalty F1* applies the same old zero rule to both. Raw original reports remain unchanged. Gemma and Qwen remain exposed to the old bookkeeping and failure behavior, so this is a mixed-implementation preliminary comparison, not a controlled estimate of a model or deliberation effect. Four retained non-adapter Codex failures also remain under the older rules. Final analysis should stratify by implementation and flag peer fallback stages.

Final checks once all 360 selected identities are terminal:

1. Verify exactly one selected result per original run identity: replacement root for the 104 selected IDs, original root for the other 256. No fallback to an old contaminated score if a replacement fails.
2. Verify frozen source/input/config/selection hashes, medium/default commands, 20 identities per model/workflow and ten per version, saved report hashes, stage completion/failure status, scientific metrics and scoring-policy labels. Keep the original 9/8 interim snapshot unchanged.
3. Reconcile Codex native events for every attempt, including retries and superseded originals. Count every completed attempt's output_tokens once; use native totals in place of coordinator totals, never add the same completion twice. Report missing usage/interruptions as unknown. Keep selected-cohort cost, superseded cost, and smoke cost separate. Record vLLM unknown costs for interrupted unreturned requests.
4. Refresh the living report with full results, then create a final frozen copy with the mixed-harness caveats, original vs replacement counts, metrics, failure and fallback counts, and token totals. Use the original paired design (one base dataset pair); do not claim across-dataset confidence intervals.
