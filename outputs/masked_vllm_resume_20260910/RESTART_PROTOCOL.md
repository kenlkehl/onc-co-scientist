# Masked Qwen/Gemma execution after September 10 repairs

The user authorized resuming jobs with the completed fixes, and explicitly chose to run masked Qwen/Gemma alongside the unmasked jobs on the shared servers.

The unmasked task already started its corrected 120-run cohort under `20260910_vllm_repair`. This task starts the 120 previously queued masked identities under `20260910_masked_vllm_repair`, with six concurrent workers alongside the 24 unmasked workers. The original masked driver continues its 240 Codex identities unchanged, so total masked concurrency can temporarily reach 36. Scientific budgets, iterations, models, workflows, repeats, datasets, mappings and seeds are unchanged.

Every Python source file in the corrected masked snapshot matches the frozen repaired unmasked snapshot. This includes the final-JSON parser, rejected-response feedback, bookkeeping fixes, available-peer fallback, and Qwen's thinking-disabled final retry. Gemma does not disable thinking. The existing masked Codex results and active calls retain their original implementation and journals.

No masked Qwen/Gemma identity had started before this handoff. The old driver's gate references the retired unfinished September 9 continuation control file. Leave that file unfinished: making it completed would release duplicate jobs using the older code. The new supervisor owns the 120 masked vLLM identities exclusively. After all 240 Codex identities are terminal and the old scheduler reports zero active jobs, the supervisor retires only that idle scheduler and its old monitor. It does not signal an active Codex worker.

Combined progress is written to `20260910_masked_vllm_repair/LIVE_PROGRESS.md` and `20260910_clinical10pct_masked_workflows/COMBINED_PROGRESS.md`. `selected_run_roots.json` assigns exactly one output root to each of the 360 identities. When all are terminal, the supervisor writes the combined report to `RESULTS.md` in both roots. It retains per-run implementation and failure/repair provenance; it does not merge journals or silently replace outcomes.

Preparation verified matching frozen hashes, identical masked inputs and mappings, and exactly 120 unique vLLM identities. The masked numerical-equivalence, all-workflow integration, leakage and endpoint-gate tests are rerun against the repaired frozen source before launch. The unmasked task's live endpoint smoke checks remain recorded under `outputs/vllm_restart_20260910/`; new live responses are recorded in the corrected masked run journals.
