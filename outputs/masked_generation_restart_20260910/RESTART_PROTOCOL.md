# Masked local-model restart with recommended generation

User authorized deleting prior masked Qwen/Gemma results and restarting with the other task's updated caller. Only the masked local-model run directories in 20260910_masked_vllm_repair were removed. Aggregate deletion metadata is retained in deleted_runs.json; it is not scored. The old masked supervisor and suspended Gemma runner were retired. Codex and the separate unmasked runner were not signaled or changed.

Authoritative replacement root: data/expected_surprising_ledger/full_runs/20260910_masked_vllm_recommended. Its 120 identities match the previous masked selection exactly; all input hashes match. Frozen source hashes match the running 20260910_vllm_recommended unmasked root. Six workers run alongside its independent 24 workers. Existing 240 Codex identities remain owned by 20260910_clinical10pct_masked_workflows. The old Codex scheduler's vLLM gate must remain closed to prevent duplicate local-model jobs.

Generation settings come from sampling_profile=auto, explicit thinking, and JSON-object output. Qwen thinking uses temperature 1, top-p .95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1. Qwen non-thinking uses .7/.8/20/0/1.5/1. Gemma uses temperature 1, top-p .95, top-k 64. Either model may disable thinking only on the final permitted retry after two consecutive output-limit truncations of the same stage/participant/round. Ordinary parse errors do not trigger that fallback. Medium reasoning and scientific settings are preserved.

Validation: six masking/integration tests passed against the actual frozen source. The matching unmasked caller had already passed 54 tests and four live normal/fallback checks. No additional diagnostic calls are part of the experimental cohort.

monitor.py aggregates the 240 original Codex and 120 replacement local-model identities into LIVE_PROGRESS.md in the replacement root and COMBINED_PROGRESS.md in the original masked root. It never selects the deleted results. New driver control and logs live in replacement-root/control.
