# H100 execution of CAA_DISCOVERY.md

Started 2026-09-15 at 21:52:50 UTC (17:52:50 America/New_York).

## Current allocation and protocol

- Python: `/data1/ken/envs/gptoss3/bin/python`.
- GPUs: physical 0 and 1, NVIDIA H100 NVL, 95,830 MiB each.
- Checkpoint: `/data1/ken/models/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343`.
- All seven checkpoint files verified against content-addressed blob hashes; receipt: `preflight.json`.
- Runtime differences from the handoff: Python 3.13.6, torch 2.13.0+cu130, OpenAI SDK 3.3.1, Uvicorn 0.52.4. Transformers 5.14.1 and Accelerate 1.14.0 match. No installed packages changed.
- Port 8765 was occupied. Added `--port` to the launcher and `--base-url` to campaign preparation; this run uses `127.0.0.1:18765`. Existing services preserved.
- BF16, balanced placement, SDPA, dynamic KV cache, CPU-staged inter-GPU transfers, 2,048-token prefill chunks.
- CAA candidate: layer 40, scale -0.05, constructed orthogonalized appraisal direction; development selection remains unvalidated as described in the plan.

## Completed preflight

- Required test suite: 35 passed (`preflight_tests.log`).
- Campaign tests after custom-port coverage: 12 passed (`port_tests.log`).
- Numerical smoke: completed; all three generations stopped normally and returned correct JSON, all validation flags true.
- Load 25.0 s; derive 26 pairs at layers 20/30/40 in 13.0 s.
- Peak smoke allocated GPU memory: 29.48 / 29.36 GiB.
- Frozen acceptance campaign verification passed, including dataset checks and row-preserving masking validation during preparation.

## Acceptance completed — gate passed

Completed 2026-09-16 at 05:24:27 UTC: 25 iterations, 100 committed stages, 102 calls. The owned server was stopped. See [acceptance_review.md](acceptance_review.md) for scientific results and limitations.

Runtime: `acceptance_runtime/`; final status: `acceptance_runtime/status.json`.

Campaign: `/data1/ken/onc-co-scientist/data/caa_discovery/20260915T1750_h100_acceptance`.

Supervisor PID at launch: 144349. Server PID: 144350. Controller PID: 144465.
Check process command lines and live status before signaling a PID.

Full 25 iterations, 100 primary stages, 100,000 generated tokens per call, up to two scientific repairs, initial thinking enabled, six-hour request timeout. One cell is scheduled for execution; seven remain queued. Supervisor has a 48-hour ceiling and releases its owned server/controller on completion or failure. It stops on a failed cell; it does not restart or reroll it.

GPU memory/utilization are sampled every 15 seconds in `gpu_samples.jsonl`. These are whole-device measurements including the approximately 2,802 MiB preexisting allocations on each GPU. They are sampled usage, not allocator-exact peaks. The monitor exits when the supervisor records an end time.

## Review and subsequent gates

This is not a completed scientific evaluation. After acceptance ends, inspect its `run.json`, `report.json`, coordination audit, calls, and scientific transcripts. Require 25 complete iterations and 100 committed stages, actual analysis execution, coherent evidence appraisal and validation handling. Summarize repairs, truncations, token counts, durations, and memory samples. Preserve a failed campaign intact and diagnose before creating a new attempt.

If acceptance passes review, launch the fresh eight-cell, one-replicate pilot with `--iterations 25 --run-limit 8 --port 18765` and the same remaining settings. Review both arms and all four conditions before the 80-cell evaluation. For the full campaign, use server `--port 18765`, manifest URL `http://127.0.0.1:18765/v1/caa`, and prepare `--base-url http://127.0.0.1:18765/v1`. Keep the 100,000-token limit and 25 iterations throughout.

No eight-cell pilot or 80-cell evaluation has been launched yet.
