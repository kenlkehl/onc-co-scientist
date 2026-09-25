# Local persistent-agent CAA pilot — launched September 14, 2026

This is a real-tool engineering pilot, not a completed efficacy experiment or a
replacement for the recent 25-iteration analysis. Live state and results are in the
artifact directories below; this document records the fixed launch configuration.

## Outcome — stopped September 15

Stopped at approximately 08:58 UTC on September 15; the owned supervisor, controller
and model server exited and both GPUs returned to idle. All frozen inputs, source,
requests and responses are preserved. No replacement campaign was launched.

The first two cells failed with zero complete iterations and 69 recorded calls each.
Their 2,048-token responses repeatedly truncated; later persistent prompts triggered
61 and 60 CUDA OOM errors respectively. The third cell was interrupted during further
OOM failures; five cells were still queued. Some analysis/tool calls worked, but this
is not evidence for or against steering efficacy. Schema-invalid hypotheses remain
invalid; the controller's scientific requirements were not relaxed.

The initial supervisor watched process liveness only, so a still-alive server returning
OOM errors did not stop the pilot. Transport retries and scientific repair attempts
then repeatedly retried infrastructure failures. Current source fixes this with
100,000-token calls independent of iteration count, no hidden CAA transport retries,
immediate infrastructure-failure propagation, campaign halting, a latched server
failure state, and one-cell acceptance gating. Chunked prefill reduces prompt-pass
memory but does not establish long-generation capacity. See `CAA_DISCOVERY.md` for
the updated launch protocol. These changes deliberately do not modify the historical
frozen campaign below or reclassify its results.

## Fixed protocol

- One replicate of each expected/surprising × unmasked/masked condition, with
  fresh unsteered and CAA arms: eight serial runs.
- Persistent-agent workflow, six iterations per run, four stages per iteration;
  24 planned primary calls per run (192 total), plus the existing repair attempts.
- 2,048 generated tokens per call, initially thinking-enabled, with the existing
  final-retry thinking fallback. Same Gemma sampling and repair policy in both arms.
- Same verified September 10 named/masked data packages and key-discovery framing.
- Gemma 4 31B IT BF16, exact local checkpoint
  `145dc2508c480a64b47242f160d286cff94a2343`; `CUDA_VISIBLE_DEVICES=0,1`.
- Balanced GPU placement, SDPA, dynamic cache, explicit CPU-staged inter-GPU
  transfers; no CPU/disk weight offload or remote inference endpoint.
- CAA: constructed `paradigm_orthogonalized` direction, zero-based layer 40,
  additive scale -0.05, applied during prefill and generation. This direction and
  scale are unvalidated engineering choices. No random-vector evaluation arm here.
- No constrained JSON decoding: common prompts, parsing, repair and controller
  validation are used for both arms.
- Detached supervisor, 12-hour wall-clock ceiling, automatic shutdown of its own
  server and controller on completion/failure. Cached calls remain on disk; terminal
  failures are retained, not rerolled. No completion-notification automation was created.

## Frozen artifacts and live status

Runtime directory:

`outputs/caa_discovery/20260915T0050_persistent_local/`

This contains `status.json`, `launch.json`, `server_manifest.json`, copied vectors,
the serving source snapshot, and `server.log`, `controller.log`, `supervisor.log`.
Launch time was 2026-09-15 00:46:47 UTC (September 14, 8:46 p.m. Eastern).

Campaign directory:

`data/caa_discovery/20260915T0050_persistent_local/`

This contains the frozen source, data, configuration, schedule and manifest. Each
view's `runs/<run_id>/` stores requests, cached calls and scientific transcripts.
Final summaries are written under `analysis/` after the schedule finishes.

Serving fingerprint:

`983df6171a154d9b18203256b4a7fa42faa1d92aae3002d953e3ad850c6c9aaf`

The frozen campaign verification passed; the serving source hashes match the live
manifest. Startup checks confirmed BF16 placement only on GPU devices 0 and 1 with
CPU-staged transfers. The first scheduled cell is expected/masked with CAA.

No efficacy conclusion should be drawn from launch or partial progress. Inspect
completed iterations, technical failures/truncations and absolute recovery before
interpreting any CAA-minus-control differences; one replicate per condition is
exploratory and the six-iteration policy differs from the full protocol.
