# Eight-cell H100 CAA pilot with scheduled pause

The user authorized eight fresh cells: control and CAA × expected/surprising × unmasked/masked, one replicate each. Each cell uses 25 persistent iterations, 100 primary stages, 100,000 maximum generated tokens per call, a six-hour request timeout, the same BF16 checkpoint and smoke-derived vectors as the accepted one-cell gate, CPU-staged transfers, dynamic cache, and 2,048-token prefill chunks. GPUs are limited to physical 0 and 1, using `/data1/ken/envs/gptoss3/bin/python`. The private server uses port 18765.

## Current session — resumed September 19 afternoon

Resumed at approximately 14:52 EDT on September 19, 2026. **The current automatic pause is Monday, September 21 at 08:55 EDT (12:55 UTC)**, if the eight-cell pilot has not already finished. Completion or pause releases the model's GPU allocations. The timer is armed in `runtime/status.json`; no automatic resume or larger campaign launch is scheduled. GPU monitoring has been restarted.

Five cells completed before this session. The sixth cell (surprising/unmasked control) completed 7 iterations and resumes at iteration 8 explore, with 30 saved responses. Two cells remain queued. At session start, 528/800 primary stages were committed and 272 remained (34%). Completed cells are skipped and durable responses replayed. The September 19 08:55 cutoff fired successfully; prior session statuses are retained under `runtime/sessions/`.

## Original session automatic pause

**September 17, 2026, at 08:55 America/New_York (EDT, UTC−04:00), which is 12:55 UTC.** This supersedes the initially requested 07:30 time.

The detached supervisor arms a POSIX real-time alarm when it starts. This is local process scheduling and does not depend on a future assistant turn. It handles the cutoff even while waiting for startup or an inference request. On the cutoff it stops its controller first, then its model server, releases the model's GPU allocations, and records `status: paused` and `resume_available: true`. Preexisting GPU users are untouched. Earlier scientific or infrastructure failure still halts the campaign and releases its server. There is no automatic resume.

Live status: `runtime/status.json`. The `pause_at_utc` and `pause_timer_armed` fields record the deadline and armed state. Logs and frozen server/controller sources are under `runtime/`.

Campaign: `/data1/ken/onc-co-scientist/data/caa_discovery/20260916T2005_h100_pilot`.

## Resume when GPU use is authorized again

From the repository root, after the status says `paused`:

```bash
bash outputs/caa_discovery/20260916T2005_h100_pilot/resume.sh
```

To impose a new deadline, pass an explicit timezone offset:

```bash
bash outputs/caa_discovery/20260916T2005_h100_pilot/resume.sh \
  --pause-at YYYY-MM-DDTHH:MM:SS-04:00
```

The helper invokes the frozen launcher, validates the frozen campaign and current serving fingerprint, skips completed cells, and reconstructs any partial cell from its saved call journal. All saved responses are checked against the reconstructed prompts and result hashes; completed requests are reused. Partial scientific transcripts are retained, and replay creates a new numbered transcript. The original launch configuration remains unchanged; prior session status is archived under `runtime/sessions/`.

A response still being generated, or not atomically saved when the cutoff occurs, must be generated again. Individual generation state/KV cache is not checkpointed. This is resumability at the saved-response boundary, not token-level continuation. Do not change the frozen source, model/vector files, or installed dependencies between sessions. A failed/halted campaign is not eligible for automatic resume.

The fallback supervisor limit is 96 hours per resumed session; reaching it also produces a resumable pause. A lock prevents concurrent supervisors for this runtime. Resume does not reuse the expired September 17 cutoff unless explicitly supplied, and an already elapsed deadline is rejected by the resume command.

## Validation and measurements

`validation.txt` and `pause_resume_tests.log` record passing checks. These include real timer interruption with actual child processes, preservation of saved calls, frozen-manifest mismatch rejection, remaining-cell limits, and interrupted scientific-workflow replay without duplicate completed model calls.

`gpu_samples.jsonl` records GPU 0/1 whole-device usage every 15 seconds. The baseline includes 2,802 MiB of unrelated allocations on each device. Monitoring ends after this session pauses/completes/fails; it does not restart inference. The prior acceptance run used about 7.5 hours per cell, so this overnight window is unlikely to complete all eight cells.
