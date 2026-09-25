# Gemma federation on eight GCP GPUs

Authorized 2026-09-21: use `profile-notes`, a Spot `g4-standard-384`, eight
independent NVIDIA Gemma 4 31B NVFP4 replicas. The initial $2,000 spending cap
and fixed runtime cutoff were explicitly removed by the user later that day.
Resume saved FP8 responses and explicitly record the quantization switch.

The dedicated instance is `gemma-fed-nvfp4-8gpu-20260921` in `us-west1-a`.
All inference servers bind to loopback (8000–8007), accessed through SSH
tunnels on the experiment host (18000–18007). No public inference firewall
rule or cloud credential inside a model container is needed.

`startup.sh` pins vLLM 0.29.0 by container digest and the NVIDIA checkpoint
by commit. It uses one GPU per container, full 262,144-token context,
64 server sequences per replica, prefix caching and asynchronous scheduling.
Google's temperature 1.0, top-p 0.95 and top-k 64 are set both in server
defaults and existing audited client requests. Both parsers are `gemma4`,
with automatic tool choice enabled. Scientific calls retain their existing
tool-free aggregate-evidence contract; parser checks do not give scientific
agents access to files, network tools or evaluation oracles.

`probe_pool.py` checks each replica's model alias, reasoning, JSON output and
automatic function-call parsing before releasing experiments.

`resume_on_pool.py` loads and verifies the original frozen driver and science
source. It changes only worker scheduling and provider transport in memory.
All 120 identities run concurrently, evenly assigned to GPUs for stable cache
locality (15 experiment workers per GPU). Existing request bodies and call IDs
are unchanged, enabling exact replay of saved successes. Each new provider
attempt has a `deployment.json` sidecar identifying its GPU, model revision and
NVFP4 quantization. Earlier attempts are untouched. The scientific report
explicitly labels the mixed FP8/NVFP4 continuation cohort.

The deployment manifest and copied scripts live under the existing experiment's
`control/gcp-nvfp4/`. Never edit its frozen science or configuration files.

`budget_guard.py` retains the completion/exit and explicit-pause shutdown
checks. With `budget_guard_enabled: false`, it monitors estimated runtime cost
but imposes no spending or time cutoff. The deployment's cap and deadline are
null; the original policy is retained in an audit backup. GCP's separate
48-hour limit must also be cleared. The $40/hour cost envelope is only an
estimate, not a GCP billing statement. On completion, stop this dedicated VM
promptly and retain its disk and logs. Gemini and unrelated instances remain
untouched. Do not automatically delete the VM or disk.

References:

- https://ai.google.dev/gemma/docs/core/model_card_4
- https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
- https://recipes.vllm.ai/Google/gemma-4-31B-it
- https://github.com/vllm-project/vllm/releases/tag/v0.29.0
- https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#G4

`live_status.py` publishes `CLOUD_LIVE_PROGRESS.md` every 30 seconds from live
server counters and the controller/budget records. This avoids scanning the
entire response archive for each heartbeat. The original `LIVE_PROGRESS.md`
retains scientific receipt totals and per-run details, with its own snapshot time.

## Parallel site scheduling trial, 2026-09-21

The user authorized trying higher concurrency on the existing eight GPUs. The
launcher now accepts `--parallel-sites --requests-per-gpu 20`. This opt-in mode
derives scheduling wrappers from the verified frozen method bodies in memory.
It concurrently executes independent site consultation/repair loops, gathers
handoffs in the original site order, and waits for every site before the central
call. Peers within each site remain sequential. Prompts, call identities, model
settings, dataset files, ledger application, validation and scoring are unchanged.

The provider retains a lock for catalog rendering and a separate lock for each
durable call identity, allowing distinct sites to issue concurrent requests while
deduplicating repeat calls. The shared call budget atomically reserves in-flight
calls, including cleanup on failure. Tests compare serial and parallel central
prompts and handoffs, exercise actual overlapping requests and replay, check budget
limits, and run the existing full single-site/federated scientific equivalence grid.

`control/gcp-nvfp4/concurrency.json` controls admission without a controller restart.
The initial limit is 20 requests per GPU, with FIFO admission. The live status
publisher reduces the limit by two (minimum 15), at most once per minute, if it
observes at least three new GPU preemptions in one sample. These changes affect
new admission only; existing calls finish normally. Saved cloud status samples
in `cloud-live-history.jsonl` support throughput comparisons after replay settles.
Original controller/provenance snapshots are in `before-parallel-20260921/`.
The controller restarted at 2026-09-21T21:34:14Z; all saved calls are replayed.
Model servers remained up; unfinished HTTP requests were cancelled and may retry.
The parallel-site trial initially retained the original deadline; the user's
subsequent instruction removes it while preserving shutdown on completion.
## September 24 preliminary cohort

`deadline_scheduler.py` is an opt-in operational scheduler for the frozen 120-run
grid. It verifies every frozen manifest hash and validates an outcome-independent
selection of repeats 1–3 in every cell (36 identities), with a nested repeats 1–2
core (24). It calls the original frozen `run_cell(..., resume=True)`, reuses saved
successes, and retains the original infrastructure-retry archival behavior. The
other 84 identities are marked deferred in execution state; their scientific
files are preserved. An operational `execution_repeat_limit=2` allows the planned
balanced fallback after a controlled restart, without changing the target lists.

Enable this mode with `resume_on_pool.py --selection SELECTION.json
--hybrid-endpoints ENDPOINTS.json --workers 36 --parallel-sites`, in addition to
the existing root/deployment/stop-on-exit arguments and frozen `PYTHONPATH`.
Endpoint records specify `id`, `kind` (`cloud` or `local`), `url`, `limit`, and
deployment provenance. `hybrid_pool.py` gives core requests admission priority,
limits local and cloud concurrency independently, and balances admitted work.
The live `concurrency.json` fields are `requests_per_gpu` and `local_requests`.
Request bodies and identities are unchanged. A per-request provider copy prevents
parallel sites from changing one another's destination during retry. Each new
attempt receives its actual endpoint's provenance; unknown local revisions stay
unknown. A terminal transport error releases admission and gives that endpoint a
60-second cooldown; the existing scientific infrastructure retry can use another
healthy endpoint.

Execution and admission counters update every five seconds. Slow historical
receipt scans run separately. The live report distinguishes experiment-admitted
requests from shared-server counters and shows selected, core, and deferred counts.
The existing completion watcher still stops only the dedicated GCP VM. During a
planned controller transition, terminate the old watcher before the controller,
preserve launch/state records, and attach a new watcher immediately after launch.
Do not stop the shared local endpoint.

On selected-cohort completion, the frozen scientific scoring functions produce
`analysis/deadline_20260924/REPORT.md` and CSV/JSON tables for both nested cohorts,
with explicit 3- and 2-repeat denominators. To take a preliminary snapshot during
execution, run `deadline_scheduler.py --root ROOT` with the frozen source on
`PYTHONPATH`. This entry point only scores existing artifacts; it does not start
model calls or change workers. The scheduled monitor handles the September 23
noon EDT fallback checkpoint and September 24 09:00 scoring / 11:00 report target.

## Approved context-fit output allowance, September 23

The user approved an explicit output-policy amendment for the selected cohort.
Pass `--context-fit-policy control/gcp-nvfp4/context-fit-policy.json` using an
absolute path. `context_fit.py` tokenizes each new request on its selected server
and reserves `min(65536, 262144 - input_tokens - 512)` output tokens. If that value
is below 32768, no generation is sent and that identity pauses without repeating
the same infrastructure retry. The full prompt, Google sampling, reasoning
settings, and scientific source/config remain unchanged. The frozen adapter still
rejects answers with `finish_reason=length`.

The provider's original `request.json` and nominal call identity remain intact,
so successful saved answers replay exactly. A call's `context-fit/` directory
records tokenizer results, both allowances, request/prompt hashes, and a separately
saved amended wire body before generation. Every new provider attempt receives a
`context-fit.json` sidecar linking to that actual request; earlier attempts are
untouched. Reports label this as a mixed output-policy continuation. Tokenization
uses the existing authenticated endpoint and never logs authorization headers or
token IDs. Context counts must match the approved 262144-token window.

For this rollout, admission was set to zero while current calls finished. Then the
watcher was stopped before the drained controller, and both were restored with the
approved policy. Cumulative hybrid counters are retained across this restart.
`context-fit-rollout.json` and its backup directory record tests, frozen/report
hashes, launch commands, and pre-transition state. Future restarts must retain the
policy flag, the shutdown watcher, and the exact selected identities.
