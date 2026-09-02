# NSCLC semantic masking × workflow grid: Qwen3.8-27B vLLM v1 protocol

## Status and independent objective

This is a new prospective experiment using `Qwen/Qwen3.8-27B` at the
OpenAI-compatible endpoint `http://camus.dfci.harvard.edu:8060/v1`. It neither
interrupts nor modifies the concurrent Luna-medium local-env v4 experiment,
and its smoke or main results will not be pooled with Luna results. A preflight
on 2026-09-02 observed vLLM `0.27.1`, a 262,144-token advertised model context,
working chat completions, and working JSON-schema response formatting. Native
tool calls were unavailable because the server was not launched with a tool-call
parser, so tool use is controlled by the client adapter described below.

The scientific task, datasets, semantic conditions, workflow conditions,
stages, iteration count, replicates, schedule seed, and scorer are held equal to
the Luna v4 grid. The design is two semantic conditions (`named`, `masked`) by
three workflows (`persistent`, `sequential`, `deliberative`), with five
replicates per cell and 20 fixed iterations of hypothesis generation, analysis,
critique, and synthesis. This is a model/runtime comparison, not a claim that
inference resources are matched across providers.

## Model and controller policy

Every model request uses `Qwen/Qwen3.8-27B`, temperature 0.2, top-p 0.95, and a
100,000-token completion ceiling shared by reasoning and the visible response.
Each model turn receives a deterministic seed derived from the harness request
identity, prompt hash, turn number, and schema name. Replicates have distinct
request identities. The adapter uses the vLLM OpenAI-compatible Chat
Completions API with a strict JSON schema on every turn.

On controller turns Qwen must choose either `python` or `final`. A Python action
contains self-contained code. The adapter runs it through bubblewrap with only
the copied public workspace mounted read-only, that session's scratch directory
mounted writable, the local `/home/klkehl/thisenv` scientific environment
mounted read-only, and network disabled. Parent experiment directories,
evaluator assets, sibling workspaces, credentials, and host network are absent.
Python is limited to 300 seconds per action, 16 MiB per output file, 256 file
descriptors, and at most 40,000 characters returned to model context per stream.
Each harness call requires at least one and permits at most 32 Python actions
and 40 controller decisions.

The adapter persists only complete pairs of prior stage prompts and accepted
stage artifacts under a hash of the harness `session_id`; within-call reasoning,
code, and tool results remain fully audited on disk but are not promoted to
cross-call memory. Persistent sessions retain the newest complete pairs up to a
180,000-character history ceiling, dropping the oldest complete pair first.
Sequential and one-round deliberative sessions remain fresh according to the
same harness session graph used for Luna.

## Output enforcement, retries, and audit

Final generation uses the exact stage-specific schema used by the Luna v4
adapter. Synthesis requires a non-empty `{conclusion,
supported_claim_indices}` object and intermediate stages require JSON null.
Client-side validation additionally requires every supported claim index to be
unique, in range, and point to a claim whose `supported` value is true.

Each harness call permits at most two artifact contract-repair turns. A repair
receives the exact validation error in the same in-memory conversation and is
instructed to reuse completed analysis without more tools. Separately, each API
request permits at most two retries after retryable transport, timeout, rate
limit, or server errors; HTTP 4xx configuration errors other than 408, 409, and
429 are terminal. The adapter has one shared 14,380-second call deadline inside
the harness's 14,400-second deadline, and an individual API generation may run
for up to 7,200 seconds within that shared deadline. Raw API requests and
responses, separate reasoning fields returned by vLLM, seeds, token usage,
every tool program and result, validation failures, retry attempts, accepted
artifacts, and bounded session state are retained. Failed physical API attempts
without provider usage metadata cannot contribute token counts and are reported
separately.

## Data isolation and scientific task

The named condition exposes the candidate indicators
`treatment_pembrolizumab`, `treatment_sotorasib`, `treatment_olaparib`, and
`treatment_osimertinib`; the masked condition exposes `feature_012`,
`feature_018`, `feature_020`, and `feature_027`. Both identify `pfs_months` as
the outcome and all remaining fields as possible modifiers or covariates. The
masked model receives no clinical mapping. Exact recovery is evaluated in each
condition's native namespace, with evaluator mappings used only by the trusted
post-run scorer.

The source hashes, 50,000 × 35 shapes, exact named/masked row-value-dtype parity,
minimal two-file public workspaces, seeded schedule, configuration fingerprint,
implementation commit, model endpoint metadata, adapter hash, and scorer are
frozen before the first scientific call. Resume may only reuse a matching
fingerprint and substrate hashes.

## Calls, gate, and endpoints

Persistent and sequential runs contain 80 harness calls; deliberative runs
contain 240. The 30-run main grid therefore contains 4,000 harness calls, 2,400
logical stage checkpoints, and 600 synthesis checkpoints. Model turns and
Python actions are additional native-resource measures. The excluded
one-iteration smoke contains six runs and 40 harness calls at concurrency two.
The main grid launches automatically at concurrency six only if all smoke runs,
calls, stage contracts, sandbox/audit invariants, and model/endpoint locks pass.

The primary endpoint is evidence-supported exact recovery at a synthesis
checkpoint on or before iteration 20. The key secondary endpoint is supported
exact recovery retained at the iteration-20 terminal synthesis. Other endpoints
remain those of the Luna v4 protocol: first recovery iteration and call,
near/component recovery, critique rescue, later loss, persistence, unsupported
convergence, malformed output, technical failure, timeout, repair incidence and
success, and recovery normalized by calls, model turns, output tokens, and wall
time. Results are reported separately for all six cells with technical failures
and truncations retained in denominators; no confirmatory workflow-superiority
p-values are used with five replicates per cell.
