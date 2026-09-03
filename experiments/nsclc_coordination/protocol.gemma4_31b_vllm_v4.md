# NSCLC semantic masking × workflow grid: Gemma-4-31B vLLM v4 protocol

## Status and reason for a fresh experiment

This is a fresh prospective run using the served alias `gemma4-31b`
(`RedHatAI/Gemma-4-31B-IT-FP8-Dynamic`) at
`http://172.24.216.113:1234/v1`. It does not resume or pool Gemma v1-v3,
Qwen, or Luna results. V3 eliminated the prior 100,000-token repetition
failure: five of six smoke cells completed, and no response ended for length.
The remaining named-deliberative smoke cell failed at its terminal synthesis
chair because Gemma attempted to submit an artifact during a controller turn.
Four similar v3 incidents recovered after one fresh-seed retry; this one did
not. V4 fixes that phase-routing failure before collecting any v4 result.

## Fixed scientific design

The tasks, public datasets, semantic conditions, workflows, stages, schedule
seed, iteration count, replicates, and scorer are unchanged. The design is two
semantic conditions (`named`, `masked`) by three workflows (`persistent`,
`sequential`, `deliberative`), five replicates per cell, and 20 fixed iterations
of hypothesis generation, analysis, critique, and synthesis. V4 uses new smoke
and main roots. Every prior result remains immutable and excluded.

## Model, sampling, and phase enforcement

Every initial request uses `gemma4-31b`, temperature 1.0, top-p 0.95, top-k 64,
repetition penalty 1.1, and an attempt-specific deterministic seed. Explicit
thinking remains enabled initially. Controller turns have a 16,000-token
completion ceiling; artifact and artifact-repair turns retain 100,000 tokens.

Every controller request explicitly instructs Gemma to call exactly one of
`run_python` or `finish_stage`, forbids `submit_stage_artifact` and artifact
text on that phase, and explains that an artifact turn follows
`finish_stage`. A retry names the exact currently available functions and
forbids every other function rather than referring generically to a required
tool. The model must complete at least one sandboxed Python action before
finalization.

As a narrow recovery for server/model phase confusion, an out-of-phase native
`submit_stage_artifact` call may be accepted only when the minimum Python-tool
count was already satisfied. Its arguments must be the artifact object itself
and must pass the exact stage JSON Schema plus the client-side semantic
contract. Wrapped, malformed, truncated, early, or semantically inconsistent
arguments remain retryable failures. This recovery is explicitly labeled and
audited; it does not add `submit_stage_artifact` to controller tools.

Native calls remain preferred. Plain-content normalization remains limited to
one bare JSON object or one complete JSON/unlabelled Markdown fence without
surrounding prose, followed by the same schema and semantic checks. Each
physical attempt saves its exact request, raw response, error or success,
finish reason, seed, timing, and token usage. Rejected output is retried with a
new deterministic seed and exact corrective feedback. A length recovery also
disables thinking for that recovery attempt only. Failed-attempt tokens remain
included in call totals.

## Sandbox, budgets, and state

Python runs through bubblewrap with the copied public workspace mounted
read-only, only the per-session scratch directory writable, the external
`/home/klkehl/thisenv` environment mounted read-only, and networking disabled.
Evaluator assets, mappings, sibling workspaces, credentials, and parent result
trees are absent. Calls require at least one and permit at most 32 Python
actions and 40 controller decisions. Python actions have a 300-second limit.
Each model turn permits two retries after its first generation; each final
artifact permits two contract-repair turns. The adapter has a 14,380-second
deadline inside the harness's 14,400-second deadline; one HTTP request may use
at most 7,200 seconds within that shared deadline.

Only complete prompt/accepted-artifact pairs enter session history. History is
keyed by the harness session graph and trimmed to 180,000 characters. Failed
responses and recovery prompts never enter later stage or session state.

## Provenance and launch gate

The named and masked public workspaces contain only `dataset.parquet` and
`dataset_description.md`. Their shapes, hashes, row/value/dtype parity,
descriptions, private evaluator manifests, mapping, configs, adapter,
environment, endpoint metadata, schedules, scorer, and implementation commit
are frozen before the first v4 scientific call.

The excluded one-iteration smoke has six runs and 40 calls at concurrency two.
The 30-run main grid has at most 4,000 calls and launches at concurrency six
only after every smoke run completes and all artifacts, phase-normalization
modes, schemas, semantic contracts, ceilings, sampling parameters, recovery
records, accounting, sandbox mounts, endpoint locks, and scoring pass. Any
technical smoke failure blocks main.

## Endpoints

The primary endpoint is evidence-supported exact recovery at a synthesis
checkpoint on or before iteration 20. The key secondary endpoint is supported
exact recovery retained at terminal synthesis. Secondary diagnostics include
near/component recovery, critique rescue, later loss, persistence, unsupported
convergence, malformed output, technical failure, timeout, normalization mode,
recovery and repair incidence, token use, model turns, tool calls, and wall
time. Results remain separated by all six cells; failures stay in denominators,
and n=5 cell comparisons are descriptive rather than confirmatory.
