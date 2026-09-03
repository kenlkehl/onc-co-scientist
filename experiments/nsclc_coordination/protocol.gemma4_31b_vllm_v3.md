# NSCLC semantic masking × workflow grid: Gemma-4-31B vLLM v3 protocol

## Status and reason for a fresh experiment

This is a fresh prospective run using the served alias `gemma4-31b`
(`RedHatAI/Gemma-4-31B-IT-FP8-Dynamic`) at
`http://172.24.216.113:1234/v1`. It does not resume or pool Gemma v1/v2, Qwen,
or Luna results. V2 demonstrated that exact-schema plain-content normalization
works, but a deliberative controller generation exhausted 100,000 completion
tokens by repeating a reasoning fragment. Its two adapter retries replayed the
same request and seed, so all three physical attempts reproduced the failure.
V3 changes the prospective runtime policy before collecting any v3 result.

## Fixed scientific design

The tasks, public datasets, semantic conditions, workflows, stages, schedule
seed, iteration count, replicates, and scorer are unchanged. The design is two
semantic conditions (`named`, `masked`) by three workflows (`persistent`,
`sequential`, `deliberative`), five replicates per cell, and 20 fixed iterations
of hypothesis generation, analysis, critique, and synthesis. Smoke and main
outputs use new v3 roots and identifiers. Prior roots remain immutable and are
excluded from v3 inference.

## Model, sampling, and response enforcement

Every initial model request uses `gemma4-31b`, temperature 1.0, top-p 0.95,
top-k 64, repetition penalty 1.1, and an attempt-specific deterministic seed.
Explicit thinking remains enabled on initial requests. Controller turns have a
16,000-token completion ceiling; artifact and artifact-repair turns retain a
100,000-token ceiling.

Controller turns offer the strict native functions `run_python` and
`finish_stage`; artifact turns force `submit_stage_artifact` with the exact
stage schema. Native parsed calls remain preferred. If and only if vLLM returns
no parsed tool call, one bare JSON object or one complete JSON/unlabelled
Markdown fence may be normalized as an artifact. It must pass both the exact
stage JSON Schema and the semantic cross-field contract. Invalid, empty,
truncated, or schema-incompatible output is never accepted.

Each physical attempt saves its exact request, raw SDK response, error or
success record, finish reason, seed, timing, and token usage. A rejected model
response is retried with a different deterministic seed and a corrective user
message containing the exact validation error. When `finish_reason=length`,
the corrective message explicitly forbids resuming or repeating the failed
reasoning and requests the required response immediately; thinking is disabled
only for that recovery attempt. Failed-attempt token usage is included in the
call total. API transport failures remain bounded by the same retry count.

## Sandbox, budgets, and state

Python runs through bubblewrap with the copied public workspace mounted
read-only, only the per-session scratch directory writable, the external
`/home/klkehl/thisenv` environment mounted read-only, and networking disabled.
Evaluator assets, mappings, sibling workspaces, credentials, and parent result
trees are absent. Each harness call requires at least one and permits at most 32
Python actions and 40 controller decisions. Python actions have a 300-second
limit. Each model turn permits two retry attempts after its first generation;
each final artifact permits two separate contract-repair turns. The adapter has
a 14,380-second deadline inside the harness's 14,400-second deadline, and one
HTTP request may consume at most 7,200 seconds within that shared deadline.

Only complete prompt/accepted-artifact pairs enter bounded session history.
History is keyed by the harness session graph and trimmed to 180,000
characters. Failed responses and recovery prompts never enter later stage or
session state.

## Isolation, provenance, and launch gate

The named and masked public workspaces contain only `dataset.parquet` and
`dataset_description.md`. Their shapes, hashes, row/value/dtype parity,
descriptions, private evaluator manifests, mapping, generated configs, adapter,
environment, endpoint metadata, deterministic schedules, scorer, and
implementation commit are frozen before the first v3 scientific call.

The excluded one-iteration smoke has six runs and 40 calls at concurrency two.
The 30-run main grid has at most 4,000 calls and launches at concurrency six
only after every smoke run completes and all call artifacts, response modes,
schemas, semantic contracts, ceilings, sampling parameters, recovery records,
token accounting, sandbox mounts, endpoint locks, and deterministic scoring
pass. Any technical smoke failure blocks the main run.

## Endpoints

The primary endpoint is evidence-supported exact recovery at a synthesis
checkpoint on or before iteration 20. The key secondary endpoint is supported
exact recovery retained at terminal synthesis. Secondary diagnostics include
near/component recovery, critique rescue, later loss, persistence, unsupported
convergence, malformed output, technical failure, timeout, normalization mode,
recovery and repair incidence, token use, model turns, tool calls, and wall
time. Results remain separated by all six cells; failures stay in denominators,
and n=5 cell comparisons are descriptive rather than confirmatory.
