# NSCLC semantic masking × workflow grid: Gemma-4-31B vLLM v2 protocol

## Status and reason for a fresh experiment

This is a fresh prospective run using the served alias `gemma4-31b`
(`RedHatAI/Gemma-4-31B-IT-FP8-Dynamic`) at
`http://172.24.216.113:1234/v1`. It does not resume or pool the failed Gemma v1
smoke or any Qwen or Luna results. Gemma v1 ended before any harness call was
accepted because vLLM 0.27.1 sometimes returned a complete artifact in
`message.content` while leaving `message.tool_calls` empty, even when the
request required or forced a function. All 17 non-empty responses implicated
by that failure passed the exact stage schema and semantic cross-field contract
when replayed; four other attempts exhausted 100,000 tokens in reasoning and
had no content. V2 changes only the response-normalization policy needed to
handle that observed server behavior.

## Fixed scientific design

The tasks, public datasets, semantic conditions, workflows, stages, schedule
seed, iteration count, replicates, and scorer are unchanged. The design is two
semantic conditions (`named`, `masked`) by three workflows (`persistent`,
`sequential`, `deliberative`), five replicates per cell, and 20 fixed iterations
of hypothesis generation, analysis, critique, and synthesis. Smoke and main
outputs use new v2 roots and identifiers. V1 remains immutable and excluded.

## Model, tools, and response enforcement

Every request uses `gemma4-31b`, temperature 0.2, top-p 0.95, a deterministic
request-derived seed, a 100,000-token completion ceiling, and explicit
`enable_thinking=true`. Controller turns offer the strict native functions
`run_python` and `finish_stage`; normal artifact turns force the strict
`submit_stage_artifact` function carrying the exact stage schema.

Native parsed function calls remain the preferred path. If and only if vLLM
returns no parsed tool call, the adapter may normalize `message.content` as an
artifact. Eligible content must be either one bare JSON object or one complete
`json`/unlabelled Markdown fence with no surrounding prose. It must pass the
exact stage JSON Schema and the client-side semantic contract, including
synthesis final-answer structure and supported-claim cross-references. A valid
artifact returned on a controller turn is accepted directly after the required
minimum Python-tool count. A valid artifact returned on an artifact or repair
turn is treated as that turn's submission. Invalid, empty, truncated, or
schema-incompatible text is retained as a failed physical attempt and follows
the same bounded API retry policy. Raw SDK-parsed responses and separate Gemma
reasoning remain saved regardless of normalization mode; success records and
turn audits label `native_tool` versus `validated_content_fallback`.

## Sandbox, budgets, and state

Python runs through bubblewrap with the copied public workspace mounted
read-only, only the per-session scratch directory writable, the external
`/home/klkehl/thisenv` environment mounted read-only, and networking disabled.
Evaluator assets, mappings, sibling workspaces, credentials, and parent result
trees are absent. Each harness call requires at least one and permits at most 32
Python actions and 40 controller decisions. Python actions have a 300-second
limit. Each API request permits two retry attempts after the first generation;
each final artifact permits two contract-repair turns. The adapter has a
14,380-second deadline within the harness's 14,400-second deadline, and one API
generation may consume at most 7,200 seconds within that shared deadline.

Only complete prompt/accepted-artifact pairs enter bounded session history.
History is keyed by the harness session graph and trimmed to 180,000
characters. Failed responses never enter subsequent stage state.

## Isolation, provenance, and launch gate

The named and masked public workspaces contain only `dataset.parquet` and
`dataset_description.md`. Their 50,000 × 35 shapes, hashes, row/value/dtype
parity, descriptions, private evaluator manifests, mapping, generated configs,
adapter, environment, endpoint metadata, deterministic schedules, scorer, and
implementation commit are frozen before the first v2 scientific call.

The excluded one-iteration smoke has six runs and 40 calls at concurrency two.
The 30-run main grid has at most 4,000 calls and launches at concurrency six
only after every smoke run completes and all call artifacts, response modes,
schemas, semantic contracts, retry limits, token accounting, sandbox mounts,
endpoint locks, and deterministic scoring pass. Any technical smoke failure
blocks the main run.

## Endpoints

The primary endpoint is evidence-supported exact recovery at a synthesis
checkpoint on or before iteration 20. The key secondary endpoint is supported
exact recovery retained at terminal synthesis. Secondary diagnostics include
near/component recovery, critique rescue, later loss, persistence, unsupported
convergence, malformed output, technical failure, timeout, normalization mode,
repair incidence, token use, model turns, tool calls, and wall time. Results
remain separated by all six cells; failures stay in denominators, and n=5 cell
comparisons are descriptive rather than confirmatory.
