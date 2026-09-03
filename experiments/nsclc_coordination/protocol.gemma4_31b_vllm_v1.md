# NSCLC semantic masking × workflow grid: Gemma-4-31B vLLM v1 protocol

## Status and independent objective

This is a fresh prospective model/runtime comparison using the served alias
`gemma4-31b` (`RedHatAI/Gemma-4-31B-IT-FP8-Dynamic`) at the OpenAI-compatible
endpoint `http://172.24.216.113:1234/v1`. It neither resumes nor pools results
from the Qwen or Luna experiments. A preflight on 2026-09-03 observed vLLM
0.27.1, a 262,144-token advertised context, and correct separation of Gemma
reasoning from strict JSON-schema content when thinking was explicitly enabled.
Preflight also found that Gemma's native function-call protocol returned compact,
well-formed controller and artifact arguments, whereas direct JSON-schema
controller output could enter an unbounded legal-whitespace loop. This
experiment therefore uses native function calls for both controller actions and
artifact submission, with client-side schema validation retained.

The scientific task, datasets, semantic conditions, workflow conditions,
stages, iteration count, replicates, schedule seed, and scorer are held equal to
the Luna v4 and Qwen v2 grids. The design is two semantic conditions (`named`,
`masked`) by three workflows (`persistent`, `sequential`, `deliberative`), with
five replicates per cell and 20 fixed iterations of hypothesis generation,
analysis, critique, and synthesis. This is a native-resource comparison; FP8
Gemma inference is not assumed to be resource-equivalent to other providers.

## Model and controller policy

Every request uses `gemma4-31b`, temperature 0.2, top-p 0.95, a 100,000-token
completion ceiling, and the explicit chat-template override
`enable_thinking=true`. Each model turn receives a deterministic seed derived
from the harness request identity, prompt hash, turn number, and schema name.
The adapter uses the vLLM Chat Completions API with required native function
calls on every turn. Controller calls expose only `run_python` and
`finish_stage`; artifact calls force the single `submit_stage_artifact`
function carrying the exact stage schema. The adapter records both the explicit
thinking override and native-tool interaction mode in requests, session
settings, and call audits.

On controller turns Gemma chooses either `python` or `final`. Python actions run
through bubblewrap with only the copied public workspace mounted read-only, the
session scratch directory mounted writable, `/home/klkehl/thisenv` mounted
read-only, and network disabled. Parent experiment directories, evaluator
assets, sibling workspaces, credentials, and the host network are absent.
Python is limited to 300 seconds per action, 16 MiB per output file, 256 file
descriptors, and 40,000 characters returned to model context per stream. Each
harness call requires at least one and permits at most 32 Python actions and 40
controller decisions.

The adapter persists only complete pairs of prior stage prompts and accepted
stage artifacts under a hash of the harness session ID. Persistent sessions
retain the newest complete pairs up to 180,000 characters; sequential and
one-round deliberative sessions follow the harness session graph used by the
other model conditions.

## Output enforcement, retries, and audit

Final artifact submission uses the exact stage-specific schema used by the Luna
and Qwen adapters as the required function's parameter schema. Synthesis
requires a non-empty `{conclusion,
supported_claim_indices}` object; intermediate stages require JSON null.
Client validation also requires supported-claim indices to be unique, in range,
and point to claims whose `supported` value is true.

Each harness call permits two artifact contract-repair turns. Each API request
permits two retries after retryable transport, timeout, rate-limit, or server
errors. The adapter has a shared 14,380-second call deadline inside the
harness's 14,400-second deadline; an individual API generation may run for up
to 7,200 seconds within that deadline. API requests and SDK-parsed responses,
separate reasoning fields, seeds, usage, tool programs and results, validation
failures, retry attempts, accepted artifacts, and bounded session state are
retained.

## Data isolation and scientific task

The named condition exposes the candidate indicators
`treatment_pembrolizumab`, `treatment_sotorasib`, `treatment_olaparib`, and
`treatment_osimertinib`; the masked condition exposes `feature_012`,
`feature_018`, `feature_020`, and `feature_027`. Both identify `pfs_months` as
the outcome and all remaining fields as possible modifiers or covariates. The
masked model receives no clinical mapping. Exact recovery is evaluated in each
condition's native namespace using evaluator mappings available only to the
trusted post-run scorer.

The source hashes, 50,000 × 35 shapes, exact named/masked row-value-dtype
parity, minimal two-file public workspaces, seeded schedule, configuration
fingerprint, implementation commit, endpoint metadata, adapter hash, and scorer
are frozen before the first scientific call. Resume may only reuse matching
fingerprints and substrate hashes.

## Calls, gate, and endpoints

Persistent and sequential runs contain 80 harness calls; deliberative runs
contain 240. The 30-run main grid therefore contains at most 4,000 harness
calls, 2,400 logical stage checkpoints, and 600 synthesis checkpoints. The
excluded one-iteration smoke contains six runs and 40 harness calls at
concurrency two. The main grid launches at concurrency six only if all smoke
runs, calls, stage contracts, sandbox/audit invariants, and endpoint locks pass.

The primary endpoint is evidence-supported exact recovery at a synthesis
checkpoint on or before iteration 20. The key secondary endpoint is supported
exact recovery retained at terminal synthesis. Other endpoints match the Luna
v4 protocol: first recovery iteration and call, near/component recovery,
critique rescue, later loss, persistence, unsupported convergence, malformed
output, technical failure, timeout, repair incidence and success, and recovery
normalized by calls, model turns, output tokens, and wall time. Results remain
separate for all six cells, with failures and truncations retained in
denominators and no confirmatory workflow-superiority claims at n=5 per cell.
