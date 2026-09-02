# NSCLC semantic masking × workflow grid: Luna-medium local-env v4 protocol

## Status and frozen objective

This is a new prospective v4 experiment. It does not resume or pool the stopped
Luna-low run or any Luna-medium predecessor. The v2 smoke completed four of six
runs and failed both deliberative runs after otherwise successful generations
produced semantically inconsistent `supported_claim_indices`. The v3 smoke
completed three of six runs: two Python adapter startups failed on transient
SSHFS `EPERM` reads from the SSHFS-hosted virtual environment, and one Codex
turn exhausted four hours after its completed analysis command was followed by
WebSocket and HTTPS transport failures. No v1, v2, or v3 main experiment
launched. All predecessor result roots remain immutable.

The replacement tests the same 2 × 3 factorial grid on the synthetic
50,000-row NSCLC cohort. The semantic factor is `named` versus `masked`; the
coordination factor is `persistent`, `sequential`, or `deliberative`. Five
replicates are planned in each of the six cells. Every healthy run completes
exactly 20 ordered iterations of:

`hypothesis_generation → analysis → critique → synthesis`.

The pinned source baseline is commit
`4a8fd25f104869d9209ec010bac504b8a91a4964`. The execution-plan documentation
commit and the v4 implementation commit are captured in the frozen machine
manifest. The model is `gpt-5.6-luna`, with reasoning effort explicitly locked
to `medium` and Codex `service_tier` explicitly locked to `fast`. The outer
per-call timeout is 14,400 seconds and the adapter has one shared 14,380-second
deadline across the original turn and any repair turns.

Each request records reasoning effort and whether the stage must produce a
final answer. The adapter rejects a reasoning mismatch before launch. It gives
Codex a closed, stage-specific JSON schema: synthesis requires a non-empty
structured final-answer object; every other stage requires JSON null. The
adapter validates the schema and cross-field semantic contract after generation
and records the exact schema hash and effective launch settings.

## Preregistered contract-repair policy

Each harness call permits at most two bounded same-session repair turns after
the original generation, for at most three Codex turns total. A repair is
allowed only when Codex exited successfully and its saved final artifact fails
JSON parsing, the supplied JSON schema, model validation, or the controller's
cross-field semantic checks. In particular, each
`supported_claim_indices` entry must be unique, in range, and point to a claim
whose `supported` field is true.

The adapter saves the thread ID before artifact validation. On an eligible
failure it resumes that exact thread, sends the exact controller error, asks for
a complete replacement artifact, and directs Luna to reuse completed analysis
without rerunning tools unless correction is otherwise impossible. Timeouts,
nonzero exits, missing runtime output, and thread-identity errors are terminal
and are never retried by this policy. The original and every repair response,
event stream, stderr log, schema, prompt hash, command, validation outcome, and
usage record are retained in numbered attempt directories. Call-level tokens,
tool calls, and duration aggregate all attempts.

This repair policy and Fast service tier remain fixed from v3. Before any v4
live call, the controller, adapter, and model-visible analysis Python are moved
to `/home/klkehl/thisenv`, a local ext4 virtual environment containing a
non-editable installation of the committed package and scientific dependencies.
No v4 prompt, schedule, iteration count, retry cap, concurrency, or scientific
endpoint may be changed after inspecting v4 scientific results.

## Conditions and scientific task

The named task exposes the four candidate indicators
`treatment_pembrolizumab`, `treatment_sotorasib`, `treatment_olaparib`, and
`treatment_osimertinib`. The masked task exposes the corresponding candidates
only as `feature_012`, `feature_018`, `feature_020`, and `feature_027`. Both
conditions identify `pfs_months` as the outcome and all remaining features as
possible modifiers or covariates. Agents search systematically for one- and
multi-feature treatment-effect heterogeneity without being told the number of
signals, target exposure, or target subgroup.

The masked agent is never given a clinical mapping. Recovery is evaluated in
the masked manifest's opaque namespace; evaluator-only mappings may be used for
clearly labeled cross-condition display after scoring.

## Workflow semantics and resources

- Persistent uses one Codex session and one copied workspace for all 80
  harness calls.
- Sequential uses one fresh session and workspace for every iteration-stage
  pair. Each receives only the immediately preceding authoritative handoff; the
  next iteration's hypothesis stage receives the previous synthesis.
- Deliberative uses two fresh peers and one fresh chair for every
  iteration-stage pair. Peers receive only the prior authoritative handoff; the
  chair receives that handoff plus both peer artifacts. Chairs are the
  scientific checkpoints.

Persistent and sequential runs have 80 planned harness calls; deliberative
runs have 240. Across 30 runs the healthy call graph contains 4,000 harness
calls, 2,400 logical stage checkpoints, and 600 synthesis checkpoints. With the
repair cap, the theoretical maximum is 12,000 Codex turns, although repairs are
expected to be exceptional. This is a native-resource comparison, not a
resource-matched comparison. Calls, original and repair turns, tokens, tool
calls, duration, timeout status, and efficiency endpoints are reported.

The excluded live smoke uses two concurrent runs. After every smoke run and all
40 planned harness calls pass the technical gate, the main run uses six
concurrent runs as explicitly authorized for v4.

## Isolation and reproducibility

Tracked source hashes, Parquet shape, and exact row/value/dtype parity after
evaluator-side inverse rename must pass before execution. Each agent workspace
contains only `dataset.parquet` and its matched `dataset_description.md`.
Private manifests, the column mapping, repository parents, sibling
workspaces/runs, network access, and grant material are denied to agents. A
fresh nonpersistent session receives a clean copied workspace and scratch root.
No private path or task target is included in model requests, prompts, or the
resolved public specification.

The run planner freezes a seeded replicate-block schedule before any model
call. Within each replicate block, the six condition-workflow cells are
shuffled with seed `20260831`. Dry run, execution, and resume consume the same
`schedule.json` and reject a changed fingerprint or run set. Atomic
`run_state.json` checkpoints are written after every successful harness call
and bind state to the specification fingerprint and public-substrate hashes.

## Preregistered endpoints

Primary endpoint:

- Evidence-supported exact recovery at a synthesis checkpoint on or before
  iteration 20.

Key secondary endpoint:

- Evidence-supported exact recovery retained in the iteration-20 terminal
  synthesis.

Other prespecified endpoints are first supported exact recovery iteration
(right-censored at 20), first recovery call, near/component recovery, critique
rescue, later loss, persistence to terminal synthesis, unsupported convergence,
malformed output, technical failure, timeout, contract-repair incidence and
success, recovery per harness call, recovery per Codex turn, recovery per 1,000
output tokens, and recovery per wall-clock hour.

Exact recovery requires the condition-native exposure, outcome, planted
direction, and all subgroup predicates without contradiction. Near recovery
omits exactly one predicate; component recovery includes one or more correct
predicates. Supported recovery additionally requires a directionally compatible
quantitative effect, a p-value or uncertainty interval, and subgroup, exposed,
and comparator sample sizes linked to evidence.

## Analysis and interpretation

Every logical checkpoint is scored by iteration. Deliberative peer artifacts
are retained as diagnostics but are not substituted for chair checkpoints.
Terminal results always use the final synthesis; a historically best analysis
is never presented as terminal.

Results are summarized separately for all six
`semantic_condition × workflow` cells with counts, rates, descriptive Wilson
intervals, trajectories, resource use, repair use, and replicate-paired
descriptive contrasts. With five replicates per cell, no confirmatory
workflow-superiority p-values are reported. Technical failures and truncated
runs remain in denominators and reports.

The one-iteration v4 live smoke is stored under a distinct
`medium-fast-repair-localenv-v4-smoke` root and excluded from the main analysis.
Main launch is automatic only after the smoke summary, normalized artifacts,
per-attempt audits, Luna/medium/Fast/local-environment locks, isolation contract,
schema hashes, and cross-field synthesis invariants all pass.
