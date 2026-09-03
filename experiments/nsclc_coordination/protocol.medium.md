# NSCLC semantic masking × workflow grid: Luna-medium resilient local-env v6 protocol

## Status and frozen objective

This is a new prospective v6 experiment. It does not resume or pool the stopped
Luna-low run or any Luna-medium predecessor. The v5 live smoke was stopped by
operator request while incomplete after its low two-way concurrency made the
technical gate impractically slow. Its root remains preserved and immutable and
is excluded from v6 analysis. The only intended change from v5 is concurrency:
the excluded live smoke uses six workers and the main run uses twelve.

The replacement tests the same 2 × 3 factorial grid on the synthetic
50,000-row NSCLC cohort. The semantic factor is `named` versus `masked`; the
coordination factor is `persistent`, `sequential`, or `deliberative`. Five
replicates are planned in each of the six cells. Every healthy run completes
exactly 20 ordered iterations of:

`hypothesis_generation → analysis → critique → synthesis`.

The pinned source baseline is commit
`4a8fd25f104869d9209ec010bac504b8a91a4964`. The execution-plan documentation
commit and the v6 implementation commit are captured in the frozen machine
manifest. The model is `gpt-5.6-luna`, with reasoning effort explicitly locked
to `medium` and Codex `service_tier` explicitly locked to `fast`. The outer
per-call timeout is 28,800 seconds and the adapter has one shared 28,780-second
deadline across original, retry, and repair turns.

Each request records reasoning effort and whether the stage must produce a
final answer. The adapter rejects a reasoning mismatch before launch. It gives
Codex a closed, stage-specific JSON schema: synthesis requires a non-empty
structured final-answer object; every other stage requires JSON null. The
adapter validates the schema and cross-field semantic contract after generation
and records the exact schema hash and effective launch settings.

## Preregistered contract-repair policy

Each harness call permits at most two bounded same-session schema-repair turns
after the original generation. A repair is
allowed only when Codex exited successfully and its saved final artifact fails
JSON parsing, the supplied JSON schema, model validation, or the controller's
cross-field semantic checks. In particular, each
`supported_claim_indices` entry must be unique, in range, and point to a claim
whose `supported` field is true.

The adapter saves the thread ID immediately after every subprocess turn,
including failed turns. Cross-reference-only failures are repaired with a
narrow schema that permits only `supported_claim_indices` and dynamically
enumerates indices of claims whose `supported` value is true; the controller
merges that patch without allowing any other scientific field to change. Other
eligible contract failures request a complete corrected artifact in the same
thread. All repairs receive the exact controller error and direct Luna to reuse
completed analysis without rerunning tools unless correction is impossible.

Classified transient transport exits have a separate budget of at most 24
retries, exponential backoff from 30 to 900 seconds, and a six-hour retry
window within the eight-hour call deadline. Two consecutive transport failures
open a shared experiment-level circuit for at least 300 seconds. Active workers
remain occupied while backing off, and newly scheduled workers consult the same
circuit before launching, preventing the scheduler from rapidly consuming the
queue during a provider outage. A retry resumes an emitted thread ID and
resubmits the exact pending task with an interruption notice. Nontransport
runtime errors, missing output, and thread-identity errors remain terminal.

Every response, event stream, stderr log, schema, prompt hash, command,
validation outcome, and usage record is retained in a numbered physical-attempt
directory. Logical schema-repair and physical transport-retry counts are
reported separately, while call-level tokens, tool calls, and duration aggregate
all attempts.

Before any v6 live call, the controller, adapter, and model-visible analysis
Python use `/home/klkehl/thisenv`, a local ext4 virtual environment containing a
non-editable installation of the committed package and scientific dependencies.
The Codex CLI package is copied to a versioned local-ext4 path and its version
and hash are frozen. No v6 prompt, schedule, iteration count, retry cap,
concurrency, or scientific endpoint may be changed after inspecting v6
scientific results.

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
calls, 2,400 logical stage checkpoints, and 600 synthesis checkpoints. Schema
repairs and transport retries are exceptional physical turns and are reported
separately. This is a native-resource comparison, not a resource-matched
comparison. Calls, original, repair, and transport-retry turns, tokens, tool
calls, duration, timeout status, and efficiency endpoints are reported.

The excluded live smoke uses six concurrent runs. After every smoke run and all
40 planned harness calls pass the technical gate, the main run uses twelve
concurrent runs as explicitly authorized for v6.

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

The one-iteration v6 live smoke is stored under a distinct
`medium-fast-resilient-localenv-v6-smoke` root and excluded from the main analysis.
Main launch is automatic only after the smoke summary, normalized artifacts,
per-attempt audits, Luna/medium/Fast/local-environment locks, isolation contract,
schema hashes, and cross-field synthesis invariants all pass.
