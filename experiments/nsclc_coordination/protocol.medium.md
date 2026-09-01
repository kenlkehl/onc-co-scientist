# NSCLC semantic masking × workflow grid: Luna-medium replacement protocol

## Status and frozen objective

This is a new prospective experiment replacing—but never resuming or pooling
with—the stopped Luna-low run. The predecessor was stopped because technical
timeouts and malformed synthesis artifacts caused excessive attrition. Its
partial artifacts and score remain under the original result root.

The replacement tests the same 2 × 3 factorial grid on the synthetic
50,000-row NSCLC cohort. The semantic factor is `named` versus `masked`; the
coordination factor is `persistent`, `sequential`, or `deliberative`. Five
replicates are planned in each of the six cells. Every healthy run completes
exactly 20 ordered iterations of:

`hypothesis_generation → analysis → critique → synthesis`.

The repository implementation baseline is commit
`4a8fd25f104869d9209ec010bac504b8a91a4964`. The execution-plan documentation
commit and the final replacement implementation commit are recorded separately
in the machine manifest. The model is `gpt-5.6-luna` at explicitly locked
medium reasoning effort. The outer per-call timeout is 14,400 seconds and the
Codex adapter timeout is 14,380 seconds.

Each harness request records its reasoning effort and whether the stage must
produce a final answer. The adapter rejects a reasoning mismatch before model
execution. It supplies a closed, stage-specific JSON schema to Codex: synthesis
requires a non-empty structured final-answer object; every other stage requires
JSON null. The adapter validates that contract again after generation and saves
the schema hash and effective model settings in the per-call audit.

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

- Persistent uses one Codex session and one copied workspace for all 80 calls.
- Sequential uses one fresh session/workspace for every iteration-stage pair.
  Each receives only the immediately preceding authoritative handoff; the next
  iteration's hypothesis stage receives the previous synthesis.
- Deliberative uses two fresh peers and one fresh chair for every
  iteration-stage pair. Peers receive only the prior authoritative handoff;
  the chair receives that handoff plus both peer artifacts. Chairs are the
  scientific checkpoints.

Persistent and sequential runs have 80 planned calls; deliberative runs have
240. Across 30 runs the maximum healthy call graph contains 4,000 model calls,
2,400 logical stage checkpoints, and 600 synthesis checkpoints. This is a
native-resource comparison, not a resource-matched comparison. Calls, input
and output tokens, tool calls, duration, timeout status, and efficiency
endpoints are reported.

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
`run_state.json` checkpoints are written after every successful call and bind
the state to the spec fingerprint and public-substrate hashes.

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
malformed output, technical failure, timeout, recovery per model call, recovery
per 1,000 output tokens, and recovery per wall-clock hour.

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

Results are summarized separately for all six `semantic_condition × workflow`
cells with counts, rates, descriptive Wilson intervals, trajectories, resource
use, and replicate-paired descriptive contrasts. With five replicates per cell,
no confirmatory workflow-superiority p-values are reported. Technical failures
and truncated runs remain in denominators and reports.

One-iteration live smoke runs are stored under a distinct medium-smoke root and
are excluded from the main analysis. Scientific results are not inspected to
change prompts, schedules, iteration count, retry policy, or concurrency.
