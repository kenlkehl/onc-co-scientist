# Codex execution plan: NSCLC semantic masking × workflow grid

## Objective

Prepare and run a reproducible experiment on the synthetic NSCLC clinical
dataset with a 2 × 3 factorial design:

- semantic condition: `named` versus `masked`;
- coordination workflow: `persistent`, `sequential`, or `deliberative`;
- five replicates in each of the six cells;
- twenty ordered scientific iterations per run, where one iteration is exactly:
  `hypothesis_generation → analysis → critique → synthesis`.

This document is an implementation and execution plan. Do not start the main
model run until all preflight gates below pass.

## Locked interpretation

1. The requested “masked” condition is the repository's tracked `anonymized`
   twin. Its clinical feature names are replaced by opaque `feature_NNN`
   identifiers. The NSCLC framing and outcome name `pfs_months` remain visible.
2. To isolate clinical-name semantics from knowledge of variable roles, give
   both conditions the same structural information: four columns are anonymous
   candidate treatment/exposure indicators and the remaining features are
   possible modifiers/covariates. In the masked condition, identify the four
   candidates only as `feature_012`, `feature_018`, `feature_020`, and
   `feature_027`; never disclose their clinical names or any other mapping.
3. A healthy run completes exactly 20 iterations. “Up to 20” is the hard cap,
   not a gold-triggered stopping rule. Do not inspect the answer key to decide
   when to stop. A timeout or unrecoverable technical failure may truncate a
   run and must be reported as such. Earliest discovery is calculated after
   execution.
4. Every synthesis is a scored checkpoint and must return a non-null
   `final_answer`. The iteration-20 synthesis is the terminal result.
5. Use `gpt-5.6-luna` with low reasoning effort. Retain the prior timeout fix:
   3,600 seconds in the outer harness and 3,580 seconds in the Codex adapter.
6. The comparison is **native-resource**, not resource-matched. Deliberative
   runs intentionally use more calls. Record calls, tokens, tool calls,
   duration, timeouts, and efficiency-normalized outcomes.
7. The main experiment contains exactly six cells and five planned replicates
   per cell. Any one-iteration live smoke runs are separate and excluded from
   the analysis.

## Repository and pinned starting point

The required repository is
[kenlkehl/onc-co-scientist](https://github.com/kenlkehl/onc-co-scientist).
The NSCLC coordination experiment and timeout-alignment fix are on branch
`codex/depmap-10k-calibrated-metadata`, not currently on `main`.

- Pinned starting commit:
  `4a8fd25f104869d9209ec010bac504b8a91a4964`
- Commit page:
  <https://github.com/kenlkehl/onc-co-scientist/commit/4a8fd25f104869d9209ec010bac504b8a91a4964>
- Commit subject: `Add NSCLC coordination experiment and align timeouts`

The earlier laptop-only file
`experiments/nsclc_coordination/nsclc_five_repeats_20260831.yaml` and its
partial results were untracked local artifacts. Do not expect them to exist on
GitHub and do not use them as inputs.

### Clone and verify on the execution machine

```bash
gh repo view kenlkehl/onc-co-scientist --json nameWithOwner,url,defaultBranchRef
git clone https://github.com/kenlkehl/onc-co-scientist.git
cd onc-co-scientist
git fetch origin codex/depmap-10k-calibrated-metadata
git switch --create codex/nsclc-semantic-workflow-grid \
  4a8fd25f104869d9209ec010bac504b8a91a4964
git rev-parse HEAD
git status --short
```

Require the printed commit to equal the pinned SHA. If `gh` is unavailable,
the `git clone` URL is sufficient. Make implementation commits on the new
branch; do not move the pinned source branch.

### Install and record the runtime

The package requires Python 3.12 or newer; use Python 3.13 for consistency
with the prior NSCLC run.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
command -v codex
codex --version
```

Before any live call, write a machine manifest containing:

- git commit and dirty/clean status;
- operating system and architecture;
- Python and installed package versions;
- Codex executable path, version, and SHA-256;
- model ID and reasoning effort;
- experiment-config SHA-256;
- dataset and private-evaluator file hashes;
- UTC start time and deterministic schedule seed.

Do not copy the laptop's absolute Python or
`/Applications/ChatGPT.app/.../codex` paths. Discover the execution machine's
absolute `.venv` Python and Codex paths and generate a machine-local YAML from
a tracked template.

## Experimental grid and resource envelope

| Semantic condition | Persistent | Sequential | Deliberative |
|---|---:|---:|---:|
| Named clinical labels | 5 replicates | 5 replicates | 5 replicates |
| Masked clinical labels | 5 replicates | 5 replicates | 5 replicates |

There are 30 main runs. With 20 iterations and four stages per iteration:

| Workflow | Calls per stage | Calls per run | Runs | Maximum calls |
|---|---:|---:|---:|---:|
| Persistent | 1 | 80 | 10 | 800 |
| Sequential | 1 | 80 | 10 | 800 |
| Deliberative: 2 peers + 1 chair | 3 | 240 | 10 | 2,400 |
| **Total** |  |  | **30** | **4,000** |

The 60-minute ceiling makes 4,000 hours the purely theoretical serial timeout
ceiling; it is not an expected duration. Check model access, usage limits,
available disk, and host uptime before launch. Begin with `max_parallel: 2`
and increase only after the smoke run demonstrates stable service behavior.

## Source data and isolation requirements

Use these tracked sources from the pinned commit:

- Named public data:
  `example_data_clinical_all_claude/ds001/nsclc/named/public/`
- Named evaluator manifest:
  `example_data_clinical_all_claude/ds001/nsclc/named/manifest.json`
- Masked public data:
  `example_data_clinical_all_claude/ds001/nsclc/anonymized/public/`
- Masked evaluator manifest:
  `example_data_clinical_all_claude/ds001/nsclc/anonymized/manifest.json`
- Evaluator-only mapping:
  `example_data_clinical_all_claude/ds001/nsclc/anonymized/column_mapping.json`

Expected hashes:

| File | SHA-256 |
|---|---|
| Named `dataset.parquet` | `c93065845b99676904f8ec902b0c1c24fb0ab98579e084c706bcdf804d025fbb` |
| Masked `dataset.parquet` | `a84474a29fd5efda1fdf109163fdbff3b374f676434ccc24fc569ac42e67ba61` |
| Named `manifest.json` | `a844619fceb456a5ef4d9b5ba3dff5e7f07363eb83554226601f845ed22ce064` |
| Masked `manifest.json` | `e78bb273b53647fdc41af4d0e7b34ffbb0d0dec86b0e09b78e689be764ca0409` |
| `column_mapping.json` | `6d291a3d653803b12ad150654635201b079c6dd37e09fd7037fd0bcbf08d9cd7` |

Add a preparation command that verifies these hashes, confirms both Parquet
files are 50,000 × 35, and proves exact row/value/dtype parity after the
evaluator-side inverse rename.

Create experiment-specific public workspaces rather than altering the tracked
source bundles. Each public workspace should contain only:

- `dataset.parquet`;
- a matched `dataset_description.md`.

The two descriptions should use parallel wording and should differ only in
the feature labels. They should identify the four candidate exposures in each
condition, state that `pfs_months` is the outcome, and identify all remaining
features as possible modifiers/covariates. The masked description must contain
no drug, gene, sex, or clinical-feature mappings.

Fail closed on gold isolation:

- never stage `manifest.json` or `column_mapping.json` in an agent workspace;
- never put private paths, mappings, or target details in `TaskSpec.metadata`;
- never include them in prompts, resolved public specs, or model request logs;
- deny agents access to the source repository, parent directories, sibling
  workspaces, other runs, network, and the grant directory;
- give every fresh session its own clean copied workspace and scratch root;
- add an automated scan that fails if a public masked workspace contains a
  clinical target term or a private filename.

## Required implementation work

### 1. Add ordered 20-iteration execution

Modify the experiment schema and orchestrator; do not repurpose
`deliberation_rounds`, which controls peer revision inside a deliberative
stage.

- [ ] Add an explicit iteration policy to `ExperimentSpec`, with a
  backward-compatible default of one iteration and a validated maximum of 20.
  For this experiment, use `iterations: 20` and `completion_mode: fixed`.
- [ ] Wrap the existing ordered stage loop in an iteration loop.
- [ ] Add one-based `iteration_index`, `max_iterations`, and zero/one-based
  stage position consistently to prompts, requests, artifacts, event records,
  call records, and `run.json`.
- [ ] Require a non-null `final_answer` for every synthesis and `null` for all
  other stages. Mark the iteration-20 synthesis as terminal.
- [ ] Include iteration in nonpersistent session IDs, workspace IDs, call
  slots, and artifact keys so later iterations cannot resume an earlier
  session accidentally.
- [ ] Persist `iterations_completed`, terminal iteration, and stop reason.
- [ ] Reject a config whose per-run call ceiling is below the calls implied by
  its workflow, stages, peers, rounds, and iterations.

Workflow semantics must be exact:

- **Persistent:** one Codex session and one workspace across all 80 ordered
  stage calls. The full session history supplies context.
- **Sequential:** a fresh session/workspace for every iteration-stage pair.
  Each receives only the immediately preceding structured handoff. Iteration
  `k + 1` hypothesis generation receives iteration `k` synthesis.
- **Deliberative:** for every iteration-stage pair, start two fresh independent
  peers and a fresh chair. Both peers receive only the prior authoritative
  handoff. The chair receives that handoff plus both peer artifacts and emits
  the authoritative handoff for the next stage. Use `agents_per_stage: 2` and
  `deliberation_rounds: 1`.

Primary files likely affected:

- `src/onc_co_scientist/harness/experiment.py`
- `src/onc_co_scientist/harness/orchestrator.py`
- `src/onc_co_scientist/harness/runtime.py`
- `scripts/codex_cli_json_adapter.py`
- the corresponding harness and adapter tests under `tests/`

### 2. Implement call-level resumability before the long run

The current `--resume` behavior only skips a fully completed run. That is not
sufficient for an 80- or 240-call run.

- [ ] Write an atomic `run_state.json` after every successful call.
- [ ] Store the spec fingerprint, deterministic call-slot cursor,
  iteration/stage/peer-or-chair position, call index, usage ledger, artifacts,
  previous authoritative handoff, partial deliberative peer results, session
  records, and workspace state needed for continuation.
- [ ] On `--resume`, verify the fingerprint and substrate hashes, restore the
  ledger and session mapping, and continue from the first missing call slot.
- [ ] Preserve failed and partial call directories. Never overwrite
  `call_0001` or silently double-count usage on restart.
- [ ] Resume an interrupted deliberative stage at the missing peer or chair;
  do not rerun already completed peers.
- [ ] Refuse incompatible or ambiguous state with an actionable error.
- [ ] Archive an irrecoverable attempt before any fresh attempt at the same
  planned replicate, retaining provenance for both.

### 3. Make scoring condition- and iteration-aware

`experiments/nsclc_coordination/score_experiment.py` is currently hard-coded
to named clinical variables and aggregates only by workflow. Do not run the
six-cell experiment until this is corrected.

- [ ] Load the appropriate private manifest by `task_id` and derive the gold
  exposure, outcome, direction, and subgroup predicates in that condition's
  own namespace.
- [ ] Use `column_mapping.json` only on the evaluator side for optional
  cross-condition display. Never require a masked agent to infer clinical
  meanings: recovery in the masked condition is scored using opaque IDs.
- [ ] Preserve the existing evidence gate: supported recovery requires a
  directionally compatible quantitative effect, uncertainty or p-value, and
  subgroup/exposed/comparator sample sizes.
- [ ] Score every logical stage checkpoint by iteration. For deliberative
  runs, chairs are the stage checkpoints and peer artifacts are diagnostics.
- [ ] Distinguish terminal recovery from best/ever recovery; never select the
  “best analysis” across all iterations and present it as the terminal state.
- [ ] Record first supported exact/near recovery iteration and call, later
  loss, critique rescue, persistence to terminal synthesis, malformed output,
  unsupported convergence, timeout, and resource use.
- [ ] Aggregate by `semantic_condition × workflow`, producing six separate
  cells with five planned runs each. Do not pool named and masked tasks.
- [ ] Remove hard-coded “Named” report labels.

Minimum scored outputs:

- `run_scores.csv`: one record per run;
- `iteration_scores.csv`: one record per logical checkpoint and iteration;
- `cell_summary.csv`: six-cell descriptive summary;
- `resource_summary.csv`: calls, tokens, tools, time, and normalized recovery;
- `report.md`: compact methods, results, failures, and provenance.

Pre-register these endpoints in the experiment protocol before live calls:

- **Primary:** evidence-supported exact recovery at a synthesis checkpoint on
  or before iteration 20.
- **Key secondary:** evidence-supported exact recovery retained in the
  iteration-20 terminal synthesis.
- First supported exact recovery iteration, censored at 20 for nonrecovery.
- Near/component recovery, rescue, loss, unsupported convergence, malformed
  outputs, and timeouts.
- Recovery per model call, per 1,000 output tokens, and per wall-clock hour.

With five replicates per cell, report counts, rates with descriptive intervals,
trajectories, and paired descriptive contrasts. Do not present confirmatory
workflow-superiority p-values.

### 4. Create a portable experiment specification

Add a tracked template such as:

`experiments/nsclc_coordination/nsclc_semantic_workflow_grid.template.yaml`

Generate a machine-local ignored YAML from the template. The generated config
must resolve absolute paths for the execution machine's venv Python, adapter,
Codex binary, prepared public workspaces, private evaluator manifests, and
output root. Do not commit machine-specific paths.

The intended high-level shape is:

```yaml
schema_version: "1"
experiment_id: nsclc-semantic-workflow-grid-20x5
iteration_policy:
  iterations: 20
  completion_mode: fixed

tasks:
  - id: ds001-nsclc-named
    semantic_condition: named
    public_workspace: <prepared-named-public>
    private_evaluation_path: <named-manifest>
  - id: ds001-nsclc-masked
    semantic_condition: masked
    public_workspace: <prepared-masked-public>
    private_evaluation_path: <masked-manifest>

models:
  - id: codex-luna-low
    model_id: gpt-5.6-luna
    adapter: cli-json
    reasoning_effort: low
    command: [<venv-python>, <absolute-adapter-path>]
    extra_args:
      - --codex
      - <absolute-codex-path>
      - --reasoning-effort
      - low
      - --analysis-python
      - <venv-python>
      - --timeout-seconds
      - "3580"

workflows:
  - {id: persistent, mode: persistent}
  - {id: sequential, mode: sequential}
  - id: deliberative
    mode: deliberative
    agents_per_stage: 2
    deliberation_rounds: 1

stages:
  - hypothesis_generation
  - analysis
  - critique
  - synthesis

replicates: 5
max_parallel: 2
budget:
  max_agent_calls: 240
  max_runtime_seconds_per_call: 3600
```

Use the full existing stage objects and safeguards rather than the abbreviated
stage list shown above. The named and masked task prompts must be parallel and
must ask agents to search the four candidate exposures systematically for
one- and multi-feature treatment-effect heterogeneity in `pfs_months`, without
stating how many signals exist or which exposure/subgroup is involved.

Add an explicit `semantic_condition` field to the task/run records rather than
inferring it from filenames. It is acceptable for an agent to know that its
column labels are masked; it must not receive the mapping.

Add deterministic schedule support to the run planner rather than merely
writing a schedule beside an independently ordered matrix. The executor and
dry run must consume the same frozen `schedule.json`, preserve its order across
resume, and refuse a schedule whose run IDs or spec fingerprint do not match
the resolved experiment. Use replicate blocks and a recorded seeded shuffle of
the six cells within each block.

## Validation and launch gates

### Gate 1: unit and integration tests

- [ ] Full repository test suite passes.
- [ ] One-iteration configs remain backward compatible.
- [ ] Two- and 20-iteration stage ordering tests pass.
- [ ] Persistent uses one session; sequential and deliberative session IDs are
  unique by iteration and stage.
- [ ] Cross-iteration handoffs are exactly as specified.
- [ ] Call limits are 80, 80, and 240 at 20 iterations.
- [ ] Every synthesis has a final answer; other stages do not.
- [ ] Interruption/resume passes after every linear stage and after each
  deliberative peer/chair position, without duplicate calls or usage.
- [ ] A stale spec fingerprint or changed substrate is rejected on resume.
- [ ] Named/masked Parquet parity and gold-leakage tests pass.
- [ ] Masked exact, near, component, contradiction, and free-text fallback
  scoring tests pass.
- [ ] Aggregation yields six condition-workflow cells, not three workflows.

### Gate 2: no-model stub matrix

Run both semantic conditions, all three workflows, one replicate, and two
iterations with the stub runtime. Require:

- persistent: 8 calls per run;
- sequential: 8 calls per run;
- deliberative: 24 calls per run;
- six total runs and 80 total stub calls;
- correct session/workspace counts, handoffs, iteration metadata, scoring, and
  resume behavior.

### Gate 3: deliberate interruption test

Interrupt a stub or disposable run in the middle of a deliberative stage,
resume it, and compare its canonical artifacts and usage accounting with an
uninterrupted reference. Also test persistent-session continuation after a
process restart. Do not proceed if completed call slots are repeated.

### Gate 4: excluded live smoke

Run one iteration, one replicate, all six cells with Luna. This is 40 maximum
live calls: 8 across the two persistent runs, 8 across the two sequential
runs, and 24 across the two deliberative runs.

Confirm:

- Codex authentication and `gpt-5.6-luna` access;
- the 3,580/3,600-second timeout relationship;
- structured artifact validity and real session resumption;
- analysis Python and package access inside every sandbox;
- zero private-path or mapping exposure;
- expected call graph and complete usage accounting;
- successful named and masked deterministic scoring.

Store smoke outputs under a distinct root containing `smoke`; the scorer must
exclude that root from the main analysis.

### Gate 5: freeze and launch the 30 main runs

1. Freeze the protocol, generated YAML, machine manifest, source/data hashes,
   scorer version, and a deterministic run schedule before the first call.
2. Construct five replicate blocks. Within each block, deterministically
   shuffle the six condition-workflow cells using a recorded seed. Save the
   complete schedule as `schedule.json` before execution.
3. Run under `tmux`, `systemd`, or another host service manager so disconnecting
   the client does not stop the job.
4. Start with no more than two concurrent runs. Use `--resume` after any host
   restart; do not create duplicate replicate IDs.
5. Monitor completion, failure, timeout, disk, and usage ledgers. Do not inspect
   scientific results to alter scheduling, prompts, iteration count, or retry
   policy.

After implementation, the command sequence should be equivalent to:

```bash
.venv/bin/ocs harness validate-experiment \
  --config experiments/nsclc_coordination/nsclc_semantic_workflow_grid.local.yaml

.venv/bin/ocs harness run-experiment \
  --config experiments/nsclc_coordination/nsclc_semantic_workflow_grid.local.yaml \
  --dry-run

.venv/bin/ocs harness run-experiment \
  --config experiments/nsclc_coordination/nsclc_semantic_workflow_grid.local.yaml \
  --resume \
  --max-parallel 2

.venv/bin/python experiments/nsclc_coordination/score_experiment.py \
  <main-results-root> \
  --out <main-results-root>/scored
```

The dry run must report two tasks × three workflows × one model × five
replicates = 30 runs and a 4,000-call maximum.

## Completion criteria

Do not declare the experiment complete until all of the following hold:

- [ ] The frozen config and provenance files identify the pinned repository,
  exact model profile, machine runtime, and all substrate hashes.
- [ ] All 30 planned main runs have terminal status, with five planned runs in
  each of the six cells; technical failures remain visible.
- [ ] Every healthy run has 20 ordered synthesis checkpoints.
- [ ] Healthy persistent and sequential runs have 80 calls each; healthy
  deliberative runs have 240 calls each.
- [ ] If all 30 runs complete normally, there are exactly 600 synthesis
  checkpoints, 2,400 logical stage checkpoints, and 4,000 model artifacts.
- [ ] No gold file, mapping, private path, sibling workspace, or prior-run
  result was exposed to an experiment agent.
- [ ] Scoring separates all six cells and distinguishes ever-recovered from
  terminally retained findings.
- [ ] The report includes failures and resource use, labels the comparison
  native-resource, and makes only descriptive claims appropriate for n=5 per
  cell.
- [ ] Tests, config, protocol, implementation, and scorer changes are committed
  before results are interpreted.

## Expected work products

Commit the iteration-aware harness, call-level resume implementation,
condition-aware scorer, preparation utility, tracked config template, updated
protocol, and tests. Keep machine-local config and large result artifacts out
of git. Preserve the frozen machine manifest, schedule, resolved public spec,
raw immutable run artifacts, and scored tables/report together in durable
experiment storage.
