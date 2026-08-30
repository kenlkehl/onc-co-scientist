# NSCLC coordination-architecture pilot

## Purpose

This experiment tests whether the repository's persistent, sequential, and
deliberative co-scientist workflows can independently analyze the same clean
synthetic NSCLC cohort and recover a deliberately buried treatment-effect
heterogeneity signal. It is a functionality and exploratory native-resource
pilot, not a resource-matched causal comparison of coordination policies.

## Locked substrate

- Dataset: named `ds001_nsclc`, 50,000 rows.
- Public files: `dataset.parquet` and `dataset_description.md` only.
- Public Parquet SHA-256:
  `c93065845b99676904f8ec902b0c1c24fb0ab98579e084c706bcdf804d025fbb`.
- Private manifest SHA-256:
  `a844619fceb456a5ef4d9b5ba3dff5e7f07363eb83554226601f845ed22ce064`.
- Model profile: `gpt-5.6-luna`, low reasoning, Codex CLI
  `0.151.0-alpha.7.1`.
- Codex executable SHA-256:
  `a5976fd714dc0801a6f40e0e2b3051f64f1e0468f6c0f279b7d2a7afb7623f43`.

The private manifest is used only by the deterministic scorer and is not
included in agent requests, copied workspaces, or the resolved public
experiment specification.

## Workflow arms

Each run has four scientific stages: hypothesis generation, analysis,
critique, and synthesis.

1. **Persistent:** one resumable Codex session performs all four stages.
2. **Sequential:** each stage uses a fresh Codex session and receives only the
   preceding structured handoff.
3. **Deliberative:** two independent peers work at each stage; a fresh chair
   synthesizes their structured artifacts. One peer round is used.

With the implemented native architecture, these arms consume 4, 4, and 12
model calls per run, respectively. Realized calls, tokens, tool calls, and
duration must therefore be reported with scientific outcomes. Recovery per
1,000 output tokens is an efficiency endpoint. A common resource ceiling is
not described as resource matching.

For copied workspaces, a persistent session keeps one workspace. Every fresh
sequential session and every deliberative peer or chair receives a separate
clean snapshot and separate writable scratch root. A custom Codex permission
profile grants model tools read/write access only to that session's public
workspace and scratch root, read-only access to the pinned Python executable
and libraries, and minimal operating-system runtime access. Model subprocesses
inherit no host environment variables and receive only a fixed executable path,
Python package path, and session-local temporary directory. Network access,
login shells, global
temporary directories, the source repository, sibling workspaces, the grant
draft, and the private manifest are denied by the operating-system sandbox.

## Execution sequence

1. Run all repository tests and a no-model stub matrix.
2. Run one live replicate of each workflow (20 total model calls).
3. Confirm the 4/4/12 call graph, true session resumption, data access,
   structured-output validity, usage accounting, and deterministic scoring.
4. Only after the smoke test passes, run eight replicates per workflow (24
   cells; 160 total model calls) in replicate-interleaved order.

The eight-replicate run is an exploratory preliminary-data pilot. Its sample
size is operational, not power-derived; report counts and run-level intervals,
not confirmatory p-values for arm comparisons.

The adapter timeout is set to 1,180 seconds, just below the harness's locked
1,200-second per-call ceiling, so timeout failures retain their audit artifacts
instead of being terminated first by the outer controller.

## Locked target and scoring

The private target is a beneficial association between sotorasib and
progression-free survival in the conjunction of:

- `kras_g12c = 1`
- `alk_fusion = 0`
- `brca2_mutation = 0`
- `sex_female = 0`

The planted effect is +5.0 months. In the locked seed-0 data, the subgroup has
3,266 patients (1,154 exposed and 2,112 comparators) and the unadjusted observed
mean difference is approximately +4.985 months. These values are never placed
in an agent prompt.

Semantic recovery levels are mutually exclusive:

- **Exact:** correct exposure, outcome, beneficial direction, and all four
  noncontradictory predicates.
- **Near:** correct exposure, outcome, and direction with exactly three of the
  four predicates and no contradiction.
- **Component:** correct exposure and outcome with one or two correct
  predicates and no contradiction.
- **None/contradictory:** anything else.

Evidence-supported exact or near recovery additionally requires a linked
quantitative claim with a positive effect estimate, a p-value or confidence
interval, and a reported subgroup sample size. Free-text assertion alone does
not receive supported-recovery credit.

## Endpoints

Primary functionality endpoint:

- Evidence-supported exact recovery in the final synthesis artifact.

Exploratory Aim 2 endpoints:

- Semantic and evidence-supported recovery at each stage.
- Survival of an analysis-stage supported near/exact finding into synthesis.
- Critique rescue: unsupported/none before critique becoming supported
  near/exact after critique or by synthesis.
- Loss: supported near/exact at analysis but not at synthesis.
- First stage/call of supported near/exact recovery, with nonrecovery treated
  as censored.
- Final recovery per 1,000 output tokens and per model call.
- Unsupported convergence and malformed-output rates.

For deliberative runs, stage-level checkpoints are the chair artifacts named
`<stage>_consensus`; peer-level results are retained as secondary diagnostics.

## Interpretation constraints

The named dataset makes feature interpretation available to agents and tests
clinical-analysis behavior, not blinded feature discovery. The synthetic PFS
outcome is an observed continuous endpoint rather than a censored survival
record. Results may show that the machinery runs and generate preliminary
hypotheses about coordination, but cannot establish superiority of one
architecture without a prospectively enforced resource-matched design and a
larger task panel.
