# Deterministic hypothesis recovery

Primary planted-finding recovery no longer uses an LLM. Research agents emit
`proposed_hypotheses[].finding` alongside their prose. LLM novelty scoring remains
available as a separate optional endpoint. Archived prose transcripts remain
readable; they receive no deterministic recovery credit without an explicit
structured claim. Use `--legacy-llm-matching` only to reproduce archived scoring.

## Definition

The default is **structured-recovery-v2**. Primary recovery measures hypothesis
identity and completeness. Statistical confirmation is a separate secondary
endpoint. `--recovery-version structured-recovery-v1` reproduces the previous
deterministic definition, including its original contrast and evidence gates.
Existing Aim 1 plans without a scorer version automatically select v1 when
regenerated with `experiments/aim1_recovery/score.py`. Older runs and reports are
preserved; a v2 rescore belongs in a new directory.

A finding specifies `outcome`, nullable `exposure`, `contrast`, signed `direction`
(-1, 0, 1), and a conjunction of `subgroup` predicates. Predicates contain
`column`, `operator`, and `value`. The schema admits equality, inequality,
numeric bounds, and membership lists. It rejects malformed, contradictory,
nonfinite, or outcome-defined subgroups. It does not infer unspecified cutoffs
from prose, complete missing modifiers, or repair a candidate using the answer key.

Supported contrasts are:

| Contrast | Confirmatory estimand |
|---|---|
| `subgroup_difference` | Outcome mean inside minus outside the submitted subgroup |
| `treatment_effect` | Exposed minus unexposed outcome mean inside the subgroup |
| `treatment_interaction` | Inside treatment difference minus outside treatment difference |

Treatment contrasts require a binary 0/1 exposure. In v2 the clinical planted
subgroup effects accept either a treatment effect within the complete subgroup
or a treatment interaction. V1 requires the interaction. DepMap requires a subgroup
mean difference. Censored survival estimands, nonlinear continuous exposures,
arbitrary Boolean expressions, multi-treatment contrasts, and equivalence tests
for null findings need additional explicitly defined adapters. An ordinary
nonsignificant test is not evidence of null recovery.

The scorer canonicalizes masked identifiers to named identifiers, and normalizes
predicate order, duplicates, and logically redundant bounds. Complete recovery
requires the correct outcome, exposure, contrast, direction, and all defining
subgroup predicates, without extra restrictive predicates. Numeric bounds may
be approximate if subgroup precision and recall both meet the fixed thresholds
(default 0.90). Missing categorical gates fail complete recovery even when a
finite evaluation sample happens to have perfect subgroup overlap. Functional
precision, recall, and F1 remain separately reported. Strict recovery requires
normalized rule equivalence, with numeric absolute tolerance 1e-9.

Numerical evidence is independently recomputed from the **submitted** claim:
Welch tests for two groups and a four-group Welch-Satterthwaite contrast for
interactions, at least 10 observations per cell. Missing subgroup covariates
are excluded from both inside and outside groups. Nonbinary treatment values
are rejected. The reported `significant` flag never confers recovery credit.
For secondary confirmation, a linked discovery-data analysis with code, a finite
effect estimate, and a valid p-value must be present in the run record. Neither
this requirement nor held-out significance gates v2 hypothesis identity. Thus
a proposed complete hypothesis can receive identity credit before it is tested;
it is not thereby a confirmed discovery. Reports retain the number of submitted
claims and show identity, strict identity, and confirmation together.

Across distinct submitted claims, alpha is allocated as `0.05 / (j * (j + 1))`
for claim index j. The sum is bounded by 0.05. Repeated identical claims reuse
the original allocation. This avoids retrospectively changing discovery times
when later candidates are submitted. The evaluator provides no feedback during
the research run. This protects held-out confirmation from adaptive feedback;
it does not make arbitrary exploratory p-values valid.

`primary_iteration` is the first complete hypothesis submission in v2.
`confirmed_iteration` is the earliest submission with linked analysis and
held-out support. `interaction_confirmed_recovered` counts only submitted
treatment interactions with confirmation; it does not relabel a treatment
effect as evidence of heterogeneity. No LLM scores any of these endpoints.

## Single or batch scoring

```bash
ocs score run --dataset path/to/named_bundle \
  --transcript path/to/transcript.json --out path/to/score \
  --evaluation-data path/to/independent_evaluation.parquet

ocs score batch --synth-root path/to/bundles --tasks-root path/to/tasks \
  --out path/to/scores --evaluation-root path/to/evaluation_tree
```

The evaluation tree mirrors bundle paths and contains `evaluation.parquet` in
each bundle directory. Without independent evaluation data, the output explicitly
labels the calculation `in_sample_reconfirmation`; it must not be described as
held-out validation. Scoring cannot prove independence of an arbitrary
user-supplied file. The prepared Aim 1 protocol makes the split before agents run.

Add `--score-novelty --judge codex-cli` (or another supported judge) for optional
novelty. Primary recovery remains deterministic. Outputs are
`structured_scores.json` and `structured_scores.md`, containing primary and
strict recovery, first discovery iterations, structural errors, functional
metrics, and evidence records. Legacy output names apply only to legacy mode.

## Research backends

Install the analysis environment before task preparation:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[analysis,dev]'
```

The interpreter path must retain the virtualenv path. Do not dereference its
symlink to the underlying base Python executable.

Prepare the 200 clinical and 50 DepMap runs from the archived preliminary cohorts:

```bash
.venv/bin/python experiments/aim1_recovery/prepare.py \
  --out data/aim1_new --python .venv/bin/python
```

The same original rows are split 80% for discovery and 20% for evaluation, with
identical splits in both naming conditions. V2 clinical tasks require 25 research
iterations and 20 repeats per condition per cohort; DepMap requires 10 iterations
and 25 repeats per condition. Workspaces contain copies of discovery data and
neutral instructions/examples, never the answer key or evaluation data.

### ChatGPT Work / Luna

```bash
.venv/bin/python experiments/aim1_recovery/run_batch.py \
  --plan data/aim1_new/plan.json --backend work \
  --model gpt-5.6-luna --reasoning-effort medium --service-tier standard \
  --work-advertised-tier standard
```

This exports one JSON task prompt per pending replicate. The **Work orchestrator**
launches each with a fresh `fork_turns=none` subagent and the requested model.
The Python process cannot call Work's collaboration tools. Never process multiple
scientific replicates in the same agent context. Fast mode is exposed by Work's
advertised priority service tier; actual response-level tier/token telemetry is
not available through this subagent interface. The `--work-advertised-tier` value
must describe the actual tool capability; it does not select or change a tier.
The exporter refuses a tier mismatch. If the tool advertises only priority,
Standard requests must wait for a Standard-capable session. Do not relabel a
priority run by editing metadata. Model, reasoning, backend, and tier must match
the frozen v2 plan. A changed endpoint URL may still serve the same frozen model.

During research, each iteration is submitted before starting the next:

```bash
/path/to/.venv/bin/python -m onc_co_scientist.harness.structured_runner submit \
  --workspace . --record iteration_record.json
/path/to/.venv/bin/python -m onc_co_scientist.harness.structured_runner finalize \
  --workspace .
```

Submissions receive sequential indexes, UTC receipts, and hashes; finalization
checks record integrity and hypothesis references. Receipts establish consistency
of retained artifacts, not an adversarial security boundary against an agent
with unrestricted filesystem access.

V2 finalization also requires the full iteration count and all four research
actions: screening, multivariable exploration, refinement, and robustness.
Each record carries `research_step` with `action`, `rationale`, `script_path`,
and `output_path`. Each iteration must link an analysis, retain its executed
script and output, and preserve their receipt hashes. Exact script reuse and
empty iterations are rejected. These checks cannot certify scientific
originality, prove execution, or establish equal token/compute budgets. Review
a separate setup pilot for protocol adherence and stopping behavior before
launching the formal batch, without selecting or tuning on recovery outcomes.

New prepared workspaces also set `require_sequential_outputs=true`. They reject
reused output paths and output files predating the preceding submission receipt
(10 ms filesystem timestamp tolerance). This catches accidental backfilling at
submission time. Earlier workspaces remain readable under their recorded
metadata. The first priority setup pilot exposed one such sequencing error;
its original records are retained, and a separate setup repeat tests the guard.

### User-provided endpoint, including vLLM

Use a **new** prepared experiment directory for a different model/backend:

```bash
.venv/bin/python experiments/aim1_recovery/prepare.py \
  --out data/aim1_local --python .venv/bin/python \
  --backend endpoint --model YOUR_SERVED_MODEL \
  --reasoning-effort unspecified --service-tier unspecified
.venv/bin/python experiments/aim1_recovery/run_batch.py \
  --plan data/aim1_local/plan.json --backend endpoint \
  --base-url http://YOUR_HOST:8000/v1 --model YOUR_SERVED_MODEL --jobs 4
```

The endpoint must support OpenAI-compatible Chat Completions with native tool
calls and usage accounting. Configure the model-appropriate tool parser/chat
template on vLLM; those are model-specific server choices. An optional credential
is read from `OPENAI_API_KEY`, or the variable named by `--api-key-env`.
The example explicitly omits reasoning effort and service tier, so local models
do not receive unsupported OpenAI-specific options. For an endpoint that supports
Standard processing, prepare with `--service-tier standard`; the runner sends
`service_tier: "default"` and requires matching returned tier metadata. Missing
or conflicting tier telemetry stops that run and remains in the request log.
The requested model and the endpoint-returned model are recorded. No model
fallback occurs. The runner has configurable per-call tokens, total generated
tokens, model turns, tool calls, request deadlines, and Python execution timeouts.
HTTP authentication and malformed-request errors are not blindly retried.

The same `execute_python` and `submit_iteration` tools are used throughout an
endpoint session. Code runs in the task workspace; credentials are excluded
from its inherited environment, subprocess groups are killed on timeout, and
code/tool outputs are retained. Both backends rely on explicit task boundaries
within a shared filesystem unless deployed inside an external container or
sandbox. Do not claim OS isolation from the workspace path alone.

The Work and endpoint runners share the output/scoring contract, but have
different agent harnesses and telemetry. A model comparison that changes the
backend is not automatically a model-only comparison.

## Reports and figures

```bash
.venv/bin/python experiments/aim1_recovery/score.py \
  --plan data/aim1_new/plan.json \
  --out experiments/aim1_recovery/results/my_run
```

The report refuses to produce a final figure with unfinished runs. It verifies
input checksums and receipts, exports submitted traces, and produces CSV/JSON
scores plus PNG/PDF/SVG recovery plots and discovery curves. `--allow-incomplete`
is for diagnostics and explicitly marks figures incomplete. Do not use it to
quietly replace the prespecified denominator.

The portable research archive retains evaluator inputs and all analysis records,
with one discovery-data copy per cohort/condition. Restore it into an empty
directory to rescore on another machine; original absolute paths are preserved
in `plan_original_paths.json` while the active plan points to the restored files:

```bash
.venv/bin/python experiments/aim1_recovery/archive.py restore \
  --archive aim1_structured_checkpoint.tar.gz --out data/restored_aim1
.venv/bin/python experiments/aim1_recovery/score.py \
  --plan data/restored_aim1/experiment/plan.json --out experiments/aim1_recovery/results/restored
```

A checkpoint can contain unfinished jobs; the same completeness gate still applies.

The rerun is a new development pilot: model, output contract, held-out confirmation,
and neutral examples differ from the archived pilot. The 0.90 threshold was a
development choice made after inspecting archived examples and frozen before
fresh runs, not an independently preregistered threshold. Keep primary, strict,
and 0.95 sensitivity results together when interpreting the naming-condition gap.

The [post hoc v2 rescore](../experiments/aim1_recovery/results/luna_20260904_v2_rescore/README.md)
reuses the completed September 4 priority runs and isolates the scoring change.
It is separate from the new Standard-service, fixed-budget experiment.
