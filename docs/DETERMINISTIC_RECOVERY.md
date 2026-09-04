# Deterministic hypothesis recovery

Primary planted-finding recovery no longer uses an LLM. Research agents emit
`proposed_hypotheses[].finding` alongside their prose. LLM novelty scoring remains
available as a separate optional endpoint. Archived prose transcripts remain
readable; they receive no deterministic recovery credit without an explicit
structured claim. Use `--legacy-llm-matching` only to reproduce archived scoring.

## Definition

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

Treatment contrasts require a binary 0/1 exposure. The pilot's clinical planted
subgroup effects require a treatment interaction; DepMap requires a subgroup
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
A linked discovery-data analysis with code, a finite effect estimate, and a
valid p-value must be present in the run record.

Across distinct submitted claims, alpha is allocated as `0.05 / (j * (j + 1))`
for claim index j. The sum is bounded by 0.05. Repeated identical claims reuse
the original allocation. This avoids retrospectively changing discovery times
when later candidates are submitted. The evaluator provides no feedback during
the research run. This protects held-out confirmation from adaptive feedback;
it does not make arbitrary exploratory p-values valid.

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
identical splits in both naming conditions. Clinical tasks retain 25-iteration
caps and 20 repeats per condition per cohort; DepMap retains a 10-iteration cap
and 25 repeats per condition. Workspaces contain copies of discovery data and
neutral instructions/examples, never the answer key or evaluation data.

### ChatGPT Work / Luna

```bash
.venv/bin/python experiments/aim1_recovery/run_batch.py \
  --plan data/aim1_new/plan.json --backend work \
  --model gpt-5.6-luna --reasoning-effort medium --service-tier priority
```

This exports one JSON task prompt per pending replicate. The **Work orchestrator**
launches each with a fresh `fork_turns=none` subagent and the requested model.
The Python process cannot call Work's collaboration tools. Never process multiple
scientific replicates in the same agent context. Fast mode is exposed by Work's
advertised priority service tier; actual response-level tier/token telemetry is
not available through this subagent interface.

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

### User-provided endpoint, including vLLM

Use a **new** prepared experiment directory for a different model/backend:

```bash
.venv/bin/python experiments/aim1_recovery/run_batch.py \
  --plan data/aim1_local/plan.json --backend endpoint \
  --base-url http://YOUR_HOST:8000/v1 --model YOUR_SERVED_MODEL --jobs 4
```

The endpoint must support OpenAI-compatible Chat Completions with native tool
calls and usage accounting. Configure the model-appropriate tool parser/chat
template on vLLM; those are model-specific server choices. An optional credential
is read from `OPENAI_API_KEY`, or the variable named by `--api-key-env`.
Reasoning effort and service tier are omitted unless explicitly supplied, so
local models do not receive unsupported OpenAI-specific options by default.
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
