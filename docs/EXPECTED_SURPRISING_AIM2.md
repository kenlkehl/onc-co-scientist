# Expected/surprising discovery across scientific workflows

Aim 2 can now compare persistent, sequential, and deliberative workflows on the
same paired discovery tasks used for Aim 1. This is an optional protocol in the
existing `ocs harness run-experiment` matrix runner. Existing named/masked and
generic artifact experiments retain their original controller and contracts.

## What changes between workflows

Every run has one scientific ledger. The shared controller registers comparisons,
runs requested analyses, releases validation evidence, records assessments, and
evaluates the final accepted claims. Agents use the current short stage forms;
they do not maintain controller IDs, repeat the accepted list, or follow a
prescribed minimum effect size. See the [scientific task guide](EXPECTED_SURPRISING_GUIDE.md).

| Workflow | Agent context | Who changes the ledger? |
|---|---|---|
| Persistent | One continuing conversation with recent committed turns, the complete current ledger, and the latest notebook | The continuing agent |
| Sequential | A fresh conversation at each stage, receiving the current ledger, research notes, and latest stage narratives | That stage's agent |
| Deliberative | Fresh peers at each stage independently receive the same ledger and handoff; later rounds can see the other peers' previous-round drafts | A fresh chair selects one complete stage form |

The four stages remain **explore → analyze → appraise → synthesize**. All modes
use 25 iterations for clinical and cell-line tasks. A six-iteration smoke override
retains automatic evidence releases at iterations 2 and 4, with reassessment two
iterations later. This response deadline is not a two-iteration run cap.

Peer drafts cannot register claims, execute analyses, release validation, or
change accepted findings. Their proposals may refer only to the same committed
ledger available to the chair. The chair can choose, revise, or combine proposals
and must explain substantive disagreement in its narrative; it is instructed to
use evidence rather than vote counting. Drafts remain in the audit even when the
chair discards them. Additional peer rounds retain each peer's own conversation
and disclose other peers' drafts only after the preceding round is complete.

Persistent here means persistent **conversation state**, carried by the controller
through the provider interface. It does not require a permanently running model
process. The Codex CLI provider still uses isolated, tool-disabled invocations;
the persistent condition supplies its retained committed conversation explicitly.
All data analysis is performed by the shared scientific controller. This differs
from the generic Pi/CLI runtime's filesystem-tool experiments.

## Scientific and resource controls

The dataset bytes, analysis limit, validation policy, independent confirmation,
effect-size reference rules, and scores are common across workflows. Private
effect cutoffs, planted identities, expected/surprising assignment, and validation
selection state are excluded from agent messages. Acceptance does not require
prior independent validation. Agents judge effect sizes and conclusions.

Each repeat has a seed identity shared across paired versions, workflows, and
model profiles. Comparison-specific validation and confirmation samples are
therefore matched whenever workflows request the same comparison. Different
investigation choices can still create different eligible validation pools.

This is a comparison of complete workflows with the same scientific opportunities.
It is **not an equal-token or equal-call comparison**:

| Healthy run | Persistent | Sequential | Deliberative: two peers, one round, one chair |
|---|---:|---:|---:|
| Six iterations | 24 calls | 24 calls | 72 calls |
| 25 iterations | 100 calls | 100 calls | 300 calls |

The configured total call ceiling includes failed calls and format repairs. Each
authoritative stage allows two repair attempts by default; each peer draft has
the same repair allowance. Chair repairs reuse the existing peer drafts. Failed
authoritative actions roll back atomically. A peer that exhausts its repairs
causes that stage to fail rather than silently shrinking the panel. A run with
an unrecovered stage error receives zero primary recovery and primary F1*.
First-attempt scores also flag repaired peer drafts.

Participant calls, prompts, replies, reported tokens, missing usage, provider
errors, and elapsed time are retained. SDK-internal retry counts are unavailable
unless the provider reports them. The runtime timeout setting bounds the configured
provider request timeout; SDK retries or Codex quota waits may extend wall time.
The persistent condition defaults to a **120,000-character history budget**,
excluding the current stage prompt. Before a call, it removes the oldest complete
conversation turns until the retained history fits. The complete current ledger
and latest notebook are always supplied. Every trim is recorded and disclosed in
the agent's context; the full original conversation remains in the evaluator audit.
This is a text budget, not a cap on scientific iterations. Set
`persistent_history_chars: null` to retain unlimited history instead. That option
can exceed model context windows because each stage contains a ledger snapshot.
A remaining context-limit error is recorded as a technical failure.

Aggregate token, tool, and dollar caps are rejected for this protocol because
the providers do not offer consistent enforcement. The generic artifact runner
keeps its existing resource controls. The paired protocol supports the independent
`federation` site-count/partition grid described in [Federated workflows](EXPECTED_SURPRISING_FEDERATION.md).
The legacy `workflow.federated` switch (which only performs final central synthesis)
and additional safeguard arms remain separate generic-runner features.

## Configure and run

Use the existing checkout and environment described in the
[refactor setup instructions](EXPECTED_SURPRISING_REFACTOR_PLAN.md).
The prepared task root is `data/expected_surprising_ledger_25_iterations`.
It contains all 20 current packages without changing frozen numerical dataset
bytes. These local data files are not shipped in Git. Restore them or repackage
the verified frozen release before running in another checkout.

The [smoke configuration](../configs/expected_surprising.aim2_smoke.yaml) runs both
NSCLC data types and both paired versions through all three workflows: 12 runs.
The [full matrix example](../configs/expected_surprising.aim2_full.yaml) includes
all ten base datasets, five model profiles, and ten repeats: **3,000 runs** and
**500,000 healthy participant calls**. It is a runnable example, not a campaign
launched by this integration. Narrow `profiles`, `pair_ids`, `models`, or
`replicates` to design a smaller study before execution.

```bash
export PYTHONNOUSERSITE=1
export PATH="/tmp/ocs-es-refactor-venv/bin:$PATH"

ocs harness validate-experiment --config configs/expected_surprising.aim2_smoke.yaml
ocs harness run-experiment --config configs/expected_surprising.aim2_smoke.yaml --dry-run
ocs harness run-experiment --config configs/expected_surprising.aim2_smoke.yaml

# Resume the same frozen configuration and implementation after interruption.
ocs harness run-experiment --config configs/expected_surprising.aim2_smoke.yaml --resume

# Rebuild the readable report without making model calls.
ocs expected-surprising summarize-workflows \
  --config configs/expected_surprising.aim2_smoke.yaml
```

Paths in YAML resolve relative to the configuration file. `expected_surprising`
imports both versions through private `assignment.json` files and verifies dataset
checksums; task IDs are never guessed. Omitting `iteration_policy` from this
protocol selects 25 iterations. Explicit budgets must allow the complete
validation response window. `max_parallel` controls concurrent matrix cells;
peers within a cell are called in order but cannot see one another's first drafts.
The existing `--max-parallel` command-line override can lower concurrency during
resume without changing the frozen scientific configuration. For example, use
`--resume --max-parallel 2` when a shared server is under memory pressure. Record
that operational change when interpreting resource use or elapsed time.

Use `adapter: provider` with any supported provider configuration: `vllm_openai`,
`codex_cli`, `gemini_vertex`, or `anthropic_vertex`. Server URLs and model names
belong only in configuration. The included Codex profiles use Luna, Sol, and Astra
separately, each at medium reasoning. No account reset is redeemed by the runner.

## Results and interrupted runs

Each matrix writes its ordinary `plan.json`, frozen `schedule.json`,
`resolved_spec.json`, `summary.json`, and per-run `run.json`. Expected/surprising
runs additionally retain:

- `provenance.json`: code, dependency versions, public files, and private evaluator hashes.
- `calls/*.json`: each participant's session, stage, prompt, response, usage, and checksum.
- `requests/*.json`: requests saved before provider calls, including work still in flight.
- `science-NNNN/transcript.jsonl`: the scientific controller's evidence and assessment audit.
- `report.json`: final R, P, F1*, E, B components and denominators, plus coordination details.
- `expected_surprising_summary.json` and `expected_surprising_report.md` at the matrix root.

Use the matrix runner's `--resume` option directly. The adapter-process supervisor
is designed for native runtime processes, not this provider conversation protocol.
Resume verifies code, dependencies, configuration, task bytes, and every reused
prompt. It reconstructs the scientific state by replaying saved responses with
the same seeds; it makes new provider calls only after the durable journal ends.
An interrupted in-flight request without a saved response may need to be reissued.
Reported tokens cannot include provider work that never returned a reply, so
interrupted requests and unreported SDK retries can add resource use beyond the
saved call totals.
Interrupted transcripts are retained in separate numbered directories. Completed
scientific runs, including scored technical failures, are reused after checksum
verification; resume does not resample an unfavorable result. Provider-initialization
failures without scientific progress can be retried. Running without `--resume`
archives existing cells rather than overwriting them.

Freeze the code used for a live run. Changes to prompts, policies, analysis, or
provider code invalidate automatic replay. The compatibility allowance for older
generic artifact experiments does not bypass these paired-run checks.

## How to read the comparison

The primary outcome is **independently confirmed focal recovery**. For each model
and workflow, report absolute expected and surprising recovery and their paired
difference, surprising minus expected. The Aim 2 workflow effect subtracts the
persistent workflow's paired difference from each other workflow's difference.
A positive change means recovery shifts toward surprising findings relative to
expected ones; it could arise by losing expected findings, so both absolute
recovery rates are essential.

Repeated runs are averaged within each paired version, and base datasets receive
equal weight. Workflow contrasts are computed within the same base datasets.
Uncertainty resamples whole base datasets, keeping versions and workflows
together and preserving clinical/cell-line strata. One base dataset per data type
is a smoke test, not enough for a confidence interval across datasets. Models and
workflow conditions are reported separately; the summary rejects accidental
pooling. Assigned failures remain in the primary denominator. Missing scientific
traces make the condition's secondary scores unavailable rather than selecting
only successful runs.

The report spells out **R — recovery/recall**, **P — confirmed-claim fraction
(precision/positive predictive value)**, **F1* — discovery performance**,
**E — exploration coverage**, and **B — evidence responsiveness**. F1* is the
harmonic mean of category-balanced R and P; the asterisk marks its difference
from ordinary F1. P considers all final accepted claims, including additional
claims beyond the planted set. No accepted claims makes P unavailable. B is
unavailable when one of its supported/excluded/ambiguous evidence classes is
absent. Machine-readable Q and D keys remain for compatibility with Aim 1.

The comparison table averages each run's scores separately. Its F1* column need
not equal the harmonic mean of the displayed average R and P; stage failures
also set the affected run's primary F1* to zero.

These outcomes describe the **workflow's shared scientific decisions**, not an
average of peer scores. A proposal contributes to recovery only if it enters the
shared record and meets the usual testing, acceptance, and confirmation rules.
In sequential mode, a later agent may revise an earlier agent's conclusion;
responsiveness then measures the workflow's adaptation, not one individual's
change of mind. Individual drafts and chair decisions remain available for
separate studies of agreement or influence.
