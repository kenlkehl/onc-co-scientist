# Aim 2 paired-workflow verification

The pre-integration checkpoint is `7bcdd63`. The implementation connects the
existing experiment matrix to the expected/surprising scientific controller.
See the [workflow guide](../../../docs/EXPECTED_SURPRISING_AIM2.md).

**Automated checks:** 174 distinct tests passed, with no failures, errors, or skips
(173 in the main suite, plus one new runtime-guard regression within a 39-test
follow-up run).
This covers all three modes, both paired versions, full 25-iteration clinical
and cell-line runs, bounded conversation history, interrupted-run replay,
chair-only scientific authority, independent first-round drafts, repairs,
failure denominators, workflow contrast signs, report regeneration, and the
existing native runtimes and semantic-grid workflows. Ruff and `git diff --check`
passed for the integration files. [Machine-readable verification](verification.json) records versions
and the test modules.

The final 16-test workflow regression suite also passed. It now checks that
regenerating a report with a different plan order produces the same summary.
Live report regeneration preserved the scientific values and produced stable
JSON and Markdown bytes; no further model calls were made.

All 20 packaged datasets match the original frozen Parquet files byte for byte.
The [dataset hash audit](dataset_hash_audit.json) records both paths and checksums.

Healthy 25-iteration runs use 100 scientific stages and 100, 100, or 300
participant calls for persistent, sequential, or two-peer deliberative workflows.
Scripted scientists making the same decisions produced identical scientific
state, validation, confirmation, and scores in all three modes. The smoke auditor
also passed on the scripted bounded-memory matrix.

The full example configuration was validated and **dry-run only**: 20 tasks ×
3 workflows × 5 model profiles × 10 repeats = 3,000 runs, with 500,000 healthy
participant calls. No full Aim 2 campaign was launched.

## Live verification

Both probes used frozen source in
`data/expected_surprising_aim2/release_20260907_bounded`, both NSCLC data types,
both paired versions, and all three workflows. Each assigned run had six
iterations, releases at iterations 2 and 4, a 125,000-token call ceiling, and two
repair attempts. Persistent history was bounded at 120,000 characters while
retaining the complete current ledger and latest notebook.

| Deployment | Finished / planned | Finished runs with stage failures | Incomplete |
|---|---:|---:|---:|
| Gemma 4 31B at `camus:8002` | 12 / 12 | 2 | 0 |
| Qwen 3.8 27B at `sn4622130540:8000` | 5 / 12 | 1 | 7 |

**Gemma:** [readable report](gemma_smoke_report.md),
[paired summary](gemma_summary.json), and [artifact audit](gemma_live_audit.json).
All 12 runs reached the end of their six-iteration budgets. Ten had no
unrecovered stage error; two surprising clinical runs retained three stage
errors involving inconsistent or duplicate claim assessments. Their primary
recovery and F1* are zero, with diagnostic outcomes preserved. The audit passed
its integrity and coverage checks across 506 participant calls. Its
`all_completed` flag is false because two terminal results have failed status.

**Qwen:** [partial report](qwen_smoke_report.md) and
[partial artifact audit](qwen_partial_audit.json). The probe is paused with all
completed results and partial checkpoints preserved. Its 309 saved calls include
36 replies that used all 125,000 allowed output tokens and returned no final
text, plus eight replies containing two identical JSON objects. Three stages
failed: one in a finished sequential run and two in unfinished deliberative
runs. No paired effect is estimated from selectively completed cells. The
partial audit verified saved-call integrity, completed report hashes, and
32 matched first-round peer pairs.

Qwen's shared server reached about 97% KV-cache use with ten queued requests and
frequent preemptions. The probe was resumed with the existing `--max-parallel 2`
override after starting with six workers, then paused when the complete Gemma
verification finished. See the [resume record](operational_resume.json),
[pause record](qwen_pause.json), and [reply diagnostics](live_diagnostics.json).
The scientific configuration and frozen source were unchanged. Reported tokens
exclude provider work without a returned reply, including interrupted requests
and unreported SDK retries. Elapsed times are diagnostic because the endpoints
were shared with other work. Other tasks' full campaigns were not modified.

The smoke contains one paired base dataset per data type and one run per version
and workflow. It verifies integration and records model reliability; it cannot
establish a general workflow ranking or a confidence interval across datasets.
The reports spell out R, P, F1*, E, and B, explain unavailable evidence classes,
and distinguish the two-iteration reassessment deadline from the run budget.

Two preliminary probes are also preserved. One exposed a null provider `metrics`
field, which was fixed and regression tested. The other motivated the tested,
auditable persistent-history limit. Their paths are in
[verification.json](verification.json).

To repeat the completed Gemma integrity audit:

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python \
  scripts/expected_surprising/audit_aim2.py \
  --root data/expected_surprising_aim2/smoke_20260907_gemma \
  --out benchmarks/expected_surprising/aim2_workflows/gemma_live_audit.json
```
