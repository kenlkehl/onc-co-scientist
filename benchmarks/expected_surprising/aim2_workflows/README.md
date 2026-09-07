# Aim 2 paired-workflow verification

The pre-integration checkpoint is `7bcdd63`. The implementation connects the
existing experiment matrix to the expected/surprising scientific controller.
See the [workflow guide](../../../docs/EXPECTED_SURPRISING_AIM2.md).

**Automated checks:** 173 tests passed, with no failures, errors, or skips.
This covers all three modes, both paired versions, full 25-iteration clinical
and cell-line runs, bounded conversation history, interrupted-run replay,
chair-only scientific authority, independent first-round drafts, repairs,
failure denominators, workflow contrast signs, report regeneration, and the
existing native runtimes and semantic-grid workflows. Ruff and `git diff --check`
also passed. [Machine-readable verification](verification.json) records versions
and the test modules.

Healthy 25-iteration runs use 100 scientific stages and 100, 100, or 300
participant calls for persistent, sequential, or two-peer deliberative workflows.
Scripted scientists making the same decisions produced identical scientific
state, validation, confirmation, and scores in all three modes. The smoke auditor
also passed on the scripted bounded-memory matrix.

The full example configuration was validated and **dry-run only**: 20 tasks ×
3 workflows × 5 model profiles × 10 repeats = 3,000 runs, with 500,000 healthy
participant calls. No full Aim 2 campaign was launched.

## Live verification

The live smoke is in progress at
`data/expected_surprising_aim2/smoke_20260907_bounded`, using the frozen source and
configuration in `data/expected_surprising_aim2/release_20260907_bounded`.
It runs Qwen `Inferact/Qwen3.8-27B-NVFP4` on `sn4622130540:8000`, six iterations,
both NSCLC data types, both paired versions, and all three workflows: 12 runs.
The server reported a 262,144-token context window. Every call retains the
125,000-token ceiling and the two-repair allowance. Scheduled releases are at
iterations 2 and 4, with two later iterations for reassessment. Persistent
conversation history is bounded at 120,000 characters; the complete current
ledger and latest notebook remain supplied.

Two preliminary attempts are preserved separately. The first exposed a null
`metrics` field in the provider response; usage parsing was fixed and regression
tested. The second showed how repeating whole ledger snapshots would exhaust
persistent context; the auditable bounded-history policy was added and tested.
Only these development smoke processes were stopped. The other task's full
clinical campaigns were not modified.

Once complete, the live batch is audited with:

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python \
  scripts/expected_surprising/audit_aim2.py \
  --root data/expected_surprising_aim2/smoke_20260907_bounded \
  --out benchmarks/expected_surprising/aim2_workflows/live_audit.json
```

The live result is not yet claimed as passing. Recovery misses and unavailable
evidence-response categories will remain visible in the final report.
