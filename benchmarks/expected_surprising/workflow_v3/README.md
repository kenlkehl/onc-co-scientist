# Appraisal-first expected/surprising development workflow

This is the archived `appraisal-3.0.0` evaluation, which exposed minimum effect sizes to the agent. The [current v3.1 evaluation](../workflow_v3_1/README.md) removes those prescriptions and preserves this earlier result separately.

This release implements [the refactor plan](../../../docs/EXPECTED_SURPRISING_REFACTOR_PLAN.md) on the existing v2 numerical datasets. The [updated guide](../../../docs/EXPECTED_SURPRISING_GUIDE.md) documents setup, contracts, provider configuration, package export, execution, and aggregation. Expert adjudication and formal evaluation locking remain separate development steps.

| Contract | Version |
|---|---|
| Workflow | `appraisal-3.0.0` |
| Prompt | `exploration-3.0.0` |
| Schema | `stage-3.0.0` |
| Validation policy | `delayed-3.0.0` |
| Scoring | `profile-3.0.0` |
| Numerical DGP | `2`, unchanged |

The controller is provider independent. The live smoke uses the configured vLLM endpoint; ordinary evaluations select any implemented provider through the existing registry. Historical task packages still dispatch to the earlier workflow, and original generation, numerical data, reports, and `StageRecord` readers are retained.

All 20 new packages live under `data/expected_surprising_workflow_refactor/{public,private}`. Their [package manifest](package_manifest.json) maps source IDs/hashes to new IDs/hashes and records private policy settings. The [package audit](package_audit.json) independently compares every package to the frozen v2 release manifest. Raw Parquet files and run artifacts remain local and excluded from Git.

Full policies release at clinical iterations 5/12/20 and cell-line iterations 3/6/8; the six-iteration smoke releases at 2/4. Selection freezes before appraisal one iteration earlier, rotates supported/excluded/ambiguous exploratory strata with seeded fallback, and uses no private targets or unseen validation outcomes. Each evidence response is due at the second subsequent iteration. The policy-derived maximum is 13 distinct validation comparisons in full runs, or 12 in smoke, shared across voluntary/automatic delivery and reversed directions.

The development environment was created with `/usr/bin/python3.12 -m venv /tmp/ocs-es-refactor-venv` and an editable install of `.[dev,vllm-openai]`. The environment is temporary. Numerical datasets were copied, not regenerated; the live manifest records sampling/runtime dependencies.

Final combined checks: **61 tests passed** (40 historical tests and 21 new workflow cases), and the requested Ruff check passed. The full-budget provider-independent cases exercise the packaged policy over 25 clinical and 10 cell-line iterations. The command included both test modules:

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python -m pytest -q \
  tests/test_expected_surprising.py tests/test_expected_surprising_workflow.py
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python -m ruff check \
  src/onc_co_scientist/expected_surprising scripts/expected_surprising \
  tests/test_expected_surprising.py tests/test_expected_surprising_workflow.py
```

The cases cover all seven requested behavior patterns (including caution resolved at the deadline), public/private separation, initial expectation exposure, frozen selections, shared samples across routes/directions/retries, canonical aliases and first acceptance, validation bounds, D arithmetic and untested/additional claims, early/fixed-budget E, unavailable B and missing decisions, interrupted/late/invalid events, and hierarchical aggregation. Existing exact/near assignment and numerical invariants remain covered by the historical tests.

The reproducible audit command is:

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python \
  scripts/expected_surprising/audit_workflow.py \
  --data data/expected_surprising_workflow_refactor \
  --smoke data/expected_surprising_workflow_refactor/smoke/20260907T171358Z \
  --out benchmarks/expected_surprising/workflow_v3/smoke_audit.json
```

The four six-iteration live runs have been executed and audited. See the [smoke report](smoke_vllm_20260907.md), [mechanical audit](smoke_audit.json), and [frozen run provenance](smoke_vllm_20260907.json) for results, errors, missing score components, and links to all raw artifacts.

An earlier partial batch, `20260907T170155Z`, was stopped during development to correct comparison-orientation diagnostics. Its partial transcripts are retained separately and excluded from the required smoke results.
