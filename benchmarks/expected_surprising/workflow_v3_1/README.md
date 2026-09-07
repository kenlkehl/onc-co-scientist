# Expected/surprising evaluation with agent judgment

This revision removes prescribed minimum effect sizes from everything the evaluated agent receives. Instructions ask the agent to judge effect sizes, uncertainty, and conclusions and explain its reasoning. Public outcome metadata includes names and units. Numerical evidence includes estimates, intervals, sample sizes, and diagnostics; evaluator cutoffs are omitted for discovery analyses, voluntary and scheduled validation, cached requests, and reoriented comparisons.

The evaluator retains its previous numerical cutoffs and scoring rules privately. Discovery confirmation and evidence responsiveness therefore describe agreement with that fixed reference; disagreements do not by themselves establish unreasonable scientific judgment. The [guide](../../../docs/EXPECTED_SURPRISING_GUIDE.md) explains the methods, metric names, setup, execution, and reporting commands.

| Contract | Version |
|---|---|
| Workflow | `appraisal-3.1.0` |
| Prompt | `exploration-3.1.0` |
| Response schema | `stage-3.0.0` |
| Validation policy | `delayed-3.0.0` |
| Scoring | `profile-3.0.0` |
| Numerical data-generating process | `2`, unchanged |

The workflow remains provider independent. The four-run live smoke uses `Inferact/Qwen3.8-27B-NVFP4` at `http://sn4622130540:8000/v1`, six iterations per run, two workers, 125,000 completion tokens per call including reasoning, two retries per stage, and a 1,800-second timeout. Scheduled validation arrives in iterations 2 and 4, with a two-iteration response window and shared paired replicate ID `smoke-replicate-0`.

All 20 new packages are under `data/expected_surprising_agent_judgment/`. The [package manifest](package_manifest.json) preserves source identities, dataset hashes, package hashes, and policies. The [package audit](package_audit.json) confirms every dataset matches the original frozen v2 release and checks the public packages contain no effect-size cutoffs. The previous prescribed-effect packages and [v3.0 smoke report](../workflow_v3/smoke_vllm_20260907.md) remain separate.

The live batch `data/expected_surprising_agent_judgment/smoke/20260907T190300Z` completed all **96/96 stages** in 71.3 minutes. Three failed attempts recovered on the first retry: one registration-contract error and two calls that exhausted the 125,000-token allowance. No stage exhausted its retries. All four run audits and all 20 package audits passed, including inspection of every recorded provider prompt for effect-cutoff exposure. See the [smoke report with full metric definitions](smoke_vllm_20260907.md), [complete audit](smoke_audit.json), and [frozen run provenance](smoke_vllm_20260907.json).

Across these four runs, F1\* is **32.22/100**, exploration coverage **20.14/100**, and evidence responsiveness **36.11/100** against the private reference. Expected focal recovery is **0%**, surprising focal recovery **50%**. The batch includes two eligible ambiguous-reference events, so aggregate evidence responsiveness is available even though three individual runs lack a complete set of evidence classes. These four development runs cannot establish an effect of the instruction change.

The clinical expected run illustrates the distinction between an agent’s conclusion and private benchmark confirmation: its three accepted BRCA2 comparisons have final intervals entirely above zero, but those intervals cross the evaluator’s 0.10 reference minimum. They are recorded as unconfirmed under that reference, rather than as evidence of zero or opposite-direction effects. The report names the evaluator reference explicitly when describing agreement and disagreement.

Local verification: **62 tests passed** across the historical and workflow test modules; Ruff passed. The tests include all seven behavior fixtures, full 25-iteration clinical and 10-iteration cell-line runs through the generic provider interface, prompt inspection across all stages, and preservation of numerical evidence while filtering private cutoffs. Scientific assessments that disagree with the private reference remain valid protocol responses.

The smoke runner writes a plain-language report using recall (R), precision (P), F1\*, exploration coverage (E), and evidence responsiveness (B). It explains the F1 analogy and its limits, and distinguishes the six-round smoke budget from the two-round reassessment deadline. Settings and detailed counts appear in an expandable section. The saved numerical records retain their original Q and D keys; the report maps them to P and F1\* without changing scores. The original report generator is preserved with the run, and the updated report records its own generator and Markdown checksums. The reproducible final audit and report commands are:

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python \
  scripts/expected_surprising/audit_workflow.py \
  --data data/expected_surprising_agent_judgment \
  --smoke data/expected_surprising_agent_judgment/smoke/20260907T190300Z \
  --out benchmarks/expected_surprising/workflow_v3_1/smoke_audit.json
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python \
  scripts/expected_surprising/report_smoke.py \
  --smoke data/expected_surprising_agent_judgment/smoke/20260907T190300Z \
  --audit benchmarks/expected_surprising/workflow_v3_1/smoke_audit.json \
  --out benchmarks/expected_surprising/workflow_v3_1/smoke_vllm_20260907.md --archive
```

An initial setup attempt (`20260907T190200Z`) failed during sandbox DNS resolution before any model calls and is excluded from evaluated runs.
