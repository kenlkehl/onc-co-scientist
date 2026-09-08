# Six models × three workflows: clinical 10% guidance

360 fresh runs: Luna, Terra, Sol, Astra, Qwen 3.8 27B, and Gemma 4 31B; persistent, sequential, and deliberative workflows; expected and surprising versions; ten repeats per version. All request medium reasoning and standard/default service tier. 25 iterations per run, at most 30 concurrent runs overall. The data bytes match the earlier paired clinical experiment.

- Persistent: shared ledger and notebook plus committed conversation, bounded to 120,000 characters of history. History trimming is disclosed and audited; the current ledger remains available.
- Sequential: a fresh conversation each stage with the shared ledger and notebook.
- Deliberative: two independent peers draft each stage's action, then a chair selects the only action applied to the scientific ledger. One deliberation round. Peer drafts never execute analyses.

The same scientific analysis/validation budgets and private scores apply to every workflow. Persistent and sequential use 100 participant calls per healthy run; deliberative uses 300. The full grid plans 60,000 calls before repairs. Each run allows up to 900 calls including repairs, at most two technical retries per stage/peer, 125,000 output tokens per call, and 1,800 seconds per call.

The common prompt states that a relative outcome difference of 10% or greater is clinically significant, with no log-scale calculation or mechanical decision rule. The unchanged private evaluator cutoff is approximately 11%, close to but not identical to the public guidance. vLLM accepts medium/default request fields but may not implement reasoning levels identically to Codex.

Every participant call stores provider-reported output tokens. Totals include peer/chair responses and retries after invalid responses; cached replay counts once. Missing usage and additional transport attempts without usage are explicitly flagged; known output totals are lower bounds in those cases. Reasoning tokens are not added on top of a provider total that already includes them. The report includes total and mean output tokens per model/workflow condition, alongside R, P, F1*, E, B, and focal recovery.

33 focused checks passed. The frozen plan was validated to contain 36 model/workflow/version cells with ten repeats each. Source, configuration, and input hashes are frozen and checked before execution. Earlier partial batches remain archived and are excluded; the previous runner's ledger-only continuity is not relabeled as this coordinator's persistent conversation condition.

[Progress](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/STATUS.md) · [Plan](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/plan.json) · [Config](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/config.yaml) · [Source/input hashes](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/frozen_manifest.json) · [Launch](launch.json)

The runner will generate the separate [full report](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/RESULTS.md). A 15-minute heartbeat will verify completion and token accounting, report results, and then pause. It stays quiet during healthy progress.
