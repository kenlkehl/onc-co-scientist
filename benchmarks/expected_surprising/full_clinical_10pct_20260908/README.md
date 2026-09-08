# Six-model clinical rerun: 10% significance guidance

120 fresh runs, 20 per model (10 expected and 10 surprising), 25 iterations per run. Models: Luna, Terra, Sol, Astra, Qwen 3.8 27B, Gemma 4 31B. Five sessions per model run concurrently. All explicitly request medium reasoning and standard service tier (`default`). vLLM accepted both request fields; their actual effect depends on the model/template, and the servers return no effective tier confirmation.

The shared prompt now says: “For this task, a relative difference in outcome of 10% or greater is considered clinically significant.” It gives no log-scale calculation or mechanical accept/reject rule. The private evaluator remains unchanged at 0.10 natural-log PFS units (approximately 11%), so the guidance and private cutoff are close but not identical.

Dataset bytes, repeat identifiers, validation policy, and full iteration budget match the original clinical comparison. These are fresh sessions and outputs, not resumed runs. The complete source and inputs are frozen with hashes in the batch manifest. Twenty focused provider, prompting, Codex, and runner tests passed before launch, and both live vLLM settings probes succeeded.

The earlier Terra and vLLM batches were stopped by request. Their transcripts, completed reports, call audits, and STOPPED.json records remain in their original directories. They are not included in this rerun's results.

[Progress](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct/STATUS.md) · [Settings and hashes](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct/manifest.json) · [Launch record](launch.json)

The runner writes the new [full report](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct/RESULTS.md) after all runs finish. A 15-minute thread heartbeat will verify it, copy it here, report the results, and pause. It remains quiet while progress is healthy.

Superseded shortly after launch when the user expanded the grid to three workflows. All partial outputs are preserved. The active experiment is [the 360-run workflow grid](../full_clinical_10pct_workflows_20260908/README.md).
