# Full clinical comparison with simplified prompts

100 fresh runs: 10 repeats × expected/surprising clinical pair × 5 models. Each run has 25 rounds (100 research stages). The package selects the common `ledger-1.0.0` prompt for every backend.

| Model | Reasoning / tier | Assigned runs | Concurrent runs |
|---|---|---:|---:|
| Qwen 3.8 27B, sn4622130540:8000 | Server settings | 20 | 10 |
| Gemma 4 31B, camus:8002 | Server settings | 20 | 10 |
| Luna (`gpt-5.6-luna`) | Medium / Priority | 20 | 5 |
| Sol (`gpt-5.6-sol`) | Medium / Priority | 20 | 5 |
| Astra (`gpt-6-astra`) | Medium / Priority | 20 | 5 |

Dispatched September 7, 2026 at 22:48 UTC from code commit `d14dfa55d5f22106686cd285bc7319aab422cab0`. The two drivers were confirmed to load their frozen workflow source. [Launch record](launch.json). Live status refreshes every 30 seconds:

- [vLLM progress](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260907T224450Z_vllm/STATUS.md)
- [Codex progress](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260907T224450Z_codex/STATUS.md)

Both batches freeze the same source and paired dataset bytes in their own output directories. Repeat IDs match across paired versions and models. The manifest records assignments, hashes, policy, settings, and the code commit. The old full experiments are separate; their sessions and incomplete results are not resumed.

The clinical pair is `es-v2-nsclc_clinical-42000`, the same numerical pair used in the earlier smoke test. Validation remains at rounds 5, 12, and 20 with a two-round reassessment deadline; voluntary validation is limited to 10 new comparisons. Each call retains 125,000 completion tokens, two technical retries, and an 1,800-second timeout. Codex uses its default sampling and a sampled-token budget rather than a vLLM generation parameter.

Each output directory contains `manifest.json`, `config.yaml`, `execution.json`, `status.json`, `STATUS.md`, API metadata, per-run transcripts and reports, and eventual model-specific summaries and `RESULTS.md`. Failed stages remain recorded and penalized under the existing scoring policy. Account usage limits pause Codex transport requests for retry; no usage-reset credit is redeemed by this runner.

[Prompt refactor and verification](../workflow_v3_2/README.md) · [Response examples](../../../docs/EXPECTED_SURPRISING_PROMPTING.md)

[Byte-level source audit](source_provenance.json): the frozen code matches the implementation commit apart from one blank line in `schemas.py`; its parsed Python AST is identical. Subsequent edits in the shared checkout cannot change these frozen runs.

## Completed results and Terra extension

[Full clinical comparison](RESULTS.md) now includes the 60 completed Luna/Sol/Astra runs and all 20 finished Gemma runs. Gemma completed 1,979 of 2,000 stages; 10 runs had unrecovered stage errors. The report retains the original failure penalties and explains why primary F1* differs from the harmonic mean of the displayed R and P.

On September 8 at 10:19 UTC, a further 20 runs were launched for `gpt-5.6-terra`, with medium reasoning, explicit normal tier (`service_tier="default"`), and five concurrent sessions. Source and input hashes exactly match the original frozen Codex batch. [Terra launch record](terra_launch.json) · [Terra progress](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908T103000Z_terra/STATUS.md). The directory identifier is a label; `launch.json` and `execution.json` record actual start times.

A thread heartbeat checks Terra every 15 minutes and will refresh the report and notify on completion or failure. The reproducible updater is `scripts/expected_surprising/update_full_clinical_comparison.py`, run with the original frozen Codex `source/src` on `PYTHONPATH` and `/tmp/ocs-es-refactor-venv/bin/python`. It includes only endpoints for which all assigned runs have finished and reports exist; pending endpoints remain explicitly listed. It preserves the E/B definitions and Astra diagnosis and refreshes the previously shared Codex report link as well as this tracked report.

[Official Codex service-tier configuration](https://learn.chatgpt.com/docs/config-file/config-reference) documents the explicit tier override; Terra's archived command confirms `service_tier="default"` and `model_reasoning_effort="medium"`.
