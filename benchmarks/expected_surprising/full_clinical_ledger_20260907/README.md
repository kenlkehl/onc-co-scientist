# Full clinical comparison with simplified prompts

100 fresh runs: 10 repeats × expected/surprising clinical pair × 5 models. Each run has 25 rounds (100 research stages). The package selects the common `ledger-1.0.0` prompt for every backend.

| Model | Reasoning / tier | Assigned runs | Concurrent runs |
|---|---|---:|---:|
| Qwen 3.8 27B, sn4622130540:8000 | Server settings | 20 | 10 |
| Gemma 4 31B, camus:8002 | Server settings | 20 | 10 |
| Luna (`gpt-5.6-luna`) | Medium / Priority | 20 | 5 |
| Sol (`gpt-5.6-sol`) | Medium / Priority | 20 | 5 |
| Astra (`gpt-6-astra`) | Medium / Priority | 20 | 5 |

Launch is prepared; this document does not yet claim processes are running. Live status files will be populated on dispatch:

- [vLLM progress](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260907T224450Z_vllm/STATUS.md)
- [Codex progress](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260907T224450Z_codex/STATUS.md)

Both batches freeze the same source and paired dataset bytes in their own output directories. Repeat IDs match across paired versions and models. The manifest records assignments, hashes, policy, settings, and the code commit. The old full experiments are separate; their sessions and incomplete results are not resumed.

The clinical pair is `es-v2-nsclc_clinical-42000`, the same numerical pair used in the earlier smoke test. Validation remains at rounds 5, 12, and 20 with a two-round reassessment deadline; voluntary validation is limited to 10 new comparisons. Each call retains 125,000 completion tokens, two technical retries, and an 1,800-second timeout. Codex uses its default sampling and a sampled-token budget rather than a vLLM generation parameter.

Each output directory contains `manifest.json`, `config.yaml`, `execution.json`, `status.json`, `STATUS.md`, API metadata, per-run transcripts and reports, and eventual model-specific summaries and `RESULTS.md`. Failed stages remain recorded and penalized under the existing scoring policy. Account usage limits pause Codex transport requests for retry; no usage-reset credit is redeemed by this runner.

[Prompt refactor and verification](../workflow_v3_2/README.md) · [Response examples](../../../docs/EXPECTED_SURPRISING_PROMPTING.md)
