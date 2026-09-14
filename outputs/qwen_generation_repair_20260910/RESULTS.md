# Qwen generation repair — September 10

Qwen's batch is paused. Gemma continues from saved calls under the existing frozen implementation, driven by `gemma_control/execution.json`. Scheduled Codex monitoring remains disabled. No Qwen scientific run was restarted or rescored, and no saved result was deleted in this repair.

## Cause and correction

The previous coordinator explicitly sent temperature 0; the provider omitted top-p, top-k and repetition controls, leaving those to server defaults. That was not the recommended Qwen3.8 sampling configuration. It also accepted unconstrained text, allowing repeated JSON objects and XML-like tag loops. The previous final-retry thinking toggle kept temperature 0 and did not switch to the recommended non-thinking sampling settings.

[Qwen's official model card](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices) specifies:

| Setting | Thinking, medium reasoning | Non-thinking final retry |
|---|---:|---:|
| Temperature | 1.0 | 0.7 |
| Top-p | 0.95 | 0.80 |
| Top-k | 20 | 20 |
| Min-p | 0.0 | 0.0 |
| Presence penalty | 0.0 | 1.5 |
| Repetition penalty | 1.0 | 1.0 |

The candidate provider supports an explicit `qwen3_8` sampling profile and opt-in `response_format: {type: json_object}`. The server constrains the final answer's JSON syntax; stage schemas and scientific checks remain independent and mandatory. Ambiguous multiple JSON objects are still rejected rather than arbitrarily choosing one. Responses stopped by the token limit are rejected with output usage retained. Effective generation settings are recorded in durable requests and provider response metadata. Other providers and existing frozen runs do not inherit this profile automatically.

The endpoint reports vLLM 0.28.0 and the expected Inferact/Qwen3.8-27B-NVFP4 model. Diagnostic responses separate reasoning from final content. Server launch flags were not independently inspected, and the server was not restarted.

## Validation

Thirteen targeted regression tests passed. Six diagnostic calls reused three actual failed prompts (sequential appraisal, persistent appraisal, sequential synthesis), each in thinking and non-thinking mode. All six returned a single valid JSON object, passed the stage schema, and ended with `finish_reason=stop`. Outputs ranged from 111 to 5,055 tokens, totaling 10,916, including reasoning. Diagnostic maximum output was 8,192 tokens per call, not the full-run 125,000-token budget.

These calls used the original first-attempt contexts; they are not replays of the entire failed trajectories. The sampling and JSON constraint changes were tested together, so these results do not isolate their individual effects. They establish a useful improvement on the failing prompts, not full-run reliability or scientifically correct conclusions. A separate live adapter smoke checks the actual provider implementation in both modes; its results and output usage are in `adapter_smoke_results.json`.

Artifacts: `qwen_provider_candidate.json`, `validated_results.json`, `replay_results.json`, request/response JSON files, `adapter_smoke_results.json`, `tests.log`, and `pause.json`. All diagnostics are excluded from the scientific comparison. Gemma's interruption may repeat an unreturned request; saved calls are replayed without regenerating them.

The live provider smoke also passed both modes, with 415 and 106 output tokens respectively. Across all eight successful diagnostic calls, observed output usage was 11,437 tokens. The actual provider metadata confirms the requested sampling parameters and JSON constraint were sent.
