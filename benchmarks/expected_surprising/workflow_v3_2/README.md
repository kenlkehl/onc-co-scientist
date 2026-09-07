# Simplified agent interface: verification

The user stopped the full model comparison after Gemma encountered avoidable interface errors. This revision changes prompting and response translation, not the numerical datasets or private scoring rules. The user subsequently approved moving on to a fresh full 100-run batch; its launch record is linked below.

- Workflow: `appraisal-3.2.0`; prompt: `ledger-1.0.0`; agent forms: `stage-forms-1.0.0`.
- Validation: `delayed-3.0.0`; scoring: `profile-3.0.0`; numerical DGP: v2.
- [What changed, examples, and limitations](../../../docs/EXPECTED_SURPRISING_PROMPTING.md).
- [20-task package inventory](package_manifest.json) and [passed byte-preservation audit](package_audit.json).
- Local packages: `data/expected_surprising_ledger/`.

## Automated verification

[81 benchmark/provider tests pass](tests.txt), including provider-route checks for vLLM, Gemini/Vertex, Anthropic/Vertex, and Codex. [Ruff passes](ruff.txt) over the benchmark source, scripts, and tests.

The checks cover stage-specific fields; short references; automatic evidence attachment; acceptance without prior validation; withdrawals; validation requests and delayed reassessment; reversed evidence; stable references under rollback; rejection of unavailable citations; retained public/private separation; compatibility with the legacy interface; and memory updates only after successful stages.

## Live verification

[Launch settings](live_launch.json) record six short runs started September 7, 2026:

| Server/model | Assigned datasets | Rounds each | Concurrency |
|---|---|---:|---:|
| Gemma 4 31B at camus:8002 | Clinical NSCLC, expected and surprising | 6 | 2 |
| Qwen 3.8 27B at sn4622130540:8000 | Clinical and cell-line NSCLC, expected and surprising | 6 | 4 |

Each round has four stages. Automatic validation is released in rounds 2 and 4, with reassessment two rounds later. Settings retain 125,000 completion tokens per call, two retries per stage, and 1,800 seconds per call. Both endpoints' served model IDs were checked before launch. The new controller remains provider independent.

The checks were interrupted at the user's request to proceed to the full comparison, so none is presented as a complete six-round smoke run. **76 successful stages were recorded**, with no bookkeeping/response-form errors and no exhausted stages. Three Qwen calls used all 125,000 completion tokens in reasoning and returned no answer; all three recovered on retry. Gemma completed 28 stages across its two runs without retries.

| Run | Successful stages before interruption |
|---|---:|
| Gemma clinical expected | 20 / 24 |
| Gemma clinical surprising | 8 / 24 |
| Qwen clinical expected | 9 / 24 |
| Qwen clinical surprising | 10 / 24 |
| Qwen cell-line expected | 18 / 24 |
| Qwen cell-line surprising | 11 / 24 |

Raw prompts, responses, translations, timing, usage, `interruption.json`, and `interruption_status.json` are retained under the paths in the launch record. The source hashes matched before interruption. The small checks support proceeding with the interface; they do not establish comparative scientific performance or eliminate interface effects. Scientific output was inspected: for example, Gemma correctly interpreted a negative signed result for a negative-direction ECOG claim as a positive raw association, rejected the original direction, and proposed a reversed claim.

## Full-run restart

The same 100-run design is retained: 10 repeats × 2 paired clinical versions × 5 models, 25 rounds per run. Both vLLM models retain their settings; Luna, Sol, and Astra use medium reasoning and explicit `service_tier="priority"`. This setting requests Fast/Priority processing, as documented in the [official Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference#service_tier). Gemini/Vertex and Anthropic/Vertex were checked through the real registry/common workflow with mocked transports; no claim of live cloud-provider testing is made.

The stopped full experiments and interrupted smoke checks are separate from the new runs. Raw datasets and run state remain local and excluded from Git.

## Actual prompt examples

These are the first successful Gemma clinical-expected responses from the new live check, with their exact supplied prompts:

- [Exploration prompt](example_explore_prompt.txt) and [response](example_explore_response.txt).
- [Synthesis prompt](example_synthesize_prompt.txt) and [response](example_synthesize_response.txt).

The first exploration prompt is 13,387 characters. Its initial synthesis prompt is 23,146 characters; the synthesis response is 1,783 characters. These are observed sizes, not a controlled before/after comparison: the model's proposed claims and reasoning can differ between runs.
