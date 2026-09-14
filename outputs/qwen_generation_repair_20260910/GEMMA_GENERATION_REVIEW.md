# Gemma 4 31B generation review

Gemma driver 3401402 was stopped at user request; Qwen remains paused and scheduled Codex monitoring remains disabled. Saved calls/results are preserved; no generation settings or scientific results were changed in this review.

| Setting | Frozen Gemma requests | Google recommendation |
|---|---|---|
| Temperature | 0 (explicit) | 1.0 |
| Top-p | Omitted; server default unknown | 0.95 |
| Top-k | Omitted; server default unknown | 64 |
| Presence/repetition penalties | Omitted | No additional universal values specified in the cited sampling section |
| Thinking | reasoning_effort=medium sent; no explicit thinking trigger in inspected system prompt or provider options | Enable through Gemma chat-template controls / system thinking token |
| Output limit | 125,000 tokens | No comparable universal limit specified in the cited sampling section |
| JSON constraint | Not requested | Separate formatting safeguard, not a sampling recommendation |

[Google model card: best practices](https://huggingface.co/google/gemma-4-31B-it#best-practices) recommends temperature 1.0, top-p 0.95 and top-k 64 across use cases. At temperature zero our requests use greedy decoding; this is a definite mismatch regardless of unspecified server sampling defaults. The exact effective server defaults were not inspected.

Google describes thinking as triggered by the system thinking token and warns against carrying thought text into ordinary conversation history. We did not explicitly configure that trigger; the generic medium setting alone does not prove thinking was active. The server/template may apply defaults, so this review does not assert that thinking was off. This needs verification before another batch.

Recommended next configuration: explicit temperature 1.0, top-p 0.95, top-k 64; explicit, verified Gemma thinking control; final-answer JSON constraint tested independently. Do not assume Qwen's non-thinking sampling schedule or presence penalty applies to Gemma. These mismatches do not establish that sampling caused the previously observed scientific/bookkeeping failures.
