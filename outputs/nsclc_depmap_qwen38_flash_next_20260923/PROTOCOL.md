# NSCLC DepMap Qwen3.8 Flash Next grid

120 fresh runs: persistent, then sequential, then deliberative; 40 runs per phase.
Each phase contains ten repeats of expected/masked, expected/unmasked, surprising/masked, and surprising/unmasked.
25 iterations per run (explore, analyze, appraise, synthesize). The existing reviewed numerical datasets are copied unchanged.
Masked twins use the existing audited bijective transform; exact frame parity is verified. Evaluator mappings remain private.
Qwen model: Inferact/Qwen3.8-Flash-Next-NVFP4 at http://sn4622130540:8001/v1; 262144-token advertised context.
Qwen thinking sampling: temperature 1.0, top_p .95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1. Reasoning xhigh throughout, including retries.
125000 completion tokens per call; 7200-second HTTP timeout; two SDK transport retries; two stage repair retries. JSON-object responses.
Originally ten concurrent runs across both masking conditions; increased to twenty and then forty at the user's request on September 23 (see [operational amendment](CONCURRENCY.md)). Scientific configurations remain frozen. Persistent history budget 120000 characters. Sequential has fresh sessions. Deliberative uses two peer drafts and one chair per stage.
Existing repaired-harness policies: chair with available peers; retain scored scientific work after stage failures. All failures remain in assigned denominators and are reported separately.
Focal primary recovery is reported alongside the existing exact discovery score D; repeats measure stochastic variability on one synthetic dataset pair, not uncertainty across datasets.
A phase must finish and receive a reporting receipt before the next phase can launch. Source, configurations, schedule, and datasets are hashed before launch.

Sampling source: https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage

Interpretation note: the offline [masking audit](masking_investigation/20260923/REPORT.md) found exact numerical/scoring parity under renaming and a large unmasked advantage at six matched iterations, alongside a masked-only row-identifier prompt defect, removed assay descriptions, and a shared relative-versus-absolute effect-threshold mismatch. Scientific settings remain unchanged; preserve these limitations when interpreting the final grid.
