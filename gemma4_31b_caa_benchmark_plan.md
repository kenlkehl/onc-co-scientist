# Gemma 4 31B CAA OpenCode AB Benchmark Plan

This runbook is for moving the CAA/OpenCode AB benchmark to a larger GPU machine and rerunning the Gemma 4 31B experiment. It avoids machine-specific paths and Python environment names. It also includes an experimental path for `nvidia/Gemma-4-31B-IT-NVFP4` on Blackwell GPUs.

## Key Constraint

The current CAA implementation derives vectors and applies steering with Transformers-level hidden states and decoder-layer hooks. The NVIDIA NVFP4 checkpoint is intended for vLLM/modelopt inference on Blackwell. Before using NVFP4 for this benchmark, verify that it can load through the repo's Transformers-based CAA path. If it only works through vLLM, the current server cannot apply CAA hooks to it without additional adapter work.

## Fixed Experiment

- Steering concept: `paradigm_orthogonalized`
- Steering layer: `40`
- Vector layers to derive: `20,30,40`
- Activation position: `last`
- Arms: `control`, `neg002`, `neg005`, `neg010`
- Scales: `0`, `-0.02`, `-0.05`, `-0.10`
- Pilot: 10 clinical bundles, 1 replicate per arm, `max_iterations=10`
- Full: same bundles, 5 replicates per arm, `max_iterations=25`
- Harness: OpenCode CLI through the CAA server's OpenAI-compatible Chat Completions API
- Judge: existing default scoring backend for comparability

## 1. Repo And Environment Setup

From the repository root:

```bash
cd /path/to/onc-co-scientist
```

Create or activate the Python environment you want to use. On a fresh machine, the repo-local `.venv` is the simplest option:

```bash
uv sync
export PYTHON="$PWD/.venv/bin/python"
```

Keep vLLM experiments in a separate environment unless you are explicitly
testing vLLM. Installing vLLM into the benchmark environment can change Torch,
Triton, and CUDA package versions.

If you use another environment, set `PYTHON` to that interpreter instead:

```bash
export PYTHON=/path/to/python
```

Set common paths and model choice:

```bash
export HF_CACHE=/path/to/huggingface-cache
export PAIRS=data/caa/clinical_all_claude_pubmed_realistic_pairs.jsonl
export VECTOR_DIR=data/caa
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Preferred experimental Blackwell target:

```bash
export MODEL_ID_OR_PATH=nvidia/Gemma-4-31B-IT-NVFP4
export MODEL_TAG=gemma4_31b_nvfp4
```

BF16 fallback target:

```bash
export MODEL_ID_OR_PATH=google/gemma-4-31B-it
export MODEL_TAG=gemma4_31b_bf16
```

If you download the model to a local snapshot, set `MODEL_ID_OR_PATH` to that local directory instead.

Set the vector output from the chosen model tag:

```bash
export VECTOR="$VECTOR_DIR/${MODEL_TAG}_clinical_pubmed_layers20_30_40.npz"
```

Confirm prerequisites:

```bash
nvidia-smi
test -f "$PAIRS"
"$PYTHON" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Download Or Verify Model Files

If the model is not already cached locally, download it with Hugging Face tooling:

```bash
huggingface-cli download "$MODEL_ID_OR_PATH" --cache-dir "$HF_CACHE"
```

If access requires authentication:

```bash
huggingface-cli login
```

## 3. NVFP4 Compatibility Gate

Run this before deriving vectors with `nvidia/Gemma-4-31B-IT-NVFP4`.

```bash
"$PYTHON" - <<'PY'
import os
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = os.environ["MODEL_ID_OR_PATH"]
cache_dir = os.environ.get("HF_CACHE")

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    device_map="auto",
    dtype="auto",
    trust_remote_code=True,
)
print(type(processor))
print(type(model))
print(getattr(model.config, "model_type", None))
print(getattr(model.config, "num_hidden_layers", None))
PY
```

Decision:

- If this succeeds, continue with the NVFP4 model through the current CAA path.
- If this fails because the checkpoint requires vLLM/modelopt-only loading, use the BF16 fallback for the current benchmark or pause to implement a vLLM-compatible CAA steering server.

Optional vLLM-only smoke test for the NVFP4 checkpoint:

```bash
vllm serve "$MODEL_ID_OR_PATH" --quantization modelopt
```

This confirms the checkpoint can run, but it does not by itself make it compatible with the current CAA server.

## 4. Generate Steering Vectors

Derive vectors for whichever model is set in `MODEL_ID_OR_PATH`.

```bash
PYTHONPATH=src "$PYTHON" -m onc_co_scientist.cli caa derive \
  --pairs "$PAIRS" \
  --out "$VECTOR" \
  --model "$MODEL_ID_OR_PATH" \
  --cache-dir "$HF_CACHE" \
  --layers 20,30,40 \
  --position last \
  --dtype bfloat16 \
  --device-map auto \
  --local-files-only \
  --trust-remote-code
```

If using a remote model ID and it is not already cached, replace `--local-files-only` with `--allow-download`.

Verify the artifact:

```bash
uv run ocs caa describe --vector-file "$VECTOR"
```

Expected concepts:

- `oncology_knowledge`
- `paradigm_adherence`
- `paradigm_orthogonalized`

Expected layers:

- `20`
- `30`
- `40`

## 5. Configure OpenCode Provider

Install OpenCode if it is not already available:

```bash
npm install -g opencode-ai
opencode --version
```

Use an inline OpenCode config for the benchmark run instead of writing a
permanent global provider file:

```bash
export OPENCODE_CONFIG_CONTENT='{
  "$schema": "https://opencode.ai/config.json",
  "enabled_providers": ["caa-local"],
  "model": "caa-local/gemma4-31b-control",
  "small_model": "caa-local/gemma4-31b-control",
  "provider": {
    "caa-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "CAA Local",
      "options": {
        "baseURL": "http://127.0.0.1:8765/v1",
        "apiKey": "EMPTY",
        "timeout": 600000
      },
      "models": {
        "gemma4-31b-control": {"name": "Gemma 4 31B CAA control"},
        "gemma4-31b-caa-l40-neg002": {"name": "Gemma 4 31B CAA neg002"},
        "gemma4-31b-caa-l40-neg005": {"name": "Gemma 4 31B CAA neg005"},
        "gemma4-31b-caa-l40-neg010": {"name": "Gemma 4 31B CAA neg010"}
      }
    }
  },
  "permission": "allow"
}'
export OPENCODE_DISABLE_AUTOUPDATE=1
export OPENCODE_DISABLE_MODELS_FETCH=1
```

Confirm OpenCode sees the CAA aliases:

```bash
opencode models caa-local
```

The model aliases stay stable even if the underlying model is NVFP4. Record the
actual `MODEL_ID_OR_PATH` and `MODEL_TAG` in the final report.

This runbook assumes the CAA server includes the structured tool-turn fix that
preserves assistant `tool_calls` and `role: tool` results. Without that fix,
Gemma 4 can call tools but may repeat the same tool call after receiving the
tool result. A plain vLLM comparison with `--tool-call-parser gemma4` should
not loop on the `pwd` smoke test below.

## 6. Start CAA Server

Start with the realistic, non-compacted request path:

```bash
PYTHONPATH=src "$PYTHON" -m onc_co_scientist.cli caa serve \
  --host 127.0.0.1 \
  --port 8765 \
  --model "$MODEL_ID_OR_PATH" \
  --vector-file "$VECTOR" \
  --dtype bfloat16 \
  --device-map auto \
  --local-files-only \
  --trust-remote-code \
  --default-max-new-tokens 4096 \
  --alias-prefix gemma4-31b \
  --steering-layer 40 \
  --concept paradigm_orthogonalized
```

Keep this server running across all arms so the model loads once.

If using a remote model ID and it is not already cached, replace `--local-files-only` with `--allow-download`.

If full OpenCode context still causes OOM, restart with the opt-in compact adapter:

```bash
PYTHONPATH=src "$PYTHON" -m onc_co_scientist.cli caa serve \
  --host 127.0.0.1 \
  --port 8765 \
  --model "$MODEL_ID_OR_PATH" \
  --vector-file "$VECTOR" \
  --dtype bfloat16 \
  --device-map auto \
  --local-files-only \
  --trust-remote-code \
  --default-max-new-tokens 4096 \
  --alias-prefix gemma4-31b \
  --steering-layer 40 \
  --concept paradigm_orthogonalized \
  --compact-agent-context
```

If using `--compact-agent-context`, report the benchmark as a compact-adapter run. Do not silently mix compact and non-compact results.

## 7. Server And OpenCode Smoke Tests

Health and model list:

```bash
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:8765/v1/models
```

Direct Chat Completions smoke test:

```bash
curl -sS http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-31b-control","messages":[{"role":"user","content":"Reply exactly READY."}],"max_tokens":16}'
```

OpenCode text smoke test:

```bash
opencode run \
  --format json \
  --model caa-local/gemma4-31b-control \
  --dangerously-skip-permissions \
  'Reply exactly SMOKE_OK.'
```

OpenCode tool smoke test:

```bash
opencode run \
  --format json \
  --model caa-local/gemma4-31b-control \
  --dangerously-skip-permissions \
  'Use bash to run pwd exactly once. After the bash result is returned, do not call any tool again. Reply exactly TOOL_OK.'
```

Gate condition: do not start the benchmark until the OpenCode text smoke returns
`SMOKE_OK`, the tool smoke shows exactly one real `bash` command execution, and
the final assistant text is `TOOL_OK`. If the model repeats `pwd`, stop and
verify the structured tool-turn fix is present in the CAA server.

## 8. Single-Bundle Gate

Before running all arms, test one clinical bundle with the control profile.

Build pilot task roots only:

```bash
uv run ocs caa run-ab \
  --root "data/caa_ab/${MODEL_TAG}_gate" \
  --stage pilot \
  --arms control \
  --replicates 1 \
  --max-iterations 10 \
  --jobs 1 \
  --harness-spec opencode \
  --harness-profile opencode \
  --model-alias-prefix gemma4-31b \
  --steering-layer 40 \
  --python-env .venv \
  --build-only
```

Run one bundle manually. Adjust the bundle path if the task set differs:

```bash
cd "data/caa_ab/${MODEL_TAG}_gate/pilot/control/tasks/aml/anonymized"

opencode run \
  --format json \
  --model caa-local/gemma4-31b-control \
  --dangerously-skip-permissions \
  'Read agent_instructions.md in the current working directory and follow its instructions exactly. Use python3 for statistical analysis. Emit transcript.json and analysis_summary.txt in this directory when done. Do not access any files outside this directory.'
```

Validate output:

```bash
test -f transcript.json
test -f analysis_summary.txt
"$PYTHON" -m json.tool transcript.json >/tmp/transcript.validated.json
```

Gate condition: continue only if the transcript is real, not simulated, and contains analyses that came from tool-executed statistical checks.

## 9. Run Pilot AB

Use a fresh root, not a failed or partially tested root:

```bash
uv run ocs caa run-ab \
  --root "data/caa_ab/${MODEL_TAG}" \
  --stage pilot \
  --arms control,neg002,neg005,neg010 \
  --replicates 1 \
  --max-iterations 10 \
  --jobs 1 \
  --harness-spec opencode \
  --harness-profile opencode \
  --model-alias-prefix gemma4-31b \
  --steering-layer 40 \
  --python-env .venv
```

Generate pilot summary:

```bash
uv run ocs caa summarize-ab \
  --root "data/caa_ab/${MODEL_TAG}" \
  --stage pilot \
  --out "data/caa_ab/${MODEL_TAG}/summary/pilot"
```

Pilot transcript count:

```bash
find "data/caa_ab/${MODEL_TAG}/pilot" -path '*/runs/run_001/transcript.json' | wc -l
```

Expected count is 40 if there are 10 bundles, 2 variants, 4 arms, and 1 replicate.

Do not proceed to full unless all expected transcripts exist and scoring completed for every arm.

## 10. Run Full AB

Run full with 5 replicates and `max_iterations=25`:

```bash
uv run ocs caa run-ab \
  --root "data/caa_ab/${MODEL_TAG}" \
  --stage full \
  --arms control,neg002,neg005,neg010 \
  --replicates 5 \
  --max-iterations 25 \
  --jobs 1 \
  --harness-spec opencode \
  --harness-profile opencode \
  --model-alias-prefix gemma4-31b \
  --steering-layer 40 \
  --python-env .venv
```

Generate final summary:

```bash
uv run ocs caa summarize-ab \
  --root "data/caa_ab/${MODEL_TAG}" \
  --stage full \
  --out "data/caa_ab/${MODEL_TAG}/summary/full"
```

Expected outputs:

- `data/caa_ab/${MODEL_TAG}/summary/full/ab_summary.md`
- `data/caa_ab/${MODEL_TAG}/summary/full/ab_summary.csv`
- `data/caa_ab/${MODEL_TAG}/summary/full/per_bundle.csv`

## 11. Optional Output Cap Or Compact Context

Use these only for smoke tests or troubleshooting. Do not silently mix them into
benchmark results without reporting the change.

For short tool smokes, OpenCode can cap output tokens:

```bash
OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=64 opencode run \
  --format json \
  --model caa-local/gemma4-31b-control \
  --dangerously-skip-permissions \
  'Reply exactly SMOKE_OK.'
```

If full agent context causes CAA server OOMs, use `--compact-agent-context` on
the server and report the benchmark as a compact-adapter run. Prefer the
non-compact path for final results now that structured tool turns are preserved.

## 12. Troubleshooting

If the NVFP4 checkpoint loads in vLLM but not through Transformers:

- The current CAA server cannot apply steering to that vLLM model.
- Use BF16 through Transformers for the benchmark, or pause to implement a vLLM/modelopt steering path.

If the server OOMs during model load:

- Confirm GPU memory with `nvidia-smi`.
- Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Try `--device-map balanced_low_0`.
- Confirm no old server process is holding GPU memory.

If OpenCode does not list the CAA models:

- Confirm `OPENCODE_CONFIG_CONTENT` is exported in the shell that launches OpenCode.
- Confirm the CAA server is listening on `http://127.0.0.1:8765/v1`.

If OpenCode calls a tool but repeats the same tool after receiving its result:

- Stop before launching the pilot.
- Confirm the CAA server preserves structured assistant `tool_calls` and `role: tool` messages.
- Re-run the `pwd` smoke; a healthy run calls `pwd` once and then returns `TOOL_OK`.

If transcripts are missing:

```bash
find "data/caa_ab/${MODEL_TAG}" -path '*/runs/run_*/harness.log' | head
```

Inspect the relevant `harness.log` and OpenCode session export before retrying.

If scoring fails:

- First preserve transcripts.
- Retry scoring with `--skip-build --skip-harness`.
- If the default judge backend is unavailable, use a fallback only if it is clearly labeled in the report.

## 13. Interpretation Targets

Compare each steered arm against control on:

- `frac_novel` for named bundles
- `buried_score_named`
- `buried_score_anonymized`
- `fraction_uncovered_named`
- `fraction_uncovered_anonymized`
- `fraction_near_or_better_named`
- `fraction_near_or_better_anonymized`
- recovery-level counts

Prefer the smallest negative scale that improves anonymized buried discovery or named novelty without reducing overall uncovering or transcript validity.

Flag degradation if steered arms show:

- fewer valid transcripts
- lower uncovered fraction
- more malformed analyses
- simulated or non-tool-backed statistical results

## References

- NVIDIA `nvidia/Gemma-4-31B-IT-NVFP4` model card: https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
