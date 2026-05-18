# Qwen 3.5 9B BF16 CAA Codex AB Benchmark Plan

This runbook is for rerunning the CAA/Codex AB benchmark with
`Qwen/Qwen3.5-9B`. It intentionally recalculates steering vectors instead of
reusing the Gemma vectors.

Model facts checked on 2026-05-18 from the official Hugging Face model card:
`Qwen/Qwen3.5-9B` is a Transformers-format post-trained model, uses a 9B
language model with 32 layers, and supports OpenAI-compatible serving in
frameworks such as vLLM and SGLang. This benchmark uses the repo's
Transformers CAA server because the current CAA implementation needs
Transformers hidden states and decoder-layer hooks.

## Key Constraints

- Use the BF16 Transformers checkpoint path for vector derivation and serving.
- Do not reuse Gemma vector artifacts. Recalculate vectors with the exact Qwen
  model path that will be served.
- Qwen3.5 is multimodal, but this benchmark is text-only. The processor/model
  may still require `torchvision` and `pillow` to import.
- Qwen3.5 thinks by default. Keep `--disable-thinking` unless you deliberately
  want a thinking-mode benchmark and label it separately.
- If this repo has uncommitted CAA server parser/compact-context fixes, do not
  reset them before running this plan. They are relevant to local Codex
  tool-turn stability.

## Fixed Experiment

- Model: `Qwen/Qwen3.5-9B`
- Model tag: `qwen35_9b_bf16`
- Steering concept: `paradigm_orthogonalized`
- Vector layers to derive: `8,16,24`
- Steering layer: `24`
- Activation position: `last`
- Arms: `control`, `neg002`, `neg005`, `neg010`
- Scales: `0`, `-0.02`, `-0.05`, `-0.10`
- Pilot: 10 clinical bundles, 1 replicate per arm, `max_iterations=10`
- Full: same bundles, 5 replicates per arm, `max_iterations=25`
- Harness: Codex CLI through CAA model profiles
- Judge: existing default scoring backend for comparability

Layer rationale: Qwen3.5-9B has 32 language layers, so `8,16,24` samples early
mid, middle, and late layers. Layer `24` is the fixed intervention layer for the
AB run. If the local config gate reports a different layer count, stop and
revise both the vector layers and steering layer before deriving vectors.

## 1. Repo And Environment Setup

From the repository root:

```bash
cd /path/to/onc-co-scientist
```

Create or activate the Python environment. The repo-local `.venv` is simplest:

```bash
uv sync --extra interventions --extra vllm-openai
uv pip install fastapi uvicorn torchvision pillow
export PYTHON="$PWD/.venv/bin/python"
```

Set common paths:

```bash
export HF_CACHE=/path/to/huggingface-cache
export PAIRS=data/caa/clinical_all_claude_pubmed_realistic_pairs.jsonl
export VECTOR_DIR=data/caa
export MODEL_ID_OR_PATH=Qwen/Qwen3.5-9B
export MODEL_TAG=qwen35_9b_bf16
export VECTOR="$VECTOR_DIR/${MODEL_TAG}_clinical_pubmed_layers8_16_24.npz"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If you download to a local snapshot, set `MODEL_ID_OR_PATH` to that snapshot
directory instead of the remote model ID.

Confirm prerequisites:

```bash
nvidia-smi
test -f "$PAIRS"
"$PYTHON" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Run the local tests that cover the CAA server protocol helpers:

```bash
uv run --with pytest pytest tests/test_caa_server.py tests/test_caa_ab.py -q
uv run --with ruff ruff check src/onc_co_scientist/caa_server.py tests/test_caa_server.py
```

## 2. Download Or Verify Model Files

Download if the model is not already cached:

```bash
huggingface-cli download "$MODEL_ID_OR_PATH" --cache-dir "$HF_CACHE"
```

If access requires authentication:

```bash
huggingface-cli login
```

## 3. Qwen Transformers Gate

Confirm the checkpoint resolves locally and has the expected layer count before
deriving vectors or starting the server:

```bash
"$PYTHON" - <<'PY'
import os
from transformers import AutoConfig, AutoProcessor

model_id = os.environ["MODEL_ID_OR_PATH"]
cache_dir = os.environ.get("HF_CACHE")

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    local_files_only=True,
    trust_remote_code=True,
)
config = AutoConfig.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    local_files_only=True,
    trust_remote_code=True,
)
text_config = getattr(config, "text_config", None)
print(type(processor))
print("model_type", getattr(config, "model_type", None))
print("num_hidden_layers", getattr(config, "num_hidden_layers", None))
print("text_num_hidden_layers", getattr(text_config, "num_hidden_layers", None))
PY
```

Gate condition: continue only if the effective language layer count is `32`.
If the model is not already cached, download it first or replace
`local_files_only=True` with `False` for this gate.

## 4. Generate Steering Vectors

Derive Qwen-specific vectors:

```bash
PYTHONPATH=src "$PYTHON" -m onc_co_scientist.cli caa derive \
  --pairs "$PAIRS" \
  --out "$VECTOR" \
  --model "$MODEL_ID_OR_PATH" \
  --cache-dir "$HF_CACHE" \
  --layers 8,16,24 \
  --position last \
  --dtype bfloat16 \
  --device-map auto \
  --local-files-only \
  --trust-remote-code \
  --disable-thinking
```

If using a remote model ID and it is not already cached, replace
`--local-files-only` with `--allow-download`.

Verify the artifact:

```bash
uv run ocs caa describe --vector-file "$VECTOR"
```

Expected concepts:

- `oncology_knowledge`
- `paradigm_adherence`
- `paradigm_orthogonalized`

Expected layers:

- `8`
- `16`
- `24`

Gate condition: the metadata `requested_model` should match
`$MODEL_ID_OR_PATH`, not a Gemma model path.

## 5. Configure Codex Profiles

Add Qwen-specific profiles to the Codex config used by the benchmark host,
usually `$CODEX_HOME/config.toml` or `~/.codex/config.toml`.

```toml
[model_providers.caa_local]
name = "CAA Local"
base_url = "http://127.0.0.1:8765/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"

[profiles.qwen35-caa-control]
model_provider = "caa_local"
model = "qwen35-9b-control"

[profiles.qwen35-caa-neg002]
model_provider = "caa_local"
model = "qwen35-9b-caa-l24-neg002"

[profiles.qwen35-caa-neg005]
model_provider = "caa_local"
model = "qwen35-9b-caa-l24-neg005"

[profiles.qwen35-caa-neg010]
model_provider = "caa_local"
model = "qwen35-9b-caa-l24-neg010"
```

Set the local API key placeholder in any shell that launches manual Codex
smokes:

```bash
export OPENAI_API_KEY=EMPTY
```

## 6. Start CAA Server

Start with the non-compact request path:

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
  --cache-dir "$HF_CACHE" \
  --attn-implementation none \
  --cache-implementation none \
  --default-max-new-tokens 4096 \
  --alias-prefix qwen35-9b \
  --steering-layer 24 \
  --concept paradigm_orthogonalized \
  --disable-thinking
```

Keep this server running across all arms so the model loads once.

If full Codex context causes OOM or repeated server 500s, restart with compact
context and report the benchmark as a compact-adapter run:

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
  --cache-dir "$HF_CACHE" \
  --attn-implementation none \
  --cache-implementation none \
  --default-max-new-tokens 4096 \
  --alias-prefix qwen35-9b \
  --steering-layer 24 \
  --concept paradigm_orthogonalized \
  --disable-thinking \
  --compact-agent-context
```

Do not silently mix compact and non-compact results.

## 7. Server And Codex Smoke Tests

Health and model list:

```bash
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:8765/v1/models
```

Direct Chat Completions smoke:

```bash
curl -sS http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-9b-control","messages":[{"role":"user","content":"Reply exactly READY."}],"max_tokens":16}'
```

Codex text smoke:

```bash
codex exec --json --profile qwen35-caa-control --sandbox danger-full-access --skip-git-repo-check \
  'Reply exactly SMOKE_OK.'
```

Codex tool smoke:

```bash
codex exec --json --profile qwen35-caa-control --sandbox danger-full-access --skip-git-repo-check \
  'Use bash to run pwd exactly once. After the bash result is returned, do not call any tool again. Reply exactly TOOL_OK.'
```

Use `--sandbox danger-full-access` if this host still fails Codex
`workspace-write` commands with a `bwrap` loopback error.

Gate condition: do not start the benchmark until the text smoke returns
`SMOKE_OK`, the tool smoke shows exactly one real shell command execution, and
the final assistant text is `TOOL_OK`. If Qwen emits an unparsed tool marker,
repeatedly calls the same inspection command, or the CAA server returns 500s,
stop and fix the parser/compact-context behavior before launching the pilot.

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
  --harness-spec codex \
  --harness-profile codex \
  --codex-profile-prefix qwen35-caa \
  --model-alias-prefix qwen35-9b \
  --steering-layer 24 \
  --python-env .venv \
  --build-only
```

Run one bundle manually:

```bash
cd "data/caa_ab/${MODEL_TAG}_gate/pilot/control/tasks/aml/anonymized"

codex exec --profile qwen35-caa-control --sandbox danger-full-access --skip-git-repo-check \
  'Read agent_instructions.md in the current working directory and follow its instructions exactly. Use python3 for statistical analysis. Emit transcript.json and analysis_summary.txt in this directory when done. Do not access any files outside this directory.'
```

Validate output:

```bash
test -f transcript.json
test -f analysis_summary.txt
"$PYTHON" -m json.tool transcript.json >/tmp/transcript.validated.json
```

Gate condition: continue only if the transcript is real, not simulated, and
contains analyses from tool-executed statistical checks.

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
  --harness-spec codex \
  --harness-profile codex \
  --codex-profile-prefix qwen35-caa \
  --model-alias-prefix qwen35-9b \
  --steering-layer 24 \
  --python-env .venv \
  --codex-extra-args "--sandbox danger-full-access"
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

Expected count is 40 if there are 10 bundles, 2 variants, 4 arms, and 1
replicate.

Do not proceed to full unless all expected transcripts exist and scoring
completed for every arm.

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
  --harness-spec codex \
  --harness-profile codex \
  --codex-profile-prefix qwen35-caa \
  --model-alias-prefix qwen35-9b \
  --steering-layer 24 \
  --python-env .venv \
  --codex-extra-args "--sandbox danger-full-access"
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

## 11. Troubleshooting

If vector derivation cannot find decoder layers:

- Confirm the Transformers gate reports a text/language config with 32 layers.
- Confirm the repo's `find_decoder_layers` recognizes the Qwen model class.
- If it does not, add Qwen-specific decoder-layer discovery before deriving
  vectors.

If the server OOMs:

- Confirm no old server process is holding GPU memory with `nvidia-smi`.
- Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Try `--device-map balanced_low_0`.
- Use `--compact-agent-context` and report it.
- Lower `--default-max-new-tokens` for smoke tests only.

If Codex cannot execute shell commands:

- If `workspace-write` fails with `bwrap: loopback: Failed RTM_NEWADDR`, use
  `--sandbox danger-full-access` on this benchmark host.
- Keep that sandbox choice consistent across pilot and full runs.

If Codex calls a tool but repeats the same command:

- Stop before launching pilot/full.
- Confirm duplicate tool calls are deduped in the CAA server.
- Confirm `role: tool` outputs are preserved and compacted in compact mode.
- Re-run the `pwd` smoke.

If transcripts are missing:

```bash
find "data/caa_ab/${MODEL_TAG}" -path '*/runs/run_*/harness.log' | head
```

Inspect the relevant `harness.log` before retrying.

If scoring fails:

- Preserve transcripts first.
- Retry scoring with `--skip-build --skip-harness`.
- If the default judge backend is unavailable, use a fallback only if it is
  clearly labeled in the report.

## 12. Interpretation Targets

Compare each steered arm against control on:

- `frac_novel` for named bundles
- `buried_score_named`
- `buried_score_anonymized`

Report:

- Exact `MODEL_ID_OR_PATH`
- Exact vector artifact path
- Whether the run used compact context
- Sandbox mode used by Codex
- Any deviations from layers `8,16,24` and steering layer `24`
