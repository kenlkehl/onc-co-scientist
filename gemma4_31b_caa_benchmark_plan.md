# Gemma 4 31B CAA Codex AB Benchmark Plan

This runbook is for moving the CAA/Codex AB benchmark to a larger GPU machine and rerunning the Gemma 4 31B experiment. It avoids machine-specific paths and Python environment names. It also includes an experimental path for `nvidia/Gemma-4-31B-IT-NVFP4` on Blackwell GPUs.

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
- Harness: Codex CLI through OpenAI Responses API
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

## 5. Configure Codex Profiles

Add this to `~/.codex/config.toml` on the new machine:

```toml
[model_providers.caa_local]
name = "CAA Local"
base_url = "http://127.0.0.1:8765/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"

[profiles.gemma-caa-control]
model_provider = "caa_local"
model = "gemma4-31b-control"

[profiles.gemma-caa-neg002]
model_provider = "caa_local"
model = "gemma4-31b-caa-l40-neg002"

[profiles.gemma-caa-neg005]
model_provider = "caa_local"
model = "gemma4-31b-caa-l40-neg005"

[profiles.gemma-caa-neg010]
model_provider = "caa_local"
model = "gemma4-31b-caa-l40-neg010"
```

The profile names stay stable even if the underlying model is NVFP4. Record the actual `MODEL_ID_OR_PATH` and `MODEL_TAG` in the final report.

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

If full Codex context still causes OOM, restart with the opt-in compact adapter:

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

## 7. Server Smoke Tests

Health and model list:

```bash
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:8765/v1/models
```

Direct Responses API smoke test:

```bash
curl -sS http://127.0.0.1:8765/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-31b-control","input":"Reply exactly READY.","max_output_tokens":32}'
```

Codex text smoke test:

```bash
OPENAI_API_KEY=EMPTY codex exec \
  --sandbox workspace-write \
  --skip-git-repo-check \
  --profile gemma-caa-control \
  'Reply exactly SMOKE_OK.'
```

Codex tool smoke test:

```bash
OPENAI_API_KEY=EMPTY codex exec \
  --sandbox workspace-write \
  --skip-git-repo-check \
  --profile gemma-caa-control \
  'Use the shell to run pwd, then reply exactly TOOL_OK.'
```

Gate condition: do not start the benchmark until both Codex smoke tests complete and the tool smoke shows an actual shell command execution.

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
  --codex-profile-prefix gemma-caa \
  --model-alias-prefix gemma4-31b \
  --steering-layer 40 \
  --build-only
```

Run one bundle manually. Adjust the bundle path if the task set differs:

```bash
cd "data/caa_ab/${MODEL_TAG}_gate/pilot/control/tasks/aml/anonymized"

OPENAI_API_KEY=EMPTY codex exec \
  --sandbox workspace-write \
  --skip-git-repo-check \
  --profile gemma-caa-control \
  'Read agent_instructions.md in the current working directory and follow its instructions exactly. Emit transcript.json and analysis_summary.txt in this directory when done. Do not access any files outside this directory.'
```

Validate output:

```bash
test -f transcript.json
test -f analysis_summary.txt
python -m json.tool transcript.json >/tmp/transcript.validated.json
```

Gate condition: continue only if the transcript is real, not simulated, and contains analyses that came from tool-executed statistical checks.

## 9. Run Pilot AB

Use a fresh root, not a failed or partially tested root:

```bash
OPENAI_API_KEY=EMPTY uv run ocs caa run-ab \
  --root "data/caa_ab/${MODEL_TAG}" \
  --stage pilot \
  --arms control,neg002,neg005,neg010 \
  --replicates 1 \
  --max-iterations 10 \
  --jobs 1 \
  --codex-profile-prefix gemma-caa \
  --model-alias-prefix gemma4-31b \
  --steering-layer 40
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
OPENAI_API_KEY=EMPTY uv run ocs caa run-ab \
  --root "data/caa_ab/${MODEL_TAG}" \
  --stage full \
  --arms control,neg002,neg005,neg010 \
  --replicates 5 \
  --max-iterations 25 \
  --jobs 1 \
  --codex-profile-prefix gemma-caa \
  --model-alias-prefix gemma4-31b \
  --steering-layer 40
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

## 11. Optional Reduced Tool Context

Only use this if full tool context OOMs but `--compact-agent-context` is not desired or not enough.

Pass reduced Codex tools through `--codex-extra-args`:

```bash
--codex-extra-args "--disable multi_agent --disable apps --disable browser_use --disable computer_use --disable image_generation"
```

Example pilot:

```bash
OPENAI_API_KEY=EMPTY uv run ocs caa run-ab \
  --root "data/caa_ab/${MODEL_TAG}_reduced_tools" \
  --stage pilot \
  --arms control,neg002,neg005,neg010 \
  --replicates 1 \
  --max-iterations 10 \
  --jobs 1 \
  --codex-profile-prefix gemma-caa \
  --model-alias-prefix gemma4-31b \
  --steering-layer 40 \
  --codex-extra-args "--disable multi_agent --disable apps --disable browser_use --disable computer_use --disable image_generation"
```

If using this path, report it as a reduced-tool-context run.

## 12. Troubleshooting

If the NVFP4 checkpoint loads in vLLM but not through Transformers:

- The current CAA server cannot apply steering to that vLLM model.
- Use BF16 through Transformers for the benchmark, or pause to implement a vLLM/modelopt steering path.

If the server OOMs during model load:

- Confirm GPU memory with `nvidia-smi`.
- Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Try `--device-map balanced_low_0`.
- Confirm no old server process is holding GPU memory.

If Codex fails to refresh models but continues:

- This warning is not necessarily fatal. Continue if smoke tests work.

If Codex does not call tools:

- Stop and inspect the session log under `~/.codex/sessions`.
- Do not launch the pilot unless the single-bundle gate produces real files.

If transcripts are missing:

```bash
find "data/caa_ab/${MODEL_TAG}" -path '*/runs/run_*/harness.log' | head
```

Inspect the relevant `harness.log` and Codex session log before retrying.

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

