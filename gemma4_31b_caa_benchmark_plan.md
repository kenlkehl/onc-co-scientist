# Gemma 4 31B BF16 CAA Codex AB Benchmark Plan

This runbook is for moving the CAA/Codex AB benchmark to a larger GPU machine
and rerunning the Gemma 4 31B BF16 experiment. It avoids machine-specific paths
and Python environment names.

## Key Constraint

The current CAA implementation derives vectors and applies steering with
Transformers-level hidden states and decoder-layer hooks. Use the BF16
Transformers checkpoint for this benchmark; quantized inference-only
checkpoints are outside this benchmark path.

## Fixed Experiment

- Steering concept: `paradigm_orthogonalized`
- Steering layer: `40`
- Vector layers to derive: `20,30,40`
- Activation position: `last`
- Arms: `control`, `neg002`, `neg005`, `neg010`
- Scales: `0`, `-0.02`, `-0.05`, `-0.10`
- Pilot: 10 clinical bundles, 1 replicate per arm, `max_iterations=10`
- Full: same bundles, 5 replicates per arm, `max_iterations=25`
- Harness: Codex CLI through CAA model profiles
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
export MODEL_ID_OR_PATH=google/gemma-4-31B-it
export MODEL_TAG=gemma4_31b_bf16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

## 3. BF16 Transformers Gate

Confirm the BF16 checkpoint resolves locally through Transformers before
deriving vectors or starting the server.

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
print(type(processor))
print(getattr(config, "model_type", None))
print(getattr(config, "num_hidden_layers", None))
print(getattr(getattr(config, "text_config", None), "num_hidden_layers", None))
PY
```

If the model is not already cached, download it first or remove
`local_files_only=True` for this gate.

## 4. Generate Steering Vectors

Derive vectors for the BF16 model set in `MODEL_ID_OR_PATH`.

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

Configure Codex profiles that point each AB arm at the CAA server's
OpenAI-compatible local API. Add the profiles to the Codex config used by the
benchmark host, usually `$CODEX_HOME/config.toml` or `~/.codex/config.toml`.

Example profile shape:

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

Set the local API key placeholder in any shell that launches manual Codex
smokes:

```bash
export OPENAI_API_KEY=EMPTY
```

The smoke tests below confirm that the profiles can reach the running CAA
server.

This runbook assumes the CAA server includes the structured tool-turn fix that
preserves assistant `tool_calls` and `role: tool` results. Without that fix,
Gemma 4 can call tools but may repeat the same tool call after receiving the
tool result.

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
  --cache-implementation none \
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
  --cache-implementation none \
  --default-max-new-tokens 4096 \
  --alias-prefix gemma4-31b \
  --steering-layer 40 \
  --concept paradigm_orthogonalized \
  --compact-agent-context
```

If using `--compact-agent-context`, report the benchmark as a compact-adapter run. Do not silently mix compact and non-compact results.

## 7. Server And Codex Smoke Tests

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

Codex text smoke test:

```bash
codex exec --json --profile gemma-caa-control --sandbox workspace-write --skip-git-repo-check \
  'Reply exactly SMOKE_OK.'
```

Codex tool smoke test:

```bash
codex exec --json --profile gemma-caa-control --sandbox workspace-write --skip-git-repo-check \
  'Use bash to run pwd exactly once. After the bash result is returned, do not call any tool again. Reply exactly TOOL_OK.'
```

Gate condition: do not start the benchmark until the Codex text smoke returns
`SMOKE_OK`, the tool smoke shows exactly one real shell command execution, and
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
  --harness-spec codex \
  --harness-profile codex \
  --codex-profile-prefix gemma-caa \
  --model-alias-prefix gemma4-31b \
  --steering-layer 40 \
  --python-env .venv \
  --build-only
```

Run one bundle manually. Adjust the bundle path if the task set differs:

```bash
cd "data/caa_ab/${MODEL_TAG}_gate/pilot/control/tasks/aml/anonymized"

codex exec --profile gemma-caa-control --sandbox workspace-write --skip-git-repo-check \
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
  --harness-spec codex \
  --harness-profile codex \
  --codex-profile-prefix gemma-caa \
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
  --harness-spec codex \
  --harness-profile codex \
  --codex-profile-prefix gemma-caa \
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

## 11. Optional Compact Context

Use these only for smoke tests or troubleshooting. Do not silently mix them into
benchmark results without reporting the change.

If full agent context causes CAA server OOMs, use `--compact-agent-context` on
the server and report the benchmark as a compact-adapter run. Prefer the
non-compact path for final results now that structured tool turns are preserved.

## 12. Troubleshooting

If the server OOMs during model load:

- Confirm GPU memory with `nvidia-smi`.
- Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Try `--device-map balanced_low_0`.
- Confirm no old server process is holding GPU memory.
- Keep `--cache-implementation none`; this avoids static-cache generation
  failures observed with this Gemma 4 31B Transformers path.

If Codex does not reach the CAA server:

- Confirm the Codex `gemma-caa-*` profiles point at
  `http://127.0.0.1:8765/v1`.
- Confirm the CAA server is listening on `http://127.0.0.1:8765/v1`.

If Codex calls a tool but repeats the same tool after receiving its result:

- Stop before launching the pilot.
- Confirm the CAA server preserves structured assistant `tool_calls` and `role: tool` messages.
- Re-run the `pwd` smoke; a healthy run calls `pwd` once and then returns `TOOL_OK`.

If transcripts are missing:

```bash
find "data/caa_ab/${MODEL_TAG}" -path '*/runs/run_*/harness.log' | head
```

Inspect the relevant `harness.log` before retrying.

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
