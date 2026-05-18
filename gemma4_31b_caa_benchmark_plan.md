# Gemma 4 31B CAA Codex AB Benchmark Plan

This runbook is for moving the CAA/Codex AB benchmark to a machine with enough GPU memory to run `google/gemma-4-31b-it` with realistic Codex tool context.

## Fixed Experiment

- Model snapshot: `/data1/ken/models/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343`
- Contrast pairs: `data/caa/clinical_all_claude_pubmed_realistic_pairs.jsonl`
- Vector artifact: `data/caa/gemma4_31b_clinical_pubmed_layers20_30_40.npz`
- Steering concept: `paradigm_orthogonalized`
- Steering layer: `40`
- Arms: `control`, `neg002`, `neg005`, `neg010`
- Scales: `0`, `-0.02`, `-0.05`, `-0.10`
- Pilot: 10 clinical bundles, 1 replicate per arm, `max_iterations=10`
- Full: same bundles, 5 replicates per arm, `max_iterations=25`
- Harness: Codex CLI through OpenAI Responses API
- Judge: existing default scoring backend for comparability

## 1. Machine Setup

From the repository root:

```bash
cd /path/to/onc-co-scientist
```

Set common paths:

```bash
export MODEL=/data1/ken/models/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343
export CACHE_DIR=/data1/ken/models
export PAIRS=data/caa/clinical_all_claude_pubmed_realistic_pairs.jsonl
export VECTOR=data/caa/gemma4_31b_clinical_pubmed_layers20_30_40.npz
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Confirm GPUs and local model files:

```bash
nvidia-smi
test -d "$MODEL"
test -f "$PAIRS"
```

Use `~/oldtorch` for Transformers/CUDA execution, matching the current local workflow. If the model is not already present, download it before running with `--local-files-only`, or remove `--local-files-only` and use `--allow-download` on commands that support it.

## 2. Generate Steering Vectors

Derive the 31B vectors at layers 20, 30, and 40:

```bash
PYTHONPATH=src ~/oldtorch/bin/python -m onc_co_scientist.cli caa derive \
  --pairs "$PAIRS" \
  --out "$VECTOR" \
  --model "$MODEL" \
  --cache-dir "$CACHE_DIR" \
  --layers 20,30,40 \
  --position last \
  --dtype bfloat16 \
  --device-map auto \
  --local-files-only
```

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

## 3. Configure Codex Profiles

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

The server exposes these aliases when started with `--alias-prefix gemma4-31b --steering-layer 40`.

## 4. Start CAA Server

Start with the realistic, non-compacted request path:

```bash
PYTHONPATH=src ~/oldtorch/bin/python -m onc_co_scientist.cli caa serve \
  --host 127.0.0.1 \
  --port 8765 \
  --model "$MODEL" \
  --vector-file "$VECTOR" \
  --dtype bfloat16 \
  --device-map auto \
  --local-files-only \
  --default-max-new-tokens 4096 \
  --alias-prefix gemma4-31b \
  --steering-layer 40 \
  --concept paradigm_orthogonalized
```

Keep this server running across all arms so the model loads once.

If full Codex context still causes OOM on the larger machine, restart with the opt-in compact adapter:

```bash
PYTHONPATH=src ~/oldtorch/bin/python -m onc_co_scientist.cli caa serve \
  --host 127.0.0.1 \
  --port 8765 \
  --model "$MODEL" \
  --vector-file "$VECTOR" \
  --dtype bfloat16 \
  --device-map auto \
  --local-files-only \
  --default-max-new-tokens 4096 \
  --alias-prefix gemma4-31b \
  --steering-layer 40 \
  --concept paradigm_orthogonalized \
  --compact-agent-context
```

If using `--compact-agent-context`, report the benchmark as a compact-adapter run. It should not be mixed silently with non-compact results.

## 5. Server Smoke Tests

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

Gate condition: do not start the benchmark until both Codex smoke tests complete, and the tool smoke shows an actual shell command execution.

## 6. Single-Bundle Gate

Before running all arms, test one clinical bundle with the control profile.

Build pilot task roots only:

```bash
uv run ocs caa run-ab \
  --root data/caa_ab/gemma31b_gate \
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

Run one bundle manually, replacing the path if needed:

```bash
cd data/caa_ab/gemma31b_gate/pilot/control/tasks/aml/anonymized

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

Gate condition: continue only if the transcript is real, not simulated, and contains analyses that actually came from tool-executed statistical checks.

## 7. Run Pilot AB

Use a fresh root, not any failed or partially tested root:

```bash
OPENAI_API_KEY=EMPTY uv run ocs caa run-ab \
  --root data/caa_ab/gemma31b \
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
  --root data/caa_ab/gemma31b \
  --stage pilot \
  --out data/caa_ab/gemma31b/summary/pilot
```

Pilot gate:

```bash
find data/caa_ab/gemma31b/pilot -path '*/runs/run_001/transcript.json' | wc -l
```

Expected count is 40 if there are 10 bundles, 2 variants, 4 arms, and 1 replicate.

Do not proceed to full unless all expected transcripts exist and scoring completed for every arm.

## 8. Run Full AB

Run full with 5 replicates and `max_iterations=25`:

```bash
OPENAI_API_KEY=EMPTY uv run ocs caa run-ab \
  --root data/caa_ab/gemma31b \
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
  --root data/caa_ab/gemma31b \
  --stage full \
  --out data/caa_ab/gemma31b/summary/full
```

Expected outputs:

- `data/caa_ab/gemma31b/summary/full/ab_summary.md`
- `data/caa_ab/gemma31b/summary/full/ab_summary.csv`
- `data/caa_ab/gemma31b/summary/full/per_bundle.csv`

## 9. Optional Reduced Tool Context

Only use this if full tool context OOMs but `--compact-agent-context` is not desired or not enough.

Pass reduced Codex tools through `--codex-extra-args`:

```bash
--codex-extra-args "--disable multi_agent --disable apps --disable browser_use --disable computer_use --disable image_generation"
```

Example pilot:

```bash
OPENAI_API_KEY=EMPTY uv run ocs caa run-ab \
  --root data/caa_ab/gemma31b_reduced_tools \
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

## 10. Troubleshooting

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
find data/caa_ab/gemma31b -path '*/runs/run_*/harness.log' | head
```

Inspect the relevant `harness.log` and Codex session log before retrying.

If scoring fails:

- First preserve transcripts.
- Retry scoring with `--skip-build --skip-harness`.
- If the default judge backend is unavailable, use a fallback only if it is clearly labeled in the report.

## 11. Interpretation Targets

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

