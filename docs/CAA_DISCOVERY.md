# Gemma 4 31B CAA discovery experiment: execution handoff

This guide gives a new operator or agent the scientific context, required assets,
commands, acceptance criteria, and reporting requirements for this experiment.
Run the commands on the GPU host from a complete copy of the repository.

## 1. Objective and method

The experiment asks whether activation steering helps an LLM co-scientist follow
numerical evidence when a discovery challenges familiar biomedical expectations.
Here, **paradigm anchoring** means allowing a prior directional expectation to
outweigh the evidence supplied by the research dataset.

**CAA means contrastive activation addition.** It is an inference-time intervention
on a model's internal hidden activations. The model weights stay fixed. A
**contrast pair** contains two conversations with identical questions, prior
expectations, and numerical evidence, followed by different assistant responses:

- The positive response privileges the prior expectation when appraising the evidence.
- The negative response appraises the supplied evidence and its uncertainty.

For example, a vignette supplies strong evidence for an unexpected association.
The anchored response rejects the association; the evidence-responsive response
accepts it. Both responses are constructed training examples.

The implementation extracts the hidden activation at the last token of each
complete conversation, computes positive minus negative for each pair, and averages
those differences into a vector for each decoder layer:

`v_layer = mean(h_anchored - h_evidence_responsive)`

The corpus contains 24 appraisal contrasts in six synthetic cancer contexts and
two auxiliary oncology-knowledge contrasts. Another eight appraisal contrasts in
two separate contexts are reserved for development. These contexts are separate
from the NSCLC evaluation task. Extraction uses thinking disabled and the complete
assistant response, with the generation-prompt suffix omitted.

The code removes the projection of the anchoring vector onto the auxiliary
knowledge vector, producing `paradigm_orthogonalized`. During generation it adds
this vector to the decoder layer's residual output at every prompt and generated
token position:

`h_steered = h_original + scale * v_orthogonalized`

The current candidate uses zero-based layer **40** and scale **-0.05**. The negative
sign moves activations away from the constructed anchoring direction. The control
arm uses ordinary generation through the same loaded BF16 model and serving stack.
BF16 means the model weights use the bfloat16 numerical format.

This is a preliminary candidate intervention. Its effect on scientific discovery,
formatting reliability, and biomedical knowledge requires measurement. The contrast
wording, response lengths, and auxiliary knowledge examples are potential confounds.
Layer and scale selection should use the separate development examples and be
recorded before a scientific evaluation.

## 2. Scientific task and comparison

The evaluated task uses a frozen synthetic non-small-cell lung cancer (NSCLC)
clinical dataset pair, identified as `es-v2-nsclc_clinical-42000`. Each dataset has
six embedded discoveries. The paired datasets share covariates and random noise;
one focal discovery reverses direction between them, while the other five
specifications stay fixed.

The agent's public research goal is to discover meaningful predictor/outcome and
subgroup associations, request analyses, assess effect sizes and uncertainty, and
follow up promising findings. The controller holds the private discovery key and
performs scoring. Keep private keys, condition assignments, and masking maps outside
the evaluated model's messages.

| Condition | Focal discovery | Variable labels |
| --- | --- | --- |
| Expected / unmasked (EU) | Literature-expected direction | Biomedical names |
| Surprising / unmasked (SU) | Reversed direction | Biomedical names |
| Expected / masked (EM) | Literature-expected direction | Anonymized names and category labels |
| Surprising / masked (SM) | Reversed direction | Anonymized names and category labels |

Masking preserves rows and statistical relationships. Preparation verifies dataset
checksums, shared private pair identity, and the exact row-preserving transformation.

A **persistent agent** retains its conversation history and research notebook across
iterations. Each iteration has four stages:

1. **Explore:** propose comparisons or refinements.
2. **Analyze:** request up to 12 registered comparisons using `run_analyses`.
3. **Appraise:** assess discovery evidence and optionally request independent validation.
4. **Synthesize:** assess validation, update conclusions, and plan further investigation.

The Python controller owns the claim/evidence ledger, executes the scientific tools,
and validates stage JSON. It supplies the current ledger and notebook at every stage.
Older conversation turns are trimmed under an audited 120,000-character history
limit. An individual **cell** is one condition × model arm × replicate.

Use these settings for the full preliminary evaluation:

- Two arms: `gemma4-control` and `gemma4-caa`.
- Four conditions, 10 replicates per condition and arm: **80 cells**.
- Persistent workflow, 25 iterations: 100 primary stage calls per cell, 8,000 overall.
- **100,000 generated tokens per call**, including thinking and special tokens.
- Up to two scientific response-repair attempts per stage.
- Thinking enabled initially; the existing adapter disables it on the final repair
  after two consecutive truncations.
- Gemma sampling: temperature 1.0, top-p 0.95, top-k 64.
- Shared JSON prompts, parsing, and strict controller validation in both arms.
- BF16 weights, balanced GPU placement, SDPA attention, dynamic KV cache, and
  2,048-token prefill chunks. Prefill is the initial processing of the input prompt.

The output allowance and prefill chunk size are independent. The KV cache stores
attention state and grows with context. The optional `--max-context-tokens` server
setting limits **prompt tokens + requested output tokens**; reserve prompt capacity
in addition to the 100,000-token output allowance.

## 3. Files and software to bring to the GPU host

The originating repository is `/data1/ken/onc-co-scientist`. The September 15 runtime
fixes were present as local working-tree changes when this handoff was written.
Transfer the updated working tree, or commit and transfer those changes before
using a checkout on another host. Verify these features with the CLI help and tests
below: `--max-tokens`, `--request-timeout`, `--run-limit`, chunked prefill, and
infrastructure-failure halting.

Copy these assets while preserving their paths relative to the repository root:

- `src/onc_co_scientist/`, `scripts/expected_surprising/`, `tests/`, `pyproject.toml`, and `README.md`.
- `data/expected_surprising_ledger/full_runs/20260910_vllm_recommended/config.yaml`.
- The complete `data/expected_surprising_ledger/full_runs/20260910_vllm_recommended/input_data/`.
- The complete `data/expected_surprising_ledger/full_runs/20260910_masked_vllm_recommended/input_data/`.

Both input packages include `public/` and `private/` directories. The repository
ignores generated data, so transfer these packages explicitly. The launcher resolves
the paths above inside its own repository copy.

Use the exact `google/gemma-4-31B-it` checkpoint revision:

`145dc2508c480a64b47242f160d286cff94a2343`

The verified source-host snapshot is:

`/home/klkehl/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343`

Copy the complete snapshot and its referenced weight blobs onto local storage.
Hugging Face snapshots can contain symlinks; verify that their targets also exist on
the destination. The two weight shards total 62,546,338,248 bytes, approximately
62.5 GB. Allow additional disk space for the environment, data, and run artifacts.
The source copy receipt is `outputs/caa_discovery/20260914_hf_cache_copy.json`.

The source runtime uses Python 3.13.7, PyTorch 2.11.0, Transformers 5.14.1,
Accelerate 1.14.0, OpenAI Python SDK 2.52.0, FastAPI 0.136.3, and Uvicorn 0.52.0.
On a new machine, provision Python 3.12+ and a CUDA-enabled PyTorch build compatible
with its GPU driver. Install the project extras in that environment:

```bash
cd /absolute/path/to/onc-co-scientist
/absolute/path/to/environment/bin/python -m pip install \
  -e '.[interventions,vllm-openai,analysis,dev]' \
  'transformers==5.14.1' 'accelerate==1.14.0'
```

The serving endpoint is local and bound to `127.0.0.1:8765`. The client uses the
placeholder API key `EMPTY`. All model inference happens on the selected local GPUs.
The model loader runs offline after the checkpoint has been staged.

## 4. Configure and check the host

Set these values for the destination machine and retain them for the later commands:

```bash
export CAA_REPO=/absolute/path/to/onc-co-scientist
export CAA_PYTHON=/absolute/path/to/environment/bin/python
export CAA_MODEL=/absolute/path/to/local/gemma-4-31B-it/snapshot
export CAA_GPUS=0,1
export CAA_TRANSFER=cpu
export CAA_RUN_ID=gemma4_caa_20260915_host1

cd "$CAA_REPO"
export PYTHONPATH="$CAA_REPO/src"
export CUDA_VISIBLE_DEVICES="$CAA_GPUS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DEACTIVATE_ASYNC_LOAD=1 OMP_NUM_THREADS=4
export CAA_SMOKE="$CAA_REPO/outputs/caa_discovery/${CAA_RUN_ID}_smoke"
export CAA_RUNTIME="$CAA_REPO/outputs/caa_discovery/${CAA_RUN_ID}_acceptance_runtime"
export CAA_CAMPAIGN="$CAA_REPO/data/caa_discovery/${CAA_RUN_ID}_acceptance"

nvidia-smi
"$CAA_PYTHON" -c 'import torch, transformers; print(torch.__version__, transformers.__version__); print("CUDA available:", torch.cuda.is_available()); print("Visible GPUs:", torch.cuda.device_count())'
"$CAA_PYTHON" scripts/expected_surprising/run_local_caa_pilot.py --help
"$CAA_PYTHON" -m onc_co_scientist.cli caa serve --help
"$CAA_PYTHON" -m pytest -q tests/test_caa_capacity.py \
  tests/test_caa_local_pilot.py tests/test_caa_discovery.py
```

Choose a fresh `CAA_RUN_ID` for each independent attempt. Check that the allocated
GPUs are available and port 8765 is free. The launcher preserves an existing server
by refusing to start when the port is occupied.

The original host has two 48 GB RTX A6000s. It loaded BF16 weights and passed a
short numerical smoke, then exhausted VRAM on growing persistent-agent prompts.
Size the replacement deployment for weights, KV cache, and generation working
memory. Full-length capacity remains a measurement to perform on the new host.

`CAA_TRANSFER=cpu` stages inter-GPU activation copies through CPU memory while keeping
weights on GPUs. This mode resolved non-finite activations seen with direct transfers
on the original host. It can reduce throughput. On another host, `direct` is available
for a separately verified numerical smoke. Use the same transfer mode for derivation,
control, and CAA runs. CPU-staged transfers require compiled forward to remain disabled.

For a checkpoint still residing on network storage, add `--stage-local /local/scratch`
to the smoke command below. The script creates and verifies a new local copy; then
set `CAA_MODEL` to the `staged_checkpoint` path recorded in `smoke.json` before serving.

## 5. Derive the vectors and run the numerical smoke

```bash
"$CAA_PYTHON" scripts/expected_surprising/caa_local_smoke.py \
  --model "$CAA_MODEL" \
  --inter-gpu-transfer "$CAA_TRANSFER" \
  --attention sdpa --max-tokens 128 --out "$CAA_SMOKE"
"$CAA_PYTHON" -m json.tool "$CAA_SMOKE/smoke.json"
```

This bounded check derives the vectors at layers 20, 30, and 40 and makes three
short JSON generations: control with thinking off, CAA with thinking off, and control
with thinking on. Its 128-token output cap applies to this readiness check only.
Persistent-agent calls use 100,000 tokens in the following steps.

Proceed when `smoke.json` reports `status: completed`, every generation has a normal
stop and the correct `{"status":"ready"}` answer, all validation flags pass, and
weights are resident on the selected GPUs. Inspect the recorded peak VRAM and timing.

Keep the entire smoke directory. The launcher requires `vectors.npz`, `vectors.json`,
and `aliases.json`. It also contains the contrast corpora and provenance. The generated
aliases include a norm-matched random-vector arm for optional development comparisons;
the two-arm campaign selects `gemma4-control` and `gemma4-caa`.

## 6. Run a full-length acceptance cell

Launch one 25-iteration cell with the scientific tools and persistent conversation:

```bash
"$CAA_PYTHON" scripts/expected_surprising/run_local_caa_pilot.py \
  --runtime "$CAA_RUNTIME" --campaign "$CAA_CAMPAIGN" \
  --model "$CAA_MODEL" --smoke-artifacts "$CAA_SMOKE" \
  --iterations 25 --max-tokens 100000 --request-timeout 21600 \
  --run-limit 1 --gpus "$CAA_GPUS" --inter-gpu-transfer "$CAA_TRANSFER" \
  --prefill-chunk-size 2048 --max-hours 48
```

The 48-hour supervisor deadline is an example allocation for acceptance testing;
choose it to match the approved machine reservation. `--request-timeout 21600` allows
up to six hours for one inference request. Both limits should reflect measured
throughput. A six-iteration engineering pilot is available with `--iterations 6`;
its per-call output allowance remains 100,000.

The launcher creates a private frozen serving/controller source copy, starts its
own server, prepares eight condition/arm cells with one replicate each, and executes
the first cell in a seeded shuffled schedule. It returns a supervisor PID and the
location of `status.json`. It releases its own server and controller when execution
ends. A successful one-cell gate leaves seven cells queued and reports `gate_passed`.

Inspect progress and results:

```bash
"$CAA_PYTHON" -m json.tool "$CAA_RUNTIME/status.json"
tail -n 60 "$CAA_RUNTIME/controller.log"
tail -n 60 "$CAA_RUNTIME/server.log"
"$CAA_PYTHON" scripts/expected_surprising/caa_discovery.py verify "$CAA_CAMPAIGN"
"$CAA_PYTHON" scripts/expected_surprising/caa_discovery.py summarize "$CAA_CAMPAIGN"
```

Acceptance requires a completed cell with 25 completed iterations and 100 committed
primary stages. Inspect its scientific report and recorded calls for actual
`run_analyses` execution, coherent evidence appraisals, and validation handling.
Record repairs, truncations, input/output tokens, elapsed time, and peak GPU memory.
Scientific claim validation stays strict, including restrictions on conditioning
on the exposure or outcome.

After reviewing that cell, run the eight-cell one-replicate pilot by repeating the
launch command with fresh runtime/campaign paths and `--run-limit 8`. Review successful
persistent-agent behavior in both arms and all four conditions before scaling.

### Failure handling

A serving failure, timeout, or response-provenance mismatch immediately stops the
workflow and writes `halt.json` in the campaign directory. Failed scientific cells
also stop subsequent scheduling. CAA transport makes one HTTP attempt per recorded
request. A generation error latches the server into a failed state; subsequent
requests receive HTTP 503 and `/health` reports failure. The supervisor then releases
its owned processes. All requests, cached responses, and partial scientific artifacts
remain available for diagnosis.

Keep each halted campaign intact. Apply fixes in the working repository, then use
fresh output directories and a newly frozen campaign. A frozen campaign records its
source, data, dependencies, serving settings, and manifest fingerprint; resumed work
requires that same environment and implementation. To stop an active supervised run,
read `supervisor_pid` from its own status file, verify that process belongs to this
runtime, and send it SIGTERM so its cleanup handler can release the model server.

## 7. Run the 80-cell preliminary evaluation

Use this manually managed sequence after the acceptance results and compute budget
have been reviewed. Set a fresh campaign path:

```bash
export CAA_FULL="$CAA_REPO/data/caa_discovery/${CAA_RUN_ID}_full"
```

In a dedicated terminal with the same environment variables, start the server and
keep that terminal available throughout the campaign:

```bash
cd "$CAA_REPO"
"$CAA_PYTHON" -m onc_co_scientist.cli caa serve \
  --model "$CAA_MODEL" --dtype bfloat16 --device-map balanced \
  --inter-gpu-transfer "$CAA_TRANSFER" \
  --vector-file "$CAA_SMOKE/vectors.npz" \
  --aliases-file "$CAA_SMOKE/aliases.json" \
  --cache-implementation dynamic --prefill-chunk-size 2048 \
  --default-max-new-tokens 100000 --enable-thinking \
  --host 127.0.0.1 --port 8765
```

In the controller terminal, fetch the live serving manifest, freeze the evaluation,
check it, and execute the scheduled cells serially:

```bash
curl --fail --silent --show-error http://127.0.0.1:8765/v1/caa \
  --output "$CAA_SMOKE/${CAA_RUN_ID}_evaluation_manifest.json"
"$CAA_PYTHON" scripts/expected_surprising/caa_discovery.py prepare \
  --out "$CAA_FULL" \
  --named "$CAA_REPO/data/expected_surprising_ledger/full_runs/20260910_vllm_recommended/input_data" \
  --masked "$CAA_REPO/data/expected_surprising_ledger/full_runs/20260910_masked_vllm_recommended/input_data" \
  --base-config "$CAA_REPO/data/expected_surprising_ledger/full_runs/20260910_vllm_recommended/config.yaml" \
  --server-manifest "$CAA_SMOKE/${CAA_RUN_ID}_evaluation_manifest.json" \
  --arm gemma4-control --arm gemma4-caa --replicates 10 --workflow persistent \
  --max-tokens 100000 --request-timeout 21600 \
  --rationale 'Preliminary constructed appraisal vector; layer 40, scale -0.05; acceptance reviewed'
"$CAA_PYTHON" scripts/expected_surprising/caa_discovery.py verify "$CAA_FULL"
"$CAA_PYTHON" scripts/expected_surprising/caa_discovery.py run "$CAA_FULL" --limit 80
"$CAA_PYTHON" scripts/expected_surprising/caa_discovery.py summarize "$CAA_FULL"
```

Write the rationale to reflect the actual selection and acceptance evidence. The
base configuration supplies 25 iterations; `prepare` overrides the response budget
and request timeout with the explicit values above. `run --limit N` bounds newly
executed cells and defaults to one. Completed cells are retained when advancing an
unchanged campaign. The manual sequence uses an operator-managed wall-clock budget;
stop its server with Ctrl-C after completion or failure. Stable code, checkpoint
paths, vector files, and package versions must be retained while the server runs.

## 8. Results and handoff report

Each campaign contains `freeze.json`, `server_manifest.json`, `schedule.json`, the
copied source, named/masked data, and configurations. Cell artifacts live under
`unmasked/runs/<run_id>/` and `masked/runs/<run_id>/`. Inspect `run.json`, `report.json`,
`requests/`, and `calls/`. Summaries are written to `analysis/summary.json` and CSVs.

Report completed and failed cells, complete iterations, scientific tool activity,
truncations, repairs, token use, timings, peak memory, and the paths to all artifacts.
The scientific summary includes precision, recall, scientific F1, exploration,
class-balanced evidence responsiveness, and focal-discovery recovery. Queued or
missing-report cells retain missing scientific values. Results are separated into
completed-only and finished-with-errors cohorts.

The main label-by-surprise interaction is:

`I = (focal_EU - focal_SU) - (focal_EM - focal_SM)`

Here each `focal` value is a recovery percentage, and EU/SU/EM/SM are the conditions
in section 2. Compute the interaction within each arm and report CAA minus control
in percentage points. Interpret it alongside absolute surprising-condition recovery,
expected-condition recovery, overall scientific accuracy, and technical reliability.
The current treatment contrast is descriptive; treatment-effect confidence intervals
remain future analysis work. Existing uncertainty estimates are conditional on this
single frozen dataset pair. Broader task coverage and reviewed development contrasts
are needed to assess generalization.
