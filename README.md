# onc-co-scientist

Initial pipeline for the **Oncology Co-Scientist Benchmark** (Aims 1.1 and 1.2 of the grant *"Do Large Language Models Entrench Biomedical Scientific Paradigms? A Study in Cancer Research"*).

The benchmark asks: when an agentic harness analyzes a synthetic patient cohort that contains a deliberately buried multi-feature treatment effect, how often does it surface novel hypotheses, and at which iteration does it uncover the buried finding?

## What's in the box

- **Synthetic dataset generator (Aim 1.1).** Large patient-level oncology cohorts (50,000 patients by default). One bundle per supported cancer type — **NSCLC, CRC, breast, prostate, AML** — each with a single buried multi-feature finding (a treatment exceptional only inside a 3–4 feature conjunction). Each cancer-type bundle ships in two parallel forms:
  - `named/` — real clinical column names.
  - `anonymized/` — non-outcome columns renamed to `feature_NNN`.
- **Harness-agnostic task builder (Aim 1.2).** Emits a generic data-mining brief that any external agent (Claude Code, Codex, custom ReAct, …) can execute against a parquet file.
- **LLM-judged scorer (Aim 1.2).** Two metrics per `(harness, dataset, replicate)`:
  - **Novelty %** — fraction of harness-proposed hypotheses an LLM judge marks as going beyond established oncology paradigm consensus.
  - **Buried discovery iteration** — earliest iteration the pipeline both proposes and tests (with a direction-correct significant analysis) a hypothesis matching the buried finding. Falls back to `max_iterations` if never uncovered.

  Per-bundle scores are reported as mean ± SD across replicates; the pipeline-level figure is the unweighted mean of bundle means. Anonymized bundles are excluded from scoring (the LLM judge can't reason about `feature_NNN` columns).

## Scope

The repo orchestrates synth → task brief → harness invocation → score, but the harness binary itself is external (`claude`, `codex`, `opencode`, `droid`, `pi`, or any ollama-launchable wrapper). Subsequent aims (paradigm-stratified probe set, fine-tuning datasets, model-panel sweep, LoRA intervention, pre-1985 foundation model) are out of scope here.

## Install

```bash
uv pip install -e ".[dev]"
```

Optional extras: `synthetic` (upstream causal-inference generator, heavy ML deps) and `providers` (LLM provider SDKs).

## Quickstart

The full pipeline is wrapped in a single script:

```bash
scripts/run_all.sh
```

That runs synth → tasks → harness → score with sensible defaults. Override anything via environment variables:

```bash
OUT=../data/ds001 \
HARNESS=claude \
REPLICATES=5 \
JOBS=4 \
JUDGE=codex-cli \
scripts/run_all.sh
```

| Variable          | Default                            | Meaning                                                     |
| ----------------- | ---------------------------------- | ----------------------------------------------------------- |
| `CONFIG`          | `configs/synthetic.example.yaml`   | Generator config YAML                                       |
| `OUT`             | `../data/ds001`                    | Output root (datasets, tasks, scores)                       |
| `SEED`            | `0`                                | Generator seed                                              |
| `CANCER_TYPES`    | `all`                              | `all` or comma list (`nsclc,crc`)                           |
| `MAX_ITERATIONS`  | `10`                               | Iteration cap baked into the task brief                     |
| `HARNESS`         | `claude`                           | First arg to `scripts/run_harness.sh` (any supported spec)  |
| `JOBS`            | `4`                                | Bundles run in parallel                                     |
| `REPLICATES`      | `5`                                | Replicate runs per bundle (idempotent top-up)               |
| `PYTHON_ENV`      | `.venv`                            | Python env prepended to PATH per harness invocation         |
| `JUDGE`           | `anthropic-vertex`                 | Scoring judge backend (`anthropic-vertex`, `claude-cli`, `codex-cli`, or `stub`) |
| `JUDGE_CLI`       | `auto`                             | CLI binary for `claude-cli`/`codex-cli` judges (`auto`, `claude`, `codex`, or a path) |
| `JUDGE_MODEL`     | unset                              | Optional model id for `anthropic-vertex` or `codex-cli`     |

## Running steps individually

The same four commands `run_all.sh` invokes:

### 1. Generate synthetic datasets

```bash
ocs synth generate \
    --config configs/synthetic.example.yaml \
    --out ../data/ds001 \
    --seed 0
```

Per cancer type, this writes:

```
../data/ds001/<cancer_type>/
├── named/
│   ├── manifest.json                    # ground truth — never shown to the agent
│   └── public/
│       ├── dataset.parquet              # agent-safe; real column names
│       └── dataset_description.md
└── anonymized/
    ├── manifest.json
    ├── column_mapping.json              # real → feature_NNN map
    └── public/
        ├── dataset.parquet              # agent-safe; opaque names
        └── dataset_description.md
```

Use `--cancer-types nsclc,crc` (etc.) to restrict the run, or `--variant named` / `--variant anonymized` to write a single twin instead of both.

### 2. Build harness task bundles

```bash
ocs harness build-task \
    --dataset ../data/ds001 \
    --max-iterations 10 \
    --out ../data/ds001/tasks
```

Writes `tasks/<cancer_type>/<variant>/agent_instructions.md` (plus dataset, schema, and example) for every bundle, mirroring the synth tree. Point `--dataset` at a single bundle directory to build a one-off task instead.

### 3. Run an agentic harness

```bash
scripts/run_harness.sh claude ../data/ds001/tasks \
    --python-env .venv \
    --jobs 4 \
    --replicates 5
```

Use the Codex CLI profile the same way for hypothesis-generation runs:

```bash
scripts/run_harness.sh codex ../data/ds001/tasks \
    --python-env .venv \
    --jobs 4 \
    --replicates 5
```

The script `cd`s into each `tasks/<ct>/<variant>/` before launching, so the harness inherits that as its working directory and cannot see the synth bundle's manifest one level up. Per-replicate outputs land under `tasks/<ct>/<variant>/runs/run_NNN/{transcript.json,analysis_summary.txt,harness.log}`. Re-invoking with the same `--replicates` tops up missing runs idempotently.

Built-in profiles: `claude`, `codex`, `opencode`, `droid`, `pi`. Local-model wrappers also work (the script auto-inserts the `--` separator that ollama needs):

```bash
scripts/run_harness.sh "ollama launch claude --model qwen3.6:27b --yes" \
    ../data/ds001/tasks --jobs 2 --replicates 5
```

### 4. Score

```bash
ocs score batch \
    --synth-root ../data/ds001 \
    --tasks-root ../data/ds001/tasks \
    --out ../data/ds001/score \
    --judge claude-cli
```

The default `claude-cli` judge shells out to `claude --dangerously-skip-permissions -p`, using whatever Claude Code auth is already on the host (no API key plumbing). Judge calls are cached on disk under `~/.cache/onc-co-scientist/judge/`; pass `--no-judge-cache` to force every call to hit the LLM.

You can also use `--judge codex-cli` to score through the OpenAI Codex CLI.
That backend shells out to `codex exec` using existing Codex CLI auth.

```bash
ocs score batch \
    --synth-root ../data/ds001 \
    --tasks-root ../data/ds001/tasks \
    --out ../data/ds001/score \
    --judge codex-cli \
    --judge-model gpt-5.4
```

For one-off scoring of a single transcript:

```bash
ocs score run \
    --dataset ../data/ds001/nsclc/named \
    --transcript ../data/ds001/tasks/nsclc/named/runs/run_001/transcript.json \
    --out ../data/ds001/nsclc/named/score \
    --judge codex-cli
```

## Prototype: CAA paradigm-bias vectors

Aim 2.2 now has a prototype under `ocs caa`. It derives residual-stream
contrastive activation addition vectors from paired prompts, subtracts the
oncology-knowledge component from the paradigm-adherence vector, and can run a
single steered generation with either additive steering or runtime projection
ablation.

### Environment

The CAA commands need a Transformers/PyTorch environment with enough VRAM for
the open-weights model. On this workstation, use the existing environment:

```bash
export PYTHONPATH="$PWD/src"
export PY=/home/kenneth_kehl/thisenv/bin/python
```

If starting from a fresh environment instead, install the optional ML stack:

```bash
uv pip install -e ".[interventions]"
```

### Download Gemma 4 31B

The public full-precision Hugging Face model ID is hyphenated:
`google/gemma-4-31B-it`. Download it into the local cache under `~/models`:

```bash
$PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/gemma-4-31B-it', cache_dir='/home/kenneth_kehl/models'))"
```

The pre-downloaded NVFP4 cache under
`~/models/models--nvidia--Gemma-4-31B-IT-NVFP4` contains quantized weights but
is not the full Google bfloat16 snapshot used for activation capture.

### Derive Vectors

Bootstrap a small synthetic pair set. These are smoke-test pairs, not the final
grant-grade named-vs-anonymized trace corpus:

```bash
$PY -m onc_co_scientist.cli caa write-pairs \
    --out ../data/caa/bootstrap_pairs.jsonl \
    --overwrite
```

Derive the CAA vectors. For a fast smoke test on a 96 GB GPU, use one middle
layer. For a broader sweep, replace `--layers 30` with a comma list such as
`20,30,40,50` or `last:8`.

```bash
$PY -m onc_co_scientist.cli caa derive \
    --pairs ../data/caa/bootstrap_pairs.jsonl \
    --out ../data/caa/gemma4_31b_caa_layer30.npz \
    --model google/gemma-4-31B-it \
    --cache-dir /home/kenneth_kehl/models \
    --layers 30 \
    --position last \
    --dtype bfloat16 \
    --local-files-only
```

Inspect the artifact:

```bash
$PY -m onc_co_scientist.cli caa describe \
    --vector-file ../data/caa/gemma4_31b_caa_layer30.npz
```

The artifact contains:

- `paradigm_adherence`: positive named-oncology-context minus anonymized/data-first context.
- `oncology_knowledge`: cancer-relevant abstract context minus cancer-irrelevant biomedical context.
- `paradigm_orthogonalized`: the component of `paradigm_adherence` orthogonal to `oncology_knowledge`.

### Try Steering

Negative additive scale pushes against the positive paradigm-adherence
direction:

```bash
$PY -m onc_co_scientist.cli caa generate \
    --vector-file ../data/caa/gemma4_31b_caa_layer30.npz \
    --model google/gemma-4-31B-it \
    --cache-dir /home/kenneth_kehl/models \
    --local-files-only \
    --dtype bfloat16 \
    --concept paradigm_orthogonalized \
    --mode add \
    --scale -0.1 \
    --max-new-tokens 128 \
    --out ../data/caa/steered_add.txt \
    --prompt "Propose oncology hypotheses for a dataset where accepted biomarkers may be misleading. Be open to high-order feature conjunctions."
```

Runtime projection ablation is a non-destructive analogue of abliteration:

```bash
$PY -m onc_co_scientist.cli caa generate \
    --vector-file ../data/caa/gemma4_31b_caa_layer30.npz \
    --model google/gemma-4-31B-it \
    --cache-dir /home/kenneth_kehl/models \
    --local-files-only \
    --dtype bfloat16 \
    --concept paradigm_orthogonalized \
    --mode ablate \
    --scale 1.0 \
    --max-new-tokens 128 \
    --out ../data/caa/steered_ablate.txt \
    --prompt "Propose oncology hypotheses for a dataset where accepted biomarkers may be misleading. Be open to high-order feature conjunctions."
```

The current implementation modifies activations at inference time only; it does
not write abliterated model weights.

## Layout

Top-level package: `src/onc_co_scientist/`. See docstrings for module-level detail.

## License

MIT.
