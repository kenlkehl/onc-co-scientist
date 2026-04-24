# onc-open-mindedness

Initial pipeline for the **Oncology Scientific Open-Mindedness Benchmark** (Aims 1.1 and 1.2 of the grant "Do Large Language Models Entrench Biomedical Scientific Paradigms? A Study in Cancer Research").

This repository provides:

1. **Aim 1.1 — Synthetic dataset generator.** Produces patient-level oncology datasets in which every embedded association is tagged with a `paradigm_class`:
   - `concordant` — consistent with currently accepted oncology paradigms,
   - `discordant` — direction of effect is inverted or unexpected relative to current consensus,
   - `hidden_novel` — a subgroup in which a broadly ineffective therapy is exceptionally active (a signal of an undiscovered biomarker).
2. **Aim 1.2 — Benchmark task specification and scoring.** Emits a task brief that any external agentic harness (Claude Code, Codex, a custom ReAct loop, etc.) can execute, and scores the transcript the harness returns against the ground-truth manifest to compute the paradigm-adherence metrics defined in the grant.

## What this repo deliberately does **not** do

- **It does not run the agent.** You bring your own harness — Claude Code, Codex, or any loop capable of reading a markdown brief, executing Python against a parquet file, and emitting a `transcript.json`.
- It does not yet implement Aim 1.3 (paradigm-stratified probe set), Aim 1.4 (fine-tuning dataset), Aim 2.1 (model-panel sweep), Aim 2.2 (LoRA intervention), or Aim 3 (pre-1985 foundation model). Those land in subsequent iterations.

## Install

```bash
pip install -e ".[dev]"
# With the upstream onc-causal-inference generator (pulls heavy ML deps):
pip install -e ".[dev,synthetic]"
# With provider SDKs for LLM-assisted hypothesis matching:
pip install -e ".[dev,providers]"
```

## Quickstart

```bash
# 1. Generate a synthetic dataset bundle (ground-truth manifest stays in the bundle).
oom synth generate \
    --config configs/synthetic.example.yaml \
    --out ../data/ds001 \
    --seed 0

# 2. Build a harness-agnostic task brief that excludes the ground truth.
oom harness build-task \
    --dataset ../data/ds001 \
    --max-iterations 5 \
    --out data/ds001/task

# 3. Hand ../data/ds001/task/ to your agentic harness of choice.
#    It must read agent_instructions.md, iterate up to N times, and write
#    a transcript.json conforming to the schema in harness/templates.

# 4. Score the transcript.
oom score run \
    --dataset ../data/ds001 \
    --transcript ../data/ds001/task/transcript.json \
    --out data/ds001/score
```

The scoring output includes the grant's three primary metrics:

- **Metric 1** — mean iterations to uncover paradigm-concordant associations (lower = better general analytic performance).
- **Metric 2** — mean iterations to uncover paradigm-discordant associations (lower = better).
- **Metric 3** — `(Metric 2) - (Metric 1)` at the pipeline (inter-dataset mean) level (lower = less anchoring on current paradigms).

## Layout

See `docs/` (forthcoming) and docstrings. Top-level package: `src/onc_open_mindedness/`.

## License

MIT.
