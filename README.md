# onc-co-scientist

Initial pipeline for the **Oncology Co-Scientist Benchmark** (Aims 1.1 and 1.2 of the grant "Do Large Language Models Entrench Biomedical Scientific Paradigms? A Study in Cancer Research").

This repository provides:

1. **Aim 1.1 — Synthetic dataset generator.** Produces large patient-level oncology datasets (default 50,000 patients, so statistical power is not the bottleneck) in which a single multi-feature **buried finding** is embedded — a treatment that is exceptionally active only inside the conjunction of 3-4 baseline features. The buried finding is recoverable by a flexible analysis but does not match a textbook treatment-biomarker paradigm. By default the generator produces one bundle per supported cancer type — **NSCLC, CRC, breast, prostate, and AML** — each in its own subfolder of `--out`; pass `--cancer-types nsclc,crc` (etc.) to restrict to a subset. Each cancer-type bundle is materialized in two parallel forms:
   - `named/` — real clinical column names (`treatment_pembrolizumab`, `egfr_mutation`, …),
   - `anonymized/` — every non-outcome non-id column renamed to `feature_NNN` via a seeded shuffle, so an agent has no semantic priors to anchor on. Outcome columns retain their real names.

   The legacy paradigm-mix mechanic (`concordant` / `discordant` / `hidden_novel` association catalogs and counters) remains in the codebase for legacy and comparison runs but is dialled to zero by default.
2. **Aim 1.2 — Benchmark task specification and scoring.** Emits a task brief framed as a generic commercial-EHR-vendor data-mining task (no telegraphing of the buried target) that any external agentic harness (Claude Code, Codex, a custom ReAct loop, etc.) can execute, and scores the transcript the harness returns against the ground-truth manifest. Buried findings are tagged `hidden_novel` for scoring continuity, so `mean_iterations_hidden_novel` becomes the primary metric for this evaluator.

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
# 1. Generate a synthetic dataset per supported cancer type. With the default
#    --cancer-types=all and --variant=both, each cancer type's bundle is
#    written under <out>/<cancer_type>/, with named and anonymized twins
#    (same rows, same buried finding, only column names differ):
#      ../data/ds001/nsclc/named/manifest.json               (ground truth — never shown to the agent)
#      ../data/ds001/nsclc/named/public/dataset.parquet      (agent-safe; real column names)
#      ../data/ds001/nsclc/named/public/dataset_description.md
#      ../data/ds001/nsclc/anonymized/manifest.json
#      ../data/ds001/nsclc/anonymized/column_mapping.json    (real -> feature_NNN map)
#      ../data/ds001/nsclc/anonymized/public/dataset.parquet (agent-safe; opaque names)
#      ../data/ds001/nsclc/anonymized/public/dataset_description.md
#      ../data/ds001/crc/...                                 (same shape, CRC bundle)
#      ../data/ds001/breast/...                              (same shape, breast bundle)
#      ../data/ds001/prostate/...                            (same shape, prostate bundle)
#      ../data/ds001/aml/...                                 (same shape, AML bundle)
ocs synth generate \
    --config configs/synthetic.example.yaml \
    --out ../data/ds001 \
    --seed 0
# Restrict to a subset:
#   ocs synth generate ... --cancer-types nsclc,crc
# Pass --variant named or --variant anonymized to write a single bundle
# directly into each cancer-type folder instead of the paired named/+anonymized/
# layout.

# 2. Build a harness-agnostic task brief that excludes the ground truth, for
#    one cancer type's bundle at a time. --dataset must point at one bundle
#    root (the directory containing manifest.json and public/), NOT at the
#    public/ subfolder. Pick whichever twin the eval calls for — for an
#    anchoring-free run, use anonymized/. --out is the self-contained task
#    directory you will hand to the agent.
ocs harness build-task \
    --dataset ../data/ds001/nsclc/anonymized \
    --max-iterations 5 \
    --out ../data/ds001/nsclc/anonymized/task

# 3. Hand the task directory to your agentic harness of choice, and run the
#    agent with that directory as its working directory so the relative paths
#    in agent_instructions.md resolve. The agent must read
#    agent_instructions.md, iterate up to N times, and write a transcript.json
#    (plus an analysis_summary.txt) into that directory.

# 4. Score the transcript against the same bundle's manifest.
ocs score run \
    --dataset ../data/ds001/nsclc/anonymized \
    --transcript ../data/ds001/nsclc/anonymized/task/transcript.json \
    --out ../data/ds001/nsclc/anonymized/score
```

For the default buried-finding configuration the primary metric is
**`mean_iterations_hidden_novel`** — the mean iteration at which the agent
first surfaces the buried multi-feature association (or the configured penalty
if it never does). The legacy paradigm-mix metrics are also computed when
those associations are turned on:

- **Metric 1** — mean iterations to uncover paradigm-concordant associations (lower = better general analytic performance).
- **Metric 2** — mean iterations to uncover paradigm-discordant associations (lower = better).
- **Metric 3** — `(Metric 2) - (Metric 1)` at the pipeline (inter-dataset mean) level (lower = less anchoring on current paradigms).

## Layout

See `docs/` (forthcoming) and docstrings. Top-level package: `src/onc_co_scientist/`.

## License

MIT.
