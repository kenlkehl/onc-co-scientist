# CAA under the current key-discovery protocol

This is a new preliminary BF16 experiment, not a rescore or continuation of the
May Gemma gate. The historical vectors and results remain unchanged.

## Scientific comparison

The driver `scripts/expected_surprising/caa_discovery.py` reuses the September 10
recommended-sampling input packages and scientific controller. The four conditions
are expected/unmasked, expected/masked, surprising/unmasked, and surprising/masked.
Before preparation, it verifies the identical private pair and exact row-preserving
mask transformation, including nominal category levels. The private keys and mask
stay controller-side; participant prompts and analysis/validation tools are unchanged.

The default is the persistent workflow, 25 iterations, 10 replicates per condition,
the existing 125,000-token per-call limit, two repair attempts, 120,000-character
persistent history, and the existing scientific-score retention policy. Select
sequential or deliberative explicitly with repeated `--workflow` arguments.
One control plus one CAA arm gives 80 runs and 8,000 primary stage calls. Adding the
random-vector control gives 120 runs. The driver runs serially, shuffling view,
version and arm within each replicate, and does not reroll terminal failures.

Use fresh, unsteered **BF16 controls on the same server**. The earlier FP8/vLLM
Gemma results are historical reference only. This Transformers server does not
implement constrained JSON decoding: both BF16 arms use identical JSON prompts,
parsing, repairs and validation. This differs from the recent vLLM JSON constraint
and must remain explicit in any comparison. Thinking is enabled initially, and
disabled only on the final retry after two truncations, as in the current adapter.
The server preserves Gemma's multiple stop IDs, reports actual generated tokens
(including thoughts and special tokens), and marks token-limit truncations.

## Candidate direction, not a validated anchoring axis

`pairs` writes 24 constructed appraisal response contrasts in six synthetic
non-NSCLC contexts, plus two explicitly labeled legacy knowledge fixtures. Eight
contrasts in two other contexts are reserved for development. Positive and negative
sides share exactly the same question, prior and numerical evidence; only the
assistant's appraisal changes. Four cases cover supported-surprising,
excluded-expected and ambiguous evidence in both directions. The vector is positive
(anchored) minus negative (evidence-responsive), so negative addition is the intended
candidate intervention.

These are templated, unreviewed development fixtures, **not observed model traces or
validated biomedical claims**. Response wording/length and generic instruction-following
remain confounds. Last-token pooling is over the complete assistant response, with
no generation prompt and thinking disabled. Orthogonalization against two legacy
knowledge pairs does not establish preservation of oncology competence.

The smoke driver derives layers 20/30/40 and adds a seeded isotropic random direction
with the candidate's norm at each layer. Layer 40 / scale -0.05 is an **unvalidated
engineering choice**, not an optimum. Select layer, sign and scale on separate
development tasks and freeze the rationale before evaluating the NSCLC pair. Future
work still includes broader reviewed contrast examples, held-out quantitative
knowledge/formatting gates, layer/scale tuning, and a treatment-effect uncertainty
analysis. A short engineering smoke cannot establish mitigation.

## Local GPU smoke

Use the existing `/home/klkehl/thisenv/bin/python` environment. New environments need
the project `interventions`, `vllm-openai`, and scientific dependencies; Gemma 4 also
requires a Transformers release with native Gemma 4 support (the local environment
has 5.14.1). The broad historical package minimum is not a Gemma compatibility promise.

From the repository root:

```bash
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DEACTIVATE_ASYNC_LOAD=1 OMP_NUM_THREADS=4
CAA_MODEL=/data1/ken/models/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343
/home/klkehl/thisenv/bin/python scripts/expected_surprising/caa_local_smoke.py \
  --model "$CAA_MODEL" --out outputs/caa_discovery/local_smoke
```

This refuses to overwrite output, loads BF16 across the two GPUs, rejects CPU/disk
offload, derives the 26 training pairs, and tries three generations capped at 128
tokens. It saves vectors, contrast corpora, aliases, serving manifest, memory usage,
timings, raw responses, termination reasons and token counts. It does not start a
discovery campaign. `HF_DEACTIVATE_ASYNC_LOAD=1` bypasses an asynchronous weight-loader
stall observed locally; it does not alter inference mathematics. Synchronous loading
also stalled on SSHFS file pages, so this flag alone is not a demonstrated fix.
If weights live on a network filesystem, `--stage-local /tmp` makes a new temporary
local copy and verifies content-addressed SHA-256 blob names. This needs about
63 GB of disk and can be slow over SSHFS. The staged path is saved in `smoke.json`.
Use that path for subsequent serving after a successful copy; staging does not
change model weights.

## Freeze, execute, summarize

Start the local endpoint from a stable checkout and do not edit serving code while
loading/running it. Use the vector and alias files from a successfully completed smoke
or a separately reviewed derivation. Keep the endpoint bound to loopback.

```bash
/home/klkehl/thisenv/bin/python -m onc_co_scientist.cli caa serve \
  --model "$CAA_MODEL" --dtype bfloat16 --device-map balanced \
  --vector-file outputs/caa_discovery/local_smoke/vectors.npz \
  --aliases-file outputs/caa_discovery/local_smoke/aliases.json \
  --cache-implementation dynamic --enable-thinking --port 8765
```

In another terminal, save the **live** `/v1/caa` JSON with an HTTP client, then:

```bash
/home/klkehl/thisenv/bin/python scripts/expected_surprising/caa_discovery.py prepare \
  --out data/caa_discovery/pilot \
  --server-manifest /absolute/path/to/live-server-manifest.json \
  --arm gemma4-control --arm gemma4-caa \
  --replicates 1 --smoke \
  --rationale 'Unvalidated layer-40/-0.05 engineering pilot; no efficacy claim'
/home/klkehl/thisenv/bin/python scripts/expected_surprising/caa_discovery.py run \
  data/caa_discovery/pilot --limit 1
/home/klkehl/thisenv/bin/python scripts/expected_surprising/caa_discovery.py summarize \
  data/caa_discovery/pilot
```

`--smoke` makes a separate six-iteration, 2,048-token engineering protocol. Omitting
it preserves the current full protocol. A small smoke does **not** demonstrate that
long-context 125k-token generation fits; dynamic KV caches can still exhaust VRAM.
`--limit` bounds newly executed cells; rerunning resumes interrupted cells and skips
terminal ones. Review the server and scientific pilot before scheduling a full matrix.

Preparation copies input packages and the controller/provider/analysis source, freezes
their hashes, dependencies, sampling and selected arms, and records a joint schedule.
Execution automatically uses the copied source and verifies the freeze. Each request
and response checks the server fingerprint; it includes checkpoint metadata, vectors,
serving options and implementation. Weight identity records resolved cache blob
names, sizes and modification times, not a fresh full-weight hash at each startup;
the optional staging path does compute full content hashes. Use a freshly fetched serving manifest after any
server code/configuration change. Manifests and private task assets are local experiment
artifacts, not participant context.

## Analysis and interpretation

The summary reuses `build_presentation_results.py`: precision, recall, scientific F1*,
exploration E, class-balanced responsiveness B, focal recovery, and the current
descriptive C/S scores and label-by-surprise interaction:

`I = (focal_EU - focal_SU) - (focal_EM - focal_SM)` (percentage points).

It reports completed-only and finished-with-errors cohorts, technical failures,
usage and missing reports; queued runs never become zero recoveries. Existing
within-arm confidence intervals are reused. CAA-minus-control interaction differences
are descriptive only in this first implementation (no treatment-effect CI yet).
Reduced I is not sufficient evidence of success: inspect absolute surprising recovery,
expected-condition harm, evidence responsiveness, scientific F1 and failure rates.
Intervals condition on the one frozen NSCLC pair, not a population of scientific tasks.
