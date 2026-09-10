# Native external co-scientists

The `external-native-1` runner evaluates an external system without running it
through the repository's persistent/sequential/deliberative scientific stages.
Biomni is the first engine. Its native A1 graph, tool retriever, Python execution,
installed resources and conversation remain active throughout a run.

The initial configuration is `configs/biomni.nsclc.yaml`: both frozen versions of
`es-v2-nsclc_clinical-42000`, ten replicates per version, 25 reporting rounds,
Qwen3.8-27B-NVFP4 with thinking enabled, xhigh effort and an adaptive ceiling of 125,000 completion tokens.
The pair name's 42000 is its seed; each input contains 50,000 observations.

## Start and verify

Use a Python >=3.12 environment for onc-co-scientist. Biomni executes separately
in its existing Python 3.11 environment. Install the repo's declared dependencies:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -e '.[dev]'
```

Start vLLM with `scripts/external/start_vllm.sh` when no server owns port 8000.
The launcher enables the Qwen reasoning parser and a 262,144-token context. Do
not restart a shared server with active requests. No other model is selected
as a fallback. The original `~/biomni/start-vllm.sh` is not modified.

```bash
.venv/bin/ocs expected-surprising external-preflight configs/biomni.nsclc.yaml --offline
.venv/bin/ocs expected-surprising external-preflight configs/biomni.nsclc.yaml
.venv/bin/ocs expected-surprising external-smoke configs/biomni.nsclc.yaml --full-length
.venv/bin/ocs expected-surprising external-run configs/biomni.nsclc.yaml --resume
```

Run the last command in a persistent terminal for a long campaign. One run is
active at a time. Limits are 900 total model requests per run and four hours per
request, without a whole-run time limit. Each completion may consume up to
125,000 tokens INCLUDING reasoning. Before generation the broker checks the
rendered prompt's token count and sets the cap to
`min(125000, 262144 - prompt_tokens - 256)`. The 256-token guard is reserved;
if fewer than 1,024 completion tokens remain, the run stops. Conversation history
is preserved. Truncated completions still fail before code can execute. Each request
audits its measured prompt length, configured ceiling, and effective cap. The
`fixed` policy remains available for reproducing the earlier protocol.
Biomni's default temperature 0.7 is explicit. Thinking effort, response size and
helper routing override upstream defaults; the native research graph remains.

A successful full-length gate is required for formal execution with the adaptive
policy. `--full-length` uses fresh 50,000-row datasets for both versions, with
25 reporting rounds and the full native research task. Its results are excluded.
The shorter default smoke uses 2,000 rows and six rounds for debugging; it cannot
unlock the adaptive campaign. To run the full-length gate and automatically start
formal execution only if it passes, use:

```bash
.venv/bin/ocs expected-surprising external-run configs/biomni.nsclc.yaml \
  --full-length-smoke --resume
```

The revised campaign writes to `data/biomni_nsclc_qwen38_xhigh_adaptive`; the
interrupted fixed-cap campaign remains in its original directory. A successful
pilot does not guarantee that every replicate will fit: native history can still
fill the context window. Code, input bytes, configuration, native source,
environment and data-lake fingerprints must match the gate. If any change, run
another smoke and use a fresh formal output root when existing results exist.

## Benchmark exchanges

`ExternalRunner` is the engine interface. `ExchangeGateway` and `Exchange` define
the framework-neutral public protocol. A runner receives a public-only task,
workspace, broker address and ephemeral credential. Private PairSpec objects,
reference effects and validation generators stay in the evaluator process.

The native scientist calls `benchmark_exchange` with `request_id`, `round`,
`action` and `payload`. Identical requests are idempotent; reusing an ID for a
different request fails. Invalid operations roll back scientific state.

| Action | Purpose |
|---|---|
| `state` | Read the public ledger and outstanding evidence obligations. |
| `register` | Register comparison, initial expectation and assessment; obtain H references. |
| `analyze` | Request canonical discovery analyses of registered H references. |
| `assess` | Record accept/reject/unresolved and investigation status, citing R evidence. |
| `validate` | Request independent evidence after discovery appraisal. |
| `prepare_close` | Select future scheduled evidence, release due evidence and seal submissions. |
| `close` | Commit the round after new evidence and deadlines have been assessed. |

Registration, testing and appraisal can repeat in a native order. The benchmark
allows at most 12 canonical analyses per round, one new voluntary validation per
round and ten total. Cached evidence is reused. Scheduled releases occur at
5/12/20; explicit reassessment of the original claim is due two rounds later.
`prepare_close` prevents new research submissions until the next round, but
Biomni retains its own memory and code. These are reporting boundaries, not
researcher/critic roles. Stage labels in underlying ledger records are internal
compatibility record types and must not be counted as native agent calls.

Large public responses are saved under `scratch/benchmark_responses`, with the
file path and due assessments returned visibly to Biomni. This avoids its
upstream observation truncation hiding a long claim ledger.

## Scientific interpretation

The task contains the exact existing public **10% relative clinical-significance
guidance**. The evaluator's private cutoff remains unchanged. Final confirmation,
R, P, F1*, E, B, category balancing and missing-value rules use the same scoring
functions as the expected/surprising controller.

R and P are fractions. F1*, E and B are on 0–100 scales. No accepted claims means
P is unavailable. B is unavailable when the required evidence classes are not
represented. These values must not be replaced with zeros. The configured
failure policy still applies to primary discovery performance.

This arm is **Biomni native + benchmark exchanges + full resources**. Additional
native analyses and external knowledge are permitted, but E counts canonical
registered tests. Its horizontal axis is reporting rounds, not custom workflow
iterations or native graph steps. The system has different evidence access and
computational opportunity from the custom harness. Compare token totals alongside
scores and retain the resource and clock labels. Initial expectations are marked
as not blinded because native data inspection cannot be ruled out.

Ten paired repeats on this one fixed dataset support descriptive comparisons;
they do not establish variation across datasets. Native early stopping or a
technical failure is retained and reported, never silently replaced by a more
successful replicate.

## Resources and isolation

Bubblewrap mounts only public task files, the Biomni package, its scientific
Python environment, the installed data lake and per-run scratch. It has a
separate PID namespace and clean environment. Neither the repository, private
input tree, other runs nor credentials are mounted. Networking remains available
for research and the local broker. There is no unsandboxed fallback.

The local Biomni installation has the base scientific environment and the full
installed data lake, not the entire optional E1 tool collection. Tools requiring
missing specialized binaries or service credentials can fail; the native trace
retains those failures. All Biomni LLM factories, including helper-specific
choices, are overridden to the same audited local endpoint. No cloud LLM
credentials are passed to the worker.

## Outputs and resumption

Each run contains `report.json`, `run.json`, `exchanges.jsonl`, per-request
`llm/*.json`, worker logs, native conversation events, saved code/results and a
native checkpoint. Top-level `summary.json`, `summary.csv` and `summary.md`
contain the comparison metrics and usage. The full reports retain per-category
coverage curves, evidence classes, validation events and accepted-claim history.

Every broker request, including failed requests and retrieval/helper work, has
an audit record. Input/output tokens use server-reported usage; reasoning is a
subset of completion tokens and is not added a second time. Raw responses retain
reasoning for audit. The broker strips separate reasoning fields and reasoning
content blocks before returning responses to Biomni and before tokenizing or
forwarding assistant history. Reasoning tags mixed into final text fail closed.
The native conversation, including persisted checkpoints, therefore receives final
content and tool calls without separate reasoning traces. If a request has no
usage receipt, totals are unavailable and known subtotals remain available.
Dollar cost is unavailable without an explicit local compute-pricing model.
Native graph nodes, tool executions and reporting exchanges are not LLM requests.

The trusted process can replay canonical benchmark exchanges; it never replays
arbitrary agent code or deserializes agent-created Python checkpoints. Native
Python namespace deserialization happens only inside the same sandbox. Resume
requires a consistent conversation/namespace checkpoint and matching exchange
and request counts. An interruption inside code execution or between incompatible
checkpoints fails closed. Completed or terminal failed cells are reused on
`--resume`, rather than rerunning selectively failed experiments.

To compare native reports with copied outputs from an existing workflow run:

```bash
.venv/bin/ocs expected-surprising external-compare \
  data/biomni_nsclc_qwen38_xhigh_adaptive data/biomni_comparison \
  --baseline /path/to/existing/experiment
```

The comparison accepts `runs/*/report.json` or individual report files, checks
scoring-profile and dataset-byte compatibility, and writes JSON, CSV and Markdown.
Conditions stay separate by model, workflow, version, resources, clock and
completion policy. Mean
metrics include available-run counts; unavailable metrics remain unavailable.
`exploration_by_tokens` in each native report provides the common token axis in
addition to reporting-round coverage.

Checkpoint restoration also preserves NumPy/Python RNG state and working
directory. Closed file variables become explicitly closed references retaining
metadata; their concrete Python IO class identity is not preserved. They are
never reopened. Live handles and other unsupported objects make a checkpoint
non-resumable rather than being silently discarded. Final run artifacts have
checksums verified on reuse. Python network connection destinations are logged
in `scratch/network_events.jsonl`; this is not full HTTP packet capture.

The checkpoint serialization test additionally uses `cloudpickle`, installed in
the Biomni environment. Install it in the test environment to exercise that test:

```bash
uv pip install --python .venv/bin/python cloudpickle
.venv/bin/python -m pytest tests/test_external_native.py
```

## Masked follow-up

`configs/biomni.nsclc.masked.yaml` runs ten expected and ten surprising masked
replicates with the same local endpoint and adaptive token budget. The GCS
`masked_expected.parquet` and `masked_surprising.parquet` files were checked
against the deterministic masking transformation of the original 50,000-row
inputs, then copied byte-for-byte into the private/public benchmark package.
The private mapping transforms hypotheses, independent samples, confirmation,
and reference targets; validation seeds are derived in the original coordinates.
Neither mapping nor original predictor names are passed to Biomni. Full-length
excluded pilot rows are also masked before reaching the worker.

The follow-up runs from an isolated worktree while the named campaign retains
its frozen source. Once all 20 named cells are terminal and its process exits,
`python -m onc_co_scientist.external.followup` fast-forwards the original checkout,
runs both masked full-length pilots, then the 20 masked cells. It preserves failed
cells and stops if the pilot gate fails. Its status is in `followup_status.json`.

After both campaigns terminate, it verifies report and artifact checksums and
writes `biomni_report.md`, `biomni_report.json`, and comparison CSV/JSON/Markdown.
Named/masked conditions remain separate; their underlying source hashes must
match. All scientific sections, curves and available diagnostic metrics are kept
in JSON, with scalar means and available-run denominators in Markdown. Missing
metrics and unpriced local dollar cost remain null. The reports are uploaded to
`gs://kehl-lab-caia/onc-co-scientist-data/reports/biomni-named-masked-20260910/`.
No custom-harness result files were present in the supplied GCS directory.
