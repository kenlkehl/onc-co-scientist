# Claude Opus 5 through Google Vertex AI

The repository now supports Claude in both experiment families:

- **Named/masked recovery:** the shared structured runner accepts
  `--backend anthropic-vertex`, with sandboxed Python and validated iteration submissions.
- **Expected/surprising:** `kind: anthropic_vertex` runs through the shared scientific
  ledger in persistent, sequential, and deliberative workflows. The preparation
  script below also creates named and masked twins of those paired datasets.

The model ID is `claude-opus-5`. Google and Anthropic document the global endpoint;
select a supported regional endpoint if your project requires one. References:
[Google model card](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/opus-5),
[Anthropic Vertex setup](https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai),
[Opus 5 migration](https://platform.claude.com/docs/en/models/opus-5/migration-guide).

## Environment and authentication

From the repository root, use a working Python 3.12+ environment. For a separate one:

```bash
uv venv --python 3.12 .venv-claude
uv pip install --python .venv-claude/bin/python -e '.[anthropic-vertex,analysis,dev]'
source .venv-claude/bin/activate

gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
export GOOGLE_CLOUD_LOCATION=global
```

Enable the Vertex AI API and Claude access in your Google project's Model Garden;
the authenticated identity needs permission to invoke that model. On a compute
service with an attached service account, use its ADC identity instead of interactive
login. Credentials stay outside manifests and logs.

Explicit project/region settings win over environment variables. Project fallback
order is `ANTHROPIC_VERTEX_PROJECT_ID`, `GOOGLE_CLOUD_PROJECT`, then SDK ADC discovery.
Region fallback is `CLOUD_ML_REGION`, `GOOGLE_CLOUD_LOCATION`, then `global`.
The structured runner names the region option `--location`; provider YAML uses `region`.

A small **live, billable** connectivity check, when ready:

```bash
python - <<'PY'
from onc_co_scientist.providers import ChatMessage, get_provider
p = get_provider({
    "kind": "anthropic_vertex", "model_id": "claude-opus-5",
    "region": "global", "reasoning_effort": "low", "timeout_s": 120,
})
r = p.chat([ChatMessage("user", "Reply with only OK.")], max_tokens=4096)
print(r.text)
print(r.raw["usage"])
PY
```

## Named/masked recovery

Linux bubblewrap and functioning user namespaces are required for the existing
Python sandbox. Preparation makes no model requests. These example setup counts
produce one named and one masked job for each of the five clinical datasets:

```bash
python experiments/aim1_recovery/prepare.py \
  --out data/claude_named_masked_setup --python "$(command -v python)" \
  --backend anthropic-vertex --model claude-opus-5 \
  --reasoning-effort medium --service-tier unspecified \
  --clinical-repeats 1 --depmap-repeats 0

python experiments/aim1_recovery/run_batch.py \
  --plan data/claude_named_masked_setup/plan.json \
  --backend anthropic-vertex --project-id "$GOOGLE_CLOUD_PROJECT" --location global \
  --jobs 1 --max-tokens-per-call 32768 --request-timeout 1800 --python-timeout 300

python experiments/aim1_recovery/score.py \
  --plan data/claude_named_masked_setup/plan.json \
  --out data/claude_named_masked_setup/scoring
```

Use a fresh output directory and your intended replicate counts for formal runs.
Keep setup jobs separate from the formal comparison. The frozen plan retains the
model, backend, effort, scorer, public inputs, and private evaluation split.
Transcripts retain native responses, signed thinking blocks, executed Python,
tool results, and output-token accounting. Completed jobs are skipped; partial
named/masked runs require review/finalization rather than silent replacement.
See [the protocol guide](NAMED_MASKED_GUIDE.md) for scoring and budget definitions.

A single already-prepared workspace can also be run with
`python scripts/run_structured_agent.py run --workspace PATH --provider anthropic-vertex
--model claude-opus-5 --project-id PROJECT --location global --reasoning-effort medium`.

## Expected/surprising, crossed with named/masked

Start with a **current reviewed ledger package** containing `public/` and `private/`.
Use the same source package as the comparison models; this command copies existing
rows and derives the masked twin without regenerating the scientific targets.
Select a pair with repeated `--pair-id` flags, or omit them to include all pairs.

```bash
python scripts/expected_surprising/prepare_claude_grid.py \
  --source /absolute/path/to/reviewed/input_data \
  --out data/claude_expected_surprising_setup \
  --project-id "$GOOGLE_CLOUD_PROJECT" --region global \
  --iterations 6 --replicates 1 --max-parallel 2 \
  --effort medium --max-tokens 32768
```

This is an offline preparation/dry run. `grid.json` lists the generated configs and
run counts. Per input pair, defaults yield **12 runs**: two semantic conditions ×
two expected/surprising versions × three workflows × one replicate. Both semantic
twins use the same schedule seed, replicate IDs, and scientific policies.

Launch explicitly when ready; these commands call Vertex:

```bash
ocs harness run-experiment \
  --config data/claude_expected_surprising_setup/named/config.yaml --resume
ocs harness run-experiment \
  --config data/claude_expected_surprising_setup/masked/config.yaml --resume
```

For formal experiments, prepare a new directory with `--iterations 25 --replicates 10`
and the intended effort/output allowance. This produces 120 runs per input pair.
Each condition has its own results directory and existing matrix reports. The
coordinator retains stage requests, usage, errors, scientific scores, and cached
responses; `--resume` reuses compatible completed work. Do not edit configs or
provider implementations mid-experiment without the existing provenance checks.
For a single semantic condition, adapt
[the example manifest](../configs/expected_surprising.claude_vertex.yaml) and run
`ocs harness validate-experiment --config PATH` before launch.

## Model controls and limits

Opus 5 uses adaptive thinking by default. `reasoning_effort` maps to
`output_config.effort` (`low`, `medium`, `high`, `xhigh`, `max`). Temperature is omitted
for Opus 5 because its API rejects sampling parameters, even though the common
scientific protocol supplies temperature zero. Effective requests are retained in
structured-runner logs. No automatic model fallback or effort reduction is enabled.

`max_tokens` includes thinking and visible output. Streaming aggregates complete
responses; all native assistant blocks are replayed unchanged during tool use.
Truncation, refusals, missing usage, and unexpected model substitutions are failures,
not accepted scientific responses. Expected/surprising repair records keep usage
from failed generations. SDK transport retries are bounded by `max_retries` (default
2); these are distinct from scientific stage repairs. Usage measures returned
responses, not a billing ledger for requests interrupted before a response arrives.

OpenAI service tiers are rejected; pass `--service-tier unspecified` at named/masked
preparation. Expected/surprising uses the existing agent-call and per-call output
budgets; it does not enforce an aggregate dollar or token cap. Before launching a
large grid, verify project access and run the separate setup grid with the intended
model and output allowance.
