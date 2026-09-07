# Gemini through Vertex AI

Install `pip install -e '.[gemini-vertex]'` in the Python environment that will run
agents. Authenticate with Google Application Default Credentials (ADC). Set
`GOOGLE_CLOUD_PROJECT` to your own project and optionally set
`GOOGLE_CLOUD_LOCATION` (defaults to `global`). Explicit project/location options take
precedence. Project discovery also accepts the existing
`ANTHROPIC_VERTEX_PROJECT_ID`, Google's ADC project, or its quota project.
No Gemini API key, Codex login, vLLM server, or OpenAI SDK is used.

The default model is `gemini-3.8-flash`. Thinking effort can be `low`, `medium`, or
`high`; omit it for the model default. GPT effort levels such as `xhigh`/`max`,
OpenAI service tiers, and vLLM thinking/template/repetition-penalty overrides are
rejected. Thinking tokens count toward generated-token budgets. Cost is not
estimated; token/tool/time limits remain available, but a dollar budget alone
cannot enforce spend for this integration.

## Internal chat and LLM judging

```python
from onc_co_scientist.providers import ChatMessage, get_provider

provider = get_provider({
    "kind": "gemini_vertex",
    "model_id": "gemini-3.8-flash",
    "location": "global",
})
response = provider.chat([ChatMessage(role="user", content="Reply OK")])
```

Scoring commands accept `--judge gemini-vertex --judge-model gemini-3.8-flash`
where an LLM judge is selected. Deterministic recovery scoring remains unchanged.
See `configs/providers.example.yaml` for configuration fields.

## Persistent, sequential, deliberative, and federated harness experiments

Use this model entry in an experiment YAML (replace absolute paths with your
installed checkout and Python environment). All workflow modes share this adapter.

```yaml
models:
  - id: gemini-flash
    model_id: gemini-3.8-flash
    adapter: cli-json
    reasoning_effort: high
    command:
      - /absolute/path/to/venv/bin/python
      - /absolute/path/to/onc-co-scientist/scripts/gemini_cli_json_adapter.py
    extra_args:
      - --location
      - global
      - --analysis-python
      - /absolute/path/to/analysis-venv/bin/python
      - --interaction-mode
      - native-tools
      - --timeout-seconds
      - '870'
      - --api-timeout-seconds
      - '120'
    env_passthrough:
      - GOOGLE_APPLICATION_CREDENTIALS
      - GOOGLE_CLOUD_PROJECT
      - GOOGLE_CLOUD_LOCATION
      - GOOGLE_CLOUD_QUOTA_PROJECT
      - CLOUDSDK_CONFIG
      - ANTHROPIC_VERTEX_PROJECT_ID
```

Set the harness per-call runtime budget above the adapter timeout (e.g. 900 seconds).
The adapter reuses the vLLM controller's sandboxed Python tool, contract validation,
repair limits, and stage handoffs. It supports both `native-tools` and `json-schema`
interaction modes. Bubblewrap and an analysis Python environment are required.
Native tool calls retain the full Gemini response parts, including thought signatures.
Gemini sessions and audit files use their own namespace and freeze model, project,
location, thinking effort, and controller settings. vLLM failover environment
variables do not redirect Gemini requests.

## Structured analysis and Aim 1 recovery

```bash
python scripts/run_structured_agent.py run \
  --workspace /path/to/prepared/workspace \
  --provider gemini-vertex --model gemini-3.8-flash \
  --location global
```

For a new Aim 1 protocol:

```bash
python experiments/aim1_recovery/prepare.py \
  --out /path/to/new/experiment --python /path/to/analysis-venv/bin/python \
  --backend gemini-vertex --model gemini-3.8-flash \
  --reasoning-effort high --service-tier unspecified
python experiments/aim1_recovery/run_batch.py \
  --plan /path/to/new/experiment/plan.json --backend gemini-vertex \
  --location global
```

Use a fresh experiment directory when changing the model/backend. Existing frozen
Codex local-CLI protocols are historical experiments; the general structured runner
is the Gemini alternative. The runner retains analysis code, tool results, validated
iteration submissions, raw native responses, and token accounting. It uses the same
Python execution mechanism as current endpoint experiments, including required
bubblewrap isolation. Public inputs are read-only, analysis artifacts are confined
to the job's analysis directory, and evaluator files and network access are
unavailable to the Python process.

## Legacy task-bundle batches

After installing the extra in the selected Python environment:

```bash
scripts/run_harness.sh ocs-gemini-agent /path/to/tasks \
  --python-env /path/to/venv --replicates 2 \
  --extra-args '--model gemini-3.8-flash'
```

The Gemini entry point copies the public task files into a fresh workspace inside
each replicate's run directory. Logs, executed code and submitted iterations stay
there; the standard transcript and summary are collected by the batch script.
Custom `--prompt` instructions are appended to the task instructions.

## Groupthink pilot and cascade

```bash
python experiments/groupthink_pilot/run_pilot.py \
  --provider gemini-vertex --model gemini-3.8-flash \
  --out /path/to/new/pilot
python experiments/groupthink_cascade/run_cascade.py \
  --provider gemini-vertex --model gemini-3.8-flash \
  --out /path/to/new/cascade
```

These remain fresh, tool-free calls with schema validation and the original
experimental evidence and response contracts (model-specific Luna wording is now neutral). Manifests and report metadata identify the Gemini backend.
Use `--dry-run` to exercise scheduling without sending prompts.

## Scope and verification

Gemini can replace the LLM calls for the experiments above. CAA activation-vector
extraction/injection and model-weight interventions inherently require accessible
local model weights; the hosted Gemini API cannot implement those operations.
Gemini can still serve as their LLM judge or a separate unsteered comparison.

Tests mock Google responses for offline validation. Tiny live requests additionally
verified GCP access, JSON-schema generation, and a signed function-call round trip.
These checks do not establish scientific performance or sustained batch quota.

API references: [generateContent and Gemini 3.8 Flash](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash),
[thought signatures](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures).
