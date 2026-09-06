# Co-Scientist Experiment Harness

The harness is a model-agnostic experiment controller. It keeps the scientific
workflow, communication policy, site boundary, resource budget, and audit trail
outside the agent runtime so those factors can be manipulated independently.

## What it implements

- **Persistent workflow:** one session performs every scientific stage.
- **Sequential workflow:** every stage starts a fresh session and receives only
  the preceding stage's written handoff.
- **Deliberative workflow:** multiple stage-specific scientists commit
  independently, optionally revise after seeing structured peer artifacts, and
  pass a consensus artifact to the next stage.
- **Federated deployment:** the selected workflow runs separately at each site.
  The central reviewer receives only each site's final structured handoff. Site
  prompts, raw outputs, and data paths are not placed in the central prompt.
- **Matched budgets:** calls, tokens, tool calls, cost, and per-call wall time are
  controlled by one experiment-level budget.
- **Provenance:** each run retains the configuration fingerprint, prompt hashes,
  normalized artifacts, runtime metadata, usage, failures, and an append-only
  JSONL event ledger.
- **Resume:** completed matrix cells with the same configuration fingerprint are
  reused.
- **Gold isolation:** `private_evaluation_path` is harness-side metadata. It is
  excluded from agent requests and from the resolved public specification.

The default stages are hypothesis generation, analysis, critique, and
synthesis. They can be replaced in the experiment YAML.

## Treatment roles

Set each task's `treatment_columns` to all treatment variables in that task's
public naming scheme. For example, use `[treatment_sotorasib]` for a dataset
with that single named treatment, or `[feature_123]` for its masked counterpart.
Every stage prompt includes these roles, including persistent follow-ups,
sequential agents, deliberative peers and chairs, and final reviewers.
Other task types can leave the list empty.

Both NSCLC grid preparers (`prepare_experiment.py` for Codex CLI and
`prepare_vllm_experiment.py`) populate the field automatically from the dataset
schema and its private rename mapping. The generated public description also
lists these roles. All four NSCLC treatments are disclosed; effect identities,
subgroup definitions, and the rename mapping stay private. Masked prompts use
only opaque treatment column names.

This applies to newly prepared experiments. Prepare into a new output directory
to test the change; archived prompts and frozen configurations retain their
original treatment-role disclosure.

## Quick smoke test

The included example uses a deterministic stub, so it makes no model calls:

```bash
ocs harness validate-experiment \
  --config configs/co_scientist.example.yaml

ocs harness run-experiment \
  --config configs/co_scientist.example.yaml
```

Use `--dry-run` to write `plan.json` and `resolved_spec.json` without invoking
agents. Use `--resume` to top up only incomplete cells.

## Pi RPC runtime

Pi is used as a pinned execution dependency, not forked. Its headless RPC mode
provides persistent sessions and JSONL events:

```yaml
models:
  - id: pi-gpt-example
    adapter: pi-rpc
    command: [pi]
    provider: openai
    model_id: openai/gpt-5
    pi_cleanroom: true
    pi_tools: [read, bash, grep, find, ls]
    extra_args: []
    env_passthrough: [OPENAI_API_KEY]
```

The controller starts `pi --mode rpc --no-session`, sends prompts over stdin,
waits for `agent_settled`, obtains the final assistant text and session usage,
and saves the raw RPC events. Persistent workflow stages reuse one process;
sequential stages receive separate processes.

By default the adapter replaces Pi's coding system prompt, disables discovery
of extensions, skills, prompt templates, themes, and context files, and
allowlists only `read`, `bash`, `grep`, `find`, and `ls`. Set
`pi_cleanroom: false` only for an explicitly prespecified extension condition.
Set `pi_tools: []` for a tool-free condition. Because `bash` can execute
arbitrary processes, protected-data runs still require an OS sandbox.

Pi does not itself create a security boundary. For protected data, launch the
harness inside site-specific containers or policy sandboxes and mount only the
corresponding public task workspace. `workspace_strategy: copy` provides
filesystem separation for manageable datasets but is not a substitute for an
OS sandbox.

## Generic JSON CLI adapter

Any agent framework can be integrated through a small executable:

```text
agent-command --request-file request.json --output response.json
```

The request contains the prompt, stage, session identifier, model identifier,
workspace, scratch directory, and non-secret experimental metadata. The output
must match the `AgentArtifact` schema:

```json
{
  "summary": "complete stage result",
  "handoff": "self-contained handoff",
  "hypotheses": [],
  "analyses": [],
  "evidence": [],
  "concerns": [],
  "minority_report": "",
  "final_answer": null
}
```

Commands are passed as an argument vector, never through a shell. Use
`{request_file}` and `{output_file}` placeholders when an adapter requires
nonstandard argument placement.

## Clinical benchmark exercises

The public question banks in `clin-genomic-analysis-benchmark/questions/` can
be imported without adding a Python dependency:

```yaml
clinical_benchmark:
  questions_root: ../../clin-genomic-analysis-benchmark/questions
  cohort_data_root: /site/public/bpc_from_synapse
  cohorts: [nsclc_2.0_public, crc_2.0_public]
  categories: [6, 7, 8]
  limit_per_cohort: 4
```

Only `id`, `category`, and question text are accepted. The importer fails
closed if a public YAML contains classification, analysis specifications,
gold answers, or disambiguation concepts.

## Federated deployment profiles

One matrix model profile can assign different models to different sites and to
the central reviewer:

```yaml
models:
  - id: mixed-commercial-open
    adapter: pi-rpc
    command: [pi]
    model_id: openai/gpt-5
    site_model_ids:
      site_a: anthropic/claude-sonnet
      site_b: google/gemini-pro
      site_c: openai/gpt-5
      site_d: local/qwen
    central_model_id: openai/gpt-5
```

This profile is one experimental matrix cell. Usage is still accumulated into
one matched run-level budget.

## Output layout

```text
<output_root>/
├── plan.json
├── resolved_spec.json
├── summary.json
└── runs/<task>__<workflow>__<model>__rNNN/
    ├── run.json
    ├── events.jsonl
    ├── artifacts.json
    ├── scratch/
    ├── workspaces/          # copied or central workspaces, when applicable
    └── calls/call_NNNN/
        ├── normalized_response.json
        └── runtime-specific logs
```

Scoring remains a separate trusted process. It should read `run.json` and
`artifacts.json` alongside evaluator-only material that is never mounted into
an agent workspace.
