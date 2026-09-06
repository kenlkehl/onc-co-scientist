# DS001 NSCLC: local Codex CLI replication handoff

This file is for the **coordinator**. Do not give it, prior results, or answer keys
to research workers. Give each worker only its generated task prompt and workspace.

## Goal

Repeat **GPT-5.6 Sol, medium reasoning, through Codex CLI**: 20 independent named
and 20 masked DS001 NSCLC sessions, plus two separately excluded setup sessions.
Each replicate has one continuing CLI context and exactly 25 research iterations.
Do not substitute Work subagents, another model, or the repo's endpoint runner.

This repeats the NSCLC experiment, not the five-cancer 200-run pilot. The source
is the same 50,000-row DS001 cohort Opus saw. Match Sol Work by giving workers the
fixed 40,000 discovery rows, with 10,000 held out (split seed 20260904). Giving
workers all 50,000 rows or removing the structured protocol would be a separately
labeled experiment, not a silent change to this one.

Keep the data, masking, research actions, schema, iteration count, and v2 scorer
fixed. Preserve every attempt. Recovery scores stay outside research contexts.
Do not tune the protocol based on setup recovery or interim results. An iteration
is a submitted research record, not a model call or fixed amount of compute.

## Existing materials and results

| Material | Repository location |
|---|---|
| Original DS001 cohorts and Opus transcripts/results | `example_data_clinical_all_claude/ds001/` |
| Initial Luna results | `experiments/aim1_recovery/results/luna_20260904/` |
| Same Luna runs rescored under v2 | `experiments/aim1_recovery/results/luna_20260904_v2_rescore/` |
| Later fixed-budget Luna experiment | `experiments/aim1_recovery/results/luna_20260905_priority_v2_restored/` |
| Completed DS001 Sol Work experiment | `experiments/aim1_recovery/results/sol_nsclc_20260905_v2/` |
| Shared submission/scoring definition | `docs/DETERMINISTIC_RECOVERY.md` |
| New local preparation and launcher | `experiments/aim1_recovery/local_cli.py` |

| Experiment | Named | Masked |
|---|---:|---:|
| Opus, pooled five cancers, legacy prose matcher | 72/100 | 92/100 |
| Opus, NSCLC only, legacy prose matcher | 8/20 | 17/20 |
| Sol Work, NSCLC, structured v2 primary | 7/20 | 2/20 |
| Later Luna Work, NSCLC, structured v2 primary | 1/20 | 0/20 |

All nine Sol Work recoveries also met strict matching and held-out confirmation.
The old Opus prose claims have not undergone a valid quantitative rescore under
the new structured rules. Missing JSON `finding` fields are a format issue, not
evidence of scientific failure. Scoring alone has not been shown to explain the
historical difference.

DS002 and DS003 data/results remain in their separately saved Google Drive
archives. No DS002/DS003 commits or files were carried into this handoff branch.
Existing local experiment branches and archives were preserved.

## Current execution status and limitations

**No CLI scientific run has started.** In the cloud session, CLI 0.153.4 installed
and `codex login status` reported ChatGPT authentication. Minimal requests for
`gpt-5.6-sol`, medium, stalled; two bounded probes timed out without a response.
This does not establish whether Sol is supported on the user's local CLI account.

The new launcher has local tests for input preparation, command construction,
interruption/resumption safeguards, and integrity checks. It has not been
validated end to end against a responding live model here. The two setup sessions
are the end-to-end gate before formal dispatch.

The CLI launcher preserves user configuration, auth, and execution rules. It
requests `workspace-write` with no interactive approvals; it does not disable
sandboxing or rules. Record CLI version, active tools, user/project instructions,
and any configuration changes before the pilot. Never record secrets. CLI default
service tier is not claimed equivalent to Work priority. Token/compute budgets
are not matched. Directory separation and the CLI write sandbox are not read
isolation: workers are instructed to inspect only their own inputs and outputs.

## Local setup and readiness

Use Linux or macOS (POSIX process groups); Windows users should use WSL. Start
from a clean checkout of this handoff. Run from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis,dev]'
codex --version
codex login status
```

If needed, use the ordinary local `codex login` flow. Test the exact model:

```bash
codex -a never -s read-only -m gpt-5.6-sol \
  -c 'model_reasoning_effort="medium"' exec --json \
  'Reply with exactly READY. Do not use tools or read or write files.' </dev/null
```

Have the coordinator bound the probe to 60 seconds. If it hangs, rejects Sol, or
reports authentication/endpoint errors, diagnose before dispatch. Retain the
error and CLI version; do not fall back to another model or Work. A model's
self-description is not independent attestation of which model served a request.

## Setup pilot: one per condition, excluded

Use the virtualenv interpreter path, not the dereferenced underlying executable:

```bash
python -m experiments.aim1_recovery.local_cli prepare \
  --out data/ds001_sol_cli_setup --python .venv/bin/python --repeats 1
python -m experiments.aim1_recovery.local_cli run \
  --plan data/ds001_sol_cli_setup/plan.json --dry-run
python -m experiments.aim1_recovery.local_cli run \
  --plan data/ds001_sol_cli_setup/plan.json --jobs 2
```

Both sessions must finalize 25 records, all four research actions, a summary, and
valid artifact receipts. Check CLI events for model/execution errors. Validate
protocol adherence without scoring setup recovery. Document any runtime fixes
and repeat a separately labeled setup if needed, preserving failed artifacts.
Freeze the working launcher and configuration before the formal batch.

## Formal batch: 20 per condition

```bash
python -m experiments.aim1_recovery.local_cli prepare \
  --out data/ds001_sol_cli --python .venv/bin/python --repeats 20
python -m experiments.aim1_recovery.local_cli run \
  --plan data/ds001_sol_cli/plan.json --dry-run
python -m experiments.aim1_recovery.local_cli run \
  --plan data/ds001_sol_cli/plan.json --jobs 4
```

Use `tmux` or another persistent terminal. Each task's `cli_logs/` retains the
command, CLI version, timestamps, event JSONL, stderr, and thread ID. Stdin is
closed explicitly. The default wall limit per attempt is three hours; timeout
preserves partial work. Four jobs means four independent scientists, not four
collaborators on one task. Workers must not inspect prior runs, answer keys,
private evaluator files, sibling directories, external sources, or delegate.

`data/` is Git-ignored. Raw CLI logs can contain context beyond scientific records;
review them before publication. Never include credentials or machine-local auth.

## Resume interruptions

```bash
python -m experiments.aim1_recovery.local_cli run \
  --plan data/ds001_sol_cli/plan.json --jobs 4 --resume
```

This skips validated completed jobs, starts untouched jobs, and resumes partial
attempts by CLI thread ID. It refuses a new context over a partial attempt without
a thread ID. Restrict dispatch with `--job-id job_0007` (repeat for more jobs).

Do not delete partial outputs or regenerate plans. Preserve the CLI's own local
session storage: the experiment archive alone does not include the complete CLI
context needed to resume. After an unclean kill, `cli_logs/running.lock` may remain.
Verify the old process has stopped before removing that specific stale lock and
document the incident. Never run two processes for one replicate.

If a context is irretrievably lost, retain the attempt in the denominator. Add an
entry keyed by job ID to `terminal_failures.json`, containing `job_id`, `reason`,
`retain_in_denominator: true`, and `recovery_credit: false`. Do not replace based
on recovery. Report interruptions and a sensitivity excluding affected attempts.

## Score, archive, and report

After all formal attempts finish or are explicitly retained as terminal failures:

```bash
python -m experiments.aim1_recovery.score \
  --plan data/ds001_sol_cli/plan.json --out data/ds001_sol_cli/final_report
python -m experiments.aim1_recovery.archive pack \
  --root data/ds001_sol_cli --archive data/ds001_sol_cli_complete.tar.gz
python -m experiments.aim1_recovery.archive restore \
  --archive data/ds001_sol_cli_complete.tar.gz --out data/ds001_sol_cli_restored
python -m experiments.aim1_recovery.score \
  --plan data/ds001_sol_cli_restored/experiment/plan.json \
  --out data/ds001_sol_cli_restored/rescored
```

Ensure all processes are idle before packing. The archive utility also packs the
sibling `ds001_sol_cli_setup` directory. Compare regenerated `run_scores.csv` and
`group_scores.csv` byte for byte with the originals. Report primary complete-rule
recovery, strict recovery, held-out confirmation, and iteration endpoints
separately. Compare NSCLC rows to Sol Work, not the pooled Opus 72%/92% figures.

Return named/masked counts out of 20, failures/resumes, CLI/model configuration,
token usage where available, and archive reproduction. Commit a compact report,
protocol/preflight/validation records and score tables under a new
`experiments/aim1_recovery/results/sol_nsclc_cli_<date>/` directory. Preserve old
runs and exclude DS002/DS003. Store the complete raw archive separately if too
large for Git. Report actual execution, not inferred access from metadata.

## Prompt for the local coordinator

> Read docs/DS001_CODEX_CLI_HANDOFF.md and carry out the DS001 NSCLC Sol 5.6 medium
> replication using local Codex CLI: two excluded setup runs, then 20 named and
> 20 masked formal runs. Follow the setup gate, preserve the fixed split/protocol/
> scorer, keep each replicate in its own persistent CLI session, retain all
> failures and resumes, and give workers no prior results or answer keys.
> Validate and score the completed batch, verify archive restoration, and commit
> and push the compact report and reproducibility records. Preserve prior
> experiments and exclude DS002/DS003. If the exact model or runtime cannot run,
> report the blocker rather than substituting a model or harness.

Official references: [non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode)
and [Codex CLI](https://learn.chatgpt.com/docs/codex/cli).
