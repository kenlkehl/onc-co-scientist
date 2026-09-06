# DS001 NSCLC: Sol medium with the archived Claude task brief

Prespecified before formal dispatch: 20 independent named and 20 independent
anonymized sessions, gpt-5.6-sol with medium reasoning, persistent Codex CLI
sessions, four concurrent workers, and one excluded setup session per condition.
No DS002, DS003, or other cancer tasks. No recovery inspection during setup or
formal execution; score the full batch after completion.

## Intervention

Use `--prompt-style claude-legacy-loose-v1`. Copy the archived per-condition
`example_data_clinical_all_claude/ds001/tasks/nsclc/{variant}/agent_instructions.md`
brief verbatim except for 50,000 -> 40,000 public discovery rows. Record source
SHA-256 hashes in the frozen protocol. Retain its propose/test/update loop,
up-to-25 iteration cap, early stopping after thorough exploration, and explicit
systematic treatment-effect heterogeneity search. This archived brief is not
instruction-free.

Append the Python location, structured-finding schema contract, contemporaneous
iteration submission commands, finalization command, and isolation instructions.
Remove the structured v2 phase allocation (20% screen / 40% multivariable / 25%
refine / remainder robustness), mandatory 25 completed iterations, four-action
quota, and distinct-script/sequential-output requirement. Do not replace these
with new scientific search advice. Code and output retention remain requested;
record receipts are checked, but per-artifact hashes/timestamps are not required.

Keep the same archived 50,000-row source, split seed 20260904, 40,000 discovery
rows and 10,000 held-out rows, masking, structured schema, deterministic v2
scorer, precision/recall thresholds, and held-out confirmation rules as the
previous local CLI experiment. Public examples retain fictional columns.

The new harness label is `codex-cli-claude-legacy-loose-v1`. The default
structured v2 preparation remains available. Recovery, strict identity,
held-out confirmation, and interaction confirmation are reported separately.
Actual iteration counts and available token usage must accompany recovery;
this intervention does not match compute. Timestamped submissions and structured
findings differ from the archived Claude output process, so this is a workflow
comparison, not an exact reproduction of Claude Code.

## Execution

The previous interpreter `/data1/ken/envs/gptoss3/bin/python` is unavailable on
this execution host. A separate environment at `data/sol_loose_env` uses Python
3.13.7 and the repository analysis/dev dependencies; record resolved package
versions with results. Installed Codex CLI is 0.153.0 (prior run: 0.153.4), using
ChatGPT authentication and the account/config default service tier. The live
Sol medium access probe returned `READY`. No account configuration is changed.

From the repository root:

```bash
data/sol_loose_env/bin/python -m experiments.aim1_recovery.local_cli prepare \
  --out data/ds001_sol_cli_loose_setup --repeats 1 \
  --python "$PWD/data/sol_loose_env/bin/python" --prompt-style claude-legacy-loose-v1
data/sol_loose_env/bin/python -m experiments.aim1_recovery.local_cli run \
  --plan data/ds001_sol_cli_loose_setup/plan.json --jobs 2
```

After both setup sessions pass input/receipt/schema/finalization checks, prepare
`data/ds001_sol_cli_loose` with `--repeats 20` and the same prompt-style option,
then run with `--jobs 4`. Do not score setup. Resume an interrupted replicate in
its original CLI thread using `--resume`; preserve every attempt. No replacement
based on scientific results. The per-session timeout is 10,800 seconds.

After every formal session is complete:

```bash
data/sol_loose_env/bin/python -m experiments.aim1_recovery.score \
  --plan data/ds001_sol_cli_loose/plan.json --out data/ds001_sol_cli_loose/scored
data/sol_loose_env/bin/python -m experiments.aim1_recovery.archive pack \
  --root data/ds001_sol_cli_loose --archive data/ds001_sol_cli_loose_complete.tar.gz
```

Restore the archive into a new directory, rescore it, and compare score CSVs.
Commit compact results and provenance; retain raw session logs and workspaces
under Git-ignored `data/`. Commit only after experiment completion, with a message
that specifies the prompt and stopping/workflow changes, then push.
