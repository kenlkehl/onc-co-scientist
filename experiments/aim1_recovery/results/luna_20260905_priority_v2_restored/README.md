# Completed reconstructed Aim 1 batch

All 250 planned Luna medium priority attempts are terminal: 246 validated
completions and four retained failures with zero recovery credit. No research
jobs remain to resume. See [report.md](report.md) for results and limitations.

A subsequent salvage check found the complete post-reconstruction archive and
the final-results commit that had not reached the remote recovery branch. The
archive was restored into a fresh directory, all 250 task inputs and 246 valid
full-budget transcripts were revalidated, and deterministic scoring reproduced
`run_scores.csv`, `group_scores.csv`, and `structured_scores.json` byte-for-byte.
No model calls or changes to experimental records or recovery criteria were
needed. The four failures and uncertainty about work lost before reconstruction
remain part of the report.

[salvage_verification.json](salvage_verification.json) records the complete
archive's identity, version, size, SHA-256, and reproduced score hashes. Version
17 of `aim1_luna_priority_v2_checkpoint.tar.gz` includes the original retained
research records, both excluded setup rounds, failure evidence, final report,
and full scores. Prior archive versions and older runs remain preserved.
Embedded earlier progress/checkpoint references describe historical snapshots;
the final verification and 250 terminal job states establish completion.

From a repository checkout with its project and analysis dependencies installed,
restore the archive to an empty directory and reproduce the results with:

```bash
python experiments/aim1_recovery/archive.py restore \
  --archive /path/to/aim1_luna_priority_v2_checkpoint.tar.gz \
  --out /path/to/empty-restore-directory
python experiments/aim1_recovery/score.py \
  --plan /path/to/empty-restore-directory/experiment/plan.json \
  --out /path/to/empty-restore-directory/rescored
```

The original full scores are under `experiment/final_report` in the archive.
Compare the three score files above against those regenerated in `rescored`.
