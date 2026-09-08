# Qwen Aim 2 smoke: incomplete, checkpoint preserved

5 of 12 planned runs finished; 1 of those finished runs had an unrecovered stage failure. The remaining cells are incomplete. **No paired workflow effect is estimated from this partial matrix.**

The probe used the frozen expected/surprising controller, both NSCLC data types, both paired versions, and all three workflows. Each assigned run had a six-iteration budget. The code supports 25 iterations for both clinical and cell-line tasks; the automated checks cover the full budget.

Qwen repeatedly produced long empty or malformed replies. The shared server also showed heavy memory pressure and request preemption. The probe was resumed with two concurrent runs after starting with six, then paused with all completed results and partial checkpoints intact. Scientific settings, the 125,000-token ceiling, and repair rules were preserved.

| Data type | Version | Workflow | Status | Successful stages / 24 | Stage failures | Saved calls |
|---|---|---|---|---:|---:|---:|
| Clinical | expected | persistent | finished | 24 / 24 | 0 | 26 |
| Clinical | expected | sequential | finished | 24 / 24 | 0 | 29 |
| Clinical | expected | deliberative | not started | 0 / 24 | 0 | 0 |
| Clinical | surprising | persistent | finished | 24 / 24 | 0 | 27 |
| Clinical | surprising | sequential | incomplete, paused | 16 / 24 | 0 | 21 |
| Clinical | surprising | deliberative | incomplete, paused | 0 / 24 | 0 | 4 |
| Cell-line | expected | persistent | finished | 24 / 24 | 0 | 28 |
| Cell-line | expected | sequential | finished with stage failure | 23 / 24 | 1 | 29 |
| Cell-line | expected | deliberative | incomplete, paused | 15 / 24 | 1 | 61 |
| Cell-line | surprising | persistent | incomplete, paused | 19 / 24 | 0 | 23 |
| Cell-line | surprising | sequential | incomplete, paused | 4 / 24 | 0 | 5 |
| Cell-line | surprising | deliberative | incomplete, paused | 13 / 24 | 1 | 56 |

Integrity checks passed for 309 saved participant calls and 32 matched first-round peer pairs. Completed scientific report hashes, dataset hashes, and participant artifact hashes were verified. Peer drafts were non-authoritative; paired peers received the same starting scientific context.

The completed per-run reports retain R (recovery/recall), P (confirmed-claim fraction, analogous to precision), F1* (category-balanced discovery performance), E (exploration coverage), and B (evidence responsiveness). These are available in the [partial audit](qwen_partial_audit.json), but are not averaged over the selectively completed cells. Unfinished cells have no final score.

The [live diagnostics](live_diagnostics.json) retain empty-reply, repeated-JSON, and stage-failure evidence. The [concurrency-resume record](operational_resume.json) and [pause record](qwen_pause.json) document interruptions. Reported token totals exclude provider work without a returned reply, including interrupted requests and unreported SDK retries. Elapsed time is diagnostic because these servers were shared with other work.

The [Gemma smoke report](gemma_smoke_report.md) covers the complete planned matrix on the other deployment. This Qwen probe remains resumable with its unchanged frozen source and configuration:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=data/expected_surprising_aim2/release_20260907_bounded/src \
/tmp/ocs-es-refactor-venv/bin/python -m onc_co_scientist.cli harness run-experiment \
  --config data/expected_surprising_aim2/release_20260907_bounded/smoke.yaml \
  --resume --max-parallel 2
```
