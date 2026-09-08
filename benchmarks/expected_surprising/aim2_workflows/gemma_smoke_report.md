# Aim 2 expected/surprising workflow results: expected_surprising_aim2_smoke_gemma

Each run used 6 iterations of explore, analyze, appraise, and synthesize. Each paired dataset version was run 1 time(s) per model and workflow. A repeat starts a separate agent conversation on the same dataset; it is not a new dataset.

The primary outcome is whether the workflow recovered the focal planted finding and the evaluator confirmed it independently. Failed assigned runs count as zero recovery. Expected and surprising percentages below show absolute performance. Their difference is surprising minus expected, in percentage points.

| Model | Workflow | Expected recovery % | Surprising recovery % | Difference (pp) | Failed / runs |
|---|---|---:|---:|---:|---:|
| gemma_4_31b | deliberative | 50.0 | 0.0 | -50.0 | 1 / 4 |
| gemma_4_31b | persistent | 50.0 | 50.0 | 0.0 | 0 / 4 |
| gemma_4_31b | sequential | 50.0 | 0.0 | -50.0 | 1 / 4 |

The workflow comparison subtracts the persistent workflow's paired difference from each other workflow's paired difference. A positive value means the new workflow shifts relative recovery toward surprising findings. It does not by itself show higher overall accuracy; read it alongside both recovery percentages.

| Model | Workflow vs persistent | Change in paired difference (pp) | 95% interval (pp) |
|---|---|---:|---|
| gemma_4_31b | deliberative vs persistent | -50.0 | unavailable |
| gemma_4_31b | sequential vs persistent | -50.0 | unavailable |

Results first average repeated runs within each dataset and version, then give each base dataset equal weight. Intervals resample whole base datasets, keeping their paired versions and workflows together. At least two base datasets in each included data type are needed for an interval; a one-dataset smoke test cannot estimate uncertainty across datasets.

The supporting outcomes describe the final shared research record:

- **R — recovery/recall:** the fraction of planted findings recovered, giving equal weight to expected, neutral, and surprising categories.
- **P — confirmed-claim fraction (precision/positive predictive value):** the fraction of final accepted claims that were tested during the run and confirmed in fresh evaluator data. Accepting a claim does not require prior validation. No accepted claims means P is unavailable.
- **F1\* — discovery performance:** the harmonic mean of R and P on a 0–100 scale. The asterisk distinguishes it from ordinary F1: R balances planted-finding categories, while P can include additional valid claims. Unrecovered stage errors set this primary discovery score to zero.
- **E — exploration coverage:** how broadly and how early the workflow tested planted findings across the full iteration budget, on a 0–100 scale.
- **B — evidence responsiveness:** balanced accuracy of responses to supportive, excluding, and ambiguous validation evidence, on a 0–100 scale. If a class never occurred, B is unavailable; this is missing evidence to assess, not a failure. The two-iteration response deadline is a reassessment checkpoint, not a run cap.

| Model | Workflow | R % | P % | F1* | E | B | Participant calls |
|---|---|---:|---:|---:|---:|---:|---:|
| gemma_4_31b | deliberative | 51.4 | 87.9 | 52.7 | 57.2 | 58.3 | 298 |
| gemma_4_31b | persistent | 43.1 | 81.8 | 56.3 | 49.3 | unavailable | 108 |
| gemma_4_31b | sequential | 38.9 | 76.2 | 41.2 | 42.4 | unavailable | 100 |

The table averages each run's R, P, and F1* separately. Its F1* column therefore need not equal the harmonic mean of the displayed average R and P. Unrecovered stage errors also set that run's primary F1* to zero.

All modes share analysis limits, validation samples, deadlines, final confirmation, and scientific scoring. Persistent mode retains recent committed conversation. Sequential mode uses a fresh conversation at each stage with the common ledger and notes. Deliberative peers see the same committed evidence and submit independent drafts; later rounds may see earlier peer drafts. The chair selects the only action that changes the research record. Drafts and disagreements remain in the call audit.

Persistent history is bounded at 120,000 characters of prior conversation. The complete current ledger and latest notebook are always provided. Oldest turns are removed only from active context, with each trim disclosed to the agent and recorded in the audit. This is not an iteration cap.

This comparison uses each workflow's natural number of participants. With two peers and one round, deliberation uses three calls per stage versus one in the other modes. It therefore compares complete workflows under equal scientific opportunity, not equal token or call spending. All attempts, including repairs and provider errors, are recorded. Missing provider token counts remain unavailable.

Machine-readable reports retain Q for P and D for F1* for compatibility. Each run includes the final scientific report, participant prompts and replies, and frozen input/code hashes. Secondary scores are unavailable for a whole condition if any assigned run lacks a scientific trace; the primary outcome still includes failures.

## Live verification notes

All 12 assigned runs reached the end of their six-iteration budgets: 10 completed
without an unrecovered stage error, and 2 retained stage failures. The
[artifact audit](gemma_live_audit.json) verified the planned coverage, report and
participant hashes, dataset hashes, release schedule, and peer/chair authority.

Both failed runs used the surprising clinical dataset. The sequential run failed
one exploration stage after inconsistent assessment updates. The deliberative
run failed its final exploration stage after an inconsistent claim rename, and
its final synthesis stage after assessing the same canonical claim twice. These
runs contribute zero primary focal recovery and F1*. Their diagnostic outcomes
remain in the underlying scientific reports.

This smoke contains one paired NSCLC clinical dataset and one paired NSCLC
cell-line dataset, with one run per version and workflow. Its differences mix
scientific decisions with protocol reliability and do not establish a general
workflow ranking. B is unavailable where an evidence category was absent.

The server was shared with other work, so elapsed times are diagnostic. SDK retry
counts were not reported. The [Qwen probe](qwen_smoke_report.md) remains incomplete
and is reported separately. See the [verification overview](README.md) for the
174 automated tests, full-budget checks, and dataset-preservation audit.
