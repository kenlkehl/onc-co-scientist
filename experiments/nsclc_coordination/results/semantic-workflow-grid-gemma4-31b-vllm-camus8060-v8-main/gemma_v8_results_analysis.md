# Gemma v8 NSCLC semantic workflow-grid analysis

Generated: 2026-09-08

Gemma completed all 30 planned runs successfully. It recovered the planted
discovery, but recovery was uncommon, concentrated in masked sequential runs,
and frequently lost at later checkpoints.

## Automated results

| Semantic condition | Workflow | Ever exact | Terminal exact | Ever near/exact |
|---|---|---:|---:|---:|
| Masked | Sequential | **3/5** | **1/5** | 4/5 |
| Masked | Persistent | 1/5 | 0/5 | 3/5 |
| Masked | Deliberative | 0/5 | 0/5 | 1/5 |
| Named | Sequential | 0/5 | 0/5 | 1/5 |
| Named | Persistent | 0/5 | 0/5 | 0/5 |
| Named | Deliberative | 0/5 | 0/5 | 2/5 |

Overall:

- Exact discovery at any synthesis checkpoint: **4/30 runs (13.3%)**.
- Exact discovery in masked runs: **4/15 (26.7%)**.
- Exact discovery in named runs: **0/15**.
- Exact discovery retained at iteration 20: **1/30 (3.3%)**.
- Supported near-or-exact recovery: **11/30**.
- Technical completion: **30/30**, with no terminal timeouts or malformed
  artifacts.

## Recovered discovery

The successful runs identified:

- Exposure: `feature_018`, corresponding to sotorasib.
- Subgroup: `feature_016=1`, `feature_005=0`, `feature_028=0`, and
  `feature_031=0`.
- Clinical translation: KRAS G12C positive, BRCA2 wild type, ALK wild type,
  and male.
- Estimated benefit: **+4.985 months**.
- Subgroup size: 3,266 patients, including 1,154 exposed and 2,112
  comparators.

An independent recomputation from the public dataset produced:

- Inside the subgroup: +4.985 months, with a p-value effectively zero.
- Outside the subgroup: -0.006 months, p=0.756.
- Overall population: +0.322 months.

This confirms genuine treatment-effect heterogeneity rather than only a
population-wide treatment main effect.

## Manual audit of scorer-positive runs

The exact synthesis results first appeared at iterations 3, 8, 12, and 15.

- The three masked sequential discoveries are scientifically credible when
  the complete four-stage iteration is considered. Their analyze/appraise
  stages explicitly demonstrated essentially no treatment effect in the
  complementary population.
- The masked persistent hit is less clean. It found all four correct modifiers
  and convincing triple interactions, but its decisive structured claim
  compared exposed clean responders with an inhibited subgroup rather than
  exposed versus unexposed patients inside the exact subgroup. It also used an
  incorrect subgroup denominator.

The most defensible reporting is therefore:

- Automated benchmark result: **4/30 exact**.
- Strictly defensible discoveries: **3/30**, all masked sequential.
- Terminally retained strict discovery: **1/30**.

The evaluator should eventually require a valid within-subgroup exposure
contrast plus an out-of-subgroup or formal interaction contrast before
awarding exact support.

## Interpretation

- **Masking helped markedly.** Named runs became anchored on the familiar
  sotorasib/KRAS G12C relationship: 267 of 300 named synthesis checkpoints
  stopped at KRAS G12C alone or KRAS G12C plus male sex. They rarely explored
  both less-obvious ALK-negative and BRCA2-negative gates.
- **Sequential coordination performed best.** Fresh contexts and compact
  handoffs appear to have reduced semantic anchoring and context saturation.
- **Deliberation did not help.** It used three times as many successful
  artifacts and never produced an exact discovery. No deliberative peer-level
  artifact contained an exact hit either.
- **Retention was the principal weakness.** Every exact-positive run dropped
  the exact result at least once later. Only masked sequential replicate 2
  regained it and retained it at iteration 20.
- **BRCA2 wild type was the main search bottleneck.** Sixty of 61 masked
  near-exact synthesis checkpoints found the other three predicates but
  omitted `feature_005=0`.

Because there were only five replicates per experimental cell, these workflow
and semantic-condition differences are descriptive rather than confirmatory.

## Resource efficiency

| Workflow | Exact runs | Successful artifacts | Attempts | Total tokens | Contract repairs |
|---|---:|---:|---:|---:|---:|
| Sequential | **3/10** | 800 | 806 | 33.3M | 4 |
| Persistent | 1/10 | 800 | 805 | 157.2M | 46 |
| Deliberative | 0/10 | 2,400 | 2,414 | 97.0M | 15 |

Persistent execution accumulated a very large context-token burden.
Deliberative execution spent substantially more output and coordination effort
without improving exact recovery.

Across the complete experiment, the scorer recorded 4,000 successful model
artifacts, 4,025 controller attempts, 65 contract repairs, approximately 258.4
million input tokens, and 29.1 million output tokens.

## Provenance note

The result root's existing `scored/` directory was written before all 30 runs
completed and is stale. The completed result set was rescored without
overwriting it. Both the experiment's frozen scorer and the current scorer
produced identical run scores, cell summaries, and report output.

Primary source files in this directory:

- `summary.json`: completed harness summary.
- `runs/`: per-run artifacts, checkpoints, calls, and execution records.
- `provenance/scorer.py`: frozen deterministic scorer.
- `private_evaluation_index.json`: evaluator-only task-to-truth routing.
