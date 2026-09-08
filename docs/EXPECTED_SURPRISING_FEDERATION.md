# Federated expected/surprising experiments

The independent `federation` grid adds site count and data partitioning to the existing task × model × within-site workflow × repeat grid. Random splitting is the default. The configuration is separate from the older generic runner's `workflow.federated` flag, which only performs central synthesis after complete site runs.

```yaml
federation:
  site_counts: [1, 2, 4]
  seed: 20260908
  partitions:
    - id: random
      mode: random
    - id: age_sorted
      mode: heterogeneous
      by: [age_years]
```

Omitting `federation` preserves existing behavior. Explicit N=1 also uses the existing coordinator, prompts, scientific seeds, analysis engine, and run IDs, with no extra central agent or calls. N=1 appears only once even when multiple partitions are listed. Each N>1 count is crossed with every configured partition. Result tables keep these conditions separate; workflow contrasts compare modes at the same N and partition.

## One goal, four stages, aggregate handoffs

For N>1, every iteration follows exploration, analysis, appraisal, and synthesis. At each stage:

1. A central orchestrator receives the common aggregate ledger and directs the sites. It may give site-specific scientific instructions. During analysis it selects up to 12 registered comparisons for evaluation across sites.
2. Each site completes that stage using the grid's persistent, sequential, or deliberative workflow. Its model context includes its own aggregate data description, its own notebook/history, central directions, and the common aggregate evidence ledger. Deliberative sites use the configured peers and chair.
3. The controller writes an aggregate handoff as each site finishes. Trusted local execution produces any directed numerical results. Site forms recommend hypotheses, assessments, and validation; they do not independently change the global registry or spend validation requests.
4. After all sites finish the stage, the orchestrator receives their handoffs, site-level results, and combined numerical summaries. It produces the one authoritative stage form and explains how it integrates the evidence. This is a stage barrier, not a central review only at the end of the experiment.

The research goal is unified. All sites evaluate each selected comparison, making global comparison coverage stable and allowing repeated analyses to use the existing canonical cache. Site-specific instructions can prioritize explanations, subgroup hypotheses, alternative interpretations, and proposed next analyses. This implementation does not give sites independent analysis queues or allow a registered comparison's site coverage to change silently over time.

The orchestrator decides whether pooled evidence, replication across sites, or site disagreement is persuasive. It is not required to vote, obtain site consensus, or apply a replication threshold. The combined numerical summary is a reference statistic, not an automatic acceptance rule. Final independent confirmation still evaluates claims about the unified population, so the benchmark remains comparable with N=1. B continues to measure agreement with the evaluator's interval rule rather than every possible scientific argument about heterogeneity.

By default the central agent uses the same provider/model as the sites. `orchestrator_model_profile` may name another profile in `models`; it changes only the N>1 central deployment. Within a grid cell, every site uses the selected workflow mode. Mixed workflow assignments among sites are not an additional dimension of this implementation.

## Partitioning and data boundaries

Random partitions reproducibly shuffle observations into approximately equal, non-overlapping sites. Heterogeneous partitions sort on the explicitly configured public covariates and divide that ordering into similarly sized sites; ties are broken reproducibly. This creates covariate-separated populations without inventing new rows. Identifiers cannot be partition covariates. Configuration, site row counts, and hashes of membership are stored in `partition.json`; row memberships and identifier values are not sent to agents.

Partition seeds depend on pair, repeat, partition, site count, and configured seed, not model or workflow. Thus all models/workflows in a matched condition use the same split. Random paired versions share the same positional split. A heterogeneous split can differ between paired versions if the chosen covariate itself differs; choose stable covariates when identical memberships across versions are required. Independent validation data use the same partition rule. Final confirmation remains private evaluator work.

Provider calls expose no rows, data-frame samples, identifiers, file paths, or site-to-site raw-data channel. Site models receive safe aggregates, and only trusted Python computes numerical handoffs from each split. The central scientific controller has a schema-only frame and cannot fall back to analyzing the pooled raw frame. Optional central model profiles inherit the same aggregate-only interface.

Site numerical handoffs contain contrast estimates, intervals, cell counts, and diagnostics. Internally, counts, means, and variances are merged to reproduce the full-population Welch contrast. Means and variances from cells smaller than 20 are suppressed. If a required site cell is suppressed, the combined result is unavailable; the engine does not silently discard that site. This prevents single-row summaries from functioning as row exports; it is not a differential-privacy guarantee.

## Shared budgets, retries, and token accounting

There is one shared analysis allowance: at most 12 distinct comparisons per iteration. A comparison evaluated at N sites is one scientific request, not N independently chosen requests. Local teams cannot add analyses beyond the central plan. The central final form must record exactly the comparisons actually executed; it cannot hide analyses or bypass the site engine.

There is also one validation service, including the existing global voluntary allowance, automatic release schedule, canonical cache, and two-iteration reassessment deadline. Site teams recommend validation; only the central form spends a slot. Independent validation returns both site-specific and combined aggregate evidence.

Participant calls share `budget.max_agent_calls`, including all sites, peers, chairs, central directions, central decisions, and repairs. With T iterations and c calls per site stage, a healthy N>1 run uses `4 × T × (N × c + 2)` calls. For N=1 it uses the existing `4 × T × c`. Here c=1 for persistent/sequential; c=`agents_per_stage × deliberation_rounds + 1` for deliberative. Configuration validation rejects a call cap below the healthy-run requirement.

Per-call output tokens are retained, summed by site and central scope, and included in condition totals and means. Unknown usage is flagged, not treated as zero. Retries count; cached replay counts once. The call journal verifies cached requests and results before replay. A central repair reuses successful site handoffs instead of spending the site budget again. The final report links participant audit artifacts, and the scientific report contains the partition and scoped resource audits.

An interrupted run can resume through the existing `--resume` option. Codex transport audits append new calls after existing attempts, including incomplete attempts, while the scientific coordinator replays verified responses. Prior audit files are preserved. Changing inputs or implementation still prevents resuming that experiment.

## Configure and verify

[Example configuration](../configs/expected_surprising.federation.example.yaml) describes 30 runs but launches nothing by itself. It requires the local packaged clinical data described in the existing setup instructions.

```bash
ocs harness validate-experiment --config configs/expected_surprising.federation.example.yaml
ocs harness run-experiment --config configs/expected_surprising.federation.example.yaml --dry-run
```

Remove `--dry-run` only when ready to launch the configured grid. Raise the shared call cap when increasing iterations, N, peer count, or deliberation rounds. Keep providers and reasoning/tier settings explicit. The already-running single-site experiment uses frozen source and is unaffected by this enhancement.

## Verification record (2026-09-08)

The targeted regression suite passed all 80 tests across federation, experiment coordination, the harness, scientific workflows, and Codex transport. Automated integration fixtures exercise all three site workflows through all four stages, checking N=1 equivalence, aggregate handoffs, shared budgets, output-token accounting, and separate condition summaries. Additional checks cover reproducible disjoint partitions, heterogeneous sorting, small-cell suppression, pooled-statistic equivalence, rejection of unapproved queries, central repairs, and replay after interruption. Federation code and tests also pass Ruff lint and formatting checks.

A bounded live smoke test used Terra through Codex with medium reasoning and the standard tier: two randomly assigned sites with sequential workflows completed exploration and analysis. It registered and executed 12 shared comparisons, all numerically valid, in eight calls with 4,630 reported output tokens and no errors. This was a two-stage smoke test, not a complete scientific experiment. The [local smoke artifact](../data/expected_surprising_federation/live_smoke_20260908/smoke_summary.json) retains calls and scoped token counts. The example grid passed configuration validation and dry-run planning; no full federated grid was launched.
