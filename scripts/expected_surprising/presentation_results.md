## Rerun

From the repository root:

```bash
python3 scripts/expected_surprising/build_presentation_results.py
```

Uses the standard library plus matplotlib and numpy (available in the project
`analysis` extra). It does not launch experiments or call models. Runtime depends
on disk speed because it reads and verifies every selected terminal report.
Rerunning replaces the outputs in `outputs/presentation_results/`.
The combined PDF additionally uses `reportlab` and `pypdf`. If these are not
installed in the active Python environment, the script uses the Codex desktop's
bundled PDF runtime. Elsewhere, install those packages or set `OCS_PDF_PYTHON`
to a Python executable that provides them.

Rebuild just the combined PDF using existing tables and figure PDFs:

```bash
python3 scripts/expected_surprising/build_presentation_results.py --report-only
```

To redraw all three figure sets from the existing table snapshot without rereading
experiments (preserves the saved snapshot date and scoring cohort):

```bash
python3 scripts/expected_surprising/build_presentation_results.py --figures-only
```

Optional successful-runs-only sensitivity analysis, in a separate directory:

```bash
python3 scripts/expected_surprising/build_presentation_results.py \
  --cohort completed --out outputs/presentation_results_completed_only
```

Use `--no-figures` to refresh tables only, `--workers 4` to reduce read concurrency,
or `--sources PATH` to use a different source manifest. `--repo PATH` rebases saved
repository paths when the data are copied to another checkout. A new experiment
batch or another replacement campaign must be explicitly added to the manifest
or its recorded selection, rather than implicitly choosing the latest directory.

## Outputs

- `results.csv`, `results.md`, `results.html`: the same wide table, grouped first
  by metric, with mean, 95% CI lower bound, and 95% CI upper bound for each of the
  four conditions as the rightmost column groups. HTML has
  sticky headers. CSV preserves unrounded values.
- `clinical_experiment_report.pdf`: the complete report, beginning with a dataset overview (N, all 43
  covariates grouped by type, planted finding categories, and the focal reversal),
  followed by clinical questions and all six planted discoveries, the exact NLR direction reversal,
  a compact dataset-construction schematic followed by a workflow diagram with
  claim and investigation states, metric and CI definitions, seven condition-metric
  table pages and a separate summary/interaction table, nine persistent-plus-Biomni figures
  first, nine all-workflow figures, and the eight
  model/harness figures, and source/model metadata. Pages are landscape 16:9,
  metric figures retain vector content, and the report has bookmarks and page numbers.
- `figures/custom_runner_claim_states.png`: the revised standalone workflow
  figure used on report page 5, starting from the user's supplied illustration.
- `figures/dataset_construction_compact.png`: the application schematic with wider,
  shallower boxes, placed on report page 4 immediately before the workflow figure.
- `clinical_context.json`: discovery comparisons, categories, injected
  coefficients, the covariate inventory and data dictionary, masking map, and frozen source hashes used to generate the
  clinical section. The script verifies that only the specified focal equation
  changes between expected and surprising versions.
- `figures/{metric}.png`, `.svg`, `.pdf`: one figure per metric, with four panels
  in table-column order and one bar per model/harness/workflow in each panel.
  PNG is high resolution; SVG and PDF remain editable/vector. Workflow colors
  and group order are consistent across figures.
- `figures/all_metrics.pdf`: seven condition-metric figures followed by summary S
  and the separate interaction I, one per page.
- `figures/persistent_only/`: nine figures in PNG, SVG, and PDF plus
  `all_metrics.pdf`, comparing persistent workflows with Qwen 3.8 27B/Biomni. These precede the broader
  figure sets in the combined report. These figures use larger grant-oriented type
  (18-32 pt on the native canvas), with enlarged ticks, values, axis labels, panel
  headings, titles, and wrapped footnotes; model/harness labels omit the repeated
  workflow name. This typography is retained on reruns. The workflow is labeled Biomni and retains its xhigh reasoning setting; the persistent runs use medium reasoning.
  Biomni is a separate comparator and is not pooled with custom-runner Qwen.
  All bars in this focused figure set are blue, including Biomni; the broader
  all-workflow and model/harness figures retain the distinct workflow colors.
  The historical `persistent_only` folder name is retained for stable file links.
- `figures/by_model_harness/`: an additional figure for each model/harness
  combination (and reasoning setting), with nine panels. Seven panels group
  workflow bars side by side within expected-unmasked, expected-masked,
  surprising-unmasked, and surprising-masked conditions. Workflow colors and
  0-100 axes are consistent across condition metrics and summary S. The two new
  panels compare workflows on summary S and signed interaction I; I uses a
  symmetric, zero-centered percentage-point axis. Biomni has its single Biomni workflow.
  Individual figures are saved as PNG, SVG, and PDF;
  `all_model_harnesses.pdf` collects them into one file.
- `persistent_results.csv`, `.md`, `.html` and `persistent_cross_condition_scores.csv`,
  `.md`, `.html`: focused tables for the same eight persistent-plus-Biomni groups
  shown in the first figure collection. The PDF tables place these eight groups
  first, separated from other workflows by a horizontal divider. Saved full-table
  CSVs retain their original metric and group ordering.
- `condition_metrics.csv`: per-condition means, 95% CI bounds, CI method and
  availability reason, valid bootstrap draw counts, planned/attempted/completed/error/
  unfinished/queued counts, included report counts, nonmissing metric counts,
  and evidence-class availability counts.
- `cross_condition_scores.csv`, `.md`, `.html`: separate across-condition summary S
  and focal interaction I, with execution counts, token medians, 95% CI bounds,
  CI method and availability reason. The Markdown results also append this table,
  and the HTML results link to it. Neither cross-condition score is placed in a
  condition column.
- `run_metrics.csv`: the selected run inventory, including pending runs, raw
  scientific metrics, saved F1 score, report paths and hashes, source generation,
  version metadata, and token coverage.
- `provenance.json`: snapshot interval, source manifest hashes, Biomni export
  date, counts, and missing-report issues.

## Selection and execution counts

The report's workflow figure shows the custom runner's four agent stages
(Explore, Analyze, Appraise, Synthesize), independent validation, and the repeating
loop. Its shared-ledger panel distinguishes each claim's judgment (accepted,
rejected, unresolved) from its investigation status (active, deferred, closed).
These fields are independent, can change, and persist when not updated; rejected
or closed claims remain in the ledger. The figure is an edited version of the
user's supplied illustration, embedded by `presentation_workflow.py` from the
durable asset `assets/custom_runner_claim_states.png`. Its exact edit prompt is
saved alongside that asset. Normal reruns reuse this image without calling an
image-generation model. The metric definitions explain the 25-round budget,
scheduled two-round evidence-response check, and final confirmation. Persistent,
sequential, and deliberative modes share this scientific loop, with different
stage-context and coordination rules; Biomni has its own workflow.

The default manifest combines the current 360-run unmasked grid, 360-run masked
grid, 60 launched Claude runs, and 40-run Biomni export (820 selected slots).
This is the recent clinical
workflow comparison, not earlier smoke tests, different dataset thresholds,
full-dataset benchmarks, or superseded execution attempts.

Unmasked selection follows the original plan, the audited 104 Codex replacements,
and the selected recommended-sampling Qwen/Gemma replacements. Masked selection
follows `selected_run_roots.json`, including the separate Gemma worker root.
Claude uses the recorded first-12 and next-48 selections; unused slots from its
original 120-run plan are excluded. Any subsequently started run in that plan is
also picked up on rerun. Add a new selection file to the manifest if its queued
runs should be included before they start. An unfinished replacement never falls back to its
superseded predecessor. Source generations remain distinguishable in the run CSV.

Attempted means a **selected run identity** has an execution directory or terminal
result. It is not a count of provider retries, discarded runs, or restart costs.
Successfully completed means `status=completed`. Ended with errors means
`status=failed`. Unfinished means a directory exists but no terminal result was
saved; this does not assert the process is currently active (it may be paused).
Queued runs have no execution directory. Counts in the wide table span all four
conditions and repeat for each metric; per-condition counts are in the long CSV.

By default, scientific means use **all ended runs with an available report**,
including runs that ended with errors. This avoids silently excluding poor
execution outcomes, while describing the findings actually present in the trace.
`--cohort completed` restricts scientific means to successful runs instead.
Unfinished and queued runs never receive zero scores. Missing reports are
explicitly flagged; checksum mismatches and inconsistent identities stop the
script rather than accepting unverified scores. Undefined metrics remain missing.

## Metrics and weighting

All presentation metrics use a 0–100 scale. Each condition is summarized separately;
there is no pooling of masked/unmasked or expected/surprising runs. Replicates
within a base dataset are equally weighted, then available base datasets are
equally weighted. The present comparison contains one base dataset pair.
Each dataset version contains six planted findings, and a run may propose
additional findings. Precision and recall count findings within a run; the table
then averages run scores, rather than pooling individual claims or patients.

- **Precision:** saved exact discovery `Q`, the independently confirmed and tested
  final accepted claims divided by all distinct final accepted claims, multiplied
  by 100. Duplicate claims count once. Three confirmed out of four accepted gives
  75%. Confirmation uses fresh simulated data and requires evidence that the effect
  in the claimed direction exceeds the prespecified cutoff, with a multiplicity
  adjustment across final accepted claims. Additional findings outside the six
  planted targets can count toward precision. No accepted claims gives an undefined
  precision, not zero. Nonmissing counts can differ from included runs.
- **Recall:** saved exact discovery `R`, which gives equal weight to expected,
  neutral, and surprising planted-finding categories. For each category, divide
  the number of recovered targets by the number available, then average the three
  fractions and multiply by 100. A recovered target must be tested, finally accepted,
  independently confirmed, and exactly matched including subgroup and direction;
  each target counts once. In the expected-focal dataset, recovering only the three
  expected findings gives `(3/3 + 0/2 + 0/1) / 3 * 100 = 33.3%`, even though these
  are three of the six targets. The expected/surprising column names describe
  dataset conditions, not these finding categories.
- **Scientific F1\*:** the saved exact `diagnostic_D`, checked against the per-run
  harmonic mean of precision and category-balanced recall. Mean per-run F1 is
  reported, not F1 calculated from aggregate precision and recall. Undefined
  precision yields zero F1 under the existing evaluator convention. This removes
  the inconsistent whole-run zero penalty from older Codex and Biomni reports;
  their original saved scores remain in `f1_saved` in the run CSV.
- **Exploration E:** how broadly and how early the agent tests the planted
  comparisons. At the end of each of the 25 iterations, calculate the fraction
  of each finding category with a valid test so far. Average across the three
  categories and all 25 iterations, then multiply by 100. Testing the exact
  comparison counts regardless of proposed direction, test result, or acceptance;
  repeated tests do not increase coverage. Earlier testing earns more credit
  because its coverage contributes to more rounds. Biomni uses 25 reporting rounds.
- **Evidence responsiveness B:** whether the agent's decision follows validation
  evidence two rounds after receiving it. A result is supported when its entire
  interval exceeds the minimum-effect cutoff (reference decision: accept), excluded
  when entirely below (reject), and ambiguous when it spans or touches the cutoff
  (unresolved). In this NSCLC benchmark the cutoff is 0.10 natural-log PFS units;
  exclusion need not imply an exactly zero or opposite effect. These evidence
  intervals are distinct from the report's run-level 95% CIs. Within each run and
  evidence class, accuracy is correct decisions / eligible validation results.
  Include voluntary and automatically delivered validation results; exclude invalid
  results and those whose two-round follow-up is beyond the budget or reached run
  history. A missing decision at an eligible follow-up counts as incorrect.
  Average each evidence class's saved per-run
  accuracy across available runs, then average supported, excluded, and ambiguous
  classes equally and multiply by 100. B is unavailable if any class has no
  eligible observations. It is deliberately not the mean of only those runs
  whose own B is defined; that would discard useful class observations. The long
  CSV supplies separate class counts, so `metric_available_runs` (runs with their
  own defined B) need not equal the number contributing to the aggregate B.
- **Focal recovery:** an exact independently-confirmed match to the focal
  milestone, without whole-run failure penalties, consistent with scientific F1.

## Confidence intervals

Every condition mean has 95% CI columns in the tables and error bars in all
three figure sets. These intervals describe repeated-run uncertainty conditional
on the fixed clinical dataset pair, not uncertainty across patients or datasets.
They use the same eligible scientific cohort as the means.

Continuous metrics use 10,000 percentile bootstrap draws, resampling whole run
records with replacement within each base dataset. Each draw repeats the point
estimate's dataset weighting and missing-value handling. For B, all three
evidence classes are resampled jointly and the class-balanced statistic is
recomputed; this preserves within-run correlations. Seeds are deterministic by
model, harness, workflow, reasoning effort, condition, and metric, with base seed
20260912. Run order is stable so reruns of the same inputs reproduce the bounds.

A bootstrap CI requires at least two runs per dataset, at least two available
observations for each necessary component overall, and valid estimates in at
least 95% of draws. Otherwise its bounds are unavailable, even if a point estimate
is defined. Constant continuous scores can have zero-width bootstrap intervals;
this does not establish certainty. Binary focal recovery uses Wilson score
intervals for this single dataset pair, including at 0% and 100% recovery.
All intervals are marginal, not paired workflow-effect tests or intervals adjusted
for multiple comparisons. Provenance records the settings and the long CSV
records each cell's method and availability.

## Tokens and comparability

The median is **observed output tokens per successfully completed run**, pooled
across the four conditions. It excludes failed and unfinished runs and is
unchanged by the scientific cohort flag. For custom/Codex runs it uses the saved
coordinator accounting, which includes peers, chairs, repairs, and returned
retries. Biomni uses its exported known-output usage. Reasoning tokens already
included in output usage are not added again. Input tokens, where available, are
retained separately in the run CSV; they are not included in the displayed median.

Missing token totals are excluded, not treated as zero. Known totals can be lower
bounds when returned calls lack usage or internal attempts were unaccounted.
Inspect `missing_token_calls` and `unknown_internal_attempts` in the run CSV.
Unknown internal-attempt coverage for the Biomni export is left blank. Superseded
attempt costs and smoke tests are outside the selected-run median.

The Codex label identifies the Codex CLI transport; its scientific workflow is
still coordinated by this project's runner. Claude and local models use the
custom runner. Biomni uses its own workflow, full resources, and
xhigh reasoning; the other displayed conditions request medium reasoning. These
differences, frozen-code generations, and uneven completion preclude interpreting
these plots as a controlled isolated harness or workflow effect.

The Gemma serving alias `gemma4-31b` is normalized to the recorded
`RedHatAI/Gemma-4-31B-IT-FP8-Dynamic` weights, based on the saved worker handoffs.
The source manifest records this mapping and its evidence; the original served
identifier is retained as `reported_model` in the run CSV.

Local runs are reread every time, so remaining Gemma/Qwen runs appear as soon as
their terminal artifacts are written. Biomni comes from a dated local GCS export;
refresh that export if upstream Biomni results change. The script does not fetch
remote data. Snapshot start/end times document the read interval rather than
claiming an instantaneous view of running experiments. No across-dataset error
bars are drawn for this single dataset pair.


## Weighted summary and separate focal interaction

These are provisional descriptive measures, with weights chosen after the current
experiments; they are not validated or preregistered endpoints. All components
below use the existing aggregate values (0-100), with the same scientific cohort.

For each condition c:

`C_c = 0.35 * F1_c + 0.25 * focal_recovery_c + 0.20 * exploration_c + 0.20 * responsiveness_c`

F1 is the mean of the per-run scientific F1 values, not F1 recomputed from mean
precision and recall. Precision and recall enter through F1; the additional focal
weight intentionally emphasizes the flipped finding. B retains its separate
supported/excluded/ambiguous class means across eligible runs; C is not calculated
from only the runs with all three responsiveness classes. The condition-score row
in `results.csv` and `condition_metrics.csv` provides the four inputs to S. C is an
aggregate statistic with component-specific denominators, so its
`metric_available_runs` field is blank rather than implying a single denominator.

The **grounded discovery and adaptation score S** is:

`S = (C_EU * C_EM * C_SU * C_SM) ** 0.25`

EU, EM, SU, SM denote expected-unmasked, expected-masked, surprising-unmasked,
surprising-masked. All four conditions receive equal weight. S ranges from 0 to
100, with higher values better. A true zero C makes S zero; any missing required
component or condition makes S unavailable without reweighting. The geometric
mean makes a weak condition harder to offset with strong performance elsewhere.

The **separate label-by-surprise interaction I** is:

`I = (focal_EU - focal_SU) - (focal_EM - focal_SM)`

I is expressed in signed percentage points (possible range -200 to +200).
Positive means the expected-to-surprising recovery drop is larger with recognizable
labels; negative means it is larger with masked labels. Zero can reflect uniformly
good or poor recovery. It is not a higher-is-better score, and is not subtracted
from S. For example, an unmasked drop of 40 points minus a masked drop of 10 points
gives I = +30 points. A missing focal condition makes I unavailable even if the
other conditions are present. Missing B can make S unavailable without affecting I.

The focal contrast avoids the changing category weights in recall and exploration
when the focal finding moves from expected to surprising. I does not establish
pretraining bias: masking also changes interpretability. Exploration covers the
six planted comparisons, rather than all possible hypothesis diversity. B includes
automatic validation and maintaining an already-correct judgment; neither directly
scores keeping an unexpected claim open or appropriately choosing validation.

### Confidence intervals for the new scores

C and S use 10,000 deterministic whole-run bootstrap draws, independently within
each condition. Every draw samples F1, focal recovery, exploration, and all three
response-class accuracies together; recomputes their means and B; then recomputes
C and the four-condition geometric mean S. This preserves within-run covariance.
Run replicate numbers are not treated as matched stochastic trials across
conditions. The experiments share a fixed dataset pair, not common agent random
trajectories. At least two available observations per required field/class and
at least 95% finite bootstrap draws are required for an interval. Do not average
component interval endpoints. Sparse classes can leave a point estimate available
but its CI unavailable. Constant observed components, including all-success focal
samples, contribute no bootstrap variation; C/S intervals have this finite-sample
limitation and can collapse for constant samples.

I uses Wilson-based **MOVER** (method of variance estimates recovery) for four
independent binomial proportions. For coefficients `a = [1, -1, -1, 1]` and
individual focal estimates `p_i` with Wilson bounds `[l_i, u_i]`, the lower bound
is `I - sqrt(sum((a_i*p_i - min(a_i*l_i, a_i*u_i))**2))`, and the upper bound is
`I + sqrt(sum((a_i*p_i - max(a_i*l_i, a_i*u_i))**2))`. Bounds are restricted to
[-200, 200]. This retains uncertainty at all-success/all-failure boundaries.
Reference: Zou GY, Huang W, Zhang X (2009), *A note on confidence interval estimation
for a linear function of binomial proportions*, Computational Statistics & Data
Analysis 53:1080-1085, https://doi.org/10.1016/j.csda.2008.09.033.

These are approximate marginal 95% intervals, without multiplicity adjustment,
conditional on this one fixed clinical dataset pair. The derived-score code
explicitly rejects multiple base dataset pairs rather than silently extending
the single-pair uncertainty model. Weights, formulas, missing-data rules, and CI
methods are recorded under `derived_scores` in `provenance.json`.

The presentation workflow label is `Biomni`. Bootstrap random seeds retain the
historical internal identifier so this label change does not change estimates
or confidence intervals on a full rerun.

## Reproducible run costs

Every build also writes `cost_estimates.json`, `run_costs.csv`, `cost_totals.csv`,
`cost_by_workflow.csv`, and `cost_by_condition.csv`. The PDF includes cost totals,
mean costs per completed run by workflow, API prices, and electricity methods.
Totals cover known usage for all selected attempts, including failures, independent
of the scientific `--cohort` option. Mean completed-run costs exclude failed runs.
Missing token counts are unavailable, not zero-cost observations.

Assumptions are versioned in `scripts/expected_surprising/presentation_costs.json`.
The API price snapshot is September 12, 2026: standard global pricing, including
recorded cache discounts and long-context surcharges where applicable. Output
counts already include reasoning tokens. Prices are not automatically fetched.
The Claude adapter does not request caching; its merged input count is priced as
uncached. Revisit this assumption if the adapter gains cache-control support.

For Gemma and Qwen (custom runner and Biomni), the default electricity assumption
is **$0.15/kWh**, **one existing GPU at 600 W**, **1,000 input tokens/second**, and
**150 output tokens/second**. This tariff is a planning assumption, not a measured
utility rate. Compute active hours as `(input/1000 + output/150)/3600`, energy as
`hours * 0.6 kW`, and cost as `energy * $0.15/kWh`. This gives $0.025 per million
input tokens and $0.166667 per million output tokens. Runtime is additive and does
not model overlapping requests, batching, prompt-cache acceleration, idle periods,
CPU/RAM, cooling, or hardware depreciation. These are marginal electricity costs,
not full serving costs or measured GPU energy. Multi-GPU configurations require
updating GPU count/power and throughput consistently.

To change assumptions and rebuild the PDF without rerunning scientific scoring:

```bash
python3 scripts/expected_surprising/build_presentation_results.py --report-only \
  --cost-assumptions scripts/expected_surprising/presentation_costs.json
```

OpenAI usage comes from saved provider audits; Claude usage comes from saved call
journals. This recovers known input even when a missing call makes a run's aggregate
input null. Local models use the complete saved input/output totals; Biomni uses
its dated export. Unknown calls and unlogged retries are not imputed, so recorded
costs may be lower bounds. No inference calls are made by the cost estimator.

`cost_usage_cache.json` caches terminal API token buckets against the complete run
inventory row (including the source path, report hash, status, and token fields).
New or changed run rows are reread; unfinished runs are never cached. Rates are
always reapplied, even when token usage is cached. After modifying audit files in
place without updating their report/run metadata, add `--refresh-costs` to force
an audit reread. A clean build without the cache also rereads all source audits.
The cost JSON records hashes of the cohort and assumptions. The PDF reuses the
same scientific snapshot and existing figures when `--report-only` is selected.

Standalone cost refresh:

```bash
python3 scripts/expected_surprising/presentation_costs.py
```
