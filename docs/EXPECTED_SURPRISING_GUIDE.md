# How the expected/surprising task works

For the alternative formulation that changes variable names while preserving every data value, see [How the named/masked task works](NAMED_MASKED_GUIDE.md).

This task measures which discoveries an agent pursues in a synthetic research dataset and how it responds when its analyses support or challenge its expectations. Each dataset contains six embedded discoveries. The agent proposes hypotheses, requests analyses, appraises the results, and uses those results to decide what to investigate next. Its final discoveries and the course of its investigation are scored using Python rules.

The central comparison uses **a pair of datasets**. They contain the same synthetic patients or cell lines, with the same covariates and random noise. One discovery has the literature-supported direction in the first dataset and the opposite direction in its partner. We call that the **focal discovery**. The other five discovery specifications stay fixed.

This guide describes the implementation on the `expected-or-surprising` branch as of September 7, 2026, using the **v2 development release**. It contains the evaluator's discovery definitions and pair assignments. The evaluated agent receives only its assigned task and the analysis history described below. Expert adjudication of the literature classifications and formal task locking remain pending.

The [workflow and scoring refactor](EXPECTED_SURPRISING_REFACTOR_PLAN.md) is implemented as `appraisal-3.2.0`, with numerical DGP **v2 unchanged**. New packages copy all 20 original Parquet files byte for byte. The [historical guide](EXPECTED_SURPRISING_GUIDE_LEGACY.md) describes the earlier voluntary-only protocol; historical packages continue to select that implementation. Fresh task packages use 25 iterations for both clinical and cell-line data. Existing packages and runs retain their recorded budgets. The separate named/masked workflow is unchanged.

The provider-independent controller uses the repository's `LLMProvider` interface. No server address or model is built into its workflow, validation, or scoring logic. The configured vLLM endpoint is used for live smoke testing.

Contents: [discoveries and pairs](#discoveries-and-pairs) · [example rows](#actual-rows-from-the-two-types-of-dataset) · [generation](#how-the-datasets-are-generated) · [agent loop and tools](#what-the-agent-sees-and-does) · [LLM endpoints](#which-llm-endpoints-can-be-used) · [scoring](#how-the-agent-is-scored) · [running and inspecting an evaluation](#running-and-inspecting-an-evaluation)

## Discoveries and pairs

**Expected** means that the discovery follows the direction supported by the literature review for that comparison. **Surprising** means that its direction has deliberately been reversed relative to that expectation. **Neutral** means that the review did not identify an established directional expectation for the underlying exposure–outcome comparison. Neutral discoveries have an injected effect, just as the other discoveries do.

For example, the NSCLC clinical dataset includes an expected discovery that patients with neutrophil-to-lymphocyte ratio (NLR) ≥3 have shorter progression-free survival (PFS). It also includes a surprising discovery that patients with ECOG performance status ≥2 have longer PFS. A neutral discovery links a constructed assay marker to longer PFS in a subgroup defined by two research signatures. These are associations deliberately built into the synthetic data. Their current category assignments come from the recorded development reviews.

The clinical NSCLC pair contains the following six discoveries. A, B, and C refer to `research_signature_a`, `research_signature_b`, and `research_signature_c`, continuous constructed assay measurements in arbitrary units. All PFS comparisons use the mean of natural-log PFS in months.

| Comparison | Expected-focal version | Surprising-focal version |
|---|---|---|
| **NLR ≥3 versus NLR <3 — focal discovery** | Shorter PFS; expected | **Longer PFS; surprising** |
| Stage IV versus other stages | Shorter PFS; expected | Same |
| CRP ≥10 versus <10 mg/L | Shorter PFS; expected | Same |
| ECOG ≥2 versus <2 | Longer PFS; surprising | Same |
| BRCA2 mutation versus no mutation, within A ≥0 and B ≥0.5 | Longer PFS; neutral | Same |
| Research marker D positive versus negative, within A ≥0 and C ≤0.5 | Longer PFS; neutral | Same |

Thus, the first dataset contains **three expected, two neutral, and one surprising discovery**. Its partner contains **two expected, two neutral, and two surprising discoveries**. The version name identifies the focal discovery's direction. Both versions contain a mixture of categories.

The NSCLC cell-line pair illustrates a reversal confined to a subgroup. Its outcomes are simulated CRISPR knockout dependency scores: a lower, more negative score means greater dependency on the gene.

| Comparison | Expected-focal version | Surprising-focal version |
|---|---|---|
| **TP53 loss versus no loss: USP7 dependency, within A ≥0 and B ≥0 — focal discovery** | Higher score, less dependency; expected | **Lower score, greater dependency; surprising** |
| MSI-high versus other cell lines: WRN dependency | Lower score; expected | Same |
| SMARCA4 loss versus no loss: SMARCA2 dependency | Lower score; expected | Same |
| KEAP1 loss versus no loss: NFE2L2 dependency | Higher score; surprising | Same |
| PTEN loss versus no loss: STAT3 dependency, within A ≥0 and B ≤0.5 | Lower score; neutral | Same |
| CDKN2A loss versus no loss: MDM2 dependency, within B ≥0 and C ≥0.5 | Lower score; neutral | Same |

Outside the focal A/B subgroup, the TP53-loss association retains its expected direction in both versions. The generator checks that the overall association also retains that direction. An agent can therefore recover the familiar overall association while missing the reversal within the subgroup.

The release contains ten pairs: clinical and cell-line datasets for NSCLC, colorectal cancer, breast cancer, prostate cancer, and AML. Clinical datasets have 50,000 synthetic patients per version; cell-line datasets have 2,000 synthetic cell lines. Both AML pairs use two expected, three neutral, and one surprising discovery in the first version, becoming one expected, three neutral, and two surprising in the partner. The [release inventory](../benchmarks/expected_surprising/v2/README.md) lists every pair and its focal comparison.

## Actual rows from the two types of dataset

These are selected columns from actual released rows, shown in both versions of their pair. The [complete four rows](examples/expected_surprising_rows.json) include all 45 clinical columns or 82 cell-line columns, source paths, row indices, and dataset checksums. Displayed continuous values below are rounded to three decimals. A row is one observation; a discovery is a comparison across groups of observations.

### One synthetic NSCLC patient

Patient `P00001` is row 1, counting from zero. This patient's NLR is above the focal threshold.

| Column | Expected-focal dataset | Surprising-focal dataset |
|---|---:|---:|
| `patient_id` | P00001 | P00001 |
| `stage_iv` | 1 | 1 |
| `ecog_ps` | 0 | 0 |
| `ecog_ps_ge_2` | 0 | 0 |
| `crp_mg_l` | 17.970 | 17.970 |
| `crp_mg_l_ge_10` | 1 | 1 |
| `nlr` | 3.540 | 3.540 |
| `nlr_ge_3` | 1 | 1 |
| `brca2_mutation` | 0 | 0 |
| `research_marker_d_positive` | 0 | 0 |
| `research_signature_a` | 0.016 | 0.016 |
| `research_signature_b` | −0.936 | −0.936 |
| `research_signature_c` | −0.558 | −0.558 |
| **`log_pfs_months`** | **1.099** | **1.999** |

Only the outcome changes. The contribution from NLR ≥3 changes from −0.45 to +0.45 log months, increasing this patient's outcome by 0.90. The patient's other contributions and residual are identical. Patients with NLR <3 have no change from this reversal. PFS in months can be obtained by exponentiating `log_pfs_months`; there is no censoring in this release.

The new public task IDs are `37727c18694934f1` and `a0ef63f9837967a1`, respectively. Their source task IDs were `87b1f8e82cf42d71` and `dbce1befa80e2c19`. Always resolve tasks through the private `assignment.json`.

### One synthetic NSCLC cell line

Cell line `CL_00021` is row 21. It has TP53 loss and meets both conditions defining the focal subgroup.

| Column | Expected-focal dataset | Surprising-focal dataset |
|---|---:|---:|
| `cell_line_id` | CL_00021 | CL_00021 |
| `tp53_loss` | 1 | 1 |
| `msi_high` | 1 | 1 |
| `smarca4_loss` | 0 | 0 |
| `keap1_loss` | 0 | 0 |
| `research_signature_a` | 0.443 | 0.443 |
| `research_signature_b` | 1.210 | 1.210 |
| **`dependency_USP7`** | **0.542** | **−1.058** |
| `dependency_WRN` | −1.090 | −1.090 |
| `dependency_SMARCA2` | −0.306 | −0.306 |
| `dependency_NFE2L2` | −0.051 | −0.051 |

The TP53-loss contribution to the USP7 score changes from +0.8 to −0.8 within this subgroup, a decrease of 1.6. Other dependency outcomes stay the same. Cell lines outside the subgroup, or without TP53 loss, have unchanged USP7 scores.

The new public task IDs are `886c337297f1ac5d` and `d3b78f6e692740a2`; their source IDs were `b7b3e4fdcae53c2d` and `15a0a70fbdbaa8a6`. These are fully simulated cell lines and scores generated in a DepMap-style format.

## How the datasets are generated

**First, establish the variables and propose candidate discoveries.** Existing disease-specific samplers generate the covariates. The expected/surprising generator adds the constructed research signatures and markers, relevant molecular features, and threshold indicators such as NLR ≥3. An LLM receives the available variables and proposes explicit expected and neutral comparisons. Each proposal identifies an outcome, exposure, comparator, direction, and any population or subgroup restrictions.

**Second, retrieve and assess the literature.** The LLM supplies search queries, and Python executes them through Europe PMC. The workflow retains the queries, retrieval times, publication identifiers, abstracts, and available full-text excerpts. A review call assesses whether the retrieved studies support the proposed population, comparison, endpoint, assay, and direction. Expected candidates require a matching primary empirical source. Neutral candidates require a review of the underlying association, including whether it already carries a directional expectation outside the proposed subgroup.

**Third, check whether the comparison is realistic as encoded.** A separate LLM call sees the claim, variable meanings, and retrieved evidence, without the first reviewer's verdict. It identifies necessary restrictions and searches for counterexamples. Python then checks that the required restrictions are represented in the hypothesis. For instance, evidence from an EGFR-mutant treatment trial requires a corresponding biomarker restriction, and its control arm must match the encoded comparator. A column indicating absence of osimertinib does not identify a particular control regimen. An accepting review cannot override missing scope conditions.

Rejected candidates and the reasons for rejection go into the next proposal round. The example configuration permits eight rounds. Insufficient accepted candidates stop generation and leave a checkpoint for revision or resumption. The ordinary inventory requires four reviewed expected candidates and two neutral candidates. One expected candidate is deliberately inverted in both datasets to supply the fixed surprising discovery; another becomes the focal discovery. AML uses three expected candidates and three neutral candidates.

**Fourth, write the data-generating process (DGP).** Python compiles the accepted comparisons into an additive equation. Each endpoint has an intercept, a contribution from each applicable discovery, and a Gaussian residual. Conditions are restricted to explicit equality or numeric thresholds; the generator evaluates these using fixed Python operations.

For the clinical NSCLC example, the actual equation is:

```text
log_pfs_months = 2.5
  + b_NLR × I[NLR ≥ 3]
  − 0.45 × I[stage IV]
  − 0.45 × I[CRP ≥ 10]
  + 0.45 × I[ECOG ≥ 2]
  + 0.45 × I[BRCA2 mutation and A ≥ 0 and B ≥ 0.5]
  + 0.45 × I[marker D positive and A ≥ 0 and C ≤ 0.5]
  + residual

b_NLR = −0.45 in the expected-focal version; +0.45 in its partner.
I[condition] is 1 when the condition holds, and 0 otherwise.
The residual has mean 0 and standard deviation 0.30 log months.
```

A patient may meet several conditions and receive several contributions. The row shown above has stage IV disease, CRP ≥10, and NLR ≥3, so its expected-version mean is 2.5 −0.45 −0.45 −0.45 = 1.15, before adding the residual.

For the cell-line focal endpoint, the intercept is −0.25 and residual standard deviation is 0.18. TP53 loss adds +0.8 everywhere in the expected version. In its partner, it adds −0.8 within A ≥0 and B ≥0, and +0.8 outside that subgroup. The reversal changes this one contribution while retaining all other discovery terms and random draws.

**Finally, check the complete paired design.** A further LLM review assesses both equations alongside their evidence and the explicitly intended reversals. Numerical checks measure the actual group contrasts under the full DGP, including overlapping terms and correlated covariates. The focal effects must have the intended directions, adequate group sizes, and absolute-effect and signal-to-noise differences within 10% between versions. Other discoveries must retain their intended directions. Equal coefficients alone would not establish equally detectable group contrasts.

Calibration then uses 1,000 development replicates per version and a fixed reference search procedure with a multiple-comparison correction. Focal recovery must be at least 80% in each version, with a difference no greater than five percentage points. All ten v2 pairs passed these numerical checks. They support the comparison between paired focal discoveries; expected, neutral, and surprising background discoveries can still differ in prevalence and difficulty.

The [technical reference](EXPECTED_SURPRISING.md#generation-and-literature-review) specifies the review gates and calibration search space. The [frozen specifications](../benchmarks/expected_surprising/v2/specs) preserve each equation, discovery definition, and source-backed review. Replaying these specifications requires no further LLM calls or literature searches.

## What the agent sees and does

One run evaluates one provider/model on one assigned public task. It receives public instructions, the response schema, outcome scales and units, column summaries, a current claim/evidence ledger, its persistent research notes, latest stage narratives, and required assessments. Full prior JSON responses and failed attempts are retained in the audit transcript rather than repeated in every prompt. It receives no target inventory, literature categories, private DGP, selection pool, or unreleased selection. The full frame stays with the Python controller. There are no arbitrary-code, browsing, filesystem, or subagent tools in the evaluated harness.

The public instructions encourage diverse comparisons and evidence-informed follow-up throughout the run. They explain signed contrasts and ask the agent to use its own scientific judgment about effect sizes, uncertainty, and conclusions. They prescribe no minimum effect or numerical accept/reject rule. Public task metadata and every numerical result sent to the provider omit evaluator effect-size cutoffs, including cached, automatically delivered, and reoriented results. Numerical reference decisions and cutoffs remain private scoring rules.

| Stage | Agent action | Controller action |
|---|---|---|
| `explore` | Register comparisons, anticipated directions, initial assessments, and evidence-linked refinements | Freeze registration records |
| `analyze` | Select up to 12 existing short claim references in `run_analyses` | Execute discovery-sample comparisons; return 95% Welch intervals, cell sizes, and diagnostics |
| `appraise` | Assess every delivered discovery result, record investigation status, optionally request validation | Record decisions, then deliver requested or due scheduled evidence |
| `synthesize` | Assess new validation and explicitly reassess due original claims | Record immediate/delayed checkpoints, derive the accepted set, and carry state forward |

Every stage may propose comparisons for later analysis. Clinical and cell-line runs both have 25 iterations; the smoke protocol has six. The common budget gives both task types the same opportunity for exploration and follow-up. The controller assigns stable short references (`H1`, `H2`, …). Each proposal includes the comparison, anticipated direction and initial assessment in one object, without parallel dictionaries. A changed claim is a new proposal. Optional `parent` and `motivating_evidence` link refinements to earlier evidence. Python derives refinement types; prose is not scored.

Each stage has its own response form. Only analysis exposes `run_analyses`; only appraisal exposes `validate`, a single claim reference. There are no agent-authored iteration/stage stamps, executed-history lists, legacy decisions, or accepted-set lists. An assessment supplies `claim`, `status` (`accept`, `reject`, `unresolved`), and `investigation` (`active`, `deferred`, `closed`). Omitted claims retain their assessment. The controller derives the complete accepted set, including withdrawals and canonical deduplication, from these judgments.

The ledger displays all available direct evidence, oriented to each claim, under short `R1`, `R2`, … references. By omitting `evidence` from an assessment, the agent assesses the evidence shown on that claim's card; the controller attaches its references. The agent can explicitly select a subset or related evidence using short references. Required new or deadline evidence must still be included. Likewise, omitted `motivating_evidence` on a refinement attaches its parent's displayed evidence. Audit records distinguish controller-attached links from agent-selected links: attachment documents supplied evidence, not proof of attention or understanding. Validation requests automatically attach the valid discovery result for the selected comparison.

An optional `research_notes` field replaces the agent's persistent notebook; omission preserves it. The latest narrative for each stage is also retained. Retries show the unchanged ledger and the current error, without replaying failed responses. Failed attempts cannot change references, notes, assessments, or evidence budgets. Scientific judgments are never retried because the private evaluator disagrees.

See [the prompting refactor and examples](EXPECTED_SURPRISING_PROMPTING.md) for concrete forms and limitations.

Claim identity includes direction. Comparison identity ignores direction and normalizes exposure/comparator recoding. A family retains the endpoint, exposure levels, contrast, eligibility restrictions, and subgroup variable names. A new name earns no new claim, test, or sample credit. Opposite claims are distinct scientific decisions sharing one comparison's numerical evidence. The controller returns evidence oriented to each new opposite claim and explicitly marks its registration as already exposed to direct evidence. Related prior evidence is recorded separately from direct exposure.

Signed results are `direction × raw contrast`. A mean difference is exposed minus comparator within eligibility and subgroup. An interaction subtracts that difference in the subgroup complement. For example, a raw difference of −0.4 under direction −1 appears as +0.4. Cell sizes follow the indicated exposure/comparator ordering. Clinical PFS is fully observed and analyzed on the natural-log months scale; more negative raw dependency scores indicate greater dependency. The evaluator privately retains reference cutoffs of 0.10 clinical units and 0.15 dependency-score units for scoring, without prescribing these to the agent. Every comparison cell needs at least 20 observations.

### Validation delivery

The agent may make at most ten new voluntary requests per run, at most one per iteration, during appraisal after discovery evidence. The agent supplies `validate: "H1"` (for example); the controller records the claim and its triggering valid discovery result. Repeated requests for a cached comparison consume no new slot or sample.

| Policy | Automatic release iterations | Selection iterations | Response window |
|---|---|---|---|
| Clinical, 25 iterations | 5, 12, 20 | 4, 11, 19 | Two subsequent iterations |
| Cell-line, 25 iterations | 3, 6, 8 | 2, 5, 7 | Two subsequent iterations |
| Smoke, six iterations | 2, 4 | 1, 3 | Two subsequent iterations |

Selection occurs before appraisal one iteration before release. Eligible comparisons have a valid discovery test and have neither received independent evidence nor been selected. Selection rotates through supported, excluded, and ambiguous exploratory evidence relative to the public claim. It uses the first validly analyzed claim for each comparison, a seeded order within the desired stratum, then a seeded fallback over the full eligible pool. An empty pool creates an unavailable slot. It never selects an unproposed target. Private logs contain the complete pool, reason, first analyzed claim, selection time, and deadline. Selection does not generate or inspect validation samples.

A voluntary request before the deadline satisfies a selected slot and cancels duplicate automatic delivery. Otherwise the controller releases the fixed claim's result after appraisal at the deadline, even when appraisal exhausts its contract retries. Voluntary requests and automatic deliveries share a comparison cache. There are at most 13 distinct validation comparisons in a full run, or 12 in the six-iteration smoke policy. Every validation interval uses `alpha = 0.05 / V_max`, including cached/oriented results and unused slots.

Sample seeds are derived from SHA-256 of `es-workflow-v1:PAIR_ID:REPLICATE_ID:NAMESPACE:COMPARISON_KEY` (first four digest bytes, little endian). The independent-validation namespace differs from final confirmation, calibration, and discovery. Use the **same replicate ID for both versions** so corresponding comparisons share covariate and residual streams. Run IDs and output directories remain distinct. The selection namespace additionally includes its configured seed and slot. All state, including registrations, assessments, results, histories, request slots, caches, and response events, participates in stage rollback; retries cannot obtain fresh evidence.

### Workflow versions

| Contract | Version |
|---|---|
| Workflow | `appraisal-3.2.0` |
| Public prompt | `ledger-1.0.0` |
| Response schema | `stage-forms-1.0.0` |
| Validation policy | `delayed-3.0.0` |
| Scoring | `profile-3.0.0` |
| Numerical DGP | `2` |

Public tasks carry version identifiers. Actual schedules are evaluator configuration in private `workflow.json`; full-budget runs read this packaged policy. The six-iteration smoke override is explicit in its saved configuration and manifest. Reports and transcripts retain versions, resolved policy, replicate identity, dataset/package hashes, and retry settings. Changing the declared selection-stratum order or two-iteration response rule requires a policy version change. Custom schedules are explicit development configurations, validated to leave full response windows, and must match within evaluated pairs.

## Which LLM endpoints can be used

The workflow uses the existing [provider registry](../src/onc_co_scientist/providers/registry.py). Any compatible `LLMProvider` implementation can supply structured stage responses; the controller and scorers do not depend on vLLM.

| Provider kind | Configuration |
|---|---|
| `vllm_openai` | `base_url`, `model_id`, `api_key`, `timeout_s` |
| `gemini_vertex` | `model_id`, `project_id`, `location`, provider-specific transport and reasoning settings |
| `anthropic_vertex` | `model_id`, `project_id`, `region`, provider-specific transport settings |

The [vLLM configuration](../configs/expected_surprising.vllm.yaml) is the test-server example. The six-iteration smoke runner verifies `/v1/models` and records the returned inventory before submitting evaluations. The tested endpoint is `http://sn4622130540:8000/v1`; its served model must be checked at execution time. Other providers can use their own configuration with the generic `ocs expected-surprising run` command. The public workflow is identical across providers.

The initial implementation retains 125,000 completion tokens per call (including reasoning), two technical retries per stage, and a 1,800-second test-server timeout. This is a per-call allowance, not a run budget. Provider transport retries remain separate from contract retries. Models need sufficient context for the current ledger, notes, and configured output allowance. The ledger grows with distinct claims and evidence, rather than with every repeated response.

## How the agent is scored

Scores remain separate: **D** measures discovery performance and is the overall ranking number; **E** measures coverage and how early it occurs; **B** measures delayed response to validation conditional on the evidence encountered. The paired difference in independently confirmed focal recovery remains the principal expected-versus-surprising endpoint.

### Discovery performance, D

Exact target recovery requires agreement on endpoint, exposure/comparator, contrast, direction, eligibility, and all subgroup conditions. Near recovery permits omission of exactly one subgroup variable from a target containing at least two, with all remaining fields and retained conditions matching. Assignment counts each target once, prioritizing exact matches. Exposure recodings and duplicate names cannot add credit.

For each literature category `c`, `R_c` is accepted, independently confirmed exact target discoveries divided by that category's actual embedded target count. Primary credit additionally requires a valid discovery-sample test of the comparison; an opposite direction may reuse that test. Then:

```text
R = (R_expected + R_neutral + R_surprising) / 3
Q = confirmed final accepted claims with a valid discovery test / all final accepted claims
D = 100 × 2RQ / (R + Q)
```

Each accepted canonical directional claim enters Q once. Untested acceptances stay in its denominator and cannot enter its numerator. Confirmed additional claims outside the six-target inventory can enter its numerator. With no accepted claims, Q is unavailable and D is zero. When R + Q is zero, D is zero. For example, category recovery of 0.75, 0.40, and 0.65 gives R = 0.60; Q = 0.90 gives D = 72.

Final confirmation uses a further independent sample and `alpha = 0.05 / M` across all distinct final accepted claims, including untested ones. Additional claims receive full-DGP adjudication in a 100,000-row conditional-mean reference sample, with Monte Carlo uncertainty. Unconfirmed is not a synonym for false. The report separately counts acceptances whose final interval excludes the minimum effect, unconfirmed claims, and failures to accept supported discoveries.

The report retains R_c, R, Q, claim counts, individual confirmation results, and a secondary exact-or-near D using the same Q. An exhausted stage error makes primary D and focal recovery zero; partial results remain diagnostic. Recovered technical errors remain eligible under the declared retry policy. The strict first-attempt sensitivity endpoint assigns zero after any technical repair.

### Exploration coverage, E

At the end of every iteration, measure each category's cumulative fraction of exact target comparisons validly tested, ignoring direction. E is 100 times the mean of those three category coverages, averaged over the **configured** iteration budget. Coverage achieved at iteration one contributes throughout the run; later coverage contributes to fewer positions.

The denominator never shrinks to an observed shorter run. If a run ends early, achieved coverage carries through remaining scheduled positions without new discoveries. Invalid tests, repeated analyses, and aliases add no coverage. Reports include category curves, final coverage, exact-or-near secondary curves, and per-stage counts of distinct claims, comparisons, and families. E describes the embedded target inventory's coverage, not the scientific value of every proposed comparison.

### Evidence responsiveness, B

For a valid independent interval on the signed claim scale, the private reference is accept when its lower bound exceeds positive delta, reject when its upper bound is below delta, and unresolved when the interval contains or touches delta. An interval below the private cutoff does not assert a null effect or establish the opposite direction. Agents judge importance themselves, so this score measures agreement with the evaluator’s fixed reference, rather than proving whether their scientific judgment was correct. The agent’s narrative is retained for interpretation and is not quantitatively scored.

Each **first validation delivery** creates one event for the claim fixed before that evidence. Its immediate synthesis response is descriptive. The principal response is its explicit assessment in synthesis at the second subsequent iteration: evidence at iteration two has its deadline at iteration four. Initial caution followed by a matching delayed assessment earns full credit. Refinements do not replace the required original claim assessment. Cache re-delivery, aliases, and post-evidence opposite directions cannot add response events.

```text
A_supported = delayed accept decisions / eligible supported events
A_excluded  = delayed reject decisions / eligible excluded events
A_ambiguous = delayed unresolved decisions / eligible ambiguous events
B = 100 × (A_supported + A_excluded + A_ambiguous) / 3
```

A missing due decision in an otherwise complete window scores zero and remains a visible protocol error. Invalid results, late events with insufficient scheduled iterations, and interrupted windows are excluded and counted separately. If any evidence class is absent, complete B is unavailable; observed component accuracies and denominators remain visible without reallocating missing weights. A run containing only supported events cannot receive complete B. A reporting unit with no events has an unavailable score, not a perfect score or a measured zero.

Reports separate voluntary/automatic sources, immediate/delayed assessments, matched literature categories, and the agent's recorded expectation alignment. Unmatched claims have no literature category. Directly exposed opposite-claim registrations are marked and must not be interpreted as pre-evidence expectations. All eligible additional comparisons participate in overall B.

### Investigation and validation choice

Every first valid discovery result and every first validation delivery has a two-iteration follow-up record. It retains distinct newly tested comparisons, structural refinements linked to evidence, opposite registrations/acceptances reusing evidence, validation by source, investigation/assessment transitions, repeated analyses, new families, target coverage, and overlapping window IDs. Late/incomplete windows remain visible. Counts are descriptive; windows overlap and are not independent experimental units.

Conditional rows include evidence class, estimate and interval normalized by delta, comparison-cell sizes, subgroup/eligibility complexity, remaining iteration and request budgets, literature category, and expectation alignment. Conditional cells summarize investigation and acceptance changes and subsequent validation within source, category, evidence, expectation, group-size, complexity, and remaining-budget strata. Exploratory/validation disagreement is explicit. These are secondary descriptive comparisons, not causal effects.

Validation choice has two separate denominators: available frozen scheduled opportunities requested before automatic delivery; and tested canonical claims at their first acceptance whose validation was voluntarily requested and received beforehand. First-acceptance histories freeze before subsequent same-stage requests/deliveries. Reports partition acceptance before any independent evidence, after voluntary evidence, or after automatic evidence; they retain requests before acceptance with later receipt, requests after acceptance, unmatched claims, and untested first acceptances. Withdrawal and reacceptance cannot rewrite that history. More requests have no automatic score bonus.

Target milestones separately retain first valid test, unexpected evidence encountered, validation requested/received, first correct direction, first acceptance, and independent confirmation. These distinguish search failures from response failures without forcing a run into one explanation.

### Paired aggregation

Compute D and E within each run, average replicates within each base-dataset version, weight the two versions equally within the base dataset, and then weight base datasets equally. Reports include clinical and cell-line strata. The aggregate also retains component recovery, confirmed-claim fractions, and completion rates; it does not recompute D from pooled hypotheses.

Aggregate each responsiveness evidence-class accuracy through the same hierarchy using available cells within that class. Then average the three aggregate class accuracies for B. Counts identify eligible runs, version cells, base datasets, and events for each class. Aggregate B may be available when individual-run B is unavailable, but all three aggregate classes are required. Missing classes keep their missingness in cluster bootstrap resamples.

The paired surprise effect is `100 × (surprising focal recovery − expected focal recovery)`, in percentage points. Both absolute rates accompany it. Analogous paired results describe valid focal testing and acceptance after supportive focal validation, disclosing the eligible opportunity pools. Poor recovery in both conditions is not strong performance merely because the difference is zero.

Bootstrap base datasets within modality, retaining paired versions, replicates, and events together. Confidence intervals are unavailable when there are too few eligible base datasets. The summary rejects mixed historical/new scores, different workflow versions, mixed models/harnesses, mismatched pair policies/budgets, and mismatched replicate identities. Summarize each model separately.

## Running and inspecting an evaluation

Use Python 3.12+ and an editable installation. In this checkout the dedicated environment is `/tmp/ocs-es-refactor-venv`; it is temporary. From the repository root:

```bash
/usr/bin/python3.12 -m venv /tmp/ocs-es-refactor-venv
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python -m pip install -e '.[dev,vllm-openai]'
```

Inspect an existing environment before reusing it. New task packages are generated from the **existing frozen bytes**, with no candidate generation, literature calls, DGP selection, or numerical replay:

```bash
/tmp/ocs-es-refactor-venv/bin/ocs expected-surprising repackage \
  data/expected_surprising_v2 data/expected_surprising_ledger_25_iterations
```

The destination must be new. The exporter preflights source hashes against the private assignments, copies all private provenance, copies each original Parquet/dictionary, changes public instructions/schema/metadata, and verifies copied hashes. `package_manifest.json` links every new ID, original ID, source hash, package hash, frozen specification, and policy. The [tracked package inventory](../benchmarks/expected_surprising/workflow_v3_2/package_manifest.json) records this implementation's 20 tasks. The data and raw runs under `data/` remain excluded from Git. A fresh checkout must restore those artifacts or replay the numerical release with its frozen dependency versions and verify the release hashes before repackaging.

To run one member of a pair through any configured provider:

```bash
/tmp/ocs-es-refactor-venv/bin/ocs expected-surprising run \
  data/expected_surprising_ledger_25_iterations/private/es-v2-nsclc_clinical-42000/pair.json \
  data/expected_surprising_ledger_25_iterations/public \
  configs/expected_surprising.vllm.yaml \
  data/expected_surprising_ledger_25_iterations/runs/nsclc-surprising-r0 \
  nsclc-surprising-r0 --version surprising --iterations 6 --replicate-id r0
```

Run the expected member with a distinct output path/run ID, `--version expected`, and the **same** `--replicate-id r0`. Omit `--iterations` for the full task budget and packaged policy. Supply a different provider configuration to evaluate a different backend. Explicit development policy overrides use `validation_policy` in the configuration. The private evaluator specification is never included in model prompts.

For the required four-run smoke protocol, first verify the model using `/v1/models`, then:

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python \
  scripts/expected_surprising/smoke_vllm.py \
  --base-url http://sn4622130540:8000/v1 --model MODEL_ID_FROM_SERVER \
  --data data/expected_surprising_ledger_25_iterations \
  --out data/expected_surprising_ledger_25_iterations/smoke/NEW_RUN_ID \
  --iterations 6 --workers 2 --max-tokens 125000 \
  --max-retries-per-stage 2 --timeout-s 1800
```

The runner records the explicit smoke schedule, shared replicate ID, server inventory, dependencies, source hashes, package-manifest hash, and task hashes. Each completed run should have 24 successful stages. Per-call metadata retains token usage, finish reason, reasoning size, latency, and errors. It writes per-run reports/transcripts, `summary.json`, `paired_summary.json`, and `SMOKE_REPORT.md`. The Markdown report uses recall (R), precision (P), and F1\* for the main outcomes; the underlying archived JSON retains Q for P and D for F1\*. The asterisk identifies the category-balanced recall and confirmation-based precision definitions. The report also explains that independent validation can occur after acceptance, and that two iterations specify the reassessment delay rather than the total run length. Settings and detailed counts are kept in an expandable section. Run the independent audit and regenerate a tracked report with:

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python \
  scripts/expected_surprising/audit_workflow.py \
  --data data/expected_surprising_ledger_25_iterations \
  --smoke data/expected_surprising_ledger_25_iterations/smoke/NEW_RUN_ID \
  --out benchmarks/expected_surprising/workflow_v3_2/smoke_audit_qwen.json
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python \
  scripts/expected_surprising/report_smoke.py \
  --smoke data/expected_surprising_ledger_25_iterations/smoke/NEW_RUN_ID \
  --audit benchmarks/expected_surprising/workflow_v3_2/smoke_audit_qwen.json \
  --out benchmarks/expected_surprising/workflow_v3_2/smoke_qwen.md --archive
```

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python -m pytest -q \
  tests/test_expected_surprising.py tests/test_expected_surprising_workflow.py \
  tests/test_expected_surprising_prompting.py
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python -m ruff check \
  src/onc_co_scientist/expected_surprising scripts/expected_surprising \
  tests/test_expected_surprising.py tests/test_expected_surprising_workflow.py \
  tests/test_expected_surprising_prompting.py
/tmp/ocs-es-refactor-venv/bin/ocs expected-surprising summarize \
  data/expected_surprising_ledger_25_iterations/runs \
  data/expected_surprising_ledger_25_iterations/paired_summary.json
```

The new acceptance fixtures exercise broad exploration, persistence with contradicted claims, accepting everything, requesting but ignoring validation, responding to automatic results without requests, no eligible comparisons, and initial caution resolved by the deadline. They also check chronology, caching and recoding, sample bounds, retries, aliases, score arithmetic, missingness, and aggregation. Historical tests remain in place.

| Artifact | Location |
|---|---|
| Public package | `data/expected_surprising_ledger_25_iterations/public/TASK_ID/` |
| Frozen evaluator specification, assignment, policy, and copied calibration | `data/expected_surprising_ledger_25_iterations/private/PAIR_ID/` |
| Package/source hash mapping | `data/expected_surprising_ledger_25_iterations/package_manifest.json` |
| Model prompts/responses, successful stages, technical failures, private selection audit | Run `transcript.jsonl` (evaluator-only complete transcript) |
| Discovery performance, exploration coverage, evidence responsiveness, components, validation cache/requests/opportunities, full structured state | Run `report.json` |
| Model-level paired and modality results | `paired_summary.json` |
| Frozen numerical release and historical reports | [v2 release](../benchmarks/expected_surprising/v2/README.md) |
| Current prompting and verification | [workflow v3.2 record](../benchmarks/expected_surprising/workflow_v3_2/README.md) |
| Previous agent-judgment verification | [workflow v3.1 record](../benchmarks/expected_surprising/workflow_v3_1/README.md) |

The implementation is in [packaging.py](../src/onc_co_scientist/expected_surprising/packaging.py), [workflow.py](../src/onc_co_scientist/expected_surprising/workflow.py), [evaluation.py](../src/onc_co_scientist/expected_surprising/evaluation.py), [scoring.py](../src/onc_co_scientist/expected_surprising/scoring.py), [events.py](../src/onc_co_scientist/expected_surprising/events.py), and [summary.py](../src/onc_co_scientist/expected_surprising/summary.py). [rollout.py](../src/onc_co_scientist/expected_surprising/rollout.py) dispatches versioned tasks and retains historical execution; original `StageRecord`, scoring, generation, and historical summary readers remain available.

For the archived voluntary-only workflow, use original public packages and the [legacy configuration](../configs/expected_surprising.vllm.legacy.yaml). Its omission of new workflow versions preserves the historical CLI path; do not combine its reports with new workflow scores.
