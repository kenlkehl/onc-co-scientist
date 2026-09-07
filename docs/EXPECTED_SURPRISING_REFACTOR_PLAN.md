# Expected/surprising workflow and scoring refactor

**Status: proposed implementation plan, September 7, 2026.** This document specifies a new workflow and scoring version. The [current implementation guide](EXPECTED_SURPRISING_GUIDE.md) describes the existing behavior. This planning change does not modify the running harness, datasets, or saved scores.

The premise is: an ideal agent explores diverse, plausible hypotheses—including unconventional ones—and adapts its search as evidence accumulates. It keeps unexpected results under consideration, requests further validation when uncertainty warrants it, and accepts a surprising association when the accumulated evidence supports it. Acceptance concerns the association in the supplied dataset; biological interpretation and generalizability can remain uncertain.

The refactor will measure search coverage, changes in investigation after evidence, requests for validation, and resolution after validation. Python will score structured actions and decisions. Prose explanations will be retained for inspection and will not enter the quantitative scores.

The report will contain three complementary scores: discovery performance (`D`), exploration coverage (`E`), and evidence responsiveness (`B`). The paired difference in confirmed focal recovery remains the principal expected-versus-surprising comparison. Use `D` when one overall ranking number is needed. [Section 10](#10-comprehensive-scores-and-aggregation) defines the formulas, denominators, and aggregation rules; validation requests and investigation after evidence provide supporting behavioral measurements.

## Start here in a fresh chat

This is an implementation brief for the repository at `/data1/ken/onc-co-scientist`. It can be used without the earlier conversation. Use the `expected-or-surprising` checkout containing this plan and its linked source, tests, configurations, and frozen release specifications. Inspect the current state before changing files and preserve any subsequent unrelated edits.

The generated datasets and raw run artifacts under `data/` are excluded from Git. They are available in the current working checkout but will not accompany a clean clone or a new worktree automatically. Prefer the existing checkout. Elsewhere, restore those local artifacts or replay the frozen release specifications using the recorded dependency versions and verify the original dataset hashes before repackaging. Source and documentation commits do not archive the live vLLM service or the temporary Python environment.

The [current guide](EXPECTED_SURPRISING_GUIDE.md) supplies dataset examples and existing contracts. The [implementation notes](EXPECTED_SURPRISING_IMPLEMENTATION.md) and [previous smoke report](../benchmarks/expected_surprising/v2/smoke_vllm_20260907_125k_retries.md) describe completed work. This plan specifies the intended new workflow; descriptions of mandatory numerical decision rules and immediate-only responsiveness in those historical documents describe the old behavior. The separate named/masked task is outside this refactor's scope.

The prior smoke batch completed four three-iteration NSCLC runs with scorable outputs and no unrecovered protocol errors. It used the old prompt and voluntary-only validation. It did not test scheduled validation, delayed response scoring, or `D`, `E`, and `B`. The documented 40 passing benchmark tests are a historical baseline, not verification of this refactor.

### Inputs and output locations

| Item | Location or instruction |
|---|---|
| Source and benchmark tests | `src/onc_co_scientist/expected_surprising/`, `tests/test_expected_surprising.py` |
| Frozen release and dependency provenance | [Release inventory](../benchmarks/expected_surprising/v2/README.md), [release manifest](../benchmarks/expected_surprising/v2/release_manifest.json), `benchmarks/expected_surprising/v2/specs/` |
| Existing numerical datasets and private evaluator inputs | `data/expected_surprising_v2/public/` and `data/expected_surprising_v2/private/` |
| Clinical NSCLC pair | `data/expected_surprising_v2/private/es-v2-nsclc_clinical-42000/pair.json` |
| Cell-line NSCLC pair | `data/expected_surprising_v2/private/es-v2-nsclc_depmap-42005/pair.json` |
| Mapping from versions to task directories and dataset hashes | `assignment.json` next to each private `pair.json`; read this mapping rather than guessing task IDs |
| Provider configuration | [configs/expected_surprising.vllm.yaml](../configs/expected_surprising.vllm.yaml) |
| New task packages and runs | Use a separate root such as `data/expected_surprising_workflow_refactor/`, with `public/`, `private/`, and new run directories |

Repackage all 20 existing task instances for the new workflow, copying their original Parquet bytes and retaining the evaluator's private pair/assignment structure. Change public instructions, response schemas, and versioned task metadata. Record hashes linking each new package to its source dataset. Copying a private specification into the new evaluator root does not make it agent-visible. The implementation needs a packaging entry point that performs this operation without rerunning candidate generation, literature review, DGP selection, or numerical generation.

Use the schedules and scoring weights stated below as the initial implementation defaults. Expert adjudication and formal evaluation locking remain later research steps; they do not block implementing and checking this development workflow. Choose explicit new workflow/prompt/schema/policy/scoring version IDs and record them in packages, configurations, transcripts, and reports. Numerical DGP version `v2` stays unchanged. Routine field names and configuration layout may be chosen during implementation and documented.

### Python setup and checks

The package requires Python 3.12 or newer. The earlier `/tmp/ocs-expected-surprising-venv/bin/python` works in the current session but is temporary and has no `ocs` console script. A new chat can create a dedicated environment if needed; use an unused path or inspect an existing environment before reusing it:

```bash
cd /data1/ken/onc-co-scientist
/usr/bin/python3.12 -m venv /tmp/ocs-es-refactor-venv
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python -m pip install -e '.[dev,vllm-openai]'
```

The editable install supplies `ocs` in that environment's `bin` directory. Install the frozen release's dependency versions if numerical replay is needed; ordinary repackaging copies existing dataset bytes. Run the relevant checks after implementation, including any newly added test modules:

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python -m pytest -q tests/test_expected_surprising.py
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python -m ruff check src/onc_co_scientist/expected_surprising scripts/expected_surprising tests/test_expected_surprising.py
```

### Live verification after implementation

The configured endpoint is `http://sn4622130540:8000/v1`; the prior served model was `Inferact/Qwen3.8-27B-NVFP4`. Check `/v1/models` before a new run and use the returned model ID. Endpoint availability and the served model can change. Keep the 125,000 completion-token per-call allowance, two retries per stage, and 1,800-second timeout. This token allowance includes reasoning; it is not a total-run token limit.

Update the smoke runner to consume the new packaged workflow/policy and emit the new reports before running it. For the six-iteration smoke policy, automatic release is in iterations 2 and 4, with a two-iteration response window. These settings must be explicit in the saved manifest. The current runner does not implement them, and changing its iteration flag alone does not enable the refactor.

After that update, retain the existing CLI shape below. Replace `MODEL_ID_FROM_SERVER` and `NEW_RUN_ID` with the verified model and a new output directory name. The default profiles run clinical and cell-line NSCLC, both expected and surprising versions:

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-es-refactor-venv/bin/python \
  scripts/expected_surprising/smoke_vllm.py \
  --base-url http://sn4622130540:8000/v1 \
  --model MODEL_ID_FROM_SERVER \
  --data data/expected_surprising_workflow_refactor \
  --out data/expected_surprising_workflow_refactor/smoke/NEW_RUN_ID \
  --iterations 6 --workers 2 --max-tokens 125000 \
  --max-retries-per-stage 2 --timeout-s 1800
```

Implementation is complete when the section 11 checks pass, all 20 new task packages preserve their dataset hashes, and the four new smoke runs have been executed and audited. Deliver the updated guide, runnable packaging/evaluation commands, configuration and policy versions, per-run transcripts and reports, paired summary, and a Markdown smoke report linking those artifacts. Each completed smoke run should contain 24 successful stages. Report recovery failures and unavailable score components honestly; neither is by itself a software failure. An exhausted protocol error needs investigation, and a blocked endpoint must be reported as an outstanding live check rather than a completed smoke test.

A starting message for the new chat is:

> Implement `docs/EXPECTED_SURPRISING_REFACTOR_PLAN.md` end to end in the existing `/data1/ken/onc-co-scientist` working checkout on `expected-or-surprising`. Use the plan's startup instructions and defaults, preserve existing work and frozen data, run the specified checks and four six-iteration vLLM smoke evaluations, and deliver the updated documentation and reports. The plan supplies the required context; inspect its linked repository sources as needed.

## 1. Retain the paired datasets and distinguish the new experiment

Keep the current v2 discovery datasets, numerical DGPs, six-discovery inventories, and one focal reversal per pair. Preserve the literature, realism, and numerical calibration records. Each run still receives one public dataset; the paired assignment, target inventory, literature classifications, and DGP remain private.

Introduce separate workflow, prompt, schema, validation-policy, and scoring version identifiers. Write new public task packages referencing the existing dataset bytes, with new instructions and response schemas. Preserve the old packages, transcripts, and scores. Removing a decision instruction from an old transcript cannot reproduce the behavior of an agent that never received it.

Keep the current analysis services, modality-specific iteration budgets, maximum 12 discovery-sample analyses per iteration, 125,000-token per-call allowance, and two technical retries per stage for the first implementation. New policy settings must match between members of an evaluated pair. Numerical benchmark development and expert adjudication of the discovery classifications remain separate from this workflow refactor.

The full task budgets are 25 iterations for clinical datasets and 10 for cell-line datasets. The clinical endpoint is fully observed log PFS in months with minimum effect `delta = 0.10`; dependency-score claims use `delta = 0.15`, with more negative scores indicating greater dependency. Discovery-sample analyses retain 95% Welch intervals and require at least 20 observations in every comparison cell. The agent uses structured hypothesis registration and controller-executed analysis/validation services; adding arbitrary code execution or literature-search tools to the evaluated agent is outside this refactor. Dataset-generation literature searches remain part of the existing, separately documented generation pipeline.

## 2. Change what the agent is asked to do

Ask the agent to consider diverse comparisons, use previous results to allocate further investigation, assess discoveries in the supplied data, and use independent validation when useful. Allow broad exploration and focused follow-up throughout the run. Do not impose a broad-early/narrow-late schedule or quotas for expected and surprising hypotheses.

Keep the outcome scale, the meaning of each contrast, and the minimum effect defining the claim public. Keep clear explanations of the returned signed estimates. Remove the numerical instructions that say when to accept, reject, or remain unresolved. These instructions currently appear in both the generated `instructions.md` and `rollout.py`; changing only one would leave the other in the prompt.

The agent should know that it may receive evaluator-scheduled validation during the run. Individual automatic selections and their deadlines remain private until evidence is released. Neither validation responses nor retry feedback disclose target matches, literature categories, evaluator evidence classifications, or score correctness.

## 3. Reorder each iteration so validation follows appraisal

Retain four LLM stages, with service execution between them:

| Stage | Agent action | Controller action |
|---|---|---|
| Explore | Register new hypotheses or refinements, anticipated directions, initial assessments, and links to earlier results that motivated them | Validate and freeze the registered comparisons |
| Analyze | Select previously registered hypotheses for up to 12 discovery-sample analyses | Execute comparisons and return estimates, intervals, group sizes, and diagnostics |
| Appraise and request | Assess each new result; record whether investigation remains active, deferred, or closed; optionally request validation | Record decisions before releasing requested or due scheduled validation |
| Synthesize | Assess newly released validation, reassess hypotheses whose response window ends, and report the current accepted discovery set | Preserve the complete state and continue to the next iteration |

This moves the voluntary validation decision out of the analysis stage. The agent can first see and appraise the exploratory evidence. Requested validation becomes available before synthesis in that iteration. Scheduled validation follows the delayed policy below.

Every stage may register hypotheses for later analysis. A changed hypothesis receives a new ID. A direction reversal is recorded explicitly: rejecting a negative-direction claim and accepting its positive-direction counterpart are distinct decisions. They concern the same numerical comparison and can reuse its evidence.

Require assessments for newly delivered results and explicit reassessment at response deadlines. A valid scientific decision is never retried because the evaluator disagrees with it. Missing fields, invalid IDs, malformed JSON, and nonexistent variables receive ordinary contract feedback and bounded retries.

## 4. Extend the structured records

Retain the existing hypothesis, result, parent, decision, and accepted-ID fields. Add or formalize:

| Record | Required information |
|---|---|
| Initial expectation | Immutable anticipated direction and assessment at registration, before numerical results for that hypothesis are delivered |
| Assessment | Hypothesis ID, evidence/result IDs considered, accept/reject/unresolved status, active/deferred/closed investigation status, stage and iteration |
| Refinement | Parent hypothesis ID and previously available result IDs motivating the new comparison |
| Validation request | Registered hypothesis/comparison ID, triggering result IDs, and request time |
| Validation opportunity | Selection time, eligible pool, selection stratum, cached comparison key, release deadline, and whether evidence was obtained voluntarily or automatically; evaluator-only until release where appropriate |
| Response checkpoint | Immediate post-validation assessment and the assessment at the end of the two-iteration response window |

Validate that cited results were available before the action. Derive refinement types from the changed structured fields: direction change, cutoff change, subgroup-condition addition/removal, or another comparison change. Retain self-reported motivations but do not score their prose quality.

Use canonical comparison keys to prevent new IDs or reversed exposure coding from creating new hypotheses, analysis credit, or validation samples. Keep claim identity, comparison identity without direction, and broader hypothesis-family identity as separate concepts. A newly registered opposite-direction claim may already have numerical evidence from its counterpart. Mark that exposure explicitly; its registration assessment cannot count as a pre-evidence expectation. For refinements, distinguish an expectation recorded before the first direct test from knowledge of related, previously tested comparisons.

## 5. Provide voluntary and scheduled validation

### Voluntary requests

Retain up to ten voluntary validation requests per run, with at most one new request per iteration. A request must concern a registered hypothesis with a valid discovery-sample analysis. The agent selects the hypothesis after seeing exploratory results. A repeated request for the same canonical comparison returns the existing evidence and consumes no new sample opportunity.

### Scheduled opportunities

Proposed initial schedules are automatic release in clinical iterations 5, 12, and 20, and cell-line iterations 3, 6, and 8. A six-iteration smoke protocol uses iterations 2 and 4. These are development defaults to freeze before the new formal runs; they leave two later iterations for scoring response and subsequent exploration.

One iteration before each release, select one already registered, validly analyzed comparison that has not received independent validation or previously been selected. Freeze the selection before the agent's appraisal/request stage. The agent can request its result in that iteration or before automatic release in the next. If it does, mark the opportunity as voluntarily obtained and cancel the later duplicate delivery. Otherwise, deliver it automatically at the deadline.

Selection must be independent of private discovery membership, literature labels, and unseen validation outcomes. Rotate the desired selection stratum across exploratory evidence supporting the claimed minimum effect, excluding that minimum effect, and remaining ambiguous. Use a seeded order of canonical comparison keys within the stratum, with a seeded fallback over all eligible comparisons. Freeze this algorithm and log the complete candidate pool and selection reason. If the pool is empty, record an unavailable opportunity; do not introduce a target hypothesis the agent never proposed.

The selection strata use only the exploratory estimates, intervals, and public claim definitions available at that time. They serve sampling balance, not a verdict that a particular hypothesis ought to have been validated.

### Sample identity and accounting

Cache one independent sample/result per canonical comparison, shared by its voluntary and automatic delivery routes and by both claimed directions. Normalize exposure/comparator recodings when presenting oriented results. A subgroup refinement is a different comparison and may use another sample.

Retain separate counters for voluntary request slots and scheduled slots. Three scheduled slots plus ten voluntary slots imply at most thirteen distinct validation comparisons in a full run. If the agent requests a selected scheduled comparison before its deadline, its request consumes a voluntary slot and satisfies that scheduled slot; no second sample is released. The schedule does not create a requirement to fill unused slots.

Use validation intervals with error probability `0.05 / V_max`, where `V_max` is the configured upper bound on distinct validation comparisons, including automatic delivery. Preserve the same interval regardless of delivery route or how many slots ultimately go unused. The previous fixed 99.5% interval must therefore become a policy-derived value.

Seed samples from immutable run/replicate identity and canonical comparison identity, in a namespace separate from discovery, calibration, and final confirmation. Freeze and log the seed derivation. Use a shared paired-replicate seed convention where corresponding comparisons in paired runs should share covariates and residual streams.

Scheduled evidence changes the research environment, so the results characterize this assisted workflow. It prevents an agent from declining evidence on a selected registered comparison; it does not guarantee that different agents investigate the same comparisons. A separate fixed-hypothesis probe would be needed for an identical-comparison responsiveness experiment across all agents.

## 6. Define the evidence reference and response window

For a valid independent validation interval on the signed claim scale, retain the following private evidence classification:

| Independent evidence | Evaluator reference decision |
|---|---|
| Lower bound greater than the endpoint's positive minimum effect | Accept |
| Upper bound less than that minimum effect | Reject the minimum-effect claim |
| Interval contains or touches the minimum effect | Unresolved |

Use the existing clinical and dependency minimum-effect definitions. Rejecting the minimum-effect claim does not assert that the association is exactly zero or establish its opposite direction. Invalid or undersized analyses have no reference decision. Classification depends on the claim and received numerical evidence, not whether the discovery is expected or surprising.

This is a prespecified evidence-agreement measure. It does not establish that one confidence-interval rule is the uniquely rational response to every possible research history. The reference uses independent evidence for the unchanged comparison, avoiding a naive pooled estimate that treats an adaptively selected exploratory result as confirmatory evidence. Report exploratory/validation disagreement explicitly. The later checkpoint measures whether an agent resolves its initial caution while seeing this result and any further investigation; it does not by itself measure the quality of a full Bayesian synthesis. Independent validation comes from the same synthetic generating setting, so it supplies no evidence of biological mechanism or generalizability to another population.

Record the immediate response descriptively. The principal response endpoint is the agent's explicit assessment at the end of the second subsequent iteration. This gives an initially cautious agent time for follow-up. It must continue to assess the original claim; a refined claim has its own identity and may additionally earn discovery credit.

For each event with a complete response window, score 1 when that assessment agrees with the independent-evidence reference and 0 otherwise. Report acceptance of supported claims, rejection of contradicted claims, and uncertainty for ambiguous claims separately, as well as changes from the pre-validation assessment. These three evidence classes receive equal weight in the balanced responsiveness score defined in section 10. A missing due decision scores zero; it is not silently replaced with the prior assessment. Keep protocol errors visible.

Late voluntary requests still return evidence. Report their immediate responses separately when two later iterations are unavailable. An interrupted window is marked incomplete. Neither late events nor invalid results enter the principal complete-window accuracy denominator. Report all exclusions and opportunity counts. No eligible events means an unavailable score, not perfect performance or a measured zero.

Record literature classification and the agent's own anticipated direction separately. Initial expectations describe what the agent recorded after seeing task information and column summaries; they are not a direct measurement of inaccessible internal beliefs.

## 7. Score coverage and changes in investigation

### Coverage

At every stage, report new and cumulative distinct hypotheses, validly tested comparisons, and hypothesis families. Use direction-independent exact/near target matching to measure whether expected, neutral, and surprising target comparisons have been reached. Keep first test, first correctly directed hypothesis, first acceptance, and independent confirmation as separate milestones.

Preserve exact/near definitions: near matching omits exactly one subgroup variable from a target containing at least two, with every other required field and retained condition matching. A category's denominator is its actual target count in that dataset.

These measures describe the breadth and reach of the search. They do not certify that an arbitrary untested hypothesis was scientifically plausible. Code checks well-formed comparisons, and numerical analyses establish support; no prose plausibility judge is added.

### Investigation after evidence

Extend the current follow-up summaries from independent validation events to every first valid discovery-sample result and every newly delivered validation result. For each event, examine the next two complete iterations and record:

- New comparisons actually tested and structural refinements linked to that evidence; separately, newly registered or accepted opposite-direction claims reusing an existing comparison's evidence.
- Further validation and whether it was voluntary or automatic.
- Movement between active, deferred, and closed investigation, and changes in assessment.
- Entry into previously untested hypothesis families and category-specific target coverage.

Count distinct validly executed comparisons. Re-running an unchanged comparison on the unchanged discovery dataset supplies no new evidence and receives no new-comparison credit. Keep repetition counts separately. Validate parent links and derive structural relatedness so arbitrary ID changes cannot masquerade as refinements.

Multiple evidence events may have overlapping follow-up windows. Preserve links and overlap indicators, deduplicate within each event, and do not treat windows as independent replicates or claim they establish causal attribution.

### Comparisons conditional on evidence

Compare subsequent investigation across expected, neutral, and surprising discoveries with comparable initial evidence, group sizes, subgroup complexity, and remaining budgets. Use prespecified evidence classes and normalized effect/interval measures within modality. Report how much support or contradiction changes continued investigation, validation, and acceptance, separately by literature category and the agent's own expectation.

The focal reversal pair supplies the strongest controlled comparison because the target variables and subgroup definition are held fixed. Conditional comparisons across all background discoveries are secondary: their difficulty and the agent-selected hypothesis pools can differ.

The exploration score in section 10 rewards reaching embedded target comparisons earlier within the fixed iteration budget. Raw hypothesis counts, more follow-up, and greater concentration have no automatic higher-is-better interpretation. Their conditional relationships with evidence remain supporting measurements of how the agent allocates investigation. Keep discovery performance, exploration coverage, and evidence responsiveness separate; no weighted average across those three scores is proposed.

## 8. Score requests for validation separately from responses

Report two primary descriptions of validation choice:

1. Among frozen scheduled opportunities, the proportion whose evidence the agent requested before automatic release, with its pre-request assessment and exploratory evidence recorded.
2. Among tested claims the agent accepts, the proportion for which it voluntarily requested and received validation before first acceptance. Partition first acceptances into those before any independent evidence, after voluntarily obtained evidence, and after automatic evidence. Also record requests made before acceptance whose results arrived later, and requests made after acceptance.

For the second measure, enter each canonical claim into the denominator once, at first acceptance, and freeze its request/receipt history at that time. A deferred claim has not yet entered this acceptance-based denominator. A new ID or withdrawal followed by reacceptance cannot change its first-acceptance history. Report unmatched and untested accepted claims separately. Neither measure awards a higher score simply for making more requests; report counts, eligible-pool sizes, and available request slots alongside the proportions.

Compare validation requests for surprising versus expected discoveries conditional on initial evidence and opportunity. Greater scrutiny of surprising discoveries is an informative behavior, particularly when they oppose recorded expectations. A larger request-rate difference is not intrinsically a better score; it could reflect insufficient scrutiny of expected discoveries.

Do not label every unrequested validation a mistake. A binary “should have requested” score requires a separate, explicit policy about evidence sufficiency, competing candidates, costs, and budget availability. The initial refactor will report the observable request decisions and their relationship to evidence. These measurements accompany `D`, `E`, and `B` and contribute no additional request-rate bonus. The post-validation response score uses the numerical reference above.

## 9. Preserve discovery confirmation and separate failure modes

Retain exact and near recovery among the current accepted claims, category counts, and independent final confirmation. Require a claimed discovery to have a valid discovery-sample test for new-version primary credit. This prevents a list of accepted but untested guesses from counting as a completed investigation.

The principal paired endpoint remains exact, independently confirmed recovery of the focal discovery. Final confirmation uses another independent sample and the existing `0.05 / M` correction over distinct final accepted claims. Additional accepted claims continue to receive independent confirmation and full-DGP adjudication; being outside the target inventory does not make them false.

Record separate milestones rather than forcing every run into a single explanatory label: target comparison tested, surprising evidence encountered, validation requested/received, correct direction registered, accepted, and confirmed. Distinguish unsupported acceptance from failure to accept supported discoveries. The target may be missed at several steps.

Keep the current retry accounting: recovered contract errors are visible but remain eligible under the declared policy; an exhausted stage error retains the run with zero primary recovery. Preserve the strict first-attempt sensitivity endpoint. All new queues, assessment histories, cached results, counters, and deadlines must participate in atomic rollback. Retries must never generate a fresh validation opportunity or expose an evaluator verdict.

Aggregate primary recovery over base-dataset pairs as in the existing summary. Extend reporting to the scores and aggregation rules in section 10, with explicit event counts and unavailable cells. Separate voluntary and scheduled evidence, immediate and delayed responses, and literature-defined and agent-recorded surprise. Report modality strata and bootstrap at the base-dataset level; do not treat correlated hypotheses or overlapping event windows as independent datasets.

## 10. Comprehensive scores and aggregation

| Reported outcome | Interpretation | Role |
|---|---|---|
| Discovery performance, `D` (0–100) | Recovery of embedded discoveries together with independent support for accepted claims | Overall performance and ranking |
| Exploration coverage, `E` (0–100) | How broadly and how early the agent validly tests embedded comparisons | Search coverage |
| Evidence responsiveness, `B` (0–100 when all evidence classes are represented) | Agreement with validation evidence after time for reconsideration | Conditional response to evidence |
| Paired surprise effect (percentage points) | Change in confirmed focal recovery when its direction is reversed | Principal expected-versus-surprising comparison |

### Discovery performance

For each run and literature category `c`, define:

```text
R_c = accepted, independently confirmed exact target discoveries in category c
      / embedded target discoveries in category c

R = (R_expected + R_neutral + R_surprising) / 3

Q = distinct final accepted claims with a valid exploratory test and independent confirmation
    / all distinct final accepted claims

D = 100 * 2 * R * Q / (R + Q)
```

Use the existing assignment rules to count each target once. A valid numerical test of the same comparison can support a subsequently registered opposite-direction claim; redundant execution is unnecessary. Count accepted claims by canonical claim identity, including direction, so renaming cannot change `Q`. Untested accepted claims remain in its denominator and cannot enter its numerator. Final confirmation retains the correction over all distinct final accepted claims specified in section 9.

The equal category weights prevent more numerous expected discoveries from dominating recovery. `Q` is the confirmed-claim fraction: additional discoveries outside the target inventory can enter its numerator after valid testing and independent confirmation. Continue their full-DGP adjudication as a separate report. Unconfirmed claims are not automatically labeled false.

The harmonic mean gives recovery and confirmation equal importance. These are prespecified benchmark choices. Report all three `R_c` values, `R`, `Q`, accepted-claim counts, and confirmation outcomes alongside `D`. If no claims are accepted, report `Q` as unavailable and `D = 0`. If `R + Q = 0`, also set `D = 0`. An exhausted stage error receives `D = 0`, consistent with zero primary focal recovery; preserve any partial results as diagnostics and report the completion rate. The strict first-attempt sensitivity analysis applies the corresponding zero to runs requiring technical repair.

For example, expected, neutral, and surprising recovery of 75%, 40%, and 65% gives `R = 0.60`. With `Q = 0.90`, `D = 72`. Accepting nothing gives zero even if every assessment remains cautious.

Use exact recovery for primary `D`. Report a secondary exact-or-near version using the same formula and assignment rules, with each target counted once and the same `Q`. Do not give near matches an arbitrary fractional weight.

### Exploration coverage

Let `C_c,t` be the fraction of embedded targets in category `c` whose comparisons have received a valid exact-matching exploratory test by the end of iteration `t`, ignoring claimed direction. For the configured iteration budget `T`:

```text
E = 100 * sum over t=1,...,T of
    [(C_expected,t + C_neutral,t + C_surprising,t) / 3] / T
```

This is the discrete area under cumulative target coverage. Reaching a comparison earlier earns more credit because more of the run remains for validation and follow-up. Testing an expected association and encountering its inverse immediately earns coverage credit; interpretation and acceptance enter `B` and `D`. Invalid analyses, repeated tests, and duplicate IDs add no coverage.

Use the configured budget as the denominator, including iterations with no new tests. If a run ends early, carry its achieved cumulative coverage through the remaining scheduled positions without adding discoveries; report termination and protocol status beside this descriptive score. Do not rescale to the shorter observed duration. Report category-specific curves and final coverage, plus an exact-or-near secondary version, to distinguish slow coverage from missing comparisons entirely.

`E` measures coverage of the embedded inventory. It does not establish that every exploratory choice was plausible or that every follow-up was worth its cost. Keep the evidence-conditional investigation measures in section 7 visible alongside it.

### Evidence responsiveness

For eligible validation events with complete response windows, calculate:

```text
A_supported = accept decisions / events whose validation supports the minimum-effect claim
A_excluded  = reject decisions / events whose validation excludes the minimum-effect claim
A_ambiguous = unresolved decisions / events whose validation includes the minimum effect

B = 100 * (A_supported + A_excluded + A_ambiguous) / 3
```

Use the explicit assessment at the second subsequent iteration. Initial uncertainty followed by the reference assessment receives full credit. Each first validation delivery and its claim fixed before that evidence contribute one event; cached re-delivery, renaming, or registering its opposite direction cannot create extra response opportunities. Retain the missing, invalid, late, and interrupted-event rules in section 6.

Equal weighting prevents a common evidence class from dominating the score. If any of the three classes has no eligible observations, the complete `B` for that reporting unit is unavailable. Report the observed component accuracies and counts without redistributing the missing class's weight. A missing required decision within an otherwise complete event remains an error, not an unavailable event.

Include all eligible comparisons in overall responsiveness, including additional claims outside the embedded inventory. Report voluntary and automatic delivery separately, and report literature-category subsets where target matching supplies a category. Keep unmatched claims unclassified by literature category. The agent's recorded prior direction supplies a separate expectation-based analysis. Neither subgroup analysis changes the overall evidence-class weights.

### Paired surprise effect

Retain the controlled focal-discovery comparison:

```text
Delta_discovery = 100 * [P(exact, confirmed focal recovery | surprising-focal version)
                        - P(exact, confirmed focal recovery | expected-focal version)]
```

Report both absolute recovery rates with the difference. Expected recovery of 80% and surprising recovery of 50% gives `Delta_discovery = -30` percentage points. A zero difference accompanied by poor recovery in both versions is not evidence of strong performance.

Report analogous paired differences for validly testing the focal comparison and, where opportunities permit, accepting it after supportive validation. These help distinguish search coverage from interpretation. The latter remains conditional on encountering the focal comparison and receiving suitable evidence; disclose different opportunity pools. Do not merge the paired difference into `D` or reward parity without reference to absolute performance.

### Aggregation and the complete report

Compute `D` and `E` within runs. Average replicates within each base-dataset version, then give the two versions equal weight within a base dataset and base datasets equal weight in the overall summary. Report clinical and cell-line strata; with the balanced current release, equal base-dataset weighting also gives the modalities equal total weight. Preserve paired versions and replicate structure when estimating the focal differences.

For responsiveness, first calculate each evidence-class accuracy within a run. Aggregate available class accuracies through the same replicate, version, and base-dataset hierarchy within each class, then average the three class summaries for `B`. Report eligible runs, versions, base datasets, and events contributing to each class; the populations can differ. A complete aggregate `B` requires all three classes, even when individual runs have unavailable `B` values. Always identify the reporting level. A model's conditional response score cannot substitute for coverage or completion when it produces few eligible opportunities.

Bootstrap base-dataset pairs, keeping their runs and events together. Preserve unavailable cells in resamples and report uncertainty as unavailable when it cannot be estimated from the eligible base datasets. Do not treat individual hypotheses or overlapping follow-up windows as independent replicates, or silently remove failed runs from discovery performance.

The complete report contains `D` with its recovery and confirmation components, `E` with coverage curves, `B` with class accuracies and denominators, and the paired surprise effect with absolute focal recovery. Validation requests and investigation after evidence accompany these scores as behavioral measurements. This specifies a compact score profile without an additional weighted average across `D`, `E`, and `B`.

## 11. Implementation order and acceptance checks

| Step | Files or modules | Deliverable |
|---|---|---|
| 1. Freeze the new contracts | [schemas.py](../src/onc_co_scientist/expected_surprising/schemas.py), configuration files | Versioned assessment, request, opportunity, checkpoint, policy, and score-component records; old readers remain available |
| 2. Separate public instructions from evaluator rules | [generation.py](../src/onc_co_scientist/expected_surprising/generation.py), [cli.py](../src/onc_co_scientist/expected_surprising/cli.py) | New task-package export from existing dataset bytes; public claim definitions and no numerical decision coaching |
| 3. Implement validation delivery | [evaluation.py](../src/onc_co_scientist/expected_surprising/evaluation.py) | Deterministic selection, delayed release, canonical caching, shared voluntary/automatic evidence, budgets, and alpha allocation |
| 4. Refactor the agent loop | [rollout.py](../src/onc_co_scientist/expected_surprising/rollout.py) | Appraisal before validation choice, evidence delivery before synthesis, explicit response deadlines, and complete transactional retry support |
| 5. Implement event and run scoring | [scoring.py](../src/onc_co_scientist/expected_surprising/scoring.py) and a dedicated event-summary module | `D`, `E`, `B`, their components, exact-or-near secondary scores, follow-up and validation-choice measures, and separate unsupported-acceptance counts |
| 6. Extend paired reports | [summary.py](../src/onc_co_scientist/expected_surprising/summary.py), CLI, documentation | Equal-weight base-dataset aggregation, evidence-class denominators and missingness, paired surprise effects with absolute recovery, source of validation, and reproducible provenance |
| 7. Verify the complete workflow | [benchmark tests](../tests/test_expected_surprising.py), [smoke runner](../scripts/expected_surprising/smoke_vllm.py) | Scripted behavior fixtures followed by separately recorded live smoke tests |

Scripted fixtures should include an agent that explores broadly and follows evidence, one that perseveres with contradicted expected claims, one that accepts everything, one that requests validation but ignores it, one that never requests validation but responds appropriately to automatic results, and one that never generates an eligible comparison. Include initial uncertainty followed by eventual correct acceptance.

Checks must establish that:

- Public prompts contain no decision thresholds expressed as mandatory accept/reject rules, target identities, category labels, or private selection information.
- Initial expectations and validation selections are frozen before their relevant evidence; decisions cite available results only.
- Voluntary and automatic routes yield the same cached comparison result, including after retries and direction reversals.
- Hypothesis renaming, duplicate requests, and repeated analyses do not generate extra credit or independent samples.
- Selection is unaffected by private target labels or unseen validation values; budgets and multiplicity bounds hold under every route.
- Early caution can resolve within the response window; missing, invalid, late, and interrupted events receive their declared treatment.
- Accepted-set identity, prior assessments, result decisions, and final reports remain consistent.
- Score arithmetic reproduces worked examples, including `R = 0.60`, `Q = 0.90`, and `D = 72`; empty accepted sets and exhausted protocol errors receive zero discovery performance.
- Category imbalance cannot change equal category weights; additional confirmed claims can enter `Q`, while untested accepted claims cannot enter its numerator.
- Earlier valid target coverage increases `E`; duplicate or invalid tests do not. A shortened run cannot improve its score by shrinking the configured iteration denominator.
- Missing evidence classes produce unavailable `B`, while missing due decisions contribute errors. A run consisting only of supported validation events cannot receive a complete balanced response score.
- Exact-or-near recovery counts each target once; aggregate reports retain equal base-dataset weights, paired assignments, and class-specific opportunity counts.
- Existing frozen datasets and historical reports remain reproducible under their own versions.

Then run six-iteration NSCLC clinical and cell-line smoke tests for both members of each pair against the configured vLLM server, retaining the 125K allowance and error-feedback retries. Inspect protocol completion, the presence and timing of voluntary/automatic evidence, `D`, `E`, `B` and all component counts, paired recovery, source hashes, and error outcomes. An unavailable `B` due to missing evidence classes must remain visible in smoke reports. These are future implementation checks, not runs performed by this planning task.

Freeze any revised schedule, selection strata, and response-window settings after independent workflow checks and before formal evaluation. Changes to delivery or scoring should receive new versions. A naturalistic arm without automatic validation and a fixed-hypothesis responsiveness probe can be added as separately labeled experiments if those comparisons are needed.
