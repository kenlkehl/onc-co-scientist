# Expected and surprising discoveries: Aim 1

For an explanation with actual dataset rows, worked scoring examples, endpoint options, and run commands, start with [How the expected/surprising task works](EXPECTED_SURPRISING_GUIDE.md). This page provides the detailed implementation contract.

Review policy 2.1.0 supersedes the original release, which is withdrawn from evaluation because of unsupported scope extrapolations and questionable neutral labels. Historical data and audit records remain under v1.

This version creates paired research datasets for the five clinical and five CRISPR/DepMap profiles. Each base dataset contains six discoveries. The default inventory uses four literature-reviewed expected associations and two neutral discoveries. AML clinical and cell-line tasks use three of each; the mix is configurable. One reviewed expected association is assigned a surprising direction in both versions. One selected expected discovery changes direction between versions. For the usual mix, the expected version has three expected, two neutral, and one surprising discovery; its partner has two expected, two neutral, and two surprising discoveries. The AML pairs have one fewer expected and one additional neutral discovery in each version. Scoring uses each dataset’s actual category denominators.

The first release is fully synthetic. Clinical cohorts contain 50,000 rows and use continuous `log_pfs_months`, the natural logarithm of fully observed PFS in months. DepMap datasets contain 2,000 cell lines and continuous knockout dependency scores. Existing covariate samplers supply clinical and cell-line features; existing outcome values and private latent columns are removed before the new DGP is evaluated. Additional constructed research signatures provide subgroup variables. Version 2 adds binary research markers D/E/F without assigned genes, pathways, or clinical roles as possible neutral exposures. Where the underlying variables are present, deterministic threshold encodings provide ECOG PS ≥2, CRP ≥10 mg/L, albumin ≤3.5 g/dL and NLR ≥3 comparisons. These are candidate encodings; their exact cutoffs still require evidence and realism review. Version 1 sampling is preserved for historical replay. The cell-line panel adds TP53, MTAP, SMARCA4, ARID1A, and KEAP1 loss, plus disease-specific AR-pathway and AML molecular features. Their enriched prevalences are simulation design values. The DepMap outcome panel also includes endpoints with no embedded target. Plasmodes, censored-survival analysis, and binary outcome adapters remain extensions of the grant design.

## Generation and literature review

```bash
ocs expected-surprising research configs/expected_surprising.example.yaml
ocs expected-surprising generate configs/expected_surprising.example.yaml
```

Both commands accept `--profile nsclc_clinical` (or another profile) and `--out PATH`. The default output is `data/expected_surprising_v2`. Provider settings are explicit in the YAML; the generation workflow uses the existing provider registry. No generation provider is used to score agents.

The LLM proposes expected and neutral candidates using the available covariate schema. The updated workflow applies three gates: source-grounded literature review, a separate candidate realism review, and review of the complete compiled DGP. For each candidate it supplies complementary queries intended to retrieve supporting and contradictory evidence. The workflow executes those searches through the [Europe PMC REST API](https://europepmc.org/RestfulWebService), retains the query, retrieval time, publication IDs, titles, DOIs, abstracts, and available full-text excerpts, and sends the retrieved evidence to the LLM for review. Expected candidates require evidence matching the population, exposure/comparator, endpoint, and direction. Reviews can cite only retrieved publication IDs. A trial restricted to biomarker-positive patients does not establish a biomarker-positive versus biomarker-negative comparison, and drug sensitivity alone does not establish genetic dependency. Genetic knockout or validated knockdown evidence can support an expectation; assay-transfer assumptions are retained in the review rationale.

The literature review records each supporting study's actual population, exposure, comparator, endpoint, assay/design, direction and limitations. Expected candidates require a primary empirical source with the correct endpoint class. Recurrence-free survival, overall survival and binary PFS landmarks cannot be relabeled as PFS evidence. A treatment flag of zero denotes absence of that drug; it does not identify a control regimen. Necessary biomarker, regimen, stage, histology and treatment-line restrictions must be expressible in the candidate.

A separate LLM call assesses realism without seeing the first reviewer's verdict. It receives the structured claim, retrieved evidence and explicit variable semantics, and searches for allowed patient strata that would receive an unjustified expected effect. The reviewer lists necessary scope conditions and unrepresented requirements. Python checks that each required condition follows from an encoded eligibility/subgroup condition; an accepting verdict cannot override a missing restriction. The `realism_provider` configuration can select a different model; the example uses a separate call to the same Gemini model with greater reasoning effort.

Neutrality requires absence of an inherited directional expectation for the parent exposure/outcome association. A newly constructed subgroup does not erase an established sex, gene or treatment association. Constructed markers with no assigned biological role may supply neutral parent comparisons; their injected nonzero effect does not itself imply a literature expectation. Uncertainty about classification triggers rejection. Nested versions of the same exposure/outcome contrast are excluded to reduce redundant discovery credit.

Unsupported or ambiguous candidates are rejected. The next proposal round receives the rejected candidates and review reasons. Generation continues until the configured inventory of three or four supported expected candidates and the remaining neutral candidates is complete, or the configured round limit is reached. Failure leaves the audit and accepted-candidate checkpoint available for resumption. Both clinical and DepMap checkpoints are re-reviewed when their policy, profile or exact claim changes. Literature review by an LLM is a development aid; formal benchmark classification still requires the expert adjudication specified in the grant.

Review policy and candidate hashes bind acceptance to the exact claim. Old reviewed candidates cannot bypass the new gates. Every model prompt/response and acceptance/rejection is recorded under `research/<profile>/events.jsonl`. Search results are cached under `searches/`. The checked candidate inventory is `<profile>_reviewed.json`. Treat these files as private evaluator material.

## The paired DGP

Each discovery specifies an outcome, exposure, exposed level, comparator level, eligibility conditions, subgroup conditions, contrast, signed coefficient, and evidence record. No model-produced code is executed. Typed predicates compile to Boolean masks and fixed numerical operations.

For a mean-difference discovery, the contribution is

```
I[eligible] * I[exposure == exposed] *
    (signed_inside_coefficient * I[subgroup] + outside_coefficient * I[not subgroup])
```

Contributions for an endpoint are added to its intercept and Gaussian residual. An interaction contribution uses `outside_coefficient + signed_inside_coefficient * I[subgroup]`; its target contrast is the exposed-minus-comparator difference inside the subgroup minus that difference in its complement. These are descriptive associations on the specified scale.

The two versions share covariates, exposure assignments, and outcome-specific residual streams. Only the signed inside coefficient of the focal discovery reverses. In a subgroup pair, the same two subgroup variables and cutoffs occur in both versions; the fixed outside coefficient preserves the expected overall direction. All other discovery specifications are identical. Overlap can change observed contrasts, which is why the entire DGP is checked.

Before materialization, the design selector considers alternative focal/background assignments among the reviewed expected candidates. It requires the correct direction for every discovery, focal absolute-effect and signal-to-noise differences within 10%, sufficient focal cell sizes, and preservation of the expected overall direction for subgroup pairs. Selection uses development draws separate from the released discovery sample. A failure requires revising the candidate inventory. After selection, a separate LLM reviews both explicit equations and their scoped evidence together. It checks whether additive terms extend expected effects outside supported populations, duplicate targets, confuse endpoints or introduce unintended discordance. The reviewer receives an explicit list of intentional reversals in each version, distinguishing the fixed surprising background from the focal discovery that switches direction. Both versions contain a mixture of discovery categories. The complete DGP review must pass before datasets are written. Its hash excludes administrative status and retained abstract text but binds the numerical model, discovery definitions and evidence attestations; changing any of these requires another review.

Historical inventories can be rechecked and replacements generated with:

```bash
ocs expected-surprising recheck configs/expected_surprising.example.yaml benchmarks/expected_surprising/v1/specs
ocs expected-surprising research configs/expected_surprising.example.yaml
ocs expected-surprising generate configs/expected_surprising.example.yaml
```

`scripts/expected_surprising/rebuild_v2.py` resumes this sequence by profile and then runs the 1,000-replicate calibration. Rejected candidates and their reasons remain in the audit. A failed scientific gate stops materialization. Expert adjudication still precedes formal task locking.

```bash
ocs expected-surprising calibrate data/expected_surprising_v2/private/PAIR_ID/pair.json \
    --replicates 1000 --reference-n 100000
```

Calibration measures every discovery under the full DGP and estimates focal recovery in a fixed, publicly defined reference search grid. The v2 grammar includes every binary exposure and endpoint, eligibility defined by up to four other binary variables at either level, and either no subgroup or two/three constructed signatures at thresholds 0 or 0.5 (greater than or equal). This includes population restrictions such as stage IV, biomarker status, and treatment exposure. Each two-sided comparison uses a Welch interval with error probability `0.05 / grid_size`. The full grid size is calculated combinatorially, including sparse and empty comparisons, without allocating millions of hypothesis objects. Its definition uses public covariates and endpoints; private discovery predicates do not define the search universe. Focal recovery can be computed from the focal test and grid size without executing unrelated tests. A focal comparison outside this grammar fails design selection. Calibration reports record the reference policy version, and a changed policy requires recalibration. Historical v1 replay retains its original treatment-only eligibility grammar.

Formal calibration requires at least 1,000 development replicates, recovery of at least 80% in each condition, and a recovery difference no greater than five percentage points. Short calibration runs are explicitly recorded as development checks. Expert review and formal task locking are separate from these numerical criteria.

## Public tasks and private evaluation

`public/<opaque_task_id>/` contains the dataset, task instructions, endpoint thresholds, data dictionary, and stage schema. The private directory contains `pair.json`, explicit equations for both versions, the assignment map, design-selection records, and calibration. Only the selected public task should be exposed to an agent. The active rollout harness requires current candidate and full-DGP review attestations. V1 can be reproduced only with the explicit `materialize --historical-replay` option, and remains ineligible for active evaluation. Directory separation is a packaging convention; deployments with filesystem tools must enforce that separation through their sandbox or container mounts.

The reference provider harness has no filesystem or arbitrary-code tool access. A trusted analysis service executes the agent's structured comparisons. It receives one public task per run and never sends paired assignments, discovery categories, candidate reviews, DGP parameters, or answer keys to the provider.

```bash
ocs expected-surprising run data/expected_surprising_v2/private/PAIR_ID/pair.json \
    data/expected_surprising_v2/public configs/expected_surprising.example.yaml \
    data/expected_surprising_v2/runs/RUN_ID RUN_ID --version surprising
```

The reference harness runs hypothesis, analysis, critique, and synthesis stages for 25 clinical or 10 DepMap iterations. Each iteration permits up to 12 structured analyses; provider output limits are configurable and identical between paired runs. `--iterations 2` allows a smoke run. The schema and scoring modules can also support external harness adapters; the legacy named/masked harness remains available through its existing commands.

Reported estimates and confidence bounds are already oriented to the registered claim: `estimate = hypothesis.direction * raw_contrast`. For a negative-direction hypothesis, a raw exposed-minus-comparator difference of -0.4 is returned as a signed estimate of +0.4. The acceptance rule always compares the reported lower bound with the positive endpoint threshold; multiplying by direction again would reverse the evidence. Every stage prompt states this convention explicitly.

The harness allows two retries per stage by default, configurable with `max_retries_per_stage` in the run YAML. Empty or malformed responses and protocol errors return to the agent in the same stage. Feedback includes the specific error, the failed response (up to 24,000 characters), and the IDs registered before that stage. Unknown variables identify the offending name and suggest close matches from the public schema. The agent must return a complete replacement stage record. Every attempt and feedback message is retained in the transcript.

A failed attempt rolls back hypothesis registrations, exploration counts, evidence decisions, accepted claims, and the private validation-service state. Failed-stage results are not released to the agent. Repeating a validation request after rollback reproduces its sample rather than providing another independent evidence opportunity. A valid but scientifically incorrect decision is scored as submitted and does not trigger corrective feedback. The report separates `attempt_errors`, `recovered_stages`, and terminal `protocol_errors` after the stage exhausts its retry budget. `protocol_complete` means all stages completed successfully, including permitted retries; `protocol_clean` additionally requires no failed attempts.

For a vLLM smoke test, `scripts/expected_surprising/smoke_vllm.py` runs both versions of the NSCLC clinical and cell-line datasets through three iterations. Its default completion allowance is 125,000 tokens, including reasoning tokens, with a configurable request timeout and two retries per stage. It retains transcripts, deterministic reports, public dataset hashes, harness source hashes, provider configuration, and per-call token/finish diagnostics. Example:

```bash
python scripts/expected_surprising/smoke_vllm.py \
  --base-url http://sn4622130540:8000/v1 \
  --model Inferact/Qwen3.8-27B-NVFP4 \
  --out data/expected_surprising_v2/smoke/NEW_RUN \
  --iterations 3 --max-tokens 125000 --max-retries-per-stage 2 --timeout-s 1800
```

The [initial September 7 vLLM smoke report](../benchmarks/expected_surprising/v2/smoke_vllm_20260907.md) covers four v2 runs at 100,000 tokens with no stage retries. All yielded scorable analyses, while four responses exhausted the entire budget in reasoning and returned no answer. Its artifacts preserve that earlier protocol.

The [125K rerun with retries](../benchmarks/expected_surprising/v2/smoke_vllm_20260907_125k_retries.md) completed every stage in all four runs. Two synthesis responses exhausted the larger token limit and then recovered on their first retries. A separate live probe also corrected the prior variable-name error. The report distinguishes first-attempt success, recovery after retries, confirmed discoveries, and unavailable scores when agents did not request independent validation.

## Deterministic scoring

- **Discovery:** exact matching requires the outcome, exposure/comparator, contrast, direction, eligibility, and all subgroup conditions. Reversing exposure/comparator coding also reverses direction. Near matching allows omission of every condition on exactly one subgroup variable when the target has at least two subgroup variables; every retained condition must match. Extra restrictions, changed cutoffs, and substituted variables receive neither exact nor near credit. Interaction targets require interaction contrasts. A global assignment prioritizes exact matches, deduplicates equivalent claims, and counts each discovery and each claim at most once.
- **Exploration:** each stage records new/cumulative distinct proposed hypotheses, successfully executed hypotheses, hypothesis families, and target coverage with anticipated direction ignored. Changing direction or a cutoff refines a family. Coverage is separate from acceptance and significance. Stage-specific recovery and the first exact hypothesis are retained.
- **Responsiveness:** agents record a prior assessment and anticipated direction before analysis. Up to ten validation requests, at most one per iteration, each draw a new independent sample. Requests are immutable and retries return the original evidence. Signed Welch intervals use 99.5% confidence. A lower bound above the endpoint's positive delta supports acceptance; an upper bound below delta supports rejection of the minimum-effect claim; intervals containing delta support continued uncertainty. Boundary equality is unresolved. Valid results with missing decisions score zero; invalid analyses are separate; no valid events yields an unavailable score. Update and maintenance accuracy are reported separately.
- **Further exploration:** the report counts new hypotheses, distinct execution-request IDs, and linked refinements from successful stage records in the next two scheduled iterations after validation. Events with fewer than two iterations remaining are excluded. An unrecovered stage error can leave incomplete records within an otherwise eligible window.
- **Final confirmation:** all distinct accepted claims are tested in a further independent sample with error probability `0.05/M`. The primary endpoint requires exact focal recovery and independent confirmation. Under retry policy 1.0.0, an unrecovered stage error retains the run in the denominator with zero primary recovery; a successfully corrected attempt remains eligible. `first_attempt_primary_recovery` separately applies the historical strict rule of zero credit after any failed attempt. Retry policy and token allowance are recorded for each run and should be held constant within a formal evaluation. Additional claims are reported as confirmed or unconfirmed and adjudicated against full-DGP conditional means in an independent 100,000-row reference sample, with Monte Carlo intervals. Being outside the target inventory does not establish that an association is false.

The reference implementation uses direct continuous-outcome contrasts. Approximate-cutoff sensitivity analyses, adjusted/censored outcome adapters, and the controlled identical-evidence probe experiment will build on these contracts. `ocs expected-surprising summarize REPORTS_ROOT OUTPUT_JSON` estimates paired recovery differences with a bootstrap over base datasets, stratified by modality. Both conditions must be present for each model/harness cell; run counts are not treated as independent dataset counts. The release reports these boundaries explicitly.

## Figure and grant text

`docs/figures/aim1_overview.svg` is editable vector artwork; PDF and 300-dpi PNG versions are included. Rebuild with `scripts/grant/draw_aim1_overview.py`. `scripts/grant/build_aim1_approach.py` produces the revised Approach with the figure in `outputs/grant_revision_2026-09-06/`. The original grant attachment is preserved.

Frozen specifications and calibration reports are in `benchmarks/expected_surprising/v2/`. Use `ocs expected-surprising materialize benchmarks/expected_surprising/v2/specs NEW_OUTPUT` to regenerate all pairs without new LLM or literature calls. The private release inventory is rebuilt by `scripts/expected_surprising/freeze_release.py`.
