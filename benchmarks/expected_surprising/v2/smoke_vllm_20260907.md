# vLLM smoke tests of the v2 reference harness

Date: September 7, 2026. Branch: `expected-or-surprising`.

All four runs produced scorable analyses and independently confirmed target discoveries. One run completed every stage without a protocol error. The other three had empty responses after exhausting the 100,000-token allowance; one also had a variable-name error and a dependent registration error.

The smoke tests used `Inferact/Qwen3.8-27B-NVFP4`, served by vLLM at `http://sn4622130540:8000/v1`. Each run allowed 100,000 completion tokens per request, including reasoning tokens, with temperature 0 and a 1,800-second request timeout. Four independent agent histories covered both versions of the NSCLC clinical and cell-line pairs. Each history had three iterations of hypothesis generation, analysis, critique, and synthesis, for 12 requests per run. The agent received the assigned public task and trusted analysis results. Discovery matching and evidence scoring used the existing Python implementation.

Both datasets in each pair contain six embedded discoveries. The version names describe the focal discovery: the expected version contains three expected, two neutral, and one surprising discovery; its partner contains two expected, two neutral, and two surprising discoveries. The clinical focal discovery concerns NLR ≥3 and PFS. The cell-line focal discovery concerns TP53 loss and USP7 dependency within the subgroup defined by research signatures A ≥0 and B ≥0. All other discovery specifications are fixed within each pair.

## Results

| NSCLC task / focal version | Valid stages | Distinct hypotheses proposed / tested | Confirmed exact discoveries | Correct validation decisions | Primary focal recovery |
|---|---:|---:|---:|---:|---:|
| Clinical / expected | 12/12 | 36 / 36 | 3/6 | 3/3 | 1 |
| Clinical / surprising | 11/12 | 35 / 35 | 3/6 | 3/3 | 0 |
| Cell-line / expected | 8/12 | 22 / 18 | 1/6 | 1/2 | 0 |
| Cell-line / surprising | 11/12 | 29 / 29 | 2/6 | 3/3 | 0 |

All 118 executed discovery-sample analyses returned valid numerical results. All 11 independent validation requests were valid. The expected cell-line run's missing critique scored zero for its validation decision. Across the 116 decisions actually recorded in successful stages, including discovery-sample decisions, every decision agreed with the numerical evidence rule. No near matches or neutral target recoveries occurred.

The proposed count of 22 for the expected cell-line run follows the current harness's registration state. Four valid hypotheses preceding the misspelled variable were registered before that first stage raised an error; 18 hypothesis IDs appeared in successful stage records. Its exploration counts therefore require the accompanying protocol-error record. The error forces primary recovery to zero.

| NSCLC task / focal version | Expected exact / available | Neutral exact / available | Surprising exact / available | Additional confirmed claims |
|---|---:|---:|---:|---:|
| Clinical / expected | 2/3 | 0/2 | 1/1 | 12 |
| Clinical / surprising | 1/2 | 0/2 | 2/2 | 0 |
| Cell-line / expected | 1/3 | 0/2 | 0/1 | 0 |
| Cell-line / surprising | 1/2 | 0/2 | 1/2 | 2 |

The clinical runs recovered NLR ≥3, CRP ≥10 mg/L, and ECOG PS ≥2. The NLR discovery was recovered in its expected direction in one version and its surprising direction in the other. CRP retained its expected direction, and ECOG retained its surprising direction in both. The cell-line runs recovered the expected SMARCA4-loss/SMARCA2-dependency discovery; the surprising version also recovered the fixed surprising KEAP1-loss/NFE2L2-dependency discovery. Neither recovered the focal TP53-loss/USP7-dependency subgroup discovery. The additional claims were independently confirmed and supported by full-DGP adjudication, but did not earn additional target-discovery credit.

The batch ran from 13:06:32 to 13:51:32 UTC, approximately 45 minutes. All 48 requests specified 100,000 completion tokens. There were 44 natural stops and four length-limit stops, using 834,818 completion tokens in total. Naturally completed calls used at most 21,342 tokens. Each length-limit stop used all 100,000 tokens in reasoning and returned an empty answer. These occurred in the final synthesis of both surprising runs and in the second-iteration critique and final synthesis of the expected cell-line run. The latter also proposed `cdn2a_loss` instead of the public column `cdkn2a_loss`, causing the initial registration error and a subsequent attempt to analyze unregistered IDs.

The clinical surprising run's primary score is zero because its final synthesis failed, despite independent confirmation of the focal discovery retained from iteration 2. Both cell-line runs also receive zero primary scores, and neither recovered its focal subgroup discovery. The clinical paired difference in primary scores consequently includes a protocol-completion failure.

## Interpretation of the scores

For this smoke test, a run is **scorable** when it produces a successfully parsed stage and at least one valid discovery-sample analysis. A **protocol-clean** run completes all 12 stages without an error. Discovery counts require exact or near matching to distinct embedded targets and confirmation in a further independent sample. Additional supported claims are reported separately. Protocol errors give the run zero primary focal-recovery credit, while its successfully recorded analyses, decisions, and confirmed matches remain available for inspection. If the final synthesis fails, confirmation uses the accepted claims from the last successful synthesis.

Validation responsiveness compares the first required critique with the numerical evidence rule. Missing decisions receive zero credit. Discovery-sample decisions were also audited against the same rule as a diagnostic; this audit is separate from the prespecified validation-responsiveness score. Follow-up exploration counts available stage records in the next two scheduled iterations, so a three-iteration smoke test can supply at most one such window per run.

The surprising clinical transcript provides a concrete exploration–response sequence. In iteration 1, the agent registered `H1-11`, anticipating shorter PFS with NLR ≥3. It rejected that claim after analysis and registered the opposite-direction claim, `H1-13`, with `parent_ids["H1-13"] = "H1-11"`. It tested and accepted the new claim in iteration 2 and independently validated it in iteration 3. This sequence can be recovered from structured records without narrative interpretation.

Follow-up exploration windows contained 20 new hypotheses and 24 tested hypotheses for clinical/expected; 21 new hypotheses and 23 tested hypotheses for clinical/surprising; and 16 new hypotheses and 17 tested hypotheses for cell-line/surprising. The expected cell-line run had no validation event early enough to supply a two-iteration follow-up window. There was one explicitly linked refinement to the originating validation hypothesis in these windows, in clinical/surprising. Refinements following discovery-sample evidence are separately visible in the stage records.

These runs demonstrate that exploration, directional revision, discovery matching, and confirmation produce auditable outputs. Their independent validation events had seven intervals agreeing with the registered directional expectation and four with unresolved direction; none clearly opposed the expectation. Agents often selected validation after seeing discovery-sample evidence. The observed validation accuracy therefore has limited coverage of responses to evidence opposing a prior expectation. Broader exploration of neutral and subgroup discoveries, responses under controlled evidence exposure, and reliable final-answer completion remain important questions for subsequent evaluation.

## Diagnostic changes before the reported batch

An initial attempt with an 8,192-token limit produced truncated reasoning and empty final responses. It was stopped when the completion allowance was increased to 100,000 tokens. A second attempt at 100,000 tokens produced structured responses but exposed ambiguity in how signed estimates were presented: the model sometimes multiplied an already signed estimate by the hypothesis direction again. For example, a reported estimate of −0.4564 for a negative-direction ECOG claim was incorrectly treated as supporting that claim.

The stage prompt now explicitly defines `reported_estimate = hypothesis.direction * raw_contrast`, gives a negative-direction example, and states that both directions use the reported interval against the positive effect threshold. The numerical analyses, DGP, matching rules, and scoring rules were unchanged. All four histories were then started fresh for the reported batch. Both stopped attempts retain their transcripts and marked manifests under the sibling directories `vllm-sn4622130540-20260907` and `vllm-sn4622130540-20260907-100k`.

The 34 benchmark tests passed after this clarification. Ruff checks passed for the harness and smoke runner. The DGP generation and numerical scoring sources are unchanged from the v2 release; the smoke manifest records the revised harness source hash.

## Reproduction and audit

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-expected-surprising-venv/bin/python \
  scripts/expected_surprising/smoke_vllm.py \
  --base-url http://sn4622130540:8000/v1 \
  --model Inferact/Qwen3.8-27B-NVFP4 \
  --out data/expected_surprising_v2/smoke/NEW_RUN \
  --iterations 3 --workers 4 --max-tokens 100000 --timeout-s 1800
```

Raw artifacts are under `data/expected_surprising_v2/smoke/vllm-sn4622130540-20260907-100k-v2/`: provider configuration, manifest, per-call token usage and finish reasons, four transcripts and deterministic reports, a combined summary, and the decision audit. The manifest records public dataset and private specification hashes, source hashes, dependency versions, server model metadata, and start/completion times. The permanent JSON summary includes hashes of the raw artifacts.

The compact permanent audit is [smoke_vllm_20260907.json](smoke_vllm_20260907.json). Its source hashes match the implementation used for these runs. Literal checks of all recorded prompts found no private pair IDs or target-discovery IDs; the provider inputs were also reviewed in the harness code.

These three-iteration runs test the workflow on one pair per modality. Formal tasks allow 25 clinical or 10 cell-line iterations. Expert literature adjudication and formal task locking remain pending.
