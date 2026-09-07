# vLLM smoke tests with 125K tokens and stage retries

Date: September 7, 2026. Branch: `expected-or-surprising`.

All four natural runs completed all 12 stages and produced scorable outputs. Three completed every stage on the first attempt. The surprising clinical run exhausted 125,000 tokens in two synthesis responses; both stages recovered on their first retry. There were no unrecovered protocol errors. A separate live check also demonstrated correction of the variable-name error from the prior smoke batch.

These runs repeat the NSCLC clinical and cell-line smoke tests using `Inferact/Qwen3.8-27B-NVFP4` at `http://sn4622130540:8000/v1`. Both versions of each pair retain the same public datasets and private specifications used in the [100K-token smoke tests](smoke_vllm_20260907.md). Each run has three iterations of hypothesis generation, analysis, critique, and synthesis. Each request allows 125,000 completion tokens, including reasoning, with temperature 0 and a 1,800-second timeout. Up to two retries are available for each stage.

The expected/surprising labels refer to the focal discovery. Every dataset contains six embedded discoveries: three expected, two neutral, and one surprising in the expected-focal version; two expected, two neutral, and two surprising in its partner. The clinical focal discovery reverses the overall NLR ≥3 association with PFS. The cell-line focal discovery reverses the TP53-loss/USP7-dependency association within the subgroup defined by research signatures A ≥0 and B ≥0.

## Results

| NSCLC task / focal version | Successful stages | Failed attempts / recovered stages | Distinct hypotheses proposed / tested | Confirmed exact discoveries | Correct independent-validation decisions |
|---|---:|---:|---:|---:|---:|
| Clinical / expected | 12/12 | 0 / 0 | 36 / 36 | 4/6 | Unavailable: none requested |
| Clinical / surprising | 12/12 | 2 / 2 | 40 / 36 | 3/6 | 2/2 |
| Cell-line / expected | 12/12 | 0 / 0 | 36 / 36 | 2/6 | 1/1 |
| Cell-line / surprising | 12/12 | 0 / 0 | 36 / 36 | 2/6 | Unavailable: none requested |

All 144 discovery-sample analyses returned valid numerical results. All 147 recorded decisions, including the three independent-validation decisions, agreed with the numerical evidence rule. No neutral discoveries or near matches were recovered. The four additional hypotheses in the surprising clinical run were registered during its final critique and remained untested.

| NSCLC task / focal version | Expected exact / available | Neutral exact / available | Surprising exact / available | Additional confirmed claims | Primary recovery / strict first-attempt recovery |
|---|---:|---:|---:|---:|---:|
| Clinical / expected | 3/3 | 0/2 | 1/1 | 12 | 1 / 1 |
| Clinical / surprising | 1/2 | 0/2 | 2/2 | 10 | 1 / 0 |
| Cell-line / expected | 1/3 | 0/2 | 1/1 | 4 | 0 / 0 |
| Cell-line / surprising | 1/2 | 0/2 | 1/2 | 1 | 0 / 0 |

The expected clinical run recovered the expected stage-IV, CRP ≥10 mg/L, and NLR ≥3 discoveries, along with the fixed surprising ECOG PS ≥2 discovery. Its partner recovered expected CRP and surprising NLR and ECOG. The clinical focal discovery was therefore recovered in both directions. Both cell-line versions recovered expected SMARCA4-loss/SMARCA2 dependency and the fixed surprising KEAP1-loss/NFE2L2 dependency. Neither recovered the focal TP53-loss/USP7-dependency subgroup discovery. All additional claims in the table were independently confirmed and supported by full-DGP adjudication; they did not earn extra target-discovery credit.

The two clinical synthesis failures occurred in iterations 2 and 3. Each used all 125,000 tokens in reasoning and returned an empty answer. The first retries returned valid records using 3,316 tokens in 32.0 seconds and 4,432 tokens in 41.8 seconds, respectively. The failed attempts remain in the audit, while the corrected stages remain eligible for recovery under the declared retry policy.

The four-run batch ran from 14:14:48 to 15:03:50 UTC, about 49 minutes. It made 50 provider calls: 48 natural stops and two length-limit stops, consuming 620,475 completion tokens. Every call specified a 125,000-token allowance. Naturally completed calls used at most 18,595 tokens. The separate variable-repair check made one additional live call using 4,960 tokens; it is excluded from those batch totals.

Independent validation was optional. The expected clinical and surprising cell-line agents never requested it, so their validation-responsiveness scores are unavailable. All three requested validation events agreed with the direction anticipated for the registered claim; two required an assessment update and one maintained the previous assessment. None of the runs requested validation in iteration 1, so none supplied the two subsequent iterations required for the validation-triggered exploration metric. Stage-level exploration and discovery-sample decisions remain available. These omissions limit the responsiveness and follow-up-exploration conclusions from this short batch.

## Retry behavior

A response with a protocol error receives feedback within the same stage. Feedback identifies the error, includes the failed response, and lists the hypotheses registered before that stage. For an unknown variable, it identifies the offending name and offers close matches from public column or outcome names. The agent supplies a complete replacement stage record. The full failed response remains in the transcript; feedback includes at most 24,000 characters and records whether it was shortened.

Stage execution is atomic: an unsuccessful attempt restores registrations, prior assessments, exploration counts, evidence decisions, accepted claims, and private validation state. Results from an unsuccessful stage are not released to the model. If a retry repeats its validation request, the restored seed and request index reproduce the same sample. This also fixes the partial-registration behavior exposed by the original cell-line smoke run.

The transcript retains model attempts, error feedback, successful stages, recovered stages, and errors that remain after retries. `protocol_complete` requires all 12 stages to succeed within their permitted attempts. `protocol_clean` additionally requires that every stage succeeded on its first attempt. The primary recovery score under retry policy 1.0.0 permits successfully repaired stages; an unrecovered stage error yields zero. The companion `first_attempt_primary_recovery` applies the historical strict rule that any failed attempt yields zero. Discovery matching, numerical evidence rules, and independent confirmation are unchanged.

Retry feedback checks the response contract and public schema. A valid but scientifically incorrect decision is scored as submitted. The test suite verifies this distinction, successful correction of unknown variables, empty and malformed responses, rollback after an analysis has already executed, reproduction of private validation evidence after rollback, and bounded retry exhaustion. All 40 benchmark tests passed; Ruff checks passed for the edited Python files.

## Reproduction and audit

The separate variable-repair check replayed the actual malformed hypothesis response from the previous expected cell-line run. The harness rejected `cdn2a_loss` and returned its normal error feedback, including the public name `cdkn2a_loss`. On the first live retry, the model returned a valid set of 12 hypotheses, corrected the exposure to `cdkn2a_loss`, and renamed the corresponding ID from `H1_CDN2A_LOSS_RIT1` to `H1_CDKN2A_LOSS_RIT1`. The outcome, comparison, direction, and conditions of that hypothesis were preserved. The live repair call used 4,960 completion tokens and finished naturally. The remaining stages in this controlled check were empty administrative stubs; its results are excluded from agent discovery and responsiveness metrics.

```bash
PYTHONNOUSERSITE=1 /tmp/ocs-expected-surprising-venv/bin/python \
  scripts/expected_surprising/smoke_vllm.py \
  --base-url http://sn4622130540:8000/v1 \
  --model Inferact/Qwen3.8-27B-NVFP4 \
  --out data/expected_surprising_v2/smoke/NEW_RUN \
  --iterations 3 --workers 4 --max-tokens 125000 \
  --max-retries-per-stage 2 --timeout-s 1800
```

Raw artifacts are under `data/expected_surprising_v2/smoke/vllm-sn4622130540-20260907-125k-retries/`. The manifest preserves provider configuration, dataset/specification hashes, source and runner hashes, dependency versions, model metadata, and execution times. Each natural run retains its transcript, deterministic report, and per-call token diagnostics. The separate `repair_probe/` directory contains a controlled replay of the prior variable-name error; it is excluded from the four natural rollouts.

The permanent [machine-readable audit](smoke_vllm_20260907_125k_retries.json) contains the run summaries, retry records, controlled-repair result, source hashes, and raw-artifact hashes. Dataset and specification hashes, and the assigned task IDs, match the previous 100K batch. Literal checks of all natural-run prompts found no private pair IDs or target-discovery IDs.

These are three-iteration workflow checks. The model budget, retry policy, and instructions changed together, so comparison with the previous smoke batch does not isolate the effect of increasing the token limit. Expert adjudication and formal task locking remain pending.
