# Living clinical workflow report — September 9, 2026

Updated **2026-09-12 01:34:44 AM EDT**. The selected comparison contains **360/360 finished runs** (350 completed all stages; 10 finished with errors), **0 active**, and **0 queued**, and **0 paused**. **104/104 Codex replacements have finished.**

This report follows the takeover of the six-model experiment. Each model has three workflows, two versions of the same clinical dataset, and ten separate runs per version: 20 runs per model/workflow. All models request medium reasoning and standard/default service. Each run has 25 iterations; the two-iteration rule is a deadline for reassessing evidence, not a limit on the run.

## What changed





- **104 Codex cells restart from the beginning:** 22 finished cells had native proof that the adapter discarded a completed response; 82 were still active or queued at the pause. Original traces remain saved. The 136 other finished Codex cells are retained, including four failures unrelated to this adapter defect. A replacement is chosen by its fixed run identity, never by whether its new score improves.
- **Adapter:** a successful terminal completion after reconnect warnings is accepted, with its output tokens retained. Genuine terminal failures still fail. This fix is in the new frozen build; the old records are unchanged.
- **Bookkeeping:** repeating a claim preserves its current scientific assessment. An unchanged assessment or investigation-only edit can proceed without evidence. These edits do not earn evidence-response credit, satisfy required reassessments, or change a scientific conclusion without evidence.
- **Peer failure:** each peer retains its existing retry budget. If a peer still fails, the chair receives an explicit missing-draft notice and proceeds with available drafts and the ledger. The run records which stages used fewer peers. Chair output must still pass scientific checks; persistence or provenance errors still stop execution.
- **Stage failure:** new runs retain discovery scores from their actual scientific record. Failed stages roll back, consume their attempts, and remain execution failures. The former whole-run zero penalty is retained as a separate diagnostic.
- **vLLM restart September 10:** all 120 Qwen/Gemma runs start fresh with the repaired bookkeeping and failure policies. Old run directories were deleted at user request; aggregate cost/status records remain. Explicit reasoning boundaries are parsed before stage validation. Retry prompts include errors and rejected-response excerpts. Both models explicitly enable thinking with recommended sampling; only two consecutive truncations permit disabling it on the final retry. Normal calls request medium reasoning. Fallback requests are visible in call journals.

The comparison still includes 136 retained original Codex runs and 104 repaired Codex runs. The new vLLM parsing/retry changes are separately versioned. Earlier Gemma threshold interpretation errors remain a scientific issue to assess, not a reason to change the hidden evaluator.

## Progress by condition

| Model | Workflow | Retained original | Replacement | Queued | Active | Paused | All stages completed | Finished with errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Astra | deliberative | 10 | 10 | 0 | 0 | 0 | 20 | 0 |
| Astra | persistent | 12 | 8 | 0 | 0 | 0 | 20 | 0 |
| Astra | sequential | 13 | 7 | 0 | 0 | 0 | 19 | 1 |
| Gemma 4 31B | deliberative | 0 | 20 | 0 | 0 | 0 | 18 | 2 |
| Gemma 4 31B | persistent | 0 | 20 | 0 | 0 | 0 | 18 | 2 |
| Gemma 4 31B | sequential | 0 | 20 | 0 | 0 | 0 | 20 | 0 |
| Luna | deliberative | 11 | 9 | 0 | 0 | 0 | 19 | 1 |
| Luna | persistent | 13 | 7 | 0 | 0 | 0 | 18 | 2 |
| Luna | sequential | 13 | 7 | 0 | 0 | 0 | 20 | 0 |
| Qwen 3.8 27B | deliberative | 0 | 20 | 0 | 0 | 0 | 20 | 0 |
| Qwen 3.8 27B | persistent | 0 | 20 | 0 | 0 | 0 | 20 | 0 |
| Qwen 3.8 27B | sequential | 0 | 20 | 0 | 0 | 0 | 20 | 0 |
| Sol | deliberative | 5 | 15 | 0 | 0 | 0 | 19 | 1 |
| Sol | persistent | 12 | 8 | 0 | 0 | 0 | 20 | 0 |
| Sol | sequential | 9 | 11 | 0 | 0 | 0 | 20 | 0 |
| Terra | deliberative | 11 | 9 | 0 | 0 | 0 | 19 | 1 |
| Terra | persistent | 13 | 7 | 0 | 0 | 0 | 20 | 0 |
| Terra | sequential | 14 | 6 | 0 | 0 | 0 | 20 | 0 |

## Scientific performance so far

Includes finished runs with errors; active and queued runs are excluded. R and P are percentages. **Scientific F1\*** uses actual accepted, tested, independently confirmed claims for every model, without the former whole-run failure penalty. **Penalty F1\*** applies that old zero rule to every model for comparison. The saved original scores are not rewritten. Both columns describe the realized trace, not what an error-free rerun would have achieved.

The two dataset versions receive equal weight among those available; runs within each version receive equal weight. A dash means insufficient observations or a missing evidence class. Uneven completion makes these interim comparisons provisional. The performance table combines retained and replacement runs; the preceding table identifies their counts.

| Model | Workflow | Finished | Exp / Sur | Recall R % | Precision P % | Scientific F1* | Penalty F1* | Exploration E | Evidence response B | Runs with peer fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Astra | deliberative | 20 | 10 / 10 | 67.5 | 99.2 | 80.3 | 80.3 | 65.5 | 97.8 | 0 |
| Astra | persistent | 20 | 10 / 10 | 67.5 | 95.6 | 79.0 | 79.0 | 65.5 | 93.8 | 0 |
| Astra | sequential | 20 | 10 / 10 | 66.7 | 96.7 | 78.3 | 74.4 | 65.9 | 91.8 | 0 |
| Gemma 4 31B | deliberative | 20 | 10 / 10 | 56.7 | 97.8 | 70.7 | 62.8 | 63.6 | 64.1 | 0 |
| Gemma 4 31B | persistent | 20 | 10 / 10 | 44.4 | 90.8 | 57.8 | 51.6 | 63.0 | 62.5 | 0 |
| Gemma 4 31B | sequential | 20 | 10 / 10 | 47.8 | 98.5 | 61.4 | 61.4 | 59.9 | 63.9 | 0 |
| Luna | deliberative | 20 | 10 / 10 | 7.2 | 64.3 | 11.5 | 11.5 | 18.3 | 80.7 | 0 |
| Luna | persistent | 20 | 10 / 10 | 17.5 | 65.9 | 24.4 | 20.8 | 31.7 | 82.0 | 0 |
| Luna | sequential | 20 | 10 / 10 | 5.6 | 67.2 | 8.0 | 8.0 | 24.1 | 78.4 | 0 |
| Qwen 3.8 27B | deliberative | 20 | 10 / 10 | 55.0 | 100.0 | 69.5 | 69.5 | 65.0 | 60.1 | 0 |
| Qwen 3.8 27B | persistent | 20 | 10 / 10 | 60.0 | 97.5 | 73.5 | 73.5 | 64.7 | 61.9 | 0 |
| Qwen 3.8 27B | sequential | 20 | 10 / 10 | 50.0 | 93.0 | 63.7 | 63.7 | 63.6 | 59.7 | 0 |
| Sol | deliberative | 20 | 10 / 10 | 43.6 | 91.1 | 54.9 | 51.1 | 47.3 | 91.6 | 0 |
| Sol | persistent | 20 | 10 / 10 | 62.5 | 93.5 | 74.0 | 74.0 | 63.2 | 83.4 | 0 |
| Sol | sequential | 20 | 10 / 10 | 56.4 | 91.8 | 67.7 | 67.7 | 60.5 | 90.2 | 0 |
| Terra | deliberative | 20 | 10 / 10 | 44.7 | 93.4 | 57.5 | 53.6 | 59.6 | 84.8 | 1 |
| Terra | persistent | 20 | 10 / 10 | 55.6 | 97.3 | 68.9 | 68.9 | 62.4 | 85.1 | 0 |
| Terra | sequential | 20 | 10 / 10 | 53.6 | 94.0 | 66.5 | 66.5 | 63.2 | 87.3 | 0 |

## Output tokens

**92,454,705 observed output tokens** across the selected comparison so far, including active runs. **42 returned calls lack output usage.** These are provider-reported totals; they are not a bill or a complete account of interrupted requests.

Counts include peers, chairs, repairs, and returned retries. Reasoning tokens are already included in reported output tokens. No Codex native event totals are added to coordinator totals here, avoiding double counting. The final audit will replace coordinator totals with reconciled native completion totals where needed. Requests interrupted before a durable response and internal attempts without usage remain unknown.

| Model | Workflow | Observed output tokens, active + finished | Mean observed / finished run | Returned calls missing usage | Internal attempts without separate usage |
|---|---|---:|---:|---:|---:|
| Astra | deliberative | 5,000,191 | 250009.5 | 4 | 2 |
| Astra | persistent | 1,651,608 | 82580.4 | 1 | 1 |
| Astra | sequential | 1,554,669 | 77733.4 | 6 | 0 |
| Gemma 4 31B | deliberative | 16,522,613 | 826130.7 | 0 | 0 |
| Gemma 4 31B | persistent | 4,117,087 | 205854.4 | 0 | 0 |
| Gemma 4 31B | sequential | 6,273,409 | 313670.5 | 0 | 0 |
| Luna | deliberative | 5,082,041 | 254102.0 | 0 | 3 |
| Luna | persistent | 1,750,649 | 87532.4 | 0 | 0 |
| Luna | sequential | 1,800,771 | 90038.6 | 0 | 1 |
| Qwen 3.8 27B | deliberative | 15,889,652 | 794482.6 | 0 | 0 |
| Qwen 3.8 27B | persistent | 3,657,188 | 182859.4 | 0 | 0 |
| Qwen 3.8 27B | sequential | 5,122,344 | 256117.2 | 0 | 0 |
| Sol | deliberative | 11,385,537 | 569276.8 | 14 | 6 |
| Sol | persistent | 3,590,003 | 179500.1 | 0 | 1 |
| Sol | sequential | 3,599,479 | 179974.0 | 0 | 4 |
| Terra | deliberative | 3,189,848 | 159492.4 | 15 | 1 |
| Terra | persistent | 1,099,211 | 54960.6 | 1 | 0 |
| Terra | sequential | 1,168,405 | 58420.2 | 1 | 0 |

**Deleted vLLM run overhead:** 233,771,882 observed output tokens in the last pre-deletion snapshot; this is a lower bound because in-flight calls may not have returned. Old vLLM traces were deleted by request. These tokens are excluded from the fresh comparison.

**Additional erased unmasked-run overhead:** 29,302,795 observed output tokens in the September 10 batch replaced by recommended-sampling runs. Excluded from current performance and token totals; interrupted unreturned calls remain unknown.

Fresh vLLM retry-smoke checks used 28,714 observed output tokens, excluded from the comparison. Other preliminary smoke attempts and the thinking-toggle check are saved separately in outputs/vllm_restart_20260910.

**Separate restart overhead:** superseded original Codex traces contain 7,701,048 provider-recorded output tokens, plus 134,662 confirmed discarded-response tokens recovered by the selection audit. Their 107 missing-usage call records overlap that recovered amount; full native reconciliation remains pending. Live prelaunch checks used 22,412 observed output tokens. These amounts are excluded from the selected comparison above, but preserved as experiment costs.

## What the outcomes mean

- **R — recovery/recall:** fraction of embedded findings recovered, with equal weight to expected, neutral, and surprising findings. Recovery requires acceptance, a valid run-time test, and independent evaluator confirmation.
- **P — confirmed-claim fraction / precision:** confirmed, tested accepted claims divided by all accepted claims. Agents can accept before validation; independent confirmation is part of scoring. Additional valid findings can count toward P.
- **F1\*:** harmonic mean of R and P, multiplied by 100. The asterisk distinguishes category-balanced recovery from ordinary F1. Scores are calculated per run before averaging; no accepted claims yields unavailable P and zero F1\*.
- **E — exploration coverage:** for each iteration, calculate the fraction of findings validly tested in each of the three finding categories. Average across categories and all 25 iterations, then multiply by 100. Early tests earn more credit; repeating a test adds none.
- **B — evidence responsiveness:** score whether the explicit assessment at the response deadline agrees with supportive evidence (accept), excluding evidence (reject), or ambiguous evidence (unresolved). Missing due assessments score zero. Compute each evidence class's accuracy within each run, average runs and versions, then average the three classes and multiply by 100. If a class has no eligible observations, B is unavailable. Invalid evidence or deadlines the run never reached are excluded.
- The public clinical threshold remains a 10% relative outcome difference. The unchanged private reference compares the claim-oriented validation interval with 0.10 log-PFS units, approximately a 10.5% geometric-mean difference. Lower bound above that reference supports acceptance; upper bound below it supports rejection; an interval spanning it calls for unresolved. This calculation is kept out of agent instructions.

Persistent mode keeps bounded committed conversation plus the complete ledger and notes. Sequential mode starts a fresh conversation for each stage with the same ledger and notes. Deliberative mode ordinarily uses two independent drafts and one chair per stage; only the chair commits actions. It uses about three times as many participant calls for the same scientific action budget. Fallback stages are flagged above.

These are preliminary results on one clinical dataset pair. They do not establish general model rankings or a causal benefit/harm of deliberation. Retained old-code runs and repaired-code runs must remain distinguishable in subsequent analyses.

## Artifacts and progress logs

- [Codex replacement progress log](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260909_codex_repair/control/progress.log)
- [Local-model continuation progress log](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260910_vllm_recommended/control/gemma_8060_expansion/continuation/progress.log)
- [Gemma camus:8060 progress log (paused)](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260910_vllm_recommended/control/gemma_8060_expansion/new_endpoint/progress.log)
- [Combined progress log](/data1/ken/onc-co-scientist/outputs/clinical_workflow_restart_20260909/progress.log)
- [Replacement selection and native evidence](/data1/ken/onc-co-scientist/outputs/clinical_workflow_restart_20260909/selection_audit.json)
- [Pause-time snapshot](/data1/ken/onc-co-scientist/outputs/clinical_workflow_restart_20260909/snapshot.json)
- [Frozen replacement configuration](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260909_codex_repair/config.yaml)
- [Frozen replacement hashes](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260909_codex_repair/frozen_manifest.json)
- [Original interim report](/data1/ken/onc-co-scientist/interim_results_9-8-26.md)
- [Deliberative review](/data1/ken/onc-co-scientist/deliberative_workflow_review_2026-09-09.md)
