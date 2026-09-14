# Why do deliberative runs look worse?

**Assessment:** much of the apparent disadvantage is execution reliability and the whole-run failure penalty. There are also concrete examples of the chair discarding useful ideas or narrowing the investigation. The present data support both explanations; they do not establish that deliberation generally reduces scientific performance.

This review uses the same fixed snapshot as [the interim report](/data1/ken/onc-co-scientist/interim_results_9-8-26.md): September 8, 2026, 10:14 p.m. EDT; 130 finished runs. It does not incorporate later completions or change any experiment scores.

## Separate the failure penalty from the scientific record

The closest comparison is **sequential**, because sequential and deliberative both start each stage with a fresh conversation and the current ledger. Persistent also differs in conversation history. Below, each deliberative run is matched to the sequential run with the same model, dataset version, and repeat number. We average within version, then give observed versions equal weight. Negative numbers favor sequential.

| Model | Matched comparisons | F1* difference, recorded score | Difference before failure penalty |
|---|---:|---:|---:|
| Luna | 8 | +4.4 | -1.5 |
| Terra | 8 | -10.7 | -0.7 |
| Sol | 7 | -47.4 | -6.6 |
| Astra | 8 | -4.1 | +6.8 |
| Qwen | 1 | +0.0 | +0.0 |
| Gemma | 5 | -43.5 | +1.5 |

**Before failure penalty** means the frozen evaluator’s existing diagnostic discovery score, calculated from the decisions actually made, before an unrecovered stage error forces F1* to zero. It is an arithmetic decomposition—not an estimate of how the agents would have behaved without errors. Earlier errors can still affect later decisions. Qwen has only one matched comparison, from the surprising version.

For Sol, about **41 of the 47-point observed gap** comes from the failure penalty. Gemma’s apparent 44-point deficit becomes a small positive difference before that penalty; Astra also changes sign. Terra becomes nearly tied. Luna is close to sequential in either view.

Among comparisons where **both runs completed every stage**, Astra deliberative is +7.5 F1* points (5 comparisons), Luna −0.2 (7), Terra −10.6 (7), and Sol −21.9 (only 2). Gemma has one such comparison, tied; Qwen has none. These selected subsets are informative case studies, not unbiased estimates. Sol’s two clean comparisons consist of one near tie and one −44.1-point loss.

## Reliability: a known adapter defect amplified by a strict peer requirement

**All eight failed Codex deliberative runs had at least one stage blocked by an exhausted peer.** Across those runs, 14 stages were blocked this way. I inspected the 42 provider attempts underlying those exhausted peers: **all 42 ended in `turn.completed`, had a saved final response, and passed the required peer-stage form validator**. The frozen adapter misclassified them as failures after warnings. This is the already identified adapter defect; the working-source fix is already committed. [Native-event evidence](/data1/ken/onc-co-scientist/outputs/deliberation_review_2026-09-09/peer_veto_native_audit.json).

The coordinator requires both peers to produce valid drafts before calling the chair. Ten of these blocked stages failed on peer 1, so peer 2 was never reached; four failed on peer 2 after peer 1 had supplied a draft. Outer stage retries revisit the same cached peer attempts rather than grant another peer retry budget. Thus a peer failure can veto the entire stage, and one failed stage can zero the entire run’s F1*. Deliberative also makes about three times as many participant calls, increasing its exposure to provider problems.

This is **intentional fail-closed behavior**, not an accidental hidden retry bug: `test_exhausted_peer_is_not_silently_dropped` explicitly requires it. I ran that test against the frozen implementation; it passed. It is reasonable if the experimental condition strictly requires two peers, but its failures must be distinguished from failed scientific reasoning. [Coordinator](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/source/src/onc_co_scientist/expected_surprising/coordination.py:194) · [Existing test](/data1/ken/onc-co-scientist/tests/test_expected_surprising_experiment.py:308).

## Additional controller friction worth addressing

The three failed Gemma deliberative runs had 15 unrecovered stages: eight evidence-required assessment errors, five duplicate-claim/current-assessment conflicts, and two unknown-variable errors. These are not the Codex adapter problem.

Of the eight evidence-required errors, **one was an exact no-op**: the chair repeated that an untested claim was unresolved and active. **Two others kept the claim unresolved and only deferred its investigation.** Requiring empirical evidence for those bookkeeping actions is unnecessary friction. The other five did change the assessment to reject without evidence, so they should not be lumped together with the harmless cases. Allowing bookkeeping would not automatically salvage all three runs, which also had other errors. [Saved requests and comparison with current state](/data1/ken/onc-co-scientist/outputs/deliberation_review_2026-09-09/empty_evidence_assessments.json) · [Controller guard](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/source/src/onc_co_scientist/expected_surprising/workflow.py:260).

For future runs, distinguish investigation management and repeated unchanged assessments from empirical changes of conclusion. Preserve the original preregistration and evidence history. Duplicate-claim conflicts also deserve clearer, reference-specific repair feedback; silently rewriting a changed scientific assessment would be inappropriate.

## A real chair-selection dynamic appears in successful runs

I examined 630 committed chair calls: the initial explore stage in 36 deliberative runs, plus every stage in six successful runs selected to investigate missed surprising findings. The chair inputs contain the peer proposals described below, and its outputs omit or change them. This is a decision made by the chair, not a message lost by the harness.

**Terra, surprising version, repeat 2, iteration 8.** Peer 2 proposed a full-cohort, positive-direction NLR claim after observing the opposite of the initial expectation. The chair explicitly declined it, reasoning that validated stage-IV refinements already covered the finding and a broad reversal would add little. The run completed all 100 stages but never recovered the focal full-cohort NLR claim. [Chair input and output](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/runs/997d9404f1e89442__deliberative__terra_medium__r002/calls/i008-explore-chair-a1.json).

**Terra, surprising version, repeat 4, iteration 16.** Peer 2 again proposed the evidence-supported NLR reversal. The chair declined because it regarded reversals of closed claims after seeing evidence as insufficiently prospective, and chose familiar treatment questions instead. The protocol explicitly permits registering an evidence-motivated reversed claim; final confirmation uses independent data. This was a model-imposed restriction, not a controller rejection. That run also completed every stage without recovering the focal claim. [Chair input and output](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/runs/997d9404f1e89442__deliberative__terra_medium__r004/calls/i016-explore-chair-a1.json) · [Prompt allowing reversed claims](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/source/src/onc_co_scientist/expected_surprising/prompting.py:84).

**Sol, surprising version, repeat 4.** At the first explore stage, peer 2 proposed the full-cohort NLR comparison. The chair omitted it and favored the other draft’s treatment-centered portfolio. Across the reviewed 25 iterations, the chair never proposed that full-cohort comparison; the final trace records it as never tested. Its matched sequential run recovered it. This one clean run drives most of Sol’s clean-comparison disadvantage. [Initial chair input and output](/data1/ken/onc-co-scientist/data/expected_surprising_ledger/full_runs/20260908_clinical10pct_workflows/runs/997d9404f1e89442__deliberative__sol_medium__r004/calls/i001-explore-chair-a1.json).

A related Terra run changed a peer’s full-cohort NLR proposal into a stage-IV comparison. That can be a scientifically defensible question, but this benchmark credits exact target comparisons. Reduced exact-target recall can therefore reflect a narrower scientific scope, not an inability to identify any association.

These cases support a plausible dynamic: **the chair can narrow the search and suppress a useful minority proposal, especially a counterintuitive reversal**. They do not prove that it happens generally. Astra’s clean matched results improve with deliberation, and the negative Terra pattern is concentrated in the surprising condition: its three clean surprising comparisons all score below sequential, while three of four expected comparisons score above it. [Matched outcomes and run IDs](/data1/ken/onc-co-scientist/outputs/deliberation_review_2026-09-09/metrics.json).

## What this review rules out—and what I would change next

- The prior full audit found consistent stage accounting, report hashes, score calculations, and scientific budgets. I found no evidence here that the chair loses access to the current ledger or that peer analyses consume the chair’s analysis budget. Only the chair executes selected actions.
- The two initial drafts were not duplicates: none of the 36 reviewed initial-stage draft pairs had identical forms or identical sets of valid proposed comparisons. Their shared model, prompt, and requested temperature of zero did not eliminate useful variation. Invalid comparison structures were excluded only from the overlap diagnostic. [Peer/chair review](/data1/ken/onc-co-scientist/outputs/deliberation_review_2026-09-09/chair_review.json).
- This grid uses **two independent drafts followed by one chair decision**. With one peer round, the peers do not debate each other’s proposals. Results apply to that design, not every possible deliberative workflow.

**For the next experiment:** use the fixed adapter; make no-op assessments and investigation-only updates safe; explicitly choose whether a missing peer should fail the stage or allow a clearly labeled degraded execution; and test a chair instruction that preserves evidence-supported alternative directions for evaluation rather than dismissing them merely because they arose after evidence. Keep the same scientific budget and avoid target-specific hints. A second peer round or a persistent record of unselected proposals would be separate experimental conditions, not silent changes to the current one.

**Current experiment:** unchanged. No frozen code, prompts, scientific records, or scores were edited. This review does not justify treating the current aggregate deliberative deficit as a clean estimate of a scientific group effect.

## Reproducibility

[Metric extraction](/data1/ken/onc-co-scientist/outputs/deliberation_review_2026-09-09/metrics.py) · [Chair review script](/data1/ken/onc-co-scientist/outputs/deliberation_review_2026-09-09/chair_review.py) · [Per-run outcomes](/data1/ken/onc-co-scientist/outputs/deliberation_review_2026-09-09/metrics.json) · [Native provider evidence](/data1/ken/onc-co-scientist/outputs/deliberation_review_2026-09-09/peer_veto_native_audit.json)
