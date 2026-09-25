# Does CAA accelerate acceptance generally?

The matched traces do not show a uniform acceptance speedup. The clearest advantage concerns formal acceptance of findings contradicting recorded prior expectations. This is suggestive, not evidence that the effect is specific to surprise.

## Method

`acceptance_latency.py`, executed with `/data1/ken/envs/gptoss3/bin/python`, generates `matched_acceptance_latency.json` from all eight final pilot reports. It matches the exact comparison within each CAA/control condition pair, takes its first discovery result, and requires valid evidence whose confidence interval lies entirely beyond the meaningful-effect threshold in either direction in both arms. Acceptance means the first formal acceptance of the evidence-supported directional claim, including a newly registered reversed claim when necessary. Delay starts at the first evidence, not at run initiation. Zero iteration delay means acceptance in the same iteration, possibly at a later stage.

There are 21 matched comparisons, all unconditional main effects with pre-evidence expectations recorded in both arms. Classification uses those recorded expectations, not the expected/surprising dataset label. One comparison has different expectations between arms and is excluded from the concordant/contradictory contrast. The other groups are:

| Relation to recorded expectation | Matched comparisons | CAA delay, iterations | Control delay, iterations |
| --- | ---: | --- | --- |
| Agrees | 4 | 0, 0, 0, 1 (mean 0.25) | 0, 0, 0, 0 (mean 0) |
| Contradicts | 3 | 1, 1, 1 (mean 1) | 5, 8, 8 (mean 7) |
| No directional expectation | 13 | Ten timing ties; one slower acceptance; two accepted only by CAA within the run | Two never accepted within 25 iterations |

The contradictory comparisons are expected/unmasked ECOG and surprising/unmasked ECOG and NLR. The concordant comparisons are expected/unmasked CRP, NLR, and stage IV, plus surprising/unmasked stage IV. In the last case CAA took one extra iteration after evidence and control accepted in the evidence iteration.

In the no-expectation group, surprising/masked feature_036 took eight iterations after evidence under CAA versus zero under control. Conversely, CAA accepted corrected negative claims for feature_025 and feature_028 that control never formally accepted. Comparisons never accepted remain censored; averaging only observed acceptances would bias the arm comparison.

## Limits and interpretation

- Concordant control findings were already accepted at the first appraisal after evidence. This floor leaves no opportunity to observe faster acceptance there. A general increase in willingness to accept could still be compatible with these results.
- The three contradictory findings come from two paired trajectories, and ECOG repeats the same embedded anomaly across the two dataset backgrounds. These are not three independent replications.
- Contradiction of the prior also requires correcting the registered hypothesis direction in these examples. Better hypothesis revision or ledger follow-through is an alternative to a mechanism specifically responsive to biomedical surprise. Both arms already described the contrary directions in their first appraisal narratives.
- This is a post-hoc subset of comparisons explored by both arms with strong initial evidence. It does not measure acceptance behavior across every proposed claim, especially weak or ambiguous findings, and the arms' preceding histories can differ.

A discriminating follow-up would use repeated matched prompts, vary prior agreement and initial claim direction separately, and include evidence strengths that avoid immediate acceptance in all controls. No additional generation was launched for this audit.
