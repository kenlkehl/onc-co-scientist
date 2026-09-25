# CAA output-behavior audit — September 20, 2026

## Scope and interpretation

Post-hoc descriptive analysis of the eight completed pilot cells, one CAA/control pair in each expected/surprising × masked/unmasked condition. It excludes the earlier acceptance run. There are four paired trajectories, not 800 independent observations. The arms use sampling and subsequently see different hypotheses, evidence, and conversational histories. These data can suggest behavioral effects but cannot establish a reproducible causal effect of the steering vector.

`analyze.py` generates `metrics.json`, `metrics.csv`, `acceptance_timelines.json`, and `excerpts.md` from saved artifacts using `/data1/ken/envs/gptoss3/bin/python`. It uses the last replay transcript in each cell, preventing resumed stages from being counted twice, and asserts 100 committed stages per cell. Generated-token totals include thought tokens and response repairs; separate committed-token totals exclude repair attempts not committed. Final-text size is measured in characters; narrative word counts use only committed stage narratives. Initial message arrays have identical hashes within all four CAA/control pairs.

## Meaning of expected/surprising dataset labels

The frozen pair identifies `nsclc_nlr_ge_3_pfs` as its focal discovery. Only NLR changes direction between the dataset versions: negative in expected, positive in surprising. ECOG is a separate discovery deliberately specified as positive, contrary to its stored negative expectation, in both versions. Therefore expected/unmasked can contain an unexpected ECOG association. “Reversal” in the acceptance tables means revising the model’s initial hypothesis after seeing evidence; it does not imply that ECOG reverses between datasets. The two ECOG comparisons concern the same embedded anomaly under two NLR backgrounds and are not independent examples of different biomedical reversals.

## Most interesting pattern: earlier formal acceptance of reversed claims

In both unmasked conditions, the initial appraisals in both arms already recognize that the numerical ECOG result contradicts their negative-direction prior. In surprising/unmasked, both also recognize the reversed NLR result in iteration 1. CAA then formally accepts revised positive-direction claims earlier:

| Condition and claim | CAA first acceptance | Control first acceptance |
| --- | --- | --- |
| Expected/unmasked, ECOG positive association | Iteration 2, appraise (H13) | Iteration 6, appraise (H47) |
| Surprising/unmasked, ECOG positive association | Iteration 2, appraise (H13) | Iteration 9, synthesize (H66) |
| Surprising/unmasked, NLR positive association | Iteration 2, appraise (H14) | Iteration 9, synthesize (H67) |

This is faster formal follow-through on recognized contrary evidence, rather than a difference in whether the model can read the effect direction. It is a candidate effect consistent with the intervention's intention, but the small unreplicated sample and differing research paths leave ordinary sampling variation and bookkeeping differences as alternatives. Final focal recovery is the same in every condition, so that endpoint has no discriminating variation in this pilot.

## The surprising/masked F1 advantage reflects ledger follow-through

CAA's scientific F1 is 80 versus control's 50 in surprising/masked. The control's final narrative correctly describes negative effects for feature_025 and feature_028, but its only unconditional claims for those exposures retain positive directions (H5 and H24). It rejects those claims and never registers/accepts the corresponding negative-direction unconditional claims. CAA registers the reversed claims and accepts them (H13 at iteration 2; H61 at iteration 6). Thus the control did recognize these associations in prose. The score difference is not evidence that only CAA noticed them, nor, by itself, that biomedical-prior anchoring caused the omission: this cell has masked labels.

## Secondary pattern: more accepted subgroup-specific claims in unmasked runs

| Condition | CAA accepted claims | Control accepted claims | CAA / control accepted conditioned claims |
| --- | --- | --- | --- |
| Expected/unmasked | 17 | 5 | 12 / 0 |
| Surprising/unmasked | 11 | 6 | 5 / 0 |
| Expected/masked | 6 | 6 | 1 / 0 |
| Surprising/masked | 6 | 4 | 0 / 0 |

The extra unmasked CAA acceptances mostly express known predictor associations within additional subsets (for example, marker effects within high-CRP, stage, histology, or treatment strata), rather than extra embedded discoveries. All 17 expected/unmasked CAA acceptances were independently confirmed; 9/11 surprising/unmasked CAA acceptances were confirmed, versus 5/6 for control. This does not establish a uniformly better-calibrated or more skeptical output style.

## No consistent general shift in verbosity, repairs, or exploration

| Measure, aggregated across four runs per arm | CAA | Control |
| --- | --- | --- |
| Generated tokens, including repairs | 646,672 | 642,052 |
| Generated tokens in committed responses | 587,491 | 586,236 |
| Final-response characters, committed responses | 605,810 | 614,899 |
| Words in committed narratives | 48,311 | 49,719 |
| Response repairs | 17 | 19 |
| Distinct comparisons tested, summed across cells | 316 | 302 |
| Voluntary validation slots used | 19 | 16 |

Total generated tokens differ by +0.72%; committed generated tokens by +0.21%. Final-response character counts differ by -1.48%. Repairs favor CAA in two conditions and control in two. Compared with control, CAA tests 5 fewer comparisons in expected/masked, 20 more in surprising/masked, 11 fewer in surprising/unmasked, and 10 more in expected/unmasked. These directions are mixed. All 800 committed responses use JSON code fences.

Both arms retain recognizable scientific prose with quantified effects, validation discussion, and final-report summaries. In the reviewed final narratives, both can overstate null interaction tests as proof of independence and describe an incomplete exploration as exhaustive. A more restrained or more evidence-calibrated rhetorical style is not demonstrated.

## Ambiguous evidence remains a weakness in both arms

Class-balanced responsiveness components: CAA scores supported 19/19, excluded 8/8, ambiguous 0/3; control scores supported 16/16, excluded 7/7, ambiguous 0/3. These denominators arise from different chosen comparisons and are descriptive, not matched event-level treatment comparisons. Both fail all three eligible ambiguous-evidence events in their respective trajectories. Missing ambiguous events in the expected cells remain missing, not zeros.

## Conclusion

A plausible signal is faster conversion of contradictory evidence into revised accepted claims, with more subgroup-specific claim retention in unmasked runs. There is no broad consistent shift in length, formatting reliability, exploration volume, or handling of ambiguous evidence. Replicated fixed-prompt comparisons would help isolate the direct response effect; replicated full trajectories would assess whether the formal-follow-through pattern survives sampling variability. Neither is run as part of this audit.

## Full condition accounting for claim reversal

The three unmasked findings above compare four runs (two conditions × two arms), not three runs. The other four runs are the two masked conditions × two arms. For the same simple directional-correction behavior:

| Condition | Claim being corrected | CAA first corrected acceptance | Control first corrected acceptance |
| --- | --- | --- | --- |
| Expected/unmasked | ECOG: negative to positive | Iteration 2 appraise | Iteration 6 appraise |
| Surprising/unmasked | ECOG: negative to positive | Iteration 2 appraise | Iteration 9 synthesize |
| Surprising/unmasked | NLR: negative to positive | Iteration 2 appraise | Iteration 9 synthesize |
| Expected/masked | feature_006: positive to negative | Iteration 2 appraise | Iteration 2 explore |
| Expected/masked | feature_025: positive to negative | Iteration 2 appraise | Iteration 2 explore |
| Expected/masked | feature_028: positive to negative | Iteration 4 appraise | Iteration 3 appraise |
| Surprising/masked | feature_025: positive to negative | Iteration 2 appraise | Never within 25 iterations |
| Surprising/masked | feature_028: positive to negative | Iteration 6 synthesize | Never within 25 iterations |

In expected/masked, feature_028's first contrary result occurred at iteration 3 for CAA and iteration 2 for control, so both corrected it the next iteration. That condition does not support faster correction by CAA. The first two masked expected corrections are also in the same iteration, with control earlier within that iteration. In surprising/masked, feature_006 was accepted as positive at iteration 1 by both arms and needed no reversal. In expected/unmasked, NLR was accepted as negative at iteration 1 by both arms and needed no reversal.

The initially registered masked hypotheses had anticipated_direction=0 (no directional expectation). Their positive directions therefore should not be described as biomedical priors. Evidence of better formal follow-through under masking cannot specifically establish a reduction in biomedical paradigm anchoring. Findings within one trajectory are correlated, and these examples are post-hoc rather than a preregistered latency endpoint.
