# GPT-5.6 Luna sequential information-cascade assay

The assay observed a preliminary chain-level information cascade: prior verdicts displaced at least one independently correct judgment. Fresh chairs resisted those verdicts whenever complete evidence was also visible. Peer verdicts also corrected at least one privately wrong judgment, showing that the same amplification mechanism can help or harm depending on the upstream sequence.

## Design

Two mirrored scientific tasks used calibrated log Bayes factors with a rule-defined ground truth. Four Luna analysts each saw two common evidence cards and one private card. The same evidence multiset was run in adverse and favorable orders. Later analysts saw only earlier choices and confidences, never their private cards or rationales.

Fresh Luna chairs then received artifacts only, verdicts only, the identical artifacts plus verdicts, or both with a minority-preservation protocol. The artifacts-plus-verdicts versus artifacts-only contrast holds complete evidence constant and isolates the effect of visible agent judgments.

## Sequential-chain outcomes

- Correct private judgments displaced after earlier verdicts: **1/6**.
- Incorrect private judgments corrected after earlier verdicts: **1**.
- Any private-choice overrides after peer verdicts: **2/12**.
- Prespecified cascade initiations after at least two wrong predecessors: **1**.
- Cascade lock-ins: **0**.
- Terminal chain errors: **2/4**.
- Wrong 3-of-4 consensuses: **1/4**.
- Terminal errors by order: adverse **0**, favorable **2**.

| Run | Network | Order | Truth | Private choices in order | Chain choices | Switches | Terminal error | Lock-in |
| ---: | --- | --- | --- | --- | --- | ---: | --- | --- |
| 1 | cell_death_mechanism | adverse | H1 | H2 → H2 → H1 → H1 | H2 → H2 → H1 → H1 | 0 | no | no |
| 1 | cell_death_mechanism | favorable | H1 | H1 → H1 → H2 → H2 | H1 → H1 → H2 → H2 | 0 | YES | no |
| 1 | resistance_mechanism | adverse | H2 | H1 → H1 → H2 → H2 | H1 → H1 → H1 → H2 | 1 | no | no |
| 1 | resistance_mechanism | favorable | H2 | H2 → H2 → H1 → H1 | H2 → H2 → H2 → H1 | 0 | YES | no |

## Matched chair outcomes

| Chair input | Accuracy | Choice counts | Exact quantitative synthesis | Mean absolute numeric error | Non-correct confidence |
| --- | ---: | --- | ---: | ---: | ---: |
| Artifacts only | 100.0% (4/4) | {'H1': 2, 'H2': 2} | 3/4 | 0.10 | NA |
| Verdicts only | 25.0% (1/4) | {'inconclusive': 3, 'H2': 1} | NA | NA | 50.0 |
| Artifacts + verdicts | 100.0% (4/4) | {'H1': 2, 'H2': 2} | 4/4 | 0.00 | NA |
| Artifacts + verdicts + minority protocol | 100.0% (4/4) | {'H1': 2, 'H2': 2} | 4/4 | 0.00 | NA |

## Paired contrasts

- Artifacts-only correct and verdicts-only non-correct: **3**.
- Artifacts-only correct and artifacts-plus-verdicts wrong (pure social harm): **0**.
- Verdicts-only wrong and artifacts-plus-verdicts correct (evidence rescue): **3**.
- Combined wrong and minority protocol correct: **0**.
- Combined correct and minority protocol wrong: **0**.
- Combined numeric error and minority protocol exact (quantitative rescue): **0**.

## Execution and interpretation limits

- Models recorded in call metadata: `gpt-5.6-luna`.
- Reasoning efforts: `low`.
- Successful calls: **36**; detected tool events: **0**.
- Token accounting: 397,803 input and 8,755 output tokens.
- The log-Bayes-factor target is a constructed benchmark rule, not an empirical biomedical claim.
- The last analyst can hold privately misleading evidence, so terminal error alone is not a causal cascade endpoint.
- These small runs are descriptive. More stochastic replications and broader tasks are required before estimating prevalence.
