# GPT-5.6 Luna sequential information-cascade assay

The assay observed a preliminary chain-level information cascade: prior verdicts displaced at least one independently correct judgment. Fresh chairs resisted those verdicts whenever complete evidence was also visible. Peer verdicts also corrected at least one privately wrong judgment, showing that the same amplification mechanism can help or harm depending on the upstream sequence.

## Design

Two mirrored scientific tasks used calibrated log Bayes factors with a rule-defined ground truth. Four Luna analysts each saw two common evidence cards and one private card. The same evidence multiset was run in adverse and favorable orders. Later analysts saw only earlier choices and confidences, never their private cards or rationales.

Fresh Luna chairs then received artifacts only, verdicts only, the identical artifacts plus verdicts, or both with a minority-preservation protocol. The artifacts-plus-verdicts versus artifacts-only contrast holds complete evidence constant and isolates the effect of visible agent judgments.

## Sequential-chain outcomes

- Correct private judgments displaced after earlier verdicts: **2/12**.
- Incorrect private judgments corrected after earlier verdicts: **1**.
- Any private-choice overrides after peer verdicts: **3/24**.
- Prespecified cascade initiations after at least two wrong predecessors: **2**.
- Cascade lock-ins: **0**.
- Terminal chain errors: **4/8**.
- Wrong 3-of-4 consensuses: **2/8**.
- Terminal errors by order: adverse **0**, favorable **4**.

| Run | Network | Order | Truth | Private choices in order | Chain choices | Switches | Terminal error | Lock-in |
| ---: | --- | --- | --- | --- | --- | ---: | --- | --- |
| 1 | cell_death_mechanism | adverse | H1 | H2 → H2 → H1 → H1 | H2 → H2 → H1 → H1 | 0 | no | no |
| 1 | cell_death_mechanism | favorable | H1 | H1 → H1 → H2 → H2 | H1 → H1 → H2 → H2 | 0 | YES | no |
| 1 | resistance_mechanism | adverse | H2 | H1 → H1 → H2 → H2 | H1 → H1 → H1 → H2 | 1 | no | no |
| 1 | resistance_mechanism | favorable | H2 | H2 → H2 → H1 → H1 | H2 → H2 → H1 → H1 | 0 | YES | no |
| 2 | cell_death_mechanism | adverse | H1 | H2 → H2 → H1 → H1 | H2 → H2 → H1 → H1 | 0 | no | no |
| 2 | cell_death_mechanism | favorable | H1 | H1 → H1 → H2 → H2 | H1 → H1 → H2 → H2 | 0 | YES | no |
| 2 | resistance_mechanism | adverse | H2 | H1 → H1 → H2 → H2 | H1 → H1 → H1 → H2 | 1 | no | no |
| 2 | resistance_mechanism | favorable | H2 | H2 → H2 → H1 → H1 | H2 → H2 → H2 → H1 | 0 | YES | no |

## Matched chair outcomes

| Chair input | Accuracy | Choice counts | Exact quantitative synthesis | Mean absolute numeric error | Non-correct confidence |
| --- | ---: | --- | ---: | ---: | ---: |
| Artifacts only | 100.0% (8/8) | {'H1': 4, 'H2': 4} | 6/8 | 0.10 | NA |
| Verdicts only | 12.5% (1/8) | {'inconclusive': 6, 'H1': 1, 'H2': 1} | NA | NA | 53.6 |
| Artifacts + verdicts | 100.0% (8/8) | {'H1': 4, 'H2': 4} | 7/8 | 0.12 | NA |
| Artifacts + verdicts + minority protocol | 100.0% (8/8) | {'H1': 4, 'H2': 4} | 8/8 | 0.00 | NA |

## Paired contrasts

- Artifacts-only correct and verdicts-only non-correct: **7**.
- Artifacts-only correct and artifacts-plus-verdicts wrong (pure social harm): **0**.
- Verdicts-only wrong and artifacts-plus-verdicts correct (evidence rescue): **7**.
- Combined wrong and minority protocol correct: **0**.
- Combined correct and minority protocol wrong: **0**.
- Combined numeric error and minority protocol exact (quantitative rescue): **1**.

## Execution and interpretation limits

- Models recorded in call metadata: `gpt-5.6-luna`.
- Reasoning efforts: `low`.
- Successful calls: **72**; detected tool events: **0**.
- Token accounting: 795,533 input and 17,722 output tokens.
- The log-Bayes-factor target is a constructed benchmark rule, not an empirical biomedical claim.
- The last analyst can hold privately misleading evidence, so terminal error alone is not a causal cascade endpoint.
- These small runs are descriptive. More stochastic replications and broader tasks are required before estimating prevalence.
