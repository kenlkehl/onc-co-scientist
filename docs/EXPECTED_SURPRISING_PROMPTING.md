# Simpler prompts for the expected/surprising task

The `appraisal-3.2.0` workflow uses `ledger-1.0.0` prompts and `stage-forms-1.0.0` responses. It moves record keeping into the controller while leaving scientific choices with the model. The numerical datasets, validation schedules and budgets, effect-size judgment instructions, and scoring formulas remain unchanged.

A subsequent budget update sets fresh full tasks to 25 iterations for both clinical and cell-line data. It does not alter these prompt forms, validation schedules, or scoring formulas. Existing packages retain their recorded budgets; current packaging instructions are in the [guide](EXPECTED_SURPRISING_GUIDE.md#running-and-inspecting-an-evaluation).

The previous interface mixed scientific work with a substantial formatting task. Models had to copy long result IDs, synchronize registration dictionaries, remember which fields were forbidden at each stage, and reconstruct an accepted list already implied by their assessments. Failed attempts were also replayed in subsequent prompts. Gemma sometimes repeated completed analyses in `executed_ids` during synthesis, causing the whole stage to be rejected.

## What changed

| Before | Now |
|---|---|
| One form for all four stages | Each stage exposes only its available actions |
| Agent invents hypothesis IDs | Controller assigns `H1`, `H2`, … |
| Long evidence hashes | Short stable references `R1`, `R2`, … |
| Separate hypothesis, direction, and initial-status dictionaries | Each proposal contains its own scientific fields |
| Assessments plus a second complete accepted list | Controller derives acceptance from assessments |
| Validation request must copy the discovery ID | Agent selects a claim; controller attaches its discovery result |
| Complete response/error history in each prompt | Current ledger, persistent research notes, latest narrative per stage, and current retry feedback |

The controller still writes the detailed original internal records for scoring and inspection. Old packages with `appraisal-3.1.0` still select the old prompt and response schema. Historical experiment outputs are not rewritten. Fresh full experiments can use the new package root; historical runs retain their original contracts.

## Examples of the new responses

An exploration response can propose a comparison without assigning an ID or maintaining parallel dictionaries. This is a schematic example using generic variable names; agents must use names from their supplied dataset.

```json
{
  "proposals": [{
    "outcome": "outcome_y",
    "exposure": "marker_positive",
    "direction": 1,
    "anticipated_direction": 0,
    "initial_status": "unresolved"
  }],
  "narrative": "I do not have a directional expectation; this comparison is worth exploring."
}
```

The default comparison is exposed value 1 versus comparator value 0. Eligibility and subgroup conditions, other exposure values, and interactions remain available. `direction` states the directional claim (+1 or −1). `anticipated_direction` records the agent's prior expectation about the raw contrast (+1, −1, or 0 for uncertain). These are different scientific concepts, so both remain explicit.

An analysis request is simply:

```json
{"run_analyses": ["H1", "H3"]}
```

An appraisal can assess the evidence displayed beside a claim and request validation:

```json
{
  "assessments": [{
    "claim": "H1",
    "status": "unresolved",
    "investigation": "active"
  }],
  "validate": "H1",
  "narrative": "The estimate is interesting, but its uncertainty warrants an independent check."
}
```

Synthesis uses the same small assessment object, without analysis or validation-request fields:

```json
{
  "assessments": [{
    "claim": "H1",
    "status": "accept",
    "investigation": "closed"
  }],
  "research_notes": "Next investigate whether the association differs across subgroups.",
  "narrative": "The accumulated evidence supports this association in the supplied dataset."
}
```

An agent can also accept based on discovery evidence alone. These examples do not prescribe when to accept, the effect size required, or whether to request validation.

## Evidence and memory

Every claim card contains its comparison, immutable initial expectation, current status, and all direct numerical evidence available so far. Evidence is signed to that card's claim direction. A reversed claim therefore shows the corresponding reversed estimate and interval without requiring another analysis. The ledger contains no evaluator cutoffs, target inventory, literature categories, or unreleased validation selections.

When `evidence` is omitted from an assessment, the controller attaches all evidence displayed on that claim's card. The prompt explicitly explains this convention. An agent can supply `"evidence": ["R1", "R3"]` to select a subset or related evidence. Evidence listed in a required new-result or response-deadline assessment must still be included. Unknown references and omitted required assessments remain errors.

A refinement names its earlier `parent`. If `motivating_evidence` is omitted, the parent's displayed evidence is attached. The audit distinguishes these controller-attached links from references explicitly selected by the agent. **Automatic attachment records what was supplied for the decision; it does not prove attention, understanding, or a specific reasoning process.** Prose remains available for inspection and is not numerically scored.

`research_notes` is an editable persistent notebook, not a required rewrite at every step. Omitting it preserves the existing notes. The latest successful narrative for each of the four stages is retained too. All earlier numerical evidence remains in the ledger. Raw prompts, responses and failures remain in the transcript, but are not all replayed to the model.

Failed attempts are rolled back. The next attempt sees the unchanged ledger and an error using the same short references. Failed proposals do not consume references, failed notes do not overwrite the notebook, and retries cannot buy new validation evidence.

## Limits and verification

This remains a structured research task. Models still must define coherent comparisons, distinguish a prior expectation from a directional claim, assess new evidence, and follow a small JSON form. The ledger grows with distinct claims and results; it is not a fixed-size memory. The interface should reduce avoidable errors, but a small smoke test cannot establish that model comparisons are free of interface effects.

Exhausted-stage scoring is unchanged: such failures remain visible and can zero primary recovery and F1*. This refactor improves the interface rather than changing the declared treatment of failures. Fresh experiments should use the new packages and be reported separately from the stopped experiments.

Implementation: [prompt forms and translation](../src/onc_co_scientist/expected_surprising/prompting.py), [controller integration](../src/onc_co_scientist/expected_surprising/workflow.py), and [verification record](../benchmarks/expected_surprising/workflow_v3_2/README.md).

The prompt builder runs in the common workflow before the provider call. All providers receive the same stage form and public ledger for a given state, including vLLM, Gemini/Vertex, Anthropic/Vertex, and Codex. Backend choice does not select a different scientific prompt.
