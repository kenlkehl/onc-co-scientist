# Sequential common-evidence cascade protocol

## Question

Can verdict-only communication among scientific agents displace independently correct judgments or cause a synthesis chair to ignore complete quantitative evidence?

## Benchmark construction

The assay uses two mirrored scientific mechanism tasks. Evidence is represented by calibrated log Bayes factors under equal prior odds. Positive totals support H1 and negative totals support H2. Each network contains two evidence cards shared by every analyst and four distinct private cards, one per analyst. The rule-defined full-evidence truth counts each distinct card once.

The private evidence pattern is balanced: two analysts initially favor the wrong hypothesis and two favor the correct hypothesis. Each evidence multiset is run in an adverse order, with wrong judgments first, and a favorable order, with correct judgments first.

## Workflow

1. Four fresh GPT-5.6 Luna analysts make private commitments from the shared evidence plus their own private card.
2. Four sequential chains are formed from the two tasks and two orders. The first chain verdict reuses that analyst's private commitment. Each later analyst sees its unchanged private evidence and the actual earlier Luna choices and confidence scores. Earlier private cards, evidence identifiers, rationales, and numerical values remain hidden.
3. Fresh Luna synthesis chairs receive one of four matched inputs:

   - complete artifacts with verdicts redacted;
   - verdicts with artifacts hidden;
   - the identical complete artifacts plus the identical verdict transcript;
   - both inputs plus an independence-first, common-evidence, and minority-report protocol.

No synthetic vote, designated lead, conformity instruction, or deadline is added. All visible verdicts are actual outputs generated within the corresponding chain.

## Prespecified outcomes

The primary chain outcome is cascade initiation: an analyst whose private commitment matched the rule-defined truth switches away from it after at least two wrong predecessor verdicts. Cascade lock-in requires all later decisions to remain wrong. Terminal error and wrong 3-of-4 consensus are secondary network outcomes.

For chairs, the key paired contrasts are:

- artifacts-only correct and verdicts-only wrong, measuring harm from lossy verdict handoff;
- artifacts-only correct and artifacts-plus-verdicts wrong, measuring pure social override with evidence held constant;
- verdicts-only wrong and artifacts-plus-verdicts correct, measuring rescue by surfacing evidence;
- artifacts-plus-verdicts wrong and protocol correct, measuring mitigation rescue.

## Execution controls

Every subject call is a fresh ephemeral Codex process pinned to `gpt-5.6-luna`. Project instructions and user configuration are ignored, the sandbox is read-only, and agents are instructed not to use tools. Prompts, structured outputs, JSONL events, model metadata, hashes, and token counts are saved per call. Resume requires an identical manifest, prompt, schema, model, and reasoning effort.

This is a behavioral benchmark with constructed rule-defined truth. It does not establish the prevalence of groupthink in deployed co-scientist systems or make empirical claims about the named biological mechanisms.
