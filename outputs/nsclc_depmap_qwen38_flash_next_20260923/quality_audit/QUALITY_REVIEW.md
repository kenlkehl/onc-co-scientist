# Interim output quality review

The visible model responses are coherent and operationally usable. This is an interface and content spot-check, not an assessment of benchmark recovery or scientific correctness. Persistent runs are still in progress; the most advanced inspected runs had reached approximately iteration 6 of 25.

At the audited snapshot, there were **205 returned responses**, **192 committed stages**, and **13 rejected attempts (6.3%)** affecting **11 stages**. All 11 affected stages recovered within the configured repair allowance. No response was empty, raw response JSON parsed successfully, and there were no recorded transport/truncation errors or exhausted stages. Counts use each run's latest reconstructed transcript and unique call records, avoiding double-counting replayed transcripts.

Eight rejected attempts omitted usable motivating evidence for a refinement. Other errors were missing assessment fields, a misspelled variable (`stkg11_loss`), an unavailable evidence reference, conditioning on the exposure/outcome, and overlapping eligibility/subgroup variables. These are interface mistakes, with successful repairs, rather than garbled generation.

Responses were spot-checked across all four version/masking cells and all four scientific stages. They contain intelligible analysis plans, select registered comparisons, assess reported intervals, track uncertainty and null findings, request validation, and revise conclusions after validation. The visible prose does not show obvious degenerate repetition. Hidden reasoning is not included in this audit.

Concrete evidence checks:

- Unmasked expected, repeat 1, iteration 4 synthesis: accurately reports WRN validation estimate +0.821 and interval [0.766, 0.877], with sample sizes 125 and 1875, and correctly converts the direction-signed result to stronger dependency. These numbers match the supplied R53 card.
- Masked surprising, repeat 1, iteration 4 synthesis: reports opposite validation estimates for H23 (-0.0091) and H26 (-0.0039), matching R25 and R50, and closes those claims rather than treating them as validated discoveries.
- A scientific judgment caveat appears in masked expected, repeat 1, iteration 4 synthesis: H15 remains accepted because the discovery and validation point estimates agree, although both intervals cross zero (validation +0.0291, interval [-0.0380, 0.0961]). The response acknowledges the uncertainty, but the acceptance is debatable. This does not prevent the experiment from proceeding.

The repeated 10% significance guidance is present in the frozen harness prompt; it was not invented by the server. No settings, prompts, or scoring were changed during this review.

[Snapshot inventory](inventory.json) · [Sampled responses](sampled_responses.json)
