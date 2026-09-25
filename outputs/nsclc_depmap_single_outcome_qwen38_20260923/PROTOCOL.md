# NSCLC DepMap composite-outcome Qwen grid

40 persistent runs: ten repeats each of expected/masked, expected/unmasked,
surprising/masked, and surprising/unmasked. Each run has 25 iterations.
Run alongside the original 16-outcome campaign with 40 additional concurrent workers.

The dataset has 2,000 rows, 65 predictors, one identifier and ONE composite dependency
outcome. All original predictor values and six effect components are retained,
including the focal subgroup reversal and outside-subgroup effect. Intercept -0.25,
residual sigma 0.18 and absolute threshold 0.15 are retained. The composite adds the
six conditional signal components and one residual draw; it does not sum six noisy
outcome columns. Its marginal variance can differ from the individual endpoints.
Source categories and literature provenance are inherited, not reinterpreted as
evidence for a shared knockout. See private derivation.json and source_pair.json.

The independent DGP review passed. Calibration used 1,000 development repeats per
version and 100,000 reference rows: focal recovery was
98.9% expected and
99.7% surprising. All calibration
criteria passed. These are numerical checks, not agent benchmark results. Oracle
confirmation recovers all six findings for every naming/version/repeat combination.

The new run uses the corrected identifier filtering, absolute threshold guidance,
and preserved assay descriptions (appraisal-3.3.0 / ledger-1.1.0 / masking v2).
The original 16-outcome run retains its frozen older protocol. Consequently the
comparison changes public context as well as outcome count and signal aggregation.

Model, sampling, xhigh reasoning, context budget, retries and failure accounting
match the original campaign. Model: Inferact/Qwen3.8-Flash-Next-NVFP4 at
http://sn4622130540:8001/v1. See sampling.json. DGP review is preparation work and
excluded from experimental call/token totals.

Run: /home/klkehl/thisenv/bin/python -u /data1/ken/onc-co-scientist/outputs/nsclc_depmap_single_outcome_qwen38_20260923/manage.py run --workflow persistent
Resume adds --resume only after verifying the prior host PID is stopped.
The manager holds execution.lock; do not launch duplicate workers. Source, packages,
plans and configurations are frozen by hash. A completed grid requires RESULTS.md
and RESULTS.json, then a report to the user and a reporting receipt.
