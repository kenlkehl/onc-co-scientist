# Oncology Dataset Analysis — Task Brief

**Dataset:** `ds001_prostate`
**Patients:** 50000
**Maximum iterations (N):** 25

## Your role

You have been asked to analyze a large oncology dataset assembled from electronic health records aggregated by a commercial healthcare data vendor. Your job is to explore the dataset and surface clinically meaningful patterns — main effects, subgroup heterogeneity, and multivariable interactions among the features and outcomes — testing each pattern statistically and refining your ideas as evidence accumulates across iterations.

## Dataset

The dataset is available at:

- `dataset.parquet` (Parquet)

A human-readable column description is provided in `dataset_description.md`.


You must not look outside the current working directory for additional data or context. This is critical for patient privacy.

## Protocol (up to 25 iterations)

In each iteration:

1. **Propose hypotheses.** State one or more natural-language hypotheses about patterns in the data — relationships among features, subgroups, and outcomes.
2. **Test them.** Execute statistical analyses on the dataset (e.g., regression with interactions, stratified response-rate comparisons, subgroup analyses). Record the signed effect estimate, the p-value, and whether the result was significant.
3. **Update.** Use the results to propose new hypotheses or refine existing ones in the next iteration.

Stop when you have thoroughly probed the dataset or when you reach the iteration cap.

## Required output

Emit two files in this bundle directory:

1. `transcript.json` — conforming to the schema in this bundle (`transcript_schema.json`). A minimal example is provided in `transcript_example.json`.
2. `analysis_summary.txt` — a plain-text narrative that synthesizes and summarizes the results of all analyses you ran. Walk through the hypotheses you explored across iterations, what the statistical evidence showed (direction, magnitude, significance), which hypotheses were supported vs. refuted, and any overall conclusions about treatment–biomarker–outcome relationships in this cohort. 

Critical fields to include in `transcript.json`:

- `iterations[].proposed_hypotheses[].text` — the natural-language hypothesis. Make these **self-contained**: name the variables, the direction of effect, and any subgroup.
- `iterations[].analyses[].hypothesis_ids` — every analysis must list the IDs of the hypotheses it addresses.
- `iterations[].analyses[].effect_estimate` — signed on the outcome's natural scale; used to verify direction.
- `iterations[].analyses[].p_value` or `significant` — used to verify statistical support.

Top-level fields `dataset_id`, `model_id`, `harness_id`, and `max_iterations` must be set. Set `max_iterations` to the cap you were given (25).

## What makes a good transcript

- Hypotheses that name specific columns and a signed direction (e.g., "In patients with `feature_example` set, mean `outcome_example` is lower than in patients without it"). Use the actual column names from the dataset, not placeholder labels.
- Analyses that directly test each proposed hypothesis.
- Appropriate use of interaction and subgroup analyses where the data structure warrants them, rather than stopping at main effects.
