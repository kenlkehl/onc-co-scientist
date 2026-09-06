# Oncology Dataset Analysis — Task Brief

**Dataset:** `ds001_nsclc`
**Patients:** 40000
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

Do not stop after main effects. Before your final transcript, run at least one systematic treatment-effect heterogeneity search for each outcome with usable treatment variation. Appropriate approaches include treatment-by-feature interaction screening, joint models over the strongest modifiers, tree/rule-based subgroup discovery, or exhaustive checks of small multi-feature subgroups. When a treatment effect appears concentrated in a subgroup, state and test the complete subgroup definition, including variables whose unfavorable value appears to suppress the treatment effect.

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
- A final best-supported treatment-effect subgroup hypothesis for each outcome where one is plausible, naming the treatment, outcome, direction, and all subgroup predicates you believe define the effect.

## Runtime and structured recording

Use `/data1/ken/onc-co-scientist/data/sol_loose_env/bin/python` for Python. Read metadata.json for the exact dataset_id, model_id, harness_id, and max_iterations. pandas, scipy, statsmodels, scikit-learn and pyarrow are available.

Every proposed hypothesis must also include a non-null structured finding using the supplied transcript_schema.json and fictional transcript_example.json. Use actual column names, signed direction, the tested contrast and all subgroup predicates. Link each analysis to its hypothesis IDs.

Save each actual iteration as an IterationRecord JSON object with index, proposed_hypotheses and analyses. Submit it before starting the next iteration; do not reconstruct, renumber or pad records after the investigation. Keep the analysis code and results you execute in this workspace. You choose the analysis methods, sequence and stopping point within the 25-iteration cap.

Submit: `/data1/ken/onc-co-scientist/data/sol_loose_env/bin/python -m onc_co_scientist.harness.structured_runner submit --workspace . --record iteration_record.json`

After writing analysis_summary.txt, create transcript.json with `/data1/ken/onc-co-scientist/data/sol_loose_env/bin/python -m onc_co_scientist.harness.structured_runner finalize --workspace .`. Only record-validation feedback is available; there is no scientific scoring feedback during research.

Inspect only this workspace's inputs and your own outputs. Do not inspect other jobs, prior research, repository source, answer keys or external sources, and do not delegate.
