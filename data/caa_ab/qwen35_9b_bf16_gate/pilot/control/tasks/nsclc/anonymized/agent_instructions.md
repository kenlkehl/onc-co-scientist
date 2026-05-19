# Oncology Dataset Analysis - Task Brief

**Dataset:** `ds001_nsclc`
**Patients:** 50000
**Maximum iterations (N):** 10

## Your role

You have been asked to analyze a large oncology dataset assembled from electronic health records aggregated by a commercial healthcare data vendor. Your job is to explore the dataset and surface clinically meaningful patterns - main effects, subgroup heterogeneity, and multivariable interactions among the features and outcomes - testing each pattern statistically and refining your ideas as evidence accumulates across iterations.

## Dataset

The dataset is available at:

- `dataset.parquet` (Parquet)

A human-readable column description is provided in `dataset_description.md`.

## Python environment

Run all Python code inside the uv-managed environment at `/home/kenneth_kehl/onc-co-scientist/.venv`. Use its interpreter directly (e.g. `/home/kenneth_kehl/onc-co-scientist/.venv/bin/python your_script.py`) or activate it first (`source /home/kenneth_kehl/onc-co-scientist/.venv/bin/activate`), or invoke it via `uv run --project /home/kenneth_kehl/onc-co-scientist/.venv python ...`. Do not install packages into a different interpreter or create a new virtualenv - this environment already has the dependencies you need.


You must not look outside the current working directory for additional data or context. This is critical for patient privacy.

## Protocol (up to 10 iterations)

In each iteration:

1. **Propose hypotheses.** State one or more natural-language hypotheses about patterns in the data - relationships among features, subgroups, and outcomes.
2. **Test them.** Execute statistical analyses on the dataset (e.g., regression with interactions, stratified comparisons, subgroup analyses). Record the signed effect estimate, the p-value, and whether the result was significant.
3. **Update.** Use the results to propose new hypotheses or refine existing ones in the next iteration.

Stop when you have thoroughly probed the dataset or when you reach the iteration cap.

Do not stop after main effects. Before your final transcript, run at least one systematic treatment-effect heterogeneity search for each outcome with usable treatment variation. Appropriate approaches include treatment-by-feature interaction screening, joint models over the strongest modifiers, tree/rule-based subgroup discovery, or exhaustive checks of small multi-feature subgroups. When a treatment effect appears concentrated in a subgroup, state and test the complete subgroup definition, including variables whose unfavorable value appears to suppress the treatment effect.

Preferred execution pattern: after reading this brief, write and run a single
end-to-end Python analysis script that performs the full protocol, records each
iteration in memory, and writes the two required artifacts. Do not repeatedly
rerun the same comparison unless you are correcting an error.

When carrying findings forward between iterations, keep column names, hypothesis
IDs, effect estimates, p-values, and significance flags in separate result
fields. Later analyses must index the dataframe by real dataset column names,
not by hypothesis IDs such as `h1_3`.

## Required output

Emit two files in this bundle directory:

1. `transcript.json` - conforming to the schema in this bundle (`transcript_schema.json`). A minimal example is provided in `transcript_example.json`.
2. `analysis_summary.txt` - a plain-text narrative that synthesizes and summarizes the results of all analyses you ran. Walk through the hypotheses you explored across iterations, what the statistical evidence showed (direction, magnitude, significance), which hypotheses were supported vs. refuted, and any overall conclusions about feature-outcome relationships in this dataset.

For reliable artifact generation, keep shell/tool calls compact. Save intermediate result files as you analyze, then generate `transcript.json` and `analysis_summary.txt` from those saved results with a short script. Avoid emitting one very large inline transcript or source block in a single tool call.

Critical fields to include in `transcript.json`:

- `iterations[].proposed_hypotheses[].text` - the natural-language hypothesis. Make these **self-contained**: name the variables, the direction of effect, and any subgroup.
- `iterations[].analyses[].hypothesis_ids` - every analysis must list the IDs of the hypotheses it addresses.
- `iterations[].analyses[].effect_estimate` - signed on the outcome's natural scale; used to verify direction.
- `iterations[].analyses[].p_value` or `significant` - used to verify statistical support.

Top-level fields `dataset_id`, `model_id`, `harness_id`, and `max_iterations` must be set. Set `max_iterations` to the cap you were given (10).

## What makes a good transcript

- Hypotheses that name specific columns and a signed direction (e.g., "In patients with `feature_example` set, mean `outcome_example` is lower than in patients without it"). Use the actual column names from the dataset, not placeholder labels.
- Analyses that directly test each proposed hypothesis.
- Appropriate use of interaction and subgroup analyses where the data structure warrants them, rather than stopping at main effects.
- A final best-supported treatment-effect subgroup hypothesis for each outcome where one is plausible, naming the treatment, outcome, direction, and all subgroup predicates you believe define the effect.
