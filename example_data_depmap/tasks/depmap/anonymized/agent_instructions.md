# CRISPR Dependency Map Analysis - Task Brief

**Dataset:** `ds001_depmap`
**Cell lines:** 50000
**Maximum iterations (N):** 10

## Your role

You have been asked to analyze a cancer cell-line CRISPR knockout screen dataset in the style of DepMap/CCLE. Your job is to explore the dataset and surface biologically meaningful patterns - main effects, subgroup heterogeneity, and multivariable interactions among cell-line features and gene dependency outcomes - testing each pattern statistically and refining your ideas as evidence accumulates across iterations.

## Dataset

The dataset is available at:

- `dataset.parquet` (Parquet)

A human-readable column description is provided in `dataset_description.md`.


You must not look outside the current working directory for additional data or context. Use only the local task bundle so every harness is evaluated against the same evidence.

## Protocol (up to 10 iterations)

In each iteration:

1. **Propose hypotheses.** State one or more natural-language hypotheses about patterns in the data - relationships among cell-line features, molecular subgroups, screen covariates, and dependency-score outcomes.
2. **Test them.** Execute statistical analyses on the dataset (e.g., regression with interactions, stratified comparisons, subgroup analyses). Record the signed effect estimate, the p-value, and whether the result was significant.
3. **Update.** Use the results to propose new hypotheses or refine existing ones in the next iteration.

Stop when you have thoroughly probed the dataset or when you reach the iteration cap.

Do not stop after main effects. Before your final transcript, run at least one systematic dependency-heterogeneity search for each dependency outcome. Appropriate approaches include feature-by-feature interaction screening, joint models over the strongest modifiers, tree/rule-based subgroup discovery, or exhaustive checks of small multi-feature subgroups. When a dependency appears concentrated in a subgroup, state and test the complete subgroup definition, including variables whose value appears to suppress or enable the dependency.

## Required output

Emit two files in this bundle directory:

1. `transcript.json` - conforming to the schema in this bundle (`transcript_schema.json`). A minimal example is provided in `transcript_example.json`.
2. `analysis_summary.txt` - a plain-text narrative that synthesizes and summarizes the results of all analyses you ran. Walk through the hypotheses you explored across iterations, what the statistical evidence showed (direction, magnitude, significance), which hypotheses were supported vs. refuted, and any overall conclusions about feature-outcome relationships in this dataset.

Critical fields to include in `transcript.json`:

- `iterations[].proposed_hypotheses[].text` - the natural-language hypothesis. Make these **self-contained**: name the variables, the direction of effect, and any subgroup.
- `iterations[].analyses[].hypothesis_ids` - every analysis must list the IDs of the hypotheses it addresses.
- `iterations[].analyses[].effect_estimate` - signed on the outcome's natural scale; used to verify direction.
- `iterations[].analyses[].p_value` or `significant` - used to verify statistical support.

Top-level fields `dataset_id`, `model_id`, `harness_id`, and `max_iterations` must be set. Set `max_iterations` to the cap you were given (10).

## What makes a good transcript

- Hypotheses that name specific columns and a signed direction (e.g., "In cell lines with `feature_example` set, `dependency_gene_example` is more negative than in cell lines without it"). Use the actual column names from the dataset, not placeholder labels.
- Analyses that directly test each proposed hypothesis.
- Appropriate use of interaction and subgroup analyses where the data structure warrants them, rather than stopping at main effects.
- A final best-supported dependency subgroup hypothesis for each dependency outcome where one is plausible, naming the dependency outcome, direction, and all subgroup predicates you believe define the effect.
