# Oncology Dataset Analysis - Task Brief

**Dataset:** `ds001_prostate`
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

Ignore any `runs/` directory or `harness.log` file. Those are harness artifacts,
not task inputs. Write `transcript.json` and `analysis_summary.txt` at this
bundle root, not under `runs/`.

Preferred execution pattern: after reading this brief, write and run a single
end-to-end Python analysis script that performs the full protocol, records each
iteration in memory, and writes the two required artifacts. Keep the script
loop-based and reusable: define helper functions, loop over selected feature
lists, and avoid hard-coding one source block per hypothesis or feature pair.
Each iteration should normally propose and test 1-3 focused hypotheses; use
ranked screening to choose follow-up hypotheses instead of enumerating every
possible combination. Do not repeatedly rerun the same comparison unless you
are correcting an error.

When carrying findings forward between iterations, keep column names, hypothesis
IDs, effect estimates, p-values, and significance flags in separate result
fields. Later analyses must index the dataframe by real dataset column names,
not by hypothesis IDs such as `h1_3`. Generate hypothesis IDs directly from the
current iteration and local loop index, or from a simple incrementing counter;
do not pop IDs from a fixed precomputed list that can be exhausted.

Use scalar-safe statistical helpers in your script. For example,
`scipy.stats.chi2_contingency` returns four values, so unpack it as
`_, p_value, _, _ = stats.chi2_contingency(table, correction=False)` and store
`float(p_value)` in the transcript. `scipy.stats.fisher_exact` returns two
values, so unpack it as `_, p_value = stats.fisher_exact(table)` and store
`float(p_value)`. `scipy.stats.ttest_ind` and `scipy.stats.pearsonr` return
result objects in recent SciPy versions; read `result.pvalue` and
`result.statistic` instead of unpacking them as four values. Store effect
estimates and p-values as plain Python floats, not NumPy arrays, tuples, or
statistical result objects.
Likewise, store significance flags as plain Python `bool` values, not NumPy
boolean scalars.
Keep helper return shapes stable. Prefer returning dictionaries from analysis
helpers over returning tuples that must be unpacked at many call sites. Do not
put conditional expressions inside f-string format specifiers; compute display
strings before formatting. Do not nest f-strings or escaped quotes inside an
outer f-string; precompute any joined or sorted display text in a separate
variable.
Do not pass placeholder values such as `None` into helpers that index dataframe
columns. Use separate helpers for feature-outcome analyses versus
feature-feature interaction analyses, or branch before indexing optional
columns.
Before formatting any numeric value with `:.3f` or similar, check for `None`,
NaN, or infinite values and use a precomputed fallback string such as `"NA"`.
Do not apply numeric format specs such as `:.3f` to strings returned by helper
functions like `safe_str`; either format a numeric value directly after
validating it, or insert the already formatted display string without another
format spec. Numeric formatting helpers must return fallback strings such as
`"NA"` before calling `float(value)`; do not call `float("NA")`.
When building summary text, remember that `list.append()` accepts one item; use
`summary_lines.extend([...])` for multiple lines.
For binary or categorical feature-outcome comparisons, prefer boolean-mask group
rates over direct `pd.crosstab(...).loc[value, ...]` indexing. A robust pattern
is: `mask = df[feature] == value`, compare `df.loc[mask, outcome].mean()` with
`df.loc[~mask, outcome].mean()`, and build a 2x2 count table from those two
groups. When counting compound masks, sum the boolean Series before converting
to `int`, e.g. `int((mask & (df[outcome] == 1)).sum())`; do not write
`int(mask & condition).sum()` or `int(mask & condition.sum())`. Do not subtract
two unequal-length outcome arrays; compare their means or fit a regression
model instead. If you write a JSON helper, include NumPy integers, floats,
booleans, and arrays and convert them to plain JSON types.
Before calling `json.dump`, run the whole transcript through that helper, e.g.
`json.dump(to_jsonable(transcript), f, indent=2)`, so no `np.bool_`, `np.int64`,
`np.float64`, `np.ndarray`, or NaN value can break final artifact generation.
Before finishing, rerun the final script and confirm it exits without a
traceback and writes parseable JSON.

## Required output

Emit two files in this bundle directory:

1. `transcript.json` - conforming to the schema in this bundle (`transcript_schema.json`). A minimal example is provided in `transcript_example.json`.
2. `analysis_summary.txt` - a plain-text narrative that synthesizes and summarizes the results of all analyses you ran. Walk through the hypotheses you explored across iterations, what the statistical evidence showed (direction, magnitude, significance), which hypotheses were supported vs. refuted, and any overall conclusions about feature-outcome relationships in this dataset.

For reliable artifact generation, keep shell/tool calls compact. Save intermediate
result files as you analyze, then generate `transcript.json` and
`analysis_summary.txt` from those saved results with a short script. Keep the
final script compact, preferably under 400 lines, and generate repetitive
summary text from loops over the transcript rather than hard-coding a long
report. Avoid emitting one very large inline transcript or source block in a
single tool call.

Critical fields to include in `transcript.json`:

- `iterations[].proposed_hypotheses[].text` - the natural-language hypothesis. Make these **self-contained**: name the variables, the direction of effect, and any subgroup.
- `iterations[].analyses[].hypothesis_ids` - every analysis must list the IDs of the hypotheses it addresses.
- `iterations[].analyses[].effect_estimate` - signed on the outcome's natural scale; used to verify direction.
- `iterations[].analyses[].p_value` or `significant` - used to verify statistical support.

The top-level JSON value in `transcript.json` must be an object, not a list.
Put your iteration records under that object's `iterations` field. Top-level
fields `dataset_id`, `model_id`, `harness_id`, and `max_iterations` must be
set. Set `max_iterations` to the cap you were given (10).

## What makes a good transcript

- Hypotheses that name specific columns and a signed direction (e.g., "In patients with `feature_example` set, mean `outcome_example` is lower than in patients without it"). Use the actual column names from the dataset, not placeholder labels.
- Analyses that directly test each proposed hypothesis.
- Appropriate use of interaction and subgroup analyses where the data structure warrants them, rather than stopping at main effects.
- A final best-supported treatment-effect subgroup hypothesis for each outcome where one is plausible, naming the treatment, outcome, direction, and all subgroup predicates you believe define the effect.
