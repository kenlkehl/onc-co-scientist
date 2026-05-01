#!/usr/bin/env bash
# End-to-end pipeline: synth → tasks → harness → score.
#
# Override defaults via environment variables, e.g.:
#   OUT=/tmp/ds001 HARNESS=claude JUDGE=codex-cli REPLICATES=3 JOBS=4 scripts/run_all.sh
#
# Each step is idempotent on top of the previous run's outputs:
#   - `ocs synth generate`   — overwrites datasets in OUT (leaves runs/ alone).
#   - `ocs harness build-task` — overwrites task bundles in OUT/tasks
#                                (leaves runs/ alone).
#   - cleanup of failed run dirs — drops any runs/run_NNN/ that lacks
#     transcript.json so the harness's top-up resumes contiguous numbering.
#   - `scripts/run_harness.sh` — tops up replicates to REPLICATES per bundle.
#   - `ocs score batch`      — re-scores from cache; cheap to re-run.
#
# Re-running this script after an interruption is safe: the cleanup step plus
# the harness's per-bundle top-up resume any task bundle that has some, but
# not all, completed replicates.

set -euo pipefail

trap 'status=$?; echo; echo "exited with status $status"; read -p "press enter to close..."' EXIT


# ---- defaults ----------------------------------------------------------

CONFIG="${CONFIG:-configs/synthetic.example.yaml}"
OUT="${OUT:-../data/ds001}"
SEED="${SEED:-0}"
CANCER_TYPES="${CANCER_TYPES:-all}"
MAX_ITERATIONS="${MAX_ITERATIONS:-25}"
HARNESS="${HARNESS:-claude}"
JOBS="${JOBS:-4}"
REPLICATES="${REPLICATES:-20}"
PYTHON_ENV="${PYTHON_ENV:-.venv}"
JUDGE="${JUDGE:-anthropic-vertex}"
JUDGE_CLI="${JUDGE_CLI:-auto}"
JUDGE_MODEL="${JUDGE_MODEL:-}"

TASKS_ROOT="$OUT/tasks"
SCORE_ROOT="$OUT/score"

# ---- step 1: synthetic dataset ----------------------------------------

echo "[1/5] Generating synthetic datasets in $OUT (cancer-types=$CANCER_TYPES, seed=$SEED)" >&2
ocs synth generate \
    --config "$CONFIG" \
    --out "$OUT" \
    --seed "$SEED" \
    --cancer-types "$CANCER_TYPES"

# ---- step 2: harness task bundles -------------------------------------

echo "[2/5] Building task bundles in $TASKS_ROOT (max-iterations=$MAX_ITERATIONS)" >&2
ocs harness build-task \
    --dataset "$OUT" \
    --max-iterations "$MAX_ITERATIONS" \
    --out "$TASKS_ROOT"

# ---- step 3: clean up failed run dirs ---------------------------------
# Drop any runs/run_NNN/ left behind without transcript.json (e.g. from a
# rate-limit or SIGINT during a prior invocation) so next_run_index() in the
# harness fills the freed slots instead of growing a sparse numbering.

echo "[3/5] Cleaning up failed run dirs under $TASKS_ROOT" >&2
removed=0
while IFS= read -r -d '' run_dir; do
    if [[ ! -f "$run_dir/transcript.json" ]]; then
        echo "  removing $run_dir (no transcript.json)" >&2
        rm -rf "$run_dir"
        (( removed++ )) || true
    fi
done < <(find "$TASKS_ROOT" -mindepth 4 -maxdepth 4 -type d -name 'run_*' -print0)
echo "  removed $removed failed run dir(s)" >&2

# ---- step 4: run the harness ------------------------------------------

echo "[4/5] Running $HARNESS across $TASKS_ROOT (jobs=$JOBS, replicates=$REPLICATES)" >&2
harness_args=(
    "$HARNESS"
    "$TASKS_ROOT"
    --jobs "$JOBS"
    --replicates "$REPLICATES"
)
if [[ -n "$PYTHON_ENV" && -d "$PYTHON_ENV" ]]; then
    harness_args+=( --python-env "$PYTHON_ENV" )
fi
scripts/run_harness.sh "${harness_args[@]}"

# ---- step 5: score ----------------------------------------------------

echo "[5/5] Scoring (judge=$JUDGE) → $SCORE_ROOT" >&2
score_args=(
    score
    batch
    --synth-root "$OUT"
    --tasks-root "$TASKS_ROOT"
    --out "$SCORE_ROOT"
    --judge "$JUDGE"
    --judge-cli "$JUDGE_CLI"
)
if [[ -n "$JUDGE_MODEL" ]]; then
    score_args+=( --judge-model "$JUDGE_MODEL" )
fi
ocs "${score_args[@]}"

echo "Done. Report: $SCORE_ROOT/batch_score.md" >&2
