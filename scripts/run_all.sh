#!/usr/bin/env bash
# End-to-end pipeline: synth → tasks → harness → score.
#
# Override defaults via environment variables, e.g.:
#   OUT=/tmp/ds001 HARNESS=claude REPLICATES=3 JOBS=4 scripts/run_all.sh
#
# Each step is idempotent on top of the previous run's outputs:
#   - `ocs synth generate`   — overwrites datasets in OUT.
#   - `ocs harness build-task` — overwrites task bundles in OUT/tasks.
#   - `scripts/run_harness.sh` — tops up replicates to REPLICATES per bundle.
#   - `ocs score batch`      — re-scores from cache; cheap to re-run.

set -euo pipefail

# ---- defaults ----------------------------------------------------------

CONFIG="${CONFIG:-configs/synthetic.example.yaml}"
OUT="${OUT:-../data/ds001}"
SEED="${SEED:-0}"
CANCER_TYPES="${CANCER_TYPES:-all}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
HARNESS="${HARNESS:-claude}"
JOBS="${JOBS:-4}"
REPLICATES="${REPLICATES:-5}"
PYTHON_ENV="${PYTHON_ENV:-.venv}"
JUDGE="${JUDGE:-claude-cli}"

TASKS_ROOT="$OUT/tasks"
SCORE_ROOT="$OUT/score"

# ---- step 1: synthetic dataset ----------------------------------------

echo "[1/4] Generating synthetic datasets in $OUT (cancer-types=$CANCER_TYPES, seed=$SEED)" >&2
ocs synth generate \
    --config "$CONFIG" \
    --out "$OUT" \
    --seed "$SEED" \
    --cancer-types "$CANCER_TYPES"

# ---- step 2: harness task bundles -------------------------------------

echo "[2/4] Building task bundles in $TASKS_ROOT (max-iterations=$MAX_ITERATIONS)" >&2
ocs harness build-task \
    --dataset "$OUT" \
    --max-iterations "$MAX_ITERATIONS" \
    --out "$TASKS_ROOT"

# ---- step 3: run the harness ------------------------------------------

echo "[3/4] Running $HARNESS across $TASKS_ROOT (jobs=$JOBS, replicates=$REPLICATES)" >&2
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

# ---- step 4: score ----------------------------------------------------

echo "[4/4] Scoring (judge=$JUDGE) → $SCORE_ROOT" >&2
ocs score batch \
    --synth-root "$OUT" \
    --tasks-root "$TASKS_ROOT" \
    --out "$SCORE_ROOT" \
    --judge "$JUDGE"

echo "Done. Report: $SCORE_ROOT/batch_score.md" >&2
