#!/usr/bin/env bash
# Resume a previously-interrupted run of scripts/run_all.sh.
#
# Skips steps 1 (synth) and 2 (build-task) — those outputs already exist on
# disk. Cleans up failed run dirs (those without transcript.json, typically
# left behind by a rate-limit interruption) so replicate numbering stays
# contiguous, then re-runs step 4 (harness top-up) and step 5 (score) from
# run_all.sh. (run_all.sh itself is also resumable — it runs the same cleanup
# step before the harness — so this script is a faster path that skips the
# always-overwriting synth + build-task steps.)
#
# Honors the same env-var contract as run_all.sh:
#   OUT HARNESS JOBS REPLICATES PYTHON_ENV JUDGE

set -euo pipefail

trap 'status=$?; echo; echo "exited with status $status"; read -p "press enter to close..."' EXIT

OUT="${OUT:-./example_data_20_iterations/ds001}"
HARNESS="${HARNESS:-claude}"
JOBS="${JOBS:-4}"
REPLICATES="${REPLICATES:-20}"
PYTHON_ENV="${PYTHON_ENV:-.venv}"
JUDGE="${JUDGE:-anthropic-vertex}"

TASKS_ROOT="$OUT/tasks"
SCORE_ROOT="$OUT/score"

if [[ ! -d "$TASKS_ROOT" ]]; then
    echo "error: tasks root does not exist: $TASKS_ROOT" >&2
    echo "       run scripts/run_all.sh first to generate datasets and bundles." >&2
    exit 2
fi

# ---- step A: clean up failed run dirs ---------------------------------

echo "[1/3] Cleaning up failed run dirs under $TASKS_ROOT" >&2
removed=0
while IFS= read -r -d '' run_dir; do
    if [[ ! -f "$run_dir/transcript.json" ]]; then
        echo "  removing $run_dir (no transcript.json)" >&2
        rm -rf "$run_dir"
        (( removed++ )) || true
    fi
done < <(find "$TASKS_ROOT" -mindepth 4 -maxdepth 4 -type d -name 'run_*' -print0)
echo "  removed $removed failed run dir(s)" >&2

# ---- step B: resume harness -------------------------------------------

echo "[2/3] Resuming $HARNESS across $TASKS_ROOT (jobs=$JOBS, replicates=$REPLICATES)" >&2
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

# ---- step C: score ----------------------------------------------------

echo "[3/3] Scoring (judge=$JUDGE) → $SCORE_ROOT" >&2
ocs score batch \
    --synth-root "$OUT" \
    --tasks-root "$TASKS_ROOT" \
    --out "$SCORE_ROOT" \
    --judge "$JUDGE"

echo "Resume complete. Report: $SCORE_ROOT/batch_score.md" >&2
