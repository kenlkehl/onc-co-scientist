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
#   OUT SYNTH_ROOT HARNESS JOBS REPLICATES PYTHON_ENV JUDGE JUDGE_CLI JUDGE_MODEL

set -euo pipefail

trap 'status=$?; echo; echo "exited with status $status"; read -p "press enter to close..."' EXIT

OUT="${OUT:-../data/ds001}"
SYNTH_ROOT="${SYNTH_ROOT:-$OUT}"
HARNESS="${HARNESS:-claude}"
JOBS="${JOBS:-4}"
REPLICATES="${REPLICATES:-20}"
PYTHON_ENV="${PYTHON_ENV:-.venv}"
JUDGE="${JUDGE:-anthropic-vertex}"
JUDGE_CLI="${JUDGE_CLI:-auto}"
JUDGE_MODEL="${JUDGE_MODEL:-}"

TASKS_ROOT="$OUT/tasks"
SCORE_ROOT="$OUT/score"

if [[ ! -d "$TASKS_ROOT" ]]; then
    echo "error: tasks root does not exist: $TASKS_ROOT" >&2
    echo "       run scripts/run_all.sh first to generate datasets and bundles." >&2
    exit 2
fi
if [[ ! -d "$SYNTH_ROOT" ]]; then
    echo "error: synth root does not exist: $SYNTH_ROOT" >&2
    echo "       scoring needs the source bundles with manifest.json ground truth." >&2
    exit 2
fi

has_source_bundle() {
    local root="$1"
    local manifest_path bundle_dir
    while IFS= read -r -d '' manifest_path; do
        bundle_dir="${manifest_path%/manifest.json}"
        if [[ -f "$bundle_dir/public/dataset.parquet" ]]; then
            return 0
        fi
    done < <(find "$root" -type f -name 'manifest.json' -print0)
    return 1
}

if ! has_source_bundle "$SYNTH_ROOT"; then
    echo "error: no source dataset bundles found under $SYNTH_ROOT" >&2
    echo "       scoring cannot use task bundles alone because tasks omit manifest.json" >&2
    echo "       to avoid leaking ground truth to the agent." >&2
    echo "       expected e.g. $SYNTH_ROOT/<cancer_type>/<variant>/manifest.json" >&2
    echo "       and $SYNTH_ROOT/<cancer_type>/<variant>/public/dataset.parquet." >&2
    echo "       If source bundles live separately, set SYNTH_ROOT=/path/to/synth-root." >&2
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
score_args=(
    score
    batch
    --synth-root "$SYNTH_ROOT"
    --tasks-root "$TASKS_ROOT"
    --out "$SCORE_ROOT"
    --judge "$JUDGE"
    --judge-cli "$JUDGE_CLI"
)
if [[ -n "$JUDGE_MODEL" ]]; then
    score_args+=( --judge-model "$JUDGE_MODEL" )
fi
ocs "${score_args[@]}"

echo "Resume complete. Report: $SCORE_ROOT/batch_score.md" >&2
