#!/usr/bin/env bash
# Run an external agentic harness against every task bundle produced by
# `ocs harness build-task` under a tasks/ root.
#
# The script always cds into the per-bundle directory before launching the
# harness, so the harness cannot reach the synth bundle's manifest.json
# (which sits one level up under <synth_root>/<ct>/<variant>/).
#
# Usage: scripts/run_harness.sh <harness-spec> <tasks-root> [flags]
# Run with -h for the full usage banner.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_harness.sh <harness-spec> <tasks-root> [flags]

Positional arguments:
  <harness-spec>    Either a bare harness name (claude, codex, opencode,
                    droid, pi) or an ollama-launch wrapper form
                    ("ollama launch claude --model qwen3.6:27b --yes").
                    The wrapper form auto-inserts the `--` separator that
                    ollama needs to pass trailing flags through to the
                    inner CLI.
  <tasks-root>      Directory containing one or more task bundles in the
                    <ct>/<variant>/ layout written by
                    `ocs harness build-task --out`.

Flags:
  --python-env DIR  Path to a Python venv. Its bin/ is prepended to PATH
                    for each harness invocation so any `python` shelled
                    out by the agent resolves there.
  --jobs N          Number of bundles to run in parallel (default 1).
  --skip-existing   Skip bundles that already have transcript.json.
  --extra-args STR  Extra args inserted between the profile flags and the
                    prompt (e.g. --extra-args "--max-turns 30").
  --prompt STR      Override the default agent prompt.
  --profile NAME    Pin the harness profile (claude, codex, opencode,
                    droid, pi) instead of auto-detecting from the spec.
  --dry-run         Print the resolved command for each bundle and exit
                    without launching anything.
  -h, --help        Show this banner.
EOF
}

DEFAULT_PROMPT='Read agent_instructions.md in the current working directory and follow its instructions exactly. Emit transcript.json and analysis_summary.txt in this directory when done. Do not access any files outside this directory.'

profile_args() {
    # Echo the per-profile argv (subcommand + permissions flags) on stdout,
    # one token per line, NUL-terminated. The prompt is appended by the
    # caller after the optional --extra-args.
    local profile="$1"
    case "$profile" in
        claude)
            printf '%s\0' '-p' '--dangerously-skip-permissions'
            ;;
        codex)
            printf '%s\0' 'exec' '--dangerously-bypass-approvals-and-sandbox'
            ;;
        opencode)
            printf '%s\0' 'run'
            ;;
        droid)
            printf '%s\0' 'exec' '--auto' 'high'
            ;;
        pi)
            : # No documented flags; user can pass --extra-args.
            ;;
        *)
            echo "warning: unknown harness profile '$profile'; passing prompt with no extra flags" >&2
            ;;
    esac
}

resolve_command() {
    # Print the full argv (one token per NUL-terminated line) for invoking
    # the harness on a single bundle. Reads the global SPEC, PROFILE,
    # EXTRA_ARGS, and PROMPT.
    local -a spec_tokens
    # shellcheck disable=SC2206  # word-splitting on SPEC is intentional.
    spec_tokens=( $SPEC )

    local is_ollama=0
    if [[ "${spec_tokens[0]:-}" == "ollama" && "${spec_tokens[1]:-}" == "launch" ]]; then
        is_ollama=1
    fi

    local tok
    for tok in "${spec_tokens[@]}"; do
        printf '%s\0' "$tok"
    done
    if (( is_ollama )); then
        printf '%s\0' '--'
    fi
    while IFS= read -r -d '' tok; do
        printf '%s\0' "$tok"
    done < <(profile_args "$PROFILE")
    if [[ -n "${EXTRA_ARGS:-}" ]]; then
        local -a extra_tokens
        # shellcheck disable=SC2206  # word-splitting on EXTRA_ARGS is intentional.
        extra_tokens=( $EXTRA_ARGS )
        for tok in "${extra_tokens[@]}"; do
            printf '%s\0' "$tok"
        done
    fi
    printf '%s\0' "$PROMPT"
}

run_one() {
    local bundle="$1"
    if [[ ! -f "$bundle/agent_instructions.md" ]]; then
        echo "skip (no agent_instructions.md): $bundle" >&2
        return 0
    fi
    if (( SKIP_EXISTING )) && [[ -f "$bundle/transcript.json" ]]; then
        echo "skip (transcript.json present): $bundle" >&2
        return 0
    fi

    local -a argv=()
    while IFS= read -r -d '' tok; do
        argv+=( "$tok" )
    done < <(resolve_command)

    if (( DRY_RUN )); then
        printf 'would run in %s:\n  ' "$bundle"
        printf '%q ' "${argv[@]}"
        printf '\n'
        return 0
    fi

    (
        cd "$bundle"
        if [[ -n "${PYTHON_ENV:-}" ]]; then
            export PATH="$PYTHON_ENV/bin:$PATH"
        fi
        exec "${argv[@]}" >harness.log 2>&1
    )
}

# ---- arg parsing ---------------------------------------------------------

SPEC=""
TASKS_ROOT=""
PYTHON_ENV=""
JOBS=1
SKIP_EXISTING=0
EXTRA_ARGS=""
PROMPT="$DEFAULT_PROMPT"
PROFILE=""
DRY_RUN=0
INTERNAL_RUN_ONE=""

positional=()
while (( $# > 0 )); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --python-env)
            PYTHON_ENV="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;
        --extra-args)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --_run-one)
            # Internal recursion target used by the parallel xargs path.
            INTERNAL_RUN_ONE="$2"
            shift 2
            ;;
        --)
            shift
            positional+=( "$@" )
            break
            ;;
        -*)
            echo "error: unknown flag: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            positional+=( "$1" )
            shift
            ;;
    esac
done

if [[ -n "$INTERNAL_RUN_ONE" ]]; then
    # Re-entered by xargs to handle one bundle. Positional args (spec,
    # tasks-root) and flags must round-trip on the recursion command line.
    if (( ${#positional[@]} < 1 )); then
        echo "error: --_run-one requires <harness-spec> on the recursion command line" >&2
        exit 2
    fi
    SPEC="${positional[0]}"
    if [[ -z "$PROFILE" ]]; then
        # Re-derive profile (matches main path).
        # shellcheck disable=SC2206
        spec_tokens=( $SPEC )
        if [[ "${spec_tokens[0]:-}" == "ollama" && "${spec_tokens[1]:-}" == "launch" ]]; then
            PROFILE="${spec_tokens[2]:-}"
        else
            PROFILE="${spec_tokens[0]:-}"
        fi
    fi
    run_one "$INTERNAL_RUN_ONE"
    exit $?
fi

if (( ${#positional[@]} < 2 )); then
    usage >&2
    exit 2
fi

SPEC="${positional[0]}"
TASKS_ROOT="${positional[1]}"

if [[ ! -d "$TASKS_ROOT" ]]; then
    echo "error: tasks root does not exist or is not a directory: $TASKS_ROOT" >&2
    exit 2
fi

# Derive the profile from the spec if the user didn't pin one.
# shellcheck disable=SC2206
spec_tokens=( $SPEC )
if [[ -z "$PROFILE" ]]; then
    if [[ "${spec_tokens[0]:-}" == "ollama" && "${spec_tokens[1]:-}" == "launch" ]]; then
        PROFILE="${spec_tokens[2]:-}"
    else
        PROFILE="${spec_tokens[0]:-}"
    fi
fi
if [[ -z "$PROFILE" ]]; then
    echo "error: could not derive a harness profile from spec: $SPEC" >&2
    exit 2
fi

# Verify the launching binary exists on PATH (skip in dry-run mode so the
# user can preview commands without installing every harness).
launcher_bin="${spec_tokens[0]:-}"
if (( ! DRY_RUN )); then
    if ! command -v "$launcher_bin" >/dev/null 2>&1; then
        echo "error: launcher binary not on PATH: $launcher_bin" >&2
        exit 2
    fi
fi

# Discover bundles: every <ct>/<variant>/agent_instructions.md is a bundle.
mapfile -d '' bundles < <(
    find "$TASKS_ROOT" -mindepth 3 -maxdepth 3 -type f -name 'agent_instructions.md' -print0 \
        | sort -z
)
bundle_dirs=()
for bf in "${bundles[@]}"; do
    bundle_dirs+=( "${bf%/agent_instructions.md}" )
done

if (( ${#bundle_dirs[@]} == 0 )); then
    echo "error: no task bundles (with agent_instructions.md) found under $TASKS_ROOT" >&2
    exit 2
fi

echo "Found ${#bundle_dirs[@]} task bundle(s) under $TASKS_ROOT" >&2
echo "Profile: $PROFILE   Jobs: $JOBS" >&2

n_ok=0
n_fail=0
n_total="${#bundle_dirs[@]}"

if (( JOBS <= 1 )); then
    for bundle in "${bundle_dirs[@]}"; do
        if run_one "$bundle"; then
            (( n_ok++ )) || true
        else
            (( n_fail++ )) || true
            echo "fail: $bundle" >&2
        fi
    done
else
    # Re-invoke ourselves once per bundle via xargs -P. We forward the spec
    # and all flags that affect a single-bundle run so the child has
    # everything it needs.
    self="${BASH_SOURCE[0]}"
    if [[ ! -x "$self" ]]; then
        # Fall back to invoking with bash explicitly if the file isn't
        # marked executable in this checkout.
        self_argv=( bash "$self" )
    else
        self_argv=( "$self" )
    fi

    extra_flags=()
    if [[ -n "$PYTHON_ENV" ]]; then extra_flags+=( --python-env "$PYTHON_ENV" ); fi
    if (( SKIP_EXISTING )); then extra_flags+=( --skip-existing ); fi
    if [[ -n "$EXTRA_ARGS" ]]; then extra_flags+=( --extra-args "$EXTRA_ARGS" ); fi
    extra_flags+=( --prompt "$PROMPT" )
    extra_flags+=( --profile "$PROFILE" )
    if (( DRY_RUN )); then extra_flags+=( --dry-run ); fi

    set +e
    printf '%s\0' "${bundle_dirs[@]}" \
        | xargs -0 -n1 -P "$JOBS" -I{} \
            "${self_argv[@]}" --_run-one "{}" "${extra_flags[@]}" "$SPEC" "$TASKS_ROOT"
    rc=$?
    set -e
    # xargs returns 123 if any child returned 1-125; 0 otherwise.
    if (( rc == 0 )); then
        n_ok=$n_total
        n_fail=0
    else
        # We don't know per-bundle counts here; recount via transcript presence.
        n_ok=0
        for bundle in "${bundle_dirs[@]}"; do
            if [[ -f "$bundle/transcript.json" ]]; then
                (( n_ok++ )) || true
            fi
        done
        n_fail=$(( n_total - n_ok ))
    fi
fi

echo "$n_ok/$n_total succeeded" >&2
if (( n_fail > 0 )); then
    exit 1
fi
