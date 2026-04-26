"""End-to-end tests for scripts/run_harness.sh using fake harness binaries."""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from onc_co_scientist.harness.task_spec import build_tasks
from onc_co_scientist.synthetic.cancer_types import CancerType
from onc_co_scientist.synthetic.generator import GeneratorConfig
from onc_co_scientist.synthetic.multi import (
    generate_multi_dataset,
    write_multi_bundle_pair,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_harness.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash is required to run scripts/run_harness.sh"
)


def _build_two_cancer_tasks(tmp_path: Path) -> Path:
    """Generate a tiny two-cancer-type tasks tree under tmp_path/tasks."""
    base = GeneratorConfig(
        dataset_id="ds_runharness",
        patient_n=40,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
        n_extra_covariates=4,
    )
    chosen = [CancerType.crc, CancerType.breast]
    bundles = generate_multi_dataset(base, chosen)
    synth_root = tmp_path / "ds"
    write_multi_bundle_pair(bundles, synth_root, anon_seed=0)
    tasks_root = tmp_path / "tasks"
    build_tasks(synth_root, tasks_root, max_iterations=2)
    return tasks_root


def _write_fake_harness(bin_dir: Path, name: str = "fake-harness") -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / name
    # Always emits a minimal transcript and an analysis summary in cwd. If
    # given a `ls-parent` token it also dumps the parent listing into the
    # summary so we can exercise the leak-guard test.
    fake.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -e
            echo '{"dataset_id":"x","model_id":"fake","harness_id":"fake@1","max_iterations":1,"iterations":[]}' > transcript.json
            {
              echo "ran in $(pwd)"
              for arg in "$@"; do
                if [[ "$arg" == "ls-parent" ]]; then
                  echo "parent listing:"
                  ls ../
                fi
              done
            } > analysis_summary.txt
            """
        )
    )
    fake.chmod(0o755)
    return fake


def _run_script(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


def _path_with_bin(bin_dir: Path) -> str:
    return f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


def test_run_harness_writes_transcript_per_bundle(tmp_path: Path) -> None:
    tasks_root = _build_two_cancer_tasks(tmp_path)
    bin_dir = tmp_path / "bin"
    _write_fake_harness(bin_dir)

    result = _run_script(
        "fake-harness",
        str(tasks_root),
        env_extra={"PATH": _path_with_bin(bin_dir)},
    )
    assert result.returncode == 0, result.stderr

    for ct in ("crc", "breast"):
        for variant in ("named", "anonymized"):
            assert (tasks_root / ct / variant / "transcript.json").exists()
            assert (tasks_root / ct / variant / "harness.log").exists()


def test_run_harness_parallel_jobs_writes_per_bundle_log(tmp_path: Path) -> None:
    tasks_root = _build_two_cancer_tasks(tmp_path)
    bin_dir = tmp_path / "bin"
    _write_fake_harness(bin_dir)

    result = _run_script(
        "fake-harness",
        str(tasks_root),
        "--jobs",
        "4",
        env_extra={"PATH": _path_with_bin(bin_dir)},
    )
    assert result.returncode == 0, result.stderr

    for ct in ("crc", "breast"):
        for variant in ("named", "anonymized"):
            assert (tasks_root / ct / variant / "transcript.json").exists()
            assert (tasks_root / ct / variant / "harness.log").exists()


def test_run_harness_skip_existing_skips_bundles_with_transcript(tmp_path: Path) -> None:
    tasks_root = _build_two_cancer_tasks(tmp_path)
    bin_dir = tmp_path / "bin"
    _write_fake_harness(bin_dir)

    sentinel_bundle = tasks_root / "crc" / "named"
    pre_existing = sentinel_bundle / "transcript.json"
    pre_existing.write_text('{"do":"not overwrite"}')

    result = _run_script(
        "fake-harness",
        str(tasks_root),
        "--skip-existing",
        env_extra={"PATH": _path_with_bin(bin_dir)},
    )
    assert result.returncode == 0, result.stderr

    # Pre-existing transcript was untouched.
    assert pre_existing.read_text() == '{"do":"not overwrite"}'
    # And the other three bundles still ran.
    for ct, variant in (("crc", "anonymized"), ("breast", "named"), ("breast", "anonymized")):
        assert (tasks_root / ct / variant / "transcript.json").exists()


def test_run_harness_cwd_isolation(tmp_path: Path) -> None:
    """The harness must be cd'd into the bundle, so a parent listing
    shows the cancer-type folder (not the tasks root or anything beyond)."""
    tasks_root = _build_two_cancer_tasks(tmp_path)
    bin_dir = tmp_path / "bin"
    _write_fake_harness(bin_dir)

    # Drop a sentinel under tasks_root that *would* be visible if cwd leaked.
    (tasks_root / "SHOULD_NOT_BE_VISIBLE").write_text("leak")

    result = _run_script(
        "fake-harness",
        str(tasks_root),
        "--extra-args",
        "ls-parent",
        env_extra={"PATH": _path_with_bin(bin_dir)},
    )
    assert result.returncode == 0, result.stderr

    for ct in ("crc", "breast"):
        for variant in ("named", "anonymized"):
            summary = (tasks_root / ct / variant / "analysis_summary.txt").read_text()
            assert "SHOULD_NOT_BE_VISIBLE" not in summary, summary
            # Parent of the bundle is the cancer-type folder; it should
            # contain the sibling variant ("named" or "anonymized").
            sibling = "anonymized" if variant == "named" else "named"
            assert sibling in summary, summary


def test_run_harness_unknown_harness_errors(tmp_path: Path) -> None:
    tasks_root = _build_two_cancer_tasks(tmp_path)
    result = _run_script(
        "this-binary-does-not-exist-xyz",
        str(tasks_root),
    )
    assert result.returncode != 0
    assert "not on PATH" in result.stderr


def test_run_harness_ollama_wrapped_inserts_double_dash(tmp_path: Path) -> None:
    """`ollama launch claude ... --yes` must auto-insert `--` so the
    profile flags reach the inner CLI per ollama's pass-through rule."""
    tasks_root = _build_two_cancer_tasks(tmp_path)
    bin_dir = tmp_path / "bin"
    argv_log = tmp_path / "ollama_argv.log"
    fake_ollama = bin_dir / "ollama"
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_ollama.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -e
            # Record full argv (one line per call, NUL-separated tokens).
            printf '%s\\0' "$@" >> "{argv_log}"
            printf '\\n' >> "{argv_log}"

            # Validate the contract: must look like
            #   launch claude <ollama-args...> -- <inner-cli-args...> <prompt>
            saw_launch_claude=0
            saw_double_dash=0
            saw_p_flag=0
            for arg in "$@"; do
              case "$arg" in
                claude) saw_launch_claude=1 ;;
                --) saw_double_dash=1 ;;
                -p) [[ $saw_double_dash -eq 1 ]] && saw_p_flag=1 ;;
              esac
            done
            if [[ $saw_launch_claude -eq 1 && $saw_double_dash -eq 1 && $saw_p_flag -eq 1 ]]; then
              echo '{{"dataset_id":"x","model_id":"fake","harness_id":"ollama-claude@1","max_iterations":1,"iterations":[]}}' > transcript.json
              echo "ollama-launched fake harness ran in $(pwd)" > analysis_summary.txt
            else
              echo "contract violation: launch=$saw_launch_claude dd=$saw_double_dash p=$saw_p_flag" >&2
              exit 17
            fi
            """
        )
    )
    fake_ollama.chmod(0o755)

    result = _run_script(
        "ollama launch claude --model fake-model --yes",
        str(tasks_root),
        env_extra={"PATH": _path_with_bin(bin_dir)},
    )
    assert result.returncode == 0, result.stderr

    # Every bundle got its transcript via the wrapped form.
    for ct in ("crc", "breast"):
        for variant in ("named", "anonymized"):
            assert (tasks_root / ct / variant / "transcript.json").exists()

    # And the recorded argv shows `--` was inserted before `-p`.
    log = argv_log.read_text()
    assert "launch" in log and "claude" in log and "--" in log and "-p" in log
