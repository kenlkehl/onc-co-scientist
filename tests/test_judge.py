"""Tests for scoring/judge.py — Judge protocol, ClaudeCliJudge, StubJudge, cache."""

from __future__ import annotations

import os
import shutil
import textwrap
from pathlib import Path

import pytest

from onc_co_scientist.scoring.judge import (
    ClaudeCliJudge,
    JudgeCache,
    NoveltyJudgment,
    StubJudge,
    _extract_json_array,
)


def test_extract_json_array_handles_plain_array():
    payload = '[{"is_novel": true, "rationale": "x"}]'
    assert _extract_json_array(payload) == [{"is_novel": True, "rationale": "x"}]


def test_extract_json_array_handles_code_fence():
    payload = textwrap.dedent(
        """\
        Sure! Here's the result:
        ```json
        [{"is_novel": false, "rationale": "standard"}]
        ```
        """
    )
    assert _extract_json_array(payload) == [
        {"is_novel": False, "rationale": "standard"}
    ]


def test_extract_json_array_handles_preamble():
    payload = "Reasoning... [{\"matches\": true, \"rationale\": \"yes\"}] done."
    assert _extract_json_array(payload) == [{"matches": True, "rationale": "yes"}]


def test_extract_json_array_raises_when_missing():
    with pytest.raises(ValueError, match="No JSON array"):
        _extract_json_array("no json here")


def test_stub_judge_novelty_marks_phrases():
    judge = StubJudge(novel_phrases=frozenset({"feature_037", "novel-thing"}))
    out = judge.judge_novelty(
        ["feature_037 increases pfs", "EGFR predicts response", "novel-thing here"]
    )
    assert [j.is_novel for j in out] == [True, False, True]


def test_stub_judge_matches_uses_association_key():
    judge = StubJudge(
        match_phrases={
            "BRCA2": frozenset({"olaparib", "parp"}),
            "EGFR": frozenset({"osimertinib"}),
        }
    )
    matches = judge.judge_matches(
        ["olaparib in BRCA2 patients", "Pembrolizumab works", "PARP inhibitor benefit"],
        association_nl="BRCA2-mutant subgroup responds to olaparib",
    )
    assert [m.matches for m in matches] == [True, False, True]


def test_judge_cache_round_trip(tmp_path: Path):
    cache = JudgeCache(cache_dir=tmp_path / "cache")
    assert cache.get("hello") is None
    cache.put("hello", '[{"x": 1}]')
    assert cache.get("hello") == '[{"x": 1}]'


def test_judge_cache_disabled_when_dir_none():
    cache = JudgeCache(cache_dir=None)
    cache.put("foo", "bar")
    assert cache.get("foo") is None


pytestmark_shell = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash required for fake claude binary"
)


def _write_fake_claude(bin_dir: Path, response: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "claude"
    # Echoes a fixed JSON-array response and increments an invocation
    # counter so tests can verify caching.
    fake.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            echo "$@" >> "{bin_dir}/claude_args.log"
            (
              flock -x 9
              n=0
              if [[ -f "{bin_dir}/counter" ]]; then n=$(<"{bin_dir}/counter"); fi
              echo $((n + 1)) > "{bin_dir}/counter"
            ) 9>"{bin_dir}/counter.lock"
            cat <<'__EOF__'
            {response}
            __EOF__
            """
        )
    )
    fake.chmod(0o755)
    return fake


@pytestmark_shell
def test_claude_cli_judge_invokes_subprocess(tmp_path: Path, monkeypatch):
    bin_dir = tmp_path / "bin"
    response = '[{"is_novel": true, "rationale": "uses obscure subgroup"}]'
    _write_fake_claude(bin_dir, response)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    judge = ClaudeCliJudge(cli="claude", batch_size=10, cache=None)
    out = judge.judge_novelty(["odd hypothesis"])
    assert out == [NoveltyJudgment(is_novel=True, rationale="uses obscure subgroup")]
    assert (bin_dir / "counter").read_text().strip() == "1"


@pytestmark_shell
def test_claude_cli_judge_uses_cache_to_skip_subprocess(tmp_path: Path, monkeypatch):
    bin_dir = tmp_path / "bin"
    response = '[{"is_novel": false, "rationale": "textbook"}]'
    _write_fake_claude(bin_dir, response)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    cache = JudgeCache(cache_dir=tmp_path / "cache")
    judge = ClaudeCliJudge(cli="claude", batch_size=10, cache=cache)
    judge.judge_novelty(["x"])
    judge.judge_novelty(["x"])  # second call should hit cache
    assert (bin_dir / "counter").read_text().strip() == "1"


@pytestmark_shell
def test_claude_cli_judge_batches_when_n_exceeds_size(tmp_path: Path, monkeypatch):
    bin_dir = tmp_path / "bin"
    # Always reply with a 2-element array (the runtime batch size below).
    response = '[{"is_novel": true, "rationale": "a"}, {"is_novel": false, "rationale": "b"}]'
    _write_fake_claude(bin_dir, response)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    judge = ClaudeCliJudge(cli="claude", batch_size=2, cache=None)
    out = judge.judge_novelty(["h1", "h2", "h3", "h4"])
    assert len(out) == 4
    # 4 hypotheses / batch_size 2 = 2 subprocess calls.
    assert (bin_dir / "counter").read_text().strip() == "2"
