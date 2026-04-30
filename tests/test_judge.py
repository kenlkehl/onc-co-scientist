"""Tests for scoring/judge.py — Judge protocol, ClaudeCliJudge, StubJudge, cache."""

from __future__ import annotations

import os
import shutil
import textwrap
from pathlib import Path

import pytest

from onc_co_scientist.scoring.judge import (
    ClaudeCliJudge,
    CodexCliJudge,
    JudgeCache,
    NoveltyJudgment,
    StubJudge,
    _extract_json_array,
)
from onc_co_scientist.synthetic.schemas import (
    AssociationForm,
    AssociationSpec,
    ParadigmClass,
)


def _spec(
    *,
    nl: str = "BRCA2-mutant subgroup responds to olaparib",
    variables: list[str] | None = None,
    outcome: str = "objective_response",
) -> AssociationSpec:
    return AssociationSpec(
        id="test_spec",
        paradigm_class=ParadigmClass.hidden_novel,
        form=AssociationForm.subgroup_conditional,
        variables=variables if variables is not None else ["treatment_olaparib", "brca2_mutation"],
        outcome=outcome,
        direction=1,
        effect_size=1.0,
        natural_language_description=nl,
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


def test_stub_judge_matches_uses_spec_block_substring():
    """The stub keys on substrings of the rendered spec block — for the named
    variant that block includes the NL description, so 'BRCA2' still works."""
    judge = StubJudge(
        match_phrases={
            "BRCA2": frozenset({"olaparib", "parp"}),
            "EGFR": frozenset({"osimertinib"}),
        }
    )
    matches = judge.judge_matches(
        ["olaparib in BRCA2 patients", "Pembrolizumab works", "PARP inhibitor benefit"],
        _spec(nl="BRCA2-mutant subgroup responds to olaparib"),
        variant="named",
    )
    assert [m.matches for m in matches] == [True, False, True]


def test_stub_judge_matches_anonymized_keys_on_feature_tokens():
    """For anonymized variants the NL description is omitted from the spec
    block, so the only stable thing to key on is the feature_NNN tokens
    in spec.variables."""
    spec = _spec(
        variables=["feature_073", "feature_201"],
    )
    judge = StubJudge(
        match_phrases={
            "feature_073": frozenset({"feature_073"}),
        }
    )
    matches = judge.judge_matches(
        ["interaction between feature_073 and feature_201", "unrelated text"],
        spec,
        variant="anonymized",
    )
    assert [m.matches for m in matches] == [True, False]


def test_stub_judge_matches_anonymized_includes_nl_description():
    """The NL description is included for BOTH variants so the judge has
    symmetric information regardless of which name space the agent saw."""
    spec = _spec(
        nl="BRCA2-mutant subgroup responds to olaparib",
        variables=["feature_073", "feature_201"],
    )
    judge = StubJudge(
        match_phrases={
            "BRCA2": frozenset({"feature_073"}),
        }
    )
    matches = judge.judge_matches(
        ["mentions feature_073"],
        spec,
        variant="anonymized",
    )
    assert [m.matches for m in matches] == [True]


def test_render_spec_block_bilingual_with_mapping():
    """When a column_mapping is supplied, both clinical and feature_NNN
    names appear in the rendered spec block (for both variants)."""
    from onc_co_scientist.scoring.judge import _render_spec_block

    spec = _spec(
        variables=["treatment_olaparib", "brca2_mutation"],
    )
    mapping = {"treatment_olaparib": "feature_073", "brca2_mutation": "feature_201"}

    named_block = _render_spec_block(spec, "named", mapping)
    anon_block = _render_spec_block(spec, "anonymized", mapping)
    for block in (named_block, anon_block):
        assert "treatment_olaparib" in block
        assert "feature_073" in block
        assert "brca2_mutation" in block
        assert "feature_201" in block
        # NL description is present in both.
        assert "BRCA2-mutant subgroup responds to olaparib" in block


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


def _write_fake_codex(bin_dir: Path, response: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "codex"
    fake.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            printf '%s\\n' "$@" >> "{bin_dir}/codex_args.log"
            cat > "{bin_dir}/codex_stdin.log"
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


@pytestmark_shell
def test_codex_cli_judge_invokes_exec_with_stdin_and_schema(
    tmp_path: Path, monkeypatch
):
    bin_dir = tmp_path / "bin"
    response = '{"judgments": [{"is_novel": true, "rationale": "uses obscure subgroup"}]}'
    _write_fake_codex(bin_dir, response)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    judge = CodexCliJudge(
        cli="codex",
        model_id="gpt-5.4",
        batch_size=10,
        cache=None,
    )

    out = judge.judge_novelty(["odd hypothesis"])

    assert out == [NoveltyJudgment(is_novel=True, rationale="uses obscure subgroup")]
    args = (bin_dir / "codex_args.log").read_text().splitlines()
    assert args[:2] == ["exec", "--sandbox"]
    assert "read-only" in args
    assert "--skip-git-repo-check" in args
    assert "--ephemeral" in args
    assert "--ignore-user-config" in args
    assert "--ignore-rules" in args
    assert "--color" in args
    assert "--model" in args
    assert "gpt-5.4" in args
    assert "--output-schema" in args
    assert args[-1] == "-"
    stdin_text = (bin_dir / "codex_stdin.log").read_text()
    assert "odd hypothesis" in stdin_text
    assert '"judgments"' in stdin_text
