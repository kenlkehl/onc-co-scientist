from onc_open_mindedness.scoring.matching import RegexMatcher
from onc_open_mindedness.synthetic.paradigms import (
    concordant_catalog,
    discordant_catalog,
    hidden_novel_catalog,
)


def test_regex_matcher_accepts_direct_phrasing():
    matcher = RegexMatcher()
    spec = concordant_catalog()[0]  # IO + EGFR + PFS, direction = -1
    assert matcher.matches(
        "Immune checkpoint inhibitor therapy produces shorter progression-free "
        "survival in patients with EGFR mutations.",
        spec,
    )


def test_regex_matcher_rejects_unrelated_hypothesis():
    matcher = RegexMatcher()
    spec = concordant_catalog()[0]
    assert not matcher.matches(
        "Patients with higher age experience more adverse events on chemotherapy.",
        spec,
    )


def test_regex_matcher_direction_check_when_enabled():
    strict = RegexMatcher(require_direction=True)
    spec = discordant_catalog()[1]  # IO + EGFR, direction = +1 (inverted)
    # Direction-correct phrasing (more/longer) should match.
    assert strict.matches(
        "Immunotherapy is associated with longer progression-free survival "
        "in EGFR-mutant lung cancer.",
        spec,
    )
    # Direction-inverted phrasing (shorter) should NOT match under direction-strict mode.
    assert not strict.matches(
        "Immunotherapy is associated with shorter progression-free survival "
        "in EGFR-mutant lung cancer.",
        spec,
    )


def test_regex_matcher_handles_hidden_novel_subgroup():
    matcher = RegexMatcher()
    spec = hidden_novel_catalog()[0]
    assert matcher.matches(
        "Drug X yields exceptional objective response rates specifically in the "
        "biomarker_z_high subgroup.",
        spec,
    )
