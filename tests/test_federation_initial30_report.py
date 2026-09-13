from scripts.expected_surprising import build_presentation_results as scoring
from scripts.expected_surprising import report_federation_initial30 as pilot


def row(sites, condition, value=100, status="completed"):
    version, view = condition.split("-")
    return dict(
        llm="gpt-5.6-sol",
        harness="Codex",
        workflow="sequential",
        reasoning_effort="medium",
        site_count=sites,
        partition_id="random",
        version=version,
        view=view,
        pair_id="fixed-pair",
        run_id=f"{sites}-{condition}",
        status=status,
        attempted=True,
        report_available=True,
        input_tokens=200,
        output_tokens=100,
        missing_token_calls=0,
        unknown_internal_attempts=0,
        **{m: value for m in scoring.METRICS},
        **{"response_" + c: value / 100 for c in scoring.CLASSES},
    )


def test_site_groups_cannot_fill_each_others_missing_conditions():
    rows = [row(2, c) for c in scoring.CONDITIONS]
    rows += [row(4, c, 0) for c in scoring.CONDITIONS[:3]]
    _, cells, cross = pilot.aggregate_sites(rows, "finished")
    summary = {r["site_count"]: r for r in cross if r["metric"] == "summary_score"}
    assert summary[2]["value"] == 100
    assert summary[4]["value"] is None
    assert summary[4]["missing_conditions"] == "surprising-masked"
    assert summary[4]["value_status"] == "missing conditions"
    assert all(
        c["value"] == 0
        for c in cells
        if c["site_count"] == 4 and c["condition"] != "surprising-masked"
    )


def test_failed_traces_retained_but_completed_sensitivity_loses_condition():
    rows = [
        row(2, c, status="failed" if c == "expected-masked" else "completed")
        for c in scoring.CONDITIONS
    ]
    _, _, main = pilot.aggregate_sites(rows, "finished")
    _, _, sensitivity = pilot.aggregate_sites(rows, "completed")
    assert next(r for r in main if r["metric"] == "summary_score")["value"] == 100
    assert next(r for r in sensitivity if r["metric"] == "summary_score")["value"] is None
    tokens = pilot.token_summary(rows, [])
    assert tokens[0]["known_output_tokens"] == 400
    assert tokens[0]["median_output_tokens_per_completed_run"] == 100
    assert tokens[0]["succeeded"] == 3
