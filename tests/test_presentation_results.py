"""Scientific aggregation and durable-run selection regressions (no live data)."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts/expected_surprising"
          / "build_presentation_results.py")
spec = importlib.util.spec_from_file_location("presentation_results", SCRIPT)
presentation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(presentation)


def report():
    return {
        "version": "expected", "model": "gpt-6-astra", "pair_id": "pair1", "versions": {},
        "scores": {"D": 0, "E": 50, "B": None, "discovery": {
            "exact": {"R": .5, "Q": 1., "diagnostic_D": 200 / 3},
            "confirmed_matches": [{"discovery_id": "focal", "match": "exact"}]}},
        "responsiveness": {"components": {
            c: {"accuracy": None} for c in presentation.CLASSES}},
        "behavior": {"milestones": [{"focal": True, "discovery_id": "focal"}]},
        "coordination": {"output_token_accounting": {
            "known_output_tokens": 100, "missing_calls": 0,
            "unaccounted_infrastructure_attempts": 0}},
    }


def plan(rid="run1"):
    return dict(run_id=rid, model_id="gpt-6-astra", workflow_id="persistent",
                semantic_condition="expected", replicate=1, iterations=25)


def row(status="completed", value=50, tokens=100):
    return dict(llm="gpt-6-astra", harness="Codex", workflow="persistent",
                reasoning_effort="medium", view="unmasked", version="expected",
                pair_id="pair1", attempted=True, status=status, output_tokens=tokens,
                report_available=status in ("completed", "failed"),
                **{k: (100 if value else 0) if k == "focal_recovery" else value
                   for k in presentation.METRICS},
                **{"response_" + c: None for c in presentation.CLASSES})


class PresentationResultsTests(unittest.TestCase):
    def test_failed_scientific_f1_retained_and_focal_unpenalized(self):
        metrics = presentation.scientific_metrics(report())
        self.assertAlmostEqual(metrics["f1"], 200 / 3)
        self.assertEqual(metrics["f1_saved"], 0)
        self.assertEqual(metrics["focal_recovery"], 100)

    def test_undefined_precision_remains_missing_with_zero_f1(self):
        data = report()
        data["scores"]["discovery"]["exact"].update(Q=None, diagnostic_D=0)
        metrics = presentation.scientific_metrics(data)
        self.assertIsNone(metrics["precision"])
        self.assertEqual(metrics["f1"], 0)

    def test_near_match_does_not_count_as_exact_focal_recovery(self):
        data = report()
        data["scores"]["discovery"]["confirmed_matches"][0]["match"] = "near"
        self.assertEqual(presentation.scientific_metrics(data)["focal_recovery"], 0)

    def test_pending_runs_not_scored_and_tokens_only_successful(self):
        rows = [row(tokens=100), row(tokens=300), row("failed", 0, 90000),
                row("unfinished", 99, 100000), row(tokens=None)]
        table, cells, _ = presentation.aggregate(rows, "finished")
        precision = table[0]
        self.assertEqual(precision["runs_attempted"], 5)
        self.assertEqual(precision["runs_successfully_completed"], 3)
        self.assertEqual(precision["runs_ended_with_errors"], 1)
        self.assertEqual(precision["median_output_tokens_per_completed_run"], 200)
        self.assertEqual(precision["expected-unmasked"], 37.5)
        self.assertIsNone(precision["expected-masked"])
        self.assertEqual(list(precision)[-12:], [c + suffix for c in presentation.CONDITIONS
                         for suffix in ("", "_ci95_low", "_ci95_high")])
        self.assertEqual(cells[0]["reports_included"], 4)
        successful, _, _ = presentation.aggregate(rows, "completed")
        self.assertEqual(successful[0]["expected-unmasked"], 50)

    def test_B_uses_class_means_not_only_complete_case_runs(self):
        rows = [row(), row()]
        rows[0].update(responsiveness=None, response_supported=1, response_excluded=0)
        rows[1].update(responsiveness=None, response_supported=0, response_ambiguous=1)
        _, cells, _ = presentation.aggregate(rows, "finished")
        b = next(r for r in cells if r["metric"] == "responsiveness"
                 and r["condition"] == "expected-unmasked")
        self.assertEqual(b["value"], 50)
        self.assertEqual(b["metric_available_runs"], 0)
        rows[1]["response_ambiguous"] = None
        _, cells, _ = presentation.aggregate(rows, "finished")
        b = next(r for r in cells if r["metric"] == "responsiveness")
        self.assertIsNone(b["value"])

    def test_base_datasets_equally_weighted(self):
        rows = [row(value=0), row(value=0), row(value=90)]
        rows[-1]["pair_id"] = "pair2"
        table, _, _ = presentation.aggregate(rows, "finished")
        self.assertEqual(table[0]["expected-unmasked"], 45)

    def test_focal_wilson_interval_does_not_collapse_at_perfect_recovery(self):
        ci = presentation.confidence_interval([row(value=100) for _ in range(10)],
                                              "focal_recovery")
        self.assertAlmostEqual(ci["ci95_low"], 72.2467200, places=5)
        self.assertAlmostEqual(ci["ci95_high"], 100)
        self.assertEqual(ci["ci_method"], "Wilson score")

    def test_bootstrap_reproducible_under_reordering_and_contains_variation(self):
        rows = [dict(row(value=v), run_id=str(i)) for i, v in enumerate([10, 30, 70, 90])]
        ci = presentation.confidence_interval(rows, "precision", seed=123)
        reversed_ci = presentation.confidence_interval(rows[::-1], "precision", seed=123)
        self.assertEqual(ci, reversed_ci)
        self.assertLess(ci["ci95_low"], 50)
        self.assertGreater(ci["ci95_high"], 50)
        self.assertEqual(ci["ci_valid_resamples"], 10000)

    def test_bootstrap_constant_scores_and_insufficient_observations(self):
        ci = presentation.confidence_interval([row(), row()], "precision")
        self.assertEqual((ci["ci95_low"], ci["ci95_high"]), (50, 50))
        for rows in ([row()], [row(), dict(row(), precision=None)]):
            ci = presentation.confidence_interval(rows, "precision")
            self.assertIsNone(ci["ci95_low"])
            self.assertIsNone(ci["ci95_high"])

    def test_B_bootstrap_preserves_correlated_evidence_classes(self):
        rows = [dict(row(), response_supported=v, response_excluded=1-v,
                     response_ambiguous=.5) for v in [0, 0, 1, 1]]
        ci = presentation.confidence_interval(rows, "responsiveness")
        self.assertEqual((ci["ci95_low"], ci["ci95_high"]), (50, 50))
        rows[1]["response_ambiguous"] = None
        rows[2]["response_ambiguous"] = None
        rows[3]["response_ambiguous"] = None
        self.assertIsNone(presentation.confidence_interval(rows, "responsiveness")["ci95_low"])

    def test_sparse_B_evidence_does_not_silently_discard_many_invalid_draws(self):
        rows = [dict(row(), response_supported=.5, response_excluded=.5,
                     response_ambiguous=.5 if i < 2 else None) for i in range(10)]
        ci = presentation.confidence_interval(rows, "responsiveness", seed=123)
        self.assertIsNone(ci["ci95_low"])
        self.assertLess(ci["ci_valid_resamples"], 9500)
        self.assertEqual(ci["ci_status"], "too many bootstrap draws lack eligible evidence")

    def test_summary_formula_equal_weights_zero_and_missing(self):
        self.assertAlmostEqual(presentation.summary_score([80] * 4), 80)
        self.assertAlmostEqual(presentation.summary_score([90, 90, 50, 85]),
                               (90 * 90 * 50 * 85) ** .25)
        self.assertEqual(presentation.summary_score([90, 90, 0, 85]), 0)
        self.assertIsNone(presentation.summary_score([90, None, 0, 85]))
        self.assertIsNone(presentation.condition_score(dict(f1=90, focal_recovery=100,
                                                          exploration=80, responsiveness=None)))
        # Sign in CSV condition order EU, EM, SU, SM, not narrative EU, SU, EM, SM.
        self.assertEqual(presentation.interaction_score([100, 100, 60, 90]), 30)
        self.assertEqual(presentation.interaction_score([100, 100, 90, 60]), -30)

    def test_condition_bootstrap_preserves_component_covariance(self):
        rows = [dict(row(value=0), run_id=str(i), f1=v, focal_recovery=100-v,
                     exploration=100-v, response_supported=v/100,
                     response_excluded=v/100, response_ambiguous=v/100)
                for i, v in enumerate([0, 0, 100, 100])]
        # C = .35*x + .25*(100-x) + .20*(100-x) + .20*x = 45 + .10*x.
        draws, enough = presentation.bootstrap_condition_score(rows, 123, 1000)
        self.assertTrue(enough)
        self.assertGreaterEqual(float(draws.min()), 45)
        self.assertLessEqual(float(draws.max()), 55)
        reverse, _ = presentation.bootstrap_condition_score(rows[::-1], 123, 1000)
        self.assertTrue((draws == reverse).all())

    def test_cross_scores_use_aggregate_B_and_keep_independent_denominators(self):
        rows = []
        for condition in presentation.CONDITIONS:
            version, view = condition.split("-")
            for i in range(6):
                item = dict(row(), run_id=str(i), view=view, version=version)
                item["response_" + presentation.CLASSES[i % 3]] = 1
                rows.append(item)
        table, cells, groups = presentation.aggregate(rows, "finished")
        cross = presentation.aggregate_cross_condition(rows, "finished", table, cells, groups)
        # All per-run B values are missing, but every evidence class has observations.
        expected = .35 * 50 + .25 * 100 + .20 * 50 + .20 * 100
        self.assertAlmostEqual(cross[0]["value"], expected)
        self.assertEqual(cross[1]["value"], 0)
        self.assertEqual(len(table), 7)
        self.assertEqual(len(cells), 28)
        self.assertIsNone(cells[-1]["metric_available_runs"])
        # Sparse class evidence must suppress the CI, not the point estimate.
        self.assertIsNone(cross[0]["ci95_low"])
        self.assertIsNotNone(cross[1]["ci95_low"])

    def test_cross_summary_missing_B_does_not_hide_focal_interaction(self):
        rows = []
        for condition in presentation.CONDITIONS:
            version, view = condition.split("-")
            rows.extend(dict(row(), run_id=str(i), view=view, version=version) for i in range(4))
        table, cells, groups = presentation.aggregate(rows, "finished")
        cross = presentation.aggregate_cross_condition(rows, "finished", table, cells, groups)
        self.assertIsNone(cross[0]["value"])
        self.assertIsNone(cross[0]["ci95_low"])
        self.assertEqual(cross[1]["value"], 0)
        self.assertLess(cross[1]["ci95_low"], 0)
        self.assertGreater(cross[1]["ci95_high"], 0)

    def test_interaction_interval_signs_and_perfect_recovery(self):
        import math
        cells = [dict(value=v, ci95_low=v-l, ci95_high=v+u)
                 for v, l, u in [(70, 10, 20), (60, 15, 25), (40, 5, 10), (50, 20, 30)]]
        ci = presentation.interaction_interval(cells)
        self.assertAlmostEqual(ci["ci95_low"], 20 - math.sqrt(10**2+25**2+10**2+20**2))
        self.assertAlmostEqual(ci["ci95_high"], 20 + math.sqrt(20**2+15**2+5**2+30**2))
        perfect = dict(value=100, **presentation.confidence_interval([row()] * 10, "focal_recovery"))
        ci = presentation.interaction_interval([perfect] * 4)
        self.assertAlmostEqual(ci["ci95_low"], -39.249052, places=4)
        self.assertAlmostEqual(ci["ci95_high"], 39.249052, places=4)

    def test_terminal_checksum_verification_and_missing_report(self):
        with tempfile.TemporaryDirectory() as temp:
            owner = Path(temp)
            folder = owner / "runs/run1"
            folder.mkdir(parents=True)
            raw = json.dumps(report()).encode()
            result = {"run_id": "run1", "status": "completed",
                      "scientific_report_sha256": hashlib.sha256(raw).hexdigest()}
            (folder / "run.json").write_text(json.dumps(result))
            missing = presentation.inspect_job((plan(), "unmasked", owner))
            self.assertFalse(missing["report_available"])
            self.assertIsNotNone(missing["report_error"])
            (folder / "report.json").write_bytes(raw)
            good = presentation.inspect_job((plan(), "unmasked", owner))
            self.assertTrue(good["report_available"])
            (folder / "report.json").write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                presentation.inspect_job((plan(), "unmasked", owner))

    def test_selection_keeps_unfinished_replacement_instead_of_old_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {
                "original/plan.json": [plan()],
                "selection.json": {"rows": [{"run_id": "run1", "replace": True}]},
                "local.json": [], "masked/plan.json": [plan("masked1")],
                "owners.json": {"masked1": "masked"},
            }
            for name, value in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            # A completed original must not win over its selected replacement.
            old = root / "original/runs/run1"
            old.mkdir(parents=True)
            (old / "run.json").write_text('{"status":"completed"}')
            config = dict(unmasked_plan="original/plan.json", codex_selection="selection.json",
                          codex_replacement_root="replacement", vllm_selection="local.json",
                          vllm_replacement_root="local", masked_plan="masked/plan.json",
                          masked_owners="owners.json", claude_roots={})
            jobs = presentation.select_jobs(presentation.Sources(root), config)
            self.assertEqual(jobs[0][2], root / "replacement")
            self.assertEqual(presentation.inspect_job(jobs[0])["status"], "queued")
            # Claude's unused original plan slots are not queued experiments.
            claude = root / "claude"
            claude.mkdir()
            (claude / "plan.json").write_text(json.dumps([
                plan("claude_selected"), plan("claude_unused"), plan("claude_new_launch")
            ]))
            (root / "claude_selection.json").write_text(json.dumps({
                "named": ["claude_selected"]
            }))
            (claude / "runs/claude_new_launch").mkdir(parents=True)
            config.update(claude_roots={"unmasked": "claude"},
                          claude_selections=["claude_selection.json"])
            jobs = presentation.select_jobs(presentation.Sources(root), config)
            claude_jobs = {p["run_id"] for p, _, owner in jobs if owner == claude}
            self.assertEqual(claude_jobs, {"claude_selected", "claude_new_launch"})
            broken = copy.deepcopy(config)
            (root / "owners.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "ownership"):
                presentation.select_jobs(presentation.Sources(root), broken)


if __name__ == "__main__":
    unittest.main()
