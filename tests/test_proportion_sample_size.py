import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.expected_surprising.proportion_sample_size import calculate, required_runs


class ProportionSampleSizeTests(unittest.TestCase):
    def test_discussed_planning_examples(self):
        cases = [
            (0.7, 0.05, 676, 638),
            (0.6, 0.05, 761, 742),
            (0.5, 0.05, 783, 783),
            (0.6, 0.025, 3031, 2993),
        ]
        for benchmark, change, decrease, increase in cases:
            with self.subTest(benchmark=benchmark, change=change):
                result = calculate(benchmark=benchmark, change=change)
                self.assertEqual(
                    [row["required_runs"] for row in result["alternatives"]],
                    [decrease, increase],
                )
                self.assertEqual(
                    result["runs_to_cover_either_direction"], max(decrease, increase)
                )

    def test_invalid_inputs(self):
        for kwargs in [
            {"change": 0},
            {"change": -0.05},
            {"change": math.nan},
            {"benchmark": 1},
            {"benchmark": math.inf},
            {"change": 0.4},
            {"power": 1},
            {"power": 0.5},
            {"alpha": 0},
            {"alpha": math.nan},
        ]:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                calculate(**kwargs)

    def test_identical_alternative_rejected(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            required_runs(0.7, 0.7, 0.8, 0.05)

    def test_script_json_from_outside_repo(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/expected_surprising/proportion_sample_size.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(script), "--json"],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
        result = json.loads(completed.stdout)
        self.assertEqual(result["runs_to_cover_either_direction"], 676)
        self.assertEqual(result["benchmark"], 0.7)


if __name__ == "__main__":
    unittest.main()
