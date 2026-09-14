import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.expected_surprising import presentation_costs as costs


class PresentationCostTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(costs.DEFAULT_CONFIG.read_text())

    def test_electricity_units_and_additive_runtime(self):
        e = self.config['electricity']
        hours, kwh, usd = costs.electricity_cost(1_000_000, 150_000, e)
        self.assertAlmostEqual(hours, 2000 / 3600)
        self.assertAlmostEqual(kwh, 1 / 3)
        self.assertAlmostEqual(usd, .05)
        self.assertAlmostEqual(costs.electricity_cost(1_000_000, 0, e)[2], .025)
        self.assertAlmostEqual(costs.electricity_cost(0, 1_000_000, e)[2], 1 / 6)
        self.assertAlmostEqual(costs.electricity_cost(1_000_000, 150_000,
                               dict(e, gpu_count=2, usd_per_kwh=.30))[2], .20)

    def test_api_buckets_and_long_context(self):
        usage = costs.empty_usage()
        usage.update(input_tokens=1000, cached_input_tokens=600,
                     cache_write_input_tokens=100, output_tokens=200,
                     long_input_tokens=300_000, long_output_tokens=1000)
        expected = (300 * 10 + 600 + 100 * 12.5 + 200 * 50
                    + 300_000 * 20 + 1000 * 75) / 1e6
        self.assertAlmostEqual(costs.api_cost(usage, 'gpt-6-astra', self.config), expected)
        usage['cached_input_tokens'] = 1001
        with self.assertRaises(ValueError):
            costs.api_cost(usage, 'gpt-6-astra', self.config)

    def test_invalid_electricity_assumptions(self):
        for key in self.config['electricity']:
            if key == 'local_models':
                continue
            for bad in (0, -1, math.nan):
                config = json.loads(costs.DEFAULT_CONFIG.read_text())
                config['electricity'][key] = bad
                with self.subTest(key=key, value=bad), self.assertRaises(ValueError):
                    costs.validate_config(config)

    def test_rebuild_reprices_cache_and_changed_row_invalidates(self):
        row = dict(llm='gpt-6-astra', harness='Codex', workflow='persistent', view='masked',
                   version='expected', run_id='r1', status='completed', source='/unused',
                   report_sha256='first', missing_token_calls='0', unknown_internal_attempts='0')
        usage = costs.empty_usage()
        usage.update(input_tokens=1000, output_tokens=100, usage_records=1)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            def inventory():
                with (out / 'run_metrics.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
            inventory()
            with patch.object(costs, 'audit_usage', return_value=usage) as audit:
                first = costs.build(out)
                second = costs.build(out)
                self.assertEqual(audit.call_count, 1)
                self.assertEqual(first, second)
                doubled = json.loads(costs.DEFAULT_CONFIG.read_text())
                doubled['api_usd_per_million_tokens']['gpt-6-astra']['input'] = 20
                custom = out / 'rates.json'
                custom.write_text(json.dumps(doubled))
                self.assertAlmostEqual(costs.build(out, custom)['known_cost_usd'], .025)
                self.assertEqual(audit.call_count, 1)
                row['report_sha256'] = 'updated'
                inventory()
                costs.build(out)
                self.assertEqual(audit.call_count, 2)
                costs.build(out, refresh=True)
                self.assertEqual(audit.call_count, 3)

    def test_missing_local_tokens_are_not_free_runs(self):
        row = dict(llm='Inferact/Qwen3.8-27B-NVFP4', harness='Custom runner',
                   workflow='persistent', view='unmasked', version='expected', run_id='r1',
                   status='completed', input_tokens='', output_tokens='100',
                   missing_token_calls='1', unknown_internal_attempts='0')
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with (out / 'run_metrics.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            total = costs.build(out)['totals'][0]
            self.assertEqual(total['runs_completed'], 1)
            self.assertEqual(total['completed_runs_with_cost'], 0)
            self.assertIsNone(total['mean_cost_per_completed_run_usd'])
            self.assertIsNone(total['total_known_cost_usd'])


if __name__ == '__main__':
    unittest.main()
