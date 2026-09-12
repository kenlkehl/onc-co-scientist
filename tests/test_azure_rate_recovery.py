"""Recovery preserves history and replays all responses dependent on a throttle."""

import yaml

from scripts.expected_surprising.recover_azure_rate_limits import recover


def test_failed_federation_replays_earliest_affected_stage(tmp_path, monkeypatch):
    from onc_co_scientist.expected_surprising import experiment
    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import build_run_plans
    from tests.test_expected_surprising_experiment import config, providers
    from tests.test_expected_surprising_federation import FederatedScientist

    _, pairs, raw, path = config(tmp_path)
    raw["federation"] = {"site_counts": [2]}
    raw["workflows"] = [raw["workflows"][0]]
    raw["budget"]["max_agent_calls"] = 120
    path.write_text(yaml.safe_dump(raw))
    grid = load_experiment_spec(path)
    plan = build_run_plans(grid)[0]

    class Throttled(FederatedScientist):
        def chat(self, messages, **kwargs):
            if len(self.messages) >= 8:
                raise RuntimeError("429 rate limit exceeded")
            return super().chat(messages, **kwargs)

    providers(monkeypatch, pairs[0], Throttled)
    result = experiment.run_cell(grid, plan, grid.output_root, grid.fingerprint(), resume=False)
    assert result["status"] == "failed"
    root = grid.output_root / "runs" / plan.run_id
    original = {p.relative_to(root): p.read_bytes() for p in (root / "calls").rglob("*.json")}
    provenance = (root / "provenance.json").read_bytes()
    failed_report = (root / "run.json").read_bytes()
    receipt = recover(root)
    assert receipt["status"] == "rolled_back"
    assert (root / "recovery/azure_rate_limits/run.json").read_bytes() == failed_report
    assert recover(root) == receipt
    archived = {entry["path"] for entry in receipt["files"]}
    for p, content in original.items():
        retained = root / p if str(p) not in archived else root / "recovery/azure_rate_limits" / p
        assert retained.read_bytes() == content
    providers(monkeypatch, pairs[0], FederatedScientist)
    result = experiment.run_cell(grid, plan, grid.output_root, grid.fingerprint(), resume=True)
    assert result["status"] == "completed", result
    assert (root / "provenance.json").read_bytes() == provenance
    assert recover(root)["status"] == "retained_completed"
