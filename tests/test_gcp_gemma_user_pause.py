import importlib.util
import json
from pathlib import Path

path = Path(__file__).parents[1] / 'scripts/gcp_gemma_federation/user_pause_deadline.py'
spec = importlib.util.spec_from_file_location('user_pause_deadline', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pause_closes_all_admission_and_preserves_runtime_and_prior_marker(tmp_path):
    control = tmp_path / 'control'; (control / 'recovery-min-output').mkdir(parents=True)
    for path in [control / 'concurrency.json', control / 'recovery-min-output/concurrency.json']:
        path.write_text(json.dumps(dict(local_requests=8, requests_per_gpu=1,
                                       endpoint_limits={'camus':8, 'retired':0}, extra='retain')))
    runtime = control / 'recovery-min-output/runtime.json'
    runtime.write_text('saved live worker counters')
    (tmp_path / 'PAUSE').write_text('Existing user hold')
    module.close_admission(tmp_path, control)
    assert (tmp_path / 'PAUSE').read_text() == 'Existing user hold'
    assert runtime.read_text() == 'saved live worker counters'
    for path in [control / 'concurrency.json', control / 'recovery-min-output/concurrency.json']:
        value = json.loads(path.read_text())
        assert value['endpoint_limits'] == {'camus':0, 'retired':0}
        assert value['local_requests'] == value['requests_per_gpu'] == 0
        assert value['extra'] == 'retain'


def test_terminal_failures_count_as_finished_but_incomplete_runs_do_not(tmp_path):
    rows = [dict(condition='masked', run_id=str(i)) for i in range(3)]
    for row, data in zip(rows, [dict(scientific_report_sha256='hash', status='failed'),
                                dict(status='active'), dict(status='paused')]):
        folder = tmp_path / 'masked/runs' / row['run_id']; folder.mkdir(parents=True)
        (folder / 'run.json').write_text(json.dumps(data))
    assert module.terminal_count(tmp_path, dict(target=rows)) == 1
