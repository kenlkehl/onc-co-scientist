"""Transport migration must preserve call bodies and prior attempt provenance."""
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pytest


MODULE = Path(__file__).parents[1] / 'scripts/gcp_gemma_federation/resume_on_pool.py'
spec = importlib.util.spec_from_file_location('gemma_pool_resume', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('fails', [False, True])
def test_routes_preserve_history_and_audit_new_attempts(tmp_path, fails):
    @dataclass(frozen=True)
    class Config:
        base_url: str = 'http://camus:8060/v1'

    class Provider:
        def _send(self, body, call):
            assert self._config.base_url == 'http://127.0.0.1:18003/v1'
            assert body == original_body
            (call / 'attempt-0002').mkdir()
            if fails:
                raise RuntimeError('network failure')
            return 'ok'

    def write(path, value):
        path.write_text(json.dumps(value))

    grid = [dict(condition='named', run_id=f'run{i}') for i in range(120)]
    deployment = dict(endpoints=[f'http://127.0.0.1:{18000+i}/v1' for i in range(8)],
                      model_repository='nvidia/Gemma-4-31B-IT-NVFP4', model_revision='pinned',
                      instance='dedicated-gemma', zone='us-west1-a', container_image='pinned')
    routes = module.install_routes(Provider, grid, tmp_path, deployment, write)
    assert all(list(routes.values()).count(gpu) == 15 for gpu in range(8))
    p = Provider(); p._config = Config()
    p.root = tmp_path / 'named/runs/run3/provider_audit'
    call = p.root / 'call-unchanged'; (call / 'attempt-0001').mkdir(parents=True)
    original_body = dict(model='gemma4-31b', messages=[{'role': 'user', 'content': 'frozen prompt'}],
                         temperature=1.0, top_p=.95, top_k=64)
    old = call / 'attempt-0001/response.json';old.write_text('saved FP8 answer')
    if fails:
        with pytest.raises(RuntimeError, match='network failure'):
            p._send(original_body.copy(), call)
    else:
        assert p._send(original_body.copy(), call) == 'ok'
    assert old.read_text() == 'saved FP8 answer'
    assert not (call / 'attempt-0001/deployment.json').exists()
    audit = json.loads((call / 'attempt-0002/deployment.json').read_text())
    assert audit['quantization'] == 'NVFP4' and audit['gpu'] == 3
