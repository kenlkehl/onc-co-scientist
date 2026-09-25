"""Resume a frozen federation grid with an explicitly recorded concurrency override.

Changes scheduling only. The original driver, provider, prompts, scientific settings,
and input provenance remain frozen and are verified by the original run function.
"""
import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path


def load_driver(root):
    driver=root/'source/scripts/expected_surprising/gemini_federation_grid.py'
    manifest=json.loads((root/'frozen_manifest.json').read_text())
    expected=manifest['hashes'][str(driver.relative_to(root))]
    if hashlib.sha256(driver.read_bytes()).hexdigest()!=expected:
        raise ValueError('Frozen driver changed')
    spec=importlib.util.spec_from_file_location('frozen_federation_driver',driver)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,required=True)
    a=p.parse_args()
    if not 1<=a.workers<=64:raise ValueError('Concurrency must be between 1 and 64')
    root=a.root.resolve()
    if (root/'PAUSE').exists():raise ValueError('Remove the pause only with user authorization before resuming')
    driver=load_driver(root)
    stamp=datetime.now(UTC).isoformat()
    driver.atomic_write_json(root/'concurrency_override.json',dict(updated_at=stamp,
        workers=a.workers,reason='User explicitly requested 64 concurrent Gemma requests',
        scientific_config_unchanged=True,launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    driver.run(root,a.workers)


if __name__=='__main__':main()
