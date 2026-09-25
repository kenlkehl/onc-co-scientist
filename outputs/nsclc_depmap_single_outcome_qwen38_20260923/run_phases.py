"""Extend the original frozen composite campaign to its authorized three phases.

Reuse its frozen manager, scientific source, and report-before-advancing gate.
The extension has its own additive manifest; the persistent freeze is untouched.
"""
import json

import manage

original_verify = manage.verify


def verify():
    original_verify()
    manifest = json.loads((manage.ROOT / "phase_extension_manifest.json").read_text())
    for name, expected in manifest["hashes"].items():
        if manage.sha(manage.ROOT / name) != expected:
            raise ValueError(f"Frozen phase extension artifact changed: {name}")


manage.WORKFLOWS = ("persistent", "sequential", "deliberative")
manage.verify = verify

if __name__ == "__main__":
    manage.main()
