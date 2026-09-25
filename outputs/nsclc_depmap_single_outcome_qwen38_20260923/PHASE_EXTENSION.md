# Composite campaign phase order

The user clarified that the one-outcome dataset follows the original three-workflow order.
Run 40 persistent experiments, report results, then 40 sequential experiments, report
results, then 40 deliberative experiments and report results. Each phase retains ten
repeats per expected/surprising by masked/unmasked cell and 25 iterations.

Only persistent is currently launched. The other phases have dry-run plans only.
Each dataset follows its own phase/report gate independently; the datasets may run
concurrently with 40 workers each, up to 80 total. Model, sampling, reasoning xhigh,
source, datasets and persistent execution are unchanged. Deliberative uses the original
two peers and one chair per stage. Preparation makes no model calls.

The initial frozen_manifest.json and grid.json remain unchanged as historical records
of the initial persistent launch. phase_extension_manifest.json and phase_extension_grid.json
add the later phases. Verify both manifests before launching a phase.

Use /home/klkehl/thisenv/bin/python -u <root>/run_phases.py run --workflow <phase>.
Use the same wrapper with reported --workflow <phase> only after reporting completed
results to the user. It reuses execution.lock and validates the preceding reporting receipt.
Add --resume only after confirming the prior host driver is stopped. Do not restart the
healthy current persistent process merely to switch wrappers.
